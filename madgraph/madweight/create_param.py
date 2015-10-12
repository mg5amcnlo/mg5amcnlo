#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MadWeight                              ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author: Mattelaer Olivier (CP3)                                    ##
##   email:  olivier.mattelaer@uclouvain.be                             ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:07/01/09                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##      +Class Param_card                                               ##
##      |     +  init                                                   ##
##      |     +  charge_card                                            ##
##      |     +  create_map_file                                        ##
##      |     +  mod_content                                            ##
##      |     |    -  mod_with_diff                                     ##
##      |     |    -  mod_with_fct                                      ##  
##      |     +  add_content                                            ##
##      |     +  write_card                                             ##
##      |     |    -  create_blok_text                                  ##
##      |     |    -  create_line_text                                  ##
##      |     |    -  create_decay_text                                 ##
##      |     |    -  create_br_text                                    ##
##      |     |_   -  create_br_line                                    ##
##      |     +  create_set_card                                        ##
##      |     |    -  create_change_tag                                 ##
##      |     |    -  generated_uncorolated_card                        ##
##      |     |    -  generated_corolated_card                          ##
##      |     |    -  del_old_param                                     ##
##      |_    |_   -  check_exist                                       ##
##                                                                      ##
##########################################################################
##
## BEGIN INCLUDE
##
try:
    import madgraph.madweight.MW_info as MW_param
except ImportError:
    import internal.madweight.MW_info as MW_param
import sys
import os
import re
import math
##
## END INCLUDE
##

#1#########################################################################
##                    START CODE
#1#########################################################################
class Param_card:

    #2#########################################################################
    def __init__(self,param_name=0,run_name=0):
        if type(param_name)==str:
            self.charge_card(param_name)
        elif type(param_name)==dict:
            self.source=1
            self.info=param_name
        else:
            self.source=0

        if run_name:
            print 'starting the creation of the param_card'
            self.create_set_card(run_name)
            
    #2#########################################################################
    def charge_card(self,name):
        """ charge the card """
        self.source=name
        self.info=MW_param.read_card(name)



    #2#########################################################################
    def mod_content(self,line_content):
        """ modify an entry in info """
        
#        line_content_old=list(line_content)
        name_blok=line_content[1]
        tag=line_content[0]
        line_content=line_content[2:]
        # treat special case
        if '_' in name_blok:
            special_tag,name_blok=name_blok.split('_')
            if special_tag=='diff':
                line_content=self.mod_with_diff(name_blok,line_content)
            elif special_tag=='fct':
                line_content=self.mod_with_fct(name_blok,line_content,tag)

        #line_content=line_content[1:]
        obj=self.info[name_blok]
        for i in range(0,len(line_content)-1):
            if line_content[i] not in obj.keys():
                self.add_content([name_blok]+line_content)
                return
            if i!=len(line_content)-2:
                obj=obj[line_content[i]]
            else:
                obj[line_content[i]]=line_content[i+1]
                #obj.update(prov)

    #3 #########################################################################
    def mod_with_diff(self,name_blok,line_content):
        """ modify the line content to fix the parameter in fixing the mass differences
            entry line_content doesn't content the name blok information

            You should prefer to use mod_with_fct (more general)
        """
        diff=line_content[-1]
        number_of_tag=(len(line_content)-1)//2 #-1 because of the value                                                                                              
        start_value=self.info[name_blok]
        for i in range(0,number_of_tag):
            start_value=prec_value[line_content[number_of_tag+i]]
        new_value=str(float(start_value)+float(diff))

        return [line_content[i] for i in range(0,number_of_tag)]+[new_value]
        
    #3 #########################################################################
    def mod_with_fct(self,name_blok,line_content,level):
        """ modify the line content to fix the parameter in fixing the mass differences
            entry line_content doesn't content the name blok information
        """        

        pat=re.compile(r'''\$(\d+)3''')
        fct=line_content[-2]
        values=pat.findall(fct)

        for param_nb in values:
            fct=fct.replace('$'+param_nb+'3','param'+param_nb)
            string0='self.info[self.MWparam[\'mw_parameter\'][\''+str(10*int(param_nb)+1)+'\'].split(\'_\')[-1]]' #select the blok name
            string=string0
            if type(self.MWparam['mw_parameter'][str(10*int(param_nb)+2)])==str:
                string+='[\''+str(self.MWparam['mw_parameter'][str(10*int(param_nb)+2)])+'\']' 
            else:
                for id in self.MWparam['mw_parameter'][str(10*int(param_nb)+2)]:
                    if id[0] not in['\'','\"']:  string+='[\''+str(id)+'\']'                       # the function start with '
            if int(param_nb)!=level: exec('param'+param_nb+'=float('+string+')')               # find the current value of the parameter
            else:                    exec('param'+param_nb+'=float('+line_content[-1]+')')
        try:
            exec('new_value='+fct[1:-1])   #supress the ' on the fct and find the correct value
        except:
            print 'WARNING: fct undefined for card ',self.creating_card,'. This card will be desactivated'
            self.wrong_generation.append(self.creating_card)
            new_value=-1

        return [line_content[i] for i in range(0,len(line_content)-2)]+[str(new_value)]

    #2#########################################################################                                
    def add_content(self,line_content):
        """ add new content in info """
        
        name_block=line_content[0]
        line_content=line_content[1:]
        #create list of dictionary
        obj=line_content[-1]
        for i in range(-2,-len(line_content)-1,-1):
            obj={line_content[i]:obj}
                    

        #put in final data
        dico=self.info[name_block]
        for i in range(0,len(line_content)-1):
            if line_content[i] not in dico.keys():
                dico[line_content[i]]=obj[line_content[i]]
                break
            elif i!=len(line_content)-2:
                dico=dico[line_content[i]]
                obj=obj[line_content[i]]
            elif(type(dico[line_content[i]])==list):                         #multiple definition of the same input
                dico[line_content[i]].append(obj[line_content[i]])
            else:
                dico[line_content[i]]=[dico[line_content[i]],line_content[i+1]]

    #2#########################################################################
    def write_card(self,name):
        """write the param_card with name $name """

        if 1:

        #try:
            num=int(name[11:-4])
            self.create_info_line(num)
        #except:
        #    pass

        MW_param.go_to_main_dir()
        ff=open('./Cards/'+name,'w')
        if self.source:
            #charge the comment from the original card
            gg=open('./Cards/'+self.source,'r')
            while 1:
                line=gg.readline()
                if line=='':
                    break
                if line[0]=="#":
                    ff.writelines(line)
                else:
                    break
            gg.close()

        decay=0
        for key in self.info.keys():
            if key in ['decay','br']:
                decay=1
                continue
            if key=="comment":
                continue
            text=self.create_blok_text(key)
            ff.writelines(text)

        if decay:
            text=self.create_decay_text()
            ff.writelines(text)
            
        ff.close()
        self.creating_card+=1
        
    #3#########################################################################
    def create_blok_text(self,blok_name):
        """write the param_card with name $name """

        text='Block '+blok_name.upper()+' '+self.info['comment'][blok_name]+'\n'
        prop_text=self.create_line_text(self.info[blok_name])
        if prop_text.count('$$'):
            print 'multiple inputs are not supported yet'
            print 'you must create your Cards by hand'
            sys.exit()

        return text+prop_text
            
    #3#########################################################################
    def create_line_text(self,obj,key_tag=""):

        text=''
        if type(obj)==dict:
            for key in obj.keys():
                text+=self.create_line_text(obj[key],key_tag+'     '+key)

        elif type(obj)==str:
            text=key_tag+'    '+obj+'\n'
        elif type(obj)==list:
            text='   $$ '
            for data in obj:
                text+=data+' $ '
            text+='$\n'
            
        return text



    #3#########################################################################
    def create_decay_text(self):
        """write the param_card with name $name """

        decay=self.info['decay']
        try:
            br=self.info['br']
        except:
            br=0
        text=''
        for key in decay.keys():
            text+='DECAY        '+key+'    '
            text+=self.create_line_text(decay[key])
            if br:
                if key in br.keys():
                    text+=self.create_br_text(br[key])
            

        return text
    
    #3#########################################################################
    def create_br_text(self,obj):
        """write the param_card with name $name """

        text=''
        space='    '
        list_data=[]
        for key in obj.keys():
            list_data+=self.create_br_line(obj[key],[key])

        for data in list_data:
            text+=space+str(data[-1])+space+str(len(data)-1)
            for i in range(0,len(data)-1):
                text+=space+data[i]
            text+='\n'
        return text

    #3#########################################################################
    def create_br_line(self,obj,begin):
        """write the param_card with name $name """


        content=[]
        if type(obj)==dict:
            for key in obj.keys():
                content_i=[key]
                content_i=self.create_br_line(obj[key],begin+[key])
                if type(content_i[0])==str:
                    content.append(content_i)
                else:
                    content+=content_i
        elif type(obj)==str:
            return begin+[obj]

        return content
    
    #2#########################################################################
    def create_set_card(self,name):
        """ create all the card from schedular in file name """
        self.creating_card=1 #tag to know card under creation
        self.wrong_generation=[] #can happen if fct is wrongly defined -> automatic desactivation
        
        if type(name)==str:
            self.MWparam=MW_param.read_card(name)
        else:
            self.MWparam=name
        
        if self.MWparam['mw_parameter']['2']:
            self.file_ParamInfo=open('./Cards/info_card.dat','a')	            
        else:
            print 'define new mapping file'
            self.file_ParamInfo=open('./Cards/info_card.dat','w')	            

        param_list=self.create_change_tag(self.MWparam)

        if not self.source:
            self.charge_card('param_card.dat')            
        if self.MWparam['mw_parameter']['1'] == 0:
            self.check_exist()
            self.define_mapping_file()
            return
        elif self.MWparam['mw_parameter']['1'] == 1:
            self.del_old_param()
            num=self.generated_uncorolated_card(param_list)
        elif self.MWparam['mw_parameter']['1'] == 2:
            self.del_old_param()
            num=self.generated_corolated_card(param_list)

        self.define_mapping_file()
        print 'we have created ',num-1,' param_card\'s'
        if self.wrong_generation:
            print 'but ',len(self.wrong_generation),' are desactivated'
                        
        if self.MWparam['mw_parameter']['2']:
            self.update_event_dir()

    #3#########################################################################
    def update_event_dir(self):
            
        #update event directory
        self.file_mapping.close()
        self.MWparam.def_actif_param()
        import create_run as Create
        create_obj=Create.create_dir(self.MWparam)
        create_obj.update_card_status()


    #3#########################################################################
    def create_info_line(self,nb_card):
        """ create the file containing the mapping between the card number and the parameter
        syntax:
        card_nb param1 param2 ... paramX valid
        """

        nb_param=1
        line=str(nb_card)+'\t'
        while self.MWparam['mw_parameter'].has_key(str(nb_param*10+1)):
            tag1=self.MWparam['mw_parameter'][str(nb_param*10+1)]
            tag2=self.MWparam['mw_parameter'][str(nb_param*10+2)]
            if 'fct_' in tag1:
                tag1=tag1[4:]
                tag2=tag2[:-1]

            value=self.info[tag1.lower()]
            if type(tag2)==str:
                value=value[tag2.lower()]
            else:
                for param in tag2:
                    value=value[param.lower()]
            line+=value+'\t'
            nb_param+=1
        line+=' \n' #ActifTag

        self.file_ParamInfo.writelines(line)

    #3#########################################################################
    def define_mapping_file(self):
        """ create the file containing the mapping between the card number and the parameter
        syntax:
        card_nb param1 param2 ... paramX valid
        """

        if self.MWparam['mw_parameter']['2']:
            print 'add card in mapping file'
            gap=self.MWparam.nb_card
            self.file_mapping=open('./Cards/mapping_card.dat','a')
            self.file_ParamInfo=open('./Cards/info_card.dat','a')	            
        else:
            print 'define new mapping file'
            gap=0
            self.file_mapping=open('./Cards/mapping_card.dat','w')
            self.file_ParamInfo=open('./Cards/info_card.dat','w')	            


        if self.MWparam['mw_parameter']['1']==0:
            self.define_mapping_file_for0gen(gap)
        elif self.MWparam['mw_parameter']['1']==1:
            self.define_mapping_file_for1gen(gap)	        
        elif self.MWparam['mw_parameter']['1']==2:	       
            self.define_mapping_file_for2gen(gap)
            
            


    #3#########################################################################
    def define_mapping_file_for0gen(self,gap=0):
        """ create the file containing the mapping between the card number and the parameter
        syntax:
        card_nb param1 param2 ... paramX valid
        """

        for i in range(1,self.MWparam.nb_card+1):
            self.file_mapping.writelines(str(i)+'\t1 \n')

    #3#########################################################################
    def define_mapping_file_for1gen(self,gap=0):
        """ create the file containing the mapping between the card number and the parameter
        syntax:
        card_nb param1 param2 ... paramX valid
        """
        
        start=1+gap
        nb_new_card=1
        nb_param=1
        while self.MWparam.info['mw_parameter'].has_key(str(nb_param*10+1)):		
            nb_new_card*=len(self.MWparam['mw_parameter'][str(nb_param*10+3)])
            nb_param+=1
                        
        for card in range(start,start+nb_new_card):
            line=str(card)+'\t'
            param_pos=self.MWparam.CardNb_to_ParameterTag(card)
            for param in range(1,nb_param):
                if type(self.MWparam['mw_parameter'][str(param*10+3)])==list:
                    line+=self.MWparam.info['mw_parameter'][str(param*10+3)][param_pos[param-1]]+'\t'
                else:
                    line+=self.MWparam.info['mw_parameter'][str(param*10+3)]+'\t'
            if card in self.wrong_generation:
                line+='0 \n'
            else:
                line+='1 \n'
            
            self.file_mapping.writelines(line)

    #3#########################################################################
    def define_mapping_file_for2gen(self,gap=0):
        """ create the file containing the mapping between the card number and the parameter
        syntax:
        card_nb param1 param2 ... paramX valid
        """
 
        start=1+gap
        nb_new_card=1
        nb_param=1
        
        nb_block,nb_data_by_block=self.MWparam.give_block_param_info()		        
        for card in range(start,start+nb_data_by_block[0]):
            line=str(card)+'\t'
            for param in range(1,nb_block+1):
                if type(self.MWparam['mw_parameter'][str(param*10+3)])==list:
                    line+=self.MWparam.info['mw_parameter'][str(param*10+3)][card-start]+'\t'
                else:
                    line+=self.MWparam.info['mw_parameter'][str(param*10+3)]

            if card in self.wrong_generation:
                line+='0 \n'
            else:
                line+='1 \n'
                                
            self.file_mapping.writelines(line)
            
    #3#########################################################################
    def create_change_tag(self,info):
        """ create list of possible modification """

        output=[]
        num=1
        while info['mw_parameter'].has_key(str(10*num+1)):
            content=[]
            tag=str(10*num+1)
            data=[num]
            data.append(info['mw_parameter'][tag].lower())
            tag=str(10*num+2)
            if type(info['mw_parameter'][tag])==list:
                data+=info['mw_parameter'][tag]
            else:
                data.append(info['mw_parameter'][tag])
            tag=str(10*num+3)
            if type(info['mw_parameter'][tag])==list:
                for i in range(0,len(info['mw_parameter'][tag])):
                    content.append(data+[info['mw_parameter'][tag][i]])
            else:
                content.append(data+[info['mw_parameter'][tag]])
            output.append(content)
            num+=1

        return output

            
    #3#########################################################################
    def generated_uncorolated_card(self,param_list,num=1):
        """ create the card in a uncoralated way """

        if self.MWparam['mw_parameter']['2']:
            gap=self.MWparam.nb_card
            self.creating_card+=gap
        else:
            gap=0
                                        
        
        new_list=param_list[1:]
        for data in param_list[0]:
            self.mod_content(data)
            if new_list:
                num=self.generated_uncorolated_card(new_list,num)
            else:
                self.write_card('param_card_'+str(num+gap)+'.dat')
                num=num+1
        return num
            


    
            
    #3#########################################################################
    def generated_corolated_card(self,param_list):
        """ create the card in a coralated way """

        # 1) check if all parameter have the same number of data:
        for i in range(0,len(param_list)-1):
            if len(param_list[i])!=len(param_list[i+1]):
                print """ERROR: all parameters don't have the same number of entries"""
                sys.exit()
                
        # 2) pass in all case
        if self.MWparam['mw_parameter']['2']:
            gap=1+self.MWparam.nb_card
            self.creating_card+=gap
        else:
            gap=1
            
        for i in range(0,len(param_list[0])):
            for j in range(0,len(param_list)):
                self.mod_content(param_list[j][i])
            self.write_card('param_card_'+str(i+gap)+'.dat')

        return len(param_list[0])+1
                                 
    #3#########################################################################
    def del_old_param(self):
        """ supress all the all param_card """

        if not self.MWparam['mw_parameter']['2']:
            os.system('rm ./Cards/param_card_?.dat &>/dev/null')
            os.system('rm ./Cards/param_card_??.dat &>/dev/null')
            os.system('rm ./Cards/param_card_???.dat &>/dev/null')

    #3#########################################################################
    def check_exist(self):
        """ check if param_card_1 exist and copy param_card if not """

        
        try:
            ff=open('Cards/param_card_1.dat','r')
            ff.close()
            os.system('ln -s  param_card_1.dat Cards/param_card.dat')
        except:
            os.system('cp ./Cards/param_card.dat ./Cards/param_card_1.dat')
            os.system('ln -s param_card_1.dat Cards/param_card.dat')
