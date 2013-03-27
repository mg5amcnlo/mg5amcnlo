#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MadWeight                              ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author: Mattelaer Olivier (CP3)                                    ##
##       email:  olivier.mattelaer@uclouvain.be                         ##
##   author: Artoisenet Pierre (CP3)                                    ##
##       email:  pierre.artoisenet@uclouvain.be                         ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:10/06/08                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##      +go_to_main_dir                                                 ##
##      +read_card                                                      ##
##      +check_for_help                                                 ##
##########################################################################
##      +Class P_info                                                   ##
##      |     +  P_listdir                                              ##
##      |     +  detectSubProcesses                                     ##
##########################################################################
##      +Class MW_info (dict,P_info)                                    ##
##      |     +  init                                                   ##
##      |     |    + init_run_opt                                       ##
##      |     +  check_info                                             ##
##      |     +  take_un_name                                           ##
##      |     +  update_nb_card                                         ##
##      |     +  number_of_P_run                                        ##
##      |     +  load_events                                            ##
##      |     +  give_block_param_info                                  ##
##      |     |    +  CardNB_to_ParameterTag                            ##
##      |     +  set_run_opt                                            ##
##      |     |    + control_opt                                        ##
##      |     +  def_actif_param                                        ##
##      |     +  find_existing_events                                   ##
##########################################################################
##      +Class move                                                     ##
##      |     +  init                                                   ##
##      |     +  to_SubProcesses                                        ##
##      |     +  to_init                                                ##
##      |     +  store                                                  ##
##      |     +  to_store                                               ##
##      |     +  to_last_pos                                            ##
##      |     +  to_main                                                ##
##                                                                      ##
##########################################################################
##
## BEGIN INCLUDE
##
import sys
import os
import re
import stat
try: 
    import madgraph.madweight.Cards as Cards
except ImportError, error:
    print error
    import internal.madweight.Cards as Cards

##
## END INCLUDE
## GLOBAL DEFINITION

num_to_tag={1:'param',2:'analyzer',3:'compilation',4:'event',5:'dir',6:'launch',7:'control',8:'collect',9:'plot',\
            -1:'relaunch',-2:'clean',-3:'refine',-4:'status'}
tag_to_num={'param':1,'analyzer':2,'compilation':3,'event':4,'dir':5,'launch':6,'control':7,'collect':8,'plot':9,\
            'relaunch':-1,'clean':-2,'refine':-3,'status':-4}
last_step=9
#1#########################################################################
##                    START CODE
#1#########################################################################
def go_to_main_dir():
    """ move to main position """
    pos=os.getcwd()
    last=pos.split(os.sep)[-1]
    if last=='bin':
        os.chdir(os.pardir)
        return
    if last=='Python':
        os.chdir(os.pardir+os.sep+os.pardir+os.sep+os.pardir)
        return
    
    list_dir=os.listdir('./')
    if 'bin' in list_dir:
        return
    else:
        sys.exit('Error: script must be executed from the main, bin or Python directory')

#1#########################################################################
def read_card(name_card):
    """put all card information in a dictionary"""

    card=Cards.Card(name_card)
    return card.info

#1#########################################################################
def check_for_help(opt):
    """ check if the user use the -h or -help option or simply invalid option """

    opt_pat=re.compile(r'''-?(?P<opt>\w*)[+-]?''',re.I)
    help=0
    authorized_opt=tag_to_num.keys()+['version']
    for i in range(0,len(num_to_tag)):
        authorized_opt+=[str(i)]
    for i in range(1,len(opt)):
        if opt_pat.search(opt[i]):
            if opt_pat.search(opt[i]).group('opt').lower() not in authorized_opt:
                try:
                    int(opt_pat.search(opt[i]).group('opt').lower())
                except:
                    os.system('cat ./Source/MadWeight/Readme.txt')
                    sys.exit()
            if opt_pat.search(opt[i]).group('opt').lower()=='version':
                print 'MadWeight Version'
                os.system('cat ./Source/MadWeight/MW_TemplateVersion.txt')
                sys.exit()

#1#########################################################################
class P_info:
    """ class containing option/routine for standard MG/ME run """
    
    #2#########################################################################
    def P_lisdir(self):
        try:
            return self.Plistdir
        except:
            self.Plistdir,nothing=self.detect_SubProcess()
            return self.Plistdir

    #2#########################################################################
    def detect_SubProcess(self,P_mode=0):
        
            
        P_SubProcess_list,MW_SubProcess_list=detect_SubProcess(P_mode)
        return P_SubProcess_list,MW_SubProcess_list 



#1#########################################################################
class MW_info(dict,P_info):
    """ class containing all the option/information from the run """

    #2#########################################################################
    def __init__(self,card_name):
        """ init all the param for the run """

        self.mw_card=Cards.Card(card_name)
        self.info=self.mw_card.info
        for key,value in self.info.items():
            self[key]=value

        dict.__init__(self.info)
        self.check_info()
        #assign special value
        self.nb_event= self.info['mw_run']['nb_exp_events']
        self.nb_card=self.number_of_P_run()
        try:
            self.name = self.info['mw_run']['name']
        except:
            self.name = self.take_run_name()
        self.P_listdir,self.MW_listdir=self.detect_SubProcess()
        self.nb_event_MW = {}
        for MW in self.MW_listdir:
            self.nb_event_MW[MW] = self.nb_event 
        self.init_run_opt()
        self.def_actif_param()
        self.Pinupdate=[]
        self.Minpudate=[]
        self.startevent=0


    #3#########################################################################
    def init_run_opt(self,value=1):
        """ init all the run scheduling paramater to value """
        self.run_opt={}
        self.run_opt['param']=value
        self.run_opt['analyzer']=value
        self.run_opt['compilation']=value
        self.run_opt['event']=value
        self.run_opt['dir']=value
        self.run_opt['launch']=value
        self.run_opt['control']=value
        self.run_opt['collect']=value
        self.run_opt['plot']=value 
        self.run_opt['madweight_main']=value
        self.run_opt['relaunch']=0           #only for bugging case... -> desactivate
        self.run_opt['refine']=0             #only for bugging case... -> desactivate
        self.run_opt['clean']=0              #dangerous... -> desactivate
        self.run_opt['status']=0             # not in the mean stream
        self.control_opt()

    #2#########################################################################
    def check_info(self):
        """ assign default value if not defined already and check the input type
            those default value and the type are defined in MW_param_default.inc
            structure of this file:
            block tag type value #comment
            additionaly check if the mw_parameter['?3'] is a list
        """
        
        #define convertissor
        def pass_in_integer(value):
            return int(value)

        def pass_in_logical(value):
            if value in ['1','t','T','.true.',1,True]:
                return 1
            else:
                return 0

        def pass_in_float(value):
            return float(value)

        for line in open('./Source/MadWeight/Python/MW_param_default.inc'):
            line=line.split('#')[0] #remove comment
            splitline=line.split()  #split the data
            if len(splitline)!=4:
                continue
            #assign element
            block=splitline[0].lower()
            tag=splitline[1].lower()
            type=splitline[2].lower()
            value=splitline[3]
            #check if exist -> default
            try:
                self[block][tag]
            except:
                if value.count('@@'):
                    value=value.split('@@')
                    value_2=self[value[0]][value[1]]
                    value=value_2
                try:
                    self[block][tag]=value
                except:
                    self[block]={tag:value}
            #change type
            if type in ['integer','logical','float']:
                self[block][tag]=eval('pass_in_'+type+'(self[block][tag])')
        self.check_mw_parameter_list_type()
        
    #2#########################################################################
    def check_mw_parameter_list_type(self):
        """ check that self['mw_parameter']['?3'] is a list """
        nb_param=1

        while self['mw_parameter'].has_key(str(nb_param*10+3)):
            if not isinstance(self['mw_parameter'][str(nb_param*10+3)], list):
                self['mw_parameter'][str(nb_param*10+3)]=[self['mw_parameter'][str(nb_param*10+3)]]
            nb_param+=1
                

    #2#########################################################################
    def take_run_name(self):
        """take the run name in run_card"""
        name="run"
        Pattern=re.compile(r'''\'(\S*)\'\s*=\s*run_tag''',re.I)
        card=open("./Cards/run_card.dat")

        while 1:
            line=card.readline()
            if line=='':
                break
            
            if Pattern.search(line):
                name=Pattern.search(line).groups()[0]
                break
        return name

    #2##########################################################################
    def update_nb_card(self):
        "take the info from MW_runcard.dat"
        self.nb_card=self.number_of_P_run()
        self.def_actif_param()
        
    #2##########################################################################
    def number_of_P_run(self):
        "take the info from the existing card in Cards"

        #check if we use different param_card.dat
#        if self.info["mw_parameter"]["1"]=="1":
        j=1
        while 1:
            if os.path.isfile('Cards/param_card_'+str(j)+'.dat'): j+=1
            elif(j==1): return j
            else: return j-1
            


    #2##########################################################################
    def load_events(self):
        "detect the number of events for P and MW run"

        self.P_nevents=self.info['mw_run']['5']
        self.MW_nevents=self.info['mw_run']['6']

        
    #2##########################################################################
    def give_block_param_info(self):
        """ return the number of modified parameter and the number of different value for each"""

        nb_block=0
        nb_values=[]
        k=0
        while 1:
            k+=1
            try:
                self.info['mw_parameter'][str(10*k+1)]
            except:
                break
            nb_block+=1
            if type(self.info['mw_parameter'][str(10*k+3)])==list:
                nb_values.append(len(self.info['mw_parameter'][str(10*k+3)]))
            else:
                nb_values.append(1)

        return nb_block,nb_values

    #3########################################################################
    def CardNb_to_ParameterTag(self,num_card):
        """ find from th card number, to which value for each block this card belong
            num_card is the number of the card in the last generation. 
            card in previous generation are not accessible by this routine
            (and are not related to this MadWeight card anyway)
         """

        nb_block,nb_data_by_block=self.give_block_param_info()

        if self['mw_parameter']['1']==2:
            return [num_card-1]*len(nb_data_by_block)

        tag=[]
        for k in range(-1,-nb_block-1,-1):
            tag.append((num_card-1)%nb_data_by_block[k])
            num_card=1+(num_card-(num_card-1)%nb_data_by_block[k])//nb_data_by_block[k]
        tag.reverse()
        return tag

    #2##########################################################################
    def set_run_opt(self,option):
        """analyze option for the run"""

        if len(option)>1:
            self.init_run_opt(0)#put everything to false
        else:
            return
        for i in range(1,len(option)):
            if option[i][0]=='-' and option[i][-1]=='+':
                num=int(option[i][1:-1])
                for j in range(num,last_step+1):
                    self.run_opt[num_to_tag[j]]=1
            elif option[i][0]=='-' and option[i][-1]=='-':
                num=int(option[i][1:-1])+1
                for j in range(1,num):
                    self.run_opt[num_to_tag[j]]=1
            elif option[i][0]=='-':
                num=int(option[i][1:])
                for i in option[i][1:]:
                    self.run_opt[num_to_tag[int(i)]]=1
            elif option[i][-1]=='+':
                num=tag_to_num[option[i][:-1]]
                for j in range(num,last_step+1):
                    self.run_opt[num_to_tag[j]]=1
            elif option[i][-1]=='-':
                num=tag_to_num[option[i][:-1]]+1
                for j in range(1,num):
                    self.run_opt[num_to_tag[j]]=1
            elif '=' in option[i]:
                obj=option[i].split('=')
                tag=obj[0]
                value=obj[1]
                self.run_opt[tag]=value
            else:
                self.run_opt[option[i]]=1
                                                
        self.control_opt()

    #3##########################################################################
    def control_opt(self):
        """analyze option for the run to have coherent input"""


        if self.run_opt['refine']:
            self.run_opt['relaunch']=1
        
        #check value for 'madweight_main'
        for i in range(3,9)+[-1,-3,-4]:
            if self.run_opt[num_to_tag[i]]==1:
                self.run_opt['madweight_main']=1
                break

        if self.run_opt['relaunch']==1:
            self.run_opt['control']=1

    #3##########################################################################
    def def_actif_param(self):
        """ find which card are still actif """

        self.param_is_actif={}
        try:
            ff=open('Cards/mapping_card.dat')
        except:
            print 'no mapping card found'
            for i in range(1,self.nb_card+1):
                self.param_is_actif[i]=1 #if no file defined all card are supose to be used
            self.actif_param=range(1,self.nb_card+1)
            return

        self.actif_param=[]
        for line in ff:
            split=line.split()
            if len(split)<2:
                continue
            nb=int(split[0])
            actif=int(split[-1])
            self.param_is_actif[nb]=actif
            if actif:
                self.actif_param.append(nb)

        if self.nb_card!=len(self.actif_param):
            print 'wrong mapping file or some card are desactivated'


    #3##########################################################################
    def find_existing_events(self,directory='',card_nb=''):
        """ find how much event are already defined """

        if directory=='':
            directory='./SubProcesses/'+self.MW_listdir[0]+'/'+self.name+'/'
        if card_nb=='':
            card_nb=self.actif_param[0]
        if os.path.isdir(directory+'/card_'+str(card_nb)+'/'):
            number=[int(eventdir.split('_')[-1]) for eventdir in os.listdir(directory+'/card_'+str(card_nb)+'/')]
        else:
            return 0
        
        if len(number):
            return max(number)+1
        else:
            return 0
        
#1 #################################################################################
class move:
    def __init__(self):
        self.initpos=os.getcwd()
        self.old=self.initpos

    def to_SubProcesses(self):
        
        old_pos=os.getcwd()+'/'
        self.old=old_pos
        if old_pos.count('/SubProcesses/')>1:
            new_pos=pos.split('/SubProcesses/')[0]+'/SubProcesses/'
        elif old_pos.count('/SubProcesses/')==0:
            new_pos=old_pos+'/SubProcesses/'
        else:
            new_pos='/SubProcesses/'.join(old_pos.split('/SubProcesses/')[:-1])+'/SubProcesses/'
            
        if new_pos!=old_pos:
            self.old=old_pos
            os.chdir(new_pos)
            
    def to_init(self):
        self.old=os.getcwd()
        os.chdir(self.initpos)

    def store(self):
        self.store=os.getcwd()

    def to_store(self):
        self.old=os.getcwd()
        os.chdir(self.store)

    def to_last_pos(self):
        pos,self.old=self.old,os.getcwd()
        os.chdir(pos)

    def to_main(self):
        old=os.getcwd()
        try:
            go_to_main_dir()
        except:
            self.to_SubProcesses()
            os.chdir('..')
        self.old=old


def detect_SubProcess(P_mode=0):
        MW_SubProcess_list=[]
        if P_mode == 1:
            print 'WARNING: automatic renormalization is not supported anymore'

        list_dir=os.listdir("./SubProcesses/")
        for name in list_dir:
            try:           
                st = os.lstat(os.path.join("./SubProcesses/", name))
            except os.error:
                continue
            if stat.S_ISDIR(st.st_mode):
                if name[0]=='P':
                    MW_SubProcess_list.append(name)                

# Pierre: set MW_SubProcess_list to SubProcess_list, set SubProcess_list to []
        return [],MW_SubProcess_list

