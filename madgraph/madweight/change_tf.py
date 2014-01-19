#!/usr/bin/env python

#Extension

import os
import re
import string
import sys
import xml.sax.handler

try:
    import madgraph.madweight.Cards as Cards
    import madgraph.madweight.mod_file as mod_file
    import madgraph.madweight.particle_class as particle_class
    import madgraph.madweight.MW_fct as MW_fct
except ImportError:
    import internal.madweight.Cards as Cards
    import internal.madweight.mod_file as mod_file
    import internal.madweight.particle_class as particle_class
    import internal.madweight.MW_fct as MW_fct
    

###f77 forbiden term
#forbiden=["goto","return","stop","call","write","read","do","while","end ","continue","asign","pause","print","rewind","backspace","endfile","open","close","inquire","entry","optional","save","equivalence","intent","target","rule","compute","system","enddo"]


def create_TF_main(name,make, MW_dir):
    print "start main program"
    TF_file=Full_TF(name)
    TF_file.read_file("./data/TF_"+name+".dat")

    print "deleting the current TFlib: "
    os.system("rm ../../../lib/libTF.a  >& /dev/null")
     
    TF_file.create_ordering_file()
    print "ordering_file.inc created"
    list_var=TF_file.create_transfer_functions()
    print "transfer_function.f created"
    TF_file.create_transfer_card(list_var)
    print "transfer_card.dat created"
    create_param_inc(list_var)
    print "TF_param created"
    create_ident_card(list_var)
    print "ident_card created"      
    create_version(name)
    print 'TransferFunctionVersion created'
    fsock = open('nb_tf.inc','w').write('       integer nb_tf\n      parameter (nb_tf=1)\n')
    os.chdir('../../../') #go to main
    
    
#    P_dir,MW_dir=MW_param.detect_SubProcess(P_mode=1)

    for directory in MW_dir:
        obj=TF_in_SubProcesses(TF_file,directory)
        obj.write_transfer_function_file() 
    print 'call_TF.f created in for all Subprocesses'
    os.chdir('./Source/MadWeight/transfer_function')
    update_dir(name,make,MW_dir)
    print 'generation completed'       
           

#1 ################################################################################# 
class XML_input(xml.sax.handler.ContentHandler):
    """ This class will organize in python obect the TF_param.dat file 
        (written in xml) 
    """

    #2 #############################################################################        
    def __init__(self):
        self.block = {}
        self.inblock=''
        self.in_variable=''
        self.buffer=''

    #2 #############################################################################  
    def startElement(self, name, attributes):
        self.buffer=''
        if name == "block":
            block_name=attributes["name"]
            self.block[block_name]=TF_block(block_name)
            self.inblock=block_name
        elif name == "variable":
            self.in_variable=attributes["name"]
            
    #2 #############################################################################  
    def characters(self, data):
        self.buffer += data
 
    #2 ############################################################################# 
    def endElement(self, name):
        if name == 'particles':
            self.block[self.inblock].def_particles(self.buffer)
        elif name == 'width_type':
            self.block[self.inblock].def_width_type(self.buffer)  
        elif name == "tf":
            self.block[self.inblock][self.in_variable].change_tf(self.buffer)
        elif name == "width":
            self.block[self.inblock][self.in_variable].change_width(self.buffer)  
        elif name == "info":
            self.block[self.inblock].def_info(self.buffer)
        elif name == "include":
            self.block[self.inblock][self.in_variable].def_include(self.buffer)
            
    #2 ############################################################################# 
    def read_file(self,filepos):
        """ parse the file and fulfill the object """
        parser = xml.sax.make_parser(  )
        parser.setContentHandler(self)
        parser.parse(filepos)
 
    #2 ############################################################################# 
    def find_label_to_block(self):
        """ return a dict {label:block_name} store it for efficiency reason"""
        
        if hasattr(self,'label_to_block'):
            return self.label_to_block
        
        self.label_to_block={}
        for name,block in self.block.items():
            for label in block.particles:
                self.label_to_block[label]=name
        
        return self.label_to_block
        
        
#1 #################################################################################
class TF_with_particles(XML_input):
    """ this class extend the XML with routine associating the particles.dat file """
    
    def __init__(self,only_add=0):
        """ standard input but add a tag if the particles.dat is loaded or not """
        
        if not only_add:
            XML_input.__init__(self)
        self.particles_file=0
        
    def load_particles_file(self,filepos='./Source/MODEL/particles.dat'):
        """ load the particles file if not already loaded """
        
        if not self.particles_file:
            self.particles_file = Cards.Particles_file(filepos)
            
    def find_pid_to_label(self):
        
        if hasattr(self, 'pid_to_label'):
            return self.pid_to_label
        
        self.pid_to_label=self.particles_file.give_pid_to_label()
        return self.pid_to_label
        
#1 ################################################################################# 
class TF_input(XML_input):
    """ This class extend the XML input containing routines containing writing output """

    #2 #############################################################################     
    def __init__(self,name):
        XML_input.__init__(self)
        self.tf_name=name

    #2 ############################################################################# 
    def create_ordering_file(self):
        """ take input form TF_input.dat and insert in ordering_file.inc"""
        
        #collect particle information
        thin=[]
        large=[]
        with_x=0
        theta = []
        phi = []
        for block in self.block.values():
            if block.order==1:
                thin+=block.particles
            elif block.order==2:
                large+=block.particles
            if "x1" in block.particles or "x2" in block.particles:
                with_x=1
            if block['THETA'].tf_code.strip() != 'tf=1d0' or \
                block['THETA'].width_code.strip() != 'width=0d0':
                theta += block.particles
            if block['PHI'].tf_code.strip() != 'tf=1d0' or \
                block['PHI'].width_code.strip() != 'width=0d0':
                phi += block.particles            

        #define the rule of how modif file transfer_function/input/ordering_file.inc        
        modif_rule={}
        modif_rule['THIN']=','.join(thin)
        modif_rule['LARGE']=','.join(large)
        modif_rule['X']=str(with_x)
        modif_rule['NAME_TF']=self.tf_name
        modif_rule['THETA'] = ','.join(theta)
        modif_rule['PHI'] =  ','.join(phi)

        #modify file
        mod_file.mod_file('./input/ordering_file.inc',modif_rule,write='./ordering_file.inc')
             
    #2 ############################################################################# 
    def create_transfer_functions(self):
        """ define for each block the 3 functions TF and the three TF width """
     
        def create_optional_variable(text,blockname,variable,list_var=[]):
            """ replace all #1,#2 by fortran name """
            
            output=''
            Pattern=re.compile(r'''#(\d*)''')
            prov=Pattern.split(text)
            i=0
            while i<len(prov)-1:
                output+=prov[i]
                i+=1
                if not('tf_'+blockname+"_"+variable+"_"+prov[i] in list_var):
                    list_var.append('tf_'+blockname+"_"+variable+"_"+prov[i])
                output+='tf_'+blockname+"_"+variable+"_"+prov[i]+"(curr_tf)"
                i+=1    
            output+=prov[-1]
            return output,list_var
        
        template = mod_file.Mod_file(rule_file='./input/mod_generic')
        list_var=[] #list all variable needed to defined in order to be set in the transfer card
        
        text='$B$ TF_HEADER $E$'
        text=mod_file.mod_text(text,template.dico)
        for block in self.block.values():
            for variable in ['E','THETA','PHI']:
                tf_var='tf_'+variable+'_'+block.name
                new_text='$B$ GENERIC_TF $E$'
                width_var='width_'+variable+'_'+block.name  
                new_text+='$B$ GENERIC_WIDTH $E$'
                new_text=mod_file.mod_text(new_text,template.dico)  #extend generic
                new_rule={'tf_var':tf_var,'width_var':width_var}
                new_rule['tf_definition']=block[variable].tf_code
                new_rule['width_definition']=block[variable].width_code
                new_rule['tf_include']=block[variable].includetext
                new_text=mod_file.mod_text(new_text,new_rule) #change default variable in real one
                new_text,list_var=create_optional_variable(new_text,block.name,variable,list_var)
                new_text=MW_fct.put_in_fortran_format(new_text)
                text+=new_text
                
        text=MW_fct.put_in_fortran_format(text)
        ff=open('./transfer_function.f','w')
        ff.writelines(text)
        return list_var
    
    #2 ###################################################################################     
    def create_transfer_card(self,list_var):
        """ create the generic transfer_card linked to this transfer_functions """
        
        in_card=open("./input/transfer_card_generic.dat",'r')
        text=in_card.read()
        in_card.close()
        text += '$b$ S-COMMENT_# $b$\n'
        text+='To change the transfer function run ./bin/change_tf.py \n'
        text+='Current parametrization :'+self.tf_name+'\n'
        text += '$e$ S-COMMENT_# $e$'
        
        current_block=''
        current_var=''
        for fortran_var in list_var:
            tag,block,var,number=fortran_var.split('_',3)
            if block != current_block:
                current_block=block
                current_var=''
                text += '$b$ S-COMMENT_# $b$'
                text += self.block[block].info()
                text += '$e$ S-COMMENT_#$e$'
            if var!=current_var:
                current_var=var
                text+='BLOCK TF_'+block+'_'+var+'\n'
            text+='\t'+number+'\t 1d0 \t #\n'
        
        text=mod_file.mod_text(text,{}) #only S-COMMENT mod   
        out=open("transfer_card.dat",'w')
        out.writelines(text)
        out.close()

class Full_TF(TF_input,TF_with_particles):
    """ class containing both extension of XML_input """
    
    def __init__(self,name):
        """ load the different module"""
        TF_input.__init__(self,name)
        TF_with_particles.__init__(self,only_add=1)

class TF_block(dict):
    """ class containing the information for a specific block from TF_param.dat """
    
    #2 #############################################################################
    def __init__(self,name):
        self.name=name
        #init all to delta
        dict.__init__(self)
        self.update({'E':TF_on_var(),'THETA':TF_on_var(),'PHI':TF_on_var()})
        self.order=0
        self.infotext=''
        self.name=name
    
    #2 #############################################################################    
    def def_particles(self,particles):
        """ define self.particles content """
        """ still to define how to store this """
        particles=particles.replace(' ','')
        particles=particles.replace('\t','')
        particles=particles.replace('\n','')
        
        self.particles=particles.split(',')

    #2 #############################################################################    
    def def_width_type(self,text):
        """ define self.order content (0:delta/1:thin/2:large) """
   
        if "thin" in text.lower():
            self.order=1
        elif "large" in text.lower():
            self.order=2
 
    #2 #############################################################################    
    def def_info(self,text):
        """ store the information on the block (comming from the xml) """
        self.infotext=text

    #2 #############################################################################    
    def info(self):
        text = 'Parameter for particles: '+','.join(self.particles)+'\n'
        text += 'Information:'+self.infotext+'\n'
        
        return text    

    

#1 ################################################################################## 
class TF_on_var:
    """ class containing the TF/WIDTH for a specific type """

    #2 #############################################################################     
    def __init__(self):
        """ initialize the content to delta """
        self.tf_code=' tf=1d0'
        self.width_code=' width=0d0' 
        self.includetext= '' 

    #2 #############################################################################    
    def def_include(self,text):
        """ store the information on the block (comming from the xml) """
        self.includetext=text
    

    #2 ############################################################################# 
    def change_tf(self,text):
        self.tf_code=text
        
    #2 #############################################################################         
    def change_width(self,text):
        self.width_code=text
        
#1 #################################################################################    
class TF_in_SubProcesses:
    """ Class for building transfert functions routine in a specific SubProcess"""

    #2 #############################################################################    
    def __init__(self,TF_data,MW_sub):
        """ TF_input: object of class TF_with_particles containing all the information of TF_param.dat
            MW_sub: string of the name of the MW subprocess in which we are going to work
        """
        
        self.TF_data=TF_data
        TF_data.load_particles_file() #load particles.dat (if not already done)
        self.sub=MW_sub
        self.dir='./SubProcesses/'+MW_sub
        pid_list=self.charge_particle_content()
        self.blockname_list=self.define_tf_block_for_particles(pid_list)        
        
    #2 #############################################################################        
    def charge_particle_content(self):
        """ find the ingoing and outgoing particles for this Subprocess
            store this information in self.particles=[list of PID]
            return the dictionary
        """
        self.particles = Cards.read_leshouches_file(self.dir+'/leshouche.inc')
        return self.particles

    #2 #############################################################################            
    def define_tf_block_for_particles(self,pid_list):
        """find for all particles in which tf_block she is in 
           return [list of tf_block]. tf_block is the string name of the tf block, 
           0 if it's delta related and -1 if it's an invisible particles
        """
        out=[]
        out.append(self.block_for_thispid('x1'))
        out.append(self.block_for_thispid('x2'))
        for pid in pid_list[2:]:
            out.append(self.block_for_thispid(pid))
        return out

    #2 #############################################################################     
    def block_for_thispid(self,pid):
        """ find in which TF, the particle pid is
            if it is delta return 0 
            if it's a invisible particles return -1
        """
        try:
          pid_to_label=self.TF_data.find_pid_to_label()     #return {pid:label} 
        except:
          pid_to_label={1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 11: 'e', 12: 've', 13: 'mu', 14: 'vm', 15: 'ta', 16: 'vt', 21: 'g', 22: 'A', 23: 'Z', 24: 'W', 25: 'h'}

        label_to_block=self.TF_data.find_label_to_block() #return {label:block}
        
        if pid in ['x1','x2']: #treat special case for initial particles
            if label_to_block.has_key(pid):
                return label_to_block[pid]
            else:
                return -1
        pid=abs(int(pid))
        if pid in particle_class.invisible_list:
            return -1
        if label_to_block.has_key(pid_to_label[pid]):
            return label_to_block[pid_to_label[pid]]
        else:
            return 0
        
    #2 #############################################################################    
    def write_transfer_function_file(self):
        """ write transfer_function.f file for the specific subprocesses """
        
        text='$B$ TF_HEADER $E$\n'
        text+=self.text_get_central_point()+'\n'
        text+=self.text_transfer_fct()+'\n'
        text+=self.text_tf_E_for_part()+'\n'
        
        template = mod_file.Mod_file(rule_file='./Source/MadWeight/transfer_function/input/mod_generic')
        text=mod_file.mod_text(text,template.dico)
        text=MW_fct.put_in_fortran_format(text)
        
        ff=open(self.dir+'/call_TF.f','w')
        ff.writelines(text)
        ff.close()
        
        
    #2 #############################################################################     
    def text_get_central_point(self):
        """ return the get_centralPoint function in a  unformated text 
            (need to pass in mod_file for comment, and to f77_format    
        """
        
        #Comment-subroutine-definition
        text='$B$ START_CENTRAL_POINT $E$\n'
        
        external_done=[]
        #add the definition for external function
        for block in self.blockname_list:
            if isinstance(block, basestring):
                name_list='width_E_'+block+', width_THETA_'+block+', width_PHI_'+block
            else:
                continue
            
            #avoiding to redefine the same thing twice
            if block not in external_done:
                text+=' external '+name_list+'\n'
                text+=' double precision '+name_list+'\n'
                external_done.append(block)
        text+= '      do perm =1,NPERM\n call get_perm(perm, perm_id)\n'
        #start the definition
        text+='\n$b$ S-COMMENT_C $b$ Start the definition $e$ S-COMMENT_C $e$\n'
        for i in range(0,len(self.blockname_list)):
            blockname=self.blockname_list[i]
            
            #define central point
            if blockname != -1: #particle is visible
                text+=' c_point(perm, %s,1,1)=theta(pexp_init(0,2+perm_id(%s)))\n' % (i+1,i-1)
                text+=' c_point(perm, %s,2,1)=phi(pexp_init(0,2+perm_id(%s)))\n' % (i+1,i-1)
                text+=' c_point(perm, %s,3,1)=rho(pexp_init(0,2+perm_id(%s)))\n' %(i+1,i-1)
            
            #define width
            variable=['THETA','PHI','E']
            for j in range(1,4):
                if blockname == -1: #particle is invisible 
                    text+=' c_point(perm,%s,%s,2)=-1d0\n' % (i+1,j)    
                elif blockname == 0: # in delta mode
                    text+=' c_point(perm,%s,%s,2)=0d0\n' % (i+1,j)
                else: #visible particles with TF define
                    text+=' c_point(perm,%s,%s,2)=width_%s_%s(pexp_init(0,2+perm_id(%s)),tag_lhco(%s))\n' % (i+1,j,variable[j-1],blockname,i-1,i+1)
            text+='\n' #space between particles definition
                 
        text+='\n enddo \n return\n end\n'
        return text

    #2 #############################################################################     
    def text_transfer_fct(self):
        """ return the transfer_fct function in a  unformated text 
            (need to pass in mod_file for comment, and to f77_format    
        """

        #Comment-subroutine-definition
        text='$B$ START_TRANSFER_FCT $E$\n' 
        
        met=0
        if self.TF_data.label_to_block.has_key('met'):
            met = self.TF_data.label_to_block['met']
            #add new definition
            text+=' double precision p_met_exp(0:3),p_met_rec(0:3)\n'
            text+=' integer tag_init(3:nexternal),type(nexternal),run_number,trigger\n'
            text+=" double precision eta_init(nexternal),phi_init(nexternal),pt_init(nexternal),j_mass(nexternal),ntrk(nexternal),btag(nexternal),had_em(nexternal),dummy1(nexternal),dummy2(nexternal)\n"
            text+=" common/LHCO_input/eta_init,phi_init,pt_init,j_mass,ntrk,btag,had_em,dummy1,dummy2,tag_init,type,run_number,trigger\n"
            text+=" integer met_lhco,opt_lhco\n   common/LHCO_met_tag/met_lhco,opt_lhco\n"
            #init met
            text+="  do i=0,3\n p_met_rec(i)=0d0\n enddo\n"
            
        text+=' weight=1d0\n'
        for i in range(0,len(self.blockname_list)):
            blockname=self.blockname_list[i]
            
            if not isinstance(blockname, basestring):
                if met and blockname == -1 and i>2: #invisible particlule but not the initial part
                    text+=' do i=0,3\n p_met_rec(i)=p_met_rec(i)+p(i,%s)\n enddo\n' %(i+1)
                continue
            
            text+=' n_lhco=tag_lhco(%s)\n' % (i+1)
            for var in ['E','THETA','PHI']:
                text+=' call tf_%s_%s(pexp(0,%s),p(0,%s),n_lhco,weight)\n' %(var,blockname,i+1,i+1)
            text+='\n'#space between particles definition
        
        if met:
            text+=' k=met_lhco\n'
            text+=' call four_momentum_set2(eta_init(k),phi_init(k),pt_init(k),j_mass(k),p_met_exp)\n'
            text+=' call tf_E_%s(p_met_exp,p_met_rec,met_lhco,weight)\n' %(met)

        text+="\n call check_nan(weight)\n return \n end\n"
        return text
        
    #2 #############################################################################     
    def text_tf_E_for_part(self):
        """ return the different tf_E_for_XX function in a  unformated text 
            (need to pass in mod_file for comment, and to f77_format    
        """    

        text2='' #text containing all the single call
        #Comment-subroutine-definition-init weight
        text='$B$ START_TF_E_FOR_PART $E$\n'
        
        for i in range(0,len(self.blockname_list)):
            text2+='\n'+self.text_tf_E_for_one_part(i)+'\n'
            blockname=self.blockname_list[i]
            
            if not isinstance(blockname, basestring):
                text+=' if(MG_num.eq.%s) then\n tf_E_for_part=1d0\n return\n endif\n' % (i+1)
            else:
                text+=' if(MG_num.eq.%s) then\n' %(i+1)
                text+=' tf_E_for_part=1d0\n'
                text+=' n_lhco=tag_lhco(%s)\n'% (i+1)
                text+=' call tf_E_%s(pexp(0,%s),momenta(0,%s),n_lhco,tf_E_for_part)\n' % (blockname,i+1,i+1)
                text+='\n return\n endif\n'        
        
        text+="\n return \n end\n"
        return text+text2
    
    #2 #############################################################################     
    def text_tf_E_for_one_part(self,i):
        """ return the different tf_E_for_XX function in a  unformated text 
            (need to pass in mod_file for comment, and to f77_format    
        """  
        
        text='$B$ S-COMMENT_C $B$      Subroutine: tf_E_for_XX()\n'
        text+='\n     purpose: returns the value of the transfer function (in energy)\n'
        text+='$E$ S-COMMENT_C $E$\n'
        text+=' double precision function tf_E_for_%s()\n\n' % (i+1)    
       
        #Comment-subroutine-definition-init weight
        text+='$B$ DEF_TF_E_FOR_ONE_PART $E$\n'
        
        blockname=self.blockname_list[i]            
        if not isinstance(blockname, basestring):        
            text+=' tf_E_for_%s=1d0\n' %(i+1) 
        else:
            text+=' tf_E_for_%s=1d0\n' %(i+1)
            text+=' n_lhco=tag_lhco(%s)\n'% (i+1)
            text+=' call tf_E_%s(pexp(0,%s),momenta(0,%s),n_lhco,tf_E_for_%s)\n' % (blockname,i+1,i+1,i+1)       
        
        text+="\n return \n end\n"
        return text    
    
    
#1 #################################################################################    
def create_param_inc(list_var):
    
    out=open("TF_param.inc",'w')
    file_in=open("./input/TF_param_generic.inc",'r')
    out.writelines(file_in.read())
    file_in.close()


    if list_var==[]:
        print "TF_param created (no input)"
        return
    
    common_text=''
    for name in list_var:
        name = name.replace('(curr_tf)','')
        line="        double precision  "+name+"(nb_tf)\n"
        out.writelines(line)
        common_text+=name+','
    common_text=common_text[:-1] #suppress last coma

    line="       Common/to_TF_param/"+common_text
    line=MW_fct.put_in_fortran_format(line)
    out.writelines(line)
    out.close()
    return

def create_ident_card(list_var):

     ff=open("./input/ident_mw_card_generic.dat",'r')
     out=open("ident_mw_card.dat",'w')
     #copy generic file
     out.writelines(ff.read())

     for name in list_var:
         data=name.split('_')
         line="TF_"+data[1].upper()+'_'+data[2].upper()
         line+='             '+data[3]+"       "+name+" real  \n" 
         out.writelines(line)

     return
 
    
def  create_rw(list_var):

    file_in=open("./input/rw_tf_generic.f","r")
    file_out=open("./rw_tf.f","w")

    Pattern=re.compile(r'''\$\$ADD_HERE\$\$''')
    while 1:
        line=file_in.readline()
        if line=="":
            break
        if not(Pattern.search(line)):
            file_out.writelines(line)
        else:
            #We are in position to put data info
            for name in list_var:
               file_out.writelines("      call get_real_t(npara,param,value,\""+name+"\"    ,"+name+",   1d0)\n")
            file_out.writelines(file_in.read())
            break
    file_in.close()
    file_out.close()

    return

def create_version(name):
    """ standard version number DIRNAME_X.Y written in Transfer_FctVersion.txt
           DIRNAME: name of the directory
           X and Y: version number coming from the new_transfer function
    """

    #load version number:
    ff=open('./Transfer_FctVersion.txt','r')
    line=ff.readline().split(':',1)
    ff.close()

#dirname:
#    dirname=os.path.basename(os.getcwd())

    #writefile:
    ff=open('./Transfer_FctVersion.txt','w')
    ff.writelines(name+':'+line[1])
    ff.close()

    return


def update_dir(name,make,MW_dir):

    main='../../../'

    os.system("cp ./ident_mw_card.dat "+main+"/Cards/")
    os.system("cp transfer_card.dat "+main+"/Cards/")
    os.system("cp data/transfer_card_"+name+".dat "+main+"/Cards/transfer_card.dat &>/dev/null")
    os.system("cp data/transfer_card_"+name+".dat "+main+"/Cards/transfer_card_default.dat &>/dev/null")
    if make:
        os.chdir(main+"/Source/")
        os.system("make")
        for directory in MW_dir:
            os.chdir(main+"/SubProcesses/"+directory)
            os.system("ln -s ../../Source/MadWeight/transfer_function/TF_param.inc TF_param.inc")
            os.system("ln -s ../../Source/MadWeight/transfer_function/nb_tf.inc nb_tf.inc")
            os.system("make")
            os.chdir('../../')            
    else:
        os.system("make ")    # always compile libTF ...
        os.chdir(main)
        for directory in MW_dir:
            os.chdir("SubProcesses/"+directory)
            os.system("ln -s ../../Source/MadWeight/transfer_function/TF_param.inc TF_param.inc")
            os.system("ln -s ../../Source/MadWeight/transfer_function/nb_tf.inc nb_tf.inc")
            os.chdir('../../')            

    #charge card
    ident=Cards.Card('./Cards/ident_mw_card.dat')
    madweight=Cards.Card('./Cards/MadWeight_card.dat')
    transfer=Cards.Card('./Cards/transfer_card.dat')

    #create output
    madweight.create_include_file(ident,'./Source/madweight_card.inc')
    transfer.create_include_file_tf(ident,'./Source/MadWeight/transfer_function')

    os.chdir('./Source/MadWeight/transfer_function')

                                
#def check_valid(text):
#    for name in forbiden:
#        Pattern=re.compile(name,re.I)
#        if Pattern.search(text):
#            print "ERROR: invalid usage of statement: ",name
#            print "for security reason transfer functions can't use this statement"
#            sys.exit()
#    return

def extract_tf_name(filepos='./Cards/proc_card.dat'):
    """ read the file to find the requested change of variable"""
    
    found=0
    for line in file(filepos):
        if found:
            name=line.split()[0] #remove blank spae,end of line,...
            return name
        if line.startswith('# Begin transfer_function'):
            found=1

        
###########################   TEST   #################################
if(__name__=="__main__"):

    import MW_param
    MW_param.go_to_main_dir()

    P_dir,MW_dir=MW_param.detect_SubProcess(P_mode=1)

    opt=sys.argv
    if len(opt)<2:
        listdir=os.listdir('./Source/MadWeight/transfer_function/data')
        print 'Available TF function:\n   ',
        print '\n    '.join([content[3:-4] for content in listdir if (content.startswith('TF') and content.endswith('dat'))])
        name=raw_input('Choose your Transfer Function\n')
    else:
        name=opt[1]
        if name in ['proc_card.dat','auto']:
            name=extract_tf_name('./Cards/proc_card.dat')
    if len(opt)==3:
        made_make=int(opt[2])
    else:
        made_make=0
    os.chdir('./Source/MadWeight/transfer_function')
    create_TF_main(name,made_make,MW_dir)
    
##    file=raw_input("file: ")
##    ff=open(file,'r')
##    gg=open(file+'_70.f','w')
##    text=ff.read()
##    text=MW_fct.put_in_fortran_format(text)
##    gg.writelines(text)
