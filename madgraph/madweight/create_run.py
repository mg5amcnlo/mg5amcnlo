#!/usr/bin/env python

#Extension
import string,os,sys,re,popen2,time,stat,filecmp

try: 
    import madgraph.madweight.mod_file as mod_file
    import madgraph.madweight.change_tf as change_tf
    #import madgraph.various.progressbar as progressbar
except ImportError:
    import internal.madweight.mod_file as mod_file
    #import internal.madweight.progressbar as progressbar


###########################################################################
##                                CONTENT                                ##
##                                -------                                ##
##                                                                       ##
##    + update_cuts_status                                               ##
##         activate/desactivate the cuts in ME/MW                        ##
##     + cut_is_active (send cut status)                                 ##
##     + check_Subprocesses_update (check if Subprocces/(MW_?)P files    ##
##         follows the last version of the file -protection against cp-) ##
##    + active_acceptance_run                                            ##
##    + desactive_acceptance_run                                         ##
##    + del_old_dir                                                      ##
##    + create_dir                                                       ##
##    + copy_file                                                        ##
##    + put_data                                                         ##
##    + restore_last_info_for_control (not use anymore)                  ##
##    + create_all_MWdir                                                 ##
##    + create_all_Pdir                                                  ##
##    + create_all_schedular                                             ##
###########################################################################


###########################################################################
###################           VARIABLE GLOBALE           ##################
###########################################################################
opt=sys.argv
write = sys.stdout.write


class NoMoreEvent(Exception):
    pass

###########################################################################
###################     GESTION OF CUT (des)ACTIVATE     ##################
###########################################################################

def update_cuts_status(MW_param):
    """ remove the call to the cuts if asked in MadWeight_Card.dat """

# Pierre: pass
    return

    #check cut status for cross section:
    if (MW_param.info['mw_run']['use_cut']+cut_is_active('cuts.f'))%2: #pass if not coherent between asked and actual status
        file_to_mod=['./SubProcesses/cuts.f']
        rule=['./Source/MadWeight/mod_file/suppress_cuts_MG']
        #modify file
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF','DESACTIVATE_BW_CUT']"""})
        #check update
        check_Subprocesses_update('cuts.f',MW_param.P_listdir,'./Source/MadWeight/mod_file/suppress_cuts_MG')

    #check cut status for Weight computation:
    if (MW_param.info['mw_run']['use_cut']+cut_is_active('cuts_MW.f'))%2: #pass if not coherent between asked and actual status
        file_to_mod=['./SubProcesses/cuts_MW.f']
        rule=['./Source/MadWeight/mod_file/suppress_cuts_MW']
        #modify file
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','TO_SPECISA','DESACTIVATE_BW_CUT']"""})
        #check update
        check_Subprocesses_update('cuts_MW.f',MW_param.MW_listdir,'./Source/MadWeight/mod_file/suppress_cuts_MW')

    #check BW cut status for cross section
    if (MW_param['mw_run']['bw_cut']+bw_cut_is_active('cuts.f'))%2: #pass if not coherent between asked and actual status
        file_to_mod=['./SubProcesses/cuts.f']
        rule=['./Source/MadWeight/mod_file/suppress_BW_cuts']
        #modify file
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','MW_NEW_DEF']"""})
        #check update 
        check_Subprocesses_update('cuts.f',MW_param.P_listdir,'./Source/MadWeight/mod_file/suppress_BW_cuts')

    #check cut status for Weight computation:
    if (MW_param.info['mw_run']['bw_cut']+bw_cut_is_active('cuts_MW.f'))%2: #pass if not coherent between asked and actual status
        file_to_mod=['./SubProcesses/cuts_MW.f']
        rule=['./Source/MadWeight/mod_file/suppress_BW_cuts']
        #modify file               
        mod_file.mod_file(file_to_mod,rule,opt={'nowarning':"""['PASSCUTS','TO_SPECISA']"""})
        #check update
        check_Subprocesses_update('cuts_MW.f',MW_param.MW_listdir,'./Source/MadWeight/mod_file/suppress_BW_cuts')





def cut_is_active(filename):
    """ check is the cut is active or not """

# Pierre: pass the fucntion 
    pass

#    for line in file('./SubProcesses/'+filename):
#        if line.count('DESACTIVATE_CUT'):
#            return 1
#        elif line.count('ACTIVATE_CUT'):
#            return 0

def bw_cut_is_active(filename):
    """ check is the bw_cutt is active or not """
    for line in file('./SubProcesses/'+filename):
        if 'DESACTIVATE_BW_CUT' in line:
            return 1
        elif 'ACTIVATE_BW_CUT' in line:
            return 0
    raise Exception



class check_Subprocesses_update:
    """ check if Subprocces/'listdir' files follows the last version of
    ./SubProcesses/'filename'. This is a protection against copy files
    -normally they are just linked-

    different mode are available
    warning:
      0: no warning apply modification
      1: warning and apply modification
      2: warning and user choice
      3: warning but no modification
      4: no warning and no modification
      5: warning and raising error
    modifrule:
      '', copy the original file
      else: apply the modification asked in modifrule

    """

    class ERROR_DifferenceInFile(Exception):pass


 
    def __init__(self,filelist,listdir,modifrule='',warning=1,run=1):
        
        #chek type input:
        if type(filelist)==str:
            self.filelist=[filelist]
        else:
            self.filelist=filelist
        if type(listdir)==str:
            self.listdir=[listdir]
        else: self.listdir=listdir

        self.modifrule=modifrule

    
        #assign tag mode
        usewarning=0
        if int(warning) in [1,2,3,5]:
            self.usewarning=1
        
        self.usemodif=0
        self.intmode=0
        self.raiseerror=0
        if int(warning) in [0,1]:
            self.usemodif=1 #apply
        elif int(warning)==2:
            self.intmode=1 #interactive mode
        elif int(warning)==5:
                self.raiseerror==1 #raising error
        
        if run:
            self.compare()
    
    
    def compare(self):
        #main loop
        for dirname in self.listdir:
            for filename in self.filelist:
                status=filecmp.cmp('./SubProcesses/'+filename,'./SubProcesses/'+dirname+'/'+filename)

            if status:
                continue #continue check if ok

            #warning
            if self.usewarning:
                self.printwarning('./SubProcesses/'+filename,'./SubProcesses/'+dirname+'/'+filename)
                
            #error   
            if self.raiseerror:
                raise self.ERROR_DifferenceInFile

            #interactive mode
            if self.intmode:
                self.usemodif=self.printintmode('./SubProcesses/'+dirname+'/'+filename)

            #modif file
            if self.usemodif:
                self.modiffile('./SubProcesses/'+dirname+'/'+filename,self.modifrule)


    def printwarning(self,filename1,filename2):
        
                print """ WARNING: those file are supposed to be identical (symbolic link):\n\
                          ./SubProcesses/"""+filename1+"""\n./SubProcesses/"""+filename2
                print """ define the tag MW_run/901 to change the rule for modification """
                if self.usemodif==0:
                    print """ no modification to the file are done """
                elif self.usemodif==1:
                    print """ modify the file """

    def printintmode(self,file):

        a=raw_input('modify file '+file+'with rule'+self.modifrule+'? (y/n)')
        if a=='y':
            return 1
        elif a=='n':
            return 0
        else:
            a=self.printintmode(file)
            return a
        
    def modiffile(self,file,rule):
        mod_file.mod_file(file,rule)

###########################################################################
#########          ACTIVATE/DESACTIVATE ACCEPTANCE TERM         ###########
###########################################################################
class AcceptanceError(Exception): pass

def activate_acceptance_run():
    """ 1. reactivate the write of events .lhe
        2. modify combine_event.f
        3. modify pythia-pgs/src/pythia.f  --> Pass in standard in special package
        4. modify pythia-pgs/src/pgs.f     --> Pass in standard in special package
        5. compile those modification
    """
    #0. test if the card/directory exist
    if not( os.path.isfile('./Cards/pythia_card.dat') and
            os.path.isfile('./Cards/pgs_card.dat') and
            os.path.isdir('../MW_pythia-pgs')):
        raise AcceptanceError, 'Cards or MW_pythia-pgs missing... impossible to load acceptance module'
        
    #1. test if the acc is already loaded
    if os.path.isfile('./Source/MadWeight_file/acc.in'):
        return
    os.system('touch ./Source/MadWeight_file/acc.in')

    #2. reactivate the write of events .lhe
    mod_file.mod_file('./SubProcesses/unwgt.f',{'S-DECOMMENT_C':''})

    #3. change combine_events.f
    mod_file.mod_file('./Source/combine_events.f','./Source/MadWeight/mod_file/mod_combine_events')

    #4. modify pythia_card.dat
    mod_file.mod_file('./Cards/pythia_card.dat','./Source/MadWeight/mod_file/mod_combine_events')

    #4. modify pythia-pgs/src/pythia.f
    #os.system('cp ../pythia-pgs/src/pythia.f ../pythia-pgs/src/pythia_default.f')
    #rule={}
    #rule['def_file']=change_tf.put_in_fortran_format("""
    # write(*,*) 'enter input/output file for MW run'
    # read(*,*) input_file,output_file
    #pythia_card='../Cards/pythia_card.dat'      
    # nfiles=1
    #""")
    #mod_file.mod_file('../pythia-pgs/src/pythia.f',rule)
    
    #4. modify pythia-pgs/src/pgs.f
    #os.system('cp ../pythia-pgs/src/pgs.f ../pythia-pgs/src/pgs_default.f')
    #rule={}
    #rule['def_file']=change_tf.put_in_fortran_format("""
    #    write(*,*) 'enter input file for MW run'
    #    read(*,*) pgs_input_file
    #    write(*,*) 'enter output file for MW run'
    #    read(*,*) pgs_output_file
    #""")
    #mod_file.mod_file('../pythia-pgs/src/pgs.f',rule)

    #5.compile the new files
    os.system("cd Source;make ../bin/combine_events;cd -")
    os.system("cd ../MW_pythia-pgs;make &>/dev/null")


def desactivate_acceptance_run():
    """ check that the event are not written """
    
    if os.path.isfile('./Source/MadWeight_file/acc.in'):
        import expand_MadWeight
        expand_MadWeight.expand_all(echap=['MadWeight_card.dat'])
        os.system('rm ./Source/MadWeight_file/acc.in')
    

def restore_pythia_pgs():
    pass
#    os.system('cp ../pythia-pgs/src/pythia_default.f ../pythia-pgs/src/pythia.f')
#    os.system('cp ../pythia-pgs/src/pgs_default.f ../pythia-pgs/src/pgs.f')
#    os.system("cd ../pythia-pgs;make &>/dev/null")
    

###########################################################################
#########          CREATION DES DOSSIER/FICHIERS DE SORTIE      ###########
###########################################################################

#1 ##################################################################################
class create_dir:
    """create all the directory for the run """

    #2 ##############################################################################
    def __init__(self,MWparam):

        self.MWparam=MWparam
        self.dir_name=MWparam.name

        self.ref_card=self.MWparam.actif_param[0] #card where the file verif.lhco are written and not linked
        self.created=0

    #2 ##############################################################################
    def all(self):
        
        print 'creating all directories'

# Pierre
#        if self.MWparam.norm_with_cross:
#            for dir in self.MWparam.P_listdir:
#                self.all_dir(dir)
#        else:
#            pass

        if self.MWparam.nb_event:
            for dir in self.MWparam.MW_listdir:
                self.all_dir(dir)

    #2 ##############################################################################
    def all_dir(self,dir):
        """creates the directory for standard run """

        self.dir_type='M' #  dir[0]  Pierre: only one dir type (MadWeight type)
        self.Sdir_pos='./SubProcesses/'+dir
        try:
            os.mkdir('./SubProcesses/'+dir)
        except:
            pass


        if self.dir_type=='M':
            if self.MWparam['mw_run']['22']:
                self.add_events()
            else:
                print 'create M dir'
                self.create_M_dir()
        else:
            if self.MWparam['mw_run']['22']:
                return
            self.create_P_dir()
            
        print 'created',self.created,'directories'
        self.created=0

    #2 ##############################################################################
    def update_card_status(self,cardref=-1,number_of_event=-1):
        """creates the directory for standard run
           number_of_event[-1]:-1:automatic (number of event of cardref if !=-1, highest existing event in other case 
        """

        #check if all card have a directory in P_
        list_card=[num for num in self.MWparam.actif_param]
        #progress bar
        pbar = progressbar.progbar('update Pdir',len(list_card)*len(self.MWparam.P_listdir))
        for directory in self.MWparam.P_listdir:
            self.dir_type=directory[0]
            self.Sdir_pos='./SubProcesses/'+directory
            for card in list_card:
                self.create_one_P_dir(card,remove_old=0)
                pbar.update()
        pbar.finish()

        #find cardref and number_of_event if not defined
        if number_of_event==-1 and cardref==-1:           #search the card with the maximum of event defined
            self.dir_type='M'
            self.Sdir_pos='./SubProcesses/'+self.MWparam.MW_listdir[0]
            for card in self.MWparam.actif_param:
                event=self.find_exist_event(card)
                if event>=number_of_event:
                    cardref=card
                    number_of_event=event
        elif cardref==-1:
            ref_num=number_of_event-1 #-1 due to the starting at zero
            for card in self.MWparam.actif_param:
                if self.find_exist_event(card)>=ref_num:
                    cardref=card
                                            
        if cardref==-1 or number_of_event==-1:
            sys.exit('impossible to update card and/or number of event not defined')
        else:
            self.ref_card=cardref
                                            
        #update events
        #progress bar
        list_card=self.MWparam.actif_param
        list_event=range(0,number_of_event)
        if (len(list_card))*(len(list_event))*len(self.MWparam.MW_listdir)>0:
            pbar = progressbar.progbar('update MWdir',(len(list_card))*(len(list_event))*len(self.MWparam.P_listdir))
        for dir in self.MWparam.MW_listdir:
            self.dir_type=dir[0]
            self.Sdir_pos='./SubProcesses/'+dir

            #then pass in update mode for all the card
            for card in list_card:
                for event in list_event:
                    self.create_one_M_dir(card,event,remove_old=0)
                    pbar.update()
        pbar.finish()
                                                                                                
                                                                                                                                    
    #2 ##############################################################################
    def add_events(self):
        
        self.file_event=open(self.Sdir_pos+'/'+self.MWparam.name+'/verif.lhco')
        self.line_event=self.file_event.readline()

        nb_exist_event=self.find_exist_event()
        list_card=self.MWparam.actif_param
        list_event=range(nb_exist_event,nb_exist_event+self.MWparam.nb_event)
        pbar = progressbar.progbar('create_dir',(len(list_card))*(len(list_event)))
        for card in list_card:
            for event in list_event:
                self.create_one_M_dir(card,event)
                pbar.update()
        pbar.finish()
                                                                
        

    #2 ###############################################################################
    def find_exist_event(self,card_nb=-1):
        """ find the maximal event number in this SubProcess"""

        if card_nb==-1:
            card_nb=self.ref_card
        directory=self.Sdir_pos+'/'+self.MWparam.name

        return self.MWparam.find_existing_events(directory,card_nb)

    #2 ##############################################################################
    def create_M_dir(self):

        self.file_event=open(self.Sdir_pos+'/'+self.MWparam.name+'/verif.lhco')
        self.line_event=self.file_event.readline()
        dirname = self.Sdir_pos.split('/')[-1]
        self.del_old_dir()
        list_card=self.MWparam.actif_param

        #progress bar
        pbar = progressbar.progbar('create_dir',(len(list_card))*(self.MWparam.nb_event_MW[dirname]))
        for card in list_card:
            list_event=range(0, self.MWparam.nb_event_MW[dirname])
            try:
                os.mkdir(self.Sdir_pos+'/'+self.MWparam.name+'/card_'+str(card))
            except:
                pass
            for event in list_event:
                try:
                    self.create_one_M_dir(card,event)
                except NoMoreEvent:
                    os.system('rm -rf %s' % self.Sdir_pos+'/'+self.MWparam.name+'/card_'+str(card))
                    self.MWparam.nb_event_MW[dirname] = event
                    break
                
                pbar.update()
        pbar.finish()

    #2 ##############################################################################
    def create_P_dir(self):

        self.del_old_dir()
        list_card=[num for num in self.MWparam.actif_param]
        #progress bar
        pbar = progressbar.progbar('create_dir',len(list_card))
        for card in list_card:
            self.create_one_P_dir(card)
            pbar.update()
        pbar.finish()

    #2 ##############################################################################
    def create_one_M_dir(self,card,event,remove_old=1):
        """ create the directory for the event \"event\" and the card nb \"card\"
        """
        dir_name=self.MWparam.name
        pos=self.Sdir_pos+'/'+dir_name+'/card_'+str(card)+'/event_'+str(event)+'/'
        #check that upper level exist
        if not os.path.isdir(self.Sdir_pos+'/'+dir_name+'/card_'+str(card)):
            os.mkdir(self.Sdir_pos+'/'+dir_name+'/card_'+str(card))
            
#        if not os.path.isdir(pos):
#            os.mkdir(pos)

        try:
          os.mkdir(pos)
        except OSError:
          if remove_old:
            os.system('rm '+pos+'/* >/dev/null')
#          else:
#            return
            
            
        ff=open(pos+'/param.dat','w')
        ff.writelines('param_card_'+str(card)+'.dat\n')
        ff.writelines(str(self.MWparam['mw_run']['MW_int_points'])+'\n')
        ff.close()

        if card==self.ref_card:
            data=self.give_new_exp_point()
            hh=open(pos+'/verif.lhco','w')
            hh.writelines(data)
            hh.close()
        else:
            try:
                os.symlink('../../card_'+str(self.ref_card)+'/event_'+str(event)+'/verif.lhco',pos+'/verif.lhco')
            except OSError:
                pass
        self.created+=1


    #2 ##############################################################################
    def create_one_P_dir(self,card,remove_old=1):
        """ create the directory for the event \"event\" and the card nb \"card\"
        """
        dir_name=self.MWparam.name
        pos=self.Sdir_pos+'/'+dir_name+'/card_'+str(card)
        try:
            os.mkdir(pos)
        except:
            if remove_old:
                os.system('rm '+pos+'/* >/dev/null')
            else:
                return
        ff=open(pos+'/param.dat','w')
        ff.writelines('param_card_'+str(card)+'.dat\n')
        ff.writelines(str(self.MWparam['mw_run']['MW_int_points'])+'\n')
        ff.close()
        os.system('ln -s ../../madevent.py '+pos+'/madevent.py')
        os.system('ln -s ../../madevent '+pos+'/madevent')
        os.system('ln -s ../../input_app.txt '+pos+'/input_app.txt')
        os.system('ln -s ../../symfact.dat '+pos+'/symfact.dat')
        #add link to a job:
        for file in os.listdir(pos+'/../../'):
            if len(file)>4 and file[:4]=='ajob':
                os.system('ln -s ../../'+file+' '+pos)
        
        self.created+=1
        
    Pattern=re.compile(r'''^\s*0\s+\d+\s+\d+\s*$''',re.I)
    #2 ##############################################################################
    def give_new_exp_point(self):

        data=self.line_event
        while 1:
            try:
                line=self.file_event.readline()
            except ValueError:
                raise NoMoreEvent
                
            if line=='':
                self.file_event.close()
                if data == '':
                    raise NoMoreEvent
                return data
            if self.Pattern.search(line):
                self.line_event=line
                return data
            else:
                data+=line
            
    #2 ##############################################################################
    def del_old_dir(self):
        ''' delete old event directory '''
        
        output=1
        #verification du format des dossiers
        list_dir=os.listdir(os.pardir)
        if self.MWparam.info['mw_run']['22']:
            print self.Sdir_pos.split('/')[-1],': no deleting'
            return output

        print self.Sdir_pos.split('/')[-1],': deleting old run directory'
        print os.getcwd()
        os.system('mkdir '+self.Sdir_pos+'/'+self.MWparam.name+'/')
        for datafile in os.listdir(self.Sdir_pos+'/'+self.MWparam.name):
            if datafile not in ['verif.lhco']:
                try: os.system('rm '+self.Sdir_pos+'/'+self.MWparam.name+'/'+datafile+' -rf')
                except:
                    print "WARNING: this directory ", os.getcwd()+'/'+self.Sdir_pos+'/'+self.MWparam.name," are not deleted"
                    output=0
                    

             
        return output
                                                                                                                                                                                                                                                                                         
###### copy file
def copy_file(list,i):
    pos=os.getcwd()
    init_dir=os.sep+pos.split(os.sep)[-1]
    
    title=[]
    content=[]

    for file in list:
        title.append(file.split('/')[-1])
        ff=open(file,'r')
        content.append(ff.readlines())
        ff.close()
        
#    for i in range(0,nmax):
    os.chdir('../'+dir_name+str(i))
    for i in range(0,len(title)):
            ff=open('./'+title[i],'w')
            ff.writelines(content[i])
            ff.close()
    os.chdir(os.pardir+init_dir)

def restore_last_info_for_control(MW_param):
    """ restore run info for controlling """
    
    pattern=re.compile(r'''condor_id/\s*(?P<run_list>.+)\s*/job/\s*(?P<job>\d*)''',re.DOTALL)
    launch_job=0
    run_job=[]
    dir_list=[]

    for dir in MW_param.P_listdir+MW_param.MW_listdir:
        ff=open('./SubProcesses/'+dir+'/schedular/condor.log','r')
        text=ff.readline()
        ff.close()
        pat=pattern.search(text)
        if pat:
            launch_job+=int(pat.group('job'))
            prov_job=eval(pat.group('run_list'))
            dir_list+=[dir]*len(prov_job)
            run_job+=prov_job
    return launch_job,run_job,dir_list


#def return_failed(dir,MW_param):
#    """read the failed process and return them in a list """
#
#    try:
#        ff=open('./SubProcesses/'+dir+'/'+MW_param.name+'/failed_job.dat','r')
#    except:
#        return []
#    out=[]
#    while 1:
#        line=ff.readline()
#        if line=='':
#            break
#        out.append(line.replace('\n',''))
#
#    return out


###########################################################################
###################             MAIN PROGRAM             ##################
###########################################################################
#def create_all_MWdir(dir,MW_param):
#    """n: number of data , name : name_run"""
#    
#    run_name=MW_param.name
#    try:
#        os.chdir('./SubProcesses/'+dir+'/schedular')
#    except:
#        os.mkdir('./SubProcesses/'+dir+'/schedular')
#        os.chdir('./SubProcesses/'+dir+'/schedular')
#
#    os.system('mkdir ../'+run_name+'&>/dev/null')
#    dir_sche=create_dir(MW_param)
#    dir_sche.all()
#    os.chdir("../../../")
#    print 'created',dir_sche.created,'directories'
#    return dir_sche.created

###########################################################################
###################            MAIN PROGRAM 2            ##################
###########################################################################

#def create_all_Pdir(dir,MW_param):
#    """n: number of data , name : name_run"""##
#
#    #re-init global parameter
#    run_name=MW_param.name
##    os.system(' mkdir ./SubProcesses/'+dir+'/schedular') 
#    os.chdir('./SubProcesses/'+dir+'/schedular')
#    os.system(' mkdir ../'+run_name+'&>/dev/null') 
#    dir_sche=create_dir(MW_param)
#    dir_sche.all()
#    os.chdir("../../../")
#    print 'created',dir_sche.created,'directories'
#    return dir_sche.created


def create_all_schedular(MWparam):

    for directory in MWparam.MW_listdir+MWparam.P_listdir:
        try:
            os.mkdir('./SubProcesses/'+directory+'/schedular')
        except:
            pass






##############################################################################################
##                                  Single Launch                                           ##
##############################################################################################
if __name__=='__main__':


#reupdate all the active card to a certain number of event
    #import global run opt
    import MW_param
    MW_param.go_to_main_dir()
    MWparam=MW_param.MW_info('MadWeight_card.dat')

    #launch the check
    create_obj=create_dir(MWparam)
    create_obj.update_card_status()
#    for dir in MWparam.MW_listdir:
#        create_obj.Sdir_pos='./SubProcesses/'+dir
#        print create_obj.find_exist_event(1)



