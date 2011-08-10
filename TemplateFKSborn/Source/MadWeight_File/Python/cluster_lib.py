#!/usr/bin/env python
##########################################################################
##                                                                      ##
##                               MadWeight                              ##
##                               ---------                              ##
##########################################################################
##                                                                      ##
##   author: Mattelaer Olivier (CP3)                                    ##
##       email:  olivier.mattelaer@uclouvain.be                         ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   license: GNU                                                       ##
##   last-modif:16/10/08                                                ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##                                                                      ##
##                         *** CLUSTER CLASS ***                        ##
##                                                                      ##
##      +cluster                                                        ##
##      |    +   init                                                   ##
##      |    C   driver                                                 ##
##      |    +   +    init                                              ##
##      |    +   +    call                                              ##
##      |    +   +    all                                               ##
##      |    +   +    create_all_dir                                    ##
##      |    +   +    compile_all                                       ##
##      |    +   +    launch_and_schedule                               ##
##      |    +   +    schedule                                          ##
##      |    +   +    launch                                            ##
##      |    +   +    control                                           ##
##      |    C   create_file                                            ##
##      |    |   +    init                                              ##
##      |    |   +    call                                              ##
##      |    |   +    all                                               ##
##      |    |   +    all_MadWeight                                     ##
##      |    |   +    all_MadEvent                                      ##
##      |    |   +    all_dir                                           ##
##      |    |   +    Event                                             ##
##      |    |   +    Card                                              ##
##      |    |   +    one                                               ##
##      |    C   launch                                                 ##
##      |    |   +    init                                              ##
##      |    |   +    call                                              ##
##      |    |   +    all                                               ##
##      |    |   +    all_MadWeight                                     ##
##      |    |   +    all_MadEvent                                      ##
##      |    |   +    directory                                         ##
##      |    |   +    one                                               ##
##      |    |   +    launch_submission_file                            ##
##      |    |   +    launch_one_submission_file                        ##
##      |    C   control                                                ##
##      |    |   +    init                                              ##
##      |    |   +    call                                              ##
##      |    |   +    all                                               ##
##      |    |   +    all_MadWeight                                     ##
##      |    |   +    all_MadEvent                                      ##
##      |    |   +    directory                                         ##
##      |    |   +    one                                               ##
##      |    |   +    str                                               ##
##                                                                      ##
##                    *** SUBMISSION FILE CLASS ***                     ##
##                                                                      ##
##      +submission_file                                                ##
##      |     +  init                                                   ##
##      |     +  activate_automatic_launch                              ##
##      |     +  clear                                                  ##
##      |     +  write                                                  ##
##      |     +  listing                                                ##
##      +submission_file_directory                                      ##
##      |     +  init                                                   ##
##      |     +  clear                                                  ##
##      |     +  write                                                  ##
##      |     +  find                                                   ##
##      |     +  listing                                                ##
##                                                                      ##
##                    *** Sub-CLUSTER ***                               ##
##      +    def_cluster                                                ##
##########################################################################
##
## BEGIN INCLUDE
##
import os
import time
import sys
import progressbar
##
## TAG
##
__author__="Mattelaer Olivier, the MadTeam"
__version__="1.1.1"
__date__="24 Mar 2009"
__copyright__= "Copyright (c) 2009 MadGraph/MadEvent team"
__license__= "GNU"
##
## BEGIN CODE
##


#################################################################################################
##                                     DEFAULT CLUSTER  CLASS                                  ##
#################################################################################################



## ERROR CLASS
class Clustererror(Exception): pass
class DirectoryError(Clustererror): pass

#1 ##############################################################################################
class cluster:
    """ default frame for cluster class """

    ##
    ## TAG VALUE
    ##
    tag=-1  #value for use this cluster type during the run.

    ##
    ## content
    ##
    ##   class driver
    ##   class create_file
    ##   class launch
    ##   class control



    #2 ##############################################################################################
    def __init__(self,MWparam):
        """ initialisation of the object """


        self.MWparam=MWparam  #object containing all run option
        self.submission_file=submission_file(MWparam,self)   #create class for treating schedulling file
        self.driver=eval('cluster.'+self.__class__.__name__+'.driver(self)') # link between object
        self.create_file=eval('cluster.'+self.__class__.__name__+'.create_file(self)')# link between object
        self.launch=eval('cluster.'+self.__class__.__name__+'.launch(self)') # link between object
        self.control=eval('cluster.'+self.__class__.__name__+'.control(self)') # link between object

    #2 ##############################################################################################
    def __str__(self):
        return "Cluster type "+str(self.tag)
        
    #2 ##############################################################################################
    #2 CLASS:  DRIVER
    #2 ##############################################################################################
    class driver:
        """ organize all the operation potentialy cluster dependant
            - creation of directory for each job
            - launch/relaunch
            - control
            - ...
        """

        #3 ##############################################################################################
        def __init__(self,mother):
            self.mother=mother
            self.MWparam=mother.MWparam
            self.Pmax_level=self.MWparam['mw_run']['nb_loop_cross']
            if self.MWparam['mw_run']['acceptance_run']:
                self.Pmax_level*=2
            
            if self.MWparam.run_opt['launch']:
                self.clean('full')
                self.compile_all()
            elif self.MWparam.run_opt['relaunch']:
                self.clean()
                self.compile_all()

        #3 ##############################################################################################
        def __call__(self):
            #print 'pass in call'
            self.all()

        #3 ##############################################################################################
        def all(self):
            """ Launch all subroutine """

            #control how to launch (by step or not)
            if hasattr(self.MWparam,'cluster_launch_mode'):
                # this variable is defined in binary
                #     ^2: create_dir (not implemented)
                #     ^1: schedule
                #     ^0: launch
                mode=self.MWparam.cluster_launch_mode
                launch=mode%2
                schedule=((mode-launch)//2)%2
                create_dir=(mode-2*schedule-launch)%2
            else:
                create_dir=1
                launch=1
                schedule=1

            #enter in mode
            self.create_all_dir()                
            if launch and schedule:
                self.launch_and_schedule()
            elif launch:
                self.launch()
            elif schedule:
                self.schedule()


            self.control()
            
            
        #3 ##############################################################################################
        def create_all_dir(self):
            """ Launch all subroutine """

            import create_run as Create

            print 'schedullar'
            Create.create_all_schedular(self.MWparam)
            if self.MWparam.run_opt['dir']:
                create_dir_obj=Create.create_dir(self.MWparam)
                create_dir_obj.all()
            return


            if self.MWparam.run_opt['dir']:
                if not self.MWparam.info['mw_run']['22']:
                    print 'creating all directories'
                    if self.MWparam.norm_with_cross:
                        for dir in self.MWparam.P_listdir:
                            Create.create_all_Pdir(dir,self.MWparam)
                else:
                    print 'creating new directories for additional events'
                    
                for dir in self.MWparam.MW_listdir:
                    Create.create_all_MWdir(dir,self.MWparam)
                print 'done'
                
        #3 ##############################################################################################
        def compile_all(self):
            """ Launch all subroutine """

            if self.MWparam.run_opt['launch'] or self.MWparam.run_opt['relaunch']:
                print 'compiling'
                #verify if everything is compiled and if include file are well defined        
                if self.MWparam.norm_with_cross:
                    dirlist=self.MWparam.P_listdir+self.MWparam.MW_listdir
                else:
                    dirlist=self.MWparam.MW_listdir

                for dir in dirlist:
                    os.chdir('./SubProcesses/'+dir)
                    os.system('make &>/dev/null')
                    os.chdir('../../')        
                print 'done'

        #3 ##############################################################################################
        def launch_and_schedule(self):
            """ create all the scheduling file and launch it directly"""

            self.mother.submission_file.activate_automatic_launch(self.mother)
            self.schedule()

        #3 ##############################################################################################
        def schedule(self):
            """ create all the scheduling file """


            if self.MWparam.run_opt['launch']:
                print """launch: create scheduling file"""
                self.mother.create_file()
                                            
            if self.MWparam.run_opt['relaunch']:
                print """relaunch: create scheduling file"""
                #clear submision file
                self.mother.submission_file.clear()
                #collect failed directory
                self.failed={}
                for dir in self.MWparam.P_listdir+self.MWparam.MW_listdir:
                    failed_job=self.return_failed(dir)
                    if failed_job:
                        self.failed[dir]=failed_job
                        self.mother.create_file.for_failed(dir,failed_job)
                    if hasattr(self.MWparam,'refine') and self.MWparam.refine and dir[0]=='P':
#                        failed_list=[val[0] for val in failed_job]
                        suceed_list=[[val,-1] for val in self.MWparam.actif_param] # if val not in failed_list]
                        self.mother.create_file.for_refine(dir,suceed_list)

                #clean all failed
                #self.mother.control.clean(self.failed)
            print 'done'
                   
        #4 ##############################################################################################
        def return_failed(self,dir):
            """read the failed process and return them in a list """

            out=[]
            try:
                for line in open('./SubProcesses/'+dir+'/'+self.MWparam.name+'/failed_job.dat'):
                    out.append([int(value) for value in line.split()])
            except IOError:
                return []

            return out

        #3 ##############################################################################################
        def launch(self):
            """ launch all the scheduling file """
            
            if self.MWparam.run_opt['launch'] or self.MWparam.run_opt['relaunch'] :
                #check the compilation
                for directory in self.MWparam.MW_listdir+self.MWparam.P_listdir:
                    os.chdir('./SubProcesses/'+directory)
                    os.system('make')
                    os.chdir('../../')
            
                if self.MWparam.run_opt['launch']:
                    self.mother.launch()
                elif self.MWparam.run_opt['relaunch']:
                    self.mother.launch.relaunch(self.failed)

        #3 ##############################################################################################
        def launch_level(self, block, level):
            """ for multiple level job (refine for example) """

            if not(self.MWparam.run_opt['launch'] or self.MWparam.run_opt['relaunch'] or self.MWparam.run_opt['control'] ):
                return
            # remove progressbar +activate automatic launch
            self.mother.submission_file.activate_automatic_launch(self.mother)
            self.mother.create_file.pbar=0
            
            # create file saying that we change of level
            ff=open('./SubProcesses/'+self.MWparam.P_listdir[0]+'/'+self.MWparam.name+'/card_'+str(block)+'/finish_level','w')
            ff.writelines(str(level))
            ff.close()

            # remove old 'done' file
            for directory in self.MWparam.P_listdir:
                os.system('rm -f ./SubProcesses/'+directory+'/'+self.MWparam.name+'/card_'+str(block)+'/done')

            # create the files
            self.mother.create_file.for_level(block,level) 
        
        #3 ##############################################################################################
        def clean(self,mode='short'):
            """ supress old output """
            
            import clean

            if mode=='short' and self.MWparam['mw_run']['acceptance_run']:
                clean.Clean_event(self.MWparam.name,'control-only')
            else:
                clean.Clean_event(self.MWparam.name,mode)
                
            clean.Clean_weight(self.MWparam.name,mode)
            
        #3 ##############################################################################################
        def control(self):
            """ control all the process """

            if self.MWparam.run_opt['control'] or self.MWparam.run_opt['status']:
                if hasattr(self.MWparam,'refine') and self.MWparam.refine:
                    self.mother.control.special_refine(self.failed)

                elif hasattr(self,'failed'):
                    self.mother.control.all_list(self.failed)
                else:
                    self.mother.control()
                print 'all job done'



    #2 ##############################################################################################
    #2 CLASS:  CREATE_FILE
    #2 ##############################################################################################
    class create_file:
        """ all the possible routine to create submission file """

        #3 ##############################################################################################
        def __init__(self,mother,clear=0):
            self.mother=mother
            self.MWparam=mother.MWparam
            self.file_number={}
            self.pbar=0
            
            if clear:
                mother.submission_file.clear() #supress ALL old submission file
        
        #3 ##############################################################################################
        def __call__(self):
            
            self.all()

        #3 ##############################################################################################
        def all(self):
            """ creates all the submition routines: Default launches MW and ME routines """

            self.mother.submission_file.clear() #supress ALL old submission file

            nb_create_file =len(self.MWparam.P_listdir)*len(self.MWparam.actif_param)*self.MWparam.norm_with_cross
            nb_create_file+=len(self.MWparam.MW_listdir)*len(self.MWparam.actif_param)*(self.MWparam.nb_event-self.MWparam.startevent)
            if hasattr(self.MWparam,'n_join'):
                nb_create_file/=self.MWparam.n_join    
            self.pbar=progressbar.progbar('Creation/Submission',nb_create_file)

            if self.MWparam.norm_with_cross:
                self.all_MadEvent()
            if self.MWparam.nb_event:
                self.all_MadWeight()

                
        #3 ##############################################################################################
        def all_MadWeight(self,main=0):
            """ creates all the submission for MadWeight """

            nb_create_file=len(self.MWparam.MW_listdir)*len(self.MWparam.actif_param)*(self.MWparam.nb_event-self.MWparam.startevent)
            if hasattr(self.MWparam,'n_join'):
                nb_create_file/=self.MWparam.n_join    
            self.pbar=progressbar.progbar('MW Creation/Submission',nb_create_file)
            
            if main:
                self.mother.submission_file.clear() #supress ALL old submission file
            for directory in self.MWparam.MW_listdir:
                for i in self.MWparam.actif_param:
                    self.Card(directory,i,'M')
        #3 ##############################################################################################
        def all_MadEvent(self,main=0):
            """ creates all the submission for MadEvent """
            nb_create_file =len(self.MWparam.P_listdir)*len(self.MWparam.actif_param)*self.MWparam.norm_with_cross
            if hasattr(self.MWparam,'n_join'):
                nb_create_file/=self.MWparam.n_join    
            self.pbar=progressbar.progbar('ME Creation/Submission',nb_create_file)
            
            if main:
                self.mother.submission_file.clear() #supress ALL old submission file
            for directory in self.MWparam.P_listdir:
                    self.Event(directory,1,'P')

        #3 ##############################################################################################
        def all_dir(self,directory,dir_type='auto'):
            """ creates all the submission for the directory """
            """ dir: directory
                dir_type: must belongs to ['MW','P','auto']
            """

            if dir_type not in ['M','P']:
                dir_type=dir[0]

            if dir_type == 'M':
                for i in self.MWparam.actif_param:
                    self.Card(directory,i,'M')
            elif(dir_type == 'P'):
                    self.Event(directory,1,'P')
            else:
                raise DirectoryError, "directory must start with MW or P :"+directory

        #3 ##############################################################################################
        def Event(self,directory,nb_event,dir_type):
            """ creates the submission for all the card for a given event in a specific directory """
            
            if dir_type=='M':
            	n_job=len(self.actif_param)
            	self._packet([directory]*n_job,               #directory
            				cardnb=self.MWparam.actif_param, #first number
            				eventnb=[nb_event]*n_job,                #second number
            				dir_type=['M']*n_job)					 # type of job
            
            elif dir_type=='P':
                n_job=len(self.MWparam.actif_param)
            	self._packet([directory]*n_job,               #directory
            				cardnb=[num for num in self.MWparam.actif_param], #second number
            				dir_type=['P']*n_job)					 # type of job
                        
        #3 ##############################################################################################            
        def Card(self,directory,nb_card,dir_type):
            """ creates the submission for all the event for a given card in a specific directory """	
                            
            n_job=self.MWparam.nb_event	
            self._packet([directory]*n_job,               #directory
            			cardnb=[nb_card]*n_job,                 #first number  
            			eventnb=range(0,self.MWparam.nb_event),  #second number
            			dir_type=[dir_type]*n_job)			     # type of job
            
        #3 ##############################################################################################
        def _packet(self,directory,cardnb,eventnb=[],dir_type=[],prog=''):
            """ usual packet with control of input """
            
            #initialization
            if dir_type[0]=='P':
                n_join=1
            elif hasattr(self.MWparam,'n_join'):
                n_join=self.MWparam.n_join
            else:
                n_join=1
                
            if not cardnb:
                return
           
            if not dir_type:
                if eventnb:
                    dir_type=['M']*len(eventnb)
                else:
                    dir_type=['P']*len(cardnb)
                    eventnb=[-1]*len(cardnb)
                
            if type(cardnb)!=list:
                return self.one(directory,nbcard,nbevent,dir_type)

            #remove unactive card
            eventnb=[eventnb[pos] for pos in range(0,len(eventnb)) if self.MWparam.param_is_actif[cardnb[pos]]]
            dir_type=[dir_type[pos] for pos in range(0,len(dir_type)) if self.MWparam.param_is_actif[cardnb[pos]]]
            cardnb=[card for card in cardnb if self.MWparam.param_is_actif[card]]

            #check if the packet are not too big...
            for i in range(0,len(cardnb)//n_join+1):
                start=i*n_join
                stop=min((i+1)*n_join,len(cardnb))
                self.packet(directory[start:stop],cardnb[start:stop],eventnb[start:stop],dir_type[start:stop],prog)
                               
        #3 ##############################################################################################
        def packet(self,directory,cardnb,eventnb=[],dir_type=[],prog=''):
            """ create the submission file for a list of event/card/directory """

	    #default use the standard submission routine
            if eventnb:
                for i in range(0,len(cardnb)):
                    if prog:
                        self.one(directory[i],cardnb[i],eventnb[i],dir_type=dir_type[i],prog=prog)
                    else:
                        self.one(directory[i],cardnb[i],eventnb[i],dir_type=dir_type[i])
            else:
                for i in range(0,len(cardnb)):
                    if prog:
                        self.one(directory[i],cardnb[i],-1,dir_type=dir_type[i],prog=prog)
                    else:
                        self.one(directory[i],cardnb[i],-1,dir_type=dir_type[i])

        #3 ##############################################################################################
        def for_level(self,block,level):
            """ launch multiple level program """

            if self.MWparam['mw_run']['acceptance_run'] and level%2==1:
                self.one(self.MWparam.P_listdir[0],block,nbevent=-1,dir_type='P',prog='acceptance.py')
                return
            for directory in self.MWparam.P_listdir:
                if self.MWparam['mw_run']['acceptance_run']:
                    prog='madevent.py' #no stop if precision reached
                    agument=''
                else:
                    prog='refine.py'
                    argument=self.MWparam['mw_run']['accuracy_cross']
                
                self.one(directory,block,nbevent=-1,dir_type='P',prog=prog,argument=argument)
                        
        #3 ##############################################################################################
        def one(self,directory,nbcard,nbevent=-1,dir_type='P',prog='madevent.py',opt='',argument=''):
            """ create the submission file for one event/card/directory """
            
            raise Clustererror , 'no way to schedule only one job in this cluster'
            #template
            #text=...
            #self.mother.submission_file.write(directory,file)  #create the submission file            
            
        #3 ##############################################################################################
        def for_failed(self,directory,failed):
            """ create the submission file for the failed job"""

            if not failed:
                return
            nb_create_file=len(failed)
            if hasattr(self.MWparam,'n_join'):
                nb_create_file/=self.MWparam.n_join    
            self.pbar=progressbar.progbar(str(directory)+' Creation/Submission',nb_create_file)
            
            nbcard=[input[0] for input in failed]
            nbevent=[input[1] for input in failed]
            self.packet([directory]*len(nbcard),nbcard,nbevent,[directory[0]]*len(nbcard))
            self.pbar.finish()


        #3 ##############################################################################################
        def for_refine(self,directory,listtorefine):
            """ create the submission file for the refine job"""

            if not listtorefine:
                return

            #put a progress bar
            nb_create_file=len(listtorefine)
            if hasattr(self.MWparam,'n_join'):
                nb_create_file/=self.MWparam.n_join    
            self.pbar=progressbar.progbar(str(directory)+' Creation/Submission',nb_create_file)

            nbcard=[input[0] for input in listtorefine]
            nbevent=[input[1] for input in listtorefine]

            self.packet([directory]*len(nbcard),nbcard,nbevent,[directory[0]]*len(nbcard),prog='madevent.py')     
            
    #2 ##############################################################################################
    #2 CLASS LAUNCH
    #2 ##############################################################################################
    class launch:
        """ all the routine to launch all the routine """

        #3 ##############################################################################################
        def __init__(self,mother):
            """store/link the usefull object """
            
            self.mother=mother
            self.MWparam=mother.MWparam
            self.pbar=0 #no progressbar
            
        #3 ##############################################################################################            
        def __call__(self):
            """ launch everything """
            self.all()
            
        #3 ##############################################################################################
        def all(self):
            """ control all the submition routines: Default launches MW and ME routines """
            
            if self.mother.submission_file.nbfile:
                self.pbar=progressbar.progbar('Submission',self.mother.submission_file.nbfile)
                
            self.all_MadWeight()
            if self.MWparam.norm_with_cross:
                self.all_MadEvent()
            if self.pbar:
                self.pbar.finish()
                
        #3 ##############################################################################################
        def all_MadWeight(self):
            """ creates all the submission for MadWeight """

            for directory in self.MWparam.MW_listdir:
                self.directory(directory,'M')
                
        #3 ##############################################################################################            
        def all_MadEvent(self):
            """ creates all the submission for MadEvent """

            for directory in self.MWparam.P_listdir:
                self.directory(directory,'P')

        #3 ##############################################################################################
        def directory(self,directory,dir_type='auto'):
            """ creates all the submission for the directory """
            """ dir: directory
                dir_type: must belongs to ['MW','P','auto']
            """

            # use the submision file if they are at least one submision file
            if self.mother.submission_file.nbfile:
                self.launch_submission_file(directory)
                return
            
            # continue -pass to each directory- if no submision file defined!
            
            if dir_type=='auto':
                dir_type=directory[0]
                
            if dir_type=='P':
                list_nbcard=[num for num in self.MWparam.actif_param]
                list_nbevent=[-1] #for compatibility
            elif dir_type=='M':
                list_nbcard=self.MWparam.actif_param
                list_nbevent=range(0,self.MWparam.nb_event)
            else:
                raise DirectoryError, "directory must start with MW or P : "+directory

            for event in list_nbevent:
                for card in list_nbcard:
                    self.one(directory,card,event,dir_type)

        #3 ##############################################################################################
        def one(self,directory,nbcard,nbevents=-1,dir_type='auto'):
            """ launch all the submission for the directory """
            """ dir: directory
                dir_type: must belongs to ['M','P','auto']
                nbcard: card number _
                nbevents: event number in P_ (-1 if not defined)
            """
			#the use of submission_file class is not compatible with the use of this routine
            raise Clustererror , 'no way to launch a single job on this cluster'
        
        #3 ##############################################################################################
        def relaunch(self,failed={}):
            """ relaunch all the failed job for the directory """

            # use the submision file if they are at least one submision file
            if self.mother.submission_file.nbfile:
                self.pbar=progressbar.progbar('Submission',self.mother.submission_file.nbfile)                
                self.launch_submission_file()
                self.pbar.finish()
                return
            
            #else use the self.one method
            for directory in failed.keys():
                dir_type=directory[0]
                for nbcard,nbevent in failed[directory]:
                    self.one(directory,nbcard,nbevent,dir_type)                                                                                                           
        #3 ##############################################################################################
        def launch_submission_file(self,directorylist=''):
            """ launch all the submission file for all the directories """

            if not directorylist:
                directorylist=self.MWparam.MW_listdir+self.MWparam.P_listdir
            elif type(directorylist)!=list:
                directorylist=[directorylist]

            for directory in directorylist:
                os.chdir('./SubProcesses/'+directory+'/schedular/')
                for file in self.mother.submission_file.listing(directory):
                    self.launch_one_submission_file(file)
                os.chdir('../../../')

        #3 ##############################################################################################
        def launch_one_submission_file(self,file):
            """ launch one specific submission file """
            try:
                os.system(self.submit_word+' '+file+' &>'+file+'.log')
            except:
                sys.exit('you need to specify how to launch scedular file in your cluster:\
                variable submit_word')
                
            if self.pbar:
                #a progress bar is defined -> update this one
                self.pbar.update()                
                                                                                                                                                        
    #2 ##############################################################################################
    #2 CLASS CONTROL
    #2 ##############################################################################################
    class control:
        """control the run: by default look for output event in directory"""

        #3 ##############################################################################################
        def __init__(self,mother,clear=1):
            self.mother=mother
            self.MWparam=mother.MWparam
            
            self.max_iden_step=240  #number of step before automatic stop of the control part if no change occur
            self.iden_step=0        # number of step without any change in the status
            
            self.idle=0
            self.running=0
            self.to_launch=0
            self.store_as_finish=0
            self.block_level={}     #if we use level submission store at which level we are for each block
            self.init_block_level() # check the initial status of the level and store_as_finish/to_launch
            self.finish=0
            self.unkown=0

            self.to_launch=0        # number of job expected to be launched after some actual job are finished
            self.store_as_finish=0  # number of job of previous finished phases
            
            self.block_level={}     #if we use level submission store at which level we are for each block
            self.init_block_level() # check the initial status of the level and store_as_finish/to_launch

        #4 ##############################################################################################
        def init_block_level(self):
            """ check the status of level for each Part of the code and update linked globals """

            for directory in self.MWparam.MW_listdir:
                #only one level so we don't care and put it always to zero (it will pass at one after the first control)
                self.block_level[directory]=0
                
            if self.MWparam.norm_with_cross:
                for cardnb in self.MWparam.actif_param:
                    # check if a level file exist
                    if os.path.isfile('./SubProcesses/'+self.MWparam.P_listdir[0]+'/'+self.MWparam.name+'/card_'+str(cardnb)+'/finish_level'):
                        level=int(open('./SubProcesses/'+self.MWparam.P_listdir[0]+'/'+self.MWparam.name+'/card_'+str(cardnb)+'/finish_level').read())
                        self.block_level[cardnb]=level
                    else:
                        self.block_level[cardnb]=0
                    #init starting_point for to_launch/store_as_finish
                    self.to_launch+=( self.mother.driver.Pmax_level - 1 - self.block_level[cardnb] ) * len( self.MWparam.P_listdir )
                    self.store_as_finish += ( self.block_level[cardnb] ) * len( self.MWparam.P_listdir )

        #3 ##############################################################################################
        def __call__(self):
            print 'start cluster control'
            self.all()

        #3 ##############################################################################################
        def all(self):
            """ control all the submission routines: Default launches MW and ME routines """

            while 1:
                self.idle,self.running,self.finish,self.unkown = 0,0,0,0
                self.all_MadWeight(main=0)
                if self.MWparam.norm_with_cross:
                    self.all_MadEvent(main=0)
                print self
                if self.to_launch+self.idle+self.running==0 or self.control_iden_state():
                    break
                time.sleep(30)
                
        #3 ##############################################################################################
        def all_MadWeight(self,main=1):
            """ control all the submission for MadWeight """

            if main:
                while 1:
                    self.idle,self.running,self.finish,self.unkown = 0,0,0,0
                    all_MadWeight(main=0)
                    print self
                    if self.to_launch+self.idle+self.running==0 or self.control_iden_state():
                        break
                    time.sleep(30)
            else:
                for directory in self.MWparam.MW_listdir:
                    if self.block_level[directory]: #only one level
                        continue
                    value=self.block_control(directory,'M')
                    if value:
                        self.store_as_finish+=value
                        self.finish-=value
                        self.block_level[directory]+=1                            
        

        #3 ##############################################################################################            
        def all_MadEvent(self,main=1):
            """ control all the submission for MadEvent """

            if main:
                while 1:
                    self.idle,self.running,self.finish,self.unkown = 0,0,0,0
                    all_MadEvent(main=0)
                    print self
                    if self.to_launch+self.idle+self.running==0 or self.control_iden_state():
                        break
                    time.sleep(30)
            else:
                for card in self.MWparam.actif_param:
                    if self.block_level[card]==self.mother.driver.Pmax_level:
                        continue
                    else:
                        value=self.block_control(card,'P',self.block_level[card])
                        if value:
                            self.store_as_finish+=value
                            self.finish-=value
                            self.block_level[card]+=1
                            if self.block_level[card]<self.mother.driver.Pmax_level:
                                if self.MWparam['mw_run']['acceptance_run'] and self.block_level[card]%2==0:
                                    self.to_launch-=value * len(self.MWparam.P_listdir)
                                else:
                                    self.to_launch-=value
                                self.idle+=value
                                self.mother.driver.launch_level(card,self.block_level[card])

        #3 ##############################################################################################
        def all_list(self,listtocontrol,main=1):
            """ control all the submission for the list """

            if main==1:
                #self.clean(listtocontrol)
                while 1:
                    self.idle,self.running,self.finish,self.unkown = 0,0,0,0
                    self.all_list(listtocontrol,main=0)
                    print self
                    if self.idle+self.running==0 or self.control_iden_state():
                        break
                    time.sleep(30)
            else:
                for directory in listtocontrol.keys():
                    for nbcard,nbevent in listtocontrol[directory]:
                        self.one(directory,nbcard,nbevent,directory[0])


        #3 ##############################################################################################
        def special_refine(self,failed,main=1):
            """ control all the submission for the list """
              
            if main:
                while 1:
                    self.idle,self.running,self.finish,self.unkown = 0,0,0,0
                    self.special_refine(failed,main=0)
                    print self
                    if self.to_launch+self.idle+self.running==0 or self.control_iden_state():
                        break
                    time.sleep(30)
            else:
                self.all_MadEvent(main=0)
                self.all_list(failed,main=0)
                
        #3 ##############################################################################################
        def control_iden_state(self):
            """ control if nothing move for a long time  return 1 if we are in this state"""
            try:
                iden_status=(    self.old_idle   ==self.idle \
                             and self.old_running==self.running \
                             and self.old_finish ==self.finish)
            except: #first pass
                iden_status=0
                
            if iden_status:
                self.iden_step+=1
                if self.iden_step<self.max_iden_step:
                    return 0
                else:
                    print 'fix status found',self.max_iden_step,'without any modification on the cluster status -> go out'
                    return 1 #fix state find
            else:
                self.iden_step=0
                #assign new value
                self.old_idle=self.idle
                self.old_running=self.running
                self.old_finish=self.finish
                return 0
                
        #3 ##############################################################################################
        def block_control(self,block,dir_type,level=0):
            """ Control all the submission in a given SubProcesses directory """

            all_suceed=0
            #A ###################################
            #A STEP A: DEFINE DEPENDENT VARIABLE
            #A ###################################
            
            if dir_type == 'M':
                list_nbcard=self.MWparam.actif_param
                list_nbevent=range(self.MWparam.startevent,self.MWparam.nb_event)
                list_dir=[block]
            elif(dir_type == 'P'):
                list_nbevent=[-1]
                list_nbcard=[block]
                if self.MWparam['mw_run']['acceptance_run'] and level%2==1:
                    list_dir=[self.MWparam.P_listdir[0]]
                else:
                    list_dir = self.MWparam.P_listdir
                list_nbcard=[block]
            else:
                raise DirectoryError, "directory must start with MW or P"

            #B ###################################
            #B STEP B: CHECK STATUS FOR EACH DIR
            #B ###################################
            name=self.MWparam.name
            all_suceed=1
            for nbcard in list_nbcard:
                for nbevent in list_nbevent:
                    for directory in list_dir:                    
                        val=self.one(directory,nbcard,nbevent,dir_type)
                        if all_suceed:
                            all_suceed=2*val/(val+1.0)*(all_suceed+1) #return 0 if val =0, add one otherwise
                        
            if all_suceed:
                return all_suceed-1 #the total number of finish job otherwise because it start at one
            else:
                return 0

        #3 ##############################################################################################
        def one(self,directory,nbcard,nbevent=-1,dir_type='auto'):
            """ control the status for a single job """

            #A ###################################
            #A STEP A: DEFINE DEPENDENT VARIABLE
            #A ###################################

            if dir_type not in ['M','P']:
                dir_type=directory[0]

            if dir_type == 'M':
                coherent_file='verif.lhco'
                start_file='start'
                output_file='stop'
                done='done'
            elif(dir_type == 'P'):
                coherent_file='param.dat'
                start_file='start'
                output_file='stop'
                done='done'
            else:
                raise DirectoryError, "directory must start with MW or P"

            #B ###################################
            #B STEP B: CHECK STATUS FOR EACH DIR
            #B ###################################
            name=self.MWparam.name
            if nbevent==-1:
                pos='./SubProcesses/'+directory+'/'+name+'/card_'+str(nbcard)+'/'
            else:
                pos='./SubProcesses/'+directory+'/'+name+'/card_'+str(nbcard)+'/event_'+str(nbevent)+'/'
            file1=pos+coherent_file
            file2=pos+start_file
            file3=pos+output_file
            file4=pos+done
            #UPDATE THE GLOBAL VALUE
            if os.path.isfile(file4):
                self.finish+=1
                return 1
            elif os.path.isfile(file3):
                self.running+=1
                os.system('rm -f '+file3+' &>/dev/null ;rm -f '+file2+' &>/dev/null')
                os.system('touch '+file4)
                return 0
            elif os.path.isfile(file2):
                self.running+=1
                return 0
            elif os.path.isfile(file1):
                self.idle+=1
                return 0
            else:
                self.unkown+=1
#                print file1,file2,file3
                print 'unknow position',pos,'from',os.getcwd()
                return 1

        #3 ##############################################################################################
        def clean(self,failed):
            """ supress start file/output file for failed job """

            for directory,nb in failed.items():
                for nbcard,nbevent in nb:
                    if nbevent==-1:
                        pos='./SubProcesses/'+directory+'/'+self.MWparam.name+'/card_'+str(nbcard)+'/'
                    else:
                        pos='./SubProcesses/'+directory+'/'+self.MWparam.name+'/card_'+str(nbcard)+'/event_'+str(nbevent)+'/'

                    try:
                        os.remove(pos+'start')
                        os.remove(pos+'stop')
                        os.remove(pos+'done')
                        os.remove(pos+'weights.out')
                    except:
                        pass
            
        #3 ##############################################################################################
        def __str__(self):
            """ return string """
            if self.unkown:
                return str(int(self.to_launch+self.idle))+'  '+str(self.running)+'  '+str(self.store_as_finish+self.finish)+'  not found: '+str(self.unkown)
            else:
                return str(int(self.to_launch+self.idle))+'  '+str(self.running)+'  '+str(int(self.store_as_finish+self.finish))
#                return str(int(self.to_launch))+' '+str(int(self.to_launch+self.idle))+'  '+str(self.running)+'  '+str(int(self.store_as_finish+self.finish))


##########################################################################
##                           SUBMISSION FILE                             ##
##########################################################################

#1 #######################################################################
class submission_file:
    """ class containing all the information to create/find/... all the
        submission file """

    #2 #######################################################################
    def __init__(self,MWparam,mother):
        """ create schedullinhg for each directory """

        self.MWparam=MWparam
        self.mother=mother
        self.object={}
        self.nbfile=0
        for directory in MWparam.MW_listdir+MWparam.P_listdir:
            self.object[directory]=submission_file_directory(directory)
            self.nbfile+=self.object[directory].number


    #2 #######################################################################
    def activate_automatic_launch(self,mother):
        """ activate automatic launch on the cluster of the submission file """
        self.launch=lambda file: mother.launch.launch_one_submission_file(file)
        

    #2 #######################################################################
    def clear(self):
        """ remove all submission file """
        [submission.clear() for submission in self.object.values()]
        self.nbfile=0
        
    #2 #######################################################################
    def write(self,directory,text):
        """ write a new submission file"""

        file=self.object[directory].write(text)
        self.nbfile+=1

        #check if we have to launch it directly
        if hasattr(self,'launch'):
            self.launch(file) #lambda function define in activate automatic launch

        #update the progessbar if exist
        if self.mother.create_file.pbar:
            self.mother.create_file.pbar.update()
        
    #2 #######################################################################
    def listing(self,directory):
        """ return the listing of all submission file in directory """
        
        return self.object[directory].listing()


#1 #######################################################################
class submission_file_directory:
    """ special routine for submission file in a specific directory"""

    #2 #######################################################################
    def __init__(self,directory):
        """update the status of the submission file"""
        
        self.directory='./SubProcesses/'+directory+'/schedular'
        try:
            os.mkdir(self.directory)
        except:
            pass
        self.number=self.find()

    #2 #######################################################################
    def clear(self):
        """ supress all the submission file """

        for file in os.listdir(self.directory):
            if 'log' in file or 'submission_file_' in file:
                os.remove(self.directory+'/'+file)
    
        self.number=0

    #2 #######################################################################
    def write(self,text):
        """ write a new submission file """

        name=self.directory+'/submission_file_'+str(self.number)+'.txt'
        ff=open(name,'w')
        ff.write(text)
        ff.close()
#        os.system('condor_submit  '+self.directory+'/submission_file_'+str(self.number)+'.txt')
        self.number+=1

        return name

    #2 #######################################################################        
    def find(self):
        """ find the number of existing submission file """

        return max([0]+[int(listdir[16:-4]) for listdir in os.listdir(self.directory) if (listdir[0:16]=='submission_file_' and listdir[-4:]=='.txt')])

    #2 #######################################################################
    def listing(self):
        """ find all the submission file """

        if os.path.basename(os.path.realpath('.'))=="schedular":
            directory='./'
        else:
            directory=self.directory
        return [listdir for listdir in os.listdir(directory) if listdir[0:16]=='submission_file_']


            

            
#####################################################################################################
##                                GESTION DERIVATIVE CLUSTER  CLASS                                ##
#####################################################################################################

import cluster


def def_cluster(MWparam):
    """ create the instance of an object in the class where:
        - the class derivates from class "cluster" (at first level)
        - the class.tag is MWparam.cluster
        - the input parameter are 'input'
    """
    mother_class="cluster"

    for object in dir(cluster):
        if hasattr(getattr(cluster,object),'__bases__') \
          and  mother_class in [obj.__name__ for obj in getattr(cluster,object).__bases__] \
          and hasattr(getattr(cluster,object),'tag')\
          and getattr(cluster,object).tag==MWparam.cluster:
                    return getattr(cluster,object)(MWparam)
    raise Clustererror, 'No cluster implemented for tag '+str(MWparam.cluster)
#           hasattr(getattr(cluster,object).__bases__[0],'__name__') and 



