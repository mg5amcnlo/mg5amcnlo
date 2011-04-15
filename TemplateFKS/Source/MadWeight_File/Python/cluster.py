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
##                       *** CLUSTER ***                                ##
##      + single_machine (tag=0)                                        ##
##      + condor         (tag=1)	             			##
##      + SGE            (tag=2)		                        ##
##        + SGE2 -for run in packet- (tag=21)                           ##
##      + BSUB           (tag=3)                                        ##
##      + multicore      (tag=4)                                        ##
##########################################################################
##
## BEGIN INCLUDE
##
import os
import time
from cluster_lib import *
##
## TAG
##
__author__="Mattelaer Olivier, the MadTeam"
__version__="1.1.2"
__date__="June 2009"
__copyright__= "Copyright (c) 2008-2009 MadGraph/MadEvent team"
__license__= "GNU"
##
## BEGIN CODE
##
#########################################################################
#   SINGLE MACHINE														#
#########################################################################
class single_machine(cluster):
    """ all the routine linked to the gestion of a single machine job """
    tag=0

    #2 ##################################################################
    class driver(cluster.driver):
        #3 ##################################################################    
        def all(self):
            self.clean()
            self.compile_all()
            self.create_all_dir()
            self.schedule()
            self.launch()
            for card in self.MWparam.actif_param:
                for i in range(1,self.Pmax_level):
                    self.launch_level(card,i)

    #2 ##################################################################
    class create_file(cluster.create_file):
        #3 ##################################################################    
        def one(self,directory,nbcard,nbevent=-1,dir_type='P',prog='madevent.py',argument=''):
            pass #no file creation needed
            
    #2 ##################################################################        
    class launch(cluster.launch):
        #3 ##################################################################
        def one(self,directory,nbcard,nbevent=-1,dir_type='P',prog=''):
            name=self.MWparam.name

            if dir_type=='M':
                if not prog: prog='../../../comp_madweight'
                pos='./SubProcesses/'+directory+'/'+name+'/card_'+str(nbcard)+'/event_'+str(nbevent)
                dirup=5
            elif dir_type=='P':
                if not prog: prog='../../madevent.py'
                pos='./SubProcesses/'+directory+'/'+name+'/card_'+str(nbcard)
                dirup=4
            else:
                print 'wrong dir_type ',dir_type

            os.chdir(pos)
            if self.MWparam['mw_run']['write_log']:
                os.system(prog)
            else:
                print 'launch '+prog
                os.system(prog+' &>/dev/null')
            os.chdir('../'*dirup)

            
            
#########################################################################
#   CONDOR CLUSTER														#
#########################################################################
class condor(cluster):
    """specific routine for condor cluster """ 
	
    tag=1 # value for use condor in the MadWeight_card.dat

    #2 ##################################################################		
    class create_file(cluster.create_file):
        """ create the condor submission file """
	
        #3 ##################################################################		
        def condor_submission(self,directory,nbsubmit,nbcard=-1,dir_type='P'):
            """ common part of all condor submission file """
            """ no standard for this functions """
			
            pos=os.path.abspath('./SubProcesses/'+directory) #to check
            dir_name=self.MWparam.name
            if dir_type=='P':
                prog='madevent.py'
                posdir=pos+'/'+dir_name+'/card_'
            elif dir_type=='M':
                prog='comp_madweight'
                posdir=pos+'/'+dir_name+'/card_'+str(nbcard)+'/event_'
                
            text=self.condor_submission_file(pos,prog,posdir+'$(PROCESS)',nbsubmit)
            self.mother.submission_file.write(directory,text)

        #3 ##################################################################		
        def condor_submission_file(self,pos,prog,posdir,nbsubmit=1,argument=''):
            """ common part of all condor submission file """
            """ no standard for this functions """
            
            text= 'Executable   = '+pos+'/'+prog+'\n'
            if self.MWparam['mw_run']['write_log']:
                text+='output       = '+posdir+'/out\n'
                text+='error        = '+posdir+'/err\n'
                text+='log          = '+posdir+'/log\n'
            else:
                text+='output       = /dev/null\n'
                text+='error        = /dev/null\n'
                text+='log          = /dev/null\n'
            if argument:
                text+='Arguments = '+str(argument)+' \n'
            text+='Universe     = vanilla\n'
            text+='notification = Error\n'
            text+='Initialdir='+posdir+'\n'

            condor_req=eval(self.MWparam.condor_req) #supress the ''
    	    if(condor_req!='' and type(condor_req)==str):
                text+='requirements ='+condor_req+'\n'         
            text+='Queue '+str(nbsubmit)+'\n'     
   
            return text
	
        #3 ##################################################################	
        def Card(self,directory,nb_card,dir_type):
            """ creates the submission for all the event for a given card in a specific directory """
			
            if dir_type=='M':
                self.condor_submission(directory,self.MWparam.nb_event,nb_card,dir_type)
            else:
                raise ClusterError, 'launch by card is foreseen only for MadWeight Job'
				
        #3 ##################################################################				
        #def Event(self,directory,nb_event,dir_type):
        #    """ creates the submission for all the card for a given event in a specific directory """
        # supress routine in order to be able to include the desactivation of param_card
        # use default and then self.one to submit
        #
        #if dir_type=='P':
        #        self.condor_submission(directory,nb_event,self.MWparam.nb_card,dir_type)
        #    else:
        #        raise ClusterError, 'launch by card is foreseen only for MadEvent Job'
				
        #3 ##################################################################				
        def one(self,directory,nbcard,nbevent=-1,dir_type='P',prog='madevent.py',argument=''):
            """ """
                                           			
            pos=os.path.abspath('./SubProcesses/'+directory) #to check
            dir_name=self.MWparam.name
            if dir_type=='P':
                posdir=pos+'/'+dir_name+'/card_'+str(nbcard)+'/'
            elif dir_type=='M':
                prog='comp_madweight'
                posdir=pos+'/'+dir_name+'/card_'+str(nbcard)+'/event_'+str(nbevent)

            text=self.condor_submission_file(pos,prog,posdir,argument=argument)
            self.mother.submission_file.write(directory,text)
			
    #2 ##################################################################			
    class launch(cluster.launch):
        """ launch all the job condor 
            use the submission file class
        """
        submit_word='condor_submit' #submit a file 'test' by "condor_submit test"


    #2 ##############################################################################################
    #2 CLASS CONTROL for multiple run
    #2 ##############################################################################################
    #class control(cluster.control):
    #    """control the run: by default look for output event in directory"""
    #                    
    #    def all(self):
    #        """ control all the submission routines: Default launches MW and ME routines """
    #        
    #        while 1:
    #            self.idle,self.running,self.finish = 0,0,0
    #            self.all_MadWeight(main=0)
    #            if self.MWparam.norm_with_cross:
    #                self.all_MadEvent(main=0)
    #            print self
    #            if self.idle<300:
    #                print 'stop waiting for this run: go to next run'
    #                break
    #            time.sleep(30)


#########################################################################
#   Condor CLUSTER  -submission by bigger packet-					  		#
#########################################################################
class condor2(cluster,condor):

    tag=11 # value to call this class from MadWeight_card.dat

    #2 ##################################################################		
    class create_file(cluster.create_file):
        """ create the condor submission file """
        
        def all_MadWeight(self,main=0):
            """ creates all the submission for MadWeight """
            if main:
                self.mother.submission_file.clear() #supress ALL old submission file
            for directory in self.MWparam.MW_listdir:
                for i in range(0,self.MWparam.nb_event):
                    self.Event(directory,i,'M')                                                                        
        
        #3 ##################################################################				
        def Event(self,directory,nb_event,dir_type):
            """ creates the submission for all the card for a given event in a specific directory """
            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name

            self._packet([directory]*len(self.MWparam.actif_param),
                        self.MWparam.actif_param,
                        [nb_event]*len(self.MWparam.actif_param),
                        [dir_type]*len(self.MWparam.actif_param))
                #_packet is the usual packet but checking first the coherence of the input (and split in smaller packet)
                
        #3 ##############################################################################################
        def packet(self,directory,nbcard,nbevent,dir_type,prog='madevent.py'):
            """ create the submission file for a list of event/card/directory """

            #if start_text:
            #    self.packet_text=''
            #    self.packet_nb=0
                
            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name

            #write the script launching the packet
            text_script=''
            posdir={}
            for i in range(0,len(dir_type)):
                if dir_type[i]=='P':
                    posdir[i]=pos+"/SubProcesses/"+directory[i]+'/'+name+'/card_'+str(nbcard[i])
                    text_script+='cd '+posdir[i]+'/\n'                    
                    text_script+="../../"+prog+"\n"
                else:
                    posdir[i]=pos+"/SubProcesses/"+directory[i]+'/'+name+'/card_'+str(nbcard[i])+'/event_'+str(nbevent[i])
                    text_script+='cd '+posdir[i]+'/\n'                    
                    text_script+="../../../madweight\n"
            ff=open(pos+'/SubProcesses/'+directory[0]+'/schedular/condor_script_'+str(nbcard[0])+'_'+str(nbevent[0]),'w')
            ff.writelines(text_script)
            ff.close()
            os.system('chmod 771 '+pos+'/SubProcesses/'+directory[0]+'/schedular/condor_script_'+str(nbcard[0])+'_'+str(nbevent[0]))

            #write the condor file
            pos=os.path.realpath(os.getcwd())+"/SubProcesses/"+'/'+directory[0]
            prog='schedular/condor_script_'+str(nbcard[0])+'_'+str(nbevent[0])
            text=self.condor_submission_text(pos,prog,posdir[0])
            self.mother.submission_file.write(directory[0],text)

            
    class launch(condor.launch): pass        

#########################################################################
#   SGE CONDOR  -submission by bigger packet-                           #
#########################################################################
class condor3(condor2,cluster):

    tag=12 # value to call this class from MadWeight_card.dat

    #2 ##################################################################        
    class create_file(cluster.create_file):
        """ create the condor submission file """                                                            
        
        #3 ##################################################################                
        def Card(self,directory,nb_card,dir_type='M'):
            """ creates the submission for all the event for a given card in a specific directory """
            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name            
            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name

            self._packet([directory]*len(self.MWparam.actif_param),
                        self.MWparam.actif_param,
                        [nb_event]*len(self.MWparam.actif_param),
                        [dir_type]*len(self.MWparam.actif_param))
                    
        all_MadWeight = condor2.create_file.all_MadWeight
        packet = condor2.create_file.packet
    
    class launch(condor2.launch): pass








#########################################################################
#   SGE CLUSTER  					  		#
#########################################################################
class SGE(cluster):

    tag=2 # value to call this class from MadWeight_card.dat

    #2 ##################################################################		
    class create_file(cluster.create_file):
        """ create the condor submission file """

        standard_text=open('./Source/MadWeight_File/Tools/sge_schedular','r').read()
        #3 ##################################################################	
        #def Card(self,directory,nb_card,dir_type):
        #    """ creates the submission for all the event for a given card in a specific directory """
        #    pass	
        #3 ##############################################################################################

       #3 ##################################################################				
        def one(self,directory,nbcard,nbevent=-1,dir_type='P',prog="madevent.py",argument=''):
            """SGE in pure local for the moment  """
            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name
            
            if self.MWparam.param_is_actif[nbcard]:
                return

            if dir_type=='P':
                posdir=+pos+"/SubProcesses/"+directory+'/'+name+'/card_'+str(nbcard)+'/'
                pos_prog=="../../"+prog+"\n"
            elif dir_type=='M':
                posdir=+pos+"/SubProcesses/"+directory+'/'+name+'/card_'+str(nbcard)+'/event_'+str(nbevent)+'/'
                pos_prog="../../../comp_madweight\n"

            
            text=self.standard_text
            text+="#$ -wd "+posdir+'/\n'
            if self.MWparam['mw_run']['write_log']:
                text+="#$ -e "+posdir+'/err\n'
                text+="#$ -o "+posdir+'/out\n'
            else:
                text+="#$ -e /dev/null\n"
                text+="#$ -o /dev/null\n"
            text+="\n"
            text+="date\n"
            if argument:
                text+=pos_prog[:-1]+' '+str(argument)+'\n' # the [:-1] is for remove the \n caracter
            else:
                text+=pos_prog
            text+="date\n"

            self.mother.submission_file.write(directory,text)
            
    #2 ##################################################################			
    class launch(cluster.launch):
        """ launch all the SGE job  
            use the submission file class
        """
        submit_word='qsub' #submit a file 'test' by "qsub test"

#########################################################################
#   SGE CLUSTER  -submission by bigger packet-					  		#
#########################################################################
class SGE2(cluster,SGE):

    tag=21 # value to call this class from MadWeight_card.dat

    #2 ##################################################################		
    class create_file(cluster.create_file):
        """ create the condor submission file """

        standard_text=open('./Source/MadWeight_File/Tools/sge_schedular','r').read()
        #3 ##################################################################	
        #def Card(self,directory,nb_card,dir_type):
        #    """ creates the submission for all the event for a given card in a specific directory """
        #    pass	
        #3 ##############################################################################################
        
        def all_MadWeight(self,main=0):
            """ creates all the submission for MadWeight """
            if main:
                self.mother.submission_file.clear() #supress ALL old submission file
            for directory in self.MWparam.MW_listdir:
                for i in range(0,self.MWparam.nb_event):
                    self.Event(directory,i,'M')                                                                        
        
        #3 ##################################################################				
        def Event(self,directory,nb_event,dir_type='M'):
            """ creates the submission for all the card for a given event in a specific directory """
            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name
            
            text=self.standard_text
            posdir=pos+"/SubProcesses/"+directory+'/'+name+'/card_'+str(self.MWparam.actif_param[0])+'/event_'+str(nb_event)+'/'
            text+="#$ -wd "+posdir+'/\n'
            text+="#$ -e "+posdir+'/err\n'
            text+="#$ -o "+posdir+'/out\n'
            text+="\n"
            text+="date\n"
            
            if dir_type=='P':
                return cluster.create_file.Event(self,directory,nb_event,dir_type)
            elif dir_type=='M':
                text+="cd ../../..\n"
                text+="make\n"
                for i in range(1,self.MWparam.actif_param):
                    text+="cd "+name+"/card_"+str(i)+"/event_"+str(nb_event)+'\n'
                    text+="../../../madweight\n"
                    text+="cd ../../..\n"
            text+="date\n"
        
            self.mother.submission_file.write(directory,text)
        
        #3 ##############################################################################################
        def packet(self,directory,nbcard,nbevent,dir_type='P',prog="madevent.py"):
            """ create the submission file for a list of event/card/directory """

            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name
            
            text=self.standard_text
            posdir=pos+"/SubProcesses/"+directory[0]+'/'+name+'/card_'+str(self.MWparam.actif_param[0])+'/event_'+str(nb_event[0])+'/'
            text+="#$ -wd "+pos+"/SubProcesses/"+directory[0]+'/\n'
            text+="#$ -e "+posdir+'/err\n'
            text+="#$ -o "+posdir+'/out\n'
            text+="\n"
            text+="date\n"
            
            for i in range(0,len(nbcard)):
                text+="cd ../"+directory[i]+'\n'
                text+="cd "+name+"/card_"+str(nbcard[i])+"/event_"+str(nb_event[i])+'\n'
                if dir_type[i]=='P':
                    text+="../../"+prog+"\n"
                    text+='cd ../../\n'
                else:
                    text+="../../../madweight\n"
                    text+='cd ../../../\n'
            text+="date\n"
            
            self.mother.submission_file.write(directory[0],text)


        
    class launch(SGE.launch): pass        

	
#########################################################################
#   BSUB CLUSTER 					  		#
#########################################################################
class Bsub(cluster):

    tag=3 # value to call this class from MadWeight_card.dat

    #2 ##################################################################		
    class create_file(SGE.create_file):
        """ create the condor submission file """

        standard_text=open('./Source/MadWeight_File/Tools/bsub_schedular','r').read()

       #3 ##################################################################				
        def one(self,directory,nbcard,nbevent=-1,dir_type='P',prog="madevent.py",argument=''):
            """BSUB in pure local for the moment  """
            pos=os.path.realpath(os.getcwd())
            name=self.MWparam.name

            if  not  self.MWparam.param_is_actif[nbcard]:
                return

            if dir_type=='P':
                posdir=pos+"/SubProcesses/"+directory+"/"+name+'/card_'+str(nbcard)+'/'
                pos_prog="../../"+prog+"\n"
            elif dir_type=='M':
                posdir=pos+"/SubProcesses/"+directory+"/"+name+'/card_'+str(nbcard)+'/event_'+str(nbevent)
                pos_prog="../../../comp_madweight\n"
                
            text=self.standard_text
            if self.MWparam["mw_run"]['write_log']:
                text+="#BSUB -e "+posdir+'/err\n'
                text+="#BSUB -o "+posdir+'/out\n'
            else:
                text+="#BSUB -e /dev/null\n"
                text+="#BSUB -o /dev/null\n"                
            text+="cd "+posdir+'/\n'
            text+="\n"
            text+="date\n"
            if argument:
                text+=pos_prog[:-1]+' '+str(argument)+'\n'
            else:
                text+=pos_prog
            text+="date\n"

            self.mother.submission_file.write(directory,text)
    #2 ##################################################################			
    class launch(cluster.launch):
        """ launch all the SGE job  
            use the submission file class
        """
        submit_word='bsub <' #submit a bash file 'test' by "bsub < test" (the < is important in order to define some run variable)

#1 ##########################################################################################################
# TS -  MULTIPROCESSEUR MODE
#1 ##########################################################################################################
class ts(cluster):
    """ all the routine linked to the gestion of a single machine job with multi-processor with ts install"""
    tag=4

    #2 ##################################################################
    class driver(cluster.driver):
        #3 ##################################################################    
        def all(self):
            self.clean()
            self.compile_all()
            self.create_all_dir()
            self.schedule()
            self.launch()
            self.control()
            
    #2 ##################################################################
    class create_file(cluster.create_file):
        #3 ##################################################################                                                                                  
        def one(self,directory,nbcard,nbevent=-1,dir_type='P',prog='madevent.py',argument=''):
            pass #no file creation needed                                                                                                                      

    #2 ##################################################################                                                                                      
    class launch(cluster.launch):
        #3 ##################################################################                                                                                  
        def one(self,directory,nbcard,nbevent=-1,dir_type='P'):
            name=self.MWparam.name

            if dir_type=='M':
                prog='../../../comp_madweight'
                pos='./SubProcesses/'+directory+'/'+name+'/card_'+str(nbcard)+'/event_'+str(nbevent)
                dirup=5
            elif dir_type=='P':
                prog='../../madevent.py'
                pos='./SubProcesses/'+directory+'/'+name+'/card_'+str(nbcard)
                dirup=4
            else:
                print 'wrong dir_type ',dir_type

            print "launching w/ TS"
            os.chdir(pos)
            os.system('ts '+prog)
            os.chdir('../'*dirup)
