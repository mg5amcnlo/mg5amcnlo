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
##   last-modif:08/08*08/08                                             ##
##                                                                      ##
##########################################################################
##                                                                      ##
##   Content                                                            ##
##   -------                                                            ##
##                                                                      ##
##                                                                      ##
##                                                                      ##
##                                                                      ##
##########################################################################
#Extension
import string
import re
import os
import sys
import math
import time
import MW_param
import create_param
import verif_event 

#1 #######################################################################
class Acceptance:
    """ create events for each param_card and collect_data """

    #2 #######################################################################
    def __init__(self,name='acceptance'):
        """read the card"""
        
        MW_param.go_to_main_dir()
        self.MWparam=MW_param.MW_info('MadWeight_card.dat')
        self.name=name
        self.dir=os.getcwd().split('/')[-1]
        os.system('mkdir Events/'+self.MWparam.name)
        os.system('mkdir Events/'+self.MWparam.name+'/Acceptance')
        create_param.Param_card(run_name=self.MWparam)
        self.MWparam.update_nb_card()        
        os.chdir('..')

    #2 #######################################################################
    def make_all(self,end_list=[]):
        """schedullar"""

        self.dir_operation(end_list)
        self.control_operation(end_list)
        self.nacc=self.find_nacc()
        self.ngen=self.find_ngen()
        self.write_acc_file()

        
    #2 #######################################################################    
    def dir_operation(self,end_list):
        """ create all directory operation """

        for i in range(0,self.MWparam.nb_card):
            if i in end_list:
                continue
            if not os.path.isdir(self.name+'_'+str(i)+'_'+self.dir):
                print 'copying directory '+self.name+'_'+str(i)+'_'+self.dir
                os.system('cp -r '+self.dir+' '+self.name+'_'+str(i)+'_'+self.dir)

            #put correct card
            os.system('cp '+self.dir+'/Cards/param_card_'+str(i+1)+'.dat '+self.name+'_'+str(i)+'_'+self.dir+'/Cards/param_card.dat')        

            os.chdir(self.name+'_'+str(i)+'_'+self.dir)
            print os.getcwd()
            os.system('./bin/generate_events 1 acc_'+str(i)+' acc_'+str(i)+' &>../'+self.dir+'/Events/'+self.MWparam.name+'/Acceptance/log_acc_'+str(i)+'.log &')
            os.chdir('..')
            time.sleep(10)


##     #2 #######################################################################
##     def define_param_card(self,dir,num_card):
##         # define the param.dat
##         for MWdir in self.MWparam.P_listdir:
##             print dir+'/SubProcesses/'+MWdir+'/param.dat'
##             ff=open(dir+'/SubProcesses/'+MWdir+'/param.dat','w')
##             ff.writelines('param_card.dat\n')
##             ff.close()
            
    #2 #######################################################################    
    def control_operation(self,end_list=[]):
        """ check if file are written """

        while 1:
            if len(end_list)==self.MWparam.nb_card:
                break
            for i in range (0,self.MWparam.nb_card):
                if i in end_list:
                    continue
                try:
                    ff=open(self.name+'_'+str(i)+'_'+self.dir+'/Events/acc_'+str(i)+'_pgs_events.lhco.gz','r')
                    ff.close()
                    end_list.append(i)
                except:
                    continue
            print 'status: ',len(end_list),'/',self.MWparam.nb_card
            if len(end_list)!=self.MWparam.nb_card:
                time.sleep(30)
            
    #2 #######################################################################    
    def find_nacc(self):
        """select the event from directory"""

        os.chdir(self.dir)
        filter=verif_event.Lhco_filter(self.MWparam.P_listdir[0],MWparam=self.MWparam)
        os.chdir('..')
        list_nacc=[]

        for i in range(0,self.MWparam.nb_card):
            os.chdir(self.name+'_'+str(i)+'_'+self.dir+'/Events')
            try:
                os.system('gunzip acc_'+str(i)+'_pgs_events.lhco.gz')
                os.system('mv acc_'+str(i)+'_pgs_events.lhco input.lhco')
            except:
                print 'continue'
            os.chdir('..')
            list_nacc.append(filter.verif_event('input.lhco'))
            os.chdir('..')
        return list_nacc

    #2 #######################################################################    
    def find_ngen(self):
        """extract top_mass/cross_setion/n_gen from banner\
            input is the num_directory"""

        out=[]
        ##Pattern to look at
        Pattern=[]
        Pattern.append(re.compile(r'''\s*Number\s+of\s*Events\s+:\s*([\d.eE]*)\s*$''',re.VERBOSE))
        
        for num_dir in range(0,self.MWparam.nb_card):

            ff=open(self.name+'_'+str(num_dir)+'_'+self.dir+'/Events/acc_'+str(num_dir)+'_banner.txt','r')
            line='init'
            
            j=0
            while line!='':
                line=ff.readline()
                for i in range(0,len(Pattern)):
                    if(Pattern[i].search(line)):
                        print line
                        value=Pattern[i].search(line).groups()
                        out.append(int(value[0]))
                        j+=1
                if(j==len(Pattern)):
                    break

            ff.close()
            
        return out
    #2 #######################################################################    
    def write_acc_file(self):
        """write acceptance file """

        nacc=self.nacc
        ngen=self.ngen
        print 'nacc',nacc
        print 'ngen',ngen
        os.chdir(self.dir)
        ff=open('./Events/'+self.MWparam.name+'/accfac.dat','w')

        for i in range(0,self.MWparam.nb_card):
            line=str(i)+'\t'+str(float(nacc[i])/float(ngen[i]))+'\t'+str(math.sqrt(nacc[i])/float(ngen[i]))
            ff.writelines(line+'\n')
                        
#############################################################################
#############################################################################
##                               Main Program                              ##
#############################################################################
#############################################################################
if (__name__=='__main__'):

    acceptance=Acceptance()
    list=[] #make here a list of completly finished directory
    acceptance.make_all(list)

    #acceptance.control_operation(list)
    #acceptance.nacc=acceptance.find_nacc()
    #acceptance.ngen=acceptance.find_ngen()
    #acceptance.write_acc_file()
