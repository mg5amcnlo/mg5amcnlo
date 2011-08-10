#!/usr/bin/env python

#Extension
import string
import sys
import os
import re
import time
import stat
import math
# patch if symbolic directory replace by real file
sys.path.append('./Source/MadWeight_File/Python')
sys.path.append('../Source/MadWeight_File/Python')
#end patch
import write_MadWeight
import create_param
import Cards
from Cards import Card
import cluster_lib as Cluster
from create_run import update_cuts_status
from MW_param import MW_info,go_to_main_dir,check_for_help
from collect_result import Collect_dir,Collect_P_dir,Collect_MW_dir,Collect_All_dir,collect_schedular
from verif_event import verif_event,restrict_event_passing_cut
from madweight import make_symb_link,compile_P_SubProcesses

####################################################
##               PARAMETER
####################################################
rel_pos='../'


####################################################
##
####################################################
def create_acc_term(file1,file2):
    ff=open(file1,'r')
    gg=open(file2,'r')
    hh=open('./Events/accfac.dat','w')
    while 1:
        line1=ff.readline().split()
        line2=gg.readline().split()
        if line1 and line1[0]==line2[0]:
            value=float(line1[1])/float(line2[1])
            err=math.sqrt(float(line1[2])**2/float(line2[1])**2+float(line1[1])**2/float(line2[1])**2*float(line2[2])**2)
            hh.writelines(line1[0]+'\t'+str(value)+'\t'+str(err)+'\n')
        else:
            break
    ff.close()
    gg.close()
    hh.close()
    os.system('cp '+file1+' '+file1+'_with_cut.out')
    os.system('cp '+file2+' '+file1)

def passing_event(file1,file2, pass_event):
    print 'write file ',file2
    hh=open(file2,'w')
    for line in open(file1):
        event=int(line.split()[0].split('.')[1])
        if event in pass_event:
            hh.writelines(line)

####################################################
##               MAIN PROGRAM
####################################################
def Launch_all_SubProcess(MWparam):

    name=MWparam.name
    print 'run name :',name
    print 'original run :',MWparam['mw_acc']['old_run']
    MWparam.nb_event=0
    P_proclist=MWparam.P_listdir
    #create banner
    if MWparam.run_opt['launch']:
        os.system('mkdir ./Events/'+name)
        os.chmod('./bin/put_banner_MW',0775)
        os.system('./bin/put_banner_MW '+name+'/'+name+'_banner.txt')

    if MWparam.run_opt['compilation']:
        print 'starting program compilation'
        make_symb_link(P_proclist)    
        compile_P_SubProcesses(P_proclist)
            

    list_pass_event=restrict_event_passing_cut(MWparam)
    print list_pass_event

    if MWparam.run_opt['refine']:
        print "collecting data to find data with a precision less than",MWparam.run_opt['refine']
        collect_schedular(MWparam)          


    #all cluster operation
    cluster=Cluster.def_cluster(MWparam)
    cluster.driver()

    if MWparam.run_opt['collect']:
        print "collecting data"
        collect_schedular(MWparam)

    return list_pass_event

#####################################################################################"
##
##               LAUNCH PROGRAM
##
######################################################################################
if(__name__=='__main__'):

    go_to_main_dir()
    check_for_help(sys.argv)
    MWparam=MW_info('MadWeight_card.dat')
    MWparam.set_run_opt(sys.argv)
    MWparam.old_name=MWparam['mw_acc']['old_run']
    
    if MWparam.run_opt['param']:
        create_param.Param_card(run_name=MWparam)
        MWparam.update_nb_card()
        Cards.create_include_file(MWparam)
        update_cuts_status(MWparam)


    pass_event=Launch_all_SubProcess(MWparam)

    if MWparam.run_opt['plot']:
        passing_event('./Events/'+MWparam.old_name+'/'+MWparam.old_name+'_weights.out',\
                      './Events/'+MWparam.name+'/'+MWparam.name+'_weights.out'
                      , pass_event)

        create_acc_term('./Events/'+MWparam.name+'/'+MWparam.name+'_cross_weights.out',\
                       './Events/'+MWparam.old_name+'/'+MWparam.old_name+'_cross_weights.out')
        import plot
        plot.Likelihood(mw_param=MWparam,auto=1)
        # This routine can/must be added when you use the creation of the histogram
        # By the way she is not fully tested and stable yet
        #plot.Differential_Graph(MWparam,auto=1)        

    if MWparam.run_opt['clean']:
        from clean import Clean_run
        if MWparam.run_opt['clean']==1:
            Clean_run(MWparam.name)
        else:
            Clean_run(MWparam.run_opt['clean'])
