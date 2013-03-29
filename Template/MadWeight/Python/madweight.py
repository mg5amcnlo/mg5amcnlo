#!/usr/bin/env python

# idea have 2 script
# madweight.py -> direct submission 
# madweight_cmd.py -> interface


#Extension
import string
import sys
import os
import re
import time
import stat
# patch if symbolic directory replace by real file
sys.path.append('./Source/MadWeight/Python')
sys.path.append('../Source/MadWeight/Python')
#end patch
import change_tf
import write_MadWeight,create_run,put_banner
import create_param
import Cards
from Cards import Card
import cluster as Cluster
from create_run import update_cuts_status
from MW_param import MW_info,go_to_main_dir,check_for_help,detect_SubProcess
from collect_result import collect_schedular
from verif_event import verif_event

####################################################
##               PARAMETER
####################################################
rel_pos='../'
####################################################
##               MAIN PROGRAM
####################################################
def Launch_all_SubProcess(MWparam):

    name=MWparam.name
    print 'name :',name
    P_proclist,MW_proclist=MWparam.P_listdir,MWparam.MW_listdir
    #create banner
    if MWparam.run_opt['launch']:
        os.system('mkdir ./Events/'+name+' &>/dev/null')
        banner=put_banner.MW_Banner('./Events/'+name+'/'+name+'_banner.txt')
        banner.write()

    if MWparam.run_opt['compilation']:
        print 'starting program compilation'
        compile_SubProcesses(MW_proclist)
            
    if MWparam.run_opt['event']:
        verif_event(MWparam)

    if MWparam.run_opt['refine']:
        print "collecting data to find data with a precision less than",MWparam.run_opt['refine']
        collect_schedular(MWparam)          


    #all cluster operation
    cluster=Cluster.def_cluster(MWparam)
    cluster.driver()

    if MWparam.run_opt['collect']:
        print "collecting data"
        collect_schedular(MWparam)          


#########################################################################
##                      FUNCTIONS
#########################################################################
def   compile_SubProcesses(process_list):
    os.chdir("./Source")
    os.system("make ../lib/libtools.a")
    os.system("make ../lib/libblocks.a")
    os.system("make ../lib/libTF.a")
    os.system("make ../lib/libpdf.a")
    os.system("make ../lib/libdhelas.a")
    os.system("make ../lib/libmodel.a")
    os.system("make ../lib/libgeneric.a")
    os.system("make ../lib/libcernlib.a")
    #os.system("make ../bin/sum_html")
    # os.system("make ../bin/gen_ximprove")

    os.chdir("../SubProcesses")
    for name in process_list:
        os.chdir("./"+name)
        #os.system(" rm madweight")
        exit_status=os.system("make > /dev/null")
        if  os.path.isfile("./comp_madweight") and exit_status==0 :
            os.chdir("..")
        else:
            print "fortran compilation error"
            sys.exit()
    os.chdir("..")   
    return

##############################################################################
def     compile_P_SubProcesses(process_list):
    if not os.path.isfile("./Cards/param_card.dat"):
        os.system('cp ./Cards/param_card_1.dat ./Cards/param_card.dat')
    os.chdir("./SubProcesses")
    for name in process_list:
        os.chdir("./"+name)
        #os.system(" rm madevent")
        os.system("make gensym > /dev/null")
        os.system("./gensym > /dev/null")
        exit_status=os.system("make > /dev/null")
        if  os.path.isfile("./madevent") and exit_status==0:
            os.chdir("..")
        else:
            print "fortran compilation error"
            sys.exit()
    os.chdir("..")   
    return  
##############################################################################
def  make_symb_link(P_proclist):
    os.chdir("./SubProcesses")
    for name in P_proclist:
        os.chdir("./"+name)
        #os.system('ln -s ../../Source/give_run_parameter.f')
        #os.system('ln -s ../../Source/update_sm_param.f')
        #os.system('ln -s ../madevent2')
        os.system('ln -s ../madevent_param')
        os.system('ln -s ../../Source/pdf.inc')
        os.chdir('..')
    os.chdir("..")   
    return


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

    for dir in MWparam.MW_listdir:
      if not os.path.exists("SubProcesses/"+dir+"/call_TF.f"): 
        print "Currently no transfer function loaded ..."
        listdir=os.listdir('./Source/MadWeight/transfer_function/data')
        print 'Available transfer functions:\n   ',
        print '\n    '.join([content[3:-4] for content in listdir if (content.startswith('TF') and content.endswith('dat'))])
        name=raw_input('Choose your transfer Function\n')
        P_dir,MW_dir=detect_SubProcess(P_mode=1)
        os.chdir('./Source/MadWeight/transfer_function')
        change_tf.create_TF_main(name,0,MW_dir)
        print os.getcwd()
        os.chdir('../../..')
        break


    if MWparam.run_opt['param']:
        create_param.Param_card(run_name=MWparam)
        MWparam.update_nb_card()
        Cards.create_include_file(MWparam)
        update_cuts_status(MWparam)

    if MWparam.run_opt['analyzer']:
        write_MadWeight.create_all_fortran_code(MWparam)
    if MWparam.run_opt['madweight_main']:    
        Launch_all_SubProcess(MWparam)

    if MWparam.run_opt['plot']:
        import plot
        plot.Likelihood(mw_param=MWparam,auto=1)
        if MWparam['mw_run']['histo']:
            plot.Differential_Graph(MWparam,auto=1)        

    if MWparam.run_opt['clean']:
        print 'cleaning in progress ....'
        from clean import Clean_run
        if MWparam.run_opt['clean']==1:
            Clean_run(MWparam.name)
        else:
            Clean_run(MWparam.run_opt['clean'])
