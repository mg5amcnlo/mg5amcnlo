#! /usr/bin/env python
################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""A sample script running a comparison between different loop ME generators using
objects and routines defined in me_comparator. To define your own test case, 
simply modify this script. Support for new ME generator is achieved through
inheritance of the MERunner class.
"""

import logging
import logging.config
import pydoc
import os
import sys

#Look for MG5/MG4 path
mg5_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
sys.path.append(mg5_path)
pickle_path = os.path.join(mg5_path,'tests','parallel_tests','input_files')

import loop_me_comparator
import me_comparator
from madgraph import MG4DIR
mg4_path = os.getcwd()



if '__main__' == __name__:

    # Helper function
    def SaveRunner(pickle_name, runner):
        """ Save a runner result to a give pickle path """

        pp=os.path.join(pickle_path,pickle_name)
        loop_me_comparator.LoopPickleRunner.store_comparison(pp,
            [runner.proc_list,runner.res_list],runner.model,
            runner.name,energy=runner.energy)

    # Get full logging info
    logging.config.fileConfig(os.path.join(mg5_path, 'tests', '.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('cmdprint').setLevel(logging.INFO)
    logging.getLogger('tutorial').setLevel(logging.ERROR)
        
    logging.basicConfig(level=logging.INFO)

    my_proc_list = []
#    my_proc_list.append(('g g > g g g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}))    
# Just to check that th general setup is ok, let's try some trivial processes
#    my_proc_list.append(('u u~ > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
#    my_proc_list.append(('d g > d g',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
#    my_proc_list.append(('g g > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
#    my_proc_list.append(('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4}))
#    my_proc_list.append(('u d~ > h t b~',{'QED':3,'QCD':0},['QCD'],{'QCD':2,'QED':6}))
## Check higgs massive tt~ processes
#    my_proc_list.append(('g g > h t t~',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2}))
#    my_proc_list.append(('g g > t t~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
#    my_proc_list.append(('g g > h h t t~',{'QCD':2,'QED':2},['QCD'],{'QCD':6,'QED':4}))
## Check of the gga R2
#    my_proc_list.append(('d~ d > g a',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}))
#    my_proc_list.append(('d~ d > g z',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}))
## Check of the ggh R2    
#    my_proc_list.append(('g g > h t t~ ',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}))
## Check of the ggvv R2    
#    my_proc_list.append(('d d~ > w+ w- g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}))
#    my_proc_list.append(('d~ d > z z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}))
#    my_proc_list.append(('d~ d > z a g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}))
## Check of the gggv R2 
#    my_proc_list.append(('d~ d > z g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}))
#    my_proc_list.append(('d~ d > a g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}))
## Now adding some masses to mess things around 
#    my_proc_list.append(('g g > z t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}))
#    my_proc_list.append(('g g > a t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}))    
## The ones below are a bit long
#    my_proc_list.append(('d d~ > d d~ d d~',{'QED':0,'QCD':4},['QCD'],{'QCD':10,'QED':0}))
## And an hardcore one for fun
#     my_proc_list.append(('d d~ > w+ w- t t~ g',{'QED':2,'QCD':3},['QCD'],{'QCD':8,'QED':4}))

    #my_proc_list = me_comparator.create_proc_list(['u', 'u~','d','d~','g'],
    #                                              initial=2, final=2)
    
    #my_proc_list = me_comparator.create_proc_list_enhanced(
    #    fermion, fermion, boson,
    #    initial=2, final_1=2, final_2 = 1)
    my_proc_list.append(('g g > t t~ g',{'QCD':3,'QED':0},['QED'],{'QCD':6,'QED':2}))
    # Set the model we are working with
    #model = 'loop_sm-no_widths'
    #model = 'loop_sm-parallel_test'
    model = 'loop_qcd_qed_sm-parallel_test'
    # the hard-coded reference value
    my_res_list=[]
    my_res_list.append(((-0.3123118956235762,0.00041827020029929103,\
                         0.022458628164095062,-2.590158624079223e-8,99),\
                        [[500.,0.,0.,500.],\
                         [500.,0.,0.,-500.],\
                         [185.32118462,30.4379583327,-55.0837756531,1.66127103109],\
                         [397.104074226,175.514562721,-296.936556388,-91.293348717],\
                         [417.574741153,-205.952521054,352.020332042,89.6320776859]]))
    # Load a saved run
    #SavedRun = me_comparator.PickleRunner.find_comparisons(
    #                     os.path.join(pickle_path,'ml5_parallel_test.pkl'))[0]
    # Create a MERunner object for MadLoop 4
    #ML4 = loop_me_comparator.LoopMG4Runner()
    #ML4.setup('/Users/Spooner/Documents/PhD/MadFKS/ML4ParrallelTest/NLOComp')
    # Create a MERunner object from Hard-Coded Reference value
    HCRvalue = loop_me_comparator.LoopHardCodedRefRunner()
    HCRvalue.setup(my_proc_list,my_res_list,model)
    # Create a MERunner object for GoSam
    #GoSam = loop_me_comparator.GoSamRunner()
    # GoSam.setup('/Users/erdissshaw/Works/GoSam/gosam_dir')
    # GoSam.setup('/Users/valentin/Documents/HEP_softs/GoSam/gosam-2.0.beta')

    # Create a MERunner object for MadLoop 5 optimized
    ML5_opt = loop_me_comparator.LoopMG5Runner()
    ML5_opt.setup(mg5_path, optimized_output=True)
    
    # Create a MERunner object for MadLoop 5 default
    ML5_default = loop_me_comparator.LoopMG5Runner()
    ML5_default.setup(mg5_path, optimized_output=False)

    # Create a MERunner object for UFO-ALOHA-MG5
#    my_mg5_ufo = me_comparator.MG5_UFO_Runner()
#    my_mg5_ufo.setup(mg5_path, mg4_path)

    # Create and setup a comparator
    my_comp = loop_me_comparator.LoopMEComparator()
    # Always put the saved run first if you use it, so that the corresponding PS
    # points will be used
    #my_comp.set_me_runners(ML5_opt,GoSam)
    my_comp.set_me_runners(HCRvalue,ML5_opt)

    # Run the actual comparison
    # energy=1000
    energy = HCRvalue.energy
    my_comp.run_comparison(my_proc_list,
                           model=model,
                           energy=energy)
    
    # Save the result of a runner
    #SaveRunner('trial.pkl',ML5_opt)

    # Do some cleanup
    #my_comp.cleanup()
    filename=model+'_results.log'

    # Print the output
    my_comp.output_result(filename=filename)

    pydoc.pager(file(filename,'r').read())

    # Print a list of non zero processes
    #print my_comp.get_non_zero_processes()
