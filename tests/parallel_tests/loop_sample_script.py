#! /usr/bin/env python
################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
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

import loop_me_comparator
import me_comparator
from madgraph import MG4DIR
mg4_path = os.getcwd()



if '__main__' == __name__: 
    # Get full logging info
    logging.config.fileConfig(os.path.join(mg5_path, 'tests', '.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('cmdprint').setLevel(logging.INFO)
    logging.getLogger('tutorial').setLevel(logging.ERROR)
        
    logging.basicConfig(level=logging.INFO)
    
    my_proc_list = [('u u~ > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                    ('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4}),
                    ('d~ d > g a',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}),
                    ('d~ d > g z',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2})]
#                    ('d d~ > u u~',{'QCD':2},['QCD'],{'QCD':6}),
#                    'u u~ > d d~',
#                    'g g > t t~',
#                    'u d > u d',
#                    'g d > g d',
#                    'd d~ > d d~ g',
#                    ('g g > u u~',{'QCD':2},['QCD'],{'QCD':6})]
#                    'd d~ > d d~ d d~',
#                    'd d > d~ d d d',
#                    'd g > d u u~']
#                    'd d~ > u u~ g',
    
    #my_proc_list = me_comparator.create_proc_list(['u', 'u~','d','d~','g'],
    #                                              initial=2, final=2)
    
    #my_proc_list = me_comparator.create_proc_list_enhanced(
    #    fermion, fermion, boson,
    #    initial=2, final_1=2, final_2 = 1)

    # Set the model we are working with
    model = 'loop_sm'

    # Create a MERunner object for MadLoop 4
    ML4 = loop_me_comparator.LoopMG4Runner()
    ML4.setup('/Users/Spooner/Documents/PhD/MadFKS/ML4ParrallelTest/NLOComp')

    # Create a MERunner object for GoSam
    # GoSam = loop_me_comparator.GoSamRunner()
    # GoSam.setup('/Users/Spooner/Documents/PhD/HEP_softs/GoSam_bis')

    # Create a MERunner object for MadLoop 5
    ML5 = loop_me_comparator.LoopMG5Runner()
    ML5.setup(mg5_path)
    
    # Create a MERunner object for UFO-ALOHA-MG5
#    my_mg5_ufo = me_comparator.MG5_UFO_Runner()
#    my_mg5_ufo.setup(mg5_path, mg4_path)

    # Create and setup a comparator
    my_comp = loop_me_comparator.LoopMEComparator()
    my_comp.set_me_runners(ML5, ML4)

    # Run the actual comparison
    my_comp.run_comparison(my_proc_list,
                           model=model,
                           energy=2000)

    # Do some cleanup
    # my_comp.cleanup()
    filename=model+'_results.log'

    # Print the output
    my_comp.output_result(filename=filename)

    pydoc.pager(file(filename,'r').read())

    # Print a list of non zero processes
    #print my_comp.get_non_zero_processes()

