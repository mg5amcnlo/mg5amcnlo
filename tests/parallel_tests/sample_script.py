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

"""A sample script running a comparison between different ME generators using
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
print mg5_path
sys.path.append(mg5_path)

import me_comparator
from madgraph import MG4DIR
mg4_path = MG4DIR



if '__main__' == __name__: 
    # Get full logging info
    logging.config.fileConfig(os.path.join(mg5_path, 'tests', '.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('cmdprint').setLevel(logging.INFO)
    logging.getLogger('tutorial').setLevel(logging.ERROR)
        
    logging.basicConfig(level=logging.INFO)
#    my_proc_list = me_comparator.create_proc_list_enhanced(
##        ['w+', 'w-', 'z'],
##        initial=2, final_1=2)
    #my_proc_list = ['w+ w+ > h1 x1+ x1+', ' w+ w+ > h+ h+ h1', ' w+ w- > el+ el- h1', ' w+ w- > el+ el- h2', ' w+ w- > h1 h1 h1', ' w+ w- > h1 h1 h2', ' w+ w- > h1 h2 h2', ' w+ w- > h1 h3 h3', ' w+ w- > h+ h- h1', ' w+ w- > h2 h2 h2', ' w+ w- > h2 h3 h3', ' w+ w- > h+ h- h2', ' el+ w+ > el+ h1 w+', ' el+ w+ > el+ h2 w+', ' el+ w+ > el+ h+ h1', ' el+ w+ > el+ h+ h2', ' el+ w+ > el+ h+ h3', ' el- w+ > el- h1 w+', ' el- w+ > el- h2 w+', ' el- w+ > el- h+ h1', ' el- w+ > el- h+ h2', ' el- w+ > el- h+ h3', ' w+ x1+ > h1 w+ x1+', ' w+ x1+ > h+ h1 x1+', ' w+ x1+ > h+ h2 x1+', ' w+ x1+ > h+ h3 x1+', ' w+ x1- > h1 w+ x1-', ' w+ x1- > h1 w- x1+', ' w+ x1- > h+ h1 x1-', ' w+ x1- > h+ h2 x1-', ' w+ x1- > h+ h3 x1-', ' h1 w+ > h1 h1 w+', ' h1 w+ > h1 h2 w+', ' h1 w+ > w- x1+ x1+', ' h1 w+ > h+ h+ w-', ' h1 w+ > h+ x1+ x1-', ' h1 w+ > h+ h1 h1', ' h1 w+ > h+ h1 h2', ' h1 w+ > h+ h1 h3', ' h1 w+ > h+ h2 h3', ' h2 w+ > h1 h2 w+', ' h2 w+ > h2 h2 w+', ' h2 w+ > h+ x1+ x1-', ' h2 w+ > h+ h1 h2', ' h2 w+ > h+ h1 h3', ' h2 w+ > h+ h2 h2', ' h2 w+ > h+ h2 h3', ' h3 w+ > h1 h3 w+', ' h3 w+ > h2 h3 w+', ' h3 w+ > h+ x1+ x1-', ' h3 w+ > h+ h1 h3', ' h3 w+ > h+ h2 h3', ' h3 w+ > h+ h3 h3', ' h+ w+ > h1 w+ w+', ' h+ w+ > h+ h1 w+', ' h+ w+ > h+ h2 w+', ' h+ w+ > h+ h+ h1', ' h+ w+ > h+ h+ h2', ' h+ w+ > h+ h+ h3', ' h- w+ > h1 w+ w-', ' h- w+ > h- h1 w+', ' h- w+ > h- h2 w+', ' h- w+ > h+ h1 w-', ' h- w+ > h1 x1+ x1-', ' h- w+ > h2 x1+ x1-', ' h- w+ > h3 x1+ x1-', ' h- w+ > h1 h1 h3', ' h- w+ > h1 h2 h3', ' h- w+ > h1 h3 h3', ' h- w+ > h+ h- h1', ' h- w+ > h2 h2 h3', ' h- w+ > h2 h3 h3', ' h- w+ > h+ h- h2', ' h- w+ > h+ h- h3', ' w- w- > h1 x1- x1-', ' w- w- > h- h- h1', ' el+ w- > el+ h1 w-', ' el+ w- > el+ h2 w-', ' el+ w- > el+ h- h1', ' el+ w- > el+ h- h2', ' el+ w- > el+ h- h3', ' el- w- > el- h1 w-', ' el- w- > el- h2 w-', ' el- w- > el- h- h1', ' el- w- > el- h- h2', ' el- w- > el- h- h3', ' w- x1+ > h1 w+ x1-', ' w- x1+ > h1 w- x1+', ' w- x1+ > h- h1 x1+', ' w- x1+ > h- h2 x1+', ' w- x1+ > h- h3 x1+', ' w- x1- > h1 w- x1-', ' w- x1- > h- h1 x1-', ' w- x1- > h- h2 x1-', ' w- x1- > h- h3 x1-', ' h1 w- > w+ x1- x1-', ' h1 w- > h- h- w+', ' h1 w- > w- x1+ x1-', ' h1 w- > h1 h1 w-', ' h1 w- > h1 h2 w-', ' h1 w- > h- x1+ x1-', ' h1 w- > h- h1 h1', ' h1 w- > h- h1 h2', ' h1 w- > h- h1 h3', ' h1 w- > h- h2 h3', ' h2 w- > h1 h2 w-', ' h2 w- > h2 h2 w-', ' h2 w- > h- x1+ x1-', ' h2 w- > h- h1 h2', ' h2 w- > h- h1 h3', ' h2 w- > h- h2 h2', ' h2 w- > h- h2 h3', ' h3 w- > h1 h3 w-', ' h3 w- > h2 h3 w-', ' h3 w- > h- x1+ x1-', ' h3 w- > h- h1 h3', ' h3 w- > h- h2 h3', ' h3 w- > h- h3 h3', ' h+ w- > h1 w+ w-', ' h+ w- > h- h1 w+', ' h+ w- > h+ h1 w-', ' h+ w- > h+ h2 w-', ' h+ w- > h1 x1+ x1-', ' h+ w- > h2 x1+ x1-', ' h+ w- > h3 x1+ x1-', ' h+ w- > h1 h1 h3', ' h+ w- > h1 h2 h3', ' h+ w- > h1 h3 h3', ' h+ w- > h+ h- h1', ' h+ w- > h2 h2 h3', ' h+ w- > h2 h3 h3', ' h+ w- > h+ h- h2', ' h+ w- > h+ h- h3', ' h- w- > h1 w- w-', ' h- w- > h- h1 w-', ' h- w- > h- h2 w-', ' h- w- > h- h- h1', ' h- w- > h- h- h2', ' h- w- > h- h- h3', ' e- x1+ > e- h1 x1+', ' e- x1+ > e- h2 x1+', ' e- x1+ > e- h3 x1+', ' e+ x1- > e+ h1 x1-', ' e+ x1- > e+ h2 x1-', ' e+ x1- > e+ h3 x1-', ' el+ el+ > e+ e+ h2', ' el+ el- > h1 w+ w-', ' el+ el- > h2 w+ w-', ' el+ el- > h- h1 w+', ' el+ el- > h- h2 w+', ' el+ el- > h- h3 w+', ' el+ el- > h+ h1 w-', ' el+ el- > h+ h2 w-', ' el+ el- > h+ h3 w-', ' el+ el- > el+ el- h3', ' el+ h1 > el+ w+ w-', ' el+ h1 > el+ h- w+', ' el+ h1 > el+ h+ w-', ' el+ h1 > el+ h+ h-', ' el+ h2 > el+ w+ w-', ' el+ h2 > el+ h- w+', ' el+ h2 > el+ h+ w-', ' el+ h3 > el+ h- w+', ' el+ h3 > el+ h+ w-', ' el+ h+ > el+ h1 w+', ' el+ h+ > el+ h2 w+', ' el+ h+ > el+ h3 w+', ' el+ h- > el+ h1 w-', ' el+ h- > el+ h2 w-', ' el+ h- > el+ h3 w-', ' el- el- > e- e- h2', ' el- h1 > el- w+ w-', ' el- h1 > el- h- w+', ' el- h1 > el- h+ w-', ' el- h2 > el- w+ w-', ' el- h2 > el- h- w+', ' el- h2 > el- h+ w-', ' el- h3 > el- h- w+', ' el- h3 > el- h+ w-', ' el- h+ > el- h1 w+', ' el- h+ > el- h2 w+', ' el- h+ > el- h3 w+', ' el- h- > el- h1 w-', ' el- h- > el- h2 w-', ' el- h- > el- h3 w-', ' x1+ x1+ > h1 w+ w+', ' x1+ x1- > h1 w+ w-', ' h1 x1+ > w+ w+ x1-', ' h1 x1+ > w+ w- x1+', ' h1 x1+ > h- w+ x1+', ' h1 x1+ > h+ w- x1+', ' h2 x1+ > h- w+ x1+', ' h2 x1+ > h+ w- x1+', ' h3 x1+ > h- w+ x1+', ' h3 x1+ > h+ w- x1+', ' h+ x1+ > h1 w+ x1+', ' h+ x1+ > h2 w+ x1+', ' h+ x1+ > h3 w+ x1+', ' h- x1+ > h1 w- x1+', ' h- x1+ > h2 w- x1+', ' h- x1+ > h3 w- x1+', ' x1- x1- > h1 w- w-', ' h1 x1- > w+ w- x1-', ' h1 x1- > h- w+ x1-', ' h1 x1- > w- w- x1+', ' h1 x1- > h+ w- x1-', ' h2 x1- > h- w+ x1-', ' h2 x1- > h+ w- x1-', ' h3 x1- > h- w+ x1-', ' h3 x1- > h+ w- x1-', ' h+ x1- > h1 w+ x1-', ' h+ x1- > h2 w+ x1-', ' h+ x1- > h3 w+ x1-', ' h- x1- > h1 w- x1-', ' h- x1- > h2 w- x1-', ' h- x1- > h3 w- x1-', ' h1 h1 > h1 w+ w-', ' h1 h1 > h2 w+ w-', ' h1 h1 > h- h1 w+', ' h1 h1 > h- h2 w+', ' h1 h1 > h- h3 w+', ' h1 h1 > h+ h1 w-', ' h1 h1 > h+ h2 w-', ' h1 h1 > h+ h3 w-', ' h1 h1 > h+ h- h2', ' h1 h2 > h1 w+ w-', ' h1 h2 > h- h1 w+', ' h1 h2 > h- h3 w+', ' h1 h2 > h+ h1 w-', ' h1 h2 > h+ h3 w-', ' h1 h2 > el+ el- h2', ' h1 h3 > h- h1 w+', ' h1 h3 > h- h2 w+', ' h1 h3 > h+ h1 w-', ' h1 h3 > h+ h2 w-', ' h+ h1 > h- w+ w+', ' h+ h1 > h+ w+ w-', ' h+ h1 > w+ x1+ x1-', ' h+ h1 > h1 h3 w+', ' h+ h1 > h2 h3 w+', ' h+ h1 > h3 h3 w+', ' h+ h1 > h+ h- w+', ' h+ h1 > h+ h+ w-', ' h+ h1 > h+ h1 h2', ' h- h1 > h- w+ w-', ' h- h1 > h- h- w+', ' h- h1 > h+ w- w-', ' h- h1 > w- x1+ x1-', ' h- h1 > h1 h3 w-', ' h- h1 > h2 h3 w-', ' h- h1 > h3 h3 w-', ' h- h1 > h+ h- w-', ' h- h1 > h- h1 h2', ' h2 h2 > h1 w+ w-', ' h2 h2 > h2 w+ w-', ' h2 h2 > h- h1 w+', ' h2 h2 > h- h2 w+', ' h2 h2 > h- h3 w+', ' h2 h2 > h+ h1 w-', ' h2 h2 > h+ h2 w-', ' h2 h2 > h+ h3 w-', ' h2 h3 > h- h1 w+', ' h2 h3 > h- h2 w+', ' h2 h3 > h+ h1 w-', ' h2 h3 > h+ h2 w-', ' h+ h2 > h+ w+ w-', ' h+ h2 > w+ x1+ x1-', ' h+ h2 > h1 h3 w+', ' h+ h2 > h2 h3 w+', ' h+ h2 > h+ h- w+', ' h+ h2 > h+ h+ w-', ' h- h2 > h- w+ w-', ' h- h2 > h- h- w+', ' h- h2 > w- x1+ x1-', ' h- h2 > h1 h3 w-', ' h- h2 > h2 h3 w-', ' h- h2 > h+ h- w-', ' h3 h3 > h1 w+ w-', ' h3 h3 > h2 w+ w-', ' h3 h3 > h- h1 w+', ' h3 h3 > h- h2 w+', ' h3 h3 > h- h3 w+', ' h3 h3 > h+ h1 w-', ' h3 h3 > h+ h2 w-', ' h3 h3 > h+ h3 w-', ' h+ h3 > w+ x1+ x1-', ' h+ h3 > h1 h2 w+', ' h+ h3 > h1 h3 w+', ' h+ h3 > h2 h3 w+', ' h+ h3 > h+ h- w+', ' h+ h3 > h+ h+ w-', ' h- h3 > h- h- w+', ' h- h3 > w- x1+ x1-', ' h- h3 > h1 h2 w-', ' h- h3 > h1 h3 w-', ' h- h3 > h2 h3 w-', ' h- h3 > h+ h- w-', ' h+ h+ > h1 w+ w+', ' h+ h+ > h+ h1 w+', ' h+ h+ > h+ h2 w+', ' h+ h+ > h+ h3 w+', ' h+ h+ > h+ h+ h1', ' h+ h+ > h+ h+ h3', ' h+ h- > h1 w+ w-', ' h+ h- > h2 w+ w-', ' h+ h- > h- h1 w+', ' h+ h- > h- h2 w+', ' h+ h- > h- h3 w+', ' h+ h- > h+ h1 w-', ' h+ h- > h+ h2 w-', ' h+ h- > h+ h3 w-', ' h+ h- > el+ el- h2', ' h+ h- > h1 h1 h1', ' h- h- > h1 w- w-', ' h- h- > h- h1 w-', ' h- h- > h- h2 w-', ' h- h- > h- h3 w-', ' h- h- > h- h- h1', ' h- h- > h- h- h3']
    my_proc_list = ['e- x1+ > e- h1 x1+','e- x1+ > e- h2 x1+','e- x1+ > e- h3 x1+',
                    'e+ x1+ > e+ h1 x1+','e+ x1+ > e+ h2 x1+','e+ x1+ > e+ h3 x1+']
    my_proc_list += ['el+ h2 > el+ w+ w-']
                   
    my_proc_list = me_comparator.create_proc_list(['g', 'go'], initial=2,
                                                  final=2)

    my_proc_list = me_comparator.create_proc_list(['g', 'h'], initial=2,
                                                  final=3)
    my_proc_list = me_comparator.create_proc_list(['g', 'h3'], initial=2,
                                                  final=3)

    # Create a MERunner object for MG4
    my_mg4 = me_comparator.MG4Runner()
    my_mg4.setup(mg4_path)

    # Create a MERunner object for MG5
    my_mg5 = me_comparator.MG5Runner()
    my_mg5.setup(mg5_path, mg4_path)

    # Create a MERunner object for UFO-ALOHA-MG5
    my_mg5_ufo = me_comparator.MG5_UFO_Runner()
    my_mg5_ufo.setup(mg5_path, mg4_path)

    # Create and setup a comparator
    my_comp = me_comparator.MEComparator()
    my_comp.set_me_runners(my_mg5, my_mg5_ufo, my_mg4)

    # Run the actual comparison
    my_comp.run_comparison(my_proc_list,
                       model='heft', orders={'QED':4, 'QCD':4, 'HIG':1, 'HIW':1}, energy=1000)

    # Do some cleanup
    #my_comp.cleanup()

    # Print the output
    my_comp.output_result(filename='mssm_results.log')

    
    pydoc.pager(file('mssm_results.log','r').read())

    # Print a list of non zero processes
    #print my_comp.get_non_zero_processes()

