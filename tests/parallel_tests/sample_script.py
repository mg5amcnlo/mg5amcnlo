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
    logging.config.fileConfig(os.path.join(mg5_path,'tests','.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('cmdprint').setLevel(logging.INFO)
    logging.getLogger('tutorial').setLevel(logging.ERROR)
    
    
    
    
    logging.basicConfig(level=logging.INFO)
    # Create a list of processes to check automatically
    my_proc_list = me_comparator.create_proc_list(
                                ['mu+','mu-','sl4+','sl4-','sl1+','sl1-'],
                                initial=2, final=2)

    # or give one
    #my_proc_list = ['e+ e- > e+ e-', 'g g> g g', 'g g> g g g', 'g g> g g g g', 'a w+ > a h1 w+/z', 'a w+ > a a w+', 'a w- > a h1 w-', 'a w- > a a w-', 'a h1 > a w+ w-', 'a a > h1 w+ w-', 'a a > a w+ w-']
    #my_proc_list += ['u u~ > d d~', 'ul ul~ > g g' , 'go go > su1 su1~', 'su1 su1~ > g g', 'su1 su1~ > g g g', 'su1 su1~ > su1 su1~' ]
    #my_proc_list += ['u u~ > g g su1 su1~', 'g g > go go  su1 su1~']
    #my_proc_list = ['su1 su1~ > g g g','su1 su1~ > g g', 'g g > su1 su1~', 'g su1 > g su1']
    #my_proc_list += ['el sve~ > g g g','el sve~ > g g', 'g g > el sve~', 'g el > g el', 'el sve~ >el sve~']
    #my_proc_list = ['mu+ mu+ > sl4+ sl4+']
    
    
    # Create a MERunner object for MG4
    my_mg4 = me_comparator.MG4Runner()
    my_mg4.setup(mg4_path)

    # Create a MERunner object for MG5
    #my_mg5 = me_comparator.MG5Runner()
    #my_mg5.setup(mg5_path, mg4_path)

    # Create a MERunner object for UFO-ALOHA-MG5
    my_mg5_ufo = me_comparator.MG5_UFO_Runner()
    my_mg5_ufo.setup(mg5_path, mg4_path)

    # Create and setup a comparator
    my_comp = me_comparator.MEComparator()
    my_comp.set_me_runners(my_mg4, my_mg5_ufo)

    # Run the actual comparison
    my_comp.run_comparison(my_proc_list,
                       model=['mssm_mg5','mssm'], orders={'QED':4, 'QCD':4}, energy=10000)

    # Do some cleanup
    #my_comp.cleanup()

    # Print the output
    my_comp.output_result(filename='sm_result.log')

    
    pydoc.pager(file('sm_result.log','r').read())

    # Print a list of non zero processes
    #print my_comp.get_non_zero_processes()

