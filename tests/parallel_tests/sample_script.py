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
mg4_path = os.getcwd()



if '__main__' == __name__: 
    # Get full logging info
    logging.config.fileConfig(os.path.join(mg5_path, 'tests', '.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('cmdprint').setLevel(logging.INFO)
    logging.getLogger('tutorial').setLevel(logging.ERROR)
        
    logging.basicConfig(level=logging.INFO)
    #my_proc_list = ['u u~ > g y $ g', 'u~ u > g y $ g ', 'y > u u~','Z >  u u~']
    #my_proc_list = ['t t > t t', 't t~ > t t~', 't t~ > z z', 't z > t z', 't~ t~ > t~ t~', 't~ z > t~ z', 'g g > y y', 'g y > g y', 'y y > g g', 'y y > z z', 'y y > a a', 'y z > t t~', 'y z > y z', 'a y > a y',' z z > t t~', 'z z > y y', 'a a > y y']
    #my_proc_list = ['t t~ > t t~','e+ e- > e+ e-','w+ w- > w+ w-', 't t~ > t t~ / a z h g', 'y > t t~', 'y > t t~ g', 't t~ > t t~ / z h g','t t~ > t t~ / a h g','t t~ > t t~ / z a g', 't t~ > t t~ / h ', 't t~ > t t~ / z','t t~ > z > t t~', 'y t > t z', 't t~ > z > t t~']
    #my_proc_list += ['t t~ > z > t t~','u u~ > z > u u~', 't t~ > t t~', 'u u > u u']
    #my_proc_list = [ 't t~ > y > t t~','z z > y > t t~','t t~ > y > z z',' u u~ > y > t t~', 't t~ > y > u u~']
    #my_proc_list = [' t t~ > t t~ y', 't t~ > t t~ g','g g > g g g','u u~ > y > u u~', 't t~ > y > t t~' ]
    my_proc_list = me_comparator.create_proc_list(['u', 'u~','g','a','e+','e-','h','ve','ve~','z','w+','w-','d','d~','y'], initial=2,
                                                  final=2)

    #my_proc_list1 = me_comparator.create_proc_list(['t', 't~','g','a','ta+','ta-','h','vt','vt~','z','w+','w-','b','b~','y'], initial=2,
    #                                              final=2)
    #fermion = ['b','b~','t','t~','ta-','ta+','vt','vt~']
    #final = ['g','a','h','z','w+','w-','y']
    fermion=['b','b~','t','t~','c','c~','s','s~']
    fermion2=['ta-','ta+','vt','vt~','mu-','mu+','vm','vm~']
    boson = ['g','y','z','a','w+','w-']

#    my_proc_list = me_comparator.create_proc_list_enhanced(
#        fermion, fermion, boson,
#        initial=2, final_1=2, final_2 = 1)
#    my_proc_list += me_comparator.create_proc_list_enhanced(
#        fermion2, fermion2, boson,
#        initial=2, final_1=2, final_2 = 1)

    my_proc_list = [p+' / h' for p in my_proc_list]
    #print len(my_proc_list)
    #print my_proc_list
    #
    #my_proc_list += me_comparator.create_proc_list(['w+','w-','z','a','x1+','x1-','n1'], initial=2,
    #                                              final=3)
    #my_proc_list += me_comparator.create_proc_list(['g','u','u~','go','ul','ul~','ur','ur~'], initial=2,
    #                                              final=3)
    #my_proc_list = ['b b~ > h t t~','e+ e- > h ta+ ta-',' y h > ta+ ta-','e+ e- > h ta+ ta- $y','e+ e- > y > h ta+ ta-']
#    my_proc_list = ['ta+ ta- > e+ ve w- / h', 'ta+ ta- > y > e+ ve w- / h','ta+ ta- > e+ ve w- / h y']
#    my_proc_list = ['ta+ ta- > e+ ve w- / h e+ e- ve ve~ w+ w-', 'e+ ve > W+ y','t t~ > e+ ve w- / h e+ e- ve ve~ w+ w-']
    my_proc_list = ['b b > a b b / h', 'b b~ > a b b~ / h', 'b b~ > g t t~ / h', 'b b~ > t t~ z / h', 'b b~ > a t t~ / h', 'b b~ > c c~ z / h', 'b b~ > a c c~ / h', 'b b~ > g s s~ / h', 'b t > b g t / h', 'b t > b t z / h', 'b t > a b t / h', 'b t~ > b g t~ / h', 'b t~ > b t~ y / h', 'b t~ > b t~ z / h', 'b t~ > a b t~ / h', 'b t~ > c~ s y / h', 'b c > b c z / h', 'b c > s t y / h', 'b c~ > b c~ z / h', 'b s~ > c~ t y / h', 'b~ b~ > a b~ b~ / h', 'b~ t > b~ g t / h', 'b~ t > b~ t y / h', 'b~ t > b~ t z / h', 'b~ t > a b~ t / h', 'b~ t > c s~ y / h', 'b~ t~ > b~ g t~ / h', 'b~ t~ > b~ t~ z / h', 'b~ t~ > a b~ t~ / h', 'b~ c > b~ c z / h', 'b~ c~ > b~ c~ z / h', 'b~ c~ > s~ t~ y / h', 'b~ s > c t~ y / h', 't t > g t t / h', 't t > t t z / h', 't t > a t t / h', 't t~ > b b~ g / h', 't t~ > b b~ y / h', 't t~ > b b~ z / h', 't t~ > a b b~ / h', 't t~ > g t t~ / h', 't t~ > t t~ y / h', 't t~ > t t~ z / h', 't t~ > a t t~ / h', 't t~ > c c~ g / h', 't t~ > c c~ y / h', 't t~ > c c~ z / h', 't t~ > a c c~ / h', 't t~ > g s s~ / h', 't t~ > s s~ z / h', 'c t > c g t / h', 'c t > c t z / h', 'c t > a c t / h', 'c~ t > b s~ y / h', 'c~ t > c~ g t / h', 'c~ t > c~ t z / h', 'c~ t > a c~ t / h', 's t > b c y / h', 's t > g s t / h', 's t > s t z / h', 's t > a s t / h', 's~ t > g s~ t / h', 's~ t > s~ t z / h', 's~ t > a s~ t / h', 't~ t~ > g t~ t~ / h', 't~ t~ > t~ t~ z / h', 't~ t~ > a t~ t~ / h', 'c t~ > b~ s y / h', 'c t~ > c g t~ / h', 'c t~ > c t~ y / h', 'c t~ > c t~ z / h', 'c t~ > a c t~ / h', 'c~ t~ > c~ g t~ / h', 'c~ t~ > c~ t~ z / h', 'c~ t~ > a c~ t~ / h', 's t~ > g s t~ / h', 's t~ > s t~ z / h', 's t~ > a s t~ / h', 's~ t~ > b~ c~ y / h', 's~ t~ > g s~ t~ / h', 's~ t~ > s~ t~ z / h', 'c c > c c g / h', 'c c > c c z / h', 'c c > a c c / h', 'c c~ > b b~ z / h', 'c c~ > a b b~ / h', 'c c~ > g t t~ / h', 'c c~ > t t~ y / h', 'c c~ > t t~ z / h', 'c c~ > a t t~ / h', 'c c~ > c c~ g / h', 'c c~ > c c~ y / h', 'c c~ > c c~ z / h', 'c c~ > a c c~ / h', 'c c~ > g s s~ / h', 'c c~ > s s~ y / h', 'c c~ > s s~ z / h', 'c c~ > a s s~ / h', 'c s > c g s / h', 'c s > c s y / h', 'c s > c s z / h', 'c s > a c s / h', 'c s~ > b~ t y / h', 'c s~ > c g s~ / h', 'c s~ > c s~ y / h', 'c s~ > c s~ z / h', 'c~ c~ > c~ c~ g / h', 'c~ c~ > c~ c~ y / h', 'c~ c~ > c~ c~ z / h', 'c~ c~ > a c~ c~ / h', 'c~ s > b t~ y / h', 'c~ s > c~ g s / h', 'c~ s > c~ s y / h', 'c~ s > c~ s z / h', 'c~ s > a c~ s / h', 'c~ s~ > c~ g s~ / h', 'c~ s~ > c~ s~ y / h', 'c~ s~ > c~ s~ z / h', 'c~ s~ > a c~ s~ / h', 's s > g s s / h', 's s > a s s / h', 's s~ > t t~ z / h', 's s~ > c c~ g / h', 's s~ > c c~ y / h', 's s~ > c c~ z / h', 's s~ > a c c~ / h', 's s~ > g s s~ / h', 's s~ > a s s~ / h', 's~ s~ > g s~ s~ / h', 's~ s~ > a s~ s~ / h', 'ta- ta- > a ta- ta- / h', 'ta- ta- > ta- vt w- / h', 'ta+ ta- > a ta+ ta- / h', 'ta+ ta- > ta+ vt w- / h', 'ta+ ta- > vt vt~ y / h', 'ta+ ta- > vt vt~ z / h', 'ta+ ta- > a vt vt~ / h', 'ta+ ta- > a mu+ mu- / h', 'ta+ ta- > mu- vm~ w+ / h', 'ta+ ta- > mu+ vm w- / h', 'ta+ ta- > vm vm~ z / h', 'ta+ ta- > a vm vm~ / h', 'ta- vt > ta- ta- w+ / h', 'ta- vt > ta- vt z / h', 'ta- vt > a ta- vt / h', 'ta- vt > vt vt w- / h', 'ta- vt~ > ta- vt~ y / h', 'ta- vt~ > a ta- vt~ / h', 'ta- vt~ > vt vt~ w- / h', 'ta- vt~ > mu+ mu- w- / h', 'ta- vt~ > vm vm~ w- / h', 'mu- ta- > ta- vm w- / h', 'mu- ta- > mu- vt w- / h', 'mu+ ta- > mu+ vt w- / h', 'mu+ ta- > vm~ vt y / h', 'ta- vm > mu- ta- w+ / h', 'ta- vm > ta- vm y / h', 'ta- vm > a ta- vm / h', 'ta- vm > mu- vt y / h', 'ta- vm > vm vt w- / h', 'ta- vm~ > mu+ ta- w- / h', 'ta- vm~ > ta- vm~ y / h', 'ta- vm~ > a ta- vm~ / h', 'ta- vm~ > vm~ vt w- / h', 'ta+ ta+ > a ta+ ta+ / h', 'ta+ ta+ > ta+ vt~ w+ / h', 'ta+ vt > ta+ ta- w+ / h', 'ta+ vt > ta+ vt y / h', 'ta+ vt > a ta+ vt / h', 'ta+ vt > mu+ mu- w+ / h', 'ta+ vt > vm vm~ w+ / h', 'ta+ vt~ > ta+ ta+ w- / h', 'ta+ vt~ > ta+ vt~ z / h', 'ta+ vt~ > a ta+ vt~ / h', 'ta+ vt~ > vt~ vt~ w+ / h', 'mu- ta+ > mu- vt~ w+ / h', 'mu- ta+ > vm vt~ y / h', 'mu+ ta+ > ta+ vm~ w+ / h', 'mu+ ta+ > mu+ vt~ w+ / h', 'ta+ vm > mu- ta+ w+ / h', 'ta+ vm > ta+ vm y / h', 'ta+ vm > a ta+ vm / h', 'ta+ vm > vm vt~ w+ / h', 'ta+ vm~ > mu+ ta+ w- / h', 'ta+ vm~ > ta+ vm~ y / h', 'ta+ vm~ > a ta+ vm~ / h', 'ta+ vm~ > mu+ vt~ y / h', 'ta+ vm~ > vm~ vt~ w+ / h', 'vt vt > ta- vt w+ / h', 'vt vt > vt vt y / h', 'vt vt > vt vt z / h', 'vt vt~ > ta+ ta- y / h', 'vt vt~ > ta+ ta- z / h', 'vt vt~ > a ta+ ta- / h', 'vt vt~ > ta+ vt w- / h', 'vt vt~ > mu+ mu- y / h', 'vt vt~ > mu+ mu- z / h', 'vt vt~ > a mu+ mu- / h', 'vt vt~ > mu- vm~ w+ / h', 'vt vt~ > vm vm~ y / h', 'mu- vt > mu- ta- w+ / h', 'mu- vt > ta- vm y / h', 'mu- vt > mu- vt y / h', 'mu- vt > a mu- vt / h', 'mu- vt > vm vt w- / h', 'mu+ vt > mu+ ta- w+ / h', 'mu+ vt > mu+ vt y / h', 'mu+ vt > a mu+ vt / h', 'mu+ vt > vm~ vt w+ / h', 'vm vt > ta- vm w+ / h', 'vm vt > vm vt y / h', 'vm vt > vm vt z / h', 'vm~ vt > mu+ ta- y / h', 'vm~ vt > ta- vm~ w+ / h', 'vt~ vt~ > ta+ vt~ w- / h', 'vt~ vt~ > vt~ vt~ y / h', 'vt~ vt~ > vt~ vt~ z / h', 'mu- vt~ > mu- ta+ w- / h', 'mu- vt~ > mu- vt~ y / h', 'mu- vt~ > a mu- vt~ / h', 'mu- vt~ > vm vt~ w- / h', 'mu+ vt~ > mu+ ta+ w- / h', 'mu+ vt~ > ta+ vm~ y / h', 'mu+ vt~ > mu+ vt~ y / h', 'mu+ vt~ > a mu+ vt~ / h', 'mu+ vt~ > vm~ vt~ w+ / h', 'vm vt~ > mu- ta+ y / h', 'vm vt~ > ta+ vm w- / h', 'vm~ vt~ > ta+ vm~ w- / h', 'vm~ vt~ > vm~ vt~ y / h', 'vm~ vt~ > vm~ vt~ z / h', 'mu- mu- > a mu- mu- / h', 'mu- mu- > mu- vm w- / h', 'mu+ mu- > a ta+ ta- / h', 'mu+ mu- > ta- vt~ w+ / h', 'mu+ mu- > ta+ vt w- / h', 'mu+ mu- > vt vt~ z / h', 'mu+ mu- > a vt vt~ / h', 'mu+ mu- > a mu+ mu- / h', 'mu+ mu- > mu+ vm w- / h', 'mu+ mu- > vm vm~ y / h', 'mu+ mu- > vm vm~ z / h', 'mu+ mu- > a vm vm~ / h', 'mu- vm > mu- mu- w+ / h', 'mu- vm > mu- vm z / h', 'mu- vm > a mu- vm / h', 'mu- vm > vm vm w- / h', 'mu- vm~ > ta+ ta- w- / h', 'mu- vm~ > vt vt~ w- / h', 'mu- vm~ > mu- vm~ y / h', 'mu- vm~ > a mu- vm~ / h', 'mu- vm~ > vm vm~ w- / h', 'mu+ mu+ > a mu+ mu+ / h', 'mu+ mu+ > mu+ vm~ w+ / h', 'mu+ vm > ta+ ta- w+ / h', 'mu+ vm > vt vt~ w+ / h', 'mu+ vm > mu+ mu- w+ / h', 'mu+ vm > mu+ vm y / h', 'mu+ vm > a mu+ vm / h', 'mu+ vm~ > mu+ mu+ w- / h', 'mu+ vm~ > mu+ vm~ z / h', 'mu+ vm~ > a mu+ vm~ / h', 'mu+ vm~ > vm~ vm~ w+ / h', 'vm vm > mu- vm w+ / h', 'vm vm > vm vm y / h', 'vm vm > vm vm z / h', 'vm vm~ > ta+ ta- y / h', 'vm vm~ > ta+ ta- z / h', 'vm vm~ > a ta+ ta- / h', 'vm vm~ > ta- vt~ w+ / h', 'vm vm~ > vt vt~ y / h', 'vm vm~ > mu+ mu- y / h', 'vm vm~ > mu+ mu- z / h', 'vm vm~ > a mu+ mu- / h', 'vm vm~ > mu+ vm w- / h', 'vm~ vm~ > mu+ vm~ w- / h', 'vm~ vm~ > vm~ vm~ y / h', 'vm~ vm~ > vm~ vm~ z / h']


    # Create a MERunner object for MG4
    my_mg4 = me_comparator.MG4Runner()
    my_mg4.setup(mg4_path)

    # Create a MERunner object for MG5
    my_mg5 = me_comparator.MG5Runner()
    my_mg5.setup(mg5_path, mg4_path)

    # Create a MERunner object for UFO-ALOHA-MG5
    my_mg5_ufo = me_comparator.MG5_UFO_Runner()
    my_mg5_ufo.setup(mg5_path, mg4_path)

    # Create a MERunner object for C++
    my_mg5_cpp = me_comparator.MG5_CPP_Runner()
    my_mg5_cpp.setup(mg5_path, mg4_path)

    # Create and setup a comparator
    my_comp = me_comparator.MEComparator()
    my_comp.set_me_runners(my_mg5_ufo, my_mg4)

    # Run the actual comparison
    my_comp.run_comparison(my_proc_list,
                           model=['RS','RS'],
                           orders={'QED':4, 'QCD':4, 'QTD':4}, energy=2000)

    # Do some cleanup
    #my_comp.cleanup()
    filename='mssm_results2.log'

    filename='mssm_results.log'

    # Print the output
    my_comp.output_result(filename=filename)

    pydoc.pager(file(filename,'r').read())

    # Print a list of non zero processes
    #print my_comp.get_non_zero_processes()

