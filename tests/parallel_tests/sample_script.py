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
import __main__

"""A sample script running a comparison between different ME generators using
objects and routines defined in me_comparator. To define your own test case, 
simply modify this script. Support for new ME generator is achieved through
inheritance of the MERunner class.
"""

import logging
import os
import me_comparator

# Get full logging info
#logging.basicConfig(level=logging.INFO)


# specify the position of different codes
mg5_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
print 'mg5_path', mg5_path
mg4_path = None
mg4_dir_possibility = [os.path.join(mg5_path, os.path.pardir),
                os.path.join(os.getcwd(), os.path.pardir),
                os.getcwd()]

for position in mg4_dir_possibility:
    if os.path.exists(os.path.join(position, 'MGMEVersion.txt')) and \
                    os.path.exists(os.path.join(position, 'UpdateNotes.txt')):
        mg4_path = os.path.realpath(position)
        break
del mg4_dir_possibility
print 'mg4_path', mg4_path


# Create a list of processes to check automatically
#my_proc_list = me_comparator.create_proc_list(['w+', 'w-', 'a', 'h', 'u', 'u~', 'd', 'g'],
#                                      initial=2, final=2)

if '__main__' == __name__:
    my_proc_list = ['w+ w+ > w+ w+', 'w+ w- > w+ w-', 'w+ w- > a a', 
                'w+ w- > h h', 'w+ w- > u u~', 'a w+ > a w+', 'a w+ > h w+', 
                'h w+ > a w+', 'h w+ > h w+', 'u w+ > u w+', 'u~ w+ > u~ w+', 
                'd w+ > d w+', 'd w+ > a u', 'd w+ > h u', 'd w+ > g u',
                'w- w- > w- w-', 'a w- > a w-', 'a w- > h w-', 'a w- > d u~', 
                'h w- > a w-', 'h w- > h w-', 'h w- > d u~', 'u w- > u w-', 
                'u w- > a d', 'u w- > d h', 'u w- > d g', 'u~ w- > u~ w-',
                'd w- > d w-', 'g w- > d u~', 'a a > w+ w-', 'a a > u u~',
                'a h > w+ w-', 'a u > d w+', 'a u > a u', 'a u > g u', 
                'a u~ > a u~', 'a u~ > g u~', 'a d > u w-', 'a d > a d', 
                'a d > d g', 'a g > u u~', 'h h > w+ w-', 'h h > h h', 
                'h u > d w+', 'd h > u w-', 'u u > u u', 'u u~ > w+ w-', 
                'u u~ > a a', 'u u~ > a g', 'u u~ > u u~', 'u u~ > g g', 
                'd u > d u', 'g u > d w+', 'g u > a u', 'g u > g u', 
                'u~ u~ > u~ u~', 'd u~ > a w-', 'd u~ > h w-', 'd u~ > g w-',
                'd u~ > d u~', 'g u~ > a u~', 'g u~ > g u~', 'd d > d d', 
                'd g > u w-', 'd g > a d', 'd g > d g', 'g g > u u~', 
                'g g > g g', 'u u~ > z h h h', 'w+ w- > z h h h', 'w+ w- > a h']

    # Create a MERunner object for MG4
    my_mg4 = me_comparator.MG4Runner()
    my_mg4.setup(mg4_path)

    # Create a MERunner object for MG5
    my_mg5 = me_comparator.MG5Runner()
    my_mg5.setup(mg5_path, mg4_path)

    # Create and setup a comparator
    my_comp = me_comparator.MEComparator()
    my_comp.set_me_runners(my_mg4, my_mg5)

    # Run the actual comparison
    my_comp.run_comparison(my_proc_list,
                        model='sm', orders={'QED':99, 'QCD':99}, energy=500)


    # Do some cleanup
    my_comp.cleanup()

    # Print the output
    my_comp.output_result(filename='sm_result.log')

    # Print a list of non zero processes
    #print my_comp.get_non_zero_processes()


