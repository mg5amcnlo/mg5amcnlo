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
import me_comparator
import os
import sys

# Get full logging info
logging.basicConfig(level=logging.INFO)

#Look for MG5/MG4 path
mg5_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
sys.path.append(mg5_path)
from madgraph import MG4DIR
mg4_path = MG4DIR





# Create a list of processes to check automatically
#my_proc_list = me_comparator.create_proc_list(['w+', 'w-'],
#                                      initial=2, final=2)

# Create a MERunner object for MG4
#my_mg4 = me_comparator.MG4Runner()
#my_mg4.setup(mg4_path)

# Create a MERunner object for MG5
#my_mg5 = me_comparator.MG5Runner()
#my_mg5.setup(mg5_path, mg4_path)

# Create and setup a comparator
#my_comp = me_comparator.MEComparator()
#my_comp.set_me_runners(my_mg4, my_mg5)

# Run the actual comparison
#my_comp.run_comparison(my_proc_list,
#                        model='sm', orders={'QED':2, 'QCD':2}, energy=500)

# Do some cleanup
#my_comp.cleanup()

# Print the output
#my_comp.output_result(filename='sm_result.log')

# Print a list of non zero processes
#print my_comp.get_non_zero_processes()

#my_proc_list = ['u u > u u', 'u u~ > u u~', 'u u~ > d d~', 'u u~ > b b~', 'u u~ > t t~', 'u u~ > w+ w-', 'u u~ > a a', 'u u~ > a z', 'u u~ > a g', 'u u~ > z z', 'u u~ > g z', 'u u~ > h z', 'u u~ > g g', 'd u > d u', 'd~ u > d~ u', 'd~ u > b~ t', 'd~ u > a w+', 'd~ u > w+ z', 'd~ u > g w+', 'd~ u > h w+', 'b u > b u', 'b u > d t', 'b~ u > b~ u', 't u > t u', 't~ u > t~ u', 't~ u > b~ d', 'u w+ > u w+', 'u w- > u w-', 'u w- > a d', 'u w- > d z', 'u w- > d g', 'u w- > d h', 'a u > a u', 'a u > u z', 'a u > g u', 'a u > d w+', 'u z > a u', 'u z > u z', 'u z > g u', 'u z > h u', 'u z > d w+', 'g u > a u', 'g u > u z', 'g u > g u', 'g u > d w+', 'h u > u z', 'h u > d w+', 'u~ u~ > u~ u~', 'd u~ > d u~', 'd u~ > b t~', 'd u~ > a w-', 'd u~ > w- z', 'd u~ > g w-', 'd u~ > h w-', 'd~ u~ > d~ u~', 'b u~ > b u~', 'b~ u~ > b~ u~', 'b~ u~ > d~ t~', 't u~ > t u~', 't u~ > b d~', 't~ u~ > t~ u~', 'u~ w+ > u~ w+', 'u~ w+ > a d~', 'u~ w+ > d~ z', 'u~ w+ > d~ g', 'u~ w+ > d~ h', 'u~ w- > u~ w-', 'a u~ > a u~', 'a u~ > u~ z', 'a u~ > g u~', 'a u~ > d~ w-', 'u~ z > a u~', 'u~ z > u~ z', 'u~ z > g u~', 'u~ z > h u~', 'u~ z > d~ w-', 'g u~ > a u~', 'g u~ > u~ z', 'g u~ > g u~', 'g u~ > d~ w-', 'h u~ > u~ z', 'h u~ > d~ w-', 'd d > d d', 'd d~ > u u~', 'd d~ > d d~', 'd d~ > b b~', 'd d~ > t t~', 'd d~ > w+ w-', 'd d~ > a a', 'd d~ > a z', 'd d~ > a g', 'd d~ > z z', 'd d~ > g z', 'd d~ > h z', 'd d~ > g g', 'b d > b d', 'b~ d > t~ u', 'b~ d > b~ d', 'd t > b u', 'd t > d t', 'd t~ > d t~', 'd w+ > a u', 'd w+ > u z', 'd w+ > g u', 'd w+ > h u', 'd w+ > d w+', 'd w- > d w-', 'a d > u w-', 'a d > a d', 'a d > d z', 'a d > d g', 'd z > u w-', 'd z > a d', 'd z > d z', 'd z > d g', 'd z > d h', 'd g > u w-', 'd g > a d', 'd g > d z', 'd g > d g', 'd h > u w-', 'd h > d z', 'd~ d~ > d~ d~', 'b d~ > t u~', 'b d~ > b d~', 'b~ d~ > b~ d~', 'd~ t > d~ t', 'd~ t~ > b~ u~', 'd~ t~ > d~ t~', 'd~ w+ > d~ w+', 'd~ w- > a u~', 'd~ w- > u~ z', 'd~ w- > g u~', 'd~ w- > h u~', 'd~ w- > d~ w-', 'a d~ > u~ w+', 'a d~ > a d~', 'a d~ > d~ z', 'a d~ > d~ g', 'd~ z > u~ w+', 'd~ z > a d~', 'd~ z > d~ z', 'd~ z > d~ g', 'd~ z > d~ h', 'd~ g > u~ w+', 'd~ g > a d~', 'd~ g > d~ z', 'd~ g > d~ g', 'd~ h > u~ w+', 'd~ h > d~ z', 'b b > b b', 'b b~ > u u~', 'b b~ > d d~', 'b b~ > b b~', 'b b~ > t t~', 'b b~ > w+ w-', 'b b~ > a a', 'b b~ > a z', 'b b~ > a g', 'b b~ > a h', 'b b~ > z z', 'b b~ > g z', 'b b~ > h z', 'b b~ > g g', 'b b~ > g h', 'b b~ > h h', 'b t > b t', 'b t~ > d u~', 'b t~ > b t~', 'b t~ > a w-', 'b t~ > w- z', 'b t~ > g w-', 'b t~ > h w-', 'b w+ > b w+', 'b w+ > a t', 'b w+ > t z', 'b w+ > g t', 'b w+ > h t', 'b w- > b w-', 'a b > a b', 'a b > b z', 'a b > b g', 'a b > b h', 'a b > t w-', 'b z > a b', 'b z > b z', 'b z > b g', 'b z > b h', 'b z > t w-', 'b g > a b', 'b g > b z', 'b g > b g', 'b g > b h', 'b g > t w-', 'b h > a b', 'b h > b z', 'b h > b g', 'b h > b h', 'b h > t w-', 'b~ b~ > b~ b~', 'b~ t > d~ u', 'b~ t > b~ t', 'b~ t > a w+', 'b~ t > w+ z', 'b~ t > g w+', 'b~ t > h w+', 'b~ t~ > b~ t~', 'b~ w+ > b~ w+', 'b~ w- > b~ w-', 'b~ w- > a t~', 'b~ w- > t~ z', 'b~ w- > g t~', 'b~ w- > h t~', 'a b~ > a b~', 'a b~ > b~ z', 'a b~ > b~ g', 'a b~ > b~ h', 'a b~ > t~ w+', 'b~ z > a b~', 'b~ z > b~ z', 'b~ z > b~ g', 'b~ z > b~ h', 'b~ z > t~ w+', 'b~ g > a b~', 'b~ g > b~ z', 'b~ g > b~ g', 'b~ g > b~ h', 'b~ g > t~ w+', 'b~ h > a b~', 'b~ h > b~ z', 'b~ h > b~ g', 'b~ h > b~ h', 'b~ h > t~ w+', 't t > t t', 't t~ > u u~', 't t~ > d d~', 't t~ > b b~', 't t~ > t t~', 't t~ > w+ w-', 't t~ > a a', 't t~ > a z', 't t~ > a g', 't t~ > a h', 't t~ > z z', 't t~ > g z', 't t~ > h z', 't t~ > g g', 't t~ > g h', 't t~ > h h', 't w+ > t w+', 't w- > a b', 't w- > b z', 't w- > b g', 't w- > b h', 't w- > t w-', 'a t > b w+', 'a t > a t', 'a t > t z', 'a t > g t', 'a t > h t', 't z > b w+', 't z > a t', 't z > t z', 't z > g t', 't z > h t', 'g t > b w+', 'g t > a t', 'g t > t z', 'g t > g t', 'g t > h t', 'h t > b w+', 'h t > a t', 'h t > t z', 'h t > g t', 'h t > h t', 't~ t~ > t~ t~', 't~ w+ > a b~', 't~ w+ > b~ z', 't~ w+ > b~ g', 't~ w+ > b~ h', 't~ w+ > t~ w+', 't~ w- > t~ w-', 'a t~ > b~ w-', 'a t~ > a t~', 'a t~ > t~ z', 'a t~ > g t~', 'a t~ > h t~', 't~ z > b~ w-', 't~ z > a t~', 't~ z > t~ z', 't~ z > g t~', 't~ z > h t~', 'g t~ > b~ w-', 'g t~ > a t~', 'g t~ > t~ z', 'g t~ > g t~', 'g t~ > h t~', 'h t~ > b~ w-', 'h t~ > a t~', 'h t~ > t~ z', 'h t~ > g t~', 'h t~ > h t~', 'w+ w+ > w+ w+', 'w+ w- > u u~', 'w+ w- > d d~', 'w+ w- > b b~', 'w+ w- > t t~', 'w+ w- > w+ w-', 'w+ w- > a a', 'w+ w- > a z', 'w+ w- > a h', 'w+ w- > z z', 'w+ w- > h z', 'w+ w- > h h', 'a w+ > d~ u', 'a w+ > b~ t', 'a w+ > a w+', 'a w+ > w+ z', 'a w+ > h w+', 'w+ z > d~ u', 'w+ z > b~ t', 'w+ z > a w+', 'w+ z > w+ z', 'w+ z > h w+', 'g w+ > d~ u', 'g w+ > b~ t', 'h w+ > d~ u', 'h w+ > b~ t', 'h w+ > a w+', 'h w+ > w+ z', 'h w+ > h w+', 'w- w- > w- w-', 'a w- > d u~', 'a w- > b t~', 'a w- > a w-', 'a w- > w- z', 'a w- > h w-', 'w- z > d u~', 'w- z > b t~', 'w- z > a w-', 'w- z > w- z', 'w- z > h w-', 'g w- > d u~', 'g w- > b t~', 'h w- > d u~', 'h w- > b t~', 'h w- > a w-', 'h w- > w- z', 'h w- > h w-', 'a a > u u~', 'a a > d d~', 'a a > b b~', 'a a > t t~', 'a a > w+ w-', 'a z > u u~', 'a z > d d~', 'a z > b b~', 'a z > t t~', 'a z > w+ w-', 'a g > u u~', 'a g > d d~', 'a g > b b~', 'a g > t t~', 'a h > b b~', 'a h > t t~', 'a h > w+ w-', 'z z > u u~', 'z z > d d~', 'z z > b b~', 'z z > t t~', 'z z > w+ w-', 'z z > z z', 'z z > h h', 'g z > u u~', 'g z > d d~', 'g z > b b~', 'g z > t t~', 'h z > u u~', 'h z > d d~', 'h z > b b~', 'h z > t t~', 'h z > w+ w-', 'h z > h z', 'g g > u u~', 'g g > d d~', 'g g > b b~', 'g g > t t~', 'g g > g g', 'g h > b b~', 'g h > t t~', 'h h > b b~', 'h h > t t~', 'h h > w+ w-', 'h h > z z', 'h h > h h']

