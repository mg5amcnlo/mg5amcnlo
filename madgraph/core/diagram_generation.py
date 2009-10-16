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

import itertools

"""Main diagram generation algorithm
"""

def generate_diagrams(proc, ref_dict_to0, ref_dict_to1):

    for leg in proc['legs']:
        leg.set('from_group', True)

    max_multi_to1 = max([len(key) for key in ref_dict_to1.keys()])

#    combine_legs(proc['legs'],ref_dict_to1,max_multi_to1)


    return []

def combine_legs(leg_list, ref_dict_to1, max_multi_to1):
    """Take a LegList as an input, with the reference dictionary n-1>1,
    and output a list of list of LegLists (allowed combination) 
    and Legs (rest).
    """

#    res = []
#    reslist=copy.copy(leg_list)
#    for comb_length in range(2, max_multi_to1 + 1):
#        for comb in itertools.combinations(leg_list,comb_length):
#            if comb in ref_dict_to1.keys() and :
#                
#                res.append([[comb]+(leg_list-comb)]))
#
#    print res




