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
import copy

import madgraph.core.base_objects as base_objects

"""Main diagram generation algorithm
"""

def generate_diagrams(proc, ref_dict_to0, ref_dict_to1):
    """Generate diagrams. For algorithm, see wiki page
    """

    for leg in proc['legs']:
        leg.set('from_group', True)

    max_multi_to1 = max([len(key) for key in ref_dict_to1.keys()])

    combine_legs([leg for leg in proc['legs']],ref_dict_to1,max_multi_to1)


    return []

def combine_legs(list_legs, ref_dict_to1, max_multi_to1):
    """Take a list of legs as an input, with the reference dictionary n-1>1,
    and output a list of list of tuples of Legs (allowed combinations) 
    and Legs (rest). For algorithm, see wiki page.
    """

    # I suggest that we change return format to be a list of pairs of lists:
    # [[tuples of Legs + Legs],[lists of vertices, same number as tuples]]
    # to make substitution trivial in the next step
    # (otherwise need to check dictionary again for the tuples)
    # I have prepared for this by letting LegList.passesTo1 return id
    # of new particle
    # We also need to add interaction id (i.e., number in InteractionList)
    # in the ref_dict:s.

    res = []

    for comb_length in range(2, max_multi_to1 + 1):
        for comb in itertools.combinations(list_legs,comb_length):
            newleg = base_objects.LegList(comb).passesTo1(ref_dict_to1)
            if newleg:
                res_list = copy.copy(list_legs)
                for leg in comb:
                    res_list.remove(leg)
                res_list.insert(0,comb)
                res.append(res_list)
                res_list1 = list_legs[0:list_legs.index(comb[0])]
                res_list2 = list_legs[list_legs.index(comb[0])+1:]
                for leg in comb[1:]:
                    res_list2.remove(leg)
                res_list=[comb]
                res_list.extend(res_list1)
                # This is where recursion happens
                for item in combine_legs(res_list2,ref_dict_to1,max_multi_to1):
                    res_list3=copy.copy(res_list)
                    res_list3.extend(item)
                    res.append(res_list3)
    return res



