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

import re
import copy

import madgraph.core.color_algebra as color_algebra

"""Classes, methods and functions required to write QCD color information 
for a diagram."""

def colorize(diagram, model):
    """Takes a diagram and a model as input and output the ColorString
    object associated to the diagram"""

    min_index = -100
    col_str = color_algebra.ColorString()

    repl_dict = {}

    for vertex in diagram['vertices']:
        if vertex['id'] == 0:
            break

        # List of pdg codes entering the vertex ordered as in
        # interactions.py
        list_pdg = [part.get_pdg_code() for part in \
               model.get_interaction(vertex['id'])['particles']]

        # Create a list of leg number following the same order
        list_numbers = []

        # Step 1: create a dictionary associating PDG code -> number
        list_leg = {}
        for pdg_code in list_pdg:
            if pdg_code not in list_leg.keys():
                list_leg[pdg_code] = [leg['number'] for leg in vertex['legs'] \
                                      if leg['id'] == pdg_code]

        # Replace the PDG code by numbers, if several times the same
        # PDG code, pop the corresponding numbers according to the vertex
        # ordering
        for pdg_code in list_pdg:
            my_number = list_leg[pdg_code].pop()
            if my_number not in list_numbers:
                list_numbers.append(my_number)
            elif my_number in repl_dict.keys():
                list_numbers.append(repl_dict[my_number])
            else:
                list_numbers.append(min_index)
                repl_dict[my_number] = min_index
                min_index = min_index - 1

        my_col_str = color_algebra.ColorString(model.get_interaction(\
                                                    vertex['id'])['color'][0])
        my_col_str = replace_index(my_col_str, list_numbers)

        col_str.extend(my_col_str)

    return col_str

def replace_index(col_str, list_indices):
    """Replace all occurences of index i in col_str by list_indices[i]"""

    ret_str = color_algebra.ColorString()

    for col_obj in col_str:
        mystr = col_obj
        for ind in range(len(list_indices)):
            match = re.sub(r'(?P<start>,|\()%i(?P<end>,|\))' % ind,
                   r'\g<start>%i\g<end>' % list_indices[ind],
                   mystr)
            if match:
                mystr = match
        ret_str.append(mystr)

    return ret_str



