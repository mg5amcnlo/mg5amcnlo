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

import copy
import re

import madgraph.core.color_algebra as color_algebra

"""Classes, methods and functions required to write QCD color information 
for a diagram."""

def colorize(diagram, model):
    """Takes a diagram and a model as input and output the ColorFactor
    object associated to the diagram"""

    # The smallest value used to create new summed indices
    min_index = -100
    # The color factor to be output
    col_fact = color_algebra.ColorFactor()
    # The dictionary for book keeping of replaced indices
    repl_dict = {}

    for vertex in diagram.get('vertices'):

        # SPECIAL VERTEX WITH ID = 0 -------------------------------------------

        if vertex['id'] == 0:
            # For vertex (i1,i2), replace all i2 by i1
            old_num = vertex.get('legs')[1].get('number')
            new_num = vertex.get('legs')[0].get('number')
            # Be careful i1 or i2 might have been replaced themselves
            if old_num in repl_dict.keys():
                old_num = repl_dict[old_num]
            if new_num in repl_dict.keys():
                new_num = repl_dict[new_num]
            # Do the replacement
            for index, col_str in enumerate(col_fact):
                ret_str = color_algebra.ColorString()
                for col_obj in col_str:
                    mystr = replace_index(col_obj, old_num, new_num)
                    mystr = clean_str(mystr)
                    ret_str.append(mystr)
                del col_fact[index]
                col_fact.insert(index, ret_str)
            # End the routine (vertex 0 should always be at the end)
            return col_fact

        # NORMAL VERTICES WITH ID != 0 -----------------------------------------
        # Create a list of pdg codes entering the vertex ordered as in
        # interactions.py
        list_pdg = [part.get_pdg_code() for part in \
               model.get_interaction(vertex.get('id')).get('particles')]
        # Create a list of associated leg number following the same order
        list_numbers = []
        # Step 1: create a dictionary associating PDG code -> number
        list_leg = {}

        # Create a copy of the current vertex where all legs labeled as initial 
        # state, except the last one (outgoing) have a flipped PDG code
        # ASK JOHAN IF IT'S NOT THE CONTRARY!!!
        flipped_vertex = copy.copy(vertex)
        for leg in flipped_vertex['legs']:
            if leg.get('state') == 'initial' and \
                flipped_vertex['legs'].index(leg) != \
                                    len(flipped_vertex['legs']) - 1:
                part = model.get('particle_dict')[leg.get('id')]
                leg.set('id', part.get_anti_pdg_code())

        for pdg_code in list_pdg:
            if pdg_code not in list_leg.keys():
                list_leg[pdg_code] = [leg['number'] for leg in \
                           flipped_vertex['legs'] if leg['id'] == pdg_code]

        # Step 2: replace the PDG code by numbers, if several times the same
        # PDG code, pop the corresponding numbers according to the vertex
        # ordering
        for pdg_code in list_pdg:
            my_number = list_leg[pdg_code].pop()
            if  my_number in repl_dict.keys():
                # If a number has already been replaced, use the new value
                list_numbers.append(repl_dict[my_number])
            elif my_number not in list_numbers:
                # Only appear once until now -> no need for a new index
                list_numbers.append(my_number)
            else:
                # If the number already appear, create a new index and save
                # it as replaced
                list_numbers.append(min_index)
                repl_dict[my_number] = min_index
                min_index = min_index - 1
        # Create a new ColorFactor to store new elements
        new_col_fact = color_algebra.ColorFactor()

        # For each new string
        for new_col_str in [color_algebra.ColorString(s) for s in \
                       model.get_interaction(vertex['id'])['color']]:

            # Create a modified string with replaced indices
            mod_str = color_algebra.ColorString()
            for col_obj in new_col_str:
                mystr = col_obj
                for ind in range(len(list_numbers)):
                    mystr = replace_index(mystr, ind, list_numbers[ind])
                mystr = clean_str(mystr)
                mod_str.append(mystr)
            # Attach it to all elements of the current color factor and
            # store the result in the new one
            if col_fact:
                for exist_str in col_fact:
                    final_str = copy.copy(exist_str)
                    final_str.extend(mod_str)
                    new_col_fact.append(final_str)
            else:
                new_col_fact.append(mod_str)

        # The new color factor becomes the current one
        col_fact = new_col_fact

    return col_fact


def replace_index(mystr, old_ind, new_ind):
    """Replace all indices old_ind by new_ind in mystr and return
    the result. Use clean_str when all replacement are done."""

    return re.sub(r'(?P<start>,|\()%i(?P<end>,|\))' % old_ind,
                   r'\g<start>X%i\g<end>' % new_ind,
                   mystr)

def clean_str(mystr):
    """Remove spurious X labels in front of replaced indices"""

    return re.sub(r'(?P<start>,|\()X(?P<index>-?\d+)',
                   r'\g<start>\g<index>',
                   mystr)

