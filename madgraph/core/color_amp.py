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

        # Create a dictionary pdg code --> leg(s)
        dict_pdg_leg = {}
        for index, leg in enumerate(vertex.get('legs')):
            curr_num = leg.get('number')
            curr_pdg = leg.get('id')
            # If this is the last leg and not the last vertex, 
            # flip part/antipart, and replace last index by a new summed index
            if index == len(vertex.get('legs')) - 1 and \
                vertex != diagram.get('vertices')[-1]:
                part = model.get('particle_dict')[curr_pdg]
                curr_pdg = \
                    model.get('particle_dict')[curr_pdg].get_anti_pdg_code()
                repl_dict[curr_num] = min_index
                min_index = min_index - 1
            if curr_num in repl_dict.keys():
                curr_num = repl_dict[curr_num]
            if curr_pdg in dict_pdg_leg.keys():
                dict_pdg_leg[curr_pdg].append(curr_num)
            else:
                dict_pdg_leg[curr_pdg] = [curr_num]

        # Create a list of associated leg number following the same order
        list_numbers = []
        for pdg_code in list_pdg:
            list_numbers.append(dict_pdg_leg[pdg_code].pop())

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

def build_color_basis(list_col_fact):
    """Takes a list of color factors and returns a dictionary with keys being
    the different structures (now tuple of strings, not ColorString!) 
    and values being a list of pairs with first element the coefficient 
    of the color factor and the second one its index."""

    color_basis = {}

    for index, col_fact in enumerate(list_col_fact):
        for col_str in col_fact:
            coeff, remain = col_str.extract_coeff()
            remain.sort()
            remain = tuple(remain)
            if remain in color_basis.keys():
                color_basis[remain].append((coeff, index))
            else:
                color_basis[remain] = [(coeff, index)]

    return color_basis



