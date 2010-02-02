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
import fractions
import operator
import re

import madgraph.core.color_algebra as color_algebra
import madgraph.core.diagram_generation as diagram_generation

"""Classes, methods and functions required to write QCD color information 
for a diagram and build a color basis, and to square a QCD color string for
squared diagrams and interference terms."""

#===============================================================================
# ColorBasis
#===============================================================================
class ColorBasis(dict):
    """The ColorBasis object is a dictionary created from an amplitude. Keys
    are the different color structures present in the amplitude. Values have
    the format (diag,(index c1, index c2,...), coeff, is_imaginary, Nc_power) 
    where diag is the diagram index, (index c1, index c2,...) the list of 
    indices corresponding to the chose color parts for each vertex in the 
    diagram, coeff the corresponding coefficient (a fraction), is_imaginary
    if this contribution is real or complex, and Nc_power the Nc power."""

    # Dictionary to save simplifications already done in a canonical form
    _canonical_dict = {}

    # Dictionary store the raw colorize information
    _list_color_dict = []

    def colorize(self, diagram, model):
        """Takes a diagram and a model and outputs a dictionary with keys being
        color coefficient index tuples and values a color string (before 
        simplification)."""

        # The smallest value used to create new summed indices
        min_index = -1000
        # The dictionary to be output
        res_dict = {}
        # The dictionary for book keeping of replaced indices
        repl_dict = {}

        for vertex in diagram.get('vertices'):

        # SPECIAL VERTEX WITH ID = 0 -------------------------------------------

            if vertex['id'] == 0:
                self.add_vertex_id_0(vertex, repl_dict, res_dict)
                # Return since this must be the last vertex
                return res_dict

        # NORMAL VERTICES WITH ID != 0 -----------------------------------------
            min_index, res_dict = self.add_vertex(vertex, diagram, model,
                            repl_dict, res_dict, min_index)

        return res_dict

    def add_vertex_id_0(self, vertex, repl_dict, res_dict):
        """Update the repl_dict and res_dict when vertex has id=0, i.e. for
        the special case of an identity vertex."""

        # For vertex (i1,i2), replace all i2 by i1
        old_num = vertex.get('legs')[1].get('number')
        new_num = vertex.get('legs')[0].get('number')
        # Be careful i1 or i2 might have been replaced themselves
        try:
            old_num = repl_dict[old_num]
        except KeyError:
            pass
        try:
            new_num = repl_dict[new_num]
        except KeyError:
            pass
        # Do the replacement
        for (ind_chain, col_str_chain) in res_dict.items():
            col_str_chain.replace_indices({old_num:new_num})

    def add_vertex(self, vertex, diagram, model,
                   repl_dict, res_dict, min_index):
        """Update repl_dict, res_dict and min_index for normal vertices.
        Returns the min_index reached and the result dictionary in a tuple."""

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
            # flip part/antipart, and replace last index by a new 
            # summed index
            if index == len(vertex.get('legs')) - 1 and \
                vertex != diagram.get('vertices')[-1]:
                part = model.get('particle_dict')[curr_pdg]
                curr_pdg = \
                    model.get('particle_dict')[curr_pdg].get_anti_pdg_code()
                repl_dict[curr_num] = min_index
                min_index = min_index - 1
            try:
                curr_num = repl_dict[curr_num]
            except KeyError:
                pass
            try:
                dict_pdg_leg[curr_pdg].append(curr_num)
            except KeyError:
                dict_pdg_leg[curr_pdg] = [curr_num]

        # Create a list of associated leg number following the same order
        list_numbers = []
        for pdg_code in list_pdg:
            list_numbers.append(dict_pdg_leg[pdg_code].pop(0))
        # ... and the associated dictionary for replacement
        match_dict = dict(enumerate(list_numbers))

        # Update the result dict using the current vertex ColorString object
        # If more than one, create different entries

        # For colorless vertices, return a copy of res_dict
        inter_color = model.get_interaction(vertex['id'])['color']
        if not inter_color:
            return (min_index, copy.copy(res_dict))
        new_res_dict = {}
        for i, col_str in \
                enumerate(inter_color):
            # Build the new element
            mod_col_str = col_str.create_copy()

            # Replace summed (negative) internal indices
            list_neg = []
            for col_obj in mod_col_str:
                list_neg.extend([ind for ind in col_obj if ind < 0])
            internal_indices_dict = {}
            # This notation is to remove duplicates
            for index in list(set(list_neg)):
                internal_indices_dict[index] = min_index
                min_index = min_index - 1
            mod_col_str.replace_indices(internal_indices_dict)

            # Replace other (positive) indices using the match_dic
            mod_col_str.replace_indices(match_dict)

            # If we are considering the first vertex, simply create
            # new entries

            if not res_dict:
                new_res_dict[tuple([i])] = mod_col_str
            #... otherwise, loop over existing elements and multiply
            # the color strings
            else:
                for ind_chain, col_str_chain in res_dict.items():
                    new_col_str_chain = col_str_chain.create_copy()
                    new_col_str_chain.product(mod_col_str)
                    new_res_dict[tuple(list(ind_chain) + [i])] = \
                        new_col_str_chain
        return (min_index, new_res_dict)


    def update_color_basis(self, colorize_dict, index):
        """Update the current color basis by adding information from 
        the colorize dictionary (produced by the colorize routine)
        associated to diagram with index index. Keep track of simplification
        results for maximal optimization."""

        # loop over possible color chains
        for col_chain, col_str in colorize_dict.items():

            # Create a canonical immutable representation of the the string
            canonical_rep, rep_dict = col_str.to_canonical()
            try:
                # If this representation has already been considered,
                # recycle the result. 
                col_fact = copy.copy(self._canonical_dict[canonical_rep])

            except KeyError:
                # If the representation is really new

                # Create and simplify a color factor for the considered chain
                col_fact = color_algebra.ColorFactor([col_str])
                col_fact = col_fact.full_simplify()

                # Save the result for further use
                canonical_col_fact = copy.copy(col_fact)
                canonical_col_fact.replace_indices(rep_dict)
                self._canonical_dict[canonical_rep] = canonical_col_fact

            else:
                # If this representation has already been considered,
                # adapt the result
                # Note that we have to replace back
                # the indices to match the initial convention. 
                col_fact.replace_indices(self._invert_dict(rep_dict))
                # Must simplify once to put traces in a canonical ordering
                col_fact = col_fact.simplify()

            # loop over color strings in the resulting color factor
            for col_str in col_fact:
                immutable_col_str = col_str.to_immutable()
                # if the color structure is already present in the present basis
                # update it
                basis_entry = (index,
                                col_chain,
                                col_str.coeff,
                                col_str.is_imaginary,
                                col_str.Nc_power)
                try:
                    self[immutable_col_str].append(basis_entry)
                except KeyError:
                    self[immutable_col_str] = [basis_entry]

    def create_color_dict_list(self, amplitude):
        """Returns a list of colorize dict for all diagrams in amplitude. Also
        update the _list_color_dict object accordingly """

        list_color_dict = []

        for diagram in amplitude.get('diagrams'):
            colorize_dict = self.colorize(diagram,
                                          amplitude.get('process').get('model'))
            list_color_dict.append(colorize_dict)

        self._list_color_dict = list_color_dict

        return list_color_dict

    def build(self, amplitude=None):
        """Build the a color basis object using information contained in
        amplitude (otherwise use info from _list_color_dict). 
        Returns a list of color """

        if amplitude:
            self.create_color_dict_list(amplitude)

        for index, color_dict in enumerate(self._list_color_dict):
            self.update_color_basis(color_dict, index)

    def __init__(self, *args):
        """Initialize a new color basis object, either empty or filled (0
        or 1 arguments). If one arguments is given, it's interpreted as 
        an amplitude."""

        if len(args) not in (0, 1):
            raise ValueError, \
                "Object ColorBasis must be initialized with 0 or 1 arguments"

        if len(args) == 1:
            if not isinstance(args[0], diagram_generation.Amplitude):
                raise TypeError, \
                        "%s is not a valid Amplitude object" % str(value)
            self.build(*args)

    def __str__(self):
        """Returns a nicely formatted string for display"""

        my_str = ""
        for k, v in self.items():
            for name, indices in k:
                my_str = my_str + name + str(indices)
            my_str = my_str + ': '
            for contrib in v:
                imag_str = ''
                if contrib[3]:
                    imag_str = 'I'
                my_str = my_str + '(diag:%i, chain:%s, coeff:%s%s, Nc:%i) ' % \
                                    (contrib[0], contrib[1], contrib[2],
                                     imag_str, contrib[4])
            my_str = my_str + '\n'
        return my_str

    def _invert_dict(self, mydict):
        """Helper method to invert dictionary dict"""

        return dict([v, k] for k, v in mydict.items())

#===============================================================================
# ColorMatrix
#===============================================================================
class ColorMatrix(dict):
    """A color matrix, i.e. a dictionary with pairs (i,j) as keys where i
    and j refer to elements of color basis objects. Values are Color Factor
    objects. Also contains two additional dictonaries, one with the fixed Nc
    representation of the matrix, and the other one with the "inverted" matrix,
    i.e. a dictionary where keys are values of the color matrix."""

    _col_basis1 = None
    _col_basis2 = None
    col_matrix_fixed_Nc = {}
    inverted_col_matrix = {}

    def __init__(self, col_basis, col_basis2=None,
                 Nc=3, Nc_power_min=None, Nc_power_max=None):
        """Initialize a color matrix with one or two color basis objects. If
        only one color basis is given, the other one is assumed to be equal.
        As options, any value of Nc and minimal/maximal power of Nc can also be 
        provided. Note that the min/max power constraint is applied
        only at the end, so that it does NOT speed up the calculation."""

        self._col_basis1 = col_basis
        if col_basis2:
            self._col_basis2 = col_basis2
            self.build_matrix(Nc, Nc_power_min, Nc_power_max)
        else:
            self._col_basis2 = col_basis
            # If the two color basis are equal, assumes the color matrix is 
            # symmetric
            self.build_matrix(Nc, Nc_power_min, Nc_power_max, is_symmetric=True)

    def build_matrix(self, Nc=3,
                     Nc_power_min=None,
                     Nc_power_max=None,
                     is_symmetric=False):
        """Create the matrix using internal color basis objects. Use the stored
        color basis objects and takes Nc and Nc_min/max parameters as __init__.
        If is_isymmetric is True, build only half of the matrix which is assumed
        to be symmetric."""

        canonical_dict = {}

        for i1, struct1 in \
                    enumerate(self._col_basis1.keys()):
            for i2, struct2 in \
                    enumerate(self._col_basis2.keys()):

                # Only scan upper right triangle if symmetric
                if is_symmetric and i2 < i1:
                    continue

                # Fix indices in struct2 knowing summed indices in struct1
                # to avoid duplicates
                new_struct2 = self.fix_summed_indices(struct1, struct2)

                # Build a canonical representation of the two immutable struct
                canonical_entry, dummy = \
                            color_algebra.ColorString().to_canonical(struct1 + \
                                                                   new_struct2)

                try:
                    # If this has already been calculated, use the result
                    result, result_fixed_Nc = canonical_dict[canonical_entry]

                except KeyError:
                    # Otherwise calculate the result
                    result, result_fixed_Nc = \
                            self.create_new_entry(struct1,
                                                  new_struct2,
                                                  Nc_power_min,
                                                  Nc_power_max,
                                                  Nc)
                    # Store both results
                    canonical_dict[canonical_entry] = (result, result_fixed_Nc)

                # Store the full result...
                self[(i1, i2)] = result
                if is_symmetric:
                    self[(i2, i1)] = result

                # the fixed Nc one ...
                self.col_matrix_fixed_Nc[(i1, i2)] = result_fixed_Nc
                if is_symmetric:
                    self.col_matrix_fixed_Nc[(i2, i1)] = result_fixed_Nc
                # and update the inverted dict
                if result_fixed_Nc in self.inverted_col_matrix.keys():
                    self.inverted_col_matrix[result_fixed_Nc].append((i1,
                                                                      i2))
                    if is_symmetric:
                        self.inverted_col_matrix[result_fixed_Nc].append((i2,
                                                                          i1))
                else:
                    self.inverted_col_matrix[result_fixed_Nc] = [(i1, i2)]
                    if is_symmetric:
                        self.inverted_col_matrix[result_fixed_Nc] = [(i2, i1)]

    def create_new_entry(self, struct1, struct2,
                         Nc_power_min, Nc_power_max, Nc):
        """ Create a new product result, and result with fixed Nc for two color
        basis entries. Implement Nc power limits."""

        # Create color string objects corresponding to color basis 
        # keys
        col_str = color_algebra.ColorString()
        col_str.from_immutable(struct1)

        col_str2 = color_algebra.ColorString()
        col_str2.from_immutable(struct2)

        # Complex conjugate the second one and multiply the two
        col_str.product(col_str2.complex_conjugate())

        # Create a color factor to store the result and simplify it
        # taking into account the limit on Nc
        col_fact = color_algebra.ColorFactor([col_str])
        result = col_fact.full_simplify()

        # Keep only terms with Nc_max >= Nc power >= Nc_min
        if Nc_power_min is not None:
            result[:] = [col_str for col_str in result \
                         if col_str.Nc_power >= Nc_power_min]
        if Nc_power_max is not None:
            result[:] = [col_str for col_str in result \
                         if col_str.Nc_power <= Nc_power_max]

        # Calculate the fixed Nc representation
        result_fixed_Nc = result.set_Nc(Nc)

        return result, result_fixed_Nc

    def __str__(self):
        """Returns a nicely formatted string with the fixed Nc representation
        of the current matrix (only the real part)"""

        mystr = '\n\t' + '\t'.join([str(i) for i in \
                                    range(len(self._col_basis2))])

        for i1 in range(len(self._col_basis1)):
            mystr = mystr + '\n' + str(i1) + '\t'
            mystr = mystr + '\t'.join(['%i/%i' % \
                        (self.col_matrix_fixed_Nc[(i1, i2)][0].numerator,
                        self.col_matrix_fixed_Nc[(i1, i2)][0].denominator) \
                        for i2 in range(len(self._col_basis2))])

        return mystr

    def get_line_denominators(self):
        """Get a list with the denominators for the different lines in
        the color matrix"""

        den_list = []
        for i1 in range(len(self._col_basis1)):
            den_list.append(self.lcmm(*[\
                        self.col_matrix_fixed_Nc[(i1, i2)][0].denominator for \
                                        i2 in range(len(self._col_basis2))]))
        return den_list

    def get_line_numerators(self, line_index, den):
        """Returns a list of numerator for line line_index, assuming a common
        denominator den."""

        return [self.col_matrix_fixed_Nc[(line_index, i2)][0].numerator * \
                den / self.col_matrix_fixed_Nc[(line_index, i2)][0].denominator \
                for i2 in range(len(self._col_basis2))]

    @classmethod
    def fix_summed_indices(self, struct1, struct2):
        """Returns a copy of the immutable Color String representation struct2 
        where summed indices are modified to avoid duplicates with those
        appearing in struct1. Assumes internal summed indices are negative."""

        # First, determines what is the smallest index appearing in struct1
        min_index = min(reduce(operator.add,
                                [list(elem[1]) for elem in struct1])) - 1
        # Second, determines the summed indices in struct2 and create a 
        # replacement dictionary
        repl_dict = {}
        list2 = reduce(operator.add,
                       [list(elem[1]) for elem in struct1])
        for summed_index in list(set([i for i in list2 \
                                      if list2.count(i) == 2])):
            repl_dict[summed_index] = min_index
            min_index -= 1

        # Three, create a new immutable struct by doing replacements in struct2
        return_list = []
        for elem in struct2:
            fix_elem = [elem[0], []]
            for index in elem[1]:
                try:
                    fix_elem[1].append(repl_dict[index])
                except:
                    fix_elem[1].append(index)
            return_list.append((elem[0], tuple(fix_elem[1])))

        return tuple(return_list)

    @staticmethod
    def lcm(a, b):
        """Return lowest common multiple."""
        return a * b // fractions.gcd(a, b)

    @staticmethod
    def lcmm(*args):
        """Return lcm of args."""
        return reduce(ColorMatrix.lcm, args)
