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

"""Classes, methods and functions required to square a QCD color string for
squared diagrams and interference terms."""

import operator

import madgraph.core.color_algebra as color_algebra

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
        provided. Be careful that the min/max power constraint is applied
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

        self.build_matrix(Nc, Nc_power_min, Nc_power_max)

    def build_matrix(self, Nc=3,
                     Nc_power_min=None,
                     Nc_power_max=None,
                     is_symmetric=False):
        """Create the matrix using internal color basis objects. Use the stored
        color basis objects and takes Nc and Nc_min/max parameters as __init__.
        If is_isymmetric is True, build only half of the matrix which is assumed
        to be symmetric."""

        canonical_dict = {}

        for i1, (struct1, contrib_list1) in \
                    enumerate(self._col_basis1.items()):
            for i2, (struct2, contrib_list2) in \
                    enumerate(self._col_basis2.items()):

                # Only scan upper right triangle if symmetric
                if is_symmetric and i2 < i1:
                    continue

                # Fix indices in struct2 knowing summed indices in struct1
                # to avoid duplicates
                new_struct2 = self.fix_summed_indices(struct1, struct2)

                # Build a canonical representation of the two immutable struct
                canonical_entry = self.to_canonical(struct1, new_struct2)

                try:
                    # If this has already been calculated, use the result
                    result = canonical_dict[canonical_entry][0]
                    result_fixed_Nc = canonical_dict[canonical_entry][1]

                except KeyError:
                    # Otherwise calculate the result

                    # Create color string objects corresponding to color basis 
                    # keys
                    col_str = color_algebra.ColorString()
                    col_str.from_immutable(struct1)

                    col_str2 = color_algebra.ColorString()
                    col_str2.from_immutable(new_struct2)

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

                    # Store both results
                    canonical_dict[canonical_entry] = (result, result_fixed_Nc)

                # Store the full result...
                self[(i1, i2)] = result
                if is_symmetric:
                    self[(i2, i1)] = result

                # the fixed Nc one ...
                self.col_matrix_fixed_Nc[(i1, i2)] = result_fixed_Nc

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

    @classmethod
    def to_canonical(self, immutable1, immutable2):
        """Returns a pair (canonical1,canonical2) where canonical corresponds
        to the canonical representation of the immutable representation (i.e.,
        first index is 1, ...)"""

        replaced_indices = {}
        curr_ind = 1
        return_list = []

        for elem in immutable1 + immutable2:
            can_elem = [elem[0], []]
            for index in elem[1]:
                try:
                    new_index = replaced_indices[index]
                except KeyError:
                    new_index = curr_ind
                    curr_ind += 1
                    replaced_indices[index] = new_index
                can_elem[1].append(new_index)
            return_list.append((can_elem[0], tuple(can_elem[1])))

        return_list.sort()

        return tuple(return_list)

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
