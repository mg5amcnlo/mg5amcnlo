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

import itertools

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

    def __init__(self, col_basis, col_basis2=None, Nc=3, Nc_limit=None):
        """Initialize a color matrix with one or two color basis objects. If
        only one color basis is given, the other one is assumed to be equal.
        As options, any value of Nc and minimal power of Nc can also be 
        provided. Be careful that the minimal power constraint is applied
        only at the end, so that it does NOT speed up the calculation."""

        self._col_basis1 = col_basis
        if col_basis2:
            self._col_basis2 = col_basis2
        else:
            self._col_basis2 = col_basis

        self.build_matrix(Nc, Nc_limit)

    def build_matrix(self, Nc=3, Nc_limit=None):
        """Create the matrix using internal color basis objects. Use the stored
        color basis objects and takes Nc and Nc_limit parameters as __init__."""

        for i1, (struct1, contrib_list1) in \
                    enumerate(self._col_basis1.items()):
            for i2, (struct2, contrib_list2) in \
                    enumerate(self._col_basis2.items()):

                # Create color string objects corresponding to color basis keys
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

                # Keep only terms with Nc power >= Nc_limit
                if Nc_limit is not None:
                    result[:] = [col_str for col_str in result \
                                 if col_str.Nc_power >= Nc_limit]

                # Calculate the fixed Nc representation
                result_fixed_Nc = result.set_Nc(Nc)

                # Store the full result...
                self[(i1, i2)] = result

                # the fixed Nc one ...
                self.col_matrix_fixed_Nc[(i1, i2)] = result_fixed_Nc

                # and update the inverted dict
                if result_fixed_Nc in self.inverted_col_matrix.keys():
                    self.inverted_col_matrix[result_fixed_Nc].append((i1, i2))
                else:
                    self.inverted_col_matrix[result_fixed_Nc] = [(i1, i2)]

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
