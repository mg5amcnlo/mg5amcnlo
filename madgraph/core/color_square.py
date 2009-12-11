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

import madgraph.core.color_algebra as color_algebra

def build_color_matrix(col_basis1, col_basis2):
    """Create a color matrix NxM starting from color_basis dictionaries of size
    N and M."""

    color_matrix = []
    for k1, v1 in col_basis1.items():
        color_line = []
        for k2, v2 in col_basis2.items():
            col_str = color_algebra.ColorString(list(k1))
            col_str2 = color_algebra.ColorString(list(k2))
            col_str.extend(col_str2.complex_conjugate())
            col_fact = color_algebra.ColorFactor([col_str])
            col_fact.simplify()
            color_line.append(col_fact)
        color_matrix.append(color_line)

    return color_matrix


