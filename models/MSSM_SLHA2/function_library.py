# This file is part of the UFO.
#
# This file contains definitions for functions that
# are extensions of the cmath library, and correspond
# either to functions that are in cmath, but inconvenient
# to access from there (e.g. z.conjugate()),
# or functions that are simply not defined.
#
#

__date__ = "22 July 2010"
__author__ = "claude.duhr@durham.ac.uk"

import cmath
from object_library import all_functions, Function

#
# shortcuts for functions from cmath
#

complexconjugate = Function(name = 'complexconjugate',
                            arguments = ('z',),
                            expression = 'z.conjugate()')


re = Function(name = 're',
              arguments = ('z',),
              expression = 'z.real')

im = Function(name = 'im',
              arguments = ('z',),
              expression = 'z.imag')

# New functions (trigonometric)

sec = Function(name = 'sec',
             arguments = ('z',),
             expression = '1./cmath.cos(z)')

asec = Function(name = 'asec',
             arguments = ('z',),
             expression = 'cmath.acos(1./z)')

csc = Function(name = 'csc',
             arguments = ('z',),
             expression = '1./cmath.sin(z)')

acsc = Function(name = 'acsc',
             arguments = ('z',),
             expression = 'cmath.asin(1./z)')




