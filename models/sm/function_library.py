# This file is part of the UFO.
#
# This file contains definitions for functions that
# are extensions of the cmath library, and correspond
# either to functions that are in cmath, but inconvenient
# to access from there (e.g. z.conjugate()),
# or functions that are simply not defined.
#
#
from __future__ import absolute_import
__date__ = "22 July 2010"
__author__ = "claude.duhr@durham.ac.uk"

from cmath import cos, sin, acos, asin
from .object_library import all_functions, Function

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
             expression = '1./cos(z)')

asec = Function(name = 'asec',
             arguments = ('z',),
             expression = 'acos(1./z)')

csc = Function(name = 'csc',
             arguments = ('z',),
             expression = '1./sin(z)')

acsc = Function(name = 'acsc',
             arguments = ('z',),
             expression = 'asin(1./z)')

