# This file is part of the UFO.
#
# This file contains definitions for functions that
# are extensions of the cmath library, and correspond
# either to functions that are in cmath, but inconvenient
# to access from there (e.g. z.conjugate()),
# or functions that are simply not defined.
#
#

__date__ = "5 June 2010"
__author__ = "claude.duhr@durham.ac.uk"

from cmath import cos, sin, acos, asin

#
# shortcuts for functions from cmath
#

def complexconjugate(z):
	"""Returns z.conjugate()"""
	return z.conjugate()


def re(z):
        """Returns z.re()"""
        return z.real

def im(z):
        """Returns z.imag"""
        return z.imag


# New functions (trigonometric)

def sec(z):
        "Returns the secant of the complex number z."""
        return 1./cos(z)

def asec(z):
        "Returns the arcsecant of the complex number z."""
        return acos(1./z)

def csc(z):
        "Returns the cosecant of the complex number z."""
        return 1./sin(z)

def acsc(z):
        "Returns the arccosecant of the complex number z."""
        return asin(1./z)




