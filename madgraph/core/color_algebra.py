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

"""Classes and methods required for all calculations related 
to QCD color algebra."""

import copy
import fractions
import itertools

#===============================================================================
# ColorObject
#===============================================================================
class ColorObject(list):
    """Parent class for all color objects like T, Tr, f, d, ... By default,
    implement a list of color indices (with get and set) and template for 
    simplify and pair_simplify. Any new color object MUST inherit
    from this class!"""

    def __init__(self, *args):
        """Initialize a color object. Any number of argument can be used, all
        of them being considered as color indices."""

        list.__init__(self)
        self.set_indices(*args)

    def append_index(self, index):
        """Append an index at the end of the current ColorObject"""
        # Check if the input value is an integer
        if not isinstance(index, int):
            raise TypeError, \
                "Object %s is not a valid integer for color index" % str(index)
        self.append(index)

    def set_indices(self, *args):
        """Use arguments (any number) to set  the current indices. Previous
        indices are lost."""

        # Remove all existing elements
        del self[:]
        # Append new ones
        for index in args:
            self.append_index(index)

    def get_indices(self):
        """Returns a list of indices."""

        return list(self)

    def __str__(self):
        """Returns a standard string representation. Can be overwritten for
        each specific child."""

        return '%s(%s)' % (self.__class__.__name__,
                           ','.join([str(i) for i in self]))

    def simplify(self):
        """Called to simplify the current object. Default behavior is to return
        None, but should be overwritten for each child. Should return a 
        ColorFactor containing the ColorStrings corresponding to the 
        simplification"""

        return None

    def pair_simplify(self, col_obj):
        """Called to simplify a pair of (multiplied) ColorObject. 
        Default behavior is to return None, but should be overwritten 
        for each child. Should return a ColorFactor containing the ColorStrings 
        corresponding to the simplification"""

        return None

#===============================================================================
# Tr
#===============================================================================
class Tr(ColorObject):
    """The trace color object"""

    def simplify(self):
        """Implement simple trace simplifications and cyclicity, and
        Tr(a,x,b,x,c) = 1/2(Tr(a,c)Tr(b)-1/Nc Tr(a,b,c))"""

        # Tr(a)=0
        if len(self) == 1:
            col_str = ColorString()
            col_str.set_coeff(fractions.Fraction(0, 1))
            return ColorFactor(col_str)
        #Tr()=Nc
        if len(self) == 0:
            col_str = ColorString()
            col_str.set_Nc_power(1)
            return ColorFactor(col_str)
        #Always order starting from smallest index
        if self[0] != min(self):
            pos = self.get_indices().index(min(self))
            new = self.get_indices()[pos:] + self.get_indices()[:pos]
            return ColorFactor(ColorString(Tr(*new)))

        # Tr(a,x,b,x,c) = 1/2(Tr(a,c)Tr(b)-1/Nc Tr(a,b,c))
        for i1, index1 in enumerate(self.get_indices()):
            for i2, index2 in enumerate(self.get_indices()[i1 + 1:]):
                if index1 == index2:
                    a = self.get_indices()[:i1]
                    b = self.get_indices()[i1 + 1:i1 + i2 + 1]
                    c = self.get_indices()[i1 + i2 + 2:]
                    col_str1 = ColorString(Tr(*(a + c)), Tr(*b))
                    col_str2 = ColorString(Tr(*(a + b + c)))
                    col_str1.set_coeff(fractions.Fraction(1, 2))
                    col_str2.set_coeff(fractions.Fraction(-1, 2))
                    col_str2.set_Nc_power(-1)
                    return ColorFactor(col_str1, col_str2)

        return None

    def pair_simplify(self, col_obj):
        """Implement Tr product simplification: 
        Tr(a,x,b)Tr(c,x,d) = 1/2(Tr(a,d,c,b)-1/Nc Tr(a,b)Tr(c,d)) and
        Tr(a,x,b)T(c,x,d,i,j) = 1/2(T(c,b,a,d,i,j)-1/Nc Tr(a,b)T(c,d,i,j))"""

        if isinstance(col_obj, Tr) or isinstance(col_obj, T):
            for i1, index1 in enumerate(self.get_indices()):
                for i2, index2 in enumerate(col_obj.get_indices()):
                    if index1 == index2:
                        a = self.get_indices()[:i1]
                        b = self.get_indices()[i1 + 1:]
                        c = col_obj.get_indices()[:i2]
                        d = col_obj.get_indices()[i2 + 1:]
                        if isinstance(col_obj, Tr):
                            col_str1 = ColorString(Tr(*(a + d + c + b)))
                        else:
                            ij = col_obj.get_ij()
                            col_str1 = ColorString(T(*(a + d + c + b + ij)))
                        if isinstance(col_obj, Tr):
                            col_str2 = ColorString(Tr(*(a + b)), Tr(*(c + d)))
                        else:
                            col_str2 = ColorString(Tr(*(a + b)),
                                                   T(*(c + d) + ij))
                        col_str1.set_coeff(fractions.Fraction(1, 2))
                        col_str2.set_coeff(fractions.Fraction(-1, 2))
                        col_str2.set_Nc_power(-1)
                        return ColorFactor(col_str1, col_str2)

        return None

#===============================================================================
# T
#===============================================================================
class T(ColorObject):
    """The T color object. Implement two additional indices i and j"""

    _ij = [0, 0]

    def __init__(self, *args):
        """Initialize a T object. Any number of argument can be used, all
        of them being considered as color indices except the two last one."""

        list.__init__(self)
        if len(args) < 2:
            raise ValueError, \
                "T objects have at least two indices!"

        self.set_ij(*args[-2:])
        self.set_indices(*args[:-2])

    def set_ij(self, i, j):
        """Set the two last indices"""
        if not isinstance(i, int) or not isinstance(j, int):
            raise TypeError, \
                "Object %s is not a valid integer for index i,j" % str(index)
        self._ij = [i, j]

    def get_ij(self):
        """Returns the two last indices"""
        return self._ij

    def __str__(self):
        """Returns a T string representation."""

        return '%s(%s;%s)' % (self.__class__.__name__,
                           ','.join([str(i) for i in self]),
                           ','.join([str(i) for i in self._ij]))

    def __eq__(self, other):
        """Compare deeply!"""
        return self.get_ij() == other.get_ij() and \
               self.get_indices() == other.get_indices()

    def simplify(self):
        """Implement T(a,b,c,...,i,i) = Tr(a,b,c,...) and
        T(a,x,b,x,c,i,j) = 1/2(T(a,c,i,j)Tr(b)-1/Nc T(a,b,c,i,j))"""

        ij = self.get_ij()

        # T(a,b,c,...,i,i) = Tr(a,b,c,...)
        if ij[0] == ij[1]:
            return ColorFactor(ColorString(Tr(*self.get_indices())))

        # T(a,x,b,x,c,i,j) = 1/2(T(a,c,i,j)Tr(b)-1/Nc T(a,b,c,i,j))
        for i1, index1 in enumerate(self.get_indices()):
            for i2, index2 in enumerate(self.get_indices()[i1 + 1:]):
                if index1 == index2:
                    a = self.get_indices()[:i1]
                    b = self.get_indices()[i1 + 1:i1 + i2 + 1]
                    c = self.get_indices()[i1 + i2 + 2:]
                    col_str1 = ColorString(T(*(a + c + ij)), Tr(*b))
                    col_str2 = ColorString(T(*(a + b + c + ij)))
                    col_str1.set_coeff(fractions.Fraction(1, 2))
                    col_str2.set_coeff(fractions.Fraction(-1, 2))
                    col_str2.set_Nc_power(-1)
                    return ColorFactor(col_str1, col_str2)

        return None

    def pair_simplify(self, col_obj, simplify_T_product=True):
        """Implement T(a,...,i,j)T(b,...,j,k) = T(a,...,b,...,i,k)
        and T(a,x,b,i,j)T(c,x,d,k,l) = 1/2(T(a,d,i,l)T(c,b,k,j)    
                                        -1/Nc T(a,b,i,j)T(c,d,k,l))
        but only if the simplify_T_product tag is True."""

        if isinstance(col_obj, T):
            ij1 = self.get_ij()
            ij2 = col_obj.get_ij()

            if ij1[1] == ij2[0]:
                return ColorFactor(ColorString(T(*(self.get_indices() + \
                                                   col_obj.get_indices() + \
                                                   [ij1[0], ij2[1]]))))
            if simplify_T_product:
                for i1, index1 in enumerate(self.get_indices()):
                    for i2, index2 in enumerate(col_obj.get_indices()):
                        if index1 == index2:
                            a = self.get_indices()[:i1]
                            b = self.get_indices()[i1 + 1:]
                            c = col_obj.get_indices()[:i2]
                            d = col_obj.get_indices()[i2 + 1:]
                            col_str1 = ColorString(T(*(a + d + \
                                                       [ij1[0], ij2[1]])),
                                                   T(*(c + b + \
                                                       [ij2[0], ij1[1]])))
                            col_str2 = ColorString(T(*(a + b + \
                                                       [ij1[0], ij1[1]])),
                                                   T(*(c + d + \
                                                       [ij2[0], ij2[1]])))
                            col_str1.set_coeff(fractions.Fraction(1, 2))
                            col_str2.set_coeff(fractions.Fraction(-1, 2))
                            col_str2.set_Nc_power(-1)
                            return ColorFactor(col_str1, col_str2)

#===============================================================================
# f
#===============================================================================
class f(ColorObject):
    """The f color object"""

    def set_indices(self, *args):
        """Overwrite default to check that there are only 3 indices"""
        if len(args) != 3:
            raise ValueError, \
                "F objects should have exactly 3 color indices!"
        # Remove all existing elements
        del self[:]
        # Append new ones
        for index in args:
            self.append_index(index)

    def simplify(self):
        """Implement only the replacement rule 
        f(a,b,c)=-2ITr(a,b,c)+2ITr(c,b,a)"""

        indices = self.get_indices()
        col_str1 = ColorString(Tr(*indices))
        indices.reverse()
        col_str2 = ColorString(Tr(*indices))

        col_str1.set_coeff(fractions.Fraction(-2, 1))
        col_str2.set_coeff(fractions.Fraction(2, 1))

        col_str1.set_is_imaginary(True)
        col_str2.set_is_imaginary(True)

        return ColorFactor(col_str1, col_str2)

#===============================================================================
# d
#===============================================================================
class d(f):
    """The d color object"""

    def simplify(self):
        """Implement only the replacement rule 
        d(a,b,c)=2Tr(a,b,c)+2Tr(c,b,a)"""

        indices = self.get_indices()
        col_str1 = ColorString(Tr(*indices))
        indices.reverse()
        col_str2 = ColorString(Tr(*indices))

        col_str1.set_coeff(fractions.Fraction(2, 1))
        col_str2.set_coeff(fractions.Fraction(2, 1))

        return ColorFactor(col_str1, col_str2)

#===============================================================================
# ColorString
#===============================================================================
class ColorString(list):
    """A list of ColorObjects with an implicit multiplication between,
    together with a Fraction coefficient and a tag
    to indicate if the coefficient is real or imaginary. ColorStrings can be
    simplified, by simplifying their elements."""

    _coeff = fractions.Fraction(1, 1)
    _is_imaginary = False
    _Nc_power = 0

    def __init__(self, *args):
        """Creates a new color string starting from one or several color
        objects."""

        list.__init__(self)
        self.set_color_objects(*args)

    def append_color_object(self, col_obj):
        """Append a color object at the end of the current ColorString"""
        # Check if the input value is a child of ColorObject
        if not isinstance(col_obj, ColorObject):
            raise TypeError, \
                "Object %s is not a valid ColorObject child" % \
                                                           str(col_obj)
        self.append(col_obj)

    def get_color_objects(self):
        """Returns a list of color objects."""

        return list(self)

    def set_color_objects(self, *args):
        """Use arguments (any number) to set  the current color objects.
        Previous color objects are lost."""

        # Remove all existing elements
        del self[:]
        # Append new ones
        for col_obj in args:
            self.append_color_object(col_obj)

    def __str__(self):
        """Returns a standard string representation based on color object
        representations"""

        coeff_str = str(self._coeff)
        if self._is_imaginary:
            coeff_str += ' I'
        if self._Nc_power > 0:
            coeff_str += ' Nc^%i' % self._Nc_power
        elif self._Nc_power < 0:
            coeff_str += ' 1/Nc^%i' % abs(self._Nc_power)
        return '%s %s' % (coeff_str,
                         ' '.join([str(col_obj) for col_obj in self]))

    def set_coeff(self, frac):
        """Set the coefficient of the ColorString"""

        if not isinstance(frac, fractions.Fraction):
            raise TypeError, \
                "Object %s is not a valid fraction for coefficient" % \
                                                                str(frac)
        self._coeff = frac

    def get_coeff(self):
        """Get the coefficient of the ColorString"""
        return self._coeff

    def set_Nc_power(self, power):
        """Set the Nc power of the ColorString"""

        if not isinstance(power, int):
            raise TypeError, \
                "Object %s is not a valid int for Nc power" % str(power)
        self._Nc_power = power

    def get_Nc_power(self):
        """Get the Nc power of the ColorString"""
        return self._Nc_power

    def set_is_imaginary(self, is_imaginary):
        """Set the is_imaginary flag of the ColorString"""

        if not isinstance(is_imaginary, bool):
            raise TypeError, \
                "Object %s is not a valid boolean for is_imaginary flag" % \
                                                            str(is_imaginary)
        self._is_imaginary = is_imaginary

    def is_imaginary(self):
        """Is the ColorString imaginary ?"""
        return self._is_imaginary

    def __eq__(self, other):
        """Compare deeply!"""

        return self.get_coeff() == other.get_coeff() and \
               self.get_Nc_power() == other.get_Nc_power() and \
               self.is_imaginary() == other.is_imaginary() and \
               self.get_color_objects() == other.get_color_objects()

    def product(self, other):
        """Multiply two ColorStrings and returns the result."""

        res_coeff = self.get_coeff() * other.get_coeff()

        res_Nc_power = self.get_Nc_power() + other.get_Nc_power()

        res_imaginary = False
        if all([self.is_imaginary(), other.is_imaginary()]):
            res_coeff = -res_coeff
        elif any([self.is_imaginary(), other.is_imaginary()]):
            res_imaginary = True

        res_col_str = ColorString(*(self.get_color_objects() + \
                                  other.get_color_objects()))

        res_col_str.set_coeff(res_coeff)
        res_col_str.set_Nc_power(res_Nc_power)
        res_col_str.set_is_imaginary(res_imaginary)

        return res_col_str

    def simplify(self):
        """Simplify the current ColorString by applying simplify rules on
        each element and building a new ColorFactor to return if necessary"""

        # First try to simplify single elements
        for i, col_obj in enumerate(self):
            res = col_obj.simplify()
            if res:
                res_col_factor = ColorFactor()
                first_col_str = copy.deepcopy(self)
                del first_col_str[i]
                for second_col_str in res:
                    prod = first_col_str.product(second_col_str)
                    prod.sort()
                    res_col_factor.append_color_string(prod)
                return res_col_factor

        # Second, try to simplify pairs
        for i1, col_obj1 in enumerate(self.get_color_objects()):
            for i2, col_obj2 in enumerate(self.get_color_objects()[i1 + 1:]):
                res = col_obj1.pair_simplify(col_obj2)
                # Try both pairing
                if not res:
                    res = col_obj2.pair_simplify(col_obj1)
                if res:
                    res_col_factor = ColorFactor()
                    first_col_str = copy.deepcopy(self)
                    del first_col_str[i1]
                    del first_col_str[i1 + i2]
                    for second_col_str in res:
                        prod = first_col_str.product(second_col_str)
                        prod.sort()
                        res_col_factor.append_color_string(prod)
                    return res_col_factor

        return None

    def is_similar(self, other):
        """Returns true if two color strings differ only by their coefficients,
        False otherwise. Only two similar strings can be added."""

        return self.get_Nc_power() == other.get_Nc_power() and \
               self.is_imaginary() == other.is_imaginary() and \
               self.get_color_objects() == other.get_color_objects()

    def add(self, other):
        """Returns the sum of two strings, i.e. a copy of the first one with
        the two coefficients added. ONLY USE WITH SIMILAR STRINGS!"""

        col_str = copy.deepcopy(self)
        col_str.set_coeff(self.get_coeff() + other.get_coeff())

        return col_str
#===============================================================================
# ColorFactor
#===============================================================================
class ColorFactor(list):
    """ColorFactor objects are list of ColorString with an implicit summation.
    They can be simplified by simplifying all their elements."""

    def __init__(self, *args):
        """Creates a new color factor starting from one or several color
        strings."""

        list.__init__(self)
        self.set_color_strings(*args)

    def append_color_string(self, col_str):
        """Append a color string at the end of the current ColorFactor"""
        # Check if the input value is a child of ColorString
        if not isinstance(col_str, ColorString):
            raise TypeError, \
                "Object %s is not a valid ColorString " % str(col_str)
        self.append(col_str)

    def extend(self, col_fact):
        """Extend with a ColorFactor (or a list of ColorStrings)
        at the end of the current ColorFactor"""

        for col_str in col_fact:
            self.append_color_string(col_str)

    def set_color_strings(self, *args):
        """Use arguments (any number) to set  the current color strings.
        Previous color strings are lost."""

        # Remove all existing elements
        del self[:]
        # Append new ones
        for col_str in args:
            self.append_color_string(col_str)

    def __str__(self):
        """Returns a nice string for print"""

        return '+'.join(['(%s)' % str(col_str) for col_str in self])

    def simplify(self):
        """Returns a new color factor where each color string has been
        simplified once and similar strings have been added."""

        new_col_factor = ColorFactor()

        # Simplify
        for col_str in self:
            res = col_str.simplify()
            if res:
                new_col_factor.extend([col_str for col_str in res \
                                       if col_str.get_coeff() != 0])
            elif col_str.get_coeff() != 0:
                new_col_factor.append_color_string(copy.deepcopy(col_str))

        for i1, col_str1 in enumerate(new_col_factor):
            for i2, col_str2 in enumerate(new_col_factor[i1 + 1:]):
                if col_str1.is_similar(col_str2):
                    new_col_factor[i1] = col_str1.add(col_str2)
                    new_col_factor[i1 + i2 + 1].set_coeff(fractions.Fraction(0, 1))

        return new_col_factor

    def full_simplify(self):
        """Simplify the current color factor until the result is stable"""

        while(True):
            ref = copy.deepcopy(self)

            self = self.simplify()
            if self == ref:
                print self
                break





