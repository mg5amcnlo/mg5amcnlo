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

"""Classes and methods required for all calculations related to SU(N) color 
algebra."""

import array
import copy
import fractions
import itertools

#===============================================================================
# ColorObject
#===============================================================================
class ColorObject(array.array):
    """Parent class for all color objects like T, Tr, f, d, ... Any new color 
    object MUST inherit from this class!"""

    def __new__(cls, *args):
        """Create a new ColorObject, assuming an integer array"""
        return super(ColorObject, cls).__new__(cls, 'i', args)

    def __reduce__(self):
        """Special method needed to pickle color objects correctly"""
        return (self.__class__, tuple([i for i in self]))

    def __str__(self):
        """Returns a standard string representation."""

        return '%s(%s)' % (self.__class__.__name__,
                           ','.join([str(i) for i in self]))

    __repr__ = __str__

    def simplify(self):
        """Simplification rules, to be overwritten for each new color object!
        Should return a color factor or None if no simplification is possible"""
        return None

    def pair_simplify(self, other):
        """Pair simplification rules, to be overwritten for each new color 
        object! Should return a color factor or None if no simplification 
        is possible"""
        return None

    def complex_conjugate(self):
        """Complex conjugation. By default, the ordering of color index is
        reversed. Can be overwritten for specific color objects like T,..."""

        self.reverse()

    def replace_indices(self, repl_dict):
        """Replace current indices following the rules listed in the replacement
        dictionary written as {old_index:new_index,...}. Deals correctly with
        the replacement by allowing only one single replacement."""

        for i, index in enumerate(self):
            try:
                self[i] = repl_dict[index]
            except KeyError:
                continue

    def create_copy(self):
        """Return a real copy of the current object."""
        return globals()[self.__class__.__name__](*self)

    __copy__ = create_copy


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
            col_str.coeff = fractions.Fraction(0, 1)
            return ColorFactor([col_str])

        # Tr()=Nc
        if len(self) == 0:
            col_str = ColorString()
            col_str.Nc_power = 1
            return ColorFactor([col_str])

        # Always order starting from smallest index
        if self[0] != min(self):
            pos = self.index(min(self))
            new = self[pos:] + self[:pos]
            return ColorFactor([ColorString([Tr(*new)])])

        # Tr(a,x,b,x,c) = 1/2(Tr(a,c)Tr(b)-1/Nc Tr(a,b,c))
        for i1, index1 in enumerate(self):
            for i2, index2 in enumerate(self[i1 + 1:]):
                if index1 == index2:
                    a = self[:i1]
                    b = self[i1 + 1:i1 + i2 + 1]
                    c = self[i1 + i2 + 2:]
                    col_str1 = ColorString([Tr(*(a + c)), Tr(*b)])
                    col_str2 = ColorString([Tr(*(a + b + c))])
                    col_str1.coeff = fractions.Fraction(1, 2)
                    col_str2.coeff = fractions.Fraction(-1, 2)
                    col_str2.Nc_power = -1
                    return ColorFactor([col_str1, col_str2])

        return None

    def pair_simplify(self, col_obj):
        """Implement Tr product simplification: 
        Tr(a,x,b)Tr(c,x,d) = 1/2(Tr(a,d,c,b)-1/Nc Tr(a,b)Tr(c,d)) and
        Tr(a,x,b)T(c,x,d,i,j) = 1/2(T(c,b,a,d,i,j)-1/Nc Tr(a,b)T(c,d,i,j))"""

        # Tr(a,x,b)Tr(c,x,d) = 1/2(Tr(a,d,c,b)-1/Nc Tr(a,b)Tr(c,d))
        if isinstance(col_obj, Tr):
            for i1, index1 in enumerate(self):
                for i2, index2 in enumerate(col_obj):
                    if index1 == index2:
                        a = self[:i1]
                        b = self[i1 + 1:]
                        c = col_obj[:i2]
                        d = col_obj[i2 + 1:]
                        col_str1 = ColorString([Tr(*(a + d + c + b))])
                        col_str2 = ColorString([Tr(*(a + b)), Tr(*(c + d))])
                        col_str1.coeff = fractions.Fraction(1, 2)
                        col_str2.coeff = fractions.Fraction(-1, 2)
                        col_str2.Nc_power = -1
                        return ColorFactor([col_str1, col_str2])

        # Tr(a,x,b)T(c,x,d,i,j) = 1/2(T(c,b,a,d,i,j)-1/Nc Tr(a,b)T(c,d,i,j))
        if isinstance(col_obj, T):
            for i1, index1 in enumerate(self):
                for i2, index2 in enumerate(col_obj[:-2]):
                    if index1 == index2:
                        a = self[:i1]
                        b = self[i1 + 1:]
                        c = col_obj[:i2]
                        d = col_obj[i2 + 1:-2]
                        ij = col_obj[-2:]
                        col_str1 = ColorString([T(*(c + b + a + d + ij))])
                        col_str2 = ColorString([Tr(*(a + b)), T(*(c + d) + ij)])
                        col_str1.coeff = fractions.Fraction(1, 2)
                        col_str2.coeff = fractions.Fraction(-1, 2)
                        col_str2.Nc_power = -1
                        return ColorFactor([col_str1, col_str2])

        return None

#===============================================================================
# T
#===============================================================================
class T(ColorObject):
    """The T color object. Last two indices have a special meaning"""

    def __init__(self, *args):
        """Check for at least two indices"""

        super(T, self).__init__()
        if len(args) < 2:
            raise ValueError, \
                "T objects must have at least two indices!"

    def simplify(self):
        """Implement T(a,b,c,...,i,i) = Tr(a,b,c,...) and
        T(a,x,b,x,c,i,j) = 1/2(T(a,c,i,j)Tr(b)-1/Nc T(a,b,c,i,j))"""

        # T(a,b,c,...,i,i) = Tr(a,b,c,...)
        if self[-2] == self[-1]:
            return ColorFactor([ColorString([Tr(*self[:-2])])])

        # T(a,x,b,x,c,i,j) = 1/2(T(a,c,i,j)Tr(b)-1/Nc T(a,b,c,i,j))
        for i1, index1 in enumerate(self[:-2]):
            for i2, index2 in enumerate(self[i1 + 1:-2]):
                if index1 == index2:
                    a = self[:i1]
                    b = self[i1 + 1:i1 + i2 + 1]
                    c = self[i1 + i2 + 2:-2]
                    ij = self[-2:]
                    col_str1 = ColorString([T(*(a + c + ij)), Tr(*b)])
                    col_str2 = ColorString([T(*(a + b + c + ij))])
                    col_str1.coeff = fractions.Fraction(1, 2)
                    col_str2.coeff = fractions.Fraction(-1, 2)
                    col_str2.Nc_power = -1
                    return ColorFactor([col_str1, col_str2])

        return None

    def pair_simplify(self, col_obj):
        """Implement T(a,...,i,j)T(b,...,j,k) = T(a,...,b,...,i,k)
        and T(a,x,b,i,j)T(c,x,d,k,l) = 1/2(T(a,d,i,l)T(c,b,k,j)    
                                        -1/Nc T(a,b,i,j)T(c,d,k,l))."""

        if isinstance(col_obj, T):
            ij1 = self[-2:]
            ij2 = col_obj[-2:]

            # T(a,...,i,j)T(b,...,j,k) = T(a,...,b,...,i,k)
            if ij1[1] == ij2[0]:
                return ColorFactor([ColorString([T(*(self[:-2] + \
                                                   col_obj[:-2] + \
                                                   array.array('i', [ij1[0],
                                                               ij2[1]])))])])
            # T(a,x,b,i,j)T(c,x,d,k,l) = 1/2(T(a,d,i,l)T(c,b,k,j)    
            #                          -1/Nc T(a,b,i,j)T(c,d,k,l))
            for i1, index1 in enumerate(self[:-2]):
                for i2, index2 in enumerate(col_obj[:-2]):
                    if index1 == index2:
                        a = self[:i1]
                        b = self[i1 + 1:-2]
                        c = col_obj[:i2]
                        d = col_obj[i2 + 1:-2]
                        col_str1 = ColorString([T(*(a + d + \
                                                   array.array('i',
                                                    [ij1[0], ij2[1]]))),
                                               T(*(c + b + \
                                                   array.array('i',
                                                    [ij2[0], ij1[1]])))])
                        col_str2 = ColorString([T(*(a + b + \
                                                   array.array('i',
                                                    [ij1[0], ij1[1]]))),
                                               T(*(c + d + \
                                                   array.array('i',
                                                    [ij2[0], ij2[1]])))])
                        col_str1.coeff = fractions.Fraction(1, 2)
                        col_str2.coeff = fractions.Fraction(-1, 2)
                        col_str2.Nc_power = -1
                        return ColorFactor([col_str1, col_str2])

    def complex_conjugate(self):
        """Complex conjugation. Overwritten here because the two last indices
        should be treated differently"""

        # T(a,b,c,i,j)* = T(c,b,a,j,i)
        l1 = self[:-2]
        l1.reverse()
        l2 = self[-2:]
        l2.reverse()
        self[:] = l1 + l2

#===============================================================================
# f
#===============================================================================
class f(ColorObject):
    """The f color object"""

    def __init__(self, *args):
        """Ensure f and d objects have strictly 3 indices"""

        super(f, self).__init__()
        if len(args) != 3:
            raise ValueError, \
                "f and d objects must have three indices!"

    def simplify(self):
        """Implement only the replacement rule 
        f(a,b,c)=-2ITr(a,b,c)+2ITr(c,b,a)"""

        indices = self[:]
        col_str1 = ColorString([Tr(*indices)])
        indices.reverse()
        col_str2 = ColorString([Tr(*indices)])

        col_str1.coeff = fractions.Fraction(-2, 1)
        col_str2.coeff = fractions.Fraction(2, 1)

        col_str1.is_imaginary = True
        col_str2.is_imaginary = True

        return ColorFactor([col_str1, col_str2])

#===============================================================================
# d
#===============================================================================
class d(f):
    """The d color object"""

    def simplify(self):
        """Implement only the replacement rule 
        d(a,b,c)=2Tr(a,b,c)+2Tr(c,b,a)"""

        indices = self[:]
        col_str1 = ColorString([Tr(*indices)])
        indices.reverse()
        col_str2 = ColorString([Tr(*indices)])

        col_str1.coeff = fractions.Fraction(2, 1)
        col_str2.coeff = fractions.Fraction(2, 1)

        return ColorFactor([col_str1, col_str2])

#===============================================================================
# epsilon
#===============================================================================
class Epsilon(ColorObject):
    """Espilon_ijk color object for three triplets"""

    def __init__(self, *args):
        """Ensure e_ijk objects have strictly 3 indices"""

        super(Epsilon, self).__init__()
        if len(args) != 3:
            raise ValueError, \
                "Epsilon objects must have three indices!"

    def pair_simplify(self, col_obj):
        """Implement e_ijk ae_klm = T(i,l)T(j,m) - T(i,m)T(j,l)"""

        if isinstance(col_obj, AEpsilon):
            eps_indices = self[:]
            aeps_indices = col_obj[:]
            if
            col_str1 = ColorString[T(self[0], col_obj[1]),
                                   T(self[1], col_obj[2])]
            col_str1 = ColorString[T(self[0], col_obj[1]),
                                   T(self[1], col_obj[0])]

#===============================================================================
# ColorString
#===============================================================================
class ColorString(list):
    """A list of ColorObjects with an implicit multiplication between,
    together with a Fraction coefficient and a tag
    to indicate if the coefficient is real or imaginary. ColorStrings can be
    simplified, by simplifying their elements."""

    coeff = fractions.Fraction(1, 1)
    is_imaginary = False
    Nc_power = 0

    def __init__(self, init_list=[],
                 coeff=fractions.Fraction(1, 1),
                 is_imaginary=False, Nc_power=0):
        """Overrides norm list constructor to implement easy modification
        of coeff, is_imaginary and Nc_power"""

        if init_list:
            self.extend(init_list)
        self.coeff = coeff
        self.is_imaginary = is_imaginary
        self.Nc_power = Nc_power

    def __str__(self):
        """Returns a standard string representation based on color object
        representations"""

        coeff_str = str(self.coeff)
        if self.is_imaginary:
            coeff_str += ' I'
        if self.Nc_power > 0:
            coeff_str += ' Nc^%i' % self.Nc_power
        elif self.Nc_power < 0:
            coeff_str += ' 1/Nc^%i' % abs(self.Nc_power)
        return '%s %s' % (coeff_str,
                         ' '.join([str(col_obj) for col_obj in self]))

    __repr__ = __str__

    def product(self, other):
        """Multiply self with other."""

        self.coeff = self.coeff * other.coeff

        self.Nc_power = self.Nc_power + other.Nc_power

        # Complex algebra
        if self.is_imaginary and other.is_imaginary:
            self.is_imaginary = False
            self.coeff = -self.coeff
        elif self.is_imaginary or other.is_imaginary:
            self.is_imaginary = True

        self.extend(other)

    def simplify(self):
        """Simplify the current ColorString by applying simplify rules on
        each element and building a new ColorFactor to return if necessary"""

        # First, try sto simplify element by element
        for i1, col_obj1 in enumerate(self):
            res = col_obj1.simplify()
            # If a simplification possibility is found...
            if res:
                # Create a color factor to store the answer...
                res_col_factor = ColorFactor()
                # Obtained my multiplying the initial string minus the color
                # object to simplify with all color strings in the result
                for second_col_str in res:
                    first_col_str = copy.copy(self)
                    del first_col_str[i1]
                    first_col_str.product(second_col_str)
                    # This sort is necessary to ensure ordering of ColorObjects
                    # remains the same for comparison
                    first_col_str.sort()
                    res_col_factor.append(first_col_str)
                return res_col_factor

        # Second, try to simplify pairs
        for i1, col_obj1 in enumerate(self):

            for i2, col_obj2 in enumerate(self[i1 + 1:]):
                res = col_obj1.pair_simplify(col_obj2)
                # Try both pairing
                if not res:
                    res = col_obj2.pair_simplify(col_obj1)
                if res:
                    res_col_factor = ColorFactor()
                    for second_col_str in res:
                        first_col_str = copy.copy(self)
                        del first_col_str[i1]
                        del first_col_str[i1 + i2]
                        first_col_str.product(second_col_str)
                        first_col_str.sort()
                        res_col_factor.append(first_col_str)
                    return res_col_factor

        return None

    def add(self, other):
        """Add string other to current string. ONLY USE WITH SIMILAR STRINGS!"""

        self.coeff = self.coeff + other.coeff

    def complex_conjugate(self):
        """Returns the complex conjugate of the current color string"""

        compl_conj_str = copy.copy(self)
        for col_obj in compl_conj_str:
            col_obj.complex_conjugate()
        if compl_conj_str.is_imaginary:
            compl_conj_str.coeff = -compl_conj_str.coeff

        return compl_conj_str

    def to_immutable(self):
        """Returns an immutable object summarizing the color structure of the
        current color string. Format is ((name1,indices1),...) where name is the
        class name of the color object and indices a tuple corresponding to its
        indices. An immutable object, in Python, is built on tuples, strings and
        numbers, i.e. objects which cannot be modified. Their crucial property
        is that they can be used as dictionary keys!"""

        ret_list = [(col_obj.__class__.__name__, tuple(col_obj)) \
                        for col_obj in self]
        ret_list.sort()
        return tuple(ret_list)

    def from_immutable(self, immutable_rep):
        """Fill the current object with Color Objects created using an immutable
        representation."""

        del self[:]

        for col_tuple in immutable_rep:
            self.append(globals()[col_tuple[0]](*col_tuple[1]))

    def replace_indices(self, repl_dict):
        """Replace current indices following the rules listed in the replacement
        dictionary written as {old_index:new_index,...}, does that for ALL 
        color objects."""

        map(lambda col_obj: col_obj.replace_indices(repl_dict), self)

    def create_copy(self):
        """Returns a real copy of self, non trivial because bug in 
        copy.deepcopy"""

        res = ColorString()
        for col_obj in self:
            res.append(col_obj.create_copy())
        res.coeff = self.coeff
        res.is_imaginary = self.is_imaginary
        res.Nc_power = self.Nc_power

        return res

    __copy__ = create_copy

    def set_Nc(self, Nc=3):
        """Returns a tuple, with the first entry being the string coefficient 
        with Nc replaced (by default by 3), and the second one being True
        or False if the coefficient is imaginary or not. Raise an error if there
        are still non trivial color objects."""

        if self:
            raise ValueError, \
                "String %s cannot be simplified to a number!" % str(self)

        if self.Nc_power >= 0:
            return (self.coeff * fractions.Fraction(\
                                            int(Nc ** self.Nc_power), 1),
                    self.is_imaginary)
        else:
            return (self.coeff * fractions.Fraction(\
                                            1, int(Nc ** abs(self.Nc_power))),
                    self.is_imaginary)

    def to_canonical(self, immutable=None):
        """Returns the canonical representation of the immutable representation 
        (i.e., first index is 1, ...). This allow for an easy comparison of
        two color strings, i.e. independently of the actual index names (only
        relative positions matter). Also returns the conversion dictionary.
        If no immutable representation is given, use the one build from self."""

        if not immutable:
            immutable = self.to_immutable()

        replaced_indices = {}
        curr_ind = 1
        return_list = []

        for elem in immutable:
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

        return (tuple(return_list), replaced_indices)

    def __eq__(self, col_str):
        """Check if two color strings are equivalent by checking if their
        canonical representations and the coefficients are equal."""

        return self.coeff == col_str.coeff and \
               self.Nc_power == col_str.Nc_power and \
               self.is_imaginary == col_str.is_imaginary and \
               self.to_canonical() == col_str.to_canonical()

    def __ne__(self, col_str):
        """Logical opposite of ea"""

        return not self.__eq__(col_str)

    def is_similar(self, col_str):
        """Check if two color strings are similar by checking if their
        canonical representations and Nc/I powers are equal."""

        return self.Nc_power == col_str.Nc_power and \
               self.is_imaginary == col_str.is_imaginary and \
               self.to_canonical() == col_str.to_canonical()

#===============================================================================
# ColorFactor
#===============================================================================
class ColorFactor(list):
    """ColorFactor objects are list of ColorString with an implicit summation.
    They can be simplified by simplifying all their elements."""

    def __str__(self):
        """Returns a nice string for print"""

        return '+'.join(['(%s)' % str(col_str) for col_str in self])

    def append_str(self, new_str):
        """Special append taking care of adding new string to strings already
        existing with the same structure."""

        for col_str in self:
            # Check if strings are similar, this IS the optimal way of doing
            # it. Note that first line only compare the lists, not the 
            # properties associated
            if col_str.is_similar(new_str):
                # Add them
                col_str.add(new_str)
                return True

        # If no correspondence is found, append anyway
        self.append(new_str)
        return False

    def extend_str(self, new_col_fact):
        """Special extend taking care of adding new strings to strings already
        existing with the same structure."""

        for col_str in new_col_fact:
            self.append_str(col_str)

    def simplify(self):
        """Returns a new color factor where each color string has been
        simplified once and similar strings have been added."""

        new_col_factor = ColorFactor()
        # Simplify
        for col_str in self:
            res = col_str.simplify()
            if res:
                new_col_factor.extend_str(res)
            else:
                new_col_factor.append_str(col_str)

        # Only returns non zero elements
        return ColorFactor([col_str for col_str in \
                            new_col_factor if col_str.coeff != 0])

    def full_simplify(self):
        """Simplify the current color factor until the result is stable"""

        result = copy.copy(self)
        while(True):
            ref = copy.copy(result)
            result = result.simplify()
            if result == ref:
                return result

    def set_Nc(self, Nc=3):
        """Returns a tuple containing real and imaginary parts of the current
        color factor, when Nc is replaced (3 by default)."""

        return (sum([cs.set_Nc(Nc)[0] for cs in self if not cs.is_imaginary]),
                sum([cs.set_Nc(Nc)[0] for cs in self if cs.is_imaginary]))


    def replace_indices(self, repl_dict):
        """Replace current indices following the rules listed in the replacement
        dictionary written as {old_index:new_index,...}, does that for ALL 
        color strings."""

        map(lambda col_str:col_str.replace_indices(repl_dict), self)

    def create_copy(self):
        """Returns a real copy of self, non trivial because bug in 
        copy.deepcopy"""

        res = ColorFactor()
        for col_str in self:
            res.append(col_str.create_copy())

        return res

    __copy__ = create_copy





