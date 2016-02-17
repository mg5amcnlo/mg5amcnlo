################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Classes and methods required for all calculations related to SU(N) color 
algebra."""

import array
import copy
import fractions
import itertools
import madgraph.various.misc as misc

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
        return self

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
# ColorOne
#===============================================================================
class ColorOne(ColorObject):
    """The one of the color object"""

    def __init__(self, *args):
        """Check for no index"""
        
        assert len(args) == 0 , "ColorOne objects must have no index!"
        
        super(ColorOne, self).__init__()

    def simplify(self):
        """"""
        assert len(self)==0, "There is argument(s) in color object ColorOne."
        col_str = ColorString()
        col_str.coeff = fractions.Fraction(1, 1)
        return ColorFactor([col_str])


    def pair_simplify(self, col_obj):
        """Implement ColorOne product simplification"""

        if any(isinstance(col_obj, c_type) for c_type in [Tr,T,f,d,ColorOne]):
            col_str = ColorString([col_obj])
            return ColorFactor([col_str])
        return None


#===============================================================================
# T
#===============================================================================
class T(ColorObject):
    """The T color object. Last two indices have a special meaning"""

    def __init__(self, *args):
        """Check for at least two indices"""
        
        assert len(args) > 1 , "T objects must have at least two indices!"
        
        super(T, self).__init__()
        
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
        return self

#===============================================================================
# f
#===============================================================================
class f(ColorObject):
    """The f color object"""

    def __init__(self, *args):
        """Ensure f and d objects have strictly 3 indices"""
        
        assert len(args) == 3, "f and d objects must have three indices!"
        
        super(f, self).__init__()
                

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
# Epsilon and EpsilonBar, the totally antisymmetric tensors of three triplet
# (antitriplet) indices
#===============================================================================
class Epsilon(ColorObject):
    """Epsilon_ijk color object for three triplets"""

    def __init__(self, *args):
        """Ensure e_ijk objects have strictly 3 indices"""

        super(Epsilon, self).__init__()
        assert len(args) == 3, "Epsilon objects must have three indices!"

    @staticmethod
    def perm_parity(lst, order=None):
        '''\                                                                                                                                                                                                 
        Given a permutation of the digits 0..N in order as a list,                                                                                                                                           
        returns its parity (or sign): +1 for even parity; -1 for odd.                                                                                                                                        
        '''
        lst = lst[:]
        sort =lst[:]
        if not order:
            order = lst[:]
            order.sort()
        parity = 1
        for i in range(0,len(lst)-1):
            if lst[i] != order[i]:
                parity *= -1
                mn = lst.index(order[i])
                lst[i],lst[mn] = lst[mn],lst[i]
        return parity

    def simplify(self):
        """Implement epsilon(i,k,j) = -epsilon(i,j,k) i<j<k"""
        
        
        
        # epsilon(i,k,j) = -epsilon(i,j,k) i<j<k
        order_list = list(self[:])
        order_list.sort()
        
        if list(self[:]) != order_list:
            col_str1 = ColorString([Epsilon(*order_list)])
            col_str1.coeff = self.perm_parity(self[:], order_list)
            return ColorFactor([col_str1])

    def pair_simplify(self, col_obj):
        """Implement e_ijk ae_ilm = T(j,l)T(k,m) - T(j,m)T(k,l) and
        e_ijk T(l,k) = e_ikl"""

        # e_ijk ae_ilm = T(j,l)T(k,m) - T(j,m)T(k,l)
        if isinstance(col_obj, EpsilonBar):

            incommon = False
            eps_indices = self[:]
            aeps_indices = col_obj[:]
            for i in self:
                if i in col_obj:
                    incommon = True
                    com_index_eps = self.index(i)
                    com_index_aeps = col_obj.index(i)

            if incommon:
                eps_indices = self[com_index_eps:] + self[:com_index_eps]
                aeps_indices = col_obj[com_index_aeps:] + col_obj[:com_index_aeps]
                col_str1 = ColorString([T(eps_indices[1], aeps_indices[1]),
                                       T(eps_indices[2], aeps_indices[2])])
                col_str2 = ColorString([T(eps_indices[1], aeps_indices[2]),
                                       T(eps_indices[2], aeps_indices[1])])

                col_str2.coeff = fractions.Fraction(-1, 1)

                return ColorFactor([col_str1, col_str2])

        # e_ijk T(l,k) = e_ikl
        if isinstance(col_obj, T) and len(col_obj) == 2 and col_obj[1] in self:

            com_index = self.index(col_obj[1])
            new_self = copy.copy(self)
            new_self[com_index] = col_obj[0]

            return ColorFactor([ColorString([new_self])])


    def complex_conjugate(self):
        """Complex conjugation. Overwritten here because complex conjugation
        interchange triplets and antitriplets."""

        return EpsilonBar(*self)


class EpsilonBar(ColorObject):
    """Epsilon_ijk color object for three antitriplets"""

    def __init__(self, *args):
        """Ensure e_ijk objects have strictly 3 indices"""

        super(EpsilonBar, self).__init__()
        assert len(args) == 3, "EpsilonBar objects must have three indices!"

    def pair_simplify(self, col_obj):
        """Implement ebar_ijk T(k,l) = e_ikl"""

        # ebar_ijk T(k,l) = ebar_ijl
        if isinstance(col_obj, T) and len(col_obj) == 2 and col_obj[0] in self:

            com_index = self.index(col_obj[0])
            new_self = copy.copy(self)
            new_self[com_index] = col_obj[1]

            return ColorFactor([ColorString([new_self])])

    def simplify(self):
        """Implement epsilon(i,k,j) = -epsilon(i,j,k) i<j<k"""
        
        # epsilon(i,k,j) = -epsilon(i,j,k) i<j<k
        order_list = list(self[:])
        order_list.sort()
        
        if list(self[:]) != order_list:
            col_str1 = ColorString([EpsilonBar(*order_list)])
            col_str1.coeff = Epsilon.perm_parity(self[:], order_list)
            return ColorFactor([col_str1])

    def complex_conjugate(self):
        """Complex conjugation. Overwritten here because complex conjugation
        interchange triplets and antitriplets."""

        return Epsilon(*self)


#===============================================================================
# Color sextet objects: K6, K6Bar, T6
#                       Note that delta3 = T, delta6 = T6, delta8 = 2 Tr
# This 2 Tr is weird and should be check why it is not the expected 1/2.
#===============================================================================

class K6(ColorObject):
    """K6, the symmetry clebsch coefficient, mapping into the symmetric
    tensor."""

    def __init__(self, *args):
        """Ensure sextet color objects have strictly 3 indices"""

        super(K6, self).__init__()
        assert len(args) == 3, "sextet color objects must have three indices!"

    def pair_simplify(self, col_obj):
        """Implement the replacement rules
        K6(m,i,j)K6Bar(m,k,l) = 1/2(delta3(l,i)delta3(k,j)
                                  + delta3(k,i)delta3(l,j))
                            = 1/2(T(l,i)T(k,j) + T(k,i)T(l,j))
        K6(m,i,j)K6Bar(n,j,i) = delta6(m,n)
        K6(m,i,j)K6Bar(n,i,j) = delta6(m,n)
        delta3(i,j)K6(m,i,k) = K6(m,j,k)
        delta3(i,k)K6(m,j,i) = K6(m,j,k)."""

        if isinstance(col_obj, K6Bar):

            m = self[0]
            n = col_obj[0]

            ij1 = self[-2:]
            ij2 = col_obj[-2:]

            # K6(m,i,j)K6Bar(m,k,l) = 1/2(T(l,i)T(k,j)
            #                           + T(k,i)T(l,j)
            if m == n:
                col_str1 = ColorString([T(ij2[1], ij1[0]),
                                        T(ij2[0], ij1[1])])
                col_str2 = ColorString([T(ij2[0], ij1[0]),
                                        T(ij2[1], ij1[1])])
                col_str1.coeff = fractions.Fraction(1, 2)
                col_str2.coeff = fractions.Fraction(1, 2)

                return ColorFactor([col_str1, col_str2])

            # K6(m,i,j)K6Bar(n,j,i) = delta6(m,n)
            if ij1[1] == ij2[0] and ij1[0] == ij2[1]:
                return ColorFactor([ColorString([T6(m, n)])])

            # K6(m,i,j)K6Bar(n,i,j) = delta6(m,n)
            if ij1[0] == ij2[0] and ij1[1] == ij2[1]:
                return ColorFactor([ColorString([T6(m, n)])])

        if isinstance(col_obj, T) and len(col_obj) == 2:
            # delta3(i,j)K6(m,i,k) = K6(m,j,k)
            # delta3(i,k)K6(m,j,i) = K6(m,j,k)
            if col_obj[0] in self[-2:]:
                index1 = self[-2:].index(col_obj[0])
                return ColorFactor([ColorString([K6(self[0],
                                                    self[2-index1],
                                                    col_obj[1])])])

    def complex_conjugate(self):
        """Complex conjugation. By default, the ordering of color index is
        reversed. Can be overwritten for specific color objects like T,..."""

        return K6Bar(*self)


class K6Bar(ColorObject):
    """K6Bar, the barred symmetry clebsch coefficient, mapping into the symmetric
    tensor."""

    def __init__(self, *args):
        """Ensure sextet color objects have strictly 3 indices"""

        super(K6Bar, self).__init__()
        assert len(args) == 3, "sextet color objects must have three indices!"

    def pair_simplify(self, col_obj):
        """Implement the replacement rules
        delta3(i,j)K6Bar(m,j,k) = K6Bar(m,i,k)
        delta3(k,j)K6Bar(m,i,j) = K6Bar(m,i,k)."""
        
        if isinstance(col_obj, T) and len(col_obj) == 2:
            # delta3(i,j)K6Bar(m,j,k) = K6Bar(m,i,k)
            # delta3(k,j)K6Bar(m,i,j) = K6Bar(m,i,k)
            if col_obj[1] in self[-2:]:
                index1 = self[-2:].index(col_obj[1])
                return ColorFactor([ColorString([K6Bar(self[0],
                                                    self[2-index1],
                                                    col_obj[0])])])

    def complex_conjugate(self):
        """Complex conjugation. By default, the ordering of color index is
        reversed. Can be overwritten for specific color objects like T,..."""

        return K6(*self)

class T6(ColorObject):
    """The T6 sextet trace color object."""

    new_index = 10000

    def __init__(self, *args):
        """Check for exactly three indices"""

        super(T6, self).__init__()
        assert len(args) >= 2 and len(args) <= 3, \
               "T6 objects must have two or three indices!"

    def simplify(self):
        """Implement delta6(i,i) = 1/2 Nc(Nc+1),
        T6(a,i,j) = 2(K6(i,ii,jj)T(a,jj,kk)K6Bar(j,kk,ii))"""

        # delta6(i,i) = Nc
        if len(self) == 2 and self[0] == self[1]:
            col_str1 = ColorString()
            col_str1.Nc_power = 2
            col_str1.coeff = fractions.Fraction(1, 2)
            col_str2 = ColorString()
            col_str2.Nc_power = 1
            col_str2.coeff = fractions.Fraction(1, 2)
            return ColorFactor([col_str1, col_str2])

        if len(self) == 2:
            return

        # Set new indices according to the Mathematica template
        ii = T6.new_index
        jj = ii + 1
        kk = jj + 1
        T6.new_index += 3
        # Create the resulting color objects
        col_string = ColorString([K6(self[1], ii, jj),
                                  T(self[0], jj, kk),
                                  K6Bar(self[2], kk, ii)])
        col_string.coeff = fractions.Fraction(2, 1)
        return ColorFactor([col_string])

    def pair_simplify(self, col_obj):
        """Implement the replacement rules
        delta6(i,j)delta6(j,k) = delta6(i,k)
        delta6(m,n)K6(n,i,j) = K6(m,i,j)
        delta6(m,n)K6Bar(m,i,j) = K6Bar(n,i,j)."""

        if len(self) == 3:
            return

        if isinstance(col_obj, T6) and len(col_obj) == 2:
            #delta6(i,j)delta6(j,k) = delta6(i,k)
            if col_obj[0] == self[1]:
                return ColorFactor([ColorString([T6(self[0],
                                                        col_obj[1])])])

        if isinstance(col_obj, K6):
            # delta6(m,n)K6(n,i,j) = K6(m,i,j)
            if col_obj[0] == self[1]:
                return ColorFactor([ColorString([K6(self[0],
                                                    col_obj[1],
                                                    col_obj[2])])])
                

        if isinstance(col_obj, K6Bar):        
            # delta6(m,n)K6Bar(m,i,j) = K6Bar(n,i,j)."""
            if col_obj[0] == self[0]:
                return ColorFactor([ColorString([K6Bar(self[1],
                                                    col_obj[1],
                                                    col_obj[2])])])

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
    # The loop_NC_power attribute is the power of Nc that comes from the
    # possible color trace that appear in loop diagrams. It is typically
    # equal to 1 if the loop diagrams is a closed fermion loop and 0 otherwise.
    loop_Nc_power = 0
    canonical = None
    immutable = None
    
    def __init__(self, init_list=[],
                 coeff=fractions.Fraction(1, 1),
                 is_imaginary=False, Nc_power=0, loop_Nc_power=0):
        """Overrides norm list constructor to implement easy modification
        of coeff, is_imaginary and Nc_power"""

        if init_list:
            for obj in init_list:
                assert type(obj) != array.array
            self.extend(init_list)
        self.coeff = coeff
        self.is_imaginary = is_imaginary
        self.Nc_power = Nc_power
        self.loop_Nc_power = loop_Nc_power

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

        # Reset "canonical", so don't get wrong result from comparison
        self.canonical = None
        self.immutable = None
        
        self.extend(other)

    def simplify(self):
        """Simplify the current ColorString by applying simplify rules on
        each element and building a new ColorFactor to return if necessary"""

        # First, try to simplify element by element
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

        compl_conj_str = ColorString([], self.coeff, self.is_imaginary,
                                     self.Nc_power)
        for col_obj in self:
            compl_conj_str.append(col_obj.complex_conjugate())
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
        
        if self.immutable:
            return self.immutable

        ret_list = [(col_obj.__class__.__name__, tuple(col_obj)) \
                        for col_obj in self]

        if not ret_list and self.coeff:
            ret_list=[("ColorOne",tuple([]))]

        ret_list.sort()
        self.immutable = tuple(ret_list)

        return self.immutable

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
            assert type(col_obj) != array.array
            res.append(col_obj.create_copy())
        res.coeff = self.coeff
        res.is_imaginary = self.is_imaginary
        res.Nc_power = self.Nc_power
        res.loop_Nc_power = self.loop_Nc_power

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

    def order_summation(self, immutable=None):
        """Force a specific order for the summation indices 
           in case we have Clebsch Gordan coefficients K6's or K6Bar's
           This is necessary to correctly recognize later on the equivalent 
           color strings (otherwise the color basis is degenerate).
           The new ordering is as follow:
                1. put K and KBar Clebsch Gordan coefficients at the end of the list of color factors
                   the other factors are re-arranged in the reversed order compared with immutable 
                2. rename the summation indices so that they are increasing (starting from 10000)
                   from left to right
                3. finally, after the summation indices have been renamed, replace
                   K6(a,i,j) by K6(a,j,i) and K6Bar(a,i,j) by K6Bar(a,j,i) IF j>i
        """

        if not immutable:
            immutable = self.to_immutable()

#       STEP 1: first scan to see whether there are some K's or KBar's,
#       and put them at the en 
        immutable_order2=[]
        go_further=0
        for  elem in immutable:
          if elem[0]=="K6" or elem[0]=="K6Bar" :
             immutable_order2.append(elem)
             go_further=1
          else: immutable_order2.insert(0,elem)

        if go_further==0: return

#       STEP 2: rename the summation indices so that they are increasing (starting from 10000)
#               from left to right
        replaced_indices = {}
        curr_ind = 10000
        return_list = []

        for elem in immutable_order2:
            can_elem = [elem[0], []]
            for index in elem[1]:
              if index>9999:  # consider only summation indices
                try:
                    new_index = replaced_indices[index]
                except KeyError:
                    new_index = curr_ind
                    curr_ind += 1
                    replaced_indices[index] = new_index
              else: new_index=index
              can_elem[1].append(new_index)
#       STEP 3.  replace K6(a,i,j) by K6(a,j,i) and K6Bar(a,i,j) by K6Bar(a,j,i) IF j>i
            if (can_elem[0]=="K6" or can_elem[0]=="K6Bar"):
               if can_elem[1][2]>can_elem[1][1]: can_elem[1]=[can_elem[1][0], can_elem[1][2], can_elem[1][1] ]
            return_list.append((can_elem[0], tuple(can_elem[1])))
        return_list.sort()

        self.from_immutable(return_list)
        self.immutable=None   # don't use the information self.immutable later on in the code, 
                              # since the summation indices have been modified 
        return

    def to_canonical(self, immutable=None):
        """Returns the canonical representation of the immutable representation 
        (i.e., first index is 1, ...). This allow for an easy comparison of
        two color strings, i.e. independently of the actual index names (only
        relative positions matter). Also returns the conversion dictionary.
        If no immutable representation is given, use the one build from self."""

        if not immutable:
            immutable = self.to_immutable()

        if self.canonical:
            return self.canonical

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

        self.canonical = (tuple(return_list), replaced_indices)
        return self.canonical

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

    def near_equivalent(self, col_str):
        """Check if two color strings are equivalent looking only at
        the color objects (used in color flow string calculation)"""

        if len(self.to_canonical()) != len(col_str.to_canonical()):
            return False

        return all([co1[0] == co2[0] and sorted(co1[1]) == sorted(co2[1]) \
                        for (co1,co2) in zip(self.to_canonical()[0],
                                             col_str.to_canonical()[0])])

#===============================================================================
# ColorFactor
#===============================================================================
class ColorFactor(list):
    """ColorFactor objects are list of ColorString with an implicit summation.
    They can be simplified by simplifying all their elements."""

    def __str__(self):
        """Returns a nice string for printing"""

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

        # Reset "canonical", so don't get wrong result from comparison
        self.canonical = None
        self.immutable = None

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





