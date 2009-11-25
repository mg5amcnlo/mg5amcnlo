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

"""Classes, methods, functions and regular expressions required 
for all calculations related to QCD color."""

import copy
import operator
import re

#===============================================================================
# Regular expression objects (compiled)
#===============================================================================

# Match only valid color string object
re_color_object = re.compile(r"""^(T\(((-?\d+)(,-?\d+)*)?\)
                            |Tr\(((-?\d+)(,-?\d+)*)?\)
                            |f\(-?\d+,-?\d+,-?\d+\)
                            |d\(-?\d+,-?\d+,-?\d+\)
                            |(1/)?Nc
                            |-?(\d+/)?\d+
                            |I)$""",
                            re.VERBOSE)

# T trace T(a,b,c,...,i,i), group start is a,b,c,...
re_T_trace = re.compile(r"""^T\((?P<start>(-?\d+,)*?(-?\d+)?),?
                            (?P<id>-?\d+),(?P=id)\)$""", re.VERBOSE)

# T product T(a,...,i,j)T(b,...,j,k), group start1 is a,...,
# group start2 is b,..., group id1 is i and group id2 is k
re_T_product1 = re.compile(r"""^T\((?P<start1>(-?\d+,)*)
                                (?P<id1>-?\d+),(?P<summed>-?\d+)\)
                                T\((?P<start2>(-?\d+,)*)
                                (?P=summed),(?P<id2>-?\d+)\)$""", re.VERBOSE)

# T product T(b,...,j,k)T(a,...,i,j), group start1 is a,...,
# group start2 is b,..., group id1 is i and group id2 is k
re_T_product2 = re.compile(r"""^T\((?P<start2>(-?\d+,)*)
                                (?P<summed>-?\d+),(?P<id2>-?\d+)\)
                                T\((?P<start1>(-?\d+,)*)
                                (?P<id1>-?\d+),(?P=summed)\)$""", re.VERBOSE)

# Tr()
re_trace_0_index = re.compile(r"^Tr\(\)$")

# Tr(i), group id is i
re_trace_1_index = re.compile(r"^Tr\((?P<id>-?\d+)\)$")

# Tr(a,b,c,...), group elems is a,b,c,...
re_trace_n_indices = re.compile(r"^Tr\((?P<elems>(-?\d+,)*(-?\d+)?)\)$")

# Match a fraction (with denominator or not), group has num/den entries
re_fraction = re.compile(r"^(?P<num>-?\d+)(/(?P<den>\d+))?$")

# Match f(a,b,c) terms, groups elements are a, b and c
re_f_term = re.compile(r"^f\((?P<a>-?\d+),(?P<b>-?\d+),(?P<c>-?\d+)\)")

# Match d(a,b,c) terms, groups elements are a, b and c
re_d_term = re.compile(r"^d\((?P<a>-?\d+),(?P<b>-?\d+),(?P<c>-?\d+)\)")

# Match T(a,...,x,b,...,x,c,...,i,j), groups element are a='a,...',b='b,...,' 
# c='c,...,' (notice commas) and id1=i, id2=j
re_T_int_sum = re.compile(r"""^T\((?P<a>(-?\d+,)*)
                                (?P<x>-?\d+),
                                (?P<b>(-?\d+,)*)
                                (?P=x),
                                (?P<c>(-?\d+,)*)
                                (?P<id1>-?\d+),(?P<id2>-?\d+)\)$""", re.VERBOSE)

# Match Tr(a,...,x,b,...,x,c,...), groups element are a='a,...',b='b,...,' 
# c='c,...,' (notice commas) 
re_trace_int_sum = re.compile(r"""^Tr\((?P<a>(-?\d+,)*)
                                (?P<x>-?\d+),
                                (?P<b>(-?\d+,)*)
                                (?P=x),?
                                (?P<c>(-?\d+,)*(-?\d+)?)\)$""", re.VERBOSE)

#===============================================================================
# ColorString
#===============================================================================
class ColorString(list):
    """Define a color string as an ordered list of strings corresponding to the
    different color structures. Different elements are implicitly linked by
    a product operator."""

    def __init__(self, init_list=None):
        """Initialization method, if init_list is given, try to fill the 
        color string with the content."""

        list.__init__(self)

        if init_list:
            if not isinstance(init_list, list):
                raise ValueError, \
                    "Object %s is not a valid list" % init_list
            for elem in init_list:
                if self.is_valid_color_object(elem):
                    self.append(elem)
                else: raise ValueError, \
                        "String %s is not a valid color object" % elem

    def is_valid_color_object(self, test_str):
        """Checks the validity of a given color object stored in 
        string test_str. Returns True if valid, False otherwise."""

        if not isinstance(test_str, str):
            raise ValueError, "Object %s is not a valid string." % str(test_str)

        # Check if the string has the right format
        if re_color_object.match(test_str):
            return True
        else:
            return False

    def append(self, mystr):
        """Appends an string, but test if valid before."""
        if not self.is_valid_color_object(mystr):
            raise ValueError, \
                        "String %s is not a valid color structure" % mystr
        else:
            list.append(self, mystr)

    def insert(self, pos, mystr):
        """Insert an string at position pos, but test if valid before."""
        if not self.is_valid_color_object(mystr):
            raise ValueError, \
                        "String %s is not a valid color structure" % mystr
        else:
            list.insert(self, pos, mystr)

    def extend(self, col_string):
        """Extend with another color string, but test if valid before."""
        if not isinstance(col_string, ColorString):
            raise ValueError, \
                        "Object %s is not a valid color string" % col_string
        else:
            list.extend(self, col_string)

    def simplify(self):
        """Simplify the current color string as much as possible, using 
        all possible identities."""

        while True:
            original = copy.copy(self)

            self.__simplify_T_traces()
            self.__simplify_T_products()
            self.__simplify_simple_traces()
            self.__simplify_trace_cyclicity()
            self.__simplify_coeffs()
            if self == original:
                break
    
    def __simplify_T_traces(self):
        """Apply the identity T(a,b,c,...,i,i) = Tr(a,b,c,...)"""
        
        for index, col_obj in enumerate(self):
            self[index] = re_T_trace.sub(lambda m: "Tr(%s)" % m.group('start'),
                                     col_obj)
    
    def __simplify_T_products(self):
        """Apply the identity T(a,...,i,j)T(b,...,j,k) = T(a,...,b,...,i,k)
        on the first matching pair"""
        
        for index1, mystr1 in enumerate(self):
            for index2, mystr2 in enumerate(self[index1 + 1:]):
                # Test both product ordering
                m = re_T_product1.match(mystr1 + mystr2)
                if not m:
                    m = re_T_product2.match(mystr1 + mystr2)
                if m:
                    self[index1] = "T(%s%s%s,%s)" % \
                                (m.group('start1'),
                                 m.group('start2'),
                                 m.group('id1'),
                                 m.group('id2'))
                    del self[index1 + index2 + 1]
                    return
    
    def __simplify_simple_traces(self):
        """Apply the identities Tr(a)=0 (AS) and Tr() = Nc (delta)"""
        
        for index, col_obj in enumerate(self):
            if re_trace_0_index.match(col_obj):
                self[index] = "Nc"
            if re_trace_1_index.match(col_obj):
                self[index] = "0"
    
    def __simplify_trace_cyclicity(self):
        """Re-order traces using cyclicity to bring smallest id in front"""
        
        for index, col_obj in enumerate(self):
            m = re_trace_n_indices.match(col_obj)
            if m:
                list_indices = [int(s) for s in m.group('elems').split(',')]
                pos = list_indices.index(min(list_indices))
                res_list = list_indices[pos:]
                res_list.extend(list_indices[:pos])
                self[index] = 'Tr(%s)' % ','.join([str(i) for i in res_list])

    def __simplify_coeffs(self):
        """Applies simple algebraic simplifications on scalar coefficients
        and bring the final result to the first position"""

        # Simplify factors Nc
        numNc = self.count('Nc') - self.count('1/Nc')
        while ('Nc' in self): self.remove('Nc')
        while ('1/Nc' in self): self.remove('1/Nc')
        if numNc > 0:
            for dummy in range(numNc):
                self.insert(0, 'Nc')
        elif numNc < 0:
            for dummy in range(abs(numNc)):
                self.insert(0, '1/Nc')

        # Simplify factors I
        numI = self.count('I')
        while ('I' in self): self.remove('I')
        if numI % 4 == 1:
            self.insert(0, 'I')
        elif numI % 4 == 2:
            self.insert(0, '-1')
        elif numI % 4 == 3:
            self.insert(0, 'I')
            self.insert(0, '-1')

        # Compute numerators and denominators
        numlist = []
        denlist = []
        for elem in self[:]:
            m = re_fraction.match(elem)
            if m:
                self.remove(elem)
                numlist.append(int(m.group('num')))
                if m.group('den'):
                    denlist.append(int(m.group('den')))
        numerator = reduce(operator.mul, numlist, 1)
        denominator = reduce(operator.mul, denlist, 1)
            
        # Deal with 0 coeff
        if numerator == 0:
            del self[:]
            self.insert(0, '0')
            return
        
        # Try to simplify further fractions
        dev = self.__gcd(numerator, denominator)
        numerator //= dev
        denominator //= dev

        # Output result in a correct way
        if denominator != 1:
            self.insert(0, "%i/%i" % (numerator, denominator))
        elif numerator != 1:
            self.insert(0, "%i" % numerator)

    def __gcd(self, a, b):
        """Find the gcd of two integers"""
        while b: a, b = b, a % b
        return a
    
    def __expand_term(self, indices, new_str1, new_str2):
        """Return two color strings built on self where all indices in the
        list indices have been removed and, for the first index of the list, 
        replaced once by new_str1 and once bynew_str2."""
        
        str1 = copy.copy(self)
        str2 = copy.copy(self)
        
        # Remove indices, pay attention to the shift induced by del
        for i, index in enumerate(indices):
            del str1[index + i]
            del str2[index + i]
        
        for i in range(len(new_str1)):
            str1.insert(indices[0] + i, new_str1[i])
        
        for i in range(len(new_str2)):
            str2.insert(indices[0] + i, new_str2[i])
        
        return [str1, str2]
    
    
    def expand_composite_terms(self):
        """Expand the first encountered composite term like f,d,... 
        and returns the corresponding list of ColorString objects. 
        This method will NOT modify the current color string. If nothing
        to expand is found, returns an empty list."""
        
        for index, col_obj in enumerate(self):
            
            # Deal with d terms
            m = re_d_term.match(col_obj)
            if m:
                return self.__expand_term([index], ['2', 'Tr(%s,%s,%s)' % \
                                                      (m.group('a'),
                                                       m.group('b'),
                                                       m.group('c'))],
                                                 ['2', 'Tr(%s,%s,%s)' % \
                                                      (m.group('c'),
                                                       m.group('b'),
                                                       m.group('a'))])
            # Deal with f terms
            m = re_f_term.match(col_obj)
            if m:
                return self.__expand_term([index], ['-2', 'I',
                                                    'Tr(%s,%s,%s)' % \
                                                      (m.group('a'),
                                                       m.group('b'),
                                                       m.group('c'))],
                                                 ['2', 'I', 'Tr(%s,%s,%s)' % \
                                                      (m.group('c'),
                                                       m.group('b'),
                                                       m.group('a'))])
            
        return []
    
    def expand_T_internal_sum(self):
        """Expand the first encountered term with an internal sum in T's using
        T(a,x,b,x,c,i,j) = 1/2(T(a,c,i,j)Tr(b)-1/Nc T(a,b,c,i,j)) and 
        returns the corresponding list of ColorString objects. 
        This method will NOT modify the current color string. If nothing
        to expand is found, returns an empty list."""

        for index, col_obj in enumerate(self):
            
            m = re_T_int_sum.match(col_obj)
            if m:
                # Since b ends with a comma, we should remove it for Tr(b)
                b = m.group('b')
                b = b.rstrip(',')
                return self.__expand_term([index], ['1/2', 'T(%s%s%s,%s)' % \
                                                      (m.group('a'),
                                                       m.group('c'),
                                                       m.group('id1'),
                                                       m.group('id2')),
                                                       'Tr(%s)' % b],
                                                 ['-1/2', '1/Nc',
                                                  'T(%s%s%s%s,%s)' % \
                                                      (m.group('a'),
                                                       m.group('b'),
                                                       m.group('c'),
                                                       m.group('id1'),
                                                       m.group('id2'))])
        return []
    
    def expand_trace_internal_sum(self):
        """Expand the first encountered term with an internal sum in a trace
        using Tr(a,x,b,x,c) = 1/2(T(a,c)Tr(b)-1/Nc T(a,b,c)) and 
        returns the corresponding list of ColorString objects. 
        This method will NOT modify the current color string. If nothing
        to expand is found, returns an empty list."""
        
        for index, col_obj in enumerate(self):
            
            m = re_trace_int_sum.match(col_obj)
            if m:
                # Here we must also be careful about final comma, since there is
                # no index after color indices
                b = m.group('b')
                b = b.rstrip(',')
                ac = m.group('a') + m.group('c')
                ac = ac.rstrip(',')
                abc = m.group('a') + m.group('b') + m.group('c')
                abc = abc.rstrip(',')
                return self.__expand_term([index], ['1/2', 'Tr(%s)' % ac,
                                                       'Tr(%s)' % b],
                                                 ['-1/2', '1/Nc',
                                                  'Tr(%s)' % abc])
        return []


##===============================================================================
## ColorString
##===============================================================================
#class ColorString(list):
#    """Define a color string as an ordered list of strings corresponding to the
#    different color structures. Different elements are implicitly linked by
#    a product operator."""
#
#    def __init__(self, init_list=None):
#
#        list.__init__(self)
#
#        if init_list:
#            if not isinstance(init_list, list):
#                raise ValueError, \
#                    "Object %s is not a valid list" % init_list
#            for elem in init_list:
#                if self.is_valid_color_structure(elem):
#                    # Be sure there is no white spaces
#                    re.sub(r'\s', '', elem)
#                    self.append(elem)
#                else: raise ValueError, \
#                        "String %s is not a valid color structure" % elem
#
#    def is_valid_color_structure(self, test_str):
#        """Checks the validity of a given color structure stored in 
#        string test_str. Returns True if valid, False otherwise."""
#
#        if not isinstance(test_str, str):
#            raise ValueError, "Object %s is not a valid string." % str(test_str)
#
#        # Check if the string has the right format
#        if not re.match(r"""^(T\(-?\d+,-?\d+\)
#                            |T\(-?\d+,-?\d+,-?\d+\)
#                            |f\(-?\d+,-?\d+,-?\d+\)
#                            |d\(-?\d+,-?\d+,-?\d+\)
#                            |(1/)?Nc
#                            |-?(\d+/)?\d+
#                            |I)$""",
#                        test_str, re.VERBOSE):
#            return False
#        else:
#            return True
#
#    def append(self, mystr):
#        """Appends an string, but test if valid before."""
#        if not self.is_valid_color_structure(mystr):
#            raise ValueError, \
#                        "String %s is not a valid color structure" % mystr
#        else:
#            list.append(self, mystr)
#
#    def insert(self, pos, mystr):
#        """Insert an string at position pos, but test if valid before."""
#        if not self.is_valid_color_structure(mystr):
#            raise ValueError, \
#                        "String %s is not a valid color structure" % mystr
#        else:
#            list.insert(self, pos, mystr)
#
#    def extend(self, col_string):
#        """Extend with another color string, but test if valid before."""
#        if not isinstance(col_string, ColorString):
#            raise ValueError, \
#                        "Object %s is not a valid color string" % col_string
#        else:
#            list.extend(self, col_string)
#
#    def simplify(self):
#        """Simplify the current color string as much as possible, using 
#        standard identities."""
#
#        for dummy in range(100):
#            original = copy.copy(self)
#
#            self.__simplify_coeffs()
#            self.__simplify_traces()
#            self.__simplify_delta()
#
#            if self == original:
#                return True
##            print 'subs:', self
#
#        raise ValueError, "Maximal iteration reached!"
#
#    def __simplify_traces(self):
#        """Applies T(i,i) = Nc and T(a,i,i)=0 on all elements of the current
#        color string."""
#
#        for index, mystr in enumerate(self[:]):
#            # T(i,i) = Nc
#            if re.match(r'T\((?P<id>-?\d+),(?P=id)\)', mystr):
#                self[index] = 'Nc'
#            # T(a,i,i) = 0
#            if re.match(r'T\(\d+,(?P<id>-?\d+),(?P=id)\)', mystr):
#                self[index] = '0'
#        return True
#
#    def __simplify_delta(self):
#        """Applies T(i,x)T(a,x,j) = T(a,i,j) and T(x,j)T(a,i,x) = T(a,i,j)
#        on the first valid pair of elements of the current color string. 
#        The first element of the pair is replaced, the second one removed.
#        Return True if one replacement is made, False otherwise."""
#
#        for index1, mystr1 in enumerate(self[:]):
#            for index2, mystr2 in enumerate(self[index1 + 1:]):
#
#                match_strings = \
#    (r"^T\((?P<i>-?\d+),(?P<x>-?\d+)\)T\((?P<a>-?\d+),(?P=x),(?P<j>-?\d+)\)$",
#     r"^T\((?P<x>-?\d+),(?P<j>-?\d+)\)T\((?P<a>-?\d+),(?P<i>-?\d+),(?P=x)\)$",
#     r"^T\((?P<a>-?\d+),(?P<x>-?\d+),(?P<j>-?\d+)\)T\((?P<i>-?\d+),(?P=x)\)$",
#     r"^T\((?P<a>-?\d+),(?P<i>-?\d+),(?P<x>-?\d+)\)T\((?P=x),(?P<j>-?\d+)\)$")
#
#                for match_str in match_strings:
#                    res_match_object = re.match(match_str, mystr1 + mystr2)
#                    if res_match_object:
#                        self[index1] = "T(%s,%s,%s)" % \
#                                (res_match_object.group('a'),
#                                 res_match_object.group('i'),
#                                 res_match_object.group('j'))
#                        del self[index1 + index2 + 1]
#                        return True
#
#                match_strings = \
#    (r"^T\((?P<i>-?\d+),(?P<x>-?\d+)\)T\((?P=x),(?P<j>-?\d+)\)$",
#     r"^T\((?P<x>-?\d+),(?P<j>-?\d+)\)T\((?P<i>-?\d+),(?P=x)\)$")
#
#                for match_str in match_strings:
#                    res_match_object = re.match(match_str, mystr1 + mystr2)
#                    if res_match_object:
#                        self[index1] = "T(%s,%s)" % \
#                                (res_match_object.group('i'),
#                                 res_match_object.group('j'))
#                        del self[index1 + index2 + 1]
#                        return True
#
#        return False
#
#    def __simplify_coeffs(self):
#        """Applies simple algebraic simplifications on scalar coefficients
#        and bring the final result to the first position"""
#
#        # Simplify factors Nc
#        numNc = self.count('Nc') - self.count('1/Nc')
#        while ('Nc' in self): self.remove('Nc')
#        while ('1/Nc' in self): self.remove('1/Nc')
#        if numNc > 0:
#            for dummy in range(numNc):
#                self.insert(0, 'Nc')
#        elif numNc < 0:
#            for dummy in range(abs(numNc)):
#                self.insert(0, '1/Nc')
#
#        # Simplify factors I
#        numI = self.count('I')
#        while ('I' in self): self.remove('I')
#        if numI % 4 == 1:
#            self.insert(0, 'I')
#        elif numI % 4 == 2:
#            self.insert(0, '-1')
#        elif numI % 4 == 3:
#            self.insert(0, 'I')
#            self.insert(0, '-1')
#
#        # Simplify numbers
#
#        # Identify numbers and numbers with denominators, remove them all
#        numlist = [elem for elem in self if re.match('^-?(\d+/)?\d+$', elem)]
#
#
#        # Extract the numerator
#        numerator = reduce(operator.mul,
#                      [int(re.match('^-?(?P<x>\d+)(/\d+)?$', elem).group('x'))\
#                      for elem in numlist], 1)
#        # Break if the numerator is 0 --> no need for further calculation
#        if numerator == 0:
#            # Empty the color string and just leave '0'
#            del self[:]
#            self.insert(0, '0')
#            return True
#
#        # Extract denominator and sign
#        numwithdenlist = [elem for elem in \
#                            numlist if re.match('^-?\d+/\d+$', elem)]
#        is_neg = len([elem for elem in numlist if re.match('^-.*$', elem)]) % 2
#        denominator = reduce(operator.mul,
#                      [int(re.match('^-?\d+/(?P<y>\d+)$', elem).group('y'))\
#                      for elem in numwithdenlist], 1)
#
#        # Try to simplify further fractions
#        dev = self.__gcd(numerator, denominator)
#        numerator //= dev
#        denominator //= dev
#
#        # Output the result with the correct format (1 is not output)
#        for num in numlist: self.remove(num)
#        if denominator != 1:
#            if is_neg:
#                self.insert(0, "-%i/%i" % (numerator, denominator))
#            else:
#                self.insert(0, "%i/%i" % (numerator, denominator))
#        elif numerator != 1 or (numerator == 1 and is_neg):
#            if is_neg:
#                self.insert(0, "-%i" % numerator)
#            else:
#                self.insert(0, "%i" % numerator)
#
#        return True
#
#    def __gcd(self, a, b):
#        """Find the gcd of two integers"""
#        while b: a, b = b, a % b
#        return a
#
#    def __find_first_free_index(self):
#        """Find the first free negative index in self"""
#
#        all_numbers = []
#        for elem in self:
#            all_numbers.extend([int(x) for x in re.findall('-?\d+', elem)])
#        return min(min(all_numbers) - 1, -1)
#
#    def expand_composite_terms(self, first_index=0):
#        """Expand the first encountered composite term like f,d,... 
#        and returns the corresponding list of ColorString objects. 
#        New summed indices are negative by convention. If first_index 
#        is specified and negative, indices are labeled first_index, 
#        first_index-1, first_index-2,... If not, first_index is chosen 
#        as the smallest (in abs value) summed index still free. 
#        This method will NOT modify the current color string. If nothing
#        to expand is found, returns an empty list."""
#
#        # Find the smallest negative index still free
#        if first_index >= 0:
#            first_index = self.__find_first_free_index()
#
#        i = first_index
#        j = first_index - 1
#        k = first_index - 2
#
#        for index, elem in enumerate(self):
#
#            # Treat d terms
#            match_object = \
#            re.match('^d\((?P<a>-?\d+),(?P<b>-?\d+),(?P<c>-?\d+)\)$', elem)
#
#            if match_object:
#
#                a = match_object.group('a')
#                b = match_object.group('b')
#                c = match_object.group('c')
#
#                color_string1 = copy.copy(self)
#                color_string1.remove(elem)
#                color_string2 = copy.copy(self)
#                color_string2.remove(elem)
#
#                color_string1.insert(index, '2')
#                color_string1.insert(index + 1, "T(%s,%i,%i)" % (a, i, j))
#                color_string1.insert(index + 2, "T(%s,%i,%i)" % (b, j, k))
#                color_string1.insert(index + 3, "T(%s,%i,%i)" % (c, k, i))
#
#                color_string2.insert(index, '2')
#                color_string2.insert(index + 1, "T(%s,%i,%i)" % (c, i, j))
#                color_string2.insert(index + 2, "T(%s,%i,%i)" % (b, j, k))
#                color_string2.insert(index + 3, "T(%s,%i,%i)" % (a, k, i))
#
#                return [color_string1, color_string2]
#
#            # Treat f terms
#            match_object = \
#            re.match('^f\((?P<a>-?\d+),(?P<b>-?\d+),(?P<c>-?\d+)\)$', elem)
#
#            if match_object:
#
#                a = match_object.group('a')
#                b = match_object.group('b')
#                c = match_object.group('c')
#
#                color_string1 = copy.copy(self)
#                color_string1.remove(elem)
#                color_string2 = copy.copy(self)
#                color_string2.remove(elem)
#
#                color_string1.insert(index, '-2')
#                color_string1.insert(index + 1, 'I')
#                color_string1.insert(index + 2, "T(%s,%i,%i)" % (a, i, j))
#                color_string1.insert(index + 3, "T(%s,%i,%i)" % (b, j, k))
#                color_string1.insert(index + 4, "T(%s,%i,%i)" % (c, k, i))
#
#                color_string2.insert(index, '2')
#                color_string2.insert(index + 1, 'I')
#                color_string2.insert(index + 2, "T(%s,%i,%i)" % (c, i, j))
#                color_string2.insert(index + 3, "T(%s,%i,%i)" % (b, j, k))
#                color_string2.insert(index + 4, "T(%s,%i,%i)" % (a, k, i))
#
#                return [color_string1, color_string2]
#
#        return []
#
#    def apply_golden_rule(self, first_index=0):
#        """Similar to expand_composite_terms, but applies the golden rule
#        T(a,i1,j1)T(a,i2,j2) = 1/2 (T[i1,j2]T[i2,j1]-1/Nc T[i1,j1]T[i2,j2])
#        on the first T product corresponding to the pattern."""
#
#        for index1, mystr1 in enumerate(self):
#            for index2, mystr2 in enumerate(self[index1 + 1:]):
#                res_match_object = re.match(r"""^T\((?P<a>-?\d+),
#                                                    (?P<i1>-?\d+),
#                                                    (?P<j1>-?\d+)\)
#                                                 T\((?P=a),
#                                                    (?P<i2>-?\d+),
#                                                    (?P<j2>-?\d+)\)$""",
#                                            mystr1 + mystr2, re.VERBOSE)
#                if res_match_object:
#                    i1 = res_match_object.group('i1')
#                    i2 = res_match_object.group('i2')
#                    j1 = res_match_object.group('j1')
#                    j2 = res_match_object.group('j2')
#
#                    color_string1 = copy.copy(self)
#                    del color_string1[index1]
#                    del color_string1[index1 + index2 ]
#
#                    color_string2 = copy.copy(self)
#                    del color_string2[index1]
#                    del color_string2[index1 + index2 ]
#
#                    color_string1.insert(index1, '1/2')
#                    color_string1.insert(index1 + 1, "T(%s,%s)" % (i1, j2))
#                    color_string1.insert(index1 + 2, "T(%s,%s)" % (i2, j1))
#
#                    color_string2.insert(index1, '-1/2')
#                    color_string2.insert(index1 + 1, '1/Nc')
#                    color_string2.insert(index1 + 2, "T(%s,%s)" % (i1, j1))
#                    color_string2.insert(index1 + 3, "T(%s,%s)" % (i2, j2))
#
#                    return [color_string1, color_string2]
#
#        return []
#
#    def extract_coeff(self):
#        """Returns a tuple with first value being a color string with only
#        numerical coefficients, and the second one a color string 
#        with the rest"""
#
#        col_str = copy.copy(self)
#
#        # Remove coeffs one by one
#        if col_str and re.match("^-?(\d+/)?\d+$", col_str[0]):
#            coeff_str = col_str.pop(0)
#            return (coeff_str, col_str)
#        elif col_str:
#            return ('1', col_str)
#
#
#    def is_similar(self, col_str):
#        """Test if self is similar to col_str, i.e. if they have the same
#        tensorial structure (taking into account possible renaming of 
#        summed indices, permutations, ...) and identical powers of Nc/I"""
#
#        col_str1 = copy.copy(self)
#        col_str2 = copy.copy(col_str)
#
#        # Get rid of coefficients
#        col_str1 = col_str1.extract_coeff()[1]
#        col_str2 = col_str2.extract_coeff()[1]
#
#
#        if col_str1 == ColorString() or col_str2 == ColorString():
#            if col_str1 == ColorString() and col_str2 == ColorString():
#                return True
#            else:
#                return False
#
#        # Built lists of indices (only match starting , or ( since findall
#        # is exclusive)
#        all_indices1 = re.findall(r"(?:,|\()(-?\d+)",
#                                  reduce(operator.add, col_str1))
#        all_indices2 = re.findall(r"(?:,|\()(-?\d+)",
#                                  reduce(operator.add, col_str2))
#
#        # If the number of indices is not the same -> cannot be similar
#        if len(all_indices1) != len(all_indices2): return False
#
#        # Build lists of summed indices
#        # list(set(...)) allows to remove double entries
#        sum_indices1 = list(set([index for index in all_indices1 \
#                        if all_indices1.count(index) == 2]))
#        sum_indices2 = list(set([index for index in all_indices2 \
#                        if all_indices2.count(index) == 2]))
#
#        # If the number of summed indices is not the same -> cannot be similar
#        if len(sum_indices1) != len(sum_indices2): return False
#
#        # Create a concatenated string where summed indices are tagged
#        # as X0, X1, ...
#        concat_str1 = reduce(operator.add, col_str1)
#        for i, index in enumerate(sum_indices1):
#            concat_str1 = re.sub(r"(,|\()%s(,|\))" % index,
#                                 r"\1X%i\2" % i,
#                                 concat_str1)
#
#        # test all permutations of str2
#        for str2_permut in itertools.permutations(col_str2):
#            concat_str2 = reduce(operator.add, str2_permut)
#
#            # Test all summed indices ordering possibilities
#            for permut_sum_indices2 in itertools.permutations(sum_indices2):
#                my_str2 = copy.copy(concat_str2)
#                for i, index in enumerate(permut_sum_indices2):
#                    my_str2 = re.sub(r"(,|\()%s(,|\))" % index,
#                                         r"\1X%i\2" % i,
#                                         my_str2)
#                if my_str2 == concat_str1:
#                    return True
#
#        return False
#
#    def add(self, col_str):
#        """Add two color similar strings, i.e. returns a new color string with
#        numerical coefficients added."""
#
#        col_str1 = copy.copy(self)
#        col_str2 = copy.copy(col_str)
#
#        # Separate coefficients
#        coeff1, col_str1 = col_str1.extract_coeff()
#        coeff2, col_str2 = col_str2.extract_coeff()
#
#        # Extract numerators and denominators
#        match1 = re.match('^(?P<num>-?\d+)(/(?P<den>\d+))?$', coeff1)
#        match2 = re.match('^(?P<num>-?\d+)(/(?P<den>\d+))?$', coeff2)
#
#        num1 = int(match1.group('num'))
#        num2 = int(match2.group('num'))
#
#        if match1.group('den'):
#            den1 = int(match1.group('den'))
#        else:
#            den1 = 1
#
#        if match2.group('den'):
#            den2 = int(match2.group('den'))
#        else:
#            den2 = 1
#
#        # Simplify the fraction
#        numerator = num1 * den2 + num2 * den1
#        denominator = den1 * den2
#
#        dev = self.__gcd(numerator, denominator)
#        numerator //= dev
#        denominator //= dev
#
#        # Output the result with the correct format (1 is not output)
#        out_str = ColorString()
#        if denominator != 1:
#            out_str.append("%i/%i" % (numerator, denominator))
#        elif numerator != 1:
#            out_str.append("%i" % numerator)
#        out_str.extend(col_str1)
#
#        return out_str
#
#
##===============================================================================
## ColorFactor
##===============================================================================
#class ColorFactor(list):
#    """A list of ColorString object used to store the final result of a given
#    color factor calculation. Different elements are implicitly linked by a
#    sum operator.
#    """
#
#    def __init__(self, init_list=None):
#        """Creates a new ColorFactor object. If a list of ColorString
#        is given, add them."""
#
#        list.__init__(self)
#
#        if init_list is not None:
#            if isinstance(init_list, list):
#                for object in init_list:
#                    self.append(object)
#            else:
#                raise ValueError, \
#                    "Object %s is not a valid list" % repr(init_list)
#
#    def append(self, object):
#        """Appends an ColorString, but test if valid before."""
#        if not isinstance(object, ColorString):
#            raise ValueError, \
#                "Object %s is not a valid ColorString" % repr(object)
#        else:
#            list.append(self, object)
#
#    def insert(self, pos, object):
#        """Insert an ColorString at position pos, but test if valid before."""
#        if not isinstance(object, ColorString):
#            raise ValueError, \
#                "Object %s is not a valid ColorString" % repr(object)
#        else:
#            list.insert(self, pos, object)
#
#    def extend(self, col_factor):
#        """Extend with another ColorFactor, but test if valid before."""
#        if not isinstance(col_factor, ColorFactor):
#            raise ValueError, \
#                        "Object %s is not a valid ColorFactor" % col_factor
#        else:
#            list.extend(self, col_factor)
#
#    def simplify(self):
#        """Simplify the current ColorFactor. First apply simplification rules
#        on all ColorStrings, then expand first composite terms and apply the
#        golden rule. Iterate until the result does not change anymore"""
#
#        while True:
#            original = copy.copy(self)
#
#            # Expand one composite term if possible
#            for index, col_str in enumerate(self[:]):
#                result = col_str.expand_composite_terms()
#                if result:
#                    self[index] = ColorString(['0'])
#                    self.extend(ColorFactor(result))
#
#            # Remove zeros
#            while(ColorString(['0']) in self): self.remove(ColorString(['0']))
#
#            # Iterate until the result does not change anymore
#            if self == original:
#                break
#
#        while True:
#            original = copy.copy(self)
#
#            # Simplify each color string
#            for col_str in self:
#                col_str.simplify()
#
#            # Apply golden rule
#            for index, col_str in enumerate(self[:]):
#                result = col_str.apply_golden_rule()
#                if result:
#                    self[index] = ColorString(['0'])
#                    self.extend(ColorFactor(result))
#
#            # Remove zeros
#            while(ColorString(['0']) in self): self.remove(ColorString(['0']))
#
#            # Iterate until the result does not change anymore
#            if self == original:
#                break
#
#        self.__collect()
#
#    def __collect(self):
#        """Do a final simplification by grouping terms which are similars"""
#
#        # Try all combinations
#        for i1, col_str1 in enumerate(self[:]):
#            for i2, col_str2 in enumerate(self[i1 + 1:]):
#                if col_str1.is_similar(col_str2):
#                    self[i1] = self[i1].add(col_str2)
#                    self[i1 + i2 + 1] = ColorString(['0'])
#
#        for elem in self:
#            elem.simplify()
#
#        while(ColorString(['0']) in self): self.remove(ColorString(['0']))











