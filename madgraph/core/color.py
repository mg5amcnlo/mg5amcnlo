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

"""Classes, methods and functions required for all calculations
related to QCD color."""

import copy
import operator
import re

#===============================================================================
# ColorString
#===============================================================================
class ColorString(list):
    """Define a color string as an ordered list of strings corresponding to the
    different color structures. Different elements are implicitly linked by
    a product operator."""
    
    def __init__(self, init_list=None):
        
        list.__init__(self)
        
        if init_list:
            if not isinstance(init_list, list):
                raise ValueError, \
                    "Object %s is not a valid list" % init_list
            for elem in init_list:
                if self.is_valid_color_structure(elem):
                    # Be sure there is no white spaces
                    re.sub(r'\s', '', elem) 
                    self.append(elem)
                else: raise ValueError, \
                        "String %s is not a valid color structure" % elem
    
    def is_valid_color_structure(self, test_str):
        """Checks the validity of a given color structure stored in 
        string test_str. Returns True if valid, False otherwise."""
        
        if not isinstance(test_str, str):
            raise ValueError, "Object %s is not a valid string." % str(test_str)
        
        # Check if the string has the right format
        if not re.match(r"""^(T\(-?\d+,-?\d+\)
                            |T\(-?\d+,-?\d+,-?\d+\)
                            |f\(-?\d+,-?\d+,-?\d+\)
                            |d\(-?\d+,-?\d+,-?\d+\)
                            |(1/)?Nc
                            |-?(\d+/)?\d+
                            |I)$""",
                        test_str, re.VERBOSE):
            return False
        else:
            return True
    
    def append(self, mystr):
        """Appends an string, but test if valid before."""
        if not self.is_valid_color_structure(mystr):
            raise ValueError, \
                        "String %s is not a valid color structure" % mystr
        else:
            list.append(self, mystr)
    
    def insert(self, pos, mystr):
        """Insert an string at position pos, but test if valid before."""
        if not self.is_valid_color_structure(mystr):
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
        standard identities."""
        
        while True:
            original = copy.copy(self)
            
            self.__simplify_coeffs()
            self.__simplify_traces()
            self.__simplify_delta()
            
            if self == original:
                break
    
    def __simplify_traces(self):
        """Applies T(i,i) = Nc and T(a,i,i)=0 on all elements of the current
        color string."""
    
        for index, mystr in enumerate(self):
            # T(i,i) = Nc
            if re.match(r'T\((?P<id>-?\d+),(?P=id)\)', mystr):
                self[index] = 'Nc'
            # T(a,i,i) = 0
            if re.match(r'T\(\d+,(?P<id>-?\d+),(?P=id)\)', mystr):
                self[index] = '0'
        return True
    
    def __simplify_delta(self):
        """Applies T(i,x)T(a,x,j) = T(a,i,j) and T(x,j)T(a,i,x) = T(a,i,j)
        on the first valid pair of elements of the current color string. 
        The first element of the pair is replaced, the second one removed.
        Return True if one replacement is made, False otherwise."""
    
        for index1, mystr1 in enumerate(self):
            for index2, mystr2 in enumerate(self[index1 + 1:]):
                
                match_strings = \
                (r"^T\((?P<i>-?\d+),(?P<x>-?\d+)\)T\((?P<a>-?\d+),(?P=x),(?P<j>-?\d+)\)$",
                 r"^T\((?P<x>-?\d+),(?P<j>-?\d+)\)T\((?P<a>-?\d+),(?P<i>-?\d+),(?P=x)\)$",
                 r"^T\((?P<a>-?\d+),(?P<x>-?\d+),(?P<j>-?\d+)\)T\((?P<i>-?\d+),(?P=x)\)$",
                 r"^T\((?P<a>-?\d+),(?P<i>-?\d+),(?P<x>-?\d+)\)T\((?P=x),(?P<j>-?\d+)\)$")
                
                for match_str in match_strings:
                    res_match_object = re.match(match_str, mystr1 + mystr2)
                    if res_match_object:
                        self[index1] = "T(%s,%s,%s)" % \
                                (res_match_object.group('a'),
                                 res_match_object.group('i'),
                                 res_match_object.group('j'))
                        del self[index1 + index2 + 1]
                        return True
        
        return False

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
        
        # Simplify numbers
        
        # Identify numbers and numbers with denominators, remove them all
        numlist = [elem for elem in self if re.match('^-?(\d+/)?\d+$', elem)]
        
        
        # Extract the numerator
        numerator = reduce(operator.mul,
                      [int(re.match('^-?(?P<x>\d+)(/\d+)?$', elem).group('x'))\
                      for elem in numlist], 1)
        # Break if the numerator is 0 --> no need for further calculation
        if numerator == 0:
            # Empty the color string and just leave '0'
            del self[:]
            self.insert(0, '0')
            return True
        
        # Extract denominator and sign
        numwithdenlist = [elem for elem in \
                            numlist if re.match('^-?\d+/\d+$', elem)]
        is_neg = len([elem for elem in numlist if re.match('^-.*$', elem)]) % 2
        denominator = reduce(operator.mul,
                      [int(re.match('^-?\d+/(?P<y>\d+)$', elem).group('y'))\
                      for elem in numwithdenlist], 1)
        
        # Try to simplify further fractions
        def gcd(a, b):
            while b: a, b = b, a % b
            return a
        dev = gcd(numerator, denominator)
        numerator //= dev
        denominator //= dev
        
        # Output the result with the correct format (1 is not output)
        for num in numlist: self.remove(num)
        if denominator != 1:
            if is_neg:
                self.insert(0, "-%i/%i" % (numerator, denominator))
            else:
                self.insert(0, "%i/%i" % (numerator, denominator))
        elif numerator != 1 or (numerator == 1 and is_neg):
            if is_neg:
                self.insert(0, "-%i" % numerator)
            else:
                self.insert(0, "%i" % numerator)
        
        return True
    
    def expand_composite_terms(self, first_index=0):
        """Expand the first encountered composite term like f,d,... 
        and returns the corresponding list of ColorString objects. 
        New summed indices are negative by convention. If first_index 
        is specified and negative, indices are labeled first_index, 
        first_index-1, first_index-2,... If not, first_index is chosen 
        as the smallest (in abs value) summed index still free. 
        This method will NOT modify the current color string. If nothing
        to expand is found, returns an empty list."""
        
        # Find the smallest negative index still free
        if first_index >= 0:
            all_numbers = []
            for elem in self:
                all_numbers.extend([int(x) for x in re.findall('-?\d+', elem)])
            first_index = min(min(all_numbers) - 1, -1)
            
        i = first_index
        j = first_index - 1
        k = first_index - 2
        
        for index, elem in enumerate(self):
            
            # Treat d terms
            match_object = \
            re.match('^d\((?P<a>-?\d+),(?P<b>-?\d+),(?P<c>-?\d+)\)$', elem)
            
            if match_object:
                
                a = match_object.group('a')
                b = match_object.group('b')
                c = match_object.group('c')
                
                color_string1 = copy.copy(self)
                color_string1.remove(elem)
                color_string2 = copy.copy(self)
                color_string2.remove(elem)
                
                color_string1.insert(index, '2')
                color_string1.insert(index + 1, "T(%s,%i,%i)" % (a, i, j))
                color_string1.insert(index + 2, "T(%s,%i,%i)" % (b, j, k))  
                color_string1.insert(index + 3, "T(%s,%i,%i)" % (c, k, i))
                
                color_string2.insert(index, '2')
                color_string2.insert(index + 1, "T(%s,%i,%i)" % (c, i, j))
                color_string2.insert(index + 2, "T(%s,%i,%i)" % (b, j, k))  
                color_string2.insert(index + 3, "T(%s,%i,%i)" % (a, k, i))
                
                return [color_string1, color_string2]
            
            # Treat f terms
            match_object = \
            re.match('^f\((?P<a>-?\d+),(?P<b>-?\d+),(?P<c>-?\d+)\)$', elem)
            
            if match_object:
                
                a = match_object.group('a')
                b = match_object.group('b')
                c = match_object.group('c')
                
                color_string1 = copy.copy(self)
                color_string1.remove(elem)
                color_string2 = copy.copy(self)
                color_string2.remove(elem)
                
                color_string1.insert(index, '-2')
                color_string1.insert(index + 1, 'I')
                color_string1.insert(index + 2, "T(%s,%i,%i)" % (a, i, j))
                color_string1.insert(index + 3, "T(%s,%i,%i)" % (b, j, k))  
                color_string1.insert(index + 4, "T(%s,%i,%i)" % (c, k, i))
                
                color_string2.insert(index, '2')
                color_string2.insert(index + 1, 'I')
                color_string2.insert(index + 2, "T(%s,%i,%i)" % (c, i, j))
                color_string2.insert(index + 3, "T(%s,%i,%i)" % (b, j, k))  
                color_string2.insert(index + 4, "T(%s,%i,%i)" % (a, k, i))
                
                return [color_string1, color_string2]
        
        return []
                
            


class ColorFactor(list):
    """A list of ColorString object used to store the final result of a given
    color factor calculation. Different elements are implicitly linked by a
    sum operator.
    """

    def __init__(self, init_list=None):
        """Creates a new ColorFactor object. If a list of ColorString
        is given, add them."""

        list.__init__(self)

        if init_list is not None:
            for object in init_list:
                self.append(object)

    def append(self, object):
        """Appends an ColorString, but test if valid before."""
        if not isinstance(object, ColorString):
            raise ValueError, \
                "Object %s is not a valid ColorString" % repr(object)
        else:
            list.append(self, object)
    
    def insert(self, pos, object):
        """Insert an ColorString at position pos, but test if valid before."""
        if not isinstance(object, ColorString):
            raise ValueError, \
                "Object %s is not a valid ColorString" % repr(object)
        else:
            list.insert(self, pos, object)
    
    def extend(self, col_string):
        """Extend with another ColorFactor, but test if valid before."""
        if not isinstance(col_string, ColorString):
            raise ValueError, \
                        "Object %s is not a valid color string" % col_string
        else:
            list.extend(self, col_string)   
        
        
        
        
                
                
                

        
    
    
