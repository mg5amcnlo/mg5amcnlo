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
import re

class ColorString(list):
    """Define a color string as an ordered list of strings corresponding to the
    different color structures."""
    
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
        if not re.match(r"""^(T\(\d+,\d+\)
                            |T\(\d+,\d+,\d+\)
                            |f\(\d+,\d+,\d+\)
                            |d\(\d+,\d+,\d+\)
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
            self.__simplify_traces()
            self.__simplify_delta()
            if self == original:
                break
    
    def __simplify_traces(self):
        """Applies T(i,i) = Nc and T(a,i,i)=0 on all elements of the current
        color string."""
    
        for index, mystr in enumerate(self):
            # T(i,i) = Nc
            if re.match(r'T\((?P<id>\d+),(?P=id)\)', mystr):
                self[index] = 'Nc'
            # T(a,i,i) = 0
            if re.match(r'T\(\d+,(?P<id>\d+),(?P=id)\)', mystr):
                self[index] = '0'
    
    def __simplify_delta(self):
        """Applies T(i,x)T(a,x,j) = T(a,i,j) and T(x,j)T(a,i,x) = T(a,i,j)
        on the first valid pair of elements of the current color string. 
        The first element of the pair is replaced, the second one removed.
        Return True if one replacement is made, False otherwise."""
    
        for index1, mystr1 in enumerate(self):
            for index2, mystr2 in enumerate(self[index1 + 1:]):
                match_strings = \
            (r"^T\((?P<i>\d+),(?P<x>\d+)\)T\((?P<a>\d+),(?P=x),(?P<j>\d+)\)$",
             r"^T\((?P<x>\d+),(?P<j>\d+)\)T\((?P<a>\d+),(?P<i>\d+),(?P=x)\)$",
             r"^T\((?P<a>\d+),(?P<x>\d+),(?P<j>\d+)\)T\((?P<i>\d+),(?P=x)\)$",
             r"^T\((?P<a>\d+),(?P<i>\d+),(?P<x>\d+)\)T\((?P=x),(?P<j>\d+)\)$")
                
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
                
                
                

        
    
    
