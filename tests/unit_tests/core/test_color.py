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

"""Unit test library for the color related routines in the core library"""

import unittest

import madgraph.core.color as color

#===============================================================================
# ColorTest
#===============================================================================
class ColorTest(unittest.TestCase):
    """Test class for code parts related to color"""

    def test_validity(self):
        """Test the color string validity check"""

        my_color_str = color.ColorString()
        
        valid_strings = ["T(101,102)",
                         "T(1,101,102)",
                         "f(1,2,3)",
                         "d(1,2,3)",
                         "Nc", "1/Nc", "I",
                         "0", "1", "-1", "1/2", "-123/345"]
        
        for valid_str in valid_strings:
            self.assert_(my_color_str.is_valid_color_structure(valid_str))
        
        wrong_strings = ["T(101,102",
                         "T 1,101,102)",
                         "T(1, 101, 102)",
                         "k(1,2,3)",
                         "d((1,2,3))",
                         "d(1,2,3,)",
                         "T(1.2)",
                         "-2/Nc"]
        
        for wrong_str in wrong_strings:
            self.assertFalse(my_color_str.is_valid_color_structure(wrong_str))
    
    def test_colorstring_init(self):
        """Test the color string initialization"""
        
        wrong_lists = [['T(101,102)', 1],
                       ['T(101,102)', 'k(1,2,3)'],
                       'T(101,102)']
    
        for wrg_list in wrong_lists:
            self.assertRaises(ValueError,
                              color.ColorString,
                              wrg_list)
    
    def test_colorstring_manip(self):
        """Test the color string manipulation (append, insert and extend)"""
        
        my_color_string = color.ColorString(['T(101,102)'])
    
        self.assertRaises(ValueError,
                          my_color_string.append,
                          'k(1,2,3)')
        self.assertRaises(ValueError,
                          my_color_string.insert,
                          0, 'k(1,2,3)')  
        self.assertRaises(ValueError,
                          my_color_string.extend,
                          ['k(1,2,3)'])
    
    def test_traces_simplify(self):
        """Test color string trace simplification"""
        
        my_color_string1 = color.ColorString(['T(101,101)',
                                             'T(102,103)',
                                             'T(1,-101,-101)',
                                             'T(2,104,105)'])
        
        my_color_string2 = color.ColorString(['0'])
        
        my_color_string1.simplify()
        
        self.assertEqual(my_color_string1, my_color_string2)
    
    def test_delta_simplify(self):
        """Test color string delta simplification"""
        
        my_color_string1 = color.ColorString(['T(101,-102)',
                                              'f(1,2,3)',
                                              'T(103,-102,104)', 'Nc'])
        
        my_color_string2 = color.ColorString(['Nc', 'T(103,101,104)',
                                              'f(1,2,3)'])
        
        my_color_string1.simplify()
        
        self.assertEqual(my_color_string1, my_color_string2)
        
        my_color_string1 = color.ColorString(['T(101,102)',
                                              'f(1,2,3)',
                                              'T(103,104,101)'])
        
        my_color_string2 = color.ColorString(['T(103,104,102)',
                                              'f(1,2,3)'])
        
        my_color_string1.simplify()
        
        self.assertEqual(my_color_string1, my_color_string2)
        
        my_color_string1 = color.ColorString(['T(-101,102)',
                                              'f(1,2,3)',
                                              'T(103,104,-101)',
                                              'T(105,104)'])
        
        my_color_string2 = color.ColorString(['T(103,105,102)',
                                              'f(1,2,3)'])
        
        my_color_string1.simplify()
        
        self.assertEqual(my_color_string1, my_color_string2)
    
    def test_coeff_simplify(self):
        """Test color string coefficient simplification"""
        
        # Test Nc simplification
        my_color_string = color.ColorString(['Nc'] * 5 + \
                                            ['f(1,2,3)'] + \
                                            ['1/Nc'] * 3)
        
        my_color_string.simplify()
        
        self.assertEqual(my_color_string, color.ColorString(['Nc',
                                                             'Nc',
                                                             'f(1,2,3)']))
        
        # Test factors I simplification
        my_color_string = color.ColorString(['I'] * 4)
        my_color_string.simplify()
        self.assertEqual(my_color_string, color.ColorString([]))
        
        my_color_string = color.ColorString(['I'] * 5)
        my_color_string.simplify()
        self.assertEqual(my_color_string, color.ColorString(['I']))
        
        my_color_string = color.ColorString(['I'] * 6)
        my_color_string.simplify()
        self.assertEqual(my_color_string, color.ColorString(['-1']))
        
        my_color_string = color.ColorString(['I'] * 7)
        my_color_string.simplify()
        self.assertEqual(my_color_string, color.ColorString(['-1', 'I']))
        
        # Test numbers simplification
        my_color_string = color.ColorString(['-1/2', '2/3', '2', '-3'])
        my_color_string.simplify()
        self.assertEqual(my_color_string, color.ColorString(['2']))
        
        # Mix everything
        my_color_string = color.ColorString(['Nc', 'I', '-4', 'I', '1/Nc',
                                             'I', 'Nc', 'd(1,2,3)',
                                             '2/3', '-2/8'])
        my_color_string.simplify()
        self.assertEqual(my_color_string,
                         color.ColorString(['-2/3', 'I', 'Nc', 'd(1,2,3)']))
        
    def test_expand_composite(self):
        """Test color string expansion in the presence of terms like f, d, ...
        """
        
        my_color_string1 = color.ColorString(['T(1,2)',
                                              'd(3,4,-5)',
                                              'T(-5,6,7)'])
        
        my_color_string2 = color.ColorString(['T(1,2)',
                                              'f(3,4,5)',
                                              'T(5,6,7)'])
        
        self.assertEqual(my_color_string1.expand_composite_terms(),
                         [color.ColorString(['T(1,2)',
                                             '2',
                                             'T(3,-6,-7)',
                                             'T(4,-7,-8)',
                                             'T(-5,-8,-6)',
                                             'T(-5,6,7)']),
                          color.ColorString(['T(1,2)',
                                             '2',
                                             'T(-5,-6,-7)',
                                             'T(4,-7,-8)',
                                             'T(3,-8,-6)',
                                             'T(-5,6,7)'])])
        
        self.assertEqual(my_color_string2.expand_composite_terms(-4),
                         [color.ColorString(['T(1,2)',
                                             '-2',
                                             'I',
                                             'T(3,-4,-5)',
                                             'T(4,-5,-6)',
                                             'T(5,-6,-4)',
                                             'T(5,6,7)']),
                          color.ColorString(['T(1,2)',
                                             '2',
                                             'I',
                                             'T(5,-4,-5)',
                                             'T(4,-5,-6)',
                                             'T(3,-6,-4)',
                                             'T(5,6,7)'])])
        
    def test_golden_rule(self):
        """Test color string golden rule implementation"""
        
        my_color_string1 = color.ColorString(['T(1,2)',
                                              'T(3,4,5)',
                                              'd(6,-7,-8)',
                                              'T(3,-7,-8)'])
        
        self.assertEqual(my_color_string1.apply_golden_rule(),
                         [color.ColorString(['T(1,2)',
                                             '1/2',
                                             'T(4,-8)',
                                             'T(-7,5)',
                                             'd(6,-7,-8)']),
                          color.ColorString(['T(1,2)',
                                             '-1/2',
                                             '1/Nc',
                                             'T(4,5)',
                                             'T(-7,-8)',
                                             'd(6,-7,-8)'])])
        
        
