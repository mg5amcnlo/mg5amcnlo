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

"""Unit test library for the color algebra related routines 
in the core library"""

import copy
import fractions
import unittest

import madgraph.core.color_algebra as color
#
class ColorObjectTest(unittest.TestCase):
    """Test class for the ColorObject objects"""

    def test_standard(self):
        """Test the standard routines of ColorObject"""

        my_color_object = color.ColorObject(1, 2, -3)
        self.assertEqual([1, 2, -3], my_color_object.get_indices())

        my_color_object.set_indices(*[-1, 2, 3])
        self.assertEqual([-1, 2, 3], my_color_object.get_indices())

        my_color_object.append_index(4)
        self.assertEqual([-1, 2, 3, 4], my_color_object.get_indices())
        self.assertEqual('ColorObject(-1,2,3,4)', str(my_color_object))

    def test_error_raising(self):
        """Test type error raising for ColorObject"""

        not_an_integer = 'a'
        my_color_object = color.ColorObject(1, 2 , -3)

        self.assertRaises(TypeError,
                          my_color_object.append_index,
                          not_an_integer)

        self.assertRaises(TypeError,
                          my_color_object.set_indices,
                          not_an_integer)

    def test_Tr_simplify(self):
        """Test simplification of trace objects"""

        # Test Tr(a)=0
        col_str = color.ColorString()
        col_str.set_coeff(fractions.Fraction(0, 1))
        self.assertEqual(color.Tr(-1).simplify(), color.ColorFactor(col_str))

        # Test Tr()=Nc
        col_str = color.ColorString()
        col_str.set_Nc_power(1)
        self.assertEqual(color.Tr().simplify(), color.ColorFactor(col_str))

        # Test cyclicity
        col_str = color.ColorString(color.Tr(1, 2, 3, 4, 5))
        self.assertEqual(color.Tr(3, 4, 5, 1, 2).simplify(),
                         color.ColorFactor(col_str))

        # Tr(a,x,b,x,c) = 1/2(Tr(a,c)Tr(b)-1/Nc Tr(a,b,c))
        col_str1 = color.ColorString(color.Tr(1, 2, 4), color.Tr(3))
        col_str2 = color.ColorString(color.Tr(1, 2, 3, 4))
        col_str1.set_coeff(fractions.Fraction(1, 2))
        col_str2.set_coeff(fractions.Fraction(-1, 2))
        col_str2.set_Nc_power(-1)
        my_tr = color.Tr(1, 2, 100, 3, 100, 4)
        self.assertEqual(my_tr.simplify(),
                         color.ColorFactor(col_str1, col_str2))

        my_tr = color.Tr(1, 2, 100, 100, 4)
        col_str1.set_color_objects(color.Tr(1, 2, 4), color.Tr())
        col_str2.set_color_objects(color.Tr(1, 2, 4))
        self.assertEqual(my_tr.simplify(),
                         color.ColorFactor(col_str1, col_str2))

        my_tr = color.Tr(100, 100)
        col_str1.set_color_objects(color.Tr(), color.Tr())
        col_str2.set_color_objects(color.Tr())
        self.assertEqual(my_tr.simplify(),
                         color.ColorFactor(col_str1, col_str2))

    def test_Tr_pair_simplify(self):
        """Test Tr object product simplification"""

        my_Tr1 = color.Tr(1, 2, 3)
        my_Tr2 = color.Tr(4, 2, 5)
        my_T = color.T(4, 2, 5, 101, 102)

        col_str1 = color.ColorString(color.Tr(1, 5, 4, 3))
        col_str2 = color.ColorString(color.Tr(1, 3), color.Tr(4, 5))
        col_str1.set_coeff(fractions.Fraction(1, 2))
        col_str2.set_coeff(fractions.Fraction(-1, 2))
        col_str2.set_Nc_power(-1)
        self.assertEqual(my_Tr1.pair_simplify(my_Tr2),
                         color.ColorFactor(col_str1, col_str2))

        col_str1.set_color_objects(color.T(1, 5, 4, 3, 101, 102))
        col_str2.set_color_objects(color.Tr(1, 3), color.T(4, 5, 101, 102))
        self.assertEqual(my_Tr1.pair_simplify(my_T),
                         color.ColorFactor(col_str1, col_str2))

    def test_T_definition(self):
        """Test the T object definition and standard routines"""

        # T should have at least two indices!
        self.assertRaises(ValueError,
                         color.T,
                         1)

        # Test get and set for i j
        my_T = color.T(1, 2, 3, 100, 101)
        self.assertEqual(my_T.get_ij(), [100, 101])
        self.assertEqual(my_T.get_indices(), [1, 2, 3])

        # Test string representation
        self.assertEqual(str(my_T), 'T(1,2,3;100,101)')

    def test_T_simplify(self):
        """Test T simplify"""

        # T(a,b,c,...,i,i) = Tr(a,b,c,...)
        self.assertEqual(color.T(1, 2, 3, 100, 100).simplify(),
                         color.ColorFactor(\
                                    color.ColorString(color.Tr(1, 2, 3))))

        # T(a,x,b,x,c,i,j) = 1/2(T(a,c,i,j)Tr(b)-1/Nc T(a,b,c,i,j))
        my_T = color.T(1, 2, 100, 3, 100, 4, 101, 102)
        col_str1 = color.ColorString(color.T(1, 2, 4, 101, 102), color.Tr(3))
        col_str2 = color.ColorString(color.T(1, 2, 3, 4, 101, 102))
        col_str1.set_coeff(fractions.Fraction(1, 2))
        col_str2.set_coeff(fractions.Fraction(-1, 2))
        col_str2.set_Nc_power(-1)
        self.assertEqual(my_T.simplify(),
                         color.ColorFactor(col_str1, col_str2))
        self.assertEqual(my_T.simplify(),
                         color.ColorFactor(col_str1, col_str2))
    def test_T_pair_simplify(self):
        """Test T object products simplifications"""

        my_T1 = color.T(1, 2, 3, 101, 102)
        my_T2 = color.T(4, 5, 102, 103)
        self.assertEqual(my_T1.pair_simplify(my_T2),
                         color.ColorFactor(color.ColorString(\
                                        color.T(1, 2, 3, 4, 5, 101, 103))))

        my_T3 = color.T(4, 2, 5, 103, 104)
        col_str1 = color.ColorString(color.T(1, 5, 101, 104),
                                     color.T(4, 3, 103, 102))
        col_str2 = color.ColorString(color.T(1, 3, 101, 102),
                                     color.T(4, 5, 103, 104))
        col_str1.set_coeff(fractions.Fraction(1, 2))
        col_str2.set_coeff(fractions.Fraction(-1, 2))
        col_str2.set_Nc_power(-1)
        self.assertEqual(my_T1.pair_simplify(my_T3),
                         color.ColorFactor(col_str1, col_str2))

    def test_f_object(self):
        """Test the f color object"""
        # T should have exactly 3 indices!
        self.assertRaises(ValueError,
                         color.f,
                         1, 2, 3, 4)

        # Simplify should always return the same ColorFactor
        my_f = color.f(1, 2, 3)
        col_str1 = color.ColorString(color.Tr(1, 2, 3))
        col_str2 = color.ColorString(color.Tr(3, 2, 1))
        col_str1.set_coeff(fractions.Fraction(-2, 1))
        col_str2.set_coeff(fractions.Fraction(2, 1))
        col_str1.set_is_imaginary(True)
        col_str2.set_is_imaginary(True)

        self.assertEqual(my_f.simplify(),
                         color.ColorFactor(col_str1, col_str2))

    def test_d_object(self):
        """Test the d color object"""
        # T should have exactly 3 indices!
        self.assertRaises(ValueError,
                         color.d,
                         1, 2)

        # Simplify should always return the same ColorFactor
        my_d = color.d(1, 2, 3)
        col_str1 = color.ColorString(color.Tr(1, 2, 3))
        col_str2 = color.ColorString(color.Tr(3, 2, 1))
        col_str1.set_coeff(fractions.Fraction(2, 1))
        col_str2.set_coeff(fractions.Fraction(2, 1))

        self.assertEqual(my_d.simplify(),
                         color.ColorFactor(col_str1, col_str2))



class ColorStringTest(unittest.TestCase):
    """Test class for the ColorString objects"""

    my_col_string = color.ColorString()

    def setUp(self):
        """Initialize the ColorString test"""
        # Create a test color string
        self.my_col_string.set_coeff(fractions.Fraction(2, 3))
        self.my_col_string.set_Nc_power(-2)
        self.my_col_string.set_is_imaginary(True)

        test_f = color.f(1, 2, 3)
        test_d = color.d(4, 5, 6)
        self.my_col_string.set_color_objects(test_f, test_d)

    def test_error_raising(self):
        """Test type error raising for ColorString"""

        self.assertRaises(TypeError,
                          self.my_col_string.append_color_object,
                          'not a color object!')

        self.assertRaises(TypeError,
                          self.my_col_string.set_coeff,
                          0.1)

        self.assertRaises(TypeError,
                          self.my_col_string.set_Nc_power,
                          0.1)

        self.assertRaises(TypeError,
                          self.my_col_string.set_is_imaginary,
                          'true')

    def test_representation(self):
        """Test ColorString representation"""

        self.assertEqual(str(self.my_col_string),
                         "2/3 I 1/Nc^2 f(1,2,3) d(4,5,6)")

    def test_product(self):
        """Test the product of two color strings"""

        self.assertEqual(str(self.my_col_string.product(self.my_col_string)),
                         "-4/9 1/Nc^4 f(1,2,3) d(4,5,6) f(1,2,3) d(4,5,6)")

    def test_simplify(self):
        """Test the simplification of a string"""

        # Simplification of one term
        self.assertEqual(str(self.my_col_string.simplify()),
            '(4/3 1/Nc^2 Tr(1,2,3) d(4,5,6))+(-4/3 1/Nc^2 Tr(3,2,1) d(4,5,6))')

        # Simplification of two terms

class ColorFactorTest(unittest.TestCase):
    """Test class for the ColorFactor objects"""

    def test_gluons(self):
        """Test simplification of chains of f"""

        my_col_fact = color.ColorFactor(color.ColorString(\
                                    color.f(1, 2, -1),
                                    color.f(-1, 3, -2),
                                    color.f(-2, 4, -3),
                                    color.f(-3, 5, -4),
                                    color.f(-4, 6, -5),
                                    color.f(-5, 7, -6),
                                    color.f(-6, 8, -7),
                                    color.f(-7, 9, 10)))

        my_col_fact.full_simplify()
        print my_col_fact
##===============================================================================
## ColorStringTest
##===============================================================================
#class ColorStringTest(unittest.TestCase):
#    """Test class for code parts related to ColorString objects"""
#
#    def test_validity(self):
#        """Test the color string validity check"""
#
#        my_color_str = color.ColorString()
#
#        valid_strings = ["T(101,102)",
#                         "T(1,101,102)",
#                         "T(1,2,3,4,101,102)",
#                         "Tr()",
#                         "Tr(1,2,3,4)",
#                         "f(1,2,3)",
#                         "d(1,2,3)",
#                         "Nc", "1/Nc", "I",
#                         "0", "1", "-1", "1/2", "-123/345"]
#
#        for valid_str in valid_strings:
#            self.assert_(my_color_str.is_valid_color_object(valid_str))
#
#        wrong_strings = ["T(101,102",
#                         "T 1,101,102)",
#                         "T(1, 101, 102)",
#                         "k(1,2,3)",
#                         "d((1,2,3))",
#                         "d(1,2,3,)",
#                         "T(1.2)",
#                         "-2/Nc",
#                         'Tr(3,)']
#
#        for wrong_str in wrong_strings:
#            self.assertFalse(my_color_str.is_valid_color_object(wrong_str))
#
#    def test_init(self):
#        """Test the color string initialization"""
#
#        wrong_lists = [['T(101,102)', 1],
#                       ['T(101,102)', 'k(1,2,3)'],
#                       'T(101,102)']
#
#        for wrg_list in wrong_lists:
#            self.assertRaises(ValueError,
#                              color.ColorString,
#                              wrg_list)
#
#    def test_manip(self):
#        """Test the color string manipulation (append, insert and extend)"""
#
#        my_color_string = color.ColorString(['T(101,102)'])
#
#        self.assertRaises(ValueError,
#                          my_color_string.append,
#                          'k(1,2,3)')
#        self.assertRaises(ValueError,
#                          my_color_string.insert,
#                          0, 'k(1,2,3)')
#        self.assertRaises(ValueError,
#                          my_color_string.extend,
#                          ['k(1,2,3)'])
#
#    def test_T_traces(self):
#        """Test identity T(a,b,c,...,i,i) = Tr(a,b,c,...)"""
#
#        my_color_string = color.ColorString(['T(1,2,3,101,101)'])
#
#        my_color_string.simplify()
#
#        self.assertEqual(my_color_string,
#                         color.ColorString(['Tr(1,2,3)']))
#
#    def test_T_products(self):
#        """Test identity T(a,...,i,j)T(b,...,j,k) = T(a,...,b,...,i,k)"""
#
#        my_color_string = color.ColorString(['T(4,102,103)',
#                                             'T(1,2,3,101,102)',
#                                             'T(103,104)',
#                                             'T(5,6,104,105)'])
#
#        my_color_string.simplify()
#
#        self.assertEqual(my_color_string,
#                         color.ColorString(['T(1,2,3,4,5,6,101,105)']))
#
#    def test_simple_traces(self):
#        """Test identities Tr(1)=0, Tr()=Nc"""
#
#        my_color_string = color.ColorString(['Tr(1)'])
#
#        my_color_string.simplify()
#
#        self.assertEqual(my_color_string,
#                         color.ColorString())
#
#        my_color_string = color.ColorString(['Tr()'])
#
#        my_color_string.simplify()
#
#        self.assertEqual(my_color_string,
#                         color.ColorString(['Nc']))
#
#    def test_trace_cyclicity(self):
#        """Test trace cyclicity"""
#
#        my_color_string = color.ColorString(['Tr(5,2,3,4,1)'])
#
#        my_color_string.simplify()
#
#        self.assertEqual(my_color_string,
#                         color.ColorString(['Tr(1,5,2,3,4)']))
#
#    def test_coeff_simplify(self):
#        """Test color string coefficient simplification"""
#
#        # Test Nc simplification
#        my_color_string = color.ColorString(['Nc'] * 5 + \
#                                            ['f(1,2,3)'] + \
#                                            ['1/Nc'] * 3)
#
#        my_color_string.simplify()
#
#        self.assertEqual(my_color_string, color.ColorString(['Nc',
#                                                             'Nc',
#                                                             'f(1,2,3)']))
#
#        # Test factors I simplification
#        my_color_string = color.ColorString(['I'] * 4)
#        my_color_string.simplify()
#        self.assertEqual(my_color_string, color.ColorString([]))
#
#        my_color_string = color.ColorString(['I'] * 5)
#        my_color_string.simplify()
#        self.assertEqual(my_color_string, color.ColorString(['I']))
#
#        my_color_string = color.ColorString(['I'] * 6)
#        my_color_string.simplify()
#        self.assertEqual(my_color_string, color.ColorString(['-1']))
#
#        my_color_string = color.ColorString(['I'] * 7)
#        my_color_string.simplify()
#        self.assertEqual(my_color_string, color.ColorString(['-1', 'I']))
#
#        # Test numbers simplification
#        my_color_string = color.ColorString(['-1/2', '2/3', '2', '-3'])
#        my_color_string.simplify()
#        self.assertEqual(my_color_string, color.ColorString(['2']))
#
#        # Mix everything
#        my_color_string = color.ColorString(['Nc', 'I', '-4', 'I', '1/Nc',
#                                             'I', 'Nc', 'd(1,2,3)',
#                                             '2/3', '-2/8'])
#        my_color_string.simplify()
#        self.assertEqual(my_color_string,
#                         color.ColorString(['-2/3', 'I', 'Nc', 'd(1,2,3)']))
#
#    def test_expand_composite(self):
#        """Test color string expansion in the presence of terms like f, d, ...
#        """
#
#        my_color_string1 = color.ColorString(['T(1,2)',
#                                              'd(3,4,-5)',
#                                              'T(-5,6,7)'])
#
#        my_color_string2 = color.ColorString(['T(1,2)',
#                                              'f(3,4,5)',
#                                              'T(5,6,7)'])
#
#        self.assertEqual(my_color_string1.expand_composite_terms(),
#                         [color.ColorString(['T(1,2)',
#                                             '2',
#                                             'Tr(3,4,-5)',
#                                             'T(-5,6,7)']),
#                          color.ColorString(['T(1,2)',
#                                             '2',
#                                             'Tr(-5,4,3)',
#                                             'T(-5,6,7)'])])
#
#        self.assertEqual(my_color_string2.expand_composite_terms(),
#                         [color.ColorString(['T(1,2)',
#                                             '-2', 'I',
#                                             'Tr(3,4,5)',
#                                             'T(5,6,7)']),
#                          color.ColorString(['T(1,2)',
#                                             '2', 'I',
#                                             'Tr(5,4,3)',
#                                             'T(5,6,7)'])])
#
#
#    def test_expand_T_int_sum(self):
#        """Test color string expansion for T(a,x,b,x,c,i,j)"""
#
#        my_color_string = color.ColorString(['T(1,2)',
#                                              'T(1,2,101,3,101,4,5,6,102,103)',
#                                              'T(-5,6,7)'])
#
#        self.assertEqual(my_color_string.expand_T_internal_sum(),
#                        [color.ColorString(['T(1,2)',
#                                             '1/2', 'T(1,2,4,5,6,102,103)',
#                                             'Tr(3)',
#                                             'T(-5,6,7)']),
#                         color.ColorString(['T(1,2)',
#                                             '-1/2', '1/Nc',
#                                             'T(1,2,3,4,5,6,102,103)',
#                                             'T(-5,6,7)'])])
#
#        my_color_string = color.ColorString(['T(1,2)',
#                                              'T(101,101,102,103)',
#                                              'T(-5,6,7)'])
#
#        self.assertEqual(my_color_string.expand_T_internal_sum(),
#                        [color.ColorString(['T(1,2)',
#                                             '1/2', 'T(102,103)',
#                                             'Tr()',
#                                             'T(-5,6,7)']),
#                         color.ColorString(['T(1,2)',
#                                             '-1/2', '1/Nc',
#                                             'T(102,103)',
#                                             'T(-5,6,7)'])])
#
#    def test_expand_trace_int_sum(self):
#        """Test color string expansion for Tr(a,x,b,x,c)"""
#
#        my_color_string = color.ColorString(['T(1,2)',
#                                              'Tr(1,2,101,3,101,4,5,6)',
#                                              'T(-5,6,7)'])
#
#        self.assertEqual(my_color_string.expand_trace_internal_sum(),
#                        [color.ColorString(['T(1,2)',
#                                             '1/2', 'Tr(1,2,4,5,6)',
#                                             'Tr(3)',
#                                             'T(-5,6,7)']),
#                         color.ColorString(['T(1,2)',
#                                             '-1/2', '1/Nc',
#                                             'Tr(1,2,3,4,5,6)',
#                                             'T(-5,6,7)'])])
#
#        my_color_string = color.ColorString(['T(1,2)',
#                                              'Tr(1,2,101,101)',
#                                              'T(-5,6,7)'])
#
#        self.assertEqual(my_color_string.expand_trace_internal_sum(),
#                        [color.ColorString(['T(1,2)',
#                                             '1/2', 'Tr(1,2)',
#                                             'Tr()',
#                                             'T(-5,6,7)']),
#                         color.ColorString(['T(1,2)',
#                                             '-1/2', '1/Nc',
#                                             'Tr(1,2)',
#                                             'T(-5,6,7)'])])
#
#        my_color_string = color.ColorString(['Tr(-1,5,-100)'])
#        self.assertEqual(my_color_string.expand_trace_internal_sum(), [])
#
#    def test_expand_trace_product(self):
#        """Test color string expansion for Tr(a,x,b)Tr(c,x,d)"""
#
#        my_color_string = color.ColorString(['T(1,2)',
#                                              'Tr(1,101,2)', 'Tr(3,101,4)',
#                                              'T(-5,6,7)'])
#
#        self.assertEqual(my_color_string.expand_trace_product(),
#                        [color.ColorString(['T(1,2)',
#                                             '1/2', 'Tr(1,4,3,2)',
#                                             'T(-5,6,7)']),
#                         color.ColorString(['T(1,2)',
#                                             '-1/2', '1/Nc',
#                                             'Tr(1,2)', 'Tr(3,4)',
#                                             'T(-5,6,7)'])])
#
#        my_color_string = color.ColorString(['Tr(1,2,3)', 'Tr(1,2,3)'])
#
#        self.assertEqual(my_color_string.expand_trace_product(),
#                        [color.ColorString(['1/2', 'Tr(1,2,1,2)']),
#                         color.ColorString(['-1/2', '1/Nc',
#                                             'Tr(1,2)', 'Tr(1,2)'])])
#
#        my_color_string = color.ColorString(['Tr(1,2,3)', 'Tr(3,2,1)'])
#
#        self.assertEqual(my_color_string.expand_trace_product(),
#                        [color.ColorString(['1/2', 'Tr(1,2,2,1)']),
#                         color.ColorString(['-1/2', '1/Nc',
#                                             'Tr(1,2)', 'Tr(2,1)'])])
#
#    def test_expand_trace_T_product(self):
#        """Test color string expansion for Tr(a,x,b)T(c,x,d,i,j)"""
#
#        my_color_string = color.ColorString(['T(1,2)',
#                                              'Tr(1,101,2)',
#                                              'T(3,101,4,102,103)',
#                                              'T(-5,6,7)'])
#
#        self.assertEqual(my_color_string.expand_trace_T_product(),
#                        [color.ColorString(['T(1,2)',
#                                             '1/2', 'T(3,2,1,4,102,103)',
#                                             'T(-5,6,7)']),
#                         color.ColorString(['T(1,2)',
#                                             '-1/2', '1/Nc',
#                                             'Tr(1,2)', 'T(3,4,102,103)',
#                                             'T(-5,6,7)'])])
#
#        my_color_string = color.ColorString(['T(1,2)',
#                                              'Tr(101,2)',
#                                              'T(3,101,102,103)',
#                                              'T(-5,6,7)'])
#
#        self.assertEqual(my_color_string.expand_trace_T_product(),
#                        [color.ColorString(['T(1,2)',
#                                             '1/2', 'T(3,2,102,103)',
#                                             'T(-5,6,7)']),
#                         color.ColorString(['T(1,2)',
#                                             '-1/2', '1/Nc',
#                                             'Tr(2)', 'T(3,102,103)',
#                                             'T(-5,6,7)'])])
#
#    def test_expand_T_product(self):
#        """Test color string expansion for T(a,x,b,i,j)T(c,x,d,k,l)"""
#
#        my_color_string = color.ColorString(['T(3,101,4,102,103)',
#                                              'T(5,101,6,104,105)'])
#
#        self.assertEqual(my_color_string.expand_T_product(),
#                        [color.ColorString(['1/2', 'T(3,6,102,105)',
#                                             'T(5,4,104,103)']),
#                         color.ColorString(['-1/2', '1/Nc',
#                                             'T(3,4,102,103)',
#                                             'T(5,6,104,105)'])])
#
#    def test_similarity(self):
#        """Test color string similarity"""
#
#        my_color_string = color.ColorString(['-1/3', 'T(3,101,4,102,103)',
#                                              'T(5,101,6,104,105)'])
#
#        my_color_string1 = color.ColorString(['1/2', 'T(3,101,4,102,103)',
#                                              'T(5,101,6,104,105)'])
#
#        my_color_string2 = color.ColorString(['1/2', 'I', 'T(3,101,4,102,103)',
#                                              'T(5,101,6,104,105)'])
#
#        my_color_string3 = color.ColorString(['1/2', 'T(7,101,4,102,103)',
#                                              'T(5,101,6,104,105)'])
#
#        self.assertTrue(my_color_string.is_similar(my_color_string1,
#                                                   check_I=True))
#
#        self.assertTrue(my_color_string.is_similar(my_color_string2,
#                                                   check_I=False))
#
#        self.assertFalse(my_color_string.is_similar(my_color_string3,
#                                                   check_I=False))
#
#    def test_extract_coeff(self):
#        """Test coefficient extraction for color string"""
#
#        my_color_string = color.ColorString(['1/2', 'I', 'T(3,101,4,102,103)',
#                                              'T(5,101,6,104,105)'])
#
#        self.assertEqual(my_color_string.extract_coeff()[0],
#                         color.ColorString(['1/2', 'I']))
#
#        self.assertEqual(my_color_string.extract_coeff()[1],
#                         color.ColorString(['T(3,101,4,102,103)',
#                                              'T(5,101,6,104,105)']))
#
#    def test_complex_conjugate(self):
#        """Test the complex conjugation of a color string"""
#
#        my_color_string = color.ColorString(['I', 'T(3,4,102,103)',
#                                              'Tr(1,2,3)'])
#
#        self.assertEqual(my_color_string.complex_conjugate(),
#                         color.ColorString(['-1', 'I',
#                                            'T(4,3,103,102)', 'Tr(1,3,2)']))
#
#class ColorFactorTest(unittest.TestCase):
#    """Test class for code parts related to ColorFactor objects"""
#
#    def test_colorfactor_init(self):
#        """Test the color factor initialization"""
#
#        wrong_lists = [1, ['T(101,102)', 'k(1,2,3)']]
#
#        for wrg_list in wrong_lists:
#            self.assertRaises(ValueError,
#                              color.ColorFactor,
#                              wrg_list)
#
#    def test_colorfactor_manip(self):
#        """Test the color factor manipulation (append, insert and extend)"""
#
#        my_color_factor = color.ColorFactor([color.ColorString(['T(101,102)'])])
#
#        self.assertRaises(ValueError,
#                          my_color_factor.append,
#                          1)
#        self.assertRaises(ValueError,
#                          my_color_factor.insert,
#                          0, 1)
#        self.assertRaises(ValueError,
#                          my_color_factor.extend,
#                          [1])
#
#    def test_basic_simplify(self):
#        """Test the color factor simplify algorithm on basic identities"""
#
#        my_color_factor = color.ColorFactor([color.ColorString(['1/4',
#                                                                'd(1,2,3)'
#                                                                ]),
#                                             color.ColorString(['1/4', 'I',
#                                                                'f(1,2,3)'
#                                                                ])])
#
#        my_color_factor.simplify()
#        self.assertEqual(my_color_factor, color.ColorFactor(\
#                                    [color.ColorString(['Tr(1,2,3)'])]))
#
#        my_color_factor = color.ColorFactor([color.ColorString(['f(1,2,3)',
#                                                                'f(1,2,3)'
#                                                                ])])
#
#        my_color_factor.simplify()
#        self.assert_(my_color_factor in \
#                         [color.ColorFactor(list(x)) for x in \
#                          itertools.permutations(\
#                                    [color.ColorString(['-1', 'Nc']),
#                                     color.ColorString(['Nc', 'Nc', 'Nc'])])])
#
#        my_color_factor = color.ColorFactor([color.ColorString(['f(1,2,3)',
#                                                                'd(1,2,3)',
#                                                                'f(5,6,7)',
#                                                                'd(5,6,7)'
#                                                                ])])
#
#        my_color_factor.simplify()
#        self.assertEqual(my_color_factor, color.ColorFactor([]))
#
#
#        my_color_factor = color.ColorFactor([color.ColorString(['d(1,2,3)',
#                                                                'd(1,2,3)'
#                                                                ])])
#        my_color_factor.simplify()
#        self.assert_(my_color_factor in [
#                    color.ColorFactor(list(x)) for x in itertools.permutations(\
#                                    [color.ColorString(['-5', 'Nc']),
#                                     color.ColorString(['4', '1/Nc']),
#                                     color.ColorString(['Nc', 'Nc', 'Nc'])])])
#
#        my_color_factor = color.ColorFactor([color.ColorString(['f(1,2,-1)',
#                                                                'f(-1,3,-2)',
#                                                                'f(-2,4,5)'
#                                                                ])])
#        my_color_factor.simplify()
#        self.assert_(my_color_factor in [
#                    color.ColorFactor(list(x)) for x in itertools.permutations(\
#                                    [color.ColorString(['2', 'I',
#                                                        'Tr(1,2,3,4,5)']),
#                                    color.ColorString(['-2', 'I',
#                                                        'Tr(1,2,3,5,4)']),
#                                    color.ColorString(['-2', 'I',
#                                                        'Tr(1,2,4,5,3)']),
#                                    color.ColorString(['2', 'I',
#                                                        'Tr(1,2,5,4,3)']),
#                                    color.ColorString(['-2', 'I',
#                                                        'Tr(1,3,4,5,2)']),
#                                    color.ColorString(['2', 'I',
#                                                        'Tr(1,3,5,4,2)']),
#                                    color.ColorString(['2', 'I',
#                                                        'Tr(1,4,5,3,2)']),
#                                    color.ColorString(['-2', 'I',
#                                                        'Tr(1,5,4,3,2)'])])])
#
#        my_color_factor = color.ColorFactor([color.ColorString(['Tr(1,2,3,4,5,6,7)',
#                                                                'Tr(1,7,6,5,4,3,2)'
#                                                                ])])
#        my_color_factor.simplify()
#        self.assertEqual(my_color_factor,
#                    color.ColorFactor([color.ColorString(['1/128', 'Nc', 'Nc', 'Nc', 'Nc', 'Nc', 'Nc', 'Nc']),
#                                     color.ColorString(['-7/128', 'Nc', 'Nc', 'Nc', 'Nc', 'Nc']),
#                                     color.ColorString(['21/128', 'Nc', 'Nc', 'Nc']),
#                                     color.ColorString(['-35/128', 'Nc']),
#                                     color.ColorString(['35/128', '1/Nc']),
#                                     color.ColorString(['-21/128', '1/Nc', '1/Nc', '1/Nc']),
#                                     color.ColorString(['3/64', '1/Nc', '1/Nc', '1/Nc', '1/Nc', '1/Nc'])]))
#
#
#
##        my_color_factor = color.ColorFactor([color.ColorString(['f(1,2,-1)',
##                                                        'f(-1,3,-2)',
##                                                        'f(-2,4,-3)',
##                                                        'f(-3,5,-4)',
##                                                        'f(-4,6,-5)',
##                                                        'f(-5,7,8)'
##                                                      #  'f(-6,8,-7)',
##                                                      #  'f(-7,9,10)'
##                                                        ])])
##        my_color_factor.simplify()

