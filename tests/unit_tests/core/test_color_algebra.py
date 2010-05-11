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

        my_color_object = color.ColorObject(-1, 2, 3)
        my_color_object.append(4)

        self.assertEqual('ColorObject(-1,2,3,4)', str(my_color_object))


    def test_Tr_simplify(self):
        """Test simplification of trace objects"""

        # Test Tr(a)=0
        self.assertEqual(color.Tr(-1).simplify(),
                         color.ColorFactor([color.ColorString(coeff=0)]))

        # Test Tr()=Nc
        col_str = color.ColorString()
        col_str.Nc_power = 1
        self.assertEqual(color.Tr().simplify(), color.ColorFactor([col_str]))

        # Test cyclicity
        col_str = color.ColorString([color.Tr(1, 2, 3, 4, 5)])
        self.assertEqual(color.Tr(3, 4, 5, 1, 2).simplify(),
                         color.ColorFactor([col_str]))

        # Tr(a,x,b,x,c) = 1/2(Tr(a,c)Tr(b)-1/Nc Tr(a,b,c))
        col_str1 = color.ColorString([color.Tr(1, 2, 4), color.Tr(3)])
        col_str2 = color.ColorString([color.Tr(1, 2, 3, 4)])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(-1, 2)
        col_str2.Nc_power = -1
        my_tr = color.Tr(1, 2, 100, 3, 100, 4)
        self.assertEqual(my_tr.simplify(),
                         color.ColorFactor([col_str1, col_str2]))

        my_tr = color.Tr(1, 2, 100, 100, 4)

        col_str1 = color.ColorString([color.Tr(1, 2, 4), color.Tr()])
        col_str2 = color.ColorString([color.Tr(1, 2, 4)])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(-1, 2)
        col_str2.Nc_power = -1
        self.assertEqual(my_tr.simplify(),
                         color.ColorFactor([col_str1, col_str2]))

        my_tr = color.Tr(100, 100)
        col_str1 = color.ColorString([color.Tr(), color.Tr()])
        col_str2 = color.ColorString([color.Tr()])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(-1, 2)
        col_str2.Nc_power = -1
        self.assertEqual(my_tr.simplify(),
                         color.ColorFactor([col_str1, col_str2]))

    def test_Tr_pair_simplify(self):
        """Test Tr object product simplification"""

        my_Tr1 = color.Tr(1, 2, 3)
        my_Tr2 = color.Tr(4, 2, 5)
        my_T = color.T(4, 2, 5, 101, 102)

        col_str1 = color.ColorString([color.Tr(1, 5, 4, 3)])
        col_str2 = color.ColorString([color.Tr(1, 3), color.Tr(4, 5)])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(-1, 2)
        col_str2.Nc_power = -1
        self.assertEqual(my_Tr1.pair_simplify(my_Tr2),
                         color.ColorFactor([col_str1, col_str2]))

        col_str1 = color.ColorString([color.T(4, 3, 1, 5, 101, 102)])
        col_str2 = color.ColorString([color.Tr(1, 3), color.T(4, 5, 101, 102)])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(-1, 2)
        col_str2.Nc_power = -1
        self.assertEqual(my_Tr1.pair_simplify(my_T),
                         color.ColorFactor([col_str1, col_str2]))


    def test_T_simplify(self):
        """Test T simplify"""

        # T(a,b,c,...,i,i) = Tr(a,b,c,...)
        self.assertEqual(color.T(1, 2, 3, 100, 100).simplify(),
                         color.ColorFactor([\
                                    color.ColorString([color.Tr(1, 2, 3)])]))

        # T(a,x,b,x,c,i,j) = 1/2(T(a,c,i,j)Tr(b)-1/Nc T(a,b,c,i,j))
        my_T = color.T(1, 2, 100, 3, 100, 4, 101, 102)
        col_str1 = color.ColorString([color.T(1, 2, 4, 101, 102), color.Tr(3)])
        col_str2 = color.ColorString([color.T(1, 2, 3, 4, 101, 102)])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(-1, 2)
        col_str2.Nc_power = -1
        self.assertEqual(my_T.simplify(),
                         color.ColorFactor([col_str1, col_str2]))
        self.assertEqual(my_T.simplify(),
                         color.ColorFactor([col_str1, col_str2]))
    def test_T_pair_simplify(self):
        """Test T object products simplifications"""

        my_T1 = color.T(1, 2, 3, 101, 102)
        my_T2 = color.T(4, 5, 102, 103)
        self.assertEqual(my_T1.pair_simplify(my_T2),
                         color.ColorFactor([color.ColorString([\
                                        color.T(1, 2, 3, 4, 5, 101, 103)])]))

        my_T3 = color.T(4, 2, 5, 103, 104)
        col_str1 = color.ColorString([color.T(1, 5, 101, 104),
                                     color.T(4, 3, 103, 102)])
        col_str2 = color.ColorString([color.T(1, 3, 101, 102),
                                     color.T(4, 5, 103, 104)])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(-1, 2)
        col_str2.Nc_power = -1
        self.assertEqual(my_T1.pair_simplify(my_T3),
                         color.ColorFactor([col_str1, col_str2]))

    def test_f_object(self):
        """Test the f color object"""
        # T should have exactly 3 indices!
        self.assertRaises(ValueError,
                         color.f,
                         1, 2, 3, 4)

        # Simplify should always return the same ColorFactor
        my_f = color.f(1, 2, 3)
        col_str1 = color.ColorString([color.Tr(1, 2, 3)])
        col_str2 = color.ColorString([color.Tr(3, 2, 1)])
        col_str1.coeff = fractions.Fraction(-2, 1)
        col_str2.coeff = fractions.Fraction(2, 1)
        col_str1.is_imaginary = True
        col_str2.is_imaginary = True

        self.assertEqual(my_f.simplify(),
                         color.ColorFactor([col_str1, col_str2]))

    def test_d_object(self):
        """Test the d color object"""
        # T should have exactly 3 indices!
        self.assertRaises(ValueError,
                         color.d,
                         1, 2)

        # Simplify should always return the same ColorFactor
        my_d = color.d(1, 2, 3)
        col_str1 = color.ColorString([color.Tr(1, 2, 3)])
        col_str2 = color.ColorString([color.Tr(3, 2, 1)])
        col_str1.coeff = fractions.Fraction(2, 1)
        col_str2.coeff = fractions.Fraction(2, 1)

        self.assertEqual(my_d.simplify(),
                         color.ColorFactor([col_str1, col_str2]))


class ColorStringTest(unittest.TestCase):
    """Test class for the ColorString objects"""

    my_col_string = color.ColorString()

    def setUp(self):
        """Initialize the ColorString test"""
        # Create a test color string

        test_f = color.f(1, 2, 3)
        test_d = color.d(4, 5, 6)

        self.my_col_string = color.ColorString([test_f, test_d],
                                               coeff=fractions.Fraction(2, 3),
                                               Nc_power= -2,
                                               is_imaginary=True)

    def test_representation(self):
        """Test ColorString representation"""

        self.assertEqual(str(self.my_col_string),
                         "2/3 I 1/Nc^2 f(1,2,3) d(4,5,6)")

    def test_product(self):
        """Test the product of two color strings"""
        test = copy.copy(self.my_col_string)
        test.product(self.my_col_string)
        self.assertEqual(str(test),
                         "-4/9 1/Nc^4 f(1,2,3) d(4,5,6) f(1,2,3) d(4,5,6)")


    def test_simplify(self):
        """Test the simplification of a string"""

        # Simplification of one term
        self.assertEqual(str(self.my_col_string.simplify()),
            '(4/3 1/Nc^2 Tr(1,2,3) d(4,5,6))+(-4/3 1/Nc^2 Tr(3,2,1) d(4,5,6))')

    def test_complex_conjugate(self):
        """Test the complex conjugation of a color string"""

        my_color_string = color.ColorString([color.T(3, 4, 102, 103),
                                             color.Tr(1, 2, 3)])
        my_color_string.is_imaginary = True

        self.assertEqual(str(my_color_string.complex_conjugate()),
                         '-1 I T(4,3,103,102) Tr(3,2,1)')

    def test_to_immutable(self):
        """Test the immutable representation of a color string structure"""

        self.assertEqual(self.my_col_string.to_immutable(),
                         (('d', (4, 5, 6)), ('f', (1, 2, 3))))

    def test_from_immutable(self):
        """Test the creation of a color string using its immutable rep"""

        test_str = copy.copy(self.my_col_string)
        test_str.from_immutable((('f', (1, 2, 3)), ('d', (4, 5, 6))))

        self.assertEqual(test_str, self.my_col_string)

    def test_replace_indices(self):
        """Test indices replacement"""

        repl_dict = {1:2, 2:3, 3:1}

        my_color_string = color.ColorString([color.T(1, 2, 3, 4),
                                             color.Tr(3, 2, 1)])

        my_color_string.replace_indices(repl_dict)
        self.assertEqual(str(my_color_string),
                         '1 T(2,3,1,4) Tr(1,3,2)')
        inv_repl_dict = dict([v, k] for k, v in repl_dict.items())
        my_color_string.replace_indices(inv_repl_dict)
        self.assertEqual(str(my_color_string),
                         '1 T(1,2,3,4) Tr(3,2,1)')

    def test_color_string_canonical(self):
        """Test the canonical representation of a immutable color string"""

        immutable1 = (('f', (2, 3, 4)), ('T', (4, 2, 5)))
        immutable2 = (('T', (3, 5)),)

        self.assertEqual(color.ColorString().to_canonical(immutable1 + \
                                                               immutable2)[0],
                         (('T', (2, 4)), ('T', (3, 1, 4)), ('f', (1, 2, 3))))

        self.assertEqual(color.ColorString().to_canonical(immutable1 + \
                                                               immutable2)[1],
                         {3:2, 5:4, 4:3, 2:1})

class ColorFactorTest(unittest.TestCase):
    """Test class for the ColorFactor objects"""

    def test_f_d_sum(self):
        """Test f and d sum with the right weights giving a Tr"""

        col_str1 = color.ColorString([color.d(1, 2, 3)])
        col_str1.coeff = fractions.Fraction(1, 4)
        col_str2 = color.ColorString([color.f(1, 2, 3)])
        col_str2.coeff = fractions.Fraction(1, 4)
        col_str2.is_imaginary = True

        my_color_factor = color.ColorFactor([col_str1, col_str2])

        self.assertEqual(str(my_color_factor.full_simplify()),
                         '(1 Tr(1,2,3))')

    def test_f_product(self):
        """Test the fully contracted product of two f's"""

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.f(1, 2, 3), color.f(1, 2, 3)])])

        self.assertEqual(str(my_color_factor.full_simplify()),
                         '(-1 Nc^1 )+(1 Nc^3 )')


    def test_d_product(self):
        """Test the fully contracted product of two d's"""

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.d(1, 2, 3), color.d(1, 2, 3)])])


        self.assertEqual(str(my_color_factor.full_simplify()),
                         '(-5 Nc^1 )+(4 1/Nc^1 )+(1 Nc^3 )')

    def test_f_d_product(self):
        """Test the fully contracted product of f and d"""

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.f(1, 2, 3), color.d(1, 2, 3)])])


        self.assertEqual(str(my_color_factor.full_simplify()), '')

    def test_three_f_chain(self):
        """Test a chain of three f's"""

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.f(1, 2, -1),
                                       color.f(-1, 3, -2),
                                       color.f(-2, 4, 5)])])

        self.assertEqual(str(my_color_factor.full_simplify()),
        "(2 I Tr(1,2,3,4,5))+(-2 I Tr(1,2,4,5,3))+(-2 I Tr(1,2,3,5,4))" + \
        "+(2 I Tr(1,2,5,4,3))+(-2 I Tr(1,3,4,5,2))+(2 I Tr(1,4,5,3,2))" + \
        "+(2 I Tr(1,3,5,4,2))+(-2 I Tr(1,5,4,3,2))")

    def test_Tr_product(self):
        """Test a non trivial product of two traces"""

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.Tr(1, 2, 3, 4, 5, 6, 7),
                                       color.Tr(1, 7, 6, 5, 4, 3, 2)])])

        self.assertEqual(str(my_color_factor.full_simplify()),
        "(1/128 Nc^7 )+(-7/128 Nc^5 )+(21/128 Nc^3 )+(-35/128 Nc^1 )" + \
        "+(35/128 1/Nc^1 )+(-21/128 1/Nc^3 )+(3/64 1/Nc^5 )")

    def test_T_f_product(self):
        """Test a non trivial T f f product"""

        my_color_factor = color.ColorFactor([\
                                    color.ColorString([color.T(-1000, 1, 2),
                                               color.f(-1, -1000, 5),
                                               color.f(-1, 4, 3)])])

        self.assertEqual(str(my_color_factor.full_simplify()),
        "(-1 T(5,4,3,1,2))+(1 T(5,3,4,1,2))+(1 T(4,3,5,1,2))+(-1 T(3,4,5,1,2))")


    def test_gluons(self):
        """Test simplification of chains of f"""

        my_col_fact = color.ColorFactor([color.ColorString([color.f(-3, 1, 2),
                                    color.f(-1, 3, 4),
                                    color.f(-1, 5, -3)
                                    ])])

        self.assertEqual(str(my_col_fact.full_simplify()),
        '(2 I Tr(1,2,3,4,5))+(-2 I Tr(1,2,5,3,4))+(-2 I Tr(1,2,4,3,5))+' + \
        '(2 I Tr(1,2,5,4,3))+(-2 I Tr(1,3,4,5,2))+(2 I Tr(1,5,3,4,2))+' + \
        '(2 I Tr(1,4,3,5,2))+(-2 I Tr(1,5,4,3,2))')


