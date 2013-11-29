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

"""Unit test library for the color algebra related routines 
in the core library"""

import copy
import fractions

import madgraph.core.color_algebra as color
import tests.unit_tests as unittest

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
        self.assertRaises(AssertionError,
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
        self.assertRaises(AssertionError,
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

    def test_epsilon_object(self):
        """Test the epsilon object"""

        # Espilon should have exactly 3 indices!
        self.assertRaises(AssertionError,
                         color.Epsilon,
                         1, 2)

        my_epsilon1 = color.Epsilon(2, 3, 1)
        my_epsilon2 = color.Epsilon(5, 1, 4)
        my_epsilon2 = my_epsilon2.complex_conjugate()

        my_goal_str1 = color.ColorString([color.T(2, 4), color.T(3, 5)])
        my_goal_str2 = color.ColorString([color.T(2, 5), color.T(3, 4)])
        my_goal_str2.coeff = fractions.Fraction(-1
                                              )
        self.assertEqual(my_epsilon1.pair_simplify(my_epsilon2),
                         color.ColorFactor([my_goal_str1, my_goal_str2]))

    def test_delta3_pair_simplify(self):
        """Test delta3 simplify"""

        self.assertEqual(color.K6(1,101,103).pair_simplify(color.T(101,102)),
                         color.ColorFactor([color.ColorString([color.K6(1,103,102)])]))
        self.assertEqual(color.K6(1,103,102).pair_simplify(color.T(102,101)),
                         color.ColorFactor([color.ColorString([color.K6(1,103,101)])]))
        self.assertEqual(color.K6Bar(1,101,103).pair_simplify(color.T(102,101)),
                         color.ColorFactor([color.ColorString([color.K6Bar(1,103,102)])]))
        self.assertEqual(color.K6Bar(1,103,101).pair_simplify(color.T(102,101)),
                         color.ColorFactor([color.ColorString([color.K6Bar(1,103,102)])]))

    def test_delta6_simplify(self):
        """Test delta6 simplify"""

        # delta6(i,i) = 1
        col_str1 = color.ColorString()
        col_str1.Nc_power = 2
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2 = color.ColorString()
        col_str2.Nc_power = 1
        col_str2.coeff = fractions.Fraction(1, 2)
        self.assertEqual(color.T6(1, 1).simplify(),
                         color.ColorFactor([col_str1, col_str2]))

    def test_K6_objects(self):
        """Test K6 product simplifications"""

        #K6(m,i,j)K6Bar(m,k,l) = 1/2(T(l,i)T(k,j)
        #                          + T(k,i)T(l,j)

        my_K6 = color.K6(1,101,102)
        my_K6Bar = color.K6Bar(1,103,104)

        col_str1 = color.ColorString([color.T(104,101),
                                      color.T(103,102)])
        col_str2 = color.ColorString([color.T(103,101),
                                      color.T(104,102)])
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2.coeff = fractions.Fraction(1, 2)

        self.assertEqual(my_K6.pair_simplify(my_K6Bar),
                         color.ColorFactor([col_str1, col_str2]))

        #K6(m,i,j)K6Bar(n,j,i) = delta6(m,n)

        my_K6 = color.K6(1,101,102)
        my_K6Bar = color.K6Bar(2,102,101)

        self.assertEqual(my_K6.pair_simplify(my_K6Bar),
                         color.ColorFactor([\
                         color.ColorString([color.T6(1,2)])]))

        #K6(m,i,j)K6Bar(n,i,j) = delta6(m,n).
        my_K6 = color.K6(1,101,102)
        my_K6Bar = color.K6Bar(2,101,102)

        self.assertEqual(my_K6.pair_simplify(my_K6Bar),
                         color.ColorFactor([\
                         color.ColorString([color.T6(1,2)])]))


    def test_T6_simplify(self):
        """Test T6 simplify"""

        # T6(a,i,j) = 2(K6(i,ii,jj)T(a,jj,kk)K6Bar(j,kk,ii))

        my_T6 = color.T6(1,101,102)

        color.T6.new_index = 10000

        k6 = color.K6(101, 10000, 10001)
        t = color.T(1, 10001, 10002)
        k6b = color.K6Bar(102, 10002, 10000)
        col_string = color.ColorString([k6, t, k6b])
        col_string.coeff = fractions.Fraction(2, 1)
        self.assertEqual(my_T6.simplify(), color.ColorFactor([col_string]))

        my_T6 = color.T6(1,101,102)

        k6 = color.K6(101, 10003, 10004)
        t = color.T(1, 10004, 10005)
        k6b = color.K6Bar(102, 10005, 10003)
        col_string = color.ColorString([k6, t, k6b])
        col_string.coeff = fractions.Fraction(2, 1)
        self.assertEqual(my_T6.simplify(), color.ColorFactor([col_string]))
        
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


    def test_sextet_products(self):
        """Test non trivial product of sextet operators"""

        # T6[2, 101, 102] T6[2, 102, 103] = (-1 + Nc) (2 + Nc) delta6[101, 103])/Nc

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.T6(2, 101, 102),
                                       color.T6(2, 102, 103)])])

        col_str1 = color.ColorString([color.T6(101,103)])
        col_str1.Nc_power = 1
        col_str2 = copy.copy(col_str1)
        col_str2.Nc_power = 0
        col_str3 = copy.copy(col_str1)
        col_str3.Nc_power = -1
        col_str3.coeff = fractions.Fraction(-2, 1)

        self.assertEqual(my_color_factor.full_simplify(),
                         color.ColorFactor([col_str1, col_str2, col_str3]))

        # T6[2, 101, 102] T6[3, 102, 101] = 1/2 (2 + Nc) delta8[2, 3]

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.T6(2, 101, 102),
                                       color.T6(3, 102, 101)])])

        col_str1 = color.ColorString([color.Tr(2,3)])
        col_str1.Nc_power = 1
        col_str1.coeff = fractions.Fraction(1)
        col_str2 = copy.copy(col_str1)
        col_str2.Nc_power = 0
        col_str2.coeff = fractions.Fraction(2)

        self.assertEqual(my_color_factor.full_simplify(),
                         color.ColorFactor([col_str1, col_str2]))

        
        # K6[1, 101, 102] T[2, 102, 103] T[2, 103, 104] K6Bar[1, 104, 101]
        #                 = 1/4 (-1 + Nc) (1 + Nc)^2
        #                 = 1/4 (-1 - Nc + Nc^2 + Nc^3)

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.K6(1, 101, 102),
                                       color.T(2, 102, 103),
                                       color.T(2, 103, 104),
                                       color.K6Bar(1, 104, 101)])])

        col_str1 = color.ColorString()
        col_str1.Nc_power = 3
        col_str1.coeff = fractions.Fraction(1, 4)
        col_str2 = color.ColorString()
        col_str2.Nc_power = 2
        col_str2.coeff = fractions.Fraction(1, 4)
        col_str3 = color.ColorString()
        col_str3.Nc_power = 1
        col_str3.coeff = fractions.Fraction(-1, 4)
        col_str4 = color.ColorString()
        col_str4.Nc_power = 0
        col_str4.coeff = fractions.Fraction(-1, 4)

        self.assertEqual(my_color_factor.full_simplify(),
                         color.ColorFactor([col_str1, col_str3,
                                            col_str2, col_str4]))

        # T6[2, 101, 102] T6[2, 102, 103] K6[103, 99, 98] K6Bar[101, 98, 99]
        #                 = 1/2 (-1 + Nc) (1 + Nc) (2 + Nc)
        #                 = 1/2 (Nc^3 + 2 Nc^2 - Nc - 2)

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.T6(2, 101, 102),
                                       color.T6(2, 102, 103),
                                       color.K6(103,99, 98),
                                       color.K6Bar(101, 98, 99)])])

        col_str1 = color.ColorString()
        col_str1.Nc_power = 3
        col_str1.coeff = fractions.Fraction(1, 2)
        col_str2 = color.ColorString()
        col_str2.Nc_power = 2
        col_str2.coeff = fractions.Fraction(1, 1)
        col_str3 = color.ColorString()
        col_str3.Nc_power = 1
        col_str3.coeff = fractions.Fraction(-1, 2)
        col_str4 = color.ColorString()
        col_str4.Nc_power = 0
        col_str4.coeff = fractions.Fraction(-1, 1)

        self.assertEqual(my_color_factor.full_simplify(),
                         color.ColorFactor([col_str2, col_str1, 
                                            col_str3, col_str4]))

        # K6[103, 99, 98] T[80, 98, 100] K6Bar[103, 100, 97] T[80, 99, 97]
        #                 = -(1/4) + Nc^2/4

        my_color_factor = color.ColorFactor([\
                    color.ColorString([color.K6(103, 99, 98),
                                       color.T(80, 98, 100),
                                       color.K6Bar(103, 100, 97),
                                       color.T(80, 99, 97)])])

        col_str1 = color.ColorString()
        col_str1.Nc_power = 2
        col_str1.coeff = fractions.Fraction(1, 4)
        col_str2 = color.ColorString()
        col_str2.Nc_power = 0
        col_str2.coeff = fractions.Fraction(-1, 4)

        self.assertEqual(my_color_factor.full_simplify(),
                         color.ColorFactor([col_str1, col_str2]))

