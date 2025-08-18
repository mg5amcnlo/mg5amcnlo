################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Test the validity of the LHE parser"""

from __future__ import absolute_import
import unittest
import madgraph.various.lhe_parser as lhe_parser
import madgraph .various.misc as misc
import tempfile
import os
import shutil
import math
from six.moves import zip
pjoin = os.path.join
from madgraph import MG5DIR
import itertools


class TestEvent(unittest.TestCase):


    def test_equiv_sequence(self):
        """ check the equiv_sequence: staticmethod"""

        l1 = [3,4,5]
        mapping = {0: "a", 1: "b", 2: "c"}
        for l2 in itertools.permutations(l1):
            out = lhe_parser.Event.equiv_sequence(l1,l2, mapping)
            if tuple(l1) == l2:
                self.assertTrue(out)
            else:
                self.assertFalse(out)

        l1 = [3,4,5]
        mapping = {0: "a", 1: "a", 2: "c"}
        l2 =[4,3,5]
        out = lhe_parser.Event.equiv_sequence(l1,l2, mapping)
        self.assertTrue(out)
        l2 =[4,5,3]
        out = lhe_parser.Event.equiv_sequence(l1,l2, mapping)
        self.assertFalse(out)        

    
    def test_get_permutation(self):
        """ check the static method get_permutation"""


        out = lhe_parser.Event.get_permutation([3,4,5], "AAA")
        self.assertEqual(len(out), 1)
        
        out = lhe_parser.Event.get_permutation([3,4,5], "AAB")
        self.assertEqual(len(out), 3)
        self.assertIn((3,4,5), out)
        check = set()
        for one in out:
            check.add(one[-1])
        self.assertEqual(len(check), 3)
        self.assertEqual(check, set([3,4,5]))

        out = lhe_parser.Event.get_permutation([3,4,5], "ABC")
        self.assertEqual(len(out), 6)
        self.assertIn((3,4,5), out)
        self.assertIn((4,3,5), out)

        out = lhe_parser.Event.get_permutation([3,4,5, 6], "AACC")
        self.assertEqual(len(out), 6)
        self.assertIn((3,4,5,6), out)
        self.assertNotIn((4,3,6,5), out)
        self.assertIn((4,5,3,6), out)

        out = lhe_parser.Event.get_permutation([3,4,5, 6], "ACCA")
        self.assertEqual(len(out), 6)
        self.assertIn((3,4,5,6), out)
        self.assertIn((4,3,6,5), out)
  

    def test_event_property(self):
        """
        test Event parsing and the function get_momenta and get_all_momenta
        """

        input = """
    10      1 +7.8873842e-07 2.27574700e+02 7.54677100e-03 1.14295200e-01
            2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +1.1463656627e+03 1.1463656627e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -2 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -2.1981849450e+02 2.1981849450e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        23  2    1    2    0    0 +5.7837338551e+01 +1.1718357879e+02 -7.4151825307e+01 1.7598875523e+02 9.1631871662e+01 0.0000e+00 0.0000e+00
        23  2    1    2    0    0 +1.1433114561e+02 +1.4041658917e+02 +7.5301024612e+02 7.8190910735e+02 1.0755924887e+02 0.0000e+00 0.0000e+00
        -11  1    1    2    0    0 -1.7327841208e+02 -1.9147557428e+02 +1.9134097396e+02 3.2140266327e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    1    2    0    0 +1.1099279128e+00 -6.6124593668e+01 +5.6347773431e+01 8.6883631360e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -11  1    4    4    0    0 +4.0361765237e+01 -1.1075255459e+01 +1.6276183383e+02 1.6805697822e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    4    4    0    0 +7.3969380376e+01 +1.5149184462e+02 +5.9024841230e+02 6.1385212913e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -11  1    3    3    0    0 -2.2637466291e+00 -1.4439319632e+01 -1.0004133265e+01 1.7711611520e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    3    3    0    0 +6.0101085180e+01 +1.3162289842e+02 -6.4147692042e+01 1.5827714371e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00

        """

        #first check that we can initialise the event
        event = lhe_parser.Event(input)
        self.assertEqual(10, len(event))

        # check that we can get the momenta for a request order 
        out = event.get_momenta([(2,-2), (-11,-11,-11,11,11,11)])
        self.assertEqual(out, [(1146.3656627, 0.0, 0.0, 1146.3656627), (219.8184945, -0.0, -0.0, -219.8184945), (321.40266327, -173.27841208, -191.47557428, 191.34097396), (168.05697822, 40.361765237, -11.075255459, 162.76183383), (17.71161152, -2.2637466291, -14.439319632, -10.004133265), (86.88363136, 1.1099279128, -66.124593668, 56.347773431), (613.85212913, 73.969380376, 151.49184462, 590.2484123), (158.27714371, 60.10108518, 131.62289842, -64.147692042)])
        
        # and for a second order:
        out2 = event.get_momenta([(2,-2), (-11,11,-11,11,-11,11)])
        
        self.assertEqual(out2, [(1146.3656627, 0.0, 0.0, 1146.3656627), (219.8184945, -0.0, -0.0, -219.8184945), (321.40266327, -173.27841208, -191.47557428, 191.34097396), (86.88363136, 1.1099279128, -66.124593668, 56.347773431), (168.05697822, 40.361765237, -11.075255459, 162.76183383), (613.85212913, 73.969380376, 151.49184462, 590.2484123), (17.71161152, -2.2637466291, -14.439319632, -10.004133265), (158.27714371, 60.10108518, 131.62289842, -64.147692042)])


        # check that first data structure for get_all_momenta is constructed correctly
        # here all particles are considered different
        mother = event.get_all_momenta([(2,-2), (-11,-11,-11,11,11,11)], debug_output=1)
        self.assertEqual(mother, {-11: {(0, 1): [2], (3, 3): [3], (2, 2): [4]}, 11: {(0, 1): [5], (3, 3): [6], (2, 2): [7]}})

        mother2 = event.get_all_momenta([(2,-2), (-11,11,-11,11,-11,11)], debug_output=1)
        self.assertNotEqual(mother2, mother)
        self.assertEqual(mother2, {-11: {(0, 1): [2], (3, 3): [4], (2, 2): [6]}, 11: {(0, 1): [3], (3, 3): [5], (2, 2): [7]}})




        perm_gen = event.get_all_momenta([(2,-2), (-11,-11,-11,11,11,11)], debug_output=2)
        self.assertEqual(len(perm_gen), 2)
        self.assertEqual(len(perm_gen[11]), 6)
        self.assertEqual(len(perm_gen[-11]), 6)
        self.assertEqual(perm_gen, {-11: [[(2, 2), (3, 3), (4, 4)], [(2, 2), (4, 3), (3, 4)], [(3, 2), (2, 3), (4, 4)], [(3, 2), (4, 3), (2, 4)], [(4, 2), (2, 3), (3, 4)], [(4, 2), (3, 3), (2, 4)]], 
                                     11: [[(5, 5), (6, 6), (7, 7)], [(5, 5), (7, 6), (6, 7)], [(6, 5), (5, 6), (7, 7)], [(6, 5), (7, 6), (5, 7)], [(7, 5), (5, 6), (6, 7)], [(7, 5), (6, 6), (5, 7)]]}
        )
        perm_gen2 = event.get_all_momenta([(2,-2), (-11,11,-11,11,-11,11)], debug_output=2)
        self.assertEqual(len(perm_gen2), 2)
        self.assertEqual(len(perm_gen2[11]), 6)
        self.assertEqual(len(perm_gen2[-11]), 6)
        self.assertNotEqual(perm_gen, perm_gen2)
        self.assertEqual(perm_gen2, {-11: [[(2, 2), (4, 4), (6, 6)], [(2, 2), (6, 4), (4, 6)], [(4, 2), (2, 4), (6, 6)], [(4, 2), (6, 4), (2, 6)], [(6, 2), (2, 4), (4, 6)], [(6, 2), (4, 4), (2, 6)]], 
                                     11: [[(3, 3), (5, 5), (7, 7)], [(3, 3), (7, 5), (5, 7)], [(5, 3), (3, 5), (7, 7)], [(5, 3), (7, 5), (3, 7)], [(7, 3), (3, 5), (5, 7)], [(7, 3), (5, 5), (3, 7)]]}
        
        )
        
        
    
        all_perms = event.get_all_momenta([(2,-2), (-11,-11,-11,11,11,11)], debug_output=0)
        self.assertEqual(len(all_perms), 36)
        for i in range(len(all_perms)):
            for j in range(i+1, len(all_perms)):
                # check that initial state are the same
                self.assertEqual(all_perms[i][0], all_perms[j][0])
                self.assertEqual(all_perms[i][1], all_perms[j][1])
                # check that each combination are unique
                self.assertNotEqual(all_perms[i][2:], all_perms[j][2:])
        # check that a given momenta is repeated the correct amount of time at a given position
        nb_identical = [0] * len(all_perms[0])
        for i in range(len(all_perms)):
            for j in range(len(all_perms[0])):
                if all_perms[i][j] == all_perms[0][j]:
                    nb_identical[j] +=1
        self.assertEqual(nb_identical, [36,36,12,12,12,12,12,12]) # for the electron permuation: 2 repetition and from the positron one 6 repetition.

        # redo the same but where two pair of electron/positron are indistiguishable
        input = """
    10      1 +7.8873842e-07 2.27574700e+02 7.54677100e-03 1.14295200e-01
            2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +1.1463656627e+03 1.1463656627e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -2 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -2.1981849450e+02 2.1981849450e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        23  2    1    2    0    0 +5.7837338551e+01 +1.1718357879e+02 -7.4151825307e+01 1.7598875523e+02 9.1631871662e+01 0.0000e+00 0.0000e+00
        23  2    1    2    0    0 +1.1433114561e+02 +1.4041658917e+02 +7.5301024612e+02 7.8190910735e+02 1.0755924887e+02 0.0000e+00 0.0000e+00
        -11  1    1    2    0    0 -1.7327841208e+02 -1.9147557428e+02 +1.9134097396e+02 3.2140266327e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    1    2    0    0 +1.1099279128e+00 -6.6124593668e+01 +5.6347773431e+01 8.6883631360e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -11  1    1    2    0    0 +4.0361765237e+01 -1.1075255459e+01 +1.6276183383e+02 1.6805697822e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    1    2    0    0 +7.3969380376e+01 +1.5149184462e+02 +5.9024841230e+02 6.1385212913e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -11  1    3    3    0    0 -2.2637466291e+00 -1.4439319632e+01 -1.0004133265e+01 1.7711611520e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    4    4    0    0 +6.0101085180e+01 +1.3162289842e+02 -6.4147692042e+01 1.5827714371e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00

        """
        #first check that we can initialise the event
        event = lhe_parser.Event(input)
        self.assertEqual(10, len(event))

        # check that we can get the momenta for a request order 
        out = event.get_momenta([(2,-2), (-11,-11,-11,11,11,11)])
        self.assertEqual(out, [(1146.3656627, 0.0, 0.0, 1146.3656627), (219.8184945, -0.0, -0.0, -219.8184945), (321.40266327, -173.27841208, -191.47557428, 191.34097396), (168.05697822, 40.361765237, -11.075255459, 162.76183383), (17.71161152, -2.2637466291, -14.439319632, -10.004133265), (86.88363136, 1.1099279128, -66.124593668, 56.347773431), (613.85212913, 73.969380376, 151.49184462, 590.2484123), (158.27714371, 60.10108518, 131.62289842, -64.147692042)])
        
        # and for a second order:
        out2 = event.get_momenta([(2,-2), (-11,11,-11,11,-11,11)])
        
        self.assertEqual(out2, [(1146.3656627, 0.0, 0.0, 1146.3656627), (219.8184945, -0.0, -0.0, -219.8184945), (321.40266327, -173.27841208, -191.47557428, 191.34097396), (86.88363136, 1.1099279128, -66.124593668, 56.347773431), (168.05697822, 40.361765237, -11.075255459, 162.76183383), (613.85212913, 73.969380376, 151.49184462, 590.2484123), (17.71161152, -2.2637466291, -14.439319632, -10.004133265), (158.27714371, 60.10108518, 131.62289842, -64.147692042)])


        # check that first data structure for get_all_momenta is constructed correctly
        # here all particles are considered different
        mother = event.get_all_momenta([(2,-2), (-11,-11,-11,11,11,11)], debug_output=1)
        self.assertEqual(mother, {-11: {(0, 1): [2, 3], (2, 2): [4]}, 11: {(0, 1): [5, 6], (3, 3): [7]}})


        mother2 = event.get_all_momenta([(2,-2), (-11,11,-11,11,-11,11)], debug_output=1)

        self.assertNotEqual(mother2, mother)
        self.assertEqual(mother2, {-11: {(0, 1): [2, 4], (2, 2): [6]}, 11: {(0, 1): [3, 5], (3, 3): [7]}})




        perm_gen = event.get_all_momenta([(2,-2), (-11,-11,-11,11,11,11)], debug_output=2)
        self.assertEqual(len(perm_gen), 2)
        self.assertEqual(len(perm_gen[11]), 3)
        self.assertEqual(len(perm_gen[-11]), 3)
        self.assertEqual(perm_gen, {-11: [[(2, 2), (3, 3), (4, 4)], [(2, 2), (4, 3), (3, 4)], [(3, 2), (4, 3), (2, 4)]], 
                                     11: [[(5, 5), (6, 6), (7, 7)], [(5, 5), (7, 6), (6, 7)], [(6, 5), (7, 6), (5, 7)]]}
        )

        perm_gen2 = event.get_all_momenta([(2,-2), (-11,11,-11,11,-11,11)], debug_output=2)
        self.assertEqual(len(perm_gen2), 2)
        self.assertEqual(len(perm_gen2[11]), 3)
        self.assertEqual(len(perm_gen2[-11]), 3)
        self.assertNotEqual(perm_gen, perm_gen2)
        self.assertEqual(perm_gen2, {-11: [[(2, 2), (4, 4), (6, 6)], [(2, 2), (6, 4), (4, 6)], [(4, 2), (6, 4), (2, 6)]], 
                                      11: [[(3, 3), (5, 5), (7, 7)], [(3, 3), (7, 5), (5, 7)], [(5, 3), (7, 5), (3, 7)]]}
        )
        
        
        all_perms = event.get_all_momenta([(2,-2), (-11,-11,-11,11,11,11)], debug_output=0)
        self.assertEqual(len(all_perms), 9)
        for i in range(len(all_perms)):
            for j in range(i+1, len(all_perms)):
                # check that initial state are the same
                self.assertEqual(all_perms[i][0], all_perms[j][0])
                self.assertEqual(all_perms[i][1], all_perms[j][1])
                # check that each combination are unique
                self.assertNotEqual(all_perms[i][2:], all_perms[j][2:])
        # check that a given momenta is repeated the correct amount of time at a given position
        nb_identical = [0] * len(all_perms[0])
        for i in range(len(all_perms)):
            for j in range(len(all_perms[0])):
                if all_perms[i][j] == all_perms[0][j]:
                    nb_identical[j] +=1
        self.assertIn(nb_identical, [[9,9,3,3,3,3,3,3],
                                     [9,9,6,3,3,6,3,3]]) # technically other combination are possible 


        # redo the same but with less particles (simpler case so should go trough)
        input = """
    8      1 +7.8873842e-07 2.27574700e+02 7.54677100e-03 1.14295200e-01
            2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +1.1463656627e+03 1.1463656627e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -2 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -2.1981849450e+02 2.1981849450e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        23  2    1    2    0    0 +5.7837338551e+01 +1.1718357879e+02 -7.4151825307e+01 1.7598875523e+02 9.1631871662e+01 0.0000e+00 0.0000e+00
        23  2    1    2    0    0 +1.1433114561e+02 +1.4041658917e+02 +7.5301024612e+02 7.8190910735e+02 1.0755924887e+02 0.0000e+00 0.0000e+00
        -11  1    1    2    0    0 -1.7327841208e+02 -1.9147557428e+02 +1.9134097396e+02 3.2140266327e+02 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    1    2    0    0 +1.1099279128e+00 -6.6124593668e+01 +5.6347773431e+01 8.6883631360e+01 0.0000000000e+00 0.0000e+00 -1.0000e+00
        -11  1    3    3    0    0 -2.2637466291e+00 -1.4439319632e+01 -1.0004133265e+01 1.7711611520e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        11  1    4    4    0    0 +6.0101085180e+01 +1.3162289842e+02 -6.4147692042e+01 1.5827714371e+02 0.0000000000e+00 0.0000e+00 -1.0000e+00

        """
        #first check that we can initialise the event
        event = lhe_parser.Event(input)
        self.assertEqual(8, len(event))


        # check that first data structure for get_all_momenta is constructed correctly
        # here all particles are considered different
        mother = event.get_all_momenta([(2,-2), (-11,-11,11,11)], debug_output=1)
        self.assertEqual(mother, {-11: {(0, 1): [2], (2, 2): [3]}, 11: {(0, 1): [4], (3, 3): [5]}})



        perm_gen = event.get_all_momenta([(2,-2), (-11,-11,11,11)], debug_output=2)
        self.assertEqual(len(perm_gen), 2)
        self.assertEqual(len(perm_gen[11]), 2)
        self.assertEqual(len(perm_gen[-11]), 2)
        self.assertEqual(perm_gen, {-11: [[(2, 2), (3, 3)], [(3, 2), (2, 3)]], 
                                     11: [[(4, 4), (5, 5)], [(5, 4), (4, 5)]]}
        )
        
        all_perms = event.get_all_momenta([(2,-2), (-11,-11,11,11)], debug_output=0)
        self.assertEqual(len(all_perms), 4)
        for i in range(len(all_perms)):
            for j in range(i+1, len(all_perms)):
                # check that initial state are the same
                self.assertEqual(all_perms[i][0], all_perms[j][0])
                self.assertEqual(all_perms[i][1], all_perms[j][1])
                # check that each combination are unique
                self.assertNotEqual(all_perms[i][2:], all_perms[j][2:])
        # check that a given momenta is repeated the correct amount of time at a given position
        nb_identical = [0] * len(all_perms[0])
        for i in range(len(all_perms)):
            for j in range(len(all_perms[0])):
                if all_perms[i][j] == all_perms[0][j]:
                    nb_identical[j] +=1
        self.assertEqual(nb_identical, [4,4,2,2,2,2])


from madgraph.various.lhe_parser import FourMomentum
class TestFourMomentum(unittest.TestCase):


    def test_boost_to_restframe(self):
        """check that we can correctly boost momenta to a restframe"""

        plep = FourMomentum(38.249416771525524,24.8053721987,27.2397493528,-10.281412774874447)
        ptop = FourMomentum(186.51892916393933,66.4591715464,-21.3845299376,10.463198859726106)
        #Want to have the lepton momenta in the frame where the top is at rest

        out = plep.boost_to_restframe(ptop) 

        self.assertAlmostEqual(out.E, 35.77191285579485)
        self.assertAlmostEqual(out.px,  11.108508511835302)
        self.assertAlmostEqual(out.py, 31.646981413441754)
        self.assertAlmostEqual(out.pz, -12.437819560808716)


        out = ptop.boost_to_restframe(ptop)
        self.assertAlmostEqual(out.E, ptop.mass) 
        self.assertAlmostEqual(out.px,  0)
        self.assertAlmostEqual(out.py, 0)
        self.assertAlmostEqual(out.pz, 0)

    def test_y_eta_massless(self):
        """test that rapidity and pseudorapidity coincide for
        a massless particle. At 45 degrees, they both should be equal to .88"""
        p = FourMomentum(100*math.sqrt(2),100.,0.,100.)
        self.assertAlmostEqual(p.rapidity, p.pseudorapidity)
        self.assertAlmostEqual(p.rapidity, 0.881373587)
        self.assertAlmostEqual(p.pseudorapidity, 0.881373587)


class TESTLHEParser(unittest.TestCase):

    def setUp(self):
        
        debugging = unittest.debug
        if debugging:
            self.path = pjoin(MG5DIR, "tmp_lhe_test")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(pjoin(MG5DIR, "tmp_test"))
        else:
            self.path = tempfile.mkdtemp(prefix='test_mg5')

    def tearDown(self):
        
        if self.path != pjoin(MG5DIR, "tmp_lhe_test"):
            shutil.rmtree(self.path)



    def test_parsing_lo_weight(self):
        """test that our parser can handle a large range of lo_weight format"""

        def parse_lo_weight_old(evt):
            """parsing for unittest onlyx"""

            
            start, stop = evt.tag.find('<mgrwt>'), evt.tag.find('</mgrwt>')
    
            if start != -1 != stop :
                text = evt.tag[start+8:stop]
    #<rscale>  3 0.29765919e+03</rscale>
    #<asrwt>0</asrwt>
    #<pdfrwt beam="1">  1       21 0.15134321e+00 0.29765919e+03</pdfrwt>
    #<pdfrwt beam="2">  1       21 0.38683649e-01 0.29765919e+03</pdfrwt>
    #<totfact> 0.17315115e+03</totfact>
                evt.loweight={}
                for line in text.split('\n'):
                    line = line.replace('<', ' <').replace("'",'"')
                    if 'rscale' in line:
                        _, nqcd, scale, _ = line.split()
                        evt.loweight['n_qcd'] = int(nqcd)
                        evt.loweight['ren_scale'] = float(scale)
                    elif '<pdfrwt beam="1"' in line:
                        args = line.split()
                        evt.loweight['n_pdfrw1'] = int(args[2])
                        npdf = evt.loweight['n_pdfrw1']
                        evt.loweight['pdf_pdg_code1'] = [int(i) for i in args[3:3+npdf]]
                        evt.loweight['pdf_x1'] = [float(i) for i in args[3+npdf:3+2*npdf]]
                        evt.loweight['pdf_q1'] = [float(i) for i in args[3+2*npdf:3+3*npdf]]
                    elif '<pdfrwt beam="2"' in line:
                        args = line.split()
                        evt.loweight['n_pdfrw2'] = int(args[2])
                        npdf = evt.loweight['n_pdfrw2']
                        evt.loweight['pdf_pdg_code2'] = [int(i) for i in args[3:3+npdf]]
                        evt.loweight['pdf_x2'] = [float(i) for i in args[3+npdf:3+2*npdf]]
                        evt.loweight['pdf_q2'] = [float(i) for i in args[3+2*npdf:3+3*npdf]]
                    elif '<asrwt>' in line:
                        args = line.replace('>','> ').split()
                        nalps = int(args[1])
                        evt.loweight['asrwt'] = [float(a) for a in args[2:2+nalps]] 
                        
                    elif 'totfact' in line:
                        args = line.replace('>','> ').split()
                        evt.loweight['tot_fact'] = float(args[1])
            else:
                return None
            return evt.loweight

        
        events=["""
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
</event>
""","""
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>  2 0.12500000E+03</rscale>
<asrwt>0</asrwt>
<pdfrwt beam="1">  1        4 0.11319990E+00 0.12500000E+03</pdfrwt>
<pdfrwt beam="2">  1       -1 0.59528052E+00 0.12500000E+03</pdfrwt>
<totfact>-0.27352270E-03</totfact>
</mgrwt>
</event>""",
"""
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>  2 0.12500000E+03</rscale>
<asrwt> 1 0.11 </asrwt>
<pdfrwt beam='1'>  1        4 0.11319990E+00 0.12500000E+03</pdfrwt>
<pdfrwt beam=2>    2      1 -1  0.2 0.11e-02 0.59528052E+00 0.12500000E+03</pdfrwt>
<totfact> 115 </totfact>
</mgrwt>
</event>""",
"""<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>2 0.12500000E+03</rscale>
<asrwt>1 0.11 </asrwt>
<pdfrwt beam='1'>  1        4 0.11319990E+00 0.12500000e+03 </pdfrwt>
<pdfrwt beam=2>    2      1 -1  0.2 0.11e-02 0.59528052E+00 0.12500000E+03 </pdfrwt>
<totfact> 115.001 </totfact>
</mgrwt>
</event>""",
"""<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>2 0.12500000E+03</rscale>
<asrwt>1 0.11 </asrwt>
<pdfrwt beam='1'>  1        4 0.11319990E+00 0.12500000e+03 </pdfrwt>
<pdfrwt beam=2>    2      1 -1  0.2 0.11e-02 0.59528052E+00 0.12500000E+03 </pdfrwt>
<totfact> 115.001 </totfact>
</mgrwt>
</event>""",
"""<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
<mgrwt>
<rscale>  2 0.26956472E+02</rscale>
    <asrwt>  1 0.46373112E+02</asrwt>
    <pdfrwt beam="1">  1       21 0.13689253E-01 0.52142986E+01</pdfrwt>
    <pdfrwt beam="2">  1       21 0.29841683E-01 0.46373112E+02</pdfrwt>
    <totfact> 0.15951072E+03</totfact>
</mgrwt>
</event>
"""]
     
        solutions = [None, 
                  {'pdf_pdg_code1': [4], 'asrwt': [], 'pdf_pdg_code2': [-1], 'pdf_q1': [125.0], 'pdf_q2': [125.0], 'n_pdfrw1': 1, 'n_pdfrw2': 1, 'tot_fact': -0.0002735227, 'pdf_x2': [0.59528052], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'ren_scale': 125.0},
                  {'pdf_pdg_code1': [4], 'asrwt': [0.11], 'pdf_pdg_code2': [1, -1], 'pdf_q1': [125.0], 'pdf_q2': [0.59528052, 125.0], 'ren_scale': 125.0, 'n_pdfrw1': 1, 'n_pdfrw2': 2, 'pdf_x2': [0.2, 0.0011], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'tot_fact': 115.0},
                  {'pdf_pdg_code1': [4], 'asrwt': [0.11], 'pdf_pdg_code2': [1, -1], 'pdf_q1': [125.0], 'pdf_q2': [0.59528052, 125.0], 'ren_scale': 125.0, 'n_pdfrw1': 1, 'n_pdfrw2': 2, 'pdf_x2': [0.2, 0.0011], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'tot_fact': 115.001},
                  {'pdf_pdg_code1': [4], 'asrwt': [0.11], 'pdf_pdg_code2': [1, -1], 'pdf_q1': [125.0], 'pdf_q2': [0.59528052, 125.0], 'ren_scale': 125.0, 'n_pdfrw1': 1, 'n_pdfrw2': 2, 'pdf_x2': [0.2, 0.0011], 'pdf_x1': [0.1131999], 'n_qcd': 2, 'tot_fact': 115.001},
                  {'pdf_pdg_code1': [21], 'asrwt': [46.373112], 'pdf_pdg_code2': [21], 'pdf_q1': [5.2142986], 'pdf_q2': [46.373112], 'ren_scale': 26.956472, 'n_pdfrw1': 1, 'n_pdfrw2': 1, 'pdf_x2': [0.029841683], 'pdf_x1': [0.013689253], 'n_qcd': 2, 'tot_fact': 159.51072},
                  None]
                  
     
        for i,evt in enumerate(events):
            evt1 = lhe_parser.Event(evt)
            evt2 = lhe_parser.Event(evt)
            lo = evt1.parse_lo_weight()
            try:
                lo2 = parse_lo_weight_old(evt2)
            except:
                pass
            else:
                if lo:
                    for key in lo2:
                        self.assertEqual(lo[key], lo2[key])
            self.assertEqual(lo, solutions[i])
         
     
     
        
        

    def test_read_write_lhe(self):
        """test that we can read/write an lhe event file"""
        
        input= """<LesHouchesEvents version="1.0">
<header>
DATA
</header>
<init>
     2212     2212  0.70000000000E+04  0.70000000000E+04 0 0 10042 10042 3  1
  0.16531958660E+02  0.18860728290E+00  0.17208000000E+00   0
</init>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
</event>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
# balbalblb
#bbbb3
</event>            
<event>
  7     66 +1.5024446e-03 3.15138740e+02 7.95774720e-02 9.66701260e-02
       21 -1    0    0  502  501 +0.0000000e+00 +0.0000000e+00 -6.4150959e+01 6.41553430e+01 7.49996552e-01 0.0000e+00 0.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 +8.1067989e+02 8.10679950e+02 3.11899968e-01 0.0000e+00 0.0000e+00
       24  2    1    2    0    0 -1.5294533e+02 +1.0783429e+01 +4.5796553e+02 4.89600040e+02 8.04190039e+01 0.0000e+00 0.0000e+00
        2  1    3    3  503    0 -1.5351296e+02 +1.2743130e+01 +4.6093709e+02 4.85995489e+02 0.00000000e+00 0.0000e+00 0.0000e+00
       -1  1    3    3    0  503 +5.6763429e-01 -1.9597014e+00 -2.9715566e+00 3.60455091e+00 0.00000000e+00 0.0000e+00 0.0000e+00
       23  1    1    2    0    0 +4.4740095e+01 +3.2658177e+01 +4.6168760e+01 1.16254200e+02 9.11880036e+01 0.0000e+00 0.0000e+00
        1  1    1    2  502    0 +1.0820523e+02 -4.3441605e+01 +2.4239464e+02 2.68981060e+02 3.22945297e-01 0.0000e+00 0.0000e+00
# 2  5  2  2  1 0.11659994e+03 0.11659994e+03 8  0  0 0.10000000e+01 0.88172677e+00 0.11416728e+01 0.00000000e+00 0.00000000e+00
  <rwgt>
    <wgt id='1001'> +9.1696000e+03 </wgt>
    <wgt id='1002'> +1.1264000e+04 </wgt>
    <wgt id='1003'> +6.9795000e+03 </wgt>
    <wgt id='1004'> +9.1513000e+03 </wgt>
    <wgt id='1005'> +1.1253000e+04 </wgt>
  </rwgt>
</event>
</LesHouchesEvents> 
        
        """
        
        open(pjoin(self.path,'event.lhe'),'w').write(input)
        
        input = lhe_parser.EventFile(pjoin(self.path,'event.lhe'))
        self.assertEqual(input.banner.lower(), """<LesHouchesEvents version="1.0">
<header>
DATA
</header>
<init>
     2212     2212  0.70000000000E+04  0.70000000000E+04 0 0 10042 10042 3  1
  0.16531958660E+02  0.18860728290E+00  0.17208000000E+00   0
</init>
""".lower())
        
        nb_event = 0
        txt = ""
        for event in input:
            nb_event +=1
            new = lhe_parser.Event(text=str(event))
            for part1,part2 in zip(event, new):
                self.assertEqual(part1, part2)
            self.assertEqual(new, event, '%s \n !=\n %s' % (new, event))
            txt += str(event)
        self.assertEqual(nb_event, 3)
    
        target = """<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000000e+00 +0.0000000000e+00 +1.1943355000e+01 1.1943354600e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 -1.0679326000e+03 1.0679326200e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155000e+00 +4.2744556000e+01 -7.9238049000e+02 7.9761999700e+02 8.0419007300e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155000e+00 -4.2744556000e+01 -2.6360878000e+02 2.8225597900e+02 9.1188003500e+01 1.8975e-26 1.0000e+00
</event>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000000e+00 +0.0000000000e+00 +1.1943355000e+01 1.1943354600e+01 0.0000000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 -1.0679326000e+03 1.0679326200e+03 0.0000000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155000e+00 +4.2744556000e+01 -7.9238049000e+02 7.9761999700e+02 8.0419007300e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155000e+00 -4.2744556000e+01 -2.6360878000e+02 2.8225597900e+02 9.1188003500e+01 1.8975e-26 1.0000e+00
# balbalblb
#bbbb3
</event>
<event>
 7     66 +1.5024446e-03 3.15138740e+02 7.95774720e-02 9.66701260e-02
       21 -1    0    0  502  501 +0.0000000000e+00 +0.0000000000e+00 -6.4150959000e+01 6.4155343000e+01 7.4999655200e-01 0.0000e+00 0.0000e+00
        2 -1    0    0  501    0 +0.0000000000e+00 +0.0000000000e+00 +8.1067989000e+02 8.1067995000e+02 3.1189996800e-01 0.0000e+00 0.0000e+00
       24  2    1    2    0    0 -1.5294533000e+02 +1.0783429000e+01 +4.5796553000e+02 4.8960004000e+02 8.0419003900e+01 0.0000e+00 0.0000e+00
        2  1    3    3  503    0 -1.5351296000e+02 +1.2743130000e+01 +4.6093709000e+02 4.8599548900e+02 0.0000000000e+00 0.0000e+00 0.0000e+00
       -1  1    3    3    0  503 +5.6763429000e-01 -1.9597014000e+00 -2.9715566000e+00 3.6045509100e+00 0.0000000000e+00 0.0000e+00 0.0000e+00
       23  1    1    2    0    0 +4.4740095000e+01 +3.2658177000e+01 +4.6168760000e+01 1.1625420000e+02 9.1188003600e+01 0.0000e+00 0.0000e+00
        1  1    1    2  502    0 +1.0820523000e+02 -4.3441605000e+01 +2.4239464000e+02 2.6898106000e+02 3.2294529700e-01 0.0000e+00 0.0000e+00
# 2  5  2  2  1 0.11659994e+03 0.11659994e+03 8  0  0 0.10000000e+01 0.88172677e+00 0.11416728e+01 0.00000000e+00 0.00000000e+00
<rwgt>
<wgt id='1001'> +9.1696000e+03 </wgt>
<wgt id='1002'> +1.1264000e+04 </wgt>
<wgt id='1003'> +6.9795000e+03 </wgt>
<wgt id='1004'> +9.1513000e+03 </wgt>
<wgt id='1005'> +1.1253000e+04 </wgt>
</rwgt>
</event>
"""


        self.assertEqual(target.split('\n'), txt.split('\n'))


    def test_read_write_gzip(self):
        """ """ 
        
        input= """<LesHouchesEvents version="1.0">
<header>
DATA
</header>
<init>
     2212     2212  0.70000000000E+04  0.70000000000E+04 0 0 10042 10042 3  1
  0.16531958660E+02  0.18860728290E+00  0.17208000000E+00   0
</init>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
</event>
<event>
 4      0 +1.7208000e-01 1.00890300e+02 7.95774700e-02 1.27947900e-01
       -1 -1    0    0    0  501 +0.0000000e+00 +0.0000000e+00 +1.1943355e+01 1.19433546e+01 0.00000000e+00 0.0000e+00 1.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 -1.0679326e+03 1.06793262e+03 0.00000000e+00 0.0000e+00 -1.0000e+00
       24  1    1    2    0    0 +6.0417155e+00 +4.2744556e+01 -7.9238049e+02 7.97619997e+02 8.04190073e+01 3.4933e-25 -1.0000e+00
       23  1    1    2    0    0 -6.0417155e+00 -4.2744556e+01 -2.6360878e+02 2.82255979e+02 9.11880035e+01 1.8975e-26 1.0000e+00
# balbalblb
#bbbb3
</event>            
<event>
  7     66 +1.5024446e-03 3.15138740e+02 7.95774720e-02 9.66701260e-02
       21 -1    0    0  502  501 +0.0000000e+00 +0.0000000e+00 -6.4150959e+01 6.41553430e+01 7.49996552e-01 0.0000e+00 0.0000e+00
        2 -1    0    0  501    0 +0.0000000e+00 +0.0000000e+00 +8.1067989e+02 8.10679950e+02 3.11899968e-01 0.0000e+00 0.0000e+00
       24  2    1    2    0    0 -1.5294533e+02 +1.0783429e+01 +4.5796553e+02 4.89600040e+02 8.04190039e+01 0.0000e+00 0.0000e+00
        2  1    3    3  503    0 -1.5351296e+02 +1.2743130e+01 +4.6093709e+02 4.85995489e+02 0.00000000e+00 0.0000e+00 0.0000e+00
       -1  1    3    3    0  503 +5.6763429e-01 -1.9597014e+00 -2.9715566e+00 3.60455091e+00 0.00000000e+00 0.0000e+00 0.0000e+00
       23  1    1    2    0    0 +4.4740095e+01 +3.2658177e+01 +4.6168760e+01 1.16254200e+02 9.11880036e+01 0.0000e+00 0.0000e+00
        1  1    1    2  502    0 +1.0820523e+02 -4.3441605e+01 +2.4239464e+02 2.68981060e+02 3.22945297e-01 0.0000e+00 0.0000e+00
# 2  5  2  2  1 0.11659994e+03 0.11659994e+03 8  0  0 0.10000000e+01 0.88172677e+00 0.11416728e+01 0.00000000e+00 0.00000000e+00
  <rwgt>
    <wgt id='1001'> +9.1696000e+03 </wgt>
    <wgt id='1002'> +1.1264000e+04 </wgt>
    <wgt id='1003'> +6.9795000e+03 </wgt>
    <wgt id='1004'> +9.1513000e+03 </wgt>
    <wgt id='1005'> +1.1253000e+04 </wgt>
  </rwgt>
</event>
</LesHouchesEvents> 
        
        """
        
        open(pjoin(self.path, 'event.lhe'),'w').write(input)
        input_lhe = lhe_parser.EventFile(pjoin(self.path, 'event.lhe.gz'))
        output_lhe = lhe_parser.EventFile(pjoin(self.path, 'event2.lhe.gz'),'wb')
        output_lhe.write(input_lhe.banner)
        for event in input_lhe:
            output_lhe.write(str(event))
        output_lhe.close()
        self.assertTrue(pjoin(self.path,'event2.lhe.gz'))
        try:
            text = open(pjoin(self.path, 'event2.lhe.gz'), 'r').read()
            self.assertFalse(text.startswith('<LesHouchesEvents version="1.0">'))
        except UnicodeDecodeError:
            pass
        misc.gunzip(pjoin(self.path,'event2.lhe.gz'))
        self.assertTrue(pjoin(self.path,'event2.lhe'))
        input_lhe = lhe_parser.EventFile(pjoin(self.path, 'event.lhe'))
        

class TESTLHEParserNLO(unittest.TestCase):

    def setUp(self):
        
        debugging = unittest.debug
        if debugging:
            self.path = pjoin(MG5DIR, "tmp_lhe_nlo_test")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(pjoin(MG5DIR, "tmp_test"))
        else:
            self.path = tempfile.mkdtemp(prefix='test_mg5')

    def tearDown(self):
        
        if self.path != pjoin(MG5DIR, "tmp_lhe_nlo_test"):
            shutil.rmtree(self.path)


    def test_parse_event_nlo(self):
        """test that the parsing of event is working"""

        input ="""5      0 0.10593147E+01 0.24441289E+03 0.78185903E-02 0.10405812E+00
        21 -1    0    0  501  502  0.00000000000000000E+00  0.00000000000000000E+00  0.42282537151686591E+03  0.          42282603668456335E+03  0.75000000000000000E+00 0.0000E+00 0.9000E+01
        21 -1    0    0  502  503  0.00000000000000000E+00  0.00000000000000000E+00 -0.18150169578842335E+03  0.          18150324535410761E+03  0.75000000000000000E+00 0.0000E+00 0.9000E+01
        25  1    1    2    0    0  0.56214411169559405E+02  0.66042662140338271E+01 -0.20826596102379959E+02  0.          13878913307812604E+03  0.12500000000000000E+03 0.0000E+00 0.9000E+01
         6  1    1    2  501    0 -0.15861007254562349E+02  0.16129295015972410E+02  0.22408759852532069E+03  0.          28339191506055153E+03  0.17200000000000000E+03 0.0000E+00 0.9000E+01
        -6  1    1    2    0  503 -0.40353403914997060E+02 -0.22733561230006238E+02  0.38062673305501825E+02  0.          18214823389999333E+03  0.17200000000000000E+03 0.0000E+00 0.9000E+01
 #aMCatNLO 1  6  2  2  1 0.78291954E+03 0.00000000E+00 -9  6  0 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+ 00 0.00000000E+00
 <mgrwgt>
  0.1031581000D-03   42    3    0
  0.277027419519390D+03 0.000000000000000D+00 0.000000000000000D+00 0.277027419519390D+03
  0.277027419519390D+03 0.000000000000000D+00 0.000000000000000D+00 -.277027419519390D+03
  0.160453950779390D+03 0.562144111695594D+02 0.660426621403383D+01 -.831672649865851D+02
  0.211503232795725D+03 -.158610072545623D+02 0.161292950159724D+02 0.120987155409990D+03
  0.182097655463665D+03 -.403534039149971D+02 -.227335612300062D+02 -.378198904234052D+02
  0.000000000000000D+00 -.000000000000000D+00 -.000000000000000D+00 -.000000000000000D+00
  0.865053245639401D+03 0.000000000000000D+00 0.000000000000000D+00 0.865053245639401D+03
  0.877093774682915D+03 0.000000000000000D+00 0.000000000000000D+00 -.877093774682915D+03
  0.350562512177784D+03 0.282442901765167D+03 0.143451344439578D+03 -.831672649865851D+02
  0.358756738289416D+03 0.236594758571710D+03 0.168841421784626D+03 0.120987155409990D+03
  0.249815653693264D+03 0.150855887921291D+03 0.929301770042065D+02 -.378198904234052D+02
  0.783012116161852D+03 -.669893548258168D+03 -.405222943228411D+03 -.120405290435139D+02
  0.277027419519390D+03 0.000000000000000D+00 0.000000000000000D+00 0.277027419519390D+03
  0.225967710793224D+04 0.000000000000000D+00 0.000000000000000D+00 -.225967710793224D+04
  0.160453950779390D+03 0.562144111695594D+02 0.660426621403383D+01 -.831672649865851D+02
  0.211503232795725D+03 -.158610072545623D+02 0.161292950159724D+02 0.120987155409990D+03
  0.182097655463665D+03 -.403534039149971D+02 -.227335612300062D+02 -.378198904234052D+02
  0.198264968841285D+04 -.000000000000000D+00 -.000000000000000D+00 -.198264968841285D+04
   0.195145760518D-04 0.000000000000D+00 0.000000000000D+00 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20400  4 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2  2  2  6  2       21 0.102125239379D+00 0.100000000000D+01
 0.236612551047D-05 0.000000000000D+00 0.000000000000D+00 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20402  4 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2  2  2  6  2       21 0.123825971681D-01 0.100000000000D+01
 0.717226692041D-07 0.000000000000D+00 0.000000000000D+00 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20404  4 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2  2  2  6  2       21 0.375344805948D-03 0.100000000000D+01
 -.914823949886D-06 0.189485494619D-05 0.160922492366D-06 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20600  6 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2  3  2  6  2       21 -.626033177164D-02 0.100000000000D+01
 0.813641128621D-06 0.000000000000D+00 0.000000000000D+00 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20600  6 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2 15  2  6  2       21 0.556791654706D-02 0.100000000000D+01
 -.110921614677D-06 0.229749527477D-06 0.195117133666D-07 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20602  6 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2  3  2  6  2       21 -.759059826334D-03 0.100000000000D+01
 0.986532848925D-07 0.000000000000D+00 0.000000000000D+00 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20602  6 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2 15  2  6  2       21 0.675105077722D-03 0.100000000000D+01
 -.336228752105D-08 0.696423299867D-08 0.591444603090D-09 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20604  6 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2  3  2  6  2       21 -.230088372698D-04 0.100000000000D+01
 0.299040642051D-08 0.000000000000D+00 0.000000000000D+00 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20604  6 0.65050228D-01 0.27923508D-01 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  1  2 15  2  6  2       21 0.204639770600D-04 0.100000000000D+01
 0.127621223942D-02 0.000000000000D+00 -.696483305066D-03 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  2  6  2       21 0.468550602584D-01 0.100000000000D+01
 0.154739633003D-03 0.000000000000D+00 -.844479998620D-04 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  2  6  2       21 0.568113563305D-02 0.100000000000D+01
 0.469051175075D-05 0.000000000000D+00 -.255981178186D-05 0.760310459950D-05 0.000000000000D+00  6 21 21 25 6 -6 21 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  2  6  2       21 0.172208198555D-03 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 -2 25 6 -6 -2 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.249942845473D-02 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 -4 25 6 -6 -4 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.645352006106D-03 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 -1 25 6 -6 -1 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.286028403423D-02 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 -3 25 6 -6 -3 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.821735293595D-03 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 -5 25 6 -6 -5 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.368470634933D-03 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 -2 25 6 -6 -2 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.303053543804D-03 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 -4 25 6 -6 -4 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.782483739757D-04 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 -1 25 6 -6 -1 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.346806971497D-03 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 -3 25 6 -6 -3 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.996346954125D-04 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 -5 25 6 -6 -5 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.446767465949D-04 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 -2 25 6 -6 -2 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.918624518319D-05 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 -4 25 6 -6 -4 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.237188695933D-05 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 -1 25 6 -6 -1 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.105125115233D-04 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 -3 25 6 -6 -3 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.302015521523D-05 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 -5 25 6 -6 -5 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  7  6  2       21 0.135425424516D-05 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 2 25 6 -6 2 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.522035338488D-01 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 4 25 6 -6 4 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.645352006106D-03 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 1 25 6 -6 1 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.239207551122D-01 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 3 25 6 -6 3 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.140655939476D-02 0.100000000000D+01
 0.555375280429D-03 0.000000000000D+00 -.301772224305D-03 0.760310459950D-05 0.000000000000D+00  6 21 5 25 6 -6 5 20600  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.368470634933D-03 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 2 25 6 -6 2 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.632963344160D-02 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 4 25 6 -6 4 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.782483739757D-04 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 1 25 6 -6 1 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.290037092020D-02 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 3 25 6 -6 3 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.170544113133D-03 0.100000000000D+01
 0.673387736129D-04 0.000000000000D+00 -.365896218490D-04 0.760310459950D-05 0.000000000000D+00  6 21 5 25 6 -6 5 20602  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.446767465949D-04 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 2 25 6 -6 2 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.191865648507D-03 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 4 25 6 -6 4 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.237188695933D-05 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 1 25 6 -6 1 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.879168679591D-04 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 3 25 6 -6 3 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.516958164594D-05 0.100000000000D+01
 0.204119205134D-05 0.000000000000D+00 -.110911502055D-05 0.760310459950D-05 0.000000000000D+00  6 21 5 25 6 -6 5 20604  6 0.65050228D-01 0.22776847D+00 0.55225000D+05 0.55225000D+05 0.55225000D+05 0.11435178D+01  3  2  5  8  6  2       21 0.135425424516D-05 0.100000000000D+01
</mgrwgt>
 """
        evt1 = lhe_parser.Event(input)
        #evt1.parse_reweight() 
        evt1.parse_nlo_weight()
        self.assertFalse(evt1.nloweight.ispureqcd())

        for cevent in evt1.nloweight.cevents:
            self.assertIn(len(cevent), (5,6))

        
        input_tt="""
   4      0 0.11274375E+04 0.20505458E+03 0.75467711E-02 0.93919079E-01
        21 -1    0    0  503  502  0.00000000000000000E+00  0.00000000000000000E+00  0.18170668562899095E+03  0.18170823344656947E+03  0.75000000000000000E+00 0.0000E+00 0.9000E+01
        21 -1    0    0  501  503 -0.00000000000000000E+00 -0.00000000000000000E+00 -0.16702087667067835E+04  0.16702089350988979E+04  0.75000000000000000E+00 0.0000E+00 0.9000E+01
         6  1    1    2  501    0 -0.32339471900291807E+03  0.38592008829917069E+03 -0.98220482100754839E+03  0.11172169750063263E+04  0.17300000000000000E+03 0.0000E+00 0.9000E+01
        -6  1    1    2    0  502  0.32339471900291772E+03 -0.38592008829917023E+03 -0.50629726007024482E+03  0.73470019353914188E+03  0.17300000000000000E+03 0.0000E+00 0.9000E+01
 #aMCatNLO 1  0  0  0  0 0.00000000E+00 0.00000000E+00 -9  5  0 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
   <mgrwgt>
  0.1116882500D+00    1    2    0
  0.550898982069431D+03 0.000000000000000D+00 0.000000000000000D+00 0.550898982069431D+03
  0.550898982069431D+03 -.000000000000000D+00 -.000000000000000D+00 -.550898982069431D+03
  0.550898982069432D+03 -.323394719002918D+03 0.385920088299171D+03 -.141570581736885D+03
  0.550898982069431D+03 0.323394719002918D+03 -.385920088299170D+03 0.141570581736884D+03
  0.000000000000000D+00 0.000000000000000D+00 0.000000000000000D+00 -.000000000000000D+00
  0.550898982069431D+03 0.000000000000000D+00 0.000000000000000D+00 0.550898982069431D+03
  0.550898982069431D+03 -.000000000000000D+00 -.000000000000000D+00 -.550898982069431D+03
  0.391939067407447D+03 -.217454558343616D+03 0.259497380216243D+03 -.951937880153362D+02
  0.436344732144414D+03 -.601832139861054D+01 -.320075966727984D+03 0.240796346734891D+03
  0.273514164587001D+03 0.223472879742226D+03 0.605785865117412D+02 -.145602558719555D+03
 0.399976491867D+00 0.000000000000D+00 0.000000000000D+00 0.293174259046D+00 0.000000000000D+00  5 21 21 6 -6 21 6  6 0.27955007D-01 0.25695533D+00 0.28344746D+06 0.28344746D+06 0.28344746D+06 0.10863802D+01  1  2  14  4  5  4       -6 0.325857975681D+02 0.100000000000D+01
   </mgrwgt>
"""

        evt2 = lhe_parser.Event(input_tt)
        #evt1.parse_reweight() 
        evt2.parse_nlo_weight()
        self.assertTrue(evt2.nloweight.ispureqcd())

        for cevent in evt2.nloweight.cevents:
            self.assertIn(len(cevent), (4,5))

        # mu+ mu [QCD]
        input_dy = """
   <event>
   5      0 0.55083926E+03 0.35472124E+02 0.75467711E-02 0.11155337E+00
         2 -1    0    0  501    0  0.00000000000000000E+00  0.00000000000000000E+00  0.10206815190231115E+02  0.10211830214390197E+02  0.32000000000000001E+00 0.0000E+00 0.9000E+01
        -2 -1    0    0    0  501  0.00000000000000000E+00  0.00000000000000000E+00 -0.12329761613404298E+03  0.12329803138873635E+03  0.32000000000000001E+00 0.0000E+00 0.9000E+01
        23  2    1    2    0    0  0.00000000000000000E+00  0.00000000000000000E+00 -0.11309080094381187E+03  0.13350986160312655E+03  0.70960227502264445E+02 0.0000E+00 0.0000E+00
        13  1    3    3    0    0 -0.35471892763126441E+02  0.72521825518040303E-01 -0.55128820698553746E+02  0.65555003401103363E+02  0.10565837150000000E+00 0.0000E+00 0.9000E+01
       -13  1    3    3    0    0  0.35471892763126441E+02 -0.72521825518040303E-01 -0.57961980245258125E+02  0.67954858202023189E+02  0.10565837150000000E+00 0.0000E+00 0.9000E+01
 #aMCatNLO 1  5  2  2  2 0.13261534E+03 0.00000000E+00 -9  5  0 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00
   <mgrwgt>
  0.5492790300D-01    5    3    0
  0.354801137511322D+02 0.000000000000000D+00 0.000000000000000D+00 0.354801137511322D+02
  0.354801137511322D+02 0.000000000000000D+00 0.000000000000000D+00 -.354801137511322D+02
  0.354801137511322D+02 -.354720500509395D+02 0.725221470909378D-01 0.752912687571613D+00
  0.354801137511322D+02 0.354720500509395D+02 -.725221470909378D-01 -.752912687571613D+00
  0.000000000000000D+00 -.000000000000000D+00 -.000000000000000D+00 -.000000000000000D+00
  0.894667454651124D+02 0.000000000000000D+00 0.000000000000000D+00 0.894667454651124D+02
  0.383454720030935D+03 0.000000000000000D+00 0.000000000000000D+00 -.383454720030935D+03
  0.327381002593379D+02 -.922917817134942D+01 0.314012515890186D+02 0.752912687571613D+00
  0.117668624328189D+03 0.943868541136740D+02 0.702599462303913D+02 -.752912687571613D+00
  0.322514740908521D+03 -.851576759423246D+02 -.101661197819410D+03 -.293987974565823D+03
  0.354801137511322D+02 0.000000000000000D+00 0.000000000000000D+00 0.354801137511322D+02
  0.666115104304532D+03 0.000000000000000D+00 0.000000000000000D+00 -.666115104304532D+03
  0.354801137511322D+02 -.354720500509395D+02 0.725221470909378D-01 0.752912687571613D+00
  0.354801137511322D+02 0.354720500509395D+02 -.725221470909378D-01 -.752912687571613D+00
  0.630634990553400D+03 -.000000000000000D+00 -.000000000000000D+00 -.630634990553400D+03
 0.150266932414D-01 0.000000000000D+00 0.000000000000D+00 0.223222474185D-02 0.000000000000D+00  5 2 -2 13 -13 21 400  0 0.15706970D-02 0.18969282D-01 0.12582716D+04 0.12582716D+04 0.12582716D+04 0.13218549D+01  1  2  2  2  5  2       -2 0.206635523771D+03 0.100000000000D+01
 -.834816239710D-03 0.000000000000D+00 -.577064586663D-04 0.223222474185D-02 0.000000000000D+00  5 2 -2 13 -13 21 402  2 0.15706970D-02 0.18969282D-01 0.12582716D+04 0.12582716D+04 0.12582716D+04 0.13218549D+01  1  2  3  2  5  2       -2 -.200585731574D+02 0.100000000000D+01
 0.820471779823D-03 0.000000000000D+00 0.000000000000D+00 0.223222474185D-02 0.000000000000D+00  5 2 -2 13 -13 21 402  2 0.15706970D-02 0.18969282D-01 0.12582716D+04 0.12582716D+04 0.12582716D+04 0.13218549D+01  1  2 15  2  5  2       -2 0.197139112014D+02 0.100000000000D+01
 0.275103314752D-01 0.000000000000D+00 -.198025401068D-01 0.223222474185D-02 0.000000000000D+00  5 2 -2 13 -13 21 402  2 0.15706970D-02 0.35613542D+00 0.12582716D+04 0.12582716D+04 0.12582716D+04 0.13218549D+01  3  2  5  2  5  2       -2 0.638029828359D+00 0.100000000000D+01
 0.383016300600D-02 0.000000000000D+00 -.630348365331D-02 0.223222474185D-02 0.000000000000D+00  5 2 21 13 -13 2 402  2 0.15706970D-02 0.35613542D+00 0.12582716D+04 0.12582716D+04 0.12582716D+04 0.13218549D+01  3  2  5  4  5  2       -2 0.836332244631D+00 0.100000000000D+01
   </mgrwgt>

"""
        evt3 = lhe_parser.Event(input_dy)
        #evt1.parse_reweight() 
        evt3.parse_nlo_weight()
        self.assertTrue(evt3.nloweight.ispureqcd())

        for cevent in evt3.nloweight.cevents:
            self.assertIn(len(cevent), (4,5))