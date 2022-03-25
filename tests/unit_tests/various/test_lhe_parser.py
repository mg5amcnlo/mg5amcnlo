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
        

