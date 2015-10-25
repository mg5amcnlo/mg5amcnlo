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

"""Unit test library for the various base objects of the core library"""

import copy
import itertools
import logging
import math


import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import models.import_ufo as import_ufo
from madgraph import MadGraph5Error, InvalidCmd

#===============================================================================
# AmplitudeTest
#===============================================================================
class AmplitudeTest(unittest.TestCase):
    """Test class for routine functions of the Amplitude object"""

    mydict = {}
    myamplitude = None

    myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':True,
                                      'from_group':False})] * 10)
    myvertexlist = base_objects.VertexList([base_objects.Vertex({'id':3,
                                      'legs':myleglist})] * 10)


    mydiaglist = base_objects.DiagramList([base_objects.Diagram(\
                                        {'vertices':myvertexlist})] * 100)

    myprocess = base_objects.Process()

    def setUp(self):

        self.mydict = {'diagrams':self.mydiaglist, 'process':self.myprocess,
                       'has_mirror_process': False}

        self.myamplitude = diagram_generation.Amplitude(self.mydict)

    def test_setget_amplitude_correct(self):
        "Test correct Amplitude object __init__, get and set"

        myamplitude2 = diagram_generation.Amplitude()

        for prop in self.mydict.keys():
            myamplitude2.set(prop, self.mydict[prop])

        self.assertEqual(self.myamplitude, myamplitude2)

        for prop in self.myamplitude.keys():
            self.assertEqual(self.myamplitude.get(prop), self.mydict[prop])

    def test_setget_amplitude_exceptions(self):
        "Test error raising in Amplitude __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(diagram_generation.Amplitude.PhysicsObjectError,
                          diagram_generation.Amplitude,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          diagram_generation.Amplitude,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.myamplitude.get,
                          a_number)
        self.assertRaises(diagram_generation.Amplitude.PhysicsObjectError,
                          self.myamplitude.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.myamplitude.set,
                          a_number, 0)
        self.assertRaises(diagram_generation.Amplitude.PhysicsObjectError,
                          self.myamplitude.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for amplitude properties"""

        test_values = [{'prop':'diagrams',
                        'right_list':[self.mydiaglist],
                        'wrong_list':['a', {}]}
                       ]

        temp_amplitude = self.myamplitude

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_amplitude.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_amplitude.set(test['prop'], x))

    def test_representation(self):
        """Test amplitude object string representation."""

        goal = "{\n"
        goal = goal + "    \'process\': %s,\n" % repr(self.myprocess)
        goal = goal + "    \'diagrams\': %s,\n" % repr(self.mydiaglist)
        goal = goal + "    \'has_mirror_process\': False\n}"

        self.assertEqual(goal, str(self.myamplitude))

#===============================================================================
# DiagramGenerationTest
#===============================================================================
class DiagramGenerationTest(unittest.TestCase):
    """Test class for all functions related to the diagram generation"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()
    myprocess = base_objects.Process()

    ref_dict_to0 = {}
    ref_dict_to1 = {}

    myamplitude = diagram_generation.Amplitude()

    def setUp(self):

        # A gluon
        self.mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A quark U and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antiu = copy.copy(self.mypartlist[1])
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(self.mypartlist[2])
        antid.set('is_part', False)

        # A photon
        self.mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A electron and positron
        self.mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^+',
                      'antitexname':'e^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antie = copy.copy(self.mypartlist[4])
        antie.set('is_part', False)

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        self.myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[4], \
                                             antie, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))


        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)

        self.ref_dict_to0 = self.myinterlist.generate_ref_dict()[0]
        self.ref_dict_to1 = self.myinterlist.generate_ref_dict()[1]


    def test_combine_legs_gluons(self):
        """Test combine_legs and merge_comb_legs: gg>gg"""

        # Four gluon legs with two initial state
        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'state':True}) \
                                              for num in range(1, 5)])
        myleglist[0].set('state', False)
        myleglist[1].set('state', False)

        l1 = myleglist[0]
        l2 = myleglist[1]
        l3 = myleglist[2]
        l4 = myleglist[3]

        # All possibilities for the first combination
        goal_combined_legs = [
                [(l1, l2), l3, l4], [(l1, l2), (l3, l4)],
                [(l1, l3), l2, l4], [(l1, l3), (l2, l4)],
                [(l1, l4), l2, l3], [(l1, l4), (l2, l3)],
                [l1, (l2, l3), l4], [l1, (l2, l4), l3], [l1, l2, (l3, l4)],
                [(l1, l2, l3), l4], [(l1, l2, l4), l3],
                [(l1, l3, l4), l2], [l1, (l2, l3, l4)]
                ]

        combined_legs = self.myamplitude.combine_legs(
                                              [leg for leg in myleglist],
                                                self.ref_dict_to1,
                                                3)
        self.assertEqual(combined_legs, goal_combined_legs)

        # Now test the reduction of legs for this
        reduced_list = self.myamplitude.merge_comb_legs(combined_legs,
                                                        self.ref_dict_to1)

        # Remaining legs should be from_group False
        l1.set('from_group', False)
        l2.set('from_group', False)
        l3.set('from_group', False)
        l4.set('from_group', False)

        # Define all possible legs obtained after merging combinations
        l12 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':True})
        l13 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':False})
        l14 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':False})
        l23 = base_objects.Leg({'id':21,
                                'number':2,
                                'state':False})
        l24 = base_objects.Leg({'id':21,
                                'number':2,
                                'state':False})
        l34 = base_objects.Leg({'id':21,
                                'number':3,
                                'state':True})
        l123 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':True})
        l124 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':True})
        l134 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':False})
        l234 = base_objects.Leg({'id':21,
                                'number':2,
                                'state':False})

        # Associated vertices
        vx12 = base_objects.Vertex({'legs':base_objects.LegList([l1, l2, l12]), 'id': 1})
        vx13 = base_objects.Vertex({'legs':base_objects.LegList([l1, l3, l13]), 'id': 1})
        vx14 = base_objects.Vertex({'legs':base_objects.LegList([l1, l4, l14]), 'id': 1})
        vx23 = base_objects.Vertex({'legs':base_objects.LegList([l2, l3, l23]), 'id': 1})
        vx24 = base_objects.Vertex({'legs':base_objects.LegList([l2, l4, l24]), 'id': 1})
        vx34 = base_objects.Vertex({'legs':base_objects.LegList([l3, l4, l34]), 'id': 1})
        vx123 = base_objects.Vertex(
            {'legs':base_objects.LegList([l1, l2, l3, l123]), 'id': 2})
        vx124 = base_objects.Vertex(
            {'legs':base_objects.LegList([l1, l2, l4, l124]), 'id': 2})
        vx134 = base_objects.Vertex(
            {'legs':base_objects.LegList([l1, l3, l4, l134]), 'id': 2})
        vx234 = base_objects.Vertex(
            {'legs':base_objects.LegList([l2, l3, l4, l234]), 'id': 2})

        # The final object which should be produced by merge_comb_legs
        goal_reduced_list = [\
                (base_objects.LegList([l12, l3, l4]), \
                 base_objects.VertexList([vx12])), \
                (base_objects.LegList([l12, l34]), \
                 base_objects.VertexList([vx12, \
                                          vx34])), \
                (base_objects.LegList([l13, l2, l4]), \
                 base_objects.VertexList([vx13])), \
                (base_objects.LegList([l13, l24]), \
                 base_objects.VertexList([vx13, \
                                          vx24])), \
                (base_objects.LegList([l14, l2, l3]), \
                 base_objects.VertexList([vx14])), \
                (base_objects.LegList([l14, l23]), \
                 base_objects.VertexList([vx14, \
                                          vx23])), \
                (base_objects.LegList([l1, l23, l4]), \
                 base_objects.VertexList([vx23])), \
                (base_objects.LegList([l1, l24, l3]), \
                 base_objects.VertexList([vx24])), \
                (base_objects.LegList([l1, l2, l34]), \
                 base_objects.VertexList([vx34])), \
                (base_objects.LegList([l123, l4]), \
                 base_objects.VertexList([vx123])), \
                (base_objects.LegList([l124, l3]), \
                 base_objects.VertexList([vx124])), \
                (base_objects.LegList([l134, l2]), \
                 base_objects.VertexList([vx134])), \
                (base_objects.LegList([l1, l234]), \
                 base_objects.VertexList([vx234])), \
                ]

        self.assertEqual(reduced_list, goal_reduced_list)

    def test_combine_legs_uux_ddx(self):
        """Test combine_legs and merge_comb_legs: uu~>dd~"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-2,
                                         'number':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'number':3,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'number':4,
                                         'state':True}))
        l1 = myleglist[0]
        l2 = myleglist[1]
        l3 = myleglist[2]
        l4 = myleglist[3]

        my_combined_legs = [\
                [(l1, l2), l3, l4], [(l1, l2), (l3, l4)], \
                [l1, l2, (l3, l4)] \
                ]

        combined_legs = self.myamplitude.combine_legs(
                                                  [leg for leg in myleglist],
                                                    self.ref_dict_to1, 3)
        self.assertEqual(combined_legs, my_combined_legs)

        reduced_list = self.myamplitude.merge_comb_legs(combined_legs,
                                                        self.ref_dict_to1)

        l1.set('from_group', False)
        l2.set('from_group', False)
        l3.set('from_group', False)
        l4.set('from_group', False)

        l12glue = base_objects.Leg({'id':21,
                                'number':1,
                                'state':True})
        l12phot = base_objects.Leg({'id':22,
                                'number':1,
                                'state':True})
        l34glue = base_objects.Leg({'id':21,
                                'number':3,
                                'state':True})
        l34phot = base_objects.Leg({'id':22,
                                'number':3,
                                'state':True})
        vx12glue = base_objects.Vertex(
            {'legs':base_objects.LegList([l1, l2, l12glue]), 'id':3})
        vx12phot = base_objects.Vertex(
            {'legs':base_objects.LegList([l1, l2, l12phot]), 'id':4})
        vx34glue = base_objects.Vertex(
            {'legs':base_objects.LegList([l3, l4, l34glue]), 'id':5})
        vx34phot = base_objects.Vertex(
            {'legs':base_objects.LegList([l3, l4, l34phot]), 'id':6})

        my_reduced_list = [\
                (base_objects.LegList([l12glue, l3, l4]),
                 base_objects.VertexList([vx12glue])),
                (base_objects.LegList([l12phot, l3, l4]),
                 base_objects.VertexList([vx12phot])),
                (base_objects.LegList([l12glue, l34glue]),
                 base_objects.VertexList([vx12glue, vx34glue])),
                (base_objects.LegList([l12glue, l34phot]),
                 base_objects.VertexList([vx12glue, vx34phot])),
                (base_objects.LegList([l12phot, l34glue]),
                 base_objects.VertexList([vx12phot, vx34glue])),
                (base_objects.LegList([l12phot, l34phot]),
                 base_objects.VertexList([vx12phot, vx34phot])),
                (base_objects.LegList([l1, l2, l34glue]),
                 base_objects.VertexList([vx34glue])),
                (base_objects.LegList([l1, l2, l34phot]),
                 base_objects.VertexList([vx34phot])),
                ]

        self.assertEqual(reduced_list, my_reduced_list)

    def test_combine_legs_uux_uuxuux(self):
        """Test combine_legs: uu~>uu~uu~"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-2,
                                         'number':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':3,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'number':4,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':5,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'number':6,
                                         'state':True}))
        l1 = myleglist[0]
        l2 = myleglist[1]
        l3 = myleglist[2]
        l4 = myleglist[3]
        l5 = myleglist[4]
        l6 = myleglist[5]

        my_combined_legs = [\
                [(l1, l2), l3, l4, l5, l6], [(l1, l2), (l3, l4), l5, l6],
                [(l1, l2), (l3, l4), (l5, l6)], [(l1, l2), (l3, l6), l4, l5],
                [(l1, l2), (l3, l6), (l4, l5)], [(l1, l2), l3, (l4, l5), l6],
                [(l1, l2), l3, l4, (l5, l6)],
                [(l1, l3), l2, l4, l5, l6], [(l1, l3), (l2, l4), l5, l6],
                [(l1, l3), (l2, l4), (l5, l6)], [(l1, l3), (l2, l6), l4, l5],
                [(l1, l3), (l2, l6), (l4, l5)], [(l1, l3), l2, (l4, l5), l6],
                [(l1, l3), l2, l4, (l5, l6)],
                [(l1, l5), l2, l3, l4, l6], [(l1, l5), (l2, l4), l3, l6],
                [(l1, l5), (l2, l4), (l3, l6)], [(l1, l5), (l2, l6), l3, l4],
                [(l1, l5), (l2, l6), (l3, l4)], [(l1, l5), l2, (l3, l4), l6],
                [(l1, l5), l2, (l3, l6), l4],
                [l1, (l2, l4), l3, l5, l6], [l1, (l2, l4), (l3, l6), l5],
                [l1, (l2, l4), l3, (l5, l6)],
                [l1, (l2, l6), l3, l4, l5], [l1, (l2, l6), (l3, l4), l5],
                [l1, (l2, l6), l3, (l4, l5)],
                [l1, l2, (l3, l4), l5, l6], [l1, l2, (l3, l4), (l5, l6)],
                [l1, l2, (l3, l6), l4, l5], [l1, l2, (l3, l6), (l4, l5)],
                [l1, l2, l3, (l4, l5), l6],
                [l1, l2, l3, l4, (l5, l6)]
                ]

        combined_legs = self.myamplitude.combine_legs(
                                              [leg for leg in myleglist],
                                                self.ref_dict_to1, 3)
        self.assertEqual(combined_legs, my_combined_legs)


    def test_diagram_generation_gluons(self):
        """Test the number of diagram generated for gg>ng with n up to 4"""

        goal_ndiags = [1, 4, 25, 220, 2485, 34300]

        # Test 1,2,3 and 4 gluons in the final state
        for ngluon in range (1, 4):

            # Create the amplitude
            myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'state':False})] * 2)

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                                'orders':{'QCD':ngluon},
                                                'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            # Call generate_diagram and output number of diagrams
            self.myamplitude.generate_diagrams()
            ndiags = len(self.myamplitude.get('diagrams'))

            logging.debug("Number of diagrams for %d gluons: %d" % (ngluon,
                                                            ndiags))

            self.assertEqual(ndiags, goal_ndiags[ngluon - 1])

    def test_diagram_generation_uux_gg(self):
        """Test the number of diagram generated for uu~>gg (s, t and u channels)
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        self.myamplitude.set('process', myproc)

        self.myamplitude.generate_diagrams()
        self.assertEqual(len(self.myamplitude.get('diagrams')), 3)

    def test_diagram_generation_uux_uuxng(self):
        """Test the number of diagram generated for uu~>uu~+ng with n up to 2
        """
        goal_ndiags = [4, 18, 120, 1074, 12120]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_ndiags[ngluons])

    def test_diagram_generation_uux_ddxng(self):
        """Test the number of diagram generated for uu~>dd~+ng with n up to 2
        """
        goal_ndiags = [2, 9, 60, 537, 6060]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_ndiags[ngluons])

    def test_diagram_generation_diagrams_ddx_uuxg(self):
        """Test the vertex list output for dd~>uu~g (so far only 2
        diagrams, due to lack of time)
        """
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':-1,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        self.myamplitude.set('process', myproc)

        self.myamplitude.generate_diagrams()
        mydiagrams = self.myamplitude.get('diagrams')

        for leg in myleglist:
            leg.set('from_group', True)

        l1 = myleglist[0]
        l2 = myleglist[1]
        l3 = myleglist[2]
        l4 = myleglist[3]
        l5 = myleglist[4]

        l1.set('id',
               self.mymodel.get('particle_dict')[l1.get('id')].get_anti_pdg_code())
        l2.set('id',
               self.mymodel.get('particle_dict')[l2.get('id')].get_anti_pdg_code())

        l12glue = base_objects.Leg({'id':21,
                                    'number':1,
                                    'state':True})
        l34glue = base_objects.Leg({'id':21,
                                    'number':3,
                                    'state':True})
        l35 = base_objects.Leg({'id':-2,
                                'number':3,
                                'state':True})

        vx12glue = base_objects.Vertex(
            {'legs':base_objects.LegList([l1, l2, l12glue]), 'id':5})
        vx34glue = base_objects.Vertex(
            {'legs':base_objects.LegList([l3, l4, l34glue]), 'id':3})
        vx12glue34glue5 = base_objects.Vertex(
            {'legs':base_objects.LegList([l12glue, l34glue, l5]), 'id':1})
        vx35 = base_objects.Vertex(
            {'legs':base_objects.LegList([l3, l5, l35]), 'id':3})
        vx12glue354 = base_objects.Vertex(
            {'legs':base_objects.LegList([l12glue, l35, l4]), 'id':3})

        goaldiagrams = base_objects.DiagramList([\
            base_objects.Diagram({'vertices': base_objects.VertexList(\
            [vx12glue, vx34glue, vx12glue34glue5]),
                                  'orders':{'QED':0, 'QCD':3, 'WEIGHTED':3}}),
            base_objects.Diagram({'vertices': base_objects.VertexList(\
            [vx12glue, vx35, vx12glue354]),
                                  'orders':{'QED':0, 'QCD':3, 'WEIGHTED':3}})\
            ])

        for diagram in mydiagrams:
            for vertex in diagram.get('vertices'):
                for leg in vertex.get('legs'):
                    leg.set('from_group', True)

        self.assertEqual(goaldiagrams[0:2], mydiagrams[0:2])


    def test_diagram_generation_nodiag(self):
        """Test charge violating processes give 0 diagram
        """

        for nquarks in range(1, 5):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':1,
                                                'state':True})] * nquarks)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.assertRaises(InvalidCmd, self.myamplitude.generate_diagrams)
            self.assertEqual(len(self.myamplitude.get('diagrams')), 0)

    def test_diagram_generation_photons(self):
        """Test the number of diagram generated for uu~>na with n up to 6"""

        # Test up to 5 photons in the final state
        for nphot in range (1, 5):

            # Create the amplitude
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))

            myleglist.extend([base_objects.Leg({'id':22,
                                                'state':True})] * nphot)

            myproc = base_objects.Process({'legs':myleglist,
                                            'orders':{'QED':nphot},
                                            'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            # Call generate_diagram and output number of diagrams
            self.myamplitude.generate_diagrams()
            ndiags = len(self.myamplitude.get('diagrams'))

            logging.debug("Number of diagrams for %d photons: %d" % (nphot,
                                                            ndiags))

            self.assertEqual(ndiags, math.factorial(nphot))

    def test_diagram_generation_electrons(self):
        """Test the number of diagram generated for e+e->n(e+e-) with n up to 3
        """

        goal_ndiags = [2, 36, 1728]
        for npairs in range (1, 3):

            # Create the amplitude
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-11,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':11,
                                             'state':False}))

            myleglist.extend([base_objects.Leg({'id':11,
                                                'state':True}),
                              base_objects.Leg({'id':-11,
                                                'state':True})] * npairs)

            myproc = base_objects.Process({'legs':myleglist,
                                            'orders':{'QED':npairs * 2},
                                            'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            # Call generate_diagram and output number of diagrams
            self.myamplitude.generate_diagrams()
            ndiags = len(self.myamplitude.get('diagrams'))

            logging.debug("Number of diagrams for %d electron pairs: %d" % \
                          (npairs, ndiags))

            self.assertEqual(ndiags, goal_ndiags[npairs - 1])

    def test_expand_list(self):
        """Test the expand_list function"""

        mylist = [[1, 2], 3, [4, 5]]
        goal_list = [[1, 3, 4], [1, 3, 5], [2, 3, 4], [2, 3, 5]]

        self.assertEqual(diagram_generation.expand_list(mylist), goal_list)

        # Also test behavior with singlets like [1]
        mylist = [1, [2]]
        goal_list = [[1, 2]]

        self.assertEqual(diagram_generation.expand_list(mylist), goal_list)

        mylist = [[1]]

        self.assertEqual(diagram_generation.expand_list(mylist), mylist)

        mylist = [[1, 2], [3]]
        goal_list = [[1, 3], [2, 3]]

        self.assertEqual(diagram_generation.expand_list(mylist), goal_list)


    def test_expand_list_list(self):
        """Test the expand_list_list function"""

        mylist = [ [1, 2], [[3, 4], [5, 6]] ]
        goal_list = [[1, 2, 3, 4], [1, 2, 5, 6]]
        self.assertEqual(diagram_generation.expand_list_list(mylist), goal_list)

        mylist = [ [[1, 2], [3, 4]], [5] ]
        goal_list = [[1, 2, 5], [3, 4, 5]]
        self.assertEqual(diagram_generation.expand_list_list(mylist), goal_list)

        mylist = [ [[1, 2], [3, 4]], [[6, 7], [8, 9]] ]
        goal_list = [[1, 2, 6, 7], [1, 2, 8, 9], [3, 4, 6, 7], [3, 4, 8, 9]]
        self.assertEqual(diagram_generation.expand_list_list(mylist), goal_list)

        mylist = [ [[1, 2], [3, 4]], [5], [[6, 7], [8, 9]] ]
        goal_list = [[1, 2, 5, 6, 7], [1, 2, 5, 8, 9], [3, 4, 5, 6, 7],
                     [3, 4, 5, 8, 9]]
        self.assertEqual(diagram_generation.expand_list_list(mylist), goal_list)

    def test_diagram_generation_ue_dve(self):
        """Test the number of diagram generated for ue->dve (t channel)
        """

        mypartlist = base_objects.ParticleList();
        myinterlist = base_objects.InteractionList();

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[len(mypartlist) - 1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^+',
                      'antitexname':'e^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # nu_e
        mypartlist.append(base_objects.Particle({'name':'ve',
                      'antiname':'ve~',
                      'spin':2,
                      'color':0,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\nu_e',
                      'antitexname':'\bar\nu_e',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':12,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        nue = mypartlist[len(mypartlist) - 1]
        nuebar = copy.copy(nue)
        nuebar.set('is_part', False)

        # W
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':0,
                      'mass':'WMASS',
                      'width':'WWIDTH',
                      'texname':'W^+',
                      'antitexname':'W^-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # Coupling of u and d to W

        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             u, \
                                             wminus]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of d and u to W

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             d, \
                                             wplus]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e- and nu_e to W

        myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [nuebar, \
                                             eminus, \
                                             wplus]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of nu_e and e+ to W

        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             nue, \
                                             wminus]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':12,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)

        self.assertEqual(len(myamplitude.get('diagrams')), 1)

    def test_coupling_orders_uux_ddxng(self):
        """Test the number of diagrams uu~>dd~+ng with different QCD
        and QED coupling orders
        """
        goal_ndiags20 = [1, 0, 0]
        goal_ndiags02 = [1, 0, 0]
        goal_ndiags21 = [1, 4, 0]
        goal_ndiags22 = [2, 4, 24]
        goal_ndiags04 = [1, 5, 36]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED':2, 'QCD':0}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_ndiags20[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED':0, 'QCD':2}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_ndiags02[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED':2, 'QCD':1}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_ndiags21[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED':2, 'QCD':2}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_ndiags22[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': {'QED':0, 'QCD':4}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_ndiags04[ngluons])               
                
    def test_squared_orders_constraints_uux_ddxuux(self):
        """ Tests that the various possible squared order constraints are 
        correctly treated at LO."""
        
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,'state':False}))
        myleglist.append(base_objects.Leg({'id':1,'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,'state':True}))
        myleglist.append(base_objects.Leg({'id':2,'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,'state':True}))
        
        SO_tests = [({},{},{},50),
                    ({},{'QED':-1},{'QED':'=='},14),
                    ({},{'QED':-2},{'QED':'=='},38),
                    ({},{'QED':-3},{'QED':'=='},50),
                    ({},{'QED':-4},{'QED':'=='},36),
                    ({},{'QED':-5},{'QED':'=='},12),
                    ({},{'QED':-6},{'QED':'=='},0),
                    ({},{'QCD':4},{'QCD':'>'},38),
                    ({},{'QCD':2},{'QCD':'<='},36),
                    ({},{'QED':2,'QCD':4},{'QED':'==','QCD':'>'},38),
                    ({'QCD':2},{'QED':4,'QCD':4},{'QED':'<=','QCD':'<='},24)]
        
        for orders, sq_orders, sq_orders_type, ndiagGoal in SO_tests:
            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders': orders,
                                           'squared_orders': sq_orders,
                                           'sqorders_types':sq_orders_type})
            self.myamplitude.set('process', myproc)
            self.myamplitude.generate_diagrams()
            self.assertEqual(len(self.myamplitude.get('diagrams')),ndiagGoal) 

    def test_forbidden_particles_uux_uuxng(self):
        """Test the number of diagrams uu~>uu~+g with different 
        forbidden particles.
        """

        goal_no_photon = [2, 10]
        goal_no_photon_quark = [2, 2]

        for ngluons in range(2):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_particles':[22]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_photon[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_particles':[22, 1]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_photon_quark[ngluons])

    def test_forbidden_onshell_s_channel_uux_uuxng(self):
        """Test diagram generation with forbidden onshell s-channel particles.
        """

        goal_no_photon = [4, 18]
        photon_none = [{1:[0]},{2:[0],4:[0],13:[1],17:[1]}]
        goal_no_quark = [2, 6]
        quark_none = [{0:[0]},{0:[0,1],1:[0,1],3:[1],5:[1]}]
        goal_no_antiquark = [2, 6]
        antiquark_none = [{},{}]

        for ngluons in range(2):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_onsh_s_channels':[22]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_photon[ngluons])

            #print self.myamplitude.nice_string()

            diagrams = self.myamplitude.get('diagrams')
            for idiag in range(len(diagrams)):
                if idiag in photon_none[ngluons]:
                    vertices = diagrams[idiag].get('vertices')
                    for ivert in range(len(vertices)):
                        if ivert in photon_none[ngluons][idiag]:
                            self.assertEqual(False,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                        else:
                            self.assertEqual(None,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                else:
                    self.assertFalse(any([vert.get('legs')[-1].get('onshell') == False\
                                          for vert in diagrams[idiag].get('vertices')]))

            # Test with u a > u a (+ g)

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_onsh_s_channels':[1]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_quark[ngluons])

            #print self.myamplitude.nice_string()

            diagrams = self.myamplitude.get('diagrams')
            for idiag in range(len(diagrams)):
                if idiag in quark_none[ngluons]:
                    vertices = diagrams[idiag].get('vertices')
                    for ivert in range(len(vertices)):
                        if ivert in quark_none[ngluons][idiag]:
                            self.assertEqual(False,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                        else:
                            self.assertEqual(None,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                else:
                    self.assertFalse(any([vert.get('legs')[-1].get('onshell') == False\
                                          for vert in diagrams[idiag].get('vertices')]))
            

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_onsh_s_channels':[-1]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_antiquark[ngluons])

            diagrams = self.myamplitude.get('diagrams')
            for idiag in range(len(diagrams)):
                if idiag in antiquark_none[ngluons]:
                    vertices = diagrams[idiag].get('vertices')
                    for ivert in range(len(vertices)):
                        if ivert in antiquark_none[ngluons][idiag]:
                            self.assertEqual(False,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                        else:
                            self.assertEqual(None,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                else:
                    self.assertFalse(any([vert.get('legs')[-1].get('onshell') == False\
                                          for vert in diagrams[idiag].get('vertices')]))
            
    def test_forbidden_s_channel_uux_uuxng(self):
        """Test diagram generation with forbidden s-channel particles.
        """

        goal_no_photon = [3, 14]
        goal_no_quark = [1, 2]
        goal_no_antiquark = [2, 6]

        for ngluons in range(2):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_s_channels':[22]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_photon[ngluons])

            # Test with u a > u a (+ g)

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_s_channels':[1]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_quark[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_s_channels':[-1]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_no_antiquark[ngluons])


    def test_required_s_channel_uux_uuxng(self):
        """Test the number of diagrams uu~>uu~+g with different 
        required s-channel particles.
        """

        goal_req_photon = [1, 4]
        goal_req_quark = [1, 4]
        goal_req_photon_or_gluon = [2, 9]
        goal_req_antiquark = [0, 0]

        for ngluons in range(2):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[22]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_photon[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[21], [22]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_photon_or_gluon[ngluons])

            # Just to make sure that diagrams are not double counted
            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[22], [22]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_photon[ngluons])

            # Test with u a > u a (+ g)

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[1]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_quark[ngluons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[-1]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_antiquark[ngluons])


    def test_required_s_channel_decay(self):
        """Test decay processes d > d u u~ + a with required s-channels.
        """

        goal_req_photon = [1, 4]
        goal_req_d = [0, 2]
        goal_req_u = [0, 1]
        goal_req_u_or_d = [0, 3]
        goal_req_antid = [0, 0]

        for nphotons in range(2):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':1,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':22,
                                                 'state':True})] * nphotons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[22]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_photon[nphotons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[21]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_photon[nphotons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[1, 22]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_d[nphotons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[2, 22]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_u[nphotons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[1, 22],
                                                                  [2, 22]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_u_or_d[nphotons])

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'required_s_channels':[[-1]]})

            self.myamplitude.set('process', myproc)

            self.myamplitude.generate_diagrams()

            self.assertEqual(len(self.myamplitude.get('diagrams')),
                             goal_req_antid[nphotons])

    def test_decay_process_generation(self):
        """Test the decay process generations d > d g g and d > g g d
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc1 = base_objects.Process({'legs':myleglist,
                                        'model':self.mymodel,
                                        'is_decay_chain': True})

        myamplitude1 = diagram_generation.Amplitude()
        myamplitude1.set('process', myproc1)
        myamplitude1.generate_diagrams()

        self.assertEqual(len(myamplitude1.get('diagrams')), 3)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))

        myproc2 = base_objects.Process({'legs':myleglist,
                                        'model':self.mymodel,
                                        'is_decay_chain': True})

        myamplitude2 = diagram_generation.Amplitude()
        myamplitude2.set('process', myproc2)
        myamplitude2.generate_diagrams()

        self.assertEqual(len(myamplitude2.get('diagrams')), 3)


    def test_diagram_generation_identical_interactions(self):
        """Test generation with multiple interactions for same particles
        """

        mypartlist = base_objects.ParticleList();
        myinterlist = base_objects.InteractionList();

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        g = mypartlist[-1]

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        u = mypartlist[-1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GUU'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GDD'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GNP'},
                      'orders':{'NP':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        myamplitude.generate_diagrams()
        diagrams = myamplitude.get('diagrams')
        self.assertEqual(len(diagrams), 2)
        self.assertEqual(diagrams[0].get('orders'),{'QCD':2, 'NP':0, 'WEIGHTED':2})
        self.assertEqual(diagrams[1].get('orders'),{'QCD':1, 'NP':1, 'WEIGHTED':2})

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        myamplitude.generate_diagrams()
        diagrams = myamplitude.get('diagrams')
        self.assertEqual(len(diagrams), 12)
        orders = [{'QCD':3, 'NP':0, 'WEIGHTED':3},
                  {'QCD':2, 'NP':1, 'WEIGHTED':3},
                  {'QCD':2, 'NP':1, 'WEIGHTED':3},
                  {'QCD':1, 'NP':2, 'WEIGHTED':3},
                  {'QCD':3, 'NP':0, 'WEIGHTED':3},
                  {'QCD':2, 'NP':1, 'WEIGHTED':3},
                  {'QCD':2, 'NP':1, 'WEIGHTED':3},
                  {'QCD':1, 'NP':2, 'WEIGHTED':3},
                  {'QCD':3, 'NP':0, 'WEIGHTED':3},
                  {'QCD':2, 'NP':1, 'WEIGHTED':3},
                  {'QCD':3, 'NP':0, 'WEIGHTED':3},
                  {'QCD':2, 'NP':1, 'WEIGHTED':3}]
        for diagram, order in zip(diagrams, orders):
            self.assertEqual(diagram.get('orders'),order)

    def test_multiple_interaction_identical_particles(self):
        """Test the case with multiple interactions for identical particles
        """

        mypartlist = base_objects.ParticleList();
        myinterlist = base_objects.InteractionList();

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        u = mypartlist[-1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        g = mypartlist[-1]

        # two different couplings  u u g

        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQCD'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GEFF'},
                      'orders':{'EFF':1}}))

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)

        self.assertEqual(len(myamplitude.get('diagrams')), 8)
        
        goal_lastvx = set([21,2,-2])
        for diag in myamplitude.get('diagrams'):
            self.assertEqual(set([l.get('id') for l in \
                                  diag.get('vertices')[-1].get('legs')]),
                             goal_lastvx)

#===============================================================================
# Muliparticle test
#===============================================================================
class MultiparticleTest(unittest.TestCase):
    """Test class for processes with multiparticle labels"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()
    myprocess = base_objects.Process()

    def setUp(self):

        # A gluon
        self.mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A quark U and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antiu = copy.copy(self.mypartlist[1])
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(self.mypartlist[2])
        antid.set('is_part', False)

        # A photon
        self.mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A electron and positron
        self.mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^+',
                      'antitexname':'e^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antie = copy.copy(self.mypartlist[4])
        antie.set('is_part', False)

        # 3 gluon vertiex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        self.myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[4], \
                                             antie, \
                                             self.mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)

#===============================================================================
# DecayChainAmplitudeTest
#===============================================================================
class DecayChainAmplitudeTest(unittest.TestCase):
    """Test class for the DecayChainAmplitude object"""

    mydict = {}
    mymodel = base_objects.Model()
    my_amplitudes = diagram_generation.AmplitudeList()
    my_decay_chains = diagram_generation.DecayChainAmplitudeList()
    my_decay_chain = diagram_generation.DecayChainAmplitude()

    def setUp(self):

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antiu = copy.copy(mypartlist[1])
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(mypartlist[2])
        antid.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^+',
                      'antitexname':'e^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antie = copy.copy(mypartlist[4])
        antie.set('is_part', False)

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[4], \
                                             antie, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)

        self.mydict = {'amplitudes': self.my_amplitudes,
                       'decay_chains': self.my_decay_chains}

        self.my_decay_chain = diagram_generation.DecayChainAmplitude(\
            self.mydict)

    def test_setget_process_correct(self):
        "Test correct DecayChainAmplitude object __init__, get and set"

        myprocess2 = diagram_generation.DecayChainAmplitude()

        for prop in self.mydict.keys():
            myprocess2.set(prop, self.mydict[prop])

        self.assertEqual(self.my_decay_chain, myprocess2)

        
    def test_setget_process_exceptions(self):
        "Test error raising in DecayChainAmplitude __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(diagram_generation.DecayChainAmplitude.PhysicsObjectError,
                          diagram_generation.DecayChainAmplitude,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          diagram_generation.DecayChainAmplitude,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.my_decay_chain.get,
                          a_number)
        self.assertRaises(diagram_generation.DecayChainAmplitude.PhysicsObjectError,
                          self.my_decay_chain.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.my_decay_chain.set,
                          a_number, 0)
        self.assertRaises(diagram_generation.DecayChainAmplitude.PhysicsObjectError,
                          self.my_decay_chain.set,
                          'wrongparam', 0)

    def test_representation(self):
        """Test process object string representation."""

        goal = "{\n"
        goal = goal + "    \'amplitudes\': %s,\n" % repr(diagram_generation.AmplitudeList())
        goal = goal + "    \'decay_chains\': %s\n}" % repr(diagram_generation.AmplitudeList())

        self.assertEqual(goal, str(self.my_decay_chain))

    def test_decay_chain_pp_jj(self):
        """Test a decay chain process pp > jj, j > jj based on
        multiparticle lists
        """

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        
        my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                'model':self.mymodel})
        my_decay_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        my_decay_leglist[0].set('state', False)
        my_decay_processes = base_objects.ProcessDefinition({\
                               'legs':my_decay_leglist,
                               'model':self.mymodel})

        my_process_definition.set('decay_chains',
                                  base_objects.ProcessDefinitionList(\
                                    [my_decay_processes]))

        my_decay_chain_amps = diagram_generation.DecayChainAmplitude(\
                                                   my_process_definition)
        
        self.assertEqual(len(my_decay_chain_amps.get('amplitudes')), 35)
        self.assertEqual(len(my_decay_chain_amps.get('decay_chains')), 1)
        self.assertEqual(len(my_decay_chain_amps.get('decay_chains')[0].\
                             get('amplitudes')), 15)
        # Check that all onshell flags are set appropriately
        for amp in my_decay_chain_amps.get('amplitudes'):
            for diagram in amp.get('diagrams'):
                external = set()
                for vertex in diagram.get('vertices'):
                    for l in vertex.get('legs'):
                        self.assertTrue(l.get('onshell') and l.get('state') and \
                                        not l.get('number') in external or \
                                        not l.get('onshell') and (not l.get('state') or \
                                        l.get('number') in external))
                        external.add(l.get('number'))
 
    def test_unused_decays_in_decay_chain_pp_jj(self):
        """Test removal of unused decays in decay chain qq > qq, j > jj
        """

        p = [1, -1, 2, -2, 21]
        q = [1, -1, 2, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': q, 'state': True});

        # Define the multiprocess
        my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        
        my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                'model':self.mymodel})
        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        my_decay_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        my_decay_leglist[0].set('state', False)
        my_decay_processes = base_objects.ProcessDefinition({\
                               'legs':my_decay_leglist,
                               'model':self.mymodel})

        my_process_definition.set('decay_chains',
                                  base_objects.ProcessDefinitionList(\
                                    [my_decay_processes]))

        decay_chain = diagram_generation.DecayChainAmplitude(\
                                                   my_process_definition)
        # Check that all decays are quarks, no gluons
        for dc_amp in decay_chain.get('decay_chains')[0].get('amplitudes'):
            self.assertTrue(dc_amp.get('process').get('legs')[0].get('id') in q)

    def test_forbidden_s_channel_decay_chain(self):
        """Test decay chains with forbidden s-channel particles.
        """

        goal_no_quark = 6
        quark_none = {0:[0,1],1:[0,1],3:[1],5:[1]}

        # Test with u a > u a (+ g)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.extend([base_objects.Leg({'id':21,
                                             'state':True})])

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel,
                                       'forbidden_onsh_s_channels':[1]})

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        mydecayproc = base_objects.Process({'legs':myleglist,
                                            'model':self.mymodel})
        myproc.set('decay_chains', base_objects.ProcessList([mydecayproc]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)

        #print myamplitude.nice_string()

        self.assertEqual(len(myamplitude.get('amplitudes')[0].get('diagrams')),
                         goal_no_quark)

        #print myamplitude.nice_string()

        diagrams = myamplitude.get('amplitudes')[0].get('diagrams')
        for idiag in range(len(diagrams)):
            if idiag in quark_none:
                vertices = diagrams[idiag].get('vertices')
                for ivert in range(len(vertices)):
                    if ivert in quark_none[idiag]:
                        self.assertEqual(False,
                                vertices[ivert].get('legs')[-1].get('onshell'))
                    else:
                        self.assertEqual(None,
                                vertices[ivert].get('legs')[-1].get('onshell'))
            else:
                self.assertFalse(any([vert.get('legs')[-1].get('onshell') == False\
                                      for vert in diagrams[idiag].get('vertices')]))
            

#===============================================================================
# MultiProcessTest
#===============================================================================
class MultiProcessTest(unittest.TestCase):
    """Test class for the MultiProcess object"""

    mydict = {}
    my_process_definition = None
    mymodel = base_objects.Model()
    my_multi_leglist = base_objects.MultiLegList()
    my_process_definitions = base_objects.ProcessDefinitionList()
    my_processes = base_objects.ProcessList()
    my_multi_process = diagram_generation.MultiProcess()

    def setUp(self):

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antiu = copy.copy(mypartlist[1])
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(mypartlist[2])
        antid.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^-',
                      'antitexname':'e^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antie = copy.copy(mypartlist[4])
        antie.set('is_part', False)

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[4], \
                                             antie, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)
        self.mymodel.set('order_hierarchy', {'QCD':1, 'QED':2})

        self.my_multi_leglist = base_objects.MultiLegList(\
            [copy.copy(base_objects.MultiLeg({'ids':[3, 4, 5],
                                              'state':True})) for \
             dummy in range(5)])

        self.my_multi_leglist[0].set('state', False)
        self.my_multi_leglist[1].set('state', False)

        mydict = {'legs':self.my_multi_leglist,
                  'orders':{'QCD':5, 'QED':1},
                  'model':self.mymodel,
                  'id':3}

        self.my_process_definition = base_objects.ProcessDefinition(mydict)
        self.my_process_definitions = base_objects.ProcessDefinitionList(\
            [self.my_process_definition])

        self.mydict = {'process_definitions':self.my_process_definitions}

        self.my_multi_process = diagram_generation.MultiProcess(\
            self.mydict)

    def test_setget_process_correct(self):
        "Test correct MultiProcess object __init__, get and set"

        myprocess2 = diagram_generation.MultiProcess()

        for prop in self.mydict.keys():
            myprocess2.set(prop, self.mydict[prop])

        self.assertEqual(self.my_multi_process, myprocess2)


    def test_setget_process_exceptions(self):
        "Test error raising in MultiProcess __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(diagram_generation.MultiProcess.PhysicsObjectError,
                          diagram_generation.MultiProcess,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          diagram_generation.MultiProcess,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.my_multi_process.get,
                          a_number)
        self.assertRaises(diagram_generation.MultiProcess.PhysicsObjectError,
                          self.my_multi_process.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.my_multi_process.set,
                          a_number, 0)
        self.assertRaises(diagram_generation.MultiProcess.PhysicsObjectError,
                          self.my_multi_process.set,
                          'wrongparam', 0)

    def test_representation(self):
        """Test process object string representation."""

        goal = "{\n"
        goal = goal + "    \'process_definitions\': %s,\n" % repr(self.my_process_definitions)
        goal = goal + "    \'amplitudes\': %s\n}" % repr(diagram_generation.AmplitudeList())

        self.assertEqual(goal, str(self.my_multi_process))

    def test_multiparticle_pp_nj(self):
        """Setting up and testing pp > nj based on multiparticle lists,
        using the amplitude functionality of MultiProcess
        (which makes partial use of crossing symmetries)
        """

        max_fs = 2 # 3

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        goal_number_processes = [219, 379]

        goal_valid_procs = []
        goal_valid_procs.append([([1, 1, 1, 1], 4),
                                 ([1, -1, 1, -1], 4),
                                 ([1, -1, 2, -2], 2),
                                 ([1, -1, 21, 21], 3),
                                 ([1, 2, 1, 2], 2),
                                 ([1, -2, 1, -2], 2),
                                 ([1, 21, 1, 21], 3),
                                 ([-1, 1, 1, -1], 4),
                                 ([-1, 1, 2, -2], 2),
                                 ([-1, 1, 21, 21], 3),
                                 ([-1, -1, -1, -1], 4),
                                 ([-1, 2, -1, 2], 2),
                                 ([-1, -2, -1, -2], 2),
                                 ([-1, 21, -1, 21], 3),
                                 ([2, 1, 1, 2], 2),
                                 ([2, -1, -1, 2], 2),
                                 ([2, 2, 2, 2], 4),
                                 ([2, -2, 1, -1], 2),
                                 ([2, -2, 2, -2], 4),
                                 ([2, -2, 21, 21], 3),
                                 ([2, 21, 2, 21], 3),
                                 ([-2, 1, 1, -2], 2),
                                 ([-2, -1, -1, -2], 2),
                                 ([-2, 2, 1, -1], 2),
                                 ([-2, 2, 2, -2], 4),
                                 ([-2, 2, 21, 21], 3),
                                 ([-2, -2, -2, -2], 4),
                                 ([-2, 21, -2, 21], 3),
                                 ([21, 1, 1, 21], 3),
                                 ([21, -1, -1, 21], 3),
                                 ([21, 2, 2, 21], 3),
                                 ([21, -2, -2, 21], 3),
                                 ([21, 21, 1, -1], 3),
                                 ([21, 21, 2, -2], 3),
                                 ([21, 21, 21, 21], 4)])
        goal_valid_procs.append([([1, 1, 1, 1, 21], 18),
                                 ([1, -1, 1, -1, 21], 18),
                                 ([1, -1, 2, -2, 21], 9),
                                 ([1, -1, 21, 21, 21], 16),
                                 ([1, 2, 1, 2, 21], 9),
                                 ([1, -2, 1, -2, 21], 9),
                                 ([1, 21, 1, 1, -1], 18),
                                 ([1, 21, 1, 2, -2], 9),
                                 ([1, 21, 1, 21, 21], 16),
                                 ([-1, 1, 1, -1, 21], 18),
                                 ([-1, 1, 2, -2, 21], 9),
                                 ([-1, 1, 21, 21, 21], 16),
                                 ([-1, -1, -1, -1, 21], 18),
                                 ([-1, 2, -1, 2, 21], 9),
                                 ([-1, -2, -1, -2, 21], 9),
                                 ([-1, 21, 1, -1, -1], 18),
                                 ([-1, 21, -1, 2, -2], 9),
                                 ([-1, 21, -1, 21, 21], 16),
                                 ([2, 1, 1, 2, 21], 9),
                                 ([2, -1, -1, 2, 21], 9),
                                 ([2, 2, 2, 2, 21], 18),
                                 ([2, -2, 1, -1, 21], 9),
                                 ([2, -2, 2, -2, 21], 18),
                                 ([2, -2, 21, 21, 21], 16),
                                 ([2, 21, 1, -1, 2], 9),
                                 ([2, 21, 2, 2, -2], 18),
                                 ([2, 21, 2, 21, 21], 16),
                                 ([-2, 1, 1, -2, 21], 9),
                                 ([-2, -1, -1, -2, 21], 9),
                                 ([-2, 2, 1, -1, 21], 9),
                                 ([-2, 2, 2, -2, 21], 18),
                                 ([-2, 2, 21, 21, 21], 16),
                                 ([-2, -2, -2, -2, 21], 18),
                                 ([-2, 21, 1, -1, -2], 9),
                                 ([-2, 21, 2, -2, -2], 18),
                                 ([-2, 21, -2, 21, 21], 16),
                                 ([21, 1, 1, 1, -1], 18),
                                 ([21, 1, 1, 2, -2], 9),
                                 ([21, 1, 1, 21, 21], 16),
                                 ([21, -1, 1, -1, -1], 18),
                                 ([21, -1, -1, 2, -2], 9),
                                 ([21, -1, -1, 21, 21], 16),
                                 ([21, 2, 1, -1, 2], 9),
                                 ([21, 2, 2, 2, -2], 18),
                                 ([21, 2, 2, 21, 21], 16),
                                 ([21, -2, 1, -1, -2], 9),
                                 ([21, -2, 2, -2, -2], 18),
                                 ([21, -2, -2, 21, 21], 16),
                                 ([21, 21, 1, -1, 21], 16),
                                 ([21, 21, 2, -2, 21], 16),
                                 ([21, 21, 21, 21, 21], 25)])


        for nfs in range(2, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * (2 + nfs)])

            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)

            my_process_definition = base_objects.ProcessDefinition({\
                                                     'legs':my_multi_leglist,
                                                     'model':self.mymodel,
                                                     'orders': {'QED': nfs}})
            my_multiprocess = diagram_generation.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            nproc = 0

            # Calculate diagrams for all processes

            amplitudes = my_multiprocess.get('amplitudes')

            valid_procs = [([leg.get('id') for leg in \
                             amplitude.get('process').get('legs')],
                            len(amplitude.get('diagrams'))) \
                           for amplitude in amplitudes]

            if nfs <= 3:
                self.assertEqual(valid_procs, goal_valid_procs[nfs-2])

            #print 'pp > ',nfs,'j (p,j = ', p, '):'
            #print 'Valid processes: ',len(filter(lambda item: item[1] > 0, valid_procs))
            #print 'Attempted processes: ',len(amplitudes)

    def test_multiparticle_stop_decay(self):
        """Test that process mirroring is not used in the process st > st g
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()
        mymodel = base_objects.Model()

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        g = mypartlist[-1]

        # Two stop squarks
        mypartlist.append(base_objects.Particle({'name':'t1',
                      'antiname':'t1~',
                      'spin':1,
                      'color':3,
                      'mass':'Mt1',
                      'width':'Wt1',
                      'texname':'t1',
                      'antitexname':'\bar t1',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':1000006,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t1 = mypartlist[-1]
        t1bar = copy.copy(t1)
        t1bar.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t2',
                      'antiname':'t2~',
                      'spin':1,
                      'color':3,
                      'mass':'Mt2',
                      'width':'Wt2',
                      'texname':'t2',
                      'antitexname':'\bar t2',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2000006,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t2 = mypartlist[-1]
        t2bar = copy.copy(t2)
        t2bar.set('is_part', False)

        # Gluon couplings to squarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [t1bar, \
                                             t1, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [t2bar, \
                                             t2, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [t1bar, \
                                             t2, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G12G'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [t2bar, \
                                             t1, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G12G'},
                      'orders':{'QCD':1}}))

        mymodel.set('particles', mypartlist)
        mymodel.set('interactions',myinterlist)
        
        max_fs = 2

        p = [1000006, 2000006, -1000006, -2000006]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        goal_number_processes = [8]

        goal_valid_procs = []
        goal_valid_procs.append([([1000006, 1000006, 21], 1), 
                                 ([1000006, 2000006, 21], 1), 
                                 ([2000006, 1000006, 21], 1), 
                                 ([2000006, 2000006, 21], 1), 
                                 ([-1000006, -1000006, 21], 1), 
                                 ([-1000006, -2000006, 21], 1), 
                                 ([-2000006, -1000006, 21], 1), 
                                 ([-2000006, -2000006, 21], 1)])

        for nfs in range(2, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 2])
            my_multi_leglist += [copy.copy(base_objects.MultiLeg({'ids':[21]}))\
                                 for n in range(2, nfs + 1)]

            my_multi_leglist[0].set('state', False)

            my_process_definition = base_objects.ProcessDefinition({\
                                                     'legs':my_multi_leglist,
                                                     'model':mymodel,
                                                     'orders': {'QED': nfs}})
            my_multiprocess = diagram_generation.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])},
                collect_mirror_procs = True)

            nproc = 0

            # Calculate diagrams for all processes

            amplitudes = my_multiprocess.get('amplitudes')

            valid_procs = [([leg.get('id') for leg in \
                             amplitude.get('process').get('legs')],
                            len(amplitude.get('diagrams'))) \
                           for amplitude in amplitudes]

#            print 'pp > ',nfs,'j (p,j = ', p, '):'
#            print 'Valid processes: ',valid_procs

            if nfs <= 3:
                self.assertEqual(valid_procs, goal_valid_procs[nfs-2])

    def test_heft_multiparticle_pp_hnj(self):
        """Test pp > h+nj in HEFT, which tests new optimize_orders
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name':'g',
                      'antiname':'g',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':21,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        g = mypartlist[-1]

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antiu = copy.copy(mypartlist[1])
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(mypartlist[2])
        antid.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A higgs
        mypartlist.append(base_objects.Particle({'name':'h',
                      'antiname':'h',
                      'spin':1,
                      'color':1,
                      'mass':'MH',
                      'width':'WH',
                      'texname':'h',
                      'antitexname':'h',
                      'line':'dashed',
                      'charge':0.,
                      'pdg_code':25,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        h = mypartlist[-1]

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 3),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 4),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[1], \
                                             antiu, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[2], \
                                             antid, \
                                             mypartlist[3]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Couplings of h to g

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [g,
                                             g, \
                                             h]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GGH'},
                      'orders':{'HIG':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [g,
                                             g,
                                             g,
                                             h]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GGGH'},
                      'orders':{'HIG':1, 'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [g,
                                             g,
                                             g,
                                             g,
                                             h]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GGGGH'},
                      'orders':{'HIG':1, 'QCD':2}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)
        mymodel.set('order_hierarchy', {'QCD':1, 'HIG':1, 'QED':2})
        mymodel.set('expansion_order', {'QCD':-1, 'HIG':1, 'QED':-1})
        max_fs = 2

        p = [21, 1, 2, -1, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        goal_number_processes = [7, 21]

        goal_valid_procs = [[([21, 21, 21, 25], 4),
                             ([21, 1, 1, 25], 1),
                             ([21, 2, 2, 25], 1),
                             ([21, -1, -1, 25], 1),
                             ([21, -2, -2, 25], 1),
                             ([1, -1, 21, 25], 1),
                             ([2, -2, 21, 25], 1)],
                            [([21, 21, 21, 21, 25], 26),
                             ([21, 21, 1, -1, 25], 8),
                             ([21, 21, 2, -2, 25], 8),
                             ([21, 1, 21, 1, 25], 8),
                             ([21, 2, 21, 2, 25], 8),
                             ([21, -1, 21, -1, 25], 8),
                             ([21, -2, 21, -2, 25], 8),
                             ([1, 1, 1, 1, 25], 2),
                             ([1, 2, 1, 2, 25], 1),
                             ([1, -1, 21, 21, 25], 8),
                             ([1, -1, 1, -1, 25], 2),
                             ([1, -1, 2, -2, 25], 1),
                             ([1, -2, 1, -2, 25], 1),
                             ([2, 2, 2, 2, 25], 2),
                             ([2, -1, 2, -1, 25], 1),
                             ([2, -2, 21, 21, 25], 8),
                             ([2, -2, 1, -1, 25], 1),
                             ([2, -2, 2, -2, 25], 2),
                             ([-1, -1, -1, -1, 25], 2),
                             ([-1, -2, -1, -2, 25], 1),
                             ([-2, -2, -2, -2, 25], 2)]]

        for nfs in range(1, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * (2 + nfs)])
            my_multi_leglist.append(base_objects.MultiLeg({'ids': [25]}))
            
            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)

            my_process_definition = base_objects.ProcessDefinition({\
                                                     'legs':my_multi_leglist,
                                                     'model':mymodel})
            my_multiprocess = diagram_generation.MultiProcess(\
                           my_process_definition, collect_mirror_procs =  True)

            nproc = 0

            # Calculate diagrams for all processes

            amplitudes = my_multiprocess.get('amplitudes')

            valid_procs = [([leg.get('id') for leg in \
                             amplitude.get('process').get('legs')],
                            len(amplitude.get('diagrams'))) \
                           for amplitude in amplitudes]
                            
            if nfs <= 3:
                self.assertEqual(valid_procs, goal_valid_procs[nfs-1])

            #print 'pp > h + ',nfs,'j (p,j = ', p, '):'
            #print 'Processes: ',len(amplitudes), \
            #      ' with ', sum([v[1] for v in valid_procs]),
            #for amplitude in amplitudes:
            #    print amplitude.get('process').nice_string()
            #print 'valid_procs = ',valid_procs

    def test_multiparticle_mirror_pp_3j(self):
        """Setting up and testing pp > 3j mirror process functionality
        """

        max_fs = 3

        p = [21, 1, 2, -1, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        goal_number_processes = 29

        goal_legs_mirror = [\
            ([21, 21, 21, 21, 21], False),
            ([21, 21, 21, 1, -1], False),
            ([21, 21, 21, 2, -2], False),
            ([21, 1, 21, 21, 1], True),
            ([21, 1, 1, 1, -1], True),
            ([21, 1, 1, 2, -2], True),
            ([21, 2, 21, 21, 2], True),
            ([21, 2, 1, 2, -1], True),
            ([21, 2, 2, 2, -2], True),
            ([21, -1, 21, 21, -1], True),
            ([21, -1, 1, -1, -1], True),
            ([21, -1, 2, -1, -2], True),
            ([21, -2, 21, 21, -2], True),
            ([21, -2, 1, -1, -2], True),
            ([21, -2, 2, -2, -2], True),
            ([1, 1, 21, 1, 1], False),
            ([1, 2, 21, 1, 2], True),
            ([1, -1, 21, 21, 21], True),
            ([1, -1, 21, 1, -1], True),
            ([1, -1, 21, 2, -2], True),
            ([1, -2, 21, 1, -2], True),
            ([2, 2, 21, 2, 2], False),
            ([2, -1, 21, 2, -1], True),
            ([2, -2, 21, 21, 21], True),
            ([2, -2, 21, 1, -1], True),
            ([2, -2, 21, 2, -2], True),
            ([-1, -1, 21, -1, -1], False),
            ([-1, -2, 21, -1, -2], True),
            ([-2, -2, 21, -2, -2], False)]

        # Define the multiprocess
        my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 5])

        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)

        my_process_definition = base_objects.ProcessDefinition({\
                                                 'legs':my_multi_leglist,
                                                 'model':self.mymodel,
                                                 'orders': {'QED': 0}})

        # Calculate diagrams for all processes
        myproc = diagram_generation.MultiProcess(my_process_definition,
                                      collect_mirror_procs = True)

        amplitudes = myproc.get('amplitudes')

        legs_mirror = [([l.get('id') for l in a.get('process').get('legs')],
                        a.get('has_mirror_process')) for a in amplitudes]

        self.assertEqual(legs_mirror, goal_legs_mirror)

    def test_find_optimal_order(self):
        """Test find_optimal_process_orders for different configurations
        """

        # First try p p > e+ e- + nj

        max_fs = 5

        p = [21, 1, -1, 2, -2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        orders = [4, 5, 6, 7, 8]
        for nfs in range(2, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * (nfs)])

            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)
            
            my_multi_leglist.append(base_objects.MultiLeg({'ids': [11],
                                                           'state': True}))
            my_multi_leglist.append(base_objects.MultiLeg({'ids': [-11],
                                                           'state': True}))

            my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                    'model':self.mymodel})

            # Check coupling orders for process
            self.assertEqual(diagram_generation.MultiProcess.\
                             find_optimal_process_orders(my_process_definition),
                             {'WEIGHTED': orders[nfs-2]})

        # Now check p p > a > p p
        max_fs = 3

        orders = [4, 5]
        for nfs in range(2, max_fs + 1):
            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for \
                                               leg in [my_multi_leg] * (2+nfs)])
            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)
            my_process_definition = base_objects.ProcessDefinition({\
                                                  'legs':my_multi_leglist,
                                                  'model':self.mymodel,
                                                  'required_s_channels':[22]})

            self.assertEqual(diagram_generation.MultiProcess.\
                             find_optimal_process_orders(my_process_definition),
                             {'WEIGHTED': orders[nfs-2]})
        
        # Now check p p > a|g > p p
        max_fs = 3
        
        orders = [2, 3]
        for nfs in range(2, max_fs + 1):
            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for \
                                               leg in [my_multi_leg] * (2+nfs)])
            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)
            my_process_definition = base_objects.ProcessDefinition({\
                                             'legs':my_multi_leglist,
                                             'model':self.mymodel,
                                             'required_s_channels':[[22],[21]]})

            self.assertEqual(diagram_generation.MultiProcess.\
                             find_optimal_process_orders(my_process_definition),
                             {'WEIGHTED': orders[nfs-2]})
        

        # Check that it works with multiple non-QCD orders.

        myoldinterlist = self.mymodel.get('interactions')
        myinterlist = copy.copy(myoldinterlist)
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            []),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'SQED':1}}))

        self.mymodel.set('interactions', myinterlist)
        self.mymodel.set('order_hierarchy', {'QCD':1, 'QED':2, 'SQED':2})
        self.assertEqual(diagram_generation.MultiProcess.\
                         find_optimal_process_orders(my_process_definition),
                         {'WEIGHTED': orders[nfs-2]})
        
        self.mymodel.set('interactions', myoldinterlist)

        # Now check decay process p > p (a|g)
        max_fs = 3
        orders = [1, 2]

        ag = [21, 22]
        my_ag_leg = base_objects.MultiLeg({'ids': ag, 'state': True});

        for nfs in range(2, max_fs + 1):
            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for \
                                               leg in [my_multi_leg] * 2])
            
            my_multi_leglist.extend([copy.copy(leg) for \
                                               leg in [my_ag_leg] * (nfs-1)])
            my_multi_leglist[0].set('state', False)
            my_process_definition = base_objects.ProcessDefinition({\
                                             'legs':my_multi_leglist,
                                             'model':self.mymodel})

            self.assertEqual(diagram_generation.MultiProcess.\
                             find_optimal_process_orders(my_process_definition),
                             {})
        
            my_process_definition.set('is_decay_chain', True)
            self.assertEqual(diagram_generation.MultiProcess.\
                             find_optimal_process_orders(my_process_definition),
                             {'WEIGHTED': orders[nfs-2]})
        

    def test_multiparticle_pp_nj_with_required_s_channel(self):
        """Setting up and testing pp > nj with required photon s-channel
        """

        max_fs = 2 # 3

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        goal_number_processes = [8, 24]

        goal_valid_procs = []
        goal_valid_procs.append([([1, -1, 1, -1], 1),
                                 ([1, -1, 2, -2], 1),
                                 ([-1, 1, 1, -1], 1),
                                 ([-1, 1, 2, -2], 1),
                                 ([2, -2, 1, -1], 1),
                                 ([2, -2, 2, -2], 1),
                                 ([-2, 2, 1, -1], 1),
                                 ([-2, 2, 2, -2], 1)])
        goal_valid_procs.append([([1, -1, 1, -1, 21], 4),
                                 ([1, -1, 2, -2, 21], 4),
                                 ([1, 21, 1, 1, -1], 4),
                                 ([1, 21, 1, 2, -2], 2),
                                 ([-1, 1, 1, -1, 21], 4),
                                 ([-1, 1, 2, -2, 21], 4),
                                 ([-1, 21, 1, -1, -1], 4),
                                 ([-1, 21, -1, 2, -2], 2),
                                 ([2, -2, 1, -1, 21], 4),
                                 ([2, -2, 2, -2, 21], 4),
                                 ([2, 21, 1, -1, 2], 2),
                                 ([2, 21, 2, 2, -2], 4),
                                 ([-2, 2, 1, -1, 21], 4),
                                 ([-2, 2, 2, -2, 21], 4),
                                 ([-2, 21, 1, -1, -2], 2),
                                 ([-2, 21, 2, -2, -2], 4),
                                 ([21, 1, 1, 1, -1], 4),
                                 ([21, 1, 1, 2, -2], 2),
                                 ([21, -1, 1, -1, -1], 4),
                                 ([21, -1, -1, 2, -2], 2),
                                 ([21, 2, 1, -1, 2], 2),
                                 ([21, 2, 2, 2, -2], 4),
                                 ([21, -2, 1, -1, -2], 2),
                                 ([21, -2, 2, -2, -2], 4)])


        for nfs in range(2, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * (2 + nfs)])

            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)

            my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                    'model':self.mymodel,
                                                                    'required_s_channels': [[22]]})
            my_multiprocess = diagram_generation.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            if nfs <= 3:
                self.assertEqual(len(my_multiprocess.get('amplitudes')),
                                 goal_number_processes[nfs - 2])

            # Calculate diagrams for all processes

            #amplitudes = my_multiprocess.get('amplitudes')

            valid_procs = [([leg.get('id') for leg in \
                             amplitude.get('process').get('legs')],
                            len(amplitude.get('diagrams'))) \
                           for amplitude in my_multiprocess.get('amplitudes')]

            if nfs <= 3:
                self.assertEqual(valid_procs, goal_valid_procs[nfs - 2])

    def test_wrong_multiparticle(self):
        """Check that an exception is raised for empty multipart amplitudes"""
        
        max_fs = 2 # 3

        p = [-1, -2]
        j = [ 1,  2]

        my_multi_init = base_objects.MultiLeg({'ids': p, 'state': False});
        my_multi_final = base_objects.MultiLeg({'ids': j, 'state': True});
        goal_number_processes = [0, 0]
        
        for nfs in range(2, max_fs + 1):
            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList(
                          [copy.copy(leg) for leg in [my_multi_init] * 2] + \
                          [copy.copy(leg) for leg in [my_multi_final] * nfs]
                          )

            my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                    'model':self.mymodel}
                                                                  )
            my_multiprocess = diagram_generation.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            if nfs <= 3:
                self.assertRaises(MadGraph5Error,
                                  my_multiprocess.get, 'amplitudes')

    def test_crossing_uux_gg(self):
        """Test the number of diagram generated for uu~>gg (s, t and u channels)
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        crossproc = base_objects.Process({'legs':myleglist,
                                          'model':self.mymodel})

        crossamp = diagram_generation.MultiProcess.cross_amplitude(myamplitude,
                                                                   crossproc,
                                                                   [3,4,2,1],
                                                                   [2,4,1,3])
        crossed_numbers =  [[[3, 1, 1], [2, 4, 1]],
                            [[3, 2, 2], [1, 4, 2]],
                            [[3, 4, 3], [1, 2, 3]]]
        crossed_states =  [[[True, False, False], [False, True, False]],
                           [[True, False, False], [False, True, False]],
                           [[True, True, True], [False, False, True]]]
        for idiag, diagram in enumerate(crossamp.get('diagrams')):
            self.assertEqual([[l.get('number') for l in v.get('legs')] \
                              for v in diagram.get('vertices')],
                             crossed_numbers[idiag])
            self.assertEqual([[l.get('state') for l in v.get('legs')] \
                              for v in diagram.get('vertices')],
                             crossed_states[idiag])
#===============================================================================
# TestDiagramTag
#===============================================================================
class TestDiagramTag(unittest.TestCase):
    """Test class for the DiagramTag class"""


    def setUp(self):
        self.base_model = import_ufo.import_model('sm')
    
    def test_diagram_tag_gg_ggg(self):
        """Test the diagram tag for gg > ggg"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude = diagram_generation.Amplitude(myproc)

        tags = []
        permutations = []
        diagram_classes = []
        for idiag, diagram in enumerate(myamplitude.get('diagrams')):
            tag = diagram_generation.DiagramTag(diagram)
            try:
                ind = tags.index(tag)
            except:
                diagram_classes.append([idiag + 1])
                permutations.append([tag.get_external_numbers()])
                tags.append(tag)
            else:
                diagram_classes[ind].append(idiag + 1)
                permutations[ind].append(tag.get_external_numbers())

        permutations = [[diagram_generation.DiagramTag.reorder_permutation(p, perms[0])\
                         for p in perms] for perms in permutations]        

        goal_classes =  [[1, 2, 3],
                         [4],
                         [5, 6, 9, 10, 13, 14],
                         [7, 11, 15],
                         [8, 12, 16],
                         [17, 18, 19],
                         [20, 21, 22],
                         [23, 24, 25]]
        goal_perms =  [[[0, 1, 2, 3, 4], [0, 1, 2, 4, 3], [0, 1, 4, 2, 3]],
                       [[0, 1, 2, 3, 4]],
                       [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3], [0, 1, 3, 2, 4],
                        [0, 1, 4, 2, 3], [0, 1, 3, 4, 2], [0, 1, 4, 3, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 1, 3, 4, 2]],
                       [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3], [0, 1, 4, 2, 3]]]

        for i in range(len(diagram_classes)):
            self.assertEqual(diagram_classes[i], goal_classes[i])
            self.assertEqual(permutations[i], goal_perms[i])

    def test_diagram_tag_uu_uug(self):
        """Test diagram tag for uu>uug"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude = diagram_generation.Amplitude(myproc)

        tags = []
        permutations = []
        diagram_classes = []
        for idiag, diagram in enumerate(myamplitude.get('diagrams')):
            tag = diagram_generation.DiagramTag(diagram)
            try:
                ind = tags.index(tag)
            except:
                diagram_classes.append([idiag + 1])
                permutations.append([tag.get_external_numbers()])
                tags.append(tag)
            else:
                diagram_classes[ind].append(idiag + 1)
                permutations[ind].append(tag.get_external_numbers())

        permutations = [[diagram_generation.DiagramTag.reorder_permutation(p, perms[0])\
                         for p in perms] for perms in permutations]        

        goal_classes = [[1, 8], [2, 9], [3, 10], [4, 11], [5, 12], [6, 13],
                        [7, 14], [15, 18], [16, 19], [17, 20], [21, 24],
                        [22, 25], [23, 26]]

        goal_perms = [[[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]],
                     [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4]]]

        for i in range(len(diagram_classes)):
            self.assertEqual(diagram_classes[i], goal_classes[i])
            self.assertEqual(permutations[i], goal_perms[i])


    def test_reorder_permutation(self):
        """Test the reorder_permutation routine"""

        perm1 = [2,3,4,5,1]
        perm2 = [3,5,2,1,4]
        goal = [3,2,4,1,0]

        self.assertEqual(diagram_generation.DiagramTag.reorder_permutation(\
            perm1, perm2), goal)

    def test_diagram_tag_to_diagram_uux_nglue(self):
        """Test diagrams from DiagramTags for u u~ > n g
        """

        # Test 2, 3, 4 and 5 gluons in the final state
        for ngluon in range (2, 4):

            # Create the amplitude
            myleglist = base_objects.LegList([\
                base_objects.Leg({'id':2, 'state':False}),
                base_objects.Leg({'id':-2, 'state':False})])

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'QCD':ngluon, 'QED': 0},
                                           'model':self.base_model})

            myamplitude = diagram_generation.Amplitude(myproc)
            diagrams = myamplitude.get('diagrams')
            diagram_tags = [diagram_generation.DiagramTag(d) \
                            for d in diagrams]

            #print myamplitude.get('process').nice_string()
            
            for i,(d,dtag) in enumerate(zip(diagrams, diagram_tags)):
                #print '%3r: ' % (i+1),d.nice_string()
                #print 'new: ',dtag.diagram_from_tag(self.base_model).nice_string()
                # Check that the resulting diagram is recreated in the same way
                # from the diagram tag (by checking the diagram tag)
                self.assertEqual(dtag,
                                 diagram_generation.DiagramTag(\
                                     dtag.diagram_from_tag(self.base_model)))

