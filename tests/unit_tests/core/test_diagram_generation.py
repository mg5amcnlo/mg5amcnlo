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

"""Unit test library for the various base objects of the core library"""

import copy
import itertools
import logging
import math
import unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

#===============================================================================
# AmplitudeTest
#===============================================================================
class AmplitudeTest(unittest.TestCase):
    """Test class for routine functions of the Amplitude object"""

    mydict = {}
    myamplitude = None

    myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':'final',
                                      'from_group':False})] * 10)
    myvertexlist = base_objects.VertexList([base_objects.Vertex({'id':3,
                                      'legs':myleglist})] * 10)


    mydiaglist = base_objects.DiagramList([base_objects.Diagram(\
                                        {'vertices':myvertexlist})] * 100)

    myprocess = base_objects.Process()

    def setUp(self):

        self.mydict = {'diagrams':self.mydiaglist, 'process':self.myprocess}

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
        self.assertRaises(diagram_generation.Amplitude.PhysicsObjectError,
                          diagram_generation.Amplitude,
                          a_number)

        # Test get
        self.assertRaises(diagram_generation.Amplitude.PhysicsObjectError,
                          self.myamplitude.get,
                          a_number)
        self.assertRaises(diagram_generation.Amplitude.PhysicsObjectError,
                          self.myamplitude.get,
                          'wrongparam')

        # Test set
        self.assertRaises(diagram_generation.Amplitude.PhysicsObjectError,
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
        goal = goal + "    \'diagrams\': %s\n}" % repr(self.mydiaglist)

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

        # 3 gluon vertiex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': ['C1'],
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
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': ['C1'],
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
                      'color': ['C1'],
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
                                              'state':'final'}) \
                                              for num in range(1, 5)])
        myleglist[0].set('state', 'initial')
        myleglist[1].set('state', 'initial')

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
                                'state':'final'})
        l13 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':'initial'})
        l14 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':'initial'})
        l23 = base_objects.Leg({'id':21,
                                'number':2,
                                'state':'initial'})
        l24 = base_objects.Leg({'id':21,
                                'number':2,
                                'state':'initial'})
        l34 = base_objects.Leg({'id':21,
                                'number':3,
                                'state':'final'})
        l123 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':'final'})
        l124 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':'final'})
        l134 = base_objects.Leg({'id':21,
                                'number':1,
                                'state':'initial'})
        l234 = base_objects.Leg({'id':21,
                                'number':2,
                                'state':'initial'})

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
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'number':3,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'number':4,
                                         'state':'final'}))
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
                                'state':'final'})
        l12phot = base_objects.Leg({'id':22,
                                'number':1,
                                'state':'final'})
        l34glue = base_objects.Leg({'id':21,
                                'number':3,
                                'state':'final'})
        l34phot = base_objects.Leg({'id':22,
                                'number':3,
                                'state':'final'})
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
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':3,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'number':4,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'number':5,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'number':6,
                                         'state':'final'}))
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
                                              'state':'initial'})] * 2)

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':'final'})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                                'orders':{'QCD':ngluon},
                                                'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            # Call generate_diagram and output number of diagrams
            ndiags = len(self.myamplitude.generate_diagrams())

            logging.debug("Number of diagrams for %d gluons: %d" % (ngluon,
                                                            ndiags))

            self.assertEqual(ndiags, goal_ndiags[ngluon - 1])

    def test_diagram_generation_uux_gg(self):
        """Test the number of diagram generated for uu~>gg (s, t and u channels)
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        self.myamplitude.set('process', myproc)

        self.assertEqual(len(self.myamplitude.generate_diagrams()), 3)

    def test_diagram_generation_uux_uuxng(self):
        """Test the number of diagram generated for uu~>uu~+ng with n up to 2
        """
        goal_ndiags = [4, 18, 120, 1074, 12120]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':'final'}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':'final'}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':'final'})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.assertEqual(len(self.myamplitude.generate_diagrams()),
                             goal_ndiags[ngluons])

    def test_diagram_generation_uux_ddxng(self):
        """Test the number of diagram generated for uu~>dd~+ng with n up to 2
        """
        goal_ndiags = [2, 9, 60, 537, 6060]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':'final'}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':'final'}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':'final'})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.assertEqual(len(self.myamplitude.generate_diagrams()),
                             goal_ndiags[ngluons])

    def test_diagram_generation_diagrams_ddx_uuxg(self):
        """Test the vertex list output for dd~>uu~g (so far only 2
        diagrams, due to lack of time)
        """
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                           'state':'initial',
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':-1,
                                           'state':'initial',
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':'final',
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':'final',
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':'final',
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        self.myamplitude.set('process', myproc)

        mydiagrams = self.myamplitude.generate_diagrams()

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
                                    'state':'final'})
        l34glue = base_objects.Leg({'id':21,
                                    'number':3,
                                    'state':'final'})
        l35 = base_objects.Leg({'id':-2,
                                'number':3,
                                'state':'final'})

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
            [vx12glue, vx34glue, vx12glue34glue5])}),
            base_objects.Diagram({'vertices': base_objects.VertexList(\
            [vx12glue, vx35, vx12glue354])})\
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
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':'final'}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':'final'}))
            myleglist.extend([base_objects.Leg({'id':1,
                                                'state':'final'})] * nquarks)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.assertEqual(len(self.myamplitude.generate_diagrams()), 0)

    def test_diagram_generation_photons(self):
        """Test the number of diagram generated for uu~>na with n up to 6"""

        # Test up to 5 photons in the final state
        for nphot in range (1, 5):

            # Create the amplitude
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':'initial'}))

            myleglist.extend([base_objects.Leg({'id':22,
                                                'state':'final'})] * nphot)

            myproc = base_objects.Process({'legs':myleglist,
                                            'orders':{'QED':nphot},
                                            'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            # Call generate_diagram and output number of diagrams
            ndiags = len(self.myamplitude.generate_diagrams())

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
                                             'state':'initial'}))
            myleglist.append(base_objects.Leg({'id':11,
                                             'state':'initial'}))

            myleglist.extend([base_objects.Leg({'id':11,
                                                'state':'final'}),
                              base_objects.Leg({'id':-11,
                                                'state':'final'})] * npairs)

            myproc = base_objects.Process({'legs':myleglist,
                                            'orders':{'QED':npairs * 2},
                                            'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            # Call generate_diagram and output number of diagrams
            ndiags = len(self.myamplitude.generate_diagrams())

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
                      'line':'waivy',
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
                      'color': ['C1'],
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
                      'color': ['C1'],
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
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':12,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', myproc)

        self.assertEqual(len(myamplitude.get('diagrams')), 1)
        
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
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': ['C1'],
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
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[3]]),
                      'color': ['C1'],
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
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)

    def test_multiparticle_pp_nj(self):
        """Setting up and testing pp > nj based on multiparticle lists,
        using the amplitude functionality of MultiProcess
        (which makes partial use of crossing symmetries)
        """

        max_fs = 2 # 3

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': 'final'});

        goal_number_processes = [219,379]

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

            my_multi_leglist[0].set('state','initial')
            my_multi_leglist[1].set('state','initial')

            my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                    'model':self.mymodel})
            my_multiprocess = base_objects.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            nproc = 0

            # Calculate diagrams for all processes
            
            amplitudes = diagram_generation.AmplitudeList(my_multiprocess)

            if nfs <= 2:
                self.assertEqual(len(amplitudes),goal_number_processes[nfs - 2])
            
            amplitudes.generate_amplitudes()

            valid_procs = [([leg.get('id') for leg in \
                             amplitude.get('process').get('legs')],
                            len(amplitude.get('diagrams'))) \
                           for amplitude in amplitudes]

            valid_procs = filter(lambda item: item[1] > 0, valid_procs)

            if nfs <= 2:
                self.assertEqual(valid_procs, goal_valid_procs[nfs - 2])

            #print 'pp > ',nfs,'j (p,j = ', p, '):'
            #print 'Valid processes: ',len(filter(lambda item: item[1] > 0, valid_procs))
            #print 'Attempted processes: ',len(amplitudes)

    def test_multiparticle_pp_nj_with_full_crossing(self):
        """Setting up and testing pp > nj based on multiparticle lists.
        Make maximum use of crossing symmetries to minimize number of
        processes for which we generate amplitudes.
        """

        #print "Testing multiparticles with crossing checks"

        max_fs = 2 # 3

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': 'final'});

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

            my_multi_leglist[0].set('state','initial')
            my_multi_leglist[1].set('state','initial')

            my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                    'model':self.mymodel})
            my_multiprocess = base_objects.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            nproc = 0

            # Setup amplitudes

            amplitudes = diagram_generation.AmplitudeList(my_multiprocess)

            nproc = 0

            # Calculate diagrams for all processes,
            # making maximum use of crossing symmetry
            valid_procs = []
            # Check for crossed processes
            valid_procs_dict = {}
            failed_procs = []
            number_valid_procs_crossed = 0
            number_failed_procs_crossed = 0
            for amplitude in amplitudes:
                model = amplitude.get('process').get('model')
                legs = amplitude.get('process').get('legs')
                if tuple(sorted(legs.get_outgoing_id_list(model))) \
                       in valid_procs_dict:

                    amplitude_number = valid_procs_dict[tuple(sorted(legs.get_outgoing_id_list(model)))]
                    number_valid_procs_crossed = number_valid_procs_crossed + 1
                    valid_procs.append(([leg.get('id') for leg in \
                                         legs],
                                        len(amplitudes[amplitude_number].get('diagrams'))))
                elif tuple(sorted(legs.get_outgoing_id_list(model))) \
                            in failed_procs:
                    number_failed_procs_crossed = number_failed_procs_crossed + 1
                else:
                    if len(amplitude.get('diagrams')) > 0:
                        nproc = nproc + 1
                        valid_procs.append(([leg.get('id') for leg in \
                                             legs],
                                            len(amplitude.get('diagrams'))))
                        valid_procs_dict[tuple(sorted(legs.get_outgoing_id_list(model)))] = amplitudes.index(amplitude)
                    else:
                        failed_procs.append(tuple(sorted(legs.get_outgoing_id_list(model))))
            if nfs < 3:
                self.assertEqual(valid_procs, goal_valid_procs[nfs - 2])

            #print 'pp > ',nfs,'j, (p,j=', p,'):'
            #print 'Valid processes generated: ',nproc
            #print 'Valid crossings found: ',number_valid_procs_crossed
            #print 'Attempted processes: ',len(amplitudes)-number_failed_procs_crossed
            #print 'Failed crossings found: ',number_failed_procs_crossed

