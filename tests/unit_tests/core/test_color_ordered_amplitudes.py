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
import os
import unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_ordered_amplitudes as color_ordered_amplitudes
import madgraph.iolibs.drawing_eps as draw

#===============================================================================
# ColorOrderedAmplitudeTest
#===============================================================================
class ColorOrderedAmplitudeTest(unittest.TestCase):
    """Test class for all functions related to the diagram generation"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()
    myprocess = base_objects.Process()

    ref_dict_to0 = {}
    ref_dict_to1 = {}

    mycolorflow = color_ordered_amplitudes.ColorOrderedFlow()
    myamplitude = color_ordered_amplitudes.ColorOrderedAmplitude()

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

    def test_combine_legs_color_ordered_gluons(self):
        """Test combine_legs and merge_comb_legs: gg>gg"""

        # Four gluon legs with two initial state
        myleglist = base_objects.LegList([\
            color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                          'number':num,
                                          'state':True,
                                          'color_ordering': {0:(num,num)}}) \
            for num in range(1, 5)])
        myleglist[0].set('state', False)
        myleglist[1].set('state', False)
        myleglist[3].set('color_ordering', {})

        l1 = myleglist[0]
        l2 = myleglist[1]
        l3 = myleglist[2]
        l4 = myleglist[3]

        myprocess = base_objects.Process()
        myprocess.set('model', self.mymodel)
        self.mycolorflow.set('process', myprocess)
        self.mycolorflow.set('max_color_orders', {0:3})

        # All possibilities for the first combination
        goal_combined_legs = [
                [(l1, l2), l3, l4], [(l1, l2), (l3, l4)],
                [(l1, l3), l2, l4], [(l1, l3), (l2, l4)],
                [(l1, l4), l2, l3], [(l1, l4), (l2, l3)],
                [l1, (l2, l3), l4], [l1, (l2, l4), l3], [l1, l2, (l3, l4)],
                [(l1, l2, l3), l4], [(l1, l2, l4), l3],
                [(l1, l3, l4), l2], [l1, (l2, l3, l4)]
                ]

        combined_legs = self.mycolorflow.combine_legs(
                                              [leg for leg in myleglist],
                                                self.ref_dict_to1,
                                                3)
        self.assertEqual(combined_legs, goal_combined_legs)

        # Now test the reduction of legs for this
        reduced_list = self.mycolorflow.merge_comb_legs(combined_legs,
                                                        self.ref_dict_to1)

        # Remaining legs should be from_group False
        l1.set('from_group', False)
        l2.set('from_group', False)
        l3.set('from_group', False)
        l4.set('from_group', False)

        # Define all possible legs obtained after merging combinations
        l12 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':1,
                                'state':True,
                                'color_ordering': {0:(1,2)}})
        l13 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':1,
                                'state':False,
                                'color_ordering': {0:(3,1)}})
        l14 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':1,
                                'state':False,
                                'color_ordering': {0:(1,1)}})
        l23 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':2,
                                'state':False,
                                'color_ordering': {0:(2,3)}})
        l24 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':2,
                                'state':False,
                                'color_ordering': {0:(2,2)}})
        l34 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':3,
                                'state':True,
                                'color_ordering': {0:(3,3)}})
        l123 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':1,
                                'state':True,
                                'color_ordering': {0:(1,3)}})
        l124 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':1,
                                'state':True,
                                'color_ordering': {0:(1,2)}})
        l134 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':1,
                                'state':False,
                                'color_ordering': {0:(3,1)}})
        l234 = color_ordered_amplitudes.ColorOrderedLeg({'id':21,
                                'number':2,
                                'state':False,
                                'color_ordering': {0:(2,3)}})

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

    def test_color_ordered_gluons(self):
        """Test the number of color ordered diagrams gg>ng with n up to 4"""

        goal_ndiags = [3, 10, 38, 154, 654, 2871, 12925]

        # Time for 7 gluons: 46 s

        # Test 2, 3, 4 and 5 gluons in the final state
        for ngluon in range (2, 6):

            # Create the amplitude
            myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'state':False})] * 2)

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'QCD':ngluon},
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()

            # Test the process after setup
            mycoleglist = base_objects.LegList([\
                color_ordered_amplitudes.ColorOrderedLeg(leg) for leg in myleglist])

            for i, leg in enumerate(mycoleglist):
                leg.set('color_ordering', {0: (i+1, i+1)})

            mycoproc = base_objects.Process({'legs':mycoleglist,
                                           'orders':{'QCD':ngluon},
                                           'model':self.mymodel})

            mycolorflow = self.myamplitude.get('color_flows')[0]

            self.assertEqual(mycolorflow.get('process'),
                             mycoproc)

            # Call generate_diagram and output number of diagrams
            ndiags = len(mycolorflow.get('diagrams'))

            #print "Number of diagrams for %d gluons: %d, cmp %d" % (ngluon,
            #                                                        ndiags,
            #                                                        goal_ndiags[ngluon-2])
            #print self.myamplitude.get('diagrams').nice_string()

            self.assertEqual(ndiags, goal_ndiags[ngluon - 2])

    def test_color_ordered_uux_nglue(self):
        """Test the number of color flows and diagrams generated for uu~>gg
        """

        goal_nflows = [1, 1, 1, 1]
        goal_ndiagrams = [2, 6, 21, 81]

        # Test 2, 3, 4 and 5 gluons in the final state
        for ngluon in range (2, 6):

            # Create the amplitude
            myleglist = base_objects.LegList([\
                base_objects.Leg({'id':2, 'state':False}),
                base_objects.Leg({'id':-2, 'state':False})])

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'QCD':ngluon, 'QED': 0},
                                           'model':self.mymodel})

            self.myamplitude = color_ordered_amplitudes.ColorOrderedAmplitude(myproc)

            for c in self.myamplitude.get('color_flows'):
                print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
                                               l in c.get('process').get('legs')]
                print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluon-2])
            self.assertEqual(len(self.myamplitude.get('color_flows')[0].\
                                 get('diagrams')), goal_ndiagrams[ngluon-2])
                             

            
    def test_color_ordered_uux_uuxng_no_singlet(self):
        """Test the number of color flows and diagram generated for uu~>uu~+ng with n up to 3
        """
        goal_ndiags = [[1], [3, 3], [10, 11, 10],[35, 41, 41, 35]]
        goal_nflows = [1, 2, 3, 4]

        for ngluons in range(0, 4):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders':{'QCD':ngluons+2, 'QED': 0}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()
            for c in self.myamplitude.get('color_flows'):
                print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
                                               l in c.get('process').get('legs')]
                print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluons])
            for iflow, flow in enumerate(self.myamplitude.get('color_flows')):
                self.assertEqual(len(self.myamplitude.get('color_flows')[iflow].get('diagrams')),
                             goal_ndiags[ngluons][iflow])

    def test_color_ordered_uux_uuxng_singlet(self):
        """Test the number of color flows and diagram generated for uu~>uu~+ng with n up to 3
        """
        goal_ndiags = [[1], [2, 2], [5, 4, 5],[16, 10, 10, 16]]
        goal_nflows = [1, 2, 3, 4]

        for ngluons in range(0, 4):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders':{'QCD':ngluons, 'QED': 2}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()
            for c in self.myamplitude.get('color_flows'):
                print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
                                               l in c.get('process').get('legs')]
                print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluons])
            for iflow, flow in enumerate(self.myamplitude.get('color_flows')):
                self.assertEqual(len(self.myamplitude.get('color_flows')[iflow].get('diagrams')),
                             goal_ndiags[ngluons][iflow])

    def test_color_ordered_uux_ddxng_no_singlet(self):
        """Test the number of color flows and diagram generated for uu~>uu~+ng with n up to 3
        """
        goal_ndiags = [[1], [3, 3], [10, 11, 10],[35, 41, 41, 35]]
        goal_nflows = [1, 2, 3, 4]

        for ngluons in range(0, 4):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders':{'QCD':ngluons+2, 'QED': 0}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()
            for c in self.myamplitude.get('color_flows'):
                print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
                                               l in c.get('process').get('legs')]
                print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluons])
            for iflow, flow in enumerate(self.myamplitude.get('color_flows')):
                self.assertEqual(len(self.myamplitude.get('color_flows')[iflow].get('diagrams')),
                             goal_ndiags[ngluons][iflow])

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

