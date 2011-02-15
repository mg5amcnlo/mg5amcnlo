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
import madgraph.core.color_algebra as color
import madgraph.core.color_ordered_amplitudes as color_ordered_amplitudes
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.export_v4 as export_v4

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

        # A quark S and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'s',
                      'antiname':'s~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'s',
                      'antitexname':'\bar s',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':3,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        s = self.mypartlist[len(self.mypartlist) - 1]
        antis = copy.copy(s)
        antis.set('is_part', False)

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
        gamma = self.mypartlist[len(self.mypartlist) - 1]

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
        e = self.mypartlist[len(self.mypartlist) - 1]
        antie = copy.copy(e)
        antie.set('is_part', False)

        # W
        self.mypartlist.append(base_objects.Particle({'name':'w+',
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

        wplus = self.mypartlist[len(self.mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [color.ColorString([color.f(0, 1, -1),
                                                   color.f(2, 3, -1)]),
                                color.ColorString([color.f(2, 0, -1),
                                                   color.f(1, 3, -1)]),
                                color.ColorString([color.f(1, 2, -1),
                                                   color.f(0, 3, -1)])],
                      'lorentz':['gggg1', 'gggg2', 'gggg3'],
                      'couplings':{(0, 0):'GG', (1, 1):'GG', (2, 2):'GG'},
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
                                             gamma]),
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
                                             gamma]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [s, 
                                             antis,
                                             self.mypartlist[0]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [s,
                                             antis, \
                                             gamma]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        self.myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [e, \
                                             antie, \
                                             gamma]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of u and d to W

        self.myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             self.mypartlist[1], \
                                             wminus]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of d and u to W

        self.myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             self.mypartlist[2], \
                                             wplus]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)

        self.ref_dict_to0 = self.myinterlist.generate_ref_dict()[0]
        self.ref_dict_to1 = self.myinterlist.generate_ref_dict()[1]

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

            #for c in self.myamplitude.get('color_flows'):
            #    print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
            #                                   l in c.get('process').get('legs')]
            #    print c.nice_string()

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
            #for c in self.myamplitude.get('color_flows'):
            #    print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
            #                                   l in c.get('process').get('legs')]
            #    print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluons])
            for iflow, flow in enumerate(self.myamplitude.get('color_flows')):
                self.assertEqual(len(self.myamplitude.get('color_flows')[iflow].get('diagrams')),
                             goal_ndiags[ngluons][iflow])

    def test_color_ordered_uux_uuxng_plus_singlet(self):
        """Test the number of color flows and diagram generated for uu~>uu~+ng with n up to 3
        """
        goal_ndiags = [[2], [5, 5], [15, 15, 15],[51, 51, 51, 51]]
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
                                           'orders':{'QCD':ngluons+2, 'QED': 2}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()
            #for c in self.myamplitude.get('color_flows'):
            #    print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
            #                                   l in c.get('process').get('legs')]
            #    print c.nice_string()

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
            #for c in self.myamplitude.get('color_flows'):
            #    print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
            #                                  l in c.get('process').get('legs')]
            #    print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluons])
            for iflow, flow in enumerate(self.myamplitude.get('color_flows')):
                self.assertEqual(len(self.myamplitude.get('color_flows')[iflow].get('diagrams')),
                             goal_ndiags[ngluons][iflow])

    def test_color_ordered_uux_ddxssxng_no_singlet(self):
        """Test number of color flows and diagram for uu~>dd~ss~+ng with n up to 3
        """
        goal_ndiags = [[4, 4], [16] * 6, [10, 11, 10],[35, 41, 41, 35]]
        goal_nflows = [2, 6, 3, 4]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':3,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-3,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders':{'QCD':ngluons+4, 'QED': 0}})

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

    def test_color_ordered_uux_uuxddxng_no_singlet(self):
        """Test number of color flows and diagram for uu~>uu~dd~+ng with n up to 3
        """
        goal_ndiags = [[4, 4], [17, 3], [10, 11, 10],[35, 41, 41, 35]]
        goal_nflows = [2, 6, 3, 4]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':1,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-1,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders':{'QCD':ngluons+4, 'QED': 0}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()
            #for c in self.myamplitude.get('color_flows'):
            #    print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
            #                                   l in c.get('process').get('legs')]
            #    print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluons])
            for iflow, flow in enumerate(self.myamplitude.get('color_flows')):
                self.assertEqual(len(self.myamplitude.get('color_flows')[iflow].get('diagrams')),
                             goal_ndiags[ngluons][iflow])

    def test_color_ordered_uux_epem_nglue(self):
        """Test the number of color flows and diagrams generated for uu~>e+e-ng
        """

        goal_nflows = [1, 1, 1, 1, 1]
        goal_ndiagrams = [1, 2, 5, 16, 58]

        # Test 2, 3, 4 and 5 gluons in the final state
        for ngluon in range (0, 5):

            # Create the amplitude
            myleglist = base_objects.LegList([\
                base_objects.Leg({'id':2, 'state':False}),
                base_objects.Leg({'id':-2, 'state':False}),
                base_objects.Leg({'id':-11}),
                base_objects.Leg({'id':11})])

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'QCD':ngluon, 'QED': 2},
                                           'model':self.mymodel})

            self.myamplitude = color_ordered_amplitudes.ColorOrderedAmplitude(myproc)

            #for c in self.myamplitude.get('color_flows'):
            #    print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
            #                                   l in c.get('process').get('legs')]
            #    print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluon])
            self.assertEqual(len(self.myamplitude.get('color_flows')[0].\
                                 get('diagrams')), goal_ndiagrams[ngluon])
                             

            
    def test_color_ordered_uux_uuxepemng_no_singlet(self):
        """Test color flows and diagrams for uu~>uu~e+e-+ng with n up to 2
        """
        goal_ndiags = [[4], [14, 14], [50, 56, 50]]
        goal_nflows = [1, 2, 3]

        for ngluons in range(0, 3):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':11,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':-11,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'orders':{'QCD':ngluons+2, 'QED': 2}})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()
            #for c in self.myamplitude.get('color_flows'):
            #    print "color flow process: ",[(l.get('number'), l.get('color_ordering')) for \
            #                                   l in c.get('process').get('legs')]
            #    print c.nice_string()

            self.assertEqual(len(self.myamplitude.get('color_flows')),
                             goal_nflows[ngluons])
            for iflow, flow in enumerate(self.myamplitude.get('color_flows')):
                self.assertEqual(len(self.myamplitude.get('color_flows')[iflow].get('diagrams')),
                             goal_ndiags[ngluons][iflow])

    def test_color_ordered_gg_h_nglue(self):
        """Test the number of color ordered diagrams gg>h+ng with n up to 3"""

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()
        mymodel = base_objects.Model()
        myprocess = base_objects.Process()
        myamplitude = color_ordered_amplitudes.ColorOrderedAmplitude()

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

        # A Higgs
        mypartlist.append(base_objects.Particle({'name':'h',
                      'antiname':'h',
                      'spin':1,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'H',
                      'antitexname':'H',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':25,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # 3 gluon vertex
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

        # Gluon couplings to Higgs
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0], \
                                             mypartlist[0], \
                                             mypartlist[1]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'HIG':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0], \
                                             mypartlist[0], \
                                             mypartlist[0], \
                                             mypartlist[1]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'HIG':1, 'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0], \
                                             mypartlist[0], \
                                             mypartlist[0], \
                                             mypartlist[0], \
                                             mypartlist[1]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'HIG':1, 'QCD':2}}))

        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        goal_ndiags = [1, 4, 19, 90]

        # Test 0-4 gluons in the final state
        for ngluon in range (0, 4):

            # Create the amplitude
            myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'state':False})] * 2)

            myleglist.append(base_objects.Leg({'id':25,
                                                'state':True}))

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'HIG':1},
                                           'model':mymodel})

            myamplitude.set('process', myproc)

            myamplitude.setup_process()

            # Test the process after setup
            mycoleglist = base_objects.LegList([\
                color_ordered_amplitudes.ColorOrderedLeg(leg) for leg in myleglist])

            for i, leg in enumerate(mycoleglist):
                leg.set('color_ordering', {0: (i+1, i+1)})

            mycolorflow = myamplitude.get('color_flows')[0]

            # Call generate_diagram and output number of diagrams
            ndiags = len(mycolorflow.get('diagrams'))

            #print "Number of diagrams for %d gluons: %d, cmp %d" % (ngluon,
            #                                                        ndiags,
            #                                                        goal_ndiags[ngluon-2])
            #print myamplitude.get('diagrams').nice_string()

            self.assertEqual(ndiags, goal_ndiags[ngluon])

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


#===============================================================================
# BGHelasMatrixElementTest
#===============================================================================
class BGHelasMatrixElementTest(unittest.TestCase):
    """Test class for functions related to the B-G matrix elements"""

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

        # A quark S and its antiparticle
        self.mypartlist.append(base_objects.Particle({'name':'s',
                      'antiname':'s~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'s',
                      'antitexname':'\bar s',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':3,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        s = self.mypartlist[len(self.mypartlist) - 1]
        antis = copy.copy(s)
        antis.set('is_part', False)

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
        gamma = self.mypartlist[len(self.mypartlist) - 1]

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
        e = self.mypartlist[len(self.mypartlist) - 1]
        antie = copy.copy(e)
        antie.set('is_part', False)

        # W
        self.mypartlist.append(base_objects.Particle({'name':'w+',
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

        wplus = self.mypartlist[len(self.mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # 3 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [color.ColorString([color.f(1, 2, -1),
                                                   color.f(-1, 0, 3)]),
                                color.ColorString([color.f(1, 3, -1),
                                                   color.f(-1, 0, 2)]),
                                color.ColorString([color.f(2, 3, -1),
                                                   color.f(-1, 0, 1)])],
                      'lorentz':['VVVV4', 'VVVV3', 'VVVV1'],
            'couplings':{(0, 0):'GG', (1, 1):'GG', (2, 2):'GG'},
                      'orders':{'QCD':2}}))

        # Gluon and photon couplings to quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             gamma]),
                      'color': [color.ColorString([color.T(0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             gamma]),
                      'color': [color.ColorString([color.T(0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [s, 
                                             antis,
                                             self.mypartlist[0]]),
                      'color': [color.ColorString([color.T(2,0,1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma

        self.myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [e, \
                                             antie, \
                                             gamma]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of u and d to W

        self.myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             self.mypartlist[1], \
                                             wminus]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of d and u to W

        self.myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             self.mypartlist[2], \
                                             wplus]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)

        self.ref_dict_to0 = self.myinterlist.generate_ref_dict()[0]
        self.ref_dict_to1 = self.myinterlist.generate_ref_dict()[1]

    def test_matrix_element_gluons(self):
        """Test the matrix element for all-gluon amplitudes"""

        goal_amplitudes = [4, 12, 38, 78, 138, 245]
        goal_wavefunctions = [6, 19, 36, 89, 211, 394]

        # Test 2, 3, 4 and 5 gluons in the final state
        for ngluon in range(2,4):

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

            mycolorflow = self.myamplitude.get('color_flows')[0]

            matrix_element = color_ordered_amplitudes.BGHelasMatrixElement(\
                mycolorflow, gen_color=False, optimization=3)

            print "\n".join(\
                helas_call_writers.FortranUFOHelasCallWriter(self.mymodel).\
                get_matrix_element_calls(matrix_element))
            print "For ",ngluon," FS gluons, there are ",\
                  len(matrix_element.get_all_amplitudes()),' amplitudes and ',\
                  len(matrix_element.get_all_wavefunctions()),\
                  ' wavefunctions for ', len(mycolorflow.get('diagrams')),\
                  ' diagrams'

            self.assertEqual(len(matrix_element.get_all_wavefunctions()),
                             goal_wavefunctions[ngluon-2])
            self.assertEqual(len(matrix_element.get_all_amplitudes()),
                             goal_amplitudes[ngluon-2])
            
            # Test JAMP (color amplitude) output
            print '\n'.join(export_v4.get_JAMP_lines(matrix_element))

    def test_matrix_element_gluons_non_optimized(self):
        """Test the matrix element for all-gluon amplitudes"""

        goal_amplitudes = [4, 15, 68]
        goal_wavefunctions = [6, 16, 30]

        # Test 2, 3, 4 and 5 gluons in the final state
        for ngluon in range(2,4):

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

            mycolorflow = self.myamplitude.get('color_flows')[0]

            matrix_element = color_ordered_amplitudes.BGHelasMatrixElement(\
                mycolorflow, gen_color=False, optimization=1)

            print "\n".join(\
                helas_call_writers.FortranUFOHelasCallWriter(self.mymodel).\
                get_matrix_element_calls(matrix_element))
            print "For ",ngluon," FS gluons, there are ",\
                  len(matrix_element.get_all_amplitudes()),' amplitudes and ',\
                  len(matrix_element.get_all_wavefunctions()),\
                  ' wavefunctions for ', len(mycolorflow.get('diagrams')),\
                  ' diagrams'

            self.assertEqual(len(matrix_element.get_all_wavefunctions()),
                             goal_wavefunctions[ngluon-2])
            self.assertEqual(len(matrix_element.get_all_amplitudes()),
                             goal_amplitudes[ngluon-2])
            
            # Test JAMP (color amplitude) output
            print '\n'.join(export_v4.get_JAMP_lines(matrix_element))

    def test_matrix_element_photons(self):
        """Test the matrix element for all-gluon amplitudes"""

        goal_amplitudes = [2, 6, 12, 30, 60, 140]
        goal_wavefunctions = [6, 11, 32, 77, 190, 429]

        # Test 2, 3, 4 and 5 photons in the final state
        for nphoton in range(2,6):

            # Create the amplitude
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))

            myleglist.extend([base_objects.Leg({'id':22,
                                              'state':True})] * nphoton)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()

            mycolorflow = self.myamplitude.get('color_flows')[0]

            matrix_element = color_ordered_amplitudes.BGHelasMatrixElement(\
                mycolorflow, gen_color=False, optimization = 3)

            print "\n".join(\
                helas_call_writers.FortranUFOHelasCallWriter(self.mymodel).\
                get_matrix_element_calls(matrix_element))
            print "For ",nphoton," FS photons, there are ",\
                  len(matrix_element.get_all_amplitudes()),' amplitudes and ',\
                  len(matrix_element.get_all_wavefunctions()),\
                  ' wavefunctions for ', len(mycolorflow.get('diagrams')),\
                  ' diagrams'

            self.assertEqual(len(matrix_element.get_all_wavefunctions()),
                             goal_wavefunctions[nphoton-2])
            self.assertEqual(len(matrix_element.get_all_amplitudes()),
                             goal_amplitudes[nphoton-2])

            # Test JAMP (color amplitude) output
            #print '\n'.join(export_v4.get_JAMP_lines(matrix_element))

    def test_matrix_element_photons_non_optimized(self):
        """Test the matrix element for all-gluon amplitudes"""

        goal_amplitudes = [2, 6, 24, 120]
        goal_wavefunctions = [6, 11, 26, 57]

        # Test 2, 3, 4 and 5 photons in the final state
        for nphoton in range(2,6):

            # Create the amplitude
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))

            myleglist.extend([base_objects.Leg({'id':22,
                                              'state':True})] * nphoton)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel})

            self.myamplitude.set('process', myproc)

            self.myamplitude.setup_process()

            mycolorflow = self.myamplitude.get('color_flows')[0]

            matrix_element = color_ordered_amplitudes.BGHelasMatrixElement(\
                mycolorflow, gen_color=False, optimization = 1)

            print "\n".join(\
                helas_call_writers.FortranUFOHelasCallWriter(self.mymodel).\
                get_matrix_element_calls(matrix_element))
            print "For ",nphoton," FS photons, there are ",\
                  len(matrix_element.get_all_amplitudes()),' amplitudes and ',\
                  len(matrix_element.get_all_wavefunctions()),\
                  ' wavefunctions for ', len(mycolorflow.get('diagrams')),\
                  ' diagrams'

            self.assertEqual(len(matrix_element.get_all_wavefunctions()),
                             goal_wavefunctions[nphoton-2])
            self.assertEqual(len(matrix_element.get_all_amplitudes()),
                             goal_amplitudes[nphoton-2])

            # Test JAMP (color amplitude) output
            print '\n'.join(export_v4.get_JAMP_lines(matrix_element))


    def test_matrix_element_uux_nglue(self):
        """Test the number of color flows and diagrams generated for uu~>gg
        """

        goal_amplitudes = [2, 7, 18, 45]
        goal_wavefunctions = [6, 10, 27, 56]

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

            mycolorflow = self.myamplitude.get('color_flows')[0]

            matrix_element = color_ordered_amplitudes.BGHelasMatrixElement(\
                mycolorflow, gen_color=False, optimization = 3)

            print "\n".join(\
                helas_call_writers.FortranUFOHelasCallWriter(self.mymodel).\
                get_matrix_element_calls(matrix_element))
            print "For ",ngluon," FS gluons, there are ",\
                  len(matrix_element.get_all_amplitudes()),' amplitudes and ',\
                  len(matrix_element.get_all_wavefunctions()),\
                  ' wavefunctions for ', len(mycolorflow.get('diagrams')),\
                  ' diagrams'

            self.assertEqual(len(matrix_element.get_all_amplitudes()),
                             goal_amplitudes[ngluon-2])
            self.assertEqual(len(matrix_element.get_all_wavefunctions()),
                             goal_wavefunctions[ngluon-2])

            # Test JAMP (color amplitude) output
            #print '\n'.join(export_v4.get_JAMP_lines(matrix_element))

    def test_matrix_element_uux_ddxng_no_singlet(self):
        """Test the number of color flows and diagrams generated for uu~>gg
        """

        goal_amplitudes = [[1],[3,3],[8,8,9]]
        goal_wavefunctions = [[5],[9,9],[18,18,16]]
        goal_ndiags = [[1], [3, 3], [10, 11, 10],[35, 41, 41, 35]]
        goal_nflows = [1, 2, 3, 4]

        # Test 2, 3, 4 and 5 gluons in the final state
        for ngluon in range (0, 5):

            # Create the amplitude
            myleglist = base_objects.LegList([\
                base_objects.Leg({'id':2, 'state':False}),
                base_objects.Leg({'id':-2, 'state':False}),
                base_objects.Leg({'id':1}),
                base_objects.Leg({'id':-1})])

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'QED': 0},
                                           'model':self.mymodel})

            self.myamplitude = color_ordered_amplitudes.ColorOrderedAmplitude(myproc)

            for iflow, mycolorflow in \
                enumerate(self.myamplitude.get('color_flows')):

                matrix_element = color_ordered_amplitudes.BGHelasMatrixElement(\
                    mycolorflow, gen_color=False, optimization = 3)

                print "\n".join(\
                    helas_call_writers.FortranUFOHelasCallWriter(self.mymodel).\
                    get_matrix_element_calls(matrix_element))
                print "For ",ngluon," FS gluons, there are ",\
                      len(matrix_element.get_all_amplitudes()),' amplitudes and ',\
                      len(matrix_element.get_all_wavefunctions()),\
                      ' wavefunctions for ', len(mycolorflow.get('diagrams')),\
                      ' diagrams'

                self.assertEqual(len(matrix_element.get_all_amplitudes()),
                                 goal_amplitudes[ngluon][iflow])
                self.assertEqual(len(matrix_element.get_all_wavefunctions()),
                                 goal_wavefunctions[ngluon][iflow])

                # Test JAMP (color amplitude) output
                print '\n'.join(export_v4.get_JAMP_lines(matrix_element))

    def test_matrix_element_uux_ddxng_no_singlet_non_optimized(self):
        """Test the number of color flows and diagrams generated for uu~>gg
        """

        goal_amplitudes = [[1],[3,3],[11,12,11],[44, 50, 50, 44]]
        goal_wavefunctions = [[5],[9,9],[16,16,15],[33, 33, 34, 30]]
        goal_ndiags = [[1], [3, 3], [10, 11, 10],[35, 41, 41, 35]]
        goal_nflows = [1, 2, 3, 4]

        # Test 0, 1, 2 and 3 gluons in the final state
        for ngluon in range (0, 4):

            # Create the amplitude
            myleglist = base_objects.LegList([\
                base_objects.Leg({'id':2, 'state':False}),
                base_objects.Leg({'id':-2, 'state':False}),
                base_objects.Leg({'id':1}),
                base_objects.Leg({'id':-1})])

            myleglist.extend([base_objects.Leg({'id':21,
                                              'state':True})] * ngluon)

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'QED': 0},
                                           'model':self.mymodel})

            self.myamplitude = color_ordered_amplitudes.ColorOrderedAmplitude(myproc)

            for iflow, mycolorflow in \
                enumerate(self.myamplitude.get('color_flows')):

                matrix_element = color_ordered_amplitudes.BGHelasMatrixElement(\
                    mycolorflow, gen_color=False, optimization = 1)

                print "\n".join(\
                    helas_call_writers.FortranUFOHelasCallWriter(self.mymodel).\
                    get_matrix_element_calls(matrix_element))
                print "For ",ngluon," FS gluons, there are ",\
                      len(matrix_element.get_all_amplitudes()),' amplitudes and ',\
                      len(matrix_element.get_all_wavefunctions()),\
                      ' wavefunctions for ', len(mycolorflow.get('diagrams')),\
                      ' diagrams'

                self.assertEqual(len(matrix_element.get_all_amplitudes()),
                                 goal_amplitudes[ngluon][iflow])
                self.assertEqual(len(matrix_element.get_all_wavefunctions()),
                                 goal_wavefunctions[ngluon][iflow])

                # Test JAMP (color amplitude) output
                print '\n'.join(export_v4.get_JAMP_lines(matrix_element))
