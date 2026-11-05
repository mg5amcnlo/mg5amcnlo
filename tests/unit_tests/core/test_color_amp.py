################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Unit test library for the routines of the core library related to writing
color information for diagrams."""

from __future__ import absolute_import
import collections
import copy
import fractions

import madgraph
import madgraph.various.misc as misc
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color
import madgraph.loop.loop_color_amp as loop_color_amp

import tests.unit_tests as unittest
class ColorAmpTest(unittest.TestCase):
    """Test class for the color_amp module"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()

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
                      'texname':'u',
                      'antitexname':'\bar u',
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
                      'texname':r'\gamma',
                      'antitexname':r'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A Higgs
        self.mypartlist.append(base_objects.Particle({'name':'h',
                      'antiname':'h',
                      'spin':1,
                      'color':1,
                      'mass':'mh',
                      'width':'wh',
                      'texname':'h',
                      'antitexname':'h',
                      'line':'dashed',
                      'charge':0.,
                      'pdg_code':25,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # 3 gluon vertiex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [color.ColorString([color.f(-1, 0, 2),
                                                   color.f(-1, 1, 3)]),
                                color.ColorString([color.f(-1, 0, 3),
                                                   color.f(-1, 1, 2)]),
                                color.ColorString([color.f(-1, 0, 1),
                                                   color.f(-1, 2, 3)])],
                      'lorentz':['L(p1,p2,p3)', 'L(p2,p3,p1)', 'L3'],
                      'couplings':{(0, 0):'G^2',
                                   (1, 1):'G^2',
                                   (2, 2):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon couplings to up and down quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Photon coupling to up
        self.myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[3]]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)

    def test_colorize_uu_gg(self):
        """Test the colorize function for uu~ > gg"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))

        myleglist.extend([base_objects.Leg({'id':21,
                                            'state':True})] * 2)

        myprocess = base_objects.Process({'legs':myleglist,
                                        'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude()

        myamplitude.set('process', myprocess)

        myamplitude.generate_diagrams()

        my_col_basis = color_amp.ColorBasis()

        # S channel
        col_dict = my_col_basis.colorize(myamplitude['diagrams'][0],
                                     self.mymodel)

        goal_dict = {(0, 0):color.ColorString([color.T(-1000, 1, 2),
                                               color.f(3, 4, -1000)])}

        self.assertEqual(col_dict, goal_dict)

        # T channel
        col_dict = my_col_basis.colorize(myamplitude['diagrams'][1],
                                     self.mymodel)

        goal_dict = {(0, 0):color.ColorString([color.T(3, 1, -1000),
                                               color.T(4, -1000, 2)])}

        self.assertEqual(col_dict, goal_dict)

        # U channel
        col_dict = my_col_basis.colorize(myamplitude['diagrams'][2],
                                     self.mymodel)

        goal_dict = {(0, 0):color.ColorString([color.T(4, 1, -1000),
                                               color.T(3,-1000, 2)])}

        self.assertEqual(col_dict, goal_dict)

    def test_colorize_uux_ggg(self):
        """Test the colorize function for uu~ > ggg"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))

        myleglist.extend([base_objects.Leg({'id':21,
                                            'state':True})] * 3)

        myprocess = base_objects.Process({'legs':myleglist,
                                        'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude()

        myamplitude.set('process', myprocess)

        # What is checked below is colorize, against diagrams picked by their
        # position, so it wants the plain generation order -- the four gluon
        # merging reorders them
        with misc.TMP_variable(madgraph, 'merge_quartic_vertices', False):
            myamplitude.generate_diagrams()

        my_col_basis = color_amp.ColorBasis()

        # First diagram with two 3-gluon vertices
        col_dict = my_col_basis.colorize(myamplitude['diagrams'][0],
                                     self.mymodel)
        goal_dict = {(0, 0, 0):color.ColorString([color.T(-1000, 2, 1),
                                               color.f(-1001, 3, 4),
                                               color.f(-1000, -1001, 5)])}

        self.assertEqual(col_dict, goal_dict)

        # Diagram with one 4-gluon vertex
        col_dict = my_col_basis.colorize(myamplitude['diagrams'][3],
                                     self.mymodel)

        goal_dict = {(0, 0): color.ColorString([color.T(-1000,2,1),
                                                color.f(-1001,3,5),
                                                color.f(-1001,4,-1000)]),
                     (0, 1): color.ColorString([color.T(-1000,2,1),
                                                color.f(-1002,3,-1000),
                                                color.f(-1002,4,5)]),
                     (0, 2): color.ColorString([color.T(-1000,2,1),
                                                color.f(-1003,3,4),
                                                color.f(-1003,5,-1000)])}

        self.assertEqual(col_dict, goal_dict)

    def test_colorize_funny_model(self):
        """Test the colorize function for uu~ > ggg"""

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

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0]] * 3),
                      'color': [color.ColorString([color.f(0, 1, 2)]),
                                color.ColorString([color.f(0, 2, 1)]),
                                color.ColorString([color.f(1, 2, 0)])],
                      'lorentz':['L1'],
            'couplings':{(0, 0):'G', (2,0):'G'},
                      'orders':{'QCD':1}}))

        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))

        myleglist.extend([base_objects.Leg({'id':21,
                                            'state':True})] * 2)

        myprocess = base_objects.Process({'legs':myleglist,
                                        'model':mymodel})

        myamplitude = diagram_generation.Amplitude()

        myamplitude.set('process', myprocess)

        myamplitude.generate_diagrams()

        my_col_basis = color_amp.ColorBasis()

        # Check that only the color structures that are actually used are included
        col_dict = my_col_basis.colorize(myamplitude['diagrams'][0],
                                         mymodel)
        goal_dict = {(2, 0): color.ColorString([color.f(1,2,-1000),
                                                color.f(3,4,-1000)]),
                     (0, 0): color.ColorString([color.f(-1000,1,2),
                                                color.f(3,4,-1000)]),
                     (0, 2): color.ColorString([color.f(-1000,1,2),
                                                color.f(4,-1000,3)]),
                     (2, 2): color.ColorString([color.f(1,2,-1000),
                                                color.f(4,-1000,3)])}
                     

        self.assertEqual(col_dict, goal_dict)


    def test_color_basis_uux_aggg(self):
        """Test the color basis building for uu~ > aggg (3! elements)"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.extend([base_objects.Leg({'id':21,
                                            'state':True})] * 3)

        myprocess = base_objects.Process({'legs':myleglist,
                                        'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude()

        myamplitude.set('process', myprocess)

        myamplitude.generate_diagrams()

        new_col_basis = color_amp.ColorBasis(myamplitude)

        self.assertEqual(len(new_col_basis), 6)

        # Test the color flow decomposition
        self.assertEqual(new_col_basis.color_flow_decomposition(
                                        {1:3, 2:-3, 3:1, 4:8, 5:8, 6:-8}, 2),
        [{1: [0, 501], 2: [504, 0], 3: [0, 0], 4: [502, 501], 5: [503, 502], 6: [504, 503]},
         {1: [0, 501], 2: [503, 0], 3: [0, 0], 4: [502, 501], 5: [503, 504], 6: [504, 502]},
         {1: [0, 501], 2: [504, 0], 3: [0, 0], 4: [502, 503], 5: [503, 501], 6: [504, 502]},
         {1: [0, 501], 2: [502, 0], 3: [0, 0], 4: [502, 504], 5: [503, 501], 6: [504, 503]},
         {1: [0, 501], 2: [503, 0], 3: [0, 0], 4: [502, 504], 5: [503, 502], 6: [504, 501]},
         {1: [0, 501], 2: [502, 0], 3: [0, 0], 4: [502, 503], 5: [503, 504], 6: [504, 501]}])

    def test_color_flow_string(self):
        """Test the color flow decomposition of various color strings"""

        # qq~>qq~
        my_cs = color.ColorString([color.T(-1000, 1, 2), color.T(-1000, 3, 4)])

        goal_cs = color.ColorString([color.T(1, 4), color.T(3, 2)])
        goal_cs.coeff = fractions.Fraction(1, 2)

        self.assertEqual(color_amp.ColorBasis.get_color_flow_string(my_cs, []),
                         goal_cs)

        # qq~>qq~g
        my_cs = color.ColorString([color.T(-1000, 1, 2),
                                   color.T(-1000, 3, 4),
                                   color.T(5, 4, 6)])
        goal_cs = color.ColorString([color.T(1, 2005),
                                     color.T(3, 2),
                                     color.T(1005, 6)])
        goal_cs.coeff = fractions.Fraction(1, 4)
        self.assertEqual(color_amp.ColorBasis.get_color_flow_string(my_cs,
                                                         [(8, 5, 1005, 2005)]),
                         goal_cs)

        # gg>gg
        my_cs = color.ColorString([color.Tr(-1000, 1, 2),
                                   color.Tr(-1000, 3, 4)])

        goal_cs = color.ColorString([color.T(1001, 2002),
                                     color.T(1002, 2003),
                                     color.T(1003, 2004),
                                     color.T(1004, 2001)])
        goal_cs.coeff = fractions.Fraction(1, 32)

        self.assertEqual(color_amp.ColorBasis.get_color_flow_string(my_cs,
                                                         [(8, 1, 1001, 2001),
                                                          (8, 2, 1002, 2002),
                                                          (8, 3, 1003, 2003),
                                                          (8, 4, 1004, 2004)]),
                         goal_cs)

    def test_color_flow_string_epsilon(self):
        """Test the color flow decomposition of strings including Epsilon
        and color sextets"""

        # g q > trip q
        my_cs = color.ColorString([color.Epsilon(-1000,2,3), color.T(1,4,-1000)])

        goal_cs = color.ColorString([color.T(4,2001), color.Epsilon(2,3,1001)])
        goal_cs.coeff = fractions.Fraction(1, 2)
        self.assertEqual(color_amp.ColorBasis.get_color_flow_string(my_cs,
                                                    [(8, 1, 1001, 2001)]),
                         goal_cs)

        # g q > six q~
        my_cs = color.ColorString([color.K6(3,-1000,4),
                                   color.T(1,-1000,2)])
        goal_cs = color.ColorString([color.T(1001,2), color.T(1003,4),
                                     color.T(3003,2001)])
        goal_cs.coeff = fractions.Fraction(1, 4)
        self.assertEqual(color_amp.ColorBasis.get_color_flow_string(my_cs,
                                                    [(8, 1, 1001, 2001),
                                                     (6, 3, 1003, 3003)]),
                         goal_cs)

        # g q~ > trip > q~ q q~

        my_cs = color.ColorString([color.Epsilon(-1000,2,4),
                                   color.EpsilonBar(-1001,3,5),
                                   color.T(1,-1001,-1000)])

        goal_cs = color.ColorString([color.Epsilon(2,4,1001),
                                     color.EpsilonBar(3,5,2001)])
        goal_cs.coeff = fractions.Fraction(1, 2)
        self.assertEqual(color_amp.ColorBasis.get_color_flow_string(my_cs,
                                                          [(8, 1, 1001, 2001)]),
                         goal_cs)

    
        

class ColorSquareTest(unittest.TestCase):
    """Test class for the color_amp module"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()

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
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        antid = copy.copy(self.mypartlist[2])
        antid.set('is_part', False)

        # 3 gluon vertiex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [color.ColorString([color.f(-1, 0, 2),
                                                   color.f(-1, 1, 3)]),
                                color.ColorString([color.f(-1, 0, 3),
                                                   color.f(-1, 1, 2)]),
                                color.ColorString([color.f(-1, 0, 1),
                                                   color.f(-1, 2, 3)])],
                      'lorentz':['L(p1,p2,p3)', 'L(p2,p3,p1)', 'L3'],
                      'couplings':{(0, 0):'G^2',
                                   (1, 1):'G^2',
                                   (2, 2):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon couplings to up and down quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))


        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)

    def test_color_matrix_multi_gluons(self):
        """Test the color matrix building for gg > n*g with n up to 3"""

        goal = [fractions.Fraction(7, 3),
                fractions.Fraction(19, 6),
                fractions.Fraction(455, 108),
                fractions.Fraction(3641, 648)]


        goal_line1 = [(fractions.Fraction(7, 3), fractions.Fraction(-2, 3)),
                     (fractions.Fraction(19, 6), fractions.Fraction(-1, 3),
                    fractions.Fraction(-1, 3), fractions.Fraction(-1, 3),
                    fractions.Fraction(-1, 3), fractions.Fraction(2, 3)),
                    (fractions.Fraction(455, 108), fractions.Fraction(-29, 54),
                    fractions.Fraction(-29, 54), fractions.Fraction(7, 54),
                    fractions.Fraction(7, 54), fractions.Fraction(17, 27),
                    fractions.Fraction(-29, 54), fractions.Fraction(-1, 27),
                    fractions.Fraction(7, 54), fractions.Fraction(-29, 54),
                    fractions.Fraction(5, 108), fractions.Fraction(-1, 27),
                    fractions.Fraction(7, 54), fractions.Fraction(5, 108),
                    fractions.Fraction(17, 27), fractions.Fraction(-1, 27),
                    fractions.Fraction(7, 54), fractions.Fraction(17, 27),
                    fractions.Fraction(-29, 54), fractions.Fraction(-1, 27),
                    fractions.Fraction(-1, 27), fractions.Fraction(17, 27),
                    fractions.Fraction(17, 27), fractions.Fraction(-10, 27))]

        for n in range(3):
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':21,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':21,
                                             'state':False}))

            myleglist.extend([base_objects.Leg({'id':21,
                                                'state':True})] * (n + 1))

            myprocess = base_objects.Process({'legs':myleglist,
                                            'model':self.mymodel})

            myamplitude = diagram_generation.Amplitude()

            myamplitude.set('process', myprocess)

            myamplitude.generate_diagrams()

            col_basis = color_amp.ColorBasis(myamplitude)

            col_matrix = color_amp.ColorMatrix(col_basis, Nc=3)
            # Check diagonal
            for i in range(len(list(col_basis.items()))):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, i)],
                                 (goal[n], 0))

            # Check first line
            for i in range(len(list(col_basis.items()))):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(0, i)],
                                 (goal_line1[n][i], 0))

    def test_color_matrix_multi_quarks(self):
        """Test the color matrix building for qq~ > n*(qq~) with n up to 2"""

        goal = [fractions.Fraction(9, 1),
                fractions.Fraction(27, 1)]



        goal_line1 = [(fractions.Fraction(9, 1), fractions.Fraction(3, 1)),
                      (fractions.Fraction(27, 1), fractions.Fraction(9, 1),
                       fractions.Fraction(9, 1), fractions.Fraction(3, 1),
                       fractions.Fraction(3, 1), fractions.Fraction(9, 1))
                      ]

        goal_den_list = [[1] * 2, [1] * 6]

        goal_first_line_num = [[9, 3],
                               [27, 9, 9, 3, 3, 9]]

        for n in range(2):
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))

            myleglist.extend([base_objects.Leg({'id':-2,
                                                'state':True}),
                             base_objects.Leg({'id':2,
                                                'state':True})] * (n + 1))

            myprocess = base_objects.Process({'legs':myleglist,
                                            'model':self.mymodel})

            myamplitude = diagram_generation.Amplitude()

            myamplitude.set('process', myprocess)

            myamplitude.generate_diagrams()

            col_basis = color_amp.ColorBasis(myamplitude)

            col_matrix = color_amp.ColorMatrix(col_basis, Nc=3)
            # Check diagonal
            for i in range(len(list(col_basis.items()))):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, i)],
                                 (goal[n], 0))

            # Check first line
            for i in range(len(list(col_basis.items()))):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(0, i)],
                                 (goal_line1[n][i], 0))

            self.assertEqual(col_matrix.get_line_denominators(),
                             goal_den_list[n])
            self.assertEqual(col_matrix.get_line_numerators(0,
                             col_matrix.get_line_denominators()[0]),
                             goal_first_line_num[n])

    def test_color_matrix_Nc_restrictions(self):
        """Test the Nc power restriction during color basis building """

        goal = [fractions.Fraction(3, 8),
                fractions.Fraction(-9, 4),
                fractions.Fraction(45, 16)]

        for n in range(3):
            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':21,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':21,
                                             'state':False}))

            myleglist.extend([base_objects.Leg({'id':21,
                                                'state':True})] * 2)

            myprocess = base_objects.Process({'legs':myleglist,
                                            'model':self.mymodel})

            myamplitude = diagram_generation.Amplitude()

            myamplitude.set('process', myprocess)

            myamplitude.generate_diagrams()

            col_basis = color_amp.ColorBasis(myamplitude)

            col_matrix = color_amp.ColorMatrix(col_basis, Nc=3,
                                                  Nc_power_min=n,
                                                  Nc_power_max=2 * n)

            for i in range(len(list(col_basis.items()))):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, i)],
                                 (goal[n], 0))


    def test_color_matrix_fixed_indices(self):
        """Test index fixing for immutable color string"""

        immutable1 = (('f', (1, -1, -3)), ('T', (-1, -3, 4)))
        immutable2 = (('d', (1, -2, -1)), ('T', (-1, -2, 4)))

        self.assertEqual(color_amp.ColorMatrix.fix_summed_indices(immutable1,
                                                               immutable2),
                         (('d', (1, -4, -5)), ('T', (-5, -4, 4))))

    def test_helper_lcm_functions(self):
        """Test the helper functions to derive common denominators for
        lines in the color matrix."""

        self.assertEqual(color_amp.ColorMatrix.lcm(6, 3), 6)
        self.assertEqual(color_amp.ColorMatrix.lcmm(6, 3, 5, 2), 30)

    def get_gluon_amplitude(self, n_final):
        """Amplitude for g g > n_final gluons, using the test model."""

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':21, 'state':False}))
        myleglist.append(base_objects.Leg({'id':21, 'state':False}))
        myleglist.extend([base_objects.Leg({'id':21,
                                            'state':True})] * n_final)
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', base_objects.Process({'legs':myleglist,
                                                      'model':self.mymodel}))
        myamplitude.generate_diagrams()
        return myamplitude

    def get_quark_amplitude(self, n_pairs):
        """Amplitude for u u~ > n_pairs (u u~), using the test model."""

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2, 'state':False}))
        myleglist.append(base_objects.Leg({'id':-2, 'state':False}))
        for _ in range(n_pairs):
            myleglist.append(base_objects.Leg({'id':2, 'state':True}))
            myleglist.append(base_objects.Leg({'id':-2, 'state':True}))
        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', base_objects.Process({'legs':myleglist,
                                                      'model':self.mymodel}))
        myamplitude.generate_diagrams()
        return myamplitude

    def test_permute_immutable(self):
        """Test the canonical form of a permuted color structure: traces are
        cyclic so they are rotated to start on their smallest index, open
        chains are not, and the color objects are sorted."""

        # Tr is cyclic: exchanging 1 and 3 has to be rotated back
        self.assertEqual(color_amp.permute_immutable((('Tr', (1, 2, 3, 4)),),
                                                     {1: 3, 3: 1}),
                         (('Tr', (1, 4, 3, 2)),))
        # identity permutation still canonicalises
        self.assertEqual(color_amp.permute_immutable((('Tr', (3, 4, 1, 2)),),
                                                     {}),
                         (('Tr', (1, 2, 3, 4)),))
        # T is an open chain, no rotation, but the factors get sorted
        self.assertEqual(color_amp.permute_immutable((('T', (5, 2, 1)),
                                                      ('T', (5, 4, 3))),
                                                     {1: 3, 3: 1}),
                         (('T', (5, 2, 3)), ('T', (5, 4, 1))))
        # indices which are not in the permutation are left alone
        self.assertEqual(color_amp.permute_immutable((('ColorOne', ()),), {}),
                         (('ColorOne', ()),))

    def test_color_basis_symmetry(self):
        """The permutations found have to map the basis onto itself, and for
        a pure gluon process they act transitively so a single row of the
        color matrix determines all the others."""

        for n_final in range(1, 4):
            col_basis = color_amp.ColorBasis(self.get_gluon_amplitude(n_final))
            keys = sorted(col_basis.keys())
            symmetry = color_amp.ColorBasisSymmetry(keys)

            positions = dict((key, i) for i, key in enumerate(keys))
            for induced in symmetry.generators1:
                # a permutation of the basis, and an involution
                self.assertEqual(sorted(induced), list(range(len(keys))))
                for i in range(len(keys)):
                    self.assertEqual(induced[induced[i]], i)

            # all gluons are equivalent, so there is a single orbit
            self.assertEqual(len(symmetry.representatives), 1)
            # every row is reachable from the representative
            self.assertEqual(symmetry.row_rep, [0] * len(keys))
            self.assertEqual(len(positions), len(keys))

    def test_color_basis_symmetry_no_symmetry(self):
        """When no index permutation maps the basis onto itself, every row is
        its own representative and the color matrix falls back to the plain
        scan, which still has to give a correct symmetric matrix."""

        # no transposition maps these bases onto themselves
        for keys in [[(('T', (1, 2, 3)),), (('Tr', (1, 2, 3)),)],
                     [(('T', (1, 2, 3)),)]]:
            symmetry = color_amp.ColorBasisSymmetry(keys)
            self.assertEqual(symmetry.generators1, [])
            self.assertFalse(symmetry.has_symmetry())
            self.assertEqual(symmetry.representatives, list(range(len(keys))))

        # the color matrix then falls back to the plain scan
        keys = [(('T', (1, 2, 3)),)]
        col_matrix = color_amp.ColorMatrix(dict((key, []) for key in keys),
                                           Nc=3)
        self.assertEqual(len(col_matrix), 1)
        self.assertEqual(col_matrix.col_matrix_fixed_Nc[(0, 0)],
                         (fractions.Fraction(4), 0))

    def test_color_matrix_matches_direct_computation(self):
        """The color matrix built by orbits must agree entry by entry with the
        one obtained by computing every entry independently."""

        for amplitude in [self.get_gluon_amplitude(2),
                          self.get_gluon_amplitude(3),
                          self.get_quark_amplitude(1)]:
            col_basis = color_amp.ColorBasis(amplitude)
            col_matrix = color_amp.ColorMatrix(col_basis, Nc=3)
            keys = sorted(col_basis.keys())

            for i1, struct1 in enumerate(keys):
                for i2, struct2 in enumerate(keys):
                    new_struct2 = color_amp.ColorMatrix.fix_summed_indices(
                                                            struct1, struct2)
                    result, result_fixed_Nc = col_matrix.create_new_entry(
                                        struct1, new_struct2, None, None, 3)
                    self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i1, i2)],
                                     result_fixed_Nc)
                    # entries are recycled between equivalent index pairs, so
                    # the terms of the color factor can come in another order
                    self.assertEqual(sorted(map(str, col_matrix[(i1, i2)])),
                                     sorted(map(str, result)))

            # the mapping interface still behaves like the dictionary it was
            self.assertEqual(len(col_matrix), len(keys) ** 2)
            self.assertTrue(col_matrix)
            self.assertTrue((0, 0) in col_matrix)
            self.assertFalse((0, len(keys)) in col_matrix)
            self.assertEqual(sorted(col_matrix.keys()),
                             sorted([(i, j) for i in range(len(keys))
                                            for j in range(len(keys))]))

    def test_relabel_canonical(self):
        """The shortcut taken when a simplified color factor is recycled with
        relabelled indices must agree with the full simplification, and it has
        to be the path actually taken for QCD processes."""

        col_basis = color_amp.ColorBasis()
        col_basis.build(self.get_gluon_amplitude(3))
        self.assertTrue(col_basis._fast_relabel_dict)
        self.assertTrue(all(col_basis._fast_relabel_dict.values()))

        # explicitly compare both paths on every recycled structure
        for color_dict in col_basis._list_color_dict:
            for col_str in color_dict.values():
                canonical_rep, rep_dict = col_str.to_canonical()
                if canonical_rep not in col_basis._canonical_dict:
                    continue
                col_fact = col_basis._canonical_dict[canonical_rep].create_copy()
                col_fact.replace_indices(col_basis._invert_dict(rep_dict))
                for one_str in col_fact:
                    one_str.coeff = one_str.coeff * col_str.coeff
                slow = col_fact.create_copy().simplify().simplify()
                fast = col_basis._canonicalize_strings(col_fact)
                self.assertEqual([s.to_immutable() for s in fast],
                                 [s.to_immutable() for s in slow])
                self.assertEqual([s.coeff for s in fast],
                                 [s.coeff for s in slow])

    def test_color_factor_simplify_merges_like_strings(self):
        """ColorFactor.simplify has to add up similar strings, keeping them in
        order of first appearance."""

        col_fact = color.ColorFactor([
                color.ColorString([color.T(1, 2, 3)],
                                  coeff=fractions.Fraction(2, 3)),
                color.ColorString([color.T(4, 5, 6)],
                                  coeff=fractions.Fraction(1, 5)),
                color.ColorString([color.T(1, 2, 3)],
                                  coeff=fractions.Fraction(1, 3))])
        result = col_fact.simplify()
        self.assertEqual([col_str.to_immutable() for col_str in result],
                         [(('T', (1, 2, 3)),), (('T', (4, 5, 6)),)])
        self.assertEqual([col_str.coeff for col_str in result],
                         [fractions.Fraction(1, 1), fractions.Fraction(1, 5)])

        # strings adding up to zero are dropped
        col_fact = color.ColorFactor([
                color.ColorString([color.T(1, 2, 3)],
                                  coeff=fractions.Fraction(1, 3)),
                color.ColorString([color.T(1, 2, 3)],
                                  coeff=fractions.Fraction(-1, 3))])
        self.assertEqual(len(col_fact.simplify()), 0)

    def test_spanning_generators(self):
        """The spanning subset has to still reach every line, and dropping the
        redundant generators has to make a real difference."""

        col_basis = color_amp.ColorBasis(self.get_gluon_amplitude(3))
        keys = sorted(col_basis.keys())
        symmetry = color_amp.ColorBasisSymmetry(keys)
        kept = symmetry.spanning_generators()
        self.assertTrue(0 < len(kept) < len(symmetry.generators1))

        representatives, representative, parent, gens = symmetry.spanning_tree()
        self.assertEqual(len(gens), len(kept))
        # a single orbit, and every line but the representative has a parent
        self.assertEqual(representatives, [0])
        self.assertEqual(representative, [0] * len(keys))
        self.assertEqual(parent[0], None)
        self.assertTrue(all(parent[i] is not None
                            for i in range(1, len(keys))))
        # every parent link is a generator applied to the parent line
        for line in range(1, len(keys)):
            origin, local = parent[line]
            self.assertEqual(gens[local][origin], line)

    def test_spanning_tree_rebuilds_color_matrix(self):
        """Following the parents back to the representative gives the column
        permutation relating the two lines, which has to rebuild the matrix."""

        for amplitude in [self.get_gluon_amplitude(2),
                          self.get_gluon_amplitude(3),
                          self.get_quark_amplitude(1)]:
            col_basis = color_amp.ColorBasis(amplitude)
            col_matrix = color_amp.ColorMatrix(col_basis, Nc=3)
            keys = sorted(col_basis.keys())
            symmetry = color_amp.ColorBasisSymmetry(keys)
            representatives, representative, parent, gens = \
                                                    symmetry.spanning_tree()
            denominator = max(col_matrix.get_line_denominators())
            rows = [col_matrix.get_line_numerators(i, denominator)
                    for i in range(len(keys))]

            for line in range(len(keys)):
                # walk up to the representative, permuting the columns
                perm = list(range(len(keys)))
                node = line
                while parent[node] is not None:
                    origin, local = parent[node]
                    perm = [gens[local][p] for p in perm]
                    node = origin
                self.assertEqual(node, representative[line])
                self.assertEqual([rows[node][perm[j]]
                                  for j in range(len(keys))], rows[line])


class DDMColorBasisTest(unittest.TestCase):
    """Test the Del Duca-Dixon-Maltoni (n-2)! adjoint color basis"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()
    mymodel = base_objects.Model()

    def setUp(self):
        # same gluon + quark model as ColorSquareTest, built only once
        if not len(self.mypartlist):
            ColorSquareTest.setUp(self)
        color_amp.set_ddm_basis(True)

    def tearDown(self):
        color_amp.set_ddm_basis(False)

    def get_amplitude(self, ids):
        """Amplitude for the process with the given pdg codes, the first two
        being in the initial state."""

        myleglist = base_objects.LegList()
        for i, pdg in enumerate(ids):
            myleglist.append(base_objects.Leg({'id': pdg, 'state': i > 1}))

        myamplitude = diagram_generation.Amplitude()
        myamplitude.set('process', base_objects.Process({'legs': myleglist,
                                                    'model': self.mymodel}))
        myamplitude.generate_diagrams()

        return myamplitude

    def test_ddm_half_ladder(self):
        """The half-ladder structures written down explicitly"""

        self.assertEqual(color_amp.ddm_half_ladder((2,), 1, 3),
                         (('f', (1, 2, 3)),))
        self.assertEqual(color_amp.ddm_half_ladder((3, 2), 1, 4),
                         (('f', (-1, 2, 4)), ('f', (1, 3, -1))))
        self.assertEqual(color_amp.ddm_half_ladder((2, 3, 4), 1, 5),
                         (('f', (-2, 4, 5)), ('f', (-1, 3, -2)),
                          ('f', (1, 2, -1))))

    def test_ddm_basis_size_multi_gluons(self):
        """The DDM basis of gg > n*g has (n+2-2)! elements instead of (n+1)!"""

        for n, size in enumerate([1, 2, 6, 24]):
            amplitude = self.get_amplitude([21] * (n + 3))
            self.assertEqual(len(color_amp.ColorBasis(amplitude)), size)

    def test_ddm_reduction_matches_trace_basis(self):
        """Expanding the DDM decomposition of every diagram back on the trace
        basis must give the direct trace decomposition."""

        def trace_expand(col_str):
            res = collections.defaultdict(fractions.Fraction)
            for cs in color.ColorFactor([col_str]).full_simplify():
                res[(cs.to_immutable(), cs.is_imaginary, cs.Nc_power)] += \
                                                                     cs.coeff
            return dict((k, v) for k, v in res.items() if v)

        for nb_gluons in range(3, 6):
            amplitude = self.get_amplitude([21] * nb_gluons)
            col_basis = color_amp.ColorBasis()
            color_dicts = col_basis.create_color_dict_list(amplitude)
            ends = col_basis.get_ddm_ends()
            self.assertEqual(ends, (1, nb_gluons))

            nb_checked = 0
            for color_dict in color_dicts:
                for col_str in color_dict.values():
                    via_ddm = collections.defaultdict(fractions.Fraction)
                    for ddm_str in color_amp.reduce_to_ddm(col_str, *ends):
                        for k, v in trace_expand(ddm_str).items():
                            via_ddm[k] += v
                    self.assertEqual(trace_expand(col_str),
                                dict((k, v) for k, v in via_ddm.items() if v))
                    nb_checked += 1
            self.assertTrue(nb_checked > 0)

    def test_ddm_color_matrix_gg_gg(self):
        """The 2x2 DDM color matrix of gg > gg, N^2(N^2-1) on the diagonal and
        half of it off diagonal"""

        col_basis = color_amp.ColorBasis(self.get_amplitude([21] * 4))
        col_matrix = color_amp.ColorMatrix(col_basis, Nc=3)

        self.assertEqual(len(col_basis), 2)
        for i, j, goal in [(0, 0, 72), (1, 1, 72), (0, 1, 36), (1, 0, 36)]:
            self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, j)],
                             (fractions.Fraction(goal, 1), 0))

    def test_ddm_color_matrix_is_positive_definite(self):
        """A Gram matrix of real color tensors: the diagonal must be positive
        and the matrix symmetric, for an even as well as an odd number of
        gluons."""

        for nb_gluons in range(3, 6):
            col_basis = color_amp.ColorBasis(self.get_amplitude([21] * nb_gluons))
            col_matrix = color_amp.ColorMatrix(col_basis, Nc=3)
            for i in range(len(col_basis)):
                real, imag = col_matrix.col_matrix_fixed_Nc[(i, i)]
                self.assertTrue(real > 0)
                self.assertEqual(imag, 0)
                for j in range(len(col_basis)):
                    self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, j)],
                                     col_matrix.col_matrix_fixed_Nc[(j, i)])

    def test_ddm_fall_back_on_trace_basis(self):
        """Processes which are not fully adjoint keep the trace basis"""

        # u u~ > g g has T objects in its color structures
        col_basis = color_amp.ColorBasis(self.get_amplitude([2, -2, 21, 21]))
        self.assertEqual(col_basis._ddm_ends, None)
        self.assertEqual(len(col_basis), 2)

        # and so has u u~ > u u~
        col_basis = color_amp.ColorBasis(self.get_amplitude([2, -2, 2, -2]))
        self.assertEqual(col_basis._ddm_ends, None)
        self.assertEqual(len(col_basis), 2)

    def test_ddm_no_color_flow_decomposition(self):
        """The color flow decomposition is not defined in the DDM basis, and
        must say so instead of returning something wrong."""

        col_basis = color_amp.ColorBasis(self.get_amplitude([21] * 4))
        self.assertRaises(color_amp.ColorBasis.ColorBasisError,
                          col_basis.color_flow_decomposition,
                          {1: 8, 2: 8, 3: 8, 4: 8}, 2)


class LoopNcMemoTest(unittest.TestCase):
    """The process-wide canonical color memo must not leak loop_Nc_power."""

    def setUp(self):
        self.saved = dict(color_amp.ColorBasis._canonical_dict)
        color_amp.ColorBasis._canonical_dict.clear()

    def tearDown(self):
        color_amp.ColorBasis._canonical_dict.clear()
        color_amp.ColorBasis._canonical_dict.update(self.saved)

    @staticmethod
    def _fill(compute_loop_nc, loop_Nc_power):
        """Feed one T(-1,1,2)T(-1,3,4) chain through a fresh LoopColorBasis."""
        basis = loop_color_amp.LoopColorBasis(compute_loop_nc=compute_loop_nc)
        col_str = color.ColorString([color.T(-1, 1, 2), color.T(-1, 3, 4)])
        col_str.loop_Nc_power = loop_Nc_power
        basis.update_color_basis({(0,): col_str}, 0)
        return basis

    def test_loop_Nc_power_survives_a_poisoned_memo(self):
        """A compute_loop_nc=False basis must not pin loop_Nc_power for the
        next basis: the memo is keyed on the canonical color string only."""

        # A [sqrvirt=]/standalone-style generation: loop_Nc_power is None.
        self._fill(False, None)
        # A later loop-induced madevent generation asks for the real value.
        basis = self._fill(True, 1)

        powers = [entry[5] for entries in basis.values() for entry in entries]
        self.assertTrue(powers)
        self.assertEqual(powers, [1] * len(powers))

    def test_memo_is_shared_across_loop_bases(self):
        """The memo must stay process-wide: that sharing is what makes
        multi-subprocess loop generation tractable."""

        self._fill(True, 1)
        self.assertTrue(color_amp.ColorBasis._canonical_dict)
        self.assertTrue(self._fill(True, 1)._canonical_dict is
                        color_amp.ColorBasis._canonical_dict)
