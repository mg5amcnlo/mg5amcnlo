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

"""Unit test library for the routines of the core library related to writing
color information for diagrams."""

import copy
import fractions

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color

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
                      'texname':'\gamma',
                      'antitexname':'\gamma',
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
            for i in range(len(col_basis.items())):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, i)],
                                 (goal[n], 0))

            # Check first line
            for i in range(len(col_basis.items())):
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
            for i in range(len(col_basis.items())):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, i)],
                                 (goal[n], 0))

            # Check first line
            for i in range(len(col_basis.items())):
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

            for i in range(len(col_basis.items())):
                self.assertEqual(col_matrix.col_matrix_fixed_Nc[(i, i)],
                                 (goal[n], 0))


    def test_color_matrix_fixed_indices(self):
        """Test index fixing for immutable color string"""

        immutable1 = (('f', (1, -1, -3)), ('T', (-1, -3, 4)))
        immutable2 = (('d', (1, -2, -1)), ('T', (-1, -2, 4)))

        self.assertEqual(color_amp.ColorMatrix.fix_summed_indices(immutable1,
                                                               immutable2),
                         (('d', (1, -2, -5)), ('T', (-5, -2, 4))))

    def test_helper_lcm_functions(self):
        """Test the helper functions to derive common denominators for
        lines in the color matrix."""

        self.assertEqual(color_amp.ColorMatrix.lcm(6, 3), 6)
        self.assertEqual(color_amp.ColorMatrix.lcmm(6, 3, 5, 2), 30)
