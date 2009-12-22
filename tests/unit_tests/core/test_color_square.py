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

"""Unit test library for the routines of the core library related to squaring
color information."""

import copy
import unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.core.color_square as color_square

class ColorSquareTest(unittest.TestCase):
    """Test class for the color_square module"""

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

        # A quark U and its antiparticle
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
                      'color': [['f(0,1,2)']],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # 4 gluon vertex
        self.myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': [['f(-1,0,2)', 'f(-1,1,3)'],
                                ['f(-1,0,3)', 'f(-1,1,2)'],
                                ['f(-1,0,1)', 'f(-1,2,3)']],
                      'lorentz':['L1', 'L2', 'L3'],
                      'couplings':{(0, 0):'G^2',
                                   (1, 1):'G^2',
                                   (2, 2):'G^2'},
                      'orders':{'QCD':2}}))

        # Gluon couplings to quarks
        self.myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': [['T(2,0,1)']],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[2], \
                                             antid, \
                                             self.mypartlist[0]]),
                      'color': [['T(2,0,1)']],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('interactions', self.myinterlist)


#    def test_square_basis_uux_ddxddx(self):
#        """Test the build_basis function for uu~ > dd~dd~"""
#
#        myleglist = base_objects.LegList()
#
#        myleglist.append(base_objects.Leg({'id':2,
#                                         'state':'initial'}))
#        myleglist.append(base_objects.Leg({'id':-2,
#                                         'state':'initial'}))
#
#        myleglist.append(base_objects.Leg({'id':1,
#                                         'state':'final'}))
#        myleglist.append(base_objects.Leg({'id':-1,
#                                         'state':'final'}))
#        myleglist.append(base_objects.Leg({'id':1,
#                                         'state':'final'}))
#        myleglist.append(base_objects.Leg({'id':-1,
#                                         'state':'final'}))
#
#
#        myprocess = base_objects.Process({'legs':myleglist,
#                                        'model':self.mymodel})
#
#        myamplitude = diagram_generation.Amplitude()
#
#        myamplitude.set('process', myprocess)
#
#        myamplitude.generate_diagrams()
#
#        col_fact_list = [color_amp.colorize(diag, self.mymodel) \
#                         for diag in myamplitude['diagrams']]
#
#        color_basis = {}
#        for index, diag in enumerate(myamplitude['diagrams']):
#            col_fact = color_amp.colorize(diag, self.mymodel)
#            color_amp.build_color_basis(col_fact,
#                                        color_basis,
#                                        index)
#        color = color_square.build_color_matrix(color_basis, color_basis)
#
#        for line in color[0]:
#            map(lambda x:x.fix_Nc(), line)
#            print line
#
#        print color[1]
#
#    def test_square_basis_gg_gg(self):
#        """Test the build_basis function for gg >gg"""
#
#        myleglist = base_objects.LegList()
#
#        myleglist.append(base_objects.Leg({'id':21,
#                                         'state':'initial'}))
#        myleglist.append(base_objects.Leg({'id':21,
#                                         'state':'initial'}))
#
#        myleglist.extend([base_objects.Leg({'id':21,
#                                         'state':'final'})] * 3)
#
#        myprocess = base_objects.Process({'legs':myleglist,
#                                        'model':self.mymodel})
#
#        myamplitude = diagram_generation.Amplitude()
#
#        myamplitude.set('process', myprocess)
#
#        myamplitude.generate_diagrams()
#
#        col_fact_list = [color_amp.colorize(diag, self.mymodel) \
#                         for diag in myamplitude['diagrams']]
#
#        color_basis = {}
#        for index, diag in enumerate(myamplitude['diagrams']):
#            col_fact = color_amp.colorize(diag, self.mymodel)
#            color_amp.build_color_basis(col_fact,
#                                        color_basis,
#                                        index)
#
#        color = color_square.build_color_matrix(color_basis,
#                                                    color_basis,
#                                                    equal=False)
#
#        for line in color[0]:
#            map(lambda x:x.fix_Nc(), line)
#            print line
#
#        print color[1]
