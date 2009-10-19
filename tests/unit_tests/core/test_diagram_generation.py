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
import unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

#===============================================================================
# DiagramGenerationTest
#===============================================================================
class DiagramGenerationTest(unittest.TestCase):
    """Test class for the diagram generation"""

    mypartlist = base_objects.ParticleList()
    myinterlist = base_objects.InteractionList()

    ref_dict_to0 = {}
    ref_dict_to1 = {}

    def setUp(self):

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

        self.myinterlist.append(base_objects.Interaction({
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 3),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        self.myinterlist.append(base_objects.Interaction({
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0]] * 4),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'G^2'},
                      'orders':{'QCD':2}}))

        self.myinterlist.append(base_objects.Interaction({
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[1], \
                                             antiu, \
                                             self.mypartlist[0]]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.ref_dict_to0 = self.myinterlist.generate_ref_dict()[0]
        self.ref_dict_to1 = self.myinterlist.generate_ref_dict()[1]


    def test_combine_legs(self):

        # Test gluon interactions

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

        my_combined_legs = [\
                [(l1, l2), l3, l4], [(l1, l2), (l3, l4)], \
                [(l1, l3), l2, l4], [(l1, l3), (l2, l4)], \
                [(l1, l4), l2, l3], [(l1, l4), (l2, l3)], \
                [(l2, l3), l1, l4], [(l2, l4), l1, l3], [(l3, l4), l1, l2], \
                [(l1, l2, l3), l4], [(l1, l2, l4), l3], \
                [(l1, l3, l4), l2], [(l2, l3, l4), l1]\
                ]

        combined_legs = diagram_generation.combine_legs([leg for leg in myleglist], \
                                                        self.ref_dict_to1, \
                                                        3)
        self.assertEqual(combined_legs, my_combined_legs)

        myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                                            'number':num,
                                                            'state':'final'}) \
                                          for num in range(1, 5)])

        # Now test with 3 quarks+3 antiquarks (already flipped sign for IS)

        myleglist[0] = base_objects.Leg({'id':-2,
                                         'number':1,
                                         'state':'initial'})
        myleglist[1] = base_objects.Leg({'id':2,
                                         'number':2,
                                         'state':'initial'})
        myleglist[2] = base_objects.Leg({'id':2,
                                         'number':3,
                                         'state':'final'})
        myleglist[3] = base_objects.Leg({'id':-2,
                                         'number':4,
                                         'state':'final'})
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
                [(l1, l2), l3, l4, l5, l6], [(l1, l2), (l3, l4), l5, l6], \
                [(l1, l2), (l3, l4), (l5, l6)], [(l1, l2), (l3, l6), l4, l5], \
                [(l1, l2), (l3, l6), (l4, l5)], [(l1, l2), (l4, l5), l3, l6], \
                [(l1, l2), (l5, l6), l3, l4], \
                [(l1, l3), l2, l4, l5, l6], [(l1, l3), (l2, l4), l5, l6], \
                [(l1, l3), (l2, l4), (l5, l6)], [(l1, l3), (l2, l6), l4, l5], \
                [(l1, l3), (l2, l6), (l4, l5)], [(l1, l3), (l4, l5), l2, l6], \
                [(l1, l3), (l5, l6), l2, l4], \
                [(l1, l5), l2, l3, l4, l6], [(l1, l5), (l2, l4), l3, l6], \
                [(l1, l5), (l2, l4), (l3, l6)], [(l1, l5), (l2, l6), l3, l4], \
                [(l1, l5), (l2, l6), (l3, l4)], [(l1, l5), (l3, l4), l2, l6], \
                [(l1, l5), (l3, l6), l2, l4], \
                [(l2, l4), l1, l3, l5, l6], [(l2, l4), l1, (l3, l6), l5], \
                [(l2, l4), l1, (l5, l6), l3], \
                [(l2, l6), l1, l3, l4, l5], [(l2, l6), l1, (l3, l4), l5], \
                [(l2, l6), l1, (l4, l5), l3], \
                [(l3, l4), l1, l2, l5, l6], [(l3, l4), l1, l2, (l5, l6)], \
                [(l3, l6), l1, l2, l4, l5], [(l3, l6), l1, l2, (l4, l5)], \
                [(l4, l5), l1, l2, l3, l6], \
                [(l5, l6), l1, l2, l3, l4]
                ]

        combined_legs = diagram_generation.combine_legs([leg for leg in myleglist], \
                                                        self.ref_dict_to1, \
                                                        3)
        self.assertEqual(combined_legs, my_combined_legs)


    def test_diagram_generation(self):

        for ngluon in range (2, 5):

            myleglist = base_objects.LegList([base_objects.Leg({'id':21,
                                              'number':num,
                                              'state':'final'}) \
                                              for num in range(1, ngluon + 3)])

            myleglist[0].set('state', 'initial')
            myleglist[1].set('state', 'initial')

            myproc = base_objects.Process({'legs':myleglist,
                                           'orders':{'QCD':ngluon}})

            print "Number of diagrams for %d gluons: %d" % (ngluon, \
                len(diagram_generation.generate_diagrams(myproc, \
                                                         self.ref_dict_to0, \
                                                         self.ref_dict_to1)))

    def test_expand_list(self):

        mylist = [[1, 2], 3, [4, 5]]

        goal_list = [[1, 3, 4], [1, 3, 5], [2, 3, 4], [2, 3, 5]]

        self.assertEqual(diagram_generation.expand_list(mylist), goal_list)

