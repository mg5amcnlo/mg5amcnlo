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
import os

import madgraph
import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import tests.unit_tests as unittest

#===============================================================================
# ParticleTest
#===============================================================================
class ParticleTest(unittest.TestCase):
    """Test class for the Particle object"""

    mydict = {}
    mypart = None

    def setUp(self):

        self.mydict = {'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'mt',
                      'width':'wt',
                      #'texname':'t',
                      #'antitexname':'\\overline{t}',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      #'propagating':True,
                      'is_part':True,
                      #'ghost':False,
                      'type': '',
                      'counterterm':{('QCD',((1,2),(3,4))):{0:'GC_0',-1:'GC_1'}},
                      'propagator':'',
                      'self_antipart':False}

        self.mypart = base_objects.Particle(self.mydict)

    def test_setget_particle_correct(self):
        "Test correct Particle object __init__, get and set"

        mypart2 = base_objects.Particle()

        # First fill mypart2 it using set
        for prop in self.mydict.keys():
            mypart2.set(prop, self.mydict[prop])

        # Check equality between Particle objects
        self.assertEqual(self.mypart, mypart2)

        # Check equality with initial dic using get
        for prop in self.mypart.keys():
            self.assertEqual(self.mypart.get(prop), self.mydict[prop])

    def test_setget_particle_exceptions(self):
        "Test error raising in Particle __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.Particle.PhysicsObjectError,
                          base_objects.Particle,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.Particle,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.mypart.get,
                          a_number)
        self.assertRaises(base_objects.Particle.PhysicsObjectError,
                          self.mypart.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.mypart.set,
                          a_number, 0)
        self.assertRaises(base_objects.Particle.PhysicsObjectError,
                          self.mypart.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for particle properties"""

        test_values = [
                       {'prop':'name',
                        'right_list':['h', 'e+', 'e-', 'u~',
                                      'k++', 'k--', 'T', 'u+~'],
                        'wrong_list':['', 'x ', 'e?', '{}', '$', 's/', '>', 
                                      '[]']},
                       {'prop':'spin',
                        'right_list':[1, 2, 3, 4, 5],
                        'wrong_list':[-1, 0, 'a', 6]},
                       {'prop':'color',
                        'right_list':[1, 3, 6, 8],
                        'wrong_list':[2, 0, 'a', 23, -1, -3, -6]},
                       {'prop':'mass',
                        'right_list':['me', 'zero', 'mm2'],
                        'wrong_list':['m+', '', ' ', 'm~']},
                       {'prop':'pdg_code',
                        'right_list':[1, 12, 80000000, -1],
                        'wrong_list':[1.2, 'a']},
                       {'prop':'line',
                        'right_list':['straight', 'wavy', 'curly', 'dashed'],
                        'wrong_list':[-1, 'wrong']},
                       {'prop':'charge',
                        'right_list':[1., -1., -2. / 3., 0.],
                        'wrong_list':[1, 'a']},
                       #{'prop':'propagating',
                       # 'right_list':[True, False],
                       # 'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'is_part',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
                       {'prop':'self_antipart',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]}
                       ]

        temp_part = self.mypart

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_part.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_part.set(test['prop'], x), '%s not allowd for %s ' %(x, test['prop']))

    def test_representation(self):
        """Test particle object string representation."""

        goal = "{\n"
        goal = goal + "    \'name\': \'t\',\n"
        goal = goal + "    \'antiname\': \'t~\',\n"
        goal = goal + "    \'spin\': 2,\n"
        goal = goal + "    \'color\': 3,\n"
        goal = goal + "    \'charge\': 0.67,\n"
        goal = goal + "    \'mass\': \'mt\',\n"
        goal = goal + "    \'width\': \'wt\',\n"
        goal = goal + "    \'pdg_code\': 6,\n"
        #goal = goal + "    \'texname\': \'t\',\n"
        #goal = goal + "    \'antitexname\': \'\\overline{t}\',\n"
        goal = goal + "    \'line\': \'straight\',\n"
        #goal = goal + "    \'propagating\': True,\n"
        goal = goal + "    \'propagator\': '',\n"
        goal = goal + "    \'is_part\': True,\n"
        goal = goal + "    \'self_antipart\': False,\n"        
        goal = goal + "    \'type\': '',\n"
        #goal = goal + "    \'ghost\': False,\n"
        goal = goal + "    \'counterterm\': {('QCD', ((1, 2), (3, 4))): {0: 'GC_0', -1: 'GC_1'}}\n}"

        self.assertEqual(goal.split('\n'), str(self.mypart).split('\n'))

    def test_get_pdg_code(self):
        """Test the get_pdg_code function of Particle"""

        test_part = copy.copy(self.mypart)
        self.assertEqual(test_part.get_pdg_code(), 6)
        test_part.set('is_part', False)
        self.assertEqual(test_part.get_pdg_code(), -6)
        test_part.set('self_antipart', True)
        self.assertEqual(test_part.get_pdg_code(), 6)

    def test_get_anti_pdg_code(self):
        """Test the get_anti_pdg_code function of Particle"""

        test_part = copy.copy(self.mypart)
        self.assertEqual(test_part.get_anti_pdg_code(), -6)
        test_part.set('is_part', False)
        self.assertEqual(test_part.get_anti_pdg_code(), 6)
        test_part.set('self_antipart', True)
        self.assertEqual(test_part.get_pdg_code(), 6)

    def test_get_helicity_states(self):
        """Test the get_anti_pdg_code function of Particle"""

        test_part = copy.copy(self.mypart)
        self.assertEqual(test_part.get_helicity_states(), [-1, 1])
        test_part.set('spin', 1)
        self.assertEqual(test_part.get_helicity_states(), [0])
        test_part.set('spin', 3)
        self.assertEqual(test_part.get_helicity_states(), [-1, 0, 1])
        test_part.set('mass', 'Zero')
        self.assertEqual(test_part.get_helicity_states(), [-1, 1])
        test_part.set('spin', 5)
        self.assertEqual(test_part.get_helicity_states(), [-2, -1, 1, 2])
        test_part.set('mass', 'M')
        self.assertEqual(test_part.get_helicity_states(), [-2, -1, 0, 1, 2])
        
    def test_particle_list(self):
        """Test particle list initialization, search and dict generation
        functions."""

        mylist = [self.mypart]
        mypartlist = base_objects.ParticleList(mylist)

        not_a_part = 1

        for part in mypartlist:
            self.assertEqual(part, self.mypart)

        self.assertRaises(AssertionError,
                          mypartlist.append,
                          not_a_part)
        # test particle search
        self.assertEqual(self.mypart,
                         mypartlist.get_copy(self.mypart['name']))
        anti_part = copy.copy(self.mypart)
        anti_part.set('is_part', False)
        self.assertEqual(anti_part,
                         mypartlist.get_copy(self.mypart['antiname']))
        self.assertEqual(None,
                         mypartlist.get_copy('none'))

        mydict = {6:self.mypart, -6:anti_part}

        self.assertEqual(mydict, mypartlist.generate_dict())

        my_ref_dict = {(6, -6):[0], (-6, 6):[0]}

        self.assertEqual(my_ref_dict, mypartlist.generate_ref_dict())


#===============================================================================
# InteractionTest
#===============================================================================
class InteractionTest(unittest.TestCase):
    """Test class for the interaction object."""

    mydict = {}
    myinter = None
    mypart = None

    def setUp(self):

        self.mypart = base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'mt',
                      'width':'wt',
                      'texname':'t',
                      'antitexname':'\\overline{t}',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True})

        self.mydict = {'id': 1,
                       'particles': base_objects.ParticleList([self.mypart] * 4),
                       'color': [color.ColorString([color.f(1, 2, 3)]),
                                 color.ColorString([color.d(1, 2, 3)])],
                       'lorentz':['L1', 'L2'],
                       'couplings':{(0, 0):'g00',
                                    (0, 1):'g01',
                                    (1, 0):'g10',
                                    (1, 1):'g11'},
                       'orders':{'QCD':1, 'QED':1},
                       'loop_particles':[[]],
                       'perturbation_type':'QCD',
                       'type':'base'}

        self.myinter = base_objects.Interaction(self.mydict)

    def test_setget_interaction_correct(self):
        "Test correct interaction object __init__, get and set"

        myinter2 = base_objects.Interaction()

        # First fill myinter2 it using set
        for prop in ['id', 'particles', 'color', 'lorentz', 'couplings',
                     'orders', 'type', 'loop_particles','perturbation_type']:
            myinter2.set(prop, self.mydict[prop])

        # Check equality between Interaction objects
        self.assertEqual(self.myinter, myinter2)

        # Check equality with initial dic using get
        for prop in self.myinter.keys():
            self.assertEqual(self.myinter.get(prop), self.mydict[prop])

    def test_setget_interaction_exceptions(self):
        "Test error raising in Interaction __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.Interaction.PhysicsObjectError,
                          base_objects.Interaction,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.Interaction,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.myinter.get,
                          a_number)
        self.assertRaises(base_objects.Interaction.PhysicsObjectError,
                          self.myinter.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.myinter.set,
                          a_number, 0)
        self.assertRaises(base_objects.Interaction.PhysicsObjectError,
                          self.myinter.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for interaction properties"""

        test_values = [
                       {'prop':'particles',
                        'right_list':[base_objects.ParticleList([]),
                                      base_objects.ParticleList([self.mypart] * 3)],
                        'wrong_list':[1, 'x ', [self.mypart, 1], [1, 2]]},
                       {'prop':'color',
                        'right_list':[[], [color.ColorString([color.f(1, 2, 3)]),
                                           color.ColorString([color.f(1, 2, 3)])]],
                        'wrong_list':[1, 'a', ['a', 1]]},
                       {'prop':'lorentz',
                        'right_list':[[], ['L1'], ['L1', 'L2']],
                        'wrong_list':[1, 'a', ['a', 1]]},
                       {'prop':'orders',
                        'right_list':[{}, {'QCD':2}, {'QED':1, 'QCD':1}],
                        'wrong_list':[1, 'a', {1:'a'}]},
                       # WARNING: Valid value should be defined with
                       # respect to the last status of myinter, i.e.
                       # the last good color and lorentz lists
                       {'prop':'couplings',
                        'right_list':[{(0, 0):'g00', (0, 1):'g01',
                                       (1, 0):'g10', (1, 1):'g11'}],
                        'wrong_list':[{(0):'g00', (0, 1):'g01',
                                       (1, 0):'g10', (1, 2):'g11'}]}
                       ]

        mytestinter = self.myinter

        for test in test_values:
            for x in test['right_list']:
                self.assert_(mytestinter.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(mytestinter.set(test['prop'], x))

    def test_representation(self):
        """Test interaction object string representation."""

        goal = "{\n"
        goal = goal + "    \'id\': %d,\n" % self.myinter['id']
        goal = goal + "    \'particles\': [%s],\n" % \
                            ','.join([str(self.mypart.get_pdg_code())]*4)
        goal = goal + "    \'color\': [1 f(1,2,3), 1 d(1,2,3)],\n"
        goal = goal + "    \'lorentz\': [\'L1\', \'L2\'],\n"
        goal = goal + "    \'couplings\': %s,\n" % \
                                    repr(self.myinter['couplings'])
        goal = goal + "    \'orders\': %s,\n" % repr(self.myinter['orders'])
        goal = goal + "    \'loop_particles\': [[]],\n"        
        goal = goal + "    \'type\': \'base\',\n"
        goal = goal + "    \'perturbation_type\': 'QCD'\n}"

        self.assertEqual(goal, str(self.myinter))

    def test_generating_dict_to_0(self):
        """Test the dictionary generation routine"""

        # Create a non trivial 4-interaction
        part1 = base_objects.Particle()
        part1.set('pdg_code', 1)
        part2 = base_objects.Particle()
        part2.set('pdg_code', 2)
        part2.set('is_part', False)
        part3 = base_objects.Particle()
        part3.set('pdg_code', 3)
        part4 = base_objects.Particle()
        part4.set('pdg_code', 4)
        part4.set('is_part', False)
        part4.set('self_antipart', True)

        myinter = base_objects.Interaction()
        myinter.set('particles', base_objects.ParticleList([part1,
                                                           part2,
                                                           part3,
                                                           part4]))
        ref_dict_to0 = {}
        ref_dict_to1 = {}

        myinter.generate_dict_entries(ref_dict_to0, ref_dict_to1)

        goal_ref_dict_to0 = { (-2, 1, 3, 4):[0]}

        self.assertEqual(ref_dict_to0, goal_ref_dict_to0)

        # Check it still work if I add a 3-interaction

        myinterlist = base_objects.InteractionList([myinter])

        add_inter = base_objects.Interaction()
        add_inter.set('particles', base_objects.ParticleList([part1,
                                                              part2,
                                                              part3]))
        myinterlist.append(add_inter)

        goal_ref_dict_to0[(-2, 1, 3)] = [0]

        self.assertEqual(myinterlist.generate_ref_dict()[0], goal_ref_dict_to0)

    def test_generating_dict_to_1(self):
        """Test the dictionary generation routine generate_ref_dict"""

        # Create a non trivial 4-interaction
        part1 = base_objects.Particle()
        part1.set('pdg_code', 1)
        part2 = base_objects.Particle()
        part2.set('pdg_code', 2)
        part2.set('is_part', False)
        part3 = base_objects.Particle()
        part3.set('pdg_code', 3)
        part4 = base_objects.Particle()
        part4.set('pdg_code', 4)
        part4.set('is_part', False)
        part4.set('self_antipart', True)

        myinter = base_objects.Interaction()
        myinter.set('particles', base_objects.ParticleList([part1,
                                                           part2,
                                                           part3,
                                                           part4]))
        ref_dict_to0 = {}
        ref_dict_to1 = {}

        myinter.generate_dict_entries(ref_dict_to0, ref_dict_to1)

        goal_ref_dict_to1 = {(-2, 3, 4):[(-1, 0)],
                            (1, 3, 4):[(2, 0)],
                            (-2, 1, 4):[(-3, 0)],
                            (-2, 1, 3):[(4, 0)]}

        self.assertEqual(ref_dict_to1, goal_ref_dict_to1)

        # Check it still work if I add a 3-interaction

        myinterlist = base_objects.InteractionList([myinter] * 10)

        add_inter = base_objects.Interaction()
        add_inter.set('particles', base_objects.ParticleList([part1,
                                                              part2,
                                                              part3]))
        myinterlist.append(add_inter)

        goal_ref_dict_to1[(-2, 1)] = [(-3, 0)]
        goal_ref_dict_to1[(1, 3)] = [(2, 0)]
        goal_ref_dict_to1[(-2, 3)] = [(-1, 0)]

        self.assertEqual(myinterlist.generate_ref_dict()[1], goal_ref_dict_to1)

    def test_interaction_list(self):
        """Test interaction list initialization"""

        # Create a dummy list of interactions with ids
        mylist = [copy.copy(inter) for inter in [self.myinter] * 3]
        for i in range(1, 4):
            mylist[i - 1].set('id', i)
        myinterlist = base_objects.InteractionList(mylist)

        # Check error raising
        not_a_inter = 1
        self.assertRaises(AssertionError,
                          myinterlist.append,
                          not_a_inter)

        # Check reference dict
        mydict = {}
        for i in range(1, 4):
            mydict[i] = myinterlist[i - 1]
        self.assertEqual(mydict, myinterlist.generate_dict())


#===============================================================================
# ModelTest
#===============================================================================
class ModelTest(unittest.TestCase):
    """Test class for the Model object"""

    mymodel = base_objects.Model()

    def setUp(self):

        self.myinterlist = base_objects.InteractionList()
        self.mypartlist = base_objects.ParticleList()

        # Create a model with gluon and top quark + a single interaction
        self.mypartlist.append(base_objects.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'mt',
                  'width':'wt',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  'self_antipart':False}))
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

        antit = copy.copy(self.mypartlist[0])
        antit.set('is_part', False)

        self.myinterlist.append(base_objects.Interaction({
                      'id':1,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0], \
                                             antit, \
                                             self.mypartlist[1]]),
                      'color': [color.ColorString([color.f(1, 2, 3),
                                                   color.d(1, 2, 3)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('interactions', self.myinterlist)
        self.mymodel.set('particles', self.mypartlist)
        self.mymodel.set('order_hierarchy', {'QCD': 1, 'QED': 2})

    def test_model_initialization(self):
        """Test the default Model class initialization"""
        mymodel = base_objects.Model()

        self.assertEqual(mymodel['particles'],
                         base_objects.ParticleList())
        self.assertEqual(mymodel['interactions'],
                         base_objects.InteractionList())

    def test_get_particle(self):
        """ test that get_particle is working """
        
        obj = self.mymodel.get_particle(6)
        self.assertEqual(obj['name'], 't')
        obj = self.mymodel.get_particle(7)
        self.assertEqual(obj, None)        

    def test_get_interaction(self):
        """ test that get_particle is working """
        
        obj = self.mymodel.get_interaction(1)
        self.assertEqual(obj['lorentz'], ['L1'])
        obj = self.mymodel.get_interaction(7)
        self.assertEqual(obj, None)  

    def test_setget_model_correct(self):
        """Test correct Model object get and set"""

        # Test the particles item
        mydict = {'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'mt',
                  'width':'wt',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True}

        mypart = base_objects.Particle(mydict)
        mypartlist = base_objects.ParticleList([mypart])
        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)

        self.assertEqual(mymodel.get('particles'), mypartlist)

    def test_setget_model_error(self):
        """Test error raising in Model object get and set"""

        mymodel = base_objects.Model()
        not_a_string = 1.

        # General
        self.assertRaises(AssertionError,
                          mymodel.get,
                          not_a_string)
        self.assertRaises(base_objects.Model.PhysicsObjectError,
                          mymodel.get,
                          'wrong_key')
        self.assertRaises(AssertionError,
                          mymodel.set,
                          not_a_string, None)
        self.assertRaises(base_objects.Model.PhysicsObjectError,
                          mymodel.set,
                          'wrong_subclass', None)

        # For each subclass
        self.assertRaises(AssertionError,
                          mymodel.set, 'particles', not_a_string)
        self.assertRaises(AssertionError,
                          mymodel.set, 'interactions', not_a_string)

    def test_dictionaries(self):
        """Test particle dictionary in Model"""

        antitop = copy.copy(self.mypartlist[0])
        antitop.set('is_part', False)
        mypartdict = {6:self.mypartlist[0], -6:antitop, 21:self.mypartlist[1]}
        self.assertEqual(mypartdict, self.mymodel.get('particle_dict'))

        myinterdict = {1:self.myinterlist[0]}
        self.assertEqual(myinterdict, self.mymodel.get('interaction_dict'))

        particles = copy.copy(self.mymodel.get('particles'))
        particles.append(base_objects.Particle({'name':'a',
                  'antiname':'a',
                  'spin':3,
                  'color':0,
                  'mass':'zero',
                  'width':'zero',
                  'texname':'\gamma',
                  'antitexname':'\gamma',
                  'line':'wavy',
                  'charge':0.,
                  'pdg_code':22,
                  'propagating':True,
                  'self_antipart':True}))
        self.mymodel.set('particles', particles)
        mypartdict[22] = particles[2]
        self.assertEqual(mypartdict, self.mymodel.get('particle_dict'))

        interactions = copy.copy(self.mymodel.get('interactions'))
        interactions.append(base_objects.Interaction({
                      'id':2,
                      'particles': base_objects.ParticleList(\
                                            [self.mypartlist[0], \
                                             antitop, \
                                             self.mymodel['particles'][2]]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))
        self.mymodel.set('interactions', interactions)
        myinterdict[2] = interactions[1]
        self.assertEqual(myinterdict, self.mymodel.get('interaction_dict'))
        
    def test_ref_dict_multiple_interactions(self):
        """Test ref_dicts with multiple interactions with same particles"""

        myinterlist = base_objects.InteractionList()
        mypartlist = base_objects.ParticleList()

        # Create a model with gluon and top quark + a single interaction
        mypartlist.append(base_objects.Particle({'name':'t',
                  'antiname':'t~',
                  'spin':2,
                  'color':3,
                  'mass':'mt',
                  'width':'wt',
                  'texname':'t',
                  'antitexname':'\\overline{t}',
                  'line':'straight',
                  'charge':2. / 3.,
                  'pdg_code':6,
                  'propagating':True,
                  'self_antipart':False}))
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

        antit = copy.copy(mypartlist[0])
        antit.set('is_part', False)

        myinterlist.append(base_objects.Interaction({
                      'id':1,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0], \
                                             antit, \
                                             mypartlist[1]]),
                      'color': [color.ColorString([color.f(1, 2, 3),
                                                   color.d(1, 2, 3)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id':2,
                      'particles': base_objects.ParticleList(\
                                            [mypartlist[0], \
                                             antit, \
                                             mypartlist[1]]),
                      'color': [color.ColorString([color.f(1, 2, 3),
                                                   color.d(1, 2, 3)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ2'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        self.assertEqual(mymodel.get('ref_dict_to0'),
                         {(-6, 6, 21): [1, 2], (6, -6): [0],
                          (21, 21): [0], (-6, 6): [0]})

        self.assertEqual(mymodel.get('ref_dict_to1'),
                         {(-6, 21): [(-6, 1), (-6, 2)],
                          (-6, 6): [(21, 1), (21, 2)],
                          (6, 21): [(6, 1), (6, 2)]})

    def test_check_majoranas(self):
        """Test the check_majoranas function"""

        # By default, mymodel has no fermion flow clashes
        self.assertFalse(self.mymodel.get('got_majoranas'))

        # Add a Majorana particle
        self.mypartlist.append(base_objects.Particle({'name':'n1',
                  'antiname':'n1',
                  'spin':2,
                  'color':[],
                  'mass':'mn1',
                  'width':'wn1',
                  'line':'straight',
                  'charge':0.,
                  'pdg_code':1000022,
                  'propagating':True,
                  'self_antipart':True}))

        self.mymodel.set('particles', self.mypartlist)
        self.assertTrue(self.mymodel.get('got_majoranas'))

        # Remove the Majorana particle again
        self.mypartlist.pop(-1)
        self.mymodel.set('particles', self.mypartlist)
        self.assertFalse(self.mymodel.get('got_majoranas'))

        # Add a new set of particles
        self.mypartlist.append(base_objects.Particle({'name':'x1',
                  'antiname':'x1~',
                  'spin':2,
                  'color':[],
                  'mass':'mx1',
                  'width':'wx1',
                  'line':'straight',
                  'charge':0.,
                  'pdg_code':10024,
                  'propagating':True,
                  'self_antipart':False}))
        
        self.mypartlist.append(base_objects.Particle({'name':'x2',
                  'antiname':'x2~',
                  'spin':2,
                  'color':[],
                  'mass':'mx2',
                  'width':'wx2',
                  'line':'straight',
                  'charge':0.,
                  'pdg_code':10025,
                  'propagating':True,
                  'self_antipart':False}))

        self.mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':[],
                      'mass':'zero',
                      'width':'zero',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':32,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        self.mymodel.set('particles', self.mypartlist)
        
        # Add a non-fermion clash 4-fermion interaction
        self.myinterlist.append(base_objects.Interaction({
                      'id':2,
                      'particles': base_objects.ParticleList(\
                                         [self.mymodel.get_particle(10024), \
                                          self.mymodel.get_particle(-10025), \
                                          self.mymodel.get_particle(10025), \
                                          self.mymodel.get_particle(-10024), \
                                          self.mymodel.get_particle(32)]),
                      'color': [color.ColorString([color.f(1, 2, 3),
                                                   color.d(1, 2, 3)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('interactions', self.myinterlist)
        self.assertFalse(self.mymodel.get('got_majoranas'))
        
        # Add a fermion clash 4-fermion interaction
        self.myinterlist.append(base_objects.Interaction({
                      'id':3,
                      'particles': base_objects.ParticleList(\
                                         [self.mymodel.get_particle(10024), \
                                          self.mymodel.get_particle(-10025), \
                                          self.mymodel.get_particle(10025), \
                                          self.mymodel.get_particle(10024), \
                                          self.mymodel.get_particle(32)]),
                      'color': [color.ColorString([color.f(1, 2, 3),
                                                   color.d(1, 2, 3)])],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('interactions', self.myinterlist)
        self.assertTrue(self.mymodel.get('got_majoranas'))


    def test_pass_in_standard_name(self):
        """Test if we can overwrite the name of the model following MG 
        convention"""
        
        model_name = [(part.get('name'), part.get('antiname')) \
                                          for part in self.mymodel['particles']]

        model2 = copy.copy(self.mymodel)
        
        # check that standard name are not modified
        model2.pass_particles_name_in_mg_default()
        model2_name = [(part.get('name'), part.get('antiname')) \
                                                for part in model2['particles']]
        self.assertEqual(set(model_name),set(model2_name))
        
        # add a particle with non conventional name
        particles = model2['particles']
        particles.append(base_objects.Particle({'name':'ee',
                  'antiname':'eex',
                  'pdg_code':1}))
        model2.set('particles', particles)

        model2.pass_particles_name_in_mg_default()

        model2_name = [(part.get('name'), part.get('antiname')) \
                                                for part in model2['particles']]        
        self.assertEqual(set(model_name + [('d','d~')]), set(model2_name))
        
        # add a particles with non conventional name with the conventional name
        #associtaed to another particle
        particles.append(base_objects.Particle({'name':'u',
                  'antiname':'d~',
                  'pdg_code':100}))
        particles.append(base_objects.Particle({'name':'ee',
                  'antiname':'eex',
                  'pdg_code':2}))
        model2.set('particles', particles)

        self.assertRaises(madgraph.MadGraph5Error, \
                                       model2.pass_particles_name_in_mg_default)

    def test_get_max_WEIGHTED(self):
        """Test get_max_WEIGHTED"""

        self.mymodel.get('interactions').append(\
            base_objects.Interaction({
                      'id':10,
                      'particles': base_objects.ParticleList(\
                                         [self.mymodel.get_particle(6), \
                                          self.mymodel.get_particle(-6), \
                                          self.mymodel.get_particle(21), \
                                          self.mymodel.get_particle(21), \
                                          self.mymodel.get_particle(21)]),
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1, 'QED':5}}))


        self.assertEqual(self.mymodel.get_max_WEIGHTED(), 11./3)

        self.mymodel.get('interactions').pop(-1)

#===============================================================================
# ModelTest
#===============================================================================
class ModelTest2(unittest.TestCase):
    """Test class for the Model object from a correct load"""
    
    def setUp(self):
        """ """
        import madgraph.interface.master_interface as Cmd
        cmd = Cmd.MasterCmd() 
        cmd.do_import('model sm')
        self.model = cmd._curr_model
        
    def test_change_to_complex_mass_scheme(self):
        """Check that a model can be converted to complex mass scheme"""
        
        model = copy.deepcopy(self.model)
        model.change_mass_to_complex_scheme()
        
        # Check that the Width of the W is not anymore in the external parameter
        # and the yukawa
        self.assertEqual(len(self.model['parameters'][('external',)]) -3,
                         len(model['parameters'][('external',)]) )
        
        
#        # Check that the Width of the W is in internal parameter
#        WW = None
#        WComplex = None
#        MW = None
#        for param in model['parameters'][('aEWM1',)]:
#            if param.name not in ['CMASS_MW', 'WW', 'MW']:
#                continue
#            elif param.name == 'CMASS_MW':
#                WComplex = param
#                self.assertFalse(WW)
#            elif param.name == 'WW':
#                WW = param
#            elif param.name == 'MW': 
#                MW = param
#                self.assertFalse(WW)
#                self.assertFalse(WComplex)
#        self.assertTrue(WW)
#        self.assertTrue(MW)
#        self.assertTrue(WComplex)
#        # Check that WW and MW are the real/imaginary part
#        self.assertEqual(WW.expr, '-1 * im(CMASS_MW**2) / MW')
#        self.assertEqual(['cmath.sqrt(re(%s**2))' % WComplex.expr], [MW.expr])
        
        # Check that MZ has a complex_mass definition
        # and that the width and the mass are external
        found = 0
        for param in model['parameters'][('external',)]:
            if param.name in ['mdl_WZ','mdl_MZ','mdl_WW','mdl_MW']:
                self.assertEqual(param.type, 'real')
                found += 1
                
        self.assertEqual(found, 4)
        
        found=0
        for param in model['parameters'][tuple([])]:
            if param.name in ['CMASS_mdl_MZ']:
                self.assertEqual(param.expr, 'cmath.sqrt(mdl_MZ**2 - complex(0,1) * mdl_MZ * mdl_WZ)')
                found += 1
                self.assertEqual(param.type, 'complex')
            
            # check that other parameter are changed correctly
            if param.name in ['mdl_MZ__exp__2']:
                self.assertEqual(param.expr, 'CMASS_mdl_MZ**2')
                found += 1
        self.assertEqual(found, 2)
        
#===============================================================================
# LegTest
#===============================================================================
class LegTest(unittest.TestCase):
    """Test class for the Leg object"""

    mydict = {}
    myleg = None

    def setUp(self):

        self.mydict = {'id':3,
                      'number':5,
                      'state':True,
                      'from_group':False,
                      'onshell':None,                       
                      'loop_line':False}

        self.myleg = base_objects.Leg(self.mydict)

    def test_setget_leg_correct(self):
        "Test correct Leg object __init__, get and set"

        myleg2 = base_objects.Leg()

        for prop in self.mydict.keys():
            myleg2.set(prop, self.mydict[prop])

        self.assertEqual(self.myleg, myleg2)

        for prop in self.myleg.keys():
            self.assertEqual(self.myleg.get(prop), self.mydict[prop])

    def test_setget_leg_exceptions(self):
        "Test error raising in Leg __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.Leg.PhysicsObjectError,
                          base_objects.Leg,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.Leg,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.myleg.get,
                          a_number)
        self.assertRaises(base_objects.Leg.PhysicsObjectError,
                          self.myleg.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.myleg.set,
                          a_number, 0)
        self.assertRaises(base_objects.Leg.PhysicsObjectError,
                          self.myleg.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for leg properties"""

        test_values = [
                       {'prop':'id',
                        'right_list':[0, 3],
                        'wrong_list':['', 0.0]},
                       {'prop':'number',
                        'right_list':[1, 2, 3, 4, 5],
                        'wrong_list':['a', {}]},
                       {'prop':'state',
                        'right_list':[False, True],
                        'wrong_list':[1, 'wrong']}
                       ]

        temp_leg = self.myleg

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_leg.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_leg.set(test['prop'], x))

    def test_representation(self):
        """Test leg object string representation."""

        goal = "{\n"
        goal = goal + "    \'id\': 3,\n"
        goal = goal + "    \'number\': 5,\n"
        goal = goal + "    \'state\': True,\n"
        goal = goal + "    \'from_group\': False,\n" 
        goal = goal + "    \'loop_line\': False,\n"
        goal = goal + "    \'onshell\': None\n}"

        self.assertEqual(goal, str(self.myleg))

    def test_leg_list(self):
        """Test leg list initialization and counting functions
        for legs with 'from_group' = True"""

        mylist = [copy.copy(self.myleg) for dummy in range(1, 4) ]
        myleglist = base_objects.LegList(mylist)

        not_a_leg = 1

        for leg in myleglist:
            self.assertEqual(leg, self.myleg)

        self.assertRaises(AssertionError,
                          myleglist.append,
                          not_a_leg)

        # Test counting functions for number of from_group elements
        # that are True
        self.assertFalse(myleglist.minimum_one_from_group())
        myleglist[0].set('from_group', True)
        self.assertTrue(myleglist.minimum_one_from_group())
        self.assertFalse(myleglist.minimum_two_from_group())
        myleglist[1].set('from_group', True)
        self.assertTrue(myleglist.minimum_two_from_group())

        # Test can_combine_to_1
        ref_dict_to1 = {}
        self.assertFalse(myleglist.can_combine_to_1(ref_dict_to1))
        ref_dict_to1 = {(3, 3, 3):[3]}
        self.assertTrue(myleglist.can_combine_to_1(ref_dict_to1))
        myleglist[0].set('from_group', False)
        myleglist[1].set('from_group', False)
        self.assertFalse(myleglist.can_combine_to_1(ref_dict_to1))

        # Test can_combine_to_0
        ref_dict_to0 = {}
        myleglist[0].set('from_group', True)
        myleglist[1].set('from_group', True)
        myleglist[2].set('from_group', True)
        self.assertFalse(myleglist.can_combine_to_0(ref_dict_to0))
        ref_dict_to0 = {(3, 3, 3):0}
        self.assertTrue(myleglist.can_combine_to_0(ref_dict_to0))
        myleglist[0].set('from_group', False)
        myleglist[1].set('from_group', False)
        self.assertFalse(myleglist.can_combine_to_0(ref_dict_to0))
        myleglist[0].set('from_group', True)
        self.assertTrue(myleglist.can_combine_to_0(ref_dict_to0))

#===============================================================================
# MultiLegTest
#===============================================================================
class MultiLegTest(unittest.TestCase):
    """Test class for the MultiLeg object"""

    mydict = {}
    my_multi_leg = None

    def setUp(self):

        self.mydict = {'ids':[3, 2, 5],
                      'state':True}

        self.my_multi_leg = base_objects.MultiLeg(self.mydict)

    def test_setget_multi_leg_correct(self):
        "Test correct MultiLeg object __init__, get and set"

        my_multi_leg2 = base_objects.MultiLeg()

        for prop in self.mydict.keys():
            my_multi_leg2.set(prop, self.mydict[prop])

        self.assertEqual(self.my_multi_leg, my_multi_leg2)

        for prop in self.my_multi_leg.keys():
            self.assertEqual(self.my_multi_leg.get(prop), self.mydict[prop])

    def test_setget_multi_leg_exceptions(self):
        "Test error raising in MultiLeg __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.MultiLeg.PhysicsObjectError,
                          base_objects.MultiLeg,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.MultiLeg,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.my_multi_leg.get,
                          a_number)
        self.assertRaises(base_objects.MultiLeg.PhysicsObjectError,
                          self.my_multi_leg.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.my_multi_leg.set,
                          a_number, 0)
        self.assertRaises(base_objects.MultiLeg.PhysicsObjectError,
                          self.my_multi_leg.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for multi_leg properties"""

        test_values = [
                       {'prop':'ids',
                        'right_list':[[0], [3, 4, 5]],
                        'wrong_list':['', 1, 0.0]},
                       {'prop':'state',
                        'right_list':[False, True],
                        'wrong_list':[0, 'wrong']}
                       ]

        temp_multi_leg = self.my_multi_leg

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_multi_leg.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_multi_leg.set(test['prop'], x))

    def test_representation(self):
        """Test multi_leg object string representation."""

        goal = "{\n"
        goal = goal + "    \'ids\': [3, 2, 5],\n"
        goal = goal + "    \'state\': True\n}"

        self.assertEqual(goal, str(self.my_multi_leg))

    def test_multi_leg_list(self):
        """Test multi_leg list initialization and counting functions
        for multi_legs with 'from_group' = True"""

        mylist = [copy.copy(self.my_multi_leg) for dummy in range(1, 4) ]
        my_multi_leglist = base_objects.MultiLegList(mylist)

        not_a_multi_leg = 1

        for multi_leg in my_multi_leglist:
            self.assertEqual(multi_leg, self.my_multi_leg)

        self.assertRaises(AssertionError,
                          my_multi_leglist.append,
                          not_a_multi_leg)

#===============================================================================
# VertexTest
#===============================================================================
class VertexTest(unittest.TestCase):
    """Test class for the Vertex object"""

    mydict = {}
    myvertex = None
    myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':True,
                                      'from_group':False})] * 10)

    def setUp(self):

        self.mydict = {'id':3,
                      'legs':self.myleglist}

        self.myvertex = base_objects.Vertex(self.mydict)

    def test_setget_vertex_correct(self):
        "Test correct Vertex object __init__, get and set"

        myvertex2 = base_objects.Vertex()

        for prop in self.mydict.keys():
            myvertex2.set(prop, self.mydict[prop])

        self.assertEqual(self.myvertex, myvertex2)

        for prop in self.myvertex.keys():
            self.assertEqual(self.myvertex.get(prop), self.mydict[prop])

    def test_setget_vertex_exceptions(self):
        "Test error raising in Vertex __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.Vertex.PhysicsObjectError,
                          base_objects.Vertex,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.Vertex,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.myvertex.get,
                          a_number)
        self.assertRaises(base_objects.Vertex.PhysicsObjectError,
                          self.myvertex.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.myvertex.set,
                          a_number, 0)
        self.assertRaises(base_objects.Vertex.PhysicsObjectError,
                          self.myvertex.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for vertex properties"""

        test_values = [
                       {'prop':'id',
                        'right_list':[0, 3],
                        'wrong_list':['', 0.0]},
                       {'prop':'legs',
                        'right_list':[self.myleglist],
                        'wrong_list':['a', {}]}
                       ]

        temp_vertex = self.myvertex

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_vertex.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_vertex.set(test['prop'], x))

    def test_representation(self):
        """Test vertex object string representation."""

        goal = "{\n"
        goal = goal + "    \'id\': 3,\n"
        goal = goal + "    \'legs\': %s\n}" % repr(self.myleglist)

        self.assertEqual(goal, str(self.myvertex))

    def test_vertex_list(self):
        """Test vertex list initialization"""

        mylist = [self.myvertex] * 10
        myvertexlist = base_objects.VertexList(mylist)

        not_a_vertex = 1

        for vertex in myvertexlist:
            self.assertEqual(vertex, self.myvertex)

        self.assertRaises(AssertionError,
                          myvertexlist.append,
                          not_a_vertex)

#===============================================================================
# DiagramTest
#===============================================================================
class DiagramTest(unittest.TestCase):
    """Test class for the Diagram object"""

    mydict = {}
    mydiagram = None
    myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':True,
                                      'from_group':False})] * 10)
    myvertexlist = base_objects.VertexList([base_objects.Vertex({'id':3,
                                      'legs':myleglist})] * 10)

    def setUp(self):

        self.mydict = {'vertices':self.myvertexlist,
                       'orders':{}}

        self.mydiagram = base_objects.Diagram(self.mydict)

    def test_setget_diagram_correct(self):
        "Test correct Diagram object __init__, get and set"

        mydiagram2 = base_objects.Diagram()

        for prop in self.mydict.keys():
            mydiagram2.set(prop, self.mydict[prop])

        self.assertEqual(self.mydiagram, mydiagram2)

        for prop in self.mydiagram.keys():
            self.assertEqual(self.mydiagram.get(prop), self.mydict[prop])

    def test_setget_diagram_exceptions(self):
        "Test error raising in Diagram __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.Diagram.PhysicsObjectError,
                          base_objects.Diagram,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.Diagram,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.mydiagram.get,
                          a_number)
        self.assertRaises(base_objects.Diagram.PhysicsObjectError,
                          self.mydiagram.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.mydiagram.set,
                          a_number, 0)
        self.assertRaises(base_objects.Diagram.PhysicsObjectError,
                          self.mydiagram.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for diagram properties"""

        test_values = [{'prop':'vertices',
                        'right_list':[self.myvertexlist],
                        'wrong_list':['a', {}]}
                       ]

        temp_diagram = self.mydiagram

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_diagram.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_diagram.set(test['prop'], x))

    def test_representation(self):
        """Test diagram object string representation."""

        goal = "{\n"
        goal = goal + "    \'vertices\': %s,\n" % repr(self.myvertexlist)
        goal = goal + "    'orders': {}\n}"
        
        self.assertEqual(goal, str(self.mydiagram))

    def test_diagram_list(self):
        """Test Diagram list initialization"""

        mylist = [self.mydiagram] * 10
        mydiagramlist = base_objects.DiagramList(mylist)

        not_a_diagram = 1

        for diagram in mydiagramlist:
            self.assertEqual(diagram, self.mydiagram)

        self.assertRaises(AssertionError,
                          mydiagramlist.append,
                          not_a_diagram)

    def test_diagram_list_nice_string(self):
        """Test Diagram and Diagram list nice_string representation"""

        mylist = [self.mydiagram] * 10
        mydiagramlist = base_objects.DiagramList(mylist)

        self.assertRaises(Exception,
                           mydiagramlist.nice_string)
        return

#===============================================================================
# ProcessTest
#===============================================================================
class ProcessTest(unittest.TestCase):
    """Test class for the Process object"""

    mydict = {}
    myprocess = None
    myleglist = base_objects.LegList()

    mymodel = base_objects.Model()

    def setUp(self):

        mypartlist = base_objects.ParticleList([
                     base_objects.Particle({'name':'c',
                                             'antiname':'c~',
                                             'pdg_code':3})])

        self.mymodel.set('particles', mypartlist)

        self.myleglist = base_objects.LegList(\
            [copy.copy(base_objects.Leg({'id':3,
                                         'number':5,
                                         'state':True,
                                         'from_group':False})) for \
             dummy in range(5)])

        self.myleglist[0].set('state', False)
        self.myleglist[1].set('state', False)

        self.mydict = {'legs':self.myleglist,
                       'orders':{'QCD':5, 'QED':1},
                       'model':self.mymodel,
                       'id': 1,
                       'uid':0,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_onsh_s_channels':[],
                       'forbidden_particles':[],
                       'perturbation_couplings':[],
                       'is_decay_chain': False,
                       'decay_chains': base_objects.ProcessList(),
                       'legs_with_decays': self.myleglist,
                       'squared_orders': {},
                       'constrained_orders':{},
                       'sqorders_types': {},
                       'has_born': True,
                       'overall_orders': {},
                       'NLO_mode':'tree',
                       'split_orders':[]}

        self.myprocess = base_objects.Process(self.mydict)

    def test_setget_process_correct(self):
        "Test correct Process object __init__, get and set"

        myprocess2 = base_objects.Process()

        for prop in self.mydict.keys():
            myprocess2.set(prop, self.mydict[prop])

        self.assertEqual(self.myprocess, myprocess2)

        for prop in self.myprocess.keys():
            self.assertEqual(self.myprocess.get(prop), self.mydict[prop])

    def test_setget_process_exceptions(self):
        "Test error raising in Process __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.Process.PhysicsObjectError,
                          base_objects.Process,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.Process,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.myprocess.get,
                          a_number)
        self.assertRaises(base_objects.Process.PhysicsObjectError,
                          self.myprocess.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.myprocess.set,
                          a_number, 0)
        self.assertRaises(base_objects.Process.PhysicsObjectError,
                          self.myprocess.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for process properties"""

        test_values = [{'prop':'legs',
                        'right_list':[self.myleglist],
                        'wrong_list':['a', {}]}
                       ]

        temp_process = self.myprocess

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_process.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_process.set(test['prop'], x))

    def test_representation(self):
        """Test process object string representation."""

        goal = "{\n"
        goal = goal + "    \'legs\': %s,\n" % repr(self.myleglist)
        goal = goal + "    \'orders\': %s,\n" % repr(self.myprocess['orders'])
        goal = goal + "    \'overall_orders\': %s,\n" % \
               repr(self.myprocess['overall_orders'])
        goal = goal + "    \'squared_orders\': %s,\n" % repr(self.myprocess['squared_orders'])
        goal = goal + "    \'constrained_orders\': %s,\n" % repr(self.myprocess['constrained_orders'])
        goal = goal + "    \'model\': %s,\n" % repr(self.myprocess['model'])
        goal = goal + "    \'id\': 1,\n"
        goal = goal + "    \'required_s_channels\': [],\n"
        goal = goal + "    \'forbidden_onsh_s_channels\': [],\n"
        goal = goal + "    \'forbidden_s_channels\': [],\n"
        goal = goal + "    \'forbidden_particles\': [],\n"
        goal = goal + "    \'is_decay_chain\': False,\n"
        goal = goal + "    \'decay_chains\': [],\n"
        goal = goal + "    \'legs_with_decays\': %s,\n" % repr(self.myleglist)
        goal = goal + "    \'perturbation_couplings\': [],\n"
        goal = goal + "    \'has_born\': True,\n"
        goal = goal + "    \'NLO_mode\': 'tree',\n"
        goal = goal + "    \'split_orders\': []\n}"

        for a, b in zip(goal.split('\n'), str(self.myprocess).split('\n')):
            self.assertEqual(a,b)
        self.assertEqual(goal, str(self.myprocess))

    def test_nice_string(self):
        """Test Process nice_string representation"""

        goal_str = "Process: c c > c c c QED<=1 QCD<=5 @1"

        self.assertEqual(goal_str, self.myprocess.nice_string())

    def test_input_string(self):
        """Test Process nice_string representation"""

        goal_str = "c c > c c c QED=1 QCD=5, (c > c c c c, c > c c c c)"

        decay = copy.copy(self.myprocess)
        decay.set('legs', copy.deepcopy(decay.get('legs')))
        decay.get('legs')[1].set('state', True)
        decay.set('is_decay_chain', True)
        decay.set('orders', {})
        decay2 = copy.copy(decay)
        self.myprocess.set('decay_chains', base_objects.ProcessList([decay]))
        decay.set('decay_chains', base_objects.ProcessList([decay2]))

        self.assertEqual(goal_str, self.myprocess.input_string())

    def test_shell_string(self):
        """Test Process shell_string representation"""

        self.myprocess.get('legs')[2].set('id', -3)
        self.myprocess.set('id', 2)
        goal_str = "2_cc_cxcc"

        self.assertEqual(goal_str, self.myprocess.shell_string())
    
    def test_long_shell_string(self):
        """Test Process nice_string representation"""

        goal_str = '1_cc_ccc_c_cccc_c_cccc_c_cccc_c_cccc_c_cccc_c_cccc_4'

        decay = copy.copy(self.myprocess)
        decay.set('legs', copy.deepcopy(decay.get('legs')))
        decay.get('legs')[1].set('state', True)
        decay.set('is_decay_chain', True)
        decay.set('orders', {})
        decay.set('forbidden_particles',[3])
        decay2 = copy.copy(decay)
        decay3 = copy.copy(decay)
        decay4 = copy.copy(decay)
        decay5 = copy.copy(decay)
        decay6 = copy.copy(decay)
        self.myprocess.set('uid',4)
        self.myprocess.set('decay_chains', base_objects.ProcessList([decay]))
        decay.set('decay_chains', base_objects.ProcessList([decay2]))
        decay2.set('decay_chains', base_objects.ProcessList([decay3]))
        decay3.set('decay_chains', base_objects.ProcessList([decay4]))
        decay4.set('decay_chains', base_objects.ProcessList([decay5]))
        decay5.set('decay_chains', base_objects.ProcessList([decay6]))
        self.assertTrue(len(self.myprocess.shell_string()) < 70)
        self.assertEqual(goal_str, self.myprocess.shell_string())
        
    def test_get_final_ids_after_decay(self):
        """check that we get the correct ids and in the correct order"""
        
        mymodel = base_objects.Model()
        mypartlist = base_objects.ParticleList([
                     base_objects.Particle({'name':'c',
                                             'antiname':'c~',
                                             'pdg_code':3}),
                     base_objects.Particle({'name':'l',
                                             'antiname':'l~',
                                             'pdg_code':11}),
                     base_objects.Particle({'name':'H',
                                             'antiname':'H',
                                             'pdg_code':25}),                                                                                                
                                                ])
        
        mymodel.set('particles', mypartlist)

        # Check for c c~ > h c, h > l l~

        myleglist = base_objects.LegList(\
            [base_objects.Leg({'id':3,
                                         'number':1,
                                         'state':False,
                                         'from_group':False}),
             base_objects.Leg({'id':3,
                                         'number':2,
                                         'state':False,
                                         'from_group':False}),
             base_objects.Leg({'id':25,
                                         'number':3,
                                         'state':True,
                                         'from_group':False}),
             base_objects.Leg({'id':3,
                                         'number':4,
                                         'state':True,
                                         'from_group':False})])

        mylegdecay = base_objects.LegList(\
            [base_objects.Leg({'id':25,
                                         'number':1,
                                         'state':False,
                                         'from_group':False}),
             base_objects.Leg({'id':11,
                                         'number':2,
                                         'state':True,
                                         'from_group':False}),
             base_objects.Leg({'id':-11,
                                         'number':3,
                                         'state':True,
                                         'from_group':False})])
        
        mydecay = {'legs':mylegdecay,
                  'orders':{'QCD':5, 'QED':1},
                  'model':mymodel,
                  'id': 1,
                  'uid':0,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_onsh_s_channels':[],
                       'forbidden_particles':[],
                       'perturbation_couplings':[],
                       'is_decay_chain': False,
                       'decay_chains': base_objects.ProcessList(),
                       'legs_with_decays': [],
                       'squared_orders': {},
                       'has_born': True,
                       'overall_orders': {},
                       'NLO_mode':'tree'}
            

        mydecay = base_objects.Process(mydecay)
        
        myprocess = copy.copy(mydecay)
        myprocess['legs'] = myleglist
        myprocess['is_decay_chain'] = True
        proclist = base_objects.ProcessList()
        proclist.append(mydecay)
        myprocess['decay_chains'] = proclist
        
        # checking
        output = myprocess.get_final_ids_after_decay()
        self.assertEqual(output, [11, -11, 3])
        
        ## c c~ > c h c~ h c, h > l l, h > l~ l~ 
        myleglist = base_objects.LegList(\
            [base_objects.Leg({'id':3,
                                         'number':1,
                                         'state':False,
                                         'from_group':False}),
             base_objects.Leg({'id':-3,
                                         'number':2,
                                         'state':False,
                                         'from_group':False}),
             base_objects.Leg({'id':3,
                                         'number':3,
                                         'state':True,
                                         'from_group':False}),
             base_objects.Leg({'id':25,
                                         'number':4,
                                         'state':True,
                                         'from_group':False}),
            base_objects.Leg({'id':-3,
                                         'number':3,
                                         'state':True,
                                         'from_group':False}),
             base_objects.Leg({'id':25,
                                         'number':4,
                                         'state':True,
                                         'from_group':False}),
              base_objects.Leg({'id':3,
                                         'number':3,
                                         'state':True,
                                         'from_group':False})])

        mylegdecay = base_objects.LegList(\
            [base_objects.Leg({'id':25,
                                         'number':1,
                                         'state':False,
                                         'from_group':False}),
             base_objects.Leg({'id':11,
                                         'number':2,
                                         'state':True,
                                         'from_group':False}),
             base_objects.Leg({'id':11,
                                         'number':3,
                                         'state':True,
                                         'from_group':False})])        
        mydecay['legs'] = mylegdecay
        
        mylegdecay2 = base_objects.LegList(\
            [base_objects.Leg({'id':25,
                                         'number':1,
                                         'state':False,
                                         'from_group':False}),
             base_objects.Leg({'id':-11,
                                         'number':2,
                                         'state':True,
                                         'from_group':False}),
             base_objects.Leg({'id':-11,
                                         'number':3,
                                         'state':True,
                                         'from_group':False})]) 
        mydecay2 = copy.copy(mydecay)
        mydecay2['legs'] = mylegdecay2
        
        
        myprocess['legs'] = myleglist
        myprocess['is_decay_chain'] = True
        proclist = base_objects.ProcessList()
        proclist.append(mydecay)
        proclist.append(mydecay2)
        myprocess['decay_chains'] = proclist       
        # checking
        output = myprocess.get_final_ids_after_decay()
        self.assertEqual(output, [3, 11, 11, -3,-11,-11,3])
        
#===============================================================================
# ProcessDefinitionTest
#===============================================================================
class ProcessDefinitionTest(unittest.TestCase):
    """Test class for the ProcessDefinition object"""

    mydict = {}
    my_process_definition = None
    mymodel = base_objects.Model()
    my_multi_leglist = base_objects.MultiLegList()

    def setUp(self):

        mypartlist = base_objects.ParticleList([
                     base_objects.Particle({'name':'c',
                                             'antiname':'c~',
                                             'pdg_code':3})])

        self.mymodel.set('particles', mypartlist)

        self.my_multi_leglist = base_objects.MultiLegList(\
            [copy.copy(base_objects.MultiLeg({'ids':[3, 4, 5],
                                              'state':True})) for \
             dummy in range(5)])

        self.my_multi_leglist[0].set('state', False)
        self.my_multi_leglist[1].set('state', False)

        self.mydict = {'legs':self.my_multi_leglist,
                       'orders':{'QCD':5, 'QED':1},
                       'model':self.mymodel,
                       'id':3,
                       'uid':0,
                       'required_s_channels':[],
                       'forbidden_s_channels':[],
                       'forbidden_onsh_s_channels':[],
                       'forbidden_particles':[],
                       'perturbation_couplings':[],
                       'is_decay_chain': False,
                       'decay_chains': base_objects.ProcessList(),
                       'squared_orders':{},
                       'constrained_orders':{},
                       'has_born': True,
                       'overall_orders':{},
                       'sqorders_types':{},
                       'NLO_mode':'tree',
                       'split_orders':[]}

        self.my_process_definition = base_objects.ProcessDefinition(self.mydict)

    def test_setget_process_definition_correct(self):
        "Test correct ProcessDefinition object __init__, get and set"

        my_process_definition2 = base_objects.ProcessDefinition()

        for prop in self.mydict.keys():
            my_process_definition2.set(prop, self.mydict[prop])

        self.assertEqual(self.my_process_definition, my_process_definition2)

        for prop in self.my_process_definition.keys():
            self.assertEqual(self.my_process_definition.get(prop), self.mydict[prop])

    def test_setget_process_definition_exceptions(self):
        "Test error raising in ProcessDefinition __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(base_objects.ProcessDefinition.PhysicsObjectError,
                          base_objects.ProcessDefinition,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          base_objects.ProcessDefinition,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          self.my_process_definition.get,
                          a_number)
        self.assertRaises(base_objects.ProcessDefinition.PhysicsObjectError,
                          self.my_process_definition.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          self.my_process_definition.set,
                          a_number, 0)
        self.assertRaises(base_objects.ProcessDefinition.PhysicsObjectError,
                          self.my_process_definition.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for process properties"""

        test_values = [{'prop':'legs',
                        'right_list':[self.my_multi_leglist],
                        'wrong_list':['a', {}]}
                       ]

        temp_process = self.my_process_definition

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_process.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_process.set(test['prop'], x))

    def test_representation(self):
        """Test process object string representation."""

        goal = "{\n"
        goal = goal + "    \'legs\': %s,\n" % repr(self.my_multi_leglist)
        goal = goal + "    \'orders\': %s,\n" % repr(self.my_process_definition['orders'])
        goal = goal + "    \'overall_orders\': %s,\n" % repr(self.my_process_definition['overall_orders'])
        goal = goal + "    \'squared_orders\': %s,\n" % repr(self.my_process_definition['squared_orders'])
        goal = goal + "    \'constrained_orders\': %s,\n" % repr(self.my_process_definition['constrained_orders'])
        goal = goal + "    \'model\': %s,\n" % repr(self.my_process_definition['model'])
        goal = goal + "    \'id\': %s,\n" % repr(self.my_process_definition['id'])
        goal = goal + "    \'required_s_channels\': [],\n"
        goal = goal + "    \'forbidden_onsh_s_channels\': [],\n"
        goal = goal + "    \'forbidden_s_channels\': [],\n"
        goal = goal + "    \'forbidden_particles\': [],\n"
        goal = goal + "    \'is_decay_chain\': False,\n"
        goal = goal + "    \'decay_chains\': [],\n"
        goal = goal + "    \'perturbation_couplings\': [],\n"
        goal = goal + "    \'has_born\': True,\n"
        goal = goal + "    \'NLO_mode\': 'tree',\n"
        goal = goal + "    \'split_orders\': []\n}"                
        self.assertEqual(goal, str(self.my_process_definition))

#===============================================================================
# HelperTest
#===============================================================================
class HelperTest(unittest.TestCase):
    """Test class for helper functions"""


    def test_make_unique(self):
        """Test the make_unique function"""

        doubletlist = [4, 6, 2, 4, 6, 2, 2, 2]
        base_objects.make_unique(doubletlist)
        self.assertEqual(doubletlist,
                         [4, 6, 2])
