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
                      'texname':'t',
                      'antitexname':'\\overline{t}',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
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
        self.assertRaises(base_objects.Particle.PhysicsObjectError,
                          base_objects.Particle,
                          a_number)

        # Test get
        self.assertRaises(base_objects.Particle.PhysicsObjectError,
                          self.mypart.get,
                          a_number)
        self.assertRaises(base_objects.Particle.PhysicsObjectError,
                          self.mypart.get,
                          'wrongparam')

        # Test set
        self.assertRaises(base_objects.Particle.PhysicsObjectError,
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
                        'wrong_list':['', 'x ', 'e?', '{}', '9x', 'd~3', 'd+g',
                                      'u~+', 'u~~']},
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
                        'right_list':[1, 12, 8000000],
                        'wrong_list':[-1, 'a']},
                       {'prop':'line',
                        'right_list':['straight', 'wavy', 'curly', 'dashed'],
                        'wrong_list':[-1, 'wrong']},
                       {'prop':'charge',
                        'right_list':[1., -1., -2. / 3., 0.],
                        'wrong_list':[1, 'a']},
                       {'prop':'propagating',
                        'right_list':[True, False],
                        'wrong_list':[1, 'a', 'true', None]},
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
                self.assertFalse(temp_part.set(test['prop'], x))

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
        goal = goal + "    \'texname\': \'t\',\n"
        goal = goal + "    \'antitexname\': \'\\overline{t}\',\n"
        goal = goal + "    \'line\': \'straight\',\n"
        goal = goal + "    \'propagating\': True,\n"
        goal = goal + "    \'is_part\': True,\n"
        goal = goal + "    \'self_antipart\': False\n}"

        self.assertEqual(goal, str(self.mypart))

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

    def test_particle_list(self):
        """Test particle list initialization and search"""

        mylist = [self.mypart] * 10
        mypartlist = base_objects.ParticleList(mylist)

        not_a_part = 1

        for part in mypartlist:
            self.assertEqual(part, self.mypart)

        self.assertRaises(base_objects.ParticleList.PhysicsObjectListError,
                          mypartlist.append,
                          not_a_part)
        # test particle search
        self.assertEqual(self.mypart,
                         mypartlist.find_name(self.mypart['name']))
        anti_part = copy.copy(self.mypart)
        anti_part.set('is_part', False)
        self.assertEqual(anti_part,
                         mypartlist.find_name(self.mypart['antiname']))
        self.assertEqual(None,
                         mypartlist.find_name('none'))

        mydict={6:self.mypart}
        self.assertEqual(mydict,mypartlist.generate_dict())
        

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
                       'color': ['C1', 'C2'],
                       'lorentz':['L1', 'L2'],
                       'couplings':{(0, 0):'g00',
                                    (0, 1):'g01',
                                    (1, 0):'g10',
                                    (1, 1):'g11'},
                       'orders':{'QCD':1, 'QED':1}}

        self.myinter = base_objects.Interaction(self.mydict)

    def test_setget_interaction_correct(self):
        "Test correct interaction object __init__, get and set"

        myinter2 = base_objects.Interaction()

        # First fill myinter2 it using set
        for prop in ['id', 'particles', 'color', 'lorentz', 'couplings', 'orders']:
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
        self.assertRaises(base_objects.Interaction.PhysicsObjectError,
                          base_objects.Interaction,
                          a_number)

        # Test get
        self.assertRaises(base_objects.Interaction.PhysicsObjectError,
                          self.myinter.get,
                          a_number)
        self.assertRaises(base_objects.Interaction.PhysicsObjectError,
                          self.myinter.get,
                          'wrongparam')

        # Test set
        self.assertRaises(base_objects.Interaction.PhysicsObjectError,
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
                        'right_list':[[], ['C1'], ['C1', 'C2']],
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
                                       (1, 0):'g10', (1, 2):'g11'},
                                      {(0, 0):'g00', (0, 1):'g01',
                                       (1, 0):'g10', (1, 2):'g11'},
                                      {(0, 0):'g00', (0, 1):'g01',
                                       (1, 0):'g10'}]}
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
        goal = goal + "    \'particles\': %s,\n" % \
                            repr(base_objects.ParticleList([self.mypart] * 4))
        goal = goal + "    \'color\': [\'C1\', \'C2\'],\n"
        goal = goal + "    \'lorentz\': [\'L1\', \'L2\'],\n"
        goal = goal + "    \'couplings\': %s,\n" % \
                                    repr(self.myinter['couplings'])
        goal = goal + "    \'orders\': %s\n}" % repr(self.myinter['orders'])

        self.assertEqual(goal, str(self.myinter))

    def test_generating_dict(self):
        """Test the dictionary generation routine"""

        # Create a non trivial interaction
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

        goal_ref_dict_to0 = { (1, -2, 3, 4):0,
                          (1, -2, 4, 3):0,
                          (1, 3, -2, 4):0,
                          (1, 3, 4, -2):0,
                          (1, 4, -2, 3):0,
                          (1, 4, 3, -2):0,
                          (-2, 1, 3, 4):0,
                          (-2, 1, 4, 3):0,
                          (-2, 3, 1, 4):0,
                          (-2, 3, 4, 1):0,
                          (-2, 4, 1, 3):0,
                          (-2, 4, 3, 1):0,
                          (3, 1, -2, 4):0,
                          (3, 1, 4, -2):0,
                          (3, -2, 1, 4):0,
                          (3, -2, 4, 1):0,
                          (3, 4, 1, -2):0,
                          (3, 4, -2, 1):0,
                          (4, 1, -2, 3):0,
                          (4, 1, 3, -2):0,
                          (4, -2, 1, 3):0,
                          (4, -2, 3, 1):0,
                          (4, 3, 1, -2):0,
                          (4, 3, -2, 1):0,
                          (-1, 2, -3, 4):0,
                          (-1, 2, 4, -3):0,
                          (-1, -3, 2, 4):0,
                          (-1, -3, 4, 2):0,
                          (-1, 4, 2, -3):0,
                          (-1, 4, -3, 2):0,
                          (2, -1, -3, 4):0,
                          (2, -1, 4, -3):0,
                          (2, -3, -1, 4):0,
                          (2, -3, 4, -1):0,
                          (2, 4, -1, -3):0,
                          (2, 4, -3, -1):0,
                          (-3, -1, 2, 4):0,
                          (-3, -1, 4, 2):0,
                          (-3, 2, -1, 4):0,
                          (-3, 2, 4, -1):0,
                          (-3, 4, -1, 2):0,
                          (-3, 4, 2, -1):0,
                          (4, -1, 2, -3):0,
                          (4, -1, -3, 2):0,
                          (4, 2, -1, -3):0,
                          (4, 2, -3, -1):0,
                          (4, -3, -1, 2):0,
                          (4, -3, 2, -1):0}

        goal_ref_dict_to1 = {(-2, 3, 4):[-1],
                            (-2, 4, 3):[-1],
                            (3, -2, 4):[-1],
                            (3, 4, -2):[-1],
                            (4, -2, 3):[-1],
                            (4, 3, -2):[-1],
                            (1, 3, 4):[2],
                            (1, 4, 3):[2],
                            (3, 1, 4):[2],
                            (3, 4, 1):[2],
                            (4, 1, 3):[2],
                            (4, 3, 1):[2],
                            (1, -2, 4):[-3],
                            (1, 4, -2):[-3],
                            (-2, 1, 4):[-3],
                            (-2, 4, 1):[-3],
                            (4, 1, -2):[-3],
                            (4, -2, 1):[-3],
                            (1, -2, 3):[4],
                            (1, 3, -2):[4],
                            (-2, 1, 3):[4],
                            (-2, 3, 1):[4],
                            (3, 1, -2):[4],
                            (3, -2, 1):[4],
                            (2, -3, 4):[1],
                            (2, 4, -3):[1],
                            (-3, 2, 4):[1],
                            (-3, 4, 2):[1],
                            (4, 2, -3):[1],
                            (4, -3, 2):[1],
                            (-1, -3, 4):[-2],
                            (-1, 4, -3):[-2],
                            (-3, -1, 4):[-2],
                            (-3, 4, -1):[-2],
                            (4, -1, -3):[-2],
                            (4, -3, -1):[-2],
                            (-1, 2, 4):[3],
                            (-1, 4, 2):[3],
                            (2, -1, 4):[3],
                            (2, 4, -1):[3],
                            (4, -1, 2):[3],
                            (4, 2, -1):[3],
                            (-1, 2, -3):[4],
                            (-1, -3, 2):[4],
                            (2, -1, -3):[4],
                            (2, -3, -1):[4],
                            (-3, -1, 2):[4],
                            (-3, 2, -1):[4]}

        self.assertEqual(ref_dict_to0, goal_ref_dict_to0)
        self.assertEqual(ref_dict_to1, goal_ref_dict_to1)

        myinterlist = base_objects.InteractionList([myinter] * 10)

        add_inter = base_objects.Interaction()
        add_inter.set('particles', base_objects.ParticleList([part1,
                                                              part2,
                                                              part3]))
        myinterlist.append(add_inter)

        goal_ref_dict_to0[(1, -2, 3)] = 0
        goal_ref_dict_to0[(1, 3, -2)] = 0
        goal_ref_dict_to0[(-2, 1, 3)] = 0
        goal_ref_dict_to0[(-2, 3, 1)] = 0
        goal_ref_dict_to0[(3, 1, -2)] = 0
        goal_ref_dict_to0[(3, -2, 1)] = 0
        goal_ref_dict_to0[(-1, 2, -3)] = 0
        goal_ref_dict_to0[(-1, -3, 2)] = 0
        goal_ref_dict_to0[(2, -1, -3)] = 0
        goal_ref_dict_to0[(2, -3, -1)] = 0
        goal_ref_dict_to0[(-3, -1, 2)] = 0
        goal_ref_dict_to0[(-3, 2, -1)] = 0

        goal_ref_dict_to1[(1, -2)] = [-3]
        goal_ref_dict_to1[(1, 3)] = [2]
        goal_ref_dict_to1[(-2, 1)] = [-3]
        goal_ref_dict_to1[(-2, 3)] = [-1]
        goal_ref_dict_to1[(3, 1)] = [2]
        goal_ref_dict_to1[(3, -2)] = [-1]
        goal_ref_dict_to1[(-1, 2)] = [3]
        goal_ref_dict_to1[(-1, -3)] = [-2]
        goal_ref_dict_to1[(2, -1)] = [3]
        goal_ref_dict_to1[(2, -3)] = [1]
        goal_ref_dict_to1[(-3, -1)] = [-2]
        goal_ref_dict_to1[(-3, 2)] = [1]

        self.assertEqual(myinterlist.generate_ref_dict()[0], goal_ref_dict_to0)
        self.assertEqual(myinterlist.generate_ref_dict()[1], goal_ref_dict_to1)

    def test_interaction_list(self):
        """Test interaction list initialization"""

        mylist = [copy.copy(inter) for inter in [self.myinter] * 3]

        for i in range(1,4):
            mylist[i-1].set('id',i)
            
        myinterlist = base_objects.InteractionList(mylist)

        not_a_inter = 1

        self.assertRaises(base_objects.InteractionList.PhysicsObjectListError,
                          myinterlist.append,
                          not_a_inter)
        mydict = {}
        
        for i in range(1,4):
            mydict[i]=myinterlist[i-1]

        self.assertEqual(mydict,myinterlist.generate_dict())


#===============================================================================
# ModelTest
#===============================================================================
class ModelTest(unittest.TestCase):
    """Test class for the Model object"""

    def test_model_initialization(self):
        """Test the default Model class initialization"""
        mymodel = base_objects.Model()

        self.assertEqual(mymodel['particles'], base_objects.ParticleList())

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
        self.assertRaises(base_objects.Model.PhysicsObjectError,
                          mymodel.get,
                          not_a_string)
        self.assertRaises(base_objects.Model.PhysicsObjectError,
                          mymodel.get,
                          'wrong_key')
        self.assertRaises(base_objects.Model.PhysicsObjectError,
                          mymodel.set,
                          not_a_string, None)
        self.assertRaises(base_objects.Model.PhysicsObjectError,
                          mymodel.set,
                          'wrong_subclass', None)

        # For each subclass
        self.assertFalse(mymodel.set('particles', not_a_string))
        self.assertFalse(mymodel.set('interactions', not_a_string))

    def test_dictionaries(self):
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
        mypartdict = {6:mypart}
        self.assertEqual(mypartdict,mymodel.get('particle_dict'))
    

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
                      'state':'final',
                      'from_group':False}

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
        self.assertRaises(base_objects.Leg.PhysicsObjectError,
                          base_objects.Leg,
                          a_number)

        # Test get
        self.assertRaises(base_objects.Leg.PhysicsObjectError,
                          self.myleg.get,
                          a_number)
        self.assertRaises(base_objects.Leg.PhysicsObjectError,
                          self.myleg.get,
                          'wrongparam')

        # Test set
        self.assertRaises(base_objects.Leg.PhysicsObjectError,
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
                        'right_list':['initial', 'final'],
                        'wrong_list':[0, 'wrong']}
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
        goal = goal + "    \'state\': \'final\',\n"
        goal = goal + "    \'from_group\': False\n}"

        self.assertEqual(goal, str(self.myleg))

    def test_leg_list(self):
        """Test leg list initialization"""

        mylist = [copy.copy(self.myleg) for item in range(1, 4)]
        myleglist = base_objects.LegList(mylist)

        not_a_leg = 1

        for leg in myleglist:
            self.assertEqual(leg, self.myleg)

        self.assertRaises(base_objects.LegList.PhysicsObjectListError,
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

        # Test passesTo1
        ref_dict_to1 = {}
        self.assertFalse(myleglist.passesTo1(ref_dict_to1))
        ref_dict_to1 = {(3, 3, 3):[3]}
        self.assertEqual(myleglist.passesTo1(ref_dict_to1), [3])
        myleglist[0].set('from_group', False)
        myleglist[1].set('from_group', False)
        self.assertFalse(myleglist.passesTo1(ref_dict_to1))

#===============================================================================
# VertexTest
#===============================================================================
class VertexTest(unittest.TestCase):
    """Test class for the Vertex object"""

    mydict = {}
    myvertex = None
    myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':'final',
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
        self.assertRaises(base_objects.Vertex.PhysicsObjectError,
                          base_objects.Vertex,
                          a_number)

        # Test get
        self.assertRaises(base_objects.Vertex.PhysicsObjectError,
                          self.myvertex.get,
                          a_number)
        self.assertRaises(base_objects.Vertex.PhysicsObjectError,
                          self.myvertex.get,
                          'wrongparam')

        # Test set
        self.assertRaises(base_objects.Vertex.PhysicsObjectError,
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

        self.assertRaises(base_objects.VertexList.PhysicsObjectListError,
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
                                      'state':'final',
                                      'from_group':False})] * 10)
    myvertexlist = base_objects.VertexList([base_objects.Vertex({'id':3,
                                      'legs':myleglist})] * 10)

    def setUp(self):

        self.mydict = {'vertices':self.myvertexlist}

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
        self.assertRaises(base_objects.Diagram.PhysicsObjectError,
                          base_objects.Diagram,
                          a_number)

        # Test get
        self.assertRaises(base_objects.Diagram.PhysicsObjectError,
                          self.mydiagram.get,
                          a_number)
        self.assertRaises(base_objects.Diagram.PhysicsObjectError,
                          self.mydiagram.get,
                          'wrongparam')

        # Test set
        self.assertRaises(base_objects.Diagram.PhysicsObjectError,
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
        goal = goal + "    \'vertices\': %s\n}" % repr(self.myvertexlist)

        self.assertEqual(goal, str(self.mydiagram))

    def test_diagram_list(self):
        """Test Diagram list initialization"""

        mylist = [self.mydiagram] * 10
        mydiagramlist = base_objects.DiagramList(mylist)

        not_a_diagram = 1

        for diagram in mydiagramlist:
            self.assertEqual(diagram, self.mydiagram)

        self.assertRaises(base_objects.DiagramList.PhysicsObjectListError,
                          mydiagramlist.append,
                          not_a_diagram)

#===============================================================================
# ProcessTest
#===============================================================================
class ProcessTest(unittest.TestCase):
    """Test class for the Process object"""

    mydict = {}
    myprocess = None
    myleglist = base_objects.LegList([base_objects.Leg({'id':3,
                                      'number':5,
                                      'state':'final',
                                      'from_group':False})] * 5)

    def setUp(self):

        self.mydict = {'legs':self.myleglist,
                       'orders':{'QCD':5, 'QED':1}}

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
        self.assertRaises(base_objects.Process.PhysicsObjectError,
                          base_objects.Process,
                          a_number)

        # Test get
        self.assertRaises(base_objects.Process.PhysicsObjectError,
                          self.myprocess.get,
                          a_number)
        self.assertRaises(base_objects.Process.PhysicsObjectError,
                          self.myprocess.get,
                          'wrongparam')

        # Test set
        self.assertRaises(base_objects.Process.PhysicsObjectError,
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
        goal = goal + "    \'orders\': %s\n}" % repr(self.myprocess['orders'])

        self.assertEqual(goal, str(self.myprocess))

