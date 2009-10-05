##############################################################################
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
##############################################################################

"""Unit test library for the various base objects of the core library"""

import unittest

import madgraph.core.base_objects as base_objects

class ParticleTest(unittest.TestCase):

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
                      'propagating':True}

        self.mypart = base_objects.Particle(self.mydict)

    def tearDown(self):
        pass

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
                                      'k++', 'k--', 'T'],
                        'wrong_list':['', 'x ', 'e?', '{}', '9x']},
                       {'prop':'spin',
                        'right_list':[1, 2, 3, 4, 5],
                        'wrong_list':[-1, 0, 'a', 6]},
                       {'prop':'color',
                        'right_list':[1, -1, 3, -3, 6, -6, 8],
                        'wrong_list':[2, 0, 'a', 23]},
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
        goal = goal + "    \'propagating\': True\n}"

        self.assertEqual(goal, str(self.mypart))

    def test_particle_list(self):
        """Test particle list initialization"""

        mylist = [self.mypart] * 10
        mypartlist = base_objects.ParticleList(mylist)

        not_a_part = 1

        for part in mypartlist:
            self.assertEqual(part, self.mypart)

        self.assertRaises(base_objects.ParticleList.PhysicsObjectListError,
                          mypartlist.append,
                          not_a_part)

class InteractionTest(unittest.TestCase):

    mydict = {}
    myinter = None

    def setUp(self):

        self.mydict = {'particles': ['a', 'b', 'c'],
                       'color': ['C1', 'C2'],
                       'lorentz':['L1', 'L2'],
                       'couplings':{(0, 0):'g00',
                                    (0, 1):'g01',
                                    (1, 0):'g10',
                                    (1, 1):'g11'},
                       'orders':['QCD', 'QED']}

        self.myinter = base_objects.Interaction(self.mydict)

    def test_setget_interaction_correct(self):
        "Test correct interaction object __init__, get and set"

        myinter2 = base_objects.Interaction()

        # First fill myinter2 it using set
        for prop in ['particles', 'color', 'lorentz', 'couplings', 'orders']:
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
                        'right_list':[[], ['a'], ['a+', 'b-']],
                        'wrong_list':[1, 'x ', ['e?'], ['a', ' ']]},
                       {'prop':'color',
                        'right_list':[[], ['C1'], ['C1', 'C2']],
                        'wrong_list':[1, 'a', ['a', 1]]},
                       {'prop':'lorentz',
                        'right_list':[[], ['L1'], ['L1', 'L2']],
                        'wrong_list':[1, 'a', ['a', 1]]},
                       {'prop':'orders',
                        'right_list':[[], ['QCD'], ['QED', 'QCD']],
                        'wrong_list':[1, 'a', ['a', 1]]},
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
        goal = goal + "    \'particles\': [\'a\', \'b\', \'c\'],\n"
        goal = goal + "    \'color\': [\'C1\', \'C2\'],\n"
        goal = goal + "    \'lorentz\': [\'L1\', \'L2\'],\n"
        goal = goal + "    \'couplings\': %s,\n" % repr(self.myinter['couplings'])
        goal = goal + "    \'orders\': [\'QCD\', \'QED\']\n}"

        self.assertEqual(goal, str(self.myinter))

    def test_interaction_list(self):
        """Test interaction list initialization"""

        mylist = [self.myinter] * 10
        myinterlist = base_objects.InteractionList(mylist)

        not_a_part = 1

        for part in myinterlist:
            self.assertEqual(part, self.myinter)

        self.assertRaises(base_objects.InteractionList.PhysicsObjectListError,
                          myinterlist.append,
                          not_a_part)

class ModelTest(unittest.TestCase):

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

    def test_setget_model_correct(self):
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

if __name__ == "__main__":
    unittest.main()
