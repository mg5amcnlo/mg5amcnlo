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
        self.assertRaises(base_objects.Particle.ParticleError,
                          base_objects.Particle,
                          wrong_dict)
        self.assertRaises(base_objects.Particle.ParticleError,
                          base_objects.Particle,
                          a_number)

        # Test get
        self.assertRaises(base_objects.Particle.ParticleError,
                          self.mypart.get,
                          a_number)
        self.assertRaises(base_objects.Particle.ParticleError,
                          self.mypart.get,
                          'wrongparam')

        # Test set
        self.assertRaises(base_objects.Particle.ParticleError,
                          self.mypart.set,
                          a_number, 0)
        self.assertRaises(base_objects.Particle.ParticleError,
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

        for test in test_values:
            print test['prop'],
            for x in test['right_list']:
                self.assert_(self.mypart.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertRaises(base_objects.Particle.ParticleError,
                         self.mypart.set,
                         test['prop'], x)

        print ' ',

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

        for part in mypartlist:
            self.assertEqual(part, self.mypart)

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
        self.assertRaises(base_objects.Model.ModelError,
                          mymodel.get,
                          not_a_string)
        self.assertRaises(base_objects.Model.ModelError,
                          mymodel.get,
                          'wrong_key')
        self.assertRaises(base_objects.Model.ModelError,
                          mymodel.set,
                          not_a_string, None)
        self.assertRaises(base_objects.Model.ModelError,
                          mymodel.set,
                          'wrong_subclass', None)

        # For each subclass
        self.assertRaises(base_objects.Model.ModelError,
                          mymodel.set,
                          'particles', not_a_string)

if __name__ == "__main__":
    unittest.main()
