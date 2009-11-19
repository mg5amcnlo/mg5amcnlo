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

"""Unit test library for the helas_objects module"""

import copy
import unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects

#===============================================================================
# HelasWavefunctionTest
#===============================================================================
class HelasWavefunctionTest(unittest.TestCase):
    """Test class for the HelasWavefunction object"""

    mydict = {}
    mywavefunction = None
    mymothers = helas_objects.HelasWavefunctionList()

    def setUp(self):

        mywavefunction = helas_objects.HelasWavefunction({'pdg_code': 12,
                                                           'interaction_id': 2,
                                                           'direction': 'incoming',
                                                           'number': 2})
        self.mymothers = helas_objects.HelasWavefunctionList([mywavefunction])
        self.mydict = {'pdg_code': 12,
                       'mothers': self.mymothers,
                       'interaction_id': 2,
                       'direction': 'incoming',
                       'number': 5}
                        

        self.mywavefunction = helas_objects.HelasWavefunction(self.mydict)

    def test_setget_wavefunction_correct(self):
        "Test correct HelasWavefunction object __init__, get and set"

        mywavefunction2 = helas_objects.HelasWavefunction()

        for prop in self.mydict.keys():
            mywavefunction2.set(prop, self.mydict[prop])

        self.assertEqual(self.mywavefunction, mywavefunction2)

        for prop in self.mywavefunction.keys():
            self.assertEqual(self.mywavefunction.get(prop), self.mydict[prop])

    def test_setget_wavefunction_exceptions(self):
        "Test error raising in HelasWavefunction __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          helas_objects.HelasWavefunction,
                          wrong_dict)
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          helas_objects.HelasWavefunction,
                          a_number)

        # Test get
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          self.mywavefunction.get,
                          a_number)
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          self.mywavefunction.get,
                          'wrongparam')

        # Test set
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          self.mywavefunction.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          self.mywavefunction.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for wavefunction properties"""

        test_values = [
                       {'prop':'interaction_id',
                        'right_list':[0, 3],
                        'wrong_list':['', 0.0]},
                       {'prop':'number',
                        'right_list':[1, 2, 3, 4, 5],
                        'wrong_list':['a', {}]},
                       {'prop':'direction',
                        'right_list':['incoming', 'outgoing'],
                        'wrong_list':[0, 'wrong']}
                       ]

        temp_wavefunction = self.mywavefunction

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_wavefunction.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_wavefunction.set(test['prop'], x))

    def test_representation(self):
        """Test wavefunction object string representation."""

        goal = "{\n"
        goal = goal + "    \'pdg_code\': 12,\n"
        goal = goal + "    \'mothers\': " + repr(self.mymothers) + ",\n"
        goal = goal + "    \'interaction_id\': 2,\n"
        goal = goal + "    \'direction\': \'incoming\',\n"
        goal = goal + "    \'number\': 5\n}"

        self.assertEqual(goal, str(self.mywavefunction))

    def test_equality(self):
        """Test that the overloaded equality operator works"""
        
        mymother = copy.copy(self.mymothers[0])
        mymother.set('pdg_code',13)
        mymothers = helas_objects.HelasWavefunctionList([mymother])
        mywavefunction = helas_objects.HelasWavefunction({'pdg_code': 12,
                                                          'interaction_id': 2,
                                                          'mothers': mymothers,
                                                          'direction': 'incoming',
                                                          'number': 2})

        self.assertTrue(self.mywavefunction == mywavefunction)
        mywavefunction.set('pdg_code', 13)
        self.assertFalse(self.mywavefunction == mywavefunction)
        mywavefunction.set('pdg_code', self.mywavefunction.get('pdg_code'))
        mywavefunction.set('mothers', helas_objects.HelasWavefunctionList())
        self.assertFalse(self.mywavefunction == mywavefunction)
        mymother.set('number', 4)
        mywavefunction.set('mothers', mymothers)
        self.assertFalse(self.mywavefunction == mywavefunction)


    def test_wavefunction_list(self):
        """Test wavefunction list initialization and counting functions
        for wavefunctions with 'from_group' = True"""

        mylist = [copy.copy(self.mywavefunction) for dummy in range(1, 4) ]
        mywavefunctionlist = helas_objects.HelasWavefunctionList(mylist)

        not_a_wavefunction = 1

        for wavefunction in mywavefunctionlist:
            self.assertEqual(wavefunction, self.mywavefunction)

        self.assertRaises(helas_objects.HelasWavefunctionList.PhysicsObjectListError,
                          mywavefunctionlist.append,
                          not_a_wavefunction)

#===============================================================================
# HelasAmplitudeTest
#===============================================================================
class HelasAmplitudeTest(unittest.TestCase):
    """Test class for the HelasAmplitude object"""

    mydict = {}
    myamplitude = None
    mywavefunctions = None

    def setUp(self):

        mydict = {'pdg_code': 10,
                  'mothers': helas_objects.HelasWavefunctionList(),
                  'interaction_id': 2,
                  'direction': 'incoming',
                  'number': 5}
                        

        self.mywavefunctions = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(mydict)] * 3)

        self.mydict = {'mothers': self.mywavefunctions,
                       'interaction_id': 2,
                       'number': 5}

        self.myamplitude = helas_objects.HelasAmplitude(self.mydict)

    def test_setget_amplitude_correct(self):
        "Test correct HelasAmplitude object __init__, get and set"

        myamplitude2 = helas_objects.HelasAmplitude()

        for prop in self.mydict.keys():
            myamplitude2.set(prop, self.mydict[prop])

        self.assertEqual(self.myamplitude, myamplitude2)

        for prop in self.myamplitude.keys():
            self.assertEqual(self.myamplitude.get(prop), self.mydict[prop])

    def test_setget_amplitude_exceptions(self):
        "Test error raising in HelasAmplitude __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          helas_objects.HelasAmplitude,
                          wrong_dict)
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          helas_objects.HelasAmplitude,
                          a_number)

        # Test get
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          self.myamplitude.get,
                          a_number)
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          self.myamplitude.get,
                          'wrongparam')

        # Test set
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          self.myamplitude.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          self.myamplitude.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for amplitude properties"""

        test_values = [
                       {'prop':'interaction_id',
                        'right_list':[0, 3],
                        'wrong_list':['', 0.0]},
                       {'prop':'number',
                        'right_list':[1, 2, 3, 4, 5],
                        'wrong_list':['a', {}]},
                       ]

        temp_amplitude = self.myamplitude

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_amplitude.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_amplitude.set(test['prop'], x))

    def test_representation(self):
        """Test amplitude object string representation."""

        goal = "{\n"
        goal = goal + "    \'mothers\': " + repr(self.mywavefunctions) + ",\n"
        goal = goal + "    \'interaction_id\': 2,\n"
        goal = goal + "    \'number\': 5\n}"

        self.assertEqual(goal, str(self.myamplitude))

    def test_amplitude_list(self):
        """Test amplitude list initialization and counting functions
        for amplitudes with 'from_group' = True"""

        mylist = [copy.copy(self.myamplitude) for dummy in range(1, 4) ]
        myamplitudelist = helas_objects.HelasAmplitudeList(mylist)

        not_a_amplitude = 1

        for amplitude in myamplitudelist:
            self.assertEqual(amplitude, self.myamplitude)

        self.assertRaises(helas_objects.HelasAmplitudeList.PhysicsObjectListError,
                          myamplitudelist.append,
                          not_a_amplitude)

#===============================================================================
# HelasDiagramTest
#===============================================================================
class HelasDiagramTest(unittest.TestCase):
    """Test class for the HelasDiagram object"""

    mydict = {}
    mywavefunctions = None
    myamplitude = None
    mydiagram = None

    def setUp(self):

        mydict = {'pdg_code': 10,
                  'mothers': helas_objects.HelasWavefunctionList(),
                  'interaction_id': 2,
                  'direction': 'incoming',
                  'number': 5}
                        

        self.mywavefunctions = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(mydict)] * 3)

        mydict = {'mothers': self.mywavefunctions,
                  'interaction_id': 2,
                  'number': 5}

        self.myamplitude = helas_objects.HelasAmplitude(self.mydict)
        
        self.mydict = {'wavefunctions': self.mywavefunctions,
                       'amplitude': self.myamplitude,
                       'fermionfactor': 1}
        self.mydiagram = helas_objects.HelasDiagram(self.mydict)

    def test_setget_diagram_correct(self):
        "Test correct HelasDiagram object __init__, get and set"

        mydiagram2 = helas_objects.HelasDiagram()

        for prop in self.mydict.keys():
            mydiagram2.set(prop, self.mydict[prop])

        self.assertEqual(self.mydiagram, mydiagram2)

        for prop in self.mydiagram.keys():
            self.assertEqual(self.mydiagram.get(prop), self.mydict[prop])

    def test_setget_diagram_exceptions(self):
        "Test error raising in HelasDiagram __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          helas_objects.HelasDiagram,
                          wrong_dict)
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          helas_objects.HelasDiagram,
                          a_number)

        # Test get
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          self.mydiagram.get,
                          a_number)
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          self.mydiagram.get,
                          'wrongparam')

        # Test set
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          self.mydiagram.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          self.mydiagram.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for diagram properties"""

        test_values = [
                       {'prop':'wavefunctions',
                        'right_list':[self.mywavefunctions],
                        'wrong_list':['', 0.0]},
                       {'prop':'amplitude',
                        'right_list':[self.myamplitude],
                        'wrong_list':['a', {}]},
                       {'prop':'fermionfactor',
                        'right_list':[-1,1],
                        'wrong_list':['a', {}, 0]},
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
        goal = goal + "    \'wavefunctions\': " + repr(self.mywavefunctions) + ",\n"
        goal = goal + "    \'amplitude\': " + repr(self.myamplitude) + ",\n"
        goal = goal + "    \'fermionfactor\': 1\n}"

        self.assertEqual(goal, str(self.mydiagram))

    def test_diagram_list(self):
        """Test diagram list initialization and counting functions
        for diagrams with 'from_group' = True"""

        mylist = [copy.copy(self.mydiagram) for dummy in range(1, 4) ]
        mydiagramlist = helas_objects.HelasDiagramList(mylist)

        not_a_diagram = 1

        for diagram in mydiagramlist:
            self.assertEqual(diagram, self.mydiagram)

        self.assertRaises(helas_objects.HelasDiagramList.PhysicsObjectListError,
                          mydiagramlist.append,
                          not_a_diagram)
