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
import madgraph.core.diagram_generation as diagram_generation

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
                                                           'state': 'incoming',
                                                           'number': 2})
        self.mymothers = helas_objects.HelasWavefunctionList([mywavefunction])
        self.mydict = {'pdg_code': 12,
                       'name': 'none',
                       'antiname': 'none',
                       'spin': 1,
                       'color': 1,
                       'mass': 'zero',
                       'width': 'zero',
                       'is_part': True,
                       'self_antipart': False,
                       'mothers': self.mymothers,
                       'interaction_id': 2,
                       'pdg_codes':[1,2,3],
                       'inter_color': [],
                       'lorentz': [],
                       'couplings': { (0, 0):'none'},
                       'state': 'incoming',
                       'number_external': 4,
                       'number': 5,
                       'fermionflow': 1}

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
                       {'prop':'state',
                        'right_list':['incoming', 'outgoing', 'intermediate'],
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
        goal = goal + "    \'name\': \'none\',\n"
        goal = goal + "    \'antiname\': \'none\',\n"
        goal = goal + "    \'spin\': 1,\n"
        goal = goal + "    \'color\': 1,\n"
        goal = goal + "    \'mass\': 'zero',\n"
        goal = goal + "    \'width\': 'zero',\n"
        goal = goal + "    \'is_part\': True,\n"
        goal = goal + "    \'self_antipart\': False,\n"
        goal = goal + "    \'interaction_id\': 2,\n"
        goal = goal + "    \'pdg_codes\': [1, 2, 3],\n"
        goal = goal + "    \'inter_color\': [],\n"
        goal = goal + "    \'lorentz\': [],\n"
        goal = goal + "    \'couplings\': {(0, 0): \'none\'},\n"
        goal = goal + "    \'state\': \'incoming\',\n"
        goal = goal + "    \'number_external\': 4,\n"
        goal = goal + "    \'number\': 5,\n"
        goal = goal + "    \'fermionflow\': 1,\n"
        goal = goal + "    \'mothers\': " + repr(self.mymothers) + "\n}"

        self.assertEqual(goal, str(self.mywavefunction))

    def test_equality(self):
        """Test that the overloaded equality operator works"""
        
        mymother = copy.copy(self.mymothers[0])
        mymother.set('pdg_code',13)
        mymothers = helas_objects.HelasWavefunctionList([mymother])
        mywavefunction = copy.copy(self.mywavefunction)
        mywavefunction.set('mothers',mymothers)
        self.assertTrue(self.mywavefunction == mywavefunction)
        mywavefunction.set('spin', 5)
        self.assertFalse(self.mywavefunction == mywavefunction)
        mywavefunction.set('spin', self.mywavefunction.get('spin'))
        mywavefunction.set('mothers', helas_objects.HelasWavefunctionList())
        self.assertFalse(self.mywavefunction == mywavefunction)
        mymother.set('number', 4)
        mywavefunction.set('mothers', mymothers)
        self.assertFalse(self.mywavefunction == mywavefunction)


    def test_wavefunction_list(self):
        """Test wavefunction list initialization"""

        mylist = [copy.copy(self.mywavefunction) for dummy in range(1, 4) ]
        mywavefunctionlist = helas_objects.HelasWavefunctionList(mylist)

        not_a_wavefunction = 1

        for wavefunction in mywavefunctionlist:
            self.assertEqual(wavefunction, self.mywavefunction)

        self.assertRaises(helas_objects.HelasWavefunctionList.PhysicsObjectListError,
                          mywavefunctionlist.append,
                          not_a_wavefunction)

    def test_equality_in_list(self):
        """Test that the overloaded equality operator works also for a list"""
        mymother = copy.copy(self.mymothers[0])
        mymothers = helas_objects.HelasWavefunctionList([mymother])
        mymother.set('pdg_code',100)
        mywavefunction = copy.copy(self.mywavefunction)
        mywavefunction.set('mothers',mymothers)
        mywavefunction.set('spin',self.mywavefunction.get('spin') + 1)

        wavefunctionlist = helas_objects.HelasWavefunctionList(\
            [copy.copy(wf) for wf in [ mywavefunction ] * 100 ])
        self.assertFalse(self.mywavefunction in wavefunctionlist)
        mywavefunction.set('spin',self.mywavefunction.get('spin'))
        self.assertFalse(self.mywavefunction in wavefunctionlist)
        wavefunctionlist.append(mywavefunction)
        self.assertTrue(self.mywavefunction in wavefunctionlist)

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
                  'name': 'none',
                  'antiname': 'none',
                  'spin': 1,
                  'color': 1,
                  'mass': 'zero',
                  'width': 'zero',
                  'is_part': True,
                  'self_antipart': False,
                  'interaction_id': 2,
                  'pdg_codes':[1,2,3],
                  'inter_color': [],
                  'lorentz': [],
                  'couplings': { (0, 0):'none'},
                  'state': 'incoming',
                  'mothers': helas_objects.HelasWavefunctionList(),
                  'number': 5}

        self.mywavefunctions = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(mydict)] * 3)

        self.mydict = {'mothers': self.mywavefunctions,
                       'interaction_id': 2,
                       'pdg_codes':[1,2,3],
                       'inter_color': [],
                       'lorentz': [],
                       'couplings': { (0, 0):'none'},
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
        goal = goal + "    \'interaction_id\': 2,\n"
        goal = goal + "    \'pdg_codes\': [1, 2, 3],\n"
        goal = goal + "    \'inter_color\': [],\n"
        goal = goal + "    \'lorentz\': [],\n"
        goal = goal + "    \'couplings\': {(0, 0): \'none\'},\n"
        goal = goal + "    \'number\': 5,\n"
        goal = goal + "    \'mothers\': " + repr(self.mywavefunctions) + "\n}"

        self.assertEqual(goal, str(self.myamplitude))

    def test_sign_flips_to_order(self):
        """Test the sign from flips to order a list"""

        mylist = []

        mylist.append(3)
        mylist.append(2)
        mylist.append(6)
        mylist.append(4)

        self.assertEqual(helas_objects.HelasAmplitude().sign_flips_to_order(mylist), 1)

        mylist[3] = 1
        self.assertEqual(helas_objects.HelasAmplitude().sign_flips_to_order(mylist), -1)

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
                  'state': 'incoming',
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
                        'right_list':[-1,1,0],
                        'wrong_list':['a', {}, 0.]},
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


#===============================================================================
# HelasMatrixElementTest
#===============================================================================
class HelasMatrixElementTest(unittest.TestCase):
    """Test class for the HelasMatrixElement object"""

    mydict = {}
    mywavefunctions = None
    myamplitude = None
    mydiagrams = None
    mymatrixelement = None
    mymodel = base_objects.Model()


    def setUp(self):

        mydict = {'pdg_code': 10,
                  'mothers': helas_objects.HelasWavefunctionList(),
                  'interaction_id': 2,
                  'state': 'incoming',
                  'number': 5}
                        

        self.mywavefunctions = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(mydict)] * 3)

        mydict = {'mothers': self.mywavefunctions,
                  'interaction_id': 2,
                  'number': 5}

        self.myamplitude = helas_objects.HelasAmplitude(self.mydict)
        
        mydict = {'wavefunctions': self.mywavefunctions,
                  'amplitude': self.myamplitude,
                  'fermionfactor': 1}
        
        self.mydiagrams = helas_objects.HelasDiagramList([helas_objects.HelasDiagram(mydict)] * 4)
        self.mydict = {'processes': base_objects.ProcessList(),
                       'diagrams': self.mydiagrams}
        self.mymatrixelement = helas_objects.HelasMatrixElement(self.mydict)

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        g = mypartlist[len(mypartlist)-1]

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
        u = mypartlist[len(mypartlist)-1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'e^-',
                      'antitexname':'e^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        eminus = mypartlist[len(mypartlist)-1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
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
        a = mypartlist[len(mypartlist)-1]


        # A E slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'sl2-',
                      'antiname':'sl2+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        seminus = mypartlist[len(mypartlist)-1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # A neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu1',
                      'width':'Wneu1',
                      'texname':'\chi_0^1',
                      'antitexname':'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist)-1]

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             a]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX15'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)        

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

    def test_setget_matrix_element_correct(self):
        "Test correct HelasMatrixElement object __init__, get and set"

        mymatrixelement2 = helas_objects.HelasMatrixElement()

        for prop in self.mydict.keys():
            mymatrixelement2.set(prop, self.mydict[prop])

        self.assertEqual(self.mymatrixelement, mymatrixelement2)

        for prop in self.mymatrixelement.keys():
            self.assertEqual(self.mymatrixelement.get(prop), self.mydict[prop])

    def test_setget_matrix_element_exceptions(self):
        "Test error raising in HelasMatrixElement __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          helas_objects.HelasMatrixElement,
                          wrong_dict)
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          helas_objects.HelasMatrixElement,
                          a_number)

        # Test get
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          self.mymatrixelement.get,
                          a_number)
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          self.mymatrixelement.get,
                          'wrongparam')

        # Test set
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          self.mymatrixelement.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          self.mymatrixelement.set,
                          'wrongparam', 0)

    def test_values_for_prop(self):
        """Test filters for matrix_element properties"""

        test_values = [
                       {'prop':'diagrams',
                        'right_list':[self.mydiagrams],
                        'wrong_list':['', 0.0]}
                       ]

        temp_matrix_element = self.mymatrixelement

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_matrix_element.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_matrix_element.set(test['prop'], x))

    def test_representation(self):
        """Test matrix_element object string representation."""

        goal = "{\n"
        goal = goal + "    \'processes\': [],\n"
        goal = goal + "    \'diagrams\': " + repr(self.mydiagrams) + "\n}"

        self.assertEqual(goal, str(self.mymatrixelement))


    def test_process_init(self):
        """Testing the process initialization using the process
        e- e+ > e- e+
        """

        # Test e+ e- > e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)
        
        self.assertEqual(matrix_element.get('processes')[0],
                         myamplitude.get('process'))

    def test_get_helicity_matrix(self):
        """Testing helicity matrix using the process
        e- e+ > z a
        """

        # A Z
        self.mymodel.get('particles').append(base_objects.Particle({'name':'Z',
                      'antiname':'Z',
                      'spin':3,
                      'color':1,
                      'mass':'MZ',
                      'width':'WZ',
                      'texname':'Z',
                      'antitexname':'Z',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        # A Higgs
        self.mymodel.get('particles').append(base_objects.Particle({'name':'H',
                      'antiname':'H',
                      'spin':1,
                      'color':1,
                      'mass':'MH',
                      'width':'WH',
                      'texname':'H',
                      'antitexname':'H',
                      'line':'dashed',
                      'charge':0.,
                      'pdg_code':25,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        self.mymodel.set('particle_dict',
                         self.mymodel.get('particles').generate_dict())

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':25,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        matrix_element = helas_objects.HelasMatrixElement()
        matrix_element.set('processes', base_objects.ProcessList([ myproc ]))

        self.assertEqual(matrix_element.get_helicity_combinations(), 24)

        goal_prods = [(-1, -1, -1, -1, 0),
                      (-1, -1, -1, 1, 0),
                      (-1, -1, 0, -1, 0),
                      (-1, -1, 0, 1, 0),
                      (-1, -1, 1, -1, 0),
                      (-1, -1, 1, 1, 0),
                      (-1, 1, -1, -1, 0),
                      (-1, 1, -1, 1, 0),
                      (-1, 1, 0, -1, 0),
                      (-1, 1, 0, 1, 0),
                      (-1, 1, 1, -1, 0),
                      (-1, 1, 1, 1, 0),
                      (1, -1, -1, -1, 0),
                      (1, -1, -1, 1, 0),
                      (1, -1, 0, -1, 0),
                      (1, -1, 0, 1, 0),
                      (1, -1, 1, -1, 0),
                      (1, -1, 1, 1, 0),
                      (1, 1, -1, -1, 0),
                      (1, 1, -1, 1, 0),
                      (1, 1, 0, -1, 0),
                      (1, 1, 0, 1, 0),
                      (1, 1, 1, -1, 0),
                      (1, 1, 1, 1, 0)]

        i = 0
        for prod in matrix_element.get_helicity_matrix():
            self.assertEqual(prod, goal_prods[i])
            i = i + 1
        
    def test_get_den_factor(self):
        """Testing helicity matrix using the process
        u u~ > a a a
        """

        # A Z
        self.mymodel.get('particles').append(base_objects.Particle({'name':'Z',
                      'antiname':'Z',
                      'spin':3,
                      'color':1,
                      'mass':'MZ',
                      'width':'WZ',
                      'texname':'Z',
                      'antitexname':'Z',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        self.mymodel.set('particle_dict',
                         self.mymodel.get('particles').generate_dict())

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        matrix_element = helas_objects.HelasMatrixElement()
        matrix_element.set('processes', base_objects.ProcessList([ myproc ]))

        self.assertEqual(matrix_element.get_denominator_factor(), 9 * 4 * 6)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        matrix_element = helas_objects.HelasMatrixElement()
        matrix_element.set('processes', base_objects.ProcessList([ myproc ]))

        self.assertEqual(matrix_element.get_denominator_factor(), 1 * 6 * 6)

    def test_fermionfactor_emep_emep(self):
        """Testing the fermion factor using the process  e- e+ > e- e+
        """

        # Test e+ e- > e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        myamplitude.get('diagrams')

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)
        
        diagrams = matrix_element.get('diagrams')

        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[1].get('fermionfactor'), -1)

    def test_fermionfactor_emep_emepa(self):
        """Testing the fermion factor using the process  e- e+ > e- e+ a
        """

        # Test e+ e- > e+ e- a

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        myamplitude.get('diagrams')

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)
        
        diagrams = matrix_element.get('diagrams')

        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[1].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[2].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[3].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[4].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[5].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[6].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[7].get('fermionfactor'), 1)

    def test_fermionfactor_emep_emepemep(self):
        """Testing the fermion factor using the process
        e- e+ > e- e+ e- e+
        Time estimates for e+e->e+e-e+e-e+e- (1728 diagrams):
        Diagram generation: 18 s
        Helas call generation (with optimization): 58 s
        Helas call generation (without optimization): 23 s
        Fermion factor calculation: 0 s
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        #print myamplitude.get('process').nice_string()

        myamplitude.get('diagrams')

        #print "diagrams: ", myamplitude.get('diagrams').nice_string()

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        #print "\n".join(helas_objects.HelasFortranModel().\
        #      get_matrix_element_calls(matrix_element))
        #print helas_objects.HelasFortranModel().\
        #      get_JAMP_line(matrix_element)
        
        diagrams = matrix_element.get('diagrams')

        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[1].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[2].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[3].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[4].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[5].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[6].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[7].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[8].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[9].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[10].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[11].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[12].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[13].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[14].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[15].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[16].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[17].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[18].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[19].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[20].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[21].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[22].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[23].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[24].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[25].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[26].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[27].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[28].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[29].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[30].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[31].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[32].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[33].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[34].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[35].get('fermionfactor'), -1)

    def test_fermionfactor_epem_sepsemepem(self):
        """Testing the fermion factor using the process
        e+ e- > se+ se- e+ e-
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'me',
                      'width':'zero',
                      'texname':'e^+',
                      'antitexname':'e^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        eminus = mypartlist[len(mypartlist)-1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A E slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'sl2-',
                      'antiname':'sl2+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        seminus = mypartlist[len(mypartlist)-1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # A neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu1',
                      'width':'Wneu1',
                      'texname':'\chi_0^1',
                      'antitexname':'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist)-1]

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        myamplitude.get('diagrams')
        
        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        diagrams = matrix_element.get('diagrams')

        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[1].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[2].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[3].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[4].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[5].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[6].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('fermionfactor') * \
                         diagrams[7].get('fermionfactor'), 1)

    def test_generate_helas_diagrams_uux_gepem(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes u u~ > g e+ e-
        """

        # Test u u~ > g e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "  ((1(-2),3(21)>1(-2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4))\n"
        goal = goal + "  ((2(2),3(21)>2(2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4))"

        self.assertEqual(goal,
                         myamplitude.get('diagrams').nice_string())

        wavefunctions1 = helas_objects.HelasWavefunctionList()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[4], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[5].set('pdg_code', 2, self.mymodel)
        wavefunctions1[5].set('number_external', 1)
        wavefunctions1[5].set('state', 'incoming')
        wavefunctions1[5].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[2]]))
        wavefunctions1[5].set('interaction_id', 3, self.mymodel)
        wavefunctions1[5].set('number', 6)
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[6].set('pdg_code', 22, self.mymodel)
        wavefunctions1[6].set('number_external', 4)
        wavefunctions1[6].set('state', 'intermediate')
        wavefunctions1[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3],wavefunctions1[4]]))
        wavefunctions1[6].set('interaction_id', 7, self.mymodel)
        wavefunctions1[6].set('number', 7)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[5], wavefunctions1[1],
                          wavefunctions1[6]]),
             'number': 1})
        amplitude1.set('interaction_id', 4, self.mymodel)

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[0].set('pdg_code', -2, self.mymodel)
        wavefunctions2[0].set('number_external', 2)
        wavefunctions2[0].set('state', 'outgoing')
        wavefunctions2[0].set('mothers', helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1],wavefunctions1[2]]))
        wavefunctions2[0].set('interaction_id', 3, self.mymodel)
        wavefunctions2[0].set('number', 8)

        amplitude2 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions2[0],
                          wavefunctions1[6]]),
             'number': 2})
        amplitude2.set('interaction_id', 4, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1,
                                               'fermionfactor': -1})

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2,
                                               'fermionfactor': -1})

        diagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(\
            myamplitude,
            1)
        
        self.assertEqual(matrix_element.get('diagrams'), diagrams)

    def test_generate_helas_diagrams_uux_gepem_no_optimization(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes u u~ > g e+ e-
        """

        # Test u u~ > g e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "  ((1(-2),3(21)>1(-2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4))\n"
        goal = goal + "  ((2(2),3(21)>2(2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4))"

        self.assertEqual(goal,
                         myamplitude.get('diagrams').nice_string())

        wavefunctions1 = helas_objects.HelasWavefunctionList()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[4], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[5].set('pdg_code', 2, self.mymodel)
        wavefunctions1[5].set('number_external', 1)
        wavefunctions1[5].set('state', 'incoming')
        wavefunctions1[5].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[2]]))
        wavefunctions1[5].set('interaction_id', 3, self.mymodel)
        wavefunctions1[5].set('number', 6)
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[6].set('pdg_code', 22, self.mymodel)
        wavefunctions1[6].set('number_external', 4)
        wavefunctions1[6].set('state', 'intermediate')
        wavefunctions1[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3],wavefunctions1[4]]))
        wavefunctions1[6].set('interaction_id', 7, self.mymodel)
        wavefunctions1[6].set('number', 7)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[5], wavefunctions1[1],
                          wavefunctions1[6]]),
             'number': 1})
        amplitude1.set('interaction_id', 4, self.mymodel)

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[4], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[5].set('pdg_code', -2, self.mymodel)
        wavefunctions2[5].set('number_external', 2)
        wavefunctions2[5].set('state', 'outgoing')
        wavefunctions2[5].set('mothers', helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1],wavefunctions1[2]]))
        wavefunctions2[5].set('interaction_id', 3, self.mymodel)
        wavefunctions2[5].set('number', 6)
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[6].set('pdg_code', 22, self.mymodel)
        wavefunctions2[6].set('number_external', 4)
        wavefunctions2[6].set('state', 'intermediate')
        wavefunctions2[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3],wavefunctions1[4]]))
        wavefunctions2[6].set('interaction_id', 7, self.mymodel)
        wavefunctions2[6].set('number', 7)

        amplitude2 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions2[0], wavefunctions2[5],
                          wavefunctions2[6]]),
             'number': 2})
        amplitude2.set('interaction_id', 4, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1,
                                               'fermionfactor': -1})

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2,
                                               'fermionfactor': -1})

        diagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(\
            myamplitude,
            0)
        
        self.assertEqual(matrix_element.get('diagrams'), diagrams)

    def test_generate_helas_diagrams_ae_ae(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes a e- > a e-
        """

        # Test a e- > a e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "  ((1(22),2(-11)>1(-11),id:7),(3(22),4(11)>3(11),id:7),(1(-11),3(11),id:0))\n"
        goal = goal + "  ((1(22),4(11)>1(11),id:7),(2(-11),3(22)>2(-11),id:7),(1(11),2(-11),id:0))"

        self.assertEqual(goal,
                         myamplitude.get('diagrams').nice_string())

        wavefunctions1 = helas_objects.HelasWavefunctionList()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[4].set('pdg_code', 11, self.mymodel)
        wavefunctions1[4].set('interaction_id', 7, self.mymodel)
        wavefunctions1[4].set('state', 'incoming')
        wavefunctions1[4].set('number_external', 1)
        wavefunctions1[4].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[1]]))
        wavefunctions1[4].set('number', 5)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[4], wavefunctions1[2],
                          wavefunctions1[3]]),
             'number': 1})
        amplitude1.set('interaction_id', 7, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1,
                                               'fermionfactor': 1})

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[0].set('pdg_code', -11, self.mymodel)
        wavefunctions2[0].set('interaction_id', 7, self.mymodel)
        wavefunctions2[0].set('state', 'outgoing')
        wavefunctions2[0].set('number_external', 1)
        wavefunctions2[0].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[3]]))
        wavefunctions2[0].set('number', 6)
        
        amplitude2 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions2[0], wavefunctions1[1],
                          wavefunctions1[2]]),
             'interaction_id': 7,
             'number': 2})
        amplitude2.set('interaction_id', 7, self.mymodel)

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2,
                                               'fermionfactor': 1})

        mydiagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)
        
        self.assertEqual(matrix_element.get('diagrams'), mydiagrams)

    def test_generate_helas_diagrams_ea_ae(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes e- a > a e-
        """

        # Test e- a > a e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "  ((1(-11),2(22)>1(-11),id:7),(3(22),4(11)>3(11),id:7),(1(-11),3(11),id:0))\n"
        goal = goal + "  ((1(-11),3(22)>1(-11),id:7),(2(22),4(11)>2(11),id:7),(1(-11),2(11),id:0))"

        self.assertEqual(goal,
                         myamplitude.get('diagrams').nice_string())
        
        wavefunctions1 = helas_objects.HelasWavefunctionList()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[4].set('pdg_code', 11, self.mymodel)
        wavefunctions1[4].set('interaction_id', 7, self.mymodel)
        wavefunctions1[4].set('number_external', 1)
        wavefunctions1[4].set('state', 'incoming')
        wavefunctions1[4].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[1]]))
        wavefunctions1[4].set('number', 5)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[4], wavefunctions1[2],
                          wavefunctions1[3]]),
             'number': 1})
        amplitude1.set('interaction_id', 7, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1,
                                               'fermionfactor': 1})

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[0].set('pdg_code', 11, self.mymodel)
        wavefunctions2[0].set('interaction_id', 7, self.mymodel)
        wavefunctions2[0].set('number_external', 1)
        wavefunctions2[0].set('state', 'incoming')
        wavefunctions2[0].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[2]]))
        wavefunctions2[0].set('number', 6)
        
        amplitude2 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions2[0], wavefunctions1[1],
                          wavefunctions1[3]]),
             'interaction_id': 7,
             'number': 2})
        amplitude2.set('interaction_id', 7, self.mymodel)

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2,
                                               'fermionfactor': 1})
        mydiagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)
        
        self.assertEqual(matrix_element.get('diagrams'), mydiagrams)

#===============================================================================
# HelasMultiProcessTest
#===============================================================================
class HelasMultiProcessTest(unittest.TestCase):
    """Test class for the HelasMultiProcess object"""

    mydict = {}
    mywavefunctions = None
    myamplitude = None
    mydiagrams = None
    mymatrixelement = None
    mymodel = base_objects.Model()


    def setUp(self):

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        g = mypartlist[len(mypartlist)-1]

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
        u = mypartlist[len(mypartlist)-1]
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
                      'charge':-1./3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[len(mypartlist)-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             g]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # 3-Gluon coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX1'},
                      'orders':{'QCD':1}}))


        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)        


    def test_helas_multi_process(self):
        """Test the HelasMultiProcess with the processes uu~>uu~
        and dd~>dd~"""
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':'final'}))

        myproc1 = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude1 = diagram_generation.Amplitude({'process': myproc1})

        myamplitude1.get('diagrams')

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':'final'}))

        myproc2 = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude2 = diagram_generation.Amplitude({'process': myproc2})

        myamplitude2.get('diagrams')

        myamplitudes = diagram_generation.AmplitudeList([ myamplitude1,
                                                          myamplitude2 ])

        my_matrix_element1 = helas_objects.HelasMatrixElement(myamplitude1)
        my_multiprocess = helas_objects.HelasMultiProcess(myamplitudes)

        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 1)
        self.assertEqual(my_multiprocess.get('matrix_elements')[0].\
                         get('processes'),
                         base_objects.ProcessList([ myproc1, myproc2 ]))
        self.assertEqual(my_multiprocess.get('matrix_elements')[0].\
                         get('diagrams'),
                         my_matrix_element1.get('diagrams'))

        myamplitudes[0].get('process').set('id', 10)

        my_multiprocess = helas_objects.HelasMultiProcess(myamplitudes)
        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 2)


    def test_helas_multiprocess_pp_nj(self):
        """Setting up and testing pp > nj based on multiparticle lists,
        using the amplitude functionality of MultiProcess
        (which makes partial use of crossing symmetries)
        """

        max_fs = 2 # 3

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': 'final'});

        goal_number_matrix_elements = [22,34]

        for nfs in range(2, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * (2 + nfs)])

            my_multi_leglist[0].set('state','initial')
            my_multi_leglist[1].set('state','initial')

            my_process_definition = base_objects.ProcessDefinition({'legs':my_multi_leglist,
                                                                    'model':self.mymodel})
            my_multiprocess = diagram_generation.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            helas_multi_proc = helas_objects.HelasMultiProcess(my_multiprocess)

            if nfs <= 3:
                self.assertEqual(len(helas_multi_proc.get('matrix_elements')),
                                     goal_number_matrix_elements[nfs - 2])

#===============================================================================
# HelasModelTest
#===============================================================================
class HelasModelTest(unittest.TestCase):
    """Test class for the HelasModel object"""

    mymodel = helas_objects.HelasModel()
    mybasemodel = base_objects.Model()    

    def setUp(self):
        self.mymodel.set('name', 'sm')

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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

        g = mypartlist[len(mypartlist)-1]

        # A quark U and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'u',
                      'antiname':'u~',
                      'spin':2,
                      'color':3,
                      'mass':'mu',
                      'width':'zero',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':2,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        u = mypartlist[len(mypartlist)-1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'me',
                      'width':'zero',
                      'texname':'e^+',
                      'antitexname':'e^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        eminus = mypartlist[len(mypartlist)-1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
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
        a = mypartlist[len(mypartlist)-1]

        # A T particle
        mypartlist.append(base_objects.Particle({'name':'T1',
                      'antiname':'T1',
                      'spin':5,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'T',
                      'antitexname':'T',
                      'line':'double',
                      'charge':0.,
                      'pdg_code':8000002,
                      'propagating':False,
                      'is_part':True,
                      'self_antipart':True}))
        T1 = mypartlist[len(mypartlist)-1]

        # A U squark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'su',
                      'antiname':'su~',
                      'spin':1,
                      'color':3,
                      'mass':'Musq2',
                      'width':'Wusq2',
                      'texname':'\tilde u',
                      'antitexname':'\bar {\tilde u}',
                      'line':'dashed',
                      'charge':2. / 3.,
                      'pdg_code':1000002,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        su = mypartlist[len(mypartlist)-1]
        antisu = copy.copy(su)
        antisu.set('is_part', False)

        # A E slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'sl2-',
                      'antiname':'sl2+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        seminus = mypartlist[len(mypartlist)-1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # A neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu1',
                      'width':'Wneu1',
                      'texname':'\chi_0^1',
                      'antitexname':'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist)-1]

        # W+ and W-
        mypartlist.append(base_objects.Particle({'name':'W+',
                      'antiname':'W-',
                      'spin':3,
                      'color':1,
                      'mass':'wmas',
                      'width':'wwid',
                      'texname':'W^+',
                      'antitexname':'W^-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[len(mypartlist)-1]
        wminus = copy.copy(u)
        wminus.set('is_part', False)

        # Z
        mypartlist.append(base_objects.Particle({'name':'Z',
                      'antiname':'Z',
                      'spin':3,
                      'color':1,
                      'mass':'zmas',
                      'width':'zwid',
                      'texname':'Z',
                      'antitexname':'Z',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[len(mypartlist)-1]

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             a]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX15'},
                      'orders':{'QED':1}}))

        # Tgg coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             T1]),
                      'color': ['C1'],
                      'lorentz':['A'],
                      'couplings':{(0, 0):'MGVX2'},
                      'orders':{'QCD':1}}))


        # ggg coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 15,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX1'},
                      'orders':{'QCD':1}}))

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        # Gluon coupling to su
        myinterlist.append(base_objects.Interaction({
                      'id': 105,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             su, \
                                             antisu]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX74'},
                      'orders':{'QCD':1}}))

        # Coupling of n1 to u and su
        myinterlist.append(base_objects.Interaction({
                      'id': 101,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             u, \
                                             antisu]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX570'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 102,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             n1, \
                                             su]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX575'},
                      'orders':{'QED':1}}))

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        # g-gamma-su-subar coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 100,
                      'particles': base_objects.ParticleList(\
                                            [a,
                                             g,
                                             su,
                                             antisu]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX89'},
                      'orders':{'QED':1, 'QCD':1}}))

        # w+w-w+w- coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [wplus,
                                             wminus,
                                             wplus,
                                             wminus]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX6'},
                      'orders':{'QED':2}}))

        # w+w-zz coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [wplus,
                                             wminus,
                                             z,
                                             z]),
                      'color': ['C1'],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX8'},
                      'orders':{'QED':2}}))


        self.mybasemodel.set('particles', mypartlist)
        self.mybasemodel.set('interactions', myinterlist)

    def test_setget_helas_model_correct(self):
        """Test correct HelasModel object get and set"""

        self.assertEqual(self.mymodel.get('name'), 'sm')

    def test_setget_helas_model_error(self):
        """Test error raising in HelasModel object get and set"""

        mymodel = helas_objects.HelasModel()
        not_a_string = 1.

        # General
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.get,
                          not_a_string)
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.get,
                          'wrong_key')
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.set,
                          not_a_string, None)
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.set,
                          'wrong_subclass', None)
        # add_wavefunction and add_amplitude
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.add_wavefunction,
                          'wrong_subclass', None)
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.add_wavefunction,
                          (1,2), "not_a_function")
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.add_amplitude,
                          'wrong_subclass', None)
        self.assertRaises(helas_objects.HelasModel.PhysicsObjectError,
                          mymodel.add_amplitude,
                          (1,2), "not_a_function")

    def test_set_wavefunctions(self):
        """Test wavefunction dictionary in HelasModel"""

        wavefunctions = {}
        # IXXXXXX.Key: (spin, state)
        key1 = (tuple([-2]),tuple(['']))
        wavefunctions[key1] = \
                          lambda wf: 'CALL IXXXXX(P(0,%d),%s,NHEL(%d),%d*IC(%d),W(1,%d))' % \
                          (wf.get('number_external'), wf.get('mass'),
                           wf.get('number_external'), -(-1)**wf.get_with_flow('is_part'),
                           wf.get('number_external'), wf.get('number'))
        # OXXXXXX.Key: (spin, state)
        key2 = (tuple([2]),tuple(['']))
        wavefunctions[key2] = \
                          lambda wf: 'CALL OXXXXX(P(0,%d),%s,NHEL(%d),%d*IC(%d),W(1,%d))' % \
                          (wf.get('number_external'), wf.get('mass'),
                           wf.get('number_external'), 1**wf.get_with_flow('is_part'),
                           wf.get('number_external'), wf.get('number'))
        
        self.assert_(self.mymodel.set('wavefunctions', wavefunctions))

        wf = helas_objects.HelasWavefunction()
        wf.set('pdg_code', -2, self.mybasemodel)
        wf.set('state', 'incoming')
        wf.set('interaction_id', 0)
        wf.set('number_external', 1)
        wf.set('lorentz', [''])
        wf.set('number', 40)

        self.assertEqual(wf.get_call_key(), key1)

        goal = 'CALL IXXXXX(P(0,1),mu,NHEL(1),-1*IC(1),W(1,40))'
        self.assertEqual(self.mymodel.get_wavefunction_call(wf), goal)

        wf.set('fermionflow', -1)

        self.assertEqual(wf.get_call_key(), key2)

        goal = 'CALL OXXXXX(P(0,1),mu,NHEL(1),1*IC(1),W(1,40))'
        self.assertEqual(self.mymodel.get_wavefunction_call(wf), goal)

