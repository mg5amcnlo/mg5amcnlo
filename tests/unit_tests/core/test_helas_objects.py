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
                                                           'state': 'initial',
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
                       'inter_color': [],
                       'lorentz': [],
                       'couplings': { (0, 0):'none'},
                       'state': 'initial',
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
                        'right_list':['initial', 'final', 'intermediate'],
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
        goal = goal + "    \'inter_color\': [],\n"
        goal = goal + "    \'lorentz\': [],\n"
        goal = goal + "    \'couplings\': {(0, 0): \'none\'},\n"
        goal = goal + "    \'state\': \'initial\',\n"
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
                  'inter_color': [],
                  'lorentz': [],
                  'couplings': { (0, 0):'none'},
                  'state': 'initial',
                  'mothers': helas_objects.HelasWavefunctionList(),
                  'number': 5}

        self.mywavefunctions = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(mydict)] * 3)

        self.mydict = {'mothers': self.mywavefunctions,
                       'interaction_id': 2,
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
        goal = goal + "    \'inter_color\': [],\n"
        goal = goal + "    \'lorentz\': [],\n"
        goal = goal + "    \'couplings\': {(0, 0): \'none\'},\n"
        goal = goal + "    \'number\': 5,\n"
        goal = goal + "    \'mothers\': " + repr(self.mywavefunctions) + "\n}"

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
                  'state': 'initial',
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
                  'state': 'initial',
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
        self.mydict = {'diagrams': self.mydiagrams}
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
        mypartlist.append(base_objects.Particle({'name':'e+',
                      'antiname':'e-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
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


        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             a]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))


        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)
        

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
        goal = goal + "    \'diagrams\': " + repr(self.mydiagrams) + "\n}"

        self.assertEqual(goal, str(self.mymatrixelement))


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
        goal = goal + "  ((1,3>1,id:3),(4,5>4,id:7),(1,2,4,id:4))\n"
        goal = goal + "  ((2,3>2,id:3),(4,5>4,id:7),(1,2,4,id:4))"

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
        wavefunctions1[5].set('pdg_code', (2, self.mymodel))
        wavefunctions1[5].set('state', 'intermediate')
        wavefunctions1[5].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[2]]))
        wavefunctions1[5].set('interaction_id', (3, self.mymodel))
        wavefunctions1[5].set('number', 6)
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[6].set('pdg_code', (22, self.mymodel))
        wavefunctions1[6].set('state', 'intermediate')
        wavefunctions1[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3],wavefunctions1[4]]))
        wavefunctions1[6].set('interaction_id', (7, self.mymodel))
        wavefunctions1[6].set('number', 7)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[5], wavefunctions1[1],
                          wavefunctions1[6]]),
             'number': 1})
        amplitude1.set('interaction_id', (4, self.mymodel))

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[0].set('pdg_code', (-2, self.mymodel))
        wavefunctions2[0].set('state', 'intermediate')
        wavefunctions2[0].set('mothers', helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1],wavefunctions1[2]]))
        wavefunctions2[0].set('interaction_id', (3, self.mymodel))
        wavefunctions2[0].set('number', 8)

        amplitude2 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions2[0],
                          wavefunctions1[6]]),
             'number': 2})
        amplitude2.set('interaction_id', (4, self.mymodel))

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1})

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2})

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
        goal = goal + "  ((1,3>1,id:3),(4,5>4,id:7),(1,2,4,id:4))\n"
        goal = goal + "  ((2,3>2,id:3),(4,5>4,id:7),(1,2,4,id:4))"

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
        wavefunctions1[5].set('pdg_code', (2, self.mymodel))
        wavefunctions1[5].set('state', 'intermediate')
        wavefunctions1[5].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[2]]))
        wavefunctions1[5].set('interaction_id', (3, self.mymodel))
        wavefunctions1[5].set('number', 6)
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[6].set('pdg_code', (22, self.mymodel))
        wavefunctions1[6].set('state', 'intermediate')
        wavefunctions1[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3],wavefunctions1[4]]))
        wavefunctions1[6].set('interaction_id', (7, self.mymodel))
        wavefunctions1[6].set('number', 7)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[5], wavefunctions1[1],
                          wavefunctions1[6]]),
             'number': 1})
        amplitude1.set('interaction_id', (4, self.mymodel))

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
        wavefunctions2[5].set('pdg_code', (-2, self.mymodel))
        wavefunctions2[5].set('state', 'intermediate')
        wavefunctions2[5].set('mothers', helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1],wavefunctions1[2]]))
        wavefunctions2[5].set('interaction_id', (3, self.mymodel))
        wavefunctions2[5].set('number', 6)
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[6].set('pdg_code', (22, self.mymodel))
        wavefunctions2[6].set('state', 'intermediate')
        wavefunctions2[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3],wavefunctions1[4]]))
        wavefunctions2[6].set('interaction_id', (7, self.mymodel))
        wavefunctions2[6].set('number', 7)

        amplitude2 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions2[0], wavefunctions2[5],
                          wavefunctions2[6]]),
             'number': 2})
        amplitude2.set('interaction_id', (4, self.mymodel))

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1})

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2})

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
        goal = goal + "  ((1,2>1,id:7),(3,4>3,id:7),(1,3,id:0))\n"
        goal = goal + "  ((1,4>1,id:7),(2,3>2,id:7),(1,2,id:0))"

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
            myleglist[1], 7, self.mymodel))
        wavefunctions1[4].set('state', 'intermediate')
        wavefunctions1[4].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[1]]))
        wavefunctions1[4].set('number', 5)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[4], wavefunctions1[2],
                          wavefunctions1[3]]),
             'number': 1})
        amplitude1.set('interaction_id', (7, self.mymodel))

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1})

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            # -11 is what comes out of algorithm - don't know if right
            base_objects.Leg({'id': -11, 'state': 'initial'}),
                              7, self.mymodel))
        wavefunctions2[0].set('state', 'intermediate')
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
        amplitude2.set('interaction_id', (7, self.mymodel))

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2})

        mydiagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)
        
        self.assertEqual(matrix_element.get('diagrams'), mydiagrams)

    def test_generate_helas_diagrams_ea_ae(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes a e- > a e-
        """

        # Test a e- > a e-

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
        goal = goal + "  ((1,2>1,id:7),(3,4>3,id:7),(1,3,id:0))\n"
        goal = goal + "  ((1,3>1,id:7),(2,4>2,id:7),(1,2,id:0))"

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
            myleglist[0], 7, self.mymodel))
        wavefunctions1[4].set('state', 'intermediate')
        wavefunctions1[4].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0],wavefunctions1[1]]))
        wavefunctions1[4].set('number', 5)

        amplitude1 = helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[4], wavefunctions1[2],
                          wavefunctions1[3]]),
             'number': 1})
        amplitude1.set('interaction_id', (7, self.mymodel))

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitude': amplitude1})

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            # 11 is what comes out of algorithm - don't know if right
            base_objects.Leg({'id': 11, 'state': 'initial'}),
                              7, self.mymodel))
        wavefunctions2[0].set('state', 'intermediate')
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
        amplitude2.set('interaction_id', (7, self.mymodel))

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitude': amplitude2})
        mydiagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)
        
        self.assertEqual(matrix_element.get('diagrams'), mydiagrams)


#===============================================================================
# HelasParticleTest
#===============================================================================
class HelasParticleTest(unittest.TestCase):
    """Test class for the HelasParticle object"""

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

        self.mypart = helas_objects.HelasParticle(self.mydict)

    def test_setget_helas_particle_correct(self):
        "Test correct HelasParticle object __init__, get and set"

        mypart2 = helas_objects.HelasParticle()

        # First fill mypart2 it using set
        for prop in self.mydict.keys():
            mypart2.set(prop, self.mydict[prop])

        # Check equality between HelasParticle objects
        self.assertEqual(self.mypart, mypart2)

        # Check equality with initial dic using get
        for prop in self.mypart.keys():
            self.assertEqual(self.mypart.get(prop), self.mydict[prop])

    def test_setget_helas_particle_exceptions(self):
        "Test error raising in HelasParticle __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasParticle.PhysicsObjectError,
                          helas_objects.HelasParticle,
                          wrong_dict)
        self.assertRaises(helas_objects.HelasParticle.PhysicsObjectError,
                          helas_objects.HelasParticle,
                          a_number)

        # Test get
        self.assertRaises(helas_objects.HelasParticle.PhysicsObjectError,
                          self.mypart.get,
                          a_number)
        self.assertRaises(helas_objects.HelasParticle.PhysicsObjectError,
                          self.mypart.get,
                          'wrongparam')

        # Test set
        self.assertRaises(helas_objects.HelasParticle.PhysicsObjectError,
                          self.mypart.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasParticle.PhysicsObjectError,
                          self.mypart.set,
                          'wrongparam', 0)

    def test_initialize_with_base_object_particle(self):
        """Test initialization with instance of mother class"""

        mybaseparticle = base_objects.Particle(self.mydict)
        myparticle = helas_objects.HelasParticle(mybaseparticle)
        self.assertEqual(myparticle, self.mypart)

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
                        'right_list':[1, 12, 80000000, -1],
                        'wrong_list':[1.2, 'a']},
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
        """Test the get_pdg_code function of HelasParticle"""

        test_part = copy.copy(self.mypart)
        self.assertEqual(test_part.get_pdg_code(), 6)
        test_part.set('is_part', False)
        self.assertEqual(test_part.get_pdg_code(), -6)
        test_part.set('self_antipart', True)
        self.assertEqual(test_part.get_pdg_code(), 6)

    def test_get_anti_pdg_code(self):
        """Test the get_anti_pdg_code function of HelasParticle"""

        test_part = copy.copy(self.mypart)
        self.assertEqual(test_part.get_anti_pdg_code(), -6)
        test_part.set('is_part', False)
        self.assertEqual(test_part.get_anti_pdg_code(), 6)
        test_part.set('self_antipart', True)
        self.assertEqual(test_part.get_pdg_code(), 6)

    def test_set_wavefunction_list(self):
        """Test setting the static Helas wavefunction call dictionnary"""

        self.assertEqual(self.mypart.get('wavefunctions'),{})

    def test_helas_particle_list(self):
        """Test particle list initialization, search and dict generation
        functions."""

        mylist = [self.mypart] * 10
        mypartlist = helas_objects.HelasParticleList(mylist)

        not_a_part = 1

        for part in mypartlist:
            self.assertEqual(part, self.mypart)

        self.assertRaises(helas_objects.HelasParticleList.PhysicsObjectListError,
                          mypartlist.append,
                          not_a_part)
        
        not_a_part = base_objects.Particle()
        self.assertRaises(helas_objects.HelasParticleList.PhysicsObjectListError,
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

        mydict = {6:self.mypart, -6:anti_part}

        self.assertEqual(mydict, mypartlist.generate_dict())

        # test initialization with base_objects.ParticleList
        mybasepart = base_objects.Particle(self.mydict)
        mybaselist = [mybasepart] * 10
        self.assertRaises(helas_objects.HelasParticleList.PhysicsObjectListError,
                          helas_objects.HelasParticleList,
                          mybaselist)
        mybaselist = base_objects.ParticleList(mybaselist)
        self.assertEqual(helas_objects.HelasParticleList(mybaselist),
                         mypartlist)
        

#===============================================================================
# HelasInteractionTest
#===============================================================================
class HelasInteractionTest(unittest.TestCase):
    """Test class for the interaction object."""

    mydict = {}
    myinter = None
    mypart = None

    def setUp(self):

        self.mypart = helas_objects.HelasParticle({'name':'t',
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
                       'particles': helas_objects.HelasParticleList([self.mypart] * 4),
                       'color': ['C1', 'C2'],
                       'lorentz':['L1', 'L2'],
                       'couplings':{(0, 0):'g00',
                                    (0, 1):'g01',
                                    (1, 0):'g10',
                                    (1, 1):'g11'},
                       'orders':{'QCD':1, 'QED':1}}

        self.myinter = helas_objects.HelasInteraction(self.mydict)

    def test_setget_helas_interaction_correct(self):
        "Test correct interaction object __init__, get and set"

        myinter2 = helas_objects.HelasInteraction()

        # First fill myinter2 it using set
        for prop in ['id', 'particles', 'color', 'lorentz', 'couplings', 'orders']:
            myinter2.set(prop, self.mydict[prop])

        # Check equality between HelasInteraction objects
        self.assertEqual(self.myinter, myinter2)

        # Check equality with initial dic using get
        for prop in self.myinter.keys():
            self.assertEqual(self.myinter.get(prop), self.mydict[prop])

    def test_setget_helas_interaction_exceptions(self):
        "Test error raising in HelasInteraction __init__, get and set"

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasInteraction.PhysicsObjectError,
                          helas_objects.HelasInteraction,
                          wrong_dict)
        self.assertRaises(helas_objects.HelasInteraction.PhysicsObjectError,
                          helas_objects.HelasInteraction,
                          a_number)

        # Test get
        self.assertRaises(helas_objects.HelasInteraction.PhysicsObjectError,
                          self.myinter.get,
                          a_number)
        self.assertRaises(helas_objects.HelasInteraction.PhysicsObjectError,
                          self.myinter.get,
                          'wrongparam')

        # Test set
        self.assertRaises(helas_objects.HelasInteraction.PhysicsObjectError,
                          self.myinter.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasInteraction.PhysicsObjectError,
                          self.myinter.set,
                          'wrongparam', 0)

    def test_initialize_with_base_object_interaction(self):
        """Test initialization with instance of mother class"""

        mybaseinter = base_objects.Interaction(self.mydict)
        myinter = helas_objects.HelasInteraction(mybaseinter)
        self.assertEqual(myinter, self.myinter)

    def test_values_for_prop(self):
        """Test filters for interaction properties"""

        test_values = [
                       {'prop':'particles',
                        'right_list':[helas_objects.HelasParticleList([]),
                                      helas_objects.HelasParticleList([self.mypart] * 3)],
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
        goal = goal + "    \'id\': %d,\n" % self.myinter['id']
        goal = goal + "    \'particles\': %s,\n" % \
                            repr(helas_objects.HelasParticleList([self.mypart] * 4))
        goal = goal + "    \'color\': [\'C1\', \'C2\'],\n"
        goal = goal + "    \'lorentz\': [\'L1\', \'L2\'],\n"
        goal = goal + "    \'couplings\': %s,\n" % \
                                    repr(self.myinter['couplings'])
        goal = goal + "    \'orders\': %s\n}" % repr(self.myinter['orders'])

        self.assertEqual(goal, str(self.myinter))

    def test_interaction_list(self):
        """Test interaction list initialization"""

        # Create a dummy list of interactions with ids
        mylist = [copy.copy(inter) for inter in [self.myinter] * 3]
        for i in range(1, 4):
            mylist[i - 1].set('id', i)
        myinterlist = helas_objects.HelasInteractionList(mylist)

        # Check error raising
        not_a_inter = 1
        self.assertRaises(helas_objects.HelasInteractionList.PhysicsObjectListError,
                          myinterlist.append,
                          not_a_inter)

        # Check error raising
        not_a_inter = base_objects.Interaction()
        self.assertRaises(helas_objects.HelasInteractionList.PhysicsObjectListError,
                          myinterlist.append,
                          not_a_inter)

        # Check reference dict
        mydict = {}
        for i in range(1, 4):
            mydict[i] = myinterlist[i - 1]
        self.assertEqual(mydict, myinterlist.generate_dict())

        # test initialization with base_objects.InteractionList
        mybaseinter = base_objects.Interaction(self.mydict)
        mybaselist = [copy.copy(inter) for inter in [mybaseinter] * 3]
        for i in range(1, 4):
            mybaselist[i - 1].set('id', i)

        self.assertRaises(helas_objects.HelasInteractionList.PhysicsObjectListError,
                          helas_objects.HelasInteractionList,
                          mybaselist)
        mybaselist = base_objects.InteractionList(mybaselist)
        self.assertEqual(helas_objects.HelasInteractionList(mybaselist),
                         myinterlist)
        

#===============================================================================
# HelasModelTest
#===============================================================================
class HelasModelTest(unittest.TestCase):
    """Test class for the HelasModel object"""

    mymodel = helas_objects.HelasModel()
    myinterlist = helas_objects.HelasInteractionList()
    mypartlist = helas_objects.HelasParticleList()

    def setUp(self):

        # Create a model with gluon and top quark + a single interaction
        self.mypartlist.append(helas_objects.HelasParticle({'name':'t',
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
        self.mypartlist.append(helas_objects.HelasParticle({'name':'g',
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

        self.myinterlist.append(helas_objects.HelasInteraction({
                      'id':1,
                      'particles': helas_objects.HelasParticleList(\
                                            [self.mypartlist[0], \
                                             antit, \
                                             self.mypartlist[1]]),
                      'color': ['C1'],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('interactions', self.myinterlist)
        self.mymodel.set('particles', self.mypartlist)

    def test_helas_model_initialization(self):
        """Test the default HelasModel class initialization"""
        mymodel = helas_objects.HelasModel()

        self.assertEqual(mymodel['particles'],
                         helas_objects.HelasParticleList())
        self.assertEqual(mymodel['interactions'],
                         helas_objects.HelasInteractionList())

    def test_setget_helas_model_correct(self):
        """Test correct HelasModel object get and set"""

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

        mypart = helas_objects.HelasParticle(mydict)
        mypartlist = helas_objects.HelasParticleList([mypart])
        mymodel = helas_objects.HelasModel()
        mymodel.set('particles', mypartlist)

        self.assertEqual(mymodel.get('particles'), mypartlist)

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

        # For each subclass
        self.assertFalse(mymodel.set('particles', not_a_string))
        self.assertFalse(mymodel.set('interactions', not_a_string))

    def test_dictionaries(self):
        """Test particle dictionary in HelasModel"""

        antitop = copy.copy(self.mypartlist[0])
        antitop.set('is_part', False)
        mypartdict = {6:self.mypartlist[0], -6:antitop, 21:self.mypartlist[1]}
        self.assertEqual(mypartdict, self.mymodel.get('particle_dict'))

        myinterdict = {1:self.myinterlist[0]}
        self.assertEqual(myinterdict, self.mymodel.get('interaction_dict'))

