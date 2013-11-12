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
from madgraph.iolibs import helas_call_writers

"""Unit test library for the helas_objects module"""

import copy

import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color
import madgraph.iolibs.export_v4 as export_v4
import models.import_ufo as import_ufo

#===============================================================================
# HelasWavefunctionTest
#===============================================================================
class HelasWavefunctionTest(unittest.TestCase):
    """Test class for the HelasWavefunction object"""

    mydict = {}
    mywavefunction = None
    mymothers = helas_objects.HelasWavefunctionList()

    def test_setget_wavefunction_exceptions(self):
        "Test error raising in HelasWavefunction __init__, get and set"

        mywavefunction = helas_objects.HelasWavefunction()

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          helas_objects.HelasWavefunction,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          helas_objects.HelasWavefunction,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          mywavefunction.get,
                          a_number)
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          mywavefunction.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          mywavefunction.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasWavefunction.PhysicsObjectError,
                          mywavefunction.set,
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

        temp_wavefunction = helas_objects.HelasWavefunction()

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_wavefunction.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_wavefunction.set(test['prop'], x))


#===============================================================================
# HelasAmplitudeTest
#===============================================================================
class HelasAmplitudeTest(unittest.TestCase):
    """Test class for the HelasAmplitude object"""

    mydict = {}
    myamplitude = None
    mywavefunctions = None

    def test_setget_amplitude_exceptions(self):
        "Test error raising in HelasAmplitude __init__, get and set"

        myamplitude = helas_objects.HelasAmplitude()

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          helas_objects.HelasAmplitude,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          helas_objects.HelasAmplitude,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          myamplitude.get,
                          a_number)
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          myamplitude.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          myamplitude.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasAmplitude.PhysicsObjectError,
                          myamplitude.set,
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
                       {'prop':'fermionfactor',
                        'right_list':[-1, 1, 0],
                        'wrong_list':['a', {}, 0.]}
                       ]

        temp_amplitude = helas_objects.HelasAmplitude()

        for test in test_values:
            for x in test['right_list']:
                self.assert_(temp_amplitude.set(test['prop'], x))
            for x in test['wrong_list']:
                self.assertFalse(temp_amplitude.set(test['prop'], x))

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

#===============================================================================
# HelasDiagramTest
#===============================================================================
class HelasDiagramTest(unittest.TestCase):
    """Test class for the HelasDiagram object"""

    mydict = {}
    mywavefunctions = None
    myamplitude = None
    mydiagram = None

    def test_setget_diagram_exceptions(self):
        "Test error raising in HelasDiagram __init__, get and set"

        mydiagram = helas_objects.HelasDiagram()

        wrong_dict = self.mydict
        wrong_dict['wrongparam'] = 'wrongvalue'

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          helas_objects.HelasDiagram,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          helas_objects.HelasDiagram,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          mydiagram.get,
                          a_number)
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          mydiagram.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          mydiagram.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasDiagram.PhysicsObjectError,
                          mydiagram.set,
                          'wrongparam', 0)

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

        g = mypartlist[-1]

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
        u = mypartlist[-1]
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
        eminus = mypartlist[-1]
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
        a = mypartlist[-1]


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
        seminus = mypartlist[-1]
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
        n1 = mypartlist[-1]

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             a]),
                      'color': [color.ColorString([color.T(0, 1)])],
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
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        # Gluon self-couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g,
                                             g]),
                      'color': [color.ColorString([color.f(0, 1, 2)]),
                                color.ColorString([color.f(1, 2, 0)]),
                                color.ColorString([color.f(2, 0, 1)])],
                      'lorentz':['gggg1', 'gggg2', 'gggg3'],
                      'couplings':{(0, 0):'GG',(1, 1):'GG',(2, 2):'GG'},
                      'orders':{'QCD':2}}))


        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

    def test_setget_matrix_element_exceptions(self):
        "Test error raising in HelasMatrixElement __init__, get and set"

        wrong_dict = {}
        wrong_dict['wrongparam'] = 'wrongvalue'

        mymatrixelement = helas_objects.HelasMatrixElement()

        a_number = 0

        # Test init
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          helas_objects.HelasMatrixElement,
                          wrong_dict)
        self.assertRaises(AssertionError,
                          helas_objects.HelasMatrixElement,
                          a_number)

        # Test get
        self.assertRaises(AssertionError,
                          mymatrixelement.get,
                          a_number)
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          mymatrixelement.get,
                          'wrongparam')

        # Test set
        self.assertRaises(AssertionError,
                          mymatrixelement.set,
                          a_number, 0)
        self.assertRaises(helas_objects.HelasMatrixElement.PhysicsObjectError,
                          mymatrixelement.set,
                          'wrongparam', 0)

#    def test_representation(self):
#        """Test matrix_element object string representation."""
#
#        goal = "{\n"
#        goal = goal + "    \'processes\': [],\n"
#        goal = goal + "    \'diagrams\': " + repr(self.mydiagrams) + "\n}"
#
#        self.assertEqual(goal, str(self.mymatrixelement))


    def test_process_init(self):
        """Testing the process initialization using the process
        e- e+ > e- e+
        """

        # Test e+ e- > e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        self.assertEqual(matrix_element.get('processes')[0],
                         myamplitude.get('process'))

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
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        matrix_element = helas_objects.HelasMatrixElement()
        matrix_element.set('processes', base_objects.ProcessList([ myproc ]))
        matrix_element.calculate_identical_particle_factor()

        self.assertEqual(matrix_element.get_denominator_factor(), 9 * 4 * 6)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        matrix_element = helas_objects.HelasMatrixElement()
        matrix_element.set('processes', base_objects.ProcessList([ myproc ]))
        matrix_element.calculate_identical_particle_factor()

        self.assertEqual(matrix_element.get_denominator_factor(), 1 * 6 * 6)

    def test_fermionfactor_emep_emep(self):
        """Testing the fermion factor using the process  e- e+ > e- e+
        """

        # Test e+ e- > e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        myamplitude.get('diagrams')

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        diagrams = matrix_element.get('diagrams')

        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[1].get('amplitudes')[0].get('fermionfactor'), -1)

    def test_fermionfactor_emep_emepa(self):
        """Testing the fermion factor using the process  e- e+ > e- e+ a
        """

        # Test e+ e- > e+ e- a

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        myamplitude.get('diagrams')

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        diagrams = matrix_element.get('diagrams')

        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[1].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[2].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[3].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[4].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[5].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[6].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[7].get('amplitudes')[0].get('fermionfactor'), 1)

    def test_fermionfactor_emep_emepemep(self):
        """Testing the fermion factor using the process e- e+ > e- e+ e- e+
        Time estimates for e+e->e+e-e+e-e+e- (1728 diagrams):
        Diagram generation: 18 s
        Helas call generation (with optimization): 58 s
        Helas call generation (without optimization): 23 s
        Fermion factor calculation: 0 s
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

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

        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[1].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[2].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[3].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[4].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[5].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[6].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[7].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[8].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[9].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[10].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[11].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[12].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[13].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[14].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[15].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[16].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[17].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[18].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[19].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[20].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[21].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[22].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[23].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[24].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[25].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[26].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[27].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[28].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[29].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[30].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[31].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[32].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[33].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[34].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[35].get('amplitudes')[0].get('fermionfactor'), -1)

    def test_fermionfactor_epem_sepsemepem(self):
        """Testing the fermion factor using the process e+ e- > se+ se- e+ e-
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
        eminus = mypartlist[-1]
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
        seminus = mypartlist[-1]
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
        n1 = mypartlist[-1]

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        myamplitude.get('diagrams')

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        diagrams = matrix_element.get('diagrams')

        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[1].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[2].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[3].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[4].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[5].get('amplitudes')[0].get('fermionfactor'), 1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[6].get('amplitudes')[0].get('fermionfactor'), -1)
        self.assertEqual(diagrams[0].get('amplitudes')[0].get('fermionfactor') * \
                         diagrams[7].get('amplitudes')[0].get('fermionfactor'), 1)

    def test_generate_helas_diagrams_uux_gepem(self):
        """Testing the helas diagram generation u u~ > g e+ e-
        """

        # Test u u~ > g e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "1  ((1(-2),3(21)>1(-2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4)) (QCD=1,QED=2,WEIGHTED=5)\n"
        goal = goal + "2  ((2(2),3(21)>2(2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4)) (QCD=1,QED=2,WEIGHTED=5)"

        self.assertEqual(goal,
                         myamplitude.get('diagrams').nice_string())

        wavefunctions1 = helas_objects.HelasWavefunctionList()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions1[-1].flip_part_antipart()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions1[-1].flip_part_antipart()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[4], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[5].set('particle', -2, self.mymodel)
        wavefunctions1[5].set('number_external', 1)
        wavefunctions1[5].set('state', 'incoming')
        wavefunctions1[5].set('is_part',
                              False)
        wavefunctions1[5].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions1[2]]))
        wavefunctions1[5].set('interaction_id', 3, self.mymodel)
        wavefunctions1[5].set('number', 6)
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[6].set('particle', 22, self.mymodel)
        wavefunctions1[6].set('number_external', 4)
        wavefunctions1[6].set('state', 'intermediate')
        wavefunctions1[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3], wavefunctions1[4]]))
        wavefunctions1[6].set('interaction_id', 7, self.mymodel)
        wavefunctions1[6].set('number', 7)

        amplitude1 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[5], wavefunctions1[1],
                          wavefunctions1[6]]),
             'number': 1,
             'fermionfactor':-1})])
        amplitude1[0].set('interaction_id', 4, self.mymodel)

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[0].set('particle', 2, self.mymodel)
        wavefunctions2[0].set('number_external', 2)
        wavefunctions2[0].set('state', 'outgoing')
        wavefunctions2[0].set('is_part', True)
        wavefunctions2[0].set('mothers', helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1], wavefunctions1[2]]))
        wavefunctions2[0].set('interaction_id', 3, self.mymodel)
        wavefunctions2[0].set('number', 8)

        amplitude2 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions2[0],
                          wavefunctions1[6]]),
             'number': 2,
             'fermionfactor':-1})])
        amplitude2[0].set('interaction_id', 4, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitudes': amplitude1,
                                               'number': 1})

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitudes': amplitude2,
                                               'number': 2})

        diagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(\
            myamplitude,
            1)

        self.assertEqual(matrix_element.get('diagrams'), diagrams)

    def test_generate_helas_diagrams_uux_gepem_no_optimization(self):
        """Testing the helas diagram generation u u~ > g e+ e-
        """
        # Test u u~ > g e+ e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "1  ((1(-2),3(21)>1(-2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4)) (QCD=1,QED=2,WEIGHTED=5)\n"
        goal = goal + "2  ((2(2),3(21)>2(2),id:3),(4(11),5(-11)>4(22),id:7),(1(-2),2(2),4(22),id:4)) (QCD=1,QED=2,WEIGHTED=5)"

        self.assertEqual(goal,
                         myamplitude.get('diagrams').nice_string())

        wavefunctions1 = helas_objects.HelasWavefunctionList()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions1[-1].flip_part_antipart()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions1[-1].flip_part_antipart()
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction(\
            myleglist[4], 0, self.mymodel))
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[5].set('particle', -2, self.mymodel)
        wavefunctions1[5].set('number_external', 1)
        wavefunctions1[5].set('state', 'incoming')
        wavefunctions1[5].set('is_part', False)
        wavefunctions1[5].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions1[2]]))
        wavefunctions1[5].set('interaction_id', 3, self.mymodel)
        wavefunctions1[5].set('number', 6)
        wavefunctions1.append(helas_objects.HelasWavefunction())
        wavefunctions1[6].set('particle', 22, self.mymodel)
        wavefunctions1[6].set('number_external', 4)
        wavefunctions1[6].set('state', 'intermediate')
        wavefunctions1[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3], wavefunctions1[4]]))
        wavefunctions1[6].set('interaction_id', 7, self.mymodel)
        wavefunctions1[6].set('number', 7)

        amplitude1 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[5], wavefunctions1[1],
                          wavefunctions1[6]]),
             'number': 1,
             'fermionfactor':-1})])
        amplitude1[0].set('interaction_id', 4, self.mymodel)

        wavefunctions2 = helas_objects.HelasWavefunctionList()
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[0], 0, self.mymodel))
        wavefunctions2[-1].flip_part_antipart()
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[1], 0, self.mymodel))
        wavefunctions2[-1].flip_part_antipart()
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[2], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[3], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction(\
            myleglist[4], 0, self.mymodel))
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[5].set('particle', 2, self.mymodel)
        wavefunctions2[5].set('number_external', 2)
        wavefunctions2[5].set('state', 'outgoing')
        wavefunctions2[5].set('is_part', True)
        wavefunctions2[5].set('mothers', helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1], wavefunctions1[2]]))
        wavefunctions2[5].set('interaction_id', 3, self.mymodel)
        wavefunctions2[5].set('number', 6)
        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[6].set('particle', 22, self.mymodel)
        wavefunctions2[6].set('number_external', 4)
        wavefunctions2[6].set('state', 'intermediate')
        wavefunctions2[6].set('mothers', helas_objects.HelasWavefunctionList(
                         [wavefunctions1[3], wavefunctions1[4]]))
        wavefunctions2[6].set('interaction_id', 7, self.mymodel)
        wavefunctions2[6].set('number', 7)

        amplitude2 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions2[0], wavefunctions2[5],
                          wavefunctions2[6]]),
             'number': 2,
             'fermionfactor':-1})])
        amplitude2[0].set('interaction_id', 4, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitudes': amplitude1,
                                               'number': 1})

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitudes': amplitude2,
                                               'number': 2})

        diagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(\
            myamplitude,
            0)
        
        self.assertEqual(matrix_element.get('diagrams')[0]['amplitudes'], amplitude1, 
                         '%s != %s' %(matrix_element.get('diagrams')[0]['amplitudes'], amplitude1))
        #self.assertEqual(matrix_element.get('diagrams'), diagrams)

    def test_generate_helas_diagrams_ae_ae(self):
        """Testing the helas diagram generation a e- > a e-
        """

        # Test a e- > a e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "1  ((1(22),2(-11)>1(-11),id:7),(3(22),4(11),1(-11),id:7)) (QCD=0,QED=2,WEIGHTED=4)\n"
        goal = goal + "2  ((1(22),4(11)>1(11),id:7),(2(-11),3(22),1(11),id:7)) (QCD=0,QED=2,WEIGHTED=4)"

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
        wavefunctions1[4].set('particle', 11, self.mymodel)
        wavefunctions1[4].set('interaction_id', 7, self.mymodel)
        wavefunctions1[4].set('state', 'incoming')
        wavefunctions1[4].set('number_external', 1)
        wavefunctions1[4].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions1[1]]))
        wavefunctions1[4].set('number', 5)

        amplitude1 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[2], wavefunctions1[3],
                          wavefunctions1[4]]),
             'number': 1,
             'color_indices': [0, 0],
             'fermionfactor': 1})])
        amplitude1[0].set('interaction_id', 7, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitudes': amplitude1,
                                               'number': 1})

        wavefunctions2 = helas_objects.HelasWavefunctionList()

        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[0].set('particle', -11, self.mymodel)
        wavefunctions2[0].set('interaction_id', 7, self.mymodel)
        wavefunctions2[0].set('state', 'outgoing')
        wavefunctions2[0].set('number_external', 1)
        wavefunctions2[0].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions1[3]]))
        wavefunctions2[0].set('number', 6)

        amplitude2 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1], wavefunctions1[2],
                          wavefunctions2[0]]),
             'interaction_id': 7,
             'number': 2,
             'color_indices': [0, 0],
             'fermionfactor': 1})])
        amplitude2[0].set('interaction_id', 7, self.mymodel)

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitudes': amplitude2,
                                               'number': 2})

        mydiagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        self.assertEqual(matrix_element.get('diagrams'), mydiagrams)

    def test_generate_helas_diagrams_ea_ae(self):
        """Testing the helas diagram generation e- a > a e-
        """

        # Test e- a > a e-

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        goal = "2 diagrams:\n"
        goal = goal + "1  ((1(-11),2(22)>1(-11),id:7),(3(22),4(11),1(-11),id:7)) (QCD=0,QED=2,WEIGHTED=4)\n"
        goal = goal + "2  ((1(-11),3(22)>1(-11),id:7),(2(22),4(11),1(-11),id:7)) (QCD=0,QED=2,WEIGHTED=4)"

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
        wavefunctions1[4].set('particle', 11, self.mymodel)
        wavefunctions1[4].set('interaction_id', 7, self.mymodel)
        wavefunctions1[4].set('number_external', 1)
        wavefunctions1[4].set('state', 'incoming')
        wavefunctions1[4].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions1[1]]))
        wavefunctions1[4].set('number', 5)

        amplitude1 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[2], wavefunctions1[3],
                          wavefunctions1[4]]),
             'number': 1,
             'color_indices': [0, 0],
             'fermionfactor': 1})])
        amplitude1[0].set('interaction_id', 7, self.mymodel)

        diagram1 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions1,
                                               'amplitudes': amplitude1,
                                               'number': 1})

        wavefunctions2 = helas_objects.HelasWavefunctionList()

        wavefunctions2.append(helas_objects.HelasWavefunction())
        wavefunctions2[0].set('particle', 11, self.mymodel)
        wavefunctions2[0].set('interaction_id', 7, self.mymodel)
        wavefunctions2[0].set('number_external', 1)
        wavefunctions2[0].set('state', 'incoming')
        wavefunctions2[0].set('mothers',
                              helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[0], wavefunctions1[2]]))
        wavefunctions2[0].set('number', 6)

        amplitude2 = helas_objects.HelasAmplitudeList([\
            helas_objects.HelasAmplitude({\
             'mothers': helas_objects.HelasWavefunctionList(\
                         [wavefunctions1[1], wavefunctions1[3],
                          wavefunctions2[0]]),
             'interaction_id': 7,
             'number': 2,
             'color_indices': [0, 0],
             'fermionfactor': 1})])
        amplitude2[0].set('interaction_id', 7, self.mymodel)

        diagram2 = helas_objects.HelasDiagram({'wavefunctions': wavefunctions2,
                                               'amplitudes': amplitude2,
                                               'number': 2})
        mydiagrams = helas_objects.HelasDiagramList([diagram1, diagram2])

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        self.assertEqual(matrix_element.get('diagrams'), mydiagrams)

    def test_generate_helas_diagrams_4g(self):
        """Testing the helas diagram generation g g > g g and g g > g g g
        """

        # Test g g > g g 

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 4)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude,
                                                          gen_color=False)

        self.assertEqual([len(diagram.get('amplitudes')) for diagram in \
                          matrix_element.get('diagrams')],
                         [3, 1, 1, 1])

        # Test g g > g g g

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 25)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude,
                                                          gen_color=False)

        self.assertEqual([len(diagram.get('amplitudes')) for diagram in \
                          matrix_element.get('diagrams')],
                         [1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3,
                          1, 1, 1, 3, 3, 3, 3, 3, 3])

        return

        # Test g g > g g g g

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 220)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude,
                                                          gen_color=False)

        self.assertEqual(sum([len(diagram.get('amplitudes')) for diagram in \
                          matrix_element.get('diagrams')]), 510)

    def test_get_base_amplitude(self):
        """Testing the functions get_base_amplitude() in HelasMatrixElement,
           and especially the case where the order of the particles in the
           base_objects.Diagram which used to generate de HelasDiagram is not
           the same as in the interaction. In this case, the 'color_indices'
           attribute of the HelasAmplitude does not necessarly matches the
           list of color_keys generated by the color_amp module. In fact, it does
           but only thanks to the fact that 'color_indices' is overwritten and
           re-initialized using get_color_indices() after HelasDiagram generation"""
        # work on a copy of the model not to affect the other tests
        localmodel = copy.deepcopy(self.mymodel)
        
        # Add three new kinds of gluons.
        
        g = localmodel.get('particles')[0] 
        
        ga = base_objects.Particle({'name':'ga',
                      'antiname':'ga',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'ga',
                      'antitexname':'ga',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':1021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True})
        localmodel.get('particles').append(ga)
        
        gb = base_objects.Particle({'name':'gb',
                      'antiname':'gb',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'gb',
                      'antitexname':'gb',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':2021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True})    
        localmodel.get('particles').append(gb)

        gc = base_objects.Particle({'name':'gc',
                      'antiname':'gc',
                      'spin':3,
                      'color':8,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'gc',
                      'antitexname':'gc',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':3021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True})
        localmodel.get('particles').append(gc)
        
        # Add g ga gb gc, g ga gb and g ga gc interactions

        localmodel.get('interactions').append(base_objects.Interaction({
                      'id': 1008,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             ga, \
                                             gb]),
                      'color': [color.ColorString([color.f(0, 1, 2)]),
                                color.ColorString([color.d(0, 1, 2)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG',(1,0):'GG'},
                      'orders':{'QCD':1}}))

        localmodel.get('interactions').append(base_objects.Interaction({
                      'id': 2008,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             ga, \
                                             gc]),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        localmodel.get('interactions').append(base_objects.Interaction({
                      'id': 1009,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             ga, \
                                             gb,
                                             gc]),
                      'color': [color.ColorString([color.f(0, 1, 2)]),
                                color.ColorString([color.f(1, 2, 0)]),
                                color.ColorString([color.f(2, 0, 1)])],
                      'lorentz':['gggg1', 'gggg2', 'gggg3'],
                      'couplings':{(0, 0):'GG',(1, 1):'GG',(2, 2):'GG'},
                      'orders':{'QCD':2}}))        


        # Reinitialize the model dictionaries
        localmodel.reset_dictionaries()
        localmodel.actualize_dictionaries()
        localmodel.set('particles',localmodel.get('particles'))        
        localmodel.set('interactions',localmodel.get('interactions'))
        
        ### Test with a single diagram g ga > ga ga g g
        
        leglist = base_objects.LegList()
        leglist.append(base_objects.Leg({'id':21,'state':True,'number':1}))
        leglist.append(base_objects.Leg({'id':1021,'state':True,'number':2}))
        leglist.append(base_objects.Leg({'id':1021,'state':False,'number':3}))
        leglist.append(base_objects.Leg({'id':1021,'state':False,'number':4}))
        leglist.append(base_objects.Leg({'id':21,'state':False,'number':5}))
        leglist.append(base_objects.Leg({'id':21,'state':False,'number':6}))
        myproc=base_objects.Process({'legs':leglist,
                                     'model':localmodel})
        
        l1=leglist[0]
        l2=leglist[1]
        l3=leglist[2]
        l4=leglist[3]
        l5=leglist[4]
        l6=leglist[5]

        # One way of constructing this diagram, with a four-point vertex followed
        # by two three-point vertex decay.
        l36 = base_objects.Leg({'id':2021,'number':3})
        l45 = base_objects.Leg({'id':3021,'number':4})

        vx36 = base_objects.Vertex({'legs':base_objects.LegList([l3, l6, l36]), 'id': 1008})
        vx45 = base_objects.Vertex({'legs':base_objects.LegList([l5, l4, l45]), 'id': 2008})
        vx1234 = base_objects.Vertex({'legs':base_objects.LegList([l1, l2, l45, l36]), 'id': 1009})

        myVertexList=base_objects.VertexList([vx45,vx36,vx1234])

        trialDiag=base_objects.Diagram({'vertices':myVertexList})
        
        
        myamp=diagram_generation.Amplitude({'process':myproc,'diagrams':base_objects.DiagramList([trialDiag])})
        
        myME = helas_objects.HelasMatrixElement(myamp)
        #print "orig diag=",trialDiag.nice_string()
        #print "reconstruct=",myME['base_amplitude']['diagrams'][0].nice_string()
        
        # Now make sure that the color key corresponding to  the three-point
        # vertex with only one color structure is always 0.
        sum=0
        for i, key in enumerate(myME.get('color_basis')):
            for diag_tuple in myME.get('color_basis')[key]:
                sum=sum+diag_tuple[1][1]
        for j, amp in enumerate(myME.get_all_amplitudes()):
            sum=sum+tuple(amp.get('color_indices'))[1]
        
        self.assertEqual(sum,0)

    def test_helas_forbidden_s_channel_uux_uuxng(self):
        """Test helas diagrams with forbidden s-channel particles."""
        goal_no_photon = [4, 18]
        photon_none = [{1:[0]},{2:[1],4:[1],13:[0],17:[0]}]
        goal_no_quark = [2, 6]
        quark_none = [{0:[0]},{0:[0,1],1:[0,1],3:[0],5:[0]}]

        for ngluons in range(2):

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':-2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_onsh_s_channels':[22]})

            myamplitude = diagram_generation.Amplitude(myproc)

            helas_amplitude = helas_objects.HelasMatrixElement(myamplitude)
            
            self.assertEqual(len(helas_amplitude.get('diagrams')),
                             goal_no_photon[ngluons])

            #print myamplitude.nice_string()

            diagrams = helas_amplitude.get('diagrams')
            for idiag in range(len(diagrams)):
                if idiag in photon_none[ngluons]:
                    vertices, tchannels = \
                       diagrams[idiag].get('amplitudes')[0].get_s_and_t_channels(2, self.mymodel, 20)
                    for ivert in range(len(vertices)):
                        if ivert in photon_none[ngluons][idiag]:
                            self.assertEqual(False,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                        else:
                            self.assertEqual(None,
                                    vertices[ivert].get('legs')[-1].get('onshell'))

            # Test with u a > u a (+ g)

            myleglist = base_objects.LegList()

            myleglist.append(base_objects.Leg({'id':2,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':False}))
            myleglist.append(base_objects.Leg({'id':2,
                                             'state':True}))
            myleglist.append(base_objects.Leg({'id':22,
                                             'state':True}))
            myleglist.extend([base_objects.Leg({'id':21,
                                                 'state':True})] * ngluons)

            myproc = base_objects.Process({'legs':myleglist,
                                           'model':self.mymodel,
                                           'forbidden_onsh_s_channels':[2]})

            myamplitude = diagram_generation.Amplitude(myproc)

            helas_amplitude = helas_objects.HelasMatrixElement(myamplitude)
            
            self.assertEqual(len(helas_amplitude.get('diagrams')),
                             goal_no_quark[ngluons])

            #print helas_amplitude.nice_string()

            diagrams = helas_amplitude.get('diagrams')
            for idiag in range(len(diagrams)):
                if idiag in quark_none[ngluons]:
                    vertices, tchannels = \
                      diagrams[idiag].get('amplitudes')[0].get_s_and_t_channels(2, self.mymodel, 20)
                    for ivert in range(len(vertices)):
                        if ivert in quark_none[ngluons][idiag]:
                            self.assertEqual(False,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
                        else:
                            self.assertEqual(None,
                                    vertices[ivert].get('legs')[-1].get('onshell'))
            
    def test_sorted_mothers(self):
        """Testing the sorted_mothers routine
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A W
        mypartlist.append(base_objects.Particle({'name':'W+',
                      'antiname':'W-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'texname':'W^+',
                      'antitexname':'W^-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        Wplus = mypartlist[-1]
        Wminus = copy.copy(Wplus)
        Wminus.set('is_part', False)

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
        a = mypartlist[-1]

        # Z
        mypartlist.append(base_objects.Particle({'name':'Z',
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
        Z = mypartlist[-1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 3,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Wplus,
                                             Wminus]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX6'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             a]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX4'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 5,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             a, \
                                             Wplus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX7'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Z, \
                                             Wplus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX8'},
            'orders':{'QED':2}}))


        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':True,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True,
                                           'number': 5}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True,
                                           'number': 4}))

        mymothers = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(leg, 0, mybasemodel) for leg in myleglist[:4]])

        amplitude = helas_objects.HelasAmplitude()
        amplitude.set('interaction_id', 5, mybasemodel)
        amplitude.set('mothers', mymothers)
        self.assertEqual(helas_objects.HelasMatrixElement.sorted_mothers(amplitude),
                         [mymothers[2], mymothers[3], mymothers[0], mymothers[1]])
        mymothers = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(leg, 0, mybasemodel) for leg in myleglist[2:]])

        wavefunction = helas_objects.HelasWavefunction(myleglist[2],
                                                       4, mybasemodel)
        wavefunction.set('mothers', mymothers)
        self.assertEqual([m.get('pdg_code') for m in \
                          helas_objects.HelasMatrixElement.\
                          sorted_mothers(wavefunction)],
                         [mymothers[1].get('pdg_code'),
                          mymothers[0].get('pdg_code'),
                          mymothers[2].get('pdg_code')])

    def test_complicated_majorana_process(self):
        """Test majorana process z e- > n2 n2 e-
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[-1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # E sleptons and their antiparticle
        mypartlist.append(base_objects.Particle({'name':'el-',
                      'antiname':'el+',
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
        elminus = mypartlist[-1]
        elplus = copy.copy(elminus)
        elplus.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'er-',
                      'antiname':'er+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde e^-',
                      'antitexname':'\tilde e^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':2000011,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        erminus = mypartlist[-1]
        erplus = copy.copy(erminus)
        erplus.set('is_part', False)

        # Neutralinos
        mypartlist.append(base_objects.Particle({'name':'n2',
                      'antiname':'n2',
                      'spin':2,
                      'color':1,
                      'mass':'mn2',
                      'width':'wn2',
                      'texname':'\chi_0^2',
                      'antitexname':'\chi_0^2',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000023,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n2 = mypartlist[-1]

        # A z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'zmass',
                      'width':'zwidth',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[-1]

        # Coupling of e to Z
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             eminus, \
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZL'},
                      'orders':{'QED':1}}))

        # Coupling of n2 to e and el/er
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n2, \
                                             elminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN2M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             eminus, \
                                             elplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN2P'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n2, \
                                             erminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GERN2M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             eminus, \
                                             erplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GERN2P'},
                      'orders':{'QED':1}}))

        # Coupling of n2 to z
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             n2, \
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZN22'},
                      'orders':{'QED':1}}))

        # Coupling of el/er to z
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [z, \
                                             elminus, \
                                             elplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZELEL'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [z, \
                                             erminus, \
                                             erplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZERER'},
                      'orders':{'QED':1}}))


        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000023,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000023,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 24)

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        self.assertEqual(sum([len(diagram.get('amplitudes')) for diagram in \
                          me.get('diagrams')]), 24)

        for i, amp in enumerate(me.get_all_amplitudes()):
            self.assertEqual(amp.get('number'), i + 1)

        for i, wf in enumerate(me.get_all_wavefunctions()):
            self.assertEqual(wf.get('number'), i + 1)        

    def test_multi_amp_majorana_process(self):
        """Test fermion clash process x1+ x1+ > w- w- with multiple amps
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # x1+ and x1-
        mypartlist.append(base_objects.Particle({'name':'x1+',
                      'antiname':'x1-',
                      'spin':2,
                      'color':1,
                      'mass':'Mch1',
                      'width':'Wch1',
                      'charge':1.,
                      'pdg_code':1000024,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        x1plus = mypartlist[-1]
        x1minus = copy.copy(x1plus)
        x1minus.set('is_part', False)

        # W+ and W-
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[-1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # Neutralinos
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
        n1 = mypartlist[-1]

        # Coupling of n1 to w and x1
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [x1minus, \
                                             n1, \
                                             wplus]),
                      'color': [],
                      'lorentz':['FFV2', 'FFV3'],
                      'couplings':{(0, 0):'GC_666', (0, 1):'GC_416'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             x1plus, \
                                             wminus]),
                      'color': [],
                      'lorentz':['FFV2', 'FFV3'],
                      'couplings':{(0, 0):'GC_424', (0, 1):'GC_630'},
                      'orders':{'QED':1}}))


        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000024,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000024,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 2)

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        helas_writer = helas_call_writers.FortranUFOHelasCallWriter(mymodel)

        self.assertEqual(len(me.get_all_amplitudes()), 2)

        self.assertEqual(len(me.get_all_wavefunctions()), 6)

        for i, amp in enumerate(me.get_all_amplitudes()):
            self.assertEqual(amp.get('number'), i + 1)

        for i, wf in enumerate(me.get_all_wavefunctions()):
            self.assertEqual(wf.get('number'), i + 1)        


    def test_get_conjugate_index(self):
        """Test the get_conjugate_index routines"""

        myleg1 = base_objects.Leg({'id':2})
        myleg2 = base_objects.Leg({'id':2})
        myleg3 = base_objects.Leg({'id':-11})
        myleg4 = base_objects.Leg({'id':11})

        wf1 = helas_objects.HelasWavefunction(myleg1, 3, self.mymodel)
        wf2 = helas_objects.HelasWavefunction(myleg2, 0, self.mymodel)
        wf3 = helas_objects.HelasWavefunction(myleg3, 0, self.mymodel)
        wf4 = helas_objects.HelasWavefunction(myleg4, 0, self.mymodel)

        mothers = helas_objects.HelasWavefunctionList([wf2, wf3, wf4])

        wf1.set('mothers', mothers)
        wf1.set('pdg_codes', [-2,2,-11,11])
        
        self.assertEqual(wf1.get_spin_state_number(), 2)
        self.assertEqual(wf1.find_outgoing_number(), 1)
        self.assertEqual(wf1.get_conjugate_index(), ())

        wf1.set('fermionflow', -1)
        self.assertEqual(wf1.get_conjugate_index(), (1,))

        wf2.set('fermionflow', -1)
        self.assertEqual(wf1.get_conjugate_index(), (1,))

        wf1.set('fermionflow', 1)
        self.assertEqual(wf1.get_conjugate_index(), (1,))

        wf2.set('fermionflow', 1)
        wf3.set('fermionflow', -1)
        self.assertEqual(wf1.get_conjugate_index(), (2,))

        wf4.set('fermionflow', -1)
        self.assertEqual(wf1.get_conjugate_index(), (2,))

        wf3.set('fermionflow', 1)
        self.assertEqual(wf1.get_conjugate_index(), (2,))

        wf1.set('fermionflow', -1)
        self.assertEqual(wf1.get_conjugate_index(), (1, 2))

        myleg5 = base_objects.Leg({'id':2, 'state': False})
        wf5 = helas_objects.HelasWavefunction(myleg5, 0, self.mymodel)

        mothers.insert(0, wf5)
        wf4.set('fermionflow', 1)

        amp = helas_objects.HelasAmplitude()
        amp.set('mothers', mothers)

        self.assertEqual([w.is_fermion() for w in mothers], [True] * 4)
        self.assertEqual(amp.get_conjugate_index(), ())
        
        wf5.set('fermionflow', -1)
        self.assertEqual(amp.get_conjugate_index(), (1,))

        wf4.set('fermionflow', -1)
        self.assertEqual(amp.get_conjugate_index(), (1,2))

        wf5.set('fermionflow', 1)
        self.assertEqual(amp.get_conjugate_index(), (2,))

    def test_get_conjugate_index_majoranas(self):
        """Test the get_conjugate_index for Majoranas"""

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # x1+ and x1-
        mypartlist.append(base_objects.Particle({'name':'x1+',
                      'antiname':'x1-',
                      'spin':2,
                      'color':1,
                      'mass':'Mch1',
                      'width':'Wch1',
                      'charge':1.,
                      'pdg_code':1000024,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        x1plus = mypartlist[-1]
        x1minus = copy.copy(x1plus)
        x1minus.set('is_part', False)

        # W+, W- and Z
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[-1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'MZ',
                      'width':'WZ',
                      'charge':1.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        z = mypartlist[-1]

        # Neutralinos
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
        n1 = mypartlist[-1]

        mypartlist.append(base_objects.Particle({'name':'n2',
                      'antiname':'n2',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu2',
                      'width':'Wneu2',
                      'texname':'\chi_0^2',
                      'antitexname':'\chi_0^2',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000023,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n2 = mypartlist[-1]

        # Coupling of n1 to w and x1
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [x1minus, \
                                             n1, \
                                             wplus]),
                      'color': [],
                      'lorentz':['FFV2', 'FFV3'],
                      'couplings':{(0, 0):'GC_666', (0, 1):'GC_416'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             x1plus, \
                                             wminus]),
                      'color': [],
                      'lorentz':['FFV2', 'FFV3'],
                      'couplings':{(0, 0):'GC_424', (0, 1):'GC_630'},
                      'orders':{'QED':1}}))

        # Coupling of n1 to n2
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             n2, \
                                             z]),
                      'color': [],
                      'lorentz':['FFV2'],
                      'couplings':{(0, 0):'GC_424'},
                      'orders':{'QED':1}}))

        # 4f coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             n2, \
                                             x1minus,
                                             x1plus]),
                      'color': [],
                      'lorentz':['FFV2'],
                      'couplings':{(0, 0):'GC_424'},
                      'orders':{'QED':1}}))
        
        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)
        mylegn1 = base_objects.Leg({'id':1000022})
        mylegn2 = base_objects.Leg({'id':1000023})
        mylegxp = base_objects.Leg({'id':1000024})
        mylegxm = base_objects.Leg({'id':-1000024})
        mylegz  = base_objects.Leg({'id':23})
        mylegwp = base_objects.Leg({'id':24})
        mylegwm = base_objects.Leg({'id':-24})

        wfn1 = helas_objects.HelasWavefunction(mylegn1, 1, mymodel)
        wfn2 = helas_objects.HelasWavefunction(mylegn2, 1, mymodel)
        wfxp = helas_objects.HelasWavefunction(mylegxp, 1, mymodel)
        wfxm = helas_objects.HelasWavefunction(mylegxm, 1, mymodel)
        wfz = helas_objects.HelasWavefunction(mylegz, 1, mymodel)
        wfwp = helas_objects.HelasWavefunction(mylegwp, 1, mymodel)
        wfwm = helas_objects.HelasWavefunction(mylegwm, 1, mymodel)
        # Dummy vertex just to create an amplitude
        vx = base_objects.Vertex({'legs': base_objects.LegList([mylegn1]),
                                  'id': 4})
        amp = helas_objects.HelasAmplitude(vx, mymodel)

        # Test wavefunctions with mothers

        mothers = helas_objects.HelasWavefunctionList([wfn2, wfz])
        wfn1.set('mothers', mothers)
        wfn1.set('pdg_codes', [1000022,1000023,23])

        wfn1.set('state', 'incoming')
        wfn2.set('state', 'incoming')
        # n2 is incoming mother (so n1 is incoming result) -> need conj.
        self.assertEqual(wfn1.get_conjugate_index(), (1,))

        wfn1.set('state', 'outgoing')
        wfn2.set('state', 'outgoing')
        # n2 is outgoing mother -> don't need conj.
        self.assertEqual(wfn1.get_conjugate_index(), ())

        mothers = helas_objects.HelasWavefunctionList([wfn1, wfz])
        wfn2.set('mothers', mothers)
        wfn2.set('pdg_codes', [1000022,1000023,23])

        wfn1.set('state', 'incoming')
        wfn2.set('state', 'incoming')
        # n1 is incoming mother -> don't need conj.
        self.assertEqual(wfn2.get_conjugate_index(), ())

        wfn1.set('state', 'outgoing')
        wfn2.set('state', 'outgoing')
        # n2 is outgoing mother -> need conj.
        self.assertEqual(wfn2.get_conjugate_index(), (1,))

        mothers = helas_objects.HelasWavefunctionList([wfxp, wfwm])
        wfn1.set('mothers', mothers)
        wfn1.set('pdg_codes', [1000022,1000024,-24])

        wfn1.set('state', 'incoming')
        # only one Majorana
        self.assertEqual(wfn1.get_conjugate_index(), ())

        wfn1.set('state', 'outgoing')
        # only one Majorana
        self.assertEqual(wfn1.get_conjugate_index(), ())

        wfxp.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfn1.get_conjugate_index(), (1,))
        wfxp.set('fermionflow', 1)

        mothers = helas_objects.HelasWavefunctionList([wfn1, wfwm])
        wfxm.set('mothers', mothers)
        wfxm.set('pdg_codes', [1000022,1000024,-24])

        wfxm.set('state', 'incoming')
        wfn1.set('state', 'incoming')
        # only one Majorana
        self.assertEqual(wfxm.get_conjugate_index(), ())

        wfxm.set('state', 'outgoing')
        wfn1.set('state', 'outgoing')
        # only one Majorana
        self.assertEqual(wfxm.get_conjugate_index(), ())

        wfxm.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfxm.get_conjugate_index(), (1,))
        wfxm.set('fermionflow', 1)

        mothers = helas_objects.HelasWavefunctionList([wfn1, wfxm, wfxp])
        wfn2.set('mothers', mothers)
        wfn2.set('pdg_codes', [1000024,-1000024,1000022,1000023])

        wfn2.set('state', 'incoming')
        # n1 is incoming mother -> don't need conj.
        self.assertEqual(wfn2.get_conjugate_index(), ())

        wfxm.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfn2.get_conjugate_index(), (1,))
        wfxm.set('fermionflow', 1)

        wfn2.set('state', 'outgoing')
        # n1 is outgoing mother -> need conj.
        self.assertEqual(wfn2.get_conjugate_index(), (2,))

        wfxm.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfn2.get_conjugate_index(), (1,2))
        wfxm.set('fermionflow', 1)

        mothers = helas_objects.HelasWavefunctionList([wfn1, wfxm, wfxp])
        wfn2.set('mothers', mothers)
        wfn2.set('pdg_codes', [1000022,1000023,1000024,-1000024])

        wfn2.set('state', 'incoming')
        # n1 is incoming mother -> don't need conj.
        self.assertEqual(wfn2.get_conjugate_index(), ())

        wfxm.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfn2.get_conjugate_index(), (2,))
        wfxm.set('fermionflow', 1)

        wfn2.set('state', 'outgoing')
        # n1 is outgoing mother -> need conj.
        self.assertEqual(wfn2.get_conjugate_index(), (1,))

        wfxm.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfn2.get_conjugate_index(), (1,2))
        wfxm.set('fermionflow', 1)

        mothers = helas_objects.HelasWavefunctionList([wfn1, wfxm, wfxp])
        wfn2.set('mothers', mothers)
        wfn2.set('pdg_codes', [1000022,1000024,1000023,-1000024])

        wfn2.set('state', 'incoming')
        # only one Majorana in fermion line -> don't need conj.
        self.assertEqual(wfn2.get_conjugate_index(), ())

        wfn2.set('state', 'outgoing')
        # only one Majorana in fermion line -> don't need conj.
        self.assertEqual(wfn2.get_conjugate_index(), ())

        wfxp.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfn2.get_conjugate_index(), (1,))
        wfxp.set('fermionflow', 1)

        wfxm.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(wfn2.get_conjugate_index(), (2,))
        wfxm.set('fermionflow', 1)

        # Test amplitudes with mothers
        mothers = helas_objects.HelasWavefunctionList([wfn1, wfn2, wfxm, wfxp])
        amp.set('mothers', mothers)
        amp.set('pdg_codes', [1000022,1000023,1000024,-1000024])

        wfn1.set('state', 'incoming')
        wfn2.set('state', 'outgoing')
        # n1 is incoming mother -> don't need conj.
        self.assertEqual(amp.get_conjugate_index(), ())

        wfxp.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(amp.get_conjugate_index(), (2,))
        wfxp.set('fermionflow', 1)

        wfn1.set('state', 'outgoing')
        wfn2.set('state', 'incoming')
        # n1 is outgoing mother -> need conj.
        self.assertEqual(amp.get_conjugate_index(), (1,))

        wfxm.set('fermionflow', -1)
        # neg. fermion flow
        self.assertEqual(amp.get_conjugate_index(), (1,2))
        wfxm.set('fermionflow', 1)

#===============================================================================
# HelasDecayChainProcessTest
#===============================================================================
class HelasDecayChainProcessTest(unittest.TestCase):
    """Test class for the HelasDecayChainProcess object"""

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

        g = mypartlist[-1]

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
        u = mypartlist[-1]
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
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
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
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX1'},
                      'orders':{'QCD':1}}))


        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)


    def test_helas_decay_chain_process(self):
        """Test a HelasDecayChainProcess pp > jj, j > jj
        """

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for leg in [my_multi_leg] * 4])
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        
        my_process_definition = base_objects.ProcessDefinition({\
                                     'legs':my_multi_leglist,
                                     'model':self.mymodel})
        my_decay_leglist = base_objects.MultiLegList([copy.copy(leg) \
                                          for leg in [my_multi_leg] * 4])
        my_decay_leglist[0].set('state', False)
        my_decay_processes = base_objects.ProcessDefinition({\
                               'legs':my_decay_leglist,
                               'model':self.mymodel})

        my_process_definition.set('decay_chains',
                                  base_objects.ProcessDefinitionList(\
                                    [my_decay_processes]))

        my_decay_chain_amps = diagram_generation.DecayChainAmplitude(\
                                                   my_process_definition)
        
        my_dc_process = helas_objects.HelasDecayChainProcess(\
                                       my_decay_chain_amps)

        self.assertEqual(len(my_dc_process.get('core_processes')), 33)
        self.assertEqual(len(my_dc_process.get('decay_chains')), 1)
        self.assertEqual(len(my_dc_process.get('decay_chains')[0].\
                             get('core_processes')), 15)

    def test_helas_forbidden_s_channel_decay_chain(self):
        """Test helas diagrams with forbidden s-channel particles for decay chain.
        """

        # Test with u g > u g g, u > u g

        goal_no_quark = 15
        quark_none = {0:[1,2],1:[1,2],2:[2],8:[1],11:[1],13:[1],14:[1]}
        quark_true = {0:0,1:0,2:1,6:0,8:0,11:0,13:0,14:0}

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel,
                                       'forbidden_onsh_s_channels':[2]})

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        mydecayproc = base_objects.Process({'legs':myleglist,
                                            'model':self.mymodel})
        myproc.set('decay_chains', base_objects.ProcessList([mydecayproc]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)
        
        helas_amplitude = helas_objects.HelasDecayChainProcess(myamplitude).\
                          combine_decay_chain_processes()[0]

        self.assertEqual(len(helas_amplitude.get('diagrams')),
                         goal_no_quark)

        diagrams = helas_amplitude.get('diagrams')
        wf_dict = {}
        for idiag in range(len(diagrams)):
            if idiag in quark_none:
                vertices, tchannels = \
                      diagrams[idiag].get('amplitudes')[0].get_s_and_t_channels(2, self.mymodel, 20)
                for ivert in range(len(vertices)):
                    if ivert in quark_none[idiag]:
                        # This is forbidden leg
                        self.assertEqual(False,
                                vertices[ivert].get('legs')[-1].get('onshell'))
                    elif ivert == quark_true[idiag]:
                        # This is the decay chain leg
                        self.assertEqual(True,
                                vertices[ivert].get('legs')[-1].get('onshell'))
                    else:
                        self.assertEqual(None,
                                vertices[ivert].get('legs')[-1].get('onshell'))
                       
        # Test with u g > u g , u > u g g

        goal_no_quark = 9
        quark_none = {0:[0],1:[0],3:[0],4:[0],6:[0],7:[0]}
        quark_true = dict(zip(range(goal_no_quark),[1]*goal_no_quark))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        mydecayproc = base_objects.Process({'legs':myleglist,
                                            'model':self.mymodel,
                                            'forbidden_onsh_s_channels':[2]})
        myproc.set('decay_chains', base_objects.ProcessList([mydecayproc]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)

        helas_amplitude = helas_objects.HelasDecayChainProcess(myamplitude).\
                          combine_decay_chain_processes()[0]

        self.assertEqual(len(helas_amplitude.get('diagrams')),
                         goal_no_quark)

        diagrams = helas_amplitude.get('diagrams')
        for idiag in range(len(diagrams)):
            if idiag in quark_none:
                vertices, tchannels = \
                      diagrams[idiag].get('amplitudes')[0].get_s_and_t_channels(2, self.mymodel, 20)
                for ivert in range(len(vertices)):
                    if ivert in quark_none[idiag]:
                        # This is forbidden leg
                        self.assertEqual(False,
                                vertices[ivert].get('legs')[-1].get('onshell'))
                    elif ivert == quark_true[idiag]:
                        # This is the decay chain leg
                        self.assertEqual(True,
                                vertices[ivert].get('legs')[-1].get('onshell'))
                    else:
                        self.assertEqual(None,
                                vertices[ivert].get('legs')[-1].get('onshell'))
                        
            

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

        g = mypartlist[-1]

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
        u = mypartlist[-1]
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
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
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
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)


    def test_helas_multi_process(self):
        """Test the HelasMultiProcess with the processes uu~>uu~
        and dd~>dd~"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        myproc1 = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude1 = diagram_generation.Amplitude({'process': myproc1})

        myamplitude1.get('diagrams')

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))

        myproc2 = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude2 = diagram_generation.Amplitude({'process': myproc2})

        myamplitude2.get('diagrams')

        myamplitudes = diagram_generation.AmplitudeList([ myamplitude1,
                                                          myamplitude2 ])
        # myamplitudes is emptied by the HelasMultiProcess
        myamplcopy = copy.copy(myamplitudes)

        my_matrix_element1 = helas_objects.HelasMatrixElement(myamplitude1)
        my_multiprocess = helas_objects.HelasMultiProcess(myamplitudes)

        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 1)
        self.assertEqual(my_multiprocess.get('matrix_elements')[0].\
                         get('processes'),
                         base_objects.ProcessList([ myproc1, myproc2 ]))
        self.assertEqual(my_multiprocess.get('matrix_elements')[0].\
                         get('diagrams'),
                         my_matrix_element1.get('diagrams'))

        myamplcopy[0].get('process').set('id', 10)

        my_multiprocess = helas_objects.HelasMultiProcess(myamplcopy)
        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 2)


    def test_helas_multiprocess_pp_nj(self):
        """Setting up and testing pp > nj based on multiparticle lists,
        using the amplitude functionality of MultiProcess
        (which makes partial use of crossing symmetries)
        """

        max_fs = 2 #3

        p = [21, 1, -1, -2, 2]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        goal_number_matrix_elements = [18, 26]

        for nfs in range(2, max_fs + 1):

            # Define the multiprocess
            my_multi_leglist = base_objects.MultiLegList([copy.copy(leg) for \
                                            leg in [my_multi_leg] * (2 + nfs)])

            my_multi_leglist[0].set('state', False)
            my_multi_leglist[1].set('state', False)

            my_process_definition = base_objects.ProcessDefinition({\
                            'legs':my_multi_leglist,
                            'model':self.mymodel})
            my_multiprocess = diagram_generation.MultiProcess(\
                {'process_definitions':\
                 base_objects.ProcessDefinitionList([my_process_definition])})

            helas_multi_proc = helas_objects.HelasMultiProcess(my_multiprocess)

            if nfs <= 3:
                self.assertEqual(len(helas_multi_proc.get('matrix_elements')),
                                     goal_number_matrix_elements[nfs - 2])

    def test_non_combine_processes(self):
        """Test that the processes uc>w+uusu~ and uc>w+usss~ are not
           combined (they have identical diagrams but with different
           relative particle order in different diagrams)"""

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

        g = mypartlist[-1]

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
        u = mypartlist[-1]
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
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # A quark C and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'c',
                      'antiname':'c~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'c',
                      'antitexname':'\bar c',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':4,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        c = mypartlist[-1]
        antic = copy.copy(c)
        antic.set('is_part', False)

        # A quark S and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'s',
                      'antiname':'s~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'s',
                      'antitexname':'\bar s',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':3,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        s = mypartlist[-1]
        antis = copy.copy(s)
        antis.set('is_part', False)

        # W+/-
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'WMASS',
                      'width':'WWIDTH',
                      'texname':'w+',
                      'antitexname':'w-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [c, \
                                             antic, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [s, \
                                             antis, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # 3-Gluon coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # u d w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antid, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             d, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # c s w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [c, \
                                             antis, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [antic, \
                                             s, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        # u c > w+ u u s u~ 

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':4,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':24}))
        myleglist.append(base_objects.Leg({'id':2}))
        myleglist.append(base_objects.Leg({'id':2}))
        myleglist.append(base_objects.Leg({'id':3}))
        myleglist.append(base_objects.Leg({'id':-2}))

        myproc1 = base_objects.Process({'legs':myleglist,
                                        'orders':{'QED':1},
                                        'model':mymodel})

        myamplitude1 = diagram_generation.Amplitude(myproc1)

        amplitude_tag1 = helas_objects.IdentifyMETag.create_tag(myamplitude1)

        # u c > w+ u s s s~

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':4,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':24}))
        myleglist.append(base_objects.Leg({'id':2}))
        myleglist.append(base_objects.Leg({'id':3}))
        myleglist.append(base_objects.Leg({'id':3}))
        myleglist.append(base_objects.Leg({'id':-3}))

        myproc2 = base_objects.Process({'legs':myleglist,
                                        'orders':{'QED':1},
                                        'model':mymodel})

        myamplitude2 = diagram_generation.Amplitude(myproc2)
        
        amplitude_tag2 = helas_objects.IdentifyMETag.create_tag(myamplitude2)
                         
        self.assertFalse(amplitude_tag1 == amplitude_tag2)


    def test_complete_decay_chain_process(self):
        """Test a complete decay chain process gp>jg,j>jjj,j>jjj
        """

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});
        my_gluon_leg = base_objects.MultiLeg({'ids': [21], 'state': True});

        # Define the multiprocess
        my_multi_leglist = base_objects.MultiLegList([copy.copy(my_gluon_leg),
                                                      copy.copy(my_multi_leg),
                                                      copy.copy(my_multi_leg),
                                                      copy.copy(my_gluon_leg)])
                                                      
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        
        my_process_definition = base_objects.ProcessDefinition({\
                                     'legs':my_multi_leglist,
                                     'model':self.mymodel})
        #my_multi_leg = base_objects.MultiLeg({'ids': [1, -1, 21],
        #                                              'state': True});
        my_decay_leglist = base_objects.MultiLegList([copy.copy(leg) \
                                          for leg in [my_multi_leg] * 3])
        my_decay_leglist[0].set('state', False)
        my_decay_leglist2 = base_objects.MultiLegList([copy.copy(leg) \
                                          for leg in [my_multi_leg] * 3] + \
                                                     [copy.copy(my_gluon_leg)])
        my_decay_leglist2[0].set('state', False)
        my_decay_processes = base_objects.ProcessDefinitionList(\
            [base_objects.ProcessDefinition({\
                               'legs':my_decay_leglist,
                               'model':self.mymodel}),
             base_objects.ProcessDefinition({\
                               'legs':my_decay_leglist2,
                               'model':self.mymodel})])

        my_process_definition.set('decay_chains',
                                  my_decay_processes)

        my_decay_chain_amps = diagram_generation.DecayChainAmplitude(\
                                                   my_process_definition)
        
        my_dc_process = helas_objects.HelasDecayChainProcess(\
                                       my_decay_chain_amps)

        matrix_elements = my_dc_process.combine_decay_chain_processes()

        self.assertEqual(len(matrix_elements), 11)

        num_processes =  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        num_amps =  [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        num_wfs =  [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        iden_factors =  [1, 1, 6, 1, 1, 6, 1, 1, 6, 2, 12]

        for i, me in enumerate(matrix_elements):
            self.assertEqual(len(me.get('processes')), num_processes[i])
            if num_amps[i] > 0:
                self.assertEqual(me.get_number_of_amplitudes(),
                                 num_amps[i])
#            if num_wfs[i] > 0:
#                self.assertEqual(me.get_number_of_wavefunctions(),
#                                 num_wfs[i])

            if iden_factors[i] > 0:
                self.assertEqual(me.get('identical_particle_factor'),
                                 iden_factors[i])

            for i, amp in enumerate(sorted(me.get_all_amplitudes(),
                                       lambda a1,a2: \
                                       a1.get('number') - a2.get('number'))):
                self.assertEqual(amp.get('number'), i + 1)
                  
            for i, wf in enumerate(sorted(me.get_all_wavefunctions(),
                                       lambda a1,a2: \
                                       a1.get('number') - a2.get('number'))):
                self.assertEqual(wf.get('number'), i + 1)

            for i, wf in enumerate(filter (lambda wf: not wf.get('mothers'),
                                           me.get_all_wavefunctions())):
                self.assertEqual(wf.get('number_external'), i + 1)

    def test_decay_chain_process_overall_orders(self):
        """Test a complete decay chain process pp>jj,j>jj with QED=2, QCD=2
        """

        mypartlist = self.mymodel.get('particles')
        myinterlist = self.mymodel.get('interactions')

        # A photon
        mypartlist.append(base_objects.Particle({'name':'a',
                      'antiname':'a',
                      'spin':3,
                      'color':0,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'a',
                      'antitexname':'a',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':22,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        a = mypartlist[-1]
        u = self.mymodel.get('particle_dict')[2]
        antiu = self.mymodel.get('particle_dict')[-2]
        d = self.mymodel.get('particle_dict')[1]
        antid = self.mymodel.get('particle_dict')[-1]

        # Photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             a]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'UUA'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             a]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'DDA'},
                      'orders':{'QED':1}}))

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)

        p = [1, -1, 2, -2, 21]

        my_multi_leg = base_objects.MultiLeg({'ids': p, 'state': True});

        # Define the multiprocess
        my_multi_leglist = base_objects.MultiLegList([copy.copy(my_multi_leg),
                                                      copy.copy(my_multi_leg),
                                                      copy.copy(my_multi_leg),
                                                      copy.copy(my_multi_leg)])
                                                      
        
        my_multi_leglist[0].set('state', False)
        my_multi_leglist[1].set('state', False)
        
        my_process_definition = base_objects.ProcessDefinition({\
                                     'legs':my_multi_leglist,
                                     'model':self.mymodel,
                                     'overall_orders':{'QED': 2, 'QCD': 2}})
        #my_multi_leg = base_objects.MultiLeg({'ids': [1, -1, 21],
        #                                              'state': True});
        my_decay_leglist = base_objects.MultiLegList([copy.copy(leg) \
                                          for leg in [my_multi_leg] * 3])
        my_decay_leglist[0].set('state', False)
        my_decay_processes = base_objects.ProcessDefinitionList(\
            [base_objects.ProcessDefinition({\
                               'legs':my_decay_leglist,
                               'model':self.mymodel})])

        my_process_definition.set('decay_chains',
                                  my_decay_processes)

        my_decay_chain_amps = diagram_generation.DecayChainAmplitude(\
                                                   my_process_definition)
        
        my_dc_process = helas_objects.HelasDecayChainProcess(\
                                       my_decay_chain_amps)

        matrix_elements = my_dc_process.combine_decay_chain_processes()

        self.assertEqual(len(matrix_elements), 20)

        num_processes = [1] * 20

        for i, me in enumerate(matrix_elements):
            self.assertEqual(len(me.get('processes')), num_processes[i])
            for i, amp in enumerate(sorted(me.get_all_amplitudes(),
                                       lambda a1,a2: \
                                       a1.get('number') - a2.get('number'))):
                self.assertEqual(amp.get('number'), i + 1)
                  
            for i, wf in enumerate(sorted(me.get_all_wavefunctions(),
                                       lambda a1,a2: \
                                       a1.get('number') - a2.get('number'))):
                self.assertEqual(wf.get('number'), i + 1)

            for i, wf in enumerate(filter (lambda wf: not wf.get('mothers'),
                                           me.get_all_wavefunctions())):
                self.assertEqual(wf.get('number_external'), i + 1)

    def test_multistage_decay_chain_process(self):
        """Test a multistage decay g g > d d~, d > g d, d~ > g d~, g > u u~ g
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        amp_core = diagram_generation.Amplitude(mycoreproc)

        me_core =  helas_objects.HelasMatrixElement(\
            amp_core)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))

        mydecay11 = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        me11 =  helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay11))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        mydecay12 = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        me12 =  helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay12))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        me2 =  helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay2))

        mydecay11.set('decay_chains', base_objects.ProcessList([mydecay2]))
        mydecay12.set('decay_chains', base_objects.ProcessList([mydecay2]))

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay11, mydecay12]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        #print matrix_elements[0].get('processes')[0].nice_string()
        #print matrix_elements[0].get('identical_particle_factor')

        #for diag in matrix_elements[0].get('diagrams'):
            #print 'Diagram ',diag.get('number')
            #print "Wavefunctions: ", len(diag.get('wavefunctions'))
            #for wf in diag.get('wavefunctions'):
            #    print wf.get('number'), wf.get('number_external'), wf.get('pdg_code'), [mother.get('number') for mother in wf.get('mothers')], wf.get('onshell')
            #print "Amplitudes: ", len(diag.get('amplitudes'))
            #for amp in diag.get('amplitudes'):
            #    print amp.get('number'), [mother.get('number') for mother in amp.get('mothers')]

        self.assertEqual(matrix_elements[0].get_number_of_amplitudes(),
                         me_core.get_number_of_amplitudes() * \
                         me11.get_number_of_amplitudes() * \
                         me12.get_number_of_amplitudes() * \
                         me2.get_number_of_amplitudes() ** 2)

        self.assertEqual(matrix_elements[0].get('identical_particle_factor'),
                         1)

        for i, amp in enumerate(sum([diag.get('amplitudes') for diag in \
                                    matrix_elements[0].get('diagrams')],[])):
            self.assertEqual(amp.get('number'), i + 1)

        for i, wf in enumerate(sum([diag.get('wavefunctions') for diag in \
                                   matrix_elements[0].get('diagrams')],[])):
            self.assertEqual(wf.get('number'), i + 1)

        for i, wf in enumerate(filter (lambda wf: not wf.get('mothers'),
                                       matrix_elements[0].get_all_wavefunctions())):
            self.assertEqual(wf.get('number_external'), i + 1)

        # Test the setting of wavefunctions as "onshell" if they
        # correspond to a decaying particle
        for i, wf in enumerate(matrix_elements[0].get_all_wavefunctions()):
            if i in [6, 8, 13, 15, 18, 19, 21, 22, 24, 25, 27, 28]:
                self.assert_(wf.get('onshell'))
            else:
                self.assert_(not wf.get('onshell'))

        # Test Process.get_legs_with_decays
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 5}))
        myleglist.append(base_objects.Leg({'id':1,
                                           'state':True,
                                           'number': 6}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True,
                                           'number': 7}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':True,
                                           'number': 8}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 9}))
        myleglist.append(base_objects.Leg({'id':-1,
                                           'state':True,
                                           'number': 10}))
        
        self.assertEqual(myleglist, matrix_elements[0].get('processes')[0].\
                         get_legs_with_decays())

    def test_majorana_decay_chain_process(self):
        """Test decay chain with majorana particles e+e->n1n1
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[-1]
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
        seminus = mypartlist[-1]
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
        n1 = mypartlist[-1]

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
        a = mypartlist[-1]

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        # Coupling of sl2 to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [a, \
                                             seplus, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX56'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':True}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        self.assertEqual(matrix_elements[0].get('identical_particle_factor'),
                         2)

        for i, diag in enumerate(matrix_elements[0].get('diagrams')):
            self.assertEqual(diag.get('number'), i + 1)

        for i, amp in enumerate(sum([diag.get('amplitudes') for diag in \
                                    matrix_elements[0].get('diagrams')],[])):
            self.assertEqual(amp.get('number'), i + 1)

        for i, wf in enumerate(sum([diag.get('wavefunctions') for diag in \
                                   matrix_elements[0].get('diagrams')],[])):
            self.assertEqual(wf.get('number'), i + 1)

        for i, wf in enumerate(filter (lambda wf: not wf.get('mothers'),
                                       matrix_elements[0].get('diagrams')[0].\
                                       get('wavefunctions'))):
            self.assertEqual(wf.get('number_external'), i + 1)

        for wf in filter (lambda wf: not wf.get('mothers'),
                                       sum([d.get('wavefunctions') for d in \
                                            matrix_elements[0].get('diagrams')\
                                            [1:]], [])):
            old_wf = filter(lambda w: w.get('number_external') == \
                            wf.get('number_external') and not w.get('mothers'),\
                            matrix_elements[0].get('diagrams')[0].\
                            get('wavefunctions'))[0]
            self.assertEqual(wf.get('particle'), old_wf.get('particle'))
            self.assert_(wf.get_with_flow('state') != old_wf.get_with_flow('state'))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay3 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel,
                                         'is_decay_chain': True})

        me3 =  helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay3), gen_color = False)
        
        #print me3.get('processes')[0].nice_string()
        #print me3.get_base_amplitude().get('diagrams').nice_string()

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay3]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        #for d in matrix_elements[0].get('diagrams'):
        #    print "Diagram number ", d.get('number')
        #    print "Wavefunctions:"
        #    for w in d.get('wavefunctions'):
        #        print w.get('number'),w.get('number_external'),w.get('pdg_code'),\
        #              [wf.get('number') for wf in w.get('mothers')]
        #    print "Amplitudes:"
        #    for a in d.get('amplitudes'):
        #        print a.get('number'),\
        #              [wf.get('number') for wf in a.get('mothers')]

        for i, diag in enumerate(matrix_elements[0].get('diagrams')):
            self.assertEqual(diag.get('number'), i + 1)

        for i, amp in enumerate(sum([diag.get('amplitudes') for diag in \
                                    matrix_elements[0].get('diagrams')],[])):
            self.assertEqual(amp.get('number'), i + 1)

        for i, wf in enumerate(sum([diag.get('wavefunctions') for diag in \
                                   matrix_elements[0].get('diagrams')],[])):
            self.assertEqual(wf.get('number'), i + 1)

        for i, wf in enumerate(filter (lambda wf: not wf.get('mothers'),
                                       matrix_elements[0].get('diagrams')[0].\
                                       get('wavefunctions'))):
            self.assertEqual(wf.get('number_external'), i + 1)

        for wf in filter (lambda wf: not wf.get('mothers'),
                                       sum([d.get('wavefunctions') for d in \
                                            matrix_elements[0].get('diagrams')\
                                            [1:]], [])):
            old_wf = filter(lambda w: w.get('number_external') == \
                            wf.get('number_external') and not w.get('mothers'),\
                            matrix_elements[0].get('diagrams')[0].\
                            get('wavefunctions'))[0]
            self.assertEqual(wf.get('particle'), old_wf.get('particle'))
            self.assert_(wf.get_with_flow('state') != old_wf.get_with_flow('state'))
        

    def test_decay_chain_different_pdgs(self):
        """Test decay chain with identical particles with different PDG
        With the new IdentifyMETag, processes with identical s-channel particles
        but different PDG code are not combined, to ensure that the particles
        are correctly included in the event file.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        eminus = mypartlist[-1]
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
        seminus = mypartlist[-1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # A E' slepton and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'sl4-',
                      'antiname':'sl4+',
                      'spin':1,
                      'color':1,
                      'mass':'Msl2',
                      'width':'Wsl2',
                      'texname':'\tilde mu^-',
                      'antitexname':'\tilde mu^+',
                      'line':'dashed',
                      'charge':1.,
                      'pdg_code':1000013,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        sepminus = mypartlist[-1]
        sepplus = copy.copy(sepminus)
        sepplus.set('is_part', False)

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
        n1 = mypartlist[-1]

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
        a = mypartlist[-1]

        # Coupling of n1 to e and se
        myinterlist.append(base_objects.Interaction({
                      'id': 103,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 104,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 105,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             sepplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX350'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 106,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             sepminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX494'},
                      'orders':{'QED':1}}))

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX12'},
                      'orders':{'QED':1}}))

        # Coupling of sl2 to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [a, \
                                             seplus, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX56'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [a, \
                                             sepplus, \
                                             sepminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX56'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[11],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-11],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[1000011,1000013],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[-1000011,-1000013],
                                         'state':True}))

        mycoreproc = base_objects.ProcessDefinition({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[1000011,1000013],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[11],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[1000022],
                                         'state':True}))

        mydecay1 = base_objects.ProcessDefinition({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[-1000011,-1000013],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-11],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[1000022],
                                         'state':True}))


        mydecay2 = base_objects.ProcessDefinition({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessDefinitionList([\
            mydecay1,mydecay2]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()
        
        self.assertEqual(len(matrix_elements), 4)

    def get_w_model(self):
        """Prepare a model with W, quarks, and leptons"""

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        u = mypartlist[-1]
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
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # A quark C and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'c',
                      'antiname':'c~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'c',
                      'antitexname':'\bar c',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':4,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        c = mypartlist[-1]
        antic = copy.copy(c)
        antic.set('is_part', False)

        # A quark S and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'s',
                      'antiname':'s~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'s',
                      'antitexname':'\bar s',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':3,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        s = mypartlist[-1]
        antis = copy.copy(s)
        antis.set('is_part', False)

        # W+/-
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'WMASS',
                      'width':'WWIDTH',
                      'texname':'w+',
                      'antitexname':'w-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # A W
        mypartlist.append(base_objects.Particle({'name':'W+',
                      'antiname':'W-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'texname':'W^+',
                      'antitexname':'W^-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        Wplus = mypartlist[-1]
        Wminus = copy.copy(Wplus)
        Wminus.set('is_part', False)

        # An electron and muon
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
        eminus = mypartlist[-1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'mu-',
                      'antiname':'mu+',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\mu^-',
                      'antitexname':'\mu^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':13,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        muminus = mypartlist[-1]
        muplus = copy.copy(muminus)
        muplus.set('is_part', False)

        # Neutrinos
        mypartlist.append(base_objects.Particle({'name':'ve',
                      'antiname':'ve~',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'v_e',
                      'antitexname':'\tilde v_e',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':12,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        ve = mypartlist[-1]
        vebar = copy.copy(ve)
        vebar.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'vm',
                      'antiname':'vm~',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'v_m',
                      'antitexname':'\tilde v_m',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':14,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        vm = mypartlist[-1]
        vmbar = copy.copy(vm)
        vmbar.set('is_part', False)

        # u d w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             u, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             d, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # c s w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [antis, \
                                             c, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [antic, \
                                             s, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # e ve w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 12,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             ve, \
                                             wminus]),
                      'color': [color.ColorString()],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_30'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 13,
                      'particles': base_objects.ParticleList(\
                                            [vebar, \
                                             eminus, \
                                             wplus]),
                      'color': [color.ColorString()],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_30'},
                      'orders':{'QED':1}}))

        # mu vm w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 14,
                      'particles': base_objects.ParticleList(\
                                            [muplus, \
                                             vm, \
                                             wminus]),
                      'color': [color.ColorString()],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_30'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 15,
                      'particles': base_objects.ParticleList(\
                                            [vmbar, \
                                             muminus, \
                                             wplus]),
                      'color': [color.ColorString()],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_30'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        return mymodel

    def test_decay_chain_different_order1(self):
        """Test one decay chain with identical particles with different FS order
        """

        mymodel = self.get_w_model()

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[1,3,2,4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-1,-3,-2,-4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[24]}))

        mycoreproc = base_objects.ProcessDefinition({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[24],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-11],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[12],
                                         'state':True}))

        mydecay1 = base_objects.ProcessDefinition({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[24],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[14],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[-13],
                                         'state':True}))


        mydecay2 = base_objects.ProcessDefinition({'legs':myleglist,
                                                   'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessDefinitionList([\
            mydecay1,mydecay2]))

        myamplitudes = diagram_generation.MultiProcess(mycoreproc)

        self.assertEqual(len(myamplitudes.get('amplitudes')), 1)

        my_multiprocess = helas_objects.HelasMultiProcess(myamplitudes)
        
        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 1)

        goal_string = ["u d~ > e+ ve",
                       "c s~ > e+ ve",
                       "u d~ > mu+ vm",
                       "c s~ > mu+ vm"]

        for i, p in enumerate(\
            my_multiprocess.get('matrix_elements')[0].get('processes')):
            self.assertEqual(p.base_string(),goal_string[i])

    def test_decay_chain_different_order2(self):
        """Test two decay chains with particles with different FS order
        """

        mymodel = self.get_w_model()

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[1,3,2,4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-1,-3,-2,-4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[24]}))

        mycoreproc1 = base_objects.ProcessDefinition({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[24],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-11],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[12],
                                         'state':True}))

        mydecay1 = base_objects.ProcessDefinition({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[1,3,2,4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-1,-3,-2,-4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-24]}))

        mycoreproc2 = base_objects.ProcessDefinition({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[-24],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-14],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[13],
                                         'state':True}))


        mydecay2 = base_objects.ProcessDefinition({'legs':myleglist,
                                                   'model':mymodel})

        mycoreproc1.set('decay_chains', base_objects.ProcessDefinitionList([\
            mydecay1]))

        mycoreproc2.set('decay_chains', base_objects.ProcessDefinitionList([\
            mydecay2]))

        myprocs = base_objects.ProcessDefinitionList([mycoreproc1,
                                                      mycoreproc2])

        myamplitudes = diagram_generation.MultiProcess(myprocs)

        self.assertEqual(len(myamplitudes.get('amplitudes')), 2)

        my_multiprocess = helas_objects.HelasMultiProcess(myamplitudes)
        
        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 2)

    def test_decay_chain_different_order3(self):
        """Test two decay chains with particles with different FS order (3)
        """

        mymodel = self.get_w_model()

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[1,3,2,4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-1,-3,-2,-4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[24]}))

        mycoreproc1 = base_objects.ProcessDefinition({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[24],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-11],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[12],
                                         'state':True}))

        mydecay1 = base_objects.ProcessDefinition({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[1,3,2,4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[-1,-3,-2,-4],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[24]}))

        mycoreproc2 = base_objects.ProcessDefinition({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.MultiLegList()

        myleglist.append(base_objects.MultiLeg({'ids':[24],
                                         'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':[14],
                                         'state':True}))
        myleglist.append(base_objects.MultiLeg({'ids':[-13],
                                         'state':True}))


        mydecay2 = base_objects.ProcessDefinition({'legs':myleglist,
                                                   'model':mymodel})

        mycoreproc1.set('decay_chains', base_objects.ProcessDefinitionList([\
            mydecay1]))

        mycoreproc2.set('decay_chains', base_objects.ProcessDefinitionList([\
            mydecay2]))

        myprocs = base_objects.ProcessDefinitionList([mycoreproc1,
                                                      mycoreproc2])

        myamplitudes = diagram_generation.MultiProcess(myprocs)

        self.assertEqual(len(myamplitudes.get('amplitudes')), 2)

        my_multiprocess = helas_objects.HelasMultiProcess(myamplitudes)
        
        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 1)

        goal_string = ["u d~ > e+ ve",
                       "c s~ > e+ ve",
                       "u d~ > mu+ vm",
                       "c s~ > mu+ vm"]

        for i, p in enumerate(\
            my_multiprocess.get('matrix_elements')[0].get('processes')):
            self.assertEqual(p.base_string(),goal_string[i])

    def test_equal_decay_chains(self):
        """Test the functions for checking equal decay chains
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))

        myproc1 = base_objects.Process({'legs':myleglist,
                                        'model':self.mymodel,
                                        'is_decay_chain': True})

        myamplitude1 = diagram_generation.Amplitude()
        myamplitude1.set('process', myproc1)
        myamplitude1.generate_diagrams()

        mymatrixelement1 = helas_objects.HelasMatrixElement(\
            myamplitude1, gen_color = False)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                         'state':True}))

        myproc2 = base_objects.Process({'legs':myleglist,
                                        'model':self.mymodel,
                                        'is_decay_chain': True})

        myamplitude2 = diagram_generation.Amplitude()
        myamplitude2.set('process', myproc2)
        myamplitude2.generate_diagrams()

        mymatrixelement2 = helas_objects.HelasMatrixElement(\
            myamplitude2, gen_color = False)

        self.assert_(helas_objects.HelasMatrixElement.\
                     check_equal_decay_processes(\
                       mymatrixelement1, mymatrixelement2))

    def test_decay_processes_different_is_particles(self):
        """Test the HelasMultiProcess with the processes w+ > u d~ and w- > u~ d"""

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        u = mypartlist[-1]
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
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[-1]
        antid = copy.copy(d)
        antid.set('is_part', False)

        # A quark C and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'c',
                      'antiname':'c~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'c',
                      'antitexname':'\bar c',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':4,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        c = mypartlist[-1]
        antic = copy.copy(c)
        antic.set('is_part', False)

        # A quark S and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'s',
                      'antiname':'s~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'s',
                      'antitexname':'\bar s',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':3,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        s = mypartlist[-1]
        antis = copy.copy(s)
        antis.set('is_part', False)

        # W+/-
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'WMASS',
                      'width':'WWIDTH',
                      'texname':'w+',
                      'antitexname':'w-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # u d w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antid, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 9,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             d, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # c s w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [c, \
                                             antis, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [antic, \
                                             s, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        # u c > w+ u u s u~ 

        myleglist = base_objects.MultiLegList()

        w = [24,-24]
        q = [1,2,3,4,-1,-2,-3,-4]

        myleglist.append(base_objects.MultiLeg({'ids':w,
                                           'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':q}))
        myleglist.append(base_objects.MultiLeg({'ids':q}))

        myproc = base_objects.ProcessDefinition({'legs':myleglist,
                                                 'orders':{'QED':1},
                                                 'model':mymodel})

        myamplitudes = diagram_generation.MultiProcess(myproc)

        self.assertEqual(len(myamplitudes.get('amplitudes')), 4)

        my_multiprocess = helas_objects.HelasMultiProcess(myamplitudes)

        self.assertEqual(len(my_multiprocess.get('matrix_elements')), 2)

#===============================================================================
# TestIdentifyMETag
#===============================================================================
class TestIdentifyMETag(unittest.TestCase):
    """Test class for the DiagramTag class"""


    def setUp(self):
        self.base_model = import_ufo.import_model('sm')
    
    def test_identify_me_tag_qq_qqg(self):
        """Test the find_symmetry function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model,
                                       'orders': {'QED': 0}})

        myamplitude1 = diagram_generation.Amplitude(myproc)


        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':1,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':1,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':1,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model,
                                       'orders': {'QED': 0}})

        myamplitude2 = diagram_generation.Amplitude(myproc)

        tags1 = sorted([helas_objects.IdentifyMETag(d, self.base_model) \
                        for d in myamplitude1.get('diagrams')])
        tags2 = sorted([helas_objects.IdentifyMETag(d, self.base_model) \
                        for d in myamplitude2.get('diagrams')])

        self.assertEqual(tags1, tags2)
        
        tags1 = sorted([helas_objects.IdentifyMETagFKS(d, self.base_model) \
                        for d in myamplitude1.get('diagrams')])
        tags2 = sorted([helas_objects.IdentifyMETagFKS(d, self.base_model) \
                        for d in myamplitude2.get('diagrams')])

        self.assertEqual(tags1, tags2)        

    def test_non_identify_me_tag_qq_qqg(self):
        """Test the find_symmetry function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude1 = diagram_generation.Amplitude(myproc)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':1,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':1,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamplitude2 = diagram_generation.Amplitude(myproc)

        tags1 = sorted([helas_objects.IdentifyMETag(d, self.base_model) \
                        for d in myamplitude1.get('diagrams')])

        tags2 = sorted([helas_objects.IdentifyMETag(d, self.base_model) \
                        for d in myamplitude2.get('diagrams')])

        self.assertFalse(tags1 == tags2)

        tags1 = sorted([helas_objects.IdentifyMETagFKS(d, self.base_model) \
                        for d in myamplitude1.get('diagrams')])

        tags2 = sorted([helas_objects.IdentifyMETagFKS(d, self.base_model) \
                        for d in myamplitude2.get('diagrams')])

        self.assertFalse(tags1 == tags2)
#===============================================================================
# TestIdentifyMETag
#===============================================================================
class TestIdentifyMETagFKS(unittest.TestCase):
    """Test class for the DiagramTag class"""


    def setUp(self):
        self.base_model = import_ufo.import_model('sm')
    
    def test_identify_me_tag_qq_qg(self):
        """Test the find_symmetry function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-1,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-4,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':-3,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model,
                                       'orders': {'QED': 2, 'QCD':0},
                                       'perturbation_couplings': ['QCD'],
                                       'NLO_mode':'tree'})


        myamplitude1 = diagram_generation.Amplitude(myproc)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-3,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-1,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':-4,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model,
                                       'orders': {'QED': 2, 'QCD':0},
                                        'perturbation_couplings': ['QCD'],
                                       'NLO_mode':'tree'})

        myamplitude2 = diagram_generation.Amplitude(myproc)

        tags1 = sorted([helas_objects.IdentifyMETag(d, self.base_model) \
                        for d in myamplitude1.get('diagrams')])
        tags2 = sorted([helas_objects.IdentifyMETag(d, self.base_model) \
                        for d in myamplitude2.get('diagrams')])

        self.assertEqual(tags1, tags2)

        tags1 = sorted([helas_objects.IdentifyMETagFKS(d, self.base_model) \
                        for d in myamplitude1.get('diagrams')])
        tags2 = sorted([helas_objects.IdentifyMETagFKS(d, self.base_model) \
                        for d in myamplitude2.get('diagrams')])

        self.assertNotEqual(tags1, tags2)




