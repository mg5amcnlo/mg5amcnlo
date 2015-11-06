################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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
from madgraph import core

import copy
import os

import aloha
from madgraph import MG5DIR

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.various.misc as misc

import tests.unit_tests as unittest
import tests.unit_tests.core.test_helas_objects as test_helas_objects
import tests.unit_tests.iolibs.test_file_writers as test_file_writers
import tests.parallel_tests.test_aloha as test_aloha


#===============================================================================
# HelasModelTestSetup
#===============================================================================
class HelasModelTestSetup(unittest.TestCase):
    """Test class for the HelasModel object"""

    mymodel = helas_call_writers.HelasCallWriter()
    mybasemodel = base_objects.Model()

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

        g = mypartlist[len(mypartlist) - 1]

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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'d',
                      'antiname':'d~',
                      'spin':2,
                      'color':3,
                      'mass':'mu',
                      'width':'zero',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        d = mypartlist[len(mypartlist) - 1]
        antid = copy.copy(d)
        antid.set('is_part', False)

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
        eminus = mypartlist[len(mypartlist) - 1]
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
        a = mypartlist[len(mypartlist) - 1]

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
        T1 = mypartlist[len(mypartlist) - 1]

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
        su = mypartlist[len(mypartlist) - 1]
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
        seminus = mypartlist[len(mypartlist) - 1]
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
        n1 = mypartlist[len(mypartlist) - 1]

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
        wplus = mypartlist[len(mypartlist) - 1]
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
        z = mypartlist[len(mypartlist) - 1]

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             g]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX15'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 10,
                      'particles': base_objects.ParticleList(\
                                            [d, \
                                             antid, \
                                             g]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             a]),
                      'color': [],
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
                      'color': [],
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
                      'color': [],
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
                      'color': [],
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
                      'color': [],
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
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX570'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 102,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             n1, \
                                             su]),
                      'color': [],
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

        # Coupling of n1 to z
        myinterlist.append(base_objects.Interaction({
                      'id': 106,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             n1,
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZN11'},
                      'orders':{'QED':1}}))

        # g-gamma-su-subar coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 100,
                      'particles': base_objects.ParticleList(\
                                            [a,
                                             g,
                                             su,
                                             antisu]),
                      'color': [],
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
                      'color': [],
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
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'MGVX8'},
                      'orders':{'QED':2}}))


        self.mybasemodel.set('particles', mypartlist)
        self.mybasemodel.set('interactions', myinterlist)
        self.mybasemodel.set('name', 'sm')
        self.mymodel.set('model', self.mybasemodel)

#===============================================================================
# HelasModelTest
#===============================================================================
class HelasModelTest(HelasModelTestSetup):
    """Test class for the HelasModel object"""

    def test_setget_helas_model_correct(self):
        """Test correct HelasModel object get and set"""

        self.assertEqual(self.mymodel.get_model_name(), 'sm')

    def test_setget_helas_model_error(self):
        """Test error raising in HelasModel object get and set"""

        mymodel = helas_call_writers.HelasCallWriter()
        not_a_string = 1.

        # General
        self.assertRaises(AssertionError,
                          mymodel.get,
                          not_a_string)
        self.assertRaises(helas_call_writers.HelasCallWriter.PhysicsObjectError,
                          mymodel.get,
                          'wrong_key')
        self.assertRaises(AssertionError,
                          mymodel.set,
                          not_a_string, None)
        self.assertRaises(helas_call_writers.HelasCallWriter.PhysicsObjectError,
                          mymodel.set,
                          'wrong_subclass', None)
        # add_wavefunction and add_amplitude
        self.assertRaises(AssertionError,
                          mymodel.add_wavefunction,
                          'wrong_subclass', None)
        self.assertRaises(AssertionError,
                          mymodel.add_wavefunction,
                          (1, 2), "not_a_function")
        self.assertRaises(AssertionError,
                          mymodel.add_amplitude,
                          'wrong_subclass', None)
        self.assertRaises(AssertionError,
                          mymodel.add_amplitude,
                          (1, 2), "not_a_function")

    def test_set_wavefunctions(self):
        """Test wavefunction dictionary in HelasModel"""

        wavefunctions = {}
        # IXXXXXX.Key: (spin, state)
        key1 = ((-2, 0), ('',))
        wavefunctions[key1] = \
                          lambda wf: 'CALL IXXXXX(P(0,%d),%s,NHEL(%d),%d*IC(%d),W(1,%d))' % \
                          (wf.get('number_external'), wf.get('mass'),
                           wf.get('number_external'), -(-1) ** wf.get_with_flow('is_part'),
                           wf.get('number_external'), wf.get('number'))
        # OXXXXXX.Key: (spin, state)
        key2 = ((2, 0), ('',))
        wavefunctions[key2] = \
                          lambda wf: 'CALL OXXXXX(P(0,%d),%s,NHEL(%d),%d*IC(%d),W(1,%d))' % \
                          (wf.get('number_external'), wf.get('mass'),
                           wf.get('number_external'), 1 ** wf.get_with_flow('is_part'),
                           wf.get('number_external'), wf.get('number'))

        self.assert_(self.mymodel.set('wavefunctions', wavefunctions))

        wf = helas_objects.HelasWavefunction()
        wf.set('particle', -2, self.mybasemodel)
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

        del wavefunctions[key1]
        del wavefunctions[key2]
        

#===============================================================================
# FortranHelasCallWriterTest
#===============================================================================
class FortranHelasCallWriterTest(HelasModelTestSetup):
    """Test class for the FortranHelasCallWriterTest object"""

    def test_generate_wavefunctions_and_amplitudes(self):
        """Test automatic generation of wavefunction and amplitude calls"""

        goal = [ \
            'CALL IXXXXX(P(0,1),me,NHEL(1),+1*IC(1),W(1,1))',
            'CALL OXXXXX(P(0,2),me,NHEL(2),-1*IC(2),W(1,2))',
            'CALL VXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,3))',
            'CALL FVOXXX(W(1,2),W(1,3),MGVX12,me,zero,W(1,1))',
            'CALL FVIXXX(W(1,1),W(1,3),MGVX12,me,zero,W(1,2))',
            'CALL JIOXXX(W(1,1),W(1,2),MGVX12,zero,zero,W(1,3))',
            'CALL IOVXXX(W(1,1),W(1,2),W(1,3),MGVX12,AMP(1))',
            'CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))',
            'CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))',
            'CALL TXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,3))',
            'CALL JVTAXX(W(1,2),W(1,3),MGVX2,zero,zero,W(1,1))',
            'CALL JVTAXX(W(1,1),W(1,3),MGVX2,zero,zero,W(1,2))',
            'CALL UVVAXX(W(1,1),W(1,2),MGVX2,zero,zero,zero,W(1,3))',
            'CALL VVTAXX(W(1,1),W(1,2),W(1,3),MGVX2,zero,AMP(2))',
            'CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))',
            'CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))',
            'CALL SXXXXX(P(0,3),-1*IC(3),W(1,3))',
            'CALL SXXXXX(P(0,4),-1*IC(4),W(1,4))',
            'CALL JVSSXX(W(1,2),W(1,3),W(1,4),MGVX89,zero,zero,W(1,1))',
            'CALL JVSSXX(W(1,1),W(1,3),W(1,4),MGVX89,zero,zero,W(1,2))',
            'CALL HVVSXX(W(1,2),W(1,1),W(1,4),MGVX89,Musq2,Wusq2,W(1,3))',
            'CALL HVVSXX(W(1,2),W(1,1),W(1,3),MGVX89,Musq2,Wusq2,W(1,4))',
            'CALL VVSSXX(W(1,2),W(1,1),W(1,3),W(1,4),MGVX89,AMP(1))']

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                           'number': 1,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                           'number': 2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'number': 3,
                                         'state':False}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 7,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = helas_call_writers.FortranHelasCallWriter()

        goal_counter = 0

        for wf in wfs:
            self.assertEqual(fortran_model.get_wavefunction_call(wf),
                             goal[goal_counter])
            goal_counter = goal_counter + 1

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers', mothers)
            if not wf.get('self_antipart'):
                wf.flip_part_antipart()
            self.assertEqual(fortran_model.get_wavefunction_call(wf),
                             goal[goal_counter])
            if not wf.get('self_antipart'):
                wf.flip_part_antipart()
            goal_counter = goal_counter + 1

        amplitude = helas_objects.HelasAmplitude({\
            'mothers': wfs,
            'number': 1})
        amplitude.set('interaction_id', 7, self.mybasemodel)

        self.assertEqual(fortran_model.get_amplitude_call(amplitude),
                         goal[goal_counter])
        goal_counter = goal_counter + 1

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'number': 1,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'number': 2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id': 8000002,
                                           'number': 3,
                                           'state':False}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 5,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = helas_call_writers.FortranHelasCallWriter()

        for wf in wfs:
            self.assertEqual(fortran_model.get_wavefunction_call(wf),
                             goal[goal_counter])
            goal_counter = goal_counter + 1

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers', mothers)
            self.assertEqual(fortran_model.get_wavefunction_call(wf),
                             goal[goal_counter])
            goal_counter = goal_counter + 1

        amplitude = helas_objects.HelasAmplitude({\
            'mothers': wfs,
            'number': 2})
        amplitude.set('interaction_id', 5, self.mybasemodel)
        self.assertEqual(fortran_model.get_amplitude_call(amplitude),
                         goal[goal_counter])
        goal_counter = goal_counter + 1






    def test_w_and_z_amplitudes(self):
        """Test wavefunction and amplitude calls for W and Z"""

        goal = [ \
            'CALL JWWWXX(W(1,2),W(1,3),W(1,4),MGVX6,wmas,wwid,W(1,1))',
            'CALL JWWWXX(W(1,1),W(1,3),W(1,4),MGVX6,wmas,wwid,W(1,2))',
            'CALL JWWWXX(W(1,1),W(1,2),W(1,4),MGVX6,wmas,wwid,W(1,3))',
            'CALL JWWWXX(W(1,1),W(1,2),W(1,3),MGVX6,wmas,wwid,W(1,4))',
            '# Amplitude(s) for diagram number 1',
            'CALL WWWWXX(W(1,1),W(1,2),W(1,3),W(1,4),MGVX6,AMP(1))',
            'CALL JW3WXX(W(1,2),W(1,3),W(1,4),MGVX8,wmas,wwid,W(1,1))',
            'CALL JW3WXX(W(1,1),W(1,3),W(1,4),MGVX8,wmas,wwid,W(1,2))',
            'CALL JW3WXX(W(1,1),W(1,2),W(1,4),MGVX8,zmas,zwid,W(1,3))',
            'CALL JW3WXX(W(1,1),W(1,2),W(1,3),MGVX8,zmas,zwid,W(1,4))',
            '# Amplitude(s) for diagram number 1',
            'CALL W3W3XX(W(1,1),W(1,2),W(1,3),W(1,4),MGVX8,AMP(1))']

        goal_counter = 0

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                           'number': 1,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                           'number': 2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id': 24,
                                           'number': 3,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                           'number': 4,
                                         'state':False}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 8,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = helas_call_writers.FortranHelasCallWriter()

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers', mothers)
            # Not yet implemented special wavefunctions for W/Z
            #self.assertEqual(fortran_model.get_wavefunction_call(wf),
            #                 goal[goal_counter])
            goal_counter = goal_counter + 1

        amplitude = helas_objects.HelasAmplitude({\
            'mothers': wfs,
            'number': 1})
        amplitude.set('interaction_id', 8, self.mybasemodel)
        # Not yet implemented special wavefunctions for W/Z
        #self.assertEqual(fortran_model.get_amplitude_call(amplitude),
        #                 goal[goal_counter])
        goal_counter = goal_counter + 1

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                           'number': 1,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                           'number': 2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id': 23,
                                           'number': 3,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id': 23,
                                           'number': 4,
                                         'state':False}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 9,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = helas_call_writers.FortranHelasCallWriter()

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers', mothers)
            # Not yet implemented special wavefunctions for W/Z
            # self.assertEqual(fortran_model.get_wavefunction_call(wf),
            #                 goal[goal_counter])
            goal_counter = goal_counter + 1


        amplitude = helas_objects.HelasAmplitude({\
            'mothers': wfs,
            'number': 1})
        amplitude.set('interaction_id', 9, self.mybasemodel)
        # Not yet implemented special wavefunctions for W/Z
        #self.assertEqual(fortran_model.get_amplitude_call(amplitude),
        #                 goal[goal_counter])
        goal_counter = goal_counter + 1


class UFOHELASCallWriterTest(unittest.TestCase):
    """Test class for the FortranHelasCallWriterTest object"""

    mybasemodel = base_objects.Model()


    def setUp(self):
        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

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
        a = mypartlist[len(mypartlist) - 1]

        # W+ and W-
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
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
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # Z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
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
        z = mypartlist[len(mypartlist) - 1]

        # a-a-w+w- 4-vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [a, \
                                             a,
                                             wminus,
                                             wplus]),
                      'color': [],
                      'lorentz':['VVVV1'],
                      'couplings':{(0, 0):'GC_51'},
                      'orders':{'QED':2}}))

        # w+w-z vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [wminus,
                                             wplus,
                                             z]),
                      'color': [],
                      'lorentz':['VVV1'],
                      'couplings':{(0, 0):'GC_12'},
                      'orders':{'QED':1}}))

        self.mybasemodel.set('particles', mypartlist)
        self.mybasemodel.set('interactions', myinterlist)
        self.mybasemodel.set('name', 'sm')
        
        
        #import madgraph.interface.cmd_interface as cmd
        #CMD = cmd.MadGraphCmdShell()
        #CMD._curr_model = self.mybasemodel
        #CMD._curr_fortran_model = helas_call_writers.FortranUFOHelasCallWriter 
        #CMD.do_generate('a w- > w- a z')
        #CMD.do_export('matrix_v4 /tmp/')
        
        
        
        leg1 = base_objects.Leg({'id':22,'state':False})
        leg2 = base_objects.Leg({'id':24,'state':False})
        leg3 = base_objects.Leg({'id':22,'state':True})
        leg4 = base_objects.Leg({'id':24,'state':True})
        leg5 = base_objects.Leg({'id':23,'state':True}) 

        legList1 = base_objects.LegList([leg1, leg2, leg3, leg4, leg5])
        
        myproc = base_objects.Process({'legs':legList1,
                                       'model':self.mybasemodel})
        
        myamplitude = diagram_generation.Amplitude({'process': myproc})
        
        self.mymatrixelement = helas_objects.HelasMatrixElement(myamplitude)
        
    def test_UFO_fortran_helas_call_writer(self):
        """Test automatic generation of UFO helas calls in Fortran"""
        
        fortran_model = helas_call_writers.FortranUFOHelasCallWriter(\
            self.mybasemodel)
        
        result = fortran_model.get_matrix_element_calls(self.mymatrixelement)
        solution =['CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))',
                   'CALL VXXXXX(P(0,2),wmas,NHEL(2),-1*IC(2),W(1,2))',
                   'CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))',
                   'CALL VXXXXX(P(0,4),wmas,NHEL(4),+1*IC(4),W(1,4))',
                   'CALL VXXXXX(P(0,5),zmas,NHEL(5),+1*IC(5),W(1,5))',
                   'CALL VVVV1_4(W(1,1),W(1,3),W(1,2),GC_51,wmas,wwid,W(1,6))',
                   '# Amplitude(s) for diagram number 1',
                   'CALL VVV1_0(W(1,6),W(1,4),W(1,5),GC_12,AMP(1))',
                   'CALL VVVV1_3(W(1,1),W(1,3),W(1,4),GC_51,wmas,wwid,W(1,6))',
                   '# Amplitude(s) for diagram number 2',
                   'CALL VVV1_0(W(1,2),W(1,6),W(1,5),GC_12,AMP(2))']

        
        for i, line in enumerate(solution):
            self.assertEqual(line, result[i])

        
    def test_UFO_CPP_helas_call_writer(self):
        """Test automatic generation of UFO helas calls in C++"""
        
        cpp_model = helas_call_writers.CPPUFOHelasCallWriter(\
            self.mybasemodel)
        
        result = cpp_model.get_matrix_element_calls(self.mymatrixelement)
        solution =['vxxxxx(p[perm[0]],mME[0],hel[0],-1,w[0]);',
                   'vxxxxx(p[perm[1]],mME[1],hel[1],-1,w[1]);',
                   'vxxxxx(p[perm[2]],mME[2],hel[2],+1,w[2]);',
                   'vxxxxx(p[perm[3]],mME[3],hel[3],+1,w[3]);',
                   'vxxxxx(p[perm[4]],mME[4],hel[4],+1,w[4]);',
                   'VVVV1_4(w[0],w[2],w[1],pars->GC_51,pars->wmas,pars->wwid,w[5]);',
                   '# Amplitude(s) for diagram number 1',
                   'VVV1_0(w[5],w[3],w[4],pars->GC_12,amp[0]);',
                   'VVVV1_3(w[0],w[2],w[3],pars->GC_51,pars->wmas,pars->wwid,w[5]);',
                   '# Amplitude(s) for diagram number 2',
                   'VVV1_0(w[1],w[5],w[4],pars->GC_12,amp[1]);']
        
        for i, line in enumerate(solution):
            self.assertEqual(line, result[i])
        

    def test_UFO_Python_helas_call_writer(self):
        """Test automatic generation of UFO helas calls in Python"""
        
        cpp_model = helas_call_writers.PythonUFOHelasCallWriter(\
            self.mybasemodel)
        
        result = cpp_model.get_matrix_element_calls(self.mymatrixelement)
        solution =['w[0] = vxxxxx(p[0],zero,hel[0],-1)',
                   'w[1] = vxxxxx(p[1],wmas,hel[1],-1)',
                   'w[2] = vxxxxx(p[2],zero,hel[2],+1)',
                   'w[3] = vxxxxx(p[3],wmas,hel[3],+1)',
                   'w[4] = vxxxxx(p[4],zmas,hel[4],+1)',
                   'w[5]= VVVV1_4(w[0],w[2],w[1],GC_51,wmas,wwid)',
                   '# Amplitude(s) for diagram number 1',
                   'amp[0]= VVV1_0(w[5],w[3],w[4],GC_12)',
                   'w[5]= VVVV1_3(w[0],w[2],w[3],GC_51,wmas,wwid)',
                   '# Amplitude(s) for diagram number 2',
                   'amp[1]= VVV1_0(w[1],w[5],w[4],GC_12)']
        
        
        for i, line in enumerate(solution):
            self.assertEqual(line, result[i])
        

class UFOHELASCALLWriterComplexMass(unittest.TestCase):
    """testing the writting in case of complex mass scheme"""

    @test_aloha.set_global(cms=True) 
    def setUp(self):
        """load the model"""

        import madgraph.interface.master_interface as interface
        cmd = interface.MasterCmd()
        cmd.do_import('model sm')
        
        self.mybasemodel = cmd._curr_model
        self.mybasemodel.change_mass_to_complex_scheme()

        leg1 = base_objects.Leg({'id':22,'state':False})
        leg2 = base_objects.Leg({'id':24,'state':False})
        leg3 = base_objects.Leg({'id':22,'state':True})
        leg4 = base_objects.Leg({'id':24,'state':True})
        leg5 = base_objects.Leg({'id':23,'state':True}) 

        legList1 = base_objects.LegList([leg1, leg2, leg3, leg4, leg5])
        
        myproc = base_objects.Process({'legs':legList1,
                                       'model':self.mybasemodel})
        
        myamplitude = diagram_generation.Amplitude({'process': myproc})
        
        self.mymatrixelement = helas_objects.HelasMatrixElement(myamplitude)

    @test_aloha.set_global(cms=True)        
    def test_UFO_fortran_helas_call_writer(self):
        """Test automatic generation of UFO helas calls in Fortran"""
        
        fortran_model = helas_call_writers.FortranUFOHelasCallWriter(\
            self.mybasemodel)
        
        result = fortran_model.get_matrix_element_calls(self.mymatrixelement)
        solution = """CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),mdl_MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),mdl_MW,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),mdl_MZ,NHEL(5),+1*IC(5),W(1,5))
CALL VVV1_3(W(1,1),W(1,2),GC_4,DCMPLX(CMASS_mdl_MW),W(1,6))
CALL VVV1_2(W(1,3),W(1,4),GC_4,DCMPLX(CMASS_mdl_MW),W(1,7))
# Amplitude(s) for diagram number 1
CALL VVV1_0(W(1,6),W(1,7),W(1,5),GC_53,AMP(1))
CALL VVV1_1(W(1,4),W(1,5),GC_53,DCMPLX(CMASS_mdl_MW),W(1,8))
# Amplitude(s) for diagram number 2
CALL VVV1_0(W(1,3),W(1,6),W(1,8),GC_4,AMP(2))
# Amplitude(s) for diagram number 3
CALL VVVV5_0(W(1,3),W(1,6),W(1,4),W(1,5),GC_57,AMP(3))
CALL VVV1_2(W(1,1),W(1,4),GC_4,DCMPLX(CMASS_mdl_MW),W(1,6))
CALL VVV1_3(W(1,3),W(1,2),GC_4,DCMPLX(CMASS_mdl_MW),W(1,9))
# Amplitude(s) for diagram number 4
CALL VVV1_0(W(1,9),W(1,6),W(1,5),GC_53,AMP(4))
CALL VVV1_2(W(1,2),W(1,5),GC_53,DCMPLX(CMASS_mdl_MW),W(1,10))
# Amplitude(s) for diagram number 5
CALL VVV1_0(W(1,3),W(1,10),W(1,6),GC_4,AMP(5))
# Amplitude(s) for diagram number 6
CALL VVVV5_0(W(1,3),W(1,2),W(1,6),W(1,5),GC_57,AMP(6))
# Amplitude(s) for diagram number 7
CALL VVV1_0(W(1,1),W(1,9),W(1,8),GC_4,AMP(7))
# Amplitude(s) for diagram number 8
CALL VVV1_0(W(1,1),W(1,10),W(1,7),GC_4,AMP(8))
CALL VVVV2_4(W(1,1),W(1,3),W(1,2),GC_5,DCMPLX(CMASS_mdl_MW),W(1,10))
# Amplitude(s) for diagram number 9
CALL VVV1_0(W(1,10),W(1,4),W(1,5),GC_53,AMP(9))
CALL VVVV5_3(W(1,1),W(1,2),W(1,5),GC_57,DCMPLX(CMASS_mdl_MW),W(1,10))
# Amplitude(s) for diagram number 10
CALL VVV1_0(W(1,3),W(1,10),W(1,4),GC_4,AMP(10))
CALL VVVV2_3(W(1,1),W(1,3),W(1,4),GC_5,DCMPLX(CMASS_mdl_MW),W(1,10))
# Amplitude(s) for diagram number 11
CALL VVV1_0(W(1,2),W(1,10),W(1,5),GC_53,AMP(11))
CALL VVVV5_2(W(1,1),W(1,4),W(1,5),GC_57,DCMPLX(CMASS_mdl_MW),W(1,10))
# Amplitude(s) for diagram number 12
CALL VVV1_0(W(1,3),W(1,2),W(1,10),GC_4,AMP(12))"""

        self.assertEqual(solution.split('\n'), result)

    @test_aloha.set_global(cms=True)   
    def test_UFO_CPP_helas_call_writer(self):
        """Test automatic generation of UFO helas calls in C++"""
        
        cpp_model = helas_call_writers.CPPUFOHelasCallWriter(\
            self.mybasemodel)
        
        result = cpp_model.get_matrix_element_calls(self.mymatrixelement)
        
        solution = """vxxxxx(p[perm[0]],mME[0],hel[0],-1,w[0]);
vxxxxx(p[perm[1]],mME[1],hel[1],-1,w[1]);
vxxxxx(p[perm[2]],mME[2],hel[2],+1,w[2]);
vxxxxx(p[perm[3]],mME[3],hel[3],+1,w[3]);
vxxxxx(p[perm[4]],mME[4],hel[4],+1,w[4]);
VVV1_3(w[0],w[1],pars->GC_4,pars->CMASS_mdl_MW,w[5]);
VVV1_2(w[2],w[3],pars->GC_4,pars->CMASS_mdl_MW,w[6]);
# Amplitude(s) for diagram number 1
VVV1_0(w[5],w[6],w[4],pars->GC_53,amp[0]);
VVV1_1(w[3],w[4],pars->GC_53,pars->CMASS_mdl_MW,w[7]);
# Amplitude(s) for diagram number 2
VVV1_0(w[2],w[5],w[7],pars->GC_4,amp[1]);
# Amplitude(s) for diagram number 3
VVVV5_0(w[2],w[5],w[3],w[4],pars->GC_57,amp[2]);
VVV1_2(w[0],w[3],pars->GC_4,pars->CMASS_mdl_MW,w[5]);
VVV1_3(w[2],w[1],pars->GC_4,pars->CMASS_mdl_MW,w[8]);
# Amplitude(s) for diagram number 4
VVV1_0(w[8],w[5],w[4],pars->GC_53,amp[3]);
VVV1_2(w[1],w[4],pars->GC_53,pars->CMASS_mdl_MW,w[9]);
# Amplitude(s) for diagram number 5
VVV1_0(w[2],w[9],w[5],pars->GC_4,amp[4]);
# Amplitude(s) for diagram number 6
VVVV5_0(w[2],w[1],w[5],w[4],pars->GC_57,amp[5]);
# Amplitude(s) for diagram number 7
VVV1_0(w[0],w[8],w[7],pars->GC_4,amp[6]);
# Amplitude(s) for diagram number 8
VVV1_0(w[0],w[9],w[6],pars->GC_4,amp[7]);
VVVV2_4(w[0],w[2],w[1],pars->GC_5,pars->CMASS_mdl_MW,w[9]);
# Amplitude(s) for diagram number 9
VVV1_0(w[9],w[3],w[4],pars->GC_53,amp[8]);
VVVV5_3(w[0],w[1],w[4],pars->GC_57,pars->CMASS_mdl_MW,w[9]);
# Amplitude(s) for diagram number 10
VVV1_0(w[2],w[9],w[3],pars->GC_4,amp[9]);
VVVV2_3(w[0],w[2],w[3],pars->GC_5,pars->CMASS_mdl_MW,w[9]);
# Amplitude(s) for diagram number 11
VVV1_0(w[1],w[9],w[4],pars->GC_53,amp[10]);
VVVV5_2(w[0],w[3],w[4],pars->GC_57,pars->CMASS_mdl_MW,w[9]);
# Amplitude(s) for diagram number 12
VVV1_0(w[2],w[1],w[9],pars->GC_4,amp[11]);"""

        self.assertEqual(solution.split('\n'), result)
        
    @test_aloha.set_global(cms=True) 
    def test_UFO_Python_helas_call_writer(self):
        """Test automatic generation of UFO helas calls in Python"""
        
        cpp_model = helas_call_writers.PythonUFOHelasCallWriter(\
            self.mybasemodel)
        
        result = cpp_model.get_matrix_element_calls(self.mymatrixelement)
        solution = """w[0] = vxxxxx(p[0],ZERO,hel[0],-1)
w[1] = vxxxxx(p[1],mdl_MW,hel[1],-1)
w[2] = vxxxxx(p[2],ZERO,hel[2],+1)
w[3] = vxxxxx(p[3],mdl_MW,hel[3],+1)
w[4] = vxxxxx(p[4],mdl_MZ,hel[4],+1)
w[5]= VVV1_3(w[0],w[1],GC_4,CMASS_mdl_MW)
w[6]= VVV1_2(w[2],w[3],GC_4,CMASS_mdl_MW)
# Amplitude(s) for diagram number 1
amp[0]= VVV1_0(w[5],w[6],w[4],GC_53)
w[7]= VVV1_1(w[3],w[4],GC_53,CMASS_mdl_MW)
# Amplitude(s) for diagram number 2
amp[1]= VVV1_0(w[2],w[5],w[7],GC_4)
# Amplitude(s) for diagram number 3
amp[2]= VVVV5_0(w[2],w[5],w[3],w[4],GC_57)
w[5]= VVV1_2(w[0],w[3],GC_4,CMASS_mdl_MW)
w[8]= VVV1_3(w[2],w[1],GC_4,CMASS_mdl_MW)
# Amplitude(s) for diagram number 4
amp[3]= VVV1_0(w[8],w[5],w[4],GC_53)
w[9]= VVV1_2(w[1],w[4],GC_53,CMASS_mdl_MW)
# Amplitude(s) for diagram number 5
amp[4]= VVV1_0(w[2],w[9],w[5],GC_4)
# Amplitude(s) for diagram number 6
amp[5]= VVVV5_0(w[2],w[1],w[5],w[4],GC_57)
# Amplitude(s) for diagram number 7
amp[6]= VVV1_0(w[0],w[8],w[7],GC_4)
# Amplitude(s) for diagram number 8
amp[7]= VVV1_0(w[0],w[9],w[6],GC_4)
w[9]= VVVV2_4(w[0],w[2],w[1],GC_5,CMASS_mdl_MW)
# Amplitude(s) for diagram number 9
amp[8]= VVV1_0(w[9],w[3],w[4],GC_53)
w[9]= VVVV5_3(w[0],w[1],w[4],GC_57,CMASS_mdl_MW)
# Amplitude(s) for diagram number 10
amp[9]= VVV1_0(w[2],w[9],w[3],GC_4)
w[9]= VVVV2_3(w[0],w[2],w[3],GC_5,CMASS_mdl_MW)
# Amplitude(s) for diagram number 11
amp[10]= VVV1_0(w[1],w[9],w[4],GC_53)
w[9]= VVVV5_2(w[0],w[3],w[4],GC_57,CMASS_mdl_MW)
# Amplitude(s) for diagram number 12
amp[11]= VVV1_0(w[2],w[1],w[9],GC_4)"""
        
        self.assertEqual(solution.split('\n'), result)        

