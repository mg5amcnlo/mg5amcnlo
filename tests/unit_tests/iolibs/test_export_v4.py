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

"""Unit test library for the export v4 format routines"""

import StringIO
import copy
import fractions
import os 
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.iolibs.misc as misc
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.save_load_object as save_load_object        
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_algebra as color
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.core.color_amp as color_amp
import tests.unit_tests.core.test_helas_objects as test_helas_objects
import tests.unit_tests.iolibs.test_file_writers as test_file_writers
import tests.unit_tests.iolibs.test_helas_call_writers as \
                                            test_helas_call_writers

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')
#===============================================================================
# IOImportV4Test
#===============================================================================
class IOExportV4Test(unittest.TestCase,
                     test_file_writers.CheckFileCreate):
    """Test class for the export v4 module"""

    mymodel = base_objects.Model()
    mymatrixelement = helas_objects.HelasMatrixElement()
    myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)
    created_files = ['test'
                    ]

    def setUp(self):

        test_file_writers.CheckFileCreate.clean_files
        # Set up model

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

        self.mymodel.set('particles', mypartlist)
        self.mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mymodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.mymatrixelement = helas_objects.HelasMatrixElement(myamplitude)
        self.myfortranmodel.downcase = False

    tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_export_matrix_element_v4_standalone(self):
        """Test the result of exporting a matrix element to file"""

        goal_matrix_f = \
"""      SUBROUTINE SMATRIX(P,ANS)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     MadGraph StandAlone Version
C     
C     Returns amplitude squared summed/avg over colors
C     and helicities
C     for the point in phase space P(0:3,NEXTERNAL)
C     
C     Process: e+ e- > a a a
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=32)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
C     
C     LOCAL VARIABLES 
C     
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T
      REAL*8 MATRIX
      INTEGER IHEL,IDEN, I
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
      DATA (NHEL(I,   1),I=1,5) /-1,-1,-1,-1,-1/
      DATA (NHEL(I,   2),I=1,5) /-1,-1,-1,-1, 1/
      DATA (NHEL(I,   3),I=1,5) /-1,-1,-1, 1,-1/
      DATA (NHEL(I,   4),I=1,5) /-1,-1,-1, 1, 1/
      DATA (NHEL(I,   5),I=1,5) /-1,-1, 1,-1,-1/
      DATA (NHEL(I,   6),I=1,5) /-1,-1, 1,-1, 1/
      DATA (NHEL(I,   7),I=1,5) /-1,-1, 1, 1,-1/
      DATA (NHEL(I,   8),I=1,5) /-1,-1, 1, 1, 1/
      DATA (NHEL(I,   9),I=1,5) /-1, 1,-1,-1,-1/
      DATA (NHEL(I,  10),I=1,5) /-1, 1,-1,-1, 1/
      DATA (NHEL(I,  11),I=1,5) /-1, 1,-1, 1,-1/
      DATA (NHEL(I,  12),I=1,5) /-1, 1,-1, 1, 1/
      DATA (NHEL(I,  13),I=1,5) /-1, 1, 1,-1,-1/
      DATA (NHEL(I,  14),I=1,5) /-1, 1, 1,-1, 1/
      DATA (NHEL(I,  15),I=1,5) /-1, 1, 1, 1,-1/
      DATA (NHEL(I,  16),I=1,5) /-1, 1, 1, 1, 1/
      DATA (NHEL(I,  17),I=1,5) / 1,-1,-1,-1,-1/
      DATA (NHEL(I,  18),I=1,5) / 1,-1,-1,-1, 1/
      DATA (NHEL(I,  19),I=1,5) / 1,-1,-1, 1,-1/
      DATA (NHEL(I,  20),I=1,5) / 1,-1,-1, 1, 1/
      DATA (NHEL(I,  21),I=1,5) / 1,-1, 1,-1,-1/
      DATA (NHEL(I,  22),I=1,5) / 1,-1, 1,-1, 1/
      DATA (NHEL(I,  23),I=1,5) / 1,-1, 1, 1,-1/
      DATA (NHEL(I,  24),I=1,5) / 1,-1, 1, 1, 1/
      DATA (NHEL(I,  25),I=1,5) / 1, 1,-1,-1,-1/
      DATA (NHEL(I,  26),I=1,5) / 1, 1,-1,-1, 1/
      DATA (NHEL(I,  27),I=1,5) / 1, 1,-1, 1,-1/
      DATA (NHEL(I,  28),I=1,5) / 1, 1,-1, 1, 1/
      DATA (NHEL(I,  29),I=1,5) / 1, 1, 1,-1,-1/
      DATA (NHEL(I,  30),I=1,5) / 1, 1, 1,-1, 1/
      DATA (NHEL(I,  31),I=1,5) / 1, 1, 1, 1,-1/
      DATA (NHEL(I,  32),I=1,5) / 1, 1, 1, 1, 1/
      DATA IDEN/24/
C     ----------
C     BEGIN CODE
C     ----------
      NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
        JC(IHEL) = +1
      ENDDO
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
          T=MATRIX(P ,NHEL(1,IHEL),JC(1))
          ANS=ANS+T
          IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)
      END


      REAL*8 FUNCTION MATRIX(P,NHEL,IC)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     Returns amplitude squared summed/avg over colors
C     for the point with external lines W(0:6,NEXTERNAL)
C     
C     Process: e+ e- > a a a
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=6)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=11, NCOLOR=1)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'coupl.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/1/
      DATA (CF(I,1),I=1,1) /1/
C     ----------
C     BEGIN CODE
C     ----------
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1*IC(2),W(1,2))
      CALL VXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
      CALL VXXXXX(P(0,4),ZERO,NHEL(4),+1*IC(4),W(1,4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1*IC(5),W(1,5))
      CALL FVOXXX(W(1,1),W(1,3),MGVX12,ZERO,ZERO,W(1,6))
      CALL FVIXXX(W(1,2),W(1,4),MGVX12,ZERO,ZERO,W(1,7))
C     Amplitude(s) for diagram number 1
      CALL IOVXXX(W(1,7),W(1,6),W(1,5),MGVX12,AMP(1))
      CALL FVIXXX(W(1,2),W(1,5),MGVX12,ZERO,ZERO,W(1,8))
C     Amplitude(s) for diagram number 2
      CALL IOVXXX(W(1,8),W(1,6),W(1,4),MGVX12,AMP(2))
      CALL FVOXXX(W(1,1),W(1,4),MGVX12,ZERO,ZERO,W(1,9))
      CALL FVIXXX(W(1,2),W(1,3),MGVX12,ZERO,ZERO,W(1,10))
C     Amplitude(s) for diagram number 3
      CALL IOVXXX(W(1,10),W(1,9),W(1,5),MGVX12,AMP(3))
C     Amplitude(s) for diagram number 4
      CALL IOVXXX(W(1,8),W(1,9),W(1,3),MGVX12,AMP(4))
      CALL FVOXXX(W(1,1),W(1,5),MGVX12,ZERO,ZERO,W(1,11))
C     Amplitude(s) for diagram number 5
      CALL IOVXXX(W(1,10),W(1,11),W(1,4),MGVX12,AMP(5))
C     Amplitude(s) for diagram number 6
      CALL IOVXXX(W(1,7),W(1,11),W(1,3),MGVX12,AMP(6))
      JAMP(1)=-AMP(1)-AMP(2)-AMP(3)-AMP(4)-AMP(5)-AMP(6)

      MATRIX = 0.D0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
      ENDDO
      END
""" % misc.get_pkg_info()

        process_exporter = export_v4.ProcessExporterFortranSA()

        process_exporter.write_matrix_element_v4(\
            writers.FortranWriter(self.give_pos('test')),
            self.mymatrixelement,
            self.myfortranmodel)

        self.assertFileContains('test', goal_matrix_f)

    def test_coeff_string(self):
        """Test the coeff string for JAMP lines"""

        process_exporter = export_v4.ProcessExporterFortran()

        self.assertEqual(process_exporter.coeff(1,
                                         fractions.Fraction(1),
                                         False, 0), '+')

        self.assertEqual(process_exporter.coeff(-1,
                                         fractions.Fraction(1),
                                         False, 0), '-')

        self.assertEqual(process_exporter.coeff(-1,
                                         fractions.Fraction(-3),
                                         False, 0), '+3*')

        self.assertEqual(process_exporter.coeff(-1,
                                         fractions.Fraction(3, 5),
                                         True, -2), '-1./15.*imag1*')

    def test_export_matrix_element_v4_madevent_group(self):
        """Test the result of exporting a subprocess group matrix element"""

        # Setup a model

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

        # A Z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
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
        z = mypartlist[-1]

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             a]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             a]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQED'},
                      'orders':{'QED':1}}))

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [g] * 3),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['VVV1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GDZ1', (0, 0):'GDZ2'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # Set parameters
        external_parameters = [\
            base_objects.ParamCardVariable('zero', 0.,'DUM', 1),
            base_objects.ParamCardVariable('MZ', 91.,'MASS', 23),
            base_objects.ParamCardVariable('WZ', 2.,'DECAY', 23)]
        couplings = [\
            base_objects.ModelVariable('GQQ', '1.', 'complex'),
            base_objects.ModelVariable('GQED', '0.1', 'complex'),
            base_objects.ModelVariable('G', '1.', 'complex'),
            base_objects.ModelVariable('GUZ1', '0.1', 'complex'),
            base_objects.ModelVariable('GUZ2', '0.1', 'complex'),
            base_objects.ModelVariable('GDZ1', '0.05', 'complex'),
            base_objects.ModelVariable('GDZ2', '0.05', 'complex')]
        mymodel.set('parameters', {('external',): external_parameters})
        mymodel.set('couplings', {(): couplings})
        mymodel.set('functions', [])
                    


        procs = [[2,-2,21,21], [2,-2,2,-2], [2,-2,1,-1]]
        amplitudes = diagram_generation.AmplitudeList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel})
            my_amplitude = diagram_generation.Amplitude(my_process)
            amplitudes.append(my_amplitude)

        # Calculate diagrams for all processes

        amplitudes[1].set('has_mirror_process', True)
        subprocess_groups = group_subprocs.SubProcessGroup.\
                           group_amplitudes(amplitudes)
        self.assertEqual(len(subprocess_groups), 2)
        self.assertEqual(subprocess_groups[0].get('name'), 'qq_gg')
        self.assertEqual(subprocess_groups[1].get('name'), 'qq_qq')

        subprocess_group = subprocess_groups[1]
        matrix_elements = subprocess_group.get('matrix_elements')

        maxflows = 0
        for me in matrix_elements:
            maxflows = max(maxflows,
                           len(me.get('color_basis')))
        
        self.assertEqual(maxflows, 2)

        exporter = export_v4.ProcessExporterFortranMEGroup()

        # Test amp2 lines
        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_elements[0],
                                          subprocess_group.get('diagram_maps')[0])
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))+AMP(4)*dconjg(AMP(4))+AMP(5)*dconjg(AMP(5))+AMP(6)*dconjg(AMP(6))',
                          'AMP2(4)=AMP2(4)+AMP(7)*dconjg(AMP(7))',
                          'AMP2(5)=AMP2(5)+AMP(8)*dconjg(AMP(8))',
                          'AMP2(6)=AMP2(6)+AMP(9)*dconjg(AMP(9))+AMP(10)*dconjg(AMP(10))+AMP(11)*dconjg(AMP(11))+AMP(12)*dconjg(AMP(12))'])
        
        # Test configs.inc

        exporter.write_configs_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group,
            subprocess_group.get('diagrams_for_configs'))

        goal_configs = """C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA SPROP(-1,1)/21/
      DATA TPRID(-1,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/4,3/
      DATA SPROP(-1,2)/22/
      DATA TPRID(-1,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/4,3/
      DATA SPROP(-1,3)/23/
      DATA TPRID(-1,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/1,3/
      DATA TPRID(-1,4)/21/
      DATA SPROP(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/-1,4/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA (IFOREST(I,-1,5),I=1,2)/1,3/
      DATA TPRID(-1,5)/22/
      DATA SPROP(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/-1,4/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA (IFOREST(I,-1,6),I=1,2)/1,3/
      DATA TPRID(-1,6)/23/
      DATA SPROP(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/-1,4/
C     Number of configs
      DATA MAPCONFIG(0)/6/
"""
        #print open(self.give_pos('test')).read()

        self.assertFileContains('test', goal_configs)

        # Test config_subproc_map.inc

        exporter.write_config_subproc_map_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group.get('diagrams_for_configs'))

        goal_confsub = """      DATA (CONFSUB(I,1),I=1,2)/1,1/
      DATA (CONFSUB(I,2),I=1,2)/2,2/
      DATA (CONFSUB(I,3),I=1,2)/3,3/
      DATA (CONFSUB(I,4),I=1,2)/4,0/
      DATA (CONFSUB(I,5),I=1,2)/5,0/
      DATA (CONFSUB(I,6),I=1,2)/6,0/
"""
        
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_confsub)

        # Test coloramps.inc
        
        exporter.write_coloramps_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group.get('diagrams_for_configs'),
            maxflows,
            matrix_elements)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""      LOGICAL ICOLAMP(2,6,2)
      DATA(ICOLAMP(I,1,1),I=1,2)/.TRUE.,.TRUE./
      DATA(ICOLAMP(I,2,1),I=1,2)/.TRUE.,.FALSE./
      DATA(ICOLAMP(I,3,1),I=1,2)/.TRUE.,.FALSE./
      DATA(ICOLAMP(I,4,1),I=1,2)/.TRUE.,.TRUE./
      DATA(ICOLAMP(I,5,1),I=1,2)/.FALSE.,.TRUE./
      DATA(ICOLAMP(I,6,1),I=1,2)/.FALSE.,.TRUE./
      DATA(ICOLAMP(I,1,2),I=1,2)/.TRUE.,.TRUE./
      DATA(ICOLAMP(I,2,2),I=1,2)/.TRUE.,.FALSE./
      DATA(ICOLAMP(I,3,2),I=1,2)/.TRUE.,.FALSE./
""")

        # Test find_matrix_elements_for_configs

        self.assertEqual(\
            diagram_symmetry.find_matrix_elements_for_configs(subprocess_group),
            ([], {}))

        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(subprocess_group)

        self.assertEqual(symmetry, [1,1,1,1,1,1])
        self.assertEqual(perms,
                         [[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]])
        self.assertEqual(ident_perms,
                         [[0,1,2,3]])

        # Test processes.dat

        files.write_to_file(self.give_pos('test'),
                            exporter.write_processes_file,
                            subprocess_group)

        goal_processes = """1       u u~ > u u~
mirror  u~ u > u u~
2       u u~ > d d~
mirror  none"""
        
        self.assertFileContains('test', goal_processes)

        # Test mirrorprocs.inc

        exporter.write_mirrorprocs(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group)

        goal_mirror_inc = \
                 "      DATA (MIRRORPROCS(I),I=1,2)/.TRUE.,.FALSE./\n"
        
        self.assertFileContains('test', goal_mirror_inc)

        # Test matrix1.f
        exporter.write_matrix_element_v4(\
            writers.FortranWriter(self.give_pos('test')),
            matrix_elements[0],
            helas_call_writers.FortranUFOHelasCallWriter(mymodel),
            "1")

        goal_matrix1 = \
"""      SUBROUTINE SMATRIX1(P,ANS)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     MadGraph for Madevent Version
C     
C     Returns amplitude squared summed/avg over colors
C     and helicities
C     for the point in phase space P(0:3,NEXTERNAL)
C     
C     Process: u u~ > u u~
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'genps.inc'
      INCLUDE 'maxconfigs.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=16)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=12)
      INTEGER    NDIAGS
      PARAMETER (NDIAGS=6)
      INTEGER    THEL
      PARAMETER (THEL=2*NCOMB)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
C     
C     LOCAL VARIABLES 
C     
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY(2)
      REAL*8 T,MATRIX1
      REAL*8 R,SUMHEL,TS(NCOMB)
      INTEGER I,IDEN
      INTEGER IPROC,JC(NEXTERNAL),II
      LOGICAL GOODHEL(2,NCOMB)
      REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
      INTEGER IDUM, NGOOD(2), IGOOD(2,NCOMB), JHEL(2), J, JJ
      REAL     XRAN1
      EXTERNAL XRAN1
C     
C     GLOBAL VARIABLES
C     
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
      COMMON/TO_AMPS/  AMP2,       JAMP2

      CHARACTER*79         HEL_BUFF
      COMMON/TO_HELICITY/  HEL_BUFF

      INTEGER IMIRROR
      COMMON/TO_MIRROR/ IMIRROR

      REAL*8 POL(2)
      COMMON/TO_POLARIZATION/ POL

      INTEGER          ISUM_HEL
      LOGICAL                    MULTI_CHANNEL
      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
      DATA NTRY,IDUM /0,0,0/
      DATA XTRY, XREJ, NGOOD /0,0,0,0/
      SAVE YFRAC, IGOOD, JHEL
      DATA GOODHEL/THEL*.FALSE./
      DATA (NHEL(I,   1),I=1,4) /-1,-1,-1,-1/
      DATA (NHEL(I,   2),I=1,4) /-1,-1,-1, 1/
      DATA (NHEL(I,   3),I=1,4) /-1,-1, 1,-1/
      DATA (NHEL(I,   4),I=1,4) /-1,-1, 1, 1/
      DATA (NHEL(I,   5),I=1,4) /-1, 1,-1,-1/
      DATA (NHEL(I,   6),I=1,4) /-1, 1,-1, 1/
      DATA (NHEL(I,   7),I=1,4) /-1, 1, 1,-1/
      DATA (NHEL(I,   8),I=1,4) /-1, 1, 1, 1/
      DATA (NHEL(I,   9),I=1,4) / 1,-1,-1,-1/
      DATA (NHEL(I,  10),I=1,4) / 1,-1,-1, 1/
      DATA (NHEL(I,  11),I=1,4) / 1,-1, 1,-1/
      DATA (NHEL(I,  12),I=1,4) / 1,-1, 1, 1/
      DATA (NHEL(I,  13),I=1,4) / 1, 1,-1,-1/
      DATA (NHEL(I,  14),I=1,4) / 1, 1,-1, 1/
      DATA (NHEL(I,  15),I=1,4) / 1, 1, 1,-1/
      DATA (NHEL(I,  16),I=1,4) / 1, 1, 1, 1/
      DATA IDEN/36/
C     ----------
C     BEGIN CODE
C     ----------
      NTRY(IMIRROR)=NTRY(IMIRROR)+1
      DO I=1,NEXTERNAL
        JC(I) = +1
      ENDDO

      IF (MULTI_CHANNEL) THEN
        DO I=1,NDIAGS
          AMP2(I)=0D0
        ENDDO
        JAMP2(0)=2
        DO I=1,INT(JAMP2(0))
          JAMP2(I)=0D0
        ENDDO
      ENDIF
      ANS = 0D0
      WRITE(HEL_BUFF,'(16I5)') (0,I=1,NEXTERNAL)
      DO I=1,NCOMB
        TS(I)=0D0
      ENDDO
      IF (ISUM_HEL .EQ. 0 .OR. NTRY(IMIRROR) .LT. 10) THEN
        DO I=1,NCOMB
          IF (GOODHEL(IMIRROR,I) .OR. NTRY(IMIRROR) .LT. 2) THEN
            T=MATRIX1(P ,NHEL(1,I),JC(1))
            DO JJ=1,NINCOMING
              IF(POL(JJ).NE.1D0.AND.NHEL(JJ,I).EQ.INT(SIGN(1D0
     $         ,POL(JJ)))) THEN
                T=T*ABS(POL(JJ))
              ELSE IF(POL(JJ).NE.1D0)THEN
                T=T*(2D0-ABS(POL(JJ)))
              ENDIF
            ENDDO
            ANS=ANS+T
            TS(I)=T
            IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IMIRROR,I)) THEN
              GOODHEL(IMIRROR,I)=.TRUE.
              NGOOD(IMIRROR) = NGOOD(IMIRROR) +1
              IGOOD(IMIRROR,NGOOD(IMIRROR)) = I
            ENDIF
          ENDIF
        ENDDO
        JHEL(IMIRROR) = 1
        ISUM_HEL=MIN(ISUM_HEL,NGOOD(IMIRROR))
      ELSE  !RANDOM HELICITY
        DO J=1,ISUM_HEL
          JHEL(IMIRROR)=JHEL(IMIRROR)+1
          IF (JHEL(IMIRROR) .GT. NGOOD(IMIRROR)) JHEL(IMIRROR)=1
          HWGT = REAL(NGOOD(IMIRROR))/REAL(ISUM_HEL)
          I = IGOOD(IMIRROR,JHEL(IMIRROR))
          T=MATRIX1(P ,NHEL(1,I),JC(1))
          DO JJ=1,NINCOMING
            IF(POL(JJ).NE.1D0.AND.NHEL(JJ,I).EQ.INT(SIGN(1D0,POL(JJ)))
     $       ) THEN
              T=T*ABS(POL(JJ))
            ELSE IF(POL(JJ).NE.1D0)THEN
              T=T*(2D0-ABS(POL(JJ)))
            ENDIF
          ENDDO
          ANS=ANS+T*HWGT
          TS(I)=T*HWGT
        ENDDO
        IF (ISUM_HEL .EQ. 1) THEN
          WRITE(HEL_BUFF,'(16i5)')(NHEL(II,I),II=1,NEXTERNAL)
        ENDIF
      ENDIF
      IF (ISUM_HEL .NE. 1) THEN
        R=XRAN1(IDUM)*ANS
        SUMHEL=0D0
        DO I=1,NCOMB
          SUMHEL=SUMHEL+TS(I)
          IF(R.LT.SUMHEL)THEN
            WRITE(HEL_BUFF,'(16i5)')(NHEL(II,I),II=1,NEXTERNAL)
            GOTO 10
          ENDIF
        ENDDO
 10     CONTINUE
      ENDIF
      IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,NDIAGS
          XTOT=XTOT+AMP2(I)
        ENDDO
        IF (XTOT.NE.0D0) THEN
          ANS=ANS*AMP2(SUBDIAG(1))/XTOT
        ELSE
          ANS=0D0
        ENDIF
      ENDIF
      ANS=ANS/DBLE(IDEN)
      END


      REAL*8 FUNCTION MATRIX1(P,NHEL,IC)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     Returns amplitude squared summed/avg over colors
C     for the point with external lines W(0:6,NEXTERNAL)
C     
C     Process: u u~ > u u~
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=12)
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INTEGER    NWAVEFUNCS,     NCOLOR
      PARAMETER (NWAVEFUNCS=12, NCOLOR=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
C     
C     GLOBAL VARIABLES
C     
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
      COMMON/TO_AMPS/  AMP2,       JAMP2
      INCLUDE 'coupl.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/1/
      DATA (CF(I,  1),I=  1,  2) /    9,    3/
C     1 T(2,1) T(3,4)
      DATA DENOM(2)/1/
      DATA (CF(I,  2),I=  1,  2) /    3,    9/
C     1 T(2,4) T(3,1)
C     ----------
C     BEGIN CODE
C     ----------
      CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1*IC(1),W(1,1))
      CALL OXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL OXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
      CALL IXXXXX(P(0,4),ZERO,NHEL(4),-1*IC(4),W(1,4))
      CALL FFV1_3(W(1,1),W(1,2),GQQ,ZERO, ZERO, W(1,5))
C     Amplitude(s) for diagram number 1
      CALL FFV1_0(W(1,4),W(1,3),W(1,5),GQQ,AMP(1))
      CALL FFV1_3(W(1,1),W(1,2),GQED,ZERO, ZERO, W(1,6))
C     Amplitude(s) for diagram number 2
      CALL FFV1_0(W(1,4),W(1,3),W(1,6),GQED,AMP(2))
      CALL FFV1_3(W(1,1),W(1,2),GUZ1,MZ, WZ, W(1,7))
      CALL FFV2_3(W(1,1),W(1,2),GUZ2,MZ, WZ, W(1,8))
C     Amplitude(s) for diagram number 3
      CALL FFV1_0(W(1,4),W(1,3),W(1,7),GUZ1,AMP(3))
      CALL FFV2_0(W(1,4),W(1,3),W(1,7),GUZ2,AMP(4))
      CALL FFV1_0(W(1,4),W(1,3),W(1,8),GUZ1,AMP(5))
      CALL FFV2_0(W(1,4),W(1,3),W(1,8),GUZ2,AMP(6))
      CALL FFV1_3(W(1,1),W(1,3),GQQ,ZERO, ZERO, W(1,9))
C     Amplitude(s) for diagram number 4
      CALL FFV1_0(W(1,4),W(1,2),W(1,9),GQQ,AMP(7))
      CALL FFV1_3(W(1,1),W(1,3),GQED,ZERO, ZERO, W(1,10))
C     Amplitude(s) for diagram number 5
      CALL FFV1_0(W(1,4),W(1,2),W(1,10),GQED,AMP(8))
      CALL FFV1_3(W(1,1),W(1,3),GUZ1,MZ, WZ, W(1,11))
      CALL FFV2_3(W(1,1),W(1,3),GUZ2,MZ, WZ, W(1,12))
C     Amplitude(s) for diagram number 6
      CALL FFV1_0(W(1,4),W(1,2),W(1,11),GUZ1,AMP(9))
      CALL FFV2_0(W(1,4),W(1,2),W(1,11),GUZ2,AMP(10))
      CALL FFV1_0(W(1,4),W(1,2),W(1,12),GUZ1,AMP(11))
      CALL FFV2_0(W(1,4),W(1,2),W(1,12),GUZ2,AMP(12))
      JAMP(1)=+1./6.*AMP(1)-AMP(2)-AMP(3)-AMP(4)-AMP(5)-AMP(6)
     $ +1./2.*AMP(7)
      JAMP(2)=-1./2.*AMP(1)-1./6.*AMP(7)+AMP(8)+AMP(9)+AMP(10)+AMP(11)
     $ +AMP(12)
      MATRIX1 = 0.D0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        MATRIX1=MATRIX1+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
      ENDDO
      AMP2(1)=AMP2(1)+AMP(1)*DCONJG(AMP(1))
      AMP2(2)=AMP2(2)+AMP(2)*DCONJG(AMP(2))
      AMP2(3)=AMP2(3)+AMP(3)*DCONJG(AMP(3))+AMP(4)*DCONJG(AMP(4))
     $ +AMP(5)*DCONJG(AMP(5))+AMP(6)*DCONJG(AMP(6))
      AMP2(4)=AMP2(4)+AMP(7)*DCONJG(AMP(7))
      AMP2(5)=AMP2(5)+AMP(8)*DCONJG(AMP(8))
      AMP2(6)=AMP2(6)+AMP(9)*DCONJG(AMP(9))+AMP(10)*DCONJG(AMP(10))
     $ +AMP(11)*DCONJG(AMP(11))+AMP(12)*DCONJG(AMP(12))
      DO I = 1, NCOLOR
        JAMP2(I)=JAMP2(I)+JAMP(I)*DCONJG(JAMP(I))
      ENDDO

      END
""" % misc.get_pkg_info()
        
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_matrix1)

        # Test auto_dsig,f
        exporter.write_auto_dsig_file(\
            writers.FortranWriter(self.give_pos('test')),
            matrix_elements[0],
            "1")

        goal_auto_dsig1 = \
"""      DOUBLE PRECISION FUNCTION DSIG1(PP,WGT,IMODE)
C     ****************************************************
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     Process: u u~ > u u~
C     
C     RETURNS DIFFERENTIAL CROSS SECTION
C     Input:
C     pp    4 momentum of external particles
C     wgt   weight from Monte Carlo
C     imode 0 run, 1 init, 2 reweight, 
C     3 finalize, 4 only PDFs
C     Output:
C     Amplitude squared and summed
C     ****************************************************
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379.66*1000)  !CONV TO PICOBARNS
      REAL*8     PI
      PARAMETER (PI=3.1415926D0)
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER IMODE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,ITYPE,LP
      DOUBLE PRECISION U1,UB1,D1,DB1,C1,CB1,S1,SB1,B1,BB1
      DOUBLE PRECISION U2,UB2,D2,DB2,C2,CB2,S2,SB2,B2,BB2
      DOUBLE PRECISION G1,G2
      DOUBLE PRECISION A1,A2
      DOUBLE PRECISION XPQ(-7:7)
      DOUBLE PRECISION DSIGUU
C     
C     EXTERNAL FUNCTIONS
C     
      LOGICAL PASSCUTS
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF
C     
C     GLOBAL VARIABLES
C     
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC

      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
      INCLUDE 'coupl.inc'
      INCLUDE 'run.inc'
C     
C     DATA
C     
      DATA U1,UB1,D1,DB1,C1,CB1,S1,SB1,B1,BB1/10*1D0/
      DATA U2,UB2,D2,DB2,C2,CB2,S2,SB2,B2,BB2/10*1D0/
      DATA A1,G1/2*1D0/
      DATA A2,G2/2*1D0/
C     ----------
C     BEGIN CODE
C     ----------
      DSIG1=0D0

C     Only run if IMODE is 0
      IF(IMODE.NE.0.AND.IMODE.NE.4) RETURN


      IF (ABS(LPP(IB(1))).GE.1) THEN
        LP=SIGN(1,LPP(IB(1)))
        U1=PDG2PDF(ABS(LPP(IB(1))),2*LP,XBK(IB(1)),DSQRT(Q2FACT(IB(1))
     $   ))
      ENDIF
      IF (ABS(LPP(IB(2))).GE.1) THEN
        LP=SIGN(1,LPP(IB(2)))
        UB2=PDG2PDF(ABS(LPP(IB(2))),-2*LP,XBK(IB(2)),DSQRT(Q2FACT(IB(2
     $   ))))
      ENDIF
      PD(0) = 0D0
      IPROC = 0
      IPROC=IPROC+1  ! u u~ > u u~
      PD(IPROC)=PD(IPROC-1) + U1*UB2
      IF (IMODE.EQ.4)THEN
        DSIG1 = PD(IPROC)
        RETURN
      ENDIF
      CALL SMATRIX1(PP,DSIGUU)
      DSIGUU=DSIGUU*REWGT(PP)
      IF (DSIGUU.LT.1D199) THEN
        DSIG1=PD(IPROC)*CONV*DSIGUU
      ELSE
        WRITE(*,*) 'Error in matrix element'
        DSIGUU=0D0
        DSIG1=0D0
      ENDIF
      IF(IMODE.EQ.0.AND.DSIG1.GT.0D0)THEN
C       Call UNWGT to unweight and store events
        CALL UNWGT(PP,DSIG1*WGT,1)
      ENDIF

      END

""" % misc.get_pkg_info()
        

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_auto_dsig1)

        # Test super auto_dsig.f
        exporter.write_super_auto_dsig_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group)

        goal_super = \
        """      DOUBLE PRECISION FUNCTION DSIG(PP,WGT,IMODE)
C     ****************************************************
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     Process: u u~ > u u~
C     Process: u u~ > d d~
C     
C     RETURNS DIFFERENTIAL CROSS SECTION 
C     FOR MULTIPLE PROCESSES IN PROCESS GROUP
C     Input:
C     pp    4 momentum of external particles
C     wgt   weight from Monte Carlo
C     imode 0 run, 1 init, 2 reweight,
C     3 finalize, 4 only PDFs
C     Output:
C     Amplitude squared and summed
C     ****************************************************
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'genps.inc'
      INCLUDE 'maxconfigs.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      REAL*8     PI
      PARAMETER (PI=3.1415926D0)
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER IMODE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J,K,LUN,IDUM,ICONF,IMIRROR,NPROC
      DATA IDUM/0/
      SAVE NPROC,IDUM
      INTEGER SYMCONF(0:LMAXCONFIGS)
      SAVE SYMCONF
      DOUBLE PRECISION SUMPROB,TOTWGT,R,XDUM
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      INCLUDE 'config_subproc_map.inc'
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'
      LOGICAL MIRRORPROCS(LMAXCONFIGS)
      INCLUDE 'mirrorprocs.inc'
C     SELPROC is vector of selection weights for the subprocesses
C     SUMWGT is vector of total weight for the subprocesses
C     NUMEVTS is vector of event calls for the subprocesses
      DOUBLE PRECISION SELPROC(2, MAXSPROC,LMAXCONFIGS)
      DOUBLE PRECISION SUMWGT(2, MAXSPROC,LMAXCONFIGS)
      INTEGER NUMEVTS(2, MAXSPROC,LMAXCONFIGS)
      INTEGER LARGEDIM
      PARAMETER (LARGEDIM=2*MAXSPROC*LMAXCONFIGS)
      DATA SELPROC/LARGEDIM*0D0/
      DATA SUMWGT/LARGEDIM*0D0/
      DATA NUMEVTS/LARGEDIM*0/
      SAVE SELPROC,SUMWGT,NUMEVTS
C     
C     EXTERNAL FUNCTIONS
C     
      INTEGER NEXTUNOPEN
      REAL XRAN1
      DOUBLE PRECISION DSIGPROC
      EXTERNAL NEXTUNOPEN,XRAN1,DSIGPROC
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'coupl.inc'
      INCLUDE 'run.inc'
C     ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     IPROC has the present process number
      INTEGER IPROC
      COMMON/TO_IPROC/IPROC
      COMMON/TO_MIRROR/ IMIRROR
C     ----------
C     BEGIN CODE
C     ----------
      DSIG=0D0

      IF(IMODE.EQ.1)THEN
C       Set up process information from file symfact
        LUN=NEXTUNOPEN()
        OPEN(UNIT=LUN,FILE='../symfact.dat',STATUS='OLD',ERR=10)
        IPROC=1
        SYMCONF(IPROC)=ICONFIG
        DO I=1,MAPCONFIG(0)
          READ(LUN,*) XDUM, ICONF
          IF(ICONF.EQ.-MAPCONFIG(ICONFIG))THEN
            IPROC=IPROC+1
            SYMCONF(IPROC)=I
          ENDIF
        ENDDO
        SYMCONF(0)=IPROC
        CLOSE(LUN)
        RETURN
 10     WRITE(*,*)'Error opening symfact.dat. No permutations will be
     $    used.'
        RETURN
      ELSE IF(IMODE.EQ.2)THEN
C       Output weights and number of events
        SUMPROB=0D0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              SUMPROB=SUMPROB+SUMWGT(K,I,J)
            ENDDO
          ENDDO
        ENDDO
        WRITE(*,*)'Relative summed weights:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(4E12.4)')((SUMWGT(K,I,J)/SUMPROB,K=1,2),I=1
     $     ,MAXSPROC)
        ENDDO
        SUMPROB=0D0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              SUMPROB=SUMPROB+NUMEVTS(K,I,J)
            ENDDO
          ENDDO
        ENDDO
        WRITE(*,*)'Relative number of events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(4E12.4)')((NUMEVTS(K,I,J)/SUMPROB,K=1,2),I=1
     $     ,MAXSPROC)
        ENDDO
        WRITE(*,*)'Events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(4I12)')((NUMEVTS(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
C       Reset weights and number of events
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              NUMEVTS(K,I,J)=0
              SUMWGT(K,I,J)=0D0
            ENDDO
          ENDDO
        ENDDO
        RETURN
      ELSE IF(IMODE.EQ.3)THEN
C       No finalize needed
        RETURN
      ENDIF

C     IMODE.EQ.0, regular run mode

C     Select among the subprocesses based on PDF weight
      SUMPROB=0D0
      DO J=1,SYMCONF(0)
        DO I=1,MAXSPROC
          IF(CONFSUB(I,SYMCONF(J)).NE.0) THEN
            DO K=1,2
              IF(K.EQ.1.OR.MIRRORPROCS(I))THEN
C               Calculate PDF weight for all subprocesses
                SELPROC(K,I,J)=DSIGPROC(PP,J,I,K,SYMCONF,CONFSUB,1D0,4)
                SUMPROB=SUMPROB+SELPROC(K,I,J)
                IF(K.EQ.2)THEN
C                 Need to flip back x values
                  XDUM=XBK(1)
                  XBK(1)=XBK(2)
                  XBK(2)=XDUM
                ENDIF
              ENDIF
            ENDDO
          ENDIF
        ENDDO
      ENDDO

C     Perform the selection
      IDUM=0
      R=XRAN1(IDUM)*SUMPROB
      ICONF=0
      IPROC=0
      TOTWGT=0D0
      DO J=1,SYMCONF(0)
        DO I=1,MAXSPROC
          DO K=1,2
            TOTWGT=TOTWGT+SELPROC(K,I,J)
            IF(R.LT.TOTWGT)THEN
C             Normalize SELPROC to selection probability
              SELPROC(K,I,J)=SELPROC(K,I,J)/SUMPROB
              IPROC=I
              ICONF=J
              IMIRROR=K
              GOTO 50
            ENDIF
          ENDDO
        ENDDO
      ENDDO
 50   CONTINUE

      IF(IPROC.EQ.0) RETURN

C     Update weigth w.r.t SELPROC
      WGT=WGT/SELPROC(IMIRROR,IPROC,ICONF)

C     Call DSIGPROC to calculate sigma for process
      DSIG=DSIGPROC(PP,ICONF,IPROC,IMIRROR,SYMCONF,CONFSUB,WGT,IMODE)

      IF(DSIG.GT.0D0)THEN
C       Update summed weight and number of events
        SUMWGT(IMIRROR,IPROC,ICONF)=SUMWGT(IMIRROR,IPROC,ICONF)
     $   +DSIG*WGT
        NUMEVTS(IMIRROR,IPROC,ICONF)=NUMEVTS(IMIRROR,IPROC,ICONF)+1
      ENDIF

      RETURN
      END

      FUNCTION DSIGPROC(PP,ICONF,IPROC,IMIRROR,SYMCONF,CONFSUB,WGT
     $ ,IMODE)
C     ****************************************************
C     RETURNS DIFFERENTIAL CROSS SECTION 
C     FOR A PROCESS
C     Input:
C     pp    4 momentum of external particles
C     wgt   weight from Monte Carlo
C     imode 0 run, 1 init, 2 reweight, 3 finalize
C     Output:
C     Amplitude squared and summed
C     ****************************************************

      IMPLICIT NONE

      INCLUDE 'genps.inc'
      INCLUDE 'maxconfigs.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'run.inc'
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION DSIGPROC
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER ICONF,IPROC,IMIRROR,IMODE
      INTEGER SYMCONF(0:LMAXCONFIGS)
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
C     
C     GLOBAL VARIABLES
C     
C     SUBDIAG is vector of diagram numbers for this config
C     IB gives which beam is which (for mirror processes)
      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
C     ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     
C     EXTERNAL FUNCTIONS
C     
      DOUBLE PRECISION DSIG1,DSIG2
      LOGICAL PASSCUTS
C     
C     LOCAL VARIABLES 
C     
      DOUBLE PRECISION P1(0:3,NEXTERNAL),XDUM
      INTEGER I,J,K,JC(NEXTERNAL)
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'

      ICONFIG=SYMCONF(ICONF)
      DO I=1,MAXSPROC
        SUBDIAG(I) = CONFSUB(I,SYMCONF(ICONF))
      ENDDO

C     Set momenta according to this permutation
      CALL SWITCHMOM(PP,P1,PERMS(1,MAPCONFIG(ICONFIG)),JC,NEXTERNAL)

      IB(1)=1
      IB(2)=2

      IF(IMIRROR.EQ.2)THEN
C       Flip momenta (rotate around x axis)
        DO I=1,NEXTERNAL
          P1(2,I)=-P1(2,I)
          P1(3,I)=-P1(3,I)
        ENDDO
C       Flip beam identity
        IB(1)=2
        IB(2)=1
C       Flip x values (to get boost right)
        XDUM=XBK(1)
        XBK(1)=XBK(2)
        XBK(2)=XDUM
      ENDIF

      DSIGPROC=0D0

      IF (PASSCUTS(P1)) THEN
        IF(IPROC.EQ.1) DSIGPROC=DSIG1(P1,WGT,IMODE)  ! u u~ > u u~
        IF(IPROC.EQ.2) DSIGPROC=DSIG2(P1,WGT,IMODE)  ! u u~ > d d~
      ENDIF
      RETURN
      END


""" % misc.get_pkg_info()
        
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal_super)

    def test_export_group_decay_chains(self):
        """Test the result of exporting a subprocess group decay chain"""

        # Setup a model

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

        # A Z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
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
        z = mypartlist[-1]

        # Gluon and photon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             g]),
                      'color': [color.ColorString([color.T(2,1,0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # 3 gluon vertiex
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [g] * 3),
                      'color': [color.ColorString([color.f(0,1,2)])],
                      'lorentz':['VVV1'],
                      'couplings':{(0, 0):'G'},
                      'orders':{'QCD':1}}))

        # Coupling of Z to quarks
        
        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GUZ1', (0, 1):'GUZ2'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFV1', 'FFV2'],
                      'couplings':{(0, 0):'GDZ1', (0, 0):'GDZ2'},
                      'orders':{'QED':1}}))

        # Z-Z-q-q~ 4-point vertex

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antid, \
                                             d, \
                                             z,
                                             z]),
                      'color': [color.ColorString([color.T(1,0)])],
                      'lorentz':['FFVV1', 'FFVV2'],
                      'couplings':{(0, 0):'ZZQQ',(0, 1):'ZZQQ'},
                      'orders':{'QED':2}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)        
        mymodel.set('name', 'sm')

        # Set parameters
        external_parameters = [\
            base_objects.ParamCardVariable('zero', 0.,'DUM', 1),
            base_objects.ParamCardVariable('MZ', 91.,'MASS', 23),
            base_objects.ParamCardVariable('WZ', 2.,'DECAY', 23)]
        couplings = [\
            base_objects.ModelVariable('GQQ', '1.', 'complex'),
            base_objects.ModelVariable('GQED', '0.1', 'complex'),
            base_objects.ModelVariable('G', '1.', 'complex'),
            base_objects.ModelVariable('GUZ1', '0.1', 'complex'),
            base_objects.ModelVariable('GUZ2', '0.1', 'complex'),
            base_objects.ModelVariable('GDZ1', '0.05', 'complex'),
            base_objects.ModelVariable('GDZ2', '0.05', 'complex'),
            base_objects.ModelVariable('ZZQQ', '0.01', 'complex')]
        mymodel.set('parameters', {('external',): external_parameters})
        mymodel.set('couplings', {(): couplings})
        mymodel.set('functions', [])
                    
        procs = [[2,-2,23,23], [1,-1,23,23]]
        decays = [[23,1,-1,21], [23,2,-2]]
        coreamplitudes = diagram_generation.AmplitudeList()
        decayamplitudes = diagram_generation.AmplitudeList()
        decayprocs = base_objects.ProcessList()

        for proc in procs:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel})
            my_amplitude = diagram_generation.Amplitude(my_process)
            my_amplitude.set('has_mirror_process', True)
            coreamplitudes.append(my_amplitude)

        for proc in decays:
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel,
                                               'is_decay_chain': True})
            my_amplitude = diagram_generation.Amplitude(my_process)
            decayamplitudes.append(my_amplitude)
            decayprocs.append(my_process)

        decays = diagram_generation.DecayChainAmplitudeList([\
                         diagram_generation.DecayChainAmplitude({\
                                            'amplitudes': decayamplitudes})])

        decay_chains = diagram_generation.DecayChainAmplitude({\
            'amplitudes': coreamplitudes,
            'decay_chains': decays})

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
                          group_amplitudes(decay_chains)

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        # Check number of groups
        self.assertEqual(len(subproc_groups), 3)
        self.assertEqual([g.get('name') for g in subproc_groups],
                         ['qq_zz_z_qqg_z_qqg',
                          'qq_zz_z_qqg_z_qq',
                          'qq_zz_z_qq_z_qq'])

        subprocess_group = subproc_groups[0]
        matrix_elements = subprocess_group.get('matrix_elements')

        # Exporter
        exporter = export_v4.ProcessExporterFortranMEGroup()

        # Test amp2 lines
        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_elements[1],
                                          subprocess_group.get('diagram_maps')[1])
        #print '\n'.join(amp2_lines)

        self.assertEqual('\n'.join(amp2_lines),
"""AMP2(5)=AMP2(5)+AMP(9)*dconjg(AMP(9))
AMP2(6)=AMP2(6)+AMP(10)*dconjg(AMP(10))
AMP2(7)=AMP2(7)+AMP(11)*dconjg(AMP(11))
AMP2(8)=AMP2(8)+AMP(12)*dconjg(AMP(12))
AMP2(9)=AMP2(9)+AMP(13)*dconjg(AMP(13))
AMP2(10)=AMP2(10)+AMP(14)*dconjg(AMP(14))
AMP2(11)=AMP2(11)+AMP(15)*dconjg(AMP(15))
AMP2(12)=AMP2(12)+AMP(16)*dconjg(AMP(16))""")
        
        # Test configs.inc

        exporter.write_configs_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group,
            subprocess_group.get('diagrams_for_configs'))

        #print open(self.give_pos('test')).read()
        
        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/8,6/
      DATA SPROP(-1,1)/1/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/7,-1/
      DATA SPROP(-2,1)/23/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/5,3/
      DATA SPROP(-3,1)/1/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/4,-3/
      DATA SPROP(-4,1)/23/
      DATA TPRID(-4,1)/0/
      DATA (IFOREST(I,-5,1),I=1,2)/1,-4/
      DATA TPRID(-5,1)/2/
      DATA SPROP(-5,1)/0/
      DATA (IFOREST(I,-6,1),I=1,2)/-5,-2/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/8,7/
      DATA SPROP(-1,2)/-1/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,6/
      DATA SPROP(-2,2)/23/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/5,3/
      DATA SPROP(-3,2)/1/
      DATA TPRID(-3,2)/0/
      DATA (IFOREST(I,-4,2),I=1,2)/4,-3/
      DATA SPROP(-4,2)/23/
      DATA TPRID(-4,2)/0/
      DATA (IFOREST(I,-5,2),I=1,2)/1,-4/
      DATA TPRID(-5,2)/2/
      DATA SPROP(-5,2)/0/
      DATA (IFOREST(I,-6,2),I=1,2)/-5,-2/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/8,6/
      DATA SPROP(-1,3)/1/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/7,-1/
      DATA SPROP(-2,3)/23/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/5,4/
      DATA SPROP(-3,3)/-1/
      DATA TPRID(-3,3)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/-3,3/
      DATA SPROP(-4,3)/23/
      DATA TPRID(-4,3)/0/
      DATA (IFOREST(I,-5,3),I=1,2)/1,-4/
      DATA TPRID(-5,3)/2/
      DATA SPROP(-5,3)/0/
      DATA (IFOREST(I,-6,3),I=1,2)/-5,-2/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/8,7/
      DATA SPROP(-1,4)/-1/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/-1,6/
      DATA SPROP(-2,4)/23/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/5,4/
      DATA SPROP(-3,4)/-1/
      DATA TPRID(-3,4)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,3/
      DATA SPROP(-4,4)/23/
      DATA TPRID(-4,4)/0/
      DATA (IFOREST(I,-5,4),I=1,2)/1,-4/
      DATA TPRID(-5,4)/2/
      DATA SPROP(-5,4)/0/
      DATA (IFOREST(I,-6,4),I=1,2)/-5,-2/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA (IFOREST(I,-1,5),I=1,2)/5,3/
      DATA SPROP(-1,5)/1/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/4,-1/
      DATA SPROP(-2,5)/23/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/8,6/
      DATA SPROP(-3,5)/1/
      DATA TPRID(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/7,-3/
      DATA SPROP(-4,5)/23/
      DATA TPRID(-4,5)/0/
      DATA (IFOREST(I,-5,5),I=1,2)/1,-4/
      DATA TPRID(-5,5)/2/
      DATA SPROP(-5,5)/0/
      DATA (IFOREST(I,-6,5),I=1,2)/-5,-2/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA (IFOREST(I,-1,6),I=1,2)/5,3/
      DATA SPROP(-1,6)/1/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/4,-1/
      DATA SPROP(-2,6)/23/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/8,7/
      DATA SPROP(-3,6)/-1/
      DATA TPRID(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/-3,6/
      DATA SPROP(-4,6)/23/
      DATA TPRID(-4,6)/0/
      DATA (IFOREST(I,-5,6),I=1,2)/1,-4/
      DATA TPRID(-5,6)/2/
      DATA SPROP(-5,6)/0/
      DATA (IFOREST(I,-6,6),I=1,2)/-5,-2/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA (IFOREST(I,-1,7),I=1,2)/5,4/
      DATA SPROP(-1,7)/-1/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/-1,3/
      DATA SPROP(-2,7)/23/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/8,6/
      DATA SPROP(-3,7)/1/
      DATA TPRID(-3,7)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/7,-3/
      DATA SPROP(-4,7)/23/
      DATA TPRID(-4,7)/0/
      DATA (IFOREST(I,-5,7),I=1,2)/1,-4/
      DATA TPRID(-5,7)/2/
      DATA SPROP(-5,7)/0/
      DATA (IFOREST(I,-6,7),I=1,2)/-5,-2/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA (IFOREST(I,-1,8),I=1,2)/5,4/
      DATA SPROP(-1,8)/-1/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,3/
      DATA SPROP(-2,8)/23/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/8,7/
      DATA SPROP(-3,8)/-1/
      DATA TPRID(-3,8)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,6/
      DATA SPROP(-4,8)/23/
      DATA TPRID(-4,8)/0/
      DATA (IFOREST(I,-5,8),I=1,2)/1,-4/
      DATA TPRID(-5,8)/2/
      DATA SPROP(-5,8)/0/
      DATA (IFOREST(I,-6,8),I=1,2)/-5,-2/
C     Number of configs
      DATA MAPCONFIG(0)/8/
""")

        # Test config_subproc_map.inc

        exporter.write_config_subproc_map_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group.get('diagrams_for_configs'))

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""      DATA (CONFSUB(I,1),I=1,2)/1,5/
      DATA (CONFSUB(I,2),I=1,2)/2,6/
      DATA (CONFSUB(I,3),I=1,2)/3,7/
      DATA (CONFSUB(I,4),I=1,2)/4,8/
      DATA (CONFSUB(I,5),I=1,2)/5,9/
      DATA (CONFSUB(I,6),I=1,2)/6,10/
      DATA (CONFSUB(I,7),I=1,2)/7,11/
      DATA (CONFSUB(I,8),I=1,2)/8,12/
""")

        # Test processes.dat

        files.write_to_file(self.give_pos('test'),
                            exporter.write_processes_file,
                            subprocess_group)

        #print open(self.give_pos('test')).read()

        goal_processes = """1       u u~ > d d~ g d d~ g
mirror  u~ u > d d~ g d d~ g
2       d d~ > d d~ g d d~ g
mirror  d~ d > d d~ g d d~ g"""
        
        self.assertFileContains('test', goal_processes)

    def test_export_group_multidiagram_decay_chains(self):
        """Test export group_amplitudes for uu~>g>gogo, go>qqn1."""

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

        # A gluino
        mypartlist.append(base_objects.Particle({'name':'go',
                      'antiname':'go',
                      'spin':2,
                      'color':8,
                      'mass':'MGO',
                      'width':'WGO',
                      'texname':'g',
                      'antitexname':'g',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':1000021,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        go = mypartlist[-1]

        # A u squark
        mypartlist.append(base_objects.Particle({'name':'ul',
                      'antiname':'ul~',
                      'spin':1,
                      'color':3,
                      'mass':'MUL',
                      'width':'WUL',
                      'texname':'u',
                      'antitexname':'\bar u',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':1000002,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        ul = mypartlist[-1]
        antiul = copy.copy(ul)
        antiul.set('is_part', False)

        # A quark D and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'dl',
                      'antiname':'dl~',
                      'spin':1,
                      'color':3,
                      'mass':'MDL',
                      'width':'WDL',
                      'texname':'d',
                      'antitexname':'\bar d',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':1000001,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        dl = mypartlist[-1]
        antidl = copy.copy(dl)
        antidl.set('is_part', False)

        # A neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'MN1',
                      'width':'zero',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[-1]

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        g = mymodel.get_particle(21)
        d = mymodel.get_particle(1)
        antid = mymodel.get_particle(-1)
        u = mymodel.get_particle(2)
        antiu = mymodel.get_particle(-2)
        
        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Gluon couplings to gluino
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             go, \
                                             g]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Gluino couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 11,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             u, \
                                             antiul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 12,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             antiu, \
                                             ul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 13,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             d, \
                                             antidl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 14,
                      'particles': base_objects.ParticleList(\
                                            [go, \
                                             antid, \
                                             dl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QCD':1}}))

        # Neutralino couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 15,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             u, \
                                             antiul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 16,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             antiu, \
                                             ul]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 17,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             d, \
                                             antidl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 18,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             antid, \
                                             dl]),
                      'color': [],
                      'lorentz':['L1'],
                      'couplings':{(0, 0):'GQQ'},
                      'orders':{'QED':1}}))

        mymodel.set('interactions', myinterlist)

        procs = [[2,-2,1000021,1000021]]
        decays = [[1000021,1,-1,1000022],[1000021,2,-2,1000022]]
        coreamplitudes = diagram_generation.AmplitudeList()
        decayamplitudes = diagram_generation.AmplitudeList()
        decayprocs = base_objects.ProcessList()
        proc_diags = [1]
        decay_diags = [2,2]
        
        for iproc, proc in enumerate(procs):
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)
            my_leglist[1].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel,
                                               'required_s_channels':[[21]]})
            my_amplitude = diagram_generation.Amplitude(my_process)
            coreamplitudes.append(my_amplitude)
            self.assertEqual(len(my_amplitude.get('diagrams')), proc_diags[iproc])

        for iproc, proc in enumerate(decays):
            # Define the multiprocess
            my_leglist = base_objects.LegList([\
                base_objects.Leg({'id': id, 'state': True}) for id in proc])

            my_leglist[0].set('state', False)

            my_process = base_objects.Process({'legs':my_leglist,
                                               'model':mymodel,
                                               'is_decay_chain': True})
                                               #'forbidden_particles':[2000001,2000002]})
            my_amplitude = diagram_generation.Amplitude(my_process)
            decayamplitudes.append(my_amplitude)
            decayprocs.append(my_process)
            self.assertEqual(len(my_amplitude.get('diagrams')), decay_diags[iproc])

        decays = diagram_generation.DecayChainAmplitudeList([\
                         diagram_generation.DecayChainAmplitude({\
                                            'amplitudes': decayamplitudes})])

        decay_chains = diagram_generation.DecayChainAmplitude({\
            'amplitudes': coreamplitudes,
            'decay_chains': decays})

        dc_subproc_group = group_subprocs.DecayChainSubProcessGroup.\
                          group_amplitudes(decay_chains)

        subproc_groups = \
                       dc_subproc_group.generate_helas_decay_chain_subproc_groups()

        self.assertEqual(len(subproc_groups), 1)

        group_name = 'qq_gogo_go_qqn1_go_qqn1'
        me_len = 3

        subprocess_group = subproc_groups[0]
        self.assertEqual(subprocess_group.get('name'),
                         group_name)
        self.assertEqual(len(subprocess_group.get('matrix_elements')), me_len)

        # Exporter
        exporter = export_v4.ProcessExporterFortranMEGroup()

        # Test config_subproc_map.inc
        exporter.write_config_subproc_map_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group.get('diagrams_for_configs'))

        self.assertFileContains('test',
"""      DATA (CONFSUB(I,1),I=1,3)/1,0,0/
      DATA (CONFSUB(I,2),I=1,3)/2,0,0/
      DATA (CONFSUB(I,3),I=1,3)/0,1,0/
      DATA (CONFSUB(I,4),I=1,3)/0,2,0/
      DATA (CONFSUB(I,5),I=1,3)/3,0,0/
      DATA (CONFSUB(I,6),I=1,3)/4,0,0/
      DATA (CONFSUB(I,7),I=1,3)/0,3,0/
      DATA (CONFSUB(I,8),I=1,3)/0,4,0/
      DATA (CONFSUB(I,9),I=1,3)/0,0,1/
      DATA (CONFSUB(I,10),I=1,3)/0,0,2/
      DATA (CONFSUB(I,11),I=1,3)/0,0,3/
      DATA (CONFSUB(I,12),I=1,3)/0,0,4/
""")

        # Test configs.inc

        exporter.write_configs_file(\
            writers.FortranWriter(self.give_pos('test')),
            subprocess_group,
            subprocess_group.get('diagrams_for_configs'))

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/5,3/
      DATA SPROP(-1,1)/1000001/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/4,-1/
      DATA SPROP(-2,1)/1000021/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/8,6/
      DATA SPROP(-3,1)/1000001/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/7,-3/
      DATA SPROP(-4,1)/1000021/
      DATA TPRID(-4,1)/0/
      DATA (IFOREST(I,-5,1),I=1,2)/-4,-2/
      DATA SPROP(-5,1)/21/
      DATA TPRID(-5,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/5,3/
      DATA SPROP(-1,2)/1000001/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/4,-1/
      DATA SPROP(-2,2)/1000021/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/8,7/
      DATA SPROP(-3,2)/-1000001/
      DATA TPRID(-3,2)/0/
      DATA (IFOREST(I,-4,2),I=1,2)/-3,6/
      DATA SPROP(-4,2)/1000021/
      DATA TPRID(-4,2)/0/
      DATA (IFOREST(I,-5,2),I=1,2)/-4,-2/
      DATA SPROP(-5,2)/21/
      DATA TPRID(-5,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/5,3/
      DATA SPROP(-1,3)/1000001/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/4,-1/
      DATA SPROP(-2,3)/1000021/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/8,6/
      DATA SPROP(-3,3)/1000002/
      DATA TPRID(-3,3)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/7,-3/
      DATA SPROP(-4,3)/1000021/
      DATA TPRID(-4,3)/0/
      DATA (IFOREST(I,-5,3),I=1,2)/-4,-2/
      DATA SPROP(-5,3)/21/
      DATA TPRID(-5,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/5,3/
      DATA SPROP(-1,4)/1000001/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/4,-1/
      DATA SPROP(-2,4)/1000021/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/8,7/
      DATA SPROP(-3,4)/-1000002/
      DATA TPRID(-3,4)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,6/
      DATA SPROP(-4,4)/1000021/
      DATA TPRID(-4,4)/0/
      DATA (IFOREST(I,-5,4),I=1,2)/-4,-2/
      DATA SPROP(-5,4)/21/
      DATA TPRID(-5,4)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA (IFOREST(I,-1,5),I=1,2)/5,4/
      DATA SPROP(-1,5)/-1000001/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/-1,3/
      DATA SPROP(-2,5)/1000021/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/8,6/
      DATA SPROP(-3,5)/1000001/
      DATA TPRID(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/7,-3/
      DATA SPROP(-4,5)/1000021/
      DATA TPRID(-4,5)/0/
      DATA (IFOREST(I,-5,5),I=1,2)/-4,-2/
      DATA SPROP(-5,5)/21/
      DATA TPRID(-5,5)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA (IFOREST(I,-1,6),I=1,2)/5,4/
      DATA SPROP(-1,6)/-1000001/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/-1,3/
      DATA SPROP(-2,6)/1000021/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/8,7/
      DATA SPROP(-3,6)/-1000001/
      DATA TPRID(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/-3,6/
      DATA SPROP(-4,6)/1000021/
      DATA TPRID(-4,6)/0/
      DATA (IFOREST(I,-5,6),I=1,2)/-4,-2/
      DATA SPROP(-5,6)/21/
      DATA TPRID(-5,6)/0/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA (IFOREST(I,-1,7),I=1,2)/5,4/
      DATA SPROP(-1,7)/-1000001/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/-1,3/
      DATA SPROP(-2,7)/1000021/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/8,6/
      DATA SPROP(-3,7)/1000002/
      DATA TPRID(-3,7)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/7,-3/
      DATA SPROP(-4,7)/1000021/
      DATA TPRID(-4,7)/0/
      DATA (IFOREST(I,-5,7),I=1,2)/-4,-2/
      DATA SPROP(-5,7)/21/
      DATA TPRID(-5,7)/0/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA (IFOREST(I,-1,8),I=1,2)/5,4/
      DATA SPROP(-1,8)/-1000001/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,3/
      DATA SPROP(-2,8)/1000021/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/8,7/
      DATA SPROP(-3,8)/-1000002/
      DATA TPRID(-3,8)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,6/
      DATA SPROP(-4,8)/1000021/
      DATA TPRID(-4,8)/0/
      DATA (IFOREST(I,-5,8),I=1,2)/-4,-2/
      DATA SPROP(-5,8)/21/
      DATA TPRID(-5,8)/0/
C     Diagram 11
      DATA MAPCONFIG(9)/11/
      DATA (IFOREST(I,-1,9),I=1,2)/5,3/
      DATA SPROP(-1,9)/1000002/
      DATA TPRID(-1,9)/0/
      DATA (IFOREST(I,-2,9),I=1,2)/4,-1/
      DATA SPROP(-2,9)/1000021/
      DATA TPRID(-2,9)/0/
      DATA (IFOREST(I,-3,9),I=1,2)/8,6/
      DATA SPROP(-3,9)/1000002/
      DATA TPRID(-3,9)/0/
      DATA (IFOREST(I,-4,9),I=1,2)/7,-3/
      DATA SPROP(-4,9)/1000021/
      DATA TPRID(-4,9)/0/
      DATA (IFOREST(I,-5,9),I=1,2)/-4,-2/
      DATA SPROP(-5,9)/21/
      DATA TPRID(-5,9)/0/
C     Diagram 12
      DATA MAPCONFIG(10)/12/
      DATA (IFOREST(I,-1,10),I=1,2)/5,3/
      DATA SPROP(-1,10)/1000002/
      DATA TPRID(-1,10)/0/
      DATA (IFOREST(I,-2,10),I=1,2)/4,-1/
      DATA SPROP(-2,10)/1000021/
      DATA TPRID(-2,10)/0/
      DATA (IFOREST(I,-3,10),I=1,2)/8,7/
      DATA SPROP(-3,10)/-1000002/
      DATA TPRID(-3,10)/0/
      DATA (IFOREST(I,-4,10),I=1,2)/-3,6/
      DATA SPROP(-4,10)/1000021/
      DATA TPRID(-4,10)/0/
      DATA (IFOREST(I,-5,10),I=1,2)/-4,-2/
      DATA SPROP(-5,10)/21/
      DATA TPRID(-5,10)/0/
C     Diagram 15
      DATA MAPCONFIG(11)/15/
      DATA (IFOREST(I,-1,11),I=1,2)/5,4/
      DATA SPROP(-1,11)/-1000002/
      DATA TPRID(-1,11)/0/
      DATA (IFOREST(I,-2,11),I=1,2)/-1,3/
      DATA SPROP(-2,11)/1000021/
      DATA TPRID(-2,11)/0/
      DATA (IFOREST(I,-3,11),I=1,2)/8,6/
      DATA SPROP(-3,11)/1000002/
      DATA TPRID(-3,11)/0/
      DATA (IFOREST(I,-4,11),I=1,2)/7,-3/
      DATA SPROP(-4,11)/1000021/
      DATA TPRID(-4,11)/0/
      DATA (IFOREST(I,-5,11),I=1,2)/-4,-2/
      DATA SPROP(-5,11)/21/
      DATA TPRID(-5,11)/0/
C     Diagram 16
      DATA MAPCONFIG(12)/16/
      DATA (IFOREST(I,-1,12),I=1,2)/5,4/
      DATA SPROP(-1,12)/-1000002/
      DATA TPRID(-1,12)/0/
      DATA (IFOREST(I,-2,12),I=1,2)/-1,3/
      DATA SPROP(-2,12)/1000021/
      DATA TPRID(-2,12)/0/
      DATA (IFOREST(I,-3,12),I=1,2)/8,7/
      DATA SPROP(-3,12)/-1000002/
      DATA TPRID(-3,12)/0/
      DATA (IFOREST(I,-4,12),I=1,2)/-3,6/
      DATA SPROP(-4,12)/1000021/
      DATA TPRID(-4,12)/0/
      DATA (IFOREST(I,-5,12),I=1,2)/-4,-2/
      DATA SPROP(-5,12)/21/
      DATA TPRID(-5,12)/0/
C     Number of configs
      DATA MAPCONFIG(0)/12/
""")

#===============================================================================
# FullHelasOutputTest
#===============================================================================
class FullHelasOutputTest(test_helas_call_writers.HelasModelTestSetup,
                          test_file_writers.CheckFileCreate):
    """Test class for the output of various processes. In practice,
    tests both HelasObject generation and MG4 output."""

    created_files = ['leshouche'
                    ]

    tearDown = test_file_writers.CheckFileCreate.clean_files

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
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(\
            helas_call_writers.FortranHelasCallWriter(self.mymodel).\
            get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),me,NHEL(1),+1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),+1*IC(4),W(1,4))
CALL FVIXXX(W(1,1),W(1,2),MGVX12,me,zero,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,5),W(1,4),W(1,3),MGVX12,AMP(1))
CALL FVIXXX(W(1,1),W(1,3),MGVX12,me,zero,W(1,6))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,6),W(1,4),W(1,2),MGVX12,AMP(2))""")

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
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(\
            myamplitude,
            0)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(\
            helas_call_writers.FortranHelasCallWriter(self.mymodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),mu,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),mu,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL FVIXXX(W(1,1),W(1,3),GG,mu,zero,W(1,6))
CALL JIOXXX(W(1,5),W(1,4),MGVX12,zero,zero,W(1,7))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,6),W(1,2),W(1,7),MGVX15,AMP(1))
CALL IXXXXX(P(0,1),mu,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),mu,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL FVOXXX(W(1,2),W(1,3),GG,mu,zero,W(1,6))
CALL JIOXXX(W(1,5),W(1,4),MGVX12,zero,zero,W(1,7))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,1),W(1,6),W(1,7),MGVX15,AMP(2))""")

    def test_generate_helas_diagrams_uux_uuxuux(self):
        """Test calls for u u~ > u u~ u u~ and MadEvent files"""

        # Set up local model

        mybasemodel = base_objects.Model()
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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

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

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # Gluon self-couplings
        my_color_string = color.ColorString([color.f(0, 1, 2)])
        my_color_string.is_imaginary = True
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [my_color_string],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        # Test Helas calls

        fortran_model = helas_call_writers.FortranHelasCallWriter(mybasemodel)

        self.assertEqual("\n".join(fortran_model.\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),zero,NHEL(4),-1*IC(4),W(1,4))
CALL OXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL IXXXXX(P(0,6),zero,NHEL(6),-1*IC(6),W(1,6))
CALL JIOXXX(W(1,1),W(1,2),GG,zero,zero,W(1,7))
CALL JIOXXX(W(1,4),W(1,3),GG,zero,zero,W(1,8))
CALL FVOXXX(W(1,5),W(1,7),GG,zero,zero,W(1,9))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,6),W(1,9),W(1,8),GG,AMP(1))
CALL FVIXXX(W(1,6),W(1,7),GG,zero,zero,W(1,10))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,10),W(1,5),W(1,8),GG,AMP(2))
CALL JIOXXX(W(1,6),W(1,5),GG,zero,zero,W(1,11))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,7),W(1,8),W(1,11),GG,AMP(3))
CALL JIOXXX(W(1,6),W(1,3),GG,zero,zero,W(1,12))
CALL FVIXXX(W(1,4),W(1,7),GG,zero,zero,W(1,13))
# Amplitude(s) for diagram number 4
CALL IOVXXX(W(1,13),W(1,5),W(1,12),GG,AMP(4))
# Amplitude(s) for diagram number 5
CALL IOVXXX(W(1,4),W(1,9),W(1,12),GG,AMP(5))
CALL JIOXXX(W(1,4),W(1,5),GG,zero,zero,W(1,14))
# Amplitude(s) for diagram number 6
CALL VVVXXX(W(1,7),W(1,12),W(1,14),GG,AMP(6))
CALL FVOXXX(W(1,3),W(1,7),GG,zero,zero,W(1,15))
# Amplitude(s) for diagram number 7
CALL IOVXXX(W(1,6),W(1,15),W(1,14),GG,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOVXXX(W(1,10),W(1,3),W(1,14),GG,AMP(8))
# Amplitude(s) for diagram number 9
CALL IOVXXX(W(1,4),W(1,15),W(1,11),GG,AMP(9))
# Amplitude(s) for diagram number 10
CALL IOVXXX(W(1,13),W(1,3),W(1,11),GG,AMP(10))
CALL JIOXXX(W(1,1),W(1,3),GG,zero,zero,W(1,16))
CALL JIOXXX(W(1,4),W(1,2),GG,zero,zero,W(1,17))
CALL FVOXXX(W(1,5),W(1,16),GG,zero,zero,W(1,18))
# Amplitude(s) for diagram number 11
CALL IOVXXX(W(1,6),W(1,18),W(1,17),GG,AMP(11))
CALL FVIXXX(W(1,6),W(1,16),GG,zero,zero,W(1,19))
# Amplitude(s) for diagram number 12
CALL IOVXXX(W(1,19),W(1,5),W(1,17),GG,AMP(12))
# Amplitude(s) for diagram number 13
CALL VVVXXX(W(1,16),W(1,17),W(1,11),GG,AMP(13))
CALL JIOXXX(W(1,6),W(1,2),GG,zero,zero,W(1,20))
CALL FVIXXX(W(1,4),W(1,16),GG,zero,zero,W(1,21))
# Amplitude(s) for diagram number 14
CALL IOVXXX(W(1,21),W(1,5),W(1,20),GG,AMP(14))
# Amplitude(s) for diagram number 15
CALL IOVXXX(W(1,4),W(1,18),W(1,20),GG,AMP(15))
# Amplitude(s) for diagram number 16
CALL VVVXXX(W(1,16),W(1,20),W(1,14),GG,AMP(16))
CALL FVOXXX(W(1,2),W(1,16),GG,zero,zero,W(1,22))
# Amplitude(s) for diagram number 17
CALL IOVXXX(W(1,6),W(1,22),W(1,14),GG,AMP(17))
# Amplitude(s) for diagram number 18
CALL IOVXXX(W(1,19),W(1,2),W(1,14),GG,AMP(18))
# Amplitude(s) for diagram number 19
CALL IOVXXX(W(1,4),W(1,22),W(1,11),GG,AMP(19))
# Amplitude(s) for diagram number 20
CALL IOVXXX(W(1,21),W(1,2),W(1,11),GG,AMP(20))
CALL JIOXXX(W(1,1),W(1,5),GG,zero,zero,W(1,23))
CALL FVOXXX(W(1,3),W(1,23),GG,zero,zero,W(1,24))
# Amplitude(s) for diagram number 21
CALL IOVXXX(W(1,6),W(1,24),W(1,17),GG,AMP(21))
CALL FVIXXX(W(1,6),W(1,23),GG,zero,zero,W(1,25))
# Amplitude(s) for diagram number 22
CALL IOVXXX(W(1,25),W(1,3),W(1,17),GG,AMP(22))
# Amplitude(s) for diagram number 23
CALL VVVXXX(W(1,23),W(1,17),W(1,12),GG,AMP(23))
# Amplitude(s) for diagram number 24
CALL IOVXXX(W(1,4),W(1,24),W(1,20),GG,AMP(24))
CALL FVIXXX(W(1,4),W(1,23),GG,zero,zero,W(1,26))
# Amplitude(s) for diagram number 25
CALL IOVXXX(W(1,26),W(1,3),W(1,20),GG,AMP(25))
# Amplitude(s) for diagram number 26
CALL VVVXXX(W(1,23),W(1,20),W(1,8),GG,AMP(26))
CALL FVOXXX(W(1,2),W(1,23),GG,zero,zero,W(1,27))
# Amplitude(s) for diagram number 27
CALL IOVXXX(W(1,6),W(1,27),W(1,8),GG,AMP(27))
# Amplitude(s) for diagram number 28
CALL IOVXXX(W(1,25),W(1,2),W(1,8),GG,AMP(28))
# Amplitude(s) for diagram number 29
CALL IOVXXX(W(1,4),W(1,27),W(1,12),GG,AMP(29))
# Amplitude(s) for diagram number 30
CALL IOVXXX(W(1,26),W(1,2),W(1,12),GG,AMP(30))
CALL FVIXXX(W(1,1),W(1,17),GG,zero,zero,W(1,28))
# Amplitude(s) for diagram number 31
CALL IOVXXX(W(1,28),W(1,5),W(1,12),GG,AMP(31))
CALL FVIXXX(W(1,1),W(1,12),GG,zero,zero,W(1,29))
# Amplitude(s) for diagram number 32
CALL IOVXXX(W(1,29),W(1,5),W(1,17),GG,AMP(32))
# Amplitude(s) for diagram number 33
CALL IOVXXX(W(1,28),W(1,3),W(1,11),GG,AMP(33))
CALL FVIXXX(W(1,1),W(1,11),GG,zero,zero,W(1,30))
# Amplitude(s) for diagram number 34
CALL IOVXXX(W(1,30),W(1,3),W(1,17),GG,AMP(34))
CALL FVIXXX(W(1,1),W(1,20),GG,zero,zero,W(1,31))
# Amplitude(s) for diagram number 35
CALL IOVXXX(W(1,31),W(1,5),W(1,8),GG,AMP(35))
CALL FVIXXX(W(1,1),W(1,8),GG,zero,zero,W(1,32))
# Amplitude(s) for diagram number 36
CALL IOVXXX(W(1,32),W(1,5),W(1,20),GG,AMP(36))
# Amplitude(s) for diagram number 37
CALL IOVXXX(W(1,31),W(1,3),W(1,14),GG,AMP(37))
CALL FVIXXX(W(1,1),W(1,14),GG,zero,zero,W(1,33))
# Amplitude(s) for diagram number 38
CALL IOVXXX(W(1,33),W(1,3),W(1,20),GG,AMP(38))
# Amplitude(s) for diagram number 39
CALL IOVXXX(W(1,32),W(1,2),W(1,11),GG,AMP(39))
# Amplitude(s) for diagram number 40
CALL IOVXXX(W(1,30),W(1,2),W(1,8),GG,AMP(40))
# Amplitude(s) for diagram number 41
CALL IOVXXX(W(1,29),W(1,2),W(1,14),GG,AMP(41))
# Amplitude(s) for diagram number 42
CALL IOVXXX(W(1,33),W(1,2),W(1,12),GG,AMP(42))""")

        exporter = export_v4.ProcessExporterFortranME()

        #print matrix_element.get('color_basis')
        # Test color matrix output
        self.assertEqual("\n".join(exporter.get_color_data_lines(matrix_element)),
                         """DATA Denom(1)/1/
DATA (CF(i,  1),i=  1,  6) /   27,    9,    9,    3,    3,    9/
C 1 T(2,1) T(3,4) T(5,6)
DATA Denom(2)/1/
DATA (CF(i,  2),i=  1,  6) /    9,   27,    3,    9,    9,    3/
C 1 T(2,1) T(3,6) T(5,4)
DATA Denom(3)/1/
DATA (CF(i,  3),i=  1,  6) /    9,    3,   27,    9,    9,    3/
C 1 T(2,4) T(3,1) T(5,6)
DATA Denom(4)/1/
DATA (CF(i,  4),i=  1,  6) /    3,    9,    9,   27,    3,    9/
C 1 T(2,4) T(3,6) T(5,1)
DATA Denom(5)/1/
DATA (CF(i,  5),i=  1,  6) /    3,    9,    9,    3,   27,    9/
C 1 T(2,6) T(3,1) T(5,4)
DATA Denom(6)/1/
DATA (CF(i,  6),i=  1,  6) /    9,    3,    3,    9,    9,   27/
C 1 T(2,6) T(3,4) T(5,1)""")

        # Test JAMP (color amplitude) output
        self.assertEqual('\n'.join(exporter.get_JAMP_lines(matrix_element)),
                         """JAMP(1)=+1./4.*(+1./9.*AMP(1)+1./9.*AMP(2)+1./3.*AMP(4)+1./3.*AMP(5)+1./3.*AMP(7)+1./3.*AMP(8)+1./9.*AMP(9)+1./9.*AMP(10)+AMP(14)-AMP(16)+AMP(17)+1./3.*AMP(19)+1./3.*AMP(20)+AMP(22)-AMP(23)+1./3.*AMP(27)+1./3.*AMP(28)+AMP(29)+AMP(31)+1./3.*AMP(33)+1./3.*AMP(34)+1./3.*AMP(35)+1./3.*AMP(36)+AMP(37)+1./9.*AMP(39)+1./9.*AMP(40))
JAMP(2)=+1./4.*(-1./3.*AMP(1)-1./3.*AMP(2)-1./9.*AMP(4)-1./9.*AMP(5)-1./9.*AMP(7)-1./9.*AMP(8)-1./3.*AMP(9)-1./3.*AMP(10)-AMP(12)+AMP(13)-1./3.*AMP(17)-1./3.*AMP(18)-AMP(19)-AMP(25)+AMP(26)-AMP(27)-1./3.*AMP(29)-1./3.*AMP(30)-1./3.*AMP(31)-1./3.*AMP(32)-AMP(33)-AMP(35)-1./3.*AMP(37)-1./3.*AMP(38)-1./9.*AMP(41)-1./9.*AMP(42))
JAMP(3)=+1./4.*(-AMP(4)+AMP(6)-AMP(7)-1./3.*AMP(9)-1./3.*AMP(10)-1./9.*AMP(11)-1./9.*AMP(12)-1./3.*AMP(14)-1./3.*AMP(15)-1./3.*AMP(17)-1./3.*AMP(18)-1./9.*AMP(19)-1./9.*AMP(20)-1./3.*AMP(21)-1./3.*AMP(22)-AMP(24)-AMP(26)-AMP(28)-1./3.*AMP(31)-1./3.*AMP(32)-1./9.*AMP(33)-1./9.*AMP(34)-AMP(36)-1./3.*AMP(39)-1./3.*AMP(40)-AMP(41))
JAMP(4)=+1./4.*(+AMP(1)+AMP(3)+1./3.*AMP(4)+1./3.*AMP(5)+AMP(10)+1./3.*AMP(11)+1./3.*AMP(12)+AMP(15)+AMP(16)+AMP(18)+1./9.*AMP(21)+1./9.*AMP(22)+1./3.*AMP(24)+1./3.*AMP(25)+1./3.*AMP(27)+1./3.*AMP(28)+1./9.*AMP(29)+1./9.*AMP(30)+1./9.*AMP(31)+1./9.*AMP(32)+1./3.*AMP(33)+1./3.*AMP(34)+AMP(38)+AMP(40)+1./3.*AMP(41)+1./3.*AMP(42))
JAMP(5)=+1./4.*(+AMP(2)-AMP(3)+1./3.*AMP(7)+1./3.*AMP(8)+AMP(9)+1./3.*AMP(11)+1./3.*AMP(12)+1./9.*AMP(14)+1./9.*AMP(15)+1./9.*AMP(17)+1./9.*AMP(18)+1./3.*AMP(19)+1./3.*AMP(20)+AMP(21)+AMP(23)+1./3.*AMP(24)+1./3.*AMP(25)+AMP(30)+AMP(32)+1./3.*AMP(35)+1./3.*AMP(36)+1./9.*AMP(37)+1./9.*AMP(38)+AMP(39)+1./3.*AMP(41)+1./3.*AMP(42))
JAMP(6)=+1./4.*(-1./3.*AMP(1)-1./3.*AMP(2)-AMP(5)-AMP(6)-AMP(8)-AMP(11)-AMP(13)-1./3.*AMP(14)-1./3.*AMP(15)-AMP(20)-1./3.*AMP(21)-1./3.*AMP(22)-1./9.*AMP(24)-1./9.*AMP(25)-1./9.*AMP(27)-1./9.*AMP(28)-1./3.*AMP(29)-1./3.*AMP(30)-AMP(34)-1./9.*AMP(35)-1./9.*AMP(36)-1./3.*AMP(37)-1./3.*AMP(38)-1./3.*AMP(39)-1./3.*AMP(40)-AMP(42))""")

        # Test configs file
        writer = writers.FortranWriter(self.give_pos('test'))
        mapconfigs, s_and_t_channels = exporter.write_configs_file(writer,
                                                                 matrix_element)

        writer.close()

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA SPROP(-1,1)/21/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/6,-1/
      DATA SPROP(-2,1)/-2/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/5,-2/
      DATA SPROP(-3,1)/21/
      DATA TPRID(-3,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/4,3/
      DATA SPROP(-1,2)/21/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/5,-1/
      DATA SPROP(-2,2)/2/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/6,-2/
      DATA SPROP(-3,2)/21/
      DATA TPRID(-3,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/4,3/
      DATA SPROP(-1,3)/21/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/6,5/
      DATA SPROP(-2,3)/21/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/-2,-1/
      DATA SPROP(-3,3)/21/
      DATA TPRID(-3,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/6,3/
      DATA SPROP(-1,4)/21/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/5,-1/
      DATA SPROP(-2,4)/2/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/4,-2/
      DATA SPROP(-3,4)/21/
      DATA TPRID(-3,4)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA (IFOREST(I,-1,5),I=1,2)/6,3/
      DATA SPROP(-1,5)/21/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/4,-1/
      DATA SPROP(-2,5)/-2/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/5,-2/
      DATA SPROP(-3,5)/21/
      DATA TPRID(-3,5)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA (IFOREST(I,-1,6),I=1,2)/6,3/
      DATA SPROP(-1,6)/21/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/5,4/
      DATA SPROP(-2,6)/21/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/-2,-1/
      DATA SPROP(-3,6)/21/
      DATA TPRID(-3,6)/0/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA (IFOREST(I,-1,7),I=1,2)/5,4/
      DATA SPROP(-1,7)/21/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/6,-1/
      DATA SPROP(-2,7)/-2/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/-2,3/
      DATA SPROP(-3,7)/21/
      DATA TPRID(-3,7)/0/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA (IFOREST(I,-1,8),I=1,2)/5,4/
      DATA SPROP(-1,8)/21/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,3/
      DATA SPROP(-2,8)/2/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/6,-2/
      DATA SPROP(-3,8)/21/
      DATA TPRID(-3,8)/0/
C     Diagram 9
      DATA MAPCONFIG(9)/9/
      DATA (IFOREST(I,-1,9),I=1,2)/6,5/
      DATA SPROP(-1,9)/21/
      DATA TPRID(-1,9)/0/
      DATA (IFOREST(I,-2,9),I=1,2)/-1,4/
      DATA SPROP(-2,9)/-2/
      DATA TPRID(-2,9)/0/
      DATA (IFOREST(I,-3,9),I=1,2)/-2,3/
      DATA SPROP(-3,9)/21/
      DATA TPRID(-3,9)/0/
C     Diagram 10
      DATA MAPCONFIG(10)/10/
      DATA (IFOREST(I,-1,10),I=1,2)/6,5/
      DATA SPROP(-1,10)/21/
      DATA TPRID(-1,10)/0/
      DATA (IFOREST(I,-2,10),I=1,2)/-1,3/
      DATA SPROP(-2,10)/2/
      DATA TPRID(-2,10)/0/
      DATA (IFOREST(I,-3,10),I=1,2)/4,-2/
      DATA SPROP(-3,10)/21/
      DATA TPRID(-3,10)/0/
C     Diagram 11
      DATA MAPCONFIG(11)/11/
      DATA (IFOREST(I,-1,11),I=1,2)/1,3/
      DATA TPRID(-1,11)/21/
      DATA SPROP(-1,11)/0/
      DATA (IFOREST(I,-2,11),I=1,2)/-1,5/
      DATA TPRID(-2,11)/2/
      DATA SPROP(-2,11)/0/
      DATA (IFOREST(I,-3,11),I=1,2)/-2,6/
      DATA TPRID(-3,11)/21/
      DATA SPROP(-3,11)/0/
      DATA (IFOREST(I,-4,11),I=1,2)/-3,4/
C     Diagram 12
      DATA MAPCONFIG(12)/12/
      DATA (IFOREST(I,-1,12),I=1,2)/1,3/
      DATA TPRID(-1,12)/21/
      DATA SPROP(-1,12)/0/
      DATA (IFOREST(I,-2,12),I=1,2)/-1,6/
      DATA TPRID(-2,12)/2/
      DATA SPROP(-2,12)/0/
      DATA (IFOREST(I,-3,12),I=1,2)/-2,5/
      DATA TPRID(-3,12)/21/
      DATA SPROP(-3,12)/0/
      DATA (IFOREST(I,-4,12),I=1,2)/-3,4/
C     Diagram 13
      DATA MAPCONFIG(13)/13/
      DATA (IFOREST(I,-1,13),I=1,2)/6,5/
      DATA SPROP(-1,13)/21/
      DATA TPRID(-1,13)/0/
      DATA (IFOREST(I,-2,13),I=1,2)/1,3/
      DATA TPRID(-2,13)/21/
      DATA SPROP(-2,13)/0/
      DATA (IFOREST(I,-3,13),I=1,2)/-2,-1/
      DATA TPRID(-3,13)/21/
      DATA SPROP(-3,13)/0/
      DATA (IFOREST(I,-4,13),I=1,2)/-3,4/
C     Diagram 14
      DATA MAPCONFIG(14)/14/
      DATA (IFOREST(I,-1,14),I=1,2)/1,3/
      DATA TPRID(-1,14)/21/
      DATA SPROP(-1,14)/0/
      DATA (IFOREST(I,-2,14),I=1,2)/-1,4/
      DATA TPRID(-2,14)/2/
      DATA SPROP(-2,14)/0/
      DATA (IFOREST(I,-3,14),I=1,2)/-2,5/
      DATA TPRID(-3,14)/21/
      DATA SPROP(-3,14)/0/
      DATA (IFOREST(I,-4,14),I=1,2)/-3,6/
C     Diagram 15
      DATA MAPCONFIG(15)/15/
      DATA (IFOREST(I,-1,15),I=1,2)/1,3/
      DATA TPRID(-1,15)/21/
      DATA SPROP(-1,15)/0/
      DATA (IFOREST(I,-2,15),I=1,2)/-1,5/
      DATA TPRID(-2,15)/2/
      DATA SPROP(-2,15)/0/
      DATA (IFOREST(I,-3,15),I=1,2)/-2,4/
      DATA TPRID(-3,15)/21/
      DATA SPROP(-3,15)/0/
      DATA (IFOREST(I,-4,15),I=1,2)/-3,6/
C     Diagram 16
      DATA MAPCONFIG(16)/16/
      DATA (IFOREST(I,-1,16),I=1,2)/5,4/
      DATA SPROP(-1,16)/21/
      DATA TPRID(-1,16)/0/
      DATA (IFOREST(I,-2,16),I=1,2)/1,3/
      DATA TPRID(-2,16)/21/
      DATA SPROP(-2,16)/0/
      DATA (IFOREST(I,-3,16),I=1,2)/-2,-1/
      DATA TPRID(-3,16)/21/
      DATA SPROP(-3,16)/0/
      DATA (IFOREST(I,-4,16),I=1,2)/-3,6/
C     Diagram 17
      DATA MAPCONFIG(17)/17/
      DATA (IFOREST(I,-1,17),I=1,2)/5,4/
      DATA SPROP(-1,17)/21/
      DATA TPRID(-1,17)/0/
      DATA (IFOREST(I,-2,17),I=1,2)/6,-1/
      DATA SPROP(-2,17)/-2/
      DATA TPRID(-2,17)/0/
      DATA (IFOREST(I,-3,17),I=1,2)/1,3/
      DATA TPRID(-3,17)/21/
      DATA SPROP(-3,17)/0/
      DATA (IFOREST(I,-4,17),I=1,2)/-3,-2/
C     Diagram 18
      DATA MAPCONFIG(18)/18/
      DATA (IFOREST(I,-1,18),I=1,2)/5,4/
      DATA SPROP(-1,18)/21/
      DATA TPRID(-1,18)/0/
      DATA (IFOREST(I,-2,18),I=1,2)/1,3/
      DATA TPRID(-2,18)/21/
      DATA SPROP(-2,18)/0/
      DATA (IFOREST(I,-3,18),I=1,2)/-2,6/
      DATA TPRID(-3,18)/2/
      DATA SPROP(-3,18)/0/
      DATA (IFOREST(I,-4,18),I=1,2)/-3,-1/
C     Diagram 19
      DATA MAPCONFIG(19)/19/
      DATA (IFOREST(I,-1,19),I=1,2)/6,5/
      DATA SPROP(-1,19)/21/
      DATA TPRID(-1,19)/0/
      DATA (IFOREST(I,-2,19),I=1,2)/-1,4/
      DATA SPROP(-2,19)/-2/
      DATA TPRID(-2,19)/0/
      DATA (IFOREST(I,-3,19),I=1,2)/1,3/
      DATA TPRID(-3,19)/21/
      DATA SPROP(-3,19)/0/
      DATA (IFOREST(I,-4,19),I=1,2)/-3,-2/
C     Diagram 20
      DATA MAPCONFIG(20)/20/
      DATA (IFOREST(I,-1,20),I=1,2)/6,5/
      DATA SPROP(-1,20)/21/
      DATA TPRID(-1,20)/0/
      DATA (IFOREST(I,-2,20),I=1,2)/1,3/
      DATA TPRID(-2,20)/21/
      DATA SPROP(-2,20)/0/
      DATA (IFOREST(I,-3,20),I=1,2)/-2,4/
      DATA TPRID(-3,20)/2/
      DATA SPROP(-3,20)/0/
      DATA (IFOREST(I,-4,20),I=1,2)/-3,-1/
C     Diagram 21
      DATA MAPCONFIG(21)/21/
      DATA (IFOREST(I,-1,21),I=1,2)/1,5/
      DATA TPRID(-1,21)/21/
      DATA SPROP(-1,21)/0/
      DATA (IFOREST(I,-2,21),I=1,2)/-1,3/
      DATA TPRID(-2,21)/2/
      DATA SPROP(-2,21)/0/
      DATA (IFOREST(I,-3,21),I=1,2)/-2,6/
      DATA TPRID(-3,21)/21/
      DATA SPROP(-3,21)/0/
      DATA (IFOREST(I,-4,21),I=1,2)/-3,4/
C     Diagram 22
      DATA MAPCONFIG(22)/22/
      DATA (IFOREST(I,-1,22),I=1,2)/1,5/
      DATA TPRID(-1,22)/21/
      DATA SPROP(-1,22)/0/
      DATA (IFOREST(I,-2,22),I=1,2)/-1,6/
      DATA TPRID(-2,22)/2/
      DATA SPROP(-2,22)/0/
      DATA (IFOREST(I,-3,22),I=1,2)/-2,3/
      DATA TPRID(-3,22)/21/
      DATA SPROP(-3,22)/0/
      DATA (IFOREST(I,-4,22),I=1,2)/-3,4/
C     Diagram 23
      DATA MAPCONFIG(23)/23/
      DATA (IFOREST(I,-1,23),I=1,2)/6,3/
      DATA SPROP(-1,23)/21/
      DATA TPRID(-1,23)/0/
      DATA (IFOREST(I,-2,23),I=1,2)/1,5/
      DATA TPRID(-2,23)/21/
      DATA SPROP(-2,23)/0/
      DATA (IFOREST(I,-3,23),I=1,2)/-2,-1/
      DATA TPRID(-3,23)/21/
      DATA SPROP(-3,23)/0/
      DATA (IFOREST(I,-4,23),I=1,2)/-3,4/
C     Diagram 24
      DATA MAPCONFIG(24)/24/
      DATA (IFOREST(I,-1,24),I=1,2)/1,5/
      DATA TPRID(-1,24)/21/
      DATA SPROP(-1,24)/0/
      DATA (IFOREST(I,-2,24),I=1,2)/-1,3/
      DATA TPRID(-2,24)/2/
      DATA SPROP(-2,24)/0/
      DATA (IFOREST(I,-3,24),I=1,2)/-2,4/
      DATA TPRID(-3,24)/21/
      DATA SPROP(-3,24)/0/
      DATA (IFOREST(I,-4,24),I=1,2)/-3,6/
C     Diagram 25
      DATA MAPCONFIG(25)/25/
      DATA (IFOREST(I,-1,25),I=1,2)/1,5/
      DATA TPRID(-1,25)/21/
      DATA SPROP(-1,25)/0/
      DATA (IFOREST(I,-2,25),I=1,2)/-1,4/
      DATA TPRID(-2,25)/2/
      DATA SPROP(-2,25)/0/
      DATA (IFOREST(I,-3,25),I=1,2)/-2,3/
      DATA TPRID(-3,25)/21/
      DATA SPROP(-3,25)/0/
      DATA (IFOREST(I,-4,25),I=1,2)/-3,6/
C     Diagram 26
      DATA MAPCONFIG(26)/26/
      DATA (IFOREST(I,-1,26),I=1,2)/4,3/
      DATA SPROP(-1,26)/21/
      DATA TPRID(-1,26)/0/
      DATA (IFOREST(I,-2,26),I=1,2)/1,5/
      DATA TPRID(-2,26)/21/
      DATA SPROP(-2,26)/0/
      DATA (IFOREST(I,-3,26),I=1,2)/-2,-1/
      DATA TPRID(-3,26)/21/
      DATA SPROP(-3,26)/0/
      DATA (IFOREST(I,-4,26),I=1,2)/-3,6/
C     Diagram 27
      DATA MAPCONFIG(27)/27/
      DATA (IFOREST(I,-1,27),I=1,2)/4,3/
      DATA SPROP(-1,27)/21/
      DATA TPRID(-1,27)/0/
      DATA (IFOREST(I,-2,27),I=1,2)/6,-1/
      DATA SPROP(-2,27)/-2/
      DATA TPRID(-2,27)/0/
      DATA (IFOREST(I,-3,27),I=1,2)/1,5/
      DATA TPRID(-3,27)/21/
      DATA SPROP(-3,27)/0/
      DATA (IFOREST(I,-4,27),I=1,2)/-3,-2/
C     Diagram 28
      DATA MAPCONFIG(28)/28/
      DATA (IFOREST(I,-1,28),I=1,2)/4,3/
      DATA SPROP(-1,28)/21/
      DATA TPRID(-1,28)/0/
      DATA (IFOREST(I,-2,28),I=1,2)/1,5/
      DATA TPRID(-2,28)/21/
      DATA SPROP(-2,28)/0/
      DATA (IFOREST(I,-3,28),I=1,2)/-2,6/
      DATA TPRID(-3,28)/2/
      DATA SPROP(-3,28)/0/
      DATA (IFOREST(I,-4,28),I=1,2)/-3,-1/
C     Diagram 29
      DATA MAPCONFIG(29)/29/
      DATA (IFOREST(I,-1,29),I=1,2)/6,3/
      DATA SPROP(-1,29)/21/
      DATA TPRID(-1,29)/0/
      DATA (IFOREST(I,-2,29),I=1,2)/4,-1/
      DATA SPROP(-2,29)/-2/
      DATA TPRID(-2,29)/0/
      DATA (IFOREST(I,-3,29),I=1,2)/1,5/
      DATA TPRID(-3,29)/21/
      DATA SPROP(-3,29)/0/
      DATA (IFOREST(I,-4,29),I=1,2)/-3,-2/
C     Diagram 30
      DATA MAPCONFIG(30)/30/
      DATA (IFOREST(I,-1,30),I=1,2)/6,3/
      DATA SPROP(-1,30)/21/
      DATA TPRID(-1,30)/0/
      DATA (IFOREST(I,-2,30),I=1,2)/1,5/
      DATA TPRID(-2,30)/21/
      DATA SPROP(-2,30)/0/
      DATA (IFOREST(I,-3,30),I=1,2)/-2,4/
      DATA TPRID(-3,30)/2/
      DATA SPROP(-3,30)/0/
      DATA (IFOREST(I,-4,30),I=1,2)/-3,-1/
C     Diagram 31
      DATA MAPCONFIG(31)/31/
      DATA (IFOREST(I,-1,31),I=1,2)/6,3/
      DATA SPROP(-1,31)/21/
      DATA TPRID(-1,31)/0/
      DATA (IFOREST(I,-2,31),I=1,2)/5,-1/
      DATA SPROP(-2,31)/2/
      DATA TPRID(-2,31)/0/
      DATA (IFOREST(I,-3,31),I=1,2)/1,-2/
      DATA TPRID(-3,31)/21/
      DATA SPROP(-3,31)/0/
      DATA (IFOREST(I,-4,31),I=1,2)/-3,4/
C     Diagram 32
      DATA MAPCONFIG(32)/32/
      DATA (IFOREST(I,-1,32),I=1,2)/6,3/
      DATA SPROP(-1,32)/21/
      DATA TPRID(-1,32)/0/
      DATA (IFOREST(I,-2,32),I=1,2)/1,-1/
      DATA TPRID(-2,32)/2/
      DATA SPROP(-2,32)/0/
      DATA (IFOREST(I,-3,32),I=1,2)/-2,5/
      DATA TPRID(-3,32)/21/
      DATA SPROP(-3,32)/0/
      DATA (IFOREST(I,-4,32),I=1,2)/-3,4/
C     Diagram 33
      DATA MAPCONFIG(33)/33/
      DATA (IFOREST(I,-1,33),I=1,2)/6,5/
      DATA SPROP(-1,33)/21/
      DATA TPRID(-1,33)/0/
      DATA (IFOREST(I,-2,33),I=1,2)/-1,3/
      DATA SPROP(-2,33)/2/
      DATA TPRID(-2,33)/0/
      DATA (IFOREST(I,-3,33),I=1,2)/1,-2/
      DATA TPRID(-3,33)/21/
      DATA SPROP(-3,33)/0/
      DATA (IFOREST(I,-4,33),I=1,2)/-3,4/
C     Diagram 34
      DATA MAPCONFIG(34)/34/
      DATA (IFOREST(I,-1,34),I=1,2)/6,5/
      DATA SPROP(-1,34)/21/
      DATA TPRID(-1,34)/0/
      DATA (IFOREST(I,-2,34),I=1,2)/1,-1/
      DATA TPRID(-2,34)/2/
      DATA SPROP(-2,34)/0/
      DATA (IFOREST(I,-3,34),I=1,2)/-2,3/
      DATA TPRID(-3,34)/21/
      DATA SPROP(-3,34)/0/
      DATA (IFOREST(I,-4,34),I=1,2)/-3,4/
C     Diagram 35
      DATA MAPCONFIG(35)/35/
      DATA (IFOREST(I,-1,35),I=1,2)/4,3/
      DATA SPROP(-1,35)/21/
      DATA TPRID(-1,35)/0/
      DATA (IFOREST(I,-2,35),I=1,2)/5,-1/
      DATA SPROP(-2,35)/2/
      DATA TPRID(-2,35)/0/
      DATA (IFOREST(I,-3,35),I=1,2)/1,-2/
      DATA TPRID(-3,35)/21/
      DATA SPROP(-3,35)/0/
      DATA (IFOREST(I,-4,35),I=1,2)/-3,6/
C     Diagram 36
      DATA MAPCONFIG(36)/36/
      DATA (IFOREST(I,-1,36),I=1,2)/4,3/
      DATA SPROP(-1,36)/21/
      DATA TPRID(-1,36)/0/
      DATA (IFOREST(I,-2,36),I=1,2)/1,-1/
      DATA TPRID(-2,36)/2/
      DATA SPROP(-2,36)/0/
      DATA (IFOREST(I,-3,36),I=1,2)/-2,5/
      DATA TPRID(-3,36)/21/
      DATA SPROP(-3,36)/0/
      DATA (IFOREST(I,-4,36),I=1,2)/-3,6/
C     Diagram 37
      DATA MAPCONFIG(37)/37/
      DATA (IFOREST(I,-1,37),I=1,2)/5,4/
      DATA SPROP(-1,37)/21/
      DATA TPRID(-1,37)/0/
      DATA (IFOREST(I,-2,37),I=1,2)/-1,3/
      DATA SPROP(-2,37)/2/
      DATA TPRID(-2,37)/0/
      DATA (IFOREST(I,-3,37),I=1,2)/1,-2/
      DATA TPRID(-3,37)/21/
      DATA SPROP(-3,37)/0/
      DATA (IFOREST(I,-4,37),I=1,2)/-3,6/
C     Diagram 38
      DATA MAPCONFIG(38)/38/
      DATA (IFOREST(I,-1,38),I=1,2)/5,4/
      DATA SPROP(-1,38)/21/
      DATA TPRID(-1,38)/0/
      DATA (IFOREST(I,-2,38),I=1,2)/1,-1/
      DATA TPRID(-2,38)/2/
      DATA SPROP(-2,38)/0/
      DATA (IFOREST(I,-3,38),I=1,2)/-2,3/
      DATA TPRID(-3,38)/21/
      DATA SPROP(-3,38)/0/
      DATA (IFOREST(I,-4,38),I=1,2)/-3,6/
C     Diagram 39
      DATA MAPCONFIG(39)/39/
      DATA (IFOREST(I,-1,39),I=1,2)/6,5/
      DATA SPROP(-1,39)/21/
      DATA TPRID(-1,39)/0/
      DATA (IFOREST(I,-2,39),I=1,2)/4,3/
      DATA SPROP(-2,39)/21/
      DATA TPRID(-2,39)/0/
      DATA (IFOREST(I,-3,39),I=1,2)/1,-2/
      DATA TPRID(-3,39)/2/
      DATA SPROP(-3,39)/0/
      DATA (IFOREST(I,-4,39),I=1,2)/-3,-1/
C     Diagram 40
      DATA MAPCONFIG(40)/40/
      DATA (IFOREST(I,-1,40),I=1,2)/4,3/
      DATA SPROP(-1,40)/21/
      DATA TPRID(-1,40)/0/
      DATA (IFOREST(I,-2,40),I=1,2)/6,5/
      DATA SPROP(-2,40)/21/
      DATA TPRID(-2,40)/0/
      DATA (IFOREST(I,-3,40),I=1,2)/1,-2/
      DATA TPRID(-3,40)/2/
      DATA SPROP(-3,40)/0/
      DATA (IFOREST(I,-4,40),I=1,2)/-3,-1/
C     Diagram 41
      DATA MAPCONFIG(41)/41/
      DATA (IFOREST(I,-1,41),I=1,2)/5,4/
      DATA SPROP(-1,41)/21/
      DATA TPRID(-1,41)/0/
      DATA (IFOREST(I,-2,41),I=1,2)/6,3/
      DATA SPROP(-2,41)/21/
      DATA TPRID(-2,41)/0/
      DATA (IFOREST(I,-3,41),I=1,2)/1,-2/
      DATA TPRID(-3,41)/2/
      DATA SPROP(-3,41)/0/
      DATA (IFOREST(I,-4,41),I=1,2)/-3,-1/
C     Diagram 42
      DATA MAPCONFIG(42)/42/
      DATA (IFOREST(I,-1,42),I=1,2)/6,3/
      DATA SPROP(-1,42)/21/
      DATA TPRID(-1,42)/0/
      DATA (IFOREST(I,-2,42),I=1,2)/5,4/
      DATA SPROP(-2,42)/21/
      DATA TPRID(-2,42)/0/
      DATA (IFOREST(I,-3,42),I=1,2)/1,-2/
      DATA TPRID(-3,42)/2/
      DATA SPROP(-3,42)/0/
      DATA (IFOREST(I,-4,42),I=1,2)/-3,-1/
C     Number of configs
      DATA MAPCONFIG(0)/42/
""")

        # Test coloramps.inc output
        self.assertEqual("\n".join(\
                       exporter.get_icolamp_lines(mapconfigs,
                                                   matrix_element, 1)),
                         """DATA(icolamp(i,1,1),i=1,6)/.true.,.true.,.false.,.true.,.false.,.true./
DATA(icolamp(i,2,1),i=1,6)/.true.,.true.,.false.,.false.,.true.,.true./
DATA(icolamp(i,3,1),i=1,6)/.false.,.false.,.false.,.true.,.true.,.false./
DATA(icolamp(i,4,1),i=1,6)/.true.,.true.,.true.,.true.,.false.,.false./
DATA(icolamp(i,5,1),i=1,6)/.true.,.true.,.false.,.true.,.false.,.true./
DATA(icolamp(i,6,1),i=1,6)/.false.,.false.,.true.,.false.,.false.,.true./
DATA(icolamp(i,7,1),i=1,6)/.true.,.true.,.true.,.false.,.true.,.false./
DATA(icolamp(i,8,1),i=1,6)/.true.,.true.,.false.,.false.,.true.,.true./
DATA(icolamp(i,9,1),i=1,6)/.true.,.true.,.true.,.false.,.true.,.false./
DATA(icolamp(i,10,1),i=1,6)/.true.,.true.,.true.,.true.,.false.,.false./
DATA(icolamp(i,11,1),i=1,6)/.false.,.false.,.true.,.true.,.true.,.true./
DATA(icolamp(i,12,1),i=1,6)/.false.,.true.,.true.,.true.,.true.,.false./
DATA(icolamp(i,13,1),i=1,6)/.false.,.true.,.false.,.false.,.false.,.true./
DATA(icolamp(i,14,1),i=1,6)/.true.,.false.,.true.,.false.,.true.,.true./
DATA(icolamp(i,15,1),i=1,6)/.false.,.false.,.true.,.true.,.true.,.true./
DATA(icolamp(i,16,1),i=1,6)/.true.,.false.,.false.,.true.,.false.,.false./
DATA(icolamp(i,17,1),i=1,6)/.true.,.true.,.true.,.false.,.true.,.false./
DATA(icolamp(i,18,1),i=1,6)/.false.,.true.,.true.,.true.,.true.,.false./
DATA(icolamp(i,19,1),i=1,6)/.true.,.true.,.true.,.false.,.true.,.false./
DATA(icolamp(i,20,1),i=1,6)/.true.,.false.,.true.,.false.,.true.,.true./
DATA(icolamp(i,21,1),i=1,6)/.false.,.false.,.true.,.true.,.true.,.true./
DATA(icolamp(i,22,1),i=1,6)/.true.,.false.,.true.,.true.,.false.,.true./
DATA(icolamp(i,23,1),i=1,6)/.true.,.false.,.false.,.false.,.true.,.false./
DATA(icolamp(i,24,1),i=1,6)/.false.,.false.,.true.,.true.,.true.,.true./
DATA(icolamp(i,25,1),i=1,6)/.false.,.true.,.false.,.true.,.true.,.true./
DATA(icolamp(i,26,1),i=1,6)/.false.,.true.,.true.,.false.,.false.,.false./
DATA(icolamp(i,27,1),i=1,6)/.true.,.true.,.false.,.true.,.false.,.true./
DATA(icolamp(i,28,1),i=1,6)/.true.,.false.,.true.,.true.,.false.,.true./
DATA(icolamp(i,29,1),i=1,6)/.true.,.true.,.false.,.true.,.false.,.true./
DATA(icolamp(i,30,1),i=1,6)/.false.,.true.,.false.,.true.,.true.,.true./
DATA(icolamp(i,31,1),i=1,6)/.true.,.true.,.true.,.true.,.false.,.false./
DATA(icolamp(i,32,1),i=1,6)/.false.,.true.,.true.,.true.,.true.,.false./
DATA(icolamp(i,33,1),i=1,6)/.true.,.true.,.true.,.true.,.false.,.false./
DATA(icolamp(i,34,1),i=1,6)/.true.,.false.,.true.,.true.,.false.,.true./
DATA(icolamp(i,35,1),i=1,6)/.true.,.true.,.false.,.false.,.true.,.true./
DATA(icolamp(i,36,1),i=1,6)/.true.,.false.,.true.,.false.,.true.,.true./
DATA(icolamp(i,37,1),i=1,6)/.true.,.true.,.false.,.false.,.true.,.true./
DATA(icolamp(i,38,1),i=1,6)/.false.,.true.,.false.,.true.,.true.,.true./
DATA(icolamp(i,39,1),i=1,6)/.true.,.false.,.true.,.false.,.true.,.true./
DATA(icolamp(i,40,1),i=1,6)/.true.,.false.,.true.,.true.,.false.,.true./
DATA(icolamp(i,41,1),i=1,6)/.false.,.true.,.true.,.true.,.true.,.false./
DATA(icolamp(i,42,1),i=1,6)/.false.,.true.,.false.,.true.,.true.,.true./"""
)

        # Test leshouche.inc output
        writer = writers.FortranWriter(self.give_pos('leshouche'))
        exporter.write_leshouche_file(writer, matrix_element)
        writer.close()

        self.assertFileContains('leshouche',
                         """      DATA (IDUP(I,1,1),I=1,6)/2,-2,2,-2,2,-2/
      DATA (MOTHUP(1,I),I=1, 6)/  0,  0,  1,  1,  1,  1/
      DATA (MOTHUP(2,I),I=1, 6)/  0,  0,  2,  2,  2,  2/
      DATA (ICOLUP(1,I,1,1),I=1, 6)/501,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,1,1),I=1, 6)/  0,501,  0,502,  0,503/
      DATA (ICOLUP(1,I,2,1),I=1, 6)/501,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,2,1),I=1, 6)/  0,501,  0,503,  0,502/
      DATA (ICOLUP(1,I,3,1),I=1, 6)/502,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,3,1),I=1, 6)/  0,501,  0,501,  0,503/
      DATA (ICOLUP(1,I,4,1),I=1, 6)/503,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,4,1),I=1, 6)/  0,501,  0,501,  0,502/
      DATA (ICOLUP(1,I,5,1),I=1, 6)/502,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,5,1),I=1, 6)/  0,501,  0,503,  0,501/
      DATA (ICOLUP(1,I,6,1),I=1, 6)/503,  0,502,  0,503,  0/
      DATA (ICOLUP(2,I,6,1),I=1, 6)/  0,501,  0,502,  0,501/
""")

        # Test pdf output (for auto_dsig.f)

        self.assertEqual(exporter.get_pdf_lines(matrix_element, 2),
                         """IF (ABS(LPP(1)) .GE. 1) THEN
LP=SIGN(1,LPP(1))
u1=PDG2PDF(ABS(LPP(1)),2*LP,XBK(1),DSQRT(Q2FACT(1)))
ENDIF
IF (ABS(LPP(2)) .GE. 1) THEN
LP=SIGN(1,LPP(2))
ub2=PDG2PDF(ABS(LPP(2)),-2*LP,XBK(2),DSQRT(Q2FACT(2)))
ENDIF
PD(0) = 0d0
IPROC = 0
IPROC=IPROC+1 ! u u~ > u u~ u u~
PD(IPROC)=PD(IPROC-1) + u1*ub2""")

        # Test mg.sym
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_mg_sym_file(writer, matrix_element)
        writer.close()
        
        self.assertFileContains('test',
                         """      2
      2
      3
      5
      2
      4
      6
""")

    def test_generate_helas_diagrams_gg_gg(self):
        """Test calls for g g > g g"""

        # Set up local model

        mybasemodel = base_objects.Model()
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

        # Gluon self-couplings
        my_color_string = color.ColorString([color.f(0, 1, 2)])
        my_color_string.is_imaginary = True
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g]),
                      'color': [my_color_string],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [g, \
                                             g, \
                                             g,
                                             g]),
                      'color': [color.ColorString([color.f(0, 1, -1),
                                                   color.f(2, 3, -1)]),
                                color.ColorString([color.f(2, 0, -1),
                                                   color.f(1, 3, -1)]),
                                color.ColorString([color.f(1, 2, -1),
                                                   color.f(0, 3, -1)])],
                      'lorentz':['gggg1', 'gggg2', 'gggg3'],
                      'couplings':{(0, 0):'GG', (1, 1):'GG', (2, 2):'GG'},
                      'orders':{'QCD':2}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

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
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        # Test Helas calls

        fortran_model = helas_call_writers.FortranHelasCallWriter(mybasemodel)

        self.assertEqual("\n".join(fortran_model.\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
# Amplitude(s) for diagram number 1
CALL GGGGXX(W(1,1),W(1,2),W(1,3),W(1,4),GG,AMP(1))
CALL GGGGXX(W(1,3),W(1,1),W(1,2),W(1,4),GG,AMP(2))
CALL GGGGXX(W(1,2),W(1,3),W(1,1),W(1,4),GG,AMP(3))
CALL JVVXXX(W(1,1),W(1,2),GG,zero,zero,W(1,5))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,3),W(1,4),W(1,5),GG,AMP(4))
CALL JVVXXX(W(1,1),W(1,3),GG,zero,zero,W(1,6))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,2),W(1,4),W(1,6),GG,AMP(5))
CALL JVVXXX(W(1,1),W(1,4),GG,zero,zero,W(1,7))
# Amplitude(s) for diagram number 4
CALL VVVXXX(W(1,2),W(1,3),W(1,7),GG,AMP(6))""")

        exporter = export_v4.ProcessExporterFortranME()

        # Test color matrix output
        self.assertEqual("\n".join(exporter.get_color_data_lines(\
                         matrix_element)),
                         """DATA Denom(1)/6/
DATA (CF(i,  1),i=  1,  6) /   19,   -2,   -2,   -2,   -2,    4/
C 1 Tr(1,2,3,4)
DATA Denom(2)/6/
DATA (CF(i,  2),i=  1,  6) /   -2,   19,   -2,    4,   -2,   -2/
C 1 Tr(1,2,4,3)
DATA Denom(3)/6/
DATA (CF(i,  3),i=  1,  6) /   -2,   -2,   19,   -2,    4,   -2/
C 1 Tr(1,3,2,4)
DATA Denom(4)/6/
DATA (CF(i,  4),i=  1,  6) /   -2,    4,   -2,   19,   -2,   -2/
C 1 Tr(1,3,4,2)
DATA Denom(5)/6/
DATA (CF(i,  5),i=  1,  6) /   -2,   -2,    4,   -2,   19,   -2/
C 1 Tr(1,4,2,3)
DATA Denom(6)/6/
DATA (CF(i,  6),i=  1,  6) /    4,   -2,   -2,   -2,   -2,   19/
C 1 Tr(1,4,3,2)""")

        # Test JAMP (color amplitude) output
        self.assertEqual("\n".join(exporter.get_JAMP_lines(matrix_element)),
                         """JAMP(1)=+2*(+AMP(3)-AMP(1)+AMP(4)-AMP(6))
JAMP(2)=+2*(+AMP(1)-AMP(2)-AMP(4)-AMP(5))
JAMP(3)=+2*(-AMP(3)+AMP(2)+AMP(5)+AMP(6))
JAMP(4)=+2*(+AMP(1)-AMP(2)-AMP(4)-AMP(5))
JAMP(5)=+2*(-AMP(3)+AMP(2)+AMP(5)+AMP(6))
JAMP(6)=+2*(+AMP(3)-AMP(1)+AMP(4)-AMP(6))""")

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(2)=AMP2(2)+AMP(4)*dconjg(AMP(4))',
                          'AMP2(3)=AMP2(3)+AMP(5)*dconjg(AMP(5))',
                          'AMP2(4)=AMP2(4)+AMP(6)*dconjg(AMP(6))'])
        
        # Test configs file
        writer = writers.FortranWriter(self.give_pos('test'))
        nconfig, s_and_t_channels = exporter.write_configs_file(writer,
                                     matrix_element)
        writer.close()
        self.assertFileContains('test',
"""C     Diagram 2
      DATA MAPCONFIG(1)/2/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA SPROP(-1,1)/21/
      DATA TPRID(-1,1)/0/
C     Diagram 3
      DATA MAPCONFIG(2)/3/
      DATA (IFOREST(I,-1,2),I=1,2)/1,3/
      DATA TPRID(-1,2)/21/
      DATA SPROP(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,4/
C     Diagram 4
      DATA MAPCONFIG(3)/4/
      DATA (IFOREST(I,-1,3),I=1,2)/1,4/
      DATA TPRID(-1,3)/21/
      DATA SPROP(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,3/
C     Number of configs
      DATA MAPCONFIG(0)/3/
""")

    def test_generate_helas_diagrams_uu_susu(self):
        """Testing the helas diagram generation u u > su su with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4, apart from the sign! (AMP 1,2)
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(self.mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL OXXXXX(P(0,1),mu,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),mu,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL FSOCXX(W(1,1),W(1,3),MGVX575,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,2),W(1,5),W(1,4),MGVX575,AMP(1))
CALL FSOCXX(W(1,1),W(1,4),MGVX575,Mneu1,Wneu1,W(1,6))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,2),W(1,6),W(1,3),MGVX575,AMP(2))""")

    def test_generate_helas_diagrams_zz_n1n1(self):
        """Testing the helas diagram generation z z > n1 n1 with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(self.mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),zmas,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zmas,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),Mneu1,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),Mneu1,NHEL(4),-1*IC(4),W(1,4))
CALL FVOXXX(W(1,3),W(1,1),GZN11,Mneu1,Wneu1,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,4),W(1,5),W(1,2),GZN11,AMP(1))
CALL FVIXXX(W(1,4),W(1,1),GZN11,Mneu1,Wneu1,W(1,6))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,6),W(1,3),W(1,2),GZN11,AMP(2))""")

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(exporter.get_JAMP_lines(matrix_element)[0],
                         "JAMP(1)=-AMP(1)-AMP(2)")


    def test_generate_helas_diagrams_epem_elpelmepem(self):
        """Testing the helas diagram generation e+ e- > sl2+ sl2- e+ e-
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'me',
                      'width':'zero',
                      'texname':'e^-',
                      'antitexname':'e^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        eminus = mypartlist[len(mypartlist) - 1]
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

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        #print myamplitude.get('process').nice_string()
        #print "\n".join(helas_call_writers.FortranHelasCallWriter().\
        #                get_matrix_element_calls(matrix_element))
        #print helas_call_writers.FortranHelasCallWriter().get_JAMP_line(matrix_element)



        # I have checked that the resulting Helas calls below give
        # identical result as MG4 (when fermionfactors are taken into
        # account)
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL OXXXXX(P(0,1),me,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),me,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL OXXXXX(P(0,6),me,NHEL(6),+1*IC(6),W(1,6))
CALL FSOXXX(W(1,1),W(1,3),MGVX350,Mneu1,Wneu1,W(1,7))
CALL FSIXXX(W(1,2),W(1,4),MGVX494,Mneu1,Wneu1,W(1,8))
CALL HIOXXX(W(1,5),W(1,7),MGVX494,Msl2,Wsl2,W(1,9))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,8),W(1,6),W(1,9),MGVX350,AMP(1))
CALL IXXXXX(P(0,1),me,NHEL(1),+1*IC(1),W(1,10))
CALL FSICXX(W(1,10),W(1,3),MGVX350,Mneu1,Wneu1,W(1,11))
CALL HIOXXX(W(1,11),W(1,6),MGVX350,Msl2,Wsl2,W(1,12))
CALL OXXXXX(P(0,2),me,NHEL(2),-1*IC(2),W(1,13))
CALL FSOCXX(W(1,13),W(1,4),MGVX494,Mneu1,Wneu1,W(1,14))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,5),W(1,14),W(1,12),MGVX494,AMP(2))
CALL FSIXXX(W(1,5),W(1,4),MGVX494,Mneu1,Wneu1,W(1,15))
CALL HIOXXX(W(1,2),W(1,7),MGVX494,Msl2,Wsl2,W(1,16))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,15),W(1,6),W(1,16),MGVX350,AMP(3))
CALL OXXXXX(P(0,5),me,NHEL(5),+1*IC(5),W(1,17))
CALL FSOCXX(W(1,17),W(1,4),MGVX494,Mneu1,Wneu1,W(1,18))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,2),W(1,18),W(1,12),MGVX494,AMP(4))
CALL FSOXXX(W(1,6),W(1,3),MGVX350,Mneu1,Wneu1,W(1,19))
CALL HIOXXX(W(1,8),W(1,1),MGVX350,Msl2,Wsl2,W(1,20))
# Amplitude(s) for diagram number 5
CALL IOSXXX(W(1,5),W(1,19),W(1,20),MGVX494,AMP(5))
CALL IXXXXX(P(0,6),me,NHEL(6),-1*IC(6),W(1,21))
CALL FSICXX(W(1,21),W(1,3),MGVX350,Mneu1,Wneu1,W(1,22))
CALL HIOXXX(W(1,22),W(1,1),MGVX350,Msl2,Wsl2,W(1,23))
# Amplitude(s) for diagram number 6
CALL IOSXXX(W(1,5),W(1,14),W(1,23),MGVX494,AMP(6))
# Amplitude(s) for diagram number 7
CALL IOSXXX(W(1,2),W(1,18),W(1,23),MGVX494,AMP(7))
CALL HIOXXX(W(1,15),W(1,1),MGVX350,Msl2,Wsl2,W(1,24))
# Amplitude(s) for diagram number 8
CALL IOSXXX(W(1,2),W(1,19),W(1,24),MGVX494,AMP(8))""")

        # Test find_outgoing_number
        goal_numbers = [1, 2, 3, 2, 3, 1, 2, 3, 1, 1, 3, 2, 3, 3]

        i = 0
        for wf in matrix_element.get_all_wavefunctions():
            if not wf.get('interaction_id'):
                continue
            self.assertEqual(wf.find_outgoing_number(), goal_numbers[i])
            i += 1

        # Test get_used_lorentz
        # Wavefunctions
        goal_lorentz_list = [('', (), 1), ('', (), 2), ('', (), 3),
                             ('', (1,), 2),('', (), 3), ('', (1,), 1),
                             ('', (), 2), ('', (), 3),('', (1,), 1),
                             ('', (), 1), ('', (), 3),('', (1,), 2),
                             ('', (), 3), ('', (), 3)]
        # Amplitudes
        goal_lorentz_list += [('', (), 0)] * 8
        self.assertEqual(matrix_element.get_used_lorentz(),
                         goal_lorentz_list)


    def test_generate_helas_diagrams_uu_susug(self):
        """Testing the helas diagram generation u u > su su with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4, apart from sign! (AMP 1,2,5,6)
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(self.mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL OXXXXX(P(0,1),mu,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),mu,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FSOCXX(W(1,1),W(1,3),MGVX575,Mneu1,Wneu1,W(1,6))
CALL FVIXXX(W(1,2),W(1,5),GG,mu,zero,W(1,7))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,7),W(1,6),W(1,4),MGVX575,AMP(1))
CALL HVSXXX(W(1,5),W(1,4),MGVX74,Musq2,Wusq2,W(1,8))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,2),W(1,6),W(1,8),MGVX575,AMP(2))
CALL FSOCXX(W(1,1),W(1,4),MGVX575,Mneu1,Wneu1,W(1,9))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,7),W(1,9),W(1,3),MGVX575,AMP(3))
CALL HVSXXX(W(1,5),W(1,3),MGVX74,Musq2,Wusq2,W(1,10))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,2),W(1,9),W(1,10),MGVX575,AMP(4))
CALL FVOCXX(W(1,1),W(1,5),GG,mu,zero,W(1,11))
CALL FSIXXX(W(1,2),W(1,3),MGVX575,Mneu1,Wneu1,W(1,12))
# Amplitude(s) for diagram number 5
CALL IOSCXX(W(1,12),W(1,11),W(1,4),MGVX575,AMP(5))
CALL FSIXXX(W(1,2),W(1,4),MGVX575,Mneu1,Wneu1,W(1,13))
# Amplitude(s) for diagram number 6
CALL IOSCXX(W(1,13),W(1,11),W(1,3),MGVX575,AMP(6))
# Amplitude(s) for diagram number 7
CALL IOSCXX(W(1,12),W(1,1),W(1,8),MGVX575,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOSCXX(W(1,13),W(1,1),W(1,10),MGVX575,AMP(8))""")

    def test_generate_helas_diagrams_enu_enu(self):
        """Testing the helas diagram generation e- nubar > e- nubar
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A electron and positron
        mypartlist.append(base_objects.Particle({'name':'e-',
                      'antiname':'e+',
                      'spin':2,
                      'color':1,
                      'mass':'me',
                      'width':'zero',
                      'texname':'e^-',
                      'antitexname':'e^+',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':11,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A neutrino
        mypartlist.append(base_objects.Particle({'name':'ve',
                      'antiname':'ve~',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\nu_e',
                      'antitexname':'\bar\nu_e',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':12,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        nu = mypartlist[len(mypartlist) - 1]
        nubar = copy.copy(nu)
        nubar.set('is_part', False)

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
        Wplus = mypartlist[len(mypartlist) - 1]
        Wminus = copy.copy(Wplus)
        Wminus.set('is_part', False)

        # Coupling of W- e+ nu_e

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             nu, \
                                             Wminus]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [nubar, \
                                             eminus, \
                                             Wplus]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-12,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-12,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),me,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),me,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),zero,NHEL(4),-1*IC(4),W(1,4))
CALL JIOXXX(W(1,1),W(1,2),MGVX27,MW,WW,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,4),W(1,3),W(1,5),MGVX27,AMP(1))""")

    def test_generate_helas_diagrams_WWWW(self):
        """Testing the helas diagram generation W+ W- > W+ W-
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
        Wplus = mypartlist[len(mypartlist) - 1]
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
        a = mypartlist[len(mypartlist) - 1]

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
        Z = mypartlist[len(mypartlist) - 1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Z]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX5'},
            'orders':{'QED':1}}))

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

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.  Note that this looks like it uses
        # incoming bosons instead of outgoing though
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MW,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),MW,NHEL(4),+1*IC(4),W(1,4))
# Amplitude(s) for diagram number 1
CALL W3W3NX(W(1,2),W(1,1),W(1,3),W(1,4),MGVX6,DUM0,AMP(1))
CALL JVVXXX(W(1,2),W(1,1),MGVX3,zero,zero,W(1,5))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,3),W(1,4),W(1,5),MGVX3,AMP(2))
CALL JVVXXX(W(1,2),W(1,1),MGVX5,MZ,WZ,W(1,6))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,3),W(1,4),W(1,6),MGVX5,AMP(3))
CALL JVVXXX(W(1,3),W(1,1),MGVX3,zero,zero,W(1,7))
# Amplitude(s) for diagram number 4
CALL VVVXXX(W(1,2),W(1,4),W(1,7),MGVX3,AMP(4))
CALL JVVXXX(W(1,3),W(1,1),MGVX5,MZ,WZ,W(1,8))
# Amplitude(s) for diagram number 5
CALL VVVXXX(W(1,2),W(1,4),W(1,8),MGVX5,AMP(5))""")

    def test_generate_helas_diagrams_WWZA(self):
        """Testing the helas diagram generation W+ W- > Z A
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
        Wplus = mypartlist[len(mypartlist) - 1]
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
        a = mypartlist[len(mypartlist) - 1]

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
        Z = mypartlist[len(mypartlist) - 1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Wplus, \
                                             Z]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX5'},
            'orders':{'QED':1}}))

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
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX7'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Z, \
                                             Wminus,
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
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MZ,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
# Amplitude(s) for diagram number 1
CALL W3W3NX(W(1,2),W(1,4),W(1,1),W(1,3),MGVX7,DUM0,AMP(1))
CALL JVVXXX(W(1,3),W(1,1),MGVX5,MW,WW,W(1,5))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,2),W(1,5),W(1,4),MGVX3,AMP(2))
CALL JVVXXX(W(1,1),W(1,4),MGVX3,MW,WW,W(1,6))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,6),W(1,2),W(1,3),MGVX5,AMP(3))""")


    def test_generate_helas_diagrams_WWWWA(self):
        """Testing the helas diagram generation W+ W- > W+ W- a
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
        Wplus = mypartlist[len(mypartlist) - 1]
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
        a = mypartlist[len(mypartlist) - 1]

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
        Z = mypartlist[len(mypartlist) - 1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Wplus, \
                                             Z]),
            'color': [],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX5'},
            'orders':{'QED':1}}))

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
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             Z]),
            'color': [],
            'lorentz':['WWVVN'],
            'couplings':{(0, 0):'MGVX7'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Z, \
                                             Wminus,
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
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        #print myamplitude.get('process').nice_string()
        #print myamplitude.get('diagrams').nice_string()

        #print "Keys:"
        #for diagram in matrix_element.get('diagrams'):
        #    for wf in diagram.get('wavefunctions'):
        #        print wf.get_call_key()
        #    print diagram.get('amplitude').get_call_key()

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.
        self.assertEqual("\n".join(helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MW,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),MW,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL JVVXXX(W(1,2),W(1,1),MGVX3,zero,zero,W(1,6))
CALL JVVXXX(W(1,5),W(1,3),MGVX3,MW,WW,W(1,7))
# Amplitude(s) for diagram number 1
CALL VVVXXX(W(1,7),W(1,4),W(1,6),MGVX3,AMP(1))
CALL JVVXXX(W(1,1),W(1,2),MGVX5,MZ,WZ,W(1,8))
# Amplitude(s) for diagram number 2
CALL VVVXXX(W(1,4),W(1,7),W(1,8),MGVX5,AMP(2))
CALL JVVXXX(W(1,4),W(1,5),MGVX3,MW,WW,W(1,9))
# Amplitude(s) for diagram number 3
CALL VVVXXX(W(1,3),W(1,9),W(1,6),MGVX3,AMP(3))
# Amplitude(s) for diagram number 4
CALL VVVXXX(W(1,9),W(1,3),W(1,8),MGVX5,AMP(4))
# Amplitude(s) for diagram number 5
CALL W3W3NX(W(1,3),W(1,5),W(1,4),W(1,6),MGVX4,DUM0,AMP(5))
# Amplitude(s) for diagram number 6
CALL W3W3NX(W(1,3),W(1,5),W(1,4),W(1,8),MGVX7,DUM0,AMP(6))
CALL JVVXXX(W(1,3),W(1,1),MGVX3,zero,zero,W(1,10))
CALL JVVXXX(W(1,5),W(1,2),MGVX3,MW,WW,W(1,11))
# Amplitude(s) for diagram number 7
CALL VVVXXX(W(1,11),W(1,4),W(1,10),MGVX3,AMP(7))
CALL JVVXXX(W(1,1),W(1,3),MGVX5,MZ,WZ,W(1,12))
# Amplitude(s) for diagram number 8
CALL VVVXXX(W(1,4),W(1,11),W(1,12),MGVX5,AMP(8))
# Amplitude(s) for diagram number 9
CALL VVVXXX(W(1,2),W(1,9),W(1,10),MGVX3,AMP(9))
# Amplitude(s) for diagram number 10
CALL VVVXXX(W(1,9),W(1,2),W(1,12),MGVX5,AMP(10))
# Amplitude(s) for diagram number 11
CALL W3W3NX(W(1,2),W(1,5),W(1,4),W(1,10),MGVX4,DUM0,AMP(11))
# Amplitude(s) for diagram number 12
CALL W3W3NX(W(1,2),W(1,5),W(1,4),W(1,12),MGVX7,DUM0,AMP(12))
CALL JVVXXX(W(1,1),W(1,5),MGVX3,MW,WW,W(1,13))
CALL JVVXXX(W(1,2),W(1,4),MGVX3,zero,zero,W(1,14))
# Amplitude(s) for diagram number 13
CALL VVVXXX(W(1,3),W(1,13),W(1,14),MGVX3,AMP(13))
CALL JVVXXX(W(1,4),W(1,2),MGVX5,MZ,WZ,W(1,15))
# Amplitude(s) for diagram number 14
CALL VVVXXX(W(1,13),W(1,3),W(1,15),MGVX5,AMP(14))
CALL JVVXXX(W(1,3),W(1,4),MGVX3,zero,zero,W(1,16))
# Amplitude(s) for diagram number 15
CALL VVVXXX(W(1,2),W(1,13),W(1,16),MGVX3,AMP(15))
CALL JVVXXX(W(1,4),W(1,3),MGVX5,MZ,WZ,W(1,17))
# Amplitude(s) for diagram number 16
CALL VVVXXX(W(1,13),W(1,2),W(1,17),MGVX5,AMP(16))
# Amplitude(s) for diagram number 17
CALL W3W3NX(W(1,2),W(1,4),W(1,3),W(1,13),MGVX6,DUM0,AMP(17))
# Amplitude(s) for diagram number 18
CALL VVVXXX(W(1,7),W(1,1),W(1,14),MGVX3,AMP(18))
# Amplitude(s) for diagram number 19
CALL VVVXXX(W(1,1),W(1,7),W(1,15),MGVX5,AMP(19))
# Amplitude(s) for diagram number 20
CALL VVVXXX(W(1,11),W(1,1),W(1,16),MGVX3,AMP(20))
# Amplitude(s) for diagram number 21
CALL VVVXXX(W(1,1),W(1,11),W(1,17),MGVX5,AMP(21))
CALL JW3WNX(W(1,3),W(1,1),W(1,2),MGVX6,DUM0,MW,WW,W(1,18))
# Amplitude(s) for diagram number 22
CALL VVVXXX(W(1,18),W(1,4),W(1,5),MGVX3,AMP(22))
CALL JW3WNX(W(1,1),W(1,2),W(1,4),MGVX6,DUM0,MW,WW,W(1,19))
# Amplitude(s) for diagram number 23
CALL VVVXXX(W(1,3),W(1,19),W(1,5),MGVX3,AMP(23))
CALL JW3WNX(W(1,1),W(1,5),W(1,2),MGVX4,DUM0,zero,zero,W(1,20))
# Amplitude(s) for diagram number 24
CALL VVVXXX(W(1,3),W(1,4),W(1,20),MGVX3,AMP(24))
CALL JW3WNX(W(1,2),W(1,5),W(1,1),MGVX7,DUM0,MZ,WZ,W(1,21))
# Amplitude(s) for diagram number 25
CALL VVVXXX(W(1,4),W(1,3),W(1,21),MGVX5,AMP(25))
CALL JW3WNX(W(1,1),W(1,3),W(1,4),MGVX6,DUM0,MW,WW,W(1,22))
# Amplitude(s) for diagram number 26
CALL VVVXXX(W(1,2),W(1,22),W(1,5),MGVX3,AMP(26))
CALL JW3WNX(W(1,1),W(1,5),W(1,3),MGVX4,DUM0,zero,zero,W(1,23))
# Amplitude(s) for diagram number 27
CALL VVVXXX(W(1,2),W(1,4),W(1,23),MGVX3,AMP(27))
CALL JW3WNX(W(1,3),W(1,5),W(1,1),MGVX7,DUM0,MZ,WZ,W(1,24))
# Amplitude(s) for diagram number 28
CALL VVVXXX(W(1,4),W(1,2),W(1,24),MGVX5,AMP(28))""")

    def test_multiple_lorentz_structures(self):
        """Testing multiple Lorentz structures for one diagram.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # An A particle
        mypartlist.append(base_objects.Particle({'name':'A',
                      'antiname':'A',
                      'spin':3,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'A',
                      'antitexname':'A',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':45,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        A = mypartlist[len(mypartlist) - 1]

        # A particle self-couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [A, \
                                             A, \
                                             A]),
                      'color': [],
                      'lorentz':['L1', 'L2'],
                      'couplings':{(0, 0):'G1', (0, 1):'G2'},
                      'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':45,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':45,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':45,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':45,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)
        self.assertEqual("\n".join(myfortranmodel.\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
CALL JVVL1X(W(1,1),W(1,2),G1,zero,zero,W(1,5))
CALL JVVL2X(W(1,1),W(1,2),G2,zero,zero,W(1,6))
# Amplitude(s) for diagram number 1
CALL VVVL1X(W(1,3),W(1,4),W(1,5),G1,AMP(1))
CALL VVVL2X(W(1,3),W(1,4),W(1,5),G2,AMP(2))
CALL VVVL1X(W(1,3),W(1,4),W(1,6),G1,AMP(3))
CALL VVVL2X(W(1,3),W(1,4),W(1,6),G2,AMP(4))
CALL JVVL1X(W(1,1),W(1,3),G1,zero,zero,W(1,7))
CALL JVVL2X(W(1,1),W(1,3),G2,zero,zero,W(1,8))
# Amplitude(s) for diagram number 2
CALL VVVL1X(W(1,2),W(1,4),W(1,7),G1,AMP(5))
CALL VVVL2X(W(1,2),W(1,4),W(1,7),G2,AMP(6))
CALL VVVL1X(W(1,2),W(1,4),W(1,8),G1,AMP(7))
CALL VVVL2X(W(1,2),W(1,4),W(1,8),G2,AMP(8))
CALL JVVL1X(W(1,1),W(1,4),G1,zero,zero,W(1,9))
CALL JVVL2X(W(1,1),W(1,4),G2,zero,zero,W(1,10))
# Amplitude(s) for diagram number 3
CALL VVVL1X(W(1,2),W(1,3),W(1,9),G1,AMP(9))
CALL VVVL2X(W(1,2),W(1,3),W(1,9),G2,AMP(10))
CALL VVVL1X(W(1,2),W(1,3),W(1,10),G1,AMP(11))
CALL VVVL2X(W(1,2),W(1,3),W(1,10),G2,AMP(12))""")

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))+AMP(2)*dconjg(AMP(2))+AMP(3)*dconjg(AMP(3))+AMP(4)*dconjg(AMP(4))',
                          'AMP2(2)=AMP2(2)+AMP(5)*dconjg(AMP(5))+AMP(6)*dconjg(AMP(6))+AMP(7)*dconjg(AMP(7))+AMP(8)*dconjg(AMP(8))',
                          'AMP2(3)=AMP2(3)+AMP(9)*dconjg(AMP(9))+AMP(10)*dconjg(AMP(10))+AMP(11)*dconjg(AMP(11))+AMP(12)*dconjg(AMP(12))'])

        # Test configs file
        writer = writers.FortranWriter(self.give_pos('test'))
        nconfig, s_and_t_channels = exporter.write_configs_file(writer,
                                     matrix_element)
        writer.close()

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA SPROP(-1,1)/45/
      DATA TPRID(-1,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/1,3/
      DATA TPRID(-1,2)/45/
      DATA SPROP(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,4/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/1,4/
      DATA TPRID(-1,3)/45/
      DATA SPROP(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,3/
C     Number of configs
      DATA MAPCONFIG(0)/3/
""")

    def test_multiple_lorentz_structures_with_fermion_flow_clash(self):
        """Testing process w+ w+ > z x1+ x1+.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # Z particle
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
                      'color':1,
                      'mass':'MZ',
                      'width':'WZ',
                      'texname':'Z',
                      'antitexname':'Z',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':23,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        z = mypartlist[len(mypartlist) - 1]

        # W particle
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w-',
                      'spin':3,
                      'color':1,
                      'mass':'MW',
                      'width':'WW',
                      'texname':'W',
                      'antitexname':'W',
                      'line':'curly',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # n1 particle
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'Mneu1',
                      'width':'Wneu1',
                      'texname':'n1',
                      'antitexname':'n1',
                      'line':'curly',
                      'charge':0.,
                      'pdg_code':1000023,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))

        n1 = mypartlist[len(mypartlist) - 1]

        # x1+ particle
        mypartlist.append(base_objects.Particle({'name':'x1+',
                      'antiname':'x1-',
                      'spin':2,
                      'color':1,
                      'mass':'Mx1p',
                      'width':'Wx1p',
                      'texname':'x1+',
                      'antitexname':'x1-',
                      'line':'curly',
                      'charge':1.,
                      'pdg_code':1000024,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        x1plus = mypartlist[len(mypartlist) - 1]
        x1minus = copy.copy(x1plus)
        x1minus.set('is_part', False)

        # Interactions

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([wminus,wplus,z]),
            'color': [],
            'lorentz': ['VVV1'],
            'couplings': {(0, 0): 'GC_214'},
            'orders': {'QED': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([n1,x1plus,wminus]),
            'color': [],
            'lorentz': ['FFV2', 'FFV3'],
            'couplings': {(0, 1): 'GC_628', (0, 0): 'GC_422'},
            'orders': {'QED': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 3,
            'particles': base_objects.ParticleList([n1,x1minus,wplus]),
            'color': [],
            'lorentz': ['FFV2', 'FFV3'],
            'couplings': {(0, 1): 'GC_628', (0, 0): 'GC_422'},
            'orders': {'QED': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList([n1,n1,z]),
            'color': [],
            'lorentz': ['FFV5'],
            'couplings': {(0, 0): 'GC_418'},
            'orders': {'QED': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000024,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000024,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 6)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, gen_color=False)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MZ,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),Mx1p,NHEL(4),-1*IC(4),W(1,4))
CALL OXXXXX(P(0,5),Mx1p,NHEL(5),+1*IC(5),W(1,5))
CALL VVV1_2(W(1,1),W(1,3),GC_214,MW, WW, W(1,6))
CALL FFV2C1_2(W(1,4),W(1,2),GC_422,Mneu1, Wneu1, W(1,7))
CALL FFV3C1_2(W(1,4),W(1,2),GC_628,Mneu1, Wneu1, W(1,8))
# Amplitude(s) for diagram number 1
CALL FFV2_0(W(1,7),W(1,5),W(1,6),GC_422,AMP(1))
CALL FFV3_0(W(1,7),W(1,5),W(1,6),GC_628,AMP(2))
CALL FFV2_0(W(1,8),W(1,5),W(1,6),GC_422,AMP(3))
CALL FFV3_0(W(1,8),W(1,5),W(1,6),GC_628,AMP(4))
CALL FFV2_1(W(1,5),W(1,2),GC_422,Mneu1, Wneu1, W(1,9))
CALL FFV3_1(W(1,5),W(1,2),GC_628,Mneu1, Wneu1, W(1,10))
# Amplitude(s) for diagram number 2
CALL FFV2C1_0(W(1,4),W(1,9),W(1,6),GC_422,AMP(5))
CALL FFV3C1_0(W(1,4),W(1,9),W(1,6),GC_628,AMP(6))
CALL FFV2C1_0(W(1,4),W(1,10),W(1,6),GC_422,AMP(7))
CALL FFV3C1_0(W(1,4),W(1,10),W(1,6),GC_628,AMP(8))
CALL FFV2C1_2(W(1,4),W(1,1),GC_422,Mneu1, Wneu1, W(1,11))
CALL FFV3C1_2(W(1,4),W(1,1),GC_628,Mneu1, Wneu1, W(1,12))
CALL VVV1_2(W(1,2),W(1,3),GC_214,MW, WW, W(1,13))
# Amplitude(s) for diagram number 3
CALL FFV2_0(W(1,11),W(1,5),W(1,13),GC_422,AMP(9))
CALL FFV3_0(W(1,11),W(1,5),W(1,13),GC_628,AMP(10))
CALL FFV2_0(W(1,12),W(1,5),W(1,13),GC_422,AMP(11))
CALL FFV3_0(W(1,12),W(1,5),W(1,13),GC_628,AMP(12))
# Amplitude(s) for diagram number 4
CALL FFV5_0(W(1,11),W(1,9),W(1,3),GC_418,AMP(13))
CALL FFV5_0(W(1,11),W(1,10),W(1,3),GC_418,AMP(14))
CALL FFV5_0(W(1,12),W(1,9),W(1,3),GC_418,AMP(15))
CALL FFV5_0(W(1,12),W(1,10),W(1,3),GC_418,AMP(16))
CALL FFV2_1(W(1,5),W(1,1),GC_422,Mneu1, Wneu1, W(1,14))
CALL FFV3_1(W(1,5),W(1,1),GC_628,Mneu1, Wneu1, W(1,15))
# Amplitude(s) for diagram number 5
CALL FFV2C1_0(W(1,4),W(1,14),W(1,13),GC_422,AMP(17))
CALL FFV3C1_0(W(1,4),W(1,14),W(1,13),GC_628,AMP(18))
CALL FFV2C1_0(W(1,4),W(1,15),W(1,13),GC_422,AMP(19))
CALL FFV3C1_0(W(1,4),W(1,15),W(1,13),GC_628,AMP(20))
# Amplitude(s) for diagram number 6
CALL FFV5_0(W(1,7),W(1,14),W(1,3),GC_418,AMP(21))
CALL FFV5_0(W(1,8),W(1,14),W(1,3),GC_418,AMP(22))
CALL FFV5_0(W(1,7),W(1,15),W(1,3),GC_418,AMP(23))
CALL FFV5_0(W(1,8),W(1,15),W(1,3),GC_418,AMP(24))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))+AMP(2)*dconjg(AMP(2))+AMP(3)*dconjg(AMP(3))+AMP(4)*dconjg(AMP(4))',
                          'AMP2(2)=AMP2(2)+AMP(5)*dconjg(AMP(5))+AMP(6)*dconjg(AMP(6))+AMP(7)*dconjg(AMP(7))+AMP(8)*dconjg(AMP(8))',
                          'AMP2(3)=AMP2(3)+AMP(9)*dconjg(AMP(9))+AMP(10)*dconjg(AMP(10))+AMP(11)*dconjg(AMP(11))+AMP(12)*dconjg(AMP(12))',
                          'AMP2(4)=AMP2(4)+AMP(13)*dconjg(AMP(13))+AMP(14)*dconjg(AMP(14))+AMP(15)*dconjg(AMP(15))+AMP(16)*dconjg(AMP(16))',
                          'AMP2(5)=AMP2(5)+AMP(17)*dconjg(AMP(17))+AMP(18)*dconjg(AMP(18))+AMP(19)*dconjg(AMP(19))+AMP(20)*dconjg(AMP(20))',
                          'AMP2(6)=AMP2(6)+AMP(21)*dconjg(AMP(21))+AMP(22)*dconjg(AMP(22))+AMP(23)*dconjg(AMP(23))+AMP(24)*dconjg(AMP(24))'])

    def test_four_fermion_vertex_normal_fermion_flow(self):
        """Testing process u u > t t g with fermion flow (u~t)(u~t)
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A u quark and its antiparticle
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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A t quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'t',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

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

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # Four fermion vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antiu,
                                             t,
                                             antiu,
                                             t]),
                      'color': [color.ColorString([color.T(1, 0),
                                                   color.T(3, 2)])],
                      'lorentz':['FFFF1'],
                      'couplings':{(0, 0):'GEFF'},
                      'orders':{'NP':2}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 4)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),zero,NHEL(2),+1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),MT,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MT,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FFV1_2(W(1,1),W(1,5),GG,zero, zero, W(1,6))
# Amplitude(s) for diagram number 1
CALL FFFF1_0(W(1,2),W(1,3),W(1,6),W(1,4),GEFF,AMP(1))
CALL FFFF1_2(W(1,1),W(1,2),W(1,3),GEFF,MT, WT, W(1,7))
# Amplitude(s) for diagram number 2
CALL FFV1_0(W(1,7),W(1,4),W(1,5),GG,AMP(2))
CALL FFFF1_2(W(1,1),W(1,2),W(1,4),GEFF,MT, WT, W(1,8))
# Amplitude(s) for diagram number 3
CALL FFV1_0(W(1,8),W(1,3),W(1,5),GG,AMP(3))
CALL FFFF1_1(W(1,3),W(1,1),W(1,4),GEFF,zero, zero, W(1,9))
# Amplitude(s) for diagram number 4
CALL FFV1_0(W(1,2),W(1,9),W(1,5),GG,AMP(4))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))',
                          'AMP2(4)=AMP2(4)+AMP(4)*dconjg(AMP(4))'])

        # Check fermion factors
        self.assertEqual([d.get('amplitudes')[0].get('fermionfactor') \
                          for d in matrix_element.get('diagrams')],
                         [1, 1, -1, 1])
        
        writer = writers.FortranWriter(self.give_pos('test'))
        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)

        # Test configs file
        nconfig, s_and_t_channels = exporter.write_configs_file(writer,
                                     matrix_element)
        writer.close()

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/4,3/
      DATA SPROP(-1,1)/21/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/1,5/
      DATA TPRID(-2,1)/2/
      DATA SPROP(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/-2,-1/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/5,4/
      DATA SPROP(-1,2)/6/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,3/
      DATA SPROP(-2,2)/21/
      DATA TPRID(-2,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/5,3/
      DATA SPROP(-1,3)/6/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/4,-1/
      DATA SPROP(-2,3)/21/
      DATA TPRID(-2,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/4,3/
      DATA SPROP(-1,4)/21/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/1,-1/
      DATA TPRID(-2,4)/2/
      DATA SPROP(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/-2,5/
C     Number of configs
      DATA MAPCONFIG(0)/4/
""")

        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_props_file(writer, matrix_element,
                                   s_and_t_channels)
        writer.close()
        #print open(self.give_pos('test')).read()
        self.assertFileContains('test',
"""      PMASS(-1,1)  = ZERO
      PWIDTH(-1,1) = ZERO
      POW(-1,1) = 2
      PMASS(-2,1)  = ZERO
      PWIDTH(-2,1) = ZERO
      POW(-2,1) = 1
      PMASS(-1,2)  = ABS(MT)
      PWIDTH(-1,2) = ABS(WT)
      POW(-1,2) = 1
      PMASS(-2,2)  = ZERO
      PWIDTH(-2,2) = ZERO
      POW(-2,2) = 2
      PMASS(-1,3)  = ABS(MT)
      PWIDTH(-1,3) = ABS(WT)
      POW(-1,3) = 1
      PMASS(-2,3)  = ZERO
      PWIDTH(-2,3) = ZERO
      POW(-2,3) = 2
      PMASS(-1,4)  = ZERO
      PWIDTH(-1,4) = ZERO
      POW(-1,4) = 2
      PMASS(-2,4)  = ZERO
      PWIDTH(-2,4) = ZERO
      POW(-2,4) = 1
""")


    def test_four_fermion_vertex_strange_fermion_flow(self):
        """Testing process u u > t t g with fermion flow (u~u~)(tt)
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A u quark and its antiparticle
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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A t quark and its antiparticle
        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'t',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

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

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':['FFV1'],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        # Four fermion vertex
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antiu,
                                             antiu,
                                             t,
                                             t]),
                      'color': [color.ColorString([color.K6(-1, 1, 0),
                                                   color.K6Bar(-1,3, 2)])],
                      'lorentz':['FFFF1'],
                      'couplings':{(0, 0):'GEFF'},
                      'orders':{'NP':2}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 4)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MT,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MT,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FFV1_2(W(1,1),W(1,5),GG,zero, zero, W(1,6))
# Amplitude(s) for diagram number 1
CALL FFFF1C1C2_0(W(1,6),W(1,2),W(1,3),W(1,4),GEFF,AMP(1))
CALL FFFF1C1C2_4(W(1,1),W(1,2),W(1,3),GEFF,MT, WT, W(1,7))
# Amplitude(s) for diagram number 2
CALL FFV1_0(W(1,7),W(1,4),W(1,5),GG,AMP(2))
CALL FFFF1C1C2_3(W(1,1),W(1,2),W(1,4),GEFF,MT, WT, W(1,8))
# Amplitude(s) for diagram number 3
CALL FFV1C1_0(W(1,3),W(1,8),W(1,5),GG,AMP(3))
CALL FFFF1C1C2_2(W(1,1),W(1,3),W(1,4),GEFF,zero, zero, W(1,9))
# Amplitude(s) for diagram number 4
CALL FFV1C1_0(W(1,9),W(1,2),W(1,5),GG,AMP(4))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

        exporter = export_v4.ProcessExporterFortranME()

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(matrix_element)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))',
                          'AMP2(4)=AMP2(4)+AMP(4)*dconjg(AMP(4))'])

        # Check fermion factors
        self.assertEqual([d.get('amplitudes')[0].get('fermionfactor') \
                          for d in matrix_element.get('diagrams')],
                         [1, 1, 1, 1])
        
    def test_multiple_lorentz_structures_with_decay_chain(self):
        """Testing process b~ t > wp+, wp+ > b~ t
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # bt sextet particle
        mypartlist.append(base_objects.Particle({'name':'wp+',
                      'antiname':'wp-',
                      'spin':3,
                      'color':1,
                      'mass':'Mwp',
                      'width':'Wwp',
                      'texname':'wp+',
                      'antitexname':'wp-',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':9000006,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        wp = mypartlist[len(mypartlist) - 1]
        wpbar = copy.copy(wp)
        wpbar.set('is_part', False)

        # b and t quarks
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

        # Interactions

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([t, antib, wpbar]),
            'color': [color.ColorString([color.T(1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([b, antit, wp]),
            'color': [color.ColorString([color.T(1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':-5,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':True}))
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-5}))
        myleglist.append(base_objects.Leg({'id':6}))
        mydecay = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myproc.set('decay_chains', base_objects.ProcessList([\
            mydecay]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('core_processes')[0])
        self.assertEqual("\n".join(result),
                         """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),Mwp,NHEL(3),+1*IC(3),W(1,3))
# Amplitude(s) for diagram number 1
CALL FFS3_0(W(1,2),W(1,1),W(1,3),GC_108,AMP(1))
CALL FFS4_0(W(1,2),W(1,1),W(1,3),GC_111,AMP(2))""")
        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('decay_chains')[0].get('core_processes')[0])
        self.assertEqual("\n".join(result),
                         """CALL VXXXXX(P(0,1),Mwp,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),MT,NHEL(3),+1*IC(3),W(1,3))
CALL FFS3_3(W(1,2),W(1,3),GC_108,Mwp, Wwp, W(1,4))
CALL FFS4_3(W(1,2),W(1,3),GC_111,Mwp, Wwp, W(1,5))
# Amplitude(s) for diagram number 1
#
#""")

        matrix_elements = matrix_element.combine_decay_chain_processes()

        matrix_element = matrix_elements[0]

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MT,NHEL(4),+1*IC(4),W(1,4))
CALL FFS3_3(W(1,3),W(1,4),GC_108,Mwp, Wwp, W(1,5))
CALL FFS4_3(W(1,3),W(1,4),GC_111,Mwp, Wwp, W(1,6))
# Amplitude(s) for diagram number 1
CALL FFS3_0(W(1,2),W(1,1),W(1,5),GC_108,AMP(1))
CALL FFS3_0(W(1,2),W(1,1),W(1,6),GC_108,AMP(2))
CALL FFS4_0(W(1,2),W(1,1),W(1,5),GC_111,AMP(3))
CALL FFS4_0(W(1,2),W(1,1),W(1,6),GC_111,AMP(4))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

    def test_multiple_lorentz_structures_with_decay_chain_and_fermion_flow(self):
        """Testing process b t > six1, six1 > b t
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # bt sextet particle
        mypartlist.append(base_objects.Particle({'name':'six1',
                      'antiname':'six1~',
                      'spin':1,
                      'color':6,
                      'mass':'Msix1',
                      'width':'Wsix1',
                      'texname':'six1',
                      'antitexname':'six1bar',
                      'line':'dashed',
                      'charge':1./3.,
                      'pdg_code':9000006,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))

        six1 = mypartlist[len(mypartlist) - 1]
        six1bar = copy.copy(six1)
        six1bar.set('is_part', False)

        # b and t quarks
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

        # Interactions

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([t, b, six1bar]),
            'color': [color.ColorString([color.K6Bar(2,1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([antib, antit, six1]),
            'color': [color.ColorString([color.K6(2,1,0)])],
            'lorentz': ['FFS3', 'FFS4'],
            'couplings': {(0, 0): 'GC_108', (0, 1): 'GC_111'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':5,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':True}))
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myleglist = base_objects.LegList()
        myleglist.append(base_objects.Leg({'id':9000006,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':5}))
        myleglist.append(base_objects.Leg({'id':6}))
        mydecay = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})
        myproc.set('decay_chains', base_objects.ProcessList([\
            mydecay]))

        myamplitude = diagram_generation.DecayChainAmplitude(myproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('core_processes')[0])
        self.assertEqual("\n".join(result),
                         """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),+1*IC(3),W(1,3))
# Amplitude(s) for diagram number 1
CALL FFS3C1_0(W(1,2),W(1,1),W(1,3),GC_108,AMP(1))
CALL FFS4C1_0(W(1,2),W(1,1),W(1,3),GC_111,AMP(2))""")
        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element.get('decay_chains')[0].get('core_processes')[0])
        self.assertEqual("\n".join(result),
                         """CALL SXXXXX(P(0,1),-1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),+1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MT,NHEL(3),-1*IC(3),W(1,3))
CALL FFS3C1_3(W(1,3),W(1,2),GC_108,Msix1, Wsix1, W(1,4))
CALL FFS4C1_3(W(1,3),W(1,2),GC_111,Msix1, Wsix1, W(1,5))
# Amplitude(s) for diagram number 1
#
#""")

        matrix_elements = matrix_element.combine_decay_chain_processes()

        matrix_element = matrix_elements[0]

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                                   get_matrix_element_calls(matrix_element)

        goal = """CALL OXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),MT,NHEL(2),+1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),MT,NHEL(4),-1*IC(4),W(1,4))
CALL FFS3C1_3(W(1,4),W(1,3),GC_108,Msix1, Wsix1, W(1,5))
CALL FFS4C1_3(W(1,4),W(1,3),GC_111,Msix1, Wsix1, W(1,6))
# Amplitude(s) for diagram number 1
CALL FFS3C1_0(W(1,2),W(1,1),W(1,5),GC_108,AMP(1))
CALL FFS3C1_0(W(1,2),W(1,1),W(1,6),GC_108,AMP(2))
CALL FFS4C1_0(W(1,2),W(1,1),W(1,5),GC_111,AMP(3))
CALL FFS4C1_0(W(1,2),W(1,1),W(1,6),GC_111,AMP(4))""".split('\n')

        for i in range(len(goal)):
            self.assertEqual(result[i], goal[i])

    def test_export_matrix_element_v4_standalone(self):
        """Test the result of exporting a matrix element to file"""

        writer = writers.FortranWriter(self.give_pos('test'))

        goal_matrix_f = \
"""      SUBROUTINE SMATRIX(P,ANS)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     MadGraph StandAlone Version
C     
C     Returns amplitude squared summed/avg over colors
C     and helicities
C     for the point in phase space P(0:3,NEXTERNAL)
C     
C     Process: e+ e- > a a a
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=32)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
C     
C     LOCAL VARIABLES 
C     
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T
      REAL*8 MATRIX
      INTEGER IHEL,IDEN, I
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
      DATA (NHEL(IHEL,   1),IHEL=1,5) /-1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,   2),IHEL=1,5) /-1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,   3),IHEL=1,5) /-1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,   4),IHEL=1,5) /-1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,   5),IHEL=1,5) /-1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,   6),IHEL=1,5) /-1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,   7),IHEL=1,5) /-1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,   8),IHEL=1,5) /-1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,   9),IHEL=1,5) /-1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  10),IHEL=1,5) /-1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  11),IHEL=1,5) /-1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  12),IHEL=1,5) /-1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  13),IHEL=1,5) /-1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  14),IHEL=1,5) /-1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  15),IHEL=1,5) /-1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  16),IHEL=1,5) /-1, 1, 1, 1, 1/
      DATA (NHEL(IHEL,  17),IHEL=1,5) / 1,-1,-1,-1,-1/
      DATA (NHEL(IHEL,  18),IHEL=1,5) / 1,-1,-1,-1, 1/
      DATA (NHEL(IHEL,  19),IHEL=1,5) / 1,-1,-1, 1,-1/
      DATA (NHEL(IHEL,  20),IHEL=1,5) / 1,-1,-1, 1, 1/
      DATA (NHEL(IHEL,  21),IHEL=1,5) / 1,-1, 1,-1,-1/
      DATA (NHEL(IHEL,  22),IHEL=1,5) / 1,-1, 1,-1, 1/
      DATA (NHEL(IHEL,  23),IHEL=1,5) / 1,-1, 1, 1,-1/
      DATA (NHEL(IHEL,  24),IHEL=1,5) / 1,-1, 1, 1, 1/
      DATA (NHEL(IHEL,  25),IHEL=1,5) / 1, 1,-1,-1,-1/
      DATA (NHEL(IHEL,  26),IHEL=1,5) / 1, 1,-1,-1, 1/
      DATA (NHEL(IHEL,  27),IHEL=1,5) / 1, 1,-1, 1,-1/
      DATA (NHEL(IHEL,  28),IHEL=1,5) / 1, 1,-1, 1, 1/
      DATA (NHEL(IHEL,  29),IHEL=1,5) / 1, 1, 1,-1,-1/
      DATA (NHEL(IHEL,  30),IHEL=1,5) / 1, 1, 1,-1, 1/
      DATA (NHEL(IHEL,  31),IHEL=1,5) / 1, 1, 1, 1,-1/
      DATA (NHEL(IHEL,  32),IHEL=1,5) / 1, 1, 1, 1, 1/
      DATA IDEN/24/
C     ----------
C     BEGIN CODE
C     ----------
      NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
        JC(IHEL) = +1
      ENDDO
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
          T=MATRIX(P ,NHEL(1,IHEL),JC(1))
          ANS=ANS+T
          IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)
      END
      
      
      REAL*8 FUNCTION MATRIX(P,NHEL,IC)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     Returns amplitude squared summed/avg over colors
C     for the point with external lines W(0:6,NEXTERNAL)
C     
C     Process: e+ e- > a a a
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=6)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=11, NCOLOR=1)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'coupl.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/1/
      DATA (CF(I,1),I=1,1) /1/
C     ----------
C     BEGIN CODE
C     ----------
      CALL OXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1*IC(2),W(1,2))
      CALL VXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
      CALL VXXXXX(P(0,4),ZERO,NHEL(4),+1*IC(4),W(1,4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1*IC(5),W(1,5))
      CALL FVOXXX(W(1,1),W(1,3),MGVX12,ZERO,ZERO,W(1,6))
      CALL FVIXXX(W(1,2),W(1,4),MGVX12,ZERO,ZERO,W(1,7))
C     Amplitude(s) for diagram number 1
      CALL IOVXXX(W(1,7),W(1,6),W(1,5),MGVX12,AMP(1))
      CALL FVIXXX(W(1,2),W(1,5),MGVX12,ZERO,ZERO,W(1,8))
C     Amplitude(s) for diagram number 2
      CALL IOVXXX(W(1,8),W(1,6),W(1,4),MGVX12,AMP(2))
      CALL FVOXXX(W(1,1),W(1,4),MGVX12,ZERO,ZERO,W(1,9))
      CALL FVIXXX(W(1,2),W(1,3),MGVX12,ZERO,ZERO,W(1,10))
C     Amplitude(s) for diagram number 3
      CALL IOVXXX(W(1,10),W(1,9),W(1,5),MGVX12,AMP(3))
C     Amplitude(s) for diagram number 4
      CALL IOVXXX(W(1,8),W(1,9),W(1,3),MGVX12,AMP(4))
      CALL FVOXXX(W(1,1),W(1,5),MGVX12,ZERO,ZERO,W(1,11))
C     Amplitude(s) for diagram number 5
      CALL IOVXXX(W(1,10),W(1,11),W(1,4),MGVX12,AMP(5))
C     Amplitude(s) for diagram number 6
      CALL IOVXXX(W(1,7),W(1,11),W(1,3),MGVX12,AMP(6))
      JAMP(1)=-AMP(1)-AMP(2)-AMP(3)-AMP(4)-AMP(5)-AMP(6)
      
      MATRIX = 0.D0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
      ENDDO
      END
""" % misc.get_pkg_info()

    def test_matrix_multistage_decay_chain_process(self):
        """Test matrix.f for multistage decay chain
        """

        # Set up local model

        mybasemodel = base_objects.Model()
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()
        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)

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
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A mu and anti-mu
        mypartlist.append(base_objects.Particle({'name':'mu+',
                      'antiname':'mu-',
                      'spin':2,
                      'color':1,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'\mu^+',
                      'antitexname':'\mu^-',
                      'line':'straight',
                      'charge':-1.,
                      'pdg_code':13,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        muminus = mypartlist[len(mypartlist) - 1]
        muplus = copy.copy(muminus)
        muplus.set('is_part', False)

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

        # Coupling of e to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [eminus, \
                                             eplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GAL'},
                      'orders':{'QED':1}}))

        # Coupling of mu to gamma
        myinterlist.append(base_objects.Interaction({
                      'id': 16,
                      'particles': base_objects.ParticleList(\
                                            [muminus, \
                                             muplus, \
                                             a]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GAL'},
                      'orders':{'QED':1}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)


        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                           'model':mybasemodel})

        me_core = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mycoreproc))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay11 = base_objects.Process({'legs':myleglist,
                                          'model':mybasemodel})

        me11 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay11))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay12 = base_objects.Process({'legs':myleglist,
                                          'model':mybasemodel})

        me12 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay12))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':22,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':13,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-13,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':True}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                         'model':mybasemodel})

        me2 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay2))

        mydecay11.set('decay_chains', base_objects.ProcessList([mydecay2]))
        mydecay12.set('decay_chains', base_objects.ProcessList([mydecay2]))

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay11, mydecay12]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_elements = helas_objects.HelasDecayChainProcess(myamplitude).\
                          combine_decay_chain_processes()

        me = matrix_elements[0]

        # Check all ingredients in file here

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(me.get_nexternal_ninitial(), (10, 2))
        self.assertEqual(me.get_helicity_combinations(), 1024)
        self.assertEqual(len(exporter.get_helicity_lines(me).split("\n")), 1024)
        # This has been tested against v4
        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,5))
CALL VXXXXX(P(0,6),zero,NHEL(6),+1*IC(6),W(1,6))
CALL FVOXXX(W(1,4),W(1,6),GAL,zero,zero,W(1,7))
CALL JIOXXX(W(1,5),W(1,7),GAL,zero,zero,W(1,8))
CALL FVOXXX(W(1,3),W(1,8),GAL,zero,zero,W(1,9))
CALL IXXXXX(P(0,7),zero,NHEL(7),-1*IC(7),W(1,10))
CALL OXXXXX(P(0,8),zero,NHEL(8),+1*IC(8),W(1,11))
CALL IXXXXX(P(0,9),zero,NHEL(9),-1*IC(9),W(1,12))
CALL VXXXXX(P(0,10),zero,NHEL(10),+1*IC(10),W(1,13))
CALL FVOXXX(W(1,11),W(1,13),GAL,zero,zero,W(1,14))
CALL JIOXXX(W(1,12),W(1,14),GAL,zero,zero,W(1,15))
CALL FVIXXX(W(1,10),W(1,15),GAL,zero,zero,W(1,16))
CALL FVOXXX(W(1,9),W(1,1),GAL,zero,zero,W(1,17))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,16),W(1,17),W(1,2),GAL,AMP(1))
CALL FVIXXX(W(1,12),W(1,13),GAL,zero,zero,W(1,18))
CALL JIOXXX(W(1,18),W(1,11),GAL,zero,zero,W(1,19))
CALL FVIXXX(W(1,10),W(1,19),GAL,zero,zero,W(1,20))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,20),W(1,17),W(1,2),GAL,AMP(2))
CALL FVIXXX(W(1,5),W(1,6),GAL,zero,zero,W(1,21))
CALL JIOXXX(W(1,21),W(1,4),GAL,zero,zero,W(1,22))
CALL FVOXXX(W(1,3),W(1,22),GAL,zero,zero,W(1,23))
CALL FVOXXX(W(1,23),W(1,1),GAL,zero,zero,W(1,24))
# Amplitude(s) for diagram number 3
CALL IOVXXX(W(1,16),W(1,24),W(1,2),GAL,AMP(3))
# Amplitude(s) for diagram number 4
CALL IOVXXX(W(1,20),W(1,24),W(1,2),GAL,AMP(4))
CALL FVIXXX(W(1,16),W(1,1),GAL,zero,zero,W(1,25))
# Amplitude(s) for diagram number 5
CALL IOVXXX(W(1,25),W(1,9),W(1,2),GAL,AMP(5))
CALL FVIXXX(W(1,20),W(1,1),GAL,zero,zero,W(1,26))
# Amplitude(s) for diagram number 6
CALL IOVXXX(W(1,26),W(1,9),W(1,2),GAL,AMP(6))
# Amplitude(s) for diagram number 7
CALL IOVXXX(W(1,25),W(1,23),W(1,2),GAL,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOVXXX(W(1,26),W(1,23),W(1,2),GAL,AMP(8))""")

        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_pmass_file(writer, me)
        writer.close()
        self.assertFileContains('test',"""      PMASS(1)=ZERO
      PMASS(2)=ZERO
      PMASS(3)=ZERO
      PMASS(4)=ZERO
      PMASS(5)=ZERO
      PMASS(6)=ZERO
      PMASS(7)=ZERO
      PMASS(8)=ZERO
      PMASS(9)=ZERO
      PMASS(10)=ZERO\n""")


    def test_matrix_4g_decay_chain_process(self):
        """Test matrix.f for multistage decay chain
        """

        # Set up local model

        mybasemodel = base_objects.Model()
        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()
        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mybasemodel)

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
                                color.ColorString([color.f(0, 1, 2)]),
                                color.ColorString([color.f(0, 1, 2)])],
                      'lorentz':['gggg1', 'gggg2', 'gggg3'],
                      'couplings':{(0, 0):'GG', (1, 1):'GG', (2, 2):'GG'},
                      'orders':{'QCD':2}}))

        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)


        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                           'model':mybasemodel})

        me_core = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mycoreproc))

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                          'model':mybasemodel})

        me1 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay1))

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_elements = helas_objects.HelasDecayChainProcess(myamplitude).\
                          combine_decay_chain_processes()

        me = matrix_elements[0]

        # Check all ingredients in file here

        exporter = export_v4.ProcessExporterFortranME()

        #exporter.generate_subprocess_directory_v4_standalone(me,
        #                                                      myfortranmodel)

        goal = """16 82 [0, 0, 0]
16 83 [0, 1, 0]
16 84 [0, 2, 0]
16 85 [1, 0, 0]
16 86 [1, 1, 0]
16 87 [1, 2, 0]
16 88 [2, 0, 0]
16 89 [2, 1, 0]
16 90 [2, 2, 0]
16 91 [0, 0, 1]
16 92 [0, 1, 1]
16 93 [0, 2, 1]
16 94 [1, 0, 1]
16 95 [1, 1, 1]
16 96 [1, 2, 1]
16 97 [2, 0, 1]
16 98 [2, 1, 1]
16 99 [2, 2, 1]
16 100 [0, 0, 2]
16 101 [0, 1, 2]
16 102 [0, 2, 2]
16 103 [1, 0, 2]
16 104 [1, 1, 2]
16 105 [1, 2, 2]
16 106 [2, 0, 2]
16 107 [2, 1, 2]
16 108 [2, 2, 2]""".split("\n")

        diagram = me.get('diagrams')[15]

        for i, amp in enumerate(diagram.get('amplitudes')):
            if diagram.get('number') == 16:
                self.assertEqual("%d %d %s" % \
                                 (diagram.get('number'), amp.get('number'), \
                                  repr(amp.get('color_indices'))),
                                 goal[i])

        self.assertEqual(me.get_nexternal_ninitial(), (8, 2))
        self.assertEqual(me.get_helicity_combinations(), 256)
        self.assertEqual(len(exporter.get_helicity_lines(me).split("\n")), 256)
        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
                         """CALL VXXXXX(P(0,1),zero,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL JVVXXX(W(1,3),W(1,4),GG,zero,zero,W(1,6))
CALL JVVXXX(W(1,6),W(1,5),GG,zero,zero,W(1,7))
CALL VXXXXX(P(0,6),zero,NHEL(6),+1*IC(6),W(1,8))
CALL VXXXXX(P(0,7),zero,NHEL(7),+1*IC(7),W(1,9))
CALL VXXXXX(P(0,8),zero,NHEL(8),+1*IC(8),W(1,10))
CALL JVVXXX(W(1,8),W(1,9),GG,zero,zero,W(1,11))
CALL JVVXXX(W(1,11),W(1,10),GG,zero,zero,W(1,12))
# Amplitude(s) for diagram number 1
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,12),GG,AMP(1))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,12),GG,AMP(2))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,12),GG,AMP(3))
CALL JVVXXX(W(1,8),W(1,10),GG,zero,zero,W(1,13))
CALL JVVXXX(W(1,13),W(1,9),GG,zero,zero,W(1,14))
# Amplitude(s) for diagram number 2
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,14),GG,AMP(4))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,14),GG,AMP(5))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,14),GG,AMP(6))
CALL JVVXXX(W(1,9),W(1,10),GG,zero,zero,W(1,15))
CALL JVVXXX(W(1,8),W(1,15),GG,zero,zero,W(1,16))
# Amplitude(s) for diagram number 3
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,16),GG,AMP(7))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,16),GG,AMP(8))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,16),GG,AMP(9))
CALL JGGGXX(W(1,10),W(1,9),W(1,8),GG,W(1,17))
CALL JGGGXX(W(1,8),W(1,10),W(1,9),GG,W(1,18))
CALL JGGGXX(W(1,9),W(1,8),W(1,10),GG,W(1,19))
# Amplitude(s) for diagram number 4
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,17),GG,AMP(10))
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,18),GG,AMP(11))
CALL GGGGXX(W(1,1),W(1,2),W(1,7),W(1,19),GG,AMP(12))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,17),GG,AMP(13))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,18),GG,AMP(14))
CALL GGGGXX(W(1,7),W(1,1),W(1,2),W(1,19),GG,AMP(15))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,17),GG,AMP(16))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,18),GG,AMP(17))
CALL GGGGXX(W(1,2),W(1,7),W(1,1),W(1,19),GG,AMP(18))
CALL JVVXXX(W(1,3),W(1,5),GG,zero,zero,W(1,20))
CALL JVVXXX(W(1,20),W(1,4),GG,zero,zero,W(1,21))
# Amplitude(s) for diagram number 5
CALL GGGGXX(W(1,1),W(1,2),W(1,21),W(1,12),GG,AMP(19))
CALL GGGGXX(W(1,21),W(1,1),W(1,2),W(1,12),GG,AMP(20))
CALL GGGGXX(W(1,2),W(1,21),W(1,1),W(1,12),GG,AMP(21))
# Amplitude(s) for diagram number 6
CALL GGGGXX(W(1,1),W(1,2),W(1,21),W(1,14),GG,AMP(22))
CALL GGGGXX(W(1,21),W(1,1),W(1,2),W(1,14),GG,AMP(23))
CALL GGGGXX(W(1,2),W(1,21),W(1,1),W(1,14),GG,AMP(24))
# Amplitude(s) for diagram number 7
CALL GGGGXX(W(1,1),W(1,2),W(1,21),W(1,16),GG,AMP(25))
CALL GGGGXX(W(1,21),W(1,1),W(1,2),W(1,16),GG,AMP(26))
CALL GGGGXX(W(1,2),W(1,21),W(1,1),W(1,16),GG,AMP(27))
# Amplitude(s) for diagram number 8
CALL GGGGXX(W(1,1),W(1,2),W(1,21),W(1,17),GG,AMP(28))
CALL GGGGXX(W(1,1),W(1,2),W(1,21),W(1,18),GG,AMP(29))
CALL GGGGXX(W(1,1),W(1,2),W(1,21),W(1,19),GG,AMP(30))
CALL GGGGXX(W(1,21),W(1,1),W(1,2),W(1,17),GG,AMP(31))
CALL GGGGXX(W(1,21),W(1,1),W(1,2),W(1,18),GG,AMP(32))
CALL GGGGXX(W(1,21),W(1,1),W(1,2),W(1,19),GG,AMP(33))
CALL GGGGXX(W(1,2),W(1,21),W(1,1),W(1,17),GG,AMP(34))
CALL GGGGXX(W(1,2),W(1,21),W(1,1),W(1,18),GG,AMP(35))
CALL GGGGXX(W(1,2),W(1,21),W(1,1),W(1,19),GG,AMP(36))
CALL JVVXXX(W(1,4),W(1,5),GG,zero,zero,W(1,22))
CALL JVVXXX(W(1,3),W(1,22),GG,zero,zero,W(1,23))
# Amplitude(s) for diagram number 9
CALL GGGGXX(W(1,1),W(1,2),W(1,23),W(1,12),GG,AMP(37))
CALL GGGGXX(W(1,23),W(1,1),W(1,2),W(1,12),GG,AMP(38))
CALL GGGGXX(W(1,2),W(1,23),W(1,1),W(1,12),GG,AMP(39))
# Amplitude(s) for diagram number 10
CALL GGGGXX(W(1,1),W(1,2),W(1,23),W(1,14),GG,AMP(40))
CALL GGGGXX(W(1,23),W(1,1),W(1,2),W(1,14),GG,AMP(41))
CALL GGGGXX(W(1,2),W(1,23),W(1,1),W(1,14),GG,AMP(42))
# Amplitude(s) for diagram number 11
CALL GGGGXX(W(1,1),W(1,2),W(1,23),W(1,16),GG,AMP(43))
CALL GGGGXX(W(1,23),W(1,1),W(1,2),W(1,16),GG,AMP(44))
CALL GGGGXX(W(1,2),W(1,23),W(1,1),W(1,16),GG,AMP(45))
# Amplitude(s) for diagram number 12
CALL GGGGXX(W(1,1),W(1,2),W(1,23),W(1,17),GG,AMP(46))
CALL GGGGXX(W(1,1),W(1,2),W(1,23),W(1,18),GG,AMP(47))
CALL GGGGXX(W(1,1),W(1,2),W(1,23),W(1,19),GG,AMP(48))
CALL GGGGXX(W(1,23),W(1,1),W(1,2),W(1,17),GG,AMP(49))
CALL GGGGXX(W(1,23),W(1,1),W(1,2),W(1,18),GG,AMP(50))
CALL GGGGXX(W(1,23),W(1,1),W(1,2),W(1,19),GG,AMP(51))
CALL GGGGXX(W(1,2),W(1,23),W(1,1),W(1,17),GG,AMP(52))
CALL GGGGXX(W(1,2),W(1,23),W(1,1),W(1,18),GG,AMP(53))
CALL GGGGXX(W(1,2),W(1,23),W(1,1),W(1,19),GG,AMP(54))
CALL JGGGXX(W(1,5),W(1,4),W(1,3),GG,W(1,24))
CALL JGGGXX(W(1,3),W(1,5),W(1,4),GG,W(1,25))
CALL JGGGXX(W(1,4),W(1,3),W(1,5),GG,W(1,26))
# Amplitude(s) for diagram number 13
CALL GGGGXX(W(1,1),W(1,2),W(1,24),W(1,12),GG,AMP(55))
CALL GGGGXX(W(1,1),W(1,2),W(1,25),W(1,12),GG,AMP(56))
CALL GGGGXX(W(1,1),W(1,2),W(1,26),W(1,12),GG,AMP(57))
CALL GGGGXX(W(1,24),W(1,1),W(1,2),W(1,12),GG,AMP(58))
CALL GGGGXX(W(1,25),W(1,1),W(1,2),W(1,12),GG,AMP(59))
CALL GGGGXX(W(1,26),W(1,1),W(1,2),W(1,12),GG,AMP(60))
CALL GGGGXX(W(1,2),W(1,24),W(1,1),W(1,12),GG,AMP(61))
CALL GGGGXX(W(1,2),W(1,25),W(1,1),W(1,12),GG,AMP(62))
CALL GGGGXX(W(1,2),W(1,26),W(1,1),W(1,12),GG,AMP(63))
# Amplitude(s) for diagram number 14
CALL GGGGXX(W(1,1),W(1,2),W(1,24),W(1,14),GG,AMP(64))
CALL GGGGXX(W(1,1),W(1,2),W(1,25),W(1,14),GG,AMP(65))
CALL GGGGXX(W(1,1),W(1,2),W(1,26),W(1,14),GG,AMP(66))
CALL GGGGXX(W(1,24),W(1,1),W(1,2),W(1,14),GG,AMP(67))
CALL GGGGXX(W(1,25),W(1,1),W(1,2),W(1,14),GG,AMP(68))
CALL GGGGXX(W(1,26),W(1,1),W(1,2),W(1,14),GG,AMP(69))
CALL GGGGXX(W(1,2),W(1,24),W(1,1),W(1,14),GG,AMP(70))
CALL GGGGXX(W(1,2),W(1,25),W(1,1),W(1,14),GG,AMP(71))
CALL GGGGXX(W(1,2),W(1,26),W(1,1),W(1,14),GG,AMP(72))
# Amplitude(s) for diagram number 15
CALL GGGGXX(W(1,1),W(1,2),W(1,24),W(1,16),GG,AMP(73))
CALL GGGGXX(W(1,1),W(1,2),W(1,25),W(1,16),GG,AMP(74))
CALL GGGGXX(W(1,1),W(1,2),W(1,26),W(1,16),GG,AMP(75))
CALL GGGGXX(W(1,24),W(1,1),W(1,2),W(1,16),GG,AMP(76))
CALL GGGGXX(W(1,25),W(1,1),W(1,2),W(1,16),GG,AMP(77))
CALL GGGGXX(W(1,26),W(1,1),W(1,2),W(1,16),GG,AMP(78))
CALL GGGGXX(W(1,2),W(1,24),W(1,1),W(1,16),GG,AMP(79))
CALL GGGGXX(W(1,2),W(1,25),W(1,1),W(1,16),GG,AMP(80))
CALL GGGGXX(W(1,2),W(1,26),W(1,1),W(1,16),GG,AMP(81))
# Amplitude(s) for diagram number 16
CALL GGGGXX(W(1,1),W(1,2),W(1,24),W(1,17),GG,AMP(82))
CALL GGGGXX(W(1,1),W(1,2),W(1,24),W(1,18),GG,AMP(83))
CALL GGGGXX(W(1,1),W(1,2),W(1,24),W(1,19),GG,AMP(84))
CALL GGGGXX(W(1,1),W(1,2),W(1,25),W(1,17),GG,AMP(85))
CALL GGGGXX(W(1,1),W(1,2),W(1,25),W(1,18),GG,AMP(86))
CALL GGGGXX(W(1,1),W(1,2),W(1,25),W(1,19),GG,AMP(87))
CALL GGGGXX(W(1,1),W(1,2),W(1,26),W(1,17),GG,AMP(88))
CALL GGGGXX(W(1,1),W(1,2),W(1,26),W(1,18),GG,AMP(89))
CALL GGGGXX(W(1,1),W(1,2),W(1,26),W(1,19),GG,AMP(90))
CALL GGGGXX(W(1,24),W(1,1),W(1,2),W(1,17),GG,AMP(91))
CALL GGGGXX(W(1,24),W(1,1),W(1,2),W(1,18),GG,AMP(92))
CALL GGGGXX(W(1,24),W(1,1),W(1,2),W(1,19),GG,AMP(93))
CALL GGGGXX(W(1,25),W(1,1),W(1,2),W(1,17),GG,AMP(94))
CALL GGGGXX(W(1,25),W(1,1),W(1,2),W(1,18),GG,AMP(95))
CALL GGGGXX(W(1,25),W(1,1),W(1,2),W(1,19),GG,AMP(96))
CALL GGGGXX(W(1,26),W(1,1),W(1,2),W(1,17),GG,AMP(97))
CALL GGGGXX(W(1,26),W(1,1),W(1,2),W(1,18),GG,AMP(98))
CALL GGGGXX(W(1,26),W(1,1),W(1,2),W(1,19),GG,AMP(99))
CALL GGGGXX(W(1,2),W(1,24),W(1,1),W(1,17),GG,AMP(100))
CALL GGGGXX(W(1,2),W(1,24),W(1,1),W(1,18),GG,AMP(101))
CALL GGGGXX(W(1,2),W(1,24),W(1,1),W(1,19),GG,AMP(102))
CALL GGGGXX(W(1,2),W(1,25),W(1,1),W(1,17),GG,AMP(103))
CALL GGGGXX(W(1,2),W(1,25),W(1,1),W(1,18),GG,AMP(104))
CALL GGGGXX(W(1,2),W(1,25),W(1,1),W(1,19),GG,AMP(105))
CALL GGGGXX(W(1,2),W(1,26),W(1,1),W(1,17),GG,AMP(106))
CALL GGGGXX(W(1,2),W(1,26),W(1,1),W(1,18),GG,AMP(107))
CALL GGGGXX(W(1,2),W(1,26),W(1,1),W(1,19),GG,AMP(108))
CALL JVVXXX(W(1,1),W(1,2),GG,zero,zero,W(1,27))
# Amplitude(s) for diagram number 17
CALL VVVXXX(W(1,7),W(1,12),W(1,27),GG,AMP(109))
# Amplitude(s) for diagram number 18
CALL VVVXXX(W(1,7),W(1,14),W(1,27),GG,AMP(110))
# Amplitude(s) for diagram number 19
CALL VVVXXX(W(1,7),W(1,16),W(1,27),GG,AMP(111))
# Amplitude(s) for diagram number 20
CALL VVVXXX(W(1,7),W(1,17),W(1,27),GG,AMP(112))
CALL VVVXXX(W(1,7),W(1,18),W(1,27),GG,AMP(113))
CALL VVVXXX(W(1,7),W(1,19),W(1,27),GG,AMP(114))
# Amplitude(s) for diagram number 21
CALL VVVXXX(W(1,21),W(1,12),W(1,27),GG,AMP(115))
# Amplitude(s) for diagram number 22
CALL VVVXXX(W(1,21),W(1,14),W(1,27),GG,AMP(116))
# Amplitude(s) for diagram number 23
CALL VVVXXX(W(1,21),W(1,16),W(1,27),GG,AMP(117))
# Amplitude(s) for diagram number 24
CALL VVVXXX(W(1,21),W(1,17),W(1,27),GG,AMP(118))
CALL VVVXXX(W(1,21),W(1,18),W(1,27),GG,AMP(119))
CALL VVVXXX(W(1,21),W(1,19),W(1,27),GG,AMP(120))
# Amplitude(s) for diagram number 25
CALL VVVXXX(W(1,23),W(1,12),W(1,27),GG,AMP(121))
# Amplitude(s) for diagram number 26
CALL VVVXXX(W(1,23),W(1,14),W(1,27),GG,AMP(122))
# Amplitude(s) for diagram number 27
CALL VVVXXX(W(1,23),W(1,16),W(1,27),GG,AMP(123))
# Amplitude(s) for diagram number 28
CALL VVVXXX(W(1,23),W(1,17),W(1,27),GG,AMP(124))
CALL VVVXXX(W(1,23),W(1,18),W(1,27),GG,AMP(125))
CALL VVVXXX(W(1,23),W(1,19),W(1,27),GG,AMP(126))
# Amplitude(s) for diagram number 29
CALL VVVXXX(W(1,24),W(1,12),W(1,27),GG,AMP(127))
CALL VVVXXX(W(1,25),W(1,12),W(1,27),GG,AMP(128))
CALL VVVXXX(W(1,26),W(1,12),W(1,27),GG,AMP(129))
# Amplitude(s) for diagram number 30
CALL VVVXXX(W(1,24),W(1,14),W(1,27),GG,AMP(130))
CALL VVVXXX(W(1,25),W(1,14),W(1,27),GG,AMP(131))
CALL VVVXXX(W(1,26),W(1,14),W(1,27),GG,AMP(132))
# Amplitude(s) for diagram number 31
CALL VVVXXX(W(1,24),W(1,16),W(1,27),GG,AMP(133))
CALL VVVXXX(W(1,25),W(1,16),W(1,27),GG,AMP(134))
CALL VVVXXX(W(1,26),W(1,16),W(1,27),GG,AMP(135))
# Amplitude(s) for diagram number 32
CALL VVVXXX(W(1,24),W(1,17),W(1,27),GG,AMP(136))
CALL VVVXXX(W(1,24),W(1,18),W(1,27),GG,AMP(137))
CALL VVVXXX(W(1,24),W(1,19),W(1,27),GG,AMP(138))
CALL VVVXXX(W(1,25),W(1,17),W(1,27),GG,AMP(139))
CALL VVVXXX(W(1,25),W(1,18),W(1,27),GG,AMP(140))
CALL VVVXXX(W(1,25),W(1,19),W(1,27),GG,AMP(141))
CALL VVVXXX(W(1,26),W(1,17),W(1,27),GG,AMP(142))
CALL VVVXXX(W(1,26),W(1,18),W(1,27),GG,AMP(143))
CALL VVVXXX(W(1,26),W(1,19),W(1,27),GG,AMP(144))
CALL JVVXXX(W(1,1),W(1,7),GG,zero,zero,W(1,28))
# Amplitude(s) for diagram number 33
CALL VVVXXX(W(1,2),W(1,12),W(1,28),GG,AMP(145))
# Amplitude(s) for diagram number 34
CALL VVVXXX(W(1,2),W(1,14),W(1,28),GG,AMP(146))
# Amplitude(s) for diagram number 35
CALL VVVXXX(W(1,2),W(1,16),W(1,28),GG,AMP(147))
# Amplitude(s) for diagram number 36
CALL VVVXXX(W(1,2),W(1,17),W(1,28),GG,AMP(148))
CALL VVVXXX(W(1,2),W(1,18),W(1,28),GG,AMP(149))
CALL VVVXXX(W(1,2),W(1,19),W(1,28),GG,AMP(150))
CALL JVVXXX(W(1,1),W(1,21),GG,zero,zero,W(1,29))
# Amplitude(s) for diagram number 37
CALL VVVXXX(W(1,2),W(1,12),W(1,29),GG,AMP(151))
# Amplitude(s) for diagram number 38
CALL VVVXXX(W(1,2),W(1,14),W(1,29),GG,AMP(152))
# Amplitude(s) for diagram number 39
CALL VVVXXX(W(1,2),W(1,16),W(1,29),GG,AMP(153))
# Amplitude(s) for diagram number 40
CALL VVVXXX(W(1,2),W(1,17),W(1,29),GG,AMP(154))
CALL VVVXXX(W(1,2),W(1,18),W(1,29),GG,AMP(155))
CALL VVVXXX(W(1,2),W(1,19),W(1,29),GG,AMP(156))
CALL JVVXXX(W(1,1),W(1,23),GG,zero,zero,W(1,30))
# Amplitude(s) for diagram number 41
CALL VVVXXX(W(1,2),W(1,12),W(1,30),GG,AMP(157))
# Amplitude(s) for diagram number 42
CALL VVVXXX(W(1,2),W(1,14),W(1,30),GG,AMP(158))
# Amplitude(s) for diagram number 43
CALL VVVXXX(W(1,2),W(1,16),W(1,30),GG,AMP(159))
# Amplitude(s) for diagram number 44
CALL VVVXXX(W(1,2),W(1,17),W(1,30),GG,AMP(160))
CALL VVVXXX(W(1,2),W(1,18),W(1,30),GG,AMP(161))
CALL VVVXXX(W(1,2),W(1,19),W(1,30),GG,AMP(162))
CALL JVVXXX(W(1,1),W(1,24),GG,zero,zero,W(1,31))
CALL JVVXXX(W(1,1),W(1,25),GG,zero,zero,W(1,32))
CALL JVVXXX(W(1,1),W(1,26),GG,zero,zero,W(1,33))
# Amplitude(s) for diagram number 45
CALL VVVXXX(W(1,2),W(1,12),W(1,31),GG,AMP(163))
CALL VVVXXX(W(1,2),W(1,12),W(1,32),GG,AMP(164))
CALL VVVXXX(W(1,2),W(1,12),W(1,33),GG,AMP(165))
# Amplitude(s) for diagram number 46
CALL VVVXXX(W(1,2),W(1,14),W(1,31),GG,AMP(166))
CALL VVVXXX(W(1,2),W(1,14),W(1,32),GG,AMP(167))
CALL VVVXXX(W(1,2),W(1,14),W(1,33),GG,AMP(168))
# Amplitude(s) for diagram number 47
CALL VVVXXX(W(1,2),W(1,16),W(1,31),GG,AMP(169))
CALL VVVXXX(W(1,2),W(1,16),W(1,32),GG,AMP(170))
CALL VVVXXX(W(1,2),W(1,16),W(1,33),GG,AMP(171))
# Amplitude(s) for diagram number 48
CALL VVVXXX(W(1,2),W(1,17),W(1,31),GG,AMP(172))
CALL VVVXXX(W(1,2),W(1,18),W(1,31),GG,AMP(173))
CALL VVVXXX(W(1,2),W(1,19),W(1,31),GG,AMP(174))
CALL VVVXXX(W(1,2),W(1,17),W(1,32),GG,AMP(175))
CALL VVVXXX(W(1,2),W(1,18),W(1,32),GG,AMP(176))
CALL VVVXXX(W(1,2),W(1,19),W(1,32),GG,AMP(177))
CALL VVVXXX(W(1,2),W(1,17),W(1,33),GG,AMP(178))
CALL VVVXXX(W(1,2),W(1,18),W(1,33),GG,AMP(179))
CALL VVVXXX(W(1,2),W(1,19),W(1,33),GG,AMP(180))
CALL JVVXXX(W(1,1),W(1,12),GG,zero,zero,W(1,34))
# Amplitude(s) for diagram number 49
CALL VVVXXX(W(1,2),W(1,7),W(1,34),GG,AMP(181))
CALL JVVXXX(W(1,1),W(1,14),GG,zero,zero,W(1,35))
# Amplitude(s) for diagram number 50
CALL VVVXXX(W(1,2),W(1,7),W(1,35),GG,AMP(182))
CALL JVVXXX(W(1,1),W(1,16),GG,zero,zero,W(1,36))
# Amplitude(s) for diagram number 51
CALL VVVXXX(W(1,2),W(1,7),W(1,36),GG,AMP(183))
CALL JVVXXX(W(1,1),W(1,17),GG,zero,zero,W(1,37))
CALL JVVXXX(W(1,1),W(1,18),GG,zero,zero,W(1,38))
CALL JVVXXX(W(1,1),W(1,19),GG,zero,zero,W(1,39))
# Amplitude(s) for diagram number 52
CALL VVVXXX(W(1,2),W(1,7),W(1,37),GG,AMP(184))
CALL VVVXXX(W(1,2),W(1,7),W(1,38),GG,AMP(185))
CALL VVVXXX(W(1,2),W(1,7),W(1,39),GG,AMP(186))
# Amplitude(s) for diagram number 53
CALL VVVXXX(W(1,2),W(1,21),W(1,34),GG,AMP(187))
# Amplitude(s) for diagram number 54
CALL VVVXXX(W(1,2),W(1,21),W(1,35),GG,AMP(188))
# Amplitude(s) for diagram number 55
CALL VVVXXX(W(1,2),W(1,21),W(1,36),GG,AMP(189))
# Amplitude(s) for diagram number 56
CALL VVVXXX(W(1,2),W(1,21),W(1,37),GG,AMP(190))
CALL VVVXXX(W(1,2),W(1,21),W(1,38),GG,AMP(191))
CALL VVVXXX(W(1,2),W(1,21),W(1,39),GG,AMP(192))
# Amplitude(s) for diagram number 57
CALL VVVXXX(W(1,2),W(1,23),W(1,34),GG,AMP(193))
# Amplitude(s) for diagram number 58
CALL VVVXXX(W(1,2),W(1,23),W(1,35),GG,AMP(194))
# Amplitude(s) for diagram number 59
CALL VVVXXX(W(1,2),W(1,23),W(1,36),GG,AMP(195))
# Amplitude(s) for diagram number 60
CALL VVVXXX(W(1,2),W(1,23),W(1,37),GG,AMP(196))
CALL VVVXXX(W(1,2),W(1,23),W(1,38),GG,AMP(197))
CALL VVVXXX(W(1,2),W(1,23),W(1,39),GG,AMP(198))
# Amplitude(s) for diagram number 61
CALL VVVXXX(W(1,2),W(1,24),W(1,34),GG,AMP(199))
CALL VVVXXX(W(1,2),W(1,25),W(1,34),GG,AMP(200))
CALL VVVXXX(W(1,2),W(1,26),W(1,34),GG,AMP(201))
# Amplitude(s) for diagram number 62
CALL VVVXXX(W(1,2),W(1,24),W(1,35),GG,AMP(202))
CALL VVVXXX(W(1,2),W(1,25),W(1,35),GG,AMP(203))
CALL VVVXXX(W(1,2),W(1,26),W(1,35),GG,AMP(204))
# Amplitude(s) for diagram number 63
CALL VVVXXX(W(1,2),W(1,24),W(1,36),GG,AMP(205))
CALL VVVXXX(W(1,2),W(1,25),W(1,36),GG,AMP(206))
CALL VVVXXX(W(1,2),W(1,26),W(1,36),GG,AMP(207))
# Amplitude(s) for diagram number 64
CALL VVVXXX(W(1,2),W(1,24),W(1,37),GG,AMP(208))
CALL VVVXXX(W(1,2),W(1,24),W(1,38),GG,AMP(209))
CALL VVVXXX(W(1,2),W(1,24),W(1,39),GG,AMP(210))
CALL VVVXXX(W(1,2),W(1,25),W(1,37),GG,AMP(211))
CALL VVVXXX(W(1,2),W(1,25),W(1,38),GG,AMP(212))
CALL VVVXXX(W(1,2),W(1,25),W(1,39),GG,AMP(213))
CALL VVVXXX(W(1,2),W(1,26),W(1,37),GG,AMP(214))
CALL VVVXXX(W(1,2),W(1,26),W(1,38),GG,AMP(215))
CALL VVVXXX(W(1,2),W(1,26),W(1,39),GG,AMP(216))""")

        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_pmass_file(writer, me)
        writer.close()

        self.assertFileContains('test',"""      PMASS(1)=ZERO
      PMASS(2)=ZERO
      PMASS(3)=ZERO
      PMASS(4)=ZERO
      PMASS(5)=ZERO
      PMASS(6)=ZERO
      PMASS(7)=ZERO
      PMASS(8)=ZERO\n""")

    def test_vector_clash_majorana_process(self):
        """Test majorana process w+ w- > n2 n2
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # Neutralino
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n2',
                      'spin':2,
                      'color':1,
                      'mass':'MN1',
                      'width':'WN1',
                      'texname':'\chi_0^2',
                      'antitexname':'\chi_0^2',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist) - 1]

        # W+/-
        mypartlist.append(base_objects.Particle({'name':'w-',
                      'antiname':'w+',
                      'spin':3,
                      'color':1,
                      'mass':'WMASS',
                      'width':'WWIDTH',
                      'texname':'w-',
                      'antitexname':'w+',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':-24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wminus.set('is_part', False)

        # chargino+/-
        mypartlist.append(base_objects.Particle({'name':'x1-',
                      'antiname':'x1+',
                      'spin':2,
                      'color':1,
                      'mass':'MX1',
                      'width':'WX1',
                      'texname':'x1-',
                      'antitexname':'x1+',
                      'line':'straight',
                      'charge':1.,
                      'pdg_code':-1000024,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        x1plus = mypartlist[len(mypartlist) - 1]
        x1minus = copy.copy(x1plus)
        x1minus.set('is_part', False)

        # Coupling of n1 to w
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             x1minus, \
                                             wplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GWN1X1'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [x1plus, \
                                             n1, \
                                             wminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GWX1N1'},
                      'orders':{'QED':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 2)

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
        """CALL VXXXXX(P(0,1),WMASS,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),WMASS,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MN1,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MN1,NHEL(4),+1*IC(4),W(1,4))
CALL FVICXX(W(1,3),W(1,1),GWN1X1,MX1,WX1,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVCXX(W(1,5),W(1,4),W(1,2),GWX1N1,AMP(1))
CALL FVOXXX(W(1,4),W(1,1),GWN1X1,MX1,WX1,W(1,6))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,3),W(1,6),W(1,2),GWX1N1,AMP(2))""")


    def test_export_majorana_decay_chain(self):
        """Test decay chain with majorana particles and MadEvent files
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
        eminus = mypartlist[len(mypartlist) - 1]
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

        # e- e+ > n1 n1 / z sl5-, n1 > e- sl2+

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

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        # This has been checked against v4
        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
                         """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL FSOXXX(W(1,3),W(1,4),MGVX350,Mneu1,Wneu1,W(1,5))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,6))
CALL SXXXXX(P(0,6),+1*IC(6),W(1,7))
CALL FSICXX(W(1,6),W(1,7),MGVX350,Mneu1,Wneu1,W(1,8))
CALL HIOXXX(W(1,1),W(1,5),MGVX494,Msl2,Wsl2,W(1,9))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,8),W(1,2),W(1,9),MGVX350,AMP(1))
CALL OXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,10))
CALL FSOXXX(W(1,10),W(1,7),MGVX350,Mneu1,Wneu1,W(1,11))
CALL HIOXXX(W(1,1),W(1,11),MGVX494,Msl2,Wsl2,W(1,12))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,13))
CALL FSICXX(W(1,13),W(1,4),MGVX350,Mneu1,Wneu1,W(1,14))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,14),W(1,2),W(1,12),MGVX350,AMP(2))""")

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(exporter.get_JAMP_lines(me)[0],
                         "JAMP(1)=+AMP(1)-AMP(2)")

        # e- e+ > n1 n1 / z sl5-, n1 > e- sl2+, n1 > e+ sl2-

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
            mydecay1, mydecay2]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        # This has been checked against v4
        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
        """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL FSOXXX(W(1,3),W(1,4),MGVX350,Mneu1,Wneu1,W(1,5))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,6))
CALL SXXXXX(P(0,6),+1*IC(6),W(1,7))
CALL FSIXXX(W(1,6),W(1,7),MGVX494,Mneu1,Wneu1,W(1,8))
CALL HIOXXX(W(1,1),W(1,5),MGVX494,Msl2,Wsl2,W(1,9))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,8),W(1,2),W(1,9),MGVX350,AMP(1))
CALL OXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,10))
CALL FSOCXX(W(1,10),W(1,7),MGVX494,Mneu1,Wneu1,W(1,11))
CALL HIOXXX(W(1,1),W(1,11),MGVX494,Msl2,Wsl2,W(1,12))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,13))
CALL FSICXX(W(1,13),W(1,4),MGVX350,Mneu1,Wneu1,W(1,14))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,14),W(1,2),W(1,12),MGVX350,AMP(2))""")

        self.assertEqual(exporter.get_JAMP_lines(me)[0],
                         "JAMP(1)=+AMP(1)-AMP(2)")


        # e- e+ > n1 n1 / z sl5-, n1 > e- sl2+ a

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
                                         'model':mymodel})

        me3 = helas_objects.HelasMatrixElement(\
            diagram_generation.Amplitude(mydecay3))

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay3]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        # This has been checked against v4
        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
                         """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),+1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),+1*IC(5),W(1,5))
CALL FVOXXX(W(1,3),W(1,5),MGVX12,zero,zero,W(1,6))
CALL FSOXXX(W(1,6),W(1,4),MGVX350,Mneu1,Wneu1,W(1,7))
CALL IXXXXX(P(0,6),zero,NHEL(6),-1*IC(6),W(1,8))
CALL SXXXXX(P(0,7),+1*IC(7),W(1,9))
CALL VXXXXX(P(0,8),zero,NHEL(8),+1*IC(8),W(1,10))
CALL FVICXX(W(1,8),W(1,10),MGVX12,zero,zero,W(1,11))
CALL FSICXX(W(1,11),W(1,9),MGVX350,Mneu1,Wneu1,W(1,12))
CALL HIOXXX(W(1,1),W(1,7),MGVX494,Msl2,Wsl2,W(1,13))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,12),W(1,2),W(1,13),MGVX350,AMP(1))
CALL HVSXXX(W(1,10),W(1,9),MGVX56,Msl2,Wsl2,W(1,14))
CALL FSICXX(W(1,8),W(1,14),MGVX350,Mneu1,Wneu1,W(1,15))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,15),W(1,2),W(1,13),MGVX350,AMP(2))
CALL HVSXXX(W(1,5),W(1,4),MGVX56,Msl2,Wsl2,W(1,16))
CALL FSOXXX(W(1,3),W(1,16),MGVX350,Mneu1,Wneu1,W(1,17))
CALL HIOXXX(W(1,1),W(1,17),MGVX494,Msl2,Wsl2,W(1,18))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,12),W(1,2),W(1,18),MGVX350,AMP(3))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,15),W(1,2),W(1,18),MGVX350,AMP(4))
CALL OXXXXX(P(0,6),zero,NHEL(6),+1*IC(6),W(1,19))
CALL FVOXXX(W(1,19),W(1,10),MGVX12,zero,zero,W(1,20))
CALL FSOXXX(W(1,20),W(1,9),MGVX350,Mneu1,Wneu1,W(1,21))
CALL HIOXXX(W(1,1),W(1,21),MGVX494,Msl2,Wsl2,W(1,22))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,23))
CALL FVICXX(W(1,23),W(1,5),MGVX12,zero,zero,W(1,24))
CALL FSICXX(W(1,24),W(1,4),MGVX350,Mneu1,Wneu1,W(1,25))
# Amplitude(s) for diagram number 5
CALL IOSXXX(W(1,25),W(1,2),W(1,22),MGVX350,AMP(5))
CALL FSOXXX(W(1,19),W(1,14),MGVX350,Mneu1,Wneu1,W(1,26))
CALL HIOXXX(W(1,1),W(1,26),MGVX494,Msl2,Wsl2,W(1,27))
# Amplitude(s) for diagram number 6
CALL IOSXXX(W(1,25),W(1,2),W(1,27),MGVX350,AMP(6))
CALL FSICXX(W(1,23),W(1,16),MGVX350,Mneu1,Wneu1,W(1,28))
# Amplitude(s) for diagram number 7
CALL IOSXXX(W(1,28),W(1,2),W(1,22),MGVX350,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOSXXX(W(1,28),W(1,2),W(1,27),MGVX350,AMP(8))""")

        # Test amp2 lines        
        amp2_lines = \
                 exporter.get_amp2_lines(me)
        self.assertEqual(amp2_lines,
                         ['AMP2(1)=AMP2(1)+AMP(1)*dconjg(AMP(1))',
                          'AMP2(2)=AMP2(2)+AMP(2)*dconjg(AMP(2))',
                          'AMP2(3)=AMP2(3)+AMP(3)*dconjg(AMP(3))',
                          'AMP2(4)=AMP2(4)+AMP(4)*dconjg(AMP(4))',
                          'AMP2(5)=AMP2(5)+AMP(5)*dconjg(AMP(5))',
                          'AMP2(6)=AMP2(6)+AMP(6)*dconjg(AMP(6))',
                          'AMP2(7)=AMP2(7)+AMP(7)*dconjg(AMP(7))',
                          'AMP2(8)=AMP2(8)+AMP(8)*dconjg(AMP(8))'])
        
        # Test jamp lines        
        self.assertEqual(exporter.get_JAMP_lines(me)[0],
                         "JAMP(1)=+AMP(1)+AMP(2)+AMP(3)+AMP(4)-AMP(5)-AMP(6)-AMP(7)-AMP(8)")

        writer = writers.FortranWriter(self.give_pos('test'))

        # Test configs file
        mapconfigs, s_and_t_channels = exporter.write_configs_file(writer,
                                     me)
        writer.close()
        
        self.assertFileContains('test',
                         """C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/8,6/
      DATA SPROP(-1,1)/11/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/7,-1/
      DATA SPROP(-2,1)/1000022/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/5,3/
      DATA SPROP(-3,1)/11/
      DATA TPRID(-3,1)/0/
      DATA (IFOREST(I,-4,1),I=1,2)/4,-3/
      DATA SPROP(-4,1)/1000022/
      DATA TPRID(-4,1)/0/
      DATA (IFOREST(I,-5,1),I=1,2)/1,-4/
      DATA TPRID(-5,1)/1000011/
      DATA SPROP(-5,1)/0/
      DATA (IFOREST(I,-6,1),I=1,2)/-5,-2/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/8,7/
      DATA SPROP(-1,2)/-1000011/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,6/
      DATA SPROP(-2,2)/1000022/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/5,3/
      DATA SPROP(-3,2)/11/
      DATA TPRID(-3,2)/0/
      DATA (IFOREST(I,-4,2),I=1,2)/4,-3/
      DATA SPROP(-4,2)/1000022/
      DATA TPRID(-4,2)/0/
      DATA (IFOREST(I,-5,2),I=1,2)/1,-4/
      DATA TPRID(-5,2)/1000011/
      DATA SPROP(-5,2)/0/
      DATA (IFOREST(I,-6,2),I=1,2)/-5,-2/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/8,6/
      DATA SPROP(-1,3)/11/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/7,-1/
      DATA SPROP(-2,3)/1000022/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/5,4/
      DATA SPROP(-3,3)/-1000011/
      DATA TPRID(-3,3)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/-3,3/
      DATA SPROP(-4,3)/1000022/
      DATA TPRID(-4,3)/0/
      DATA (IFOREST(I,-5,3),I=1,2)/1,-4/
      DATA TPRID(-5,3)/1000011/
      DATA SPROP(-5,3)/0/
      DATA (IFOREST(I,-6,3),I=1,2)/-5,-2/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/8,7/
      DATA SPROP(-1,4)/-1000011/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/-1,6/
      DATA SPROP(-2,4)/1000022/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/5,4/
      DATA SPROP(-3,4)/-1000011/
      DATA TPRID(-3,4)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,3/
      DATA SPROP(-4,4)/1000022/
      DATA TPRID(-4,4)/0/
      DATA (IFOREST(I,-5,4),I=1,2)/1,-4/
      DATA TPRID(-5,4)/1000011/
      DATA SPROP(-5,4)/0/
      DATA (IFOREST(I,-6,4),I=1,2)/-5,-2/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA (IFOREST(I,-1,5),I=1,2)/5,3/
      DATA SPROP(-1,5)/11/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/4,-1/
      DATA SPROP(-2,5)/1000022/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/8,6/
      DATA SPROP(-3,5)/11/
      DATA TPRID(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/7,-3/
      DATA SPROP(-4,5)/1000022/
      DATA TPRID(-4,5)/0/
      DATA (IFOREST(I,-5,5),I=1,2)/1,-4/
      DATA TPRID(-5,5)/1000011/
      DATA SPROP(-5,5)/0/
      DATA (IFOREST(I,-6,5),I=1,2)/-5,-2/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA (IFOREST(I,-1,6),I=1,2)/5,3/
      DATA SPROP(-1,6)/11/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/4,-1/
      DATA SPROP(-2,6)/1000022/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/8,7/
      DATA SPROP(-3,6)/-1000011/
      DATA TPRID(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/-3,6/
      DATA SPROP(-4,6)/1000022/
      DATA TPRID(-4,6)/0/
      DATA (IFOREST(I,-5,6),I=1,2)/1,-4/
      DATA TPRID(-5,6)/1000011/
      DATA SPROP(-5,6)/0/
      DATA (IFOREST(I,-6,6),I=1,2)/-5,-2/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA (IFOREST(I,-1,7),I=1,2)/5,4/
      DATA SPROP(-1,7)/-1000011/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/-1,3/
      DATA SPROP(-2,7)/1000022/
      DATA TPRID(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/8,6/
      DATA SPROP(-3,7)/11/
      DATA TPRID(-3,7)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/7,-3/
      DATA SPROP(-4,7)/1000022/
      DATA TPRID(-4,7)/0/
      DATA (IFOREST(I,-5,7),I=1,2)/1,-4/
      DATA TPRID(-5,7)/1000011/
      DATA SPROP(-5,7)/0/
      DATA (IFOREST(I,-6,7),I=1,2)/-5,-2/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA (IFOREST(I,-1,8),I=1,2)/5,4/
      DATA SPROP(-1,8)/-1000011/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/-1,3/
      DATA SPROP(-2,8)/1000022/
      DATA TPRID(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/8,7/
      DATA SPROP(-3,8)/-1000011/
      DATA TPRID(-3,8)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,6/
      DATA SPROP(-4,8)/1000022/
      DATA TPRID(-4,8)/0/
      DATA (IFOREST(I,-5,8),I=1,2)/1,-4/
      DATA TPRID(-5,8)/1000011/
      DATA SPROP(-5,8)/0/
      DATA (IFOREST(I,-6,8),I=1,2)/-5,-2/
C     Number of configs
      DATA MAPCONFIG(0)/8/
""")

        writer = writers.FortranWriter(self.give_pos('test'))

        # Test decayBW file
        exporter.write_decayBW_file(writer,
                                     s_and_t_channels)

        writer.close()
        self.assertFileContains('test',
                         """      DATA GFORCEBW(-1,1)/.FALSE./
      DATA GFORCEBW(-2,1)/.TRUE./
      DATA GFORCEBW(-3,1)/.FALSE./
      DATA GFORCEBW(-4,1)/.TRUE./
      DATA GFORCEBW(-1,2)/.FALSE./
      DATA GFORCEBW(-2,2)/.TRUE./
      DATA GFORCEBW(-3,2)/.FALSE./
      DATA GFORCEBW(-4,2)/.TRUE./
      DATA GFORCEBW(-1,3)/.FALSE./
      DATA GFORCEBW(-2,3)/.TRUE./
      DATA GFORCEBW(-3,3)/.FALSE./
      DATA GFORCEBW(-4,3)/.TRUE./
      DATA GFORCEBW(-1,4)/.FALSE./
      DATA GFORCEBW(-2,4)/.TRUE./
      DATA GFORCEBW(-3,4)/.FALSE./
      DATA GFORCEBW(-4,4)/.TRUE./
      DATA GFORCEBW(-1,5)/.FALSE./
      DATA GFORCEBW(-2,5)/.TRUE./
      DATA GFORCEBW(-3,5)/.FALSE./
      DATA GFORCEBW(-4,5)/.TRUE./
      DATA GFORCEBW(-1,6)/.FALSE./
      DATA GFORCEBW(-2,6)/.TRUE./
      DATA GFORCEBW(-3,6)/.FALSE./
      DATA GFORCEBW(-4,6)/.TRUE./
      DATA GFORCEBW(-1,7)/.FALSE./
      DATA GFORCEBW(-2,7)/.TRUE./
      DATA GFORCEBW(-3,7)/.FALSE./
      DATA GFORCEBW(-4,7)/.TRUE./
      DATA GFORCEBW(-1,8)/.FALSE./
      DATA GFORCEBW(-2,8)/.TRUE./
      DATA GFORCEBW(-3,8)/.FALSE./
      DATA GFORCEBW(-4,8)/.TRUE./
""")

        fortran_model = helas_call_writers.FortranHelasCallWriter(mymodel)

        # Test dname.mg
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_dname_file(writer,
                                  "P"+me.get('processes')[0].shell_string())
        writer.close()
        self.assertFileContains('test', "DIRNAME=P0_emep_n1n1_n1_emsl2pa_n1_emsl2pa\n")
        # Test iproc.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_iproc_file(writer, 0)
        writer.close()
        self.assertFileContains('test', "      1\n")
        # Test maxamps.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        # Extract ncolor
        ncolor = max(1, len(me.get('color_basis')))
        exporter.write_maxamps_file(writer,
                                     len(me.get_all_amplitudes()),
                                     ncolor,
                                     len(me.get('processes')),
                                     1)
        writer.close()
        self.assertFileContains('test',
                                "      INTEGER    MAXAMPS, MAXFLOW, " + \
                                "MAXPROC, MAXSPROC\n" + \
                                "      PARAMETER (MAXAMPS=8, MAXFLOW=1)\n" + \
                                "      PARAMETER (MAXPROC=1, MAXSPROC=1)\n")
        # Test mg.sym
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_mg_sym_file(writer, me)
        writer.close()
        self.assertFileContains('test', """      3
      2
      3
      6
      2
      4
      7
      2
      5
      8\n""")
        # Test ncombs.inc
        nexternal, ninitial = me.get_nexternal_ninitial()
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_ncombs_file(writer, nexternal)
        writer.close()
        self.assertFileContains('test',
                         """      INTEGER    N_MAX_CL
      PARAMETER (N_MAX_CL=256)\n""")
        # Test nexternal.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_nexternal_file(writer, nexternal, ninitial)
        writer.close()
        self.assertFileContains('test',
                         """      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=8)
      INTEGER    NINCOMING
      PARAMETER (NINCOMING=2)\n""")
        # Test ngraphs.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_ngraphs_file(writer, len(mapconfigs))
        writer.close()
        self.assertFileContains('test',
                         """      INTEGER    N_MAX_CG
      PARAMETER (N_MAX_CG=8)\n""")
        # Test props.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_props_file(writer, me, s_and_t_channels)
        writer.close()
        self.assertFileContains('test',
                         """      PMASS(-1,1)  = ZERO
      PWIDTH(-1,1) = ZERO
      POW(-1,1) = 1
      PMASS(-2,1)  = ABS(MNEU1)
      PWIDTH(-2,1) = ABS(WNEU1)
      POW(-2,1) = 1
      PMASS(-3,1)  = ZERO
      PWIDTH(-3,1) = ZERO
      POW(-3,1) = 1
      PMASS(-4,1)  = ABS(MNEU1)
      PWIDTH(-4,1) = ABS(WNEU1)
      POW(-4,1) = 1
      PMASS(-5,1)  = ABS(MSL2)
      PWIDTH(-5,1) = ABS(WSL2)
      POW(-5,1) = 2
      PMASS(-1,2)  = ABS(MSL2)
      PWIDTH(-1,2) = ABS(WSL2)
      POW(-1,2) = 2
      PMASS(-2,2)  = ABS(MNEU1)
      PWIDTH(-2,2) = ABS(WNEU1)
      POW(-2,2) = 1
      PMASS(-3,2)  = ZERO
      PWIDTH(-3,2) = ZERO
      POW(-3,2) = 1
      PMASS(-4,2)  = ABS(MNEU1)
      PWIDTH(-4,2) = ABS(WNEU1)
      POW(-4,2) = 1
      PMASS(-5,2)  = ABS(MSL2)
      PWIDTH(-5,2) = ABS(WSL2)
      POW(-5,2) = 2
      PMASS(-1,3)  = ZERO
      PWIDTH(-1,3) = ZERO
      POW(-1,3) = 1
      PMASS(-2,3)  = ABS(MNEU1)
      PWIDTH(-2,3) = ABS(WNEU1)
      POW(-2,3) = 1
      PMASS(-3,3)  = ABS(MSL2)
      PWIDTH(-3,3) = ABS(WSL2)
      POW(-3,3) = 2
      PMASS(-4,3)  = ABS(MNEU1)
      PWIDTH(-4,3) = ABS(WNEU1)
      POW(-4,3) = 1
      PMASS(-5,3)  = ABS(MSL2)
      PWIDTH(-5,3) = ABS(WSL2)
      POW(-5,3) = 2
      PMASS(-1,4)  = ABS(MSL2)
      PWIDTH(-1,4) = ABS(WSL2)
      POW(-1,4) = 2
      PMASS(-2,4)  = ABS(MNEU1)
      PWIDTH(-2,4) = ABS(WNEU1)
      POW(-2,4) = 1
      PMASS(-3,4)  = ABS(MSL2)
      PWIDTH(-3,4) = ABS(WSL2)
      POW(-3,4) = 2
      PMASS(-4,4)  = ABS(MNEU1)
      PWIDTH(-4,4) = ABS(WNEU1)
      POW(-4,4) = 1
      PMASS(-5,4)  = ABS(MSL2)
      PWIDTH(-5,4) = ABS(WSL2)
      POW(-5,4) = 2
      PMASS(-1,5)  = ZERO
      PWIDTH(-1,5) = ZERO
      POW(-1,5) = 1
      PMASS(-2,5)  = ABS(MNEU1)
      PWIDTH(-2,5) = ABS(WNEU1)
      POW(-2,5) = 1
      PMASS(-3,5)  = ZERO
      PWIDTH(-3,5) = ZERO
      POW(-3,5) = 1
      PMASS(-4,5)  = ABS(MNEU1)
      PWIDTH(-4,5) = ABS(WNEU1)
      POW(-4,5) = 1
      PMASS(-5,5)  = ABS(MSL2)
      PWIDTH(-5,5) = ABS(WSL2)
      POW(-5,5) = 2
      PMASS(-1,6)  = ZERO
      PWIDTH(-1,6) = ZERO
      POW(-1,6) = 1
      PMASS(-2,6)  = ABS(MNEU1)
      PWIDTH(-2,6) = ABS(WNEU1)
      POW(-2,6) = 1
      PMASS(-3,6)  = ABS(MSL2)
      PWIDTH(-3,6) = ABS(WSL2)
      POW(-3,6) = 2
      PMASS(-4,6)  = ABS(MNEU1)
      PWIDTH(-4,6) = ABS(WNEU1)
      POW(-4,6) = 1
      PMASS(-5,6)  = ABS(MSL2)
      PWIDTH(-5,6) = ABS(WSL2)
      POW(-5,6) = 2
      PMASS(-1,7)  = ABS(MSL2)
      PWIDTH(-1,7) = ABS(WSL2)
      POW(-1,7) = 2
      PMASS(-2,7)  = ABS(MNEU1)
      PWIDTH(-2,7) = ABS(WNEU1)
      POW(-2,7) = 1
      PMASS(-3,7)  = ZERO
      PWIDTH(-3,7) = ZERO
      POW(-3,7) = 1
      PMASS(-4,7)  = ABS(MNEU1)
      PWIDTH(-4,7) = ABS(WNEU1)
      POW(-4,7) = 1
      PMASS(-5,7)  = ABS(MSL2)
      PWIDTH(-5,7) = ABS(WSL2)
      POW(-5,7) = 2
      PMASS(-1,8)  = ABS(MSL2)
      PWIDTH(-1,8) = ABS(WSL2)
      POW(-1,8) = 2
      PMASS(-2,8)  = ABS(MNEU1)
      PWIDTH(-2,8) = ABS(WNEU1)
      POW(-2,8) = 1
      PMASS(-3,8)  = ABS(MSL2)
      PWIDTH(-3,8) = ABS(WSL2)
      POW(-3,8) = 2
      PMASS(-4,8)  = ABS(MNEU1)
      PWIDTH(-4,8) = ABS(WNEU1)
      POW(-4,8) = 1
      PMASS(-5,8)  = ABS(MSL2)
      PWIDTH(-5,8) = ABS(WSL2)
      POW(-5,8) = 2\n""")

        # Test reversed order of decay specifications
        # e- e+ > se+ se-, se- > e- n1, se+ > e+ n1

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000011,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mydecay2 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay2, mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        self.assertEqual("\n".join(myfortranmodel.get_matrix_element_calls(me)),
                         """CALL IXXXXX(P(0,1),zero,NHEL(1),+1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),Mneu1,NHEL(4),-1*IC(4),W(1,4))
CALL HIOXXX(W(1,4),W(1,3),MGVX350,Msl2,Wsl2,W(1,5))
CALL IXXXXX(P(0,5),zero,NHEL(5),-1*IC(5),W(1,6))
CALL OXXXXX(P(0,6),Mneu1,NHEL(6),+1*IC(6),W(1,7))
CALL HIOXXX(W(1,6),W(1,7),MGVX494,Msl2,Wsl2,W(1,8))
CALL JIOXXX(W(1,1),W(1,2),MGVX12,zero,zero,W(1,9))
# Amplitude(s) for diagram number 1
CALL VSSXXX(W(1,9),W(1,8),W(1,5),MGVX56,AMP(1))
CALL FSIXXX(W(1,1),W(1,5),MGVX494,Mneu1,Wneu1,W(1,10))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,10),W(1,2),W(1,8),MGVX350,AMP(2))""")

    def test_export_complicated_majorana_decay_chain(self):
        """Test complicated decay chain z e+ > n2 el+, n2 > e- e+ n1
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
        eminus = mypartlist[len(mypartlist) - 1]
        eplus = copy.copy(eminus)
        eplus.set('is_part', False)

        # A E slepton and its antiparticle
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
        seminus = mypartlist[len(mypartlist) - 1]
        seplus = copy.copy(seminus)
        seplus.set('is_part', False)

        # Neutralinos
        mypartlist.append(base_objects.Particle({'name':'n1',
                      'antiname':'n1',
                      'spin':2,
                      'color':1,
                      'mass':'mn1',
                      'width':'zero',
                      'texname':'\chi_0^1',
                      'antitexname':'\chi_0^1',
                      'line':'straight',
                      'charge':0.,
                      'pdg_code':1000022,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        n1 = mypartlist[len(mypartlist) - 1]

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
        n2 = mypartlist[len(mypartlist) - 1]

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
        z = mypartlist[len(mypartlist) - 1]

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

        # Coupling of n1 to n2 and z
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             n2, \
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZN12'},
                      'orders':{'QED':1}}))

        # Coupling of n1 and n2 to e and el
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n1, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN1M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [n1, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN1P'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 5,
                      'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             n2, \
                                             seminus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN2M'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 6,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             eminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GELN2P'},
                      'orders':{'QED':1}}))

        # Coupling of n2 to z
        myinterlist.append(base_objects.Interaction({
                      'id': 7,
                      'particles': base_objects.ParticleList(\
                                            [n2, \
                                             n2, \
                                             z]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZN22'},
                      'orders':{'QED':1}}))

        # Coupling of el to z
        myinterlist.append(base_objects.Interaction({
                      'id': 8,
                      'particles': base_objects.ParticleList(\
                                            [z, \
                                             seminus, \
                                             seplus]),
                      'color': [],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GZELEL'},
                      'orders':{'QED':1}}))


        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':23,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000023,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-1000011,
                                         'state':True}))

        mycoreproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel,
                                           'forbidden_particles':[1000022]})

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':1000023,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-11,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000022,
                                         'state':True}))

        mydecay1 = base_objects.Process({'legs':myleglist,
                                         'model':mymodel})

        mycoreproc.set('decay_chains', base_objects.ProcessList([\
            mydecay1]))

        myamplitude = diagram_generation.DecayChainAmplitude(mycoreproc)

        matrix_element = helas_objects.HelasDecayChainProcess(myamplitude)

        matrix_elements = matrix_element.combine_decay_chain_processes()

        me = matrix_elements[0]

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)

        result = myfortranmodel.get_matrix_element_calls(me)
        goal = """CALL VXXXXX(P(0,1),zmass,NHEL(1),-1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),zero,NHEL(3),+1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),zero,NHEL(4),-1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),mn1,NHEL(5),-1*IC(5),W(1,5))
CALL JIOXXX(W(1,4),W(1,3),GZL,zmass,zwidth,W(1,6))
CALL FVIXXX(W(1,5),W(1,6),GZN12,mn2,wn2,W(1,7))
CALL SXXXXX(P(0,6),+1*IC(6),W(1,8))
CALL FVOXXX(W(1,2),W(1,1),GZL,zero,zero,W(1,9))
# Amplitude(s) for diagram number 1
CALL IOSXXX(W(1,7),W(1,9),W(1,8),GELN2P,AMP(1))
CALL HIOXXX(W(1,5),W(1,3),GELN1P,Msl2,Wsl2,W(1,10))
CALL FSIXXX(W(1,4),W(1,10),GELN2M,mn2,wn2,W(1,11))
# Amplitude(s) for diagram number 2
CALL IOSXXX(W(1,11),W(1,9),W(1,8),GELN2P,AMP(2))
CALL OXXXXX(P(0,5),mn1,NHEL(5),+1*IC(5),W(1,12))
CALL HIOXXX(W(1,4),W(1,12),GELN1M,Msl2,Wsl2,W(1,13))
CALL IXXXXX(P(0,3),zero,NHEL(3),-1*IC(3),W(1,14))
CALL FSICXX(W(1,14),W(1,13),GELN2P,mn2,wn2,W(1,15))
# Amplitude(s) for diagram number 3
CALL IOSXXX(W(1,15),W(1,9),W(1,8),GELN2P,AMP(3))
CALL FVIXXX(W(1,7),W(1,1),GZN22,mn2,wn2,W(1,16))
# Amplitude(s) for diagram number 4
CALL IOSXXX(W(1,16),W(1,2),W(1,8),GELN2P,AMP(4))
CALL FVIXXX(W(1,11),W(1,1),GZN22,mn2,wn2,W(1,17))
# Amplitude(s) for diagram number 5
CALL IOSXXX(W(1,17),W(1,2),W(1,8),GELN2P,AMP(5))
CALL FVIXXX(W(1,15),W(1,1),GZN22,mn2,wn2,W(1,18))
# Amplitude(s) for diagram number 6
CALL IOSXXX(W(1,18),W(1,2),W(1,8),GELN2P,AMP(6))
CALL HVSXXX(W(1,1),W(1,8),-GZELEL,Msl2,Wsl2,W(1,19))
# Amplitude(s) for diagram number 7
CALL IOSXXX(W(1,7),W(1,2),W(1,19),GELN2P,AMP(7))
# Amplitude(s) for diagram number 8
CALL IOSXXX(W(1,11),W(1,2),W(1,19),GELN2P,AMP(8))
# Amplitude(s) for diagram number 9
CALL IOSXXX(W(1,15),W(1,2),W(1,19),GELN2P,AMP(9))""".split('\n')

        for i in range(max(len(result), len(goal))):
            self.assertEqual(result[i], goal[i])

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(exporter.get_JAMP_lines(me)[0],
                         "JAMP(1)=+AMP(1)-AMP(2)-AMP(3)+AMP(4)-AMP(5)-AMP(6)+AMP(7)-AMP(8)-AMP(9)")


    def test_duplicate_lorentz_structures(self):
        """Test duplicate Lorentz structure with only one color structure.
        """

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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        # A z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
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
        z = mypartlist[len(mypartlist) - 1]

        # u ubar z coupling
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [u, \
                                             antiu, \
                                             z]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L4', 'L7'],
                      'couplings':{(0,0):'GC_23',(0,1):'GC_24'},
                      'orders':{'QED':1}}))



        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        self.assertEqual(len(myamplitude.get('diagrams')), 2)

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=True)

        self.assertEqual(sum([len(diagram.get('amplitudes')) for diagram in \
                          me.get('diagrams')]), 8)

        for i, amp in enumerate(me.get_all_amplitudes()):
            self.assertEqual(amp.get('number'), i + 1)

        self.assertEqual(len(me.get('color_basis')), 2)

        exporter = export_v4.ProcessExporterFortranME()

        self.assertEqual(exporter.get_JAMP_lines(me),
                         ["JAMP(1)=-AMP(1)-AMP(2)-AMP(3)-AMP(4)",
                         "JAMP(2)=+AMP(5)+AMP(6)+AMP(7)+AMP(8)"])

    def test_generate_helas_diagrams_gg_gogo(self):
        """Testing the v4 helas diagram generation g g > go go,
        where there is no extra sign.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # A gluino
        mypartlist.append(base_objects.Particle({'name': 'go',
                                                 'antiname': 'go',
                                                 'spin': 2,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'MGO',
                                                 'width': 'WGO',
                                                 'pdg_code': 1000021,
                                                 'texname': 'go',
                                                 'antitexname': 'go',
                                                 'line': 'straight',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))
        go = mypartlist[len(mypartlist) - 1]

        # Triple glue coupling
        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([g, g, g]),
            'lorentz': [''],
            'couplings': {(0, 0): 'G'},
            'orders': {'QCD': 1}
            }))

        # go-go-g coupling
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([go, go, g]),
            'lorentz': [''],
            'couplings': {(0, 0): 'GGI'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude,
                                                          gen_color=False)

        goal_string = """CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MGO,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MGO,NHEL(4),+1*IC(4),W(1,4))
CALL JVVXXX(W(1,1),W(1,2),G,ZERO,ZERO,W(1,5))
# Amplitude(s) for diagram number 1
CALL IOVXXX(W(1,3),W(1,4),W(1,5),GGI,AMP(1))
CALL FVIXXX(W(1,3),W(1,1),GGI,MGO,WGO,W(1,6))
# Amplitude(s) for diagram number 2
CALL IOVXXX(W(1,6),W(1,4),W(1,2),GGI,AMP(2))
CALL FVOXXX(W(1,4),W(1,1),GGI,MGO,WGO,W(1,7))
# Amplitude(s) for diagram number 3
CALL IOVXXX(W(1,3),W(1,7),W(1,2),GGI,AMP(3))""".split('\n')

        result = helas_call_writers.FortranHelasCallWriter(mybasemodel).\
                 get_matrix_element_calls(matrix_element)
        for i in range(max(len(goal_string),len(result))):
            self.assertEqual(result[i], goal_string[i])

    def test_generate_ufo_helas_diagrams_gg_gogo(self):
        """Testing minus sign on the FFV_1 UFO helas go-go-g coupling.
        """

        # Set up model

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # A gluino
        mypartlist.append(base_objects.Particle({'name': 'go',
                                                 'antiname': 'go',
                                                 'spin': 2,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'MGO',
                                                 'width': 'WGO',
                                                 'pdg_code': 1000021,
                                                 'texname': 'go',
                                                 'antitexname': 'go',
                                                 'line': 'straight',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))
        go = mypartlist[len(mypartlist) - 1]

        # Triple glue coupling
        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList([g, g, g]),
            'lorentz': ['VVV1'],
            'couplings': {(0, 0): 'G'},
            'orders': {'QCD': 1}
            }))

        # go-go-g coupling
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList([go, go, g]),
            'lorentz': ['FFV1'],
            'couplings': {(0, 0): 'GGI'},
            'orders': {'QCD': 1}
            }))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                         'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude(myproc)

        matrix_element = helas_objects.HelasMatrixElement(myamplitude,
                                                          gen_color=False)

        goal_string = """CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
CALL IXXXXX(P(0,3),MGO,NHEL(3),-1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),MGO,NHEL(4),+1*IC(4),W(1,4))
CALL VVV1_1(W(1,1),W(1,2),G,ZERO, ZERO, W(1,5))
# Amplitude(s) for diagram number 1
CALL FFV1_0(W(1,3),W(1,4),W(1,5),GGI,AMP(1))
CALL FFV1_2(W(1,3),W(1,1),-GGI,MGO, WGO, W(1,6))
# Amplitude(s) for diagram number 2
CALL FFV1_0(W(1,6),W(1,4),W(1,2),GGI,AMP(2))
CALL FFV1_1(W(1,4),W(1,1),GGI,MGO, WGO, W(1,7))
# Amplitude(s) for diagram number 3
CALL FFV1_0(W(1,3),W(1,7),W(1,2),GGI,AMP(3))""".split('\n')

        result = helas_call_writers.FortranUFOHelasCallWriter(mybasemodel).\
                 get_matrix_element_calls(matrix_element)
        for i in range(max(len(goal_string),len(result))):
            self.assertEqual(result[i], goal_string[i])

    def test_configs_ug_ttxz(self):
        """Test configs.inc which previously failed.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # u and t quarks
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
        u = mypartlist[len(mypartlist) - 1]
        antiu = copy.copy(u)
        antiu.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

        # A z
        mypartlist.append(base_objects.Particle({'name':'z',
                      'antiname':'z',
                      'spin':3,
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
        z = mypartlist[len(mypartlist) - 1]

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # t tbar z couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [t, \
                                             antit, \
                                             z]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antiu, \
                                             u, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':6}))
        myleglist.append(base_objects.Leg({'id':-6}))
        myleglist.append(base_objects.Leg({'id':23}))
        myleglist.append(base_objects.Leg({'id':2}))

        myproc = base_objects.Process({'legs':myleglist,
                                           'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)
        writer = writers.FortranWriter(self.give_pos('test'))

        exporter = export_v4.ProcessExporterFortranME()

        # Test configs file
        nconfig, s_and_t_channels = exporter.write_configs_file(writer,
                                                                 me)
        writer.close()

        # 2 21 > 6 -6 23  2
        # 1  2   3  4  5  6
        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/5,3/
      DATA SPROP(-1,1)/6/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/4,-1/
      DATA SPROP(-2,1)/21/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/6,-2/
      DATA SPROP(-3,1)/2/
      DATA TPRID(-3,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/5,4/
      DATA SPROP(-1,2)/-6/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/-1,3/
      DATA SPROP(-2,2)/21/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/6,-2/
      DATA SPROP(-3,2)/2/
      DATA TPRID(-3,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/1,6/
      DATA TPRID(-1,3)/21/
      DATA SPROP(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,4/
      DATA TPRID(-2,3)/6/
      DATA SPROP(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/-2,5/
      DATA TPRID(-3,3)/6/
      DATA SPROP(-3,3)/0/
      DATA (IFOREST(I,-4,3),I=1,2)/-3,3/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/5,4/
      DATA SPROP(-1,4)/-6/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/1,6/
      DATA TPRID(-2,4)/21/
      DATA SPROP(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/-2,-1/
      DATA TPRID(-3,4)/6/
      DATA SPROP(-3,4)/0/
      DATA (IFOREST(I,-4,4),I=1,2)/-3,3/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA (IFOREST(I,-1,5),I=1,2)/1,6/
      DATA TPRID(-1,5)/21/
      DATA SPROP(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/-1,3/
      DATA TPRID(-2,5)/6/
      DATA SPROP(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/-2,5/
      DATA TPRID(-3,5)/6/
      DATA SPROP(-3,5)/0/
      DATA (IFOREST(I,-4,5),I=1,2)/-3,4/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA (IFOREST(I,-1,6),I=1,2)/5,3/
      DATA SPROP(-1,6)/6/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/1,6/
      DATA TPRID(-2,6)/21/
      DATA SPROP(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/-2,-1/
      DATA TPRID(-3,6)/6/
      DATA SPROP(-3,6)/0/
      DATA (IFOREST(I,-4,6),I=1,2)/-3,4/
C     Diagram 7
      DATA MAPCONFIG(7)/7/
      DATA (IFOREST(I,-1,7),I=1,2)/5,3/
      DATA SPROP(-1,7)/6/
      DATA TPRID(-1,7)/0/
      DATA (IFOREST(I,-2,7),I=1,2)/1,6/
      DATA TPRID(-2,7)/21/
      DATA SPROP(-2,7)/0/
      DATA (IFOREST(I,-3,7),I=1,2)/-2,4/
      DATA TPRID(-3,7)/6/
      DATA SPROP(-3,7)/0/
      DATA (IFOREST(I,-4,7),I=1,2)/-3,-1/
C     Diagram 8
      DATA MAPCONFIG(8)/8/
      DATA (IFOREST(I,-1,8),I=1,2)/5,4/
      DATA SPROP(-1,8)/-6/
      DATA TPRID(-1,8)/0/
      DATA (IFOREST(I,-2,8),I=1,2)/1,6/
      DATA TPRID(-2,8)/21/
      DATA SPROP(-2,8)/0/
      DATA (IFOREST(I,-3,8),I=1,2)/-2,3/
      DATA TPRID(-3,8)/6/
      DATA SPROP(-3,8)/0/
      DATA (IFOREST(I,-4,8),I=1,2)/-3,-1/
C     Diagram 9
      DATA MAPCONFIG(9)/9/
      DATA (IFOREST(I,-1,9),I=1,2)/5,3/
      DATA SPROP(-1,9)/6/
      DATA TPRID(-1,9)/0/
      DATA (IFOREST(I,-2,9),I=1,2)/4,-1/
      DATA SPROP(-2,9)/21/
      DATA TPRID(-2,9)/0/
      DATA (IFOREST(I,-3,9),I=1,2)/1,-2/
      DATA TPRID(-3,9)/2/
      DATA SPROP(-3,9)/0/
      DATA (IFOREST(I,-4,9),I=1,2)/-3,6/
C     Diagram 10
      DATA MAPCONFIG(10)/10/
      DATA (IFOREST(I,-1,10),I=1,2)/5,4/
      DATA SPROP(-1,10)/-6/
      DATA TPRID(-1,10)/0/
      DATA (IFOREST(I,-2,10),I=1,2)/-1,3/
      DATA SPROP(-2,10)/21/
      DATA TPRID(-2,10)/0/
      DATA (IFOREST(I,-3,10),I=1,2)/1,-2/
      DATA TPRID(-3,10)/2/
      DATA SPROP(-3,10)/0/
      DATA (IFOREST(I,-4,10),I=1,2)/-3,6/
C     Number of configs
      DATA MAPCONFIG(0)/10/
""")
        # Test props.inc
        writer = writers.FortranWriter(self.give_pos('test'))
        exporter.write_props_file(writer, me, s_and_t_channels)
        writer.close()
        self.assertFileContains('test',
"""      PMASS(-1,1)  = ABS(MT)
      PWIDTH(-1,1) = ABS(WT)
      POW(-1,1) = 1
      PMASS(-2,1)  = ZERO
      PWIDTH(-2,1) = ZERO
      POW(-2,1) = 2
      PMASS(-3,1)  = ZERO
      PWIDTH(-3,1) = ZERO
      POW(-3,1) = 1
      PMASS(-1,2)  = ABS(MT)
      PWIDTH(-1,2) = ABS(WT)
      POW(-1,2) = 1
      PMASS(-2,2)  = ZERO
      PWIDTH(-2,2) = ZERO
      POW(-2,2) = 2
      PMASS(-3,2)  = ZERO
      PWIDTH(-3,2) = ZERO
      POW(-3,2) = 1
      PMASS(-1,3)  = ZERO
      PWIDTH(-1,3) = ZERO
      POW(-1,3) = 2
      PMASS(-2,3)  = ABS(MT)
      PWIDTH(-2,3) = ABS(WT)
      POW(-2,3) = 1
      PMASS(-3,3)  = ABS(MT)
      PWIDTH(-3,3) = ABS(WT)
      POW(-3,3) = 1
      PMASS(-1,4)  = ABS(MT)
      PWIDTH(-1,4) = ABS(WT)
      POW(-1,4) = 1
      PMASS(-2,4)  = ZERO
      PWIDTH(-2,4) = ZERO
      POW(-2,4) = 2
      PMASS(-3,4)  = ABS(MT)
      PWIDTH(-3,4) = ABS(WT)
      POW(-3,4) = 1
      PMASS(-1,5)  = ZERO
      PWIDTH(-1,5) = ZERO
      POW(-1,5) = 2
      PMASS(-2,5)  = ABS(MT)
      PWIDTH(-2,5) = ABS(WT)
      POW(-2,5) = 1
      PMASS(-3,5)  = ABS(MT)
      PWIDTH(-3,5) = ABS(WT)
      POW(-3,5) = 1
      PMASS(-1,6)  = ABS(MT)
      PWIDTH(-1,6) = ABS(WT)
      POW(-1,6) = 1
      PMASS(-2,6)  = ZERO
      PWIDTH(-2,6) = ZERO
      POW(-2,6) = 2
      PMASS(-3,6)  = ABS(MT)
      PWIDTH(-3,6) = ABS(WT)
      POW(-3,6) = 1
      PMASS(-1,7)  = ABS(MT)
      PWIDTH(-1,7) = ABS(WT)
      POW(-1,7) = 1
      PMASS(-2,7)  = ZERO
      PWIDTH(-2,7) = ZERO
      POW(-2,7) = 2
      PMASS(-3,7)  = ABS(MT)
      PWIDTH(-3,7) = ABS(WT)
      POW(-3,7) = 1
      PMASS(-1,8)  = ABS(MT)
      PWIDTH(-1,8) = ABS(WT)
      POW(-1,8) = 1
      PMASS(-2,8)  = ZERO
      PWIDTH(-2,8) = ZERO
      POW(-2,8) = 2
      PMASS(-3,8)  = ABS(MT)
      PWIDTH(-3,8) = ABS(WT)
      POW(-3,8) = 1
      PMASS(-1,9)  = ABS(MT)
      PWIDTH(-1,9) = ABS(WT)
      POW(-1,9) = 1
      PMASS(-2,9)  = ZERO
      PWIDTH(-2,9) = ZERO
      POW(-2,9) = 2
      PMASS(-3,9)  = ZERO
      PWIDTH(-3,9) = ZERO
      POW(-3,9) = 1
      PMASS(-1,10)  = ABS(MT)
      PWIDTH(-1,10) = ABS(WT)
      POW(-1,10) = 1
      PMASS(-2,10)  = ZERO
      PWIDTH(-2,10) = ZERO
      POW(-2,10) = 2
      PMASS(-3,10)  = ZERO
      PWIDTH(-3,10) = ZERO
      POW(-3,10) = 1
""")
        
    def test_configs_long_decay(self):
        """Test configs.inc which previously failed.
        """

        mypartlist = base_objects.ParticleList()
        myinterlist = base_objects.InteractionList()

        # b and t quarks
        mypartlist.append(base_objects.Particle({'name':'b',
                      'antiname':'b~',
                      'spin':2,
                      'color':3,
                      'mass':'zero',
                      'width':'zero',
                      'texname':'b',
                      'antitexname':'\bar b',
                      'line':'straight',
                      'charge':-1. / 3.,
                      'pdg_code':5,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        b = mypartlist[len(mypartlist) - 1]
        antib = copy.copy(b)
        antib.set('is_part', False)

        mypartlist.append(base_objects.Particle({'name':'t',
                      'antiname':'t~',
                      'spin':2,
                      'color':3,
                      'mass':'MT',
                      'width':'WT',
                      'texname':'y',
                      'antitexname':'\bar t',
                      'line':'straight',
                      'charge':2. / 3.,
                      'pdg_code':6,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':False}))
        t = mypartlist[len(mypartlist) - 1]
        antit = copy.copy(t)
        antit.set('is_part', False)

        # A w
        mypartlist.append(base_objects.Particle({'name':'w+',
                      'antiname':'w+',
                      'spin':3,
                      'mass':'wmass',
                      'width':'wwidth',
                      'texname':'\gamma',
                      'antitexname':'\gamma',
                      'line':'wavy',
                      'charge':1.,
                      'pdg_code':24,
                      'propagating':True,
                      'is_part':True,
                      'self_antipart':True}))
        wplus = mypartlist[len(mypartlist) - 1]
        wminus = copy.copy(wplus)
        wplus.set('is_part', False)

        # A gluon
        mypartlist.append(base_objects.Particle({'name': 'g',
                                                 'antiname': 'g',
                                                 'spin': 3,
                                                 'color': 8,
                                                 'charge': 0.00,
                                                 'mass': 'ZERO',
                                                 'width': 'ZERO',
                                                 'pdg_code': 21,
                                                 'texname': '_',
                                                 'antitexname': '_',
                                                 'line': 'curly',
                                                 'propagating': True,
                                                 'is_part': True,
                                                 'self_antipart': True}))

        g = mypartlist[len(mypartlist) - 1]

        # t b w couplings
        myinterlist.append(base_objects.Interaction({
                      'id': 1,
                      'particles': base_objects.ParticleList(\
                                            [t, \
                                             antib, \
                                             wminus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 2,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             b, \
                                             wplus]),
                      'color': [color.ColorString([color.T(0, 1)])],
                      'lorentz':['L1'],
                      'couplings':{(0,0):'GC_23'},
                      'orders':{'QED':1}}))

        # Gluon couplings to quarks
        myinterlist.append(base_objects.Interaction({
                      'id': 3,
                      'particles': base_objects.ParticleList(\
                                            [antib, \
                                             b, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        myinterlist.append(base_objects.Interaction({
                      'id': 4,
                      'particles': base_objects.ParticleList(\
                                            [antit, \
                                             t, \
                                             g]),
                      'color': [color.ColorString([color.T(2, 1, 0)])],
                      'lorentz':[''],
                      'couplings':{(0, 0):'GG'},
                      'orders':{'QCD':1}}))

        mymodel = base_objects.Model()
        mymodel.set('particles', mypartlist)
        mymodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':6,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21}))
        myleglist.append(base_objects.Leg({'id':5}))
        myleglist.append(base_objects.Leg({'id':24}))
        myleglist.append(base_objects.Leg({'id':21}))
        
        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        myfortranmodel = helas_call_writers.FortranHelasCallWriter(mymodel)
        writer = writers.FortranWriter(self.give_pos('test'))

        exporter = export_v4.ProcessExporterFortranME()

        # Test configs file
        nconfig, s_and_t_channels = exporter.write_configs_file(writer, me)
        writer.close()

        self.assertFileContains('test',
"""C     Diagram 1
      DATA MAPCONFIG(1)/1/
      DATA (IFOREST(I,-1,1),I=1,2)/3,2/
      DATA SPROP(-1,1)/5/
      DATA TPRID(-1,1)/0/
      DATA (IFOREST(I,-2,1),I=1,2)/4,-1/
      DATA SPROP(-2,1)/6/
      DATA TPRID(-2,1)/0/
      DATA (IFOREST(I,-3,1),I=1,2)/5,-2/
      DATA SPROP(-3,1)/6/
      DATA TPRID(-3,1)/0/
C     Diagram 2
      DATA MAPCONFIG(2)/2/
      DATA (IFOREST(I,-1,2),I=1,2)/3,2/
      DATA SPROP(-1,2)/5/
      DATA TPRID(-1,2)/0/
      DATA (IFOREST(I,-2,2),I=1,2)/5,-1/
      DATA SPROP(-2,2)/5/
      DATA TPRID(-2,2)/0/
      DATA (IFOREST(I,-3,2),I=1,2)/4,-2/
      DATA SPROP(-3,2)/6/
      DATA TPRID(-3,2)/0/
C     Diagram 3
      DATA MAPCONFIG(3)/3/
      DATA (IFOREST(I,-1,3),I=1,2)/4,3/
      DATA SPROP(-1,3)/6/
      DATA TPRID(-1,3)/0/
      DATA (IFOREST(I,-2,3),I=1,2)/-1,2/
      DATA SPROP(-2,3)/6/
      DATA TPRID(-2,3)/0/
      DATA (IFOREST(I,-3,3),I=1,2)/5,-2/
      DATA SPROP(-3,3)/6/
      DATA TPRID(-3,3)/0/
C     Diagram 4
      DATA MAPCONFIG(4)/4/
      DATA (IFOREST(I,-1,4),I=1,2)/4,3/
      DATA SPROP(-1,4)/6/
      DATA TPRID(-1,4)/0/
      DATA (IFOREST(I,-2,4),I=1,2)/5,-1/
      DATA SPROP(-2,4)/6/
      DATA TPRID(-2,4)/0/
      DATA (IFOREST(I,-3,4),I=1,2)/-2,2/
      DATA SPROP(-3,4)/6/
      DATA TPRID(-3,4)/0/
C     Diagram 5
      DATA MAPCONFIG(5)/5/
      DATA (IFOREST(I,-1,5),I=1,2)/5,3/
      DATA SPROP(-1,5)/5/
      DATA TPRID(-1,5)/0/
      DATA (IFOREST(I,-2,5),I=1,2)/-1,2/
      DATA SPROP(-2,5)/5/
      DATA TPRID(-2,5)/0/
      DATA (IFOREST(I,-3,5),I=1,2)/4,-2/
      DATA SPROP(-3,5)/6/
      DATA TPRID(-3,5)/0/
C     Diagram 6
      DATA MAPCONFIG(6)/6/
      DATA (IFOREST(I,-1,6),I=1,2)/5,3/
      DATA SPROP(-1,6)/5/
      DATA TPRID(-1,6)/0/
      DATA (IFOREST(I,-2,6),I=1,2)/4,-1/
      DATA SPROP(-2,6)/6/
      DATA TPRID(-2,6)/0/
      DATA (IFOREST(I,-3,6),I=1,2)/-2,2/
      DATA SPROP(-3,6)/6/
      DATA TPRID(-3,6)/0/
C     Number of configs
      DATA MAPCONFIG(0)/6/
""")

    def test_configs_8fs(self):
        """Test configs.inc for 8fs process which previously failed.
        """

        diagrams = save_load_object.load_from_file(\
                                              os.path.join(_input_file_path,
                                                           'test_8fs.pkl'))

        goal_schannels = [[[8, 6, -1], [7, -1, -2], [-2, 5, -3],
                           [-3, 3, -4], [4, -4, -5]],
                          [],
                          [[6, 5, -1], [8, -1, -2], [7, -2, -3], [-3, 3, -4]],
                          [[6, 5, -1]]]
        goal_tchannels = [[[1, -5, -6]],
                          [[1, 4, -1], [-1, 7, -2], [-2, 3, -3],
                           [-3, 6, -4], [-4, 5, -5], [-5, 8, -6]],
                          [[1, 4, -5], [-5, -4, -6]],
                          [[1, 4, -2], [-2, 7, -3], [-3, 8, -4],
                           [-4, 3, -5], [-5, -1, -6]]]
                          

        for (idiag, diagram) in enumerate(diagrams):

            schannels, tchannels = diagram.get('amplitudes')[0].\
                                         get_s_and_t_channels(2)

            self.assertEqual([[l.get('number') for l in v.get('legs')] for v \
                              in schannels],
                             goal_schannels[idiag])
            self.assertEqual([[l.get('number') for l in v.get('legs')] for v \
                              in tchannels],
                             goal_tchannels[idiag]) 

class AlohaFortranWriterTest(unittest.TestCase):
    """ A basic test to see if the Aloha Fortran Writter is working """
    
    def setUp(self):
        """ check that old file are remove """
        try:
            os.remove('/tmp/FFV1_1.f')
        except:
            pass
    
    def test_header(self):
        """ test the header of a file """
        
        from models.sm.object_library import Lorentz
        import aloha.create_aloha as create_aloha
        
        FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')
        
        solution="""C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Gamma(3,2,1)
C     
      SUBROUTINE FFV1_1(F2, V3, C, M1, W1, F1)
      IMPLICIT NONE
      DOUBLE COMPLEX F1(6)
      DOUBLE COMPLEX F2(6)
      DOUBLE COMPLEX V3(6)
      DOUBLE COMPLEX C
      DOUBLE COMPLEX DENOM
      DOUBLE PRECISION M1, W1
      DOUBLE PRECISION P1(0:3)

      F1(5)= F2(5)+V3(5)
      F1(6)= F2(6)+V3(6)
      P1(0) =  DBLE(F1(5))
      P1(1) =  DBLE(F1(6))
      P1(2) =  DIMAG(F1(6))
      P1(3) =  DIMAG(F1(5))"""

        abstract_M = create_aloha.AbstractRoutineBuilder(FFV1).compute_routine(1)
        abstract_M.add_symmetry(2)
        abstract_M.write('/tmp','Fortran')
        
        self.assertTrue(os.path.exists('/tmp/FFV1_1.f'))
        textfile = open('/tmp/FFV1_1.f','r')
        split_sol = solution.split('\n')
        for i in range(len(split_sol)):
            self.assertEqual(split_sol[i]+'\n', textfile.readline())


class UFO_model_to_mg4_Test(unittest.TestCase):
    """Check the conversion model from UFO to MG4"""
    
    def setUp(self):
        import models.import_ufo
        self.mymodel = models.import_ufo.import_model('sm')
    
    
    def test_refactorize(self):
        """ test the separation of variable """
        
        mg4_model = export_v4.UFO_model_to_mg4(self.mymodel,'/dev/null')
        mg4_model.refactorize()
        
        # external parameters
        expected = ['aEWM1', 'Gf', 'aS', 'ymb', 'ymt', 'ymtau', 'MTA', 'MT', 'MB', 'MZ', 'MH', 'WT', 'WZ', 'WW', 'WH']
        expected.sort()
        solution = [param.name for param in mg4_model.params_ext]
        solution.sort()
        self.assertEqual(expected, solution)
        
        #  internal params
        self.assertEqual(len(mg4_model.params_dep), 1)
        self.assertEqual(len(mg4_model.params_indep), 32)
        
        # couplings
        self.assertEqual(len(mg4_model.coups_dep), 3)
        sol= ['GC_1', 'GC_2', 'GC_3', 'GC_5', 'GC_6', 'GC_7', 'GC_8', 'GC_13', 'GC_14', 'GC_15', 'GC_16', 'GC_17', 'GC_18', 'GC_19', 'GC_20', 'GC_21', 'GC_22', 'GC_23', 'GC_33', 'GC_34', 'GC_35', 'GC_36', 'GC_37', 'GC_38', 'GC_39', 'GC_40', 'GC_41', 'GC_42', 'GC_43', 'GC_44', 'GC_45', 'GC_46', 'GC_47', 'GC_48', 'GC_49', 'GC_50', 'GC_51', 'GC_52', 'GC_53', 'GC_54', 'GC_55', 'GC_56', 'GC_57', 'GC_58', 'GC_59', 'GC_60', 'GC_61', 'GC_62', 'GC_63', 'GC_64', 'GC_67', 'GC_68', 'GC_69', 'GC_72', 'GC_96', 'GC_97', 'GC_100', 'GC_101', 'GC_102', 'GC_103', 'GC_104', 'GC_135', 'GC_136']
        
        self.assertEqual(sol, [ p.name for p in mg4_model.coups_indep])

        
        # MG4 use G and not aS as it basic object for alphas related computation
        #Pass G in the  independant list
        self.assertTrue('G' not in [p.name for p in mg4_model.params_dep])
        self.assertTrue('G' in [p.name for p in mg4_model.params_indep])
        self.assertTrue('sqrt__aS' not in [p.name for p in mg4_model.params_dep])
        self.assertTrue('sqrt__aS' in [p.name for p in mg4_model.params_indep])
        
        
    def test_case_sensitive(self):
        """ test that the case clash are dealt correctly """  
        
        mg4_model = export_v4.UFO_model_to_mg4(self.mymodel,'/dev/null')
        
        #check that they are no crash for normal model
        mg4_model.pass_parameter_to_case_insensitive()
        
        # edit model in order to add new parameter with name: CW / Cw / Mz / Mz2
        CW = base_objects.ModelVariable( 'CW', 'Mz**2 * Mz2' , 'real')
        Cw = base_objects.ModelVariable( 'Cw', 'Mz**2 * Mz2 * CW' , 'real')
        Mz = base_objects.ParamCardVariable('Mz', 100, 'MASS', 41)
        Mz2 = base_objects.ParamCardVariable('Mz2', 100, 'MASS', 43)
        
        mg4_model.model['parameters'][()].append(CW)
        mg4_model.model['parameters'][()].append(Cw)
        mg4_model.model['parameters'][('external',)].append(Mz)
        mg4_model.model['parameters'][('external',)].append(Mz2)


        mg4_model.pass_parameter_to_case_insensitive()

        self.assertEqual(CW.name,'cw__2')
        self.assertEqual(CW.expr,'mz__2**2 * Mz2')
        self.assertEqual(Cw.name,'cw__3')
        self.assertEqual(Cw.expr,'mz__2**2 * Mz2 * cw__2')
        
        self.assertEqual(Mz.name,'mz__2')
        


if __name__ == '__main__':
        """Write out pkl file with helas diagram for test_configs_8fs
        """

        import models.import_ufo
        mymodel = models.import_ufo.import_model('sm')
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':False}))
        myleglist.append(base_objects.Leg({'id':1}))
        myleglist.append(base_objects.Leg({'id':24}))
        myleglist.append(base_objects.Leg({'id':5}))
        myleglist.append(base_objects.Leg({'id':-5}))
        myleglist.append(base_objects.Leg({'id':21}))
        myleglist.append(base_objects.Leg({'id':21}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mymodel})
        myamplitude = diagram_generation.Amplitude({'process': myproc})

        me = helas_objects.HelasMatrixElement(myamplitude,
                                              gen_color=False)

        import madgraph.iolibs.drawing_eps as draw
        filename = os.path.join('diagrams_' + \
                                myamplitude.get('process').shell_string() + ".eps")
        plot = draw.MultiEpsDiagramDrawer(myamplitude.get('diagrams'),
                                          filename,
                                          model=mymodel,
                                                amplitude='',
                                          legend=myamplitude.get('process').input_string())

        plot.draw()  



        me = save_load_object.save_to_file(\
                       os.path.join(_input_file_path, 'test_8fs.pkl'),
                       [me.get('diagrams')[323], me.get('diagrams')[954],
                        me.get('diagrams')[1123], me.get('diagrams')[1139]])
        
        
        


    
    
    



