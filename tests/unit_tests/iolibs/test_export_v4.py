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

import tests.unit_tests as unittest

import madgraph.iolibs.misc as misc
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import tests.unit_tests.iolibs.test_file_writers as test_file_writers

#===============================================================================
# IOImportV4Test
#===============================================================================
class IOExportV4Test(unittest.TestCase,
                     test_file_writers.CheckFileCreate):
    """Test class for the export v4 module"""

    mymodel = base_objects.Model()
    mymatrixelement = helas_objects.HelasMatrixElement()
    myfortranmodel = helas_call_writers.FortranHelasCallWriter()
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

        export_v4.write_matrix_element_v4_standalone(\
            writers.FortranWriter(self.give_pos('test')),
            self.mymatrixelement,
            self.myfortranmodel)

        self.assertFileContains('test', goal_matrix_f)

    def test_coeff_string(self):
        """Test the coeff string for JAMP lines"""

        self.assertEqual(export_v4.coeff(1,
                                         fractions.Fraction(1),
                                         False, 0), '+')

        self.assertEqual(export_v4.coeff(-1,
                                         fractions.Fraction(1),
                                         False, 0), '-')

        self.assertEqual(export_v4.coeff(-1,
                                         fractions.Fraction(-3),
                                         False, 0), '+3*')

        self.assertEqual(export_v4.coeff(-1,
                                         fractions.Fraction(3, 5),
                                         True, -2), '-1./15.*complex(0,1)*')

