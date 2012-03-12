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

"""Unit test library for the export realfks format routines"""

import StringIO
import copy
import fractions
import os 
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.export_fks_born as export_fks_born
import madgraph.fks.fks_born as fks_born
import madgraph.fks.fks_born_helas_objects as fks_born_helas
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.save_load_object as save_load_object        
import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_algebra as color
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.process_checks as process_checks
import madgraph.core.color_amp as color_amp
import tests.unit_tests.core.test_helas_objects as test_helas_objects
import tests.unit_tests.iolibs.test_file_writers as test_file_writers
import tests.unit_tests.iolibs.test_helas_call_writers as \
                                            test_helas_call_writers
import models.import_ufo as import_ufo

_file_path = os.path.dirname(os.path.realpath(__file__))
_input_file_path = os.path.join(_file_path, os.path.pardir, os.path.pardir,
                                'input_files')
#===============================================================================
# IOImportV4Test
#===============================================================================
class IOExportBornFKSTest(unittest.TestCase,
                     test_file_writers.CheckFileCreate):
    """Test class for the export bornfks module"""

    mymatrixelement = helas_objects.HelasMatrixElement()
    created_files = ['test'
                    ]

    def setUp(self):
        self.mymodel = import_ufo.import_model('sm')
        self.myfortranmodel = helas_call_writers.FortranUFOHelasCallWriter(self.mymodel)
    
        myleglist = MG.MultiLegList()
        
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[21], 'state':False}))
        myleglist.append(MG.MultiLeg({'ids':[2], 'state':True}))
        myleglist.append(MG.MultiLeg({'ids':[21], 'state':True}))
    
        myproc = MG.ProcessDefinition({'legs': myleglist,
                             'model': self.mymodel,
                             'orders':{'QCD': 2, 'QED': 0},
                             'perturbation_couplings': ['QCD'],
                             'NLO_mode': 'real'})
        my_process_definitions = MG.ProcessDefinitionList([myproc])

        self.myfksmulti = fks_born.FKSMultiProcessFromBorn(\
                {'process_definitions': my_process_definitions})
        
        self.myfks_me = fks_born_helas.FKSHelasMultiProcessFromBorn(\
                self.myfksmulti)['matrix_elements'][0]

        #self.myfortranmodel.downcase = False

        tearDown = test_file_writers.CheckFileCreate.clean_files

    def test_get_fks_conf_lines_B(self):
        """Checks the lines corresponding to each fks confs, to be 
        written in fks.inc
        """
        lines_list = [
"""c     FKS configuration number  1
DATA FKS_I(1) / 5 /
DATA FKS_J(1) / 1 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 3 /
DATA FKS_J(1) / 1 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 5 /
DATA FKS_J(1) / 2 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 3 /
DATA FKS_J(1) / 2 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 3 /
DATA FKS_J(1) / 2 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 3 /
DATA FKS_J(1) / 2 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 4 /
DATA FKS_J(1) / 2 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 5 /
DATA FKS_J(1) / 3 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 5 /
DATA FKS_J(1) / 4 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 5 /
DATA FKS_J(1) / 4 /
""",
"""c     FKS configuration number  1
DATA FKS_I(1) / 5 /
DATA FKS_J(1) / 4 /
"""]
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        for lines, real in zip(lines_list, self.myfks_me.real_processes):
            self.assertEqual(lines, process_exporter.get_fks_conf_lines(real))

    def test_get_fks_j_from_i_lines_B(self):
        """Test that the lines corresponding to the fks_j_from_i array, to be 
        written in fks.inc. 
        The real process is gg_uxug_i3_j1 (myfks_me.real_processes[1])
        """
        lines = \
"""DATA (FKS_J_FROM_I(3, JPOS), JPOS = 0, 3)  / 3, 1, 2, 4 /
DATA (FKS_J_FROM_I(4, JPOS), JPOS = 0, 2)  / 2, 1, 2 /
DATA (FKS_J_FROM_I(5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        self.assertEqual(lines, process_exporter.get_fks_j_from_i_lines(self.myfks_me.real_processes[1]))


    def test_write_fks_inc_B(self):
        """Tests the correct writing of the fks.inc file, containing informations 
        for all the different fks configurations
        The real process is gg_uxug_i3_j1 (myfks_me.real_processes[1])
        """
        goal = \
"""      INTEGER FKS_CONFIGS, IPOS, JPOS
      DATA FKS_CONFIGS / 1 /
      INTEGER FKS_I(1), FKS_J(1)
      INTEGER FKS_J_FROM_I(NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE(NEXTERNAL), PDG_TYPE(NEXTERNAL)

C     FKS configuration number  1
      DATA FKS_I(1) / 3 /
      DATA FKS_J(1) / 1 /

      DATA (FKS_J_FROM_I(3, JPOS), JPOS = 0, 3)  / 3, 1, 2, 4 /
      DATA (FKS_J_FROM_I(4, JPOS), JPOS = 0, 2)  / 2, 1, 2 /
      DATA (FKS_J_FROM_I(5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /
C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE(IPOS), IPOS=1, NEXTERNAL) / 8, 8, -3, 3, 8 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE(IPOS), IPOS=1, NEXTERNAL) / 21, 21, -2, 2, 21 /

"""

        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        process_exporter.write_fks_inc(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me.real_processes[1],
            self.myfortranmodel)
        self.assertFileContains('test', goal)


    def test_write_mirrorprocs_B(self):
        """Tests the correct writing of the mirrorprocs.inc file"""
        goal = \
"""      LOGICAL MIRRORPROC
      DATA MIRRORPROC /.FALSE./
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        process_exporter.write_mirrorprocs(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me)
        self.assertFileContains('test', goal)


    def test_write_lh_order_B(self):
        """tests the correct writing of the B-LH order file"""

        goal = \
"""#OLE_order written by MadGraph 5

MatrixElementSquareType CHsummed
CorrectionType          QCD
IRregularisation        CDR
AlphasPower             2
AlphaPower              0
NJetSymmetrizeFinal     Yes

# process
2 21 -> 2 21 
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        process_exporter.write_lh_order(\
            self.give_pos('test'),\
            self.myfks_me)
        self.assertFileContains('test', goal)


    def test_get_pdf_lines_mir_false_B(self):
        """tests the correct writing of the pdf lines for a non-mirror configuration,
        i.e. with beam indices 1,2 in the usual position.
        The real process used is uux_uxug (real_processes[5])
        """
        lines = \
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
IPROC=IPROC+1 ! u u~ > u~ u g
PD(IPROC)=PD(IPROC-1) + u1*ub2"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        self.assertEqual(lines, 
                         process_exporter.get_pdf_lines_mir( \
                             self.myfks_me.real_processes[5].matrix_element, 2, False, False))

    def test_get_pdf_lines_mir_true_B(self):
        """tests the correct writing of the pdf lines for a mirror configuration,
        i.e. with exchanged beam indices 1,2D0
        The real process used is uux_uxug (real_processes[5])
        """
        lines = \
"""IF (ABS(LPP(2)) .GE. 1) THEN
LP=SIGN(1,LPP(2))
u1=PDG2PDF(ABS(LPP(2)),2*LP,XBK(2),DSQRT(Q2FACT(2)))
ENDIF
IF (ABS(LPP(1)) .GE. 1) THEN
LP=SIGN(1,LPP(1))
ub2=PDG2PDF(ABS(LPP(1)),-2*LP,XBK(1),DSQRT(Q2FACT(1)))
ENDIF
PD(0) = 0d0
IPROC = 0
IPROC=IPROC+1 ! u u~ > u~ u g
PD(IPROC)=PD(IPROC-1) + u1*ub2"""

        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        self.assertEqual(lines, 
                         process_exporter.get_pdf_lines_mir( \
                             self.myfks_me.real_processes[5].matrix_element, 2, False, True))

    def test_write_sborn_sf_dum_B(self):
        """Tests the correct writing of the sborn_sf file, containing the calls 
        to the different color linked borns. In this case the real process has no 
        soft singularities, so a dummy function is written"""
        
        goal = \
"""      SUBROUTINE SBORN_SF(P_BORN,M,N,WGT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_BORN(0:3,NEXTERNAL-1),WGT
      DOUBLE COMPLEX WGT1(2)
      INTEGER M,N

C     This is a dummy function because
C     this subdir has no soft singularities
      WGT = 0D0

      RETURN
      END
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        process_exporter.write_sborn_sf(\
            writers.FortranWriter(self.give_pos('test')),
            [],
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)

    def test_write_sborn_sf_B(self):
        """Tests the correct writing of the sborn_sf file, containing the calls 
        to the different color linked borns."""
        
        goal = \
"""      SUBROUTINE SBORN_SF(P_BORN,M,N,WGT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_BORN(0:3,NEXTERNAL-1),WGT
      DOUBLE COMPLEX WGT1(2)
      INTEGER M,N

C     b_sf_001 links partons 1 and 2 
      IF (M.EQ.1 .AND. N.EQ.2) THEN
        CALL SB_SF_001(P_BORN,WGT)

C       b_sf_002 links partons 1 and 3 
      ELSEIF (M.EQ.1 .AND. N.EQ.3) THEN
        CALL SB_SF_002(P_BORN,WGT)

C       b_sf_003 links partons 1 and 4 
      ELSEIF (M.EQ.1 .AND. N.EQ.4) THEN
        CALL SB_SF_003(P_BORN,WGT)

C       b_sf_004 links partons 2 and 1 
      ELSEIF (M.EQ.2 .AND. N.EQ.1) THEN
        CALL SB_SF_004(P_BORN,WGT)

C       b_sf_005 links partons 2 and 3 
      ELSEIF (M.EQ.2 .AND. N.EQ.3) THEN
        CALL SB_SF_005(P_BORN,WGT)

C       b_sf_006 links partons 2 and 4 
      ELSEIF (M.EQ.2 .AND. N.EQ.4) THEN
        CALL SB_SF_006(P_BORN,WGT)

C       b_sf_007 links partons 3 and 1 
      ELSEIF (M.EQ.3 .AND. N.EQ.1) THEN
        CALL SB_SF_007(P_BORN,WGT)

C       b_sf_008 links partons 3 and 2 
      ELSEIF (M.EQ.3 .AND. N.EQ.2) THEN
        CALL SB_SF_008(P_BORN,WGT)

C       b_sf_009 links partons 3 and 4 
      ELSEIF (M.EQ.3 .AND. N.EQ.4) THEN
        CALL SB_SF_009(P_BORN,WGT)

C       b_sf_010 links partons 4 and 1 
      ELSEIF (M.EQ.4 .AND. N.EQ.1) THEN
        CALL SB_SF_010(P_BORN,WGT)

C       b_sf_011 links partons 4 and 2 
      ELSEIF (M.EQ.4 .AND. N.EQ.2) THEN
        CALL SB_SF_011(P_BORN,WGT)

C       b_sf_012 links partons 4 and 3 
      ELSEIF (M.EQ.4 .AND. N.EQ.3) THEN
        CALL SB_SF_012(P_BORN,WGT)

      ELSE
        WGT = 0D0
      ENDIF

      RETURN
      END
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        process_exporter.write_sborn_sf(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me.color_links,
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)


    def test_write_matrix_element_fks_B(self):
        """Tests the correct writing of the matrix.f file, containing
        the real emission matrix element.
        The real emission process is ud_dug (real_processes[4]).
        """
        goal = \
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
C     Process: u d > d u g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     Process: u s > s u g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     Process: u c > c u g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'nexternal.inc'
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
      DATA IDEN/36/
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
C     Process: u d > d u g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     Process: u s > s u g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     Process: u c > c u g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=5)
      INCLUDE 'nexternal.inc'
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=11, NCOLOR=4)
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
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0D0, 0D0), (1D0, 0D0)/
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'coupl.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/1/
      DATA (CF(I,  1),I=  1,  4) /   12,    4,    4,    0/
C     1 T(3,1) T(5,4,2)
      DATA DENOM(2)/1/
      DATA (CF(I,  2),I=  1,  4) /    4,   12,    0,    4/
C     1 T(3,2) T(5,4,1)
      DATA DENOM(3)/1/
      DATA (CF(I,  3),I=  1,  4) /    4,    0,   12,    4/
C     1 T(4,1) T(5,3,2)
      DATA DENOM(4)/1/
      DATA (CF(I,  4),I=  1,  4) /    0,    4,    4,   12/
C     1 T(4,2) T(5,3,1)
C     ----------
C     BEGIN CODE
C     ----------
      CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1*IC(1),W(1,1))
      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1*IC(2),W(1,2))
      CALL OXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
      CALL OXXXXX(P(0,4),ZERO,NHEL(4),+1*IC(4),W(1,4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1*IC(5),W(1,5))
      CALL FFV1_3(W(1,1),W(1,4),GC_5,ZERO, ZERO, W(1,6))
      CALL FFV1_3(W(1,2),W(1,3),GC_5,ZERO, ZERO, W(1,7))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(1,6),W(1,7),W(1,5),GC_4,AMP(1))
      CALL FFV1_2(W(1,2),W(1,5),GC_5,ZERO, ZERO, W(1,8))
C     Amplitude(s) for diagram number 2
      CALL FFV1_0(W(1,8),W(1,3),W(1,6),GC_5,AMP(2))
      CALL FFV1_1(W(1,3),W(1,5),GC_5,ZERO, ZERO, W(1,9))
C     Amplitude(s) for diagram number 3
      CALL FFV1_0(W(1,2),W(1,9),W(1,6),GC_5,AMP(3))
      CALL FFV1_2(W(1,1),W(1,5),GC_5,ZERO, ZERO, W(1,10))
C     Amplitude(s) for diagram number 4
      CALL FFV1_0(W(1,10),W(1,4),W(1,7),GC_5,AMP(4))
      CALL FFV1_1(W(1,4),W(1,5),GC_5,ZERO, ZERO, W(1,11))
C     Amplitude(s) for diagram number 5
      CALL FFV1_0(W(1,1),W(1,11),W(1,7),GC_5,AMP(5))
      JAMP(1)=+1D0/2D0*(+IMAG1*AMP(1)+AMP(2)+AMP(5))
      JAMP(2)=+1D0/2D0*(-1D0/3D0*AMP(4)-1D0/3D0*AMP(5))
      JAMP(3)=+1D0/2D0*(-1D0/3D0*AMP(2)-1D0/3D0*AMP(3))
      JAMP(4)=+1D0/2D0*(-IMAG1*AMP(1)+AMP(3)+AMP(4))

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
        
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        process_exporter.write_matrix_element_fks(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me.real_processes[4].matrix_element,
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)
        

    def test_write_born_fks_no_ijglu_B(self):
        """Tests the correct writing of the born.f file, when the particle
        which splits into i and j is not a gluon."""
        
        goal = \
"""      SUBROUTINE SBORN(P1,ANS)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     AND HELICITIES
C     FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C     
C     
C     BORN AMPLITUDE IS 
C     Process: u g > u g WEIGHTED=2 QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
      INTEGER                 NCOMB,     NCROSS
      PARAMETER (             NCOMB=  16, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   3)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(NCROSS*2)
C     
C     LOCAL VARIABLES 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY
      COMPLEX*16 T,T1
      REAL*8 BORN
      REAL*8 ZERO
      PARAMETER(ZERO=0D0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGOOD,IGOOD(NCOMB),JHEL
      DATA NGOOD /0/
      SAVE IGOOD,JHEL
      REAL*8 HWGT
      REAL*8 XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
      INTEGER IDUM, J, JJ
      LOGICAL WARNED
      REAL     XRAN1
      EXTERNAL XRAN1
C     
C     GLOBAL VARIABLES
C     
C     Double Precision amp2(bmaxamps), jamp2(0:bmaxamps)
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXAMPS)
      COMMON/TO_AMPS/  AMP2,       JAMP2

      INCLUDE 'born_nhel.inc'
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM

      CHARACTER*79         HEL_BUFF(2)
      COMMON/TO_HELICITY/  HEL_BUFF

      REAL*8 POL(2)
      COMMON/TO_POLARIZATION/ POL

      INTEGER          ISUM_HEL
      LOGICAL                    MULTI_CHANNEL
      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
      DATA NTRY,IDUM /0,-1/
      DATA XTRY, XREJ /0,0/
C     DATA warned, isum_hel/.false.,0/
C     DATA multi_channel/.true./
      SAVE YFRAC
      DATA JAMP2(0) /   1/
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
      INCLUDE 'den_factor.inc'
      DOUBLE PRECISION HEL_FAC
      LOGICAL CALCULATEDBORN
      INTEGER GET_HEL,SKIP
      COMMON/CBORN/HEL_FAC,CALCULATEDBORN,GET_HEL,SKIP
C     ----------
C     BEGIN CODE
C     ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
        DO IHEL=1,NEXTERNAL-1
          JC(IHEL) = +1
        ENDDO

        IF (MULTI_CHANNEL) THEN
          DO IHEL=1,NGRAPHS
            AMP2(IHEL)=0D0
            JAMP2(IHEL)=0D0
          ENDDO
          DO IHEL=1,INT(JAMP2(0))
            JAMP2(IHEL)=0D0
          ENDDO
        ENDIF
        IF (CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $       ,J)) THEN
              CALCULATEDBORN=.FALSE.
C             write (*,*) "momenta not the same in Born"
            ENDIF
          ENDDO
        ENDIF
        IF (.NOT.CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            SAVEMOM(J,1)=P1(0,J)
            SAVEMOM(J,2)=P1(3,J)
          ENDDO
          DO J=1,MAX_BHEL
            DO JJ=1,NGRAPHS
              SAVEAMP(JJ,J)=(0D0,0D0)
            ENDDO
          ENDDO
        ENDIF
        ANS(IPROC) = 0D0
        WRITE(HEL_BUFF(1),'(16i5)') (0,I=1,NEXTERNAL-1)
        IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 2) THEN
          HEL_FAC=1D0
          DO IHEL=1,NCOMB
            IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
              T=BORN(P1,NHEL(1,IHEL),IHEL,JC(1))
              DO JJ=1,NINCOMING
                IF(POL(JJ).NE.1D0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1D0
     $           ,POL(JJ)))) THEN
                  T=T*ABS(POL(JJ))
                ELSE IF(POL(JJ).NE.1D0)THEN
                  T=T*(2D0-ABS(POL(JJ)))
                ENDIF
              ENDDO
              ANS(IPROC)=ANS(IPROC)+T
              IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                GOODHEL(IHEL,IPROC)=.TRUE.
                NGOOD = NGOOD +1
                IGOOD(NGOOD) = IHEL
              ENDIF
            ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
        ELSE  !RANDOM HELICITY
          DO J=1,ISUM_HEL
            HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
            HEL_FAC=HWGT
            IF (GET_HEL.EQ.0) THEN
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              IHEL = IGOOD(JHEL)
              GET_HEL=IHEL
            ELSE
              IHEL=GET_HEL
            ENDIF
            IF(GOODHEL(IHEL,IPROC)) THEN
              T=BORN(P1,NHEL(1,IHEL),IHEL,JC(1))
              DO JJ=1,NINCOMING
                IF(POL(JJ).NE.1D0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1D0
     $           ,POL(JJ)))) THEN
                  T=T*ABS(POL(JJ))
                ELSE IF(POL(JJ).NE.1D0)THEN
                  T=T*(2D0-ABS(POL(JJ)))
                ENDIF
              ENDDO
              ANS(IPROC)=ANS(IPROC)+T*HWGT
            ENDIF
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
            WRITE(HEL_BUFF(1),'(16i5)')(NHEL(I,IHEL),I=1,NEXTERNAL-1)
          ENDIF
        ENDIF
        ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      CALCULATEDBORN=.TRUE.
      END


      REAL*8 FUNCTION BORN(P,NHEL,HELL,IC)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

C     Process: u g > u g WEIGHTED=2 QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS,    NEIGEN
      PARAMETER (NGRAPHS=   3,NEIGEN=  1)
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
C     INCLUDE 'born_maxamps.inc'
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=7, NCOLOR=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1), IC(NEXTERNAL-1), HELL
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1 = (0D0,1D0))
C     
C     GLOBAL VARIABLES
C     
C     Double Precision amp2(bmaxamps), jamp2(0:bmaxamps)
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXAMPS)
      COMMON/TO_AMPS/  AMP2,       JAMP2
      INCLUDE 'born_nhel.inc'
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION HEL_FAC
      LOGICAL CALCULATEDBORN
      INTEGER GET_HEL,SKIP
      COMMON/CBORN/HEL_FAC,CALCULATEDBORN,GET_HEL,SKIP
      INCLUDE 'coupl.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/3/
      DATA (CF(I,  1),I=  1,  2) /   16,   -2/
C     1 T(2,4,3,1)
      DATA DENOM(2)/3/
      DATA (CF(I,  2),I=  1,  2) /   -2,   16/
C     1 T(4,2,3,1)
C     ----------
C     BEGIN CODE
C     ----------
      IF (.NOT. CALCULATEDBORN) THEN
        CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1*IC(1),W(1,1))
        CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
        CALL OXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
        CALL VXXXXX(P(0,4),ZERO,NHEL(4),+1*IC(4),W(1,4))
        CALL FFV1_2(W(1,1),W(1,2),GC_5,ZERO, ZERO, W(1,5))
C       Amplitude(s) for diagram number 1
        CALL FFV1_0(W(1,5),W(1,3),W(1,4),GC_5,AMP(1))
        CALL FFV1_3(W(1,1),W(1,3),GC_5,ZERO, ZERO, W(1,6))
C       Amplitude(s) for diagram number 2
        CALL VVV1_0(W(1,2),W(1,4),W(1,6),GC_4,AMP(2))
        CALL FFV1_2(W(1,1),W(1,4),GC_5,ZERO, ZERO, W(1,7))
C       Amplitude(s) for diagram number 3
        CALL FFV1_0(W(1,7),W(1,3),W(1,2),GC_5,AMP(3))
        DO I=1,NGRAPHS
          SAVEAMP(I,HELL)=AMP(I)
        ENDDO
      ELSEIF (CALCULATEDBORN) THEN
        DO I=1,NGRAPHS
          AMP(I)=SAVEAMP(I,HELL)
        ENDDO
      ENDIF
      JAMP(1)=-IMAG1*AMP(2)+AMP(3)
      JAMP(2)=+AMP(1)+IMAG1*AMP(2)
      BORN = 0.D0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        BORN =BORN+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
      ENDDO
      AMP2(1)=AMP2(1)+AMP(1)*DCONJG(AMP(1))
      AMP2(2)=AMP2(2)+AMP(2)*DCONJG(AMP(2))
      AMP2(3)=AMP2(3)+AMP(3)*DCONJG(AMP(3))
      DO I = 1, NCOLOR
        JAMP2(I)=JAMP2(I)+JAMP(I)*DCONJG(JAMP(I))
      ENDDO
      END



""" % misc.get_pkg_info()
        
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        process_exporter.write_born_fks(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me.born_matrix_element, 0,
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)


    def test_write_born_fks_ijglu_B(self):
        """Tests the correct writing of the born.f file, when the particle
        which splits into i and j is a gluon.
        """
        goal = \
"""      SUBROUTINE SBORN(P1,ANS)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     AND HELICITIES
C     FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C     
C     
C     BORN AMPLITUDE IS 
C     Process: u g > u g WEIGHTED=2 QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
C     Include 'born_maxamps.inc'
      INTEGER                 NCOMB,     NCROSS
      PARAMETER (             NCOMB=  16, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   3)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(NCROSS*2)
C     
C     LOCAL VARIABLES 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY
      COMPLEX*16 T,T1
      REAL*8 BORN
      REAL*8 ZERO
      PARAMETER(ZERO=0D0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS)
      INTEGER NGOOD,IGOOD(NCOMB),JHEL
      DATA NGOOD /0/
      SAVE IGOOD,JHEL
      REAL*8 HWGT
      REAL*8 XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
      INTEGER IDUM, J, JJ
      LOGICAL WARNED
      REAL     XRAN1
      EXTERNAL XRAN1
C     
C     GLOBAL VARIABLES
C     
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXAMPS)
      COMMON/TO_AMPS/  AMP2,       JAMP2

      INCLUDE 'born_nhel.inc'
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM

      CHARACTER*79         HEL_BUFF(2)
      COMMON/TO_HELICITY/  HEL_BUFF

      REAL*8 POL(2)
      COMMON/TO_POLARIZATION/ POL

      INTEGER          ISUM_HEL
      LOGICAL                    MULTI_CHANNEL
      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
      DATA NTRY,IDUM /0,-1/
      DATA XTRY, XREJ /0,0/
C     DATA warned, isum_hel/.false.,0/
C     DATA multi_channel/.true./
      SAVE YFRAC
      DATA JAMP2(0) /   1/
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
      INCLUDE 'den_factor.inc'
      DOUBLE PRECISION HEL_FAC
      LOGICAL CALCULATEDBORN
      INTEGER GET_HEL,SKIP
      COMMON/CBORN/HEL_FAC,CALCULATEDBORN,GET_HEL,SKIP
      INTEGER GLU_IJ
      INCLUDE 'glu_ij.inc'
C     ----------
C     BEGIN CODE
C     ----------
      NTRY=NTRY+1
      IF (NTRY.LT.2) THEN
        SKIP=1
        DO WHILE(NHEL(GLU_IJ ,SKIP).NE.1)
          SKIP=SKIP+1
        ENDDO
        SKIP=SKIP-1
      ENDIF
      DO IPROC=1,NCROSS
        DO IHEL=1,NEXTERNAL-1
          JC(IHEL) = +1
        ENDDO

        IF (MULTI_CHANNEL) THEN
          DO IHEL=1,NGRAPHS
            AMP2(IHEL)=0D0
            JAMP2(IHEL)=0D0
          ENDDO
          DO IHEL=1,INT(JAMP2(0))
            JAMP2(IHEL)=0D0
          ENDDO
        ENDIF
        IF (CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $       ,J)) THEN
              CALCULATEDBORN=.FALSE.
C             write (*,*) "momenta not the same in Born"
            ENDIF
          ENDDO
        ENDIF
        IF (.NOT.CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            SAVEMOM(J,1)=P1(0,J)
            SAVEMOM(J,2)=P1(3,J)
          ENDDO
          DO J=1,MAX_BHEL
            DO JJ=1,NGRAPHS
              SAVEAMP(JJ,J)=(0D0,0D0)
            ENDDO
          ENDDO
        ENDIF
        ANS(IPROC) = 0D0
        ANS(IPROC+1) = 0D0
        WRITE(HEL_BUFF(1),'(16i5)') (0,I=1,NEXTERNAL-1)
        IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 2) THEN
          HEL_FAC=1D0
          DO IHEL=1,NCOMB
            IF ((GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2).AND.NHEL(GLU_IJ
     $        ,IHEL).EQ.-1) THEN
              T=BORN(P1,NHEL(1,IHEL),IHEL,JC(1),T1)
              DO JJ=1,NINCOMING
                IF(POL(JJ).NE.1D0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1D0
     $           ,POL(JJ)))) THEN
                  T=T*ABS(POL(JJ))
                  T1=T1*ABS(POL(JJ))
                ELSE IF(POL(JJ).NE.1D0)THEN
                  T=T*(2D0-ABS(POL(JJ)))
                  T1=T1*(2D0-ABS(POL(JJ)))
                ENDIF
              ENDDO
              ANS(IPROC)=ANS(IPROC)+T
              ANS(IPROC+1)=ANS(IPROC+1)+T1
              IF ( (T .NE. 0D0 .OR. T1 .NE. 0D0) .AND. .NOT. GOODHEL(IH
     $         EL,IPROC)) THEN
                GOODHEL(IHEL,IPROC)=.TRUE.
                NGOOD = NGOOD +1
                IGOOD(NGOOD) = IHEL
              ENDIF
            ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD)
        ELSE  !RANDOM HELICITY
          DO J=1,ISUM_HEL
            HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
            HEL_FAC=HWGT
            IF (GET_HEL.EQ.0) THEN
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD) JHEL=1
              IHEL = IGOOD(JHEL)
              GET_HEL=IHEL
            ELSE
              IHEL=GET_HEL
            ENDIF
            IF(GOODHEL(IHEL,IPROC)) THEN
              T=BORN(P1,NHEL(1,IHEL),IHEL,JC(1),T1)
              DO JJ=1,NINCOMING
                IF(POL(JJ).NE.1D0.AND. NHEL(JJ,IHEL).EQ.INT(SIGN(1D0
     $           ,POL(JJ)))) THEN
                  T=T*ABS(POL(JJ))
                  T1=T1*ABS(POL(JJ))
                ELSE IF(POL(JJ).NE.1D0)THEN
                  T=T*(2D0-ABS(POL(JJ)))
                  T1=T1*(2D0-ABS(POL(JJ)))
                ENDIF
              ENDDO
              ANS(IPROC)=ANS(IPROC)+T*HWGT
              ANS(IPROC+1)=ANS(IPROC+1)+T1*HWGT
            ENDIF
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
            WRITE(HEL_BUFF(1),'(16i5)')(NHEL(I,IHEL),I=1,NEXTERNAL-1)
            WRITE(HEL_BUFF(2),'(16i5)')(NHEL(I,IHEL+SKIP),I=1
     $       ,NEXTERNAL-1)
          ENDIF
        ENDIF
        ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
        ANS(IPROC+1)=ANS(IPROC+1)/DBLE(IDEN(IPROC))
      ENDDO
      CALCULATEDBORN=.TRUE.
      END


      REAL*8 FUNCTION BORN(P,NHEL,HELL,IC,BORNTILDE)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

C     Process: u g > u g WEIGHTED=2 QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS,    NEIGEN
      PARAMETER (NGRAPHS=   3,NEIGEN=  1)
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
C     INCLUDE 'born_maxamps.inc'
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=7, NCOLOR=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1), IC(NEXTERNAL-1), HELL
      COMPLEX *16 BORNTILDE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
      COMPLEX*16 IMAG1
      INTEGER IHEL, BACK_HEL
      PARAMETER (IMAG1 = (0D0,1D0))
      COMPLEX *16 JAMPH(-1:1, NCOLOR)
C     
C     GLOBAL VARIABLES
C     
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXAMPS)
      COMMON/TO_AMPS/  AMP2,       JAMP2
      INCLUDE 'born_nhel.inc'
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION HEL_FAC
      LOGICAL CALCULATEDBORN
      INTEGER GET_HEL,SKIP
      COMMON/CBORN/HEL_FAC,CALCULATEDBORN,GET_HEL,SKIP
      INCLUDE 'coupl.inc'
      INTEGER GLU_IJ
      INCLUDE 'glu_ij.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/3/
      DATA (CF(I,  1),I=  1,  2) /   16,   -2/
C     1 T(2,4,3,1)
      DATA DENOM(2)/3/
      DATA (CF(I,  2),I=  1,  2) /   -2,   16/
C     1 T(4,2,3,1)
C     ----------
C     BEGIN CODE
C     ----------
      BORN = 0D0
      BORNTILDE = (0D0,0D0)
      BACK_HEL = NHEL(GLU_IJ)
      DO IHEL=-1,1,2
        NHEL(GLU_IJ) = IHEL
        IF (.NOT. CALCULATEDBORN) THEN
          CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1*IC(1),W(1,1))
          CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
          CALL OXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
          CALL VXXXXX(P(0,4),ZERO,NHEL(4),+1*IC(4),W(1,4))
          CALL FFV1_2(W(1,1),W(1,2),GC_5,ZERO, ZERO, W(1,5))
C         Amplitude(s) for diagram number 1
          CALL FFV1_0(W(1,5),W(1,3),W(1,4),GC_5,AMP(1))
          CALL FFV1_3(W(1,1),W(1,3),GC_5,ZERO, ZERO, W(1,6))
C         Amplitude(s) for diagram number 2
          CALL VVV1_0(W(1,2),W(1,4),W(1,6),GC_4,AMP(2))
          CALL FFV1_2(W(1,1),W(1,4),GC_5,ZERO, ZERO, W(1,7))
C         Amplitude(s) for diagram number 3
          CALL FFV1_0(W(1,7),W(1,3),W(1,2),GC_5,AMP(3))
          DO I=1,NGRAPHS
            IF(IHEL.EQ.-1)THEN
              SAVEAMP(I,HELL)=AMP(I)
            ELSEIF(IHEL.EQ.1)THEN
              SAVEAMP(I,HELL+SKIP)=AMP(I)
            ELSE
              WRITE(*,*) 'ERROR #1 in born.f'
              STOP
            ENDIF
          ENDDO
        ELSEIF (CALCULATEDBORN) THEN
          DO I=1,NGRAPHS
            IF(IHEL.EQ.-1)THEN
              AMP(I)=SAVEAMP(I,HELL)
            ELSEIF(IHEL.EQ.1)THEN
              AMP(I)=SAVEAMP(I,HELL+SKIP)
            ELSE
              WRITE(*,*) 'ERROR #1 in born.f'
              STOP
            ENDIF
          ENDDO
        ENDIF
        JAMP(1)=-IMAG1*AMP(2)+AMP(3)
        JAMP(2)=+AMP(1)+IMAG1*AMP(2)
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
            ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          BORN =BORN+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
        ENDDO
        DO I = 1, NGRAPHS
          AMP2(I)=AMP2(I)+AMP(I)*DCONJG(AMP(I))
        ENDDO
        DO I = 1, NCOLOR
          JAMP2(I)=JAMP2(I)+JAMP(I)*DCONJG(JAMP(I))
          JAMPH(IHEL,I)=JAMP(I)
        ENDDO

      ENDDO
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMPH(1,J)
        ENDDO
        BORNTILDE = BORNTILDE + ZTEMP*DCONJG(JAMPH(-1,I))/DENOM(I)
      ENDDO
      NHEL(GLU_IJ) = BACK_HEL
      END



""" % misc.get_pkg_info()
        
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        process_exporter.write_born_fks(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me.born_matrix_element, 1,
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)


    def test_write_b_sf_fks_B(self):
        """Tests the correct writing of a b_sf_xxx.f file, containing one color
        linked born.
        """
        goal = \
"""      SUBROUTINE SB_SF_001(P1,ANS)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     AND HELICITIES
C     FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL-1)
C     
C     
C     BORN AMPLITUDE IS 
C     Process: u g > u g WEIGHTED=2 QED=0 QCD=2 [ QCD ]
C     spectators: 1 2 

C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'nexternal.inc'
      INTEGER                 NCOMB,     NCROSS
      PARAMETER (             NCOMB=  16, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   3)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1),ANS(NCROSS)
C     
C     LOCAL VARIABLES 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY
      REAL*8 T
      REAL*8 B_SF_001
      REAL*8 ZERO
      PARAMETER(ZERO=0D0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS)
      DATA NTRY/0/
      INTEGER NGOOD,IGOOD(NCOMB),JHEL
      DATA NGOOD /0/
      SAVE IGOOD,JHEL
      REAL*8 HWGT
      INTEGER J,JJ
      INCLUDE 'born_nhel.inc'
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM

      CHARACTER*79         HEL_BUFF(2)
      COMMON/TO_HELICITY/  HEL_BUFF

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
      INCLUDE 'den_factor.inc'
      DOUBLE PRECISION HEL_FAC
      LOGICAL CALCULATEDBORN
      INTEGER GET_HEL,SKIP
      COMMON/CBORN/HEL_FAC,CALCULATEDBORN,GET_HEL,SKIP
C     ----------
C     BEGIN CODE
C     ----------
      NTRY=NTRY+1
      DO IPROC=1,NCROSS
        DO IHEL=1,NEXTERNAL-1
          JC(IHEL) = +1
        ENDDO
        IF (CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $       ,J)) THEN
              CALCULATEDBORN=.FALSE.
C             write (*,*) "momenta not the same in Born"
            ENDIF
          ENDDO
        ENDIF
        IF (.NOT.CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            SAVEMOM(J,1)=P1(0,J)
            SAVEMOM(J,2)=P1(3,J)
          ENDDO
          DO J=1,MAX_BHEL
            DO JJ=1,NGRAPHS
              SAVEAMP(JJ,J)=(0D0,0D0)
            ENDDO
          ENDDO
        ENDIF
        ANS(IPROC) = 0D0
        IF (GET_HEL .EQ. 0 .OR. NTRY .LT. 2) THEN
          DO IHEL=1,NCOMB
            IF (GOODHEL(IHEL,IPROC) .OR. NTRY .LT. 2) THEN
              T=B_SF_001(P1,NHEL(1,IHEL),IHEL,JC(1))
              ANS(IPROC)=ANS(IPROC)+T
              IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC)) THEN
                GOODHEL(IHEL,IPROC)=.TRUE.
                NGOOD = NGOOD +1
                IGOOD(NGOOD) = IHEL
              ENDIF
            ENDIF
          ENDDO
        ELSE  !RANDOM HELICITY
          HWGT = REAL(NGOOD)
          IHEL=GET_HEL
          T=B_SF_001(P1,NHEL(1,IHEL),IHEL,JC(1))
          ANS(IPROC)=ANS(IPROC)+T*HWGT
        ENDIF
        ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      CALCULATEDBORN=.TRUE.
      END


      REAL*8 FUNCTION B_SF_001(P,NHEL,HELL,IC)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

C     Process: u g > u g WEIGHTED=2 QED=0 QCD=2 [ QCD ]
C     spectators: 1 2 

C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS,    NEIGEN
      PARAMETER (NGRAPHS=   3,NEIGEN=  1)
      INCLUDE 'nexternal.inc'
      INTEGER    NWAVEFUNCS, NCOLOR1, NCOLOR2
      PARAMETER (NWAVEFUNCS=7, NCOLOR1=2, NCOLOR2=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1), IC(NEXTERNAL-1), HELL
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR1), CF(NCOLOR2,NCOLOR1)
      COMPLEX*16 AMP(NGRAPHS), JAMP1(NCOLOR1), JAMP2(NCOLOR2)
      COMPLEX*16 W(18,NWAVEFUNCS)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1 = (0D0,1D0))
C     
C     GLOBAL VARIABLES
C     
      INCLUDE 'born_nhel.inc'
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION HEL_FAC
      LOGICAL CALCULATEDBORN
      INTEGER GET_HEL,SKIP
      COMMON/CBORN/HEL_FAC,CALCULATEDBORN,GET_HEL,SKIP
      INCLUDE 'coupl.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/3/
      DATA (CF(I,  1),I=  1,  2) /    6,   -2/
      DATA DENOM(2)/3/
      DATA (CF(I,  2),I=  1,  2) /    6,   16/
C     ----------
C     BEGIN CODE
C     ----------

      IF (.NOT. CALCULATEDBORN) THEN
        CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1*IC(1),W(1,1))
        CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
        CALL OXXXXX(P(0,3),ZERO,NHEL(3),+1*IC(3),W(1,3))
        CALL VXXXXX(P(0,4),ZERO,NHEL(4),+1*IC(4),W(1,4))
        CALL FFV1_2(W(1,1),W(1,2),GC_5,ZERO, ZERO, W(1,5))
C       Amplitude(s) for diagram number 1
        CALL FFV1_0(W(1,5),W(1,3),W(1,4),GC_5,AMP(1))
        CALL FFV1_3(W(1,1),W(1,3),GC_5,ZERO, ZERO, W(1,6))
C       Amplitude(s) for diagram number 2
        CALL VVV1_0(W(1,2),W(1,4),W(1,6),GC_4,AMP(2))
        CALL FFV1_2(W(1,1),W(1,4),GC_5,ZERO, ZERO, W(1,7))
C       Amplitude(s) for diagram number 3
        CALL FFV1_0(W(1,7),W(1,3),W(1,2),GC_5,AMP(3))
        DO I=1,NGRAPHS
          SAVEAMP(I,HELL)=AMP(I)
        ENDDO
      ELSEIF (CALCULATEDBORN) THEN
        DO I=1,NGRAPHS
          AMP(I)=SAVEAMP(I,HELL)
        ENDDO
      ENDIF
      JAMP1(1)=-IMAG1*AMP(2)+AMP(3)
      JAMP1(2)=+AMP(1)+IMAG1*AMP(2)
      JAMP2(1)=+1D0/2D0*(-IMAG1*AMP(2)+AMP(3))
      JAMP2(2)=+1D0/2D0*(-3D0*AMP(1)-3D0*IMAG1*AMP(2))
      B_SF_001 = 0.D0
      DO I = 1, NCOLOR1
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR2
          ZTEMP = ZTEMP + CF(J,I)*JAMP2(J)
        ENDDO
        B_SF_001 =B_SF_001+ZTEMP*DCONJG(JAMP1(I))/DENOM(I)
      ENDDO
      END



""" % misc.get_pkg_info()
        
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        process_exporter.write_b_sf_fks(\
            writers.FortranWriter(self.give_pos('test')),
            copy.copy(self.myfks_me.born_matrix_element),
            self.myfks_me.color_links[0], 1,
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)

    
    def test_write_auto_dsig_fks_B(self):
        """test if the auto_dsig.f, containing (for MadFKS) the parton luminosity,
        is correctly written
        The real process used is uux_uxug (real_processes[5])
        """
        goal = \
"""      DOUBLE PRECISION FUNCTION DLUM()
C     ****************************************************            
C         
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     RETURNS PARTON LUMINOSITIES FOR MADFKS                          
C        
C     
C     Process: u u~ > u~ u g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     
C     ****************************************************            
C         
      IMPLICIT NONE
C     
C     CONSTANTS                                                       
C         
C     
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379.66*1000)  !CONV TO PICOBARNS             
      REAL*8     PI
      PARAMETER (PI=3.1415926D0)
C     
C     ARGUMENTS                                                       
C         
C     
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
C     
C     LOCAL VARIABLES                                                 
C         
C     
      INTEGER I, ICROSS,ITYPE,LP
      DOUBLE PRECISION P1(0:3,NEXTERNAL)
      DOUBLE PRECISION U1,UB1,D1,DB1,C1,CB1,S1,SB1,B1,BB1
      DOUBLE PRECISION U2,UB2,D2,DB2,C2,CB2,S2,SB2,B2,BB2
      DOUBLE PRECISION G1,G2
      DOUBLE PRECISION A1,A2
      DOUBLE PRECISION XPQ(-7:7)
C     
C     EXTERNAL FUNCTIONS                                              
C         
C     
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF
C     
C     GLOBAL VARIABLES                                                
C         
C     
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      INCLUDE 'coupl.inc'
      INCLUDE 'run.inc'
      INTEGER IMIRROR
      COMMON/CMIRROR/IMIRROR
C     
C     DATA                                                            
C         
C     
      DATA U1,UB1,D1,DB1,C1,CB1,S1,SB1,B1,BB1/10*1D0/
      DATA U2,UB2,D2,DB2,C2,CB2,S2,SB2,B2,BB2/10*1D0/
      DATA A1,G1/2*1D0/
      DATA A2,G2/2*1D0/
      DATA IPROC,ICROSS/1,1/
C     ----------                                                      
C         
C     BEGIN CODE                                                      
C         
C     ----------                                                      
C         
      DLUM = 0D0
      IF (IMIRROR.EQ.2) THEN
        IF (ABS(LPP(2)) .GE. 1) THEN
          LP=SIGN(1,LPP(2))
          U1=PDG2PDF(ABS(LPP(2)),2*LP,XBK(2),DSQRT(Q2FACT(2)))
        ENDIF
        IF (ABS(LPP(1)) .GE. 1) THEN
          LP=SIGN(1,LPP(1))
          UB2=PDG2PDF(ABS(LPP(1)),-2*LP,XBK(1),DSQRT(Q2FACT(1)))
        ENDIF
        PD(0) = 0D0
        IPROC = 0
        IPROC=IPROC+1  ! u u~ > u~ u g
        PD(IPROC)=PD(IPROC-1) + U1*UB2
      ELSE
        IF (ABS(LPP(1)) .GE. 1) THEN
          LP=SIGN(1,LPP(1))
          U1=PDG2PDF(ABS(LPP(1)),2*LP,XBK(1),DSQRT(Q2FACT(1)))
        ENDIF
        IF (ABS(LPP(2)) .GE. 1) THEN
          LP=SIGN(1,LPP(2))
          UB2=PDG2PDF(ABS(LPP(2)),-2*LP,XBK(2),DSQRT(Q2FACT(2)))
        ENDIF
        PD(0) = 0D0
        IPROC = 0
        IPROC=IPROC+1  ! u u~ > u~ u g
        PD(IPROC)=PD(IPROC-1) + U1*UB2
      ENDIF
      DLUM = PD(IPROC) * CONV
      RETURN
      END

""" % misc.get_pkg_info()
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        nflows = \
            process_exporter.write_auto_dsig_fks(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element,
                    self.myfortranmodel)  

        self.assertFileContains('test', goal) 

    def test_write_leshouche_file_B(self):
        """tests if the leshouche.inc file is correctly written.
        The real process used is uux_uxug (real_processes[5])
        """
        goal = \
"""      DATA (IDUP(I,1),I=1,5)/2,-2,-2,2,21/
      DATA (MOTHUP(1,I,  1),I=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP(2,I,  1),I=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP(1,I,  1),I=1, 5)/501,  0,  0,502,503/
      DATA (ICOLUP(2,I,  1),I=1, 5)/  0,501,503,  0,502/
      DATA (ICOLUP(1,I,  2),I=1, 5)/503,  0,  0,502,503/
      DATA (ICOLUP(2,I,  2),I=1, 5)/  0,501,501,  0,502/
      DATA (ICOLUP(1,I,  3),I=1, 5)/502,  0,  0,502,503/
      DATA (ICOLUP(2,I,  3),I=1, 5)/  0,501,503,  0,501/
      DATA (ICOLUP(1,I,  4),I=1, 5)/503,  0,  0,502,503/
      DATA (ICOLUP(2,I,  4),I=1, 5)/  0,501,502,  0,501/
"""    
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        nflows = \
            process_exporter.write_leshouche_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element,
                    self.myfortranmodel)  

        self.assertFileContains('test', goal) 


    def test_write_born_nhel_file_B(self):
        """tests if the born_nhel.inc file is correctly written"""
        goal = \
"""      INTEGER    MAX_BHEL, MAX_BCOL
      PARAMETER (MAX_BHEL=16)
      PARAMETER(MAX_BCOL=2)
"""        
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        calls, ncolor = \
            process_exporter.write_born_fks(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.born_matrix_element,0,
                    self.myfortranmodel)    

        nflows = \
            process_exporter.write_leshouche_file(
                    writers.FortranWriter(self.give_pos('test2')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)  
                
        process_exporter.write_born_nhel_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.born_matrix_element,
                    nflows,
                    self.myfortranmodel,
                    ncolor)  

        self.assertFileContains('test', goal) 

    def test_write_maxamps_file_B(self):
        """tests if the maxamps.inc file is correctly written.
        The real process used is uux_uxug (real_processes[5])
        """
        goal = \
"""      INTEGER    MAXAMPS, MAXFLOW
      PARAMETER (MAXAMPS=10, MAXFLOW=4)
"""        
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        calls, ncolor = \
            process_exporter.write_matrix_element_fks(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.real_processes[5].matrix_element,
                    self.myfortranmodel)    

        process_exporter.write_maxamps_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element,
                    self.myfortranmodel,
                    ncolor)  

        self.assertFileContains('test', goal) 
    
    def test_write_mg_sym_file_B(self):
        """tests if the mg.sym file is correctly written.
        The real process used is uux_uxug (real_processes[5])
        """
        goal = \
""""""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()   
        
        process_exporter.write_mg_sym_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element,
                    self.myfortranmodel)        

        self.assertFileContains('test', goal) 

    
    def test_write_nexternal_file_B(self):
        """tests if the nexternal.inc file is correctly written.
        The real process used is uux_uxug (real_processes[5])
        """
        goal = \
"""      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NINCOMING
      PARAMETER (NINCOMING=2)
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        process_exporter.write_nexternal_file(
                    writers.FortranWriter(self.give_pos('test')),
                    5, 2)        

        self.assertFileContains('test', goal)  


    def test_write_ngraphs_file_B(self):
        """tests if the ngraphs.inc file is correctly written.
        The function called is the one of the FortranProcessExporterV4 class.
        The real process used is uux_uxug (real_processes[5]).
        """
        goal = \
"""      INTEGER    N_MAX_CG
      PARAMETER (N_MAX_CG=10)
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        nconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.real_processes[5].matrix_element,
                    False,
                    self.myfortranmodel)
        
        process_exporter.write_ngraphs_file(
                    writers.FortranWriter(self.give_pos('test')),
                    nconfigs)        

        self.assertFileContains('test', goal)    


    def test_write_pmass_file_B(self):
        """tests if the pmass.inc file is correctly written.
        The function called is the one of the FortranProcessExporterV4 class.
        The real process used is uux_uxug (real_processes[5]).
        """
        goal = \
"""      PMASS(1)=ZERO
      PMASS(2)=ZERO
      PMASS(3)=ZERO
      PMASS(4)=ZERO
      PMASS(5)=ZERO
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        process_exporter.write_pmass_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element)        

        self.assertFileContains('test', goal) 
        

    def test_write_configs_file_noinv_B(self):
        """Tests if the configs.inc file is corretly written, 
        without the t-channel inversion.
        The (real) process used is gg_uxug (real_processes[1]).
        """
        goal = \
"""C     Diagram 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(I, -1,   1),I=1,2)/  4,  3/
      DATA SPROP(  -1,   1)/      21/
      DATA (IFOREST(I, -2,   1),I=1,2)/  5, -1/
      DATA SPROP(  -2,   1)/      21/
C     Diagram 2
      DATA MAPCONFIG(   2)/   2/
      DATA (IFOREST(I, -1,   2),I=1,2)/  5,  3/
      DATA SPROP(  -1,   2)/      -2/
      DATA (IFOREST(I, -2,   2),I=1,2)/  4, -1/
      DATA SPROP(  -2,   2)/      21/
C     Diagram 3
      DATA MAPCONFIG(   3)/   3/
      DATA (IFOREST(I, -1,   3),I=1,2)/  5,  4/
      DATA SPROP(  -1,   3)/       2/
      DATA (IFOREST(I, -2,   3),I=1,2)/ -1,  3/
      DATA SPROP(  -2,   3)/      21/
C     Diagram 4
      DATA MAPCONFIG(   4)/   4/
      DATA (IFOREST(I, -1,   4),I=1,2)/  1,  3/
      DATA TPRID(  -1,   4)/       2/
      DATA (IFOREST(I, -2,   4),I=1,2)/ -1,  5/
      DATA TPRID(  -2,   4)/       2/
      DATA (IFOREST(I, -3,   4),I=1,2)/ -2,  4/
C     Diagram 5
      DATA MAPCONFIG(   5)/   5/
      DATA (IFOREST(I, -1,   5),I=1,2)/  1,  3/
      DATA TPRID(  -1,   5)/       2/
      DATA (IFOREST(I, -2,   5),I=1,2)/ -1,  4/
      DATA TPRID(  -2,   5)/      21/
      DATA (IFOREST(I, -3,   5),I=1,2)/ -2,  5/
C     Diagram 6
      DATA MAPCONFIG(   6)/   6/
      DATA (IFOREST(I, -1,   6),I=1,2)/  5,  4/
      DATA SPROP(  -1,   6)/       2/
      DATA (IFOREST(I, -2,   6),I=1,2)/  1,  3/
      DATA TPRID(  -2,   6)/       2/
      DATA (IFOREST(I, -3,   6),I=1,2)/ -2, -1/
C     Diagram 7
      DATA MAPCONFIG(   7)/   7/
      DATA (IFOREST(I, -1,   7),I=1,2)/  1,  4/
      DATA TPRID(  -1,   7)/       2/
      DATA (IFOREST(I, -2,   7),I=1,2)/ -1,  5/
      DATA TPRID(  -2,   7)/       2/
      DATA (IFOREST(I, -3,   7),I=1,2)/ -2,  3/
C     Diagram 8
      DATA MAPCONFIG(   8)/   8/
      DATA (IFOREST(I, -1,   8),I=1,2)/  1,  4/
      DATA TPRID(  -1,   8)/       2/
      DATA (IFOREST(I, -2,   8),I=1,2)/ -1,  3/
      DATA TPRID(  -2,   8)/      21/
      DATA (IFOREST(I, -3,   8),I=1,2)/ -2,  5/
C     Diagram 9
      DATA MAPCONFIG(   9)/   9/
      DATA (IFOREST(I, -1,   9),I=1,2)/  5,  3/
      DATA SPROP(  -1,   9)/      -2/
      DATA (IFOREST(I, -2,   9),I=1,2)/  1,  4/
      DATA TPRID(  -2,   9)/       2/
      DATA (IFOREST(I, -3,   9),I=1,2)/ -2, -1/
C     Diagram 10
      DATA MAPCONFIG(  10)/  10/
      DATA (IFOREST(I, -1,  10),I=1,2)/  1,  5/
      DATA TPRID(  -1,  10)/      21/
      DATA (IFOREST(I, -2,  10),I=1,2)/ -1,  4/
      DATA TPRID(  -2,  10)/       2/
      DATA (IFOREST(I, -3,  10),I=1,2)/ -2,  3/
C     Diagram 11
      DATA MAPCONFIG(  11)/  11/
      DATA (IFOREST(I, -1,  11),I=1,2)/  1,  5/
      DATA TPRID(  -1,  11)/      21/
      DATA (IFOREST(I, -2,  11),I=1,2)/ -1,  3/
      DATA TPRID(  -2,  11)/       2/
      DATA (IFOREST(I, -3,  11),I=1,2)/ -2,  4/
C     Diagram 12
      DATA MAPCONFIG(  12)/  12/
      DATA (IFOREST(I, -1,  12),I=1,2)/  4,  3/
      DATA SPROP(  -1,  12)/      21/
      DATA (IFOREST(I, -2,  12),I=1,2)/  1,  5/
      DATA TPRID(  -2,  12)/      21/
      DATA (IFOREST(I, -3,  12),I=1,2)/ -2, -1/
C     Diagram 13
      DATA MAPCONFIG(  13)/  13/
      DATA (IFOREST(I, -1,  13),I=1,2)/  5,  4/
      DATA SPROP(  -1,  13)/       2/
      DATA (IFOREST(I, -2,  13),I=1,2)/  1, -1/
      DATA TPRID(  -2,  13)/       2/
      DATA (IFOREST(I, -3,  13),I=1,2)/ -2,  3/
C     Diagram 14
      DATA MAPCONFIG(  14)/  14/
      DATA (IFOREST(I, -1,  14),I=1,2)/  5,  3/
      DATA SPROP(  -1,  14)/      -2/
      DATA (IFOREST(I, -2,  14),I=1,2)/  1, -1/
      DATA TPRID(  -2,  14)/       2/
      DATA (IFOREST(I, -3,  14),I=1,2)/ -2,  4/
C     Diagram 15
      DATA MAPCONFIG(  15)/  15/
      DATA (IFOREST(I, -1,  15),I=1,2)/  4,  3/
      DATA SPROP(  -1,  15)/      21/
      DATA (IFOREST(I, -2,  15),I=1,2)/  1, -1/
      DATA TPRID(  -2,  15)/      21/
      DATA (IFOREST(I, -3,  15),I=1,2)/ -2,  5/
C     Number of configs
      DATA MAPCONFIG(0)/  15/
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        nconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[1].matrix_element,
                    False,
                    self.myfortranmodel)

        self.assertFileContains('test', goal)    


    def test_write_configs_file_inv_B(self):
        """Tests if the configs.inc file is corretly written, 
        without the t-channel inversion.
        The real process used is uux_uxug (real_processes[5]).
        """
        goal = \
"""C     Diagram 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(I, -1,   1),I=1,2)/  4,  3/
      DATA SPROP(  -1,   1)/      21/
      DATA (IFOREST(I, -2,   1),I=1,2)/  5, -1/
      DATA SPROP(  -2,   1)/      21/
C     Diagram 2
      DATA MAPCONFIG(   2)/   2/
      DATA (IFOREST(I, -1,   2),I=1,2)/  5,  3/
      DATA SPROP(  -1,   2)/      -2/
      DATA (IFOREST(I, -2,   2),I=1,2)/  4, -1/
      DATA SPROP(  -2,   2)/      21/
C     Diagram 3
      DATA MAPCONFIG(   3)/   3/
      DATA (IFOREST(I, -1,   3),I=1,2)/  5,  4/
      DATA SPROP(  -1,   3)/       2/
      DATA (IFOREST(I, -2,   3),I=1,2)/ -1,  3/
      DATA SPROP(  -2,   3)/      21/
C     Diagram 4
      DATA MAPCONFIG(   4)/   4/
      DATA (IFOREST(I, -1,   4),I=1,2)/  2,  3/
      DATA TPRID(  -1,   4)/      21/
      DATA (IFOREST(I, -2,   4),I=1,2)/ -1,  5/
      DATA TPRID(  -2,   4)/      21/
      DATA (IFOREST(I, -3,   4),I=1,2)/ -2,  4/
C     Diagram 5
      DATA MAPCONFIG(   5)/   5/
      DATA (IFOREST(I, -1,   5),I=1,2)/  2,  5/
      DATA TPRID(  -1,   5)/       2/
      DATA (IFOREST(I, -2,   5),I=1,2)/ -1,  3/
      DATA TPRID(  -2,   5)/      21/
      DATA (IFOREST(I, -3,   5),I=1,2)/ -2,  4/
C     Diagram 6
      DATA MAPCONFIG(   6)/   6/
      DATA (IFOREST(I, -1,   6),I=1,2)/  5,  3/
      DATA SPROP(  -1,   6)/      -2/
      DATA (IFOREST(I, -2,   6),I=1,2)/  2, -1/
      DATA TPRID(  -2,   6)/      21/
      DATA (IFOREST(I, -3,   6),I=1,2)/ -2,  4/
C     Diagram 7
      DATA MAPCONFIG(   7)/   7/
      DATA (IFOREST(I, -1,   7),I=1,2)/  2,  3/
      DATA TPRID(  -1,   7)/      21/
      DATA (IFOREST(I, -2,   7),I=1,2)/ -1,  4/
      DATA TPRID(  -2,   7)/       2/
      DATA (IFOREST(I, -3,   7),I=1,2)/ -2,  5/
C     Diagram 8
      DATA MAPCONFIG(   8)/   8/
      DATA (IFOREST(I, -1,   8),I=1,2)/  4,  3/
      DATA SPROP(  -1,   8)/      21/
      DATA (IFOREST(I, -2,   8),I=1,2)/  2, -1/
      DATA TPRID(  -2,   8)/       2/
      DATA (IFOREST(I, -3,   8),I=1,2)/ -2,  5/
C     Diagram 9
      DATA MAPCONFIG(   9)/   9/
      DATA (IFOREST(I, -1,   9),I=1,2)/  5,  4/
      DATA SPROP(  -1,   9)/       2/
      DATA (IFOREST(I, -2,   9),I=1,2)/  2,  3/
      DATA TPRID(  -2,   9)/      21/
      DATA (IFOREST(I, -3,   9),I=1,2)/ -2, -1/
C     Diagram 10
      DATA MAPCONFIG(  10)/  10/
      DATA (IFOREST(I, -1,  10),I=1,2)/  4,  3/
      DATA SPROP(  -1,  10)/      21/
      DATA (IFOREST(I, -2,  10),I=1,2)/  2,  5/
      DATA TPRID(  -2,  10)/       2/
      DATA (IFOREST(I, -3,  10),I=1,2)/ -2, -1/
C     Number of configs
      DATA MAPCONFIG(0)/  10/
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        nconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element,
                    True,
                    self.myfortranmodel)

        self.assertFileContains('test', goal)    


    def test_write_props_file_noinv_B(self):
        """Tests if the props.inc file is corretly written, 
        without the t-channel inversion.
        The (real) process used is gg_uxug (real_processes[1]).
        """
        goal = \
"""      PMASS( -1,   1)  = ZERO
      PWIDTH( -1,   1) = ZERO
      POW( -1,   1) = 2
      PMASS( -2,   1)  = ZERO
      PWIDTH( -2,   1) = ZERO
      POW( -2,   1) = 2
      PMASS( -1,   2)  = ZERO
      PWIDTH( -1,   2) = ZERO
      POW( -1,   2) = 1
      PMASS( -2,   2)  = ZERO
      PWIDTH( -2,   2) = ZERO
      POW( -2,   2) = 2
      PMASS( -1,   3)  = ZERO
      PWIDTH( -1,   3) = ZERO
      POW( -1,   3) = 1
      PMASS( -2,   3)  = ZERO
      PWIDTH( -2,   3) = ZERO
      POW( -2,   3) = 2
      PMASS( -1,   4)  = ZERO
      PWIDTH( -1,   4) = ZERO
      POW( -1,   4) = 1
      PMASS( -2,   4)  = ZERO
      PWIDTH( -2,   4) = ZERO
      POW( -2,   4) = 1
      PMASS( -1,   5)  = ZERO
      PWIDTH( -1,   5) = ZERO
      POW( -1,   5) = 1
      PMASS( -2,   5)  = ZERO
      PWIDTH( -2,   5) = ZERO
      POW( -2,   5) = 2
      PMASS( -1,   6)  = ZERO
      PWIDTH( -1,   6) = ZERO
      POW( -1,   6) = 1
      PMASS( -2,   6)  = ZERO
      PWIDTH( -2,   6) = ZERO
      POW( -2,   6) = 1
      PMASS( -1,   7)  = ZERO
      PWIDTH( -1,   7) = ZERO
      POW( -1,   7) = 1
      PMASS( -2,   7)  = ZERO
      PWIDTH( -2,   7) = ZERO
      POW( -2,   7) = 1
      PMASS( -1,   8)  = ZERO
      PWIDTH( -1,   8) = ZERO
      POW( -1,   8) = 1
      PMASS( -2,   8)  = ZERO
      PWIDTH( -2,   8) = ZERO
      POW( -2,   8) = 2
      PMASS( -1,   9)  = ZERO
      PWIDTH( -1,   9) = ZERO
      POW( -1,   9) = 1
      PMASS( -2,   9)  = ZERO
      PWIDTH( -2,   9) = ZERO
      POW( -2,   9) = 1
      PMASS( -1,  10)  = ZERO
      PWIDTH( -1,  10) = ZERO
      POW( -1,  10) = 2
      PMASS( -2,  10)  = ZERO
      PWIDTH( -2,  10) = ZERO
      POW( -2,  10) = 1
      PMASS( -1,  11)  = ZERO
      PWIDTH( -1,  11) = ZERO
      POW( -1,  11) = 2
      PMASS( -2,  11)  = ZERO
      PWIDTH( -2,  11) = ZERO
      POW( -2,  11) = 1
      PMASS( -1,  12)  = ZERO
      PWIDTH( -1,  12) = ZERO
      POW( -1,  12) = 2
      PMASS( -2,  12)  = ZERO
      PWIDTH( -2,  12) = ZERO
      POW( -2,  12) = 2
      PMASS( -1,  13)  = ZERO
      PWIDTH( -1,  13) = ZERO
      POW( -1,  13) = 1
      PMASS( -2,  13)  = ZERO
      PWIDTH( -2,  13) = ZERO
      POW( -2,  13) = 1
      PMASS( -1,  14)  = ZERO
      PWIDTH( -1,  14) = ZERO
      POW( -1,  14) = 1
      PMASS( -2,  14)  = ZERO
      PWIDTH( -2,  14) = ZERO
      POW( -2,  14) = 1
      PMASS( -1,  15)  = ZERO
      PWIDTH( -1,  15) = ZERO
      POW( -1,  15) = 2
      PMASS( -2,  15)  = ZERO
      PWIDTH( -2,  15) = ZERO
      POW( -2,  15) = 2
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        nconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.real_processes[1].matrix_element,
                    False,
                    self.myfortranmodel)
        
        process_exporter.write_props_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[1].matrix_element,
                    self.myfortranmodel,
                    s_and_t_channels)        

        self.assertFileContains('test', goal)    


    def test_write_props_file_inv_B(self):
        """Tests if the props.inc file is corretly written, 
        with the t-channel inversion
        The (real) process used is uux_uxug (real_processes[5]).
        """
        goal = \
"""      PMASS( -1,   1)  = ZERO
      PWIDTH( -1,   1) = ZERO
      POW( -1,   1) = 2
      PMASS( -2,   1)  = ZERO
      PWIDTH( -2,   1) = ZERO
      POW( -2,   1) = 2
      PMASS( -1,   2)  = ZERO
      PWIDTH( -1,   2) = ZERO
      POW( -1,   2) = 1
      PMASS( -2,   2)  = ZERO
      PWIDTH( -2,   2) = ZERO
      POW( -2,   2) = 2
      PMASS( -1,   3)  = ZERO
      PWIDTH( -1,   3) = ZERO
      POW( -1,   3) = 1
      PMASS( -2,   3)  = ZERO
      PWIDTH( -2,   3) = ZERO
      POW( -2,   3) = 2
      PMASS( -1,   4)  = ZERO
      PWIDTH( -1,   4) = ZERO
      POW( -1,   4) = 2
      PMASS( -2,   4)  = ZERO
      PWIDTH( -2,   4) = ZERO
      POW( -2,   4) = 2
      PMASS( -1,   5)  = ZERO
      PWIDTH( -1,   5) = ZERO
      POW( -1,   5) = 1
      PMASS( -2,   5)  = ZERO
      PWIDTH( -2,   5) = ZERO
      POW( -2,   5) = 2
      PMASS( -1,   6)  = ZERO
      PWIDTH( -1,   6) = ZERO
      POW( -1,   6) = 1
      PMASS( -2,   6)  = ZERO
      PWIDTH( -2,   6) = ZERO
      POW( -2,   6) = 2
      PMASS( -1,   7)  = ZERO
      PWIDTH( -1,   7) = ZERO
      POW( -1,   7) = 2
      PMASS( -2,   7)  = ZERO
      PWIDTH( -2,   7) = ZERO
      POW( -2,   7) = 1
      PMASS( -1,   8)  = ZERO
      PWIDTH( -1,   8) = ZERO
      POW( -1,   8) = 2
      PMASS( -2,   8)  = ZERO
      PWIDTH( -2,   8) = ZERO
      POW( -2,   8) = 1
      PMASS( -1,   9)  = ZERO
      PWIDTH( -1,   9) = ZERO
      POW( -1,   9) = 1
      PMASS( -2,   9)  = ZERO
      PWIDTH( -2,   9) = ZERO
      POW( -2,   9) = 2
      PMASS( -1,  10)  = ZERO
      PWIDTH( -1,  10) = ZERO
      POW( -1,  10) = 2
      PMASS( -2,  10)  = ZERO
      PWIDTH( -2,  10) = ZERO
      POW( -2,  10) = 1
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        nconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.real_processes[5].matrix_element,
                    True,
                    self.myfortranmodel)
        
        process_exporter.write_props_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element,
                    self.myfortranmodel,
                    s_and_t_channels)        

        self.assertFileContains('test', goal)    


    def test_write_coloramps_file_B(self):
        """Tests if the coloramps.inc file is corretly written
        The (real) process used is uux_uxug (real_processes[5]).
        """
        goal = \
"""      LOGICAL ICOLAMP(10,4)
      DATA(ICOLAMP(I,1),I=1,10)/.FALSE.,.TRUE.,.TRUE.,.TRUE.,.FALSE.
     $ ,.TRUE.,.FALSE.,.FALSE.,.TRUE.,.FALSE./
      DATA(ICOLAMP(I,2),I=1,10)/.TRUE.,.FALSE.,.TRUE.,.FALSE.,.FALSE.
     $ ,.FALSE.,.TRUE.,.TRUE.,.TRUE.,.FALSE./
      DATA(ICOLAMP(I,3),I=1,10)/.TRUE.,.TRUE.,.FALSE.,.FALSE.,.TRUE.
     $ ,.TRUE.,.FALSE.,.FALSE.,.FALSE.,.TRUE./
      DATA(ICOLAMP(I,4),I=1,10)/.FALSE.,.FALSE.,.FALSE.,.TRUE.,.TRUE.
     $ ,.FALSE.,.TRUE.,.TRUE.,.FALSE.,.TRUE./
"""
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()
        
        process_exporter.write_coloramps_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[5].matrix_element,
                    self.myfortranmodel)        

        self.assertFileContains('test', goal)    


    def test_write_bornfromreal_file_B(self):
        """Tests if the bornfromreal.inc file is corretly written, for all 
        the real emission processes possible for this born.
        """
        goal = [\
"""      DATA B_FROM_R(13) / 1 /
      DATA R_FROM_B(1) / 13 /
      DATA B_FROM_R(12) / 2 /
      DATA R_FROM_B(2) / 12 /
      DATA B_FROM_R(11) / 3 /
      DATA R_FROM_B(3) / 11 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(6) / 1 /
      DATA R_FROM_B(1) / 6 /
      DATA B_FROM_R(5) / 2 /
      DATA R_FROM_B(2) / 5 /
      DATA B_FROM_R(4) / 3 /
      DATA R_FROM_B(3) / 4 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(16) / 1 /
      DATA R_FROM_B(1) / 16 /
      DATA B_FROM_R(5) / 2 /
      DATA R_FROM_B(2) / 5 /
      DATA B_FROM_R(9) / 3 /
      DATA R_FROM_B(3) / 9 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(5) / 1 /
      DATA R_FROM_B(1) / 5 /
      DATA B_FROM_R(1) / 2 /
      DATA R_FROM_B(2) / 1 /
      DATA B_FROM_R(4) / 3 /
      DATA R_FROM_B(3) / 4 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(5) / 1 /
      DATA R_FROM_B(1) / 5 /
      DATA B_FROM_R(1) / 2 /
      DATA R_FROM_B(2) / 1 /
      DATA B_FROM_R(4) / 3 /
      DATA R_FROM_B(3) / 4 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(9) / 1 /
      DATA R_FROM_B(1) / 9 /
      DATA B_FROM_R(4) / 2 /
      DATA R_FROM_B(2) / 4 /
      DATA B_FROM_R(7) / 3 /
      DATA R_FROM_B(3) / 7 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(10) / 1 /
      DATA R_FROM_B(1) / 10 /
      DATA B_FROM_R(1) / 2 /
      DATA R_FROM_B(2) / 1 /
      DATA B_FROM_R(8) / 3 /
      DATA R_FROM_B(3) / 8 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(2) / 1 /
      DATA R_FROM_B(1) / 2 /
      DATA B_FROM_R(15) / 2 /
      DATA R_FROM_B(2) / 15 /
      DATA B_FROM_R(10) / 3 /
      DATA R_FROM_B(3) / 10 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(3) / 1 /
      DATA R_FROM_B(1) / 3 /
      DATA B_FROM_R(6) / 2 /
      DATA R_FROM_B(2) / 6 /
      DATA B_FROM_R(14) / 3 /
      DATA R_FROM_B(3) / 14 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(1) / 1 /
      DATA R_FROM_B(1) / 1 /
      DATA B_FROM_R(4) / 2 /
      DATA R_FROM_B(2) / 4 /
      DATA B_FROM_R(5) / 3 /
      DATA R_FROM_B(3) / 5 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
""",
"""      DATA B_FROM_R(2) / 1 /
      DATA R_FROM_B(1) / 2 /
      DATA B_FROM_R(5) / 2 /
      DATA R_FROM_B(2) / 5 /
      DATA B_FROM_R(9) / 3 /
      DATA R_FROM_B(3) / 9 /
      INTEGER MAPB
      DATA (MAPBCONF(MAPB), MAPB=0, 3) / 3, 1, 2, 3 /
"""]

        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        for i, real in enumerate(self.myfks_me.real_processes):
        
            process_exporter.write_bornfromreal_file(
                    writers.FortranWriter(self.give_pos('test%d' % i)),
                    real,
                    self.myfortranmodel)        

            self.assertFileContains('test%d' % i, goal[i])    


    def test_write_decayBW_file_B(self):
        """Tests if the decayBW.inc file is correctly written.
        The (real) process used is uux_uxug (real_processes[5]).
        """
        goal = \
"""      DATA GFORCEBW(-1,1)/.FALSE./
      DATA GFORCEBW(-2,1)/.FALSE./
      DATA GFORCEBW(-1,2)/.FALSE./
      DATA GFORCEBW(-2,2)/.FALSE./
      DATA GFORCEBW(-1,3)/.FALSE./
      DATA GFORCEBW(-2,3)/.FALSE./
      DATA GFORCEBW(-1,6)/.FALSE./
      DATA GFORCEBW(-1,8)/.FALSE./
      DATA GFORCEBW(-1,9)/.FALSE./
      DATA GFORCEBW(-1,10)/.FALSE./
"""

        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        nconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.real_processes[5].matrix_element,
                    False,
                    self.myfortranmodel)

        process_exporter.write_decayBW_file(
                    writers.FortranWriter(self.give_pos('test')),
                    s_and_t_channels)        

        self.assertFileContains('test', goal)    


    def test_get_color_data_lines_from_color_matrix_B(self):
        """tests if the color data lines are correctly extracted from a given
        color matrix. 
        The first color link is used.
        """
        
        goal = ["DATA DENOM(1)/3/",
                "DATA (CF(I,  1),I=  1,  2) /    6,   -2/",
                "DATA DENOM(2)/3/",
                "DATA (CF(I,  2),I=  1,  2) /    6,   16/",
                ]
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        lines = process_exporter.get_color_data_lines_from_color_matrix(
                    self.myfks_me.color_links[0]['link_matrix'])
        
        for line, goalline in zip(lines, goal):
            self.assertEqual(line.upper(), goalline)        


    def test_den_factor_line_B(self):
        """Tests if the den_factor line for a given matrix element is correctly 
        returned.
        The (real) process used is uux_uxug (real_processes[5]).
        """
        
        goal = "DATA IDEN/36/"
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        self.assertEqual(goal,
                process_exporter.get_den_factor_line(
                        self.myfks_me.real_processes[5].matrix_element))


    def test_write_den_factor_file_B(self):
        """Testst the den_factor.inc file. It is included in the born ME, with identical
        particle factor as the real ME, and it is (in principle) different for each
        real emission ME.
        """
        goal = [ \
"      DATA IDEN/192/\n",
"      DATA IDEN/96/\n",
"      DATA IDEN/192/\n",
"      DATA IDEN/96/\n",
"      DATA IDEN/96/\n",
"      DATA IDEN/96/\n",
"      DATA IDEN/192/\n",
"      DATA IDEN/192/\n",
"      DATA IDEN/192/\n",
"      DATA IDEN/96/\n",
"      DATA IDEN/192/\n",
        ]

        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        for i, real in enumerate(self.myfks_me.real_processes):
        
            process_exporter.write_den_factor_file(
                    writers.FortranWriter(self.give_pos('test%d' % i)),
                    self.myfks_me.born_matrix_element, real.matrix_element,
                    self.myfortranmodel)        

            self.assertFileContains('test%d' % i, goal[i])    


    def test_write_glu_ij_file_B(self):
        """Testst the glu_ij.inc file. It is included in the born ME, 
        when the born leg that splits is a gluon, with the leg number.
        If the leg is not a gluon, ij is set to 0.
        """
        goal = [ \
"      DATA GLU_IJ/ 0/\n",
"      DATA GLU_IJ/ 0/\n",
"      DATA GLU_IJ/ 2/\n",
"      DATA GLU_IJ/ 2/\n",
"      DATA GLU_IJ/ 2/\n",
"      DATA GLU_IJ/ 2/\n",
"      DATA GLU_IJ/ 2/\n",
"      DATA GLU_IJ/ 0/\n",
"      DATA GLU_IJ/ 4/\n",
"      DATA GLU_IJ/ 4/\n",
"      DATA GLU_IJ/ 4/\n",
        ]

        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        for i, real in enumerate(self.myfks_me.real_processes):
        
            process_exporter.write_glu_ij_file(
                    writers.FortranWriter(self.give_pos('test%d' % i)),
                    real.ijglu,
                    self.myfortranmodel)        

            self.assertFileContains('test%d' % i, goal[i])    


    def test_get_icolamp_lines_B(self):
        """Tests if the icolamp lines for a given matrix element are correctly 
        returned.
        The real process is gg_uxug_i3_j1 (myfks_me.real_processes[1])
        """
        goal = ['logical icolamp(18,6)',
        'DATA(icolamp(i,1),i=1,18)/.true.,.true.,.false.,.false.,.false.,.false.,.false.,.true.,.true.,.false.,.false.,.false.,.false.,.false.,.true.,.true.,.false.,.true./',
        'DATA(icolamp(i,2),i=1,18)/.false.,.false.,.false.,.false.,.false.,.false.,.true.,.true.,.false.,.true.,.false.,.true.,.false.,.false.,.true.,.true.,.true.,.false./',
        'DATA(icolamp(i,3),i=1,18)/.true.,.true.,.false.,.false.,.false.,.false.,.false.,.false.,.false.,.false.,.true.,.true.,.false.,.true.,.false.,.false.,.true.,.true./',
        'DATA(icolamp(i,4),i=1,18)/.false.,.false.,.false.,.true.,.true.,.false.,.false.,.false.,.false.,.false.,.true.,.true.,.false.,.false.,.true.,.true.,.true.,.false./',
        'DATA(icolamp(i,5),i=1,18)/.true.,.false.,.true.,.false.,.false.,.false.,.false.,.false.,.false.,.true.,.false.,.true.,.true.,.false.,.false.,.false.,.true.,.true./',
        'DATA(icolamp(i,6),i=1,18)/.true.,.false.,.true.,.false.,.true.,.true.,.false.,.false.,.false.,.false.,.false.,.false.,.false.,.false.,.true.,.true.,.false.,.true./']       
       
        process_exporter = export_fks_born.ProcessExporterFortranFKS_born()

        self.assertEqual(goal,
                process_exporter.get_icolamp_lines( \
                    self.myfks_me.real_processes[1].matrix_element))

        
