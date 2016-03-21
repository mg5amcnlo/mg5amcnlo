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

"""Unit test library for the export_FKS format routines"""

import StringIO
import copy
import fractions
import os 
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.various.misc as misc
import madgraph.iolibs.export_fks as export_fks
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_helas_objects as fks_helas
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

from functools import wraps
def PostponeToEW(f):
   @wraps(f)
   def postpone(*args,**opts):
     print "\n Test '%s' ignored as it should be fixed with mixed couplings expansion."%f.__name__
     return
   return postpone

class IOExportFKSEWTest(unittest.TestCase,\
                        test_file_writers.CheckFileCreate):
    """Test class for the export fks module for EW corrections"""

    def setUp(self):
        if not hasattr(self, 'myfks_me') or \
           not hasattr(self, 'myfortranmodel') or \
           not hasattr(self, 'myfkshelasmulti') or \
           not hasattr(self, 'myreals'):

            IOExportFKSEWTest.created_files = ['test']

            mymodel = import_ufo.import_model('sm')
            IOExportFKSEWTest.myfortranmodel = helas_call_writers.FortranUFOHelasCallWriter(mymodel)

            myleglist = MG.MultiLegList()

        # we test a a > t t~
            myleglist.append(MG.MultiLeg({'ids':[22], 'state':False}))
            myleglist.append(MG.MultiLeg({'ids':[22], 'state':False}))
            myleglist.append(MG.MultiLeg({'ids':[6], 'state':True}))
            myleglist.append(MG.MultiLeg({'ids':[-6], 'state':True}))

            myproc = MG.ProcessDefinition({'legs': myleglist,
                                 'model': mymodel,
                                 'orders':{'QCD': 0, 'QED': 2},
                                 'perturbation_couplings': ['QED'],
                                 'NLO_mode': 'real'})
            my_process_definitions = MG.ProcessDefinitionList([myproc])

            myfksmulti = fks_base.FKSMultiProcess(\
                    {'process_definitions': my_process_definitions})
            

            fkshelasmulti = fks_helas.FKSHelasMultiProcess(myfksmulti)
            IOExportFKSEWTest.myfkshelasmulti = fkshelasmulti
            IOExportFKSEWTest.myfks_me = fkshelasmulti['matrix_elements'][0]
            IOExportFKSEWTest.myreals = fkshelasmulti['real_matrix_elements']


        #tearDown = test_file_writers.CheckFileCreate.clean_files        

    def test_write_maxconfigs_EW(self):
        goal = \
        """      INTEGER LMAXCONFIGS
      PARAMETER (LMAXCONFIGS=10)"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_maxconfigs_file(\
            writers.FortranWriter(self.give_pos(self.created_files[0])),\
            self.myfkshelasmulti.get_max_configs())
        self.assertFileContains(self.created_files[0], goal)
    
    @PostponeToEW
    def test_write_mparticles_EW(self):
        goal = \
"""      INTEGER MAX_PARTICLES
      PARAMETER (MAX_PARTICLES=5)"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_maxparticles_file(\
            writers.FortranWriter(self.give_pos(self.created_files[0])),\
            self.myreals)
        self.assertFileContains(self.created_files[0], goal)

    def test_write_lh_order_EW(self):
        """tests the correct writing of the B-LH order file"""

        goal = \
"""#OLE_order written by MadGraph5_aMC@NLO

MatrixElementSquareType CHaveraged
CorrectionType          QED
IRregularisation        CDR
AlphasPower             0
AlphaPower              2
NJetSymmetrizeFinal     Yes
ModelFile               ./param_card.dat
Parameters              alpha_s

# process
22 22 -> 6 -6
"""
        process_list = []
        for me in self.myfkshelasmulti['matrix_elements']:
            process_list.append(me.born_matrix_element.get('processes')[0])
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_lh_order(\
            self.give_pos(self.created_files[0]),\
            process_list)
        self.assertFileContains(self.created_files[0], goal)
        
    def test_write_coloramps_file_EW(self):
        """Tests if the coloramps.inc file is correctly written 
        for the born process
        """
        goal = \
"""      LOGICAL ICOLAMP(1,2,1)
      DATA(ICOLAMP(I,1,1),I=1,1)/.TRUE./
      DATA(ICOLAMP(I,2,1),I=1,1)/.TRUE./
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()

        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)
        
        process_exporter.write_coloramps_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    mapconfigs,
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)        

        self.assertFileContains(self.created_files[0], goal)
        
    def test_den_factor_lines_EW(self):
        """Tests if the den_factor lines for a given matrix element are correctly 
        returned.
        """
        
        goal = \
            ["INTEGER IDEN_VALUES(14)",
             "DATA IDEN_VALUES /4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4/"]
        process_exporter = export_fks.ProcessExporterFortranFKS()

        self.assertEqual(goal,
                process_exporter.get_den_factor_lines(
                        self.myfks_me))
        
    def test_write_ij_lines_EW(self):
        """Tests if the ij lines for a given matrix element are correctly 
        returned.
        """
        
        goal = \
            ["INTEGER IJ_VALUES(14)",
             "DATA IJ_VALUES /1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4/"]
        process_exporter = export_fks.ProcessExporterFortranFKS()

        self.assertEqual(goal,
                process_exporter.get_ij_lines(
                        self.myfks_me))
       
    @PostponeToEW
    def test_write_real_me_wrapper_EW(self):
        """tests the correct writing of the real_me_chooser file, 
        that chooses among the different real emissions"""

        goal = \
"""      SUBROUTINE SMATRIX_REAL(P, WGT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P(0:3, NEXTERNAL)
      DOUBLE PRECISION WGT
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      IF (NFKSPROCESS.EQ.1) THEN
        CALL SMATRIX_1(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.2) THEN
        CALL SMATRIX_2(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.3) THEN
        CALL SMATRIX_3(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.4) THEN
        CALL SMATRIX_4(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.5) THEN
        CALL SMATRIX_5(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.6) THEN
        CALL SMATRIX_6(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.7) THEN
        CALL SMATRIX_7(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.8) THEN
        CALL SMATRIX_8(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.9) THEN
        CALL SMATRIX_9(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.10) THEN
        CALL SMATRIX_10(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.11) THEN
        CALL SMATRIX_11(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.12) THEN
        CALL SMATRIX_12(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.13) THEN
        CALL SMATRIX_13(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.14) THEN
        CALL SMATRIX_13(P, WGT)
      ELSE
        WRITE(*,*) 'ERROR: invalid n in real_matrix :', NFKSPROCESS
        STOP
      ENDIF

      RETURN
      END

"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_real_me_wrapper(\
            writers.FortranWriter(self.give_pos(self.created_files[0])),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains(self.created_files[0], goal)
        
    def test_write_pdf_wrapper_EW(self):
        """tests the correct writing of the parton_lum_chooser file, 
        that chooses the pdfs for the different real emissions"""

        goal = \
"""      DOUBLE PRECISION FUNCTION DLUM()
      IMPLICIT NONE
      INCLUDE 'timing_variables.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      CALL CPU_TIME(TBEFORE)
      IF (NFKSPROCESS.EQ.1) THEN
        CALL DLUM_1(DLUM)
      ELSEIF (NFKSPROCESS.EQ.2) THEN
        CALL DLUM_2(DLUM)
      ELSEIF (NFKSPROCESS.EQ.3) THEN
        CALL DLUM_3(DLUM)
      ELSEIF (NFKSPROCESS.EQ.4) THEN
        CALL DLUM_4(DLUM)
      ELSEIF (NFKSPROCESS.EQ.5) THEN
        CALL DLUM_5(DLUM)
      ELSEIF (NFKSPROCESS.EQ.6) THEN
        CALL DLUM_6(DLUM)
      ELSEIF (NFKSPROCESS.EQ.7) THEN
        CALL DLUM_7(DLUM)
      ELSEIF (NFKSPROCESS.EQ.8) THEN
        CALL DLUM_8(DLUM)
      ELSEIF (NFKSPROCESS.EQ.9) THEN
        CALL DLUM_9(DLUM)
      ELSEIF (NFKSPROCESS.EQ.10) THEN
        CALL DLUM_10(DLUM)
      ELSEIF (NFKSPROCESS.EQ.11) THEN
        CALL DLUM_11(DLUM)
      ELSEIF (NFKSPROCESS.EQ.12) THEN
        CALL DLUM_12(DLUM)
      ELSEIF (NFKSPROCESS.EQ.13) THEN
        CALL DLUM_13(DLUM)
      ELSEIF (NFKSPROCESS.EQ.14) THEN
        CALL DLUM_13(DLUM)
      ELSE
        WRITE(*,*) 'ERROR: invalid n in dlum :', NFKSPROCESS
        STOP
      ENDIF
      CALL CPU_TIME(TAFTER)
      TPDF = TPDF + (TAFTER-TBEFORE)
      RETURN
      END

"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_pdf_wrapper(\
            writers.FortranWriter(self.give_pos(self.created_files[0])),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains(self.created_files[0], goal)

    def test_write_leshouche_info_file_EW(self):
        """tests the correct writing of fks_info.inc file, containing the 
        relevant informations for all the splittings"""

        goal = \
"""# I -> IDUP_D
# M -> MOTHUP_D
# C -> ICOLUP_D
I      1      1       -11 22 6 -6 -11
M      1      1      1        0   0   1   1   1
M      1      2      1        0   0   2   2   2
C      1      1      1        0   0 501   0   0
C      1      2      1        0   0   0 501   0
I      1      2       -13 22 6 -6 -13
M      1      1      2        0   0   1   1   1
M      1      2      2        0   0   2   2   2
I      2      1       11 22 6 -6 11
M      2      1      1        0   0   1   1   1
M      2      2      1        0   0   2   2   2
C      2      1      1        0   0 501   0   0
C      2      2      1        0   0   0 501   0
I      2      2       13 22 6 -6 13
M      2      1      2        0   0   1   1   1
M      2      2      2        0   0   2   2   2
I      3      1       -1 22 6 -6 -1
M      3      1      1        0   0   1   1   1
M      3      2      1        0   0   2   2   2
C      3      1      1        0   0 502   0   0
C      3      2      1      501   0   0 502 501
I      3      2       -3 22 6 -6 -3
M      3      1      2        0   0   1   1   1
M      3      2      2        0   0   2   2   2
I      4      1       1 22 6 -6 1
M      4      1      1        0   0   1   1   1
M      4      2      1        0   0   2   2   2
C      4      1      1      502   0 501   0 502
C      4      2      1        0   0   0 501   0
I      4      2       3 22 6 -6 3
M      4      1      2        0   0   1   1   1
M      4      2      2        0   0   2   2   2
I      5      1       -2 22 6 -6 -2
M      5      1      1        0   0   1   1   1
M      5      2      1        0   0   2   2   2
C      5      1      1        0   0 502   0   0
C      5      2      1      501   0   0 502 501
I      5      2       -4 22 6 -6 -4
M      5      1      2        0   0   1   1   1
M      5      2      2        0   0   2   2   2
I      6      1       2 22 6 -6 2
M      6      1      1        0   0   1   1   1
M      6      2      1        0   0   2   2   2
C      6      1      1      502   0 501   0 502
C      6      2      1        0   0   0 501   0
I      6      2       4 22 6 -6 4
M      6      1      2        0   0   1   1   1
M      6      2      2        0   0   2   2   2
I      7      1       22 -11 6 -6 -11
M      7      1      1        0   0   1   1   1
M      7      2      1        0   0   2   2   2
C      7      1      1        0   0 501   0   0
C      7      2      1        0   0   0 501   0
I      7      2       22 -13 6 -6 -13
M      7      1      2        0   0   1   1   1
M      7      2      2        0   0   2   2   2
I      8      1       22 11 6 -6 11
M      8      1      1        0   0   1   1   1
M      8      2      1        0   0   2   2   2
C      8      1      1        0   0 501   0   0
C      8      2      1        0   0   0 501   0
I      8      2       22 13 6 -6 13
M      8      1      2        0   0   1   1   1
M      8      2      2        0   0   2   2   2
I      9      1       22 -1 6 -6 -1
M      9      1      1        0   0   1   1   1
M      9      2      1        0   0   2   2   2
C      9      1      1        0   0 502   0   0
C      9      2      1        0 501   0 502 501
I      9      2       22 -3 6 -6 -3
M      9      1      2        0   0   1   1   1
M      9      2      2        0   0   2   2   2
I     10      1       22 1 6 -6 1
M     10      1      1        0   0   1   1   1
M     10      2      1        0   0   2   2   2
C     10      1      1        0 502 501   0 502
C     10      2      1        0   0   0 501   0
I     10      2       22 3 6 -6 3
M     10      1      2        0   0   1   1   1
M     10      2      2        0   0   2   2   2
I     11      1       22 -2 6 -6 -2
M     11      1      1        0   0   1   1   1
M     11      2      1        0   0   2   2   2
C     11      1      1        0   0 502   0   0
C     11      2      1        0 501   0 502 501
I     11      2       22 -4 6 -6 -4
M     11      1      2        0   0   1   1   1
M     11      2      2        0   0   2   2   2
I     12      1       22 2 6 -6 2
M     12      1      1        0   0   1   1   1
M     12      2      1        0   0   2   2   2
C     12      1      1        0 502 501   0 502
C     12      2      1        0   0   0 501   0
I     12      2       22 4 6 -6 4
M     12      1      2        0   0   1   1   1
M     12      2      2        0   0   2   2   2
I     13      1       22 22 6 -6 22
M     13      1      1        0   0   1   1   1
M     13      2      1        0   0   2   2   2
C     13      1      1        0   0 501   0   0
C     13      2      1        0   0   0 501   0
I     14      1       22 22 6 -6 22
M     14      1      1        0   0   1   1   1
M     14      2      1        0   0   2   2   2
C     14      1      1        0   0 501   0   0
C     14      2      1        0   0   0 501   0
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_leshouche_info_file(\
            self.give_pos(self.created_files[0]),
            self.myfks_me)
        self.assertFileContains(self.created_files[0], goal)

    @PostponeToEW
    def test_write_fks_info_file_EW(self):
        """tests the correct writing of fks_info.inc file, containing the 
        relevant informations for all the splittings"""

        goal = \
"""      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(14), FKS_J_D(14)
      INTEGER FKS_J_FROM_I_D(14, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(14, NEXTERNAL), PDG_TYPE_D(14, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(14, NEXTERNAL)

      DATA FKS_I_D / 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 /
      DATA FKS_J_D / 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(1, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(2, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(3, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(4, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(5, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(6, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(7, 5, JPOS), JPOS = 0, 1)  / 1, 2 /

      DATA (FKS_J_FROM_I_D(8, 5, JPOS), JPOS = 0, 1)  / 1, 2 /

      DATA (FKS_J_FROM_I_D(9, 5, JPOS), JPOS = 0, 1)  / 1, 2 /

      DATA (FKS_J_FROM_I_D(10, 5, JPOS), JPOS = 0, 1)  / 1, 2 /

      DATA (FKS_J_FROM_I_D(11, 5, JPOS), JPOS = 0, 1)  / 1, 2 /

      DATA (FKS_J_FROM_I_D(12, 5, JPOS), JPOS = 0, 1)  / 1, 2 /

      DATA (FKS_J_FROM_I_D(13, 5, JPOS), JPOS = 0, 2)  / 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(14, 5, JPOS), JPOS = 0, 2)  / 2, 3, 4 /

C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / 1, 1, 3, 
     $ -3, 1 /
      DATA (PARTICLE_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / 1, 1, 3, 
     $ -3, 1 /
      DATA (PARTICLE_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / -3, 1, 3, 
     $ -3, -3 /
      DATA (PARTICLE_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / 3, 1, 3, 
     $ -3, 3 /
      DATA (PARTICLE_TYPE_D(5, IPOS), IPOS=1, NEXTERNAL) / -3, 1, 3, 
     $ -3, -3 /
      DATA (PARTICLE_TYPE_D(6, IPOS), IPOS=1, NEXTERNAL) / 3, 1, 3, 
     $ -3, 3 /
      DATA (PARTICLE_TYPE_D(7, IPOS), IPOS=1, NEXTERNAL) / 1, 1, 3, 
     $ -3, 1 /
      DATA (PARTICLE_TYPE_D(8, IPOS), IPOS=1, NEXTERNAL) / 1, 1, 3, 
     $ -3, 1 /
      DATA (PARTICLE_TYPE_D(9, IPOS), IPOS=1, NEXTERNAL) / 1, -3, 3, 
     $ -3, -3 /
      DATA (PARTICLE_TYPE_D(10, IPOS), IPOS=1, NEXTERNAL) / 1, 3, 3, 
     $ -3, 3 /
      DATA (PARTICLE_TYPE_D(11, IPOS), IPOS=1, NEXTERNAL) / 1, -3, 3, 
     $ -3, -3 /
      DATA (PARTICLE_TYPE_D(12, IPOS), IPOS=1, NEXTERNAL) / 1, 3, 3, 
     $ -3, 3 /
      DATA (PARTICLE_TYPE_D(13, IPOS), IPOS=1, NEXTERNAL) / 1, 1, 3, 
     $ -3, 1 /
      DATA (PARTICLE_TYPE_D(14, IPOS), IPOS=1, NEXTERNAL) / 1, 1, 3, 
     $ -3, 1 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / -11, 22, 6, -6, 
     $ -11 /
      DATA (PDG_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / 11, 22, 6, 
     $ -6, 11 /
      DATA (PDG_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / -1, 22, 6, -6, 
     $ -1 /
      DATA (PDG_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / 1, 22, 6, -6, 1 /
      DATA (PDG_TYPE_D(5, IPOS), IPOS=1, NEXTERNAL) / -2, 22, 6, -6, 
     $ -2 /
      DATA (PDG_TYPE_D(6, IPOS), IPOS=1, NEXTERNAL) / 2, 22, 6, -6, 2 /
      DATA (PDG_TYPE_D(7, IPOS), IPOS=1, NEXTERNAL) / 22, -11, 6, -6, 
     $ -11 /
      DATA (PDG_TYPE_D(8, IPOS), IPOS=1, NEXTERNAL) / 22, 11, 6, 
     $ -6, 11 /
      DATA (PDG_TYPE_D(9, IPOS), IPOS=1, NEXTERNAL) / 22, -1, 6, -6, 
     $ -1 /
      DATA (PDG_TYPE_D(10, IPOS), IPOS=1, NEXTERNAL) / 22, 1, 6, 
     $ -6, 1 /
      DATA (PDG_TYPE_D(11, IPOS), IPOS=1, NEXTERNAL) / 22, -2, 6, -6, 
     $ -2 /
      DATA (PDG_TYPE_D(12, IPOS), IPOS=1, NEXTERNAL) / 22, 2, 6, 
     $ -6, 2 /
      DATA (PDG_TYPE_D(13, IPOS), IPOS=1, NEXTERNAL) / 22, 22, 6, 
     $ -6, 22 /
      DATA (PDG_TYPE_D(14, IPOS), IPOS=1, NEXTERNAL) / 22, 22, 6, 
     $ -6, 22 /

C     
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
      DATA (PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL) / 1D0, 0D0
     $ , 2D0/3D0, -2D0/3D0, 1D0 /
      DATA (PARTICLE_CHARGE_D(2, IPOS), IPOS=1, NEXTERNAL) / 
     $ -1D0, 0D0, 2D0/3D0, -2D0/3D0, -1D0 /
      DATA (PARTICLE_CHARGE_D(3, IPOS), IPOS=1, NEXTERNAL) / 1D0
     $ /3D0, 0D0, 2D0/3D0, -2D0/3D0, 1D0/3D0 /
      DATA (PARTICLE_CHARGE_D(4, IPOS), IPOS=1, NEXTERNAL) / 
     $ -1D0/3D0, 0D0, 2D0/3D0, -2D0/3D0, -1D0/3D0 /
      DATA (PARTICLE_CHARGE_D(5, IPOS), IPOS=1, NEXTERNAL) / 
     $ -2D0/3D0, 0D0, 2D0/3D0, -2D0/3D0, -2D0/3D0 /
      DATA (PARTICLE_CHARGE_D(6, IPOS), IPOS=1, NEXTERNAL) / 2D0
     $ /3D0, 0D0, 2D0/3D0, -2D0/3D0, 2D0/3D0 /
      DATA (PARTICLE_CHARGE_D(7, IPOS), IPOS=1, NEXTERNAL) / 0D0, 1D0
     $ , 2D0/3D0, -2D0/3D0, 1D0 /
      DATA (PARTICLE_CHARGE_D(8, IPOS), IPOS=1, NEXTERNAL) / 0D0, 
     $ -1D0, 2D0/3D0, -2D0/3D0, -1D0 /
      DATA (PARTICLE_CHARGE_D(9, IPOS), IPOS=1, NEXTERNAL) / 0D0, 1D0
     $ /3D0, 2D0/3D0, -2D0/3D0, 1D0/3D0 /
      DATA (PARTICLE_CHARGE_D(10, IPOS), IPOS=1, NEXTERNAL) / 0D0, 
     $ -1D0/3D0, 2D0/3D0, -2D0/3D0, -1D0/3D0 /
      DATA (PARTICLE_CHARGE_D(11, IPOS), IPOS=1, NEXTERNAL) / 0D0, 
     $ -2D0/3D0, 2D0/3D0, -2D0/3D0, -2D0/3D0 /
      DATA (PARTICLE_CHARGE_D(12, IPOS), IPOS=1, NEXTERNAL) / 0D0, 2D0
     $ /3D0, 2D0/3D0, -2D0/3D0, 2D0/3D0 /
      DATA (PARTICLE_CHARGE_D(13, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 2D0/3D0, -2D0/3D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(14, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 2D0/3D0, -2D0/3D0, 0D0 /
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_fks_info_file(\
            writers.FortranWriter(self.give_pos(self.created_files[0])),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains(self.created_files[0], goal)

    def test_write_leshouche_file_EW(self):
        """tests if the leshouche.inc file is correctly written for the born process
        """
        goal = \
"""      DATA (IDUP(I,1),I=1,4)/22,22,6,-6/
      DATA (MOTHUP(1,I,  1),I=1, 4)/  0,  0,  1,  1/
      DATA (MOTHUP(2,I,  1),I=1, 4)/  0,  0,  2,  2/
      DATA (ICOLUP(1,I,  1),I=1, 4)/  0,  0,501,  0/
      DATA (ICOLUP(2,I,  1),I=1, 4)/  0,  0,  0,501/
"""    
        process_exporter = export_fks.ProcessExporterFortranFKS()

        nflows = \
            process_exporter.write_leshouche_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)  

        self.assertFileContains(self.created_files[0], goal) 

    def test_write_pmass_file_EW(self):
        """tests if the pmass.inc file is correctly written.
        The function called is the one of the FortranProcessExporterV4 class.
        """
        goal = \
"""      PMASS(1)=ZERO
      PMASS(2)=ZERO
      PMASS(3)=ABS(MDL_MT)
      PMASS(4)=ABS(MDL_MT)
      PMASS(5)=ZERO
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        
        process_exporter.write_pmass_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me.real_processes[0].matrix_element)        

        self.assertFileContains(self.created_files[0], goal) 

    @PostponeToEW
    def test_write_born_fks_EW(self):
        """tests if the born.f file containing the born matrix element
        is correctly written
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
C     Process: a a > t t~ QED=2 QCD=0 [ QED ]
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
      PARAMETER (THEL=NCOMB*NCROSS*14)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   2)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(NCROSS*2)
C     
C     LOCAL VARIABLES 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY(14)
      COMPLEX*16 T,T1
      REAL*8 BORN
      REAL*8 ZERO
      PARAMETER(ZERO=0D0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS,14)
      INTEGER NGOOD(14),IGOOD(NCOMB,14),JHEL
      DATA NGOOD /14*0/
      SAVE IGOOD,JHEL
      REAL*8 HWGT
      REAL*8 XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
      INTEGER J, JJ
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
      DATA NTRY /14*0/
      DATA XTRY, XREJ /0,0/
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
      DOUBLE PRECISION HEL_FAC
      INTEGER GET_HEL,SKIP(14)
      COMMON/CBORN/HEL_FAC,GET_HEL,SKIP
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER GLU_IJ
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IDEN_VALUES(14)
      DATA IDEN_VALUES /4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4/
      INTEGER IJ_VALUES(14)
      DATA IJ_VALUES /1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4/
C     ----------
C     BEGIN CODE
C     ----------
      IDEN(1)=IDEN_VALUES(NFKSPROCESS)
      GLU_IJ = IJ_VALUES(NFKSPROCESS)
      NTRY(NFKSPROCESS)=NTRY(NFKSPROCESS)+1
      IF (NTRY(NFKSPROCESS).LT.2) THEN
        SKIP(NFKSPROCESS)=1
        DO WHILE(NHEL(GLU_IJ ,SKIP(NFKSPROCESS)).NE.1)
          SKIP(NFKSPROCESS)=SKIP(NFKSPROCESS)+1
        ENDDO
        SKIP(NFKSPROCESS)=SKIP(NFKSPROCESS)-1
      ENDIF
      DO IPROC=1,NCROSS
        DO IHEL=1,NEXTERNAL-1
          JC(IHEL) = +1
        ENDDO

        DO IHEL=1,NGRAPHS
          AMP2(IHEL)=0D0
        ENDDO
        DO IHEL=1,INT(JAMP2(0))
          JAMP2(IHEL)=0D0
        ENDDO
        IF (CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $       ,J)) THEN
              CALCULATEDBORN=.FALSE.
              WRITE (*,*) 'momenta not the same in Born'
              STOP
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
        IF (ISUM_HEL .EQ. 0 .OR. NTRY(NFKSPROCESS) .LT. 2) THEN
          HEL_FAC=1D0
          DO IHEL=1,NCOMB
            IF ((GOODHEL(IHEL,IPROC,NFKSPROCESS) .OR. NTRY(NFKSPROCESS
     $       ) .LT. 2).AND.NHEL(GLU_IJ ,IHEL).EQ.-1) THEN
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
     $         EL,IPROC,NFKSPROCESS)) THEN
                GOODHEL(IHEL,IPROC,NFKSPROCESS)=.TRUE.
                NGOOD(NFKSPROCESS) = NGOOD(NFKSPROCESS) +1
                IGOOD(NGOOD(NFKSPROCESS),NFKSPROCESS) = IHEL
              ENDIF
            ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD(NFKSPROCESS))
        ELSE  !RANDOM HELICITY
          DO J=1,ISUM_HEL
            HWGT = REAL(NGOOD(NFKSPROCESS))/REAL(ISUM_HEL)
            HEL_FAC=HWGT
            IF (GET_HEL.EQ.0) THEN
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD(NFKSPROCESS)) JHEL=1
              IHEL = IGOOD(JHEL,NFKSPROCESS)
              GET_HEL=IHEL
            ELSE
              IHEL=GET_HEL
            ENDIF
            IF(GOODHEL(IHEL,IPROC,NFKSPROCESS)) THEN
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
            WRITE(HEL_BUFF(2),'(16i5)')(NHEL(I,IHEL+SKIP(NFKSPROCESS))
     $       ,I=1,NEXTERNAL-1)
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

C     Process: a a > t t~ QED=2 QCD=0 [ QED ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS,    NEIGEN
      PARAMETER (NGRAPHS=   2,NEIGEN=  1)
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
C     INCLUDE 'born_maxamps.inc'
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=6, NCOLOR=1)
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
      INTEGER GET_HEL,SKIP(14)
      COMMON/CBORN/HEL_FAC,GET_HEL,SKIP
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INCLUDE 'coupl.inc'
      INTEGER GLU_IJ
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IJ_VALUES(14)
      DATA IJ_VALUES /1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 4/
C     
C     COLOR DATA
C     
      DATA DENOM(1)/1/
      DATA (CF(I,  1),I=  1,  1) /    3/
C     1 T(3,4)
C     ----------
C     BEGIN CODE
C     ----------
      GLU_IJ = IJ_VALUES(NFKSPROCESS)
      BORN = 0D0
      BORNTILDE = (0D0,0D0)
      BACK_HEL = NHEL(GLU_IJ)
      DO IHEL=-1,1,2
        NHEL(GLU_IJ) = IHEL
        IF (.NOT. CALCULATEDBORN) THEN
          CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
          CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
          CALL OXXXXX(P(0,3),MT,NHEL(3),+1*IC(3),W(1,3))
          CALL IXXXXX(P(0,4),MT,NHEL(4),-1*IC(4),W(1,4))
          CALL FFV1_1(W(1,3),W(1,1),GC_2,MT,WT,W(1,5))
C         Amplitude(s) for diagram number 1
          CALL FFV1_0(W(1,4),W(1,5),W(1,2),GC_2,AMP(1))
          CALL FFV1_2(W(1,4),W(1,1),GC_2,MT,WT,W(1,5))
C         Amplitude(s) for diagram number 2
          CALL FFV1_0(W(1,5),W(1,3),W(1,2),GC_2,AMP(2))
          DO I=1,NGRAPHS
            IF(IHEL.EQ.-1)THEN
              SAVEAMP(I,HELL)=AMP(I)
            ELSEIF(IHEL.EQ.1)THEN
              SAVEAMP(I,HELL+SKIP(NFKSPROCESS))=AMP(I)
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
              AMP(I)=SAVEAMP(I,HELL+SKIP(NFKSPROCESS))
            ELSE
              WRITE(*,*) 'ERROR #1 in born.f'
              STOP
            ENDIF
          ENDDO
        ENDIF
        JAMP(1)=-AMP(1)-AMP(2)
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




"""  % misc.get_pkg_info()

        process_exporter = export_fks.ProcessExporterFortranFKS()

        nflows = \
            process_exporter.write_born_fks(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me, 
                    self.myfortranmodel)  

        self.assertFileContains(self.created_files[0], goal) 

    @PostponeToEW
    def test_write_matrix_element_fks_EW(self):
        """tests if the matrix_x.f file containing the matrix element 
        for a given real process is correctly written.
        The real process tested is aa > ttxa (real_processes[-1])
        """
        goal = \
"""      SUBROUTINE SMATRIX_1(P,ANS)
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
C     Process: a a > t t~ a WEIGHTED=6 QED=3 QCD=0 [ QED ]
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
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY,T_IDENT(NCOMB)
      REAL*8 T,T_SAVE(NCOMB)
      SAVE T_SAVE,T_IDENT
      REAL*8 MATRIX_1
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
      DATA IDEN/ 4/
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
          IF (NTRY.LT.2) THEN
C           for the first ps-point, check for helicities that give
C           identical matrix elements
            T=MATRIX_1(P ,NHEL(1,IHEL),JC(1))
            T_SAVE(IHEL)=T
            T_IDENT(IHEL)=-1
            DO I=1,IHEL-1
              IF (T.EQ.0D0) EXIT
              IF (T_SAVE(I).EQ.0D0) CYCLE
              IF (ABS(T/T_SAVE(I)-1D0) .LT. 1D-12) THEN
C               WRITE (*,*) 'FOUND IDENTICAL',T,IHEL,T_SAVE(I),I
                T_IDENT(IHEL) = I
              ENDIF
            ENDDO
          ELSE
            IF (T_IDENT(IHEL).GT.0) THEN
C             if two helicity states are identical, dont recompute
              T=T_SAVE(T_IDENT(IHEL))
              T_SAVE(IHEL)=T
            ELSE
              T=MATRIX_1(P ,NHEL(1,IHEL),JC(1))
              T_SAVE(IHEL)=T
            ENDIF
          ENDIF
C         add to the sum of helicities
          ANS=ANS+T
          IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)
      END


      REAL*8 FUNCTION MATRIX_1(P,NHEL,IC)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     
C     Returns amplitude squared summed/avg over colors
C     for the point with external lines W(0:6,NEXTERNAL)
C     
C     Process: a a > t t~ a WEIGHTED=6 QED=3 QCD=0 [ QED ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=6)
      INCLUDE 'nexternal.inc'
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=9, NCOLOR=1)
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
      DATA (CF(I,  1),I=  1,  1) /    3/
C     1 T(3,4)
C     ----------
C     BEGIN CODE
C     ----------
      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL OXXXXX(P(0,3),MT,NHEL(3),+1*IC(3),W(1,3))
      CALL IXXXXX(P(0,4),MT,NHEL(4),-1*IC(4),W(1,4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1*IC(5),W(1,5))
      CALL FFV1_1(W(1,3),W(1,1),GC_2,MT,WT,W(1,6))
      CALL FFV1_2(W(1,4),W(1,2),GC_2,MT,WT,W(1,7))
C     Amplitude(s) for diagram number 1
      CALL FFV1_0(W(1,7),W(1,6),W(1,5),GC_2,AMP(1))
      CALL FFV1_2(W(1,4),W(1,5),GC_2,MT,WT,W(1,8))
C     Amplitude(s) for diagram number 2
      CALL FFV1_0(W(1,8),W(1,6),W(1,2),GC_2,AMP(2))
      CALL FFV1_2(W(1,4),W(1,1),GC_2,MT,WT,W(1,6))
      CALL FFV1_1(W(1,3),W(1,2),GC_2,MT,WT,W(1,4))
C     Amplitude(s) for diagram number 3
      CALL FFV1_0(W(1,6),W(1,4),W(1,5),GC_2,AMP(3))
      CALL FFV1_1(W(1,3),W(1,5),GC_2,MT,WT,W(1,9))
C     Amplitude(s) for diagram number 4
      CALL FFV1_0(W(1,6),W(1,9),W(1,2),GC_2,AMP(4))
C     Amplitude(s) for diagram number 5
      CALL FFV1_0(W(1,8),W(1,4),W(1,1),GC_2,AMP(5))
C     Amplitude(s) for diagram number 6
      CALL FFV1_0(W(1,7),W(1,9),W(1,1),GC_2,AMP(6))
      JAMP(1)=-AMP(1)-AMP(2)-AMP(3)-AMP(4)-AMP(5)-AMP(6)

      MATRIX_1 = 0.D0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        MATRIX_1 = MATRIX_1+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
      ENDDO
      END


""" % misc.get_pkg_info()

        process_exporter = export_fks.ProcessExporterFortranFKS()
        
        nflows = \
            process_exporter.write_matrix_element_fks(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me.real_processes[-1].matrix_element, 1,
                    self.myfortranmodel)  

        self.assertFileContains(self.created_files[0], goal) 

    def test_get_fks_j_from_i_lines_EW(self):
        """Test that the lines corresponding to the fks_j_from_i array, to be 
        written in fks.inc. 
        """
        lines = ['DATA (FKS_J_FROM_I_D(2, 5, JPOS), JPOS = 0, 1)  / 1, 1 /','']

        process_exporter = export_fks.ProcessExporterFortranFKS()
        self.assertEqual(lines, process_exporter.get_fks_j_from_i_lines(self.myfks_me.real_processes[1], 2))

    @PostponeToEW
    def test_write_pdf_file_EW(self):
        """tests if the parton_lum_x.f file containing the parton distributions 
        for a given real process is correctly written.
        The real process tested is gg > ttxg (real_processes[0])
        """
        goal = \
"""      SUBROUTINE DLUM_1(LUM)
C     ****************************************************            
C         
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     RETURNS PARTON LUMINOSITIES FOR MADFKS                          
C        
C     
C     Process: a a > t t~ a WEIGHTED=6 QED=3 QCD=0 [ QED ]
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
      PARAMETER (CONV=389379660D0)  !CONV TO PICOBARNS             
C     
C     ARGUMENTS                                                       
C         
C     
      DOUBLE PRECISION PP(0:3,NEXTERNAL), LUM
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
      DATA ICROSS/1/
C     ----------                                                      
C         
C     BEGIN CODE                                                      
C         
C     ----------                                                      
C         
      LUM = 0D0
      IF (IMIRROR.EQ.2) THEN
        IF (ABS(LPP(2)) .GE. 1) THEN
          LP=SIGN(1,LPP(2))
          A1=PDG2PDF(ABS(LPP(2)),7*LP,XBK(2),DSQRT(Q2FACT(2)))
        ENDIF
        IF (ABS(LPP(1)) .GE. 1) THEN
          LP=SIGN(1,LPP(1))
          A2=PDG2PDF(ABS(LPP(1)),7*LP,XBK(1),DSQRT(Q2FACT(1)))
        ENDIF
        PD(0) = 0D0
        IPROC = 0
        IPROC=IPROC+1  ! a a > t t~ a
        PD(IPROC) = A1*A2
      ELSE
        IF (ABS(LPP(1)) .GE. 1) THEN
          LP=SIGN(1,LPP(1))
          A1=PDG2PDF(ABS(LPP(1)),7*LP,XBK(1),DSQRT(Q2FACT(1)))
        ENDIF
        IF (ABS(LPP(2)) .GE. 1) THEN
          LP=SIGN(1,LPP(2))
          A2=PDG2PDF(ABS(LPP(2)),7*LP,XBK(2),DSQRT(Q2FACT(2)))
        ENDIF
        PD(0) = 0D0
        IPROC = 0
        IPROC=IPROC+1  ! a a > t t~ a
        PD(IPROC) = A1*A2
      ENDIF
      DO I=1,IPROC
        LUM = LUM + PD(I) * CONV
      ENDDO
      RETURN
      END

""" % misc.get_pkg_info()

        process_exporter = export_fks.ProcessExporterFortranFKS()

        nflows = \
            process_exporter.write_pdf_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me.real_processes[-1].matrix_element, 1,
                    self.myfortranmodel)  

        self.assertFileContains(self.created_files[0], goal) 

    @PostponeToEW
    def test_write_configs_file_born_EW(self):
        """Tests if the configs.inc file is corretly written 
        for the born matrix element.
        """
        goal = \
"""C     Diagram 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(I, -1,   1),I=1,2)/  1,  3/
      DATA TPRID(  -1,   1)/       6/
      DATA (IFOREST(I, -2,   1),I=1,2)/ -1,  4/
C     Diagram 2
      DATA MAPCONFIG(   2)/   2/
      DATA (IFOREST(I, -1,   2),I=1,2)/  1,  4/
      DATA TPRID(  -1,   2)/       6/
      DATA (IFOREST(I, -2,   2),I=1,2)/ -1,  3/
C     Number of configs
      DATA MAPCONFIG(0)/   2/
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        
        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)

        self.assertFileContains(self.created_files[0], goal) 

    def test_write_decayBW_file_EW(self):
        """Tests if the decayBW.inc file is correctly written 
        for the born process.
        """
        goal = \
"""
"""

        process_exporter = export_fks.ProcessExporterFortranFKS()

        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)

        process_exporter.write_decayBW_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    s_and_t_channels)        

        self.assertFileContains(self.created_files[0], goal)  

    def test_write_nfksconfigs_file_EW(self):
        """tests if the nFKSconfigs.inc file is correctly written"""
        goal = \
"""      INTEGER FKS_CONFIGS
      PARAMETER (FKS_CONFIGS=14)


"""        
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_nfksconfigs_file(\
            writers.FortranWriter(self.give_pos(self.created_files[0])),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains(self.created_files[0], goal)

    def test_write_props_file_born_EW(self):
        """Tests if the props.inc file is correctly written 
        for the born matrix element.
        """
        goal = \
"""      PMASS( -1,   1)  = ABS(MDL_MT)
      PWIDTH( -1,   1) = ABS(MDL_WT)
      POW( -1,   1) = 1
      PMASS( -1,   2)  = ABS(MDL_MT)
      PWIDTH( -1,   2) = ABS(MDL_WT)
      POW( -1,   2) = 1
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        
        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)
        
        process_exporter.write_props_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel,
                    s_and_t_channels)        

        self.assertFileContains(self.created_files[0], goal) 

    def test_get_color_data_lines_from_color_matrix_EW(self):
        """tests if the color data lines are correctly extracted from a given
        color matrix. 
        The first color link is used.
        """
        
        goal = ["DATA DENOM(1)/1/",
                "DATA (CF(I,  1),I=  1,  1) /    3/"
                ]
        process_exporter = export_fks.ProcessExporterFortranFKS()

        lines = process_exporter.get_color_data_lines_from_color_matrix(
                    self.myfks_me.color_links[0]['link_matrix'])
        
        for line, goalline in zip(lines, goal):
            self.assertEqual(line.upper(), goalline)

    @PostponeToEW
    def test_write_b_sf_fks_EW(self):
        """Tests the correct writing of a b_sf_xxx.f file, containing one color
        linked born.
        """

        goal = \
"""      SUBROUTINE SB_SF_001(P1,ANS)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     AND HELICITIES
C     FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL-1)
C     
C     Process: a a > t t~ QED=2 QCD=0 [ QED ]
C     spectators: 3 3 

C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'nexternal.inc'
      INTEGER                 NCOMB,     NCROSS
      PARAMETER (             NCOMB=  16, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS*14)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   2)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1),ANS(NCROSS)
C     
C     LOCAL VARIABLES 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY(14)
      REAL*8 T
      REAL*8 B_SF_001
      REAL*8 ZERO
      PARAMETER(ZERO=0D0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS,14)
      DATA NTRY /14*0/
      INTEGER NGOOD(14),IGOOD(NCOMB,14),JHEL
      DATA NGOOD /14*0/
      SAVE IGOOD,JHEL
      REAL*8 HWGT
      INTEGER J,JJ
      INCLUDE 'born_nhel.inc'
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM

      CHARACTER*79         HEL_BUFF(2)
      COMMON/TO_HELICITY/  HEL_BUFF

      DATA GOODHEL/THEL*.FALSE./
      DOUBLE PRECISION HEL_FAC
      INTEGER GET_HEL,SKIP(14)
      COMMON/CBORN/HEL_FAC,GET_HEL,SKIP
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IDEN_VALUES(14)
      DATA IDEN_VALUES /4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4/
C     ----------
C     BEGIN CODE
C     ----------
      IDEN(1)=IDEN_VALUES(NFKSPROCESS)
      NTRY(NFKSPROCESS)=NTRY(NFKSPROCESS)+1
      DO IPROC=1,NCROSS
        DO IHEL=1,NEXTERNAL-1
          JC(IHEL) = +1
        ENDDO
        IF (CALCULATEDBORN) THEN
          DO J=1,NEXTERNAL-1
            IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $       ,J)) THEN
              CALCULATEDBORN=.FALSE.
              WRITE(*,*) 'Error in sb_sf: momenta not the same in the
     $          born'
              STOP
            ENDIF
          ENDDO
        ENDIF
        IF (.NOT.CALCULATEDBORN) THEN
          WRITE(*,*) 'Error in sb_sf: color_linked borns should be
     $      called only with calculatedborn = true'
          STOP
        ENDIF
        ANS(IPROC) = 0D0
        IF (GET_HEL .EQ. 0 .OR. NTRY(NFKSPROCESS) .LT. 2) THEN
          DO IHEL=1,NCOMB
            IF (GOODHEL(IHEL,IPROC,NFKSPROCESS) .OR. NTRY(NFKSPROCESS
     $       ) .LT. 2) THEN
              T=B_SF_001(P1,NHEL(1,IHEL),IHEL,JC(1))
              ANS(IPROC)=ANS(IPROC)+T
              IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC,NFKSPROCESS
     $         )) THEN
                GOODHEL(IHEL,IPROC,NFKSPROCESS)=.TRUE.
                NGOOD(NFKSPROCESS) = NGOOD(NFKSPROCESS) +1
                IGOOD(NGOOD(NFKSPROCESS),NFKSPROCESS) = IHEL
              ENDIF
            ENDIF
          ENDDO
        ELSE  !RANDOM HELICITY
          HWGT = REAL(NGOOD(NFKSPROCESS))
          IHEL=GET_HEL
          T=B_SF_001(P1,NHEL(1,IHEL),IHEL,JC(1))
          ANS(IPROC)=ANS(IPROC)+T*HWGT
        ENDIF
        ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END


      REAL*8 FUNCTION B_SF_001(P,NHEL,HELL,IC)
C     
C     Generated by MadGraph 5 v. %(version)s, %(date)s
C     By the MadGraph Development Team
C     Please visit us at https://launchpad.net/madgraph5
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

C     Process: a a > t t~ QED=2 QCD=0 [ QED ]
C     spectators: 3 3 

C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS,    NEIGEN
      PARAMETER (NGRAPHS=   2,NEIGEN=  1)
      INCLUDE 'nexternal.inc'
      INTEGER    NWAVEFUNCS, NCOLOR1, NCOLOR2
      PARAMETER (NWAVEFUNCS=6, NCOLOR1=1, NCOLOR2=1)
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
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INCLUDE 'coupl.inc'
C     
C     COLOR DATA
C     
      DATA DENOM(1)/1/
      DATA (CF(I,  1),I=  1,  1) /    3/
C     ----------
C     BEGIN CODE
C     ----------

      IF (.NOT. CALCULATEDBORN) THEN
        WRITE(*,*) 'Error in b_sf: color_linked borns should be called
     $    only with calculatedborn = true'
        STOP
      ELSEIF (CALCULATEDBORN) THEN
        DO I=1,NGRAPHS
          AMP(I)=SAVEAMP(I,HELL)
        ENDDO
      ENDIF
      JAMP1(1)=-AMP(1)-AMP(2)
      JAMP2(1)=+4D0/9D0*(-AMP(1)-AMP(2))
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
        
        process_exporter = export_fks.ProcessExporterFortranFKS()

        process_exporter.write_b_sf_fks(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me, 0, self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)


    def test_write_born_nhel_file_EW(self):
        """tests if the born_nhel.inc file is correctly written"""
        goal = \
"""      INTEGER    MAX_BHEL, MAX_BCOL
      PARAMETER (MAX_BHEL=16)
      PARAMETER(MAX_BCOL=1)
"""        
        process_exporter = export_fks.ProcessExporterFortranFKS()

        calls, ncolor = \
            process_exporter.write_born_fks(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me,
                    self.myfortranmodel)    

        nflows = \
            process_exporter.write_leshouche_file(
                    writers.FortranWriter(self.give_pos('test2')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)  
                
        process_exporter.write_born_nhel_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    self.myfks_me.born_matrix_element,
                    nflows,
                    self.myfortranmodel,
                    ncolor)  

        self.assertFileContains(self.created_files[0], goal) 

    def test_write_nexternal_file_EW(self):
        """tests if the nexternal.inc file is correctly written.
        The real process used is uux_uxug (real_processes[5])
        """
        goal = \
"""      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=5)
      INTEGER    NINCOMING
      PARAMETER (NINCOMING=2)
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        
        process_exporter.write_nexternal_file(
                    writers.FortranWriter(self.give_pos(self.created_files[0])),
                    5, 2)        

        self.assertFileContains(self.created_files[0], goal)  

    def test_write_sborn_sf_EW(self):
        """Tests the correct writing of the sborn_sf file, containing the calls 
        to the different color linked borns."""
        
        goal = \
"""      SUBROUTINE SBORN_SF(P_BORN,M,N,WGT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P_BORN(0:3,NEXTERNAL-1),WGT
      DOUBLE COMPLEX WGT1(2)
      INTEGER M,N

C     b_sf_001 links partons 3 and 3 
      IF (M.EQ.3 .AND. N.EQ.3) THEN
        CALL SB_SF_001(P_BORN,WGT)

C       b_sf_002 links partons 3 and 4 
      ELSEIF ((M.EQ.3 .AND. N.EQ.4).OR.(M.EQ.4 .AND. N.EQ.3)) THEN
        CALL SB_SF_002(P_BORN,WGT)

C       b_sf_003 links partons 4 and 4 
      ELSEIF (M.EQ.4 .AND. N.EQ.4) THEN
        CALL SB_SF_003(P_BORN,WGT)

      ELSE
        WGT = 0D0
      ENDIF

      RETURN
      END
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()

        process_exporter.write_sborn_sf(\
            writers.FortranWriter(self.give_pos(self.created_files[0])),
            self.myfks_me.color_links,
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains(self.created_files[0], goal)
