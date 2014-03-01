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
#===============================================================================
# IOExportFKSTest
#===============================================================================
class IOExportFKSTest(unittest.TestCase,
                     test_file_writers.CheckFileCreate):
    """Test class for the export fks module"""

    def setUp(self):
        if not hasattr(self, 'myfks_me') or \
           not hasattr(self, 'myfortranmodel') or \
           not hasattr(self, 'myreals'):

            created_files = ['test']

            mymodel = import_ufo.import_model('sm')
            IOExportFKSTest.myfortranmodel = helas_call_writers.FortranUFOHelasCallWriter(mymodel)

            myleglist = MG.MultiLegList()

        # we test g g > t t~
            myleglist.append(MG.MultiLeg({'ids':[21], 'state':False}))
            myleglist.append(MG.MultiLeg({'ids':[21], 'state':False}))
            myleglist.append(MG.MultiLeg({'ids':[6], 'state':True}))
            myleglist.append(MG.MultiLeg({'ids':[-6], 'state':True}))

            myproc = MG.ProcessDefinition({'legs': myleglist,
                                 'model': mymodel,
                                 'orders':{'QCD': 2, 'QED': 0},
                                 'perturbation_couplings': ['QCD'],
                                 'NLO_mode': 'real'})
            my_process_definitions = MG.ProcessDefinitionList([myproc])

            myfksmulti = fks_base.FKSMultiProcess(\
                    {'process_definitions': my_process_definitions})
            

            fkshelasmulti = fks_helas.FKSHelasMultiProcess(myfksmulti)
            IOExportFKSTest.myfks_me = fkshelasmulti['matrix_elements'][0]
            IOExportFKSTest.myreals = fkshelasmulti['real_matrix_elements']


        tearDown = test_file_writers.CheckFileCreate.clean_files


    def test_write_maxconfigs(self):
        goal = \
"""      INTEGER LMAXCONFIGS
      PARAMETER (LMAXCONFIGS=22)"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_maxconfigs_file(\
            writers.FortranWriter(self.give_pos('test')),\
            self.myreals)
        self.assertFileContains('test', goal)


    def test_write_mparticles(self):
        goal = \
"""      INTEGER MAX_PARTICLES, MAX_BRANCH
      PARAMETER (MAX_PARTICLES=5)
      PARAMETER (MAX_BRANCH=MAX_PARTICLES-1)"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_maxparticles_file(\
            writers.FortranWriter(self.give_pos('test')),\
            self.myreals)
        self.assertFileContains('test', goal)


    def test_write_lh_order(self):
        """tests the correct writing of the B-LH order file"""

        goal = \
"""#OLE_order written by MadGraph5_aMC@NLO

MatrixElementSquareType CHaveraged
CorrectionType          QCD
IRregularisation        CDR
AlphasPower             2
AlphaPower              0
NJetSymmetrizeFinal     Yes
ModelFile               ./param_card.dat
Parameters              alpha_s

# process
21 21 -> 6 -6 
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_lh_order(\
            self.give_pos('test'),\
            self.myfks_me, 'MadLoop')
        self.assertFileContains('test', goal)


    def test_write_real_me_wrapper(self):
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
        CALL SMATRIX_1(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.3) THEN
        CALL SMATRIX_1(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.4) THEN
        CALL SMATRIX_1(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.5) THEN
        CALL SMATRIX_2(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.6) THEN
        CALL SMATRIX_3(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.7) THEN
        CALL SMATRIX_4(P, WGT)
      ELSEIF (NFKSPROCESS.EQ.8) THEN
        CALL SMATRIX_5(P, WGT)
      ELSE
        WRITE(*,*) 'ERROR: invalid n in real_matrix :', NFKSPROCESS
        STOP
      ENDIF
      RETURN
      END

"""

        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_real_me_wrapper(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains('test', goal)


    def test_write_pdf_wrapper(self):
        """tests the correct writing of the parton_lum_chooser file, 
        that chooses thepdfs for the different real emissions"""

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
        CALL DLUM_1(DLUM)
      ELSEIF (NFKSPROCESS.EQ.3) THEN
        CALL DLUM_1(DLUM)
      ELSEIF (NFKSPROCESS.EQ.4) THEN
        CALL DLUM_1(DLUM)
      ELSEIF (NFKSPROCESS.EQ.5) THEN
        CALL DLUM_2(DLUM)
      ELSEIF (NFKSPROCESS.EQ.6) THEN
        CALL DLUM_3(DLUM)
      ELSEIF (NFKSPROCESS.EQ.7) THEN
        CALL DLUM_4(DLUM)
      ELSEIF (NFKSPROCESS.EQ.8) THEN
        CALL DLUM_5(DLUM)
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
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains('test', goal)


    def test_write_leshouche_info_file(self):
        """tests the correct writing of fks_info.inc file, containing the 
        relevant informations for all the splittings"""

        goal = \
"""      INTEGER MAXPROC_USED, MAXFLOW_USED
      PARAMETER (MAXPROC_USED = 4)
      PARAMETER (MAXFLOW_USED = 6)
      INTEGER IDUP_D(8,5,MAXPROC_USED)
      INTEGER MOTHUP_D(8,2,5,MAXPROC_USED)
      INTEGER ICOLUP_D(8,2,5,MAXFLOW_USED)
      INTEGER ILH

      DATA (IDUP_D(1,ILH,1),ILH=1,5)/21,21,6,-6,21/
      DATA (MOTHUP_D(1,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(1,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(1,1,ILH,  1),ILH=1, 5)/501,502,501,  0,504/
      DATA (ICOLUP_D(1,2,ILH,  1),ILH=1, 5)/502,503,  0,504,503/
      DATA (ICOLUP_D(1,1,ILH,  2),ILH=1, 5)/501,504,501,  0,504/
      DATA (ICOLUP_D(1,2,ILH,  2),ILH=1, 5)/502,503,  0,503,502/
      DATA (ICOLUP_D(1,1,ILH,  3),ILH=1, 5)/503,501,501,  0,504/
      DATA (ICOLUP_D(1,2,ILH,  3),ILH=1, 5)/502,503,  0,504,502/
      DATA (ICOLUP_D(1,1,ILH,  4),ILH=1, 5)/504,501,501,  0,504/
      DATA (ICOLUP_D(1,2,ILH,  4),ILH=1, 5)/502,503,  0,502,503/
      DATA (ICOLUP_D(1,1,ILH,  5),ILH=1, 5)/504,502,501,  0,504/
      DATA (ICOLUP_D(1,2,ILH,  5),ILH=1, 5)/502,503,  0,503,501/
      DATA (ICOLUP_D(1,1,ILH,  6),ILH=1, 5)/503,504,501,  0,504/
      DATA (ICOLUP_D(1,2,ILH,  6),ILH=1, 5)/502,503,  0,502,501/

      DATA (IDUP_D(2,ILH,1),ILH=1,5)/21,21,6,-6,21/
      DATA (MOTHUP_D(2,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(2,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(2,1,ILH,  1),ILH=1, 5)/501,502,501,  0,504/
      DATA (ICOLUP_D(2,2,ILH,  1),ILH=1, 5)/502,503,  0,504,503/
      DATA (ICOLUP_D(2,1,ILH,  2),ILH=1, 5)/501,504,501,  0,504/
      DATA (ICOLUP_D(2,2,ILH,  2),ILH=1, 5)/502,503,  0,503,502/
      DATA (ICOLUP_D(2,1,ILH,  3),ILH=1, 5)/503,501,501,  0,504/
      DATA (ICOLUP_D(2,2,ILH,  3),ILH=1, 5)/502,503,  0,504,502/
      DATA (ICOLUP_D(2,1,ILH,  4),ILH=1, 5)/504,501,501,  0,504/
      DATA (ICOLUP_D(2,2,ILH,  4),ILH=1, 5)/502,503,  0,502,503/
      DATA (ICOLUP_D(2,1,ILH,  5),ILH=1, 5)/504,502,501,  0,504/
      DATA (ICOLUP_D(2,2,ILH,  5),ILH=1, 5)/502,503,  0,503,501/
      DATA (ICOLUP_D(2,1,ILH,  6),ILH=1, 5)/503,504,501,  0,504/
      DATA (ICOLUP_D(2,2,ILH,  6),ILH=1, 5)/502,503,  0,502,501/

      DATA (IDUP_D(3,ILH,1),ILH=1,5)/21,21,6,-6,21/
      DATA (MOTHUP_D(3,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(3,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(3,1,ILH,  1),ILH=1, 5)/501,502,501,  0,504/
      DATA (ICOLUP_D(3,2,ILH,  1),ILH=1, 5)/502,503,  0,504,503/
      DATA (ICOLUP_D(3,1,ILH,  2),ILH=1, 5)/501,504,501,  0,504/
      DATA (ICOLUP_D(3,2,ILH,  2),ILH=1, 5)/502,503,  0,503,502/
      DATA (ICOLUP_D(3,1,ILH,  3),ILH=1, 5)/503,501,501,  0,504/
      DATA (ICOLUP_D(3,2,ILH,  3),ILH=1, 5)/502,503,  0,504,502/
      DATA (ICOLUP_D(3,1,ILH,  4),ILH=1, 5)/504,501,501,  0,504/
      DATA (ICOLUP_D(3,2,ILH,  4),ILH=1, 5)/502,503,  0,502,503/
      DATA (ICOLUP_D(3,1,ILH,  5),ILH=1, 5)/504,502,501,  0,504/
      DATA (ICOLUP_D(3,2,ILH,  5),ILH=1, 5)/502,503,  0,503,501/
      DATA (ICOLUP_D(3,1,ILH,  6),ILH=1, 5)/503,504,501,  0,504/
      DATA (ICOLUP_D(3,2,ILH,  6),ILH=1, 5)/502,503,  0,502,501/

      DATA (IDUP_D(4,ILH,1),ILH=1,5)/21,21,6,-6,21/
      DATA (MOTHUP_D(4,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(4,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(4,1,ILH,  1),ILH=1, 5)/501,502,501,  0,504/
      DATA (ICOLUP_D(4,2,ILH,  1),ILH=1, 5)/502,503,  0,504,503/
      DATA (ICOLUP_D(4,1,ILH,  2),ILH=1, 5)/501,504,501,  0,504/
      DATA (ICOLUP_D(4,2,ILH,  2),ILH=1, 5)/502,503,  0,503,502/
      DATA (ICOLUP_D(4,1,ILH,  3),ILH=1, 5)/503,501,501,  0,504/
      DATA (ICOLUP_D(4,2,ILH,  3),ILH=1, 5)/502,503,  0,504,502/
      DATA (ICOLUP_D(4,1,ILH,  4),ILH=1, 5)/504,501,501,  0,504/
      DATA (ICOLUP_D(4,2,ILH,  4),ILH=1, 5)/502,503,  0,502,503/
      DATA (ICOLUP_D(4,1,ILH,  5),ILH=1, 5)/504,502,501,  0,504/
      DATA (ICOLUP_D(4,2,ILH,  5),ILH=1, 5)/502,503,  0,503,501/
      DATA (ICOLUP_D(4,1,ILH,  6),ILH=1, 5)/503,504,501,  0,504/
      DATA (ICOLUP_D(4,2,ILH,  6),ILH=1, 5)/502,503,  0,502,501/

      DATA (IDUP_D(5,ILH,1),ILH=1,5)/-1,21,6,-6,-1/
      DATA (MOTHUP_D(5,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(5,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(5,1,ILH,  1),ILH=1, 5)/  0,502,502,  0,  0/
      DATA (ICOLUP_D(5,2,ILH,  1),ILH=1, 5)/501,503,  0,501,503/
      DATA (ICOLUP_D(5,1,ILH,  2),ILH=1, 5)/  0,502,502,  0,  0/
      DATA (ICOLUP_D(5,2,ILH,  2),ILH=1, 5)/501,503,  0,503,501/
      DATA (ICOLUP_D(5,1,ILH,  3),ILH=1, 5)/  0,501,502,  0,  0/
      DATA (ICOLUP_D(5,2,ILH,  3),ILH=1, 5)/501,503,  0,503,502/
      DATA (ICOLUP_D(5,1,ILH,  4),ILH=1, 5)/  0,501,502,  0,  0/
      DATA (ICOLUP_D(5,2,ILH,  4),ILH=1, 5)/501,503,  0,502,503/
      DATA (IDUP_D(5,ILH,2),ILH=1,5)/-3,21,6,-6,-3/
      DATA (MOTHUP_D(5,1,ILH,  2),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(5,2,ILH,  2),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(5,ILH,3),ILH=1,5)/-2,21,6,-6,-2/
      DATA (MOTHUP_D(5,1,ILH,  3),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(5,2,ILH,  3),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(5,ILH,4),ILH=1,5)/-4,21,6,-6,-4/
      DATA (MOTHUP_D(5,1,ILH,  4),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(5,2,ILH,  4),ILH=1, 5)/  0,  0,  2,  2,  2/

      DATA (IDUP_D(6,ILH,1),ILH=1,5)/1,21,6,-6,1/
      DATA (MOTHUP_D(6,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(6,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(6,1,ILH,  1),ILH=1, 5)/503,501,501,  0,502/
      DATA (ICOLUP_D(6,2,ILH,  1),ILH=1, 5)/  0,503,  0,502,  0/
      DATA (ICOLUP_D(6,1,ILH,  2),ILH=1, 5)/502,501,501,  0,502/
      DATA (ICOLUP_D(6,2,ILH,  2),ILH=1, 5)/  0,503,  0,503,  0/
      DATA (ICOLUP_D(6,1,ILH,  3),ILH=1, 5)/503,502,501,  0,502/
      DATA (ICOLUP_D(6,2,ILH,  3),ILH=1, 5)/  0,503,  0,501,  0/
      DATA (ICOLUP_D(6,1,ILH,  4),ILH=1, 5)/501,502,501,  0,502/
      DATA (ICOLUP_D(6,2,ILH,  4),ILH=1, 5)/  0,503,  0,503,  0/
      DATA (IDUP_D(6,ILH,2),ILH=1,5)/3,21,6,-6,3/
      DATA (MOTHUP_D(6,1,ILH,  2),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(6,2,ILH,  2),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(6,ILH,3),ILH=1,5)/2,21,6,-6,2/
      DATA (MOTHUP_D(6,1,ILH,  3),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(6,2,ILH,  3),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(6,ILH,4),ILH=1,5)/4,21,6,-6,4/
      DATA (MOTHUP_D(6,1,ILH,  4),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(6,2,ILH,  4),ILH=1, 5)/  0,  0,  2,  2,  2/

      DATA (IDUP_D(7,ILH,1),ILH=1,5)/21,-1,6,-6,-1/
      DATA (MOTHUP_D(7,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(7,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(7,1,ILH,  1),ILH=1, 5)/501,  0,502,  0,  0/
      DATA (ICOLUP_D(7,2,ILH,  1),ILH=1, 5)/503,501,  0,503,502/
      DATA (ICOLUP_D(7,1,ILH,  2),ILH=1, 5)/501,  0,502,  0,  0/
      DATA (ICOLUP_D(7,2,ILH,  2),ILH=1, 5)/503,501,  0,502,503/
      DATA (ICOLUP_D(7,1,ILH,  3),ILH=1, 5)/502,  0,502,  0,  0/
      DATA (ICOLUP_D(7,2,ILH,  3),ILH=1, 5)/503,501,  0,503,501/
      DATA (ICOLUP_D(7,1,ILH,  4),ILH=1, 5)/502,  0,502,  0,  0/
      DATA (ICOLUP_D(7,2,ILH,  4),ILH=1, 5)/503,501,  0,501,503/
      DATA (IDUP_D(7,ILH,2),ILH=1,5)/21,-3,6,-6,-3/
      DATA (MOTHUP_D(7,1,ILH,  2),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(7,2,ILH,  2),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(7,ILH,3),ILH=1,5)/21,-2,6,-6,-2/
      DATA (MOTHUP_D(7,1,ILH,  3),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(7,2,ILH,  3),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(7,ILH,4),ILH=1,5)/21,-4,6,-6,-4/
      DATA (MOTHUP_D(7,1,ILH,  4),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(7,2,ILH,  4),ILH=1, 5)/  0,  0,  2,  2,  2/

      DATA (IDUP_D(8,ILH,1),ILH=1,5)/21,1,6,-6,1/
      DATA (MOTHUP_D(8,1,ILH,  1),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(8,2,ILH,  1),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (ICOLUP_D(8,1,ILH,  1),ILH=1, 5)/501,503,501,  0,502/
      DATA (ICOLUP_D(8,2,ILH,  1),ILH=1, 5)/503,  0,  0,502,  0/
      DATA (ICOLUP_D(8,1,ILH,  2),ILH=1, 5)/501,502,501,  0,502/
      DATA (ICOLUP_D(8,2,ILH,  2),ILH=1, 5)/503,  0,  0,503,  0/
      DATA (ICOLUP_D(8,1,ILH,  3),ILH=1, 5)/502,503,501,  0,502/
      DATA (ICOLUP_D(8,2,ILH,  3),ILH=1, 5)/503,  0,  0,501,  0/
      DATA (ICOLUP_D(8,1,ILH,  4),ILH=1, 5)/502,501,501,  0,502/
      DATA (ICOLUP_D(8,2,ILH,  4),ILH=1, 5)/503,  0,  0,503,  0/
      DATA (IDUP_D(8,ILH,2),ILH=1,5)/21,3,6,-6,3/
      DATA (MOTHUP_D(8,1,ILH,  2),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(8,2,ILH,  2),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(8,ILH,3),ILH=1,5)/21,2,6,-6,2/
      DATA (MOTHUP_D(8,1,ILH,  3),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(8,2,ILH,  3),ILH=1, 5)/  0,  0,  2,  2,  2/
      DATA (IDUP_D(8,ILH,4),ILH=1,5)/21,4,6,-6,4/
      DATA (MOTHUP_D(8,1,ILH,  4),ILH=1, 5)/  0,  0,  1,  1,  1/
      DATA (MOTHUP_D(8,2,ILH,  4),ILH=1, 5)/  0,  0,  2,  2,  2/

"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_leshouche_info_file(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains('test', goal)


    def test_write_fks_info_file(self):
        """tests the correct writing of fks_info.inc file, containing the 
        relevant informations for all the splittings"""

        goal = \
"""      INTEGER IPOS, JPOS
      INTEGER FKS_I_D(8), FKS_J_D(8)
      INTEGER FKS_J_FROM_I_D(8, NEXTERNAL, 0:NEXTERNAL)
      INTEGER PARTICLE_TYPE_D(8, NEXTERNAL), PDG_TYPE_D(8, NEXTERNAL)
      REAL*8 PARTICLE_CHARGE_D(8, NEXTERNAL)

      DATA FKS_I_D / 5, 5, 5, 5, 5, 5, 5, 5 /
      DATA FKS_J_D / 1, 2, 3, 4, 1, 1, 2, 2 /

      DATA (FKS_J_FROM_I_D(1, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(2, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(3, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(4, 5, JPOS), JPOS = 0, 4)  / 4, 1, 2, 3, 4 /

      DATA (FKS_J_FROM_I_D(5, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(6, 5, JPOS), JPOS = 0, 1)  / 1, 1 /

      DATA (FKS_J_FROM_I_D(7, 5, JPOS), JPOS = 0, 1)  / 1, 2 /

      DATA (FKS_J_FROM_I_D(8, 5, JPOS), JPOS = 0, 1)  / 1, 2 /


C     
C     Particle type:
C     octet = 8, triplet = 3, singlet = 1
      DATA (PARTICLE_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / 8, 8, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / 8, 8, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / 8, 8, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / 8, 8, 3, 
     $ -3, 8 /
      DATA (PARTICLE_TYPE_D(5, IPOS), IPOS=1, NEXTERNAL) / -3, 8, 3, 
     $ -3, -3 /
      DATA (PARTICLE_TYPE_D(6, IPOS), IPOS=1, NEXTERNAL) / 3, 8, 3, 
     $ -3, 3 /
      DATA (PARTICLE_TYPE_D(7, IPOS), IPOS=1, NEXTERNAL) / 8, -3, 3, 
     $ -3, -3 /
      DATA (PARTICLE_TYPE_D(8, IPOS), IPOS=1, NEXTERNAL) / 8, 3, 3, 
     $ -3, 3 /

C     
C     Particle type according to PDG:
C     
      DATA (PDG_TYPE_D(1, IPOS), IPOS=1, NEXTERNAL) / 21, 21, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(2, IPOS), IPOS=1, NEXTERNAL) / 21, 21, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(3, IPOS), IPOS=1, NEXTERNAL) / 21, 21, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(4, IPOS), IPOS=1, NEXTERNAL) / 21, 21, 6, 
     $ -6, 21 /
      DATA (PDG_TYPE_D(5, IPOS), IPOS=1, NEXTERNAL) / -1, 21, 6, -6, 
     $ -1 /
      DATA (PDG_TYPE_D(6, IPOS), IPOS=1, NEXTERNAL) / 1, 21, 6, -6, 1 /
      DATA (PDG_TYPE_D(7, IPOS), IPOS=1, NEXTERNAL) / 21, -1, 6, -6, 
     $ -1 /
      DATA (PDG_TYPE_D(8, IPOS), IPOS=1, NEXTERNAL) / 21, 1, 6, -6, 1 /

C     
C     Particle charge:
C     charge is set 0. with QCD corrections, which is irrelevant
      DATA (PARTICLE_CHARGE_D(1, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(2, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(3, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(4, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(5, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(6, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(7, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /
      DATA (PARTICLE_CHARGE_D(8, IPOS), IPOS=1, NEXTERNAL) / 0D0, 0D0
     $ , 0D0, 0D0, 0D0 /

"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_fks_info_file(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains('test', goal)


    def test_write_sborn_sf(self):
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
      IF ((M.EQ.1 .AND. N.EQ.2).OR.(M.EQ.2 .AND. N.EQ.1)) THEN
        CALL SB_SF_001(P_BORN,WGT)

C       b_sf_002 links partons 1 and 3 
      ELSEIF ((M.EQ.1 .AND. N.EQ.3).OR.(M.EQ.3 .AND. N.EQ.1)) THEN
        CALL SB_SF_002(P_BORN,WGT)

C       b_sf_003 links partons 1 and 4 
      ELSEIF ((M.EQ.1 .AND. N.EQ.4).OR.(M.EQ.4 .AND. N.EQ.1)) THEN
        CALL SB_SF_003(P_BORN,WGT)

C       b_sf_004 links partons 2 and 3 
      ELSEIF ((M.EQ.2 .AND. N.EQ.3).OR.(M.EQ.3 .AND. N.EQ.2)) THEN
        CALL SB_SF_004(P_BORN,WGT)

C       b_sf_005 links partons 2 and 4 
      ELSEIF ((M.EQ.2 .AND. N.EQ.4).OR.(M.EQ.4 .AND. N.EQ.2)) THEN
        CALL SB_SF_005(P_BORN,WGT)

C       b_sf_006 links partons 3 and 3 
      ELSEIF (M.EQ.3 .AND. N.EQ.3) THEN
        CALL SB_SF_006(P_BORN,WGT)

C       b_sf_007 links partons 3 and 4 
      ELSEIF ((M.EQ.3 .AND. N.EQ.4).OR.(M.EQ.4 .AND. N.EQ.3)) THEN
        CALL SB_SF_007(P_BORN,WGT)

C       b_sf_008 links partons 4 and 4 
      ELSEIF (M.EQ.4 .AND. N.EQ.4) THEN
        CALL SB_SF_008(P_BORN,WGT)

      ELSE
        WGT = 0D0
      ENDIF

      RETURN
      END
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()

        process_exporter.write_sborn_sf(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me.color_links,
            self.myfortranmodel)

        #print open(self.give_pos('test')).read()
        self.assertFileContains('test', goal)


    def test_write_leshouche_file(self):
        """tests if the leshouche.inc file is correctly written for the born process
        """
        goal = \
"""      DATA (IDUP(I,1),I=1,4)/21,21,6,-6/
      DATA (MOTHUP(1,I,  1),I=1, 4)/  0,  0,  1,  1/
      DATA (MOTHUP(2,I,  1),I=1, 4)/  0,  0,  2,  2/
      DATA (ICOLUP(1,I,  1),I=1, 4)/501,502,501,  0/
      DATA (ICOLUP(2,I,  1),I=1, 4)/502,503,  0,503/
      DATA (ICOLUP(1,I,  2),I=1, 4)/503,501,501,  0/
      DATA (ICOLUP(2,I,  2),I=1, 4)/502,503,  0,502/
"""    
        process_exporter = export_fks.ProcessExporterFortranFKS()

        nflows = \
            process_exporter.write_leshouche_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)  

        self.assertFileContains('test', goal) 


    def test_write_nexternal_file(self):
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
                    writers.FortranWriter(self.give_pos('test')),
                    5, 2)        

        self.assertFileContains('test', goal)  


    def test_write_pmass_file(self):
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
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[0].matrix_element)        

        self.assertFileContains('test', goal) 


    def test_write_pdf_file(self):
        """tests if the parton_lum_x.f file containing the parton distributions 
        for a given real process is correctly written.
        The real process tested is gg > ttxg (real_processes[0])
        """
        goal = \
"""      SUBROUTINE DLUM_1(LUM)
C     ****************************************************            
C         
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     RETURNS PARTON LUMINOSITIES FOR MADFKS                          
C        
C     
C     Process: g g > t t~ g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
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
      DOUBLE PRECISION G1
      DOUBLE PRECISION G2
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
      DATA G1/1*1D0/
      DATA G2/1*1D0/
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
          G1=PDG2PDF(ABS(LPP(2)),0*LP,XBK(2),DSQRT(Q2FACT(2)))
        ENDIF
        IF (ABS(LPP(1)) .GE. 1) THEN
          LP=SIGN(1,LPP(1))
          G2=PDG2PDF(ABS(LPP(1)),0*LP,XBK(1),DSQRT(Q2FACT(1)))
        ENDIF
        PD(0) = 0D0
        IPROC = 0
        IPROC=IPROC+1  ! g g > t t~ g
        PD(IPROC) = G1*G2
      ELSE
        IF (ABS(LPP(1)) .GE. 1) THEN
          LP=SIGN(1,LPP(1))
          G1=PDG2PDF(ABS(LPP(1)),0*LP,XBK(1),DSQRT(Q2FACT(1)))
        ENDIF
        IF (ABS(LPP(2)) .GE. 1) THEN
          LP=SIGN(1,LPP(2))
          G2=PDG2PDF(ABS(LPP(2)),0*LP,XBK(2),DSQRT(Q2FACT(2)))
        ENDIF
        PD(0) = 0D0
        IPROC = 0
        IPROC=IPROC+1  ! g g > t t~ g
        PD(IPROC) = G1*G2
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
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[0].matrix_element, 1,
                    self.myfortranmodel)  

        self.assertFileContains('test', goal) 


    def test_write_matrix_element_fks(self):
        """tests if the matrix_x.f file containing the matrix element 
        for a given real process is correctly written.
        The real process tested is gg > ttxg (real_processes[0])
        """
        goal = \
"""      SUBROUTINE SMATRIX_1(P,ANS)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     Returns amplitude squared summed/avg over colors
C     and helicities
C     for the point in phase space P(0:3,NEXTERNAL)
C     
C     Process: g g > t t~ g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'nexternal.inc'
      INTEGER     NCOMB
      PARAMETER ( NCOMB=32)
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL),ANS
C     
C     LOCAL VARIABLES 
C     
      INTEGER IHEL,IDEN,I,T_IDENT(NCOMB)
      REAL*8 MATRIX_1
      REAL*8 T,T_SAVE(NCOMB)
      SAVE T_SAVE,T_IDENT
      INTEGER NHEL(NEXTERNAL,NCOMB)
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
      LOGICAL GOODHEL(NCOMB)
      DATA GOODHEL/NCOMB*.FALSE./
      INTEGER NTRY
      DATA NTRY/0/
      DATA IDEN/256/
C     ----------
C     BEGIN CODE
C     ----------
      NTRY=NTRY+1
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
          IF (NTRY.LT.2) THEN
C           for the first ps-point, check for helicities that give
C           identical matrix elements
            T=MATRIX_1(P ,NHEL(1,IHEL))
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
              T=MATRIX_1(P ,NHEL(1,IHEL))
              T_SAVE(IHEL)=T
            ENDIF
          ENDIF
C         add to the sum of helicities
          ANS=ANS+T
          IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IHEL)) THEN
            GOODHEL(IHEL)=.TRUE.
          ENDIF
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)
      END


      REAL*8 FUNCTION MATRIX_1(P,NHEL)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     Returns amplitude squared summed/avg over colors
C     for the point with external lines W(0:6,NEXTERNAL)
C     
C     Process: g g > t t~ g WEIGHTED=3 QED=0 QCD=3 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=18)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=12, NCOLOR=6)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      INTEGER IC(NEXTERNAL)
      DATA IC /NEXTERNAL*1/
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR), W(20,NWAVEFUNCS)
C     
C     COLOR DATA
C     
      DATA DENOM(1)/9/
      DATA (CF(I,  1),I=  1,  6) /   64,   -8,   -8,    1,    1,   10/
C     1 T(1,2,5,3,4)
      DATA DENOM(2)/9/
      DATA (CF(I,  2),I=  1,  6) /   -8,   64,    1,   10,   -8,    1/
C     1 T(1,5,2,3,4)
      DATA DENOM(3)/9/
      DATA (CF(I,  3),I=  1,  6) /   -8,    1,   64,   -8,   10,    1/
C     1 T(2,1,5,3,4)
      DATA DENOM(4)/9/
      DATA (CF(I,  4),I=  1,  6) /    1,   10,   -8,   64,    1,   -8/
C     1 T(2,5,1,3,4)
      DATA DENOM(5)/9/
      DATA (CF(I,  5),I=  1,  6) /    1,   -8,   10,    1,   64,   -8/
C     1 T(5,1,2,3,4)
      DATA DENOM(6)/9/
      DATA (CF(I,  6),I=  1,  6) /   10,    1,    1,   -8,   -8,   64/
C     1 T(5,2,1,3,4)
C     ----------
C     BEGIN CODE
C     ----------
      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
      CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
      CALL OXXXXX(P(0,3),MDL_MT,NHEL(3),+1*IC(3),W(1,3))
      CALL IXXXXX(P(0,4),MDL_MT,NHEL(4),-1*IC(4),W(1,4))
      CALL VXXXXX(P(0,5),ZERO,NHEL(5),+1*IC(5),W(1,5))
      CALL VVV1P0_1(W(1,1),W(1,2),GC_10,ZERO,ZERO,W(1,6))
      CALL FFV1P0_3(W(1,4),W(1,3),GC_11,ZERO,ZERO,W(1,7))
C     Amplitude(s) for diagram number 1
      CALL VVV1_0(W(1,6),W(1,7),W(1,5),GC_10,AMP(1))
      CALL FFV1_1(W(1,3),W(1,5),GC_11,MDL_MT,MDL_WT,W(1,8))
C     Amplitude(s) for diagram number 2
      CALL FFV1_0(W(1,4),W(1,8),W(1,6),GC_11,AMP(2))
      CALL FFV1_2(W(1,4),W(1,5),GC_11,MDL_MT,MDL_WT,W(1,9))
C     Amplitude(s) for diagram number 3
      CALL FFV1_0(W(1,9),W(1,3),W(1,6),GC_11,AMP(3))
      CALL FFV1_1(W(1,3),W(1,1),GC_11,MDL_MT,MDL_WT,W(1,6))
      CALL FFV1_2(W(1,4),W(1,2),GC_11,MDL_MT,MDL_WT,W(1,10))
C     Amplitude(s) for diagram number 4
      CALL FFV1_0(W(1,10),W(1,6),W(1,5),GC_11,AMP(4))
      CALL VVV1P0_1(W(1,2),W(1,5),GC_10,ZERO,ZERO,W(1,11))
C     Amplitude(s) for diagram number 5
      CALL FFV1_0(W(1,4),W(1,6),W(1,11),GC_11,AMP(5))
C     Amplitude(s) for diagram number 6
      CALL FFV1_0(W(1,9),W(1,6),W(1,2),GC_11,AMP(6))
      CALL FFV1_2(W(1,4),W(1,1),GC_11,MDL_MT,MDL_WT,W(1,6))
      CALL FFV1_1(W(1,3),W(1,2),GC_11,MDL_MT,MDL_WT,W(1,12))
C     Amplitude(s) for diagram number 7
      CALL FFV1_0(W(1,6),W(1,12),W(1,5),GC_11,AMP(7))
C     Amplitude(s) for diagram number 8
      CALL FFV1_0(W(1,6),W(1,3),W(1,11),GC_11,AMP(8))
C     Amplitude(s) for diagram number 9
      CALL FFV1_0(W(1,6),W(1,8),W(1,2),GC_11,AMP(9))
      CALL VVV1P0_1(W(1,1),W(1,5),GC_10,ZERO,ZERO,W(1,6))
C     Amplitude(s) for diagram number 10
      CALL FFV1_0(W(1,4),W(1,12),W(1,6),GC_11,AMP(10))
C     Amplitude(s) for diagram number 11
      CALL FFV1_0(W(1,10),W(1,3),W(1,6),GC_11,AMP(11))
C     Amplitude(s) for diagram number 12
      CALL VVV1_0(W(1,6),W(1,2),W(1,7),GC_10,AMP(12))
C     Amplitude(s) for diagram number 13
      CALL FFV1_0(W(1,9),W(1,12),W(1,1),GC_11,AMP(13))
C     Amplitude(s) for diagram number 14
      CALL FFV1_0(W(1,10),W(1,8),W(1,1),GC_11,AMP(14))
C     Amplitude(s) for diagram number 15
      CALL VVV1_0(W(1,1),W(1,11),W(1,7),GC_10,AMP(15))
      CALL VVVV1P0_1(W(1,1),W(1,2),W(1,5),GC_12,ZERO,ZERO,W(1,11))
      CALL VVVV3P0_1(W(1,1),W(1,2),W(1,5),GC_12,ZERO,ZERO,W(1,7))
      CALL VVVV4P0_1(W(1,1),W(1,2),W(1,5),GC_12,ZERO,ZERO,W(1,10))
C     Amplitude(s) for diagram number 16
      CALL FFV1_0(W(1,4),W(1,3),W(1,11),GC_11,AMP(16))
      CALL FFV1_0(W(1,4),W(1,3),W(1,7),GC_11,AMP(17))
      CALL FFV1_0(W(1,4),W(1,3),W(1,10),GC_11,AMP(18))
      JAMP(1)=-AMP(1)+IMAG1*AMP(3)+IMAG1*AMP(5)-AMP(6)+AMP(15)-AMP(18)
     $ +AMP(16)
      JAMP(2)=-AMP(4)-IMAG1*AMP(5)+IMAG1*AMP(11)+AMP(12)-AMP(15)
     $ -AMP(17)-AMP(16)
      JAMP(3)=+AMP(1)-IMAG1*AMP(3)+IMAG1*AMP(10)-AMP(12)-AMP(13)
     $ +AMP(18)+AMP(17)
      JAMP(4)=-AMP(7)+IMAG1*AMP(8)-IMAG1*AMP(10)+AMP(12)-AMP(15)
     $ -AMP(17)-AMP(16)
      JAMP(5)=+AMP(1)+IMAG1*AMP(2)-IMAG1*AMP(11)-AMP(12)-AMP(14)
     $ +AMP(18)+AMP(17)
      JAMP(6)=-AMP(1)-IMAG1*AMP(2)-IMAG1*AMP(8)-AMP(9)+AMP(15)-AMP(18)
     $ +AMP(16)
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
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.real_processes[0].matrix_element, 1,
                    self.myfortranmodel)  

        self.assertFileContains('test', goal) 



    def test_write_born_fks(self):
        """tests if the born.f file containing the born matrix element
        is correctly written
        """
        goal = \
"""      SUBROUTINE SBORN(P1,ANS)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     AND HELICITIES
C     FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C     
C     Process: g g > t t~ QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
      INCLUDE 'genps.inc'
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  16 )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*8)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   3)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(2)
C     
C     LOCAL VARIABLES 
C     
      INTEGER IHEL,IDEN,I,J,JJ,GLU_IJ
      REAL*8 BORN,BORNS(2)
      COMPLEX*16 BORNTILDE
      INTEGER NTRY(8)
      DATA NTRY /8*0/
      INTEGER NHEL(NEXTERNAL-1,NCOMB)
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
      INTEGER IDEN_VALUES(8)
      DATA IDEN_VALUES /256, 256, 256, 256, 256, 256, 256, 256/
      INTEGER IJ_VALUES(8)
      DATA IJ_VALUES /1, 2, 3, 4, 1, 1, 2, 2/
C     
C     GLOBAL VARIABLES
C     
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXAMPS)
      COMMON/TO_AMPS/  AMP2,       JAMP2
      DATA JAMP2(0) /   2/
      LOGICAL GOODHEL(NCOMB,8)
      COMMON /C_GOODHEL/GOODHEL
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM
      DOUBLE PRECISION HEL_FAC
      INTEGER GET_HEL,SKIP(8)
      COMMON/CBORN/HEL_FAC,GET_HEL,SKIP
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
C     ----------
C     BEGIN CODE
C     ----------
      IDEN=IDEN_VALUES(NFKSPROCESS)
      GLU_IJ = IJ_VALUES(NFKSPROCESS)
      NTRY(NFKSPROCESS)=NTRY(NFKSPROCESS)+1
      IF (NTRY(NFKSPROCESS).LT.2) THEN
        SKIP(NFKSPROCESS)=1
        DO WHILE(NHEL(GLU_IJ ,SKIP(NFKSPROCESS)).NE.1)
          SKIP(NFKSPROCESS)=SKIP(NFKSPROCESS)+1
        ENDDO
        SKIP(NFKSPROCESS)=SKIP(NFKSPROCESS)-1
      ENDIF
      DO JJ=1,NGRAPHS
        AMP2(JJ)=0D0
      ENDDO
      DO JJ=1,INT(JAMP2(0))
        JAMP2(JJ)=0D0
      ENDDO
      IF (CALCULATEDBORN) THEN
        DO J=1,NEXTERNAL-1
          IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $     ,J)) THEN
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
      ANS(1) = 0D0
      ANS(2) = 0D0
      HEL_FAC=1D0
      DO IHEL=1,NCOMB
        IF (NHEL(GLU_IJ,IHEL).LE.0) THEN
          IF ((GOODHEL(IHEL,NFKSPROCESS) .OR. GOODHEL(IHEL+SKIP(NFKSPRO
     $     CESS),NFKSPROCESS) .OR. NTRY(NFKSPROCESS) .LT. 2) ) THEN
            ANS(1)=ANS(1)+BORN(P1,NHEL(1,IHEL),IHEL,BORNTILDE,BORNS)
            ANS(2)=ANS(2)+BORNTILDE
            IF ( BORNS(1).NE.0D0 .AND. .NOT. GOODHEL(IHEL,NFKSPROCESS
     $       ) ) THEN
              GOODHEL(IHEL,NFKSPROCESS)=.TRUE.
            ENDIF
            IF ( BORNS(2).NE.0D0 .AND. .NOT. GOODHEL(IHEL+SKIP(NFKSPROC
     $       ESS),NFKSPROCESS) ) THEN
              GOODHEL(IHEL+SKIP(NFKSPROCESS),NFKSPROCESS)=.TRUE.
            ENDIF
          ENDIF
        ENDIF
      ENDDO
      ANS(1)=ANS(1)/DBLE(IDEN)
      ANS(2)=ANS(2)/DBLE(IDEN)
      CALCULATEDBORN=.TRUE.
      END


      REAL*8 FUNCTION BORN(P,NHEL,HELL,BORNTILDE,BORNS)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

C     Process: g g > t t~ QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NGRAPHS,    NEIGEN
      PARAMETER (NGRAPHS=   3,NEIGEN=  1)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=7, NCOLOR=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1 = (0D0,1D0))
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'genps.inc'
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL-1),BORNS(2)
      INTEGER NHEL(NEXTERNAL-1), HELL
      COMPLEX*16 BORNTILDE
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J,IHEL,BACK_HEL,GLU_IJ
      INTEGER IC(NEXTERNAL-1),NMO
      PARAMETER (NMO=NEXTERNAL-1)
      DATA IC /NMO*1/
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR), W(20,NWAVEFUNCS)
     $ , JAMPH(2, NCOLOR)
C     
C     GLOBAL VARIABLES
C     
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXAMPS)
      COMMON/TO_AMPS/  AMP2,       JAMP2
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      DOUBLE PRECISION HEL_FAC
      INTEGER GET_HEL,SKIP(8)
      COMMON/CBORN/HEL_FAC,GET_HEL,SKIP
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      INTEGER IJ_VALUES(8)
      DATA IJ_VALUES /1, 2, 3, 4, 1, 1, 2, 2/
C     
C     COLOR DATA
C     
      DATA DENOM(1)/3/
      DATA (CF(I,  1),I=  1,  2) /   16,   -2/
C     1 T(1,2,3,4)
      DATA DENOM(2)/3/
      DATA (CF(I,  2),I=  1,  2) /   -2,   16/
C     1 T(2,1,3,4)
C     ----------
C     BEGIN CODE
C     ----------
      GLU_IJ = IJ_VALUES(NFKSPROCESS)
      BORN = 0D0
      BORNTILDE = (0D0,0D0)
      BACK_HEL = NHEL(GLU_IJ)
      BORNS(1) = 0D0
      BORNS(2) = 0D0
      DO IHEL=-1,1,2
        IF (IHEL.EQ.-1.OR.NHEL(GLU_IJ).NE.0) THEN
          IF (NHEL(GLU_IJ).NE.0) NHEL(GLU_IJ) = IHEL
          IF (.NOT. CALCULATEDBORN) THEN
            CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1*IC(1),W(1,1))
            CALL VXXXXX(P(0,2),ZERO,NHEL(2),-1*IC(2),W(1,2))
            CALL OXXXXX(P(0,3),MDL_MT,NHEL(3),+1*IC(3),W(1,3))
            CALL IXXXXX(P(0,4),MDL_MT,NHEL(4),-1*IC(4),W(1,4))
            CALL VVV1P0_1(W(1,1),W(1,2),GC_10,ZERO,ZERO,W(1,5))
C           Amplitude(s) for diagram number 1
            CALL FFV1_0(W(1,4),W(1,3),W(1,5),GC_11,AMP(1))
            CALL FFV1_1(W(1,3),W(1,1),GC_11,MDL_MT,MDL_WT,W(1,5))
C           Amplitude(s) for diagram number 2
            CALL FFV1_0(W(1,4),W(1,5),W(1,2),GC_11,AMP(2))
            CALL FFV1_2(W(1,4),W(1,1),GC_11,MDL_MT,MDL_WT,W(1,5))
C           Amplitude(s) for diagram number 3
            CALL FFV1_0(W(1,5),W(1,3),W(1,2),GC_11,AMP(3))
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
          JAMP(1)=+IMAG1*AMP(1)-AMP(2)
          JAMP(2)=-IMAG1*AMP(1)-AMP(3)
          DO I = 1, NCOLOR
            ZTEMP = (0.D0,0.D0)
            DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
            ENDDO
            BORNS(2-(1-IHEL)/2)=BORNS(2-(1-IHEL)/2)+ZTEMP*DCONJG(JAMP(I
     $       ))/DENOM(I)
          ENDDO
          DO I = 1, NGRAPHS
            AMP2(I)=AMP2(I)+AMP(I)*DCONJG(AMP(I))
          ENDDO
          DO I = 1, NCOLOR
            JAMP2(I)=JAMP2(I)+JAMP(I)*DCONJG(JAMP(I))
            JAMPH(2-(1-IHEL)/2,I)=JAMP(I)
          ENDDO
        ENDIF
      ENDDO
      BORN=BORNS(1)+BORNS(2)
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMPH(2,J)
        ENDDO
        BORNTILDE = BORNTILDE + ZTEMP*DCONJG(JAMPH(1,I))/DENOM(I)
      ENDDO
      NHEL(GLU_IJ) = BACK_HEL
      END


      BLOCK DATA GOODHELS
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  16 )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*8)
      LOGICAL GOODHEL(NCOMB,8)
      COMMON /C_GOODHEL/GOODHEL
      DATA GOODHEL/THEL*.FALSE./
      END



"""  % misc.get_pkg_info()

        process_exporter = export_fks.ProcessExporterFortranFKS()

        nflows = \
            process_exporter.write_born_fks(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me, 
                    self.myfortranmodel)  

        self.assertFileContains('test', goal) 


    def test_write_born_hel(self):
        """tests if the born_hel.f file containing the born matrix element
        is correctly written
        """
        goal = \
"""      SUBROUTINE SBORN_HEL(P1,ANS)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     AND HELICITIES
C     FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C     
C     Process: g g > t t~ QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  16 )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*8)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS = 3)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1),ANS
C     
C     LOCAL VARIABLES 
C     
      INTEGER IHEL,IDEN,J
      REAL*8 BORN_HEL
      INTEGER IDEN_VALUES(8)
      DATA IDEN_VALUES /256, 256, 256, 256, 256, 256, 256, 256/
C     
C     GLOBAL VARIABLES
C     
      LOGICAL GOODHEL(NCOMB,8)
      COMMON /C_GOODHEL/ GOODHEL
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      DOUBLE PRECISION WGT_HEL(MAX_BHEL)
      COMMON/C_BORN_HEL/WGT_HEL

C     ----------
C     BEGIN CODE
C     ----------
      IDEN=IDEN_VALUES(NFKSPROCESS)
      IF (CALCULATEDBORN) THEN
        DO J=1,NEXTERNAL-1
          IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $     ,J)) THEN
            CALCULATEDBORN=.FALSE.
            WRITE (*,*) 'momenta not the same in Born_hel'
            STOP
          ENDIF
        ENDDO
      ELSE
        WRITE(*,*) 'Error in born_hel: should be called only wit'
     $   //'h calculatedborn = true'
        STOP
      ENDIF
      ANS = 0D0
      DO IHEL=1,NCOMB
        WGT_HEL(IHEL)=0D0
        IF (GOODHEL(IHEL,NFKSPROCESS)) THEN
          WGT_HEL(IHEL)=BORN_HEL(P1,IHEL)/DBLE(IDEN)
          ANS=ANS+WGT_HEL(IHEL)
        ENDIF
      ENDDO
      END


      REAL*8 FUNCTION BORN_HEL(P,HELL)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

C     Process: g g > t t~ QED=0 QCD=2 [ QCD ]
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER     NGRAPHS
      PARAMETER ( NGRAPHS = 3 )
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=2)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1 = (0D0,1D0))
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER HELL
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR)
C     
C     GLOBAL VARIABLES
C     
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
C     
C     COLOR DATA
C     
      DATA DENOM(1)/3/
      DATA (CF(I,  1),I=  1,  2) /   16,   -2/
C     1 T(1,2,3,4)
      DATA DENOM(2)/3/
      DATA (CF(I,  2),I=  1,  2) /   -2,   16/
C     1 T(2,1,3,4)
C     ----------
C     BEGIN CODE
C     ----------
      IF (.NOT. CALCULATEDBORN) THEN
        WRITE(*,*) 'Error in born_hel.f: this should be called onl'
     $   //'y with calculatedborn = true'
        STOP
      ELSEIF (CALCULATEDBORN) THEN
        DO I=1,NGRAPHS
          AMP(I)=SAVEAMP(I,HELL)
        ENDDO
      ENDIF
      JAMP(1)=+IMAG1*AMP(1)-AMP(2)
      JAMP(2)=-IMAG1*AMP(1)-AMP(3)
      BORN_HEL = 0.D0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        BORN_HEL =BORN_HEL+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
      ENDDO
      END



"""  % misc.get_pkg_info()

        process_exporter = export_fks.ProcessExporterFortranFKS()

        nflows = \
            process_exporter.write_born_hel(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me, 
                    self.myfortranmodel)  

        self.assertFileContains('test', goal) 


    def test_write_b_sf_fks(self):
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
C     Process: g g > t t~ QED=0 QCD=2 [ QCD ]
C     spectators: 1 2 

C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INCLUDE 'nexternal.inc'
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  16 )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*8)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   3)
C     
C     ARGUMENTS 
C     
      REAL*8 P1(0:3,NEXTERNAL-1),ANS
C     
C     LOCAL VARIABLES 
C     
      INTEGER IHEL,IDEN,J
      REAL*8 B_SF_001
      INTEGER IDEN_VALUES(8)
      DATA IDEN_VALUES /256, 256, 256, 256, 256, 256, 256, 256/
C     
C     GLOBAL VARIABLES
C     
      LOGICAL GOODHEL(NCOMB,8)
      COMMON /C_GOODHEL/ GOODHEL
      DOUBLE PRECISION SAVEMOM(NEXTERNAL-1,2)
      COMMON/TO_SAVEMOM/SAVEMOM
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
C     ----------
C     BEGIN CODE
C     ----------
      IDEN=IDEN_VALUES(NFKSPROCESS)
      IF (CALCULATEDBORN) THEN
        DO J=1,NEXTERNAL-1
          IF (SAVEMOM(J,1).NE.P1(0,J) .OR. SAVEMOM(J,2).NE.P1(3
     $     ,J)) THEN
            CALCULATEDBORN=.FALSE.
            WRITE(*,*) 'Error in sb_sf: momenta not the same in th'
     $       //'e born'
            STOP
          ENDIF
        ENDDO
      ELSE
        WRITE(*,*) 'Error in sb_sf: color_linked borns should b'
     $   //'e called only with calculatedborn = true'
        STOP
      ENDIF
      ANS = 0D0
      DO IHEL=1,NCOMB
        IF (GOODHEL(IHEL,NFKSPROCESS)) THEN
          ANS=ANS+B_SF_001(P1,IHEL)
        ENDIF
      ENDDO
      ANS=ANS/DBLE(IDEN)
      END


      REAL*8 FUNCTION B_SF_001(P,HELL)
C     
C     Generated by MadGraph5_aMC@NLO v. %(version)s, %(date)s
C     By the MadGraph5_aMC@NLO Development Team
C     Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
C     RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C     FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

C     Process: g g > t t~ QED=0 QCD=2 [ QCD ]
C     spectators: 1 2 

C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER     NGRAPHS
      PARAMETER ( NGRAPHS = 3 )
      INTEGER NCOLOR1, NCOLOR2
      PARAMETER (NCOLOR1=2, NCOLOR2=3)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1 = (0D0,1D0))
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
C     
C     ARGUMENTS 
C     
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER HELL
C     
C     LOCAL VARIABLES 
C     
      INTEGER I,J
      REAL*8 DENOM(NCOLOR1), CF(NCOLOR2,NCOLOR1)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP1(NCOLOR1), JAMP2(NCOLOR2)
C     
C     GLOBAL VARIABLES
C     
      DOUBLE COMPLEX SAVEAMP(NGRAPHS,MAX_BHEL)
      COMMON/TO_SAVEAMP/SAVEAMP
      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN
C     
C     COLOR DATA
C     
      DATA DENOM(1)/3/
      DATA (CF(I,  1),I=  1,  3) /   16,   -2,    6/
      DATA DENOM(2)/3/
      DATA (CF(I,  2),I=  1,  3) /   -2,   16,    6/
C     ----------
C     BEGIN CODE
C     ----------
      IF (.NOT. CALCULATEDBORN) THEN
        WRITE(*,*) 'Error in b_sf: color_linked borns should be calle'
     $   //'d only with calculatedborn = true'
        STOP
      ELSEIF (CALCULATEDBORN) THEN
        DO I=1,NGRAPHS
          AMP(I)=SAVEAMP(I,HELL)
        ENDDO
      ENDIF
      JAMP1(1)=+IMAG1*AMP(1)-AMP(2)
      JAMP1(2)=-IMAG1*AMP(1)-AMP(3)
      JAMP2(1)=+1D0/2D0*(-3D0*IMAG1*AMP(1)+3D0*AMP(2))
      JAMP2(2)=+1D0/2D0*(+3D0*IMAG1*AMP(1)+3D0*AMP(3))
      JAMP2(3)=+1D0/2D0*(+AMP(2)+AMP(3))
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


    def test_write_born_nhel_file(self):
        """tests if the born_nhel.inc file is correctly written"""
        goal = \
"""      INTEGER    MAX_BHEL, MAX_BCOL
      PARAMETER (MAX_BHEL=16)
      PARAMETER(MAX_BCOL=2)
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
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.born_matrix_element,
                    nflows,
                    self.myfortranmodel,
                    ncolor)  

        self.assertFileContains('test', goal) 


    def test_write_nfksconfigs_file(self):
        """tests if the nFKSconfigs.inc file is correctly written"""
        goal = \
"""      INTEGER FKS_CONFIGS
      PARAMETER (FKS_CONFIGS=8)


"""        
        process_exporter = export_fks.ProcessExporterFortranFKS()
        process_exporter.write_nfksconfigs_file(\
            writers.FortranWriter(self.give_pos('test')),
            self.myfks_me,
            self.myfortranmodel)
        self.assertFileContains('test', goal)


    def test_write_configs_file_born(self):
        """Tests if the configs.inc file is corretly written 
        for the born matrix element.
        """
        goal = \
"""C     Diagram 1, Amplitude 1
      DATA MAPCONFIG(   1)/   1/
      DATA (IFOREST(I, -1,   1),I=1,2)/  4,  3/
      DATA SPROP(  -1,   1)/      21/
C     Diagram 2, Amplitude 2
      DATA MAPCONFIG(   2)/   2/
      DATA (IFOREST(I, -1,   2),I=1,2)/  1,  3/
      DATA TPRID(  -1,   2)/       6/
      DATA (IFOREST(I, -2,   2),I=1,2)/ -1,  4/
C     Diagram 3, Amplitude 3
      DATA MAPCONFIG(   3)/   3/
      DATA (IFOREST(I, -1,   3),I=1,2)/  1,  4/
      DATA TPRID(  -1,   3)/       6/
      DATA (IFOREST(I, -2,   3),I=1,2)/ -1,  3/
C     Number of configs
      DATA MAPCONFIG(0)/   3/
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        
        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)

        self.assertFileContains('test', goal)    

    
    def test_write_props_file_born(self):
        """Tests if the props.inc file is corretly written 
        for the born matrix element.
        """
        goal = \
"""      PMASS( -1,   1)  = ZERO
      PWIDTH( -1,   1) = ZERO
      POW( -1,   1) = 2
      PMASS( -1,   2)  = ABS(MDL_MT)
      PWIDTH( -1,   2) = ABS(MDL_WT)
      POW( -1,   2) = 1
      PMASS( -1,   3)  = ABS(MDL_MT)
      PWIDTH( -1,   3) = ABS(MDL_WT)
      POW( -1,   3) = 1
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()
        
        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)
        
        process_exporter.write_props_file(
                    writers.FortranWriter(self.give_pos('test')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel,
                    s_and_t_channels)        

        self.assertFileContains('test', goal)    


    def test_write_coloramps_file(self):
        """Tests if the coloramps.inc file is corretly written 
        for the born process
        """
        goal = \
"""      LOGICAL ICOLAMP(2,3,1)
      DATA(ICOLAMP(I,1,1),I=1,2)/.TRUE.,.TRUE./
      DATA(ICOLAMP(I,2,1),I=1,2)/.TRUE.,.FALSE./
      DATA(ICOLAMP(I,3,1),I=1,2)/.FALSE.,.TRUE./
"""
        process_exporter = export_fks.ProcessExporterFortranFKS()

        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)
        
        process_exporter.write_coloramps_file(
                    writers.FortranWriter(self.give_pos('test')),
                    mapconfigs,
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)        

        self.assertFileContains('test', goal)    


    def test_write_decayBW_file(self):
        """Tests if the decayBW.inc file is correctly written 
        for the born process.
        """
        goal = \
"""      DATA GFORCEBW(-1,1)/.FALSE./
"""

        process_exporter = export_fks.ProcessExporterFortranFKS()

        nconfigs, mapconfigs, s_and_t_channels = \
            process_exporter.write_configs_file(
                    writers.FortranWriter(self.give_pos('test1')),
                    self.myfks_me.born_matrix_element,
                    self.myfortranmodel)

        process_exporter.write_decayBW_file(
                    writers.FortranWriter(self.give_pos('test')),
                    s_and_t_channels)        

        self.assertFileContains('test', goal)    




    def test_get_fks_j_from_i_lines(self):
        """Test that the lines corresponding to the fks_j_from_i array, to be 
        written in fks.inc. 
        """
        lines = ['DATA (FKS_J_FROM_I_D(2, 5, JPOS), JPOS = 0, 1)  / 1, 1 /','']

        process_exporter = export_fks.ProcessExporterFortranFKS()
        self.assertEqual(lines, process_exporter.get_fks_j_from_i_lines(self.myfks_me.real_processes[1], 2))


    def test_get_color_data_lines_from_color_matrix(self):
        """tests if the color data lines are correctly extracted from a given
        color matrix. 
        The first color link is used.
        """
        
        goal = ["DATA DENOM(1)/3/",
                "DATA (CF(I,  1),I=  1,  3) /   16,   -2,    6/",
                "DATA DENOM(2)/3/",
                "DATA (CF(I,  2),I=  1,  3) /   -2,   16,    6/"
                ]
        process_exporter = export_fks.ProcessExporterFortranFKS()

        lines = process_exporter.get_color_data_lines_from_color_matrix(
                    self.myfks_me.color_links[0]['link_matrix'])
        
        for line, goalline in zip(lines, goal):
            self.assertEqual(line.upper(), goalline)        


    def test_den_factor_lines(self):
        """Tests if the den_factor lines for a given matrix element are correctly 
        returned.
        """
        
        goal = \
            ["INTEGER IDEN_VALUES(8)",
             "DATA IDEN_VALUES /256, 256, 256, 256, 256, 256, 256, 256/"]
        process_exporter = export_fks.ProcessExporterFortranFKS()

        self.assertEqual(goal,
                process_exporter.get_den_factor_lines(
                        self.myfks_me))


    def test_write_ij_lines(self):
        """Tests if the ij lines for a given matrix element are correctly 
        returned.
        """
        
        goal = \
            ["INTEGER IJ_VALUES(8)",
             "DATA IJ_VALUES /1, 2, 3, 4, 1, 1, 2, 2/"]
        process_exporter = export_fks.ProcessExporterFortranFKS()

        self.assertEqual(goal,
                process_exporter.get_ij_lines(
                        self.myfks_me))


        
