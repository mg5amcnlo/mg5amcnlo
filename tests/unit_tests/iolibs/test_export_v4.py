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
import unittest
import copy

import madgraph.iolibs.export_v4 as export_v4
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import tests.unit_tests.core.test_helas_objects as test_helas_objects

#===============================================================================
# IOImportV4Test
#===============================================================================
class IOExportV4Test(unittest.TestCase):
    """Test class for the export v4 module"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_export_matrix_element_v4_standalone(self):
        """Test the result of exporting a matrix element to file"""

        fsock = StringIO.StringIO()

        line = " call aaaaaa(bbb, ccc, ddd, eee, fff, ggg, hhhhhhhhhh + asdasdasdasd, wspedfteispd)"

        #self.assertEqual(import_v4.read_particles_v4(fsock), goal_part_list)

#===============================================================================
# IOImportV4Test
#===============================================================================
class FortranWriterTest(unittest.TestCase):
    """Test class for the Fortran writer object"""

    def test_write_fortran_line(self):
        """Test writing a fortran line"""
        
        fsock = StringIO.StringIO()
        
        lines = []
        lines.append(" call aaaaaa(bbb, ccc, ddd, eee, fff, ggg, hhhhhhhhhhhhhh+asdasd, wspedfteispd)")

        lines.append("  IF (Test) then")
        lines.append(" if(mutt) call hej")
        lines.append(" else if(test) then")
        lines.append("c      Test")
        lines.append(" Call hej")
        lines.append("# Test")
        lines.append("else")
        lines.append("bah=2")
        lines.append(" endif")
        lines.append("test")

        goal_string = """      CALL AAAAAA(BBB, CCC, DDD, EEE, FFF, GGG, HHHHHHHHHHHHHH
     $ +ASDASD, WSPEDFTEISPD)
      IF (TEST) THEN
        IF(MUTT) CALL HEJ
      ELSE IF(TEST) THEN
C       Test
        CALL HEJ
C       Test
      ELSE
        BAH=2
      ENDIF
      TEST\n"""

        writer = export_v4.FortranWriter()
        for line in lines:
            writer.write_fortran_line(fsock, line)

        self.assertEqual(fsock.getvalue(),
                         goal_string)

#===============================================================================
# HelasFortranModelTest
#===============================================================================
class HelasFortranModelTest(test_helas_objects.HelasModelTest):
    """Test class for the HelasFortranModel object"""

    def test_generate_wavefunctions_and_amplitudes(self):
        """Test automatic generation of wavefunction and amplitude calls"""

        goal = [ \
            'CALL IXXXXX(P(0,1),me,NHEL(1),1*IC(1),W(1,1))',
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
                                           'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-11,
                                           'number': 2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'number': 3,
                                         'state':'initial'}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 7,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = export_v4.HelasFortranModel()

        goal_counter = 0

        for wf in wfs:
            self.assertEqual(fortran_model.get_wavefunction_call(wf),
                             goal[goal_counter])
            goal_counter = goal_counter + 1

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers',mothers)
            if not wf.get('self_antipart'):
                wf.set('pdg_code',-wf.get('pdg_code'))
            self.assertEqual(fortran_model.get_wavefunction_call(wf),
                             goal[goal_counter])
            if not wf.get('self_antipart'):
                wf.set('pdg_code',-wf.get('pdg_code'))
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
                                           'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'number': 2,
                                           'state':'initial'}))
        myleglist.append(base_objects.Leg({'id': 8000002,
                                           'number': 3,
                                           'state':'initial'}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 5,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = export_v4.HelasFortranModel()

        for wf in wfs:
            self.assertEqual(fortran_model.get_wavefunction_call(wf),
                             goal[goal_counter])
            goal_counter = goal_counter + 1

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers',mothers)
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
            'CALL WWWWXX(W(1,1),W(1,2),W(1,3),W(1,4),MGVX6,AMP(1))',
            'CALL JW3WXX(W(1,2),W(1,3),W(1,4),MGVX8,wmas,wwid,W(1,1))',
            'CALL JW3WXX(W(1,1),W(1,3),W(1,4),MGVX8,wmas,wwid,W(1,2))',
            'CALL JW3WXX(W(1,1),W(1,2),W(1,4),MGVX8,zmas,zwid,W(1,3))',
            'CALL JW3WXX(W(1,1),W(1,2),W(1,3),MGVX8,zmas,zwid,W(1,4))',
            'CALL W3W3XX(W(1,1),W(1,2),W(1,3),W(1,4),MGVX8,AMP(1))']

        goal_counter = 0

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                           'number': 1,
                                           'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-24,
                                           'number': 2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id': 24,
                                           'number': 3,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id': -24,
                                           'number': 4,
                                         'state':'initial'}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 8,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = export_v4.HelasFortranModel()

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers',mothers)
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
                                           'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-24,
                                           'number': 2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id': 23,
                                           'number': 3,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id': 23,
                                           'number': 4,
                                         'state':'initial'}))

        wfs = helas_objects.HelasWavefunctionList(\
            [ helas_objects.HelasWavefunction(leg, 9,
                                              self.mybasemodel) \
              for leg in myleglist ])

        fortran_model = export_v4.HelasFortranModel()

        for wf in wfs:
            mothers = copy.copy(wfs)
            mothers.remove(wf)
            wf.set('mothers',mothers)
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
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)
        
        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),me,NHEL(1),1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),1*IC(4),W(1,4))
CALL FVIXXX(W(1,1),W(1,2),MGVX12,me,zero,W(1,5))
CALL IOVXXX(W(1,5),W(1,4),W(1,3),MGVX12,AMP(1))
CALL FVIXXX(W(1,1),W(1,3),MGVX12,me,zero,W(1,6))
CALL IOVXXX(W(1,6),W(1,4),W(1,2),MGVX12,AMP(2))""")

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
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(\
            myamplitude,
            0)
        
        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),mu,NHEL(1),1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),mu,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL FVIXXX(W(1,1),W(1,3),GG,mu,zero,W(1,6))
CALL JIOXXX(W(1,5),W(1,4),MGVX12,zero,zero,W(1,7))
CALL IOVXXX(W(1,6),W(1,2),W(1,7),MGVX15,AMP(1))
CALL IXXXXX(P(0,1),mu,NHEL(1),1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),mu,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),1*IC(3),W(1,3))
CALL OXXXXX(P(0,4),me,NHEL(4),1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL FVOXXX(W(1,2),W(1,3),GG,mu,zero,W(1,6))
CALL JIOXXX(W(1,5),W(1,4),MGVX12,zero,zero,W(1,7))
CALL IOVXXX(W(1,1),W(1,6),W(1,7),MGVX15,AMP(2))""")

    def test_generate_helas_diagrams_uux_ggg(self):
        """Test calls for u u~ > g g g"""
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude)
        
        # The expression below should be correct, if the color factors
        # for gluon pairs are correctly ordered.

        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                        get_matrix_element_calls(matrix_element)),
                    """CALL IXXXXX(P(0,1),mu,NHEL(1),1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),mu,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),zero,NHEL(3),1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),1*IC(5),W(1,5))
CALL JIOXXX(W(1,1),W(1,2),GG,zero,zero,W(1,6))
CALL UVVAXX(W(1,3),W(1,4),MGVX2,zero,zero,zero,W(1,7))
CALL VVTAXX(W(1,6),W(1,5),W(1,7),MGVX2,zero,AMP(1))
CALL JVVXXX(W(1,3),W(1,4),MGVX1,zero,zero,W(1,8))
CALL VVVXXX(W(1,6),W(1,8),W(1,5),MGVX1,AMP(2))
CALL UVVAXX(W(1,3),W(1,5),MGVX2,zero,zero,zero,W(1,9))
CALL VVTAXX(W(1,6),W(1,4),W(1,9),MGVX2,zero,AMP(3))
CALL JVVXXX(W(1,3),W(1,5),MGVX1,zero,zero,W(1,10))
CALL VVVXXX(W(1,6),W(1,10),W(1,4),MGVX1,AMP(4))
CALL UVVAXX(W(1,4),W(1,5),MGVX2,zero,zero,zero,W(1,11))
CALL VVTAXX(W(1,6),W(1,3),W(1,11),MGVX2,zero,AMP(5))
CALL JVVXXX(W(1,4),W(1,5),MGVX1,zero,zero,W(1,12))
CALL VVVXXX(W(1,6),W(1,3),W(1,12),MGVX1,AMP(6))
CALL FVIXXX(W(1,1),W(1,3),GG,mu,zero,W(1,13))
CALL FVOXXX(W(1,2),W(1,4),GG,mu,zero,W(1,14))
CALL IOVXXX(W(1,13),W(1,14),W(1,5),GG,AMP(7))
CALL FVOXXX(W(1,2),W(1,5),GG,mu,zero,W(1,15))
CALL IOVXXX(W(1,13),W(1,15),W(1,4),GG,AMP(8))
CALL IOVXXX(W(1,13),W(1,2),W(1,12),GG,AMP(9))
CALL FVIXXX(W(1,1),W(1,4),GG,mu,zero,W(1,16))
CALL FVOXXX(W(1,2),W(1,3),GG,mu,zero,W(1,17))
CALL IOVXXX(W(1,16),W(1,17),W(1,5),GG,AMP(10))
CALL IOVXXX(W(1,16),W(1,15),W(1,3),GG,AMP(11))
CALL IOVXXX(W(1,16),W(1,2),W(1,10),GG,AMP(12))
CALL FVIXXX(W(1,1),W(1,5),GG,mu,zero,W(1,18))
CALL IOVXXX(W(1,18),W(1,17),W(1,4),GG,AMP(13))
CALL IOVXXX(W(1,18),W(1,14),W(1,3),GG,AMP(14))
CALL IOVXXX(W(1,18),W(1,2),W(1,8),GG,AMP(15))
CALL IOVXXX(W(1,1),W(1,17),W(1,12),GG,AMP(16))
CALL IOVXXX(W(1,1),W(1,14),W(1,10),GG,AMP(17))
CALL IOVXXX(W(1,1),W(1,15),W(1,8),GG,AMP(18))""")

    def test_generate_helas_diagrams_uu_susu(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes u u > su su with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)
        
        # I have checked that the resulting Helas calls
        # below give identical result as MG4, apart from the sign! (AMP 1,2)
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL OXXXXX(P(0,1),mu,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),mu,NHEL(2),1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),1*IC(4),W(1,4))
CALL FSOCXX(W(1,1),W(1,3),MGVX575,Mneu1,Wneu1,W(1,5))
CALL IOSXXX(W(1,2),W(1,5),W(1,4),MGVX575,AMP(1))
CALL FSOCXX(W(1,1),W(1,4),MGVX575,Mneu1,Wneu1,W(1,6))
CALL IOSXXX(W(1,2),W(1,6),W(1,3),MGVX575,AMP(2))""")


    def test_generate_helas_diagrams_epem_elpelmepem(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes e+ e- > sl2+ sl2- e+ e-
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

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        #print myamplitude.get('process').nice_string()
        #print "\n".join(export_v4.HelasFortranModel().\
        #                get_matrix_element_calls(matrix_element))
        #print export_v4.HelasFortranModel().get_JAMP_line(matrix_element)
            


        # I have checked that the resulting Helas calls below give
        # identical result as MG4 (when fermionfactors are taken into
        # account)
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL OXXXXX(P(0,1),me,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),me,NHEL(2),1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),1*IC(4),W(1,4))
CALL IXXXXX(P(0,5),me,NHEL(5),-1*IC(5),W(1,5))
CALL OXXXXX(P(0,6),me,NHEL(6),1*IC(6),W(1,6))
CALL FSOXXX(W(1,1),W(1,3),MGVX350,Mneu1,Wneu1,W(1,7))
CALL FSIXXX(W(1,2),W(1,4),MGVX494,Mneu1,Wneu1,W(1,8))
CALL HIOXXX(W(1,5),W(1,7),MGVX494,Msl2,Wsl2,W(1,9))
CALL IOSXXX(W(1,8),W(1,6),W(1,9),MGVX350,AMP(1))
CALL IXXXXX(P(0,1),me,NHEL(1),1*IC(1),W(1,10))
CALL FSICXX(W(1,10),W(1,3),MGVX350,Mneu1,Wneu1,W(1,11))
CALL HIOXXX(W(1,11),W(1,6),MGVX350,Msl2,Wsl2,W(1,12))
CALL OXXXXX(P(0,2),me,NHEL(2),-1*IC(2),W(1,13))
CALL FSOCXX(W(1,13),W(1,4),MGVX494,Mneu1,Wneu1,W(1,14))
CALL IOSXXX(W(1,5),W(1,14),W(1,12),MGVX494,AMP(2))
CALL FSIXXX(W(1,5),W(1,4),MGVX494,Mneu1,Wneu1,W(1,15))
CALL HIOXXX(W(1,2),W(1,7),MGVX494,Msl2,Wsl2,W(1,16))
CALL IOSXXX(W(1,15),W(1,6),W(1,16),MGVX350,AMP(3))
CALL OXXXXX(P(0,5),me,NHEL(5),1*IC(5),W(1,17))
CALL FSOCXX(W(1,17),W(1,4),MGVX494,Mneu1,Wneu1,W(1,18))
CALL IOSXXX(W(1,2),W(1,18),W(1,12),MGVX494,AMP(4))
CALL FSOXXX(W(1,6),W(1,3),MGVX350,Mneu1,Wneu1,W(1,19))
CALL HIOXXX(W(1,8),W(1,1),MGVX350,Msl2,Wsl2,W(1,20))
CALL IOSXXX(W(1,5),W(1,19),W(1,20),MGVX494,AMP(5))
CALL IXXXXX(P(0,6),me,NHEL(6),-1*IC(6),W(1,21))
CALL FSICXX(W(1,21),W(1,3),MGVX350,Mneu1,Wneu1,W(1,22))
CALL HIOXXX(W(1,22),W(1,1),MGVX350,Msl2,Wsl2,W(1,23))
CALL IOSXXX(W(1,5),W(1,14),W(1,23),MGVX494,AMP(6))
CALL IOSXXX(W(1,2),W(1,18),W(1,23),MGVX494,AMP(7))
CALL HIOXXX(W(1,15),W(1,1),MGVX350,Msl2,Wsl2,W(1,24))
CALL IOSXXX(W(1,2),W(1,19),W(1,24),MGVX494,AMP(8))""")
        

    def test_generate_helas_diagrams_uu_susug(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes u u > su su with t-channel n1
        """

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':2,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':1000002,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':21,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)
        
        # I have checked that the resulting Helas calls
        # below give identical result as MG4, apart from sign! (AMP 1,2,5,6)
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL OXXXXX(P(0,1),mu,NHEL(1),-1*IC(1),W(1,1))
CALL IXXXXX(P(0,2),mu,NHEL(2),1*IC(2),W(1,2))
CALL SXXXXX(P(0,3),1*IC(3),W(1,3))
CALL SXXXXX(P(0,4),1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),1*IC(5),W(1,5))
CALL FSOCXX(W(1,1),W(1,3),MGVX575,Mneu1,Wneu1,W(1,6))
CALL FVIXXX(W(1,2),W(1,5),GG,mu,zero,W(1,7))
CALL IOSXXX(W(1,7),W(1,6),W(1,4),MGVX575,AMP(1))
CALL HVSXXX(W(1,5),W(1,4),MGVX74,Musq2,Wusq2,W(1,8))
CALL IOSXXX(W(1,2),W(1,6),W(1,8),MGVX575,AMP(2))
CALL FSOCXX(W(1,1),W(1,4),MGVX575,Mneu1,Wneu1,W(1,9))
CALL IOSXXX(W(1,7),W(1,9),W(1,3),MGVX575,AMP(3))
CALL HVSXXX(W(1,5),W(1,3),MGVX74,Musq2,Wusq2,W(1,10))
CALL IOSXXX(W(1,2),W(1,9),W(1,10),MGVX575,AMP(4))
CALL FVOCXX(W(1,1),W(1,5),GG,mu,zero,W(1,11))
CALL FSIXXX(W(1,2),W(1,3),MGVX575,Mneu1,Wneu1,W(1,12))
CALL IOSCXX(W(1,12),W(1,11),W(1,4),MGVX575,AMP(5))
CALL FSIXXX(W(1,2),W(1,4),MGVX575,Mneu1,Wneu1,W(1,13))
CALL IOSCXX(W(1,13),W(1,11),W(1,3),MGVX575,AMP(6))
CALL IOSCXX(W(1,12),W(1,1),W(1,8),MGVX575,AMP(7))
CALL IOSCXX(W(1,13),W(1,1),W(1,10),MGVX575,AMP(8))""")
                         
    def test_generate_helas_diagrams_enu_enu(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes e- nubar > e- nubar
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
        eminus = mypartlist[len(mypartlist)-1]
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
        nu = mypartlist[len(mypartlist)-1]
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
        Wplus = mypartlist[len(mypartlist)-1]
        Wminus = copy.copy(Wplus)
        Wminus.set('is_part', False)

        # Coupling of W- e+ nu_e

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [eplus, \
                                             nu, \
                                             Wminus]),
            'color': ['C1'],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))
        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [nubar, \
                                             eminus, \
                                             Wplus]),
            'color': ['C1'],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX27'},
            'orders':{'QED':1}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-12,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':11,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-12,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls
        # below give identical result as MG4
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL IXXXXX(P(0,1),me,NHEL(1),1*IC(1),W(1,1))
CALL OXXXXX(P(0,2),zero,NHEL(2),-1*IC(2),W(1,2))
CALL OXXXXX(P(0,3),me,NHEL(3),1*IC(3),W(1,3))
CALL IXXXXX(P(0,4),zero,NHEL(4),-1*IC(4),W(1,4))
CALL JIOXXX(W(1,1),W(1,2),MGVX27,MW,WW,W(1,5))
CALL IOVXXX(W(1,4),W(1,3),W(1,5),MGVX27,AMP(1))""")

    def test_generate_helas_diagrams_WWWW(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes W+ W- > W+ W-
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
        Wplus = mypartlist[len(mypartlist)-1]
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
        a = mypartlist[len(mypartlist)-1]

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
        Z = mypartlist[len(mypartlist)-1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': ['C1'],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Z]),
            'color': ['C1'],
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
            'color': ['C1'],
            'lorentz':['WWWWN',''],
            'couplings':{(0, 0):'MGVX6',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.  Note that this looks like it uses
        # incoming bosons instead of outgoing though
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MW,NHEL(3),1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),MW,NHEL(4),1*IC(4),W(1,4))
CALL WWWWNX(W(1,2),W(1,1),W(1,3),W(1,4),MGVX6,DUM0,AMP(1))
CALL JVVXXX(W(1,2),W(1,1),MGVX3,zero,zero,W(1,5))
CALL VVVXXX(W(1,3),W(1,4),W(1,5),MGVX3,AMP(2))
CALL JVVXXX(W(1,2),W(1,1),MGVX5,MZ,WZ,W(1,6))
CALL VVVXXX(W(1,3),W(1,4),W(1,6),MGVX5,AMP(3))
CALL JVVXXX(W(1,3),W(1,1),MGVX3,zero,zero,W(1,7))
CALL VVVXXX(W(1,2),W(1,4),W(1,7),MGVX3,AMP(4))
CALL JVVXXX(W(1,3),W(1,1),MGVX5,MZ,WZ,W(1,8))
CALL VVVXXX(W(1,2),W(1,4),W(1,8),MGVX5,AMP(5))""")

    def test_generate_helas_diagrams_WWZA(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes W+ W- > W+ W-
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
        Wplus = mypartlist[len(mypartlist)-1]
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
        a = mypartlist[len(mypartlist)-1]

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
        Z = mypartlist[len(mypartlist)-1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': ['C1'],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Wplus, \
                                             Z]),
            'color': ['C1'],
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
            'color': ['C1'],
            'lorentz':['WWWWN',''],
            'couplings':{(0, 0):'MGVX6',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             a]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX4',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 5,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             Z]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX7',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Z, \
                                             Wminus,
                                             Z]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX8',(0,1):'DUM0'},
            'orders':{'QED':2}}))


        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':mybasemodel})

        myamplitude = diagram_generation.Amplitude({'process': myproc})

        matrix_element = helas_objects.HelasMatrixElement(myamplitude, 1)

        # I have checked that the resulting Helas calls below give
        # identical result as MG4.
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MZ,NHEL(3),1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),zero,NHEL(4),1*IC(4),W(1,4))
CALL W3W3NX(W(1,2),W(1,4),W(1,1),W(1,3),MGVX7,DUM0,AMP(1))
CALL JVVXXX(W(1,3),W(1,1),MGVX5,MW,WW,W(1,5))
CALL VVVXXX(W(1,2),W(1,5),W(1,4),MGVX3,AMP(2))
CALL JVVXXX(W(1,1),W(1,4),MGVX3,MW,WW,W(1,6))
CALL VVVXXX(W(1,6),W(1,2),W(1,3),MGVX5,AMP(3))""")

    def test_sorted_mothers(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes W+ W- > W+ W- a
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
        Wplus = mypartlist[len(mypartlist)-1]
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
        a = mypartlist[len(mypartlist)-1]

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
        Z = mypartlist[len(mypartlist)-1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 3,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             Wplus,
                                             Wminus]),
            'color': ['C1'],
            'lorentz':['WWWWN',''],
            'couplings':{(0, 0):'MGVX6',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             a]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX4',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 5,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             a, \
                                             Wplus,
                                             Z]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX7',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Z, \
                                             Wplus,
                                             Z]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX8',(0,1):'DUM0'},
            'orders':{'QED':2}}))


        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                           'state':'initial',
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':23,
                                         'state':'final',
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':'initial',
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final',
                                           'number': 5}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final',
                                           'number': 4}))

        mymothers = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(leg, 0, mybasemodel) for leg in myleglist[:4]])

        amplitude = helas_objects.HelasAmplitude()
        amplitude.set('interaction_id', 5, mybasemodel)
        amplitude.set('mothers',mymothers)
        self.assertEqual(export_v4.HelasFortranModel.sorted_mothers(amplitude),
                         [mymothers[2], mymothers[3], mymothers[0], mymothers[1]])
        mymothers = helas_objects.HelasWavefunctionList(\
            [helas_objects.HelasWavefunction(leg, 0, mybasemodel) for leg in myleglist[2:]])

        wavefunction = helas_objects.HelasWavefunction(myleglist[2],
                                                       4, mybasemodel)
        wavefunction.set('mothers', mymothers)
        self.assertEqual(export_v4.HelasFortranModel.\
                         sorted_mothers(wavefunction),
                         [mymothers[2], mymothers[0], mymothers[1]])
        

    def test_generate_helas_diagrams_WWWWA(self):
        """Testing the helas diagram generation based on Diagrams
        using the processes W+ W- > W+ W- a
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
        Wplus = mypartlist[len(mypartlist)-1]
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
        a = mypartlist[len(mypartlist)-1]

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
        Z = mypartlist[len(mypartlist)-1]


        # WWZ and WWa couplings

        myinterlist.append(base_objects.Interaction({
            'id': 1,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Wminus, \
                                             a]),
            'color': ['C1'],
            'lorentz':[''],
            'couplings':{(0, 0):'MGVX3'},
            'orders':{'QED':1}}))

        myinterlist.append(base_objects.Interaction({
            'id': 2,
            'particles': base_objects.ParticleList(\
                                            [Wminus, \
                                             Wplus, \
                                             Z]),
            'color': ['C1'],
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
            'color': ['C1'],
            'lorentz':['WWWWN',''],
            'couplings':{(0, 0):'MGVX6',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 4,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             a]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX4',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 5,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             a, \
                                             Wminus,
                                             Z]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX7',(0,1):'DUM0'},
            'orders':{'QED':2}}))

        myinterlist.append(base_objects.Interaction({
            'id': 6,
            'particles': base_objects.ParticleList(\
                                            [Wplus, \
                                             Z, \
                                             Wminus,
                                             Z]),
            'color': ['C1'],
            'lorentz':['WWVVN',''],
            'couplings':{(0, 0):'MGVX8',(0,1):'DUM0'},
            'orders':{'QED':2}}))


        mybasemodel = base_objects.Model()
        mybasemodel.set('particles', mypartlist)
        mybasemodel.set('interactions', myinterlist)

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':24,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':'initial'}))
        myleglist.append(base_objects.Leg({'id':24,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':-24,
                                         'state':'final'}))
        myleglist.append(base_objects.Leg({'id':22,
                                         'state':'final'}))

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
        self.assertEqual("\n".join(export_v4.HelasFortranModel().\
                                   get_matrix_element_calls(matrix_element)),
                         """CALL VXXXXX(P(0,1),MW,NHEL(1),-1*IC(1),W(1,1))
CALL VXXXXX(P(0,2),MW,NHEL(2),-1*IC(2),W(1,2))
CALL VXXXXX(P(0,3),MW,NHEL(3),1*IC(3),W(1,3))
CALL VXXXXX(P(0,4),MW,NHEL(4),1*IC(4),W(1,4))
CALL VXXXXX(P(0,5),zero,NHEL(5),1*IC(5),W(1,5))
CALL JVVXXX(W(1,2),W(1,1),MGVX3,zero,zero,W(1,6))
CALL JVVXXX(W(1,5),W(1,3),MGVX3,MW,WW,W(1,7))
CALL VVVXXX(W(1,7),W(1,4),W(1,6),MGVX3,AMP(1))
CALL JVVXXX(W(1,1),W(1,2),MGVX5,MZ,WZ,W(1,8))
CALL VVVXXX(W(1,4),W(1,7),W(1,8),MGVX5,AMP(2))
CALL JVVXXX(W(1,4),W(1,5),MGVX3,MW,WW,W(1,9))
CALL VVVXXX(W(1,3),W(1,9),W(1,6),MGVX3,AMP(3))
CALL VVVXXX(W(1,9),W(1,3),W(1,8),MGVX5,AMP(4))
CALL W3W3NX(W(1,3),W(1,6),W(1,4),W(1,5),MGVX4,DUM0,AMP(5))
CALL W3W3NX(W(1,3),W(1,5),W(1,4),W(1,8),MGVX7,DUM0,AMP(6))
CALL JVVXXX(W(1,3),W(1,1),MGVX3,zero,zero,W(1,10))
CALL JVVXXX(W(1,5),W(1,2),MGVX3,MW,WW,W(1,11))
CALL VVVXXX(W(1,11),W(1,4),W(1,10),MGVX3,AMP(7))
CALL JVVXXX(W(1,1),W(1,3),MGVX5,MZ,WZ,W(1,12))
CALL VVVXXX(W(1,4),W(1,11),W(1,12),MGVX5,AMP(8))
CALL VVVXXX(W(1,2),W(1,9),W(1,10),MGVX3,AMP(9))
CALL VVVXXX(W(1,9),W(1,2),W(1,12),MGVX5,AMP(10))
CALL W3W3NX(W(1,2),W(1,10),W(1,4),W(1,5),MGVX4,DUM0,AMP(11))
CALL W3W3NX(W(1,2),W(1,5),W(1,4),W(1,12),MGVX7,DUM0,AMP(12))
CALL JVVXXX(W(1,1),W(1,5),MGVX3,MW,WW,W(1,13))
CALL JVVXXX(W(1,2),W(1,4),MGVX3,zero,zero,W(1,14))
CALL VVVXXX(W(1,3),W(1,13),W(1,14),MGVX3,AMP(13))
CALL JVVXXX(W(1,4),W(1,2),MGVX5,MZ,WZ,W(1,15))
CALL VVVXXX(W(1,13),W(1,3),W(1,15),MGVX5,AMP(14))
CALL JVVXXX(W(1,3),W(1,4),MGVX3,zero,zero,W(1,16))
CALL VVVXXX(W(1,2),W(1,13),W(1,16),MGVX3,AMP(15))
CALL JVVXXX(W(1,4),W(1,3),MGVX5,MZ,WZ,W(1,17))
CALL VVVXXX(W(1,13),W(1,2),W(1,17),MGVX5,AMP(16))
CALL WWWWNX(W(1,2),W(1,13),W(1,3),W(1,4),MGVX6,DUM0,AMP(17))
CALL VVVXXX(W(1,7),W(1,1),W(1,14),MGVX3,AMP(18))
CALL VVVXXX(W(1,1),W(1,7),W(1,15),MGVX5,AMP(19))
CALL VVVXXX(W(1,11),W(1,1),W(1,16),MGVX3,AMP(20))
CALL VVVXXX(W(1,1),W(1,11),W(1,17),MGVX5,AMP(21))
CALL JWWWNX(W(1,2),W(1,1),W(1,3),MGVX6,DUM0,MW,WW,W(1,18))
CALL VVVXXX(W(1,18),W(1,4),W(1,5),MGVX3,AMP(22))
CALL JWWWNX(W(1,1),W(1,2),W(1,4),MGVX6,DUM0,MW,WW,W(1,19))
CALL VVVXXX(W(1,3),W(1,19),W(1,5),MGVX3,AMP(23))
CALL JW3WNX(W(1,1),W(1,5),W(1,2),MGVX4,DUM0,zero,zero,W(1,20))
CALL VVVXXX(W(1,3),W(1,4),W(1,20),MGVX3,AMP(24))
CALL JW3WNX(W(1,2),W(1,5),W(1,1),MGVX7,DUM0,MZ,WZ,W(1,21))
CALL VVVXXX(W(1,4),W(1,3),W(1,21),MGVX5,AMP(25))
CALL JWWWNX(W(1,1),W(1,3),W(1,4),MGVX6,DUM0,MW,WW,W(1,22))
CALL VVVXXX(W(1,2),W(1,22),W(1,5),MGVX3,AMP(26))
CALL JW3WNX(W(1,1),W(1,5),W(1,3),MGVX4,DUM0,zero,zero,W(1,23))
CALL VVVXXX(W(1,2),W(1,4),W(1,23),MGVX3,AMP(27))
CALL JW3WNX(W(1,3),W(1,5),W(1,1),MGVX7,DUM0,MZ,WZ,W(1,24))
CALL VVVXXX(W(1,4),W(1,2),W(1,24),MGVX5,AMP(28))""")
