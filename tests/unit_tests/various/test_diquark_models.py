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
"""Unit test Library for the objects in decay module."""
from __future__ import division

import math
import copy
import os
import sys
import time

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.various.process_checks as process_checks
import models.import_ufo as import_ufo
import models.model_reader as model_reader

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# TestColorSextetModel
#===============================================================================
class TestColorSextetModel(unittest.TestCase):
    """Test class for the sextet diquark implementation"""


    def setUp(self):
        m_path = import_ufo.find_ufo_path('sextet_diquarks')
        self.base_model = import_ufo.import_model(m_path)
    
    def test_uu_to_six_g(self):
        """Test the process u u > six g against literature expression"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model,
                                                          reuse = False)
        
        p, w_rambo = evaluator.get_momenta(myproc)

        amplitude = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(amplitude)

        mg5_me_value, amp2 = evaluator.evaluate_matrix_element(matrix_element,
                                                               p)

        comparison_value = uu_Dg(p, 6, evaluator.full_model)

        self.assertAlmostEqual(mg5_me_value, comparison_value, 12)

    def test_check_u_u_six_g(self):
        """Test the process u u > six g against literature expression"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        comparison_results, used_lorentz = \
                            process_checks.check_processes(myproc,
                                                           quick=True)
        self.assertTrue(comparison_results[0]['passed'])

    def disabled_test_gu_to_ux_six(self):
        """Test the process u u > six g against literature expression"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':600001,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model,
                                                          reuse = False)
        
        p, w_rambo = evaluator.get_momenta(myproc)

        amplitude = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(amplitude)

        mg5_me_value, amp2 = evaluator.evaluate_matrix_element(matrix_element,
                                                               p)

        comparison_value = gu_uxD(p, 6, self.full_model)

        self.assertAlmostEqual(mg5_me_value, comparison_value, 14)


    def test_sextet_color_flow_output(self):
        """Test the color flow output for color sextets"""

        # Test u u~ > six six~

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':-2,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':-9000006,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamp = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(myamp)
        
        # First build a color representation dictionnary
        repr_dict = {}
        for l in myleglist:
            repr_dict[l.get('number')] = \
                self.base_model.get_particle(l.get('id')).get_color()

        # Get the color flow decomposition
        col_flow = \
           matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                      2)
        self.assertEqual(len(col_flow), len(matrix_element.get('color_basis')))
        self.assertEqual(col_flow,
                         [{1: [503, 0], 2: [0, 501], 
                           3: [502, -503], 4: [-501, 502]}, 
                          {1: [502, 0], 2: [0, 501], 
                           3: [502, -503], 4: [-503, 501]}, 
                          {1: [501, 0], 2: [0, 501], 
                           3: [502, -503], 4: [-502, 503]}])

        # Test u u > six g

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamp = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(myamp)
        
        # First build a color representation dictionnary
        repr_dict = {}
        for l in myleglist:
            repr_dict[l.get('number')] = \
                self.base_model.get_particle(l.get('id')).get_color()

        # Get the color flow decomposition
        col_flow = \
           matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                      2)
        self.assertEqual(len(col_flow), len(matrix_element.get('color_basis')))
        self.assertEqual(col_flow,
                          [{1: [503, 0], 2: [502, 0], 
                            3: [501, -503], 4: [502, 501]}, 
                           {1: [502, 0], 2: [503, 0], 
                            3: [501, -503], 4: [502, 501]}, 
                           {1: [501, 0], 2: [502, 0], 
                            3: [501, -503], 4: [502, 503]}, 
                           {1: [502, 0], 2: [501, 0], 
                            3: [501, -503], 4: [502, 503]}])

        # Test u u > six > u u g

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
                                       'required_s_channels': [[9000006]]})

        myamp = diagram_generation.Amplitude(myproc)
        self.assertEqual(len(myamp.get('diagrams')), 5)
        matrix_element = helas_objects.HelasMatrixElement(myamp)
        
        # First build a color representation dictionnary
        repr_dict = {}
        for l in myleglist:
            repr_dict[l.get('number')] = \
                self.base_model.get_particle(l.get('id')).get_color()

        # Get the color flow decomposition
        col_flow = \
           matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                      2)
        self.assertEqual(len(col_flow), len(matrix_element.get('color_basis')))
        self.assertEqual(col_flow,
                          [{1: [501, 0], 2: [503, 0], 3: [501, 0], 
                            4: [502, 0], 5: [503, 502]}, 
                           {1: [503, 0], 2: [501, 0], 3: [501, 0], 
                            4: [502, 0], 5: [503, 502]}, 
                           {1: [502, 0], 2: [503, 0], 3: [501, 0], 
                            4: [502, 0], 5: [503, 501]}, {1: [503, 0], 
                            2: [502, 0], 3: [501, 0], 
                            4: [502, 0], 5: [503, 501]}])

        # Test six > u u

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':9000006,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2}))
        myleglist.append(base_objects.Leg({'id':2}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamp = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(myamp)
        
        # First build a color representation dictionnary
        repr_dict = {}
        for l in myleglist:
            repr_dict[l.get('number')] = \
                self.base_model.get_particle(l.get('id')).get_color()\
                * (-1)**(1+l.get('state'))

        # Get the color flow decomposition
        col_flow = \
           matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                      2)
        self.assertEqual(len(col_flow), len(matrix_element.get('color_basis')))
        self.assertEqual(col_flow,
                          [{1: [501, -502], 2: [0, 501], 3: [502, 0]}])

        # Test the size of the color basis in g g  > g six six 

        myleglist = base_objects.LegList()


        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':9000006,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':-9000006,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamp = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(myamp)
        self.assertEqual(13, len(matrix_element.get('color_basis')))


#===============================================================================
# TestColorTripletModel
#===============================================================================
class TestColorTripletModel(unittest.TestCase):
    """Test class for the triplet diquark implementation"""


    def setUp(self):
        m_path = import_ufo.find_ufo_path('triplet_diquarks')
        self.base_model = import_ufo.import_model(m_path)
        self.full_model = model_reader.ModelReader(self.base_model)
        self.full_model.set_parameters_and_couplings()
        # Set top quark mass to 0 to compare with literature expression
        self.full_model.get('parameter_dict')['mdl_MT'] = 0.
        self.full_model.get('parameter_dict')['mdl_WT'] = 0.

    
    def test_ut_to_antitrip_g(self):
        """Test the process u t > antitrip g against literature expression"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':6,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-9000006,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model,
                                                          reuse = False)
        evaluator.full_model = self.full_model
        
        p, w_rambo = evaluator.get_momenta(myproc)

        amplitude = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(amplitude)

        self.assertTrue(matrix_element.get('diagrams'))

        mg5_me_value, amp2 = evaluator.evaluate_matrix_element(matrix_element,
                                                               p)

        comparison_value = uu_Dg(p, 3, self.full_model)

        self.assertAlmostEqual(mg5_me_value, comparison_value, 12)

    def test_check_u_t_antitrip_g(self):
        """Test the process u u > antitrip g against literature expression"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':6,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-9000006,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        comparison_results, used_lorentz = \
                            process_checks.check_processes(myproc,
                                                           quick=True)

        self.assertTrue(comparison_results[0]['passed'])

    def test_triplet_color_flow_output(self):
        """Test the color flow output for color triplets"""

        # Test u u > trip~ g

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':6,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':-9000006,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        myamp = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(myamp)
        
        # First build a color representation dictionnary
        repr_dict = {}
        for l in myleglist:
            repr_dict[l.get('number')] = \
                self.base_model.get_particle(l.get('id')).get_color()

        # Get the color flow decomposition
        col_flow = \
           matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                      2)
        self.assertEqual(col_flow,
                         [{1: [501, 0], 2: [502, 0],
                           3: [0, 504], 4: [504, 503]},
                          {1: [501, 0], 2: [504, 0],
                           3: [0, 502], 4: [504, 503]},
                          {1: [504, 0], 2: [501, 0],
                           3: [0, 502], 4: [504, 503]}])


        # Test u u > trip~ > u u g

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':2,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':6,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':2,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':6,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model,
                                       'required_s_channels': [[-9000006]]})

        myamp = diagram_generation.Amplitude(myproc)
        self.assertEqual(len(myamp.get('diagrams')), 5)
        matrix_element = helas_objects.HelasMatrixElement(myamp)
        
        # First build a color representation dictionnary
        repr_dict = {}
        for l in myleglist:
            repr_dict[l.get('number')] = \
                self.base_model.get_particle(l.get('id')).get_color()

        # Get the color flow decomposition
        col_flow = \
           matrix_element.get('color_basis').color_flow_decomposition(repr_dict,
                                                                      2)
        self.assertEqual(col_flow,
                         [{1: [501, 0], 2: [502, 0], 3: [504, 0],
                           4: [505, 0], 5: [506, 503]},
                          {1: [501, 0], 2: [503, 0], 3: [501, 0],
                           4: [502, 0], 5: [503, 502]},
                          {1: [503, 0], 2: [501, 0], 3: [501, 0],
                           4: [502, 0], 5: [503, 502]},
                          {1: [502, 0], 2: [503, 0], 3: [501, 0],
                           4: [502, 0], 5: [503, 501]},
                          {1: [503, 0], 2: [502, 0], 3: [501, 0],
                           4: [502, 0], 5: [503, 501]}])


#===============================================================================
# Global helper functions
#===============================================================================

def uu_Dg(P, color_rep, full_model):
    """Process: u u > six g
       From 0909.2666, Eq. (B.8)
       |Mqq|^2 = 16 lambda^2 g_s^2 N_D (2\tau/(1-\tau)^2 + 1) *
                 (C_F*4/(sin\theta)^2 - C_D)
       lambda^2=|GC_24|^2, g_s^2=GC_4^2, N_D=6, N_C=3, C_F=4/3, C_D=10/3, 
       (for antitriplet diquark, N_D=3, C_D=4/3) 
       \tau=m_D^2/shat, \cos\theta=p1p4/|p1||p4|"""

    if color_rep == 6:
        N_D = 6.
        C_D = 10./3.
        lamb = abs(full_model.get('coupling_dict')['GC_24'])
        G = full_model.get('coupling_dict')['GC_4']
        M_D = full_model.get('parameter_dict')['mdl_MSIX']
    else:
        N_D = 3.
        C_D = 4./3.
        lamb = abs(full_model.get('coupling_dict')['GC_12'])
        G = full_model.get('coupling_dict')['GC_4']
        M_D = full_model.get('parameter_dict')['mdl_MSIX']

    # Define auxiliary quantities
    shat=dot(P[0],P[0])+dot(P[1],P[1])+2*dot(P[0],P[1])
    tau=M_D**2/shat
    cos2theta=dot3(P[0],P[3])**2/(dot3(P[0],P[0])*dot3(P[3],P[3]))

    # Calculate matrix element
    ANS = 16*lamb**2*G**2*N_D*(2*tau/(1-tau)**2 + 1)
    ANS = ANS*(4./3.*4./(1-cos2theta)-C_D)

    #   Divide by color and spin factors for final state
    ANS = ANS/N_D/8./3.
      
    return ANS

def gu_uxD(P, color_rep, full_model):
    """Process: g u > u~ six
       From 0909.2666, Eq. (B.23)
       |Mqq|^2 = 8 lambda^2 g_s^2 N_D (C_F(4/(1-cos\theta)*(1/(1-\tau) - 2\tau)
                                       -(3+cos\theta)(1-\tau)) +
                 2C_D(1-4\tau/((1+\tau)(1+\beta\cos\theta)) + 
                      8\tau^2/((1+\tau)^2(1+\beta\cos\theta)^2)))
       lambda^2=|GC_24|^2, g_s^2=GC_4^2, N_D=6, N_C=3, C_F=4/3, C_D=10/3, 
       (for antitriplet diquark, N_D=3, C_D=4/3) 
       \tau=m_D^2/shat, \cos\theta=p1p3/|p1||p3|, \beta = (1-\tau)/(1+\tau)"""

    if color_rep == 6:
        N_D = 6.
        C_D = 10./3.
    else:
        N_D = 3.
        C_D = 4./3.

    # Define auxiliary quantities
    shat=dot(P[0],P[0])+dot(P[1],P[1])+2*dot(P[0],P[1])
    if color_rep == 6:
        tau=full_model.get('parameter_dict')['mdl_MSIX']**2/shat
    else:
        tau=full_model.get('parameter_dict')['mdl_MANTI3']**2/shat        
    costheta=dot3(P[0],P[2])/math.sqrt(dot3(P[0],P[0])*dot3(P[2],P[2]))
    beta=(1-tau)/(1+tau)

    # Calculate matrix element
    ANS = 8 * abs(full_model.get('coupling_dict')['GC_24'])**2 * \
          full_model.get('coupling_dict')['GC_4']**2 * N_D
    ANS = ANS*(4./3.*(4./(1-costheta)*(1/(1-tau) - 2*tau) \
                      -(3+costheta)*(1-tau)) + \
               2*C_D*(1-4*tau/((1+tau)*(1+beta*costheta)) + \
                      8*tau**2/((1+tau)**2*(1+beta*costheta)**2)))

    #   Divide by color and spin factors for final state
    ANS = ANS/N_D/3./2.
      
    return ANS

def dot(P1, P2):
    """Scalar product of two 4-vectors"""
    return P1[0]*P2[0]-P1[1]*P2[1]-P1[2]*P2[2]-P1[3]*P2[3]

def dot3(P1, P2):
    """Scalar product of 3-components of two 4-vectors"""
    return P1[1]*P2[1]+P1[2]*P2[2]+P1[3]*P2[3]

        
if __name__ == '__main__':
    unittest.unittest.main()
