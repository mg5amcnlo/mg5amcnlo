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
# TestModelReader
#===============================================================================
class TestColorSextetModel(unittest.TestCase):
    """Test class for the sextet diquark implementation"""


    def setUp(self):
        self.base_model = import_ufo.import_model('sextets')
        self.full_model = model_reader.ModelReader(self.base_model)
        self.full_model.set_parameters_and_couplings()
    
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

        p, w_rambo = process_checks.get_momenta(myproc, self.full_model)
        helas_writer = \
                   helas_call_writers.PythonUFOHelasCallWriter(self.base_model)
        
        amplitude = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(amplitude)

        stored_quantities = {}

        mg5_me_value = process_checks.evaluate_matrix_element(matrix_element,
                                                           stored_quantities,
                                                           helas_writer,
                                                           self.full_model,
                                                           p)

        comparison_value = uu_Dg(p, 6, self.full_model)

        self.assertAlmostEqual(mg5_me_value, comparison_value, 12)

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

        p, w_rambo = process_checks.get_momenta(myproc, self.full_model)
        helas_writer = \
                   helas_call_writers.PythonUFOHelasCallWriter(self.base_model)
        
        amplitude = diagram_generation.Amplitude(myproc)
        matrix_element = helas_objects.HelasMatrixElement(amplitude)

        stored_quantities = {}

        mg5_me_value = process_checks.evaluate_matrix_element(matrix_element,
                                                           stored_quantities,
                                                           helas_writer,
                                                           self.full_model,
                                                           p)

        comparison_value = gu_uxD(p, 6, self.full_model)

        self.assertAlmostEqual(mg5_me_value, comparison_value, 14)

# Global helper functions

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
    else:
        N_D = 3.
        C_D = 4./3.

    # Define auxiliary quantities
    shat=dot(P[0],P[0])+dot(P[1],P[1])+2*dot(P[0],P[1])
    if color_rep == 6:
        tau=full_model.get('parameter_dict')['MSIX']**2/shat
    else:
        tau=full_model.get('parameter_dict')['MANTI3']**2/shat        
    cos2theta=dot3(P[0],P[3])**2/(dot3(P[0],P[0])*dot3(P[3],P[3]))

    # Calculate matrix element
    ANS = 16*abs(full_model.get('coupling_dict')['GC_24'])**2*\
          full_model.get('coupling_dict')['GC_4']**2*N_D*(2*tau/(1-tau)**2 + 1)
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
        tau=full_model.get('parameter_dict')['MSIX']**2/shat
    else:
        tau=full_model.get('parameter_dict')['MANTI3']**2/shat        
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
