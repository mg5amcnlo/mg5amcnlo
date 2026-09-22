################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
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
from __future__ import absolute_import
import os

import tests.unit_tests as unittest

from madgraph import MG4DIR, MG5DIR

import madgraph.various.Density_functions as dens
import madgraph.various.misc as misc


class TestDensityLibrary(unittest.TestCase):
        
    def setUp(self):
        pass
    
    def test_utilities(self):
        """Test the functions outside the classes of Density_functions.py"""
        
        #momenta from a random event
        ptest = [[1.7161156244e+01, +0.0000000000e+00, +0.0000000000e+00, +1.7161156244e+01],
                [6.2580362598e+02, -0.0000000000e+00, -0.0000000000e+00, -6.2580362598e+02],
                [2.6935060716e+02, -5.6043661480e+01, +2.8570090576e+01, -2.4924965708e+02],
                [3.7361417506e+02, +5.6043661480e+01, -2.8570090576e+01, -3.5939281265e+0]]
        
        #opp_mom
        opposite_mom = dens.opp_momentum(ptest[3])
        opposite_mom_ref = [3.7361417506e+02, -5.6043661480e+01, +2.8570090576e+01, +3.5939281265e+0]
        for i in range(len(opposite_mom)):
            self.assertAlmostEqual(opposite_mom[i], opposite_mom_ref[i], places=7)

        #norm_momentum
        norm_mom_ref = 368.2628048266349
        self.assertAlmostEqual(dens.norm_momentum(ptest[3]), norm_mom_ref, places=7)

        #invert_momenta
        pinv_ref = [[1.7161156244e+01, 6.2580362598e+02, 2.6935060716e+02, 3.7361417506e+02],
                    [+0.0000000000e+00, -0.0000000000e+00, -5.6043661480e+01, +5.6043661480e+01],
                    [+0.0000000000e+00, -0.0000000000e+00, +2.8570090576e+01, -2.8570090576e+01],
                    [+1.7161156244e+01, -6.2580362598e+02, -2.4924965708e+02, -3.5939281265e+0]]
        pinv = dens.invert_momenta(ptest)
        for i in range(len(pinv_ref)):
            for j in range(len(pinv_ref[0])):
                self.assertAlmostEqual(pinv_ref[i][j], pinv[i][j], places=7)


        #trace_distance and fidelity_distance
        rho_BD = [[0.46859656, 0.0095551j, -0.0095551j, 0.36650362],
                  [-0.0095551j, 0.03140344, 0.02276234, 0.0095551j],
                  [0.0095551j, 0.02276234, 0.03140344, -0.0095551j],
                  [0.36650362, -0.0095551j, 0.0095551j, 0.46859656+0]]
        sigma_BD=[[0.37611649, 0.03209884j, -0.03209884j, 0.12687239],
                  [-0.03209884j, 0.12388351, 0.04135733, 0.03209884j],
                  [0.03209884j, 0.04135733, 0.12388351, -0.03209884j],
                  [0.12687239, -0.03209884j, 0.03209884j, 0.37611649]]

        trace_distance_ref = 0.3321113000000001
        self.assertAlmostEqual(dens.trace_distance(rho_BD, sigma_BD), trace_distance_ref, places=7)
        fidelity_ref = 0.9283097655262389
        self.assertAlmostEqual(dens.Fidelity(rho_BD, sigma_BD), fidelity_ref, places=7)
        fidelity_distance_ref = 0.3718077180864046
        self.assertAlmostEqual(dens.Fidelity_distance(rho_BD, sigma_BD), fidelity_distance_ref, places=7)
    

    def test_DensityMatrixObservables(self):
        """Test the methods of the class DensityMatrixObservables in Density_functions.py"""

        #first check that it redirects density matrices to the correct class
        rho22 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        rho23 = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        rho33 = [[1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]]
        
        self.assertIsInstance(dens.DensityMatrixObservables(rho22), dens.DensityMatrixObservables22)
        self.assertIsInstance(dens.DensityMatrixObservables(rho23), dens.DensityMatrixObservables23)
        self.assertIsInstance(dens.DensityMatrixObservables(rho33), dens.DensityMatrixObservables33)

        
        rho_line_ref = [0.5, 0, 0, -0.5 + 0.2j, 0, 0, 0, 0, 0, 0.5]
        rho_line_instance = dens.DensityMatrixObservables(rho_line_ref)
        rho_square_ref = [[0.5, 0, 0, -0.5 - 0.2j], [0, 0, 0, 0], [0, 0, 0, 0], [-0.5 + 0.2j ,0, 0, 0.5]]
        rho_square_instance = dens.DensityMatrixObservables(rho_square_ref)


        #square_matrix
        rho_square = rho_line_instance.square_matrix()
        for i in range(len(rho_square_ref)):
            for j in range(len(rho_square_ref[0])):
                self.assertEqual(rho_square[i][j], rho_square_ref[i][j])

        #get_trace
        trace_line = rho_line_instance.get_trace()
        trace_square = rho_square_instance.get_trace()
        trace_ref = 1.
        self.assertEqual(trace_line, trace_ref)
        self.assertEqual(trace_square, trace_ref)

        #get_rho_normalised
        rho_non_normalised = dens.DensityMatrixObservables([0.2261061537657934, -0.02718853972747267-0.016925257981990807j, -0.029434628475716068-0.01830653640502701j, 0.010463408605257556+0.02069659668158197j, 0.14665879186677117, -0.06929673002563852+1.0360907010814602e-17j, 0.029617672608582495+0.01800888941336576j, 0.14665879186677114, 0.027371583860338788+0.016627610990330017j, 0.22610615376579346])
        rho_normalised_ref = [0.30328248, -0.03646875-0.02270232j, -0.03948149-0.02455507j, 0.01403486+0.02776092j, 0.19671752, -0.09294963, 0.03972701+0.02415582j, 0.19671752, 0.03671427+0.02230308j, 0.30328248]

        rho_normalised = rho_non_normalised.get_rho_normalised()
        self.assertAlmostEqual(dens.DensityMatrixObservables(rho_normalised).get_trace(), 1, places=7)

        for i in range(len(rho_non_normalised)):
            self.assertAlmostEqual(rho_normalised[i], rho_normalised_ref[i], places=7)
        
        #Get_Purity
        purity_ref = 0.296599066462279
        self.assertAlmostEqual(dens.DensityMatrixObservables(rho_normalised).Get_Purity(), purity_ref, places=7)

        #Get_Normalised_Purity
        normalised_purity_ref = 0.06213208861637202
        self.assertAlmostEqual(dens.DensityMatrixObservables(rho_normalised).Get_Normalised_Purity(), normalised_purity_ref, places=7)

        #Shannon_Entropy
        x = 0.3
        self.assertAlmostEqual(rho_line_instance.Shannon_Entropy(x), 0.8812908992306927, places=7)

        #Von_Neumann_entropy
        self.assertAlmostEqual(dens.DensityMatrixObservables(rho_normalised).Von_Neumann_entropy(), 1.8268234912200536, places=7)

        #Partial_Transpose
        rho_bell_state = [[0.5, 0, 0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [-0.5, 0, 0, 0.5]]
        rho_bell_state_instance = dens.DensityMatrixObservables(rho_bell_state)
        rhoT_ref = [[0.5, 0, 0, 0], [0, 0, -0.5, 0], [0, -0.5, 0, 0], [0, 0, 0, 0.5]]
        rho1T = rho_bell_state_instance.Partial_Transpose(1, ["fermion", "fermion"])
        rho2T = rho_bell_state_instance.Partial_Transpose(2, ["fermion", "fermion"])
        for i in range(len(rhoT_ref)):
            for j in range(len(rhoT_ref[0])):
                self.assertAlmostEqual(rho1T[i][j], rhoT_ref[i][j], places=7)
                self.assertAlmostEqual(rho2T[i][j], rhoT_ref[i][j], places=7)
        
        #Partial_Trace
        rho_traced_ref = [[0.5, 0], [0, 0.5]]
        rho_traced1 = rho_bell_state_instance.Partial_Trace(1, ["fermion", "fermion"])
        rho_traced2 = rho_bell_state_instance.Partial_Trace(2, ["fermion", "fermion"])
        for i in range(len(rho_traced_ref)):
            for j in range(len(rho_traced_ref[0])):
                self.assertAlmostEqual(rho_traced_ref[i][j], rho_traced1[i][j], places=7)
                self.assertAlmostEqual(rho_traced_ref[i][j], rho_traced2[i][j], places=7)

        #Negativity
        rho_negativity = [[0.47644727, 0.00761944j, -0.00761944j, 0.40012663],
                          [-0.00761944j, 0.02355273, 0.01578616, 0.00761944j],
                          [0.00761944j, 0.01578616, 0.02355273, -0.00761944j],
                          [0.40012663, -0.00761944j, 0.00761944j, 0.47644727]]
        negativity_ref = 0.3765739
        log_negativity_ref = 0.8099476283499019
        self.assertAlmostEqual(dens.DensityMatrixObservables(rho_negativity).Negativity(["fermion", "fermion"])[0], negativity_ref)
        self.assertAlmostEqual(dens.DensityMatrixObservables(rho_negativity).Negativity(["fermion", "fermion"])[1], log_negativity_ref)

        #PeresHorodecki_criterion
        PeresHorodecki_flag_ref = True
        PeresHorodecki_eig_ref = [-0.3765739, 0.42044455, 0.46066111, 0.49546824]
        self.assertTrue(dens.DensityMatrixObservables(rho_negativity).PeresHorodecki_criterion(["fermion", "fermion"])[0])
        for i in range(len(PeresHorodecki_eig_ref)):
            self.assertAlmostEqual(dens.DensityMatrixObservables(rho_negativity).PeresHorodecki_criterion(["fermion", "fermion"])[1][i], PeresHorodecki_eig_ref[i])

        #shift_clock
        shift_ref = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        clock_ref = [[1, 0, 0], [0, -0.5+0.8660254j, 0], [0, 0, -0.5-0.8660254j]]
        clock, shift = rho_line_instance.shift_clock(3)

        for i in range(len(shift_ref)):
            for j in range(len(clock_ref)):
                self.assertAlmostEqual(shift_ref[i][j], shift[i][j], places=7)
                self.assertAlmostEqual(clock_ref[i][j], clock[i][j], places=7)
        
        #Displacement_Operator
        Displacement_ref =[
        [[1.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 1.+0.j, 0.+0.j],
         [0.+0.j, 0.+0.j, 1.+0.j]],

        [[ 1. +0.j,  0. +0.j,  0. +0.j],
         [ 0. +0.j, -0.5+0.8660254j,  0. +0.j],
         [ 0. +0.j,  0. +0.j, -0.5-0.8660254j]],

        [[ 1. +0.j,  0. +0.j,  0. +0.j],
         [ 0. +0.j, -0.5-0.8660254j,  0. +0.j],
         [ 0. +0.j,  0. +0.j,-0.5+0.8660254j]],
        
        [[0.+0.j, 0.+0.j, 1.+0.j],
         [1.+0.j, 0.+0.j, 0.+0.j],
         [0.+0.j, 1.+0.j, 0.+0.j]],
        
        [[ 0. +0.j,  0. +0.j, -0.5+0.8660254j],
         [-0.5-0.8660254j,  0. +0.j,  0. +0.j],
         [ 0. +0.j,  1. +0.j,  0. +0.j]],
        
        [[ 0. +0.j,  0. +0.j, -0.5-0.8660254j],
         [-0.5+0.8660254j,  0. +0.j,  0. +0.j],
         [ 0. +0.j,  1. +0.j,0. +0.j]],
        
        [[0.+0.j, 1.+0.j, 0.+0.j],
         [0.+0.j, 0.+0.j, 1.+0.j],
         [1.+0.j, 0.+0.j, 0.+0.j]],
    
        [[ 0. +0.j, -0.5-0.8660254j,  0. +0.j],
         [ 0. +0.j,  0. +0.j,  1. +0.j],
         [-0.5+0.8660254j,  0. +0.j,  0. +0.j]],
    
        [[ 0. +0.j, -0.5+0.8660254j,  0. +0.j],
         [ 0. +0.j,  0. +0.j,  1. +0.j],
         [-0.5-0.8660254j,  0. +0.j,  0. +0.j]]
        ]

        Displacement_operators = rho_line_instance.Displacement_Operator(3)
        for o in range(len(Displacement_operators)):
            for i in range(len(Displacement_operators[0])):
                for j in range(len(Displacement_operators[0][0])):
                    self.assertAlmostEqual(Displacement_operators[o][i][j], Displacement_ref[o][i][j], places=7)
        
        #Discrete_phase_point_operator_00
        A00 = rho_line_instance.Discrete_phase_point_operator_00(3, 3)
        A00_ref =  [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                    [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]

        for i in range(len(A00_ref)):
            for j in range(len(A00_ref[0])):
                self.assertAlmostEqual(A00[i][j], A00_ref[i][j], places=7)

        #Discrete_phase_point_operators, oto long to check all 81 matrices, so we check 1 of them.
        Discrete_operator_test = rho_line_instance.Discrete_phase_point_operators(3, 3)[34]

        Discrete_operator_ref =[[0. +0.j, 0. +0.j, 0. +0.j, 0. +0., 0. +0.j, 0. +0.j, 0. +0.j, -0.5-0.8660254j, 0. +0.j],
                                [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, -0.5+0.8660254j, 0. +0.j, 0. +0.j],
                                [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 1. +0.j],
                                [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, -0.5-0.8660254j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
                                [0. +0.j, 0. +0.j, 0. +0.j, -0.5+0.8660254j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
                                [0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 1. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
                                [0. +0.j, -0.5-0.8660254j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
                                [-0.5+0.8660254j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j],
                                [0. +0.j, 0. +0.j, 1. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j, 0. +0.j]]
        
        for i in range(len(Discrete_operator_ref)):
            for j in range(len(Discrete_operator_ref[0])):
                self.assertAlmostEqual(Discrete_operator_test[i][j], Discrete_operator_ref[i][j], places=7)

        
        #Sum_Discrete_Wigner
        # I need to give a real 9x9 matrix here
        rho99 = [6.609265854757986e-05, -0.000188914946191952, -0.002147590256111751, 0.00016962969113726122, 0.00013218531712450824, -0.0001889149461919511, -0.005668590122388772, 0.00016962969113726414, 6.609265854757994e-05, 0.0005466070797326989, 0.004928990195802296, -0.00047823353580555276, -0.0003778298924677921, 0.000546607079732697, 0.01741227021163045, -0.00047823353580556165, -0.00018891494619195222, 0.29061473903772117, -0.00672142584783465, -0.004295180513177092, 0.004928990195802183, -0.03663868194212968, -0.006721425847834642, -0.002147590256111756, 0.00044198691568819124, 0.0003392593823498464, -0.0004782335358055501, -0.013339142036928018, 0.00044198691568819823, 0.00016962969113726146, 0.00026437063430771344, -0.00037782989246779036, -0.011337180247294737, 0.00033925938234985226, 0.0001321853171245084, 0.000546607079732695, 0.017412270211630464, -0.00047823353580555905, -0.00018891494619195136, 0.7070115170200342, -0.013339142036928368, -0.005668590122388777, 0.00044198691568820506, 0.0001696296911372644, 6.609265854758002e-05]
        rho99_instance = dens.DensityMatrixObservables(rho99)
        Wigner_ref = 1.1496740811277208
        self.assertAlmostEqual(rho99_instance.Sum_Discrete_Wigner(3, 3), Wigner_ref, places=7)

        #Get_Coherence
        ref_coherence = 0.317614817148771
        self.assertAlmostEqual(rho99_instance.Get_Coherence(), ref_coherence, places=7)
        
    
    
    def test_DensityMatrixObservables22(self):
        """Test the methods of the class DensityMatrixObservables22 in Density_functions.py"""

        #first check that it redirects density matrices to the correct class. Even if you call the class with bad dimensions, the code should redirect to the correct one
        rho22 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        rho23 = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        rho33 = [[1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]]
        
        self.assertIsInstance(dens.DensityMatrixObservables22(rho22), dens.DensityMatrixObservables22)
        self.assertIsInstance(dens.DensityMatrixObservables22(rho23), dens.DensityMatrixObservables23)
        self.assertIsInstance(dens.DensityMatrixObservables22(rho33), dens.DensityMatrixObservables33)

        #Get_Correlations
        rho22_ref = [0.30328248, -0.03646875-0.02270232j, -0.03948149-0.02455507j, 0.01403486+0.02776092j, 0.19671752, -0.09294963, 0.03972701+0.02415582j, 0.19671752, 0.03671427+0.02230308j, 0.30328248]
        rho22_ref_instance = dens.DensityMatrixObservables22(rho22_ref)

        Cmatrix_ref = [ [-0.15782954,  0.05552184, -0.158417  ],
                        [ 0.05552184, -0.21396898, -0.09742178],
                        [-0.14636604, -0.0900108 ,  0.21312992]]
        Cmatrix2_ref = [[ 0.04941586, -0.00746841, -0.01160114],
                        [-0.00746841,  0.05696734, -0.00713436],
                        [-0.01160114, -0.00713436,  0.08001131]]

        Cmatrix, Cmatrix2 = rho22_ref_instance.Get_Correlations()
        for i in range(len(Cmatrix)):
            for j in range(len(Cmatrix[0])):
                self.assertAlmostEqual(Cmatrix[i][j], Cmatrix_ref[i][j], places=6) #only 6 digits, why ?
                self.assertAlmostEqual(Cmatrix2[i][j], Cmatrix2_ref[i][j], places=6)

        #Get_Polarisations
        Pol1_ref = [ 0.00049104, -0.0007985, 0]
        Pol2_ref = [ 0.00049104, -0.00079848, 0]

        Pol1, Pol2 = rho22_ref_instance.Get_Polarisations()
        for i in range(len(Pol1)):
            self.assertAlmostEqual(Pol1[i], Pol1_ref[i], places=7)
            self.assertAlmostEqual(Pol2[i], Pol2_ref[i], places=7)

        #spin_expectation
        Spin1_ref = [0.00024552, -0.00039925,  0.]
        Spin2_ref = [0.00024552, -0.00039924,  0.]
        
        Spin1, Spin2 = rho22_ref_instance.spin_expectation()
        for i in range(len(Pol1)):
            self.assertAlmostEqual(Spin1[i], Spin1_ref[i], places=7)
            self.assertAlmostEqual(Spin2[i], Spin2_ref[i], places=7)

        #spinspin_expectation
        Spinspin_ref = [[-0.03945739,  0.01388046, -0.03960425],
                        [ 0.01388046, -0.05349225, -0.02435544],
                        [-0.03659151, -0.0225027,   0.05328248]]
        
        Spinspin = rho22_ref_instance.spinspin_expectation()
        for i in range(len(Spinspin)):
            for j in range(len(Spinspin[0])):
                self.assertAlmostEqual(Spinspin[i][j], Spinspin_ref[i][j], places=6) #only 6 digits, why ?

        #add_phase_density
        rho22_change_phase_ref = [[0.30328248+0.j, -0.03646875+0.02270232j, 0.03948149-0.02455507j, -0.01403486+0.02776092j],
                                  [-0.03646875-0.02270232j, 0.19671752+0.j, 0.09294963+0.j, -0.03972701+0.02415582j],
                                  [0.03948149+0.02455507j, 0.09294963+0.j, 0.19671752+0.j, 0.03671427-0.02230308j],
                                  [-0.01403486-0.02776092j, -0.03972701-0.02415582j, 0.03671427+0.02230308j, 0.30328248+0.j]]
        rho22_ref_instance.add_phase_density([0, 3.141592653589, 0, 0])
        for i in range(len(rho22_change_phase_ref)):
            for j in range(len(rho22_change_phase_ref[0])):
                self.assertAlmostEqual(rho22_change_phase_ref[i][j], rho22_ref_instance.density_matrix[i][j], places=7)
        
        #CHSH_inequality
        rho22_ref_instance = dens.DensityMatrixObservables22(rho22_ref)
        CHSH_ref = 0.8982721341461017
        self.assertFalse(rho22_ref_instance.CHSH_inequality()[1])
        self.assertAlmostEqual(rho22_ref_instance.CHSH_inequality()[0], CHSH_ref, places=7)
        
        #Get_Concurrence
        rho22_concurrence = [0.030976625443915044, -0.07739362425863269j, 0.07739362425863269j, -0.030976625443915044, 0.469023374556085, 0.08229480163704686, -0.07739362425863269j, 0.469023374556085, 0.07739362425863269j, 0.030976625443915044]
        rho22_concurrence_instance = dens.DensityMatrixObservables22(rho22_concurrence)
        concurrence_ref = 0.10263634300735841
        self.assertAlmostEqual(rho22_concurrence_instance.Get_Concurrence(), concurrence_ref, places=7)

        #Get_Dcoef
        D_coeff = rho22_concurrence_instance.Get_Dcoef()
        D_coeff_ref = [-0.18230476389205083, 0.25072899881622657, 0.3333333333333333, -0.40175756825750913]
        self.assertTrue(D_coeff[4])
        for i in range(4):
            self.assertAlmostEqual(D_coeff[i], D_coeff_ref[i], places=7)
        
        #Get_Entanglement_Formation
        entf_ref = 0.026420446091385563
        self.assertAlmostEqual(rho22_concurrence_instance.Get_Entanglement_Formation(), entf_ref, places=7)

        #get_Pauli_string
        Pauli_string_ref =[[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]],
                           [[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]],
                           [[0.+0.j, 0.-1.j, 0.+0.j, 0.-0.j],
                            [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j],
                            [0.+0.j, 0.-0.j, 0.+0.j, 0.-1.j],
                            [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j]],
                           [[ 1,  0,  0,  0],
                            [ 0, -1,  0,  0],
                            [ 0,  0,  1,  0],
                            [ 0,  0,  0, -1]],
                           [[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0]],
                           [[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]],
                           [[0.+0.j, 0.-0.j, 0.+0.j, 0.-1.j],
                            [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
                            [0.+0.j, 0.-1.j, 0.+0.j, 0.-0.j],
                            [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]],
                           [[ 0,  0,  1,  0],
                            [ 0,  0,  0, -1],
                            [ 1,  0,  0,  0],
                            [ 0, -1,  0,  0]],
                           [[0.+0.j, 0.+0.j, 0.-1.j, 0.-0.j],
                            [0.+0.j, 0.+0.j, 0.-0.j, 0.-1.j],
                            [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j],
                            [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j]],  
                           [[0.+0.j, 0.+0.j, 0.-0.j, 0.-1.j],
                            [0.+0.j, 0.+0.j, 0.-1.j, 0.-0.j],
                            [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j],
                            [0.+1.j, 0.+0.j, 0.+0.j, 0.+0.j]],
                           [[ 0.+0.j,  0.-0.j,  0.-0.j, -1.+0.j],
                            [ 0.+0.j,  0.+0.j,  1.-0.j,  0.-0.j],
                            [ 0.+0.j,  1.-0.j,  0.+0.j,  0.-0.j],
                            [-1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]],
                           [[ 0.+0.j,  0.+0.j,  0.-1.j,  0.-0.j],
                            [ 0.+0.j, -0.+0.j,  0.-0.j,  0.+1.j],
                            [ 0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j],
                            [ 0.+0.j, -0.-1.j,  0.+0.j, -0.+0.j]],    
                           [[ 1,  0,  0,  0],
                            [ 0,  1,  0,  0],
                            [ 0,  0, -1,  0],
                            [ 0,  0,  0, -1]],
                           [[ 0,  1,  0,  0],
                            [ 1,  0,  0,  0],
                            [ 0,  0,  0, -1],
                            [ 0,  0, -1,  0]],
                           [[ 0.+0.j,  0.-1.j,  0.+0.j,  0.-0.j],
                            [ 0.+1.j,  0.+0.j,  0.+0.j,  0.+0.j],
                            [ 0.+0.j,  0.-0.j, -0.+0.j,  0.+1.j],
                            [ 0.+0.j,  0.+0.j, -0.-1.j, -0.+0.j]],
                           [[ 1,  0,  0,  0],
                            [ 0, -1,  0,  0],
                            [ 0,  0, -1,  0],
                            [ 0,  0,  0,  1]]]
        Pauli_string = rho22_concurrence_instance.get_Pauli_string(2)
        for o in range(len(Pauli_string_ref)):
            for i in range(len(Pauli_string_ref[0])):
                for j in range(len(Pauli_string_ref[0][0])):
                    self.assertAlmostEqual(Pauli_string[o][i][j], Pauli_string_ref[o][i][j], places=7)

        #Magic_Pure
        rho_pure = [[0.25, -0.25, -0.25, 0.25], [-0.25, 0.25, 0.25, -0.25], [-0.25, 0.25, 0.25, -0.25], [0.25, -0.25, -0.25, 0.25]]
        rho_pure_instance = dens.DensityMatrixObservables22(rho_pure)
        self.assertAlmostEqual(rho_pure_instance.Magic_Pure(), 0, places=7) #have not found a good pure state with non zero magic

        #Magic_Mixed
        magic_ref = 0.32785008450868186
        self.assertAlmostEqual(rho22_concurrence_instance.Magic_Mixed(), magic_ref, places=7)

        #Get_Discord
        discord_ref = 0.007612231907878438
        self.assertAlmostEqual(rho22_concurrence_instance.Get_Discord(maxiter=1000), discord_ref, places=4) #this test can sometimes fail because the minimisation can get stuck in a local minimum. I need to improve the minimisation.

    
    
    def test_DensityMatrixObservables23(self):
        """Test the methods of the class DensityMatrixObservables23 in Density_functions.py"""

        #first check that it redirects density matrices to the correct class. Even if you call the class with bad dimensions, the code should redirect to the correct one
        rho22 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        rho23 = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        rho33 = [[1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]]
        
        self.assertIsInstance(dens.DensityMatrixObservables22(rho22), dens.DensityMatrixObservables22)
        self.assertIsInstance(dens.DensityMatrixObservables22(rho23), dens.DensityMatrixObservables23)
        self.assertIsInstance(dens.DensityMatrixObservables22(rho33), dens.DensityMatrixObservables33)


        rho23_ref = [0.11827190404641179, 0.09132426886575808j, -0.042267170618961764, -0.12574583437968376j, 0.06910006105562824, 0.11964231988177088j, 0.13316205558575803, 0.046079163029620104j, -0.13708660214841104, -0.16237356656283053j, -0.025078225433441292, 0.017989570175999288, 0.0535193976498959j, -0.04808724868389498, -0.01755246214871665j, 0.1592214764880326, 0.14306070456249606j, -0.05221899091345484, 0.23008711466715007, -0.1345077392674271j, 0.3412678790366483]
        rho23_ref_instance = dens.DensityMatrixObservables23(rho23_ref)

        #Get_Correlations
        Corr, Corr2 = rho23_ref_instance.Get_Correlations(["fermion", "boson"])
        Corr_ref = [[-0.06798654,  0.,  0.,  0.,  0.06612292, -0.07316547,  0.,  0.],
                    [ 0., -0.20618666,  0.03662773,  0.17316172,  0., 0., -0.02300902, -0.14607798],
                    [ 0., -0.05173644,  0.02798774,  0.00995182,  0., 0.,  0.1805869 ,  0.14684384]]
        Corr2_ref = [[ 0.00462217,  0.,  0.,  0., -0.00449547,  0.00497427,  0.,  0.],
                     [ 0.,  0.0451896 , -0.00900014, -0.03621851,  0., 0., -0.00459877,  0.02252215],
                     [ 0., -0.00900014,  0.0021249 ,  0.00662105,  0., 0.,  0.00421145, -0.00124068],
                     [ 0., -0.03621851,  0.00662105,  0.03008402,  0., 0., -0.00218711, -0.02383375],
                     [-0.00449547, 0., 0.,  0.,  0.00437224, -0.00483791,  0.,  0.],
                     [ 0.00497427,  0.,  0.,  0., -0.00483791, 0.00535319,  0.,  0.],
                     [ 0., -0.00459877,  0.00421145, -0.00218711,  0., 0.,  0.03314104,  0.02987919],
                     [ 0.,  0.02252215, -0.00124068, -0.02383375,  0., 0.,  0.02987919,  0.04290189]]
        
        for i in range(len(Corr_ref)):
            for j in range(len(Corr_ref[0])):
                self.assertAlmostEqual(Corr_ref[i][j], Corr[i][j], places=6)
        
        for i in range(len(Corr2_ref)):
            for j in range(len(Corr2_ref[0])):
                self.assertAlmostEqual(Corr2_ref[i][j], Corr2[i][j], places=6)

        #Get_Polarisations
        Pol1, Pol2 = rho23_ref_instance.Get_Polarisations(["fermion", "boson"])
        Pol1_ref = [ 0. -4.51028104e-17j, -0.61134373, -0.46115294]
        Pol2_ref = [ 0.+0.j,  0.23438497+0.j, -0.04287789+0.j, -0.09448616+0.j,  0.+0.j,  0.+0.j, -0.08842858+0.j, -0.02245094+0.j]
        for i in range(len(Pol1_ref)):
            self.assertAlmostEqual(Pol1_ref[i], Pol1[i], places=7)
        for j in range(len(Pol2_ref)):
            self.assertAlmostEqual(Pol2_ref[j], Pol2[j], places=7)

        #spin_expectation
        spin1, spin2 = rho23_ref_instance.spin_expectation(["fermion", "boson"])
        spin1_ref = [0., -0.61134373, -0.46115294]
        spin2_ref = [0., 0.20641352, -0.08176407]
        for i in range(len(spin1_ref)):
            self.assertAlmostEqual(spin1_ref[i], spin1[i], places=7)
            self.assertAlmostEqual(spin2_ref[i], spin2[i], places=7)

        #spinspin_expectation
        spinspin = rho23_ref_instance.spinspin_expectation(["fermion", "boson"])
        spin_spin_ref = [[-0.09980955,  0.,          0.        ],
                         [ 0.,         -0.16206582, -0.10819337],
                         [ 0.,          0.09111104,  0.14116437]]
        
        for i in range(len(spinspin)):
            for j in range(len(spinspin[0])):
                self.assertAlmostEqual(spinspin[i][j], spin_spin_ref[i][j], places=7)

    
    def test_DensityMatrixObservables33(self):
        """Test the methods of the class DensityMatrixObservables33 in Density_functions.py"""

        #first check that it redirects density matrices to the correct class. Even if you call the class with bad dimensions, the code should redirect to the correct one
        rho22 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        rho23 = [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        rho33 = [[1, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]]
        
        self.assertIsInstance(dens.DensityMatrixObservables22(rho22), dens.DensityMatrixObservables22)
        self.assertIsInstance(dens.DensityMatrixObservables22(rho23), dens.DensityMatrixObservables23)
        self.assertIsInstance(dens.DensityMatrixObservables22(rho33), dens.DensityMatrixObservables33)


        rho33_ref = [0.00013649499087990615, 0.0006255321592898754j, -0.0013951394040219625, 0.000625532159305027j, -0.0026700764297943554, -0.004575850399497401j, -0.0013951394040186487, -0.00457585039940916j, -0.009095235374164576, 0.0031158346387996673, 0.006378646903237555j, 0.0031158346388664984, 0.012159472975998514j, -0.020974043743924517, 0.006378646903222683j, -0.020974043743520084, 0.04168183891712595j, 0.014260872680680886, -0.0063786469033925795j, 0.027295967956988434, 0.04677034940924241j, 0.014260872680646998, 0.046770349408340475j, 0.09296400679286349, 0.00311583463893333, 0.012159472976295712j, -0.02097404374443242, 0.006378646903377708j, -0.020974043744027986, 0.041681838918135566j, 0.05225508534832526, 0.08951033638354483j, 0.027295967956923517, 0.0895103363818187j, 0.1779184235218975, 0.15340060761394816, -0.046770349409131315j, 0.15340060761098998, -0.3049081592819143j, 0.014260872680613107, 0.046770349408229384j, 0.09296400679264268, 0.15340060760803179, -0.3049081592760344j, 0.606053789799788]
        rho33_ref_instance = dens.DensityMatrixObservables33(rho33_ref)

        #Get_Correlations
        Corr, Corr2 = rho33_ref_instance.Get_Correlations()
        Corr_ref = [[ 0.00022288, 0. ,0. ,0. ,0.0009014 ,0.00316096 ,0. ,0.],
                    [ 0., 0.00289296, -0.00576697, -0.00547725, 0., 0., 0.02413501, -0.02331216],
                    [ 0., -0.00576697, 0.01153998, 0.00978945, 0., 0., -0.04156584, 0.03264351],
                    [ 0., -0.00547725 ,0.00978945, 0.00258282, 0., 0., -0.00254426, -0.06013022],
                    [ 0.0009014, 0., 0., 0., 0.01167805, 0.04422609, 0., 0.],
                    [ 0.00316096, 0., 0., 0., 0.04422609, 0.16565952, 0., 0.],
                    [ 0., 0.02413501, -0.04156584, -0.00254426, 0., 0., -0.01225891, 0.20371957],
                    [ 0., -0.02331216, 0.03264351, -0.06013022, 0., 0. ,0.20371957 ,0.15101604]]
        
        Corr2_ref = [[ 1.08538753e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.50524488e-04, 5.64213288e-04, 0.00000000e+00, 0.00000000e+00],
                     [ 0.00000000e+00, 1.19758274e-03, -1.90103625e-03, 1.25391211e-03, 0.00000000e+00, 0.00000000e+00, -4.72154623e-03,  1.46991584e-03],
                     [ 0.00000000e+00, -1.90103625e-03, 3.05558068e-03, -1.68726586e-03, 0.00000000e+00,  0.00000000e+00, 6.51591237e-03, -3.61557832e-03],
                     [ 0.00000000e+00,  1.25391211e-03, -1.68726586e-03,  3.75462133e-03, 0.00000000e+00, 0.00000000e+00, -1.27641848e-02, -9.30699940e-03],
                     [ 1.50524488e-04,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00, 2.09313687e-03, 7.84579734e-03,  0.00000000e+00,  0.00000000e+00],
                     [ 5.64213288e-04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 7.84579734e-03,  2.94090142e-02,  0.00000000e+00,  0.00000000e+00],
                     [ 0.00000000e+00, -4.72154623e-03,  6.51591237e-03, -1.27641848e-02, 0.00000000e+00,  0.00000000e+00,  4.39686364e-02,  2.65010362e-02],
                     [ 0.00000000e+00,  1.46991584e-03, -3.61557832e-03, -9.30699940e-03, 0.00000000e+00,  0.00000000e+00, 2.65010362e-02,  6.95322081e-02]]

        for i in range(len(Corr_ref)):
            for j in range(len(Corr_ref[0])):
                self.assertAlmostEqual(Corr_ref[i][j], Corr[i][j], places=5) #lack number of digits to put more
                self.assertAlmostEqual(Corr2_ref[i][j], Corr2[i][j], places=5)

        #Get_Polarisations
        Pol1, Pol2 = rho33_ref_instance.Get_Polarisations()
        Pol1_ref = [ 0.+0.0000000e+00j,  0.05955535+0.0000000e+00j, -0.09562916+0.0000000e+00j,  0.07059482+0.0000000e+00j, 0.-6.9388939e-18j,  0.+0.0000000e+00j, -0.20901918+0.0000000e+00j, -0.38138194+0.0000000e+00j]
        Pol2_ref = [ 0.+1.73472348e-18j, 0.05955535+0.00000000e+00j, -0.09562916+0.00000000e+00j,  0.07059482+0.00000000e+00j, 0.+0.00000000e+00j,  0.+2.77555756e-17j, -0.20901918+0.00000000e+00j, -0.38138194+0.00000000e+00j]
        
        for i in range(len(Pol1_ref)):
            self.assertAlmostEqual(Pol1_ref[i], Pol1[i], places=7)
            self.assertAlmostEqual(Pol2_ref[i], Pol2[i], places=7)
        
        #spin_expectation
        spin1, spin2 = rho33_ref_instance.spin_expectation()
        spin1_ref = [ 0., -0.21137376, -0.75620207]
        spin2_ref = [ 0., -0.21137376, -0.75620207]

        for i in range(len(spin1_ref)):
            self.assertAlmostEqual(spin1_ref[i], spin1[i], places=7)
            self.assertAlmostEqual(spin2_ref[i], spin2[i], places=7)

        #spinspin_expectation
        spinspin = rho33_ref_instance.spinspin_expectation()
        spinspin_ref = [[0.34440864, 0.,         0.        ],
                        [0.,         0.07780812, 0.3749674 ],
                        [0.,         0.3749674,  0.57766854]]

        for i in range(len(spinspin)):
            for j in range(len(spinspin[0])):
                self.assertAlmostEqual(spinspin[i][j], spinspin_ref[i][j], places=7)

        #add_phase_density
        rho33_ref_instance.add_phase_density([0, 3.14159265, 0, 0, 0, 0])
        
        rho_add_density_ref = [[ 1.36494991e-04+0.j, 0.00000000e+00-0.00062553j, -1.39513940e-03-0.j, 0.00000000e+00-0.00062553j, -2.67007643e-03-0.j, -0.00000000e+00+0.00457585j, -1.39513940e-03-0.j, -0.00000000e+00+0.00457585j, -9.09523537e-03-0.j],
                               [ 0.00000000e+00+0.00062553j, 3.11583464e-03+0.j, 0.00000000e+00-0.00637865j, 3.11583464e-03-0.j, 0.00000000e+00-0.01215947j, -2.09740437e-02-0.j, 0.00000000e+00-0.00637865j, -2.09740437e-02-0.j, 0.00000000e+00-0.04168184j],
                               [-1.39513940e-03+0.j, 0.00000000e+00+0.00637865j, 1.42608727e-02+0.j, -0.00000000e+00+0.00637865j, 2.72959680e-02-0.j, 0.00000000e+00-0.04677035j, 1.42608727e-02-0.j, 0.00000000e+00-0.04677035j, 9.29640068e-02-0.j],
                               [ 0.00000000e+00+0.00062553j, 3.11583464e-03+0.j, -0.00000000e+00-0.00637865j, 3.11583464e-03+0.j, 0.00000000e+00-0.01215947j, -2.09740437e-02-0.j, 0.00000000e+00-0.00637865j, -2.09740437e-02-0.j, 0.00000000e+00-0.04168184j],
                               [-2.67007643e-03+0.j, 0.00000000e+00+0.01215947j, 2.72959680e-02+0.j, 0.00000000e+00+0.01215947j, 5.22550853e-02+0.j, 0.00000000e+00-0.08951034j, 2.72959680e-02-0.j, 0.00000000e+00-0.08951034j, 1.77918424e-01-0.j],
                               [-0.00000000e+00-0.00457585j, -2.09740437e-02+0.j, 0.00000000e+00+0.04677035j, -2.09740437e-02+0.j, 0.00000000e+00+0.08951034j, 1.53400608e-01+0.j, -0.00000000e+00+0.04677035j, 1.53400608e-01-0.j, -0.00000000e+00+0.30490816j],
                               [-1.39513940e-03+0.j, 0.00000000e+00+0.00637865j, 1.42608727e-02+0.j, 0.00000000e+00+0.00637865j, 2.72959680e-02+0.j, -0.00000000e+00-0.04677035j, 1.42608727e-02+0.j, 0.00000000e+00-0.04677035j, 9.29640068e-02-0.j],
                               [-0.00000000e+00-0.00457585j, -2.09740437e-02+0.j, 0.00000000e+00+0.04677035j, -2.09740437e-02+0.j, 0.00000000e+00+0.08951034j, 1.53400608e-01+0.j, 0.00000000e+00+0.04677035j, 1.53400608e-01+0.j, -0.00000000e+00+0.30490816j],
                               [-9.09523537e-03+0.j, 0.00000000e+00+0.04168184j, 9.29640068e-02+0.j, 0.00000000e+00+0.04168184j, 1.77918424e-01+0.j, -0.00000000e+00-0.30490816j, 9.29640068e-02+0.j, -0.00000000e+00-0.30490816j, 6.06053790e-01+0.j]]
        
        rho_add_density = rho33_ref_instance.square_matrix()
        for i in range(len(rho_add_density_ref)):
            for j in range(len(rho_add_density_ref[0])):
                self.assertAlmostEqual(rho_add_density[i][j], rho_add_density_ref[i][j], places=5) #Why does it work only up to 5 digits ?

        #ConcLB2
        rho33_ref = [0.00013649499087990615, 0.0006255321592898754j, -0.0013951394040219625, 0.000625532159305027j, -0.0026700764297943554, -0.004575850399497401j, -0.0013951394040186487, -0.00457585039940916j, -0.009095235374164576, 0.0031158346387996673, 0.006378646903237555j, 0.0031158346388664984, 0.012159472975998514j, -0.020974043743924517, 0.006378646903222683j, -0.020974043743520084, 0.04168183891712595j, 0.014260872680680886, -0.0063786469033925795j, 0.027295967956988434, 0.04677034940924241j, 0.014260872680646998, 0.046770349408340475j, 0.09296400679286349, 0.00311583463893333, 0.012159472976295712j, -0.02097404374443242, 0.006378646903377708j, -0.020974043744027986, 0.041681838918135566j, 0.05225508534832526, 0.08951033638354483j, 0.027295967956923517, 0.0895103363818187j, 0.1779184235218975, 0.15340060761394816, -0.046770349409131315j, 0.15340060761098998, -0.3049081592819143j, 0.014260872680613107, 0.046770349408229384j, 0.09296400679264268, 0.15340060760803179, -0.3049081592760344j, 0.606053789799788]
        rho33_ref_instance = dens.DensityMatrixObservables33(rho33_ref)
        ConcLB2 = rho33_ref_instance.ConcLB2()
        ConcLB2_ref = 0.5039731497865665
        self.assertAlmostEqual(ConcLB2_ref, ConcLB2, places=7)
        
        #ConcUB2
        ConcUB2 = rho33_ref_instance.ConcUB2()
        ConcUB2_ref = 0.5060668944206046
        self.assertAlmostEqual(ConcUB2_ref, ConcUB2, places=7)
        
        #Get_Mana
        Mana = rho33_ref_instance.Get_Mana()
        Mana_ref = 1.1717878855241677
        self.assertAlmostEqual(Mana_ref, Mana, places=7)
    
    
    
    