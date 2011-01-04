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
"""Unit test Library for the quantum_number module."""
from __future__ import division

import copy
import os
import sys

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import models.import_ufo as import_ufo
import decay.decay_objects as decay_objects
import tests.input_files.import_vertexlist as import_vertexlist
import quantum_number.quantum_number as quantum_number

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# QNumTest
#===============================================================================
class Test_QNumber_Add (unittest.TestCase):
    """Test for QNumber"""
    qrange = 1
    my_testmodel = import_ufo.import_model('sm')
    my_qnum = quantum_number.QNumber_Add(my_testmodel, qrange)
    charge = quantum_number.QNumber_Add(my_testmodel, 3)

    def setUp(self):
        #Setup charge from particle property 'charge' 
        #as an example of quantum number.

        for part in self.my_testmodel['particles']:
            self.charge[part.get_pdg_code()] = int(round(3 *part.get('charge')))


    def test_generate(self):
        """Test if the QNumber_Add is in the given range and the random property"""
        #Test if value are in [-qrange, qrange]
        for pid, qnum in self.my_qnum.items():
            self.assertTrue(qnum <= self.qrange)
            self.assertTrue(qnum >= -self.qrange)

        #print 'QNumber_Add before global change:', self.my_qnum
        my_qnum_old = copy.copy(self.my_qnum)
        self.my_qnum.generate_global()
        #print 'QNumber_Add after global change:', self.my_qnum
        self.assertFalse( my_qnum_old == self.my_qnum)

        #Test the local generation
        tquark_qnum = self.my_qnum[6]
        self.my_qnum.generate_local(6)
        self.assertFalse( self.my_qnum[6] == tquark_qnum)
        self.assertTrue( self.my_qnum[6] >= -self.qrange)
        self.assertTrue( self.my_qnum[6] <= self.qrange)

    def test_antiparticle(self):
        """Test if the anti_pdg_code gives inverse quantum_number"""
        
        #Test for quantum number for anti-particle
        self.assertEqual(self.my_qnum[-6], -self.my_qnum[6])
        self.assertEqual(self.my_qnum.get(-6), -self.my_qnum.get(6))
        self.assertRaises(KeyError, self.my_qnum.get, 100)
        self.assertRaises(KeyError, self.my_qnum.get, -50)

    def test_conservation_measure(self):
        """Test if the conservation returns correctly"""

        #Setup (full) lepton number for the test
        lepton_num = copy.copy(self.my_qnum)
        for i in range(1,7):
            lepton_num[i] = 0
        for i in range(21, 26):
            lepton_num[i] = 0
        for i in range(11, 17):
            lepton_num[i] = 1
        #lepton_num[12] = -1
        #print lepton_num
        #self.assertEqual(self.charge.conservation_measure(self.my_testmodel), 0)
        self.assertEqual(lepton_num.conservation_measure(self.my_testmodel), 0)

    def test_find_qnum_add(self):
        """Test the result of find_QNumber"""

        qnumber_list = quantum_number.find_QNumber_Add(self.my_testmodel)
        #print qnumlist
        self.assertFalse( len(qnumber_list) == 0)
        print '\n Finding ends here! \n'
        print '\n Total number of quantum number types:', len(qnumber_list)
        for num, qnumber in enumerate(qnumber_list):
            #print num, qnumber, qnum.conservation_measure(qnumber, self.my_testmodel)
            self.assertEqual(qnumber.conservation_measure(self.my_testmodel), 0)

    def test_find_qnum_parity(self):
        mssm = import_ufo.import_model('mssm')
        #Test the Monte Carlo find several times and see the percentage
        #that catch the correct quantum number
        iteration_times = 1
        total_num = []
        for i in range(1, iteration_times+1):
            parity_list = quantum_number.find_QNumber_Parity(mssm)
            total_num.append(len(parity_list))

        print '\n Number of quantum numbers found: \n ', total_num,\
            '\n Successful rate (for running %d times):' % iteration_times, \
            ((sum(total_num)-iteration_times)/iteration_times)

        parity = quantum_number.QNumber_Parity(mssm)
        #print parity, parity[6], parity[-6]
        self.assertEqual(parity[24], parity[-24])

        self.assertFalse(len(parity_list) == 0)
        print '\n Parity finding ends here! \n'
        print '\n Total number of parity quantum number types:',\
            len(parity_list)
        #self.assertEqual(quantum_number.reduced_interactions[0], 
        for num, qnumber in enumerate(parity_list):
            self.assertEqual(qnumber.conservation_measure(mssm), 0)

    def test_find_qnum_parity_modified_mssm(self):
        mssm = import_ufo.import_model('mssm')
        particles = mssm.get('particles')
        no_want_particle_codes = [1000022, 1000023, 1000024, -1000024, 
                                  1000025, 1000035, 1000037, -1000037]
        no_want_particles = [p for p in particles if p.get('pdg_code') in \
                                 no_want_particle_codes]

        for particle in no_want_particles:
            particles.remove(particle)

        interactions = mssm.get('interactions')
        inter_list = copy.copy(interactions)
        for interaction in inter_list:
            if any([p.get('pdg_code') in no_want_particle_codes for p in \
                        interaction.get('particles')]):
                interactions.remove(interaction)
        
        mssm.set('particles', particles)
        mssm.set('interactions', interactions)
        
        #Test the Monte Carlo find several times and see the percentage
        #that catch the correct quantum number
        iteration_times = 1
        total_num = []
        for i in range(1, iteration_times+1):
            parity_list = quantum_number.find_QNumber_Parity(mssm)
            total_num.append(len(parity_list))

        print '\n Number of quantum numbers found: \n ', total_num,\
            '\n Successful rate (for running %d times):' % iteration_times, \
            ((sum(total_num)-iteration_times)/iteration_times)

        parity = quantum_number.QNumber_Parity(mssm)
        #print parity, parity[6], parity[-6]
        self.assertEqual(parity[24], parity[-24])

        self.assertFalse(len(parity_list) == 0)
        print '\n Parity finding ends here! \n'
        print '\n Total number of parity quantum number types:',\
            len(parity_list)
        #self.assertEqual(quantum_number.reduced_interactions[0], 
        for num, qnumber in enumerate(parity_list):
            self.assertEqual(qnumber.conservation_measure(mssm), 0)

    def test_find_qnum_parity_old(self):
        mssm = import_ufo.import_model('mssm')
        #Test the Monte Carlo find several times and see the percentage
        #that catch the correct quantum number
        iteration_times = 1
        total_num = []
        for i in range(1, iteration_times+1):
            parity_list = quantum_number.find_QNumber_Parity_old(mssm)
            total_num.append(len(parity_list))

        print '\n Number of quantum numbers found: \n ', total_num,\
            '\n Successful rate (for running %d times):' % iteration_times, \
            ((sum(total_num)-iteration_times)/iteration_times)

        self.assertFalse(len(parity_list) == 0)
        print '\n Parity finding ends here! \n'
        print '\n Total number of parity quantum number types:',\
            len(parity_list)
        #self.assertEqual(quantum_number.reduced_interactions[0], 
        for num, qnumber in enumerate(parity_list):
            self.assertEqual(qnumber.conservation_measure(mssm), 0)
