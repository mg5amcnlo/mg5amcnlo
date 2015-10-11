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
""" A test suite to compare the complex-mass scheme with the fixed-width scheme, 
and the Feynman gauge with the unitary gauge. The comparison between the four 
possibilities is performed since there is a strong relation between complex-mass
scheme and gauge invariance. 
"""
import itertools
import logging
import os
import shutil
import me_comparator
import madevent_comparator
import unittest
import subprocess
import loop_me_comparator

from madgraph import MG5DIR
import tests.parallel_tests.test_aloha as test_aloha

pjoin = os.path.join
_file_path = os.path.dirname(os.path.realpath(__file__))
_pickle_path = pjoin(_file_path, 'input_files')


class GaugeComparator(unittest.TestCase):
    """A class to compare the value of a old MG5 version and the current one"""
    
    nb_test = 0
    
    def compare_processes(self, my_proc_list = [], orders = {}, model = 'sm',
                        energy = 500, filename = "", pickle_file = "",
                        tolerance = 1e-06):
        """ """
        
        cmsunit_runner = me_comparator.MG5_UFO_gauge_Runner(cms='True',gauge='unitary')
        cmsunit_runner.setup(MG5DIR, MG5DIR)
            
        mg5unit_runner = me_comparator.MG5_UFO_gauge_Runner(cms='False',gauge='unitary')
        mg5unit_runner.setup(MG5DIR, MG5DIR)
        
        cmsfeynman_runner = me_comparator.MG5_UFO_gauge_Runner(cms='True',gauge='Feynman')
        cmsfeynman_runner.setup(MG5DIR, MG5DIR)
        
        mg5feynman_runner = me_comparator.MG5_UFO_gauge_Runner(cms='False',gauge='Feynman')
        mg5feynman_runner.setup(MG5DIR, MG5DIR)
        
        # ADD FOR Feynmam and CMS + Feynman
                
        # Create and setup a comparator
        my_comp = me_comparator.MEComparatorGauge()
        my_comp.set_me_runners(cmsunit_runner, cmsfeynman_runner,mg5feynman_runner,mg5unit_runner) # can add Feynman+...
        

        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                               model, orders, energy)

        # Print the output
        if filename:
            my_comp.output_result(filename=filename)
        
        # Store output to a pickle file in the input_files directory
        if pickle_file:
            me_comparator.PickleRunner.store_comparison(\
                os.path.join(_pickle_path, pickle_file),
                my_comp.get_non_zero_processes(),
                my_comp.me_runners[0].model,
                my_comp.me_runners[0].name,
                my_comp.me_runners[0].orders,
                my_comp.me_runners[0].energy)

        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)
            
        # Do some cleanup
        my_comp.cleanup()
       
    def compare_cross_section(self, my_proc_list = [], orders = {}, model = 'sm',
                        filename = "", print_result = False,
                        tolerance = 1e-02):
        """ """
              
        cmsunit_runner = madevent_comparator.MG5gaugeRunner(cms='True',gauge='unitary')
        cmsunit_runner.setup(MG5DIR)
        mg5unit_runner = madevent_comparator.MG5gaugeRunner(cms='False',gauge='unitary')
        mg5unit_runner.setup(MG5DIR)
        cmsfeyn_runner = madevent_comparator.MG5gaugeRunner(cms='True',gauge='Feynman')
        cmsfeyn_runner.setup(MG5DIR)
        mg5feyn_runner = madevent_comparator.MG5gaugeRunner(cms='False',gauge='Feynman')
        mg5feyn_runner.setup(MG5DIR)
        
        #self.nb_test +=1      
        #if os.path.exists(pjoin(MG5DIR,'models','paralel_test_model_%s' % model)):
        #    shutil.rmtree(pjoin(MG5DIR,'models','paralel_test_model_%s' % model))
        #os.system('cp -rf %s %s' % (pjoin(MG5DIR,'models',model) ,
        #                            pjoin(MG5DIR,'models','paralel_test_model_%s' % model)))
        
        # Create and setup a comparator
        my_comp = madevent_comparator.MadEventComparatorGauge()
        my_comp.set_me_runners(cmsunit_runner, mg5unit_runner, cmsfeyn_runner, mg5feyn_runner)

        # Run the actual comparison
        #my_comp.run_comparison(my_proc_list,
        #                       ['paralel_test_model_%s' % model, model], orders)
        my_comp.run_comparison(my_proc_list,model, orders)
        
        # Print the output
        if filename:
            my_comp.output_result(filename=filename)
        
        # Store output to a pickle file in the input_files directory
        #if print_result:
        #   print my_comp.results[0]

        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)
            
        # Do some cleanup
        my_comp.cleanup()
        return my_comp.results
            
    ############################################################################
    # Short test for the evaluation of the cross-section
    ############################################################################
    def test_short_cross_gauge(self):
        """Test the cross section of a short list of sm processes"""
        # Create a list of processes to check automatically                                                                                                                             
        my_proc_list = ['p p > d d~']        
        #my_proc_list = ['u u~ > c c~', 'e+ e- > u u~']
        # Store list of non-zero processes and results in file                                                                                                                          
        self.compare_cross_section(my_proc_list,
                             orders = {'QED':99, 'QCD':99},model = 'sm',
                             filename = "short_cs_sm_gauge.log")

    @test_aloha.set_global(loop=False, unitary=False, mp=False, cms=False)
    def test_short_gauge(self):
        """Test a semi-complete list of sm 2->4 processes"""
        # Create a list of processes to check automatically       
        my_proc_list = ['e+ e- > u u~ d d~','u u~ > u u~ d d~', 'c s~ > u u~ c s~',
             'e+ e- > ta+ ta- vta vta~','b b~ > b b~']

        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             model = "sm",
                             energy = 90,
                             filename = "sm_gauge_short_e90.log",
                             tolerance = 1e-3)
        
    ############################################################################    
    #  EXTENSIVE GAUGE TEST FOR THE SM
    ############################################################################ 
    def test_cross_gauge_p1(self):
        """Test the cross section of a short list of sm processes"""
        # Create a list of processes to check automatically                                                                                                                             
        my_proc_list = ['d d~ > u d~ s c~']        
        # Store list of non-zero processes and results in file                                                                                                                          
        self.compare_cross_section(my_proc_list,
                             orders = {'QED':99, 'QCD':99},model = 'sm',
                             filename = "short_cs_sm_gauge_p1.log")

    def test_cross_gauge_p2(self):
        """Test the cross section of a short list of sm processes"""
        # Create a list of processes to check automatically        
        my_proc_list = ['p p > b b~ e- ve~ mu+ vm']
        # Store list of non-zero processes and results in file                                                                                                                          
        self.compare_cross_section(my_proc_list,
                             orders = {'QED':99, 'QCD':99},model = 'sm',
                             filename = "short_cs_sm_gauge_p2.log")
    
    def test_gauge_2(self):
        """Test a semi-complete list of sm 2->2 processes"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list(\
            ['a', 'g', 'u', 'u~', 'd', 'd~',
            'b', 'b~', 'ta+', 'ta-', 'vt', 'vt~','ve','ve~'],
            initial=2, final=2)
        
        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             model = "sm",
                             energy = 90,
                             filename = "sm_gauge_2_e90.log",
                             tolerance = 1e-3)   
        # Do the test for high energy
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             model = "sm",
                             energy = 500,
                             filename = "sm_gauge_2_e500.log",
                             tolerance = 1e-3)   
        
    def test_gauge_3(self):
        """Test a semi-complete list of sm 2->3 processes"""        
        # Create a list of processes to check automatically
        my_proc_list = ['u u~ > u u~ g', 'u u~ > d d~ g', 'u u~ > g g g',
                        'u d~ > u d~ g', 'u g > u u~ u', 'u u~ > b b~ g',
                        'u u > u u g', 'b g > b u u~', 'b b~ > b b~ g']

        # Store list of non-zero processes and results in file 
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             model = "sm",
                             energy = 90,
                             filename = "sm_gauge_3_e90.log",
                             tolerance = 1e-3)   

        # Do the test for high energy 
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             model = "sm",
                             energy = 500,
                             filename = "sm_gauge_3_e500.log",
                             tolerance = 1e-3)   
    
    def test_gauge_4_e500(self):
        """Test a semi-complete list of sm 2->4 processes"""
        # Create a list of processes to check automatically       
        my_proc_list = ['e+ e- > e+ ve d u~','u u~ > u u~ c c~', 
            'c c~ > u d~ s c~', 'c s~ > ta+ vta vta vta~']

        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             model = "sm",
                             energy = 500,
                             filename = "sm_gauge_4_e500.log",
                             tolerance = 1e-3)

    def test_gauge_6_e90(self):
        """Test a semi-complete list of sm 2->4 processes"""
        # Create a list of processes to check automatically       
        my_proc_list = ['g g > b b~ e+ ve mu- vm~','u u~ > b b~ e+ ve mu- vm~',
              'u u~ > b b~ u d~ mu- vm~']

        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             model = "sm",
                             energy = 90,
                             filename = "sm_gauge_6_e90.log",
                             tolerance = 1e-3)
        
    def test_gauge_6_e500(self):
        """Test a semi-complete list of sm 2->4 processes"""
        # Create a list of processes to check automatically       
#        my_proc_list = ['g g > b b~ u u~ d d~','u u~ > b b~ e+ ve mu- vm~',
#              'u u~ > g g u d~ mu- vm~']
        my_proc_list = ['u u~ > b b~ e+ ve mu- vm~']

        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':99, 'QCD':0},
                             model = "sm",
                             energy = 500,
                             filename = "sm_gauge_6_e500_2.log",
                             tolerance = 1e-3)          

class GaugeComparatorLoop(unittest.TestCase):
    """A class to compare the values of unitary and Feynman gauge in complex and fixed width schemes"""
    
    nb_test = 0
    
    def compare_processes(self, my_proc_list = [], model = 'loop_sm',
                        energy = 500, filename = "", pickle_file = "",
                        tolerance = 1e-06):
        """ """
        
        cmsunit_runner = loop_me_comparator.LoopMG5Runner_gauge(cms='True',gauge='unitary')
        cmsunit_runner.setup(MG5DIR, MG5DIR)
            
        mg5unit_runner = loop_me_comparator.LoopMG5Runner_gauge(cms='False',gauge='unitary')
        mg5unit_runner.setup(MG5DIR, MG5DIR)
        
        cmsfeynman_runner = loop_me_comparator.LoopMG5Runner_gauge(cms='True',gauge='Feynman')
        cmsfeynman_runner.setup(MG5DIR, MG5DIR)
        
        mg5feynman_runner = loop_me_comparator.LoopMG5Runner_gauge(cms='False',gauge='Feynman')
        mg5feynman_runner.setup(MG5DIR, MG5DIR)
        
        # ADD FOR Feynmam and CMS + Feynman
                
        # Create and setup a comparator
        my_comp = loop_me_comparator.LoopMEComparatorGauge()
        my_comp.set_me_runners(cmsunit_runner, cmsfeynman_runner,mg5feynman_runner,mg5unit_runner)
        

        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                               model, energy)

        # Print the output
        if filename:
            my_comp.output_result(filename=filename)
        
                # Store output to a pickle file in the input_files directory
        if pickle_file:
            me_comparator.PickleRunner.store_comparison(\
                os.path.join(_pickle_path, pickle_file),
                my_comp.get_non_zero_processes(),
                my_comp.me_runners[0].model,
                my_comp.me_runners[0].name,
                my_comp.me_runners[0].orders,
                my_comp.me_runners[0].energy)

        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)
            
        # Do some cleanup
        my_comp.cleanup()
       
    @test_aloha.set_global(loop=False, unitary=False, mp=False, cms=False)
    def test_short_gauge_loop(self):
        """Test a semi-complete list of sm 2->4 processes"""
        # Create a list of processes to check automatically       
        #my_proc_list = ['e+ e- > u u~ d d~','u u~ > u u~ d d~', 'c s~ > u u~ c s~',
        #     'e+ e- > ta+ ta- vta vta~','b b~ > b b~']
        
        my_proc_list = [('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4})]
        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             model = "loop_sm",
                             energy = 90,
                             filename = "loop_gauge_e90.log",
                             tolerance = 1e-3)
        
    ############################################################################    
    #  EXTENSIVE GAUGE TEST FOR THE SM
    ############################################################################ 

    def test_gauge_loop_p1(self):
        """Test a semi-complete list of sm 2->2 processes"""
        # Create a list of processes to check automatically
        my_proc_list = []
        my_proc_list.append(('u u~ > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
        my_proc_list.append(('u u~ > d d~',{'QCD':0,'QED':2},['QCD'],{'QCD':2,'QED':4}))
        my_proc_list.append(('b u > b d e+ ve',{'QCD':0,'QED':4},['QCD'],{'QCD':2,'QED':8}))
        #my_proc_list.append(('d g > d g',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
        #my_proc_list.append(('g g > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
        #my_proc_list.append(('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4}))
        #my_proc_list.append(('d~ d > g a',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}))
        
        
        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             model = "loop_sm",
                             energy = 90,
                             filename = "loop_gauge_e90_p1.log",
                             tolerance = 1e-3)   
        # Do the test for high energy
        self.compare_processes(my_proc_list,
                             model = "loop_sm",
                             energy = 500,
                             filename = "loop_gauge_e500_p1.log",
                             tolerance = 1e-3)   
        
    def test_gauge_loop_p2(self):
        """Test a semi-complete list of sm 2->2 processes"""
        # Create a list of processes to check automatically
        my_proc_list = []
        my_proc_list.append(('u u~ > b b~ e+ ve mu- vm~',{'QED':4,'QCD':2},['QCD'],{'QCD':6,'QED':8}))
        #my_proc_list.append(('g g > b b~ e+ ve mu- vm~',{'QED':4,'QCD':2},['QCD'],{'QCD':6,'QED':8}))
        #my_proc_list.append(('e+ e- > b b~ e+ ve mu- vm~',{'QED':6,'QCD':0},['QCD'],{'QCD':2,'QED':12}))
        #my_proc_list.append(('g g > b b~ e+ ve mu- vm~',{'QED':6,'QCD':0},['QCD'],{'QCD':2,'QED':12}))
        my_proc_list.append(('d d~ > d~ u u~ d',{'QED':4,'QCD':0},['QCD'],{'QCD':2,'QED':8}))        
        #my_proc_list.append(('g g > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}))
        #my_proc_list.append(('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4}))
        #my_proc_list.append(('d~ d > g a',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}))
        #my_proc_list.append(('d d~ > d~ u u~ d',{'QED':4,'QCD':0},['QCD'],{'QCD':2,'QED':8}))
        
        
        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             model = "loop_sm",
                             energy = 90,
                             filename = "loop_gauge_e90_p2.log",
                             tolerance = 1e-3)   
        # Do the test for high energy
        self.compare_processes(my_proc_list,
                             model = "loop_sm",
                             energy = 500,
                             filename = "loop_gauge_e500_p2.log",
                             tolerance = 1e-3)   

