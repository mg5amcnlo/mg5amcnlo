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
""" A test suite to compare the current version of MG5 with a version of 
reference. The model is taken from the reference version. So the only 
constraints to this test is that the current version can still read the old 
model format. 
The reference version is given here as a argument which can be changed by hand.
"""
import itertools
import logging
import os
import shutil
import me_comparator
import madevent_comparator
import unittest
import subprocess

from madgraph import MG5DIR

pjoin = os.path.join
_file_path = os.path.dirname(os.path.realpath(__file__))
_pickle_path = pjoin(_file_path, 'input_files')


class OLDMG5Comparator(unittest.TestCase):
    """A class to compare the value of a old MG5 version and the current one"""
    
    old_mg5 = None # link to the previous version of MG5 (prevent multiple build)
    reference_number = 350 #2.0.0
    nb_test = 0
    
    
    @classmethod
    def build_old_mg5(cls):
        """build the directory for mg5 run in the old template"""
        if cls.old_mg5:
            return cls.old_mg5
        print 'create new parralel test'
        init_dir = os.getcwd()
        os.chdir(MG5DIR)    
        # 1. bzr branch the present directory to a new directory
        #    parralel_tests_mg5

        filepath = 'parralel_tests_mg5'
        if os.path.exists(filepath):
            logging.info("Removing existing directory " + filepath)
            shutil.rmtree(filepath)
        logging.info("Branching " + MG5DIR + " to directory " + filepath)
        devnull = open(os.devnull,'w')   
        status = subprocess.call(['bzr', 'branch', MG5DIR, filepath], stdout=devnull, stderr=devnull)   
        if status:
            raise Exception, 'Impossible to configure old run'
        
        #2. reverse the directory to its old status
        os.chdir(filepath)
        status = subprocess.call(['bzr', 'revert', '-r', str(cls.reference_number)], stdout=devnull, stderr=devnull)
        if status:
            raise Exception, 'Impossible to configure old run'
        
        cls.old_mg5 = pjoin(MG5DIR,filepath)
        os.chdir(init_dir)
        return cls.old_mg5 

    def compare_processes(self, my_proc_list = [], orders = {}, model = 'sm',
                        energy = 500, filename = "", pickle_file = "",
                        tolerance = 1e-06):
        """ """
        
        mg5_path = self.build_old_mg5()
        
        if 'v4' in model:
            old_mg5 = me_comparator.MG5OldRunner()
            old_mg5.setup(mg5_path)
            current_mg5 = me_comparator.MG5Runner()
            current_mg5.setup(MG5DIR, MG5DIR)
            current_mg5.store_proc_card = True
        else:
            old_mg5 = me_comparator.MG5_UFO_OldRunner()
            old_mg5.setup(mg5_path)
            current_mg5 = me_comparator.MG5_UFO_Runner()
            current_mg5.setup(MG5DIR,MG5DIR)
            current_mg5.store_proc_card = True
        
        if os.path.exists(pjoin(MG5DIR,'models','paralel_test_model_%s' % model)):
            shutil.rmtree(pjoin(MG5DIR,'models','paralel_test_model_%s' % model))
        os.system('cp -rf %s %s' % (pjoin(mg5_path,'models',model) ,
                                    pjoin(MG5DIR,'models','paralel_test_model_%s' % model)))

        
        # Create and setup a comparator
        my_comp = me_comparator.MEComparator()
        my_comp.set_me_runners(current_mg5, old_mg5)

        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                               ['paralel_test_model_%s' % model,  model], orders, energy)

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
        mg5_path = self.build_old_mg5()
        
        if 'v4' in model:
            raise Exception, 'Not implemented'
            #old_mg5 = me_comparator.MG5OldRunner()
            #old_mg5.setup(mg5_path)
            #current_mg5 = me_comparator.MG5Runner()
            #current_mg5.setup(MG5DIR, MG5DIR)
            #current_mg5.store_proc_card = True
        else:
            old_mg5 = madevent_comparator.MG5OldRunner()
            old_mg5.setup(mg5_path)
            current_mg5 = madevent_comparator.MG5Runner()
            current_mg5.setup(MG5DIR)
            current_mg5.store_proc_card = True
        
        self.nb_test +=1      
        if os.path.exists(pjoin(MG5DIR,'models','paralel_test_model_%s' % model)):
            shutil.rmtree(pjoin(MG5DIR,'models','paralel_test_model_%s' % model))
        os.system('cp -rf %s %s' % (pjoin(mg5_path,'models',model) ,
                                    pjoin(MG5DIR,'models','paralel_test_model_%s' % model)))
        
        # Create and setup a comparator
        my_comp = madevent_comparator.MadEventComparator()
        my_comp.set_me_runners(old_mg5,current_mg5)

        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                               ['paralel_test_model_%s' % model, model], orders)

        # Print the output
        if filename:
            my_comp.output_result(filename=filename)
        
                # Store output to a pickle file in the input_files directory
        if print_result:
            print my_comp.results[0]
        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)
            
        # Do some cleanup
        my_comp.cleanup()
        return my_comp.results
       
    def compare_cross_section_to_values( self, values, my_proc_list = [], 
                        orders = {}, model = 'sm',
                        filename = "", print_result = False,
                        tolerance = 1e-02):   
                
        if 'v4' in model:
            raise Exception, 'Not implemented'
            #old_mg5 = me_comparator.MG5OldRunner()
            #old_mg5.setup(mg5_path)
            #current_mg5 = me_comparator.MG5Runner()
            #current_mg5.setup(MG5DIR, MG5DIR)
            #current_mg5.store_proc_card = True
        else:
            current_mg5 = madevent_comparator.MG5Runner()
            current_mg5.setup(MG5DIR)
            current_mg5.store_proc_card = True
        

        # Create and setup a comparator
        my_comp = madevent_comparator.MadEventComparator()
        my_comp.set_me_runners(current_mg5)

        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                               [model], orders)

        # add the default value to the comparison
        my_comp.results.append(values)
        my_comp.me_runners =(my_comp.me_runners[0], madevent_comparator.FakeRunner())
        
        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)
            
        # Do some cleanup
        my_comp.cleanup()
    
    ############################################################################    
    #  ROUTINE FOR CREATING THE SHORT TEST (USE by the release script)
    ############################################################################    
    def test_create_all_pickle(self):
        """re-create all the pickle for the short test (those call in the release).
           Note that if you need to redo this, this is potentially due to a change
           in the model. In consequence, you need to change the old MG5 comparison
           point. (Since the default use another model)."""

#        return # By default no need this
        self.create_short_parallel_sqso()
        self.create_short_paralel_sm()
        self.create_short_paralel_mssm()
        self.create_short_paralel_heft()
        

    def create_short_paralel_sm(self):
        """Test a short list of sm processes"""
        # Create a list of processes to check automatically
        my_proc_list = ['e+ e- > e+ e-', 
                        'h h > h h', 
                        'g g > t t~',
                        'w+ w- > w+ w-',
                        'b b~ > t t~',
                        'u u~ > z u u~',
                        'g g > t t~ h',
                        'u u~ > d d~ w+ w-']
        
        # Store list of non-zero processes and results in file
        pickle_file = os.path.join(_pickle_path, "mg5_short_paralleltest_sm.pkl")
        self.compare_processes(my_proc_list,
                               model='sm',
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_sm.log",
                             pickle_file = pickle_file)

    def create_short_paralel_mssm(self):
        """Test a short list of mssm processes"""
        # Create a list of processes to check automatically
        my_proc_list = [' g g > go go', 
                        'u u~ > go go', 
                        'e+ e- > n1 n2',
                        'd d~ > el+ el-',
                        'b b~ > h1 h2',
                        'u u~ > z u u~',
                        'e+ e- > n1 n2 z',
                        'd d~ > x1+ x1- g',
                        'h1 h1 > x1+ x1- h2']
        
        # Store list of non-zero processes and results in file
        pickle_file = os.path.join(_pickle_path, "mg5_short_paralleltest_mssm.pkl")
        self.compare_processes(my_proc_list,
                             model='mssm',
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_mssm.log",
                             pickle_file = pickle_file)

    def create_short_paralel_heft(self):
        """Test a short list of mssm processes"""
        # Create a list of processes to check automatically
        my_proc_list = ['h h > w+ w-', 
                        'g g > h g', 
                        'g g > h g g',
                        'g g > h g g g']
        
        # Store list of non-zero processes and results in file
        pickle_file = os.path.join(_pickle_path, "mg5_short_paralleltest_heft.pkl")
        self.compare_processes(my_proc_list,
                             model='heft',
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_heft.log",
                             pickle_file = pickle_file)


    def create_short_parallel_sqso(self):
        """Test a short list of processes with squared order constraints"""
        # Create a list of processes to check automatically
        my_proc_list = ['u u~ > d d~', 
                        'u u~ > d d~ g', 
                        'u u~ > d d~ a',
                        'u u~ > d d~ c c~']
        
        # Store list of non-zero processes and results in file
        pickle_file = os.path.join(_pickle_path, 
                                              "mg5_short_paralleltest_sqso.pkl")
        self.compare_processes(my_proc_list,
                             model='sm',
                             orders = {'WEIGHTED^2<=':-2},
                             filename = "short_sqso.log",
                             pickle_file = pickle_file)


    ############################################################################    
    #  ROUTINE FOR THE SHORT TEST (USE by the release script)
    ############################################################################
    def test_short_sm(self):
        """Test a minimal list of sm 2->2 processes, mainly to test the test"""
        # Create a list of processes to check automatically
        comparisons = me_comparator.PickleRunner.find_comparisons(\
            os.path.join(_pickle_path, "mg5_short_paralleltest_sm.pkl"))
        for stored_runner in comparisons:
            # Create a MERunner object for MG5
            my_mg5 = me_comparator.MG5_UFO_Runner()
            my_mg5.setup(MG5DIR, MG5DIR)

            # Create and setup a comparator
            my_comp = me_comparator.MEComparator()
            my_comp.set_me_runners(stored_runner, my_mg5)

            # Run the actual comparison
            my_comp.run_comparison(stored_runner.proc_list,'sm',
                                   stored_runner.orders,
                                   stored_runner.energy)

            my_comp.assert_processes(self)

            # Do some cleanup
            my_comp.cleanup()

    def test_short_sqso(self): 
        """Test a short list of processes with squared order constraints"""

        comparisons = me_comparator.PickleRunner.find_comparisons(\
            os.path.join(_pickle_path, "mg5_short_paralleltest_sqso.pkl"))

        for stored_runner in comparisons:
            # Create a MERunner object for MG5
            my_mg5 = me_comparator.MG5_UFO_Runner()
            my_mg5.setup(MG5DIR, MG5DIR)

            # Create and setup a comparator
            my_comp = me_comparator.MEComparator()
            my_comp.set_me_runners(stored_runner, my_mg5)

            # Run the actual comparison
            my_comp.run_comparison(stored_runner.proc_list,'sm',
                                   stored_runner.orders,
                                   stored_runner.energy)

            my_comp.assert_processes(self)

            # Do some cleanup
            my_comp.cleanup()
            
    def test_short_mssm(self):
        """Test a minimal list of sm 2->2 processes, mainly to test the test"""
        # Create a list of processes to check automatically
        comparisons = me_comparator.PickleRunner.find_comparisons(\
            os.path.join(_pickle_path, "mg5_short_paralleltest_mssm.pkl"))
        for stored_runner in comparisons:
            # Create a MERunner object for MG5
            my_mg5 = me_comparator.MG5_UFO_Runner()
            my_mg5.setup(MG5DIR, MG5DIR)

            # Create and setup a comparator
            my_comp = me_comparator.MEComparator()
            my_comp.set_me_runners(stored_runner, my_mg5)

            # Run the actual comparison
            my_comp.run_comparison(stored_runner.proc_list,
                                   'mssm',
                                   stored_runner.orders,
                                   stored_runner.energy)

            my_comp.assert_processes(self)

            # Do some cleanup
            my_comp.cleanup()           

    def test_short_heft(self):
        """Test a minimal list of sm 2->2 processes, mainly to test the test"""
        # Create a list of processes to check automatically
        comparisons = me_comparator.PickleRunner.find_comparisons(\
            os.path.join(_pickle_path, "mg5_short_paralleltest_heft.pkl"))
        for stored_runner in comparisons:
            # Create a MERunner object for MG5
            my_mg5 = me_comparator.MG5_UFO_Runner()
            my_mg5.setup(MG5DIR, MG5DIR)

            # Create and setup a comparator
            my_comp = me_comparator.MEComparator()
            my_comp.set_me_runners(stored_runner, my_mg5)

            # Run the actual comparison
            my_comp.run_comparison(stored_runner.proc_list,
                                   'heft',
                                   stored_runner.orders,
                                   stored_runner.energy)

            my_comp.assert_processes(self)

            # Do some cleanup
            my_comp.cleanup()   

    
    ############################################################################
    # Short test for the evaluation of the cross-section
    ############################################################################
    def test_short_cross_sm1(self):
        """Test a short list of sm processes"""
        # Create a list of processes to check automatically                                                                                                                             
        my_proc_list = ['p p > t t~']
        values = {'number_of_P0': '2', 
                  'cross_P0_qq_ttx': '0.65258E+02', 
                  'cross_P0_gg_ttx': '0.43817E+03'}

        # Store list of non-zero processes and results in file                                                                                                                          
        self.compare_cross_section_to_values(values, my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_cs_sm1.log")

    def test_short_cross_sqso1(self):
        """Test a process with definite squared order constraints. In this case
        only the QCD-QED interference."""
        # Create a list of processes to check automatically                                                                                                                             
        my_proc_list = ['p p > j j']
        values = {'number_of_P0': '1',
                  'cross_P0_qq_qq': '0.20466E+06'}

        # Store list of non-zero processes and results in file                                                                                                                          
        self.compare_cross_section_to_values(values, my_proc_list,
                             orders = {'QED^2==':2, 'QCD^2==':2},
                             filename = "short_cs_sqso1.log")
 
    def test_short_cross_sm2(self):
        """Test a short list of sm processes""" 
        my_proc_list = ['u j > W+ g', 'g g > W+ j j']

        values = {'number_of_P0': '1', 
         'number_of_P1': '1', 
         'cross_P0_qq_wpg': '0.25882E+04', 
         'cross_P1_gg_wpqq': '0.39004E+03'}      
        self.compare_cross_section_to_values(values, my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_cs_sm2.log")

    def test_short_cross_sm3(self):
        """Test a short list of sm processes""" 
        my_proc_list = ['g g > t t~, (t > b W+, W+ > e+ ve)']

        values =  {'number_of_P0': '1', 
                   'cross_P0_gg_ttx_t_bwp_wp_lvl': '0.41434E+02'} 
                  
        self.compare_cross_section_to_values(values, my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_cs_sm3.log")

    def test_short_cross_mssm1(self):
        """Test a short list of sm processes""" 
        my_proc_list = ['g g > go go']

        values = {'number_of_P0': '1', 'cross_P0_gg_gogo': '0.46066E+01'}
        
        self.compare_cross_section_to_values(values, my_proc_list,
                             model='mssm',
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_cs_sm3.log")        
        
    ############################################################################    
    #  ROUTINE FOR CHECKING THE PARRALEL TEST
    ############################################################################           
    def test_mg5_minitest_sm(self):
        """Test a minimal list of sm 2->2 processes, mainly to test the test"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list(\
            ['u'],
            initial=2, final=2)
        my_proc_list = ['e+ e- > a > e+ e-', 'h h > h h', 'e+ e+ > e- e-', 
                        'u w+ > u w+']
        # Store list of non-zero processes and results in file
        #pickle_file = "mg4_sm_%sminitest.pkl" % self.suffix_name
        self.compare_processes(my_proc_list, model='sm',
                             orders = {'QED':2, 'QCD':2},
                             filename = "sm_mini.log",
                             energy = 1000)
        
    def test_mg5_minitest_mssm(self):
        """Test a minimal list of sm 2->2 processes, mainly to test the test"""
        # Create a list of processes to check automatically
        my_proc_list = ['g g > go go', 'e+ e-  > n1 n2', 'g t~ > go t1~']
        # Store list of non-zero processes and results in file
        #pickle_file = "mg4_sm_%sminitest.pkl" % self.suffix_name
        self.compare_processes(my_proc_list, model='mssm',
                             orders = {'QED':2, 'QCD':2},
                             filename = "mssm_mini.log",
                             energy = 2000)

    ############################################################################    
    #  EXTENSIVE TEST FOR THE SM
    ############################################################################ 
    def test_mg5_sm_22(self):
        """Test a semi-complete list of sm 2->2 processes"""
        # Create a list of processes to check automatically
        #my_proc_list = me_comparator.create_proc_list(\
        #    ['w+', 'w-', 'a', 'z', 'h', 'g', 'u', 'u~', 'd', 'd~',
        #    'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~'],
        #    initial=2, final=2)
        my_proc_list = me_comparator.create_proc_list(\
            ['w+', 'w-', 'a', 'z', 'h', 'g', 'u', 'u~', 'd', 'd~',
            'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~'],
            initial=2, final=2)
        #my_proc_list = ['e+ e- > e+ e-','e+ e- > e+ e- a']
        
        # Store list of non-zero processes and results in file
        for i in range(len(my_proc_list)//500):
            print 'step %s/%s' %(i+1,len(my_proc_list)//500 )
        # Store list of non-zero processes and results in file
            self.compare_processes(my_proc_list[500*i:500*(i+1)],
                             orders = {'QED':2, 'QCD':2},
                             model = "sm",
                             energy = 1000,
                             filename = "sm_22.log")   
            
    def test_mg5_sm_13(self):
        """Test a semi-complete list of sm 1->3 processes"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list_enhanced(
                          init_part_list = ['t','t~','ta+','ta-'], 
                          final_part_list_1 = ['b','b~','vt','vt~'],
                          final_part_list_2 = ['u', 'u~', 'd', 'd~', 'c','c~', 'e+', 've', 'e-', 've~','mu-','vm'], 
                          initial=1, final_1=1, final_2=2, charge_conservation=True)
        
        
        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':4, 'QCD':4},
                             model = "sm",
                             energy = 1000,
                             filename = "sm_13.log")   

    def test_mg5_sm_23_p1(self):
        """Test a semi-complete list of sm 2->3 processes"""
        # Create a list of processes to check automatically
        particles = ['w+', 'w-','a', 'z', 'h', 'g']
        
        def get_process(pos):
            proc = ''
            for i,ind in enumerate(pos):
                proc += ' '
                if i == 2:
                    proc += '> '
                proc += particles[ind]
            return proc
                        
        iter = itertools.product(range(len(particles)), repeat=5)
        for i in range(len(particles)**5//500):
            my_proc_list = []
            for j in range(0,500):
                try:
                    my_proc_list.append(get_process(iter.next()))
                except:
                    break
            print 'step %s/%s' %(i+1,len(particles)**5//500 )
        # Store list of non-zero processes and results in file
            self.compare_processes(my_proc_list,
                             orders = {'QED':3, 'QCD':3},
                             filename = "sm_23_p1.log")


    def test_mg5_sm_23_p2(self):
        """Test a semi-complete list of sm 2->3 processes"""
        # Create a list of processes to check automatically
        particles = ['u', 'u~', 'ta+', 'ta-', 'vt', 'vt~',
             'b', 'b~', 't', 't~']
        last_particles =  ['w+', 'w-','a', 'z', 'h', 'g']
        
        def get_process(pos, last):           
            proc = ''
            for i,ind in enumerate(pos):
                proc += ' '
                if i == 2:
                    proc += '> '
               
                proc += particles[ind]
            return proc + ' %s' % last
                        
        
        for i, last in enumerate(last_particles):
            iter = itertools.product(range(len(particles)), repeat=4)
            for j in range(len(particles)**4//500):
                my_proc_list = []
                for k in range(0,500):
                    try:
                        my_proc_list.append(get_process(iter.next(), last))
                    except:
                        break
                print 'step %s/%s' %(i*(len(particles)**4)//500+j+1, len(particles)**4//500 * len(last_particles))
        # Store list of non-zero processes and results in file
                self.compare_processes(my_proc_list,
                             orders = {'QED':3, 'QCD':3},
                             filename = "sm_23_p2.log")

    def test_mg5_sm_23_p3(self):
        """Test a semi-complete list of sm 2->3 processes"""
        # Create a list of processes to check automatically
        fermion = ['u', 'u~', 'd', 'd~',
             'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~']
        boson =  ['w+', 'w-','a', 'z', 'h', 'g']
        
        def get_process(pos_f, pos_b):           
            proc = ''
            for i,ind in enumerate(pos_f):
                proc += ' '
                proc += fermion[ind]
            proc += ' > '
            for i,ind in enumerate(pos_b):
                proc += ' '
                proc += boson[ind]            
            return proc 
        
        iter_f = itertools.product(range(len(fermion)), repeat=2)
        f_comb=-1
        for fermions in iter_f:
            f_comb+=1
            iter = itertools.product(range(len(boson)), repeat=3)
            if not f_comb % 2:
                print 'step %s/%s' % ((f_comb)//2 + 1, len(fermion)**2//2) 
                my_proc_list = []
            for j in range(len(boson)**3):
                my_proc_list.append( get_process(fermions, iter.next()))
        # Store list of non-zero processes and results in file
            if f_comb %2:
                self.compare_processes(my_proc_list,
                             orders = {'QED':3, 'QCD':3},
                             filename = "sm_23_p3.log")





    ############################################################################    
    #  EXTENSIVE TEST FOR THE MSSM
    ############################################################################ 
    def test_mg5_mssm_22(self):
        """Test a semi-complete list of mssm 2->2 processes"""
        # Create a list of processes to check automatically
        sm_parts = ['w+', 'w-', 'a', 'z', 'h1', 'h+', 'h-', 'g', 'u', 'u~',
            'd', 'd~', 'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~']
        mssm_parts = ['dl', 'dl~', 'dr', 'dr~', 'ul', 'ul~', 'ur', 'ur~', 'b1',
                      'b1~', 'b2', 'b2~', 't1', 't1~', 'ta1-', 'ta1+', 'ta2-',
                      'ta2+', 'svt', 'svt~', 'x1-', 'x1+', 'x2-', 'x2+',
                      'go', 'n1']
        # Generate 2 -> 2 processes, with MSSM particles in pairs in
        # final state
        my_proc_list = me_comparator.create_proc_list_enhanced(\
            sm_parts, mssm_parts)

        for i in range(len(my_proc_list)//500):
            print 'step %s/%s' %(i+1,len(my_proc_list)//500 )
        # Store list of non-zero processes and results in file
            self.compare_processes(my_proc_list[500*i:500*(i+1)],
                             orders = {'QED':2, 'QCD':2},
                             model = "mssm",
                             energy = 2000,
                             filename = "mssm_22.log")   

    def test_mg5_mssm_13(self):
        """Test a semi-complete list of mssm 1->3 processes"""
        # Create a list of processes to check automatically
        my_proc_list = me_comparator.create_proc_list_enhanced(
                          init_part_list = ['t1','t1~','t2', 't2~','b2','b2~'], 
                          final_part_list_1 = ['b1','b1~','b2','b2~','n1','n2'],
                          final_part_list_2 = ['u', 'u~', 'd', 'd~', 'c','c~', 'b', 'b~', 't','t~'], 
                          initial=1, final_1=1, final_2=2, charge_conservation=False)
        my_proc_list += ['t2 > b1 d~ u', 't2 > b2 d~ u', 't1 > b1 d~ u', 't1 > b2 d~ u',
                        't2~ > b1~ d u~', 't2~ > b2~ d u~', 't1~ > b1~ d u~', 't1~ > b2~ d u~']

        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':4, 'QCD':4},
                             model = "mssm",
                             energy = 2000,
                             filename = "mssm_13_%s.log") 


    def test_mg5_mssm_23_p1(self):
        """Test a semi-complete list of mssm 2->3 processes"""
        # Create a list of processes to check automatically
        
        init_part_list1 = ['w+', 'a', 'z', 'h1', 'h+', 'g', 'u~',
            'd~', 'b~', 't~', 'ta+', 'vt~']
        init_part_list2 = ['w-', 'a', 'z', 'h1', 'h-', 'g', 'u',
            'd', 'b', 't', 'ta+', 'vt']        
        final_part_list1 = ['w-', 'w+', 'a', 'z', 'h1', 'h-', 'h+', 'g'] 
        final_part_list2 = ['dl~', 'dr~', 'ul~', 'ur~']
        final_part_list3 = ['dl', 'dr', 'ul', 'ur']
        
        my_proc_list = me_comparator.create_proc_list_2_3(
                    init_part_list1, init_part_list2, final_part_list1,
                      final_part_list2,final_part_list3)                                   
                                                                  
        sm_parts = ['w+', 'w-', 'a', 'z', 'h1', 'h+', 'h-', 'g', 'u', 'u~',
            'd', 'd~', 'b', 'b~', 't', 't~', 'ta+', 'ta-', 'vt', 'vt~']
        mssm_parts = ['dl', 'dl~', 'dr', 'dr~', 'ul', 'ul~', 'ur', 'ur~', 'b1',
                      'b1~', 'b2', 'b2~', 't1', 't1~', 'ta1-', 'ta1+', 'ta2-',
                      'ta2+', 'svt', 'svt~', 'x1-', 'x1+', 'x2-', 'x2+',
                      'go', 'n1']
        # Generate 2 -> 2+1 processes, with MSSM particles in pairs in
        # final state
        #my_proc_list = me_comparator.create_proc_list_enhanced(\
        #    sm_parts, mssm_parts, sm_parts)
        for i in range(len(my_proc_list)//500):
            print 'step %s/%s' %(i+1,len(my_proc_list)//500 )
        # Store list of non-zero processes and results in file
            self.compare_processes(my_proc_list[500*i:500*(i+1)],
                             orders = {'QED':3, 'QCD':3},
                             model = "mssm",
                             energy = 2000,
                             filename = "mssm_23_p1.log")    

    def test_mg5_mssm_23_p2(self):
        """Test a semi-complete list of mssm 2->3 processes"""
        # Create a list of processes to check automatically
        
        init_part_list1 = ['w+', 'a', 'z', 'h1', 'h+', 'g', 'u~',
            'd~', 'b~', 't~', 'ta+', 'vt~']
        init_part_list2 = ['w-', 'a', 'z', 'h1', 'h-', 'g', 'u',
            'd', 'b', 't', 'ta+', 'vt']        
        final_part_list1 = ['w-', 'w+', 'a', 'z', 'h1', 'h-', 'h+', 'g'] 
        final_part_list2 = ['x1-', 'x1+', 'x2-', 'x2+', 'go', 'n1']
        final_part_list3 = ['x1-', 'x1+', 'x2-', 'x2+', 'go', 'n1']
        
        my_proc_list = me_comparator.create_proc_list_2_3(
                    init_part_list1, init_part_list2, final_part_list1,
                      final_part_list2,final_part_list3)

        for i in range(len(my_proc_list)//500):
            print 'step %s/%s' %(i+1,len(my_proc_list)//500 )
        # Store list of non-zero processes and results in file
            self.compare_processes(my_proc_list[500*i:500*(i+1)],
                             orders = {'QED':3, 'QCD':3},
                             model = "mssm",
                             energy = 2000,
                             filename = "mssm_23_p2.log")   

        
    ############################################################################    
    #  EXTENSIVE TEST FOR THE HEFT
    ############################################################################     
    def test_mg5_heft_23(self):
        """Test a heft 2->3 processes"""
        # Create a list of processes to check automatically
        sm_parts = ['g', 'a', 'w+', 'w-']
        heft_parts = ['h']
        # Generate 2 -> 1+2 processes, with one Higgs in final state
        # final state
        my_proc_list = me_comparator.create_proc_list_enhanced(\
            sm_parts, sm_parts, heft_parts)

        # Store list of non-zero processes and results in file
        self.compare_processes(my_proc_list,
                             orders = {'QED':2, 'QCD':0, 'HIG':1, 'HIW':1},
                             model = "heft",
                             energy = 500,
                             filename = "heft_23.log")

    ############################################################################
    # Short test for the evaluation of the cross-section
    ############################################################################
    def test_paralel_cross_sm(self):
        """Test a short list of sm processes"""
        # Create a list of processes to check automatically                                                                                                                             
        proc_lists = [['p p > t t~'], ['u d~ > W+ j', 'u d~ > W+ j j']]

        # Store list of non-zero processes and results in file                                                                                                                          
        pickle_file = os.path.join(_pickle_path, "mg5_short_parraleltest_cross_sm.pkl")
        for my_proc_list in proc_lists:
            print '.'
            self.compare_cross_section(my_proc_list,
                             orders = {'QED':99, 'QCD':99},
                             filename = "short_cs_sm.log")






        
