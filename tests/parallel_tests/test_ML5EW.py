import logging.config
import logging
import pydoc
import os
import unittest
import sys
import re
#Look for MG5/MG4 path
_mg5_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
sys.path.append(_mg5_path)
_file_path = os.path.dirname(os.path.realpath(__file__))
_pickle_path = os.path.join(_file_path, 'input_files', 'ML_parallel_saved_runs')

from madgraph import MG5DIR
from madgraph import MadGraph5Error
from madgraph.iolibs.files import cp
#import madgraph.iolibs.save_load_object as save_load_object
import loop_me_comparator
import madgraph.various.banner as banner_mod
import madgraph.various.misc as misc
import me_comparator
from test_ML5 import procToFolderName
# The processes below are treated all together because they are relatively quick

HCR_processes_short = []

ML5EW_processes_short = []

ML5EW_processes_short_sqso =  [
   ('u u~ > d d~',{},['QCD QED'],{'QCD^2==':6,'QED^2==':0})
   ,('u u~ > d d~',{},['QCD QED'],{'QCD^2==':4,'QED^2==':2})
   ,('u u~ > d d~',{},['QCD QED'],{'QCD^2==':2,'QED^2==':4})
   ,('u u~ > d d~',{},['QCD QED'],{'QCD^2==':0,'QED^2==':6})
   ,('u u~ > d d~',{},['QCD QED'],{'QCD^2=':99,'QED^2=':99})
   ]

# The longer processes below are treated one by one so that they can be better
# independently checked/updated (especially the corresponding reference pickle.)

HCR_processes_long =  [
                       # The process below is for testing the parallel tests only
                       ('g g > t t~',{'QCD':2,'QED':0},['QED'],{'QCD':4,'QED':2}),
                       ('a a > t t~',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('a a > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u u~ > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u u~ > z a',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u u~ > z z',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u u~ > a a',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('t t~ > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('ve ve~ > e+ e-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('w+ w- > h h',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('h h > h h',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u u~ > e+ e-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('e+ e- > t t~ g',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6}),
                       ('e+ e- > t t~ a',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       ('g g > t t~ g',{'QCD':3,'QED':0},['QED'],{'QCD':6,'QED':2}),
                       ('g g > t t~ h',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2}), 
                       ('g g > t t~ h',{'QCD':2,'QED':1},['QED'],{'QCD':4,'QED':4}),
                       ('a a > t t~ a',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       ('h h > h h h',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       ('u u~ > g a',{},['QCD QED'],{}),
                       # for massive b quark
                       ('g b > t w-',{'QCD':1,'QED':1},['QED'],{'QCD':2,'QED':4}),
                       ('a b > t w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('g g > t w- b~',{'QCD':2,'QED':1},['QED'],{'QCD':4,'QED':4}),
                       ('a g > t w- b~',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6}),
                       ('a a > t w- b~',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       ('u d~ > t b~',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u b > t d',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u g > t d b~',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6}),
                       ('u a > t d b~',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       # for loop-induced
                       ('g g > h h',{},['QCD'],{}),
                       # for splitting orders
                       ('u u~ > u u~',{},['QCD QED'],{'QCD':99,'QED':99}),
                       ('u u~ > u u~ g',{},['QCD QED'],{'QCD':99,'QED':99}),
                       ('u u~ > u u~ a',{},['QCD QED'],{'QCD':99,'QED':99}),
                       # checking with arXiv:1307.4331
                       ('u~ u > w+ w-',{},['QCD'],{}),
                       ('u~ u > w+ w-',{},['QED'],{}),
                       ('d~ d > w+ w-',{},['QCD'],{}),
                       ('d~ d > w+ w-',{},['QED'],{}),
                       ('u~ u > z z',{},['QCD'],{}),
                       ('u~ u > z z',{},['QED'],{}),
                       ('d~ d > z z',{},['QCD'],{}),
                       ('d~ d > z z',{},['QED'],{}),
                       ('u~ d > w- z',{},['QCD'],{}),
                       ('u~ d > w- z',{},['QED'],{})
                       ]

HCR_processes_long_dic = dict((procToFolderName(elem[0])+'_'+'_'.join(elem[2][0].split()),elem)\
                               for elem in HCR_processes_long)


ML5EW_processes_long =  [
                         ('g g > t t~',{'QCD':2,'QED':0},['QED'],{'QCD':4,'QED':2}),
                         ('a a > t t~',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('a a > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('u u~ > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('u u~ > z a',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('u u~ > z z',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('u u~ > a a',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('t t~ > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('ve ve~ > e+ e-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('w+ w- > h h',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('h h > h h',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('u u~ > e+ e-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                         ('e+ e- > t t~ g',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6}),
                         ('e+ e- > t t~ a',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                         ('g g > t t~ g',{'QCD':3,'QED':0},['QED'],{'QCD':6,'QED':2}),
                       ('g g > t t~ h',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2}), 
                       ('g g > t t~ h',{'QCD':2,'QED':1},['QED'],{'QCD':4,'QED':4}),
                       ('a a > t t~ a',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       ('h h > h h h',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       ('u u~ > g a',{},['QCD QED'],{}),
                       # for massive b quark
                       ('g b > t w-',{'QCD':1,'QED':1},['QED'],{'QCD':2,'QED':4}),
                       ('a b > t w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('g g > t w- b~',{'QCD':2,'QED':1},['QED'],{'QCD':4,'QED':4}),
                       ('a g > t w- b~',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6}),
                       ('a a > t w- b~',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       ('u d~ > t b~',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u b > t d',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6}),
                       ('u g > t d b~',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6}),
                       ('u a > t d b~',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8}),
                       # for loop-induced
                         ('g g > h h',{},['QCD'],{}),
                         # for splitting orders
                       ('u u~ > u u~',{},['QCD QED'],{'QCD':99,'QED':99}),
                       ('u u~ > u u~ g',{},['QCD QED'],{'QCD':99,'QED':99}),
                       ('u u~ > u u~ a',{},['QCD QED'],{'QCD':99,'QED':99}),
                      # checking with arXiv:1307.4331
                       ('u~ u > w+ w-',{},['QCD'],{}),
                       ('u~ u > w+ w-',{},['QED'],{}),
                       ('d~ d > w+ w-',{},['QCD'],{}),
                       ('d~ d > w+ w-',{},['QED'],{}),
                         ('u~ u > z z',{},['QCD'],{}),
                         ('u~ u > z z',{},['QED'],{}),
                         ('d~ d > z z',{},['QCD'],{}),
                         ('d~ d > z z',{},['QED'],{}),
                         ('u~ d > w- z',{},['QCD'],{}),
                         ('u~ d > w- z',{},['QED'],{})
                         ]

ML5EW_processes_long_dic = dict((procToFolderName(elem[0])+'_'+'_'.join(elem[2][0].split()),elem)\
                                for elem in ML5EW_processes_long)


def create_pickle(my_proc_list, pickle_file, runner, ref_runner=None,
                        model = 'loop_qcd_qed_sm-parallel_test', energy = 1000):
    """ Create a pickle with name 'pickle_file' on the specified processes
    and also possibly using the PS points provided by the reference runner """
    
    my_comp = loop_me_comparator.LoopMEComparator()
    if not ref_runner is None:
        my_comp.set_me_runners(ref_runner,runner)
    else:
        my_comp.set_me_runners(runner)
    my_comp.run_comparison(my_proc_list,model=model,energy=energy)

    loop_me_comparator.LoopPickleRunner.store_comparison( 
        os.path.join(_pickle_path,pickle_file),
        [runner.proc_list,runner.res_list],
        runner.model,runner.name,energy=runner.energy)
    
def compare_processes(testInstance, my_proc_list = [], model = 'loop_qcd_qed_sm-parallel_test',
        pickle_file = "", energy = 2000, tolerance = 3e-06, filename = "",
        chosen_runner = "ML5_opt",loop_induced = False,mu_r=0.0):
    """ A helper function to compare processes. 
    Note that the chosen_runner is what runner should to create the reference
    pickle if missing"""
    
    # Print out progress if it is a run for an individual process
    if len(my_proc_list)==1:
        print "\n== %s %s =="%(my_proc_list[0][0],my_proc_list[0][2])
    else:
        print "\n== %s =="%filename
    
    # Check if pickle exists, if not create it        
    if pickle_file!="" and not os.path.isfile(os.path.join(_pickle_path,pickle_file)):
        print " => Computing reference evaluation with %s"%chosen_runner
        create_loop_pickle(my_proc_list, model,
                                             pickle_file, energy, chosen_runner)
        print "\n => Done with %s evaluation"%chosen_runner
    # Load the stored runner
    if pickle_file != "":
        stored_runner = me_comparator.PickleRunner.find_comparisons(
                          os.path.join(_pickle_path,pickle_file))[0]
        energy = stored_runner.energy
    
    # Check if the process has squared order constraints
    has_sqso = any('^2' in key for proc in my_proc_list for key in \
                                                             proc[3].keys())

    MLCard = banner_mod.MadLoopParam(os.path.join(_mg5_path,'Template','loop_material',
                                                 'StandAlone','Cards','MadLoopParams.dat'))
    MLredstr=MLCard['MLReductionLib'][0:1]
 
    # Create a MERunner object for MadLoop 5 optimized
    ML5_opt = loop_me_comparator.LoopMG5Runner()
    ML5_opt.setup(_mg5_path, optimized_output=True, temp_dir=filename,\
                      mu_r=mu_r)

    if MLredstr=="1" and not has_sqso:
        # Create a MERunner object for MadLoop 5 default
        ML5_default = loop_me_comparator.LoopMG5Runner()
        ML5_default.setup(_mg5_path, optimized_output=False, temp_dir=filename,\
                      mu_r=mu_r) 

    # Create and setup a comparator
    my_comp = loop_me_comparator.LoopMEComparator()
    
    if MLredstr=="1" and not has_sqso:
    # Always put the saved run first if you use it, so that the corresponding PS
    # points will be used.
        if pickle_file != "" and not loop_induced:
            my_comp.set_me_runners(stored_runner,ML5_opt,ML5_default)
        elif pickle_file !="" and loop_induced:
            my_comp.set_me_runners(stored_runner,ML5_default)
        elif pickle_file == "" and not loop_induced:
            my_comp.set_me_runners(ML5_opt,ML5_default)
        else:
            raise MadGraph5Error, \
                'Cannot find pickle_file for loop induced process.'
    else:
        if pickle_file !="":
            my_comp.set_me_runners(stored_runner,ML5_opt)
        else:
            raise MadGraph5Error,"CANNOT find the stored result with TIR"
    
    # Run the actual comparison
    my_comp.run_comparison(my_proc_list,
                       model=model,
                       energy=energy)
    
    # Print the output
    my_comp.output_result(filename=os.path.join(_mg5_path,filename+'.log'),\
                          tolerance = tolerance)

    # Assert that all process comparisons passed the tolerance cut
    my_comp.assert_processes(testInstance, tolerance)

    # Do some cleanup
    my_comp.cleanup()

def create_loop_pickle(my_proc_list, model, pickle_file, energy, \
                                                             chosen_runner):
    """ Create the pickle file for reference for the arguments here."""
#       print "Creating loop pickle for chosen_runner=",chosen_runner
    allowed_chosen_runners = ['ML5_opt','ML5_default'] 
    if chosen_runner not in allowed_chosen_runners:
        raise MadGraph5Error, 'The reference runner can only be in %s.'%\
                                                      allowed_chosen_runners
    
    runner = None
    if chosen_runner == 'ML5_opt':
        runner = loop_me_comparator.LoopMG5Runner()
        runner.setup(_mg5_path, optimized_output=True)
    if chosen_runner == 'ML5_default':
        runner = loop_me_comparator.LoopMG5Runner()
        runner.setup(_mg5_path, optimized_output=False)
    
    create_pickle(my_proc_list,pickle_file, runner, ref_runner=None, \
                                                  model=model,energy=energy)
    
    runner.cleanup()

class ML5EWTest(unittest.TestCase):
    """ A class to test ML5 EW corrections versus runs from hard-coded reference process. """

    test_model_name = 'loop_qcd_qed_sm-parallel_test'

    def setUp(self):
        """ Here we just copy the hidden restrict_card to a regular one.
        And we don't bother making it hidden again after the test."""
        cp(os.path.join(_mg5_path,'models','loop_qcd_qed_sm',
                        '.restrict_parallel_test.dat'),
           os.path.join(_mg5_path,'models','loop_qcd_qed_sm',
                        'restrict_parallel_test.dat'))
        cp(os.path.join(_mg5_path,'models','loop_qcd_qed_sm',
                        '.restrict_parallel_test_MB.dat'),
           os.path.join(_mg5_path,'models','loop_qcd_qed_sm',
                        'restrict_parallel_test_MB.dat'))
        cp(os.path.join(_mg5_path,'models','loop_qcd_qed_sm_Gmu',
                        '.restrict_parallel_test_WW.dat'),
           os.path.join(_mg5_path,'models','loop_qcd_qed_sm_Gmu',
                        'restrict_parallel_test_WW.dat'))
        cp(os.path.join(_mg5_path,'models','loop_qcd_qed_sm_Gmu',
                        '.restrict_parallel_test_ZZ.dat'),
           os.path.join(_mg5_path,'models','loop_qcd_qed_sm_Gmu',
                        'restrict_parallel_test_ZZ.dat'))
        cp(os.path.join(_mg5_path,'models','loop_qcd_qed_sm_Gmu',
                        '.restrict_parallel_test_WZ.dat'),
           os.path.join(_mg5_path,'models','loop_qcd_qed_sm_Gmu',
                        'restrict_parallel_test_WZ.dat'))

    #===========================================================================
    # First tests consisting in a list of quick 2>2 processes to be run together
    #===========================================================================

    def test__ML5EW_sm_vs_stored_ML5EW(self):
        if ML5EW_processes_short:
            compare_processes(self,ML5EW_processes_short,model = self.test_model_name,
                                   pickle_file = 'ml5ew_short_parallel_tests.pkl',
                                        filename = 'ptest_short_ml5ew_vs_old_ml5ew',
                                                        chosen_runner='ML5_opt')

    #===========================================================================
    # First tests consisting in a list of quick 2>2 processes to be run together
    #===========================================================================

    def test_ML5EW_sm_vs_stored_ML5EW_sqso(self):
        if ML5EW_processes_short_sqso:
            compare_processes(self,ML5EW_processes_short_sqso,
                    model = self.test_model_name,
                    pickle_file = 'ml5ew_short_parallel_tests_sqso.pkl',
                    filename = 'ptest_short_ml5ew_vs_old_ml5ew_sqso', 
                    chosen_runner='ML5_opt')

    # The tests below probe one quite long process at a time individually, so
    # one can better manage them.
    
    #===========================================================================
    # First the long checks against results available in Hard-Coded Reference
    #===========================================================================
#   ('g g > t t~',{'QCD':2,'QED':0},['QED'],{'QCD':4,'QED':2})
    def test_long_sm_vs_stored_HCR_gg_ttx_QED(self):
        proc = 'gg_ttx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('a a > t t~',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_aa_ttx_QED(self):
        proc = 'aa_ttx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('a a > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_aa_wpwm_QED(self):
        proc = 'aa_wpwm_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('u u~ > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_uux_wpwm_QED(self):
        proc = 'uux_wpwm_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('u u~ > z a',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_uux_za_QED(self):
        proc = 'uux_za_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('u u~ > z z',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_uux_zz_QED(self):
        proc = 'uux_zz_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('u u~ > a a',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_uux_aa_QED(self):
        proc = 'uux_aa_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')

#   ('t t~ > w+ w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_ttx_wpwm_QED(self):
        proc = 'ttx_wpwm_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('ve ve~ > e+ e-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_vevex_epem_QED(self):
        proc = 'vevex_epem_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('w+ w- > h h',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_wpwm_hh_QED(self):
        proc = 'wpwm_hh_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('h h > h h',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_hh_hh_QED(self):
        proc = 'hh_hh_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('u u~ > e+ e-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_uux_epem_QED(self):
        proc = 'uux_epem_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('e+ e- > t t~ g',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6})
    def test_long_sm_vs_stored_HCR_epem_ttxg_QED(self):
        proc = 'epem_ttxg_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('e+ e- > t t~ a',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8})
    def test_long_sm_vs_stored_HCR_epem_ttxa_QED(self):
        proc = 'epem_ttxa_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')
        
#   ('g g > t t~ g',{'QCD':3,'QED':0},['QED'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_HCR_gg_ttxg_QED(self):
        proc = 'gg_ttxg_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_HCR_%s'%proc, chosen_runner = 'HCR')

#   ('g g > t t~ h',{'QCD':2,'QED':1},['QED'],{'QCD':4,'QED':4})    
    def test_long_sm_vs_stored_HCR_gg_ttxh_QED(self):
        proc = 'gg_ttxh_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
                                      filename = 'ptest_long_sm_vs_hcr_%s'%proc)

#   ('g g > t t~ h',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_HCR_gg_ttxh_QCD(self):
        proc = 'gg_ttxh_QCD'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR')

#   ('a a > t t~ a',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8})
    def test_long_sm_vs_stored_HCR_aa_ttxa_QED(self):
        proc = 'aa_ttxa_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR')
        
#   ('h h > h h h',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8})
    def test_long_sm_vs_stored_HCR_hh_hhh_QED(self):
        proc = 'hh_hhh_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR')
        
#   ('u u~ > g a',{},['QCD QED'],{})
    def test_long_sm_vs_stored_HCR_uux_ga_QCD_QED(self):
        proc = 'uux_ga_QCD_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR')

    #===========================================================================
    # Now the long checks against results previsouly generated in MadLoop 5.
    #===========================================================================

#   ('g g > h t t~',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2})
# it cannot be used since the parameter of loop_qcd_qed_sm is different with loop_sm
# ml5_sm_%s.pkl is generated by loop_sm-parallel_tests 
    def test_long_sm_vs_stored_ML5EW_gg_ttxh_QCD(self):
        return
        proc = "gg_httx"
        compare_processes(self,[ML5EW_processes_long_dic[proc+'_QCD']],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s_QCD'%proc,
                                                      chosen_runner = 'ML5_opt')

#  test loop induced processes
#  ('g g > h h',{},['QCD'],{})
    def test_long_sm_vs_stored_HCR_gg_hh_QCD(self):
        proc = 'gg_hh_QCD'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = True)
        
    # test splitting orders (now only test the sum)
    # ('u u~ > u u~',{},['QCD QED'],{'QCD':99,'QED':99}
    def test_long_sm_vs_stored_HCR_uux_uux_QCD_QED(self):
        proc = 'uux_uux_QCD_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)
        
    # test splitting orders (now only test the sum)
    # ('u u~ > u u~ g',{},['QCD QED'],{'QCD':99,'QED':99}
    def test_long_sm_vs_stored_HCR_uux_uuxg_QCD_QED(self):
        proc = 'uux_uuxg_QCD_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # test splitting orders (now only test the sum)
    # ('u u~ > u u~ a',{},['QCD QED'],{'QCD':99,'QED':99}
    def test_long_sm_vs_stored_HCR_uux_uuxa_QCD_QED(self):
        proc = 'uux_uuxa_QCD_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)
        
    # checking with arXiv:1307.4331
    # thanks to Ninh for providing us the PS results
    # ('u~ u > w+ w-',{},['QCD'],{})
    def test_long_sm_vs_stored_HCR_uxu_wpwm_QCD(self):
        proc = 'uxu_wpwm_QCD'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_WW", 
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat
        
    # ('u~ u > w+ w-',{},['QED'],{})
    def test_long_sm_vs_stored_HCR_uxu_wpwm_QED(self):
        proc = 'uxu_wpwm_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_WW", 
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat
        
    # ('d~ d > w+ w-',{},['QCD'],{})
    def test_long_sm_vs_stored_HCR_dxd_wpwm_QCD(self):
        proc = 'dxd_wpwm_QCD'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_WW", 
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat
        
    # ('d~ d > w+ w-',{},['QED'],{})
    def test_long_sm_vs_stored_HCR_dxd_wpwm_QED(self):
        proc = 'dxd_wpwm_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_WW", 
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat

    # ('u~ u > z z',{},['QCD'],{})
    def test_long_sm_vs_stored_HCR_uxu_zz_QCD(self):
        proc = 'uxu_zz_QCD'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_ZZ",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat

    # ('u~ u > z z',{},['QED'],{})
    def test_long_sm_vs_stored_HCR_uxu_zz_QED(self):
        proc = 'uxu_zz_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_ZZ",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat

    # ('d~ d > z z',{},['QCD'],{})
    def test_long_sm_vs_stored_HCR_dxd_zz_QCD(self):
        proc = 'dxd_zz_QCD'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_ZZ",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat

    # ('d~ d > z z',{},['QED'],{})
    def test_long_sm_vs_stored_HCR_dxd_zz_QED(self):
        proc = 'dxd_zz_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_ZZ",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat

    # ('u~ d > w- z',{},['QCD'],{})
    def test_long_sm_vs_stored_HCR_uxd_wmz_QCD(self):
        proc = 'uxd_wmz_QCD'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_WZ",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat

    # ('u~ d > w- z',{},['QED'],{})
    def test_long_sm_vs_stored_HCR_uxd_wmz_QED(self):
        proc = 'uxd_wmz_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm_Gmu-parallel_test_WZ",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False,mu_r=-1.0) # mu_r<0, use MU_R in param_card.dat

    # check the massive b quark in loop_qcd_qed_sm
    # ('g b > t w-',{'QCD':1,'QED':1},['QED'],{'QCD':2,'QED':4})
    def test_long_sm_vs_stored_HCR_gb_twm_QED(self):
        proc = 'gb_twm_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('a b > t w-',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_ab_twm_QED(self):
        proc = 'ab_twm_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('g g > t w- b~',{'QCD':2,'QED':1},['QED'],{'QCD':4,'QED':4})
    def test_long_sm_vs_stored_HCR_gg_twmbx_QED(self):
        proc = 'gg_twmbx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('a g > t w- b~',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6})
    def test_long_sm_vs_stored_HCR_ag_twmbx_QED(self):
        proc = 'ag_twmbx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('a a > t w- b~',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8})
    def test_long_sm_vs_stored_HCR_aa_twmbx_QED(self):
        proc = 'aa_twmbx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('u d~ > t b~',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_udx_tbx_QED(self):
        proc = 'udx_tbx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('u b > t d',{'QCD':0,'QED':2},['QED'],{'QCD':0,'QED':6})
    def test_long_sm_vs_stored_HCR_ub_td_QED(self):
        proc = 'ub_td_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('u g > t d b~',{'QCD':1,'QED':2},['QED'],{'QCD':2,'QED':6})
    def test_long_sm_vs_stored_HCR_ug_tdbx_QED(self):
        proc = 'ug_tdbx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)

    # ('u a > t d b~',{'QCD':0,'QED':3},['QED'],{'QCD':0,'QED':8})
    def test_long_sm_vs_stored_HCR_ua_tdbx_QED(self):
        proc = 'ua_tdbx_QED'
        compare_processes(self,[HCR_processes_long_dic[proc]],
               model = "loop_qcd_qed_sm-parallel_test_MB",
               pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induced = False)


if '__main__' == __name__:
    # Get full logging info
    logging.config.fileConfig(os.path.join(_mg5_path, 'tests', '.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('cmdprint').setLevel(logging.INFO)
    logging.getLogger('tutorial').setLevel(logging.ERROR)
        
    logging.basicConfig(level=logging.INFO)
    
    # save hard-coded reference results
    # Replace here the path of your HCR output file, below is just an example.
    HCRpath = '/Users/erdissshaw/Works/FLibatM/check-ML/OutputML'
    # Replace the correct model and resctrict card
    model = 'loop_qcd_qed_sm-parallel_test'
    #model = 'loop_qcd_qed_sm_Gmu-parallel_test_WW'
    #model = 'loop_qcd_qed_sm_Gmu-parallel_test_ZZ'
    #model = 'loop_qcd_qed_sm_Gmu-parallel_test_WZ'
    model = 'loop_qcd_qed_sm-parallel_test_MB'
    for savefile in HCR_processes_long_dic.keys():
        res_list = []
        proc_list = []
        if os.path.isfile(os.path.join(_pickle_path,"hcr_"+savefile+".pkl")):
            continue
        else:
            pickle_file = "hcr_"+savefile+".pkl"
        if not os.path.isfile(os.path.join(HCRpath,savefile+'.dat')):
            continue
        proc = HCR_processes_long_dic[savefile]
        proc_list.append(proc)
        res_list.append(loop_me_comparator.LoopMG5Runner.\
        parse_check_output(file(os.path.join(HCRpath,savefile+'.dat'))))
        runner = loop_me_comparator.LoopHardCodedRefRunner()
        runner.setup(proc_list,res_list,model)
        create_pickle(proc_list,pickle_file,runner,ref_runner=None,\
                                model=runner.model,energy=runner.energy)
        #loop_me_comparator.LoopPickleRunner.store_comparison( 
        #    os.path.join(_pickle_path,pickle_file),
        #    [runner.proc_list,runner.res_list],
        #    runner.model,runner.name,energy=runner.energy)
    
    # runner=save_load_object.load_from_file(os.path.join(_pickle_path,"hcr_gg_ttxh_QED.pkl"))  
    unittest.main() # necessary for unittest
    #ml5ew = ML5EWTest()
        
