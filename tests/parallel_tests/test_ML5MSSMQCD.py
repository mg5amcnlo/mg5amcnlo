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
import me_comparator
from test_ML5 import procToFolderName
# The processes below are treated all together because they are relatively quick

HCR_processes_short = []


# The longer processes below are treated one by one so that they can be better
# independently checked/updated (especially the corresponding reference pickle.)

HCR_processes_long =  [
                       # The process below is for testing the parallel tests only
                       ('g g > t1 t1~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('g g > t1 t1~ g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}),
                       ('u u~ > t1 t1~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('u u~ > t1 t1~ g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}),
                       ('g g > go go',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('u u~ > go go',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('u u~ > go go g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}),
                       # loop-induced process
                       ('g g > n1 n1',{},['QCD'],{})
                       ]

HCR_processes_long_dic = dict((procToFolderName(elem[0])+'_mssm_'+'_'.join(elem[2][0].split()),elem)\
                               for elem in HCR_processes_long)


ML5MSSMQCD_processes_long =  [
                         ('g g > t1 t1~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                         ('g g > t1 t1~ g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}),
                         ('u u~ > t1 t1~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                         ('u u~ > t1 t1~ g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}),
                         ('g g > go go',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                         ('u u~ > go go',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                         ('u u~ > go go g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}),
                         # loop induced process
                         ('g g > n1 n1',{},['QCD'],{})
                         ]

ML5MSSMQCD_processes_long_dic = dict((procToFolderName(elem[0])+'_mssm_'+'_'.join(elem[2][0].split()),elem)\
                                for elem in ML5MSSMQCD_processes_long)

class ML5MSSMQCDTest(unittest.TestCase):
    """ A class to test ML5 QCD corrections in MSSM versus runs from hard-coded reference process. """

    test_model_name = 'loop_MSSM-parallel_test'

    def setUp(self):
        """ Here we just copy the hidden restrict_card to a regular one.
        And we don't bother making it hidden again after the test."""
        cp(os.path.join(_mg5_path,'models','loop_MSSM',
                        '.restrict_parallel_test.dat'),
           os.path.join(_mg5_path,'models','loop_MSSM',
                        'restrict_parallel_test.dat'))
        cp(os.path.join(_mg5_path,'models','loop_MSSM',
                        '.restrict_parallel_test_gogo.dat'),
           os.path.join(_mg5_path,'models','loop_MSSM',
                        'restrict_parallel_test_gogo.dat'))
    @staticmethod
    def create_pickle(my_proc_list, pickle_file, runner, ref_runner=None,
                      model = 'loop_MSSM-parallel_test', energy = 1000):
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
        
    def compare_processes(self, my_proc_list = [], model = 'loop_MSSM-parallel_test',
            pickle_file = "", energy = 2000, tolerance = 3e-06, filename = "",
            chosen_runner = "ML5_opt",loop_induce = False,mu_r=0.0):
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
            self.create_loop_pickle(my_proc_list, model,
                                             pickle_file, energy, chosen_runner)
            print "\n => Done with %s evaluation"%chosen_runner
        # Load the stored runner
        if pickle_file != "":
            stored_runner = me_comparator.PickleRunner.find_comparisons(
                              os.path.join(_pickle_path,pickle_file))[0]
            energy = stored_runner.energy

        file = open(os.path.join(_mg5_path,'Template','loop_material','StandAlone',
                                 'Cards','MadLoopParams.dat'), 'r')
        
        # Check if the process has squared order constraints
        has_sqso = any('^2' in key for proc in my_proc_list for key in \
                                                                 proc[3].keys())

        MLParams = file.read()
        MLred = re.search(r'#MLReductionLib\n',MLParams)
        MLredstr=MLParams[MLred.end():MLred.end()+1]
        if loop_induce and MLredstr!="1":return            
        # Create a MERunner object for MadLoop 5 optimized
        # Open Loops is not avaiable for loop induced processes
        if not loop_induce:
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
            if pickle_file != "" and not loop_induce:
                my_comp.set_me_runners(stored_runner,ML5_opt,ML5_default)
            elif pickle_file !="" and loop_induce:
                my_comp.set_me_runners(stored_runner,ML5_default)
            elif pickle_file == "" and not loop_induce:
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
        my_comp.assert_processes(self, tolerance)

        # Do some cleanup
        my_comp.cleanup()

    def create_loop_pickle(self, my_proc_list, model, pickle_file, energy, \
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
        
        self.create_pickle(my_proc_list,pickle_file, runner, ref_runner=None, \
                                                      model=model,energy=energy)
        
        runner.cleanup()

    # The tests below probe one quite long process at a time individually, so
    # one can better manage them.
    
    #===========================================================================
    # First the long checks against results available in Hard-Coded Reference
    #===========================================================================
#   ('g g > t1 t1~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0})
    def test_long_mssm_vs_stored_HCR_gg_t1t1x_QCD(self):
        proc = 'gg_t1t1x_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_mssm_vs_HCR_%s'%proc, chosen_runner = 'HCR')

#   ('g g > t1 t1~ g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0})
    def test_long_mssm_vs_stored_HCR_gg_t1t1xg_QCD(self):
        proc = 'gg_t1t1xg_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_mssm_vs_HCR_%s'%proc, chosen_runner = 'HCR')

#   ('u u~ > t1 t1~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}) 
    def test_long_mssm_vs_stored_HCR_uux_t1t1x_QCD(self):
        proc = 'uux_t1t1x_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_mssm_vs_HCR_%s'%proc, chosen_runner = 'HCR')

#   ('u u~ > t1 t1~ g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0})
    def test_long_mssm_vs_stored_HCR_uux_t1t1xg_QCD(self):
        proc = 'uux_t1t1xg_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
               filename = 'ptest_long_mssm_vs_HCR_%s'%proc, chosen_runner = 'HCR')

#   ('g g > go go',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0})
    def test_long_mssm_vs_stored_HCR_gg_gogo_QCD(self):
        proc = 'gg_gogo_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]],
                               model = "loop_MSSM-parallel_test_gogo",
                               pickle_file = 'hcr_%s.pkl'%proc,
                               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induce = False)

#   ('u u~ > go go',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0})
    def test_long_mssm_vs_stored_HCR_uux_gogo_QCD(self):
        proc = 'uux_gogo_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]],
                               model = "loop_MSSM-parallel_test_gogo",
                               pickle_file = 'hcr_%s.pkl'%proc,
                               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induce = False)

#   ('u u~ > go go g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0})
    def test_long_mssm_vs_stored_HCR_uux_gogog_QCD(self):
        proc = 'uux_gogog_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]],
                               model = "loop_MSSM-parallel_test_gogo",
                                                      pickle_file = 'hcr_%s.pkl'%proc,
                               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induce = False)

#   test loop induced processes
#   ('g g > n1 n1',{},['QCD'],{})
    def test_long_mssm_vs_stored_HCR_gg_n1n1_QCD(self):
        proc = 'gg_n1n1_mssm_QCD'
        self.compare_processes([HCR_processes_long_dic[proc]],
                               model = self.test_model_name, pickle_file = 'hcr_%s.pkl'%proc,
                               filename = 'ptest_long_sm_vs_hcr_%s'%proc, chosen_runner = 'HCR',
                               loop_induce = True)    
        

if '__main__' == __name__:
    # Get full logging info
    logging.config.fileConfig(os.path.join(_mg5_path, 'tests', '.mg5_logging.conf'))
    logging.root.setLevel(logging.INFO)
    logging.getLogger('madgraph').setLevel(logging.INFO)
    logging.getLogger('cmdprint').setLevel(logging.INFO)
    logging.getLogger('tutorial').setLevel(logging.ERROR)
        
    logging.basicConfig(level=logging.INFO)
    
    # save hard-coded reference results
    # Replace here the path of your HCR output file
    HCRpath = '/Users/erdissshaw/Works/FLibatM/check-ML/OutputML'
    # Replace the correct model and resctrict card
    model = 'loop_MSSM-parallel_test_gogo'
    #model = 'loop_qcd_qed_sm_Gmu-parallel_test_WW'
    #model = 'loop_qcd_qed_sm_Gmu-parallel_test_ZZ'
    #model = 'loop_qcd_qed_sm_Gmu-parallel_test_WZ' 
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
        ML5MSSMQCDTest.create_pickle(proc_list,pickle_file,runner,ref_runner=None,\
                                model=runner.model,energy=runner.energy)
        #loop_me_comparator.LoopPickleRunner.store_comparison( 
        #    os.path.join(_pickle_path,pickle_file),
        #    [runner.proc_list,runner.res_list],
        #    runner.model,runner.name,energy=runner.energy)
    
    # runner=save_load_object.load_from_file(os.path.join(_pickle_path,"hcr_gg_ttxh_QED.pkl"))  
    unittest.main() # necessary for unittest
    #ml5ew = ML5EWTest()
        
