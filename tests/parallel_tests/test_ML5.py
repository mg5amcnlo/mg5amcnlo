import logging
import logging.config
import pydoc
import os
import loop_me_comparator
import me_comparator
import unittest


from madgraph import MG5DIR

#Look for MG5/MG4 path
_mg5_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
_file_path = os.path.dirname(os.path.realpath(__file__))
_pickle_path = os.path.join(_file_path, 'input_files')

ML4_processes_short = [('u u~ > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('d g > d g',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('g g > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4}),
                       ('g g > t t~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('d~ d > g a',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}),
                       ('d~ d > g z',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2})]
ML4_processes_long =  [('g g > h t t~',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2}), 
                       ('g g > h h t t~',{'QCD':2,'QED':2},['QCD'],{'QCD':6,'QED':4}),
                       ('d d~ > w+ w- g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),
                       ('d~ d > z z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),
                       ('d~ d > z g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('d~ d > a g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('g g > z t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('g g > a t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})]

ML5_processes_short = [('u u~ > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('d g > d g',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('d~ u~ > d~ u~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}), 
                       ('g u~ > g u~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('g g > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('g g > t t~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('g g > g g',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('d~ d > g a',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}),
                       ('u~ u > g z',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}),
                       ('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4}),
                       ('d u~ > w- g',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2})]
ML5_processes_long =  [('g g > h t t~',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2}), 
                       ('d d~ > w+ w- g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),
                       ('d~ d > z z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),
                       ('s s~ > a z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),                       
                       ('d~ d > z g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('d~ d > a g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('d~ u > w+ g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('g g > w- d~ u',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),                       
                       ('g g > z t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('g g > a t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('g g > h h t t~',{'QCD':2,'QED':2},['QCD'],{'QCD':6,'QED':4}),
                       ('u u~ > w+ w- b b~',{'QCD':2,'QED':2},['QCD'],{'QCD':6,'QED':4}),                       
                       ('g g > g g g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0})]

#ML4_processes_short = [('u u~ > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
#                       ('d g > d g',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0})]
#
#ML5_processes_short = ML4_processes_short 
#ML5_processes_long = ML4_processes_short + [('d~ d > z z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4})]
#ML4_processes_short = ML4_processes_short
#ML4_processes_long = ML4_processes_short + [('g g > h t t~',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2})]

class ML5Test(unittest.TestCase):
    """ A class to test ML5 versus runs from older versions or ML4 """

    @staticmethod
    def create_pickle(my_proc_list, pickle_file, runner, ref_runner=None,
                      model = 'loop_sm-no_twidth', energy = 2000):
        """ Create a pickle with name 'pickle_file' on the specified processes
        and also possibly using the PS points provided by the reference runner """
        
        my_comp = loop_me_comparator.LoopMEComparator()
        if ref_runner:
            my_comp.set_me_runners(ref_runner,runner)
        else:
            my_comp.set_me_runners(runner)
        my_comp.run_comparison(my_proc_list,model=model,energy=energy)

        loop_me_comparator.LoopPickleRunner.store_comparison( 
            os.path.join(_pickle_path,pickle_file),
            [runner.proc_list,runner.res_list],
            runner.model,runner.name,energy=runner.energy)
        
        my_comp.cleanup()
        

    def compare_processes(self, my_proc_list = [], model = 'loop_sm-no_twidth',
                      pickle_file = "", energy = 2000, tolerance = 1e-06, filename = ""):
        """ A helper function to compare processes. """
        
        # Check if pickle exists, if not create it        
        if pickle_file!="" and not os.path.isfile(os.path.join(_pickle_path,pickle_file)): 
            self.test_create_loop_pickles()

        # Load the stored runner
        if pickle_file != "":
            stored_runner = me_comparator.PickleRunner.find_comparisons(
                              os.path.join(_pickle_path,pickle_file))[0]

        # Create a MERunner object for MadLoop 5 optimized
        ML5_opt = loop_me_comparator.LoopMG5Runner()
        ML5_opt.setup(_mg5_path, optimized_output=True)
    
        # Create a MERunner object for MadLoop 5 default
        ML5_default = loop_me_comparator.LoopMG5Runner()
        ML5_default.setup(_mg5_path, optimized_output=False) 

        # Create and setup a comparator
        my_comp = loop_me_comparator.LoopMEComparator()
        
        # Always put the saved run first if you use it, so that the corresponding PS
        # points will be used.
        if pickle_file != "":
            my_comp.set_me_runners(stored_runner,ML5_opt,ML5_default)
        else:
            my_comp.set_me_runners(ML5_opt,ML5_default)
        
        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                           model=model,
                           energy=energy)
        # Print the output
        my_comp.output_result(filename=filename)

        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)

        # Do some cleanup
        my_comp.cleanup()

    def test_create_loop_pickles(self):

        if not os.path.isfile(os.path.join(_pickle_path,'ml5_parallel_test.pkl')):
            ML5_opt = loop_me_comparator.LoopMG5Runner()
            ML5_opt.setup(_mg5_path, optimized_output=True)       
            # Replace here the path of your ML4 installation
            self.create_pickle(ML5_processes_short+ML5_processes_long,'ml5_parallel_test.pkl',
                           ML5_opt, ref_runner=None, model='loop_sm-no_twidth',energy=2000)

        if not os.path.isfile(os.path.join(_pickle_path,'ml4_parallel_test.pkl')): 
            ML4 = loop_me_comparator.LoopMG4Runner()
            # Replace here the path of your ML4 installation
            ML4.setup('/Users/Spooner/Documents/PhD/MadFKS/ML4ParrallelTest/NLOComp')
            self.create_pickle(ML4_processes_short+ML4_processes_long,'ml4_parallel_test.pkl',
                           ML4, ref_runner=None, model='loop_sm-no_twidth',energy=2000) 


    def test_short_sm_vs_stored_ML5(self):
        self.compare_processes(ML5_processes_short, model = 'loop_sm-no_twidth',
          pickle_file = 'ml5_parallel_test.pkl',filename = 'short_ml5_vs_old_ml5_parallel_test.log')

    # The test below is a bit lengthy, so only run it when you want full proof check of ML5.
    def notest_long_sm_vs_stored_ML5(self):
        self.compare_processes(ML5_processes_long, model = 'loop_sm-no_twidth',
          pickle_file = 'ml5_parallel_test.pkl',filename = 'long_ml5_vs_old_ml5_parallel_test.log')
    # In principle since previous version of ML5 has been validated against ML4, it is not 
    # necessary to test both against ML4 and the old ML5.
    def notest_short_sm_vs_stored_ML4(self):
        self.compare_processes(ML4_processes_short, model = 'loop_sm-no_twidth',
          pickle_file = 'ml4_parallel_test.pkl',filename = 'short_ml5_vs_ml4_parallel_test.log')

    def notest_long_sm_vs_stored_ML4(self):
        self.compare_processes(ML4_processes_long, model = 'loop_sm-no_twidth',
          pickle_file = 'ml4_parallel_test.pkl',filename = 'long_ml5_vs_ml4_parallel_test.log')
