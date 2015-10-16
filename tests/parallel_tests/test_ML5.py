import logging.config
import logging
import pydoc
import os
import loop_me_comparator
import me_comparator
import unittest
import re
import copy


from madgraph import MG5DIR
from madgraph import MadGraph5Error
import madgraph.various.banner as banner_mod
from madgraph.iolibs.files import cp
import madgraph.various.misc as misc

#Look for MG5/MG4 path
_mg5_path = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3])
_file_path = os.path.dirname(os.path.realpath(__file__))
_pickle_path = os.path.join(_file_path, 'input_files', 'ML_parallel_saved_runs')

# The processes below are treated all together because they are relatively quick

ML4_processes_short = [('u u~ > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('d g > d g',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('g g > d d~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('e+ e- > d d~',{'QED':2,'QCD':0},['QCD'],{'QCD':2,'QED':4}),
                       ('g g > t t~',{'QCD':2,'QED':0},['QCD'],{'QCD':6,'QED':0}),
                       ('d~ d > g a',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2}),
                       ('d~ d > g z',{'QED':1,'QCD':1},['QCD'],{'QCD':4,'QED':2})]

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

# The longer processes below are treated one by one so that they can be better
# independently checked/updated (especially the corresponding reference pickle.)

ML4_processes_long =  [
                       # The process below is for testing the parallel tests only
                       ('e+ e- > d d~',{'QCD':0,'QED':2},['QCD'],{'QCD':2,'QED':4}),
                       # Below are the processes really tested
                       ('g g > h t t~',{'QCD':2,'QED':2},['QCD'],{'QCD':6,'QED':4}), 
                       ('d d~ > w+ w- g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),
                       ('d~ d > z z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),
                       ('d~ d > z g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('d~ d > a g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('g g > z t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}),
                       ('g g > a t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})]

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
                       ('g g > g g g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0}),
                       ('u u~ > z z z',{'QED':3,'QCD':0},['QCD'],{'QCD':2,'QED':6}),
                       ('u d~ > h t b~',{'QED':3,'QCD':0},['QCD'],{'QCD':2,'QED':6}),
                       ('u u~ > w+ w- z',{'QED':3,'QCD':0},['QCD'],{'QCD':2,'QED':6}),
                       ('g g > g t t~',{'QED':0,'QCD':3},['QCD'],{'QCD':8,'QED':0}),
                       ('g s > e- ve~ c',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4}),
                       ('g g > z c c~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})]

ML5_uuxddx_SplitOrders =  [
   ('u u~ > d d~',{},['QCD'],{'QCD^2==':6,'QED^2==':0})
   ,('u u~ > d d~',{},['QCD'],{'QCD^2==':4,'QED^2==':2})
   ,('u u~ > d d~',{},['QCD'],{'QCD^2==':2,'QED^2==':4})
   ,('u u~ > d d~',{},['QCD'],{'QCD^2>':2})
   ,('u u~ > d d~',{},['QCD'],{'QED^2<=':2})
   ,('u u~ > d d~',{},['QCD'],{'QCD^2=':99,'QED^2=':99})
   ]

ML5_SplitOrders_long =  [
   ('u d~ > d d~ g w+' ,{},['QCD'],{'QCD^2==':6,'QED^2==':4}),
   ('d d~ > d d~' ,{},['QCD'],{'WEIGHTED^2>':6})]


def procToFolderName(proc, sqso = {}):
    """ Transform a string proc like 'u u~ > e+ e-' to a string for a folder name
    which would be uux_epem. Also adds a suffix to this name according to the 
    squared split orders sqso if specified."""
    res=''.join(proc.split(' '))
    equiv_strings = [('+','p'),('-','m'),('~','x'),('>','_')]
    for eq in equiv_strings:
        res=res.replace(eq[0],eq[1])
    
    if sqso=={}:
        return res
    
    sq_order_re = re.compile(
          r"^\s*(?P<coup_name>\w*)\s*\^2\s*(?P<logical_operator>(==)|(<=)|=|>)")
    sqso_strings = {'==':'_eq_','<=':'_le_','=':'_le_','>':'_gt_'}
    for coup, value in sqso.items(): 
        parsed = sq_order_re.match(coup)
        if parsed is None:
            raise MadGraph5Error, " Could not parse squared orders %s"%coup
        res = res + "_%ssq%s%i"%(parsed.group('coup_name'),
                          sqso_strings[parsed.group('logical_operator')],value)
    return res


ML4_processes_long_dic = dict((procToFolderName(elem[0]),elem) for elem in \
                                                             ML4_processes_long)
ML5_processes_long_dic = dict((procToFolderName(elem[0]),elem) for elem in \
                                                             ML5_processes_long)
ML5_SplitOrders_long_dic = dict((procToFolderName(elem[0], sqso=elem[3]),elem) \
                                              for elem in ML5_SplitOrders_long)

class ML5Test(unittest.TestCase):
    """ A class to test ML5 versus runs from older versions or ML4 """

    test_model_name = 'loop_sm-parallel_test'
    test_model_name_c_massive = 'loop_sm-parallel_test_c_massive'

    def setUp(self):
        """ Here we just copy the hidden restrict_card to a regular one.
        And we don't bother making it hidden again after the test."""
        cp(os.path.join(_mg5_path,'models','loop_sm','.restrict_parallel_test.dat'),
           os.path.join(_mg5_path,'models','loop_sm','restrict_parallel_test.dat'))
        cp(os.path.join(_mg5_path,'models','loop_sm',
                                       '.restrict_parallel_test_c_massive.dat'),
           os.path.join(_mg5_path,'models','loop_sm',
                                        'restrict_parallel_test_c_massive.dat'))

    @staticmethod
    def create_pickle(my_proc_list, pickle_file, runner, ref_runner=None,
                      model = 'loop_sm-parallel_test', energy = 2000):
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
        
    def compare_processes(self, my_proc_list = [], model = 'loop_sm-parallel_test',
            pickle_file = "", energy = 2000, tolerance = 3e-06, filename = "",
            chosen_runner = "ML5_opt", compare_with = ['ML5_opt','ML5_def']):
        """ A helper function to compare processes. 
        Note that the chosen_runner is what runner should to create the reference
        pickle if missing"""            

        # Print out progress if it is a run for an individual process
        if len(my_proc_list)==1:
            print "\n== %s =="%my_proc_list[0][0]
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

        # Create a MERunner object for MadLoop 5 optimized
        ML5_opt = loop_me_comparator.LoopMG5Runner()
        ML5_opt.setup(_mg5_path, optimized_output=True, temp_dir=filename)
        
        MLCard = banner_mod.MadLoopParam(os.path.join(_mg5_path,'Template','loop_material',
                                                 'StandAlone','Cards','MadLoopParams.dat'))
        MLredstr=MLCard['MLReductionLib'][0:1]

        if MLredstr=="1":
            # Create a MERunner object for MadLoop 5 default
            ML5_default = loop_me_comparator.LoopMG5Runner()
            ML5_default.setup(_mg5_path, optimized_output=False, temp_dir=filename) 

        # Create and setup a comparator
        my_comp = loop_me_comparator.LoopMEComparator()
        
        # Remove the default ML5 comparator if using TIR
        to_compare_with = copy.copy(compare_with)
        if MLredstr=="1":
            try:
                to_compare_with.pop(to_compare_with.index('ML5_def'))
            except ValueError:
                pass  

        runners_available = {'ML5_opt':ML5_opt, 'ML5_def':ML5_opt, 
                                                         'Stored':stored_runner}
        # Always put the saved run first if you use it, so that the corresponding PS
        # points will be used.
        runners_to_compare = []
        if pickle_file != "":
            runners_to_compare.append(stored_runner)
        runners_to_compare.extend([runners_available[runner] for runner in \
                                                               to_compare_with])
        
        if len(runners_to_compare)<=1:
                raise MadGraph5Error,\
"""  Only one runner to compute the result with, so there is no possible comparison.
  This is most likely due to the fact that you are running a TIR only comparison
  and the reference pickle cannot be found."""            
        
        # Set the runners to include
        my_comp.set_me_runners(*runners_to_compare)
        
        # Run the actual comparison
        my_comp.run_comparison(my_proc_list,
                           model=model,
                           energy=energy)
        
        # Print the output
        my_comp.output_result(filename=os.path.join(_mg5_path,filename+'.log'))

        # Assert that all process comparisons passed the tolerance cut
        my_comp.assert_processes(self, tolerance)

        # Do some cleanup
        my_comp.cleanup()

    def create_loop_pickle(self, my_proc_list, model, pickle_file, energy, \
                                                                 chosen_runner):
        """ Create the pickle file for reference for the arguments here."""
#       print "Creating loop pickle for chosen_runner=",chosen_runner
        allowed_chosen_runners = ['ML4','ML5_opt','ML5_default'] 
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
        if chosen_runner == 'ML4':
            runner = loop_me_comparator.LoopMG4Runner()
            # Replace here the path of your ML4 installation
            runner.setup('/Users/valentin/Documents/Work/aMC@NLO_v4/ML4ParrallelTest/NLOComp')
        
        self.create_pickle(my_proc_list,pickle_file, runner, ref_runner=None, \
                                                      model=model,energy=energy)
        # Clean up the runner only if it is not ML4
        if chosen_runner != 'ML4':
            runner.cleanup()

    #===========================================================================
    # First tests consisting in a list of quick 2>2 processes to be run together
    #===========================================================================

    def test_short_ML5_sm_vs_stored_ML5(self):
        self.compare_processes(ML5_processes_short,model = self.test_model_name,
                                   pickle_file = 'ml5_short_parallel_tests.pkl',
                                        filename = 'ptest_short_ml5_vs_old_ml5',
                                                        chosen_runner='ML5_opt')

    # In principle since previous version of ML5 has been validated against ML4, 
    # it is not necessary to test both against ML4 and the old ML5.
    def test_short_ML5_sm_vs_stored_ML4(self):
        self.compare_processes(ML4_processes_short,model = self.test_model_name,
                                    pickle_file = 'ml4_short_parallel_test.pkl',
                                            filename = 'ptest_short_ml5_vs_ml4',
                                                            chosen_runner='ML4')

    def test_uuxddx_SplitOrders_vs_stored_ML5(self):
        self.compare_processes(ML5_uuxddx_SplitOrders,model = self.test_model_name,
                      pickle_file = 'ml5_uuxddx_SplitOrders_parallel_tests.pkl',
                                filename = 'ptest_uuxddx_SplitOrders_vs_old_ml5',
                             chosen_runner='ML5_opt',compare_with = ['ML5_opt'])

    # The tests below probe one quite long process at a time individually, so
    # one can better manage them.
    
    #===========================================================================
    # First the long checks against results available in MadLoop4
    #===========================================================================

#   Use the quick process below for testing the parallel test
    def notest_long_sm_vs_stored_ML4_epem_ddx(self):
        proc = 'epem_ddx'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_ml4_%s'%proc, chosen_runner = 'ML4')

#   ('g g > h t t~',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML4_gg_httx(self):
        proc = 'gg_httx'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_ml4_%s'%proc, chosen_runner = 'ML4')

#   ('d d~ > w+ w- g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4})    
    def test_long_sm_vs_stored_ML4_ddx_wpwmg(self):
        proc = 'ddx_wpwmg'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
                                      filename = 'ptest_long_sm_vs_ml4_%s'%proc)

#   ('d~ d > z z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4})
    def test_long_sm_vs_stored_ML4_dxd_zzg(self):
        proc = 'dxd_zzg'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_ml4_%s'%proc, chosen_runner = 'ML4')

#   ('d~ d > z g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML4_dxd_zgg(self):
        proc = 'dxd_zgg'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_ml4_%s'%proc, chosen_runner = 'ML4')

#   ('d~ d > a g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML4_dxd_agg(self):
        proc = 'dxd_agg'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_ml4_%s'%proc, chosen_runner = 'ML4')

#   ('g g > z t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML4_gg_zttx(self):
        proc = 'gg_zttx'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_ml4_%s'%proc, chosen_runner = 'ML4')

#   ('g g > a t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML4_gg_attx(self):
        proc = 'gg_zttx'
        self.compare_processes([ML4_processes_long_dic[proc]], 
               model = self.test_model_name, pickle_file = 'ml4_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_ml4_%s'%proc, chosen_runner = 'ML4')

    #===========================================================================
    # Now the long checks against results previsouly generated in MadLoop 5.
    #===========================================================================

#   ('g g > g t t~ ',{'QED':0,'QCD':3},['QCD'],{'QCD':8,'QED':0})
    def test_long_sm_vs_stored_ML5_gg_gttx(self):
        proc = "gg_gttx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('g g > h t t~',{'QCD':2,'QED':1},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML5_gg_httx(self):
        proc = "gg_httx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('d d~ > w+ w- g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4})
    def test_long_sm_vs_stored_ML5_ddx_wpwmg(self):
        proc = "ddx_wpwmg"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('d~ d > z z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4})
    def test_long_sm_vs_stored_ML5_dxd_zzg(self):
        proc = "dxd_zzg"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')
    
#   ('s s~ > a z g',{'QED':2,'QCD':1},['QCD'],{'QCD':4,'QED':4})
    def test_long_sm_vs_stored_ML5_ssx_azg(self):
        proc = "ssx_azg"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')
    
#   ('d~ d > z g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML5_dxd_zgg(self):
        proc = "dxd_zgg"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('d~ d > a g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML5_dxd_agg(self):
        proc = "dxd_agg"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('d~ u > w+ g g',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML5_dxu_wpgg(self):
        proc = "dxu_wpgg"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('g g > w- d~ u',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2}) 
    def test_long_sm_vs_stored_ML5_gg_wmdxu(self):
        proc = "gg_wmdxu"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')
                      
#   ('g g > z t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML5_gg_zttx(self):
        proc = "gg_zttx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('g g > a t t~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML5_gg_attx(self):
        proc = "gg_attx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('g g > h h t t~',{'QCD':2,'QED':2},['QCD'],{'QCD':6,'QED':4})
    def test_long_sm_vs_stored_ML5_gg_hhttx(self):
        proc = "gg_hhttx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('u u~ > w+ w- b b~',{'QCD':2,'QED':2},['QCD'],{'QCD':6,'QED':4})
    def test_long_sm_vs_stored_ML5_uux_wpwmbbx(self):
        proc = "uux_wpwmbbx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')
             
#   ('g g > g g g',{'QCD':3,'QED':0},['QCD'],{'QCD':8,'QED':0})
#   The chosen PS point for this process turns out to be  unstable, so that 
#   MadLoop goes to quadruple precision. The agreement is then fine (10e-14!)
#   but it takes several hours for the non-optimized evaluation. So better skip
#   it unless you explicitly want to try out quad prec behaviors.
    def notest_long_sm_vs_stored_ML5_gg_ggg(self):
        proc = "gg_ggg"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('u u~ > z z z',{'QED':3,'QCD':0},['QCD'],{'QCD':2,'QED':6})
    def test_long_sm_vs_stored_ML5_uux_zzz(self):
        proc = "uux_zzz"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('u d~ > h t b~',{'QED':3,'QCD':0},['QCD'],{'QCD':2,'QED':6}),
    def test_long_sm_vs_stored_ML5_udx_htbx(self):
        proc = "udx_htbx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')

#   ('u u~ > w+ w- z',{'QED':3,'QCD':0},['QCD'],{'QCD':2,'QED':6})
    def test_long_sm_vs_stored_ML5_uux_wpwmz(self):
        proc = "uux_wpwmz"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name, pickle_file = 'ml5_sm_%s.pkl'%proc,
                                  filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
                                                      chosen_runner = 'ML5_opt')
        
#    ('g s > e- ve~ c',{'QED':2,'QCD':1},['QCD'],{'QCD':3,'QED':2})
    def test_long_sm_vs_stored_ML5_gs_emvexc(self):
        proc = "gs_emvexc"
        self.compare_processes([ML5_processes_long_dic[proc]],
            model = self.test_model_name_c_massive, 
            pickle_file = 'ml5_sm_%s.pkl'%proc,
            filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
            chosen_runner = 'ML5_opt')
        
#   ('g g > z c c~',{'QED':1,'QCD':2},['QCD'],{'QCD':6,'QED':2})
    def test_long_sm_vs_stored_ML5_gg_zccx(self):
        proc = "gg_zccx"
        self.compare_processes([ML5_processes_long_dic[proc]],
               model = self.test_model_name_c_massive,
               pickle_file = 'ml5_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_old_ml5_%s'%proc,
               chosen_runner = 'ML5_opt')

    #===========================================================================
    # Now some longer checks against older version of MadLoop 5 for processes
    # with squared order constraints
    #===========================================================================

#   ('u d~ > d d~ g w+' ,{},['QCD'],{'QCD^2==':6,'QED^2==':4})
    def test_long_sm_vs_stored_ML5_sqso_udx_ddxgwp_QCDeq6_QEDeq4(self):
        proc = "udx_ddxgwp_QCDsq_eq_6_QEDsq_eq_4"
        self.compare_processes([ML5_SplitOrders_long_dic[proc]],
               model = self.test_model_name_c_massive,
               pickle_file = 'ml5_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_old_ml5_sqso_%s'%proc,
               chosen_runner = 'ML5_opt',compare_with = ['ML5_opt'])
        
#   ('d d~ > d d~' ,{},['QCD'],{'WEIGHTED^2>':6})]
    def test_long_sm_vs_stored_ML5_sqso_ddx_ddx_WEIGHTEDgt6(self):
        proc = "ddx_ddx_WEIGHTEDsq_gt_6"
        self.compare_processes([ML5_SplitOrders_long_dic[proc]],
               model = self.test_model_name_c_massive,
               pickle_file = 'ml5_sm_%s.pkl'%proc,
               filename = 'ptest_long_sm_vs_old_ml5_sqso_%s'%proc,
               chosen_runner = 'ML5_opt',compare_with = ['ML5_opt'])

