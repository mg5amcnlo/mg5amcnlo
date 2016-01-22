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
"""A set of objects to allow for easy comparisons of results from various ME
generators (e.g., MG v5 against v4, ...) and output nice reports in different
formats (txt, tex, ...).
"""

import datetime
import glob
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import time

pjoin = os.path.join
# Get the grand parent directory (mg5 root) of the module real path 
# (tests/acceptance_tests) and add it to the current PYTHONPATH to allow
# for easy import of MG5 tools

_file_path = os.path.dirname(os.path.realpath(__file__))

import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.save_load_object as save_load_object
import madgraph.interface.master_interface as cmd_interface

import madgraph.various.misc as misc

from madgraph import MadGraph5Error, MG5DIR
import me_comparator

class MadEventComparator(me_comparator.MEComparator):
    """Base object to run comparison tests. Take standard Runner objects and
    a list of proc as an input and return detailed comparison tables in various
    formats."""

    def run_comparison(self, proc_list, model='sm', orders={}):
        """Run the codes and store results."""

        if isinstance(model, basestring):
            model= [model] * len(self.me_runners)

        self.results = []
        self.proc_list = proc_list

        logging.info(\
            "Running on %i processes with order: %s, in model %s" % \
            (len(proc_list),
             me_comparator.MERunner.get_coupling_definitions(orders),
             '/'.join([onemodel for onemodel in model])))

        pass_proc = False
        for i,runner in enumerate(self.me_runners):
            cpu_time1 = time.time()
            logging.info("Now running %s" % runner.name)
            if pass_proc:
                runner.pass_proc = pass_proc 

            self.results.append(runner.run(proc_list, model[i], orders))
            cpu_time2 = time.time()
            logging.info(" Done in %0.3f s" % (cpu_time2 - cpu_time1))
#            logging.info(" (%i/%i with zero ME)" % \
#                    (len([res for res in self.results[-1] if res[0][0] == 0.0]),
#                     len(proc_list)))

    def cleanup(self):
        """Call cleanup for each MERunner."""

        for runner in self.me_runners:
            logging.info("Cleaning code %s runner" % runner.name)
            runner.cleanup()
    def output_result(self, filename=None, tolerance=3e-02):
        """Output result as a nicely formated table. If filename is provided,
        write it to the file, else to the screen. Tolerance can be adjusted."""

        def detect_type(data):
            """check if the type is an integer/float/string"""
            

            if data.isdigit():
                return 'int'
            elif len(data) and data[0] == '-' and data[1:].isdigit():
                return 'int'
            
            try:
                float(data)
                return 'float'
            except:
                return 'str'


        proc_col_size = 17
        for proc in self.results[0]:
            if len(proc) + 1 > proc_col_size:
                proc_col_size = len(proc) + 1
        
        col_size = 17

        pass_test = 0
        fail_test = 0

        failed_prop_list = []

        res_str = "\n" + self._fixed_string_length("Checked", proc_col_size) + \
                ''.join([self._fixed_string_length(runner.name, col_size) for \
                           runner in self.me_runners]) + \
                  self._fixed_string_length("Relative diff.", col_size) + \
                  "Result"

        for prop in self.results[0]:
            loc_results = []
            succeed = True
            for i in range(len(self.results)):
                if not self.results[i].has_key(prop):
                    loc_results.append('not present')
                    succeed = False
                else:
                    loc_results.append(self.results[i][prop])
            res_str += '\n' + self._fixed_string_length(proc, proc_col_size)+ \
                       ''.join([self._fixed_string_length(str(res),
                                               col_size) for res in loc_results])
            if not succeed:
                res_str += self._fixed_string_length("NAN", col_size)
                res_str += 'failed'
                fail_test += 1
                failed_prop_list.append(prop) 
            else:
                # check the type (integer/float/string)
                type = detect_type(loc_results[0])
                if type == 'float':
                    if max(loc_results) == 0.0 and min(loc_results) == 0.0:
                        res_str += self._fixed_string_length("0", col_size)
                        res_str += 'passed'
                        pass_test +=1
                    else:
                        loc_results = [float(d) for d in loc_results]
                        diff = (max(loc_results) - min(loc_results)) / \
                                                 (max(loc_results) + min(loc_results))
                        res_str += self._fixed_string_length("%1.10e" % diff, col_size)
                        if diff >= tolerance:
                            res_str += 'failed'
                            failed_prop_list.append(prop) 
                            fail_test += 1
                        else:
                            res_str += 'passed'
                            pass_test +=1
                else:
                    for value in loc_results:
                        if value != loc_results[0]:
                            res_str += self._fixed_string_length("differ", col_size)
                            res_str += 'failed'
                            failed_prop_list.append(prop) 
                            fail_test += 1
                            break
                    res_str += self._fixed_string_length("identical", col_size)
                    res_str += 'passed'
                    pass_test +=1
        
        res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                    (pass_test, pass_test + fail_test,
                     fail_test, pass_test + fail_test)

        if fail_test != 0:
            res_str += "\nFailed processes: %s" % ', '.join(failed_prop_list)

        logging.info(res_str)

        if filename:
            file = open(filename, 'w')
            file.write(res_str)
            file.close()
        
        return fail_test, failed_prop_list


    def assert_processes(self, test_object, tolerance = 1e-06):
        """Run assert to check that all processes passed comparison""" 

        fail_test, fail_prop = self.output_result('', tolerance)

        test_object.assertEqual(fail_test, 0, "Failed for processes: %s" % ', '.join(fail_prop))
        
class MadEventComparatorGauge(me_comparator.MEComparatorGauge):
    """Base object to run comparison tests. Take standard Runner objects and
    a list of proc as an input and return detailed comparison tables in various
    formats."""

    def run_comparison(self, proc_list, model='sm', orders={}):
        """Run the codes and store results."""

        #if isinstance(model, basestring):
        #    model= [model] * len(self.me_runners)

        self.results = []
        self.proc_list = proc_list

        logging.info(\
            "Running on %i processes with order: %s, in model %s" % \
            (len(proc_list),
             ' '.join(["%s=%i" % (k, v) for k, v in orders.items()]),
             model))

        pass_proc = False
        for i,runner in enumerate(self.me_runners):
            cpu_time1 = time.time()
            logging.info("Now running %s" % runner.name)
            if pass_proc:
                runner.pass_proc = pass_proc 
            self.results.append(runner.run(proc_list, model, orders))
            cpu_time2 = time.time()
            logging.info(" Done in %0.3f s" % (cpu_time2 - cpu_time1))
#            logging.info(" (%i/%i with zero ME)" % \
#                    (len([res for res in self.results[-1] if res[0][0] == 0.0]),
#                     len(proc_list)))

    def cleanup(self):
        """Call cleanup for each MERunner."""

        for runner in self.me_runners:
            logging.info("Cleaning code %s runner" % runner.name)
            runner.cleanup()
            
    def output_result(self, filename=None, tolerance=3e-03):
        """Output result as a nicely formated table. If filename is provided,
        write it to the file, else to the screen. Tolerance can be adjusted."""

        def detect_type(data):
            """check if the type is an integer/float/string"""
            

            if data.isdigit():
                return 'int'
            elif len(data) and data[0] == '-' and data[1:].isdigit():
                return 'int'
            
            try:
                float(data)
                return 'float'
            except:
                return 'str'


        proc_col_size = 17
        for proc in self.results[0]:
            if len(proc) + 1 > proc_col_size:
                proc_col_size = len(proc) + 1
        
        col_size = 17

        pass_test = 0
        fail_test = 0

        failed_proc_list = []

        res_str = "\n" + self._fixed_string_length("Process", proc_col_size) + \
                ''.join([self._fixed_string_length(runner.name, col_size) for \
                           runner in self.me_runners]) + \
                  self._fixed_string_length("Diff both unit", col_size) + \
                  self._fixed_string_length("Diff both cms", col_size) + \
                  self._fixed_string_length("Diff both fixw", col_size) + \
                  self._fixed_string_length("Diff both feyn", col_size) + \
                  "Result"

        for proc in self.results[0]:
            loc_results = []
            succeed = True
            for i in range(len(self.results)):
                if not self.results[i].has_key(proc):
                    loc_results.append('not present')
                    succeed = False
                else:
                    loc_results.append(self.results[i][proc])
            res_str += '\n' + self._fixed_string_length(proc, proc_col_size)+ \
                       ''.join([self._fixed_string_length(str(res),
                                               col_size) for res in loc_results])
            if not succeed:
                res_str += self._fixed_string_length("NAN", col_size)
                res_str += 'failed'
                fail_test += 1
                failed_proc_list.append(proc) 
            else:
                # check the type (integer/float/string)
                type = detect_type(loc_results[0])
                if type == 'float':
                    if max(loc_results) == 0.0 and min(loc_results) == 0.0:
                        res_str += self._fixed_string_length("0", col_size)
                        res_str += 'passed'
                        pass_test +=1
                    else:
                        loc_results = [float(d) for d in loc_results]                        
                        diff_feyn = abs(loc_results[1] - loc_results[2]) / \
                          (loc_results[1] + loc_results[2] + 1e-99)
                        diff_unit = abs(loc_results[0] - loc_results[3]) / \
                          (loc_results[0] + loc_results[3] + 1e-99)
                        diff_cms = abs(loc_results[0] - loc_results[1]) / \
                          (loc_results[0] + loc_results[1] + 1e-99)
                        diff_fixw = abs(loc_results[2] - loc_results[3]) / \
                          (loc_results[2] + loc_results[3] + 1e-99)
                        
                        res_str += self._fixed_string_length("%1.10e" % diff_unit, col_size)
                        res_str += self._fixed_string_length("%1.10e" % diff_cms, col_size)
                        res_str += self._fixed_string_length("%1.10e" % diff_fixw, col_size)
                        res_str += self._fixed_string_length("%1.10e" % diff_feyn, col_size)
                        
                        if diff_feyn < 4e-2 and diff_cms < 1e-2 and diff_fixw < 1e-2 and \
                         diff_unit < 4e-2:
                            pass_test += 1
                            res_str += "Pass"
                        else:
                           fail_test += 1
                           failed_proc_list.append(proc)
                           res_str += "Fail"

                else:
                    for value in loc_results:
                        if value != loc_results[0]:
                            res_str += self._fixed_string_length("differ", col_size)
                            res_str += 'failed'
                            failed_proc_list.append(proc) 
                            fail_test += 1
                            break
                    res_str += self._fixed_string_length("identical", col_size)
                    res_str += 'passed'
                    pass_test +=1
        
        res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                    (pass_test, pass_test + fail_test,
                     fail_test, pass_test + fail_test)

        if fail_test != 0:
            res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)

        logging.info(res_str)

        if filename:
            file = open(filename, 'w')
            file.write(res_str)
            file.close()
        
        return fail_test, failed_proc_list


    def assert_processes(self, test_object, tolerance = 1e-06):
        """Run assert to check that all processes passed comparison""" 

        fail_test, fail_prop = self.output_result('', tolerance)

        test_object.assertEqual(fail_test, 0, "Failed for processes: %s" % ', '.join(fail_prop))


class FakeRunner(object):
     temp_dir_name = ""
     proc_list = []
     res_list = []
     setup_flag = False
     name = 'Store'
     type = 'Store'
     model_dir = os.path.join(MG5DIR,'models')   
     
     def cleanup(self):
         pass

class MadEventRunner(object):
    """Base class to containing default function to setup, run and access results
    produced with a specific ME generator. 
    """

    temp_dir_name = ""
    proc_list = []
    res_list = []
    setup_flag = False
    name = 'None'
    model_dir = os.path.join(MG5DIR,'models')

    class MERunnerException(Exception):
        """Default Exception class for MERunner objects"""

    def setup(self):
        """Empty method to define all warming up operations to be executed before
        actually running the generator.
        """
        pass

    def run(self, proc_list, model, orders, energy):
        """Run the generator for a specific list of processes (see below for
           conventions) and store the result.
        """
        pass

    def get_result(self, proc_id):
        """Return the result (i.e., ME value for a particular PS point) for a 
        specific process identified with its id."""

        return self.proc_list[proc_id]

    def cleanup(self):
        """Perform some clean up procedure to leave the ME code directory in
        the same state as it was initially (e.g., remove temp dirs, ...)
        """
        pass 
            

class MG5Runner(MadEventRunner):
    """Runner object for the MG5 Matrix Element generator."""

    mg5_path = ""
    name = 'MadGraph v5'
    type = 'v5'
        

    def setup(self, mg5_path, temp_dir=None):
        """Wrapper for the mg4 setup, also initializing the mg5 path variable"""

        self.mg5_path = os.path.abspath(mg5_path)

        if not temp_dir:
            i=0
            while os.path.exists(os.path.join(mg5_path, 
                                              "p_ME_test_%s_%s" % (self.type, i))):
                i += 1
            temp_dir = "p_ME_test_%s_%s" % (self.type, i)         

        self.temp_dir_name = temp_dir

    def run(self, proc_list, model, orders={}):
        """Execute MG5 on the list of processes mentioned in proc_list, using
        the specified model, the specified maximal coupling orders and a certain
        energy for incoming particles (for decay, incoming particle is at rest).
        """
        self.res_list = [] # ensure that to be void, and avoid pointer problem 
        self.proc_list = proc_list
        self.model = model
        self.orders = orders
        self.non_zero = 0 

        dir_name = os.path.join(self.mg5_path, self.temp_dir_name)

        # Create a proc_card.dat in the v5 format
        proc_card_location = os.path.join(self.mg5_path, 'proc_card_%s.dat' % \
                                          self.temp_dir_name)
        proc_card_file = open(proc_card_location, 'w')
        proc_card_file.write(self.format_mg5_proc_card(proc_list, model, orders))
        proc_card_file.close()

        logging.info("proc_card.dat file for %i processes successfully created in %s" % \
                     (len(proc_list), os.path.join(dir_name, 'Cards')))

        # Run mg5
        logging.info("Running MG5")
        #proc_card = open(proc_card_location, 'r').read()
        new_proc_list = []
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('import command %s' %proc_card_location)
        #for line in proc_card.split('\n'):
        #    cmd.exec_cmd(line, errorhandling=False)
        os.remove(proc_card_location)

        values = self.get_values()
        self.res_list.append(values)
        return values

    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        if model != 'mssm':
            v5_string = "import model %s\n" % os.path.join(self.model_dir, model)
        else:
            v5_string = "import model %s\n" % model
        v5_string += "set automatic_html_opening False\n"
        couplings = me_comparator.MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output %s -f\n" % \
                     os.path.join(self.mg5_path, self.temp_dir_name)
        v5_string += "launch -i --multicore\n"
        v5_string += " set automatic_html_opening False\n"
        v5_string += "edit_cards\n"
        v5_string += "set ickkw 0\n"
        v5_string += "set LHC 13\n"
        v5_string += "set xqcut 0\n"
        v5_string += "set cut_decays True\n"
        v5_string += "survey run_01; refine 0.01; refine 0.01\n" 
        #v5_string += "print_results\n"
        return v5_string
    
    def get_values(self):
    
        dir_name = os.path.join(self.mg5_path, self.temp_dir_name)
        SubProc=[name for name in os.listdir(dir_name + '/SubProcesses') 
                 if name[0]=='P' and 
                 os.path.isdir(dir_name + '/SubProcesses/'+name) and \
                  name[1].isdigit()]

        output = {}
        
        #Part1: number of SubProcesses                                                                                                                           
        numsubProc={}
        for name in SubProc :
            tag=name.split('_')[0][1:]
            if numsubProc.has_key(tag):
                numsubProc[tag]+=1
            else: numsubProc[tag]=1

        for key,value in numsubProc.items():
            output['number_of_P'+key]=str(value)

        #Part 2: cross section                                                                                                                                   
        for name in SubProc:
            if os.path.exists(dir_name+'/SubProcesses/'+name+'/run_01_results.dat'):
                filepath = dir_name+'/SubProcesses/'+name+'/run_01_results.dat'
            else:
                filepath = dir_name+'/SubProcesses/'+name+'/results.dat'
            for line in file(filepath):
                splitline=line.split()
                #if len(splitline)==8:
                output['cross_'+name]=splitline[0]
                print "found %s %s" % (splitline[0], splitline[1])
        return output

class MG5OldRunner(MG5Runner):
    """Runner object for the MG5 Matrix Element generator."""

    mg5_path = ""
    name = 'v5 Ref'
    type = 'v5_ref'
    
    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = "import model %s\n" % os.path.join(self.model_dir, model)
        v5_string += "set automatic_html_opening False\n"
        couplings =  me_comparator.MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output %s -f\n" % \
                     os.path.join(self.mg5_path, self.temp_dir_name)
        v5_string += "launch -f \n"
        return v5_string
    
    def run(self, proc_list, model, orders={}):
        """Execute MG5 on the list of processes mentioned in proc_list, using
        the specified model, the specified maximal coupling orders and a certain
        energy for incoming particles (for decay, incoming particle is at rest).
        """
        self.res_list = [] # ensure that to be void, and avoid pointer problem 
        self.proc_list = proc_list
        self.model = model
        self.orders = orders
        self.non_zero = 0 
        dir_name = os.path.join(self.mg5_path, self.temp_dir_name)

        # Create a proc_card.dat in the v5 format
        proc_card_location = os.path.join(self.mg5_path, 'proc_card_%s.dat' % \
                                          self.temp_dir_name)
        proc_card_file = open(proc_card_location, 'w')
        proc_card_file.write(self.format_mg5_proc_card(proc_list, model, orders))
        proc_card_file.close()

        logging.info("proc_card.dat file for %i processes successfully created in %s" % \
                     (len(proc_list), os.path.join(dir_name, 'Cards')))

        # Run mg5
        logging.info("Running MG5")
        devnull = open(os.devnull,'w') 

        if logging.root.level >=20:
            subprocess.call([pjoin(self.mg5_path,'bin','mg5'), proc_card_location],
                        stdout=devnull, stderr=devnull)
        else:       
            subprocess.call([pjoin(self.mg5_path,'bin','mg5'), proc_card_location])
        os.remove(proc_card_location)

        values = self.get_values()
        self.res_list.append(values)
        return values
    
class MG5gaugeRunner(MG5Runner):
    """Runner object for the MG5 Matrix Element generator."""

    def __init__(self, cms, gauge):
        self.cms = cms
        self.gauge = gauge
        self.mg5_path = ""
        self.name = 'MG_%s_%s' %(self.cms, self.gauge)
        self.type = '%s_%s' %(self.cms, self.gauge)
    
    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = 'import model sm \n'
        v5_string += 'set automatic_html_opening False\n'
        v5_string += 'set complex_mass_scheme %s \n' % self.cms
        v5_string += 'set gauge %s \n' % self.gauge
        v5_string += "import model %s \n" % os.path.join(self.model_dir, model)

        couplings = me_comparator.MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output %s -f\n" % \
                     os.path.join(self.mg5_path, self.temp_dir_name)
        v5_string += "launch -f \n"
        
        v5_string += 'set complex_mass_scheme False \n'
        v5_string += 'set gauge unitary'
        return v5_string
