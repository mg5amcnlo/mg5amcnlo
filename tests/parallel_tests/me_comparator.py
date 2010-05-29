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

"""A set of objects to allow for easy comparisons of results from various ME
generators (e.g., MG v5 against v4, ...) and output nice reports in different
formats (txt, tex, ...).
"""

import sys
import os
import datetime
import shutil
import subprocess
import logging
import re
import time
import itertools

# Get the grand parent directory (mg5 root) of the module real path 
# (tests/acceptance_tests) and add it to the current PYTHONPATH to allow
# for easy import of MG5 tools

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.sep.join(script_path.split(os.sep)[:-2]))

import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.misc as misc

class MERunner(object):
    """Base class to containing default function to setup, run and access results
    produced with a specific ME generator. 
    """

    temp_dir_name = ""

    proc_list = []
    res_list = []

    setup_flag = False

    name = 'None'

    class MERunnerException(Exception):
        """Default Exception class for MERunner objects"""

    def setup(self):
        """Empty method to define all warming up operations to be executed before
        actually running the generator.
        """
        pass

    def run(self, proc_list):
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

class MG4Runner(MERunner):
    """Runner object for the MG4 Matrix Element generator."""

    mg4_path = ""
    
    name = 'MadGraph v4'
    
    compilator ='f77'
    if misc.which('gfortran'):
        print 'use gfortran'
        compilator = 'gfortran'

    def setup(self, mg4_path, temp_dir=None):
        """Setup routine: create a temporary copy of Template and execute the
        proper script to use the standalone mode. the temp_dir variable
        can be given to specify the name of the process directory, otherwise
        a temporary one is created."""

        self.proc_list = []
        self.res_list = []

        self.setup_flag = False

        # Create a copy of Template
        if not os.path.isdir(mg4_path):
            raise IOError, "Path %s is not valid" % str(mg4_path)

        self.mg4_path = os.path.abspath(mg4_path)

        if not temp_dir:
            temp_dir = "test_" + \
                    datetime.datetime.now().strftime("%f")

        if os.path.exists(os.path.join(mg4_path, temp_dir)):
            raise IOError, "Path %s for test already exist" % \
                                    str(os.path.join(mg4_path, temp_dir))

        shutil.copytree(os.path.join(mg4_path, 'Template'),
                        os.path.join(mg4_path, temp_dir))

        self.temp_dir_name = temp_dir

        # Execute the standalone script in it
        subprocess.call(os.path.join('bin', 'standalone'),
                        cwd=os.path.join(mg4_path, temp_dir),
                        stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT)

        # Set the setup flag to true to tell other routines everything is OK
        self.setup_flag = True

        # print some info
        logging.info("Temporary standalone directory %s successfully created" % \
                     temp_dir)

    def cleanup(self):
        """Clean up temporary directories"""

        if not self.setup_flag:
            raise self.MERunnerException, \
                    "MERunner setup should be called first"

        if os.path.isdir(os.path.join(self.mg4_path, self.temp_dir_name)):
            shutil.rmtree(os.path.join(self.mg4_path, self.temp_dir_name))
            logging.info("Temporary standalone directory %s successfully removed" % \
                     self.temp_dir_name)

    def run(self, proc_list, model, orders={}, energy=1000):
        """Execute MG4 on the list of processes mentioned in proc_list, using
        the specified model, the specified maximal coupling orders and a certain
        energy for incoming particles (for decay, incoming particle is at rest).
        """

        # Due to the limitation for the number of proc defined in proc_card,
        # work with bunches of fixed number of proc.

        bunch_size = 1000
        curr_index = 0

        self.proc_list = proc_list
        dir_name = os.path.join(self.mg4_path, self.temp_dir_name)

        self.fix_energy_in_check(dir_name, energy)

        while (curr_index < len(proc_list)):

            temp_proc_list = proc_list[curr_index:min(curr_index + bunch_size,
                                                      len(proc_list))]
            # Create a proc_card.dat in the v4 format
            proc_card_file = open(os.path.join(dir_name, 'Cards', 'proc_card.dat'), 'w')
            proc_card_file.write(self.format_mg4_proc_card(temp_proc_list, model, orders))
            proc_card_file.close()

            logging.info("proc_card.dat file for %i processes successfully created in %s" % \
                         (len(temp_proc_list), os.path.join(dir_name, 'Cards')))

            # Run the newprocess script
            logging.info("Running newprocess script")
            subprocess.call(os.path.join('bin', 'newprocess'),
                            cwd=dir_name,
                            stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT)

            # Get the ME value
            for i, proc in enumerate(temp_proc_list):
                self.res_list.append(self.get_me_value(proc, i))

            curr_index += bunch_size

        return self.res_list

    def format_mg4_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v4 conventions. Does not
        support v5 decay chain format for the moment."""

        # TODO: fix the decay chain notation

        proc_card_template = template_files.mg4_proc_card.mg4_template
        process_template = template_files.mg4_proc_card.process_template

        proc_string = ""
        couplings = '\n'.join(["%s=%i" % (k, v) for k, v in orders.items()])
        for i, proc in enumerate(proc_list):
            proc_string += process_template.substitute({'process': proc + ' @%i' % i,
                                                        'coupling': couplings})

        return proc_card_template.substitute({'process': proc_string,
                                        'model': model,
                                        'multiparticle':''})

    def get_me_value(self, proc, proc_id):
        """Compile and run ./check, then parse the output and return the result
        for process with id = proc_id."""

        sys.stdout.write('.')
        sys.stdout.flush()

        shell_name = proc

        shell_name = shell_name.replace(' ', '')
        shell_name = shell_name.replace('>', '_')
        shell_name = shell_name.replace('~', 'x')
        shell_name = "P%i_" % proc_id + shell_name

        logging.info("Working on process %s in dir %s" % (proc,
                                                          shell_name))
        
        dir_name = os.path.join(self.mg4_path, self.temp_dir_name, 'SubProcesses', shell_name)
        # If directory doesn't exist, skip and return 0
        if not os.path.isdir(dir_name):
            logging.info("Directory %s hasn't been created, skipping process %s" % \
                            (shell_name, proc))
            return ((0.0, 0), [])

        # Run make
        retcode = subprocess.call('make',
                        cwd=dir_name,
                        stdout=open('/dev/null', 'w'))#, stderr=subprocess.STDOUT)
        if retcode != 0:
            logging.info("Error while executing make in %s" % shell_name)
            return ((0.0, 0), [])

        # Run ./check
        try:
            output = subprocess.Popen('./check',
                        cwd=dir_name,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            return self.parse_check_output(output.read())
            output.close()
        except IOError:
            logging.warning("Error while executing ./check in %s" % shell_name)
            return ((0.0, 0), [])

    def parse_check_output(self, output):
        """Parse the output string and return a pair where first value is 
        the ME value and GeV exponent and the second value is a list of 4 
        momenta for all particles involved."""
        print output
        res_p = []
        value = 0.0
        gev_pow = 0
        momentum_pattern = re.compile(r"""\s*\d+\s+(?P<p0>-?\d*\.\d*E[+-]?\d*)\s+
                                                (?P<p1>-?\d*\.\d*E[+-]?\d*)\s+
                                                (?P<p2>-?\d*\.\d*E[+-]?\d*)\s+
                                                (?P<p3>-?\d*\.\d*E[+-]?\d*)""",
                                                re.IGNORECASE | re.VERBOSE)

        me_value_pattern = re.compile(r"""\sMatrix\selement\s=\s*(?P<value>-?
                                          \d*\.\d*(E[+-]?\d*)?)\sGeV\^\s*(?P<pow>-?\d+)""",
                                      re.IGNORECASE | re.VERBOSE)
        for line in output.split('\n'):

            match_momentum = momentum_pattern.match(line)
            if match_momentum:
                res_p.append([float(s) for s in match_momentum.groups()])

            match_value = me_value_pattern.match(line)
            if match_value:
                value = float(match_value.group('value'))
                gev_pow = int(match_value.group('pow'))

        return ((value, gev_pow), res_p)

    def fix_energy_in_check(self, dir_name, energy):
        """Replace the hard coded collision energy in check_sa.f by the given
        energy, assuming a working dir dir_name"""

        file = open(os.path.join(dir_name, 'SubProcesses', 'check_sa.f'), 'r')
        check_sa = file.read()
        file.close()

        file = open(os.path.join(dir_name, 'SubProcesses', 'check_sa.f'), 'w')
        file.write(re.sub("SQRTS=1000d0", "SQRTS=%id0" % int(energy), check_sa))
        file.close()

class MG5Runner(MG4Runner):
    """Runner object for the MG5 Matrix Element generator."""

    mg5_path = ""

    name = 'MadGraph v5'

    def setup(self, mg5_path, mg4_path, temp_dir=None):
        """Wrapper for the mg4 setup, also initializing the mg5 path variable"""

        super(MG5Runner, self).setup(mg4_path, temp_dir)

        if not os.path.isdir(mg5_path):
            raise IOError, "Path %s is not valid" % str(mg5_path)

        self.mg5_path = os.path.abspath(mg5_path)

    def run(self, proc_list, model, orders={}, energy=1000):
        """Execute MG5 on the list of processes mentioned in proc_list, using
        the specified model, the specified maximal coupling orders and a certain
        energy for incoming particles (for decay, incoming particle is at rest).
        """

        self.proc_list = proc_list
        dir_name = os.path.join(self.mg4_path, self.temp_dir_name)

        self.fix_energy_in_check(dir_name, energy)

        # Create a proc_card.dat in the v5 format
        proc_card_file = open(os.path.join(dir_name, 'Cards', 'proc_card_v5.dat'), 'w')
        proc_card_file.write(self.format_mg5_proc_card(proc_list, model, orders))
        proc_card_file.close()

        logging.info("proc_card.dat file for %i processes successfully created in %s" % \
                     (len(proc_list), os.path.join(dir_name, 'Cards')))

        # Run mg5
        logging.info("Running mg5")
        devnull = os.open(os.devnull, os.O_RDWR)
        subprocess.call([os.path.join(self.mg5_path, 'bin', 'mg5'),
                        "-f%s" % os.path.join(dir_name, 'Cards', 'proc_card_v5.dat')],
                        stdout=devnull, stderr=subprocess.STDOUT)

        # Perform some setup (normally done by newprocess_sa)

        # Copy HELAS
        for file in os.listdir(os.path.join(self.mg4_path, 'HELAS')):
            if not os.path.isdir(os.path.join(self.mg4_path, 'HELAS', file)):
                shutil.copy(os.path.join(self.mg4_path, 'HELAS', file),
                            os.path.join(dir_name, 'Source', 'DHELAS'))
        shutil.move(os.path.join(dir_name, 'Source', 'DHELAS', 'Makefile.template'),
                    os.path.join(dir_name, 'Source', 'DHELAS', 'Makefile'))

        # Copy MODEL
        for file in os.listdir(os.path.join(self.mg4_path, 'Models', model)):
            if not os.path.isdir(os.path.join(self.mg4_path, 'Models', model, file)):
                shutil.copy(os.path.join(self.mg4_path, 'Models', model, file),
                            os.path.join(dir_name, 'Source', 'MODEL'))
        os.symlink(os.path.join(dir_name, 'Source', 'MODEL', 'coupl.inc'),
                   os.path.join(dir_name, 'Source', 'coupl.inc'))
        os.symlink(os.path.join(dir_name, 'Source', 'coupl.inc'),
                   os.path.join(dir_name, 'SubProcesses', 'coupl.inc'))
        shutil.copy(os.path.join(dir_name, 'Source', 'MODEL', 'param_card.dat'),
                   os.path.join(dir_name, 'Cards'))

        #Pass to gfortran if needed.
        if self.compilator == 'gfortran':
            retcode = subprocess.call(['python', os.path.join('bin','Passto_gfortran.py')],
                        cwd=os.path.join(self.mg4_path, self.temp_dir_name),
                        stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT)
            if retcode != 0:
                print 'out gfortran'
                logging.info("Error while passing to gfortran in %s" % shell_name)
                return ((0.0, 0), [])

        # Run make
        retcode = subprocess.call(['make', '../lib/libdhelas3.a'],
                        cwd=os.path.join(dir_name, 'Source'),
                        stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT)
        if retcode != 0:
            logging.warning("Error while executing make HELAS")

        retcode = subprocess.call(['make', '../lib/libmodel.a'],
                        cwd=os.path.join(dir_name, 'Source'),
                        stdout=open('/dev/null', 'w'), stderr=subprocess.STDOUT)
        if retcode != 0:
            logging.warning("Error while executing make HELAS")

        # Get the ME value
        for i, proc in enumerate(proc_list):
            self.res_list.append(self.get_me_value(proc, i))

        return self.res_list

    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = "import model_v4 %s\n" % os.path.join(self.mg4_path, 'Models', model)

        couplings = ' '.join(["%s=%i" % (k, v) for k, v in orders.items()])

        dir_name = os.path.join(self.mg4_path, self.temp_dir_name, 'SubProcesses')

        for i, proc in enumerate(proc_list):
            v5_string += 'generate ' + proc + ' ' + couplings + '@%i' % i + '\n'
            v5_string += 'export sa_dirs_v4 %s\n' % dir_name

        return v5_string

class MEComparator(object):
    """Base object to run comparison tests. Take standard MERunner objects and
    a list of proc as an input and return detailed comparison tables in various
    formats."""

    me_runners = []
    results = []
    proc_list = []

    def set_me_runners(self, *args):
        """Set the list of MERunner objects (properly set up!)."""

        self.me_runners = args

        for runner in self.me_runners:
            print "Code %s added" % runner.name

    def run_comparison(self, proc_list, model='sm', orders={}, energy=1000):
        """Run the codes and store results."""

        self.results = []
        self.proc_list = proc_list

        print "Running on %i processes with order: %s, in model %s @ %i GeV" % \
            (len(proc_list),
             ' '.join(["%s=%i" % (k, v) for k, v in orders.items()]),
             model,
             energy)

        for runner in self.me_runners:
            cpu_time1 = time.time()
            print "Now running %s: " % runner.name,
            sys.stdout.flush()
            self.results.append(runner.run(proc_list, model, orders, energy))
            cpu_time2 = time.time()
            print " Done in %0.3f s" % (cpu_time2 - cpu_time1),
            print " (%i/%i with zero ME)" % (len([res for res in self.results[-1] if res[0][0] == 0.0]),
                                             len(proc_list))


    def cleanup(self):
        """Call cleanup for each MERunner."""

        for runner in self.me_runners:
            print "Cleaning code %s runner" % runner.name
            runner.cleanup()

    def _fixed_string_length(self, mystr, length):
        """Helper function to fix the length of a string by cutting it 
        or adding extra space."""

        if len(mystr) > length:
            return mystr[0:length]
        else:
            return mystr + " " * (length - len(mystr))

    def output_result(self, filename=None, tolerance=1e-06, skip_zero=True):
        """Output result as a nicely formated table. If filename is provided,
        write it to the file, else to the screen. Tolerance can be adjusted."""

        col_size = 17

        pass_proc = 0
        fail_proc = 0

        failed_proc_list = []

        res_str = self._fixed_string_length("\nProcess", col_size) + \
                  ''.join([self._fixed_string_length(runner.name, col_size) for \
                           runner in self.me_runners]) + \
                  self._fixed_string_length("Relative diff.", col_size) + \
                  self._fixed_string_length("Result", col_size) + '\n'

        for i, proc in enumerate(self.proc_list):
            list_res = [res[i][0][0] for res in self.results]
            if max(list_res) == 0.0 and min(list_res) == 0.0:
                diff = 0.0
                if skip_zero:
                    continue
            else:
                diff = (max(list_res) - min(list_res)) / (max(list_res) + min(list_res))

            res_str += self._fixed_string_length('\n' + proc, col_size) + \
                       ''.join([self._fixed_string_length("%1.10e" % res, col_size) for res in list_res])

            res_str += self._fixed_string_length("%1.10e" % diff, col_size)

            if diff < tolerance:
                pass_proc += 1
                res_str += self._fixed_string_length("Pass", col_size)
            else:
                fail_proc += 1
                failed_proc_list.append(proc)
                res_str += self._fixed_string_length("Fail", col_size)

        res_str += "\n\n Summary: %i/%i passed, %i/%i failed" % \
                    (pass_proc, pass_proc + fail_proc,
                     fail_proc, pass_proc + fail_proc)

        if fail_proc != 0:
            res_str += "\n\nFailed processes: %s" % ', '.join(failed_proc_list)

        print res_str

        if filename:
            file = open(filename, 'w')
            file.write(res_str)
            file.close()

    def get_non_zero_processes(self):
        """Return a list of processes which have non zero ME for at least
        one generator."""

        non_zero_proc = []

        for i, proc in enumerate(self.proc_list):
            list_res = [res[i][0][0] for res in self.results]
            if sum([abs(res) for res in list_res]) != 0.0:
                non_zero_proc.append(proc)

        return non_zero_proc


def create_proc_list(part_list, initial=2, final=2):
    """Helper function to automatically create process lists starting from 
    a particle list."""

    proc_list = []
    res_list = []
    for product in itertools.product(part_list, repeat=initial + final):
        sorted_product = sorted(product[0:initial]) + sorted(product[initial:])
        if  sorted_product not in proc_list:
            proc_list.append(sorted_product)

    for proc in proc_list:
        proc.insert(initial, '>')
        res_list.append(' '.join(proc))

    return res_list





