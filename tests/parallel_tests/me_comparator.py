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

class MERunner(object):
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
    
    @staticmethod
    def get_coupling_definitions(orders):
        """ Return a string specifying the orders specified by the dictionary
        orders. It makes sure to support the specification of squared order
        constraints with this syntax: 
          {'QCD^2==':3, 'QED':4} -> 'QED=4 QCD^2==3'
        """
        
        # The squared order couplings can be specified via the dictionary
        # squared_orders using either of the two syntax:
        # {'QCD':2,'QED':4} or {'QCD^2<=':2,'QED^2<=':4}
        # The latter syntax allowing for specifying the comparator. 
        # The reg. exp. below makes sure one can separate the two syntaxes.       
        sq_order_re = re.compile(
          r"^\s*(?P<coup_name>\w*)\s*\^2\s*(?P<logical_operator>(==)|(<=)|=|>)")

        squared_couplings = []        
        for coup, value in orders.items():
            parsed = sq_order_re.match(coup)
            if not parsed is None:
                operator = parsed.group('logical_operator')
                coup_name = parsed.group('coup_name')
            else:
                operator = '=' 
                coup_name = coup
            squared_couplings.append('%s^2%s%i'% (coup_name,operator,value))
        return "%s "%' '.join(squared_couplings)

class MG4Runner(MERunner):
    """Runner object for the MG4 Matrix Element generator."""

    mg4_path = ""
    
    type='v4'
    name = 'MadGraph v4'
    compilator ='f77'
    if misc.which('gfortran'):
        compilator = 'gfortran'
    model = ''
    orders = {}
    energy = ""

    def setup(self, mg4_path, temp_dir=None):
        """Setup routine: create a temporary copy of Template and execute the
        proper script to use the standalone mode. the temp_dir variable
        can be given to specify the name of the process directory, otherwise
        a temporary one is created."""

        self.proc_list = []
        self.res_list = []

        self.setup_flag = False

        self.check_path(mg4_path)

        # Create a copy of Template
        if not os.path.isdir(os.path.join(mg4_path, "Template")) or \
               not os.path.isdir(os.path.join(mg4_path, "HELAS")):
            raise IOError, "Path %s is not a valid MG4 path" % str(mg4_path)

        self.mg4_path = os.path.abspath(mg4_path)

        if not temp_dir:
            i=0
            while os.path.exists(os.path.join(mg4_path, 
                                              "ptest_%s_%s" % (self.type, i))):
                i += 1
            temp_dir = "ptest_%s_%s" % (self.type, i)         

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

    def check_path(self, mg4_path):
        """Check the path for necessary directories"""

        if not os.path.isdir(os.path.join(mg4_path, "Template")) or \
               not os.path.isdir(os.path.join(mg4_path, "HELAS")):
            raise IOError, "Path %s is not a valid MG4 path" % str(mg4_path)

    def cleanup(self):
        """Clean up temporary directories"""

        #if not self.setup_flag:
        #    raise self.MERunnerException, \
        #            "MERunner setup should be called first"
        try:
            if os.path.isdir(os.path.join(self.mg4_path, self.temp_dir_name)):
                shutil.rmtree(os.path.join(self.mg4_path, self.temp_dir_name))
                logging.info("Temporary standalone directory %s successfully removed" % \
                     self.temp_dir_name)
        except:
            pass
            
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
        self.model = model
        self.orders = orders
        self.energy = energy

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
            devnull = open(os.devnull, 'w')
            logging.info("Running newprocess script")
            subprocess.call(os.path.join('.','bin', 'newprocess'),
                            cwd=dir_name,
                            stdout=devnull, stderr=devnull
                            )

            # Get the ME value
            for i, proc in enumerate(temp_proc_list):
                self.res_list.append(self.get_me_value(proc, i))

            curr_index += bunch_size

        return self.res_list

    def format_mg4_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v4 conventions. Does not
        support v5 decay chain format for the moment not squared order 
        constraints."""

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
        working_dir = os.path.join(self.mg4_path, self.temp_dir_name)
         
        shell_name = None
        directories = glob.glob(os.path.join(working_dir, 'SubProcesses',
                                  'P%i_*' % proc_id))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s" % (proc))
            return ((0.0, 0), [])

        logging.info("Working on process %s in dir %s" % (proc, shell_name))
        
        dir_name = os.path.join(working_dir, 'SubProcesses', shell_name)
        # Run make
        devnull = open(os.devnull, 'w')
        retcode = subprocess.call('make',
                        cwd=dir_name,
                        stdout=devnull, stderr=devnull)
                        
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

        res_p = []
        value = 0.0
        gev_pow = 0
        momentum_pattern = re.compile(r"""\s*\d+\s+(?P<p0>-?\d*\.\d*E[+-]?\d*)\s+
                                                (?P<p1>-?\d*\.\d*E[+-]?\d*)\s+
                                                (?P<p2>-?\d*\.\d*E[+-]?\d*)\s+
                                                (?P<p3>-?\d*\.\d*E[+-]?\d*)""",
                                                re.IGNORECASE | re.VERBOSE)

        me_value_pattern = re.compile(r"\sMatrix\selement\s=\s*(?P<value>-?\d*\.\d*(E[+-]?\d*)?)\s*GeV\^\s*(?P<pow>-?\d+)",
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

        for check_sa_path in glob.glob(\
                       os.path.join(dir_name, 'SubProcesses','*','check_sa.f')):

            file = open(check_sa_path, 'r')
            check_sa = file.read()
            file.close()
    
            file = open(check_sa_path, 'w')
            file.write(re.sub("SQRTS=1000d0", "SQRTS=%id0" % int(energy), check_sa))
            file.close()

class MG5Runner(MG4Runner):
    """Runner object for the MG5 Matrix Element generator."""

    mg5_path = ""

    name = 'MadGraph v5'
    type = 'v5'
        

    def setup(self, mg5_path, mg4_path, temp_dir=None):
        """Wrapper for the mg4 setup, also initializing the mg5 path variable"""

        self.mg4_path = os.path.abspath(mg4_path)

        if not temp_dir:
            i=0
            while os.path.exists(os.path.join(mg4_path, 
                                              "ptest_%s_%s" % (self.type, i))):
                i += 1
            temp_dir = "ptest_%s_%s" % (self.type, i)         

        self.temp_dir_name = temp_dir

    def run(self, proc_list, model, orders={}, energy=1000):
        """Execute MG5 on the list of processes mentioned in proc_list, using
        the specified model, the specified maximal coupling orders and a certain
        energy for incoming particles (for decay, incoming particle is at rest).
        """
        self.res_list = [] # ensure that to be void, and avoid pointer problem 
        self.proc_list = proc_list
        self.model = model
        self.orders = orders
        self.energy = energy
        self.non_zero = 0 

        dir_name = os.path.join(self.mg4_path, self.temp_dir_name)

        # Create a proc_card.dat in the v5 format
        proc_card_location = os.path.join(self.mg4_path, 'proc_card_%s.dat' % \
                                          self.temp_dir_name)
        proc_card_file = open(proc_card_location, 'w')
        proc_card_file.write(self.format_mg5_proc_card(proc_list, model, orders))
        proc_card_file.close()

        logging.info("proc_card.dat file for %i processes successfully created in %s" % \
                     (len(proc_list), os.path.join(dir_name, 'Cards')))

        # Run mg5
        logging.info("Running mg5")
        proc_card = open(proc_card_location, 'r').read()
        new_proc_list = []
        cmd = cmd_interface.MasterCmd()
        cmd.no_notification()
        for line in proc_card.split('\n'):
            try:
                cmd.exec_cmd(line, errorhandling=False)
            except MadGraph5Error:
                pass
            else:
                if line.startswith('add'):
                    self.non_zero += 1
                    new_proc_list.append(line)

        if hasattr(self, 'store_proc_card'):
            self.new_proc_list = '\n'.join(new_proc_list)

        # Remove the temporary proc_card
        os.remove(proc_card_location)
        if self.non_zero:
            self.fix_energy_in_check(dir_name, energy)

            # Get the ME value
            for i, proc in enumerate(proc_list):
                value = self.get_me_value(proc, i)
                self.res_list.append(value)

            return self.res_list
        else:
            self.res_list = [((0.0, 0), [])] * len(proc_list)
            return self.res_list
        
        
    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = "import model_v4 %s\n" % os.path.join(self.model_dir, model)
        v5_string += "set automatic_html_opening False\n"
        couplings = MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output standalone %s -f\n" % \
                     os.path.join(self.mg4_path, self.temp_dir_name)

        return v5_string
    
class MG5_UFO_Runner(MG5Runner):
    
    name = 'UFO-ALOHA-MG5'
    type = 'ufo'
    
    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        if model != 'mssm':
            v5_string = "import model %s \n" % os.path.join(self.model_dir, model)
        else:
            v5_string = "import model %s \n" %  model
        v5_string += "set automatic_html_opening False\n"
        couplings = MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output standalone %s -f\n" % \
                     os.path.join(self.mg4_path, self.temp_dir_name)

        return v5_string

class MG5_UFO_gauge_Runner(MG5Runner):
    
    #name = 'MG5_gauge'
    #type = 'ufo_cms'
    
    def __init__(self, cms, gauge):
        self.cms = cms
        self.gauge = gauge
        self.type =  '%s_%s' %(self.cms, self.gauge)
        self.name =  'MG5_%s_%s' %(self.cms, self.gauge)
    
    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = 'import model sm \n'
        v5_string += "set automatic_html_opening False\n"
        v5_string += 'set complex_mass_scheme %s \n' % self.cms
        v5_string += 'set gauge %s \n' % self.gauge
        v5_string += "import model %s \n" % os.path.join(self.model_dir, model)

        couplings = MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output standalone %s -f\n" % \
                     os.path.join(self.mg4_path, self.temp_dir_name)
                     
        v5_string += 'set complex_mass_scheme False \n'
        v5_string += 'set gauge unitary \n'

        return v5_string

class MG5OldRunner(MG5Runner):
    """ """
    
    mg5_path = ""
    name = 'MadGraph5 Reference'
    type = 'ufo_ref'
    
    def setup(self, mg5_path, temp_dir=None):
        """ initializing the mg5 path variable"""
        self.mg5_path = os.path.abspath(mg5_path)

        if not temp_dir:
            i=0
            while os.path.exists(os.path.join(MG5DIR, 
                                              "ptest_%s_%s" % (self.type, i))):
                i += 1
            temp_dir = "ptest_%s_%s" % (self.type, i)         
        self.temp_dir_name = temp_dir    

    def run(self, proc_list, model, orders={}, energy=1000):
        """Execute MG5 on the list of processes mentioned in proc_list, using
        the specified model, the specified maximal coupling orders and a certain
        energy for incoming particles (for decay, incoming particle is at rest).
        """
        self.res_list = [] # ensure that to be void, and avoid pointer problem 
        self.proc_list = proc_list
        self.model = model
        self.orders = orders
        self.energy = energy

        dir_name = os.path.join(MG5DIR, self.temp_dir_name)

        # Create a proc_card.dat in the v5 format
        proc_card_location = os.path.join(self.mg4_path, 'proc_card_%s.dat' % \
                                          self.temp_dir_name)
        proc_card_file = open(proc_card_location, 'w')
        if not hasattr(self, 'pass_proc'):
            proc_card_file.write(self.format_mg5_proc_card(proc_list, model, orders))
        else:
            v5_string = "import model %s \n" % model
            proc_card_file.write(v5_string)
            proc_card_file.write(self.pass_proc)
            proc_card_file.write("\n output standalone %s -f\n" % dir_name)
        proc_card_file.close()

        logging.info("proc_card.dat file for %i processes successfully created in %s" % \
                     (len(proc_list), os.path.join(dir_name, 'Cards')))

        # Run mg5
        logging.info("Running mg5")

        devnull = open(os.devnull,'w') 
        if logging.root.level >=20:
            subprocess.call([pjoin(self.mg5_path,'bin','mg5'), proc_card_location],
                        stdout=devnull, stderr=devnull)
        else:       
            subprocess.call([pjoin(self.mg5_path,'bin','mg5'), proc_card_location])
                        
        
        # Remove the temporary proc_card
        os.remove(proc_card_location)
        try:
            self.fix_energy_in_check(dir_name, energy)
        except:
            return [((0.0, 0), [])] * len(proc_list)
        # Get the ME value
        for i, proc in enumerate(proc_list):
            value = self.get_me_value(proc, i)
            self.res_list.append(value)

        return self.res_list


class MG5_UFO_OldRunner(MG5OldRunner):
    
    name = 'UFO-ALOHA-MG5-REF'
    type = 'ufo_ref'
    
    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = "import model %s \n" % model
        v5_string += "set automatic_html_opening False\n"
        couplings = MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output standalone %s -f\n" % \
                     os.path.join(self.mg4_path, self.temp_dir_name)

        return v5_string


class MG5_CPP_Runner(MG5Runner):
    """Runner object for the MG5 C++ Standalone output."""

    mg5_path = ""
    
    type='cpp'
    name = 'MG5-C++'
    compilator ='g++'

    def format_mg5_proc_card(self, proc_list, model, orders):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = "import model %s \n" % model
        v5_string += "set automatic_html_opening False\n"
        couplings = MERunner.get_coupling_definitions(orders)

        for i, proc in enumerate(proc_list):
            v5_string += 'add process ' + proc + ' ' + couplings + \
                         '@%i' % i + '\n'
        v5_string += "output standalone_cpp %s -f\n" % \
                     os.path.join(self.mg4_path, self.temp_dir_name)

        return v5_string

    def fix_energy_in_check(self, dir_name, energy):
        """Replace the hard coded collision energy in check_sa.f by the given
        energy, assuming a working dir dir_name"""

        file = open(os.path.join(dir_name, 'SubProcesses', 'check_sa.cpp'), 'r')
        check_sa = file.read()
        file.close()

        file = open(os.path.join(dir_name, 'SubProcesses', 'check_sa.cpp'), 'w')
        file.write(re.sub("energy = 1000", "energy = %i" % int(energy),
                          check_sa))
        file.close()

class PickleRunner(MERunner):
    """Runner object for the stored comparison results."""

    name = 'Stored result'
    model = ''
    orders = {}
    energy = 1000

#    def run(self, proc_list, model, orders, energy):
    def run(self, *arg, **opt):
        """Simulate a run by simply returning res_list
        """

        return self.res_list

    @staticmethod
    def find_comparisons(pickle_path, model = "", orders={}, energy=0,
                         proc_list = []):
        """Find a stored comparison object which corresponds to the
        parameters given. If file given in pickle_path, simply return
        pickled PickleRunner."""

        if os.path.isfile(pickle_path):
            # File given. Simply return pickle object from file
            pickle_runner = save_load_object.load_from_file(pickle_path)
            if isinstance(pickle_runner, PickleRunner):
                return [pickle_runner]
            else:
                return []

        if os.path.isdir(pickle_path):
            # Directory given. Return list of comparisons which
            # correspond to the parameters given

            object_list = []
            # NOT YET FINISHED
            for pickle_file in glob.glob(os.path.join(pickle_path,"*")):
                # Ignore directories
                if os.path.isdir(pickle_file):
                    continue
                # try loading a PickleRunner from the file
                try:
                    pickle_runner = save_load_object.load_from_file(pickle_file)
                    if isinstance(pickle_runner, PickleRunner):
                        object_list.append(pickle_runner)
                        logging.info("Loaded comparison runner from file %s" % \
                                     pickle_file)
                except:
                    pass
                object_list = filter(lambda runner:\
                                  (not model or runner.model == model) and \
                                  (not orders or runner.orders == orders) and \
                                  (not energy or runner.energy == energy) and \
                                  (not proc_list or \
                                   runner.proc_list == proc_list),
                                     object_list)

            return object_list

        raise IOError, "Path %s is not valid pickle directory" % \
              str(pickle_path)

    @staticmethod
    def store_comparison(pickle_path, proc_list, model, name,
                         orders={}, energy=1000):
        """Store a comparison corresponding to the parameters given."""

        new_runner = PickleRunner()
        new_runner.proc_list = proc_list[0]
        new_runner.res_list = proc_list[1]
        new_runner.model = model
        new_runner.name = "Stored " + name
        new_runner.orders = orders
        new_runner.energy = energy

        save_load_object.save_to_file(pickle_path, new_runner)

        logging.info("Stored comparison object in %s" % pickle_path)

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
            logging.info("Code %s added" % runner.name)

    def run_comparison(self, proc_list, model='sm', orders={}, energy=1000):
        """Run the codes and store results."""

        if isinstance(model, basestring):
            model= [model] * len(self.me_runners)

        self.results = []
        self.proc_list = proc_list

        logging.info(\
            "Running on %i processes with order: %s, in model %s @ %i GeV" % \
            (len(proc_list),
             ' '.join(["%s=%i" % (k, v) for k, v in orders.items()]),
             '/'.join([onemodel for onemodel in model]),
             energy))

        pass_proc = False
        for i,runner in enumerate(self.me_runners):
            cpu_time1 = time.time()
            logging.info("Now running %s" % runner.name)
            if pass_proc:
                runner.pass_proc = pass_proc 
            self.results.append(runner.run(proc_list, model[i], orders, energy))
            if hasattr(runner, 'new_proc_list'):
                pass_proc = runner.new_proc_list
            cpu_time2 = time.time()
            logging.info(" Done in %0.3f s" % (cpu_time2 - cpu_time1))
            logging.info(" (%i/%i with zero ME)" % \
                    (len([res for res in self.results[-1] if res[0][0] == 0.0]),
                     len(proc_list)))         

    def cleanup(self):
        """Call cleanup for each MERunner."""

        for runner in self.me_runners:
            logging.info("Cleaning code %s runner" % runner.name)
            runner.cleanup()

    def _fixed_string_length(self, mystr, length):
        """Helper function to fix the length of a string by cutting it 
        or adding extra space."""

        if len(mystr) > length:
            return mystr[0:length]
        else:
            return mystr + " " * (length - len(mystr))

    def output_result(self, filename=None, tolerance=3e-06, skip_zero=True):
        """Output result as a nicely formated table. If filename is provided,
        write it to the file, else to the screen. Tolerance can be adjusted."""

        proc_col_size = 17

        for proc in self.proc_list:
            if len(proc) + 1 > proc_col_size:
                proc_col_size = len(proc) + 1
        
        col_size = 17

        pass_proc = 0
        fail_proc = 0

        failed_proc_list = []

        res_str = "\n" + self._fixed_string_length("Process", proc_col_size) + \
                ''.join([self._fixed_string_length(runner.name, col_size) for \
                           runner in self.me_runners]) + \
                  self._fixed_string_length("Relative diff.", col_size) + \
                  "Result"

        for i, proc in enumerate(self.proc_list):
            list_res = [res[i][0][0] for res in self.results]
            if max(list_res) == 0.0 and min(list_res) == 0.0:
                diff = 0.0
                if skip_zero:
                    continue
            else:
                diff = (max(list_res) - min(list_res)) / \
                       (max(list_res) + min(list_res))

            res_str += '\n' + self._fixed_string_length(proc, proc_col_size)+ \
                       ''.join([self._fixed_string_length("%1.10e" % res,
                                               col_size) for res in list_res])

            res_str += self._fixed_string_length("%1.10e" % diff, col_size)

            if diff < tolerance:
                pass_proc += 1
                res_str += "Pass"
            else:
                fail_proc += 1
                failed_proc_list.append(proc)
                res_str += "Fail"

        res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                    (pass_proc, pass_proc + fail_proc,
                     fail_proc, pass_proc + fail_proc)

        if fail_proc != 0:
            res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)

        logging.info(res_str)

        if filename:
            file = open(filename, 'w')
            file.write(res_str)
            file.write(str(failed_proc_list))
            file.close()

    def get_non_zero_processes(self):
        """Return a list of processes which have non zero ME for at least
        one generator."""

        non_zero_proc = []
        non_zero_res = []
        
        for i, proc in enumerate(self.proc_list):
            list_res = [res[i] for res in self.results]
            if sum([abs(res[0][0]) for res in list_res]) != 0.0:
                non_zero_proc.append(proc)
                non_zero_res.append(list_res[0])

        return non_zero_proc, non_zero_res

    def assert_processes(self, test_object, tolerance = 1e-06):
        """Run assert to check that all processes passed comparison""" 

        col_size = 17
        fail_proc = 0
        fail_str = "Failed for processes:"
        for i, proc in enumerate(self.proc_list):
            list_res = [res[i][0][0] for res in self.results]
            if max(list_res) == 0.0 and min(list_res) == 0.0:
                diff = 0.0
            else:
                diff = (max(list_res) - min(list_res)) / \
                       (max(list_res) + min(list_res))

            if diff >= tolerance:
                fail_str += self._fixed_string_length('\n' + proc, col_size) + \
                            ''.join([self._fixed_string_length("%1.10e" % res,
                                                               col_size) for \
                                     res in list_res])
                
                fail_str += self._fixed_string_length("%1.10e" % diff, col_size)

        test_object.assertEqual(fail_str, "Failed for processes:")

class MEComparatorGauge(MEComparator):
    """Base object to run comparison tests. Take standard MERunner objects and
    a list of proc as an input and return detailed comparison tables in various
    formats."""
    
    def output_result(self, filename=None, tolerance=3e-06, skip_zero=True):
        """Output result as a nicely formated table. If filename is provided,
        write it to the file, else to the screen. Tolerance can be adjusted."""

        proc_col_size = 18

        for proc in self.proc_list:
            if len(proc) + 1 > proc_col_size:
                proc_col_size = len(proc) + 1
        
        col_size = 18

        pass_proc = 0
        fail_proc = 0

        failed_proc_list = []

        res_str = "\n" + self._fixed_string_length("Process", proc_col_size) + \
                ''.join([self._fixed_string_length(runner.name, col_size) for \
                           runner in self.me_runners]) + \
                  self._fixed_string_length("Diff both unit", col_size) + \
                  self._fixed_string_length("Diff both cms", col_size) + \
                  self._fixed_string_length("Diff both fixw", col_size) + \
                  self._fixed_string_length("Diff both feyn", col_size) + \
                  "Result"

        for i, proc in enumerate(self.proc_list):
            list_res = [res[i][0][0] for res in self.results]            
            
            diff_feyn = abs(list_res[1] - list_res[2]) / \
                       (list_res[1] + list_res[2] + 1e-99)
            diff_unit = abs(list_res[0] - list_res[3]) / \
                       (list_res[0] + list_res[3] + 1e-99)
            diff_cms = abs(list_res[0] - list_res[1]) / \
                       (list_res[0] + list_res[1] + 1e-99)
            diff_fixw = abs(list_res[2] - list_res[3]) / \
                       (list_res[2] + list_res[3] + 1e-99)

            res_str += '\n' + self._fixed_string_length(proc, proc_col_size)+ \
                       ''.join([self._fixed_string_length("%1.10e" % res,
                                               col_size) for res in list_res])

            res_str += self._fixed_string_length("%1.10e" % diff_unit, col_size)
            res_str += self._fixed_string_length("%1.10e" % diff_cms, col_size)
            res_str += self._fixed_string_length("%1.10e" % diff_fixw, col_size)
            res_str += self._fixed_string_length("%1.10e" % diff_feyn, col_size)
                        
            if diff_feyn < 1e-2 and diff_cms < 1e-6 and diff_fixw < 1e-3 and \
               diff_unit < 1e-2:
                pass_proc += 1
                res_str += "Pass"
            else:
                fail_proc += 1
                failed_proc_list.append(proc)
                res_str += "Fail"

        res_str += "\nSummary: %i/%i passed, %i/%i failed" % \
                    (pass_proc, pass_proc + fail_proc,
                     fail_proc, pass_proc + fail_proc)

        if fail_proc != 0:
            res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)

        logging.info(res_str)

        if filename:
            file = open(filename, 'w')
            file.write(res_str)
            file.write(str(failed_proc_list))
            file.close()

    def assert_processes(self, test_object, tolerance = 1e-06):
        """Run assert to check that all processes passed comparison""" 

        col_size = 17
        fail_proc = 0
        fail_str = ""
        for i, proc in enumerate(self.proc_list):
            list_res = [res[i][0][0] for res in self.results]
            if max(list_res) == 0.0 and min(list_res) == 0.0:
                diff = 0.0           
                continue
            
            diff_feyn = abs(list_res[1] - list_res[2]) / \
                       (list_res[1] + list_res[2] + 1e-99)
            diff_unit = abs(list_res[0] - list_res[3]) / \
                       (list_res[0] + list_res[3] + 1e-99)
            diff_cms = abs(list_res[0] - list_res[1]) / \
                       (list_res[0] + list_res[1] + 1e-99)
            diff_fixw = abs(list_res[2] - list_res[3]) / \
                       (list_res[2] + list_res[3] + 1e-99)
                       
            if diff_feyn > 1e-2 or diff_cms > 1e-6 or diff_fixw > 1e-3 or \
               diff_unit > 1e-2:          
                fail_str += proc+" "

        test_object.assertEqual(fail_str, "")    
    
def create_proc_list(part_list, initial=2, final=2, charge_conservation=True):
    """Helper function to automatically create process lists starting from 
    a particle list."""

    proc_list = []
    res_list = []
    for product in itertools.product(part_list, repeat=initial + final):
        sorted_product = sorted(product[0:initial]) + sorted(product[initial:])
        if  sorted_product not in proc_list:
            proc_list.append(sorted_product)

    for proc in proc_list:
        #check charge conservation
        if charge_conservation:
            init_plus= ''.join(proc[:initial]).count('+')
            init_minus=''.join(proc[:initial]).count('-')
            final_plus=''.join(proc[initial:]).count('+')
            final_minus=''.join(proc[initial:]).count('-')
            if init_plus-init_minus-final_plus+final_minus:
                continue
        proc.insert(initial, '>')
        res_list.append(' '.join(proc))

    return res_list

def create_proc_list_enhanced(init_part_list, final_part_list_1,
                              final_part_list_2 = [], initial=2, final_1=2,
                              final_2=1, charge_conservation=True):
    """Helper function to automatically create process lists starting from 
    a particle list."""

    proc_list = []
    res_list = []
    for iprod in itertools.product(init_part_list, repeat=initial):
        for fprod1 in itertools.product(final_part_list_1, repeat=final_1):
            if final_part_list_2:
                for fprod2 in itertools.product(final_part_list_2,
                                                repeat=final_2):                
                    sorted_product = sorted(iprod) + sorted(fprod1 + fprod2)
                    if  sorted_product not in proc_list:
                        proc_list.append(sorted_product)
            else:
                sorted_product = sorted(iprod) + sorted(fprod1)

                if  sorted_product not in proc_list:
                    proc_list.append(sorted_product)

    for proc in proc_list:
        #check charge conservation
        if charge_conservation:
            init_plus= ''.join(proc[:initial]).count('+')
            init_minus=''.join(proc[:initial]).count('-')
            final_plus=''.join(proc[initial:]).count('+')
            final_minus=''.join(proc[initial:]).count('-')
            if init_plus-init_minus-final_plus+final_minus:
                continue
        proc.insert(initial, '>')
        res_list.append(' '.join(proc))

    return res_list


def create_proc_list_2_3(init_part_list1, 
                         init_part_list2, 
                         final_part_list_1,
                         final_part_list_2,
                         final_part_list_3,
                         charge_conservation=True):
    """Helper function to automatically create process lists starting from 
    a particle list."""

    proc_list = []
    res_list = []
    for i,a in enumerate(init_part_list1):
        for b in init_part_list2:
            for c in final_part_list_1:
                for d in final_part_list_2:
                    for e in final_part_list_3:
                        proc = [a,b,'>',c,d,e]
                        res_list.append(' '.join(proc))

    return res_list

