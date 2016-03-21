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
"""A set of objects to allow for easy comparisons of results for loop processes
from various ME generators (e.g., MadLoop v5 against v4, ...) and output nice 
reports in different formats (txt, tex, ...).
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
import re
import operator
import math

pjoin = os.path.join
# Get the grand parent directory (mg5 root) of the module real path 
# (tests/acceptance_tests) and add it to the current PYTHONPATH to allow
# for easy import of MG5 tools

_file_path = os.path.dirname(os.path.realpath(__file__))

import madgraph.iolibs.template_files as template_files
import madgraph.various.misc as misc
import madgraph.iolibs.save_load_object as save_load_object

import madgraph.interface.master_interface as cmd_interface
from madgraph.interface.madevent_interface import MadLoopInitializer

import madgraph.various.process_checks as process_checks
import madgraph.various.misc as misc
import madgraph.various.banner as banner_mod

import me_comparator
from madgraph.iolibs.files import mv

from madgraph import MadGraph5Error, MG5DIR

class LoopPickleRunner(me_comparator.PickleRunner):
    """Loop Pickle Runner with special features in run() for loops"""

    # Simulate a run
    def run(self, proc_list, model, energy=1000, PSpoints=[], **opts):
        
        # Make sure the comparison is valid (for now, ignore the restrict card)
        if self.energy != energy or self.model.split('-')[0]!=model.split('-')[0]:
            return [((0.0, 0.0, 0.0, 0.0, 0), [])] * len(proc_list)
        
        res=[]
        # Order the processes as asked for
        for proc in proc_list:
            try:
                ind=self.proc_list.index(proc)
                # Make sure the PS point is the same
                if PSpoints==[] or PSpoints[ind]==[] or \
                                          self.res_list[ind][-1]==PSpoints[ind]:
                    res.append(self.res_list[ind])
                else:
                    res.append(((0.0, 0.0, 0.0, 0.0, 0), []))
            except ValueError:
                res.append(((0.0, 0.0, 0.0, 0.0, 0), []))
        return res

    @staticmethod
    def store_comparison(pickle_path, proc_list, model, name,
                         orders={}, energy=1000):
        """Store a comparison corresponding to the parameters given."""
    
        new_runner = LoopPickleRunner()
        new_runner.proc_list = proc_list[0]
        new_runner.res_list = proc_list[1]
        new_runner.model = model
        new_runner.name = "Stored " + name
        new_runner.orders = orders
        new_runner.energy = energy
    
        save_load_object.save_to_file(pickle_path, new_runner)
    
        logging.info("Stored comparison object in %s" % pickle_path)

class LoopMG5RunnerError(Exception):
        """class for error in LoopMG5Runner"""
        pass

class LoopMG5Runner(me_comparator.MG5Runner):
    """Runner object for the MG5 Matrix Element generator for loop processes."""

    mg5_path = ""
    optimized_output = True

    name = 'ML5 opt'
    type = 'MLv5_opt'
    compilator ='gfortran'
    mu_r_value = 0.0

    def setup(self, mg5_path, optimized_output=True, temp_dir=None,mu_r=0.0):
        """Initialization of the temp directory"""

        self.mg5_path = os.path.abspath(mg5_path)
        self.optimized_output = optimized_output
        if not self.optimized_output:
            self.name = 'ML5 default'
            self.type = 'MLv5_default'
        
        if not temp_dir:
            temp_dir = "ptest_%s"%self.type
        else:
            temp_dir = temp_dir+('_opt' if self.optimized_output else '_default')
        base_dir = temp_dir
        i=0
        while os.path.exists(os.path.join(mg5_path,temp_dir)):
            i += 1
            temp_dir = base_dir + '_%d'%i
        self.mu_r_value = mu_r
        self.temp_dir_name = temp_dir

    def run(self, proc_list, model, energy=1000, PSpoints=[]):
        """Execute MG5 on the list of processes mentioned in proc_list, using
        the specified model and the specified orders with a given c.o.m energy
        or the given Phase-Space points if specified.
        """
        self.res_list = [] # ensure that to be void, and avoid pointer problem 
        self.proc_list = proc_list
        self.model = model
        self.energy = energy
        self.non_zero = 0 

        dir_name = os.path.join(self.mg5_path, self.temp_dir_name)
        # Create a proc_card.dat in the v5 format
        proc_card_location = os.path.join(self.mg5_path, 'proc_card_%s.dat' % \
                                          self.temp_dir_name)
        proc_card_file = open(proc_card_location, 'w')
        proc_card_file.write(self.format_mg5_proc_card(proc_list, model))
        proc_card_file.close()
        logging.info("proc_card.dat file for %i processes successfully created in %s" % \
                     (len(proc_list), os.path.join(dir_name, 'Cards')))
        # Run mg5
        logging.info("Running mg5")
        proc_card = open(proc_card_location, 'r').read()
        new_proc_list = []
        cmd = cmd_interface.MasterCmd()
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

        # Move the proc_card to the created folder
        mv(proc_card_location,os.path.join(dir_name,'Cards','proc_card_mg5.dat'))
        if self.non_zero:
            # Initialize the processes (i.e. HelFilter)
            initializations = []
            for i, proc in enumerate(proc_list):
                init = LoopMG5Runner.initialize_process(\
                      proc,i,os.path.join(self.mg5_path,self.temp_dir_name))
                initializations.append(init)
            self.fix_MadLoopParamCard(dir_name, reduction_mode=('1' if self.optimized_output else '1'))
            if PSpoints==[]:
                self.fix_energy_in_check(dir_name, energy)          

            # Get the ME value
            for i, proc in enumerate(proc_list):
                if initializations[i]:
                    value = LoopMG5Runner.get_me_value(proc, i, os.path.join(\
                                              self.mg5_path,self.temp_dir_name), 
                                          ([] if PSpoints==[] else PSpoints[i]),\
                                          mu_r=self.mu_r_value)
                    self.res_list.append(value)
                else:
                    self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))

            return self.res_list
        else:
            self.res_list = [((0.0, 0.0, 0.0, 0.0, 0), [])] * len(proc_list)
            return self.res_list

    def format_mg5_proc_card(self, proc_list, model):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = "set loop_optimized_output %s\n"%str(self.optimized_output) 

        v5_string += "import model %s\n" % os.path.join(self.model_dir, model)
        
        # The squared order couplings can be specified via the dictionary
        # squared_orders using either of the two syntax:
        # {'QCD':2,'QED':4} or {'QCD^2<=':2,'QED^2<=':4}
        # The latter syntax allowing for specifying the comparator. 
        # The reg. exp. below makes sure one can separate the two syntaxes.       
        sq_order_re = re.compile(
          r"^\s*(?P<coup_name>\w*)\s*\^2\s*(?P<logical_operator>(==)|(<=)|=|>)")
        
        for i, (proc, born_orders, perturbation_orders, squared_orders) in \
            enumerate(proc_list):      
            
            born_couplings = ' '.join(["%s=%i" % (k, v) for k, v \
                                   in born_orders.items()])
            perturbations = ' '.join([k for k \
                                   in perturbation_orders])
            
            squared_couplings = []
            for coup, value in squared_orders.items():
                parsed = sq_order_re.match(coup)
                if not parsed is None:
                    operator = parsed.group('logical_operator')
                    coup_name = parsed.group('coup_name')
                else:
                    operator = '='
                    coup_name = coup
                squared_couplings.append('%s^2%s%i'% (coup_name,operator,value))
            squared_couplings = ' '.join(squared_couplings)    
            v5_string += 'add process ' + proc + ' ' + born_couplings + \
                         ' [virt=' + perturbations + '] ' + squared_couplings + \
                         (' @%i\n'%i)
        v5_string += "output standalone %s -f\n"%\
                     os.path.join(self.mg5_path, self.temp_dir_name)
        return v5_string

    @staticmethod
    def initialize_process(proc, proc_id, working_dir, verbose=True):
        """Compile and run ./check, then parse the output and return the result
        for process with id = proc_id and PSpoint if specified."""  
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
         
        shell_name = None
        directories = glob.glob(os.path.join(working_dir, 'SubProcesses',
                                                             'P%i_*' % proc_id))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s"%str(proc))
            return False 

        if verbose: logging.info("Initializing process %s in dir %s"%(str(proc), shell_name))
        
        dir_name = os.path.join(working_dir, 'SubProcesses', shell_name)

        init = MadLoopInitializer.run_initialization(\
          run_dir=dir_name, SubProc_dir=os.path.join(working_dir, 'SubProcesses'),
          attempts = [3,15,25])
       
        if init is None:
            return False
        else:
            return True

    @staticmethod
    def get_me_value(proc, proc_id, working_dir, PSpoint=[], verbose=True,mu_r=0.0):
        """Compile and run ./check, then parse the output and return the result
        for process with id = proc_id and PSpoint if specified.""" 
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
         
        shell_name = None
        directories = glob.glob(os.path.join(working_dir, 'SubProcesses',
                                  'P%i_*' % proc_id))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s"%str(proc))
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

        if verbose: logging.info("Working on process %s in dir %s"%(str(proc), shell_name))
        
        dir_name = os.path.join(working_dir, 'SubProcesses', shell_name)
        LoopMG5Runner.fix_PSPoint_in_check(dir_name, PSpoint!=[],mu_r=mu_r)
        # Make sure the modified source file are recompiled
        if os.path.isfile(os.path.join(dir_name,'check')):
            os.remove(os.path.join(dir_name,'check'))
        if os.path.isfile(os.path.join(dir_name,'check_sa.o')):        
            os.remove(os.path.join(dir_name,'check_sa.o'))
        # Run make
        devnull = open(os.devnull, 'w')
        retcode = subprocess.call(['make','check'],
                        cwd=dir_name,
                        stdout=devnull, stderr=devnull)
                        
        if retcode != 0:
            logging.info("Error while executing make in %s"%shell_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

        # If a PS point is specified, write out the corresponding PS.input
        if PSpoint:
            misc.write_PS_input(os.path.join(dir_name, 'PS.input'),PSpoint)
        
        # Run ./check
        try:
            output = subprocess.Popen('./check',
                        cwd=dir_name,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            output.read()
            output.close()
            if os.path.exists(os.path.join(dir_name,'result.dat')):
                return LoopMG5Runner.parse_check_output(file(dir_name+'/result.dat'))  
            else:
                logging.warning("Error while looking for file %s"%str(os.path\
                                                  .join(dir_name,'result.dat')))
                return ((0.0, 0.0, 0.0, 0.0, 0), [])
        except IOError:
            logging.warning("Error while executing ./check in %s" % shell_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

    @staticmethod
    def parse_check_output(output):
        """Parse the output string and return a pair where first four values are 
        the finite, born, single and double pole of the ME and the fourth is the
        GeV exponent and the second value is a list of 4 momenta for all particles 
        involved."""

        res_p = []
        value = [0.0,0.0,0.0,0.0]
        gev_pow = 0

        for line in output:
            splitline=line.split()
            if splitline[0]=='PS':
                res_p.append([float(s) for s in splitline[1:]])
            elif splitline[0]=='BORN':
                value[1]=float(splitline[1])
            elif splitline[0]=='FIN':
                value[0]=float(splitline[1])
            elif splitline[0]=='1EPS':
                value[2]=float(splitline[1])
            elif splitline[0]=='2EPS':
                value[3]=float(splitline[1])
            elif splitline[0]=='EXP':
                gev_pow=int(splitline[1])

        return ((value[0],value[1],value[2],value[3],gev_pow), res_p)

    def cleanup(self):
        """Clean up temporary directories"""

        #if not self.setup_flag:
        #    raise self.MERunnerException, \
        #            "MERunner setup should be called first"
        try:
            if os.path.isdir(os.path.join(self.mg5_path, self.temp_dir_name)):
                shutil.rmtree(os.path.join(self.mg5_path, self.temp_dir_name))
                logging.info("Temporary standalone directory %s successfully removed" % \
                     self.temp_dir_name)
        except:
            pass

    @staticmethod
    def fix_PSPoint_in_check(check_sa_dir, read_ps = True,mu_r=0.0):
        """Set check_sa.f to be reading PS.input assuming a working dir dir_name"""
        
        file_path = os.path.join(check_sa_dir,'check_sa.f')
        if not os.path.isfile(file_path):
            raise MadGraph5Error('Could not find check_sa.f in path %s.'%str(file_path))

        file = open(file_path, 'r')
        check_sa = file.read()
        file.close()

        file = open(file_path, 'w')
        check_sa = re.sub(r"READPS = \S+\)","READPS = %s)"%('.TRUE.' if read_ps \
                                                      else '.FALSE.'), check_sa)
        check_sa = re.sub(r"NPSPOINTS = \d+","NPSPOINTS = 1", check_sa)
        if mu_r > 0.0:
            check_sa = re.sub(r"MU_R=SQRTS","MU_R=%s"%\
                                        (("%.17e"%mu_r).replace('e','d')),\
                                        check_sa)
        elif mu_r < 0.0:
            check_sa = re.sub(r"MU_R=SQRTS","",check_sa)
                    
        file.write(check_sa)
        file.close()
        
    @staticmethod
    def fix_MadLoopParamCard(dir_name,mp=False,reduction_mode='1'):
        """ Set parameters in MadLoopParams.dat suited for these checks."""

        MLCardPath = os.path.join(dir_name,'Cards','MadLoopParams.dat')
        MLOutPath  = os.path.join(dir_name,'SubProcesses','MadLoopParams.dat')
        MLCard = banner_mod.MadLoopParam()
        MLCard['CTModeRun'] = 4 if mp else -1
        MLCard['CTModeInit'] = 4 if mp else 1
        MLCard['MLReductionLib'] = reduction_mode 
        MLCard['UseLoopFilter'] = False
        MLCard['DoubleCheckHelicityFilter'] = False
        MLCard.write(MLCardPath)
        MLCard.write(MLOutPath)
        
class LoopMG5Runner_gauge(LoopMG5Runner):
    
    #name = 'MG5_gauge'
    #type = 'ufo_cms'
    
    def __init__(self, cms, gauge):
        self.cms = cms
        self.gauge = gauge
        self.type =  '%s_%s' %(self.cms, self.gauge)
        self.name =  'MG5_%s_%s' %(self.cms, self.gauge)
    
    def format_mg5_proc_card(self, proc_list, model):
        """Create a proc_card.dat string following v5 conventions."""

        v5_string = "set loop_optimized_output %s\n"%str(self.optimized_output) 

        v5_string += "import model %s\n" % os.path.join(self.model_dir, model)
        v5_string += "set automatic_html_opening False\n"
        v5_string += 'set complex_mass_scheme %s \n' % self.cms
        v5_string += 'set gauge %s \n' % self.gauge
        
        for i, (proc, born_orders, perturbation_orders, squared_orders) in \
            enumerate(proc_list):      
            
            born_couplings = ' '.join(["%s=%i" % (k, v) for k, v \
                                   in born_orders.items()])
            perturbations = ' '.join([k for k \
                                   in perturbation_orders])

            squared_couplings = ' '.join(["%s=%i" % (k, v) for k, v \
                                   in squared_orders.items()])      
            v5_string += 'add process ' + proc + ' ' + born_couplings + \
                         ' [virt=' + perturbations + '] ' + squared_couplings + \
                         (' @%i\n'%i)
        v5_string += "output standalone %s -f\n"%\
                     os.path.join(self.mg5_path, self.temp_dir_name)
                     
        v5_string += 'set complex_mass_scheme False \n'
        v5_string += 'set gauge unitary \n'
        
        return v5_string

class LoopMG4RunnerError(Exception):
        """class for error in LoopMG4Runner"""
        pass

class LoopMG4Runner(me_comparator.MERunner):
    """Runner object for the MadLoop4 Matrix Element generator for loop processes."""

    mg4_path = ""

    name = 'MadLoop v4'
    type = 'MLv4'
    compilator ='gfortran'

    def setup(self, mg4_path, temp_dir=None):
        """Initialization of the temporary directory"""

        self.mg4_path = os.path.abspath(mg4_path)

        if not temp_dir:
            i=0
            while os.path.exists(os.path.join(mg4_path, 
                                              "ptest_%s_%s" % (self.type, i))):
                i += 1
            temp_dir = "ptest_%s_%s" % (self.type, i)         

        self.temp_dir_name = temp_dir

    def run(self, proc_list, model, energy=1000, PSpoints=[]):
        """Execute MadLoop4 on the list of processes mentioned in proc_list, using
        the specified model and the specified orders with a given c.o.m energy
        or the given Phase-Space points if specified.
        """
        self.res_list = [] # ensure that to be void, and avoid pointer problem 
        self.proc_list = proc_list
        self.model = model
        self.energy = energy

        dir_name = os.path.join(self.mg4_path, self.temp_dir_name)

        # Get the ME value for each process
        for i, (proc, born_orders, perturbation_orders, squared_orders) in \
            enumerate(proc_list):
            self.orders = (born_orders,perturbation_orders,squared_orders)
            # Make sure that MadLoop4 can handle this process
            if not self.filter_process(proc, born_orders, perturbation_orders, \
                                       squared_orders):
                logging.warning("The process %s cannot be handled by MadLoop4," % proc)
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))          
            # Create a proc_card.dat in the MadLoop4 format
            proc_card_location = os.path.join(self.mg4_path, 'Cards', 
                                    'order.lh')
            proc_card_file = open(proc_card_location, 'w')
            card, proc_name = self.format_ml4_proc_card(i,proc, model,\
                born_orders)
            proc_card_file.write(card)
            proc_card_file.close()
            # Run MadLoop4
            logging.info("Running MadLoop4 for process %s"%proc)
            devnull = open(os.devnull, 'w')
            retcode = subprocess.call(['./bin/newprocess_nlo','-nocompile'],
                    cwd=self.mg4_path,
                    stdout=devnull, stderr=devnull)
                    
            if retcode != 0:
                logging.info("Error while running MadLoop4 in %s" % self.mg4_path)
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))
            
            if os.path.isdir(os.path.join(self.mg4_path,'NLO_'+proc_name)):
                shutil.move(os.path.join(self.mg4_path,'NLO_'+proc_name),
                            os.path.join(dir_name,proc_name))
            else:
                logging.info("Could not find produced directory %s" %str(
                              os.path.join(self.mg4_path,'NLO_'+proc_name)))
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))
            value = self.get_me_value(proc, i, energy, ([] if PSpoints==[] \
                                                else PSpoints[i]))
            self.res_list.append(value)
                                
        return self.res_list
        
    def format_ml4_proc_card(self, procID, proc, model, born_orders):
        """Create a les-houches order file, like in MadLoop4"""

        particle_dictionary = {'d':(1,'d'),'d~':(-1,'dx'),
                               'u':(2,'u'),'u~':(-2,'ux'),
                               's':(3,'s'),'s~':(-3,'sx'),
                               'c':(4,'c'),'c~':(-4,'cx'),
                               'b':(5,'b'),'b~':(-5,'bx'),
                               't':(6,'t'),'t~':(-6,'tx'),
                               'g':(21,'g'),
                               'a':(22,'a'),
                               'z':(23,'z'),
                               'h':(25,'h'),
                               'w+':(24,'wp'),'w-':(-24,'wm'),
                               'e+':(-11,'ep'),'e-':(11,'em'),
                               'mu+':(-13,'mup'),'mu-':(13,'mum'),
                               'ta+':(-15,'tap'),'ta-':(15,'tam'),
                               've':(12,'ve'),'ve~':(-12,'vex'),
                               'vm':(14,'vm'),'vm~':(-14,'vmx'),
                               'vt':(16,'vt'),'vt~':(-16,'vtx')}

        order_file ="MatrixElementSquareType CHsummed\n"
        order_file+="IRregularisation        CDR\n"
        order_file+="ModelFile               ../../Cards/param_card.dat\n"        
        order_file+="SubdivideSubprocess     No\n"
        order_file+="AlphasPower             %(QCD)i\n"
        order_file+="AlphaPower              %(QED)i\n"
        order_file+="OutputBaseName          %(BaseName)s\n"
        order_file+="CorrectionType          QCD\n"
        order_file+="LoopParticles           1 -1 2 -2 3 -3 4 -4 5 -5 6 -6 21\n"
        order_file+="PhysicsModel            %(ModelName)s\n"
        order_file+="\n%(Process)s"        
        
        replace_dict={}
        # QCD order, 99 by default
        if 'QCD' in born_orders.keys():
            replace_dict['QCD']=born_orders['QCD']
        else:
            replace_dict['QCD']=99
            
        # The exact QED order at born level must be supplied. If it is not
        # present it is then assumed to be zero
        if 'QED' in born_orders.keys():
            replace_dict['QED']=born_orders['QED']
        else:
            replace_dict['QED']=0
        
        proc_parts=proc.split()
        if '>' in proc_parts:
            incoming_parts=proc_parts[:proc_parts.index('>')]
            outcoming_parts=proc_parts[proc_parts.index('>')+1:]            
        else:
            raise self.LoopMG4RunnerError, \
                        "The process %s is ill-formated."%proc
        
        for part in incoming_parts+outcoming_parts:
            if part not in particle_dictionary.keys():
                raise Exception, \
                        "Particle %s is not recognized by MadLoop4."%part                
        
        proc_name ='P%i_'%procID+''.join([particle_dictionary[inc][1] for \
                                   inc in incoming_parts])+'_'+ \
                   ''.join([particle_dictionary[out][1] for out in \
                            outcoming_parts])
        replace_dict['BaseName']=proc_name
        
        # MadLoop4 can only work with the smNLO (loop Standard Model) which was
        # copied in a stone-fixed version called ML5_parallel_test.
        replace_dict['ModelName']='ML5_parallel_test'

        replace_dict['Process']=' '+' '.join([str(particle_dictionary[inc][0]) \
                                              for inc in incoming_parts])+' -> '\
                                +' '.join([str(particle_dictionary[out][0]) \
                                              for out in outcoming_parts])

        return order_file%replace_dict, proc_name

    def get_me_value(self, proc, proc_id, energy, PSpoint=[]):
        """Compile and run ./NLOComp_sa, then parse the output and return the
        result for process with id = proc_id and PSpoint if specified."""

        sys.stdout.write('.')
        sys.stdout.flush()
        working_dir = os.path.join(self.mg4_path, self.temp_dir_name)
         
        shell_name = None
        directories = glob.glob(os.path.join(working_dir,'P%i_*' %proc_id))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s" %proc)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

        logging.info("Working on process %s in dir %s" %(proc, shell_name))
        
        dir_name = os.path.join(working_dir, shell_name, 'SigVirt')
        
        if not os.path.isfile(os.path.join(dir_name,'NLOComp_sa.f')):
            logging.info("File NLOComp_sa.f hasn't been created for process %s" %proc)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])
        
        if PSpoint==[]:
            self.fix_energy_in_check(dir_name, energy)
        else:
            self.fix_PSPoint_in_check(dir_name)

        # If a PS point is specified, write out the corresponding PS.input
        if PSpoint!=[]:
            misc.write_PS_input(os.path.join(dir_name, 'PS.input'),PSpoint)
        
        # Run make
        devnull = open(os.devnull, 'w')
        retcode = subprocess.call(['make','NLOComp_sa'],
                        cwd=dir_name,
                        stdout=devnull, stderr=devnull)
                        
        if retcode != 0:
            logging.info("Error while executing make in %s" % dir_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])
        
        # Run ./check for the finite part
        answer=[]
        self.change_finite_single(dir_name,'first')
        try:
            output = subprocess.Popen('./NLOComp_sa',
                        cwd=dir_name,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            output.read()
            output.close()
            if os.path.exists(os.path.join(dir_name,'result.dat')):
                buff=self.parse_check_output(file(dir_name+'/result.dat'))
                answer=[list(buff[0]),buff[1]]
            else:
                logging.warning("Error while looking for file %s"%str(os.path\
                                                  .join(dir_name,'result.dat')))
                return ((0.0, 0.0, 0.0, 0.0, 0), [])
        except IOError:
            logging.warning("Error while executing ./check in %s" % shell_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])
        
        # Now ./check for the single part
        self.change_finite_single(dir_name,'singlePole')
        try:
            output = subprocess.Popen('./NLOComp_sa',
                        cwd=dir_name,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            output.read()
            output.close()
            if os.path.exists(os.path.join(dir_name,'result.dat')):
                answer[0][2]=self.parse_check_output(file(dir_name+'/result.dat'))[0][2]  
            else:
                logging.warning("Error while looking for file %s"%str(os.path\
                                                  .join(dir_name,'result.dat')))
                return ((0.0, 0.0, 0.0, 0.0, 0), [])
        except IOError:
            logging.warning("Error while executing ./check in %s" % shell_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])
        
        return (tuple(answer[0]),answer[1])

    def parse_check_output(self, output):
        """Parse the output string and return a pair where first four values are 
        the finite, born, single and double pole of the ME and the fourth is the
        GeV exponent and the second value is a list of 4 momenta for all particles 
        involved."""

        res_p = []
        value = [0.0,0.0,0.0,0.0]
        gev_pow = 0

        for line in output:
            splitline=line.split()
            if splitline[0]=='PS':
                res_p.append([float(s) for s in splitline[1:]])
            elif splitline[0]=='BORN':
                value[1]=float(splitline[1])
            elif splitline[0]=='FIN':
                value[0]=float(splitline[1])
            elif splitline[0]=='1EPS':
                value[2]=float(splitline[1])
            elif splitline[0]=='2EPS':
                value[3]=float(splitline[1])
            elif splitline[0]=='EXP':
                gev_pow=int(splitline[1])

        return ((value[0],value[1],value[2],value[3],gev_pow), res_p)

    def cleanup(self):
        """Clean up temporary directories"""

        try:
            if os.path.isdir(os.path.join(self.mg4_path, self.temp_dir_name)):
                shutil.rmtree(os.path.join(self.mg4_path, self.temp_dir_name))
                logging.info("Temporary standalone directory %s successfully removed" % \
                     self.temp_dir_name)
        except:
            pass

    def filter_process(self,proc,born_orders,perturbation_orders,squared_orders):
        """ Returns true if the process does not violate any limitation of ML4."""
        
        proc_parts=proc.split()
        if '>' in proc_parts:
            incoming_parts=proc_parts[:proc_parts.index('>')]
            outcoming_parts=proc_parts[proc_parts.index('>')+1:]            
        else:
            raise self.LoopMG4RunnerError, \
                        "The process %s is ill-formated."%proc
        
        # Check that the process only considers QCD perturbations
        if perturbation_orders!=['QCD']:
            logging.info("Only QCD perturbations are handled by ML4.")
            return False
        
        # Check that the process does not only contains gluons.
        if set(incoming_parts+outcoming_parts)==set(['g']):
            logging.info("The process %s only contains gluons." % proc)
            return False
                
        # Check that QCD fermion number is respected (ML4 not protected agains this)
        fermions=['d','u','c','s','b','t']
        antifermions=[fermion+'~' for fermion in fermions]
        if(len([1 for f in incoming_parts if f in fermions])-\
           len([1 for f in incoming_parts if f in antifermions])+\
           len([1 for f in outcoming_parts if f in antifermions])-\
           len([1 for f in outcoming_parts if f in fermions]))!=0:
             logging.info("The process %s does not conserve QCD fermion number." % proc)
             return False
         
        # Check that there is no 4-gluon vertices
        if (len([1 for f in incoming_parts+outcoming_parts if f in fermions+\
                antifermions])+len([1 for p in incoming_parts+outcoming_parts \
                if p=='g'])*2)>=8:
            logging.warning("The process %s most likely contains a 4-gluon vertex." % proc)
            # Still, maybe the user want to check the single pole
            return True
        
        return True

    def fix_PSPoint_in_check(self, dir_name):
        """Set check_sa.f to be reading PS.input assuming a working dir dir_name"""

        file = open(os.path.join(dir_name,'NLOComp_sa.f'), 'r')
        nlocomp_sa = file.read()
        file.close()

        file = open(os.path.join(dir_name,'NLOCOmp_sa.f'), 'w')
        file.write(re.sub("FixedPs = .false.", "FixedPs = .true.", nlocomp_sa))
        file.close()

    def fix_energy_in_check(self, dir_name, energy):
        """Replace the hard coded collision energy in check_sa.f by the given
        energy, assuming a working dir dir_name"""

        file = open(os.path.join(dir_name,'NLOComp_sa.f'), 'r')
        check_sa = file.read()
        file.close()

        file = open(os.path.join(dir_name,'NLOComp_sa.f'), 'w')
        file.write(re.sub("SQRTS=1000d0", "SQRTS=%id0" % int(energy), check_sa))        
        file.close()

    def change_finite_single(self,dir_name,mode):
        """ Replace in HelasNLO.inc and MadLoop.param the logical controlling
        whether to get the finite part or the single pole correct """
        
        file = open(os.path.join(dir_name,'MadLoop.param'), 'r')
        MadLoopParam = file.read()
        file.close()

        file = open(os.path.join(dir_name,'MadLoop.param'), 'w')        
        if mode=='first':
            file.write(re.sub("###MODE###\ndefaultRun\n", "###MODE###\nsinglePSCheck\nfinite\n", MadLoopParam))
        if mode=='singlePole':
            file.write(re.sub("###MODE###\nsinglePSCheck\nfinite\n", "###MODE###\nsinglePSCheck\nsinglePole\n", MadLoopParam))
        if mode=='finite':
            file.write(re.sub("###MODE###\nsinglePSCheck\nsinglePole\n", "###MODE###\nsinglePSCheck\nfinite\n", MadLoopParam))
        file.close()

        if mode=='first':
            return

        file = open(os.path.join(dir_name,'HelasNLO.input'), 'r')
        HelasNLO = file.read()
        file.close()

        file = open(os.path.join(dir_name,'HelasNLO.input'), 'w')
        if mode=='finite':
            file.write(re.sub("#finiteCT\n.false.\n", "#finiteCT\n.true.\n", HelasNLO))
        if mode=='singlePole':
            file.write(re.sub("#finiteCT\n.true.\n", "#finiteCT\n.false.\n", HelasNLO))
        file.close()

class LoopGoSamRunnerError(Exception):
        """class for error in LoopGoSamRunner"""
        pass

class GoSamRunner(me_comparator.MERunner):
    """Runner object for the GoSam Matrix Element generator for loop processes."""

    GoSam_path = ""

    name = 'GoSam v1.0'
    type = 'GoSam'
    compilator ='gfortran'
    use_dred = True

    def setup(self, GoSam_path, temp_dir=None):
        """Initialization of the temporary directory"""

        self.GoSam_path = os.path.abspath(GoSam_path)

        if not temp_dir:
            i=0
            while os.path.exists(os.path.join(GoSam_path, 
                                              "ptest_%s_%s" % (self.type, i))):
                i += 1
            temp_dir = "ptest_%s_%s" % (self.type, i)         

        self.temp_dir_name = temp_dir
        shutil.os.mkdir(os.path.join(self.GoSam_path, self.temp_dir_name))

    def run(self, proc_list, model, energy=1000, PSpoints=[]):
        """Execute GoSam on the list of processes mentioned in proc_list, using
        the specified model and the specified orders with a given c.o.m energy
        or the given Phase-Space points if specified.
        """
        
        self.res_list = [] # ensure that to be void, and avoid pointer problem 
        self.proc_list = proc_list
        self.model = model
        self.energy = energy

        dir_name = os.path.join(self.GoSam_path, self.temp_dir_name)

        # Get the ME value for each process
        for i, (proc, born_orders, perturbation_orders, squared_orders) in \
            enumerate(proc_list):
            self.orders = (born_orders,perturbation_orders,squared_orders)
            # Create the GoSam proc_card.in
            logging.info("Running GoSam to generate template proc card. %s"%proc)
            devnull = open(os.devnull, 'w')
            retcode = subprocess.call(['gosam.py','--template','myproc.in'],
                    cwd=dir_name,
                    stdout=devnull, stderr=devnull)
            if retcode != 0:
                logging.info("Error while running gosam.py --template in %s" % dir_name)
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))
            proc_card_location = os.path.join(dir_name,'myproc.in')
            file = open(proc_card_location, 'r')
            order_specified, proc_card_out, proc_name = \
                self.format_gosam_proc_card(i,proc, model, born_orders, \
                perturbation_orders, squared_orders, file)
            if not order_specified:
                logging.info("Error because GoSam needs the precise order "+\
                             +"specifications for process %s"%proc)
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))
            file.close()
            proc_card_file = open(proc_card_location, 'w')
            proc_card_file.write(proc_card_out)
            proc_card_file.close()
            # Run GoSam
            proc_dir=os.path.join(dir_name, proc_name)
            shutil.os.mkdir(proc_dir)
            logging.info("Running GoSam for process %s"%proc)
            devnull = open(os.devnull, 'w')
            retcode = subprocess.call(['gosam.py','myproc.in'],
                    cwd=dir_name,
                    stdout=devnull, stderr=devnull)
            devnull.close()        
            if retcode != 0:
                logging.info("Error while running GoSam in %s" % dir_name)
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))
            
            if os.path.isfile(os.path.join(dir_name,'myproc.in')):
                shutil.move(os.path.join(dir_name,'myproc.in'),
                            os.path.join(proc_dir,'myproc.in'))
            else:
                logging.info("Could not find produced order file %s" %str(
                              os.path.join(dir_name,'myproc.in')))
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))
            if 'QCD' not in born_orders.keys():
                logging.info("For GoSam run, the born QCD order must be provided.")
                self.res_list.append(((0.0, 0.0, 0.0, 0.0, 0), []))
                
            value = self.get_me_value(proc, i, energy, ([] if PSpoints==[] \
                                        else PSpoints[i]),born_orders['QCD'])
            self.res_list.append(value)
                                
        return self.res_list
    
    def dred_to_cdr(self,proc):
        """ Give the fraction (in float) to be added to the dred result to 
        match the cdr one. It is assumed that the factor aso2pi*Born is factored
        out """
        
        conversion_factor=float(0.0)
        
        gluons = ['g']
        fermions=['d','u','c','s','b','t']
        antifermions=[fermion+'~' for fermion in fermions]
        
        proc_parts=proc.split()
        if '>' in proc_parts:
            incoming_parts=proc_parts[:proc_parts.index('>')]
            outcoming_parts=proc_parts[proc_parts.index('>')+1:]            
        else:
            raise self.LoopGoSamRunnerError, \
                        "The process %s is ill-formated."%proc
                        
        for gluon in gluons:
            for part in incoming_parts+outcoming_parts:
                if gluon==part:
                    conversion_factor-=0.5
        
        for fermion in fermions+antifermions:
            for part in incoming_parts+outcoming_parts:
                if fermion==part:
                    conversion_factor-=2.0/3.0
        
        return conversion_factor
    
    def format_gosam_proc_card(self, procID, proc, model, born_orders,\
                               perturbation_orders, squared_orders, proc_card):
        """Create a gosam '.in' proc card"""

        order_specified=False

        particle_dictionary = {'d':(1,'D'),'d~':(-1,'Dbar'),
                               'u':(2,'U'),'u~':(-2,'Ubar'),
                               's':(3,'S'),'s~':(-3,'Sbar'),
                               'c':(4,'C'),'c~':(-4,'Cbar'),
                               'b':(5,'B'),'b~':(-5,'Bbar'),
                               't':(6,'T'),'t~':(-6,'Tbar'),
                               'g':(21,'g'),
                               'a':(22,'A'),
                               'z':(23,'Z'),
                               'h':(25,'h'),
                               'w+':(24,'Wp'),'w-':(-24,'Wm'),
                               'e+':(-11,'ep'),'e-':(11,'em'),
                               'mu+':(-13,'mup'),'mu-':(13,'mum'),
                               'ta+':(-15,'taup'),'ta-':(15,'taum'),
                               've':(12,'ne'),'ve~':(-12,'nebar'),
                               'vm':(14,'nmu'),'vm~':(-14,'nmubar'),
                               'vt':(16,'ntau'),'vt~':(-16,'ntaubar')}
        
        proc_parts=proc.split()
        if '>' in proc_parts:
            incoming_parts=proc_parts[:proc_parts.index('>')]
            outcoming_parts=proc_parts[proc_parts.index('>')+1:]            
        else:
            raise self.LoopGoSamRunnerError, \
                        "The process %s is ill-formated."%proc
        try:
            proc_name = 'P'+str(procID)+'_'+''.join([particle_dictionary[p][1] \
              for p in incoming_parts])+'_'+''.join([particle_dictionary[p][1] \
              for p in outcoming_parts])
        except KeyError:
            raise self.LoopGoSamRunnerError, \
                        "Some particles in the process %s are not recognized."%proc
        
        proc_card_out=""
        for line in proc_card:
            if line.find("# process_name=")==0:
                proc_card_out+="process_name="+proc_name+'\n'
            elif line.find("# process_path=")==0:
                proc_card_out+="process_path="+proc_name+'\n'
            elif line.find("# in=")==0:
                proc_card_out+="in="+','.join([particle_dictionary[p][1] for \
                                                p in incoming_parts])+'\n'
            elif line.find("# out=")==0:
                proc_card_out+="out="+','.join([particle_dictionary[p][1] for \
                                                p in outcoming_parts])+'\n'
            elif line.find("# extensions=")==0:
                if self.use_dred:
                    proc_card_out+="extensions=dred,"+line[11:]+'\n'
                else:
                    proc_card_out+=line
            elif line.find("# filter.lo=")==0:
                proc_card_out+="filter.lo=lambda d: d.vertices([H], [ep], [em]) == 0\n"
            elif line.find("# filter.nlo=")==0:
                proc_card_out+="filter.nlo=lambda d: d.vertices([H], [ep], [em]) == 0\n"    
            elif line.find("# model=")==0:
                # Irrespectively of what is the user-desired model, we use the sm
                # here for now.
                proc_card_out+="model=sm\n"
            elif line.find("# abbrev.level=")==0:
                proc_card_out+="abbrev.level=diagram\n"
            elif line.find("# abbrev.limit=")==0:
                proc_card_out+="abbrev.limit=500\n"
            elif line.find("# group=")==0:
                proc_card_out+="group=false\n"
            elif line.find("# order=")==0:
                # Please always put 'QCD' first in the list below
                orders_considered=['QCD','QED']
                gosam_born_orders=dict([(order,-1) for order in orders_considered])
                gosam_loop_orders=dict([(order,-1) for order in orders_considered])
                for key, value in born_orders.items():
                    for order in orders_considered:
                        if key==order and value!=99:
                            gosam_born_orders[order]=value
                            if order in perturbation_orders:
                                gosam_loop_orders[order]=gosam_born_orders[order]+2
                            else:
                                gosam_loop_orders[order]=value
                # Change of conventions as GoSam sets the squared order at the
                # amplitude level
                for key, value in squared_orders.items():
                    if key=="WEIGHTED":
                        raise self.LoopGoSamRunnerError, \
                            "Squared order 'WEIGHTED' not supported by GoSam"
                    for order in orders_considered:
                        if key==order and value!=99:
                            if order in perturbation_orders:
                                gosam_loop_orders[order]=value/2+1
                            else:
                                gosam_loop_orders[order]=value/2                                
                # Now write out the orders obtained
                for order in orders_considered:
                    # GoSam only accepts one order specification, so we choose
                    # here QCD if defined and otherwise the first available
                    if not order_specified:
                        if gosam_born_orders[order]!=-1:
                            proc_card_out+="order="+', '.join([order,\
                                            str(gosam_born_orders[order]),\
                                            str(gosam_loop_orders[order])])+'\n'
                            order_specified=True
                        elif gosam_loop_orders[order]!=-1:
                            proc_card_out+="order="+', '.join([order,\
                                            str(gosam_loop_orders[order]),\
                                            str(gosam_loop_orders[order])])+'\n'
                            order_specified=True
                    
            elif line.find("# zero=")==0:
                proc_card_out+="zero=wB,wT,mU,mD,mC,mS,me,mmu,"
                proc_card_out+="VUS,CVSU,VUB,CVBU,VCD,CVDC,"
                proc_card_out+="VCB,CVBC,VTD,CVDT,VTS,CVST"+'\n'
            elif line.find("# one=")==0:
                proc_card_out+="one=VUD,CVDU,VCS,CVSC,VTB,CVBT"+'\n'               
            elif line.find("# qgraf.options=")==0:
                proc_card_out+="qgraf.options=nosnail ,notadpole ,onshell"+'\n'
            elif line.find("# qgraf.verbatim=")==0:
                if 'QED' in born_orders.keys() and 'QED' in squared_orders.keys() and \
                   born_orders['QED']==0 and squared_orders['QED']==0:
                    # Forbid QED particles all together
                    proc_card_out+="qgraf.verbatim=\\\n"
                    proc_card_out+="true=iprop[ep,em,ne,nebar, 0, 0];\\n\\\n"
                    proc_card_out+="true=iprop[mup,mum,nmu,nmubar, 0, 0];\\n\\\n"
                    proc_card_out+="true=iprop[taup,taum,ntau,ntaubar, 0, 0];\\n\\\n"
                    proc_card_out+="true=iprop[A,Z,H,Wp,Wm, 0, 0];\\n\\\n"
                    proc_card_out+="true=iprop[phim,phip,chi,ghA,ghAbar, 0, 0];\\n\\\n"
                    proc_card_out+="true=iprop[ghZ,ghZbar,ghWp,ghWpbar,ghWm, ghWmbar,0, 0];\\n\\\n"
                elif 'QED' not in perturbation_orders:
                    # Only forbid QED particles to be in the loop
                    # /!\ You might not want this, i.e. for VBF pentagons for example
                    proc_card_out+="qgraf.verbatim=\\\n"
                    proc_card_out+="true=chord[ep,em,ne,nebar, 0, 0];\\n\\\n"
                    proc_card_out+="true=chord[mup,mum,nmu,nmubar, 0, 0];\\n\\\n"
                    proc_card_out+="true=chord[taup,taum,ntau,ntaubar, 0, 0];\\n\\\n"
                    proc_card_out+="true=chord[A,Z,H,Wp,Wm, 0, 0];\\n\\\n"
                    proc_card_out+="true=chord[phim,phip,chi,ghA,ghAbar, 0, 0];\\n\\\n"
                    proc_card_out+="true=chord[ghZ,ghZbar,ghWp,ghWpbar,ghWm, ghWmbar,0, 0];\\n\\\n"
                else:
                    proc_card_out+="# No qgraf specific options\n"
            # For custom path of the GoSam dependencies, one might want to
            # uncomment and edit the lines below.
#            elif line.find("# qgraf.bin=")==0:
#                proc_card_out+="qgraf.bin=/Users/erdissshaw/Works/qgraf/run\n"
#            elif line.find("# golem95.fcflags=")==0:
#                proc_card_out+="golem95.fcflags=-I/Users/erdissshaw/Works/GoSam/gosam_contrib_dir/include/gosam-contrib\n"
#            elif line.find("# golem95.ldflags=")==0:
#                proc_card_out+="golem95.ldflags=-L/Users/erdissshaw/Works/GoSam/gosam_contrib_dir/lib -lgolem95\n"
#            elif line.find("# samurai.ldflags=")==0:
#                proc_card_out+="samurai.ldflags=-L/Users/erdissshaw/Works/GoSam/gosam_contrib_dir/lib -lsamurai\n"
#            elif line.find("# samurai.fcflags=")==0:
#                proc_card_out+="samurai.fcflags=-I/Users/erdissshaw/Works/GoSam/gosam_contrib_dir/include/gosam-contrib\n"
            else:
                proc_card_out+=line

        return order_specified, proc_card_out, proc_name

    def get_me_value(self, proc, proc_id, energy, PSpoint=[], bornQCDorder=0):
        """Compile and run ./NLOComp_sa, then parse the output and return the
        result for process with id = proc_id and PSpoint if specified.
        The bornQCDorder is an adhoc hack to get the born result correct in 
        GoSam due to their odd order conventions."""

        sys.stdout.write('.')
        sys.stdout.flush()
        working_dir = os.path.join(self.GoSam_path, self.temp_dir_name)

         
        shell_name = None
        directories = glob.glob(os.path.join(working_dir,'P%i_*' % proc_id))
        if directories and os.path.isdir(directories[0]):
            shell_name = os.path.basename(directories[0])

        # If directory doesn't exist, skip and return 0
        if not shell_name:
            logging.info("Directory hasn't been created for process %s" % (proc))
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

        logging.info("Working on process %s in dir %s" % (proc, shell_name))
        
        dir_name = os.path.join(working_dir, shell_name)
        logging.info("Making GoSam sources for process %s"%proc)
        devnull = open(os.devnull, 'w')
        retcode = subprocess.call(['make','source'],
                                  cwd=dir_name,
                                  stdout=devnull, stderr=devnull)
        if retcode != 0:
            logging.info("Error while making source in %s" % dir_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])
        self.fix_common_file(os.path.join(dir_name,'common'))
        logging.info("Compiling GoSam sources for process %s"%proc)
        devnull = open(os.devnull, 'w')
        retcode = subprocess.call(['make','compile'],
                                  cwd=dir_name,
                                  stdout=devnull, stderr=devnull)
        if retcode != 0:
            logging.info("Error while compiling source in %s" % dir_name)
            devnull.close()
            return ((0.0, 0.0, 0.0, 0.0, 0), [])
        devnull.close()
        matrix_dir_name = os.path.join(dir_name, 'matrix')
        self.fix_output(matrix_dir_name,bornQCDorder)
        self.write_parameters(matrix_dir_name)
        if PSpoint==[]:
            self.fix_energy_in_check(matrix_dir_name, energy)
        else:
            self.fix_PSPoint_in_check(matrix_dir_name)  

        # If a PS point is specified, write out the corresponding PS.input
        if PSpoint!=[]:
            misc.write_PS_input(os.path.join(matrix_dir_name, 'PS.input'),PSpoint)
        
        # Run make
        devnull = open(os.devnull, 'w')
        retcode = subprocess.call(['make','test.exe'],
                        cwd=matrix_dir_name,
                        stdout=devnull, stderr=devnull)
                        
        if retcode != 0:
            logging.info("Error while executing make in %s" % matrix_dir_name)
            devnull.close()
            return ((0.0, 0.0, 0.0, 0.0, 0), [])
        devnull.close()
        
        # Run ./check
        try:
            output = subprocess.Popen('./test.exe',
                        cwd=matrix_dir_name,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            output.read()
            output.close()
            if os.path.exists(os.path.join(matrix_dir_name,'result.dat')):
                return self.parse_check_output(proc,file(os.path.join(\
                                                matrix_dir_name,'result.dat')))  
            else:
                logging.warning("Error while looking for file %s"%str(os.path\
                                        .join(matrix_dir_name,'result.dat')))
                return ((0.0, 0.0, 0.0, 0.0, 0), [])
        except IOError:
            logging.warning("Error while executing ./test.exe in %s" % matrix_dir_name)
            return ((0.0, 0.0, 0.0, 0.0, 0), [])

    def parse_check_output(self, proc, output):
        """Parse the output string and return a pair where first four values are 
        the finite, born, single and double pole of the ME and the fourth is the
        GeV exponent and the second value is a list of 4 momenta for all particles 
        involved."""

        res_p = []
        value = [0.0,0.0,0.0,0.0]
        gev_pow = 0

        for line in output:
            splitline=line.split()
            if splitline[0]=='PS':
                res_p.append([float(s) for s in splitline[1:]])
            elif splitline[0]=='BORN':
                value[1]=float(splitline[1])
            elif splitline[0]=='FIN':
                if self.use_dred:
                    value[0]=float(float(splitline[1])+float(self.dred_to_cdr(proc)))
                else:
                    value[0]=float(splitline[1])
            elif splitline[0]=='1EPS':
                value[2]=float(splitline[1])
            elif splitline[0]=='2EPS':
                value[3]=float(splitline[1])
            elif splitline[0]=='EXP':
                gev_pow=int(splitline[1])

        return ((value[0],value[1],value[2],value[3],gev_pow), res_p)

    def cleanup(self):
        """Clean up temporary directories"""

        try:
            if os.path.isdir(os.path.join(self.mg4_path, self.temp_dir_name)):
                shutil.rmtree(os.path.join(self.mg4_path, self.temp_dir_name))
                logging.info("Temporary standalone directory %s successfully removed" % \
                     self.temp_dir_name)
        except:
            pass

    def write_parameters(self, dir_name):
        """ Writes out the param.dat used by test.exe to define the several
        quantities taken to the default for this test, hardcoded here."""

        file = open(os.path.join(dir_name,'param.dat'), 'w')
        file.write("mW=80.419\n")
        file.write("mZ=91.188\n") 
        file.write("mT=174.3\n")
        file.write("mB=4.62\n")
        file.write("Nf=4\n")
        file.write("Nfgen=4\n")
        file.write("#gs=1.2177157847767197\n")
        file.write("alpha=0.0075467711139788835\n")       
        file.close()        
        
    def fix_output(self, dir_name,bornQCDorder):
        """Modify test.f90 from GoSam to have it output his result in a file, 
        assuming working directory dir_name."""

        file = open(os.path.join(dir_name,'test.f90'), 'r')
        new_test=""
        for line in file:
            new_test+=line
            if line.find('   implicit none')==0:
                new_test+='!     Added from MadLoop5\n'                
                new_test+='   integer :: k\n'
                new_test+='!     End of MadLoop5 Addition\n'
            elif line.find('      call samplitude(')==0:
                new_test+='!     Added from MadLoop5\n'                
                new_test+="      open(69, file=\"result.dat\", err=611, "
                new_test+="action='WRITE')\n"
                new_test+='      do k=1,4\n'
                new_test+="        write (69,'(a2,1x,5e25.15)') 'PS',"
                new_test+='vecs(k,1),vecs(k,2),vecs(k,3),vecs(k,4)\n'
                new_test+='      enddo\n'
                new_test+="      write (69,'(a3,1x,i2)') 'EXP',99\n"
                new_test+="      write (69,'(a4,1x,1e25.15)') 'BORN',amp(0)"
                # Unfortunately there is the need of a hardcoded hack to get 
                # around the odd GoSam conventions for the orders.
                new_test+="*1.2177157847767197_ki**%i\n"%(2*bornQCDorder)
                new_test+="      write (69,'(a3,1x,1e25.15)') 'FIN',amp(1)/amp(0)\n"
                new_test+="      write (69,'(a4,1x,1e25.15)') '1EPS',amp(2)/amp(0)\n"
                new_test+="      write (69,'(a4,1x,1e25.15)') '2EPS',amp(3)/amp(0)\n"
                new_test+='      goto 321\n'
                new_test+='  611 continue\n'
                new_test+="      stop 'Could not write out the results.'\n"
                new_test+='  321 continue\n'
                new_test+='      close(69)\n'
                new_test+='!     End of MadLoop5 Addition\n'
        file.close()

        file = open(os.path.join(dir_name,'test.f90'), 'w')
        file.write(new_test)
        file.close()

    def fix_PSPoint_in_check(self, dir_name):
        """Set test.f90 to be reading PS.input assuming a working dir dir_name"""

        file = open(os.path.join(dir_name,'test.f90'), 'r')
        new_test=""
        for line in file:
            new_test+=line
            if line.find('   implicit none')==0:
                new_test+='!     Added from MadLoop5\n'                
                new_test+='   integer :: i\n'
                new_test+='!     End of MadLoop5 Addition\n'
            elif line.find('      call ramb(')==0:
                new_test+='!     Added from MadLoop5\n'                
                new_test+="      open(967, file=\"PS.input\", err=976, "
                new_test+="status='OLD', action='READ')\n"
                new_test+='        do i=1,4\n'
                new_test+='          read(967,*,end=978) vecs(i,1),vecs(i,2),'
                new_test+='vecs(i,3),vecs(i,4)\n'
                new_test+='        enddo\n'
                new_test+='      goto 978\n'
                new_test+='  976 continue\n'
                new_test+="      stop 'Could not read the PS.input phase-space point.'\n"
                new_test+='  978 continue\n'
                new_test+='      close(967)\n'
                new_test+='!     End of MadLoop5 Addition\n'
        file.close()

        file = open(os.path.join(dir_name,'test.f90'), 'w')
        file.write(new_test)
        file.close()

    def fix_energy_in_check(self, dir_name, energy):
        """Replace the hard coded collision energy in check_sa.f by the given
        energy, assuming a working dir dir_name"""

        file = open(os.path.join(dir_name,'test.f90'), 'r')
        test = file.read()
        file.close()

        file = open(os.path.join(dir_name,'test.f90'), 'w')
        file.write(re.sub("5.0E\+02","%i.0E+00"% int(energy), test))
        file.close()

    def fix_common_file(self, dir_name):
        """Fix some compile-time parameters in common/config.90. It is assumed
        that dir_name is the 'common' directory."""

        file = open(os.path.join(dir_name,'config.f90'), 'r')
        config = file.read()
        file.close()
        
        # Make GoSam include the symmetry factors.
        file = open(os.path.join(dir_name,'config.f90'), 'w')
        file.write(re.sub("include_symmetry_factor = .false.",\
                          "include_symmetry_factor = .true.", config))
        file.close()

class LoopMEComparator(me_comparator.MEComparator):
    """Base object to run comparison tests for loop processes. Take standard 
    MERunner objects and a list of loop proc as an input and return detailed 
    comparison tables in various formats."""

    me_runners = []
    results    = []
    proc_list  = []
    orders     = []
    
    def run_comparison(self, proc_list, model='loop_SM_QCD', energy=1000):
        """Run the codes and store results.
        Notice that the proc list is a list of tuples formated like this:
        (process,born_orders,perturbation_orders,squared_orders)"""

        if isinstance(model, basestring):
            model= [model] * len(self.me_runners)

        self.results = []
        self.proc_list = proc_list

        logging.info(\
            "Running on %i processes in model %s @ %i GeV" % \
            (len(proc_list),\
             '/'.join([onemodel for onemodel in model]),\
             energy))

        pass_proc = False
        for i,runner in enumerate(self.me_runners):
            cpu_time1 = time.time()
            logging.info("Now running %s" % runner.name)
            if pass_proc:
                runner.pass_proc = pass_proc 
            # If this is the first runner, then let it generate the PS points,
            # otherwise reuse the already generated ones
            if self.results!=[]:
                PSpoints=[PS[1] for PS in self.results[0]]
            else:
                PSpoints=[]
            self.results.append(runner.run(proc_list, model[i], energy,
                                PSpoints))

            if hasattr(runner, 'new_proc_list'):
                pass_proc = runner.new_proc_list
            cpu_time2 = time.time()
            logging.info(" Done in %0.3f s" % (cpu_time2 - cpu_time1))
            logging.info(" (%i/%i with zero ME)" % \
                    (len([res for res in self.results[-1] if res[0][0] == 0.0]),
                     len(proc_list)))

    def output_result(self, filename=None, tolerance=1e-04, skip_zero=True):
        """Output result as a nicely formated table. If filename is provided,
        write it to the file, else to the screen. Tolerance can be adjusted."""

        if any('^2' in sqso for proc in self.proc_list for sqso in proc[3].keys()):
            proc_col_size = 42
        else:
            proc_col_size = 24

        for proc in self.proc_list:
            if len(proc) + 2 > proc_col_size:
                proc_col_size = len(proc) + 2
        
        col_size = 24

        pass_proc = 0
        fail_proc = 0

        failed_proc_list = []
        
        res_str = ""
        
        testing_list=['Finite', 'Born', 'Single pole', 'Double pole']
        for index in range(0,4):
            res_str +=("\n\n" if index!=0 else "")
#            res_str +=''.join(['=']*(len(testing_list[index])+8))
            res_str +="=== "+testing_list[index]+" ==="
#            res_str +="\n"+''.join(['=']*(len(testing_list[index])+8))
            res_str += "\n" + self._fixed_string_length("Process", proc_col_size) + \
                    ''.join([self._fixed_string_length(runner.name, col_size) for \
                               runner in self.me_runners]) + \
                      self._fixed_string_length("Relative diff.", col_size) + \
                      "Result"
            
            for i, (proc, born_orders, perturbation_orders, squared_orders) \
              in enumerate(self.proc_list):
                list_res = [res[i][0][index] for res in self.results]
                if index==0:maxfin=max(max(map(abs,list_res)),1e-99)
                if max(list_res) == 0.0 and min(list_res) == 0.0:
                    diff = 0.0
                    if skip_zero:
                        continue
                else:
                    diff = (max(list_res) - min(list_res)) / \
                           abs((max(list_res) + min(list_res)))

                proc_string = ""
                if any('^2' in key for key in squared_orders.keys()):
                    # In this case, it is necessary to also detail the squared
                    # order constraints
                    proc_string = self._fixed_string_length(proc+' %s'%' '.join(
                        '%s%i'%(key, value) for key, value in 
                                          squared_orders.items()),proc_col_size)
                else:
                    proc_string = self._fixed_string_length(proc, proc_col_size)
                res_str += '\n' + proc_string+ \
                           ''.join([self._fixed_string_length("%1.10e" % res,
                                                 col_size) for res in list_res])
    
                res_str += self._fixed_string_length("%1.10e" % diff, col_size)
                if diff < tolerance or max(map(abs,list_res))/maxfin<tolerance and index>1:
                    if index==3 and proc not in failed_proc_list:
                        pass_proc += 1
                    res_str += "Pass"
                else:
                    if proc not in failed_proc_list:
                        fail_proc += 1
                        failed_proc_list.append(proc)
                    res_str += "Fail"

        res_str +="\n\n=== Summary ==="

        res_str += "\n %i/%i passed, %i/%i failed" % \
                    (pass_proc, pass_proc + fail_proc,
                     fail_proc, pass_proc + fail_proc)

        if fail_proc != 0:
            res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)

        logging.info("\n"+res_str)

        if filename:
            file = open(filename, 'w')
            file.write(res_str)
            if failed_proc_list:
                file.write('\n'+str(failed_proc_list))
            file.close()

    def assert_processes(self, test_object, tolerance = 3e-06):
        """Run assert to check that all processes passed comparison""" 

        col_size = 17
        fail_proc = 0
        fail_str = "Failed for processes:"
        failed_proc_list = []

        for index in range(0,4):        
            for i, (proc, born_orders, perturbation_orders, squared_orders) \
                                                 in enumerate(self.proc_list):
                list_res = [res[i][0][index] for res in self.results]
                if index==0:maxfin=max(max(map(abs,list_res)),1e-99)
                if max(list_res) == 0.0 and min(list_res) == 0.0:
                    diff = 0.0
                else:
                    diff = (max(list_res) - min(list_res)) / \
                           abs((max(list_res) + min(list_res)))

                if diff >= tolerance and proc not in failed_proc_list and\
                         (max(map(abs,list_res))/maxfin>=tolerance or index<=1):
                    failed_proc_list.append(proc)
                    fail_str += self._fixed_string_length(proc, col_size) + \
                                ''.join([self._fixed_string_length("%1.10e" % res,
                                         col_size) for res in list_res])
                
                    fail_str += self._fixed_string_length("%1.10e" % diff, col_size)

        test_object.assertEqual(fail_str, "Failed for processes:")



class LoopHardCodedRefRunnerError(Exception):
        """class for error in LoopHardCodedRefRunner"""
        pass
    
class LoopHardCodedRefRunner(me_comparator.MERunner):
    """Runner object for hard-coded reference loop processes."""
    name = 'Hard-Coded Ref.'
    type = 'HCR'
    def setup(self,proc_list,res_list,model,decay=False):
        self.proc_list=proc_list
        self.res_list=res_list
        self.model=model
        PS = res_list[0][-1]
        if not decay:
            PSinit = list(itertools.imap(operator.add,PS[0],PS[1]))
        else:
            PSinit = PS[0]
        energy = math.sqrt(PSinit[0]**2-PSinit[1]**2-PSinit[2]**2-PSinit[3]**2)
        self.energy = energy

    def run(self,proc_list, model, energy=1000, PSpoints=[]):
        if PSpoints!=[] and PSpoints!=self.res_list[0][-1]:
            raise self.LoopHardCodedRefRunnerError,\
                 "Phase space point is not provided !"
        return self.res_list

class LoopMEComparatorGauge(LoopMEComparator):
    """Base object to run comparison tests for loop processes. Take standard 
    MERunner objects and a list of loop proc as an input and return detailed 
    comparison tables in various formats."""

    me_runners = []
    results    = []
    proc_list  = []
    orders     = []
    
    def output_result(self, filename=None, tolerance=3e-06, skip_zero=True):
        """Output result as a nicely formated table. If filename is provided,
        write it to the file, else to the screen. Tolerance can be adjusted."""

        proc_col_size = 20

        for proc in self.proc_list:
            if len(proc) + 2 > proc_col_size:
                proc_col_size = len(proc) + 2
        
        col_size = 20

        pass_proc = 0
        fail_proc = 0

        failed_proc_list = []
        
        res_str = ""
        
        testing_list=['Finite', 'Born', 'Single pole', 'Double pole']
        for index in range(0,4):
            res_str +=("\n\n" if index!=0 else "")
#            res_str +=''.join(['=']*(len(testing_list[index])+8))
            res_str +="=== "+testing_list[index]+" ==="
#            res_str +="\n"+''.join(['=']*(len(testing_list[index])+8))
            res_str += "\n" + self._fixed_string_length("Process", proc_col_size) + \
                    ''.join([self._fixed_string_length(runner.name, col_size) for \
                               runner in self.me_runners]) + \
                      self._fixed_string_length("Diff both unit", col_size) + \
                      self._fixed_string_length("Diff both cms", col_size) + \
                      self._fixed_string_length("Diff both fixw", col_size) + \
                      self._fixed_string_length("Diff both feyn", col_size) + \
                      "Result"
    
            for i, (proc, born_orders, perturbation_orders, squared_orders) \
              in enumerate(self.proc_list):
                list_res = [res[i][0][index] for res in self.results]
                
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
                    if proc not in failed_proc_list:
                        fail_proc += 1
                        failed_proc_list.append(proc)
                    res_str += "Fail"
                    
        res_str +="\n\n=== Summary ==="

        res_str += "\n %i/%i passed, %i/%i failed" % \
                    (pass_proc, pass_proc + fail_proc,
                     fail_proc, pass_proc + fail_proc)

        if fail_proc != 0:
            res_str += "\nFailed processes: %s" % ', '.join(failed_proc_list)

        logging.info("\n"+res_str)

        if filename:
            file = open(filename, 'w')
            file.write(res_str)
            if failed_proc_list:
                file.write('\n'+str(failed_proc_list))
            file.close()

    def assert_processes(self, test_object, tolerance = 1e-06):
        """Run assert to check that all processes passed comparison""" 

        col_size = 17
        fail_proc = 0
        fail_str = "Failed for processes:"
        failed_proc_list = []

        for index in range(0,4):        
            for i, (proc, born_orders, perturbation_orders, squared_orders) \
                                                 in enumerate(self.proc_list):
                list_res = [res[i][0][index] for res in self.results]
                if max(list_res) == 0.0 and min(list_res) == 0.0:
                    diff = 0.0
                else:
                    diff_feyn = abs(list_res[1] - list_res[2]) / \
                       (list_res[1] + list_res[2] + 1e-99)
                    diff_unit = abs(list_res[0] - list_res[3]) / \
                       (list_res[0] + list_res[3] + 1e-99)
                    diff_cms = abs(list_res[0] - list_res[1]) / \
                       (list_res[0] + list_res[1] + 1e-99)
                    diff_fixw = abs(list_res[2] - list_res[3]) / \
                       (list_res[2] + list_res[3] + 1e-99)
                
                if diff_feyn > 1e-2 or diff_cms > 1e-6 or diff_fixw > 1e-4 or \
                  diff_unit > 1e-2 and proc not in failed_proc_list:
                    failed_proc_list.append(proc)
                    fail_str += self._fixed_string_length(proc, col_size) + \
                                ''.join([self._fixed_string_length("%1.10e" % res,
                                         col_size) for res in list_res])

        test_object.assertEqual(fail_str, "Failed for processes:")
