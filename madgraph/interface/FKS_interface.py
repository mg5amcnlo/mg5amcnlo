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
"""A user friendly command line interface to access all MadGraph features.
   Uses the cmd package for command interpretation and tab completion.
"""

import os
import logging
import time

import madgraph
from madgraph import MG4DIR, MG5DIR, MadGraph5Error
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.fks.fks_real as fks_real
import madgraph.fks.fks_real_helas_objects as fks_real_helas
import madgraph.fks.fks_born as fks_born
import madgraph.fks.fks_born_helas_objects as fks_born_helas
import madgraph.iolibs.export_fks_real as export_fks_real
import madgraph.iolibs.export_fks_born as export_fks_born

#usefull shortcut
pjoin = os.path.join


logger = logging.getLogger('cmdprint') # -> stdout
logger_stderr = logging.getLogger('fatalerror') # ->stderr
logger_tuto = logging.getLogger('tutorial') # -> stdout include instruction in  
                                            #order to learn MG5

class CheckFKS(mg_interface.CheckValidForCmd):
    pass

class CheckFKSWeb(mg_interface.CheckValidForCmdWeb, CheckFKS):
    pass

class CompleteFKS(mg_interface.CompleteForCmd):
    pass

class HelpFKS(mg_interface.HelpToCmd):
    pass

class FKSInterface(CheckFKS, CompleteFKS, HelpFKS, mg_interface.MadGraphCmd):

    def check_output(self, args):
        """ check the validity of the line"""
          
        if args and args[0] in self._nlo_export_formats:
            self._export_format = args.pop(0)
        else:
            self._export_format = 'NLO'

        if not self._fks_multi_proc:
            text = 'No processes generated. Please generate a process first.'
            raise self.InvalidCmd(text)

        if not self._curr_model:
            text = 'No model found. Please import a model first and then retry.'
            raise self.InvalidCmd(text)

        if args and args[0][0] != '-':
            # This is a path
            path = args.pop(0)
            # Check for special directory treatment
            if path == 'auto' and self._export_format in \
                     self._nlo_export_formats:
                self.get_default_path()
            elif path != 'auto':
                self._export_dir = path
        else:
            # No valid path
            self.get_default_path()

        self._export_dir = os.path.realpath(self._export_dir)

    def do_add(self, line, *args,**opt):
        
        args = self.split_arg(line)
        # Check the validity of the arguments
        self.check_add(args)

        if args[0] != 'process': 
            raise self.InvalidCmd("The add command can only be used with a process")
        else:
            line = ' '.join(args[1:])

        if self._curr_model['perturbation_couplings']=={}:
            if self._curr_model['name']=='sm':
                mg_interface.logger.warning(\
                  "The default sm model does not allow to generate"+
                  " loop processes. MG5 now loads 'loop_sm' instead.")
                mg_interface.MadGraphCmd.do_import(self,"model loop_sm")
            else:
                raise MadGraph5Error(
                  "The model %s cannot generate loop processes"\
                  %self._curr_model['name'])
        
        orders=self.extract_process_type(line)[2]
        if orders!=['QCD']:
                raise MadGraph5Error, 'FKS for reals only available in QCD for now, you asked %s' \
                        % ', '.join(orders)
                        
        #now generate the amplitudes as usual
        self._options['group_subprocesses'] = 'NLO'
        collect_mirror_procs = False
        ignore_six_quark_processes = self._options['ignore_six_quark_processes']
#        super(FKSInterface, self).do_generate(line)
        if ',' in line:
            myprocdef, line = mg_interface.MadGraphCmd.extract_decay_chain_process(self,line)
            if myprocdef.are_decays_perturbed():
                raise MadGraph5Error("Decay processes cannot be perturbed")
        else:
            myprocdef = mg_interface.MadGraphCmd.extract_process(self,line)

        myprocdef['perturbation_couplings'] = ['QCD']

        if self._options['fks_mode'] == 'born':
            self._fks_multi_proc = fks_born.FKSMultiProcessFromBorn(myprocdef,
                                       collect_mirror_procs,
                                       ignore_six_quark_processes)
        elif self._options['fks_mode'] == 'real':
            self._fks_multi_proc = fks_real.FKSMultiProcessFromReals(myprocdef,
                                       collect_mirror_procs,
                                       ignore_six_quark_processes)
            # this is for testing, to be removed
        else: 
            raise MadGraph5Error, 'Unknown FKS mode: %s' % self._options['fks_mode']

    def do_output(self, line):
        """Initialize a new Template or reinitialize one"""

        args = self.split_arg(line)
        # Check Argument validity
        self.check_output(args)

        # Remove previous outputs from history
        self.clean_history(to_remove=['display','open','history','launch','output'],
                           remove_bef_lb1='generate',
                           keep_last=True)
        
        noclean = '-noclean' in args
        force = '-f' in args 
        nojpeg = '-nojpeg' in args
        main_file_name = ""
        try:
            main_file_name = args[args.index('-name') + 1]
        except:
            pass

        # initialize the writer
        if self._export_format in self._nlo_export_formats:
            if self._options['fks_mode'] == 'real':
                logger.info("Exporting in MadFKS format, starting from real emission process")
                self._curr_exporter = export_fks_real.ProcessExporterFortranFKS_real(\
                                          self._mgme_dir, self._export_dir,
                                          not noclean, 
                                          os.path.join(self._mgme_dir, 'loop_material'),
                                          self._cuttools_dir)
    
            if self._options['fks_mode'] == 'born':
                logger.info("Exporting in MadFKS format, starting from born process")
                self._curr_exporter = export_fks_born.ProcessExporterFortranFKS_born(\
                                          self._mgme_dir, self._export_dir,
                                          not noclean, 
                                          os.path.join(self._mgme_dir, 'loop_material'),
                                          self._cuttools_dir)
            
        # check if a dir with the same name already exists
        if not force and not noclean and os.path.isdir(self._export_dir)\
               and self._export_format in self._nlo_export_formats:
            # Don't ask if user already specified force or noclean
            logger.info('INFO: directory %s already exists.' % self._export_dir)
            logger.info('If you continue this directory will be cleaned')
            answer = self.ask('Do you want to continue?', 'y', ['y','n'], 
                                                           timeout=self.timeout)
            if answer != 'y':
                raise self.InvalidCmd('Stopped by user request')
    
        # Make a Template Copy
        if self._export_format in self._nlo_export_formats:
            self._curr_exporter.copy_fkstemplate()

        # Reset _done_export, since we have new directory
        self._done_export = False

        # Perform export and finalize right away
        self.export(nojpeg, main_file_name)

        # Automatically run finalize
        self.finalize(nojpeg)
            
        # Remember that we have done export
        self._done_export = (self._export_dir, self._export_format)

        # Reset _export_dir, so we don't overwrite by mistake later
        self._export_dir = None


    # Export a matrix element
    
    def export(self, nojpeg = False, main_file_name = ""):
        """Export a generated amplitude to file"""

        def generate_matrix_elements(self):
            """Helper function to generate the matrix elements before
            exporting"""

            # Sort amplitudes according to number of diagrams,
            # to get most efficient multichannel output
            self._curr_amps.sort(lambda a1, a2: a2.get_number_of_diagrams() - \
                                 a1.get_number_of_diagrams())

            # Check if we need to group the SubProcesses or not
            group = True
            if self._options['group_subprocesses'] in [False, 'NLO']:
                group = False
            elif self._options['group_subprocesses'] == 'Auto' and \
                                         self._curr_amps[0].get_ninitial() == 1:
                   group = False 

            cpu_time1 = time.time()
            ndiags = 0
            if not self._curr_matrix_elements.get_matrix_elements():
                if group:
                    raise MadGraph5Error, "Cannot group subprocesses when exporting to NLO"
                else:
                    if self._options['fks_mode'] == 'real':
                        self._curr_matrix_elements = \
                                 fks_real_helas.FKSHelasMultiProcessFromReals(\
                                    self._fks_multi_proc)
                    elif self._options['fks_mode'] == 'born':
                        self._curr_matrix_elements = \
                                 fks_born_helas.FKSHelasMultiProcessFromBorn(\
                                    self._fks_multi_proc)
                    else:
                        self._curr_matrix_elements = \
                                 helas_objects.HelasMultiProcess(\
                                               self._curr_amps)

                    ndiags = sum([len(me.get('diagrams')) for \
                                  me in self._curr_matrix_elements.\
                                  get_matrix_elements()])
                    # assign a unique id number to all process
                    uid = 0 
                    for me in self._curr_matrix_elements.get_matrix_elements():
                        uid += 1 # update the identification number
                        me.get('processes')[0].set('uid', uid)

            cpu_time2 = time.time()
            return ndiags, cpu_time2 - cpu_time1

        # Start of the actual routine

        ndiags, cpu_time = generate_matrix_elements(self)

        calls = 0

        path = self._export_dir

        if self._export_format in ['NLO']:
            path = os.path.join(path, 'SubProcesses')

        if self._export_format == 'NLO' and self._options['fks_mode'] == 'real':
            #_curr_matrix_element is a FKSHelasMultiProcessFromRealObject 
            self._fks_directories = []
            for ime, me in \
                enumerate(self._curr_matrix_elements.get_matrix_elements()):
                #me is a FKSHelasProcessFromReals
                calls = calls + \
                        self._curr_exporter.generate_born_directories_fks(\
                            me, self._curr_fortran_model, ime, path)
                self._fks_directories.extend(self._curr_exporter.fksdirs)
            card_path = os.path.join(path, os.path.pardir, 'SubProcesses', \
                                     'procdef_mg5.dat')
            if self._generate_info:
                self._curr_exporter.write_procdef_mg5(card_path, #
                                self._curr_model['name'],
                                self._generate_info)
                try:
                    cmd.Cmd.onecmd(self, 'history .')
                except:
                    pass
            
        cpu_time1 = time.time()

        if self._export_format == 'NLO' and self._options['fks_mode'] == 'born':
            #_curr_matrix_element is a FKSHelasMultiProcessFromBornObject 
            self._fks_directories = []
            for ime, me in \
                enumerate(self._curr_matrix_elements.get('matrix_elements')):
                #me is a FKSHelasProcessFromReals
                calls = calls + \
                        self._curr_exporter.generate_real_directories_fks(\
                            me, self._curr_fortran_model, ime, path)
                self._fks_directories.extend(self._curr_exporter.fksdirs)
            card_path = os.path.join(path, os.path.pardir, 'SubProcesses', \
                                     'procdef_mg5.dat')
            if self._generate_info:
                self._curr_exporter.write_procdef_mg5(card_path, #
                                self._curr_model['name'],
                                self._generate_info)
                try:
                    cmd.Cmd.onecmd(self, 'history .')
                except:
                    pass
            
        cpu_time1 = time.time()

   
class FKSInterfaceWeb(mg_interface.CheckValidForCmdWeb, FKSInterface):
    pass

