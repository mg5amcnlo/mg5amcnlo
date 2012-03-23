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
import time
import logging

import madgraph
from madgraph import MG4DIR, MG5DIR, MadGraph5Error
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.export_v4 as export_v4
import madgraph.loop.loop_exporters as loop_exporters
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.file_writers as writers


# Special logger for the Cmd Interface
logger = logging.getLogger('cmdprint')

#useful shortcut
pjoin = os.path.join

class CheckLoop(mg_interface.CheckValidForCmd):

    def check_display(self, args):
        """ Check the arguments of the display diagrams command in the context
        of the Loop interface."""
        
        mg_interface.MadGraphCmd.check_display(self,args)
        
        if args[0]=='diagrams' and len(args)>=3 and args[1] not in ['born','loop']:
            raise self.InvalidCmd("Can only display born or loop diagrams, not %s."%args[1])

class CheckLoopWeb(mg_interface.CheckValidForCmdWeb, CheckLoop):
    pass

class CompleteLoop(mg_interface.CompleteForCmd):
    
    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command in the context of the Loop interface"

        args = self.split_arg(line[0:begidx])

        if len(args) == 2 and args[1] == 'diagrams':
            return self.list_completion(text, ['born', 'loop'])
        else:
            return mg_interface.MadGraphCmd.complete_display(self, text, line,
                                                                 begidx, endidx)

class HelpLoop(mg_interface.HelpToCmd):

    def help_display(self):   
        mg_interface.MadGraphCmd.help_display(self)
        logger.info("   In ML5, after display diagrams, the user can add the option")
        logger.info("   \"born\" or \"loop\" to display only the corresponding diagrams.")

class LoopInterface(CheckLoop, CompleteLoop, HelpLoop, mg_interface.MadGraphCmd):
    
    def do_generate(self, line, *args,**opt):

        # Check args validity
        self.model_validity()    
        # Extract process from process definition
        if ',' in line:
            myprocdef, line = self.extract_decay_chain_process(line)
        else:
            myprocdef = self.extract_process(line)
        self.proc_validity(myprocdef)
                
        mg_interface.MadGraphCmd.do_generate(self, line, *args,**opt)
    
    def do_display(self,line, *argss, **opt):
        """ Display born or loop diagrams, otherwise refer to the default display
        command """
        
        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_display(args)
        
        if args[0]=='diagrams':
            if len(args)>=2 and args[1] in ['loop','born']:
                self.draw(' '.join(args[2:]),args[1])
            else:
                self.draw(' '.join(args[1:]),'all')
        else:
            mg_interface.MadGraphCmd.do_display(self,line,*argss,**opt)

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

        if self._export_format not in ['standalone','matrix']:
            raise MadGraph5Error('ML5 only support standalone and matrix  as export format.')

        if not os.path.isdir(self._export_dir) and \
           self._export_format in ['matrix']:
            raise MadGraph5Error('Specified export directory %s does not exist.'%str(self._export_dir))

        if not force and not noclean and os.path.isdir(self._export_dir)\
               and self._export_format in ['standalone']:
            # Don't ask if user already specified force or noclean
            logger.info('INFO: directory %s already exists.' % self._export_dir)
            logger.info('If you continue this directory will be cleaned')
            answer = self.ask('Do you want to continue?', 'y', ['y','n'], 
                                                           timeout=self.timeout)
            if answer != 'y':
                raise self.InvalidCmd('Stopped by user request')

        if os.path.isdir(os.path.join(self._mgme_dir, 'loop_material')):
            self._curr_exporter = loop_exporters.LoopProcessExporterFortranSA(\
                                        self._mgme_dir, self._export_dir, not noclean,\
                                        os.path.join(self._mgme_dir, 'loop_material'),\
                                        self._cuttools_dir)
        else:
            raise MadGraph5Error('MG5 cannot find the \'loop_material\' directory'+\
                                 ' in %s'%str(self._mgme_dir))                                                           

        if self._export_format in ['standalone']:
            self._curr_exporter.copy_v4template(modelname=self._curr_model.get('name'))

        # Reset _done_export, since we have new directory
        self._done_export = False

        # Perform export and finalize right away
        self.ML5export(nojpeg, main_file_name)

        # Automatically run finalize
        self.ML5finalize(nojpeg)
            
        # Remember that we have done export
        self._done_export = (self._export_dir, self._export_format)

        # Reset _export_dir, so we don't overwrite by mistake later
        self._export_dir = None

    # Export a matrix element
    
    def ML5export(self, nojpeg = False, main_file_name = ""):
        """Export a generated amplitude to file"""

        def generate_matrix_elements(self):
            """Helper function to generate the matrix elements before
            exporting"""

            # Sort amplitudes according to number of diagrams,
            # to get most efficient multichannel output
            self._curr_amps.sort(lambda a1, a2: a2.get_number_of_diagrams() - \
                                 a1.get_number_of_diagrams())

            cpu_time1 = time.time()
            ndiags = 0
            if not self._curr_matrix_elements.get_matrix_elements():
                self._curr_matrix_elements = \
                    helas_objects.HelasMultiProcess(self._curr_amps)
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
        if self._export_format in ['standalone']:
            path = pjoin(path, 'SubProcesses')
            
        cpu_time1 = time.time()

        # Pick out the matrix elements in a list
        matrix_elements = \
                        self._curr_matrix_elements.get_matrix_elements()

        # Fortran MadGraph Standalone
        if self._export_format == 'standalone':
            for me in matrix_elements:
                calls = calls + \
                        self._curr_exporter.generate_subprocess_directory_v4(\
                            me, self._curr_fortran_model)

        # Just the matrix.f files
        if self._export_format == 'matrix':
            for me in matrix_elements:
                filename = pjoin(path, 'matrix_' + \
                           me.get('processes')[0].shell_string() + ".f")
                if os.path.isfile(filename):
                    logger.warning("Overwriting existing file %s" % filename)
                else:
                    logger.info("Creating new file %s" % filename)
                calls = calls + self._curr_exporter.write_matrix_element_v4(\
                    writers.FortranWriter(filename),\
                    me, self._curr_fortran_model)
                
        cpu_time2 = time.time() - cpu_time1

        logger.info(("Generated helas calls for %d subprocesses " + \
              "(%d diagrams) in %0.3f s") % \
              (len(matrix_elements),
               ndiags, cpu_time))

        if calls:
            if "cpu_time2" in locals():
                logger.info("Wrote files for %d OPP calls in %0.3f s" % \
                            (calls, cpu_time2))
            else:
                logger.info("Wrote files for %d OPP calls" % \
                            (calls))

        # Replace the amplitudes with the actual amplitudes from the
        # matrix elements, which allows proper diagram drawing also of
        # decay chain processes
        self._curr_amps = diagram_generation.AmplitudeList(\
               [me.get('base_amplitude') for me in \
                matrix_elements])

    def ML5finalize(self, nojpeg, online = False):
        """Copy necessary sources and output the ps representation of 
        the diagrams, if needed"""
        
        if self._export_format in ['standalone']:
            logger.info('Export UFO model to MG4 format')
            # wanted_lorentz are the lorentz structures which are
            # actually used in the wavefunctions and amplitudes in
            # these processes
            wanted_lorentz = self._curr_matrix_elements.get_used_lorentz()
            wanted_couplings = self._curr_matrix_elements.get_used_couplings()
            self._curr_exporter.convert_model_to_mg4(self._curr_model,
                                           wanted_lorentz,
                                           wanted_couplings)

        if self._export_format in ['standalone']:
            self._curr_exporter.finalize_v4_directory( \
                                           self._curr_matrix_elements,
                                           [self.history_header] + \
                                           self.history,
                                           not nojpeg,
                                           online,
                                           self.options['fortran_compiler'])

        if self._export_format in ['standalone']:
            logger.info('Output to directory ' + self._export_dir + ' done.')

    def do_launch(self, line, *args,**opt):
        """ Check that the type of launch is fine before proceeding with the
        mother function. """
                
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = mg_interface._launch_parser.parse_args(argss)
        self.check_launch(argss, options)
        
        if not argss[0].startswith('standalone'):
            raise self.InvalidCmd('ML5 can only launch standalone runs.')
        return mg_interface.MadGraphCmd.do_launch(self, line, *args,**opt)

    def do_check(self, line, *args,**opt):
        """Check a given process or set of processes"""

        argss = self.split_arg(line, *args,**opt)
        # Check args validity
        self.model_validity()
        param_card = self.check_check(argss)
        proc = " ".join(argss[1:])
        myprocdef = self.extract_process(proc)
        self.proc_validity(myprocdef)
        
        return mg_interface.MadGraphCmd.do_check(self, line, *args,**opt)
    
    def do_add(self, line, *args,**opt):
        """Generate an amplitude for a given process and add to
        existing amplitudes
        """
        args = self.split_arg(line)
        
        # Check the validity of the arguments
        self.check_add(args)
        self.model_validity()

        if args[0] == 'process':            
            # Rejoin line
            line = ' '.join(args[1:])
            
            # store the first process (for the perl script)
            if not self._generate_info:
                self._generate_info = line
                
            # Reset Helas matrix elements
            self._curr_matrix_elements = helas_objects.HelasMultiProcess()

            # Extract process from process definition
            if ',' in line:
                myprocdef, line = self.extract_decay_chain_process(line)
            else:
                myprocdef = self.extract_process(line)
                
            self.proc_validity(myprocdef)

            cpu_time1 = time.time()

            # Decide here wether one needs a LoopMultiProcess or a MultiProcess
            multiprocessclass=None
            if myprocdef['perturbation_couplings']!=[]:
                multiprocessclass=loop_diagram_generation.LoopMultiProcess
            else:
                multiprocessclass=diagram_generation.MultiProcess

            myproc = multiprocessclass(myprocdef, collect_mirror_procs = False,
                                       ignore_six_quark_processes = False)

            for amp in myproc.get('amplitudes'):
                if amp not in self._curr_amps:
                    self._curr_amps.append(amp)
                else:
                    warning = "Warning: Already in processes:\n%s" % \
                                                amp.nice_string_processes()
                    logger.warning(warning)


            # Reset _done_export, since we have new process
            self._done_export = False

            cpu_time2 = time.time()

            ndiags = sum([len(amp.get('loop_diagrams')) for \
                              amp in myproc.get('amplitudes')])
            logger.info("Process generated in %0.3f s" % \
                  (cpu_time2 - cpu_time1))

    def proc_validity(self, proc):
        """ Check that the process or processDefinition describes a process that 
        ML5 can handle"""

        # Check that we have something    
        if not proc:
            raise self.InvalidCmd("Empty or wrong format process, please try again.")
        
        # Check that we have the same number of initial states as
        # existing processes
        if self._curr_amps and self._curr_amps[0].get_ninitial() != \
            proc.get_ninitial():
            raise self.InvalidCmd("Can not mix processes with different number of initial states.")               
            

        if isinstance(proc, base_objects.ProcessDefinition):
            if proc.has_multiparticle_label():
                raise self.InvalidCmd(
                  "When running ML5 standalone, multiparticle labels cannot be"+\
                  " employed. Please use the FKS5 interface instead.")
        
        if proc['decay_chains']:
            raise self.InvalidCmd(
                  "ML5 cannot yet decay a core process including loop corrections.")
        
        if proc.are_decays_perturbed():
            raise self.InvalidCmd(
                  "The processes defining the decay of the core process cannot"+\
                  " include loop corrections.")
        
        if not proc['perturbation_couplings']:
            raise self.InvalidCmd(
                "Please perform tree-level generations within default MG5 interface.")
        
        if proc['perturbation_couplings'] and not \
           isinstance(self._curr_model,loop_base_objects.LoopModel):
            raise self.InvalidCmd(
                "The current model does not allow for loop computations.")
        else:
            for pert_order in proc['perturbation_couplings']:
                if pert_order not in self._curr_model['perturbation_couplings']:
                    raise self.InvalidCmd(
                        "Perturbation order %s is not among" % pert_order + \
                        " the perturbation orders allowed for by the loop model.")

    def model_validity(self):
        """ Upgrade the model sm to loop_sm if needed """
    
        if self._curr_model['perturbation_couplings']==[]:
            if self._curr_model['name']=='sm':
                logger.warning(\
                  "The default sm model does not allow to generate"+
                  " loop processes. MG5 now loads 'loop_sm' instead.")
                mg_interface.MadGraphCmd.do_import(self,"model loop_sm")
            else:
                raise self.InvalidCmd(
                  "The model %s cannot handle loop processes"\
                  %self._curr_model['name'])
   
class LoopInterfaceWeb(mg_interface.CheckValidForCmdWeb, LoopInterface):
    pass

