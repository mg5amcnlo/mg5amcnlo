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
import sys
import time
import optparse
import subprocess

import madgraph
from madgraph import MG4DIR, MG5DIR, MadGraph5Error
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.madevent_interface as me_interface
import madgraph.interface.Loop_interface as Loop_interface
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.iolibs.export_fks as export_fks
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.various.cluster as cluster

#usefull shortcut
pjoin = os.path.join


logger = logging.getLogger('cmdprint') # -> stdout
logger_stderr = logging.getLogger('fatalerror') # ->stderr
logger_tuto = logging.getLogger('tutorial') # -> stdout include instruction in  
                                            #order to learn MG5

class CheckFKS(mg_interface.CheckValidForCmd):


    def check_display(self, args):
        """ Check the arguments of the display diagrams command in the context
        of the Loop interface."""
        
        mg_interface.MadGraphCmd.check_display(self,args)
        
        if args[0] in ['diagrams', 'processes'] and len(args)>=3 \
                and args[1] not in ['born','loop','virt','real']:
            raise self.InvalidCmd("Can only display born, loop (virt) or real diagrams, not %s."%args[1])


    def check_output(self, args):
        """ check the validity of the line"""
          
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
            if path == 'auto':
                self.get_default_path()
            elif path != 'auto':
                self._export_dir = path
        else:
            # No valid path
            self.get_default_path()

        self._export_dir = os.path.realpath(self._export_dir)

                
    def check_launch(self, args, options):
        """check the validity of the line. args are DIR and MODE
        MODE being NLO, aMC@NLO or aMC@LO. If no mode is passed, aMC@NLO is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if not args:
            if self._done_export:
                args.append(self._done_export[0])
                args.append('aMC@NLO')

                return
            else:
                self.help_launch()
                raise self.InvalidCmd, \
                       'No default location available, please specify location.'
        
        if len(args) > 2:
            self.help_launch()
            return self.InvalidCmd, 'Invalid Syntax: Too many argument'

        elif len(args) == 2:
            if not args[1] in ['NLO', 'aMC@NLO', 'aMC@LO']:
                raise self.InvalidCmd, '%s is not a valid mode, please use "NLO", "aMC@NLO" or "aMC@LO"' % args[1]
        else:
            #check if args[0] is path or mode
            if args[0] in ['NLO', 'aMC@NLO', 'aMC@LO'] and self._done_export:
                args.insert(0, self._done_export[0])
            elif os.path.isdir(args[0]) or os.path.isdir(pjoin(MG5dir, args[0]))\
                    or os.path.isdir(pjoin(MG4dir, args[0])):
                args.append('aMC@NLO')
        mode = args[1]
        
        # search for a valid path
        if os.path.isdir(args[0]):
            path = os.path.realpath(args[0])
        elif os.path.isdir(pjoin(MG5DIR,args[0])):
            path = pjoin(MG5DIR,args[0])
        elif  MG4DIR and os.path.isdir(pjoin(MG4DIR,args[0])):
            path = pjoin(MG4DIR,args[0])
        else:    
            raise self.InvalidCmd, '%s is not a valid directory' % args[0]
                
        # inform where we are for future command
        self._done_export = [path, mode]


    def validate_model(self, loop_type):
        """ Upgrade the model sm to loop_sm if needed """

        if not isinstance(self._curr_model,loop_base_objects.LoopModel) or \
           self._curr_model['perturbation_couplings']==[]:
            if loop_type == 'real':
                    logger.warning(\
                      "Beware that real corrections are generated from a tree-level model.")     
            else:
                if self._curr_model['name']=='sm':
                    logger.warning(\
                      "The default sm model does not allow to generate"+
                      " loop processes. MG5 now loads 'loop_sm' instead.")
                    mg_interface.MadGraphCmd.do_import(self,"model loop_sm")
                else:
                    raise MadGraph5Error(
                      "The model %s cannot handle loop processes"\
                      %self._curr_model['name'])            
    pass

class CheckFKSWeb(mg_interface.CheckValidForCmdWeb, CheckFKS):
    pass

class CompleteFKS(mg_interface.CompleteForCmd):
    
    def complete_display(self, text, line, begidx, endidx):
        "Complete the display command in the context of the FKS interface"

        args = self.split_arg(line[0:begidx])

        if len(args) == 2 and args[1] in ['diagrams', 'processes']:
            return self.list_completion(text, ['born', 'loop', 'virt', 'real'])
        else:
            return mg_interface.MadGraphCmd.complete_display(self, text, line,
                                                                 begidx, endidx)

class HelpFKS(mg_interface.HelpToCmd):

    def help_display(self):   
        mg_interface.MadGraphCmd.help_display(self)
        logger.info("   In aMC@NLO5, after display diagrams, the user can add the option")
        logger.info("   \"born\", \"virt\" or \"real\" to display only the corresponding diagrams.")

    def help_launch(self):
        """help for launch command"""
        _launch_parser.print_help()

class FKSInterface(CheckFKS, CompleteFKS, HelpFKS, mg_interface.MadGraphCmd):
    _fks_display_opts = ['real_diagrams', 'born_diagrams', 'virt_diagrams', 
                         'real_processes', 'born_processes', 'virt_processes']

    def __init__(self, mgme_dir = '', *completekey, **stdin):
        """ Special init tasks for the Loop Interface """

        mg_interface.MadGraphCmd.__init__(self, mgme_dir = '', *completekey, **stdin)
        self.setup()

    def setup(self):
        """ Special tasks when switching to this interface """

        # Refresh all the interface stored value as things like generated
        # processes and amplitudes are not to be reused in between different
        # interfaces
        # Clear history, amplitudes and matrix elements when a model is imported
        # Remove previous imports, generations and outputs from history
        self.clean_history(remove_bef_lb1='import')
        # Reset amplitudes and matrix elements
        self._done_export=False
        self._curr_amps = diagram_generation.AmplitudeList()
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()
        self._v4_export_formats = []
        self._export_formats = [ 'madevent' ]
        if self.options['loop_optimized_output'] and \
                                           not self.options['gauge']=='Feynman':
            # In the open loops method, in order to have a maximum loop numerator
            # rank of 1, one must work in the Feynman gauge
            mg_interface.MadGraphCmd.do_set(self,'gauge Feynman')
        # Set where to look for CutTools installation.
        # In further versions, it will be set in the same manner as _mgme_dir so that
        # the user can chose its own CutTools distribution.
        self._cuttools_dir=str(pjoin(self._mgme_dir,'vendor','CutTools'))
        if not os.path.isdir(pjoin(self._cuttools_dir, 'src','cts')):
            logger.warning(('Warning: Directory %s is not a valid CutTools directory.'+\
                           'Using default CutTools instead.') % \
                             self._cuttools_dir)
            self._cuttools_dir=str(pjoin(self._mgme_dir,'vendor','CutTools'))

    def do_set(self, line, log=True):
        """Set the loop optimized output while correctly switching to the
        Feynman gauge if necessary.
        """

        mg_interface.MadGraphCmd.do_set(self,line,log)
        
        args = self.split_arg(line)
        self.check_set(args)

        if args[0] == 'loop_optimized_output' and eval(args[1]) and \
                                           not self.options['gauge']=='Feynman':
            mg_interface.MadGraphCmd.do_set(self,'gauge Feynman')

    def do_display(self, line, output=sys.stdout):
        # if we arrive here it means that a _fks_display_opts has been chosen
        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_display(args)

        if args[0] in ['diagrams', 'processes']:
            get_amps_dict = {'real': self._fks_multi_proc.get_real_amplitudes,
                             'born': self._fks_multi_proc.get_born_amplitudes,
                             'loop': self._fks_multi_proc.get_virt_amplitudes,
                             'virt': self._fks_multi_proc.get_virt_amplitudes}
        if args[0] == 'diagrams':
            if len(args)>=2 and args[1] in get_amps_dict.keys():
                get_amps = get_amps_dict[args[1]]
                self._curr_amps = get_amps()
                #check that if one requests the virt diagrams, there are virt_amplitudes
                if args[1] in ['virt', 'loop'] and len(self._curr_amps) == 0:
                    raise self.InvalidCmd('No virtuals have been generated')
                self.draw(' '.join(args[2:]))
            else:
                self._curr_amps = get_amps_dict['born']() + get_amps_dict['real']() + \
                                  get_amps_dict['virt']()
                self.draw(' '.join(args[1:]))
            # set _curr_amps back to empty
            self._curr_amps = diagram_generation.AmplitudeList()
                
        elif args[0] == 'processes':
            if len(args)>=2 and args[1] in get_amps_dict.keys():
                get_amps = get_amps_dict[args[1]]
                self._curr_amps = get_amps()
                #check that if one requests the virt diagrams, there are virt_amplitudes
                if args[1] in ['virt', 'loop'] and len(self._curr_amps) == 0:
                    raise self.InvalidCmd('No virtuals have been generated')
                print '\n'.join(amp.nice_string_processes() for amp in self._curr_amps)
            else:
                self._curr_amps = get_amps_dict['born']() + get_amps_dict['real']() + \
                                  get_amps_dict['virt']()
                print '\n'.join(amp.nice_string_processes() for amp in self._curr_amps)
            # set _curr_amps back to empty
            self._curr_amps = diagram_generation.AmplitudeList()

        else:
            mg_interface.MadGraphCmd.do_display(self,line,output)

    def do_add(self, line, *args,**opt):
        
        args = self.split_arg(line)
        # Check the validity of the arguments
        self.check_add(args)

        if args[0] != 'process': 
            raise self.InvalidCmd("The add command can only be used with a process")
        else:
            line = ' '.join(args[1:])
            
        proc_type=self.extract_process_type(line)
        self.validate_model(proc_type[1])

        #now generate the amplitudes as usual
        self.options['group_subprocesses'] = 'False'
        collect_mirror_procs = False
        ignore_six_quark_processes = self.options['ignore_six_quark_processes']
#        super(FKSInterface, self).do_generate(line)
        if ',' in line:
            myprocdef, line = mg_interface.MadGraphCmd.extract_decay_chain_process(self,line)
            if myprocdef.are_decays_perturbed():
                raise MadGraph5Error("Decay processes cannot be perturbed")
        else:
            myprocdef = mg_interface.MadGraphCmd.extract_process(self,line)

        if myprocdef['perturbation_couplings']!=['QCD']:
                raise self.InvalidCmd("FKS for reals only available in QCD for now, you asked %s" \
                        % ', '.join(myprocdef['perturbation_couplings']))

        self._fks_multi_proc.add(fks_base.FKSMultiProcess(myprocdef,
                                   collect_mirror_procs,
                                   ignore_six_quark_processes))

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

        self.options['group_subprocesses'] = False
        # initialize the writer
        if self._export_format in ['NLO']:
            if not self.options['loop_optimized_output']:
                logger.info("Exporting in MadFKS format, starting from born process")
                self._curr_exporter = export_fks.ProcessExporterFortranFKS(\
                                          self._mgme_dir, self._export_dir,
                                          not noclean, 
                                          self.options['complex_mass_scheme'], 
                                          #use MP for HELAS only if there are virtual amps 
                                          len(self._fks_multi_proc.get_virt_amplitudes()) > 0, 
                                          os.path.join(self._mgme_dir, 'Template', 'loop_material'),
                                          self._cuttools_dir)
            
            else:
                logger.info("Exporting in MadFKS format, starting from born process using Optimized Loops")
                self._curr_exporter = export_fks.ProcessOptimizedExporterFortranFKS(\
                                          self._mgme_dir, self._export_dir,
                                          not noclean, 
                                          self.options['complex_mass_scheme'],
                                          #use MP for HELAS only if there are virtual amps 
                                          len(self._fks_multi_proc.get_virt_amplitudes()) > 0, 
                                          os.path.join(self._mgme_dir,'Template/loop_material'),
                                          self._cuttools_dir)
            
        # check if a dir with the same name already exists
        if not force and not noclean and os.path.isdir(self._export_dir)\
               and self._export_format in ['NLO']:
            # Don't ask if user already specified force or noclean
            logger.info('INFO: directory %s already exists.' % self._export_dir)
            logger.info('If you continue this directory will be cleaned')
            answer = self.ask('Do you want to continue?', 'y', ['y','n'], 
                                                timeout=self.options['timeout'])
            if answer != 'y':
                raise self.InvalidCmd('Stopped by user request')
    
        # Make a Template Copy
        if self._export_format in ['NLO']:
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
            if self.options['group_subprocesses'] in [False]:
                group = False
            elif self.options['group_subprocesses'] == 'Auto' and \
                                         self._curr_amps[0].get_ninitial() == 1:
                   group = False 

            cpu_time1 = time.time()
            ndiags = 0
            if not self._curr_matrix_elements.get_matrix_elements():
                if group:
                    raise MadGraph5Error, "Cannot group subprocesses when exporting to NLO"
                else:
                    self._curr_matrix_elements = \
                             fks_helas.FKSHelasMultiProcess(\
                                self._fks_multi_proc, 
                                loop_optimized= self.options['loop_optimized_output'])

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

            #_curr_matrix_element is a FKSHelasMultiProcess Object 
            self._fks_directories = []
            for ime, me in \
                enumerate(self._curr_matrix_elements.get('matrix_elements')):
                #me is a FKSHelasProcessFromReals
                calls = calls + \
                        self._curr_exporter.generate_directories_fks(\
                            me, self._curr_fortran_model, ime, path)
                self._fks_directories.extend(self._curr_exporter.fksdirs)
            card_path = os.path.join(path, os.path.pardir, 'SubProcesses', \
                                     'procdef_mg5.dat')
            if self.options['loop_optimized_output'] and \
                    len(self._curr_matrix_elements.get_virt_matrix_elements()) > 0:
                print    len(self._curr_matrix_elements.get_virt_matrix_elements()) > 0
                self._curr_exporter.write_coef_specs_file(\
                        self._curr_matrix_elements.get_virt_matrix_elements())
            if self._generate_info:
                self._curr_exporter.write_procdef_mg5(card_path, #
                                self._curr_model['name'],
                                self._generate_info)
                try:
                    cmd.Cmd.onecmd(self, 'history .')
                except:
                    pass
            
        cpu_time1 = time.time()


    def do_launch(self, line):
        """Ask for editing the parameters and then execute the code (NLO or aMC@(N)LO)
        """
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _launch_parser.parse_args(argss)
        self.check_launch(argss, options)
        mode = argss[1]
        self.orig_dir = os.path.join(os.getcwd())
        self.me_dir = os.path.join(os.getcwd(), argss[0])
        self.ask_run_configuration(mode)
        self.compile(mode, options)
        self.run(mode, options)

        os.chdir(self.orig_dir)


    def update_random_seed(self):
        """Update random number seed with the value from the run_card. 
        If this is 0, update the number according to a fresh one"""
        run_card = pjoin(self.me_dir, 'Cards', 'run_card.dat')
        if not os.path.exists(run_card):
            raise MadGraph5Error('%s is not a valid run_card' % run_card)
        file = open(run_card)
        lines = file.read().split('\n')
        file.close()
        iseed = 0
        for line in lines:
            if len(line.split()) > 2:
                if line.split()[2] == 'iseed':
                    iseed = int(line.split()[0])
        if iseed != 0:
            os.system('echo "r=%d > %s"' \
                    % (iseed, pjoin(self.me_dir, 'SubProcesses', 'randinit')))
        else:
            randinit = open(pjoin(self.me_dir, 'SubProcesses', 'randinit'))
            iseed = int(randinit.read()[2:]) + 1
            randinit.close()
            randinit = open(pjoin(self.me_dir, 'SubProcesses', 'randinit'), 'w')
            randinit.write('r=%d' % iseed)
            randinit.close()


    def run(self, mode, options):
        """runs aMC@NLO"""
        logger.info('Starting run')

        if self.options['run_mode'] == '1':
            cluster_name = self.options['cluster_type']
            self.cluster = cluster.from_name[cluster_name](self.options['cluster_queue'])
        self.update_random_seed()
        os.chdir(pjoin(self.me_dir, 'SubProcesses'))
        #find and keep track of all the jobs
        job_dict = {}
        p_dirs = [file for file in os.listdir('.') if file.startswith('P') and os.path.isdir(file)]
        for dir in p_dirs:
            os.chdir(pjoin(self.me_dir, 'SubProcesses', dir))
            job_dict[dir] = [file for file in os.listdir('.') if file.startswith('ajob')] 

        os.chdir(pjoin(self.me_dir, 'SubProcesses'))
        mcatnlo_status = ['Setting up grid', 'Computing upper envelope', 'Generating events']

        if mode == 'NLO':
            logger.info('Doing fixed order NLO')
            logger.info('   Cleaning previous results')
            os.system('rm -rf P*/grid_G* P*/novB_G* P*/viSB_G*')

            logger.info('   Setting up grid')
            self.run_all(job_dict, ['0', 'grid', '0'])
            self.wait_for_complete()
            os.system('./combine_results_FO.sh grid*')

            logger.info('   Runnning subtracted reals')
            self.run_all(job_dict, ['0', 'novB', '0', 'grid'])

            logger.info('   Runnning virtuals')
            self.run_all(job_dict, ['0', 'viSB', '0', 'grid'])

            self.wait_for_complete()
            os.system('./combine_results_FO.sh viSB* novB*')
            return

        elif mode == 'aMC@NLO':
            logger.info('Doing NLO matched to parton shower')
            shower, nevents = self.read_shower_events(pjoin(self.me_dir, 'Cards', 'run_card.dat'))
            logger.info('   Cleaning previous results')
            os.system('rm -rf P*/GF* P*/GV*')

            for i, status in enumerate(mcatnlo_status):
                logger.info('   %s' % status)
                logger.info('     Running subtracted reals')
                self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), shower, 'novB', i) 
                self.run_all(job_dict, ['2', 'F', '%d' % i])

                logger.info('     Running virtuals')
                self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), shower, 'viSB', i) 
                self.run_all(job_dict, ['2', 'V', '%d' % i])

                self.wait_for_complete()
                if i < 2:
                    os.system('./combine_results.sh %d %d GF* GV*' % (i, nevents))

        elif mode == 'aMC@LO':
            logger.info('Doing LO matched to parton shower')
            shower, nevents = self.read_shower_events(
                        pjoin(self.me_dir, 'Cards', 'run_card.dat'))
            logger.info('   Cleaning previous results')
            os.system('rm -rf P*/GB*')
            for i, status in enumerate(mcatnlo_status):
                logger.info('   %s at LO' % status)
                self.write_madinMMC_file(
                        pjoin(self.me_dir, 'SubProcesses'), shower, 'born', i) 
                self.run_all(job_dict, ['2', 'B', '%d' % i])
                self.wait_for_complete()

                if i < 2:
                    os.system('./combine_results.sh %d %d GB*' % (i, nevents))

        if self.options['run_mode'] == '1':
            #if cluster run, wait 15 sec so that event files are transferred back
            logger.info('Waiting while files are trasferred back from the cluster nodes')
            time.sleep(15)

        self.run_reweight()

        os.system('make collect_events > %s' % \
                pjoin (self.me_dir, 'log_collect_events.txt'))
        os.system('echo "1" | ./collect_events > %s' % \
                pjoin (self.me_dir, 'log_collect_events.txt'))

        count = 1
        if not os.path.exists(pjoin(self.me_dir, 'SubProcesses', 'allevents_0_001')):
            raise MadGraph5Error('An error occurred during event generation. ' + \
                    'The event file has not been created. Check log_collect_events.txt')
        while os.path.exists(pjoin(self.me_dir, 'Events', 'events_%d' % count)):
            count += 1
        evt_file = pjoin(self.me_dir, 'Events', 'events_%d' % count)
        os.system('mv %s %s' % 
            (pjoin(self.me_dir, 'SubProcesses', 'allevents_0_001'), evt_file))
        logger.info('The %s file has been generated.\nIt contains %d %s events to be showered' \
                % (evt_file, nevents, mode[4:]))


    def run_reweight(self):
        """runs the reweight_xsec_events eecutables on each sub-event file generated
        to compute on the fly scale and/or PDF uncertainities"""

        reweight_log = pjoin(self.me_dir, 'compile_reweight.log')
        #read the nevents_unweighted file to get the list of event files
        file = open(pjoin(self.me_dir, 'SubProcesses', 'nevents_unweighted'))
        lines = file.read().split('\n')
        file.close()
        evt_files = [line.split()[0] for line in lines if line]
        for i, evt_file in enumerate(evt_files):
            path, evt = os.path.split(evt_file)
            if self.options['run_mode'] == '0':
                logger.info('Reweighting file %s (%d/%d)' \
                        %(evt_file, i + 1, len(evt_files)))
                os.chdir(pjoin(self.me_dir, 'SubProcesses', path))
                os.system('echo "%s \n 1" |../reweight_xsec_events >> %s' \
                        % (evt, reweight_log))
                #check that the new event file is complete
                last_line = subprocess.Popen('tail -n1 %s.rwgt ' % evt ,
                    shell = True, stdout = subprocess.PIPE).stdout.read().strip()
                if last_line != "</LesHouchesEvents>":
                    raise MadGraph5Error('An error occurred during reweight.' + \
                            ' Check %s for details' % reweight_log)

        newfile = open(pjoin(self.me_dir, 'SubProcesses', 'nevents_unweighted'), 'w')
        for line in lines:
            if line:
                newfile.write(line.replace(line.split()[0], line.split()[0] + '.rwgt') + '\n')
        newfile.close()



        os.chdir(pjoin(self.me_dir, 'SubProcesses'))


    def wait_for_complete(self):
        """this function waits for jobs on cluster to complete their run."""

        # do nothing if running serially
        if self.options['run_mode'] == '0':
            return

        idle = 1
        run = 1
        logger.info('     Waiting for submitted jobs to complete')
        while idle + run > 0:
            idle, run, finish, fail = self.cluster.control('')
            time.sleep(5)
            logger.info('     Job status: %d idle, %d running, %d failed, %d completed' \
                    % (idle, run, fail, finish))
        #reset the cluster after completion
#        self.cluster.submitted = 0
#        self.cluster.submitted_ids = []


    def run_all(self, job_dict, args):
        """runs the jobs in job_dict (organized as folder: [job_list]), with arguments args"""
        njobs = sum(len(jobs) for jobs in job_dict.values())
        ijob = 0
        for dir, jobs in job_dict.items():
            os.chdir(pjoin(self.me_dir, 'SubProcesses', dir))
            for job in jobs:
                self.run_exe(job, args)
                # print some statistics if running serially
                ijob += 1
                if self.options['run_mode'] == '0':
                    logger.info('%d/%d completed\n' \
                            % (ijob, njobs))

        os.chdir(pjoin(self.me_dir, 'SubProcesses'))


    def run_exe(self, exe, args):
        """this basic function launch locally/on cluster exe with args as argument."""
        # first test that exe exists:
        if not os.path.exists(exe):
            raise MadGraph5Error('Cannot find executable %s in %s' \
                % (exe, os.getcwd()))
        # check that the executable has exec permissions
        if not os.access(exe, os.X_OK):
            os.system('chmod +x %s' % exe)
        # finally run it
        if self.options['run_mode'] == '0':
            #this is for the serial run
            os.system('./%s %s ' % (exe, ' '.join(args)))
        elif self.options['run_mode'] == '1':
            #this is for the cluster run
            self.cluster.submit(exe, args)

        elif self.options['run_mode'] == '2':
            #this is for the multicore run
            raise MadGraph5Error('Multicore run not yet available for aMC@NLO')







    def read_shower_events(self, run_card, verbose=True):
        """read the parton shower and the requested number of events in the run_card"""
        if not os.path.exists(run_card):
            raise MadGraph5Error('%s is not a valid run_card' % run_card)
        file = open(run_card)
        lines = file.read().split('\n')
        file.close()
        nevents = 0
        shower = ''
        for line in lines:
            if len(line.split()) > 2:
                if line.split()[2] == 'parton_shower':
                    shower = line.split()[0]
                if line.split()[2] == 'nevents':
                    nevents = int(line.split()[0])

        if shower and nevents and verbose:
            logger.info('Input read from the run_card.dat: \n Generating %d events for shower %s' \
                    %(nevents, shower))
        elif not shower or not nevents:
            raise MadGraph5Error('Falied to read shower and number of events from the run_card.dat')

        return shower, nevents


    def write_madinMMC_file(self, path, shower, run_mode, mint_mode):
        """writes the madinMMC_?.2 file"""
        #check the validity of the arguments
        shower_list = ['HERWIG6', 'HERWIGPP', 'PYTHIA6Q', 'PYTHIA6PT', 'PYTHIA8']
        if not shower in shower_list:
            raise MadGraph5Error('%s is not a valid parton shower. Please use one of the following: %s' \
                    % (shower, ', '.join(shower_list)))
        run_modes = ['born', 'virt', 'novi', 'all', 'viSB', 'novB']
        if run_mode not in run_modes:
            raise MadGraph5Error('%s is not a valid mode for run. Please use one of the following: %s' \
                    % (run_mode, ', '.join(run_modes)))
        mint_modes = [0, 1, 2]
        if mint_mode not in mint_modes:
            raise MadGraph5Error('%s is not a valid mode for mintMC. Please use one of the following: %s' \
                    % (mint_mode, ', '.join(mint_modes)))
        if run_mode in ['born']:
            name_suffix = 'B'
        elif run_mode in ['virt', 'viSB']:
            name_suffix = 'V'
        else:
            name_suffix = 'F'

        content = \
"""-1 12      ! points, iterations
0.05       ! desired fractional accuracy
1 -0.1     ! alpha, beta for Gsoft
-1 -0.1    ! alpha, beta for Gazi
1          ! Suppress amplitude (0 no, 1 yes)?
0          ! Exact helicity sum (0 yes, n = number/event)?
1          ! Enter Configuration Number:
%1d          ! MINT imode: 0 to set-up grids, 1 to perform integral, 2 generate events
1 1 1      ! if imode is 1: Folding parameters for xi_i, phi_i and y_ij
%s        ! all, born, real, virt
""" \
                    % (mint_mode, run_mode)
        file = open(pjoin(path, 'madinMMC_%s.2' % name_suffix), 'w')
        file.write(content)
        file.close()



    def compile(self, mode, options):
        """compiles aMC@NLO to compute either NLO or NLO matched to shower, as
        specified in mode"""
        #define a bunch of log files
        amcatnlo_log = pjoin(self.me_dir, 'compile_amcatnlo.log')
        madloop_log = pjoin(self.me_dir, 'compile_madloop.log')
        reweight_log = pjoin(self.me_dir, 'compile_reweight.log')
        gensym_log = pjoin(self.me_dir, 'gensym.log')
        test_log = pjoin(self.me_dir, 'test.log')

        #define which executable/tests to compile
        if mode =='NLO':
            exe = 'madevent_vegas'
            tests = ['test_ME']
        if mode in ['aMC@NLO', 'aMC@LO']:
            exe = 'madevent_mintMC'
            tests = ['test_ME', 'test_MC']
        #directory where to compile exe
        os.chdir(pjoin(self.me_dir, 'SubProcesses'))
        p_dirs = [file for file in os.listdir('.') if file.startswith('P') and os.path.isdir(file)]
        # if --nocompile option is specified, check here that all exes exists. 
        # If they exists, return
        if all([os.path.exists(pjoin(self.me_dir, 'SubProcesses', p_dir, exe)) \
                for p_dir in p_dirs]) and options.__dict__['nocompile'] \
                and not options.__dict__['tests']:
            return

        libdir = pjoin(self.me_dir, 'lib')
        # read the run_card to find if lhapdf is used or not
        run_card_file = open(pjoin(self.me_dir, 'Cards','run_card.dat'))
        found = False
        while not found:
            line = run_card_file.readline()
            if 'pdlabel' in line:
                found = True
        run_card_file.close()
        # rm links to lhapdflib/ PDFsets if exist
        if os.path.islink(pjoin(libdir, 'libLHAPDF.a')):
            os.remove(pjoin(libdir, 'libLHAPDF.a'))
        if os.path.islink(pjoin(libdir, 'PDFsets')):
            os.remove(pjoin(libdir, 'PDFsets'))

        if line.split()[0] == '\'lhapdf\'':
            logger.info('Using LHAPDF interface for PDFs')
            lhalibdir = subprocess.Popen('%s --libdir' % self.options['lhapdf'],
                    shell = True, stdout = subprocess.PIPE).stdout.read().strip()
            lhasetsdir = subprocess.Popen('%s --pdfsets-path' % self.options['lhapdf'], 
                    shell = True, stdout = subprocess.PIPE).stdout.read().strip()
            os.symlink(pjoin(lhalibdir, 'libLHAPDF.a'), pjoin(libdir, 'libLHAPDF.a'))
            os.symlink(lhasetsdir, pjoin(libdir, 'PDFsets'))
            os.putenv('lhapdf', 'True')
        else:
            logger.info('Using built-in libraries for PDFs')
            os.unsetenv('lhapdf')

        # make Source
        logger.info('Compiling source...')
        os.chdir(pjoin(self.me_dir, 'Source'))
        os.system('make > %s 2>&1' % amcatnlo_log)
        if os.path.exists(pjoin(libdir, 'libdhelas.a')) \
          and os.path.exists(pjoin(libdir, 'libgeneric.a')) \
          and os.path.exists(pjoin(libdir, 'libmodel.a')) \
          and os.path.exists(pjoin(libdir, 'libpdf.a')):
            logger.info('          ...done, continuing with P* directories')
        else:
            raise MadGraph5Error('Compilation failed, check %s for details' % amcatnlo_log)

        # make and run tests (if asked for), gensym and make madevent in each dir
        for p_dir in p_dirs:
            logger.info(p_dir)
            this_dir = pjoin(self.me_dir, 'SubProcesses', p_dir) 
            os.chdir(this_dir)
            # compile and run tests if asked for
            if options.__dict__['tests']:
                for test in tests:
                    logger.info('   Compiling %s...' % test)
                    os.system('make %s >> %s 2>&1 ' % (test, test_log))
                    if not os.path.exists(pjoin(this_dir, test)):
                        raise MadGraph5Error('Compilation failed, check %s for details' \
                                % test_log)
                    logger.info('   Running %s...' % test)
                    self.write_test_input(test)
                    input = pjoin(self.me_dir, '%s_input.txt' % test)
                    #this can be improved/better written to handle the output
                    os.system('./%s < %s | tee -a %s | grep "Fraction of failures"' \
                            % (test, input, test_log))
                    os.system('rm -f %s' % input)

            logger.info('   Compiling gensym...')
            os.system('make gensym >> %s 2>&1 ' % amcatnlo_log)
            if not os.path.exists(pjoin(this_dir, 'gensym')):
                raise MadGraph5Error('Compilation failed, check %s for details' % amcatnlo_log)

            logger.info('   Running gensym...')
            os.system('echo %s | ./gensym >> %s' % (self.options['run_mode'], gensym_log)) 
            #compile madloop library
            v_dirs = [file for file in os.listdir('.') if file.startswith('V') and os.path.isdir(file)]
            for v_dir in v_dirs:
                os.putenv('madloop', 'true')
                logger.info('   Compiling MadLoop library in %s' % v_dir)
                madloop_dir = pjoin(this_dir, v_dir)
                os.chdir(madloop_dir)
                os.system('make >> %s 2>&1' % madloop_log)
                if not os.path.exists(pjoin(this_dir, 'libMadLoop.a')):
                    raise MadGraph5Error('Compilation failed, check %s for details' % madloop_log)
            os.chdir(this_dir)
            logger.info('   Compiling %s' % exe)
            os.system('make %s >> %s 2>&1' % (exe, amcatnlo_log))
            os.unsetenv('madloop')
            if not os.path.exists(pjoin(this_dir, exe)):
                raise MadGraph5Error('Compilation failed, check %s for details' % amcatnlo_log)
            if mode in ['aMC@NLO', 'aMC@LO']:
                logger.info('   Compiling reweight_xsec_events')
                os.system('make reweight_xsec_events >> %s 2>&1' % (reweight_log))
                if not os.path.exists(pjoin(this_dir, 'reweight_xsec_events')):
                    raise MadGraph5Error('Compilation failed, check %s for details' % reweight_log)

        os.chdir(pjoin(self.me_dir))


    def write_test_input(self, test):
        """write the input files to run test_ME/MC"""
        content = "-2 -2\n" #generate randomly energy/angle
        content+= "100 100\n" #run 100 points for soft and collinear tests
        content+= "0\n" #sum over helicities
        content+= "0\n" #all FKS configs
        content+= '\n'.join(["-1"] * 50) #random diagram
        
        file = open(pjoin(self.me_dir, '%s_input.txt' % test), 'w')
        if test == 'test_ME':
            file.write(content)
        elif test == 'test_MC':
            shower, events = self.read_shower_events(\
                    pjoin(self.me_dir, 'Cards', 'run_card.dat'), verbose=False)
            MC_header = "%s\n " % shower + \
                        "1 \n1 -0.1\n-1 -0.1\n"
            file.write(MC_header + content)
        file.close()


    ############################################################################
    def ask_run_configuration(self, mode):
        """Ask the question when launching generate_events/multi_run"""
        
        logger.info('Will run in mode %s' % mode)
        cards = ['param_card.dat', 'run_card.dat']

        def get_question(mode):
            # Ask the user if he wants to edit any of the files
            #First create the asking text
            question = """Do you want to edit one cards (press enter to bypass editing)?
  1 / param   : param_card.dat (be carefull about parameter consistency, especially widths)
  2 / run     : run_card.dat\n"""
            possible_answer = ['0','done', 1, 'param', 2, 'run']
            card = {0:'done', 1:'param', 2:'run'}
            # Add the path options
            question += '  Path to a valid card.\n'
            return question, possible_answer, card
        
        # Loop as long as the user is not done.
        answer = 'no'
        while answer != 'done':
            question, possible_answer, card = get_question(mode)
            answer = self.ask(question, '0', possible_answer, timeout=int(1.5*self.options['timeout']), path_msg='enter path')
            if answer.isdigit():
                answer = card[int(answer)]
            if answer == 'done':
                return
            if not os.path.isfile(answer):
                if answer != 'trigger':
                    path = pjoin(self.me_dir,'Cards','%s_card.dat' % answer)
                else:
                    path = pjoin(self.me_dir,'Cards','delphes_trigger.dat')
                self.exec_cmd('open %s' % path)                    
            else:
                # detect which card is provided
                card_name = answer + 'card.dat'
                    
   
class FKSInterfaceWeb(mg_interface.CheckValidForCmdWeb, FKSInterface):
    pass

_launch_usage = "launch [DIRPATH] [MODE] [options]\n" + \
                "-- execute the aMC@NLO output present in DIRPATH\n" + \
                "   By default DIRPATH is the latest created directory\n" + \
                "   MODE can be either NLO, aMC@NLO or aMC@LO (if omitted, it is set to aMC@NLO)\n"

_launch_parser = optparse.OptionParser(usage=_launch_usage)
_launch_parser.add_option("-n", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found, " + \
                            "or with --tests")
_launch_parser.add_option("-t", "--tests", default=False, action='store_true',
                            help="Run soft/collinear tests to check the NLO/MC subtraction terms." + \
                                 " MC tests are skipped in NLO mode.") 


