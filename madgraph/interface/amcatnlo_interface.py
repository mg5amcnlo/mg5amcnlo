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
import pydoc
import sys
import time
import optparse
import subprocess

import madgraph
from madgraph import MG4DIR, MG5DIR, MadGraph5Error
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.madevent_interface as me_interface
import madgraph.interface.amcatnlo_run_interface as run_interface
import madgraph.interface.Loop_interface as Loop_interface
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.export_v4 as export_v4
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.various.cluster as cluster
import madgraph.various.misc as misc

#usefull shortcut
pjoin = os.path.join


logger = logging.getLogger('cmdprint') # -> stdout
logger_stderr = logging.getLogger('fatalerror') # ->stderr

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
        MODE being LO, NLO, aMC@NLO or aMC@LO. If no mode is passed, aMC@NLO is used"""
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
            if not args[1] in ['LO', 'NLO', 'aMC@NLO', 'aMC@LO']:
                raise self.InvalidCmd, '%s is not a valid mode, please use "LO", "NLO", "aMC@NLO" or "aMC@LO"' % args[1]
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

        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd, 'options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.'
        if options['noreweight'] and options['reweightonly']:
            raise self.InvalidCmd, 'options -R (--noreweight) and -R (--reweightonly)' + \
                    ' are not compatible. Please choose one.'
        if mode == 'NLO' and options['reweightonly']:
            raise self.InvalidCmd, 'option -r (--reweightonly) needs mode "aMC@NLO" or "aMC@LO"'



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
        """Complete the display command in the context of the FKS interface"""

        args = self.split_arg(line[0:begidx])

        if len(args) == 2 and args[1] in ['diagrams', 'processes']:
            return self.list_completion(text, ['born', 'loop', 'virt', 'real'])
        else:
            return mg_interface.MadGraphCmd.complete_display(self, text, line,
                                                                 begidx, endidx)


    def complete_output(self, text, line, begidx, endidx):
        """Complete the output command in the context of the FKS interface"""
        #don't propose directory use by MG_ME
        forbidden_names = ['MadGraphII', 'Template', 'pythia-pgs', 'CVS',
                            'Calculators', 'MadAnalysis', 'SimpleAnalysis',
                            'mg5', 'DECAY', 'EventConverter', 'Models',
                            'ExRootAnalysis', 'HELAS', 'Transfer_Fct', 'aloha', 
                            'madgraph', 'bin', 'tests', 'input', 'vendor', 'models']
        
        #name of the run =>proposes old run name
        args = self.split_arg(line[0:begidx])
        if len(args) >= 1: 
            if len(args) > 1 and args[1] == 'aloha':
                try:
                    return self.aloha_complete_output(text, line, begidx, endidx)
                except Exception, error:
                    print error
            # Directory continuation
            if args[-1].endswith(os.path.sep):
                return [name for name in self.path_completion(text,
                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                        only_dirs = True) if name not in forbidden_names]

            # directory names
            content = [name for name in self.path_completion(text, '.', only_dirs = True) \
                       if name not in forbidden_names]
            return self.list_completion(text, content)

class HelpFKS(mg_interface.HelpToCmd):

    def help_display(self):   
        mg_interface.MadGraphCmd.help_display(self)
        logger.info("   In aMC@NLO5, after display diagrams, the user can add the option")
        logger.info("   \"born\", \"virt\" or \"real\" to display only the corresponding diagrams.")

    def help_launch(self):
        """help for launch command"""
        _launch_parser.print_help()

class aMCatNLOInterface(CheckFKS, CompleteFKS, HelpFKS, mg_interface.MadGraphCmd):
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
        self.clean_history(remove_bef_last='import')
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

        if args[0] in ['diagrams', 'processes', 'diagrams_text']:
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

            if args[0] == 'diagrams_text':
                if len(args)>=2 and args[1] in get_amps_dict.keys():
                    get_amps = get_amps_dict[args[1]]
                    self._curr_amps = get_amps()
                    #check that if one requests the virt diagrams, there are virt_amplitudes
                    if args[1] in ['virt', 'loop'] and len(self._curr_amps) == 0:
                        raise self.InvalidCmd('No virtuals have been generated')
                    text = "\n".join([amp.nice_string() for amp in self._curr_amps])
                else:
                    text = 'Born diagrams:\n' 
                    text += '\n'.join(amp.nice_string() for amp in get_amps_dict['born']())
                    text += '\n\nReal diagrams:'
                    text += '\n'.join(amp.nice_string() for amp in get_amps_dict['real']())
                    text += '\n\nLoop diagrams:\n'
                    text += '\n'.join(amp.nice_string() for amp in get_amps_dict['virt']())
                pydoc.pager(text)

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
                    print 'Born processes:'
                    print '\n'.join(amp.nice_string_processes() for amp in get_amps_dict['born']())
                    print 'Real processes:'
                    print '\n'.join(amp.nice_string_processes() for amp in get_amps_dict['real']())
                    print 'Loop processes:'
                    print '\n'.join(amp.nice_string_processes() for amp in get_amps_dict['virt']())
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
                           remove_bef_last='generate')
        
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
            self._curr_exporter = export_v4.ExportV4Factory(\
                                          self, noclean, output_type='amcatnlo')
            
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
        old_cwd = os.getcwd()
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _launch_parser.parse_args(argss)
        options = options.__dict__
        self.check_launch(argss, options)
        run = run_interface.aMCatNLOCmd(me_dir = pjoin(os.getcwd(), argss[0]))
        
        mode = argss[1]
        if options['multicore']:
            run.cluster_mode = 2
        elif options['cluster']:
            run.cluster_mode = 1
        else:
            run.cluster_mode = int(self.options['run_mode'])

        if self.options['automatic_html_opening']:
            misc.open_file(os.path.join(os.getcwd(), argss[0], 'crossx.html'))
            self.options['automatic_html_opening'] = False

        run.ask_run_configuration(mode)
        run.compile(mode, options) 
        evt_file = run.run(mode, options)
        if run.check_mcatnlo_dir():
            run.run_mcatnlo(evt_file)
        os.chdir(old_cwd)
                    
   
class aMCatNLOInterfaceWeb(mg_interface.CheckValidForCmdWeb, aMCatNLOInterface):
    pass

_launch_usage = "launch [DIRPATH] [MODE] [options]\n" + \
                "-- execute the aMC@NLO output present in DIRPATH\n" + \
                "   By default DIRPATH is the latest created directory\n" + \
                "   MODE can be either LO, NLO, aMC@NLO or aMC@LO (if omitted, it is set to aMC@NLO)\n"

_launch_parser = optparse.OptionParser(usage=_launch_usage)
_launch_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_launch_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_launch_parser.add_option("-n", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found")
_launch_parser.add_option("-r", "--reweightonly", default=False, action='store_true',
                            help="Skip integration and event generation, just run reweight on the" + \
                                 " latest generated event files (see list in SubProcesses/nevents_unweighted)")
_launch_parser.add_option("-R", "--noreweight", default=False, action='store_true',
                            help="Skip file reweighting")

