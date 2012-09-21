################################################################################
#
# Copyright (c) 2011 The MadGraph Development team and Contributors
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
"""A user friendly command line interface to access MadGraph features.
   Uses the cmd package for command interpretation and tab completion.
"""
from __future__ import division

import atexit
import glob
import logging
import math
import optparse
import os
import pydoc
import random
import re
import shutil
import subprocess
import sys
import traceback
import time

try:
    import readline
    GNU_SPLITTING = ('GNU' in readline.__doc__)
except:
    GNU_SPLITTING = True

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.split(root_path)[0]
sys.path.insert(0, os.path.join(root_path,'bin'))

# usefull shortcut
pjoin = os.path.join
# Special logger for the Cmd Interface
logger = logging.getLogger('madevent.stdout') # -> stdout
logger_stderr = logging.getLogger('madevent.stderr') # ->stderr
 
try:
    # import from madgraph directory
    import madgraph.interface.extended_cmd as cmd
    import madgraph.iolibs.files as files
    import madgraph.various.cluster as cluster
    import madgraph.various.misc as misc

    import models.check_param_card as check_param_card    
    from madgraph import InvalidCmd, aMCatNLOError
    aMCatNLO = False
except Exception, error:
    if __debug__:
        print error
    # import from madevent directory
    import internal.extended_cmd as cmd
    import internal.misc as misc    
    from internal import InvalidCmd, MadGraph5Error
    import internal.files as files
    import internal.cluster as cluster
    aMCatNLO = True



class aMCatNLOError(Exception):
    pass

#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(cmd.Cmd):
    """Particularisation of the cmd command for aMCatNLO"""

    #suggested list of command
    next_possibility = {
        'start': [],
    }
    
    debug_output = 'ME5_debug'
    error_debug = 'Please report this bug on https://bugs.launchpad.net/madgraph5\n'
    error_debug += 'More information is found in \'%(debug)s\'.\n' 
    error_debug += 'Please attach this file to your report.'

    config_debug = 'If you need help with this issue please contact us on https://answers.launchpad.net/madgraph5\n'


    keyboard_stop_msg = """stopping all operation
            in order to quit madevent please enter exit"""
    
    # Define the Error
    InvalidCmd = InvalidCmd
    ConfigurationError = aMCatNLOError

    def __init__(self, *arg, **opt):
        """Init history and line continuation"""
        
        # Tag allowing/forbiding question
        self.force = False
        
        # If possible, build an info line with current version number 
        # and date, from the VERSION text file
        info = misc.get_pkg_info()
        info_line = ""
        if info and info.has_key('version') and  info.has_key('date'):
            len_version = len(info['version'])
            len_date = len(info['date'])
            if len_version + len_date < 30:
                info_line = "#*         VERSION %s %s %s         *\n" % \
                            (info['version'],
                            (30 - len_version - len_date) * ' ',
                            info['date'])
        else:
            version = open(pjoin(root_path,'MGMEVersion.txt')).readline().strip()
            info_line = "#*         VERSION %s %s                *\n" % \
                            (version, (24 - len(version)) * ' ')    

        # Create a header for the history file.
        # Remember to fill in time at writeout time!
        self.history_header = \
        '#************************************************************\n' + \
        '#*                    MadGraph/aMC@NLO 5                    *\n' + \
        '#*                                                          *\n' + \
        "#*                *                       *                 *\n" + \
        "#*                  *        * *        *                   *\n" + \
        "#*                    * * * * 5 * * * *                     *\n" + \
        "#*                  *        * *        *                   *\n" + \
        "#*                *                       *                 *\n" + \
        "#*                                                          *\n" + \
        "#*                                                          *\n" + \
        info_line + \
        "#*                                                          *\n" + \
        "#*    The MadGraph Development Team - Please visit us at    *\n" + \
        "#*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        '#*                                                          *\n' + \
        '#************************************************************\n' + \
        '#*                                                          *\n' + \
        '#*               Command File for aMCatNLO                  *\n' + \
        '#*                                                          *\n' + \
        '#*     run as ./bin/madevent.py filename                    *\n' + \
        '#*                                                          *\n' + \
        '#************************************************************\n'
        
        if info_line:
            info_line = info_line[1:]

        logger.info(\
        "************************************************************\n" + \
        "*                                                          *\n" + \
        "*           W E L C O M E  to  M A D G R A P H  5          *\n" + \
        "*                       a M C @ N L O                      *\n" + \
        "*                                                          *\n" + \
        "*                 *                       *                *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                     * * * * 5 * * * *                    *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                 *                       *                *\n" + \
        "*                                                          *\n" + \
        info_line + \
        "*                                                          *\n" + \
        "*    The MadGraph Development Team - Please visit us at    *\n" + \
        "*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        "*                                                          *\n" + \
        "*               Type 'help' for in-line help.              *\n" + \
        "*                                                          *\n" + \
        "************************************************************")
        
        cmd.Cmd.__init__(self, *arg, **opt)
        



    def get_history_header(self):
        """return the history header""" 
        return self.history_header % misc.get_time_info()
    
    def stop_on_keyboard_stop(self):
        """action to perform to close nicely on a keyboard interupt"""
        try:
            if hasattr(self, 'results'):
                self.update_status('Stop by the user', level=None, makehtml=False, error=True)
                self.add_error_log_in_html()
        except:
            pass
    
    def postcmd(self, stop, line):
        """ Update the status of  the run for finishing interactive command """
        
        # relaxing the tag forbidding question
        self.force = False
        
        if not self.use_rawinput:
            return stop
        
        
        arg = line.split()
        if  len(arg) == 0:
            return stop
        elif str(arg[0]) in ['exit','quit','EOF']:
            return stop
        
        try:
            self.update_status('Command \'%s\' done.<br> Waiting for instruction.' % arg[0], 
                               level=None, error=True)
        except:
            pass
        

    
    def nice_user_error(self, error, line):
        """If a ME run is currently running add a link in the html output"""

#        self.add_error_log_in_html()
        cmd.Cmd.nice_user_error(self, error, line)            
        
    def nice_config_error(self, error, line):
        """If a ME run is currently running add a link in the html output"""

#        self.add_error_log_in_html()
        cmd.Cmd.nice_config_error(self, error, line)

    def nice_error_handling(self, error, line):
        """If a ME run is currently running add a link in the html output"""

#        self.add_error_log_in_html()            
        cmd.Cmd.nice_error_handling(self, error, line)

        
        
#===============================================================================
# HelpToCmd
#===============================================================================
class HelpToCmd(object):
    """ The Series of help routine for the aMCatNLOCmd"""
    
    def help_launch(self):
        """help for launch command"""
        _launch_parser.print_help()

    def help_compile(self):
        """help for compile command"""
        _compile_parser.print_help()

    def help_generate_events(self):
        """help for generate_events command"""
        _generate_events_parser.print_help()

    def help_calculate_xsect(self):
        """help for generate_events command"""
        _calculate_xsect_parser.print_help()

    def help_run_mcatnlo(self):
        """help for run_mcatnlo command"""
        logger.info('syntax: run_mcatnlo EVT_FILE')
        logger.info('-- do shower/hadronization on parton-level file EVT_FILE')
        logger.info('   all the information (e.g. number of events, MonteCarlo, ...)')
        logger.info('   are directly read from the header of the file')

    
    def help_open(self):
        logger.info("syntax: open FILE  ")
        logger.info("-- open a file with the appropriate editor.")
        logger.info('   If FILE belongs to index.html, param_card.dat, run_card.dat')
        logger.info('   the path to the last created/used directory is used')

    def help_set(self):
        logger.info("syntax: set %s argument" % "|".join(self._set_options))
        logger.info("-- set options")
        logger.info("   stdout_level DEBUG|INFO|WARNING|ERROR|CRITICAL")
        logger.info("     change the default level for printed information")
        logger.info("   timeout VALUE")
        logger.info("      (default 20) Seconds allowed to answer questions.")
        logger.info("      Note that pressing tab always stops the timer.")        

    def run_options_help(self, data):
        if data:
            logger.info('-- local options:')
            for name, info in data:
                logger.info('      %s : %s' % (name, info))
        
        logger.info("-- session options:")
        logger.info("      Note that those options will be kept for the current session")      
        logger.info("      --cluster : Submit to the  cluster. Current cluster: %s" % self.options['cluster_mode'])
        logger.info("      --multicore : Run in multi-core configuration")
        logger.info("      --nb_core=X : limit the number of core to use to X.")
        


       
#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(object):
    """ The Series of check routine for the aMCatNLOCmd"""

    def check_set(self, args):
        """ check the validity of the line"""
        
        if len(args) < 2:
            self.help_set()
            raise self.InvalidCmd('set needs an option and an argument')

        if args[0] not in self._set_options + self.options.keys():
            self.help_set()
            raise self.InvalidCmd('Possible options for set are %s' % \
                                  self._set_options)
        
        if args[0] in ['stdout_level']:
            if args[1] not in ['DEBUG','INFO','WARNING','ERROR','CRITICAL']:
                raise self.InvalidCmd('output_level needs ' + \
                                      'a valid level')  
                
        if args[0] in ['timeout']:
            if not args[1].isdigit():
                raise self.InvalidCmd('timeout values should be a integer')   


    def check_open(self, args):
        """ check the validity of the line """
        
        if len(args) != 1:
            self.help_open()
            raise self.InvalidCmd('OPEN command requires exactly one argument')

        if args[0].startswith('./'):
            if not os.path.isfile(args[0]):
                raise self.InvalidCmd('%s: not such file' % args[0])
            return True

        # if special : create the path.
        if not self.me_dir:
            if not os.path.isfile(args[0]):
                self.help_open()
                raise self.InvalidCmd('No MadEvent path defined. Unable to associate this name to a file')
            else:
                return True
            
        path = self.me_dir
        if os.path.isfile(os.path.join(path,args[0])):
            args[0] = os.path.join(path,args[0])
        elif os.path.isfile(os.path.join(path,'Cards',args[0])):
            args[0] = os.path.join(path,'Cards',args[0])
        elif os.path.isfile(os.path.join(path,'HTML',args[0])):
            args[0] = os.path.join(path,'HTML',args[0])
        # special for card with _default define: copy the default and open it
        elif '_card.dat' in args[0]:   
            name = args[0].replace('_card.dat','_card_default.dat')
            if os.path.isfile(os.path.join(path,'Cards', name)):
                files.cp(path + '/Cards/' + name, path + '/Cards/'+ args[0])
                args[0] = os.path.join(path,'Cards', args[0])
            else:
                raise self.InvalidCmd('No default path for this file')
        elif not os.path.isfile(args[0]):
            raise self.InvalidCmd('No default path for this file') 


    def check_run_mcatnlo(self, args, options):
        """Check the validity of the line. args[0] is the event file"""
        if len(args) == 0:
            self.help_run_mcatnlo()
            raise self.InvalidCmd, 'Invalid syntax, please specify the file to shower'
        if not os.path.exists(pjoin(os.getcwd(), args[0])):
            raise self.InvalidCmd, 'file %s does not exists' % \
                            pjoin(os.getcwd(), args[0])
    

    def check_calculate_xsect(self, args, options):
        """check the validity of the line. args is ORDER,
        ORDER being LO or NLO. If no mode is passed, NLO is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if not args:
            args.append('NLO')
            return
        
        if len(args) > 1:
            self.help_calculate_xsect()
            raise self.InvalidCmd, 'Invalid Syntax: Too many argument'

        elif len(args) == 1:
            if not args[0] in ['NLO', 'LO']:
                raise self.InvalidCmd, '%s is not a valid mode, please use "LO" or "NLO"' % args[1]
        mode = args[0]
        
        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd, 'options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.'


    def check_generate_events(self, args, options):
        """check the validity of the line. args is ORDER,
        ORDER being LO or NLO. If no mode is passed, NLO is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if not args:
            args.append('NLO')
            return
        
        if len(args) > 1:
            self.help_generate_events()
            raise self.InvalidCmd, 'Invalid Syntax: Too many argument'

        elif len(args) == 1:
            if not args[0] in ['NLO', 'LO']:
                raise self.InvalidCmd, '%s is not a valid mode, please use "LO" or "NLO"' % args[1]
        mode = args[0]
        
        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd, 'options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.'
        if options['noreweight'] and options['reweightonly']:
            raise self.InvalidCmd, 'options -R (--noreweight) and -R (--reweightonly)' + \
                    ' are not compatible. Please choose one.'


    def check_launch(self, args, options):
        """check the validity of the line. args is MODE
        MODE being LO, NLO, aMC@NLO or aMC@LO. If no mode is passed, aMC@NLO is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if not args:
            args.append('aMC@NLO')
            return
        
        if len(args) > 1:
            self.help_launch()
            raise self.InvalidCmd, 'Invalid Syntax: Too many argument'

        elif len(args) == 1:
            if not args[0] in ['LO', 'NLO', 'aMC@NLO', 'aMC@LO']:
                raise self.InvalidCmd, '%s is not a valid mode, please use "LO", "NLO", "aMC@NLO" or "aMC@LO"' % args[0]
        mode = args[0]
        
        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd, 'options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.'
        if options['noreweight'] and options['reweightonly']:
            raise self.InvalidCmd, 'options -R (--noreweight) and -R (--reweightonly)' + \
                    ' are not compatible. Please choose one.'
        if mode == 'NLO' and options['reweightonly']:
            raise self.InvalidCmd, 'option -r (--reweightonly) needs mode "aMC@NLO" or "aMC@LO"'


    def check_compile(self, args, options):
        """check the validity of the line. args is MODE
        MODE being FO or MC. If no mode is passed, MC is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if not args:
            args.append('MC')
            return
        
        if len(args) > 1:
            self.help_compile()
            raise self.InvalidCmd, 'Invalid Syntax: Too many argument'

        elif len(args) == 1:
            if not args[0] in ['MC', 'FO']:
                raise self.InvalidCmd, '%s is not a valid mode, please use "FO" or "MC"' % args[0]
        mode = args[0]
        
        # check for incompatible options/modes


#===============================================================================
# CompleteForCmd
#===============================================================================
class CompleteForCmd(CheckValidForCmd):
    """ The Series of help routine for the MadGraphCmd"""

    def complete_run_mcatnlo(self, text, line, begidx, endidx):
        args = self.split_arg(line[0:begidx])
        bad_files = ['../Events/banner_header.txt']
        if not text.strip():
            text = '../Events'
        if len(args) == 1:
            return [name for name in self.path_completion( text,
                        '.', only_dirs = False) \
                                if not name in bad_files]


class aMCatNLOAlreadyRunning(InvalidCmd):
    pass

#===============================================================================
# aMCatNLOCmd
#===============================================================================
class aMCatNLOCmd(CmdExtended, HelpToCmd, CompleteForCmd):
    """The command line processor of MadGraph"""    
    
    # Truth values
    true = ['T','.true.',True,'true']
    # Options and formats available
    _run_options = ['--cluster','--multicore','--nb_core=','--nb_core=2', '-c', '-m']
    _generate_options = ['-f', '--laststep=parton', '--laststep=pythia', '--laststep=pgs', '--laststep=delphes']
    _calculate_decay_options = ['-f', '--accuracy=0.']
    _set_options = ['stdout_level','fortran_compiler','timeout']
    _plot_mode = ['all', 'parton','pythia','pgs','delphes','channel', 'banner']
    _clean_mode = _plot_mode
    _display_opts = ['run_name', 'options', 'variable']
    # survey options, dict from name to type, default value, and help text
    # Variables to store object information
    web = False
    prompt = 'aMC@NLO>'
    cluster_mode = 0
    queue  = 'madgraph'
    nb_core = None
    
    next_possibility = {
        'start': ['generate_events [OPTIONS]', 'multi_run [OPTIONS]',
                  'calculate_decay_widths [OPTIONS]',
                  'help generate_events'],
        'generate_events': ['generate_events [OPTIONS]', 'multi_run [OPTIONS]', 'pythia', 'pgs','delphes'],
        'calculate_decay_widths': ['calculate_decay_widths [OPTIONS]',
                                   'generate_events [OPTIONS]'],
        'multi_run': ['generate_events [OPTIONS]', 'multi_run [OPTIONS]'],
        'survey': ['refine'],
        'refine': ['combine_events'],
        'combine_events': ['store'],
        'store': ['pythia'],
        'pythia': ['pgs', 'delphes'],
        'pgs': ['generate_events [OPTIONS]', 'multi_run [OPTIONS]'],
        'delphes' : ['generate_events [OPTIONS]', 'multi_run [OPTIONS]']
    }
    
    
    
    ############################################################################
    def __init__(self, me_dir = None, *completekey, **stdin):
        """ add information to the cmd """

        CmdExtended.__init__(self, *completekey, **stdin)
        
        # Define current aMCatNLO directory
        if me_dir is None and aMCatNLO:
            me_dir = root_path
        
        self.me_dir = me_dir

        # usefull shortcut
        self.status = pjoin(self.me_dir, 'status')
        self.error =  pjoin(self.me_dir, 'error')
        self.dirbin = pjoin(self.me_dir, 'bin', 'internal')
        
        self.to_store = []
        self.run_name = None
        self.run_tag = None
        self.banner = None
        # Load the configuration file
        self.set_configuration()

        
    ############################################################################    
    def split_arg(self, line, error=True):
        """split argument and"""
        
        args = CmdExtended.split_arg(line)

        return args
    
  
    ############################################################################
    def set_configuration(self, config_path=None):
        """ assign all configuration variable from file 
            ./Cards/mg5_configuration.txt. assign to default if not define """
            
        self.options = {'pythia8_path': './pythia8',
                              'pythia-pgs_path': '../pythia-pgs',
                              'delphes_path': '../Delphes',
                              'madanalysis_path': '../MadAnalysis',
                              'exrootanalysis_path': '../ExRootAnalysis',
                              'td_path': '../',
                              'web_browser':None,
                              'eps_viewer':None,
                              'text_editor':None,
                              'fortran_compiler':None,
                              'cluster_mode':'pbs',
                              'automatic_html_opening':True,
                              'run_mode':0,
                              'cluster_queue':'madgraph',
                              'nb_core':None,
                              'timeout':20}
        
        if os.environ.has_key('MADGRAPH_BASE'):
            config_file = open(os.path.join(os.environ['MADGRAPH_BASE'],'mg5_configuration.txt'))
        elif not config_path:
            try:
                config_file = open(os.path.join(os.environ['HOME'],'.mg5','mg5_configuration.txt'))
            except:
                if self.me_dir:
                    config_file = open(os.path.relpath(
                          os.path.join(self.me_dir, 'Cards', 'amcatnlo_configuration.txt')))
                    main = self.me_dir
                elif not aMCatNLO:
                    config_file = open(os.path.relpath(
                          os.path.join(MG5DIR, 'input', 'mg5_configuration.txt')))                    
                    main = MG5DIR
        else:
            config_file = open(config_path)
            if self.me_dir:
                main = self.me_dir
            else:
                main = MG5DIR

        # read the file and extract information
        logger.info('load configuration from %s ' % config_file.name)
        for line in config_file:
            if '#' in line:
                line = line.split('#',1)[0]
            line = line.replace('\n','').replace('\r\n','')
            try:
                name, value = line.split('=')
            except ValueError:
                pass
            else:
                name = name.strip()
                value = value.strip()
                self.options[name] = value
                if value.lower() == "none":
                    self.options[name] = None

        # Treat each expected input
        # delphes/pythia/... path
        for key in self.options:
            if key.endswith('path'):
                if self.options[key] in ['None', None]:
                    self.options[key] = ''
                    continue
                path = os.path.join(self.me_dir, self.options[key])
                if os.path.isdir(path):
                    self.options[key] = os.path.realpath(path)
                    continue
                if os.path.isdir(self.options[key]):
                    self.options[key] = os.path.realpath(self.options[key])
                    continue
                elif not os.path.isdir(self.options[key]):
                    self.options[key] = ''
            elif key.startswith('cluster'):
                pass              
            elif key == 'automatic_html_opening':
                if self.options[key] in ['False', 'True']:
                    self.options[key] =eval(self.options[key])
            elif key not in ['text_editor','eps_viewer','web_browser']:
                # Default: try to set parameter
                try:
                    self.do_set("%s %s" % (key, self.options[key]))
                except self.InvalidCmd:
                    logger.warning("Option %s from config file not understood" \
                                   % key)
        
        # Configure the way to open a file:
        misc.open_file.configure(self.options)
          
        return self.options
    
    ############################################################################ 
    def do_open(self, line):
        """Open a text file/ eps file / html file"""
        
        args = self.split_arg(line)
        # Check Argument validity and modify argument to be the real path
        self.check_open(args)
        file_path = args[0]
        
        misc.open_file(file_path)
  

    ############################################################################
    def do_set(self, line):
        """Set an option, which will be default for coming generations/outputs
        """

        args = self.split_arg(line) 
        # Check the validity of the arguments
        self.check_set(args)

        if args[0] == "stdout_level":
            logging.root.setLevel(eval('logging.' + args[1]))
            logging.getLogger('madgraph').setLevel(eval('logging.' + args[1]))
            logger.info('set output information to level: %s' % args[1])
        elif args[0] == "fortran_compiler":
            self.options['fortran_compiler'] = args[1]
        elif args[0] == "run_mode":
            if not args[1] in [0,1,2,'0','1','2']:
                raise self.InvalidCmd, 'run_mode should be 0, 1 or 2.'
            self.cluster_mode = int(args[1])
            self.options['cluster_mode'] =  self.cluster_mode
        elif args[0] == 'cluster_type':
            self.options['cluster_mode'] = args[1]
            self.cluster = cluster.from_name[args[1]](self.options['cluster_queue'])
        elif args[0] == 'nb_core':
            if args[1] == 'None':
                import multiprocessing
                self.nb_core = multiprocessing.cpu_count()
                self.options['nb_core'] = self.nb_core
                return
            if not args[1].isdigit():
                raise self.InvalidCmd('nb_core should be a positive number') 
            self.nb_core = int(args[1])
            self.options['nb_core'] = self.nb_core
        elif args[0] == 'timeout':
            self.options[args[0]] = int(args[1]) 
        elif args[0] in self.options:
            if args[1] in ['None','True','False']:
                self.options[args[0]] = eval(args[1])
            elif args[0].endswith('path'):
                if os.path.exists(args[1]):
                    self.options[args[0]] = args[1]
                elif os.path.exists(pjoin(self.me_dir, args[1])):
                    self.options[args[0]] = pjoin(self.me_dir, args[1])
                else:
                    raise self.InvalidCmd('Not a valid path: keep previous value: \'%s\'' % self.options[args[0]])
            else:
                self.options[args[0]] = args[1]             

    ############################################################################      
    def do_run_mcatnlo(self, line):
        """ calculates LO/NLO cross-section, using madevent_vegas """
        argss = self.split_arg(line)
        self.check_run_mcatnlo(argss, {})
        evt_file = pjoin(os.getcwd(), argss[0])
        # check argument validity and normalise argument
        self.run_mcatnlo(evt_file)

 

    ############################################################################      
    def do_calculate_xsect(self, line):
        """ calculates LO/NLO cross-section, using madevent_vegas """
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _generate_events_parser.parse_args(argss)
        options = options.__dict__
        options['reweightonly'] = False
        options['noreweight'] = False
        self.check_calculate_xsect(argss, options)
        
        if options['multicore']:
            self.cluster_mode = 2
        elif options['cluster']:
            self.cluster_mode = 1
        else:
            self.cluster_mode = int(self.options['run_mode'])
        mode = argss[0]
        self.ask_run_configuration(mode)
        self.compile(mode, options) 
        self.run(mode, options)

        
    ############################################################################      
    def do_generate_events(self, line):
        """ generate events """
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _generate_events_parser.parse_args(argss)
        options = options.__dict__
        options['reweightonly'] = False
        self.check_generate_events(argss, options)
        
        if options['multicore']:
            self.cluster_mode = 2
            print 'multicore'
        elif options['cluster']:
            self.cluster_mode = 1
        else:
            self.cluster_mode = int(self.options['run_mode'])
        mode = 'aMC@' + argss[0]
        self.ask_run_configuration(mode)
        self.compile(mode, options) 
        self.run(mode, options)

        
    ############################################################################      
    def do_launch(self, line):
        """ launch the full chain """
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _launch_parser.parse_args(argss)
        options = options.__dict__
        self.check_launch(argss, options)
        
        if options['multicore']:
            self.cluster_mode = 2
            print 'multicore'
        elif options['cluster']:
            self.cluster_mode = 1
        else:
            self.cluster_mode = int(self.options['run_mode'])
        mode = argss[0]
        self.ask_run_configuration(mode)
        self.compile(mode, options) 
        evt_file = self.run(mode, options)
        self.run_mcatnlo(evt_file)


    ############################################################################      
    def do_compile(self, line):
        """ just compile the executables """
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _compile_parser.parse_args(argss)
        options = options.__dict__
        options['reweightonly'] = False
        options['nocompile'] = False
        self.check_compile(argss, options)
        
        mode = {'FO': 'NLO', 'MC': 'aMC@NLO'}[argss[0]]
        self.ask_run_configuration(mode)
        self.compile(mode, options) 


    def update_random_seed(self):
        """Update random number seed with the value from the run_card. 
        If this is 0, update the number according to a fresh one"""
        run_card = pjoin(self.me_dir, 'Cards', 'run_card.dat')
        if not os.path.exists(run_card):
            raise aMCatNLOError('%s is not a valid run_card' % run_card)
        iseed = int(self.read_run_card(run_card)['iseed'])
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
        """runs aMC@NLO. Returns the name of the event file created"""
        logger.info('Starting run')

        if self.cluster_mode == 1:
            cluster_name = self.options['cluster_type']
            self.cluster = cluster.from_name[cluster_name](self.options['cluster_queue'])
        if self.cluster_mode == 2:
            import multiprocessing
            try:
                self.nb_core = int(self.options['nb_core'])
            except TypeError:
                self.nb_core = multiprocessing.cpu_count()
            logger.info('Using %d cores' % self.nb_core)
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

        if options['reweightonly']:
            nevents = int(self.read_run_card(\
                        pjoin(self.me_dir, 'Cards', 'run_card.dat')['nevents']))
            self.reweight_and_collect_events(options, mode, nevents)
            return

        if mode == 'LO':
            logger.info('Doing fixed order LO')
            logger.info('   Cleaning previous results')
            os.system('rm -rf P*/born_G*')

            logger.info('   Computing cross-section')
            self.run_all(job_dict, [['0', 'born', '0']])
            os.system('./combine_results_FO.sh born_G*')
            return

        if mode == 'NLO':
            logger.info('Doing fixed order NLO')
            logger.info('   Cleaning previous results')
            os.system('rm -rf P*/grid_G* P*/novB_G* P*/viSB_G*')

            logger.info('   Setting up grid')
            self.run_all(job_dict, [['0', 'grid', '0']])
            os.system('./combine_results_FO.sh grid*')

            logger.info('   Computing cross-section')
            self.run_all(job_dict, [['0', 'viSB', '0', 'grid'], ['0', 'novB', '0', 'grid']])
            os.system('./combine_results_FO.sh viSB* novB*')
            return

        elif mode in ['aMC@NLO', 'aMC@LO']:
            shower = self.read_run_card(pjoin(self.me_dir, 'Cards', 'run_card.dat'))['parton_shower']
            nevents = int(self.read_run_card(pjoin(self.me_dir, 'Cards', 'run_card.dat'))['nevents'])
            #shower_list = ['HERWIG6', 'HERWIGPP', 'PYTHIA6Q', 'PYTHIA6PT', 'PYTHIA8']
            shower_list = ['HERWIG6', 'HERWIGPP', 'PYTHIA6Q']
            if not shower in shower_list:
                raise aMCatNLOError('%s is not a valid parton shower. Please use one of the following: %s' \
                    % (shower, ', '.join(shower_list)))

            if mode == 'aMC@NLO':
                logger.info('Doing NLO matched to parton shower')
                logger.info('   Cleaning previous results')
                os.system('rm -rf P*/GF* P*/GV*')

                for i, status in enumerate(mcatnlo_status):
                    logger.info('   %s' % status)
                    self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), 'novB', i) 
                    self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), 'viSB', i) 
                    self.run_all(job_dict, [['2', 'V', '%d' % i], ['2', 'F', '%d' % i]])

                    if i < 2:
                        os.system('./combine_results.sh %d %d GF* GV*' % (i, nevents))

            elif mode == 'aMC@LO':
                logger.info('Doing LO matched to parton shower')
                logger.info('   Cleaning previous results')
                os.system('rm -rf P*/GB*')
                for i, status in enumerate(mcatnlo_status):
                    logger.info('   %s at LO' % status)
                    self.write_madinMMC_file(
                            pjoin(self.me_dir, 'SubProcesses'), 'born', i) 
                    self.run_all(job_dict, [['2', 'B', '%d' % i]])

                    if i < 2:
                        os.system('./combine_results.sh %d %d GB*' % (i, nevents))

        if self.cluster_mode == 1:
            #if cluster run, wait 15 sec so that event files are transferred back
            logger.info('Waiting while files are transferred back from the cluster nodes')
            time.sleep(15)

        return self.reweight_and_collect_events(options, mode, nevents)


    def reweight_and_collect_events(self, options, mode, nevents):
        """this function calls the reweighting routines and creates the event file in the 
        Event dir. Return the name of the event file created
        """
        if not options['noreweight']:
            self.run_reweight(options['reweightonly'])

        logger.info('   Collecting events')
        os.system('make collect_events > %s' % \
                pjoin (self.me_dir, 'log_collect_events.txt'))
        os.system('echo "1" | ./collect_events > %s' % \
                pjoin (self.me_dir, 'log_collect_events.txt'))

        count = 1
        if not os.path.exists(pjoin(self.me_dir, 'SubProcesses', 'allevents_0_001')):
            raise aMCatNLOError('An error occurred during event generation. ' + \
                    'The event file has not been created. Check log_collect_events.txt')
        while os.path.exists(pjoin(self.me_dir, 'Events', 'events_%d' % count)):
            count += 1
        evt_file = pjoin(self.me_dir, 'Events', 'events_%d' % count)
        os.system('mv %s %s' % 
            (pjoin(self.me_dir, 'SubProcesses', 'allevents_0_001'), evt_file))
        logger.info('The %s file has been generated.\nIt contains %d %s events to be showered' \
                % (evt_file, nevents, mode[4:]))
        return evt_file


    def run_mcatnlo(self, evt_file):
        """runs mcatnlo on the generated event file, to produce showered-events"""
        logger.info('   Prepairing MCatNLO run')
        shower = self.evt_file_to_mcatnlo(evt_file)
        os.chdir(pjoin(self.me_dir, 'MCatNLO'))
        oldcwd = os.getcwd()

        mcatnlo_log = pjoin(self.me_dir, 'mcatnlo.log')
        logger.info('   Compiling MCatNLO for %s...' % shower) 
        misc.call(['./MCatNLO_MadFKS.inputs > %s 2>&1' % mcatnlo_log], \
                    cwd = os.getcwd(), shell=True)
        if not os.path.exists('MCATNLO_EXE'):
            raise aMCatNLOError('Compilation failed, check %s for details' % mcatnlo_log)
        logger.info('                     ... done')
        logger.info('   Running MCatNLO (this may take some time)...')
        misc.call(['./MCATNLO_EXE < MCATNLO_input >> %s 2>&1' % mcatnlo_log], \
                    cwd = os.getcwd(), shell=True)
        os.chdir(oldcwd)


    def evt_file_to_mcatnlo(self, evt_file):
        """creates the mcatnlo input script using the values set in the header of the event_file.
        It also checks if the lhapdf library is used"""
        file = open(evt_file)
        nevents = 0
        shower = ''
        pdlabel = ''
        itry = 0
        while True:
            line = file.readline()
            #print line
            if not nevents and 'nevents' in line.split():
                nevents = int(line.split()[0])
            if not shower and 'parton_shower' in line.split():
                shower = line.split()[0]
            if not pdlabel and 'pdlabel' in line.split():
                pdlabel = line.split()[0]
            if nevents and shower and pdlabel:
                break
            itry += 1
            if itry > 300:
                file.close()
                raise aMCatNLOError('Event file does not contain run information')
        file.close()

        # check if need to link lhapdf
        if pdlabel =='\'lhapdf\'':
            self.link_lhapdf(pjoin(self.me_dir, 'lib'))
                
        input = open(pjoin(self.me_dir, 'MCatNLO', 'MCatNLO_MadFKS.inputs'))
        lines = input.read().split('\n')
        input.close()
        for i in range(len(lines)):
            if lines[i].startswith('EVPREFIX'):
                lines[i]='EVPREFIX=%s' % os.path.split(evt_file)[1]
            if lines[i].startswith('NEVENTS'):
                lines[i]='NEVENTS=%d' % nevents
            if lines[i].startswith('MCMODE'):
                lines[i]='MCMODE=%s' % shower
        
        output = open(pjoin(self.me_dir, 'MCatNLO', 'MCatNLO_MadFKS.inputs'), 'w')
        output.write('\n'.join(lines))
        output.close()
        return shower

        


        



    def run_reweight(self, only):
        """runs the reweight_xsec_events eecutables on each sub-event file generated
        to compute on the fly scale and/or PDF uncertainities"""
        logger.info('   Doing reweight')

        nev_unw = pjoin(self.me_dir, 'SubProcesses', 'nevents_unweighted')
        # if only doing reweight, copy back the nevents_unweighted file
        if only:
            if os.path.exists(nev_unw + '.orig'):
                os.system('cp %s %s' % (nev_unw + '.orig', nev_unw))
            else:
                raise aMCatNLOError('Cannot find event file information')

        reweight_log = pjoin(self.me_dir, 'reweight.log')
        #read the nevents_unweighted file to get the list of event files
        file = open(nev_unw)
        lines = file.read().split('\n')
        file.close()
        # make copy of the original nevent_unweighted file
        os.system('cp %s %s' % (nev_unw, nev_unw + '.orig'))
        evt_files = [line.split()[0] for line in lines if line]
        #prepare the job_dict
        job_dict = {}
        for i, evt_file in enumerate(evt_files):
            path, evt = os.path.split(evt_file)
            os.chdir(pjoin(self.me_dir, 'SubProcesses', path))
            if self.cluster_mode == 0 or self.cluster_mode == 2:
                exe = 'reweight_xsec_events.local'
            elif self.cluster_mode == 1:
                exe = 'reweight_xsec_events.cluster'
            os.system('ln -sf ../../%s .' % exe)
            job_dict[path] = [exe]

        self.run_all(job_dict, [[evt, '1']])
        os.chdir(pjoin(self.me_dir, 'SubProcesses'))

        #check that the new event files are complete
        for evt_file in evt_files:
            last_line = subprocess.Popen('tail -n1 %s.rwgt ' % evt_file, \
                shell = True, stdout = subprocess.PIPE).stdout.read().strip()
            if last_line != "</LesHouchesEvents>":
                raise aMCatNLOError('An error occurred during reweight.' + \
                      ' Check %s for details' % reweight_log)

        #update file name in nevents_unweighted
        newfile = open(nev_unw, 'w')
        for line in lines:
            if line:
                newfile.write(line.replace(line.split()[0], line.split()[0] + '.rwgt') + '\n')
        newfile.close()


    def wait_for_complete(self):
        """this function waits for jobs on cluster to complete their run."""

        # if running serially nothing to do
        if self.cluster_mode == 0:
            return

        # if running on cluster just use the control function of the module
        elif self.cluster_mode == 1:
            idle = 1
            run = 1
            logger.info('     Waiting for submitted jobs to complete (will update each 10s)')
            while idle + run > 0:
                time.sleep(10)
                idle, run, finish, fail = self.cluster.control('')
                logger.info('     Job status: %d idle, %d running, %d failed, %d completed' \
                        % (idle, run, fail, finish))
            #reset the cluster after completion
            self.cluster.submitted = 0
            self.cluster.submitted_ids = []
        elif self.cluster_mode == 2:
            while self.ijob != self.njobs:
                time.sleep(10)

    def run_all(self, job_dict, arg_list):
        """runs the jobs in job_dict (organized as folder: [job_list]), with arguments args"""
        self.njobs = sum(len(jobs) for jobs in job_dict.values()) * len(arg_list)
        self.ijob = 0
        for args in arg_list:
            for dir, jobs in job_dict.items():
                os.chdir(pjoin(self.me_dir, 'SubProcesses', dir))
                for job in jobs:
                    self.run_exe(job, args)
                    # print some statistics if running serially

        os.chdir(pjoin(self.me_dir, 'SubProcesses'))
        self.wait_for_complete()


    def run_exe(self, exe, args):
        """this basic function launch locally/on cluster exe with args as argument.
        """

        def launch_in_thread(exe, argument, cwd, stdout, control_thread):
            """ way to launch for multicore.
            """

            start = time.time()
            if (cwd and os.path.exists(pjoin(cwd, exe))) or os.path.exists(exe):
                exe = './' + exe
            misc.call([exe] + argument, cwd=cwd, stdout=stdout,
                        stderr=subprocess.STDOUT)
            self.ijob += 1
            logger.info('     Jobs completed: %d/%d' %(self.ijob, self.njobs))
            
            # release the lock for allowing to launch the next job      
            while not control_thread[1].locked():
                # check that the status is locked to avoid coincidence unlock
                if not control_thread[2]:
                    # Main is not yet locked
                    control_thread[0] -= 1
                    return 
                time.sleep(1)
            control_thread[0] -= 1 # upate the number of running thread
            control_thread[1].release()

        # first test that exe exists:
        if not os.path.exists(exe):
            raise aMCatNLOError('Cannot find executable %s in %s' \
                % (exe, os.getcwd()))
        # check that the executable has exec permissions
        if not os.access(exe, os.X_OK):
            os.system('chmod +x %s' % exe)
        # finally run it
        if self.cluster_mode == 0:
            #this is for the serial run
            misc.call(['./'+exe] + args, cwd= os.getcwd())
            self.ijob += 1
            logger.info('     Jobs completed: %d/%d' %(self.ijob, self.njobs))
        elif self.cluster_mode == 1:
            #this is for the cluster run
            self.cluster.submit(exe, args)
        elif self.cluster_mode == 2:
            #this is for the multicore run
            import thread
            if not hasattr(self, 'control_thread'):
                self.control_thread = [0] # [used_thread]
                self.control_thread.append(thread.allocate_lock()) # The lock
                self.control_thread.append(False) # True if all thread submit 
                                                  #-> waiting mode
            if self.control_thread[2]:
#                self.update_status((remaining + 1, self.control_thread[0], 
#                                self.total_jobs - remaining - self.control_thread[0] - 1, run_type), 
#                                   level=None, force=False)
                self.control_thread[1].acquire()
                self.control_thread[0] += 1 # upate the number of running thread
                thread.start_new_thread(launch_in_thread,(exe, args, os.getcwd(), None, self.control_thread))
            elif self.control_thread[0] <  self.nb_core -1:
                self.control_thread[0] += 1 # upate the number of running thread 
                thread.start_new_thread(launch_in_thread,(exe, args, os.getcwd(), None, self.control_thread))
            elif self.control_thread[0] ==  self.nb_core -1:
                self.control_thread[0] += 1 # upate the number of running thread
                thread.start_new_thread(launch_in_thread,(exe, args, os.getcwd(), None, self.control_thread))
                self.control_thread[2] = True
                self.control_thread[1].acquire() # Lock the next submission
                                                 # Up to a release





    def write_madinMMC_file(self, path, run_mode, mint_mode):
        """writes the madinMMC_?.2 file"""
        #check the validity of the arguments
        run_modes = ['born', 'virt', 'novi', 'all', 'viSB', 'novB']
        if run_mode not in run_modes:
            raise aMCatNLOError('%s is not a valid mode for run. Please use one of the following: %s' \
                    % (run_mode, ', '.join(run_modes)))
        mint_modes = [0, 1, 2]
        if mint_mode not in mint_modes:
            raise aMCatNLOError('%s is not a valid mode for mintMC. Please use one of the following: %s' \
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

        #clean files
        os.system('rm -f %s' % 
                ' '.join([amcatnlo_log, madloop_log, reweight_log, gensym_log, test_log]))

        #define which executable/tests to compile
        if mode in ['NLO', 'LO']:
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
                for p_dir in p_dirs]) and options['nocompile']:
            return

        libdir = pjoin(self.me_dir, 'lib')

        # rm links to lhapdflib/ PDFsets if exist
        if os.path.islink(pjoin(libdir, 'libLHAPDF.a')):
            os.remove(pjoin(libdir, 'libLHAPDF.a'))
        if os.path.islink(pjoin(libdir, 'PDFsets')):
            os.remove(pjoin(libdir, 'PDFsets'))

        # read the run_card to find if lhapdf is used or not
        if self.read_run_card(pjoin(self.me_dir, 'Cards','run_card.dat'))['pdlabel'] == \
                'lhapdf':
            self.link_lhapdf(libdir)
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
            raise aMCatNLOError('Compilation failed, check %s for details' % amcatnlo_log)

        # make and run tests (if asked for), gensym and make madevent in each dir
        for p_dir in p_dirs:
            logger.info(p_dir)
            this_dir = pjoin(self.me_dir, 'SubProcesses', p_dir) 
            os.chdir(this_dir)
            # compile and run tests
            for test in tests:
                logger.info('   Compiling %s...' % test)
                os.system('make %s >> %s 2>&1 ' % (test, test_log))
                if not os.path.exists(pjoin(this_dir, test)):
                    raise aMCatNLOError('Compilation failed, check %s for details' \
                            % test_log)
                logger.info('   Running %s...' % test)
                self.write_test_input(test)
                input = pjoin(self.me_dir, '%s_input.txt' % test)
                #this can be improved/better written to handle the output
                os.system('./%s < %s | tee -a %s | grep "Fraction of failures"' \
                        % (test, input, test_log))
                os.system('rm -f %s' % input)
            #check that none of the tests failed
            file = open(test_log)
            content = file.read()
            file.close()
            if 'FAILED' in content:
                raise aMCatNLOError('Some tests failed, run cannot continue')

            if not options['reweightonly']:
                logger.info('   Compiling gensym...')
                os.system('make gensym >> %s 2>&1 ' % amcatnlo_log)
                if not os.path.exists(pjoin(this_dir, 'gensym')):
                    raise aMCatNLOError('Compilation failed, check %s for details' % amcatnlo_log)

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
                        raise aMCatNLOError('Compilation failed, check %s for details' % madloop_log)
                os.chdir(this_dir)
                logger.info('   Compiling %s' % exe)
                os.system('make %s >> %s 2>&1' % (exe, amcatnlo_log))
                os.unsetenv('madloop')
                if not os.path.exists(pjoin(this_dir, exe)):
                    raise aMCatNLOError('Compilation failed, check %s for details' % amcatnlo_log)
            if mode in ['aMC@NLO', 'aMC@LO'] and not options['noreweight']:
                logger.info('   Compiling reweight_xsec_events')
                os.system('make reweight_xsec_events >> %s 2>&1' % (reweight_log))
                if not os.path.exists(pjoin(this_dir, 'reweight_xsec_events')):
                    raise aMCatNLOError('Compilation failed, check %s for details' % reweight_log)

        os.chdir(pjoin(self.me_dir))


    def link_lhapdf(self, libdir):
        """links lhapdf into libdir"""
        logger.info('Using LHAPDF interface for PDFs')
        lhalibdir = subprocess.Popen('%s --libdir' % self.options['lhapdf'],
                shell = True, stdout = subprocess.PIPE).stdout.read().strip()
        lhasetsdir = subprocess.Popen('%s --pdfsets-path' % self.options['lhapdf'], 
                shell = True, stdout = subprocess.PIPE).stdout.read().strip()
        if not os.path.exists(pjoin(libdir, 'libLHAPDF.a')):
            os.symlink(pjoin(lhalibdir, 'libLHAPDF.a'), pjoin(libdir, 'libLHAPDF.a'))
        if not os.path.exists(pjoin(libdir, 'PDFsets')):
            os.symlink(lhasetsdir, pjoin(libdir, 'PDFsets'))
        os.putenv('lhapdf', 'True')


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
            shower = self.read_run_card(
                    pjoin(self.me_dir, 'Cards', 'run_card.dat'))['parton_shower']
            MC_header = "%s\n " % shower + \
                        "1 \n1 -0.1\n-1 -0.1\n"
            file.write(MC_header + content)
        file.close()


    ############################################################################
    ##  HELPING ROUTINE
    ############################################################################
    def read_run_card(self, run_card):
        """ """
        output={}
        for line in file(run_card,'r'):
            line = line.split('#')[0]
            line = line.split('!')[0]
            line = line.split('=')
            if len(line) != 2:
                continue
            output[line[1].strip()] = line[0].replace('\'','').strip()
        return output


    ############################################################################
    def ask_run_configuration(self, mode):
        """Ask the question when launching generate_events/multi_run"""
        
        logger.info('Will run in mode %s' % mode)
        cards = ['param_card.dat', 'run_card.dat', 'mcatnlo_card.dat']

        def get_question(mode):
            # Ask the user if he wants to edit any of the files
            #First create the asking text
            question = """Do you want to edit one cards (press enter to bypass editing)?
  1 / param   : param_card.dat (be carefull about parameter consistency, especially widths)
  2 / run     : run_card.dat\n
  3 / mcatnlo : mcatnlo_card.dat\n"""
            possible_answer = ['0','done', 1, 'param', 2, 'run', 3, 'mcatnlo']
            card = {0:'done', 1:'param', 2:'run', 3:'mcatnlo'}
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




#===============================================================================
# aMCatNLOCmd
#===============================================================================
class aMCatNLOCmdShell(aMCatNLOCmd, cmd.CmdShell):
    """The command line processor of MadGraph"""  

_compile_usage = "compile [MODE] [options]\n" + \
                "-- compiles aMC@NLO \n" + \
                "   MODE can be either FO, for fixed-order computations, \n" + \
                "   or MC for matching with parton-shower monte-carlos. \n" + \
                "   (if omitted, it is set to MC)\n"
_compile_parser = optparse.OptionParser(usage=_compile_usage)

_compile_parser.add_option("-R", "--noreweight", default=False, action='store_true',
                            help="Skip compiling reweight executable")


_launch_usage = "launch [MODE] [options]\n" + \
                "-- execute aMC@NLO \n" + \
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


_calculate_xsect_usage = "calculate_xsect [ORDER] [options]\n" + \
                "-- calculate cross-section up to ORDER.\n" + \
                "   ORDER can be either LO or NLO (if omitted, it is set to NLO). \n"

_calculate_xsect_parser = optparse.OptionParser(usage=_calculate_xsect_usage)
_calculate_xsect_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_calculate_xsect_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_calculate_xsect_parser.add_option("-n", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found, " + \
                            "or with --tests")


_generate_events_usage = "generate_events [ORDER] [options]\n" + \
                "-- generate events to be showered, corresponding to a cross-section computed up to ORDER.\n" + \
                "   ORDER can be either LO or NLO (if omitted, it is set to NLO). \n" + \
                "   The number of events and the specific parton shower MC can be specified \n" + \
                "   in the run_card.dat\n"

_generate_events_parser = optparse.OptionParser(usage=_generate_events_usage)
_generate_events_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_generate_events_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_generate_events_parser.add_option("-n", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found, " + \
                            "or with --tests")
_generate_events_parser.add_option("-R", "--noreweight", default=False, action='store_true',
                            help="Skip file reweighting")
