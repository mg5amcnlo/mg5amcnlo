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
import signal

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
logger = logging.getLogger('madgraph.stdout') # -> stdout
logger_stderr = logging.getLogger('madgraph.stderr') # ->stderr
 
try:
    # import from madgraph directory
    import madgraph.interface.extended_cmd as cmd
    import madgraph.interface.common_run_interface as common_run
    import madgraph.iolibs.files as files
    import madgraph.iolibs.save_load_object as save_load_object
    import madgraph.various.banner as banner_mod
    import madgraph.various.cluster as cluster
    import madgraph.various.misc as misc
    import madgraph.various.gen_crossxhtml as gen_crossxhtml
    import madgraph.various.sum_html as sum_html
    import madgraph.various.shower_card as shower_card

    from madgraph import InvalidCmd, aMCatNLOError
    aMCatNLO = False
except ImportError, error:
    logger.debug(error)
    # import from madevent directory
    import internal.extended_cmd as cmd
    import internal.common_run_interface as common_run
    import internal.banner as banner_mod
    import internal.misc as misc    
    from internal import InvalidCmd, MadGraph5Error
    import internal.files as files
    import internal.cluster as cluster
    import internal.save_load_object as save_load_object
    import internal.gen_crossxhtml as gen_crossxhtml
    import internal.sum_html as sum_html
    import internal.shower_card as shower_card
    aMCatNLO = True

class aMCatNLOError(Exception):
    pass

def init_worker():
    """this is to catch ctrl+c with Pool""" 
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def compile_dir(arguments):
    """compile the direcory p_dir
    arguments is the tuple (me_dir, p_dir, mode, options, tests, exe, run_mode)
    this function needs not to be a class method in order to use pool to do
    the compilation on multicore"""

    (me_dir, p_dir, mode, options, tests, exe, run_mode) = arguments
    logger.info(' Compiling %s...' % p_dir)

    this_dir = pjoin(me_dir, 'SubProcesses', p_dir) 
    #compile everything

    # compile and run tests
    for test in tests:
        misc.compile([test], cwd = this_dir, job_specs = False)
        if not os.path.exists(pjoin(this_dir, test)):
            raise aMCatNLOError('%s compilation failed' % test)
        input = pjoin(me_dir, '%s_input.txt' % test)
        #this can be improved/better written to handle the output
        misc.call(['./%s' % (test)], cwd=this_dir, 
                stdin = open(input), stdout=open(pjoin(this_dir, '%s.log' % test), 'w'))
        
    if not options['reweightonly']:
        misc.compile(['gensym'], cwd=this_dir, job_specs = False)
        if not os.path.exists(pjoin(this_dir, 'gensym')):
            raise aMCatNLOError('gensym compilation failed')

        open(pjoin(this_dir, 'gensym_input.txt'), 'w').write('%s\n' % run_mode)
        misc.call(['./gensym'],cwd= this_dir,
                 stdin=open(pjoin(this_dir, 'gensym_input.txt')),
                 stdout=open(pjoin(this_dir, 'gensym.log'), 'w')) 
        #compile madevent_mintMC/vegas
        misc.compile([exe], cwd=this_dir, job_specs = False)
        if not os.path.exists(pjoin(this_dir, exe)):
            raise aMCatNLOError('%s compilation failed' % exe)
    if mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO']:
        misc.compile(['reweight_xsec_events'], cwd=this_dir, job_specs = False)
        if not os.path.exists(pjoin(this_dir, 'reweight_xsec_events')):
            raise aMCatNLOError('reweight_xsec_events compilation failed')
    logger.info('    %s done.' % p_dir) 


def check_compiler(options, block=False):
    """check that the current fortran compiler is gfortran 4.6 or later.
    If block, stops the execution, otherwise just print a warning"""

    msg = 'In order to be able to run MadGraph @NLO, you need to have ' + \
            'gfortran 4.6 or later installed.\n%s has been detected'
    #first check that gfortran is installed
    if options['fortran_compiler']:
        compiler = options['fortran_compiler']
    elif misc.which('gfortran'):
        compiler = 'gfortran'
        
    if 'gfortran' not in compiler:
        if block:
            raise aMCatNLOError(msg % compiler)
        else:
            logger.warning(msg % compiler)
    else:
        curr_version = misc.get_gfortran_version(compiler)
        if not ''.join(curr_version.split('.')) >= '46':
            if block:
                raise aMCatNLOError(msg % (compiler + ' ' + curr_version))
            else:
                logger.warning(msg % (compiler + ' ' + curr_version))
            


#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(common_run.CommonRunCmd):
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

    def __init__(self, me_dir, options, *arg, **opt):
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
        "*                 http://amcatnlo.cern.ch                  *\n" + \
        "*                                                          *\n" + \
        "*               Type 'help' for in-line help.              *\n" + \
        "*                                                          *\n" + \
        "************************************************************")
        super(CmdExtended, self).__init__(me_dir, options, *arg, **opt)
        

    def get_history_header(self):
        """return the history header""" 
        return self.history_header % misc.get_time_info()
    
    def stop_on_keyboard_stop(self):
        """action to perform to close nicely on a keyboard interupt"""
        try:
            if hasattr(self, 'results'):
                self.update_status('Stop by the user', level=None, makehtml=True, error=True)
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
        except Exception:
            misc.sprint('self.update_status fails', log=logger)
            pass
            
    def nice_user_error(self, error, line):
        """If a ME run is currently running add a link in the html output"""

        self.add_error_log_in_html()
        cmd.Cmd.nice_user_error(self, error, line)            
        
    def nice_config_error(self, error, line):
        """If a ME run is currently running add a link in the html output"""

        self.add_error_log_in_html()
        cmd.Cmd.nice_config_error(self, error, line)

    def nice_error_handling(self, error, line):
        """If a ME run is currently running add a link in the html output"""

        self.add_error_log_in_html()            
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

    def help_shower(self):
        """help for shower command"""
        _shower_parser.print_help()

    
    def help_open(self):
        logger.info("syntax: open FILE  ")
        logger.info("-- open a file with the appropriate editor.")
        logger.info('   If FILE belongs to index.html, param_card.dat, run_card.dat')
        logger.info('   the path to the last created/used directory is used')

    def run_options_help(self, data):
        if data:
            logger.info('-- local options:')
            for name, info in data:
                logger.info('      %s : %s' % (name, info))
        
        logger.info("-- session options:")
        logger.info("      Note that those options will be kept for the current session")      
        logger.info("      --cluster : Submit to the  cluster. Current cluster: %s" % self.options['cluster_type'])
        logger.info("      --multicore : Run in multi-core configuration")
        logger.info("      --nb_core=X : limit the number of core to use to X.")
        


       
#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(object):
    """ The Series of check routine for the aMCatNLOCmd"""

    def check_shower(self, args, options):
        """Check the validity of the line. args[0] is the run_directory"""
        
        if options['force']:
            self.force = True
        
        if len(args) == 0:
            self.help_shower()
            raise self.InvalidCmd, 'Invalid syntax, please specify the run name'
        if not os.path.isdir(pjoin(self.me_dir, 'Events', args[0])):
            raise self.InvalidCmd, 'Directory %s does not exists' % \
                            pjoin(os.getcwd(), 'Events',  args[0])
        args[0] = pjoin(self.me_dir, 'Events', args[0])
    
    def check_plot(self, args):
        """Check the argument for the plot command
        plot run_name modes"""

        if options['force']:
            self.force = True

        madir = self.options['madanalysis_path']
        td = self.options['td_path']
        
        if not madir or not td:
            logger.info('Retry to read configuration file to find madanalysis/td')
            self.set_configuration()

        madir = self.options['madanalysis_path']
        td = self.options['td_path']        
        
        if not madir:
            error_msg = 'No Madanalysis path correctly set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)  
        if not  td:
            error_msg = 'No path to td directory correctly set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)  
                     
        if len(args) == 0:
            if not hasattr(self, 'run_name') or not self.run_name:
                self.help_plot()
                raise self.InvalidCmd('No run name currently define. Please add this information.')             
            args.append('all')
            return

        
        if args[0] not in self._plot_mode:
            self.set_run_name(args[0], level='plot')
            del args[0]
            if len(args) == 0:
                args.append('all')
        elif not self.run_name:
            self.help_plot()
            raise self.InvalidCmd('No run name currently define. Please add this information.')                             
        
        for arg in args:
            if arg not in self._plot_mode and arg != self.run_name:
                 self.help_plot()
                 raise self.InvalidCmd('unknown options %s' % arg)        
    
    def check_pgs(self, arg):
        """Check the argument for pythia command
        syntax: pgs [NAME] 
        Note that other option are already remove at this point
        """
        
        # If not pythia-pgs path
        if not self.options['pythia-pgs_path']:
            logger.info('Retry to read configuration file to find pythia-pgs path')
            self.set_configuration()
      
        if not self.options['pythia-pgs_path'] or not \
            os.path.exists(pjoin(self.options['pythia-pgs_path'],'src')):
            error_msg = 'No pythia-pgs path correctly set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)          
        
        tag = [a for a in arg if a.startswith('--tag=')]
        if tag: 
            arg.remove(tag[0])
            tag = tag[0][6:]
        
        
        if len(arg) == 0 and not self.run_name:
            if self.results.lastrun:
                arg.insert(0, self.results.lastrun)
            else:
                raise self.InvalidCmd('No run name currently define. Please add this information.')             
        
        if len(arg) == 1 and self.run_name == arg[0]:
            arg.pop(0)
        
        if not len(arg) and \
           not os.path.exists(pjoin(self.me_dir,'Events','pythia_events.hep')):
            self.help_pgs()
            raise self.InvalidCmd('''No file file pythia_events.hep currently available
            Please specify a valid run_name''')
                
        lock = None              
        if len(arg) == 1:
            prev_tag = self.set_run_name(arg[0], tag, 'pgs')
            filenames = glob.glob(pjoin(self.me_dir, 'Events', self.run_name,
                                            'events_*.hep.gz'))
            if not filenames:
                raise self.InvalidCmd('No events file corresponding to %s run with tag %s. '% (self.run_name, prev_tag))
            else:
                input_file = filenames[0]
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                lock = cluster.asyncrone_launch('gunzip',stdout=open(output_file,'w'), 
                                                    argument=['-c', input_file])
        else:
            if tag: 
                self.run_card['run_tag'] = tag
            self.set_run_name(self.run_name, tag, 'pgs')
        
        return lock
            

    def check_delphes(self, arg):
        """Check the argument for pythia command
        syntax: delphes [NAME] 
        Note that other option are already remove at this point
        """
        
        # If not pythia-pgs path
        if not self.options['delphes_path']:
            logger.info('Retry to read configuration file to find delphes path')
            self.set_configuration()
      
        if not self.options['delphes_path']:
            error_msg = 'No delphes path correctly set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)  

        tag = [a for a in arg if a.startswith('--tag=')]
        if tag: 
            arg.remove(tag[0])
            tag = tag[0][6:]
            
                  
        if len(arg) == 0 and not self.run_name:
            if self.results.lastrun:
                arg.insert(0, self.results.lastrun)
            else:
                raise self.InvalidCmd('No run name currently define. Please add this information.')             
        
        if len(arg) == 1 and self.run_name == arg[0]:
            arg.pop(0)
        
        if not len(arg) and \
           not os.path.exists(pjoin(self.me_dir,'Events','pythia_events.hep')):
            self.help_pgs()
            raise self.InvalidCmd('''No file file pythia_events.hep currently available
            Please specify a valid run_name''')
                              
        if len(arg) == 1:
            prev_tag = self.set_run_name(arg[0], tag, 'delphes')
            filenames = glob.glob(pjoin(self.me_dir, 'Events', self.run_name,
                                            'events_*.hep.gz'))
            if not filenames:
                raise self.InvalidCmd('No events file corresponding to %s run with tag %s.:%s '\
                    % (self.run_name, prev_tag, 
                       pjoin(self.me_dir,'Events',self.run_name, '%s_pythia_events.hep.gz' % prev_tag)))
            else:
                input_file = filenames[0]
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                lock = cluster.asyncrone_launch('gunzip',stdout=open(output_file,'w'), 
                                                    argument=['-c', input_file])
        else:
            if tag:
                self.run_card['run_tag'] = tag
            self.set_run_name(self.run_name, tag, 'delphes')               

    def check_calculate_xsect(self, args, options):
        """check the validity of the line. args is ORDER,
        ORDER being LO or NLO. If no mode is passed, NLO is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if options['force']:
            self.force = True
        
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


    def check_launch(self, args, options):
        """check the validity of the line. args is MODE
        MODE being LO, NLO, aMC@NLO or aMC@LO. If no mode is passed, auto is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if options['force']:
            self.force = True
        
        
        if not args:
            args.append('auto')
            return
        
        if len(args) > 1:
            self.help_launch()
            raise self.InvalidCmd, 'Invalid Syntax: Too many argument'

        elif len(args) == 1:
            if not args[0] in ['LO', 'NLO', 'aMC@NLO', 'aMC@LO','auto']:
                raise self.InvalidCmd, '%s is not a valid mode, please use "LO", "NLO", "aMC@NLO" or "aMC@LO"' % args[0]
        mode = args[0]
        
        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd, 'options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.'
        if mode == 'NLO' and options['reweightonly']:
            raise self.InvalidCmd, 'option -r (--reweightonly) needs mode "aMC@NLO" or "aMC@LO"'


    def check_compile(self, args, options):
        """check the validity of the line. args is MODE
        MODE being FO or MC. If no mode is passed, MC is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if options['force']:
            self.force = True
        
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

    def complete_launch(self, text, line, begidx, endidx):
        """auto-completion for launch command"""
        
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return mode
            return self.list_completion(text,['LO','NLO','aMC@NLO','aMC@LO'],line)
        elif len(args) == 2 and line[begidx-1] == '@':
            return self.list_completion(text,['LO','NLO'],line)
        else:
            opts = []
            for opt in _launch_parser.option_list:
                opts += opt._long_opts + opt._short_opts
            return self.list_completion(text, opts, line)
            
    def complete_compile(self, text, line, begidx, endidx):
        """auto-completion for launch command"""
        
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return mode
            return self.list_completion(text,['FO','MC'],line)
        else:
            opts = []
            for opt in _compile_parser.option_list:
                opts += opt._long_opts + opt._short_opts
            return self.list_completion(text, opts, line)        

    def complete_calculate_xsect(self, text, line, begidx, endidx):
        """auto-completion for launch command"""
        
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return mode
            return self.list_completion(text,['LO','NLO'],line)
        else:
            opts = []
            for opt in _calculate_xsect_parser.option_list:
                opts += opt._long_opts + opt._short_opts
            return self.list_completion(text, opts, line) 

    def complete_generate_events(self, text, line, begidx, endidx):
        """auto-completion for launch command"""
        
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return mode
            return self.list_completion(text,['LO','NLO'],line)
        else:
            opts = []
            for opt in _generate_events_parser.option_list:
                opts += opt._long_opts + opt._short_opts
            return self.list_completion(text, opts, line) 


    def complete_shower(self, text, line, begidx, endidx):
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*','events.lhe.gz'))
            data = [n.rsplit('/',2)[1] for n in data]
            tmp1 =  self.list_completion(text, data)
            if not self.run_name:
                return tmp1

    def complete_plot(self, text, line, begidx, endidx):
        """ Complete the plot command """
        
        args = self.split_arg(line[0:begidx], error=False)

        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*','events.lhe*'))
            data = [n.rsplit('/',2)[1] for n in data]
            tmp1 =  self.list_completion(text, data)
            if not self.run_name:
                return tmp1

        if len(args) > 1:
            return self.list_completion(text, self._plot_mode)        

    def complete_pgs(self,text, line, begidx, endidx):
        "Complete the pgs command"
        args = self.split_arg(line[0:begidx], error=False) 
        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*', 'events_*.hep.gz'))
            data = [n.rsplit('/',2)[1] for n in data]
            tmp1 =  self.list_completion(text, data)
            if not self.run_name:
                return tmp1
            else:
                tmp2 = self.list_completion(text, self._run_options + ['-f', 
                                                '--tag=' ,'--no_default'], line)
                return tmp1 + tmp2        
        else:
            return self.list_completion(text, self._run_options + ['-f', 
                                                 '--tag=','--no_default'], line)

    complete_delphes = complete_pgs        

class aMCatNLOAlreadyRunning(InvalidCmd):
    pass

#===============================================================================
# aMCatNLOCmd
#===============================================================================
class aMCatNLOCmd(CmdExtended, HelpToCmd, CompleteForCmd, common_run.CommonRunCmd):
    """The command line processor of MadGraph"""    
    
    # Truth values
    true = ['T','.true.',True,'true']
    # Options and formats available
    _run_options = ['--cluster','--multicore','--nb_core=','--nb_core=2', '-c', '-m']
    _generate_options = ['-f', '--laststep=parton', '--laststep=pythia', '--laststep=pgs', '--laststep=delphes']
    _calculate_decay_options = ['-f', '--accuracy=0.']
    _set_options = ['stdout_level','fortran_compiler','timeout']
    _plot_mode = ['all', 'parton','shower','pgs','delphes']
    _clean_mode = _plot_mode + ['channel', 'banner']
    _display_opts = ['run_name', 'options', 'variable']
    # survey options, dict from name to type, default value, and help text
    # Variables to store object information
    web = False
    prompt = 'aMC@NLO_run>'
    cluster_mode = 0
    queue  = 'madgraph'
    nb_core = None
    
    next_possibility = {
        'start': ['generate_events [OPTIONS]', 'calculate_crossx [OPTIONS]', 'launch [OPTIONS]',
                  'help generate_events'],
        'generate_events': ['generate_events [OPTIONS]', 'shower'],
        'launch': ['launch [OPTIONS]', 'shower'],
        'shower' : ['generate_events [OPTIONS]']
    }
    
    
    ############################################################################
    def __init__(self, me_dir = None, options = {}, *completekey, **stdin):
        """ add information to the cmd """

        self.start_time = 0
        CmdExtended.__init__(self, me_dir, options, *completekey, **stdin)
        #common_run.CommonRunCmd.__init__(self, me_dir, options)

        run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
        self.run_card = banner_mod.RunCardNLO(run_card)
        self.mode = 'aMCatNLO'
        self.nb_core = 0

        # load the current status of the directory
        if os.path.exists(pjoin(self.me_dir,'HTML','results.pkl')):
            self.results = save_load_object.load_from_file(pjoin(self.me_dir,'HTML','results.pkl'))
            self.results.resetall(self.me_dir)
        else:
            model = self.find_model_name()
            process = self.process # define in find_model_name
            self.results = gen_crossxhtml.AllResultsNLO(model, process, self.me_dir)
        self.results.def_web_mode(self.web)
        # check that compiler is gfortran 4.6 or later if virtuals have been exported
        proc_card = open(pjoin(self.me_dir, 'Cards', 'proc_card_mg5.dat')).read()

        if not '[real=QCD]' in proc_card:
            check_compiler(self.options, block=True)

        
    ############################################################################      
    def do_shower(self, line):
        """ run the shower on a given parton level file """
        argss = self.split_arg(line)
        (options, argss) = _generate_events_parser.parse_args(argss)
        # check argument validity and normalise argument
        options = options.__dict__
        options['reweightonly'] = False
        self.check_shower(argss, options)
        evt_file = pjoin(os.getcwd(), argss[0], 'events.lhe')
        self.ask_run_configuration('onlyshower', options)
        self.run_mcatnlo(evt_file)

        self.update_status('', level='all', update_results=True)

    ################################################################################
    def do_plot(self, line):
        """Create the plot for a given run"""

        # Since in principle, all plot are already done automaticaly
        args = self.split_arg(line)
        # Check argument's validity
        self.check_plot(args)
        logger.info('plot for run %s' % self.run_name)
        
        self.ask_edit_cards([], args, plot=True)
                
        if any([arg in ['parton'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe')
            if os.path.exists(filename+'.gz'):
                os.system('gunzip -f %s' % (filename+'.gz') )
            if  os.path.exists(filename):
                logger.info('Found events.lhe file for run %s' % self.run_name) 
                shutil.move(filename, pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'))
                self.create_plot('parton')
                shutil.move(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'), filename)
                os.system('gzip -f %s' % filename)
                
        if any([arg in ['all','parton'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 'MADatNLO.top')
            if  os.path.exists(filename):
                logger.info('Found MADatNLO.top file for run %s' % \
                             self.run_name) 
                output = pjoin(self.me_dir, 'HTML',self.run_name, 'plots_parton.html')
                plot_dir = pjoin(self.me_dir, 'HTML', self.run_name, 'plots_parton')
                
                if not os.path.isdir(plot_dir):
                    os.makedirs(plot_dir) 
                top_file = pjoin(plot_dir, 'plots.top')
                files.cp(filename, top_file)
                madir = self.options['madanalysis_path']
                tag = self.run_card['run_tag']  
                td = self.options['td_path']
                misc.call(['%s/plot' % self.dirbin, madir, td],
                                stdout = open(pjoin(plot_dir, 'plot.log'),'a'),
                                stderr = subprocess.STDOUT,
                                cwd=plot_dir)

                misc.call(['%s/plot_page-pl' % self.dirbin, 
                                    os.path.basename(plot_dir),
                                    'parton'],
                                stdout = open(pjoin(plot_dir, 'plot.log'),'a'),
                                stderr = subprocess.STDOUT,
                                cwd=pjoin(self.me_dir, 'HTML', self.run_name))
                shutil.move(pjoin(self.me_dir, 'HTML',self.run_name ,'plots.html'),
                                                                             output)

                os.remove(pjoin(self.me_dir, 'Events', 'plots.top'))
                
        if any([arg in ['all','shower'] for arg in args]):
            filenames = glob.glob(pjoin(self.me_dir, 'Events', self.run_name,
                                        'events_*.lhe.gz'))
            if len(filenames) != 1:
                filenames = glob.glob(pjoin(self.me_dir, 'Events', self.run_name,
                                            'events_*.hep.gz'))
                if len(filenames) != 1:
                    logger.info('No shower level file found for run %s' % \
                                self.run_name)
                    return
                filename = filenames[0]
                os.system('gunzip -c -f %s > %s' % (filename,
                          pjoin(self.me_dir, 'Events','pythia_events.hep')))

                if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pythia_card.dat')):
                    files.cp(pjoin(self.me_dir, 'Cards', 'pythia_card_default.dat'),
                             pjoin(self.me_dir, 'Cards', 'pythia_card.dat'))
                self.run_hep2lhe()
            else:
                filename = filenames[0]
                os.system('gunzip -c -f %s > %s' % (filename,
                          pjoin(self.me_dir, 'Events','pythia_events.lhe')))
            self.create_plot('Pythia')
            lhe_file_name = filename.replace('.hep.gz', '.lhe')
            shutil.move(pjoin(self.me_dir, 'Events','pythia_events.lhe'), 
                        lhe_file_name)
            os.system('gzip -f %s' % lhe_file_name)
                    
        if any([arg in ['all','pgs'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 
                                            '%s_pgs_events.lhco' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                os.system('gunzip -f %s' % (filename+'.gz') )
            if  os.path.exists(filename):
                self.create_plot('PGS')
                os.system('gzip -f %s' % filename)                
            else:
                logger.info('No valid files for pgs plot')
                
        if any([arg in ['all','delphes'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 
                                        '%s_delphes_events.lhco' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                os.system('gunzip -f %s' % (filename+'.gz') )
            if  os.path.exists(filename):
                #shutil.move(filename, pjoin(self.me_dir, 'Events','delphes_events.lhco'))
                self.create_plot('Delphes')
                #shutil.move(pjoin(self.me_dir, 'Events','delphes_events.lhco'), filename)
                os.system('gzip -f %s' % filename)                
            else:
                logger.info('No valid files for delphes plot')


    ############################################################################      
    def do_calculate_xsect(self, line):
        """Main commands: calculates LO/NLO cross-section, using madevent_vegas """
        
        self.start_time = time.time()
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _generate_events_parser.parse_args(argss)
        options = options.__dict__
        options['reweightonly'] = False
        options['parton'] = True
        self.check_calculate_xsect(argss, options)
        
        if options['multicore']:
            self.cluster_mode = 2
        elif options['cluster']:
            self.cluster_mode = 1
        
        mode = argss[0]
        self.ask_run_configuration(mode, options)

        self.update_status('Starting run', level=None, update_results=True)

        if self.options_madevent['automatic_html_opening']:
            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
            self.options_madevent['automatic_html_opening'] = False

        self.compile(mode, options) 
        self.run(mode, options)
        self.update_status('', level='all', update_results=True)

        
    ############################################################################      
    def do_generate_events(self, line):
        """Main commands: generate events """
        
        self.start_time = time.time()
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _generate_events_parser.parse_args(argss)
        options = options.__dict__
        options['reweightonly'] = False
        self.check_generate_events(argss, options)
        
        if options['multicore']:
            self.cluster_mode = 2
        elif options['cluster']:
            self.cluster_mode = 1

        mode = 'aMC@' + argss[0]
        if options['parton'] and mode == 'aMC@NLO':
            mode = 'noshower'
        elif options['parton'] and mode == 'aMC@LO':
            mode = 'noshowerLO'
        self.ask_run_configuration(mode, options)

        self.update_status('Starting run', level=None, update_results=True)

        if self.options_madevent['automatic_html_opening']:
            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
            self.options_madevent['automatic_html_opening'] = False

        self.compile(mode, options) 
        evt_file = self.run(mode, options)
        if not options['parton']:
            self.run_mcatnlo(evt_file)

        self.update_status('', level='all', update_results=True)

    ############################################################################
    def do_treatcards(self, line, amcatnlo=True):
        """Advanced commands: this is for creating the correct run_card.inc from the nlo format"""
        return super(aMCatNLOCmd,self).do_treatcards(line, amcatnlo)
    
    ############################################################################
    def set_configuration(self, amcatnlo=True, **opt):
        """this is for creating the correct run_card.inc from the nlo format"""
        return super(aMCatNLOCmd,self).set_configuration(amcatnlo=amcatnlo, **opt)
    
    ############################################################################      
    def do_launch(self, line):
        """Main commands: launch the full chain """
        
        self.start_time = time.time()
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _launch_parser.parse_args(argss)
        options = options.__dict__
        self.check_launch(argss, options)

        if options['multicore']:
            self.cluster_mode = 2
        elif options['cluster']:
            self.cluster_mode = 1

        mode = argss[0]
        if mode in ['LO', 'NLO']:
            options['parton'] = True
        mode = self.ask_run_configuration(mode, options)

        self.update_status('Starting run', level=None, update_results=True)

        if self.options['automatic_html_opening']:
            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
            self.options['automatic_html_opening'] = False

        if '+' in mode:
            mode = mode.split('+')[0]
        self.compile(mode, options) 
        evt_file = self.run(mode, options)
        
        if int(self.run_card['nevents']) == 0:
            logger.info('No event file generated: grids have been set-up with a '\
                            'relative precision of %s' % self.run_card['req_acc'])
            return

        if not mode in ['LO', 'NLO']:
            assert evt_file == pjoin(self.me_dir,'Events', self.run_name, 'events.lhe'), '%s != %s' %(evt_file, pjoin(self.me_dir,'Events', self.run_name, 'events.lhe.gz'))
            self.exec_cmd('decay_events -from_cards', postcmd=False)
            evt_file = pjoin(self.me_dir,'Events', self.run_name, 'events.lhe')
        
        if not mode in ['LO', 'NLO', 'noshower', 'noshowerLO'] \
                                                      and not options['parton']:
            self.run_mcatnlo(evt_file)
        elif mode == 'noshower':
            logger.warning("""You have chosen not to run a parton shower. NLO events without showering are NOT physical.
Please, shower the Les Houches events before using them for physics analyses.""")

        self.update_status('', level='all', update_results=True)


    ############################################################################      
    def do_compile(self, line):
        """Advanced commands: just compile the executables """
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _compile_parser.parse_args(argss)
        options = options.__dict__
        options['reweightonly'] = False
        options['nocompile'] = False
        self.check_compile(argss, options)
        
        mode = {'FO': 'NLO', 'MC': 'aMC@NLO'}[argss[0]]
        self.ask_run_configuration(mode, options)
        self.compile(mode, options) 


        self.update_status('', level='all', update_results=True)

    def update_status(self, status, level, makehtml=True, force=True, 
                      error=False, starttime = None, update_results=False):
        
        common_run.CommonRunCmd.update_status(self, status, level, makehtml, 
                                        force, error, starttime, update_results)


    def update_random_seed(self):
        """Update random number seed with the value from the run_card. 
        If this is 0, update the number according to a fresh one"""
        iseed = int(self.run_card['iseed'])
        if iseed != 0:
            misc.call(['echo "r=%d" > %s' \
                    % (iseed, pjoin(self.me_dir, 'SubProcesses', 'randinit'))],
                    cwd=self.me_dir, shell=True)
        else:
            randinit = open(pjoin(self.me_dir, 'SubProcesses', 'randinit'))
            iseed = int(randinit.read()[2:]) + 1
            randinit.close()
            randinit = open(pjoin(self.me_dir, 'SubProcesses', 'randinit'), 'w')
            randinit.write('r=%d' % iseed)
            randinit.close()


    def get_characteristics(self, file):
        """reads the proc_characteristics file and initialises the correspondent
        dictionary"""
        lines = [l for l in open(file).read().split('\n') if l and not l.startswith('#')]
        self.proc_characteristics = {}
        for l in lines:
            key, value = l.split('=')
            self.proc_characteristics[key.strip()] = value.strip()
            
        
    def run(self, mode, options):
        """runs aMC@NLO. Returns the name of the event file created"""
        logger.info('Starting run')

        if not 'only_generation' in options.keys():
            options['only_generation'] = False

        os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))

        self.get_characteristics(pjoin(self.me_dir, 'SubProcesses', 'proc_characteristics.dat'))

        if self.cluster_mode == 1:
            cluster_name = self.options['cluster_type']
            self.cluster = cluster.from_name[cluster_name](**self.options)
        if self.cluster_mode == 2:
            try:
                import multiprocessing
                if not self.nb_core:
                    try:
                        self.nb_core = int(self.options['nb_core'])
                    except TypeError:
                        self.nb_core = multiprocessing.cpu_count()
                logger.info('Using %d cores' % self.nb_core)
            except ImportError:
                self.nb_core = 1
                logger.warning('Impossible to detect the number of cores => Using One.\n'+
                        'Use set nb_core X in order to set this number and be able to'+
                        'run in multicore.')

            self.cluster = cluster.MultiCore(**self.options)
        self.update_random_seed()
        #find and keep track of all the jobs
        folder_names = {'LO': ['born_G*'], 'NLO': ['viSB_G*', 'novB_G*'],
                    'aMC@LO': ['GB*'], 'aMC@NLO': ['GV*', 'GF*']}
        folder_names['noshower'] = folder_names['aMC@NLO']
        folder_names['noshowerLO'] = folder_names['aMC@LO']
        job_dict = {}
        p_dirs = [file for file in os.listdir(pjoin(self.me_dir, 'SubProcesses')) 
                    if file.startswith('P') and \
                    os.path.isdir(pjoin(self.me_dir, 'SubProcesses', file))]
        #find jobs and clean previous results
        if not options['only_generation'] and not options['reweightonly']:
            self.update_status('Cleaning previous results', level=None)
        for dir in p_dirs:
            job_dict[dir] = [file for file in \
                                 os.listdir(pjoin(self.me_dir, 'SubProcesses', dir)) \
                                 if file.startswith('ajob')] 
            if not options['only_generation'] and not options['reweightonly']:
                for obj in folder_names[mode]:
                    to_rm = [file for file in \
                                 os.listdir(pjoin(self.me_dir, 'SubProcesses', dir)) \
                                 if file.startswith(obj[:-1]) and \
                                (os.path.isdir(pjoin(self.me_dir, 'SubProcesses', dir, file)) or \
                                 os.path.exists(pjoin(self.me_dir, 'SubProcesses', dir, file)))] 
                    files.rm([pjoin(self.me_dir, 'SubProcesses', dir, d) for d in to_rm])

        mcatnlo_status = ['Setting up grid', 'Computing upper envelope', 'Generating events']

        if options['reweightonly']:
            event_norm=self.run_card['event_norm']
            nevents=int(self.run_card['nevents'])
            self.reweight_and_collect_events(options, mode, nevents, event_norm)
            return

        devnull = os.open(os.devnull, os.O_RDWR) 
        if mode in ['LO', 'NLO']:
            logger.info('Doing fixed order %s' % mode)
            if mode == 'LO':
                if not options['only_generation']:
                    npoints = self.run_card['npoints_FO_grid']
                    niters = self.run_card['niters_FO_grid']
                    self.write_madin_file(pjoin(self.me_dir, 'SubProcesses'), 'born', 1, npoints, niters) 
                    self.update_status('Setting up grids', level=None)
                    self.run_all(job_dict, [['0', 'born', '0']], 'Setting up grids')
                p = misc.Popen(['./combine_results_FO.sh', 'born_G*'], \
                                   stdout=subprocess.PIPE, \
                                   cwd=pjoin(self.me_dir, 'SubProcesses'))
                output = p.communicate()
                self.cross_sect_dict = self.read_results(output, mode)
                self.print_summary(options, 0, mode)
                cross, error = sum_html.make_all_html_results(self, ['grid*'])
                self.results.add_detail('cross', cross)
                self.results.add_detail('error', error) 

                npoints = self.run_card['npoints_FO']
                niters = self.run_card['niters_FO']
                self.write_madin_file(pjoin(self.me_dir, 'SubProcesses'), 'born', 3, npoints, niters) 
                self.update_status('Computing cross-section', level=None)
                self.run_all(job_dict, [['0', 'born', '0']], 'Computing cross-section')
            elif mode == 'NLO':
                if not options['only_generation']:
                    self.update_status('Setting up grid', level=None)
                    npoints = self.run_card['npoints_FO_grid']
                    niters = self.run_card['niters_FO_grid']
                    self.write_madin_file(pjoin(self.me_dir, 'SubProcesses'), 'grid', 1, npoints, niters) 
                    self.run_all(job_dict, [['0', 'grid', '0']], 'Setting up grid using Born')
                p = misc.Popen(['./combine_results_FO.sh', 'grid_G*'], \
                                    stdout=subprocess.PIPE, 
                                    cwd=pjoin(self.me_dir, 'SubProcesses'))
                output = p.communicate()
                self.cross_sect_dict = self.read_results(output, mode)
                self.print_summary(options,0, mode)
                cross, error = sum_html.make_all_html_results(self, ['grid*'])
                self.results.add_detail('cross', cross)
                self.results.add_detail('error', error) 

                if not options['only_generation']:
                    npoints = self.run_card['npoints_FO_grid']
                    niters = self.run_card['niters_FO_grid']
                    self.write_madin_file(pjoin(self.me_dir, 'SubProcesses'), 'novB', 3, npoints, niters) 
                    self.update_status('Improving grid using NLO', level=None)
                    self.run_all(job_dict, [['0', 'novB', '0', 'grid']], \
                                     'Improving grids using NLO')
                p = misc.Popen(['./combine_results_FO.sh', 'novB_G*'], \
                                   stdout=subprocess.PIPE, cwd=pjoin(self.me_dir, 'SubProcesses'))
                output = p.communicate()
                self.cross_sect_dict = self.read_results(output, mode)
                self.print_summary(options, 0, mode)
                cross, error = sum_html.make_all_html_results(self, ['novB*'])
                self.results.add_detail('cross', cross)
                self.results.add_detail('error', error) 

                npoints = self.run_card['npoints_FO']
                niters = self.run_card['niters_FO']
                self.write_madin_file(pjoin(self.me_dir, 'SubProcesses'), 'novB', 3, npoints, niters) 
                npoints = self.run_card['npoints_FO_virt']
                niters = self.run_card['niters_FO_virt']
                self.write_madin_file(pjoin(self.me_dir, 'SubProcesses'), 'viSB', 3, npoints, niters) 
                self.update_status('Computing cross-section', level=None)
                self.run_all(job_dict, [['0', 'viSB', '0', 'grid'], ['0', 'novB', '0', 'novB']], \
                        'Computing cross-section')

            p = misc.Popen(['./combine_results_FO.sh'] + folder_names[mode], \
                                stdout=subprocess.PIPE, 
                                cwd=pjoin(self.me_dir, 'SubProcesses'))
            output = p.communicate()
            self.cross_sect_dict = self.read_results(output, mode)
            self.print_summary(options, 1, mode)
            misc.call(['./combine_plots_FO.sh'] + folder_names[mode], \
                                stdout=devnull, 
                                cwd=pjoin(self.me_dir, 'SubProcesses'))

            files.cp(pjoin(self.me_dir, 'SubProcesses', 'MADatNLO.top'),
                     pjoin(self.me_dir, 'Events', self.run_name))
            files.cp(pjoin(self.me_dir, 'SubProcesses', 'res.txt'),
                     pjoin(self.me_dir, 'Events', self.run_name))
            logger.info('The results of this run and the TopDrawer file with the plots' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))


            cross, error = sum_html.make_all_html_results(self, folder_names[mode])
            self.results.add_detail('cross', cross)
            self.results.add_detail('error', error) 
            self.update_status('Run completed', level='parton', update_results=True)
            return

        elif mode in ['aMC@NLO','aMC@LO','noshower','noshowerLO']:
            shower = self.run_card['parton_shower'].upper()
            nevents = int(self.run_card['nevents'])
            req_acc = self.run_card['req_acc']
            if nevents == 0 and float(req_acc) < 0 :
                raise aMCatNLOError('Cannot determine the required accuracy from the number '\
                                        'of events, because 0 events requested. Please set '\
                                        'the "req_acc" parameter in the run_card to a value between 0 and 1')
            elif float(req_acc) >1 or float(req_acc) == 0 :
                raise aMCatNLOError('Required accuracy ("req_acc" in the run_card) should '\
                                        'be between larger than 0 and smaller than 1, '\
                                        'or set to -1 for automatic determination. Current value is %s' % req_acc)

            shower_list = ['HERWIG6', 'HERWIGPP', 'PYTHIA6Q', 'PYTHIA6PT', 'PYTHIA8']

            if not shower in shower_list:
                raise aMCatNLOError('%s is not a valid parton shower. Please use one of the following: %s' \
                    % (shower, ', '.join(shower_list)))

# check that PYTHIA6PT is not used for processes with FSR
            if shower == 'PYTHIA6PT' and \
                self.proc_characteristics['has_fsr'] == '.true.':
                raise aMCatNLOError('PYTHIA6PT does not support processes with FSR')

            if mode in ['aMC@NLO', 'aMC@LO']:
                logger.info('Doing %s matched to parton shower' % mode[4:])
            elif mode in ['noshower','noshowerLO']:
                logger.info('Generating events without running the shower.')
            elif options['only_generation']:
                logger.info('Generating events starting from existing results')
            

            for i, status in enumerate(mcatnlo_status):
                if i == 2 or not options['only_generation']:
                    # if the number of events requested is zero,
                    # skip mint step 2
                    if i==2 and nevents==0:
                        self.print_summary(options, 2,mode)
                        return

                    self.update_status(status, level='parton')
                    if mode in ['aMC@NLO', 'noshower']:
                        self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), 'novB', i) 
                        self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), 'viSB', i) 
                        self.run_all(job_dict, [['2', 'V', '%d' % i], ['2', 'F', '%d' % i]], status)
                        
                    elif mode in ['aMC@LO', 'noshowerLO']:
                        self.write_madinMMC_file(
                            pjoin(self.me_dir, 'SubProcesses'), 'born', i) 
                        self.run_all(job_dict, [['2', 'B', '%d' % i]], '%s at LO' % status)

                if (i < 2 and not options['only_generation']) or i == 1 :
                    p = misc.Popen(['./combine_results.sh'] + [ '%d' % i,'%d' % nevents, '%s' % req_acc ] + folder_names[mode],
                            stdout=subprocess.PIPE, cwd = pjoin(self.me_dir, 'SubProcesses'))
                    output = p.communicate()
                    files.cp(pjoin(self.me_dir, 'SubProcesses', 'res_%d_abs.txt' % i), \
                             pjoin(self.me_dir, 'Events', self.run_name))
                    files.cp(pjoin(self.me_dir, 'SubProcesses', 'res_%d_tot.txt' % i), \
                             pjoin(self.me_dir, 'Events', self.run_name))

                    self.cross_sect_dict = self.read_results(output, mode)
                    self.print_summary(options, i, mode)

                    cross, error = sum_html.make_all_html_results(self, folder_names[mode])
                    self.results.add_detail('cross', cross)
                    self.results.add_detail('error', error) 

        if self.cluster_mode == 1:
            #if cluster run, wait 15 sec so that event files are transferred back
            self.update_status(
                    'Waiting while files are transferred back from the cluster nodes',
                    level='parton')
            time.sleep(10)

        event_norm=self.run_card['event_norm']
        return self.reweight_and_collect_events(options, mode, nevents, event_norm)

    def read_results(self, output, mode):
        """extract results (cross-section, absolute cross-section and errors)
        from output, which should be formatted as
            Found 4 correctly terminated jobs 
            random seed found in 'randinit' is 33
            Integrated abs(cross-section)
            7.94473937e+03 +- 2.9953e+01  (3.7702e-01%)
            Found 4 correctly terminated jobs 
            Integrated cross-section
            6.63392298e+03 +- 3.7669e+01  (5.6782e-01%)
        for aMC@NLO/aMC@LO, and as

        for NLO/LO
        The cross_sect_dict is returned"""
        res = {}
        if mode in ['aMC@LO', 'aMC@NLO', 'noshower', 'noshowerLO']:
            pat = re.compile(\
'''Found (\d+) correctly terminated jobs 
random seed found in 'randinit' is (\d+)
Integrated abs\(cross-section\)
\s*(\d+\.\d+e[+-]\d+) \+\- (\d+\.\d+e[+-]\d+)  \((\d+\.\d+e[+-]\d+)\%\)
Found (\d+) correctly terminated jobs 
Integrated cross-section
\s*(\-?\d+\.\d+e[+-]\d+) \+\- (\d+\.\d+e[+-]\d+)  \((\-?\d+\.\d+e[+-]\d+)\%\)''')
        else:
            pat = re.compile(\
'''Found (\d+) correctly terminated jobs 
\s*(\-?\d+\.\d+e[+-]\d+) \+\- (\d+\.\d+e[+-]\d+)  \((\-?\d+\.\d+e[+-]\d+)\%\)''')
            pass

        match = re.search(pat, output[0])
        if not match or output[1]:
            logger.info('Return code of the event collection: '+str(output[1]))
            logger.info('Output of the event collection:\n'+output[0])
            raise aMCatNLOError('An error occurred during the collection of results')
#        if int(match.groups()[0]) != self.njobs:
#            raise aMCatNLOError('Not all jobs terminated successfully')
        if mode in ['aMC@LO', 'aMC@NLO', 'noshower', 'noshowerLO']:
            return {'randinit' : int(match.groups()[1]),
                    'xseca' : float(match.groups()[2]),
                    'erra' : float(match.groups()[3]),
                    'xsect' : float(match.groups()[6]),
                    'errt' : float(match.groups()[7])}
        else:
            return {'xsect' : float(match.groups()[1]),
                    'errt' : float(match.groups()[2])}

    def print_summary(self, options, step, mode, scale_pdf_info={}):
        """print a summary of the results contained in self.cross_sect_dict.
        step corresponds to the mintMC step, if =2 (i.e. after event generation)
        some additional infos are printed"""
        # find process name
        proc_card_lines = open(pjoin(self.me_dir, 'Cards', 'proc_card_mg5.dat')).read().split('\n')
        process = ''
        for line in proc_card_lines:
            if line.startswith('generate'):
                process = line.replace('generate ', '')
        lpp = {'0':'l', '1':'p', '-1':'pbar'}
        proc_info = '\n      Process %s\n      Run at %s-%s collider (%s + %s GeV)' % \
        (process, lpp[self.run_card['lpp1']], lpp[self.run_card['lpp1']], 
                self.run_card['ebeam1'], self.run_card['ebeam2'])
        
        # Gather some basic statistics for the run and extracted from the log files.
        # > UPS is a dictionary of tuples with this format {channel:[nPS,nUPS]}
        # > Errors is a list of tuples with this format (log_file,nErrors)
        stats = {'UPS':{}, 'Errors':[]}
        if mode in ['aMC@NLO', 'noshower']: 
            log_GV_files =  glob.glob(pjoin(self.me_dir, \
                                    'SubProcesses', 'P*','GV*','log_MINT*.txt'))
            all_log_files = sum([glob.glob(pjoin(self.me_dir, 'SubProcesses', 'P*',\
                                    'G%s*'%foldname,'log*.txt')) for foldname in ['V','F']],[])
        elif mode in ['aMC@LO', 'noshowerLO']: 
            log_GV_files = ''
            all_log_files = glob.glob(pjoin(self.me_dir, \
                                          'SubProcesses', 'P*','GB*','log*.txt'))
        elif mode == 'NLO':
            log_GV_files =  glob.glob(pjoin(self.me_dir, \
                                    'SubProcesses', 'P*','viSB_G*','log*.txt'))
            all_log_files = sum([glob.glob(pjoin(self.me_dir,'SubProcesses', 'P*',
              '%sG*'%foldName,'log*.txt')) for foldName in ['grid_','novB_',\
                                                                   'viSB_']],[])
        elif mode == 'LO':
            log_GV_files = ''
            all_log_files = sum([glob.glob(pjoin(self.me_dir,'SubProcesses', 'P*',
              '%sG*'%foldName,'log*.txt')) for foldName in ['grid_','born_']],[])
        else:
            raise aMCatNLOError, 'Running mode %s not supported.'%mode
        # Find the number of potential errors found in all log files
        # This re is a simple match on a case-insensitve 'error' but there is 
        # also some veto added for excluding the sentence 
        #  "See Section 6 of paper for error calculation."
        # which appear in the header of lhapdf in the logs.
        err_finder = re.compile(\
             r"(?<!of\spaper\sfor\s)\bERROR\b(?!\scalculation\.)",re.IGNORECASE)
        for log in all_log_files:
            logfile=open(log,'r')
            nErrors = len(re.findall(err_finder, logfile.read()))
            logfile.close()
            if nErrors != 0:
                stats['Errors'].append((str(log),nErrors))
            
        
        if mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO']:
            status = ['Determining the number of unweighted events per channel',
                      'Updating the number of unweighted events per channel',
                      'Summary:']
            if step != 2:
                message = status[step] + '\n\n      Intermediate results:' + \
                    ('\n      Random seed: %(randinit)d' + \
                     '\n      Total cross-section:      %(xsect)8.3e +- %(errt)6.1e pb' + \
                     '\n      Total abs(cross-section): %(xseca)8.3e +- %(erra)6.1e pb \n') \
                     % self.cross_sect_dict
            else:
        
                message = '\n      ' + status[step] + proc_info + \
                          '\n      Total cross-section: %(xsect)8.3e +- %(errt)6.1e pb' % \
                        self.cross_sect_dict

                if int(self.run_card['nevents'])>=10000 and self.run_card['reweight_scale']=='.true.':
                   message = message + \
                       ('\n      Ren. and fac. scale uncertainty: +%0.1f%% -%0.1f%%') % \
                       (scale_pdf_info['scale_upp'], scale_pdf_info['scale_low'])
                if int(self.run_card['nevents'])>=10000 and self.run_card['reweight_PDF']=='.true.':
                   message = message + \
                       ('\n      PDF uncertainty: +%0.1f%% -%0.1f%%') % \
                       (scale_pdf_info['pdf_upp'], scale_pdf_info['pdf_low'])

                neg_frac = (self.cross_sect_dict['xseca'] - self.cross_sect_dict['xsect'])/\
                       (2. * self.cross_sect_dict['xseca'])
                message = message + \
                    ('\n      Number of events generated: %s' + \
                     '\n      Parton shower to be used: %s' + \
                     '\n      Fraction of negative weights: %4.2f' + \
                     '\n      Total running time : %s') % \
                        (self.run_card['nevents'],
                         self.run_card['parton_shower'],
                         neg_frac, 
                         misc.format_timer(time.time()-self.start_time))

        elif mode in ['NLO', 'LO']:
            status = ['Results after grid setup (cross-section is non-physical):',
                      'Final results and run summary:']
            if step == 0:
                message = '\n      ' + status[step] + \
                     '\n      Total cross-section:      %(xsect)8.3e +- %(errt)6.1e pb' % \
                             self.cross_sect_dict
            elif step == 1:
                message = '\n      ' + status[step] + proc_info + \
                     '\n      Total cross-section:      %(xsect)8.3e +- %(errt)6.1e pb' % \
                             self.cross_sect_dict
        
        if (mode in ['NLO', 'LO'] and step!=1) or \
           (mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO'] and step!=2):
            logger.info(message+'\n')
            return

# Now display the general statistics
# Recuperate the fraction of unstable PS points found in the runs for
# the virtuals
        UPS_stat_finder = re.compile(r".*Total points tried\:\s+(?P<ntot>\d+).*"+\
             r".*Stability unknown\:\s+(?P<nsun>\d+).*"+\
             r".*Stable PS point\:\s+(?P<nsps>\d+).*"+\
             r".*Unstable PS point \(and rescued\)\:\s+(?P<nups>\d+).*"+\
             r".*Exceptional PS point \(unstable and not rescued\)\:\s+(?P<neps>\d+).*"+\
             r".*Double precision used\:\s+(?P<nddp>\d+).*"+\
             r".*Quadruple precision used\:\s+(?P<nqdp>\d+).*"+\
             r".*Initialization phase\-space points\:\s+(?P<nini>\d+).*"+\
             r".*Unknown return code \(100\)\:\s+(?P<n100>\d+).*"+\
             r".*Unknown return code \(10\)\:\s+(?P<n10>\d+).*"+\
             r".*Unknown return code \(1\)\:\s+(?P<n1>\d+).*",re.DOTALL)
#        UPS_stat_finder = re.compile(r".*Total points tried\:\s+(?P<nPS>\d+).*"+\
#                      r"Unstable points \(check UPS\.log for the first 10\:\)"+\
#                                                r"\s+(?P<nUPS>\d+).*",re.DOTALL)
        for gv_log in log_GV_files:
            logfile=open(gv_log,'r')             
            UPS_stats = re.search(UPS_stat_finder,logfile.read())
            logfile.close()
            if not UPS_stats is None:
                channel_name = '/'.join(gv_log.split('/')[-5:-1])
                try:
                    stats['UPS'][channel_name][0] += int(UPS_stats.group('ntot'))
                    stats['UPS'][channel_name][1] += int(UPS_stats.group('nsun'))
                    stats['UPS'][channel_name][2] += int(UPS_stats.group('nsps'))
                    stats['UPS'][channel_name][3] += int(UPS_stats.group('nups'))
                    stats['UPS'][channel_name][4] += int(UPS_stats.group('neps'))
                    stats['UPS'][channel_name][5] += int(UPS_stats.group('nddp'))
                    stats['UPS'][channel_name][6] += int(UPS_stats.group('nqdp'))
                    stats['UPS'][channel_name][7] += int(UPS_stats.group('nini'))
                    stats['UPS'][channel_name][8] += int(UPS_stats.group('n100'))
                    stats['UPS'][channel_name][9] += int(UPS_stats.group('n10'))
                    stats['UPS'][channel_name][10] += int(UPS_stats.group('n1'))
                except KeyError:
                    stats['UPS'][channel_name] = [int(UPS_stats.group('ntot')),
                      int(UPS_stats.group('nsun')),int(UPS_stats.group('nsps')),
                      int(UPS_stats.group('nups')),int(UPS_stats.group('neps')),
                      int(UPS_stats.group('nddp')),int(UPS_stats.group('nqdp')),
                      int(UPS_stats.group('nini')),int(UPS_stats.group('n100')),
                      int(UPS_stats.group('n10')),int(UPS_stats.group('n1'))]
        debug_msg = ""
        if len(stats['UPS'].keys())>0:
            nTotPS  = sum([chan[0] for chan in stats['UPS'].values()],0)
            nTotsun = sum([chan[1] for chan in stats['UPS'].values()],0)
            nTotsps = sum([chan[2] for chan in stats['UPS'].values()],0)
            nTotups = sum([chan[3] for chan in stats['UPS'].values()],0)
            nToteps = sum([chan[4] for chan in stats['UPS'].values()],0)
            nTotddp = sum([chan[5] for chan in stats['UPS'].values()],0)
            nTotqdp = sum([chan[6] for chan in stats['UPS'].values()],0)
            nTotini = sum([chan[7] for chan in stats['UPS'].values()],0)
            nTot100 = sum([chan[8] for chan in stats['UPS'].values()],0)
            nTot10  = sum([chan[9] for chan in stats['UPS'].values()],0)
            nTot1  = sum([chan[10] for chan in stats['UPS'].values()],0)
            UPSfracs = [(chan[0] , 0.0 if chan[1][0]==0 else \
                 float(chan[1][4]*100)/chan[1][0]) for chan in stats['UPS'].items()]
            maxUPS = max(UPSfracs, key = lambda w: w[1])
            if maxUPS[1]>0.001:
                message += '\n      Number of loop ME evaluations (by MadLoop): %d'%nTotPS
                message += '\n          Stability unknown:                   %d'%nTotsun
                message += '\n          Stable PS point:                     %d'%nTotsps
                message += '\n          Unstable PS point (and rescued):     %d'%nTotups
                message += '\n          Unstable PS point (and not rescued): %d'%nToteps
                message += '\n          Only double precision used:          %d'%nTotddp
                message += '\n          Quadruple precision used:            %d'%nTotqdp
                message += '\n          Initialization phase-space points:   %d'%nTotini
                if nTot100 != 0:
                    message += '\n          Unknown return code (100):           %d'%nTot100
                if nTot10 != 0:
                    message += '\n          Unknown return code (10):            %d'%nTot10
                if nTot1 != 0:
                    message += '\n          Unknown return code (1):             %d'%nTot1
                message += '\n      Total number of unstable PS point detected:'+\
                                 ' %d (%4.2f%%)'%(nToteps,float(100*nToteps)/nTotPS)
                message += '\n      Maximum fraction of UPS points in '+\
                          'channel %s (%4.2f%%)'%maxUPS
                message += '\n      Please report this to the authors while '+\
                                                                'providing the file'
                message += '\n      %s'%str(pjoin(os.path.dirname(self.me_dir),
                                                               maxUPS[0],'UPS.log'))
            else:
                debug_msg += '\n      Number of loop ME evaluations (by MadLoop): %d'%nTotPS
                debug_msg += '\n          Stability unknown:                   %d'%nTotsun
                debug_msg += '\n          Stable PS point:                     %d'%nTotsps
                debug_msg += '\n          Unstable PS point (and rescued):     %d'%nTotups
                debug_msg += '\n          Unstable PS point (and not rescued): %d'%nToteps
                debug_msg += '\n          Only double precision used:          %d'%nTotddp
                debug_msg += '\n          Quadruple precision used:            %d'%nTotqdp
                debug_msg += '\n          Initialization phase-space points:   %d'%nTotini
                if nTot100 != 0:
                    debug_msg += '\n          Unknown return code (100):           %d'%nTot100
                if nTot10 != 0:
                    debug_msg += '\n          Unknown return code (10):            %d'%nTot10
                if nTot1 != 0:
                    debug_msg += '\n          Unknown return code (1):             %d'%nTot1
        logger.info(message+'\n')
                 
        nErrors = sum([err[1] for err in stats['Errors']],0)
        if nErrors != 0:
            debug_msg += '\n      WARNING:: A total of %d error%s ha%s been '\
              %(nErrors,'s' if nErrors>1 else '','ve' if nErrors>1 else 's')+\
              'found in the following log file%s:'%('s' if \
                                                 len(stats['Errors'])>1 else '')
            for error in stats['Errors'][:3]:
                log_name = '/'.join(error[0].split('/')[-5:])
                debug_msg += '\n       > %d error%s in %s'%\
                                   (error[1],'s' if error[1]>1 else '',log_name)
            if len(stats['Errors'])>3:
                nRemainingErrors = sum([err[1] for err in stats['Errors']][3:],0)
                nRemainingLogs = len(stats['Errors'])-3
                debug_msg += '\n      And another %d error%s in %d other log file%s'%\
                           (nRemainingErrors, 's' if nRemainingErrors>1 else '',
                               nRemainingLogs, 's ' if nRemainingLogs>1 else '')
        logger.debug(debug_msg)




    def reweight_and_collect_events(self, options, mode, nevents, event_norm):
        """this function calls the reweighting routines and creates the event file in the 
        Event dir. Return the name of the event file created
        """
        scale_pdf_info={}
        if self.run_card['reweight_scale'] == '.true.' or self.run_card['reweight_PDF'] == '.true.':
            scale_pdf_info = self.run_reweight(options['reweightonly'])

        self.update_status('Collecting events', level='parton', update_results=True)
        misc.compile(['collect_events'], 
                    cwd=pjoin(self.me_dir, 'SubProcesses'))
        p = misc.Popen(['./collect_events'], cwd=pjoin(self.me_dir, 'SubProcesses'),
                stdin=subprocess.PIPE, 
                stdout=open(pjoin(self.me_dir, 'collect_events.log'), 'w'))
        if event_norm == 'sum':
            p.communicate(input = '1\n')
        else:
            p.communicate(input = '2\n')

        #get filename from collect events
        filename = open(pjoin(self.me_dir, 'collect_events.log')).read().split()[-1]

        if not os.path.exists(pjoin(self.me_dir, 'SubProcesses', filename)):
            raise aMCatNLOError('An error occurred during event generation. ' + \
                    'The event file has not been created. Check collect_events.log')
        evt_file = pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe')
        files.mv(pjoin(self.me_dir, 'SubProcesses', filename), evt_file)
        misc.call(['gzip %s' % evt_file], shell=True)
        if not options['reweightonly']:
            self.print_summary(options, 2, mode, scale_pdf_info)
        logger.info('The %s.gz file has been generated.\n' \
                % (evt_file))
        self.results.add_detail('nb_event', nevents)
        self.update_status('Events generated', level='parton', update_results=True)
        return evt_file


    def run_mcatnlo(self, evt_file):
        """runs mcatnlo on the generated event file, to produce showered-events"""
        logger.info('   Prepairing MCatNLO run')
        self.update_status('Showering events', level='parton', update_results = True) 
        self.run_name = os.path.split(\
                    os.path.relpath(evt_file, pjoin(self.me_dir, 'Events')))[0]

        try:
            misc.call(['gunzip %s.gz' % evt_file], shell=True)
        except Exception:
            pass

        self.banner = banner_mod.Banner(evt_file)
        shower = self.banner.get_detail('run_card', 'parton_shower').upper()
        self.banner_to_mcatnlo(evt_file)
        shower_card_path = pjoin(self.me_dir, 'MCatNLO', 'shower_card.dat')
        
        # set environmental variables for the run
        if 'LD_LIBRARY_PATH' in os.environ.keys():
            ldlibrarypath = os.environ['LD_LIBRARY_PATH']
        else:
            ldlibrarypath = ''
        for path in self.shower_card['extrapaths'].split():
            ldlibrarypath += ':%s' % path
        if shower == 'HERWIGPP':
            ldlibrarypath += ':%s' % pjoin(self.options['hepmc_path'], 'lib')
        os.putenv('LD_LIBRARY_PATH', ldlibrarypath)

        self.shower_card.write_card(shower, shower_card_path)

        mcatnlo_log = pjoin(self.me_dir, 'mcatnlo.log')
        self.update_status('   Compiling MCatNLO for %s...' % shower, level='parton') 
        misc.call(['./MCatNLO_MadFKS.inputs'], stdout=open(mcatnlo_log, 'w'),
                    stderr=open(mcatnlo_log, 'w'), 
                    cwd=pjoin(self.me_dir, 'MCatNLO'))
        exe = 'MCATNLO_%s_EXE' % shower
        if not os.path.exists(pjoin(self.me_dir, 'MCatNLO', exe)) and \
            not os.path.exists(pjoin(self.me_dir, 'MCatNLO', 'Pythia8.exe')):
            print open(mcatnlo_log).read()
            raise aMCatNLOError('Compilation failed, check %s for details' % mcatnlo_log)
        logger.info('                     ... done')
        # create an empty dir where to run
        count = 1
        while os.path.isdir(pjoin(self.me_dir, 'MCatNLO', 'RUN_%s_%d' % \
                        (shower, count))):
            count += 1
        rundir = pjoin(self.me_dir, 'MCatNLO', 'RUN_%s_%d' % \
                        (shower, count))
        os.mkdir(rundir)
        files.cp(shower_card_path, rundir)

        self.update_status('Running MCatNLO in %s (this may take some time)...' % rundir,
                level='parton')
        if shower != 'PYTHIA8':
            files.mv(pjoin(self.me_dir, 'MCatNLO', exe), rundir)
            files.mv(pjoin(self.me_dir, 'MCatNLO', 'MCATNLO_%s_input' % shower), rundir)
        # special treatment for pythia8
        else:
            files.mv(pjoin(self.me_dir, 'MCatNLO', 'Pythia8.cmd'), rundir)
            files.mv(pjoin(self.me_dir, 'MCatNLO', 'Pythia8.exe'), rundir)
        #link the hwpp exe in the rundir
        if shower == 'HERWIGPP':
            try:
                misc.call(['ln -s %s %s' % \
                (pjoin(self.options['hwpp_path'], 'bin', 'Herwig++'), rundir)], shell=True)
            except Exception:
                raise aMCatNLOError('The Herwig++ path set in the configuration file is not valid.')

            if os.path.exists(pjoin(self.me_dir, 'MCatNLO', 'HWPPAnalyzer', 'HepMCFortran.so')):
                files.cp(pjoin(self.me_dir, 'MCatNLO', 'HWPPAnalyzer', 'HepMCFortran.so'), rundir)

        evt_name = os.path.basename(evt_file)
        misc.call(['ln -s %s %s' % (os.path.split(evt_file)[0], 
            pjoin(rundir,self.run_name))], shell=True)
        # special treatment for pythia8
        if shower=='PYTHIA8':
            open(pjoin(rundir, exe), 'w').write(\
                 '#!/bin/bash\nsource %s\n./Pythia8.exe Pythia8.cmd\n'\
                % pjoin(self.options['pythia8_path'], 'examples', 'config.sh'))
            os.system('chmod  +x %s' % pjoin(rundir,exe))
            misc.call(['./%s' % exe], cwd = rundir, 
                stdout=open(pjoin(rundir,'mcatnlo_run.log'), 'w'),
                stderr=open(pjoin(rundir,'mcatnlo_run.log'), 'w'),
                shell=True)
        else:
            misc.call(['./%s' % exe], cwd = rundir, 
                stdin=open(pjoin(rundir,'MCATNLO_%s_input' % shower)),
                stdout=open(pjoin(rundir,'mcatnlo_run.log'), 'w'),
                stderr=open(pjoin(rundir,'mcatnlo_run.log'), 'w'))
        #copy the showered stdhep file back in events
        if not self.shower_card['analyse']:
            if os.path.exists(pjoin(rundir, self.run_name, evt_name + '.hep')):
                hep_file = '%s_%s_0.hep' % (evt_file[:-4], shower)
                count = 0
                while os.path.exists(hep_file + '.gz'):
                    count +=1
                    hep_file = '%s_%s_%d.hep' % (evt_file[:-4], shower, count)

                misc.call(['mv %s %s' % (pjoin(rundir, self.run_name, evt_name + '.hep'), hep_file)], shell=True) 
                misc.call(['gzip %s' % evt_file], shell=True)
                misc.call(['gzip %s' % hep_file], shell=True)

                logger.info(('The file %s.gz has been generated. \nIt contains showered' + \
                            ' and hadronized events in the StdHEP format obtained' + \
                            ' showering the parton-level event file %s.gz with %s') % \
                            (hep_file, evt_file, shower))
            #this is for hw++
            elif os.path.exists(pjoin(rundir, 'MCATNLO_HERWIGPP.hepmc')):
                hep_file = '%s_%s_0.hepmc' % (evt_file[:-4], shower)
                count = 0
                while os.path.exists(hep_file + '.gz'):
                    count +=1
                    hep_file = '%s_%s_%d.hepmc' % (evt_file[:-4], shower, count)

                misc.call(['mv %s %s' % \
                    (pjoin(rundir, 'MCATNLO_HERWIGPP.hepmc'), hep_file)], shell=True) 
                misc.call(['gzip %s' % evt_file], shell=True)
                misc.call(['gzip %s' % hep_file], shell=True)
                logger.info(('The file %s.gz has been generated. \nIt contains showered' + \
                            ' and hadronized events in the HEPMC format obtained' + \
                            ' showering the parton-level event file %s.gz with %s') % \
                            (hep_file, evt_file, shower))
            #this is for pythia8
            elif os.path.exists(pjoin(rundir, 'Pythia8.hep')):
                hep_file = '%s_%s_0.hep' % (evt_file[:-4], shower)
                count = 0
                while os.path.exists(hep_file + '.gz'):
                    count +=1
                    hep_file = '%s_%s_%d.hepmc' % (evt_file[:-4], shower, count)

                misc.call(['mv %s %s' % \
                    (pjoin(rundir, 'Pythia8.hep'), hep_file)], shell=True) 
                misc.call(['gzip %s' % evt_file], shell=True)
                misc.call(['gzip %s' % hep_file], shell=True)
                logger.info(('The file %s.gz has been generated. \nIt contains showered' + \
                            ' and hadronized events in the HEPMC format obtained' + \
                            ' showering the parton-level event file %s.gz with %s') % \
                            (hep_file, evt_file, shower))

            else:
                raise aMCatNLOError('No file has been generated, an error occurred. More information in %s' % pjoin(os.getcwd(), 'amcatnlo_run.log'))
        else:
            topfiles = [n for n in os.listdir(pjoin(rundir)) \
                                            if n.lower().endswith('.top')]
            if not topfiles:
                misc.call(['gzip %s' % evt_file], shell=True)
                logger.warning('No .top file has been generated. For the results of your ' +\
                               'run, please check inside %s' % rundir)

	    else:    
                filename = 'plot_%s_%d_' % (shower, 1)
                count = 1
                while os.path.exists(pjoin(self.me_dir, 'Events', 
                    self.run_name, '%s0.top' % filename)):
                    count += 1
                    filename = 'plot_%s_%d_' % (shower, count)
                plotfiles = [] 
                for i, file in enumerate(topfiles):
                    plotfile = pjoin(self.me_dir, 'Events', self.run_name, 
                              '%s%d.top' % (filename, i))
                    misc.call(['mv %s %s' % \
                        (pjoin(rundir, file), plotfile)], shell=True) 

                    plotfiles.append(plotfile)

                ffiles = 'files'
                have = 'have'
                if len(plotfiles) == 1:
                    ffiles = 'file'
                    have = 'has'

                misc.call(['gzip %s' % evt_file], shell=True)
                logger.info(('The %s %s %s been generated, with histograms in the' + \
                        ' TopDrawer format, obtained by showering the parton-level' + \
                        ' file %s.gz with %s') % (ffiles, ', '.join(plotfiles), have, \
                        evt_file, shower))

        self.update_status('Run completed', level='hadron', update_results=True)



    ############################################################################
    def set_run_name(self, name, tag=None, level='parton', reload_card=False):
        """define the run name, the run_tag, the banner and the results."""
        
        # when are we force to change the tag new_run:previous run requiring changes
        upgrade_tag = {'parton': ['parton','pythia','pgs','delphes'],
                       'pythia': ['pythia','pgs','delphes'],
                       'pgs': ['pgs'],
                       'delphes':['delphes'],
                       'plot':[]}
        
        

        if name == self.run_name:        
            if reload_card:
                run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
                self.run_card = banner_mod.RunCardNLO(run_card)

            #check if we need to change the tag
            if tag:
                self.run_card['run_tag'] = tag
                self.run_tag = tag
                self.results.add_run(self.run_name, self.run_card)
            else:
                for tag in upgrade_tag[level]:
                    if getattr(self.results[self.run_name][-1], tag):
                        tag = self.get_available_tag()
                        self.run_card['run_tag'] = tag
                        self.run_tag = tag
                        self.results.add_run(self.run_name, self.run_card)                        
                        break
            return # Nothing to do anymore
        
        # save/clean previous run
        if self.run_name:
            self.store_result()
        # store new name
        self.run_name = name
        
        # Read run_card
        run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
        self.run_card = banner_mod.RunCardNLO(run_card)

        new_tag = False
        # First call for this run -> set the banner
        self.banner = banner_mod.recover_banner(self.results, level)
        if tag:
            self.run_card['run_tag'] = tag
            new_tag = True
        elif not self.run_name in self.results and level =='parton':
            pass # No results yet, so current tag is fine
        elif not self.run_name in self.results:
            #This is only for case when you want to trick the interface
            logger.warning('Trying to run data on unknown run.')
            self.results.add_run(name, self.run_card)
            self.results.update('add run %s' % name, 'all', makehtml=True)
        else:
            for tag in upgrade_tag[level]:
                
                if getattr(self.results[self.run_name][-1], tag):
                    # LEVEL is already define in the last tag -> need to switch tag
                    tag = self.get_available_tag()
                    self.run_card['run_tag'] = tag
                    new_tag = True
                    break
            if not new_tag:
                # We can add the results to the current run
                tag = self.results[self.run_name][-1]['tag']
                self.run_card['run_tag'] = tag # ensure that run_tag is correct                
             
                    
        if name in self.results and not new_tag:
            self.results.def_current(self.run_name)
        else:
            self.results.add_run(self.run_name, self.run_card)

        self.run_tag = self.run_card['run_tag']

        # Return the tag of the previous run having the required data for this
        # tag/run to working wel.
        if level == 'parton':
            return
        elif level == 'pythia':
            return self.results[self.run_name][0]['tag']
        else:
            for i in range(-1,-len(self.results[self.run_name])-1,-1):
                tagRun = self.results[self.run_name][i]
                if tagRun.pythia:
                    return tagRun['tag']


    def store_result(self):
        """ tar the pythia results. This is done when we are quite sure that 
        the pythia output will not be use anymore """

        if not self.run_name:
            return

        self.results.save()

        if not self.to_store:
            return 
        
        tag = self.run_card['run_tag']
#        if 'pythia' in self.to_store:
#            self.update_status('Storing Pythia files of Previous run', level='pythia', error=True)
#            os.system('mv -f %(path)s/pythia_events.hep %(path)s/%(name)s/%(tag)s_pythia_events.hep' % 
#                  {'name': self.run_name, 'path' : pjoin(self.me_dir,'Events'),
#                   'tag':tag})
#            os.system('gzip -f %s/%s_pythia_events.hep' % ( 
#                                pjoin(self.me_dir,'Events',self.run_name), tag))
#            self.to_store.remove('pythia')
#            self.update_status('Done', level='pythia',makehtml=False,error=True)
        
        self.to_store = []


    def get_init_dict(self, evt_file):
        """reads the info in the init block and returns them in a dictionary"""
        ev_file = open(evt_file)
        init = ""
        found = False
        while True:
            line = ev_file.readline()
            if "<init>" in line:
                found = True
            elif found and not line.startswith('#'):
                init += line
            if "</init>" in line or "<event>" in line:
                break
        ev_file.close()

#       IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2), PDFGUP(1),PDFGUP(2),
#       PDFSUP(1),PDFSUP(2),IDWTUP,NPRUP
# these are not included (so far) in the init_dict
#       XSECUP(1),XERRUP(1),XMAXUP(1),LPRUP(1)
            
        init_dict = {}
        init_dict['idbmup1'] = int(init.split()[0])
        init_dict['idbmup2'] = int(init.split()[1])
        init_dict['ebmup1'] = float(init.split()[2])
        init_dict['ebmup2'] = float(init.split()[3])
        init_dict['pdfgup1'] = int(init.split()[4])
        init_dict['pdfgup2'] = int(init.split()[5])
        init_dict['pdfsup1'] = int(init.split()[6])
        init_dict['pdfsup2'] = int(init.split()[7])
        init_dict['idwtup'] = int(init.split()[8])
        init_dict['nprup'] = int(init.split()[9])

        return init_dict


    def banner_to_mcatnlo(self, evt_file):
        """creates the mcatnlo input script using the values set in the header of the event_file.
        It also checks if the lhapdf library is used"""
        shower = self.banner.get('run_card', 'parton_shower').upper()
        pdlabel = self.banner.get('run_card', 'pdlabel')
        itry = 0
        nevents = self.shower_card['nevents']
        init_dict = self.get_init_dict(evt_file)

        if nevents < 0 or nevents > self.banner.get_detail('run_card', 'nevents'):
            nevents = self.banner.get_detail('run_card', 'nevents')
        mcmass_dict = {}
        for line in [l for l in self.banner['montecarlomasses'].split('\n') if l]:
            pdg = int(line.split()[0])
            mass = float(line.split()[1])
            mcmass_dict[pdg] = mass

        # check if need to link lhapdf
        if pdlabel =='\'lhapdf\'':
            self.link_lhapdf(pjoin(self.me_dir, 'lib'))

        content = 'EVPREFIX=%s\n' % pjoin(self.run_name, os.path.split(evt_file)[1])
        content += 'NEVENTS=%s\n' % nevents
        content += 'MCMODE=%s\n' % shower
        content += 'PDLABEL=%s\n' % pdlabel
        content += 'ALPHAEW=%s\n' % self.banner.get_detail('param_card', 'sminputs', 1).value
        #content += 'PDFSET=%s\n' % self.banner.get_detail('run_card', 'lhaid')
        content += 'PDFSET=%s\n' % max([init_dict['pdfsup1'],init_dict['pdfsup2']])
        content += 'TMASS=%s\n' % self.banner.get_detail('param_card', 'mass', 6).value
        content += 'TWIDTH=%s\n' % self.banner.get_detail('param_card', 'decay', 6).value
        content += 'ZMASS=%s\n' % self.banner.get_detail('param_card', 'mass', 23).value
        content += 'ZWIDTH=%s\n' % self.banner.get_detail('param_card', 'decay', 23).value
        content += 'WMASS=%s\n' % self.banner.get_detail('param_card', 'mass', 24).value
        content += 'WWIDTH=%s\n' % self.banner.get_detail('param_card', 'decay', 24).value
        try:
            content += 'HGGMASS=%s\n' % self.banner.get_detail('param_card', 'mass', 25).value
            content += 'HGGWIDTH=%s\n' % self.banner.get_detail('param_card', 'decay', 25).value
        except KeyError:
            content += 'HGGMASS=120.\n'
            content += 'HGGWIDTH=0.00575308848\n'
        content += 'beammom1=%s\n' % self.banner.get_detail('run_card', 'ebeam1')
        content += 'beammom2=%s\n' % self.banner.get_detail('run_card', 'ebeam2')
        content += 'BEAM1=%s\n' % self.banner.get_detail('run_card', 'lpp1')
        content += 'BEAM2=%s\n' % self.banner.get_detail('run_card', 'lpp2')
        content += 'DMASS=%s\n' % mcmass_dict[1]
        content += 'UMASS=%s\n' % mcmass_dict[2]
        content += 'SMASS=%s\n' % mcmass_dict[3]
        content += 'CMASS=%s\n' % mcmass_dict[4]
        content += 'BMASS=%s\n' % mcmass_dict[5]
        content += 'GMASS=%s\n' % mcmass_dict[21]
        content += 'EVENT_NORM=%s\n' % self.banner.get_detail('run_card', 'event_norm')
        lhapdfpath = subprocess.Popen('%s --prefix' % self.options['lhapdf'], 
                shell = True, stdout = subprocess.PIPE).stdout.read().strip()
        if lhapdfpath:
            content += 'LHAPDFPATH=%s\n' % lhapdfpath
        else:
            #overwrite the PDFCODE variable in order to use internal lhapdf
            content += 'LHAPDFPATH=\n' 
            content += 'PDFCODE=0\n'
        # add the pythia8/hwpp path(s)
        if self.options['pythia8_path']:
            content+='PY8PATH=%s\n' % self.options['pythia8_path']
        if self.options['hwpp_path']:
            content+='HWPPPATH=%s\n' % self.options['hwpp_path']
        if self.options['thepeg_path']:
            content+='THEPEGPATH=%s\n' % self.options['thepeg_path']
        if self.options['hepmc_path']:
            content+='HEPMCPATH=%s\n' % self.options['hepmc_path']

        
        output = open(pjoin(self.me_dir, 'MCatNLO', 'banner.dat'), 'w')
        output.write(content)
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
                files.cp(nev_unw + '.orig', nev_unw)
            else:
                raise aMCatNLOError('Cannot find event file information')

        #read the nevents_unweighted file to get the list of event files
        file = open(nev_unw)
        lines = file.read().split('\n')
        file.close()
        # make copy of the original nevent_unweighted file
        files.cp(nev_unw, nev_unw + '.orig')
        # loop over lines (all but the last one whith is empty) and check that the
        #  number of events is not 0
        evt_files = [line.split()[0] for line in lines[:-1] if line.split()[1] != '0']
        #prepare the job_dict
        job_dict = {}
        exe = 'reweight_xsec_events.local'
        for i, evt_file in enumerate(evt_files):
            path, evt = os.path.split(evt_file)
            files.ln(pjoin(self.me_dir, 'SubProcesses', exe), \
                     pjoin(self.me_dir, 'SubProcesses', path))
            job_dict[path] = [exe]

        self.run_all(job_dict, [[evt, '1']], 'Running reweight')

        #check that the new event files are complete
        for evt_file in evt_files:
            last_line = subprocess.Popen('tail -n1 %s.rwgt ' % \
                    pjoin(self.me_dir, 'SubProcesses', evt_file), \
                shell = True, stdout = subprocess.PIPE).stdout.read().strip()
            if last_line != "</LesHouchesEvents>":
                raise aMCatNLOError('An error occurred during reweight. Check the' + \
                        '\'reweight_xsec_events.output\' files inside the ' + \
                        '\'SubProcesses/P*/G*/ directories for details')

        #update file name in nevents_unweighted
        newfile = open(nev_unw, 'w')
        for line in lines:
            if line:
                newfile.write(line.replace(line.split()[0], line.split()[0] + '.rwgt') + '\n')
        newfile.close()

        return self.pdf_scale_from_reweighting(evt_files)

    def pdf_scale_from_reweighting(self, evt_files):
        """This function takes the files with the scale and pdf values
        written by the reweight_xsec_events.f code
        (P*/G*/pdf_scale_uncertainty.dat) and computes the overall
        scale and PDF uncertainty (the latter is computed using the
        Hessian method (if lhaid<90000) or Gaussian (if lhaid>90000))
        and returns it in percents.  The expected format of the file
        is: n_scales xsec_scale_central xsec_scale1 ...  n_pdf
        xsec_pdf0 xsec_pdf1 ...."""
        scale_pdf_info={}
        scales=[]
        pdfs=[]
        numofpdf = 0
        numofscales = 0
        for evt_file in evt_files:
            path, evt=os.path.split(evt_file)
            data_file=open(pjoin(self.me_dir, 'SubProcesses', path, 'scale_pdf_dependence.dat')).read()
            lines = data_file.replace("D", "E").split("\n")
            if not numofscales:
                numofscales = int(lines[0])
            if not numofpdf:
                numofpdf = int(lines[2])
            scales_this = [float(val) for val in lines[1].split()]
            pdfs_this = [float(val) for val in lines[3].split()]

            if numofscales != len(scales_this) or numofpdf !=len(pdfs_this):
                # the +1 takes the 0th (central) set into account
                logger.info(data_file)
                logger.info((' Expected # of scales: %d\n'+
                             ' Found # of scales: %d\n'+
                             ' Expected # of pdfs: %d\n'+
                             ' Found # of pdfs: %d\n') %
                        (numofscales, len(scales_this), numofpdf, len(pdfs_this)))
                raise aMCatNLOError('inconsistent scale_pdf_dependence.dat')
            if not scales:
                scales = [0.] * numofscales
            if not pdfs:
                pdfs = [0.] * numofpdf

            scales = [a + b for a, b in zip(scales, scales_this)]
            pdfs = [a + b for a, b in zip(pdfs, pdfs_this)]

        # get the central value
        if numofscales>0 and numofpdf==0:
            cntrl_val=scales[0]
        elif numofpdf>0 and numofscales==0:
            cntrl_val=pdfs[0]
        elif numofpdf>0 and numofscales>0:
            if abs(1-scales[0]/pdfs[0])>0.0001:
                raise aMCatNLOError('Central values for scale and PDF variation not identical')
            else:
                cntrl_val=scales[0]

        # get the scale uncertainty in percent
        scale_upp=0.0
        scale_low=0.0
        if numofscales>0:
            scale_pdf_info['scale_upp'] = (max(scales)/cntrl_val-1)*100
            scale_pdf_info['scale_low'] = (1-min(scales)/cntrl_val)*100

        # get the pdf uncertainty in percent (according to the Hessian method)
        lhaid=int(self.run_card['lhaid'])
        pdf_upp=0.0
        pdf_low=0.0
        if lhaid <= 90000:
            # use Hessian method (CTEQ & MSTW)
            if numofpdf>1:
                for i in range(int(numofpdf/2)):
                    pdf_upp=pdf_upp+math.pow(max(0.0,pdfs[2*i+1]-cntrl_val,pdfs[2*i+2]-cntrl_val),2)
                    pdf_low=pdf_low+math.pow(max(0.0,cntrl_val-pdfs[2*i+1],cntrl_val-pdfs[2*i+2]),2)
                scale_pdf_info['pdf_upp'] = math.sqrt(pdf_upp)/cntrl_val*100
                scale_pdf_info['pdf_low'] = math.sqrt(pdf_low)/cntrl_val*100
        else:
            # use Gaussian method (NNPDF)
            pdf_stdev=0.0
            for i in range(int(numofpdf-1)):
                pdf_stdev = pdf_stdev + pow(pdfs[i+1] - cntrl_val,2)
            pdf_stdev = math.sqrt(pdf_stdev/int(numofpdf-2))
            scale_pdf_info['pdf_upp'] = pdf_stdev/cntrl_val*100
            scale_pdf_info['pdf_low'] = scale_pdf_info['pdf_upp']
        return scale_pdf_info


    def wait_for_complete(self, run_type):
        """this function waits for jobs on cluster to complete their run."""

        starttime = time.time()
        #logger.info('     Waiting for submitted jobs to complete')
        update_status = lambda i, r, f: self.update_status((i, r, f, run_type), 
                      starttime=starttime, level='parton', update_results=True)
        try:
            self.cluster.wait(self.me_dir, update_status)
        except:
            self.cluster.remove()
            raise

    def run_all(self, job_dict, arg_list, run_type='monitor'):
        """runs the jobs in job_dict (organized as folder: [job_list]), with arguments args"""
        self.njobs = sum(len(jobs) for jobs in job_dict.values()) * len(arg_list)
        self.ijob = 0
        if self.cluster_mode == 0:
            self.update_status((self.njobs - 1, 1, 0, run_type), level='parton')
        for args in arg_list:
            for Pdir, jobs in job_dict.items():
                for job in jobs:
                    self.run_exe(job, args, run_type, cwd=pjoin(self.me_dir, 'SubProcesses', Pdir) )
                    # print some statistics if running serially
        if self.cluster_mode == 2:
            time.sleep(1) # security to allow all jobs to be launched
        self.wait_for_complete(run_type)


    def run_exe(self, exe, args, run_type, cwd=None):
        """this basic function launch locally/on cluster exe with args as argument.
        """
        
        # first test that exe exists:
        execpath = None
        if cwd and os.path.exists(pjoin(cwd, exe)):
            execpath = pjoin(cwd, exe)
        elif not cwd and os.path.exists(exe):
            execpath = exe
        else:
            raise aMCatNLOError('Cannot find executable %s in %s' \
                % (exe, os.getcwd()))
        # check that the executable has exec permissions
        if self.cluster_mode == 1 and not os.access(execpath, os.X_OK):
            subprocess.call(['chmod', '+x', exe], cwd=cwd)
        # finally run it
        if self.cluster_mode == 0:
            #this is for the serial run
            misc.call(['./'+exe] + args, cwd=cwd)
            self.ijob += 1
            self.update_status((max([self.njobs - self.ijob - 1, 0]), 
                                min([1, self.njobs - self.ijob]),
                                self.ijob, run_type), level='parton')
        elif 'reweight' in exe:
                #Find the correct PDF input file
                input_files, output_files = [], []
                input_files.append(self.get_pdf_input_filename())
                input_files.append(pjoin(os.path.dirname(exe), os.path.pardir, 'reweight_xsec_events'))
                input_files.append(args[0])
                output_files.append('%s.rwgt' % os.path.basename(args[0]))
                output_files.append('reweight_xsec_events.output')
                output_files.append('scale_pdf_dependence.dat')
    
                return self.cluster.submit2(exe, args, cwd=cwd, 
                                 input_files=input_files, output_files=output_files) 

            #this is for the cluster/multicore run
        elif 'ajob' in exe:
            input_files, output_files, args = self.getIO_ajob(exe,cwd, args)
            #submitting
            self.cluster.submit2(exe, args, cwd=cwd, 
                         input_files=input_files, output_files=output_files)
        else:
            return self.cluster.submit(exe, args, cwd=cwd)

    def getIO_ajob(self,exe,cwd, args):
        # use local disk if possible => need to stands what are the 
        # input/output files
        
        keep_fourth_arg = False
        output_files = []
        input_files = [pjoin(self.me_dir, 'MGMEVersion.txt'),
                     pjoin(self.me_dir, 'SubProcesses', 'randinit'),
                     pjoin(cwd, 'symfact.dat'),
                     pjoin(cwd, 'iproc.dat'),
                     pjoin(cwd, 'FKS_params.dat')]
        
        if os.path.exists(pjoin(self.me_dir,'SubProcesses','OLE_order.olc')):
            input_files.append(pjoin(cwd, 'OLE_order.olc'))
      
        # File for the loop (might not be present if MadLoop is not used)
        if os.path.exists(pjoin(cwd, 'MadLoopParams.dat')):
            to_add = ['MadLoopParams.dat', 'ColorDenomFactors.dat', 
                                   'ColorNumFactors.dat','HelConfigs.dat']
            for name in to_add:
                input_files.append(pjoin(cwd, name))

                to_check = ['HelFilter.dat','LoopFilter.dat']
            for name in to_check:
                if os.path.exists(pjoin(cwd, name)):
                    input_files.append(pjoin(cwd, name))

        Ire = re.compile("for i in ([\d\s]*) ; do")
        try : 
            fsock = open(exe)
        except IOError:
            fsock = open(pjoin(cwd,exe))
        text = fsock.read()
        data = Ire.findall(text)
        subdir = ' '.join(data).split()
               
        if args[0] == '0':
            # MADEVENT VEGAS MODE
            input_files.append(pjoin(cwd, 'madevent_vegas'))
            input_files.append(pjoin(self.me_dir, 'SubProcesses','madin.%s' % args[1]))
            #j=$2\_G$i
            for i in subdir:
                current = '%s_G%s' % (args[1],i)
                if os.path.exists(pjoin(cwd,current)):
                    input_files.append(pjoin(cwd, current))
                output_files.append(current)
                if len(args) == 4:
                    # use a grid train on another part
                    base = '%s_G%s' % (args[3],i)
                    if args[0] == '0':
                        to_move = [n for n in os.listdir(pjoin(cwd, base)) 
                                                      if n.endswith('.sv1')]
                        to_move.append('grid.MC_integer')
                    elif args[0] == '1':
                        to_move = ['mint_grids', 'grid.MC_integer']
                    else: 
                        to_move  = []
                    if not os.path.exists(pjoin(cwd,current)):
                        os.mkdir(pjoin(cwd,current))
                        input_files.append(pjoin(cwd, current))
                    for name in to_move:
                        files.cp(pjoin(cwd,base, name), 
                                        pjoin(cwd,current))
                    files.cp(pjoin(cwd,base, 'grid.MC_integer'), 
                                        pjoin(cwd,current))
                            
        elif args[0] == '2':
            # MINTMC MODE
            input_files.append(pjoin(cwd, 'madevent_mintMC'))
            if args[2] in ['0','2']:
                input_files.append(pjoin(self.me_dir, 'SubProcesses','madinMMC_%s.2' % args[1]))

            for i in subdir:
                current = 'G%s%s' % (args[1], i)
                if os.path.exists(pjoin(cwd,current)):
                    input_files.append(pjoin(cwd, current))
                output_files.append(current)
                if len(args) == 4 and args[3] in ['H','S','V','B','F']:
                    # use a grid train on another part
                    base = '%s_%s' % (args[3],i)
                    files.ln(pjoin(cwd,base,'mint_grids'), name = 'preset_mint_grids', 
                                            starting_dir=pjoin(cwd,current))
                    files.ln(pjoin(cwd,base,'grid.MC_integer'), 
                                          starting_dir=pjoin(cwd,current))
                elif len(args) ==4:
                    keep_fourth_arg = True
               

        else:
            raise aMCatNLOError, 'not valid arguments: %s' %(', '.join(args))

        #Find the correct PDF input file
        input_files.append(self.get_pdf_input_filename())

        if len(args) == 4 and not keep_fourth_arg:
            args = args[:3]
            
        return input_files, output_files, args
            
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
1          ! Exact helicity sum (0 yes, n = number/event)?
1          ! Enter Configuration Number:
%1d          ! MINT imode: 0 to set-up grids, 1 to perform integral, 2 generate events
1 1 1      ! if imode is 1: Folding parameters for xi_i, phi_i and y_ij
%s        ! all, born, real, virt
""" \
                    % (mint_mode, run_mode)
        file = open(pjoin(path, 'madinMMC_%s.2' % name_suffix), 'w')
        file.write(content)
        file.close()

    def write_madin_file(self, path, run_mode, vegas_mode, npoints, niters):
        """writes the madin.run_mode file"""
        #check the validity of the arguments
        run_modes = ['born', 'virt', 'novi', 'all', 'viSB', 'novB', 'grid']
        if run_mode not in run_modes:
            raise aMCatNLOError('%s is not a valid mode for run. Please use one of the following: %s' \
                    % (run_mode, ', '.join(run_modes)))
        name_suffix = run_mode

        content = \
"""%s %s  ! points, iterations
0 ! accuracy
2 ! 0 fixed grid 2 adjust
1 ! 1 suppress amp, 0 doesnt
1 ! 0 for exact hel sum
1 ! hel configuration numb
'test'
1 ! 1 to save grids
%s ! 0 to exclude, 1 for new run, 2 to restart, 3 to reset w/ keeping grid
%s        ! all, born, real, virt
""" \
                    % (npoints,niters,vegas_mode,run_mode)
        file = open(pjoin(path, 'madin.%s' % name_suffix), 'w')
        file.write(content)
        file.close()

    def compile(self, mode, options):
        """compiles aMC@NLO to compute either NLO or NLO matched to shower, as
        specified in mode"""

        #define a bunch of log files
        amcatnlo_log = pjoin(self.me_dir, 'compile_amcatnlo.log')
        madloop_log = pjoin(self.me_dir, 'compile_madloop.log')
        reweight_log = pjoin(self.me_dir, 'compile_reweight.log')
        test_log = pjoin(self.me_dir, 'test.log')

        self.update_status('Compiling the code', level=None, update_results=True)


        libdir = pjoin(self.me_dir, 'lib')
        sourcedir = pjoin(self.me_dir, 'Source')

        #clean files
        misc.call(['rm -f %s' % 
                ' '.join([amcatnlo_log, madloop_log, reweight_log, test_log])], \
                  cwd=self.me_dir, shell=True)

        #define which executable/tests to compile
        if '+' in mode:
            mode = mode.split('+')[0]
        if mode in ['NLO', 'LO']:
            exe = 'madevent_vegas'
            tests = ['test_ME']
        elif mode in ['aMC@NLO', 'aMC@LO','noshower','noshowerLO']:
            exe = 'madevent_mintMC'
            tests = ['test_ME', 'test_MC']

        #directory where to compile exe
        p_dirs = [file for file in os.listdir(pjoin(self.me_dir, 'SubProcesses')) 
                    if file.startswith('P') and \
                    os.path.isdir(pjoin(self.me_dir, 'SubProcesses', file))]
        # create param_card.inc and run_card.inc
        self.do_treatcards('', amcatnlo=True)
        # if --nocompile option is specified, check here that all exes exists. 
        # If they exists, return
        if all([os.path.exists(pjoin(self.me_dir, 'SubProcesses', p_dir, exe)) \
                for p_dir in p_dirs]) and options['nocompile']:
            return

        # rm links to lhapdflib/ PDFsets if exist
        if os.path.islink(pjoin(libdir, 'libLHAPDF.a')):
            os.remove(pjoin(libdir, 'libLHAPDF.a'))
        if os.path.islink(pjoin(libdir, 'PDFsets')):
            os.remove(pjoin(libdir, 'PDFsets'))

        # read the run_card to find if lhapdf is used or not
        if self.run_card['pdlabel'] == 'lhapdf':
            self.link_lhapdf(libdir)
        else:
            if self.run_card['lpp1'] == '1' ==self.run_card['lpp2']:
                logger.info('Using built-in libraries for PDFs')
            try:
                del os.environ['lhapdf']
            except KeyError:
                pass


        os.environ['fastjet_config'] = self.options['fastjet']
        
        # make Source
        self.update_status('Compiling source...', level=None)
        misc.compile(['clean4pdf'], cwd = sourcedir)
        misc.compile(cwd = sourcedir)
        if os.path.exists(pjoin(libdir, 'libdhelas.a')) \
          and os.path.exists(pjoin(libdir, 'libgeneric.a')) \
          and os.path.exists(pjoin(libdir, 'libmodel.a')) \
          and os.path.exists(pjoin(libdir, 'libpdf.a')):
            logger.info('          ...done, continuing with P* directories')
        else:
            raise aMCatNLOError('Compilation failed')

        # check if virtuals have been generated
        proc_card = open(pjoin(self.me_dir, 'Cards', 'proc_card_mg5.dat')).read()
        if not '[real=QCD]' in proc_card and \
                          not os.path.exists(pjoin(self.me_dir,'OLP_virtuals')):
            os.environ['madloop'] = 'true'
            if mode in ['NLO', 'aMC@NLO', 'noshower']:
                tests.append('check_poles')
                hasvirt = True
        else:
            os.unsetenv('madloop')

        # make and run tests (if asked for), gensym and make madevent in each dir
        self.update_status('Compiling directories...', level=None)

        for test in tests:
            self.write_test_input(test)

        try:
            import multiprocessing
            if not self.nb_core:
                try:
                    self.nb_core = int(self.options['nb_core'])
                except TypeError:
                    self.nb_core = multiprocessing.cpu_count()

            mypool = multiprocessing.Pool(self.nb_core, init_worker) 
            logger.info('Compiling on %d cores' % self.nb_core)
            mypool.map(compile_dir,
                    ((self.me_dir, p_dir, mode, options, 
                        tests, exe, self.options['run_mode']) for p_dir in p_dirs))
            time.sleep(1) # sleep one second to make sure all ajob* files are written
            mypool.terminate() # kill all the members of the multiprocessing pool
        except ImportError: 
            self.nb_core = 1
            logger.info('Multiprocessing module not found. Compiling on 1 core')
            for p_dir in p_dirs:
                compile_dir(self.me_dir, p_dir, mode, options, 
                        tests, exe, self.options['run_mode'])

        logger.info('Checking test output:')
        for p_dir in p_dirs:
            logger.info(p_dir)
            for test in tests:
                logger.info(' Result for %s:' % test)

                this_dir = pjoin(self.me_dir, 'SubProcesses', p_dir) 
                #check that none of the tests failed
                self.check_tests(test, this_dir)

        os.unsetenv('madloop')




    def check_tests(self, test, dir):
        """just call the correct parser for the test log"""
        if test in ['test_ME', 'test_MC']:
            return self.parse_test_mx_log(pjoin(dir, '%s.log' % test)) 
        elif test == 'check_poles':
            return self.parse_check_poles_log(pjoin(dir, '%s.log' % test)) 


    def parse_test_mx_log(self, log):
        """read and parse the test_ME/MC.log file"""
        content = open(log).read()
        if 'FAILED' in content:
            logger.info('Output of the failing test:\n'+content[:-1],'$MG:color:BLACK')
            raise aMCatNLOError('Some tests failed, run cannot continue.\n' + \
                'Please check that widths of final state particles (e.g. top) have been' + \
                ' set to 0 in the param_card.dat.')
        else:
            lines = [l for l in content.split('\n') if 'PASSED' in l]
            logger.info('   Passed.')
            logger.debug('\n'+'\n'.join(lines))


    def parse_check_poles_log(self, log):
        """reads and parse the check_poles.log file"""
        content = open(log).read()
        npass = 0
        nfail = 0
        for line in content.split('\n'):
            if 'PASSED' in line:
                npass +=1
                tolerance = float(line.split()[1])
            if 'FAILED' in line:
                nfail +=1
                tolerance = float(line.split()[1])

        if nfail + npass == 0:
            logger.warning('0 points have been tried')
            return

        if float(nfail)/float(nfail+npass) > 0.1:
            raise aMCatNLOError('Poles do not cancel, run cannot continue')
        else:
            logger.info('   Poles successfully cancel for %d points over %d (tolerance=%2.1e)' \
                    %(npass, nfail+npass, tolerance))



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
        os.environ['lhapdf'] = 'True'
        os.environ['lhapdf_config'] = self.options['lhapdf']


    def write_test_input(self, test):
        """write the input files to run test_ME/MC or check_poles"""
        if test in ['test_ME', 'test_MC']:
            content = "-2 -2\n" #generate randomly energy/angle
            content+= "100 100\n" #run 100 points for soft and collinear tests
            content+= "0\n" #sum over helicities
            content+= "0\n" #all FKS configs
            content+= '\n'.join(["-1"] * 50) #random diagram
        elif test == 'check_poles':
            content = '20 \n -1\n'
        
        file = open(pjoin(self.me_dir, '%s_input.txt' % test), 'w')
        if test == 'test_MC':
            shower = self.run_card['parton_shower']
            MC_header = "%s\n " % shower + \
                        "1 \n1 -0.1\n-1 -0.1\n"
            file.write(MC_header + content)
        else:
            file.write(content)
        file.close()



    ############################################################################
    def find_model_name(self):
        """ return the model name """
        if hasattr(self, 'model_name'):
            return self.model_name
        
        model = 'sm'
        proc = []
        for line in open(os.path.join(self.me_dir,'Cards','proc_card_mg5.dat')):
            line = line.split('#')[0]
            #line = line.split('=')[0]
            if line.startswith('import') and 'model' in line:
                model = line.split()[2]   
                proc = []
            elif line.startswith('generate'):
                proc.append(line.split(None,1)[1])
            elif line.startswith('add process'):
                proc.append(line.split(None,2)[2])
       
        self.model = model
        self.process = proc 
        return model



    ############################################################################
    def ask_run_configuration(self, mode, options):
        """Ask the question when launching generate_events/multi_run"""
        
        if 'parton' not in options:
            options['parton'] = False
        if 'reweightonly' not in options:
            options['reweightonly'] = False
        
        
        void = 'NOT INSTALLED'
        switch_order = ['order', 'fixed_order', 'shower','madspin']
        switch = {'order': 'NLO', 'fixed_order': 'OFF', 'shower': void,
                  'madspin': void}
        default_switch = ['ON', 'OFF']
        allowed_switch_value = {'order': ['LO', 'NLO'],
                                'fixed_order': default_switch,
                                'shower': default_switch,
                                'madspin': default_switch}
        
        description = {'order':  'Perturbative order of the calculation:',
                       'fixed_order': 'Fixed order (no event generation and no MC@[N]LO matching):',
                       'shower': 'Shower the generated events:',
                       'madspin': 'Decay particles with the MadSpin module:' }

        force_switch = {('shower', 'ON'): {'fixed_order': 'OFF'},
                       ('madspin', 'ON'): {'fixed_order':'OFF'},
                       ('fixed_order', 'ON'): {'shower': 'OFF', 'madspin': 'OFF'}
                       }
        special_values = ['LO', 'NLO', 'aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO']

        assign_switch = lambda key, value: switch.__setitem__(key, value if switch[key] != void else void )
        

        if mode == 'auto': 
            mode = None
        if not mode and (options['parton'] or options['reweightonly']):
            mode = 'noshower'         
        
        # Init the switch value according to the current status
        available_mode = ['0', '1', '2']
        available_mode.append('3')
        if os.path.exists(pjoin(self.me_dir, 'Cards', 'shower_card.dat')):
            switch['shower'] = 'ON'
        else:
            switch['shower'] = 'OFF'
                
        if not aMCatNLO or self.options['mg5_path']:
            available_mode.append('4')
            if os.path.exists(pjoin(self.me_dir,'Cards','madspin_card.dat')):
                switch['madspin'] = 'ON'
            else:
                switch['madspin'] = 'OFF'
            
        answers = list(available_mode) + ['auto', 'done']
        alias = {}
        for id, key in enumerate(switch_order):
            if switch[key] != void:
                answers += ['%s=%s' % (key, s) for s in allowed_switch_value[key]]
                #allow lower case for on/off
                alias.update(dict(('%s=%s' % (key, s.lower()), '%s=%s' % (key, s))
                                   for s in allowed_switch_value[key]))
        answers += special_values
        
        def create_question(switch):
            switch_format = " %i %-60s %12s=%s\n"
            question = "The following switches determine which operations are executed:\n"
            for id, key in enumerate(switch_order):
                question += switch_format % (id+1, description[key], key, switch[key])
            question += '  Either type the switch number (1 to %s) to change its default setting,\n' % (id+1)
            question += '  or set any switch explicitly (e.g. type \'order=LO\' at the prompt)\n'
            question += '  Type \'0\', \'auto\', \'done\' or just press enter when you are done.\n'
            return question

        if not self.force:
            answer = ''
            while answer not in ['0', 'done', 'auto', 'onlyshower']:
                question = create_question(switch)
                if mode:
                    answer = mode
                else:
                    answer = self.ask(question, '0', answers, alias=alias)
                if answer.isdigit() and answer != '0':
                    key = switch_order[int(answer) - 1]
                    opt1 = allowed_switch_value[key][0]
                    opt2 = allowed_switch_value[key][1]
                    answer = '%s=%s' % (key, opt1 if switch[key] == opt2 else opt2)
                if '=' in answer:
                    key, status = answer.split('=')
                    switch[key] = status
                    if (key, status) in force_switch:
                        for key2, status2 in force_switch[(key, status)].items():
                            if switch[key2] not in  [status2, void]:
                                logger.info('For coherence \'%s\' is set to \'%s\''
                                            % (key2, status2), '$MG:color:BLACK')
                                switch[key2] = status2
                elif answer in ['0', 'auto', 'done']:
                    break
                elif answer in special_values:
                    logger.info('Enter mode value: Go to the related mode', '$MG:color:BLACK')
                    if answer == 'LO':
                        switch['order'] = 'LO'
                        switch['fixed_order'] = 'ON'
                        assign_switch('shower', 'OFF')
                        assign_switch('madspin', 'OFF')
                    elif answer == 'NLO':
                        switch['order'] = 'NLO'
                        switch['fixed_order'] = 'ON'
                        assign_switch('shower', 'OFF')
                        assign_switch('madspin', 'OFF')
                    elif answer == 'aMC@NLO':
                        switch['order'] = 'NLO'
                        switch['fixed_order'] = 'OFF'
                        assign_switch('shower', 'ON')
                        assign_switch('madspin', 'OFF')
                    elif answer == 'aMC@LO':
                        switch['order'] = 'LO'
                        switch['fixed_order'] = 'OFF'
                        assign_switch('shower', 'ON')
                        assign_switch('madspin', 'OFF')
                    elif answer == 'noshower':
                        switch['order'] = 'NLO'
                        switch['fixed_order'] = 'OFF'
                        assign_switch('shower', 'OFF')
                        assign_switch('madspin', 'OFF')                                                    
                    elif answer == 'noshowerLO':
                        switch['order'] = 'LO'
                        switch['fixed_order'] = 'OFF'
                        assign_switch('shower', 'OFF')
                        assign_switch('madspin', 'OFF')
                    if mode:
                        break

        #assign the mode depending of the switch
        if not mode or mode == 'auto':
            if switch['order'] == 'LO':
                if switch['shower'] == 'ON':
                    mode = 'aMC@LO'
                elif switch['fixed_order'] == 'ON':
                    mode = 'LO'
                else:
                    mode =  'noshowerLO'
            elif switch['order'] == 'NLO':
                if switch['shower'] == 'ON':
                    mode = 'aMC@NLO'
                elif switch['fixed_order'] == 'ON':
                    mode = 'NLO'
                else:
                    mode =  'noshower'  
        logger.info('will run in mode: %s' % mode)                

        if mode == 'noshower':
            logger.warning("""You have chosen not to run a parton shower. NLO events without showering are NOT physical.
Please, shower the Les Houches events before using them for physics analyses.""")            
            
        
        # specify the cards which are needed for this run.
        cards = ['param_card.dat', 'run_card.dat']
        if mode in ['LO', 'NLO']:
            options['parton'] = True
        elif switch['madspin'] == 'ON':
            cards.append('madspin_card.dat')
        if 'aMC@' in mode:
            cards.append('shower_card.dat')
        if mode == 'onlyshower':
            cards = ['shower_card.dat']
        if options['reweightonly']:
            cards = ['run_card.dat']

        self.keep_cards(cards)
        
        if mode =='onlyshower':
            cards = ['shower_card.dat']
        
        if not options['force'] and not  self.force:
            self.ask_edit_cards(cards, plot=False)
            


        run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
        self.run_card = banner_mod.RunCardNLO(run_card)
        self.run_tag = self.run_card['run_tag']
        self.run_name = self.find_available_run_name(self.me_dir)
        #add a tag in the run_name for distinguish run_type
        if self.run_name.startswith('run_'):
            if mode in ['LO','aMC@LO','noshowerLO']:
                self.run_name += '_LO' 
        self.set_run_name(self.run_name, self.run_tag, 'parton')
        if 'aMC@' in mode or mode == 'onlyshower':
            shower_card_path = pjoin(self.me_dir, 'Cards','shower_card.dat')
            self.shower_card = shower_card.ShowerCard(shower_card_path)
        
        return mode


    def do_quit(self, line):
        """ """
        try:
            os.remove(pjoin(self.me_dir,'RunWeb'))
        except Exception:
            pass
        try:
            self.store_result()
        except:
            # If nothing runs they they are no result to update
            pass
        try:
            self.update_status('', level=None)
        except Exception, error:         
            pass
        devnull = os.open(os.devnull, os.O_RDWR) 
        try:
            misc.call(['./bin/internal/gen_cardhtml-pl'], cwd=self.me_dir,
                        stdout=devnull, stderr=devnull)
        except Exception:
            pass

        return super(aMCatNLOCmd, self).do_quit(line)
    
    # Aliases
    do_EOF = do_quit
    do_exit = do_quit




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
_compile_parser = misc.OptionParser(usage=_compile_usage)
_compile_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory for the launch, without editing them")

_launch_usage = "launch [MODE] [options]\n" + \
                "-- execute aMC@NLO \n" + \
                "   MODE can be either LO, NLO, aMC@NLO or aMC@LO (if omitted, it is asked in a separate question)\n" + \
                "     If mode is set to LO/NLO, no event generation will be performed, but only the \n" + \
                "     computation of the total cross-section and the filling of parton-level histograms \n" + \
                "     specified in the DIRPATH/SubProcesses/madfks_plot.f file.\n" + \
                "     If mode is set to aMC@LO/aMC@NLO, after the cross-section computation, a .lhe \n" + \
                "     event file is generated which will be showered with the MonteCarlo specified \n" + \
                "     in the run_card.dat\n"

_launch_parser = misc.OptionParser(usage=_launch_usage)
_launch_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory for the launch, without editing them")
_launch_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_launch_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_launch_parser.add_option("-n", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found")
_launch_parser.add_option("-r", "--reweightonly", default=False, action='store_true',
                            help="Skip integration and event generation, just run reweight on the" + \
                                 " latest generated event files (see list in SubProcesses/nevents_unweighted)")
_launch_parser.add_option("-p", "--parton", default=False, action='store_true',
                            help="Stop the run after the parton level file generation (you need " + \
                                    "to shower the file in order to get physical results)")
_launch_parser.add_option("-o", "--only_generation", default=False, action='store_true',
                            help="Skip grid set up, just generate events starting from " + \
                            "the last available results")


_calculate_xsect_usage = "calculate_xsect [ORDER] [options]\n" + \
                "-- calculate cross-section up to ORDER.\n" + \
                "   ORDER can be either LO or NLO (if omitted, it is set to NLO). \n"

_calculate_xsect_parser = misc.OptionParser(usage=_calculate_xsect_usage)
_calculate_xsect_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory for the launch, without editing them")
_calculate_xsect_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_calculate_xsect_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_calculate_xsect_parser.add_option("-n", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found, " + \
                            "or with --tests")

_shower_usage = 'shower run_name [options]\n' + \
        '-- do shower/hadronization on parton-level file generated for run run_name\n' + \
        '   all the information (e.g. number of events, MonteCarlo, ...\n' + \
        '   are directly read from the header of the event file\n'
_shower_parser = misc.OptionParser(usage=_shower_usage)
_shower_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the shower_card present in the directory for the launch, without editing")


_generate_events_usage = "generate_events [ORDER] [options]\n" + \
                "-- generate events to be showered, corresponding to a cross-section computed up to ORDER.\n" + \
                "   ORDER can be either LO or NLO (if omitted, it is set to NLO). \n" + \
                "   The number of events and the specific parton shower MC can be specified \n" + \
                "   in the run_card.dat\n"

_generate_events_parser = misc.OptionParser(usage=_generate_events_usage)
_generate_events_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory for the launch, without editing them")
_generate_events_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_generate_events_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_generate_events_parser.add_option("-n", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found, " + \
                            "or with --tests")
_generate_events_parser.add_option("-o", "--only-generation", default=False, action='store_true',
                            help="Skip grid set up, just generate events starting from" + \
                            "the last available results")
_generate_events_parser.add_option("-p", "--parton", default=False, action='store_true',
                            help="Stop the run after the parton level file generation (you need " + \
                                    "to shower the file in order to get physical results)")
