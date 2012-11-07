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
    import internal.shower_card as shower_card
    aMCatNLO = True




class aMCatNLOError(Exception):
    pass


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

    def complete_launch(self, text, line, begidx, endidx):
        """auto-completion for launch command"""
        
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return mode
            return self.list_completion(text,['LO','NLO','aMC@NLO', 'aMC@LO'],line)
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
            data = glob.glob(pjoin(self.me_dir, 'Events', '*','events.lhe.gz'))
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
    
    # The three options categories are treated on a different footage when a 
    # set/save configuration occur. current value are kept in self.options
    options_configuration = {'pythia8_path': './pythia8',
                       'madanalysis_path': './MadAnalysis',
                       'pythia-pgs_path':'./pythia-pgs',
                       'td_path':'./td',
                       'delphes_path':'./Delphes',
                       'exrootanalysis_path':'./ExRootAnalysis',
                       'MCatNLO-utilities_path':'./MCatNLO-utilities',
                       'timeout': 60,
                       'web_browser':None,
                       'eps_viewer':None,
                       'text_editor':None,
                       'fortran_compiler':None,
                       'auto_update':7,
                       'cluster_type': 'condor'}
    
    options_madgraph= {'stdout_level':None}
    
    options_madevent = {'automatic_html_opening':True,
                         'run_mode':2,
                         'cluster_queue':'madgraph',
                         'nb_core': None,
                         'cluster_temp_path':None}
    
    
    ############################################################################
    def __init__(self, me_dir = None, options = {}, *completekey, **stdin):
        """ add information to the cmd """

        self.start_time = 0
        CmdExtended.__init__(self, *completekey, **stdin)
        common_run.CommonRunCmd.__init__(self, me_dir, options)
       
        run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
        self.run_card = banner_mod.RunCardNLO(run_card)

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
        self.check_shower(argss, options)
        options['parton'] = 'onlyshower'
        evt_file = pjoin(os.getcwd(), argss[0], 'events.lhe')
        self.ask_run_configuration('', options)
        if self.check_mcatnlo_dir():
            self.run_mcatnlo(evt_file)
        os.chdir(root_path)

    ################################################################################
    def do_plot(self, line):
        """Create the plot for a given run"""

        # Since in principle, all plot are already done automaticaly
        args = self.split_arg(line)
        # Check argument's validity
        self.check_plot(args)
        logger.info('plot for run %s' % self.run_name)
        
        self.ask_edit_cards([], args)
                
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
        options['noreweight'] = False
        options['parton'] = True
        self.check_calculate_xsect(argss, options)
        
        if options['multicore']:
            self.cluster_mode = 2
        elif options['cluster']:
            self.cluster_mode = 1

#        if self.options_madevent['automatic_html_opening']:
#            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
#            self.options_madevent['automatic_html_opening'] = False

        
        mode = argss[0]
        self.ask_run_configuration(mode, options)
        self.compile(mode, options) 
        self.run(mode, options)
        os.chdir(root_path)

        
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

#        if self.options_madevent['automatic_html_opening']:
#            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
#            self.options_madevent['automatic_html_opening'] = False

        mode = 'aMC@' + argss[0]
        self.ask_run_configuration(mode, options)
        self.compile(mode, options) 
        evt_file = self.run(mode, options)
        if self.check_mcatnlo_dir() and not options['parton']:
            self.run_mcatnlo(evt_file)
        os.chdir(root_path)

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

#        if self.options['automatic_html_opening']:
#            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
#            self.options['automatic_html_opening'] = False

        mode = argss[0]
        if mode in ['LO', 'NLO']:
            options['parton'] = True
        self.ask_run_configuration(mode, options)
        self.compile(mode, options) 
        evt_file = self.run(mode, options)
        if self.check_mcatnlo_dir() and not options['parton']:
            self.run_mcatnlo(evt_file)
        os.chdir(root_path)


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
        os.chdir(root_path)

    def check_mcatnlo_dir(self):
        """Check that the MCatNLO dir (with files to run the parton-shower has 
        been copied inside the exported direcotry"""
        if os.path.isdir(pjoin(self.me_dir, 'MCatNLO')):
            #the folder has been exported after installation of MCatNLO-utilities
            return True
        elif self.options['MCatNLO-utilities_path']:
            # if the option is not none, the path should already exist
            # the folder has been exported before installation of MCatNLO-utilities
            # and they have been installed
            misc.call(['cp -r %s %s' % \
                     (pjoin(self.options['MCatNLO-utilities_path'], 'MCatNLO'), self.me_dir)],
                     shell=True)
            return True

        else:
            logger.warning('MCatNLO is needed to shower the generated event samples.\n' + \
                'You can install it by typing "install MCatNLO-utilities" in the MadGraph' + \
                ' shell')
            return False

    def update_status(self, status, level, makehtml=False, force=True, 
                      error=False, starttime = None, update_results=False):
        
        common_run.CommonRunCmd.update_status(self, status, level, makehtml, 
                                        force, error, starttime, update_results)


    def update_random_seed(self):
        """Update random number seed with the value from the run_card. 
        If this is 0, update the number according to a fresh one"""
        iseed = int(self.run_card['iseed'])
        if iseed != 0:
            misc.call(['echo "r=%d > %s"' \
                    % (iseed, pjoin(self.me_dir, 'SubProcesses', 'randinit'))],
                    cwd=self.me_dir, shell=True)
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

        if not 'only_generation' in options.keys():
            options['only_generation'] = False

        os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))
        old_cwd = os.getcwd()

        if self.cluster_mode == 1:
            cluster_name = self.options['cluster_type']
            self.cluster = cluster.from_name[cluster_name](self.options['cluster_queue'],
                                              self.options['cluster_temp_path'])
        if self.cluster_mode == 2:
            import multiprocessing
            try:
                self.nb_core = int(self.options['nb_core'])
            except TypeError:
                self.nb_core = multiprocessing.cpu_count()
            logger.info('Using %d cores' % self.nb_core)
            self.cluster = cluster.MultiCore(self.nb_core, 
                                     temp_dir=self.options['cluster_temp_path'])
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
            nevents = self.run_card['nevents']
            self.reweight_and_collect_events(options, mode, nevents)
            os.chdir(old_cwd)
            return

        devnull = os.open(os.devnull, os.O_RDWR) 
        if mode == 'LO':
            logger.info('Doing fixed order LO')
            self.update_status('Cleaning previous results', level=None)
            misc.call(['rm -rf P*/born_G*'], shell=True)

            self.update_status('Computing cross-section', level=None)
            self.run_all(job_dict, [['0', 'born', '0']], 'Computing cross-section')
            p = misc.Popen(['./combine_results_FO.sh born_G* '], \
                                stdout=subprocess.PIPE, shell=True)
            output = p.communicate()
            self.cross_sect_dict = self.read_results(output, mode)
            self.print_summary(1, mode)
            misc.call(['./combine_plots_FO.sh born_G*'], stdout=devnull, shell=True)
            misc.call(['cp MADatNLO.top res.txt %s' % \
                    pjoin(self.me_dir, 'Events', self.run_name)], shell=True)
            logger.info('The results of this run and the TopDrawer file with the plots' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))
            os.chdir(old_cwd)
            return

        if mode == 'NLO':
            self.update_status('Doing fixed order NLO', level=None)
            logger.info('   Cleaning previous results')
            misc.call(['rm -rf P*/grid_G* P*/novB_G* P*/viSB_G*'], shell=True)

            self.update_status('Setting up grid', level=None)
            self.run_all(job_dict, [['0', 'grid', '0']], 'Setting up grid')
            p = misc.Popen(['./combine_results_FO.sh grid*'], \
                                stdout=subprocess.PIPE, shell=True)
            output = p.communicate()
            self.cross_sect_dict = self.read_results(output, mode)
            self.print_summary(0, mode)

            self.update_status('Computing cross-section', level=None)
            self.run_all(job_dict, [['0', 'viSB', '0', 'grid'], ['0', 'novB', '0', 'grid']], \
                    'Computing cross-section')
            p = misc.Popen(['./combine_results_FO.sh viSB* novB*'], \
                                stdout=subprocess.PIPE, shell=True)
            output = p.communicate()
            self.cross_sect_dict = self.read_results(output, mode)
            self.print_summary(1, mode)

            misc.call(['./combine_plots_FO.sh viSB* novB*'], stdout=devnull, shell=True)
            misc.call(['cp MADatNLO.top res.txt %s' % \
                    pjoin(self.me_dir, 'Events', self.run_name)], shell=True)
            logger.info('The results of this run and the TopDrawer file with the plots' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))
            os.chdir(old_cwd)
            return

        elif mode in ['aMC@NLO', 'aMC@LO']:
            shower = self.run_card['parton_shower']
            nevents = int(self.run_card['nevents'])
            #shower_list = ['HERWIG6', 'HERWIGPP', 'PYTHIA6Q', 'PYTHIA6PT', 'PYTHIA8']
            shower_list = ['HERWIG6', 'HERWIGPP', 'PYTHIA6Q']
            if not shower in shower_list:
                raise aMCatNLOError('%s is not a valid parton shower. Please use one of the following: %s' \
                    % (shower, ', '.join(shower_list)))

            if mode == 'aMC@NLO':
                if not options['only_generation']:
                    logger.info('Doing NLO matched to parton shower')
                    logger.info('   Cleaning previous results')
                    misc.call(['rm -rf P*/GF* P*/GV*'], shell=True)

                for i, status in enumerate(mcatnlo_status):
                    if i == 2 or not options['only_generation']:
                        self.update_status(status, level='parton')
                        self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), 'novB', i) 
                        self.write_madinMMC_file(pjoin(self.me_dir, 'SubProcesses'), 'viSB', i) 
                        self.run_all(job_dict, [['2', 'V', '%d' % i], ['2', 'F', '%d' % i]], status)

                    if (i < 2 and not options['only_generation'])  or i == 1 :
                        p = misc.Popen(['./combine_results.sh %d %d GF* GV*' % (i, nevents)],
                                stdout=subprocess.PIPE, shell=True)
                        output = p.communicate()
                        misc.call(['cp res_%d_abs.txt res_%d_tot.txt %s' % \
                           (i, i, pjoin(self.me_dir, 'Events', self.run_name))], shell=True)
                        self.cross_sect_dict = self.read_results(output, mode)
                        self.print_summary(i, mode)

            elif mode == 'aMC@LO':
                if not options['only_generation']:
                    logger.info('Doing LO matched to parton shower')
                    logger.info('   Cleaning previous results')
                    misc.call(['rm -rf P*/GB*'], shell=True)
                for i, status in enumerate(mcatnlo_status):
                    if i == 2 or not options['only_generation']:
                        self.update_status('%s at LO' % status, level='parton')
                        self.write_madinMMC_file(
                                pjoin(self.me_dir, 'SubProcesses'), 'born', i) 
                        self.run_all(job_dict, [['2', 'B', '%d' % i]], '%s at LO' % status)

                    if (i < 2 and not options['only_generation'])  or i == 1 :
                        p = misc.Popen(['./combine_results.sh %d %d GB*' % (i, nevents)],
                                stdout=subprocess.PIPE, shell=True)
                        output = p.communicate()
                        misc.call(['cp res_%d_abs.txt res_%d_tot.txt %s' % \
                           (i, i, pjoin(self.me_dir, 'Events', self.run_name))], shell=True)
                        self.cross_sect_dict = self.read_results(output, mode)
                        self.print_summary(i, mode)

        if self.cluster_mode == 1:
            #if cluster run, wait 15 sec so that event files are transferred back
            self.update_status(
                    'Waiting while files are transferred back from the cluster nodes',
                    level='parton')
            time.sleep(10)

        # chancge back to the original pwd
        os.chdir(old_cwd)

        return self.reweight_and_collect_events(options, mode, nevents)

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
        if mode in ['aMC@LO', 'aMC@NLO']:
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
        if mode in ['aMC@LO', 'aMC@NLO']:
            return {'randinit' : int(match.groups()[1]),
                    'xseca' : float(match.groups()[2]),
                    'erra' : float(match.groups()[3]),
                    'xsect' : float(match.groups()[6]),
                    'errt' : float(match.groups()[7])}
        else:
            return {'xsect' : float(match.groups()[1]),
                    'errt' : float(match.groups()[2])}

    def print_summary(self, step, mode):
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
        if mode in ['aMC@NLO', 'aMC@LO']: 
            log_GV_files =  glob.glob(pjoin(self.me_dir, \
                                    'SubProcesses', 'P*','GV*','log_MINT*.txt'))
            all_log_files = glob.glob(pjoin(self.me_dir, \
                                          'SubProcesses', 'P*','G*','log*.txt'))
        elif mode in ['NLO', 'LO']:
            log_GV_files =  glob.glob(pjoin(self.me_dir, \
                                    'SubProcesses', 'P*','viSB_G*','log*.txt'))
            all_log_files = sum([glob.glob(pjoin(self.me_dir,'SubProcesses', 'P*',
              '%sG*'%foldName,'log*.txt')) for foldName in ['grid_','novB_',\
                                                                   'viSB_']],[])
        else:
            raise aMCatNLOError, 'Running mode %s not supported.'%mode
        # Recuperate the fraction of unstable PS points found in the runs for the
        # virtuals
        UPS_stat_finder = re.compile(r".*Total points tried\:\s+(?P<nPS>\d+).*"+\
                      r"Unstable points \(check UPS\.log for the first 10\:\)"+\
                                                r"\s+(?P<nUPS>\d+).*",re.DOTALL)
        for gv_log in log_GV_files:
            logfile=open(gv_log,'r')             
            UPS_stats = re.search(UPS_stat_finder,logfile.read())
            logfile.close()
            if not UPS_stats is None:
                channel_name = '/'.join(gv_log.split('/')[-5:-1])
                try:
                    stats['UPS'][channel_name][0] += int(UPS_stats.group('nPS'))
                    stats['UPS'][channel_name][1] += int(UPS_stats.group('nUPS'))
                except KeyError:
                    stats['UPS'][channel_name] = [int(UPS_stats.group('nPS')),
                                                   int(UPS_stats.group('nUPS'))]
        # Find the number of potential errors found in all log files
        for log in all_log_files:
            logfile=open(log,'r')
            nErrors = len(re.findall(re.compile(r"\bERROR\b",\
                                                re.IGNORECASE), logfile.read()))
            logfile.close()
            if nErrors != 0:
                stats['Errors'].append((str(log),nErrors))
            
        
        if mode in ['aMC@NLO', 'aMC@LO']:
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
            status = ['Results after grid setup (correspond roughly to LO):',
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
           (mode in ['aMC@NLO', 'aMC@LO'] and step!=2):
            logger.info(message+'\n')
            return

        # Now display the general statistics
        debug_msg = ""
        if len(stats['UPS'].keys())>0:
            nTotUPS = sum([chan[1] for chan in stats['UPS'].values()],0)
            nTotPS  = sum([chan[0] for chan in stats['UPS'].values()],0)
            UPSfracs = [(chan[0] , 0.0 if chan[1][0]==0 else \
                 float(chan[1][1]*100)/chan[1][0]) for chan in stats['UPS'].items()]
            maxUPS = max(UPSfracs, key = lambda w: w[1])
            if maxUPS[1]>0.1:
                message += '\n      Number of loop ME evaluations: %d'%nTotPS
                message += '\n      Total number of unstable PS point detected:'+\
                                 ' %d (%4.2f%%)'%(nTotUPS,float(100*nTotUPS)/nTotPS)
                message += '\n      Maximum fraction of UPS points in '+\
                          'channel %s (%4.2f%%)'%maxUPS
                message += '\n      Please report this to the authors while '+\
                                                                'providing the file'
                message += '\n      %s'%str(pjoin(os.path.dirname(self.me_dir),
                                                               maxUPS[0],'UPS.log'))
            else:
                debug_msg += '\n      Number of loop ME evaluations: %d'%nTotPS
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




    def reweight_and_collect_events(self, options, mode, nevents):
        """this function calls the reweighting routines and creates the event file in the 
        Event dir. Return the name of the event file created
        """
        if not options['noreweight']:
            self.run_reweight(options['reweightonly'])

        self.update_status('Collecting events', level='parton')
        misc.call(['make collect_events > %s' % \
                pjoin(self.me_dir, 'log_collect_events.txt')], shell=True)
        misc.call(['echo "1" | ./collect_events > %s' % \
                pjoin(self.me_dir, 'log_collect_events.txt')], shell=True)

        #get filename from collect events
        filename = open(pjoin(self.me_dir, 'log_collect_events.txt')).read().split()[-1]

        if not os.path.exists(pjoin(self.me_dir, 'SubProcesses', filename)):
            raise aMCatNLOError('An error occurred during event generation. ' + \
                    'The event file has not been created. Check log_collect_events.txt')
        evt_file = pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe')
        misc.call(['mv %s %s' % 
            (pjoin(self.me_dir, 'SubProcesses', filename), evt_file)], shell=True )
        misc.call(['gzip %s' % evt_file], shell=True)
        self.print_summary(2, mode)
        logger.info('The %s.gz file has been generated.\n' \
                % (evt_file))
        return evt_file


    def run_mcatnlo(self, evt_file):
        """runs mcatnlo on the generated event file, to produce showered-events"""
        logger.info('   Prepairing MCatNLO run')
        self.run_name = os.path.split(\
                    os.path.relpath(evt_file, pjoin(self.me_dir, 'Events')))[0]

        try:
            misc.call(['gunzip %s.gz' % evt_file], shell=True)
        except Exception:
            pass
        shower = self.evt_file_to_mcatnlo(evt_file)
        oldcwd = os.getcwd()
        os.chdir(pjoin(self.me_dir, 'MCatNLO'))
        shower_card_path = pjoin(self.me_dir, 'MCatNLO', 'shower_card.dat')

        if 'LD_LIBRARY_PATH' in os.environ.keys():
            ldlibrarypath = os.environ['LD_LIBRARY_PATH']
        else:
            ldlibrarypath = ''
        for path in self.shower_card['extrapaths'].split():
            ldlibrarypath += ':%s' % path
        if shower == 'HERWIGPP':
            ldlibrarypath += ':%s' % pjoin(self.shower_card['hepmcpath'], 'lib')
        os.putenv('LD_LIBRARY_PATH', ldlibrarypath)

        self.shower_card.write_card(shower, shower_card_path)

        mcatnlo_log = pjoin(self.me_dir, 'mcatnlo.log')
        self.update_status('   Compiling MCatNLO for %s...' % shower, level='parton') 
        misc.call(['./MCatNLO_MadFKS.inputs > %s 2>&1' % mcatnlo_log], \
                    cwd = os.getcwd(), shell=True)
        exe = 'MCATNLO_%s_EXE' % shower
        if not os.path.exists(exe):
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
        misc.call(['cp %s %s' % (shower_card_path, rundir)], shell=True)

        self.update_status('Running MCatNLO in %s (this may take some time)...' % rundir,
                level='parton')
        os.chdir(rundir)
        misc.call(['mv ../%s ../MCATNLO_%s_input .' % (exe, shower)], shell=True)
        #link the hwpp exe in the rundir
        if shower == 'HERWIGPP':
            try:
                misc.call(['ln -s %s %s' % \
                (pjoin(self.shower_card['hwpppath'], 'bin', 'Herwig++'), rundir)], shell=True)
            except Exception:
                raise aMCatNLOError('The Herwig++ path set in the shower_card is not valid.')

            if os.path.exists(pjoin(self.me_dir, 'MCatNLO', 'HWPPAnalyzer', 'HepMCFortran.so')):
                misc.call(['cp %s %s' % \
                    (pjoin(self.me_dir, 'MCatNLO', 'HWPPAnalyzer', 'HepMCFortran.so'), rundir)], 
                                                                                     shell=True)
        evt_name = os.path.basename(evt_file)
        misc.call(['ln -s %s %s' % (os.path.split(evt_file)[0], self.run_name)], shell=True)
        misc.call(['./%s < MCATNLO_%s_input > amcatnlo_run.log 2>&1' % \
                    (exe, shower)], cwd = os.getcwd(), shell=True)
        #copy the showered stdhep file back in events
        if not self.shower_card['analyse']:
            if os.path.exists(pjoin(self.run_name, evt_name + '.hep')):
                hep_file = '%s_%s_0.hep' % (evt_file[:-4], shower)
                count = 0
                while os.path.exists(hep_file + '.gz'):
                    count +=1
                    hep_file = '%s_%s_%d.hep' % (evt_file[:-4], shower, count)

                misc.call(['mv %s %s' % (pjoin(self.run_name, evt_name + '.hep'), hep_file)], shell=True) 
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

            else:
                raise aMCatNLOError('No file has been generated, an error occurred')
        else:
            topfiles = [n for n in os.listdir(pjoin(rundir)) \
                                            if n.lower().endswith('.top')]
            if not topfiles:
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

                files = 'files'
                have = 'have'
                if len(plotfiles) == 1:
                    files = 'file'
                    have = 'has'

                misc.call(['gzip %s' % evt_file], shell=True)
                logger.info(('The %s %s %s been generated, with histograms in the' + \
                        ' TopDrawer format, obtained by showering the parton-level' + \
                        ' file %s.gz with %s') % (files, ', '.join(plotfiles), have, \
                        evt_file, shower))


        os.chdir(oldcwd)


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
            self.results.update('add run %s' % name, 'all', makehtml=False)
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
            
            


    def evt_file_to_mcatnlo(self, evt_file):
        """creates the mcatnlo input script using the values set in the header of the event_file.
        It also checks if the lhapdf library is used"""
        file = open(evt_file)
        shower = ''
        pdlabel = ''
        itry = 0
        nevents = self.shower_card['nevents']
        while True:
            line = file.readline()
            #print line
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
                lines[i]='EVPREFIX=%s' % pjoin(self.run_name, os.path.split(evt_file)[1])
            if lines[i].startswith('NEVENTS'):
                lines[i]='NEVENTS=%d' % nevents
            if lines[i].startswith('MCMODE'):
                lines[i]='MCMODE=%s' % shower
            #the following variables are actually relevant only if running hw++
        
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
                misc.call(['cp %s %s' % (nev_unw + '.orig', nev_unw)], shell=True)
            else:
                raise aMCatNLOError('Cannot find event file information')

        #read the nevents_unweighted file to get the list of event files
        file = open(nev_unw)
        lines = file.read().split('\n')
        file.close()
        # make copy of the original nevent_unweighted file
        misc.call(['cp %s %s' % (nev_unw, nev_unw + '.orig')], shell=True)
        # loop over lines (all but the last one whith is empty) and check that the
        #  number of events is not 0
        evt_files = [line.split()[0] for line in lines[:-1] if line.split()[1] != '0']
        #prepare the job_dict
        job_dict = {}
        for i, evt_file in enumerate(evt_files):
            path, evt = os.path.split(evt_file)
            os.chdir(pjoin(self.me_dir, 'SubProcesses', path))
            if self.cluster_mode == 0 or self.cluster_mode == 2:
                exe = 'reweight_xsec_events.local'
            elif self.cluster_mode == 1:
                exe = 'reweight_xsec_events.cluster'
            misc.call(['ln -sf ../../%s .' % exe], shell=True)
            job_dict[path] = [exe]

        self.run_all(job_dict, [[evt, '1']], 'Running reweight')
        os.chdir(pjoin(self.me_dir, 'SubProcesses'))

        #check that the new event files are complete
        for evt_file in evt_files:
            last_line = subprocess.Popen('tail -n1 %s.rwgt ' % evt_file, \
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


    def wait_for_complete(self, run_type):
        """this function waits for jobs on cluster to complete their run."""

        starttime = time.time()
        #logger.info('     Waiting for submitted jobs to complete')
        update_status = lambda i, r, f: self.update_status((i, r, f, run_type), 
                      starttime=starttime, level='parton', update_results=False)
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
        os.chdir(pjoin(self.me_dir, 'SubProcesses'))


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
        else:
            #this is for the cluster/multicore run
            if 'ajob' not  in exe:
                return self.cluster.submit(exe, args, cwd=cwd)

            # use local disk if possible => need to stands what are the 
            # input/output files
            keep_fourth_arg = False
            output_files = []
            input_files = [pjoin(self.me_dir, 'MGMEVersion.txt'),
                           pjoin(self.me_dir, 'SubProcesses', 'randinit'),
                           pjoin(cwd, 'symfact.dat'),
                           pjoin(cwd, 'iproc.dat')]
            
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
                            files.ln(pjoin(cwd,base, name), 
                                            starting_dir=pjoin(cwd,current))
                        files.ln(pjoin(cwd,base, 'grid.MC_integer'), 
                                            starting_dir=pjoin(cwd,current))
                                  
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
            if hasattr(self, 'pdffile'):
                input_files.append(self.pdffile)
            else:
                for line in open(pjoin(self.me_dir,'Source','PDF','pdf_list.txt')):
                    data = line.split()
                    if len(data) < 4:
                        continue
                    if data[1].lower() == self.run_card['pdlabel'].lower():
                        self.pdffile = pjoin(self.me_dir, 'lib', 'Pdfdata', data[2])
                        input_files.append(self.pdffile) 
                        break
                else:
                    # possible when using lhapdf
                    self.pdffile = subprocess.Popen('%s --pdfsets-path' % self.options['lhapdf'], 
                            shell = True, stdout = subprocess.PIPE).stdout.read().strip()
                    #self.pdffile = pjoin(self.me_dir, 'lib', 'PDFsets')
                    input_files.append(self.pdffile)
                    
            
            if len(args) == 4 and not keep_fourth_arg:
                args = args[:3]

            #submitting
            self.cluster.submit2(exe, args, cwd=cwd, 
                         input_files=input_files, output_files=output_files)
            
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

        old_cwd = os.getcwd()

        #clean files
        misc.call(['rm -f %s' % 
                ' '.join([amcatnlo_log, madloop_log, reweight_log, gensym_log, test_log])], \
                  cwd=self.me_dir, shell=True)

        import multiprocessing
        nb_core = multiprocessing.cpu_count()

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

        # create param_card.inc and run_card.inc
        self.do_treatcards('', amcatnlo=True)
        if all([os.path.exists(pjoin(self.me_dir, 'SubProcesses', p_dir, exe)) \
                for p_dir in p_dirs]) and options['nocompile']:
            os.chdir(old_cwd)
            return

        libdir = pjoin(self.me_dir, 'lib')

        # rm links to lhapdflib/ PDFsets if exist
        if os.path.islink(pjoin(libdir, 'libLHAPDF.a')):
            os.remove(pjoin(libdir, 'libLHAPDF.a'))
        if os.path.islink(pjoin(libdir, 'PDFsets')):
            os.remove(pjoin(libdir, 'PDFsets'))

        # read the run_card to find if lhapdf is used or not
        if self.run_card['pdlabel'] == 'lhapdf':
            self.link_lhapdf(libdir)
        else:
            logger.info('Using built-in libraries for PDFs')
            os.unsetenv('lhapdf')

        # make Source
        self.update_status('Compiling source...', level=None)
        os.chdir(pjoin(self.me_dir, 'Source'))
        misc.call(['make -j%d > %s 2>&1' % (nb_core, amcatnlo_log)], shell=True)
        if os.path.exists(pjoin(libdir, 'libdhelas.a')) \
          and os.path.exists(pjoin(libdir, 'libgeneric.a')) \
          and os.path.exists(pjoin(libdir, 'libmodel.a')) \
          and os.path.exists(pjoin(libdir, 'libpdf.a')):
            logger.info('          ...done, continuing with P* directories')
        else:
            raise aMCatNLOError('Compilation failed, check %s for details' % amcatnlo_log)

        # make and run tests (if asked for), gensym and make madevent in each dir
        self.update_status('Compiling directories...', level=None)
        for p_dir in p_dirs:
            logger.info(p_dir)
            this_dir = pjoin(self.me_dir, 'SubProcesses', p_dir) 
            os.chdir(this_dir)
            # compile and run tests
            for test in tests:
                logger.info('   Compiling %s...' % test)
                misc.call(['make -j%d %s >> %s 2>&1 ' % (nb_core, test, test_log)], shell=True)
                if not os.path.exists(pjoin(this_dir, test)):
                    raise aMCatNLOError('Compilation failed, check %s for details' \
                            % test_log)
            for test in tests:
                logger.info('   Running %s...' % test)
                self.write_test_input(test)
                input = pjoin(self.me_dir, '%s_input.txt' % test)
                #this can be improved/better written to handle the output
                p = misc.Popen(['./%s < %s | tee -a %s | grep "Fraction of failures"'\
                         % (test, input, test_log)],stdout=subprocess.PIPE, shell=True)
                output = p.communicate()
                #check that none of the tests failed
                file = open(test_log)
                content = file.read()
                file.close()
                if 'FAILED' in content:
                    logger.info('Output of the failing test:\n'+output[0][:-1],'$MG:color:BLACK')
                    raise aMCatNLOError('Some tests failed, run cannot continue.\n' + \
                        'Please check that widths of final state particles (e.g. top) have been' + \
                        ' set to 0 in the param_card.dat.')
                else:
                    logger.info('   Passed.')
                    logger.debug('\n'+output[0][:-1])
#                misc.call(['./%s < %s | tee -a %s | grep "Fraction of failures"' \
#                        % (test, input, test_log)], shell=True)
                
            if not options['reweightonly']:
                logger.info('   Compiling gensym...')
                misc.call(['make -j%d gensym >> %s 2>&1 ' % (nb_core, amcatnlo_log)], shell=True)
                if not os.path.exists(pjoin(this_dir, 'gensym')):
                    raise aMCatNLOError('Compilation failed, check %s for details' % amcatnlo_log)

                logger.info('   Running gensym...')
                misc.call(['echo %s | ./gensym >> %s' % (self.options['run_mode'], gensym_log)], shell=True) 
                #compile madloop library
                v_dirs = [file for file in os.listdir('.') if file.startswith('V') and os.path.isdir(file)]
                for v_dir in v_dirs:
                    os.putenv('madloop', 'true')
                    logger.info('   Compiling MadLoop library in %s' % v_dir)
                    madloop_dir = pjoin(this_dir, v_dir)
                    os.chdir(madloop_dir)
                    misc.call(['make -j%d >> %s 2>&1' % (nb_core, madloop_log)], shell=True)
                    if not os.path.exists(pjoin(this_dir, 'libMadLoop.a')):
                        raise aMCatNLOError('Compilation failed, check %s for details' % madloop_log)
                #compile and run check_poles if the virtuals have been exported
                proc_card = open(pjoin(self.me_dir, 'Cards', 'proc_card_mg5.dat')).read()
                os.chdir(this_dir)
                if not '[real=QCD]' in proc_card:
                    logger.info('   Compiling check_poles...')
                    misc.call(['make -j%d check_poles >> %s 2>&1 ' % (nb_core, test_log)], shell=True)
                    if not os.path.exists(pjoin(this_dir, 'check_poles')):
                        raise aMCatNLOError('Compilation failed, check %s for details' \
                                % test_log)
                    logger.info('   Running check_poles...')
                    open('./check_poles.input', 'w').write('20 \n -1\n') 
                    misc.call(['./check_poles <check_poles.input >> %s' % (test_log)], shell=True) 
                    self.parse_check_poles_log(os.getcwd())
                #compile madevent_mintMC/vegas
                logger.info('   Compiling %s' % exe)
                misc.call(['make -j%d %s >> %s 2>&1' % (nb_core, exe, amcatnlo_log)], shell=True)
                os.unsetenv('madloop')
                if not os.path.exists(pjoin(this_dir, exe)):
                    raise aMCatNLOError('Compilation failed, check %s for details' % amcatnlo_log)
            if mode in ['aMC@NLO', 'aMC@LO'] and not options['noreweight']:
                logger.info('   Compiling reweight_xsec_events')
                misc.call(['make -j%d reweight_xsec_events >> %s 2>&1' % (nb_core, reweight_log)], shell=True)
                if not os.path.exists(pjoin(this_dir, 'reweight_xsec_events')):
                    raise aMCatNLOError('Compilation failed, check %s for details' % reweight_log)

        os.chdir(old_cwd)

    def parse_check_poles_log(self, dir):
        """reads and parse the check_poles.log file"""
        content = open(pjoin(dir, 'check_poles.log')).read()
        npass = 0
        nfail = 0
        for line in content.split('\n'):
            if 'PASSED' in line:
                npass +=1
                tolerance = float(line.split()[1])
            if 'FAILED' in line:
                nfail +=1
                tolerance = float(line.split()[1])

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
            shower = self.run_card['parton_shower']
            MC_header = "%s\n " % shower + \
                        "1 \n1 -0.1\n-1 -0.1\n"
            file.write(MC_header + content)
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
        
        if 'parton' in options.keys():
            if options['parton'] == False:
                cards = ['param', 'run', 'shower']
            elif options['parton'] == 'onlyshower':
                cards = ['shower']
            else:  
                cards = ['param', 'run']
        else:  
            cards = ['param', 'run', 'shower']

        def get_question(mode, cards):
            # Ask the user if he wants to edit any of the files
            #First create the asking text
            question = "Do you want to edit a card (press enter to bypass editing)?\n" + \
                       "(be careful about parameter consistency, especially widths)\n"
            card = {0:'done'}
            for i, c in enumerate(cards):
                card[i+1] = c

            possible_answer = []
            for i, c in card.items():
                if i > 0:
                    question += '%d / %6s : %s_card.dat\n' % (i, c, c)
                else:
                    question += '%d / %6s \n' % (i, c)
                possible_answer.extend([i,c])

            # Add the path options
            question += '  Path to a valid card.\n'
            return question, possible_answer, card
        
        # Loop as long as the user is not done.
        answer = 'no'
        if options['force'] or self.force:
            answer='done'
        while answer != 'done':
            question, possible_answer, card = get_question(mode, cards)
            answer = self.ask(question, '0', possible_answer, timeout=int(1.5*self.options['timeout']), path_msg='enter path')
            if answer.isdigit():
                answer = card[int(answer)]
            if answer == 'done':
                break
            if not os.path.isfile(answer):
                if answer != 'trigger':
                    path = pjoin(self.me_dir,'Cards','%s_card.dat' % answer)
                else:
                    path = pjoin(self.me_dir,'Cards','delphes_trigger.dat')
                self.exec_cmd('open %s' % path)                    
            else:
                # detect which card is provided
                card_name = self.detect_card_type(answer)
                if card_name == 'unknown':
                    card_name = self.ask('Fail to determine the type of the file. Please specify the format',
                   'param_card.dat', choices=['param_card.dat', 'run_card.dat','pythia_card.dat','pgs_card.dat',
                    'delphes_card.dat', 'delphes_trigger.dat','plot_card.dat'])
                if card_name != 'banner':
                    logger.info('copy %s as %s' % (answer, card_name))
                    files.cp(answer, pjoin(self.me_dir, 'Cards', card_name))
                    #if card_name == 'param_card.dat':
                    #    self.check_param_card(pjoin(self.me_dir, 'Cards', card_name))                        
                elif card_name == 'banner':
                    banner_mod.split_banner(answer, self.me_dir, proc_card=False)
                    logger.info('Splitting the banner in it\'s component')
                    #if 0:
                    #    # Re-compute the current mode
                    #    mode = 'parton'
                    #    for level in ['delphes','pgs','pythia']:
                    #        if os.path.exists(pjoin(self.me_dir,'Cards','%s_card.dat' % level)):
                    #            mode = level
                    #            break
                    #else:
                    #    self.clean_pointless_card(mode)

        run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
        self.run_card = banner_mod.RunCardNLO(run_card)
        self.run_tag = self.run_card['run_tag']
        self.run_name = self.find_available_run_name(self.me_dir)
        self.set_run_name(self.run_name, self.run_tag, 'parton')
        shower_card_path = pjoin(self.me_dir, 'Cards','shower_card.dat')
        self.shower_card = shower_card.ShowerCard(shower_card_path)
        #self.do_treatcards_nlo('')
        return

    def do_quit(self, line):
        """ """
        try:
            os.remove(pjoin(self.me_dir,'RunWeb'))
        except Exception:
            pass
#        try:
#            self.store_result()
#        except:
#            # If nothing runs they they are no result to update
#            pass
#        try:
#            self.update_status('', level=None)
#        except Exception, error:         
#            pass
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
_compile_parser.add_option("-R", "--noreweight", default=False, action='store_true',
                            help="Skip compiling reweight executable")


_launch_usage = "launch [MODE] [options]\n" + \
                "-- execute aMC@NLO \n" + \
                "   MODE can be either LO, NLO, aMC@NLO or aMC@LO (if omitted, it is set to aMC@NLO)\n" + \
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
_launch_parser.add_option("-R", "--noreweight", default=False, action='store_true',
                            help="Skip file reweighting")
_launch_parser.add_option("-p", "--parton", default=False, action='store_true',
                            help="Stop the run after the parton level file generation (you need " + \
                                    "to shower the file in order to get physical results)")


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
_generate_events_parser.add_option("-R", "--noreweight", default=False, action='store_true',
                            help="Skip file reweighting")
_generate_events_parser.add_option("-p", "--parton", default=False, action='store_true',
                            help="Stop the run after the parton level file generation (you need " + \
                                    "to shower the file in order to get physical results)")
