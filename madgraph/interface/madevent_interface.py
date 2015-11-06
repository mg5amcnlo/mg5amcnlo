################################################################################
#
# Copyright (c) 2011 The MadGraph5_aMC@NLO Development team and Contributors
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
"""A user friendly command line interface to access MadGraph5_aMC@NLO features.
   Uses the cmd package for command interpretation and tab completion.
"""
from __future__ import division

import atexit
import cmath
import glob
import logging
import math
import optparse
import os
import pydoc
import random
import re
import signal
import shutil
import stat
import subprocess
import sys
import traceback
import time
import tarfile

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
    import madgraph

    
except ImportError: 
    # import from madevent directory
    MADEVENT = True
    import internal.extended_cmd as cmd
    import internal.common_run_interface as common_run
    import internal.banner as banner_mod
    import internal.misc as misc    
    from internal import InvalidCmd, MadGraph5Error, ReadWrite
    import internal.files as files
    import internal.gen_crossxhtml as gen_crossxhtml
    import internal.gen_ximprove as gen_ximprove
    import internal.save_load_object as save_load_object
    import internal.cluster as cluster
    import internal.check_param_card as check_param_card
    import internal.sum_html as sum_html
    import internal.combine_runs as combine_runs
    import internal.lhe_parser as lhe_parser
else:
    # import from madgraph directory
    MADEVENT = False
    import madgraph.interface.extended_cmd as cmd
    import madgraph.interface.common_run_interface as common_run
    import madgraph.iolibs.files as files
    import madgraph.iolibs.save_load_object as save_load_object
    import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
    import madgraph.madevent.gen_ximprove as gen_ximprove
    import madgraph.madevent.sum_html as sum_html
    import madgraph.various.banner as banner_mod
    import madgraph.various.cluster as cluster
    import madgraph.various.misc as misc
    import madgraph.madevent.combine_runs as combine_runs
    import madgraph.various.lhe_parser as lhe_parser

    import models.check_param_card as check_param_card    
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR, ReadWrite




class MadEventError(Exception):
    pass

class ZeroResult(MadEventError):
    pass

class SysCalcError(InvalidCmd): pass

MadEventAlreadyRunning = common_run.MadEventAlreadyRunning

#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(common_run.CommonRunCmd):
    """Particularisation of the cmd command for MadEvent"""

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
            in order to quit MadGraph5_aMC@NLO please enter exit"""
    
    # Define the Error
    InvalidCmd = InvalidCmd
    ConfigurationError = MadGraph5Error

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
        '#*               MadGraph5_aMC@NLO/MadEvent                 *\n' + \
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
        "#*    The MadGraph5_aMC@NLO Development Team - Find us at   *\n" + \
        "#*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
        '#*                                                          *\n' + \
        '#************************************************************\n' + \
        '#*                                                          *\n' + \
        '#*               Command File for MadEvent                  *\n' + \
        '#*                                                          *\n' + \
        '#*     run as ./bin/madevent.py filename                    *\n' + \
        '#*                                                          *\n' + \
        '#************************************************************\n'
        
        if info_line:
            info_line = info_line[1:]

        logger.info(\
        "************************************************************\n" + \
        "*                                                          *\n" + \
        "*                      W E L C O M E to                    *\n" + \
        "*             M A D G R A P H 5 _ a M C @ N L O            *\n" + \
        "*                      M A D E V E N T                     *\n" + \
        "*                                                          *\n" + \
        "*                 *                       *                *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                     * * * * 5 * * * *                    *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                 *                       *                *\n" + \
        "*                                                          *\n" + \
        info_line + \
        "*                                                          *\n" + \
        "*    The MadGraph5_aMC@NLO Development Team - Find us at   *\n" + \
        "*    https://server06.fynu.ucl.ac.be/projects/madgraph     *\n" + \
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
            if hasattr(self, 'cluster'):
                logger.info('rm jobs on queue')
                self.cluster.remove()
            if hasattr(self, 'results'):
                self.update_status('Stop by the user', level=None, makehtml=False, error=True)
                self.add_error_log_in_html(KeyboardInterrupt)
        except:
            pass
    
    def postcmd(self, stop, line):
        """ Update the status of  the run for finishing interactive command """
        
        stop = super(CmdExtended, self).postcmd(stop, line)   
        # relaxing the tag forbidding question
        self.force = False
        
        if not self.use_rawinput:
            return stop
        
        if self.results and not self.results.current:
            return stop
        
        arg = line.split()
        if  len(arg) == 0:
            return stop
        if isinstance(self.results.status, str) and self.results.status.startswith('Error'):
            return stop
        if isinstance(self.results.status, str) and self.results.status == 'Stop by the user':
            self.update_status('%s Stop by the user' % arg[0], level=None, error=True)
            return stop        
        elif not self.results.status:
            return stop
        elif str(arg[0]) in ['exit','quit','EOF']:
            return stop
        
        try:
            self.update_status('Command \'%s\' done.<br> Waiting for instruction.' % arg[0], 
                               level=None, error=True)
        except Exception:
            misc.sprint('update_status fails')
            pass
        
    
    def nice_user_error(self, error, line):
        """If a ME run is currently running add a link in the html output"""

        self.add_error_log_in_html()
        cmd.Cmd.nice_user_error(self, error, line)            
        
    def nice_config_error(self, error, line):
        """If a ME run is currently running add a link in the html output"""

        self.add_error_log_in_html()
        cmd.Cmd.nice_config_error(self, error, line)
        
        
        try:
            debug_file = open(self.debug_output, 'a')
            debug_file.write(open(pjoin(self.me_dir,'Cards','proc_card_mg5.dat')))
            debug_file.close()
        except:
            pass 
            

    def nice_error_handling(self, error, line):
        """If a ME run is currently running add a link in the html output"""

        if isinstance(error, ZeroResult):
            self.add_error_log_in_html(error)
            logger.warning('Zero result detected: %s' % error)
            # create a banner if needed
            try:
                if not self.banner:
                    self.banner = banner_mod.Banner()
                if 'slha' not in self.banner:
                    self.banner.add(pjoin(self.me_dir,'Cards','param_card.dat'))
                if 'mgruncard' not in self.banner:
                    self.banner.add(pjoin(self.me_dir,'Cards','run_card.dat'))
                if 'mg5proccard' not in self.banner:
                    proc_card = pjoin(self.me_dir,'Cards','proc_card_mg5.dat')
                    if os.path.exists(proc_card):
                        self.banner.add(proc_card)
                
                out_dir = pjoin(self.me_dir, 'Events', self.run_name)
                if not os.path.isdir(out_dir):
                    os.mkdir(out_dir)
                output_path = pjoin(out_dir, '%s_%s_banner.txt' % \
                                                  (self.run_name, self.run_tag))
                self.banner.write(output_path)
            except Exception:
                if __debug__:
                    raise
                else:
                    pass
        else:
            self.add_error_log_in_html()            
            cmd.Cmd.nice_error_handling(self, error, line)
            try:
                debug_file = open(self.debug_output, 'a')
                debug_file.write(open(pjoin(self.me_dir,'Cards','proc_card_mg5.dat')))
                debug_file.close()
            except:
                pass
        
        
#===============================================================================
# HelpToCmd
#===============================================================================
class HelpToCmd(object):
    """ The Series of help routine for the MadEventCmd"""
    
    def help_banner_run(self):
        logger.info("syntax: banner_run Path|RUN [--run_options]")
        logger.info("-- Reproduce a run following a given banner")
        logger.info("   One of the following argument is require:")
        logger.info("   Path should be the path of a valid banner.")
        logger.info("   RUN should be the name of a run of the current directory")
        self.run_options_help([('-f','answer all question by default'),
                               ('--name=X', 'Define the name associated with the new run')]) 
    
    def help_open(self):
        logger.info("syntax: open FILE  ")
        logger.info("-- open a file with the appropriate editor.")
        logger.info('   If FILE belongs to index.html, param_card.dat, run_card.dat')
        logger.info('   the path to the last created/used directory is used')
        logger.info('   The program used to open those files can be chosen in the')
        logger.info('   configuration file ./input/mg5_configuration.txt')   
        
        
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
        

    def help_generate_events(self):
        logger.info("syntax: generate_events [run_name] [options]",)
        logger.info("-- Launch the full chain of script for the generation of events")
        logger.info("   Including possible plotting, shower and detector resolution.")
        logger.info("   Those steps are performed if the related program are installed")
        logger.info("   and if the related card are present in the Cards directory.")
        self.run_options_help([('-f', 'Use default for all questions.'),
                               ('--laststep=', 'argument might be parton/pythia/pgs/delphes and indicate the last level to be run.'),
                               ('-M', 'in order to add MadSpin'),
                               ('-R', 'in order to add the reweighting module')])

    def help_initMadLoop(self):
        logger.info("syntax: initMadLoop [options]",'$MG:color:GREEN')
        logger.info(
"""-- Command only useful when MadEvent simulates loop-induced processes. This command compiles and run
   the MadLoop output for the matrix element computation so as to initialize the filter for analytically
   zero helicity configurations and loop topologies. If you suspect that a change you made in the model
   parameters can have affected these filters, this command allows you to automatically refresh them. """)
        logger.info("   The available options are:",'$MG:color:BLUE')
        logger.info("     -f          : Bypass the edition of MadLoopParams.dat.",'$MG:color:BLUE')
        logger.info("     -r          : Refresh of the existing filters (erasing them if already present).",'$MG:color:BLUE')
        logger.info("     --nPS=<int> : Specify how many phase-space points should be tried to set up the filters.",'$MG:color:BLUE')
        
    def help_add_time_of_flight(self):
        logger.info("syntax: add_time_of_flight [run_name|path_to_file] [--threshold=]")
        logger.info('-- Add in the lhe files the information')
        logger.info('   of how long it takes to a particle to decay.')
        logger.info('   threshold option allows to change the minimal value required to')
        logger.info('   a non zero value for the particle (default:1e-12s)')

    def help_calculate_decay_widths(self):
        
        if self.ninitial != 1:
            logger.warning("This command is only valid for processes of type A > B C.")
            logger.warning("This command can not be run in current context.")
            logger.warning("")
        
        logger.info("syntax: calculate_decay_widths [run_name] [options])")
        logger.info("-- Calculate decay widths and enter widths and BRs in param_card")
        logger.info("   for a series of processes of type A > B C ...")
        self.run_options_help([('-f', 'Use default for all questions.'),
                               ('--accuracy=', 'accuracy (for each partial decay width).'\
                                + ' Default is 0.01.')])

    def help_multi_run(self):
        logger.info("syntax: multi_run NB_RUN [run_name] [--run_options])")
        logger.info("-- Launch the full chain of script for the generation of events")
        logger.info("   NB_RUN times. This chains includes possible plotting, shower")
        logger.info(" and detector resolution.")
        self.run_options_help([('-f', 'Use default for all questions.'),
                               ('--laststep=', 'argument might be parton/pythia/pgs/delphes and indicate the last level to be run.')])

    def help_survey(self):
        logger.info("syntax: survey [run_name] [--run_options])")
        logger.info("-- evaluate the different channel associate to the process")
        self.run_options_help([("--" + key,value[-1]) for (key,value) in \
                               self._survey_options.items()])
     
    def help_launch(self):
        """exec generate_events for 2>N and calculate_width for 1>N"""
        logger.info("syntax: launch [run_name] [options])")
        logger.info("    --alias for either generate_events/calculate_decay_widths")
        logger.info("      depending of the number of particles in the initial state.")
        
        if self.ninitial == 1:
            logger.info("For this directory this is equivalent to calculate_decay_widths")
            self.help_calculate_decay_widths()
        else:
            logger.info("For this directory this is equivalent to $generate_events")
            self.help_generate_events()
                 
    def help_refine(self):
        logger.info("syntax: refine require_precision [max_channel] [--run_options]")
        logger.info("-- refine the LAST run to achieve a given precision.")
        logger.info("   require_precision: can be either the targeted number of events")
        logger.info('                      or the required relative error')
        logger.info('   max_channel:[5] maximal number of channel per job')
        self.run_options_help([])
        
    def help_combine_events(self):
        """ """
        logger.info("syntax: combine_events [run_name] [--tag=tag_name] [--run_options]")
        logger.info("-- Combine the last run in order to write the number of events")
        logger.info("   asked in the run_card.")
        self.run_options_help([])
        
    def help_store_events(self):
        """ """
        logger.info("syntax: store_events [--run_options]")
        logger.info("-- Write physically the events in the files.")
        logger.info("   should be launch after \'combine_events\'")
        self.run_options_help([])

    def help_create_gridpack(self):
        """ """
        logger.info("syntax: create_gridpack [--run_options]")
        logger.info("-- create the gridpack. ")
        logger.info("   should be launch after \'store_events\'")
        self.run_options_help([])

    def help_import(self):
        """ """
        logger.info("syntax: import command PATH")
        logger.info("-- Execute the command present in the file")
        self.run_options_help([])
        
    def help_syscalc(self):
        logger.info("syntax: syscalc [RUN] [%s] [-f | --tag=]" % '|'.join(self._plot_mode))
        logger.info("-- calculate systematics information for the RUN (current run by default)")
        logger.info("     at different stages of the event generation for scale/pdf/...")

    def help_remove(self):
        logger.info("syntax: remove RUN [all|parton|pythia|pgs|delphes|banner] [-f] [--tag=]")
        logger.info("-- Remove all the files linked to previous run RUN")
        logger.info("   if RUN is 'all', then all run will be cleaned.")
        logger.info("   The optional argument precise which part should be cleaned.")
        logger.info("   By default we clean all the related files but the banners.")
        logger.info("   the optional '-f' allows to by-pass all security question")
        logger.info("   The banner can be remove only if all files are removed first.")



#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(object):
    """ The Series of check routine for the MadEventCmd"""

    def check_banner_run(self, args):
        """check the validity of line"""
        
        if len(args) == 0:
            self.help_banner_run()
            raise self.InvalidCmd('banner_run requires at least one argument.')
        
        tag = [a[6:] for a in args if a.startswith('--tag=')]
        
        
        if os.path.exists(args[0]):
            type ='banner'
            format = self.detect_card_type(args[0])
            if format != 'banner':
                raise self.InvalidCmd('The file is not a valid banner.')
        elif tag:
            args[0] = pjoin(self.me_dir,'Events', args[0], '%s_%s_banner.txt' % \
                                    (args[0], tag))                  
            if not os.path.exists(args[0]):
                raise self.InvalidCmd('No banner associates to this name and tag.')
        else:
            name = args[0]
            type = 'run'
            banners = glob.glob(pjoin(self.me_dir,'Events', args[0], '*_banner.txt'))
            if not banners:
                raise self.InvalidCmd('No banner associates to this name.')    
            elif len(banners) == 1:
                args[0] = banners[0]
            else:
                #list the tag and propose those to the user
                tags = [os.path.basename(p)[len(args[0])+1:-11] for p in banners]
                tag = self.ask('which tag do you want to use?', tags[0], tags)
                args[0] = pjoin(self.me_dir,'Events', args[0], '%s_%s_banner.txt' % \
                                    (args[0], tag))                
                        
        run_name = [arg[7:] for arg in args if arg.startswith('--name=')]
        if run_name:
            try:
                self.exec_cmd('remove %s all banner -f' % run_name)
            except Exception:
                pass
            self.set_run_name(args[0], tag=None, level='parton', reload_card=True)
        elif type == 'banner':
            self.set_run_name(self.find_available_run_name(self.me_dir))
        elif type == 'run':
            if not self.results[name].is_empty():
                run_name = self.find_available_run_name(self.me_dir)
                logger.info('Run %s is not empty so will use run_name: %s' % \
                                                               (name, run_name))
                self.set_run_name(run_name)
            else:
                try:
                    self.exec_cmd('remove %s all banner -f' % run_name)
                except Exception:
                    pass
                self.set_run_name(name)
            
    def check_history(self, args):
        """check the validity of line"""
        
        if len(args) > 1:
            self.help_history()
            raise self.InvalidCmd('\"history\" command takes at most one argument')
        
        if not len(args):
            return
        elif args[0] != 'clean':
                dirpath = os.path.dirname(args[0])
                if dirpath and not os.path.exists(dirpath) or \
                       os.path.isdir(args[0]):
                    raise self.InvalidCmd("invalid path %s " % dirpath)
                
    def check_save(self, args):
        """ check the validity of the line"""
        
        if len(args) == 0:
            args.append('options')

        if args[0] not in self._save_opts:
            raise self.InvalidCmd('wrong \"save\" format')
        
        if args[0] != 'options' and len(args) != 2:
            self.help_save()
            raise self.InvalidCmd('wrong \"save\" format')
        elif args[0] != 'options' and len(args) == 2:
            basename = os.path.dirname(args[1])
            if not os.path.exists(basename):
                raise self.InvalidCmd('%s is not a valid path, please retry' % \
                                                                        args[1])
        
        if args[0] == 'options':
            has_path = None
            for arg in args[1:]:
                if arg in ['--auto', '--all']:
                    continue
                elif arg.startswith('--'):
                    raise self.InvalidCmd('unknow command for \'save options\'')
                else:
                    basename = os.path.dirname(arg)
                    if not os.path.exists(basename):
                        raise self.InvalidCmd('%s is not a valid path, please retry' % \
                                                                        arg)
                    elif has_path:
                        raise self.InvalidCmd('only one path is allowed')
                    else:
                        args.remove(arg)
                        args.insert(1, arg)
                        has_path = True
            if not has_path:
                if '--auto' in arg and self.options['mg5_path']:
                    args.insert(1, pjoin(self.options['mg5_path'],'input','mg5_configuration.txt'))  
                else:
                    args.insert(1, pjoin(self.me_dir,'Cards','me5_configuration.txt'))  

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
            if args[1] not in ['DEBUG','INFO','WARNING','ERROR','CRITICAL'] \
                                                       and not args[1].isdigit():
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
                files.cp(os.path.join(path,'Cards', name), os.path.join(path,'Cards', args[0]))
                args[0] = os.path.join(path,'Cards', args[0])
            else:
                raise self.InvalidCmd('No default path for this file')
        elif not os.path.isfile(args[0]):
            raise self.InvalidCmd('No default path for this file') 
   
    def check_initMadLoop(self, args):
        """ check initMadLoop command arguments are valid."""
        
        opt = {'refresh': False, 'nPS': None, 'force': False}
        
        for arg in args:
            if arg in ['-r','--refresh']:
                opt['refresh'] = True
            if arg in ['-f','--force']:
                opt['force'] = True
            elif arg.startswith('--nPS='):
                n_attempts = arg.split('=')[1]
                try:
                    opt['nPS'] = int(n_attempts)
                except ValueError:
                    raise InvalidCmd("The number of attempts specified "+
                                      "'%s' is not a valid integer."%n_attempts)
        
        return opt
        
    def check_treatcards(self, args):
        """check that treatcards arguments are valid
           [param|run|all] [--output_dir=] [--param_card=] [--run_card=]
        """
        
        opt = {'output_dir':pjoin(self.me_dir,'Source'),
               'param_card':pjoin(self.me_dir,'Cards','param_card.dat'),
               'run_card':pjoin(self.me_dir,'Cards','run_card.dat'),
               'forbid_MadLoopInit': False}
        mode = 'all'
        for arg in args:
            if arg.startswith('--') and '=' in arg:
                key,value =arg[2:].split('=',1)
                if not key in opt:
                    self.help_treatcards()
                    raise self.InvalidCmd('Invalid option for treatcards command:%s ' \
                                          % key)
                if key in ['param_card', 'run_card']:
                    if os.path.isfile(value):
                        card_name = self.detect_card_type(value)
                        if card_name != key:
                            raise self.InvalidCmd('Format for input file detected as %s while expecting %s' 
                                                  % (card_name, key))
                        opt[key] = value
                    elif os.path.isfile(pjoin(self.me_dir,value)):
                        card_name = self.detect_card_type(pjoin(self.me_dir,value))
                        if card_name != key:
                            raise self.InvalidCmd('Format for input file detected as %s while expecting %s' 
                                                  % (card_name, key))                        
                        opt[key] = value
                    else:
                        raise self.InvalidCmd('No such file: %s ' % value)
                elif key in ['output_dir']:
                    if os.path.isdir(value):
                        opt[key] = value
                    elif os.path.isdir(pjoin(self.me_dir,value)):
                        opt[key] = pjoin(self.me_dir, value)
                    else:
                        raise self.InvalidCmd('No such directory: %s' % value)
            elif arg in ['loop','param','run','all']:
                mode = arg
            elif arg == '--no_MadLoopInit':
                opt['forbid_MadLoopInit'] = True
            else:
                self.help_treatcards()
                raise self.InvalidCmd('Unvalid argument %s' % arg)
                        
        return mode, opt 
    
    
    def check_survey(self, args, cmd='survey'):
        """check that the argument for survey are valid"""
        
        
        self.opts = dict([(key,value[1]) for (key,value) in \
                          self._survey_options.items()])

        # Treat any arguments starting with '--'
        while args and args[-1].startswith('--'):
            arg = args.pop(-1)
            try:
                for opt,value in self._survey_options.items():
                    if arg.startswith('--%s=' % opt):
                        exec('self.opts[\'%s\'] = %s(arg.split(\'=\')[-1])' % \
                             (opt, value[0]))
                        arg = ""
                if arg != "": raise Exception
            except Exception:
                self.help_survey()
                raise self.InvalidCmd('invalid %s argument'% arg)

        if len(args) > 1:
            self.help_survey()
            raise self.InvalidCmd('Too many argument for %s command' % cmd)
        elif not args:
            # No run name assigned -> assigned one automaticaly 
            self.set_run_name(self.find_available_run_name(self.me_dir))
        else:
            self.set_run_name(args[0], None,'parton', True)
            args.pop(0)
            
        return True

    def check_generate_events(self, args):
        """check that the argument for generate_events are valid"""
        
        run = None
        if args and args[-1].startswith('--laststep='):
            run = args[-1].split('=')[-1]
            if run not in ['auto','parton', 'pythia', 'pgs', 'delphes']:
                self.help_generate_events()
                raise self.InvalidCmd('invalid %s argument'% args[-1])
            if run != 'parton' and not self.options['pythia-pgs_path']:                
                raise self.InvalidCmd('''pythia-pgs not install. Please install this package first. 
                To do so type: \'install pythia-pgs\' in the mg5 interface''')
            if run == 'delphes' and not self.options['delphes_path']:
                raise self.InvalidCmd('''delphes not install. Please install this package first. 
                To do so type: \'install Delphes\' in the mg5 interface''')
            del args[-1]

                                
        #if len(args) > 1:
        #    self.help_generate_events()
        #    raise self.InvalidCmd('Too many argument for generate_events command: %s' % cmd)
                    
        return run

    def check_add_time_of_flight(self, args):
        """check that the argument are correct"""
        
        
        if len(args) >2:
            self.help_time_of_flight()
            raise self.InvalidCmd('Too many arguments')
        
        # check if the threshold is define. and keep it's value
        if args and args[-1].startswith('--threshold='):
            try:
                threshold = float(args[-1].split('=')[1])
            except ValueError:
                raise self.InvalidCmd('threshold options require a number.')
            args.remove(args[-1])
        else:
            threshold = 1e-12
            
        if len(args) == 1 and  os.path.exists(args[0]): 
                event_path = args[0]
        else:
            if len(args) and self.run_name != args[0]:
                self.set_run_name(args.pop(0))
            elif not self.run_name:            
                self.help_add_time_of_flight()
                raise self.InvalidCmd('Need a run_name to process')            
            event_path = pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe.gz')
            if not os.path.exists(event_path):
                event_path = event_path[:-3]
                if not os.path.exists(event_path):    
                    raise self.InvalidCmd('No unweighted events associate to this run.')


        
        #reformat the data
        args[:] = [event_path, threshold]

    def check_calculate_decay_widths(self, args):
        """check that the argument for calculate_decay_widths are valid"""
        
        if self.ninitial != 1:
            raise self.InvalidCmd('Can only calculate decay widths for decay processes A > B C ...')

        accuracy = 0.01
        run = None
        if args and args[-1].startswith('--accuracy='):
            try:
                accuracy = float(args[-1].split('=')[-1])
            except Exception:
                raise self.InvalidCmd('Argument error in calculate_decay_widths command')
            del args[-1]
        if len(args) > 1:
            self.help_calculate_decay_widths()
            raise self.InvalidCmd('Too many argument for calculate_decay_widths command: %s' % cmd)
                    
        return accuracy



    def check_multi_run(self, args):
        """check that the argument for survey are valid"""

        run = None
        
        if not len(args):
            self.help_multi_run()
            raise self.InvalidCmd("""multi_run command requires at least one argument for
            the number of times that it call generate_events command""")
            
        if args[-1].startswith('--laststep='):
            run = args[-1].split('=')[-1]
            if run not in ['parton', 'pythia', 'pgs', 'delphes']:
                self.help_multi_run()
                raise self.InvalidCmd('invalid %s argument'% args[-1])
            if run != 'parton' and not self.options['pythia-pgs_path']:                
                raise self.InvalidCmd('''pythia-pgs not install. Please install this package first. 
                To do so type: \'install pythia-pgs\' in the mg5 interface''')
            if run == 'delphes' and not self.options['delphes_path']:
                raise self.InvalidCmd('''delphes not install. Please install this package first. 
                To do so type: \'install Delphes\' in the mg5 interface''')
            del args[-1]
            

        elif not args[0].isdigit():
            self.help_multi_run()
            raise self.InvalidCmd("The first argument of multi_run should be a integer.")
        nb_run = args.pop(0)
        self.check_survey(args, cmd='multi_run')
        args.insert(0, int(nb_run))
        
        return run

    def check_refine(self, args):
        """check that the argument for survey are valid"""

        # if last argument is not a number -> it's the run_name (Not allow anymore)
        try:
            float(args[-1])
        except ValueError:
            self.help_refine()
            raise self.InvalidCmd('Not valid arguments')
        except IndexError:
            self.help_refine()
            raise self.InvalidCmd('require_precision argument is require for refine cmd')

    
        if not self.run_name:
            if self.results.lastrun:
                self.set_run_name(self.results.lastrun)
            else:
                raise self.InvalidCmd('No run_name currently define. Unable to run refine')

        if len(args) > 2:
            self.help_refine()
            raise self.InvalidCmd('Too many argument for refine command')
        else:
            try:
                [float(arg) for arg in args]
            except ValueError:         
                self.help_refine()    
                raise self.InvalidCmd('refine arguments are suppose to be number')
            
        return True
        
    def check_combine_events(self, arg):
        """ Check the argument for the combine events command """
        
        tag = [a for a in arg if a.startswith('--tag=')]
        if tag: 
            arg.remove(tag[0])
            tag = tag[0][6:]
        elif not self.run_tag:
            tag = 'tag_1'
        else:
            tag = self.run_tag
        self.run_tag = tag
     
        if len(arg) > 1:
            self.help_combine_events()
            raise self.InvalidCmd('Too many argument for combine_events command')
        
        if len(arg) == 1:
            self.set_run_name(arg[0], self.run_tag, 'parton', True)
        
        if not self.run_name:
            if not self.results.lastrun:
                raise self.InvalidCmd('No run_name currently define. Unable to run combine')
            else:
                self.set_run_name(self.results.lastrun)
        
        return True
    
    def check_pythia(self, args):
        """Check the argument for pythia command
        syntax: pythia [NAME] 
        Note that other option are already remove at this point
        """
        
        mode = None
        laststep = [arg for arg in args if arg.startswith('--laststep=')]
        if laststep and len(laststep)==1:
            mode = laststep[0].split('=')[-1]
            if mode not in ['auto', 'pythia', 'pgs', 'delphes']:
                self.help_pythia()
                raise self.InvalidCmd('invalid %s argument'% args[-1])     
        elif laststep:
            raise self.InvalidCmd('only one laststep argument is allowed')
     
        # If not pythia-pgs path
        if not self.options['pythia-pgs_path']:
            logger.info('Retry to read configuration file to find pythia-pgs path')
            self.set_configuration()
            
        if not self.options['pythia-pgs_path'] or not \
            os.path.exists(pjoin(self.options['pythia-pgs_path'],'src')):
            error_msg = 'No valid pythia-pgs path set.\n'
            error_msg += 'Please use the set command to define the path and retry.\n'
            error_msg += 'You can also define it in the configuration file.\n'
            raise self.InvalidCmd(error_msg)
     
     
     
        tag = [a for a in args if a.startswith('--tag=')]
        if tag: 
            args.remove(tag[0])
            tag = tag[0][6:]

        if len(args) == 0 and not self.run_name:
            if self.results.lastrun:
                args.insert(0, self.results.lastrun)
            else:
                raise self.InvalidCmd('No run name currently define. Please add this information.')             
        
        if len(args) >= 1:
            if args[0] != self.run_name and\
             not os.path.exists(pjoin(self.me_dir,'Events',args[0], 'unweighted_events.lhe.gz')):
                raise self.InvalidCmd('No events file corresponding to %s run. '% args[0])
            self.set_run_name(args[0], tag, 'pythia')
        else:
            if tag:
                self.run_card['run_tag'] = tag
            self.set_run_name(self.run_name, tag, 'pythia')

        input_file = pjoin(self.me_dir,'Events',self.run_name, 'unweighted_events.lhe')
        output_file = pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')
        if  not os.path.exists('%s.gz' % input_file):
            if not os.path.exists(input_file):
                raise self.InvalidCmd('No events file corresponding to %s run. '% self.run_name)
            files.ln(input_file, os.path.dirname(output_file))
        else:
            misc.gunzip(input_file, keep=True, stdout=output_file)
        
        args.append(mode)
    
    def check_remove(self, args):
        """Check that the remove command is valid"""

        tmp_args = args[:]

        tag = [a[6:] for a in tmp_args if a.startswith('--tag=')]
        if tag:
            tag = tag[0]
            tmp_args.remove('--tag=%s' % tag)


        if len(tmp_args) == 0:
            self.help_remove()
            raise self.InvalidCmd('clean command require the name of the run to clean')
        elif len(tmp_args) == 1:
            return tmp_args[0], tag, ['all']
        else:
            for arg in tmp_args[1:]:
                if arg not in self._clean_mode:
                    self.help_remove()
                    raise self.InvalidCmd('%s is not a valid options for clean command'\
                                              % arg)
            return tmp_args[0], tag, tmp_args[1:]

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
            error_msg = 'No valid MadAnalysis path set.\n'
            error_msg += 'Please use the set command to define the path and retry.\n'
            error_msg += 'You can also define it in the configuration file.\n'
            raise self.InvalidCmd(error_msg)  
        if not  td:
            error_msg = 'No valid td path set.\n'
            error_msg += 'Please use the set command to define the path and retry.\n'
            error_msg += 'You can also define it in the configuration file.\n'
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
    
    def check_syscalc(self, args):
        """Check the argument for the syscalc command
        syscalc run_name modes"""

        scdir = self.options['syscalc_path']
        
        if not scdir:
            logger.info('Retry to read configuration file to find SysCalc')
            self.set_configuration()

        scdir = self.options['syscalc_path']
        
        if not scdir:
            error_msg = 'No valid SysCalc path set.\n'
            error_msg += 'Please use the set command to define the path and retry.\n'
            error_msg += 'You can also define it in the configuration file.\n'
            error_msg += 'Please note that you need to compile SysCalc first.'
            raise self.InvalidCmd(error_msg)  
                     
        if len(args) == 0:
            if not hasattr(self, 'run_name') or not self.run_name:
                self.help_syscalc()
                raise self.InvalidCmd('No run name currently defined. Please add this information.')             
            args.append('all')
            return

        #deal options
        tag = [a for a in args if a.startswith('--tag=')]
        if tag: 
            args.remove(tag[0])
            tag = tag[0][6:]
        
        if args[0] not in self._syscalc_mode:
            self.set_run_name(args[0], tag=tag, level='syscalc')
            del args[0]
            if len(args) == 0:
                args.append('all')
        elif not self.run_name:
            self.help_syscalc()
            raise self.InvalidCmd('No run name currently defined. Please add this information.')                             
        elif tag and tag != self.run_tag:
            self.set_run_name(self.run_name, tag=tag, level='syscalc')
            
        for arg in args:
            if arg not in self._syscalc_mode and arg != self.run_name:
                 self.help_syscalc()
                 raise self.InvalidCmd('unknown options %s' % arg)        

        if self.run_card['use_syst'] not in self.true:
            raise self.InvalidCmd('Run %s does not include ' % self.run_name + \
                                  'systematics information needed for syscalc.')
        
    
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
            error_msg = 'No valid pythia-pgs path set.\n'
            error_msg += 'Please use the set command to define the path and retry.\n'
            error_msg += 'You can also define it in the configuration file.\n'
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
            if  not os.path.exists(pjoin(self.me_dir,'Events',self.run_name,'%s_pythia_events.hep.gz' % prev_tag)):
                raise self.InvalidCmd('No events file corresponding to %s run with tag %s. '% (self.run_name, prev_tag))
            else:
                input_file = pjoin(self.me_dir,'Events', self.run_name, '%s_pythia_events.hep.gz' % prev_tag)
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
            error_msg = 'No valid Delphes path set.\n'
            error_msg += 'Please use the set command to define the path and retry.\n'
            error_msg += 'You can also define it in the configuration file.\n'
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
            prev_tag = self.set_run_name(arg[0], tag, 'delphes')
            if  not os.path.exists(pjoin(self.me_dir,'Events',self.run_name, '%s_pythia_events.hep.gz' % prev_tag)):
                raise self.InvalidCmd('No events file corresponding to %s run with tag %s.:%s '\
                    % (self.run_name, prev_tag, 
                       pjoin(self.me_dir,'Events',self.run_name, '%s_pythia_events.hep.gz' % prev_tag)))
            else:
                input_file = pjoin(self.me_dir,'Events', self.run_name, '%s_pythia_events.hep.gz' % prev_tag)
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                lock = cluster.asyncrone_launch('gunzip',stdout=open(output_file,'w'), 
                                                    argument=['-c', input_file])
        else:
            if tag:
                self.run_card['run_tag'] = tag
            self.set_run_name(self.run_name, tag, 'delphes')
            
        return lock               

    def check_display(self, args):
        """check the validity of line
        syntax: display XXXXX
        """
            
        if len(args) < 1 or args[0] not in self._display_opts:
            self.help_display()
            raise self.InvalidCmd
        
        if args[0] == 'variable' and len(args) !=2:
            raise self.InvalidCmd('variable need a variable name')





    def check_import(self, args):
        """check the validity of line"""
         
        if not args:
            self.help_import()
            raise self.InvalidCmd('wrong \"import\" format')
        
        if args[0] != 'command':
            args.insert(0,'command')
        
        
        if not len(args) == 2 or not os.path.exists(args[1]):
            raise self.InvalidCmd('PATH is mandatory for import command\n')
        

#===============================================================================
# CompleteForCmd
#===============================================================================
class CompleteForCmd(CheckValidForCmd):
    """ The Series of help routine for the MadGraphCmd"""
    
    
    def complete_add_time_of_flight(self, text, line, begidx, endidx):
        "Complete command"
       
        args = self.split_arg(line[0:begidx], error=False)

        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*','unweighted_events.lhe.gz'))
            data = [n.rsplit('/',2)[1] for n in data]
            return  self.list_completion(text, data + ['--threshold='], line)
        elif args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        os.path.join('.',*[a for a in args \
                                                    if a.endswith(os.path.sep)]))
        else:
            return self.list_completion(text, ['--threshold='], line)
    
    def complete_banner_run(self, text, line, begidx, endidx):
       "Complete the banner run command"
       try:
  
        
        args = self.split_arg(line[0:begidx], error=False)
        
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        os.path.join('.',*[a for a in args \
                                                    if a.endswith(os.path.sep)]))        
        
        
        if len(args) > 1:
            # only options are possible
            tags = glob.glob(pjoin(self.me_dir, 'Events' , args[1],'%s_*_banner.txt' % args[1]))
            tags = ['%s' % os.path.basename(t)[len(args[1])+1:-11] for t in tags]

            if args[-1] != '--tag=':
                tags = ['--tag=%s' % t for t in tags]
            else:
                return self.list_completion(text, tags)
            return self.list_completion(text, tags +['--name=','-f'], line)
        
        # First argument
        possibilites = {} 

        comp = self.path_completion(text, os.path.join('.',*[a for a in args \
                                                    if a.endswith(os.path.sep)]))
        if os.path.sep in line:
            return comp
        else:
            possibilites['Path from ./'] = comp

        run_list =  glob.glob(pjoin(self.me_dir, 'Events', '*','*_banner.txt'))
        run_list = [n.rsplit('/',2)[1] for n in run_list]
        possibilites['RUN Name'] = self.list_completion(text, run_list)
        
        return self.deal_multiple_categories(possibilites)
    
        
       except Exception, error:
           print error


    def complete_history(self, text, line, begidx, endidx):
        "Complete the history command"

        args = self.split_arg(line[0:begidx], error=False)

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        os.path.join('.',*[a for a in args \
                                                    if a.endswith(os.path.sep)]))

        if len(args) == 1:
            return self.path_completion(text)
        
    def complete_open(self, text, line, begidx, endidx): 
        """ complete the open command """

        args = self.split_arg(line[0:begidx])
        
        # Directory continuation
        if os.path.sep in args[-1] + text:
            return self.path_completion(text,
                                    os.path.join('.',*[a for a in args if \
                                                      a.endswith(os.path.sep)]))

        possibility = []
        if self.me_dir:
            path = self.me_dir
            possibility = ['index.html']
            if os.path.isfile(os.path.join(path,'README')):
                possibility.append('README')
            if os.path.isdir(os.path.join(path,'Cards')):
                possibility += [f for f in os.listdir(os.path.join(path,'Cards')) 
                                    if f.endswith('.dat')]
            if os.path.isdir(os.path.join(path,'HTML')):
                possibility += [f for f in os.listdir(os.path.join(path,'HTML')) 
                                  if f.endswith('.html') and 'default' not in f]
        else:
            possibility.extend(['./','../'])
        if os.path.exists('ME5_debug'):
            possibility.append('ME5_debug')
        if os.path.exists('MG5_debug'):
            possibility.append('MG5_debug')
        return self.list_completion(text, possibility)
    
    def complete_set(self, text, line, begidx, endidx):
        "Complete the set command"

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            return self.list_completion(text, self._set_options + self.options.keys() )

        if len(args) == 2:
            if args[1] == 'stdout_level':
                return self.list_completion(text, ['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
            else:
                first_set = ['None','True','False']
                # directory names
                second_set = [name for name in self.path_completion(text, '.', only_dirs = True)]
                return self.list_completion(text, first_set + second_set)
        elif len(args) >2 and args[-1].endswith(os.path.sep):
                return self.path_completion(text,
                        os.path.join('.',*[a for a in args if a.endswith(os.path.sep)]),
                        only_dirs = True) 
    
    def complete_survey(self, text, line, begidx, endidx):
        """ Complete the survey command """
        
        if line.endswith('nb_core=') and not text:
            import multiprocessing
            max = multiprocessing.cpu_count()
            return [str(i) for i in range(2,max+1)]
            
        return  self.list_completion(text, self._run_options, line)
    
    complete_refine = complete_survey
    complete_combine_events = complete_survey
    complite_store = complete_survey
    complete_generate_events = complete_survey
    complete_create_gridpack = complete_survey
    
    def complete_generate_events(self, text, line, begidx, endidx):
        """ Complete the generate events"""
        
        if line.endswith('nb_core=') and not text:
            import multiprocessing
            max = multiprocessing.cpu_count()
            return [str(i) for i in range(2,max+1)]
        if line.endswith('laststep=') and not text:
            return ['parton','pythia','pgs','delphes']
        elif '--laststep=' in line.split()[-1] and line and line[-1] != ' ':
            return self.list_completion(text,['parton','pythia','pgs','delphes'],line)
        
        opts = self._run_options + self._generate_options
        return  self.list_completion(text, opts, line)


    def complete_initMadLoop(self, text, line, begidx, endidx):
        "Complete the initMadLoop command"
       
        numbers = [str(i) for i in range(10)]
        opts    = ['-f','-r','--nPS=']
       
        args = self.split_arg(line[0:begidx], error=False)
        if len(line) >=6 and line[begidx-6:begidx]=='--nPS=':
            return self.list_completion(text, numbers, line)
        else:
            return self.list_completion(text, [opt for opt in opts if not opt in
                                                                    line], line)

    def complete_launch(self, *args, **opts):

        if self.ninitial == 1:
            return self.complete_calculate_decay_widths(*args, **opts)
        else:
            return self.complete_generate_events(*args, **opts)

    def complete_calculate_decay_widths(self, text, line, begidx, endidx):
        """ Complete the calculate_decay_widths command"""
        
        if line.endswith('nb_core=') and not text:
            import multiprocessing
            max = multiprocessing.cpu_count()
            return [str(i) for i in range(2,max+1)]
        
        opts = self._run_options + self._calculate_decay_options
        return  self.list_completion(text, opts, line)
    
    def complete_display(self, text, line, begidx, endidx):
        """ Complete the display command"""    
        
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) >= 2 and args[1] =='results':
            start = line.find('results')
            return self.complete_print_results(text, 'print_results '+line[start+7:], begidx+2+start, endidx+2+start)
        return super(CompleteForCmd, self).complete_display(text, line, begidx, endidx)

    def complete_multi_run(self, text, line, begidx, endidx):
        """complete multi run command"""
        
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) == 1:
            data = [str(i) for i in range(0,20)]
            return  self.list_completion(text, data, line)
        
        if line.endswith('run=') and not text:
            return ['parton','pythia','pgs','delphes']
        elif '--laststep=' in line.split()[-1] and line and line[-1] != ' ':
            return self.list_completion(text,['parton','pythia','pgs','delphes'],line)
        
        opts = self._run_options + self._generate_options
        return  self.list_completion(text, opts, line)
        
        
        
        if line.endswith('nb_core=') and not text:
            import multiprocessing
            max = multiprocessing.cpu_count()
            return [str(i) for i in range(2,max+1)]
        opts = self._run_options + self._generate_options
        return  self.list_completion(text, opts, line)
    
    def complete_plot(self, text, line, begidx, endidx):
        """ Complete the plot command """
        
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) > 1:
            return self.list_completion(text, self._plot_mode)
        else:
            return self.list_completion(text, self._plot_mode + self.results.keys())
        
    def complete_syscalc(self, text, line, begidx, endidx):
        """ Complete the syscalc command """
        
        output = {}
        args = self.split_arg(line[0:begidx], error=False)
                
        if len(args) <=1:
            output['RUN_NAME'] = self.list_completion(self.results.keys())
        output['MODE'] =  self.list_completion(text, self._syscalc_mode)
        output['options'] = ['-f']
        if len(args) > 1 and (text.startswith('--t')):
            run = args[1]
            if run in self.results:
                tags = ['--tag=%s' % tag['tag'] for tag in self.results[run]]
                output['options'] += tags
        
        return self.deal_multiple_categories(output)
        
    def complete_remove(self, text, line, begidx, endidx):
        """Complete the remove command """
     
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) > 1 and (text.startswith('--t')):
            run = args[1]
            tags = ['--tag=%s' % tag['tag'] for tag in self.results[run]]
            return self.list_completion(text, tags)
        elif len(args) > 1 and '--' == args[-1]:
            run = args[1]
            tags = ['tag=%s' % tag['tag'] for tag in self.results[run]]
            return self.list_completion(text, tags)
        elif len(args) > 1 and '--tag=' == args[-1]:
            run = args[1]
            tags = [tag['tag'] for tag in self.results[run]]
            return self.list_completion(text, tags)
        elif len(args) > 1:
            return self.list_completion(text, self._clean_mode + ['-f','--tag='])
        else:
            data = glob.glob(pjoin(self.me_dir, 'Events','*','*_banner.txt'))
            data = [n.rsplit('/',2)[1] for n in data]
            return self.list_completion(text, ['all'] + data)
         
        
    def complete_pythia(self,text, line, begidx, endidx):
        "Complete the pythia command"
        args = self.split_arg(line[0:begidx], error=False)

        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*','unweighted_events.lhe.gz'))
            data = [n.rsplit('/',2)[1] for n in data]
            tmp1 =  self.list_completion(text, data)
            if not self.run_name:
                return tmp1
            else:
                tmp2 = self.list_completion(text, self._run_options + ['-f', 
                                                '--no_default', '--tag='], line)
                return tmp1 + tmp2
        elif line[-1] != '=':
            return self.list_completion(text, self._run_options + ['-f', 
                                                 '--no_default','--tag='], line)

    def complete_pgs(self,text, line, begidx, endidx):
        "Complete the pythia command"
        args = self.split_arg(line[0:begidx], error=False) 
        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*', '*_pythia_events.hep.gz'))
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





#===============================================================================
# MadEventCmd
#===============================================================================
class MadEventCmd(CompleteForCmd, CmdExtended, HelpToCmd, common_run.CommonRunCmd):

    """The command line processor of MadGraph"""    
    
    # Truth values
    true = ['T','.true.',True,'true']
    # Options and formats available
    _run_options = ['--cluster','--multicore','--nb_core=','--nb_core=2', '-c', '-m']
    _generate_options = ['-f', '--laststep=parton', '--laststep=pythia', '--laststep=pgs', '--laststep=delphes']
    _calculate_decay_options = ['-f', '--accuracy=0.']
    _set_options = ['stdout_level','fortran_compiler','timeout']
    _plot_mode = ['all', 'parton','pythia','pgs','delphes','channel', 'banner']
    _syscalc_mode = ['all', 'parton','pythia']
    _clean_mode = _plot_mode
    _display_opts = ['run_name', 'options', 'variable', 'results']
    _save_opts = ['options']
    _initMadLoop_opts = ['-f','-r','--nPS=']
    # survey options, dict from name to type, default value, and help text
    _survey_options = {'points':('int', 1000,'Number of points for first iteration'),
                       'iterations':('int', 5, 'Number of iterations'),
                       'accuracy':('float', 0.1, 'Required accuracy'),
                       'gridpack':('str', '.false.', 'Gridpack generation')}
    # Variables to store object information
    true = ['T','.true.',True,'true', 1, '1']
    web = False
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
    def __init__(self, me_dir = None, options={}, *completekey, **stdin):
        """ add information to the cmd """

        CmdExtended.__init__(self, me_dir, options, *completekey, **stdin)
        #common_run.CommonRunCmd.__init__(self, me_dir, options)

        self.mode = 'madevent'
        self.nb_refine=0
        if self.web:
            os.system('touch %s' % pjoin(self.me_dir,'Online'))

        
        # load the current status of the directory
        if os.path.exists(pjoin(self.me_dir,'HTML','results.pkl')):
            try:
                self.results = save_load_object.load_from_file(pjoin(self.me_dir,'HTML','results.pkl'))
            except Exception:
                #the pickle fail -> need to recreate the library
                model = self.find_model_name()
                process = self.process # define in find_model_name
                self.results = gen_crossxhtml.AllResults(model, process, self.me_dir)
                self.results.resetall(self.me_dir)
            else:                                
                self.results.resetall(self.me_dir)
        else:
            model = self.find_model_name()
            process = self.process # define in find_model_name
            self.results = gen_crossxhtml.AllResults(model, process, self.me_dir)
            self.results.resetall(self.me_dir)
        self.results.def_web_mode(self.web)
        
        self.prompt = "%s>"%os.path.basename(pjoin(self.me_dir))
        self.configured = 0 # time for reading the card
        self._options = {} # for compatibility with extended_cmd
        
    
    def pass_in_web_mode(self):
        """configure web data"""
        self.web = True
        self.results.def_web_mode(True)
        self.force = True
        if os.environ['MADGRAPH_BASE']:
            self.options['mg5_path'] = pjoin(os.environ['MADGRAPH_BASE'],'MG5')

    ############################################################################            
    def check_output_type(self, path):
        """ Check that the output path is a valid madevent directory """
        
        bin_path = os.path.join(path,'bin')
        if os.path.isfile(os.path.join(bin_path,'generate_events')):
            return True
        else: 
            return False

    ############################################################################
    def set_configuration(self, amcatnlo=False, final=True, **opt):
        """assign all configuration variable from file 
            loop over the different config file if config_file not define """
        
        super(MadEventCmd,self).set_configuration(amcatnlo=amcatnlo, 
                                                            final=final, **opt)

        if not final:
            return self.options # the return is usefull for unittest


        # Treat each expected input
        # delphes/pythia/... path
        # ONLY the ONE LINKED TO Madevent ONLY!!!
        for key in (k for k in self.options if k.endswith('path')):
            path = self.options[key]
            if path is None or key.startswith("cluster"):
                continue
            if not os.path.isdir(path):
                path = pjoin(self.me_dir, self.options[key])
            if os.path.isdir(path):
                self.options[key] = None
                if key == "pythia-pgs_path":
                    if not os.path.exists(pjoin(path, 'src','pythia')):
                        logger.info("No valid pythia-pgs path found")
                        continue
                elif key == "delphes_path":
                    if not os.path.exists(pjoin(path, 'Delphes')) and not\
                                     os.path.exists(pjoin(path, 'DelphesSTDHEP')):
                        logger.info("No valid Delphes path found")
                        continue
                elif key == "madanalysis_path":
                    if not os.path.exists(pjoin(path, 'plot_events')):
                        logger.info("No valid MadAnalysis path found")
                        continue
                elif key == "td_path":
                    if not os.path.exists(pjoin(path, 'td')):
                        logger.info("No valid td path found")
                        continue
                elif key == "syscalc_path":
                    if not os.path.exists(pjoin(path, 'sys_calc')):
                        logger.info("No valid SysCalc path found")
                        continue
                # No else since the next line reinitialize the option to the 
                #previous value anyway
                self.options[key] = os.path.realpath(path)
                continue
            else:
                self.options[key] = None
                
                          
        return self.options

    ############################################################################
    def do_add_time_of_flight(self, line):

        args = self.split_arg(line)
        #check the validity of the arguments and reformat args
        self.check_add_time_of_flight(args)
        
        event_path, threshold = args
        #gunzip the file
        if event_path.endswith('.gz'):
            need_zip = True
            misc.gunzip(event_path)
            event_path = event_path[:-3]
        else:
            need_zip = False
            
        import random
        try:
            import madgraph.various.lhe_parser as lhe_parser
        except:
            import internal.lhe_parser as lhe_parser 
            
        logger.info('Add time of flight information on file %s' % event_path)
        lhe = lhe_parser.EventFile(event_path)
        output = open('%s_2vertex.lhe' % event_path, 'w')
        #write the banner to the output file
        output.write(lhe.banner)

        # get the associate param_card
        begin_param = lhe.banner.find('<slha>')
        end_param = lhe.banner.find('</slha>')
        param_card = lhe.banner[begin_param+6:end_param].split('\n')
        param_card = check_param_card.ParamCard(param_card)

        cst = 6.58211915e-25 # hbar in GeV s
        c = 299792458000 # speed of light in mm/s
        # Loop over all events
        for event in lhe:
            for particle in event:
                id = particle.pid
                width = param_card['decay'].get((abs(id),)).value
                if width:
                    vtim = c * random.expovariate(width/cst)
                    if vtim > threshold:
                        particle.vtim = vtim
            #write this modify event
            output.write(str(event))
        output.write('</LesHouchesEvents>\n')
        output.close()
        
        files.mv('%s_2vertex.lhe' % event_path, event_path)
        
        if need_zip:
            misc.gzip(event_path)
        
    ############################################################################
    def do_banner_run(self, line): 
        """Make a run from the banner file"""
        
        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_banner_run(args)    
                     
        # Remove previous cards
        for name in ['delphes_trigger.dat', 'delphes_card.dat',
                     'pgs_card.dat', 'pythia_card.dat', 'madspin_card.dat',
                     'reweight_card.dat']:
            try:
                os.remove(pjoin(self.me_dir, 'Cards', name))
            except Exception:
                pass
            
        banner_mod.split_banner(args[0], self.me_dir, proc_card=False)
        
        # Check if we want to modify the run
        if not self.force:
            ans = self.ask('Do you want to modify the Cards?', 'n', ['y','n'])
            if ans == 'n':
                self.force = True
        
        # Call Generate events
        self.exec_cmd('generate_events %s %s' % (self.run_name, self.force and '-f' or ''))
 
 
 
    ############################################################################
    def do_display(self, line, output=sys.stdout):
        """Display current internal status"""

        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_display(args)

        if args[0] == 'run_name':
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*','*_banner.txt'))
            data = [n.rsplit('/',2)[1:] for n in data]
            
            if data:
                out = {}
                for name, tag in data:
                    tag = tag[len(name)+1:-11]
                    if name in out:
                        out[name].append(tag)
                    else:
                        out[name] = [tag]
                print 'the runs available are:'
                for run_name, tags in out.items():
                    print '  run: %s' % run_name
                    print '       tags: ', 
                    print ', '.join(tags)
            else:
                print 'No run detected.'
                
        elif  args[0] == 'options':
            outstr = "                              Run Options    \n"
            outstr += "                              -----------    \n"
            for key, default in self.options_madgraph.items():
                value = self.options[key]
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)
            outstr += "\n"
            outstr += "                         MadEvent Options    \n"
            outstr += "                         ----------------    \n"
            for key, default in self.options_madevent.items():
                if key in self.options:
                    value = self.options[key]
                else:
                    default = ''
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)  
            outstr += "\n"                 
            outstr += "                      Configuration Options    \n"
            outstr += "                      ---------------------    \n"
            for key, default in self.options_configuration.items():
                value = self.options[key]
                if value == default:
                    outstr += "  %25s \t:\t%s\n" % (key,value)
                else:
                    outstr += "  %25s \t:\t%s (user set)\n" % (key,value)
            output.write(outstr)
        elif  args[0] == 'results':
            self.do_print_results(' '.join(args[1:]))
        else:
            super(MadEventCmd, self).do_display(line, output)
 
    def do_save(self, line, check=True, to_keep={}):
        """Not in help: Save information to file"""  

        args = self.split_arg(line)
        # Check argument validity
        if check:
            self.check_save(args)
        
        if args[0] == 'options':
            # First look at options which should be put in MG5DIR/input
            to_define = {}
            for key, default in self.options_configuration.items():
                if self.options[key] != self.options_configuration[key]:
                    to_define[key] = self.options[key]

            if not '--auto' in args:
                for key, default in self.options_madevent.items():
                    if self.options[key] != self.options_madevent[key]:
                        to_define[key] = self.options[key]
            
            if '--all' in args:
                for key, default in self.options_madgraph.items():
                    if self.options[key] != self.options_madgraph[key]:
                        to_define[key] = self.options[key]
            elif not '--auto' in args:
                for key, default in self.options_madgraph.items():
                    if self.options[key] != self.options_madgraph[key]:
                        logger.info('The option %s is modified [%s] but will not be written in the configuration files.' \
                                    % (key,self.options_madgraph[key]) )
                        logger.info('If you want to make this value the default for future session, you can run \'save options --all\'')
            if len(args) >1 and not args[1].startswith('--'):
                filepath = args[1]
            else:
                filepath = pjoin(self.me_dir, 'Cards', 'me5_configuration.txt')
            basefile = pjoin(self.me_dir, 'Cards', 'me5_configuration.txt')
            basedir = self.me_dir
            
            if to_keep:
                to_define = to_keep
            self.write_configuration(filepath, basefile, basedir, to_define)
  
 


    def do_edit_cards(self, line):
        """Advanced commands: Basic edition of the cards"""
        args = self.split_arg(line)
        # Check argument's validity
        mode = self.check_generate_events(args)
        self.ask_run_configuration(mode)
        
        return

    ############################################################################

    ############################################################################
    def do_generate_events(self, line):
        """Main Commands: launch the full chain """
        
        self.banner = None
        args = self.split_arg(line)
        # Check argument's validity
        mode = self.check_generate_events(args)
        self.ask_run_configuration(mode, args)
        if not args:
            # No run name assigned -> assigned one automaticaly 
            self.set_run_name(self.find_available_run_name(self.me_dir), None, 'parton')
        else:
            self.set_run_name(args[0], None, 'parton', True)
            args.pop(0)
            
        if self.run_card['gridpack'] in self.true:        
            # Running gridpack warmup
            gridpack_opts=[('accuracy', 0.01),
                           ('points', 2000),
                           ('iterations',8),
                           ('gridpack','.true.')]
            logger.info('Generating gridpack with run name %s' % self.run_name)
            self.exec_cmd('survey  %s %s' % \
                          (self.run_name,
                           " ".join(['--' + opt + '=' + str(val) for (opt,val) \
                                     in gridpack_opts])),
                          postcmd=False)
            self.exec_cmd('combine_events', postcmd=False)
            self.exec_cmd('store_events', postcmd=False)
            self.exec_cmd('decay_events -from_cards', postcmd=False)
            self.exec_cmd('create_gridpack', postcmd=False)
        else:
            # Regular run mode
            logger.info('Generating %s events with run name %s' %
                        (self.run_card['nevents'], self.run_name))
        
            self.exec_cmd('survey  %s %s' % (self.run_name,' '.join(args)),
                          postcmd=False)
            nb_event = self.run_card['nevents']
            self.exec_cmd('refine %s' % nb_event, postcmd=False)
            if not float(self.results.current['cross']):
                # Zero cross-section. Try to guess why
                text = '''Survey return zero cross section. 
   Typical reasons are the following:
   1) A massive s-channel particle has a width set to zero.
   2) The pdf are zero for at least one of the initial state particles
      or you are using maxjetflavor=4 for initial state b:s.
   3) The cuts are too strong.
   Please check/correct your param_card and/or your run_card.'''
                logger_stderr.critical(text)
                raise ZeroResult('See https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-14')

            self.exec_cmd('refine %s' % nb_event, postcmd=False)
            
            self.exec_cmd('combine_events', postcmd=False)
            self.print_results_in_shell(self.results.current)

            
            self.run_syscalc('parton')
            self.create_plot('parton')            
            self.exec_cmd('store_events', postcmd=False)            
            self.exec_cmd('reweight -from_cards', postcmd=False)            
            self.exec_cmd('decay_events -from_cards', postcmd=False)
            if self.run_card['time_of_flight']>=0:
                self.exec_cmd("add_time_of_flight --threshold=%s" % self.run_card['time_of_flight'] ,postcmd=False)
            self.exec_cmd('pythia --no_default', postcmd=False, printcmd=False)
            # pythia launches pgs/delphes if needed    
            self.store_result()
            
            if self.param_card_iterator:
                param_card_iterator = self.param_card_iterator
                self.param_card_iterator = []
                with misc.TMP_variable(self, 'allow_notification_center', False):
                    param_card_iterator.store_entry(self.run_name, self.results.current['cross'])
                    #check if the param_card defines a scan.
                    orig_name = self.run_name
                    for card in param_card_iterator:
                        card.write(pjoin(self.me_dir,'Cards','param_card.dat'))
                        self.exec_cmd("generate_events -f ",precmd=True, postcmd=True,errorhandling=False)
                        param_card_iterator.store_entry(self.run_name, self.results.current['cross'])
                    param_card_iterator.write(pjoin(self.me_dir,'Cards','param_card.dat'))
                    name = misc.get_scan_name(orig_name, self.run_name)
                    path = pjoin(self.me_dir, 'Events','scan_%s.txt' % name)
                    logger.info("write all cross-section results in %s" % path ,'$MG:color:BLACK')
                    param_card_iterator.write_summary(path)

            
            if self.allow_notification_center:    
                misc.apple_notify('Run %s finished' % os.path.basename(self.me_dir), 
                              '%s: %s +- %s ' % (self.results.current['run_name'], 
                                                 self.results.current['cross'],
                                                 self.results.current['error']))
    
    def do_initMadLoop(self,line):
        """Compile and run MadLoop for a certain number of PS point so as to 
        initialize MadLoop (setup the zero helicity and loop filter.)"""
        
        args = line.split()
        # Check argument's validity
        options = self.check_initMadLoop(args)
        
        if not options['force']:
            self.ask_edit_cards(['MadLoopParams.dat'], mode='fixed', plot=False)
            self.exec_cmd('treatcards loop --no_MadLoopInit')

        if options['refresh']:
            for filter in glob.glob(pjoin(
                   self.me_dir,'SubProcesses','MadLoop5_resources','*Filter*')):
                logger.debug("Resetting filter '%s'."%os.path.basename(filter))
                os.remove(filter)

        MLCard = banner_mod.MadLoopParam(pjoin(self.me_dir,
                                                   'Cards','MadLoopParams.dat'))
        if options['nPS'] is None:
            options['nPS'] = MLCard['CheckCycle']+2
        elif options['nPS'] < MLCard['CheckCycle']+2:
            new_n_PS = MLCard['CheckCycle']+2
            logger.debug('Hard-setting user-defined n_PS (%d) to %d, because '\
              %(options['nPS'],new_n_PS)+"of the 'CheckCycle' value (%d) "%MLCard['CheckCycle']+\
                                             "specified in the ML param card.")
            options['nPS'] = new_n_PS

        MadLoopInitializer.init_MadLoop(self.me_dir,n_PS=options['nPS'],
                   subproc_prefix='PV', MG_options=self.options, interface=self)

    def do_launch(self, line, *args, **opt):
        """Main Commands: exec generate_events for 2>N and calculate_width for 1>N"""
        if self.ninitial == 1:
            logger.info("Note that since 2.3. The launch for 1>N pass in event generation\n"+
                           "    To have the previous behavior use the calculate_decay_widths function")
            self.do_calculate_decay_widths(line, *args, **opt)
        else:
            self.do_generate_events(line, *args, **opt)
            
    def print_results_in_shell(self, data):
        """Have a nice results prints in the shell,
        data should be of type: gen_crossxhtml.OneTagResults"""

        if not data:
            return

        if data['run_statistics']:
            globalstat = sum_html.RunStatistics()
            
            logger.info(" " )
            logger.debug(" === Run statistics summary ===")
            for key, value in data['run_statistics'].items():
                globalstat.aggregate_statistics(value)
                level = 5
                if value.has_warning():
                    level = 10
                logger.log(level, value.nice_output(str('/'.join([key[0],'G%s'%key[1]]))).\
                  replace(' statistics',''))
            logger.info(" " )
            logger.debug(globalstat.nice_output('combined', no_warning=True))
            if globalstat.has_warning():
                logger.warning(globalstat.get_warning_text())
            logger.info(" ")

            
        logger.info("  === Results Summary for run: %s tag: %s ===\n" % (data['run_name'],data['tag']))
        
        total_time = int(sum(_['cumulative_timing'] for _ in data['run_statistics'].values()))
        if total_time > 0:
            logger.info("     Cumulative sequential time for this run: %s"%misc.format_time(total_time))
        
        if self.ninitial == 1:
            logger.info("     Width :   %.4g +- %.4g GeV" % (data['cross'], data['error']))
        else:
            logger.info("     Cross-section :   %.4g +- %.4g pb" % (data['cross'], data['error']))
        logger.info("     Nb of events :  %s" % data['nb_event'] )
        if data['cross_pythia'] and data['nb_event_pythia']:
            if self.ninitial == 1:
                logger.info("     Matched Width :   %.4g +- %.4g GeV" % (data['cross_pythia'], data['error_pythia']))
            else:
                logger.info("     Matched Cross-section :   %.4g +- %.4g pb" % (data['cross_pythia'], data['error_pythia']))            
            logger.info("     Nb of events after Matching :  %s" % data['nb_event_pythia'])
            if self.run_card['use_syst'] in self.true:
                logger.info("     Be carefull that matched information are here NOT for the central value. Refer to SysCalc output for it")
            
        logger.info(" " )

    def print_results_in_file(self, data, path, mode='w', format='full'):
        """Have a nice results prints in the shell,
        data should be of type: gen_crossxhtml.OneTagResults"""
        if not data:
            return
        
        fsock = open(path, mode)
        
        if data['run_statistics']:
            logger.debug(" === Run statistics summary ===")
            for key, value in data['run_statistics'].items():
                logger.debug(value.nice_output(str('/'.join([key[0],'G%s'%key[1]]))).\
                  replace(' statistics',''))
            logger.info(" " )

        if format == "full":
            fsock.write("  === Results Summary for run: %s tag: %s  process: %s ===\n" % \
                        (data['run_name'],data['tag'], os.path.basename(self.me_dir)))
            
            if self.ninitial == 1:
                fsock.write("     Width :   %.4g +- %.4g GeV\n" % (data['cross'], data['error']))
            else:
                fsock.write("     Cross-section :   %.4g +- %.4g pb\n" % (data['cross'], data['error']))
            fsock.write("     Nb of events :  %s\n" % data['nb_event'] )
            if data['cross_pythia'] and data['nb_event_pythia']:
                if self.ninitial == 1:
                    fsock.write("     Matched Width :   %.4g +- %.4g GeV\n" % (data['cross_pythia'], data['error_pythia']))
                else:
                    fsock.write("     Matched Cross-section :   %.4g +- %.4g pb\n" % (data['cross_pythia'], data['error_pythia']))            
                fsock.write("     Nb of events after Matching :  %s\n" % data['nb_event_pythia'])
            fsock.write(" \n" )
        elif format == "short":
            if mode == "w":
                fsock.write("# run_name tag cross error Nb_event cross_after_matching nb_event_after matching\n")
                
            if data['cross_pythia'] and data['nb_event_pythia']:
                text = "%(run_name)s %(tag)s %(cross)s %(error)s %(nb_event)s %(cross_pythia)s %(nb_event_pythia)s\n"
            else:
                text = "%(run_name)s %(tag)s %(cross)s %(error)s %(nb_event)s\n"
            fsock.write(text % data)
                
    ############################################################################      
    def do_calculate_decay_widths(self, line):
        """Main Commands: launch decay width calculation and automatic inclusion of
        calculated widths and BRs in the param_card."""

        args = self.split_arg(line)
        # Check argument's validity
        accuracy = self.check_calculate_decay_widths(args)
        self.ask_run_configuration('parton')
        self.banner = None
        if not args:
            # No run name assigned -> assigned one automaticaly 
            self.set_run_name(self.find_available_run_name(self.me_dir))
        else:
            self.set_run_name(args[0], reload_card=True)
            args.pop(0)

        self.configure_directory()
        
        # Running gridpack warmup
        opts=[('accuracy', accuracy), # default 0.01
              ('points', 1000),
              ('iterations',9)]

        logger.info('Calculating decay widths with run name %s' % self.run_name)
        
        self.exec_cmd('survey  %s %s' % \
                      (self.run_name,
                       " ".join(['--' + opt + '=' + str(val) for (opt,val) \
                                 in opts])),
                      postcmd=False)
        self.refine_mode = "old" # specify how to combine event
        self.exec_cmd('combine_events', postcmd=False)
        self.exec_cmd('store_events', postcmd=False)
        
        self.collect_decay_widths()
        self.print_results_in_shell(self.results.current)
        self.update_status('calculate_decay_widths done', 
                                                 level='parton', makehtml=False)   

    
    ############################################################################
    def collect_decay_widths(self):
        """ Collect the decay widths and calculate BRs for all particles, and put 
        in param_card form. 
        """
        
        particle_dict = {} # store the results
        run_name = self.run_name

        # Looping over the Subprocesses
        for P_path in SubProcesses.get_subP(self.me_dir):
            ids = SubProcesses.get_subP_ids(P_path)
            # due to grouping we need to compute the ratio factor for the 
            # ungroup resutls (that we need here). Note that initial particles
            # grouping are not at the same stage as final particle grouping
            nb_output = len(ids) / (len(set([p[0] for p in ids])))
            results = open(pjoin(P_path, run_name + '_results.dat')).read().split('\n')[0]
            result = float(results.strip().split(' ')[0])
            for particles in ids:
                try:
                    particle_dict[particles[0]].append([particles[1:], result/nb_output])
                except KeyError:
                    particle_dict[particles[0]] = [[particles[1:], result/nb_output]]
    
        self.update_width_in_param_card(particle_dict,
                        initial = pjoin(self.me_dir, 'Cards', 'param_card.dat'),
                        output=pjoin(self.me_dir, 'Events', run_name, "param_card.dat"))
    
    @staticmethod
    def update_width_in_param_card(decay_info, initial=None, output=None):
        # Open the param_card.dat and insert the calculated decays and BRs
        
        if not output:
            output = initial
        
        param_card_file = open(initial)
        param_card = param_card_file.read().split('\n')
        param_card_file.close()

        decay_lines = []
        line_number = 0
        # Read and remove all decays from the param_card                     
        while line_number < len(param_card):
            line = param_card[line_number]
            if line.lower().startswith('decay'):
                # Read decay if particle in decay_info 
                # DECAY  6   1.455100e+00                                    
                line = param_card.pop(line_number)
                line = line.split()
                particle = 0
                if int(line[1]) not in decay_info:
                    try: # If formatting is wrong, don't want this particle
                        particle = int(line[1])
                        width = float(line[2])
                    except Exception:
                        particle = 0
                # Read BRs for this decay
                line = param_card[line_number]
                while line.startswith('#') or line.startswith(' '):
                    line = param_card.pop(line_number)
                    if not particle or line.startswith('#'):
                        line=param_card[line_number]
                        continue
                    #    6.668201e-01   3    5  2  -1
                    line = line.split()
                    try: # Remove BR if formatting is wrong
                        partial_width = float(line[0])*width
                        decay_products = [int(p) for p in line[2:2+int(line[1])]]
                    except Exception:
                        line=param_card[line_number]
                        continue
                    try:
                        decay_info[particle].append([decay_products, partial_width])
                    except KeyError:
                        decay_info[particle] = [[decay_products, partial_width]]
                    if line_number == len(param_card):
                        break
                    line=param_card[line_number]
                if particle and particle not in decay_info:
                    # No decays given, only total width       
                    decay_info[particle] = [[[], width]]
            else: # Not decay
                line_number += 1
        # Clean out possible remaining comments at the end of the card
        while not param_card[-1] or param_card[-1].startswith('#'):
            param_card.pop(-1)

        # Append calculated and read decays to the param_card   
        param_card.append("#\n#*************************")
        param_card.append("#      Decay widths      *")
        param_card.append("#*************************")
        for key in sorted(decay_info.keys()):
            width = sum([r for p,r in decay_info[key]])
            param_card.append("#\n#      PDG        Width")
            param_card.append("DECAY  %i   %e" % (key, width.real))
            if not width:
                continue
            if decay_info[key][0][0]:
                param_card.append("#  BR             NDA  ID1    ID2   ...")
                brs = [[(val[1]/width).real, val[0]] for val in decay_info[key] if val[1]]
                for val in sorted(brs, reverse=True):
                    param_card.append("   %e   %i    %s # %s" % 
                                      (val[0].real, len(val[1]),
                                       "  ".join([str(v) for v in val[1]]),
                                       val[0] * width
                                       ))
        decay_table = open(output, 'w')
        decay_table.write("\n".join(param_card) + "\n")
        decay_table.close()
        logger.info("Results written to %s" %  output)


    ############################################################################
    def do_multi_run(self, line):
        
        args = self.split_arg(line)
        # Check argument's validity
        mode = self.check_multi_run(args)
        nb_run = args.pop(0)
        if nb_run == 1:
            logger.warn("'multi_run 1' command is not optimal. Think of using generate_events instead")
        self.ask_run_configuration(mode)
        main_name = self.run_name

        # check if the param_card requires a scan over parameter.
        path=pjoin(self.me_dir, 'Cards', 'param_card.dat')
        self.check_param_card(path, run=False)
        #store it locally to avoid relaunch
        param_card_iterator, self.param_card_iterator = self.param_card_iterator, []
        
        crossoversig = 0
        inv_sq_err = 0
        nb_event = 0
        for i in range(nb_run):
            self.nb_refine = 0
            self.exec_cmd('generate_events %s_%s -f' % (main_name, i), postcmd=False)
            # Update collected value
            nb_event += int(self.results[self.run_name][-1]['nb_event'])  
            self.results.add_detail('nb_event', nb_event , run=main_name)            
            cross = self.results[self.run_name][-1]['cross']
            error = self.results[self.run_name][-1]['error'] + 1e-99
            crossoversig+=cross/error**2
            inv_sq_err+=1.0/error**2
            self.results[main_name][-1]['cross'] = crossoversig/inv_sq_err
            self.results[main_name][-1]['error'] = math.sqrt(1.0/inv_sq_err)
        self.results.def_current(main_name)
        self.run_name = main_name
        self.update_status("Merging LHE files", level='parton')
        try:
            os.mkdir(pjoin(self.me_dir,'Events', self.run_name))
        except Exception:
            pass
        os.system('%(bin)s/merge.pl %(event)s/%(name)s_*/unweighted_events.lhe.gz %(event)s/%(name)s/unweighted_events.lhe.gz %(event)s/%(name)s_banner.txt' 
                  % {'bin': self.dirbin, 'event': pjoin(self.me_dir,'Events'),
                     'name': self.run_name})

        eradir = self.options['exrootanalysis_path']
        if eradir and misc.is_executable(pjoin(eradir,'ExRootLHEFConverter')):
            self.update_status("Create Root file", level='parton')
            misc.gunzip('%s/%s/unweighted_events.lhe.gz' % 
                                  (pjoin(self.me_dir,'Events'), self.run_name))

            self.create_root_file('%s/unweighted_events.lhe' % self.run_name,
                                  '%s/unweighted_events.root' % self.run_name)
            
        path = pjoin(self.me_dir, "Events", self.run_name, "unweighted_events.lhe")        
        self.create_plot('parton', path,
                         pjoin(self.me_dir, 'HTML',self.run_name, 'plots_parton.html')
                         )
        

        if not os.path.exists('%s.gz' % path):        
            misc.gzip(path)

        self.update_status('', level='parton')
        self.print_results_in_shell(self.results.current)   
        
        if param_card_iterator:

            param_card_iterator.store_entry(self.run_name, self.results.current['cross'])
            #check if the param_card defines a scan.
            orig_name=self.run_name
            for card in param_card_iterator:
                card.write(pjoin(self.me_dir,'Cards','param_card.dat'))
                self.exec_cmd("multi_run %s -f " % nb_run ,precmd=True, postcmd=True,errorhandling=False)
                param_card_iterator.store_entry(self.run_name, self.results.current['cross'])
            param_card_iterator.write(pjoin(self.me_dir,'Cards','param_card.dat'))
            scan_name = misc.get_scan_name(orig_name, self.run_name)
            path = pjoin(self.me_dir, 'Events','scan_%s.txt' % scan_name)
            logger.info("write all cross-section results in %s" % path, '$MG:color:BLACK')
            param_card_iterator.write_summary(path)
    

    ############################################################################      
    def do_treatcards(self, line, mode=None, opt=None):
        """Advanced commands: create .inc files from param_card.dat/run_card.dat"""

        if not mode and not opt:
            args = self.split_arg(line)
            mode,  opt  = self.check_treatcards(args)
            
        # To decide whether to refresh MadLoop's helicity filters, it is necessary
        # to check if the model parameters where modified or not, before doing
        # anything else.        
        need_MadLoopFilterUpdate = False
        # Just to record what triggered the reinitialization of MadLoop for a 
        # nice debug message.
        type_of_change           = ''
        if not opt['forbid_MadLoopInit'] and self.proc_characteristics['loop_induced'] \
                                                    and mode in ['loop', 'all']:
            paramDat = pjoin(self.me_dir, 'Cards','param_card.dat')
            paramInc = pjoin(opt['output_dir'], 'param_card.inc')
            if (not os.path.isfile(paramDat)) or (not os.path.isfile(paramInc)) or \
               (os.path.getmtime(paramDat)-os.path.getmtime(paramInc)) > 0.0:
                need_MadLoopFilterUpdate = True
                type_of_change           = 'model'
            
            ML_in = pjoin(self.me_dir, 'Cards', 'MadLoopParams.dat')
            ML_out = pjoin(self.me_dir,"SubProcesses",
                                      "MadLoop5_resources", "MadLoopParams.dat")         
            if (not os.path.isfile(ML_in)) or (not os.path.isfile(ML_out)) or \
                (os.path.getmtime(ML_in)-os.path.getmtime(ML_out)) > 0.0:
                need_MadLoopFilterUpdate = True
                type_of_change           = 'MadLoop'

        #check if no 'Auto' are present in the file
        self.check_param_card(pjoin(self.me_dir, 'Cards','param_card.dat'))
        
        if mode in ['param', 'all']:
            model = self.find_model_name()
            tmp_model = os.path.basename(model)
            if tmp_model == 'mssm' or tmp_model.startswith('mssm-'):
                if not '--param_card=' in line:
                    param_card = pjoin(self.me_dir, 'Cards','param_card.dat')
                    mg5_param = pjoin(self.me_dir, 'Source', 'MODEL', 'MG5_param.dat')
                    check_param_card.convert_to_mg5card(param_card, mg5_param)
                    check_param_card.check_valid_param_card(mg5_param)
                    opt['param_card'] = pjoin(self.me_dir, 'Source', 'MODEL', 'MG5_param.dat')
            else:
                check_param_card.check_valid_param_card(opt['param_card'])            
            
            logger.debug('write compile file for card: %s' % opt['param_card']) 
            param_card = check_param_card.ParamCard(opt['param_card'])
            outfile = pjoin(opt['output_dir'], 'param_card.inc')
            ident_card = pjoin(self.me_dir,'Cards','ident_card.dat')
            if os.path.isfile(pjoin(self.me_dir,'bin','internal','ufomodel','restrict_default.dat')):
                default = pjoin(self.me_dir,'bin','internal','ufomodel','restrict_default.dat')
            elif os.path.isfile(pjoin(self.me_dir,'bin','internal','ufomodel','param_card.dat')):
                default = pjoin(self.me_dir,'bin','internal','ufomodel','param_card.dat')
            elif not os.path.exists(pjoin(self.me_dir,'bin','internal','ufomodel')):
                fsock = open(pjoin(self.me_dir,'Source','param_card.inc'),'w')
                fsock.write(' ')
                fsock.close()
                if mode == 'all':
                    self.do_treatcards('', 'run', opt)
                return
            else:
                devnull = open(os.devnull,'w')
                subprocess.call([sys.executable, 'write_param_card.py'],
                             cwd=pjoin(self.me_dir,'bin','internal','ufomodel'),
                             stdout=devnull)
                devnull.close()
                default = pjoin(self.me_dir,'bin','internal','ufomodel','param_card.dat')

            need_mp = self.proc_characteristics['loop_induced']                
            param_card.write_inc_file(outfile, ident_card, default, need_mp=need_mp)
      
      
        if mode in ['run', 'all']:
            if not hasattr(self, 'run_card'):
                run_card = banner_mod.RunCard(opt['run_card'])
            else:
                run_card = self.run_card
            if self.ninitial == 1:
                run_card['lpp1'] =  0
                run_card['lpp2'] =  0
                run_card['ebeam1'] = 0
                run_card['ebeam2'] = 0
            
            run_card.write_include_file(pjoin(opt['output_dir'],'run_card.inc'))
        

        if self.proc_characteristics['loop_induced'] and mode in ['loop', 'all']:
            self.MadLoopparam = banner_mod.MadLoopParam(pjoin(self.me_dir, 
                                                  'Cards', 'MadLoopParams.dat'))
            # The writing out of MadLoop filter is potentially dangerous
            # when running in multi-core with a central disk. So it is turned
            # off here. If these filters were not initialized then they will 
            # have to be re-computed at the beginning of each run.
            if 'WriteOutFilters' in self.MadLoopparam.user_set and \
                                       self.MadLoopparam.get('WriteOutFilters'):
                logger.info(
"""You chose to have MadLoop writing out filters. 
Beware that this can be dangerous for local multicore runs.""")
            self.MadLoopparam.set('WriteOutFilters',False, ifnotdefault=False)
            
            # The conservative settings below for 'CTModeInit' and 'ZeroThres'
            # help adress issues for processes like g g > h z, and g g > h g
            # where there are some helicity configuration heavily suppressed 
            # (by several orders of magnitude) so that the helicity filter 
            # needs high numerical accuracy to correctly handle this spread in
            # magnitude. Also, because one cannot use the Born as a reference
            # scale, it is better to force quadruple precision *for the 
            # initialization points only*. This avoids numerical accuracy issues
            # when setting up the helicity filters and does not significantly
            # slow down the run.
#            self.MadLoopparam.set('CTModeInit',4, ifnotdefault=False)
            # Consequently, we can allow for a finer threshold for vanishing
            # helicity configuration
#            self.MadLoopparam.set('ZeroThres',1.0e-11, ifnotdefault=False)

#           It is a bit superficial to use the level 2 which tries to numerically
#           map matching helicities (because of CP symmetry typically) together.
#           It is useless in the context of MC over helicities and it can 
#           potentially make the helicity double checking fail.
            self.MadLoopparam.set('HelicityFilterLevel',1, ifnotdefault=False)

#           To be on the safe side however, we ask for 4 consecutive matching
#           helicity filters.
            self.MadLoopparam.set('CheckCycle',4, ifnotdefault=False)
            
            # For now it is tricky to have eahc channel performing the helicity
            # double check. What we will end up doing is probably some kind
            # of new initialization round at the beginning of each launch
            # command, to reset the filters.    
            self.MadLoopparam.set('DoubleCheckHelicityFilter',False,
                                                             ifnotdefault=False)
          
            # Thanks to TIR recycling, TIR is typically much faster for Loop-induced
            # processes, so that we place OPP last.
            if not hasattr(self, 'run_card'):
                run_card = banner_mod.RunCard(opt['run_card'])
            else:
                run_card = self.run_card
            if run_card['nhel'] == 0:
                if 'MLReductionLib' in self.MadLoopparam.user_set and \
                            self.MadLoopparam.get('MLReductionLib').startswith('1'):
                    logger.warning(
    """You chose to set the preferred reduction technique in MadLoop to be OPP (see parameter MLReductionLib).
    Beware that this can bring significant slowdown; the optimal choice --when not MC over helicity-- being to first start with TIR reduction.""")
                # We do not include GOLEM for now since it cannot recycle TIR coefs yet.
                self.MadLoopparam.set('MLReductionLib','2|3|1', ifnotdefault=False)
            else:
                if 'MLReductionLib' in self.MadLoopparam.user_set and \
                    not self.MadLoopparam.get('MLReductionLib').startswith('1'):
                    logger.warning(
    """You chose to set the preferred reduction technique in MadLoop to be different than OPP (see parameter MLReductionLib).
    Beware that this can bring significant slowdown; the optimal choice --when MC over helicity-- being to first start with OPP reduction.""")
                self.MadLoopparam.set('MLReductionLib','1|2|3|4', ifnotdefault=False)

            # Also TIR cache will only work when NRotations_DP=0 (but only matters
            # when not MC-ing over helicities) so it will be hard-reset by MadLoop
            # to zero when not MC-ing over helicities, unless the parameter
            # Force_ML_Helicity_Sum is set to True in the matrix<i>.f codes.
            if run_card['nhel'] == 0:
                if ('NRotations_DP' in self.MadLoopparam.user_set and \
                                     self.MadLoopparam.get('NRotations_DP')!=0) or \
                   ('NRotations_QP' in self.MadLoopparam.user_set and \
                                         self.MadLoopparam.get('NRotations_QP')!=0):
                    logger.warning(
    """You chose to also use a lorentz rotation for stability tests (see parameter NRotations_[DP|QP]).
    Beware that, for optimization purposes, MadEvent uses manual TIR cache clearing which is not compatible
    with the lorentz rotation stability test. The number of these rotations to be used will be reset to 
    zero by MadLoop. You can avoid this by changing the parameter 'FORCE_ML_HELICITY_SUM' int he matrix<i>.f
    files to be .TRUE. so that the sum over helicity configurations is performed within MadLoop (in which case
    the helicity of final state particles cannot be speicfied in the LHE file.""")
                self.MadLoopparam.set('NRotations_DP',0,ifnotdefault=False)
                self.MadLoopparam.set('NRotations_QP',0,ifnotdefault=False)
            else:
                # When MC-ing over helicities, the manual TIR cache clearing is
                # not necessary, so that one can use the lorentz check
                # Using NRotations_DP=1 slows down the code by close to 100%
                # but it is typicaly safer.
                # self.MadLoopparam.set('NRotations_DP',0,ifnotdefault=False)
                # Revert to the above to be slightly less robust but twice faster.
                self.MadLoopparam.set('NRotations_DP',1,ifnotdefault=False)
                self.MadLoopparam.set('NRotations_QP',0,ifnotdefault=False)                
            
            # Finally, the stability tests are slightly less reliable for process
            # with less or equal than 4 final state particles because the 
            # accessible kinematic is very limited (i.e. lorentz rotations don't
            # shuffle invariants numerics much). In these cases, we therefore
            # increase the required accuracy to 10^-7.
            # This is important for getting g g > z z [QCD] working with a
            # ptheavy cut as low as 1 GeV.  
            if self.proc_characteristics['nexternal']<=4:
                if ('MLStabThres' in self.MadLoopparam.user_set and \
                                   self.MadLoopparam.get('MLStabThres')>1.0e-7):
                    logger.warning(
    """You chose to increase the default value of the MadLoop parameter 'MLStabThres' above 1.0e-7.
    Stability tests can be less reliable on the limited kinematic of processes with less or equal
    than four external legs, so this is not recommended (especially not for g g > z z).""")
                self.MadLoopparam.set('MLStabThres',1.0e-7,ifnotdefault=False)
            else:
                self.MadLoopparam.set('MLStabThres',1.0e-4,ifnotdefault=False)            

            #write the output file
            self.MadLoopparam.write(pjoin(self.me_dir,"SubProcesses","MadLoop5_resources",
                                          "MadLoopParams.dat"))
            
        if self.proc_characteristics['loop_induced'] and mode in ['loop', 'all']:
            # Now Update MadLoop filters if necessary (if modifications were made to
            # the model parameters).
            if need_MadLoopFilterUpdate:
                logger.debug('Changes to the %s parameters'%type_of_change+\
                  ' have been detected. Madevent will then now reinitialize'+\
                                                            ' MadLoop filters.')
                self.exec_cmd('initMadLoop -r -f')
            # The need_MadLoopInit condition is just there so as to avoid useless
            # printout if there is not initialization to be performed. But even
            # without it, and because we call 'initMadLoop' without the '-r' option
            # no time would be wasted anyway, since the existing filters would not
            # be overwritten.
            elif not opt['forbid_MadLoopInit'] and \
                                   MadLoopInitializer.need_MadLoopInit(self.me_dir):
                self.exec_cmd('initMadLoop -f')
         
    ############################################################################      
    def do_survey(self, line):
        """Advanced commands: launch survey for the current process """
        
          
        args = self.split_arg(line)
        # Check argument's validity
        self.check_survey(args)
        # initialize / remove lhapdf mode

        if os.path.exists(pjoin(self.me_dir,'error')):
            os.remove(pjoin(self.me_dir,'error'))
                        
        self.configure_directory()
        # Save original random number
        self.random_orig = self.random
        logger.info("Using random number seed offset = %s" % self.random)
        # Update random number
        self.update_random()
        self.save_random()
        self.update_status('Running Survey', level=None)
        if self.cluster_mode:
            logger.info('Creating Jobs')

        self.total_jobs = 0
        subproc = [l.strip() for l in open(pjoin(self.me_dir,
                                                 'SubProcesses', 'subproc.mg'))]

        P_zero_result = [] # check the number of times where they are no phase-space

        # File for the loop (for loop induced)
        if os.path.exists(pjoin(self.me_dir,'SubProcesses', 
                 'MadLoop5_resources')) and cluster.need_transfer(self.options):
            tf=tarfile.open(pjoin(self.me_dir, 'SubProcesses', 
                         'MadLoop5_resources.tar.gz'), 'w:gz', dereference=True)
            tf.add(pjoin(self.me_dir,'SubProcesses','MadLoop5_resources'),
                                                   arcname='MadLoop5_resources')
            tf.close()

        logger.info('Working on SubProcesses')
        ajobcreator = gen_ximprove.gensym(self)

        #check difficult PS case
        if float(self.run_card['mmjj']) > 0.01 * (float(self.run_card['ebeam1'])+float(self.run_card['ebeam2'])):
            self.pass_in_difficult_integration_mode()
        
        jobs, P_zero_result = ajobcreator.launch()
        
        # Check if all or only some fails
        if P_zero_result:
            if len(P_zero_result) == len(subproc):
                Pdir = pjoin(self.me_dir, 'SubProcesses',subproc[0].strip())
                raise ZeroResult, '%s' % \
                    open(pjoin(Pdir,'ajob.no_ps.log')).read()
            else:
                logger.warning(''' %s SubProcesses doesn\'t have available phase-space.
            Please check mass spectrum.''' % ','.join(P_zero_result))
                
        
        self.monitor(run_type='All jobs submitted for survey', html=True)
        if not self.history or 'survey' in self.history[-1] or self.ninitial ==1  or \
           self.run_card['gridpack']:
            #will be done during the refine (more precisely in gen_ximprove)
            cross, error = sum_html.make_all_html_results(self)
            self.results.add_detail('cross', cross)
            self.results.add_detail('error', error)  
            self.exec_cmd("print_results %s" % self.run_name,
                           errorhandling=False, printcmd=False, precmd=False, postcmd=False)      
        
        self.results.add_detail('run_statistics', dict(ajobcreator.run_statistics))
        self.update_status('End survey', 'parton', makehtml=False)

    ############################################################################
    def pass_in_difficult_integration_mode(self):
        """be more secure for the integration to not miss it due to strong cut"""
        
        # improve survey options if default
        if self.opts['points'] == self._survey_options['points'][1]:
            self.opts['points'] = 2 * self._survey_options['points'][1]
        if self.opts['iterations'] == self._survey_options['iterations'][1]:
            self.opts['iterations'] = 1 + self._survey_options['iterations'][1]
        if self.opts['accuracy'] == self._survey_options['accuracy'][1]:
            self.opts['accuracy'] = self._survey_options['accuracy'][1]/2  
        
        # Modify run_config.inc in order to improve the refine
        #conf_path = pjoin(self.me_dir, 'Source','run_config.inc')
        #files.cp(conf_path, conf_path + '.bk')
        #
        #text = open(conf_path).read()
        #text = re.sub('''\(min_events = \d+\)''', '''(min_events = 7500 )''', text)
        #text = re.sub('''\(max_events = \d+\)''', '''(max_events = 20000 )''', text)
        #fsock = open(conf_path, 'w')
        #fsock.write(text)
        #fsock.close()
        
        # Compile
        for name in ['../bin/internal/gen_ximprove', 'all', 
                     '../bin/internal/combine_events']:
            self.compile(arg=[name], cwd=os.path.join(self.me_dir, 'Source'))
        
        
    ############################################################################      
    def do_refine(self, line):
        """Advanced commands: launch survey for the current process """
        devnull = open(os.devnull, 'w')  
        self.nb_refine += 1
        args = self.split_arg(line)
        # Check argument's validity
        self.check_refine(args)
        
        refine_opt = {'err_goal': args[0], 'split_channels': True}   
        precision = args[0]
        if len(args) == 2:
            refine_opt['max_process']= args[1]


        # initialize / remove lhapdf mode
        self.configure_directory()

        # Update random number
        self.update_random()
        self.save_random()

        if self.cluster_mode:
            logger.info('Creating Jobs')
        self.update_status('Refine results to %s' % precision, level=None)
        
        self.total_jobs = 0
        subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                 'subproc.mg'))]
    
        # cleanning the previous job
        for nb_proc,subdir in enumerate(subproc):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses', subdir)
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(match)

        x_improve = gen_ximprove.gen_ximprove(self, refine_opt)
        # Load the run statistics from the survey
        survey_statistics = dict(self.results.get_detail('run_statistics'))
        # Printout survey statistics
        if __debug__ and survey_statistics:
            globalstat = sum_html.RunStatistics()
            logger.debug(" === Survey statistics summary ===")
            for key, value in survey_statistics.items():
                globalstat.aggregate_statistics(value)
                level = 5
                if value.has_warning():
                    level = 10
                logger.log(level, 
                  value.nice_output(str('/'.join([key[0],'G%s'%key[1]]))).
                                                      replace(' statistics',''))
            logger.debug(globalstat.nice_output('combined', no_warning=True))
                        
        if survey_statistics:
            x_improve.run_statistics = survey_statistics
        
        x_improve.launch() # create the ajob for the refinment.
        if not self.history or 'refine' not in self.history[-1]:
            cross, error = x_improve.update_html() #update html results for survey
            if  cross == 0:
                return
            logger.info("Current estimate of cross-section: %s +- %s" % (cross, error))
        

        if isinstance(x_improve, gen_ximprove.gen_ximprove_v4):
            # Non splitted mode is based on writting ajob so need to track them
            # Splitted mode handle the cluster submition internally.
            for nb_proc,subdir in enumerate(subproc):
                subdir = subdir.strip()
                Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
                bindir = pjoin(os.path.relpath(self.dirbin, Pdir))
                               
                logger.info('    %s ' % subdir)
    
                if os.path.exists(pjoin(Pdir, 'ajob1')):
                    self.compile(['madevent'], cwd=Pdir)
                    
                    alljobs = glob.glob(pjoin(Pdir,'ajob*'))
                    
                    #remove associated results.dat (ensure to not mix with all data)
                    Gre = re.compile("\s*j=(G[\d\.\w]+)")
                    for job in alljobs:
                        Gdirs = Gre.findall(open(job).read())
                        for Gdir in Gdirs:
                            if os.path.exists(pjoin(Pdir, Gdir, 'results.dat')):
                                os.remove(pjoin(Pdir, Gdir,'results.dat'))
                    
                    nb_tot = len(alljobs)            
                    self.total_jobs += nb_tot
                    for i, job in enumerate(alljobs):
                        job = os.path.basename(job)
                        self.launch_job('%s' % job, cwd=Pdir, remaining=(nb_tot-i-1), 
                                 run_type='Refine number %s on %s (%s/%s)' % 
                                 (self.nb_refine, subdir, nb_proc+1, len(subproc)))

        self.monitor(run_type='All job submitted for refine number %s' % self.nb_refine, 
                     html=True)
        
        self.update_status("Combining runs", level='parton')
        try:
            os.remove(pjoin(Pdir, 'combine_runs.log'))
        except Exception:
            pass
        
        if isinstance(x_improve, gen_ximprove.gen_ximprove_v4):
            # the merge of the events.lhe is handle in the x_improve class
            # for splitted runs. (and partly in store_events).        
            combine_runs.CombineRuns(self.me_dir)
            self.refine_mode = "old"
        else:
            self.refine_mode = "new"
            
        cross, error = sum_html.make_all_html_results(self)
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)

        self.results.add_detail('run_statistics', 
                                dict(self.results.get_detail('run_statistics')))

        self.update_status('finish refine', 'parton', makehtml=False)
        devnull.close()
    
    ############################################################################ 
    def do_combine_iteration(self, line):
        """Not in help: Combine a given iteration combine_iteration Pdir Gdir S|R step
            S is for survey 
            R is for refine
            step is the iteration number (not very critical)""" 

        self.set_run_name("tmp")
        self.configure_directory(html_opening=False)
        Pdir, Gdir, mode, step = self.split_arg(line)
        if Gdir.startswith("G"):
            Gdir = Gdir[1:]
        if "SubProcesses" not in Pdir:
            Pdir = pjoin(self.me_dir, "SubProcesses", Pdir)
        if mode == "S":
            self.opts = dict([(key,value[1]) for (key,value) in \
                          self._survey_options.items()])
            gensym = gen_ximprove.gensym(self)
            gensym.combine_iteration(Pdir, Gdir, int(step))
        elif mode == "R":
            refine = gen_ximprove.gen_ximprove_share(self)
            refine.combine_iteration(Pdir, Gdir, int(step))             
        
            

      
    ############################################################################ 
    def do_combine_events(self, line):
        """Advanced commands: Launch combine events"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_combine_events(args)

        self.update_status('Combining Events', level='parton')

        
        if not hasattr(self, "refine_mode") or self.refine_mode == "old":
            try:
                os.remove(pjoin(self.me_dir,'SubProcesses', 'combine.log'))
            except Exception:
                pass

            tmpcluster = cluster.MultiCore(nb_core=1)
            tmpcluster.launch_and_wait('../bin/internal/run_combine', 
                                       args=[self.run_name],
                                       cwd=pjoin(self.me_dir,'SubProcesses'),
                                       stdout=pjoin(self.me_dir,'SubProcesses', 'combine.log'),
                                       required_output=[pjoin(self.me_dir,'SubProcesses', 'combine.log')])

            #self.cluster.launch_and_wait('../bin/internal/run_combine', 
            #                                cwd=pjoin(self.me_dir,'SubProcesses'),
            #                                stdout=pjoin(self.me_dir,'SubProcesses', 'combine.log'),
            #                                required_output=[pjoin(self.me_dir,'SubProcesses', 'combine.log')])
            
            output = misc.mult_try_open(pjoin(self.me_dir,'SubProcesses','combine.log')).read()
            # Store the number of unweighted events for the results object
            pat = re.compile(r'''\s*Unweighting\s*selected\s*(\d+)\s*events''')
            try:      
                nb_event = pat.search(output).groups()[0]
            except AttributeError:
                time.sleep(10)
                output = misc.mult_try_open(pjoin(self.me_dir,'SubProcesses','combine.log')).read()
                try:
                    nb_event = pat.search(output).groups()[0]
                except AttributeError:
                    logger.warning('Fail to read the number of unweighted events in the combine.log file')
                    nb_event = 0

            self.results.add_detail('nb_event', nb_event)
            
            
            # Define The Banner
            tag = self.run_card['run_tag']
            
            # Update the banner with the pythia card
            if not self.banner:
                self.banner = banner_mod.recover_banner(self.results, 'parton')
            self.banner.load_basic(self.me_dir)
            # Add cross-section/event information
            self.banner.add_generation_info(self.results.current['cross'], nb_event)
            if not hasattr(self, 'random_orig'): self.random_orig = 0
            self.banner.change_seed(self.random_orig)
            if not os.path.exists(pjoin(self.me_dir, 'Events', self.run_name)):
                os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))
            self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, 
                                         '%s_%s_banner.txt' % (self.run_name, tag)))
            
            
            self.banner.add_to_file(pjoin(self.me_dir,'Events', 'events.lhe'),
                                    out=pjoin(self.me_dir,'Events', self.run_name, 'events.lhe'))
            self.banner.add_to_file(pjoin(self.me_dir,'Events', 'unweighted_events.lhe'),        
                                    out=pjoin(self.me_dir,'Events', self.run_name, 'unweighted_events.lhe'))        

                    
        elif self.refine_mode == "new":
            # Define The Banner
            tag = self.run_card['run_tag']
            # Update the banner with the pythia card
            if not self.banner:
                self.banner = banner_mod.recover_banner(self.results, 'parton')
            self.banner.load_basic(self.me_dir)
            # Add cross-section/event information
            self.banner.add_generation_info(self.results.current['cross'], self.run_card['nevents'])
            if not hasattr(self, 'random_orig'): self.random_orig = 0
            self.banner.change_seed(self.random_orig)
            if not os.path.exists(pjoin(self.me_dir, 'Events', self.run_name)):
                os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))
            self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, 
                                    '%s_%s_banner.txt' % (self.run_name, tag)))
            
            
            AllEvent = lhe_parser.MultiEventFile()
            AllEvent.banner = self.banner
            
            for Gdir,mfactor in self.get_Gdir():
                if os.path.exists(pjoin(Gdir, 'events.lhe')):
                    result = sum_html.OneResult('')
                    result.read_results(pjoin(Gdir, 'results.dat'))
                    AllEvent.add(pjoin(Gdir, 'events.lhe'), 
                                 result.get('xsec'),
                                 result.get('xerru'),
                                 result.get('axsec')
                                 )
                    
            get_wgt = lambda event: event.wgt
            nb_event = AllEvent.unweight(pjoin(self.me_dir, "Events", self.run_name, "unweighted_events.lhe.gz"),
                              get_wgt, trunc_error=1e-2, event_target=self.run_card['nevents'],
                              log_level=logging.DEBUG)
            
            self.results.add_detail('nb_event', nb_event)
        
        eradir = self.options['exrootanalysis_path']
        madir = self.options['madanalysis_path']
        td = self.options['td_path']
        if eradir and misc.is_executable(pjoin(eradir,'ExRootLHEFConverter'))  and\
           os.path.exists(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')):
                if not os.path.exists(pjoin(self.me_dir, 'Events', self.run_name)):
                    os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))
                self.create_root_file(output='%s/unweighted_events.root' % \
                                                                  self.run_name)
    
    ############################################################################ 
    def do_store_events(self, line):
        """Advanced commands: Launch store events"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_combine_events(args)
        self.update_status('Storing parton level results', level='parton')

        run = self.run_name
        tag = self.run_card['run_tag']
        devnull = open(os.devnull, 'w')

        if not os.path.exists(pjoin(self.me_dir, 'Events', run)):
            os.mkdir(pjoin(self.me_dir, 'Events', run))
        if not os.path.exists(pjoin(self.me_dir, 'HTML', run)):
            os.mkdir(pjoin(self.me_dir, 'HTML', run))    
        
        # 1) Store overall process information
        input = pjoin(self.me_dir, 'SubProcesses', 'results.dat')
        output = pjoin(self.me_dir, 'SubProcesses', '%s_results.dat' % run)
        files.cp(input, output) 


        # 2) Treat the files present in the P directory
        # Ensure that the number of events is different of 0 
        if self.results.current['nb_event'] == 0:
            logger.warning("No event detected. No cleaning performed! This should allow to run:\n" +
                           "    cd Subprocesses; ../bin/internal/combine_events\n"+
                           "  to have your events if those one are missing.")
        else:
            for P_path in SubProcesses.get_subP(self.me_dir):
                G_dir = [G for G in os.listdir(P_path) if G.startswith('G') and 
                                                    os.path.isdir(pjoin(P_path,G))]
                for G in G_dir:
                    G_path = pjoin(P_path,G)
                    try:
                        # Remove events file (if present)
                        if os.path.exists(pjoin(G_path, 'events.lhe')):
                            os.remove(pjoin(G_path, 'events.lhe'))
                    except Exception:
                        continue
                    try:
                        # Store results.dat
                        if os.path.exists(pjoin(G_path, 'results.dat')):
                            input = pjoin(G_path, 'results.dat')
                            output = pjoin(G_path, '%s_results.dat' % run)
                            files.cp(input, output) 
                    except Exception:
                        continue                    
                    # Store log
                    try:
                        if os.path.exists(pjoin(G_path, 'log.txt')):
                            input = pjoin(G_path, 'log.txt')
                            output = pjoin(G_path, '%s_log.txt' % run)
                            files.mv(input, output) 
                    except Exception:
                        continue
                    try:   
                        # Grid
                        for name in ['ftn26']:
                            if os.path.exists(pjoin(G_path, name)):
                                if os.path.exists(pjoin(G_path, '%s_%s.gz'%(run,name))):
                                    os.remove(pjoin(G_path, '%s_%s.gz'%(run,name)))
                                input = pjoin(G_path, name)
                                output = pjoin(G_path, '%s_%s' % (run,name))
                                files.mv(input, output)
                                misc.gzip(pjoin(G_path, output), error=None) 
                    except Exception:
                        continue
                    # Delete ftn25 to ensure reproducible runs
                    if os.path.exists(pjoin(G_path, 'ftn25')):
                        os.remove(pjoin(G_path, 'ftn25'))

        # 3) Update the index.html
        misc.call(['%s/gen_cardhtml-pl' % self.dirbin],
                            cwd=pjoin(self.me_dir))
        
        
        # 4) Move the Files present in Events directory
        E_path = pjoin(self.me_dir, 'Events')
        O_path = pjoin(self.me_dir, 'Events', run)
        
        # The events file
        for name in ['events.lhe', 'unweighted_events.lhe']:
            finput = pjoin(E_path, name)
            foutput = pjoin(O_path, name)
            if os.path.exists(finput):
                logger.debug("File %s exists BAAAAD. Not move anymore!" % pjoin(E_path, name))
            if os.path.exists(foutput):
                misc.gzip(foutput, stdout="%s.gz" % foutput, error=False)
        #        if os.path.exists(pjoin(O_path, '%s.gz' % name)):
        #            os.remove(pjoin(O_path, '%s.gz' % name))
        #        input = pjoin(E_path, name)
        ##        output = pjoin(O_path, name)
              

        self.update_status('End Parton', level='parton', makehtml=False)
        devnull.close()
    
        
    ############################################################################ 
    def do_create_gridpack(self, line):
        """Advanced commands: Create gridpack from present run"""

        self.update_status('Creating gridpack', level='parton')
        # compile gen_ximprove
        misc.compile(['../bin/internal/gen_ximprove'], cwd=pjoin(self.me_dir, "Source"))
        args = self.split_arg(line)
        self.check_combine_events(args)
        if not self.run_tag: self.run_tag = 'tag_1'
        os.system("sed -i.bak \"s/ *.false.*=.*GridRun/  .true.  =  GridRun/g\" %s/Cards/grid_card.dat" \
                  % self.me_dir)
        misc.call(['./bin/internal/restore_data', self.run_name],
                        cwd=self.me_dir)
        misc.call(['./bin/internal/store4grid',
                         self.run_name, self.run_tag],
                        cwd=self.me_dir)
        misc.call(['./bin/internal/clean'], cwd=self.me_dir)
        misc.call(['./bin/internal/make_gridpack'], cwd=self.me_dir)
        files.mv(pjoin(self.me_dir, 'gridpack.tar.gz'), 
                pjoin(self.me_dir, '%s_gridpack.tar.gz' % self.run_name))
        os.system("sed -i.bak \"s/\s*.true.*=.*GridRun/  .false.  =  GridRun/g\" %s/Cards/grid_card.dat" \
                  % self.me_dir)
        self.update_status('gridpack created', level='gridpack')
        
    ############################################################################      
    def do_pythia(self, line):
        """launch pythia"""
        
        # Check argument's validity
        args = self.split_arg(line)
        if '--no_default' in args:
            if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pythia_card.dat')):
                return
            no_default = True
            args.remove('--no_default')
        else:
            no_default = False
            
        if not self.run_name:
            self.check_pythia(args)
            self.configure_directory(html_opening =False)
        else:
            # initialize / remove lhapdf mode        
            self.configure_directory(html_opening =False)
            self.check_pythia(args)        
        
        
        # the args are modify and the last arg is always the mode 
        if not no_default:
            self.ask_pythia_run_configuration(args[-1])

        if self.options['automatic_html_opening']:
            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
            self.options['automatic_html_opening'] = False

        # Update the banner with the pythia card
        if not self.banner or len(self.banner) <=1:
            self.banner = banner_mod.recover_banner(self.results, 'pythia')
                     
   

        pythia_src = pjoin(self.options['pythia-pgs_path'],'src')
        
        self.update_status('Running Pythia', 'pythia')
        try:
            os.remove(pjoin(self.me_dir,'Events','pythia.done'))
        except Exception:
            pass        
        
        ## LAUNCHING PYTHIA
        tag = self.run_tag
        pythia_log = pjoin(self.me_dir, 'Events', self.run_name , '%s_pythia.log' % tag)
        self.cluster.launch_and_wait('../bin/internal/run_pythia', 
                        argument= [pythia_src], stdout= pythia_log,
                        stderr=subprocess.STDOUT,
                        cwd=pjoin(self.me_dir,'Events'))

        os.remove(pjoin(self.me_dir, "Events", "unweighted_events.lhe"))



        if not os.path.exists(pjoin(self.me_dir,'Events','pythia.done')):
            logger.warning('Fail to produce pythia output. More info in \n     %s' % pythia_log)
            return
        else:
            os.remove(pjoin(self.me_dir,'Events','pythia.done'))
        
        self.to_store.append('pythia')
        
        # Find the matched cross-section
        if int(self.run_card['ickkw']):
            # read the line from the bottom of the file
            pythia_log = misc.BackRead(pjoin(self.me_dir,'Events', self.run_name, 
                                                         '%s_pythia.log' % tag))
            pythiare = re.compile("\s*I\s+0 All included subprocesses\s+I\s+(?P<generated>\d+)\s+(?P<tried>\d+)\s+I\s+(?P<xsec>[\d\.D\-+]+)\s+I")            
            for line in pythia_log:
                info = pythiare.search(line)
                if not info:
                    continue
                try:
                    # Pythia cross section in mb, we want pb
                    sigma_m = float(info.group('xsec').replace('D','E')) *1e9
                    Nacc = int(info.group('generated'))
                    Ntry = int(info.group('tried'))
                except ValueError:
                    # xsec is not float - this should not happen
                    self.results.add_detail('cross_pythia', 0)
                    self.results.add_detail('nb_event_pythia', 0)
                    self.results.add_detail('error_pythia', 0)
                else:
                    self.results.add_detail('cross_pythia', sigma_m)
                    self.results.add_detail('nb_event_pythia', Nacc)
                    #compute pythia error
                    error = self.results[self.run_name].return_tag(self.run_tag)['error']
                    if Nacc:                    
                        error_m = math.sqrt((error * Nacc/Ntry)**2 + sigma_m**2 *(1-Nacc/Ntry)/Nacc)
                    else:
                        error_m = 10000 * sigma_m
                    # works both for fixed number of generated events and fixed accepted events
                    self.results.add_detail('error_pythia', error_m)
                break                 

            pythia_log.close()
        
        pydir = pjoin(self.options['pythia-pgs_path'], 'src')
        eradir = self.options['exrootanalysis_path']
        madir = self.options['madanalysis_path']
        td = self.options['td_path']
        
        
 
        
        #Update the banner
        self.banner.add(pjoin(self.me_dir, 'Cards','pythia_card.dat'))
        if int(self.run_card['ickkw']):
            # Add the matched cross-section
            if 'MGGenerationInfo' in self.banner:
                self.banner['MGGenerationInfo'] += '#  Matched Integrated weight (pb)  :  %s\n' % self.results.current['cross_pythia']
            else:
                self.banner['MGGenerationInfo'] = '#  Matched Integrated weight (pb)  :  %s\n' % self.results.current['cross_pythia']
        banner_path = pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, tag))
        self.banner.write(banner_path)
        
        # Creating LHE file
        self.run_hep2lhe(banner_path)
        if int(self.run_card['ickkw']):
            misc.gzip(pjoin(self.me_dir,'Events','beforeveto.tree'),
                      stdout=pjoin(self.me_dir,'Events',self.run_name, tag+'_pythia_beforeveto.tree.gz'))           
        if self.run_card['use_syst'] in self.true:
            # Calculate syscalc info based on syst.dat
            try:
                self.run_syscalc('Pythia')
            except SysCalcError, error:
                logger.error(str(error))
            else:
                # Store syst.dat
                misc.gzip(pjoin(self.me_dir,'Events', 'syst.dat'),
                          stdout=pjoin(self.me_dir,'Events',self.run_name, tag + '_pythia_syst.dat.gz'))
                         
                # Store syscalc.dat
                if os.path.exists(pjoin(self.me_dir, 'Events', 'syscalc.dat')):
                    filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                              '%s_syscalc.dat' % self.run_tag)
                    misc.gzip(pjoin(self.me_dir, 'Events','syscalc.dat'),
                              stdout = "%s.gz" % filename)

        # Plot for pythia
        self.create_plot('Pythia')

        if os.path.exists(pjoin(self.me_dir,'Events','pythia_events.lhe')):
            misc.gzip(pjoin(self.me_dir,'Events','pythia_events.lhe'),
                      stdout=pjoin(self.me_dir,'Events', self.run_name,'%s_pythia_events.lhe.gz' % tag))
                
        self.update_status('finish', level='pythia', makehtml=False)
        self.exec_cmd('pgs --no_default', postcmd=False, printcmd=False)
        if self.options['delphes_path']:
            self.exec_cmd('delphes --no_default', postcmd=False, printcmd=False)
        self.print_results_in_shell(self.results.current)
    

    ################################################################################
    def do_remove(self, line):
        """Remove one/all run or only part of it"""

        args = self.split_arg(line)
        run, tag, mode = self.check_remove(args)
        if 'banner' in mode:
            mode.append('all')
        
        
        if run == 'all':
            # Check first if they are not a run with a name run.
            if os.path.exists(pjoin(self.me_dir, 'Events', 'all')):
                logger.warning('A run with name all exists. So we will not supress all processes.')
            else:
                for match in glob.glob(pjoin(self.me_dir, 'Events','*','*_banner.txt')):
                    run = match.rsplit(os.path.sep,2)[1]
                    if self.force:
                        args.append('-f')
                    try:
                        self.exec_cmd('remove %s %s' % (run, ' '.join(args[1:]) ) )
                    except self.InvalidCmd, error:
                        logger.info(error)
                        pass # run already clear
                return
            
        # Check that run exists
        if not os.path.exists(pjoin(self.me_dir, 'Events', run)):
            raise self.InvalidCmd('No run \'%s\' detected' % run)

        try:
            self.resuls.def_current(run)
            self.update_status(' Cleaning %s' % run, level=None)
        except Exception:
            misc.sprint('fail to update results or html status')
            pass # Just ensure that html never makes crash this function


        # Found the file to delete
        
        to_delete = glob.glob(pjoin(self.me_dir, 'Events', run, '*'))
        to_delete += glob.glob(pjoin(self.me_dir, 'HTML', run, '*'))
        # forbid the banner to be removed
        to_delete = [os.path.basename(f) for f in to_delete if 'banner' not in f]
        if tag:
            to_delete = [f for f in to_delete if tag in f]
            if 'parton' in mode or 'all' in mode:
                try:
                    if self.results[run][0]['tag'] != tag:
                        raise Exception, 'dummy'
                except Exception:
                    pass
                else:
                    nb_rm = len(to_delete)
                    if os.path.exists(pjoin(self.me_dir, 'Events', run, 'events.lhe.gz')):
                        to_delete.append('events.lhe.gz')
                    if os.path.exists(pjoin(self.me_dir, 'Events', run, 'unweighted_events.lhe.gz')):
                        to_delete.append('unweighted_events.lhe.gz')
                    if os.path.exists(pjoin(self.me_dir, 'HTML', run,'plots_parton.html')):
                        to_delete.append(pjoin(self.me_dir, 'HTML', run,'plots_parton.html'))                       
                    if nb_rm != len(to_delete):
                        logger.warning('Be carefull that partonic information are on the point to be removed.')
        if 'all' in mode:
            pass # delete everything
        else:
            if 'pythia' not in mode:
                to_delete = [f for f in to_delete if 'pythia' not in f]
            if 'pgs' not in mode:
                to_delete = [f for f in to_delete if 'pgs' not in f]
            if 'delphes' not in mode:
                to_delete = [f for f in to_delete if 'delphes' not in f]
            if 'parton' not in mode:
                to_delete = [f for f in to_delete if 'delphes' in f 
                                                      or 'pgs' in f 
                                                      or 'pythia' in f]
        if not self.force and len(to_delete):
            question = 'Do you want to delete the following files?\n     %s' % \
                               '\n    '.join(to_delete)
            ans = self.ask(question, 'y', choices=['y','n'])
        else:
            ans = 'y'
        
        if ans == 'y':
            for file2rm in to_delete:
                if os.path.exists(pjoin(self.me_dir, 'Events', run, file2rm)):
                    try:
                        os.remove(pjoin(self.me_dir, 'Events', run, file2rm))
                    except Exception:
                        shutil.rmtree(pjoin(self.me_dir, 'Events', run, file2rm))
                else:
                    try:
                        os.remove(pjoin(self.me_dir, 'HTML', run, file2rm))
                    except Exception:
                        shutil.rmtree(pjoin(self.me_dir, 'HTML', run, file2rm))



        # Remove file in SubProcess directory
        if 'all' in mode or 'channel' in mode:
            try:
                if tag and self.results[run][0]['tag'] != tag:
                    raise Exception, 'dummy'
            except Exception:
                pass
            else:
                to_delete = glob.glob(pjoin(self.me_dir, 'SubProcesses', '%s*' % run))
                to_delete += glob.glob(pjoin(self.me_dir, 'SubProcesses', '*','%s*' % run))
                to_delete += glob.glob(pjoin(self.me_dir, 'SubProcesses', '*','*','%s*' % run))

                if self.force or len(to_delete) == 0:
                    ans = 'y'
                else:
                    question = 'Do you want to delete the following files?\n     %s' % \
                               '\n    '.join(to_delete)
                    ans = self.ask(question, 'y', choices=['y','n'])

                if ans == 'y':
                    for file2rm in to_delete:
                        os.remove(file2rm)
                        
        if 'banner' in mode:
            to_delete = glob.glob(pjoin(self.me_dir, 'Events', run, '*'))
            if tag:
                # remove banner
                try:
                    os.remove(pjoin(self.me_dir, 'Events',run,'%s_%s_banner.txt' % (run,tag)))
                except Exception:
                    logger.warning('fail to remove the banner')
                # remove the run from the html output
                if run in self.results:
                    self.results.delete_run(run, tag)
                    return
            elif any(['banner' not in os.path.basename(p) for p in to_delete]):
                if to_delete:
                    raise MadGraph5Error, '''Some output still exists for this run. 
                Please remove those output first. Do for example: 
                remove %s all banner
                ''' % run
            else:
                shutil.rmtree(pjoin(self.me_dir, 'Events',run))
                if run in self.results:
                    self.results.delete_run(run)
                    return
        else:
            logger.info('''The banner is not removed. In order to remove it run:
    remove %s all banner %s''' % (run, tag and '--tag=%s ' % tag or '')) 

        # update database.
        self.results.clean(mode, run, tag)
        self.update_status('', level='all')



    ############################################################################
    def do_plot(self, line):
        """Create the plot for a given run"""

        # Since in principle, all plot are already done automaticaly
        self.store_result()
        args = self.split_arg(line)
        # Check argument's validity
        self.check_plot(args)
        logger.info('plot for run %s' % self.run_name)
        if not self.force:
            self.ask_edit_cards([], args, plot=True)
                
        if any([arg in ['all','parton'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe')
            if os.path.exists(filename+'.gz'):
                misc.gunzip('%s.gz' % filename, keep=True)
            if  os.path.exists(filename):
                files.ln(filename, pjoin(self.me_dir, 'Events'))
                self.create_plot('parton')
                if not os.path.exists(filename+'.gz'):
                    misc.gzip(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'),
                          stdout= "%s.gz" % filename)
                else:
                    try:
                        os.remove(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'))
                        os.remove(filename)
                    except Exception:
                        pass
            else:
                logger.info('No valid files for partonic plot') 
                
        if any([arg in ['all','pythia'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                          '%s_pythia_events.lhe' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                misc.gunzip("%s.gz" % filename)
            if  os.path.exists(filename):
                shutil.move(filename, pjoin(self.me_dir, 'Events','pythia_events.lhe'))
                self.create_plot('Pythia')
                misc.gzip(pjoin(self.me_dir, 'Events','pythia_events.lhe'),
                          stdout= "%s.gz" % filename)
            else:
                logger.info('No valid files for pythia plot')
                
                    
        if any([arg in ['all','pgs'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 
                                            '%s_pgs_events.lhco' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                misc.gunzip("%s.gz" % filename)
            if  os.path.exists(filename):
                self.create_plot('PGS')
                misc.gzip(filename)
            else:
                logger.info('No valid files for pgs plot')
                
        if any([arg in ['all','delphes'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 
                                        '%s_delphes_events.lhco' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                misc.gunzip("%s.gz" % filename)
            if  os.path.exists(filename):
                self.create_plot('Delphes')
                misc.gzip(filename)              
            else:
                logger.info('No valid files for delphes plot')

    ############################################################################
    def do_syscalc(self, line):
        """Evaluate systematics variation weights for a given run"""

        # Since in principle, all systematics run are already done automaticaly
        self.store_result()
        args = self.split_arg(line)
        # Check argument's validity
        self.check_syscalc(args)
        if self.ninitial == 1:
            logger.error('SysCalc can\'t be run for decay processes')
            return
        
        logger.info('Calculating systematics for run %s' % self.run_name)
        
        self.ask_edit_cards(['run_card'], args)
        self.run_card = banner_mod.RunCard(pjoin(self.medir, 'Cards', 'run_card.dat'))
                
        if any([arg in ['all','parton'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe')
            if os.path.exists(filename+'.gz'):
                misc.gunzip("%s.gz" % filename)
            if  os.path.exists(filename):
                shutil.move(filename, pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'))
                self.run_syscalc('parton')
                misc.gzip(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'),
                          stdout="%s.gz" % filename)
            else:
                logger.info('No valid files for parton level systematics run.')
                
        if any([arg in ['all','pythia'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                          '%s_pythia_syst.dat' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                misc.gunzip("%s.gz" % filename)
            if  os.path.exists(filename):
                shutil.move(filename, pjoin(self.me_dir, 'Events','syst.dat'))
                try:
                    self.run_syscalc('Pythia')
                except SysCalcError, error:
                    logger.warning(str(error))
                    return
                misc.gzip(pjoin(self.me_dir, 'Events','syst.dat'), "%s.gz" % filename)
                filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                          '%s_syscalc.dat' % self.run_tag)
                misc.gzip(pjoin(self.me_dir, 'Events','syscalc.dat'),
                          stdout=filename)
            else:
                logger.info('No valid files for pythia level')

    
    def store_result(self):
        """ tar the pythia results. This is done when we are quite sure that 
        the pythia output will not be use anymore """

        if not self.run_name:
            return
        
        self.results.save()
        
        
        if not self.to_store:
            return 
        
        tag = self.run_card['run_tag']
        self.update_status('storring files of previous run', level=None,\
                                                     error=True)
        if 'event' in self.to_store:
            if not os.path.exists(pjoin(self.me_dir, 'Events',self.run_name, 'unweighted_events.lhe.gz')):
                misc.gzip(pjoin(self.me_dir,'Events',self.run_name,"unweighted_events.lhe"))
        
        if 'pythia' in self.to_store:
            self.update_status('Storing Pythia files of previous run', level='pythia', error=True)
            
            p = pjoin(self.me_dir,'Events')
            n = self.run_name
            t = tag
            misc.gzip(pjoin(p,'pythia_events.hep'), 
                      stdout=pjoin(p,'%s/%s_pythia_events.hep' % (n,t)))

            self.to_store.remove('pythia')
        self.update_status('Done', level='pythia',makehtml=False,error=True)
        
        self.to_store = []

    def launch_job(self,exe, cwd=None, stdout=None, argument = [], remaining=0, 
                    run_type='', mode=None, **opt):
        """ """
        argument = [str(arg) for arg in argument]
        if mode is None:
            mode = self.cluster_mode
        
        # ensure that exe is executable
        if os.path.exists(exe) and not os.access(exe, os.X_OK):
            os.system('chmod +x %s ' % exe)
        elif (cwd and os.path.exists(pjoin(cwd, exe))) and not \
                                            os.access(pjoin(cwd, exe), os.X_OK):
            os.system('chmod +x %s ' % pjoin(cwd, exe))
                    
        if mode == 0:
            self.update_status((remaining, 1, 
                                self.total_jobs - remaining -1, run_type), level=None, force=False)
            start = time.time()
            #os.system('cd %s; ./%s' % (cwd,exe))
            status = misc.call([exe] + argument, cwd=cwd, stdout=stdout, **opt)
            logger.info('%s run in %f s' % (exe, time.time() -start))
            if status:
                raise MadGraph5Error, '%s didn\'t stop properly. Stop all computation' % exe


        elif mode in [1,2]:
            # For condor cluster, create the input/output files
            if 'ajob' in exe: 
                input_files = ['madevent','input_app.txt','symfact.dat','iproc.dat',
                               pjoin(self.me_dir, 'SubProcesses','randinit')]
                if os.path.exists(pjoin(self.me_dir,'SubProcesses', 
                  'MadLoop5_resources.tar.gz')) and cluster.need_transfer(self.options):
                    input_files.append(pjoin(self.me_dir,'SubProcesses', 'MadLoop5_resources.tar.gz'))
                
                output_files = []
                required_output = []
                

                #Find the correct PDF input file
                input_files.append(self.get_pdf_input_filename())
                        
                #Find the correct ajob
                Gre = re.compile("\s*j=(G[\d\.\w]+)")
                origre = re.compile("grid_directory=(G[\d\.\w]+)")
                try : 
                    fsock = open(exe)
                except Exception:
                    fsock = open(pjoin(cwd,exe))
                text = fsock.read()
                output_files = Gre.findall(text)
                if not output_files:
                    Ire = re.compile("for i in ([\d\.\s]*) ; do")
                    data = Ire.findall(text)
                    data = ' '.join(data).split()
                    for nb in data:
                        output_files.append('G%s' % nb)
                        required_output.append('G%s/results.dat' % nb)
                else:
                    for G in output_files:
                        if os.path.isdir(pjoin(cwd,G)):
                            input_files.append(G)
                            required_output.append('%s/results.dat' % G)
                
                if origre.search(text):
                    G_grid = origre.search(text).groups()[0]
                    input_files.append(pjoin(G_grid, 'ftn26'))
                
                #submitting
                self.cluster.submit2(exe, stdout=stdout, cwd=cwd, 
                             input_files=input_files, output_files=output_files,
                             required_output=required_output)
            elif 'survey' in exe:
                input_files = ['madevent','input_app.txt','symfact.dat','iproc.dat',
                               pjoin(self.me_dir, 'SubProcesses','randinit')]                 
                if os.path.exists(pjoin(self.me_dir,'SubProcesses', 
                  'MadLoop5_resources.tar.gz')) and cluster.need_transfer(self.options):
                    input_files.append(pjoin(self.me_dir,'SubProcesses', 
                                                   'MadLoop5_resources.tar.gz'))

                #Find the correct PDF input file
                input_files.append(self.get_pdf_input_filename())


                output_files = []
                required_output = []
                
                #Find the correct ajob
                suffix = "_%s" % int(float(argument[0]))
                if suffix == '_0':
                    suffix = ''
                output_files = ['G%s%s' % (i, suffix) for i in argument[1:]]
                for G in output_files:
                    required_output.append('%s/results.dat' % G)

                # add the grid information if needed
                for G in output_files:
                    if '.' in argument[0]:
                        offset = int(str(argument[0]).split('.')[1])
                    else:
                        offset = 0                    
                        
                    if offset ==0 or offset == int(float(argument[0])):
                        if os.path.exists(pjoin(cwd, G, 'input_app.txt')):
                            os.remove(pjoin(cwd, G, 'input_app.txt'))
                    
                    if os.path.exists(os.path.realpath(pjoin(cwd, G, 'ftn25'))):
                        if offset == 0 or offset == int(float(argument[0])):
                            os.remove(pjoin(cwd, G, 'ftn25'))
                            continue
                        else:
                            input_files.append(pjoin(cwd, G, 'ftn25'))
                            input_files.remove('input_app.txt')
                            input_files.append(pjoin(cwd, G, 'input_app.txt'))
                    elif os.path.lexists(pjoin(cwd, G, 'ftn25')):
                        try:
                            os.remove(pjoin(cwd,G,'ftn25'))
                        except:
                            pass

                #submitting
                self.cluster.cluster_submit(exe, stdout=stdout, cwd=cwd, argument=argument,  
                             input_files=input_files, output_files=output_files,
                             required_output=required_output, **opt)
            elif "refine_splitted.sh" in exe:
                input_files = ['madevent','symfact.dat','iproc.dat',
                               pjoin(self.me_dir, 'SubProcesses','randinit')]                 
                
                if os.path.exists(pjoin(self.me_dir,'SubProcesses',
                  'MadLoop5_resources.tar.gz')) and cluster.need_transfer(self.options):
                    input_files.append(pjoin(self.me_dir,'SubProcesses', 
                                                   'MadLoop5_resources.tar.gz'))

                #Find the correct PDF input file
                input_files.append(self.get_pdf_input_filename())


                output_files = [argument[0]]
                required_output = []
                for G in output_files:
                    required_output.append('%s/results.dat' % G)                
                input_files.append(pjoin(argument[1], "input_app.txt"))
                input_files.append(pjoin(argument[1], "ftn26"))
                
                #submitting
                self.cluster.cluster_submit(exe, stdout=stdout, cwd=cwd, argument=argument,  
                             input_files=input_files, output_files=output_files,
                             required_output=required_output, **opt)                

                
            
            else:
                self.cluster.submit(exe, stdout=stdout, cwd=cwd, **opt)
            

    ############################################################################
    def find_madevent_mode(self):
        """Find if Madevent is in Group mode or not"""
        
        # The strategy is too look in the files Source/run_configs.inc
        # if we found: ChanPerJob=3 then it's a group mode.
        file_path = pjoin(self.me_dir, 'Source', 'run_config.inc')
        text = open(file_path).read()
        if re.search(r'''s*parameter\s+\(ChanPerJob=2\)''', text, re.I+re.M):
            return 'group'
        else:
            return 'v4'
    
    ############################################################################
    def monitor(self, run_type='monitor', mode=None, html=False):
        """ monitor the progress of running job """
        

        starttime = time.time()
        if mode is None:
            mode = self.cluster_mode
        if mode > 0:
            if html:
                update_status = lambda idle, run, finish: \
                    self.update_status((idle, run, finish, run_type), level=None,
                                       force=False, starttime=starttime)
                update_first = lambda idle, run, finish: \
                    self.update_status((idle, run, finish, run_type), level=None,
                                       force=True, starttime=starttime)
            else:
                update_status = lambda idle, run, finish: None
                update_first = None
            try:   
                self.cluster.wait(self.me_dir, update_status, update_first=update_first)            
            except Exception, error:
                logger.info(error)
                if not self.force:
                    ans = self.ask('Cluster Error detected. Do you want to clean the queue? ("c"=continue the run anyway)',
                             default = 'y', choices=['y','n', 'c'])
                else:
                    ans = 'y'
                if ans == 'y':
                    self.cluster.remove()
                elif ans == 'c':
                    return self.monitor(run_type=run_type, mode=mode, html=html)
                raise
            except KeyboardInterrupt, error:
                self.cluster.remove()
                raise                            
        
        

    ############################################################################   
    def configure_directory(self, html_opening=True):
        """ All action require before any type of run """   


        # Basic check
        assert os.path.exists(pjoin(self.me_dir,'SubProcesses'))
        
        #see when the last file was modified
        time_mod = max([os.path.getctime(pjoin(self.me_dir,'Cards','run_card.dat')),
                        os.path.getctime(pjoin(self.me_dir,'Cards','param_card.dat'))])
        if self.configured > time_mod and hasattr(self, 'random'):
            return
        else:
            self.configured = time.time()
        self.update_status('compile directory', level=None, update_results=True)
        if self.options['automatic_html_opening'] and html_opening:
            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
            self.options['automatic_html_opening'] = False
            #open only once the web page
        # Change current working directory
        self.launching_dir = os.getcwd()
        
        # Check if we need the MSSM special treatment
        model = self.find_model_name()
        if model == 'mssm' or model.startswith('mssm-'):
            param_card = pjoin(self.me_dir, 'Cards','param_card.dat')
            mg5_param = pjoin(self.me_dir, 'Source', 'MODEL', 'MG5_param.dat')
            check_param_card.convert_to_mg5card(param_card, mg5_param)
            check_param_card.check_valid_param_card(mg5_param)
        
        # limit the number of event to 100k
        self.check_nb_events()

        # this is in order to avoid conflicts between runs with and without
        # lhapdf
        misc.compile(['clean4pdf'], cwd = pjoin(self.me_dir, 'Source'))
        
        # set environment variable for lhapdf.
        if self.run_card['pdlabel'] == "lhapdf":
            os.environ['lhapdf'] = 'True'
            self.link_lhapdf(pjoin(self.me_dir,'lib'))
            pdfsetsdir = self.get_lhapdf_pdfsetsdir()
            lhaid_list = [int(self.run_card['lhaid'])]
            self.copy_lhapdf_set(lhaid_list, pdfsetsdir)
        elif 'lhapdf' in os.environ.keys():
            del os.environ['lhapdf']
        if self.run_card['pdlabel'] != "lhapdf":
            self.pdffile = None
            
        # set random number
        if self.run_card['iseed'] != 0:
            self.random = int(self.run_card['iseed'])
            self.run_card['iseed'] = 0
            # Reset seed in run_card to 0, to ensure that following runs
            # will be statistically independent
            self.run_card.write(pjoin(self.me_dir, 'Cards','run_card.dat'))
        elif os.path.exists(pjoin(self.me_dir,'SubProcesses','randinit')):
            for line in open(pjoin(self.me_dir,'SubProcesses','randinit')):
                data = line.split('=')
                assert len(data) ==2
                self.random = int(data[1])
                break
        else:
            self.random = random.randint(1, 30107)
                                                               
        if self.run_card['ickkw'] == 2:
            logger.info('Running with CKKW matching')
            self.treat_CKKW_matching()
            
        # create param_card.inc and run_card.inc
        self.do_treatcards('')
        
        logger.info("compile Source Directory")
        # Compile
        for name in [ 'all', '../bin/internal/combine_events']:
            self.compile(arg=[name], cwd=os.path.join(self.me_dir, 'Source'))
        
        
    ############################################################################
    ##  HELPING ROUTINE
    ############################################################################
    @staticmethod
    def check_dir(path, default=''):
        """check if the directory exists. if so return the path otherwise the 
        default"""
         
        if os.path.isdir(path):
            return path
        else:
            return default

    ############################################################################
    def get_Pdir(self):
        """get the list of Pdirectory if not yet saved."""
        
        if hasattr(self, "Pdirs"):
            if self.me_dir in self.Pdirs[0]:
                return self.Pdirs
        self.Pdirs = [pjoin(self.me_dir, 'SubProcesses', l.strip()) 
                     for l in open(pjoin(self.me_dir,'SubProcesses', 'subproc.mg'))]
        return self.Pdirs
        
    ############################################################################
    def get_Gdir(self):
        """get the list of Gdirectory if not yet saved."""
        
        if hasattr(self, "Gdirs"):
            if self.me_dir in self.Gdirs[0]:
                return self.Gdirs

        Pdirs = self.get_Pdir()
        Gdirs = []        
        for P in Pdirs:
            for line in open(pjoin(P, "symfact.dat")):
                tag, mfactor = line.split()
                Gdirs.append( (pjoin(P, "G%s" % tag), int(mfactor)) )
            
        
        self.Gdirs = Gdirs
        return self.Gdirs
                
    ############################################################################
    def set_run_name(self, name, tag=None, level='parton', reload_card=False,
                     allow_new_tag=True):
        """define the run name, the run_tag, the banner and the results."""
        
        # when are we force to change the tag new_run:previous run requiring changes
        upgrade_tag = {'parton': ['parton','pythia','pgs','delphes'],
                       'pythia': ['pythia','pgs','delphes'],
                       'pgs': ['pgs'],
                       'delphes':['delphes'],
                       'plot':[],
                       'syscalc':[]}
        
        

        if name == self.run_name:        
            if reload_card:
                run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
                self.run_card = banner_mod.RunCard(run_card)

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
        
        new_tag = False
        # First call for this run -> set the banner
        self.banner = banner_mod.recover_banner(self.results, level, name)
        if 'mgruncard' in self.banner:
            self.run_card = self.banner.charge_card('run_card')
        else:
            # Read run_card
            run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
            self.run_card = banner_mod.RunCard(run_card)   
        
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
                   
        if allow_new_tag and (name in self.results and not new_tag):
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
    def check_nb_events(self):
        """Find the number of event in the run_card, and check that this is not 
        too large"""

        
        nb_event = int(self.run_card['nevents'])
        if nb_event > 1000000:
            logger.warning("Attempting to generate more than 1M events")
            logger.warning("Limiting number to 1M. Use multi_run for larger statistics.")
            path = pjoin(self.me_dir, 'Cards', 'run_card.dat')
            os.system(r"""perl -p -i.bak -e "s/\d+\s*=\s*nevents/1000000 = nevents/" %s""" \
                                                                         % path)
            self.run_card['nevents'] = 1000000

        return

  
    ############################################################################    
    def update_random(self):
        """ change random number"""
        
        self.random += 3
        if self.random > 30081*30081: # can't use too big random number
            raise MadGraph5Error,\
                  'Random seed too large ' + str(self.random) + ' > 30081*30081'

    ############################################################################
    def save_random(self):
        """save random number in appropirate file"""

        fsock = open(pjoin(self.me_dir, 'SubProcesses','randinit'),'w')
        fsock.writelines('r=%s\n' % self.random)

    def do_quit(self, *args, **opts):
        
        common_run.CommonRunCmd.do_quit(self, *args, **opts)
        return CmdExtended.do_quit(self, *args, **opts)
        
    ############################################################################
    def treat_ckkw_matching(self):
        """check for ckkw"""
        
        lpp1 = self.run_card['lpp1']
        lpp2 = self.run_card['lpp2']
        e1 = self.run_card['ebeam1']
        e2 = self.run_card['ebeam2']
        pd = self.run_card['pdlabel']
        lha = self.run_card['lhaid']
        xq = self.run_card['xqcut']
        translation = {'e1': e1, 'e2':e2, 'pd':pd, 
                       'lha':lha, 'xq':xq}

        if lpp1 or lpp2:
            # Remove ':s from pd          
            if pd.startswith("'"):
                pd = pd[1:]
            if pd.endswith("'"):
                pd = pd[:-1]                

            if xq >2 or xq ==2:
                xq = 2
            
            # find data file
            if pd == "lhapdf":
                issudfile = 'lib/issudgrid-%(e1)s-%(e2)s-%(pd)s-%(lha)s-%(xq)s.dat.gz'
            else:
                issudfile = 'lib/issudgrid-%(e1)s-%(e2)s-%(pd)s-%(xq)s.dat.gz'
            if self.web:
                issudfile = pjoin(self.webbin, issudfile % translation)
            else:
                issudfile = pjoin(self.me_dir, issudfile % translation)
            
            logger.info('Sudakov grid file: %s' % issudfile)
            
            # check that filepath exists
            if os.path.exists(issudfile):
                path = pjoin(self.me_dir, 'lib', 'issudgrid.dat')
                misc.gunzip(issudfile, keep=True, stdout=path)
            else:
                msg = 'No sudakov grid file for parameter choice. Start to generate it. This might take a while'
                logger.info(msg)
                self.update_status('GENERATE SUDAKOF GRID', level='parton')
                
                for i in range(-2,6):
                    self.cluster.submit('%s/gensudgrid ' % self.dirbin, 
                                    arguments = [i],
                                    cwd=self.me_dir, 
                                    stdout=open(pjoin(self.me_dir, 'gensudgrid%s.log' % i,'w')))
                self.monitor()
                for i in range(-2,6):
                    path = pjoin(self.me_dir, 'lib', 'issudgrid.dat')
                    os.system('cat %s/gensudgrid%s.log >> %s' % (self.me_dir, path))
                    misc.gzip(path, stdout=issudfile)
                                     
    ############################################################################
    def create_root_file(self, input='unweighted_events.lhe', 
                                              output='unweighted_events.root' ):
        """create the LHE root file """
        self.update_status('Creating root files', level='parton')

        eradir = self.options['exrootanalysis_path']
        try:
            misc.call(['%s/ExRootLHEFConverter' % eradir, 
                             input, output],
                            cwd=pjoin(self.me_dir, 'Events'))
        except Exception:
            logger.warning('fail to produce Root output [problem with ExRootAnalysis]')
    
    def run_syscalc(self, mode='parton', event_path=None, output=None):
        """create the syscalc output""" 

        if self.run_card['use_syst'] not in self.true:
            return
        
        logger.info('running syscalc on mode %s' % mode)        
        scdir = self.options['syscalc_path']
        tag = self.run_card['run_tag']  
        card = pjoin(self.me_dir, 'bin','internal', 'syscalc_card.dat')
        template = open(pjoin(self.me_dir, 'bin','internal', 'syscalc_template.dat')).read()
        self.run_card['sys_pdf'] = self.run_card['sys_pdf'].split('#',1)[0].replace('&&',' \n ')
        
        if self.run_card['sys_pdf'].lower() in ['', 'f', 'false', 'none', '.false.']:
            self.run_card['sys_pdf'] = ''
        if self.run_card['sys_alpsfact'].lower() in ['', 'f', 'false', 'none','.false.']:
            self.run_card['sys_alpsfact'] = ''
        
        # check if the scalecorrelation parameter is define:
        if not 'sys_scalecorrelation' in self.run_card:
            self.run_card['sys_scalecorrelation'] = -1
        open(card,'w').write(template % self.run_card)
        
        if not scdir or \
            not os.path.exists(card):
            return False
        event_dir = pjoin(self.me_dir, 'Events')

        if not event_path:
            if mode == 'parton':
                event_path = pjoin(event_dir,self.run_name, 'unweighted_events.lhe')
                if not (os.path.exists(event_path) or os.path.exists(event_path+".gz")):
                    event_path = pjoin(event_dir, 'unweighted_events.lhe')
                output = pjoin(event_dir, 'syscalc.lhe')
            elif mode == 'Pythia':
                if 'mgpythiacard' in self.banner:
                    pat = re.compile('''^\s*qcut\s*=\s*([\+\-\d.e]*)''', re.M+re.I)
                    data = pat.search(self.banner['mgpythiacard'])
                    if data:
                        qcut = float(data.group(1))
                        xqcut = abs(self.run_card['xqcut'])
                        for value in self.run_card['sys_matchscale'].split():
                            if float(value) < qcut:
                                raise SysCalcError, 'qcut value for sys_matchscale lower than qcut in pythia_card. Bypass syscalc'
                            if float(value) < xqcut:
                                raise SysCalcError, 'qcut value for sys_matchscale lower than xqcut in run_card. Bypass syscalc'
                        
                        
                event_path = pjoin(event_dir,'syst.dat')
                output = pjoin(event_dir, 'syscalc.dat')
            else:
                raise self.InvalidCmd, 'Invalid mode %s' % mode
            
        if not os.path.exists(event_path):
            if os.path.exists(event_path+'.gz'):
                misc.gunzip(event_path+'.gz')
            else:
                raise SysCalcError, 'Events file %s does not exits' % event_path
        
        self.update_status('Calculating systematics for %s level' % mode, level = mode.lower())
        try:
            proc = misc.call([os.path.join(scdir, 'sys_calc'),
                               event_path, card, output],
                            stdout = open(pjoin(event_dir, self.run_name, '%s_%s_syscalc.log' % (tag,mode)),'w'),
                            stderr = subprocess.STDOUT,
                            cwd=event_dir)
            # Wait 5 s to make sure file is finished writing
            time.sleep(5)            
        except OSError, error:
            logger.error('fail to run syscalc: %s. Please check that SysCalc is correctly installed.' % error)
        else:
            if mode == 'parton' and os.path.exists(output):
                files.mv(output, event_path)
            else:
                logger.warning('SysCalc Failed. Please read the associate log to see the reason. Did you install the associate PDF set?')
        self.update_status('End syscalc for %s level' % mode, level = mode.lower(),
                                                                 makehtml=False)
        
        return True   



    ############################################################################
    def ask_run_configuration(self, mode=None, args=[]):
        """Ask the question when launching generate_events/multi_run"""
        
        available_mode = ['0']
        void = 'NOT INSTALLED'
        switch_order = ['pythia', 'pgs', 'delphes', 'madspin', 'reweight']
        switch = {'pythia': void, 'pgs': void, 'delphes': void,
                  'madspin': void, 'reweight': void}
        description = {'pythia': 'Run the pythia shower/hadronization:',
                       'pgs': 'Run PGS as detector simulator:',
                       'delphes':'Run Delphes as detector simulator:',
                       'madspin':'Decay particles with the MadSpin module:',
                       'reweight':'Add weights to the events based on changing model parameters:',
                       }
        force_switch = {('pythia', 'OFF'): {'pgs': 'OFF', 'delphes': 'OFF'},
                       ('pgs', 'ON'): {'pythia':'ON'},
                       ('delphes', 'ON'): {'pythia': 'ON'}}
        switch_assign = lambda key, value: switch.__setitem__(key, value if switch[key] != void else void )

        
        # Init the switch value according to the current status
        if self.options['pythia-pgs_path']:
            available_mode.append('1')
            available_mode.append('2')
            if os.path.exists(pjoin(self.me_dir,'Cards','pythia_card.dat')):
                switch['pythia'] = 'ON'
            else:
                switch['pythia'] = 'OFF'
            if os.path.exists(pjoin(self.me_dir,'Cards','pgs_card.dat')):
                switch['pgs'] = 'ON'
            else:
                switch['pgs'] = 'OFF'                
            if self.options['delphes_path']:
                available_mode.append('3')
                if os.path.exists(pjoin(self.me_dir,'Cards','delphes_card.dat')):
                    switch['delphes'] = 'ON'
                else:
                    switch['delphes'] = 'OFF'
                    
        # Check switch status for MS/reweight
        if not MADEVENT or ('mg5_path' in self.options and self.options['mg5_path']):
            available_mode.append('4')
            available_mode.append('5')
            if os.path.exists(pjoin(self.me_dir,'Cards','madspin_card.dat')):
                switch['madspin'] = 'ON'
            else:
                switch['madspin'] = 'OFF'
            if misc.has_f2py() or self.options['f2py_compiler']:
                if os.path.exists(pjoin(self.me_dir,'Cards','reweight_card.dat')):
                    switch['reweight'] = 'ON'
                else:
                    switch['reweight'] = 'OFF'
            else: 
                switch['reweight'] = 'Not available (requires NumPy)'
                 
        if '-R' in args or '--reweight' in args:
            if switch['reweight'] == 'OFF':
                switch['reweight'] = 'ON'
            elif switch['reweight'] != 'ON':
                logger.critical("Cannot run reweight: %s", switch['reweight'])
        if '-M' in args or '--madspin' in args:
            switch['madspin'] = 'ON'

        options = list(available_mode) + ['auto', 'done']
        for id, key in enumerate(switch_order):
            if switch[key] not in [void, 'Not available (requires NumPy)']:
                options += ['%s=%s' % (key, s) for s in ['ON','OFF']]
                options.append(key)
        options.append('parton')    
        
        #ask the question
        if mode or not self.force:
            answer = ''
            while answer not in ['0', 'done', 'auto']:
                if mode:
                    answer = mode
                else:      
                    switch_format = " %i %-61s %10s=%s\n"
                    question = "The following switches determine which programs are run:\n"
                    for id, key in enumerate(switch_order):
                        question += switch_format % (id+1, description[key], key, switch[key])
                    question += '  Either type the switch number (1 to %s) to change its default setting,\n' % (id+1)
                    question += '  or set any switch explicitly (e.g. type \'madspin=ON\' at the prompt)\n'
                    question += '  Type \'0\', \'auto\', \'done\' or just press enter when you are done.\n'
                    answer = self.ask(question, '0', options)
                if answer.isdigit() and answer != '0':
                    key = switch_order[int(answer) - 1]
                    answer = '%s=%s' % (key, 'ON' if switch[key] == 'OFF' else 'OFF')

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
                    continue
                else:
                    logger.info('pass in %s only mode' % answer, '$MG:color:BLACK')
                    switch_assign('madspin', 'OFF')
                    switch_assign('reweight', 'OFF')
                    if answer == 'parton':
                        switch_assign('pythia', 'OFF')
                        switch_assign('pgs', 'OFF')
                        switch_assign('delphes', 'OFF')
                    elif answer == 'pythia':
                        switch_assign('pythia', 'ON')
                        switch_assign('pgs', 'OFF')
                        switch_assign('delphes', 'OFF')
                    elif answer == 'pgs':
                        switch_assign('pythia', 'ON')
                        switch_assign('pgs', 'ON')
                        switch_assign('delphes', 'OFF')
                    elif answer == 'delphes':
                        switch_assign('pythia', 'ON')
                        switch_assign('pgs', 'OFF')
                        switch_assign('delphes', 'ON')
                    elif answer == 'madspin':
                        switch_assign('madspin', 'ON')
                        switch_assign('pythia', 'OFF')
                        switch_assign('pgs', 'OFF')
                        switch_assign('delphes', 'OF')                        
                    elif answer == 'reweight':
                        switch_assign('reweight', 'ON')
                        switch_assign('pythia', 'OFF')
                        switch_assign('pgs', 'OFF')
                        switch_assign('delphes', 'OFF')
                    
                    
                if mode:
                    answer =  '0' #mode auto didn't pass here (due to the continue)
            else:
                answer = 'auto'                        
                                                                     
        # Now that we know in which mode we are check that all the card
        #exists (copy default if needed)

        cards = ['param_card.dat', 'run_card.dat']
        if switch['pythia'] == 'ON':
            cards.append('pythia_card.dat')
        if switch['pgs'] == 'ON':
            cards.append('pgs_card.dat')
        if switch['delphes'] == 'ON':
            cards.append('delphes_card.dat')
            delphes3 = True
            if os.path.exists(pjoin(self.options['delphes_path'], 'data')):
                delphes3 = False
                cards.append('delphes_trigger.dat')
        if switch['madspin'] == 'ON':
            cards.append('madspin_card.dat')
        if switch['reweight'] == 'ON':
            cards.append('reweight_card.dat')
        self.keep_cards(cards)
        
        if os.path.isfile(pjoin(self.me_dir,'Cards','MadLoopParams.dat')):
            cards.append('MadLoopParams.dat')
        
        if self.force:
            self.check_param_card(pjoin(self.me_dir,'Cards','param_card.dat' ))
            return

        if answer == 'auto':
            self.ask_edit_cards(cards, mode='auto')
        else:
            self.ask_edit_cards(cards)
        return
    
    ############################################################################
    def ask_pythia_run_configuration(self, mode=None):
        """Ask the question when launching pythia"""
        
        available_mode = ['0', '1', '2']
        if self.options['delphes_path']:
                available_mode.append('3')
        name = {'0': 'auto', '1': 'pythia', '2':'pgs', '3':'delphes'}
        options = available_mode + [name[val] for val in available_mode]
        question = """Which programs do you want to run?
    0 / auto    : running existing card
    1 / pythia  : Pythia 
    2 / pgs     : Pythia + PGS\n"""
        if '3' in available_mode:
            question += """    3 / delphes  : Pythia + Delphes.\n"""

        if not self.force:
            if not mode:
                mode = self.ask(question, '0', options)
        elif not mode:
            mode = 'auto'
            
        if mode.isdigit():
            mode = name[mode]
             
        auto = False
        if mode == 'auto':
            auto = True
            if os.path.exists(pjoin(self.me_dir, 'Cards', 'pgs_card.dat')):
                mode = 'pgs'
            elif os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_card.dat')):
                mode = 'delphes'
            else: 
                mode = 'pythia'
        logger.info('Will run in mode %s' % mode)
                                               
        # Now that we know in which mode we are check that all the card
        #exists (copy default if needed) remove pointless one
        cards = ['pythia_card.dat']
        if mode == 'pgs':
            cards.append('pgs_card.dat')
        if mode == 'delphes':
            cards.append('delphes_card.dat')
            delphes3 = True
            if os.path.exists(pjoin(self.options['delphes_path'], 'data')):
                delphes3 = False
                cards.append('delphes_trigger.dat')
        self.keep_cards(cards)
        
        if self.force:
            return mode
        
        if auto:
            self.ask_edit_cards(cards, mode='auto')
        else:
            self.ask_edit_cards(cards)

        return mode
                
  
            

                
#===============================================================================
# MadEventCmd
#===============================================================================
class MadEventCmdShell(MadEventCmd, cmd.CmdShell):
    """The command line processor of MadGraph"""  



#===============================================================================
# HELPING FUNCTION For Subprocesses
#===============================================================================
class SubProcesses(object):

    name_to_pdg = {}

    @classmethod
    def clean(cls):
        cls.name_to_pdg = {}
    
    @staticmethod
    def get_subP(me_dir):
        """return the list of Subprocesses"""
        
        out = []
        for line in open(pjoin(me_dir,'SubProcesses', 'subproc.mg')):
            if not line:
                continue
            name = line.strip()
            if os.path.exists(pjoin(me_dir, 'SubProcesses', name)):
                out.append(pjoin(me_dir, 'SubProcesses', name))
        
        return out
        


    @staticmethod
    def get_subP_info(path):
        """ return the list of processes with their name"""

        nb_sub = 0
        names = {}
        old_main = ''

        if not os.path.exists(os.path.join(path,'processes.dat')):
            return SubProcesses.get_subP_info_v4(path)

        for line in open(os.path.join(path,'processes.dat')):
            main = line[:8].strip()
            if main == 'mirror':
                main = old_main
            if line[8:].strip() == 'none':
                continue
            else:
                main = int(main)
                old_main = main

            sub_proccess = line[8:]
            nb_sub += sub_proccess.count(',') + 1
            if main in names:
                names[main] += [sub_proccess.split(',')]
            else:
                names[main]= [sub_proccess.split(',')]

        return names

    @staticmethod
    def get_subP_info_v4(path):
        """ return the list of processes with their name in case without grouping """

        nb_sub = 0
        names = {'':[[]]}
        path = os.path.join(path, 'auto_dsig.f')
        found = 0
        for line in open(path):
            if line.startswith('C     Process:'):
                found += 1
                names[''][0].append(line[15:])
            elif found >1:
                break
        return names


    @staticmethod
    def get_subP_ids(path):
        """return the pdg codes of the particles present in the Subprocesses"""

        all_ids = []
        for line in open(pjoin(path, 'leshouche.inc')):
            if not 'IDUP' in line:
                continue
            particles = re.search("/([\d,-]+)/", line)
            all_ids.append([int(p) for p in particles.group(1).split(',')])
        return all_ids
    
    
#===============================================================================                                                                              
class GridPackCmd(MadEventCmd):
    """The command for the gridpack --Those are not suppose to be use interactively--"""

    def __init__(self, me_dir = None, nb_event=0, seed=0, *completekey, **stdin):
        """Initialize the command and directly run"""

        # Initialize properly
        
        MadEventCmd.__init__(self, me_dir, *completekey, **stdin)
        self.run_mode = 0
        self.random = seed
        self.random_orig = self.random
        self.options['automatic_html_opening'] = False
        # Now it's time to run!
        if me_dir and nb_event and seed:
            self.launch(nb_event, seed)
        else:
            raise MadGraph5Error,\
                  'Gridpack run failed: ' + str(me_dir) + str(nb_event) + \
                  str(seed)

    def launch(self, nb_event, seed):
        """ launch the generation for the grid """

        # 1) Restore the default data
        logger.info('generate %s events' % nb_event)
        self.set_run_name('GridRun_%s' % seed)
        self.update_status('restoring default data', level=None)
        misc.call([pjoin(self.me_dir,'bin','internal','restore_data'),
                         'default'],
            cwd=self.me_dir)

        # 2) Run the refine for the grid
        self.update_status('Generating Events', level=None)
        #misc.call([pjoin(self.me_dir,'bin','refine4grid'),
        #                str(nb_event), '0', 'Madevent','1','GridRun_%s' % seed],
        #                cwd=self.me_dir)
        self.refine4grid(nb_event)

        # 3) Combine the events/pythia/...
        self.exec_cmd('combine_events')
        self.exec_cmd('store_events')
        self.print_results_in_shell(self.results.current)
        self.exec_cmd('decay_events -from_cards', postcmd=False)

    def refine4grid(self, nb_event):
        """Special refine for gridpack run."""
        self.nb_refine += 1
        
        precision = nb_event

        # initialize / remove lhapdf mode
        # self.configure_directory() # All this has been done before
        self.cluster_mode = 0 # force single machine

        # Store seed in randinit file, to be read by ranmar.f
        self.save_random()
        
        self.update_status('Refine results to %s' % precision, level=None)
        logger.info("Using random number seed offset = %s" % self.random)
        
        self.total_jobs = 0
        subproc = [P for P in os.listdir(pjoin(self.me_dir,'SubProcesses')) if 
                   P.startswith('P') and os.path.isdir(pjoin(self.me_dir,'SubProcesses', P))]
        devnull = open(os.devnull, 'w')
        for nb_proc,subdir in enumerate(subproc):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            bindir = pjoin(os.path.relpath(self.dirbin, Pdir))
                           
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(pjoin(Pdir, match))
            

            logfile = pjoin(Pdir, 'gen_ximprove.log')
            misc.call([pjoin(bindir, 'gen_ximprove')],
                                    stdin=subprocess.PIPE,
                                    stdout=open(logfile,'w'),
                                    cwd=Pdir)

            if os.path.exists(pjoin(Pdir, 'ajob1')):
                alljobs = glob.glob(pjoin(Pdir,'ajob*'))
                nb_tot = len(alljobs)            
                self.total_jobs += nb_tot
                for i, job in enumerate(alljobs):
                    job = os.path.basename(job)
                    self.launch_job('%s' % job, cwd=Pdir, remaining=(nb_tot-i-1), 
                             run_type='Refine number %s on %s (%s/%s)' %
                             (self.nb_refine, subdir, nb_proc+1, len(subproc)))
                    if os.path.exists(pjoin(self.me_dir,'error')):
                        self.monitor(html=True)
                        raise MadEventError, \
                            'Error detected in dir %s: %s' % \
                            (Pdir, open(pjoin(self.me_dir,'error')).read())
        self.monitor(run_type='All job submitted for refine number %s' % 
                                                                 self.nb_refine)
        
        self.update_status("Combining runs", level='parton')
        try:
            os.remove(pjoin(Pdir, 'combine_runs.log'))
        except Exception:
            pass
        
        bindir = pjoin(os.path.relpath(self.dirbin, pjoin(self.me_dir,'SubProcesses')))
        combine_runs.CombineRuns(self.me_dir)
        
        #update html output
        cross, error = sum_html.make_all_html_results(self)
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)
        
        
        self.update_status('finish refine', 'parton', makehtml=False)
        devnull.close()


class MadLoopInitializer(object):
    """ A container class for the various methods for initializing MadLoop. It is
    placed in MadEventInterface because it is used by Madevent for loop-induced 
    simulations. """

    @staticmethod
    def make_and_run(dir_name,checkRam=False):
        """ Compile the check program in the directory dir_name.
        Return the compilation and running time. """

        # Make sure to recreate the executable and modified source
        # (The time stamps are sometimes not actualized if it is too fast)
        if os.path.isfile(pjoin(dir_name,'check')):
            os.remove(pjoin(dir_name,'check'))
            os.remove(pjoin(dir_name,'check_sa.o'))
            os.remove(pjoin(dir_name,'loop_matrix.o'))            
        # Now run make
        devnull = open(os.devnull, 'w')
        start=time.time()
        retcode = subprocess.call(['make','check'],
                                   cwd=dir_name, stdout=devnull, stderr=devnull)
        compilation_time = time.time()-start
        if retcode != 0:
            logging.info("Error while executing make in %s" % dir_name)
            return None, None, None

        if not checkRam:
            start=time.time()
            retcode = subprocess.call('./check',
                                   cwd=dir_name, stdout=devnull, stderr=devnull)

            run_time = time.time()-start
            ram_usage = None
        else:
            ptimer = misc.ProcessTimer(['./check'], cwd=dir_name, shell=False, \
                                 stdout=devnull, stderr=devnull, close_fds=True)
            try:
                ptimer.execute()
                #poll as often as possible; otherwise the subprocess might 
                # "sneak" in some extra memory usage while you aren't looking
                # Accuracy of .2 seconds is enough for the timing.
                while ptimer.poll():
                    time.sleep(.2)
            finally:
                #make sure that we don't leave the process dangling.
                ptimer.close()
            # Notice that ptimer.max_vms_memory is also available if needed.
            ram_usage = ptimer.max_rss_memory
            # Unfortunately the running time is less precise than with the
            # above version
            run_time = (ptimer.t1 - ptimer.t0)
            retcode = ptimer.p.returncode

        devnull.close()

        if retcode != 0:
            logging.warning("Error while executing ./check in %s" % dir_name)
            return None, None, None

        return compilation_time, run_time, ram_usage

    @staticmethod
    def fix_PSPoint_in_check(dir_path, read_ps = True, npoints = 1,
                             hel_config = -1, mu_r=0.0, split_orders=-1):
        """Set check_sa.f to be reading PS.input assuming a working dir dir_name.
        if hel_config is different than -1 then check_sa.f is configured so to
        evaluate only the specified helicity.
        If mu_r > 0.0, then the renormalization constant value will be hardcoded
        directly in check_sa.f, if is is 0 it will be set to Sqrt(s) and if it
        is < 0.0 the value in the param_card.dat is used.
        If the split_orders target (i.e. the target squared coupling orders for 
        the computation) is != -1, it will be changed in check_sa.f via the
        subroutine CALL SET_COUPLINGORDERS_TARGET(split_orders)."""

        file_path = dir_path
        if not os.path.isfile(dir_path) or \
                                   not os.path.basename(dir_path)=='check_sa.f':
            file_path = pjoin(dir_path,'check_sa.f')
            if not os.path.isfile(file_path):
                directories = [d for d in glob.glob(pjoin(dir_path,'P*_*')) \
                         if (re.search(r'.*P\d+_\w*$', d) and os.path.isdir(d))]
                if len(directories)>0:
                     file_path = pjoin(directories[0],'check_sa.f')
        if not os.path.isfile(file_path):
            raise MadGraph5Error('Could not find the location of check_sa.f'+\
                                  ' from the specified path %s.'%str(file_path))    

        file = open(file_path, 'r')
        check_sa = file.read()
        file.close()
        
        file = open(file_path, 'w')
        check_sa = re.sub(r"READPS = \S+\)","READPS = %s)"%('.TRUE.' if read_ps \
                                                      else '.FALSE.'), check_sa)
        check_sa = re.sub(r"NPSPOINTS = \d+","NPSPOINTS = %d"%npoints, check_sa)
        if hel_config != -1:
            check_sa = re.sub(r"SLOOPMATRIX\S+\(\S+,MATELEM,",
                      "SLOOPMATRIXHEL_THRES(P,%d,MATELEM,"%hel_config, check_sa)
        else:
            check_sa = re.sub(r"SLOOPMATRIX\S+\(\S+,MATELEM,",
                                        "SLOOPMATRIX_THRES(P,MATELEM,",check_sa)
        if mu_r > 0.0:
            check_sa = re.sub(r"MU_R=SQRTS","MU_R=%s"%\
                                        (("%.17e"%mu_r).replace('e','d')),check_sa)
        elif mu_r < 0.0:
            check_sa = re.sub(r"MU_R=SQRTS","",check_sa)
        
        if split_orders > 0:
            check_sa = re.sub(r"SET_COUPLINGORDERS_TARGET\(-?\d+\)",
                     "SET_COUPLINGORDERS_TARGET(%d)"%split_orders,check_sa) 
        
        file.write(check_sa)
        file.close()

    @staticmethod    
    def run_initialization(run_dir=None, SubProc_dir=None, infos=None,\
                            req_files = ['HelFilter.dat','LoopFilter.dat'],
                            attempts = [4,15]):
        """ Run the initialization of the process in 'run_dir' with success 
        characterized by the creation of the files req_files in this directory.
        The directory containing the driving source code 'check_sa.f'.
        The list attempt gives the successive number of PS points the 
        initialization should be tried with before calling it failed.
        Returns the number of PS points which were necessary for the init.
        Notice at least run_dir or SubProc_dir must be provided.
        A negative attempt number given in input means that quadprec will be
        forced for initialization."""
        
        # If the user does not want detailed info, then set the dictionary
        # to a dummy one.
        if infos is None:
            infos={}
        
        if SubProc_dir is None and run_dir is None:
            raise MadGraph5Error, 'At least one of [SubProc_dir,run_dir] must'+\
                                           ' be provided in run_initialization.'
        
        # If the user does not specify where is check_sa.f, then it is assumed
        # to be one levels above run_dir
        if SubProc_dir is None:
            SubProc_dir = os.path.abspath(pjoin(run_dir,os.pardir))
            
        if run_dir is None:
            directories =[ dir for dir in glob.glob(pjoin(SubProc_dir,\
                                             'P[0-9]*')) if os.path.isdir(dir) ]
            if directories:
                run_dir = directories[0]
            else:
                raise MadGraph5Error, 'Could not find a valid running directory'+\
                                                      ' in %s.'%str(SubProc_dir)

        # Use the presence of the file born_matrix.f to decide if it is a 
        # loop-induced process or not. It's not crucial, but just that because
        # of the dynamic adjustment of the ref scale used for deciding what are
        # the zero contributions, more points are neeeded for loop-induced.
        if not os.path.isfile(pjoin(run_dir,'born_matrix.f')):
            if len(attempts)>=1 and attempts[0]<8:
                attempts[0]=8
            if len(attempts)>=2 and attempts[1]<25:
                attempts[1]=25

        to_attempt = list(attempts)
        to_attempt.reverse()
        my_req_files = list(req_files)

        MLCardPath = pjoin(SubProc_dir,'MadLoopParams.dat')
        if not os.path.isfile(MLCardPath):
            raise MadGraph5Error, 'Could not find MadLoopParams.dat at %s.'\
                                                                     %MLCardPath
        else:
            MLCard      = banner_mod.MadLoopParam(MLCardPath) 
            MLCard_orig = banner_mod.MadLoopParam(MLCard)
        
        # Make sure that LoopFilter really is needed.
        if not MLCard['UseLoopFilter']:
            try:
                my_req_files.remove('LoopFilter.dat')
            except ValueError:
                pass
        
        if MLCard['HelicityFilterLevel']==0:
            try:
                my_req_files.remove('HelFilter.dat')
            except ValueError:
                pass

        def need_init():
            """ True if init not done yet."""
            proc_prefix_file = open(pjoin(run_dir,'proc_prefix.txt'),'r')
            proc_prefix = proc_prefix_file.read()
            proc_prefix_file.close()
            return any([not os.path.exists(pjoin(run_dir,'MadLoop5_resources',
                            proc_prefix+fname)) for fname in my_req_files]) or \
                         not os.path.isfile(pjoin(run_dir,'check')) or \
                         not os.access(pjoin(run_dir,'check'), os.X_OK)
        
        # Check if this is a process without born by checking the presence of the
        # file born_matrix.f
        is_loop_induced = os.path.exists(pjoin(run_dir,'born_matrix.f'))
        
        # For loop induced processes, always attempt quadruple precision if
        # double precision attempts fail and the user didn't specify himself
        # quadruple precision initializations attempts
        if not any(attempt<0 for attempt in to_attempt):
            to_attempt = [-attempt for attempt in to_attempt] + to_attempt
        use_quad_prec = 1
        curr_attempt = 1

        MLCard.set('WriteOutFilters',True)  
        
        while to_attempt!=[] and need_init():
            curr_attempt = to_attempt.pop()
            # if the attempt is a negative number it means we must force 
            # quadruple precision at initialization time
            if curr_attempt < 0:
                use_quad_prec = -1
                # In quadruple precision we can lower the ZeroThres threshold
                MLCard.set('CTModeInit',4)
                MLCard.set('ZeroThres',1e-11)
            else:
                # Restore the default double precision intialization params
                MLCard.set('CTModeInit',1)
                MLCard.set('ZeroThres',1e-9)
            # Plus one because the filter are written on the next PS point after
            curr_attempt = abs(curr_attempt+1)
            MLCard.set('MaxAttempts',curr_attempt) 
            MLCard.write(pjoin(SubProc_dir,'MadLoopParams.dat'))

            # initialization is performed.
            MadLoopInitializer.fix_PSPoint_in_check(run_dir, read_ps = False, 
                                                         npoints = curr_attempt)
            compile_time, run_time, ram_usage = \
                                        MadLoopInitializer.make_and_run(run_dir)
            if compile_time==None:
                logging.error("Failed at running the process in %s."%run_dir)
                attempts = None
                return None
            # Only set process_compilation time for the first compilation.
            if 'Process_compilation' not in infos.keys() or \
                                             infos['Process_compilation']==None:
                infos['Process_compilation'] = compile_time
            infos['Initialization'] = run_time
        
        MLCard_orig.write(pjoin(SubProc_dir,'MadLoopParams.dat'))
        if need_init():
            return None
        else:
            return use_quad_prec*(curr_attempt-1)

    @staticmethod
    def need_MadLoopInit(proc_dir, subproc_prefix='PV'):
        """Checks whether the necessary filters are present or not."""

        def need_init(ML_resources_path, proc_prefix, r_files):
            """ Returns true if not all required files are present. """
            return any([not os.path.exists(pjoin(ML_resources_path,
                            proc_prefix+fname)) for fname in r_files])

        MLCardPath = pjoin(proc_dir,'SubProcesses','MadLoopParams.dat')
        if not os.path.isfile(MLCardPath):
            raise MadGraph5Error, 'Could not find MadLoopParams.dat at %s.'\
                                                                     %MLCardPath        
        MLCard      = banner_mod.MadLoopParam(MLCardPath) 

        req_files = ['HelFilter.dat','LoopFilter.dat']
        # Make sure that LoopFilter really is needed.
        if not MLCard['UseLoopFilter']:
            try:
                req_files.remove('LoopFilter.dat')
            except ValueError:
                pass
        if MLCard['HelicityFilterLevel']==0:
            try:
                req_files.remove('HelFilter.dat')
            except ValueError:
                pass
        
        for v_folder in glob.iglob(pjoin(proc_dir,'SubProcesses',
                                                         '%s*'%subproc_prefix)):        
            # Make sure it is a valid MadLoop directory
            if not os.path.isdir(v_folder) or not os.path.isfile(\
                                               pjoin(v_folder,'loop_matrix.f')):
                continue
            proc_prefix_file = open(pjoin(v_folder,'proc_prefix.txt'),'r')
            proc_prefix = proc_prefix_file.read()
            proc_prefix_file.close()
            if need_init(pjoin(proc_dir,'SubProcesses','MadLoop5_resources'),
                                                        proc_prefix, req_files):
                return True
        
        return False

    @staticmethod
    def init_MadLoop(proc_dir, n_PS=None, subproc_prefix='PV', MG_options=None,
                                                              interface = None):
        """Advanced commands: Compiles and run MadLoop on RAMBO random PS points to initilize the
        filters."""

        logger.debug('Compiling Source materials necessary for MadLoop '+
                                                              'initialization.')
        # Initialize all the virtuals directory, so as to generate the necessary
        # filters (essentially Helcity filter).
        # Make sure that the MadLoopCard has the loop induced settings
        if interface is None:
            misc.compile(arg=['treatCardsLoopNoInit'], cwd=pjoin(proc_dir,'Source'))
        else:
            interface.do_treatcards('all --no_MadLoopInit')
        
        # First make sure that IREGI and CUTTOOLS are compiled if needed
        if os.path.exists(pjoin(proc_dir,'Source','CUTTOOLS')):
            misc.compile(arg=['libcuttools'],cwd=pjoin(proc_dir,'Source'))
        if os.path.exists(pjoin(proc_dir,'Source','IREGI')):
            misc.compile(arg=['libiregi'],cwd=pjoin(proc_dir,'Source'))
        # Then make sure DHELAS and MODEL are compiled
        misc.compile(arg=['libmodel'],cwd=pjoin(proc_dir,'Source'))
        misc.compile(arg=['libdhelas'],cwd=pjoin(proc_dir,'Source'))        
        
        # Now initialize the MadLoop outputs
        logger.info('Initializing MadLoop loop-induced matrix elements '+\
                                                 '(this can take some time)...')

        # Setup parallelization
        if MG_options:
            mcore = cluster.MultiCore(**MG_options)
        else:
            mcore = cluster.MultiCore(nb_core=1)
        def run_initialization_wrapper(run_dir, infos, attempts):
                if attempts is None:
                    n_PS = MadLoopInitializer.run_initialization(
                                                   run_dir=run_dir, infos=infos)
                else:
                    n_PS = MadLoopInitializer.run_initialization(
                                run_dir=run_dir, infos=infos, attempts=attempts)                    
                infos['nPS'] = n_PS
                return 0
    
        def wait_monitoring(Idle, Running, Done):
            if Idle+Running+Done == 0:
                return
            logger.debug('MadLoop initialization jobs: %d Idle, %d Running, %d Done'\
                                                         %(Idle, Running, Done))

        init_info = {}
        # List all virtual folders while making sure they are valid MadLoop folders
        VirtualFolders = [f for f in glob.iglob(pjoin(proc_dir,'SubProcesses',
                    '%s*'%subproc_prefix)) if (os.path.isdir(f) or 
                                      os.path.isfile(pjoin(f,'loop_matrix.f')))]
        logger.debug("Now Initializing MadLoop matrix element in %d folder%s:"%\
                         (len(VirtualFolders),'s' if len(VirtualFolders)>1 else ''))
        logger.debug(', '.join("'%s'"%os.path.basename(v_folder) for v_folder in 
                                                                    VirtualFolders))
        for v_folder in VirtualFolders:
            init_info[v_folder] = {}
            
            # We try all multiples of n_PS from 1 to max_mult, first in DP and then
            # in QP before giving up, or use default values if n_PS is None.
            max_mult = 3
            if n_PS is None:
                # Then use the default list of number of PS points to try
                mcore.submit(run_initialization_wrapper, 
                                   [pjoin(v_folder), init_info[v_folder], None])
            else:
                # Use specific set of PS points
                mcore.submit(run_initialization_wrapper, [pjoin(v_folder),
                       init_info[v_folder],
                       [n_PS*multiplier for multiplier in range(1,max_mult+1)]])

        # Wait for all jobs to finish.
        mcore.wait('',wait_monitoring,update_first=wait_monitoring)
        for v_folder in VirtualFolders:
            init = init_info[v_folder]
            if init['nPS'] is None:
                raise MadGraph5Error, 'Failed the initialization of'+\
                  " loop-induced matrix element '%s'%s."%\
                  (os.path.basename(v_folder),' (using default n_PS points)' if\
                    n_PS is None else ' (trying with a maximum of %d PS points)'\
                                                               %(max_mult*n_PS))
            if init['nPS']==0:
                logger.debug("Nothing to be done in '%s', all filters already "%\
                             os.path.basename(v_folder)+\
                   "present (use the '-r' option to force their recomputation)")
            else:
                logger.debug("'%s' finished using "%os.path.basename(v_folder)+
                  '%d PS points (%s), in %.3g(compil.) + %.3g(init.) secs.'%(
                  abs(init['nPS']),'DP' if init['nPS']>0 else 'QP',
                  init['Process_compilation'],init['Initialization']))
        
        logger.info('MadLoop initialization finished.')        

AskforEditCard = common_run.AskforEditCard

