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
    import madgraph.interface.common_run_interface as common_run
    import madgraph.iolibs.files as files
    import madgraph.iolibs.save_load_object as save_load_object
    import madgraph.various.banner as banner_mod
    import madgraph.various.cluster as cluster
    import madgraph.various.gen_crossxhtml as gen_crossxhtml
    import madgraph.various.sum_html as sum_html
    import madgraph.various.misc as misc
    import madgraph.various.combine_runs as combine_runs

    import models.check_param_card as check_param_card    
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR, ReadWrite
    MADEVENT = False
except ImportError, error:
    if __debug__:
        print error
    # import from madevent directory
    import internal.extended_cmd as cmd
    import internal.common_run_interface as common_run
    import internal.banner as banner_mod
    import internal.misc as misc    
    from internal import InvalidCmd, MadGraph5Error, ReadWrite
    import internal.files as files
    import internal.gen_crossxhtml as gen_crossxhtml
    import internal.save_load_object as save_load_object
    import internal.cluster as cluster
    import internal.check_param_card as check_param_card
    import internal.sum_html as sum_html
    import internal.combine_runs as combine_runs
    MADEVENT = True

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
        logger.info("syntax: generate_events [run_name] [options])")
        logger.info("-- Launch the full chain of script for the generation of events")
        logger.info("   Including possible plotting, shower and detector resolution.")
        logger.info("   Those steps are performed if the related program are installed")
        logger.info("   and if the related card are present in the Cards directory.")
        self.run_options_help([('-f', 'Use default for all questions.'),
                               ('--laststep=', 'argument might be parton/pythia/pgs/delphes and indicate the last level to be run.')])

    
    def help_add_time_of_flight(self):
        logger.info("syntax: add_time_of_flight [run_name|path_to_file] [--treshold=]")
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
        
    def help_compute_widths(self):
        logger.info("syntax: compute_widths Particle [Particles] [--precision=] [--path=Param_card] [--output=PATH]")
        logger.info("-- Compute the widths for the particles specified.")
        logger.info("   By default, this takes the current param_card and overwrites it.") 
        logger.info("   Precision allows to define when to include three/four/... body decays.")
        logger.info("   If this number is an integer then all N-body decay will be included.")      

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

    def help_print_result(self):
        logger.info("syntax: print_result [RUN] [TAG]")
        logger.info("-- show in text format the status of the run (cross-section/nb-event/...)")


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
    
    def check_treatcards(self, args):
        """check that treatcards arguments are valid
           [param|run|all] [--output_dir=] [--param_card=] [--run_card=]
        """
        
        opt = {'output_dir':pjoin(self.me_dir,'Source'),
               'param_card':pjoin(self.me_dir,'Cards','param_card.dat'),
               'run_card':pjoin(self.me_dir,'Cards','run_card.dat')}
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
            elif arg in ['param','run','all']:
                mode = arg
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
                                
        if len(args) > 1:
            self.help_generate_events()
            raise self.InvalidCmd('Too many argument for generate_events command: %s' % cmd)
                    
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
    
    def check_compute_widths(self, args):
        """check that the model is loadable and check that the format is of the
        type: PART PATH --output=PATH -f --precision=N
        return the model.
        """
        
        # Check that MG5 directory is present .
        if MADEVENT and not self.options['mg5_path']:
            raise self.InvalidCmd, '''The automatic computations of widths requires that MG5 is installed on the system.
            You can install it and set his path in ./Cards/me5_configuration.txt'''
        elif MADEVENT:
            sys.path.append(self.options['mg5_path'])
        try:
            import models.model_reader as model_reader
            import models.import_ufo as import_ufo
        except ImportError:
            raise self.ConfigurationError, '''Can\'t load MG5.
            The variable mg5_path should not be correctly configure.'''
        
        ufo_path = pjoin(self.me_dir,'bin','internal', 'ufomodel')
        # Import model
        if not MADEVENT:
            modelname = self.find_model_name()
            #restrict_file = None
            #if os.path.exists(pjoin(ufo_path, 'restrict_default.dat')):
            #    restrict_file = pjoin(ufo_path, 'restrict_default.dat')
            model = import_ufo.import_model(modelname, decay=True, 
                        restrict=True)
            if self.mother and self.mother.options['complex_mass_scheme']:
                model.change_mass_to_complex_scheme()
        else:
            model = import_ufo.import_model(pjoin(self.me_dir,'bin','internal', 'ufomodel'),
                                        decay=True)
            #pattern for checking complex mass scheme.
            has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
            if has_cms.search(open(pjoin(self.me_dir,'Cards','proc_card_mg5.dat')\
                                   ).read()):
                model.change_mass_to_complex_scheme()
   
            
#        if not hasattr(model.get('particles')[0], 'partial_widths'):
#            raise self.InvalidCmd, 'The UFO model does not include partial widths information. Impossible to compute widths automatically'
            
        # check if the name are passed to default MG5
        if '-modelname' in open(pjoin(self.me_dir,'Cards','proc_card_mg5.dat')).read():
            model.pass_particles_name_in_mg_default()        
        model = model_reader.ModelReader(model)
        particles_name = dict([(p.get('name'), p.get('pdg_code'))
                                               for p in model.get('particles')])
        particles_name.update(dict([(p.get('antiname'), p.get('pdg_code'))
                                               for p in model.get('particles')]))        
        
        output = {'model': model, 'force': False, 'output': None, 
                  'path':None, 'particles': set(), 'body_decay':4.0025,
                  'min_br':None, 'precision_channel':0.01}
        for arg in args:
            if arg.startswith('--output='):
                output_path = arg.split('=',1)[1]
                if not os.path.exists(output_path):
                    raise self.InvalidCmd, 'Invalid Path for the output. Please retry.'
                if not os.path.isfile(output_path):
                    output_path = pjoin(output_path, 'param_card.dat')
                output['output'] = output_path       
            elif arg == '-f':
                output['force'] = True
            elif os.path.isfile(arg):
                type = self.detect_card_type(arg)
                if type != 'param_card.dat':
                    raise self.InvalidCmd , '%s is not a valid param_card.' % arg
                output['path'] = arg
            elif arg.startswith('--path='):
                arg = arg.split('=',1)[1]
                type = self.detect_card_type(arg)
                if type != 'param_card.dat':
                    raise self.InvalidCmd , '%s is not a valid param_card.' % arg
                output['path'] = arg
            elif arg.startswith('--'):
                name, value = arg.split('=',1)
                try:
                    value = float(value)
                except Exception:
                    raise self.InvalidCmd, '--%s requires integer or a float' % name
                output[name[2:]] = float(value)                
            elif arg in particles_name:
                # should be a particles
                output['particles'].add(particles_name[arg])
            elif arg.isdigit() and int(arg) in particles_name.values():
                output['particles'].add(eval(arg))
            elif arg == 'all':
                output['particles'] = set(['all'])
            else:
                self.help_compute_widths()
                raise self.InvalidCmd, '%s is not a valid argument for compute_widths' % arg
        if self.force:
            output['force'] = True

        if not output['particles']:
            raise self.InvalidCmd, '''This routines requires at least one particle in order to compute
            the related width'''
            
        if output['output'] is None:
            output['output'] = output['path']

        return output
    
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
            files.cp(input_file, output_file)
        else:
            os.system('gunzip -c %s > %s' % (input_file, output_file))
        
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

    def complete_launch(self, *args, **opts):

        if self.ninitial == 1:
            return self.complete_calculate_decay_widths(*args, **opts)
        else:
            return self.complete_generate_events(*args, **opts)

    def complete_compute_widths(self, text, line, begidx, endidx):
        "Complete the compute_widths command"

        args = self.split_arg(line[0:begidx])
        
        if args[-1] in  ['--path=', '--output=']:
            completion = {'path': self.path_completion(text)}
        elif line[begidx-1] == os.path.sep:
            current_dir = pjoin(*[a for a in args if a.endswith(os.path.sep)])
            if current_dir.startswith('--path='):
                current_dir = current_dir[7:]
            if current_dir.startswith('--output='):
                current_dir = current_dir[9:]                
            completion = {'path': self.path_completion(text, current_dir)}
        else:
            completion = {}            
            completion['options'] = self.list_completion(text, 
                            ['--path=', '--output=', '--min_br=0.\$'
                             '--precision_channel=0.\$', '--body_decay='])            
        
        return self.deal_multiple_categories(completion)

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

    def complete_print_results(self,text, line, begidx, endidx):
        "Complete the print results command"
        args = self.split_arg(line[0:begidx], error=False) 
        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*','unweighted_events.lhe.gz'))
            data = [n.rsplit('/',2)[1] for n in data]
            tmp1 =  self.list_completion(text, data)
            return tmp1        
        else:
            data = glob.glob(pjoin(self.me_dir, 'Events', args[0], '*_pythia_events.hep.gz'))
            data = [os.path.basename(p).rsplit('_',1)[0] for p in data]
            tmp1 =  self.list_completion(text, data)
            return tmp1
            



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
            self.results = save_load_object.load_from_file(pjoin(self.me_dir,'HTML','results.pkl'))
            self.results.resetall(self.me_dir)
        else:
            model = self.find_model_name()
            process = self.process # define in find_model_name
            self.results = gen_crossxhtml.AllResults(model, process, self.me_dir)
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
            if path is None:
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
            subprocess.call(['gunzip', event_path])
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

        cst = 6.58211915e-25
        # Loop over all events
        for event in lhe:
            for particle in event:
                id = particle.pid
                width = param_card['decay'].get((abs(id),)).value
                if width:
                    vtim = random.expovariate(width/cst)
                    if vtim > threshold:
                        particle.vtim = vtim
            #write this modify event
            output.write(str(event))
        output.write('</LesHouchesEvents>\n')
        output.close()
        
        files.mv('%s_2vertex.lhe' % event_path, event_path)
        
        if need_zip:
            subprocess.call(['gzip', event_path])
        
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
  
 

    ############################################################################ 
    def do_print_results(self, line):
        """Not in help:Print the cross-section/ number of events for a given run"""
        
        args = self.split_arg(line)
        options={'path':None, 'mode':'w'}
        for arg in list(args):
            if arg.startswith('--') and '=' in arg:
                name,value=arg.split('=',1)
                name = name [2:]
                options[name] = value
                args.remove(arg)
        
        
        if len(args) > 0:
            run_name = args[0]
        else:
            if not self.results.current:
                raise self.InvalidCmd('no run currently defined. Please specify one.')
            else:
                run_name = self.results.current['run_name']
        if run_name not in self.results:
            raise self.InvalidCmd('%s is not a valid run_name or it doesn\'t have any information' \
                                  % run_name)

            
        if len(args) == 2:
            tag = args[1]
            if tag.isdigit():
                tag = int(tag) - 1
                if len(self.results[run_name]) < tag:
                    raise self.InvalidCmd('Only %s different tag available' % \
                                                    len(self.results[run_name]))
                data = self.results[run_name][tag]
            else:
                data = self.results[run_name].return_tag(tag)
        else:
            data = self.results[run_name].return_tag(None) # return the last
        
        if options['path']:
            self.print_results_in_file(data, options['path'], options['mode'])
        else:
            self.print_results_in_shell(data)
        

    ############################################################################

    ############################################################################
    def do_generate_events(self, line):
        """Main Commands: launch the full chain """
        
        args = self.split_arg(line)
        # Check argument's validity
        mode = self.check_generate_events(args)
        self.ask_run_configuration(mode)
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
            self.exec_cmd('create_gridpack', postcmd=False)
        else:
            # Regular run mode
            logger.info('Generating %s events with run name %s' %
                        (self.run_card['nevents'], self.run_name))
        
            self.exec_cmd('survey  %s %s' % (self.run_name,' '.join(args)),
                          postcmd=False)
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
            nb_event = self.run_card['nevents']
            self.exec_cmd('refine %s' % nb_event, postcmd=False)
            self.exec_cmd('refine %s' % nb_event, postcmd=False)
            self.exec_cmd('combine_events', postcmd=False)
            self.print_results_in_shell(self.results.current)
            self.run_syscalc('parton')
            self.create_plot('parton')
            self.exec_cmd('store_events', postcmd=False)
            self.exec_cmd('reweight -from_cards', postcmd=False)
            self.exec_cmd('decay_events -from_cards', postcmd=False)
            self.exec_cmd('pythia --no_default', postcmd=False, printcmd=False)
            # pythia launches pgs/delphes if needed
            self.store_result()
            
    
    def do_launch(self, line, *args, **opt):
        """Main Commands: exec generate_events for 2>N and calculate_width for 1>N"""
        if self.ninitial == 1:
            self.do_calculate_decay_widths(line, *args, **opt)
        else:
            self.do_generate_events(line, *args, **opt)
            
    def print_results_in_shell(self, data):
        """Have a nice results prints in the shell,
        data should be of type: gen_crossxhtml.OneTagResults"""
        if not data:
            return
        logger.info("  === Results Summary for run: %s tag: %s ===\n" % (data['run_name'],data['tag']))
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

    def print_results_in_file(self, data, path, mode='w'):
        """Have a nice results prints in the shell,
        data should be of type: gen_crossxhtml.OneTagResults"""
        if not data:
            return
        
        fsock = open(path, mode)
        
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
    
    ############################################################################      
    def do_calculate_decay_widths(self, line):
        """Main Commands: launch decay width calculation and automatic inclusion of
        calculated widths and BRs in the param_card."""

        args = self.split_arg(line)
        # Check argument's validity
        accuracy = self.check_calculate_decay_widths(args)
        self.ask_run_configuration('parton')
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
        self.exec_cmd('combine_events', postcmd=False)
        self.exec_cmd('store_events', postcmd=False)
        
        self.collect_decay_widths()
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
            os.system('gunzip %s/%s/unweighted_events.lhe.gz' % 
                                  (pjoin(self.me_dir,'Events'), self.run_name))
            self.create_root_file('%s/unweighted_events.lhe' % self.run_name,
                                  '%s/unweighted_events.root' % self.run_name)
            
        
        self.create_plot('parton', '%s/%s/unweighted_events.lhe' %
                         (pjoin(self.me_dir, 'Events'),self.run_name),
                         pjoin(self.me_dir, 'HTML',self.run_name, 'plots_parton.html')
                         )
        
        os.system('gzip -f %s/%s/unweighted_events.lhe' % 
                                  (pjoin(self.me_dir, 'Events'), self.run_name))

        self.update_status('', level='parton')
        self.print_results_in_shell(self.results.current)   
        

    ############################################################################      
    def do_treatcards(self, line, mode=None, opt=None):
        """Advanced commands: create .inc files from param_card.dat/run_card.dat"""


        if not mode and not opt:
            args = self.split_arg(line)
            mode,  opt  = self.check_treatcards(args)
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
            param_card.write_inc_file(outfile, ident_card, default)
      
      
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

        logger.info('Working on SubProcesses')
        self.total_jobs = 0
        subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                 'subproc.mg'))]
        #check difficult PS case
        if float(self.run_card['mmjj']) > 0.01 * (float(self.run_card['ebeam1'])+float(self.run_card['ebeam2'])):
            self.pass_in_difficult_integration_mode()
          
        P_zero_result = [] # check the number of times where they are no phase-space

        nb_tot_proc = len(subproc)
        for nb_proc,subdir in enumerate(subproc):
            self.update_status('Compiling for process %s/%s. <br> (previous processes already running)' % \
                               (nb_proc+1,nb_tot_proc), level=None)
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(match)
            for match in glob.glob(pjoin(Pdir, 'G*')):
                if os.path.exists(pjoin(match,'results.dat')):
                    os.remove(pjoin(match, 'results.dat'))
            
            #compile gensym
            self.compile(['gensym'], cwd=Pdir)
            if not os.path.exists(pjoin(Pdir, 'gensym')):
                raise MadEventError, 'Error make gensym not successful'

            # Launch gensym
            p = misc.Popen(['./gensym'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, cwd=Pdir)
            sym_input = "%(points)d %(iterations)d %(accuracy)f \n" % self.opts
            (stdout, stderr) = p.communicate(sym_input)
            if os.path.exists(pjoin(self.me_dir,'error')):
                files.mv(pjoin(self.me_dir,'error'), pjoin(Pdir,'ajob.no_ps.log'))
                P_zero_result.append(subdir)
                continue
            
            if not os.path.exists(pjoin(Pdir, 'ajob1')) or p.returncode:
                logger.critical(stdout)
                raise MadEventError, 'Error gensym run not successful'


            self.compile(['madevent'], cwd=Pdir)
            
            alljobs = glob.glob(pjoin(Pdir,'ajob*'))
            self.total_jobs += len(alljobs)
            for i, job in enumerate(alljobs):
                job = os.path.basename(job)
                self.launch_job('%s' % job, cwd=Pdir, remaining=(len(alljobs)-i-1), 
                                                    run_type='survey on %s (%s/%s)' % (subdir,nb_proc+1,len(subproc)))
                if os.path.exists(pjoin(self.me_dir,'error')):
                    self.monitor(html=False)
                    raise MadEventError, 'Error detected Stop running: %s' % \
                                         open(pjoin(self.me_dir,'error')).read()
                                         
        # Check if all or only some fails
        if P_zero_result:
            if len(P_zero_result) == len(subproc):
                raise ZeroResult, '%s' % \
                    open(pjoin(Pdir,'ajob.no_ps.log')).read()
            else:
                logger.warning(''' %s SubProcesses doesn\'t have available phase-space.
            Please check mass spectrum.''' % ','.join(P_zero_result))
                
        
        self.monitor(run_type='All jobs submitted for survey', html=True)
        cross, error = sum_html.make_all_html_results(self)
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error) 
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
        conf_path = pjoin(self.me_dir, 'Source','run_config.inc')
        files.cp(conf_path, conf_path + '.bk')

        text = open(conf_path).read()
        text = re.sub('''\(min_events = \d+\)''', '''(min_events = 7500 )''', text)
        text = re.sub('''\(max_events = \d+\)''', '''(max_events = 20000 )''', text)
        fsock = open(conf_path, 'w')
        fsock.write(text)
        fsock.close()
        
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
        
        precision = args[0]
        if len(args) == 2:
            max_process = args[1]
        else:
            max_process = 5

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
        for nb_proc,subdir in enumerate(subproc):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            bindir = pjoin(os.path.relpath(self.dirbin, Pdir))
                           
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(match)
            
            proc = misc.Popen([pjoin(bindir, 'gen_ximprove')],
                                    stdout=devnull,
                                    stdin=subprocess.PIPE,
                                    cwd=Pdir)
            proc.communicate('%s %s T\n' % (precision, max_process))

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
        
        bindir = pjoin(os.path.relpath(self.dirbin, pjoin(self.me_dir,'SubProcesses')))

        combine_runs.CombineRuns(self.me_dir)
        
        cross, error = sum_html.make_all_html_results(self)
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)   

        self.update_status('finish refine', 'parton', makehtml=False)
        devnull.close()
        
    ############################################################################ 
    def do_combine_events(self, line):
        """Advanced commands: Launch combine events"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_combine_events(args)

        self.update_status('Combining Events', level='parton')
        try:
            os.remove(pjoin(self.me_dir,'SubProcesses', 'combine.log'))
        except Exception:
            pass
        self.cluster.launch_and_wait('../bin/internal/run_combine', 
                                        cwd=pjoin(self.me_dir,'SubProcesses'),
                                        stdout=pjoin(self.me_dir,'SubProcesses', 'combine.log'),
                                        required_output=[pjoin(self.me_dir,'SubProcesses', 'combine.log')])
        
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
        
        
        self.banner.add_to_file(pjoin(self.me_dir,'Events', 'events.lhe'))
        self.banner.add_to_file(pjoin(self.me_dir,'Events', 'unweighted_events.lhe'))        

        
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
    def do_compute_widths(self, line):
        """Require MG5 directory: Compute automatically the widths of a set 
        of particles"""

        args = self.split_arg(line)
        opts = self.check_compute_widths(args)
        
        
        from madgraph.interface.master_interface import MasterCmd
        cmd = MasterCmd()
        self.define_child_cmd_interface(cmd, interface=False)
        cmd.exec_cmd('set automatic_html_opening False --no_save')
        if not opts['path']:
            opts['path'] = pjoin(self.me_dir, 'Cards', 'param_card.dat')
            if not opts['force'] :
                self.ask_edit_cards(['param_card'],[], plot=False)
        
        
        line = 'compute_widths %s %s' % \
                (' '.join([str(i) for i in opts['particles']]),
                 ' '.join('--%s=%s' % (key,value) for (key,value) in opts.items()
                        if key not in ['model', 'force', 'particles'] and value))
        
        cmd.exec_cmd(line, model=opts['model'])
        self.child = None
        del cmd


    
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
        for P_path in SubProcesses.get_subP(self.me_dir):
            G_dir = [G for G in os.listdir(P_path) if G.startswith('G') and 
                                                os.path.isdir(pjoin(P_path,G))]
            for G in G_dir:
                G_path = pjoin(P_path,G)
                # Remove events file (if present)
                if os.path.exists(pjoin(G_path, 'events.lhe')):
                    os.remove(pjoin(G_path, 'events.lhe'))
                # Store results.dat
                if os.path.exists(pjoin(G_path, 'results.dat')):
                    input = pjoin(G_path, 'results.dat')
                    output = pjoin(G_path, '%s_results.dat' % run)
                    files.cp(input, output) 
                # Store log
                if os.path.exists(pjoin(G_path, 'log.txt')):
                    input = pjoin(G_path, 'log.txt')
                    output = pjoin(G_path, '%s_log.txt' % run)
                    files.mv(input, output) 
                # Grid
                for name in ['ftn26']:
                    if os.path.exists(pjoin(G_path, name)):
                        if os.path.exists(pjoin(G_path, '%s_%s.gz'%(run,name))):
                            os.remove(pjoin(G_path, '%s_%s.gz'%(run,name)))
                        input = pjoin(G_path, name)
                        output = pjoin(G_path, '%s_%s' % (run,name))
                        files.mv(input, output) 
                        misc.call(['gzip', output], stdout=devnull, 
                                        stderr=devnull, cwd=G_path)
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
            if os.path.exists(pjoin(E_path, name)):
                if os.path.exists(pjoin(O_path, '%s.gz' % name)):
                    os.remove(pjoin(O_path, '%s.gz' % name))
                input = pjoin(E_path, name)
                output = pjoin(O_path, name)
                files.mv(input, output) 
                misc.call(['gzip', output], stdout=devnull, stderr=devnull, 
                                                                     cwd=O_path) 
        self.update_status('End Parton', level='parton', makehtml=False)
        devnull.close()
    
    ############################################################################
    def do_reweight(self, line):
        """ Allow to reweight the events generated with a new choices of model
            parameter.
        """
        
        if '-from_cards' in line and not os.path.exists(pjoin(self.me_dir, 'Cards', 'reweight_card.dat')):
            return
        
        # Check that MG5 directory is present .
        if MADEVENT and not self.options['mg5_path']:
            raise self.InvalidCmd, '''The module reweight requires that MG5 is installed on the system.
            You can install it and set its path in ./Cards/me5_configuration.txt'''
        elif MADEVENT:
            sys.path.append(self.options['mg5_path'])
        try:
            import madgraph.interface.reweight_interface as reweight_interface
        except ImportError:
            raise self.ConfigurationError, '''Can\'t load Reweight module.
            The variable mg5_path might not be correctly configured.'''
        
        self.to_store.append('event')
        if not '-from_cards' in line:
            self.keep_cards(['reweight_card.dat'])
            self.ask_edit_cards(['reweight_card.dat'], 'fixed', plot=False)        

        # forbid this function to create an empty item in results.
        if self.results.current['cross'] == 0 and self.run_name:
            self.results.delete_run(self.run_name, self.run_tag)

        # load the name of the event file
        args = self.split_arg(line) 
        self.check_decay_events(args) 
        # args now alway content the path to the valid files
        reweight_cmd = reweight_interface.ReweightInterface(args[0])
        reweight_cmd. mother = self
        self.update_status('Running Reweight', level='madspin')
        
        
        path = pjoin(self.me_dir, 'Cards', 'reweight_card.dat')
        reweight_cmd.me_dir = self.me_dir
        reweight_cmd.import_command_file(path)
        
        # re-define current run
        try:
            self.results.def_current(self.run_name, self.run_tag)
        except Exception:
            pass
        
    ############################################################################ 
    def do_create_gridpack(self, line):
        """Advanced commands: Create gridpack from present run"""

        self.update_status('Creating gridpack', level='parton')
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
        if not self.banner:
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
                    error_m = math.sqrt((error * Nacc/Ntry)**2 + sigma_m**2 *(1-Nacc/Ntry)/Nacc)
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
            misc.call(['gzip','-f','beforeveto.tree'], 
                                                cwd=pjoin(self.me_dir,'Events'))
            files.mv(pjoin(self.me_dir,'Events','beforeveto.tree.gz'), 
                     pjoin(self.me_dir,'Events',self.run_name, tag+'_pythia_beforeveto.tree.gz'))
             
        if self.run_card['use_syst'] in self.true:
            # Calculate syscalc info based on syst.dat
            try:
                self.run_syscalc('Pythia')
            except SysCalcError, error:
                logger.error(str(error))
            else:
                # Store syst.dat
                subprocess.call(['gzip','-f','syst.dat'],
                                cwd=pjoin(self.me_dir,'Events'))          
                files.mv(pjoin(self.me_dir,'Events','syst.dat.gz'), 
                         pjoin(self.me_dir,'Events',self.run_name, tag + '_pythia_syst.dat.gz'))
                # Store syscalc.dat
                if os.path.exists(pjoin(self.me_dir, 'Events', 'syscalc.dat')):
                    filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                              '%s_syscalc.dat' % self.run_tag)
                    shutil.move(pjoin(self.me_dir, 'Events','syscalc.dat'), 
                                filename)
                    os.system('gzip -f %s' % filename)

        # Plot for pythia
        self.create_plot('Pythia')

        if os.path.exists(pjoin(self.me_dir,'Events','pythia_events.lhe')):
            shutil.move(pjoin(self.me_dir,'Events','pythia_events.lhe'),
            pjoin(self.me_dir,'Events', self.run_name,'%s_pythia_events.lhe' % tag))
            os.system('gzip -f %s' % pjoin(self.me_dir,'Events', self.run_name,
                                        '%s_pythia_events.lhe' % tag))      

        
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
                os.system('gunzip -f %s' % (filename+'.gz') )
            if  os.path.exists(filename):
                shutil.move(filename, pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'))
                self.create_plot('parton')
                shutil.move(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'), filename)
                os.system('gzip -f %s' % filename)
            else:
                logger.info('No valid files for partonic plot') 
                
        if any([arg in ['all','pythia'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                          '%s_pythia_events.lhe' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                os.system('gunzip -f %s' % (filename+'.gz') )
            if  os.path.exists(filename):
                shutil.move(filename, pjoin(self.me_dir, 'Events','pythia_events.lhe'))
                self.create_plot('Pythia')
                shutil.move(pjoin(self.me_dir, 'Events','pythia_events.lhe'), filename)
                os.system('gzip -f %s' % filename)                
            else:
                logger.info('No valid files for pythia plot')
                
                    
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
                
        if any([arg in ['all','parton'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe')
            if os.path.exists(filename+'.gz'):
                os.system('gunzip -f %s' % (filename+'.gz') )
            if  os.path.exists(filename):
                shutil.move(filename, pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'))
                self.run_syscalc('parton')
                shutil.move(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'), filename)
                os.system('gzip -f %s' % filename)
            else:
                logger.info('No valid files for parton level systematics run.')
                
        if any([arg in ['all','pythia'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                          '%s_pythia_syst.dat' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                os.system('gunzip -f %s' % (filename+'.gz') )
            if  os.path.exists(filename):
                shutil.move(filename, pjoin(self.me_dir, 'Events','syst.dat'))
                try:
                    self.run_syscalc('Pythia')
                except SysCalcError, error:
                    logger.warning(str(error))
                    return
                    
                shutil.move(pjoin(self.me_dir, 'Events','syst.dat'), filename)
                os.system('gzip -f %s' % filename)                
                filename = pjoin(self.me_dir, 'Events' ,self.run_name,
                                          '%s_syscalc.dat' % self.run_tag)
                shutil.move(pjoin(self.me_dir, 'Events','syscalc.dat'), 
                            filename)
                os.system('gzip -f %s' % filename)
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
        self.update_status('storring files of Previous run', level=None,\
                                                     error=True)
        if 'event' in self.to_store:
            if not os.path.exists(pjoin(self.me_dir, 'Events',self.run_name, 'unweighted_events.lhe.gz')):
                os.system('gzip -f %s/unweighted_events.lhe' % \
                                     pjoin(self.me_dir,'Events',self.run_name) )                
            
        
        if 'pythia' in self.to_store:
            self.update_status('Storing Pythia files of Previous run', level='pythia', error=True)
            os.system('mv -f %(path)s/pythia_events.hep %(path)s/%(name)s/%(tag)s_pythia_events.hep' % 
                  {'name': self.run_name, 'path' : pjoin(self.me_dir,'Events'),
                   'tag':tag})
            os.system('gzip -f %s/%s_pythia_events.hep' % ( 
                                pjoin(self.me_dir,'Events',self.run_name), tag))
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
            status = misc.call(['./'+exe] + argument, cwd=cwd, 
                                                           stdout=stdout, **opt)
            logger.info('%s run in %f s' % (exe, time.time() -start))
            if status:
                raise MadGraph5Error, '%s didn\'t stop properly. Stop all computation' % exe


        elif mode in [1,2]:
            # For condor cluster, create the input/output files
            if 'ajob' in exe: 
                input_files = ['madevent','input_app.txt','symfact.dat','iproc.dat',
                               pjoin(self.me_dir, 'SubProcesses','randinit')]
                output_files = []
                required_output = []
                

                #Find the correct PDF input file
                input_files.append(self.get_pdf_input_filename())
                        
                #Find the correct ajob
                Gre = re.compile("\s*j=(G[\d\.\w]+)")
                Ire = re
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
                
                #submitting
                self.cluster.submit2(exe, stdout=stdout, cwd=cwd, 
                             input_files=input_files, output_files=output_files,
                             required_output=required_output)
            
            else:
                self.cluster.submit(exe, stdout=stdout, cwd=cwd)
            
            
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
            else:
                update_status = lambda idle, run, finish: None
            try:    
                self.cluster.wait(self.me_dir, update_status)            
            except Exception, error:
                logger.info(error)
                if not self.force:
                    ans = self.ask('Cluster Error detected. Do you want to clean the queue?',
                             default = 'y', choices=['y','n'])
                else:
                    ans = 'y'
                if ans == 'y':
                    self.cluster.remove()
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

        # set environment variable for lhapdf.
        if self.run_card['pdlabel'] == "lhapdf":
            os.environ['lhapdf'] = 'True'
            self.link_lhapdf(pjoin(self.me_dir,'lib'))
            pdfsetsdir = subprocess.Popen('%s --pdfsets-path' % self.options['lhapdf'],
                    shell = True, stdout = subprocess.PIPE).stdout.read().strip()
            lhaid_list = [int(self.run_card['lhaid'])]
            self.copy_lhapdf_set(lhaid_list, pdfsetsdir)
        elif 'lhapdf' in os.environ.keys():
            del os.environ['lhapdf']
        if self.run_card['pdlabel'] != "lhapdf":
            self.pdffile = None
            #remove lhapdf stuff
            self.compile(arg=['clean_lhapdf'], cwd=os.path.join(self.me_dir, 'Source'))
            
        # set random number
        if self.run_card['iseed'] != '0':
            self.random = int(self.run_card['iseed'])
            self.run_card['iseed'] = '0'
            # Reset seed in run_card to 0, to ensure that following runs
            # will be statistically independent
            text = open(pjoin(self.me_dir, 'Cards','run_card.dat')).read()
            (t,n) = re.subn(r'\d+\s*= iseed','0 = iseed',text)
            open(pjoin(self.me_dir, 'Cards','run_card.dat'),'w').write(t)
        elif os.path.exists(pjoin(self.me_dir,'SubProcesses','randinit')):
            for line in open(pjoin(self.me_dir,'SubProcesses','randinit')):
                data = line.split('=')
                assert len(data) ==2
                self.random = int(data[1])
                break
        else:
            self.random = random.randint(1, 30107)
                                                               
        if self.run_card['ickkw'] == '2':
            logger.info('Running with CKKW matching')
            self.treat_CKKW_matching()
            
        # create param_card.inc and run_card.inc
        self.do_treatcards('')
        
        # Compile
        for name in ['../bin/internal/gen_ximprove', 'all', 
                     '../bin/internal/combine_events']:
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
                os.system('gunzip -fc %s > %s' % (issudfile, path))
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
                    os.system('gzip -fc %s > %s' % (path, issudfile))
                                     
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

        logger.info('running syscalc on mode %s' % mode)
        if self.run_card['use_syst'] not in self.true:
            return
        
        scdir = self.options['syscalc_path']
        tag = self.run_card['run_tag']  
        card = pjoin(self.me_dir, 'bin','internal', 'syscalc_card.dat')
        template = open(pjoin(self.me_dir, 'bin','internal', 'syscalc_template.dat')).read()
        self.run_card['sys_pdf'] = self.run_card['sys_pdf'].split('#',1)[0].replace('&&',' \n ')
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
                event_path = pjoin(event_dir,'unweighted_events.lhe')
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
                os.system('gzip -f %s.gz ' % event_path)
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
    def ask_run_configuration(self, mode=None):
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
                       'reweight':'Add weight to events based on coupling parameters:',
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
        if not MADEVENT or self.options['mg5_path']:
            available_mode.append('4')
            available_mode.append('5')
            if os.path.exists(pjoin(self.me_dir,'Cards','madspin_card.dat')):
                switch['madspin'] = 'ON'
            else:
                switch['madspin'] = 'OFF'
            if os.path.exists(pjoin(self.me_dir,'Cards','reweight_card.dat')):
                switch['reweight'] = 'ON'
            else:
                switch['reweight'] = 'OFF'
                 


        options = list(available_mode) + ['auto', 'done']
        for id, key in enumerate(switch_order):
            if switch[key] != void:
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
                    switch_format = " %i %-50s %10s=%s\n"
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
                
  
            
    def check_param_card(self, path, run=True):
        """Check that all the width are define in the param_card.
        If some width are set on 'Auto', call the computation tools."""
        
        pattern = re.compile(r'''decay\s+(\+?\-?\d+)\s+auto''',re.I)
        text = open(path).read()
        pdg = pattern.findall(text)
        if pdg:
            if run:
                logger.info('Computing the width set on auto in the param_card.dat')
                self.do_compute_widths('%s %s' % (' '.join(pdg), path))
            else:
                logger.info('''Some width are on Auto in the card. 
    Those will be computed as soon as you have finish the edition of the cards.
    If you want to force the computation right now and being able to re-edit
    the cards afterwards, you can type \"compute_wdiths\".''')
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
            proc = misc.Popen([pjoin(bindir, 'gen_ximprove')],
                                    stdin=subprocess.PIPE,
                                    stdout=open(logfile,'w'),
                                    cwd=Pdir)
            proc.communicate('%s 1 F\n' % (precision))

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



AskforEditCard = common_run.AskforEditCard

