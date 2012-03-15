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
    import madgraph.iolibs.save_load_object as save_load_object
    import madgraph.various.banner as banner_mod
    import madgraph.various.cluster as cluster
    import madgraph.various.gen_crossxhtml as gen_crossxhtml
    import madgraph.various.sum_html as sum_html
    import madgraph.various.misc as misc

    import models.check_param_card as check_param_card    
    from madgraph import InvalidCmd, MadGraph5Error
    MADEVENT = False
except Exception, error:
    if __debug__:
        print error
    # import from madevent directory
    import internal.extended_cmd as cmd
    import internal.banner as banner_mod
    import internal.misc as misc    
    from internal import InvalidCmd, MadGraph5Error
    import internal.files as files
    import internal.gen_crossxhtml as gen_crossxhtml
    import internal.save_load_object as save_load_object
    import internal.cluster as cluster
    import internal.check_param_card as check_param_card
    import internal.sum_html as sum_html
    MADEVENT = True



class MadEventError(Exception):
    pass

#===============================================================================
# CmdExtended
#===============================================================================
class CmdExtended(cmd.Cmd):
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
            in order to quit madevent please enter exit"""
    
    # Define the Error
    InvalidCmd = InvalidCmd
    ConfigurationError = MadGraph5Error

    
    def __init__(self, *arg, **opt):
        """Init history and line continuation"""
        
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
        '#*                    MadGraph/MadEvent 5                   *\n' + \
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
        "*           W E L C O M E  to  M A D G R A P H  5          *\n" + \
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
                        
        if not self.use_rawinput:
            return stop
        
        if self.results and not self.results.current:
            return stop
        
        arg = line.split()
        if  len(arg) == 0:
            return stop
        if self.results.status.startswith('Error'):
            return stop
        if self.results.status == 'Stop by the user':
            self.update_status('%s Stop by the user' % arg[0], level=None, error=True)
            return stop        
        elif not self.results.status:
            return stop
        elif str(arg[0]) in ['exit','quit','EOF']:
            return stop
        
        try:
            self.update_status('Command \'%s\' done.<br> Waiting for instruction.' % arg[0], 
                               level=None, error=True)
        except:
            pass
        
    def add_error_log_in_html(self):
        """If a ME run is currently running add a link in the html output"""

        # Be very carefull to not raise any error here (the traceback 
        #will modify in that case.
        if hasattr(self, 'results') and hasattr(self.results, 'current') and\
                self.results.current and 'run_name' in self.results.current and \
                hasattr(self, 'me_dir'):
            name = self.results.current['run_name']
            tag = self.results.current['tag']
            self.debug_output = pjoin(self.me_dir, '%s_%s_debug.log' % (name,tag))
            self.results.current.debug = self.debug_output
        else:
            #Force class default
            self.debug_output = MadEventCmd.debug_output
        if os.path.exists('ME5_debug') and not 'ME5_debug' in self.debug_output:
            os.remove('ME5_debug')
        if not 'ME5_debug' in self.debug_output:
            os.system('ln -s %s ME5_debug &> /dev/null' % self.debug_output)

    
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
        

    def help_generate_events(self):
        logger.info("syntax: generate_events [run_name] [options])")
        logger.info("-- Launch the full chain of script for the generation of events")
        logger.info("   Including possible plotting, shower and detector resolution.")
        logger.info("   Those steps are performed if the related program are installed")
        logger.info("   and if the related card are present in the Cards directory.")
        self.run_options_help([('-f', 'Use default for all questions.'),
                               ('--laststep=', 'argument might be parton/pythia/pgs/delphes and indicate the last level to be run.')])

    def help_calculate_decay_widths(self):
        logger.info("syntax: calculate_decay_widths [run_name] [options])")
        logger.info("-- Calculate decay widths and enter widths and BRs in param_card")
        logger.info("   for a series of processes of type A > B C ...")
        logger.info("   Note that you want to do \"set group_subprocesses False\"")
        logger.info("   before generating the processes.")
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
        
    def help_refine(self):
        logger.info("syntax: refine require_precision [max_channel] [--run_options]")
        logger.info("-- refine the LAST run to achieve a given precision.")
        logger.info("   require_precision: can be either the targeted number of events")
        logger.info('                      or the required relative error')
        logger.info('   max_channel:[5] maximal number of channel per job')
        self.run_options_help([])
        
    def help_combine_events(self):
        """ """
        logger.info("syntax: combine_events [--run_options]")
        logger.info("-- Combine the last run in order to write the number of events")
        logger.info("   require in the run_card.")
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
        
    def help_plot(self):
        logger.info("syntax: help [RUN] [%s] [-f]" % '|'.join(self._plot_mode))
        logger.info("-- create the plot for the RUN (current run by default)")
        logger.info("     at the different stage of the event generation")
        logger.info("     Note than more than one mode can be specified in the same command.")
        logger.info("   This require to have MadAnalysis and td require. By default")
        logger.info("     if those programs are installed correctly, the creation")
        logger.info("     will be performed automaticaly during the event generation.")
        logger.info("   -f options: answer all question by default.")
        
    def help_remove(self):
        logger.info("syntax: remove RUN [all|parton|pythia|pgs|delphes|banner] [-f] [--tag=]")
        logger.info("-- Remove all the files linked to previous run RUN")
        logger.info("   if RUN is 'all', then all run will be cleaned.")
        logger.info("   The optional argument precise which part should be cleaned.")
        logger.info("   By default we clean all the related files but the banners.")
        logger.info("   the optional '-f' allows to by-pass all security question")
        logger.info("   The banner can be remove only if all files are removed first.")

    def help_pythia(self):
        logger.info("syntax: pythia [RUN] [--run_options]")
        logger.info("-- run pythia on RUN (current one by default)")
        self.run_options_help([('-f','answer all question by default'),
                               ('--tag=', 'define the tag for the pythia run'),
                               ('--no_default', 'not run if pythia_card not present')])        
                
    def help_pgs(self):
        logger.info("syntax: pgs [RUN] [--run_options]")
        logger.info("-- run pgs on RUN (current one by default)")
        self.run_options_help([('-f','answer all question by default'),
                               ('--tag=', 'define the tag for the pgs run'),
                               ('--no_default', 'not run if pgs_card not present')]) 

    def help_delphes(self):
        logger.info("syntax: delphes [RUN] [--run_options]")
        logger.info("-- run delphes on RUN (current one by default)")
        self.run_options_help([('-f','answer all question by default'),
                               ('--tag=', 'define the tag for the delphes run'),
                               ('--no_default', 'not run if delphes_card not present')]) 
       
#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(object):
    """ The Series of check routine for the MadEventCmd"""

    def check_banner_run(self, args):
        """check the validity of line"""
        
        if len(args) == 0:
            self.help_banner_run()
            raise self.InvalidCmd('banner_run reauires at least one argument.')
        
        tag = [a[6:] for a in args if a.startswith('--tag=')]
        
        
        if os.path.isfile(args[0]):
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
                os.exec_cmd('remove %s all banner' % run_name)
            except:
                pass
            self.set_run_name(args[0], tag=None, level='parton', reload_card=True)
        elif type == 'banner':
            self.set_run_name(self.find_available_run_name(self.me_dir))
        elif type == 'run':
            if os.path.exists(pjoin(self.me_dir, 'Events', name)):
                run_name = self.find_available_run_name(self.me_dir)
                logger.info('Run %s is not empty so will use run_name: %s' % \
                                                               (name, run_name))
                self.set_run_name(run_name)
            else:
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
                raise self.InvalidCmd('No MadEvent path defined. Impossible to associate this name to a file')
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
        
        force = False
        run = None
        if '-f' in args:
            force = True
            args.remove('-f')
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
                    
        return force, run

    def check_calculate_decay_widths(self, args):
        """check that the argument for calculate_decay_widths are valid"""
        
        if self.ninitial != 1:
            raise self.InvalidCmd('Can only calculate decay widths for decay processes A > B C ...')
        force = False
        accuracy = 0.01
        run = None
        if '-f' in args:
            force = True
            args.remove('-f')
        if args and args[-1].startswith('--accuracy='):
            try:
                accuracy = float(args[-1].split('=')[-1])
            except:
                self.InvalidCmd('Argument error in calculate_decay_widths command')
            del args[-1]
        if len(args) > 1:
            self.help_calculate_decay_widths()
            raise self.InvalidCmd('Too many argument for calculate_decay_widths command: %s' % cmd)
                    
        return force, accuracy



    def check_multi_run(self, args):
        """check that the argument for survey are valid"""

        force = False
        run = None
        if '-f' in args:
            force = True
            args.remove('-f')
        
        if not len(args):
            self.help_multi_run()
            raise self.InvalidCmd("""multi_run command requires at least one argument for
            the number of times that it call generate_events command""")
            
        if args[-1].startswith('--laststep='):
            run = args[-1][6:]
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
        
        return force, run

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
                raise self.InvalidCmd('No run_name currently define. Impossible to run refine')

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
        
        if len(arg):
            self.help_combine_events()
            raise self.InvalidCmd('Too many argument for combine_events command')
        
        if not self.run_name:
            if not self.results.lastrun:
                raise self.InvalidCmd('No run_name currently define. Impossible to run combine')
            else:
                self.set_run_name(self.results.lastrun)
            self.set_run_name(arg[0])
        
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
     
        tag = [a for a in args if a.startswith('--tag=')]
        if tag: 
            args.remove(tag[0])
            tag = tag[0][6:]
     
        if len(args) == 0 and not self.run_name:
            if self.results.lastrun:
                args.insert(0, self.results.lastrun)
            else:
                raise self.InvalidCmd('No run name currently define. Please add this information.')             
        
        if len(args) == 1:
            if args[0] != self.run_name and\
             not os.path.exists(pjoin(self.me_dir,'Events',args[0], 'unweighted_events.lhe.gz')):
                raise self.InvalidCmd('No events file corresponding to %s run. '% args[0])
            self.set_run_name(args[0], tag, 'pythia')
        else:
            if tag:
                self.run_card['run_tag'] = tag
            self.set_run_name(self.run_name, tag, 'pythia')
            

        if  not os.path.exists(pjoin(self.me_dir,'Events',self.run_name,'unweighted_events.lhe.gz')):
            raise self.InvalidCmd('No events file corresponding to %s run. '% self.run_name)

        
        input_file = pjoin(self.me_dir,'Events',self.run_name, 'unweighted_events.lhe')
        output_file = pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')
        os.system('gunzip -c %s > %s' % (input_file, output_file))
            

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
        
        args.append(mode)
    
    def check_remove(self, args):
        """Check that the remove command is valid"""

        tmp_args = args[:]

        tag = [a[6:] for a in tmp_args if a.startswith('--tag=')]
        if tag:
            tag = tag[0]
            tmp_args.remove('--tag=%s' % tag)

        try:
            tmp_args.remove('-f')
        except:
            pass
        else:
            if args[0] == '-f':
                args.pop(0)
                args.append('-f')

        if len(tmp_args) == 0:
            self.help_remove()
            raise self.InvalidCmd('clean command require the name of the run to clean')
        elif len(tmp_args) == 1:
            return tmp_args[0], tag, ['all']
        else:
            for arg in tmp_args[1:]:
                if arg not in self._clean_mode and arg != '-f':
                    self.help_clean()
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

        force = False
        if '-f' in args:
            args.remove('-f')
            force = True
        
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
    
        if force:
            args.append('-f')
    
    
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
                args.insert(0, self.results.lastrun)
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
            if  not os.path.exists(pjoin(self.me_dir,'Events',self.run_name,'%s_pythia_events.hep.gz' % prev_tag)):
                raise self.InvalidCmd('No events file corresponding to %s run with tag %s. '% (self.run_name, prev_tag))
            else:
                input_file = pjoin(self.me_dir,'Events', self.run_name, '%s_pythia_events.hep.gz' % prev_tag)
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                self.launch_job('gunzip',stdout=open(output_file,'w'), 
                                 argument=['-c', input_file], mode=2)
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
            if  not os.path.exists(pjoin(self.me_dir,'Events',self.run_name, '%s_pythia_events.hep.gz' % prev_tag)):
                raise self.InvalidCmd('No events file corresponding to %s run with tag %s.:%s '\
                    % (self.run_name, prev_tag, 
                       pjoin(self.me_dir,'Events',self.run_name, '%s_pythia_events.hep.gz' % prev_tag)))
            else:
                input_file = pjoin(self.me_dir,'Events', self.run_name, '%s_pythia_events.hep.gz' % prev_tag)
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                self.launch_job('gunzip',stdout=open(output_file,'w'), 
                                 argument=['-c', input_file], mode=2)
        else:
            if tag:
                self.run_card['run_tag'] = tag
            self.set_run_name(self.run_name, tag, 'delphes')               

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

    def complete_calculate_decay_widths(self, text, line, begidx, endidx):
        """ Complete the calculate_decay_widths command"""
        
        if line.endswith('nb_core=') and not text:
            import multiprocessing
            max = multiprocessing.cpu_count()
            return [str(i) for i in range(2,max+1)]
        
        opts = self._run_options + self._calculate_decay_options
        return  self.list_completion(text, opts, line)

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


class MadEventAlreadyRunning(InvalidCmd):
    pass

#===============================================================================
# MadEventCmd
#===============================================================================
class MadEventCmd(CmdExtended, HelpToCmd, CompleteForCmd):
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
    _survey_options = {'points':('int', 1000,'Number of points for first iteration'),
                       'iterations':('int', 5, 'Number of iterations'),
                       'accuracy':('float', 0.1, 'Required accuracy'),
                       'gridpack':('str', '.false.', 'Gridpack generation')}
    # Variables to store object information
    true = ['T','.true.',True,'true', 1, '1']
    web = False
    prompt = 'MGME5>'
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
        
        # Define current MadEvent directory
        if me_dir is None and MADEVENT:
            me_dir = root_path
        
        self.me_dir = me_dir

        # usefull shortcut
        self.status = pjoin(self.me_dir, 'status')
        self.error =  pjoin(self.me_dir, 'error')
        self.dirbin = pjoin(self.me_dir, 'bin', 'internal')
        
        # Check that the directory is not currently running
        if os.path.exists(pjoin(me_dir,'RunWeb')): 
            message = '''Another instance of madevent is currently running.
            Please wait that all instance of madevent are closed. If this message
            is an error in itself, you can suppress the files: 
            %s.''' % pjoin(me_dir,'RunWeb')
            raise MadEventAlreadyRunning, message
        else:
            os.system('touch %s' % pjoin(me_dir,'RunWeb'))
            subprocess.Popen([pjoin(self.dirbin, 'gen_cardhtml-pl')], cwd=me_dir)
      
        self.to_store = []
        self.run_name = None
        self.run_tag = None
        self.banner = None

        # Get number of initial states
        try:
            nexternal = open(pjoin(me_dir,'Source','nexternal.inc')).read()
            found = re.search("PARAMETER\s*\(NINCOMING=(\d)\)", nexternal)
            self.ninitial = int(found.group(1))
        except:
            # in gridpack this file might be remove. In that case, let suppose
            # that ninitial is 2
            self.ninitial = 2
            
        # Load the configuration file
        self.set_configuration()
        self.nb_refine=0
        if self.web:
            os.system('touch %s' % pjoin(self.me_dir,'Online'))

        
        # load the current status of the directory
        if os.path.exists(pjoin(self.me_dir,'HTML','results.pkl')):
            self.results = save_load_object.load_from_file(pjoin(self.me_dir,'HTML','results.pkl'))
            self.results.resetall()
        else:
            model = self.find_model_name()
            process = self.process # define in find_model_name
            self.results = gen_crossxhtml.AllResults(model, process, self.me_dir)
        self.results.def_web_mode(self.web)

        
        self.configured = 0 # time for reading the card
        self._options = {} # for compatibility with extended_cmd
        
    ############################################################################    
    def split_arg(self, line, error=True):
        """split argument and remove run_options"""
        
        args = CmdExtended.split_arg(line)
        
        for arg in args[:]:
            if not arg.startswith('-'):
                continue
            elif arg == '-c':
                self.cluster_mode = 1
            elif arg == '-m':
                self.cluster_mode = 2
            elif arg == '-f':
                continue
            elif not arg.startswith('--'):
                if error:
                    raise self.InvalidCmd('%s argument cannot start with - symbol' % arg)
                else:
                    continue
            elif arg.startswith('--cluster'):
                self.cluster_mode = 1
            elif arg.startswith('--multicore'):
                self.cluster_mode = 2
            elif arg.startswith('--nb_core'):
                self.cluster_mode = 2
                self.nb_core = int(arg.split('=',1)[1])
            elif arg.startswith('--web'):
                self.web = True
                self.cluster_mode = 1
                self.results.def_web_mode(True)
                if not '-f' in args:
                    args.append('-f')
            else:
                continue
            args.remove(arg)
        
        if self.cluster_mode == 2 and not self.nb_core:
            import multiprocessing
            self.nb_core = multiprocessing.cpu_count()
            
        if self.cluster_mode == 1 and not hasattr(self, 'cluster'):
            cluster_name = self.options['cluster_type']
            self.cluster = cluster.from_name[cluster_name](self.options['cluster_queue'])
        return args
    
    ############################################################################            
    def check_output_type(self, path):
        """ Check that the output path is a valid madevent directory """
        
        bin_path = os.path.join(path,'bin')
        if os.path.isfile(os.path.join(bin_path,'generate_events')):
            return True
        else: 
            return False
            
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
                              'nb_core':None}
        
        if os.environ.has_key('MADGRAPH_BASE'):
            config_file = open(os.path.join(os.environ['MADGRAPH_BASE'],'mg5_configuration.txt'))
        elif not config_path:
            try:
                config_file = open(os.path.join(os.environ['HOME'],'.mg5','mg5_configuration.txt'))
            except:
                if self.me_dir:
                    config_file = open(os.path.relpath(
                          os.path.join(self.me_dir, 'Cards', 'me5_configuration.txt')))
                    main = self.me_dir
                elif not MADEVENT:
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
    def do_banner_run(self, line): 
        """Make a run from the banner file"""
        
        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_banner_run(args)
        if '-f' in args:
            force = True
        else:
            force = False     
             
        banner_mod.split_banner(args[0], self.me_dir, proc_card=False)
        
        # Remove previous cards
        for name in ['delphes_trigger.dat', 'delphes_card.dat',
                     'pgs_card.dat', 'pythia_card.dat']:
            try:
                os.remove(pjoin(self.me_dir, 'Cards', name))
            except:
                pass
        
        
        # Check if we want to modify the run
        if not force:
            ans = self.ask('Do you want to modify the Cards?', 'n', ['y','n'])
            if ans == 'n':
                force = True
        
        # Call Generate events
        self.exec_cmd('generate_events %s %s' % (self.run_name, force and '-f' or ''))
 
 
 
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
        else:
            super(MadEventCmd, self).do_display(line, output)
 
  
    ############################################################################
    def do_import(self, line):
        """Advanced commands: Import command files"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_import(args)
        # Remove previous imports, generations and outputs from history
        self.clean_history()
        
        # Execute the card
        self.import_command_file(args[1])  
  
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
    def update_status(self, status, level, makehtml=True, force=True, error=False):
        """ update the index status """
        

        if makehtml and not force:
            if hasattr(self, 'next_update') and time.time() < self.next_update:
                return
            else:
                self.next_update = time.time() + 3
        

        if isinstance(status, str):
            if '<br>' not  in status:
                logger.info(status)
        else:
            logger.info(' Idle: %s Running: %s Finish: %s' % status[:3])
        
        self.last_update = time
        self.results.update(status, level, makehtml=makehtml, error=error)
        
    ############################################################################      
    def do_generate_events(self, line):
        """ launch the full chain """
        
        args = self.split_arg(line)
        # Check argument's validity
        force, mode = self.check_generate_events(args)
        self.ask_run_configuration(mode, force)
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
                raise MadGraph5Error('''Survey return zero cross section. 
   Typical reasons are the following:
   1) A massive s-channel particle has a width set to zero.
   2) The pdf are zero for at least one of the initial state particles.
   3) The cuts are too strong.
   Please check/correct your param_card and/or your run_card.''')
            
            nb_event = self.run_card['nevents']
            self.exec_cmd('refine %s' % nb_event, postcmd=False)
            self.exec_cmd('refine %s' % nb_event, postcmd=False)
            self.exec_cmd('combine_events', postcmd=False)
            self.create_plot('parton')
            self.exec_cmd('store_events', postcmd=False)
            self.exec_cmd('pythia --no_default', postcmd=False, printcmd=False)
            # pythia launches pgs/delphes if needed
            self.store_result()
        
        self.print_results_in_shell(self.results.current)
            
            
    def print_results_in_shell(self, data):
        """Have a nice results prints in the shell,
        data should be of type: gen_crossxhtml.OneTagResults"""

        logger.info("  === Results Summary for run: %s tag: %s ===\n" % (data['run_name'],data['tag']))
        if self.ninitial == 1:
            logger.info("     Width :   %.4g +- %.4g GeV" % (data['cross'], data['error']))
        else:
            logger.info("     Cross-section :   %.4g +- %.4g pb" % (data['cross'], data['error']))
        logger.info("     Nb of events :  %s" % data['nb_event'] )
        if data['cross_pythia']:
            error = data.get_pythia_error(data['cross'], data['error'], 
                                        data['cross_pythia'], data['nb_event'])
            nb_event = 0
            if data['cross']:
                nb_event = int(0.5+(data['nb_event'] * data['cross_pythia'] /data['cross']))
            
            if self.ninitial == 1:
                logger.info("     Matched Width :   %.4g +- %.4g GeV" % (data['cross_pythia'], error))
            else:
                logger.info("     Matched Cross-section :   %.4g +- %.4g pb" % (data['cross'], error))            
            logger.info("     Nb of events after Matching :  %s" % nb_event)
        logger.info(" " )
        
            
    
    ############################################################################      
    def do_calculate_decay_widths(self, line):
        """ launch decay width calculation and automatic inclusion of
        calculated widths and BRs in the param_card."""
        
        args = self.split_arg(line)
        # Check argument's validity
        force, accuracy = self.check_calculate_decay_widths(args)
        self.ask_run_configuration('parton', force)
        if not args:
            # No run name assigned -> assigned one automaticaly 
            self.set_run_name(self.find_available_run_name(self.me_dir))
        else:
            self.set_run_name(args[0], reload_card=True)
            args.pop(0)
        
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

        # Open the param_card.dat and insert the calculated decays and BRs
        param_card_file = open(pjoin(self.me_dir, 'Cards', 'param_card.dat'))
        param_card = param_card_file.read().split('\n')
        param_card_file.close()

        decay_lines = []
        line_number = 0
        # Read and remove all decays from the param_card                     
        while line_number < len(param_card):
            line = param_card[line_number]
            if line.lower().startswith('decay'):
                # Read decay if particle in particle_dict 
                # DECAY  6   1.455100e+00                                    
                line = param_card.pop(line_number)
                line = line.split()
                particle = 0
                if int(line[1]) not in particle_dict:
                    try: # If formatting is wrong, don't want this particle
                        particle = int(line[1])
                        width = float(line[2])
                    except:
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
                    except:
                        line=param_card[line_number]
                        continue
                    try:
                        particle_dict[particle].append([decay_products, partial_width])
                    except KeyError:
                        particle_dict[particle] = [[decay_products, partial_width]]
                    line=param_card[line_number]
                if particle and particle not in particle_dict:
                    # No decays given, only total width       
                    particle_dict[particle] = [[[], width]]
            else: # Not decay                              
                line_number += 1
        # Clean out possible remaining comments at the end of the card
        while not param_card[-1] or param_card[-1].startswith('#'):
            param_card.pop(-1)

        # Append calculated and read decays to the param_card                                   
        param_card.append("#\n#*************************")
        param_card.append("#      Decay widths      *")
        param_card.append("#*************************")
        for key in sorted(particle_dict.keys()):
            width = sum([r for p,r in particle_dict[key]])
            param_card.append("#\n#      PDG        Width")
            param_card.append("DECAY  %i   %e" % (key, width))
            if not width:
                continue
            if particle_dict[key][0][0]:
                param_card.append("#  BR             NDA  ID1    ID2   ...")
                brs = [[val[1]/width, val[0]] for val in particle_dict[key] if val[1]]
                for val in sorted(brs, reverse=True):
                    param_card.append("   %e   %i    %s" % (val[0], len(val[1]),
                                           "  ".join([str(v) for v in val[1]])))
        output_name = pjoin(self.me_dir, 'Events', run_name, "param_card.dat")
        decay_table = open(output_name, 'w')
        decay_table.write("\n".join(param_card) + "\n")
        logger.info("Results written to %s" %  output_name)


            




    ############################################################################
    def do_multi_run(self, line):
        
        args = self.split_arg(line)
        # Check argument's validity
        force, mode = self.check_multi_run(args)
        self.ask_run_configuration(mode, force)
        main_name = self.run_name
        nb_run = args.pop(0)
        crossoversig = 0
        inv_sq_err = 0
        nb_event = 0
        for i in range(nb_run):
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

        
        self.run_name = main_name
        self.results.def_current(main_name)
        self.update_status("Merging LHE files", level='parton')
        try:
            os.mkdir(pjoin(self.me_dir,'Events', self.run_name))
        except:
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
        self.results.def_current(None)
            
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
        # treat random number
        self.update_random()
        self.save_random()
        self.update_status('Running Survey', level=None)
        if self.cluster_mode:
            logger.info('Creating Jobs')

        logger.info('Working on SubProcesses')
        self.total_jobs = 0
        subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                 'subproc.mg'))]
        for nb_proc,subdir in enumerate(subproc):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(pjoin(Pdir, match))
            
            #compile gensym
            misc.compile(['gensym'], cwd=Pdir)
            if not os.path.exists(pjoin(Pdir, 'gensym')):
                raise MadEventError, 'Error make gensym not successful'

            # Launch gensym
            p = subprocess.Popen(['./gensym'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, cwd=Pdir)
            sym_input = "%(points)d %(iterations)d %(accuracy)f %(gridpack)s\n" % self.opts
            (stdout, stderr) = p.communicate(sym_input)
            if not os.path.exists(pjoin(Pdir, 'ajob1')) or p.returncode:
                logger.critical(stdout)
                raise MadEventError, 'Error gensym run not successful'
            #
            os.system("chmod +x %s/ajob*" % Pdir)
        
            misc.compile(['madevent'], cwd=Pdir)
            
            alljobs = glob.glob(pjoin(Pdir,'ajob*'))
            self.total_jobs += len(alljobs)
            for i, job in enumerate(alljobs):
                job = os.path.basename(job)
                self.launch_job('./%s' % job, cwd=Pdir, remaining=(len(alljobs)-i-1), 
                                                    run_type='survey on %s (%s/%s)' % (subdir,nb_proc+1,len(subproc)))
                if os.path.exists(pjoin(self.me_dir,'error')):
                    self.monitor()
                    raise MadEventError, 'Error detected Stop running: %s' % \
                                         open(pjoin(self.me_dir,'error')).read()
        self.monitor(run_type='All jobs submitted for survey')
        self.update_status('End survey', 'parton', makehtml=False)

    ############################################################################      
    def do_refine(self, line):
        """Advanced commands: launch survey for the current process """
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

        if self.cluster_mode:
            logger.info('Creating Jobs')
        self.update_status('Refine results to %s' % precision, level=None)
        logger.info("Using random number seed offset = %s" % self.random)
        
        self.total_jobs = 0
        subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                 'subproc.mg'))]
        devnull = os.open(os.devnull, os.O_RDWR)
        for nb_proc,subdir in enumerate(subproc):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            bindir = pjoin(os.path.relpath(self.dirbin, Pdir))
                           
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(pjoin(Pdir, match))
            
            proc = subprocess.Popen([pjoin(bindir, 'gen_ximprove')],
                                    stdout=devnull,
                                    stdin=subprocess.PIPE,
                                    cwd=Pdir)
            proc.communicate('%s %s T\n' % (precision, max_process))

            if os.path.exists(pjoin(Pdir, 'ajob1')):
                misc.compile(['madevent'], cwd=Pdir)
                #
                os.system("chmod +x %s/ajob*" % Pdir)
                alljobs = glob.glob(pjoin(Pdir,'ajob*'))
                nb_tot = len(alljobs)            
                self.total_jobs += nb_tot
                for i, job in enumerate(alljobs):
                    job = os.path.basename(job)
                    self.launch_job('./%s' % job, cwd=Pdir, remaining=(nb_tot-i-1), 
                             run_type='Refine number %s on %s (%s/%s)' % (self.nb_refine, subdir, nb_proc+1, len(subproc)))
        self.monitor(run_type='All job submitted for refine number %s' % self.nb_refine)
        
        self.update_status("Combining runs", level='parton')
        try:
            os.remove(pjoin(Pdir, 'combine_runs.log'))
        except:
            pass
        
        bindir = pjoin(os.path.relpath(self.dirbin, pjoin(self.me_dir,'SubProcesses')))
        subprocess.call([pjoin(bindir, 'combine_runs')], 
                                          cwd=pjoin(self.me_dir,'SubProcesses'),
                                          stdout=devnull)
        
        self.update_status('finish refine', 'parton', makehtml=False)
        
    ############################################################################ 
    def do_combine_events(self, line):
        """Advanced commands: Launch combine events"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_combine_events(args)

        self.update_status('Combining Events', level='parton')
        try:
            os.remove(pjoin(self.me_dir,'SubProcesses', 'combine.log'))
        except:
            pass
        if self.cluster_mode == 1:
            out = self.cluster.launch_and_wait('../bin/internal/run_combine', 
                                        cwd=pjoin(self.me_dir,'SubProcesses'),
                                        stdout=pjoin(self.me_dir,'SubProcesses', 'combine.log'))
        else:
            out = subprocess.call(['../bin/internal/run_combine'],
                         cwd=pjoin(self.me_dir,'SubProcesses'), 
                         stdout=open(pjoin(self.me_dir,'SubProcesses','combine.log'),'w'))
            
        output = open(pjoin(self.me_dir,'SubProcesses','combine.log')).read()
        # Store the number of unweighted events for the results object
        pat = re.compile(r'''\s*Unweighting selected\s*(\d+)\s*events''',re.MULTILINE)
        if output:
            nb_event = pat.search(output).groups()[0]
            self.results.add_detail('nb_event', nb_event)
        
        # Define The Banner
        tag = self.run_card['run_tag']
        self.banner.load_basic(self.me_dir)
        self.banner.change_seed(self.random)
        if not os.path.exists(pjoin(self.me_dir, 'Events', self.run_name)):
            os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))
        self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, 
                                     '%s_%s_banner.txt' % (self.run_name, tag)))
        self.banner.add(pjoin(self.me_dir, 'Cards', 'param_card.dat'))
        self.banner.add(pjoin(self.me_dir, 'Cards', 'run_card.dat'))
        
        
        subprocess.call(['%s/put_banner' % self.dirbin, 'events.lhe'],
                            cwd=pjoin(self.me_dir, 'Events'))
        subprocess.call(['%s/put_banner'% self.dirbin, 'unweighted_events.lhe'],
                            cwd=pjoin(self.me_dir, 'Events'))
        
        #if os.path.exists(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')):
        #    subprocess.call(['%s/extract_banner-pl' % self.dirbin, 
        #                     'unweighted_events.lhe', 'banner.txt'],
        #                    cwd=pjoin(self.me_dir, 'Events'))
        
        eradir = self.options['exrootanalysis_path']
        madir = self.options['madanalysis_path']
        td = self.options['td_path']
        if eradir and misc.is_executable(pjoin(eradir,'ExRootLHEFConverter'))  and\
           os.path.exists(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')):
                if not os.path.exists(pjoin(self.me_dir, 'Events', self.run_name)):
                    os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))
                self.create_root_file(output='%s/unweighted_events.root' % self.run_name)
        
    ############################################################################ 
    def do_store_events(self, line):
        """Advanced commands: Launch store events"""

        args = self.split_arg(line)
        # Check argument's validity
        self.check_combine_events(args)
        self.update_status('Storing parton level results', level='parton')

        run = self.run_name
        tag = self.run_card['run_tag']
        devnull = os.open(os.devnull, os.O_RDWR)

        if not os.path.exists(pjoin(self.me_dir, 'Events', run)):
            os.mkdir(pjoin(self.me_dir, 'Events', run))
        if not os.path.exists(pjoin(self.me_dir, 'HTML', run)):
            os.mkdir(pjoin(self.me_dir, 'HTML', run))    
        
        # 2) Treat the files present in the P directory
        for P_path in SubProcesses.get_subP(self.me_dir):
            G_dir = [G for G in os.listdir(P_path) if G.startswith('G') and 
                                                os.path.isdir(pjoin(P_path,G))]
            for G in G_dir:
                G_path = pjoin(P_path,G)
                # Remove events file (if present)
                if os.path.exists(pjoin(G_path, 'events.lhe')):
                    os.remove(pjoin(G_path, 'events.lhe'))
                # Remove results.dat (but not for gridpack)
                if self.run_card['gridpack'] not in self.true:
                    if os.path.exists(pjoin(G_path, 'results.dat')):
                        os.remove(pjoin(G_path, 'results.dat'))
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
                        subprocess.call(['gzip', output], stdout=devnull, 
                                        stderr=devnull, cwd=G_path)
        
        # 2) restore links in local this is require due to chrome over-security
        if not self.web: 
            results = pjoin(self.me_dir, 'HTML', run, 'results.html')
            text = open(results).read()
            text = text.replace('''if ( ! UrlExists(alt)){
         obj.href = alt;''','''if ( ! UrlExists(alt)){
         obj.href = url;''')
            open(results, 'w').write(text)
            
        # 3) Update the index.html
        subprocess.call(['%s/gen_cardhtml-pl' % self.dirbin],
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
                subprocess.call(['gzip', output], stdout=devnull, stderr=devnull, 
                                                                     cwd=O_path)
                
        self.update_status('End Parton', level='parton', makehtml=False)

    ############################################################################ 
    def do_create_gridpack(self, line):
        """Advanced commands: Create gridpack from present run"""

        self.update_status('Creating gridpack', level='parton')
        args = self.split_arg(line)
        self.check_combine_events(args)
        os.system("sed -i.bak \"s/ *.false.*=.*GridRun/  .true.  =  GridRun/g\" %s/Cards/grid_card.dat" \
                  % self.me_dir)
        subprocess.call(['./bin/internal/restore_data', self.run_name],
                        cwd=self.me_dir)
        subprocess.call(['./bin/internal/store4grid',
                         self.run_name, self.run_tag],
                        cwd=self.me_dir)
        subprocess.call(['./bin/internal/clean'], cwd=self.me_dir)
        misc.compile(['gridpack.tar.gz'], cwd=self.me_dir)
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
        if '-f' in args:
            force = True
            args.remove('-f')
        else:
            force = False
        if '--no_default' in args:
            if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pythia_card.dat')):
                return
            force = True
            no_default = True
            args.remove('--no_default')
        else:
            no_default = False
                                    
        self.check_pythia(args)        
        # the args are modify and the last arg is always the mode
        
        self.ask_pythia_run_configuration(args[-1], force)

        # Update the banner with the pythia card
        if not self.banner:
            self.banner = banner_mod.recover_banner(self.results, 'pythia')
                     
        # initialize / remove lhapdf mode        
        self.configure_directory()

        #if not force:
        #    if os.path.exists(pjoin(self.me_dir, 'Events', self.run_name, '%s_pythia.log' % tag)):
        #        question = 'Previous run of pythia detected. Do you want to remove it?'
        #        ans = self.ask(question, 'y', choices=['y','n'], timeout = 20)
        #        if ans == 'n':
        #            return
         
        #self.exec_cmd('remove %s pythia -f' % self.run_name)
        
        pythia_src = pjoin(self.options['pythia-pgs_path'],'src')
        
        self.update_status('Running Pythia', 'pythia')
        try:
            os.remove(pjoin(self.me_dir,'Events','pythia.done'))
        except:
            pass
        
        ## LAUNCHING PYTHIA
        tag = self.run_tag
        
        if self.cluster_mode == 1:
            pythia_log = pjoin(self.me_dir, 'Events', self.run_name , '%s_pythia.log' % tag)
            self.cluster.launch_and_wait('../bin/internal/run_pythia', 
                        argument= [pythia_src], stdout= pythia_log,
                        stderr=subprocess.STDOUT,
                        cwd=pjoin(self.me_dir,'Events'))
        else:
            pythia_log = open(pjoin(self.me_dir, 'Events',  self.run_name , '%s_pythia.log' % tag), 'w')
            subprocess.call(['../bin/internal/run_pythia', pythia_src],
                           stdout=pythia_log,
                           stderr=subprocess.STDOUT,
                           cwd=pjoin(self.me_dir,'Events'))

        if not os.path.exists(pjoin(self.me_dir,'Events','pythia.done')):
            if self.cluster_mode == 1:
                logger.warning('Fail to produce pythia output. More info in \n     %s' % pythia_log)
            else:
                logger.warning('Fail to produce pythia output. More info in \n     %s' % pythia_log.name)
            return
        else:
            os.remove(pjoin(self.me_dir,'Events','pythia.done'))
        
        self.to_store.append('pythia')
        
        # Find the matched cross-section
        if int(self.run_card['ickkw']):    
            pythia_log = open(pjoin(self.me_dir,'Events', self.run_name, '%s_pythia.log' % tag))
            cs_info = misc.get_last_line(pythia_log)
            # line should be of type: Cross section (pb):    1840.20000000006 
            cs_info = cs_info.split(':')[1]
            self.results.add_detail('cross_pythia', cs_info)
        
        pydir = pjoin(self.options['pythia-pgs_path'], 'src')
        eradir = self.options['exrootanalysis_path']
        madir = self.options['madanalysis_path']
        td = self.options['td_path']
        
        
        self.banner.add(pjoin(self.me_dir, 'Cards','pythia_card.dat'))
        banner_path = pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, tag))
        self.banner.write(banner_path)
        
        # Creating LHE file
        if misc.is_executable(pjoin(pydir, 'hep2lhe')):
            self.update_status('Creating Pythia LHE File', level='pythia')
            # Write the banner to the LHE file
            out = open(pjoin(self.me_dir,'Events','pythia_events.lhe'), 'w')
            #out.writelines('<LesHouchesEvents version=\"1.0\">\n')    
            out.writelines('<!--\n')
            out.writelines('# Warning! Never use this file for detector studies!\n')
            out.writelines('-->\n<!--\n')
            out.writelines(open(banner_path).read())
            out.writelines('\n-->\n')
            out.close()
            
            if self.cluster_mode == 1:
                self.cluster.launch_and_wait(self.dirbin+'/run_hep2lhe', 
                                         argument= [pydir],
                                        cwd=pjoin(self.me_dir,'Events'))
            else:
                subprocess.call([self.dirbin+'/run_hep2lhe', pydir],
                             cwd=pjoin(self.me_dir,'Events'))
                
            # Creating ROOT file
            if eradir and misc.is_executable(pjoin(eradir, 'ExRootLHEFConverter')):
                self.update_status('Creating Pythia LHE Root File', level='pythia')
                subprocess.call([eradir+'/ExRootLHEFConverter', 
                             'pythia_events.lhe', 
                             pjoin(self.run_name, '%s_pythia_lhe_events.root' % tag)],
                            cwd=pjoin(self.me_dir,'Events')) 
            

        if int(self.run_card['ickkw']):
            self.update_status('Create matching plots for Pythia', level='pythia')
            subprocess.call([self.dirbin+'/create_matching_plots.sh', self.run_name, tag],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            cwd=pjoin(self.me_dir,'Events'))
            #Clean output
            subprocess.call(['gzip','-f','events.tree'], 
                                                cwd=pjoin(self.me_dir,'Events'))          
            files.mv(pjoin(self.me_dir,'Events','events.tree.gz'), 
                     pjoin(self.me_dir,'Events',self.run_name, tag + '_pythia_events.tree.gz'))
            subprocess.call(['gzip','-f','beforeveto.tree'], 
                                                cwd=pjoin(self.me_dir,'Events'))
            files.mv(pjoin(self.me_dir,'Events','beforeveto.tree.gz'), 
                     pjoin(self.me_dir,'Events',self.run_name, tag+'_pythia_beforeveto.tree.gz'))
            files.mv(pjoin(self.me_dir,'Events','xsecs.tree'), 
                     pjoin(self.me_dir,'Events',self.run_name, tag+'_pythia_xsecs.tree'))            
             

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
    
    def get_available_tag(self):
        """create automatically a tag"""
        
        used_tags = [r['tag'] for r in self.results[self.run_name]]
        i=0
        while 1:
            i+=1
            if 'tag_%s' %i not in used_tags:
                return 'tag_%s' % i
   
    
    
    ################################################################################
    def do_remove(self, line):
        """Remove one/all run or only part of it"""

        args = self.split_arg(line)
        run, tag, mode = self.check_remove(args)
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
        except:
            pass # Just ensure that html never makes crash this function


        # Found the file to suppress
        
        to_suppress = glob.glob(pjoin(self.me_dir, 'Events', run, '*'))
        to_suppress += glob.glob(pjoin(self.me_dir, 'HTML', run, '*'))
        # forbid the banner to be removed
        to_suppress = [os.path.basename(f) for f in to_suppress if 'banner' not in f]
        if tag:
            to_suppress = [f for f in to_suppress if tag in f]
            if 'parton' in mode or 'all' in mode:
                try:
                    if self.results[run][0]['tag'] != tag:
                        raise Exception, 'dummy'
                except:
                    pass
                else:
                    nb_rm = len(to_suppress)
                    if os.path.exists(pjoin(self.me_dir, 'Events', run, 'events.lhe.gz')):
                        to_suppress.append('events.lhe.gz')
                    if os.path.exists(pjoin(self.me_dir, 'Events', run, 'unweighted_events.lhe.gz')):
                        to_suppress.append('unweighted_events.lhe.gz')
                    if nb_rm != len(to_suppress):
                        logger.warning('Be carefull that partonic information are on the point to be removed.')
        if 'all' in mode:
            pass # suppress everything
        else:
            if 'pythia' not in mode:
                to_suppress = [f for f in to_suppress if 'pythia' not in f]
            if 'pgs' not in mode:
                to_suppress = [f for f in to_suppress if 'pgs' not in f]
            if 'delphes' not in mode:
                to_suppress = [f for f in to_suppress if 'delphes' not in f]
            if 'parton' not in mode:
                to_suppress = [f for f in to_suppress if 'delphes' in f 
                                                      or 'pgs' in f 
                                                      or 'pythia' in f]
        if '-f' not in args and len(to_suppress):
            question = 'Do you want to suppress the following files?\n     %s' % \
                               '\n    '.join(to_suppress)
            ans = self.ask(question, 'y', choices=['y','n'])
        else:
            ans = 'y'
        
        if ans == 'y':
            for file2rm in to_suppress:
                if os.path.exists(pjoin(self.me_dir, 'Events', run, file2rm)):
                    try:
                        os.remove(pjoin(self.me_dir, 'Events', run, file2rm))
                    except:
                        shutil.rmtree(pjoin(self.me_dir, 'Events', run, file2rm))
                else:
                    try:
                        os.remove(pjoin(self.me_dir, 'HTML', run, file2rm))
                    except:
                        shutil.rmtree(pjoin(self.me_dir, 'HTML', run, file2rm))



        # Remove file in SubProcess directory
        if 'all' in mode or 'channel' in mode:
            try:
                if self.results[run][0]['tag'] != tag:
                    raise Exception, 'dummy'
            except:
                pass
            else:
                to_suppress = glob.glob(pjoin(self.me_dir, 'SubProcesses', '%s*' % run))
                to_suppress += glob.glob(pjoin(self.me_dir, 'SubProcesses', '*','%s*' % run))
                to_suppress += glob.glob(pjoin(self.me_dir, 'SubProcesses', '*','*','%s*' % run))

                if '-f' in args or len(to_suppress) == 0:
                    ans = 'y'
                else:
                    question = 'Do you want to suppress the following files?\n     %s' % \
                               '\n    '.join(to_suppress)
                    ans = self.ask(question, 'y', choices=['y','n'])

                if ans == 'y':
                    for file2rm in to_suppress:
                        os.remove(file2rm)
                        
        if 'banner' in mode:
            to_suppress = glob.glob(pjoin(self.me_dir, 'Events', run, '*'))
            if tag:
                # remove banner
                try:
                    os.remove(pjoin(self.me_dir, 'Events',run,'%s_%s_banner.txt' % (run,tag)))
                except:
                    logger.warning('fail to remove the banner')
                # remove the run from the html output
                if run in self.results:
                    self.results.delete_run(run, tag)
                    return
            elif any(['banner' not in os.path.basename(p) for p in to_suppress]):
                if to_suppress:
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



    ################################################################################
    def do_plot(self, line):
        """Create the plot for a given run"""

        # Since in principle, all plot are already done automaticaly
        self.store_result()
        args = self.split_arg(line)
        # Check argument's validity
        self.check_plot(args)
        logger.info('plot for run %s' % self.run_name)
        
        self.ask_edit_cards([], args)
                
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

                
    
    def store_result(self):
        """ tar the pythia results. This is done when we are quite sure that 
        the pythia output will not be use anymore """


        if not self.run_name:
            return
        
        self.results.save()
        
        if not self.to_store:
            return 
        
        tag = self.run_card['run_tag']
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
            
        
    ############################################################################      
    def do_pgs(self, line):
        """launch pgs"""
        
        args = self.split_arg(line)
        # Check argument's validity
        if '-f' in args:
            force = True
            args.remove('-f')
        else:
            force = False
        if '--no_default' in args:
            no_default = True
            args.remove('--no_default')
        else:
            no_default = False

        # Check all arguments
        # This might launch a gunzip in another thread. After the question
        # This thread need to be wait for completion. (This allow to have the 
        # question right away and have the computer working in the same time)
        self.check_pgs(args) 

        # Check that the pgs_card exists. If not copy the default 
        if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pgs_card.dat')):
            if no_default:
                logger.info('No pgs_card detected, so not run pgs')
                return 
            
            files.cp(pjoin(self.me_dir, 'Cards', 'pgs_card_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'pgs_card.dat'))
            logger.info('No pgs card found. Take the default one.')        
        
        if not (no_default or force):
            self.ask_edit_cards(['pgs'], args)
            
        self.update_status('prepare PGS run', level=None)  
        # Wait that the gunzip of the files is finished (if any)
        if hasattr(self, 'control_thread') and self.control_thread[0]:
            self.monitor(mode=2,html=False)

        pgsdir = pjoin(self.options['pythia-pgs_path'], 'src')
        eradir = self.options['exrootanalysis_path']
        madir = self.options['madanalysis_path']
        td = self.options['td_path']
        
        # Compile pgs if not there       
        if not misc.is_executable(pjoin(pgsdir, 'pgs')):
            logger.info('No PGS executable -- running make')
            misc.compile(cwd=pgsdir)
        

            

            
        self.update_status('Running PGS', level='pgs')
        
        tag = self.run_tag
        # Update the banner with the pgs card        
        self.banner.add(pjoin(self.me_dir, 'Cards','pgs_card.dat'))
        banner_path = pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, self.run_tag))
        self.banner.write(banner_path)            

        ########################################################################
        # now pass the event to a detector simulator and reconstruct objects
        ########################################################################
        
        # Prepare the output file with the banner
        ff = open(pjoin(self.me_dir, 'Events', 'pgs_events.lhco'), 'w')
        text = open(banner_path).read()
        text = '#%s' % text.replace('\n','\n#')
        dico = self.results[self.run_name].get_current_info()
        text +='\n##  Integrated weight (pb)  : %.4g' % dico['cross']
        text +='\n##  Number of Event         : %s\n' % dico['nb_event']
        ff.writelines(text)
        ff.close()

        try: 
            os.remove(pjoin(self.me_dir, 'Events', 'pgs.done'))
        except:
            pass
        if self.cluster_mode == 1:
            pgs_log = pjoin(self.me_dir, 'Events', self.run_name, "%s_pgs.log" % tag)
            self.cluster.launch_and_wait('../bin/internal/run_pgs', 
                            argument=[pgsdir], cwd=pjoin(self.me_dir,'Events'),
                            stdout=pgs_log, stderr=subprocess.STDOUT)
        else:
            pgs_log = open(pjoin(self.me_dir, 'Events', self.run_name,"%s_pgs.log" % tag),'w')
            subprocess.call([self.dirbin+'/run_pgs', pgsdir], stdout= pgs_log,
                                               stderr=subprocess.STDOUT,
                                               cwd=pjoin(self.me_dir, 'Events')) 
        
        if not os.path.exists(pjoin(self.me_dir, 'Events', 'pgs.done')):
            logger.error('Fail to create LHCO events')
            return 
        else:
            os.remove(pjoin(self.me_dir, 'Events', 'pgs.done'))
            
        if os.path.getsize(banner_path) == os.path.getsize(pjoin(self.me_dir, 'Events','pgs_events.lhco')):
            subprocess.call(['cat pgs_uncleaned_events.lhco >>  pgs_events.lhco'], 
                            cwd=pjoin(self.me_dir, 'Events'))
            os.remove(pjoin(self.me_dir, 'Events', 'pgs_uncleaned_events.lhco '))

        # Creating Root file
        if eradir and misc.is_executable(pjoin(eradir, 'ExRootLHCOlympicsConverter')):
            self.update_status('Creating PGS Root File', level='pgs')
            subprocess.call([eradir+'/ExRootLHCOlympicsConverter', 
                             'pgs_events.lhco',pjoin('%s/%s_pgs_events.root' % (self.run_name, tag))],
                            cwd=pjoin(self.me_dir, 'Events')) 
        
        if os.path.exists(pjoin(self.me_dir, 'Events', 'pgs_events.lhco')):
            # Creating plots
            self.create_plot('PGS')
            files.mv(pjoin(self.me_dir, 'Events', 'pgs_events.lhco'), 
                    pjoin(self.me_dir, 'Events', self.run_name, '%s_pgs_events.lhco' % tag))
            subprocess.call(['gzip','-f', pjoin(self.me_dir, 'Events', 
                                                self.run_name, '%s_pgs_events.lhco' % tag)])


        
        self.update_status('finish', level='pgs', makehtml=False)

    ############################################################################
    def do_delphes(self, line):
        """ run delphes and make associate root file/plot """
 
        args = self.split_arg(line)
        # Check argument's validity
        if '-f' in args:
            force = True
            args.remove('-f')
        else:
            force = False
        if '--no_default' in args:
            no_default = True
            args.remove('--no_default')
        else:
            no_default = False
        self.check_delphes(args) 
        self.update_status('prepare delphes run', level=None)
                
        # Check that the delphes_card exists. If not copy the default and
        # ask for edition of the card.
        if not os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_card.dat')):
            if no_default:
                logger.info('No delphes_card detected, so not run Delphes')
                return
            
            files.cp(pjoin(self.me_dir, 'Cards', 'delphes_card_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'delphes_card.dat'))
            logger.info('No delphes card found. Take the default one.')
        if not os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_trigger.dat')):    
            files.cp(pjoin(self.me_dir, 'Cards', 'delphes_trigger_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'delphes_trigger.dat'))
        if not (no_default or force):
            self.ask_edit_cards(['delphes', 'trigger'], args)
            
        self.update_status('Running Delphes', level=None)  
        # Wait that the gunzip of the files is finished (if any)
        if hasattr(self, 'control_thread') and self.control_thread[0]:
            self.monitor(mode=2,html=False)        


 
        delphes_dir = self.options['delphes_path']
        tag = self.run_tag
        self.banner.add(pjoin(self.me_dir, 'Cards','delphes_card.dat'))
        self.banner.add(pjoin(self.me_dir, 'Cards','delphes_trigger.dat'))
        self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, tag)))
        
        cross = self.results[self.run_name].get_current_info()['cross']
                    
        if self.cluster_mode == 1:
            delphes_log = pjoin(self.me_dir, 'Events', self.run_name, "%s_delphes.log" % tag)
            self.cluster.launch_and_wait('../bin/internal/run_delphes', 
                        argument= [delphes_dir, self.run_name, tag, str(cross)],
                        stdout=delphes_log, stderr=subprocess.STDOUT,
                        cwd=pjoin(self.me_dir,'Events'))
        else:
            delphes_log = open(pjoin(self.me_dir, 'Events', self.run_name, "%s_delphes.log" % tag),'w')
            subprocess.call(['../bin/internal/run_delphes', delphes_dir, 
                                self.run_name, tag, str(cross)],
                                stdout= delphes_log, stderr=subprocess.STDOUT,
                                cwd=pjoin(self.me_dir,'Events'))
                
        if not os.path.exists(pjoin(self.me_dir, 'Events', 
                                self.run_name, '%s_delphes_events.lhco' % tag)):
            logger.error('Fail to create LHCO events from DELPHES')
            return 
        
        if os.path.exists(pjoin(self.me_dir,'Events','delphes.root')):
            source = pjoin(self.me_dir,'Events','delphes.root')
            target = pjoin(self.me_dir,'Events', self.run_name, "%s_delphes_events.root" % tag)
            files.mv(source, target)
            
        #eradir = self.options['exrootanalysis_path']
        madir = self.options['madanalysis_path']
        td = self.options['td_path']

        # Creating plots
        self.create_plot('Delphes')

        if os.path.exists(pjoin(self.me_dir, 'Events', self.run_name,  '%s_delphes_events.lhco' % tag)):
            subprocess.call(['gzip','-f', pjoin(self.me_dir, 'Events', self.run_name, '%s_delphes_events.lhco' % tag)])


        
        self.update_status('delphes done', level='delphes', makehtml=False)   

    def launch_job(self,exe, cwd=None, stdout=None, argument = [], remaining=0, 
                    run_type='', mode=None, **opt):
        """ """
        argument = [str(arg) for arg in argument]
        if mode is None:
            mode = self.cluster_mode

        def launch_in_thread(exe, argument, cwd, stdout, control_thread):
            """ way to launch for multicore"""

            start = time.time()
            if (cwd and os.path.exists(pjoin(cwd, exe))) or os.path.exists(exe):
                exe = './' + exe
            subprocess.call([exe] + argument, cwd=cwd, stdout=stdout,
                        stderr=subprocess.STDOUT, **opt)
            #logger.info('%s run in %f s' % (exe, time.time() -start))
            
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

        
        
        if mode == 0:
            self.update_status((remaining, 1, 
                                self.total_jobs - remaining -1, run_type), level=None, force=False)
            start = time.time()
            #os.system('cd %s; ./%s' % (cwd,exe))
            status = subprocess.call(['./'+exe] + argument, cwd=cwd, 
                                                           stdout=stdout, **opt)
            logger.info('%s run in %f s' % (exe, time.time() -start))
            if status:
                raise MadGraph5Error, '%s didn\'t stop properly. Stop all computation' % exe


        elif mode == 1:
            self.cluster.submit(exe, stdout=stdout, cwd=cwd)

        elif mode == 2:
            import thread
            if not hasattr(self, 'control_thread'):
                self.control_thread = [0] # [used_thread]
                self.control_thread.append(thread.allocate_lock()) # The lock
                self.control_thread.append(False) # True if all thread submit 
                                                  #-> waiting mode

            if self.control_thread[2]:
                self.update_status((remaining + 1, self.control_thread[0], 
                                self.total_jobs - remaining - self.control_thread[0] - 1, run_type), 
                                   level=None, force=False)
                self.control_thread[1].acquire()
                self.control_thread[0] += 1 # upate the number of running thread
                thread.start_new_thread(launch_in_thread,(exe, argument, cwd, stdout, self.control_thread))
            elif self.control_thread[0] <  self.nb_core -1:
                self.control_thread[0] += 1 # upate the number of running thread
                thread.start_new_thread(launch_in_thread,(exe, argument, cwd, stdout, self.control_thread))
            elif self.control_thread[0] ==  self.nb_core -1:
                self.control_thread[0] += 1 # upate the number of running thread
                thread.start_new_thread(launch_in_thread,(exe, argument, cwd, stdout, self.control_thread))
                self.control_thread[2] = True
                self.control_thread[1].acquire() # Lock the next submission
                                                 # Up to a release
            
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
    def monitor(self, run_type='monitor', mode=None, html=True):
        """ monitor the progress of running job """
        
        if mode is None:
            mode = self.cluster_mode
        
        if mode == 1:
            if html:
                update_status = lambda idle, run, finish: \
                    self.update_status((idle, run, finish, run_type), level=None)
            else:
                update_status = lambda idle, run, finish: None
            self.cluster.wait(self.me_dir, update_status)            

        if mode == 2:
            # Wait that all thread finish
            if not self.control_thread[2]:
#                time.sleep(1)
                nb = self.control_thread[0]
                while self.control_thread[0]:
                    time.sleep(5)
                    if nb != self.control_thread[0] and html:
                        self.update_status((0, self.control_thread[0], 
                                           self.total_jobs - self.control_thread[0], run_type), 
                                           level=None, force=False)
                        nb = self.control_thread[0]
                try:
                    del self.next_update
                except:
                    pass
            else:    
                for i in range(0,self.nb_core):
                    if html:
                        self.update_status((0, self.control_thread[0], 
                                           self.total_jobs - self.control_thread[0], run_type), 
                                           level=None, force=False)
                    self.control_thread[1].acquire()
                self.control_thread[2] = False
                self.control_thread[1].release()
                try:
                    del self.next_update
                except:
                    pass
        if not html:
            return

        #######################################################################
        cross, error = sum_html.make_all_html_results(self)
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)   
        
        
        
        
    @staticmethod
    def find_available_run_name(me_dir):
        """ find a valid run_name for the current job """
        
        name = 'run_%02d'
        data = [int(s[4:6]) for s in os.listdir(pjoin(me_dir,'Events')) if
                        s.startswith('run_') and len(s)>5 and s[4:6].isdigit()]
        return name % (max(data+[0])+1) 

    ############################################################################   
    def configure_directory(self):
        """ All action require before any type of run """   


        # Basic check
        assert os.path.exists(pjoin(self.me_dir,'SubProcesses'))
        
        #see when the last file was modified
        time_mod = max([os.path.getctime(pjoin(self.me_dir,'Cards','run_card.dat')),
                        os.path.getctime(pjoin(self.me_dir,'Cards','param_card.dat'))])
        
        if self.configured > time_mod:
            return
        else:
            self.configured = time.time()
        self.update_status('compile directory', level=None)
        if self.options['automatic_html_opening']:
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
        elif 'lhapdf' in os.environ.keys():
            del os.environ['lhapdf']
        
        # Compile
        out = subprocess.call([pjoin(self.dirbin, 'compile_Source')],
                              cwd = self.me_dir)
        if out:
            raise MadEventError, 'Impossible to compile'
        
        # set random number
        if self.run_card['iseed'] != '0':
            self.random = int(self.run_card['iseed'])
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
    @staticmethod
    def check_dir(path, default=''):
        """check if the directory exists. if so return the path otherwise the 
        default"""
         
        if os.path.isdir(path):
            return path
        else:
            return default
        
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
                self.run_card = self.read_run_card(run_card)

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
        self.run_card = self.read_run_card(run_card)

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
        if nb_event > 100000:
            logger.warning("Attempting to generate more than 100K events")
            logger.warning("Limiting number to 100K. Use multi_run for larger statistics.")
            path = pjoin(self.me_dir, 'Cards', 'run_card.dat')
            os.system(r"""perl -p -i.bak -e "s/\d+\s*=\s*nevents/100000 = nevents/" %s""" \
                                                                         % path)
            self.run_card['nevents'] = 100000

        return

  
    ############################################################################    
    def update_random(self):
        """ change random number"""
        
        self.random += 5
        assert self.random < 31328*30081 # cann't use too big random number

    ############################################################################
    def save_random(self):
        """save random number in appropirate file"""
        
        fsock = open(pjoin(self.me_dir, 'SubProcesses','randinit'),'w')
        fsock.writelines('r=%s\n' % self.random)

    def do_quit(self, line):
        """ """
  
        try:
            os.remove(pjoin(self.me_dir,'RunWeb'))
        except:
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
        try:
            subprocess.call(['./bin/internal/gen_cardhtml-pl'], cwd=self.me_dir,
                        stdout=devnull, stderr=devnull)
        except:
            pass

        return super(MadEventCmd, self).do_quit(line)
    
    # Aliases
    do_EOF = do_quit
    do_exit = do_quit
        
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
                self.update_status('GENERATE SUDAKOF GRID', level='parton')
                
                for i in range(-2,6):
                    self.launch_job('%s/gensudgrid ' % self.dirbin, 
                                    arguments = [i],
                                    cwd=self.me_dir, 
                                    stdout=open(pjoin(self.me_dir, 'gensudgrid%s.log' % s,'w')))
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
        subprocess.call(['%s/ExRootLHEFConverter' % eradir, 
                             input, output],
                            cwd=pjoin(self.me_dir, 'Events'))
        
    ############################################################################
    def create_plot(self, mode='parton', event_path=None, output=None):
        """create the plot""" 

        madir = self.options['madanalysis_path']
        td = self.options['td_path']
        if not madir or not  td or \
            not os.path.exists(pjoin(self.me_dir, 'Cards', 'plot_card.dat')):
            return False

        tag = self.run_card['run_tag']    
        if not event_path:
            if mode == 'parton':
                event_path = pjoin(self.me_dir, 'Events','unweighted_events.lhe')
                output = pjoin(self.me_dir, 'HTML',self.run_name, 'plots_parton.html')
            elif mode == 'Pythia':
                event_path = pjoin(self.me_dir, 'Events','pythia_events.lhe')
                output = pjoin(self.me_dir, 'HTML',self.run_name, 
                              'plots_pythia_%s.html' % tag)                                   
            elif mode == 'PGS':
                event_path = pjoin(self.me_dir, 'Events', 'pgs_events.lhco')
                output = pjoin(self.me_dir, 'HTML',self.run_name, 
                              'plots_pgs_%s.html' % tag)  
            elif mode == 'Delphes':
                event_path = pjoin(self.me_dir, 'Events', self.run_name,'%s_delphes_events.lhco' % tag)
                output = pjoin(self.me_dir, 'HTML',self.run_name, 
                              'plots_delphes_%s.html' % tag) 
            else:
                raise self.InvalidCmd, 'Invalid mode %s' % mode

            
            
        if not os.path.exists(event_path):
            raise self.InvalidCmd, 'Events file %s does not exits' % event_path
        
        self.update_status('Creating Plots for %s level' % mode, level = mode.lower())
               
        plot_dir = pjoin(self.me_dir, 'HTML', self.run_name,'plots_%s_%s' % (mode.lower(),tag))
                
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir) 
        
        files.ln(pjoin(self.me_dir, 'Cards','plot_card.dat'), plot_dir, 'ma_card.dat')
        proc = subprocess.Popen([os.path.join(madir, 'plot_events')],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'w'),
                            stderr = subprocess.STDOUT,
                            stdin=subprocess.PIPE,
                            cwd=plot_dir)
        proc.communicate('%s\n' % event_path)
        del proc
        #proc.wait()
        subprocess.call(['%s/plot' % self.dirbin, madir, td],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'a'),
                            stderr = subprocess.STDOUT,
                            cwd=plot_dir)
    
        subprocess.call(['%s/plot_page-pl' % self.dirbin, 
                                os.path.basename(plot_dir),
                                mode],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'a'),
                            stderr = subprocess.STDOUT,
                            cwd=pjoin(self.me_dir, 'HTML', self.run_name))
       
        shutil.move(pjoin(self.me_dir, 'HTML',self.run_name ,'plots.html'),
                                                                         output)
        
        self.update_status('End Plots for %s level' % mode, level = mode.lower(),
                                                                 makehtml=False)
        
        return True   

    
    def clean_pointless_card(self, mode):
        """ Clean the pointless card """
        if mode == 'parton':
            if os.path.exists(pjoin(self.me_dir,'Cards','pythia_card.dat')):
                os.remove(pjoin(self.me_dir,'Cards','pythia_card.dat'))
        elif mode in ['parton', 'pythia', 'delphes']:
            if os.path.exists(pjoin(self.me_dir,'Cards','pgs_card.dat')):
                    os.remove(pjoin(self.me_dir,'Cards','pgs_card.dat'))
        elif mode in ['pythia', 'pgs']:
            if os.path.exists(pjoin(self.me_dir,'Cards','delphes_card.dat')):
                    os.remove(pjoin(self.me_dir,'Cards','delphes_card.dat'))
            if os.path.exists(pjoin(self.me_dir,'Cards','delphes_trigger.dat')):
                    os.remove(pjoin(self.me_dir,'Cards','delphes_trigger.dat'))


    ############################################################################
    def ask_run_configuration(self, mode=None, force=False):
        """Ask the question when launching generate_events/multi_run"""
        
        available_mode = ['0', '1']
        if self.options['pythia-pgs_path']:
            available_mode.append('2')
            available_mode.append('3')

            if self.options['delphes_path']:
                available_mode.append('4')
        
        if len(available_mode) == 2:
            mode = 'parton'
        else:
            name = {'0': 'auto', '1': 'parton', '2':'pythia', '3':'pgs', '4':'delphes'}
            options = available_mode + [name[val] for val in available_mode]
            question = """Which programs do you want to run?
  0 / auto    : running existing card
  1 / parton  :  Madevent\n"""
            if '2' in available_mode:
                question += """  2 / pythia  : MadEvent + Pythia.
  3 / pgs     : MadEvent + Pythia + PGS.\n"""
            if '4' in available_mode:
                question += """  4 / delphes :  MadEvent + Pythia + Delphes.\n"""

            if not force:
                if not mode:
                    mode = self.ask(question, '0', options)
            elif not mode:
                mode = 'auto'
                
            if mode.isdigit():
                mode = name[mode]
            auto = False
            if mode == 'auto':
                auto = True
                if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pythia_card.dat')):
                    mode = 'parton'
                elif os.path.exists(pjoin(self.me_dir, 'Cards', 'pgs_card.dat')):
                    mode = 'pgs'
                elif os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_card.dat')):
                    mode = 'delphes'
                else: 
                    mode = 'pythia'
        logger.info('Will run in mode %s' % mode)
                                                                     
        self.clean_pointless_card(mode)
        # Now that we know in which mode we are check that all the card
        #exists (copy default if needed)

        cards = ['param_card.dat', 'run_card.dat']
        if mode in ['pythia', 'pgs', 'delphes']:
            self.add_card_to_run('pythia')
            cards.append('pythia_card.dat')
        if mode == 'pgs':
            self.add_card_to_run('pgs')
            cards.append('pgs_card.dat')
        elif mode == 'delphes':
            self.add_card_to_run('delphes')
            self.add_card_to_run('trigger')
            cards.append('delphes_card.dat')

        if force:
            return

        def get_question(mode):
            # Ask the user if he wants to edit any of the files
            #First create the asking text
            question = """Do you want to edit one cards (press enter to bypass editing)?
  1 / param   : param_card.dat (be carefull about parameter consistency, especially widths)
  2 / run     : run_card.dat\n"""
            possible_answer = ['0','done', 1, 'param', 2, 'run']
            if mode in ['pythia', 'pgs', 'delphes']:
                question += '  3 / pythia  : pythia_card.dat\n'
                possible_answer.append(3)
                possible_answer.append('pythia')
            if mode == 'pgs':
                question += '  4 / pgs     : pgs_card.dat\n'
                possible_answer.append(4)
                possible_answer.append('pgs')            
            elif mode == 'delphes':
                question += '  5 / delphes : delphes_card.dat\n'
                question += '  6 / trigger : delphes_trigger.dat\n'
                possible_answer.append(5)
                possible_answer.append('delphes')
                possible_answer.append(6)
                possible_answer.append('trigger')
            if self.options['madanalysis_path']:
                question += '  9 / plot    : plot_card.dat\n'
                possible_answer.append(9)
                possible_answer.append('plot')
            card = {0:'done', 1:'param', 2:'run', 3:'pythia', 
                      4: 'pgs', 5: 'delphes', 6:'trigger',9:'plot'}
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
                    print path
                self.exec_cmd('open %s' % path)                    
            else:
                # detect which card is provide
                card_name = self.detect_card_type(answer)
                if card_name == 'unknown':
                    card_name = self.ask('Fail to determine the type of the file. Please specify the format',
                   ['param_card.dat', 'run_card.dat','pythia_card.dat','pgs_card.dat',
                    'delphes_card.dat', 'delphes_trigger.dat','plot_card.dat'])
                elif card_name != 'banner':
                    logger.info('copy %s as %s' % (answer, card_name))
                    files.cp(answer, pjoin(self.me_dir, 'Cards', card_name))
                elif card_name == 'banner':
                    banner_mod.split_banner(answer, self.me_dir, proc_card=False)
                    logger.info('Splitting the banner in it\'s component')
                    if auto:
                        # Re-compute the current mode
                        mode = 'parton'
                        for level in ['delphes','pgs','pythia']:
                            if os.path.exists(pjoin(self.me_dir,'Cards','%s_card.dat' % level)):
                                mode = level
                                break
                    else:
                        clean_pointless_card(mode)
                    
                    
    ############################################################################
    def ask_pythia_run_configuration(self, mode=None, force=False):
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
            question += """  3 / delphes  : Pythia + Delphes.\n"""

        if not force:
            if not mode:
                mode = self.ask(question, '0', options)
        elif not mode:
            mode = 'auto'
            
        if mode.isdigit():
            mode = name[mode]
             
        if mode == 'auto':
            if os.path.exists(pjoin(self.me_dir, 'Cards', 'pgs_card.dat')):
                mode = 'pgs'
            elif os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_card.dat')):
                mode = 'delphes'
            else: 
                mode = 'pythia'
        logger.info('Will run in mode %s' % mode)
        
        self.clean_pointless_card(mode)
                                                 
        # Now that we know in which mode we are check that all the card
        #exists (copy default if needed)
        
        cards = ['pythia_card.dat']
        self.add_card_to_run('pythia')
        if mode == 'pgs':
            self.add_card_to_run('pgs')
            cards.append('pgs_card.dat')
        if mode == 'delphes':
            self.add_card_to_run('delphes')
            self.add_card_to_run('trigger')
            cards.append('delphes_card.dat')
        
        if force:
            return mode
        
        # Ask the user if he wants to edit any of the files
        #First create the asking text
        question = """Do you want to edit one cards (press enter to bypass editing)?\n"""
        question += """  1 / pythia   : pythia_card.dat\n""" 
        possible_answer = ['0','done', '1', 'pythia']
        card = {0:'done', 1:'pythia', 9:'plot'}
        if mode == 'pgs':
             question += '  2 / pgs     : pgs_card.dat\n'
             possible_answer.append(2)
             possible_answer.append('pgs') 
             card[2] = 'pgs'           
        if mode == 'delphes':
             question += '  2 / delphes : delphes_card.dat\n'
             question += '  3 / trigger : delphes_trigger.dat\n'
             possible_answer.append(2)
             possible_answer.append('delphes')
             possible_answer.append(3)
             possible_answer.append('trigger')
             card[2] = 'delphes'
             card[3] = 'trigger'
        if self.options['madanalysis_path']:
             question += '  9 / plot : plot_card.dat\n'
             possible_answer.append(9)
             possible_answer.append('plot')
        
        # Add the path options
        question += '  Path to a valid card.\n'
        
        # Loop as long as the user is not done.
        answer = 'no'
        while answer != 'done':
             answer = self.ask(question, '0', possible_answer, timeout=int(1.5*self.options['timeout']), path_msg='enter path')
             if answer.isdigit():
                 answer = card[int(answer)]
             if answer == 'done':
                 return
             if os.path.isfile(answer):
                 # detect which card is provide
                 card_name = self.detect_card_type(answer)
                 if card_name == 'unknown':
                     card_name = self.ask('Fail to determine the type of the file. Please specify the format',
                  ['pythia_card.dat','pgs_card.dat',
                   'delphes_card.dat', 'delphes_trigger.dat','plot_card.dat'])
        
                 logger.info('copy %s as %s' % (answer, card_name))
                 files.cp(answer, pjoin(self.me_dir, 'Cards', card_name))
                 continue
             if answer != 'trigger':
                 path = pjoin(self.me_dir,'Cards','%s_card.dat' % answer)
             else:
                 path = pjoin(self.me_dir,'Cards','delphes_trigger.dat')
             self.exec_cmd('open %s' % path)                    
                 
        return mode

    def ask_edit_cards(self, cards, fct_args):
        """Question for cards editions (used for pgs/delphes)"""

        if '-f' in fct_args or '--no_default' in fct_args:
            return
        
        card_name = {'pgs': 'pgs_card.dat',
                     'delphes': 'delphes_card.dat',
                     'trigger': 'delphes_trigger.dat'
                     }

        # Ask the user if he wants to edit any of the files
        #First create the asking text
        question = """Do you want to edit one cards (press enter to bypass editing)?\n""" 
        possible_answer = ['0', 'done']
        card = {0:'done'}
        
        for i, mode in enumerate(cards):
            possible_answer.append(i+1)
            possible_answer.append(mode)
            question += '  %s / %-9s : %s\n' % (i+1, mode, card_name[mode])
            card[i+1] = mode
        
        if self.options['madanalysis_path']:
             question += '  9 / %-9s : plot_card.dat\n' % 'plot'
             possible_answer.append(9)
             possible_answer.append('plot')
             card[9] = 'plot'

        # Add the path options
        question += '  Path to a valid card.\n'
        
        # Loop as long as the user is not done.
        answer = 'no'
        while answer != 'done':
             answer = self.ask(question, '0', possible_answer, timeout=int(1.5*self.options['timeout']), path_msg='enter path')
             if answer.isdigit():
                 answer = card[int(answer)]
             if answer == 'done':
                 return
             if os.path.isfile(answer):
                 # detect which card is provide
                 card_name = self.detect_card_type(answer)
                 if card_name == 'unknown':
                     card_name = self.ask('Fail to determine the type of the file. Please specify the format',
                  ['pgs_card.dat', 'delphes_card.dat', 'delphes_trigger.dat'])
        
                 logger.info('copy %s as %s' % (answer, card_name))
                 files.cp(answer, pjoin(self.me_dir, 'Cards', card_name))
                 continue
             if answer != 'trigger':
                 path = pjoin(self.me_dir,'Cards','%s_card.dat' % answer)
             else:
                 path = pjoin(self.me_dir,'Cards','delphes_trigger.dat')
             self.exec_cmd('open %s' % path)                    
                 
        return mode


        question = """Do you want to edit the %s?""" % card
        answer = self.ask(question, 'n', ['y','n'],path_msg='enter path')
        if answer == 'y':
            path = pjoin(self.me_dir,'Cards', card)
            self.exec_cmd('open %s' % path)
        elif answer != 'n':
            card_name = self.detect_card_type(answer)
            if card_name != card:
                raise self.InvalidCmd('Invalid File Format for a %s' % card)
            logger.info('copy %s as %s' % (answer, card_name))
            files.cp(answer, pjoin(self.me_dir, 'Cards', card_name))   



    def add_card_to_run(self, name):
        """ensure that card name is define. If not use the default one"""
        dico = {'dir': self.me_dir, 'name': name }

        if name != 'trigger':
            if not os.path.isfile('%(dir)s/Cards/%(name)s_card.dat' % dico):
                files.cp('%(dir)s/Cards/%(name)s_card_default.dat' % dico,
                         '%(dir)s/Cards/%(name)s_card.dat' % dico)
        else:
            if not os.path.isfile('%(dir)s/Cards/delphes_trigger.dat' % dico):
                files.cp('%(dir)s/Cards/delphes_trigger_default.dat' % dico,
                         '%(dir)s/Cards/delphes_trigger.dat' % dico) 
            
    @staticmethod
    def detect_card_type(path):
        """detect the type of the card. Return value are
           banner
           param_card.dat
           run_card.dat
           pythia_card.dat
           plot_card.dat
           pgs_card.dat
           delphes_card.dat
           delphes_trigger.dat
        """
        
        text = open(path).read()
        text = re.findall('(<MGVersion>|CEN_max_tracker|#TRIGGER CARD|parameter set name|muon eta coverage|MSTP|MSTU|Begin Minpts|gridpack|ebeam1|BLOCK|DECAY)', text, re.I)
        text = [t.lower() for t in text]
        if '<mgversion>' in text:
            return 'banner'
        elif 'cen_max_tracker' in text:
            return 'delphes_card.dat'
        elif '#trigger card' in text:
            return 'delphes_trigger.dat'
        elif 'parameter set name' in text:
            return 'pgs_card.dat'
        elif 'muon eta coverage' in text:
            return 'pgs_card.dat'
        elif 'mstp' in text:
            return 'pythia_card.dat'
        elif 'mstu' in text:
            return 'pythia_param_card.dat'
        elif 'begin minpts' in text:
            return 'plot_card.dat'
        elif 'gridpack' in text and 'ebeam1' in text:
            return 'run_card.dat'
        elif 'block' in text and 'decay' in text: 
            return 'param_card.dat'
        else:
            return 'unknown'


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
            return make_info_html.get_subprocess_info_v4(path)

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
        subprocess.call([pjoin(self.me_dir,'bin','internal','restore_data'),
                         'default'],
            cwd=self.me_dir)

        # 2) Run the refine for the grid
        self.update_status('Generating Events', level=None)
        #subprocess.call([pjoin(self.me_dir,'bin','refine4grid'),
        #                str(nb_event), '0', 'Madevent','1','GridRun_%s' % seed],
        #                cwd=self.me_dir)
        self.refine4grid(nb_event)

        # 3) Combine the events/pythia/...
        self.exec_cmd('combine_events')
        self.exec_cmd('store_events')
        self.exec_cmd('pythia --no_default -f')
        self.print_results_in_shell(self.results.current)

    def refine4grid(self, nb_event):
        """Special refine for gridpack run."""
        self.nb_refine += 1
        
        precision = nb_event

        # initialize / remove lhapdf mode
        # self.configure_directory() # All this has been done before
        self.cluster_mode = 0 # force single machine
        
        self.update_status('Refine results to %s' % precision, level=None)
        logger.info("Using random number seed offset = %s" % self.random)
        
        self.total_jobs = 0
        subproc = [P for P in os.listdir(pjoin(self.me_dir,'SubProcesses')) if 
                   P.startswith('P') and os.path.isdir(pjoin(self.me_dir,'SubProcesses', P))]
        for nb_proc,subdir in enumerate(subproc):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            bindir = pjoin(os.path.relpath(self.dirbin, Pdir))
                           
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(pjoin(Pdir, match))
            
            devnull = os.open(os.devnull, os.O_RDWR)
            logfile = pjoin(Pdir, 'gen_ximprove.log')
            proc = subprocess.Popen([pjoin(bindir, 'gen_ximprove')],
                                    stdin=subprocess.PIPE,
                                    stdout=open(logfile,'w'),
                                    cwd=Pdir)
            proc.communicate('%s 1 F\n' % (precision))

            if os.path.exists(pjoin(Pdir, 'ajob1')):
                # misc.compile(['madevent'], cwd=Pdir) # Done before
                #
                os.system("chmod +x %s/ajob*" % Pdir)
                alljobs = glob.glob(pjoin(Pdir,'ajob*'))
                nb_tot = len(alljobs)            
                self.total_jobs += nb_tot
                for i, job in enumerate(alljobs):
                    job = os.path.basename(job)
                    self.launch_job('./%s' % job, cwd=Pdir, remaining=(nb_tot-i-1), 
                             run_type='Refine number %s on %s (%s/%s)' % (self.nb_refine, subdir, nb_proc+1, len(subproc)))
        self.monitor(run_type='All job submitted for refine number %s' % self.nb_refine,
                     html=True)
        
        self.update_status("Combining runs", level='parton')
        try:
            os.remove(pjoin(Pdir, 'combine_runs.log'))
        except:
            pass
        
        bindir = pjoin(os.path.relpath(self.dirbin, pjoin(self.me_dir,'SubProcesses')))
        subprocess.call([pjoin(bindir, 'combine_runs')], 
                                          cwd=pjoin(self.me_dir,'SubProcesses'),
                                          stdout=devnull)
        
        #subprocess.call([pjoin(self.dirbin, 'sumall')], 
        #                                 cwd=pjoin(self.me_dir,'SubProcesses'),
        #                                 stdout=devnull)
        
        self.update_status('finish refine', 'parton', makehtml=False)

    
