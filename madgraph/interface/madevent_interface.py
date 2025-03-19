###############################################################################
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

from __future__ import absolute_import
import collections
import itertools
import glob
import logging
import math
import os
import random
import re

import stat
import subprocess
import sys
import time
import tarfile
import shutil
import copy
from six.moves import range
import six
StringIO = six
try:
    import readline
    GNU_SPLITTING = ('GNU' in readline.__doc__)
except:
    GNU_SPLITTING = True

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.split(root_path)[0]
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(root_path,'bin'))

# usefull shortcut
pjoin = os.path.join
# Special logger for the Cmd Interface
logger = logging.getLogger('madevent.stdout') # -> stdout
logger_stderr = logging.getLogger('madevent.stderr') # ->stderr
 
try:
    import madgraph
except ImportError as error: 
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
#    import internal.histograms as histograms # imported later to not slow down the loading of the code
    from internal.files import ln
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
#    import madgraph.various.histograms as histograms  # imported later to not slow down the loading of the code
    import models.check_param_card as check_param_card
    from madgraph.iolibs.files import ln    
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR, ReadWrite



class MadEventError(Exception): pass
ZeroResult = common_run.ZeroResult
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
    error_debug = 'Please report this bug on https://bugs.launchpad.net/mg5amcnlo\n'
    error_debug += 'More information is found in \'%(debug)s\'.\n' 
    error_debug += 'Please attach this file to your report.'

    config_debug = 'If you need help with this issue please contact us on https://answers.launchpad.net/mg5amcnlo\n'


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
        if info and 'version' in info and  'date' in info:
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
        return cmd.Cmd.nice_user_error(self, error, line)            
        
    def nice_config_error(self, error, line):
        """If a ME run is currently running add a link in the html output"""

        self.add_error_log_in_html()
        stop = cmd.Cmd.nice_config_error(self, error, line)
        
        
        try:
            debug_file = open(self.debug_output, 'a')
            debug_file.write(open(pjoin(self.me_dir,'Cards','proc_card_mg5.dat')))
            debug_file.close()
        except:
            pass 
        return stop
            

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
            stop = cmd.Cmd.nice_error_handling(self, error, line)
            try:
                debug_file = open(self.debug_output, 'a')
                debug_file.write(open(pjoin(self.me_dir,'Cards','proc_card_mg5.dat')))
                debug_file.close()
            except:
                pass
            return stop
        
        
#===============================================================================
# HelpToCmd
#===============================================================================
class HelpToCmd(object):
    """ The Series of help routine for the MadEventCmd"""
    
    def help_pythia(self):
        logger.info("syntax: pythia [RUN] [--run_options]")
        logger.info("-- run pythia on RUN (current one by default)")
        self.run_options_help([('-f','answer all question by default'),
                               ('--tag=', 'define the tag for the pythia run'),
                               ('--no_default', 'not run if pythia_card not present')])

    def help_pythia8(self):
        logger.info("syntax: pythia8 [RUN] [--run_options]")
        logger.info("-- run pythia8 on RUN (current one by default)")
        self.run_options_help([('-f','answer all question by default'),
                               ('--tag=', 'define the tag for the pythia8 run'),
                               ('--no_default', 'not run if pythia8_card not present')])

    def help_rivet(self):
        logger.info("syntax: rivet [RUN] [--run_options]")
        logger.info("-- run rivet on RUN (current one by default)")
        self.run_options_help([('-f','answer all question by default'),
                               ('--tag=', 'define the tag for the rivet run'),
                               ('--no_default', 'not run if rivet_card not present')])

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
     
     
    def help_restart_gridpack(self):
        logger.info("syntax: restart_gridpack --precision= --restart_zero")


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

class AskRun(cmd.ControlSwitch):
    """a class for the question on what to do on a madevent run"""

    to_control = [('shower', 'Choose the shower/hadronization program'),
                      ('detector', 'Choose the detector simulation program'),
                      ('analysis', 'Choose an analysis package (plot/convert)'),
                      ('madspin', 'Decay onshell particles'),
                      ('reweight', 'Add weights to events for new hypp.')
                ]
    
    def __init__(self, question, line_args=[], mode=None, force=False,
                                                                  *args, **opt):
        
        self.check_available_module(opt['mother_interface'].options)
        self.me_dir = opt['mother_interface'].me_dir
        super(AskRun,self).__init__(self.to_control, opt['mother_interface'],
                                     *args, **opt)
        
        
    def check_available_module(self, options):
        
        self.available_module = set()
        if options['pythia-pgs_path']:
            self.available_module.add('PY6')
            self.available_module.add('PGS')
        if options['pythia8_path']:
            self.available_module.add('PY8')
        if options['madanalysis_path']:
            self.available_module.add('MA4')
        if options['madanalysis5_path']:
            self.available_module.add('MA5')
        if options['exrootanalysis_path']:
            self.available_module.add('ExRoot')
        if options['delphes_path']:
            if 'PY6' in self.available_module or 'PY8' in self.available_module:
                self.available_module.add('Delphes')
            else:
                logger.warning("Delphes program installed but no parton shower module detected.\n    Please install pythia8")
        if options['rivet_path']:
            if 'PY8' in self.available_module:
                self.available_module.add('Rivet')
            else:
                logger.warning("Rivet program installed but no parton shower with hepmc output detected.\n    Please install pythia8")
        
        if not MADEVENT or ('mg5_path' in options and options['mg5_path']):
            self.available_module.add('MadSpin')
            if misc.has_f2py() or options['f2py_compiler']:
                self.available_module.add('reweight')

#   old mode to activate the shower            
    def ans_parton(self, value=None):
        """None: means that the user type 'pythia'
           value: means that the user type pythia=value"""
        
        if value is None:
            self.set_all_off()
        else:
            logger.warning('Invalid command: parton=%s' % value)
            
                
#
#   HANDLING SHOWER 
#
    def get_allowed_shower(self):
        """return valid entry for the shower switch"""
        
        if hasattr(self, 'allowed_shower'):
            return self.allowed_shower
        
        self.allowed_shower = []
        if 'PY6' in self.available_module:
            self.allowed_shower.append('Pythia6')
        if 'PY8' in self.available_module:
            self.allowed_shower.append('Pythia8')
        if self.allowed_shower:
            self.allowed_shower.append('OFF')
        return self.allowed_shower
    
    def set_default_shower(self):
        
        if 'PY6' in self.available_module and\
                   os.path.exists(pjoin(self.me_dir,'Cards','pythia_card.dat')):
            self.switch['shower'] = 'Pythia6'
        elif 'PY8' in self.available_module and\
                  os.path.exists(pjoin(self.me_dir,'Cards','pythia8_card.dat')):
            self.switch['shower'] = 'Pythia8'
        elif self.get_allowed_shower():
            self.switch['shower'] = 'OFF'
        else:
            self.switch['shower'] = 'Not Avail.'

    def check_value_shower(self, value):
        """check an entry is valid. return the valid entry in case of shortcut"""
        
        if value in self.get_allowed_shower():
            return True
        
        value =value.lower()
        if value in ['py6','p6','pythia_6'] and 'PY6' in self.available_module:
            return 'Pythia6'
        elif value in ['py8','p8','pythia_8'] and 'PY8' in self.available_module:
            return 'Pythia8'
        else:
            return False
            
            
#   old mode to activate the shower            
    def ans_pythia(self, value=None):
        """None: means that the user type 'pythia'
           value: means that the user type pythia=value"""
        
        if 'PY6' not in self.available_module:
            logger.info('pythia-pgs not available. Ignore commmand')
            return

        if value is None:
            self.set_all_off()
            self.switch['shower'] = 'Pythia6'
        elif value == 'on':
            self.switch['shower'] = 'Pythia6'
        elif value == 'off':
            self.set_switch('shower', 'OFF')
        else:
            logger.warning('Invalid command: pythia=%s' % value)
            
            
    def consistency_shower_detector(self, vshower, vdetector):
        """consistency_XX_YY(val_XX, val_YY)
           -> XX is the new key set by the user to a new value val_XX
           -> YY is another key
           -> return value should be None or "replace_YY" 
        """

        if vshower == 'OFF':
            if self.check_value('detector', vdetector) and  vdetector!= 'OFF':
                return 'OFF'
        if vshower == 'Pythia8' and vdetector == 'PGS':
            return 'OFF'
        
        return None


        
#
#   HANDLING DETECTOR
#
    def get_allowed_detector(self):
        """return valid entry for the switch"""
 
        if hasattr(self, 'allowed_detector'):
            return self.allowed_detector 
        
        self.allowed_detector = []
        if 'PGS' in self.available_module:
            self.allowed_detector.append('PGS')
        if 'Delphes' in self.available_module:
            self.allowed_detector.append('Delphes')

            
        if self.allowed_detector:
            self.allowed_detector.append('OFF')
        return self.allowed_detector  

    def set_default_detector(self):
        
        self.set_default_shower() #ensure that this one is called first!
        
        if 'PGS' in self.available_module and self.switch['shower'] == 'Pythia6'\
                  and os.path.exists(pjoin(self.me_dir,'Cards','pgs_card.dat')):
            self.switch['detector'] = 'PGS'
        elif 'Delphes' in self.available_module and self.switch['shower'] != 'OFF'\
              and os.path.exists(pjoin(self.me_dir,'Cards','delphes_card.dat')):
            self.switch['detector'] = 'Delphes'
        elif self.get_allowed_detector():
            self.switch['detector'] = 'OFF'
        else:
            self.switch['detector'] =  'Not Avail.'
                
#   old mode to activate pgs            
    def ans_pgs(self, value=None):
        """None: means that the user type 'pgs'
           value: means that the user type pgs=value"""        
        
        if 'PGS' not in self.available_module:
            logger.info('pythia-pgs not available. Ignore commmand')
            return
        
        if value is None:
            self.set_all_off()
            self.switch['shower'] = 'Pythia6'
            self.switch['detector'] = 'PGS'
        elif value == 'on':
            self.switch['shower'] = 'Pythia6'
            self.switch['detector'] = 'PGS'
        elif value == 'off':
            self.set_switch('detector', 'OFF')
        else:
            logger.warning('Invalid command: pgs=%s' % value)

            
#   old mode to activate Delphes
    def ans_delphes(self, value=None):
        """None: means that the user type 'delphes'
           value: means that the user type delphes=value"""          
        
        if 'Delphes' not in self.available_module:
            logger.warning('Delphes not available. Ignore commmand')
            return
        
        if value is None:
            self.set_all_off()
            if 'PY6' in self.available_module:
                self.switch['shower'] = 'Pythia6'
            else:
                self.switch['shower'] = 'Pythia8'
            self.switch['detector'] = 'Delphes'
        elif value == 'on':
            return self.ans_delphes(None)
        elif value == 'off':
            self.set_switch('detector', 'OFF')
        else:
            logger.warning('Invalid command: pgs=%s' % value)        

    def consistency_detector_shower(self,vdetector, vshower):
        """consistency_XX_YY(val_XX, val_YY)
           -> XX is the new key set by the user to a new value val_XX
           -> YY is another key
           -> return value should be None or "replace_YY" 
        """
        
        if vdetector == 'PGS' and vshower != 'Pythia6':
            return 'Pythia6'
        if vdetector == 'Delphes' and vshower  not in ['Pythia6', 'Pythia8']:
            if 'PY8' in self.available_module:
                return 'Pythia8'
            elif 'PY6' in self.available_module:
                return 'Pythia6'
            else:
                raise Exception
        return None


#
#   HANDLING ANALYSIS
#
    def get_allowed_analysis(self):
        """return valid entry for the shower switch"""
        
        if hasattr(self, 'allowed_analysis'):
            return self.allowed_analysis
        
        self.allowed_analysis = []
        if 'ExRoot' in self.available_module:
            self.allowed_analysis.append('ExRoot')
        if 'MA4' in self.available_module:
            self.allowed_analysis.append('MadAnalysis4')
        if 'MA5' in self.available_module:
            self.allowed_analysis.append('MadAnalysis5') 
        if 'Rivet' in self.available_module:
            self.allowed_analysis.append('Rivet') 
            
        if self.allowed_analysis:
            self.allowed_analysis.append('OFF')
            
        return self.allowed_analysis
   
    def check_analysis(self, value):
        """check an entry is valid. return the valid entry in case of shortcut"""
        
        if value in self.get_allowed_analysis():
            return True
        if value.lower() in ['ma4', 'madanalysis4', 'madanalysis_4','4']:
            return 'MadAnalysis4'
        if value.lower() in ['ma5', 'madanalysis5', 'madanalysis_5','5']:
            return 'MadAnalysis5'
        if value.lower() in ['ma', 'madanalysis']:
            if 'MA5' in self.available_module:
                return 'MadAnalysis5'
            elif 'MA4' in self.available_module:
                return 'MadAnalysis4'
            else:
                return False
        else:
            return False

    def consistency_shower_analysis(self, vshower, vanalysis):
        """consistency_XX_YY(val_XX, val_YY)
           -> XX is the new key set by the user to a new value val_XX
           -> YY is another key
           -> return value should be None or "replace_YY" 
        """

        if vshower != 'Pythia8' and vanalysis == 'Rivet':
            return 'OFF' #new value for analysis
        
        return None
        
    def consistency_analysis_shower(self, vanalysis, vshower):
        """consistency_XX_YY(val_XX, val_YY)
           -> XX is the new key set by the user to a new value val_XX
           -> YY is another key
           -> return value should be None or "replace_YY" 
        """

        if vshower != 'Pythia8' and vanalysis == 'Rivet':
            return 'Pythia8' #new value for analysis
        
        return None


    def set_default_analysis(self):
        """initialise the switch for analysis"""
        
        if 'MA4' in self.available_module and \
                     os.path.exists(pjoin(self.me_dir,'Cards','plot_card.dat')):
            self.switch['analysis'] = 'MadAnalysis4'
        elif 'MA5' in self.available_module and\
             (os.path.exists(pjoin(self.me_dir,'Cards','madanalysis5_parton_card.dat'))\
             or os.path.exists(pjoin(self.me_dir,'Cards', 'madanalysis5_hadron_card.dat'))):
            self.switch['analysis'] = 'MadAnalysis5'
        elif 'ExRoot' in self.available_module:
            self.switch['analysis'] = 'ExRoot'   
        elif self.get_allowed_analysis(): 
            self.switch['analysis'] = 'OFF'
        else:
            self.switch['analysis'] = 'Not Avail.'
            
#
#   MADSPIN handling
#
    def get_allowed_madspin(self):
        """ ON|OFF|onshell """
        
        if hasattr(self, 'allowed_madspin'):
            return self.allowed_madspin
        
        self.allowed_madspin = []
        if 'MadSpin'  in self.available_module:
            self.allowed_madspin = ['OFF',"ON",'onshell',"full"]
        return self.allowed_madspin
    
    def check_value_madspin(self, value):
        """handle alias and valid option not present in get_allowed_madspin"""
        
        if value.upper() in self.get_allowed_madspin():
            return True
        elif value.lower() in self.get_allowed_madspin():
            return True
        
        if 'MadSpin' not in self.available_module:
            return False
             
        if value.lower() in ['madspin', 'full']:
            return 'full'
        elif value.lower() in ['none']:
            return 'none'
        
    
    def set_default_madspin(self):
        """initialise the switch for madspin"""
        
        if 'MadSpin' in self.available_module:
            if os.path.exists(pjoin(self.me_dir,'Cards','madspin_card.dat')):
                self.switch['madspin'] = 'ON'
            else:
                self.switch['madspin'] = 'OFF'
        else:
            self.switch['madspin'] = 'Not Avail.'
            
    def get_cardcmd_for_madspin(self, value):
        """set some command to run before allowing the user to modify the cards."""
        
        if value == 'onshell':
            return ["edit madspin_card --replace_line='set spinmode' --before_line='decay' set spinmode onshell"]
        elif value in ['full', 'madspin']:
            return ["edit madspin_card --replace_line='set spinmode' --before_line='decay' set spinmode full"]
        elif value == 'none':
            return ["edit madspin_card --replace_line='set spinmode' --before_line='decay' set spinmode none"]
        else:
            return []
        
#
#   ReWeight handling
#
    def get_allowed_reweight(self):
        """ return the list of valid option for reweight=XXX """
        
        if hasattr(self, 'allowed_reweight'):
            return getattr(self, 'allowed_reweight')
        
        if 'reweight' not in self.available_module:
            self.allowed_reweight = []
            return
        self.allowed_reweight = ['OFF', 'ON']
        
        # check for plugin mode
        plugin_path = self.mother_interface.plugin_path
        opts = misc.from_plugin_import(plugin_path, 'new_reweight', warning=False)
        self.allowed_reweight += opts
        
    def set_default_reweight(self):
        """initialise the switch for reweight"""
        
        if 'reweight' in self.available_module:
            if os.path.exists(pjoin(self.me_dir,'Cards','reweight_card.dat')):
                self.switch['reweight'] = 'ON'
            else:
                self.switch['reweight'] = 'OFF'
        else:
            self.switch['reweight'] = 'Not Avail.'        

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
            banners = misc.glob('*_banner.txt', pjoin(self.me_dir,'Events', args[0]))
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

        if args[0] not in self._set_options + list(self.options.keys()):
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
                                (opt, value[0]), globals(), {'self':self, 'arg':arg})
                        arg = ""
                if arg != "": raise Exception
            except Exception as error:
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
        #pass nb run to an integer
        nb_run = args.pop(0)
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
        Note that other option are already removed at this point
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
    
    def check_pythia8(self, args):
        """Check the argument for pythia command
        syntax: pythia8 [NAME] 
        Note that other option are already removed at this point
        """        
        mode = None
        laststep = [arg for arg in args if arg.startswith('--laststep=')]
        if laststep and len(laststep)==1:
            mode = laststep[0].split('=')[-1]
            if mode not in ['auto', 'pythia','pythia8','delphes']:
                self.help_pythia8()
                raise self.InvalidCmd('invalid %s argument'% args[-1])     
        elif laststep:
            raise self.InvalidCmd('only one laststep argument is allowed')

        # If not pythia-pgs path
        if not self.options['pythia8_path']:
            logger.info('Retry reading configuration file to find pythia8 path')
            self.set_configuration()
            
        if not self.options['pythia8_path'] or not \
            os.path.exists(pjoin(self.options['pythia8_path'],'bin','pythia8-config')):
            error_msg = 'No valid pythia8 path set.\n'
            error_msg += 'Please use the set command to define the path and retry.\n'
            error_msg += 'You can also define it in the configuration file.\n'
            error_msg += 'Finally, it can be installed automatically using the'
            error_msg += ' install command.\n'
            raise self.InvalidCmd(error_msg)

        tag = [a for a in args if a.startswith('--tag=')]
        if tag: 
            args.remove(tag[0])
            tag = tag[0][6:]

        if len(args) == 0 and not self.run_name:
            if self.results.lastrun:
                args.insert(0, self.results.lastrun)
            else:
                raise self.InvalidCmd('No run name currently define. '+
                                                 'Please add this information.')             
        
        if len(args) >= 1:
            if args[0] != self.run_name and\
             not os.path.exists(pjoin(self.me_dir,'Events',args[0], 
                                                  'unweighted_events.lhe.gz')):
                raise self.InvalidCmd('No events file corresponding to %s run. '
                                                                      % args[0])
            self.set_run_name(args[0], tag, 'pythia8')
        else:
            if tag:
                self.run_card['run_tag'] = tag
            self.set_run_name(self.run_name, tag, 'pythia8')

        input_file = pjoin(self.me_dir,'Events',self.run_name, 'unweighted_events.lhe')
        if not os.path.exists('%s.gz'%input_file):
            if os.path.exists(input_file):
                misc.gzip(input_file, stdout='%s.gz'%input_file)
            else:
                raise self.InvalidCmd('No event file corresponding to %s run. '
                                                                % self.run_name)
        
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
        
    
    def check_pgs(self, arg, no_default=False):
        """Check the argument for pythia command
        syntax is  "pgs [NAME]" 
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
            if not no_default:
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

    def check_display(self, args):
        """check the validity of line
        syntax is "display XXXXX"
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
    
    
    def complete_banner_run(self, text, line, begidx, endidx, formatting=True):
       "Complete the banner run command"
       try:
  
        
        args = self.split_arg(line[0:begidx], error=False)
        
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        os.path.join('.',*[a for a in args \
                                                    if a.endswith(os.path.sep)]))        
        
        
        if len(args) > 1:
            # only options are possible
            tags = misc.glob('%s_*_banner.txt' % args[1], pjoin(self.me_dir, 'Events' , args[1]))
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

        run_list =  misc.glob(pjoin('*','*_banner.txt'), pjoin(self.me_dir, 'Events'))
        run_list = [n.rsplit('/',2)[1] for n in run_list]
        possibilites['RUN Name'] = self.list_completion(text, run_list)
        
        return self.deal_multiple_categories(possibilites, formatting)
    
        
       except Exception as error:
           print(error)


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
            return self.list_completion(text, self._set_options + list(self.options.keys()) )

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
            return self.list_completion(text, self._plot_mode + list(self.results.keys()))
        
    def complete_syscalc(self, text, line, begidx, endidx, formatting=True):
        """ Complete the syscalc command """
        
        output = {}
        args = self.split_arg(line[0:begidx], error=False)
                
        if len(args) <=1:
            output['RUN_NAME'] = self.list_completion(list(self.results.keys()))
        output['MODE'] =  self.list_completion(text, self._syscalc_mode)
        output['options'] = ['-f']
        if len(args) > 1 and (text.startswith('--t')):
            run = args[1]
            if run in self.results:
                tags = ['--tag=%s' % tag['tag'] for tag in self.results[run]]
                output['options'] += tags
        
        return self.deal_multiple_categories(output, formatting)
        
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
            data = misc.glob(pjoin('*','*_banner.txt'), pjoin(self.me_dir, 'Events'))
            data = [n.rsplit('/',2)[1] for n in data]
            return self.list_completion(text, ['all'] + data)
         
    
    def complete_shower(self,text, line, begidx, endidx):
        "Complete the shower command"
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) == 1:
            return self.list_completion(text, self._interfaced_showers)
        elif len(args)>1 and args[1] in self._interfaced_showers:
            return getattr(self, 'complete_%s' % text)\
                (text, args[1],line.replace(args[0]+' ',''), 
                 begidx-len(args[0])-1, endidx-len(args[0])-1)

    def complete_pythia8(self,text, line, begidx, endidx):
        "Complete the pythia8 command"
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) == 1:
            #return valid run_name
            data = misc.glob(pjoin('*','unweighted_events.lhe.gz'),pjoin(self.me_dir, 'Events'))
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

    def complete_madanalysis5_parton(self,text, line, begidx, endidx):
        "Complete the madanalysis5 command"
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) == 1:
            #return valid run_name
            data = []
            for name in ['unweighted_events.lhe']:
                data += misc.glob(pjoin('*','%s'%name), pjoin(self.me_dir, 'Events'))
                data += misc.glob(pjoin('*','%s.gz'%name), pjoin(self.me_dir, 'Events'))
            data = [n.rsplit('/',2)[1] for n in data]
            tmp1 =  self.list_completion(text, data)
            if not self.run_name:
                return tmp1
            else:
                tmp2 = self.list_completion(text, ['-f',
                '--MA5_stdout_lvl=','--no_default','--tag='], line)
                return tmp1 + tmp2            
        elif '--MA5_stdout_lvl=' in line and not any(arg.startswith(
                                          '--MA5_stdout_lvl=') for arg in args):
            return self.list_completion(text, 
                ['--MA5_stdout_lvl=%s'%opt for opt in 
                ['logging.INFO','logging.DEBUG','logging.WARNING',
                                                'logging.CRITICAL','90']], line)
        else:
            return self.list_completion(text,  ['-f', 
                             '--MA5_stdout_lvl=','--no_default','--tag='], line)

    def complete_pythia(self,text, line, begidx, endidx):
        "Complete the pythia command"     
        args = self.split_arg(line[0:begidx], error=False)

        if len(args) == 1:
            #return valid run_name
            data = misc.glob(pjoin('*','unweighted_events.lhe.gz'), pjoin(self.me_dir, 'Events'))
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
            data = misc.glob(pjoin('*', '*_pythia_events.hep.gz'), pjoin(self.me_dir, 'Events'))
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
    complete_rivet = complete_pgs     

#===============================================================================
# MadEventCmd
#===============================================================================
class MadEventCmd(CompleteForCmd, CmdExtended, HelpToCmd, common_run.CommonRunCmd):

    """The command line processor of Mad Graph"""    
    

    LO = True
    # Truth values
    true = ['T','.true.',True,'true']
    # Options and formats available
    _run_options = ['--cluster','--multicore','--nb_core=','--nb_core=2', '-c', '-m']
    _generate_options = ['-f', '--laststep=parton', '--laststep=pythia', '--laststep=pgs', '--laststep=delphes']
    _calculate_decay_options = ['-f', '--accuracy=0.']
    _interfaced_showers = ['pythia','pythia8']
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
    
    asking_for_run = AskRun
    
    ############################################################################
    def __init__(self, me_dir = None, options={}, *completekey, **stdin):
        """ add information to the cmd """

        CmdExtended.__init__(self, me_dir, options, *completekey, **stdin)
        #common_run.CommonRunCmd.__init__(self, me_dir, options)

        self.mode = 'madevent'
        self.nb_refine=0
        if self.web:
            os.system('touch %s' % pjoin(self.me_dir,'Online'))

        self.load_results_db()        
        self.results.def_web_mode(self.web)

        self.Gdirs = None
        
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
                elif key == "rivet_path":
                    if not os.path.exists(pjoin(path, 'bin', 'rivet')):
                        logger.info("No valid rivet path found")
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
            data = misc.glob(pjoin('*','*_banner.txt'), pjoin(self.me_dir, 'Events'))
            data = [n.rsplit('/',2)[1:] for n in data]
            
            if data:
                out = {}
                for name, tag in data:
                    tag = tag[len(name)+1:-11]
                    if name in out:
                        out[name].append(tag)
                    else:
                        out[name] = [tag]
                print('the runs available are:')
                for run_name, tags in out.items():
                    print('  run: %s' % run_name)
                    print('       tags: ', end=' ') 
                    print(', '.join(tags))
            else:
                print('No run detected.')
                
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
    def do_restart_gridpack(self, line):
        """ syntax restart_gridpack --precision=1.0 --restart_zero
        collect the result of the current run and relaunch each channel
        not completed or optionally a completed one with a precision worse than 
        a threshold (and/or the zero result channel)"""
        
    
        args = self.split_arg(line)
        # Check argument's validity
        self.check_survey(args)
    
        # initialize / remove lhapdf mode
        #self.run_card = banner_mod.RunCard(pjoin(self.me_dir, 'Cards', 'run_card.dat'))
        #self.configure_directory()
        
        gensym = gen_ximprove.gensym(self)
        
        min_precision = 1.0
        resubmit_zero=False
        if '--precision=' in line:
            s = line.index('--precision=') + len('--precision=')
            arg=line[s:].split(1)[0]
            min_precision = float(arg)
        
        if '--restart_zero' in line:
            resubmit_zero = True
            
            
        gensym.resubmit(min_precision, resubmit_zero)
        self.monitor(run_type='All jobs submitted for gridpack', html=True)

                        #will be done during the refine (more precisely in gen_ximprove)
        cross, error = sum_html.make_all_html_results(self)
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)  
        self.exec_cmd("print_results %s" % self.run_name,
                       errorhandling=False, printcmd=False, precmd=False, postcmd=False)      
        
        self.results.add_detail('run_statistics', dict(gensym.run_statistics))

        
        #self.exec_cmd('combine_events', postcmd=False)
        #self.exec_cmd('store_events', postcmd=False)
        self.exec_cmd('decay_events -from_cards', postcmd=False)
        self.exec_cmd('create_gridpack', postcmd=False)
        
    

    ############################################################################    

    ############################################################################
    def do_generate_events(self, line):
        """Main Commands: launch the full chain """
        
        self.banner = None
        self.Gdirs = None
        
        args = self.split_arg(line)
        # Check argument's validity
        mode = self.check_generate_events(args)
        switch_mode = self.ask_run_configuration(mode, args)
        with misc.TMP_variable(banner_mod.RunCard, 'allow_scan', True):
            run_card = banner_mod.RunCard(pjoin(self.me_dir, 'Cards', 'run_card.dat'), consistency=False)
        if not run_card.scan_set:
            if not args:
                # No run name assigned -> assigned one automaticaly 
                self.set_run_name(self.find_available_run_name(self.me_dir), None, 'parton')
            else:
                self.set_run_name(args[0], None, 'parton', True)
                args.pop(0) 

        
        self.run_generate_events(switch_mode, args)

        self.postprocessing()


    # postprocessing : runs after all the event generation has been done
    # even for the 'scan' mode, madevent->pythia->madevent->pythia->...->POSTPROCESSING
    def postprocessing(self):

        # Run Rivet postprocessor
        cmd_do_rivet = common_run.CommonRunCmd.do_rivet(self,"--no_default",True)
        if cmd_do_rivet:
            rivet_config = cmd_do_rivet[0]
            postprocess_RIVET = cmd_do_rivet[1]
            postprocess_CONTUR = cmd_do_rivet[2]
            if postprocess_RIVET or postprocess_CONTUR:
                self.rivet_postprocessing(rivet_config, postprocess_RIVET, postprocess_CONTUR)

    def rivet_postprocessing(self, rivet_config, postprocess_RIVET, postprocess_CONTUR):

        # Check number of Rivet jobs to run 
        run_dirs = [pjoin(self.me_dir, 'Events',run_name) 
                      for run_name in self.postprocessing_dirs]

        nb_rivet = len(run_dirs)

        if postprocess_RIVET:

            # Submit Rivet jobs
            for i_rivet in range(nb_rivet):
                self.cluster.submit2(pjoin(run_dirs[i_rivet], "run_rivet.sh"), argument=[str(i_rivet)])

            startRivet = time.time()

            def wait_monitoring(Idle, Running, Done):
                if Idle+Running+Done == 0:
                    return
                logger.info('Rivet analysis jobs: %d Idle, %d Running, %d Done [%s]'\
                             %(Idle, Running, Done, misc.format_time(time.time() - startRivet)))
            self.cluster.wait(pjoin(self.me_dir, 'Events'),wait_monitoring)

            self.update_status("postprocessing rivet done", level="rivet")

        if postprocess_CONTUR:

            self.update_status("Starting postprocess contur", level="rivet")

            set_env = "#!{0}\n".format(misc.which('bash' if misc.get_shell_type() in ['bash',None] else 'tcsh'))
            rivet_path = self.options['rivet_path']
            yoda_path = self.options['yoda_path']
            set_env = set_env + "# RIVET/YODA PATH SETUP\n"
            set_env = set_env + "export PATH={0}:{1}:$PATH\n"\
                                             .format(pjoin(rivet_path, 'bin'),\
                                                     pjoin(yoda_path, 'bin'))
            set_env = set_env + "export LD_LIBRARY_PATH={0}:{1}:{2}:{3}:$LD_LIBRARY_PATH\n"\
                                                        .format(pjoin(rivet_path, 'lib'),\
                                                                pjoin(rivet_path, 'lib64'),\
                                                                pjoin(yoda_path, 'lib'),\
                                                                pjoin(yoda_path, 'lib64'))
            major, minor = sys.version_info[0:2]
            set_env = set_env + "export PYTHONPATH={0}:{1}:{2}:{3}:$PYTHONPATH\n\n"\
                                                   .format(pjoin(rivet_path, 'lib', 'python%s.%s' %(major,minor), 'site-packages'),\
                                                           pjoin(rivet_path, 'lib64', 'python%s.%s' %(major,minor), 'site-packages'),\
                                                           pjoin(yoda_path, 'lib', 'python%s.%s' %(major,minor), 'site-packages'),\
                                                           pjoin(yoda_path, 'lib64', 'python%s.%s' %(major,minor), 'site-packages'))

            contur_path = self.options['contur_path']
            set_env = set_env + "# CONTUR PATH SETUP\n"
            set_env = set_env + "export PATH={0}:$PATH\n".format(pjoin(contur_path, 'python%s.%s' %(major,minor), 'bin'))
            set_env = set_env + "export PYTHONPATH={0}:$PYTHONPATH\n".format(pjoin(contur_path, 'python%s.%s' %(major,minor)))

            set_env = set_env + "source {0} >> contur.log 2>&1\n\n".format(pjoin(contur_path, "contur", "setupContur.sh"))

            os.system("mkdir -p {0}".format(pjoin(self.me_dir, 'Analysis', 'contur')))

            if nb_rivet == 1:
                this_yoda_file = pjoin(run_dirs[0], "rivet_result.yoda")
                os.system("ln -s {0} {1}".format(this_yoda_file, pjoin(self.me_dir, 'Analysis', 'contur', 'rivet_result.yoda')))
                if not rivet_config["weight_name"] == "None":
                    contur_cmd = 'contur --wn "{0}" {1}\n'.format(rivet_config["weight_name"], pjoin(self.me_dir, 'Analysis', 'contur', 'rivet_result.yoda'))
                else:
                    contur_cmd = 'contur {0}\n'.format(pjoin(self.me_dir, 'Analysis', 'contur', 'rivet_result.yoda'))
            else:
                # Link yoda and params files inside analysis/contur/scan directory
                scan_subdirs = []
                for i_rivet in range(nb_rivet):
                    this_scan_dir = pjoin(self.me_dir, 'Analysis', 'contur', 'scan', rivet_config["contur_ra"])
                    os.system("mkdir -p {0}".format(this_scan_dir))

                    this_scan_subdir = pjoin(this_scan_dir, str(i_rivet+1).zfill(4))
                    scan_subdirs.append(this_scan_subdir)
                    os.mkdir(this_scan_subdir)

                    this_yoda_file = pjoin(run_dirs[i_rivet], "rivet_result.yoda")
                    this_param_file = pjoin(run_dirs[i_rivet], "params.dat")
                    os.system("ln -s {0} {1}".format(this_yoda_file, pjoin(this_scan_subdir, "runpoint_"+str(i_rivet+1).zfill(4)+".yoda")))
                    os.system("ln -s {0} {1}".format(this_param_file, pjoin(this_scan_subdir, "params.dat")))

                    if rivet_config['xaxis_relvar'] or rivet_config['yaxis_relvar']:
                        f_params = open(pjoin(run_dirs[i_rivet], "params.dat"))
                        f_relparams = open(pjoin(run_dirs[i_rivet], "params_replace.dat"), "w")
                        rivet_config.setRelevantParamCard(f_params=f_params,f_relparams=f_relparams)
                        f_params.close()
                        f_relparams.close()

                        files.mv(pjoin(run_dirs[i_rivet], "params_replace.dat"), pjoin(run_dirs[i_rivet], "params.dat"))

                contur_add = ""
                if not (rivet_config["contur_add"] == "default" or rivet_config["contur_add"]  == None):
                    contur_add = " " + rivet_config["contur_add"]

                if rivet_config["weight_name"] == "None":
                    contur_cmd = 'contur --nomultip -g scan >> contur.log 2>&1\n'
                else:
                    contur_cmd = 'contur --nomultip -g scan --wn "{0}" >> contur.log 2>&1\n'.format(rivet_config["weight_name"] + contur_add)

                if rivet_config["draw_contur_heatmap"]:

                    axis_log = ""
                    if rivet_config["xaxis_log"]:
                        axis_log = axis_log + " --xlog"
                    if rivet_config["yaxis_log"]:
                        axis_log = axis_log + " --ylog"

                    axis_label = ""
                    if rivet_config["xaxis_label"]:
                        axis_label = axis_label + " -x " + rivet_config["xaxis_label"]
                    if rivet_config["yaxis_label"]:
                        axis_label = axis_label + " -y " + rivet_config["yaxis_label"]

                    if rivet_config["xaxis_relvar"]:
                        if rivet_config["xaxis_label"]:
                            xaxis_var = rivet_config["xaxis_label"]
                        else:
                            xaxis_var = "xaxis_relvar"
                    else:
                        xaxis_var = rivet_config["xaxis_var"]
                    if rivet_config["yaxis_relvar"]:
                        if rivet_config["yaxis_label"]:
                            yaxis_var = rivet_config["yaxis_label"]
                        else:
                            yaxis_var = "yaxis_relvar"
                    else:
                        yaxis_var = rivet_config["yaxis_var"]

                    contur_cmd = contur_cmd + 'contur-plot ANALYSIS/contur.map {0} {1} {2} {3}' \
                                                        .format(xaxis_var, yaxis_var,axis_label, axis_log)

            wrapper = open(pjoin(self.me_dir, "Analysis", "contur", "run_contur.sh"), "w")
            wrapper.write(set_env)
 
            wrapper.write('{0}\n'.format(contur_cmd))
            wrapper.close()
 
            misc.call(["run_contur.sh"], cwd=(pjoin(self.me_dir, "Analysis", "contur")))

            logger.info("Contur outputs are stored in {0}".format(pjoin(self.me_dir, "Analysis", "contur","conturPlot")))
            self.update_status("postprocessing contur done", level="rivet")

    # this decorator handle the loop related to scan.
    @common_run.scanparamcardhandling(run_card_scan=True)
    def run_generate_events(self, switch_mode, args):

        if self.proc_characteristics['loop_induced'] and self.options['run_mode']==0:
            # Also the single core mode is not supported for loop-induced.
            # We therefore emulate it with multi-core mode with one core
            logger.warning(
"""Single-core mode not supported for loop-induced processes.
Beware that MG5aMC now changes your runtime options to a multi-core mode with only one active core.""")
            self.do_set('run_mode 2')
            self.do_set('nb_core 1')

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
            with misc.TMP_variable(self, 'run_name', self.run_name):
                self.exec_cmd('decay_events -from_cards', postcmd=False)
            self.exec_cmd('create_gridpack', postcmd=False)
        else:
            # Regular run mode
            logger.info('Generating %s events with run name %s' %
                        (self.run_card['nevents'], self.run_name))
            self.exec_cmd('survey  %s %s' % (self.run_name,' '.join(args)),
                          postcmd=False)
            nb_event = self.run_card['nevents']
            bypass_run=False
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
                if not self.param_card_iterator:
                    raise ZeroResult('See https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/FAQ-General-14')
                else:
                    bypass_run = True
            
            #we can bypass the following if scan and first result is zero
            if not bypass_run:
                self.exec_cmd('refine %s --treshold=%s' % (nb_event,self.run_card['second_refine_treshold'])
                              , postcmd=False)
            
                self.exec_cmd('combine_events', postcmd=False,printcmd=False)
                self.print_results_in_shell(self.results.current)

                if self.run_card['use_syst']:
                    if self.run_card['systematics_program'] == 'auto':                        
                        scdir = self.options['syscalc_path']
                        if not scdir or not os.path.exists(scdir):
                            to_use = 'systematics'
                        else:
                            to_use = 'syscalc'
                    elif self.run_card['systematics_program'].lower() in ['systematics','syscalc', 'none']:
                        to_use = self.run_card['systematics_program']
                    else:
                        logger.critical('Unvalid options for systematics_program: bypass computation of systematics variations.')
                        to_use = 'none'
                        
                    if to_use == 'systematics':
                        if self.run_card['systematics_arguments'] != ['']:
                            self.exec_cmd('systematics %s %s ' % (self.run_name,
                                          ' '.join(self.run_card['systematics_arguments'])),                  
                                          postcmd=False, printcmd=False)
                        else:
                            self.exec_cmd('systematics %s --from_card' % self.run_name,
                                           postcmd=False,printcmd=False)    
                    elif to_use == 'syscalc':
                        self.run_syscalc('parton')
                
                    
                self.create_plot('parton')            
                self.exec_cmd('store_events', postcmd=False) 
                if self.run_card['boost_event'].strip()  and self.run_card['boost_event'] != 'False':
                    self.boost_events()
                            
                                       
                self.exec_cmd('reweight -from_cards', postcmd=False)            
                self.exec_cmd('decay_events -from_cards', postcmd=False)
                if self.run_card['time_of_flight']>=0:
                    self.exec_cmd("add_time_of_flight --threshold=%s" % self.run_card['time_of_flight'] ,postcmd=False)

                if switch_mode['analysis'] == 'ExRoot':
                    input = pjoin(self.me_dir, 'Events', self.run_name,'unweighted_events.lhe.gz')
                    output = pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.root')
                    self.create_root_file(input , output)

                self.exec_cmd('madanalysis5_parton --no_default', postcmd=False, printcmd=False)
                # shower launches pgs/delphes if needed    
                self.exec_cmd('shower --no_default', postcmd=False, printcmd=False)
                self.exec_cmd('madanalysis5_hadron --no_default', postcmd=False, printcmd=False)
                self.exec_cmd('rivet --no_default', postcmd=False, printcmd=False)
                self.store_result()
                        
            if self.allow_notification_center:    
                misc.system_notify('Run %s finished' % os.path.basename(self.me_dir), 
                              '%s: %s +- %s ' % (self.results.current['run_name'], 
                                                 self.results.current['cross'],
                                                 self.results.current['error']))
    
    def boost_events(self):
        
        if not self.run_card['boost_event']:
            return
        
        if self.run_card['boost_event'].startswith('lambda'):
            if not isinstance(self, cmd.CmdShell):
                raise Exception("boost not allowed online")
            filter = eval(self.run_card['boost_event'])
        else:
            raise Exception
            
        path = [pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe.gz'),
                    pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe'),
                    pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe.gz'),
                    pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe')]
            
        for p in path:
            if os.path.exists(p):
                event_path = p
                break
        else:
            raise Exception("fail to find event file for the boost")
            
            
        lhe = lhe_parser.EventFile(event_path)
        with misc.TMP_directory() as tmp_dir:
            output =  lhe_parser.EventFile(pjoin(tmp_dir, os.path.basename(event_path)), 'w')
            #write the banner to the output file
            output.write(lhe.banner)
            # Loop over all events
            for event in lhe:
                event.boost(filter)
                #write this modify event
                output.write(str(event))
            output.write('</LesHouchesEvent>\n') 
            lhe.close()
            files.mv(pjoin(tmp_dir, os.path.basename(event_path)), event_path) 
         
            
            
            
    
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
            for filter in misc.glob('*Filter*', 
                       pjoin(self.me_dir,'SubProcesses','MadLoop5_resources')):
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
        #    self.do_calculate_decay_widths(line, *args, **opt)
        #else:
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

        if data['run_mode']=='madevent':
            if data['cross_pythia'] and data['nb_event_pythia']:
                if data['cross_pythia'] == -1:
                    path = pjoin(self.me_dir, 'Events', self.run_name, '%s_merged_xsecs.txt' % self.run_tag)
                    cross_sections = {}
                    if os.path.exists(path):
                        for line in open(path):
                            split = line.split()
                            if len(split)!=3:
                                continue
                            scale, cross, error = split
                            cross_sections[float(scale)] = (float(cross), float(error))   
                    if len(cross_sections)>0:
                        logger.info('     Pythia8 merged cross-sections are:')
                        for scale in sorted(cross_sections.keys()):
                            logger.info('      > Merging scale = %-6.4g : %-11.5g +/- %-7.2g [pb]'%\
                                        (scale,cross_sections[scale][0],cross_sections[scale][1]))
                    
                else:
                    if self.ninitial == 1:
                        logger.info("     Matched width :   %.4g +- %.4g GeV" % (data['cross_pythia'], data['error_pythia']))
                    else:
                        logger.info("     Matched cross-section :   %.4g +- %.4g pb" % (data['cross_pythia'], data['error_pythia']))            
                        logger.info("     Nb of events after matching/merging :  %d" % int(data['nb_event_pythia']))
                if self.run_card['use_syst'] in self.true and \
                   (int(self.run_card['ickkw'])==1 or self.run_card['ktdurham']>0.0
                                                    or self.run_card['ptlund']>0.0):
                    logger.info("     Notice that because Systematics computation is turned on, the merging did not veto events but modified their weights instead.\n"+\
                                "     The resulting hepmc/stdhep file should therefore be use with those weights.")
                else:
                    logger.info("     Nb of events after merging :  %s" % data['nb_event_pythia'])

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
        self.Gdirs = None

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
        comment = collections.defaultdict(str)
        line_number = 0
        # Read and remove all decays from the param_card                     
        while line_number < len(param_card):
            line = param_card[line_number]
            if line.lower().startswith('decay'):
                # Read decay if particle in decay_info 
                # DECAY  6   1.455100e+00                                    
                line = param_card.pop(line_number)
                line = line.split()
                comment[int(line[1])] = ' '.join(line).split('#',1)[1] if '#' in line else ''
                particle = 0
                if int(line[1]) not in decay_info:
                    try: # If formatting is wrong, don't want this particle
                        particle = int(line[1])
                        width = float(line[2])
                    except Exception:
                        particle = 0
                # Read BRs for this decay
                line = param_card[line_number]
                while re.search(r'^(#|\s|\d)', line):
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
            param_card.append("DECAY  %i   %e # %s" % (key, width.real, comment[int(key)]))
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

        self.check_survey(args, cmd='multi_run')
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
            path = '%s/%s/unweighted_events.lhe.gz' % (pjoin(self.me_dir,'Events'), self.run_name)

            if os.path.exists(path):
                misc.gunzip(path)

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
        
        cpath = pjoin(self.me_dir,'Cards','param_card.dat')
        if param_card_iterator:

            param_card_iterator.store_entry(self.run_name, self.results.current['cross'],param_card_path=cpath)
            #check if the param_card defines a scan.
            orig_name=self.run_name
            for card in param_card_iterator:
                card.write(cpath)
                self.exec_cmd("multi_run %s -f " % nb_run ,precmd=True, postcmd=True,errorhandling=False)
                param_card_iterator.store_entry(self.run_name, self.results.current['cross'], param_card_path=cpath)
            param_card_iterator.write(pjoin(self.me_dir,'Cards','param_card.dat'))
            scan_name = misc.get_scan_name(orig_name, self.run_name)
            path = pjoin(self.me_dir, 'Events','scan_%s.txt' % scan_name)
            logger.info("write all cross-section results in %s" % path, '$MG:BOLD')
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
                run_card = banner_mod.RunCard(opt['run_card'], path=pjoin(self.me_dir, 'Cards', 'run_card.dat'))
            else:
                run_card = self.run_card
            self.run_card = run_card
            if self.cluster:
                self.cluster.modify_interface(self)
            if self.ninitial == 1:
                run_card['lpp1'] =  0
                run_card['lpp2'] =  0
                run_card['ebeam1'] = 0
                run_card['ebeam2'] = 0
                
            # Ensure that the bias parameters has all the required input from the
            # run_card
            if run_card['bias_module'].lower() not in ['dummy','none']:
                # Using basename here means that the module will not be overwritten if already existing.
                bias_module_path = pjoin(self.me_dir,'Source','BIAS',
                                         os.path.basename(run_card['bias_module']))
                if not os.path.isdir(bias_module_path):
                    if not os.path.isdir(run_card['bias_module']):
                        raise InvalidCmd("The bias module at '%s' cannot be found."%run_card['bias_module'])
                    else:
                        for mandatory_file in ['makefile','%s.f'%os.path.basename(run_card['bias_module'])]:
                            if not os.path.isfile(pjoin(run_card['bias_module'],mandatory_file)):
                                raise InvalidCmd("Could not find the mandatory file '%s' in bias module '%s'."%(
                                                                         mandatory_file,run_card['bias_module']))
                        misc.copytree(run_card['bias_module'], pjoin(self.me_dir,'Source','BIAS',
                                                                     os.path.basename(run_card['bias_module'])))
                
                #check expected parameters for the module.
                default_bias_parameters = {}
                start, last = False,False
                for line in open(pjoin(bias_module_path,'%s.f'%os.path.basename(bias_module_path))):
                    if start and last:
                        break
                    if not start and not re.search(r'c\s*parameters\s*=\s*{',line, re.I):
                        continue
                    start = True
                    if not line.startswith('C'):
                        continue
                    line = line[1:]
                    if '{' in line:
                        line = line.split('{')[-1]
                    # split for } ! #
                    split_result = re.split(r'(\}|!|\#)', line,1, re.M)
                    line = split_result[0]
                    sep = split_result[1] if len(split_result)>1 else None
                    if sep == '}':
                        last = True
                    if ',' in line:
                        for pair in line.split(','):
                            if not pair.strip():
                                continue
                            x,y =pair.split(':') 
                            x=x.strip()
                            if x.startswith(('"',"'")) and x.endswith(x[0]):
                                x = x[1:-1] 
                            default_bias_parameters[x] = y
                    elif ':' in line:
                        x,y = line.split(':')
                        x = x.strip()
                        if x.startswith(('"',"'")) and x.endswith(x[0]):
                            x = x[1:-1] 
                        default_bias_parameters[x] = y
                for key,value in run_card['bias_parameters'].items():
                    if key not in default_bias_parameters:
                        logger.warning('%s not supported by the bias module. We discard this entry.', key)
                    else:
                        default_bias_parameters[key] = value
                run_card['bias_parameters'] = default_bias_parameters  
              
              
            # Finally write the include file          
            run_card.write_include_file(opt['output_dir'])
        

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
            self.MadLoopparam.set('WriteOutFilters',False, changeifuserset=False)
            
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
#            self.MadLoopparam.set('CTModeInit',4, changeifuserset=False)
            # Consequently, we can allow for a finer threshold for vanishing
            # helicity configuration
#            self.MadLoopparam.set('ZeroThres',1.0e-11, changeifuserset=False)

#           It is a bit superficial to use the level 2 which tries to numerically
#           map matching helicities (because of CP symmetry typically) together.
#           It is useless in the context of MC over helicities and it can 
#           potentially make the helicity double checking fail.
            self.MadLoopparam.set('HelicityFilterLevel',1, changeifuserset=False)

#           To be on the safe side however, we ask for 4 consecutive matching
#           helicity filters.
            self.MadLoopparam.set('CheckCycle',4, changeifuserset=False)
            
            # For now it is tricky to have each channel performing the helicity
            # double check. What we will end up doing is probably some kind
            # of new initialization round at the beginning of each launch
            # command, to reset the filters.    
            self.MadLoopparam.set('DoubleCheckHelicityFilter',False,
                                                             changeifuserset=False)
          
            # Thanks to TIR recycling, TIR is typically much faster for Loop-induced
            # processes when not doing MC over helicities, so that we place OPP last.
            if not hasattr(self, 'run_card'):
                run_card = banner_mod.RunCard(opt['run_card'])
            else:
                run_card = self.run_card
            if run_card['nhel'] == 0:
                if 'MLReductionLib' in self.MadLoopparam.user_set and \
                    (self.MadLoopparam.get('MLReductionLib').startswith('1') or
                     self.MadLoopparam.get('MLReductionLib').startswith('6')):
                    logger.warning(
    """You chose to set the preferred reduction technique in MadLoop to be OPP (see parameter MLReductionLib).
    Beware that this can bring significant slowdown; the optimal choice --when not MC over helicity-- being to first start with TIR reduction.""")
                # We do not include GOLEM for now since it cannot recycle TIR coefs yet.
                self.MadLoopparam.set('MLReductionLib','7|6|1', changeifuserset=False)
            else:
                if 'MLReductionLib' in self.MadLoopparam.user_set and \
                    not (self.MadLoopparam.get('MLReductionLib').startswith('1') or
                         self.MadLoopparam.get('MLReductionLib').startswith('6')):
                    logger.warning(
    """You chose to set the preferred reduction technique in MadLoop to be different than OPP (see parameter MLReductionLib).
    Beware that this can bring significant slowdown; the optimal choice --when MC over helicity-- being to first start with OPP reduction.""")
                self.MadLoopparam.set('MLReductionLib','6|7|1', changeifuserset=False)

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
                self.MadLoopparam.set('NRotations_DP',0,changeifuserset=False)
                self.MadLoopparam.set('NRotations_QP',0,changeifuserset=False)
            else:
                # When MC-ing over helicities, the manual TIR cache clearing is
                # not necessary, so that one can use the lorentz check
                # Using NRotations_DP=1 slows down the code by close to 100%
                # but it is typicaly safer.
                # self.MadLoopparam.set('NRotations_DP',0,changeifuserset=False)
                # Revert to the above to be slightly less robust but twice faster.
                self.MadLoopparam.set('NRotations_DP',1,changeifuserset=False)
                self.MadLoopparam.set('NRotations_QP',0,changeifuserset=False)                
            
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
                self.MadLoopparam.set('MLStabThres',1.0e-7,changeifuserset=False)
            else:
                self.MadLoopparam.set('MLStabThres',1.0e-4,changeifuserset=False)            

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
        elif self.run_card['hard_survey']:
            self.pass_in_difficult_integration_mode(self.run_card['hard_survey'])

        if self.proc_characteristics['hel_recycling'] and self.run_card['hel_recycling']:
            jobs, P_zero_result = ajobcreator.get_helicity()
        else:
            for p in subproc:
                for f in misc.glob('matrix*_orig.f', pjoin(self.me_dir, 'SubProcesses', p)):
                    new_file = f.replace('_orig','_optim')
                    files.cp(f, f.replace('_orig','_optim'))
                    f = '%s.o' % f[:-2]
                    if os.path.exists(f):
                        files.cp(f, f.replace('_orig','_optim'))
            try:
                os.remove(pjoin(self.me_dir, 'SubProcesses', p, 'Hel', 'selection'))
            except Exception as error:
                logger.debug(error)
                pass
                                
        jobs, P_zero_result = ajobcreator.launch()
        # Check if all or only some fails
        if P_zero_result:
            if len(P_zero_result) == len(subproc):
                Pdir = pjoin(self.me_dir, 'SubProcesses',subproc[0].strip())
                raise ZeroResult('%s' % \
                    open(pjoin(Pdir,'ajob.no_ps.log')).read())
            else:
                logger.warning(''' %s SubProcesses doesn\'t have available phase-space.
            Please check mass spectrum.''' % ','.join(P_zero_result))
            self.get_Gdir()
            for P in P_zero_result:
                self.Gdirs[0][pjoin(self.me_dir,'SubProcesses',P)] = []
        
        self.monitor(run_type='All jobs submitted for survey', html=True)
        if not self.history or 'survey' in self.history[-1] or self.ninitial ==1  or \
           self.run_card['gridpack']:
            #will be done during the refine (more precisely in gen_ximprove)
            cross, error = self.make_make_all_html_results()
            self.results.add_detail('cross', cross)
            self.results.add_detail('error', error)  
            self.exec_cmd("print_results %s" % self.run_name,
                           errorhandling=False, printcmd=False, precmd=False, postcmd=False)      
        
        self.results.add_detail('run_statistics', dict(ajobcreator.run_statistics))
        self.update_status('End survey', 'parton', makehtml=False)

    ############################################################################
    def pass_in_difficult_integration_mode(self, rate=1):
        """be more secure for the integration to not miss it due to strong cut"""
        
        # improve survey options if default
        if self.opts['points'] == self._survey_options['points'][1]:
            self.opts['points'] = (rate+2) * self._survey_options['points'][1]
        if self.opts['iterations'] == self._survey_options['iterations'][1]:
            self.opts['iterations'] = 1 + rate + self._survey_options['iterations'][1]
        if self.opts['accuracy'] == self._survey_options['accuracy'][1]:
            self.opts['accuracy'] = self._survey_options['accuracy'][1]/(rate+2)  
        
        # Modify run_config.inc in order to improve the refine
        conf_path = pjoin(self.me_dir, 'Source','run_config.inc')
        files.cp(conf_path, conf_path + '.bk')
        #
        text = open(conf_path).read()
        min_evt, max_evt = 2500 *(2+rate), 10000*(rate+1) 
        
        text = re.sub(r'''\(min_events = \d+\)''', '(min_events = %i )' % min_evt, text)
        text = re.sub(r'''\(max_events = \d+\)''', '(max_events = %i )' % max_evt, text)
        fsock = open(conf_path, 'w')
        fsock.write(text)
        fsock.close()
        
        # Compile
        for name in ['../bin/internal/gen_ximprove', 'all']:
            self.compile(arg=[name], cwd=os.path.join(self.me_dir, 'Source'))
        
        
    ############################################################################      
    def do_refine(self, line):
        """Advanced commands: launch survey for the current process """
        devnull = open(os.devnull, 'w')  
        self.nb_refine += 1
        args = self.split_arg(line)
        treshold=None
               

        
        for a in args:
            if a.startswith('--treshold='):
                treshold = float(a.split('=',1)[1])
                old_xsec = self.results.current['prev_cross']
                new_xsec = self.results.current['cross']
                if old_xsec > new_xsec * treshold:
                    logger.info('No need for second refine due to stability of cross-section')
                    return
                else:
                    args.remove(a)
                    break
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
            for match in misc.glob('*ajob*', Pdir):
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
            logger.info("- Current estimate of cross-section: %s +- %s" % (cross, error))
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
                    
                    alljobs = misc.glob('ajob*', Pdir)
                    
                    #remove associated results.dat (ensure to not mix with all data)
                    Gre = re.compile(r"\s*j=(G[\d\.\w]+)")
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
            
        cross, error, across = self.make_make_all_html_results(get_attr=('xsec','xerru','axsec'))
        
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)
        self.results.add_detail('axsec', across)

        self.results.add_detail('run_statistics', 
                                dict(self.results.get_detail('run_statistics')))

        self.update_status('finish refine', 'parton', makehtml=False)
        devnull.close()
    
    ############################################################################ 
    def do_comine_iteration(self, line):
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
        start=time.time()
        args = self.split_arg(line)
        start = time.time()
        # Check argument's validity
        self.check_combine_events(args)
        self.update_status('Combining Events', level='parton')

        
        if self.run_card['gridpack'] and isinstance(self, GridPackCmd):
            return GridPackCmd.do_combine_events(self, line)

    
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
        

        if self.run_card['gridpack']:
            return 
        
        
        get_wgt = lambda event: event.wgt            
        AllEvent = lhe_parser.MultiEventFile()
        AllEvent.banner = self.banner
        
        partials = 0 # if too many file make some partial unweighting
        sum_xsec, sum_xerru, sum_axsec = 0,[],0
        Gdirs = self.get_Gdir()
        Gdirs.sort()
        partials_info = []
        try:
            p = subprocess.Popen(["ulimit", "-n"], stdout=subprocess.PIPE)
            out, err = p.communicate()
            max_G = out.decode()
            if max_G == "unlimited":
                max_G =2500
            else:
                max_G = int(max_G) - 40
        except Exception as  error:
            logger.debug(error)
            max_G = 80 # max(20, len(Gdirs)/self.options['nb_core'])

        if not hasattr(self,'proc_characteristic'):
            self.proc_characteristic = self.get_characteristics()
        mycluster = cluster.MultiCore(nb_core=self.options['nb_core'])

        def split(a, n):
            """split a list "a" into n chunk of same size (or nearly same size)"""
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

        partials_info = [] 
        if len(Gdirs) >= max_G:
            start_unweight= time.perf_counter()
            # first check in how many chunk we have to split (always use a multiple of nb_core)
            nb_split = 1
            nb_G = len(Gdirs) // (2* self.options['nb_core'])
            while nb_G > min(80, max_G):
               nb_split += 1 
               nb_G = len(Gdirs)//(nb_split*2*self.options['nb_core'])
               if nb_G < 10:
                   nb_split -=1
                   nb_G = len(Gdirs)//(nb_split*2*self.options['nb_core']) 
            
            #enforce at least 10 directory per thread
            if nb_G > 10 or nb_split>1: 
                # do the unweighting of each chunk on their own thread
                nb_chunk = (nb_split*2*self.options['nb_core'])
            else:
                nb_chunk = len(Gdirs) // 10 
                nb_G =10
            
            # security that the number of combine events is too large
            if nb_chunk >= max_G:
                nb_chunk = max_G -1
                nb_G = len(Gdirs) // nb_chunk 

            for i, local_G in enumerate(split(Gdirs, nb_chunk)):
                line = [pjoin(self.me_dir, "Events", self.run_name, "partials%d.lhe.gz" % i)]
                line.append(pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, tag)))
                line.append(str(self.results.current.get('axsec')))
                line += local_G
                partials_info.append(self.do_combine_events_partial(' '.join(line), preprocess_only=True))
                mycluster.submit(sys.executable, 
                    [pjoin(self.me_dir, 'bin', 'internal', 'madevent_interface.py'), 'combine_events_partial'] + line,
                    stdout='/dev/null'
                )
                
            starttime = time.time()
            update_status = lambda idle, run, finish: \
                    self.update_status((idle, run, finish, 'unweight'), level=None,
                                       force=False, starttime=starttime)
            mycluster.wait(self.me_dir, update_status)
            # do the final combination
            for data in partials_info:
                AllEvent.add(*data)
            
            start_unweight= time.perf_counter()
            nb_event = AllEvent.unweight(pjoin(self.me_dir, "Events", self.run_name, "unweighted_events.lhe.gz"),
                          get_wgt, trunc_error=1e-2, event_target=self.run_card['nevents'],
                          log_level=logging.DEBUG, normalization=self.run_card['event_norm'],
                          proc_charac=self.proc_characteristic)
            
            #cleaning
            for data in partials_info:
                path = data[0]
                try:
                    os.remove(path)
                except Exception as error:
                    try: 
                        os.remove(path[:-3]) # try without the .gz
                    except:
                        misc.sprint('no file ', path, 'to clean')
        else:
            for Gdir in Gdirs:
                if os.path.exists(pjoin(Gdir, 'events.lhe')):
                    result = sum_html.OneResult('')
                    result.read_results(pjoin(Gdir, 'results.dat'))
                    sum_xsec += result.get('xsec')
                    sum_xerru.append(result.get('xerru'))
                    sum_axsec += result.get('axsec')

                    if self.run_card['gridpack'] or self.run_card['nevents']==0:
                        os.remove(pjoin(Gdir, 'events.lhe'))
                        continue

                    AllEvent.add(pjoin(Gdir, 'events.lhe'), 
                                result.get('xsec'),
                                result.get('xerru'),
                                result.get('axsec')
                                )
                    
            if len(AllEvent) == 0:
                nb_event = 0 
            else:
                nb_event = AllEvent.unweight(pjoin(self.me_dir, "Events", self.run_name, "unweighted_events.lhe.gz"),
                                get_wgt, trunc_error=1e-2, event_target=self.run_card['nevents'],
                                log_level=logging.DEBUG, normalization=self.run_card['event_norm'],
                                proc_charac=self.proc_characteristic)

        if nb_event < self.run_card['nevents']:
            logger.warning("failed to generate enough events. Please follow one of the following suggestions to fix the issue:")
            logger.warning("  - set in the run_card.dat 'sde_strategy' to %s", 1 + self.run_card['sde_strategy'] % 2)
            logger.warning("  - set in the run_card.dat  'hard_survey' to 1 or 2.")
            logger.warning("  - reduce the number of requested events (if set too high)")
            logger.warning("  - check that you do not have -integrable- singularity in your amplitude.")
            logger.warning("  - regenerate your process directory by selecting another gauge (in particular try FD gauge).")


                   
        self.results.add_detail('nb_event', nb_event)
    
        if self.run_card['bias_module'].lower() not in  ['dummy', 'none'] and nb_event:
            self.correct_bias()
        elif self.run_card['custom_fcts']:
            self.correct_bias()
        logger.info("combination of events done in %s s ", time.time()-start)
        
        self.to_store.append('event')

    ############################################################################ 
    def do_combine_events_partial(self, line, preprocess_only=False):
        """ """

    
        AllEvent = lhe_parser.MultiEventFile()

        sum_xsec, sum_xerru, sum_axsec = 0,[],0
        output, banner_path,cross = line.split()[:3]
        Gdirs = line.split()[3:]

        cross = float(cross)
        if not self.banner:
            self.banner = banner_mod.Banner(banner_path)
        if not hasattr(self, 'run_card'):
            self.run_card = banner_mod.RunCard(self.banner['mgruncard'])
        AllEvent.banner = self.banner

        for Gdir in Gdirs:
            if os.path.exists(pjoin(Gdir, 'events.lhe')):
                result = sum_html.OneResult('')
                result.read_results(pjoin(Gdir, 'results.dat'))
                sum_xsec += result.get('xsec')
                sum_xerru.append(result.get('xerru'))
                sum_axsec += result.get('axsec')

                if self.run_card['gridpack'] or self.run_card['nevents']==0:
                    os.remove(pjoin(Gdir, 'events.lhe'))
                    continue
                if not preprocess_only:
                    AllEvent.add(pjoin(Gdir, 'events.lhe'), 
                             result.get('xsec'),
                             result.get('xerru'),
                             result.get('axsec')
                             ) 
        if preprocess_only:
            return output, sum_xsec, math.sqrt(sum(x**2 for x in sum_xerru)), sum_axsec
        nb_event = max(min(abs(1.01*self.run_card['nevents']*sum_axsec/cross),self.run_card['nevents']), 10)
        get_wgt = lambda event: event.wgt   
        AllEvent.unweight(output,
                          get_wgt, log_level=5,  trunc_error=1e-2, event_target=nb_event)  
        return output, sum_xsec, math.sqrt(sum(x**2 for x in sum_xerru)), sum_axsec

    ############################################################################ 
    def correct_bias(self):
        """check the first event and correct the weight by the bias 
           and correct the cross-section.
           If the event do not have the bias tag it means that the bias is 
           one modifying the cross-section/shape so we have nothing to do
        """

        lhe = lhe_parser.EventFile(pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe.gz'))
        init = False
        cross = collections.defaultdict(float)
        nb_event = 0
        for event in lhe:
            rwgt_info = event.parse_reweight()
            if not init:
                if 'bias' in rwgt_info:
                    output = lhe_parser.EventFile(pjoin(self.me_dir, 'Events', self.run_name, '.unweighted_events.lhe.tmp.gz'),'w')
                    #output.write(lhe.banner)
                    init = True
                else:
                    return
            #change the weight
            event.wgt /= rwgt_info['bias']
            #remove the bias info
            del event.reweight_data['bias']
            # compute the new cross-section
            cross[event.ievent] += event.wgt
            nb_event +=1
            output.write(str(event))
        output.write('</LesHouchesEvents>')
        output.close()
        lhe.close()
                
        # MODIFY THE BANNER i.e. INIT BLOCK
        # ensure information compatible with normalisation choice
        total_cross = sum(cross[key] for key in cross)
        if 'event_norm' in self.run_card: # if not this is "sum"
            if self.run_card['event_norm'] == 'average':
                total_cross = total_cross / nb_event
                for key in cross:
                    cross[key] /= nb_event
            elif self.run_card['event_norm'] == 'unity':
                total_cross = self.results.current['cross'] * total_cross / nb_event
                for key in cross:
                    cross[key] *= total_cross / nb_event              
                
        bannerfile = lhe_parser.EventFile(pjoin(self.me_dir, 'Events', self.run_name, '.banner.tmp.gz'),'w')
        banner = banner_mod.Banner(lhe.banner)
        banner.modify_init_cross(cross)
        banner.set_lha_strategy(-4)
        banner.write(bannerfile, close_tag=False)
        bannerfile.close()
        # replace the lhe file by the new one
        if lhe.name.endswith('.gz'):
            os.system('cat %s %s > %s' %(bannerfile.name, output.name, lhe.name))
        else:
            os.system('cat %s %s > %s.gz' %(bannerfile.name, output.name, lhe.name))
            os.remove(lhe.name)
        os.remove(bannerfile.name)
        os.remove(output.name)
        
                
        self.results.current['cross'] = total_cross
        self.results.current['error'] = 0
         
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
        #input = pjoin(self.me_dir, 'SubProcesses', 'results.dat')
        #output = pjoin(self.me_dir, 'SubProcesses', '%s_results.dat' % run)
        #files.cp(input, output) 


        # 2) Treat the files present in the P directory
        # Ensure that the number of events is different of 0 
        if self.results.current['nb_event'] == 0 and not self.run_card['gridpack']:
            logger.warning("No event detected. No cleaning performed! This should allow to run:\n" +
                           "    cd Subprocesses; ../bin/internal/combine_events\n"+
                           "  to have your events if those one are missing.")
        else:
            for G_path in self.get_Gdir():
                try:
                    # Remove events file (if present)
                    if os.path.exists(pjoin(G_path, 'events.lhe')):
                        os.remove(pjoin(G_path, 'events.lhe'))
                except Exception:
                    continue
                #try:
                #    # Store results.dat
                #    if os.path.exists(pjoin(G_path, 'results.dat')):
                #        input = pjoin(G_path, 'results.dat')
                #        output = pjoin(G_path, '%s_results.dat' % run)
                #        files.cp(input, output) 
                #except Exception:
                #    continue                    
                # Store log
                input = pjoin(G_path, 'log.txt')
                if os.path.exists(input): 
                    if self.run_card['keep_log'] not in ["none", "minimal"]:
                        output = pjoin(G_path, '%s_log.txt' % run)
                        try:
                            files.mv(input, output) 
                        except Exception:
                            continue
                    elif self.run_card['keep_log'] == "none":
                        try:
                            os.remove(input)
                        except Exception:
                            continue 
                #try:   
                #    # Grid
                #    for name in ['ftn26']:
                #        if os.path.exists(pjoin(G_path, name)):
                #            if os.path.exists(pjoin(G_path, '%s_%s.gz'%(run,name))):
                #                os.remove(pjoin(G_path, '%s_%s.gz'%(run,name)))
                #            input = pjoin(G_path, name)
                #            output = pjoin(G_path, '%s_%s' % (run,name))
                #            files.mv(input, output)
                #            misc.gzip(pjoin(G_path, output), error=None) 
                #except Exception:
                #    continue
                # Delete ftn25 to ensure reproducible runs
                if os.path.exists(pjoin(G_path, 'ftn25')):
                    os.remove(pjoin(G_path, 'ftn25'))

        # 3) Update the index.html
        self.gen_card_html()

        
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
                if os.path.exists("%s.gz" % foutput):
                    os.remove(foutput)
                else:
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
        
        Gdir = self.get_Gdir()
        Pdir = set([os.path.dirname(G) for G in Gdir])
        for P in Pdir: 
            allG = misc.glob('G*', path=P)
            for G in allG:
                # avoid case where some file starts with G (madgraph5/madgraph4gpu#947)
                if not os.path.isdir(G): 
                    continue
                if pjoin(P, G) not in Gdir:
                    logger.debug('removing %s', pjoin(P,G))
                    shutil.rmtree(pjoin(P,G))
                    
        
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
        os.system("sed -i.bak \"s/\\s*.true.*=.*GridRun/  .false.  =  GridRun/g\" %s/Cards/grid_card.dat" \
                  % self.me_dir)
        self.update_status('gridpack created', level='gridpack')
        
    ############################################################################      
    def do_shower(self, line):
        """launch the shower"""

        args = self.split_arg(line)
        if len(args)>1 and args[0] in self._interfaced_showers:
            chosen_showers = [args.pop(0)]
        elif '--no_default' in line:
            # If '--no_default' was specified in the arguments, then only one 
            # shower will be run, depending on which card is present.
            # but we each of them are called. (each of them check if the file exists)
            chosen_showers = list(self._interfaced_showers)
        else:
            chosen_showers = list(self._interfaced_showers)
            # It is preferable to run only one shower, even if several are available and no
            # specific one has been selected
            shower_priority = ['pythia8','pythia']
            chosen_showers = [sorted(chosen_showers,key=lambda sh:
                shower_priority.index(sh) if sh in shower_priority else len(shower_priority)+1)[0]]
        
        for shower in chosen_showers:
            self.exec_cmd('%s %s'%(shower,' '.join(args)), 
                                                  postcmd=False, printcmd=False)

    def do_madanalysis5_parton(self, line):
        """launch MadAnalysis5 at the parton level."""
        return self.run_madanalysis5(line,mode='parton')

    #===============================================================================
    # Return a warning (if applicable) on the consistency of the current Pythia8
    # and MG5_aMC version specified. It is placed here because it should be accessible
    # from both madgraph5_interface and madevent_interface
    #===============================================================================
    @staticmethod
    def mg5amc_py8_interface_consistency_warning(options):
        """ Check the consistency of the mg5amc_py8_interface installed with
        the current MG5 and Pythia8 versions. """
    
        # All this is only relevant is Pythia8 is interfaced to MG5
        if not options['pythia8_path']:
            return None
        
        if not options['mg5amc_py8_interface_path']:
            return \
    """
    A Pythia8 path is specified via the option 'pythia8_path' but no path for option
    'mg5amc_py8_interface_path' is specified. This means that Pythia8 cannot be used
    leading order simulations with MadEvent.
    Consider installing the MG5_aMC-PY8 interface with the following command:
     MG5_aMC>install mg5amc_py8_interface
    """
       
        mg5amc_py8_interface_path = options['mg5amc_py8_interface_path']
        py8_path                  = options['pythia8_path']
        # If the specified interface path is relative, make it absolut w.r.t MGDIR if
        # avaialble.
        if not MADEVENT:
            mg5amc_py8_interface_path = pjoin(MG5DIR,mg5amc_py8_interface_path)
            py8_path                  = pjoin(MG5DIR,py8_path)

        # Retrieve all the on-install and current versions  
        fsock =  open(pjoin(mg5amc_py8_interface_path, 'MG5AMC_VERSION_ON_INSTALL'))
        MG5_version_on_install = fsock.read().replace('\n','')
        fsock.close()
        if MG5_version_on_install == 'UNSPECIFIED':
            MG5_version_on_install = None
        fsock = open(pjoin(mg5amc_py8_interface_path, 'PYTHIA8_VERSION_ON_INSTALL'))
        PY8_version_on_install = fsock.read().replace('\n','')
        fsock.close()
        MG5_curr_version =misc.get_pkg_info()['version']
        try:
            p = subprocess.Popen(['./get_pythia8_version.py',py8_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                             cwd=mg5amc_py8_interface_path)
            (out, err) = p.communicate()
            out = out.decode(errors='ignore').replace('\n','')
            PY8_curr_version = out
            # In order to test that the version is correctly formed, we try to cast
            # it to a float
            float(out)
        except:
            PY8_curr_version = None
    
        if not MG5_version_on_install is None and not MG5_curr_version is None:
            if MG5_version_on_install != MG5_curr_version:
                return \
    """
    The current version of MG5_aMC (v%s) is different than the one active when
    installing the 'mg5amc_py8_interface_path' (which was MG5aMC v%s). 
    Please consider refreshing the installation of this interface with the command:
     MG5_aMC>install mg5amc_py8_interface
    """%(MG5_curr_version, MG5_version_on_install)
    
        if not PY8_version_on_install is None and not PY8_curr_version is None:
            if PY8_version_on_install != PY8_curr_version:
                return \
    """
    The current version of Pythia8 (v%s) is different than the one active when
    installing the 'mg5amc_py8_interface' tool (which was Pythia8 v%s). 
    Please consider refreshing the installation of this interface with the command:
     MG5_aMC>install mg5amc_py8_interface
    """%(PY8_curr_version,PY8_version_on_install)
    
        return None

    def setup_Pythia8RunAndCard(self, PY8_Card, run_type):
        """ Setup the Pythia8 Run environment and card. In particular all the process and run specific parameters
        of the card are automatically set here. This function returns the path where HEPMC events will be output,
        if any."""
        
        HepMC_event_output = None
        tag = self.run_tag
        
        if self.options ['run_mode'] == 0 :
            if hasattr(self, 'run_card') and PY8_Card['Main:numberOfEvents'] in [0,-1]:
                PY8_Card.userSet('Main:numberOfEvents', self.run_card['nevents'])

        PY8_Card.subruns[0].systemSet('Beams:LHEF',"unweighted_events.lhe.gz")

        hepmc_format = PY8_Card['HEPMCoutput:file'].lower()
        if hepmc_format == "auto":
            hepmc_format = "hepmc.gz"
        elif hepmc_format == "autoremove":
            hepmc_format = "hepmcremove"

        # output format : hepmc/fifo
        if hepmc_format.startswith("hepmc"):

            hepmc_specs = hepmc_format.split('@')
            hepmc_path = pjoin(self.me_dir,'Events', self.run_name, '%s_pythia8_events.hepmc'%tag)

            # In case @ is given (output path)
            if len(hepmc_specs) > 1:
                if os.path.isabs(hepmc_specs[1]):
                    if os.path.exists(hepmc_specs[1]):
                        os.mkdir(pjoin(hepmc_specs[1], self.run_name))
                        self.to_store.append("moveHEPMC@" + pjoin(hepmc_specs[1], self.run_name))
                    else:
                        logger.warning("%s does not exist, using default output path"%hepmc_specs[1])
                else:
                    self.to_store.append("moveHEPMC@" + pjoin(self.me_dir, 'Events', hepmc_specs[1], self.run_name))
                    os.mkdir(pjoin(self.me_dir, 'Events', hepmc_specs[1], self.run_name))

            # Compress if .gz is given
            if hepmc_specs[0].endswith(".gz"):
                if not 'compressHEPMC' in self.to_store:
                    self.to_store.append('compressHEPMC')
            else:
                if 'compressHEPMC' in self.to_store:
                    self.to_store.remove('compressHEPMC')

            # Remove if remove is given
            if hepmc_specs[0].endswith("remove"):
                if not 'removeHEPMC' in self.to_store:
                    self.to_store.append('removeHEPMC')
            else:
                if 'removeHEPMC' in self.to_store:
                    self.to_store.remove('removeHEPMC')

            HepMC_event_output=hepmc_path
            PY8_Card.MadGraphSet('HEPMCoutput:file','%s_pythia8_events.hepmc'%tag, force=True)

        elif hepmc_format.startswith('fifo'):
            fifo_specs = hepmc_format.split('@')
            fifo_path  = None
            if len(fifo_specs)<=1:
                fifo_path = pjoin(self.me_dir,'Events', self.run_name,'PY8.hepmc.fifo')
                if os.path.exists(fifo_path):
                    os.remove(fifo_path)
                misc.mkfifo(fifo_path)
                # Use defaultSet not to overwrite the current userSet status
                PY8_Card.defaultSet('HEPMCoutput:file','PY8.hepmc.fifo')
            else:
                fifo_path = fifo_specs[1]
                if os.path.exists(fifo_path):
                    if stat.S_ISFIFO(os.stat(fifo_path).st_mode):
                        logger.warning('PY8 will be reusing already existing '+
                                         'custom fifo file at:\n  %s'%fifo_path)
                    else:
                        raise InvalidCmd(
"""The fifo path speficied for the PY8 parameter 'HEPMCoutput:file':
   %s
already exists and is not a fifo file."""%fifo_path)
                else:
                    misc.mkfifo(fifo_path)
                # Use defaultSet not to overwrite the current userSet status
                PY8_Card.defaultSet('HEPMCoutput:file',fifo_path)
            HepMC_event_output=fifo_path    
        elif hepmc_format in ['','/dev/null','None']:
            logger.warning('User disabled the HepMC output of Pythia8.')
            HepMC_event_output = None
        else:
            raise InvalidCmd("Unknow HEPMCoutput:file setting, hepmc/hepmc.gz/hepmcremove/fifo")

        # We specify by hand all necessary parameters, so that there is no
        # need to read parameters from the Banner.
        PY8_Card.MadGraphSet('JetMatching:setMad', False)
        if run_type=='MLM':
            # When running MLM make sure that we do not write out the parameter
            # Merging:xxx as this can interfere with the MLM merging in older
            # versions of the driver.
            PY8_Card.vetoParamWriteOut('Merging:TMS')
            PY8_Card.vetoParamWriteOut('Merging:Process')
            PY8_Card.vetoParamWriteOut('Merging:nJetMax')
            # MadGraphSet sets the corresponding value (in system mode)
            # only if it is not already user_set.
            if PY8_Card['JetMatching:qCut']==-1.0:
                PY8_Card.MadGraphSet('JetMatching:qCut',1.5*self.run_card['xqcut'], force=True)
            
            if PY8_Card['JetMatching:qCut']<(1.5*self.run_card['xqcut']):
                logger.error(
    'The MLM merging qCut parameter you chose (%f) is less than '%PY8_Card['JetMatching:qCut']+
    '1.5*xqcut, with xqcut your run_card parameter (=%f).\n'%self.run_card['xqcut']+
    'It would be better/safer to use a larger qCut or a smaller xqcut.')

            # Also make sure to use the shower starting scales specified in the LHE
            # unless the user specified it
            PY8_Card.systemSet('Beams:setProductionScalesFromLHEF',True)

            # Automatically set qWeed to xqcut if not defined by the user.
            if PY8_Card['SysCalc:qWeed']==-1.0:
                PY8_Card.MadGraphSet('SysCalc:qWeed',self.run_card['xqcut'], force=True)

            if PY8_Card['SysCalc:qCutList']=='auto':
                if self.run_card['use_syst']:
                    if self.run_card['sys_matchscale']=='auto':
                        qcut = PY8_Card['JetMatching:qCut']
                        value = [factor*qcut for factor in [0.5,0.75,1.0,1.5,2.0] if\
                                 factor*qcut> 1.5*self.run_card['xqcut'] ]
                        PY8_Card.MadGraphSet('SysCalc:qCutList', value, force=True)
                    else:
                        qCutList = [float(qc) for qc in self.run_card['sys_matchscale'].split()]
                        if PY8_Card['JetMatching:qCut'] not in qCutList:
                            qCutList.append(PY8_Card['JetMatching:qCut'])
                        PY8_Card.MadGraphSet('SysCalc:qCutList', qCutList, force=True)
            

            if PY8_Card['SysCalc:qCutList']!='auto':
                for scale in PY8_Card['SysCalc:qCutList']:
                    if scale<(1.5*self.run_card['xqcut']):
                        logger.error(
            'One of the MLM merging qCut parameter you chose (%f) in the variation list'%scale+\
            " (either via 'SysCalc:qCutList' in the PY8 shower card or "+\
            "'sys_matchscale' in the run_card) is less than 1.5*xqcut, where xqcut is"+
            ' the run_card parameter (=%f)\n'%self.run_card['xqcut']+
            'It would be better/safer to use a larger qCut or a smaller xqcut.')
                
            # Specific MLM settings
            # PY8 should not implement the MLM veto since the driver should do it
            # if merging scale variation is turned on
            if self.run_card['use_syst']:
                # We do no force it here, but it is clear that the user should know what
                # he's doing if he were to force it to True.
                PY8_Card.MadGraphSet('JetMatching:doVeto',False)
            PY8_Card.MadGraphSet('JetMatching:merge',True)
            PY8_Card.MadGraphSet('JetMatching:scheme',1)
            # Use the parameter maxjetflavor for JetMatching:nQmatch which specifies
            # up to which parton must be matched.Merging:nQuarksMerge
            PY8_Card.MadGraphSet('JetMatching:nQmatch',self.run_card['maxjetflavor'])
            # For MLM, a cone radius of 1.0 is to be prefered.
            PY8_Card.MadGraphSet('JetMatching:coneRadius',1.0)
            # And the value of etaj_max is already infinity by default.
            # PY8_Card.MadGraphSet('JetMatching:etaJetMax',1000.0)
            if not hasattr(self,'proc_characteristic'):
                self.proc_characteristic = self.get_characteristics()
            nJetMax = self.proc_characteristic['max_n_matched_jets']
            if PY8_Card['JetMatching:nJetMax'.lower()] == -1:
                logger.info("No user-defined value for Pythia8 parameter "+
            "'JetMatching:nJetMax'. Setting it automatically to %d."%nJetMax)
                PY8_Card.MadGraphSet('JetMatching:nJetMax',nJetMax, force=True)
        # We use the positivity of 'ktdurham' cut as a CKKWl marker.
        elif run_type=='CKKW':

            # Make sure the user correctly filled in the lowest order process to be considered
            if PY8_Card['Merging:Process']=='<set_by_user>':
                raise self.InvalidCmd('When running CKKWl merging, the user must'+
                    " specifiy the option 'Merging:Process' in pythia8_card.dat.\n"+
                    "Read section 'Defining the hard process' of "+\
                    "http://home.thep.lu.se/~torbjorn/pythia81html/CKKWLMerging.html for more information.")

            # When running CKKWL make sure that we do not write out the parameter
            # JetMatching:xxx as this can interfere with the MLM merging in older
            # versions of the driver.
            PY8_Card.vetoParamWriteOut('JetMatching:qCut')
            PY8_Card.vetoParamWriteOut('JetMatching:doShowerKt')
            PY8_Card.vetoParamWriteOut('JetMatching:nJetMax')

            CKKW_cut = None
            # Specific CKKW settings
            if self.run_card['ptlund']<=0.0 and self.run_card['ktdurham']>0.0:
                PY8_Card.subruns[0].MadGraphSet('Merging:doKTMerging',True)
                PY8_Card.subruns[0].MadGraphSet('Merging:Dparameter',
                                                    self.run_card['dparameter'])
                CKKW_cut = 'ktdurham'
            elif self.run_card['ptlund']>0.0 and self.run_card['ktdurham']<=0.0:
                PY8_Card.subruns[0].MadGraphSet('Merging:doPTLundMerging',True)
                CKKW_cut = 'ptlund'                
            else:
                raise InvalidCmd("*Either* the 'ptlund' or 'ktdurham' cut in "+\
                  " the run_card must be turned on to activate CKKW(L) merging"+
                  " with Pythia8, but *both* cuts cannot be turned on at the same time."+
                  "\n ptlund=%f, ktdurham=%f."%(self.run_card['ptlund'],self.run_card['ktdurham']))

            
            # Automatically set qWeed to the CKKWL cut if not defined by the user.
            if PY8_Card['SysCalc:qWeed']==-1.0:
                PY8_Card.MadGraphSet('SysCalc:qWeed',self.run_card[CKKW_cut], force=True)
            
            # MadGraphSet sets the corresponding value (in system mode)
            # only if it is not already user_set.
            if PY8_Card['Merging:TMS']==-1.0:
                if self.run_card[CKKW_cut]>0.0:
                    PY8_Card.MadGraphSet('Merging:TMS',self.run_card[CKKW_cut], force=True)
                else:
                    raise self.InvalidCmd('When running CKKWl merging, the user'+\
                 " select a '%s' cut larger than 0.0 in the run_card."%CKKW_cut)
            if PY8_Card['Merging:TMS']<self.run_card[CKKW_cut]:
                logger.error(
    'The CKKWl merging scale you chose (%f) is less than '%PY8_Card['Merging:TMS']+
    'the %s cut specified in the run_card parameter (=%f).\n'%(CKKW_cut,self.run_card[CKKW_cut])+
    'It is incorrect to use a smaller CKKWl scale than the generation-level %s cut!'%CKKW_cut)
    
            PY8_Card.MadGraphSet('TimeShower:pTmaxMatch',1)
            PY8_Card.MadGraphSet('SpaceShower:pTmaxMatch',1)
            PY8_Card.MadGraphSet('SpaceShower:rapidityOrder',False)
            # PY8 should not implement the CKKW veto since the driver should do it.
            if self.run_card['use_syst']:
                # We do no force it here, but it is clear that the user should know what
                # he's doing if he were to force it to True.
                PY8_Card.MadGraphSet('Merging:applyVeto',False)
                PY8_Card.MadGraphSet('Merging:includeWeightInXsection',False)
            # Use the parameter maxjetflavor for Merging:nQuarksMerge which specifies
            # up to which parton must be matched.
            PY8_Card.MadGraphSet('Merging:nQuarksMerge',self.run_card['maxjetflavor'])
            if not hasattr(self,'proc_characteristic'):
                self.proc_characteristic = self.get_characteristics()
            nJetMax = self.proc_characteristic['max_n_matched_jets']
            if PY8_Card['Merging:nJetMax'.lower()] == -1:
                logger.info("No user-defined value for Pythia8 parameter "+
                "'Merging:nJetMax'. Setting it automatically to %d."%nJetMax)
                PY8_Card.MadGraphSet('Merging:nJetMax',nJetMax, force=True)
            if PY8_Card['SysCalc:tmsList']=='auto':
                if self.run_card['use_syst']:
                    if self.run_card['sys_matchscale']=='auto':
                        tms = PY8_Card["Merging:TMS"]
                        value = [factor*tms for factor in [0.5,0.75,1.0,1.5,2.0]
                                 if factor*tms > self.run_card[CKKW_cut]]
                        PY8_Card.MadGraphSet('SysCalc:tmsList', value, force=True)
                    else:
                        tmsList = [float(tms) for tms in self.run_card['sys_matchscale'].split()]
                        if PY8_Card['Merging:TMS'] not in tmsList:
                            tmsList.append(PY8_Card['Merging:TMS'])
                        PY8_Card.MadGraphSet('SysCalc:tmsList', tmsList, force=True)
                #else:
                #    PY8_Card.MadGraphSet('SysCalc:tmsList', [], force=True)
            if PY8_Card['SysCalc:tmsList']!='auto':
                for scale in PY8_Card['SysCalc:tmsList']:
                    if float(scale)<float(self.run_card[CKKW_cut]):
                        logger.error(
        'One of the CKKWl merging scale you chose (%f) in the variation list'%scale+\
        " (either via 'SysCalc:tmsList' in the PY8 shower card or "+\
        "'sys_matchscale' in the run_card) is less than %f, "%self.run_card[CKKW_cut]+
        'the %s cut specified in the run_card parameter.\n'%CKKW_cut+
        'It is incorrect to use a smaller CKKWl scale than the generation-level %s cut!'%CKKW_cut)
        else:
            # When not performing any merging, make sure that we do not write out the parameter
            # JetMatching:xxx or Merging:xxx as this can trigger undesired vetos in an unmerged
            # simulation.
            PY8_Card.vetoParamWriteOut('Merging:TMS')
            PY8_Card.vetoParamWriteOut('Merging:Process')
            PY8_Card.vetoParamWriteOut('Merging:nJetMax')
            PY8_Card.vetoParamWriteOut('JetMatching:qCut')
            PY8_Card.vetoParamWriteOut('JetMatching:doShowerKt')
            PY8_Card.vetoParamWriteOut('JetMatching:nJetMax')

        return HepMC_event_output

    def do_pythia8(self, line):
        """launch pythia8"""


        try:
            import madgraph
        except ImportError:  
            import internal.histograms as histograms
        else:
            import madgraph.various.histograms as histograms

        # Check argument's validity
        args = self.split_arg(line)
        if '--no_default' in args:
            if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pythia8_card.dat')):
                return
            no_default = True
            args.remove('--no_default')
        else:
            no_default = False

        if not self.run_name:
            self.check_pythia8(args)
            self.configure_directory(html_opening =False)
        else:
            # initialize / remove lhapdf mode        
            self.configure_directory(html_opening =False)
            self.check_pythia8(args)        

        # Update the banner with the pythia card
        if not self.banner or len(self.banner) <=1:
            # Here the level keyword 'pythia' must not be changed to 'pythia8'.
            self.banner = banner_mod.recover_banner(self.results, 'pythia')

        # the args are modify and the last arg is always the mode 
        if not no_default:
            self.ask_pythia_run_configuration(args[-1], pythia_version=8, banner=self.banner)

        if self.options['automatic_html_opening']:
            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
            self.options['automatic_html_opening'] = False

        if self.run_card['event_norm'] not in ['unit','average']:
            logger.critical("Pythia8 does not support normalization to the sum. Not running Pythia8")
            return
             #\n"+\
             #"The normalisation of the hepmc output file will be wrong (i.e. non-standard).\n"+\
             #"Please use 'event_norm = average' in the run_card to avoid this problem.")


        
        if not self.options['mg5amc_py8_interface_path'] or not \
             os.path.exists(pjoin(self.options['mg5amc_py8_interface_path'],
                                                       'MG5aMC_PY8_interface')):
            raise self.InvalidCmd(
"""The MG5aMC_PY8_interface tool cannot be found, so that MadEvent cannot steer Pythia8 shower.
Please install this tool with the following MG5_aMC command:
  MG5_aMC> install mg5amc_py8_interface_path""")
        else:
            pythia_main = pjoin(self.options['mg5amc_py8_interface_path'],
                                                         'MG5aMC_PY8_interface')
            warnings = MadEventCmd.mg5amc_py8_interface_consistency_warning(self.options)
            if warnings:
                logger.warning(warnings)

        self.results.add_detail('run_mode', 'madevent')

        # Again here 'pythia' is just a keyword for the simulation level.
        self.update_status('\033[92mRunning Pythia8 [arXiv:1410.3012]\033[0m', 'pythia8')
        
        tag = self.run_tag        
        # Now write Pythia8 card
        # Start by reading, starting from the default one so that the 'user_set'
        # tag are correctly set.
        PY8_Card = banner_mod.PY8Card(pjoin(self.me_dir, 'Cards', 
                                                    'pythia8_card_default.dat'))
        PY8_Card.read(pjoin(self.me_dir, 'Cards', 'pythia8_card.dat'),
                                                                  setter='user')
        
        run_type = 'default'
        merged_run_types = ['MLM','CKKW']
        if int(self.run_card['ickkw'])==1:
            run_type = 'MLM'
        elif int(self.run_card['ickkw'])==2 or \
                   self.run_card['ktdurham']>0.0 or self.run_card['ptlund']>0.0:
            run_type = 'CKKW'

        # Edit the card and run environment according to the run specification
        HepMC_event_output = self.setup_Pythia8RunAndCard(PY8_Card, run_type)

        # Now write the card.
        pythia_cmd_card = pjoin(self.me_dir, 'Events', self.run_name ,
                                                         '%s_pythia8.cmd' % tag)
        cmd_card = StringIO.StringIO()
        PY8_Card.write(cmd_card,pjoin(self.me_dir,'Cards','pythia8_card_default.dat'),
                                                       direct_pythia_input=True)
        
        # Now setup the preamble to make sure that everything will use the locally
        # installed tools (if present) even if the user did not add it to its
        # environment variables.
        if 'heptools_install_dir' in self.options:
            preamble = misc.get_HEPTools_location_setter(
                                     self.options['heptools_install_dir'],'lib')
        else:
            if MADEVENT:
                preamble = misc.get_HEPTools_location_setter(
                pjoin(self.options['mg5amc_py8_interface_path'],os.pardir),'lib')
            else:
                preamble = misc.get_HEPTools_location_setter(
                                                 pjoin(MG5DIR,'HEPTools'),'lib')
        #preamble += "\n unset PYTHIA8DATA\n"
        
        open(pythia_cmd_card,'w').write("""!
! It is possible to run this card manually with:
!    %s %s
!
"""%(preamble+pythia_main,os.path.basename(pythia_cmd_card))+cmd_card.getvalue())
       
        # launch pythia8
        pythia_log = pjoin(self.me_dir , 'Events', self.run_name ,
                                                         '%s_pythia8.log' % tag)

        # Write a bash wrapper to run the shower with custom environment variables
        wrapper_path = pjoin(self.me_dir,'Events',self.run_name,'run_shower.sh')
        wrapper = open(wrapper_path,'w')
        shell = 'bash' if misc.get_shell_type() in ['bash',None] else 'tcsh'
        shell_exe = None
        if os.path.exists('/usr/bin/env'):
            shell_exe = '/usr/bin/env %s'%shell
        else: 
            shell_exe = misc.which(shell)
            if not shell_exe:
                raise self.InvalidCmd('No s hell could be found in your environment.\n'+
                  "Make sure that either '%s' is in your path or that the"%shell+\
                  " command '/usr/bin/env %s' exists and returns a valid path."%shell)
                
        exe_cmd = "#!%s\n%s"%(shell_exe,' '.join(
                     [preamble+pythia_main,
                      os.path.basename(pythia_cmd_card)]))

        wrapper.write(exe_cmd)
        wrapper.close()

        # Set it as executable
        st = os.stat(wrapper_path)
        os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)

        # If the target HEPMC output file is a fifo, don't hang MG5_aMC and let
        # it proceed.
        is_HepMC_output_fifo = False if not HepMC_event_output else \
                              ( os.path.exists(HepMC_event_output) and \
                              stat.S_ISFIFO(os.stat(HepMC_event_output).st_mode))
        startPY8timer = time.time()
        
        # Information that will be extracted from this PY8 run
        PY8_extracted_information={ 'sigma_m':None, 'Nacc':None, 'Ntry':None,
                                    'cross_sections':{} }

        if is_HepMC_output_fifo:
            logger.info(
"""Pythia8 is set to output HEPMC events to to a fifo file.
You can follow PY8 run with the following command (in a separate terminal):
    tail -f %s"""%pythia_log )
            py8_log = open( pythia_log,'w')
            py8_bkgrd_proc = misc.Popen([wrapper_path],
                    stdout=py8_log,stderr=py8_log,
                                  cwd=pjoin(self.me_dir,'Events',self.run_name))
            # Now directly return to madevent interactive interface if we are piping PY8
            if not no_default:
                logger.info('You can now run a tool that reads the following fifo file:'+\
                '\n   %s\nwhere PY8 outputs HEPMC events (e.g. MadAnalysis5).'
                                          %HepMC_event_output,'$MG:color:GREEN')
            return
        else:
            if self.options ['run_mode']!=0:
                # Start a parallelization instance (stored in self.cluster)
                self.configure_run_mode(self.options['run_mode'])
                if self.options['run_mode']==1:
                    n_cores = max(int(self.options['cluster_size']),1)
                elif self.options['run_mode']==2:
                    n_cores = max(int(self.cluster.nb_core),1)
                
                lhe_file_name = os.path.basename(PY8_Card.subruns[0]['Beams:LHEF'])
                lhe_file = lhe_parser.EventFile(pjoin(self.me_dir,'Events',
                                                    self.run_name,PY8_Card.subruns[0]['Beams:LHEF']))
                n_available_events = len(lhe_file)
                if PY8_Card['Main:numberOfEvents']==-1:
                    n_events = n_available_events
                else:
                    n_events = PY8_Card['Main:numberOfEvents']
                    if n_events > n_available_events:
                        raise self.InvalidCmd('You specified more events (%d) in the PY8 parameter'%n_events+\
                            "'Main:numberOfEvents' than the total number of events available (%d)"%n_available_events+\
                            ' in the event file:\n %s'%pjoin(self.me_dir,'Events',self.run_name,PY8_Card.subruns[0]['Beams:LHEF']))

                # Implement a security to insure a minimum numbe of events per job
                if self.options['run_mode']==2:
                    min_n_events_per_job = 100
                elif self.options['run_mode']==1:
                    min_n_events_per_job = 1000                   
                min_n_core = n_events//min_n_events_per_job
                n_cores = max(min(min_n_core,n_cores),1)

            if self.options['run_mode']==0 or (self.options['run_mode']==2 and self.options['nb_core']==1):
                # No need for parallelization anymore
                self.cluster = None
                logger.info('Follow Pythia8 shower by running the '+
                    'following command (in a separate terminal):\n    tail -f %s'%pythia_log)

                if self.options['run_mode']==2 and self.options['nb_core']>1:    
                    ret_code = self.cluster.launch_and_wait(wrapper_path, 
                        argument= [], stdout= pythia_log, stderr=subprocess.STDOUT,
                                      cwd=pjoin(self.me_dir,'Events',self.run_name))
                else:
                    stdout = open(pythia_log,'w')
                    ret_code = misc.call(wrapper_path, stdout=stdout, stderr=subprocess.STDOUT,
                                      cwd=pjoin(self.me_dir,'Events',self.run_name))
                    stdout.close()
                if ret_code != 0:
                    raise self.InvalidCmd('Pythia8 shower interrupted with return'+\
                        ' code %d.\n'%ret_code+\
                        'You can find more information in this log file:\n%s'%pythia_log)
            else:
                if self.run_card['event_norm']=='sum':
                    logger.error("")
                    logger.error("Either run in single core or change event_norm to 'average'.")
                    raise InvalidCmd("Pythia8 parallelization with event_norm set to 'sum' is not supported."
                                    "Either run in single core or change event_norm to 'average'.")

                # Create the parallelization folder
                parallelization_dir = pjoin(self.me_dir,'Events',self.run_name,'PY8_parallelization')
                if os.path.isdir(parallelization_dir):
                    shutil.rmtree(parallelization_dir)
                os.mkdir(parallelization_dir)
                # Copy what should be the now standalone executable for PY8
                shutil.copy(pythia_main,parallelization_dir)
                # Add a safe card in parallelization
                ParallelPY8Card = copy.copy(PY8_Card)
                assert ParallelPY8Card['JetMatching:nJetMax'] ==  PY8_Card['JetMatching:nJetMax']

                # Normalize the name of the HEPMCouput and lhe input
                if HepMC_event_output:
                    ParallelPY8Card['HEPMCoutput:file']='events.hepmc'
                else:
                    ParallelPY8Card['HEPMCoutput:file']='/dev/null'

                ParallelPY8Card.subruns[0].systemSet('Beams:LHEF','events.lhe.gz')
                ParallelPY8Card.write(pjoin(parallelization_dir,'PY8Card.dat'),
                                      pjoin(self.me_dir,'Cards','pythia8_card_default.dat'),
                                                                    direct_pythia_input=True)
                # Write the wrapper
                wrapper_path = pjoin(parallelization_dir,'run_PY8.sh')
                wrapper = open(wrapper_path,'w')
                if self.options['cluster_temp_path'] is None:
                    exe_cmd = \
"""#!%s 
./%s PY8Card.dat >& PY8_log.txt
"""
                else: 
                    exe_cmd = \
"""#!%s
ln -s ./events_$1.lhe.gz ./events.lhe.gz
./%s PY8Card_$1.dat >& PY8_log.txt
mkdir split_$1
if [ -f ./events.hepmc ];
then
   mv ./events.hepmc ./split_$1/
fi
if [ -f ./pts.dat ];
then
   mv ./pts.dat ./split_$1/
fi
if [ -f ./djrs.dat ];
then
   mv ./djrs.dat ./split_$1/
fi
if [ -f ./PY8_log.txt ];
then
   mv ./PY8_log.txt ./split_$1/
fi
tar -czf split_$1.tar.gz split_$1
"""
                exe_cmd = exe_cmd%(shell_exe,os.path.basename(pythia_main))
                wrapper.write(exe_cmd)
                wrapper.close()
                # Set it as executable
                st = os.stat(wrapper_path)
                os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)
                
                # Split the .lhe event file, create event partition
                partition=[n_available_events//n_cores]*n_cores
                for i in range(n_available_events%n_cores):
                    partition[i] += 1
                
                # Splitting according to the total number of events requested by the user
                # Will be used to determine the number of events to indicate in the PY8 split cards.
                partition_for_PY8=[n_events//n_cores]*n_cores
                for i in range(n_events%n_cores):
                    partition_for_PY8[i] += 1
                
                logger.info('Splitting .lhe event file for PY8 parallelization...')    
                n_splits = lhe_file.split(partition=partition, cwd=parallelization_dir, zip=True)                
                
                if n_splits!=len(partition):
                    raise MadGraph5Error('Error during lhe file splitting. Expected %d files but obtained %d.'
                                                                            %(len(partition),n_splits))
                # Distribute the split events
                split_files    = []
                split_dirs     = []
                for split_id in range(n_splits):
                    split_files.append('events_%s.lhe.gz'%split_id)
                    split_dirs.append(pjoin(parallelization_dir,'split_%d'%split_id))
                    # Add the necessary run content
                    shutil.move(pjoin(parallelization_dir,lhe_file.name+'_%d.lhe.gz'%split_id),
                                pjoin(parallelization_dir,split_files[-1]))
                
                logger.info('Submitting Pythia8 jobs...')
                for i, split_file in enumerate(split_files):
                    # We must write a PY8Card tailored for each split so as to correct the normalization
                    # HEPMCoutput:scaling of each weight since the lhe showered will not longer contain the
                    # same original number of events
                    split_PY8_Card = banner_mod.PY8Card(pjoin(parallelization_dir,'PY8Card.dat'))
                    # Make sure to sure the number of split_events determined during the splitting.
                    split_PY8_Card.systemSet('Main:numberOfEvents',partition_for_PY8[i])
                    split_PY8_Card.systemSet('HEPMCoutput:scaling',split_PY8_Card['HEPMCoutput:scaling']*
                                                             (float(partition_for_PY8[i])))
                    # Add_missing set to False so as to be sure not to add any additional parameter w.r.t
                    # the ones in the original PY8 param_card copied.
                    split_PY8_Card.write(pjoin(parallelization_dir,'PY8Card_%d.dat'%i),
                                         pjoin(parallelization_dir,'PY8Card.dat'), add_missing=False)
                    in_files = [pjoin(parallelization_dir,os.path.basename(pythia_main)),
                                pjoin(parallelization_dir,'PY8Card_%d.dat'%i), 
                                pjoin(parallelization_dir,split_file)]
                    if self.options['cluster_temp_path'] is None:
                        out_files = []
                        os.mkdir(pjoin(parallelization_dir,'split_%d'%i))
                        selected_cwd = pjoin(parallelization_dir,'split_%d'%i)
                        for in_file in in_files+[pjoin(parallelization_dir,'run_PY8.sh')]:
                            # Make sure to rename the split_file link from events_<x>.lhe.gz to events.lhe.gz
                            # and similarly for PY8Card
                            if os.path.basename(in_file)==split_file:
                                ln(in_file,selected_cwd,name='events.lhe.gz')
                            elif os.path.basename(in_file).startswith('PY8Card'):
                                ln(in_file,selected_cwd,name='PY8Card.dat')                                
                            else:
                                ln(in_file,selected_cwd)                                
                        in_files  = []
                        wrapper_path = os.path.basename(wrapper_path)
                    else:
                        out_files = ['split_%d.tar.gz'%i]
                        selected_cwd = parallelization_dir

                    self.cluster.submit2(wrapper_path, 
                            argument=[str(i)], cwd=selected_cwd, 
                            input_files=in_files,
                            output_files=out_files,
                            required_output=out_files)
                
                def wait_monitoring(Idle, Running, Done):
                    if Idle+Running+Done == 0:
                        return
                    logger.info('Pythia8 shower jobs: %d Idle, %d Running, %d Done [%s]'\
                                %(Idle, Running, Done, misc.format_time(time.time() - startPY8timer)))
                self.cluster.wait(parallelization_dir,wait_monitoring)
                
                logger.info('Merging results from the split PY8 runs...')
                if self.options['cluster_temp_path']:
                    # Decompressing the output
                    for i, split_file in enumerate(split_files):
                        misc.call(['tar','-xzf','split_%d.tar.gz'%i],cwd=parallelization_dir)
                        os.remove(pjoin(parallelization_dir,'split_%d.tar.gz'%i))
                
                # Now merge logs
                pythia_log_file = open(pythia_log,'w')
                n_added = 0
                for split_dir in split_dirs:
                    log_file = pjoin(split_dir,'PY8_log.txt')
                    pythia_log_file.write('='*35+'\n')
                    pythia_log_file.write(' -> Pythia8 log file for run %d <-'%i+'\n')
                    pythia_log_file.write('='*35+'\n')
                    pythia_log_file.write(open(log_file,'r').read()+'\n')
                    if run_type in merged_run_types:
                        sigma_m, Nacc, Ntry = self.parse_PY8_log_file(log_file)
                        if any(elem is None for elem in [sigma_m, Nacc, Ntry]):
                            continue
                        n_added += 1
                        if PY8_extracted_information['sigma_m'] is None:
                           PY8_extracted_information['sigma_m'] = sigma_m
                        else:
                           PY8_extracted_information['sigma_m'] += sigma_m
                        if PY8_extracted_information['Nacc'] is None:
                           PY8_extracted_information['Nacc'] = Nacc
                        else:
                           PY8_extracted_information['Nacc'] += Nacc
                        if PY8_extracted_information['Ntry'] is None:
                           PY8_extracted_information['Ntry'] = Ntry
                        else:
                           PY8_extracted_information['Ntry'] += Ntry

                # Normalize the values added
                if n_added>0:
                    PY8_extracted_information['sigma_m'] /= float(n_added)
                pythia_log_file.close()
                
                # djr plots
                djr_HwU = None
                n_added = 0
                for split_dir in split_dirs:
                    djr_file = pjoin(split_dir,'djrs.dat')
                    if not os.path.isfile(djr_file):
                        continue
                    xsecs = self.extract_cross_sections_from_DJR(djr_file)
                    if len(xsecs)>0:
                        n_added += 1
                        if len(PY8_extracted_information['cross_sections'])==0:
                            PY8_extracted_information['cross_sections'] = xsecs
                            # Square the error term
                            for key in PY8_extracted_information['cross_sections']:
                                PY8_extracted_information['cross_sections'][key][1] = \
                                    PY8_extracted_information['cross_sections'][key][1]**2
                        else:
                            for key, value in xsecs.items():
                                PY8_extracted_information['cross_sections'][key][0] += value[0]
                                # Add error in quadrature
                                PY8_extracted_information['cross_sections'][key][1] += value[1]**2
                    new_djr_HwU = histograms.HwUList(djr_file,run_id=0)
                    if djr_HwU is None:
                        djr_HwU = new_djr_HwU
                    else:
                        for i, hist in enumerate(djr_HwU):
                            djr_HwU[i] = hist + new_djr_HwU[i]


                if not djr_HwU is None:
                    djr_HwU.output(pjoin(self.me_dir,'Events',self.run_name,'djrs'),format='HwU')
                    shutil.move(pjoin(self.me_dir,'Events',self.run_name,'djrs.HwU'),
                                pjoin(self.me_dir,'Events',self.run_name,'%s_djrs.dat'%tag))

                if n_added>0:
                    for key in PY8_extracted_information['cross_sections']:
                        # The cross-sections in the DJR are normalized for the original number of events, so we should not
                        # divide by n_added anymore for the cross-section value
                        # PY8_extracted_information['cross_sections'][key][0] /= float(n_added)
                        PY8_extracted_information['cross_sections'][key][1] = \
                         math.sqrt(PY8_extracted_information['cross_sections'][key][1]) / float(n_added)

                # pts plots
                pts_HwU = None
                for split_dir in split_dirs:
                    pts_file = pjoin(split_dir,'pts.dat')
                    if not os.path.isfile(pts_file):
                        continue
                    new_pts_HwU = histograms.HwUList(pts_file,run_id=0)
                    if pts_HwU is None:
                        pts_HwU = new_pts_HwU
                    else:
                        for i, hist in enumerate(pts_HwU):
                            pts_HwU[i] = hist + new_pts_HwU[i]
                if not pts_HwU is None:
                    pts_HwU.output(pjoin(self.me_dir,'Events',self.run_name,'pts'),format='HwU')
                    shutil.move(pjoin(self.me_dir,'Events',self.run_name,'pts.HwU'),
                                pjoin(self.me_dir,'Events',self.run_name,'%s_pts.dat'%tag))

                # HepMC events now.
                all_hepmc_files = []
                for split_dir in split_dirs:
                    hepmc_file = pjoin(split_dir,'events.hepmc')
                    if not os.path.isfile(hepmc_file):
                        continue
                    all_hepmc_files.append(hepmc_file)
                
                if len(all_hepmc_files)>0:
                    hepmc_output = pjoin(self.me_dir,'Events',self.run_name,HepMC_event_output)
                    with misc.TMP_directory() as tmp_dir:
                        # Use system calls to quickly put these together
                        header = open(pjoin(tmp_dir,'header.hepmc'),'w')
                        n_head = 0
                        for line in open(all_hepmc_files[0],'r'):
                            if not line.startswith('E'):
                                n_head += 1
                                header.write(line)
                            else:
                                break
                        header.close()
                        tail = open(pjoin(tmp_dir,'tail.hepmc'),'w')
                        n_tail = 0  
                                     
                        for line in misc.reverse_readline(all_hepmc_files[-1]):
                            if line.startswith('HepMC::'):
                                n_tail += 1
                                tail.write(line)
                            else:
                                break
                        tail.close()
                        if n_tail>1:
                            raise MadGraph5Error('HEPMC files should only have one trailing command.')
                        ###################################################################### 
                        # This is the most efficient way of putting together HEPMC's, *BUT*  #
                        #    WARNING: NEED TO RENDER THE CODE BELOW SAFE TOWARDS INJECTION   #
                        ######################################################################
                        for hepmc_file in all_hepmc_files:
                            # Remove in an efficient way the starting and trailing HEPMC tags
                            # check for support of negative argument in head
                            devnull = open(os.path.devnull, 'w')
                            pid = misc.call(['head','-n', '-1', __file__], stdout=devnull, stderr=devnull)
                            devnull.close()
                            if pid == 0:
                                misc.call('head -n -1 %s | tail -n +%d > %s/tmpfile' %
                                          (hepmc_file, n_head+1, os.path.dirname(hepmc_file)), shell=True)
                                misc.call(['mv', 'tmpfile', os.path.basename(hepmc_file)], cwd=os.path.dirname(hepmc_file))
                            elif sys.platform == 'darwin':
                                # sed on MAC has slightly different synthax than on
                                os.system(' '.join(['sed','-i',"''","'%s;$d'"%
                                        (';'.join('%id'%(i+1) for i in range(n_head))),hepmc_file]))          
                            else:                                
                                # other UNIX systems 
                                os.system(' '.join(['sed','-i']+["-e '%id'"%(i+1) for i in range(n_head)]+
                                                                            ["-e '$d'",hepmc_file]))
                            
                        os.system(' '.join(['cat',pjoin(tmp_dir,'header.hepmc')]+all_hepmc_files+
                                                    [pjoin(tmp_dir,'tail.hepmc'),'>',hepmc_output]))

                # We are done with the parallelization directory. Clean it.
                if os.path.isdir(parallelization_dir):
                    shutil.rmtree(parallelization_dir)

        # Properly rename the djr and pts output if present.
        djr_output = pjoin(self.me_dir,'Events', self.run_name, 'djrs.dat')
        if os.path.isfile(djr_output):
            shutil.move(djr_output, pjoin(self.me_dir,'Events',
                                            self.run_name, '%s_djrs.dat' % tag))
        pt_output = pjoin(self.me_dir,'Events', self.run_name, 'pts.dat')
        if os.path.isfile(pt_output):
            shutil.move(pt_output, pjoin(self.me_dir,'Events',
                                            self.run_name, '%s_pts.dat' % tag))

        if not os.path.isfile(pythia_log) or \
             'Inclusive cross section:' not in '\n'.join(open(pythia_log,'r').readlines()[-20:]):
            logger.warning('Fail to produce a pythia8 output. More info in \n     %s'%pythia_log)
            return
        
        # Plot for Pythia8
        successful = self.create_plot('Pythia8')
        if not successful:
            logger.warning('Failed to produce Pythia8 merging plots.')
        
        self.to_store.append('pythia8')

        # Study matched cross-sections
        if run_type in merged_run_types:
            # From the log file
            if all(PY8_extracted_information[_] is None for _ in ['sigma_m','Nacc','Ntry']):
                # When parallelization is enable we shouldn't have cannot look in the log in this way
                if self.options['run_mode']==0 or (self.options['run_mode']==2 and self.options['nb_core']==1):
                    PY8_extracted_information['sigma_m'],PY8_extracted_information['Nacc'],\
                        PY8_extracted_information['Ntry'] = self.parse_PY8_log_file(
                        pjoin(self.me_dir,'Events', self.run_name,'%s_pythia8.log' % tag))      
                else:
                    logger.warning('Pythia8 cross-section could not be retreived.\n'+
                       'Try turning parallelization off by setting the option nb_core to 1. YYYYY')

            if not any(PY8_extracted_information[_] is None for _ in ['sigma_m','Nacc','Ntry']):
                self.results.add_detail('cross_pythia', PY8_extracted_information['sigma_m'])
                self.results.add_detail('nb_event_pythia', PY8_extracted_information['Nacc'])
                # Shorthands
                Nacc = PY8_extracted_information['Nacc']
                Ntry = PY8_extracted_information['Ntry']
                sigma_m = PY8_extracted_information['sigma_m']
                # Compute pythia error
                error = self.results[self.run_name].return_tag(self.run_tag)['error'] 
                try:                   
                    error_m = math.sqrt((error * Nacc/Ntry)**2 + sigma_m**2 *(1-Nacc/Ntry)/Nacc)
                except ZeroDivisionError:
                    # Cannot compute error
                    error_m = -1.0
                # works both for fixed number of generated events and fixed accepted events
                self.results.add_detail('error_pythia', error_m)

            if self.run_card['use_syst']:
                    self.results.add_detail('cross_pythia', -1)
                    self.results.add_detail('error_pythia', 0)

            # From the djr file generated
            djr_output = pjoin(self.me_dir,'Events',self.run_name,'%s_djrs.dat'%tag)
            if os.path.isfile(djr_output) and len(PY8_extracted_information['cross_sections'])==0:
                # When parallelization is enable we shouldn't have cannot look in the log in this way
                if self.options['run_mode']==0 or (self.options['run_mode']==2 and self.options['nb_core']==1):
                    PY8_extracted_information['cross_sections'] = self.extract_cross_sections_from_DJR(djr_output)
                else:
                    logger.warning('Pythia8 merged cross-sections could not be retreived.\n'+
                       'Try turning parallelization off by setting the option nb_core to 1.XXXXX')
                    PY8_extracted_information['cross_sections'] = {} 
                    
            cross_sections = PY8_extracted_information['cross_sections']
            if cross_sections:
                # Filter the cross_sections specified an keep only the ones 
                # with central parameters and a different merging scale
                a_float_re = r'[\+|-]?\d+(\.\d*)?([EeDd][\+|-]?\d+)?'
                central_merging_re = re.compile(
                  r'^\s*Weight_MERGING\s*=\s*(?P<merging>%s)\s*$'%a_float_re,
                                                                  re.IGNORECASE)                
                cross_sections = dict(
                    (float(central_merging_re.match(xsec).group('merging')),value)
                        for xsec, value in cross_sections.items() if not 
                                         central_merging_re.match(xsec) is None)
                central_scale = PY8_Card['JetMatching:qCut'] if \
                        int(self.run_card['ickkw'])==1 else PY8_Card['Merging:TMS']
                if central_scale in cross_sections:
                    self.results.add_detail('cross_pythia8', cross_sections[central_scale][0])
                    self.results.add_detail('error_pythia8', cross_sections[central_scale][1])
                
                #logger.info('Pythia8 merged cross-sections are:')
                #for scale in sorted(cross_sections.keys()):
                #   logger.info(' > Merging scale = %-6.4g : %-11.5g +/- %-7.2g [pb]'%\
                #               (scale,cross_sections[scale][0],cross_sections[scale][1]))       
            
            xsecs_file = open(pjoin(self.me_dir,'Events',self.run_name,
                                                 '%s_merged_xsecs.txt'%tag),'w')
            if cross_sections:
                xsecs_file.write('%-20s%-20s%-20s\n'%('Merging scale',
                                    'Cross-section [pb]','MC uncertainty [pb]'))
                for scale in sorted(cross_sections.keys()):
                    xsecs_file.write('%-20.4g%-20.6e%-20.2e\n'%
                      (scale,cross_sections[scale][0],cross_sections[scale][1]))
            else:
                xsecs_file.write('Cross-sections could not be read from the'+\
                    "XML node 'xsection' of the .dat file produced by Pythia8.")
            xsecs_file.close()
        
        #Update the banner
        # We add directly the pythia command card because it has the full 
        # information
        self.banner.add(pythia_cmd_card)

        if int(self.run_card['ickkw']):
            # Add the matched cross-section
            if 'MGGenerationInfo' in self.banner:
                self.banner['MGGenerationInfo'] += '#  Matched Integrated weight (pb)  :  %s\n' % self.results.current['cross_pythia']
            else:
                self.banner['MGGenerationInfo'] = '#  Matched Integrated weight (pb)  :  %s\n' % self.results.current['cross_pythia']
        banner_path = pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, tag))
        self.banner.write(banner_path)

        self.update_status('Pythia8 shower finished after %s.'%misc.format_time(time.time() - startPY8timer), level='pythia8')
        if self.options['delphes_path']:
            self.exec_cmd('delphes --no_default', postcmd=False, printcmd=False)
        self.print_results_in_shell(self.results.current)
    
    def parse_PY8_log_file(self, log_file_path):
        """ Parse a log file to extract number of event and cross-section. """
        pythiare = re.compile(r"Les Houches User Process\(es\)\s*\d+\s*\|\s*(?P<tried>\d+)\s*(?P<selected>\d+)\s*(?P<generated>\d+)\s*\|\s*(?P<xsec>[\d\.e\-\+]+)\s*(?P<xsec_error>[\d\.e\-\+]+)")
        pythia_xsec_re = re.compile(r"Inclusive cross section\s*:\s*(?P<xsec>[\d\.e\-\+]+)\s*(?P<xsec_error>[\d\.e\-\+]+)")
        sigma_m, Nacc, Ntry = None, None, None
        for line in misc.BackRead(log_file_path): 
            info = pythiare.search(line)
            if not info:
                # Also try to obtain the cross-section and error from the final xsec line of pythia8 log
                # which is more reliable, in general for example when there is merging and the last event
                # is skipped.
                final_PY8_xsec = pythia_xsec_re.search(line)
                if not final_PY8_xsec:
                    continue
                else:
                    sigma_m = float(final_PY8_xsec.group('xsec')) *1e9
                    continue
            else:
                try:
                    # Pythia cross section in mb, we want pb
                    if sigma_m is None:
                        sigma_m = float(info.group('xsec')) *1e9
                    if Nacc is None:
                        Nacc = int(info.group('generated'))
                    if Ntry is None:
                        Ntry = int(info.group('tried'))
                    if Nacc==0:
                        raise self.InvalidCmd('Pythia8 shower failed since it'+\
                          ' did not accept any event from the MG5aMC event file.')
                    return sigma_m, Nacc, Ntry
                except ValueError:
                    return None,None,None

        raise self.InvalidCmd("Could not find cross-section and event number information "+\
                         "in Pythia8 log\n  '%s'."%log_file_path)
    
    def extract_cross_sections_from_DJR(self,djr_output):
        """Extract cross-sections from a djr XML output."""
        import xml.dom.minidom as minidom
        run_nodes = minidom.parse(djr_output).getElementsByTagName("run")
        all_nodes = dict((int(node.getAttribute('id')),node) for
                                                      node in run_nodes)
        try:
            selected_run_node = all_nodes[0]
        except:
            return {}
        xsections = selected_run_node.getElementsByTagName("xsection")
        # In the DJR, the conversion to pb is already performed
        return dict((xsec.getAttribute('name'),
            [float(xsec.childNodes[0].data.split()[0]),
             float(xsec.childNodes[0].data.split()[1])])
                                              for xsec in xsections)
    
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

        if self.run_card['event_norm'] != 'sum':
            logger.error('pythia-pgs require event_norm to be on sum. Do not run pythia6')
            return

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
        
        self.results.add_detail('run_mode', 'madevent')

        self.update_status('Running Pythia', 'pythia')
        try:
            os.remove(pjoin(self.me_dir,'Events','pythia.done'))
        except Exception:
            pass        
        
        ## LAUNCHING PYTHIA
        # check that LHAPATH is define.
        if not re.search(r'^\s*LHAPATH=%s/PDFsets'  % pythia_src,
                          open(pjoin(self.me_dir,'Cards','pythia_card.dat')).read(), 
                          re.M):
            f = open(pjoin(self.me_dir,'Cards','pythia_card.dat'),'a')
            f.write('\n     LHAPATH=%s/PDFsets' % pythia_src)
            f.close()
        tag = self.run_tag
        pythia_log = pjoin(self.me_dir, 'Events', self.run_name , '%s_pythia.log' % tag)
        #self.cluster.launch_and_wait('../bin/internal/run_pythia', 
        #                argument= [pythia_src], stdout= pythia_log,
        #                stderr=subprocess.STDOUT,
        #                cwd=pjoin(self.me_dir,'Events'))
        output_files = ['pythia_events.hep']
        if self.run_card['use_syst']:
            output_files.append('syst.dat')
        if self.run_card['ickkw'] == 1: 
            output_files += ['beforeveto.tree', 'xsecs.tree', 'events.tree']
        
        os.environ['PDG_MASS_TBL'] = pjoin(pythia_src,'mass_width_2004.mc')
        self.cluster.launch_and_wait(pjoin(pythia_src, 'pythia'),
                                     input_files=[pjoin(self.me_dir, "Events", "unweighted_events.lhe"),
                                                  pjoin(self.me_dir,'Cards','pythia_card.dat'),
                                                  pjoin(pythia_src,'mass_width_2004.mc')],
                                     output_files=output_files,
                                     stdout= pythia_log,
                                     stderr=subprocess.STDOUT,
                                     cwd=pjoin(self.me_dir,'Events'))
            

        os.remove(pjoin(self.me_dir, "Events", "unweighted_events.lhe"))

        if not os.path.exists(pjoin(self.me_dir,'Events','pythia_events.hep')):
            logger.warning('Fail to produce pythia output. More info in \n     %s' % pythia_log)
            return
        
        self.to_store.append('pythia')
        
        # Find the matched cross-section
        if int(self.run_card['ickkw']):
            # read the line from the bottom of the file
            #pythia_log = misc.BackRead(pjoin(self.me_dir,'Events', self.run_name, 
            #                                             '%s_pythia.log' % tag))
            pythiare = re.compile(r"\s*I\s+0 All included subprocesses\s+I\s+(?P<generated>\d+)\s+(?P<tried>\d+)\s+I\s+(?P<xsec>[\d\.D\-+]+)\s+I")            
            for line in misc.reverse_readline(pjoin(self.me_dir,'Events', self.run_name, 
                                                         '%s_pythia.log' % tag)):
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

            #pythia_log.close()
        
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
            except SysCalcError as error:
                logger.error(str(error))
            else:
                if os.path.exists(pjoin(self.me_dir,'Events', 'syst.dat')):
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
                for match in misc.glob(pjoin('*','*_banner.txt'), pjoin(self.me_dir, 'Events')):
                    run = match.rsplit(os.path.sep,2)[1]
                    if self.force:
                        args.append('-f')
                    try:
                        self.exec_cmd('remove %s %s' % (run, ' '.join(args[1:]) ) )
                    except self.InvalidCmd as error:
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
        
        to_delete = misc.glob('*', pjoin(self.me_dir, 'Events', run))
        to_delete += misc.glob('*', pjoin(self.me_dir, 'HTML', run))
        # forbid the banner to be removed
        to_delete = [os.path.basename(f) for f in to_delete if 'banner' not in f]
        if tag:
            to_delete = [f for f in to_delete if tag in f]
            if 'parton' in mode or 'all' in mode:
                try:
                    if self.results[run][0]['tag'] != tag:
                        raise Exception('dummy')
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
                    raise Exception('dummy')
            except Exception:
                pass
            else:
                to_delete = misc.glob('%s*' % run, pjoin(self.me_dir, 'SubProcesses'))
                to_delete += misc.glob(pjoin('*','%s*' % run), pjoin(self.me_dir, 'SubProcesses'))
                to_delete += misc.glob(pjoin('*','*','%s*' % run), pjoin(self.me_dir, 'SubProcesses'))

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
            to_delete = misc.glob('*', pjoin(self.me_dir, 'Events', run))
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
                    raise MadGraph5Error('''Some output still exists for this run. 
                Please remove those output first. Do for example: 
                remove %s all banner
                ''' % run)
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
            self.ask_edit_cards(['plot_card.dat'], args, plot=True)
                
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
        
        self.ask_edit_cards(['run_card.dat'], args, plot=False)
        self.run_card = banner_mod.RunCard(pjoin(self.me_dir, 'Cards', 'run_card.dat'))
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
                except SysCalcError as error:
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
 
        if not self.to_store:
            return 
        
        tag = self.run_card['run_tag']
        self.update_status('storing files of previous run', level=None,\
                                                     error=True)
        if 'event' in self.to_store:
            if not os.path.exists(pjoin(self.me_dir, 'Events',self.run_name, 'unweighted_events.lhe.gz')) and\
               os.path.exists(pjoin(self.me_dir, 'Events',self.run_name, 'unweighted_events.lhe')):
                logger.info("gzipping output file: unweighted_events.lhe")
                misc.gzip(pjoin(self.me_dir,'Events',self.run_name,"unweighted_events.lhe"))
            if os.path.exists(pjoin(self.me_dir,'Events','reweight.lhe')):
                os.remove(pjoin(self.me_dir,'Events', 'reweight.lhe'))
        
        if 'pythia' in self.to_store:
            self.update_status('Storing Pythia files of previous run', level='pythia', error=True)
            p = pjoin(self.me_dir,'Events')
            n = self.run_name
            t = tag
            self.to_store.remove('pythia')
            misc.gzip(pjoin(p,'pythia_events.hep'), 
                      stdout=pjoin(p, str(n),'%s_pythia_events.hep' % t),forceexternal=True)

        if 'pythia8' in self.to_store:
            p = pjoin(self.me_dir,'Events')
            n = self.run_name
            t = tag
            file_path = pjoin(p, n ,'%s_pythia8_events.hepmc'%t)
            self.to_store.remove('pythia8')
            if os.path.isfile(file_path):
                if 'removeHEPMC' in self.to_store:
                    os.remove(file_path)
                else:
                    self.update_status('Storing Pythia8 files of previous run', level='pythia', error=True)
                    if 'compressHEPMC' in self.to_store:
                        misc.gzip(file_path,stdout=file_path)
                        hepmc_fileformat = ".gz"

                    moveHEPMC_in_to_store = None
                    for to_store in self.to_store:
                        if "moveHEPMC" in to_store:
                           moveHEPMC_in_to_store = to_store

                    if not moveHEPMC_in_to_store == None:
                        move_hepmc_path = moveHEPMC_in_to_store.split("@")[1]
                        os.system("mv " + file_path + hepmc_fileformat + " " + move_hepmc_path)

        self.update_status('Done', level='pythia',makehtml=False,error=True)
        self.results.save()        
        
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
                raise MadGraph5Error('%s didn\'t stop properly. Stop all computation' % exe)


        elif mode in [1,2]:
            exename = os.path.basename(exe)
            # For condor cluster, create the input/output files
            if 'ajob' in exename: 
                input_files = ['madevent','input_app.txt','symfact.dat','iproc.dat','dname.mg',
                               pjoin(self.me_dir, 'SubProcesses','randinit')]
                if os.path.exists(pjoin(self.me_dir,'SubProcesses', 
                  'MadLoop5_resources.tar.gz')) and cluster.need_transfer(self.options):
                    input_files.append(pjoin(self.me_dir,'SubProcesses', 'MadLoop5_resources.tar.gz'))
                
                output_files = []
                required_output = []
                

                #Find the correct PDF input file
                input_files.append(self.get_pdf_input_filename())
                        
                #Find the correct ajob
                Gre = re.compile(r"\s*j=(G[\d\.\w]+)")
                origre = re.compile(r"grid_directory=(G[\d\.\w]+)")
                try : 
                    fsock = open(exe)
                except Exception:
                    fsock = open(pjoin(cwd,exe))
                text = fsock.read()
                output_files = Gre.findall(text)
                if not output_files:
                    Ire = re.compile(r"for i in ([\d\.\s]*) ; do")
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
            elif 'survey' in exename:
                input_files = ['madevent','input_app.txt','symfact.dat','iproc.dat', 'dname.mg',
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
            elif "refine_splitted.sh" in exename:
                input_files = ['madevent','symfact.dat','iproc.dat', 'dname.mg',
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
                self.cluster.submit(exe, argument=argument, stdout=stdout, cwd=cwd, **opt)
            

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
            except Exception as error:
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
            except KeyboardInterrupt as error:
                self.cluster.remove()
                raise                            
        
        

    ############################################################################   
    def configure_directory(self, html_opening=True):
        """ All action require before any type of run """   

        # Basic check
        assert os.path.exists(pjoin(self.me_dir,'SubProcesses'))

        # environmental variables to be included in make_opts
        self.make_opts_var = {}
        
        #see when the last file was modified
        time_mod = max([os.path.getmtime(pjoin(self.me_dir,'Cards','run_card.dat')),
                        os.path.getmtime(pjoin(self.me_dir,'Cards','param_card.dat'))])
        
        if self.configured >= time_mod and hasattr(self, 'random') and hasattr(self, 'run_card'):
            #just ensure that cluster specific are correctly handled
            if self.cluster:
                self.cluster.modify_interface(self)
            return
        else:
            self.configured = time_mod
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
        # lhapdf. not needed anymore the makefile handles it automaticallu
        #misc.compile(['clean4pdf'], cwd = pjoin(self.me_dir, 'Source'))
        
        self.make_opts_var['pdlabel1'] = ''
        self.make_opts_var['pdlabel2'] = ''
        if self.run_card['pdlabel1'] in ['eva', 'iww']:
            self.make_opts_var['pdlabel1'] = 'eva'
        if self.run_card['pdlabel2'] in ['eva', 'iww']:
            self.make_opts_var['pdlabel2'] = 'eva'
        if self.run_card['pdlabel1'] in ['edff','chff']:
            self.make_opts_var['pdlabel1'] = self.run_card['pdlabel1']
        if self.run_card['pdlabel2'] in ['edff','chff']:
            self.make_opts_var['pdlabel2'] = self.run_card['pdlabel2']


        # set  lhapdf.
        if self.run_card['pdlabel'] == "lhapdf":
            self.make_opts_var['lhapdf'] = 'True'
            self.link_lhapdf(pjoin(self.me_dir,'lib'))
            pdfsetsdir = self.get_lhapdf_pdfsetsdir()
            lhaid_list = [int(self.run_card['lhaid'])]
            self.copy_lhapdf_set(lhaid_list, pdfsetsdir)
        if self.run_card['pdlabel'] != "lhapdf":
            self.pdffile = None
            self.make_opts_var['lhapdf'] = ""

            # this is the case of collision with dressed leptons
            if abs(self.run_card['lpp1']) == abs(self.run_card['lpp2']) in [3,4]:
                # copy the files for the chosen density
                if self.run_card['pdlabel'] in  sum(self.run_card.allowed_lep_densities.values(),[]):
                    self.copy_lep_densities(self.run_card['pdlabel'], pjoin(self.me_dir, 'Source'))
                    self.make_opts_var['pdlabel1'] = 'ee'
                    self.make_opts_var['pdlabel2'] = 'ee'
        
        # set random number
        if self.run_card['iseed'] != 0:
            self.random = int(self.run_card['iseed'])
            self.run_card['iseed'] = 0
            # Reset seed in run_card to 0, to ensure that following runs
            # will be statistically independent
            self.run_card.write(pjoin(self.me_dir, 'Cards','run_card.dat'), template=pjoin(self.me_dir, 'Cards','run_card.dat'))
            time_mod = max([os.path.getmtime(pjoin(self.me_dir,'Cards','run_card.dat')),
                        os.path.getmtime(pjoin(self.me_dir,'Cards','param_card.dat'))])
            self.configured = time_mod
        elif os.path.exists(pjoin(self.me_dir,'SubProcesses','randinit')):
            for line in open(pjoin(self.me_dir,'SubProcesses','randinit')):
                data = line.split('=')
                assert len(data) ==2
                self.random = int(data[1])
                break
        else:
            self.random = random.randint(1, 30107)
        
        #set random seed for python part of the code
        if self.run_card['python_seed'] == -2: #-2 means same as run_card
            import random
            if not hasattr(random, 'mg_seedset'):
                random.seed(self.run_card['python_seed'])  
                random.mg_seedset = self.run_card['python_seed']  
        elif self.run_card['python_seed'] >= 0:
            import random
            if not hasattr(random, 'mg_seedset'):
                random.seed(self.run_card['python_seed'])  
                random.mg_seedset = self.run_card['python_seed']  
        if self.run_card['ickkw'] == 2:
            logger.info('Running with CKKW matching')
            self.treat_ckkw_matching()

        # add the make_opts_var to make_opts
        self.update_make_opts(self.run_card)
        # reset list of Gdirectory
        self.Gdirs = None
            
        # create param_card.inc and run_card.inc
        self.do_treatcards('')
        
        logger.info("compile Source Directory")
        
        # Compile
        for name in [ 'all']:#, '../bin/internal/combine_events']:
            self.compile(arg=[name], cwd=os.path.join(self.me_dir, 'Source'))

        force_subproc_clean = False

        bias_name = os.path.basename(self.run_card['bias_module'])
        if bias_name.lower()=='none':
            bias_name = 'dummy'

        # Always refresh the bias dependencies file
        if os.path.exists(pjoin(self.me_dir, 'SubProcesses','bias_dependencies')):
            os.remove(pjoin(self.me_dir, 'SubProcesses','bias_dependencies'))
        if os.path.exists(pjoin(self.me_dir, 'Source','BIAS',bias_name,'bias_dependencies')):
            files.ln(pjoin(self.me_dir, 'Source','BIAS',bias_name,'bias_dependencies'),
                                                        pjoin(self.me_dir, 'SubProcesses'))

        if self.proc_characteristics['bias_module']!=bias_name and \
             os.path.isfile(pjoin(self.me_dir, 'lib','libbias.a')):
                os.remove(pjoin(self.me_dir, 'lib','libbias.a'))
                force_subproc_clean = True

            
        # Finally compile the bias module as well
        if self.run_card['bias_module'] not in ['dummy',None]:
            logger.debug("Compiling the bias module '%s'"%bias_name)
            # Verify the compatibility of the specified module
            bias_module_valid = misc.Popen(['make','requirements'],
                       cwd=os.path.join(self.me_dir, 'Source','BIAS',bias_name),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')
            if 'VALID' not in str(bias_module_valid).upper() or \
               'INVALID' in str(bias_module_valid).upper():
                raise InvalidCmd("The bias module '%s' cannot be used because of:\n%s"%
                                                          (bias_name,bias_module_valid))
        
        self.compile(arg=[], cwd=os.path.join(self.me_dir, 'Source','BIAS',bias_name))
        self.proc_characteristics['bias_module']=bias_name
        # Update the proc_characterstics file
        self.proc_characteristics.write(
                   pjoin(self.me_dir,'SubProcesses','proc_characteristics'))

        if force_subproc_clean:
            # Make sure that madevent will be recompiled
            subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                'subproc.mg'))]
            for nb_proc,subdir in enumerate(subproc):
                Pdir = pjoin(self.me_dir, 'SubProcesses',subdir.strip())
                self.compile(['clean'], cwd=Pdir)

        #see when the last file was modified
        time_mod = max([os.path.getmtime(pjoin(self.me_dir,'Cards','run_card.dat')),
                        os.path.getmtime(pjoin(self.me_dir,'Cards','param_card.dat'))])

        self.configured = time_mod

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
    def get_Gdir(self, Pdir=None, symfact=None):
        """get the list of Gdirectory if not yet saved."""
        
        if hasattr(self, "Gdirs") and self.Gdirs:
            if self.me_dir in self.Gdirs[0]:
                if Pdir is None:
                    if not symfact:
                        return list(itertools.chain(*list(self.Gdirs[0].values())))
                    else:
                        return list(itertools.chain(*list(self.Gdirs[0].values()))), self.Gdirs[1]
                else:
                    if not symfact:
                        return self.Gdirs[0][Pdir]
                    else:
                        return self.Gdirs[0][Pdir], self.Gdirs[1]


        Pdirs = self.get_Pdir()
        Gdirs = {self.me_dir:[]}   
        mfactors = {}     
        for P in Pdirs:
            Gdirs[P] = []
            #for the next line do not use P, since in readonly mode it might not have symfact
            for line in open(pjoin(self.me_dir, 'SubProcesses',os.path.basename(P), "symfact.dat")):
                tag, mfactor = line.split()
                if int(mfactor) > 0:
                    Gdirs[P].append( pjoin(P, "G%s" % tag) )
                    mfactors[pjoin(P, "G%s" % tag)] = mfactor
        self.Gdirs = (Gdirs, mfactors)
        return self.get_Gdir(Pdir, symfact=symfact)
                
    ############################################################################
    def set_run_name(self, name, tag=None, level='parton', reload_card=False,
                     allow_new_tag=True):
        """define the run name, the run_tag, the banner and the results."""

        def get_last_tag(self, level):
            # Return the tag of the previous run having the required data for this
            # tag/run to working wel.
            if level == 'parton':
                return
            elif level in ['pythia','pythia8','madanalysis5_parton','madanalysis5_hadron']:
                return self.results[self.run_name][0]['tag']
            else:
                for i in range(-1,-len(self.results[self.run_name])-1,-1):
                    tagRun = self.results[self.run_name][i]
                    if tagRun.pythia or tagRun.shower or tagRun.pythia8 :
                        return tagRun['tag']
    
        
        # when are we force to change the tag new_run:previous run requiring changes
        upgrade_tag = {'parton': ['parton','pythia','pgs','delphes','madanalysis5_hadron','madanalysis5_parton', 'rivet'],
                       'pythia': ['pythia','pgs','delphes','madanalysis5_hadron'],
                       'pythia8': ['pythia8','pgs','delphes','madanalysis5_hadron', 'rivet'],
                       'pgs': ['pgs'],
                       'delphes':['delphes'],
                       'madanalysis5_hadron':['madanalysis5_hadron'],
                       'madanalysis5_parton':['madanalysis5_parton'],
                       'plot':[],
                       'syscalc':[],
                       'rivet':['rivet']}

        if name and name == self.run_name:        
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
            return get_last_tag(self, level)

        
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

        return get_last_tag(self, level)
            
    
    
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
            raise MadGraph5Error('Random seed too large ' + str(self.random) + ' > 30081*30081')
        if self.run_card['python_seed'] == -2: 
            import random
            if not hasattr(random, 'mg_seedset'):
                random.seed(self.random)  
                random.mg_seedset = self.random
            
    ############################################################################
    def save_random(self):
        """save random number in appropirate file"""

        fsock = open(pjoin(self.me_dir, 'SubProcesses','randinit'),'w')
        fsock.writelines('r=%s\n' % self.random)

    def do_quit(self, *args, **opts):
        
        return common_run.CommonRunCmd.do_quit(self, *args, **opts)
        #return CmdExtended.do_quit(self, *args, **opts)
        
    ############################################################################
    def treat_CKKW_matching(self):
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
                self.update_status('GENERATE SUDAKOV GRID', level='parton')
                
                for i in range(-2,6):
                    self.cluster.submit('%s/gensudgrid ' % self.dirbin, 
                    argument = ['%d'%i],
                    cwd=self.me_dir, 
                    stdout=open(pjoin(self.me_dir, 'gensudgrid%s.log' % i),'w'))
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
        totar = False
        torm = False
        if input.endswith('.gz'):
            if not os.path.exists(input) and os.path.exists(input[:-3]):
                totar = True
                input = input[:-3]
            else:
                misc.gunzip(input, keep=True)
                totar = False
                torm = True
                input = input[:-3]
            
        try:
            misc.call(['%s/ExRootLHEFConverter' % eradir, 
                             input, output],
                            cwd=pjoin(self.me_dir, 'Events'))
        except Exception:
            logger.warning('fail to produce Root output [problem with ExRootAnalysis]')
    
        if totar:
            if os.path.exists('%s.gz' % input):
                try:
                    os.remove('%s.gz' % input)
                except:
                    pass
            else:
                misc.gzip(input)
        if torm:
            os.remove(input)
    
    def run_syscalc(self, mode='parton', event_path=None, output=None):
        """create the syscalc output""" 

        if self.run_card['use_syst'] not in self.true:
            return
        
        scdir = self.options['syscalc_path']
        if not scdir or not os.path.exists(scdir):
            return

        if self.run_card['event_norm'] != 'sum':
            logger.critical('SysCalc works only when event_norm is on \'sum\'.')
            return
        logger.info('running SysCalc on mode %s' % mode)    
    
        # Restore the old default for SysCalc+PY6
        if self.run_card['sys_matchscale']=='auto':
            self.run_card['sys_matchscale'] = "30 50"
    
        # Check that all pdfset are correctly installed
        lhaid = [self.run_card.get_lhapdf_id()]
        if '&&' in self.run_card['sys_pdf']:
            line = ' '.join(self.run_card['sys_pdf'])
            sys_pdf = line.split('&&')
            lhaid += [l.split()[0] for l in sys_pdf]
        else:
            lhaid += [l for l in self.run_card['sys_pdf'].split() if not l.isdigit() or int(l) > 500]
        try:
            pdfsets_dir = self.get_lhapdf_pdfsetsdir()
        except Exception as error:
            logger.debug(str(error))
            logger.warning('Systematic computation requires lhapdf to run. Bypass SysCalc')
            return
        
        # Copy all the relevant PDF sets
        [self.copy_lhapdf_set([onelha], pdfsets_dir) for onelha in lhaid]
        
        to_syscalc={'sys_scalefact': self.run_card['sys_scalefact'],
                    'sys_alpsfact': self.run_card['sys_alpsfact'],
                    'sys_matchscale': self.run_card['sys_matchscale'],
                    'sys_scalecorrelation': self.run_card['sys_scalecorrelation'],
                    'sys_pdf': self.run_card['sys_pdf']}
        
        tag = self.run_card['run_tag']  
        card = pjoin(self.me_dir, 'bin','internal', 'syscalc_card.dat')
        template = open(pjoin(self.me_dir, 'bin','internal', 'syscalc_template.dat')).read()
        
        if '&&' in to_syscalc['sys_pdf']:
            to_syscalc['sys_pdf'] = to_syscalc['sys_pdf'].split('#',1)[0].replace('&&',' \n ')
        else:
            data = to_syscalc['sys_pdf'].split()
            new = []
            for d in data:
                if not d.isdigit():
                    new.append(d)
                elif int(d) > 500:
                    new.append(d)
                else:
                    new[-1] += ' %s' % d
            to_syscalc['sys_pdf'] = '\n'.join(new)  
        
        if to_syscalc['sys_pdf'].lower() in ['', 'f', 'false', 'none', '.false.']:
            to_syscalc['sys_pdf'] = ''
        if to_syscalc['sys_alpsfact'].lower() in ['', 'f', 'false', 'none','.false.']:
            to_syscalc['sys_alpsfact'] = ''



        
        # check if the scalecorrelation parameter is define:
        if not 'sys_scalecorrelation' in self.run_card:
            self.run_card['sys_scalecorrelation'] = -1
        open(card,'w').write(template % self.run_card)
        
        if not os.path.exists(card):
            return False

        
        
        event_dir = pjoin(self.me_dir, 'Events')

        if not event_path:
            if mode == 'parton':
                event_path = pjoin(event_dir,self.run_name, 'unweighted_events.lhe')
                if not (os.path.exists(event_path) or os.path.exists(event_path+".gz")):
                    event_path = pjoin(event_dir, 'unweighted_events.lhe')
                output = pjoin(event_dir, 'syscalc.lhe')
                stdout = open(pjoin(event_dir, self.run_name, '%s_systematics.log' % (mode)),'w')
            elif mode == 'Pythia':
                stdout = open(pjoin(event_dir, self.run_name, '%s_%s_syscalc.log' % (tag,mode)),'w')
                if 'mgpythiacard' in self.banner:
                    pat = re.compile(r'''^\s*qcut\s*=\s*([\+\-\d.e]*)''', re.M+re.I)
                    data = pat.search(self.banner['mgpythiacard'])
                    if data:
                        qcut = float(data.group(1))
                        xqcut = abs(self.run_card['xqcut'])
                        for value in self.run_card['sys_matchscale'].split():
                            if float(value) < qcut:
                                raise SysCalcError('qcut value for sys_matchscale lower than qcut in pythia_card. Bypass syscalc')
                            if float(value) < xqcut:
                                raise SysCalcError('qcut value for sys_matchscale lower than xqcut in run_card. Bypass syscalc')
                        
                        
                event_path = pjoin(event_dir,'syst.dat')
                output = pjoin(event_dir, 'syscalc.dat')
            else:
                raise self.InvalidCmd('Invalid mode %s' % mode)
            
        if not os.path.exists(event_path):
            if os.path.exists(event_path+'.gz'):
                misc.gunzip(event_path+'.gz')
            else:
                raise SysCalcError('Events file %s does not exits' % event_path)
        
        self.update_status('Calculating systematics for %s level' % mode, level = mode.lower())
        try:
            proc = misc.call([os.path.join(scdir, 'sys_calc'),
                               event_path, card, output],
                            stdout = stdout,
                            stderr = subprocess.STDOUT,
                            cwd=event_dir)
            # Wait 5 s to make sure file is finished writing
            time.sleep(5)            
        except OSError as error:
            logger.error('fail to run syscalc: %s. Please check that SysCalc is correctly installed.' % error)
        else:
            if not os.path.exists(output):
                logger.warning('SysCalc Failed. Please read the associate log to see the reason. Did you install the associate PDF set?')
            elif mode == 'parton':
                files.mv(output, event_path)
                
        self.update_status('End syscalc for %s level' % mode, level = mode.lower(),
                                                                 makehtml=False)
        
        return True   


    action_switcher = AskRun
    ############################################################################
    def ask_run_configuration(self, mode=None, args=[]):
        """Ask the question when launching generate_events/multi_run"""

        passing_cmd = []
        if '-R' in args or '--reweight' in args:
            passing_cmd.append('reweight=ON')
        if '-M' in args or '--madspin' in args:
            passing_cmd.append('madspin=ON')
        
        switch, cmd_switch = self.ask('', '0', [], ask_class = self.action_switcher,
                              mode=mode, line_args=args, force=self.force,
                              first_cmd=passing_cmd, return_instance=True)
        #
        self.switch = switch # store the value of the switch for plugin purpose 
        if 'dynamical' in switch:
            mode = 'auto'
        
        # Now that we know in which mode we are check that all the card
        #exists (copy default if needed)
    
        cards = ['param_card.dat', 'run_card.dat']
        if switch['shower'] == 'Pythia6':
            cards.append('pythia_card.dat')
        if switch['shower'] == 'Pythia8':
            cards.append('pythia8_card.dat')            
        if switch['detector'] in  ['PGS','DELPHES+PGS']:
            cards.append('pgs_card.dat')
        if switch['detector'] in ['Delphes', 'DELPHES+PGS']:
            cards.append('delphes_card.dat')
            delphes3 = True
            if os.path.exists(pjoin(self.options['delphes_path'], 'data')):
                delphes3 = False
                cards.append('delphes_trigger.dat')
        if switch['madspin'] != 'OFF':
            cards.append('madspin_card.dat')
        if switch['reweight'] != 'OFF':
            cards.append('reweight_card.dat')
        if switch['analysis'].upper() in ['MADANALYSIS5']:
            cards.append('madanalysis5_parton_card.dat')
        if switch['analysis'].upper() in ['MADANALYSIS5'] and not switch['shower']=='OFF':
            cards.append('madanalysis5_hadron_card.dat')
        elif switch['analysis'].upper() in ['MADANALYSIS4']:
            cards.append('plot_card.dat')
        elif switch['analysis'].upper() in ['RIVET']:
            cards.append('rivet_card.dat')

        self.keep_cards(cards)
        
        first_cmd = cmd_switch.get_cardcmd()
        
        if os.path.isfile(pjoin(self.me_dir,'Cards','MadLoopParams.dat')):
            cards.append('MadLoopParams.dat')
        
        if self.force:
            self.check_param_card(pjoin(self.me_dir,'Cards','param_card.dat' ))
            return switch
        

        if 'dynamical' in switch and switch['dynamical']:
            self.ask_edit_cards(cards, plot=False, mode='auto', first_cmd=first_cmd)
        else:
            self.ask_edit_cards(cards, plot=False, first_cmd=first_cmd)
        return switch
    
    ############################################################################
    def ask_pythia_run_configuration(self, mode=None, pythia_version=6, banner=None):
        """Ask the question when launching pythia"""
        
        pythia_suffix = '' if pythia_version==6 else '%d'%pythia_version
        
        available_mode = ['0', '1']
        if pythia_version==6:
            available_mode.append('2')
        if self.options['delphes_path']:
                available_mode.append('3')
        name = {'0': 'auto', '2':'pgs', '3':'delphes'}
        name['1'] = 'pythia%s'%pythia_suffix
        options = available_mode + [name[val] for val in available_mode]
        question = """Which programs do you want to run?
    0 / auto    : running existing cards\n"""
        if pythia_version==6:
            question += """    1. pythia  : Pythia\n"""
            question += """    2. pgs     : Pythia + PGS\n"""
        else:
            question += """    1. pythia8  : Pythia8\n"""

        if '3' in available_mode:
            question += """    3. delphes  : Pythia%s + Delphes.\n"""%pythia_suffix

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
            if pythia_version==6 and os.path.exists(pjoin(self.me_dir,
                                                      'Cards', 'pgs_card.dat')):
                mode = 'pgs'
            elif os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_card.dat')):
                mode = 'delphes'
            else: 
                mode = 'pythia%s'%pythia_suffix
        logger.info('Will run in mode %s' % mode)
        # Now that we know in which mode we are check that all the card
        #exists (copy default if needed) remove pointless one
        cards = ['pythia%s_card.dat'%pythia_suffix]
        if mode == 'pgs' and pythia_version==6:
            cards.append('pgs_card.dat')
        if mode == 'delphes':
            cards.append('delphes_card.dat')
            delphes3 = True
            if os.path.exists(pjoin(self.options['delphes_path'], 'data')):
                delphes3 = False
                cards.append('delphes_trigger.dat')
        self.keep_cards(cards, ignore=['madanalysis5_parton_card.dat','madanalysis5_hadron_card.dat',
                      'plot_card.dat'])
        
        if self.force:
            return mode
        
        if not banner:
            banner = self.banner
            
        if auto:
            self.ask_edit_cards(cards, from_banner=['param', 'run'], 
                                mode='auto', plot=(pythia_version==6), banner=banner
                                )
        else:
            self.ask_edit_cards(cards, from_banner=['param', 'run'],
                                 plot=(pythia_version==6), banner=banner)

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
            particles = re.search(r"/([\d,-]+)/", line)
            all_ids.append([int(p) for p in particles.group(1).split(',')])
        return all_ids
    
    
#===============================================================================                                                                              
class GridPackCmd(MadEventCmd):
    """The command for the gridpack --Those are not suppose to be use interactively--"""

    def __init__(self, me_dir = None, nb_event=0, seed=0, gran=-1, nprocs=1, maxevts=2500, *completekey, **stdin):
        """Initialize the command and directly run"""

        # Initialize properly
        self.readonly = False
        MadEventCmd.__init__(self, me_dir, *completekey, **stdin)
        self.run_mode = 0
        self.random = seed
        self.random_orig = self.random
        self.granularity = gran
        self.nprocs = nprocs
        self.maxevts = maxevts
        
        self.options['automatic_html_opening'] = False
        #write the grid_card.dat on disk
        self.nb_event = int(nb_event)
        self.write_gridcard(nb_event, seed, gran) # set readonly on True if needed
        self.prepare_local_dir()                  # move to gridpack dir or create local structure
        # Now it's time to run!
        if me_dir and nb_event and seed:
            self.launch(nb_event, seed)
        else:
            raise MadGraph5Error('Gridpack run failed: ' + str(me_dir) + str(nb_event) + \
                  str(seed))


    def update_status(self, *args, **opts):
        return

    def load_results_db(self):
        """load the current results status"""
        model = self.find_model_name()
        process = self.process # define in find_model_name
        self.results = gen_crossxhtml.AllResults(model, process, self.me_dir)
        self.last_mode=''

    def save_random(self):
        """save random number in appropirate file"""

        if not self.readonly:
            fsock = open(pjoin(self.me_dir, 'SubProcesses','randinit'),'w')
        else:
            fsock = open('randinit','w')
        fsock.writelines('r=%s\n' % self.random)

    def write_RunWeb(self, me_dir):
        try:
            super(GridPackCmd, self).write_RunWeb(me_dir)
        except IOError:
            self.readonly  =True

    def write_gridcard(self, nb_event, seed, gran):
        """write the grid_card.dat file at appropriate location"""
        
        # first try to write grid_card within the gridpack.
        print("WRITE GRIDCARD", self.me_dir)
        if self.readonly:
            if not os.path.exists('Cards'):
                os.mkdir('Cards')
            fsock = open('grid_card.dat','w')
        else:
            fsock = open(pjoin(self.me_dir, 'Cards', 'grid_card.dat'),'w')
                
        gridpackcard = banner_mod.GridpackCard()
        gridpackcard['GridRun'] = True
        gridpackcard['gevents'] = nb_event
        gridpackcard['gseed'] = seed
        gridpackcard['ngran'] = gran
        
        gridpackcard.write(fsock)

    ############################################################################
    def get_Pdir(self):
        """get the list of Pdirectory if not yet saved."""
        
        if hasattr(self, "Pdirs"):
            if self.me_dir in self.Pdirs[0]:
                return self.Pdirs
            
        if not self.readonly:
            self.Pdirs = [pjoin(self.me_dir, 'SubProcesses', l.strip()) 
                     for l in open(pjoin(self.me_dir,'SubProcesses', 'subproc.mg'))]
        else:
            self.Pdirs = [l.strip() 
                     for l in open(pjoin(self.me_dir,'SubProcesses', 'subproc.mg'))] 
          
        return self.Pdirs
        
    def prepare_local_dir(self):
        """create the P directory structure in the local directory"""
        
        if not self.readonly:
            os.chdir(self.me_dir)
        else:
            for line in open(pjoin(self.me_dir,'SubProcesses','subproc.mg')):
                p = line.strip()
                os.mkdir(p)
                files.cp(pjoin(self.me_dir,'SubProcesses',p,'symfact.dat'),
                         pjoin(p, 'symfact.dat'))
            

    def launch(self, nb_event, seed):
        """ launch the generation for the grid """

        # 1) Restore the default data
        logger.info('generate %s events' % nb_event)
        self.set_run_name('GridRun_%s' % seed)
        if not self.readonly:
            self.update_status('restoring default data', level=None)
            misc.call([pjoin(self.me_dir,'bin','internal','restore_data'),
                         'default'], cwd=self.me_dir)

        if self.run_card['python_seed'] == -2:
            import random
            if not hasattr(random, 'mg_seedset'):
                random.seed(seed)  
                random.mg_seedset = seed
        elif self.run_card['python_seed'] > 0:
            import random
            if not hasattr(random, 'mg_seedset'):
                random.seed(self.run_card['python_seed'])  
                random.mg_seedset = self.run_card['python_seed']         
        # 2) Run the refine for the grid
        self.update_status('Generating Events', level=None)
        #misc.call([pjoin(self.me_dir,'bin','refine4grid'),
        #                str(nb_event), '0', 'Madevent','1','GridRun_%s' % seed],
        #                cwd=self.me_dir)
        self.gridpack_cross = self.refine4grid(nb_event)

        # 3) Combine the events/pythia/...
        self.exec_cmd('combine_events')
        if not self.readonly:
            self.exec_cmd('store_events')
            self.print_results_in_shell(self.results.current)
            if self.run_card['systematics_program'] == 'systematics' and self.run_card['use_syst']:
                self.exec_cmd('systematics %s --from_card' % self.run_name,
                                               postcmd=False,printcmd=False)
            self.exec_cmd('decay_events -from_cards', postcmd=False)
            if self.run_card['time_of_flight']>=0:
                    self.exec_cmd("add_time_of_flight --threshold=%s" % self.run_card['time_of_flight'] ,postcmd=False)
        else:
            path = pjoin('Events', self.run_name, 'unweighted_events.lhe.gz')
            if self.run_card['use_syst'] and self.run_card['systematics_program'] == 'systematics':
                self.options['nb_core']  = 1
                self.exec_cmd('systematics %s --from_card' % path, postcmd=False, printcmd=False)
            if self.run_card['time_of_flight']>=0:
                self.exec_cmd("add_time_of_flight %s --threshold=%s" % (path, self.run_card['time_of_flight']) ,postcmd=False)
            

    def refine4grid(self, nb_event):
        """Special refine for gridpack run."""
        self.nb_refine += 1
        
        precision = nb_event

        across= self.make_make_all_html_results(get_attr='axsec')

        self.opts = dict([(key,value[1]) for (key,value) in \
                          self._survey_options.items()])
        
        # initialize / remove lhapdf mode
        # self.configure_directory() # All this has been done before
        self.cluster_mode = 0 # force single machine

        # Store seed in randinit file, to be read by ranmar.f
        self.save_random()
        
        self.update_status('Refine results to %s' % precision, level=None)
        logger.info("Using random number seed offset = %s" % self.random)

        refine_opt = {'err_goal': nb_event, 'split_channels': True,
                      'ngran':self.granularity, 'readonly': self.readonly,
                      'nprocs': self.nprocs, 'maxevts': self.maxevts}
        x_improve = gen_ximprove.gen_ximprove_gridpack(self, refine_opt)
        x_improve.launch() # create the ajob for the refinment and run those!
        self.gscalefact = x_improve.gscalefact #store jacobian associate to the gridpack 
        
        
        #bindir = pjoin(os.path.relpath(self.dirbin, pjoin(self.me_dir,'SubProcesses')))
        #print 'run combine!!!'
        #combine_runs.CombineRuns(self.me_dir)
        
        return across
        #update html output
        Presults = sum_html.collect_result(self)
        cross, error = Presults.xsec, Presults.xerru
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)
        
        
        #self.update_status('finish refine', 'parton', makehtml=False)
        #devnull.close()
        
        
        
        return
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
            for match in misc.glob('*ajob*', Pdir):
                if os.path.basename(match)[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(pjoin(Pdir, match))
            

            logfile = pjoin(Pdir, 'gen_ximprove.log')
            misc.call([pjoin(bindir, 'gen_ximprove')],
                                    stdin=subprocess.PIPE,
                                    stdout=open(logfile,'w'),
                                    cwd=Pdir)

            if os.path.exists(pjoin(Pdir, 'ajob1')):
                alljobs = misc.glob('ajob*', Pdir)
                nb_tot = len(alljobs)            
                self.total_jobs += nb_tot
                for i, job in enumerate(alljobs):
                    job = os.path.basename(job)
                    self.launch_job('%s' % job, cwd=Pdir, remaining=(nb_tot-i-1), 
                             run_type='Refine number %s on %s (%s/%s)' %
                             (self.nb_refine, subdir, nb_proc+1, len(subproc)))
                    if os.path.exists(pjoin(self.me_dir,'error')):
                        self.monitor(html=True)
                        raise MadEventError('Error detected in dir %s: %s' % \
                            (Pdir, open(pjoin(self.me_dir,'error')).read()))
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
        cross, error = self.make_make_all_html_results()
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)
        
        
        self.update_status('finish refine', 'parton', makehtml=False)
        devnull.close()

    def do_combine_events(self, line):
        """Advanced commands: Launch combine events""" 

        if self.readonly:
            outdir = 'Events'
            if not os.path.exists(outdir):
                os.mkdir(outdir)
        else:
            outdir = pjoin(self.me_dir, 'Events')
        args = self.split_arg(line)
        # Check argument's validity
        self.check_combine_events(args)
        gscalefact = self.gscalefact # {(C.get('name')): jac}
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
        
        
        if not os.path.exists(pjoin(outdir, self.run_name)):
                os.mkdir(pjoin(outdir, self.run_name))
        self.banner.write(pjoin(outdir, self.run_name, 
                                '%s_%s_banner.txt' % (self.run_name, tag)))
        
        get_wgt = lambda event: event.wgt            
        AllEvent = lhe_parser.MultiEventFile()
        AllEvent.banner = self.banner
        
        partials = 0 # if too many file make some partial unweighting
        sum_xsec, sum_xerru, sum_axsec = 0,[],0
        partials_info = []
        Gdirs = self.get_Gdir()
        Gdirs.sort()
        for Gdir in Gdirs:
            #mfactor already taken into accoun in auto_dsig.f
            if os.path.exists(pjoin(Gdir, 'events.lhe')):
                result = sum_html.OneResult('')
                result.read_results(pjoin(Gdir, 'results.dat'))
                AllEvent.add(pjoin(Gdir, 'events.lhe'), 
                             result.get('xsec')*gscalefact[Gdir],
                             result.get('xerru')*gscalefact[Gdir],
                             result.get('axsec')*gscalefact[Gdir]
                             )

                sum_xsec += result.get('xsec')*gscalefact[Gdir]
                sum_xerru.append(result.get('xerru')*gscalefact[Gdir])
                sum_axsec += result.get('axsec')*gscalefact[Gdir]
                
                if len(AllEvent) >= 80: #perform a partial unweighting
                    if not self.results.current.get('axsec'):
                        if self.run_card['gridpack'] and self.gridpack_cross:
                            nb_event = min(abs(1.05*self.nb_event*sum_axsec/self.gridpack_cross),self.nb_event)
                        else:
                            nb_event= self.nb_event
                    else:
                        nb_event = min(abs(1.01*self.nb_event*sum_axsec/self.results.current.get('axsec')),self.run_card['nevents'], self.nb_event, self.gridpack_cross, sum_axsec)
                    misc.sprint(nb_event, self.results.current.get('axsec'), self.gridpack_cross)
                    AllEvent.unweight(pjoin(outdir, self.run_name, "partials%s.lhe.gz" % partials),
                          get_wgt, log_level=5,  trunc_error=1e-2, event_target=nb_event)
                    AllEvent = lhe_parser.MultiEventFile()
                    AllEvent.banner = self.banner
                    partials_info.append((pjoin(outdir, self.run_name, "partials%s.lhe.gz" % partials),
                                 sum_xsec,
                                 math.sqrt(sum(x**2 for x in sum_xerru)),
                                 sum_axsec) )  
                    sum_xsec, sum_xerru, sum_axsec = 0,[],0
                    partials +=1
        
        for data in partials_info:
            AllEvent.add(*data)
            sum_xsec += data[1]

        if not hasattr(self,'proc_characteristic'):
            self.proc_characteristic = self.get_characteristics()
        
        self.banner.add_generation_info(sum_xsec, self.nb_event)
        nb_event = AllEvent.unweight(pjoin(outdir, self.run_name, "unweighted_events.lhe.gz"),
                          get_wgt, trunc_error=1e-2, event_target=self.nb_event,
                          log_level=logging.DEBUG, normalization=self.run_card['event_norm'],
                          proc_charac=self.proc_characteristic)
        
        
        if partials:
            for i in range(partials):
                try:
                    os.remove(pjoin(outdir, self.run_name, "partials%s.lhe.gz" % i))
                except Exception:
                    os.remove(pjoin(outdir, self.run_name, "partials%s.lhe" % i))
                   
        self.results.add_detail('nb_event', nb_event)
        self.banner.add_generation_info(sum_xsec, nb_event)
        if self.run_card['bias_module'].lower() not in  ['dummy', 'none']:
            self.correct_bias()


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
        retcode = misc.compile(arg=['-j1','check'], cwd=dir_name, nb_core=1)
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
                directories = [d for d in misc.glob('P*_*', dir_path) \
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
            raise MadGraph5Error('At least one of [SubProc_dir,run_dir] must'+\
                                           ' be provided in run_initialization.')
        
        # If the user does not specify where is check_sa.f, then it is assumed
        # to be one levels above run_dir
        if SubProc_dir is None:
            SubProc_dir = os.path.abspath(pjoin(run_dir,os.pardir))
            
        if run_dir is None:
            directories =[ dir for dir in misc.glob('P[0-9]*', SubProc_dir)
                                                         if os.path.isdir(dir) ]
            if directories:
                run_dir = directories[0]
            else:
                raise MadGraph5Error('Could not find a valid running directory'+\
                                                      ' in %s.'%str(SubProc_dir))

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
            raise MadGraph5Error('Could not find MadLoopParams.dat at %s.'\
                                                                     %MLCardPath)
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
            if 'Process_compilation' not in list(infos.keys()) or \
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
            raise MadGraph5Error('Could not find MadLoopParams.dat at %s.'\
                                                                     %MLCardPath)        
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
        if os.path.exists(pjoin(proc_dir,'Source','CutTools')):
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
            if interface and  hasattr(interface, 'cluster') and isinstance(interface.cluster, cluster.MultiCore):
                mcore = interface.cluster
            else: 
                mcore = cluster.MultiCore(**MG_options)
        else:
            mcore = cluster.onecore

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
                raise MadGraph5Error('Failed the initialization of'+\
                  " loop-induced matrix element '%s'%s."%\
                  (os.path.basename(v_folder),' (using default n_PS points)' if\
                    n_PS is None else ' (trying with a maximum of %d PS points)'\
                                                               %(max_mult*n_PS)))
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


if '__main__' == __name__:
    # Launch the interface without any check if one code is already running.
    # This can ONLY run a single command !!
    import sys
    if sys.version_info < (3, 7):
        sys.exit('MadGraph/MadEvent 5 works only with python 3.7 or later).\n'+\
               'Please upgrate your version of python.')

    import os
    import optparse
    # Get the directory of the script real path (bin)                                                                                                                                                           
    # and add it to the current PYTHONPATH                                                                                                                                                                      
    #root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath( __file__ ))))
    sys.path.insert(0, root_path)

    class MyOptParser(optparse.OptionParser):    
        class InvalidOption(Exception): pass
        def error(self, msg=''):
            raise MyOptParser.InvalidOption(msg)
    # Write out nice usage message if called with -h or --help                                                                                                                                                  
    usage = "usage: %prog [options] [FILE] "
    parser = MyOptParser(usage=usage)
    parser.add_option("-l", "--logging", default='INFO',
                      help="logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL) [%default]")
    parser.add_option("","--web", action="store_true", default=False, dest='web', \
                     help='force toce to be in secure mode')
    parser.add_option("","--debug", action="store_true", default=False, dest='debug', \
                     help='force to launch debug mode')
    parser_error = ''
    done = False
    
    for i in range(len(sys.argv)-1):
        try:
            (options, args) = parser.parse_args(sys.argv[1:len(sys.argv)-i])
            done = True
        except MyOptParser.InvalidOption as error:
            pass
        else:
            args += sys.argv[len(sys.argv)-i:]
    if not done:
        # raise correct error:                                                                                                                                                                                  
        try:
            (options, args) = parser.parse_args()
        except MyOptParser.InvalidOption as error:
            print(error)
            sys.exit(2)

    if len(args) == 0:
        args = ''

    import subprocess
    import logging
    import logging.config
    # Set logging level according to the logging level given by options                                                                                                                                         
    #logging.basicConfig(level=vars(logging)[options.logging])                                                                                                                                                  
    import internal
    import internal.coloring_logging
    # internal.file = XXX/bin/internal/__init__.py
    # => need three dirname to get XXX
    # we use internal to have any issue with pythonpath finding the wrong file
    me_dir = os.path.dirname(os.path.dirname(os.path.dirname(internal.__file__)))
    print("me_dir is", me_dir)
    try:
        if __debug__ and options.logging == 'INFO':
            options.logging = 'DEBUG'
        if options.logging.isdigit():
            level = int(options.logging)
        else:
            level = eval('logging.' + options.logging)
        log_path = os.path.join(me_dir, 'bin', 'internal', 'me5_logging.conf')
        logging.config.fileConfig(log_path)
        logging.root.setLevel(level)
        logging.getLogger('madgraph').setLevel(level)
    except:
        raise
        pass

    # Call the cmd interface main loop                                                                                                                                                                          
    try:
        if args:
            # a single command is provided
            if '--web' in args:
                i = args.index('--web') 
                args.pop(i)                                                                                                                                                                     
                cmd_line = MadEventCmd(me_dir, force_run=True)
            else:
                cmd_line = MadEventCmdShell(me_dir, force_run=True)
            if not hasattr(cmd_line, 'do_%s' % args[0]):
                if parser_error:
                    print(parser_error)
                    print('and %s  can not be interpreted as a valid command.' % args[0])
                else:
                    print('ERROR: %s  not a valid command. Please retry' % args[0])
            else:
                cmd_line.use_rawinput = False
                cmd_line.run_cmd(' '.join(args))
                cmd_line.run_cmd('quit')

    except KeyboardInterrupt:
        print('quit on KeyboardInterrupt')
        pass


    
    
    
    
    
    
    
    


