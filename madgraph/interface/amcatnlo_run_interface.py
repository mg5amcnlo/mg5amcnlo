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

from __future__ import absolute_import
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
import tarfile
import copy
import datetime
import tarfile
import traceback
import six
StringIO = six
from six.moves import range
from six.moves import zip
try:
    import cpickle as pickle
except:
    import pickle

try:
    import readline
    GNU_SPLITTING = ('GNU' in readline.__doc__)
except:
    GNU_SPLITTING = True

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.split(root_path)[0]
sys.path.insert(0, os.path.join(root_path,'bin'))

__maxint__ = 2**31 - 1

# usefull shortcut
pjoin = os.path.join
# Special logger for the Cmd Interface
logger = logging.getLogger('madgraph.stdout') # -> stdout
logger_stderr = logging.getLogger('madgraph.stderr') # ->stderr
 
try:
    import madgraph
except ImportError: 
    aMCatNLO = True 
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
    import internal.FO_analyse_card as analyse_card 
    import internal.lhe_parser as lhe_parser
else:
    # import from madgraph directory
    aMCatNLO = False
    import madgraph.interface.extended_cmd as cmd
    import madgraph.interface.common_run_interface as common_run
    import madgraph.iolibs.files as files
    import madgraph.iolibs.save_load_object as save_load_object
    import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
    import madgraph.madevent.sum_html as sum_html
    import madgraph.various.banner as banner_mod
    import madgraph.various.cluster as cluster
    import madgraph.various.misc as misc
    import madgraph.various.shower_card as shower_card
    import madgraph.various.FO_analyse_card as analyse_card
    import madgraph.various.lhe_parser as lhe_parser
    from madgraph import InvalidCmd, aMCatNLOError, MadGraph5Error,MG5DIR

class aMCatNLOError(Exception):
    pass


def compile_dir(*arguments):
    """compile the direcory p_dir
    arguments is the tuple (me_dir, p_dir, mode, options, tests, exe, run_mode)
    this function needs not to be a class method in order to do
    the compilation on multicore"""

    if len(arguments) == 1:
        (me_dir, p_dir, mode, options, tests, exe, run_mode) = arguments[0]
    elif len(arguments)==7:
        (me_dir, p_dir, mode, options, tests, exe, run_mode) = arguments
    else:
        raise aMCatNLOError('Wrong number of arguments')
    logger.info(' Compiling %s...' % p_dir)

    this_dir = pjoin(me_dir, 'SubProcesses', p_dir) 

    try:
        #compile everything
        # compile and run tests
        for test in tests:
            # skip check_poles for LOonly dirs
            if test == 'check_poles' and os.path.exists(pjoin(this_dir, 'parton_lum_0.f')):
                continue
            if test == 'test_ME' or test == 'test_MC':
                test_exe='test_soft_col_limits'
            else:
                test_exe=test
            misc.compile([test_exe], cwd = this_dir, job_specs = False)
            input = pjoin(me_dir, '%s_input.txt' % test)
            #this can be improved/better written to handle the output
            misc.call(['./%s' % (test_exe)], cwd=this_dir, 
                    stdin = open(input), stdout=open(pjoin(this_dir, '%s.log' % test), 'w'),
                    close_fds=True)
            if test == 'check_poles' and os.path.exists(pjoin(this_dir,'MadLoop5_resources')) :
                tf=tarfile.open(pjoin(this_dir,'MadLoop5_resources.tar.gz'),'w:gz',
                                                                 dereference=True)
                tf.add(pjoin(this_dir,'MadLoop5_resources'),arcname='MadLoop5_resources')
                tf.close()
            
        if not options['reweightonly']:
            misc.compile(['gensym'], cwd=this_dir, job_specs = False)
            misc.call(['./gensym', mode],cwd= this_dir,
                     stdout=open(pjoin(this_dir, 'gensym.log'), 'w'),
                     close_fds=True) 
            #compile madevent_mintMC/mintFO
            misc.compile([exe], cwd=this_dir, job_specs = False)
        if mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO']:
            misc.compile(['reweight_xsec_events'], cwd=this_dir, job_specs = False)

        logger.info('    %s done.' % p_dir) 
        return 0
    except MadGraph5Error as msg:
        raise aMCatNLOError(msg)


def check_compiler(options, block=False):
    """check that the current fortran compiler is gfortran 4.6 or later.
    If block, stops the execution, otherwise just print a warning"""

    msg = 'In order to be able to run MadGraph5_aMC@NLO at NLO, you need to have ' + \
            'gfortran 4.6 or later installed.\n%s has been detected.\n'+\
            'Note that you can still run MadEvent (at LO) without any problem!'
    #first check that gfortran is installed
    if options['fortran_compiler']:
        compiler = options['fortran_compiler']
    elif misc.which('gfortran'):
        compiler = 'gfortran'
    else: 
        compiler = ''
        
    if 'gfortran' not in compiler:
        if block:
            raise aMCatNLOError(msg % compiler)
        else:
            logger.warning(msg % compiler)
    else:
        curr_version = misc.get_gfortran_version(compiler)
        curr_version = curr_version.split('.')
        if len(curr_version) == 1:
            curr_version.append(0)

        if int(curr_version[0]) < 5:
            if int(curr_version[0]) == 4 and int(curr_version[1]) > 5:
                return
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
    error_debug = 'Please report this bug on https://bugs.launchpad.net/mg5amcnlo\n'
    error_debug += 'More information is found in \'%(debug)s\'.\n' 
    error_debug += 'Please attach this file to your report.'

    config_debug = 'If you need help with this issue, please, contact us on https://answers.launchpad.net/mg5amcnlo\n'


    keyboard_stop_msg = """stopping all operation
            in order to quit MadGraph5_aMC@NLO please enter exit"""
    
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
        '#*                    MadGraph5_aMC@NLO                     *\n' + \
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
        "#*                           and                            *\n" + \
        "#*                http://amcatnlo.cern.ch                   *\n" + \
        '#*                                                          *\n' + \
        '#************************************************************\n' + \
        '#*                                                          *\n' + \
        '#*               Command File for aMCatNLO                  *\n' + \
        '#*                                                          *\n' + \
        '#*     run as ./bin/aMCatNLO.py filename                    *\n' + \
        '#*                                                          *\n' + \
        '#************************************************************\n'
        
        if info_line:
            info_line = info_line[1:]

        logger.info(\
        "************************************************************\n" + \
        "*                                                          *\n" + \
        "*           W E L C O M E  to  M A D G R A P H 5           *\n" + \
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
        "*    The MadGraph5_aMC@NLO Development Team - Find us at   *\n" + \
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
            if hasattr(self, 'cluster'):
                logger.info('rm jobs on queue')
                self.cluster.remove()
            if hasattr(self, 'results'):
                self.update_status('Stop by the user', level=None, makehtml=True, error=True)
                self.add_error_log_in_html(KeyboardInterrupt)
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

    def help_banner_run(self):
        logger.info("syntax: banner_run Path|RUN [--run_options]")
        logger.info("-- Reproduce a run following a given banner")
        logger.info("   One of the following argument is require:")
        logger.info("   Path should be the path of a valid banner.")
        logger.info("   RUN should be the name of a run of the current directory")
        self.run_options_help([('-f','answer all question by default'),
                               ('--name=X', 'Define the name associated with the new run')]) 


    def help_compile(self):
        """help for compile command"""
        _compile_parser.print_help()

    def help_generate_events(self):
        """help for generate_events commandi
        just call help_launch"""
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
            raise self.InvalidCmd('Invalid syntax, please specify the run name')
        if not os.path.isdir(pjoin(self.me_dir, 'Events', args[0])):
            raise self.InvalidCmd('Directory %s does not exists' % \
                            pjoin(os.getcwd(), 'Events',  args[0]))

        self.set_run_name(args[0], level= 'shower')
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
            error_msg = 'No valid Madanalysis path set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)  
        if not  td:
            error_msg = 'No valid path to your topdrawer directory set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)  
                     
        if len(args) == 0:
            if not hasattr(self, 'run_name') or not self.run_name:
                self.help_plot()
                raise self.InvalidCmd('No run name defined. Please add this information.')             
            args.append('all')
            return

        
        if args[0] not in self._plot_mode:
            self.set_run_name(args[0], level='plot')
            del args[0]
            if len(args) == 0:
                args.append('all')
        elif not self.run_name:
            self.help_plot()
            raise self.InvalidCmd('No run name defined. Please add this information.')                             
        
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
            error_msg = 'No valid pythia-pgs path set.'
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
                raise self.InvalidCmd('No run name defined. Please add this information.')             
        
        if len(arg) == 1 and self.run_name == arg[0]:
            arg.pop(0)
        
        if not len(arg) and \
           not os.path.exists(pjoin(self.me_dir,'Events','pythia_events.hep')):
            self.help_pgs()
            raise self.InvalidCmd('''No file pythia_events.hep currently available. Please specify a valid run_name''')
                
        lock = None              
        if len(arg) == 1:
            prev_tag = self.set_run_name(arg[0], tag, 'pgs')
            filenames = misc.glob('events_*.hep.gz', pjoin(self.me_dir, 'Events', self.run_name)) 

            if not filenames:
                raise self.InvalidCmd('No event file corresponding to %s run with tag %s. '% (self.run_name, prev_tag))
            else:
                input_file = filenames[0]
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                lock = cluster.asyncrone_launch('gunzip',stdout=open(output_file,'w'), 
                                                    argument=['-c', input_file],
                                                    close_fds=True)
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
            error_msg = 'No valid delphes path set.'
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
                raise self.InvalidCmd('No run name defined. Please add this information.')             
        
        if len(arg) == 1 and self.run_name == arg[0]:
            arg.pop(0)
        
        if not len(arg) and \
           not os.path.exists(pjoin(self.me_dir,'Events','pythia_events.hep')):
            self.help_pgs()
            raise self.InvalidCmd('''No file pythia_events.hep currently available. Please specify a valid run_name''')
                              
        if len(arg) == 1:
            prev_tag = self.set_run_name(arg[0], tag, 'delphes')
            filenames = misc.glob('events_*.hep.gz', pjoin(self.me_dir, 'Events')) 
            
            
            if not filenames:
                raise self.InvalidCmd('No event file corresponding to %s run with tag %s.:%s '\
                    % (self.run_name, prev_tag, 
                       pjoin(self.me_dir,'Events',self.run_name, '%s_pythia_events.hep.gz' % prev_tag)))
            else:
                input_file = filenames[0]
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                lock = cluster.asyncrone_launch('gunzip',stdout=open(output_file,'w'), 
                                                    argument=['-c', input_file],
                                                    close_fds=True)
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
            raise self.InvalidCmd( 'Invalid Syntax: too many arguments')

        elif len(args) == 1:
            if not args[0] in ['NLO', 'LO']:
                raise self.InvalidCmd('%s is not a valid mode, please use "LO" or "NLO"' % args[1])
        mode = args[0]
        
        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd('options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.')


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
            raise self.InvalidCmd('Invalid Syntax: too many arguments')
        
        elif len(args) == 1:
            if not args[0] in ['NLO', 'LO']:
                raise self.InvalidCmd('%s is not a valid mode, please use "LO" or "NLO"' % args[1])
        mode = args[0]
        
        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd('options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.')

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
            raise self.InvalidCmd( 'Invalid Syntax: too many arguments')

        elif len(args) == 1:
            if not args[0] in ['LO', 'NLO', 'aMC@NLO', 'aMC@LO','auto']:
                raise self.InvalidCmd('%s is not a valid mode, please use "LO", "NLO", "aMC@NLO" or "aMC@LO"' % args[0])
        mode = args[0]
        
        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd('options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.')
        if mode == 'NLO' and options['reweightonly']:
            raise self.InvalidCmd('option -r (--reweightonly) needs mode "aMC@NLO" or "aMC@LO"')


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
            raise self.InvalidCmd('Invalid Syntax: too many arguments')

        elif len(args) == 1:
            if not args[0] in ['MC', 'FO']:
                raise self.InvalidCmd('%s is not a valid mode, please use "FO" or "MC"' % args[0])
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
            tags = misc.glob('%s_*_banner.txt' % args[1],pjoin(self.me_dir, 'Events' , args[1]))
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

        run_list = misc.glob(pjoin('*','*_banner.txt'), pjoin(self.me_dir, 'Events')) 
        run_list = [n.rsplit('/',2)[1] for n in run_list]
        possibilites['RUN Name'] = self.list_completion(text, run_list)
        
        return self.deal_multiple_categories(possibilites, formatting)
    
        
       except Exception as error:
           print(error)

 
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
        """auto-completion for generate_events command
        call the compeltion for launch"""
        self.complete_launch(text, line, begidx, endidx)


    def complete_shower(self, text, line, begidx, endidx):
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return valid run_name
            data = misc.glob(pjoin('*','events.lhe.gz', pjoin(self.me_dir, 'Events')))
            data = [n.rsplit('/',2)[1] for n in data]
            tmp1 =  self.list_completion(text, data)
            if not self.run_name:
                return tmp1

    def complete_plot(self, text, line, begidx, endidx):
        """ Complete the plot command """
        
        args = self.split_arg(line[0:begidx], error=False)

        if len(args) == 1:
            #return valid run_name
            data = misc.glob(pjoin('*','events.lhe*', pjoin(self.me_dir, 'Events')))
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
            data = misc.glob(pjoin('*', 'events_*.hep.gz'),
                             pjoin(self.me_dir, 'Events'))
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

class AskRunNLO(cmd.ControlSwitch):
    
    to_control = [('order', 'Type of perturbative computation'),
                  ('fixed_order', 'No MC@[N]LO matching / event generation'),
                  ('shower', 'Shower the generated events'),
                  ('madspin', 'Decay onshell particles'),
                  ('reweight', 'Add weights to events for new hypp.'),
                   ('madanalysis','Run MadAnalysis5 on the events generated')]
    
    quit_on = cmd.ControlSwitch.quit_on + ['onlyshower']
    
    def __init__(self, question, line_args=[], mode=None, force=False,
                                                                  *args, **opt):
        
        self.me_dir = opt['mother_interface'].me_dir
        self.check_available_module(opt['mother_interface'].options)
        self.last_mode = opt['mother_interface'].last_mode
        self.proc_characteristics = opt['mother_interface'].proc_characteristics
        with misc.TMP_variable(banner_mod.RunCard, 'allow_scan', True):
            self.run_card = banner_mod.RunCard(pjoin(self.me_dir,'Cards', 'run_card.dat'),
                                           consistency='warning')

        hide_line = []
        if 'QED' in self.proc_characteristics['splitting_types']:
            hide_line = ['madspin', 'shower', 'reweight', 'madanalysis']        
        
        super(AskRunNLO,self).__init__(self.to_control, opt['mother_interface'],
                                     *args, hide_line=hide_line, force=force,
                                      **opt)

    @property
    def answer(self):
        
        out = super(AskRunNLO, self).answer
        if out['shower'] == 'HERWIG7':
            out['shower'] = 'HERWIGPP'
        
        if out['shower'] not in self.get_allowed('shower') or out['shower'] =='OFF':
            out['runshower'] = False
        else:
            out['runshower'] = True
        return out

    def default(self,*args, **opts):
        self.nb_fo_warning = 0
        super(AskRunNLO, self).default(*args, **opts)

    def check_available_module(self, options):
        
        self.available_module = set()
        if options['madanalysis5_path']:
            self.available_module.add('MA5')
        if not aMCatNLO or ('mg5_path' in options and options['mg5_path']):
            
            self.available_module.add('MadSpin')
            if misc.has_f2py()  or options['f2py_compiler']:
                self.available_module.add('reweight')
        if options['pythia8_path']:
            self.available_module.add('PY8')
        if options['hwpp_path'] and options['thepeg_path'] and options['hepmc_path']:
            self.available_module.add('HW7')
            
        MCatNLO_libdir = pjoin(self.me_dir, 'MCatNLO', 'lib')
        if os.path.exists(os.path.realpath(pjoin(MCatNLO_libdir, 'libstdhep.a'))):
            self.available_module.add('StdHEP')
#
#   shorcut
#
    def ans_lo(self, value):
        """ function called if the user type lo=value. or lo (then value is None)"""
        
        if value is None:            
            self.switch['order'] = 'LO'
            self.switch['fixed_order'] = 'ON'
            self.set_switch('shower', 'OFF')
        else:
            logger.warning('Invalid command: lo=%s' % value)

    def ans_nlo(self, value):
        if value is None:
            self.switch['order'] = 'NLO'
            self.switch['fixed_order'] = 'ON'
            self.set_switch('shower', 'OFF')    
        else:
            logger.warning('Invalid command: nlo=%s' % value)
            
    def ans_amc__at__nlo(self, value):
        if value is None:
            self.switch['order'] = 'NLO'
            self.switch['fixed_order'] = 'OFF'
            self.set_switch('shower', 'ON')
        else:
            logger.warning('Invalid command: aMC@NLO=%s' % value)
            
    def ans_amc__at__lo(self, value):
        if value is None:
            self.switch['order'] = 'LO'
            self.switch['fixed_order'] = 'OFF'
            self.set_switch('shower', 'ON')    
        else:
            logger.warning('Invalid command: aMC@LO=%s' % value)  
                  
    def ans_noshower(self, value):
        if value is None:
            self.switch['order'] = 'NLO'
            self.switch['fixed_order'] = 'OFF'
            self.set_switch('shower', 'OFF')
        else:
            logger.warning('Invalid command: noshower=%s' % value)  
        
    def ans_onlyshower(self, value):
        if value is None:
            self.switch['mode'] = 'onlyshower'
            self.switch['madspin'] = 'OFF'
            self.switch['reweight'] = 'OFF'
        else:
            logger.warning('Invalid command: onlyshower=%s' % value)     
              
    def ans_noshowerlo(self, value):
        if value is None:
            self.switch['order'] = 'LO'
            self.switch['fixed_order'] = 'OFF'
            self.set_switch('shower', 'OFF')
        else:
            logger.warning('Invalid command: noshowerlo=%s' % value)    
            
    def ans_madanalysis5(self, value):
        """ shortcut madanalysis5 -> madanalysis """
        
        if value is None:
            return self.onecmd('madanalysis')
        else:
            self.set_switch('madanalysis', value)                
#
#   ORDER   
#   
    def get_allowed_order(self):
        return ["LO", "NLO"]
    
    def set_default_order(self):  
        
        if self.last_mode in ['LO', 'aMC@L0', 'noshowerLO']:
            self.switch['order'] = 'LO'       
        self.switch['order'] = 'NLO'
        
    def set_switch_off_order(self):
        return
#
#   Fix order
#    
    def get_allowed_fixed_order(self):
        """ """
        
        if self.proc_characteristics['ninitial'] == 1 or \
           'QED' in self.proc_characteristics['splitting_types']:
            return ['ON']
        else:
            return ['ON', 'OFF']
        
    def set_default_fixed_order(self):
          
        if self.last_mode in ['LO', 'NLO']:
            self.switch['fixed_order'] = 'ON'
        elif self.proc_characteristics['ninitial'] == 1:
            self.switch['fixed_order'] = 'ON'
        elif 'QED' in self.proc_characteristics['splitting_types']:
             self.switch['fixed_order'] = 'ON'
        else:
            self.switch['fixed_order'] = 'OFF'

    def color_for_fixed_order(self, switch_value):
         
        if switch_value in ['OFF']:
            return self.green % switch_value
        else:
            return self.red % switch_value
    
    def print_options_fixed_order(self):
        
        if 'QED' in self.proc_characteristics['splitting_types']:
            return "No NLO+PS available for EW correction"
        else:
            return self.print_options('fixed_order', keep_default=True)
    
    def color_for_shower(self, switch_value):
         
        if switch_value in ['ON']:
            return self.green % switch_value
        elif switch_value in self.get_allowed('shower'):
            return self.green % switch_value
        else:
            return self.red % switch_value    
    
   
    def consistency_QED(self, key, value, switch):
        """ temporary way to forbid event generation due to lack of validation"""
        
#        if True:
        if 'QED' in self.proc_characteristics['splitting_types']:
            out = {}
            to_check ={'fixed_order': ['ON'],
                       'shower': ['OFF'],
                       'madanalysis': ['OFF'],
                       'madspin': ['OFF','onshell','none'],
                       'reweight': ['OFF']}
            for key, allowed  in to_check.items():        
                if switch[key] not in allowed:
                    out[key] = allowed[0]
                    if not self.nb_fo_warning:
                        if 'QED' in self.proc_characteristics['splitting_types']:
                            logger.warning("NLO+PS mode is not allowed for processes including electroweak corrections")
                        self.nb_fo_warning = 1
        else: 
            return self.check_consistency_with_all(key, value)
        return out 
    #apply to all related to the group 
    consistency_fixed_order = lambda self, *args, **opts: self.consistency_QED('fixed_order', *args, **opts)
    consistency_shower = lambda self, *args, **opts: self.consistency_QED('shower', *args, **opts)
    consistency_madanalysis = lambda self, *args, **opts: self.consistency_QED('madanalysis', *args, **opts)
    consistency_madspin = lambda self, *args, **opts: self.consistency_QED('madspin', *args, **opts)
    consistency_reweight = lambda self, *args, **opts: self.consistency_QED('reweight', *args, **opts)
      
    def consistency_fixed_order_shower(self, vfix, vshower):
        """ consistency_XX_YY(val_XX, val_YY)
           -> XX is the new key set by the user to a new value val_XX
           -> YY is another key set by the user.
           -> return value should be None or "replace_YY" 
        """
        
        if vfix == 'ON' and vshower != 'OFF' :
            return 'OFF'
        return None
    
    consistency_fixed_order_madspin = consistency_fixed_order_shower
    consistency_fixed_order_reweight = consistency_fixed_order_shower


    def consistency_fixed_order_madanalysis(self, vfix, vma5):
        
        if vfix == 'ON' and vma5 == 'ON' :
            return 'OFF'
        return None    


    def consistency_shower_fixed_order(self, vshower, vfix):
        """ consistency_XX_YY(val_XX, val_YY)
           -> XX is the new key set by the user to a new value val_XX
           -> YY is another key set by the user.
           -> return value should be None or "replace_YY" 
        """    
        
        if vshower != 'OFF' and vfix == 'ON':
            return 'OFF'
        return None

    consistency_madspin_fixed_order = consistency_shower_fixed_order
    consistency_reweight_fixed_order = consistency_shower_fixed_order
    consistency_madanalysis_fixed_order = consistency_shower_fixed_order

            
#
#   Shower
#
    def get_allowed_shower(self):
        """ """
        
        if hasattr(self, 'allowed_shower'):
            return self.allowed_shower
        
        if 'QED' in self.proc_characteristics['splitting_types']:
            self.allowed_shower = ['OFF']
            return self.allowed_shower

        if not misc.which('bc'):
            return ['OFF']
        
        if self.proc_characteristics['ninitial'] == 1:
            self.allowed_shower = ['OFF']
            return ['OFF']
        else:
            if 'StdHEP' in self.available_module:
                allowed = ['HERWIG6','OFF', 'PYTHIA6Q', 'PYTHIA6PT', ]
            else:
                allowed = ['OFF']
            if 'PY8' in self.available_module:
                allowed.append('PYTHIA8')
            if 'HW7' in self.available_module:
                allowed.append('HERWIGPP')
            
            
            self.allowed_shower = allowed
            
            return allowed
    
    def check_value_shower(self, value):
        """ """
        
        if value.upper() in self.get_allowed_shower():
            return True
        if value.upper() in ['PYTHIA8', 'HERWIGPP']:
            return True
        if value.upper() == 'ON':
            return self.run_card['parton_shower']
        if value.upper() in ['P8','PY8','PYTHIA_8']:
            return 'PYTHIA8'
        if value.upper() in ['PY6','P6','PY6PT', 'PYTHIA_6', 'PYTHIA_6PT','PYTHIA6PT','PYTHIA6_PT']:
            return 'PYTHIA6PT'
        if value.upper() in ['PY6Q', 'PYTHIA_6Q','PYTHIA6Q', 'PYTHIA6_Q']:
            return 'PYTHIA6Q'
        if value.upper() in ['HW7', 'HERWIG7']:
            return 'HERWIG7'
        if value.upper() in ['HW++', 'HWPP', 'HERWIG++']:
            return 'HERWIGPP'
        if value.upper() in ['HW6', 'HERWIG_6']:
            return 'HERWIG6'
    
    def set_default_shower(self): 
        
        if 'QED' in self.proc_characteristics['splitting_types']:
            self.switch['shower'] = 'Not Avail'
        elif self.last_mode in ['LO', 'NLO', 'noshower', 'noshowerLO']:
            self.switch['shower'] = 'OFF'         
        elif self.proc_characteristics['ninitial'] == 1:
            self.switch['shower'] = 'OFF'
        elif os.path.exists(pjoin(self.me_dir, 'Cards', 'shower_card.dat')):
            if self.switch['fixed_order'] == "OFF":
                self.switch['shower'] = self.run_card['parton_shower']  
            elif self.switch['fixed_order'] == "ON":
                self.switch['shower'] = "OFF" 
        else:
            if self.switch['fixed_order'] == "ON":
                self.switch['shower'] = 'OFF'
            else:
                self.switch['shower'] = 'OFF (%s)' % self.run_card['parton_shower']

        if self.last_mode in ['LO', 'NLO', 'noshower', 'noshowerLO']:
            self.switch['shower'] = 'OFF'
            return 
        
        if self.proc_characteristics['ninitial'] == 1:
            self.switch['shower'] = 'OFF'
            return
        
        if not misc.which('bc'):
            logger.warning('bc command not available. Forbids to run the shower. please install it if you want to run the shower. (sudo apt-get install bc)')
            self.switch['shower'] = 'OFF'
            return
         
        if os.path.exists(pjoin(self.me_dir, 'Cards', 'shower_card.dat')):
            self.switch['shower'] = self.run_card['parton_shower']  
            #self.switch['shower'] = 'ON'
            self.switch['fixed_order'] = "OFF"


    def consistency_shower_madanalysis(self, vshower, vma5):
        """ MA5 only possible with (N)LO+PS if shower is run"""
        
        if vshower == 'OFF' and vma5 == 'ON':
            return 'OFF'
        return None
    
    def consistency_madanalysis_shower(self, vma5, vshower):
        
        if vma5=='ON' and vshower == 'OFF':
            return 'ON'
        return None
    
    def get_cardcmd_for_shower(self, value):
        """ adpat run_card according to this setup. return list of cmd to run"""
        
        if value != 'OFF':
            return  ['set parton_shower %s' % self.switch['shower']]
        return []
    
#
#   madspin
#
    def get_allowed_madspin(self):
        """ """
        
        if hasattr(self, 'allowed_madspin'):
            return self.allowed_madspin
        
        self.allowed_madspin = []
        
        
        if 'MadSpin' not in self.available_module:
            return self.allowed_madspin
        if self.proc_characteristics['ninitial'] == 1:
            self.available_module.remove('MadSpin')
            self.allowed_madspin = ['OFF']
            return self.allowed_madspin
        else:        
            if 'QED' in self.proc_characteristics['splitting_types']:
                self.allowed_madspin = ['OFF', 'onshell']
            else:
                self.allowed_madspin = ['OFF', 'ON', 'onshell']
            return  self.allowed_madspin
        
    def check_value_madspin(self, value):
        """handle alias and valid option not present in get_allowed_madspin
        remember that this mode should always be OFF for 1>N. (ON not in allowed value)"""
        
        if value.upper() in self.get_allowed_madspin():
            if value == value.upper():
                return True
            else:
                return value.upper()
        elif value.lower() in self.get_allowed_madspin():
            if value == value.lower():
                return True
            else:
                return value.lower()
                        
        if 'MadSpin' not in self.available_module or \
           'ON' not in self.get_allowed_madspin():
            return False   
        
        if value.lower() in ['madspin', 'full']:
            return 'full'
        elif value.lower() in ['none']:
            return 'none'
            
    def set_default_madspin(self):
        
        if 'MadSpin' in self.available_module:
            if os.path.exists(pjoin(self.me_dir,'Cards','madspin_card.dat')):
                self.switch['madspin'] = 'ON'
            else:
                self.switch['madspin'] = 'OFF'
        else:
            self.switch['madspin'] = 'Not Avail'  
            
    def get_cardcmd_for_madspin(self, value):
        """set some command to run before allowing the user to modify the cards."""
        
        if value == 'onshell':
            return ["edit madspin_card --replace_line='set spinmode' --before_line='decay' set spinmode onshell"]
        elif value in ['full', 'madspin']:
            return ["edit madspin_card --replace_line='set spinmode' --before_line='decay' set spinmode madspin"]
        elif value == 'none':
            return ["edit madspin_card --replace_line='set spinmode' --before_line='decay' set spinmode none"]
        else:
            return []            
        
#
#   reweight
#
    def get_allowed_reweight(self):
        """set the valid (visible) options for reweight"""
        
        if hasattr(self, 'allowed_reweight'):
            return getattr(self, 'allowed_reweight')
        


        self.allowed_reweight = []
        if 'QED' in self.proc_characteristics['splitting_types']:
            return self.allowed_reweight 
        if 'reweight' not in self.available_module:
            return self.allowed_reweight
        if self.proc_characteristics['ninitial'] == 1:
            self.available_module.remove('reweight')
            self.allowed_reweight.append('OFF')
            return self.allowed_reweight
        else:
            self.allowed_reweight = [ 'OFF', 'ON', 'NLO', 'NLO_TREE','LO']
            return self.allowed_reweight     
    
    def set_default_reweight(self):
        """initialise the switch for reweight"""
        
        if 'QED' in self.proc_characteristics['splitting_types']:
            self.switch['reweight'] = 'Not Avail'
        elif 'reweight' in self.available_module:
            if os.path.exists(pjoin(self.me_dir,'Cards','reweight_card.dat')):
                self.switch['reweight'] = 'ON'
            else:
                self.switch['reweight'] = 'OFF'
        else:
            self.switch['reweight'] = 'Not Avail'      
            
    def get_cardcmd_for_reweight(self, value):
        """ adpat run_card according to this setup. return list of cmd to run"""
        
        if value == 'LO':
            return ["edit reweight_card --replace_line='change mode' --before_line='launch' change mode LO"]
        elif  value == 'NLO':
            return ["edit reweight_card --replace_line='change mode' --before_line='launch' change mode NLO",
                    "set store_rwgt_info T"]
        elif value == 'NLO_TREE':
            return ["edit reweight_card --replace_line='change mode' --before_line='launch' change mode NLO_tree",
                    "set store_rwgt_info T"]            
        return []
            
#
#   MadAnalysis5
#    
    def get_allowed_madanalysis(self):
        
        if hasattr(self, 'allowed_madanalysis'):
            return self.allowed_madanalysis
        
        self.allowed_madanalysis = []
        
        if 'QED' in self.proc_characteristics['splitting_types']:
            return self.allowed_madanalysis
        
        if 'MA5' not in self.available_module:
            return self.allowed_madanalysis
        
        if self.proc_characteristics['ninitial'] == 1:
            self.available_module.remove('MA5')
            self.allowed_madanalysis = ['OFF']
            return self.allowed_madanalysis
        else:
            self.allowed_madanalysis = ['OFF', 'ON']
            return  self.allowed_madanalysis
    
    def set_default_madanalysis(self):
        """initialise the switch for reweight"""
        
        if 'QED' in self.proc_characteristics['splitting_types']:
            self.switch['madanalysis'] = 'Not Avail'
        elif 'MA5' not in self.available_module: 
            self.switch['madanalysis'] =  'Not Avail'
        elif os.path.exists(pjoin(self.me_dir,'Cards', 'madanalysis5_hadron_card.dat')):
            self.switch['madanalysis'] = 'ON'
        else:
            self.switch['madanalysis'] = 'OFF'
            
    def check_value_madanalysis(self, value):
        """check an entry is valid. return the valid entry in case of shortcut"""
        
        if value.upper() in self.get_allowed('madanalysis'):
            return True
        value = value.lower()
        if value == 'hadron':
            return 'ON' if 'ON' in self.get_allowed_madanalysis5 else False
        else:
            return False
        
        
#===============================================================================
# aMCatNLOCmd
#===============================================================================
class aMCatNLOCmd(CmdExtended, HelpToCmd, CompleteForCmd, common_run.CommonRunCmd):
    """The command line processor of MadGraph"""    
    

    LO = False
    # Truth values
    true = ['T','.true.',True,'true']
    # Options and formats available
    _run_options = ['--cluster','--multicore','--nb_core=','--nb_core=2', '-c', '-m']
    _generate_options = ['-f', '--laststep=parton', '--laststep=pythia', '--laststep=pgs', '--laststep=delphes']
    _calculate_decay_options = ['-f', '--accuracy=0.']
    _set_options = ['stdout_level','fortran_compiler','cpp_compiler','timeout']
    _plot_mode = ['all', 'parton','shower','pgs','delphes']
    _clean_mode = _plot_mode + ['channel', 'banner']
    _display_opts = ['run_name', 'options', 'variable']
    # survey options, dict from name to type, default value, and help text
    # Variables to store object information
    web = False
    cluster_mode = 0
    queue  = 'madgraph'
    nb_core = None
    make_opts_var = {}
    
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

        self.mode = 'aMCatNLO'
        self.nb_core = 0
        self.prompt = "%s>"%os.path.basename(pjoin(self.me_dir))


        self.load_results_db()
        self.results.def_web_mode(self.web)
        # check that compiler is gfortran 4.6 or later if virtuals have been exported
        proc_card = open(pjoin(self.me_dir, 'Cards', 'proc_card_mg5.dat')).read()

        if not '[real=QCD]' in proc_card:
            check_compiler(self.options, block=True)

        
    ############################################################################      
    def do_shower(self, line):
        """ run the shower on a given parton level file """
        argss = self.split_arg(line)
        (options, argss) = _launch_parser.parse_args(argss)
        # check argument validity and normalise argument
        options = options.__dict__
        options['reweightonly'] = False
        self.check_shower(argss, options)
        evt_file = pjoin(os.getcwd(), argss[0], 'events.lhe')
        self.ask_run_configuration('onlyshower', options)
        self.run_mcatnlo(evt_file, options)

        self.update_status('', level='all', update_results=True)

    ################################################################################
    def do_plot(self, line):
        """Create the plot for a given run"""

        # Since in principle, all plot are already done automaticaly
        args = self.split_arg(line)
        # Check argument's validity
        self.check_plot(args)
        logger.info('plot for run %s' % self.run_name)
        
        if not self.force:
            self.ask_edit_cards([], args, plot=True)
                
        if any([arg in ['parton'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe')
            if os.path.exists(filename+'.gz'):
                misc.gunzip(filename)
            if  os.path.exists(filename):
                logger.info('Found events.lhe file for run %s' % self.run_name) 
                shutil.move(filename, pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'))
                self.create_plot('parton')
                shutil.move(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'), filename)
                misc.gzip(filename)
                
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
            filenames = misc.glob('events_*.lhe.gz', pjoin(self.me_dir, 'Events', self.run_name))
            if len(filenames) != 1:
                filenames = misc.glob('events_*.hep.gz', pjoin(self.me_dir, 'Events', self.run_name)) 
                if len(filenames) != 1:
                    logger.info('No shower-level event file found for run %s' % \
                                self.run_name)
                    return
                filename = filenames[0]
                misc.gunzip(filename, keep=True, stdout=pjoin(self.me_dir, 'Events','pythia_events.hep'))
                
                if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pythia_card.dat')):
                    if aMCatNLO and not self.options['mg5_path']:
                        raise Exception("plotting NLO HEP files requires MG5 utilities.")
                    
                    files.cp(pjoin(self.options['mg5_path'], 'Template','LO', 'Cards', 'pythia_card_default.dat'),
                             pjoin(self.me_dir, 'Cards', 'pythia_card.dat'))
                self.run_hep2lhe()
            else:
                filename = filenames[0]
                misc.gunzip(filename, keep=True, stdout=pjoin(self.me_dir, 'Events','pythia_events.hep'))

            self.create_plot('shower')
            lhe_file_name = filename.replace('.hep.gz', '.lhe')
            shutil.move(pjoin(self.me_dir, 'Events','pythia_events.lhe'), 
                        lhe_file_name)
            misc.gzip(lhe_file_name)
                    
        if any([arg in ['all','pgs'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 
                                            '%s_pgs_events.lhco' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                misc.gunzip(filename)
            if  os.path.exists(filename):
                self.create_plot('PGS')
                misc.gzip(filename)                
            else:
                logger.info('No valid files to make PGS plots')
                
        if any([arg in ['all','delphes'] for arg in args]):
            filename = pjoin(self.me_dir, 'Events', self.run_name, 
                                        '%s_delphes_events.lhco' % self.run_tag)
            if os.path.exists(filename+'.gz'):
                misc.gunzip(filename)
            if  os.path.exists(filename):
                #shutil.move(filename, pjoin(self.me_dir, 'Events','delphes_events.lhco'))
                self.create_plot('Delphes')
                #shutil.move(pjoin(self.me_dir, 'Events','delphes_events.lhco'), filename)
                misc.gzip(filename)                
            else:
                logger.info('No valid files to make Delphes plots')


    ############################################################################      
    def do_calculate_xsect(self, line):
        """Main commands: calculates LO/NLO cross-section, using madevent_mintFO 
        this function wraps the do_launch one"""
        
        self.start_time = time.time()
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _calculate_xsect_parser.parse_args(argss)
        options = options.__dict__
        options['reweightonly'] = False
        options['parton'] = True
        self.check_calculate_xsect(argss, options)
        self.do_launch(line, options, argss)
        
    ############################################################################
    def do_banner_run(self, line): 
        """Make a run from the banner file"""
        
        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_banner_run(args)    
                     
        # Remove previous cards
        for name in ['shower_card.dat', 'madspin_card.dat']:
            try:
                os.remove(pjoin(self.me_dir, 'Cards', name))
            except Exception:
                pass
            
        banner_mod.split_banner(args[0], self.me_dir, proc_card=False)
        
        # Check if we want to modify the run
        if not self.force:
            ans = self.ask('Do you want to modify the Run type and/or any of the Cards?', 'n', ['y','n'])
            if ans == 'n':
                self.force = True
        
        # Compute run mode:
        if self.force:
            mode_status = {'order': 'NLO', 'fixed_order': False, 'madspin':False, 'shower':True}
            banner = banner_mod.Banner(args[0])
            for line in banner['run_settings']:
                if '=' in line:
                    mode, value = [t.strip() for t in line.split('=')]
                    mode_status[mode] = value
        else:
            mode_status = {}

        # Call Generate events
        self.do_launch('-n %s %s' % (self.run_name, '-f' if self.force else ''),
                       switch=mode_status)
        
    ############################################################################      
    def do_generate_events(self, line):
        """Main commands: generate events  
        this function just wraps the do_launch one"""
        self.do_launch(line)


    ############################################################################
    def do_treatcards(self, line, amcatnlo=True,mode=''):
        """Advanced commands: this is for creating the correct run_card.inc from the nlo format"""
                #check if no 'Auto' are present in the file
        self.check_param_card(pjoin(self.me_dir, 'Cards','param_card.dat'))
        
        # propagate the FO_card entry FO_LHE_weight_ratio to the run_card.
        # this variable is system only in the run_card 
        # can not be done in EditCard since this parameter is not written in the 
        # run_card directly.
        if mode in ['LO', 'NLO']: 
            name = 'fo_lhe_weight_ratio'
            FO_card = analyse_card.FOAnalyseCard(pjoin(self.me_dir,'Cards', 'FO_analyse_card.dat'))
            if name in FO_card:
                self.run_card.set(name, FO_card[name], user=False)
            name = 'fo_lhe_postprocessing'
            if name in FO_card:
                self.run_card.set(name, FO_card[name], user=False)

        return super(aMCatNLOCmd,self).do_treatcards(line, amcatnlo)
    
    ############################################################################
    def set_configuration(self, amcatnlo=True, **opt):
        """assign all configuration variable from file 
            loop over the different config file if config_file not define """
        return super(aMCatNLOCmd,self).set_configuration(amcatnlo=amcatnlo, **opt)
    
    ############################################################################      
    def do_launch(self, line, options={}, argss=[], switch={}):
        """Main commands: launch the full chain 
        options and args are relevant if the function is called from other 
        functions, such as generate_events or calculate_xsect
        mode gives the list of switch needed for the computation (usefull for banner_run)
        """

        if not argss and not options:
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
        
        if not switch:
            mode = argss[0]

            if mode in ['LO', 'NLO']:
                options['parton'] = True
            mode = self.ask_run_configuration(mode, options)
        else:
            mode = self.ask_run_configuration('auto', options, switch)

        if 'run_name' in list(options.keys()) and options['run_name']:
            self.run_name = options['run_name']
            # if a dir with the given run_name already exists
            # remove it and warn the user
            if os.path.isdir(pjoin(self.me_dir, 'Events', self.run_name)):
                logger.warning('removing old run information in \n'+
                                pjoin(self.me_dir, 'Events', self.run_name))
                files.rm(pjoin(self.me_dir, 'Events', self.run_name))
                self.results.delete_run(self.run_name)
        else:
            self.run_name = ''
            check = 0
            suffix = ''
            if mode == 'LO':
                suffix = '_%s' % mode
            while True:
                check += 1
                name = 'run_%02i%s' % (check, suffix)
                if not misc.glob('run_%02i*' % check, path= pjoin(self.me_dir, 'Events')): 
                    self.run_name = name
                    break

        if self.options['automatic_html_opening']:
            misc.open_file(os.path.join(self.me_dir, 'crossx.html'))
            self.options['automatic_html_opening'] = False

        self.run_generate_events(mode, options, argss, switch)
 
        #check if the param_card defines a scan.
        if False:# self.param_card_iterator:
            cpath = pjoin(self.me_dir,'Cards','param_card.dat')
            param_card_iterator = self.param_card_iterator
            self.param_card_iterator = [] #avoid to next generate go trough here
            param_card_iterator.store_entry(self.run_name, self.results.current['cross'],
                                            error=self.results.current['error'],
                                            param_card_path=cpath)
            orig_name = self.run_name
            #go trough the scan
            with misc.TMP_variable(self, 'allow_notification_center', False):
                for i,card in enumerate(param_card_iterator):
                    card.write(cpath)
                    self.check_param_card(cpath, dependent=True)
                    if not options['force']:
                        options['force'] = True
                    if options['run_name']:
                        options['run_name'] = '%s_%s' % (orig_name, i+1)
                    if not argss:
                        argss = [mode, "-f"]
                    elif argss[0] == "auto":
                        argss[0] = mode
                    self.do_launch("", options=options, argss=argss, switch=switch)
                    #self.exec_cmd("launch -f ",precmd=True, postcmd=True,errorhandling=False)
                    param_card_iterator.store_entry(self.run_name, self.results.current['cross'],
                                                    error=self.results.current['error'],
                                                    param_card_path=cpath)
            #restore original param_card
            param_card_iterator.write(pjoin(self.me_dir,'Cards','param_card.dat'))
            name = misc.get_scan_name(orig_name, self.run_name)
            path = pjoin(self.me_dir, 'Events','scan_%s.txt' % name)
            logger.info("write all cross-section results in %s" % path, '$MG:BOLD')
            param_card_iterator.write_summary(path)
            
        if self.allow_notification_center:    
            misc.system_notify('Run %s finished' % os.path.basename(self.me_dir), 
                              '%s: %s +- %s ' % (self.results.current['run_name'], 
                                                 self.results.current['cross'],
                                                 self.results.current['error']))
    
    # this decorator handle the loop related to scan.
    @common_run.scanparamcardhandling(run_card_scan=True)
    def run_generate_events(self, mode, options, args, switch): 

        self.results.add_detail('run_mode', mode)
        self.update_status('Starting run', level=None, update_results=True)

        if '+' in mode:
            mode = mode.split('+')[0]
        self.compile(mode, options) 
        evt_file = self.run(mode, options)
        
        if self.run_card['nevents'] == 0 and not mode in ['LO', 'NLO']:
            logger.info('No event file generated: grids have been set-up with a '\
                            'relative precision of %s' % self.run_card['req_acc'])
            return

        if not mode in ['LO', 'NLO']:
            assert evt_file == pjoin(self.me_dir,'Events', self.run_name, 'events.lhe'), '%s != %s' %(evt_file, pjoin(self.me_dir,'Events', self.run_name, 'events.lhe.gz'))
            
            if self.run_card['systematics_program'] == 'systematics':
                self.exec_cmd('systematics %s %s ' % (self.run_name, ' '.join(self.run_card['systematics_arguments'])))
            
            self.exec_cmd('reweight -from_cards', postcmd=False)
            self.exec_cmd('decay_events -from_cards', postcmd=False)
            evt_file = pjoin(self.me_dir,'Events', self.run_name, 'events.lhe')
            if self.run_card['time_of_flight']>=0:
                self.exec_cmd("add_time_of_flight --threshold=%s" % self.run_card['time_of_flight'] ,postcmd=False)

        if not mode in ['LO', 'NLO', 'noshower', 'noshowerLO'] \
                                                      and not options['parton']:
            self.run_mcatnlo(evt_file, options)
            self.exec_cmd('madanalysis5_hadron --no_default', postcmd=False, printcmd=False)

        elif mode == 'noshower':
            logger.warning("""You have chosen not to run a parton shower. NLO events without showering are NOT physical. Please, shower the Les Houches events before using them for physics analyses.""")


        self.update_status('', level='all', update_results=True)
        if self.run_card['ickkw'] == 3 and \
           (mode in ['noshower'] or \
            (('PYTHIA8' not in self.run_card['parton_shower'].upper()) and (mode in ['aMC@NLO']))):
            logger.warning("""You are running with FxFx merging enabled. To be able to merge samples of various multiplicities without double counting, you have to remove some events after showering 'by hand'. Please read http://amcatnlo.cern.ch/FxFx_merging.htm for more details.""")

        self.store_result()


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


    def update_random_seed(self):
        """Update random number seed with the value from the run_card. 
        If this is 0, update the number according to a fresh one"""
        iseed = self.run_card['iseed']
        if iseed == 0:
            randinit = open(pjoin(self.me_dir, 'SubProcesses', 'randinit'))
            iseed = int(randinit.read()[2:]) + 1
            randinit.close()
        randinit = open(pjoin(self.me_dir, 'SubProcesses', 'randinit'), 'w')
        randinit.write('r=%d' % iseed)
        randinit.close()
            
        
    def run(self, mode, options):
        """runs aMC@NLO. Returns the name of the event file created"""
        logger.info('Starting run')

        if not 'only_generation' in list(options.keys()):
            options['only_generation'] = False

        self.get_characteristics(pjoin(self.me_dir, 'SubProcesses', 'proc_characteristics'))
        self.setup_cluster_or_multicore()
        self.update_random_seed()
        #find and keep track of all the jobs
        folder_names = {'LO': ['born_G*'], 'NLO': ['all_G*'],
                    'aMC@LO': ['GB*'], 'aMC@NLO': ['GF*']}
        folder_names['noshower'] = folder_names['aMC@NLO']
        folder_names['noshowerLO'] = folder_names['aMC@LO']
        p_dirs = [d for d in \
                open(pjoin(self.me_dir, 'SubProcesses', 'subproc.mg')).read().split('\n') if d]
        #Clean previous results
        self.clean_previous_results(options,p_dirs,folder_names[mode])

        mcatnlo_status = ['Setting up grids', 'Computing upper envelope', 'Generating events']


        if options['reweightonly']:
            event_norm=self.run_card['event_norm']
            nevents=self.run_card['nevents']
            return self.reweight_and_collect_events(options, mode, nevents, event_norm)

        if mode in ['LO', 'NLO']:
            # this is for fixed order runs
            mode_dict = {'NLO': 'all', 'LO': 'born'}
            logger.info('Doing fixed order %s' % mode)
            req_acc = self.run_card['req_acc_FO']

            # create a list of dictionaries "jobs_to_run" with all the
            # jobs that need to be run
            integration_step=-1
            jobs_to_run,jobs_to_collect,integration_step = self.create_jobs_to_run(options,p_dirs, \
                                req_acc,mode_dict[mode],integration_step,mode,fixed_order=True)
            self.prepare_directories(jobs_to_run,mode)

            # loop over the integration steps. After every step, check
            # if we have the required accuracy. If this is the case,
            # stop running, else do another step.
            while True:
                integration_step=integration_step+1
                self.run_all_jobs(jobs_to_run,integration_step)
                self.collect_log_files(jobs_to_run,integration_step)
                jobs_to_run,jobs_to_collect=self.collect_the_results(options,req_acc,jobs_to_run, \
                                  jobs_to_collect,integration_step,mode,mode_dict[mode])
                if not jobs_to_run:
                    # there are no more jobs to run (jobs_to_run is empty)
                    break
            # We are done.
            self.finalise_run_FO(folder_names[mode],jobs_to_collect)
            self.update_status('Run complete', level='parton', update_results=True)
            return

        elif mode in ['aMC@NLO','aMC@LO','noshower','noshowerLO']:
            if self.ninitial == 1:
                raise aMCatNLOError('Decay processes can only be run at fixed order.')
            mode_dict = {'aMC@NLO': 'all', 'aMC@LO': 'born',\
                         'noshower': 'all', 'noshowerLO': 'born'}
            shower = self.run_card['parton_shower'].upper()
            nevents = self.run_card['nevents']
            req_acc = self.run_card['req_acc']
            if nevents == 0 and req_acc < 0 :
                raise aMCatNLOError('Cannot determine the required accuracy from the number '\
                                        'of events, because 0 events requested. Please set '\
                                        'the "req_acc" parameter in the run_card to a value '\
                                        'between 0 and 1')
            elif req_acc >1 or req_acc == 0 :
                raise aMCatNLOError('Required accuracy ("req_acc" in the run_card) should '\
                                        'be between larger than 0 and smaller than 1, '\
                                        'or set to -1 for automatic determination. Current '\
                                        'value is %f' % req_acc)
# For more than 1M events, set req_acc to 0.001 (except when it was explicitly set in the run_card)
            elif req_acc < 0 and nevents > 1000000 :
                req_acc=0.001

            shower_list = ['HERWIG6', 'HERWIGPP', 'PYTHIA6Q', 'PYTHIA6PT', 'PYTHIA8']

            if not shower in shower_list:
                raise aMCatNLOError('%s is not a valid parton shower. '\
                                    'Please use one of the following: %s' \
                                    % (shower, ', '.join(shower_list)))

# check that PYTHIA6PT is not used for processes with FSR
            if shower == 'PYTHIA6PT' and self.proc_characteristics['has_fsr']:
                raise aMCatNLOError('PYTHIA6PT does not support processes with FSR')

            if mode in ['aMC@NLO', 'aMC@LO']:
                logger.info('Doing %s matched to parton shower' % mode[4:])
            elif mode in ['noshower','noshowerLO']:
                logger.info('Generating events without running the shower.')
            elif options['only_generation']:
                logger.info('Generating events starting from existing results')
            
            jobs_to_run,jobs_to_collect,integration_step = self.create_jobs_to_run(options,p_dirs, \
                                            req_acc,mode_dict[mode],1,mode,fixed_order=False)
            # Make sure to update all the jobs to be ready for the event generation step
            if options['only_generation']:
                jobs_to_run,jobs_to_collect=self.collect_the_results(options,req_acc,jobs_to_run, \
                                jobs_to_collect,1,mode,mode_dict[mode],fixed_order=False)
            else:
                self.prepare_directories(jobs_to_run,mode,fixed_order=False)


            # Main loop over the three MINT generation steps:
            for mint_step, status in enumerate(mcatnlo_status):
                if options['only_generation'] and mint_step < 2:
                    continue
                self.update_status(status, level='parton')
                self.run_all_jobs(jobs_to_run,mint_step,fixed_order=False)
                self.collect_log_files(jobs_to_run,mint_step)
                jobs_to_run,jobs_to_collect=self.collect_the_results(options,req_acc,jobs_to_run, \
                                jobs_to_collect,mint_step,mode,mode_dict[mode],fixed_order=False)
                if mint_step+1==2 and nevents==0:
                    self.print_summary(options,2,mode)
                    return

            # Sanity check on the event files. If error the jobs are resubmitted
            self.check_event_files(jobs_to_collect)

            if self.cluster_mode == 1:
            #if cluster run, wait 10 sec so that event files are transferred back
                self.update_status(
                    'Waiting while files are transferred back from the cluster nodes',
                    level='parton')
                time.sleep(10)

            event_norm=self.run_card['event_norm']
            # gather the various orders tag and write include files (only needed once)
            if not os.path.exists(pjoin(self.me_dir, 'SubProcesses', 'orderstags_glob.dat')):
                raise aMCatNLOError('Error, cannot find orderstag_glob.dat')
                self.write_orders_tag_info()

            return self.reweight_and_collect_events(options, mode, nevents, event_norm)


    def write_orders_tag_info(self):
        """Collects the informations on the orders_tag variable from the 
        different channels and writes a file, linked to the P0 dirs
        """
        log = pjoin(self.me_dir, 'Events', self.run_name, 'alllogs_0.html')
        content = open(log).read()
        taglines = [l for l in content.split('\n') if 'orders_tag_plot=' in l]
        orderstags = []
        for l in taglines:
            tag = int(l.split()[1])
            if not tag in orderstags:
                orderstags.append(tag)
        # now write a fortran include file with all the informations
        content = '%d\n' % len(orderstags)
        content+= '%s\n' % ' '.join(['%d' % v for v in orderstags])
        outfile = open(pjoin(self.me_dir, 'SubProcesses', 'orderstags_glob.dat'), 'w')
        outfile.write(content)
        outfile.close()

        # finally link it into the p dirs
        p_dirs = [d for d in \
                open(pjoin(self.me_dir, 'SubProcesses', 'subproc.mg')).read().split('\n') if d]

        for p_dir in p_dirs:
            if not os.path.isfile(pjoin(self.me_dir, 'SubProcesses', p_dir, 'orderstags_glob.dat')):
                files.ln(pjoin(self.me_dir, 'SubProcesses', 'orderstags_glob.dat'),
                         pjoin(self.me_dir, 'SubProcesses', p_dir))

        return


    def create_jobs_to_run(self,options,p_dirs,req_acc,run_mode,\
                           integration_step,mode,fixed_order=True):
        """Creates a list of dictionaries with all the jobs to be run"""
        jobs_to_run=[]
        if not options['only_generation']:
            # Fresh, new run. Check all the P*/channels.txt files
            # (created by the 'gensym' executable) to set-up all the
            # jobs using the default inputs.
            npoints = self.run_card['npoints_FO_grid']
            niters = self.run_card['niters_FO_grid']
            for p_dir in p_dirs:
                try:
                    with open(pjoin(self.me_dir,'SubProcesses',p_dir,'channels.txt')) as chan_file:
                        channels=chan_file.readline().split()
                except IOError:
                    logger.warning('No integration channels found for contribution %s (too large masses/low energy?)' % p_dir)
                    continue
                if fixed_order:
                    lch=len(channels)
                    maxchannels=20    # combine up to 20 channels in a single job
                    njobs=(int(lch/maxchannels)+1 if lch%maxchannels!= 0 \
                           else int(lch/maxchannels))
                    for nj in range(1,njobs+1):
                        job={}
                        job['p_dir']=p_dir
                        job['channel']=str(nj)
                        job['nchans']=(int(lch/njobs)+1 if nj <= lch%njobs else int(lch/njobs))
                        job['configs']=' '.join(channels[:job['nchans']])
                        del channels[:job['nchans']]
                        job['split']=0
                        if req_acc == -1:
                            job['accuracy']=0
                            job['niters']=niters
                            job['npoints']=npoints
                        elif req_acc > 0:
                            job['accuracy']=0.05
                            job['niters']=6
                            job['npoints']=-1
                        else:
                            raise aMCatNLOError('No consistent "req_acc_FO" set. Use a value '+
                                                'between 0 and 1 or set it equal to -1.')
                        job['mint_mode']=0
                        job['run_mode']=run_mode
                        job['wgt_frac']=1.0
                        job['wgt_mult']=1.0
                        jobs_to_run.append(job)
                    if channels:
                        raise aMCatNLOError('"channels" is not empty %s' % channels)
                else:
                    for channel in channels:
                        job={}
                        job['p_dir']=p_dir
                        job['channel']=channel
                        job['split']=0
                        job['accuracy']=0.03
                        job['niters']=12
                        job['npoints']=-1
                        job['mint_mode']=0
                        job['run_mode']=run_mode
                        job['wgt_frac']=1.0
                        jobs_to_run.append(job)
            jobs_to_collect=copy.copy(jobs_to_run) # These are all jobs
        else:
            # if options['only_generation'] is true, just read the current jobs from file
            try:
                with open(pjoin(self.me_dir,"SubProcesses","job_status.pkl"),'rb') as f:
                    jobs_to_collect=pickle.load(f)
                    for job in jobs_to_collect:
                        job['dirname']=pjoin(self.me_dir,'SubProcesses',job['dirname'].rsplit('/SubProcesses/',1)[1])
                jobs_to_run=copy.copy(jobs_to_collect)
            except:
                raise aMCatNLOError('Cannot reconstruct jobs from saved job status in %s' % \
                                    pjoin(self.me_dir,'SubProcesses','job_status.pkl'))
            # Update cross sections and determine which jobs to run next
            if fixed_order:
                jobs_to_run,jobs_to_collect=self.collect_the_results(options,req_acc,jobs_to_run,
                                                 jobs_to_collect,integration_step,mode,run_mode)
                # Update the integration_step to make sure that nothing will be overwritten
                integration_step=1
                for job in jobs_to_run:
                    while os.path.exists(pjoin(job['dirname'],'res_%s.dat' % integration_step)):
                        integration_step=integration_step+1
                integration_step=integration_step-1
            else:
                self.append_the_results(jobs_to_collect,integration_step)
        return jobs_to_run,jobs_to_collect,integration_step

    def prepare_directories(self,jobs_to_run,mode,fixed_order=True):
        """Set-up the G* directories for running"""
        name_suffix={'born' :'B' , 'all':'F'}
        for job in jobs_to_run:
            if job['split'] == 0:
                if fixed_order :
                    dirname=pjoin(self.me_dir,'SubProcesses',job['p_dir'],
                                  job['run_mode']+'_G'+job['channel'])
                else:
                    dirname=pjoin(self.me_dir,'SubProcesses',job['p_dir'],
                                  'G'+name_suffix[job['run_mode']]+job['channel'])
            else:
                if fixed_order :
                    dirname=pjoin(self.me_dir,'SubProcesses',job['p_dir'],
                            job['run_mode']+'_G'+job['channel']+'_'+str(job['split']))
                else:
                    dirname=pjoin(self.me_dir,'SubProcesses',job['p_dir'],
                            'G'+name_suffix[job['run_mode']]+job['channel']+'_'+str(job['split']))
            job['dirname']=dirname
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            self.write_input_file(job,fixed_order)
            # link or copy the grids from the base directory to the split directory:
            if not fixed_order:
                if job['split'] != 0:
                    for f in ['grid.MC_integer','mint_grids','res_1']:
                        if not os.path.isfile(pjoin(job['dirname'],f)):
                            files.ln(pjoin(job['dirname'].rsplit("_",1)[0],f),job['dirname'])
            else:
                if job['split'] != 0:
                    for f in ['grid.MC_integer','mint_grids']:
                        files.cp(pjoin(job['dirname'].rsplit("_",1)[0],f),job['dirname'])


    def write_input_file(self,job,fixed_order):
        """write the input file for the madevent_mint* executable in the appropriate directory"""
        if fixed_order:
            content= \
"""NPOINTS = %(npoints)s
NITERATIONS = %(niters)s
ACCURACY = %(accuracy)s
ADAPT_GRID = 2
MULTICHANNEL = 1
SUM_HELICITY = 1
NCHANS = %(nchans)s
CHANNEL = %(configs)s
SPLIT = %(split)s
WGT_MULT= %(wgt_mult)s
RUN_MODE = %(run_mode)s
RESTART = %(mint_mode)s
""" \
              % job
        else:
            if job['mint_mode'] == 0:
                job['fold_string']='1 1 1'
            else:
                job['fold_string']=' '.join(map(str, self.run_card['folding']))
            content = \
"""-1 12      ! points, iterations
%(accuracy)s       ! desired fractional accuracy
1 -0.1     ! alpha, beta for Gsoft
-1 -0.1    ! alpha, beta for Gazi
1          ! Suppress amplitude (0 no, 1 yes)?
1          ! Exact helicity sum (0 yes, n = number/event)?
%(channel)s          ! Enter Configuration Number:
%(mint_mode)s          ! MINT imode: 0 to set-up grids, 1 to perform integral, 2 generate events
%(fold_string)s      ! if imode is 1: Folding parameters for xi_i, y_ij and phi_i
%(run_mode)s        ! all, born, real, virt
""" \
                    % job
        with open(pjoin(job['dirname'], 'input_app.txt'), 'w') as input_file:
            input_file.write(content)


    def run_all_jobs(self,jobs_to_run,integration_step,fixed_order=True):
        """Loops over the jobs_to_run and executes them using the function 'run_exe'"""
        if fixed_order:
            if integration_step == 0:
                self.update_status('Setting up grids', level=None)
            else:
                self.update_status('Refining results, step %i' % integration_step, level=None)
        self.ijob = 0
        name_suffix={'born' :'B', 'all':'F'}
        if fixed_order:
            run_type="Fixed order integration step %s" % integration_step
        else:
            run_type="MINT step %s" % integration_step
        self.njobs=len(jobs_to_run)            
        for job in jobs_to_run:
            executable='ajob1'
            if fixed_order:
                arguments=[job['channel'],job['run_mode'], \
                                    str(job['split']),str(integration_step)]
            else:
                arguments=[job['channel'],name_suffix[job['run_mode']], \
                                    str(job['split']),str(integration_step)]
            self.run_exe(executable,arguments,run_type,
                         cwd=pjoin(self.me_dir,'SubProcesses',job['p_dir']))

        if self.cluster_mode == 2:
            time.sleep(1) # security to allow all jobs to be launched
        self.wait_for_complete(run_type)


    def collect_the_results(self,options,req_acc,jobs_to_run,jobs_to_collect,\
                            integration_step,mode,run_mode,fixed_order=True):
        """Collect the results, make HTML pages, print the summary and
           determine if there are more jobs to run. Returns the list
           of the jobs that still need to be run, as well as the
           complete list of jobs that need to be collected to get the
           final answer.
        """
# Get the results of the current integration/MINT step
        self.append_the_results(jobs_to_run,integration_step)
        self.cross_sect_dict = self.write_res_txt_file(jobs_to_collect,integration_step)
# Update HTML pages
        if fixed_order:
            cross, error = self.make_make_all_html_results(folder_names=['%s*' % run_mode], 
                                                           jobs=jobs_to_collect)
        else:
            name_suffix={'born' :'B' , 'all':'F'}
            cross, error = self.make_make_all_html_results(folder_names=['G%s*' % name_suffix[run_mode]])
        self.results.add_detail('cross', cross)
        self.results.add_detail('error', error)
# Combine grids from split fixed order jobs
        if fixed_order:
            jobs_to_run=self.combine_split_order_run(jobs_to_run)
# Set-up jobs for the next iteration/MINT step
        jobs_to_run_new=self.update_jobs_to_run(req_acc,integration_step,jobs_to_run,fixed_order)
        # IF THERE ARE NO MORE JOBS, WE ARE DONE!!!
        if fixed_order:
            # Write the jobs_to_collect directory to file so that we
            # can restart them later (with only-generation option)
            with open(pjoin(self.me_dir,"SubProcesses","job_status.pkl"),'wb') as f:
                pickle.dump(jobs_to_collect,f)
# Print summary
        if (not jobs_to_run_new) and fixed_order:
            # print final summary of results (for fixed order)
            scale_pdf_info=self.collect_scale_pdf_info(options,jobs_to_collect)
            self.print_summary(options,integration_step,mode,scale_pdf_info,done=True)
            return jobs_to_run_new,jobs_to_collect
        elif jobs_to_run_new:
            # print intermediate summary of results
            scale_pdf_info=[]
            self.print_summary(options,integration_step,mode,scale_pdf_info,done=False)
        else:
            # When we are done for (N)LO+PS runs, do not print
            # anything yet. This will be done after the reweighting
            # and collection of the events
            scale_pdf_info=[]
# Prepare for the next integration/MINT step
        if (not fixed_order) and integration_step+1 == 2 :
            # Write the jobs_to_collect directory to file so that we
            # can restart them later (with only-generation option)
            with open(pjoin(self.me_dir,"SubProcesses","job_status.pkl"),'wb') as f:
                pickle.dump(jobs_to_collect,f)
            # next step is event generation (mint_step 2)
            jobs_to_run_new,jobs_to_collect_new= \
                    self.check_the_need_to_split(jobs_to_run_new,jobs_to_collect)
            self.prepare_directories(jobs_to_run_new,mode,fixed_order)
            self.write_nevents_unweighted_file(jobs_to_collect_new,jobs_to_collect)
            self.write_nevts_files(jobs_to_run_new)
        else:
            if fixed_order and self.run_card['req_acc_FO'] > 0:
                jobs_to_run_new,jobs_to_collect= \
                    self.split_jobs_fixed_order(jobs_to_run_new,jobs_to_collect)
            self.prepare_directories(jobs_to_run_new,mode,fixed_order)
            jobs_to_collect_new=jobs_to_collect
        if fixed_order:
            # Write the jobs_to_collect directory to file so that we
            # can collect plots (by hand) even if there was some
            # error. Mainly for debugging only. Normally this file
            # should not be used. (Rather, use 'job_status.pkl' which
            # is only written if all jobs are finished correctly)
            with open(pjoin(self.me_dir,"SubProcesses","job_status2.pkl"),'wb') as f:
                pickle.dump(jobs_to_collect_new,f)
        return jobs_to_run_new,jobs_to_collect_new


    def write_nevents_unweighted_file(self,jobs,jobs0events):
        """writes the nevents_unweighted file in the SubProcesses directory.
           We also need to write the jobs that will generate 0 events,
           because that makes sure that the cross section from those channels
           is taken into account in the event weights (by collect_events.f).
        """
        content=[]
        for job in jobs:
            path=pjoin(job['dirname'].split('/')[-2],job['dirname'].split('/')[-1])
            lhefile=pjoin(path,'events.lhe')
            content.append(' %s     %d     %9e     %9e' % \
                (lhefile.ljust(40),job['nevents'],job['resultABS']*job['wgt_frac'],job['wgt_frac']))
        for job in jobs0events:
            if job['nevents']==0:
                path=pjoin(job['dirname'].split('/')[-2],job['dirname'].split('/')[-1])
                lhefile=pjoin(path,'events.lhe')
                content.append(' %s     %d     %9e     %9e' % \
                               (lhefile.ljust(40),job['nevents'],job['resultABS'],1.))
        with open(pjoin(self.me_dir,'SubProcesses',"nevents_unweighted"),'w') as f:
            f.write('\n'.join(content)+'\n')

    def write_nevts_files(self,jobs):
        """write the nevts files in the SubProcesses/P*/G*/ directories"""
        for job in jobs:
            with open(pjoin(job['dirname'],'nevts'),'w') as f:
                if self.run_card['event_norm'].lower()=='bias':
                    f.write('%i %f\n' % (job['nevents'],self.cross_sect_dict['xseca']))
                else:
                    f.write('%i\n' % job['nevents'])

    def combine_split_order_run(self,jobs_to_run):
        """Combines jobs and grids from split jobs that have been run"""
        # combine the jobs that need to be combined in job
        # groups. Simply combine the ones that have the same p_dir and
        # same channel. 
        jobgroups_to_combine=[]
        jobs_to_run_new=[]
        for job in jobs_to_run:
            if job['split'] == 0:
                job['combined']=1
                jobs_to_run_new.append(job) # this jobs wasn't split
            elif job['split'] == 1:
                jobgroups_to_combine.append([j for j in jobs_to_run if j['p_dir'] == job['p_dir'] and \
                                            j['channel'] == job['channel']])
            else:
                continue
        for job_group in jobgroups_to_combine:
            # Combine the grids (mint-grids & MC-integer grids) first
            self.combine_split_order_grids(job_group)
            jobs_to_run_new.append(self.combine_split_order_jobs(job_group))
        return jobs_to_run_new

    def combine_split_order_jobs(self,job_group):
        """combine the jobs in job_group and return a single summed job"""
        # first copy one of the jobs in 'jobs'
        sum_job=copy.copy(job_group[0])
        # update the information to have a 'non-split' job:
        sum_job['dirname']=pjoin(sum_job['dirname'].rsplit('_',1)[0])
        sum_job['split']=0
        sum_job['wgt_mult']=1.0
        sum_job['combined']=len(job_group)
        # information to be summed:
        keys=['niters_done','npoints_done','niters','npoints',\
              'result','resultABS','time_spend']
        keys2=['error','errorABS']
        # information to be summed in quadrature:
        for key in keys2:
            sum_job[key]=math.pow(sum_job[key],2)
        # Loop over the jobs and sum the information
        for i,job in enumerate(job_group):
            if i==0 : continue # skip the first
            for key in keys:
                sum_job[key]+=job[key]
            for key in keys2:
                sum_job[key]+=math.pow(job[key],2)
        for key in keys2:
            sum_job[key]=math.sqrt(sum_job[key])
        sum_job['err_percABS'] = sum_job['errorABS']/sum_job['resultABS']*100.
        sum_job['err_perc'] = sum_job['error']/sum_job['result']*100.
        sum_job['niters']=int(sum_job['niters_done']/len(job_group))
        sum_job['niters_done']=int(sum_job['niters_done']/len(job_group))
        return sum_job

    
    def combine_split_order_grids(self,job_group):
        """Combines the mint_grids and MC-integer grids from the split order
        jobs (fixed order only).
        """
        files_mint_grids=[]
        files_MC_integer=[]
        location=None
        for job in job_group:
            files_mint_grids.append(pjoin(job['dirname'],'mint_grids'))
            files_MC_integer.append(pjoin(job['dirname'],'grid.MC_integer'))
            if not location:
                location=pjoin(job['dirname'].rsplit('_',1)[0])
            else:
                if location != pjoin(job['dirname'].rsplit('_',1)[0]) :
                    raise aMCatNLOError('Not all jobs have the same location. '\
                                        +'Cannot combine them.')
        # Needed to average the grids (both xgrids, ave_virt and
        # MC_integer grids), but sum the cross section info. The
        # latter is only the only line that contains integers.
        for j,fs in enumerate([files_mint_grids,files_MC_integer]):
            linesoffiles=[]
            polyfit_data=[]
            for f in fs:
                with open(f,'r+') as fi:
                    data=fi.readlines()
                    linesoffiles.append([ dat for dat in data if 'POL' not in dat.split()[0] ])
                    polyfit_data.append([ dat for dat in data if 'POL'     in dat.split()[0] ])
            to_write=[]
            for rowgrp in zip(*linesoffiles):
                action=misc.make_unique([row.strip().split()[0] for row in rowgrp]) 
                floatsbyfile = [[float(a) for a in row.strip().split()[1:]] for row in rowgrp]
                floatgrps = list(zip(*floatsbyfile))
                if len(action) != 1:
                    raise aMCatNLOError('"mint_grids" files not in correct format. '+\
                                        'Cannot combine them.')
                if 'AVE' in action:
                    # average
                    write_string = [sum(floatgrp)/len(floatgrp) for floatgrp in floatgrps]
                elif 'SUM' in action:
                    # sum
                    write_string = [sum(floatgrp) for floatgrp in floatgrps]
                elif 'MAX' in action:
                    # take maximum
                    write_string = [max(floatgrp) for floatgrp in floatgrps]
                elif 'QSM' in action:
                    # sum in quadrature
                    write_string = [math.sqrt(sum([err**2 for err in floatgrp])) for floatgrp in floatgrps]
                elif 'IDE' in action:
                    # they should all be identical (INTEGERS)
                    write_string = [int(round(floatgrp[0])) for floatgrp in floatgrps]
                elif 'SPE' in action:
                    # special: average first; sum second; average third (ALL INTEGERS)
                    write_string=[]
                    for i,floatgrp in enumerate(floatgrps):
                        if i==0: # average number of PS points per iterations
                            write_string.append(int(sum(floatgrp)/len(floatgrp)))
                        elif i==1: # sum te number of iterations
                            write_string.append(int(sum(floatgrp)))
                        elif i==2: # average the nhits_in_grids
                            write_string.append(int(sum(floatgrp)/len(floatgrp)))
                else:
                    raise aMCatNLOError('Unknown action for combining grids: %s' % action[0])
                to_write.append(action[0] + " " + (" ".join(str(ws) for ws in write_string)) + "\n")

            if polyfit_data:
                # special for data regarding virtuals. Need to simply append
                # all the data, but skipping doubles.
                for i,onefile in enumerate(polyfit_data):
                    # Get the number of channels, and the number of PS points per channel
                    data_amount_in_file=[int(oneline.split()[1]) for oneline in onefile if len(oneline.split())==2]
                    if i==0:
                        filtered_list=[ [] for i in range(len(data_amount_in_file)) ]
                    start=len(data_amount_in_file)
                    for channel,channel_size in enumerate(data_amount_in_file):
                        end=start+channel_size
                        for data_channel in onefile[start:end]:
                            if data_channel not in filtered_list[channel]:
                                filtered_list[channel].append(data_channel)
                        start=end
                # The amount of data in each file (per channel):
                for channel in filtered_list:
                    to_write.append("POL " + str(len(channel)) + "\n")
                # All the data:
                for ch in filtered_list:
                    for dat in ch:
                        to_write.append(dat)

            # write the data over the master location
            if j==0:
                with open(pjoin(location,'mint_grids'),'w') as f:
                    f.writelines(to_write)
            elif j==1:
                with open(pjoin(location,'grid.MC_integer'),'w') as f:
                    f.writelines(to_write)

                
    def split_jobs_fixed_order(self,jobs_to_run,jobs_to_collect):
        """Looks in the jobs_to_run to see if there is the need to split the
           jobs, depending on the expected time they take. Updates
           jobs_to_run and jobs_to_collect to replace the split-job by
           its splits.
        """
        # determine the number jobs we should have (this is per p_dir)
        if self.options['run_mode'] ==2:
            nb_submit = int(self.options['nb_core'])
        elif self.options['run_mode'] ==1:
            nb_submit = int(self.options['cluster_size'])
        else:
            nb_submit =1 
        # total expected aggregated running time
        time_expected=0
        for job in jobs_to_run:
            time_expected+=job['time_spend']*(job['niters']*job['npoints'])/  \
                           (job['niters_done']*job['npoints_done'])
        # this means that we must expect the following per job (in
        # ideal conditions)
        time_per_job=time_expected/(nb_submit*(1+len(jobs_to_run)/2))
        jobs_to_run_new=[]
        jobs_to_collect_new=[job for job in jobs_to_collect if job['resultABS']!=0]
        for job in jobs_to_run:
            # remove current job from jobs_to_collect. Make sure
            # to remove all the split ones in case the original
            # job had been a split one (before it was re-combined)
            for j in [j for j in jobs_to_collect_new if j['p_dir'] == job['p_dir'] and \
                                j['channel'] == job['channel']]:
                jobs_to_collect_new.remove(j)
            time_expected=job['time_spend']*(job['niters']*job['npoints'])/  \
                           (job['niters_done']*job['npoints_done'])
            # if the time expected for this job is (much) larger than
            # the time spend in the previous iteration, and larger
            # than the expected time per job, split it
            if time_expected > max(2*job['time_spend']/job['combined'],time_per_job) \
                    or job['npoints'] >= __maxint__:
                # determine the number of splits needed; the second condition 
                # (job['npoints'] >= __maxint__) prevents integer overflow in fortran
                nsplit = min(max(int(time_expected/max(2*job['time_spend']/job['combined'],time_per_job)),2),nb_submit)
                nsplit*= int(job['npoints'] / __maxint__) + 1
                for i in range(1,nsplit+1):
                    job_new=copy.copy(job)
                    job_new['split']=i
                    job_new['wgt_mult']=1./float(nsplit)
                    job_new['dirname']=job['dirname']+'_%i' % job_new['split']
                    job_new['accuracy']=min(job['accuracy']*math.sqrt(float(nsplit)),0.1)
                    if nsplit >= job['niters']:
                        job_new['npoints']=int(job['npoints']*job['niters']/nsplit)
                        job_new['niters']=1
                    else:
                        job_new['npoints']=int(job['npoints']/nsplit)

                    if job_new['npoints'] > __maxint__:
                        raise aMCatNLOError('Too many point for the job. Fortran will likely crash' + \
                                        'for integer overflow. %d' % job_new['npoints'])

                    jobs_to_collect_new.append(job_new)
                    jobs_to_run_new.append(job_new)
            else:
                jobs_to_collect_new.append(job)
                jobs_to_run_new.append(job)
        return jobs_to_run_new,jobs_to_collect_new
                
        
    def check_the_need_to_split(self,jobs_to_run,jobs_to_collect):
        """Looks in the jobs_to_run to see if there is the need to split the
           event generation step. Updates jobs_to_run and
           jobs_to_collect to replace the split-job by its
           splits. Also removes jobs that do not need any events.
        """
        nevt_job=self.run_card['nevt_job']
        if nevt_job > 0:
            jobs_to_collect_new=copy.copy(jobs_to_collect)
            for job in jobs_to_run:
                nevents=job['nevents']
                if nevents == 0:
                    jobs_to_collect_new.remove(job)
                elif nevents > nevt_job:
                    jobs_to_collect_new.remove(job)
                    if nevents % nevt_job != 0 :
                        nsplit=int(nevents/nevt_job)+1
                    else:
                        nsplit=int(nevents/nevt_job)
                    for i in range(1,nsplit+1):
                        job_new=copy.copy(job)
                        left_over=nevents % nsplit
                        if i <= left_over:
                            job_new['nevents']=int(nevents/nsplit)+1
                            job_new['wgt_frac']=float(job_new['nevents'])/float(nevents)
                        else:
                            job_new['nevents']=int(nevents/nsplit)
                            job_new['wgt_frac']=float(job_new['nevents'])/float(nevents)
                        job_new['split']=i
                        job_new['dirname']=job['dirname']+'_%i' % job_new['split']
                        jobs_to_collect_new.append(job_new)
            jobs_to_run_new=copy.copy(jobs_to_collect_new)
        else:
            jobs_to_run_new=copy.copy(jobs_to_collect)
            for job in jobs_to_collect:
                if job['nevents'] == 0:
                    jobs_to_run_new.remove(job)
            jobs_to_collect_new=copy.copy(jobs_to_run_new)

        return jobs_to_run_new,jobs_to_collect_new
    

    def update_jobs_to_run(self,req_acc,step,jobs,fixed_order=True):
        """
        For (N)LO+PS:    determines the number of events and/or the required
                         accuracy per job.
        For fixed order: determines which jobs need higher precision and
                         returns those with the newly requested precision.
        """
        err=self.cross_sect_dict['errt']
        tot=self.cross_sect_dict['xsect']
        errABS=self.cross_sect_dict['erra']
        totABS=self.cross_sect_dict['xseca']
        jobs_new=[]
        if fixed_order:
            if req_acc == -1:
                if step+1 == 1:
                    npoints = self.run_card['npoints_FO']
                    niters = self.run_card['niters_FO']
                    for job in jobs:
                        job['mint_mode']=-1
                        job['niters']=niters
                        job['npoints']=npoints
                        jobs_new.append(job)
                elif step+1 == 2:
                    pass
                elif step+1 > 2:
                    raise aMCatNLOError('Cannot determine number of iterations and PS points '+
                                        'for integration step %i' % step )
            elif ( req_acc > 0 and err/abs(tot) > req_acc*1.2 ) or step <= 0:
                req_accABS=req_acc*abs(tot)/totABS # overal relative required accuracy on ABS Xsec.
                for job in jobs:
                    # skip jobs with 0 xsec
                    if job['resultABS'] == 0.:
                        continue
                    job['mint_mode']=-1
                    # Determine relative required accuracy on the ABS for this job
                    job['accuracy']=req_accABS*math.sqrt(totABS/job['resultABS'])
                    # If already accurate enough, skip the job
                    if job['accuracy'] > job['errorABS']/job['resultABS'] and step != 0:
                            continue
                    # Update the number of PS points based on errorABS, ncall and accuracy
                    itmax_fl=job['niters_done']*math.pow(job['errorABS']/
                                                         (job['accuracy']*job['resultABS']),2)
                    itmax_fl=itmax_fl*1.1 # add 10% to make sure to have enough
                    if itmax_fl <= 4.0 :
                        job['niters']=max(int(round(itmax_fl)),2)
                        job['npoints']=job['npoints_done']*2
                    elif itmax_fl > 4.0 and itmax_fl <= 16.0 :
                        job['niters']=4
                        job['npoints']=int(round(job['npoints_done']*itmax_fl/4.0))*2
                    else:
                        if itmax_fl > 100.0 : itmax_fl=50.0
                        job['niters']=int(round(math.sqrt(itmax_fl)))
                        job['npoints']=int(round(job['npoints_done']*itmax_fl/
                                                 round(math.sqrt(itmax_fl))))*2
                    # Add the job to the list of jobs that need to be run
                    jobs_new.append(job)
            return jobs_new
        elif step+1 <= 2:
            nevents=self.run_card['nevents']
            # Total required accuracy for the upper bounding envelope
            if req_acc<0: 
                req_acc2_inv=nevents
            else:
                req_acc2_inv=1/(req_acc*req_acc)
            if step+1 == 1 or step+1 == 2 :
                # determine the req. accuracy for each of the jobs for Mint-step = 1
                for job in jobs:
                    accuracy=min(math.sqrt(totABS/(req_acc2_inv*job['resultABS'])),0.2)
                    job['accuracy']=accuracy
            if step+1 == 2:
                # Randomly (based on the relative ABS Xsec of the job) determine the 
                # number of events each job needs to generate for MINT-step = 2.
                r=self.get_randinit_seed()
                if not hasattr(random, 'mg_seedset'):
                    random.seed(r)  
                    random.mg_seedset = r
                totevts=nevents
                for job in jobs:
                    job['nevents'] = 0
                while totevts :
                    target = random.random() * totABS
                    crosssum = 0.
                    i = 0
                    while i<len(jobs) and crosssum < target:
                        job = jobs[i]
                        crosssum += job['resultABS']
                        i += 1            
                    totevts -= 1
                    i -= 1
                    jobs[i]['nevents'] += 1
            for job in jobs:
                job['mint_mode']=step+1 # next step
            return jobs
        else:
            return []


    def get_randinit_seed(self):
        """ Get the random number seed from the randinit file """
        with open(pjoin(self.me_dir,"SubProcesses","randinit")) as randinit:
            # format of the file is "r=%d".
            iseed = int(randinit.read()[2:])
        return iseed


    def append_the_results(self,jobs,integration_step):
        """Appends the results for each of the jobs in the job list"""
        error_found=False
        for job in jobs:
            try:
                if integration_step >= 0 :
                    with open(pjoin(job['dirname'],'res_%s.dat' % integration_step)) as res_file:
                        results=res_file.readline().split()
                else:
                # should only be here when doing fixed order with the 'only_generation'
                # option equal to True. Take the results from the final run done.
                    with open(pjoin(job['dirname'],'res.dat')) as res_file:
                        results=res_file.readline().split()
            except IOError:
                if not error_found:
                    error_found=True
                    error_log=[]
                error_log.append(pjoin(job['dirname'],'log.txt'))
                continue
            job['resultABS']=float(results[0])
            job['errorABS']=float(results[1])
            job['result']=float(results[2])
            job['error']=float(results[3])
            job['niters_done']=int(results[4])
            job['npoints_done']=int(results[5])
            job['time_spend']=float(results[6])
            if job['resultABS'] != 0:
                job['err_percABS'] = job['errorABS']/job['resultABS']*100.
                job['err_perc'] = job['error']/job['result']*100.
            else:
                job['err_percABS'] = 0.
                job['err_perc'] = 0.
        if error_found:
            raise aMCatNLOError('An error occurred during the collection of results.\n' + 
                   'Please check the .log files inside the directories which failed:\n' +
                                '\n'.join(error_log)+'\n')



    def write_res_txt_file(self,jobs,integration_step):
        """writes the res.txt files in the SubProcess dir"""
        jobs.sort(key = lambda job: -job['errorABS'])
        content=[]
        content.append('\n\nCross section per integration channel:')
        for job in jobs:
            content.append('%(p_dir)20s  %(channel)15s   %(result)10.8e    %(error)6.4e       %(err_perc)6.4f%%  ' %  job)
        content.append('\n\nABS cross section per integration channel:')
        for job in jobs:
            content.append('%(p_dir)20s  %(channel)15s   %(resultABS)10.8e    %(errorABS)6.4e       %(err_percABS)6.4f%%  ' %  job)
        # print also statistics for each directory
        dir_dict={}
        for job in jobs:
            try:
                dir_dict[job['p_dir']]['result'] += job['result']*job['wgt_frac']
                dir_dict[job['p_dir']]['resultABS'] += job['resultABS']*job['wgt_frac']
                # store the error ^2
                dir_dict[job['p_dir']]['error'] += math.pow(job['error'], 2)*job['wgt_frac']
                dir_dict[job['p_dir']]['errorABS'] += math.pow(job['errorABS'], 2)*job['wgt_frac']
            except KeyError:
                dir_dict[job['p_dir']] = {
                        'result' : job['result']*job['wgt_frac'],
                        'resultABS' : job['resultABS']*job['wgt_frac'],
                        'error' : math.pow(job['error'], 2)*job['wgt_frac'],
                        'errorABS' : math.pow(job['errorABS'], 2)*job['wgt_frac']}

        for dir_res in dir_dict.values():
            dir_res['error'] = math.sqrt(dir_res['error'])
            dir_res['errorABS'] = math.sqrt(dir_res['errorABS'])
        content.append('\n\nABS cross section per dir')
        for ddir, res in dir_dict.items():
            content.append(('%20s' % ddir) + '   %(resultABS)10.8e    %(errorABS)6.4e ' %  res)
        content.append('\n\nCross section per dir')
        for ddir, res in dir_dict.items():
            content.append(('%20s' % ddir) + '   %(result)10.8e    %(error)6.4e ' %  res)

        totABS=0
        errABS=0
        tot=0
        err=0
        for job in jobs:
            totABS+= job['resultABS']*job['wgt_frac']
            errABS+= math.pow(job['errorABS'],2)*job['wgt_frac']
            tot+= job['result']*job['wgt_frac']
            err+= math.pow(job['error'],2)*job['wgt_frac']
        if jobs:
            content.append('\nTotal ABS and \nTotal: \n                      %10.8e +- %6.4e  (%6.4e%%)\n                      %10.8e +- %6.4e  (%6.4e%%) \n' %\
                           (totABS, math.sqrt(errABS), math.sqrt(errABS)/totABS *100.,\
                            tot, math.sqrt(err), math.sqrt(err)/tot *100.))
        with open(pjoin(self.me_dir,'SubProcesses','res_%s.txt' % integration_step),'w') as res_file:
            res_file.write('\n'.join(content))
        randinit=self.get_randinit_seed()
        return {'xsect':tot,'xseca':totABS,'errt':math.sqrt(err),\
                'erra':math.sqrt(errABS),'randinit':randinit}
        

    def collect_scale_pdf_info(self,options,jobs):
        """read the scale_pdf_dependence.dat files and collects there results"""
        scale_pdf_info=[]
        if any(self.run_card['reweight_scale']) or any(self.run_card['reweight_PDF']) or \
           len(self.run_card['dynamical_scale_choice']) > 1 or len(self.run_card['lhaid']) > 1:
            evt_files=[]
            evt_wghts=[]
            for job in jobs:
                evt_files.append(pjoin(job['dirname'],'scale_pdf_dependence.dat'))
                evt_wghts.append(job['wgt_frac'])
            scale_pdf_info = self.pdf_scale_from_reweighting(evt_files,evt_wghts)
        return scale_pdf_info


    def combine_plots_FO(self,folder_name,jobs):
        """combines the plots and puts then in the Events/run* directory"""
        devnull = open(os.devnull, 'w') 
        
        if self.analyse_card['fo_analysis_format'].lower() == 'topdrawer':
            topfiles = []
            for job in jobs:
                if job['dirname'].endswith('.top'):
                    topfiles.append(job['dirname'])
                else:
                    topfiles.append(pjoin(job['dirname'],'MADatNLO.top'))
            misc.call(['./combine_plots_FO.sh'] + topfiles, \
                      stdout=devnull, 
                      cwd=pjoin(self.me_dir, 'SubProcesses'))
            files.cp(pjoin(self.me_dir, 'SubProcesses', 'MADatNLO.top'),
                     pjoin(self.me_dir, 'Events', self.run_name))
            logger.info('The results of this run and the TopDrawer file with the plots' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))
        elif self.analyse_card['fo_analysis_format'].lower() == 'hwu':
            out=pjoin(self.me_dir,'Events',self.run_name,'MADatNLO')
            self.combine_plots_HwU(jobs,out)
            try:
                misc.call(['gnuplot','MADatNLO.gnuplot'],\
                          stdout=devnull,stderr=devnull,\
                          cwd=pjoin(self.me_dir, 'Events', self.run_name))
            except Exception:
                pass
            logger.info('The results of this run and the HwU and GnuPlot files with the plots' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))
        elif self.analyse_card['fo_analysis_format'].lower() == 'root':
            rootfiles = []
            for job in jobs:
                if job['dirname'].endswith('.root'):
                    rootfiles.append(job['dirname'])
                else:
                    rootfiles.append(pjoin(job['dirname'],'MADatNLO.root'))
            misc.call(['./combine_root.sh'] + folder_name + rootfiles, \
                      stdout=devnull, 
                      cwd=pjoin(self.me_dir, 'SubProcesses'))
            files.cp(pjoin(self.me_dir, 'SubProcesses', 'MADatNLO.root'),
                     pjoin(self.me_dir, 'Events', self.run_name))
            logger.info('The results of this run and the ROOT file with the plots' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))
        elif self.analyse_card['fo_analysis_format'].lower() == 'lhe':
            self.combine_FO_lhe(jobs)
            logger.info('The results of this run and the LHE File (to be used for plotting only)' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))            
        else:
            logger.info('The results of this run' + \
                        ' have been saved in %s' % pjoin(self.me_dir, 'Events', self.run_name))

    def combine_FO_lhe(self,jobs):
        """combine the various lhe file generated in each directory.
           They are two steps:
           1) banner 
           2) reweight each sample by the factor written at the end of each file
           3) concatenate each of the new files (gzip those).
        """
        
        logger.info('Combining lhe events for plotting analysis')
        start = time.time()
        self.run_card['fo_lhe_postprocessing'] = [i.lower() for i in self.run_card['fo_lhe_postprocessing']]
        output = pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe.gz')
        if os.path.exists(output):
            os.remove(output)
        

        
        
        # 1. write the banner
        text = open(pjoin(jobs[0]['dirname'],'header.txt'),'r').read()
        i1, i2 = text.find('<initrwgt>'),text.find('</initrwgt>') 
        self.banner['initrwgt'] = text[10+i1:i2]
#        
#        <init>
#        2212 2212 6.500000e+03 6.500000e+03 0 0 247000 247000 -4 1
#        8.430000e+02 2.132160e+00 8.430000e+02 1
#        <generator name='MadGraph5_aMC@NLO' version='2.5.2'>please cite 1405.0301 </generator>
#        </init>

        cross = sum(j['result'] for j in jobs)
        error = math.sqrt(sum(j['error'] for j in jobs))
        self.banner['init'] = "0 0 0e0 0e0 0 0 0 0 -4 1\n  %s %s %s 1" % (cross, error, cross)
        self.banner.write(output[:-3], close_tag=False)
        misc.gzip(output[:-3])
        
        
        
        fsock = lhe_parser.EventFile(output,'a')
        if 'nogrouping' in self.run_card['fo_lhe_postprocessing']:
            fsock.eventgroup = False
        else:
            fsock.eventgroup = True
        
        if 'norandom' in self.run_card['fo_lhe_postprocessing']:
            for job in jobs:
                dirname = job['dirname']
                #read last line
                lastline = misc.BackRead(pjoin(dirname,'events.lhe')).readline()
                nb_event, sumwgt, cross = [float(i) for i in lastline.split()]
                # get normalisation ratio 
                ratio = cross/sumwgt
                lhe = lhe_parser.EventFile(pjoin(dirname,'events.lhe'))
                lhe.eventgroup = True # read the events by eventgroup
                for eventsgroup in lhe:
                    neweventsgroup = []
                    for i,event in enumerate(eventsgroup):
                        event.rescale_weights(ratio)
                        if i>0 and 'noidentification' not in self.run_card['fo_lhe_postprocessing'] \
                                                and event == neweventsgroup[-1]:
                            neweventsgroup[-1].wgt += event.wgt
                            for key in event.reweight_data:
                                neweventsgroup[-1].reweight_data[key] += event.reweight_data[key]
                        else:
                            neweventsgroup.append(event)
                    fsock.write_events(neweventsgroup)
                lhe.close()
                os.remove(pjoin(dirname,'events.lhe'))
        else:
            lhe = []
            lenlhe = []     
            misc.sprint('Need to combine %s event files' % len(jobs))
            globallhe = lhe_parser.MultiEventFile()
            globallhe.eventgroup = True
            for job in jobs:
                dirname = job['dirname']
                lastline = misc.BackRead(pjoin(dirname,'events.lhe')).readline()
                nb_event, sumwgt, cross = [float(i) for i in lastline.split()]
                lastlhe = globallhe.add(pjoin(dirname,'events.lhe'),cross, 0, cross,
                                        nb_event=int(nb_event), scale=cross/sumwgt)
            for eventsgroup in globallhe:
                neweventsgroup = []
                for i,event in enumerate(eventsgroup):
                    event.rescale_weights(event.sample_scale)
                    if i>0 and 'noidentification' not in self.run_card['fo_lhe_postprocessing'] \
                                            and event == neweventsgroup[-1]:
                        neweventsgroup[-1].wgt += event.wgt
                        for key in event.reweight_data:
                            neweventsgroup[-1].reweight_data[key] += event.reweight_data[key]
                    else:
                        neweventsgroup.append(event) 
                fsock.write_events(neweventsgroup)               
            globallhe.close()
            fsock.write('</LesHouchesEvents>\n') 
            fsock.close()
            misc.sprint('The combining of the LHE files has taken ', time.time()-start)
            for job in jobs:
                dirname = job['dirname']
                os.remove(pjoin(dirname,'events.lhe'))
                                
                                           
                        
        misc.sprint('The combining of the LHE files has taken ', time.time()-start)
                        
                        
                    

            
            
    def combine_plots_HwU(self,jobs,out,normalisation=None):
        """Sums all the plots in the HwU format."""
        logger.debug('Combining HwU plots.')

        command =  [sys.executable]
        command.append(pjoin(self.me_dir, 'bin', 'internal','histograms.py'))
        for job in jobs:
            if job['dirname'].endswith('.HwU'):
                command.append(job['dirname'])
            else:
                command.append(pjoin(job['dirname'],'MADatNLO.HwU'))
        command.append("--out="+out)
        command.append("--gnuplot")
        command.append("--band=[]")
        command.append("--lhapdf-config="+self.options['lhapdf'])
        if normalisation:
            command.append("--multiply="+(','.join([str(n) for n in normalisation])))
        command.append("--sum")
        command.append("--keep_all_weights")
        command.append("--no_open")

        p = misc.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, cwd=self.me_dir)

        while p.poll() is None:
            line = p.stdout.readline().decode(errors='ignore')
            #misc.sprint(type(line))
            if any(t in line for t in ['INFO:','WARNING:','CRITICAL:','ERROR:','KEEP:']):
                print(line[:-1])
            elif __debug__ and line:
                logger.debug(line[:-1])

            
    def pineappl_combine(self,cross,error,jobs):
        """Combines the PineAPPL grids in all the SubProcess/P*/all_G*/ directories"""
        logger.debug('Combining PineAPPL grids \n')
        all_jobs=[]
        for job in jobs:
            if job['resultABS'] == 0.0:
                logger.warning('pineappl_combine: Job\n%s\nwill be skipped, as it returned zero cross-section' %
                        job['dirname'])
                continue
            all_jobs.append(job['dirname'])
        ngrids=len(all_jobs)
        nobs  =len([name for name in os.listdir(all_jobs[0]) if name.endswith("_out.pineappl")])
        for obs in range(0,nobs):
            gdir = [pjoin(job,"grid_obs_"+str(obs)+"_out.pineappl") for job in all_jobs]
            # combine PineAPPL grid from different channels for observable 'obs'
            if self.run_card["pineappl"]:
                unc2_inv=pow(cross/error,2)
                unc2_inv_ngrids=pow(cross/error,2)*ngrids
                misc.call(['pineappl','merge', pjoin(self.me_dir,"Events",
                        self.run_name,"amcblast_obs_"+str(obs)+".pineappl")]+ gdir)

            # after combining, delete the original grids
            for ggdir in gdir:
                os.remove(ggdir)


    def collect_log_files(self, jobs, integration_step):
        """collect the log files and put them in a single, html-friendly file
        inside the Events/run_.../ directory"""
        log_file = pjoin(self.me_dir, 'Events', self.run_name, 
                         'alllogs_%d.html' % integration_step)
        outfile = open(log_file, 'w')

        content = ''
        content += '<HTML><BODY>\n<font face="courier" size=2>'
        for job in jobs:
            # put an anchor
            log=pjoin(job['dirname'],'log_MINT%s.txt' % integration_step)
            content += '<a name=%s></a>\n' % (os.path.dirname(log).replace(
                                          pjoin(self.me_dir,'SubProcesses'),''))
            # and put some nice header
            content += '<font color="red">\n'
            content += '<br>LOG file for integration channel %s, %s <br>' % \
                    (os.path.dirname(log).replace(pjoin(self.me_dir,
                                                           'SubProcesses'), ''), 
                     integration_step)
            content += '</font>\n'
            #then just flush the content of the small log inside the big log
            #the PRE tag prints everything verbatim
            with open(log) as l:
                content += '<PRE>\n' + l.read() + '\n</PRE>'
            content +='<br>\n'
            outfile.write(content)
            content=''

        outfile.write('</font>\n</BODY></HTML>\n')
        outfile.close()


    def finalise_run_FO(self,folder_name,jobs):
        """Combine the plots and put the res*.txt files in the Events/run.../ folder."""
        # Copy the res_*.txt files to the Events/run* folder
        res_files = misc.glob('res_*.txt', pjoin(self.me_dir, 'SubProcesses'))
        for res_file in res_files:
            files.mv(res_file,pjoin(self.me_dir, 'Events', self.run_name))
        # Collect the plots and put them in the Events/run* folder
        self.combine_plots_FO(folder_name,jobs)
        # If PineAPPL is linked, combine the grid to be put inside Events/run_XX
        if self.run_card['pineappl']:
            cross=self.cross_sect_dict['xsect']
            error=self.cross_sect_dict['errt']
            self.pineappl_combine(cross,error,jobs)


    def setup_cluster_or_multicore(self):
        """setup the number of cores for multicore, and the cluster-type for cluster runs"""
        if self.cluster_mode == 1:
            cluster_name = self.options['cluster_type']
            try:
                self.cluster = cluster.from_name[cluster_name](**self.options)
            except KeyError:
                # Check if a plugin define this type of cluster
                # check for PLUGIN format
                cluster_class = misc.from_plugin_import(self.plugin_path, 
                                            'new_cluster', cluster_name,
                                            info = 'cluster handling will be done with PLUGIN: %(plug)s' )
                if cluster_class:
                    self.cluster = cluster_class(**self.options)
        
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
                logger.warning('Impossible to detect the number of cores => Using one.\n'+
                        'Use set nb_core X in order to set this number and be able to '+
                        'run in multicore.')
            if not (hasattr(self, 'cluster') or not isinstance(self.cluster, cluster.MultiCore)):                
                self.cluster = cluster.MultiCore(**self.options)
                self.cluster.nb_core = self.nb_core
            else:
                self.cluster.nb_core = self.nb_core



    def clean_previous_results(self,options,p_dirs,folder_name):
        """Clean previous results.
             o.  If doing only the reweighting step, do not delete anything and return directlty.
             o.  Always remove all the G*_* files (from split event generation).
             o.  Remove the G* (or born_G* or all_G*) only when NOT doing only_generation or reweight_only."""
        if options['reweightonly']:
            return
        if not options['only_generation']:
            self.update_status('Cleaning previous results', level=None)
        for dir in p_dirs:
            #find old folders to be removed
            for obj in folder_name:
                # list all the G* (or all_G* or born_G*) directories
                to_rm = [file for file in \
                             os.listdir(pjoin(self.me_dir, 'SubProcesses', dir)) \
                             if file.startswith(obj[:-1]) and \
                            (os.path.isdir(pjoin(self.me_dir, 'SubProcesses', dir, file)) or \
                             os.path.exists(pjoin(self.me_dir, 'SubProcesses', dir, file)))] 
                # list all the G*_* directories (from split event generation)
                to_always_rm = [file for file in \
                             os.listdir(pjoin(self.me_dir, 'SubProcesses', dir)) \
                             if file.startswith(obj[:-1]) and
                             '_' in file and not '_G' in file and \
                            (os.path.isdir(pjoin(self.me_dir, 'SubProcesses', dir, file)) or \
                             os.path.exists(pjoin(self.me_dir, 'SubProcesses', dir, file)))]

                if not options['only_generation']:
                    to_always_rm.extend(to_rm)
                    if os.path.exists(pjoin(self.me_dir, 'SubProcesses', dir,'MadLoop5_resources.tar.gz')):
                        to_always_rm.append(pjoin(self.me_dir, 'SubProcesses', dir,'MadLoop5_resources.tar.gz'))
                files.rm([pjoin(self.me_dir, 'SubProcesses', dir, d) for d in to_always_rm])
        return


    def print_summary(self, options, step, mode, scale_pdf_info=[], done=True):
        """print a summary of the results contained in self.cross_sect_dict.
        step corresponds to the mintMC step, if =2 (i.e. after event generation)
        some additional infos are printed"""
        # find process name
        proc_card_lines = open(pjoin(self.me_dir, 'Cards', 'proc_card_mg5.dat')).read().split('\n')
        process = ''
        for line in proc_card_lines:
            if line.startswith('generate') or line.startswith('add process'):
                process = process+(line.replace('generate ', '')).replace('add process ','')+' ; '
        lpp = {0:'l', 1:'p', -1:'pbar', 2:'elastic photon from p', 3:'e-', 4:'mu-', -3:'e+', -4:'mu+'}
        if self.ninitial == 1:
            proc_info = '\n      Process %s' % process[:-3]
        else:
            proc_info = '\n      Process %s\n      Run at %s-%s collider (%s + %s GeV)' % \
                (process[:-3], lpp[self.run_card['lpp1']], lpp[self.run_card['lpp2']], 
                 self.run_card['ebeam1'], self.run_card['ebeam2'])

        if self.ninitial == 1:
            self.cross_sect_dict['unit']='GeV'
            self.cross_sect_dict['xsec_string']='(Partial) decay width'
            self.cross_sect_dict['axsec_string']='(Partial) abs(decay width)'
        else:
            self.cross_sect_dict['unit']='pb'
            self.cross_sect_dict['xsec_string']='Total cross section'
            self.cross_sect_dict['axsec_string']='Total abs(cross section)'
        if self.run_card['event_norm'].lower()=='bias':
            self.cross_sect_dict['xsec_string']+=', incl. bias (DO NOT USE)'

        if mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO']:
            status = ['Determining the number of unweighted events per channel',
                      'Updating the number of unweighted events per channel',
                      'Summary:']
            computed='(computed from LHE events)'
        elif mode in ['NLO', 'LO']:
            status = ['Results after grid setup:','Current results:',
                      'Final results and run summary:']
            computed='(computed from histogram information)'

        if step != 2 and mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO']:
            message = status[step] + '\n\n      Intermediate results:' + \
                      ('\n      Random seed: %(randinit)d' + \
                       '\n      %(xsec_string)s:      %(xsect)8.3e +- %(errt)6.1e %(unit)s' + \
                       '\n      %(axsec_string)s: %(xseca)8.3e +- %(erra)6.1e %(unit)s \n') \
                      % self.cross_sect_dict
        elif mode in ['NLO','LO'] and not done:
            if step == 0:
                message = '\n      ' + status[0] + \
                          '\n      %(xsec_string)s:      %(xsect)8.3e +- %(errt)6.1e %(unit)s' % \
                          self.cross_sect_dict
            else:
                message = '\n      ' + status[1] + \
                          '\n      %(xsec_string)s:      %(xsect)8.3e +- %(errt)6.1e %(unit)s' % \
                          self.cross_sect_dict
                
        else:
            message = '\n   --------------------------------------------------------------'
            message = message + \
                      '\n      ' + status[2] + proc_info 
            if mode not in ['LO', 'NLO']:
                message = message + \
                      '\n      Number of events generated: %s' % self.run_card['nevents'] 
            message = message + \
                      '\n      %(xsec_string)s: %(xsect)8.3e +- %(errt)6.1e %(unit)s' % \
                      self.cross_sect_dict
            message = message + \
                      '\n   --------------------------------------------------------------'
            if scale_pdf_info and (self.run_card['nevents']>=10000 or mode in ['NLO', 'LO']):
                if scale_pdf_info[0]:
                        # scale uncertainties
                    message = message + '\n      Scale variation %s:' % computed
                    for s in scale_pdf_info[0]:
                        if s['unc']:
                            if self.run_card['ickkw'] != -1:
                                message = message + \
                                          ('\n          Dynamical_scale_choice %(label)i (envelope of %(size)s values): '\
                                           '\n              %(cen)8.3e pb  +%(max)0.1f%% -%(min)0.1f%%') % s
                            else:
                                message = message + \
                                          ('\n          Soft and hard scale dependence (added in quadrature): '\
                                           '\n              %(cen)8.3e pb  +%(max_q)0.1f%% -%(min_q)0.1f%%') % s
                                    
                        else:
                            message = message + \
                                          ('\n          Dynamical_scale_choice %(label)i: '\
                                           '\n              %(cen)8.3e pb') % s
                                
                if scale_pdf_info[1]:
                    message = message + '\n      PDF variation %s:' % computed
                    for p in scale_pdf_info[1]:
                        if p['unc']=='none':
                            message = message + \
                                          ('\n          %(name)s (central value only): '\
                                           '\n              %(cen)8.3e pb') % p
                            
                        elif p['unc']=='unknown':
                            message = message + \
                                          ('\n          %(name)s (%(size)s members; combination method unknown): '\
                                           '\n              %(cen)8.3e pb') % p
                        else:
                            message = message + \
                                          ('\n          %(name)s (%(size)s members; using %(unc)s method): '\
                                           '\n              %(cen)8.3e pb  +%(max)0.1f%% -%(min)0.1f%%') % p
                        # pdf uncertainties
                message = message + \
                          '\n   --------------------------------------------------------------'

        
        if (mode in ['NLO', 'LO'] and not done) or \
           (mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO'] and step!=2):
            logger.info(message+'\n')
            return

        # Some advanced general statistics are shown in the debug message at the
        # end of the run
        # Make sure it never stops a run
        # Gather some basic statistics for the run and extracted from the log files.
        if mode in ['aMC@NLO', 'aMC@LO', 'noshower', 'noshowerLO']: 
            log_GV_files =  misc.glob(pjoin('P*','G*','log_MINT*.txt'), 
                                      pjoin(self.me_dir, 'SubProcesses'))
            all_log_files = log_GV_files
        elif mode == 'NLO':
            log_GV_files = misc.glob(pjoin('P*','all_G*','log_MINT*.txt'), 
                                      pjoin(self.me_dir, 'SubProcesses')) 
            all_log_files = log_GV_files

        elif mode == 'LO':
            log_GV_files = ''
            all_log_files = misc.glob(pjoin('P*','born_G*','log_MINT*.txt'), 
                                      pjoin(self.me_dir, 'SubProcesses')) 
        else:
            raise aMCatNLOError( 'Run mode %s not supported.'%mode)

        try:
            message, debug_msg = \
               self.compile_advanced_stats(log_GV_files, all_log_files, message)
        except Exception as e:
            debug_msg = 'Advanced statistics collection failed with error "%s"\n'%str(e)
            err_string = StringIO.StringIO()
            traceback.print_exc(limit=4, file=err_string)
            debug_msg += 'Please report this backtrace to a MG5_aMC developer:\n%s'\
                                                          %err_string.getvalue()

        logger.debug(debug_msg+'\n')
        logger.info(message+'\n')
        
        # Now copy relevant information in the Events/Run_<xxx> directory
        evt_path = pjoin(self.me_dir, 'Events', self.run_name)
        open(pjoin(evt_path, 'summary.txt'),'w').write(message+'\n')
        open(pjoin(evt_path, '.full_summary.txt'), 
                                       'w').write(message+'\n\n'+debug_msg+'\n')
                                       
        self.archive_files(evt_path,mode)

    def archive_files(self, evt_path, mode):
        """ Copies in the Events/Run_<xxx> directory relevant files characterizing
        the run."""

        files_to_arxiv = [pjoin('Cards','param_card.dat'),
                          pjoin('Cards','MadLoopParams.dat'),
                          pjoin('Cards','FKS_params.dat'),
                          pjoin('Cards','run_card.dat'),                          
                          pjoin('Subprocesses','setscales.f'),
                          pjoin('Subprocesses','cuts.f')]

        if mode in ['NLO', 'LO']:
            files_to_arxiv.append(pjoin('Cards','FO_analyse_card.dat'))

        if not os.path.exists(pjoin(evt_path,'RunMaterial')):
            os.mkdir(pjoin(evt_path,'RunMaterial'))

        for path in files_to_arxiv:
            if os.path.isfile(pjoin(self.me_dir,path)):
                files.cp(pjoin(self.me_dir,path),pjoin(evt_path,'RunMaterial'))
        misc.call(['tar','-czpf','RunMaterial.tar.gz','RunMaterial'],cwd=evt_path)
        shutil.rmtree(pjoin(evt_path,'RunMaterial'))

    def compile_advanced_stats(self,log_GV_files,all_log_files,message):
        """ This functions goes through the log files given in arguments and 
        compiles statistics about MadLoop stability, virtual integration 
        optimization and detection of potential error messages into a nice
        debug message to printed at the end of the run """
        
        def safe_float(str_float):
            try:
                return float(str_float)
            except ValueError:
                logger.debug('Could not convert the following float during'+
                             ' advanced statistics printout: %s'%str(str_float))
                return -1.0
        
        
        # > UPS is a dictionary of tuples with this format {channel:[nPS,nUPS]}
        # > Errors is a list of tuples with this format (log_file,nErrors)
        stats = {'UPS':{}, 'Errors':[], 'virt_stats':{}, 'timings':{}}
        mint_search = re.compile(r"MINT(?P<ID>\d*).txt")

        # ==================================     
        # == MadLoop stability statistics ==
        # ==================================
    
        # Recuperate the fraction of unstable PS points found in the runs for
        # the virtuals
        UPS_stat_finder = re.compile(
             r"Satistics from MadLoop:.*"+\
             r"Total points tried\:\s+(?P<ntot>\d+).*"+\
             r"Stability unknown\:\s+(?P<nsun>\d+).*"+\
             r"Stable PS point\:\s+(?P<nsps>\d+).*"+\
             r"Unstable PS point \(and rescued\)\:\s+(?P<nups>\d+).*"+\
             r"Exceptional PS point \(unstable and not rescued\)\:\s+(?P<neps>\d+).*"+\
             r"Double precision used\:\s+(?P<nddp>\d+).*"+\
             r"Quadruple precision used\:\s+(?P<nqdp>\d+).*"+\
             r"Initialization phase\-space points\:\s+(?P<nini>\d+).*"+\
             r"Unknown return code \(100\)\:\s+(?P<n100>\d+).*"+\
             r"Unknown return code \(10\)\:\s+(?P<n10>\d+).*",re.DOTALL)
    
        unit_code_meaning = { 0 : 'Not identified (CTModeRun != -1)',
                              1 : 'CutTools (double precision)',
                              2 : 'PJFry++',
                              3 : 'IREGI',
                              4 : 'Golem95',
                              5 : 'Samurai',
                              6 : 'Ninja (double precision)',
                              7 : 'COLLIER',
                              8 : 'Ninja (quadruple precision)',
                              9 : 'CutTools (quadruple precision)'}
        RetUnit_finder =re.compile(
                           r"#Unit\s*(?P<unit>\d+)\s*=\s*(?P<n_occurences>\d+)")
    #Unit
    
        for gv_log in log_GV_files:
            channel_name = '/'.join(gv_log.split('/')[-5:-1])
            log=open(gv_log,'r').read()                
            UPS_stats = re.search(UPS_stat_finder,log)
            for retunit_stats in re.finditer(RetUnit_finder, log):
                if channel_name not in list(stats['UPS'].keys()):
                    stats['UPS'][channel_name] = [0]*10+[[0]*10]
                stats['UPS'][channel_name][10][int(retunit_stats.group('unit'))] \
                                     += int(retunit_stats.group('n_occurences'))
            if not UPS_stats is None:
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
                except KeyError:
                    stats['UPS'][channel_name] = [int(UPS_stats.group('ntot')),
                      int(UPS_stats.group('nsun')),int(UPS_stats.group('nsps')),
                      int(UPS_stats.group('nups')),int(UPS_stats.group('neps')),
                      int(UPS_stats.group('nddp')),int(UPS_stats.group('nqdp')),
                      int(UPS_stats.group('nini')),int(UPS_stats.group('n100')),
                      int(UPS_stats.group('n10')),[0]*10]
        debug_msg = ""
        if len(list(stats['UPS'].keys()))>0:
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
            nTot1  = [sum([chan[10][i] for chan in stats['UPS'].values()],0) \
                                                             for i in range(10)]
            UPSfracs = [(chan[0] , 0.0 if chan[1][0]==0 else \
              safe_float(chan[1][4]*100)/chan[1][0]) for chan in stats['UPS'].items()]
            maxUPS = max(UPSfracs, key = lambda w: w[1])

            tmpStr = ""
            tmpStr += '\n  Number of loop ME evaluations (by MadLoop): %d'%nTotPS
            tmpStr += '\n    Stability unknown:                   %d'%nTotsun
            tmpStr += '\n    Stable PS point:                     %d'%nTotsps
            tmpStr += '\n    Unstable PS point (and rescued):     %d'%nTotups
            tmpStr += '\n    Unstable PS point (and not rescued): %d'%nToteps
            tmpStr += '\n    Only double precision used:          %d'%nTotddp
            tmpStr += '\n    Quadruple precision used:            %d'%nTotqdp
            tmpStr += '\n    Initialization phase-space points:   %d'%nTotini
            tmpStr += '\n    Reduction methods used:'
            red_methods = [(unit_code_meaning[i],nTot1[i]) for i in \
                                         unit_code_meaning.keys() if nTot1[i]>0]
            for method, n in sorted(red_methods, key= lambda l: l[1], reverse=True):
                tmpStr += '\n      > %s%s%s'%(method,' '*(33-len(method)),n)                
            if nTot100 != 0:
                debug_msg += '\n  Unknown return code (100):             %d'%nTot100
            if nTot10 != 0:
                debug_msg += '\n  Unknown return code (10):              %d'%nTot10
            nUnknownUnit = sum(nTot1[u] for u in range(10) if u \
                                                not in list(unit_code_meaning.keys()))
            if nUnknownUnit != 0:
                debug_msg += '\n  Unknown return code (1):               %d'\
                                                                   %nUnknownUnit

            if maxUPS[1]>0.001:
                message += tmpStr
                message += '\n  Total number of unstable PS point detected:'+\
                        ' %d (%4.2f%%)'%(nToteps,safe_float(100*nToteps)/nTotPS)
                message += '\n    Maximum fraction of UPS points in '+\
                          'channel %s (%4.2f%%)'%maxUPS
                message += '\n    Please report this to the authors while '+\
                                                                'providing the file'
                message += '\n    %s'%str(pjoin(os.path.dirname(self.me_dir),
                                                               maxUPS[0],'UPS.log'))
            else:
                debug_msg += tmpStr

    
        # ====================================================
        # == aMC@NLO virtual integration optimization stats ==
        # ====================================================
    
        virt_tricks_finder = re.compile(
          r"accumulated results Virtual ratio\s*=\s*-?(?P<v_ratio>[\d\+-Eed\.]*)"+\
            r"\s*\+/-\s*-?[\d\+-Eed\.]*\s*\(\s*-?(?P<v_ratio_err>[\d\+-Eed\.]*)\s*\%\)\s*\n"+\
          r"accumulated results ABS virtual\s*=\s*-?(?P<v_abs_contr>[\d\+-Eed\.]*)"+\
            r"\s*\+/-\s*-?[\d\+-Eed\.]*\s*\(\s*-?(?P<v_abs_contr_err>[\d\+-Eed\.]*)\s*\%\)")
    
        virt_frac_finder = re.compile(r"update virtual fraction to\s*:\s*"+\
                     r"-?(?P<v_frac>[\d\+-Eed\.]*)\s*")
        
        channel_contr_finder = re.compile(r"Final result \[ABS\]\s*:\s*-?(?P<v_contr>[\d\+-Eed\.]*)")
        
        channel_contr_list = {}
        for gv_log in log_GV_files:
            logfile=open(gv_log,'r')
            log = logfile.read()
            logfile.close()
            channel_name = '/'.join(gv_log.split('/')[-3:-1])
            vf_stats = None
            for vf_stats in re.finditer(virt_frac_finder, log):
                pass
            if not vf_stats is None:
                v_frac = safe_float(vf_stats.group('v_frac'))
                ###v_average = safe_float(vf_stats.group('v_average'))
                try:
                    if v_frac < stats['virt_stats']['v_frac_min'][0]:
                        stats['virt_stats']['v_frac_min']=(v_frac,channel_name)
                    if v_frac > stats['virt_stats']['v_frac_max'][0]:
                        stats['virt_stats']['v_frac_max']=(v_frac,channel_name)
                    stats['virt_stats']['v_frac_avg'][0] += v_frac
                    stats['virt_stats']['v_frac_avg'][1] += 1
                except KeyError:
                    stats['virt_stats']['v_frac_min']=[v_frac,channel_name]
                    stats['virt_stats']['v_frac_max']=[v_frac,channel_name]
                    stats['virt_stats']['v_frac_avg']=[v_frac,1]


            ccontr_stats = None
            for ccontr_stats in re.finditer(channel_contr_finder, log):
                pass
            if not ccontr_stats is None:
                contrib = safe_float(ccontr_stats.group('v_contr'))
                try:
                    if contrib>channel_contr_list[channel_name]:
                        channel_contr_list[channel_name]=contrib
                except KeyError:
                    channel_contr_list[channel_name]=contrib
                
                
        # Now build the list of relevant virt log files to look for the maxima
        # of virt fractions and such.
        average_contrib = 0.0
        for value in channel_contr_list.values():
            average_contrib += value
        if len(list(channel_contr_list.values())) !=0:
            average_contrib = average_contrib / len(list(channel_contr_list.values()))
        
        relevant_log_GV_files = []
        excluded_channels = set([])
        all_channels = set([])
        for log_file in log_GV_files:
            channel_name = '/'.join(log_file.split('/')[-3:-1])
            all_channels.add(channel_name)
            try:
                if channel_contr_list[channel_name] > (0.1*average_contrib):
                    relevant_log_GV_files.append(log_file)
                else:
                    excluded_channels.add(channel_name)
            except KeyError:
                    relevant_log_GV_files.append(log_file)
        
        # Now we want to use the latest occurence of accumulated result in the log file
        for gv_log in relevant_log_GV_files:
            logfile=open(gv_log,'r')
            log = logfile.read()
            logfile.close()
            channel_name = '/'.join(gv_log.split('/')[-3:-1])
            
            vt_stats = None
            for vt_stats in re.finditer(virt_tricks_finder, log):
                pass
            if not vt_stats is None:
                vt_stats_group = vt_stats.groupdict()
                v_ratio = safe_float(vt_stats.group('v_ratio'))
                v_ratio_err = safe_float(vt_stats.group('v_ratio_err'))
                v_contr = safe_float(vt_stats.group('v_abs_contr'))
                v_contr_err = safe_float(vt_stats.group('v_abs_contr_err'))
                try:
                    if v_ratio < stats['virt_stats']['v_ratio_min'][0]:
                        stats['virt_stats']['v_ratio_min']=(v_ratio,channel_name)
                    if v_ratio > stats['virt_stats']['v_ratio_max'][0]:
                        stats['virt_stats']['v_ratio_max']=(v_ratio,channel_name)
                    if v_ratio < stats['virt_stats']['v_ratio_err_min'][0]:
                        stats['virt_stats']['v_ratio_err_min']=(v_ratio_err,channel_name)
                    if v_ratio > stats['virt_stats']['v_ratio_err_max'][0]:
                        stats['virt_stats']['v_ratio_err_max']=(v_ratio_err,channel_name)
                    if v_contr < stats['virt_stats']['v_contr_min'][0]:
                        stats['virt_stats']['v_contr_min']=(v_contr,channel_name)
                    if v_contr > stats['virt_stats']['v_contr_max'][0]:
                        stats['virt_stats']['v_contr_max']=(v_contr,channel_name)
                    if v_contr_err < stats['virt_stats']['v_contr_err_min'][0]:
                        stats['virt_stats']['v_contr_err_min']=(v_contr_err,channel_name)
                    if v_contr_err > stats['virt_stats']['v_contr_err_max'][0]:
                        stats['virt_stats']['v_contr_err_max']=(v_contr_err,channel_name)
                except KeyError:
                    stats['virt_stats']['v_ratio_min']=[v_ratio,channel_name]
                    stats['virt_stats']['v_ratio_max']=[v_ratio,channel_name]
                    stats['virt_stats']['v_ratio_err_min']=[v_ratio_err,channel_name]
                    stats['virt_stats']['v_ratio_err_max']=[v_ratio_err,channel_name]
                    stats['virt_stats']['v_contr_min']=[v_contr,channel_name]
                    stats['virt_stats']['v_contr_max']=[v_contr,channel_name]
                    stats['virt_stats']['v_contr_err_min']=[v_contr_err,channel_name]
                    stats['virt_stats']['v_contr_err_max']=[v_contr_err,channel_name]
        
            vf_stats = None
            for vf_stats in re.finditer(virt_frac_finder, log):
                pass
            ##if not vf_stats is None:
            ##    v_frac = safe_float(vf_stats.group('v_frac'))
            ##    v_average = safe_float(vf_stats.group('v_average'))
            ##    try:
            ##        if v_average < stats['virt_stats']['v_average_min'][0]:
            ##            stats['virt_stats']['v_average_min']=(v_average,channel_name)
            ##        if v_average > stats['virt_stats']['v_average_max'][0]:
            ##            stats['virt_stats']['v_average_max']=(v_average,channel_name)
            ##        stats['virt_stats']['v_average_avg'][0] += v_average
            ##        stats['virt_stats']['v_average_avg'][1] += 1
            ##    except KeyError:
            ##        stats['virt_stats']['v_average_min']=[v_average,channel_name]
            ##        stats['virt_stats']['v_average_max']=[v_average,channel_name]
            ##        stats['virt_stats']['v_average_avg']=[v_average,1]
        
        try:
            debug_msg += '\n\n  Statistics on virtual integration optimization : '
            
            debug_msg += '\n    Maximum virt fraction computed         %.3f (%s)'\
                                       %tuple(stats['virt_stats']['v_frac_max'])
            debug_msg += '\n    Minimum virt fraction computed         %.3f (%s)'\
                                       %tuple(stats['virt_stats']['v_frac_min'])
            debug_msg += '\n    Average virt fraction computed         %.3f'\
              %safe_float(stats['virt_stats']['v_frac_avg'][0]/safe_float(stats['virt_stats']['v_frac_avg'][1]))
            debug_msg += '\n  Stats below exclude negligible channels (%d excluded out of %d)'%\
                 (len(excluded_channels),len(all_channels))
            ##debug_msg += '\n    Maximum virt ratio used                %.2f (%s)'\
            ##                        %tuple(stats['virt_stats']['v_average_max'])          
            debug_msg += '\n    Maximum virt ratio found from grids    %.2f (%s)'\
                                     %tuple(stats['virt_stats']['v_ratio_max'])
            tmpStr = '\n    Max. MC err. on virt ratio from grids  %.1f %% (%s)'\
                                  %tuple(stats['virt_stats']['v_ratio_err_max'])
            debug_msg += tmpStr
            # After all it was decided that it is better not to alarm the user unecessarily
            # with such printout of the statistics.
#            if stats['virt_stats']['v_ratio_err_max'][0]>100.0 or \
#                                stats['virt_stats']['v_ratio_err_max'][0]>100.0:
#                message += "\n  Suspiciously large MC error in :"
#            if stats['virt_stats']['v_ratio_err_max'][0]>100.0:
#                message += tmpStr

            tmpStr = '\n    Maximum MC error on abs virt           %.1f %% (%s)'\
                                  %tuple(stats['virt_stats']['v_contr_err_max'])
            debug_msg += tmpStr
#            if stats['virt_stats']['v_contr_err_max'][0]>100.0:
#                message += tmpStr
            

        except KeyError:
            debug_msg += '\n  Could not find statistics on the integration optimization. '
    
        # =======================================
        # == aMC@NLO timing profile statistics ==
        # =======================================
    
        timing_stat_finder = re.compile(r"\s*Time spent in\s*(?P<name>\w*)\s*:\s*"+\
                     r"(?P<time>[\d\+-Eed\.]*)\s*")

        for logf in log_GV_files:
            logfile=open(logf,'r')
            log = logfile.read()
            logfile.close()
            channel_name = '/'.join(logf.split('/')[-3:-1])
            mint = re.search(mint_search,logf)
            if not mint is None:
               channel_name =   channel_name+' [step %s]'%mint.group('ID')

            for time_stats in re.finditer(timing_stat_finder, log):
                try:
                    stats['timings'][time_stats.group('name')][channel_name]+=\
                                                 safe_float(time_stats.group('time'))
                except KeyError:
                    if time_stats.group('name') not in list(stats['timings'].keys()):
                        stats['timings'][time_stats.group('name')] = {}
                    stats['timings'][time_stats.group('name')][channel_name]=\
                                                 safe_float(time_stats.group('time'))
        
        # useful inline function
        Tstr = lambda secs: str(datetime.timedelta(seconds=int(secs)))
        try:
            totTimeList = [(time, chan) for chan, time in \
                                              stats['timings']['Total'].items()]
        except KeyError:
            totTimeList = []

        totTimeList.sort()
        if len(totTimeList)>0:
            debug_msg += '\n\n  Inclusive timing profile :'
            debug_msg += '\n    Overall slowest channel          %s (%s)'%\
                                     (Tstr(totTimeList[-1][0]),totTimeList[-1][1])
            debug_msg += '\n    Average channel running time     %s'%\
                       Tstr(sum([el[0] for el in totTimeList])/len(totTimeList))
            debug_msg += '\n    Aggregated total running time    %s'%\
                                        Tstr(sum([el[0] for el in totTimeList]))       
        else:            
            debug_msg += '\n\n  Inclusive timing profile non available.'
        
        sorted_keys = sorted(list(stats['timings'].keys()), key= lambda stat: \
                              sum(stats['timings'][stat].values()), reverse=True)
        for name in sorted_keys:
            if name=='Total':
                continue
            if sum(stats['timings'][name].values())<=0.0:
                debug_msg += '\n  Zero time record for %s.'%name
                continue
            try:
                TimeList = [((100.0*time/stats['timings']['Total'][chan]), 
                     chan) for chan, time in stats['timings'][name].items()]
            except KeyError as ZeroDivisionError:
                debug_msg += '\n\n  Timing profile for %s unavailable.'%name
                continue
            TimeList.sort()
            debug_msg += '\n  Timing profile for <%s> :'%name
            try:
                debug_msg += '\n    Overall fraction of time         %.3f %%'%\
                       safe_float((100.0*(sum(stats['timings'][name].values())/
                                      sum(stats['timings']['Total'].values()))))
            except KeyError as ZeroDivisionError:
                debug_msg += '\n    Overall fraction of time unavailable.'
            debug_msg += '\n    Largest fraction of time         %.3f %% (%s)'%\
                                             (TimeList[-1][0],TimeList[-1][1])
            debug_msg += '\n    Smallest fraction of time        %.3f %% (%s)'%\
                                             (TimeList[0][0],TimeList[0][1])

        # =============================     
        # == log file eror detection ==
        # =============================
        
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
                           
        return message, debug_msg


    def reweight_and_collect_events(self, options, mode, nevents, event_norm):
        """this function calls the reweighting routines and creates the event file in the 
        Event dir. Return the name of the event file created
        """
        scale_pdf_info=[]
        if any(self.run_card['reweight_scale']) or any(self.run_card['reweight_PDF']) or \
           len(self.run_card['dynamical_scale_choice']) > 1 or len(self.run_card['lhaid']) > 1\
           or self.run_card['store_rwgt_info']:
            scale_pdf_info = self.run_reweight(options['reweightonly'])
        self.update_status('Collecting events', level='parton', update_results=True)
        misc.compile(['collect_events'], 
                    cwd=pjoin(self.me_dir, 'SubProcesses'), nocompile=options['nocompile'])
        p = misc.Popen(['./collect_events'], cwd=pjoin(self.me_dir, 'SubProcesses'),
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        if event_norm.lower() == 'sum':
            out, err = p.communicate(input = '1\n'.encode())
        elif event_norm.lower() == 'unity':
            out, err = p.communicate(input = '3\n'.encode())
        elif event_norm.lower() == 'bias':
            out, err = p.communicate(input = '0\n'.encode())
        else:
            out, err = p.communicate(input = '2\n'.encode())
        
        out = out.decode(errors='ignore')
        data = str(out)
        #get filename from collect events
        filename = data.split()[-1].strip().replace('\\n','').replace('"','').replace("'",'')
        
        if not os.path.exists(pjoin(self.me_dir, 'SubProcesses', filename)):
            raise aMCatNLOError('An error occurred during event generation. ' + \
                    'The event file has not been created: \n %s' % data)
        evt_file = pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe.gz')
        misc.gzip(pjoin(self.me_dir, 'SubProcesses', filename), stdout=evt_file)
        if not options['reweightonly']:
            self.print_summary(options, 2, mode, scale_pdf_info)
            res_files = misc.glob('res*.txt', pjoin(self.me_dir, 'SubProcesses'))
            for res_file in res_files:
                files.mv(res_file,pjoin(self.me_dir, 'Events', self.run_name))

        logger.info('The %s file has been generated.\n' % (evt_file))
        self.results.add_detail('nb_event', nevents)
        self.update_status('Events generated', level='parton', update_results=True)
        return evt_file[:-3]


    def run_mcatnlo(self, evt_file, options):
        """runs mcatnlo on the generated event file, to produce showered-events
        """
        logger.info('Preparing MCatNLO run')
        try:     
            misc.gunzip(evt_file)
        except Exception:
            pass

        self.banner = banner_mod.Banner(evt_file)
        shower = self.banner.get_detail('run_card', 'parton_shower').upper()

        #check that the number of split event files divides the number of
        # events, otherwise set it to 1
        if int(self.banner.get_detail('run_card', 'nevents') / \
                self.shower_card['nsplit_jobs']) * self.shower_card['nsplit_jobs'] \
                != self.banner.get_detail('run_card', 'nevents'):
            logger.warning(\
                'nsplit_jobs in the shower card is not a divisor of the number of events.\n' + \
                'Setting it to 1.')
            self.shower_card['nsplit_jobs'] = 1

        # don't split jobs if the user asks to shower only a part of the events
        if self.shower_card['nevents'] > 0 and \
           self.shower_card['nevents'] < self.banner.get_detail('run_card', 'nevents') and \
           self.shower_card['nsplit_jobs'] != 1:
            logger.warning(\
                'Only a part of the events will be showered.\n' + \
                'Setting nsplit_jobs in the shower_card to 1.')
            self.shower_card['nsplit_jobs'] = 1

        self.banner_to_mcatnlo(evt_file)

        # if fastjet has to be linked (in extralibs) then
        # add lib /include dirs for fastjet if fastjet-config is present on the
        # system, otherwise add fjcore to the files to combine
        if 'fastjet' in self.shower_card['extralibs']:
            #first, check that stdc++ is also linked
            if not 'stdc++' in self.shower_card['extralibs']:
                logger.warning('Linking FastJet: adding stdc++ to EXTRALIBS')
                self.shower_card['extralibs'] += ' stdc++'
            # then check if options[fastjet] corresponds to a valid fj installation
            try:
                #this is for a complete fj installation
                p = subprocess.Popen([self.options['fastjet'], '--prefix'], \
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = p.communicate()
                #remove the line break from output (last character)
                output = output.decode(errors='ignore')[:-1]
                # add lib/include paths
                if not pjoin(output, 'lib') in self.shower_card['extrapaths']:
                    logger.warning('Linking FastJet: updating EXTRAPATHS')
                    self.shower_card['extrapaths'] += ' ' + pjoin(output, 'lib')
                if not pjoin(output, 'include') in self.shower_card['includepaths']:
                    logger.warning('Linking FastJet: updating INCLUDEPATHS')
                    self.shower_card['includepaths'] += ' ' + pjoin(output, 'include')
                # to be changed in the fortran wrapper
                include_line = '#include "fastjet/ClusterSequence.hh"//INCLUDE_FJ' 
                namespace_line = 'namespace fj = fastjet;//NAMESPACE_FJ'
            except Exception:
                logger.warning('Linking FastJet: using fjcore')
                # this is for FJcore, so no FJ library has to be linked
                self.shower_card['extralibs'] = self.shower_card['extralibs'].replace('fastjet', '')
                if not 'fjcore.o' in self.shower_card['analyse']:
                    self.shower_card['analyse'] += ' fjcore.o'
                # to be changed in the fortran wrapper
                include_line = '#include "fjcore.hh"//INCLUDE_FJ' 
                namespace_line = 'namespace fj = fjcore;//NAMESPACE_FJ'
            # change the fortran wrapper with the correct namespaces/include
            fjwrapper_lines = open(pjoin(self.me_dir, 'MCatNLO', 'srcCommon', 'myfastjetfortran.cc')).read().split('\n')
            for line in fjwrapper_lines:
                if '//INCLUDE_FJ' in line:
                    fjwrapper_lines[fjwrapper_lines.index(line)] = include_line
                if '//NAMESPACE_FJ' in line:
                    fjwrapper_lines[fjwrapper_lines.index(line)] = namespace_line
            with open(pjoin(self.me_dir, 'MCatNLO', 'srcCommon', 'myfastjetfortran.cc'), 'w') as fsock:
                fsock.write('\n'.join(fjwrapper_lines) + '\n')

        extrapaths = self.shower_card['extrapaths'].split()

        # check that the path needed by HW++ and PY8 are set if one uses these shower
        if shower in ['HERWIGPP', 'PYTHIA8']:
            path_dict = {'HERWIGPP': ['hepmc_path',
                                      'thepeg_path',
                                      'hwpp_path'],
                         'PYTHIA8': ['pythia8_path']}

            if not all([self.options[ppath] and os.path.exists(self.options[ppath]) for ppath in path_dict[shower]]):
                raise aMCatNLOError('Some paths are missing or invalid in the configuration file.\n' + \
                        ('Please make sure you have set these variables: %s' % ', '.join(path_dict[shower])))

        if shower == 'HERWIGPP':
            extrapaths.append(pjoin(self.options['hepmc_path'], 'lib'))
            self.shower_card['extrapaths'] += ' %s' % pjoin(self.options['hepmc_path'], 'lib')

        # add the HEPMC path of the pythia8 installation
        if shower == 'PYTHIA8':
            hepmc = subprocess.Popen([pjoin(self.options['pythia8_path'], 'bin', 'pythia8-config'), '--hepmc2'],
                         stdout = subprocess.PIPE).stdout.read().decode(errors='ignore').strip()
            #this gives all the flags, i.e.
            #-I/Path/to/HepMC/include -L/Path/to/HepMC/lib -lHepMC
            # we just need the path to the HepMC libraries
            extrapaths.append(hepmc.split()[1].replace('-L', '')) 

        # check that if FxFx is activated the correct shower plugin is present
        if shower == 'PYTHIA8' and self.run_card['ickkw'] == 3:
            misc.sprint(self.options['pythia8_path'])
            f1 = open(pjoin(self.me_dir, "MCatNLO","Scripts", "JetMatching.h"), "r").read()
            py8_plugin = pjoin(self.options['pythia8_path'], "include","Pythia8Plugins","JetMatching.h")
            if not os.path.exists(py8_plugin):
                raise Exception("FxFx requires a dedicated plugin to be installed in Pythia8. See http://cern.ch/amcatnlo/FxFx_merging.htm")
            f2 = open(py8_plugin).read()
            if f1 != f2:
                raise Exception("FxFx requires a dedicated plugin to be installed in Pythia8. Incorrect plugin detected. See http://cern.ch/amcatnlo/FxFx_merging.htm")

        if shower == 'PYTHIA8' and not os.path.exists(pjoin(self.options['pythia8_path'], 'xmldoc')):
            extrapaths.append(pjoin(self.options['pythia8_path'], 'lib'))

        # HwU.f now contains some calls to the PineAPPL functions. One should add the 
        #  dummy interface to have them linked
        if "HwU.o" in self.shower_card['analyse'] and \
          not "pineappl_interface_dummy.o" in self.shower_card['analyse']: 
            self.shower_card['analyse'] += " pineappl_interface_dummy.o"

        # set the PATH for the dynamic libraries
        if sys.platform == 'darwin':
            ld_library_path = 'DYLD_LIBRARY_PATH'
        else:
            ld_library_path = 'LD_LIBRARY_PATH'
        if ld_library_path in list(os.environ.keys()):
            paths = os.environ[ld_library_path]
        else:
            paths = ''
        paths += ':' + ':'.join(extrapaths)
        os.putenv(ld_library_path, paths)

        shower_card_path = pjoin(self.me_dir, 'MCatNLO', 'shower_card.dat')
        self.shower_card.write_card(shower, shower_card_path)

        # overwrite if shower_card_set.dat exists in MCatNLO
        if os.path.exists(pjoin(self.me_dir, 'MCatNLO', 'shower_card_set.dat')):
            files.mv(pjoin(self.me_dir, 'MCatNLO', 'shower_card_set.dat'),
                     pjoin(self.me_dir, 'MCatNLO', 'shower_card.dat'))
        
        mcatnlo_log = pjoin(self.me_dir, 'mcatnlo.log')
        self.update_status('Compiling MCatNLO for %s...' % shower, level='shower') 

        
        # libdl may be needded for pythia 82xx
        #if shower == 'PYTHIA8' and not \
        #   os.path.exists(pjoin(self.options['pythia8_path'], 'xmldoc')) and \
        #   'dl' not in self.shower_card['extralibs'].split():
        #    # 'dl' has to be linked with the extralibs
        #    self.shower_card['extralibs'] += ' dl'
        #    logger.warning("'dl' was added to extralibs from the shower_card.dat.\n" + \
        #                  "It is needed for the correct running of PY8.2xx.\n" + \
        #                  "If this library cannot be found on your system, a crash will occur.")

        misc.call(['./MCatNLO_MadFKS.inputs'], stdout=open(mcatnlo_log, 'w'),
                    stderr=open(mcatnlo_log, 'w'), 
                    cwd=pjoin(self.me_dir, 'MCatNLO'),
                    close_fds=True)

        exe = 'MCATNLO_%s_EXE' % shower
        if not os.path.exists(pjoin(self.me_dir, 'MCatNLO', exe)) and \
            not os.path.exists(pjoin(self.me_dir, 'MCatNLO', 'Pythia8.exe')):
            print(open(mcatnlo_log).read())
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

        #look for the event files (don't resplit if one asks for the 
        # same number of event files as in the previous run)
        event_files = misc.glob('events.lhe_*', pjoin(self.me_dir, 'Events', self.run_name))
        if max(len(event_files), 1) != self.shower_card['nsplit_jobs']:
            logger.info('Cleaning old files and splitting the event file...')
            #clean the old files
            files.rm([f for f in event_files if 'events.lhe' not in f])
            if self.shower_card['nsplit_jobs'] > 1:
                eventfile = lhe_parser.EventFile(evt_file)
                eventfile.split(self.banner.get_detail('run_card', 'nevents') / self.shower_card['nsplit_jobs'], cwd = pjoin(self.me_dir, 'Events', self.run_name))
                logger.info('Splitting done.')
            event_files = misc.glob('events.lhe_*', pjoin(self.me_dir, 'Events', self.run_name)) 

        event_files.sort()

        self.update_status('Showering events...', level='shower')
        logger.info('(Running in %s)' % rundir)
        if shower != 'PYTHIA8':
            files.mv(pjoin(self.me_dir, 'MCatNLO', exe), rundir)
            files.mv(pjoin(self.me_dir, 'MCatNLO', 'MCATNLO_%s_input' % shower), rundir)
        else:
        # special treatment for pythia8
            files.mv(pjoin(self.me_dir, 'MCatNLO', 'Pythia8.cmd'), rundir)
            files.mv(pjoin(self.me_dir, 'MCatNLO', 'Pythia8.exe'), rundir)
            if os.path.exists(pjoin(self.options['pythia8_path'], 'xmldoc')): # this is PY8.1xxx
                files.ln(pjoin(self.options['pythia8_path'], 'examples', 'config.sh'), rundir)
                files.ln(pjoin(self.options['pythia8_path'], 'xmldoc'), rundir)
            else: # this is PY8.2xxx
                files.ln(pjoin(self.options['pythia8_path'], 'share/Pythia8/xmldoc'), rundir)
        #link the hwpp exe in the rundir
        if shower == 'HERWIGPP':
            try:
                if os.path.exists(pjoin(self.options['hwpp_path'], 'bin', 'Herwig++')):
                    files.ln(pjoin(self.options['hwpp_path'], 'bin', 'Herwig++'), rundir)
                if os.path.exists(pjoin(self.options['hwpp_path'], 'bin', 'Herwig')):
                    files.ln(pjoin(self.options['hwpp_path'], 'bin', 'Herwig'), rundir)
            except Exception:
                raise aMCatNLOError('The Herwig++ path set in the configuration file is not valid.')

            if os.path.exists(pjoin(self.me_dir, 'MCatNLO', 'HWPPAnalyzer', 'HepMCFortran.so')):
                files.cp(pjoin(self.me_dir, 'MCatNLO', 'HWPPAnalyzer', 'HepMCFortran.so'), rundir)

        files.ln(evt_file, rundir, 'events.lhe')
        for i, f in enumerate(event_files):
            files.ln(f, rundir,'events_%d.lhe' % (i + 1))

        if not self.shower_card['analyse']:
            # an hep/hepmc file as output
            out_id = 'HEP'
        else:
            # one or more .top file(s) as output
            if "HwU" in self.shower_card['analyse']:
                out_id = 'HWU'
            else:
                out_id = 'TOP'

        # write the executable
        with open(pjoin(rundir, 'shower.sh'), 'w') as fsock:
            # set the PATH for the dynamic libraries
            if sys.platform == 'darwin':
                ld_library_path = 'DYLD_LIBRARY_PATH'
            else:
                ld_library_path = 'LD_LIBRARY_PATH'
            fsock.write(open(pjoin(self.me_dir, 'MCatNLO', 'shower_template.sh')).read() \
                % {'ld_library_path': ld_library_path,
                   'extralibs': ':'.join(extrapaths)})
        subprocess.call(['chmod', '+x', pjoin(rundir, 'shower.sh')])

        if event_files:
            arg_list = [[shower, out_id, self.run_name, '%d' % (i + 1)] \
                    for i in range(len(event_files))]
        else:
            arg_list = [[shower, out_id, self.run_name]]

        self.run_all({rundir: 'shower.sh'}, arg_list, 'shower')
        #self.njobs = 1
        #self.wait_for_complete('shower')

        # now collect the results
        message = ''
        warning = ''
        to_gzip = [evt_file]
        if out_id == 'HEP':
            #copy the showered stdhep/hepmc file back in events
            if shower in ['PYTHIA8', 'HERWIGPP']:
                hep_format = 'HEPMC'
                ext = 'hepmc'
            else:
                hep_format = 'StdHEP'
                ext = 'hep'

            hep_file = '%s_%s_0.%s.gz' % \
                    (pjoin(os.path.dirname(evt_file), 'events'), shower, ext)
            count = 0

            # find the first available name for the output:
            # check existing results with or without event splitting
            while os.path.exists(hep_file) or \
                  os.path.exists(hep_file.replace('.%s.gz' % ext, '__1.%s.gz' % ext)) :
                count +=1
                hep_file = '%s_%s_%d.%s.gz' % \
                    (pjoin(os.path.dirname(evt_file), 'events'), shower, count, ext)

            try:
                if self.shower_card['nsplit_jobs'] == 1:
                    files.mv(os.path.join(rundir, 'events.%s.gz' % ext), hep_file) 
                    message = ('The file %s has been generated. \nIt contains showered' + \
                     ' and hadronized events in the %s format obtained by' + \
                     ' showering the parton-level event file %s.gz with %s') % \
                     (hep_file, hep_format, evt_file, shower)
                else:
                    hep_list = []
                    for i in range(self.shower_card['nsplit_jobs']):
                        hep_list.append(hep_file.replace('.%s.gz' % ext, '__%d.%s.gz' % (i + 1, ext)))
                        files.mv(os.path.join(rundir, 'events_%d.%s.gz' % (i + 1, ext)), hep_list[-1]) 
                    message = ('The following files have been generated:\n  %s\nThey contain showered' + \
                     ' and hadronized events in the %s format obtained by' + \
                     ' showering the (split) parton-level event file %s.gz with %s') % \
                     ('\n  '.join(hep_list), hep_format, evt_file, shower)

            except OSError as IOError:
                raise aMCatNLOError('No file has been generated, an error occurred.'+\
             ' More information in %s' % pjoin(os.getcwd(), 'amcatnlo_run.log'))

            # run the plot creation in a secure way
            if hep_format == 'StdHEP':
                try:
                    self.do_plot('%s -f' % self.run_name)
                except Exception as error:
                    logger.info("Fail to make the plot. Continue...")
                    pass

        elif out_id == 'TOP' or out_id == 'HWU':
            #copy the topdrawer or HwU file(s) back in events
            if out_id=='TOP':
                ext='top'
            elif out_id=='HWU':
                ext='HwU'
            topfiles = []
            top_tars = [tarfile.TarFile(f) for f in misc.glob('histfile*.tar', rundir)]
            for top_tar in top_tars:
                topfiles.extend(top_tar.getnames())

            # safety check
            if len(top_tars) != self.shower_card['nsplit_jobs']:
                raise aMCatNLOError('%d job(s) expected, %d file(s) found' % \
                                     (self.shower_card['nsplit_jobs'], len(top_tars)))

            # find the first available name for the output:
            # check existing results with or without event splitting
            filename = 'plot_%s_%d_' % (shower, 1)
            count = 1
            while os.path.exists(pjoin(self.me_dir, 'Events', 
                      self.run_name, '%s0.%s' % (filename,ext))) or \
                  os.path.exists(pjoin(self.me_dir, 'Events', 
                      self.run_name, '%s0__1.%s' % (filename,ext))):
                count += 1
                filename = 'plot_%s_%d_' % (shower, count)

            if out_id=='TOP':
                hist_format='TopDrawer format'
            elif out_id=='HWU':
                hist_format='HwU and GnuPlot formats'

            if not topfiles:
                # if no topfiles are found just warn the user
                warning = 'No .top file has been generated. For the results of your ' +\
                               'run, please check inside %s' % rundir
            elif self.shower_card['nsplit_jobs'] == 1:
                # only one job for the shower
                top_tars[0].extractall(path = rundir) 
                plotfiles = [] 
                for i, file in enumerate(topfiles):
                    if out_id=='TOP':
                        plotfile = pjoin(self.me_dir, 'Events', self.run_name, 
                                         '%s%d.top' % (filename, i))
                        files.mv(pjoin(rundir, file), plotfile) 
                    elif out_id=='HWU':
                        out=pjoin(self.me_dir,'Events',
                                  self.run_name,'%s%d'% (filename,i))
                        histos=[{'dirname':pjoin(rundir,file)}]
                        self.combine_plots_HwU(histos,out)
                        try:
                            misc.call(['gnuplot','%s%d.gnuplot' % (filename,i)],\
                                      stdout=os.open(os.devnull, os.O_RDWR),\
                                      stderr=os.open(os.devnull, os.O_RDWR),\
                                      cwd=pjoin(self.me_dir, 'Events', self.run_name))
                        except Exception:
                            pass
                        plotfile=pjoin(self.me_dir,'Events',self.run_name,
                                                    '%s%d.HwU'% (filename,i))
                    plotfiles.append(plotfile)

                ffiles = 'files'
                have = 'have'
                if len(plotfiles) == 1:
                    ffiles = 'file'
                    have = 'has'

                message = ('The %s %s %s been generated, with histograms in the' + \
                        ' %s, obtained by showering the parton-level' + \
                        ' file %s.gz with %s.') % (ffiles, ', '.join(plotfiles), have, \
                        hist_format, evt_file, shower)
            else:
                # many jobs for the shower have been run
                topfiles_set = set(topfiles)
                plotfiles = [] 
                for j, top_tar in enumerate(top_tars):
                    top_tar.extractall(path = rundir) 
                    for i, file in enumerate(topfiles_set):
                        plotfile = pjoin(self.me_dir, 'Events', self.run_name, 
                                             '%s%d__%d.%s' % (filename, i, j + 1,ext))
                        files.mv(pjoin(rundir, file), plotfile) 
                        plotfiles.append(plotfile)

                # check if the user asked to combine the .top into a single file
                if self.shower_card['combine_td']:
                    misc.compile(['sum_plots'], cwd = pjoin(self.me_dir, 'Utilities'))

                    if self.banner.get('run_card', 'event_norm').lower() == 'sum':
                        norm = 1.
                    else:
                        norm = 1./float(self.shower_card['nsplit_jobs'])

                    plotfiles2 = []
                    for i, file in enumerate(topfiles_set):
                        filelist = ['%s%d__%d.%s' % (filename, i, j + 1,ext) \
                                    for j in range(self.shower_card['nsplit_jobs'])]
                        if out_id=='TOP':
                            infile="%d\n%s\n%s\n" % \
                                (self.shower_card['nsplit_jobs'],
                                 '\n'.join(filelist),
                                 '\n'.join([str(norm)] * self.shower_card['nsplit_jobs']))
                            p = misc.Popen([pjoin(self.me_dir, 'Utilities', 'sum_plots')],
                                           stdin=subprocess.PIPE,
                                           stdout=os.open(os.devnull, os.O_RDWR), 
                                           cwd=pjoin(self.me_dir, 'Events', self.run_name))
                            p.communicate(input = infile.encode())
                            files.mv(pjoin(self.me_dir, 'Events', self.run_name, 'sum.top'),
                                     pjoin(self.me_dir, 'Events', self.run_name, '%s%d.top' % (filename, i)))
                        elif out_id=='HWU':
                            out=pjoin(self.me_dir,'Events',
                                      self.run_name,'%s%d'% (filename,i))
                            histos=[]
                            norms=[]
                            for plotfile in plotfiles:
                                histos.append({'dirname':plotfile})
                                norms.append(norm)
                            self.combine_plots_HwU(histos,out,normalisation=norms)
                            try:
                                misc.call(['gnuplot','%s%d.gnuplot' % (filename, i)],\
                                          stdout=os.open(os.devnull, os.O_RDWR),\
                                          stderr=os.open(os.devnull, os.O_RDWR),\
                                          cwd=pjoin(self.me_dir, 'Events',self.run_name))
                            except Exception:
                                pass

                        plotfiles2.append(pjoin(self.me_dir, 'Events', self.run_name, '%s%d.%s' % (filename, i,ext)))
                        tar = tarfile.open(
                                pjoin(self.me_dir, 'Events', self.run_name, '%s%d.tar.gz' % (filename, i)), 'w:gz')
                        for f in filelist:
                            tar.add(pjoin(self.me_dir, 'Events', self.run_name, f), arcname=f)
                        files.rm([pjoin(self.me_dir, 'Events', self.run_name, f) for f in filelist])

                    tar.close()

                    ffiles = 'files'
                    have = 'have'
                    if len(plotfiles2) == 1:
                        ffiles = 'file'
                        have = 'has'

                    message = ('The %s %s %s been generated, with histograms in the' + \
                            ' %s, obtained by showering the parton-level' + \
                            ' file %s.gz with %s.\n' + \
                            'The files from the different shower ' + \
                            'jobs (before combining them) can be found inside %s.') % \
                            (ffiles, ', '.join(plotfiles2), have, hist_format,\
                             evt_file, shower, 
                             ', '.join([f.replace('%s' % ext, 'tar.gz') for f in plotfiles2]))

                else:
                    message = ('The following files have been generated:\n  %s\n' + \
                            'They contain histograms in the' + \
                            ' %s, obtained by showering the parton-level' + \
                            ' file %s.gz with %s.') % ('\n  '.join(plotfiles), \
                            hist_format, evt_file, shower)
                
        # Now arxiv the shower card used if RunMaterial is present
        run_dir_path = pjoin(rundir, self.run_name)
        if os.path.exists(pjoin(run_dir_path,'RunMaterial.tar.gz')):
            misc.call(['tar','-xzpf','RunMaterial.tar.gz'],cwd=run_dir_path)
            files.cp(pjoin(self.me_dir,'Cards','shower_card.dat'),
               pjoin(run_dir_path,'RunMaterial','shower_card_for_%s_%d.dat'\
                                                          %(shower, count)))
            misc.call(['tar','-czpf','RunMaterial.tar.gz','RunMaterial'], 
                                                           cwd=run_dir_path)
            shutil.rmtree(pjoin(run_dir_path,'RunMaterial'))
        
        if self.run_card['ickkw'] >0 :
            if self.run_card['ickkw'] != 3 or shower != 'PYTHIA8':
                logger.warning("Merged cross-section not retrieved by MadGraph. Please check the parton-shower log to get the correct cross-section after merging")
            else:
                pythia_log = misc.BackRead(pjoin(rundir, "mcatnlo_run.log") )
                
                pythiare = re.compile(r"\s*Les Houches User Process\(es\)\s+9999\s*\|\s*(?P<generated>\d+)\s+(?P<tried>\d+)\s+(?P<accepted>\d+)\s*\|\s*(?P<xsec>[\d\.DeE\-+]+)\s+(?P<xerr>[\d\.DeE\-+]+)\s*\|")    
                # | Les Houches User Process(es)                  9999 |       10000      10000       7115 |   1.120e-04  0.000e+00 |     
                                                         
                for line in pythia_log:
                    info = pythiare.search(line)
                    if not info:
                        continue
                    try:
                        # Pythia cross section in mb, we want pb
                        sigma_m = float(info.group('xsec').replace('D','E')) *1e9
                        sigma_err = float(info.group('xerr').replace('D','E')) *1e9
                        Nacc = int(info.group('accepted'))
                        #Ntry = int(info.group('accepted'))
                    except:
                        logger.warning("Merged cross-section not retrieved by MadGraph. Please check the parton-shower log to get the correct cross-section after merging")
                        break
                    
                    self.results.add_detail('cross_pythia', sigma_m)
                    self.results.add_detail('nb_event_pythia', Nacc)
                    self.results.add_detail('error_pythia', sigma_err)
                    self.results.add_detail('shower_dir', os.path.basename(rundir))
                    logger.info("\nFxFx  Cross-Section:\n"+\
                                "======================\n"+\
                                "    %f pb.\n"
                                "    Number of events after merging: %s\n", sigma_m, Nacc, '$MG:BOLD')
                    break
                else:
                    logger.warning("Merged cross-section not retrieved by MadGraph. Please check the parton-shower log to get the correct cross-section after merging")
                                    
                
            
            
            
            
        # end of the run, gzip files and print out the message/warning
        for f in to_gzip:
            misc.gzip(f)
        if message:
            logger.info(message)
        if warning:
            logger.warning(warning)

        self.update_status('Run complete', level='shower', update_results=True)

    ############################################################################
    def set_run_name(self, name, tag=None, level='parton', reload_card=False,**opts):
        """define the run name, the run_tag, the banner and the results."""
        
        # when are we force to change the tag new_run:previous run requiring changes
        upgrade_tag = {'parton': ['parton','delphes','shower','madanalysis5_hadron'],
                       'shower': ['shower','delphes','madanalysis5_hadron'],
                       'delphes':['delphes'],
                       'madanalysis5_hadron':['madanalysis5_hadron'],
                       'plot':[]}
        
        if name == self.run_name:        
            if reload_card:
                run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
                with misc.TMP_variable(banner_mod.RunCardNLO, 'allow_scan', True): 
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
        self.banner = banner_mod.recover_banner(self.results, level, self.run_name, tag)
        if 'mgruncard' in self.banner:
            self.run_card = self.banner.charge_card('run_card')
        else:
            self.banner.add(pjoin(self.me_dir, 'Cards', 'run_card.dat'))
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

        if 'event' in self.to_store:
            if os.path.exists(pjoin(self.me_dir,'Events', self.run_name, 'events.lhe')):
                if not  os.path.exists(pjoin(self.me_dir,'Events', self.run_name, 'events.lhe.gz')):
                    self.update_status('gzipping output file: events.lhe', level='parton', error=True)
                    misc.gzip(pjoin(self.me_dir,'Events', self.run_name, 'events.lhe'))
                else:
                    os.remove(pjoin(self.me_dir,'Events', self.run_name, 'events.lhe'))
            if os.path.exists(pjoin(self.me_dir,'Events','reweight.lhe')):
                os.remove(pjoin(self.me_dir,'Events', 'reweight.lhe'))
                
        
        tag = self.run_card['run_tag']
        
        self.to_store = []


    ############################################################################
    def get_Gdir(self, Pdir=None):
        """get the list of Gdirectory if not yet saved."""
        
        if hasattr(self, "Gdirs"):
            if self.me_dir in self.Gdirs:
                if Pdir is None:
                    return sum(self.Gdirs.values())
                else:
                    return self.Gdirs[Pdir]
                
        Pdirs = self.get_Pdir()
        Gdirs = {self.me_dir:[]}             
        for P in Pdirs:
            Gdirs[P] = [pjoin(P,G) for G in os.listdir(P) if G.startswith('G') and 
                                                os.path.isdir(pjoin(P,G))]

        self.Gdirs = Gdirs
        return self.getGdir(Pdir)


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

        if nevents < 0 or \
           nevents > self.banner.get_detail('run_card', 'nevents'):
            nevents = self.banner.get_detail('run_card', 'nevents')

        nevents = nevents / self.shower_card['nsplit_jobs']

        mcmass_dict = {}
        for line in [l for l in self.banner['montecarlomasses'].split('\n') if l]:
            pdg = int(line.split()[0])
            mass = float(line.split()[1])
            mcmass_dict[pdg] = mass

        content = 'EVPREFIX=%s\n' % pjoin(os.path.split(evt_file)[1])
        content += 'NEVENTS=%d\n' % nevents
        content += 'NEVENTS_TOT=%d\n' % (self.banner.get_detail('run_card', 'nevents') /\
                                             self.shower_card['nsplit_jobs'])
        content += 'MCMODE=%s\n' % shower
        content += 'PDLABEL=%s\n' % pdlabel

        try:
            aewm1 = self.banner.get_detail('param_card', 'sminputs', 1).value
            raise KeyError
        except KeyError:
            mod = self.get_model()
            if not hasattr(mod, 'parameter_dict'):
                from models import model_reader
                mod = model_reader.ModelReader(mod)
                if "set EWscheme MZ_MW_alpha" in self.banner.get_detail('proc_card'):
                    mod.change_electroweak_mode("MZ_MW_alpha")
                #misc.sprint(self.banner.get_detail('proc_card'))
                mod.set_parameters_and_couplings(self.banner.param_card)
            aewm1 = 0
            for key in ['aEWM1', 'AEWM1', 'aEWm1', 'aewm1']:
                if key in mod['parameter_dict']:
                    aewm1 = mod['parameter_dict'][key]
                    break
                elif 'mdl_%s' % key in mod['parameter_dict']:
                    aewm1 = mod['parameter_dict']['mod_%s' % key]
                    break
            else:
                for key in ['aEW', 'AEW', 'aEw', 'aew']:
                    if key in mod['parameter_dict']:
                        aewm1 = 1./mod['parameter_dict'][key]
                        break
                    elif 'mdl_%s' % key in mod['parameter_dict']:
                        aewm1 = 1./mod['parameter_dict']['mod_%s' % key]
                        break 
           
        content += 'ALPHAEW=%s\n' % aewm1
        #content += 'PDFSET=%s\n' % self.banner.get_detail('run_card', 'lhaid')
        #content += 'PDFSET=%s\n' % max([init_dict['pdfsup1'],init_dict['pdfsup2']])
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
        try:
            content += 'EMASS=%s\n' % mcmass_dict[11]
            content += 'MUMASS=%s\n' % mcmass_dict[13]
            content += 'TAUMASS=%s\n' % mcmass_dict[15]
        except KeyError:
            # this is for backward compatibility
            mcmass_lines = [l for l in \
                    open(pjoin(self.me_dir, 'SubProcesses', 'MCmasses_%s.inc' % shower.upper())
                            ).read().split('\n') if l]
            new_mcmass_dict = {}
            for l in mcmass_lines:
                key, val = l.split('=')
                new_mcmass_dict[key.strip()] = val.replace('d', 'e').strip()
            content += 'EMASS=%s\n' % new_mcmass_dict['mcmass(11)']
            content += 'MUMASS=%s\n' % new_mcmass_dict['mcmass(13)']
            content += 'TAUMASS=%s\n' % new_mcmass_dict['mcmass(15)']

        content += 'GMASS=%s\n' % mcmass_dict[21]
        content += 'EVENT_NORM=%s\n' % self.banner.get_detail('run_card', 'event_norm').lower()
        # check if need to link lhapdf
        if int(self.shower_card['pdfcode']) > 1 or \
            (pdlabel=='lhapdf' and int(self.shower_card['pdfcode'])==1) or \
            shower=='HERWIGPP' : 
            # Use LHAPDF (should be correctly installed, because
            # either events were already generated with them, or the
            # user explicitly gives an LHAPDF number in the
            # shower_card).
            self.link_lhapdf(pjoin(self.me_dir, 'lib'))
            lhapdfpath = subprocess.Popen([self.options['lhapdf'], '--prefix'], 
                                          stdout = subprocess.PIPE).stdout.read().decode(errors='ignore').strip()
            content += 'LHAPDFPATH=%s\n' % lhapdfpath
            pdfsetsdir = self.get_lhapdf_pdfsetsdir()
            if self.shower_card['pdfcode']==0:
                lhaid_list = ''
                content += ''
            elif self.shower_card['pdfcode']==1:
                lhaid_list = [max([init_dict['pdfsup1'],init_dict['pdfsup2']])]
                content += 'PDFCODE=%s\n' % max([init_dict['pdfsup1'],init_dict['pdfsup2']])
            else:
                lhaid_list = [abs(int(self.shower_card['pdfcode']))]
                content += 'PDFCODE=%s\n' % self.shower_card['pdfcode']
            self.copy_lhapdf_set(lhaid_list, pdfsetsdir)
        elif int(self.shower_card['pdfcode'])==1 or \
            int(self.shower_card['pdfcode'])==-1 and True:
            # Try to use LHAPDF because user wants to use the same PDF
            # as was used for the event generation. However, for the
            # event generation, LHAPDF was not used, so non-trivial to
            # see if if LHAPDF is available with the corresponding PDF
            # set. If not found, give a warning and use build-in PDF
            # set instead.
            try:
                lhapdfpath = subprocess.Popen([self.options['lhapdf'], '--prefix'], 
                                              stdout = subprocess.PIPE).stdout.read().decode(errors='ignore').strip()
                self.link_lhapdf(pjoin(self.me_dir, 'lib'))
                content += 'LHAPDFPATH=%s\n' % lhapdfpath
                pdfsetsdir = self.get_lhapdf_pdfsetsdir()
                lhaid_list = [max([init_dict['pdfsup1'],init_dict['pdfsup2']])]
                content += 'PDFCODE=%s\n' % max([init_dict['pdfsup1'],init_dict['pdfsup2']])
                self.copy_lhapdf_set(lhaid_list, pdfsetsdir)
            except Exception:
                logger.warning('Trying to shower events using the same PDF in the shower as used in the generation'+\
                                   ' of the events using LHAPDF. However, no valid LHAPDF installation found with the'+\
                                   ' needed PDF set. Will use default internal PDF for the shower instead. To use the'+\
                                   ' same set as was used in the event generation install LHAPDF and set the path using'+\
                                   ' "set /path_to_lhapdf/bin/lhapdf-config" from the MadGraph5_aMC@NLO python shell')
                content += 'LHAPDFPATH=\n' 
                content += 'PDFCODE=0\n'
        else:
            content += 'LHAPDFPATH=\n' 
            content += 'PDFCODE=0\n'

        content += 'ICKKW=%s\n' % self.banner.get_detail('run_card', 'ickkw')
        if self.banner.get_detail('run_card', 'mcatnlo_delta') :
            content += 'DELTA=ON\n'
        else:
            content += 'DELTA=OFF\n'
        content += 'PTJCUT=%s\n' % self.banner.get_detail('run_card', 'ptj')
        # add the pythia8/hwpp path(s)
        if self.options['pythia8_path']:
            content+='PY8PATH=%s\n' % self.options['pythia8_path']
        if self.options['hwpp_path']:
            content+='HWPPPATH=%s\n' % self.options['hwpp_path']
        if self.options['thepeg_path'] and self.options['thepeg_path'] != self.options['hwpp_path']:
            content+='THEPEGPATH=%s\n' % self.options['thepeg_path']
        if self.options['hepmc_path'] and self.options['hepmc_path'] != self.options['hwpp_path']:
            content+='HEPMCPATH=%s\n' % self.options['hepmc_path']
        
        output = open(pjoin(self.me_dir, 'MCatNLO', 'banner.dat'), 'w')
        output.write(content)
        output.close()
        return shower


    def run_reweight(self, only):
        """runs the reweight_xsec_events executables on each sub-event file generated
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
        evt_wghts = [float(line.split()[3]) for line in lines[:-1] if line.split()[1] != '0']
        if self.run_card['event_norm'].lower()=='bias' and self.run_card['nevents'] != 0:
            evt_wghts[:]=[1./float(self.run_card['nevents']) for wgt in evt_wghts]
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
            last_line = subprocess.Popen(['tail',  '-n1', '%s.rwgt' % \
                    pjoin(self.me_dir, 'SubProcesses', evt_file)], \
                    stdout = subprocess.PIPE).stdout.read().decode(errors='ignore').strip()
            if last_line != "</LesHouchesEvents>":
                raise aMCatNLOError('An error occurred during reweighting. Check the' + \
                        '\'reweight_xsec_events.output\' files inside the ' + \
                        '\'SubProcesses/P*/G*/\' directories for details')

        #update file name in nevents_unweighted
        newfile = open(nev_unw, 'w')
        for line in lines:
            if line:
                newfile.write(line.replace(line.split()[0], line.split()[0] + '.rwgt') + '\n')
        newfile.close()
        return self.pdf_scale_from_reweighting(evt_files,evt_wghts)

    def pdf_scale_from_reweighting(self, evt_files,evt_wghts):
        """This function takes the files with the scale and pdf values
        written by the reweight_xsec_events.f code
        (P*/G*/pdf_scale_dependence.dat) and computes the overall
        scale and PDF uncertainty (the latter is computed using the
        Hessian method (if lhaid<90000) or Gaussian (if lhaid>90000))
        and returns it in percents.  The expected format of the file
        is: n_scales xsec_scale_central xsec_scale1 ...  n_pdf
        xsec_pdf0 xsec_pdf1 ...."""

        scales=[]
        pdfs=[]
        for i,evt_file in enumerate(evt_files):
            path, evt=os.path.split(evt_file)
            with open(pjoin(self.me_dir, 'SubProcesses', path, 'scale_pdf_dependence.dat'),'r') as f:
                data_line=f.readline()
                if "scale variations:" in data_line:
                    for j,scale in enumerate(self.run_card['dynamical_scale_choice']):
                        data_line = f.readline().split()
                        scales_this = [float(val)*evt_wghts[i] for val in f.readline().replace("D", "E").split()]
                        try:
                            scales[j] = [a + b for a, b in zip(scales[j], scales_this)]
                        except IndexError:
                            scales+=[scales_this]
                    data_line=f.readline()
                if "pdf variations:" in data_line:
                    for j,pdf in enumerate(self.run_card['lhaid']):
                        data_line = f.readline().split()
                        pdfs_this = [float(val)*evt_wghts[i] for val in f.readline().replace("D", "E").split()]
                        try:
                            pdfs[j] = [a + b for a, b in zip(pdfs[j], pdfs_this)]
                        except IndexError:
                            pdfs+=[pdfs_this]

        # get the scale uncertainty in percent
        scale_info=[]
        for j,scale in enumerate(scales):
            s_cen=scale[0]
            if s_cen != 0.0 and self.run_card['reweight_scale'][j]:
                # max and min of the full envelope
                s_max=(max(scale)/s_cen-1)*100
                s_min=(1-min(scale)/s_cen)*100
                # ren and fac scale dependence added in quadrature
                ren_var=[]
                fac_var=[]
                for i in range(len(self.run_card['rw_rscale'])):
                    ren_var.append(scale[i]-s_cen) # central fac scale
                for i in range(len(self.run_card['rw_fscale'])):
                    fac_var.append(scale[i*len(self.run_card['rw_rscale'])]-s_cen) # central ren scale
                s_max_q=((s_cen+math.sqrt(math.pow(max(ren_var),2)+math.pow(max(fac_var),2)))/s_cen-1)*100
                s_min_q=(1-(s_cen-math.sqrt(math.pow(min(ren_var),2)+math.pow(min(fac_var),2)))/s_cen)*100
                s_size=len(scale)
            else:
                s_max=0.0
                s_min=0.0
                s_max_q=0.0
                s_min_q=0.0
                s_size=len(scale)
            scale_info.append({'cen':s_cen, 'min':s_min, 'max':s_max, \
                               'min_q':s_min_q, 'max_q':s_max_q, 'size':s_size, \
                               'label':self.run_card['dynamical_scale_choice'][j], \
                               'unc':self.run_card['reweight_scale'][j]})

        # check if we can use LHAPDF to compute the PDF uncertainty
        if any(self.run_card['reweight_pdf']):
            lhapdf = misc.import_python_lhapdf(self.options['lhapdf'])
            if lhapdf:
                use_lhapdf = True
            else:
                logger.warning("Failed to access python version of LHAPDF: "\
                                   "cannot compute PDF uncertainty from the "\
                                   "weights in the events. The weights in the LHE " \
                                   "event files will still cover all PDF set members, "\
                                   "but there will be no PDF uncertainty printed in the run summary. \n "\
                                   "If the python interface to LHAPDF is available on your system, try "\
                                   "adding its location to the PYTHONPATH environment variable and the"\
                                   "LHAPDF library location to LD_LIBRARY_PATH (linux) or DYLD_LIBRARY_PATH (mac os x).")
                use_lhapdf=False                
            
        # turn off lhapdf printing any messages
        if any(self.run_card['reweight_pdf']) and use_lhapdf: lhapdf.setVerbosity(0)

        pdf_info=[]
        for j,pdfset in enumerate(pdfs):
            p_cen=pdfset[0]
            if p_cen != 0.0 and self.run_card['reweight_pdf'][j]:
                if use_lhapdf:
                    pdfsetname=self.run_card['lhapdfsetname'][j]
                    try:
                        p=lhapdf.getPDFSet(pdfsetname)
                        ep=p.uncertainty(pdfset,-1)
                        p_cen=ep.central
                        p_min=abs(ep.errminus/p_cen)*100
                        p_max=abs(ep.errplus/p_cen)*100
                        p_type=p.errorType
                        p_size=p.size
                        p_conf=p.errorConfLevel
                    except:
                        logger.warning("Could not access LHAPDF to compute uncertainties for %s" % pdfsetname)
                        p_min=0.0
                        p_max=0.0
                        p_type='unknown'
                        p_conf='unknown'
                        p_size=len(pdfset)
                else:
                    p_min=0.0
                    p_max=0.0
                    p_type='unknown'
                    p_conf='unknown'
                    p_size=len(pdfset)
                    pdfsetname=self.run_card['lhaid'][j]
            else:
                p_min=0.0
                p_max=0.0
                p_type='none'
                p_conf='unknown'
                p_size=len(pdfset)
                pdfsetname=self.run_card['lhaid'][j]
            pdf_info.append({'cen':p_cen, 'min':p_min, 'max':p_max, \
                             'unc':p_type, 'name':pdfsetname, 'size':p_size, \
                             'label':self.run_card['lhaid'][j], 'conf':p_conf})

        scale_pdf_info=[scale_info,pdf_info]
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

    def run_all(self, job_dict, arg_list, run_type='monitor', split_jobs = False):
        """runs the jobs in job_dict (organized as folder: [job_list]), with arguments args"""
        self.ijob = 0
        if run_type != 'shower':
            self.njobs = sum(len(jobs) for jobs in job_dict.values()) * len(arg_list)
            for args in arg_list:
                for Pdir, jobs in job_dict.items():
                    for job in jobs:
                        self.run_exe(job, args, run_type, cwd=pjoin(self.me_dir, 'SubProcesses', Pdir) )
            if self.cluster_mode == 2:
                time.sleep(1) # security to allow all jobs to be launched
        else:
            self.njobs = len(arg_list)
            for args in arg_list:
                [(cwd, exe)] = list(job_dict.items())
                self.run_exe(exe, args, run_type, cwd)

        if self.njobs:
            self.wait_for_complete(run_type)



    def check_event_files(self,jobs):
        """check the integrity of the event files after splitting, and resubmit 
        those which are not nicely terminated"""
        jobs_to_resubmit = []
        for job in jobs:
            last_line = ''
            try:
                last_line = subprocess.Popen(
                        ['tail', '-n1', pjoin(job['dirname'], 'events.lhe')], \
                    stdout = subprocess.PIPE).stdout.read().decode(errors='ignore').strip()
            except IOError:
                pass
            if last_line != "</LesHouchesEvents>":
                jobs_to_resubmit.append(job)
        self.njobs = 0
        if jobs_to_resubmit:
            run_type = 'Resubmitting broken jobs'
            logger.info('Some event files are broken, corresponding jobs will be resubmitted.')
            for job in jobs_to_resubmit:
                logger.debug('Resubmitting ' + job['dirname'] + '\n')
            self.run_all_jobs(jobs_to_resubmit,2,fixed_order=False)


    def find_jobs_to_split(self, pdir, job, arg):
        """looks into the nevents_unweighed_splitted file to check how many
        split jobs are needed for this (pdir, job). arg is F, B or V"""
        # find the number of the integration channel
        splittings = []
        ajob = open(pjoin(self.me_dir, 'SubProcesses', pdir, job)).read()
        pattern = re.compile(r'for i in (\d+) ; do')
        match = re.search(pattern, ajob)
        channel = match.groups()[0]
        # then open the nevents_unweighted_splitted file and look for the 
        # number of splittings to be done
        nevents_file = open(pjoin(self.me_dir, 'SubProcesses', 'nevents_unweighted_splitted')).read()
        # This skips the channels with zero events, because they are
        # not of the form GFXX_YY, but simply GFXX
        pattern = re.compile(r"%s_(\d+)/events.lhe" % \
                          pjoin(pdir, 'G%s%s' % (arg,channel)))
        matches = re.findall(pattern, nevents_file)
        for m in matches:
            splittings.append(m)
        return splittings


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

        #this is for the cluster/multicore run
        elif 'reweight' in exe:
            # a reweight run
            # Find the correct PDF input file
            input_files, output_files = [], []
            pdfinput = self.get_pdf_input_filename()
            if os.path.exists(pdfinput):
                input_files.append(pdfinput)
            input_files.append(pjoin(os.path.dirname(exe), os.path.pardir, 'reweight_xsec_events'))
            input_files.append(pjoin(cwd, os.path.pardir, 'leshouche_info.dat'))
            input_files.append(pjoin(cwd, os.path.pardir, 'orderstags_glob.dat'))
            input_files.append(args[0])
            output_files.append('%s.rwgt' % os.path.basename(args[0]))
            output_files.append('reweight_xsec_events.output')
            output_files.append('scale_pdf_dependence.dat')

            return self.cluster.submit2(exe, args, cwd=cwd, 
                             input_files=input_files, output_files=output_files,
                             required_output=output_files) 

        elif 'ajob' in exe:
            # the 'standard' amcatnlo job
            # check if args is a list of string 
            if type(args[0]) == str:
                input_files, output_files, required_output, args = self.getIO_ajob(exe,cwd,args)
                #submitting
                self.cluster.submit2(exe, args, cwd=cwd, 
                             input_files=input_files, output_files=output_files,
                             required_output=required_output)

#                # keep track of folders and arguments for splitted evt gen
#                subfolder=output_files[-1].split('/')[0]
#                if len(args) == 4 and '_' in subfolder:
#                    self.split_folders[pjoin(cwd,subfolder)] = [exe] + args

        elif 'shower' in exe:
            # a shower job
            # args are [shower, output(HEP or TOP), run_name]
            # cwd is the shower rundir, where the executable are found
            input_files, output_files = [], []
            shower = args[0]
            # the input files
            if shower == 'PYTHIA8':
                input_files.append(pjoin(cwd, 'Pythia8.exe'))
                input_files.append(pjoin(cwd, 'Pythia8.cmd'))
                if os.path.exists(pjoin(self.options['pythia8_path'], 'xmldoc')):
                    input_files.append(pjoin(cwd, 'config.sh'))
                    input_files.append(pjoin(self.options['pythia8_path'], 'xmldoc'))
                else:
                    input_files.append(pjoin(self.options['pythia8_path'], 'share/Pythia8/xmldoc'))
            else:
                input_files.append(pjoin(cwd, 'MCATNLO_%s_EXE' % shower))
                input_files.append(pjoin(cwd, 'MCATNLO_%s_input' % shower))
            if shower == 'HERWIGPP':
                if os.path.exists(pjoin(self.options['hwpp_path'], 'bin', 'Herwig++')):
                    input_files.append(pjoin(cwd, 'Herwig++'))
                if os.path.exists(pjoin(self.options['hwpp_path'], 'bin', 'Herwig')):
                    input_files.append(pjoin(cwd, 'Herwig'))
                input_files.append(pjoin(cwd, 'HepMCFortran.so'))
            if len(args) == 3:
                if os.path.exists(pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe.gz')):
                    input_files.append(pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe.gz'))
                elif os.path.exists(pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe')):
                    input_files.append(pjoin(self.me_dir, 'Events', self.run_name, 'events.lhe'))
                else:
                    raise aMCatNLOError('Event file not present in %s' % \
                            pjoin(self.me_dir, 'Events', self.run_name))
            else: 
                input_files.append(pjoin(cwd, 'events_%s.lhe' % args[3]))
            # the output files
            if len(args) == 3:
                output_files.append('mcatnlo_run.log')
            else:
                output_files.append('mcatnlo_run_%s.log' % args[3]) 
            if args[1] == 'HEP':
                if len(args) == 3:
                    fname = 'events'
                else:
                    fname = 'events_%s' % args[3]
                if shower in ['PYTHIA8', 'HERWIGPP']:
                    output_files.append(fname + '.hepmc.gz')
                else:
                    output_files.append(fname + '.hep.gz')
            elif args[1] == 'TOP' or args[1] == 'HWU':
                if len(args) == 3:
                    fname = 'histfile'
                else:
                    fname = 'histfile_%s' % args[3]
                output_files.append(fname + '.tar')
            else:
                raise aMCatNLOError('Not a valid output argument for shower job :  %d' % args[1])
            #submitting
            self.cluster.submit2(exe, args, cwd=cwd, 
                    input_files=input_files, output_files=output_files)

        else:
            return self.cluster.submit(exe, args, cwd=cwd)

    def getIO_ajob(self,exe,cwd, args):
        # use local disk if possible => need to stands what are the 
        # input/output files
        
        output_files = []
        required_output = []
        input_files = [pjoin(self.me_dir, 'SubProcesses', 'randinit'),
                     pjoin(cwd, 'symfact.dat'),
                     pjoin(cwd, 'iproc.dat'),
                     pjoin(cwd, 'initial_states_map.dat'),
                     pjoin(cwd, 'configs_and_props_info.dat'),
                     pjoin(cwd, 'leshouche_info.dat'),
                     pjoin(cwd, 'FKS_params.dat')]

        # For GoSam interface, we must copy the SLHA card as well
        if os.path.exists(pjoin(self.me_dir,'OLP_virtuals','gosam.rc')):
            input_files.append(pjoin(self.me_dir, 'Cards', 'param_card.dat'))

        if os.path.exists(pjoin(cwd,'nevents.tar')):
            input_files.append(pjoin(cwd,'nevents.tar'))
        
        if os.path.exists(pjoin(self.me_dir,'SubProcesses','OLE_order.olc')):
            input_files.append(pjoin(cwd, 'OLE_order.olc'))

        # File for the loop (might not be present if MadLoop is not used)
        if os.path.exists(pjoin(cwd,'MadLoop5_resources.tar.gz')) and \
                                            cluster.need_transfer(self.options):
            input_files.append(pjoin(cwd, 'MadLoop5_resources.tar.gz'))
        elif os.path.exists(pjoin(cwd,'MadLoop5_resources')) and \
                                            cluster.need_transfer(self.options):
            tf=tarfile.open(pjoin(cwd,'MadLoop5_resources.tar.gz'),'w:gz',
                                                           dereference=True)
            tf.add(pjoin(cwd,'MadLoop5_resources'),arcname='MadLoop5_resources')
            tf.close()
            input_files.append(pjoin(cwd, 'MadLoop5_resources.tar.gz'))
               
        if args[1] == 'born' or args[1] == 'all':
            # MADEVENT MINT FO MODE
            input_files.append(pjoin(cwd, 'madevent_mintFO'))
            if args[2] == '0':
                current = '%s_G%s' % (args[1],args[0])
            else:
                current = '%s_G%s_%s' % (args[1],args[0],args[2])
            if os.path.exists(pjoin(cwd,current)):
                input_files.append(pjoin(cwd, current))
            output_files.append(current)

            required_output.append('%s/results.dat' % current)
            required_output.append('%s/res_%s.dat' % (current,args[3]))
            required_output.append('%s/log_MINT%s.txt' % (current,args[3]))
            required_output.append('%s/mint_grids' % current)
            required_output.append('%s/grid.MC_integer' % current)
            if args[3] != '0':
                required_output.append('%s/scale_pdf_dependence.dat' % current)
                            
        elif args[1] == 'F' or args[1] == 'B':
            # MINTMC MODE
            input_files.append(pjoin(cwd, 'madevent_mintMC'))

            if args[2] == '0':
                current = 'G%s%s' % (args[1],args[0])
            else:
                current = 'G%s%s_%s' % (args[1],args[0],args[2])
            if os.path.exists(pjoin(cwd,current)):
                input_files.append(pjoin(cwd, current))
            output_files.append(current)
            if args[2] > '0':
                # this is for the split event generation
                output_files.append('G%s%s_%s' % (args[1], args[0], args[2]))
                required_output.append('G%s%s_%s/log_MINT%s.txt' % (args[1],args[0],args[2],args[3]))

            else:
                required_output.append('%s/log_MINT%s.txt' % (current,args[3]))
            if args[3] in ['0','1']:
                required_output.append('%s/results.dat' % current)
            if args[3] == '1':
                output_files.append('%s/results.dat' % current)

        else:
            raise aMCatNLOError('not valid arguments: %s' %(', '.join(args)))

        #Find the correct PDF input file
        pdfinput = self.get_pdf_input_filename()
        if os.path.exists(pdfinput):
            input_files.append(pdfinput)            
        return input_files, output_files, required_output,  args

    def activate_Pythia8_compilation(self, mode, options):
        """ Overwrite the files 'pythia8_control_setup.inc' and 'pythia8_opts' 
        # according to the interface options. If pythia_path is not specified, pythia8 will be considered
        # as unavaialable and dummy entries will be filled in. Otherwise, these files will be set accordingly. """
        # Also write dummies if we do fixed-order for now
        if mode in ['NLO', 'LO'] or not self.run_card['mcatnlo_delta'] :
            # Write dummy entries
            open(pjoin(self.me_dir, 'SubProcesses', 'pythia8_opts'),'w').write(
"""PYTHIA8INCLUDE=.
PYTHIA8TARGETS=pythia8_fortran_dummy.o
PYTHIA8LINKLIBS=""")
            open(pjoin(self.me_dir, 'SubProcesses', 'pythia8_control_setup.inc'),'w').write(
"""      data is_pythia_active/-1/
      data pythia_cmd_file/500*' '/""")
        else:
            if not self.options['pythia8_path']:
                raise aMCatNLOError('Cannot find Pythia8 path in configuration file')
            os.environ["PYTHIA8DATA"] = pjoin(self.options['pythia8_path'], 'share/Pythia8/xmldoc')
            # Write entries accoridng to the pythia8_path
            # Probably need to do something a bit more careful to asses when '-lz' really is necessary
            open(pjoin(self.me_dir, 'SubProcesses', 'pythia8_opts'),'w').write(
"""PYTHIA8INCLUDE=-I%(pythia8_prefix)s/include
PYTHIA8TARGETS=pythia8_wrapper.o
PYTHIA8LINKLIBS=%(pythia8_prefix)s/lib/libpythia8.a -lz -ldl"""%{'pythia8_prefix':self.options['pythia8_path']})
            # Initialize Pythia8 flag to 'available but not yet initialised" (==0)
            # For now, we don't use any pythia8.cmd card for initialization
            open(pjoin(self.me_dir, 'SubProcesses', 'pythia8_control_setup.inc'),'w').write(
"""      data is_pythia_active/0/
      data pythia_cmd_file/500*' '/""")

    def link_and_copy_epdf(self, pdlabel, lhaid, libdir):
        """links and copies the libraries/PDFs from ePDF/eMELA
        pdlabel is in the form epdf:setname
        """
        logger.info('Using eMELA for leptonic densities')

        epdflibdir = subprocess.Popen([self.options['eMELA'], '--libdir'],
                 stdout = subprocess.PIPE).stdout.read().decode().strip()

        epdfdatadir = subprocess.Popen([self.options['eMELA'], '--data'],
                 stdout = subprocess.PIPE).stdout.read().decode().strip()

        # update the LHAPDF data path

        if 'LHAPDF_DATA_PATH' in os.environ and os.environ['LHAPDF_DATA_PATH']:
            os.environ['LHAPDF_DATA_PATH'] = '%s:%s' % (epdfdatadir, os.environ['LHAPDF_DATA_PATH'])
        else:
            os.environ['LHAPDF_DATA_PATH'] = epdfdatadir

        pdfsets = self.get_lhapdf_pdfsets_list_static(epdfdatadir, '6.2')
        pdfsetname = [pdfsets[i]['filename'] for i in lhaid]

        # link the static library in lib
        lib = 'libeMELA.a'

        if os.path.exists(pjoin(libdir, lib)):
            files.rm(pjoin(libdir, lib))
        files.ln(pjoin(epdflibdir, lib), libdir)

        # create the PDFsets dir
        if not os.path.isdir(pjoin(libdir, 'PDFsets')):
            os.mkdir(pjoin(libdir, 'PDFsets'))

        #clean previous set of pdf used
        for name in os.listdir(pjoin(libdir, 'PDFsets')):
            if name not in pdfsetname:
                try:
                    if os.path.isdir(pjoin(libdir, 'PDFsets', name)):
                        shutil.rmtree(pjoin(libdir, 'PDFsets', name))
                    else:
                        os.remove(pjoin(libdir, 'PDFsets', name))
                except Exception as error:
                    logger.debug('%s', error)

        # copy the ePDF set in the PDFsets dir
        for setname in pdfsetname:
            shutil.copytree(os.path.join(epdfdatadir, setname), os.path.join(libdir, 'PDFsets', setname))

        # finally return the parsed info file
        return banner_mod.eMELA_info(os.path.join(libdir, 'PDFsets', setname, setname + '.info'), self.me_dir)


    def copy_lep_densities(self, name, sourcedir):
        """copies the leptonic densities so that they are correctly compiled
        """
        lep_d_path = os.path.join(sourcedir, 'PDF', 'lep_densities', name)
        pdf_path = os.path.join(sourcedir, 'PDF')
        # check that the name is correct, ie that the path exists
        if not os.path.isdir(lep_d_path):
            raise aMCatNLOError(('Invalid name for the dressed-lepton PDFs: %s\n' % (name)) + \
                    'The corresponding directory cannot be found in \n' + \
                    'Source/PDF/lep_densities')

        # now copy the files
        for filename in ['eepdf.f', 'gridpdfaux.f']:
            files.cp(os.path.join(lep_d_path, filename), pdf_path)


    def link_and_copy_epdf(self, pdlabel, lhaid, libdir):
        """links and copies the libraries/PDFs from ePDF/eMELA
        pdlabel is in the form epdf:setname
        """
        logger.info('Using eMELA for leptonic densities')

        epdflibdir = subprocess.Popen([self.options['eMELA'], '--libdir'],
                 stdout = subprocess.PIPE).stdout.read().decode().strip()

        epdfdatadir = subprocess.Popen([self.options['eMELA'], '--data'],
                 stdout = subprocess.PIPE).stdout.read().decode().strip()

        # update the LHAPDF data path

        if 'LHAPDF_DATA_PATH' in os.environ and os.environ['LHAPDF_DATA_PATH']:
            os.environ['LHAPDF_DATA_PATH'] = '%s:%s' % (epdfdatadir, os.environ['LHAPDF_DATA_PATH'])
        else:
            os.environ['LHAPDF_DATA_PATH'] = epdfdatadir

        pdfsets = self.get_lhapdf_pdfsets_list_static(epdfdatadir, '6.2')
        pdfsetname = [pdfsets[i]['filename'] for i in lhaid]

        # link the static library in lib
        lib = 'libeMELA.a'

        if os.path.exists(pjoin(libdir, lib)):
            files.rm(pjoin(libdir, lib))
        files.ln(pjoin(epdflibdir, lib), libdir)

        # create the PDFsets dir
        if not os.path.isdir(pjoin(libdir, 'PDFsets')):
            os.mkdir(pjoin(libdir, 'PDFsets'))

        #clean previous set of pdf used
        for name in os.listdir(pjoin(libdir, 'PDFsets')):
            if name not in pdfsetname:
                try:
                    if os.path.isdir(pjoin(libdir, 'PDFsets', name)):
                        shutil.rmtree(pjoin(libdir, 'PDFsets', name))
                    else:
                        os.remove(pjoin(libdir, 'PDFsets', name))
                except Exception as error:
                    logger.debug('%s', error)

        # copy the ePDF set in the PDFsets dir
        for setname in pdfsetname:
            shutil.copytree(os.path.join(epdfdatadir, setname), os.path.join(libdir, 'PDFsets', setname))

        # finally return the parsed info file
        return banner_mod.eMELA_info(os.path.join(libdir, 'PDFsets', setname, setname + '.info'), self.me_dir)


    def copy_lep_densities(self, name, sourcedir):
        """copies the leptonic densities so that they are correctly compiled
        """
        lep_d_path = os.path.join(sourcedir, 'PDF', 'lep_densities', name)
        pdf_path = os.path.join(sourcedir, 'PDF')
        # check that the name is correct, ie that the path exists
        if not os.path.isdir(lep_d_path):
            raise aMCatNLOError(('Invalid name for the dressed-lepton PDFs: %s\n' % (name)) + \
                    'The corresponding directory cannot be found in \n' + \
                    'Source/PDF/lep_densities')

        # now copy the files
        for filename in ['eepdf.f', 'gridpdfaux.f']:
            files.cp(os.path.join(lep_d_path, filename), pdf_path)


    def compile(self, mode, options):
        """compiles aMC@NLO to compute either NLO or NLO matched to shower, as
        specified in mode"""
        os.mkdir(pjoin(self.me_dir, 'Events', self.run_name))

        self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, 
                          '%s_%s_banner.txt' % (self.run_name, self.run_tag)))

        self.get_characteristics(pjoin(self.me_dir, 
                                        'SubProcesses', 'proc_characteristics'))

        #define a bunch of log files
        amcatnlo_log = pjoin(self.me_dir, 'compile_amcatnlo.log')
        madloop_log = pjoin(self.me_dir, 'compile_madloop.log')
        reweight_log = pjoin(self.me_dir, 'compile_reweight.log')
        test_log = pjoin(self.me_dir, 'test.log')

        # environmental variables to be included in make_opts
        self.make_opts_var = {}
        if self.proc_characteristics['has_loops'] and \
                          not os.path.exists(pjoin(self.me_dir,'OLP_virtuals')):
            self.make_opts_var['madloop'] = 'true'

        self.update_status('Compiling the code', level=None, update_results=True)

        libdir = pjoin(self.me_dir, 'lib')
        sourcedir = pjoin(self.me_dir, 'Source')

        #clean files
        files.rm([amcatnlo_log, madloop_log, reweight_log, test_log])
        #define which executable/tests to compile
        if '+' in mode:
            mode = mode.split('+')[0]
        if mode in ['NLO', 'LO']:
            exe = 'madevent_mintFO'
            tests = ['test_ME']
            self.analyse_card.write_card(pjoin(self.me_dir, 'SubProcesses', 'analyse_opts'))
            self.analyse_card.update_FO_extrapaths_ajob(pjoin(self.me_dir, 'SubProcesses', 'ajob_template'))
        elif mode in ['aMC@NLO', 'aMC@LO','noshower','noshowerLO']:
            exe = 'madevent_mintMC'
            tests = ['test_ME', 'test_MC']
            # write an analyse_opts with a dummy analysis so that compilation goes through
            with open(pjoin(self.me_dir, 'SubProcesses', 'analyse_opts'),'w') as fsock:
                fsock.write('FO_ANALYSE=analysis_dummy.o dbook.o open_output_files_dummy.o HwU_dummy.o\n')

        # Overwrite the files 'pythia8_control_setup.inc' and 'pythia8_opts' 
        # according to the interface options. If pythia_path is not specified, pythia8 will be considered
        # as unavaialable and dummy entries will be filled in. Otherwise, these files will be set accordingly. 
        self.activate_Pythia8_compilation(mode, options)

        #directory where to compile exe
        p_dirs = [d for d in \
                open(pjoin(self.me_dir, 'SubProcesses', 'subproc.mg')).read().split('\n') if d]

        # if --nocompile option is specified, check here that all exes exists. 
        # If they exists, return
        if all([os.path.exists(pjoin(self.me_dir, 'SubProcesses', p_dir, exe)) \
                for p_dir in p_dirs]) and options['nocompile']:
            return

        # rm links to lhapdflib/ PDFsets if exist
        if os.path.exists(pjoin(libdir, 'PDFsets')):
            files.rm(pjoin(libdir, 'PDFsets'))

        # read the run_card to find if lhapdf is used or not
        if self.run_card['pdlabel'] == 'lhapdf' and \
                (abs(self.banner.get_detail('run_card', 'lpp1')) not in [0, 3, 4] or \
                 abs(self.banner.get_detail('run_card', 'lpp2')) not in [0, 3, 4]):

            self.link_lhapdf(libdir, [pjoin('SubProcesses', p) for p in p_dirs])
            pdfsetsdir = self.get_lhapdf_pdfsetsdir()
            lhaid_list = self.run_card['lhaid']
            self.copy_lhapdf_set(lhaid_list, pdfsetsdir)

        # this is the case of collision with dressed leptons
        elif abs(self.banner.get_detail('run_card', 'lpp1')) == \
             abs(self.banner.get_detail('run_card', 'lpp2')) in [3,4]:

            # force not to use LHAPDF in this case
            if self.run_card['pdlabel'] == 'lhapdf':
                raise aMCatNLOError('Usage of LHAPDF with dressed-lepton collisions not possible')

            elif self.run_card['pdlabel'].startswith('emela'):
                logger.warning('For informations on the simulation of leptonic collisions at NLO accuracy')
                logger.warning('please refer to arXiv:2207.03265 and to the FAQ')
                logger.warning('https://answers.launchpad.net/mg5amcnlo/+faq/3324')

                # this is if the PDFs from ePDF/eMELA are employed
                self.make_opts_var['epdf'] = self.options['eMELA']
                self.update_make_opts()
                # link the LHAPDF libraries, but unset the corresponding keys in make_opts
                self.link_lhapdf(libdir)
                for kk in ['lhapdf', 'lhapdfversion', 'lhapdfsubversion', 'lhapdf_config']:
                    self.make_opts_var[kk] = None
                # and link eMELA
                emela_info = self.link_and_copy_epdf(self.run_card['pdlabel'], self.run_card['lhaid'], libdir)

                # find the uv scheme of the model. if a file called 'TOYXS' exists, use msbar
                #  (for testing purposes only) 
                if os.path.exists(pjoin(self.me_dir, 'TOYXS')):
                    uvscheme = 0
                else:
                    try:
                        Gmu = self.banner.get_detail('param_card', 'sminputs', 2)
                        uvscheme = 2 # Gmu scheme
                    except KeyError:
                        uvscheme = 1 # Alpha(mz) scheme
                # update the run_card variables (PDFscheme, alpha running params, etc) accordingly
                emela_info.update_epdf_emela_variables(self.banner, uvscheme)
                self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, 
                          '%s_%s_banner2.txt' % (self.run_name, self.run_tag)))

            elif self.run_card['pdlabel'] in  sum(self.run_card.allowed_lep_densities.values(),[]):
                # using internal densities: copy the files for the chosen density
                self.copy_lep_densities(self.run_card['pdlabel'], sourcedir)

        # bare leptons, or anything else
        else:
            if self.run_card['lpp1'] == 1 == self.run_card['lpp2']:
                logger.info('Using built-in libraries for PDFs')

            self.make_opts_var['lhapdf'] = ""

        # create param_card.inc and run_card.inc
        self.do_treatcards('', amcatnlo=True, mode=mode)

        # read the run_card to find if PineAPPL is used or not
        if self.run_card['pineappl']:
            self.make_opts_var['pineappl'] = 'True'
            # check validity of the PineAPPL installation
            for code in ['pineappl']:
                try:
                    p = subprocess.Popen([self.options[code], '--version'], \
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except OSError:
                    raise aMCatNLOError(('No valid %s installation found. \n' + \
                       'Please set the path to %s-config by using \n' + \
                       'MG5_aMC> set <absolute-path-to-%s>/bin/%s \n') % (code,code,code,code))
        else:
            self.make_opts_var['pineappl'] = ""

        if 'fastjet' in list(self.options.keys()) and self.options['fastjet']:
            self.make_opts_var['fastjet_config'] = self.options['fastjet']

        # add the make_opts_var to make_opts
        self.update_make_opts()
        
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
        
        # make StdHep (only necessary with MG option output_dependencies='internal')
        MCatNLO_libdir = pjoin(self.me_dir, 'MCatNLO', 'lib')
        if not os.path.exists(os.path.realpath(pjoin(MCatNLO_libdir, 'libstdhep.a'))) or \
            not os.path.exists(os.path.realpath(pjoin(MCatNLO_libdir, 'libFmcfio.a'))):  
            if  os.path.exists(pjoin(sourcedir,'StdHEP')):
                logger.info('Compiling StdHEP (can take a couple of minutes) ...')
                try:
                    misc.compile(['StdHEP'], cwd = sourcedir)
                except Exception as error:
                    logger.debug(str(error))
                    logger.warning("StdHep failed to compiled. This forbids to run NLO+PS with PY6 and Herwig6")
                    logger.info("details on the compilation error are available if the code is run with --debug flag")
                else:
                    logger.info('          ...done.')      
            else:
                logger.warning('Could not compile StdHEP because its'+\
                   ' source directory could not be found in the SOURCE folder.\n'+\
                             " Check the MG5_aMC option 'output_dependencies'.\n"+\
                   " This will prevent the use of HERWIG6/Pythia6 shower.")

        # make CutTools (only necessary with MG option output_dependencies='internal')
        if not os.path.exists(os.path.realpath(pjoin(libdir, 'libcts.a'))) or \
            not os.path.exists(os.path.realpath(pjoin(libdir, 'mpmodule.mod'))):
            if  os.path.exists(pjoin(sourcedir,'CutTools')):
                logger.info('Compiling CutTools (can take a couple of minutes) ...')
                misc.compile(['CutTools','-j1'], cwd = sourcedir, nb_core=1)
                logger.info('          ...done.')
            else:
                raise aMCatNLOError('Could not compile CutTools because its'+\
                   ' source directory could not be found in the SOURCE folder.\n'+\
                             " Check the MG5_aMC option 'output_dependencies.'")
        if not os.path.exists(os.path.realpath(pjoin(libdir, 'libcts.a'))) or \
            not os.path.exists(os.path.realpath(pjoin(libdir, 'mpmodule.mod'))):
            raise aMCatNLOError('CutTools compilation failed.')            

        # Verify compatibility between current compiler and the one which was
        # used when last compiling CutTools (if specified).
        compiler_log_path = pjoin(os.path.dirname((os.path.realpath(pjoin(
                                  libdir, 'libcts.a')))),'compiler_version.log')
        if os.path.exists(compiler_log_path):
            compiler_version_used = open(compiler_log_path,'r').read()
            if not str(misc.get_gfortran_version(misc.detect_current_compiler(\
                       pjoin(sourcedir,'make_opts')))) in compiler_version_used:
                if os.path.exists(pjoin(sourcedir,'CutTools')):
                    logger.info('CutTools was compiled with a different fortran'+\
                                            ' compiler. Re-compiling it now...')
                    misc.compile(['cleanCT'], cwd = sourcedir)
                    misc.compile(['CutTools','-j1'], cwd = sourcedir, nb_core=1)
                    logger.info('          ...done.')
                else:
                    raise aMCatNLOError("CutTools installation in %s"\
                                 %os.path.realpath(pjoin(libdir, 'libcts.a'))+\
                 " seems to have been compiled with a different compiler than"+\
                    " the one specified in MG5_aMC. Please recompile CutTools.")

        # make IREGI (only necessary with MG option output_dependencies='internal')
        if not os.path.exists(os.path.realpath(pjoin(libdir, 'libiregi.a'))) \
           and os.path.exists(pjoin(sourcedir,'IREGI')):
                logger.info('Compiling IREGI (can take a couple of minutes) ...')
                misc.compile(['IREGI'], cwd = sourcedir)
                logger.info('          ...done.')

        if os.path.exists(pjoin(libdir, 'libiregi.a')):
            # Verify compatibility between current compiler and the one which was
            # used when last compiling IREGI (if specified).
            compiler_log_path = pjoin(os.path.dirname((os.path.realpath(pjoin(
                                libdir, 'libiregi.a')))),'compiler_version.log')
            if os.path.exists(compiler_log_path):
                compiler_version_used = open(compiler_log_path,'r').read()
                if not str(misc.get_gfortran_version(misc.detect_current_compiler(\
                       pjoin(sourcedir,'make_opts')))) in compiler_version_used:
                    if os.path.exists(pjoin(sourcedir,'IREGI')):
                        logger.info('IREGI was compiled with a different fortran'+\
                                            ' compiler. Re-compiling it now...')
                        misc.compile(['cleanIR'], cwd = sourcedir)
                        misc.compile(['IREGI'], cwd = sourcedir)
                        logger.info('          ...done.')
                    else:
                        raise aMCatNLOError("IREGI installation in %s"\
                                %os.path.realpath(pjoin(libdir, 'libiregi.a'))+\
                 " seems to have been compiled with a different compiler than"+\
                    " the one specified in MG5_aMC. Please recompile IREGI.")

        # check if MadLoop virtuals have been generated
        if self.proc_characteristics['has_loops'] and \
                          not os.path.exists(pjoin(self.me_dir,'OLP_virtuals')):
            if mode in ['NLO', 'aMC@NLO', 'noshower']:
                tests.append('check_poles')
            pass

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
        except ImportError: 
            self.nb_core = 1

        compile_options = copy.copy(self.options)
        compile_options['nb_core'] = self.nb_core
        compile_cluster = cluster.MultiCore(**compile_options)
        logger.info('Compiling on %d cores' % self.nb_core)

        update_status = lambda i, r, f: self.donothing(i,r,f)
        for p_dir in p_dirs:
            compile_cluster.submit(prog = compile_dir, 
                               argument = [self.me_dir, p_dir, mode, options, 
                    tests, exe, self.options['run_mode']])
        try:
            compile_cluster.wait(self.me_dir, update_status)
        except Exception as  error:
            logger.warning("Compilation of the Subprocesses failed")
            if __debug__:
                raise
            compile_cluster.remove()
            self.do_quit('')

        logger.info('Checking test output:')
        for p_dir in p_dirs:
            logger.info(p_dir)
            for test in tests:
                logger.info(' Result for %s:' % test)

                this_dir = pjoin(self.me_dir, 'SubProcesses', p_dir) 
                #check that none of the tests failed
                self.check_tests(test, this_dir)


    def donothing(*args):
        pass


    def check_tests(self, test, dir):
        """just call the correct parser for the test log.
        Skip check_poles for LOonly folders"""
        if test in ['test_ME', 'test_MC']:
            return self.parse_test_mx_log(pjoin(dir, '%s.log' % test)) 
        elif test == 'check_poles' and not os.path.exists(pjoin(dir,'parton_lum_0.f')):
            return self.parse_check_poles_log(pjoin(dir, '%s.log' % test)) 


    def parse_test_mx_log(self, log):
        """read and parse the test_ME/MC.log file"""
        content = open(log).read()
        if 'FAILED' in content:
            logger.info('Output of the failing test:\n'+content[:-1],'$MG:BOLD')
            raise aMCatNLOError('Some tests failed, run cannot continue. Please search on https://answers.launchpad.net/mg5amcnlo for more information, and in case there is none, report the problem there.')
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


    def write_test_input(self, test):
        """write the input files to run test_ME/MC or check_poles"""
        if test in ['test_ME', 'test_MC']:
            content = "-2 -2\n" #generate randomly energy/angle
            content+= "100 100\n" #run 100 points for soft and collinear tests
            content+= "0\n" #all FKS configs
            content+= '\n'.join(["-1"] * 50) #random diagram (=first diagram)
        elif test == 'check_poles':
            content = '20 \n -1\n'
        elif test == 'check_sudakov':
            content = '7 \n -1\n'
        
        file = open(pjoin(self.me_dir, '%s_input.txt' % test), 'w')
        if test == 'test_MC':
            shower = self.run_card['parton_shower'].upper()
            header = "1 \n %s\n 1 -0.1\n-1 -0.1\n" % shower
            file.write(header + content)
        elif test == 'test_ME':
            header = "2 \n"
            file.write(header + content)
        else:
            file.write(content)
        file.close()


    action_switcher = AskRunNLO
    ############################################################################
    def ask_run_configuration(self, mode, options, switch={}):
        """Ask the question when launching generate_events/multi_run"""
        
        if 'parton' not in options:
            options['parton'] = False
        if 'reweightonly' not in options:
            options['reweightonly'] = False
        
        if mode == 'auto': 
            mode = None
        if not mode and (options['parton'] or options['reweightonly']):
            mode = 'noshower'  
        
        passing_cmd = []
        for key,value in switch.keys():
            passing_cmd.append('%s=%s' % (key,value))
        
        if 'do_reweight' in options and options['do_reweight']:
            passing_cmd.append('reweight=ON')
        if 'do_madspin' in options and  options['do_madspin']:
            passing_cmd.append('madspin=ON')

        force = self.force
        if mode == 'onlyshower':
            passing_cmd.append('onlyshower')
            force = True
        elif mode:
            passing_cmd.append(mode)
        ####mode = None # allow to overwrite it due to EW

        switch, cmd_switch = self.ask('', '0', [], ask_class = self.action_switcher,
                              mode=mode, force=(force or mode),
                              first_cmd=passing_cmd,
                              return_instance=True)

        if 'mode' in switch:
            mode = switch['mode']
        #assign the mode depending of the switch
        if not mode or mode == 'auto':
            if switch['order'] == 'LO':
                if switch['runshower']:
                    mode = 'aMC@LO'
                elif switch['fixed_order'] == 'ON':
                    mode = 'LO'
                else:
                    mode =  'noshowerLO'
            elif switch['order'] == 'NLO':
                if switch['runshower']:
                    mode = 'aMC@NLO'
                elif switch['fixed_order'] == 'ON':
                    mode = 'NLO'
                else:
                    mode =  'noshower'  
        logger.info('will run in mode: %s' % mode)                

        if mode == 'noshower':
            if switch['shower'] == 'OFF':
                logger.warning("""You have chosen not to run a parton shower. NLO events without showering are NOT physical. Please, shower the LesHouches events before using them for physics analyses. You have to choose NOW which parton-shower you WILL use and specify it in the run_card.""")   
            else:
                logger.info("""Your parton-shower choice is not available for running. Events will be generated for the associated parton shower. Remember that NLO events without showering are NOT physical.""", '$MG:BOLD')           

        
        # specify the cards which are needed for this run.
        cards = ['param_card.dat', 'run_card.dat']
        ignore = []
        if mode in ['LO', 'NLO']:
            options['parton'] = True
            ignore = ['shower_card.dat', 'madspin_card.dat']
            cards.append('FO_analyse_card.dat')
        else:
            if switch['madspin'] != 'OFF':
                cards.append('madspin_card.dat')
            if switch['reweight'] != 'OFF':
                cards.append('reweight_card.dat')
            if switch['madanalysis'] in ['HADRON', 'ON']:
                cards.append('madanalysis5_hadron_card.dat')                
        if 'aMC@' in mode:
            cards.append('shower_card.dat')
        if mode == 'onlyshower':
            cards = ['shower_card.dat']
        if options['reweightonly']:
            cards = ['run_card.dat']

        self.keep_cards(cards, ignore)
        
        if mode =='onlyshower':
            cards = ['shower_card.dat']
        
        
        # automatically switch to keep_wgt option
        first_cmd = cmd_switch.get_cardcmd()
                
        if not options['force'] and not self.force:
            self.ask_edit_cards(cards, plot=False, first_cmd=first_cmd, switch=switch)

        self.banner = banner_mod.Banner()

        # store the cards in the banner
        for card in cards:
            self.banner.add(pjoin(self.me_dir, 'Cards', card))
        # and the run settings
        run_settings = '\n'.join(['%s = %s' % (k, v) for (k, v) in switch.items()])
        self.banner.add_text('run_settings', run_settings)

        if not mode =='onlyshower':
            self.run_card = self.banner.charge_card('run_card')
            self.run_tag = self.run_card['run_tag']
            #this is if the user did not provide a name for the current run
            if not hasattr(self, 'run_name') or not self.run_name:
                self.run_name = self.find_available_run_name(self.me_dir)
                #add a tag in the run_name for distinguish run_type
                if self.run_name.startswith('run_'):
                    if mode in ['LO','aMC@LO','noshowerLO']:
                        self.run_name += '_LO' 
            self.set_run_name(self.run_name, self.run_tag, 'parton')
            if self.run_card['ickkw'] == 3 and mode in ['LO', 'aMC@LO', 'noshowerLO']:
                raise self.InvalidCmd("""FxFx merging (ickkw=3) not allowed at LO""")
            elif self.run_card['ickkw'] == 3 and ((mode in ['aMC@NLO'] and self.run_card['parton_shower'].upper() != 'PYTHIA8') or mode in ['noshower']):
                logger.warning("""You are running with FxFx merging enabled.  To be able to merge
    samples of various multiplicities without double counting, you
    have to remove some events after showering 'by hand'.  Please
    read http://amcatnlo.cern.ch/FxFx_merging.htm for more details.""")

                if self.run_card['parton_shower'].upper() == 'PYTHIA6Q':
                    raise self.InvalidCmd("""FxFx merging does not work with Pythia6's Q-squared ordered showers.""")
                elif self.run_card['parton_shower'].upper() != 'HERWIG6' and self.run_card['parton_shower'].upper() != 'PYTHIA8' and self.run_card['parton_shower'].upper() != 'HERWIGPP':
                    question="FxFx merging not tested for %s shower. Do you want to continue?\n"  % self.run_card['parton_shower'] + \
                        "Type \'n\' to stop or \'y\' to continue"
                    answers = ['n','y']
                    answer = self.ask(question, 'n', answers)
                    if answer == 'n':
                        error = '''Stop opertation'''
                        self.ask_run_configuration(mode, options)
    #                    raise aMCatNLOError(error)
            elif self.run_card['ickkw'] == -1 and mode in ['aMC@NLO', 'noshower']:
                    # NNLL+NLO jet-veto only possible for LO event generation or fNLO runs.
                raise self.InvalidCmd("""NNLL+NLO jet veto runs (ickkw=-1) only possible for fNLO or LO.""")
        if 'aMC@' in mode or mode == 'onlyshower':
            self.shower_card = self.banner.charge_card('shower_card')
            
        elif mode in ['LO', 'NLO']:
            analyse_card_path = pjoin(self.me_dir, 'Cards','FO_analyse_card.dat')
            self.analyse_card = self.banner.charge_card('FO_analyse_card')

        return mode


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
                "     computation of the total cross section and the filling of parton-level histograms.\n" + \
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
_launch_parser.add_option("-x", "--nocompile", default=False, action='store_true',
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
_launch_parser.add_option("-n", "--name", default=False, dest='run_name',
                            help="Provide a name to the run")
_launch_parser.add_option("-R", "--reweight", default=False, dest='do_reweight', action='store_true',
                            help="Run the reweight module (reweighting by different model parameters)")
_launch_parser.add_option("-M", "--madspin", default=False, dest='do_madspin', action='store_true',
                            help="Run the madspin package")



_generate_events_usage = "generate_events [MODE] [options]\n" + \
                "-- execute aMC@NLO \n" + \
                "   MODE can be either LO, NLO, aMC@NLO or aMC@LO (if omitted, it is asked in a separate question)\n" + \
                "     If mode is set to LO/NLO, no event generation will be performed, but only the \n" + \
                "     computation of the total cross section and the filling of parton-level histograms.\n" + \
                "     If mode is set to aMC@LO/aMC@NLO, after the cross-section computation, a .lhe \n" + \
                "     event file is generated which will be showered with the MonteCarlo specified \n" + \
                "     in the run_card.dat\n"

_generate_events_parser = misc.OptionParser(usage=_generate_events_usage)
_generate_events_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory for the generate_events, without editing them")
_generate_events_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_generate_events_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_generate_events_parser.add_option("-x", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found")
_generate_events_parser.add_option("-r", "--reweightonly", default=False, action='store_true',
                            help="Skip integration and event generation, just run reweight on the" + \
                                 " latest generated event files (see list in SubProcesses/nevents_unweighted)")
_generate_events_parser.add_option("-p", "--parton", default=False, action='store_true',
                            help="Stop the run after the parton level file generation (you need " + \
                                    "to shower the file in order to get physical results)")
_generate_events_parser.add_option("-o", "--only_generation", default=False, action='store_true',
                            help="Skip grid set up, just generate events starting from " + \
                            "the last available results")
_generate_events_parser.add_option("-n", "--name", default=False, dest='run_name',
                            help="Provide a name to the run")



_calculate_xsect_usage = "calculate_xsect [ORDER] [options]\n" + \
                "-- calculate cross section up to ORDER.\n" + \
                "   ORDER can be either LO or NLO (if omitted, it is set to NLO). \n"

_calculate_xsect_parser = misc.OptionParser(usage=_calculate_xsect_usage)
_calculate_xsect_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory for the launch, without editing them")
_calculate_xsect_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_calculate_xsect_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_calculate_xsect_parser.add_option("-x", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found")
_calculate_xsect_parser.add_option("-n", "--name", default=False, dest='run_name',
                            help="Provide a name to the run")
_calculate_xsect_parser.add_option("-o", "--only_generation", default=False, action='store_true',
                            help="Skip grid set up, just generate events starting from " + \
                            "the last available results")

_shower_usage = 'shower run_name [options]\n' + \
        '-- do shower/hadronization on parton-level file generated for run run_name\n' + \
        '   all the information (e.g. number of events, MonteCarlo, ...\n' + \
        '   are directly read from the header of the event file\n'
_shower_parser = misc.OptionParser(usage=_shower_usage)
_shower_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the shower_card present in the directory for the launch, without editing")

if '__main__' == __name__:
    # Launch the interface without any check if one code is already running.
    # This can ONLY run a single command !!
    import sys

    if sys.version_info < (3, 7):
        sys.exit('MadGraph5_aMc@NLO works only with python 3.7 and later.\n'+\
               'Please upgrade your version of python or specify a compatible version.')

    import os
    import optparse
    # Get the directory of the script real path (bin)
    # and add it to the current PYTHONPATH
    #root_path = os.path.dirname(os.path.dirname(os.path.realpath( __file__ )))
    #root_path = os.path.split(root_path)[0]
    sys.path.insert(0, os.path.join(root_path,'bin'))                                                     


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
    import internal.coloring_logging
    try:
        if __debug__ and options.logging == 'INFO':
            options.logging = 'DEBUG'
        if options.logging.isdigit():
            level = int(options.logging)
        else:
            level = eval('logging.' + options.logging)
        print(os.path.join(root_path, 'bin','internal', 'me5_logging.conf'))
        logging.config.fileConfig(os.path.join(root_path,'bin', 'internal', 'me5_logging.conf'))
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
                cmd_line =  aMCatNLOCmd(me_dir=root_path, force_run=True)
            else:
                cmd_line =  aMCatNLOCmdShell(me_dir=root_path, force_run=True)

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

