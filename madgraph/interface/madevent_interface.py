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

import atexit
import glob
import logging
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
devnull = os.open(os.devnull, os.O_RDWR)
 
try:
    # import from madgraph directory
    import madgraph.interface.extended_cmd as cmd
    import madgraph.iolibs.misc as misc
    import madgraph.iolibs.files as files
    import Template as madevent
    from madgraph import MG5DIR
    MADEVENT = False
except:
    # import from madevent directory
    import internal.extended_cmd as cmd
    import internal.misc as misc    
    import internal as madevent
    import internal.files as files
    MADEVENT = True


# Special logger for the Cmd Interface
logger = logging.getLogger('madevent.stdout') # -> stdout
logger_stderr = logging.getLogger('madevent.stderr') # ->stderr


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
    error_debug += 'More information is found in \'%s\'.\n' 
    error_debug += 'Please attach this file to your report.'
    
    keyboard_stop_msg = """stopping all operation
            in order to quit madevent please enter exit"""
    
    # Define the Error
    InvalidCmd = madevent.InvalidCmd
    
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
        "*           Type 'tutorial' to learn how MG5 works         *\n" + \
        "*                                                          *\n" + \
        "************************************************************")
        
        cmd.Cmd.__init__(self, *arg, **opt)
        
    def get_history_header(self):
        """return the history header""" 
        return self.history_header % misc.get_time_info()
    
    
#===============================================================================
# HelpToCmd
#===============================================================================
class HelpToCmd(object):
    """ The Series of help routine for the MadEventCmd"""
    
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
        
    def help_shell(self):
        logger.info("syntax: shell CMD (or ! CMD)")
        logger.info("-- run the shell command CMD and catch output")


    def run_options_help(self):
        logger.info("-- options:")
        logger.info("      --cluster= :[%s] cluster choice: 0 single machine" % 
                                                              self.cluster_mode)
        logger.info("                                      1 pbs cluster")
        logger.info("                                      2 multicore")
        logger.info("      --queue= :[%s] queue name on the pbs cluster" % 
                                                                     self.queue)
        logger.info("      --nb_core= : number of core to use in multicore.")
        logger.info("                        The default is all available core")
        logger.info("      Note that those options will be kept for the current session")      

    def help_survey(self):
        logger.info("syntax: survey [run_name] [--run_options])")
        logger.info("-- evaluate the different channel associate to the process")
        self.run_options_help()
        
    def help_refine(self):
        logger.info("syntax: refine require_precision [max_channel] [--run_options]")
        logger.info("-- refine the LAST run to achieve a given precision.")
        logger.info("   require_precision: can be either the targeted number of events")
        logger.info('                      or the required relative error')
        logger.info('   max_channel:[5] maximal number of channel per job')
        self.run_options_help()
        
    def help_combine_events(self):
        """ """
        logger.info("syntax: combine_events [GOAL] [--run_options]")
        logger.info("-- Combine the last run in order to write GOAL events")
        logger.info("   GOAL is taken in the run_card by default")
        self.run_options_help()
        
        
    def help_pythia(self):
        logger.info("syntax: pythia [RUN] [--run_options]")
        logger.info("-- run pythia on RUN (current one by default)")
        self.run_options_help()        
                
    def help_pythia(self):
        logger.info("syntax: pgs [RUN] [--run_options]")
        logger.info("-- run pgs on RUN (current one by default)")
        self.run_options_help() 
       
#===============================================================================
# CheckValidForCmd
#===============================================================================
class CheckValidForCmd(object):
    """ The Series of check routine for the MadEventCmd"""


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

        if args[0] not in self._set_options:
            self.help_set()
            raise self.InvalidCmd('Possible options for set are %s' % \
                                  self._set_options)
        
        if args[0] in ['stdout_level']:
            if args[1] not in ['DEBUG','INFO','WARNING','ERROR','CRITICAL']:
                raise self.InvalidCmd('output_level needs ' + \
                                      'a valid level')  
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
    
    def check_survey(self, args):
        """check that the argument for survey are valid"""
        
        if len(args) > 1:
            self.help_survey()
            raise self.InvalidCmd('Too many argument for survey command')
        elif not args:
            # No run name assigned -> assigned one automaticaly 
            self.run_name = self.find_available_run_name()
        else:
            self.run_name = args[0]
            args.pop(0)
            
        return True

    def check_refine(self, args):
        """check that the argument for survey are valid"""

        if len(args) > 2:
            self.help_refine()
            raise self.InvalidCmd('Too many argument for refine command')
        elif not args:
            self.help_refine()
            raise self.InvalidCmd('require_precision argument is require for refine cmd')
        else:
            try:
                [float(arg) for arg in args]
            except ValueError:                
                raise self.InvalidCmd('refine arguments are suppose to be number')
            
        return True
    
    def check_pythia(self, arg):
        """Check the argument for pythia command
        syntax: pythia [NAME] 
        Note that other option are already remove at this point
        """
               
     
        if len(arg) == 0 and not hasattr(self, 'run_name'):
            self.help_pythia()
            raise self.InvalidCmd('No run name currently define. Please add this information.')             
        
        if len(arg) == 1:
            self.run_name = arg[0]
            
            if  not os.path.exists(pjoin(self.me_dir,'Events','%s_unweighted_events.lhe.gz' % self.run_name)):
                raise self.InvalidCmd('No events file corresponding to %s run. '% self.run_name)
        
        
        input_file = pjoin(self.me_dir,'Events', '%s_unweighted_events.lhe.gz' % self.run_name)
        output_file = pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')
        os.system('gunzip -c %s > %s' % (input_file, output_file))
            
        if  not os.path.exists(pjoin(self.me_dir,'Events','%s_unweighted_events.lhe.gz' % self.run_name)):
            raise self.InvalidCmd('No events file corresponding to %s run. '% self.run_name)

        # If not pythia-pgs path
        if not self.configuration['pythia-pgs_path']:
            logger.info('Retry to read configuration file to find pythia-pgs path')
            self.set_configuration()
            
        if not self.configuration['pythia-pgs_path'] or not \
            os.path.exists(pjoin(self.configuration['pythia-pgs_path'],'src')):
            error_msg = 'No pythia-pgs path correctly set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)

    def check_pgs(self, arg):
        """Check the argument for pythia command
        syntax: pythia [NAME] 
        Note that other option are already remove at this point
        """
        
        # If not pythia-pgs path
        if not self.configuration['pythia-pgs_path']:
            logger.info('Retry to read configuration file to find pythia-pgs path')
            self.set_configuration()
      
        if not self.configuration['pythia-pgs_path'] or not \
            os.path.exists(pjoin(self.configuration['pythia-pgs_path'],'src')):
            error_msg = 'No pythia-pgs path correctly set.'
            error_msg += 'Please use the set command to define the path and retry.'
            error_msg += 'You can also define it in the configuration file.'
            raise self.InvalidCmd(error_msg)  
                  
        if len(arg) == 0 and not hasattr(self, 'run_name'):
            self.help_pgs()
            raise self.InvalidCmd('No run name currently define. Please add this information.')             
        
        if len(arg) == 1 and hasattr(self, 'run_name') and self.run_name == arg[0]:
            arg.pop(0)
        
        if not len(arg) and \
           not os.path.exists(pjoin(self.me_dir,'Events','pythia_events.hep')):
            self.help_pgs()
            raise self.InvalidCmd('''No file file pythia_events.hep currently available
            Please specify a valid run_name''')
                              
        if len(arg) == 1:
            self.run_name = arg[0]
            if  not os.path.exists(pjoin(self.me_dir,'Events','%s_pythia_events.hep.gz' % self.run_name)):
                raise self.InvalidCmd('No events file corresponding to %s run. '% self.run_name)
            else:
                input_file = pjoin(self.me_dir,'Events', '%s_pythia_events.hep.gz' % self.run_name)
                output_file = pjoin(self.me_dir, 'Events', 'pythia_events.hep')
                os.system('gunzip -c %s > %s' % (input_file, output_file))



    def check_import(self, args):
        """check the validity of line"""
         
        if not args or args[0] not in ['command']:
            self.help_import()
            raise self.InvalidCmd('wrong \"import\" format')
        
        if not len(args) == 2 or not os.path.exists(args[1]):
            raise self.InvalidCmd('PATH is mandatory for import command\n')
        

#===============================================================================
# CompleteForCmd
#===============================================================================
class CompleteForCmd(CheckValidForCmd):
    """ The Series of help routine for the MadGraphCmd"""
    
    def complete_history(self, text, line, begidx, endidx):
        "Complete the history command"

        args = self.split_arg(line[0:begidx])

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
            return self.list_completion(text, self._set_options)

        if len(args) == 2:
            if args[1] == 'stdout_level':
                return self.list_completion(text, ['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
    
    def complete_pythia(self,text, line, begidx, endidx):
        "Complete the pythia command"
        
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*_unweighted_events.lhe.gz'))
            data = [n.rsplit('/',1)[1][:-25] for n in data]
            return self.list_completion(text, data)
        
    def complete_pgs(self,text, line, begidx, endidx):
        "Complete the pythia command"
        
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            #return valid run_name
            data = glob.glob(pjoin(self.me_dir, 'Events', '*_pythia_events.hep.gz'))
            data = [n.rsplit('/',1)[1][:-21] for n in data]
            return self.list_completion(text, data)        
        
#===============================================================================
# MadEventCmd
#===============================================================================
class MadEventCmd(CmdExtended, HelpToCmd, CompleteForCmd):
    """The command line processor of MadGraph"""    

    # Options and formats available
    _set_options = ['stdout_level']
    # Variables to store object information
    true = ['T','.true.',True,'true', 1, '1']
    web = False
    prompt = 'MGME5>'
    cluster_mode = 0
    queue  = 'madgraph'
    nb_core = None
    
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

                
        self._options = {}        
        self.to_store = []
        # Load the configuration file
        self.set_configuration()

        if self.web:
            os.system('touch Online')
        
        self.configured = 0 # time for reading the card

    ############################################################################    
    def split_arg(self, line):
        """split argument and remove run_options"""
        
        args = CmdExtended.split_arg(line)
        
        for arg in args:
            if not arg.startswith('--'):
                continue
            if arg.startswith('--cluster='):
                self.cluster_mode = int(arg.split('=',1)[1])
            elif arg.startswith('--queue='):
                self.queue = arg.split('=',1)[1].strip()
            elif arg.startswith('--nb_core'):
                self.nb_core = int(arg.split('=',1)[1])
            else:
                continue
            args.remove(arg)
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
            
        self.configuration = {'pythia8_path': './pythia8',
                              'pythia-pgs_path': '../pythia-pgs',
                              'delphes_path': '../Delphes',
                              'madanalysis_path': '../MadAnalysis',
                              'exrootanalysis_path': '../ExRootAnalysis',
                              'td_path': '../',
                              'web_browser':None,
                              'eps_viewer':None,
                              'text_editor':None}
        
        if not config_path:
            try:
                config_file = open(os.path.join(os.environ['HOME'],'.mg5','mg5_config'))
            except:
                if self.me_dir:
                    config_file = open(os.path.relpath(
                          os.path.join(self.dirbin, 'me5_configuration.txt')))
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
                self.configuration[name] = value
                if value.lower() == "none":
                    self.configuration[name] = None

        # Treat each expected input
        # delphes/pythia/... path
        for key in self.configuration:
            if key.endswith('path'):
                path = os.path.join(self.me_dir, self.configuration[key])
                if os.path.isdir(path):
                    self.configuration[key] = path
                    continue
                if os.path.isdir(self.configuration[key]):
                    continue
                else:
                    self.configuration[key] = ''
            elif key not in ['text_editor','eps_viewer','web_browser']:
                # Default: try to set parameter
                try:
                    self.do_set("%s %s" % (key, self.configuration[key]))
                except self.InvalidCmd:
                    logger.warning("Option %s from config file not understood" \
                                   % key)
        
        # Configure the way to open a file:
        misc.open_file.configure(self.configuration)
          
        return self.configuration
  
    ############################################################################
    def do_import(self, line):
        """Import files with external formats"""

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
 
 
    ############################################################################
    def update_status(self, status):
        """ update the index status """
        logger.info(status)
        os.system('echo \"%s \" > %s' % (status, self.status))
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.run_name))
        os.system('%s/gen_cardhtml-pl' % (self.dirbin))
        
        
    ############################################################################                       
#    def do_survey_up(self):
#        """ make the survey """
#        
#        os.system('touch %s' % pjoin(self.me_dir, 'survey'))
#        self.update_status("Starting jobs")
#        
#        if self.cluster_mode == 0:
#            args = "0 %s" % self.name
#        elif self.cluster_mode == 1:
#            args = "1 %s %s " % (self.cluster_queue, self.name) 
#        elif self.cluster_mode == 2:
#            args = "2 %s %s " % (self.nb_core, self.name) 
#        
#        os.system('%s/survey %s ' % (self.bin, args))
#
#        if os.path.exists(self.error):
#            logger.error(open(self.error).read())
#            logger.info(time.ctime())
#            os.remove(pjoin(self.me_dir, 'survey'))
#            os.remove(pjoin(self.me_dir, 'RunWeb'))
#            os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.name))
#            os.system('%s/gen_cardhtml-pl' % (self.dirbin))
#        
#        os.remove(pjoin(self.me_dir, 'survey')) 


     
    ############################################################################      
    def do_survey(self, line):
        """ launch survey for the current process """
        
        
        args = self.split_arg(line)
        # Check argument's validity
        self.check_survey(args)
        self.update_status('compile directory')
        # initialize / remove lhapdf mode
        self.configure_directory()

        if self.cluster_mode:
            logger.info('Creating Jobs')

        # treat random number
        self.update_random()
        self.save_random()

        logger.info('Working on SubProcesses')
        for subdir in open(pjoin(self.me_dir, 'SubProcesses', 'subproc.mg')):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if match[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(pjoin(Pdir, match))
            
            #compile gensym
            misc.compile(['gensym'], cwd=Pdir)
            if not os.path.exists(pjoin(Pdir, 'gensym')):
                raise MadEventError, 'Error make gensym not successful'

            # Launch gensym
            subprocess.call(['./gensym'], cwd=Pdir)
            if not os.path.exists(pjoin(Pdir, 'ajob1')):
                raise MadEventError, 'Error gensym run not successful'

            #
            os.system("chmod +x %s/ajob*" % Pdir)
        
            out = subprocess.call(['make','madevent'], cwd=Pdir)
            if out:
                raise MadEventError, 'Error make madevent not successful'

            for job in glob.glob(pjoin(Pdir,'ajob*')):
                job = os.path.basename(job)
                os.system('touch %s/wait.%s' %(Pdir,job))
                self.launch_job(job, cwd=Pdir,stdout=devnull)
        self.monitor()

    ############################################################################      
    def do_refine(self, line):
        """ launch survey for the current process """

        if not hasattr(self, 'run_name'):
            self.run_name = 'last'

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

        logger.info("Using random number seed offset = %s" % self.random)

        for subdir in open(pjoin(self.me_dir, 'SubProcesses', 'subproc.mg')):
            subdir = subdir.strip()
            Pdir = pjoin(self.me_dir, 'SubProcesses',subdir)
            bindir = pjoin(os.path.relpath(self.dirbin, Pdir))
                           
            logger.info('    %s ' % subdir)
            # clean previous run
            for match in glob.glob(pjoin(Pdir, '*ajob*')):
                if match[:4] in ['ajob', 'wait', 'run.', 'done']:
                    os.remove(pjoin(Pdir, match))

            out = None #open(pjoin(Pdir, 'gen_ximprove.log'),'w')
            proc = subprocess.Popen([pjoin(bindir, 'gen_ximprove')],
                                    stdout=out,stderr=subprocess.STDOUT,
                                    stdin=subprocess.PIPE,
                                    cwd=Pdir)
            proc.communicate('%s %s T\n' % (precision, max_process))
            proc.wait()
            if os.path.exists(pjoin(Pdir, 'ajob1')):
                out = subprocess.call(['make','madevent'], cwd=Pdir)
                if out:
                    raise MadEventError, 'Error make madevent not successful'

                #
                os.system("chmod +x %s/ajob*" % Pdir)            
                for job in glob.glob(pjoin(Pdir,'ajob*')):
                    job = os.path.basename(job)
                    os.system('touch %s/wait.%s' %(Pdir, job))
                    self.launch_job(job, cwd=Pdir,stdout=devnull)
                    
        self.monitor()
        
        self.update_status("Combining runs")
        try:
            os.remove(pjoin(Pdir, 'combine_runs.log'))
        except:
            pass
        
        bindir = pjoin(os.path.relpath(self.dirbin, pjoin(self.me_dir,'SubProcesses')))
        subprocess.call([pjoin(bindir, 'combine_runs')], 
                                          cwd=pjoin(self.me_dir,'SubProcesses'))
        
        subprocess.call([pjoin(self.dirbin, 'sumall')], 
                                         cwd=pjoin(self.me_dir,'SubProcesses'))
        subprocess.call([pjoin(self.dirbin, 'gen_crossxhtml-pl'), self.run_name], 
                                                                cwd=self.me_dir)     
 
    ############################################################################ 
    def do_combine_events(self, line):
        """Launch combine events"""

        if not hasattr(self, 'run_name'):
            self.run_name = 'last'

        self.update_status('Combining Events')
        subprocess.call(['%s/run_combine' % self.dirbin, str(self.cluster_mode)],
                            cwd=pjoin(self.me_dir, 'SubProcesses'))

        shutil.move(pjoin(self.me_dir, 'SubProcesses', 'events.lhe'),
                    pjoin(self.me_dir, 'Events', 'events.lhe'))
        shutil.move(pjoin(self.me_dir, 'SubProcesses', 'unweighted_events.lhe'),
                    pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')) 
        
        subprocess.call(['%s/put_banner' % self.dirbin, 'events.lhe'],
                            cwd=pjoin(self.me_dir, 'Events'))
        subprocess.call(['%s/put_banner'% self.dirbin, 'unweighted_events.lhe'],
                            cwd=pjoin(self.me_dir, 'Events'))
        
        if os.path.exists(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')):
            subprocess.call(['%s/extract_banner-pl' % self.dirbin, 
                             'unweighted_events.lhe', 'banner.txt'],
                            cwd=pjoin(self.me_dir, 'Events'))
        
        eradir = self.configuration['exrootanalysis_path']
        madir = self.configuration['madanalysis_path']
        td = self.configuration['td_path']
        if misc.is_executable(pjoin(eradir,'ExRootLHEFConverter'))  and\
           os.path.exists(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')):
                self.create_root_file()
        
        if madir and td and \
            os.path.exists(pjoin(self.me_dir, 'Events', 'unweighted_events.lhe')) and \
            os.path.exists(pjoin(self.me_dir, 'Cards', 'plot_card.dat')):
                self.create_plot()
        
        #
        # STORE
        #
        self.update_status('Storing parton level results')
        subprocess.call(['%s/store' % self.dirbin, self.run_name],
                            cwd=pjoin(self.me_dir, 'Events'))
        os.system('%s/gen_crossxhtml-pl %s' % (self.dirbin, self.run_name))
        shutil.copy(pjoin(self.me_dir, 'Events', self.run_name+'_banner.txt'),
                    pjoin(self.me_dir, 'Events', 'banner.txt')) 


        
        



    ############################################################################      
    def do_pythia(self, line):
        """launch pythia"""
        
        args = self.split_arg(line)
        # Check argument's validity
        self.check_pythia(args) 
        # initialize / remove lhapdf mode
        self.configure_directory()
        
        # Check that the pythia_card exists. If not copy the default and
        # ask for edition of the card.
        if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pythia_card.dat')):
            files.cp(pjoin(self.me_dir, 'Cards', 'pythia_card_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'pythia_card.dat'))
            
            logger.info('No pythia card found. Take the default one.')
            answer = self.ask('Do you want to edit this card?','n', ['y','n'],
                              timeout=20)
            if answer == 'y':
                misc.open_file(pjoin(self.me_dir, 'Cards', 'pythia_card.dat'))
        
        pythia_src = pjoin(self.configuration['pythia-pgs_path'],'src')
        
        logger.info('Launching pythia')

        subprocess.call(['../bin/internal/run_pythia', 
                         pythia_src,
                         str(self.cluster_mode)],
                         cwd=pjoin(self.me_dir,'Events'))


        if not os.path.exists(pjoin(self.me_dir,'Events','pythia_events.hep')):
            logger.warning('Fail to produce pythia output')
            return
        
        self.to_store.append('pythia')


        
        pydir = pjoin(self.configuration['pythia-pgs_path'], 'src')
        eradir = self.configuration['exrootanalysis_path']
        madir = self.configuration['madanalysis_path']
        td = self.configuration['td_path']
        
        # Update the banner with the pythia card
        banner = open(pjoin(self.me_dir,'Events','banner.txt'),'a')
        banner.writelines('<MGPythiaCard>')
        banner.writelines(open(pjoin(self.me_dir, 'Cards','pythia_card.dat')).read())
        banner.writelines('</MGPythiaCard>')
        banner.close()
        
        # Creating LHE file
        print pjoin(pydir, 'hep2lhe')
        if misc.is_executable(pjoin(pydir, 'hep2lhe')):
            self.update_status('Creating Pythia LHE File')
            subprocess.call([self.dirbin+'/run_hep2lhe', pydir, 
                             str(self.cluster_mode)],
                             cwd=pjoin(self.me_dir,'Events'))       

        # Creating ROOT file
        if misc.is_executable(pjoin(eradir, 'ExRootLHEFConverter')):
            self.update_status('Creating Pythia LHE Root File')
            subprocess.call([eradir+'/ExRootLHEFConverter', 
                             'pythia_events.lhe', 'pythia_lhe_events.root'],
                            cwd=pjoin(self.me_dir,'Events')) 

        if int(self.run_card['ickkw']):
            self.update_status('Create matching plots for Pythia')
            subprocess.call([self.dirbin+'/create_matching_plots.sh', self.run_name],
                            cwd=pjoin(self.me_dir,'Events')) 

        # Plot for pythia
        if misc.is_executable(pjoin(madir, 'plot_events')) and td:
            self.update_status('Creating Plots for Pythia')
            self.create_plot('Pythia')
    
    def store_result(self):
        """ tar the pythia results. This is done when we are quite sure that 
        the pythia output will not be use anymore """
        
        if not self.to_store or not hasattr(self, 'run_name'):
            return
        
        if 'pythia' in self.to_store:
            self.update_status('Storing Pythia files of Previous run')
            os.system('mv -f %(path)s/pythia_events.hep %(path)s/%(name)s_pythia_events.hep' % 
                  {'name': self.run_name, 'path' : pjoin(self.me_dir,'Events')})
            os.sytem('gzip -f %s_pythia_events.hep' % self.run_name)
            self.to_store.remove('pythia')
            
        
    ############################################################################      
    def do_pgs(self, line):
        """launch pgs"""
        
        args = self.split_arg(line)
        # Check argument's validity
        self.check_pgs(args) 
        
        pgsdir = pjoin(self.configuration['pythia-pgs_path'], 'src')
        eradir = self.configuration['exrootanalysis_path']
        madir = self.configuration['madanalysis_path']
        td = self.configuration['td_path']
        
        # Compile pgs if not there   
        print pgsdir     
        if not misc.is_executable(pjoin(pgsdir, 'pgs')):
            logger.info('No PGS executable -- running make')
            subprocess.call(['make'], cwd=pgsdir) 

        if not misc.is_executable(pjoin(pgsdir, 'pgs')):
            logger.error('Fail to compile PGS')
            return
        
        # Check that the pgs_card exists. If not copy the default and
        # ask for edition of the card.
        if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pgs_card.dat')):
            files.cp(pjoin(self.me_dir, 'Cards', 'pgs_card_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'pgs_card.dat'))
            
            logger.info('No pgs card found. Take the default one.')
            answer = self.ask('Do you want to edit this card?','n', ['y','n'],
                              timeout=20)
            if answer == 'y':
                misc.open_file(pjoin(self.me_dir, 'Cards', 'pgs_card.dat'))
        
        
        
        self.update_status('Running PGS')
        # now pass the event to a detector simulator and reconstruct objects
        subprocess.call([self.dirbin+'/run_pgs', pgsdir, str(self.cluster_mode)],
                            cwd=pjoin(self.me_dir, 'Events')) 

        if not os.path.exists(pjoin(self.me_dir, 'Events', 'pgs_events.lhco')):
            logger.error('Fail to create LHCO events')
            return 
        
        # Creating Root file
        if misc.is_executable(pjoin(eradir, 'ExRootLHCOlympicsConverter')):
            self.update_status('Creating PGS Root File')
            subprocess.call([eradir+'/ExRootLHCOlympicsConverter', 
                             'pgs_events.lhco','pgs_events.root'],
                            cwd=pjoin(self.me_dir, 'Events')) 

        
        # Creating plots
        if misc.is_executable(pjoin(madir, 'plot_events')) and td:
            self.update_status('Creating Plots for PGS')
            self.create_plot('PGS')

       
    def launch_job(self,exe, cwd=None, stdout=None, **opt):
        """ """
        
        if self.cluster_mode == 0:
            start = time.time()
            subprocess.call(['./'+exe], cwd=cwd, stdout=stdout, **opt)
            logger.info('run in %f s' % (time.time() -start))
            #print 'sum_html'
            #subprocess.call(['./sum_html'], cwd=self.dirbin)
        elif self.cluster_mode == 1:
            os.system("qsub -N %s %s >> %s" % (self.queue, exe, stdout))
        elif self.cluster_mode == 2:
            os.system("%s/multicore %s %s" % (self.dirbin, self.nb_core, pjoin()))
            time.sleep(1)
            
    
    def monitor(self):
        """ monitor the progress of running job """
        
        if self.cluster_mode:
            subprocess.call([pjoin(self.dirbin, 'monitor'), self.run_name], 
                                                                cwd=self.me_dir)
        subprocess.call([pjoin(self.dirbin, 'sumall')], 
                                          cwd=pjoin(self.me_dir,'SubProcesses'))   
        subprocess.call([pjoin(self.dirbin, 'gen_crossxhtml-pl'), self.run_name], 
                                          cwd=pjoin(self.me_dir,'SubProcesses'))
        

    def find_available_run_name(self):
        """ find a valid run_name for the current job """
        
        name = 'run_%03d'
        data = [int(s[4:7]) for s in os.listdir(pjoin(self.me_dir,'Events')) if
                        s.startswith('run_') and len(s)>6 and s[4:7].isdigit()]
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

        # Change current working directory
        self.launching_dir = os.getcwd()
        os.chdir(self.me_dir)
        
        # Check if we need the MSSM special treatment
        model = self.find_model_name()
        if model == 'mssm' or model.startswith('mssm-'):
            param_card = pjoin(self.me_dir, 'Cards','param_card.dat')
            mg5_param = pjoin(self.me_dir, 'Source', 'MODEL', 'MG5_param.dat')
            check_param_card.convert_to_mg5card(param_card, mg5_param)
            check_param_card.check_valid_param_card(mg5_param)

        # Read run_card
        run_card = pjoin(self.me_dir, 'Cards','run_card.dat')
        self.run_card = self.read_run_card(run_card)
        
        # limit the number of event to 100k
        self.check_nb_events()

        # set environment variable for lhapdf.
        if self.run_card['pdlabel'] == "'lhapdf'":
            os.environ['lhapdf'] = True
        elif 'lhapdf' in os.environ.keys():
            del os.environ['lhapdf']
        
        # Compile
        out = subprocess.call([pjoin(self.dirbin, 'compile_Source')],
                              cwd = self.me_dir)
        if out:
            raise MadEventError, 'Impossible to compile'
        
        # set random number
        if os.path.exists(pjoin(self.me_dir,'SubProcesses','randinit')):
            for line in open(pjoin(self.me_dir,'SubProcesses','randinit')):
                data = line.split('=')
                assert len(data) ==2
                self.random = int(data[1])
                break
        else:
            self.random = random.randint(1, 30107) # 30107 maximal allow 
                                                   # random number for ME
                                                   
        if self.run_card['ickkw'] == 2:
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
            output[line[1].strip()] = line[0].strip()
        return output

    ############################################################################
    @staticmethod
    def check_dir(path, default=''):
        """check if the directory exists. if so return the path otherwise the default"""
         
        if os.path.isdir(path):
            return path
        else:
            return default

    ############################################################################
    def find_model_name(self):
        """ return the model name """
        if hasattr(self, 'model_name'):
            return self.model_name
        
        model = 'sm'
        for line in file(os.path.join(self.me_dir,'Cards','proc_card_mg5.dat'),'r'):
            line = line.split('#')[0]
            line = line.split('=')[0]
            if line.startswith('import') and 'model' in line:
                model = line.split()[2]       
       
        self.model = model
        return model
    
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
        self.random = self.random % 30107 # cann't use too big random number


    ############################################################################
    def save_random(self):
        """save random number in appropirate file"""
        
        fsock = open(pjoin(self.me_dir, 'SubProcesses','randinit'),'w')
        fsock.writelines('r=%s\n' % self.random)

    def do_quit(self, line):
        """ """
        self.store_result()
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
                logger.info(error_msg)
                if self.cluster_mode not in [0, 1]:
                    msg = 'No sudakov grid file for parameter choice and not possible to create it automaticaly for the cluster choice'
                    logger.error(msg)
                    os.system("echo %s > %s" % (msg, self.error))
                    shutil.copy(self.error, self.status)             
                    os.remove(pjoin(self.me_dir, 'RunWeb'))
                    os.system('%s/gen_cardhtml-pl' % self.dirbin)    
                    return
                self.update_status(msg)
                os.system('%s/run_genissud %s' % (self.dirbin, self.cluster_mode))


    ############################################################################
    def create_root_file(self):
        """create the LHE root file """
        self.update_status('Creating root files')

        eradir = self.configuration['exrootanalysis_path']
        subprocess.call(['%s/ExRootLHEFConverter' % eradir, 
                             'unweighted_events.lhe', 'unweighted_events.root'],
                            cwd=pjoin(self.me_dir, 'Events'))
        
    ############################################################################
    def create_plot(self, mode='parton', event_path=None):
        """create the plot""" 

        if not event_path:
            if mode == 'parton':
                event_path = pjoin(self.me_dir, 'Events','unweighted_events.lhe')
            elif mode == 'Pythia':
                event_path = pjoin(self.me_dir, 'Events','pythia_events.lhe')
            elif mode == 'PGS':
                event_path = pjoin(self.me_dir, 'Events', 'pgs_events.lhco')
                
        plot_dir = pjoin(self.me_dir, 'Events', self.run_name)
        if mode == 'Pythia':
            plot_dir += '_pythia'
        elif mode == 'PGS':
            plot_dir += '_pgs'
        
        self.update_status("Creating Plots")
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir) 
        
        madir = self.configuration['madanalysis_path']
        td = self.configuration['td_path']


        files.ln(pjoin(self.me_dir, 'Cards','plot_card.dat'), plot_dir, 'ma_card.dat')
        proc = subprocess.Popen(['%s/plot_events' % madir],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'w'),
                            stderr = subprocess.STDOUT,
                            stdin=subprocess.PIPE,
                            cwd=plot_dir)
        proc.communicate('%s\n' % event_path)
        proc.wait()
        subprocess.call(['%s/plot' % self.dirbin, madir, td],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'a'),
                            stderr = subprocess.STDOUT,
                            cwd=plot_dir)

    
        subprocess.call(['%s/plot_page-pl' % self.dirbin, 
                                os.path.basename(plot_dir),
                                mode],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = subprocess.STDOUT,
                            cwd=pjoin(self.me_dir, 'Events'))
       
        shutil.move(pjoin(self.me_dir, 'Events', 'plots.html'),
                   pjoin(self.me_dir, 'Events', '%s_plots.html' % self.run_name))       
        
        
        
        
        
        
        
        
        
        
  
    
    