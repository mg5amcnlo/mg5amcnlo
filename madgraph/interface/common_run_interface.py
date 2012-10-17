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
logger = logging.getLogger('madgraph.stdout') # -> stdout
logger_stderr = logging.getLogger('madgraph.stderr') # ->stderr


try:
    # import from madgraph directory
    import madgraph.various.banner as banner_mod
    import madgraph.various.misc as misc
    import madgraph.iolibs.files as files
    import madgraph.various.cluster as cluster
    import models.check_param_card as check_param_card
    MADEVENT=False    
except Exception, error:
    if __debug__:
        print error
    # import from madevent directory
    import internal.banner as banner_mod
    import internal.misc as misc    
    import internal.cluster as cluster
    import internal.check_param_card as check_param_card
    import internal.files as files
    MADEVENT=True
#try:
#    os.sys.path.append("../MadSpin")
#    import decay
#except: 
#    pass

#===============================================================================
# HelpToCmd
#===============================================================================
class HelpToCmd(object):
    """ The Series of help routins in common between amcatnlo_run and 
    madevent interface"""

    def help_treatcards(self):
        logger.info("syntax: treatcards [param|run] [--output_dir=] [--param_card=] [--run_card=]")
        logger.info("-- create the .inc files containing the cards information." )
 
    def help_set(self):
        logger.info("syntax: set %s argument" % "|".join(self._set_options))
        logger.info("-- set options")
        logger.info("   stdout_level DEBUG|INFO|WARNING|ERROR|CRITICAL")
        logger.info("     change the default level for printed information")
        logger.info("   timeout VALUE")
        logger.info("      (default 20) Seconds allowed to answer questions.")
        logger.info("      Note that pressing tab always stops the timer.")        
        logger.info("   cluster_temp_path PATH")
        logger.info("      (default None) Allow to perform the run in PATH directory")
        logger.info("      This allow to not run on the central disk. This is not used")
        logger.info("      by condor cluster (since condor has it's own way to prevent it).")


class CheckValidForCmd(object):
    """ The Series of check routines in common between amcatnlo_run and 
    madevent interface"""

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
                files.cp(path + '/Cards/' + name, path + '/Cards/'+ args[0])
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


#===============================================================================
# CommonRunCmd
#===============================================================================
class CommonRunCmd(HelpToCmd, CheckValidForCmd):

    debug_output = 'ME5_debug'

    def do_treatcards_nlo(self,line):
        """this is for creating the correct run_card.inc from the nlo format run_card"""
        return self.do_treatcards(line, amcatnlo=True)


    ############################################################################      
    def do_treatcards(self, line, amcatnlo=False):
        """Advanced commands: create .inc files from param_card.dat/run_card.dat"""

        args = self.split_arg(line)
        mode,  opt  = self.check_treatcards(args)

        if mode in ['run', 'all']:
            if not hasattr(self, 'run_card'):
                if amcatnlo:
                    run_card = banner_mod.RunCardNLO(opt['run_card'])
                else:
                    run_card = banner_mod.RunCard(opt['run_card'])
            else:
                run_card = self.run_card
            run_card.write_include_file(pjoin(opt['output_dir'],'run_card.inc'))
        
        if mode in ['param', 'all']: 
            if os.path.exists(pjoin(self.me_dir, 'Source', 'MODEL', 'mp_coupl.inc')):
                param_card = check_param_card.ParamCardMP(opt['param_card'])
            else:
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
                return
            else:
                subprocess.call(['python', 'write_param_card.py'], 
                             cwd=pjoin(self.me_dir,'bin','internal','ufomodel'))
                default = pjoin(self.me_dir,'bin','internal','ufomodel','param_card.dat')
            param_card.write_inc_file(outfile, ident_card, default)

  
    ############################################################################ 
    def do_open(self, line):
        """Open a text file/ eps file / html file"""
        
        args = self.split_arg(line)
        # Check Argument validity and modify argument to be the real path
        self.check_open(args)
        file_path = args[0]
        
        misc.open_file(file_path)

    ############################################################################
    def do_set(self, line, log=True):
        """Set an option, which will be default for coming generations/outputs
        """
        # cmd calls automaticaly post_set after this command.


        args = self.split_arg(line) 
        # Check the validity of the arguments
        self.check_set(args)
        # Check if we need to save this in the option file
        if args[0] in self.options_configuration and '--no_save' not in args:
            self.do_save('options --auto')
        
        if args[0] == "stdout_level":
            if args[1].isdigit():
                logging.root.setLevel(int(args[1]))
                logging.getLogger('madgraph').setLevel(int(args[1]))
            else:
                logging.root.setLevel(eval('logging.' + args[1]))
                logging.getLogger('madgraph').setLevel(eval('logging.' + args[1]))
            if log: logger.info('set output information to level: %s' % args[1])
        elif args[0] == "fortran_compiler":
            self.options['fortran_compiler'] = args[1]
            current = misc.detect_current_compiler(pjoin(self.me_dir,'Source','make_opts'))
            if current != args[1] and args[1] != 'None':
                misc.mod_compilator(self.me_dir, args[1], current)
        elif args[0] == "run_mode":
            if not args[1] in [0,1,2,'0','1','2']:
                raise self.InvalidCmd, 'run_mode should be 0, 1 or 2.'
            self.cluster_mode = int(args[1])
            self.options['run_mode'] =  self.cluster_mode
        elif args[0] in  ['cluster_type', 'cluster_queue', 'cluster_temp_path']:
            self.options[args[0]] = args[1]
            opt = self.options
            self.cluster = cluster.from_name[opt['cluster_type']](\
                                 opt['cluster_queue'], opt['cluster_temp_path'])
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


    def add_error_log_in_html(self, errortype=None):
        """If a ME run is currently running add a link in the html output"""

        # Be very carefull to not raise any error here (the traceback 
        #will be modify in that case.)
        if hasattr(self, 'results') and hasattr(self.results, 'current') and\
                self.results.current and 'run_name' in self.results.current and \
                hasattr(self, 'me_dir'):
            name = self.results.current['run_name']
            tag = self.results.current['tag']
            self.debug_output = pjoin(self.me_dir, '%s_%s_debug.log' % (name,tag))
            if errortype:
                self.results.current.debug = errortype
            else:
                self.results.current.debug = self.debug_output
            
        else:
            #Force class default
            self.debug_output = CommonRunCmd.debug_output
        if os.path.exists('ME5_debug') and not 'ME5_debug' in self.debug_output:
            os.remove('ME5_debug')
        if not 'ME5_debug' in self.debug_output:
            os.system('ln -s %s ME5_debug &> /dev/null' % self.debug_output)


  

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
            logger.info(' Idle: %s,  Running: %s,  Completed: %s' % status[:3])
        
        self.last_update = time
        self.results.update(status, level, makehtml=makehtml, error=error)
        

    ############################################################################
    def set_configuration(self, config_path=None, final=True, initdir=None, amcatnlo=False):
        """ assign all configuration variable from file 
            ./Cards/mg5_configuration.txt. assign to default if not define """

        if not hasattr(self, 'options') or not self.options:  
            self.options = dict(self.options_configuration)
            self.options.update(self.options_madgraph)
            self.options.update(self.options_madevent) 
        if not config_path:
            if os.environ.has_key('MADGRAPH_BASE'):
                config_path = pjoin(os.environ['MADGRAPH_BASE'],'mg5_configuration.txt')
                self.set_configuration(config_path, final)
                return
            if 'HOME' in os.environ:
                config_path = pjoin(os.environ['HOME'],'.mg5', 
                                                        'mg5_configuration.txt')
                if os.path.exists(config_path):
                    self.set_configuration(config_path, final=False)
            if amcatnlo:
                me5_config = pjoin(self.me_dir, 'Cards', 'amcatnlo_configuration.txt')
            else:
                me5_config = pjoin(self.me_dir, 'Cards', 'me5_configuration.txt')
            self.set_configuration(me5_config, final=False, initdir=self.me_dir)
                
            if self.options.has_key('mg5_path'):
                MG5DIR = self.options['mg5_path']
                config_file = pjoin(MG5DIR, 'input', 'mg5_configuration.txt')
                self.set_configuration(config_file, final=False,initdir=MG5DIR)
            return self.set_configuration(me5_config, final,initdir=self.me_dir)

        config_file = open(config_path)

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
                if name.endswith('_path'):
                    path = value
                    if os.path.isdir(path):
                        self.options[name] = os.path.realpath(path)
                        continue
                    if not initdir:
                        continue
                    path = pjoin(initdir, value)
                    if os.path.isdir(path):
                        self.options[name] = os.path.realpath(path)
                        continue
                else:
                    self.options[name] = value
                    if value.lower() == "none":
                        self.options[name] = None

        if not final:
            return self.options # the return is usefull for unittest


        # Treat each expected input
        # delphes/pythia/... path
        for key in self.options:
            # Final cross check for the path
            if key.endswith('path'):
                path = self.options[key]
                if path is None:
                    continue
                if os.path.isdir(path):
                    self.options[key] = os.path.realpath(path)
                    continue
                path = pjoin(self.me_dir, self.options[key])
                if os.path.isdir(path):
                    self.options[key] = os.path.realpath(path)
                    continue
                elif self.options.has_key('mg5_path') and self.options['mg5_path']: 
                    path = pjoin(self.options['mg5_path'], self.options[key])
                    if os.path.isdir(path):
                        self.options[key] = os.path.realpath(path)
                        continue
                self.options[key] = None
            elif key.startswith('cluster'):
                pass              
            elif key == 'automatic_html_opening':
                if self.options[key] in ['False', 'True']:
                    self.options[key] =eval(self.options[key])
            elif key not in ['text_editor','eps_viewer','web_browser','stdout_level']:
                # Default: try to set parameter
                try:
                    self.do_set("%s %s --no_save" % (key, self.options[key]), log=False)
                except self.InvalidCmd:
                    logger.warning("Option %s from config file not understood" \
                                   % key)
        
        # Configure the way to open a file:
        misc.open_file.configure(self.options)
          
        return self.options

    @staticmethod
    def find_available_run_name(me_dir):
        """ find a valid run_name for the current job """
        
        name = 'run_%02d'
        data = [int(s[4:6]) for s in os.listdir(pjoin(me_dir,'Events')) if
                        s.startswith('run_') and len(s)>5 and s[4:6].isdigit()]
        return name % (max(data+[0])+1) 
    

   ############################################################################      
    def do_decay_events(self,line):
        """Require MG5 directory: decay events with spin correlations
        """
        
        # First need to load MadSpin
        
        # Check that MG5 directory is present .
        if MADEVENT and not self.options['mg5_path']:
            raise self.InvalidCmd, '''The module decay_events requires that MG5 is installed on the system.
            You can install it and set its path in ./Cards/me5_configuration.txt'''
        elif MADEVENT:
            sys.path.append(self.options['mg5_path'])
        try:
            import MadSpin.decay as decay
        except:
            raise self.ConfigurationError, '''Can\'t load MadSpin
            The variable mg5_path might not be correctly configured.'''
        
        
        logger.info("This functionality allows for the decay of  resonances")
        logger.info("in a .lhe file, keeping track of the spin correlation effets.")
        logger.info("BE AWARE OF THE CURRENT LIMITATIONS:")
        logger.info("  (1) exclusive decay channel only, ")
        logger.info("      YOU CANNOT USE MULTI-PARTICLE TAG")
        logger.info("  (2) the overall branching fraction must")
        logger.info("      specified by the user (otherwise the")
        logger.info("      is not rescaled)")
        logger.info("  (3) only decay channels with splitting of the type")
        logger.info("      A -> B(stable) + C(unstable)")
        logger.info("      have been tested so far")
        logger.info("  (4) when entering the decay chain structure,")
        logger.info("      please note that the reading module is ")
        logger.info("      CASE SENSITIVE")

        # load the name of the event file
        args = self.split_arg(line) 
        self.check_decay_events(args)

        evt_file=pjoin(self.me_dir,'Events',self.run_name, 'unweighted_events.lhe.gz')

        
        try:
            misc.call(['gunzip %s' % evt_file], shell=True)
            evt_file=evt_file.replace(".gz","")
        except:
            pass
        
        print evt_file
        try:
            inputfile = open(evt_file, 'r')       
        except:
            print "cannot open the file"
        
         #    load the tools
        decay_tools=decay.decay_misc()
        
#
# Read the banner from the input event file
#
        logger.info("Extracting the banner ...")
        mybanner=decay.Banner(inputfile)
        mybanner.ReadBannerFromFile()
        mybanner.proc["generate"], proc_option=decay_tools.format_proc_line(\
                                        mybanner.proc["generate"])
        logger.info("process: "+mybanner.proc["generate"])
        logger.info("options: "+proc_option)

        if not mybanner.proc.has_key('model'):
            logger.info('No model found in the banner, set the model to sm')
            mybanner.proc['model']='sm'
# Read the final state of the production process:
#     "_full" means with the complete decay chain syntax 
#     "_compact" means without the decay chain syntax 
        final_state_full=\
             mybanner.proc["generate"][mybanner.proc["generate"].find(">")+1:]
        final_state_compact, prod_branches=\
             decay_tools.get_final_state_compact(final_state_full)

        decay_processes={}


        logger.info('Please enter the definition of each branch  ')
        logger.info('associated with the decay channel you want to consider')
        logger.info('Please use the mg5 syntax, e.g. ')
        logger.info(' A > B C , ( C > D E , E > F G )')
        list_branches={}
        while 1: 
            decaybranch=self.ask('New branch: (if you are done, type enter) \n','done')
            if decaybranch == 'done' or decaybranch == '':
                break
            decay_process, init_part=\
                decay_tools.reorder_branch(decaybranch)
            list_branches[init_part]=decay_process
            del decay_process, init_part    
        
        if len(list_branches)==0:
            logger.info("Nothing to decay ...")
            return
# Ask the user which particle should be decayed        
        particle_index=2
        to_decay={}
        for particle in final_state_compact.split():
            particle_index+=1
            if list_branches.has_key(str(particle)):
                to_decay[particle_index]=particle
                decay_processes[particle_index]=list_branches[str(particle)]

        if len(to_decay)==0:
            logger.info("Nothing to decay ...")
            return
            
        logger.info("An estimation of the maximum weight is needed for the unweighting ")
        answer=self.ask("Enter the maximum weight, or enter a negative number if unknown \n",-1.0)
        max_weight=-1
        try:
            answer=answer.replace("\n","") 
            if float(answer)>0:
                max_weight=float(answer)
            else:
                max_weight=-1
                logger.info("The maximum weight will be evaluated")
        except: 
            pass
     
        BW_effects=1

# by default set the branching fraction to 1
        branching_fraction=1.0
        
        answer=self.ask( "Branching fraction ? (type a negative number is unknown ) \n",-1.0)

        try:
            if float(answer) >0.0: branching_fraction=float(answer)
        except: 
            pass

        current_dir=os.getcwd()
        os.chdir("../MadSpin")
        generate_all=decay.decay_all_events(inputfile,mybanner,to_decay,decay_processes,\
                 prod_branches, proc_option, max_weight, BW_effects,branching_fraction)
        os.chdir(current_dir)
        
        misc.call(['gzip %s' % evt_file], shell=True)
        decayed_evt_file=evt_file.replace('.lhe', '_decayed.lhe')
        shutil.move('../MadSpin/decayed_events.lhe', decayed_evt_file)
        misc.call(['gzip %s' % decayed_evt_file], shell=True)
        logger.info("Decayed events have been written in %s" % decayed_evt_file)
        
        
    def check_decay_events(self,args):
        """Check the argument for decay_events command
        syntax: decay_events [NAME]
        Note that other option are already remove at this point
        """
        
        if len(args) == 0:
            if self.run_name:
                args.insert(0, self.results.lastrun)
            elif self.results.lastrun:
                args.insert(0, self.results.lastrun)
            else:
                raise self.InvalidCmd('No run name currently define. Please add this information.')
            return

        if args[0] != self.run_name:
            if not os.path.exists(pjoin(self.me_dir,'Events',args[0], 'unweighted_events.lhe.gz'))\
            and not os.path.exists(pjoin(self.me_dir,'Events',args[0], 'unweighted_events.lhe')):
                raise self.InvalidCmd('No events file corresponding to %s run. '% args[0])
        self.run_name=args[0]
        
