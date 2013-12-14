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
import cmd
import glob
import logging
import math
import optparse
import os
import pydoc
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback


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
    import madgraph.various.banner as banner_mod
    import madgraph.various.misc as misc
    import madgraph.iolibs.files as files
    import madgraph.various.cluster as cluster
    import models.check_param_card as check_param_card
    from madgraph import InvalidCmd, MadGraph5Error, MG5DIR
    MADEVENT=False    
except Exception, error:
    if __debug__:
        print error
    # import from madevent directory
    import internal.extended_cmd as cmd
    import internal.banner as banner_mod
    import internal.misc as misc    
    import internal.cluster as cluster
    import internal.check_param_card as check_param_card
    import internal.files as files
    from internal import InvalidCmd, MadGraph5Error
    MADEVENT=True

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

    def help_plot(self):
        logger.info("syntax: help [RUN] [%s] [-f]" % '|'.join(self._plot_mode))
        logger.info("-- create the plot for the RUN (current run by default)")
        logger.info("     at the different stage of the event generation")
        logger.info("     Note than more than one mode can be specified in the same command.")
        logger.info("   This require to have MadAnalysis and td require. By default")
        logger.info("     if those programs are installed correctly, the creation")
        logger.info("     will be performed automaticaly during the event generation.")
        logger.info("   -f options: answer all question by default.")
        
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
    
    def help_decay_events(self, skip_syntax=False):
        if not skip_syntax:
            logger.info("syntax: decay_events [RUN]")
        logger.info("This functionality allows for the decay of resonances")
        logger.info("in a .lhe file, keeping track of the spin correlation effets.")
        logger.info("BE AWARE OF THE CURRENT LIMITATIONS:")
        logger.info("  (1) Only a succession of 2 body decay are currently allowed")



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
    
    def check_decay_events(self,args):
        """Check the argument for decay_events command
        syntax: decay_events [NAME]
        Note that other option are already remove at this point
        """
        
        opts = []
        if '-from_cards' in args:
            args.remove('-from_cards')
            opts.append('-from_cards')
        
        if len(args) == 0:
            if self.run_name:
                args.insert(0, self.run_name)
            elif self.results.lastrun:
                args.insert(0, self.results.lastrun)
            else:
                raise self.InvalidCmd('No run name currently defined. Please add this information.')
                return

        if args[0] != self.run_name:
            self.set_run_name(args[0])
        
        if self.mode == 'madevent':
            possible_path = [
                pjoin(self.me_dir,'Events',args[0], 'unweighted_events.lhe.gz'),
                pjoin(self.me_dir,'Events',args[0], 'unweighted_events.lhe')]
        else:
            possible_path = [
                           pjoin(self.me_dir,'Events',args[0], 'events.lhe.gz'),
                           pjoin(self.me_dir,'Events',args[0], 'events.lhe')]

        for path in possible_path:
            if os.path.exists(path):
                correct_path = path
                break
        else:
            raise self.InvalidCmd('No events file corresponding to %s run. ' % args[0])
        args[0] = correct_path
        
        args += opts
     

class MadEventAlreadyRunning(InvalidCmd):
    pass

#===============================================================================
# CommonRunCmd
#===============================================================================
class CommonRunCmd(HelpToCmd, CheckValidForCmd, cmd.Cmd):

    debug_output = 'ME5_debug'
    helporder = ['Main Commands', 'Documented commands', 'Require MG5 directory',
                   'Advanced commands']

    # The three options categories are treated on a different footage when a 
    # set/save configuration occur. current value are kept in self.options
    options_configuration = {'pythia8_path': './pythia8',
                       'hwpp_path': './herwigPP',
                       'thepeg_path': './thepeg',
                       'hepmc_path': './hepmc',
                       'madanalysis_path': './MadAnalysis',
                       'pythia-pgs_path':'./pythia-pgs',
                       'td_path':'./td',
                       'delphes_path':'./Delphes',
                       'exrootanalysis_path':'./ExRootAnalysis',
                       'syscalc_path': './SysCalc',
                       'timeout': 60,
                       'web_browser':None,
                       'eps_viewer':None,
                       'text_editor':None,
                       'fortran_compiler':None,
                       'auto_update':7,
                       'cluster_type': 'condor',
                       'cluster_status_update': (600, 30),
                       'cluster_nb_retry':1,
                       'cluster_retry_wait':300}
    
    options_madgraph= {'stdout_level':None}
    
    options_madevent = {'automatic_html_opening':True,
                         'run_mode':2,
                         'cluster_queue':'madgraph',
                         'cluster_time':None,
                         'cluster_memory':None,
                         'nb_core': None,
                         'cluster_temp_path':None}
    


    def __init__(self, me_dir, options, *args, **opts):
        """common"""
        
        cmd.Cmd.__init__(self, *args, **opts)
        # Define current MadEvent directory
        if me_dir is None and MADEVENT:
            me_dir = root_path
        
        self.me_dir = me_dir
        self.options = options  
        
        # usefull shortcut
        self.status = pjoin(self.me_dir, 'status')
        self.error =  pjoin(self.me_dir, 'error')
        self.dirbin = pjoin(self.me_dir, 'bin', 'internal')
        
        # Check that the directory is not currently running
        if os.path.exists(pjoin(me_dir,'RunWeb')): 
            message = '''Another instance of madevent is currently running.
            Please wait that all instance of madevent are closed. If no
            instance is running, you can delete the file
            %s and try again.''' % pjoin(me_dir,'RunWeb')
            raise MadEventAlreadyRunning, message
        else:
            pid = os.getpid()
            fsock = open(pjoin(me_dir,'RunWeb'),'w')
            fsock.write(`pid`)
            fsock.close()
            misc.Popen([pjoin(self.dirbin, 'gen_cardhtml-pl')], cwd=me_dir)
        
        self.to_store = []
        self.run_name = None
        self.run_tag = None
        self.banner = None
        # Load the configuration file
        self.set_configuration()
        self.configure_run_mode(self.options['run_mode'])
        
        
        # Get number of initial states
        nexternal = open(pjoin(self.me_dir,'Source','nexternal.inc')).read()
        found = re.search("PARAMETER\s*\(NINCOMING=(\d)\)", nexternal)
        self.ninitial = int(found.group(1))
        

    ############################################################################    
    def split_arg(self, line, error=False):
        """split argument and remove run_options"""
        
        args = cmd.Cmd.split_arg(line)
        for arg in args[:]:
            if not arg.startswith('-'):
                continue
            elif arg == '-c':
                self.configure_run_mode(1)
            elif arg == '-m':
                self.configure_run_mode(2)
            elif arg == '-f':
                self.force = True
            elif not arg.startswith('--'):
                if error:
                    raise self.InvalidCmd('%s argument cannot start with - symbol' % arg)
                else:
                    continue
            elif arg.startswith('--cluster'):
                self.configure_run_mode(1)
            elif arg.startswith('--multicore'):
                self.configure_run_mode(2)
            elif arg.startswith('--nb_core'):
                self.nb_core = int(arg.split('=',1)[1])
                self.configure_run_mode(2)
            elif arg.startswith('--web'):
                self.pass_in_web_mode()
                self.configure_run_mode(1)
            else:
                continue
            args.remove(arg)

        return args
    
    ############################################################################      
    def do_treatcards(self, line, amcatnlo=False):
        """Advanced commands: create .inc files from param_card.dat/run_card.dat"""


        keepwidth = False
        if '--keepwidth' in line:
            keepwidth = True
            line = line.replace('--keepwidth', '')
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

            if amcatnlo and int(run_card['ickkw']) == 3:
                # For FxFx merging, make sure that:
                # 1. Renormalization and factorization (and ellis-sexton scales) are not fixed
                scales=['fixed_ren_scale','fixed_fac_scale','fixed_QES_scale']
                for scale in scales:
                    if  banner_mod.RunCard.format('bool',run_card[scale]) == '.true.':
                        logger.info('''For consistency in the FxFx merging, \'%s\' has been set to false'''
                                    % scale,'$MG:color:BLACK')
                        run_card[scale]='F'
                # 2. Use kT algorithm for jets with pseudo-code size R=1.0
                jetparams=['jetradius','jetalgo']
                for jetparam in jetparams:
                    if float(run_card[jetparam]) != 1.0:
                        logger.info('''For consistency in the FxFx merging, \'%s\' has been set to 1.0'''
                               % jetparam ,'$MG:color:BLACK')
                        run_card[jetparam]='1.0'

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
                
                
            if amcatnlo and not keepwidth:
                # force particle in final states to have zero width
                pids = self.get_pid_final_states()
                # check those which are charged under qcd
                if not MADEVENT and pjoin(self.me_dir,'bin','internal') not in sys.path:                    
                        sys.path.insert(0,pjoin(self.me_dir,'bin','internal'))

                #Ensure that the model that we are going to load is the current
                #one.                    
                to_del = [name  for name in sys.modules.keys() 
                                                if name.startswith('internal.ufomodel')
                                                or name.startswith('ufomodel')]
                for name in to_del:
                    del(sys.modules[name]) 

                import ufomodel as ufomodel
                zero = ufomodel.parameters.ZERO
                no_width = [p for p in ufomodel.all_particles 
                        if (str(p.pdg_code) in pids or str(-p.pdg_code) in pids)
                           and p.color != 1 and p.width != zero]
                done = []
                for part in no_width:
                    if abs(part.pdg_code) in done:
                        continue
                    done.append(abs(part.pdg_code))
                    param = param_card['decay'].get((part.pdg_code,))
                    
                    if  param.value != 0:
                        logger.info('''For gauge cancellation, the width of \'%s\' has been set to zero.'''
                                    % part.name,'$MG:color:BLACK')
                        param.value = 0
                
            param_card.write_inc_file(outfile, ident_card, default)


    def ask_edit_cards(self, cards, mode='fixed', plot=True):
        """ """
        if not self.options['madanalysis_path']:
            plot = False
            
        self.ask_edit_card_static(cards, mode, plot, self.options['timeout'],
                                  self.ask)
    
    @staticmethod
    def ask_edit_card_static(cards, mode='fixed', plot=True, 
                             timeout=0, ask=None, **opt):
        if not ask:
            ask = CommonRunCmd.ask

        def path2name(path):
            if '_card' in path:
                return path.split('_card')[0]
            elif path == 'delphes_trigger.dat':
                return 'trigger'
            else:
                raise Exception, 'Unknow cards name %s' % path
            
        # Ask the user if he wants to edit any of the files
        #First create the asking text
        question = """Do you want to edit a card (press enter to bypass editing)?\n""" 
        possible_answer = ['0', 'done']
        card = {0:'done'}
        
        for i, card_name in enumerate(cards):
            imode = path2name(card_name)
            possible_answer.append(i+1)
            possible_answer.append(imode)
            question += '  %s / %-10s : %s\n' % (i+1, imode, card_name)
            card[i+1] = imode
        if plot:
            question += '  9 / %-10s : plot_card.dat\n' % 'plot'
            possible_answer.append(9)
            possible_answer.append('plot')
            card[9] = 'plot'        

        if 'param_card.dat' in cards:
            # Add the path options
            question += ' you can also\n'
            question += '   - enter the path to a valid card or banner.\n'
            question += '   - use the \'set\' command to modify a parameter directly.\n'
            question += '     The set option works only for param_card and run_card.\n'
            question += '     Type \'help set\' for more information on this command.\n'
        else:
            question += ' you can also\n'
            question += '   - enter the path to a valid card.\n'
        
        out = 'to_run'
        while out not in ['0', 'done']:
            out = ask(question, '0', possible_answer, timeout=int(1.5*timeout), 
                              path_msg='enter path', ask_class = AskforEditCard,
                              cards=cards, mode=mode, **opt)

            
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
           shower_card.dat
           madspin_card.dat
        """
        
        text = open(path).read(50000)
        if text == '':
            logger.warning('File %s is empty' % path)
            return 'unknown'
        text = re.findall('(<MGVersion>|ParticlePropagator|<mg5proccard>|CEN_max_tracker|#TRIGGER CARD|parameter set name|muon eta coverage|QES_over_ref|MSTP|Herwig\+\+|MSTU|Begin Minpts|gridpack|ebeam1|BLOCK|DECAY|launch|madspin|set)', text, re.I)
        text = [t.lower() for t in text]
        if '<mgversion>' in text or '<mg5proccard>' in text:
            return 'banner'
        elif 'particlepropagator' in text:
            return 'delphes_card.dat'
        elif 'cen_max_tracker' in text:
            return 'delphes_card.dat'
        elif '#trigger card' in text:
            return 'delphes_trigger.dat'
        elif 'parameter set name' in text:
            return 'pgs_card.dat'
        elif 'muon eta coverage' in text:
            return 'pgs_card.dat'
        elif 'mstp' in text and not 'herwig++' in text:
            return 'pythia_card.dat'
        elif 'begin minpts' in text:
            return 'plot_card.dat'
        elif ('gridpack' in text and 'ebeam1' in text) or \
                ('qes_over_ref' in text and 'ebeam1' in text):
            return 'run_card.dat'
        elif 'block' in text and 'decay' in text: 
            return 'param_card.dat'
        elif 'herwig++' in text:
            return 'shower_card.dat'
        elif 'decay' in text and 'launch' in text:
            return 'madspin_card.dat'
        elif 'launch' in text and 'set' in text:
            return 'reweight_card.dat'
        else:
            return 'unknown'


    ############################################################################
    def get_available_tag(self):
        """create automatically a tag"""
        
        used_tags = [r['tag'] for r in self.results[self.run_name]]
        i=0
        while 1:
            i+=1
            if 'tag_%s' %i not in used_tags:
                return 'tag_%s' % i
   
    
    ############################################################################
    def create_plot(self, mode='parton', event_path=None, output=None, tag=None):
        """create the plot""" 

        madir = self.options['madanalysis_path']
        if not tag:
            tag = self.run_card['run_tag']  
        td = self.options['td_path']

        if not madir or not td or \
            not os.path.exists(pjoin(self.me_dir, 'Cards', 'plot_card.dat')):
            return False

        if 'ickkw' in self.run_card and int(self.run_card['ickkw']) and \
                mode == 'Pythia':
            self.update_status('Create matching plots for Pythia', level='pythia')
            # recover old data if none newly created
            if not os.path.exists(pjoin(self.me_dir,'Events','events.tree')):
                misc.call(['gunzip', '-c', pjoin(self.me_dir,'Events', 
                      self.run_name, '%s_pythia_events.tree.gz' % tag)],
                      stdout=open(pjoin(self.me_dir,'Events','events.tree'),'w')
                          )
                files.mv(pjoin(self.me_dir,'Events',self.run_name, tag+'_pythia_xsecs.tree'),
                     pjoin(self.me_dir,'Events','xsecs.tree'))
                
            # Generate the matching plots
            misc.call([self.dirbin+'/create_matching_plots.sh', 
                       self.run_name, tag, madir],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            cwd=pjoin(self.me_dir,'Events'))

            #Clean output
            misc.call(['gzip','-f','events.tree'], 
                                                cwd=pjoin(self.me_dir,'Events'))          
            files.mv(pjoin(self.me_dir,'Events','events.tree.gz'), 
                     pjoin(self.me_dir,'Events',self.run_name, tag + '_pythia_events.tree.gz'))
            files.mv(pjoin(self.me_dir,'Events','xsecs.tree'), 
                     pjoin(self.me_dir,'Events',self.run_name, tag+'_pythia_xsecs.tree'))
                        

        if not event_path:
            if mode == 'parton':
                possibilities=[
                    pjoin(self.me_dir, 'Events', 'unweighted_events.lhe'),
                    pjoin(self.me_dir, 'Events', 'unweighted_events.lhe.gz'),  
                    pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe'),
                    pjoin(self.me_dir, 'Events', self.run_name, 'unweighted_events.lhe.gz')]
                for event_path in possibilities:
                    if os.path.exists(event_path):
                        break
                output = pjoin(self.me_dir, 'HTML',self.run_name, 'plots_parton.html')
             
            elif mode == 'Pythia':
                event_path = pjoin(self.me_dir, 'Events','pythia_events.lhe')
                output = pjoin(self.me_dir, 'HTML',self.run_name, 
                              'plots_pythia_%s.html' % tag)                                   
            elif mode == 'PGS':
                event_path = pjoin(self.me_dir, 'Events', self.run_name, 
                                   '%s_pgs_events.lhco' % tag)
                output = pjoin(self.me_dir, 'HTML',self.run_name, 
                              'plots_pgs_%s.html' % tag)  
            elif mode == 'Delphes':
                event_path = pjoin(self.me_dir, 'Events', self.run_name,'%s_delphes_events.lhco' % tag)
                output = pjoin(self.me_dir, 'HTML',self.run_name, 
                              'plots_delphes_%s.html' % tag) 
            else:
                raise self.InvalidCmd, 'Invalid mode %s' % mode
        elif mode == 'reweight' and not output:
                output = pjoin(self.me_dir, 'HTML',self.run_name, 
                              'plots_%s.html' % tag)   

            
            
        if not os.path.exists(event_path):
            if os.path.exists(event_path+'.gz'):
                os.system('gunzip -f %s.gz ' % event_path)
            else:
                raise self.InvalidCmd, 'Events file %s does not exits' % event_path
        
        self.update_status('Creating Plots for %s level' % mode, level = mode.lower())
        
        mode = mode.lower()
        if mode not in ['parton', 'reweight']:
            plot_dir = pjoin(self.me_dir, 'HTML', self.run_name,'plots_%s_%s' % (mode.lower(),tag))
        elif mode == 'parton':
            plot_dir = pjoin(self.me_dir, 'HTML', self.run_name,'plots_parton')
        else:
            plot_dir =pjoin(self.me_dir, 'HTML', self.run_name,'plots_%s' % (tag))
             
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir) 
        
        files.ln(pjoin(self.me_dir, 'Cards','plot_card.dat'), plot_dir, 'ma_card.dat')
                
        try:
            proc = misc.Popen([os.path.join(madir, 'plot_events')],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'w'),
                            stderr = subprocess.STDOUT,
                            stdin=subprocess.PIPE,
                            cwd=plot_dir)
            proc.communicate('%s\n' % event_path)
            del proc
            #proc.wait()
            misc.call(['%s/plot' % self.dirbin, madir, td],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'a'),
                            stderr = subprocess.STDOUT,
                            cwd=plot_dir)
    
            misc.call(['%s/plot_page-pl' % self.dirbin, 
                                os.path.basename(plot_dir),
                                mode],
                            stdout = open(pjoin(plot_dir, 'plot.log'),'a'),
                            stderr = subprocess.STDOUT,
                            cwd=pjoin(self.me_dir, 'HTML', self.run_name))

            shutil.move(pjoin(self.me_dir, 'HTML',self.run_name ,'plots.html'),
                                                                         output)

            logger.info("Plots for %s level generated, see %s" % \
                         (mode, output))
        except OSError, error:
            logger.error('fail to create plot: %s. Please check that MadAnalysis is correctly installed.' % error)
        
        self.update_status('End Plots for %s level' % mode, level = mode.lower(),
                                                                 makehtml=False)
        
        return True   

    def run_hep2lhe(self, banner_path = None):
        """Run hep2lhe on the file Events/pythia_events.hep"""

        if not self.options['pythia-pgs_path']:
            raise self.InvalidCmd, 'No pythia-pgs path defined'
            
        pydir = pjoin(self.options['pythia-pgs_path'], 'src')
        eradir = self.options['exrootanalysis_path']

        # Creating LHE file
        if misc.is_executable(pjoin(pydir, 'hep2lhe')):
            self.update_status('Creating Pythia LHE File', level='pythia')
            # Write the banner to the LHE file
            out = open(pjoin(self.me_dir,'Events','pythia_events.lhe'), 'w')
            #out.writelines('<LesHouchesEvents version=\"1.0\">\n')    
            out.writelines('<!--\n')
            out.writelines('# Warning! Never use this file for detector studies!\n')
            out.writelines('-->\n<!--\n')
            if banner_path:
                out.writelines(open(banner_path).read().replace('<LesHouchesEvents version="1.0">',''))
            out.writelines('\n-->\n')
            out.close()
            
            self.cluster.launch_and_wait(self.dirbin+'/run_hep2lhe', 
                                         argument= [pydir],
                                        cwd=pjoin(self.me_dir,'Events'))

            logger.info('Warning! Never use this pythia lhe file for detector studies!')
            # Creating ROOT file
            if eradir and misc.is_executable(pjoin(eradir, 'ExRootLHEFConverter')):
                self.update_status('Creating Pythia LHE Root File', level='pythia')
                try:
                    misc.call([eradir+'/ExRootLHEFConverter', 
                             'pythia_events.lhe', 
                             pjoin(self.run_name, '%s_pythia_lhe_events.root' % self.run_tag)],
                            cwd=pjoin(self.me_dir,'Events'))              
                except Exception, error:
                    misc.sprint('ExRootLHEFConverter fails', str(error), 
                                                                     log=logger)
                    pass

    def store_result(self):
        """Dummy routine, to be overwritten by daughter classes"""

        pass

    ############################################################################      
    def do_pgs(self, line):
        """launch pgs"""
        
        args = self.split_arg(line)
        # Check argument's validity
        if '--no_default' in args:
            no_default = True
            args.remove('--no_default')
        else:
            no_default = False

        # Check all arguments
        # This might launch a gunzip in another thread. After the question
        # This thread need to be wait for completion. (This allow to have the 
        # question right away and have the computer working in the same time)
        # if lock is define this a locker for the completion of the thread
        lock = self.check_pgs(args) 

        # Check that the pgs_card exists. If not copy the default 
        if not os.path.exists(pjoin(self.me_dir, 'Cards', 'pgs_card.dat')):
            if no_default:
                logger.info('No pgs_card detected, so not run pgs')
                return 
            
            files.cp(pjoin(self.me_dir, 'Cards', 'pgs_card_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'pgs_card.dat'))
            logger.info('No pgs card found. Take the default one.')        
        
        if not (no_default or self.force):
            self.ask_edit_cards(['pgs_card.dat'])
            
        self.update_status('prepare PGS run', level=None)  

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
        banner_path = pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, self.run_tag))
        if os.path.exists(pjoin(self.me_dir, 'Source', 'banner_header.txt')):
            self.banner.add(pjoin(self.me_dir, 'Cards','pgs_card.dat'))
            self.banner.write(banner_path)
        else:
            open(banner_path, 'w').close()

        ########################################################################
        # now pass the event to a detector simulator and reconstruct objects
        ########################################################################
        if lock:
            lock.acquire()
        # Prepare the output file with the banner
        ff = open(pjoin(self.me_dir, 'Events', 'pgs_events.lhco'), 'w')
        if os.path.exists(pjoin(self.me_dir, 'Source', 'banner_header.txt')):
            text = open(banner_path).read()
            text = '#%s' % text.replace('\n','\n#')
            dico = self.results[self.run_name].get_current_info()
            text +='\n##  Integrated weight (pb)  : %.4g' % dico['cross']
            text +='\n##  Number of Event         : %s\n' % dico['nb_event']
            ff.writelines(text)
        ff.close()

        try: 
            os.remove(pjoin(self.me_dir, 'Events', 'pgs.done'))
        except Exception:
            pass
         
        pgs_log = pjoin(self.me_dir, 'Events', self.run_name, "%s_pgs.log" % tag)
        self.cluster.launch_and_wait('../bin/internal/run_pgs', 
                            argument=[pgsdir], cwd=pjoin(self.me_dir,'Events'),
                            stdout=pgs_log, stderr=subprocess.STDOUT)
        
        if not os.path.exists(pjoin(self.me_dir, 'Events', 'pgs.done')):
            logger.error('Fail to create LHCO events')
            return 
        else:
            os.remove(pjoin(self.me_dir, 'Events', 'pgs.done'))
            
        if os.path.getsize(banner_path) == os.path.getsize(pjoin(self.me_dir, 'Events','pgs_events.lhco')):
            misc.call(['cat pgs_uncleaned_events.lhco >>  pgs_events.lhco'], 
                            cwd=pjoin(self.me_dir, 'Events'))
            os.remove(pjoin(self.me_dir, 'Events', 'pgs_uncleaned_events.lhco '))

        # Creating Root file
        if eradir and misc.is_executable(pjoin(eradir, 'ExRootLHCOlympicsConverter')):
            self.update_status('Creating PGS Root File', level='pgs')
            try:
                misc.call([eradir+'/ExRootLHCOlympicsConverter', 
                             'pgs_events.lhco',pjoin('%s/%s_pgs_events.root' % (self.run_name, tag))],
                            cwd=pjoin(self.me_dir, 'Events')) 
            except Exception:
                logger.warning('fail to produce Root output [problem with ExRootAnalysis')
        if os.path.exists(pjoin(self.me_dir, 'Events', 'pgs_events.lhco')):
            # Creating plots
            files.mv(pjoin(self.me_dir, 'Events', 'pgs_events.lhco'), 
                    pjoin(self.me_dir, 'Events', self.run_name, '%s_pgs_events.lhco' % tag))
            self.create_plot('PGS')
            misc.call(['gzip','-f', pjoin(self.me_dir, 'Events', 
                                                self.run_name, '%s_pgs_events.lhco' % tag)])

        self.update_status('finish', level='pgs', makehtml=False)

    ############################################################################
    def do_delphes(self, line):
        """ run delphes and make associate root file/plot """
 
        args = self.split_arg(line)
        # Check argument's validity
        if '--no_default' in args:
            no_default = True
            args.remove('--no_default')
        else:
            no_default = False
        # Check all arguments
        # This might launch a gunzip in another thread. After the question
        # This thread need to be wait for completion. (This allow to have the 
        # question right away and have the computer working in the same time)
        # if lock is define this a locker for the completion of the thread
        lock = self.check_delphes(args) 
        self.update_status('prepare delphes run', level=None)
        
        
        if os.path.exists(pjoin(self.options['delphes_path'], 'data')):
            delphes3 = False
            prog = '../bin/internal/run_delphes'
        else:
            delphes3 = True
            prog =  '../bin/internal/run_delphes3'
                
        # Check that the delphes_card exists. If not copy the default and
        # ask for edition of the card.
        if not os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_card.dat')):
            if no_default:
                logger.info('No delphes_card detected, so not run Delphes')
                return
            
            files.cp(pjoin(self.me_dir, 'Cards', 'delphes_card_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'delphes_card.dat'))
            logger.info('No delphes card found. Take the default one.')
        if not delphes3 and not os.path.exists(pjoin(self.me_dir, 'Cards', 'delphes_trigger.dat')):    
            files.cp(pjoin(self.me_dir, 'Cards', 'delphes_trigger_default.dat'),
                     pjoin(self.me_dir, 'Cards', 'delphes_trigger.dat'))
        if not (no_default or self.force):
            if delphes3:
                self.ask_edit_cards(['delphes_card.dat'], args)
            else:
                self.ask_edit_cards(['delphes_card.dat', 'delphes_trigger.dat'], args)
            
        self.update_status('Running Delphes', level=None)  
        # Wait that the gunzip of the files is finished (if any)
        if lock:
            lock.acquire()


 
        delphes_dir = self.options['delphes_path']
        tag = self.run_tag
        if os.path.exists(pjoin(self.me_dir, 'Source', 'banner_header.txt')):
            self.banner.add(pjoin(self.me_dir, 'Cards','delphes_card.dat'))
            if not delphes3:
                self.banner.add(pjoin(self.me_dir, 'Cards','delphes_trigger.dat'))
            self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, tag)))
        
        cross = self.results[self.run_name].get_current_info()['cross']
                    
        delphes_log = pjoin(self.me_dir, 'Events', self.run_name, "%s_delphes.log" % tag)
        self.cluster.launch_and_wait(prog, 
                        argument= [delphes_dir, self.run_name, tag, str(cross)],
                        stdout=delphes_log, stderr=subprocess.STDOUT,
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
            misc.call(['gzip','-f', pjoin(self.me_dir, 'Events', self.run_name, '%s_delphes_events.lhco' % tag)])


        
        self.update_status('delphes done', level='delphes', makehtml=False)   

    ############################################################################ 
    def get_pid_final_states(self):
        """Find the pid of all particles in the final states"""
        pids = set()
        subproc = [l.strip() for l in open(pjoin(self.me_dir,'SubProcesses', 
                                                                 'subproc.mg'))]
        nb_init = self.ninitial
        pat = re.compile(r'''DATA \(IDUP\(I,\d+\),I=1,\d+\)/([\+\-\d,\s]*)/''', re.I)
        for Pdir in subproc:
            text = open(pjoin(self.me_dir, 'SubProcesses', Pdir, 'born_leshouche.inc')).read()
            group = pat.findall(text)
            for particles in group:
                particles = particles.split(',')
                pids.update(set(particles[nb_init:]))
        
        return pids
                
    ############################################################################
    def get_pdf_input_filename(self):
        """return the name of the file which is used by the pdfset"""
        
        if hasattr(self, 'pdffile') and self.pdffile:
            return self.pdffile
        else:
            for line in open(pjoin(self.me_dir,'Source','PDF','pdf_list.txt')):
                data = line.split()
                if len(data) < 4:
                    continue
                if data[1].lower() == self.run_card['pdlabel'].lower():
                    self.pdffile = pjoin(self.me_dir, 'lib', 'Pdfdata', data[2])
                    return self.pdffile 
            else:
                # possible when using lhapdf
                self.pdffile = subprocess.Popen('%s --pdfsets-path' % self.options['lhapdf'], 
                        shell = True, stdout = subprocess.PIPE).stdout.read().strip()
                #self.pdffile = pjoin(self.me_dir, 'lib', 'PDFsets')
                return self.pdffile
                
                
        
  
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
            if args[1] == 'None':
                args[1] = None
            self.options['fortran_compiler'] = args[1]
            current = misc.detect_current_compiler(pjoin(self.me_dir,'Source','make_opts'))
            if current != args[1] and args[1] != None:
                misc.mod_compilator(self.me_dir, args[1], current)
        elif args[0] == "run_mode":
            if not args[1] in [0,1,2,'0','1','2']:
                raise self.InvalidCmd, 'run_mode should be 0, 1 or 2.'
            self.cluster_mode = int(args[1])
            self.options['run_mode'] =  self.cluster_mode
        elif args[0] in  ['cluster_type', 'cluster_queue', 'cluster_temp_path']:
            if args[1] == 'None':
                args[1] = None
            self.options[args[0]] = args[1]
            # cluster (re)-initialization done later
            # self.cluster update at the end of the routine
        elif args[0] in ['cluster_nb_retry', 'cluster_retry_wait']:
            self.options[args[0]] = int(args[1])
            # self.cluster update at the end of the routine
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
        elif args[0] == 'cluster_status_update':
            if '(' in args[1]:
                data = ' '.join([a for a in args[1:] if not a.startswith('-')])
                data = data.replace('(','').replace(')','').replace(',',' ').split()
                first, second = data[:2]
            else: 
                first, second = args[1:3]            
            
            self.options[args[0]] = (int(first), int(second))
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
                
    def post_set(self, stop, line):
        """Check if we need to save this in the option file"""
        try:
            args = self.split_arg(line)
            if 'cluster' in args[0] or args[0] == 'run_mode':
                self.configure_run_mode(self.options['run_mode'])             


            # Check the validity of the arguments
            self.check_set(args)
            
            if args[0] in self.options_configuration and '--no_save' not in args:
                self.exec_cmd('save options --auto')
            elif args[0] in self.options_madevent:
                logger.info('This option will be the default in any output that you are going to create in this session.')
                logger.info('In order to keep this changes permanent please run \'save options\'')
            return stop
        except self.InvalidCmd:
            return stop

    def configure_run_mode(self, run_mode):
        """change the way to submit job 0: single core, 1: cluster, 2: multicore"""
        
        self.cluster_mode = run_mode
        
        if run_mode == 2:
            if not self.nb_core:
                import multiprocessing
                self.nb_core = multiprocessing.cpu_count()
            nb_core =self.nb_core
        elif run_mode == 0:
            nb_core = 1 
        
        
        
        if run_mode in [0, 2]:
            self.cluster = cluster.MultiCore( 
                             **self.options)
                             #cluster_temp_path=self.options['cluster_temp_path'],
            
        if self.cluster_mode == 1:
            opt = self.options
            cluster_name = opt['cluster_type']
            self.cluster = cluster.from_name[cluster_name](**opt)

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


    def do_quit(self, line):
        """Not in help: exit """
  
        try:
            os.remove(pjoin(self.me_dir,'RunWeb'))
        except Exception:
            pass
        try:
            self.store_result()
        except Exception:
            # If nothing runs they they are no result to update
            pass
        
        try:
            self.update_status('', level=None)
        except Exception, error:        
            pass
        devnull = open(os.devnull, 'w')
        try:
            misc.call(['./bin/internal/gen_cardhtml-pl'], cwd=self.me_dir,
                        stdout=devnull, stderr=devnull)
        except Exception:
            pass
        devnull.close()

        return super(CommonRunCmd, self).do_quit(line)
    
    # Aliases
    do_EOF = do_quit
    do_exit = do_quit
  

    def update_status(self, status, level, makehtml=True, force=True, 
                      error=False, starttime = None, update_results=True,
                      print_log=True):
        """ update the index status """
        
        if makehtml and not force:
            if hasattr(self, 'next_update') and time.time() < self.next_update:
                return
            else:
                self.next_update = time.time() + 3
                
        if print_log:
            if isinstance(status, str):
                if '<br>' not  in status:
                    logger.info(status)
            elif starttime:
                running_time = misc.format_timer(time.time()-starttime)
                logger.info(' Idle: %s,  Running: %s,  Completed: %s [ %s ]' % \
                           (status[0], status[1], status[2], running_time))
            else: 
                logger.info(' Idle: %s,  Running: %s,  Completed: %s' % status[:3])
        
        if update_results:
            self.results.update(status, level, makehtml=makehtml, error=error)
        
        
    ############################################################################
    def keep_cards(self, need_card=[], ignore=[]):
        """Ask the question when launching generate_events/multi_run"""
        
        check_card = ['pythia_card.dat', 'pgs_card.dat','delphes_card.dat',
                      'delphes_trigger.dat', 'madspin_card.dat', 'shower_card.dat',
                      'reweight_card.dat']
        
        cards_path = pjoin(self.me_dir,'Cards')
        for card in check_card:
            if card in ignore:
                continue
            if card not in need_card:
                if os.path.exists(pjoin(cards_path, card)):
                    os.remove(pjoin(cards_path, card))
            else:
                if not os.path.exists(pjoin(cards_path, card)):
                    default = card.replace('.dat', '_default.dat')
                    files.cp(pjoin(cards_path, default),pjoin(cards_path, card)) 
                
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
                self.set_configuration(config_path=config_path, final=final)
                return
            if 'HOME' in os.environ:
                config_path = pjoin(os.environ['HOME'],'.mg5', 
                                                        'mg5_configuration.txt')
                if os.path.exists(config_path):
                    self.set_configuration(config_path=config_path,  final=False)
            if amcatnlo:
                me5_config = pjoin(self.me_dir, 'Cards', 'amcatnlo_configuration.txt')
            else:
                me5_config = pjoin(self.me_dir, 'Cards', 'me5_configuration.txt')
            self.set_configuration(config_path=me5_config, final=False, initdir=self.me_dir)
                
            if self.options.has_key('mg5_path') and self.options['mg5_path']:
                MG5DIR = self.options['mg5_path']
                config_file = pjoin(MG5DIR, 'input', 'mg5_configuration.txt')
                self.set_configuration(config_path=config_file, final=False,initdir=MG5DIR)
            else:
                self.options['mg5_path'] = None
            return self.set_configuration(config_path=me5_config, final=final,initdir=self.me_dir)

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
            elif key.startswith('cluster') and key != 'cluster_status_update':
                if key in ('cluster_nb_retry','cluster_wait_retry'):
                    self.options[key] = int(self.options[key])
                if hasattr(self,'cluster'):
                    del self.cluster
                pass              
            elif key == 'automatic_html_opening':
                if self.options[key] in ['False', 'True']:
                    self.options[key] =eval(self.options[key])
            elif key not in ['text_editor','eps_viewer','web_browser','stdout_level',
                              'complex_mass_scheme', 'gauge', 'group_subprocesses']:
                # Default: try to set parameter
                try:
                    self.do_set("%s %s --no_save" % (key, self.options[key]), log=False)
                except self.InvalidCmd:
                    logger.warning("Option %s from config file not understood" \
                                   % key)
        
        # Configure the way to open a file:
        misc.open_file.configure(self.options)
        self.configure_run_mode(self.options['run_mode'])
          
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

        if '-from_cards' in line and not os.path.exists(pjoin(self.me_dir, 'Cards', 'madspin_card.dat')):
            return
                
        # First need to load MadSpin
        
        # Check that MG5 directory is present .
        if MADEVENT and not self.options['mg5_path']:
            raise self.InvalidCmd, '''The module decay_events requires that MG5 is installed on the system.
            You can install it and set its path in ./Cards/me5_configuration.txt'''
        elif MADEVENT:
            sys.path.append(self.options['mg5_path'])
        try:
            import MadSpin.decay as decay
            import MadSpin.interface_madspin as interface_madspin
        except ImportError:
            raise self.ConfigurationError, '''Can\'t load MadSpin
            The variable mg5_path might not be correctly configured.'''
        
        self.update_status('Running MadSpin', level='madspin')        
        if not '-from_cards' in line:
            self.keep_cards(['madspin_card.dat'])
            self.ask_edit_cards(['madspin_card.dat'], 'fixed', plot=False)        
        self.help_decay_events(skip_syntax=True)

        # load the name of the event file
        args = self.split_arg(line) 
        self.check_decay_events(args) 
        # args now alway content the path to the valid files
        madspin_cmd = interface_madspin.MadSpinInterface(args[0]) 
        madspin_cmd.update_status = lambda *x,**opt: self.update_status(*x, level='madspin',**opt) 

        path = pjoin(self.me_dir, 'Cards', 'madspin_card.dat')
        
        madspin_cmd.import_command_file(path)
                
        # create a new run_name directory for this output
        i = 1
        while os.path.exists(pjoin(self.me_dir,'Events', '%s_decayed_%i' % (self.run_name,i))):
            i+=1
        new_run = '%s_decayed_%i' % (self.run_name,i)
        evt_dir = pjoin(self.me_dir, 'Events')
        
        os.mkdir(pjoin(evt_dir, new_run))
        current_file = args[0].replace('.lhe', '_decayed.lhe')
        new_file = pjoin(evt_dir, new_run, os.path.basename(args[0]))
        if not os.path.exists(current_file):
            if os.path.exists(current_file+'.gz'):
                current_file += '.gz'
                new_file += '.gz'
            else:
                logger.error('MadSpin fails to create any decayed file.')
                return
        
        files.mv(current_file, new_file)
        logger.info("The decayed event file has been moved to the following location: ")
        logger.info(new_file)        
 
        if hasattr(self, 'results'):
            current = self.results.current
            nb_event = self.results.current['nb_event']
            if not nb_event:
                current = self.results[self.run_name][0]
                nb_event = current['nb_event']
            
            cross = current['cross']
            error = current['error']
            self.results.add_run( new_run, self.run_card)
            self.results.add_detail('nb_event', nb_event)
            self.results.add_detail('cross', cross * madspin_cmd.branching_ratio)
            self.results.add_detail('error', error * madspin_cmd.branching_ratio)
            self.results.add_detail('run_mode', current['run_mode'])
    
        self.run_name = new_run
        self.banner.add(path)
        self.banner.write(pjoin(self.me_dir,'Events',self.run_name, '%s_%s_banner.txt' %
                                (self.run_name, self.run_tag)))
        self.update_status('MadSpin Done', level='parton', makehtml=False)
        if 'unweighted' in os.path.basename(args[0]):
            self.create_plot('parton')
    
    def complete_decay_events(self, text, line, begidx, endidx):
        args = self.split_arg(line[0:begidx], error=False)
        if len(args) == 1:
            return self.complete_plot(text, line, begidx, endidx)
        else:
            return
        

class AskforEditCard(cmd.OneLinePathCompletion):
    """A class for asking a question where in addition you can have the 
    set command define and modifying the param_card/run_card correctly"""
    
    def __init__(self, question, cards=[], mode='auto', *args, **opt):

        # Initiation
        if 'pwd' in opt:
            self.me_dir = opt['pwd']
            del opt['pwd']

        cmd.OneLinePathCompletion.__init__(self, question, *args, **opt)

        if not hasattr(self, 'me_dir') or not self.me_dir:
            self.me_dir = self.mother_interface.me_dir

        # read the card

        try:
            self.run_card = banner_mod.RunCard(pjoin(self.me_dir,'Cards','run_card.dat'))
        except IOError:
            self.run_card = {}
        try:
            self.param_card = check_param_card.ParamCard(pjoin(self.me_dir,'Cards','param_card.dat'))   
        except (check_param_card.InvalidParamCard, ValueError) as e:
            logger.error('Current param_card is not valid. We are going to use the default one.')
            logger.error('problem detected: %s' % e)
            files.cp(pjoin(self.me_dir,'Cards','param_card_default.dat'), 
                     pjoin(self.me_dir,'Cards','param_card.dat'))
            self.param_card = check_param_card.ParamCard(pjoin(self.me_dir,'Cards','param_card.dat'))
        default_param = check_param_card.ParamCard(pjoin(self.me_dir,'Cards','param_card_default.dat'))
        try:
            run_card_def = banner_mod.RunCard(pjoin(self.me_dir,'Cards','run_card_default.dat'))
        except IOError:
            run_card_def = {}
            
        self.pname2block = {}
        self.conflict = []
        self.restricted_value = {}
        self.mode = mode
        self.cards = cards
        
        # Read the comment of the param_card_default to find name variable for 
        # the param_card also check which value seems to be constrained in the
        # model.
        for bname, block in default_param.items():
            for lha_id, param in block.param_dict.items():
                all_var = []
                comment = param.comment
                # treat merge parameter
                if comment.strip().startswith('set of param :'):
                    all_var = list(re.findall(r'''[^-]1\*(\w*)\b''', comment))
                # just the variable name as comment
                elif len(comment.split()) == 1:
                    all_var = [comment.strip().lower()]
                # either contraction or not formatted
                else:
                    split = comment.split()
                    if len(split) >2 and split[1] == ':':
                        # NO VAR associated
                        self.restricted_value[(bname, lha_id)] = ' '.join(split[1:])
                    elif len(split) == 2:
                        if re.search(r'''\[[A-Z]\]eV\^''', split[1]):
                            all_var = [comment.strip().lower()]
                    else:
                        # not recognized format
                        continue
                    
                for var in all_var:
                    var = var.lower()
                    if var in self.pname2block:
                        self.pname2block[var].append((bname, lha_id))
                    else:
                        self.pname2block[var] = [(bname, lha_id)]
        
        if run_card_def:
            self.run_set = run_card_def.keys() + self.run_card.hidden_param
        elif self.run_card:
            self.run_set = self.run_card.keys()
        else:
            self.run_set = []            
        # check for conflict with run_card
        for var in self.pname2block:                
            if var in self.run_set:
                self.conflict.append(var)        
                            
    
    def complete_set(self, text, line, begidx, endidx):
        """ Complete the set command"""

        prev_timer = signal.alarm(0) # avoid timer if any
        if prev_timer:
            nb_back = len(line)
            self.stdout.write('\b'*nb_back + '[timer stopped]\n')
            self.stdout.write(line)
            self.stdout.flush()
        
        possibilities = {}
        allowed = {}
        args = self.split_arg(line[0:begidx])
        if len(args) == 1:
            allowed = {'category':'', 'run_card':'', 'block':'all', 'param_card':''}
        elif len(args) == 2:
            if args[1] == 'run_card':
                allowed = {'run_card':'default'}
            elif args[1] == 'param_card':
                allowed = {'block':'all', 'param_card':'default'}
            elif args[1] in self.param_card.keys():
                allowed = {'block':args[1]}
            elif args[1] == 'width':
                allowed = {'block': 'decay'}
            else:
                allowed = {'value':''}
        else:
            start = 1
            if args[1] in  ['run_card', 'param_card']:
                start = 2
            if args[start] in self.param_card.keys():
                if args[start+1:]:
                    allowed = {'block':(args[start], args[start+1:])}
                else:
                    allowed = {'block':args[start]}
            elif len(args) == start +1:
                    allowed['value'] = ''

            
        if 'category' in allowed.keys():
            possibilities['category of parameter (optional)'] = \
                          self.list_completion(text, ['run_card', 'param_card'])
        
        if 'run_card' in allowed.keys():
            opts = self.run_set
            if allowed['run_card'] == 'default':
                opts.append('default')
            
            possibilities['Run Card'] = self.list_completion(text, opts)

        if 'param_card' in allowed.keys():
            opts = self.pname2block.keys()
            if allowed['param_card'] == 'default':
                opts.append('default')
            possibilities['Param Card'] = self.list_completion(text, opts)
                                
        if 'value' in allowed.keys():
            opts = ['default']
            if 'decay' in args:
                opts.append('Auto')
            if args[-1] in self.pname2block and self.pname2block[args[-1]][0][0] == 'decay':
                opts.append('Auto')
            possibilities['Special Value'] = self.list_completion(text, opts)
                 

        if 'block' in allowed.keys():
            if allowed['block'] == 'all':
                allowed_block = [i for i in self.param_card.keys() if 'qnumbers' not in i]
                allowed_block.append('width')
                possibilities['Param Card Block' ] = \
                                       self.list_completion(text, allowed_block)
            elif isinstance(allowed['block'], basestring):
                block = self.param_card[allowed['block']].param_dict
                ids = [str(i[0]) for i in block 
                          if (allowed['block'], i) not in self.restricted_value]
                possibilities['Param Card id' ] = self.list_completion(text, ids)
                varname = [name for name, all_var in self.pname2block.items()
                                               if any((bname == allowed['block'] 
                                                   for bname,lhaid in all_var))]
                possibilities['Param card variable'] = self.list_completion(text,
                                                                        varname)
            else:
                block = self.param_card[allowed['block'][0]].param_dict
                nb = len(allowed['block'][1])
                ids = [str(i[nb]) for i in block if len(i) > nb and \
                            [str(a) for a in i[:nb]] == allowed['block'][1]]
                
                if not ids:
                    if tuple([int(i) for i in allowed['block'][1]]) in block:
                        opts = ['default']
                        if allowed['block'][0] == 'decay':
                            opts.append('Auto')
                        possibilities['Special value'] = self.list_completion(text, opts)
                possibilities['Param Card id' ] = self.list_completion(text, ids)        

        return self.deal_multiple_categories(possibilities)
           
    def do_set(self, line):
        """ edit the value of one parameter in the card"""

        args = self.split_arg(line.lower())
        if '=' in args[-1]:
            arg1, arg2 = args.pop(-1).split('=')
            args += [arg1, arg2]

        start = 0
        if len(args) < 2:
            logger.warning('Invalid set command %s (need two arguments)' % line)
            return

        # Special case for the qcut value
        if args[0].lower() == 'qcut':
            pythia_path = pjoin(self.me_dir, 'Cards','pythia_card.dat')
            if os.path.exists(pythia_path):
                logger.info('add line QCUT = %s in pythia_card.dat' % args[1])
                p_card = open(pythia_path,'r').read()
                p_card, n = re.subn('''^\s*QCUT\s*=\s*[\de\+\-\.]*\s*$''', 
                                    ''' QCUT = %s ''' % args[1], \
                                    p_card, flags=(re.M+re.I))
                if n==0:
                    p_card = '%s \n QCUT= %s' % (p_card, args[1])
                open(pythia_path, 'w').write(p_card)
                return

        card = '' #store which card need to be modify (for name conflict)
        if args[0] in ['run_card', 'param_card']:
            if args[1] == 'default':
                logging.info('replace %s by the default card' % args[0])
                files.cp(pjoin(self.me_dir,'Cards','%s_default.dat' % args[0]),
                        pjoin(self.me_dir,'Cards','%s.dat'% args[0]))                
                if args[0] == 'param_card':
                    self.param_card = check_param_card.ParamCard(pjoin(self.me_dir,'Cards','param_card.dat'))
                elif args[0] == 'run_card':
                    self.run_card = banner_mod.RunCard(pjoin(self.me_dir,'Cards','run_card.dat'))
                return
            else:
                card = args[0]
            start=1
            if len(args) < 3:
                logger.warning('Invalid set command: %s (not enough arguments)' % line)
                return

        #### RUN CARD
        if args[start] in [l.lower() for l in self.run_set] and card != 'param_card':
            if args[start] not in self.run_set:
                args[start] = [l for l in self.run_set \
                                                 if l.lower() == args[start]][0]
            
            if args[start+1] in self.conflict and card == '':
                text = 'ambiguous name (present in both param_card and run_card. Please specify'
                logger.warning(text)
                return
                
            if args[start+1] == 'default':
                default = banner_mod.RunCard(pjoin(self.me_dir,'Cards','run_card_default.dat'))
                if args[start] in default.keys():
                    self.setR(args[start],default[args[start]]) 
                else:
                    logger.info('remove information %s from the run_card' % args[start])
                    del self.run_card[args[start]]
            elif  args[start+1].lower() in ['t','.true.','true']:
                self.setR(args[start], '.true.')
            elif  args[start+1].lower() in ['f','.false.','false']:
                self.setR(args[start], '.false.')
            else:
                if args[0].startswith('sys_'):
                    val = ' '.join(args[start+1:])
                    val = val.split('#')[0]
                else: 
                    try:
                        val = eval(args[start+1])
                    except NameError:
                        val = args[start+1]
                self.setR(args[start], val)
            self.run_card.write(pjoin(self.me_dir,'Cards','run_card.dat'),
                                pjoin(self.me_dir,'Cards','run_card_default.dat'))
            
        ### PARAM_CARD WITH BLOCK NAME
        elif (args[start] in self.param_card or args[start] == 'width') \
                                                         and card != 'run_card':
            if args[start] == 'width':
                args[start] = 'decay'
                
            if args[start+1] in self.conflict and card == '':
                text = 'ambiguous name (present in both param_card and run_card. Please specify'
                logger.warning(text)
                return
            
            if args[start+1] in self.pname2block:
                all_var = self.pname2block[args[start+1]]
                key = None
                for bname, lhaid in all_var:
                    if bname == args[start]:
                        key = lhaid
                        break
                else:
                    logger.warning('%s is not part of block "%s" but "%s". please correct.' %
                                    (args[start+1], args[start], bname))
                    return
            else:
                try:
                    key = tuple([int(i) for i in args[start+1:-1]])
                except ValueError:
                    if args[start] == 'decay' and args[start+1:-1] == ['all']:
                        for key in self.param_card[args[start]].param_dict:
                            if (args[start], key) in self.restricted_value:
                                continue
                            else:
                                self.setP(args[start], key, args[-1])
                        self.param_card.write(pjoin(self.me_dir,'Cards','param_card.dat'))
                        return
                    logger.warning('invalid set command %s (failed to identify LHA information)' % line)
                    return 

            if key in self.param_card[args[start]].param_dict:
                if (args[start], key) in self.restricted_value:
                    text = "Note that this parameter seems to be ignore by MG.\n"
                    text += "MG will use instead the expression: %s\n" % \
                                      self.restricted_value[(args[start], key)]
                    text += "You need to match this expression for external program (such pythia)."
                    logger.warning(text)
                
                if args[-1].lower() in ['default', 'auto']:
                    self.setP(args[start], key, args[-1])   
                else:
                    try:
                        value = float(args[-1])
                    except Exception:
                        logger.warning('Invalid input: Expected number and not \'%s\'' \
                                                                     % args[-1])
                        return
                    self.setP(args[start], key, value)
            else:
                logger.warning('invalid set command %s' % line)
                return
            self.param_card.write(pjoin(self.me_dir,'Cards','param_card.dat'))
        
        # PARAM_CARD NO BLOCK NAME
        elif args[start] in self.pname2block and card != 'run_card':
            all_var = self.pname2block[args[start]]
            for bname, lhaid in all_var:
                new_line = 'param_card %s %s %s' % (bname, 
                   ' '.join([ str(i) for i in lhaid]), ' '.join(args[start+1:]))
                self.do_set(new_line)
            if len(all_var) > 1:
                logger.warning('This variable correspond to more than one parameter in the param_card.')
                for bname, lhaid in all_var: 
                    logger.warning('   %s %s' % (bname, ' '.join([str(i) for i in lhaid])))
                logger.warning('all listed variables have been modified')
        #INVALID
        else:
            logger.warning('invalid set command %s ' % line)
            return            
    
    def setR(self, name, value):
        logger.info('modify parameter %s of the run_card.dat to %s' % (name, value))
        self.run_card[name] = value
        
    def setP(self, block, lhaid, value):
        if isinstance(value, str):
            value = value.lower()
            if value == 'default':
                default = check_param_card.ParamCard(pjoin(self.me_dir,'Cards','param_card_default.dat'))   
                value = default[block].param_dict[lhaid].value
        
            elif value == 'auto':
                value = 'Auto'
                if block != 'decay':
                    logger.warning('Invalid input: \'Auto\' value only valid for DECAY')
                    return
            else:
                try:
                    value = float(value)
                except ValueError:
                    logger.warning('Invalid input: \'%s\' not valid intput.'% value)
                    
        logger.info('modify param_card information BLOCK %s with id %s set to %s' %\
                    (block, lhaid, value))
        self.param_card[block].param_dict[lhaid].value = value
        
      
    def help_set(self):
        '''help message for set'''
        
        logger.info('********************* HELP SET ***************************')
        logger.info("syntax: set [run_card|param_card] NAME [VALUE|default]")
        logger.info("syntax: set [param_card] BLOCK ID(s) [VALUE|default]")
        logger.info('')
        logger.info('-- Edit the param_card/run_card and replace the value of the')
        logger.info('    parameter by the value VALUE.')
        logger.info('   ')
        logger.info('-- Example:')
        logger.info('     set run_card ebeam1 4000')
        logger.info('     set ebeam2 4000')
        logger.info('     set lpp1 0')
        logger.info('     set ptj default')
        logger.info('')
        logger.info('     set param_card mass 6 175')
        logger.info('     set mass 25 125.3')
        logger.info('     set mass mh 125')
        logger.info('     set mh 125')
        logger.info('     set decay 25 0.004')
        logger.info('     set decay wh 0.004')
        logger.info('     set vmix 2 1 2.326612e-01')
        logger.info('')
        logger.info('     set param_card default #return all parameter to default')
        logger.info('     set run_card default')
        logger.info('********************* HELP SET ***************************')


    def default(self, line):
        """Default action if line is not recognized"""

        line = line.strip()
        args = line.split()
        if line == '' and self.default_value is not None:
            self.value = self.default_value        
        # check if input is a file
        elif hasattr(self, 'do_%s' % args[0]):
            self.do_set(' '.join(args[1:]))
        elif os.path.exists(line):
            self.copy_file(line)
            self.value = 'repeat'
        elif line.strip() != '0' and line.strip() != 'done' and \
            str(line) != 'EOF' and line.strip() in self.allow_arg:
            self.open_file(line)
            self.value = 'repeat'
        else:
            self.value = line
        
        return line
    
    def do_compute_widths(self, line):
        signal.alarm(0) # avoid timer if any
        path = pjoin(self.me_dir,'Cards','param_card.dat')
        pattern = re.compile(r'''decay\s+(\+?\-?\d+)\s+auto''',re.I)
        text = open(path).read()
        pdg = pattern.findall(text)
        line = '%s %s' % (line, ' '.join(pdg))
        if not '--path' in line:
            line += ' --path=%s' % path
        try:
            return self.mother_interface.do_compute_widths(line)
        except InvalidCmd, error:
            logger.error("Invalid command: %s " % error)

    def help_compute_widths(self):
        signal.alarm(0) # avoid timer if any
        return self.mother_interface.help_compute_widths()

    def complete_compute_widths(self, *args, **opts):
        signal.alarm(0) # avoid timer if any
        return self.mother_interface.complete_compute_widths(*args,**opts)
        
    def copy_file(self, path):
        """detect the type of the file and overwritte the current file"""
        
        card_name = CommonRunCmd.detect_card_type(path)
        
        if card_name == 'unknown':
            logger.warning('Fail to determine the type of the file. Not copied')
        if card_name != 'banner':
            logger.info('copy %s as %s' % (path, card_name))
            files.cp(path, pjoin(self.mother_interface.me_dir, 'Cards', card_name))                     
        elif card_name == 'banner':
            banner_mod.split_banner(path, self.mother_interface.me_dir, proc_card=False)
            logger.info('Splitting the banner in it\'s component')
            if not self.mode == 'auto':
                self.mother_interface.keep_cards(self.cards)

    def open_file(self, answer):
        """open the file"""
        me_dir = self.mother_interface.me_dir
        if answer.isdigit():
            if answer == '9':
                answer = 'plot'
            else:
                answer = self.cards[int(answer)-1]
        if not '.dat' in answer:
            if answer != 'trigger':
                path = pjoin(me_dir,'Cards','%s_card.dat' % answer)
            else:
                path = pjoin(me_dir,'Cards','delphes_trigger.dat')
        else:
            path = pjoin(me_dir, 'Cards', answer)
        self.mother_interface.exec_cmd('open %s' % path)
 

