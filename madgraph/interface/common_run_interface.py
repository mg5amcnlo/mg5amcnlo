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
    import madgraph.interface.extended_cmd as cmd
    import madgraph.various.banner as banner_mod
    import madgraph.various.misc as misc
    import madgraph.iolibs.files as files
    import madgraph.various.cluster as cluster
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
    import internal.cluster as cluster
    import internal.check_param_card as check_param_card
    import internal.files as files
    from internal import InvalidCmd, MadGraph5Error
    MADEVENT = True

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

class MadEventAlreadyRunning(InvalidCmd):
    pass

#===============================================================================
# CommonRunCmd
#===============================================================================
class CommonRunCmd(HelpToCmd, CheckValidForCmd):

    debug_output = 'ME5_debug'

    def __init__(self, me_dir, options):
        """common"""
        
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
                if not MADEVENT and pjoin(self.me_dir,'bin') not in sys.path:
                        sys.path.append(pjoin(self.me_dir,'bin'))                    
                import internal.ufomodel as ufomodel
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

    def ask_edit_cards(self, cards, fct_args, plot=True):
        """Question for cards editions (used for pgs/delphes/compute_widths)"""

        if self.force or '--no_default' in fct_args:
            return
        
        card_name = {'pgs': 'pgs_card.dat',
                     'delphes': 'delphes_card.dat',
                     'trigger': 'delphes_trigger.dat',
                     'param': 'param_card.dat'
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
        
        if plot and self.options['madanalysis_path']:
            question += '  9 / %-9s : plot_card.dat\n' % 'plot'
            possible_answer.append(9)
            possible_answer.append('plot')
            card[9] = 'plot'

        # Add the path options
        question += '  Path to a valid card.\n'
        
        # Loop as long as the user is not done.
        answer = 'no'
        while answer != 'done':
            answer = self.ask(question, '0', possible_answer, timeout=int(1.5*self.options['timeout']), 
                              path_msg='enter path')
            if answer.isdigit():
                answer = card[int(answer)]
            if answer == 'done':
                return
            if os.path.exists(answer):
                # detect which card is provide
                card_name = self.detect_card_type(answer)
                if card_name == 'unknown':
                    card_name = self.ask('Fail to determine the type of the file. Please specify the format',
                  'pgs_card.dat', choices=['pgs_card.dat', 'delphes_card.dat', 'delphes_trigger.dat'])
        
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
           shower_card.dat
        """
        
        text = open(path).read()
        text = re.findall('(<MGVersion>|CEN_max_tracker|#TRIGGER CARD|parameter set name|muon eta coverage|QES_over_ref|MSTP|Herwig\+\+|MSTU|Begin Minpts|gridpack|ebeam1|BLOCK|DECAY)', text, re.I)
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
        else:
            return 'unknown'

    ############################################################################
    def create_plot(self, mode='parton', event_path=None, output=None):
        """create the plot""" 

        madir = self.options['madanalysis_path']
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
                event_path = pjoin(self.me_dir, 'Events','unweighted_events.lhe')
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

            
            
        if not os.path.exists(event_path):
            if os.path.exists(event_path+'.gz'):
                os.system('gunzip -f %s.gz ' % event_path)
            else:
                raise self.InvalidCmd, 'Events file %s does not exits' % event_path
        
        self.update_status('Creating Plots for %s level' % mode, level = mode.lower())
               
        plot_dir = pjoin(self.me_dir, 'HTML', self.run_name,'plots_%s_%s' % (mode.lower(),tag))
                
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
                             pjoin(self.run_name, '%s_pythia_lhe_events.root' % tag)],
                            cwd=pjoin(self.me_dir,'Events'))              
                except:
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
            self.ask_edit_cards(['pgs'], args)
            
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
        except:
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
            except:
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
        if not (no_default or self.force):
            self.ask_edit_cards(['delphes', 'trigger'], args)
            
        self.update_status('Running Delphes', level=None)  
        # Wait that the gunzip of the files is finished (if any)
        if lock:
            lock.acquire()


 
        delphes_dir = self.options['delphes_path']
        tag = self.run_tag
        if os.path.exists(pjoin(self.me_dir, 'Source', 'banner_header.txt')):
            self.banner.add(pjoin(self.me_dir, 'Cards','delphes_card.dat'))
            self.banner.add(pjoin(self.me_dir, 'Cards','delphes_trigger.dat'))
            self.banner.write(pjoin(self.me_dir, 'Events', self.run_name, '%s_%s_banner.txt' % (self.run_name, tag)))
        
        cross = self.results[self.run_name].get_current_info()['cross']
                    
        delphes_log = pjoin(self.me_dir, 'Events', self.run_name, "%s_delphes.log" % tag)
        self.cluster.launch_and_wait('../bin/internal/run_delphes', 
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
            text = open(pjoin(Pdir, 'born_leshouche.inc')).read()
            group = pat.findall(text)
            for particles in group:
                particles = particles.split(',')
                pids.update(set(particles[nb_init:]))
        
        return pids
                
                
                
        
  
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
            self.cluster = cluster.MultiCore(nb_core, 
                                     temp_dir=self.options['cluster_temp_path'])
            
        if self.cluster_mode == 1:
            opt = self.options
            cluster_name = opt['cluster_type']
            self.cluster = cluster.from_name[cluster_name](opt['cluster_queue'],
                                                        opt['cluster_temp_path'])

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


  

    def update_status(self, status, level, makehtml=True, force=True, 
                      error=False, starttime = None, update_results=False):
        """ update the index status """
        
        if makehtml and not force:
            if hasattr(self, 'next_update') and time.time() < self.next_update:
                return
            else:
                self.next_update = time.time() + 3
        
        if isinstance(status, str):
            if '<br>' not  in status:
                logger.info(status)
        elif starttime:
            running_time = time.time()-starttime
            if running_time < 1e-2:
                running_time = ''
            elif running_time < 10:
                running_time = '[ %.2gs ]' % running_time
            elif 60 > running_time >= 10:
                running_time = '[ %.3gs ]' % running_time
            elif 3600 > running_time >= 60:
                running_time = '[ %im %is ]' % (running_time // 60, int(running_time % 60))
            else:
                running_time = '[ %ih %im ]' % (running_time // 3600, (running_time//60 % 60))
                
            logger.info(' Idle: %s,  Running: %s,  Completed: %s %s' % \
                       (status[0], status[1], status[2], running_time))
        else: 
            logger.info(' Idle: %s,  Running: %s,  Completed: %s' % status[:3])
        
        if update_results:
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
                
            if self.options.has_key('mg5_path'):
                MG5DIR = self.options['mg5_path']
                config_file = pjoin(MG5DIR, 'input', 'mg5_configuration.txt')
                self.set_configuration(config_path=config_file, final=False,initdir=MG5DIR)
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
