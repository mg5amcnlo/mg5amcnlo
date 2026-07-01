################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
""" Command interface for MadSpin """
from __future__ import division
from __future__ import absolute_import
import collections
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import glob
from itertools import chain, filterfalse, product

pjoin = os.path.join
if '__main__' == __name__:
    import sys
    sys.path.append(pjoin(os.path.dirname(__file__), '..'))

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as master_interface
import madgraph.interface.madevent_interface as madevent_interface
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.interface.reweight_interface as rwgt_interface
import madgraph.various.misc as misc
import madgraph.iolibs.files as files
import madgraph.iolibs.export_v4 as export_v4
import madgraph.various.banner as banner
import madgraph.various.lhe_parser as lhe_parser

import models.import_ufo as import_ufo
import models.check_param_card as check_param_card
import MadSpin.decay as madspin

logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print

class MadSpinOptions(banner.ConfigFile):
    
    def default_setup(self):

        self.add_param("max_weight", -1)
        self.add_param('curr_dir', os.path.realpath(os.getcwd()))
        self.add_param('Nevents_for_max_weight', 0)
        self.add_param("max_weight_ps_point", 400)
        self.add_param('BW_cut', -1)
        self.add_param('nb_sigma', 0.)
        self.add_param('ms_dir', '')
        self.add_param('max_running_process', 100)
        self.add_param('onlyhelicity', False)
        self.add_param('ME_mode', 'auto', allowed=['auto', 'decay_chain', 'density'])
        self.add_param('spinmode', "PA", allowed=['full','madspin','none','onshell','PA'])
        self.add_param('use_old_dir', False, comment='should be use only for faster debugging')
        self.add_param('run_card', '' , comment='define cut for spinmode==none. Path to run_card to use')
        self.add_param('fixed_order', False, comment='to activate fixed order handling of counter-event')
        self.add_param('seed', 0, comment='control the seed of madspin')
        self.add_param('cross_section', {'__type__':0.}, comment="forcing normalization of cross-section after MS (for none/onshell)" )
        self.add_param('new_wgt', 'cross-section' ,allowed=['cross-section', 'BR'], comment="if not consistent number of particles, choose what to do for the weight. (BR: means local according to number of part, cross use the force cross-section")
        self.add_param('input_format', 'auto', allowed=['auto','lhe', 'hepmc', 'lhe_no_banner'])
        self.add_param('frame_id', 6)
        self.add_param('global_order_coupling', '')
        self.add_param('identical_particle_in_prod_and_decay', 'average')
        self.add_param('beampol', [0.5, 0.5], comment='beam polarization')
        self.add_param('density_debug', False, comment='Turn on check against full ME calculation')
        self.add_param('density_tolerance', 1E-4, comment='Tolerance for deviation between density and full ME')
        self.add_param('decay_event_mult', 1E0, comment='Produce more events than needed so that MadSpin does not have to regenerate decay events')
        self.add_param('density_keep_jacobian', False, comment='keep track of the phase-space volume change related to the offshell reshuffling')
        self.add_param('density_do_reshuffle', True, comment='In density mode with pole approximation, sample Breit-Wigner masses and reshuffle the accepted event. Disable to keep pure onshell kinematics.')
        self.add_param('density_pole_approximation', True, comment='In density mode, use the multiple approximation, leaving offshell as pure BW (via reshuffling). Set on False, means using offshell matrix-element that impacts the shape of the Breit-Wigner. False is equivalent to the old Madspin mode/decay chain syntax of MadGraph')

    ############################################################################
    ##  Special post-processing of the options                                ## 
    ############################################################################
    def post_set_ms_dir(self, value, change_userdefine, raiseerror, *opts):
        """ special handling for set ms_dir """
        
        self.__setitem__('curr_dir', value, change_userdefine=change_userdefine)
        
    ############################################################################
    def post_set_seed(self, value, change_userdefine, raiseerror):
        """ special handling for set seed """
        
        if not hasattr(random, 'mg_seedset'):
            random.seed(self['seed'])  
            random.mg_seedset = self['seed']  

    ############################################################################        
    def post_set_run_card(self, value, change_userdefine, raiseerror, *opts):
        """ special handling for set run_card """
        
        if value == 'default':
            self.run_card = None
        elif not value:
            self.run_card = None
        elif os.path.isfile(value):
            self.run_card = banner.RunCard(value)
        else:
            misc.sprint(value)
            args = value.split()
            if  len(args) >1:
                if not hasattr(self, 'run_card'):
                    misc.sprint("init run_card")
                    self.run_card =  banner.RunCardLO()
                    self.run_card.remove_all_cut()
                self.run_card[args[0]] = ' '.join(args[1:])
            else:
                raise Exception("wrong syntax for \"set run_card %s\"" % value)
            
        
    ############################################################################
    def post_fixed_order(self, value, change_userdefine, raiseerror):
        """ special handling for set fixed_order """
        
        if value:
            logger.warning('Fix order madspin fails to have the correct scale information. This can bias the results!')
            logger.warning('Not all functionalities of MadSpin handle this mode correctly (only onshell mode so far).')

    ############################################################################
    def post_identical_particle_in_prod_and_decay(self, value, change_userdefine, raiseerror):
        """ special handling for set fixed_order """
        if value not in ["crash", 'average', 'max', 'first']:
            raise Exception("value %s not supported for this parameter identical_in_prod_and_decay")

class MadSpinInterface(extended_cmd.Cmd):
    """Basic interface for madspin"""

    prompt = 'MadSpin>'
    debug_output = 'MS_debug'

    # Process-wide counter used to make each MadSpinInterface instance use
    # its own ``madspin_me`` output subdirectory. The actual fortran/f2py
    # artefacts (the ``.so`` extension and the ``liball_2me`` shared
    # library) are recompiled to that subdirectory each call, and
    # ``dlopen`` caches loaded libraries by absolute path: without a fresh
    # path the second MadSpin call in the same process (e.g. inline
    # MadSpin followed by ``decay_events``) keeps the *first* call's
    # matrix elements in memory and ``pdg2prefix`` ends up missing the
    # second card's decay channels (KeyError in ``get_pdir``).
    _ms_run_counter = 0


    @misc.mute_logger()
    def __init__(self, event_path=None, *completekey, **stdin):
        """initialize the interface with potentially an event_path"""

        cmd_logger.info('************************************************************')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('*           W E L C O M E  to  M A D S P I N               *')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('************************************************************')
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)

        MadSpinInterface._ms_run_counter += 1
        self._ms_run_id = MadSpinInterface._ms_run_counter
        # First call keeps the historical ``madspin_me`` name (the
        # decay/import code still hard-codes that in a few places and the
        # vast majority of MadSpin uses only sees one run per process).
        # Subsequent calls use a unique suffix so dlopen sees a fresh
        # file path.
        if self._ms_run_id == 1:
            self.ms_me_subdir = 'madspin_me'
        else:
            self.ms_me_subdir = 'madspin_me_%d' % self._ms_run_id

        self.decay = madspin.decay_misc()
        self.model = None
        #self.mode = "madspin" # can be flat/bridge change the way the decay is done.
        #                      # note amc@nlo does not support bridge.
        
        self.options = MadSpinOptions()
        
        self.events_file = None
        self.decay_processes = {}
        self.list_branches = {}
        self.to_decay={}
        self.mg5cmd = master_interface.MasterCmd()
        self.seed = None
        self.err_branching_ratio = 0
        self.me_run_name = "" # Events diretory name where to stotre the events (used by madevent) not use internally
        self.all_iden = {}
        
        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path)
            
    
    def setup_for_pure_decay(self):
        """this is for spinmode=None -> simple decay
           We go here if they are no banner.
           -> this requires that a command import model appears in the card!
        """

        logger.info("Setup the code for pure decay mode")
        self.proc_option = []
        self.final_state_full = ''
        self.final_state_compact = ''
        self.prod_branches = ''
        self.final_state = set()

    def _load_f2py_matrix_module(self, sp_path):
        """Load the freshly-compiled ``all_matrix2py`` extension under
        ``sp_path``.

        Each MadSpin run compiles its matrix elements into its own
        ``madspin_me_<N>`` subdir, and (from the second call onwards)
        ``decay.compile()`` overrides the makefile's ``PROCNAME`` so the
        resulting Fortran shared library
        (``liball<PROCNAME>_2me.{so,dylib}``) has a unique SONAME /
        install_name. The combination of a unique wrapper path *and* a
        unique dependent-library identity is what stops the dynamic
        loader from returning the first call's already-loaded matrix
        elements on the second call.

        This helper just picks the loadable ``.so`` and loads it via
        ``importlib.util.spec_from_file_location`` to bypass the
        ``sys.modules`` cache (which would otherwise short-circuit
        ``__import__('all_matrix2py')`` to the first call's module
        object).
        """
        import importlib.util
        import glob

        # The actual loadable file is the cpython-tagged ``.so``; on some
        # builds the unsuffixed ``all_matrix2py.so`` is a 0-byte stub. Pick
        # the largest matching file so we always load real code.
        patterns = [
            'all_matrix2py.cpython*.so',
            'all_matrix2py.cpython*.dylib',
            'all_matrix2py.so',
            'all_matrix2py.dylib',
        ]
        candidates = []
        for pat in patterns:
            for hit in glob.glob(pjoin(sp_path, pat)):
                if os.path.getsize(hit) > 0:
                    candidates.append(hit)
        if not candidates:
            # Fall back to the historical ``__import__`` so we at least
            # produce a meaningful error if nothing got compiled.
            return __import__('all_matrix2py')
        candidates.sort(key=os.path.getsize, reverse=True)
        so_path = candidates[0]

        # Load via spec_from_file_location to bypass the sys.modules cache
        # while keeping the module name as ``all_matrix2py`` (the .so's
        # PyInit_all_matrix2py init symbol is baked in at compile time).
        spec = importlib.util.spec_from_file_location('all_matrix2py', so_path)
        if spec is None or spec.loader is None:
            return __import__('all_matrix2py')
        mymod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mymod)
        return mymod

    def _log_lhe_timers(self):
        if not getattr(lhe_parser, "_ENABLE_LHE_TIMERS", False):
            return
        timers, counts = lhe_parser.get_lhe_timers()
        if not timers:
            print("LHE parser timing enabled but no samples were collected.")
            return
        print("LHE parser timing summary:")
        for key in sorted(timers):
            total = timers[key]
            count = counts.get(key, 1)
            print("  %s: %.6fs total over %d call(s) (avg %.6fs)" %
                  (key, total, count, total / max(1, count)))
        
     
    def do_import(self, inputfile):
        """import the event file"""
        
        args = self.split_arg(inputfile)
        if not args:
            return self.InvalidCmd, 'import requires arguments'
        elif args[0] == 'model':
            return self.import_model(args[1:])
        
        # change directory where to write the output
        self.options['curr_dir'] = os.path.realpath(os.path.dirname(inputfile))
        if os.path.basename(os.path.dirname(os.path.dirname(inputfile))) == 'Events':
            self.options['curr_dir'] = pjoin(self.options['curr_dir'], 
                                                      os.path.pardir, os.pardir)
        
        if not os.path.exists(inputfile):
            if inputfile.endswith('.gz'):
                if not os.path.exists(inputfile[:-3]):
                    misc.sprint(os.getcwd(), os.listdir('.'), inputfile, os.path.exists(inputfile), os.path.exists(inputfile[:-3]))
                    raise self.InvalidCmd('No such file or directory : %s' % inputfile)
                else: 
                    inputfile = inputfile[:-3]
            elif os.path.exists(inputfile + '.gz'):
                inputfile = inputfile + '.gz'
            else: 
                raise self.InvalidCmd('No such file or directory : %s' % inputfile)

        self.inputfile = inputfile
        if self.options['spinmode'] == 'none' and \
           (self.options['input_format'] not in ['lhe','auto'] or 
             (self.options['input_format'] == 'auto' and '.lhe'  not in inputfile[-7:])):  
            self.banner = banner.Banner()
            self.setup_for_pure_decay()
            return  
        
        if inputfile.endswith('.gz'):
            misc.gunzip(inputfile)
            inputfile = inputfile[:-3]
        # Read the banner of the inputfile
        self.events_file = open(os.path.realpath(inputfile))
        self.banner = banner.Banner(self.events_file)


        # Check the validity of the banner:
        if 'slha' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')

        
        if 'madspin' in self.banner:
            raise self.InvalidCmd('This event file was already decayed by MS. This is not possible to add to it a second decay')
        
        if 'mgruncard' in self.banner:
            run_card = self.banner.charge_card('run_card')
            if not self.options['Nevents_for_max_weight']:
                nevents = run_card['nevents']
                N_weight = max([75, int(3*nevents**(1/3))])
                self.options['Nevents_for_max_weight'] = N_weight
                N_sigma = max(4.5, math.log(nevents,7.7))
                self.options['nb_sigma'] = N_sigma
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = float(self.banner.get_detail('run_card', 'bwcutoff'))
                if self.options['BW_cut'] > 25:
                    logger.critical("value of bwcutoff set to %s from the input file. This is much too large value for Madspin and the validity of the Narrow-width-Approximation. Please ensure that you overwrite that value via \"set BW_cut X\"  to a smaller value (like X=10)", self.options['BW_cut'])
            
            if isinstance(run_card, banner.RunCardLO):
                run_card.update_system_parameter_for_include()
                self.options['frame_id'] = run_card['frame_id']
                beampol = [.5,.5]
                beampol[0] =  (-1./200)* run_card['polbeam1'] + 0.5
                beampol[1] =  (-1./200)* run_card['polbeam2'] + 0.5
                self.options['beampol'] = beampol
            else:
                self.options['frame_id'] = 6
                self.options['beampol'] = [.5,.5]

        else:
            if not self.options['Nevents_for_max_weight']:
                self.options['Nevents_for_max_weight'] = 75
                self.options['nb_sigma'] = 4.5
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = 15.0
                
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception(msg)
        process, option = mg_interface.MadGraphCmd.split_process_line(process)
        self.proc_option = option
        
        logger.info("process: %s" % process)
        logger.info("options: %s" % option)

        if not hasattr(self,'multiparticles_ms'):
            for key, value in self.banner.get_detail('proc_card','multiparticles'):
                try:
                    self.do_define('%s = %s' % (key, value))
                except self.mg5cmd.InvalidCmd:  
                    pass
                
        # Read the final state of the production process:
        #     "_full" means with the complete decay chain syntax 
        #     "_compact" means without the decay chain syntax 
        self.final_state_full = process[process.find(">")+1:]
        self.final_state_compact, self.prod_branches=\
                 self.decay.get_final_state_compact(self.final_state_full)
                
        # Load the model
        complex_mass = False   
        has_cms = re.compile(r'''set\s+complex_mass_scheme\s*(True|T|1|true|$|;)''')
        for line in self.banner.proc_card:
            if line.startswith('set'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                if has_cms.search(line):
                    complex_mass = True
        
          
        info = self.banner.get('proc_card', 'full_model_line')
        if '-modelname' in info:
            mg_names = False
        else:
            mg_names = True
        model_name = self.banner.get('proc_card', 'model')
        if model_name:
            model_name = os.path.expanduser(model_name)
            self.load_model(model_name, mg_names, complex_mass)
        else:
            raise self.InvalidCmd('Only UFO model can be loaded in MadSpin.')
        # check particle which can be decayed:
        self.final_state = set()
        final_model = False
        for line in self.banner.proc_card:
            line = ' '.join(line.strip().split())
            if line.startswith('generate'):
                self.final_state.update(self.mg5cmd.get_final_part(line[8:]))
            elif line.startswith('add process'):
                self.final_state.update(self.mg5cmd.get_final_part(line[11:]))
            elif line.startswith('define'):
                try:
                    self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                except self.mg5cmd.InvalidCmd:
                    if final_model:
                        raise
                    else:
                        key = line.split()[1]
                        if key in self.multiparticles_ms:
                            del self.multiparticles_ms[key]            
            elif line.startswith('set') and not line.startswith('set gauge'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
            elif line.startswith('import model'):
                if model_name in line:
                    final_model = True
                    
                
    def import_model(self, args):
        """syntax: import model NAME CARD_PATH
            args didn't include import model"""
        
        bypass_check = False
        if '--bypass_check' in args:
            args.remove('--bypass_check')
            bypass_check = True
        if len(args) == 1:  
            logger.warning("""No param_card defined for the new model. We will use the default one but this might completely wrong.""")
        elif len(args) != 2:
            return self.InvalidCmd, 'import model requires two arguments'
        
        model_name = args[0]
        self.load_model(model_name, False, False)
        
        if len(args) == 2:
            card = args[1]
            if not os.path.exists(card):
                raise self.InvalidCmd('%s: no such file' % card)
        else:
            card = "madspin_param_card.dat"
            export_v4.UFO_model_to_mg4.create_param_card_static(self.model,
                                card, rule_card_path=None)

        

        #Check the param_card
        if not (bypass_check or self.options['input_format'] in ['hepmc', 'lhe_no_banner']):
            if not hasattr(self.banner, 'param_card'):
                self.banner.charge_card('slha')
            param_card = check_param_card.ParamCard(card)
            # checking that all parameter of the old param card are present in 
            #the new one with the same value
            try:
                diff = self.banner.param_card.create_diff(param_card)
            except Exception:
                raise self.InvalidCmd('''The two param_card seems very different. 
    So we prefer not to proceed. If you are sure about what you are doing, 
    you can use the command \'import model MODELNAME PARAM_CARD_PATH --bypass_check\'''')
            if diff:
                raise self.InvalidCmd('''Original param_card differs on some parameters:
    %s
    Due to those differences, we prefer not to proceed. If you are sure about what you are doing, 
    you can use the command \'import model MODELNAME PARAM_CARD_PATH --bypass_check\''''
    % diff.replace('\n','\n    '))
   
   
                
        #OK load the new param_card (but back up the old one)
        if 'slha' in self.banner:
            self.banner['slha_original'] = self.banner['slha']
        self.banner['slha'] = open(card).read()
        if hasattr(self.banner, 'param_card'):
            del self.banner.param_card
        self.banner.charge_card('slha')
                


    @extended_cmd.debug()
    def complete_import(self, text, line, begidx, endidx):
        "Complete the import command"
        
        args=self.split_arg(line[0:begidx])
        
        
        if len(args) == 1:
            base_dir = '.'
        else:
            base_dir = args[1]
        
        return self.path_completion(text, base_dir)
        
        # Directory continuation
        if os.path.sep in args[-1] + text:
            return self.path_completion(text,
                                    pjoin(*[a for a in args if \
                                                      a.endswith(os.path.sep)]))

    def do_decay(self, decaybranch):
        """add a process in the list of decayed particles"""
        
        #if self.model and not self.model['case_sensitive']:
        #    decaybranch = decaybranch.lower()

        if self.options['spinmode'] not in  ['full','madspin'] and '{' in decaybranch:
            if self.options['spinmode'] == 'none':
                logger.warning("polarization option used with spinmode=none. The polarization definition will be done according to the rest-frame of the decaying particles (which is likely not what you expect).")
            else:
                logger.warning("polarization option used with spinmode=onshell. This combination is not validated and is by construction using sub-optimal method which can likely lead to bias in some situation. Use at your own risk.")
        if "=" in decaybranch and self.options['spinmode'] in['full','madspin']:
            logger.warning("Note that coupling order restriction are not associated to specific Branching Ratio. The total cross-section might therefore use the wrong branching ratio.")
        decay_process, init_part = self.decay.reorder_branch(decaybranch)
        if init_part not in self.list_branches:
            self.list_branches[init_part] = []
        self.list_branches[init_part].append(decay_process)
        del decay_process, init_part    
        
    
    def check_set(self, args):
        """checking the validity of the set command"""
        
        if len(args) < 2:
            if args and '=' in args[0]:
                name, value = args[0].split('=')
                args[0]= name
                args.append(value)
            elif len(args) == 1 and args[0] in ['onlyhelicity']:
                args.append('True')
            else:
                raise self.InvalidCmd('set command requires at least two argument.')
        
        if args[1].strip() == '=':
            args.pop(1)
        
        valid = ['max_weight','seed','curr_dir', 'spinmode', 'run_card']
        if args[0] not in self.options and args[0] not in valid:
            raise self.InvalidCmd('Unknown options %s' % args[0])        
    
        if args[0] == 'max_weight':
            try:
                args[1] = float(args[1].replace('d','e'))
            except ValueError:
                raise self.InvalidCmd('second argument should be a real number.')
        
        elif args[0] == 'curr_dir':
            if not os.path.isdir(args[1]):
                raise self.InvalidCmd('second argument should be a path to a existing directory')
        
        elif args[0] == "spinmode":
            if args[1].lower() not in ["full", "onshell", "none", "madspin", "density", "pa"]:
                raise self.InvalidCmd("spinmode can only take one of those 5 values: full/onshell/none/density/PA")
             
        elif args[0] == "run_card":
            if self.options['spinmode'] == "madspin":
                raise self.InvalidCmd("edition of the run_card is not allowed within normal mode")
            if "=" in args:
                args.remove("=")
            if len(args)==2 and "=" in args[1]:
                data = args.pop(1)
                arg, value = data.split("=")
                args.append(arg)
                args.append(value)
        elif args[0] == 'Nevents_for_max_weigth':
            args[0] = 'Nevents_for_max_weight'
        
    def do_set(self, line):
        """ add one of the options """
        
        args = self.split_arg(line)
        self.check_set(args)

        self.options[args[0]] = ' '.join(args[1:])
        

    def complete_set(self,  text, line, begidx, endidx):
        

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            opts = list(self.options.keys()) + ['seed', "spinmode"]
            return self.list_completion(text, opts) 
        elif len(args) == 2:
            if args[1] == 'ms_dir':
                return self.path_completion(text, '.', only_dirs = True)
        elif args[1] == 'ms_dir':
            curr_path = pjoin(*[a for a in args \
                                                   if a.endswith(os.path.sep)])
            return self.path_completion(text, curr_path, only_dirs = True)
        elif args[1] == "spinmode":
            return self.list_completion(text, ["full","onshell", "none"], line)
         
    def help_set(self):
        """help the set command"""
        
        print('syntax: set OPTION VALUE')
        print('')
        print('-- assign to a given option a given value')
        print('   - set max_weight VALUE: pre-define the maximum_weight for the reweighting')
        print('   - set seed VALUE: fix the value of the seed to a given value.')
        print('       by default use the current time to set the seed. random number are')
        print('       generated by the python module random using the Mersenne Twister generator.')
        print('       It has a period of 2**19937-1.')
        print('   - set max_running_process VALUE: allow to limit the number of open file used by the code')
        print('       The number of running is raising like 2*VALUE')
        print('   - set spinmode=none: mode with simple file merging. No spin correlation attempt.')
        print('       This mode allows 3 (and more) body decay.')
    
    def do_define(self, line):
        """ """

        try:
            self.mg5cmd.exec_cmd('define %s' % line)
        except:
            #cleaning if the error is recover later
            key = line.split()[0]
            if hasattr(self, 'multiparticles_ms') and key in self.multiparticles_ms:
                del self.multiparticles_ms[key]
            raise
           
        self.multiparticles_ms = dict([(k,list(pdgs)) for k, pdgs in \
                                        self.mg5cmd._multiparticles.items()])
    
    
    def update_status(self, *args, **opts):
        """ """
        pass # function overwritten for MS launched by ME
    
    def complete_define(self, *args):
        """ """
        try:
            return self.mg5cmd.complete_define(*args)
        except Exception as error:
            misc.sprint(error)
            
    def complete_decay(self, *args):
        """ """
        try:
            return self.mg5cmd.complete_generate(*args)
        except Exception as error:
            misc.sprint(error)
            
    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.list_branches and not self.options['onlyhelicity']:
            raise self.InvalidCmd("Nothing to decay ... Please specify some decay")
        if not self.events_file:
            raise self.InvalidCmd("No events files defined.")
        
        # Validity check. Need lhe version 3 if matching is on
        if self.banner.get("run_card", "lhe_version") < 3 and \
            self.banner.get("run_card", "ickkw") > 0:
            raise Exception("MadSpin requires LHEF version 3 when running with matching/merging")

    def help_launch(self):
        """help for the launch command"""
        
        print('''Running Madspin on the loaded events, following the decays enter
        An example of a full run is the following:
        import ../mssm_events.lhe.gz
        define sq = ur ur~
        decay go > sq j
        launch
        ''')
        
        self.parser_launch.print_help()

    def parser_launch(self):
        usage = """launch [-n RUN_NAME]   
        """
        parser = misc.OptionParser(usage=usage)
        parser.add_option("-n", "--name",
                  default="",
                  help="When NOT run in standalone instruct MG5aMC where to store the events file")
        return parser
    
    def parse_launch(self, line):
        
        args = self.split_arg(line)
        return self.parser_launch().parse_args(args)
        

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""
        
        (options, args) = self.parse_launch(line)
        if getattr(lhe_parser, "_ENABLE_LHE_TIMERS", False):
            lhe_parser.reset_lhe_timers()
        
        if options.name:
            self.me_run_name = options.name # Only use by MG5aMC
        else:
            self.me_run_name = ''

        if self.options['onlyhelicity']:
            self.options['spinmode'] = 'full'
            self.options['ME_mode'] = 'decay_chain'



        if self.options["spinmode"] in ["none"]:
            out = self.run_bridge(line)
            self._log_lhe_timers()
            return out
        elif self.options["spinmode"] == "onshell":
            if self.options['ME_mode'] in ['auto', 'decay_chain']:
                out = self.run_onshell(line)
            else:
                out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif self.options["spinmode"] == "PA":
            self.options['density_pole_approximation'] = True
            out = self.run_onshell(line, density_method=True)
            self._log_lhe_timers()
            return out
        elif self.options["spinmode"] == "madspin":
            # legacy MadSpin / decay-chain path: fall through to decay_all_events below
            pass
        elif self.options["spinmode"] == "full":
            if self.options['ME_mode'] in ['auto', 'density']:
                self.options['density_pole_approximation'] = False 
                out = self.run_onshell(line, density_method=True)
                self._log_lhe_timers()
                return out
            else:
                pass
        elif self.options["spinmode"] == "bridge":
            raise Exception("Bridge mode not available.")
        else:
            raise Exception("spinmode %s not supported" % self.options["spinmode"])
        
        if self.options['ms_dir'] and os.path.exists(pjoin(self.options['ms_dir'], 'madspin.pkl')):
            out = self.run_from_pickle()
            self._log_lhe_timers()
            return out
        
    
        args = self.split_arg(line)
        self.check_launch(args)
        for part in list(self.list_branches.keys()):
            if part in self.mg5cmd._multiparticles:
                if any(pid in self.final_state for pid in self.mg5cmd._multiparticles[part]):
                    break
            else:
                try:
                    pid = self.mg5cmd._curr_model.get('name2pdg')[part]
                except KeyError:
                    pid = self.mg5cmd._curr_model.get('name2pdg')[part.lower()]
                    self.list_branches[part.lower()] = self.list_branches[part]
                    del self.list_branches[part]
                    particle = self.mg5cmd._curr_model.get_particle(pid)
                    if particle.get('antiname').upper() in self.list_branches:
                        self.list_branches[particle.get('antiname').lower()] = \
                            self.list_branches[particle.get('antiname').upper()]
                        del self.list_branches[particle.get('antiname').upper()]
                if pid in self.final_state:
                    break
        else:
            if not self.options['onlyhelicity']:
                logger.info("Nothing to decay ...")
                return
        
        if self.options['BW_cut'] > 100:
            raise Exception("BW_cut parameter is much too large (>100) for narrow width approximation. Please set it up to a smaller value in your madspin_card.dat")

        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.options['seed']:
            self.options['seed'] = random.randint(0, int(30081*30081))
            #self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.options['seed'])
            self.history.insert(0, 'set seed %s' % self.options['seed'])

        if self.options['seed'] > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.options['seed']) + ' > 30081*30081'
            raise Exception(msg)

        #self.options['seed'] = self.seed
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)
            
        time_me_generation = time.time()
        self.update_status('generating Madspin matrix element')
        generate_all = madspin.decay_all_events(self, self.banner, self.events_file, 
                                                    self.options)
        logger.critical(f"Time for ME: {time.time()-time_me_generation:.2f} sec")        
        self.update_status('running MadSpin')
        generate_all.run()
                        
        self.branching_ratio = generate_all.branching_ratio
        self.cross = generate_all.cross
        self.error = generate_all.error
        self.efficiency = generate_all.efficiency
        try:
            self.err_branching_ratio = generate_all.err_branching_ratio
        except Exception:
            self.err_branching_ratio = 0
            
        evt_path = self.events_file.name
        try:
            self.events_file.close()
        except:
            pass
        misc.gzip(evt_path)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        misc.gzip(pjoin(self.options['curr_dir'],'decayed_events.lhe'),
                  stdout=decayed_evt_file)
        if not self.mother:
            logger.info("Decayed events have been written in %s.gz" % decayed_evt_file)

        # Now arxiv the shower card used if RunMaterial is present
        ms_card_path = pjoin(self.options['curr_dir'],'Cards','madspin_card.dat')
        run_dir = os.path.realpath(os.path.dirname(decayed_evt_file))
        if os.path.exists(ms_card_path):
            if os.path.exists(pjoin(run_dir,'RunMaterial.tar.gz')):
                misc.call(['tar','-xzpf','RunMaterial.tar.gz'], cwd=run_dir)
                base_path = pjoin(run_dir,'RunMaterial')
            else:
                base_path = pjoin(run_dir)

            evt_name = os.path.basename(decayed_evt_file).replace('.lhe', '')
            ms_card_to_copy = pjoin(base_path,'madspin_card_for_%s.dat'%evt_name)
            count = 0    
            while os.path.exists(ms_card_to_copy):
                count += 1
                ms_card_to_copy = pjoin(base_path,'madspin_card_for_%s_%d.dat'%\
                                                               (evt_name,count))
            files.cp(str(ms_card_path),str(ms_card_to_copy))
            
            if os.path.exists(pjoin(run_dir,'RunMaterial.tar.gz')):
                misc.call(['tar','-czpf','RunMaterial.tar.gz','RunMaterial'], 
                                                                    cwd=run_dir)
                shutil.rmtree(pjoin(run_dir,'RunMaterial'))
        self._log_lhe_timers()

    def run_from_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        
        generate_all = save_load_object.load_from_file(pjoin(self.options['ms_dir'], 'madspin.pkl'))
        
        #restore data passed to string to help pickle
        generate_all.all_decay = eval(generate_all.all_decay)
        for me in generate_all.all_ME:
            for d in generate_all.all_ME[me]['decays']:
                if isinstance(d['decay_struct'], str):
                    d['decay_struct'] = eval(d['decay_struct'])


        # Re-create information which are not save in the pickle.
        generate_all.evtfile = self.events_file
        generate_all.curr_event = madspin.Event(self.events_file, self.banner ) 
        generate_all.mgcmd = self.mg5cmd
        generate_all.mscmd = self 
        #generate_all.pid2width = lambda pid: generate_all.banner.get('param_card', 'decay', abs(pid)).value
        #generate_all.pid2mass = lambda pid: generate_all.banner.get('param_card', 'mass', abs(pid)).value
        if generate_all.path_me != self.options['ms_dir']:
            for decay in generate_all.all_ME.values():
                decay['path'] = decay['path'].replace(generate_all.path_me, self.options['ms_dir'])
                for decay2 in decay['decays']:
                    if decay2['path']: 
                        decay2['path'] = decay2['path'].replace(generate_all.path_me, self.options['ms_dir'])
            generate_all.path_me = self.options['ms_dir'] # directory can have been move
            generate_all.ms_dir = generate_all.path_me
        
        if not hasattr(self.banner, 'param_card'):
            self.banner.charge_card('slha')
        
        # Special treatment for the mssm. Convert the param_card to the correct
        # format
        if self.banner.get('model').startswith('mssm-') or self.banner.get('model')=='mssm':
            self.banner.param_card = check_param_card.convert_to_mg5card(\
                    self.banner.param_card, writting=False)
            
        for name, block in self.banner.param_card.items():
            if name.startswith('decay'):
                continue
                        
            orig_block = generate_all.banner.param_card[name]
            if block != orig_block:                
                raise Exception("""The directory %s is specific to a mass spectrum. 
                Your event file is not compatible with this one. (Different param_card: %s different)
                orig block:
                %s
                new block:
                %s""" \
                % (self.options['ms_dir'], name, orig_block, block))

        #replace init information
        generate_all.banner['init'] = self.banner['init']

        #replace run card if present in header (to make sure correct random seed is recorded in output file)
        if 'mgruncard' in self.banner:
            generate_all.banner['mgruncard'] = self.banner['mgruncard']   
        
        # NOW we have all the information available for RUNNING
        
        if self.options['seed']:
            #seed is specified need to use that one:
            open(pjoin(self.options['ms_dir'],'seeds.dat'),'w').write('%s\n'%self.options['seed'])
            #remove all ranmar_state
            for name in misc.glob(pjoin('*', 'SubProcesses','*','ranmar_state.dat'), 
                                                        self.options['ms_dir']):
                os.remove(name)    
        
        generate_all.ending_run()
        self.branching_ratio = generate_all.branching_ratio
        self.cross = generate_all.cross
        self.error = generate_all.error
        self.efficiency = generate_all.efficiency
        try:
            self.err_branching_ratio = generate_all.err_branching_ratio
        except Exception:
            # might not be define in some gridpack mode
            self.err_branching_ratio = 0 
        evt_path = self.events_file.name
        try:
            self.events_file.close()
        except:
            pass
        misc.gzip(evt_path)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        misc.gzip(pjoin(self.options['curr_dir'],'decayed_events.lhe'),
                  stdout=decayed_evt_file)
        if not self.mother:
            logger.info("Decayed events have been written in %s.gz" % decayed_evt_file)    
    
    

    def run_bridge(self, line):
        """Run the Bridge Algorithm"""
        
        # 1. Read the event file to check which decay to perform and the number
        #   of event to generate for each type of particle.
        # 2. Generate the events requested
        # 3. perform the merge of the events.
        #    if not enough events. re-generate the missing one.
        
        args = self.split_arg(line)


        asked_to_decay = set()
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                for pdg in self.mg5cmd._multiparticles[part]:
                    asked_to_decay.add(pdg)
            else:
                asked_to_decay.add(self.mg5cmd._curr_model.get('name2pdg')[part])

        #0. Define the path where to write the file
        self.path_me = os.path.realpath(self.options['curr_dir']) 
        if self.options['ms_dir']:
            self.path_me = os.path.realpath(self.options['ms_dir'])
            if not os.path.exists(self.path_me):
                os.mkdir(self.path_me) 
        else:
            # cleaning
            for name in misc.glob("decay_*_*", self.path_me):
                shutil.rmtree(name)

        if self.events_file:
            self.events_file.close()
            filename = self.events_file.name
        else:
            filename = self.inputfile

        if self.options['input_format'] == 'auto':
            if '.lhe' in filename :
                self.options['input_format']  = 'lhe'
            elif '.hepmc' in filename:
                self.options['input_format']  = 'hepmc'
            else:
                raise Exception("fail to recognized input format automatically")
                
        if self.options['input_format'] in ['lhe', 'lhe_no_banner']:
            orig_lhe = lhe_parser.EventFile(filename)
            if self.options['input_format'] == 'lhe_no_banner':
                orig_lhe.allow_empty_event = True
                
        elif self.options['input_format'] in ['hepmc']:
            import madgraph.various.hepmc_parser as hepmc_parser
            orig_lhe = hepmc_parser.HEPMC_EventFile(filename)
            orig_lhe.allow_empty_event = True
            logger.info("Parsing input event to know how many decay to generate. This can takes few minuts.")
        else:
            raise Exception
            
        to_decay = collections.defaultdict(int)
        nb_event = 0
 
        for event in orig_lhe:
            nb_event +=1
            for particle in event:
                if particle.status == 1 and particle.pdg in asked_to_decay:
                    # final state and tag as to decay
                    to_decay[particle.pdg] += 1
            if self.options['input_format'] == 'hepmc' and nb_event == 250:
                currpos = orig_lhe.tell()
                filesize = orig_lhe.getfilesize()
                for key in to_decay:
                    to_decay[key] *= 1.05 * filesize/ currpos 
                    # 1.05 to avoid accidental coincidence with nevents
                break

        # Handle the banner of the output file
        if not self.options['seed']:
            self.options['seed'] = random.randint(0, int(30081*30081))
            #self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.options['seed'])
            self.history.insert(0, 'set seed %s' % self.options['seed'])

        if self.options['seed'] > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.options['seed']) + ' > 30081*30081'
            raise Exception(msg)

        #self.options['seed'] = self.options['seed']
        
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)


        # 2. Generate the events requested
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)      
            evt_decayfile = {} 
            for pdg, nb_needed in to_decay.items():
                #check if a splitting is needed
                if nb_needed == nb_event:
                    evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5)
                elif nb_needed %  nb_event == 0:
                    nb_mult = nb_needed // nb_event
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches:
                        continue
                    elif len(self.list_branches[name]) == nb_mult:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_event,100000), mg5)
                    else:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                elif self.options['cross_section']:
                    #cross-section hard-coded -> allow 
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    
                    if name not in self.list_branches:
                        continue
                    else:
                        try:
                            evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                        except common_run_interface.ZeroResult:
                            logger.warning("Branching ratio is zero for this particle. Not decaying it")
                            del to_decay[pdg]                    
                else:
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches or len(self.list_branches[name]) == 0:
                        continue
                    #raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed. One workaround is to force the final cross-section manually.")
                    if len(self.list_branches[name]) == 1:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_event,100000), mg5)
                    else:
                        evt_decayfile[pdg] = self.generate_events(pdg, min(nb_needed,100000), mg5, cumul=True)
                    
                     
        # Compute the branching ratio.
        if not self.options['cross_section']:
            br = 1
            multi_br = [ ]
            multi_totevt = 0
            for (pdg, event_files) in evt_decayfile.items():
                if not event_files:
                    continue
                totwidth = float(self.banner.get('param', 'decay', abs(pdg)).value)
                if to_decay[pdg] == nb_event:
                    # Exactly one particle of this type to decay by event
                    pwidth = sum([event_files[k].cross for k in event_files])
                    if pwidth > 1.01 * totwidth:
                        logger.critical("Branching ratio larger than one for %s " % pdg) 
                    br *= pwidth / totwidth
                elif to_decay[pdg] % nb_event == 0:
                    # More than one particle of this type to decay by event
                    # Need to check the number of event file to check if we have to 
                    # make separate type of decay or not.
                    nb_mult = to_decay[pdg] // nb_event
                    if nb_mult == len(event_files):
                        for k in event_files:
                            pwidth = event_files[k].cross
                            if pwidth > 1.01 * totwidth:
                                logger.critical("Branching ratio larger than one for %s " % pdg)                       
                            br *= pwidth / totwidth
                        br *= math.factorial(nb_mult)
                    else:
                        pwidth = sum(event_files[k].cross for k in event_files)
                        if pwidth > 1.01 * totwidth:
                            logger.critical("Branching ratio larger than one for %s " % pdg) 
                        br *= (pwidth / totwidth)**nb_mult
                else:
                    pwidth = sum([event_files[k].cross for k in event_files])        
                    multi_br.append(pwidth / totwidth) 
                    multi_totevt += to_decay[pdg] % nb_event
            if multi_br and multi_totevt % nb_event == 0:
                if all(misc.equal(br,multi_br[0], 2) for br in multi_br): 
                    logger.warning("not all event are decaying the same particle, this is only supported if each event have ONE decaying particle (not checked) and that all particles have the same BR")        
                else:
                    raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed: [%s %s ] " %(multi_br, multi_totevt))
            elif multi_br:
                raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed. (%s %s)" % (multi_br, multi_totevt))
        else:
            br = 1
        self.branching_ratio = br
        self.efficiency = 1
        try:
            self.cross, self.error = self.banner.get_cross(witherror=True)
        except:
            if self.options['input_format'] != 'lhe':
                self.cross, self.error = 0, 0
        self.cross *= br
        self.error *= br
        
        # modify the cross-section in the init block of the banner
        if not self.options['cross_section']:
            self.banner.scale_init_cross(self.branching_ratio)
        else:
            
            if self.options['input_format'] in ['lhe_no_banner','hepmc'] and 'init' not in self.banner:
                self.cross = sum(self.options['cross_section'].values())
                self.error = 0
                self.branching_ratio = 1
            else:  
                self.banner.modify_init_cross(self.options['cross_section'])
                new_cross, new_error =   self.banner.get_cross(witherror=True)
                self.branching_ratio = new_cross / self.cross
                self.cross = new_cross   
                self.error = new_error

        # 3. Merge the various file together.
        if self.options['input_format'] == 'hepmc':
            name = orig_lhe.name.replace('.hepmc', '_decayed.lhe')
            if not name.endswith('.gz'):
                name = '%s.gz' % name
            
            output_lhe = lhe_parser.EventFile(name, 'w')
        else:
            name = orig_lhe.name.replace('.lhe', '_decayed.lhe')
            if not name.endswith('.gz'):
                name = '%s.gz' % name
            output_lhe = lhe_parser.EventFile(name, 'w')
        try:
            self.banner.write(output_lhe, close_tag=False)
        except Exception:
            if self.options['input_format'] == 'lhe':
                raise
        
        # initialise object which store not use event due to wrong helicity
        bufferedEvents_decay = {}
        for pdg in evt_decayfile:
            bufferedEvents_decay[pdg] = [{}] * len(evt_decayfile[pdg])
        
        import time
        start = time.time()
        counter = 0
        orig_lhe.seek(0)

        for event in orig_lhe:
            if counter and counter % 100 == 0 and float(str(counter)[1:]) ==0:
                print("decaying event number %s [%s s]" % (counter, time.time()-start))
            counter +=1
            
            # use random order for particles to avoid systematics when more than 
            # one type of decay is asked.
            particles = [p for p in event if int(p.status) == 1.0]
            random.shuffle(particles)
            ids = [particle.pid for particle in particles]
            br = 1 #br for that particular events (for special/weighted case)
            hepmc_output = lhe_parser.Event() #for hepmc case: collect the decay particle
            for i,particle in enumerate(particles):
                # check if we need to decay the particle 
                if self.final_state and particle.pdg not in self.final_state:
                    continue # nothing to do for this particle
                if particle.pdg not in evt_decayfile:
                    continue # nothing to do for this particle
                
                # check how the decay need to be done
                nb_decay = len(evt_decayfile[particle.pdg])
                if nb_decay == 0:
                    continue #nothing to do for this particle
                if nb_decay == 1:
                    decay_file = evt_decayfile[particle.pdg][0]
                    decay_file_nb = 0
                elif ids.count(particle.pdg) == nb_decay:
                    decay_file = evt_decayfile[particle.pdg][ids[:i].count(particle.pdg)]
                    decay_file_nb = ids[:i].count(particle.pdg)
                else:
                    #need to select the file according to the associate cross-section
                    r = random.random()
                    tot = sum(evt_decayfile[particle.pdg][key].cross for key in evt_decayfile[particle.pdg])
                    r = r * tot
                    cumul = 0
                    for j,events in evt_decayfile[particle.pdg].items():
                        cumul += events.cross
                        if r <= cumul:
                            decay_file = events
                            decay_file_nb = j
                            break
                    else:
                        # security for numerical accuracy issue... (unlikely but better safe)
                        if (cumul-tot)/tot < 1e-5:
                            decay_file = events
                            decay_file_nb = j
                        else:
                            misc.sprint(j,cumul, events.cross, tot, (tot-cumul)/tot)
                            raise Exception
                
                if self.options['new_wgt'] == 'BR':
                    tot_width = float(self.banner.get('param', 'decay', abs(pdg)).value)
                    if tot_width:
                        br = decay_file.cross / tot_width
                # ok start the procedure
                if hasattr(particle,'helicity'):
                    helicity = particle.helicity
                else:
                    helicity = 9
                bufferedEvents = bufferedEvents_decay[particle.pdg][decay_file_nb]
                
                # now that we have the file to read. find the associate event
                # checks if we have one event in memory
                if helicity in bufferedEvents and bufferedEvents[helicity]:
                    decay = bufferedEvents[helicity].pop()
                else:
                    # read the event file up to completion
                    while 1:
                        try:
                            decay = next(decay_file)
                        except StopIteration:
                            # check how far we are
                            ratio = counter / nb_event 
                            needed = 1.05 * to_decay[particle.pdg] - counter
                            needed = min(100000, max(needed, 6000))
                            with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
                                new_file = self.generate_events(particle.pdg, needed, mg5, [decay_file_nb])
                            evt_decayfile[particle.pdg].update(new_file)
                            decay_file = evt_decayfile[particle.pdg][decay_file_nb]
                            continue

                        if helicity == decay[0].helicity or helicity==9 or \
                                            self.options["spinmode"] == "none":
                            break # use that event
                        # not valid event store it for later
                        if helicity not in bufferedEvents:
                            bufferedEvents[helicity] = [decay]
                        elif len(bufferedEvents[helicity]) < 200:
                            # only add to the buffering if the buffer is not too large
                            bufferedEvents[helicity].append(decay)
                # now that we have the event make the merge
                if self.options['input_format'] != 'hepmc':
                    particle.add_decay(decay)
                else:
                    if len(hepmc_output) == 0:
                        hepmc_output.append(lhe_parser.Particle(event=hepmc_output))
                        hepmc_output[0].color2 = 0
                        hepmc_output[0].status = -1
                        hepmc_output.nexternal+=1
                    decayed_particle = lhe_parser.Particle(particle, hepmc_output)
                    decayed_particle.mother1 = hepmc_output[0]
                    decayed_particle.mother2 = hepmc_output[0]
                    hepmc_output.append(decayed_particle)
                    hepmc_output.nexternal+=1
                    decayed_particle.add_decay(decay)
            # change the weight associate to the event
            if self.options['new_wgt'] == 'cross-section':
                event.wgt *= self.branching_ratio
                br = self.branching_ratio
            else:
                event.wgt *= br
                
            if self.options['input_format'] != 'hepmc':
                wgts = event.parse_reweight()
                for key in wgts:
                    wgts[key] *= br
                # all particle have been decay if needed
                output_lhe.write(str(event))
            else:
                hepmc_output.wgt = event.wgt
                hepmc_output.nexternal = len(hepmc_output) # the append does not update nexternal
                output_lhe.write(str(hepmc_output))
        else:
            if counter==0:
                raise Exception
        output_lhe.write('</LesHouchesEvents>\n')        
                    
    
    def load_model(self, name, use_mg_default, complex_mass=False):
        """load the model"""
        
        loop = False
        #if (name.startswith('loop_')):
        #    logger.info("The model in the banner is %s" % name)
        #    logger.info("Set the model to %s since only" % name[:5])
        #    logger.info("tree-level amplitudes are used for the decay ")
        #    name = name[5:]
        #    self.banner.proc_card.info['full_model_line'].replace('loop_','')

        logger.info('detected model: %s. Loading...' % name)
        model_path = name
        #base_model = import_ufo.import_model(model_path)

        # Import model
        base_model = import_ufo.import_model(name, decay=True,
                                               complex_mass_scheme=complex_mass)

        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()
        
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        self.mg5cmd.process_model()

    def generate_events(self, pdg, nb_event, mg5, restrict_file=None, cumul=False,
                        output_width=False):
        """generate new events for this particle
           restrict_file allow to only generate a subset of the definition
           cumul allow to merge all the definition in one run (add process)
                 to generate events according to cross-section
        """
        if not hasattr(self, 'me_int'):
            self.me_int = {}
            
        
        
        nb_event = int(nb_event) # in case of hepmc request the nb_event is not an integer
        if cumul:
            width = 0.
        else:   
            width = 1.
        part = self.model.get_particle(pdg)
        if not part:
            return {}# this particle is not defined in the current model so ignore it
        name = part.get_name()
        out = {}
        time_gen_dec = time.time()
        logger.info("generate %s decay event for particle %s" % (int(nb_event), name))
        if name not in self.list_branches:
            return out
        for i,proc in enumerate(self.list_branches[name]):
            if restrict_file and i not in restrict_file:
                continue
            decay_dir = pjoin(self.path_me, "decay_%s_%s" %(str(pdg).replace("-","x"),i))
            if not os.path.exists(decay_dir):
                if cumul:
                    mg5.exec_cmd("generate %s" % proc)
                    for j,proc2 in enumerate(self.list_branches[name][1:]):
                        misc.sprint(proc2)
                        if restrict_file and j not in restrict_file:
                            raise Exception # Do not see how this can happen
                        mg5.exec_cmd("add process %s" % proc2)
                    # Force the Fortran madevent output: the decay directory is
                    # driven below through MadEventCmdShell, so it must have the
                    # madevent structure regardless of MG5's default output mode
                    # (which is 'mg7' in MadGraph7).
                    mg5.exec_cmd("output madevent %s -f" % decay_dir)
                else:
                    misc.sprint(proc)
                    mg5.exec_cmd("generate %s" % proc)
                    mg5.exec_cmd("output madevent %s -f" % decay_dir)
                
                options = dict(mg5.options)
                if self.options['ms_dir']:
                    # we are in gridpack mode -> create it
                    if decay_dir in self.me_int:
                        me5_cmd = self.me_int[decay_dir]
                    else:
                        me5_cmd = madevent_interface.MadEventCmdShell(me_dir=os.path.realpath(\
                                                decay_dir), options=options)
                        me5_cmd.options["automatic_html_opening"] = False
                        me5_cmd.options["madanalysis5_path"] = None
                        me5_cmd.options["madanalysis_path"] = None
                        me5_cmd.allow_notification_center = False
                        try:
                            os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card_default.dat'))
                            os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card.dat'))
                        except Exception as error:
                            logger.debug(error)
                            pass 
                        self.me_int[decay_dir] = me5_cmd

                    if self.options["run_card"]:
                        run_card = self.run_card
                    else:
                        run_card = banner.RunCard(pjoin(decay_dir, "Cards", "run_card.dat"))                        
                    run_card["iseed"] = self.options['seed']
                    run_card['gridpack'] = True
                    run_card['systematics_program'] = 'False'
                    run_card['use_syst'] = False
                    run_card.__setitem__('allow_overshoot_events', True, change_userdefine=True)
                    run_card.__setitem__('refine_evt_by_job', 5000, change_userdefine=True)
                    run_card.write(pjoin(decay_dir, "Cards", "run_card.dat"))
                    param_card = self.banner['slha']
                    open(pjoin(decay_dir, "Cards", "param_card.dat"),"w").write(param_card)
                    self.options['seed'] += 1
                    self.seed = self.options['seed'] 
                    # actually creation
                    me5_cmd.exec_cmd("generate_events run_01 -f")
                    if output_width:
                        if cumul:
                            width += me5_cmd.results.current['cross']
                        else:
                            width *= me5_cmd.results.current['cross']
                    me5_cmd.exec_cmd("exit")                        
                    #remove pointless informat
                    if not os.path.exists(pjoin(decay_dir, 'run.sh')):
                        devnull = open('/dev/null','w')
                        misc.call(["rm", "Cards", "bin", 'Source', 'SubProcesses'], cwd=decay_dir,stdout=devnull, stderr=-2)
                        misc.call(['tar', '-xzpvf', 'run_01_gridpack.tar.gz'], cwd=decay_dir,stdout=devnull, stderr=-2)
                        devnull.close()
            # Now generate the events
            if not self.options['ms_dir']:
                if decay_dir in self.me_int:
                        me5_cmd = self.me_int[decay_dir]
                else:
                    me5_cmd = madevent_interface.MadEventCmdShell(me_dir=os.path.realpath(\
                                                    decay_dir), options=mg5.options)
                    me5_cmd.options["automatic_html_opening"] = False
                    me5_cmd.options["automatic_html_opening"] = False
                    me5_cmd.options["madanalysis5_path"] = None
                    me5_cmd.options["madanalysis_path"] = None
                    me5_cmd.allow_notification_center = False
                    try:
                        os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card_default.dat'))
                        os.remove(pjoin(decay_dir, 'Cards', 'madanalysis5_parton_card.dat'))
                    except Exception as error:
                        logger.debug(error)
                        pass                 
                    self.me_int[decay_dir] = me5_cmd
                if self.options["run_card"]:
                    if hasattr(self, 'run_card'):
                        run_card = self.run_card
                    elif hasattr(self.options, 'run_card'):
                        run_card = self.options.run_card
                    else:
                        self.run_card = banner.RunCard(self.options["run_card"])
                        run_card = self.run_card 
                else:
                    run_card = banner.RunCard(pjoin(decay_dir, "Cards", "run_card.dat"))
                run_card["nevents"] = int(0.8*nb_event)
                run_card.__setitem__('allow_overshoot_events', True, change_userdefine=True)
                run_card.__setitem__('refine_evt_by_job', 5000, change_userdefine=True)
                # Handle the banner of the output file
                if not self.seed:
                    self.seed = random.randint(0, int(30081*30081))
                    self.do_set('seed %s' % self.seed)
                    logger.info('Will use seed %s' % self.seed)
                    self.history.insert(0, 'set seed %s' % self.seed)
                run_card["iseed"] = self.seed
                run_card["systematics_program"] = 'None'
                run_card['use_syst'] = False
                run_card.write(pjoin(decay_dir, "Cards", "run_card.dat"))
                param_card = self.banner['slha']
                open(pjoin(decay_dir, "Cards", "param_card.dat"),"w").write(param_card)
                self.seed += 1
                me5_cmd.exec_cmd("generate_events run_01 -f")
                if output_width:
                    if cumul:    
                        width += me5_cmd.results.current['cross']
                    else:
                        width *= me5_cmd.results.current['cross']
                if run_card["nevents"] > 1.01 * me5_cmd.results.current['nb_event']:
                    logger.critical('The number of event generated is only %s/%s. This typically indicates that you need specify cut on the decay process.',me5_cmd.results.current['nb_event'], run_card["nevents"])
                    logger.critical('We strongly suggest that you cancel/discard this run.')
                me5_cmd.exec_cmd("exit")
                out[i] = lhe_parser.EventFile(pjoin(decay_dir, "Events", 'run_01', 'unweighted_events.lhe.gz'))        
            else:
                if not self.seed:
                    if hasattr(self, 'mother'):
                        try:
                            self.seed = 100 + self.mother.run_card['iseed']
                        except:
                            self.seed = random.randint(0, int(30081*30081))
                self.seed += 1
                if self.seed > 30081*30081:
                    self.seed -= 30081*30081        
                logger.info('Will use seed %s' % (self.seed))
                misc.call(['run.sh', str(int(1.2*nb_event)), str(self.seed), '-p', str(self.mg5cmd.options['nb_core'])], cwd=decay_dir)
                out[i] = lhe_parser.EventFile(pjoin(decay_dir, 'events.lhe.gz'))     
            if cumul:
                break
        time_gen_dec = time.time()-time_gen_dec
        logger.critical(f"Time for decay event generation = {time_gen_dec:.1f} sec")
        if not output_width:
            return out
        else:
            return out, width

    def run_onshell(self, line, density_method=False):
        """Run the onshell Algorithm"""
        
        # 1. Read the event file to check which decay to perform and the number
        #   of event to generate for each type of particle. (assume efficiency=1 for spin 0
        #   otherwise efficiency=2
        # 2. Generate the associated events
        # 3. generate the various matrix-element (production/decay/production+decay) 
        #    => no production+decay if density_method on True
        # 4. determine the maxwgt
        # 5. generate the decay (for each production event)
        # 6. perform the merge of the events.
        #    if not enough events. re-generate the missing one.
        
        # Spyros: this is not used - remove?
        args = self.split_arg(line)

        # First define an utility function for generating events when needed
        # Spyros what should be done here? 

        # Find which particles should be decayed
        asked_to_decay = set()
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                for pdg in self.mg5cmd._multiparticles[part]:
                    asked_to_decay.add(pdg)
            else:
                asked_to_decay.add(self.mg5cmd._curr_model.get('name2pdg')[part])

        # 0. Define the path where to write the file
        self.path_me = os.path.realpath(self.options['curr_dir']) 
        if self.options['ms_dir']:
            self.path_me = os.path.realpath(self.options['ms_dir'])
            if not os.path.exists(self.path_me):
                os.mkdir(self.path_me) 
        else:
            # cleaning
            for name in misc.glob("decay_*_*", self.path_me):
                shutil.rmtree(name)

        self.events_file.close()
        if self.events_file.name.endswith('.gz'):
            misc.gunzip(self.events_file.name)
        orig_lhe = lhe_parser.EventFile(self.events_file.name)
        if self.options['fixed_order']:
            orig_lhe.eventgroup = True

        # Dictionary with particle properties
        decay_dict = {}
        
        # 1. Open input event file and check which particles to decay
        # - count the number of particles to be decayed.
        to_decay = collections.defaultdict(int)	
        nb_event = 0
        for event in orig_lhe:
            if self.options['fixed_order']:
                event = event[0]
            nb_event +=1
            for particle in event:
                if particle.status == 1 and particle.pdg in asked_to_decay:
                    # final state and tag as to decay
                    to_decay[particle.pdg] += 1
                    # Properties of decaying particle
                    width = self.banner.get('param_card', 'decay', abs(particle.pdg)).value
                    mass = self.banner.get('param_card', 'mass', abs(particle.pdg)).value
                    color = self.model.get_particle(particle.pdg).get('color')
                    spin = self.model.get_particle(particle.pdg).get('spin')
                    decay_dict[particle.pdg] = [width, mass, color, spin]
        #print(f"to_decay = {to_decay}")
                	
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)  
                self.model = mg5._curr_model

        # Handle the banner of the output file
        if not self.seed:
            self.seed = random.randint(0, int(30081*30081))
            self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.seed)
            self.history.insert(0, 'set seed %s' % self.seed)

        if self.seed > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.seed) + ' > 30081*30081'
            raise Exception(msg)

        self.options['seed'] = self.seed
        #print(f"from run onshell seed = {self.seed}")
        
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)


        # 2. Generate the events requested
        nevents_for_max = self.options['Nevents_for_max_weight']
        if nevents_for_max == 0 :
            nevents_for_max = 75
        nevents_for_max *= self.options['max_weight_ps_point']
        
        with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
            mg5 = self.mg5cmd
            if not self.model:
                modelpath = self.model.get('modelpath+restriction')
                mg5.exec_cmd("import model %s" % modelpath)      
            evt_decayfile = {}
            br = 1.
            # pdg -> br_pdg for the "mixed final-state" case (events do not
            # all share the same set of decaying particles). Filled in the
            # else-branch below and consumed after the loop to compute the
            # per-pdg drop probability that equalizes BRs across productions.
            mixed_pdgs_br = {}
            for pdg, nb_needed in to_decay.items():
                # muliply by expected effeciency of generation
                spin = self.model.get_particle(pdg).get('spin')
                if spin == 1:
                    efficiency = 1.1
                else:
                    efficiency = 2.0
              
                totwidth = self.banner.get('param_card', 'decay', abs(pdg)).value
				
                #check if a splitting is needed
                if nb_needed == nb_event:
                    nb_needed = (int(efficiency*nb_needed) + nevents_for_max)*self.options['decay_event_mult'] 
                    evt_decayfile[pdg], pwidth = self.generate_events(pdg, nb_needed, mg5, output_width=True, cumul=True)
                    if pwidth > 1.01*totwidth:
                        logger.warning('partial width (%s) larger than total width (%s) --from param_card--', pwidth, totwidth)
                    elif pwidth > totwidth:
                        pwidth = totwidth
                    br *= pwidth / totwidth
                elif nb_needed %  nb_event == 0:
                    nb_mult = nb_needed // nb_event
                    nb_needed = (int(efficiency*nb_needed) + nevents_for_max*nb_mult)*self.options['decay_event_mult']
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches:
                        continue
                    elif len(self.list_branches[name]) == nb_mult:
                        evt_decayfile[pdg], pwidth = self.generate_events(pdg, nb_event*self.options['decay_event_mult'], mg5, output_width=True)
                        if pwidth > 1.01*totwidth:
                            logger.warning('partial width (%s) larger than total width (%s) --from param_card--')
                        elif pwidth > totwidth:
                            pwidth = totwidth
                        br *= pwidth / totwidth**nb_mult
                        br *= math.factorial(nb_mult)
                    else:
                        evt_decayfile[pdg],pwidth = self.generate_events(pdg, nb_needed, mg5, cumul=True, output_width=True)
                        if pwidth > 1.01*totwidth:
                            logger.warning('partial width (%s) larger than total width (%s) --from param_card--')
                        elif pwidth > totwidth:
                            pwidth = totwidth
                        br *= (pwidth / totwidth)**nb_mult                      
                else:
                    # Mixed case: events do not all share the same final-state
                    # particles to be decayed. We collect this pdg here and, once
                    # the loop is done, equalize BRs via the legacy
                    # add_loose_decay mechanism (drop events sampled to the
                    # "fake" decay channel so the output stays unweighted).
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches or len(self.list_branches[name]) == 0:
                        continue
                    nb_gen = (int(efficiency*nb_needed) + nevents_for_max) \
                                * self.options['decay_event_mult']
                    evt_decayfile[pdg], pwidth = self.generate_events(
                        pdg, nb_gen, mg5, cumul=True, output_width=True)
                    if pwidth > 1.01*totwidth:
                        logger.warning('partial width (%s) larger than total width (%s) --from param_card--', pwidth, totwidth)
                    elif pwidth > totwidth:
                        pwidth = totwidth
                    mixed_pdgs_br[pdg] = pwidth / totwidth

        # Equalize branching ratios across mixed productions (legacy
        # add_loose_decay mechanism): pick max_br as the global BR factor and
        # drop, per event, with probability 1 - br_pdg / max_br so the output
        # sample stays unweighted. The banner cross-section is corrected after
        # the decay loop (below) once the actual number of kept events is
        # known.
        drop_prob_per_pdg = {}
        if mixed_pdgs_br:
            max_mixed_br = max(mixed_pdgs_br.values())
            br *= max_mixed_br
            for pdg, pdg_br in mixed_pdgs_br.items():
                drop_prob_per_pdg[pdg] = 1.0 - pdg_br / max_mixed_br
            if any(d > 1e-9 for d in drop_prob_per_pdg.values()):
                logger.warning(
                    "Mixed-pdg production processes have different total BRs "
                    "(per-pdg BR=%s, max=%g). Equalizing by dropping events; "
                    "the output sample stays unweighted and the banner "
                    "cross-section reflects the effective BR.",
                    {k: '%.4g' % v for k, v in mixed_pdgs_br.items()},
                    max_mixed_br,
                )
        mixed_pdgs_set = set(drop_prob_per_pdg.keys())

        self.branching_ratio = br
        self.efficiency = 1
        self.cross, self.error = self.banner.get_cross(witherror=True)
        self.cross *= self.branching_ratio
        self.error *= self.branching_ratio
        

        density_needs_reshuffle = (
            density_method
            and self.options['density_pole_approximation']
            and self.options['density_do_reshuffle']
        )

        # 3. generate the various matrix-elements
        time_me_generation = time.time()
        self.update_status('generating Madspin matrix element (density_method=%s)' % density_method)
        if density_method:
            self.generate_all = madspin.decay_all_events_density(self, self.banner, self.events_file,self.options)
        else:
            self.generate_all = madspin.decay_all_events_onshell(self, self.banner, self.events_file,self.options)

        self.generate_all.compile()
        self.all_me = self.generate_all.all_me
        self.all_f2py = {}
        # Build raw-PDG -> merged-particle-ID mapping so event tags can be
        # remapped to match all_me keys (which are built from process legs,
        # i.e. merged-particle IDs when apply_flavor_grouping is on).
        self._revert_merged = {}
        model_merged = getattr(self.generate_all, 'model', None)
        model_merged = model_merged.get('merged_particles') if model_merged else None
        if model_merged:
            for merged_id, members in model_merged.items():
                for pdg in members:
                    self._revert_merged[pdg] = merged_id
                    self._revert_merged[-pdg] = -merged_id

        self.all_amp = {}
        self.all_nhel = {}
        self.all_jamp = {}
        self.all_inter = {}
        self.all_density = {}
        self.all_matrix = {}
        time_me_generation = time.time() - time_me_generation
        logger.critical(f"Time ME generation: {time_me_generation:.2f} sec")

        #4. determine the maxwgt
        maxwgt = self.get_maxwgt_for_onshell(orig_lhe, evt_decayfile, decay_dict)

        #5. generate the decay (for each production event)

        orig_lhe.seek(0)
        output_lhe = lhe_parser.EventFile(orig_lhe.name.replace('.lhe', '_decayed.lhe'), 'w')
        if self.options['fixed_order']:
            output_lhe.eventgroup = True
        
        self.banner.scale_init_cross(self.branching_ratio)
        self.banner.write(output_lhe, close_tag=False)       
        
        self.efficiency =1.
        nb_try = 0
        nb_loose_skip = 0  # events dropped to equalize BRs (fake-decay path)
        #nb_event = len(orig_lhe)
        nb_event = orig_lhe.get_banner().run_card['nevents']

        start = time.time()
        logger.info("Start generating decays")
        for curr_event,production in enumerate(orig_lhe):
            if self.options['fixed_order']:
                production, counterevt = production[0], production[1:]
            if curr_event and self.efficiency and curr_event % 10 == 0 and float(str(curr_event)[1:]) == 0:
                logger.info("decaying event number %s. Efficiency: %s [%s s]" % (curr_event, 1/self.efficiency, time.time()-start))

            # BR-equalization: drop this event with probability
            # 1 - br_pdg / max_br when this production process has a smaller
            # total BR than the largest one in the mixed sample. Done before
            # any matrix-element work so dropped events are cheap.
            if drop_prob_per_pdg:
                evt_mixed_pdgs = [p.pid for p in production
                                  if int(p.status) == 1 and p.pid in mixed_pdgs_set]
                if len(evt_mixed_pdgs) == 1:
                    drop = drop_prob_per_pdg[evt_mixed_pdgs[0]]
                    if drop > 0 and random.random() < drop:
                        nb_loose_skip += 1
                        continue
                elif len(evt_mixed_pdgs) > 1:
                    raise self.InvalidCmd(
                        "BR equalization for events with more than one "
                        "mixed-pdg decaying particle is not implemented yet "
                        "(event %d has pdgs=%s). Please report this case." %
                        (curr_event, evt_mixed_pdgs))

            # Per-production-event cache reused across rejection retries.
            prod_density_cached = None

            while 1:
                nb_try += 1
                decays = self.get_decay_from_file(production, evt_decayfile, nb_event-curr_event)
                # In density mode do not do full event construction before accept/reject
                build_event = (not density_method) or self.options['fixed_order']
                
                if prod_density_cached is None or not self.options['density_pole_approximation']:
                    full_evt, wgt, prod_density_cached = self.get_onshell_evt_and_wgt(
                        production, decays, decay_dict, build_event=build_event)
                else:
                    full_evt, wgt, _ = self.get_onshell_evt_and_wgt(
                        production, decays, decay_dict, prod_density_cached, build_event=build_event)
                jac = 1
                if density_needs_reshuffle and self.options['density_keep_jacobian']:
                    # Build the full Event for correct jacobian handling
                    # already done if density_pole_approximation is False, 
                    # but need to be done here if density_pole_approximation is True and density_keep_jacobian is True
                    full_evt = lhe_parser.Event(str(production))
                    full_evt = full_evt.add_decays(decays)
                    jac = full_evt.reshuffle_production()
                        
                if random.random()*maxwgt < wgt*jac:
                    if density_needs_reshuffle and not self.options['density_keep_jacobian']:
                        # Build the full Event only after acceptance in density mode.
                        if self.options['density_pole_approximation']:
                            full_evt = lhe_parser.Event(str(production))
                        else:
                            full_evt = production
                        full_evt = full_evt.add_decays(decays)
                        if self.options['density_pole_approximation']:
                            jac = full_evt.reshuffle_production()
                    elif full_evt is None:
                        # No-reshuffle density mode still needs a concrete event to write out.
                        if density_method and self.options['density_pole_approximation']:
                            full_evt = lhe_parser.Event(str(production))
                        else:
                            full_evt = production
                        full_evt = full_evt.add_decays(decays)
                    if self.options['fixed_order']:
                        full_evt = [full_evt] + [evt.add_decays(decays) for evt in counterevt]
                    break
                #else:
                #    misc.sprint('fail-> retry')
            # Efficiency = accepted / trials (+1 because current event is already accepted)
            self.efficiency = float(curr_event + 1) / nb_try
            #if density_method:
            #    full_evt.reshuffle_production()
            if self.options['fixed_order']:
                for evt in full_evt:
                    # change the weight associated to the event
                    evt.wgt *= self.branching_ratio
                    wgts = evt.parse_reweight()
                    for key in wgts:
                        wgts[key] *= self.branching_ratio 
            else:
                # change the weight associated to the event
                full_evt.wgt *= self.branching_ratio
                wgts = full_evt.parse_reweight()
                for key in wgts:
                    wgts[key] *= self.branching_ratio            
            
            output_lhe.write_events(full_evt)

        output_lhe.write('</LesHouchesEvents>\n')
        # Log unweighting efficiency (can be turned off)
        n_processed = curr_event + 1
        n_written = n_processed - nb_loose_skip
        eff = float(n_written) / nb_try if nb_try else 0.0
        logger.critical(
            "MadSpin unweight efficiency: %.4f (%d written / %d trials, %.2f trials/event)",
            eff, n_written, nb_try, (1.0 / eff if eff else float("inf"))
        )
        if nb_loose_skip > 0:
            # Rewrite the banner with the corrected cross-section so it
            # matches the actual sum of kept-event weights. Each kept event
            # already has wgt = orig_wgt * max_br; we need the banner to read
            # σ * max_br * (n_written / n_processed) ≈ σ * <br>.
            br_correction = float(n_written) / n_processed
            self._rewrite_lhe_banner_cross(output_lhe.name, br_correction,
                                           n_written=n_written)
            self.branching_ratio *= br_correction
            self.cross *= br_correction
            self.error *= br_correction
            logger.info(
                "BR equalization: dropped %d/%d events (effective BR rescale = %.4g).",
                nb_loose_skip, n_processed, br_correction,
            )
            # Downstream sets nb_event = int(original_nb_event * efficiency)
            # so the kept-fraction needs to be communicated as the efficiency.
            self.efficiency = br_correction
        else:
            self.efficiency = 1 # to let me5 to write the correct number of events
        # Re-gzip the input events file (gunzipped at the start of this
        # routine) and the decayed output, matching the legacy MadSpin path
        # so downstream code (banners, crossx.html) finds the *.lhe.gz files
        # it expects.
        try:
            output_lhe.close()
        except Exception:
            pass
        try:
            input_evt_path = self.events_file.name
            if input_evt_path.endswith('.lhe') and os.path.exists(input_evt_path):
                misc.gzip(input_evt_path)
        except Exception as exc:
            logger.warning('Could not re-gzip MadSpin input file %s: %s',
                           getattr(self.events_file, 'name', '?'), exc)
        try:
            decayed_path = output_lhe.name
            if decayed_path.endswith('.lhe') and os.path.exists(decayed_path):
                misc.gzip(decayed_path)
        except Exception as exc:
            logger.warning('Could not gzip MadSpin decayed output %s: %s',
                           output_lhe.name, exc)
        logger.info('Done so far. output written in %s' % output_lhe.name)
        logger.critical(f"Time for decay = {time.time()-start:.2f} sec")

    def _rewrite_lhe_banner_cross(self, path, ratio, n_written=None):
        """Rewrite an already-written LHE file, multiplying every <init> line
        cross-section / error / xmax by ``ratio`` and (optionally) replacing
        the ``Number of Events`` entry in the MGGenerationInfo block with
        ``n_written``. Mirrors decay_all_events.write_banner_information for
        the PA-mode (run_onshell) code path."""

        tmp_path = path + '.tmp_brfix'
        shutil.move(path, tmp_path)
        with open(tmp_path, 'r') as src, open(path, 'w') as dst:
            in_init = False
            in_mggen = False
            for line in src:
                stripped = line.strip()
                lstripped = stripped.lower()
                if lstripped.startswith('<init'):
                    in_init = True
                    dst.write(line)
                    continue
                if in_init:
                    if lstripped.startswith('</init'):
                        in_init = False
                        dst.write(line)
                        continue
                    parts = stripped.split()
                    if len(parts) == 4:
                        try:
                            xsec, xerr, xmax = (float(parts[0]), float(parts[1]), float(parts[2]))
                            pid = int(parts[3])
                            dst.write("   %+13.7e %+13.7e %+13.7e %i\n" %
                                      (ratio*xsec, ratio*xerr, ratio*xmax, pid))
                            continue
                        except ValueError:
                            pass
                    dst.write(line)
                    continue
                # MGGenerationInfo block: update Number of Events and any
                # ":" -separated numeric field with the BR correction ratio.
                if lstripped.startswith('<mggenerationinfo'):
                    in_mggen = True
                    dst.write(line)
                    continue
                if in_mggen:
                    if lstripped.startswith('</mggenerationinfo'):
                        in_mggen = False
                        dst.write(line)
                        continue
                    if 'Number of Events' in line and n_written is not None:
                        dst.write('#  Number of Events        :       %i\n' % n_written)
                        continue
                    if ':' in line:
                        head, tail = line.rsplit(':', 1)
                        try:
                            value = float(tail.strip())
                            dst.write('%s : %s\n' % (head, value * ratio))
                            continue
                        except ValueError:
                            pass
                    dst.write(line)
                    continue
                dst.write(line)
        os.remove(tmp_path)

    def get_decay_from_file(self,production, evt_decayfile, nb_remain):
        """return a dictionary PDG -> list of associated decay"""
        
        out = collections.defaultdict(list)
        particles = [p for p in production if int(p.status) == 1.0]
        ids = [particle.pid for particle in particles]
        for i,particle in enumerate(particles):
            # check if we need to decay the particle 
            if particle.pdg not in evt_decayfile:
                continue # nothing to do for this particle
            # check how the decay need to be done
            nb_decay = len(evt_decayfile[particle.pdg])
            if nb_decay == 0:
                continue #nothing to do for this particle
            # Determine the file to read in order to get the decay [decay_file]
            if nb_decay == 1:
                decay_file = evt_decayfile[particle.pdg][0]
                decay_file_nb = 0
            elif ids.count(particle.pdg) == nb_decay:
                decay_file = evt_decayfile[particle.pdg][ids[:i].count(particle.pdg)]
                decay_file_nb = ids[:i].count(particle.pdg)
            else:
                #need to select the file according to the associate cross-section
                r = random.random()
                tot = sum(evt_decayfile[particle.pdg][key].cross for key in evt_decayfile[particle.pdg])
                r = r * tot
                cumul = 0
                for j,events in evt_decayfile[particle.pdg].items():
                    
                    cumul += events.cross
                    if r < cumul:
                        decay_file = events
                        decay_file_nb = j
                        break
                    else:
                        continue
                else:
                    raise Exception
            # So now we know which file to read. Do it and re-generate events for that 
            # file if needed.
            while 1:
                try:
                    decay = next(decay_file)
                    break
                except StopIteration:
                    # Estimate refill size from remaining production events
                    # efficiency and per-trial consumption if decaying particles
                    # Take into account identical parents
                    # Oversample by 10% to reduce refill frequency; cap to limit one refill cost.
                    eff = max(self.efficiency, 1e-12)
                    same_pdg = ids.count(particle.pdg)
                    if nb_decay == 1:
                        burn = same_pdg
                    elif nb_decay == same_pdg:
                        burn = 1.0
                    else:
                        burn = max(1.0, float(same_pdg) / float(nb_decay))
                    needed = int(math.ceil(1.10 * burn * nb_remain / eff))
                    needed = min(200000, max(needed, 1000))
                    with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
                        new_file = self.generate_events(particle.pdg, needed, self.mg5cmd, [decay_file_nb])
                    evt_decayfile[particle.pdg].update(new_file)
                    decay_file = evt_decayfile[particle.pdg][decay_file_nb]
                    continue
            out[particle.pdg].append(decay)
                        
        return out
        
    
    def get_maxwgt_for_onshell(self, orig_lhe, evt_decayfile, decay_dict):
        """determine the maximum weight for the onshell (or similar) strategy"""
        #print(f"decay_dict = {decay_dict} - length = {len(decay_dict)}")
        # event_decay is a dict pdg -> list of event file (contain the decay)
                
        if self.options['ms_dir'] and os.path.exists(pjoin(self.options['ms_dir'], 'max_wgt')):
            return float(open(pjoin(self.options['ms_dir'], 'max_wgt'),'r').read())
        
        nevents = self.options['Nevents_for_max_weight']
        if nevents == 0 :
            nevents = 75
        
        all_maxwgt = []
        logger.info("Estimating the maximum weight")
        logger.info("*****************************")
        logger.info("Probing the first %s events with %s phase space points" % (nevents, self.options['max_weight_ps_point']))

        self.efficiency = 1. / self.options['max_weight_ps_point']
        start = time.time()

        orig_lhe.seek(0)
        
        # Loop over production events
        for i in range(nevents):
            if i % 5 ==1:
                logger.info( "Event %s/%s :  %2fs" % (i, nevents, time.time()-start))
            maxwgt = 0
            try:
                base_event = next(orig_lhe)
            except StopIteration:
                break
            if self.options['fixed_order']:
                base_event = base_event[0]
            # Cache production density matrix
            density_matrix_prod = None
            # Loop over decays
            for j in range(self.options['max_weight_ps_point']):
                decays = self.get_decay_from_file(base_event, evt_decayfile, nevents-i)   
                #carefull base_event is modified by the following function 
                if density_matrix_prod is None:
                    _, wgt, density_matrix_prod = self.get_onshell_evt_and_wgt(
                        base_event, decays, decay_dict, build_event=False)
                    #print(f"wgt1 = {wgt}")
                else:
                    wgt = self.get_onshell_evt_and_wgt(
                        base_event, decays, decay_dict, density_matrix_prod, build_event=False)[1]
                    #print(f"wgt2 = {wgt}")
                #print(f"Event {i} , PS point {j}, wgt for max = {wgt}")
                jac = 1 
                if (self.options['density_keep_jacobian'] and
                        self.options['density_pole_approximation'] and
                        self.options['density_do_reshuffle']):
                    # Build the full Event for tracking associated jacobian
                    full_evt = lhe_parser.Event(str(base_event))
                    full_evt = full_evt.add_decays(decays)
                    jac = full_evt.reshuffle_production()
                maxwgt = max(wgt*jac, maxwgt)
            all_maxwgt.append(maxwgt.real)
        all_maxwgt.sort(reverse=True)
        assert all_maxwgt[0] >= all_maxwgt[1], "ERROR: "
        decay_tools=madspin.decay_misc()
        ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt)
        base_max_weight = 1.05 * (ave_weight+self.options['nb_sigma']*std_weight)

        for i in [20, 30, 40, 50]:
            if len(all_maxwgt) < i:
                break
            ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt[:i])
            base_max_weight = max(base_max_weight, 1.05 * (ave_weight+self.options['nb_sigma']*std_weight))
                
            if all_maxwgt[1] > base_max_weight:
                base_max_weight = 1.05 * all_maxwgt[1]
        if self.options['ms_dir']:
            open(pjoin(self.options['ms_dir'], 'max_wgt'),'w').write(str(base_max_weight))
        return base_max_weight

            
    def get_onshell_evt_and_wgt(self, production, decays, decay_dict, prod_density_cached=None, build_event=True):
        """ return the onshell wgt for the production event associated to the decays
            return also the full event with decay.
            Carefull this modifies production event (pass to the full one)
            build_event: if False (density mode) compute weight without building event"""
        decay_me = 1.0
        decay_me_debug = 1.0
        jac = 1.0
        tag, order = production.get_tag_and_order(self._revert_merged or None)
        try:
            info = self.generate_all.all_me[tag]
        except:
            misc.sprint(self.generate_all.all_me)
            misc.sprint(production)
            misc.sprint(decays)
            raise
        
        # Calculate decay ME
        if self.generate_all.mode == 'onshell':
            #print(f"len(decays) = {len(decays)}")
            for pdg in decays:
                for dec in decays[pdg]:
                    #print(f"dec = {dec}")
                    decay_me *= self.calculate_matrix_element(dec)
        else:
            if self.options['density_debug']:
                #print(f"len(decays) = {len(decays)}")
                for pdg in decays:
                    for dec in decays[pdg]:
                        #print(f"dec = {dec}")
                        decay_me_debug *= self.calculate_matrix_element(dec)

        # Calculate production*decay ME
        if self.generate_all.mode == 'onshell':
            full_event = lhe_parser.Event(str(production))
            full_event = full_event.add_decays(decays)
            #print(f"full_event = {full_event}")
            full_me = self.calculate_matrix_element(full_event)
            #print(f"full_me = {full_me}")
        else:
            #offshell mode
            full_dqrts = production.sqrts 
            jac = 1 
            if (not self.options['density_pole_approximation'] or
                    self.options['density_do_reshuffle']):
                for pdg in decays:
                    for dec in decays[pdg]:
                        pole = self.banner.get('param', 'mass', abs(pdg)).value
                        width = self.banner.get('param', 'decay', abs(pdg)).value 
                        if self.options['BW_cut'] <0: 
                           bw_cut = 15
                        else:
                           bw_cut = self.options['BW_cut']     
                        min_mass = pole - bw_cut * width
                        max_mass = min(pole + bw_cut * width,full_dqrts) 
                        dec[0].new_mass = lhe_parser.Event.generate_random_mass(pole, width, min_mass, max_mass)
                        dec[0].reshuffle_info = (pole, width, min_mass, max_mass)

                        full_dqrts -= dec[0].new_mass
                        gap = math.atan((pole**2-min_mass**2)/pole*width)
                        gap += math.atan((max_mass**2-pole**2)/pole*width)
                        jac *= gap/math.pi 
            if prod_density_cached is None:
                full_me, prod_density_cached, prod_diag, dec_diag = self.calculate_matrix_element_from_density(production, decays, decay_dict)
            else:                
                full_me, _, prod_diag, dec_diag = self.calculate_matrix_element_from_density(production, decays, decay_dict, prod_density_cached)
            #print(f"full_me from density = {full_me}")
   
            full_event = None
            if build_event or self.options['density_debug']:
                # Create full event from production and decays
                if self.options['density_pole_approximation']:
                    full_event = lhe_parser.Event(str(production))
                else:
                    full_event = production          
                # CAUTION: the next line removes everything from decays dictionary
                full_event = full_event.add_decays(decays)
            
                #print(f"full event 2 = {full_event}")
                if self.options['density_debug']:
                    me1 = self.calculate_matrix_element(full_event)
                    #print(f"me1 = {me1} , me2 = {full_me} , ratio = {me1/full_me}")
                    if abs(1-me1/full_me) > self.options['density_tolerance']:
                        print(f"full = {me1} , density = {full_me} , ratio = {me1/full_me}")	    
                        print(full_event)
                        print(production)
                        print(decays)

                        print("ERROR matrix element from density does not match with full matrix element")	
                        raise RuntimeError("ERROR matrix element from density does not match with full matrix element")	 
    
        # Calculate production ME and cache it so that if we reject 
        # the decay the production ME will not be recalculated
        if hasattr(production, 'me_wgt'):
            production_me = production.me_wgt
        else:
            production_me = self.calculate_matrix_element(production) if self.generate_all.mode == 'onshell' \
                            else prod_diag
            production.me_wgt = production_me

        if self.generate_all.mode == 'density' and self.options['density_debug']:
            prod_me = self.calculate_matrix_element(production)
            #print(f"prod_diag = {prod_diag} , prod_me = {prod_me}")
            if abs(1-prod_diag/prod_me) > self.options['density_tolerance']:
                print(f"prod_me = {prod_me} , prod_diag = {prod_diag} , ratio = {prod_diag/prod_me}")	    
                raise RuntimeError("ERROR production matrix element from density does not match with diagonal")	     
            if abs(1-dec_diag/decay_me_debug) > self.options['density_tolerance']:
                print(f"decay_me = {decay_me_debug} , dec_diag = {dec_diag} , ratio = {dec_diag/decay_me_debug}")	    
                raise RuntimeError("ERROR decay matrix element from density does not match with diagonal")	   
        
        if self.generate_all.mode == 'density':
            decay_me = dec_diag

        #print(f"full_event = {full_event}")
        #print(f"full_me = {full_me}")
        #print(f"production_me = {production_me}")
        #print(f"decay_me = {decay_me}")
        #print(f"wgt = {full_me/(production_me*decay_me)}")
        
        return full_event, full_me/(production_me*decay_me)*jac, prod_density_cached

           
    def calculate_matrix_element_from_density(self, production, decays, decay_dict, prod_density_cached=None):
        """routine to return the matrix element from density matrices"""

        # ------------------------------------------------------------------
        # Load f2py module and build pdg2prefix map if needed (unchanged logic)
        # ------------------------------------------------------------------
        if not hasattr(self, 'f2py_module'):
            sp_path = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses')
            if sys.path[0] != sp_path:
                sys.path.insert(0, sp_path)

            mymod = self._load_f2py_matrix_module(sp_path)
            self.f2py_module = mymod

            all_prefix = self.f2py_module.get_prefix()
            all_pdg, all_procid = self.f2py_module.get_pdg_order()
            self.pdg2prefix = {}
            for i, pdg in enumerate(all_pdg):
                pdg = tuple([x for x in pdg if x != 0])
                self.pdg2prefix[pdg] = (str(all_prefix[i].decode()).strip(), i)

            if self.model_init:
                self.model_init = False
                with misc.chdir(sp_path):
                    if (not os.path.exists(pjoin(self.path_me, 'Cards', 'param_card.dat'))
                            and os.path.exists(pjoin(self.path_me, 'param_card.dat'))):
                        mymod.initialise(pjoin(self.path_me, 'param_card.dat'))
                    else:
                        mymod.initialise(pjoin(self.path_me, 'Cards', 'param_card.dat'))

        # ------------------------------------------------------------------
        # Cache production-only metadata reused across rejection retries
        # ------------------------------------------------------------------
        decays_key = tuple(decays.keys())
        MEdenom_prod, MEdenom_decay = None, None
        prod_static = getattr(production, '_ms_density_static', None)
        if not self.options['density_pole_approximation'] or \
            (not prod_static or prod_static.get('decays_key') != decays_key):
            # Production averaging factor (spin/color initial state) from standalone
            iden_p = self.get_iden(production)

            # Symmetry factor for identical final states in production
            final_pdgs = [int(p.pid) for p in production if getattr(p, "status", None) == 1]
            counts_final = collections.Counter(final_pdgs)
            sym_factor_prod_ident = 1
            for n in counts_final.values():
                if n > 1:
                    sym_factor_prod_ident *= math.factorial(n)

            # Find particles that should decay (status==1 and pid in decays keys)
            init_part = [part for pdg in decays_key for part in production
                         if part.pid == pdg and part.status == 1]
            nchanging = len(init_part)

            # Allowed helicities per spin
            hel_dict = {1: [0], 2: [1, -1], 3: [-1, 0, 1]}

            # Decaying-particle positions (+1 for Fortran), spins, helicities
            position = [i + 1 for pdg in decays_key
                        for i in range(len(production))
                        if production[i].pid == pdg and production[i].status == 1]
            decaying_pdg = [int(production[i - 1].pid) for i in position]
            decaying_spins = [self.model.get_particle(i).get('spin') for i in decaying_pdg]
            helicities = [hel_dict[i] for i in decaying_spins]

            use_new_mass = (
                not self.options['density_pole_approximation'] or
                self.options['density_do_reshuffle']
            )
            if use_new_mass:
                new_mass = {}
                for key in decays:
                    new_mass[key] = [getattr(dec[0], 'new_mass', dec[0].mass)
                                     for dec in decays[key]]

                for particle in production:
                    if particle.status == 1 and particle.pid in new_mass:
                        particle.new_mass = new_mass[particle.pid].pop(0)
            else:
                for particle in production:
                    if hasattr(particle, 'new_mass'):
                        del particle.new_mass
                    if hasattr(particle, 'reshuffle_info'):
                        del particle.reshuffle_info

            MEdenom_prod, MEdenom_decay = None, None
            if not self.options['density_pole_approximation']:
                # compute the denominator and then reshuffle the event before 
                # computing the numerator 
                MEdenom_prod = self.calculate_matrix_element(production)  
                MEdenom_decay = 1.0              
                for key in decays:
                    for dec in decays[key]:
                        MEdenom_decay *= self.calculate_matrix_element(dec)
                # now doing the reshuffling
                # doing the reshuffling for each part:
                jac = 1.0
                jac *= production.reshuffle_production()
                for key in decays:
                    for dec in decays[key]:
                        jac *= dec.reshuffle_decayevt()
                if jac == 0:
                    raise Exception

            allowed_hel_pairs, allowed_hel = self.get_allowed_hel(helicities)

            prod_static = {
                'decays_key': decays_key,
                'iden_p': iden_p,
                'sym_factor_prod_ident': sym_factor_prod_ident,
                'init_part': init_part,
                'nchanging': nchanging,
                'position': position,
                'helicities': helicities,
                'allowed_hel': allowed_hel,
                'ncomb': len(allowed_hel_pairs),
                'dimension': math.prod(len(i) for i in helicities),
            }
            production._ms_density_static = prod_static

        iden_p = prod_static['iden_p']
        sym_factor_prod_ident = prod_static['sym_factor_prod_ident']
        init_part = prod_static['init_part']
        position = prod_static['position']
        helicities = prod_static['helicities']
        allowed_hel = prod_static['allowed_hel']
        ncomb = prod_static['ncomb']
        dimension = prod_static['dimension']

        # ------------------------------------------------------------------
        # Normalization
        # ------------------------------------------------------------------
        dec_diag = 1.0
        prod_color = 1
        prod_denominators = 1

        density_prod = self.get_density(production,
                                        position,
                                        allowed_hel,
                                        ncomb,
                                        dimension) \
            if prod_density_cached is None else prod_density_cached

        # ------------------------------------------------------------------
        # Symmetry factor:
        # For each parent-PDG group with N identical parents and decay-channel
        # multiplicities {n_k}, the factor that belongs to the denominator is:
        #   sym_group = (Π_k n_k!) / (N!)
        # and sym_factor_decay = Π_groups sym_group.
        # ------------------------------------------------------------------
        sym_factor_decay = 1.0

        # Canonical decay-channel signature: sorted final-state PDGs only.
        def _decay_signature(dec_evt):
            pdgs = []
            for p in dec_evt:
                if p.status == 1:
                    pdgs.append(int(p.pid))
            pdgs.sort()
            return tuple(pdgs)
        
        # ------------------------------------------------------------------
        # Build total decay density matrix as tensor product
        # ------------------------------------------------------------------
        decaying_idx = 0
        density_dec = None

        for pdg, decay_event_list in decays.items():
            N = len(decay_event_list)

            # decay symmetry for this PDG group
            if N > 1:
                # Fast path for N==2 avoids building multiplicity maps.
                if N == 2:
                    if _decay_signature(decay_event_list[0]) != _decay_signature(decay_event_list[1]):
                        sym_factor_decay *= 0.5
                else:
                    sig_counts = {}
                    for evt in decay_event_list:
                        sig = _decay_signature(evt)
                        sig_counts[sig] = sig_counts.get(sig, 0) + 1
                    sym = 1
                    for nk in sig_counts.values():
                        if nk > 1:
                            sym *= math.factorial(nk)
                    sym_factor_decay *= (sym / float(math.factorial(N)))

            # particle properties for this parent PDG
            width = decay_dict[pdg][0]
            mass = decay_dict[pdg][1]
            color = decay_dict[pdg][2]
            spin = decay_dict[pdg][3]

            for i_decay_event in range(N):
                current_decay_event = decay_event_list[i_decay_event]

                # boost to lab frame using corresponding production particle momentum
                part = init_part[decaying_idx + i_decay_event]
                boost = -1 * lhe_parser.FourMomentum(part)
                boost.E *= -1
                current_decay_event.boost(boost)

                density_dec_tmp = self.get_density(
                    current_decay_event,
                    position=[1],
                    allow_hel=helicities[decaying_idx + i_decay_event],
                    ncomb=len(helicities[decaying_idx + i_decay_event]),
                    dimension=len(helicities[decaying_idx + i_decay_event])
                )

                if density_dec is None:
                    density_dec = density_dec_tmp
                else:
                    density_dec = density_dec.tensor_product(density_dec_tmp)

                # keep your normalization updates
                if MEdenom_decay is None:
                    dec_diag *= density_dec_tmp.trace().real
                dec_diag /= (color * spin)
                prod_color *= color
                D = complex(0, mass * width)
                prod_denominators *= (D * D.conjugate())
            

            decaying_idx += N

        # ------------------------------------------------------------------
        # Contract production and decay density matrices
        # ------------------------------------------------------------------
        me = density_dec.scalar_multiplication(density_prod)

        # ------------------------------------------------------------------
        # include production identical-final-state symmetry factor
        # ------------------------------------------------------------------
        denominator = iden_p * sym_factor_prod_ident * prod_color * prod_denominators * sym_factor_decay
        me = me.real / denominator

        #print(f"production = {production}")
        #print(f"decays = {decays}")
        if MEdenom_prod is None:
            prod_diag = density_prod.trace().real 
        else: 
            prod_diag = MEdenom_prod
        prod_diag /= (iden_p * sym_factor_prod_ident)
        if MEdenom_decay is not None:
            dec_diag *= MEdenom_decay 
        return me, density_prod, prod_diag, dec_diag


    def get_density_matrix_indices(self, nhel_decay):
        #print("------")
        #print(f"get_density_matrix_indices , nhel_decay = {nhel_decay}")
        diag = [sum(range(nhel_decay, nhel_decay - i, -1)) for i in range(nhel_decay)]
        off_diag = [i for i in list(range(nhel_decay * (nhel_decay + 1) // 2)) if i not in diag]
        return diag, off_diag

    def get_density_matrix_element_from_label(matrix, label):
        if label in label_to_index:
            i, j = label_to_index[label]
            return matrix[i, j]
        else:
            raise ValueError(f"Label {label} is not valid for this matrix size: {matrix.shape}.")

    def get_allowed_hel(self, list_hels):
        # list_hels is a list of lists with all possible helicities of the decaying particles, e.g.
        # [[1,-1], [1,0,-1]] - we need to construct a list of lists with all possible helicities
        # [ [1,1] , [1,0], [1,-1], [-1,1], ... ] which should eventually be converted into a flat list
        # [ 1, 1, 1, 0, 1, -1, ... ]
        key = tuple(tuple(hels) for hels in list_hels)
        # Cache allowed helicities - they depend only on spins
        # avoid rebuilding allowed helicities per trial
        if not hasattr(self, '_allowed_hel_cache'):
            self._allowed_hel_cache = {}
        if key in self._allowed_hel_cache:
            return self._allowed_hel_cache[key]

        helicity_combinations = [list(l) for l in product(*list_hels)]
        concatenated_hel_list = list(chain.from_iterable(helicity_combinations))
        out = (helicity_combinations, concatenated_hel_list)
        self._allowed_hel_cache[key] = out
        return out  

    def get_density(self, event, position, allow_hel, ncomb, dimension):
        orig_order = getattr(event, '_ms_orig_order_for_density', None)
        if orig_order is None:
            _, orig_order, _, _ = self.get_pdir(event)
            event._ms_orig_order_for_density = orig_order

        # Fast path: single-point momentum extraction without permutation
        # construction. With apply_flavor_grouping, orig_order contains the
        # merged-particle PDGs (e.g. 81) while the event carries the raw
        # PDGs (e.g. 1, 2, 3): get_momenta must be told about the merge so
        # that final.index(pdg) succeeds.
        try:
            p = event.get_momenta(orig_order, merged_map=self._revert_merged or None)
        except Exception:
            # Safety fallback for unusual event structures.
            all_p = event.get_all_momenta(orig_order, merged_map=self._revert_merged or None)
            assert len(all_p) == 1, "Error: get_density can only be called for a single phase-space point"
            p = all_p[0]
        P = rwgt_interface.ReweightInterface.invert_momenta(p)
        # f77_density runs `flavormapping` on the pdgs we pass: when the ME
        # legs use merged-particle IDs, we must pass the concrete raw PDGs
        # from the event for this permutation so the flavormapping picks
        # the right per-particle flavor index for GET_DENSITY. Same logic
        # as calculate_matrix_element's pdg_for_call handling.
        pdg_template = list(orig_order[0]) + list(orig_order[1])
        merged_particles = self.model.get('merged_particles') or {}
        need_raw_pdg = (self._revert_merged and
                        any(abs(pid) in merged_particles for pid in pdg_template))
        if need_raw_pdg:
            pdgs = event.get_pdg(p)
        else:
            pdgs = pdg_template
        n_changing = len(position)
        if n_changing == 0:
            raise ValueError("Error in get_density: 'position' must contain at least one position index")
        if len(allow_hel) % n_changing != 0:
            raise ValueError("Error in get_density: inconsistent 'allow_hel' and 'position' lengths")
        density_array = self.f2py_module.py_get_density(pdgs=pdgs,
                                                        procid=-1,
                                                        p=P,
                                                        pos=position,
                                                        allow_hel=allow_hel,
                                                        n_comb=ncomb,
                                                        alphas=event.aqcd,
                                                        npdg=len(pdgs))
        density_matrix = madspin.DensityMatrix(density_array,
                                               n_changing,
                                               allow_hel,
                                               dimension)
        return density_matrix

   
    def get_inter_value(self,event,nhel):
        """routine to return all the possible inter for an event"""
        
        pdir,orig_order = self.get_pdir(event)
        	
        if pdir in self.all_amp:
            all_p = event.get_all_momenta(orig_order)
            for p in all_p:
#                print(pdir,'Momenta=',p)
                P = rwgt_interface.ReweightInterface.invert_momenta(p)
#               print("Momenta =",P,"\n")
                IC = [1]*len(p)
                amp = []
                jamp = []
                inter = []
           
                for i,hel in enumerate(nhel):
                    #print(f"hel = {hel}")		
                    amp.append(self.all_amp[pdir](P,hel,IC))
                    jamp.append(self.all_jamp[pdir](amp[i]))
                #print(f"len(jamp) = {len(jamp)}")
                for i in range(len(jamp)): 
                    for j in range(len(jamp)): 
                        inter.append(self.all_inter[pdir](jamp[i],jamp[j]))
                return inter
        else : 
            self.all_amp[pdir],self.all_jamp[pdir],self.all_inter[pdir],self.all_matrix[pdir]= self.get_mymod(pdir,'INTER')

        return self.get_inter_value(event,nhel) 


    def get_nhel(self,event,position):

        pdir,orig_order, prefix, pos = self.get_pdir(event)
        if pdir in self.all_nhel:
            iden,NHEL = self.all_nhel[pdir]
            if position == -1:
                return iden
            nhel = rwgt_interface.ReweightInterface.invert_momenta(NHEL)
            groups = {} 
            nhel = sorted(nhel) 
            for item in nhel:
                a = item.copy()
                del a[position]
                t = tuple(a)
                groups.setdefault(t, []).append(item)
                grouped = list(groups.values())
            return grouped,iden
        else:
            #transer nhel information from fortran to wrapper
            getattr(self.f2py_module, '%sget_nhel_entry' % prefix.lower())()
            #transer now to python dictionary
            nhel = getattr(getattr(self.f2py_module, '%sprocess_nhel' % prefix.lower()), '%snhel' %prefix.lower())
            iden = getattr(self.f2py_module, 'get_idens')()[pos]
            self.all_nhel[pdir] = (iden, nhel)
            return self.get_nhel(event,position)


    def get_iden(self, event):
        # DEBUGGING REMOVE
        #print("---- DEBUG ---")
        #pdgs, allproc = self.f2py_module.get_pdg_order()
        #idens = self.f2py_module.get_idens()
        #
        #print("len(pdgs) =", len(pdgs))
        #print("len(idens) =", len(idens))
        #   
        #for i in range(len(idens)):
        #    print(i, pdgs[i], idens[i])
        #print("--- END")
        # END REMOVE

        # get_pdir returns (pdir, orig_order, prefix, pos)
        _, _, _, pos = self.get_pdir(event)
        idens = self.f2py_module.get_idens()
        #print(f"idens = {idens} , pos = {pos}")
        return idens[pos]
    

    def get_mymod(self,pdir,MODE): 
        
        all_prefix = self.f2py_module.get_prefix()
        tag = [t for t in self.all_me if self.all_me[t]['pdir'] == pdir][0]
        return 



    def get_pdir(self,event):
        # Use the merged-PDG tag (same as calculate_matrix_element). MadSpin's
        # all_me is keyed by the merged-particle representation when
        # apply_flavor_grouping is on; passing raw event PDGs here would miss
        # the lookup with KeyError, e.g. ((-2, 2), (21, 23)) when the table
        # is keyed by ((-81, 81), (21, 23)).
        tag, order = event.get_tag_and_order(self._revert_merged or None)
        try:
            orig_order = self.all_me[tag]['order']
        except Exception:
            # try to pass to full anti-particles for 1->N
            init, final = tag
            if len(init) == 2:
                raise
            init = (-init[0],)
            final = tuple(-i for i in final)
            tag = (init, final)
            orig_order = self.all_me[tag]['order']
        pdir = self.all_me[tag]['pdir']
        prefix, pos = self.pdg2prefix[tuple(list(orig_order[0]) + list(orig_order[1]))]
        #misc.sprint(f"get_pdir: pdir = {pdir} , orig_order = {orig_order} , prefix = {prefix}")
        return pdir,orig_order, prefix, pos

    model_init = True
    def calculate_matrix_element(self, event):
        """routine to return the matrix element"""

        tag, order = event.get_tag_and_order(self._revert_merged or None)
        try:
            orig_order = self.all_me[tag]['order']
        except Exception:
            # try to pass to full anti-particles for 1->N
            init, final = tag
            if len(init) == 2:
                raise
            init = (-init[0],)
            final = tuple(-i for i in final)
            tag = (init, final)
            orig_order = self.all_me[tag]['order']
        pdir = self.all_me[tag]['pdir']
        if pdir in self.all_f2py:
            all_p = event.get_all_momenta(orig_order, merged_map=self._revert_merged or None)
            if self.options['identical_particle_in_prod_and_decay'] == "crash" and\
                len(all_p)> 1:
                raise Exception("Ambiguous particle in production and decay. crash as requested by 'identical_particle_in_prod_and_decay'")
            # When the ME legs use merged-particle IDs (apply_flavor_grouping
            # is on for MadSpin), smatrixhel still needs the concrete raw PDGs
            # from the event for the current permutation; otherwise it cannot
            # pick the right flavor branch and returns 0.  See the analogous
            # logic in ReweightInterface.calculate_matrix_element.
            pdg_template = list(orig_order[0]) + list(orig_order[1])
            merged_particles = self.model.get('merged_particles') or {}
            need_raw_pdg = (self._revert_merged and
                            any(abs(p) in merged_particles
                                for p in pdg_template))
            out = 0
            for p in all_p:
                pdg_for_call = event.get_pdg(p) if need_raw_pdg else pdg_template
                p_inv = rwgt_interface.ReweightInterface.invert_momenta(p)
                if event[0].color1 == 599 and event.aqcd==0:
                    new_value = self.all_f2py[pdir](pdg_for_call, p_inv, 0.113, 0)
                else:
                    new_value = self.all_f2py[pdir](pdg_for_call, p_inv, event.aqcd, event.scale, -1)
                if self.options['identical_particle_in_prod_and_decay'] == "average":
                    out += new_value
                else:
                    if abs(out)< abs(new_value):
                        out = new_value
                if self.options['identical_particle_in_prod_and_decay'] == 'first':
                    return out
            if self.options['identical_particle_in_prod_and_decay'] == "average":
                return out/len(all_p)
            else:
                return out
        else:
            # First time we see a new ``pdir`` for this MadSpin instance:
            # load the freshly-compiled f2py extension once and cache a
            # smatrixhel lambda per pdir. The .so / pdg2prefix only need
            # to be loaded the first time we hit this branch — subsequent
            # ``pdir``s reuse ``self.f2py_module`` so we don't re-trigger
            # the spec-from-file-location load (which is fine on Linux
            # but wasteful, and on macOS would re-walk the install_name
            # bookkeeping every time).
            if not hasattr(self, 'f2py_module'):
                sp_path = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses')
                if sys.path[0] != sp_path:
                    sys.path.insert(0, sp_path)

                mymod = self._load_f2py_matrix_module(sp_path)
                self.f2py_module = mymod

                all_prefix = self.f2py_module.get_prefix()
                all_pdg, all_procid = self.f2py_module.get_pdg_order()
                self.pdg2prefix = {}
                for i, pdg in enumerate(all_pdg):
                    pdg = tuple([x for x in pdg if x != 0])
                    self.pdg2prefix[tuple(pdg)] = (str(all_prefix[i].decode()).strip(), i)

                if self.model_init:
                    self.model_init = False
                    with misc.chdir(sp_path):
                        if not os.path.exists(pjoin(self.path_me, 'Cards','param_card.dat')) and \
                                os.path.exists(pjoin(self.path_me,'param_card.dat')):
                            mymod.initialise(pjoin(self.path_me,'param_card.dat'))
                        else:
                            mymod.initialise(pjoin(self.path_me, 'Cards','param_card.dat'))
            mymod = self.f2py_module

            #if Rpath linking is not working the below code can be an alternative:
            #import ctypes
            #exts = ['so','dylib','dll']
            #for ext in exts:
            #    me_library = pjoin(self.path_me, self.ms_me_subdir, 'SubProcesses', pdir, 'libme%s.%s' % (pdir, ext))
            #    if os.path.exists(me_library):
            #        break
            # ctypes.CDLL(me_library)

            if self.model_init:
                self.model_init = False
                with misc.chdir(pjoin(self.path_me, 'madspin_me', 'SubProcesses', pdir)):
                    with misc.stdchannel_redirected(sys.stdout, os.devnull):
                        if not os.path.exists(pjoin(self.path_me, 'Cards','param_card.dat')) and \
                                os.path.exists(pjoin(self.path_me,'param_card.dat')):
                            mymod.initialise(pjoin(self.path_me,'param_card.dat'))
                        else:
                            mymod.initialise(pjoin(self.path_me, 'Cards','param_card.dat'))
            # pdg is passed per call (not captured here) so that callers can
            # substitute concrete event PDGs when the ME legs use merged IDs.
            self.all_f2py[pdir] = lambda pdg_list, *args : mymod.smatrixhel(pdg_list, 0, *args)
            return self.calculate_matrix_element(event)
        
        
    def generate_all_matrix_element(self):
        
        # 1. compute the production matrix element -----------------------------
        processes = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
        processes += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search(r'^\s*add\s+process', line)]
        # 2. compute the decay matrix-element
        decay_text = []
        processes_decay = []
        for decays in self.list_branches.values():
            for decay in  decays:
                if '=' not in decay:
                    decay += ' QCD=99'
                if ',' in decay:
                    decay_text.append('(%s)' % decay)
                else:
                    decay_text.append(decay)
                processes_decay.append(decay)            
        decay_text = ', '.join(decay_text)
        processes += []
        
        #handle NLO
        new_processes = []
        for proc in processes:
            # deal with @ syntax need to move it after the decay specification
            if '@' in proc:
                proc, proc_nb = proc.split('@')
                try:
                    proc_nb = int(proc_nb)
                except ValueError:
                    raise MadSpinError('MadSpin didn\'t allow order restriction after the @ comment: \"%s\" not valid' % proc_nb)
                proc_nb = '@ %i' % proc_nb 
                if self.options['global_order_coupling']:
                    proc_nb = '%s %s' % (proc_nb, self.options['global_order_coupling'])
            else:
                if self.options['global_order_coupling']:
                    proc_nb = '@0 %s ' % self.options['global_order_coupling']    
                
            rwgt_interface.ReweightInterface.get_LO_definition_from_NLO()        
        
        raise Exception

if __name__ == '__main__':
    
    a = MadSpinInterface()
    a.cmdloop()
    
    


        
