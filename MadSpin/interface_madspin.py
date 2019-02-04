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
        self.add_param('Nevents_for_max_weigth', 0)
        self.add_param("max_weight_ps_point", 400)
        self.add_param('BW_cut', -1)
        self.add_param('nb_sigma', 0.)
        self.add_param('ms_dir', '')
        self.add_param('max_running_process', 100)
        self.add_param('onlyhelicity', False)
        self.add_param('spinmode', "madspin", allowed=['madspin','none','onshell'])
        self.add_param('use_old_dir', False, comment='should be use only for faster debugging')
        self.add_param('run_card', '' , comment='define cut for spinmode==none. Path to run_card to use')
        self.add_param('fixed_order', False, comment='to activate fixed order handling of counter-event')
        self.add_param('seed', 0, comment='control the seed of madspin')
        self.add_param('cross_section', {'__type__':0.}, comment="forcing normalization of cross-section after MS (for none/onshell)" )
        self.add_param('new_wgt', 'cross-section' ,allowed=['cross-section', 'BR'], comment="if not consistent number of particles, choose what to do for the weight. (BR: means local according to number of part, cross use the force cross-section")
        self.add_param('input_format', 'auto', allowed=['auto','lhe', 'hepmc', 'lhe_no_banner'])
        
        
    ############################################################################
    ##  Special post-processing of the options                                ## 
    ############################################################################
    def post_set_ms_dir(self, value, change_userdefine, raiseerror):
        """ special handling for set ms_dir """
        
        self.__setitem__('curr_dir', value, change_userdefine=change_userdefine)
        
    ############################################################################
    def post_set_seed(self, value, change_userdefine, raiseerror):
        """ special handling for set seed """
        
        random.seed(value)

    ############################################################################        
    def post_set_run_card(self, value, change_userdefine, raiseerror):
        """ special handling for set run_card """
        
        if value == 'default':
            self.run_card = None
        elif os.path.isfile(value):
            self.run_card = banner.RunCard(value)
            
        args = value.split()
        if  len(args) >2:
            if not self.options['run_card']:
                self.run_card =  banner.RunCardLO()
                self.run_card.remove_all_cut()
            self.run_card[args[0]] = ' '.join(args[1:])
            
        
    ############################################################################
    def post_fixed_order(self, value, change_userdefine, raiseerror):
        """ special handling for set fixed_order """
        
        if value:
            logger.warning('Fix order madspin fails to have the correct scale information. This can bias the results!')
            logger.warning('Not all functionalities of MadSpin handle this mode correctly (only onshell mode so far).')


class MadSpinInterface(extended_cmd.Cmd):
    """Basic interface for madspin"""
    
    prompt = 'MadSpin>'
    debug_output = 'MS_debug'
    
    
    @misc.mute_logger()
    def __init__(self, event_path=None, *completekey, **stdin):
        """initialize the interface with potentially an event_path"""
        
        cmd_logger.info('************************************************************')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('*           W E L C O M E  to  M A D S P I N               *')
        cmd_logger.info('*                                                          *')
        cmd_logger.info('************************************************************')
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)
        
        self.decay = madspin.decay_misc()
        self.model = None
        self.mode = "madspin" # can be flat/bridge change the way the decay is done.
                              # note amc@nlo does not support bridge.
        
        self.options = MadSpinOptions()
        
        self.events_file = None
        self.decay_processes = {}
        self.list_branches = {}
        self.to_decay={}
        self.mg5cmd = master_interface.MasterCmd()
        self.seed = None
        self.err_branching_ratio = 0
        
        
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
            if not self.options['Nevents_for_max_weigth']:
                nevents = int(self.banner.get_detail('run_card', 'nevents'))
                N_weight = max([75, int(3*nevents**(1/3))])
                self.options['Nevents_for_max_weigth'] = N_weight
                N_sigma = max(4.5, math.log(nevents,7.7))
                self.options['nb_sigma'] = N_sigma
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = float(self.banner.get_detail('run_card', 'bwcutoff'))
        else:
            if not self.options['Nevents_for_max_weigth']:
                self.options['Nevents_for_max_weigth'] = 75
                self.options['nb_sigma'] = 4.5
            if self.options['BW_cut'] == -1:
                self.options['BW_cut'] = 15.0
                
                
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
        if not process:
            msg = 'Invalid proc_card information in the file (no generate line):\n %s' % self.banner['mg5proccard']
            raise Exception, msg
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
            elif line.startswith('set'):
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

        decay_process, init_part = self.decay.reorder_branch(decaybranch)
        if not self.list_branches.has_key(init_part):
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
        
        elif args[0] == 'BW_effect':
            if args[1] in [0, False,'.false.', 'F', 'f', 'False', 'no']:
                args[1] = 0
            elif args[1] in [1, True,'.true.', 'T', 't', 'True', 'yes']:
                args[1] = 1
            else:
                raise self.InvalidCmd('second argument should be either T or F.')
        
        elif args[0] == 'curr_dir':
            if not os.path.isdir(args[1]):
                raise self.InvalidCmd('second argument should be a path to a existing directory')
        
        elif args[0] == "spinmode":
            if args[1].lower() not in ["full", "onshell", "none"]:
                raise self.InvalidCmd("spinmode can only take one of those 3 value: full/onshell/none")
             
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
        
    def do_set(self, line):
        """ add one of the options """
        
        args = self.split_arg(line)
        self.check_set(args)

        self.options[args[0]] = ' '.join(args[1:])
        

    def complete_set(self,  text, line, begidx, endidx):
        

        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            opts = self.options.keys() + ['seed', "spinmode"]
            return self.list_completion(text, opts) 
        elif len(args) == 2:
            if args[1] == 'BW_effect':
                return self.list_completion(text, ['True', 'False']) 
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
        
        print 'syntax: set OPTION VALUE'
        print ''
        print '-- assign to a given option a given value'
        print '   - set max_weight VALUE: pre-define the maximum_weight for the reweighting'
        print '   - set BW_effect True|False: [default:True] reshuffle the momenta to describe'
        print '       corrrectly the Breit-Wigner of the decayed particle'
        print '   - set seed VALUE: fix the value of the seed to a given value.'
        print '       by default use the current time to set the seed. random number are'
        print '       generated by the python module random using the Mersenne Twister generator.'
        print '       It has a period of 2**19937-1.'
        print '   - set max_running_process VALUE: allow to limit the number of open file used by the code'
        print '       The number of running is raising like 2*VALUE'
        print '   - set spinmode=none: mode with simple file merging. No spin correlation attempt.'
        print '       This mode allows 3 (and more) body decay.'
    
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
        except Exception,error:
            misc.sprint(error)
            
    def complete_decay(self, *args):
        """ """
        try:
            return self.mg5cmd.complete_generate(*args)
        except Exception,error:
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
            raise Exception, "MadSpin requires LHEF version 3 when running with matching/merging"

    def help_launch(self):
        """help for the launch command"""
        
        print '''Running Madspin on the loaded events, following the decays enter
        An example of a full run is the following:
        import ../mssm_events.lhe.gz
        define sq = ur ur~
        decay go > sq j
        launch
        '''

    @misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""
        
        if self.options["spinmode"] in ["none"]:
            return self.run_bridge(line)
        elif self.options["spinmode"] == "onshell":
            return self.run_onshell(line)
        elif self.options["spinmode"] == "bridge":
            raise Exception, "Bridge mode not available."
        
        if self.options['ms_dir'] and os.path.exists(pjoin(self.options['ms_dir'], 'madspin.pkl')):
            return self.run_from_pickle()
        
    
        args = self.split_arg(line)
        self.check_launch(args)
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                if any(pid in self.final_state for pid in self.mg5cmd._multiparticles[part]):
                    break
            pid = self.mg5cmd._curr_model.get('name2pdg')[part]
            if pid in self.final_state:
                break
#        else:
#            logger.info("Nothing to decay ...")
#            return
        

        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.options['seed']:
            self.options['seed'] = random.randint(0, int(30081*30081))
            #self.do_set('seed %s' % self.seed)
            logger.info('Will use seed %s' % self.options['seed'])
            self.history.insert(0, 'set seed %s' % self.options['seed'])

        if self.options['seed'] > 30081*30081: # can't use too big random number
            msg = 'Random seed too large ' + str(self.options['seed']) + ' > 30081*30081'
            raise Exception, msg

        #self.options['seed'] = self.seed
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)
        
        
        self.update_status('generating Madspin matrix element')
        generate_all = madspin.decay_all_events(self, self.banner, self.events_file, 
                                                    self.options)
        
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

    def run_from_pickle(self):
        import madgraph.iolibs.save_load_object as save_load_object
        
        generate_all = save_load_object.load_from_file(pjoin(self.options['ms_dir'], 'madspin.pkl'))
        # Re-create information which are not save in the pickle.
        generate_all.evtfile = self.events_file
        generate_all.curr_event = madspin.Event(self.events_file, self.banner ) 
        generate_all.mgcmd = self.mg5cmd
        generate_all.mscmd = self 
        generate_all.pid2width = lambda pid: generate_all.banner.get('param_card', 'decay', abs(pid)).value
        generate_all.pid2mass = lambda pid: generate_all.banner.get('param_card', 'mass', abs(pid)).value
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
                raise Exception, """The directory %s is specific to a mass spectrum. 
                Your event file is not compatible with this one. (Different param_card: %s different)
                orig block:
                %s
                new block:
                %s""" \
                % (self.options['ms_dir'], name, orig_block, block)

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
                raise Exception, "fail to recognized input format automatically"
                
        if self.options['input_format'] in ['lhe', 'lhe_no_banner']:
            orig_lhe = lhe_parser.EventFile(filename)
            if self.options['input_format'] == 'lhe_no_banner':
                orig_lhe.allow_empty_event = True
                
        elif self.options['input_format'] in ['hepmc']:
            import madgraph.various.hepmc_parser as hepmc_parser
            orig_lhe = hepmc_parser.HEPMC_EventFile(filename)
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
            raise Exception, msg

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
                    raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed. One workaround is to force the final cross-section manually.")
                    
                     
        # Compute the branching ratio.
        if not self.options['cross_section']:
            br = 1
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
                    raise self.InvalidCmd("The bridge mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed.")
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
                print "decaying event number %s [%s s]" % (counter, time.time()-start)
            counter +=1
            
            # use random order for particles to avoid systematics when more than 
            # one type of decay is asked.
            particles = [p for p in event if int(p.status) == 1.0]
            random.shuffle(particles)
            ids = [particle.pid for particle in particles]
            br = 1 #br for that particular events (for special/weighted case)
            hepmc_output = lhe_parser.Event() #for hepmc case: collect the decay particle
            for i,particle in enumerate(particles):
                #misc.sprint(i, particle.pdg, particle.pid)
                #misc.sprint(self.final_state, evt_decayfile)
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
                            decay = decay_file.next()
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
                        if restrict_file and j not in restrict_file:
                            raise Exception # Do not see how this can happen
                        mg5.exec_cmd("add process %s" % proc2)
                    mg5.exec_cmd("output %s -f" % decay_dir)
                else:
                    mg5.exec_cmd("generate %s" % proc)
                    mg5.exec_cmd("output %s -f" % decay_dir)
                
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
                        except Exception,error:
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
                    run_card.write(pjoin(decay_dir, "Cards", "run_card.dat"))
                    param_card = self.banner['slha']
                    open(pjoin(decay_dir, "Cards", "param_card.dat"),"w").write(param_card)
                    self.options['seed'] += 1
                    # actually creation
                    me5_cmd.exec_cmd("generate_events run_01 -f")
                    if output_width:
                        if cumul:
                            width += me5_cmd.results.current['cross']
                        else:
                            width *= me5_cmd.results.current['cross']
                    me5_cmd.exec_cmd("exit")                        
                    #remove pointless informat
                    misc.call(["rm", "Cards", "bin", 'Source', 'SubProcesses'], cwd=decay_dir)
                    misc.call(['tar', '-xzpvf', 'run_01_gridpack.tar.gz'], cwd=decay_dir)
            
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
                    except Exception,error:
                        logger.debug(error)
                        pass                 
                    self.me_int[decay_dir] = me5_cmd
                if self.options["run_card"]:
                    if hasattr(self, 'run_card'):
                        run_card = self.run_card
                    else:
                        self.run_card = banner.RunCard(self.options["run_card"])
                        run_card = self.run_card 
                else:
                    run_card = banner.RunCard(pjoin(decay_dir, "Cards", "run_card.dat"))
                run_card["nevents"] = int(1.2*nb_event)
                # Handle the banner of the output file
                if not self.seed:
                    self.seed = random.randint(0, int(30081*30081))
                    self.do_set('seed %s' % self.seed)
                    logger.info('Will use seed %s' % self.seed)
                    self.history.insert(0, 'set seed %s' % self.seed)
                run_card["iseed"] = self.seed
                run_card["systematics_program"] = 'None'
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
                misc.call(['run.sh', str(int(1.2*nb_event)), str(self.seed)], cwd=decay_dir)     
                out[i] = lhe_parser.EventFile(pjoin(decay_dir, 'events.lhe.gz'))            
            if cumul:
                break
        
        if not output_width:
            return out
        else:
            return out, width

    def run_onshell(self, line):
        """Run the onshell Algorithm"""
        
        # 1. Read the event file to check which decay to perform and the number
        #   of event to generate for each type of particle. (assume efficiency=1 for spin 0
        #   otherwise efficiency=2
        # 2. Generate the associated events
        # 3. generate the various matrix-element (production/decay/production+decay) 
        
        # 3. comput
        # 3. perform the merge of the events.
        #    if not enough events. re-generate the missing one.
        # First define an utility function for generating events when needed

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

        self.events_file.close()
        orig_lhe = lhe_parser.EventFile(self.events_file.name)
        if self.options['fixed_order']:
            orig_lhe.eventgroup = True

        #count the number of particle need to be decayed.
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
            raise Exception, msg

        self.options['seed'] = self.seed
        
        text = '%s\n' % '\n'.join([ line for line in self.history if line])
        self.banner.add_text('madspin' , text)


        # 2. Generate the events requested
        nevents_for_max = self.options['Nevents_for_max_weigth']
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
            for pdg, nb_needed in to_decay.items():
                # muliply by expected effeciency of generation
                spin = self.model.get_particle(pdg).get('spin')
                if spin ==1:
                    efficiency = 1.1
                else:
                    efficiency = 2.0
                    
                totwidth = self.banner.get('param_card', 'decay', abs(pdg)).value
                #check if a splitting is needed
                if nb_needed == nb_event:
                    nb_needed = int(efficiency*nb_needed) + nevents_for_max   
                    evt_decayfile[pdg], pwidth = self.generate_events(pdg, nb_needed, mg5, output_width=True)
                    if pwidth > 1.01*totwidth:
                        logger.warning('partial width (%s) larger than total width (%s) --from param_card--', pwidth, totwidth)
                    elif pwidth > totwidth:
                        pwidth = totwidth
                    br *= pwidth / totwidth
                elif nb_needed %  nb_event == 0:
                    nb_mult = nb_needed // nb_event
                    nb_needed = int(efficiency*nb_needed) +nevents_for_max *nb_mult
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches:
                        continue
                    elif len(self.list_branches[name]) == nb_mult:
                        evt_decayfile[pdg], pwidth = self.generate_events(pdg, nb_event, mg5, output_width=True)
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
                    part = self.model.get_particle(pdg)
                    name = part.get_name()
                    if name not in self.list_branches or len(self.list_branches[name]) == 0:
                        continue
                    raise self.InvalidCmd("The onshell mode of MadSpin does not support event files where events do not *all* share the same set of final state particles to be decayed.")
        self.branching_ratio = br
        self.efficiency = 1
        self.cross, self.error = self.banner.get_cross(witherror=True)
        self.cross *= self.branching_ratio
        self.error *= self.branching_ratio
        
        # 3. generate the various matrix-element
        self.update_status('generating Madspin matrix element')
        self.generate_all = madspin.decay_all_events_onshell(self, self.banner, self.events_file, 
                                                    self.options)
        self.generate_all.compile()
        self.all_me = self.generate_all.all_me
        self.all_f2py = {}
        
        #4. determine the maxwgt
        maxwgt = self.get_maxwgt_for_onshell(orig_lhe, evt_decayfile)
        
        
        #5. generate the decay 
        orig_lhe.seek(0)
        output_lhe = lhe_parser.EventFile(orig_lhe.name.replace('.lhe', '_decayed.lhe'), 'w')
        if self.options['fixed_order']:
            output_lhe.eventgroup = True
        
        self.banner.scale_init_cross(self.branching_ratio)
        self.banner.write(output_lhe, close_tag=False)
        
        
        self.efficiency =1.
        nb_try, nb_event = 0, len(orig_lhe)
        
        start = time.time()
        for curr_event,production in enumerate(orig_lhe):
            if self.options['fixed_order']:
                production, counterevt= production[0], production[1:]
            if curr_event and curr_event % 1000 == 0 and float(str(curr_event)[1:]) ==0:
                logger.info("decaying event number %s. Efficiency: %s [%s s]" % (curr_event, 1/self.efficiency, time.time()-start))
            while 1:
                nb_try +=1
                decays = self.get_decay_from_file(production, evt_decayfile, nb_event-curr_event)
                full_evt, wgt = self.get_onshell_evt_and_wgt(production, decays)
                if random.random()*maxwgt < wgt:
                    if self.options['fixed_order']:
                        full_evt = [full_evt] + [evt.add_decays(decays) for evt in counterevt]
                    break
            self.efficiency = curr_event/nb_try
            if self.options['fixed_order']:
                for evt in full_evt:
                    # change the weight associate to the event
                    evt.wgt *= self.branching_ratio
                    wgts = evt.parse_reweight()
                    for key in wgts:
                        wgts[key] *= self.branching_ratio 
            else:
                # change the weight associate to the event
                full_evt.wgt *= self.branching_ratio
                wgts = full_evt.parse_reweight()
                for key in wgts:
                    wgts[key] *= self.branching_ratio            
            output_lhe.write_events(full_evt)
            
        output_lhe.write('</LesHouchesEvents>\n')    
        self.efficiency = 1 # to let me5 to write the correct number of events
#        misc.sprint('Done so far. output written in %s' % output_lhe.name)
        
    
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
                    decay = decay_file.next()
                    break
                except StopIteration:
                    eff = self.efficiency
                    needed = 1.05 * nb_remain / eff
                    needed = min(50000, max(needed, 1000))
                    with misc.MuteLogger(["madgraph", "madevent", "ALOHA", "cmdprint"], [50,50,50,50]):
                        new_file = self.generate_events(particle.pdg, needed, self.mg5cmd, [decay_file_nb])
                    evt_decayfile[particle.pdg].update(new_file)
                    decay_file = evt_decayfile[particle.pdg][decay_file_nb]
                    continue
            
            out[particle.pdg].append(decay)
                        
        return out
        
    
    def get_maxwgt_for_onshell(self, orig_lhe, evt_decayfile):
        """determine the maximum weight for the onshell (or similar) strategy"""

        # event_decay is a dict pdg -> list of event file (contain the decay)
        
        
        nevents = self.options['Nevents_for_max_weigth']
        if nevents == 0 :
            nevents = 75
        
        all_maxwgt = []
        logger.info("Estimating the maximum weight")
        logger.info("*****************************")
        logger.info("Probing the first %s events with %s phase space points" % (nevents, self.options['max_weight_ps_point']))


        self.efficiency = 1. / self.options['max_weight_ps_point']
        start = time.time()
        for i in range(nevents):
            if i % 5 ==1:
                logger.info( "Event %s/%s :  %2fs" % (i, nevents, time.time()-start))
            maxwgt = 0
            orig_lhe.seek(0)
            base_event = orig_lhe.next()
            if self.options['fixed_order']:
                base_event = base_event[0]
            for j in range(self.options['max_weight_ps_point']):
                decays = self.get_decay_from_file(base_event, evt_decayfile, nevents-i)   
                #carefull base_event is modified by the following function                  
                _, wgt = self.get_onshell_evt_and_wgt(base_event, decays)
                maxwgt = max(wgt, maxwgt)
            all_maxwgt.append(maxwgt)
            
        all_maxwgt.sort(reverse=True)
        assert all_maxwgt[0] >= all_maxwgt[1]
        decay_tools=madspin.decay_misc()
        ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt)
        base_max_weight = 1.05 * (ave_weight+self.options['nb_sigma']*std_weight)

#        misc.sprint(all_maxwgt)
        for i in [20, 30, 40, 50]:
            if len(all_maxwgt) < i:
                break
            ave_weight, std_weight = decay_tools.get_mean_sd(all_maxwgt[:i])
            base_max_weight = max(base_max_weight, 1.05 * (ave_weight+self.options['nb_sigma']*std_weight))
                
            if all_maxwgt[1] > base_max_weight:
                base_max_weight = 1.05 * all_maxwgt[1]
        return base_max_weight
            
            
            

            
    def get_onshell_evt_and_wgt(self, production, decays):
        """ return the onshell wgt for the production event associated to the decays
            return also the full event with decay. 
            Carefull this modifies production event (pass to the full one)"""
        
        tag, order = production.get_tag_and_order()
        try:
            info = self.generate_all.all_me[tag]
        except:
            misc.sprint(self.generate_all.all_me)
            misc.sprint(production)
            misc.sprint(decays)
            raise
        import copy
        
        
        if hasattr(production, 'me_wgt'):
            production_me = production.me_wgt
        else:
            production_me = self.calculate_matrix_element(production)
            production.me_wgt = production_me
            
        decay_me = 1.0
        for pdg in decays:
            for dec in decays[pdg]:
                decay_me *= self.calculate_matrix_element(dec)
            random.shuffle(decays[pdg])

        full_event = lhe_parser.Event(str(production))
        full_event = full_event.add_decays(decays)
        full_me = self.calculate_matrix_element(full_event)
        return full_event, full_me/(production_me*decay_me)
        
        
    def calculate_matrix_element(self, event):
        """routine to return the matrix element"""        
        
        tag, order = event.get_tag_and_order()
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
            p = event.get_momenta(orig_order)
            p = rwgt_interface.ReweightInterface.invert_momenta(p)
            if event[0].color1 == 599 and event.aqcd==0:
                return self.all_f2py[pdir](p, 0.113, 0)
            else:
                return self.all_f2py[pdir](p, event.aqcd, 0)
        else:
            if sys.path[0] != pjoin(self.path_me, 'madspin_me', 'SubProcesses'):
                sys.path.insert(0, pjoin(self.path_me, 'madspin_me', 'SubProcesses'))
            
            mymod = __import__("%s.matrix2py" % (pdir))
            reload(mymod)
            mymod = getattr(mymod, 'matrix2py')  
            with misc.chdir(pjoin(self.path_me, 'madspin_me', 'SubProcesses', pdir)):
                with misc.stdchannel_redirected(sys.stdout, os.devnull):
                    mymod.initialisemodel(pjoin(self.path_me, 'Cards','param_card.dat'))
            self.all_f2py[pdir] = mymod.get_value  
            return self.calculate_matrix_element(event)
        
        
    def generate_all_matrix_element(self):
        
        # 1. compute the production matrix element -----------------------------
        processes = [line[9:].strip() for line in self.banner.proc_card
                     if line.startswith('generate')]
        processes += [' '.join(line.split()[2:]) for line in self.banner.proc_card
                      if re.search('^\s*add\s+process', line)]
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
                    raise MadSpinError, 'MadSpin didn\'t allow order restriction after the @ comment: \"%s\" not valid' % proc_nb
                proc_nb = '@ %i' % proc_nb 
            else:
                proc_nb = ''     
                
            rwgt_interface.ReweightInterface.get_LO_definition_from_NLO()        
        


if __name__ == '__main__':
    
    a = MadSpinInterface()
    a.cmdloop()
    
    


        
