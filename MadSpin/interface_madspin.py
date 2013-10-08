################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
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
""" Command interface for MadSpin """
from __future__ import division
import logging
import math
import os
import re
import shutil


pjoin = os.path.join
if '__main__' == __name__:
    import sys
    sys.path.append(pjoin(os.path.dirname(__file__), '..'))

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.master_interface as master_interface
import madgraph.various.misc as misc
import madgraph.various.banner as banner

import models.import_ufo as import_ufo
import MadSpin.decay as madspin

logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr
cmd_logger = logging.getLogger('cmdprint2') # -> print



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
        
        self.options = {'max_weight': -1, 'BW_effect': 1, 
                        'curr_dir': os.path.realpath(os.getcwd()),
                        'Nevents_for_max_weigth': 0,
                        'max_weight_ps_point': 400,
                        'BW_cut':-1,
                        'zeromass_for_max_weight':5,
                        'nb_sigma':0}
        

        
        self.events_file = None
        self.decay_processes = {}
        self.list_branches = {}
        self.to_decay={}
        self.mg5cmd = master_interface.MasterCmd()
        self.seed = None
        
        
        if event_path:
            logger.info("Extracting the banner ...")
            self.do_import(event_path)
            
            
    def do_import(self, inputfile):
        """import the event file"""
        
        # change directory where to write the output
        self.options['curr_dir'] = os.path.realpath(os.path.dirname(inputfile))
        if os.path.basename(os.path.dirname(os.path.dirname(inputfile))) == 'Events':
            self.options['curr_dir'] = pjoin(self.options['curr_dir'], 
                                                      os.path.pardir, os.pardir)
        
        if not os.path.exists(inputfile):
            if inputfile.endswith('.gz'):
                if not os.path.exists(inputfile[:-3]):
                    raise self.InvalidCmd('No such file or directory : %s' % inputfile)
                else: 
                    inputfile = inputfile[:-3]
            elif os.path.exists(inputfile + '.gz'):
                inputfile = inputfile + '.gz'
            else: 
                raise self.InvalidCmd('No such file or directory : %s' % inputfile)
        
        if inputfile.endswith('.gz'):
            misc.call(['gunzip', inputfile])
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
            self.load_model(model_name, mg_names, complex_mass)
        else:
            raise self.InvalidCmd('Only UFO model can be loaded in MadSpin.')
        
        # check particle which can be decayed:
        self.final_state = set()
        for line in self.banner.proc_card:
            line = ' '.join(line.strip().split())
            if line.startswith('generate'):
                self.final_state.update(self.mg5cmd.get_final_part(line[8:]))
            elif line.startswith('add process'):
                self.final_state.update(self.mg5cmd.get_final_part(line[11:]))
            elif line.startswith('define'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)            
            elif line.startswith('set'):
                self.mg5cmd.exec_cmd(line, printcmd=False, precmd=False, postcmd=False)
                    
            
                

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
        
        if self.model and not self.model['case_sensitive']:
            decaybranch = decaybranch.lower()
        
        decay_process, init_part = self.decay.reorder_branch(decaybranch)
        if not self.list_branches.has_key(init_part):
            self.list_branches[init_part] = []
        self.list_branches[init_part].append(decay_process)
        del decay_process, init_part    
        
    
    def check_set(self, args):
        """checking the validity of the set command"""
        
        if len(args) < 2:
            raise self.InvalidCmd('set command requires at least two argument.')
        
        valid = ['maz_weight','seed','curr_dir']
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

        
    def do_set(self, line):
        """ add one of the options """
        
        args = self.split_arg(line)
        self.check_set(args)
        
        if args[0] in  ['max_weight', 'BW_effect']:
            self.options[args[0]] = args[1]
        elif args[0] == 'seed':
            import random
            random.seed(int(args[1]))
            self.seed = int(args[1])
        else:
             self.options[args[0]] = int(args[1])
    
    def complete_set(self,  text, line, begidx, endidx):
        
     try:
        args = self.split_arg(line[0:begidx])

        # Format
        if len(args) == 1:
            opts = self.options.keys() + ['seed']
            return self.list_completion(text, opts) 
        elif len(args) == 2:
            if args[1] == 'BW_effect':
                return self.list_completion(text, ['True', 'False']) 
     except Exception, error:
         print error
         
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
        
    
    def do_define(self, line):
        """ """
        return self.mg5cmd.do_define(line)
    
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
        
        if not self.list_branches:
            raise self.InvalidCmd("Nothing to decay ... Please specify some decay")
        if not self.events_file:
            raise self.InvalidCmd("No events files defined.")

    def help_launch(self):
        """help for the launch command"""
        
        print '''Running Madspin on the loaded events, following the decays enter
        An example of a full run is the following:
        import ../mssm_events.lhe.gz
        define sq = ur ur~
        decay go > sq j
        launch
        '''

    #@misc.mute_logger()
    def do_launch(self, line):
        """end of the configuration launched the code"""
        
        args = self.split_arg(line)
        self.check_launch(args)
        for part in self.list_branches.keys():
            if part in self.mg5cmd._multiparticles:
                if any(pid in self.final_state for pid in self.mg5cmd._multiparticles[part]):
                    break
            pid = self.mg5cmd._curr_model.get('name2pdg')[part]
            if pid in self.final_state:
                break
        else:
            logger.info("Nothing to decay1 ...")
            return
        

        model_line = self.banner.get('proc_card', 'full_model_line')

        if not self.seed:
            import random
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
        

        generate_all = madspin.decay_all_events(self, self.banner, self.events_file, 
                                                    self.options)
        generate_all.run()
                        
        self.branching_ratio = generate_all.branching_ratio
        evt_path = self.events_file.name
        try:
            self.events_file.close()
        except:
            pass
        misc.call(['gzip -f %s' % evt_path], shell=True)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        shutil.move(pjoin(self.options['curr_dir'],'decayed_events.lhe'), decayed_evt_file)
        misc.call(['gzip -f %s' % decayed_evt_file], shell=True)
        if not self.mother:
            logger.info("Decayed events have been written in %s.gz" % decayed_evt_file)
    
    
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
        base_model = import_ufo.import_model(name, decay=True)
        if not hasattr(base_model.get('particles')[0], 'partial_widths'):
            msg = 'The UFO model does not include partial widths information.\n'
            msg += 'Impossible to use analytical formula, will use MG5/MadEvent (slower).'
            logger.warning(msg)

        if use_mg_default:
            base_model.pass_particles_name_in_mg_default()
        if complex_mass:
            base_model.change_mass_to_complex_scheme()
        
        self.model = base_model
        self.mg5cmd._curr_model = self.model
        self.mg5cmd.process_model()
        

if __name__ == '__main__':
    
    a = MadSpinInterface()
    a.cmdloop()
    
    


        
