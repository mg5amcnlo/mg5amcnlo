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

import logging
import os
import shutil

pjoin = os.path.join
if '__main__' == __name__:
    import sys
    sys.path.append(pjoin(os.path.dirname(__file__), '..'))

import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.various.misc as misc
import MadSpin.decay as madspin
import madgraph.various.banner as banner


logger = logging.getLogger('decay.stdout') # -> stdout
logger_stderr = logging.getLogger('decay.stderr') # ->stderr



class MadSpinInterface(extended_cmd.Cmd):
    """Basic interface for madspin"""
    
    prompt = 'MadSpin>'
    debug_output = 'MS_debug'
    
    def __init__(self, event_path=None,*completekey, **stdin):
        """initialize the interface with potentially an event_path"""
        
        extended_cmd.Cmd.__init__(self, *completekey, **stdin)
        
        self.decay = madspin.decay_misc()
        self.model = None
        
        self.options = {'max_weight': -1, 'BW_effect': 1, 
                        'curr_dir': os.getcwd()}
        
        logger.info("Extracting the banner ...")
        
        self.events_file = None
        self.decay_processes = {}
        self.list_branches = {}
        self.to_decay={}
        if event_path:
            self.do_import(event_path)
            
            
    def do_import(self, inputfile):
        """import the event file"""
        
        if not os.path.exists(inputfile):
            raise self.InvalidCmd('No such file or directory : %s' % inputfile)
        
        if inputfile.endswith('.gz'):
            misc.call(['gunzip', inputfile])
            inputfile = inputfile.replace(".gz","")
        
        # Read the banner of the inputfile
        self.events_file = open(inputfile)
        self.banner = banner.Banner(self.events_file)
        
        # Check the validity of the banner:
        if 'slha' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain model information')
        elif 'mg5proccard' not in self.banner:
            self.events_file = None
            raise self.InvalidCmd('Event file does not contain generation information')
        
        # load information
        process = self.banner.get_detail('proc_card', 'generate')
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
        
        decay_process, init_part = self.decay.reorder_branch(decaybranch)
        if not self.list_branches.has_key(init_part):
            self.list_branches[init_part] = []
        self.list_branches[init_part].append(decay_process)
        del decay_process, init_part    
        
    
    def check_set(self, args):
        """checking the validity of the set command"""
        
        if len(args) < 2:
            raise self.InvalidCmd('set command requires at least two argument.')
         
        if args[0] not in self.options:
            raise self.InvalidCmd('Unknown options')        
    
        if args[0] == 'max_weight':
            try:
                args[1] == float(args[1])
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
        
        if args[0] not in self.options:
            raise self.InvalidCmd('Unknown options')
        
        if args[0] in  ['max_weight', 'BW_effect']:
            self.options[args[0]] = args[1] 

    def check_launch(self, args):
        """check the validity of the launch command"""
        
        if not self.list_branches:
            raise self.InvalidCmd("Nothing to decay ... Please specify some decay")
        if not self.events_file:
            raise self.InvalidCmd("No events files defined.")

    def do_launch(self, line):
        """end of the configuration launched the code"""
        
        args = self.split_arg(line)
        self.check_launch(args)
    
        # Ask the user which particle should be decayed        
        particle_index=2
        counter=0
        for particle in self.final_state_compact.split():
            particle_index+=1
            if self.list_branches.has_key(str(particle)):
                counter+=1
                self.decay_processes[counter]=self.list_branches[str(particle)][0]
                # if there are several decay branches initiated by the same particle:
                #  use each of them in turn:
                if len(self.list_branches[str(particle)])>1: 
                    del self.list_branches[str(particle)][0] 
                    
        if not self.decay_processes:
            logger.info("Nothing to decay ...")
            return

        generate_all = madspin.decay_all_events(self.events_file,
                                              self.banner,
                                              self.decay_processes,
                                              self.prod_branches, 
                                              self.proc_option, 
                                              self.options['max_weight'], 
                                              self.options['BW_effect'],
                                              self.options['curr_dir'])

        evt_path = self.events_file.name
        misc.call(['gzip %s' % evt_path], shell=True)
        decayed_evt_file=evt_path.replace('.lhe', '_decayed.lhe')
        shutil.move(pjoin(self.options['curr_dir'],'decayed_events.lhe'), decayed_evt_file)
        misc.call(['gzip %s' % decayed_evt_file], shell=True)
        logger.info("Decayed events have been written in %s" % decayed_evt_file)
    

if __name__ == '__main__':
    
    a = MadSpinInterface()
    a.cmdloop()
    
    


        
