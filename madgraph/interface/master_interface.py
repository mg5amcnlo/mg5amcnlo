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
"""A user friendly command line interface to access all MadGraph5_aMC@NLO features.
   Uses the cmd package for command interpretation and tab completion.
"""


import atexit
import logging
import optparse
import os
import pydoc
import re
import subprocess
import sys
import traceback
import time

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.split(root_path)[0]
sys.path.insert(0, root_path)

#usefull shortcut
pjoin = os.path.join

import madgraph
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.interface.extended_cmd as cmd
import madgraph.interface.madgraph_interface as MGcmd
import madgraph.interface.loop_interface as LoopCmd
import madgraph.interface.amcatnlo_interface as amcatnloCmd
import madgraph.fks.fks_base as fks_base
import madgraph.iolibs.files as files
import madgraph.various.misc as misc

from madgraph import MG4DIR, MG5DIR, MadGraph5Error

logger = logging.getLogger('cmdprint') # -> stdout


class Switcher(object):
    """ Helping class containing all the switching routine """

    def __init__(self, main='MadGraph', *args, **opt):
            
        # define the interface
        self.change_principal_cmd(main)
        self.cmd.__init__(self, *args, **opt)       

    interface_names= {'MadGraph':('MG5_aMC',MGcmd.MadGraphCmd),
                      'MadLoop':('MG5_aMC',LoopCmd.LoopInterface),
                      'aMC@NLO':('MG5_aMC',amcatnloCmd.aMCatNLOInterface)}

    _switch_opts = interface_names.keys()
    current_interface = None
   
    # Helper functions
   
    def setup(self, *args, **opts):
        """ Function to initialize the interface when switched to it. It is not
        the same as __init__ as this latter functions would call its mother
        from madgraph_interface and this is only desirable for the first
        initialization when launching MG5 """
        return self.cmd.setup(self, *args, **opts)
   
    def debug_link_to_command(self):
        """redefine all the command to call directly the appropriate child"""
        
        correct = True
        # function which should be self.cmd dependent but which doesn't start
        # by do_xxxx, help_xxx, check_xxxx or complete_xxx 
        overwritable = []        
        # list of item overwritten by the MasterClass
        self.to_preserve = [key for key,method in Switcher.__dict__.items() if
                       hasattr(method, '__call__') ]
        self.to_preserve += ['do_shell', 'help_shell', 'complete_shell']

        ff = open(pjoin(os.getcwd(), 'additional_command'), 'w')
        
        for key in dir(self):
            # by pass all not over-writable command
            if key in self.to_preserve:
                continue
            if not (key.startswith('do_') or key.startswith('complete_') or \
               key.startswith('help_') or key.startswith('check_') or \
               key in overwritable):
                continue
            text = """\
    def %(key)s(self, *args, **opts):
        return self.cmd.%(key)s(self, *args, **opts)
        
""" % {'key': key}
            logger.warning("""Command %s not define in the Master. 
            The line to add to the master_interface.py are written in 'additional_command' file""" % key)
            ff.write(text)
            correct = False
               
            
        # Check that all function define in more than one subclass is define
        # in the Switcher or in one of the MasterClass
        define = {}
        for mother in MasterCmd.__mro__:
            if mother.__name__ in ['Cmd', 'BasicCmd', 'ExtendedCmd']:
                continue
            
            
            for data in mother.__dict__:
                #check if  define in Switcher
                if data in Switcher.__dict__ or data.startswith('__'):
                    continue
                if data in MasterCmd.__dict__:
                    #always overwritten in the  main class
                    continue 
                if data not in define:
                    define[data] = mother.__name__
                else:
                    logger.warning('%s define in %s and in %s but not in Switcher.' % (data, define[data], mother.__name__))
                    correct = False
                    
        # Do the same for the WEb MasterClass
        define = {}
        for mother in MasterCmdWeb.__mro__:
            if mother.__name__ in ['Cmd', 'BasicCmd', 'ExtendedCmd']:
                continue
            
            for data in mother.__dict__:
                #check if  define in Switcher
                if data in Switcher.__dict__ or data.startswith('__'):
                    continue
                if data in MasterCmdWeb.__dict__:
                    #always overwritten in the  main class
                    continue                
                if data not in define:
                    define[data] = mother.__name__
                else:
                    logger.warning('%s define in %s and in %s but not in Switcher.' % (data, define[data], mother.__name__))
                    correct = False                    
                    
        if not correct:
            raise Exception, 'The Cmd interface has dangerous features. Please see previous warnings and correct those.' 

    

    @staticmethod
    def extract_process_type(line):
        """Extract from a string what is the type of the computation. This 
        returns a tuple (mode, option, pert_orders) where mode can be either 'NLO' or 'tree'
        and option 'all', 'real' or 'virt'."""

        # Perform sanity modifications on the lines:
        # Add a space before and after any > , $ / | [ ]
        space_before = re.compile(r"(?P<carac>\S)(?P<tag>[\\[\\]/\,\\$\\>|])(?P<carac2>\S)")
        line2 = space_before.sub(r'\g<carac> \g<tag> \g<carac2>', line)       
        
        # Use regular expressions to extract the loop mode (if present) and its
        # option, specified in the line with format [ option = loop_orders ] or
        # [ loop_orders ] which implicitly select the 'all' option.
        loopRE = re.compile(r"^(.*)(?P<loop>\[(\s*(?P<option>\w+)\s*=)?(?P<orders>.+)?\])(.*)$")
        res=loopRE.search(line)
        if res:
            orders=res.group('orders').split() if res.group('orders') else []
            if res.group('option') and len(res.group('option').split())==1:
                if res.group('option').split()[0]=='tree':
                    return ('tree',res.group('option').split()[0],orders)
                else:
                    return ('NLO',res.group('option').split()[0],orders)
            else:
                # If not option is set the convention is that the mode is 'all'
                # unless no perturbation orders is defined.
                if len(orders)>0:
                    return ('NLO','all',orders)
                else:
                    return ('tree',None,[])               
        else:
            return ('tree',None,[])    
    
    # Wrapping functions possibly switching to new interfaces

    def do_add(self, line, *args, **opts):
        
        argss = cmd.Cmd.split_arg(line)
        if len(argss)>=1 and argss[0] in ['process','timing','profile']:
            proc_line = ' '.join(argss[1:])
            (type,nlo_mode,orders)=self.extract_process_type(proc_line)
            if type=='NLO':
                if not nlo_mode in self._valid_nlo_modes: raise self.InvalidCMD( \
                    'The NLO mode %s is not valid. Please choose one among: %s' \
                    % (nlo_mode, ' '.join(self._valid_nlo_modes)))
                elif nlo_mode in ['all', 'real', 'LOonly']:
                    self.change_principal_cmd('aMC@NLO')
                elif nlo_mode in ['virt', 'sqrvirt']:
                    self.change_principal_cmd('MadLoop')
                elif nlo_mode == 'noborn': 
                    self.change_principal_cmd('MadLoop')
                    self.cmd.validate_model(self, loop_type=nlo_mode,
                                                            coupling_type=orders)
                    self.change_principal_cmd('MadGraph')
                    return self.cmd.create_loop_induced(self, line, *args, **opts)
        try:
            return  self.cmd.do_add(self, line, *args, **opts)
        except fks_base.NoBornException:
            logger.info("------------------------------------------------------------------------", '$MG:color:BLACK')
            logger.info(" No Born diagrams found. Now switching to the loop-induced mode.        ", '$MG:color:BLACK')
            logger.info(" Please cite ref. 'arXiv:1507.00020' when using results from this mode. ", '$MG:color:BLACK')
            logger.info("------------------------------------------------------------------------", '$MG:color:BLACK')            
            self.change_principal_cmd('MadGraph')
            return self.cmd.create_loop_induced(self, line, *args, **opts)

        
    def do_check(self, line, *args, **opts):

        argss = self.split_arg(line)
        proc_line = " ".join(argss[1:])
        (type,nlo_mode,orders)=self.extract_process_type(proc_line)
        if type=='NLO':
            if not nlo_mode in self._valid_nlo_modes: raise self.InvalidCMD(\
                'The NLO mode %s is not valid. Please chose one among: %s' \
                % (nlo_mode, ' '.join(self._valid_nlo_modes)))
            elif nlo_mode == 'all':
                self.change_principal_cmd('MadLoop')
            elif nlo_mode == 'real':
                raise self.InvalidCMD('Mode [real=...] not valid for checking processes.')
                self.change_principal_cmd('aMC@NLO')
            elif nlo_mode == 'virt' or nlo_mode == 'sqrvirt':
                self.change_principal_cmd('MadLoop')
        else:
            self.change_principal_cmd('MadGraph')
        
        return self.cmd.do_check(self, line, *args, **opts)

    def do_generate(self, line, *args, **opts):

        argss = cmd.Cmd.split_arg(line)
        # Make sure to switch to the right interface.
        if len(argss)>=1:            
            proc_line = ' '.join(argss[1:])
            (type,nlo_mode,orders)=self.extract_process_type(proc_line)
            if type=='NLO':
                if not nlo_mode in self._valid_nlo_modes: raise self.InvalidCmd( \
                    'The NLO mode %s is not valid. Please chose one among: %s' \
                    % (nlo_mode, ' '.join(self._valid_nlo_modes)))
                elif nlo_mode in ['all', 'real', 'LOonly']:
                    self._fks_multi_proc = fks_base.FKSMultiProcess()
                    self.change_principal_cmd('aMC@NLO')
                elif nlo_mode == 'virt' or nlo_mode == 'virtsqr':
                    self.change_principal_cmd('MadLoop')
            else:
                self.change_principal_cmd('MadGraph')        
        return self.cmd.do_generate(self, line, *args, **opts)

    def do_import(self, *args, **opts):
        self.cmd.do_import(self, *args, **opts)
        if self._curr_model:
            if isinstance(self._curr_model, loop_base_objects.LoopModel) and \
               self._curr_model['perturbation_couplings']!=[] and \
                            self.current_interface not in ['aMC@NLO','MadLoop']:
                self.change_principal_cmd('aMC@NLO')
            if (not isinstance(self._curr_model, loop_base_objects.LoopModel) or \
               self._curr_model['perturbation_couplings']==[]) and \
                                          self.current_interface in ['MadLoop']:
                self.change_principal_cmd('MadGraph')
        import madgraph.various.misc as misc
        return
    
    def do_output(self, line, *args, **opts):
        """ treat output aloha in order to use always the one in MG5 """
        if line.strip().startswith('aloha'):
            MGcmd.MadGraphCmd.do_output(self, line, *args, **opts)
        else:
            self.cmd.do_output(self, line, *args, **opts)
    
    def check_output(self, arg, *args, **opts):
        if arg and arg[0] == 'aloha':
            MGcmd.MadGraphCmd.check_output(self, arg, *args, **opts)
        else:
            self.cmd.check_output(self, arg, *args, **opts)
            
    
        

    # Dummy functions, not triggering any switch of interfaces
            
    def export(self, *args, **opts):
        return self.cmd.export(self, *args, **opts)
    
    def check_add(self, *args, **opts):
        return self.cmd.check_add(self, *args, **opts)
        
    def check_answer_in_input_file(self, *args, **opts):
        return self.cmd.check_answer_in_input_file(self, *args, **opts)  
    
    def check_check(self, *args, **opts):
        return self.cmd.check_check(self, *args, **opts)
        
    def check_define(self, *args, **opts):
        return self.cmd.check_define(self, *args, **opts)

    def check_decay_diagram(self, *args, **opts):
        return self.cmd.check_decay_diagram(self, *args, **opts)

    def complete_decay_diagram(self, *args, **opts):
        return self.cmd.complete_decay_diagram(self, *args, **opts)

    def do_decay_diagram(self, *args, **opts):
        return self.cmd.do_decay_diagram(self, *args, **opts)

    def help_decay_diagram(self, *args, **opts):
        return self.cmd.help_decay_diagram(self, *args, **opts)

    def check_compute_widths(self, *args, **opts):
        return self.cmd.check_compute_widths(self, *args, **opts)
    
    def complete_compute_widths(self, *args, **opts):
        return self.cmd.complete_compute_widths(self, *args, **opts)

    def do_compute_widths(self, *args, **opts):
        return self.cmd.do_compute_widths(self, *args, **opts)

    def help_compute_widths(self, *args, **opts):
        return self.cmd.help_compute_widths(self, *args, **opts)

    def check_display(self, *args, **opts):
        return self.cmd.check_display(self, *args, **opts)
        
    def check_draw(self, *args, **opts):
        return self.cmd.check_draw(self, *args, **opts)
        
    def check_for_export_dir(self, *args, **opts):
        return self.cmd.check_for_export_dir(self, *args, **opts)
        
    def check_generate(self, *args, **opts):
        return self.cmd.check_generate(self, *args, **opts)

    def check_tutorial(self, *args, **opts):
        return self.cmd.check_tutorial(self, *args, **opts)
        
    def check_history(self, *args, **opts):
        return self.cmd.check_history(self, *args, **opts)
        
    def check_import(self, *args, **opts):
        return self.cmd.check_import(self, *args, **opts)
        
    def check_install(self, *args, **opts):
        return self.cmd.check_install(self, *args, **opts)
        
    def check_launch(self, *args, **opts):
        return self.cmd.check_launch(self, *args, **opts)
        
    def check_load(self, *args, **opts):
        return self.cmd.check_load(self, *args, **opts)
        
    def check_open(self, *args, **opts):
        return self.cmd.check_open(self, *args, **opts)
        
    def check_process_format(self, *args, **opts):
        return self.cmd.check_process_format(self, *args, **opts)
    
    def check_save(self, *args, **opts):
        return self.cmd.check_save(self, *args, **opts)
        
    def check_set(self, *args, **opts):
        return self.cmd.check_set(self, *args, **opts)
        
    def get_stored_line(self, *args, **opts):
        return self.cmd.get_stored_line(self, *args, **opts)
        
    def complete_add(self, *args, **opts):
        return self.cmd.complete_add(self, *args, **opts)

    def complete_switch(self, *args, **opts):
        return self.cmd.complete_switch(self, *args, **opts)

    def complete_check(self, *args, **opts):
        return self.cmd.complete_check(self, *args, **opts)
        
    def complete_define(self, *args, **opts):
        return self.cmd.complete_define(self, *args, **opts)
        
    def complete_display(self, *args, **opts):
        return self.cmd.complete_display(self, *args, **opts)
        
    def complete_draw(self, *args, **opts):
        return self.cmd.complete_draw(self, *args, **opts)
        
    def complete_generate(self, *args, **opts):
        return self.cmd.complete_generate(self, *args, **opts)
        
    def complete_help(self, *args, **opts):
        return self.cmd.complete_help(self, *args, **opts)
        
    def complete_history(self, *args, **opts):
        return self.cmd.complete_history(self, *args, **opts)
        
    def complete_import(self, *args, **opts):
        return self.cmd.complete_import(self, *args, **opts)
        
    def complete_install(self, *args, **opts):
        return self.cmd.complete_install(self, *args, **opts)
        
    def complete_launch(self, *args, **opts):
        return self.cmd.complete_launch(self, *args, **opts)
        
    def complete_load(self, *args, **opts):
        return self.cmd.complete_load(self, *args, **opts)
        
    def complete_open(self, *args, **opts):
        return self.cmd.complete_open(self, *args, **opts)
        
    def complete_output(self, *args, **opts):
        return self.cmd.complete_output(self, *args, **opts)
        
    def complete_save(self, *args, **opts):
        return self.cmd.complete_save(self, *args, **opts)
        
    def complete_set(self, *args, **opts):
        return self.cmd.complete_set(self, *args, **opts)
        
    def complete_tutorial(self, *args, **opts):
        return self.cmd.complete_tutorial(self, *args, **opts)

    def do_switch(self, *args, **opts):
        """Not in help """
        return self.cmd.do_switch(self, *args, **opts)
      
    def do_EOF(self, *args, **opts):
        return self.cmd.do_EOF(self, *args, **opts)
        
    def do_define(self, *args, **opts):
        return self.cmd.do_define(self, *args, **opts)
        
    def do_display(self, *args, **opts):
        return self.cmd.do_display(self, *args, **opts)
        
    def do_exit(self, *args, **opts):
        return self.cmd.do_exit(self, *args, **opts)
        
    def do_help(self, *args, **opts):
        return self.cmd.do_help(self, *args, **opts)
        
    def do_history(self, *args, **opts):
        return self.cmd.do_history(self, *args, **opts) 
        
    def do_install(self, *args, **opts):
        self.cmd.do_install(self, *args, **opts)
        
    def do_launch(self, line, *argss, **opts):
        args = cmd.Cmd.split_arg(line)
        # check if a path is given
        if len(args) >=1:
            if os.path.isdir(args[0]):
                path = os.path.realpath(args[0])
            elif os.path.isdir(pjoin(MG5DIR,args[0])):
                path = pjoin(MG5DIR,args[0])
            elif  MG4DIR and os.path.isdir(pjoin(MG4DIR,args[0])):
                path = pjoin(MG4DIR,args[0])
            else:
                path=None
        # if there is a path, find what output has been done
            if path:
                type = self.cmd.find_output_type(self, path) 
                if type in ['standalone', 'standalone_cpp', 'pythia8', 'madevent']:
                    self.change_principal_cmd('MadGraph')
                elif type == 'aMC@NLO':
                    self.change_principal_cmd('aMC@NLO')
                elif type == 'MadLoop':
                    self.change_principal_cmd('MadLoop')

        return self.cmd.do_launch(self, line, *argss, **opts)
        
    def do_load(self, *args, **opts):
        return self.cmd.do_load(self, *args, **opts)
        
    def do_open(self, *args, **opts):
        return self.cmd.do_open(self, *args, **opts)
        
    def do_quit(self, *args, **opts):
        return self.cmd.do_quit(self, *args, **opts)
        
    def do_save(self, *args, **opts):
        return self.cmd.do_save(self, *args, **opts)
        
    def do_set(self, *args, **opts):
        return self.cmd.do_set(self, *args, **opts)
    
    def do_tutorial(self, *args, **opts):
        return self.cmd.do_tutorial(self, *args, **opts)

    def help_EOF(self, *args, **opts):
        return self.cmd.help_EOF(self, *args, **opts)
        
    def help_add(self, *args, **opts):
        return self.cmd.help_add(self, *args, **opts)
        
    def help_check(self, *args, **opts):
        return self.cmd.help_check(self, *args, **opts)
        
    def help_define(self, *args, **opts):
        return self.cmd.help_define(self, *args, **opts)
        
    def help_display(self, *args, **opts):
        return self.cmd.help_display(self, *args, **opts)
        
    def help_generate(self, *args, **opts):
        return self.cmd.help_generate(self, *args, **opts)
        
    def help_help(self, *args, **opts):
        return self.cmd.help_help(self, *args, **opts)
        
    def help_history(self, *args, **opts):
        return self.cmd.help_history(self, *args, **opts)
        
    def help_import(self, *args, **opts):
        return self.cmd.help_import(self, *args, **opts)
        
    def help_install(self, *args, **opts):
        return self.cmd.help_install(self, *args, **opts)
        
    def help_launch(self, *args, **opts):
        return self.cmd.help_launch(self, *args, **opts)
        
    def help_load(self, *args, **opts):
        return self.cmd.help_load(self, *args, **opts)
        
    def help_open(self, *args, **opts):
        return self.cmd.help_open(self, *args, **opts)
        
    def help_output(self, *args, **opts):
        return self.cmd.help_output(self, *args, **opts)
        
    def help_quit(self, *args, **opts):
        return self.cmd.help_quit(self, *args, **opts)
        
    def help_save(self, *args, **opts):
        return self.cmd.help_save(self, *args, **opts)
        
    def help_set(self, *args, **opts):
        return self.cmd.help_set(self, *args, **opts)
        
    def help_tutorial(self, *args, **opts):
        return self.cmd.help_tutorial(self, *args, **opts)
        
    def test_interface(self, *args, **opts):
        return self.cmd.test_interface(self, *args, **opts)

    def set_configuration(self, *args, **opts):
        return self.cmd.set_configuration(self, *args, **opts)

    def check_customize_model(self, *args, **opts):
        return self.cmd.check_customize_model(self, *args, **opts)

    def complete_customize_model(self, *args, **opts):
        return self.cmd.complete_customize_model(self, *args, **opts)

    def do_customize_model(self, *args, **opts):
        return self.cmd.do_customize_model(self, *args, **opts)

    def help_customize_model(self, *args, **opts):
        return self.cmd.help_customize_model(self, *args, **opts)

class MasterCmd(Switcher, LoopCmd.LoopInterface, amcatnloCmd.aMCatNLOInterface, cmd.CmdShell):

    def __init__(self, main='MadGraph', *args, **opt):
            
        # define the interface
        if main in self.interface_names.keys():
            self.prompt= self.interface_names[main][0]+'>'
            self.cmd= self.interface_names[main][1]
            self.current_interface=main
        else:
            raise MadGraph5Error, 'Type of interface not valid: %s' % main  
        self.cmd.__init__(self, *args, **opt)     
        self.current_interface = main  
    
    def complete_switch(self, text, line, begidx, endidx):
        """Complete the switch command"""
        return self.list_completion(text,self._switch_opts)
        
    def do_switch(self, line):
        """Not in help: Allow to switch to any given interface from command line """

        args = cmd.Cmd.split_arg(line)
        if len(args)==1 and args[0] in self.interface_names.keys():
            self.change_principal_cmd(args[0])
        else:
            raise self.InvalidCmd("Invalid switch command or non existing interface %s."\
                            %args[0]+" Valid interfaces are %s"\
                            %','.join(interface_quick_name.keys()))
        
    def change_principal_cmd(self, name):
        old_cmd=self.current_interface
        if name in self.interface_names.keys():
            self.prompt= self.interface_names[name][0]+'>'
            self.cmd= self.interface_names[name][1]
            self.current_interface=name
        else:
            raise MadGraph5Error, 'Type of interface not valid: %s' % name  
        
        if self.interface_names[old_cmd][0]!=self.interface_names[name][0]:
            logger.info("Switching from interface %s to %s"\
                        %(self.interface_names[old_cmd][0],\
                          self.interface_names[name][0]))
            # Setup the interface
            self.cmd.setup(self)
        
        if __debug__:
            self.debug_link_to_command()      
        

class MasterCmdWeb(MGcmd.MadGraphCmdWeb, Switcher, LoopCmd.LoopInterfaceWeb):
   
    def __init__(self, *arg, **opt):
    
        if os.environ.has_key('_CONDOR_SCRATCH_DIR'):
            self.writing_dir = pjoin(os.environ['_CONDOR_SCRATCH_DIR'], \
                                                                 os.path.pardir)
        else:
            self.writing_dir = pjoin(os.environ['MADGRAPH_DATA'],
                               os.environ['REMOTE_USER'])
            
        
        #standard initialization
        Switcher.__init__(self, mgme_dir = '', *arg, **opt)
        
        self.options['timeout'] = 1 # time authorize to answer question [0 is no time limit]
        
    def change_principal_cmd(self, name):
        if name == 'MadGraph':
            self.cmd = MGcmd.MadGraphCmdWeb
        elif name == 'Loop':
            self.cmd = LoopCmd.LoopInterfaceWeb
        else:
            raise MadGraph5Error, 'Type of interface not valid'  
        
        if __debug__:
            self.debug_link_to_command() 
    
    def do_shell(self, *args):
        raise Exception
    
    def finalize(self, nojpeg, flaglist=[]):
        """Finalize web generation"""

        if flaglist != []:
            raise Exception
        self.cmd.finalize(self, nojpeg, online = True)
    
    def finalize(self, nojpeg, **opts):
        """Finalize web generation""" 
        
        opts['online'] = True
        self.cmd.finalize(self, nojpeg, opts)

    # Generate a new amplitude
    def do_generate(self, line):
        """Generate an amplitude for a given process"""

        try:
            Switcher.do_generate(self, line)
        except:
            # put the stop logo on the web
            files.cp(self._export_dir+'/HTML/stop.jpg',self._export_dir+'/HTML/card.jpg')
            raise
    
    # Add a process to the existing multiprocess definition
    def do_add(self, line):
        """Generate an amplitude for a given process and add to
        existing amplitudes
        syntax:
        """
        try:
           Switcher.do_add(self, line)
        except:
            # put the stop logo on the web
            files.cp(self._export_dir+'/HTML/stop.jpg',self._export_dir+'/HTML/card.jpg')
            raise
        
    # Use the cluster file for the configuration
    def set_configuration(self, config_path=None, final=False):
        
        """Force to use the web configuration file only"""
        config_path = pjoin(os.environ['MADGRAPH_BASE'], 'mg5_configuration.txt')
        return Switcher.set_configuration(self, config_path=config_path, final=final)
    
    def do_save(self, line, check=True, **opt):
        """Save information to file"""
        
        if check:
            self.check_save([])
            raise #useless but full security
        
        args = self.split_arg(line)
        if args[0] != 'options':
            Switcher.do_save(self, line,check, opt)
        else:
            # put default options since 
            # in the web the local file is not used
            # in download the default file is more usefull
            files.cp(pjoin(MG5DIR,'input','mg5_configuration.txt'), args[1])
            
    def do_install(self, line):
        """block all install"""
        return

