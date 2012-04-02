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
"""A user friendly command line interface to access all MadGraph features.
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
import madgraph.interface.extended_cmd as cmd
import madgraph.interface.madgraph_interface as MGcmd
import madgraph.iolibs.files as files

from madgraph import MG4DIR, MG5DIR, MadGraph5Error

logger = logging.getLogger('cmdprint') # -> stdout


class Switcher(object):
    """ Helping class containing all the switching routine """

    def __init__(self, main='MadGraph', *args, **opt):
            
        # define the interface
        self.change_principal_cmd(main)
        self.cmd.__init__(self, *args, **opt)       
        


        
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

        ff = open(pjoin(MG5DIR, 'additional_command'), 'w')
        
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
        
        
    
    
    
    def check_add(self, *args, **opts):
        return self.cmd.check_add(self, *args, **opts)
        
    def check_answer_in_input_file(self, *args, **opts):
        return self.cmd.check_answer_in_input_file(self, *args, **opts)
        
    def check_check(self, *args, **opts):
        return self.cmd.check_check(self, *args, **opts)
        
    def check_define(self, *args, **opts):
        return self.cmd.check_define(self, *args, **opts)
        
    def check_display(self, *args, **opts):
        return self.cmd.check_display(self, *args, **opts)
        
    def check_draw(self, *args, **opts):
        return self.cmd.check_draw(self, *args, **opts)
        
    def check_for_export_dir(self, *args, **opts):
        return self.cmd.check_for_export_dir(self, *args, **opts)
        
    def check_generate(self, *args, **opts):
        return self.cmd.check_generate(self, *args, **opts)
        
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
        
    def check_output(self, *args, **opts):
        return self.cmd.check_output(self, *args, **opts)
        
    def check_process_format(self, *args, **opts):
        return self.cmd.check_process_format(self, *args, **opts)
        
    def check_save(self, *args, **opts):
        return self.cmd.check_save(self, *args, **opts)
        
    def check_set(self, *args, **opts):
        return self.cmd.check_set(self, *args, **opts)
        
    def check_stored_line(self, *args, **opts):
        return self.cmd.check_stored_line(self, *args, **opts)
        
    def complete_add(self, *args, **opts):
        return self.cmd.complete_add(self, *args, **opts)
        
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
        
    def do_EOF(self, *args, **opts):
        return self.cmd.do_EOF(self, *args, **opts)
        
    def do_add(self, *args, **opts):
        return self.cmd.do_add(self, *args, **opts)
        
    def do_check(self, *args, **opts):
        return self.cmd.do_check(self, *args, **opts)
        
    def do_define(self, *args, **opts):
        return self.cmd.do_define(self, *args, **opts)
        
    def do_display(self, *args, **opts):
        return self.cmd.do_display(self, *args, **opts)
        
    def do_exit(self, *args, **opts):
        return self.cmd.do_exit(self, *args, **opts)
        
    def do_generate(self, *args, **opts):
        return self.cmd.do_generate(self, *args, **opts)
        
    def do_help(self, *args, **opts):
        return self.cmd.do_help(self, *args, **opts)
        
    def do_history(self, *args, **opts):
        return self.cmd.do_history(self, *args, **opts)
        
    def do_import(self, *args, **opts):
        return self.cmd.do_import(self, *args, **opts)
        
    def do_install(self, *args, **opts):
        return self.cmd.do_install(self, *args, **opts)
        
    def do_launch(self, *args, **opts):
        return self.cmd.do_launch(self, *args, **opts)
        
    def do_load(self, *args, **opts):
        return self.cmd.do_load(self, *args, **opts)
        
    def do_open(self, *args, **opts):
        return self.cmd.do_open(self, *args, **opts)
        
    def do_output(self, *args, **opts):
        return self.cmd.do_output(self, *args, **opts)
        
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


class MasterCmd(Switcher, MGcmd.MadGraphCmd, cmd.CmdShell):


    def change_principal_cmd(self, name):
        if name == 'MadGraph':
            self.cmd = MGcmd.MadGraphCmd
        else:
            raise MadGraph5Error, 'Type of interface not valid'  
        
        if __debug__:
            self.debug_link_to_command()      
        
class MasterCmdWeb(Switcher, MGcmd.MadGraphCmdWeb):
   
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
#        elif name == 'Loop':
#            self.cmd = NLOcmd.LoopInterfaceWeb
        else:
            raise MadGraph5Error, 'Type of interface not valid'  
        
        if __debug__:
            self.debug_link_to_command() 
    
    def finalize(self, nojpeg):
        """Finalize web generation""" 
        
        self.cmd.finalize(self, nojpeg, online = True)

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
    def set_configuration(self, config_path=None):
        
        """Force to use the web configuration file only"""
        config_path = pjoin(os.environ['MADGRAPH_BASE'], 'mg5_configuration.txt')
        return Switcher.set_configuration(self, config_path=config_path)
    

    def do_save(self, line, check=True):
        """Save information to file"""
        
        if check:
            self.check_save([])
            raise #useless but full security
        
        args = self.split_arg(line)
        if args[0] != 'options':
            Switcher.do_save(self, line,check)
        else:
            # put default options since 
            # in the web the local file is not used
            # in download the default file is more usefull
            files.cp(pjoin(MG5DIR,'input','mg5_configuration.txt'), args[1])
            


