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

import glob
import logging
import os
import pydoc
import re
import sys
import subprocess
import thread
import time

import madgraph.iolibs.files as files
import madgraph.interface.extended_cmd as cmd
import madgraph.interface.madevent_interface as me_cmd
import madgraph.various.misc as misc

from madgraph import MG4DIR, MG5DIR, MadGraph5Error
from madgraph.iolibs.files import cp



logger = logging.getLogger('cmdprint.ext_program')

class ExtLauncher(object):
    """ Generic Class for executing external program """
    
    program_dir = ''
    executable = ''  # path from program_dir
    
    force = False
    
    def __init__(self, cmd, running_dir, card_dir='', **options):
        """ initialize an object """
        
        self.running_dir = running_dir
        self.card_dir = os.path.join(self.running_dir, card_dir)
        self.cmd_int = cmd
        #include/overwrite options
        for key,value in options.items():
            setattr(self, key, value)
            
        self.cards = [] # files can be modified (path from self.card_dir)
            
    def run(self):
        """ execute the main code """

        self.prepare_run()        
        for card in self.cards:
            self.treat_input_file(card, default = 'n')

        self.launch_program()

        
    def prepare_run(self):
        """ aditional way to prepare the run"""
        pass
    
    def launch_program(self):
        """launch the main program"""
        subprocess.call([self.executable], cwd=self.running_dir)
    
    def edit_file(self, path):
        """edit a file"""

        path = os.path.realpath(path)
        open_file(path)
    

    # Treat Nicely the timeout
    def timeout_fct(self,timeout):
        if timeout:
            # avoid to always wait a given time for the next answer
            self.force = True
   
    def ask(self, question, default, choices=[], path_msg=None):
        """nice handling of question"""
     
        if not self.force:
            return self.cmd_int.ask(question, default, choices=choices, 
                                path_msg=path_msg, fct_timeout=self.timeout_fct)
        else:
            return str(default)
         
        
    def treat_input_file(self, filename, default=None, msg=''):
        """ask to edit a file"""
        
        if msg == '' and filename == 'param_card.dat':
            msg = \
            """WARNING: If you edit this file don\'t forget to modify 
            consistently the different parameters, especially 
            the width of all particles.""" 
                                         
        if not self.force:
            if msg:  print msg
            question = 'Do you want to edit file: %(card)s?' % {'card':filename}
            choices = ['y', 'n']
            path_info = 'path of the new %(card)s' % {'card':os.path.basename(filename)}
            ans = self.ask(question, default, choices, path_info)
        else:
            ans = default
        
        if ans == 'y':
            path = os.path.join(self.card_dir, filename)
            self.edit_file(path)
        elif ans == 'n':
            return
        else:
            path = os.path.join(self.card_dir, filename)
            files.cp(ans, path)
            
   
                
                    
class SALauncher(ExtLauncher):
    """ A class to launch a simple Standalone test """
    
    def __init__(self, cmd_int, running_dir, **options):
        """ initialize the StandAlone Version"""
        
        ExtLauncher.__init__(self, cmd_int, running_dir, './Cards', **options)
        self.cards = ['param_card.dat']

    
    def launch_program(self):
        """launch the main program"""
        sub_path = os.path.join(self.running_dir, 'SubProcesses')
        for path in os.listdir(sub_path):
            if path.startswith('P') and \
                                   os.path.isdir(os.path.join(sub_path, path)):
                cur_path =  os.path.join(sub_path, path)
                # make
                misc.compile(cwd=cur_path, mode='unknown')
                # check
                subprocess.call(['./check'], cwd=cur_path)
                
        
class MELauncher(ExtLauncher):
    """A class to launch MadEvent run"""
    
    def __init__(self, running_dir, cmd_int , unit='pb', **option):
        """ initialize the StandAlone Version"""

        ExtLauncher.__init__(self, cmd_int, running_dir, './Cards', **option)
        #self.executable = os.path.join('.', 'bin','generate_events')
        self.pythia = cmd_int.options['pythia-pgs_path']
        self.delphes = cmd_int.options['delphes_path'],
        self.options = cmd_int.options

        assert hasattr(self, 'cluster')
        assert hasattr(self, 'multicore')
        assert hasattr(self, 'name')
        assert hasattr(self, 'shell')

        self.unit = unit
        
        if self.cluster:
            self.cluster = 1
        if self.multicore:
            self.cluster = 2
        
        self.cards = []

        # Assign a valid run name if not put in options
        if self.name == '':
            self.name = me_cmd.MadEventCmd.find_available_run_name(self.running_dir)
    
    def launch_program(self):
        """launch the main program"""
        
        # Check for number of cores if multicore mode
        mode = str(self.cluster)
        nb_node = 1
        if mode == "2":
            import multiprocessing
            max_node = multiprocessing.cpu_count()
            if max_node == 1:
                logger.warning('Only one core is detected on your computer! Pass in single machine')
                self.cluster = 0
                self.launch_program()
                return
            elif max_node == 2:
                nb_node = 2
            elif not self.force:
                nb_node = self.ask('How many core do you want to use?', max_node, range(2,max_node+1))
            else:
                nb_node=max_node
                
        import madgraph.interface.madevent_interface as ME
        
        stdout_level = self.cmd_int.options['stdout_level']
        if self.shell:
            usecmd = ME.MadEventCmdShell(me_dir=self.running_dir, options=self.options)
        else:
            usecmd = ME.MadEventCmd(me_dir=self.running_dir, options=self.options)
            usecmd.pass_in_web_mode()
        #Check if some configuration were overwritten by a command. If so use it    
        set_cmd = [l for l in self.cmd_int.history if l.strip().startswith('set')]
        for line in set_cmd:
            try:
                usecmd.exec_cmd(line)
            except:
                pass
        usecmd.exec_cmd('set stdout_level %s'  % stdout_level)
        #ensure that the logger level 
        launch = self.cmd_int.define_child_cmd_interface(
                     usecmd, interface=False)
        #launch.me_dir = self.running_dir
        if self.unit == 'pb':
            command = 'generate_events %s' % self.name
        else:
            warning_text = '''\
This command will create a new param_card with the computed width. 
This param_card makes sense only if you include all processes for
the computation of the width.'''
            logger.warning(warning_text)

            command = 'calculate_decay_widths %s' % self.name
        if mode == "1":
            command += " --cluster"
        elif mode == "2":
            command += " --nb_core=%s" % nb_node
        
        if self.force:
            command+= " -f"
        
        if self.laststep:
            command += ' --laststep=%s' % self.laststep
        
        try:
            os.remove('ME5_debug')
        except:
           pass
        launch.run_cmd(command)
        launch.run_cmd('quit')
        
        if os.path.exists('ME5_debug'):
            return True
        
        # Display the cross-section to the screen
        path = os.path.join(self.running_dir, 'SubProcesses', 'results.dat') 
        if not os.path.exists(path):
            logger.error('Generation failed (no results.dat file found)')
            return
        fsock = open(path)
        line = fsock.readline()
        cross, error = line.split()[0:2]
        
        logger.info('more information in %s' 
                                 % os.path.join(self.running_dir, 'index.html'))
                

class Pythia8Launcher(ExtLauncher):
    """A class to launch Pythia8 run"""
    
    def __init__(self, running_dir, cmd_int, **option):
        """ initialize launching Pythia 8"""

        running_dir = os.path.join(running_dir, 'examples')
        ExtLauncher.__init__(self, cmd_int, running_dir, '.', **option)
        self.cards = []
    
    def prepare_run(self):
        """ ask for pythia-pgs/delphes run """

        # Find all main_model_process.cc files
        date_file_list = []
        for file in glob.glob(os.path.join(self.running_dir,'main_*_*.cc')):
            # retrieves the stats for the current file as a tuple
            # (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
            # the tuple element mtime at index 8 is the last-modified-date
            stats = os.stat(file)
            # create tuple (year yyyy, month(1-12), day(1-31), hour(0-23), minute(0-59), second(0-59),
            # weekday(0-6, 0 is monday), Julian day(1-366), daylight flag(-1,0 or 1)) from seconds since epoch
            # note:  this tuple can be sorted properly by date and time
            lastmod_date = time.localtime(stats[8])
            date_file_list.append((lastmod_date, os.path.split(file)[-1]))

        if not date_file_list:
            raise MadGraph5Error, 'No Pythia output found'
        # Sort files according to date with newest first
        date_file_list.sort()
        date_file_list.reverse()
        files = [d[1] for d in date_file_list]
        
        answer = ''
        answer = self.ask('Select a main file to run:', files[0], files)

        self.cards.append(answer)
    
        self.executable = self.cards[-1].replace(".cc","")

        # Assign a valid run name if not put in options
        if self.name == '':
            for i in range(1000):
                path = os.path.join(self.running_dir, '',
                                    '%s_%02i.log' % (self.executable, i))
                if not os.path.exists(path):
                    self.name = '%s_%02i.log' % (self.executable, i)
                    break
        
        if self.name == '':
            raise MadGraph5Error, 'too many runs in this directory'

        # Find all exported models
        models = glob.glob(os.path.join(self.running_dir,os.path.pardir,
                                        "Processes_*"))
        models = [os.path.split(m)[-1].replace("Processes_","") for m in models]
        # Extract model name from executable
        models.sort(key=len)
        models.reverse()
        model_dir = ""
        for model in models:
            if self.executable.replace("main_", "").startswith(model):
                model_dir = "Processes_%s" % model
                break
        if model_dir:
            self.model = model
            self.model_dir = os.path.realpath(os.path.join(self.running_dir,
                                                           os.path.pardir,
                                                           model_dir))
            self.cards.append(os.path.join(self.model_dir,
                                           "param_card_%s.dat" % model))
        
    def launch_program(self):
        """launch the main program"""

        # Make pythia8
        print "Running make for pythia8 directory"
        misc.compile(cwd=os.path.join(self.running_dir, os.path.pardir), mode='cpp')
        if self.model_dir:
            print "Running make in %s" % self.model_dir
            misc.compile(cwd=self.model_dir, mode='cpp')
        # Finally run make for executable
        makefile = self.executable.replace("main_","Makefile_")
        print "Running make with %s" % makefile
        misc.compile(arg=['-f', makefile], cwd=self.running_dir, mode='cpp')
        
        print "Running " + self.executable
        
        output = open(os.path.join(self.running_dir, self.name), 'w')
        if not self.executable.startswith('./'):
            self.executable = os.path.join(".", self.executable)
        subprocess.call([self.executable], stdout = output, stderr = output,
                        cwd=self.running_dir)
        
        # Display the cross-section to the screen
        path = os.path.join(self.running_dir, self.name) 
        pydoc.pager(open(path).read())

        print "Output of the run is found at " + \
              os.path.realpath(os.path.join(self.running_dir, self.name))

# old compatibility shortcut
open_file = misc.open_file


