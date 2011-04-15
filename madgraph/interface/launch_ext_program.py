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
import subprocess
import time

import madgraph.iolibs.files as files
import madgraph.iolibs.misc as misc
import madgraph.interface.extended_cmd as cmd
from madgraph import MG4DIR, MG5DIR, MadGraph5Error
from madgraph.iolibs.files import cp

logger = logging.getLogger('cmdprint.ext_program')

class ExtLauncher(object):
    """ Generic Class for executing external program """
    
    program_dir = ''
    executable = ''  # path from program_dir
    
    cards = [] # files can be modified (path from self.card_dir)
    force = False
    
    def __init__(self, running_dir, card_dir='', timeout=None, **options):
        """ initialize an object """
        
        self.running_dir = running_dir
        self.card_dir = os.path.join(self.running_dir, card_dir)
        self.timeout = timeout
        self.found_editor()
        
        #include/overwrite options
        for key,value in options.items():
            setattr(self, key, value)
            
            
    def found_editor(self):
        """ found a (shell) program for editing file """
        
        # let first try to use the prefer editor (if EDITOR is define)
        # if not define use the first define in a pre-define list
            
        if os.environ.has_key('EDITOR') and misc.which(os.environ['EDITOR']):
            self.editor =  os.environ['EDITOR']
            return
        
        logger.info('INFO: You can choose your prefered editor by defining the shell variable EDITOR')
        
        possibility = ['vi', 'emacs', 'vim', 'gedit', 'nano']
        for editor in possibility:
            if misc.which(editor):
                self.editor = editor
                return
        
        logger.warning(
         'No valid editor detected. Please configure the shell variable EDITOR')
            
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
        subprocess.call([self.editor, path], cwd=os.getcwd())
        
    def ask(self, question, default, choices=[], path_info=[]):
        """ ask a question """
        
        assert type(path_info) == list
        
        if not self.force:
            # add choice info to the question
            if choices + path_info:
                question += ' ['
                
                for data in choices[:9] + path_info:
                    if default == data:
                        question += "\033[%dm%s\033[0m" % (4, data)
                    else:
                        question += "%s" % data
                    question += ', '
                if len(choices) > 9:
                    question += '... , ' 
                question = question[:-2]+']'
                
            if path_info:
                fct = lambda q: cmd.raw_path_input(q, allow_arg=choices, default=default)
            else:
                fct = lambda q: cmd.smart_input(q, allow_arg=choices, default=default)
            try:
                out =  misc.timed_input(question, default, timeout=self.timeout,
                                        noerror=False, fct=fct)
            except misc.TimeOutError:
                # avoid to always wait a given time for the next answer
                self.force = True
            else:
                self.timeout=None # answer at least one question so wait...
                return out
        else:
            return default
        
    def treat_input_file(self, filename, default=None, msg=''):
        """ask to edit a file"""
        
        if msg == '' and filename == 'param_card.dat':
            msg = \
            """WARNING: If you edit this file don\'t forget to modify 
            consistently the different parameters, especially 
            the width of all particles.""" 
        
        fct = lambda q: cmd.raw_path_input(q, allow_arg=['y','n'])     
                                    
        if not self.force:
            if msg:  print msg
            question = 'Do you want to edit file: %(card)s?' % {'card':filename}
            choices = ['y', 'n']
            path_info = ['path of the new %(card)s' % {'card':os.path.basename(filename)}]
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
    
    cards = ['param_card.dat']

    def __init__(self, running_dir, timeout, **options):
        """ initialize the StandAlone Version"""
        
        ExtLauncher.__init__(self, running_dir, './Cards', timeout, **options)
    
    def launch_program(self):
        """launch the main program"""
        sub_path = os.path.join(self.running_dir, 'SubProcesses')
        for path in os.listdir(sub_path):
            if path.startswith('P') and \
                                   os.path.isdir(os.path.join(sub_path, path)):
                cur_path =  os.path.join(sub_path, path)
                # make
                subprocess.call(['make'], cwd=cur_path)
                # check
                subprocess.call(['./check'], cwd=cur_path)
                
        
class MELauncher(ExtLauncher):
    """A class to launch MadEvent run"""
    
    cards = ['param_card.dat', 'run_card.dat']

    def __init__(self, running_dir, timeout, **option):
        """ initialize the StandAlone Version"""
        
        
        ExtLauncher.__init__(self, running_dir, './Cards', timeout, **option)
        self.executable = os.path.join('.', 'bin','generate_events')

        assert hasattr(self, 'cluster')
        assert hasattr(self, 'name')
         
        # Check for pythia-pgs directory
        if os.path.isdir(os.path.join(MG5DIR,'pythia-pgs')):
            self.pythia = os.path.join(MG5DIR,'pythia-pgs')
        elif MG4DIR and os.path.isdir(os.path.join(MG4DIR,'pythia-pgs')):
            self.pythia = os.path.join(MG4DIR,'pythia-pgs')
        else:
            self.pythia = None
            
        # Check for DELPHES directory
        if os.path.isdir(os.path.join(MG5DIR,'Delphes')):
            self.delphes = os.path.join(MG5DIR,'Delphes')
        elif MG4DIR and os.path.isdir(os.path.join(MG4DIR,'Delphes')):
            self.delphes = os.path.join(MG4DIR,'Delphes')
        else:
            self.delphes = None
        
        
        # Assign a valid run name if not put in options
        if self.name == '':
            for i in range(1000):
                path = os.path.join(self.running_dir, 'Events','run_%02i_banner.txt' % i)
                if not os.path.exists(path):
                    self.name = 'run_%02i' % i
                    break
        
        if self.name == '':
            raise MadGraph5Error, 'too much run in this directory'
    
    
    def copy_default_card(self, name):

        dico = {'dir': self.card_dir, 'name': name }

        if not os.path.exists('%(dir)s/%(name)s_card.dat' % dico):
            cp('%(dir)s/%(name)s_card_default.dat' % dico,
                '%(dir)s/%(name)s_card.dat' % dico)
    
          
    def prepare_run(self):
        """ ask for pythia-pgs/delphes run """
        
        # Check If we Need to run pythia 
        if not self.pythia or self.force:
            return
        
        answer = self.ask('Do you want to run pythia?','auto', ['y','n','auto'])
        if answer == 'y':
            self.copy_default_card('pythia')
            self.cards.append('pythia_card.dat')
        elif answer == 'n':
            path = os.path.join(self.card_dir, 'pythia_card.dat')
            try: os.remove(path)
            except OSError: pass
            return # no Need to ask for PGS
        
        answer = self.ask('Do you want to run PGS?','auto', ['y','n','auto'])
        if answer == 'y':
            self.copy_default_card('pgs')
            self.cards.append('pgs_card.dat')
            return # No Need to ask for Delphes
        elif answer == 'n':
            path = os.path.join(self.card_dir, 'pgs_card.dat')
            try: os.remove(path)
            except OSError: pass
    
        if not self.delphes:
            return
        
        answer = self.ask('Do you want to run Delphes?','n', ['y','n','auto'])
        if answer == 'y':
            self.copy_default_card('delphes')
            self.cards.append('delphes_card.dat')
        elif answer == 'n':
            path = os.path.join(self.card_dir, 'delphes_card.dat')
            try: os.remove(path)
            except OSError: pass        
    
    def launch_program(self):
        """launch the main program"""
        
        mode = str(self.cluster)
        if mode == "0":
            subprocess.call([self.executable, mode, self.name], 
                                                           cwd=self.running_dir)
        elif mode == "1":
            subprocess.call([self.executable, mode, self.name, self.name], 
                                                           cwd=self.running_dir)
        elif mode == "2":
            import multiprocessing
            max_node = multiprocessing.cpu_count()
            if max_node == 1:
                logger.warning('Only one core is detected on your computer! Pass in single machine')
                self.cluster = 0
                self.launch_program()
                return
            nb_node = self.ask('How many core do you want to use?', max_node, range(max_node+1))
            subprocess.call([self.executable, mode, nb_node, self.name], 
                                                           cwd=self.running_dir)
        
        # Display the cross-section to the screen
        path = os.path.join(self.running_dir, 'SubProcesses', '%s_results.dat' 
                                                                    % self.name) 
        fsock = open(path)
        line = fsock.readline()
        cross, error = line.split()[0:2]
        
        logger.info('The total cross-section is %s +- %s pb' % (cross, error))
        logger.info('more information in %s' 
                                 % os.path.join(self.running_dir, 'index.html'))
                

class Pythia8Launcher(ExtLauncher):
    """A class to launch Pythia8 run"""
    
    cards = []

    def __init__(self, running_dir, timeout, **option):
        """ initialize launching Pythia 8"""

        running_dir = os.path.join(running_dir, 'examples')
        ExtLauncher.__init__(self, running_dir, '.', timeout, **option)

    
    def prepare_run(self):
        """ ask for pythia-pgs/delphes run """

        self.cards = []
        
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
        status = subprocess.call(['make'], stdout = open(os.devnull, 'w'),
                        stderr = open(os.devnull, 'w'),
                        cwd=os.path.join(self.running_dir, os.path.pardir))
        if status != 0:
            raise MadGraph5Error, "make failed for pythia8 directory"
        if self.model_dir:
            print "Running make in %s" % self.model_dir
            status = subprocess.call(['make'], stdout = open(os.devnull, 'w'),
                            stderr = open(os.devnull, 'w'),
                            cwd=self.model_dir)
            if status != 0:
                raise MadGraph5Error, "make failed for %s directory" % self.model_dir
        # Finally run make for executable
        makefile = self.executable.replace("main_","Makefile_")
        print "Running make with %s" % makefile
        status = subprocess.call(['make', '-f', makefile],
                                 stdout = open(os.devnull, 'w'),
                                 cwd=self.running_dir)
        if status != 0:
            raise MadGraph5Error, "make failed for %s" % self.executable
        
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
        
        
        
        
                    
            
