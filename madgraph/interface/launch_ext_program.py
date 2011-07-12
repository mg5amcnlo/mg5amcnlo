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
    
    force = False
    
    def __init__(self, running_dir, card_dir='', timeout=None,
                 **options):
        """ initialize an object """
        
        self.running_dir = running_dir
        self.card_dir = os.path.join(self.running_dir, card_dir)
        self.timeout = timeout
        
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
            return str(default)
        
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
    
    def __init__(self, running_dir, timeout, **options):
        """ initialize the StandAlone Version"""
        
        ExtLauncher.__init__(self, running_dir, './Cards', timeout, **options)
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
    
    def __init__(self, running_dir, timeout, unit='pb', **option):
        """ initialize the StandAlone Version"""
        
        ExtLauncher.__init__(self, running_dir, './Cards', timeout, **option)
        self.executable = os.path.join('.', 'bin','generate_events')

        assert hasattr(self, 'cluster')
        assert hasattr(self, 'name')
        self.unit = unit
        
        self.cards = ['param_card.dat', 'run_card.dat']
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
            for i in range(1,1000):
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
        
        # Open the corresponding crossx.html page
        os.system('touch %s' % os.path.join(self.running_dir,'RunWeb'))
        subprocess.call([os.path.join('bin','gen_crossxhtml-pl')], 
                         cwd=self.running_dir)
        open_file(os.path.join(self.running_dir, 'HTML', 'crossx.html'))

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
        
        if self.unit != 'GeV':
            logger.info('The total cross-section is %s +- %s %s' % (cross, error, self.unit))
        else:
            logger.info('The width is %s +- %s GeV' % (cross, error))
        logger.info('more information in %s' 
                                 % os.path.join(self.running_dir, 'index.html'))
                

class Pythia8Launcher(ExtLauncher):
    """A class to launch Pythia8 run"""
    
    def __init__(self, running_dir, timeout, **option):
        """ initialize launching Pythia 8"""

        running_dir = os.path.join(running_dir, 'examples')
        ExtLauncher.__init__(self, running_dir, '.', timeout, **option)
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

#
# Global function to open supported file types
#
class open_file(object):
    """ a convinient class to open a file """
    
    web_browser = None
    eps_viewer = None
    text_editor = None 
    configured = False
    
    def __init__(self, filename):
        """open a file"""
        
        # Check that the class is correctly configure
        if not self.configured:
            self.configure()
        
        try:
            extension = filename.rsplit('.',1)[1]
        except IndexError:
            extension = ''   
    
    
        # dispatch method
        if extension in ['html','htm','php']:
            self.open_program(self.web_browser, filename)
        elif extension in ['ps','eps']:
            self.open_program(self.eps_viewer, filename)
        else:
            self.open_program(self.text_editor,filename, mac_check=False)
            # mac_check to False avoid to use open cmd in mac
    
    @classmethod
    def configure(cls, configuration=None):
        """ configure the way to open the file """
        
        cls.configured = True
        
        # start like this is a configuration for mac
        cls.configure_mac(configuration)
        if sys.platform == 'darwin':
            return # done for MAC
        
        # on Mac some default (eps/web) might be kept on None. This is not
        #suitable for LINUX which doesn't have open command.
        
        # first for eps_viewer
        if not cls.eps_viewer:
           cls.eps_viewer = cls.find_valid(['gv', 'ggv', 'evince'], 'eps viewer') 
            
        # Second for web browser
        if not cls.web_browser:
            cls.web_browser = cls.find_valid(
                                    ['firefox', 'chrome', 'safari','opera'], 
                                    'web browser')

    @classmethod
    def configure_mac(cls, configuration=None):
        """ configure the way to open a file for mac """
    
        if configuration is None:
            configuration = {'text_editor': None,
                             'eps_viewer':None,
                             'web_browser':None}
        
        for key in configuration:
            if key == 'text_editor':
                # Treat text editor ONLY text base editor !!
                if configuration[key] and not misc.which(configuration[key]):
                    logger.warning('Specified text editor %s not valid.' % \
                                                             configuration[key])
                elif configuration[key]:
                    # All is good
                    cls.text_editor = configuration[key]
                    continue
                #Need to find a valid default
                if os.environ.has_key('EDITOR'):
                    cls.text_editor = os.environ['EDITOR']
                else:
                    cls.text_editor = cls.find_valid(
                                        ['vi', 'emacs', 'vim', 'gedit', 'nano'],
                                         'text editor')
              
            elif key in ['eps_viewer', 'web_browser']:
                if configuration[key]:
                    cls.eps_viewer = configuration[key]
                    continue
                # else keep None. For Mac this will use the open command.


    @staticmethod
    def find_valid(possibility, program='program'):
        """find a valid shell program in the list"""
        
        for p in possibility:
            if misc.which(p):
                logger.warning('Using default %s \"%s\". ' % (program, p) + \
                             'Set another one in ./input/mg5_configuration.txt')
                return p
        
        logger.warning('No valid %s found. ' % program + \
                                   'Please set in ./input/mg5_configuration.txt')
        return None
        
        
    def open_program(self, program, file_path, mac_check=True):
      """ open a file with a given program """
      
      if mac_check==True and sys.platform == 'darwin':
          return self.open_mac_program(program, file_path)
      
      # Shell program only
      if program:
          subprocess.call([program, file_path])
      else:
          logger.warning('Not able to open file %s since no program configured.' % file_path + \
                              'Please set one in ./input/mg5_configuration.txt') 
    
    def open_mac_program(self, program, file_path):
      """ open a text with the text editor """
      
      if not program:
          # Ask to mac manager
          os.system('open %s' % file_path)
      elif misc.which(program):
          # shell program
          subprocess.call([program, file_path])
      else:
         # not shell program
         os.system('open -a %s %s' % (program, file_path))
      

