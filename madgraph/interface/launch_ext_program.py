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

import logging
import os
import subprocess


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
            
        if os.environ['EDITOR']:
            self.editor =  os.environ['EDITOR']
            return
        
        possibility = ['vi', 'emacs', 'vim', 'gedit', 'nano']
        for editor in possibility:
            if misc.which(editor):
                self.editor = editor
                return
            
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
        
    def ask(self, question, default):
        """ ask a question """
        
        if not self.force:
            try:
                out =  misc.timed_input(question, default, timeout=self.timeout,
                                        noerror=False)
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
            question = 'Do you want to edit file: %(card)s? [y/n/path of the new %(card)s]' 
            question = question % {'card':filename}
            try:
                ans =  misc.timed_input(question, default, timeout=self.timeout,
                                        noerror=False, fct=fct)
            except misc.TimeOutError:
                # avoid to always wait a given time for the next answer
                self.force = True
            else:
                self.timeout=None # answer at least one question so wait...
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
                path = os.path.join(self.running_dir, 'Events','run%s_banner.txt' % i)
                if not os.path.exists(path):
                    self.name = 'run%s' % i
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
        
        answer = self.ask('Do you want to run pythia? [y/n]','n')
        if answer == 'y':
            self.copy_default_card('pythia')
            self.cards.append('pythia_card.dat')
        else:
            path = os.path.join(self.card_dir, 'pythia_card.dat')
            try: os.remove(path)
            except OSError: pass
            return # no Need to ask for PGS
        
        answer = self.ask('Do you want to run PGS? [y/n]','n')
        if answer == 'y':
            self.copy_default_card('pgs')
            self.cards.append('pgs_card.dat')
            return # No Need to ask for Delphes
        else:
            path = os.path.join(self.card_dir, 'pgs_card.dat')
            try: os.remove(path)
            except OSError: pass
    
        if not self.delphes:
            return
        
        answer = self.ask('Do you want to run Delphes? [y/n]','n')
        if answer == 'y':
            self.copy_default_card('delphes')
            self.cards.append('delphes_card.dat')
        else:
            path = os.path.join(self.card_dir, 'delphes_card.dat')
            try: os.remove(path)
            except OSError: pass        
    
    def launch_program(self):
        """launch the main program"""
        
        mode = str(self.cluster)
        if mode == "0":
            print [self.executable, mode, self.name], self.running_dir
            subprocess.call([self.executable, mode, self.name], 
                                                           cwd=self.running_dir)
        elif mode == "1":
            subprocess.call([self.executable, mode, self.name, self.name], 
                                                           cwd=self.running_dir)
        elif mode == "2":
            nb_node = self.ask('How many core do you want to use?', '2')
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
        
        
        
        
        
                    
            