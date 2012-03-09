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

"""A set of functions performing routine administrative I/O tasks."""

import logging
import os
import re
import signal
import subprocess
import sys
import StringIO
import sys
import time

try:
    # Use in MadGraph
    import madgraph
    from madgraph import MadGraph5Error
    import madgraph.iolibs.files as files
except:
    # Use in MadEvent
    import internal as madgraph
    from internal import MadGraph5Error
    import internal.files as files
    
logger = logging.getLogger('cmdprint.ext_program')

   
#===============================================================================
# parse_info_str
#===============================================================================
def parse_info_str(fsock):
    """Parse a newline separated list of "param=value" as a dictionnary
    """

    info_dict = {}
    pattern = re.compile("(?P<name>\w*)\s*=\s*(?P<value>.*)",
                         re.IGNORECASE | re.VERBOSE)
    for entry in fsock:
        entry = entry.strip()
        if len(entry) == 0: continue
        m = pattern.match(entry)
        if m is not None:
            info_dict[m.group('name')] = m.group('value')
        else:
            raise IOError, "String %s is not a valid info string" % entry

    return info_dict


#===============================================================================
# get_pkg_info
#===============================================================================
def get_pkg_info(info_str=None):
    """Returns the current version information of the MadGraph package, 
    as written in the VERSION text file. If the file cannot be found, 
    a dictionary with empty values is returned. As an option, an info
    string can be passed to be read instead of the file content.
    """

    if info_str is None:
        info_dict = files.read_from_file(os.path.join(madgraph.__path__[0],
                                                  "VERSION"),
                                                  parse_info_str, 
                                                  print_error=False)
    else:
        info_dict = parse_info_str(StringIO.StringIO(info_str))

    return info_dict

#===============================================================================
# get_time_info
#===============================================================================
def get_time_info():
    """Returns the present time info for use in MG5 command history header.
    """

    creation_time = time.asctime() 
    time_info = {'time': creation_time,
                 'fill': ' ' * (26 - len(creation_time))}

    return time_info

#===============================================================================
# find a executable
#===============================================================================
def which(program):
    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    if not program:
        return None

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

#===============================================================================
# Return Nice display for a random variable
#===============================================================================
def nice_representation(var, nb_space=0):
    """ Return nice information on the current variable """
    
    #check which data to put:
    info = [('type',type(var)),('str', var)]
    if hasattr(var, 'func_doc'):
        info.append( ('DOC', var.func_doc) )
    if hasattr(var, '__doc__'):
        info.append( ('DOC', var.__doc__) )
    if hasattr(var, '__dict__'):
        info.append( ('ATTRIBUTE', var.__dict__.keys() ))
    
    spaces = ' ' * nb_space

    outstr=''
    for name, value in info:
        outstr += '%s%3s : %s\n' % (spaces,name, value)

    return outstr


#===============================================================================
# Compiler which returns smart output error in case of trouble
#===============================================================================
def compile(arg=[], cwd=None, mode='fortran', **opt):
    """compile a given directory"""

    try:
        p = subprocess.Popen(['make'] + arg, stdout=subprocess.PIPE, 
                             stderr=subprocess.STDOUT, cwd=cwd, **opt)
        (out, err) = p.communicate()
    except OSError, error:
        if cwd and not os.path.exists(cwd):
            raise OSError, 'Directory %s doesn\'t exists. Impossible to run make' % cwd
        else:
            error_text = "Impossible to compile %s directory\n" % cwd
            error_text += "Trying to launch make command returns:\n"
            error_text += "    " + str(error) + "\n"
            error_text += "In general this means that your computer is not able to compile."
            if sys.platform == "darwin":
                error_text += "Note that MacOSX doesn\'t have gmake/gfortan install by default.\n"
                error_text += "Xcode3 contains those require program"
            raise MadGraph5Error, error_text

    if p.returncode:
        # Check that makefile exists
        if not cwd:
            cwd = os.getcwd()
        all_file = [f.lower() for f in os.listdir(cwd)]
        if 'makefile' not in all_file:
            raise OSError, 'no makefile present in %s' % os.path.realpath(cwd)

        if mode == 'fortran' and  not (which('g77') or which('gfortran')):
            error_msg = 'A fortran compilator (g77 or gfortran) is required to create this output.\n'
            error_msg += 'Please install g77 or gfortran on your computer and retry.'
            raise MadGraph5Error, error_msg
        elif mode == 'cpp' and not which('g++'):            
            error_msg ='A C++ compilator (g++) is required to create this output.\n'
            error_msg += 'Please install g++ (which is part of the gcc package)  on your computer and retry.'
            raise MadGraph5Error, error_msg

        # Other reason

        error_text = 'A compilation Error occurs '
        if cwd:
            error_text += 'when trying to compile %s.\n' % cwd
        error_text += 'The compilation fails with the following output message:\n'
        error_text += '    '+out.replace('\n','\n    ')+'\n'
        error_text += 'Please try to fix this compilations issue and retry.\n'
        error_text += 'Help might be found at https://answers.launchpad.net/madgraph5.\n'
        error_text += 'If you think that this is a bug, you can report this at https://bugs.launchpad.net/madgraph5'

        raise MadGraph5Error, error_text
    return p.returncode


################################################################################
# TAIL FUNCTION
################################################################################
def tail(f, n, offset=None):
    """Reads a n lines from f with an offset of offset lines.  The return
    value is a tuple in the form ``lines``.
    """
    avg_line_length = 74
    to_read = n + (offset or 0)

    while 1:
        try:
            f.seek(-(avg_line_length * to_read), 2)
        except IOError:
            # woops.  apparently file is smaller than what we want
            # to step back, go to the beginning instead
            f.seek(0)
        pos = f.tell()
        lines = f.read().splitlines()
        if len(lines) >= to_read or pos == 0:
            return lines[-to_read:offset and -offset or None]
        avg_line_length *= 1.3

################################################################################
# LAST LINE FUNCTION
################################################################################
def get_last_line(fsock):
    """return the last line of a file"""
    
    return tail(fsock, 1)[0]
    


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
            self.open_program(self.web_browser, filename, background=True)
        elif extension in ['ps','eps']:
            self.open_program(self.eps_viewer, filename, background=True)
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
                if configuration[key]:
                    program = configuration[key].split()[0]                    
                    if not which(program):
                        logger.warning('Specified text editor %s not valid.' % \
                                                             configuration[key])
                    else:
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
              
            elif key == 'eps_viewer':
                if configuration[key]:
                    cls.eps_viewer = configuration[key]
                    continue
                # else keep None. For Mac this will use the open command.
            elif key == 'web_browser':
                if configuration[key]:
                    cls.web_browser = configuration[key]
                    continue
                # else keep None. For Mac this will use the open command.

    @staticmethod
    def find_valid(possibility, program='program'):
        """find a valid shell program in the list"""
        
        for p in possibility:
            if which(p):
                logger.info('Using default %s \"%s\". ' % (program, p) + \
                             'Set another one in ./input/mg5_configuration.txt')
                return p
        
        logger.info('No valid %s found. ' % program + \
                                   'Please set in ./input/mg5_configuration.txt')
        return None
        
        
    def open_program(self, program, file_path, mac_check=True, background=False):
      """ open a file with a given program """

      if mac_check==True and sys.platform == 'darwin':
          return self.open_mac_program(program, file_path)

      # Shell program only                                                                                                                                                                 
      if program:
          arguments = program.split() # allow argument in program definition
          arguments.append(file_path)

          if not background:
              subprocess.call(arguments)
          else:
              import thread
              thread.start_new_thread(subprocess.call,(arguments,))
      else:
          logger.warning('Not able to open file %s since no program configured.' % file_path + \
                              'Please set one in ./input/mg5_configuration.txt')
    
    def open_mac_program(self, program, file_path):
      """ open a text with the text editor """
      
      if not program:
          # Ask to mac manager
          os.system('open %s' % file_path)
      elif which(program):
          # shell program
          arguments = program.split() # Allow argument in program definition
          arguments.append(file_path)
          subprocess.call(arguments)
      else:
         # not shell program
         os.system('open -a %s %s' % (program, file_path))

def is_executable(path):
    """ check if a path is executable"""
    try: 
        return os.access(path, os.X_OK)
    except:
        return False        

