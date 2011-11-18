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

import os
import re
import signal
import subprocess
import sys
import StringIO
import time

import madgraph
from madgraph import MadGraph5Error
import madgraph.iolibs.files as files

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
                                                  parse_info_str)
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
    return None

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
# Compiler which returns smart output error in case of trouble
#===============================================================================
def compile(arg=[], cwd=None, mode='fortran', **opt):
    """compile a given directory"""

    try:
        p = subprocess.Popen(['make'] + arg, stdout=subprocess.PIPE, 
                             stderr=subprocess.STDOUT, cwd=cwd, **opt)
        p.wait()
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

        if mode == 'fortran' and (not which('g77') or not which('gfortran')):
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
        error_text += ''.join(['  '+l for l in p.stdout.readlines()])+'\n'
        error_text += 'Please try to fix this compilations issue and retry.\n'
        error_text += 'Help might be found at https://answers.launchpad.net/madgraph5.\n'
        error_text += 'If you think that this is a bug, you can report this at https://bugs.launchpad.net/madgraph5'

        raise MadGraph5Error, error_text

    
#===============================================================================
# Ask a question with a maximum amount of time to answer
#===============================================================================
class TimeOutError(Exception):
    """Class for run-time error"""
         
def timed_input(question, default, timeout=None, noerror=True, fct=None):
    """ a question with a maximal time to answer take default otherwise"""
    
    def handle_alarm(signum, frame): 
            raise TimeOutError
        
    signal.signal(signal.SIGALRM, handle_alarm)
    
    if fct is None:
        fct = raw_input
        
    if timeout:
        signal.alarm(timeout)
        question += '[%ss to answer] ' % (timeout)    
    try:
        result = fct(question)
    except TimeOutError:
        if noerror:
            print '\nuse %s' % default
            return default
        else:
            signal.alarm(0)
            raise
    finally:
        signal.alarm(0)
    return result

