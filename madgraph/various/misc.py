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
"""A set of functions performing routine administrative I/O tasks."""

from __future__ import absolute_import
import collections
import contextlib
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import optparse
import time
import shutil
import stat
import traceback
import gzip as ziplib
import six
from six.moves import zip_longest
from six.moves import range
from six.moves import zip
from six.moves import input
StringIO = six
if six.PY3:
    import io
    file = io.IOBase
try:
    # Use in MadGraph
    import madgraph
except Exception as error:
    # Use in MadEvent
    import internal
    from internal import MadGraph5Error, InvalidCmd
    import internal.files as files
    MADEVENT = True    
else:
    from madgraph import MadGraph5Error, InvalidCmd
    import madgraph.iolibs.files as files
    MADEVENT = False

    
logger = logging.getLogger('cmdprint.ext_program')
logger_stderr = logging.getLogger('madevent.misc')
pjoin = os.path.join
misc = locals
   
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
            raise IOError("String %s is not a valid info string" % entry)

    return info_dict


def glob(name, path=''):
    """call to glob.glob with automatic security on path"""
    import glob as glob_module
    path = re.sub('(?P<name>\?|\*|\[|\])', '[\g<name>]', path)
    return glob_module.glob(pjoin(path, name))

#===============================================================================
# mute_logger (designed to be a decorator)
#===============================================================================
def mute_logger(names=['madgraph','ALOHA','cmdprint','madevent'], levels=[50,50,50,50]):
    """change the logger level and restore those at their initial value at the
    end of the function decorated."""
    def control_logger(f):
        def restore_old_levels(names, levels):
            for name, level in zip(names, levels):
                log_module = logging.getLogger(name)
                log_module.setLevel(level)            
        
        def f_with_no_logger(self, *args, **opt):
            old_levels = []
            for name, level in zip(names, levels):
                log_module = logging.getLogger(name)
                old_levels.append(log_module.level)
                log_module.setLevel(level)
            try:
                out = f(self, *args, **opt)
                restore_old_levels(names, old_levels)
                return out
            except:
                restore_old_levels(names, old_levels)
                raise
            
        return f_with_no_logger
    return control_logger

PACKAGE_INFO = {}
#===============================================================================
# get_pkg_info
#===============================================================================
def get_pkg_info(info_str=None):
    """Returns the current version information of the MadGraph5_aMC@NLO package, 
    as written in the VERSION text file. If the file cannot be found, 
    a dictionary with empty values is returned. As an option, an info
    string can be passed to be read instead of the file content.
    """
    global PACKAGE_INFO

    if info_str:
        info_dict = parse_info_str(StringIO.StringIO(info_str))
        return info_dict

    if PACKAGE_INFO:
        return PACKAGE_INFO
    
    elif MADEVENT:
        info_dict ={}
        info_dict['version'] = open(pjoin(internal.__path__[0],'..','..','MGMEVersion.txt')).read().strip()
        info_dict['date'] = '20xx-xx-xx'
        PACKAGE_INFO = info_dict                        
    else:
        if PACKAGE_INFO:
            return PACKAGE_INFO
        info_dict = files.read_from_file(os.path.join(madgraph.__path__[0],
                                                  "VERSION"),
                                                  parse_info_str, 
                                                  print_error=False)
        if info_dict:                                          
           PACKAGE_INFO = info_dict
        else:
           info_dict ={}
           info_dict['version'] = '3.x.x'
           info_dict['date'] = '20xx-xx-xx'
           PACKAGE_INFO = info_dict        
        return PACKAGE_INFO
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
# Test the compatibility of a given version of MA5 with this version of MG5
#===============================================================================
def is_MA5_compatible_with_this_MG5(ma5path):
    """ Returns None if compatible or, it not compatible, a string explaining 
    why it is so."""

    class version:
        def __init__(input):
            self.info = input.split('.')

        def __lt__(input):
            if isinstance(input, str):
                input = version(input)
            
            for a,b in zip(self.info, input.info):
                both_int = True
                try:
                    a = int(a)
                except Exception:
                     both_int = False
                try:
                    b = int(b)
                except Exception:
                    both_int = False
                if both_int:
                    if a !=b:
                        return a<b
                    continue
                elif isinstance(a, int):
                    return False
                elif isinstance(a, False):
                    return False
                elif a == b:
                    continue
                else:
                    return a < b
            else:
                return len(self.info) < len(input.info)                   

    def LooseVersion(input):
        return version(input)

    ma5_version = None
    try:
        for line in open(pjoin(ma5path,'version.txt'),'r').read().split('\n'):
            if line.startswith('MA5 version :'):
                ma5_version=LooseVersion(line[13:].strip())
                break
    except:
        ma5_version = None

    if ma5_version is None:
        reason = "No MadAnalysis5 version number could be read from the path supplied '%s'."%ma5path
        reason += "\nThe specified version of MadAnalysis5 will not be active in your session."
        return reason
        
    mg5_version = None
    try:
        info = get_pkg_info()        
        mg5_version = LooseVersion(info['version'])
    except:
        mg5_version = None
    
    # If version not reckognized, then carry on as it's probably a development version
    if not mg5_version:
        return None
    
    if mg5_version < LooseVersion("2.6.1") and ma5_version >= LooseVersion("1.6.32"):
        reason =  "This active MG5aMC version is too old (v%s) for your selected version of MadAnalysis5 (v%s)"%(mg5_version,ma5_version)
        reason += "\nUpgrade MG5aMC or re-install MA5 from within MG5aMC to fix this compatibility issue."
        reason += "\nThe specified version of MadAnalysis5 will not be active in your session."
        return reason

    if mg5_version >= LooseVersion("2.6.1") and ma5_version < LooseVersion("1.6.32"):
        reason = "Your selected version of MadAnalysis5 (v%s) is too old for this active version of MG5aMC (v%s)."%(ma5_version,mg5_version)
        reason += "\nRe-install MA5 from within MG5aMC to fix this compatibility issue."
        reason += "\nThe specified version of MadAnalysis5 will not be active in your session."
        return reason

    return None

#===============================================================================
# Find the subdirectory which includes the files ending with a given extension 
#===============================================================================
def find_includes_path(start_path, extension):
    """Browse the subdirectories of the path 'start_path' and returns the first
    one found which contains at least one file ending with the string extension
    given in argument."""
    
    if not os.path.isdir(start_path):
        return None
    subdirs=[pjoin(start_path,dir) for dir in os.listdir(start_path)]
    for subdir in subdirs:
        if os.path.isfile(subdir):
            if os.path.basename(subdir).endswith(extension):
                return start_path
        elif os.path.isdir(subdir):
            path = find_includes_path(subdir, extension)
            if path:
                return path
    return None

#===============================================================================
# Given the path of a ninja installation, this function determines if it 
# supports quadruple precision or not. 
#===============================================================================
def get_ninja_quad_prec_support(ninja_lib_path):
    """ Get whether ninja supports quad prec in different ways"""
    
    # First try with the ninja-config executable if present
    ninja_config = os.path.abspath(pjoin(
                                 ninja_lib_path,os.pardir,'bin','ninja-config'))
    if os.path.exists(ninja_config):
        try:    
            p = Popen([ninja_config, '-quadsupport'], stdout=subprocess.PIPE, 
                                                         stderr=subprocess.PIPE)
            output, error = p.communicate()
            return 'TRUE' in output.decode(errors='ignore').upper()
        except Exception:
            pass
    
    # If no ninja-config is present, then simply use the presence of
    # 'quadninja' in the include
    return False

#===============================================================================
# find a executable
#===============================================================================
def which(program):
    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(\
                                               os.path.realpath(fpath), os.X_OK)

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

def has_f2py():
    has_f2py = False
    if which('f2py'):
        has_f2py = True
    elif which('f2py%d.%d' %(sys.version_info.major, sys.version_info.minor)):
        has_f2py = True
    elif which('f2py%d' %(sys.version_info.major)):
        has_f2py = True

    return has_f2py       
        
#===============================================================================
#  Activate dependencies if possible. Mainly for tests
#===============================================================================

def deactivate_dependence(dependency, cmd=None, log = None):
    """ Make sure to turn off some dependency of MG5aMC. """
    
    def tell(msg):
        if log == 'stdout':
            print(msg)
        elif callable(log):
            log(msg)
    

    if dependency in ['pjfry','golem','samurai','ninja','collier']:
        if dependency not in cmd.options:
            return
        if cmd.options[dependency] not in ['None',None,'']:
            tell("Deactivating MG5_aMC dependency '%s'"%dependency)
            cmd.options[dependency] = None

def activate_dependence(dependency, cmd=None, log = None, MG5dir=None):
    """ Checks whether the specfieid MG dependency can be activated if it was
    not turned off in MG5 options."""
    
    def tell(msg):
        if log == 'stdout':
            print(msg)
        elif callable(log):
            log(msg)

    if cmd is None:
        cmd = MGCmd.MasterCmd()

    if dependency=='pjfry':
        if cmd.options['pjfry'] in ['None',None,''] or \
         (cmd.options['pjfry'] == 'auto' and which_lib('libpjfry.a') is None) or\
          which_lib(pjoin(cmd.options['pjfry'],'libpjfry.a')) is None:
            tell("Installing PJFry...")
            cmd.do_install('PJFry')

    if dependency=='golem':
        if cmd.options['golem'] in ['None',None,''] or\
         (cmd.options['golem'] == 'auto' and which_lib('libgolem.a') is None) or\
         which_lib(pjoin(cmd.options['golem'],'libgolem.a')) is None:
            tell("Installing Golem95...")
            cmd.do_install('Golem95')
    
    if dependency=='samurai':
        raise MadGraph5Error('Samurai cannot yet be automatically installed.') 

    if dependency=='ninja':
        if cmd.options['ninja'] in ['None',None,''] or\
         (cmd.options['ninja'] == './HEPTools/lib' and not MG5dir is None and\
         which_lib(pjoin(MG5dir,cmd.options['ninja'],'lib','libninja.a')) is None):
            tell("Installing ninja...")
            cmd.do_install('ninja')
 
    if dependency=='collier':
        if cmd.options['collier'] in ['None',None,''] or\
         (cmd.options['collier'] == 'auto' and which_lib('libcollier.a') is None) or\
         which_lib(pjoin(cmd.options['collier'],'libcollier.a')) is None:
            tell("Installing COLLIER...")
            cmd.do_install('collier')

#===============================================================================
# find a library
#===============================================================================
def which_lib(lib):
    def is_lib(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.R_OK)

    if not lib:
        return None

    fpath, fname = os.path.split(lib)
    if fpath:
        if is_lib(lib):
            return lib
    else:
        locations = sum([os.environ[env_path].split(os.pathsep) for env_path in
           ["DYLD_LIBRARY_PATH","LD_LIBRARY_PATH","LIBRARY_PATH","PATH"] 
                                                  if env_path in os.environ],[])
        for path in locations:
            lib_file = os.path.join(path, lib)
            if is_lib(lib_file):
                return lib_file
    return None

#===============================================================================
# Return Nice display for a random variable
#===============================================================================
def nice_representation(var, nb_space=0):
    """ Return nice information on the current variable """
    
    #check which data to put:
    info = [('type',type(var)),('str', var)]
    if hasattr(var, 'func_doc'):
        info.append( ('DOC', var.__doc__) )
    if hasattr(var, '__doc__'):
        info.append( ('DOC', var.__doc__) )
    if hasattr(var, '__dict__'):
        info.append( ('ATTRIBUTE', list(var.__dict__.keys()) ))
    
    spaces = ' ' * nb_space

    outstr=''
    for name, value in info:
        outstr += '%s%3s : %s\n' % (spaces,name, value)

    return outstr

#
# Decorator for re-running a crashing function automatically.
#
wait_once = False
def multiple_try(nb_try=5, sleep=20):

    def deco_retry(f):
        def deco_f_retry(*args, **opt):
            my_error = None
            for i in range(nb_try):
                try:
                    return f(*args, **opt)
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    my_error = error
                    global wait_once
                    if not wait_once:
                        text = """Start waiting for update. (more info in debug mode)"""
                        logger.info(text)
                        logger_stderr.debug('fail to do %s function with %s args. %s try on a max of %s (%s waiting time)' %
                                 (str(f), ', '.join([str(a) for a in args]), i+1, nb_try, sleep * (i+1)))
                        logger_stderr.debug('error is %s' % str(error))
                        if __debug__: logger_stderr.debug('and occurred at :'+traceback.format_exc())
                    wait_once = True
                    time.sleep(sleep * (i+1))

            if __debug__:
                raise
            raise my_error.__class__('[Fail %i times] \n %s ' % (i+1, my_error))
        return deco_f_retry
    return deco_retry

#===============================================================================
# helper for scan. providing a nice formatted string for the scan name
#===============================================================================
def get_scan_name(first, last):
    """return a name of the type xxxx[A-B]yyy
        where xxx and yyy are the common part between the two names.
    """
    
    # find the common string at the beginning     
    base = [first[i] for i in range(len(first)) if first[:i+1] == last[:i+1]]
    # remove digit even if in common
    while base and base[0].isdigit():
        base = base[1:] 
    # find the common string at the end 
    end = [first[-(i+1)] for i in range(len(first)) if first[-(i+1):] == last[-(i+1):]]
    # remove digit even if in common    
    while end and end[-1].isdigit():
        end = end[:-1] 
    end.reverse()
    #convert to string
    base, end = ''.join(base), ''.join(end)
    if end:
        name = "%s[%s-%s]%s" % (base, first[len(base):-len(end)], last[len(base):-len(end)],end)
    else:
        name = "%s[%s-%s]%s" % (base, first[len(base):], last[len(base):],end)
    return name

def copytree(*args, **opts):

    if 'copy_function' not in opts:
        opts['copy_function'] = shutil.copy
    return shutil.copytree(*args, **opts)

#===============================================================================
# Compiler which returns smart output error in case of trouble
#===============================================================================
def compile(arg=[], cwd=None, mode='fortran', job_specs = True, nb_core=1 ,**opt):
    """compile a given directory"""

    if 'nocompile' in opt:
        if opt['nocompile'] == True:
            if not arg:
                return
            if cwd:
                executable = pjoin(cwd, arg[0])
            else:
                executable = arg[0]
            if os.path.exists(executable):
                return
        del opt['nocompile']

    cmd = ['make']
    try:
        if nb_core > 1:
            cmd.append('-j%s' % nb_core)
        cmd += arg
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                             stderr=subprocess.STDOUT, cwd=cwd, **opt)
        (out, err) = p.communicate()
    except OSError as error:
        if cwd and not os.path.exists(cwd):
            raise OSError('Directory %s doesn\'t exists. Impossible to run make' % cwd)
        else:
            error_text = "Impossible to compile %s directory\n" % cwd
            error_text += "Trying to launch make command returns:\n"
            error_text += "    " + str(error) + "\n"
            error_text += "In general this means that your computer is not able to compile."
            if sys.platform == "darwin":
                error_text += "Note that MacOSX doesn\'t have gmake/gfortan install by default.\n"
                error_text += "Xcode contains gmake. For gfortran we advise: http://hpc.sourceforge.net/"
            raise MadGraph5Error(error_text)

    if p.returncode:
        # Check that makefile exists
        if not cwd:
            cwd = os.getcwd()
        all_file = [f.lower() for f in os.listdir(cwd)]
        if 'makefile' not in all_file and '-f' not in arg:
            raise OSError('no makefile present in %s' % os.path.realpath(cwd))

        if mode == 'fortran' and  not (which('g77') or which('gfortran')):
            error_msg = 'A fortran compiler (g77 or gfortran) is required to create this output.\n'
            error_msg += 'Please install g77 or gfortran on your computer and retry.'
            raise MadGraph5Error(error_msg)
        elif mode == 'cpp' and not which('g++'):            
            error_msg ='A C++ compiler (g++) is required to create this output.\n'
            error_msg += 'Please install g++ (which is part of the gcc package)  on your computer and retry.'
            raise MadGraph5Error(error_msg)

        try:
            out = out.decode('utf-8', errors='ignore')
        except Exception:
            out = str(out)

        # Check if this is due to the need of gfortran 4.6 for quadruple precision
        if any(tag.upper() in out.upper() for tag in ['real(kind=16)','real*16',
            'complex*32']) and mode == 'fortran' and not \
                             ''.join(get_gfortran_version().split('.')) >= '46':
            if not which('gfortran'):
                raise MadGraph5Error('The fortran compiler gfortran v4.6 or later '+\
                  'is required to compile %s.\nPlease install it and retry.'%cwd)
            else:
                logger_stderr.error('ERROR, you could not compile %s because'%cwd+\
             ' your version of gfortran is older than 4.6. MadGraph5_aMC@NLO will carry on,'+\
                              ' but will not be able to compile an executable.')
                return p.returncode
        # Other reason
        error_text = 'A compilation Error occurs '
        if cwd:
            error_text += 'when trying to compile %s.\n' % cwd
        error_text += 'The compilation fails with the following output message:\n'
        error_text += '    '+out.replace('\n','\n    ')+'\n'
        error_text += 'Please try to fix this compilations issue and retry.\n'
        error_text += 'Help might be found at https://answers.launchpad.net/mg5amcnlo.\n'
        error_text += 'If you think that this is a bug, you can report this at https://bugs.launchpad.net/mg5amcnlo'
        raise MadGraph5Error(error_text)
    return p.returncode

def get_gfortran_version(compiler='gfortran'):
    """ Returns the gfortran version as a string.
        Returns '0' if it failed."""
    try:    
        p = Popen([compiler, '-dumpversion'], stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)
        output, error = p.communicate()
        output = output.decode("utf-8",errors='ignore')
        version_finder=re.compile(r"(?P<version>\d[\d.]*)")
        version = version_finder.search(output).group('version')
        return version
    except Exception as error:
        raise error
        return '0'

def mod_compilator(directory, new='gfortran', current=None, compiler_type='gfortran'):
    #define global regular expression
    if type(directory)!=list:
        directory=[directory]

    #search file
    file_to_change=find_makefile_in_dir(directory)
    if compiler_type == 'gfortran':
        comp_re = re.compile('^(\s*)FC\s*=\s*(.+)\s*$')
        var = 'FC'
    elif compiler_type == 'cpp':
        comp_re = re.compile('^(\s*)CXX\s*=\s*(.+)\s*$')
        var = 'CXX'
    else:
        MadGraph5Error, 'Unknown compiler type: %s' % compiler_type

    mod = False
    for name in file_to_change:
        lines = open(name,'r').read().split('\n')
        for iline, line in enumerate(lines):
            result = comp_re.match(line)
            if result:
                if new != result.group(2) and '$' not in result.group(2):
                    mod = True
                    lines[iline] = result.group(1) + var + "=" + new
            elif compiler_type == 'gfortran' and line.startswith('DEFAULT_F_COMPILER'):
                lines[iline] = "DEFAULT_F_COMPILER = %s" % new
            elif compiler_type == 'cpp' and line.startswith('DEFAULT_CPP_COMPILER'):    
                lines[iline] = "DEFAULT_CPP_COMPILER = %s" % new
                
        if mod:
            open(name,'w').write('\n'.join(lines))
            # reset it to change the next file
            mod = False

def pid_exists(pid):
    """Check whether pid exists in the current process table.
    UNIX only.
    https://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid-in-python
    """
    import errno
    
    if pid < 0:
        return False
    if pid == 0:
        # According to "man 2 kill" PID 0 refers to every process
        # in the process group of the calling process.
        # On certain systems 0 is a valid PID but we have no way
        # to know that in a portable fashion.
        raise ValueError('invalid PID 0')
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True



#===============================================================================
# mute_logger (designed to work as with statement)
#===============================================================================
class MuteLogger(object):
    """mute_logger (designed to work as with statement),
       files allow to redirect the output of the log to a given file.
    """

    def __init__(self, names, levels, files=None, **opt):
        assert isinstance(names, list)
        assert isinstance(names, list)
        
        self.names = names
        self.levels = levels
        if isinstance(files, list):
            self.files = files
        else:
            self.files = [files] * len(names)
        self.logger_saved_info = {}
        self.opts = opt

    def __enter__(self):
        old_levels = []
        for name, level, path in zip(self.names, self.levels, self.files):
            if path:
                self.setup_logFile_for_logger(path, name, **self.opts)
            log_module = logging.getLogger(name)
            old_levels.append(log_module.level)
            log_module = logging.getLogger(name)
            log_module.setLevel(level)
        self.levels = old_levels
        
    def __exit__(self, ctype, value, traceback ):
        for name, level, path in zip(self.names, self.levels, self.files):

            if path:
                if 'keep' in self.opts and not self.opts['keep']:
                    self.restore_logFile_for_logger(name, level, path=path)
                else:
                    self.restore_logFile_for_logger(name, level)
            else:
                log_module = logging.getLogger(name)
                log_module.setLevel(level)         
        
    def setup_logFile_for_logger(self, path, full_logname, **opts):
        """ Setup the logger by redirecting them all to logfiles in tmp """
        
        logs = full_logname.split('.')
        lognames = [ '.'.join(logs[:(len(logs)-i)]) for i in\
                                            range(len(full_logname.split('.')))]
        for logname in lognames:
            try:
                os.remove(path)
            except Exception as error:
                pass
            my_logger = logging.getLogger(logname)
            hdlr = logging.FileHandler(path)            
            # I assume below that the orders of the handlers in my_logger.handlers
            # remains the same after having added/removed the FileHandler
            self.logger_saved_info[logname] = [hdlr, my_logger.handlers]
            #for h in my_logger.handlers:
            #    h.setLevel(logging.CRITICAL)
            for old_hdlr in list(my_logger.handlers):
                my_logger.removeHandler(old_hdlr)
            my_logger.addHandler(hdlr)
            #my_logger.setLevel(level)
            my_logger.debug('Log of %s' % logname)

    def restore_logFile_for_logger(self, full_logname, level, path=None, **opts):
        """ Setup the logger by redirecting them all to logfiles in tmp """
        
        logs = full_logname.split('.')
        lognames = [ '.'.join(logs[:(len(logs)-i)]) for i in\
                                            range(len(full_logname.split('.')))]
        for logname in lognames:
            if path:
                try:
                    os.remove(path)
                except Exception as error:
                    pass
            my_logger = logging.getLogger(logname)
            if logname in self.logger_saved_info:
                my_logger.removeHandler(self.logger_saved_info[logname][0])
                for old_hdlr in self.logger_saved_info[logname][1]:
                    my_logger.addHandler(old_hdlr)
            else:
                my_logger.setLevel(level)
        
            #for i, h in enumerate(my_logger.handlers):
            #    h.setLevel(cls.logger_saved_info[logname][2][i])

nb_open =0
@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """                                                                                                                                                                                                     
    A context manager to temporarily redirect stdout or stderr                                                                                                                                              
                                                                                                                                                                                                            
    e.g.:                                                                                                                                                                                                   
                                                                                                                                                                                                            
                                                                                                                                                                                                            
    with stdchannel_redirected(sys.stderr, os.devnull):                                                                                                                                                     
        if compiler.has_function('clock_gettime', libraries=['rt']):                                                                                                                                        
            libraries.append('rt')                                                                                                                                                                          
    """

    if logger.getEffectiveLevel()>5:
        #deactivate this for hard-core debugging level
        try:
            oldstdchannel = os.dup(stdchannel.fileno())
            dest_file = open(dest_filename, 'w')
            os.dup2(dest_file.fileno(), stdchannel.fileno())
            yield
        finally:
            if oldstdchannel is not None:
                os.dup2(oldstdchannel, stdchannel.fileno())
                os.close(oldstdchannel)
            if dest_file is not None:
                dest_file.close()
    else:
        try:
            logger.debug('no stdout/stderr redirection due to debug level')
            yield
        finally:
            return
        
        
def get_open_fds():
    '''
    return the number of open file descriptors for current process

    .. warning: will only work on UNIX-like os-es.
    '''
    import subprocess
    import os

    pid = os.getpid()
    procs = subprocess.check_output( 
        [ "lsof", '-w', '-Ff', "-p", str( pid ) ] )
    nprocs = [s for s in procs.split( '\n' ) if s and s[ 0 ] == 'f' and s[1: ].isdigit()]
        
    return nprocs

def detect_if_cpp_compiler_is_clang(cpp_compiler):
    """ Detects whether the specified C++ compiler is clang."""
    
    try:
        p = Popen([cpp_compiler, '--version'], stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)
        output, error = p.communicate()
    except Exception as error:
        # Cannot probe the compiler, assume not clang then
        return False

    output = output.decode(errors='ignore')
    
    return 'LLVM' in str(output) or "clang" in str(output)


def detect_cpp_std_lib_dependence(cpp_compiler):
    """ Detects if the specified c++ compiler will normally link against the C++
    standard library -lc++ or -libstdc++."""

    is_clang = detect_if_cpp_compiler_is_clang(cpp_compiler)
    if is_clang:
        try:
            import platform
            v, _,_ = platform.mac_ver()
            if not v:
                # We will not attempt to support clang elsewhere than on macs, so
                # we venture a guess here.
                return '-lc++'
            else:
                maj, v = [float(x) for x in v.rsplit('.')[:2]]
                if maj >=11 or (maj ==10 and v >= 9):
                   return '-lc++'
                else:
                   return '-lstdc++'
        except:
            return '-lstdc++'
    return '-lstdc++'

def detect_current_compiler(path, compiler_type='fortran'):
    """find the current compiler for the current directory"""
    
#    comp = re.compile("^\s*FC\s*=\s*(\w+)\s*")
#   The regular expression below allows for compiler definition with absolute path
    if compiler_type == 'fortran':
        comp = re.compile("^\s*FC\s*=\s*([\w\/\\.\-]+)\s*")
    elif compiler_type == 'cpp':
        comp = re.compile("^\s*CXX\s*=\s*([\w\/\\.\-]+)\s*")
    else:
        MadGraph5Error, 'Unknown compiler type: %s' % compiler_type

    for line in open(path):
        if comp.search(line):
            compiler = comp.search(line).groups()[0]
            return compiler
        elif compiler_type == 'fortran' and line.startswith('DEFAULT_F_COMPILER'):
            return line.split('=')[1].strip()
        elif compiler_type == 'cpp' and line.startswith('DEFAULT_CPP_COMPILER'):
            return line.split('=')[1].strip()

def find_makefile_in_dir(directory):
    """ return a list of all file starting with makefile in the given directory"""

    out=[]
    #list mode
    if type(directory)==list:
        for name in directory:
            out+=find_makefile_in_dir(name)
        return out

    #single mode
    for name in os.listdir(directory):
        if os.path.isdir(directory+'/'+name):
            out+=find_makefile_in_dir(directory+'/'+name)
        elif os.path.isfile(directory+'/'+name) and name.lower().startswith('makefile'):
            out.append(directory+'/'+name)
        elif os.path.isfile(directory+'/'+name) and name.lower().startswith('make_opt'):
            out.append(directory+'/'+name)
    return out

def rm_old_compile_file():

    # remove all the .o files
    os.path.walk('.', rm_file_extension, '.o')
    
    # remove related libraries
    libraries = ['libblocks.a', 'libgeneric_mw.a', 'libMWPS.a', 'libtools.a', 'libdhelas3.a',
                 'libdsample.a', 'libgeneric.a', 'libmodel.a', 'libpdf.a', 'libgammaUPC.a','libdhelas3.so', 'libTF.a', 
                 'libdsample.so', 'libgeneric.so', 'libmodel.so', 'libpdf.so', 'libgammaUPC.so']
    lib_pos='./lib'
    [os.remove(os.path.join(lib_pos, lib)) for lib in libraries \
                                 if os.path.exists(os.path.join(lib_pos, lib))]


def format_time(n_secs):
    m, s = divmod(n_secs, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return "%d day%s,%dh%02dm%02ds" % (d,'' if d<=1 else 's',h, m, s)
    elif h > 0:
        return "%dh%02dm%02ds" % (h, m, s)
    elif m > 0:
        return "%dm%02ds" % (m, s)                
    else:
        return "%d second%s" % (s, '' if s<=1 else 's')   

def rm_file_extension( ext, dirname, names):

    [os.remove(os.path.join(dirname, name)) for name in names if name.endswith(ext)]



def multiple_replacer(*key_values):
    replace_dict = dict(key_values)
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = re.compile("|".join([re.escape(k) for k, v in key_values]), re.M)
    return lambda string: pattern.sub(replacement_function, string)

def multiple_replace(string, *key_values):
    return multiple_replacer(*key_values)(string)

# Control
def check_system_error(value=1):
    def deco_check(f):
        def deco_f(arg, *args, **opt):
            try:
                return f(arg, *args, **opt)
            except OSError as error:
                logger.debug('try to recover from %s' % error)
                if isinstance(arg, (list,tuple)):
                    prog =  arg[0]
                else:
                    prog = arg
                
                # Permission denied
                if error.errno == 13:     
                    if os.path.exists(prog):
                        os.system('chmod +x %s' % prog)
                    elif 'cwd' in opt and opt['cwd'] and \
                                       os.path.isfile(pjoin(opt['cwd'],arg[0])):
                        os.system('chmod +x %s' % pjoin(opt['cwd'],arg[0]))
                    return f(arg, *args, **opt)
                # NO such file or directory
                elif error.errno == 2:
                    # raise a more meaningfull error message
                    raise Exception('%s fails with no such file or directory' \
                                                                           % arg)            
                else:
                    raise
        return deco_f
    return deco_check


@check_system_error()
def call(arg, *args, **opt):
    """nice way to call an external program with nice error treatment"""
    try:
        return subprocess.call(arg, *args, **opt)
    except OSError:
        arg[0] = './%s' % arg[0]
        return subprocess.call(arg, *args, **opt)
        
@check_system_error()
def Popen(arg, *args, **opt):
    """nice way to call an external program with nice error treatment"""
    return subprocess.Popen(arg, *args, **opt)

@check_system_error()
def call_stdout(arg, *args, **opt):
    """nice way to call an external program with nice error treatment"""
    try:
        out = subprocess.Popen(arg, *args, stdout=subprocess.PIPE, **opt)
    except OSError:
        arg[0] = './%s' % arg[0]
        out = subprocess.call(arg, *args,  stdout=subprocess.PIPE, **opt)
        
    str_out = out.stdout.read().decode(errors='ignore').strip()
    return str_out



def copytree(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if symlinks and os.path.islink(s):
      if os.path.lexists(d):
        os.remove(d)
      os.symlink(os.readlink(s), d)
      try:
        st = os.lstat(s)
        mode = stat.S_IMODE(st.st_mode)
        os.lchmod(d, mode)
      except:
        pass # lchmod not available
    elif os.path.isdir(s):
      copytree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)

    

@multiple_try()
def mult_try_open(filepath, *args, **opt):
    """try to open a file with multiple try to ensure that filesystem is sync"""  
    return open(filepath, *args, ** opt)

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
        avg_line_length = int(avg_line_length)

def mkfifo(fifo_path):
    """ makes a piping fifo (First-in First-out) file and nicely intercepts 
    error in case the file format of the target drive doesn't suppor tit."""

    try:
        os.mkfifo(fifo_path)
    except:
        raise OSError('MadGraph5_aMCatNLO could not create a fifo file at:\n'+
          '   %s\n'%fifo_path+'Make sure that this file does not exist already'+
          ' and that the file format of the target drive supports fifo file (i.e not NFS).')

################################################################################
# LAST LINE FUNCTION
################################################################################
def get_last_line(fsock):
    """return the last line of a file"""
    
    return tail(fsock, 1)[0]


#https://stackoverflow.com/questions/2301789/read-a-file-in-reverse-order-using-python
def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment




def write_PS_input(filePath, PS):
    """ Write out in file filePath the PS point to be read by the MadLoop."""
    try:
        PSfile = open(filePath, 'w')
        # Add a newline in the end as the implementation fortran 'read'
        # command on some OS is problematic if it ends directly with the
        # floating point number read.

        PSfile.write('\n'.join([' '.join(['%.16E'%pi for pi in p]) \
                                                             for p in PS])+'\n')
        PSfile.close()
    except Exception:
        raise MadGraph5Error('Could not write out the PS point to file %s.'\
                                                                  %str(filePath))

def format_timer(running_time):
    """ return a nicely string representing the time elapsed."""
    if running_time < 2e-2:
        running_time = running_time = 'current time: %02dh%02d' % (time.localtime().tm_hour, time.localtime().tm_min) 
    elif running_time < 10:
        running_time = ' %.2gs ' % running_time
    elif 60 > running_time >= 10:
        running_time = ' %.3gs ' % running_time
    elif 3600 > running_time >= 60:
        running_time = ' %im %is ' % (running_time // 60, int(running_time % 60))
    else:
        running_time = ' %ih %im ' % (running_time // 3600, (running_time//60 % 60))
    return running_time
    

#===============================================================================
# TMP_directory (designed to work as with statement)
#===============================================================================
class TMP_directory(object):
    """create a temporary directory and ensure this one to be cleaned.
    """

    def __init__(self, suffix='', prefix='tmp', dir=None):
        self.nb_try_remove = 0
        import tempfile   
        self.path = tempfile.mkdtemp(suffix, prefix, dir)

    
    def __exit__(self, ctype, value, traceback ):
        #True only for debugging:
        if False and isinstance(value, Exception):
            sprint("Directory %s not cleaned. This directory can be removed manually" % self.path)
            return False
        try:
            shutil.rmtree(self.path)
        except OSError:
            self.nb_try_remove += 1
            if self.nb_try_remove < 3:
                time.sleep(10)
                self.__exit__(ctype, value, traceback)
            else:
                logger.warning("Directory %s not completely cleaned. This directory can be removed manually" % self.path)
        
    def __enter__(self):
        return self.path
    
class TMP_variable(object):
    """replace an attribute of a class with another value for the time of the
       context manager
    """

    def __init__(self, cls, attribute, value):

        self.cls = cls
        self.attribute = attribute        
        if isinstance(attribute, list):
            self.old_value = []
            for key, onevalue in zip(attribute, value):
                self.old_value.append(getattr(cls, key))
                setattr(self.cls, key, onevalue)
        else:
            self.old_value = getattr(cls, attribute)
            setattr(self.cls, self.attribute, value)
    
    def __exit__(self, ctype, value, traceback ):
        
        if isinstance(self.attribute, list):
            for key, old_value in zip(self.attribute, self.old_value):
                setattr(self.cls, key, old_value)
        else:
            setattr(self.cls, self.attribute, self.old_value)
        
    def __enter__(self):
        return self.old_value 
    
    #allow to use as decorator
    def __call__(self, func):
        def wrapper(*args, **kwds):
            with self:
                return func(*args, **kwds)
    
#
# GUNZIP/GZIP
#
def gunzip(path, keep=False, stdout=None):
    """ a standard replacement for os.system('gunzip -f %s.gz ' % event_path)"""

    
    if not path.endswith(".gz"):
        if os.path.exists("%s.gz" % path):
            path = "%s.gz" % path
        else:
            raise Exception("%(path)s does not finish by .gz and the file %(path)s.gz does not exists" %\
                              {"path": path})         

    
    #for large file (>1G) it is faster and safer to use a separate thread
    if os.path.getsize(path) > 1e8:
        if stdout:
            os.system('gunzip -c %s > %s' % (path, stdout))
        else:
            os.system('gunzip  %s' % path) 
        return 0
    
    if not stdout:
        stdout = path[:-3]
    try:
        gfile = ziplib.open(path, "r")
    except IOError:
        raise
    else:    
        try:    
            open(stdout,'w').write(gfile.read().decode(errors='ignore'))
        except IOError as error:
            sprint(error)
            # this means that the file is actually not gzip
            if stdout == path:
                return
            else:
                files.cp(path, stdout)
    finally:
        gfile.close()    
    if not keep:
        os.remove(path)
    return 0

def gzip(path, stdout=None, error=True, forceexternal=False):
    """ a standard replacement for os.system('gzip %s ' % path)"""
 
    #for large file (>1G) it is faster and safer to use a separate thread
    if os.path.getsize(path) > 1e9 or forceexternal:
        call(['gzip', '-f', path])
        if stdout:
            if not stdout.endswith(".gz"):
                stdout = "%s.gz" % stdout
            shutil.move('%s.gz' % path, stdout)
        return
    
    if not stdout:
        stdout = "%s.gz" % path
    elif not stdout.endswith(".gz"):
        stdout = "%s.gz" % stdout

    try:
        with ziplib.open(stdout, 'wb') as f:
            f.write(open(path).read().encode())
    except OverflowError:
        gzip(path, stdout, error=error, forceexternal=True)
    except Exception:
        if error:
            raise
        else:
            return
    finally:
        os.remove(path)
    
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
           cls.eps_viewer = cls.find_valid(['evince','gv', 'ggv'], 'eps viewer') 
            
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
                if 'EDITOR' in os.environ:
                    cls.text_editor = os.environ['EDITOR']
                else:
                    cls.text_editor = cls.find_valid(
                                        ['vi', 'emacs', 'vim', 'gedit', 'nano'],
                                         'text editor')
              
            elif key == 'eps_viewer':
                if configuration[key]:
                    cls.eps_viewer = configuration[key]
                if sys.platform == 'darwin':
                    import platform
                    ver, _, _ =  platform.mac_ver()
                    # open does not support eps anymore since 13.0
                    # pass by a converter first
                    if int(ver.split('.')[0]) > 12: 
                        if which('pstopdf'):
                            cls.eps_viewer = 'pstopdf'
                        elif  which('ps2pdf'):
                            cls.eps_viewer = 'ps2pdf'

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
                import six.moves._thread
                six.moves._thread.start_new_thread(subprocess.call,(arguments,))
        else:
            logger.warning('Not able to open file %s since no program configured.' % file_path + \
                                'Please set one in ./input/mg5_configuration.txt')

    def open_mac_program(self, program, file_path):
        """ open a text with the text editor """

        if not program:
            # Ask to mac manager
            os.system('open %s' % file_path)
        elif program == 'pstopdf':
            output = file_path.rsplit('.',1)[0]+ '.pdf'
            arguments = [program, file_path, '-o', output]
            subprocess.call(arguments, stdout=open(os.devnull,"w"))
            return self.open_mac_program(None, output)
        elif program == 'ps2pdf':
            output = file_path.rsplit('.',1)[0]+ '.pdf'
            arguments = [program, file_path, output]
            subprocess.call(arguments, stdout=open(os.devnull,"w"))
            return self.open_mac_program(None, output)
        elif which(program): 
            # shell program
            arguments = program.split() # Allow argument in program definition
            arguments.append(file_path)
            subprocess.call(arguments)
        else:
            # not shell program
            os.system('open -a %s %s' % (program, file_path))

def get_HEPTools_location_setter(HEPToolsDir,type):
    """ Checks whether mg5dir/HEPTools/<type> (which is 'lib', 'bin' or 'include')
    is in the environment paths of the user. If not, it returns a preamble that
    sets it before calling the exectuable, for example:
       <preamble> ./my_exe
    with <preamble> -> DYLD_LIBRARY_PATH=blabla:$DYLD_LIBRARY_PATH"""
    
    assert(type in ['bin','include','lib'])
    
    target_env_var = 'PATH' if type in ['bin','include'] else \
          ('DYLD_LIBRARY_PATH' if sys.platform=='darwin' else 'LD_LIBRARY_PATH')
    
    target_path = os.path.abspath(pjoin(HEPToolsDir,type))
    
    if target_env_var not in os.environ or \
                target_path not in os.environ[target_env_var].split(os.pathsep):
        return "%s=%s:$%s "%(target_env_var,target_path,target_env_var)
    else:
        return ''

def get_shell_type():
    """ Try and guess what shell type does the user use."""
    try:
        if os.environ['SHELL'].endswith('bash'):
            return 'bash'
        elif os.environ['SHELL'].endswith('tcsh'):
            return 'tcsh'
        else:
            # If unknown, return None
            return None 
    except KeyError:
        return None

def is_executable(path):
    """ check if a path is executable"""
    try: 
        return os.access(path, os.X_OK)
    except Exception:
        return False        
    
class OptionParser(optparse.OptionParser):
    """Option Peaser which raise an error instead as calling exit"""
    
    def exit(self, status=0, msg=None):
        if msg:
            raise InvalidCmd(msg)
        else:
            raise InvalidCmd

def sprint(*args, **opt):
    """Returns the current line number in our program."""
    
    if not __debug__:
        return
    

    import inspect
    if 'cond' in opt and not opt['cond']:
        return

    use_print = False    
    if 'use_print' in opt and opt['use_print']:
        use_print = True
    
    if 'log' in opt:
        log = opt['log']
    else:
        log = logging.getLogger('madgraph')
    if 'level' in opt:
        level = opt['level']
    else:
        level = logging.getLogger('madgraph').level
        if level == 0:
            use_print = True
        #print  "madgraph level",level
        #if level == 20:
        #    level = 10 #avoid info level
        #print "use", level
    if 'wait' in opt:
        wait = bool(opt['wait'])
    else:
        wait = False
        
    lineno  =  inspect.currentframe().f_back.f_lineno
    fargs =  inspect.getframeinfo(inspect.currentframe().f_back)
    filename, lineno = fargs[:2]
    #file = inspect.currentframe().f_back.co_filename
    #print type(file)
    try:
        source = inspect.getsourcelines(inspect.currentframe().f_back)
        line = source[0][lineno-source[1]]
        line = re.findall(r"misc\.sprint\(\s*(.*)\)\s*($|#)", line)[0][0]
        if line.startswith("'") and line.endswith("'") and line.count(",") ==0:
            line= ''
        elif line.startswith("\"") and line.endswith("\"") and line.count(",") ==0:
            line= ''
        elif line.startswith(("\"","'")) and len(args)==1 and "%" in line:
            line= ''        
    except Exception:
        line=''

    if line:
        intro = ' %s = \033[0m' % line
    else:
        intro = ''
    
    
    if not use_print:
        log.log(level, ' '.join([intro]+[str(a) for a in args]) + \
                   ' \033[1;30m[%s at line %s]\033[0m' % (os.path.basename(filename), lineno))
    else:
        print(' '.join([intro]+[str(a) for a in args]) + \
                   ' \033[1;30m[%s at line %s]\033[0m' % (os.path.basename(filename), lineno))

    if wait:
        input('press_enter to continue')
    elif 'sleep' in opt:
        time.sleep(int(opt['sleep']))

    return 

class misc(object):
    @staticmethod
    def sprint(*args, **opt):
        return sprint(*args, **opt)
################################################################################
# function to check if two float are approximatively equal
################################################################################
def equal(a,b,sig_fig=6, zero_limit=True):
    """function to check if two float are approximatively equal"""
    import math

    if isinstance(sig_fig, int):
        if not a or not b:
            if zero_limit:
                if zero_limit is not True:
                    power = zero_limit
                else:
                    power = sig_fig + 1
            else:
                return a == b  
        else:
            power = sig_fig - int(math.log10(abs(a)))

        return ( a==b or abs(int(a*10**power) - int(b*10**power)) < 10)
    else:
        return abs(a-b) < sig_fig

################################################################################
# class to change directory with the "with statement"
# Exemple:
# with chdir(path) as path:
#     pass
################################################################################
class chdir:
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

################################################################################
# Timeout FUNCTION
################################################################################

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''This function will spwan a thread and run the given function using the args, kwargs and 
    return the given default value if the timeout_duration is exceeded 
    ''' 
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except Exception as error:
                print(error)
                self.result = default
    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    return it.result

def mmin(iter, default=None):
    
    if six.PY3:
        return min(iter, default=default)
    else:
        return min(iter, default)


################################################################################
# TAIL FUNCTION
################################################################################
class digest:

    def test_all(self):
        try:
            return self.test_hashlib()
        except Exception:
            pass
        try:
            return self.test_md5()
        except Exception:
            pass
        try:
            return self.test_zlib()
        except Exception:
            pass
                
    def test_hashlib(self):
        import hashlib
        def digest(text):
            """using mg5 for the hash"""
            t = hashlib.md5()
            t.update(text)
            return t.hexdigest()
        return digest
    
    def test_md5(self):
        import md5
        def digest(text):
            """using mg5 for the hash"""
            t = md5.md5()
            t.update(text)
            return t.hexdigest()
        return digest
    
    def test_zlib(self):
        import zlib
        def digest(text):
            return zlib.adler32(text)
    
digest = digest().test_all()

#===============================================================================
# Helper class for timing and RAM flashing of subprocesses.
#===============================================================================
class ProcessTimer:
  def __init__(self,*args,**opts):
    self.cmd_args = args
    self.cmd_opts = opts
    self.execution_state = False

  def execute(self):
    self.max_vms_memory = 0
    self.max_rss_memory = 0

    self.t1 = None
    self.t0 = time.time()
    self.p = subprocess.Popen(*self.cmd_args,**self.cmd_opts)
    self.execution_state = True

  def poll(self):
    if not self.check_execution_state():
      return False

    self.t1 = time.time()
    # I redirect stderr to void, because from MacOX snow leopard onward, this
    # ps -p command writes a million times the following stupid warning
    # dyld: DYLD_ environment variables being ignored because main executable (/bin/ps) is setuid or setgid
    flash = subprocess.Popen("ps -p %i -o rss"%self.p.pid,
                  shell=True,stdout=subprocess.PIPE,stderr=open(os.devnull,"w"))
    stdout_list = flash.communicate()[0].decode(errors='ignore').split('\n')
    rss_memory = int(stdout_list[1])
    # for now we ignore vms
    vms_memory = 0

    # This is the neat version using psutil
#    try:
#      pp = psutil.Process(self.p.pid)
#
#      # obtain a list of the subprocess and all its descendants
#      descendants = list(pp.get_children(recursive=True))
#      descendants = descendants + [pp]
#
#      rss_memory = 0
#      vms_memory = 0
#
#      # calculate and sum up the memory of the subprocess and all its descendants 
#      for descendant in descendants:
#        try:
#          mem_info = descendant.get_memory_info()
#
#          rss_memory += mem_info[0]
#          vms_memory += mem_info[1]
#        except psutil.error.NoSuchProcess:
#          # sometimes a subprocess descendant will have terminated between the time
#          # we obtain a list of descendants, and the time we actually poll this
#          # descendant's memory usage.
#          pass
#
#    except psutil.error.NoSuchProcess:
#      return self.check_execution_state()

    self.max_vms_memory = max(self.max_vms_memory,vms_memory)
    self.max_rss_memory = max(self.max_rss_memory,rss_memory)

    return self.check_execution_state()

  def is_running(self):
    # Version with psutil
#    return psutil.pid_exists(self.p.pid) and self.p.poll() == None
    return self.p.poll() == None

  def check_execution_state(self):
    if not self.execution_state:
      return False
    if self.is_running():
      return True
    self.executation_state = False
    self.t1 = time.time()
    return False

  def close(self,kill=False):

    if self.p.poll() == None:
        if kill:
            self.p.kill()
        else:
            self.p.terminate()

    # Again a neater handling with psutil
#    try:
#      pp = psutil.Process(self.p.pid)
#      if kill:
#        pp.kill()
#      else:
#        pp.terminate()
#    except psutil.error.NoSuchProcess:
#      pass

## Define system_notify (in a way which is system independent
class Notification(object):

    def __init__(self):
        self.init = False
        self.working = True

    def load_notification(self):
        self.init = True
        self.working = False
        if sys.platform == 'darwin':
            try:
                import Foundation
                import objc
                self.NSUserNotification = objc.lookUpClass('NSUserNotification')
                self.NSUserNotificationCenter = objc.lookUpClass('NSUserNotificationCenter')
                self.working = "Foundation"
            except:
                if which('osascript'):
                    self.working = 'osascript'
                return
        elif sys.platform == 'linux':
            if which('notify-send'):
                self.working = 'notify-send'
            

    def __call__(self,subtitle, info_text, userInfo={}):
        
        if not self.init:
            self.load_notification()
        if not self.working:
            return
        elif self.working == "Foundation":
            try:
                notification = self.NSUserNotification.alloc().init()
                notification.setTitle_('MadGraph5_aMC@NLO')
                notification.setSubtitle_(subtitle)
                notification.setInformativeText_(info_text)
                try:
                    notification.setUserInfo_(userInfo)
                except:
                    pass
                self.NSUserNotificationCenter.defaultUserNotificationCenter().scheduleNotification_(notification)
            except:
                pass
            
        elif self.working=='osascript':
            try:
                os.system("""
              osascript -e 'display notification "{}" with title "MadGraph5_aMC@NLO" subtitle "{}"'
              """.format(info_text, subtitle))
            except:
                pass

        elif self.working == 'notify-send':
            try:
                os.system(""" notify-send "MadGraph5_aMC@NLO" "{}"  &> /dev/null """.format(info_text,subtitle))
            except:
                pass



system_notify = Notification()

class EasterEgg(object):
    
    done_notification = False
    message_aprilfirst =\
        {'error': ['Be careful, a cat is eating a lot of fish today. This makes the code unstable.',
                   'Really, this sounds fishy.',
                   'A Higgs boson walks into a church. The priest says "We don\'t allow Higgs bosons in here." The Higgs boson replies, "But without me, how can you have mass?"',
                   "Why does Heisenberg detest driving cars? Because, every time he looks at the speedometer he gets lost!",
                   "May the mass times acceleration be with you.",
                   "NOTE: This product may actually be nine-dimensional. If this is the case, functionality is not affected by the extra five dimensions.",
                   "IMPORTANT: This product is composed of 100%% matter: It is the responsibility of the User to make sure that it does not come in contact with antimatter.",
                   "",
                   'The fish are out of jokes. See you next year for more!'],
#         'loading': ['Hi %(user)s, You are Loading Madgraph. Please be patient, we are doing the work.'],
#         'quit': ['Thanks %(user)s for using MadGraph5_aMC@NLO, even on April 1st!']
               }

    default_banner_1 =  "************************************************************\n" + \
        "*                                                          *\n" + \
        "*                     W E L C O M E to                     *\n" + \
        "*              M A D G R A P H 5 _ a M C @ N L O           *\n" + \
        "*                                                          *\n" + \
        "*                                                          *\n" 


    default_banner_2 =        "*                                                          *\n" + \
        "%s" + \
        "*                                                          *\n" + \
        "*    The MadGraph5_aMC@NLO Development Team - Find us at   *\n" + \
        "*              http://madgraph.phys.ucl.ac.be/             *\n" + \
        "*                            and                           *\n" + \
        "*            http://amcatnlo.web.cern.ch/amcatnlo/         *\n" + \
        "*                                                          *\n" + \
        "*               Type 'help' for in-line help.              *\n" + \
        "*           Type 'tutorial' to learn how MG5 works         *\n" + \
        "*    Type 'tutorial aMCatNLO' to learn how aMC@NLO works   *\n" + \
        "*    Type 'tutorial MadLoop' to learn how MadLoop works    *\n" + \
        "*                                                          *\n" + \
        "************************************************************"

    May4_banner = "*                           _____                          *\n" + \
        "*                       ,-~\"     \"~-.                      *\n" + \
        "*        *            ,^ ___         ^.             *      *\n" + \
        "*          *         / .^   ^.         \         *         *\n" + \
        "*            *      Y  l  o  !          Y      *           *\n" + \
        "*              *   l_  `.___.'         _,[   *             *\n" + \
        "*                * |^~\"--------------~\"\"^| *               *\n" + \
        "*              *   !     May the 4th     !   *             *\n" + \
        "*            *       \                 /       *           *\n" + \
        "*          *          ^.             .^          *         *\n" + \
        "*        *              \"-.._____.,-\"              *       *\n"

    Zcommezorglub =  "* M:::::::::M         M:::::::::M                          *\n" + \
        "* M:::::::::M         M:::::::::M                          *\n" + \
        "* M::::::::::M       M::::::::::M                          *\n" + \
        "* M:::::::::::M     M:::::::::::M   (_)___                 *\n" + \
        "* M:::::::M::::M   M::::M:::::::M   | / __|                *\n" + \
        "* M::::::M M::::M M::::M M::::::M   | \__ \                *\n" + \
        "* M::::::M  M::::M::::M  M::::::M   |_|___/                *\n" + \
        "* M::::::M   M:::::::M   M::::::M                          *\n" + \
        "* M::::::M    M:::::M    M::::::M    / _| ___  _ __        *\n" + \
        "* M::::::M     MMMMM     M::::::M   | |_ / _ \| '__|       *\n" + \
        "* M::::::M               M::::::M   |  _| (_) | |          *\n" + \
        "* M::::::M               M::::::M   |_/\/\___/|_|          *\n" + \
        "* M::::::M               M::::::M                          *\n" + \
        "* MMMMMMMM               MMMMMMMM                          *\n" + \
        "*                                                          *\n" + \
        "*     https://en.wikipedia.org/wiki/Z_comme_Zorglub        *\n"    





    special_banner = {(4,5): May4_banner, (14,10): Zcommezorglub}

    
    def __init__(self, msgtype):

        try:
            now = time.localtime()
            date = now.tm_mday, now.tm_mon 
            if date in [(1,4)]:
                madgraph.iolibs.drawing_eps.EpsDiagramDrawer.april_fool = True
                if msgtype in EasterEgg.message_aprilfirst:
                    choices = EasterEgg.message_aprilfirst[msgtype]
                    if len(choices) == 0:
                        return
                    elif len(choices) == 1:
                        msg = choices[0]
                    else:
                        import random
                        msg = choices[random.randint(0,len(choices)-2)]
                    EasterEgg.message_aprilfirst[msgtype].remove(msg)
                else:
                    return
            elif msgtype=='loading' and date in self.special_banner:
                self.change_banner(date)
                return
            else:
                return
            if MADEVENT:
                return
            
            import os
            import pwd
            username =pwd.getpwuid( os.getuid() )[ 0 ] 
            msg = msg % {'user': username}
            if sys.platform == "darwin":
                self.call_apple(msg)
            else:
                self.call_linux(msg)
        except Exception as error:
            sprint(error)
            pass
    
    def __call__(self, msg):
        try:
            self.call_apple(msg)
        except:
            pass
            
    def change_banner(self, date):
        if MADEVENT:
            return
        import madgraph.interface.madgraph_interface as madgraph_interface
        if date == (14,10):
            def flip(text):
                new_text = []
                for line in text.split('\n'):
                    new_line = []
                    for word in line.split(' '):
                        if "%" in word:
                            new_line.append(word)
                            continue
                        new_line.append(''.join(list(reversed(list(word)))))
                    new_text.append(' '.join(new_line))
                return '\n'.join(new_text)

            madgraph_interface.CmdExtended.intro_banner= flip(self.default_banner_1) + \
                                                        self.special_banner[date] + \
                                                        flip(self.default_banner_2) #+ \
                                                        #"\n* homage to Franquin (text above in zorgland)              *" +\
                                                        #"\n************************************************************"
        else:
            madgraph_interface.CmdExtended.intro_banner= self.default_banner_1 + self.special_banner[date] + self.default_banner_2
        

    def call_apple(self, msg):
        
        #1. control if the volume is on or not
        p = subprocess.Popen("osascript -e 'get volume settings'", stdout=subprocess.PIPE, shell=True)
        output, _  = p.communicate()
        output = output.decode(errors='ignore')
        #output volume:25, input volume:71, alert volume:100, output muted:true
        info = dict([[a.strip() for a in l.split(':',1)] for l in output.strip().split(',')])
        muted = False
        if 'output muted' in info and info['output muted'] == 'true':
            muted = True
        elif 'output volume' in info and info['output volume'] == '0':
            muted = True
        
        if muted:
            if not EasterEgg.done_notification:
                system_notify('On April first','turn up your volume!')
                EasterEgg.done_notification = True
        else:
            os.system('say %s' % msg)


    def call_linux(self, msg):
        # check for fishing path
        fishPath = madgraph.MG5DIR+"/input/.cowgraph.cow"
        if os.path.exists(fishPath):
            fishPath = " -f " + fishPath
            #sprint("got fishPath: ",fishPath)

        # check for fishing pole
        fishPole = which('cowthink')
        if not os.path.exists(fishPole):
            if os.path.exists(which('cowsay')):
                fishPole = which('cowsay')
            else:
                return

        # go fishing
        fishCmd = fishPole + fishPath + " " + msg
        os.system(fishCmd)


def get_older_version(v1, v2):
    """ return v2  if v1>v2
        return v1 if v1<v2
        return v1 if v1=v2 
        return v1 if v2 is not in 1.2.3.4.5 format
        return v2 if v1 is not in 1.2.3.4.5 format
    """
    
    for a1, a2 in zip_longest(v1, v2, fillvalue=0):
        try:
            a1= int(a1)
        except:
            return v2
        try:
            a2= int(a2)
        except:
            return v1        
        if a1 > a2:
            return v2
        elif a1 < a2:
            return v1
    return v1

    

plugin_support = {}
def is_plugin_supported(obj):
    global plugin_support
    
    name = obj.__name__
    if name in plugin_support:
        return plugin_support[name]
    
    # get MG5 version
    if '__mg5amcnlo__' in plugin_support:
        mg5_ver = plugin_support['__mg5amcnlo__']
    else:
        info = get_pkg_info()
        mg5_ver = info['version'].split('.')
    try:
        min_ver = obj.minimal_mg5amcnlo_version
        max_ver = obj.maximal_mg5amcnlo_version
        val_ver = obj.latest_validated_version
    except:
        logger.error("Plugin %s misses some required info to be valid. It is therefore discarded" % name)
        plugin_support[name] = False
        return
    
    if get_older_version(min_ver, mg5_ver) == min_ver and \
       get_older_version(mg5_ver, max_ver) == mg5_ver:
        plugin_support[name] = True
        if get_older_version(mg5_ver, val_ver) == val_ver:
            logger.warning("""Plugin %s has marked as NOT being validated with this version. 
It has been validated for the last time with version: %s""",
                                        name, '.'.join(str(i) for i in val_ver))
    else:
        if __debug__:
            logger.error("Plugin %s seems not supported by this version of MG5aMC. Keep it active (please update status)" % name)
            plugin_support[name] = True            
        else:
            logger.error("Plugin %s is not supported by this version of MG5aMC." % name)
            plugin_support[name] = False
    return plugin_support[name]
    

#decorator
def set_global(loop=False, unitary=True, mp=False, cms=False):
    from functools import wraps
    import aloha
    import aloha.aloha_lib as aloha_lib
    def deco_set(f):
        @wraps(f)
        def deco_f_set(*args, **opt):
            old_loop = aloha.loop_mode
            old_gauge = aloha.unitary_gauge
            old_mp = aloha.mp_precision
            old_cms = aloha.complex_mass
            aloha.loop_mode = loop
            aloha.unitary_gauge = unitary
            aloha.mp_precision = mp
            aloha.complex_mass = cms
            aloha_lib.KERNEL.clean()
            try:
                out =  f(*args, **opt)
            except:
                aloha.loop_mode = old_loop
                aloha.unitary_gauge = old_gauge
                aloha.mp_precision = old_mp
                aloha.complex_mass = old_cms
                raise
            aloha.loop_mode = old_loop
            aloha.unitary_gauge = old_gauge
            aloha.mp_precision = old_mp
            aloha.complex_mass = old_cms
            aloha_lib.KERNEL.clean()
            return out
        return deco_f_set
    return deco_set
   
    
    

def plugin_import(module, error_msg, fcts=[]):
    """convenient way to import a plugin file/function"""
    
    if six.PY2:
        level = -1
    else:
        level = 0

    try:
        _temp = __import__('PLUGIN.%s' % module, globals(), locals(), fcts, level)
    except ImportError:
        try:
            _temp = __import__('MG5aMC_PLUGIN.%s' % module, globals(), locals(), fcts, level)
        except ImportError:
            raise MadGraph5Error(error_msg)
    
    if not fcts:
        return _temp
    elif len(fcts) == 1:
        return getattr(_temp,fcts[0])
    else:
        return [getattr(_temp,name) for name in fcts]

def from_plugin_import(plugin_path, target_type, keyname=None, warning=False,
                       info=None):
    """return the class associated with keyname for a given plugin class
    if keyname is None, return all the name associated"""
    
    validname = []
    for plugpath in plugin_path:
        plugindirname = os.path.basename(plugpath)
        for plug in os.listdir(plugpath):
            if os.path.exists(pjoin(plugpath, plug, '__init__.py')):
                try:
                    with stdchannel_redirected(sys.stdout, os.devnull):
                        __import__('%s.%s' % (plugindirname,plug))
                except Exception as error:
                    if warning:
                        logger.warning("error detected in plugin: %s.", plug)
                        logger.warning("%s", error)
                    continue
                plugin = sys.modules['%s.%s' % (plugindirname,plug)] 
                if hasattr(plugin, target_type):
                    if not is_plugin_supported(plugin):
                        continue
                    if keyname is None:
                        validname += list(getattr(plugin, target_type).keys())
                    else:
                        if keyname in getattr(plugin, target_type):
                            if not info:
                                logger.info('Using from plugin %s mode %s' % (plug, keyname), '$MG:BOLD')
                            else:
                                logger.info(info % {'plug': plug, 'key':keyname}, '$MG:BOLD')
                            return getattr(plugin, target_type)[keyname]
                        
    if not keyname:
        return validname
    
    
    

python_lhapdf=None
def import_python_lhapdf(lhapdfconfig):
    """load the python module of lhapdf return None if it can not be loaded"""

    #save the result to have it faster and avoid the segfault at the second try if lhapdf is not compatible
    global python_lhapdf
    if python_lhapdf:
        if python_lhapdf == -1:
            return None
        else:
            return python_lhapdf

    use_lhapdf=False
    try:
        
        lhapdf_libdir=subprocess.Popen([lhapdfconfig,'--libdir'],\
                                           stdout=subprocess.PIPE).stdout.read().decode(errors='ignore').strip()
    except:
        use_lhapdf=False
        return False
    else:
        if sys.platform != "darwin":
            if not 'LD_LIBRARY_PATH' in os.environ:
                os.environ['LD_LIBRARY_PATH'] = lhapdf_libdir
            else:
                os.environ['LD_LIBRARY_PATH'] = '%s:%s' %(lhapdf_libdir,os.environ['LD_LIBRARY_PATH'])
        
        try:
            candidates=[dirname for dirname in os.listdir(lhapdf_libdir) \
                            if os.path.isdir(os.path.join(lhapdf_libdir,dirname))]
        except OSError:
            candidates=[]
        for candidate in candidates:
            if os.path.isdir(os.path.join(lhapdf_libdir,candidate,'site-packages')):
                sys.path.insert(0,os.path.join(lhapdf_libdir,candidate,'site-packages'))
                try:
                    import lhapdf
                    use_lhapdf=True
                    break
                except ImportError:
                    sys.path.pop(0)
                    continue
    if not use_lhapdf:
        try:
            candidates=[dirname for dirname in os.listdir(lhapdf_libdir+'64') \
                            if os.path.isdir(os.path.join(lhapdf_libdir+'64',dirname))]
        except OSError:
            candidates=[]

        for candidate in candidates:
            if os.path.isdir(os.path.join(lhapdf_libdir+'64',candidate,'site-packages')):
                sys.path.insert(0,os.path.join(lhapdf_libdir+'64',candidate,'site-packages'))
                try:
                    import lhapdf
                    use_lhapdf=True
                    break
                except ImportError:
                    sys.path.pop(0)
                    continue
        if not use_lhapdf:
            try:
                import lhapdf
                use_lhapdf=True
            except ImportError:
                print('fail')
                logger.warning("Failed to access python version of LHAPDF: "\
                                   "If the python interface to LHAPDF is available on your system, try "\
                                   "adding its location to the PYTHONPATH environment variable and the"\
                                   "LHAPDF library location to LD_LIBRARY_PATH (linux) or DYLD_LIBRARY_PATH (mac os x)."\
                                   "The required LD_LIBRARY_PATH is "+ lhapdf_libdir 
                                   )
        
    if use_lhapdf:
        python_lhapdf = lhapdf
        try:
            python_lhapdf.setVerbosity(0)
        except Exception:
            pass
    else:
        python_lhapdf = None
    return python_lhapdf

def newtonmethod(f, df, x0, error=1e-10,maxiter=10000):
    """implement newton method for solving f(x)=0, df is the derivate"""
    x = x0
    iter=0
    while abs(f(x)) > error:
        iter+=1
        x = x - f(x)/df(x)
        if iter ==maxiter:
            sprint('fail to solve equation')
            raise Exception
    return x

def wget(http, path, *args, **opt):
    """a wget function for both unix and mac"""

    if sys.platform == "darwin":
        return call(['curl', '-L', http, '-o%s' % path], *args, **opt)
    else:
        return call(['wget', http, '--output-document=%s'% path], *args, **opt)


def make_unique(input, keepordering=None):
    "remove duplicate in a list "

    if keepordering is None:
        if MADEVENT:
            keepordering = False
        else:
            keepordering = madgraph.ordering
    if not keepordering:
        return list(set(input))
    else:
        return list(dict.fromkeys(input)) 

if six.PY3:
    try:
        from collections import MutableSet
    except ImportError: # this is for python3.10
        from collections.abc import  MutableSet
    
    class OrderedSet(collections.OrderedDict, MutableSet):

        def __init__(self, arg=None):
            super( OrderedSet, self).__init__()
            if arg:
                self.update(arg)

        def update(self, *args, **kwargs):
            if kwargs:
                raise TypeError("update() takes no keyword arguments")

            for s in args:
                for e in s:
                    self.add(e)

        def add(self, elem):
            self[elem] = None

        def discard(self, elem):
            self.pop(elem, None)

        def pop(self, *args):
            if args:
                return super().pop(*args)
            else:
                key = next(iter(self))
                del self[key]
                return key

        def __le__(self, other):
            return all(e in other for e in self)

        def __lt__(self, other):
            return self <= other and self != other

        def __ge__(self, other):
            return all(e in self for e in other)

        def __gt__(self, other):
            return self >= other and self != other

        def __repr__(self):
            return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

        def __str__(self):
            return '{%s}' % (', '.join(map(repr, self.keys())))

        def __eq__(self, other):
            try:
                return set(self) == set(other)
            except TypeError:
                return False
        difference = property(lambda self: self.__sub__)
        difference_update = property(lambda self: self.__isub__)
        intersection = property(lambda self: self.__and__)
        intersection_update = property(lambda self: self.__iand__)
        issubset = property(lambda self: self.__le__)
        issuperset = property(lambda self: self.__ge__)
        symmetric_difference = property(lambda self: self.__xor__)
        symmetric_difference_update = property(lambda self: self.__ixor__)
        union = property(lambda self: self.__or__)
else:
    OrderedSet = set


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function (for using python2 type of sort)'

    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def smallest_diff_key(A, B):
    """return the smallest key adiff in A such that adiff not in B or A[adiff] != B[bdiff]"""
    diff_keys = [k for k in A if k not in B or A[k] != B[k]]
    return min(diff_keys)

def dict_cmp(A, B, level=1):
    if len(A) != len(B):
        return (len(A) > len(B)) - (len(A) < len(B))
    try:
        adiff = smallest_diff_key(A, B)
    except ValueError:
        # No difference.
        return 0
    bdiff = smallest_diff_key(B, A)
    if adiff != bdiff:
        a = adiff
        b = bdiff
        return (a > b) - (a < b)        
    a = A[adiff]
    b = B[bdiff]
    if isinstance(a, dict):
        return dict_cmp(a,b,level=level+1)
    else:
        return (a > b) - (a < b)
        #return cmp(A[adiff], B[bdiff])

if six.PY3:
    import io
    file = io.FileIO
        
class BackRead(file):
    """read a file returning the lines in reverse order for each call of readline()
This actually just reads blocks (4096 bytes by default) of data from the end of
the file and returns last line in an internal buffer."""


    def readline(self):
        """ readline in a backward way """

        while len(self.data) == 1 and ((self.blkcount * self.blksize) < self.size):
          self.blkcount = self.blkcount + 1
          line = self.data[0]
          try:
            self.seek(-self.blksize * self.blkcount, 2) # read from end of file
            self.data = (self.read(self.blksize).decode(errors='ignore') + line).split('\n')
          except IOError:  # can't seek before the beginning of the file
            self.seek(0)
            data = self.read(self.size - (self.blksize * (self.blkcount-1))).decode(errors='ignore') + line
            self.data = data.split('\n')

        if len(self.data) == 0:
          return ""

        line = self.data.pop()
        return line + '\n'

    def __init__(self, filepos, blksize=4096):
        """initialize the internal structures"""

        # get the file size
        self.size = os.stat(filepos)[6]
        # how big of a block to read from the file...
        self.blksize = blksize
        # how many blocks we've read
        self.blkcount = 1
        file.__init__(self, filepos, 'rb')
        # if the file is smaller than the blocksize, read a block,
        # otherwise, read the whole thing...
        if self.size > self.blksize:
          self.seek(-self.blksize * self.blkcount, 2) # read from end of file
        self.data = self.read(self.blksize).decode(errors='ignore').split('\n')
        # strip the last item if it's empty...  a byproduct of the last line having
        # a newline at the end of it
        if not self.data[-1]:
          self.data.pop()

    def next(self):
        line = self.readline()
        if line:
            return line
        else:
            raise StopIteration
def tqdm(iterator, **opts):
    return iterator

        
############################### TRACQER FOR OPEN FILE
#openfiles = set()
#oldfile = __builtin__.file
#
#class newfile(oldfile):
#    done = 0
#    def __init__(self, *args):
#        self.x = args[0]
#        if 'matplotlib' in self.x:
#            raise Exception
#        print "### OPENING %s ### %s " % (str(self.x) , time.time()-start)
#        oldfile.__init__(self, *args)
#        openfiles.add(self)
#
#    def close(self):
#        print "### CLOSING %s ### %s" % (str(self.x), time.time()-start)
#        oldfile.close(self)
#        openfiles.remove(self)
#oldopen = __builtin__.open
#def newopen(*args):
#    return newfile(*args)
#__builtin__.file = newfile
#__builtin__.open = newopen


