################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
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
import tempfile
import traceback
import gzip as ziplib
import io
from itertools import zip_longest
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
# configuration file locations
#===============================================================================
CONFIG_NAME = 'mg7_configuration.txt'
LEGACY_CONFIG_NAME = 'mg5_configuration.txt'
CONFIG_TEMPLATE_NAME = '.mg7_configuration_default.txt'

def install_config_file(root):
    """The configuration file of the MadGraph installation rooted at *root*."""

    return pjoin(root, 'input', CONFIG_NAME)

def base_config_file():
    """The $MADGRAPH_BASE configuration file, or None if that is not set.

    A base directory set up for MadGraph5_aMC@NLO holds mg5_configuration.txt
    rather than the MadGraph7 name, so fall back to it when only that one is
    there. With neither present the MadGraph7 name is returned, which is what
    the callers that create a missing file need.
    """

    base = os.environ.get('MADGRAPH_BASE')
    if not base:
        return None
    config_path = pjoin(base, CONFIG_NAME)
    if not os.path.exists(config_path):
        legacy_path = pjoin(base, LEGACY_CONFIG_NAME)
        if os.path.exists(legacy_path):
            return legacy_path
    return config_path

def user_config_dir(create=False):
    """MadGraph7's per-user configuration directory.

    MadGraph7's ~/.mg5 is deliberately never consulted: a config file shared
    between installations is what makes one of them write absolute paths into
    another one's HEPTools folder. Returns None without HOME or XDG_CONFIG_HOME.
    """

    xdg = os.environ.get('XDG_CONFIG_HOME')
    if xdg:
        path = pjoin(xdg, 'mg7')
    elif os.environ.get('HOME'):
        path = pjoin(os.environ['HOME'], '.mg7')
    else:
        return None
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            logger.warning('could not create %s: %s', path, error)
            return None
    return path

def user_config_file(create=False):
    """MadGraph7's per-user configuration file, or None if it has no home."""

    config_dir = user_config_dir(create=create)
    return pjoin(config_dir, CONFIG_NAME) if config_dir else None

def mg_root():
    """The root of this MadGraph installation, deduced from this file."""

    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#===============================================================================
# parse_info_str
#===============================================================================
def parse_info_str(fsock):
    """Parse a newline separated list of "param=value" as a dictionary
    """

    info_dict = {}
    pattern = re.compile(r"(?P<name>\w*)\s*=\s*(?P<value>.*)",
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
    path = re.sub(r'(?P<name>\?|\*|\[|\])', r'[\g<name>]', path)
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
    """Returns the current version information of the MadGraph7 package, 
    as written in the VERSION text file. If the file cannot be found, 
    a dictionary with empty values is returned. As an option, an info
    string can be passed to be read instead of the file content.
    """
    global PACKAGE_INFO

    if info_str:
        info_dict = parse_info_str(io.StringIO(info_str))
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
        def __init__(self, input):
            self.info = input.split('.')

        def __lt__(self, input):
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
        text = open(pjoin(ma5path,'bin', 'ma5'),'r').read()
        pattern = re.compile(r'\s*version\s*=\s*["\'](.*)["\']\s*')
        ma5_version = pattern.findall(text)[0]
        ma5_version=LooseVersion(ma5_version)
    except Exception as error:
        misc.sprint(error)   
        ma5_version = None
    # fallback to version.txt if the above method fails.
    # for the transition period where that file still exists (from HEPinstaller)
    if not ma5_version and os.path.exists(pjoin(ma5path,'version.txt')):
        try:
            for line in open(pjoin(ma5path,'version.txt'),'r').read().split('\n'):
                if line.startswith('MA5 version :'):
                    ma5_version=LooseVersion(line[13:].strip())
                    break
        except Exception:
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
    
    if mg5_version < LooseVersion("2.6.1") and ma5_version > LooseVersion("1.6.31"):
        reason =  "This active MadGraph7 version is too old (v%s) for your selected version of MadAnalysis5 (v%s)"%(mg5_version,ma5_version)
        reason += "\nUpgrade MadGraph7 or re-install MA5 from within MadGraph7 to fix this compatibility issue."
        reason += "\nThe specified version of MadAnalysis5 will not be active in your session."
        return reason

    if mg5_version > LooseVersion("2.6.0") and ma5_version < LooseVersion("1.6.32"):
        reason = "Your selected version of MadAnalysis5 (v%s) is too old for this active version of MadGraph7 (v%s)."%(ma5_version,mg5_version)
        reason += "\nRe-install MA5 from within MadGraph7 to fix this compatibility issue."
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

def default_f2py():
    """The f2py to use when none is configured (f2py_compiler): the one
    installed with this Python (next to sys.executable, e.g. in a virtual
    environment), else f2pyX.Y, f2pyX or f2py on the PATH -- which may
    belong to another Python, whose modules this one cannot import. None
    when there is none."""
    for name in ('f2py%d.%d' % sys.version_info[:2], 'f2py%d' % sys.version_info[0], 'f2py'):
        path = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    for name in ('f2py%d.%d' % sys.version_info[:2], 'f2py%d' % sys.version_info[0], 'f2py'):
        if which(name):
            return name
    return None

def has_f2py():
    return default_f2py() is not None
        
#===============================================================================
#  Activate dependencies if possible. Mainly for tests
#===============================================================================

def deactivate_dependence(dependency, cmd=None, log = None):
    """ Make sure to turn off some dependency of MadGraph7. """
    
    def tell(msg):
        if log == 'stdout':
            print(msg)
        elif callable(log):
            log(msg)
    

    if dependency in ['pjfry','golem','samurai','ninja','collier']:
        if dependency not in cmd.options:
            return
        if cmd.options[dependency] not in ['None',None,'']:
            tell("Deactivating MadGraph7 dependency '%s'"%dependency)
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
        # the option points to the library directory itself (./HEPTools/lib),
        # but some installations keep ninja in its own subdirectory, so look
        # for libninja.a both directly there and one 'lib' level below.
        if cmd.options['ninja'] in ['None',None,''] or\
         (cmd.options['ninja'] == './HEPTools/lib' and not MG5dir is None and\
         all(which_lib(pjoin(MG5dir,cmd.options['ninja'],subdir,'libninja.a'))\
                                is None for subdir in ['lib',''])):
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
           ["MG5_LIBRARY_PATH", "DYLD_LIBRARY_PATH","LD_LIBRARY_PATH","LIBRARY_PATH","PATH"] 
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
                # Re-raise the original error, traceback included. A bare
                # `raise` here is *outside* the except block, so it does not
                # re-raise my_error: it fails with "RuntimeError: No active
                # exception to reraise" and hides the real cause (which, after
                # nb_try attempts and their sleeps, is the only thing that
                # explains what went wrong).
                raise my_error
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
    raise Exception('this copytree is overwritten by another function')
    if 'copy_function' not in opts:
        opts['copy_function'] = shutil.copy
    return shutil.copytree(*args, **opts)

#===============================================================================
# Atomic file replacement
#===============================================================================
def atomic_write(path, content):
    """Write ``content`` to ``path`` so that any concurrent reader sees either
    the complete old file or the complete new one, but never a truncated one.

    ``open(path, 'w')`` (and therefore ``shutil.copy``) truncates the
    destination before the first byte is written, so a reader that opens the
    file in that window gets a short read.  For the files this is used on --
    ``Source/make_opts`` above all, which is shared by every MG5 process on the
    machine and re-written by each of them -- that failure is silent: a
    truncated make_opts still *parses*, so ``make`` falls back to its builtins
    ($(FC)=f77, no $(libext), no -ffixed-line-length-132) and the build dies
    much later with unrelated column-72 Fortran errors.

    The temporary file is created in the same directory as the destination:
    os.replace is only atomic within one filesystem.
    """
    path = os.path.abspath(path)
    dirname = os.path.dirname(path)
    binary = isinstance(content, bytes)
    fd, tmp = tempfile.mkstemp(dir=dirname,
                               prefix='.%s.' % os.path.basename(path),
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb' if binary else 'w') as fsock:
            fsock.write(content)
            fsock.flush()
            os.fsync(fsock.fileno())
        # mkstemp creates the file 0600; make_opts and friends have to stay
        # readable by whoever else uses the install.
        if os.path.exists(path):
            shutil.copymode(path, tmp)
        else:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_copy(src, dst):
    """``shutil.copy(src, dst)``, but the destination is replaced atomically.

    Like shutil.copy, ``dst`` may be a directory, in which case the basename of
    ``src`` is used.  See :func:`atomic_write` for why this matters.  The source
    is read into memory, so this is for configuration-sized files.
    """
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    with open(src, 'rb') as fsock:
        content = fsock.read()
    atomic_write(dst, content)
    return dst

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
             ' your version of gfortran is older than 4.6. MadGraph7 will carry on,'+\
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
        comp_re = re.compile(r'^(\s*)FC\s*=\s*(.+)\s*$')
        var = 'FC'
    elif compiler_type == 'cpp':
        comp_re = re.compile(r'^(\s*)CXX\s*=\s*(.+)\s*$')
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

def pool_chunksize(pool, nitems):
    """chunksize to pass to pool.map / pool.map_async: ceil(nitems / (4 * nprocs)),
    as CPython computes when no chunksize is given, but from the number of
    processes the pool was created with. CPython uses len(pool._pool) instead,
    which drops to zero (ZeroDivisionError) while a maxtasksperchild pool
    replaces the workers that exited after their last task.
    """
    nprocs = max(1, pool._processes)
    return max(1, -(-nitems // (4 * nprocs)))



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
        logger.debug('no stdout/stderr redirection due to debug level')
        yield
        
        
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
        comp = re.compile("^\\s*FC\\s*=\\s*([\\w\\/\\.\\-]+)\\s*")
    elif compiler_type == 'cpp':
        comp = re.compile("^\\s*CXX\\s*=\\s*([\\w\\/\\.\\-]+)\\s*")
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


_FORMAT_PATTERN = re.compile(
    r'%\((\w+)\)([-+# 0]*\d*\.?\d*[hlL]?[diouxXeEfgGcrsa])'
)

def apply_template(template, mapping):
    """Substitute %(name)<spec> placeholders in template with values from mapping.

    Only matches valid Python %-format specifiers like %(name)s, %(name)d, or
    %(name)3.3d. Any other '%' character is left untouched, so templates can
    safely contain Fortran object member access (e.g., W(I)%P) without needing
    to be escaped as '%%P'.
    """
    def replace(match):
        name, spec = match.group(1), match.group(2)
        return ('%' + spec) % mapping[name]
    return _FORMAT_PATTERN.sub(replace, template)

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
                    # raise a more meaningful error message
                    raise Exception('%s fails with no such file or directory' \
                                                                           % arg)            
                # Checkpoint, requeued automatically
                elif error.errno == 85:
                    logger.info('%s created a checkpoint' % arg)
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
    try:
        return subprocess.Popen(arg, *args, **opt)
    except ValueError as error:
        if 'I/O operation on closed file' not in str(error):
            raise
        # In some CI/threaded contexts sys.__stdout__ can be closed while a
        # subprocess is started with stderr=STDOUT and no explicit stdout.
        # Retry with explicit /dev/null redirection in that case.
        if opt.get('stderr') == subprocess.STDOUT and opt.get('stdout') is None:
            retry_opt = dict(opt)
            retry_opt['stdout'] = subprocess.DEVNULL
            return subprocess.Popen(arg, *args, **retry_opt)
        raise

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
    try:
      shutil.copystat(src, dst)
    except PermissionError:
        if os.path.realpath(src).startswith('/cvmfs') and os.path.realpath(dst).startswith('/afs'):
           # allowing mismatch from cvmfs to afs since sounds to not create issue --at least in general--
           logger.critical(f'Ignoring that we could not copy permissions from {src} to {dst}')
        else:
           logger.critical(f'Permission error detected from {src} to {dst}.\n'+\
                          'If you are using WSL with windows partition, please try using python3.12\n'+\
                          'or avoid moving your data from the WSL partition to the UNIX one')
           # we do not have enough experience in WSL to allow it to get through.
           raise
      
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
    error in case the file format of the target drive doesn't support it."""

    try:
        os.mkfifo(fifo_path)
    except:
        raise OSError('MadGraph7 could not create a fifo file at:\n'+
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

       A dict is addressed by key instead: TMP_variable(cmd.options, 'foo', 1)
       swaps cmd.options['foo'], which is where MG5 keeps its own settings.

       Note that the new value is installed by __init__, not by __enter__, so
       this can also be driven by hand -- construct it to swap, and call
       __exit__(None, None, None) to restore -- for a scope that is not a
       single block.
    """

    def __init__(self, cls, attribute, value):

        self.cls = cls
        self.attribute = attribute
        self.is_dict = isinstance(cls, dict)
        if isinstance(attribute, list):
            self.old_value = []
            for key, onevalue in zip(attribute, value):
                self.old_value.append(self._get(key))
                self._set(key, onevalue)
        else:
            self.old_value = self._get(attribute)
            self._set(attribute, value)

    def _get(self, key):
        if self.is_dict:
            return self.cls.get(key)
        return getattr(self.cls, key)

    def _set(self, key, value):
        if self.is_dict:
            self.cls[key] = value
        else:
            setattr(self.cls, key, value)

    def __exit__(self, ctype, value, traceback ):
        
        if isinstance(self.attribute, list):
            for key, old_value in zip(self.attribute, self.old_value):
                self._set(key, old_value)
        else:
            self._set(self.attribute, self.old_value)
        
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
            raise Exception("%(path)s does not finish by .gz and the file %(path)s.gz does not exist" %\
                              {"path": path})         

    
    #for large file (>1G) it is faster and safer to use a separate thread
    if os.path.getsize(path) > 1e8:
        if keep:
            options = '-k '
        else:
            options = ''

        if stdout:
            os.system('gunzip  %s -c %s > %s' % (options, path, stdout))
        else:
            # -f: without it gunzip asks "already exists -- do you wish to
            # overwrite (y or n)?" as soon as the uncompressed file is already
            # there. Nothing answers that prompt here, so gunzip would silently
            # decompress nothing and leave a stale file behind.
            os.system('gunzip -f %s %s' % (options, path))
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

_gzip_tool = 'gzip'
_gzip_tool_supports_multithreading = False
_gzip_tool_max_cores = None

def configure_gzip(configuration=None):
    if not configuration:
        configuration = {}

    # Set some default values, in case the keys are missing from the configuration
    configuration = {'use_pigz': None, 'nb_core': None} | configuration

    global _gzip_tool, _gzip_tool_supports_multithreading, _gzip_tool_max_cores
    _gzip_tool_supports_multithreading = False
    _gzip_tool_max_cores = None

    use_pigz = configuration['use_pigz']
    if use_pigz is None:
        # Try using pigz if possible, fall back to gzip otherwise
        _gzip_tool = open_file.find_valid(['pigz', 'gzip'], 'gzip') 
    elif use_pigz:
        _gzip_tool = open_file.find_valid(['pigz'], 'pigz')
        if not _gzip_tool:
            logger.warning('Could not find valid `pigz` executable; using gzip.')
            _gzip_tool = open_file.find_valid(['gzip'], 'gzip')
    else:
        _gzip_tool = open_file.find_valid(['gzip'], 'gzip')
        if not _gzip_tool:
            logger.warning('Could not find valid `gzip` executable.')

    # If we use the pigz binary, enable multithreading support
    if _gzip_tool and 'pigz' in _gzip_tool:
        _gzip_tool_supports_multithreading = True
        if configuration['nb_core'] is not None:
            _gzip_tool_max_cores = configuration['nb_core']

# Compression level for the in-process branch of gzip() below. The gzip module
# defaults to 9, which is a poor trade here: on a 172 MB LHE, level 9 takes
# 18.6 s against 4.5 s at level 6, and buys 4% (38.1 MB against 39.7 MB). Level
# 6 is also what the external tool this function shells out to for large files
# uses, so the two branches now agree instead of compressing the same data
# differently depending on its size.
GZIP_COMPRESSLEVEL = 6


def _remove_partial_gz(path):
    """Delete the truncated archive left behind by a failed compression."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def gzip(path, stdout=None, error=True, forceexternal=False,
         compresslevel=GZIP_COMPRESSLEVEL, keep=False):
    """ a standard replacement for os.system('gzip %s ' % path)"""

    # For large files (>256M), it is faster and safer to use a separate tool.
    if _gzip_tool and (os.path.getsize(path) > 256e6 or forceexternal):
        command = [_gzip_tool]
        if _gzip_tool_supports_multithreading and _gzip_tool_max_cores is not None:
            command += ['-p', str(_gzip_tool_max_cores)]
        if keep:
            command.append('-k')
        if call(command + ['-f', path]) != 0:
            _remove_partial_gz('%s.gz' % path)
            if error:
                raise RuntimeError('compressing %s with %s failed' % (path, _gzip_tool))
            return

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
        # Stream it: reading the whole file in as a str and encoding it made a
        # 172 MB LHE cost two extra full-size copies in memory.
        with open(path, 'rb') as fsock, \
             ziplib.open(stdout, 'wb', compresslevel=compresslevel) as f:
            shutil.copyfileobj(fsock, f, 4 * 1024 * 1024)
    except BaseException as err:
        # Only a complete archive may replace the source: a failed or
        # interrupted copy leaves no truncated .gz and keeps the source.
        _remove_partial_gz(stdout)
        if isinstance(err, OverflowError):
            if not _gzip_tool:
                raise
            return gzip(path, stdout, error=error, forceexternal=True,
                        compresslevel=compresslevel, keep=keep)
        if error or not isinstance(err, Exception):
            raise
        return
    if not keep:
        os.remove(path)
    
#
# Global function to open supported file types
#
class open_file(object):
    """ a convenient class to open a file """
    
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

    # tried in this order when neither the text_editor option nor $EDITOR
    # names one
    DEFAULT_TEXT_EDITORS = ['vi', 'emacs', 'vim', 'gedit', 'nano']

    @classmethod
    def resolve_text_editor(cls, configured=None, quiet=False):
        """The text editor a card will open in.

        The `text_editor` option if that program exists, else $EDITOR, else the
        first of DEFAULT_TEXT_EDITORS found on the machine.  With `quiet`, it
        says nothing and sets nothing -- for telling someone in advance which
        editor they are about to get.
        """

        if configured:
            if which(configured.split()[0]):
                return configured
            if not quiet:
                logger.warning('Specified text editor %s not valid.' % configured)
        if 'EDITOR' in os.environ:
            return os.environ['EDITOR']
        if quiet:
            for candidate in cls.DEFAULT_TEXT_EDITORS:
                if which(candidate):
                    return candidate
            return None
        return cls.find_valid(cls.DEFAULT_TEXT_EDITORS, 'text editor')

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
                cls.text_editor = cls.resolve_text_editor(configuration[key])
              
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
                             'Set another one in ./input/mg7_configuration.txt')
                return p
        
        logger.info('No valid %s found. ' % program + \
                                   'Please set in ./input/mg7_configuration.txt')
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
                import _thread
                _thread.start_new_thread(subprocess.call,(arguments,))
        else:
            logger.warning('Not able to open file %s since no program configured.' % file_path + \
                                'Please set one in ./input/mg7_configuration.txt')

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
    sets it before calling the executable, for example:
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

def find_pythia8_main164(pythia8_path):
    """Locate main164, the Pythia8 example program used to shower LO events.
    Returns (executable, examples_dir): executable is None if main164 is not
    compiled, examples_dir is None if main164.cc cannot be found either."""

    examples_dir = None
    for examples in (pjoin(pythia8_path, 'share', 'Pythia8', 'examples'),
                     pjoin(pythia8_path, 'examples')):
        executable = pjoin(examples, 'main164')
        if os.path.isfile(executable) and os.access(executable, os.X_OK):
            return executable, examples
        if examples_dir is None and os.path.isfile(pjoin(examples, 'main164.cc')):
            examples_dir = examples
    return None, examples_dir

def get_pythia8_merging_plugin(pythia8_path):
    """The 'Init:plugins' entry ('<library>::aMCatNLOHooks') of the merging
    hook that main164 of this Pythia8 installation loads as a plugin, as the
    main164 of Pythia8 8.318 does, or None if its main164 does not (up to
    Pythia8 8.317, main164 sets the hook itself) or the plugin is missing.
    The library is given by its absolute path, since the shower does not run
    with the Pythia8 library directory in the library search path."""

    for examples in (pjoin(pythia8_path, 'share', 'Pythia8', 'examples'),
                     pjoin(pythia8_path, 'examples')):
        try:
            source = open(pjoin(examples, 'main164.cc')).read()
        except IOError:
            continue
        # join the adjacent string literals of the C++ source
        source = re.sub(r'"\s*"', '', source)
        if 'Init:plugins += {libpythia8amcatnloHooks.so::aMCatNLOHooks}' not in source:
            return None
        library = os.path.abspath(pjoin(pythia8_path, 'lib', 'libpythia8amcatnloHooks.so'))
        if not os.path.isfile(library):
            return None
        # a path that the Pythia8 card (or MadGraph7's reading of it) would cut
        if re.search(r'[\s,{}"!=]', library):
            library = os.path.basename(library)
        return '%s::aMCatNLOHooks' % library
    return None

_cpp_token = re.compile(r'"(?:\\.|[^"\\])*"'      # string literal
                        r"|'(?:\\.|[^'\\])*'"     # character literal
                        r'|//[^\n]*'                 # line comment
                        r'|/\*.*?\*/', re.S)         # block comment

def strip_cpp_comments(source):
    """The C/C++ source without its comments and with its whitespace
    collapsed, so that two copies of a file differing only in comments or
    layout compare equal. String and character literals are kept as is."""

    code = _cpp_token.sub(lambda m: m.group(0) if m.group(0)[0] in '"\'' else ' ',
                          source)
    return ' '.join(code.split())

def same_cpp_code(source1, source2):
    """True if the two C/C++ sources differ at most by comments and whitespace"""
    return strip_cpp_comments(source1) == strip_cpp_comments(source2)

def get_pythia8_version(pythia8_path):
    """The version of a Pythia8 installation (e.g. '8.317'), from its
    pythia8-config or its xmldoc, or None if it cannot be told."""

    try:
        out = subprocess.Popen([pjoin(pythia8_path, 'bin', 'pythia8-config'), '--version'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
        out = out.decode(errors='ignore').strip()
        float(out)
        return out
    except (OSError, ValueError):
        pass
    for xmldoc in (pjoin(pythia8_path, 'share', 'Pythia8', 'xmldoc'),
                   pjoin(pythia8_path, 'xmldoc')):
        try:
            match = re.search(r'"Pythia:versionNumber"\s+default="([\d.]+)"',
                              open(pjoin(xmldoc, 'Version.xml')).read())
        except IOError:
            continue
        if match:
            return match.group(1)
    return None

def _pythia8_makefile_inc(examples_dir):
    """The Makefile.inc holding the configuration of a Pythia8 examples
    directory: next to its Makefile (installation, or built source tree), or
    at the top of a source tree that was configured but not built. None if
    Pythia8 was not configured."""

    for makefile_inc in (pjoin(examples_dir, 'Makefile.inc'),
                         pjoin(examples_dir, os.pardir, 'Makefile.inc')):
        if os.path.isfile(makefile_inc):
            return makefile_inc
    return None

def pythia8_hepmc_version(examples_dir):
    """The HepMC version (2 or 3) main164 is compiled against with the
    configuration of this Pythia8 examples directory (main164.cc takes HepMC3
    when both are enabled), None if neither is."""

    use = {}
    makefile_inc = _pythia8_makefile_inc(examples_dir)
    if makefile_inc:
        for line in open(makefile_inc):
            match = re.match(r'\s*HEPMC([23])_USE\s*=\s*(\S+)', line)
            if match:
                use[int(match.group(1))] = match.group(2).lower() == 'true'
    for version in (3, 2):
        if use.get(version):
            return version
    return None

def find_hepmc(version, candidates=()):
    """Return (prefix, libdir) of a HepMC<version> installation (version 2 or
    3), trying the candidate prefixes and then HepMC3-config, or None."""

    if version == 3:
        header, library, config = pjoin('HepMC3', 'GenEvent.h'), 'libHepMC3', 'HepMC3-config'
    else:
        header, library, config = pjoin('HepMC', 'GenEvent.h'), 'libHepMC', 'HepMC-config'
    candidates = list(candidates)
    if which(config):
        candidates.append(os.path.dirname(os.path.dirname(os.path.realpath(which(config)))))
    for prefix in candidates:
        if not prefix or not os.path.isfile(pjoin(prefix, 'include', header)):
            continue
        for libdir in ('lib', 'lib64'):
            if glob('%s.*' % library, pjoin(prefix, libdir)):
                return os.path.realpath(prefix), os.path.realpath(pjoin(prefix, libdir))
    return None

def get_pythia8_hepmc_flags(pythia8_path, hepmc_version, hepmc_paths=()):
    """Compiler flags (-I, -L, -rpath, -l) linking a program that uses Pythia8
    to HepMC<hepmc_version> (2 or 3): the HepMC setup of Pythia8 if it has
    that version, else the installation found in hepmc_paths or next to Pythia8."""

    config = pjoin(pythia8_path, 'bin', 'pythia8-config')
    if os.path.isfile(config):
        try:
            out = subprocess.Popen([config, '--hepmc%d' % hepmc_version],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
            out = out.decode(errors='ignore').strip()
        except OSError:
            out = ''
        if out:
            return out
    default = 'hepmc3' if hepmc_version == 3 else 'hepmc'
    hepmc = find_hepmc(hepmc_version, list(hepmc_paths) + [pjoin(pythia8_path, os.pardir, default)])
    if not hepmc:
        raise MadGraph5Error('No HepMC%d installation was found (it can be ' % hepmc_version +
                             "installed with 'install %s')." % default)
    prefix, libdir = hepmc
    return '-I%s -L%s -Wl,-rpath,%s -l%s' % (pjoin(prefix, 'include'), libdir, libdir,
                                             'HepMC3' if hepmc_version == 3 else 'HepMC')

# Version of the per-jet-sample DJR patch of main164 below. It is part of the
# build stamp, so that a change to the patch rebuilds the patched main164.
MAIN164_DJR_PATCH_VERSION = 2

def pythia8_main164_writes_jet_sample_djrs(source):
    """Whether a main164.cc can histogram the differential jet rates per jet
    sample by itself (its InternalAnalysis:jetSamples setting)."""

    return '"InternalAnalysis:jetSamples"' in source

def pythia8_main164_knows_jet_samples(executable):
    """Whether the main164 executable was built from a main164.cc that knows
    InternalAnalysis:jetSamples (a Pythia8 setting it would refuse otherwise).
    The source sits next to the executable, as get_pythia8_main164 builds it."""

    try:
        source = open(pjoin(os.path.dirname(executable), 'main164.cc')).read()
    except (IOError, OSError):
        return False
    return pythia8_main164_writes_jet_sample_djrs(source)

def patch_pythia8_main164_djrs(source):
    """The main164.cc of Pythia8 with the change proposed to Pythia8: the
    setting InternalAnalysis:jetSamples (default off) also writes the
    differential jet rate histograms of its internal analysis per jet sample,
    in <jetsample njet="n"> blocks after the totals of each <run>, and the
    '<run id="0"">' of Pythia8 <= 8.318 is fixed. The jet sample is the
    number of matrix-element partons of MLM matching (nMEpartons().first,
    getProcessSubset().size() for shower-kT, npNLO() for FxFx), or
    Merging:nRequested less Merging:nRecluster for merging.

    Until a Pythia8 release has it, MadGraph7 builds this patched copy for
    its MLM showers; Pythia8's own main164.cc is never modified. Returns None
    when the source is not the main164.cc this was written against (Pythia8
    8.316 to 8.318): the stock main164 is then used, and the DJR plots show
    the summed histograms only."""

    edits = [
        ('  pythia.settings.addWord("InternalAnalysis:output", "analysis.dat");\n',
         '  // Optionally also write the histograms of each jet sample separately.\n'
         '  pythia.settings.addFlag("InternalAnalysis:jetSamples", false);\n'),
        ("  vector<Hist> d01Hists, d12Hists, d23Hists, d34Hists;\n",
         "  // Optionally also for each jet sample, as known by merging or by\n"
         "  // MadGraph-style (not Alpgen-style) jet matching.\n"
         '  bool doJetSamples = pythia.flag("InternalAnalysis:jetSamples");\n'
         "  shared_ptr<JetMatchingMadgraph> jetMatchingPtr\n"
         "    = dynamic_pointer_cast<JetMatchingMadgraph>(jetMatchingHook.hook);\n"
         "  vector<map<int, vector<Hist> > > jetSampleHists;\n"),
        ('      d34Hists.push_back(Hist("d34", 100., 0., 3.));\n',
         "      jetSampleHists.push_back(map<int, vector<Hist> >());\n"),
        ("        if (dij.size() > 3) d34Hists.back().fill(log10(dij[3]), w);\n",
         "        // Optionally also for the jet sample of the event: for jet matching\n"
         "        // the number of matrix-element partons (only those to be matched\n"
         "        // for shower-kT, those of the Born for FxFx), for merging the\n"
         "        // requested number of jets less the reclustered ones.\n"
         "        int nJetSample = -1;\n"
         "        if (doJetSamples && jetMatchingPtr) {\n"
         '          if (pythia.flag("JetMatching:doFxFx"))\n'
         "            nJetSample = jetMatchingPtr->npNLO();\n"
         '          else if (pythia.flag("JetMatching:doShowerKt"))\n'
         "            nJetSample = jetMatchingPtr->getProcessSubset().size();\n"
         "          else\n"
         "            nJetSample = jetMatchingPtr->nMEpartons().first;\n"
         "        } else if (doJetSamples && doMerging) {\n"
         '          nJetSample = pythia.mode("Merging:nRequested")\n'
         '            - pythia.mode("Merging:nRecluster");\n'
         "        }\n"
         "        if (nJetSample >= 0) {\n"
         "          vector<Hist>& sampleHists = jetSampleHists.back()[nJetSample];\n"
         "          if (sampleHists.empty())\n"
         '            sampleHists = { Hist("d01", 100., 0., 3.),\n'
         '              Hist("d12", 100., 0., 3.), Hist("d23", 100., 0., 3.),\n'
         '              Hist("d34", 100., 0., 3.) };\n'
         "          for (int iHist = 0; iHist < 4 && iHist < int(dij.size()); ++iHist)\n"
         "            sampleHists[iHist].fill(log10(dij[iHist]), w);\n"
         "        }\n"),
    ]
    for anchor, addition in edits:
        if source.count(anchor) != 1:
            return None
        source = source.replace(anchor, anchor + addition)

    run_header = '      output << "<run id=\\"" << i << "\\"" << "\\">\\n";\n'
    run_end = '      output << "</run>\\n";\n'
    if source.count(run_header) != 1 or source.count(run_end) != 1:
        return None
    source = source.replace(run_header,
                            '      output << "<run id=\\"" << i << "\\">\\n";\n')
    source = source.replace(run_end,
        "      // Optionally print the histograms of each jet sample.\n"
        "      for (const auto& jetSample : jetSampleHists[i]) {\n"
        '        output << "<jetsample njet=\\"" << jetSample.first << "\\">\\n";\n'
        "        for (const Hist& hist : jetSample.second) {\n"
        '          output << "<histogram name=\\"" << "log10" << hist.getTitle() << "\\""\n'
        '                 << " unit=\\"" << "[]" << "\\""\n'
        '                 << " weight=\\"" << "all" << "\\">\\n";\n'
        "          hist.table(output, false, false);\n"
        '          output << "</histogram>\\n";\n'
        "        }\n"
        '        output << "</jetsample>\\n";\n'
        "      }\n"
        "\n" + run_end)
    return ('// Patched by MadGraph7 (version %d of the per-jet-sample DJR patch,\n'
            '// see patch_pythia8_main164_djrs in madgraph/various/misc.py).\n'
            % MAIN164_DJR_PATCH_VERSION) + source

def _pythia8_main164_djr_source(examples):
    """The patched main164.cc for per-jet-sample DJRs, or None when it is not
    needed (the installed main164 does it) or not possible (an unknown
    main164.cc), see patch_pythia8_main164_djrs."""

    try:
        source = open(pjoin(examples, 'main164.cc')).read()
    except (IOError, OSError):
        return None
    if pythia8_main164_writes_jet_sample_djrs(source):
        return None
    patched = patch_pythia8_main164_djrs(source)
    if patched is None:
        logger.info('The main164 of this Pythia8 version cannot be patched to ' +
                    'histogram the jet rates per jet sample: the DJR plots show ' +
                    'their sum only.')
    return patched

def get_pythia8_main164(pythia8_path, fallback_dir=None, hepmc_version=None,
                        hepmc_paths=(), jet_sample_djrs=False):
    """Return the path of a compiled main164, compiling it if needed.

    main164 is built in the Pythia8 examples directory with the Makefile that
    Pythia8 configured there. If that directory is read-only (e.g. a central
    installation), main164.cc and the Makefiles are copied to fallback_dir and
    main164 is built there instead.

    main164 writes its events in the format of the HepMC library it is linked
    to. If hepmc_version (2 or 3) differs from the one Pythia8 was configured
    with, a dedicated main164 is built against the HepMC installation found in
    hepmc_paths or next to Pythia8: in <examples>/main164_hepmc<version>, so
    that it is shared by all processes, or in <fallback_dir>_hepmc<version>
    if the examples directory is read-only.

    With jet_sample_djrs, main164 histograms the jet rates of an MLM shower
    per jet sample too: when the installed main164.cc does not, a patched copy
    (patch_pythia8_main164_djrs) is built in the same way, in a directory
    named with _djr appended. When the patch does not apply, the stock main164
    is returned."""

    executable, examples = find_pythia8_main164(pythia8_path)
    if not examples:
        version = get_pythia8_version(pythia8_path)
        if version and float(version) < 8.311:
            raise MadGraph5Error('The Pythia8 installation %s is version %s. ' % (pythia8_path, version) +
                'MadGraph7 showers LO events with main164, an example program of Pythia8 '+
                "that exists since Pythia 8.311: please install a more recent Pythia8 "+
                "(e.g. with 'install pythia8').")
        raise MadGraph5Error('main164.cc cannot be found in the Pythia8 installation ' +
            '%s. MadGraph7 showers LO events with main164, an example program of ' % pythia8_path +
            "Pythia8 that exists since Pythia 8.311 (it can be installed with 'install pythia8').")

    source = _pythia8_main164_djr_source(examples) if jet_sample_djrs else None

    hepmc = None
    if hepmc_version and hepmc_version != pythia8_hepmc_version(examples):
        if not os.path.isfile(pjoin(examples, 'main164.cc')):
            raise MadGraph5Error('main164.cc cannot be found in %s, so that ' % examples +
                                 'main164 cannot be compiled for HepMC%d.' % hepmc_version)
        default = 'hepmc3' if hepmc_version == 3 else 'hepmc'
        hepmc = find_hepmc(hepmc_version, list(hepmc_paths) +
                           [pjoin(pythia8_path, os.pardir, default)])
        if not hepmc:
            raise MadGraph5Error('Pythia8 cannot write HepMC%d: ' % hepmc_version +
                'no HepMC%d installation was found (it can be installed with ' % hepmc_version +
                "'install %s'), and Pythia8 was configured with another version." % default)
        hepmc = (hepmc_version,) + hepmc

    if hepmc or source:
        suffix = ('_hepmc%d' % hepmc[0] if hepmc else '') + ('_djr' if source else '')
        stamp = _pythia8_main164_stamp(pythia8_path, examples, hepmc)
        if source:
            stamp += '\ndjr patch %d' % MAIN164_DJR_PATCH_VERSION
        shared_dir = pjoin(examples, 'main164' + suffix)
        local_dir = fallback_dir + suffix if fallback_dir else None
        for build_dir in (shared_dir, local_dir):
            if build_dir and _pythia8_main164_is_built(build_dir, stamp):
                return pjoin(build_dir, 'main164')
        if os.access(examples, os.W_OK):
            build_dir = shared_dir
        elif local_dir:
            build_dir = local_dir
        else:
            raise MadGraph5Error('No writable directory to compile main164 ' +
                (('for HepMC%d.' % hepmc[0]) if hepmc else 'with per-jet-sample jet rates.'))
        return _compile_pythia8_main164(examples, build_dir, stamp,
                                        _pythia8_main164_makefile_inc(examples, hepmc),
                                        source=source)

    if executable:
        return executable
    if os.access(examples, os.W_OK):
        return _compile_pythia8_main164(examples, examples)
    if not fallback_dir:
        raise MadGraph5Error('main164 is not compiled in the read-only ' +
            'directory %s. Please compile it with "make main164" there.' % examples)
    stamp = _pythia8_main164_stamp(pythia8_path, examples)
    if _pythia8_main164_is_built(fallback_dir, stamp):
        return pjoin(fallback_dir, 'main164')
    return _compile_pythia8_main164(examples, fallback_dir, stamp,
                                    _pythia8_main164_makefile_inc(examples))

def _pythia8_lib_include(examples, makefile_inc):
    """The Pythia8 library and include directories that the examples Makefile
    uses: those of the source tree for the examples of a built source tree (the
    Makefile then takes ../lib), PREFIX_LIB and PREFIX_INCLUDE otherwise."""

    if glob('libpythia8.*', pjoin(examples, os.pardir, 'lib')):
        return (os.path.realpath(pjoin(examples, os.pardir, 'lib')),
                os.path.realpath(pjoin(examples, os.pardir, 'include')))
    prefix = {}
    for line in open(makefile_inc):
        match = re.match(r'\s*(PREFIX_LIB|PREFIX_INCLUDE)\s*=\s*(\S+)', line)
        if match:
            prefix[match.group(1)] = match.group(2)
    return prefix.get('PREFIX_LIB'), prefix.get('PREFIX_INCLUDE')

def _pythia8_main164_makefile_inc(examples, hepmc=None):
    """The Makefile.inc of a copy of the Pythia8 examples directory: the
    configuration of Pythia8, with the location of its library made explicit
    (the examples Makefile of a source tree finds it as ../lib, which a copy
    does not have) and, for hepmc = (version, prefix, libdir), its HepMC setup
    replaced by that one."""

    makefile_inc = _pythia8_makefile_inc(examples)
    if not makefile_inc:
        raise MadGraph5Error('No Makefile.inc found for the Pythia8 examples in ' +
            '%s: Pythia8 must be configured and compiled to be used.' % examples)
    lines = []
    for line in open(makefile_inc).read().splitlines():
        key = line.split('=', 1)[0].strip()
        if key in ('PREFIX_LIB', 'PREFIX_INCLUDE'):
            continue
        if hepmc and re.match(r'HEPMC[23]_(USE|INCLUDE|LIB)$', key):
            continue
        if hepmc and key == 'CXX_COMMON':
            line = ' '.join(word for word in line.split(' ')
                      if word not in ('-DHEPMC2HACK', '-DHEPMC2', '-DHEPMC3'))
        lines.append(line)
    lib, include = _pythia8_lib_include(examples, makefile_inc)
    if lib:
        lines.append('PREFIX_LIB=%s' % lib)
    if include:
        lines.append('PREFIX_INCLUDE=%s' % include)
    if hepmc:
        version, prefix, libdir = hepmc
        lines += ['HEPMC%d_USE=false' % (5 - version),
                  'HEPMC%d_USE=true' % version,
                  'HEPMC%d_INCLUDE=-I%s' % (version, pjoin(prefix, 'include')),
                  'HEPMC%d_LIB=-L%s -Wl,-rpath,%s -l%s' % (version, libdir, libdir,
                                              'HepMC3' if version == 3 else 'HepMC')]
    return '\n'.join(lines) + '\n'

def _pythia8_main164_stamp(pythia8_path, examples, hepmc=None):
    """What a copied main164 build depends on, so that it is rebuilt when
    Pythia8 or HepMC is reinstalled or upgraded in place: the path, size and
    modification time of main164.cc, Makefile.inc and the Pythia8 (and HepMC)
    libraries."""

    files = [pjoin(examples, 'main164.cc')]
    makefile_inc = _pythia8_makefile_inc(examples)
    if makefile_inc:
        files.append(makefile_inc)
        lib = _pythia8_lib_include(examples, makefile_inc)[0]
        if lib:
            files += sorted(glob('libpythia8.*', lib))
    if hepmc:
        version, prefix, libdir = hepmc
        files += sorted(glob('libHepMC%s.*' % ('3' if version == 3 else ''), libdir))
    lines = [os.path.realpath(pythia8_path)]
    for path in files:
        try:
            info = os.stat(path)
            lines.append('%s %d %d' % (os.path.realpath(path), info.st_size, int(info.st_mtime)))
        except OSError:
            lines.append('%s missing' % path)
    return '\n'.join(lines)

def _pythia8_main164_is_built(build_dir, stamp):
    """Whether build_dir holds a main164 built with this stamp."""

    stamp_file = pjoin(build_dir, 'BUILD_STAMP')
    return os.path.isfile(pjoin(build_dir, 'main164')) and \
           os.path.isfile(stamp_file) and open(stamp_file).read() == stamp

def _compile_pythia8_main164(examples, build_dir, stamp=None, makefile_inc=None,
                             source=None):
    """Compile main164 in build_dir and return its path. If build_dir is not the
    Pythia8 examples directory, main164.cc (or the given source in its place)
    and the Makefile are copied there, with makefile_inc as Makefile.inc, and
    the build is stamped."""

    executable = pjoin(build_dir, 'main164')
    stamp_file = pjoin(build_dir, 'BUILD_STAMP')
    if build_dir != examples:
        if os.path.exists(executable):
            # stale build: make would consider it up to date
            os.remove(executable)
        elif not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        for name in ('main164.cc', 'Makefile'):
            shutil.copy(pjoin(examples, name), build_dir)
        if source is not None:
            with open(pjoin(build_dir, 'main164.cc'), 'w') as fsock:
                fsock.write(source)
        with open(pjoin(build_dir, 'Makefile.inc'), 'w') as fsock:
            fsock.write(makefile_inc)

    # HEPToolsInstaller adds a 'mainMG' rule building main164 with Rivet support
    target = 'main164'
    if re.search(r'^mainMG\s*:', open(pjoin(build_dir, 'Makefile')).read(), re.M):
        target = 'mainMG'
    logger.info('Compiling main164 from Pythia8 in %s' % build_dir)
    compile([target], cwd=build_dir, mode='cpp')
    if not os.path.isfile(executable):
        raise MadGraph5Error('Compilation of main164 in %s did not produce ' % build_dir +
                             'an executable, so that Pythia8 cannot be used.')
    if build_dir != examples:
        with open(stamp_file, 'w') as fsock:
            fsock.write(stamp)
    return executable

def hepmc_file_version(path):
    """The HepMC version (2 or 3) of a HepMC ASCII file (possibly gzipped),
    read from its header; None if it cannot be told (e.g. for a fifo)."""

    if not os.path.isfile(path):
        return None
    try:
        with (ziplib.open(path, 'rt') if path.endswith('.gz') else open(path)) as fsock:
            for _, line in zip(range(10), fsock):
                if line.startswith('HepMC::Asciiv3') or line.startswith('HepMC::Version 3'):
                    return 3
                if line.startswith('HepMC::IO_GenEvent') or line.startswith('HepMC::Version 2'):
                    return 2
    except (IOError, OSError, UnicodeDecodeError):
        pass
    return None

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

    no_color = bool(os.environ.get('MG7_NO_COLOR'))

    if line:
        intro = ' %s = ' % line if no_color else ' %s = \033[0m' % line
    else:
        intro = ''

    if no_color:
        suffix = ' [%s at line %s]' % (os.path.basename(filename), lineno)
    else:
        suffix = ' \033[1;30m[%s at line %s]\033[0m' % (os.path.basename(filename), lineno)

    if not use_print:
        log.log(level, ' '.join([intro]+[str(a) for a in args]) + suffix)
    else:
        print(' '.join([intro]+[str(a) for a in args]) + suffix)

    if wait:
        input('press_enter to continue')
    elif 'sleep' in opt:
        time.sleep(int(opt['sleep']))

    return 

class misc(object):
    @staticmethod
    def sprint(*args, **opt):
        return sprint(*args, **opt)
spirnt = sprint
################################################################################
# function to check if two float are approximately equal
################################################################################
def equal(a,b,sig_fig=6, zero_limit=True):
    """function to check if two float are approximately equal"""
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
# Example:
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
    '''This function will spawn a thread and run the given function using the args, kwargs and
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
    
    return min(iter, default=default)


################################################################################
# MODEL FINGERPRINT
################################################################################
def hash_model_files(model_path):
    """Return an md5 hex digest of the UFO model's python source (the ``*.py``
    files in ``model_path``). Used to detect that the model on disk still
    matches the one a process was generated with. Returns None if the path has
    no python file (or cannot be read)."""
    import hashlib
    try:
        pyfiles = sorted(f for f in os.listdir(model_path) if f.endswith('.py'))
    except OSError:
        return None
    if not pyfiles:
        return None
    h = hashlib.md5()
    for name in pyfiles:
        h.update(name.encode('utf-8'))
        try:
            with open(os.path.join(model_path, name), 'rb') as f:
                h.update(f.read())
        except OSError:
            return None
    return h.hexdigest()


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
                notification.setTitle_('MadGraph7')
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
              osascript -e 'display notification "{}" with title "MadGraph7" subtitle "{}"'
              """.format(info_text, subtitle))
            except:
                pass

        elif self.working == 'notify-send':
            try:
                os.system(""" notify-send "MadGraph7" "{}"  &> /dev/null """.format(info_text,subtitle))
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
#         'quit': ['Thanks %(user)s for using MadGraph7, even on April 1st!']
               }

    default_banner_1 =  "************************************************************\n" + \
        "*                                                          *\n" + \
        "*                     W E L C O M E to                     *\n" + \
        "*                    M A D G R A P H 7                     *\n" + \
        "*                                                          *\n" + \
        "*                                                          *\n" 


    default_banner_2 =        "*                                                          *\n" + \
        "%s" + \
        "*                                                          *\n" + \
        "*       The MadGraph7 Development Team - Find us at        *\n" + \
        "*              http://madgraph.phys.ucl.ac.be/             *\n" + \
        "*                            and                           *\n" + \
        "*            http://amcatnlo.web.cern.ch/amcatnlo/         *\n" + \
        "*                                                          *\n" + \
        "*               Type 'help' for in-line help.              *\n" + \
        "*       Type 'tutorial' to learn how MadGraph7 works       *\n" + \
        "*                                                          *\n" + \
        "************************************************************"

    May4_banner = "*                           _____                          *\n" + \
        "*                       ,-~\"     \"~-.                      *\n" + \
        "*        *            ,^ ___         ^.             *      *\n" + \
        "*          *         / .^   ^.         \\         *         *\n" + \
        "*            *      Y  l  o  !          Y      *           *\n" + \
        "*              *   l_  `.___.'         _,[   *             *\n" + \
        "*                * |^~\"--------------~\"\"^| *               *\n" + \
        "*              *   !     May the 4th     !   *             *\n" + \
        "*            *       \\                 /       *           *\n" + \
        "*          *          ^.             .^          *         *\n" + \
        "*        *              \"-.._____.,-\"              *       *\n"

    Zcommezorglub =  "* M:::::::::M         M:::::::::M                          *\n" + \
        "* M:::::::::M         M:::::::::M                          *\n" + \
        "* M::::::::::M       M::::::::::M                          *\n" + \
        "* M:::::::::::M     M:::::::::::M   (_)___                 *\n" + \
        "* M:::::::M::::M   M::::M:::::::M   | / __|                *\n" + \
        "* M::::::M M::::M M::::M M::::::M   | \\__ \\                *\n" + \
        "* M::::::M  M::::M::::M  M::::::M   |_|___/                *\n" + \
        "* M::::::M   M:::::::M   M::::::M                          *\n" + \
        "* M::::::M    M:::::M    M::::::M    / _| ___  _ __        *\n" + \
        "* M::::::M     MMMMM     M::::::M   | |_ / _ \\| '__|       *\n" + \
        "* M::::::M               M::::::M   |  _| (_) | |          *\n" + \
        "* M::::::M               M::::::M   |_/\\/\\___/|_|          *\n" + \
        "* M::::::M               M::::::M                          *\n" + \
        "* MMMMMMMM               MMMMMMMM                          *\n" + \
        "*                                                          *\n" + \
        "*     https://en.wikipedia.org/wiki/Z_comme_Zorglub        *\n"    

    towel_day_banner = "*      .---------.                                         *\n" + \
        "*      |  DON'T  |    The Guide: you're reading it.        *\n" + \
        "*      |  PANIC  |    Type 'help' for in-line help.        *\n" + \
        "*      '---------'                                         *\n" + \
        "*                                                          *\n" + \
        "*      .---------.                                         *\n" + \
        "*      |~~~~~~~~~|    The Towel: useful for wrapping       *\n" + \
        "*      |~ TOWEL ~|    your phase space cuts, or just       *\n" + \
        "*      |~~~~~~~~~|    crying when your job crashes.        *\n" + \
        "*      '---------'                                         *\n" + \
        "*                                                          *\n" + \
        "*         ) ) )                                            *\n" + \
        "*        ( ( (        A cup of tea: feed its Brownian      *\n" + \
        "*      .-------.-.    motion to The Infinite Improbability *\n" + \
        "*      |  tea  |-'    Drive (the Monte Carlo integration)  *\n" + \
        "*      '-------'      to obtain cross sections (probably). *\n"


    # The original MadGraph5 banner (the '5' diagram, now a 7), shown on the
    # anniversary of the MadGraph 5 paper, arXiv:1106.0522, 2 June 2011.
    mg5_paper_banner = \
        "*                 *                       *                *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                     * * * * 7 * * * *                    *\n" + \
        "*                   *        * *        *                  *\n" + \
        "*                 *                       *                *\n" + \
        "*                                                          *\n" + \
        "*    On this day in 2011 the MadGraph 5 paper appeared.    *\n" + \
        "*        Happy birthday!   arXiv:1106.0522 [hep-ph]        *\n"

    special_banner = {(4,5): May4_banner, (25,5): towel_day_banner,
                     (14,10): Zcommezorglub, (2,6): mg5_paper_banner}

    
    def __init__(self, msgtype):

        try:
            now = time.localtime()
            date = now.tm_mday, now.tm_mon 
            if date in [(1,4)]:
                madgraph.iolibs.drawing_eps.EpsDiagramDrawer.april_fool = True
                if msgtype == 'loading':
                   self.post_banner(date) 
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
                self.post_banner(date)
                return
            else:
                self.post_banner(date)
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
        
    def post_banner(self, date):

        if MADEVENT:
            return ""
        from madgraph import MG5DIR
        import madgraph.interface.madgraph_interface as madgraph_interface
        # written by bin/create_release.py, so a git checkout has none.  That is
        # normal, not a failure: without it there is simply no contributor to
        # celebrate, and reporting it printed a DEBUG line under every error
        # message the user got (EasterEgg('error') comes through here).
        authors = pjoin(MG5DIR, 'input', 'authors.md')
        if not os.path.exists(authors):
            return ""
        to_add = []
        ff = open(authors, 'r')
        for line in ff:
            author, fdate = line.split()
            year, month, day = [int(i) for i in fdate.split('-')]
            if (day, month) == date:
                to_add.append((author, year))
        ff.close()

        if not to_add:
            return ""
        
        text= """
*                CONTRIBUTOR OF THE DAY:                    *
"""
        for author, year in to_add:
            one = "*  - %s (first contribution in %s)" % (author, year) 
            text += '%s%s*\n' % (one, ' '* (59-len(one)))
        text += "************************************************************"

        madgraph_interface.CmdExtended.intro_banner += text


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
    """ return v2 if v1>v2
        return v1 if v1<v2
        return v1 if v1=v2 
        return v1 if v2 is not in 1.2.3.4.5 format (treat '<n>_text' as '<n>' if n is an integer)
        return v2 if v1 is not in 1.2.3.4.5 format (treat '<n>_text' as '<n>' if n is an integer)
    """
    
    for a1, a2 in zip_longest(v1, v2, fillvalue=0):
        if '_' in str(a1) : a1 = str(a1)[:str(a1).index('_')]
        if '_' in str(a2) : a2 = str(a2)[:str(a2).index('_')]
        try:
            a1 = int(a1)
        except:
            return v2
        try:
            a2 = int(a2)
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
    
    # get the MadGraph7 version
    if '__mg5amcnlo__' in plugin_support:
        mg5_ver = plugin_support['__mg5amcnlo__']
    else:
        info = get_pkg_info()
        mg5_ver = info['version'].split('.')

    # A MadGraph7 plugin gives the range of MadGraph7 versions it supports
    # with minimal_mg7_version and maximal_mg7_version (and optionally
    # latest_validated_mg7_version). The MG5_aMC@NLO attributes
    # (minimal_mg5amcnlo_version, ...) refer to the MG5_aMC@NLO numbering,
    # which can not be compared to the MadGraph7 one.
    if hasattr(obj, 'minimal_mg7_version') or hasattr(obj, 'maximal_mg7_version'):
        try:
            min_ver = obj.minimal_mg7_version
            max_ver = obj.maximal_mg7_version
        except AttributeError:
            logger.error("Plugin %s defines only one of minimal_mg7_version and "
                         "maximal_mg7_version. It is therefore discarded" % name)
            plugin_support[name] = False
            return False
        val_ver = getattr(obj, 'latest_validated_mg7_version',
                          getattr(obj, 'latest_validated_version', None))
    elif hasattr(obj, 'minimal_mg5amcnlo_version') and \
                                    hasattr(obj, 'maximal_mg5amcnlo_version'):
        logger.warning("Plugin %s only gives the MG5_aMC@NLO versions it supports "
                       "(minimal_mg5amcnlo_version, maximal_mg5amcnlo_version). It "
                       "is used without version check. Define minimal_mg7_version "
                       "and maximal_mg7_version for MadGraph7." % name)
        plugin_support[name] = True
        return True
    else:
        logger.error("Plugin %s misses some required info to be valid "
                     "(minimal_mg7_version, maximal_mg7_version). It is therefore "
                     "discarded" % name)
        plugin_support[name] = False
        return False
    
    logger.debug("Validating plugin against this version: %s." % '.'.join(str(i) for i in mg5_ver) )
    if get_older_version(min_ver, mg5_ver) == min_ver and \
       get_older_version(mg5_ver, max_ver) == mg5_ver:
        plugin_support[name] = True
        if val_ver and get_older_version(mg5_ver, val_ver) == val_ver:
            logger.warning("""Plugin %s has marked as NOT being validated with this version: %s. 
It has been validated for the last time with version: %s""",
			   name, '.'.join(str(i) for i in mg5_ver), '.'.join(str(i) for i in val_ver) )
    else:
        if __debug__:
            logger.error("Plugin %s seems not supported by this version of MadGraph7. Keep it active (please update status)" % name)
            plugin_support[name] = True            
        else:
            logger.error("Plugin %s is not supported by this version of MadGraph7." % name)
            plugin_support[name] = False
    return plugin_support[name]
    

#decorator
def set_global(loop=False, unitary=True, mp=False, cms=False,
               dual=0, npwave=(0,)):
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
            old_dual = aloha.dual_mode
            # npwave is a list mutated in place, so keep a copy of it
            old_npwave = list(aloha.npwave)
            aloha.loop_mode = loop
            aloha.unitary_gauge = unitary
            aloha.mp_precision = mp
            aloha.complex_mass = cms
            aloha.dual_mode = dual
            aloha.npwave = list(npwave)
            aloha_lib.KERNEL.clean()
            try:
                out =  f(*args, **opt)
            except:
                aloha.loop_mode = old_loop
                aloha.unitary_gauge = old_gauge
                aloha.mp_precision = old_mp
                aloha.complex_mass = old_cms
                aloha.dual_mode = old_dual
                aloha.npwave = old_npwave
                raise
            aloha.loop_mode = old_loop
            aloha.unitary_gauge = old_gauge
            aloha.mp_precision = old_mp
            aloha.complex_mass = old_cms
            aloha.dual_mode = old_dual
            aloha.npwave = old_npwave
            aloha_lib.KERNEL.clean()
            return out
        return deco_f_set
    return deco_set
   
    
    

def plugin_import(module, error_msg, fcts=[]):
    """convenient way to import a plugin file/function"""
    
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

def iter_plugins(plugin_path, warning=False, only=None):
    """yield (name, module) for each plugin of the plugin directories that can
    be imported (with a warning for the others if warning is set). If only is
    set, consider only the plugin of that name.
    The support of the plugin for this version is not checked here."""

    for plugpath in plugin_path:
        plugindirname = os.path.basename(plugpath)
        for plug in os.listdir(plugpath):
            if only is not None and plug != only:
                continue
            if not os.path.exists(pjoin(plugpath, plug, '__init__.py')):
                continue
            try:
                with stdchannel_redirected(sys.stdout, os.devnull):
                    __import__('%s.%s' % (plugindirname, plug))
            except Exception as error:
                if warning:
                    logger.warning("error detected in plugin: %s.", plug)
                    logger.warning("%s", error)
                continue
            yield plug, sys.modules['%s.%s' % (plugindirname, plug)]

def from_plugin_import(plugin_path, target_type, keyname=None, warning=False,
                       info=None):
    """return the class associated with keyname for a given plugin class
    if keyname is None, return all the name associated"""
    
    validname = []
    for plug, plugin in iter_plugins(plugin_path, warning):
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
    
    
    
def plugin_diagram_filter(plugin_path, name):
    """return the diagram filter function of the plugin called `name` (the
    `new_diagram_filter` attribute of its __init__.py, a function
    remove_diag(diag, model) returning True for the diagrams to remove; one
    filter per plugin, selected by the name of the plugin).
    If name is None, return the names of the plugins providing one.
    Raise MadGraph5Error if the plugin does not provide a valid filter."""

    if name is None:
        return [plug for plug, plugin in iter_plugins(plugin_path)
                if callable(getattr(plugin, 'new_diagram_filter', None)) and
                   is_plugin_supported(plugin)]

    if not name:
        raise MadGraph5Error("No plugin name given for the diagram filter "
                             "(--diagram_filter=PLUGINNAME).")
    for plug, plugin in iter_plugins(plugin_path, warning=True, only=name):
        if not hasattr(plugin, 'new_diagram_filter'):
            raise MadGraph5Error("Plugin %s does not provide a diagram filter "
                                 "(new_diagram_filter in its __init__.py)." % name)
        if not is_plugin_supported(plugin):
            raise MadGraph5Error("Plugin %s is not supported by this version "
                                 "of MadGraph7." % name)
        filter_fct = plugin.new_diagram_filter
        if not callable(filter_fct):
            raise MadGraph5Error("new_diagram_filter of plugin %s must be a "
                   "function remove_diag(diag, model), not %r" % (name, filter_fct))
        logger.info('Using diagram filter from plugin %s' % name, '$MG:BOLD')
        return filter_fct
    raise MadGraph5Error("No plugin '%s' found (or it can not be imported) to "
                         "provide the diagram filter." % name)

#===============================================================================
# LHAPDF locations
#===============================================================================
class LhapdfPaths(collections.namedtuple('LhapdfPaths',
                                    ['config', 'data_paths', 'download_path'])):
    """Where LHAPDF lives for one run: the lhapdf-config executable, the
    directories to search for PDF sets, and the one a missing set may be
    downloaded into. Any field may be None or empty."""

    def find_set(self, pdf_set):
        """The first data path holding <pdf_set>/, or None."""

        for base in self.data_paths:
            if os.path.isdir(pjoin(base, pdf_set)):
                return base
        return None

    def with_data_path(self, path):
        """A copy with *path* first in the search list (used after a download)."""

        return self._replace(
            data_paths=[path] + [p for p in self.data_paths if p != path])


def _tool_executable(value, root):
    """Turn a configuration value into a usable executable path, or None.

    Accepts an absolute path, a path relative to the MadGraph root
    ('./HEPTools/...'), and a bare program name looked up on PATH
    ('lhapdf-config', the shipped default). A trailing '--python=X.Y' is a
    filter for the 'set' command, not part of the path, so it is dropped.
    """

    if not value:
        return None
    exe = str(value).split()[0].strip('"\'')
    if exe.lower() in ('none', 'auto'):
        return None
    if not os.path.isabs(exe) and (os.sep in exe or exe.startswith('.')):
        return which(os.path.normpath(pjoin(root, exe))) or which(exe)
    return which(exe)


_lhapdf_datadirs_cache = {}
def _lhapdf_datadirs(exe):
    """The PDF-set directories a lhapdf-config reports, in its own order.
    '--datadir' is the LHAPDF 6 spelling, '--pdfsets-path' the LHAPDF 5 one."""

    if exe not in _lhapdf_datadirs_cache:
        dirs = []
        for flag in ('--datadir', '--pdfsets-path'):
            try:
                out = subprocess.check_output([exe, flag],
                                              stderr=subprocess.DEVNULL)
            except (OSError, subprocess.CalledProcessError):
                continue
            out = out.decode(errors='ignore').strip()
            if out:
                dirs = [p for p in out.split(os.pathsep) if p]
                break
        _lhapdf_datadirs_cache[exe] = dirs
    return _lhapdf_datadirs_cache[exe]


# The LHAPDF sets mirrored on CVMFS. Where it is mounted -- most grid and
# laboratory clusters -- it is a complete, read-only, node-local copy of the
# sets, so a set found there needs neither a download nor a transfer to the
# worker node.
CVMFS_LHAPDF_PATH = '/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current'

def get_cvmfs_lhapdf_path(options=None):
    """Return the CVMFS PDF-set mirror to use, or None.

    The location is configurable ('cvmfs_lhapdf_path'); setting it to None
    switches the fallback off. A configured path that is not mounted is
    simply ignored, so the default value is safe on any machine.
    """

    path = CVMFS_LHAPDF_PATH
    if options is not None:
        try:
            path = options.get('cvmfs_lhapdf_path', CVMFS_LHAPDF_PATH)
        except AttributeError:
            pass
    if not path or str(path).strip().lower() in ('none', 'false', ''):
        return None
    path = str(path).strip()
    return path if os.path.isdir(path) else None


def _writable_dir(path):
    """True if *path* is a writable directory or can be created as one."""

    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return os.access(probe, os.W_OK)


def resolve_lhapdf(options=None, root=None, use_env=True, create=False):
    """Locate LHAPDF for a run and return a LhapdfPaths.

    Both the 'launch' command and a standalone bin/generate_events resolve
    through here, so they cannot disagree about where the PDF sets live.
    Never raises: a missing or broken LHAPDF just yields empty fields.

    ``options``  an MadGraph7 option mapping; 'lhapdf', 'lhapdf_py3',
                 'lhapdf_py2', 'heptools_install_dir' and 'mg5_path' are read
    ``root``     what a relative option value is resolved against
    ``use_env``  honour $MADGRAPH_LHAPDF_CONFIG and $LHAPDF_DATA_PATH
    ``create``   create the download directory (only needed before a download)
    """

    options = options or {}
    root = root or options.get('mg5_path') or mg_root()

    candidates = []
    if use_env:
        candidates.append(os.environ.get('MADGRAPH_LHAPDF_CONFIG'))
    candidates += [options.get(key)
                   for key in ('lhapdf', 'lhapdf_py3', 'lhapdf_py2')]
    candidates.append('lhapdf-config')

    # Accept the first candidate that actually answers a data-directory query;
    # remember the first one that merely exists in case none of them answers.
    config, data_dirs, fallback, seen = None, [], None, set()
    for value in candidates:
        exe = _tool_executable(value, root)
        if not exe or exe in seen:
            continue
        seen.add(exe)
        fallback = fallback or exe
        dirs = _lhapdf_datadirs(exe)
        if dirs:
            config, data_dirs = exe, dirs
            break
    config = config or fallback

    heptools = options.get('heptools_install_dir') or pjoin('.', 'HEPTools')
    if not os.path.isabs(heptools):
        heptools = pjoin(root, heptools)

    search = []
    if use_env and os.environ.get('LHAPDF_DATA_PATH'):
        search += os.environ['LHAPDF_DATA_PATH'].split(os.pathsep)
    search += data_dirs
    search.append(pjoin(heptools, 'lhapdf_pdfsets'))
    search.append(pjoin(root, 'lhapdf_pdfsets'))
    # last resort before a download: the read-only CVMFS mirror. It is never a
    # download target -- _writable_dir rejects it -- so it only ever spares us
    # from fetching a set that is already on the machine.
    cvmfs = get_cvmfs_lhapdf_path(options)
    if cvmfs:
        search.append(cvmfs)
    search = [os.path.abspath(p) for p in search]

    data_paths = []
    for path in search:
        if path not in data_paths and os.path.isdir(path):
            data_paths.append(path)

    download_path = next((p for p in search if _writable_dir(p)), None)
    if download_path and create:
        try:
            os.makedirs(download_path, exist_ok=True)
        except OSError:
            download_path = None

    return LhapdfPaths(config, data_paths, download_path)


# Where 'lhapdf get' itself fetches the sets and the id index from.
LHAPDF_SETS_URL = 'https://lhapdfsets.web.cern.ch/current'

class LhapdfServerError(MadGraph5Error):
    """The LHAPDF server refused a request (e.g. an unknown set name): asking
    again, or with another client, gets the same answer."""

def _fetch_url(url, path, timeout=60):
    """Download *url* to *path*: urllib first, then curl/wget (e.g. for a
    python without usable SSL certificates). Raises MadGraph5Error on failure."""

    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, \
                open(path, 'wb') as out:
            shutil.copyfileobj(response, out)
            # a dropped connection just ends the stream early
            expected = response.headers.get('Content-Length')
            if expected and out.tell() != int(expected):
                raise IOError('truncated: got %d of %s bytes'
                              % (out.tell(), expected))
        return
    except urllib.error.HTTPError as error:
        raise LhapdfServerError('could not download %s: %s' % (url, error))
    except Exception as error:
        reason = error
    for tool, cmd in (('curl', ['curl', '-fsL', url, '-o', path]),
                      ('wget', ['wget', '-q', url, '-O', path])):
        if which(tool) and subprocess.call(cmd) == 0:
            return
    raise MadGraph5Error('could not download %s: %s' % (url, reason))


def download_lhapdf_set(pdf_set, target, base_url=LHAPDF_SETS_URL):
    """Install the LHAPDF set *pdf_set* into the directory *target* straight
    from the LHAPDF server, without needing LHAPDF itself (its .info and .dat
    grids are all madspace reads). pdfsets.index comes along when *target* has
    none, so an lhaid can be turned into a set name later. The set appears in
    one rename, so an interrupted download never leaves a partial one behind.
    Raises MadGraph5Error on failure."""

    import tarfile
    os.makedirs(target, exist_ok=True)
    if os.path.isdir(pjoin(target, pdf_set)):
        return pjoin(target, pdf_set)
    tmpdir = tempfile.mkdtemp(prefix='.%s.' % pdf_set, dir=target)
    try:
        tarball = pjoin(tmpdir, '%s.tar.gz' % pdf_set)
        for attempt in range(3):
            try:
                _fetch_url('%s/%s.tar.gz' % (base_url, pdf_set), tarball)
                with tarfile.open(tarball) as tar:
                    tar.extractall(tmpdir, filter='data')
                break
            except LhapdfServerError:
                raise
            except (MadGraph5Error, tarfile.TarError, EOFError, OSError) as error:
                # a dropped or corrupted transfer: fetch it again
                reason = error
                logger.debug('download of %s failed: %s', pdf_set, reason)
                shutil.rmtree(pjoin(tmpdir, pdf_set), ignore_errors=True)
        else:
            raise MadGraph5Error('could not install %s: %s' % (pdf_set, reason))
        if not os.path.exists(pjoin(tmpdir, pdf_set, '%s.info' % pdf_set)):
            raise MadGraph5Error('%s.tar.gz does not contain the set %s'
                                 % (pdf_set, pdf_set))
        try:
            os.rename(pjoin(tmpdir, pdf_set), pjoin(target, pdf_set))
        except OSError:
            # another process installed it meanwhile
            if not os.path.isdir(pjoin(target, pdf_set)):
                raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    download_lhapdf_index(target, base_url)
    return pjoin(target, pdf_set)


def download_lhapdf_index(target, base_url=LHAPDF_SETS_URL):
    """Fetch LHAPDF's pdfsets.index (lhaid -> set name) into *target* unless
    it is already there. Returns its path, or None if it could not be had."""

    index = pjoin(target, 'pdfsets.index')
    if os.path.exists(index):
        return index
    try:
        os.makedirs(target, exist_ok=True)
        _fetch_url('%s/pdfsets.index' % base_url, index + '.part')
        os.replace(index + '.part', index)
        return index
    except Exception as error:
        logger.debug('could not download pdfsets.index: %s', error)
        if os.path.exists(index + '.part'):
            os.remove(index + '.part')
        return None


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
        # LD_LIBRARY_PATH set now only reaches child processes: the python
        # module (built without an rpath, e.g. by HEPTools) would not find
        # libLHAPDF in this one. Load it first, globally, so it does.
        for libname in ('libLHAPDF.so', 'libLHAPDF.dylib'):
            libpath = os.path.join(lhapdf_libdir, libname)
            if os.path.exists(libpath):
                try:
                    import ctypes
                    ctypes.CDLL(libpath, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break
        try:
            candidates=[dirname for dirname in os.listdir(lhapdf_libdir) \
                            if os.path.isdir(os.path.join(lhapdf_libdir,dirname))]
        except OSError:
            candidates=[]
        if os.path.isdir(pjoin(lhapdf_libdir, os.pardir, 'local', 'lib')):
            candidates += [pjoin(os.pardir,'local', 'lib', dirname) for dirname in os.listdir(pjoin(lhapdf_libdir, os.pardir, 'local', 'lib'))
                           if os.path.isdir(os.path.join(lhapdf_libdir,os.pardir, 'local', 'lib', dirname))]
        for candidate in candidates:
            for subdir in ['site-packages', 'dist-packages']:
                if os.path.isdir(os.path.join(lhapdf_libdir,candidate, subdir)):
                    sys.path.insert(0,os.path.join(lhapdf_libdir,candidate, subdir))
                    try:
                        import lhapdf
                        use_lhapdf=True
                        break
                    except ImportError as  error:
                        sys.path.pop(0)
                        continue
            else:
                continue
            break
    if not use_lhapdf:
        try:
            candidates=[dirname for dirname in os.listdir(lhapdf_libdir+'64') \
                            if os.path.isdir(os.path.join(lhapdf_libdir+'64',dirname))]
        except OSError:
            candidates=[]
        if os.path.isdir(pjoin(lhapdf_libdir, os.pardir, 'local', 'lib64')):
            candidates += [pjoin(os.pardir,'local', 'lib64', dirname) for dirname in os.listdir(pjoin(lhapdf_libdir, os.pardir, 'local', 'lib'))
                           if os.path.isdir(os.path.join(lhapdf_libdir,os.pardir, 'local', 'lib64', dirname))]
        for candidate in candidates:
            for subdir in ['site-packages', 'dist-packages']:
                if os.path.isdir(os.path.join(lhapdf_libdir+'64',candidate, subdir)):
                    sys.path.insert(0,os.path.join(lhapdf_libdir+'64',candidate, subdir))
                    try:
                        import lhapdf
                        use_lhapdf=True
                        break
                    except ImportError as error:
                        sys.path.pop(0)
                        continue
            else:
                continue
            break
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

