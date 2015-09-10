#! /usr/bin/env python

__author__ = 'Valentin Hirschi'
__email__  = "valentin.hirschi[at]gmail[dot]com"

import sys
import os
import subprocess
import shutil

pjoin = os.path.join


_lib_extensions = ['a']
if sys.platform == "darwin":
   _lib_extensions.append('dylib')
else:
   _lib_extensions.append('so')

_HepTools = {'hepmc':
               {'install_mode':'Default',
                'version':       '2.06.09',
                'tarball':      ['online','http://lcgapp.cern.ch/project/simu/HepMC/download/HepMC-%(version)s.tar.gz'],
                'mandatory_dependencies': [],
                'optional_dependencies' : [],
                'libraries' : ['libHepMC.%(libextension)s'],
                'install_path':  '%(prefix)s/hepmc/'},
             'boost':
               {'install_mode':'Default',
                'version':       '1.59.0',
                'tarball':      ['online','http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.tar.gz'],
                'mandatory_dependencies': [],
                'optional_dependencies' : [],
                'libraries' : ['libboost_system-mt.%(libextension)s','libboost_system.%(libextension)s'],
                'install_path':  '%(prefix)s/boost/'},
             'pythia8':
               {'install_mode':'Default',
                'version':       '',
                'tarball':      ['online',''],
                'mandatory_dependencies': ['hepmc'],
                'optional_dependencies' : ['lhapdf'],
                'libraries' : ['libpythia8.%(libextension)s'],
                'install_path':  '%(prefix)s/pythia8/'},
             'lhapdf':
               {'install_mode':'Default',
                'version':       '',
                'tarball':      ['online',''],
                'mandatory_dependencies': ['boost'],
                'optional_dependencies' : [],
                'libraries' : ['libLHAPDF.%(libextension)s'],
                'install_path':  '%(prefix)s/lhapdf6/'},
            }

_cwd             = os.getcwd()
_installers_path = os.path.abspath(os.path.dirname(os.path.realpath( __file__ )))
_cpp             = 'g++'
_gfrotran        = 'gfortran'
_prefix          = pjoin(_cwd,'HEPTools')

if len(sys.argv)>1 and sys.argv[1].lower() not in _HepTools.keys():
    print "HEPToolInstaller does not support the installation of %s"%sys.argv[1]
    sys.argv[1] = 'help'

if len(sys.argv)<2 or sys.argv[1]=='help':
    
    print """
./HEPToolInstaller <target> <options>"
     Possible values and meaning for the various <options> are:
           
     |           option           |    value           |                          meaning
     -============================-====================-===================================================
     | <target>                   | Any <TOOL>         | Which HEPTool to install
     | --prefix=<path>            | install root path  | Specify where to install target and dependencies
     | --fortran_compiler=<path>  | path to gfortran   | Specify fortran compiler
     | --cpp_compiler=<path>      | path to g++        | Specify C++ compiler     
     | --with_<DEP>=<DepMode>     | <path>             | Use the specified path for dependency DEP
     |                            | Default            | Link against DEP if present otherwise install it
     |                            | OFF                | Do not link against dependency DEP
     | --<DEP>_tarball=<path>     | A path to .tar.gz  | Path of the tarball to be used if DEP is installed

     <TOOL> and <DEP> can be any of the following:\n        %s\n
Example of usage:
    ./HEPToolInstaller Pythia8 --prefix=~/MyTools --with_lhapdf=OFF --Pythia8_tarball=~/MyTarball.tar.gz
"""%', '.join(_HepTools.keys())
    sys.exit(0)

target_tool = sys.argv[1].lower()

available_options = ['--prefix','--fortran_compiler','--gfortran_compiler']+\
                    ['--%s_tarball'%tool for tool in _HepTools.keys()]+\
                    ['--with_%s'%tool for tool in _HepTools.keys()]

# Now parse the options
for option in sys.argv[2:]:
    try:
        option, value = option.split('=')
    except:
        print "Option '%s' not of the form '<opt>=<value>'."%option
    if option not in available_options:
        print "Option '%s' not reckognized options"
        sys.exit(0) 
    if option=='--prefix':
        if not os.path.isdir(value):
            print "Creating root directory '%s'."%os.path.abspath(value)
            os.mkdir(os.path.abspath(value))
        _prefix = os.path.abspath(value)
    elif option=='--fortran_compiler':
        _gfortran = value
    elif option=='--cpp_compiler':
        _cpp = value
    elif option.startswith('--with_'):
        _HepTools[option[7:]]['install_path'] = value if value!='OFF' else None
    elif option.endswith('_tarball'):
        access_mode = 'onine' if '//' in value else 'local'
        if access_mode=='local':
            value = os.path.abspath(value)
        _HepTools[option[2:-8]]['tarball'] = [access_mode, value]

# Apply substitutions if necessary:
for tool in _HepTools:
    _HepTools[tool]['install_path']=_HepTools[tool]['install_path']%{'prefix':_prefix}
    if _HepTools[tool]['tarball'][0]=='online':
        _HepTools[tool]['tarball'][1]=_HepTools[tool]['tarball'][1]%{'version':_HepTools[tool]['version']}
    
    new_libs = []
    for lib in _HepTools[tool]['libraries']:
        for libext in _lib_extensions:
            if lib%{'libextension':libext} not in new_libs:
                new_libs.append(lib%{'libextension':libext})
    _HepTools[tool]['libraries'] = new_libs

# TMP_directory (designed to work as with statement) and go to it
class TMP_directory(object):
    """create a temporary directory, goes to it, and ensure this one to be cleaned.
    """

    def __init__(self, suffix='', prefix='tmp', dir=None):
        self.nb_try_remove = 0
        import tempfile   
        self.path = tempfile.mkdtemp(suffix, prefix, dir)
        self.orig_path = os.getcwd()
        os.chdir(os.path.abspath(self.path))
    
    def __exit__(self, ctype, value, traceback ):
        os.chdir(self.orig_path)
        #True only for debugging:
        if False and isinstance(value, Exception):
            print "Directory %s not cleaned. This directory can be removed manually" % self.path
            return False
        try:
            shutil.rmtree(self.path)
        except OSError:
            import time
            self.nb_try_remove += 1
            if self.nb_try_remove < 3:
                time.sleep(10)
                self.__exit__(ctype, value, traceback)
            else:
                logger.warning("Directory %s not completely cleaned. This directory can be removed manually" % self.path)
        
    def __enter__(self):
        return self.path

# Now define the installation function
def install_hepmc(tmp_path):
    """Installation operations of hepmc""" 
    subprocess.call(' '.join([pjoin(_installers_path,'installHEPMC2.sh'),
                     _HepTools['hepmc']['install_path'],
                     _HepTools['hepmc']['version'],
                     _HepTools['hepmc']['tarball'][1]
                      ]), shell=True)

def install_boost(tmp_path):
    """Installation operations of boost"""
    subprocess.call(' '.join([pjoin(_installers_path,'installBOOST.sh'),
                     _HepTools['boost']['install_path'],
                     _HepTools['boost']['version'],
                     _HepTools['boost']['tarball'][1]
                      ]), shell=True)

def install_pythia8(tmp_path):
    """Installation operations of pythia8"""

def install_lhpadf6(tmp_path):
    """Installation operations of lhapdf6"""

def get_data(link):
    """ Pulls up a tarball from the web """
    if sys.platform == "darwin":
        program = "curl -OL"
    else:
        program = "wget"
    subprocess.call('%s %s'%(program,link), shell=True)
    return pjoin(_cwd,os.path.basename(link))

# find a library in common paths 
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
           ["LIBRARY_PATH","PATH","DYLD_LIBRARY_PATH","LD_LIBRARY_PATH"] 
                                                  if env_path in os.environ],[])
        for path in locations:
            lib_file = os.path.join(path, lib)
            if is_lib(lib_file):
                return lib_file
    return None

# Find a dependency
def find_dependency(tool):
    """ Check if a tool is already installed or needs to be."""
    if _HepTools[tool]['install_path'] is None:
        return None
    elif _HepTools[tool]['install_path'].lower() != 'default':
        return _HepTools[tool]['install_path']
    else:
        # Treat the default case which is "install dependency if not found
        # otherwise use the one found".
        lib_found = None
        for lib in _HepTools[tool]['library']['libraries']:
            lib_search = which_lib()
            if not lib_search is None:
                lib_found = lib_search
                break
        if lib_found is None:
            return 'TO_INSTALL'
        else:
            # Return the root folder which is typically the install dir
            if os.path.dirname(lib_found).endswith('lib')
                return os.path.abspath(pjoin(os.path.dirname(lib_found),'..'))
            else:
                return os.path.abspath(os.path.dirname(lib_found))

def install_with_dependencies(target):
    """ Recursive install function for a given tool, taking care of its dependencies"""

    for dependency in _HepTools[target]['mandatory_dependencies']+_HepTools[target]['optional_dependencies']:
        path = find_dependency(dependency)
        if path is None:
            if dependency in _HepTools[target]['optional_dependencies']:
                print "Optional dependency '%s' of tool '%s' is disabled and will not be available."%(dependency,target)
                _HepTools[target]['optional_dependencies'].remove(dependency)
            else:
                print "Mandatory dependency '%s' of tool '%s' unavailable. Exiting now."%(dependency,target)
                sys.exit(0)
        elif path=='TO_INSTALL':
            install_with_dependencies(dependency)
        else:
            _HepTools[dependency]['install_path']=path

    with TMP_directory() as tmp_path:
        # Get the source tarball if online
        if _HepTools[target]['tarball'][0]=='online':
            try:
                tarball_path = get_data(_HepTools[target]['tarball'][1])
            except Exception as e:
                print "Could not download data at '%s' because of:\n%s\n"%(_HepTools[target]['tarball'][1],str(e))
                sys.exit(9)
            _HepTools[target]['tarball'] = ('local',tarball_path)
        
        if not os.path.isdir(_HepTools[target]['install_path']):
            os.mkdir(_HepTools[target]['install_path'])
        exec('install_%s(tmp_path)'%target)



_environ = dict(os.environ)
try:   
    os.environ["CXX"]     = _cpp
    os.environ["FC"]      = _gfrotran
    install_with_dependencies(target_tool)
except ZeroDivisionError as e:
    os.environ.clear()
    os.environ.update(_environ)
    print "The following error occured during the installation of '%s' (and its dependencies):\n%s"%(target_tool,str(e))
    sys.exit(9)

os.environ.clear()
os.environ.update(_environ)
print "Successful installation of '%s' in '%s'."%(target_tool,_prefix)
