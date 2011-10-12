#!/usr/bin/env python
import re, sys, os


g77 = re.compile('g77')
gfortran = re.compile('gfortran')

def mod_compilator(directory, new='gfortran'):
    #define global regular expression
    rule={'S-REGEXP_g77+gfortran+re.I':''}
    if type(directory)!=list:
        directory=[directory]

    #search file
    file_to_change=find_makefile_in_dir(directory)
    for name in file_to_change:
        text = open(name,'r').read()
        if new == 'g77':
            text= gfortran.sub('g77', text)
        else:
            text= g77.sub('gfortran', text)
        open(name,'w').write(text)

def detect_current_compiler(path):
    """find the currnt compiler for the current directory"""
    
    comp = re.compile("^\s*FC\s*=(g77|gfortran)\s*")
    for line in open(path):
        if comp.search(line):
            compiler = comp.search(line).groups()[0]
            if compiler == 'g77':
                print 'Currently using g77, Will pass to gfortran'
            elif compiler == "gfortran":
                print 'Currently using gfortran, Will pass to g77'
            else:
                raise Exception, 'unrecognize compiler'
            return compiler



def find_makefile_in_dir(directory):
    """ retrun a list of all file startinf with makefile in the given directory"""

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
                 'libdsample.a', 'libgeneric.a', 'libmodel.a', 'libpdf.a', 'libdhelas3.so', 'libTF.a', 
                 'libdsample.so', 'libgeneric.so', 'libmodel.so', 'libpdf.so']
    lib_pos='./lib'
    [os.remove(os.path.join(lib_pos, lib)) for lib in libraries \
                                 if os.path.exists(os.path.join(lib_pos, lib))]


def rm_file_extension( ext, dirname, names):

    [os.remove(os.path.join(dirname, name)) for name in names if name.endswith(ext)]


def go_to_main_dir():
    """ move to main position """
    pos=os.getcwd()
    last=pos.split(os.sep)[-1]
    if last=='bin':
        os.chdir(os.pardir)
        return
    
    list_dir=os.listdir('./')
    if 'bin' in list_dir:
        return
    else:
        sys.exit('Error: script must be executed from the main, bin or Python directory')


    
if "__main__"==__name__:

    # Collect position to modify
    directory=['Source']
    pypgs = os.path.join(os.path.pardir, 'pythia-pgs')
    madanalysis = os.path.join(os.path.pardir, 'MadAnalysis')
    for d in [pypgs, madanalysis]:
        if os.path.isdir(d):
            directory.append(d)
        
    # start the real work
    go_to_main_dir()

    if len(sys.argv) > 1:
        if '-h' in sys.argv:
            print 'change the compilator from g77/gfortran to the other one'
            print 'If you want to force one run as ./bin/change_compiler.py g77'
            sys.exit()
        assert len(sys.argv) == 2
        
        if sys.argv[-1] == 'g77':
            print 'pass to g77'
            new_comp = 'g77'
        else:
            print 'pass to gfortran'
            new_comp = 'gfortran'
    else:
        old_comp = detect_current_compiler('./Source/make_opts')
        if old_comp == 'g77':
            new_comp = 'gfortran'
        else:
            new_comp = 'g77'
    mod_compilator(directory, new_comp)
    rm_old_compile_file()
    print 'Done'
