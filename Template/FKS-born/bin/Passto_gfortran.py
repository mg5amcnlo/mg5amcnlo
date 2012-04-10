#!/usr/bin/env python
import sys, os
sys.path+=['../'*i+'./Source/MadWeight_File/Python' for i in range(0,2)]
sys.path+=['../'*i+'Template/Source/MadWeight_File/Python' for i in range(0,2)]
import mod_file, MW_param

def mod_dir_to_gfortran(directory):
    #define global regular expression
    rule={'S-REGEXP_f77+gfortran+re.I':'','S-REGEXP_g77+gfortran+re.I':''}

    if type(directory)!=list:
        directory=[directory]

    #search file
    file_to_change=find_makefile_in_dir(directory)

    #modify all those makefile
    mod_file.mod_file(file_to_change,rule,opt={'nowarning':'\'all\''})
    print 'remove old compile file'
    
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



    
if "__main__"==__name__:
    print 'start main'
    value=0
#    while value not in ['1','2']:
#        value=raw_input('On which directory do you want to apply this script? (1/2)\n' + \
#                               ' 1: On this copy of the Template directory\n' + \
#                               ' 2: On the full MG_ME directory\n')
    value='1'
    if value == '1':
        directory=['Source','SubProcesses']
    else:
        directory=['..']

    MW_param.go_to_main_dir()
    mod_dir_to_gfortran(directory)
    rm_old_compile_file()
