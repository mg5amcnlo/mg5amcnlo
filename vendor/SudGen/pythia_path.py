#! /usr/bin/env python3
import os
from pathlib import Path

pjoin=os.path.join

# Check the configuration file in ~/.mg5/mg5_configuration.txt
home = str(Path.home())
conf_file=pjoin(home,'.mg5','mg5_configuration.txt')
try:
    with open(conf_file) as f:
        data=f.readlines()
    for line in data:
        if 'pythia8_path' in line and not line.lstrip().startswith('#') :
            pythia8_path=line.split('=')[1].split()[0]
except OSError:
    pass

# Check the configuration file in ~/.mg5/mg5_configuration.txt
curr_path=os.getcwd()
conf_file=pjoin(curr_path,'..','..','input','mg5_configuration.txt')
try:
    with open(conf_file) as f:
        data=f.readlines()
    for line in data:
        if 'pythia8_path' in line and not line.lstrip().startswith('#') :
            pythia8_path=line.split('=')[1].split()[0]
except OSError:
    pass

if not pythia8_path:
    print('Pythia8 path not found in input/mg5_configuration.txt file. Cannot compile SudGen.')
else:
    makefile_inc=pjoin(curr_path,'makefile.inc')
    try:
        os.remove(makefile_inc)
    except OSError:
        pass
    with open(makefile_inc,'w') as f:
        f.write('WORK='+pythia8_path+'\n')
        f.write('PYTHIA8INCLUDE=\$(WORK)/include \n')
        f.write('PYTHIA8LIB=\$(WORK)/lib \n')
        f.write('PYTHIA8FLAGS=-lstdc++ -lz -ldl -fPIC \n')
    

