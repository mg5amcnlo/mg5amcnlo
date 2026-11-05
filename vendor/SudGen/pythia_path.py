#! /usr/bin/env python3
import os
from pathlib import Path

pjoin=os.path.join

pythia8_path=None

# Check the per-user configuration file (see misc.user_config_file)
xdg=os.environ.get('XDG_CONFIG_HOME')
if xdg:
    conf_file=pjoin(xdg,'mg7','mg7_configuration.txt')
else:
    conf_file=pjoin(str(Path.home()),'.mg7','mg7_configuration.txt')
try:
    with open(conf_file) as f:
        data=f.readlines()
    for line in data:
        if 'pythia8_path' in line and not line.lstrip().startswith('#') :
            pythia8_path=line.split('=')[1].split()[0]
except OSError:
    pass

# Check the configuration file in input/mg7_configuration.txt
curr_path=os.getcwd()
conf_file=pjoin(curr_path,'..','..','input','mg7_configuration.txt')
try:
    with open(conf_file) as f:
        data=f.readlines()
    for line in data:
        if 'pythia8_path' in line and not line.lstrip().startswith('#') :
            pythia8_path=line.split('=')[1].split()[0]
except OSError:
    pass

if not pythia8_path:
    print('Pythia8 path not found in input/mg7_configuration.txt file. Cannot compile SudGen.')
else:
    makefile_inc=pjoin(curr_path,'makefile.inc')
    try:
        os.remove(makefile_inc)
    except OSError:
        pass
    with open(makefile_inc,'w') as f:
        f.write('WORK='+pythia8_path+'\n')
        f.write('PYTHIA8INCLUDE=\\$(WORK)/include \n')
        f.write('PYTHIA8LIB=\\$(WORK)/lib \n')
        f.write('PYTHIA8FLAGS=-lstdc++ -lz -ldl -fPIC \n')
    

