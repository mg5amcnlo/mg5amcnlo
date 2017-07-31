#! /usr/bin/env python

# This is an example of how to run MadLoop from Python using the f2py compilation of the wrapper file 'f2py_wrapper.f'.

import os
import sys
import subprocess

pjoin = os.path.join
root_path = os.path.dirname(os.path.realpath( __file__ ))
sys.path.insert(0, root_path)

try:
    import allmatrix2py
except:
    if os.path.isfile(pjoin(root_path,'makefile')) and \
       os.path.isfile(pjoin(root_path,'all_matrix.f')) and \
       not os.path.isfile(pjoin(root_path,'allmatrix2py.so')):
        print "Trying to automatically generate the python module 'allmatrix2py.so' with f2py..."
        p = subprocess.Popen(['make','allmatrix2py.so'], stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, cwd=root_path)
        (out, err) = p.communicate()
        if p.returncode or not os.path.isfile(pjoin(root_path,'allmatrix2py.so')):
            print "ERROR: Failed to produce 'allmatrix2py.so' with 'make allmatrix2py.so' in '%s'. The error was:\n%s"%(root_path,err)
            sys.exit(0)
        try:
            import allmatrix2py
        except:
            print "ERROR: Could not load the f2py module 'allmatrix2py.so'. The following error occurred:\n",sys.exc_info()[0]
            sys.exit(0)
    else:
        if os.path.exists(pjoin(root_path,'allmatrix2py.so')):
            print "ERROR: Could not load the f2py module 'matrix2py.so'. The following error occurred:\n",sys.exc_info()[0]
            sys.exit(0)
        else:
            print "ERROR: Could not find the 'matrix2py.so' f2py module. Please generate it by running:\n"+\
                  "  > make matrix2py.so\n"+\
                  "in the <PROC_OUTPUT>/SubProcesses/P<chosen_proc> directory."
            sys.exit(0)

# Now we can use this MadLoop python module.

# This is a handy way of looking at what is available in the imported f2py module
# print help(matrix2py)

# Read the model parameters
allmatrix2py.initialise(os.path.abspath(pjoin(root_path,os.pardir,'Cards','param_card.dat')))
# Specify where MadLoop5_resources is
allmatrix2py.set_madloop_path(os.path.abspath(pjoin(root_path,'MadLoop5_resources')))
# List all processes available in this module

all_procs_available = [[pdg for pdg in proc_pdgs if pdg!=0] for proc_pdgs in allmatrix2py.get_pdg_order()]
print "Available processes are:"
for process_pdgs in all_procs_available:
    print "   --> %s"%str(process_pdgs)

def invert_momenta(p):
    """ fortran/C-python do not order table in the same order"""
    return [[p[j][i] for j in range(len(p))] for i in range(len(p[0]))]

# Now choose MadLoop's inputs for this evaluation

# The kinematic configuration in the convention (E, px, py, pz) and with particles ordered as in the process definition.
# This is the structure of this dictionary.
p= dict( (tuple(process_pdgs),[[None,]*4]*len(process_pdgs)) for process_pdgs in all_procs_available)

if (p.values()[0])[0][0] is None:
    if not os.path.isfile(pjoin(root_path,'PS.input')):
        print "\n\n==================================================================================================================="
        print "*                                           No kinematics defined!                                                *"
        print "*                                   -------------------------------------                                         *"
        print "* Please either define your kinematic configuration directly in check_sa.py or in a file 'PS.input'. Exiting now. *"
        print " Here is an example of the content of such a file for a process output containing g g > h and g g > h g."
        print \
"""

\"\"\"
Process=[21,21,25]
6.250000000000000E+001        0.000000000000000E+000        0.000000000000000E+000        6.250000000000000E+001
6.250000000000000E+001        0.000000000000000E+000        0.000000000000000E+000       -6.250000000000000E+001
1.250000000000000E+002        0.000000000000000E+000        0.000000000000000E+000        0.000000000000000E+000

Process=[21,21,25,21]
5.000000000000000E+002        0.000000000000000E+000        0.000000000000000E+000        5.000000000000000E+002
5.000000000000000E+002        0.000000000000000E+000        0.000000000000000E+000       -5.000000000000000E+002
5.078125000000001E+002       -1.567603625592635E+002        2.039552324827553E+002        4.196152845517468E+002
4.921874999999999E+002        1.567603625592635E+002       -2.039552324827552E+002       -4.196152845517467E+002
\"\"\"
"""
        print "===================================================================================================================\n\n"
        sys.exit(0)
    try:
        p               = {}
        current_process = None
        current_p       = []
        for line in open(pjoin(root_path,'PS.input'),'r').readlines():
            if line.startswith('Process='):
                if not current_process is None:
                    p[current_process] = current_p
                    current_p = []
                current_process = tuple(eval(line[8:]))
                continue
            if len(line.split())>0:
                current_p.append([float(line.split()[j]) for j in range(4)])
        p[current_process] = current_p
    except:
        print "ERROR: File PS.input is malformed. Error was:\n",sys.exc_info()[0]
        sys.exit(0)


for iProc, (process_pdgs, kinematic_config) in enumerate(p.items()):
    print '='*112
    P =invert_momenta(kinematic_config)
    # Alpha_s value
    alphas = 0.118
    # -1 means that the MadLoop averages/sum over helicity configuration. 
    # If a positive integer is picked it corresponds to the position of the helicity configuration listed in the file
    # 'MadLoop5_resouces/ML5_<id>_HelConfigs.dat'
    nhel = -1 # means sum over all helicity
    # Choice of renormalization scale
    renormalization_scale = 91.188

    finite_loop_me, return_code = allmatrix2py.smatrixhel(pdgs=list(process_pdgs),
		                          p=P, alphas=alphas,scales2=renormalization_scale,nhel=nhel)

    print '* %-108s *'%' MadLoop evaluation for the process '
    print '* %-108s *'%('  %s'%str(process_pdgs))
    print '* %-108s *'%' and the kinematic configuration:'
    print '* %-108s *'%((' %-3s'+' %-25s'*4)%('#','E',' p_x',' p_y',' p_z'))
    for i,k in enumerate(kinematic_config):
        # The complicated printout below is just so as to align the digits negative numbers with positive ones
        print '* %-108s *'%((' %-3d%s')%(i,
          ''.join([' %-25.15e'%e if j==0 or e<0.0 else '  %-24.15e'%e for j,e in enumerate(k)])))
    print '* %-108s *'%('-'*108)
    print '* %-108s *'%(' Finite part obtained for the loop matrix element (Madloop return_code=%d )'%return_code)
    print '* %-108s *'%''
    print '* %-108s *'%('      %.18e'%finite_loop_me)
    print '* %-108s *'%''
    print '='*112
