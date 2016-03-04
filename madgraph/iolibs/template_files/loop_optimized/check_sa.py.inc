#! /usr/bin/env python

# This is an example of how to run MadLoop from Python using the f2py compilation of the wrapper file 'f2py_wrapper.f'.

import os
import sys
import subprocess

pjoin = os.path.join
root_path = os.path.dirname(os.path.realpath( __file__ ))
sys.path.insert(0, root_path)

try:
    import matrix2py
except:
    if os.path.isfile(pjoin(root_path,'makefile')) and \
       os.path.isfile(pjoin(root_path,'f2py_wrapper.f')) and \
       not os.path.isfile(pjoin(root_path,'matrix2py.so')):
        print "Trying to automatically generate the python module 'matrix2py.so' with f2py..."
        p = subprocess.Popen(['make','matrix2py.so'], stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, cwd=root_path)
        (out, err) = p.communicate()
        if p.returncode or not os.path.isfile(pjoin(root_path,'matrix2py.so')):
            print "ERROR: Failed to produce 'matrix2py.so' with 'make matrix2py.so' in '%s'. The error was:\n%s"%(root_path,err)
            sys.exit(0)
        try:
            import matrix2py
        except:
            print "ERROR: Could not load the f2py module 'matrix2py.so'. The following error occurred:\n",sys.exc_info()[0]
            sys.exit(0)
    else:
        if os.path.exists(pjoin(root_path,'matrix2py.so')):
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
matrix2py.initialise(os.path.abspath(pjoin(root_path,os.pardir,os.pardir,'Cards','param_card.dat')))


def invert_momenta(p):
        """ fortran/C-python do not order table in the same order"""
        new_p = []
        for i in range(len(p[0])):  new_p.append([0]*len(p))
        for i, onep in enumerate(p):
            for j, x in enumerate(onep):
                new_p[j][i] = x
        return new_p

# Now chose MadLoop's inputs for this evaluation

# The kinematic configuration in the convention (E, px, py, pz) and with particles ordered as in the process definition.
# This is here a random trial PS point.
{0}
if p[0][0] is None:
    if not os.path.isfile(pjoin(root_path,'PS.input')):
        print "\n\n==================================================================================================================="
        print "*                                           No kinematics defined!                                                *"
        print "*                                   -------------------------------------                                         *"
        print "* Please either define your kinematic configuration directly in check_sa.py or in a file 'PS.input'. Exiting now. *"
        print "===================================================================================================================\n\n"
        sys.exit(0)
    try:
        for i, line in enumerate(open(pjoin(root_path,'PS.input'),'r').readlines()):
            if i==len(p): break        
            p[i]=[float(line.split()[j]) for j in range(4)]
    except:
        print "ERROR: File PS.input is malformed. Error was:\n",sys.exc_info()[0]
        sys.exit(0)
P =invert_momenta(p)
# Alpha_s value
alphas = 0.118
# -1 means that the MadLoop averages/sum over helicity configuration. 
# If a positive integer is picked it corresponds to the position of the helicity configuration listed in the file
# 'MadLoop5_resouces/ML5_<id>_HelConfigs.dat'
nhel = -1 # means sum over all helicity
# Choice of renormalization scale
renormalization_scale = 91.188

finite_loop_me, return_code = matrix2py.get_me(P, alphas, renormalization_scale, nhel)


print '='*112
print '* %-108s *'%' MadLoop evaluation for the process '
print '* %-108s *'%'  {1}'
print '* %-108s *'%' and the kinematic configuration:'
print '* %-108s *'%((' %-3s'+' %-25s'*4)%('#','E',' p_x',' p_y',' p_z'))
for i,k in enumerate(p):
    # The complicated printout below is just so as to align the digits negative numbers with positive ones
    print '* %-108s *'%((' %-3d%s')%(i,
      ''.join([' %-25.15e'%e if j==0 or e<0.0 else '  %-24.15e'%e for j,e in enumerate(k)])))
print '* %-108s *'%('-'*108)
print '* %-108s *'%(' Finite part obtained for the loop matrix element (Madloop return_code=%d )'%return_code)
print '* %-108s *'%''
print '* %-108s *'%('      %.18e'%finite_loop_me)
print '* %-108s *'%''
print '='*112
