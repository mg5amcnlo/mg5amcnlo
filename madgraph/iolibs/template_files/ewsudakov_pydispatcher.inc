import importlib
import sys
import numpy as np 

# update the python path
sys.path.insert(0, '%(path)s')

pdir_list = [%(pdir_list)s]

pdg2ewsud_dict = {%(pdg2sud)s}


def ewsudakov(pdgs, p, g):
    """wraps the call to the ewsudakov function of a 
    given directory. The momenta are also transposed 
    to match the fortran<>python convention"""

    p_transp =np.array([[pp[i] for pp in p] for i in range(4)], order='F')
    res = np.array([0.,0.,0.], order='F')

    this_pdg = 0
    n_inc = len(pdgs[0])
    for pdg in pdg2ewsud_dict:
        if pdgs[0] == pdg[0:n_inc]:
            if sorted(list(pdgs[1])) == sorted(list(pdg[n_inc:])):
                pdg2ewsud_dict[pdg].ewsudakov_py(p_transp,  g, res)
                this_pdg = pdg

    return res,this_pdg

