import importlib
import sys
import numpy as np 

# update the python path
sys.path.insert(0, '%(path)s')

pdir_list = [%(pdir_list)s]

pdg2ewsud_dict = {%(pdg2sud)s}


def ewsudakov(pdgs, p, g):
    """wraps the call to the ewsudakov function of a 
    given directory"""
    res = np.array([0.,0.,0.], order='F')

    pdg2ewsud_dict[pdgs].ewsudakov(p, g, res)

    return res
