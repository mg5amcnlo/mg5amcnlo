import importlib
import sys
import numpy as np 

# update the python path
sys.path.insert(0, '%(path)s')

# the list of P0* directories
pdir_list = [%(pdir_list)s]

# a dictionary which maps pdgs, in the format ((initial), (sorted final))
# and the corresponding f2py modules
pdg2ewsud_dict = {%(pdg2sud)s}

# a dictionary which maps pdgs, in the format ((initial), (sorted final))
# and the unsorted version (corresponding to the ordering in the Pdirs) 
original_pdg_list_dict = {%(pdgsorted)s}


def ewsudakov(pdgs, p, g):
    """wraps the call to the ewsudakov function of a 
    given directory. The momenta are also transposed 
    to match the fortran<>python convention"""

    p_transp =np.array([[pp[i] for pp in p] for i in range(4)], order='F')
    res = np.array([0.,0.,0.,0.,0.,0.], order='F')

    pdg2ewsud_dict[pdgs].ewsudakov_py(p_transp,  g, res)

    return res

