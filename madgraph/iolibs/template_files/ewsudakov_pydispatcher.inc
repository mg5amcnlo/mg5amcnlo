import importlib
import sys, os
import numpy as np 

# update the python path
sys.path.insert(0, '%(path)s')

# the list of P0* directories
pdir_list = [%(pdir_list)s]

def import_lib(name):
    old_dir = os.getcwd()
    sys.path.insert(0, os.path.join('%(path)s',name))
    try:
        os.chdir(os.path.join('%(path)s',name))
        out = importlib.import_module('ewsudpy_%%s' %% name)
        os.chdir(old_dir)
    except:
        os.chdir(old_dir)
        sys.path = sys.path[1:]
        raise
    return out



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

    res = pdg2ewsud_dict[pdgs].ewsudakov_py(p_transp,  g)

    return res

