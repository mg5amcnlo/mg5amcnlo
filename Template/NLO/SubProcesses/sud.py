import importlib

pdg2ewsud_dict = {(-2,2,6,-6):importlib.import_module('ewsudpy')} 

import numpy as np
p_in = [[201.78976897296093, 0.0000000000000000, 0.0000000000000000, 201.78976897296093],
                 [201.78976897296093, 0.0000000000000000, 0.0000000000000000, -201.78976897296093],
                 [201.78976897296093, 84.555515249845499, -7.3837982863936427, 59.009026647284550],
                 [201.78976897296093, -84.555515249845499, 7.3837982863936427, -59.009026647284550]]

p_transp = [[pp[i] for pp in p_in] for i in range(4)]
print(p_transp)

p_arr = np.array(p_transp, order='F')

gstr=1.1587268699383517

res = np.array([0.,0.,0.], order='F')

pdg2ewsud_dict[(-2,2,6,-6)].ewsudakov(p_arr, gstr, res)

print(res)


