from ctypes import cdll, c_double, c_int
import numpy.ctypeslib as npct
import numpy as np
import os
import sys


double_arr = npct.ndpointer(dtype=np.float64, ndim=1, flags="CONTIGUOUS")
int_arr = npct.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")


nlo = cdll.LoadLibrary('../libmadnis_nlo.so')
nlo.madgraph_nlo_init.argtypes = None
nlo.madgraph_nlo_init.restype = None
nlo.madgraph_nlo_init()

nlo.call_magraph_nlo.argtypes = [
            double_arr,
            c_int,
            c_int,
            double_arr,
            int_arr,
        ]
nlo.call_magraph_nlo.restype = None


nbatch = 10000
ndim = 7
vegas_weight = np.ones((nbatch,))
x = np.random.rand(nbatch * ndim).astype(np.float64)
   
# prepare outputs
w_out = np.empty((nbatch,), dtype=np.float64)    
chan_out = np.empty((nbatch,), dtype=np.int32)   

nlo.GetNChannels.argtype = None
nlo.GetNChannels.restype = c_int
# check number of channels
nchans = nlo.GetNChannels()
print(f'number of channels: {nchans}' )
   
# call api
nlo.call_magraph_nlo(x, nbatch, ndim, w_out, chan_out)


sigma = w_out.mean(axis=0)
error = np.sqrt(w_out.var(axis=0)/nbatch)
print(f'\n===============================' )
print(f'{sigma} +- {error} pb' )
print(f'===============================' )

# # nlo.madnis_nlo_terminate_.argtypes = None
# # nlo.madnis_nlo_terminate_.restype = None
# # nlo.madnis_nlo_terminate_()