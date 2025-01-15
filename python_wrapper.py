from ctypes import cdll, c_double, c_int
import numpy.ctypeslib as npct
import numpy as np


double_arr = npct.ndpointer(dtype=np.float64, ndim=1, flags="CONTIGUOUS")
int_arr = npct.ndpointer(dtype=np.int32, ndim=1, flags="CONTIGUOUS")


nlo = cdll.LoadLibrary('../madnis_NLO_FO.so')
print('hello')
print(dir(nlo))
nlo.madnis_nlo_initialise_.argtypes = None
nlo.madnis_nlo_initialise_.restype = None
nlo.madnis_nlo_initialise_()


ichan = np.array([0], dtype=np.int32)
nlo.madnis_get_channel_.argtypes = [int_arr]
nlo.madnis_get_channel_.restype = None
nlo.madnis_get_channel_(ichan)
print('ICHAN', ichan)


for i in range(10):
    print('ICHAN', ichan)
    x = np.random.rand(60,)
    f = np.empty(26,)
    vegas_wgt = np.array([1.])
    ifl = np.array([0], dtype=np.int32)

    nlo.madnis_nlo_evaluate_.argtypes = [double_arr, double_arr, int_arr, double_arr] 
    nlo.madnis_nlo_evaluate_.restypes = None 
    nlo.madnis_nlo_evaluate_(x,vegas_wgt,ifl,f)
    print('F', f)



nlo.madnis_nlo_terminate_.argtypes = None
nlo.madnis_nlo_terminate_.restype = None
nlo.madnis_nlo_terminate_()
print('bye')


