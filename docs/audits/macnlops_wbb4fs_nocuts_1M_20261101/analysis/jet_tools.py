"""Shared no-cut Wbb observables for first-emission LHE and full showers."""
from pathlib import Path
import ctypes as C
import json
import numpy as np

DIRECTORY=Path(__file__).resolve().parent
FIELDS=json.loads((DIRECTORY/'observables.json').read_text())['fields']
_LIBRARY=C.CDLL(str(DIRECTORY/'wbb_observables.so'))
_CALL=_LIBRARY.wbb_observables_
_CALL.argtypes=[C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_double),C.POINTER(C.c_double),C.POINTER(C.c_int)]
_CALL.restype=None

def observables(particles):
    """Input rows are (PDG,px,py,pz,E) for all final-state particles."""
    rows=np.asarray(particles,dtype=np.float64)
    ids=np.ascontiguousarray(rows[:,0],dtype=np.int32)
    momentum=np.ascontiguousarray(rows[:,1:5],dtype=np.float64)
    values=np.full(len(FIELDS),np.nan)
    n,status=C.c_int(len(rows)),C.c_int(1)
    _CALL(C.byref(n),ids.ctypes.data_as(C.POINTER(C.c_int)),momentum.ctypes.data_as(C.POINTER(C.c_double)),values.ctypes.data_as(C.POINTER(C.c_double)),C.byref(status))
    if status.value:
        raise RuntimeError('Wbb observable failure status %d, final PDGs %s' % (status.value,list(ids)))
    return values
