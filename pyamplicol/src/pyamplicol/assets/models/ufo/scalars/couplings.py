# type: ignore
from __future__ import absolute_import
from .object_library import all_couplings, Coupling


SCALAR_COUPLING = Coupling(name='SCALAR_COUPLING',
                           value='lam*complex(0,1)',
                           order={'QCD': 1})
