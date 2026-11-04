# type: ignore
from __future__ import absolute_import
from .object_library import all_lorentz, Lorentz

from .function_library import complexconjugate, re, im, csc, sec, acsc, asec
from . import parameters as Param

try:
    import form_factors as ForFac
except ImportError:
    pass

scalar_lorentz_structures = {}
for n_point_interaction in Param.N_POINT_INTERACTIONS:
    scalar_lorentz_structures[n_point_interaction] = Lorentz(name='SCALAR_%d_LORENTZ_STRUCTURE' % n_point_interaction,
                                                             spins=[
                                                                 1, ]*n_point_interaction,
                                                             structure='1')
