# type: ignore
# This file was automatically created by FeynRules 1.7.69
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Mon 1 Oct 2012 14:58:25


from __future__ import absolute_import
from .object_library import all_vertices, Vertex
from . import particles as P
from . import couplings as C
from . import lorentz as L
from . import parameters as Param
import itertools

for n_point_interaction in Param.N_POINT_INTERACTIONS:
    for i_inter, interaction in enumerate(sorted(list(set(tuple(sorted(inter)) for inter in itertools.product(*[list(range(Param.N_SCALARS)),]*n_point_interaction))))):
        # Remember that the constructor automatically adds the objects to all_vertices
        Vertex(name='V_%d_SCALAR_%s' % (n_point_interaction, ''.join('%d' % scalar_id for scalar_id in interaction)),
               particles=[P.scalar_particles[scalar_id]
                          for scalar_id in interaction],
               color=['1'],
               lorentz=[L.scalar_lorentz_structures[n_point_interaction]],
               couplings={(0, 0): C.SCALAR_COUPLING})
