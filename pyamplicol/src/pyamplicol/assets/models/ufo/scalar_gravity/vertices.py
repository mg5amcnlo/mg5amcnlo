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

for i_scalar in list(range(Param.N_SCALARS)):
    Vertex(name='V_S%sS%sGr' % (i_scalar, i_scalar),
           particles=[P.scalar_particles[i_scalar],
                      P.scalar_particles[i_scalar], P.graviton],
           color=['1'],
           lorentz=[L.SSTmpart, L.SST],
           couplings={(0, 0): getattr(
               C, ("SSTmpart%d" % i_scalar)), (0, 1): C.SST}
           )
    Vertex(name='V_S%sS%sGrGr' % (i_scalar, i_scalar),
           particles=[P.scalar_particles[i_scalar],
                      P.scalar_particles[i_scalar], P.graviton, P.graviton],
           color=['1'],
           lorentz=[L.SSTTmpart, L.SSTT],
           couplings={(0, 0): getattr(
               C, ("SSTTmpart%d" % i_scalar)), (0, 1): C.SSTT}
           )
    Vertex(name='V_S%sS%sGrGrGr' % (i_scalar, i_scalar),
           particles=[P.scalar_particles[i_scalar],
                      P.scalar_particles[i_scalar], P.graviton, P.graviton, P.graviton],
           color=['1'],
           lorentz=[L.SSTTTmpart, L.SSTTT],
           couplings={(0, 0): getattr(
               C, ("SSTTTmpart%d" % i_scalar)), (0, 1): C.SSTTT}
           )

Vertex(name='V_GrGrGr',
       particles=[P.graviton, P.graviton, P.graviton],
       color=['1'],
       lorentz=[L.TTT],
       couplings={(0, 0): C.TTT}
       )

Vertex(name='V_GrGrGrGr',
       particles=[P.graviton, P.graviton, P.graviton, P.graviton],
       color=['1'],
       lorentz=[L.TTTT],
       couplings={(0, 0): C.TTTT}
       )
