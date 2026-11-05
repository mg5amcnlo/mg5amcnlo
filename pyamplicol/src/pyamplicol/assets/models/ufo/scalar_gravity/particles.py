# type: ignore
# This file was automatically created by FeynRules 1.7.69
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Mon 1 Oct 2012 14:58:25


from __future__ import division
from __future__ import absolute_import
from .object_library import all_particles, Particle  # pylint: disable=unused-import
from . import parameters as Param

scalar_particles = []
for i_scalar in range(Param.N_SCALARS):
    scalar_particles.append(Particle(pdg_code=2000+i_scalar,
                                     name='scalar_%d' % i_scalar,
                                     antiname='scalar_%d' % i_scalar,
                                     spin=1,
                                     color=1,
                                     mass=Param.scalar_mass_parameters[i_scalar],
                                     width=Param.scalar_width_parameters[i_scalar],
                                     texname='\\phi_%d' % i_scalar,
                                     antitexname='\\phi_%d' % i_scalar,
                                     charge=0,
                                     GhostNumber=0,
                                     LeptonNumber=0,
                                     Y=0))

graviton = Particle(pdg_code=1337,
                    name='graviton',
                    antiname='graviton',
                    spin=5,
                    color=1,
                    mass=Param.ZERO,
                    width=Param.ZERO,
                    texname='\\tilde{g}',
                    antitexname='\\tilde{g}',
                    charge=0,
                    GhostNumber=0,
                    LeptonNumber=0,
                    Y=0)
