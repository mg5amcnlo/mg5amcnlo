# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param

u = Particle(pdg_code = 2,
             name = 'u',
             antiname = 'u~',
             spin = 2,
             color = 3,
             mass = Param.MU,
             width = Param.ZERO,
             texname = 'u',
             antitexname = 'u',
             charge = 2/3,
             perturbation = ['QCD',],
             LeptonNumber = 0,
             GhostNumber = 0)

u__tilde__ = u.anti()

d = Particle(pdg_code = 1,
             name = 'd',
             antiname = 'd~',
             spin = 2,
             color = 3,
             mass = Param.MD,
             width = Param.ZERO,
             texname = 'd',
             antitexname = 'd',
             charge = -1/3,
             perturbation = ['QCD',],
             LeptonNumber = 0,
             GhostNumber = 0)

d__tilde__ = d.anti()

G = Particle(pdg_code = 21,
             name = 'G',
             antiname = 'G',
             spin = 3,
             color = 8,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'G',
             antitexname = 'G',
             charge = 0,
             perturbation = ['QCD',],
             LeptonNumber = 0,
             GhostNumber = 0)

gh = Particle(pdg_code = 666,
             name = 'gh',
             antiname = 'gh~',
             spin = 1,
             color = 8,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'gh',
             antitexname = 'gh',
             charge = 0,
             perturbation = ['QCD',],
             LeptonNumber = 0,
             line='dashed',
             GhostNumber = 1)

gh__tilde__ = gh.anti()
