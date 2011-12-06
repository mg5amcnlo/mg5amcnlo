# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param
import CT_parameters as CTParam

d = Particle(pdg_code = 1,
             name = 'd',
             antiname = 'd~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'd',
             antitexname = 'd',
             charge = -1/3,
             LeptonNumber = 0,
             GhostNumber = 0)

d__tilde__ = d.anti()

u = Particle(pdg_code = 2,
             name = 'u',
             antiname = 'u~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'u',
             antitexname = 'u',
             charge = 2/3,
             LeptonNumber = 0,
             GhostNumber = 0)

u__tilde__ = u.anti()

s = Particle(pdg_code = 3,
             name = 's',
             antiname = 's~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 's',
             antitexname = 's',
             charge = -1/3,
             LeptonNumber = 0,
             GhostNumber = 0)

s__tilde__ = s.anti()

c = Particle(pdg_code = 4,
             name = 'c',
             antiname = 'c~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'c',
             antitexname = 'c',
             charge = 2/3,
             LeptonNumber = 0,
             GhostNumber = 0)

c__tilde__ = c.anti()

b = Particle(pdg_code = 5,
             name = 'b',
             antiname = 'b~',
             spin = 2,
             color = 3,
             mass = Param.MB,
             width = Param.ZERO,
             texname = 'b',
             antitexname = 'b',
             charge = -1/3,
             LeptonNumber = 0,
             GhostNumber = 0)

b__tilde__ = b.anti()

t = Particle(pdg_code = 6,
             name = 't',
             antiname = 't~',
             spin = 2,
             color = 3,
             mass = Param.MT,
             width = Param.ZERO,
             texname = 't',
             antitexname = 't',
             charge = 2/3,
             LeptonNumber = 0,
             GhostNumber = 0)

t__tilde__ = t.anti()

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
             LeptonNumber = 0,
             line='dashed',
             GhostNumber = 1)

gh__tilde__ = gh.anti()

# Wavefunction renormalization

b.loop_particles = [[[b,G]]]
b.counterterm = {(1,0):CTParam.bWcft_UV.value}

t.loop_particles = [[[t,G]]]
t.counterterm = {(1,0):CTParam.tWcft_UV.value}

G.loop_particles = [[[b],[t]]]
G.counterterm = {(1,0):CTParam.GWcft_UV_b.value,(1,1):CTParam.GWcft_UV_t.value}

