# This file was automatically created by FeynRules $Revision: 216 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Thu 22 Jul 2010 16:38:04


from __future__ import division
from object_library import all_particles, Particle

ve = Particle(pdg_code = 12,
              name = 've',
              antiname = 've~',
              spin = 2,
              color = 1,
              mass = 'ZERO',
              width = 'ZERO',
              texname = 've',
              antitexname = 've',
              line = 'straight',
              charge = 0,
              LeptonNumber = 1,
              GhostNumber = 0)

ve__tilde__ = ve.anti()

e__minus__ = Particle(pdg_code = 11,
                      name = 'e-',
                      antiname = 'e+',
                      spin = 2,
                      color = 1,
                      mass = 'ZERO',
                      width = 'ZERO',
                      texname = 'e-',
                      antitexname = 'e-',
                      line = 'straight',
                      charge = -1,
                      LeptonNumber = 1,
                      GhostNumber = 0)

e__plus__ = e__minus__.anti()

t = Particle(pdg_code = 6,
             name = 't',
             antiname = 't~',
             spin = 2,
             color = 3,
             mass = 'MT',
             width = 'WT',
             texname = 't',
             antitexname = 't',
             line = 'straight',
             charge = 2/3,
             LeptonNumber = 0,
             GhostNumber = 0)

t__tilde__ = t.anti()

b = Particle(pdg_code = 5,
             name = 'b',
             antiname = 'b~',
             spin = 2,
             color = 3,
             mass = 'MB',
             width = 'ZERO',
             texname = 'b',
             antitexname = 'b',
             line = 'straight',
             charge = -1/3,
             LeptonNumber = 0,
             GhostNumber = 0)

b__tilde__ = b.anti()

A = Particle(pdg_code = 22,
             name = 'A',
             antiname = 'A',
             spin = 3,
             color = 1,
             mass = 'ZERO',
             width = 'ZERO',
             texname = 'A',
             antitexname = 'A',
             line = 'wavy',
             charge = 0,
             LeptonNumber = 0,
             GhostNumber = 0)

W__plus__ = Particle(pdg_code = 24,
                     name = 'W+',
                     antiname = 'W-',
                     spin = 3,
                     color = 1,
                     mass = 'MW',
                     width = 'WW',
                     texname = 'W+',
                     antitexname = 'W+',
                     line = 'wavy',
                     charge = 1,
                     LeptonNumber = 0,
                     GhostNumber = 0)

W__minus__ = W__plus__.anti()
