# This file was automatically created by FeynRules $Revision: 356 $
# Mathematica version: 7.0 for Microsoft Windows (32-bit) (November 10, 2008)
# Date: Tue 26 Oct 2010 21:29:11


from __future__ import division
from object_library import all_particles, Particle

ve = Particle(pdg_code = 12,
              name = 've',
              antiname = 've~',
              spin = 2,
              color = 1,
              mass = 'Mve',
              width = 'ZERO',
              texname = 've',
              antitexname = 've',
              line = 'straight',
              charge = 0,
              LeptonNumber = 1,
              GhostNumber = 0)

ve__tilde__ = ve.anti()

vm = Particle(pdg_code = 14,
              name = 'vm',
              antiname = 'vm~',
              spin = 2,
              color = 1,
              mass = 'Mvm',
              width = 'ZERO',
              texname = 'vm',
              antitexname = 'vm',
              line = 'straight',
              charge = 0,
              LeptonNumber = 1,
              GhostNumber = 0)

vm__tilde__ = vm.anti()

vt = Particle(pdg_code = 16,
              name = 'vt',
              antiname = 'vt~',
              spin = 2,
              color = 1,
              mass = 'Mvt',
              width = 'ZERO',
              texname = 'vt',
              antitexname = 'vt',
              line = 'straight',
              charge = 0,
              LeptonNumber = 1,
              GhostNumber = 0)

vt__tilde__ = vt.anti()

e__minus__ = Particle(pdg_code = 11,
                      name = 'e-',
                      antiname = 'e+',
                      spin = 2,
                      color = 1,
                      mass = 'Me',
                      width = 'ZERO',
                      texname = 'e-',
                      antitexname = 'e-',
                      line = 'straight',
                      charge = -1,
                      LeptonNumber = 1,
                      GhostNumber = 0)

e__plus__ = e__minus__.anti()

m__minus__ = Particle(pdg_code = 13,
                      name = 'm-',
                      antiname = 'm+',
                      spin = 2,
                      color = 1,
                      mass = 'MM',
                      width = 'ZERO',
                      texname = 'm-',
                      antitexname = 'm-',
                      line = 'straight',
                      charge = -1,
                      LeptonNumber = 1,
                      GhostNumber = 0)

m__plus__ = m__minus__.anti()

tt__minus__ = Particle(pdg_code = 15,
                       name = 'tt-',
                       antiname = 'tt+',
                       spin = 2,
                       color = 1,
                       mass = 'MTA',
                       width = 'ZERO',
                       texname = 'tt-',
                       antitexname = 'tt-',
                       line = 'straight',
                       charge = -1,
                       LeptonNumber = 1,
                       GhostNumber = 0)

tt__plus__ = tt__minus__.anti()

u = Particle(pdg_code = 2,
             name = 'u',
             antiname = 'u~',
             spin = 2,
             color = 3,
             mass = 'MU',
             width = 'ZERO',
             texname = 'u',
             antitexname = 'u',
             line = 'straight',
             charge = 2/3,
             LeptonNumber = 0,
             GhostNumber = 0)

u__tilde__ = u.anti()

c = Particle(pdg_code = 4,
             name = 'c',
             antiname = 'c~',
             spin = 2,
             color = 3,
             mass = 'MC',
             width = 'ZERO',
             texname = 'c',
             antitexname = 'c',
             line = 'straight',
             charge = 2/3,
             LeptonNumber = 0,
             GhostNumber = 0)

c__tilde__ = c.anti()

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

d = Particle(pdg_code = 1,
             name = 'd',
             antiname = 'd~',
             spin = 2,
             color = 3,
             mass = 'MD',
             width = 'ZERO',
             texname = 'd',
             antitexname = 'd',
             line = 'straight',
             charge = -1/3,
             LeptonNumber = 0,
             GhostNumber = 0)

d__tilde__ = d.anti()

s = Particle(pdg_code = 3,
             name = 's',
             antiname = 's~',
             spin = 2,
             color = 3,
             mass = 'MS',
             width = 'ZERO',
             texname = 's',
             antitexname = 's',
             line = 'straight',
             charge = -1/3,
             LeptonNumber = 0,
             GhostNumber = 0)

s__tilde__ = s.anti()

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

ghA = Particle(pdg_code = 9000001,
               name = 'ghA',
               antiname = 'ghA~',
               spin = -1,
               color = 1,
               mass = 'ZERO',
               width = 'WghA',
               texname = 'ghA',
               antitexname = 'ghA',
               line = 'dotted',
               charge = 0,
               LeptonNumber = 0,
               GhostNumber = 1)

ghA__tilde__ = ghA.anti()

ghZ = Particle(pdg_code = 9000002,
               name = 'ghZ',
               antiname = 'ghZ~',
               spin = -1,
               color = 1,
               mass = 'MZ',
               width = 'WghZ',
               texname = 'ghZ',
               antitexname = 'ghZ',
               line = 'dotted',
               charge = 0,
               LeptonNumber = 0,
               GhostNumber = 1)

ghZ__tilde__ = ghZ.anti()

ghWp = Particle(pdg_code = 9000003,
                name = 'ghWp',
                antiname = 'ghWp~',
                spin = -1,
                color = 1,
                mass = 'MW',
                width = 'WghWp',
                texname = 'ghWp',
                antitexname = 'ghWp',
                line = 'dotted',
                charge = 1,
                LeptonNumber = 0,
                GhostNumber = 1)

ghWp__tilde__ = ghWp.anti()

ghWm = Particle(pdg_code = 9000004,
                name = 'ghWm',
                antiname = 'ghWm~',
                spin = -1,
                color = 1,
                mass = 'MW',
                width = 'WghWm',
                texname = 'ghWm',
                antitexname = 'ghWm',
                line = 'dotted',
                charge = -1,
                LeptonNumber = 0,
                GhostNumber = 1)

ghWm__tilde__ = ghWm.anti()

ghG = Particle(pdg_code = 9000005,
               name = 'ghG',
               antiname = 'ghG~',
               spin = -1,
               color = 8,
               mass = 'ZERO',
               width = 'WghG',
               texname = 'ghG',
               antitexname = 'ghG',
               line = 'dotted',
               charge = 0,
               LeptonNumber = 0,
               GhostNumber = 1)

ghG__tilde__ = ghG.anti()

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

Z = Particle(pdg_code = 23,
             name = 'Z',
             antiname = 'Z',
             spin = 3,
             color = 1,
             mass = 'MZ',
             width = 'WZ',
             texname = 'Z',
             antitexname = 'Z',
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

G = Particle(pdg_code = 21,
             name = 'G',
             antiname = 'G',
             spin = 3,
             color = 8,
             mass = 'ZERO',
             width = 'ZERO',
             texname = 'G',
             antitexname = 'G',
             line = 'curly',
             charge = 0,
             LeptonNumber = 0,
             GhostNumber = 0)

H = Particle(pdg_code = 25,
             name = 'H',
             antiname = 'H',
             spin = 1,
             color = 1,
             mass = 'MH',
             width = 'WH',
             texname = '\\phi',
             antitexname = '\\phi',
             line = 'dashed',
             charge = 0,
             LeptonNumber = 0,
             GhostNumber = 0)

phi0 = Particle(pdg_code = 250,
                name = 'phi0',
                antiname = 'phi0',
                spin = 1,
                color = 1,
                mass = 'MZ',
                width = 'Wphi',
                texname = 'phi0',
                antitexname = 'phi0',
                line = 'dashed',
                GoldstoneBoson = True,
                charge = 0,
                LeptonNumber = 0,
                GhostNumber = 0)

phi__plus__ = Particle(pdg_code = 251,
                       name = 'phi+',
                       antiname = 'phi-',
                       spin = 1,
                       color = 1,
                       mass = 'MW',
                       width = 'Wphi2',
                       texname = '\\phi^+',
                       antitexname = '\\phi^+',
                       line = 'dashed',
                       GoldstoneBoson = True,
                       charge = 1,
                       LeptonNumber = 0,
                       GhostNumber = 0)

phi__minus__ = phi__plus__.anti()

