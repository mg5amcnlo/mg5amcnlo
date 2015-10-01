# This file was automatically created by FeynRules $Revision: 999 $
# Mathematica version: 7.0 for Linux x86 (64-bit) (February 18, 2009)
# Date: Mon 30 Jan 2012 19:57:04


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param
import CT_parameters as CTParam

A = Particle(pdg_code = 22,
             name = 'A',
             antiname = 'A',
             spin = 3,
             color = 1,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'A',
             antitexname = 'A',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0)

Z = Particle(pdg_code = 23,
             name = 'Z',
             antiname = 'Z',
             spin = 3,
             color = 1,
             mass = Param.MZ,
             width = Param.WZ,
             texname = 'Z',
             antitexname = 'Z',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0)

W__plus__ = Particle(pdg_code = 24,
                     name = 'W+',
                     antiname = 'W-',
                     spin = 3,
                     color = 1,
                     mass = Param.MW,
                     width = Param.WW,
                     texname = 'W+',
                     antitexname = 'W+',
                     charge = 1,
                     GhostNumber = 0,
                     LeptonNumber = 0)

W__minus__ = W__plus__.anti()

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
             GhostNumber = 0,
             LeptonNumber = 0)

ghA = Particle(pdg_code = 9000001,
               name = 'ghA',
               antiname = 'ghA~',
               spin = -1,
               color = 1,
               mass = Param.ZERO,
               width = Param.ZERO,
               texname = 'ghA',
               antitexname = 'ghA',
               charge = 0,
               GhostNumber = 1,
               LeptonNumber = 0)

ghA__tilde__ = ghA.anti()

ghZ = Particle(pdg_code = 9000002,
               name = 'ghZ',
               antiname = 'ghZ~',
               spin = -1,
               color = 1,
               mass = Param.MZ,
               width = Param.ZERO,
               texname = 'ghZ',
               antitexname = 'ghZ',
               charge = 0,
               GhostNumber = 1,
               LeptonNumber = 0)

ghZ__tilde__ = ghZ.anti()

ghWp = Particle(pdg_code = 9000003,
                name = 'ghWp',
                antiname = 'ghWp~',
                spin = -1,
                color = 1,
                mass = Param.MW,
                width = Param.ZERO,
                texname = 'ghWp',
                antitexname = 'ghWp',
                charge = 1,
                GhostNumber = 1,
                LeptonNumber = 0)

ghWp__tilde__ = ghWp.anti()

ghWm = Particle(pdg_code = 9000004,
                name = 'ghWm',
                antiname = 'ghWm~',
                spin = -1,
                color = 1,
                mass = Param.MW,
                width = Param.ZERO,
                texname = 'ghWm',
                antitexname = 'ghWm',
                charge = -1,
                GhostNumber = 1,
                LeptonNumber = 0)

ghWm__tilde__ = ghWm.anti()

ghG = Particle(pdg_code = 9000005,
              name = 'ghG',
              antiname = 'ghG~',
              spin = -1,
              color = 8,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 'ghG',
              antitexname = 'ghG',
              charge = 0,
              GhostNumber = 1,
              LeptonNumber = 0)

ghG__tilde__ = ghG.anti()

ve = Particle(pdg_code = 12,
              name = 've',
              antiname = 've~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 've',
              antitexname = 've',
              charge = 0,
              GhostNumber = 0,
              LeptonNumber = 1)

ve__tilde__ = ve.anti()

vm = Particle(pdg_code = 14,
              name = 'vm',
              antiname = 'vm~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 'vm',
              antitexname = 'vm',
              charge = 0,
              GhostNumber = 0,
              LeptonNumber = 1)

vm__tilde__ = vm.anti()

vt = Particle(pdg_code = 16,
              name = 'vt',
              antiname = 'vt~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 'vt',
              antitexname = 'vt',
              charge = 0,
              GhostNumber = 0,
              LeptonNumber = 1)

vt__tilde__ = vt.anti()

e__minus__ = Particle(pdg_code = 11,
                      name = 'e-',
                      antiname = 'e+',
                      spin = 2,
                      color = 1,
                      mass = Param.Me,
                      width = Param.ZERO,
                      texname = 'e-',
                      antitexname = 'e-',
                      charge = -1,
                      GhostNumber = 0,
                      LeptonNumber = 1)

e__plus__ = e__minus__.anti()

m__minus__ = Particle(pdg_code = 13,
                       name = 'm-',
                       antiname = 'm+',
                       spin = 2,
                       color = 1,
                       mass = Param.MM,
                       width = Param.ZERO,
                       texname = 'm-',
                       antitexname = 'm-',
                       charge = -1,
                       GhostNumber = 0,
                       LeptonNumber = 1)

m__plus__ = m__minus__.anti()

tt__minus__ = Particle(pdg_code = 15,
                       name = 'tt-',
                       antiname = 'tt+',
                       spin = 2,
                       color = 1,
                       mass = Param.MTA,
                       width = Param.ZERO,
                       texname = 'tt-',
                       antitexname = 'tt-',
                       charge = -1,
                       GhostNumber = 0,
                       LeptonNumber = 1)

tt__plus__ = tt__minus__.anti()

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
             GhostNumber = 0,
             LeptonNumber = 0)

u__tilde__ = u.anti()

c = Particle(pdg_code = 4,
             name = 'c',
             antiname = 'c~',
             spin = 2,
             color = 3,
             mass = Param.MC,
             width = Param.ZERO,
             texname = 'c',
             antitexname = 'c',
             charge = 2/3,
             GhostNumber = 0,
             LeptonNumber = 0)

c__tilde__ = c.anti()

t = Particle(pdg_code = 6,
             name = 't',
             antiname = 't~',
             spin = 2,
             color = 3,
             mass = Param.MT,
             width = Param.WT,
             texname = 't',
             antitexname = 't',
             charge = 2/3,
             GhostNumber = 0,
             LeptonNumber = 0)

t__tilde__ = t.anti()

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
             GhostNumber = 0,
             LeptonNumber = 0)

d__tilde__ = d.anti()

s = Particle(pdg_code = 3,
             name = 's',
             antiname = 's~',
             spin = 2,
             color = 3,
             mass = Param.MS,
             width = Param.ZERO,
             texname = 's',
             antitexname = 's',
             charge = -1/3,
             GhostNumber = 0,
             LeptonNumber = 0)

s__tilde__ = s.anti()

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
             GhostNumber = 0,
             LeptonNumber = 0)

b__tilde__ = b.anti()

H = Particle(pdg_code = 25,
             name = 'H',
             antiname = 'H',
             spin = 1,
             color = 1,
             mass = Param.MH,
             width = Param.WH,
             texname = 'H',
             antitexname = 'H',
             charge = 0,
             GhostNumber = 0,
             LeptonNumber = 0)

G0 = Particle(pdg_code = 250,
              name = 'G0',
              antiname = 'G0',
              spin = 1,
              color = 1,
              mass = Param.MZ,
              width = Param.WZ,
              texname = 'G0',
              antitexname = 'G0',
              goldstoneboson = True,
              charge = 0,
              GhostNumber = 0,
              LeptonNumber = 0)

G__plus__ = Particle(pdg_code = 251,
                     name = 'G+',
                     antiname = 'G-',
                     spin = 1,
                     color = 1,
                     mass = Param.MW,
                     width = Param.WW,
                     texname = 'G+',
                     antitexname = 'G+',
                     goldstoneboson = True,
                     charge = 1,
                     GhostNumber = 0,
                     LeptonNumber = 0)

G__minus__ = G__plus__.anti()

# Wavefunction renormalization

b.loop_particles = [[[5,21]]]
b.counterterm = {(1,0,0):CTParam.bWcft_UV.value}

t.loop_particles = [[[6,21]]]
t.counterterm = {(1,0,0):CTParam.tWcft_UV.value}

G.loop_particles = [[[5]],[[6]]]
G.counterterm = {(1,0,0):CTParam.GWcft_UV_b.value,(1,0,1):CTParam.GWcft_UV_t.value}

Param.MB.loop_particles= [[[5,21]]]
Param.MB.counterterm = {(1,0,0):CTParam.bMass_UV.value}

Param.MT.loop_particles= [[[6,21]]]
Param.MT.counterterm = {(1,0,0):CTParam.tMass_UV.value}

Param.G.loop_particles = [[[1],[2],[3],[4]],[[5]],[[6]],[[21]]],
Param.G.counterterm = {(1,0,0):CTParam.G_UVq.value,(1,0,1):CTParam.G_UVb.value,(1,0,2):CTParam.G_UVt.value,(1,0,3):CTParam.G_UVg.value}
