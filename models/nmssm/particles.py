# This file was automatically created by FeynRules 1.7.214
# Mathematica version: 9.0 for Mac OS X x86 (64-bit) (January 24, 2013)
# Date: Thu 22 Aug 2013 18:21:53


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param

import propagators as Prop

a = Particle(pdg_code = 22,
             name = 'a',
             antiname = 'a',
             spin = 3,
             color = 1,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'a',
             antitexname = 'a',
             charge = 0,
             GhostNumber = 0,
             Y = 0)

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
             Y = 0)

W__plus__ = Particle(pdg_code = 24,
                     name = 'W+',
                     antiname = 'W-',
                     spin = 3,
                     color = 1,
                     mass = Param.MW,
                     width = Param.WW,
                     texname = 'W+',
                     antitexname = 'W-',
                     charge = 1,
                     GhostNumber = 0,
                     Y = 0)

W__minus__ = W__plus__.anti()

g = Particle(pdg_code = 21,
             name = 'g',
             antiname = 'g',
             spin = 3,
             color = 8,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'g',
             antitexname = 'g',
             charge = 0,
             GhostNumber = 0,
             Y = 0)

n1 = Particle(pdg_code = 1000022,
              name = 'n1',
              antiname = 'n1',
              spin = 2,
              color = 1,
              mass = Param.Mneu1,
              width = Param.ZERO,
              texname = 'n1',
              antitexname = 'n1',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

n2 = Particle(pdg_code = 1000023,
              name = 'n2',
              antiname = 'n2',
              spin = 2,
              color = 1,
              mass = Param.Mneu2,
              width = Param.Wneu2,
              texname = 'n2',
              antitexname = 'n2',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

n3 = Particle(pdg_code = 1000025,
              name = 'n3',
              antiname = 'n3',
              spin = 2,
              color = 1,
              mass = Param.Mneu3,
              width = Param.Wneu3,
              texname = 'n3',
              antitexname = 'n3',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

n4 = Particle(pdg_code = 1000035,
              name = 'n4',
              antiname = 'n4',
              spin = 2,
              color = 1,
              mass = Param.Mneu4,
              width = Param.Wneu4,
              texname = 'n4',
              antitexname = 'n4',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

n5 = Particle(pdg_code = 1000045,
              name = 'n5',
              antiname = 'n5',
              spin = 2,
              color = 1,
              mass = Param.Mneu5,
              width = Param.Wneu5,
              texname = 'n5',
              antitexname = 'n5',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

x1__plus__ = Particle(pdg_code = 1000024,
                      name = 'x1+',
                      antiname = 'x1-',
                      spin = 2,
                      color = 1,
                      mass = Param.Mch1,
                      width = Param.Wch1,
                      texname = 'x1+',
                      antitexname = 'x1-',
                      charge = 1,
                      GhostNumber = 0,
                      Y = 0)

x1__minus__ = x1__plus__.anti()

x2__plus__ = Particle(pdg_code = 1000037,
                      name = 'x2+',
                      antiname = 'x2-',
                      spin = 2,
                      color = 1,
                      mass = Param.Mch2,
                      width = Param.Wch2,
                      texname = 'x2+',
                      antitexname = 'x2-',
                      charge = 1,
                      GhostNumber = 0,
                      Y = 0)

x2__minus__ = x2__plus__.anti()

go = Particle(pdg_code = 1000021,
              name = 'go',
              antiname = 'go',
              spin = 2,
              color = 8,
              mass = Param.Mgo,
              width = Param.Wgo,
              texname = 'go',
              antitexname = 'go',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

h01 = Particle(pdg_code = 25,
               name = 'h01',
               antiname = 'h01',
               spin = 1,
               color = 1,
               mass = Param.MH01,
               width = Param.WH01,
               texname = 'h01',
               antitexname = 'h01',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

h02 = Particle(pdg_code = 35,
               name = 'h02',
               antiname = 'h02',
               spin = 1,
               color = 1,
               mass = Param.MH02,
               width = Param.WH02,
               texname = 'h02',
               antitexname = 'h02',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

h03 = Particle(pdg_code = 45,
               name = 'h03',
               antiname = 'h03',
               spin = 1,
               color = 1,
               mass = Param.MH03,
               width = Param.WH03,
               texname = 'h03',
               antitexname = 'h03',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

a01 = Particle(pdg_code = 36,
               name = 'a01',
               antiname = 'a01',
               spin = 1,
               color = 1,
               mass = Param.MxA01,
               width = Param.WA01,
               texname = 'a01',
               antitexname = 'a01',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

a02 = Particle(pdg_code = 46,
               name = 'a02',
               antiname = 'a02',
               spin = 1,
               color = 1,
               mass = Param.MxA02,
               width = Param.WA02,
               texname = 'a02',
               antitexname = 'a02',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

H__plus__ = Particle(pdg_code = 37,
                     name = 'H+',
                     antiname = 'H-',
                     spin = 1,
                     color = 1,
                     mass = Param.MH,
                     width = Param.WH,
                     texname = 'H+',
                     antitexname = 'H-',
                     charge = 1,
                     GhostNumber = 0,
                     Y = 0)

H__minus__ = H__plus__.anti()

G0 = Particle(pdg_code = 250,
              name = 'G0',
              antiname = 'G0',
              spin = 1,
              color = 1,
              mass = Param.MZ,
              width = Param.ZERO,
              texname = 'G0',
              antitexname = 'G0',
              goldstone = True,
              charge = 0,
              GhostNumber = 0,
              Y = 0)

G__plus__ = Particle(pdg_code = 251,
                     name = 'G+',
                     antiname = 'G-',
                     spin = 1,
                     color = 1,
                     mass = Param.MW,
                     width = Param.ZERO,
                     texname = 'G+',
                     antitexname = 'G-',
                     goldstone = True,
                     charge = 1,
                     GhostNumber = 0,
                     Y = 0)

G__minus__ = G__plus__.anti()

ve = Particle(pdg_code = 12,
              name = 've',
              antiname = 've~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 've',
              antitexname = 've~',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

ve__tilde__ = ve.anti()

vm = Particle(pdg_code = 14,
              name = 'vm',
              antiname = 'vm~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 'vm',
              antitexname = 'vm~',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

vm__tilde__ = vm.anti()

vt = Particle(pdg_code = 16,
              name = 'vt',
              antiname = 'vt~',
              spin = 2,
              color = 1,
              mass = Param.ZERO,
              width = Param.ZERO,
              texname = 'vt',
              antitexname = 'vt~',
              charge = 0,
              GhostNumber = 0,
              Y = 0)

vt__tilde__ = vt.anti()

e__minus__ = Particle(pdg_code = 11,
                      name = 'e-',
                      antiname = 'e+',
                      spin = 2,
                      color = 1,
                      mass = Param.ZERO,
                      width = Param.ZERO,
                      texname = 'e-',
                      antitexname = 'e+',
                      charge = -1,
                      GhostNumber = 0,
                      Y = 0)

e__plus__ = e__minus__.anti()

mu__minus__ = Particle(pdg_code = 13,
                       name = 'mu-',
                       antiname = 'mu+',
                       spin = 2,
                       color = 1,
                       mass = Param.ZERO,
                       width = Param.ZERO,
                       texname = 'mu-',
                       antitexname = 'mu+',
                       charge = -1,
                       GhostNumber = 0,
                       Y = 0)

mu__plus__ = mu__minus__.anti()

tau__minus__ = Particle(pdg_code = 15,
                        name = 'tau-',
                        antiname = 'tau+',
                        spin = 2,
                        color = 1,
                        mass = Param.Mta,
                        width = Param.ZERO,
                        texname = 'tau-',
                        antitexname = 'tau+',
                        charge = -1,
                        GhostNumber = 0,
                        Y = 0)

tau__plus__ = tau__minus__.anti()

u = Particle(pdg_code = 2,
             name = 'u',
             antiname = 'u~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'u',
             antitexname = 'u~',
             charge = 2/3,
             GhostNumber = 0,
             Y = 0)

u__tilde__ = u.anti()

c = Particle(pdg_code = 4,
             name = 'c',
             antiname = 'c~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'c',
             antitexname = 'c~',
             charge = 2/3,
             GhostNumber = 0,
             Y = 0)

c__tilde__ = c.anti()

t = Particle(pdg_code = 6,
             name = 't',
             antiname = 't~',
             spin = 2,
             color = 3,
             mass = Param.MT,
             width = Param.WT,
             texname = 't',
             antitexname = 't~',
             charge = 2/3,
             GhostNumber = 0,
             Y = 0)

t__tilde__ = t.anti()

d = Particle(pdg_code = 1,
             name = 'd',
             antiname = 'd~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 'd',
             antitexname = 'd~',
             charge = -1/3,
             GhostNumber = 0,
             Y = 0)

d__tilde__ = d.anti()

s = Particle(pdg_code = 3,
             name = 's',
             antiname = 's~',
             spin = 2,
             color = 3,
             mass = Param.ZERO,
             width = Param.ZERO,
             texname = 's',
             antitexname = 's~',
             charge = -1/3,
             GhostNumber = 0,
             Y = 0)

s__tilde__ = s.anti()

b = Particle(pdg_code = 5,
             name = 'b',
             antiname = 'b~',
             spin = 2,
             color = 3,
             mass = Param.MB,
             width = Param.ZERO,
             texname = 'b',
             antitexname = 'b~',
             charge = -1/3,
             GhostNumber = 0,
             Y = 0)

b__tilde__ = b.anti()

sv1 = Particle(pdg_code = 1000012,
               name = 'sv1',
               antiname = 'sv1~',
               spin = 1,
               color = 1,
               mass = Param.Msn1,
               width = Param.Wsn1,
               texname = 'sv1',
               antitexname = 'sv1~',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

sv1__tilde__ = sv1.anti()

sv2 = Particle(pdg_code = 1000014,
               name = 'sv2',
               antiname = 'sv2~',
               spin = 1,
               color = 1,
               mass = Param.Msn2,
               width = Param.Wsn2,
               texname = 'sv2',
               antitexname = 'sv2~',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

sv2__tilde__ = sv2.anti()

sv3 = Particle(pdg_code = 1000016,
               name = 'sv3',
               antiname = 'sv3~',
               spin = 1,
               color = 1,
               mass = Param.Msn3,
               width = Param.Wsn3,
               texname = 'sv3',
               antitexname = 'sv3~',
               charge = 0,
               GhostNumber = 0,
               Y = 0)

sv3__tilde__ = sv3.anti()

sl1__minus__ = Particle(pdg_code = 1000011,
                        name = 'sl1-',
                        antiname = 'sl1+',
                        spin = 1,
                        color = 1,
                        mass = Param.Msl1,
                        width = Param.Wsl1,
                        texname = 'sl1-',
                        antitexname = 'sl1+',
                        charge = -1,
                        GhostNumber = 0,
                        Y = 0)

sl1__plus__ = sl1__minus__.anti()

sl2__minus__ = Particle(pdg_code = 1000013,
                        name = 'sl2-',
                        antiname = 'sl2+',
                        spin = 1,
                        color = 1,
                        mass = Param.Msl2,
                        width = Param.Wsl2,
                        texname = 'sl2-',
                        antitexname = 'sl2+',
                        charge = -1,
                        GhostNumber = 0,
                        Y = 0)

sl2__plus__ = sl2__minus__.anti()

sl3__minus__ = Particle(pdg_code = 1000015,
                        name = 'sl3-',
                        antiname = 'sl3+',
                        spin = 1,
                        color = 1,
                        mass = Param.Msl3,
                        width = Param.Wsl3,
                        texname = 'sl3-',
                        antitexname = 'sl3+',
                        charge = -1,
                        GhostNumber = 0,
                        Y = 0)

sl3__plus__ = sl3__minus__.anti()

sl4__minus__ = Particle(pdg_code = 2000011,
                        name = 'sl4-',
                        antiname = 'sl4+',
                        spin = 1,
                        color = 1,
                        mass = Param.Msl4,
                        width = Param.Wsl4,
                        texname = 'sl4-',
                        antitexname = 'sl4+',
                        charge = -1,
                        GhostNumber = 0,
                        Y = 0)

sl4__plus__ = sl4__minus__.anti()

sl5__minus__ = Particle(pdg_code = 2000013,
                        name = 'sl5-',
                        antiname = 'sl5+',
                        spin = 1,
                        color = 1,
                        mass = Param.Msl5,
                        width = Param.Wsl5,
                        texname = 'sl5-',
                        antitexname = 'sl5+',
                        charge = -1,
                        GhostNumber = 0,
                        Y = 0)

sl5__plus__ = sl5__minus__.anti()

sl6__minus__ = Particle(pdg_code = 2000015,
                        name = 'sl6-',
                        antiname = 'sl6+',
                        spin = 1,
                        color = 1,
                        mass = Param.Msl6,
                        width = Param.Wsl6,
                        texname = 'sl6-',
                        antitexname = 'sl6+',
                        charge = -1,
                        GhostNumber = 0,
                        Y = 0)

sl6__plus__ = sl6__minus__.anti()

su1 = Particle(pdg_code = 1000002,
               name = 'su1',
               antiname = 'su1~',
               spin = 1,
               color = 3,
               mass = Param.Msu1,
               width = Param.Wsu1,
               texname = 'su1',
               antitexname = 'su1~',
               charge = 2/3,
               GhostNumber = 0,
               Y = 0)

su1__tilde__ = su1.anti()

su2 = Particle(pdg_code = 1000004,
               name = 'su2',
               antiname = 'su2~',
               spin = 1,
               color = 3,
               mass = Param.Msu2,
               width = Param.Wsu2,
               texname = 'su2',
               antitexname = 'su2~',
               charge = 2/3,
               GhostNumber = 0,
               Y = 0)

su2__tilde__ = su2.anti()

su3 = Particle(pdg_code = 1000006,
               name = 'su3',
               antiname = 'su3~',
               spin = 1,
               color = 3,
               mass = Param.Msu3,
               width = Param.Wsu3,
               texname = 'su3',
               antitexname = 'su3~',
               charge = 2/3,
               GhostNumber = 0,
               Y = 0)

su3__tilde__ = su3.anti()

su4 = Particle(pdg_code = 2000002,
               name = 'su4',
               antiname = 'su4~',
               spin = 1,
               color = 3,
               mass = Param.Msu4,
               width = Param.Wsu4,
               texname = 'su4',
               antitexname = 'su4~',
               charge = 2/3,
               GhostNumber = 0,
               Y = 0)

su4__tilde__ = su4.anti()

su5 = Particle(pdg_code = 2000004,
               name = 'su5',
               antiname = 'su5~',
               spin = 1,
               color = 3,
               mass = Param.Msu5,
               width = Param.Wsu5,
               texname = 'su5',
               antitexname = 'su5~',
               charge = 2/3,
               GhostNumber = 0,
               Y = 0)

su5__tilde__ = su5.anti()

su6 = Particle(pdg_code = 2000006,
               name = 'su6',
               antiname = 'su6~',
               spin = 1,
               color = 3,
               mass = Param.Msu6,
               width = Param.Wsu6,
               texname = 'su6',
               antitexname = 'su6~',
               charge = 2/3,
               GhostNumber = 0,
               Y = 0)

su6__tilde__ = su6.anti()

sd1 = Particle(pdg_code = 1000001,
               name = 'sd1',
               antiname = 'sd1~',
               spin = 1,
               color = 3,
               mass = Param.Msd1,
               width = Param.Wsd1,
               texname = 'sd1',
               antitexname = 'sd1~',
               charge = -1/3,
               GhostNumber = 0,
               Y = 0)

sd1__tilde__ = sd1.anti()

sd2 = Particle(pdg_code = 1000003,
               name = 'sd2',
               antiname = 'sd2~',
               spin = 1,
               color = 3,
               mass = Param.Msd2,
               width = Param.Wsd2,
               texname = 'sd2',
               antitexname = 'sd2~',
               charge = -1/3,
               GhostNumber = 0,
               Y = 0)

sd2__tilde__ = sd2.anti()

sd3 = Particle(pdg_code = 1000005,
               name = 'sd3',
               antiname = 'sd3~',
               spin = 1,
               color = 3,
               mass = Param.Msd3,
               width = Param.Wsd3,
               texname = 'sd3',
               antitexname = 'sd3~',
               charge = -1/3,
               GhostNumber = 0,
               Y = 0)

sd3__tilde__ = sd3.anti()

sd4 = Particle(pdg_code = 2000001,
               name = 'sd4',
               antiname = 'sd4~',
               spin = 1,
               color = 3,
               mass = Param.Msd4,
               width = Param.Wsd4,
               texname = 'sd4',
               antitexname = 'sd4~',
               charge = -1/3,
               GhostNumber = 0,
               Y = 0)

sd4__tilde__ = sd4.anti()

sd5 = Particle(pdg_code = 2000003,
               name = 'sd5',
               antiname = 'sd5~',
               spin = 1,
               color = 3,
               mass = Param.Msd5,
               width = Param.Wsd5,
               texname = 'sd5',
               antitexname = 'sd5~',
               charge = -1/3,
               GhostNumber = 0,
               Y = 0)

sd5__tilde__ = sd5.anti()

sd6 = Particle(pdg_code = 2000005,
               name = 'sd6',
               antiname = 'sd6~',
               spin = 1,
               color = 3,
               mass = Param.Msd6,
               width = Param.Wsd6,
               texname = 'sd6',
               antitexname = 'sd6~',
               charge = -1/3,
               GhostNumber = 0,
               Y = 0)

sd6__tilde__ = sd6.anti()

ghG = Particle(pdg_code = 9000001,
               name = 'ghG',
               antiname = 'ghG~',
               spin = -1,
               color = 8,
               mass = Param.ZERO,
               width = Param.ZERO,
               texname = 'ghG',
               antitexname = 'ghG~',
               charge = 0,
               GhostNumber = 1,
               Y = 0)

ghG__tilde__ = ghG.anti()

ghA = Particle(pdg_code = 9000002,
               name = 'ghA',
               antiname = 'ghA~',
               spin = -1,
               color = 1,
               mass = Param.ZERO,
               width = Param.ZERO,
               texname = 'ghA',
               antitexname = 'ghA~',
               charge = 0,
               GhostNumber = 1,
               Y = 0)

ghA__tilde__ = ghA.anti()

ghZ = Particle(pdg_code = 9000003,
               name = 'ghZ',
               antiname = 'ghZ~',
               spin = -1,
               color = 1,
               mass = Param.MZ,
               width = Param.WZ,
               texname = 'ghZ',
               antitexname = 'ghZ~',
               charge = 0,
               GhostNumber = 1,
               Y = 0)

ghZ__tilde__ = ghZ.anti()

ghWp = Particle(pdg_code = 9000004,
                name = 'ghWp',
                antiname = 'ghWp~',
                spin = -1,
                color = 1,
                mass = Param.MW,
                width = Param.WW,
                texname = 'ghWp',
                antitexname = 'ghWp~',
                charge = 1,
                GhostNumber = 1,
                Y = 0)

ghWp__tilde__ = ghWp.anti()

ghWm = Particle(pdg_code = 9000005,
                name = 'ghWm',
                antiname = 'ghWm~',
                spin = -1,
                color = 1,
                mass = Param.MW,
                width = Param.WW,
                texname = 'ghWm',
                antitexname = 'ghWm~',
                charge = -1,
                GhostNumber = 1,
                Y = 0)

ghWm__tilde__ = ghWm.anti()

