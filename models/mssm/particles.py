# This file was automatically created by FeynRules $Revision: 216 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Wed 11 Aug 2010 14:22:44


from __future__ import division
from object_library import all_particles, Particle

ve = Particle(pdg_code = 12,
              name = 've',
              antiname = 've~',
              spin = 2,
              color = 1,
              mass = 'Mnue',
              width = 'ZERO',
              texname = 've',
              antitexname = 've',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

ve__tilde__ = ve.anti()

vm = Particle(pdg_code = 14,
              name = 'vm',
              antiname = 'vm~',
              spin = 2,
              color = 1,
              mass = 'Mnum',
              width = 'ZERO',
              texname = 'vm',
              antitexname = 'vm',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

vm__tilde__ = vm.anti()

vt = Particle(pdg_code = 16,
              name = 'vt',
              antiname = 'vt~',
              spin = 2,
              color = 1,
              mass = 'Mnut',
              width = 'ZERO',
              texname = 'vt',
              antitexname = 'vt',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

vt__tilde__ = vt.anti()

e__minus__ = Particle(pdg_code = 11,
                      name = 'e-',
                      antiname = 'e+',
                      spin = 2,
                      color = 1,
                      mass = 'ME',
                      width = 'ZERO',
                      texname = 'e-',
                      antitexname = 'e-',
                      line = 'straight',
                      charge = -1,
                      GhostNumber = 0)

e__plus__ = e__minus__.anti()

mu__minus__ = Particle(pdg_code = 13,
                       name = 'mu-',
                       antiname = 'mu+',
                       spin = 2,
                       color = 1,
                       mass = 'MM',
                       width = 'ZERO',
                       texname = 'mu-',
                       antitexname = 'mu-',
                       line = 'straight',
                       charge = -1,
                       GhostNumber = 0)

mu__plus__ = mu__minus__.anti()

tau__minus__ = Particle(pdg_code = 15,
                        name = 'tau-',
                        antiname = 'tau+',
                        spin = 2,
                        color = 1,
                        mass = 'MTA',
                        width = 'ZERO',
                        texname = 'tau-',
                        antitexname = 'tau-',
                        line = 'straight',
                        charge = -1,
                        GhostNumber = 0)

tau__plus__ = tau__minus__.anti()

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
             GhostNumber = 0)

b__tilde__ = b.anti()

n1 = Particle(pdg_code = 1000022,
              name = 'n1',
              antiname = 'n1',
              spin = 2,
              color = 1,
              mass = 'Mneu1',
              width = 'Wneu1',
              texname = 'n1',
              antitexname = 'n1',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

n2 = Particle(pdg_code = 1000023,
              name = 'n2',
              antiname = 'n2',
              spin = 2,
              color = 1,
              mass = 'Mneu2',
              width = 'Wneu2',
              texname = 'n2',
              antitexname = 'n2',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

n3 = Particle(pdg_code = 1000025,
              name = 'n3',
              antiname = 'n3',
              spin = 2,
              color = 1,
              mass = 'Mneu3',
              width = 'Wneu3',
              texname = 'n3',
              antitexname = 'n3',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

n4 = Particle(pdg_code = 1000035,
              name = 'n4',
              antiname = 'n4',
              spin = 2,
              color = 1,
              mass = 'Mneu4',
              width = 'Wneu4',
              texname = 'n4',
              antitexname = 'n4',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

x1__plus__ = Particle(pdg_code = 1000024,
                      name = 'x1+',
                      antiname = 'x1-',
                      spin = 2,
                      color = 1,
                      mass = 'Mch1',
                      width = 'Wch1',
                      texname = 'x1+',
                      antitexname = 'x1+',
                      line = 'straight',
                      charge = 1,
                      GhostNumber = 0)

x1__minus__ = x1__plus__.anti()

x2__plus__ = Particle(pdg_code = 1000037,
                      name = 'x2+',
                      antiname = 'x2-',
                      spin = 2,
                      color = 1,
                      mass = 'Mch2',
                      width = 'Wch2',
                      texname = 'x2+',
                      antitexname = 'x2+',
                      line = 'straight',
                      charge = 1,
                      GhostNumber = 0)

x2__minus__ = x2__plus__.anti()

go = Particle(pdg_code = 1000021,
              name = 'go',
              antiname = 'go',
              spin = 2,
              color = 8,
              mass = 'Mglu',
              width = 'Wglu',
              texname = 'go',
              antitexname = 'go',
              line = 'straight',
              charge = 0,
              GhostNumber = 0)

sv1 = Particle(pdg_code = 1000012,
               name = 'sv1',
               antiname = 'sv1~',
               spin = 1,
               color = 1,
               mass = 'Msn1',
               width = 'Wsn1',
               texname = 'sv1',
               antitexname = 'sv1',
               line = 'dashed',
               charge = 0,
               GhostNumber = 0)

sv1__tilde__ = sv1.anti()

sv2 = Particle(pdg_code = 1000014,
               name = 'sv2',
               antiname = 'sv2~',
               spin = 1,
               color = 1,
               mass = 'Msn2',
               width = 'Wsn2',
               texname = 'sv2',
               antitexname = 'sv2',
               line = 'dashed',
               charge = 0,
               GhostNumber = 0)

sv2__tilde__ = sv2.anti()

sv3 = Particle(pdg_code = 1000016,
               name = 'sv3',
               antiname = 'sv3~',
               spin = 1,
               color = 1,
               mass = 'Msn3',
               width = 'Wsn3',
               texname = 'sv3',
               antitexname = 'sv3',
               line = 'dashed',
               charge = 0,
               GhostNumber = 0)

sv3__tilde__ = sv3.anti()

sl1__minus__ = Particle(pdg_code = -1000011,
                        name = 'sl1-',
                        antiname = 'sl1+',
                        spin = 1,
                        color = 1,
                        mass = 'Msl1',
                        width = 'Wsl1',
                        texname = 'sl1-',
                        antitexname = 'sl1-',
                        line = 'dashed',
                        charge = -1,
                        GhostNumber = 0)

sl1__plus__ = sl1__minus__.anti()

sl2__minus__ = Particle(pdg_code = -1000013,
                        name = 'sl2-',
                        antiname = 'sl2+',
                        spin = 1,
                        color = 1,
                        mass = 'Msl2',
                        width = 'Wsl2',
                        texname = 'sl2-',
                        antitexname = 'sl2-',
                        line = 'dashed',
                        charge = -1,
                        GhostNumber = 0)

sl2__plus__ = sl2__minus__.anti()

sl3__minus__ = Particle(pdg_code = -1000015,
                        name = 'sl3-',
                        antiname = 'sl3+',
                        spin = 1,
                        color = 1,
                        mass = 'Msl3',
                        width = 'Wsl3',
                        texname = 'sl3-',
                        antitexname = 'sl3-',
                        line = 'dashed',
                        charge = -1,
                        GhostNumber = 0)

sl3__plus__ = sl3__minus__.anti()

sl4__minus__ = Particle(pdg_code = -2000011,
                        name = 'sl4-',
                        antiname = 'sl4+',
                        spin = 1,
                        color = 1,
                        mass = 'Msl4',
                        width = 'Wsl4',
                        texname = 'sl4-',
                        antitexname = 'sl4-',
                        line = 'dashed',
                        charge = -1,
                        GhostNumber = 0)

sl4__plus__ = sl4__minus__.anti()

sl5__minus__ = Particle(pdg_code = -2000013,
                        name = 'sl5-',
                        antiname = 'sl5+',
                        spin = 1,
                        color = 1,
                        mass = 'Msl5',
                        width = 'Wsl5',
                        texname = 'sl5-',
                        antitexname = 'sl5-',
                        line = 'dashed',
                        charge = -1,
                        GhostNumber = 0)

sl5__plus__ = sl5__minus__.anti()

sl6__minus__ = Particle(pdg_code = -2000015,
                        name = 'sl6-',
                        antiname = 'sl6+',
                        spin = 1,
                        color = 1,
                        mass = 'Msl6',
                        width = 'Wsl6',
                        texname = 'sl6-',
                        antitexname = 'sl6-',
                        line = 'dashed',
                        charge = -1,
                        GhostNumber = 0)

sl6__plus__ = sl6__minus__.anti()

su1 = Particle(pdg_code = 1000002,
               name = 'su1',
               antiname = 'su1~',
               spin = 1,
               color = 3,
               mass = 'Musq1',
               width = 'Wusq1',
               texname = 'su1',
               antitexname = 'su1',
               line = 'dashed',
               charge = 2/3,
               GhostNumber = 0)

su1__tilde__ = su1.anti()

su2 = Particle(pdg_code = 1000004,
               name = 'su2',
               antiname = 'su2~',
               spin = 1,
               color = 3,
               mass = 'Musq2',
               width = 'Wusq2',
               texname = 'su2',
               antitexname = 'su2',
               line = 'dashed',
               charge = 2/3,
               GhostNumber = 0)

su2__tilde__ = su2.anti()

su3 = Particle(pdg_code = 1000006,
               name = 'su3',
               antiname = 'su3~',
               spin = 1,
               color = 3,
               mass = 'Musq3',
               width = 'Wusq3',
               texname = 'su3',
               antitexname = 'su3',
               line = 'dashed',
               charge = 2/3,
               GhostNumber = 0)

su3__tilde__ = su3.anti()

su4 = Particle(pdg_code = 2000002,
               name = 'su4',
               antiname = 'su4~',
               spin = 1,
               color = 3,
               mass = 'Musq4',
               width = 'Wusq4',
               texname = 'su4',
               antitexname = 'su4',
               line = 'dashed',
               charge = 2/3,
               GhostNumber = 0)

su4__tilde__ = su4.anti()

su5 = Particle(pdg_code = 2000004,
               name = 'su5',
               antiname = 'su5~',
               spin = 1,
               color = 3,
               mass = 'Musq5',
               width = 'Wusq5',
               texname = 'su5',
               antitexname = 'su5',
               line = 'dashed',
               charge = 2/3,
               GhostNumber = 0)

su5__tilde__ = su5.anti()

su6 = Particle(pdg_code = 2000006,
               name = 'su6',
               antiname = 'su6~',
               spin = 1,
               color = 3,
               mass = 'Musq6',
               width = 'Wusq6',
               texname = 'su6',
               antitexname = 'su6',
               line = 'dashed',
               charge = 2/3,
               GhostNumber = 0)

su6__tilde__ = su6.anti()

sd1 = Particle(pdg_code = 1000001,
               name = 'sd1',
               antiname = 'sd1~',
               spin = 1,
               color = 3,
               mass = 'Mdsq1',
               width = 'Wdsq1',
               texname = 'sd1',
               antitexname = 'sd1',
               line = 'dashed',
               charge = -1/3,
               GhostNumber = 0)

sd1__tilde__ = sd1.anti()

sd2 = Particle(pdg_code = 1000003,
               name = 'sd2',
               antiname = 'sd2~',
               spin = 1,
               color = 3,
               mass = 'Mdsq2',
               width = 'Wdsq2',
               texname = 'sd2',
               antitexname = 'sd2',
               line = 'dashed',
               charge = -1/3,
               GhostNumber = 0)

sd2__tilde__ = sd2.anti()

sd3 = Particle(pdg_code = 1000005,
               name = 'sd3',
               antiname = 'sd3~',
               spin = 1,
               color = 3,
               mass = 'Mdsq3',
               width = 'Wdsq3',
               texname = 'sd3',
               antitexname = 'sd3',
               line = 'dashed',
               charge = -1/3,
               GhostNumber = 0)

sd3__tilde__ = sd3.anti()

sd4 = Particle(pdg_code = 2000001,
               name = 'sd4',
               antiname = 'sd4~',
               spin = 1,
               color = 3,
               mass = 'Mdsq4',
               width = 'Wdsq4',
               texname = 'sd4',
               antitexname = 'sd4',
               line = 'dashed',
               charge = -1/3,
               GhostNumber = 0)

sd4__tilde__ = sd4.anti()

sd5 = Particle(pdg_code = 2000003,
               name = 'sd5',
               antiname = 'sd5~',
               spin = 1,
               color = 3,
               mass = 'Mdsq5',
               width = 'Wdsq5',
               texname = 'sd5',
               antitexname = 'sd5',
               line = 'dashed',
               charge = -1/3,
               GhostNumber = 0)

sd5__tilde__ = sd5.anti()

sd6 = Particle(pdg_code = 2000005,
               name = 'sd6',
               antiname = 'sd6~',
               spin = 1,
               color = 3,
               mass = 'Mdsq6',
               width = 'Wdsq6',
               texname = 'sd6',
               antitexname = 'sd6',
               line = 'dashed',
               charge = -1/3,
               GhostNumber = 0)

sd6__tilde__ = sd6.anti()

h1 = Particle(pdg_code = 25,
              name = 'h1',
              antiname = 'h1',
              spin = 1,
              color = 1,
              mass = 'Mh01',
              width = 'Wh01',
              texname = 'h1',
              antitexname = 'h1',
              line = 'dashed',
              charge = 0,
              GhostNumber = 0)

h2 = Particle(pdg_code = 35,
              name = 'h2',
              antiname = 'h2',
              spin = 1,
              color = 1,
              mass = 'Mh02',
              width = 'Wh02',
              texname = 'h2',
              antitexname = 'h2',
              line = 'dashed',
              charge = 0,
              GhostNumber = 0)

h3 = Particle(pdg_code = 36,
              name = 'h3',
              antiname = 'h3',
              spin = 1,
              color = 1,
              mass = 'MA0',
              width = 'WA0',
              texname = 'h3',
              antitexname = 'h3',
              line = 'dashed',
              charge = 0,
              GhostNumber = 0)

H__plus__ = Particle(pdg_code = 37,
                     name = 'H+',
                     antiname = 'H-',
                     spin = 1,
                     color = 1,
                     mass = 'MH',
                     width = 'WH',
                     texname = 'H+',
                     antitexname = 'H+',
                     line = 'dashed',
                     charge = 1,
                     GhostNumber = 0)

H__minus__ = H__plus__.anti()

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
                       GhostNumber = 0)

phi__minus__ = phi__plus__.anti()

a = Particle(pdg_code = 22,
             name = 'a',
             antiname = 'a',
             spin = 3,
             color = 1,
             mass = 'ZERO',
             width = 'ZERO',
             texname = 'a',
             antitexname = 'a',
             line = 'wavy',
             charge = 0,
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
                     GhostNumber = 0)

W__minus__ = W__plus__.anti()

g = Particle(pdg_code = 21,
             name = 'g',
             antiname = 'g',
             spin = 3,
             color = 8,
             mass = 'ZERO',
             width = 'ZERO',
             texname = 'g',
             antitexname = 'g',
             line = 'curly',
             charge = 0,
             GhostNumber = 0)

ghG = Particle(pdg_code = 9000001,
               name = 'ghG',
               antiname = 'ghG~',
               spin = -1,
               color = 8,
               mass = 'ZERO',
               width = 'ZERO',
               texname = 'ghG',
               antitexname = 'ghG',
               line = 'dotted',
               charge = 0,
               GhostNumber = 1)

ghG__tilde__ = ghG.anti()

ghA = Particle(pdg_code = 9000002,
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
               GhostNumber = 1)

ghA__tilde__ = ghA.anti()

ghZ = Particle(pdg_code = 9000003,
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
               GhostNumber = 1)

ghZ__tilde__ = ghZ.anti()

ghWp = Particle(pdg_code = 9000004,
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
                GhostNumber = 1)

ghWp__tilde__ = ghWp.anti()

ghWm = Particle(pdg_code = 9000005,
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
                GhostNumber = 1)

ghWm__tilde__ = ghWm.anti()

