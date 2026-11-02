# This file was automatically created by FeynRules 1.7.69
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Mon 1 Oct 2012 14:58:25


from __future__ import division
from __future__ import absolute_import
from .object_library import all_particles, Particle
from . import parameters as Param

a = Particle(pdg_code=22,
             name='a',
             antiname='a',
             spin=3,
             color=1,
             mass=Param.ZERO,
             width=Param.ZERO,
             texname=r'{\gamma}',
             antitexname=r'{\gamma}',
             charge=0,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

Z = Particle(pdg_code=23,
             name='Z',
             antiname='Z',
             spin=3,
             color=1,
             mass=Param.MZ,
             width=Param.WZ,
             texname=r'{Z}',
             antitexname=r'{Z}',
             charge=0,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

W__plus__ = Particle(pdg_code=24,
                     name='W+',
                     antiname='W-',
                     spin=3,
                     color=1,
                     mass=Param.MW,
                     width=Param.WW,
                     texname=r'{W^+}',
                     antitexname=r'{W^-}',
                     charge=1,
                     GhostNumber=0,
                     LeptonNumber=0,
                     Y=0)

W__minus__ = W__plus__.anti()

g = Particle(pdg_code=21,
             name='g',
             antiname='g',
             spin=3,
             color=8,
             mass=Param.ZERO,
             width=Param.ZERO,
             texname=r'{g}',
             antitexname=r'{g}',
             charge=0,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

ghA = Particle(pdg_code=9000001,
               name='ghA',
               antiname='ghA~',
               spin=-1,
               color=1,
               mass=Param.ZERO,
               width=Param.ZERO,
               texname=r'\tilde{\gamma}',
               antitexname=r'\overline{\tilde{\gamma}}',
               charge=0,
               GhostNumber=1,
               LeptonNumber=0,
               Y=0)

ghA__tilde__ = ghA.anti()

ghZ = Particle(pdg_code=9000002,
               name='ghZ',
               antiname='ghZ~',
               spin=-1,
               color=1,
               mass=Param.MZ,
               width=Param.WZ,
               texname=r'\tilde{Z}',
               antitexname=r'\bar{\tilde{Z}}',
               charge=0,
               GhostNumber=1,
               LeptonNumber=0,
               Y=0)

ghZ__tilde__ = ghZ.anti()

ghWp = Particle(pdg_code=9000003,
                name='ghWp',
                antiname='ghWp~',
                spin=-1,
                color=1,
                mass=Param.MW,
                width=Param.WW,
                texname=r'{\tilde{W}^+}',
                antitexname=r'{\bar{\tilde{W}}^+}',
                charge=1,
                GhostNumber=1,
                LeptonNumber=0,
                Y=0)

ghWp__tilde__ = ghWp.anti()

ghWm = Particle(pdg_code=9000004,
                name='ghWm',
                antiname='ghWm~',
                spin=-1,
                color=1,
                mass=Param.MW,
                width=Param.WW,
                texname=r'{\tilde{W}^-}',
                antitexname=r'{\bar{\tilde{W}}^-}',
                charge=-1,
                GhostNumber=1,
                LeptonNumber=0,
                Y=0)

ghWm__tilde__ = ghWm.anti()

ghG = Particle(pdg_code=9000005,
               name='ghG',
               antiname='ghG~',
               spin=-1,
               color=8,
               mass=Param.ZERO,
               width=Param.ZERO,
               texname=r'{\tilde{g}}',
               antitexname=r'{\bar{\tilde{g}}}',
               charge=0,
               GhostNumber=1,
               LeptonNumber=0,
               Y=0)

ghG__tilde__ = ghG.anti()

ve = Particle(pdg_code=12,
              name='ve',
              antiname='ve~',
              spin=2,
              color=1,
              mass=Param.ZERO,
              width=Param.ZERO,
              texname=r'{\nu_e}',
              antitexname=r'{\overline{\nu}_e}',
              charge=0,
              GhostNumber=0,
              LeptonNumber=1,
              Y=0)

ve__tilde__ = ve.anti()

vm = Particle(pdg_code=14,
              name='vm',
              antiname='vm~',
              spin=2,
              color=1,
              mass=Param.ZERO,
              width=Param.ZERO,
              texname=r'{\nu_\mu}',
              antitexname=r'{\overline{\nu}_\mu}',
              charge=0,
              GhostNumber=0,
              LeptonNumber=1,
              Y=0)

vm__tilde__ = vm.anti()

vt = Particle(pdg_code=16,
              name='vt',
              antiname='vt~',
              spin=2,
              color=1,
              mass=Param.ZERO,
              width=Param.ZERO,
              texname=r'{\nu_\tau}',
              antitexname=r'{\overline{\nu}_\tau}',
              charge=0,
              GhostNumber=0,
              LeptonNumber=1,
              Y=0)

vt__tilde__ = vt.anti()

u = Particle(pdg_code=2,
             name='u',
             antiname='u~',
             spin=2,
             color=3,
             mass=Param.ZERO,
             width=Param.ZERO,
             texname=r'{u}',
             antitexname=r'{\overline{u}}',
             charge=2/3,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

u__tilde__ = u.anti()

c = Particle(pdg_code=4,
             name='c',
             antiname='c~',
             spin=2,
             color=3,
             mass=Param.MC,
             width=Param.ZERO,
             texname=r'{c}',
             antitexname=r'{\overline{c}}',
             charge=2/3,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

c__tilde__ = c.anti()

t = Particle(pdg_code=6,
             name='t',
             antiname='t~',
             spin=2,
             color=3,
             mass=Param.MT,
             width=Param.WT,
             texname=r'{t}',
             antitexname=r'{\overline{t}}',
             charge=2/3,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

t__tilde__ = t.anti()

d = Particle(pdg_code=1,
             name='d',
             antiname='d~',
             spin=2,
             color=3,
             mass=Param.ZERO,
             width=Param.ZERO,
             texname=r'{d}',
             antitexname=r'{\overline{d}}',
             charge=-1/3,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

d__tilde__ = d.anti()

s = Particle(pdg_code=3,
             name='s',
             antiname='s~',
             spin=2,
             color=3,
             mass=Param.ZERO,
             width=Param.ZERO,
             texname=r'{s}',
             antitexname=r'{\overline{s}}',
             charge=-1/3,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

s__tilde__ = s.anti()

b = Particle(pdg_code=5,
             name='b',
             antiname='b~',
             spin=2,
             color=3,
             mass=Param.MB,
             width=Param.ZERO,
             texname=r'{b}',
             antitexname=r'{\overline{b}}',
             charge=-1/3,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

b__tilde__ = b.anti()

H = Particle(pdg_code=25,
             name='H',
             antiname='H',
             spin=1,
             color=1,
             mass=Param.MH,
             width=Param.WH,
             texname=r'{H}',
             antitexname=r'{H}',
             charge=0,
             GhostNumber=0,
             LeptonNumber=0,
             Y=0)

G0 = Particle(pdg_code=250,
              name='G0',
              antiname='G0',
              spin=1,
              color=1,
              mass=Param.MZ,
              width=Param.WZ,
              texname=r'{G_0}',
              antitexname=r'{G_0}',
              GoldstoneBoson=True,
              charge=0,
              GhostNumber=0,
              LeptonNumber=0,
              Y=0)

G__plus__ = Particle(pdg_code=251,
                     name='G+',
                     antiname='G-',
                     spin=1,
                     color=1,
                     mass=Param.MW,
                     width=Param.WW,
                     texname=r'{G^+}',
                     antitexname=r'{G^-}',
                     GoldstoneBoson=True,
                     charge=1,
                     GhostNumber=0,
                     LeptonNumber=0,
                     Y=0)

G__minus__ = G__plus__.anti()

e__minus__ = Particle(pdg_code=11,
                      name='e-',
                      antiname='e+',
                      spin=2,
                      color=1,
                      mass=Param.Me,
                      width=Param.ZERO,
                      texname=r'{e^-}',
                      antitexname=r'{e^+}',
                      charge=-1,
                      GhostNumber=0,
                      LeptonNumber=1,
                      Y=0)

e__plus__ = e__minus__.anti()

mu__minus__ = Particle(pdg_code=13,
                       name='mu-',
                       antiname='mu+',
                       spin=2,
                       color=1,
                       mass=Param.MM,
                       width=Param.ZERO,
                       texname=r'{\mu^-}',
                       antitexname=r'{\mu^+}',
                       charge=-1,
                       GhostNumber=0,
                       LeptonNumber=1,
                       Y=0)

mu__plus__ = mu__minus__.anti()

ta__minus__ = Particle(pdg_code=15,
                       name='ta-',
                       antiname='ta+',
                       spin=2,
                       color=1,
                       mass=Param.MTA,
                       width=Param.WTau,
                       texname=r'{\tau^-}',
                       antitexname=r'{\tau^+}',
                       charge=-1,
                       GhostNumber=0,
                       LeptonNumber=1,
                       Y=0)

ta__plus__ = ta__minus__.anti()
