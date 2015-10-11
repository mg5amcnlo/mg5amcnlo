# This file was automatically created by FeynRules 2.0.25
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 8 May 2014 12:30:33


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param

import propagators as Prop

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
              antitexname = 'vm~',
              charge = 0,
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
              antitexname = 'vt~',
              charge = 0,
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
                      antitexname = 'e+',
                      charge = -1,
                      LeptonNumber = 1)

e__plus__ = e__minus__.anti()

mu__minus__ = Particle(pdg_code = 13,
                       name = 'mu-',
                       antiname = 'mu+',
                       spin = 2,
                       color = 1,
                       mass = Param.MMU,
                       width = Param.ZERO,
                       texname = 'mu-',
                       antitexname = 'mu+',
                       charge = -1,
                       LeptonNumber = 1)

mu__plus__ = mu__minus__.anti()

ta__minus__ = Particle(pdg_code = 15,
                       name = 'ta-',
                       antiname = 'ta+',
                       spin = 2,
                       color = 1,
                       mass = Param.MTA,
                       width = Param.WTA,
                       texname = 'ta-',
                       antitexname = 'ta+',
                       charge = -1,
                       LeptonNumber = 1)

ta__plus__ = ta__minus__.anti()

pi__plus__ = Particle(pdg_code = 211,
                      name = 'pi+',
                      antiname = 'pi-',
                      spin = 1,
                      color = 1,
                      mass = Param.Mpic,
                      width = Param.ZERO,
                      texname = 'pi+',
                      antitexname = 'pi-',
                      charge = 1,
                      LeptonNumber = 0)

pi__minus__ = pi__plus__.anti()

pi0 = Particle(pdg_code = 111,
               name = 'pi0',
               antiname = 'pi0',
               spin = 1,
               color = 1,
               mass = Param.Mpi0,
               width = Param.ZERO,
               texname = 'pi0',
               antitexname = 'pi0',
               charge = 0,
               LeptonNumber = 0)

