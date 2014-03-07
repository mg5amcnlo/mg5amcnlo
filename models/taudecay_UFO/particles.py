# This file was automatically created by FeynRules 2.0.6
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Wed 18 Dec 2013 14:14:59


from __future__ import division
from object_library import all_particles, Particle
import parameters as Param

import propagators as Prop

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

