# This file was automatically created by FeynRules $Revision: 634 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Wed 6 Jul 2011 14:07:37
from __future__ import division
from object_library import all_decays, Decay
import particles as P

#Decay(P.Z,
#     {(P.u, P.u__tilde__): '''(ee**2*(9*cw**4*(-MU**2 + MZ**2) - 6*cw**2*(11*MU**2 + MZ**2)*sw**2 \
#              + (7*MU**2 + 17*MZ**2)*sw**4)*cmath.sqrt(-4*MU**2*MZ**2 + \
#               MZ**4))/(96.*cw**2*cmath.pi*MZ**3*sw**2)'''
#     } 
#     )

#Decay(P.Z,
#     {(P.u, P.u__tilde__): '''(ee**2*(9*cw**4*( MZ**2) - 6*cw**2*MZ**2*sw**2 \
#              + (17*MZ**2)*sw**4)*cmath.sqrt(MZ**4))/(96.*cw**2*cmath.pi*MZ**3*sw**2)'''
#     }
#     )

Decay(P.Z,
      {(P.u, P.u__tilde__): '''(ee**2*(9*cw**4*MZ**2 - 6*cw**2*MZ**2*sw**2 + \
              17*MZ**2*sw**4)*cmath.sqrt(MZ**4))/(288.*cw**2*cmath.pi*MZ**3*sw**2)'''
      } 
     )