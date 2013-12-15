# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

R2_GG_1 = Lorentz(name = 'R2_GG_1',
               spins = [ 3, 3 ],
               structure = 'P(-1,1)*P(-1,1)*Metric(1,2)')

R2_GG_2 = Lorentz(name = 'R2_GG_2',
               spins = [ 3, 3 ],
               structure = 'P(1,1)*P(2,1)')

R2_QQ = Lorentz(name = 'R2_QQ',
               spins = [ 2, 2 ],
               structure = 'P(-1,1)*Gamma(-1,2,1)')

FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                 spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

GHGHG = Lorentz(name = 'GHGHG',
                 spins = [ 1, 1, 3 ],
                structure = 'P(3,1)')

#=============================================================================================
#  4-gluon R2 vertex
#=============================================================================================


R2_4G_1234 = Lorentz(name = 'R2_4G_1234',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,2)*Metric(3,4)')

R2_4G_1324 = Lorentz(name = 'R2_4G_1324',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4)')

R2_4G_1423 = Lorentz(name = 'R2_4G_1423',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3)')

#=============================================================================================

