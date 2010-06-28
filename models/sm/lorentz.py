# This file was automatically created by FeynRules $Revision: 216 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Sat 26 Jun 2010 18:16:42


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec



UUS_1 = Lorentz(name = 'UUS_1',
                spins = [ -1, -1, 1 ],
                structure = '1')

UUV_2 = Lorentz(name = 'UUV_2',
                spins = [ -1, -1, 3 ],
                structure = 'P(3,1) + P(3,3)')

UUV_3 = Lorentz(name = 'UUV_3',
                spins = [ -1, -1, 3 ],
                structure = 'P(3,2) + P(3,3)')

SSS_4 = Lorentz(name = 'SSS_4',
                spins = [ 1, 1, 1 ],
                structure = '1')

FFS_5 = Lorentz(name = 'FFS_5',
                spins = [ 2, 2, 1 ],
                structure = 'Gamma5(1,2)')

FFS_6 = Lorentz(name = 'FFS_6',
                spins = [ 2, 2, 1 ],
                structure = 'Identity(1,2)')

FFS_7 = Lorentz(name = 'FFS_7',
                spins = [ 2, 2, 1 ],
                structure = 'ProjM(1,2)')

FFS_8 = Lorentz(name = 'FFS_8',
                spins = [ 2, 2, 1 ],
                structure = 'ProjP(1,2)')

FFV_9 = Lorentz(name = 'FFV_9',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,2)')

FFV_10 = Lorentz(name = 'FFV_10',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)')

FFV_11 = Lorentz(name = 'FFV_11',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) - 2*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')

FFV_12 = Lorentz(name = 'FFV_12',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) + 2*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')

FFV_13 = Lorentz(name = 'FFV_13',
                 spins = [ 2, 2, 3 ],
                 structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) + 4*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')

VSS_14 = Lorentz(name = 'VSS_14',
                 spins = [ 3, 1, 1 ],
                 structure = 'P(1,2) - P(1,3)')

VVS_15 = Lorentz(name = 'VVS_15',
                 spins = [ 3, 3, 1 ],
                 structure = 'Metric(1,2)')

VVV_16 = Lorentz(name = 'VVV_16',
                 spins = [ 3, 3, 3 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

SSSS_17 = Lorentz(name = 'SSSS_17',
                  spins = [ 1, 1, 1, 1 ],
                  structure = '1')

VVSS_18 = Lorentz(name = 'VVSS_18',
                  spins = [ 3, 3, 1, 1 ],
                  structure = 'Metric(1,2)')

VVVV_19 = Lorentz(name = 'VVVV_19',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV_20 = Lorentz(name = 'VVVV_20',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV_21 = Lorentz(name = 'VVVV_21',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV_22 = Lorentz(name = 'VVVV_22',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV_23 = Lorentz(name = 'VVVV_23',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

