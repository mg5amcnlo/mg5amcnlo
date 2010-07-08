# This file was automatically created by FeynRules $Revision: 216 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Wed 7 Jul 2010 09:55:56


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec



SSS_1 = Lorentz(name = 'SSS_1',
                spins = [ 1, 1, 1 ],
                structure = '1')

FFS_2 = Lorentz(name = 'FFS_2',
                spins = [ 2, 2, 1 ],
                structure = 'Identity(1,2)')

FFV_3 = Lorentz(name = 'FFV_3',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,2)')

FFV_4 = Lorentz(name = 'FFV_4',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2)')

FFV_5 = Lorentz(name = 'FFV_5',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) - 2*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')

FFV_6 = Lorentz(name = 'FFV_6',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) + 2*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')

FFV_7 = Lorentz(name = 'FFV_7',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,1,\'s1\')*ProjM(\'s1\',2) + 4*Gamma(3,1,\'s1\')*ProjP(\'s1\',2)')

VVS_8 = Lorentz(name = 'VVS_8',
                spins = [ 3, 3, 1 ],
                structure = 'Metric(1,2)')

VVV_9 = Lorentz(name = 'VVV_9',
                spins = [ 3, 3, 3 ],
                structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

SSSS_10 = Lorentz(name = 'SSSS_10',
                  spins = [ 1, 1, 1, 1 ],
                  structure = '1')

VVSS_11 = Lorentz(name = 'VVSS_11',
                  spins = [ 3, 3, 1, 1 ],
                  structure = 'Metric(1,2)')

VVVV_12 = Lorentz(name = 'VVVV_12',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV_13 = Lorentz(name = 'VVVV_13',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV_14 = Lorentz(name = 'VVVV_14',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV_15 = Lorentz(name = 'VVVV_15',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV_16 = Lorentz(name = 'VVVV_16',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

