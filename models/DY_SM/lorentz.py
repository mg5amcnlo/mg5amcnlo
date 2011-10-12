# This file was automatically created by FeynRules $Revision: 634 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 28 Jul 2011 16:28:57


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec



SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'Gamma5(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'Identity(2,1)')

FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')

FFV2 = Lorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma5(-1,1)*Gamma(3,2,-1)')

FFV3 = Lorentz(name = 'FFV3',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFV4 = Lorentz(name = 'FFV4',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV5 = Lorentz(name = 'FFV5',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV6 = Lorentz(name = 'FFV6',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 4*Gamma(3,2,-1)*ProjP(-1,1)')

FFT1 = Lorentz(name = 'FFT1',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,1)*Gamma(1003,2,1) + P(1003,1)*Gamma(2003,2,1)')

FFT2 = Lorentz(name = 'FFT2',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,1)*Gamma(1003,2,1) - P(2003,2)*Gamma(1003,2,1) + P(1003,1)*Gamma(2003,2,1) - P(1003,2)*Gamma(2003,2,1)')

FFT3 = Lorentz(name = 'FFT3',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,2)*Gamma(1003,2,1) + P(1003,2)*Gamma(2003,2,1)')

FFT4 = Lorentz(name = 'FFT4',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,1)*Gamma(1003,2,1) + P(2003,2)*Gamma(1003,2,1) + P(1003,1)*Gamma(2003,2,1) + P(1003,2)*Gamma(2003,2,1)')

FFT5 = Lorentz(name = 'FFT5',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,1)*Gamma5(-1,1)*Gamma(1003,2,-1) + P(1003,1)*Gamma5(-1,1)*Gamma(2003,2,-1)')

FFT6 = Lorentz(name = 'FFT6',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,1)*Gamma5(-1,1)*Gamma(1003,2,-1) - P(2003,2)*Gamma5(-1,1)*Gamma(1003,2,-1) + P(1003,1)*Gamma5(-1,1)*Gamma(2003,2,-1) - P(1003,2)*Gamma5(-1,1)*Gamma(2003,2,-1)')

FFT7 = Lorentz(name = 'FFT7',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,2)*Gamma5(-1,1)*Gamma(1003,2,-1) + P(1003,2)*Gamma5(-1,1)*Gamma(2003,2,-1)')

FFT8 = Lorentz(name = 'FFT8',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,1)*Gamma5(-1,1)*Gamma(1003,2,-1) + P(2003,2)*Gamma5(-1,1)*Gamma(1003,2,-1) + P(1003,1)*Gamma5(-1,1)*Gamma(2003,2,-1) + P(1003,2)*Gamma5(-1,1)*Gamma(2003,2,-1)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVS2 = Lorentz(name = 'VVS2',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,2)*P(2,1)*P(3,3) - P(-1,1)*P(-1,2)*P(3,3)*Metric(1,2)')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVT1 = Lorentz(name = 'VVT1',
               spins = [ 3, 3, 5 ],
               structure = 'P(1003,2)*P(2003,1)*Metric(1,2) + P(1003,1)*P(2003,2)*Metric(1,2) - P(2,1)*P(2003,2)*Metric(1,1003) - P(2,1)*P(1003,2)*Metric(1,2003) - P(1,2)*P(2003,1)*Metric(2,1003) + P(-1,1)*P(-1,2)*Metric(1,2003)*Metric(2,1003) - P(1,2)*P(1003,1)*Metric(2,2003) + P(-1,1)*P(-1,2)*Metric(1,1003)*Metric(2,2003)')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

VVVS1 = Lorentz(name = 'VVVS1',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,1)*P(4,4)*Metric(1,2) - P(3,2)*P(4,4)*Metric(1,2) - P(2,1)*P(4,4)*Metric(1,3) + P(2,3)*P(4,4)*Metric(1,3) + P(1,2)*P(4,4)*Metric(2,3) - P(1,3)*P(4,4)*Metric(2,3)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV6 = Lorentz(name = 'VVVV6',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVT1 = Lorentz(name = 'VVVT1',
                spins = [ 3, 3, 3, 5 ],
                structure = 'P(2004,2)*Metric(1,1004)*Metric(2,3) - P(2004,3)*Metric(1,1004)*Metric(2,3) + P(1004,2)*Metric(1,2004)*Metric(2,3) - P(1004,3)*Metric(1,2004)*Metric(2,3) - P(2004,1)*Metric(1,3)*Metric(2,1004) + P(2004,3)*Metric(1,3)*Metric(2,1004) + P(3,1)*Metric(1,2004)*Metric(2,1004) - P(3,2)*Metric(1,2004)*Metric(2,1004) - P(1004,1)*Metric(1,3)*Metric(2,2004) + P(1004,3)*Metric(1,3)*Metric(2,2004) + P(3,1)*Metric(1,1004)*Metric(2,2004) - P(3,2)*Metric(1,1004)*Metric(2,2004) + P(2004,1)*Metric(1,2)*Metric(3,1004) - P(2004,2)*Metric(1,2)*Metric(3,1004) - P(2,1)*Metric(1,2004)*Metric(3,1004) + P(2,3)*Metric(1,2004)*Metric(3,1004) + P(1,2)*Metric(2,2004)*Metric(3,1004) - P(1,3)*Metric(2,2004)*Metric(3,1004) + P(1004,1)*Metric(1,2)*Metric(3,2004) - P(1004,2)*Metric(1,2)*Metric(3,2004) - P(2,1)*Metric(1,1004)*Metric(3,2004) + P(2,3)*Metric(1,1004)*Metric(3,2004) + P(1,2)*Metric(2,1004)*Metric(3,2004) - P(1,3)*Metric(2,1004)*Metric(3,2004)')

