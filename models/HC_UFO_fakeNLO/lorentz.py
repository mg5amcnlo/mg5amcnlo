# This file was automatically created by FeynRules 2.0.6
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Wed 11 Dec 2013 19:27:12


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot


SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'Gamma5(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) + ProjP(2,1)')

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
               structure = 'Identity(2,1)*Metric(1003,2003)')

FFT2 = Lorentz(name = 'FFT2',
               spins = [ 2, 2, 5 ],
               structure = 'P(2003,1)*Gamma(1003,2,1) - P(2003,2)*Gamma(1003,2,1) + P(1003,1)*Gamma(2003,2,1) - P(1003,2)*Gamma(2003,2,1) - 2*P(-1,1)*Gamma(-1,2,1)*Metric(1003,2003) + 2*P(-1,2)*Gamma(-1,2,1)*Metric(1003,2003)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVS2 = Lorentz(name = 'VVS2',
               spins = [ 3, 3, 1 ],
               structure = '-4*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1) + 4*Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVS3 = Lorentz(name = 'VVS3',
               spins = [ 3, 3, 1 ],
               structure = '-(Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1)) + Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVS4 = Lorentz(name = 'VVS4',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVS5 = Lorentz(name = 'VVS5',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,1)*P(2,1) - P(-1,1)**2*Metric(1,2)')

VVS6 = Lorentz(name = 'VVS6',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVS7 = Lorentz(name = 'VVS7',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,2)*P(2,2) - P(-1,2)**2*Metric(1,2)')

VVS8 = Lorentz(name = 'VVS8',
               spins = [ 3, 3, 1 ],
               structure = 'P(1,1)*P(2,1) + P(1,2)*P(2,2) - P(-1,1)**2*Metric(1,2) - P(-1,2)**2*Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2)')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,3))')

VVV3 = Lorentz(name = 'VVV3',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3)')

VVV4 = Lorentz(name = 'VVV4',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3)')

VVV5 = Lorentz(name = 'VVV5',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(1,2)*Metric(2,3)')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVV8 = Lorentz(name = 'VVV8',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVV9 = Lorentz(name = 'VVV9',
               spins = [ 3, 3, 3 ],
               structure = 'P(2,3)*Metric(1,3) + P(1,3)*Metric(2,3)')

VVT1 = Lorentz(name = 'VVT1',
               spins = [ 3, 3, 5 ],
               structure = 'Metric(1,2003)*Metric(2,1003) + Metric(1,1003)*Metric(2,2003) - Metric(1,2)*Metric(1003,2003)')

VVT2 = Lorentz(name = 'VVT2',
               spins = [ 3, 3, 5 ],
               structure = 'P(1003,2)*P(2003,1)*Metric(1,2) + P(1003,1)*P(2003,2)*Metric(1,2) - P(2,1)*P(2003,2)*Metric(1,1003) - P(2,1)*P(1003,2)*Metric(1,2003) - P(1,2)*P(2003,1)*Metric(2,1003) + P(-1,1)*P(-1,2)*Metric(1,2003)*Metric(2,1003) - P(1,2)*P(1003,1)*Metric(2,2003) + P(-1,1)*P(-1,2)*Metric(1,1003)*Metric(2,2003) + P(1,2)*P(2,1)*Metric(1003,2003) - P(-1,1)*P(-1,2)*Metric(1,2)*Metric(1003,2003)')

VVT3 = Lorentz(name = 'VVT3',
               spins = [ 3, 3, 5 ],
               structure = 'P(1003,2)*P(2003,1)*Metric(1,2) + P(1003,1)*P(2003,2)*Metric(1,2) - P(2,1)*P(2003,2)*Metric(1,1003) - P(2,2)*P(2003,2)*Metric(1,1003) - P(2,1)*P(1003,2)*Metric(1,2003) - P(2,2)*P(1003,2)*Metric(1,2003) - P(1,1)*P(2003,1)*Metric(2,1003) - P(1,2)*P(2003,1)*Metric(2,1003) + P(-1,1)*P(-1,2)*Metric(1,2003)*Metric(2,1003) - P(1,1)*P(1003,1)*Metric(2,2003) - P(1,2)*P(1003,1)*Metric(2,2003) + P(-1,1)*P(-1,2)*Metric(1,1003)*Metric(2,2003) + P(1,1)*P(2,1)*Metric(1003,2003) + P(1,2)*P(2,1)*Metric(1003,2003) + P(1,1)*P(2,2)*Metric(1003,2003) + P(1,2)*P(2,2)*Metric(1003,2003) - P(-1,1)*P(-1,2)*Metric(1,2)*Metric(1003,2003)')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

FFVT1 = Lorentz(name = 'FFVT1',
                spins = [ 2, 2, 3, 5 ],
                structure = 'Gamma(2004,2,1)*Metric(3,1004) + Gamma(1004,2,1)*Metric(3,2004) - 2*Gamma(3,2,1)*Metric(1004,2004)')

FFVT2 = Lorentz(name = 'FFVT2',
                spins = [ 2, 2, 3, 5 ],
                structure = 'Gamma(2004,2,-1)*Metric(3,1004)*ProjM(-1,1) + Gamma(1004,2,-1)*Metric(3,2004)*ProjM(-1,1) - 2*Gamma(3,2,-1)*Metric(1004,2004)*ProjM(-1,1)')

FFVT3 = Lorentz(name = 'FFVT3',
                spins = [ 2, 2, 3, 5 ],
                structure = '-(Gamma(2004,2,-1)*Metric(3,1004)*ProjM(-1,1))/2. - (Gamma(1004,2,-1)*Metric(3,2004)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(1004,2004)*ProjM(-1,1)')

FFVT4 = Lorentz(name = 'FFVT4',
                spins = [ 2, 2, 3, 5 ],
                structure = 'Gamma(2004,2,-1)*Metric(3,1004)*ProjM(-1,1) + Gamma(1004,2,-1)*Metric(3,2004)*ProjM(-1,1) - 2*Gamma(3,2,-1)*Metric(1004,2004)*ProjM(-1,1) + 2*Gamma(2004,2,-1)*Metric(3,1004)*ProjP(-1,1) + 2*Gamma(1004,2,-1)*Metric(3,2004)*ProjP(-1,1) - 4*Gamma(3,2,-1)*Metric(1004,2004)*ProjP(-1,1)')

FFVT5 = Lorentz(name = 'FFVT5',
                spins = [ 2, 2, 3, 5 ],
                structure = '-(Gamma(2004,2,-1)*Metric(3,1004)*ProjM(-1,1))/2. - (Gamma(1004,2,-1)*Metric(3,2004)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(1004,2004)*ProjM(-1,1) + Gamma(2004,2,-1)*Metric(3,1004)*ProjP(-1,1) + Gamma(1004,2,-1)*Metric(3,2004)*ProjP(-1,1) - 2*Gamma(3,2,-1)*Metric(1004,2004)*ProjP(-1,1)')

FFVT6 = Lorentz(name = 'FFVT6',
                spins = [ 2, 2, 3, 5 ],
                structure = '-(Gamma(2004,2,-1)*Metric(3,1004)*ProjM(-1,1))/2. - (Gamma(1004,2,-1)*Metric(3,2004)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(1004,2004)*ProjM(-1,1) - (Gamma(2004,2,-1)*Metric(3,1004)*ProjP(-1,1))/2. - (Gamma(1004,2,-1)*Metric(3,2004)*ProjP(-1,1))/2. + Gamma(3,2,-1)*Metric(1004,2004)*ProjP(-1,1)')

FFVT7 = Lorentz(name = 'FFVT7',
                spins = [ 2, 2, 3, 5 ],
                structure = '-(Gamma(2004,2,-1)*Metric(3,1004)*ProjM(-1,1))/2. - (Gamma(1004,2,-1)*Metric(3,2004)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(1004,2004)*ProjM(-1,1) - Gamma(2004,2,-1)*Metric(3,1004)*ProjP(-1,1) - Gamma(1004,2,-1)*Metric(3,2004)*ProjP(-1,1) + 2*Gamma(3,2,-1)*Metric(1004,2004)*ProjP(-1,1)')

FFVT8 = Lorentz(name = 'FFVT8',
                spins = [ 2, 2, 3, 5 ],
                structure = '-(Gamma(2004,2,-1)*Metric(3,1004)*ProjM(-1,1))/2. - (Gamma(1004,2,-1)*Metric(3,2004)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(1004,2004)*ProjM(-1,1) - 2*Gamma(2004,2,-1)*Metric(3,1004)*ProjP(-1,1) - 2*Gamma(1004,2,-1)*Metric(3,2004)*ProjP(-1,1) + 4*Gamma(3,2,-1)*Metric(1004,2004)*ProjP(-1,1)')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = '-4*Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1) + 4*Epsilon(1,2,-1,-2)*P(-2,1)*P(-1,2)')

VVSS2 = Lorentz(name = 'VVSS2',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

VVSS3 = Lorentz(name = 'VVSS3',
                spins = [ 3, 3, 1, 1 ],
                structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVVS1 = Lorentz(name = 'VVVS1',
                spins = [ 3, 3, 3, 1 ],
                structure = '-4*Epsilon(1,2,3,-1)*P(-1,1) - 4*Epsilon(1,2,3,-1)*P(-1,2) - 4*Epsilon(1,2,3,-1)*P(-1,3)')

VVVS2 = Lorentz(name = 'VVVS2',
                spins = [ 3, 3, 3, 1 ],
                structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVT1 = Lorentz(name = 'VVVT1',
                spins = [ 3, 3, 3, 5 ],
                structure = 'P(2004,2)*Metric(1,1004)*Metric(2,3) - P(2004,3)*Metric(1,1004)*Metric(2,3) + P(1004,2)*Metric(1,2004)*Metric(2,3) - P(1004,3)*Metric(1,2004)*Metric(2,3) - P(2004,1)*Metric(1,3)*Metric(2,1004) + P(2004,3)*Metric(1,3)*Metric(2,1004) + P(3,1)*Metric(1,2004)*Metric(2,1004) - P(3,2)*Metric(1,2004)*Metric(2,1004) - P(1004,1)*Metric(1,3)*Metric(2,2004) + P(1004,3)*Metric(1,3)*Metric(2,2004) + P(3,1)*Metric(1,1004)*Metric(2,2004) - P(3,2)*Metric(1,1004)*Metric(2,2004) + P(2004,1)*Metric(1,2)*Metric(3,1004) - P(2004,2)*Metric(1,2)*Metric(3,1004) - P(2,1)*Metric(1,2004)*Metric(3,1004) + P(2,3)*Metric(1,2004)*Metric(3,1004) + P(1,2)*Metric(2,2004)*Metric(3,1004) - P(1,3)*Metric(2,2004)*Metric(3,1004) + P(1004,1)*Metric(1,2)*Metric(3,2004) - P(1004,2)*Metric(1,2)*Metric(3,2004) - P(2,1)*Metric(1,1004)*Metric(3,2004) + P(2,3)*Metric(1,1004)*Metric(3,2004) + P(1,2)*Metric(2,1004)*Metric(3,2004) - P(1,3)*Metric(2,1004)*Metric(3,2004) - P(3,1)*Metric(1,2)*Metric(1004,2004) + P(3,2)*Metric(1,2)*Metric(1004,2004) + P(2,1)*Metric(1,3)*Metric(1004,2004) - P(2,3)*Metric(1,3)*Metric(1004,2004) - P(1,2)*Metric(2,3)*Metric(1004,2004) + P(1,3)*Metric(2,3)*Metric(1004,2004)')

VVVSS1 = Lorentz(name = 'VVVSS1',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = '-4*Epsilon(1,2,3,-1)*P(-1,1) - 4*Epsilon(1,2,3,-1)*P(-1,2) - 4*Epsilon(1,2,3,-1)*P(-1,3)')

VVVSS2 = Lorentz(name = 'VVVSS2',
                 spins = [ 3, 3, 3, 1, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVVS1 = Lorentz(name = 'VVVVS1',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVS2 = Lorentz(name = 'VVVVS2',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVS3 = Lorentz(name = 'VVVVS3',
                 spins = [ 3, 3, 3, 3, 1 ],
                 structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVVT1 = Lorentz(name = 'VVVVT1',
                 spins = [ 3, 3, 3, 3, 5 ],
                 structure = 'Metric(1,2005)*Metric(2,4)*Metric(3,1005) - Metric(1,4)*Metric(2,2005)*Metric(3,1005) + Metric(1,1005)*Metric(2,4)*Metric(3,2005) - Metric(1,4)*Metric(2,1005)*Metric(3,2005) - Metric(1,2005)*Metric(2,3)*Metric(4,1005) + Metric(1,3)*Metric(2,2005)*Metric(4,1005) - Metric(1,1005)*Metric(2,3)*Metric(4,2005) + Metric(1,3)*Metric(2,1005)*Metric(4,2005) + Metric(1,4)*Metric(2,3)*Metric(1005,2005) - Metric(1,3)*Metric(2,4)*Metric(1005,2005)')

VVVVT2 = Lorentz(name = 'VVVVT2',
                 spins = [ 3, 3, 3, 3, 5 ],
                 structure = 'Metric(1,2005)*Metric(2,1005)*Metric(3,4) + Metric(1,1005)*Metric(2,2005)*Metric(3,4) - Metric(1,4)*Metric(2,2005)*Metric(3,1005) - Metric(1,4)*Metric(2,1005)*Metric(3,2005) - Metric(1,2005)*Metric(2,3)*Metric(4,1005) + Metric(1,2)*Metric(3,2005)*Metric(4,1005) - Metric(1,1005)*Metric(2,3)*Metric(4,2005) + Metric(1,2)*Metric(3,1005)*Metric(4,2005) + Metric(1,4)*Metric(2,3)*Metric(1005,2005) - Metric(1,2)*Metric(3,4)*Metric(1005,2005)')

VVVVT3 = Lorentz(name = 'VVVVT3',
                 spins = [ 3, 3, 3, 3, 5 ],
                 structure = 'Metric(1,2005)*Metric(2,1005)*Metric(3,4) + Metric(1,1005)*Metric(2,2005)*Metric(3,4) - Metric(1,2005)*Metric(2,4)*Metric(3,1005) - Metric(1,1005)*Metric(2,4)*Metric(3,2005) - Metric(1,3)*Metric(2,2005)*Metric(4,1005) + Metric(1,2)*Metric(3,2005)*Metric(4,1005) - Metric(1,3)*Metric(2,1005)*Metric(4,2005) + Metric(1,2)*Metric(3,1005)*Metric(4,2005) + Metric(1,3)*Metric(2,4)*Metric(1005,2005) - Metric(1,2)*Metric(3,4)*Metric(1005,2005)')

VVVVSS1 = Lorentz(name = 'VVVVSS1',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVSS2 = Lorentz(name = 'VVVVSS2',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVSS3 = Lorentz(name = 'VVVVSS3',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

