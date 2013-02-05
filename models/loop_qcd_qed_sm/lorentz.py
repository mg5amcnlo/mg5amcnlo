# This file was automatically created by FeynRules $Revision: 999 $
# Mathematica version: 7.0 for Linux x86 (64-bit) (February 18, 2009)
# Date: Mon 30 Jan 2012 19:57:04


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

###################################
# CounterTerms Lorentz structures #
###################################

R2_GG_1 = Lorentz(name = 'R2_GG_1',
               spins = [ 3, 3 ],
               structure = 'P(-1,1)*P(-1,1)*Metric(1,2)')

R2_GG_2 = Lorentz(name = 'R2_GG_2',
               spins = [ 3, 3 ],
               structure = 'P(1,1)*P(2,1)')

R2_GG_3 = Lorentz(name = 'R2_GG_3',
               spins = [ 3, 3 ],
               structure = 'Metric(1,2)')

R2_QQ_1 = Lorentz(name = 'R2_QQ_1',
               spins = [ 2, 2 ],
               structure = 'P(-1,1)*Gamma(-1,2,1)')

R2_QQ_2 = Lorentz(name = 'R2_QQ_2',
               spins = [ 2, 2 ],
               structure = 'Identity(1,2)')

R2_QQ_3 = Lorentz(name = 'R2_QQ_3',
               spins = [ 2, 2 ],
               structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

R2_QQ_4 = Lorentz(name = 'R2_QQ_4',
                spins = [ 2, 2 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)')

R2_SS_1 = Lorentz(name = 'R2_SS_1',
                  spins = [ 1, 1 ],
                  structure = '1')

R2_SS_2 = Lorentz(name = 'R2_SS_2',
                  spins = [ 1, 1 ],
                  structure = 'P(-1,1)*P(-1,1)')

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

# From FeynRules

R2RGA_VVVV10 = Lorentz(name = 'R2RGA_VVVV10',
                       spins = [ 3, 3, 3, 3 ],
                       structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4)\
                       + Metric(1,2)*Metric(3,4)')

R2RGA_VVVV2 = Lorentz(name = 'R2RGA_VVVV2',
                      spins = [ 3, 3, 3, 3 ],
                      structure = 'Metric(1,4)*Metric(2,3)')

R2RGA_VVVV3 = Lorentz(name = 'R2RGA_VVVV3',
                      spins = [ 3, 3, 3, 3 ],
                      structure = 'Metric(1,3)*Metric(2,4)')

R2RGA_VVVV5 = Lorentz(name = 'R2RGA_VVVV5',
                      spins = [ 3, 3, 3, 3 ],
                      structure = 'Metric(1,2)*Metric(3,4)')

#=============================================================================================

R2_GGZ = Lorentz(name = 'R2_GGZ',
                 spins = [ 3, 3, 3 ],
                 structure = 'Epsilon(3,1,2,-1)*P(-1,2)-Epsilon(3,1,2,-1)*P(-1,1)') 

R2_GGVV = Lorentz(name = 'R2_GGVV',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,2)*Metric(3,4)+Metric(1,3)*Metric(2,4)+Metric(1,4)*Metric(2,3)')

R2_GGHH = Lorentz(name = 'R2_GGHH',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'Metric(1,2)')

R2_GGGVa = Lorentz(name = 'R2_GGGVa',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Epsilon(4,1,2,3)')

R2_VVVV1 = Lorentz(name = 'R2_VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Metric(1,2)*Metric(3,4)+Metric(1,3)*Metric(2,4)+Metric(1,4)*Metric(2,3)')

R2_VVVV2 = Lorentz(name = 'R2_VVVV2',
                   spins = [ 3, 3, 3, 3 ],
                   structure = 'Metric(1,2)*Metric(3,4)')

R2_VVVV3 = Lorentz(name = 'R2_VVVV3',
                   spins = [ 3, 3, 3, 3 ],
                   structure = 'Metric(1,3)*Metric(2,4)+Metric(1,4)*Metric(2,3)')

###################
# Base structures #
###################


UUS1 = Lorentz(name = 'UUS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

UUV1 = Lorentz(name = 'UUV1',
               spins = [ 1, 1, 3 ],
               structure = 'P(3,2) + P(3,3)')

SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) - ProjP(2,1)')

FFS3 = Lorentz(name = 'FFS3',
               spins = [ 2, 2, 1 ],
               structure = 'ProjP(2,1)')

FFS4 = Lorentz(name = 'FFS4',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) + ProjP(2,1)')

FFS5 = Lorentz(name = 'FFS5',
               spins = [ 2, 2, 1 ],
               structure = 'Identity(2,1)')

FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')

FFV2 = Lorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFV3 = Lorentz(name = 'FFV3',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV4 = Lorentz(name = 'FFV4',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV5 = Lorentz(name = 'FFV5',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 4*Gamma(3,2,-1)*ProjP(-1,1)')

FFV6 = Lorentz(name = 'FFV6',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

VSS1 = Lorentz(name = 'VSS1',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

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


##############################################
# UV CounterTerms Lorentz structures for QED #
# Generate by WriteUFO automatically         # 
##############################################

l_WmWpMass1 = Lorentz(name = 'l_WmWpMass1',
                      spins = [ 3, 3 ],
                      structure = '-Metric(1,2)*P(-1,1)*P(-1,1)')


l_WmWpMass2 = Lorentz(name = 'l_WmWpMass2',
                      spins = [ 3, 3 ],
                      structure = 'Metric(1,2)')


l_WmWpMass3 = Lorentz(name = 'l_WmWpMass3',
                      spins = [ 3, 3 ],
                      structure = '-P(1,1)*P(2,1)')


l_GpWmMass4 = Lorentz(name = 'l_GpWmMass4',
                      spins = [ 1, 3 ],
                      structure = 'P(2,1)')


l_GpWmMass5 = Lorentz(name = 'l_GpWmMass5',
                      spins = [ 1, 3 ],
                      structure = 'P(2,2)')


l_HMass6 = Lorentz(name = 'l_HMass6',
                   spins = [ 1, 1 ],
                   structure = '-P(-1,1)*P(-1,1)')


l_HMass7 = Lorentz(name = 'l_HMass7',
                   spins = [ 1, 1 ],
                   structure = '1')


l_vevexMass8 = Lorentz(name = 'l_vevexMass8',
                       spins = [ 2, 2 ],
                       structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)')


l_vevexMass9 = Lorentz(name = 'l_vevexMass9',
                       spins = [ 2, 2 ],
                       structure = 'P(-1,2)*Gamma(-1,2,-2)*ProjP(-2,1)')


l_vevexMass10 = Lorentz(name = 'l_vevexMass10',
                        spins = [ 2, 2 ],
                        structure = 'ProjM(2,1)')


l_vevexMass11 = Lorentz(name = 'l_vevexMass11',
                        spins = [ 2, 2 ],
                        structure = 'ProjP(2,1)')


l_WpWpWmWm12 = Lorentz(name = 'l_WpWpWmWm12',
                       spins = [ 3, 3, 3, 3 ],
                       structure = 'Metric(1,2)*Metric(3,4)')


l_WpWpWmWm13 = Lorentz(name = 'l_WpWpWmWm13',
                       spins = [ 3, 3, 3, 3 ],
                       structure = 'Metric(1,3)*Metric(2,4)')


l_WpWpWmWm14 = Lorentz(name = 'l_WpWpWmWm14',
                       spins = [ 3, 3, 3, 3 ],
                       structure = 'Metric(1,4)*Metric(2,3)')


l_AWpWm15 = Lorentz(name = 'l_AWpWm15',
                    spins = [ 3, 3, 3 ],
                    structure = 'Metric(1,2)*(P(3,2)-P(3,1))+Metric(2,3)*(P(1,3)-P(1,2))+Metric(3,1)*(P(2,1)-P(2,3))')


l_HHHH16 = Lorentz(name = 'l_HHHH16',
                   spins = [ 1, 1, 1, 1 ],
                   structure = '1')


l_HHH17 = Lorentz(name = 'l_HHH17',
                  spins = [ 1, 1, 1 ],
                  structure = '1')


l_HHWmWp18 = Lorentz(name = 'l_HHWmWp18',
                     spins = [ 1, 1, 3, 3 ],
                     structure = 'Metric(3,4)')


l_G0HA19 = Lorentz(name = 'l_G0HA19',
                   spins = [ 1, 1, 3 ],
                   structure = 'P(3,1)-P(3,2)')


l_HWpWm20 = Lorentz(name = 'l_HWpWm20',
                    spins = [ 1, 3, 3 ],
                    structure = 'Metric(2,3)')


l_vexveA21 = Lorentz(name = 'l_vexveA21',
                     spins = [ 2, 2, 3 ],
                     structure = 'Gamma(3,2,-1)*ProjM(-1,1)')


l_vexveA22 = Lorentz(name = 'l_vexveA22',
                     spins = [ 2, 2, 3 ],
                     structure = 'Gamma(3,2,-1)*ProjP(-1,1)')


l_epemH23 = Lorentz(name = 'l_epemH23',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_epemH24 = Lorentz(name = 'l_epemH24',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_umumxA25 = Lorentz(name = 'l_umumxA25',
                     spins = [ -1, -1, 3 ],
                     structure = 'P(3,1)')


l_umumxA26 = Lorentz(name = 'l_umumxA26',
                     spins = [ -1, -1, 3 ],
                     structure = 'P(3,2)')


l_HuZuZx27 = Lorentz(name = 'l_HuZuZx27',
                     spins = [ 1, -1, -1 ],
                     structure = '1')


