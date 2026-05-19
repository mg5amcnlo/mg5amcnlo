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













FFFF17 = Lorentz(name = 'FFFF17',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjM(-3,1)*ProjM(-2,3)')

FFFF18 = Lorentz(name = 'FFFF18',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjM(-3,3)*ProjM(-2,1)')

FFFF19 = Lorentz(name = 'FFFF19',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,2,-6)*Gamma(-1,-6,-5)*Gamma(-1,4,-4)*ProjM(-5,1)*ProjM(-3,3)')

FFFF20 = Lorentz(name = 'FFFF20',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,4,-6)*Gamma(-1,-6,-5)*Gamma(-1,2,-4)*ProjM(-5,3)*ProjM(-3,1)')

FFFF21 = Lorentz(name = 'FFFF21',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-6,-5)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*Gamma(-1,4,-6)*ProjM(-5,3)*ProjM(-3,1)')

FFFF22 = Lorentz(name = 'FFFF22',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'ProjM(4,3)*ProjP(2,1)')

FFFF23 = Lorentz(name = 'FFFF23',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'ProjM(2,1)*ProjP(4,3)')

FFFF24 = Lorentz(name = 'FFFF24',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-3)*Gamma(-1,4,-2)*ProjM(-2,3)*ProjP(-3,1)')

FFFF25 = Lorentz(name = 'FFFF25',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjM(-2,3)*ProjP(-3,1)')

FFFF26 = Lorentz(name = 'FFFF26',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjP(-3,1)*ProjP(-2,3)')

FFFF27 = Lorentz(name = 'FFFF27',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-3)*Gamma(-1,4,-2)*ProjM(-2,1)*ProjP(-3,3)')

FFFF28 = Lorentz(name = 'FFFF28',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjM(-2,1)*ProjP(-3,3)')

FFFF29 = Lorentz(name = 'FFFF29',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjP(-3,3)*ProjP(-2,1)')

FFFF30 = Lorentz(name = 'FFFF30',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,2,-6)*Gamma(-1,-6,-5)*Gamma(-1,4,-4)*ProjP(-5,1)*ProjP(-3,3)')

FFFF31 = Lorentz(name = 'FFFF31',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-4,-3)*Gamma(-2,4,-6)*Gamma(-1,-6,-5)*Gamma(-1,2,-4)*ProjP(-5,3)*ProjP(-3,1)')

FFFF32 = Lorentz(name = 'FFFF32',
                 spins = [ 2, 2, 2, 2 ],
                 structure = 'Gamma(-2,-6,-5)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*Gamma(-1,4,-6)*ProjP(-5,3)*ProjP(-3,1)')




VVV8 = Lorentz(name = 'VVV8',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')



VVV9 = Lorentz(name = 'VVV9',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2) - P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) + P(1,3)*Metric(2,3)')

VVV10 = Lorentz(name = 'VVV10',
                spins = [ 3, 3, 3 ],
                structure = '-(P(1,2)*P(2,3)*P(3,1)) + P(1,3)*P(2,1)*P(3,2) + P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2) - P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2) - P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3) + P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3) + P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3) - P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3)')


VVS11 = Lorentz(name = 'VVS11',
                spins = [ 3, 3, 1 ],
                structure = 'Metric(1,2)')

VVS12 = Lorentz(name = 'VVS12',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')


VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(1,2)*Metric(2,3)')


VSS5 = Lorentz(name = 'VSS5',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')


VSS6 = Lorentz(name = 'VSS6',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) + P(1,3)/3.')


VSS4 = Lorentz(name = 'VSS4',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2)')


SSS6 = Lorentz(name = 'SSS6',
               spins = [ 1, 1, 1 ],
               structure = '1')

SSS7 = Lorentz(name = 'SSS7',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2)')


SSS9 = Lorentz(name = 'SSS9',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')


FFV70 = Lorentz(name = 'FFV70',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1)')


FFV83 = Lorentz(name = 'FFV83',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)')


FFV87 = Lorentz(name = 'FFV87',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')


FFV82 = Lorentz(name = 'FFV82',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFV109 = Lorentz(name = 'FFV109',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')


FFV80 = Lorentz(name = 'FFV80',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')



FFV91 = Lorentz(name = 'FFV91',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 4*Gamma(3,2,-1)*ProjP(-1,1)')


FFS45 = Lorentz(name = 'FFS45',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS52 = Lorentz(name = 'FFS52',
                spins = [ 2, 2, 1 ],
                structure = 'ProjP(2,1)')


FFS41 = Lorentz(name = 'FFS41',
                spins = [ 2, 2, 1 ],
                structure = 'ProjM(2,1)')


FFS51 = Lorentz(name = 'FFS51',
                spins = [ 2, 2, 1 ],
                structure = 'ProjM(2,1) - ProjP(2,1)')

FFS55 = Lorentz(name = 'FFS55',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')


FFS37 = Lorentz(name = 'FFS37',
                spins = [ 2, 2, 1 ],
                structure = 'Identity(2,1)')


FFV110 = Lorentz(name = 'FFV110',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')
