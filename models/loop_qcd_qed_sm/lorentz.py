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


l_GmWpMass4 = Lorentz(name = 'l_GmWpMass4',
                      spins = [ 1, 3 ],
                      structure = 'P(2,1)')


l_GmWpMass5 = Lorentz(name = 'l_GmWpMass5',
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
                       structure = 'Metric(1,4)*Metric(2,3)')


l_WpWpWmWm14 = Lorentz(name = 'l_WpWpWmWm14',
                       spins = [ 3, 3, 3, 3 ],
                       structure = 'Metric(1,3)*Metric(2,4)')


l_AWpWm15 = Lorentz(name = 'l_AWpWm15',
                    spins = [ 3, 3, 3 ],
                    structure = 'Metric(1,2)*(P(3,2)-P(3,1))+Metric(2,3)*(P(1,3)-P(1,2))+Metric(3,1)*(P(2,1)-P(2,3))')


l_HHHH16 = Lorentz(name = 'l_HHHH16',
                   spins = [ 1, 1, 1, 1 ],
                   structure = '1')


l_HHHH17 = Lorentz(name = 'l_HHHH17',
                   spins = [ 1, 1, 1, 1 ],
                   structure = '1')


l_HHG0G018 = Lorentz(name = 'l_HHG0G018',
                     spins = [ 1, 1, 1, 1 ],
                     structure = '1')


l_HHG0G019 = Lorentz(name = 'l_HHG0G019',
                     spins = [ 1, 1, 1, 1 ],
                     structure = '1')


l_HHGmGp20 = Lorentz(name = 'l_HHGmGp20',
                     spins = [ 1, 1, 1, 1 ],
                     structure = '1')


l_HHGmGp21 = Lorentz(name = 'l_HHGmGp21',
                     spins = [ 1, 1, 1, 1 ],
                     structure = '1')


l_G0G0G0G022 = Lorentz(name = 'l_G0G0G0G022',
                       spins = [ 1, 1, 1, 1 ],
                       structure = '1')


l_G0G0G0G023 = Lorentz(name = 'l_G0G0G0G023',
                       spins = [ 1, 1, 1, 1 ],
                       structure = '1')


l_G0G0GmGp24 = Lorentz(name = 'l_G0G0GmGp24',
                       spins = [ 1, 1, 1, 1 ],
                       structure = '1')


l_G0G0GmGp25 = Lorentz(name = 'l_G0G0GmGp25',
                       spins = [ 1, 1, 1, 1 ],
                       structure = '1')


l_GmGmGpGp26 = Lorentz(name = 'l_GmGmGpGp26',
                       spins = [ 1, 1, 1, 1 ],
                       structure = '1')


l_GmGmGpGp27 = Lorentz(name = 'l_GmGmGpGp27',
                       spins = [ 1, 1, 1, 1 ],
                       structure = '1')


l_HHH28 = Lorentz(name = 'l_HHH28',
                  spins = [ 1, 1, 1 ],
                  structure = '1')


l_HHH29 = Lorentz(name = 'l_HHH29',
                  spins = [ 1, 1, 1 ],
                  structure = '1')


l_HG0G030 = Lorentz(name = 'l_HG0G030',
                    spins = [ 1, 1, 1 ],
                    structure = '1')


l_HG0G031 = Lorentz(name = 'l_HG0G031',
                    spins = [ 1, 1, 1 ],
                    structure = '1')


l_GmHGp32 = Lorentz(name = 'l_GmHGp32',
                    spins = [ 1, 1, 1 ],
                    structure = '1')


l_GmHGp33 = Lorentz(name = 'l_GmHGp33',
                    spins = [ 1, 1, 1 ],
                    structure = '1')


l_HHWmWp34 = Lorentz(name = 'l_HHWmWp34',
                     spins = [ 1, 1, 3, 3 ],
                     structure = 'Metric(3,4)')


l_G0HA35 = Lorentz(name = 'l_G0HA35',
                   spins = [ 1, 1, 3 ],
                   structure = 'P(3,1)-P(3,2)')


l_HWpWm36 = Lorentz(name = 'l_HWpWm36',
                    spins = [ 1, 3, 3 ],
                    structure = 'Metric(2,3)')


l_vexveA37 = Lorentz(name = 'l_vexveA37',
                     spins = [ 2, 2, 3 ],
                     structure = 'Gamma(3,2,-1)*ProjM(-1,1)')


l_vexveA38 = Lorentz(name = 'l_vexveA38',
                     spins = [ 2, 2, 3 ],
                     structure = 'Gamma(3,2,-1)*ProjP(-1,1)')


l_epemH39 = Lorentz(name = 'l_epemH39',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_epemH40 = Lorentz(name = 'l_epemH40',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_epemH41 = Lorentz(name = 'l_epemH41',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_epemH42 = Lorentz(name = 'l_epemH42',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_epmmH43 = Lorentz(name = 'l_epmmH43',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_epmmH44 = Lorentz(name = 'l_epmmH44',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_epttmH45 = Lorentz(name = 'l_epttmH45',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_epttmH46 = Lorentz(name = 'l_epttmH46',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_mpemH47 = Lorentz(name = 'l_mpemH47',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_mpemH48 = Lorentz(name = 'l_mpemH48',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_mpmmH49 = Lorentz(name = 'l_mpmmH49',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_mpmmH50 = Lorentz(name = 'l_mpmmH50',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_mpmmH51 = Lorentz(name = 'l_mpmmH51',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_mpmmH52 = Lorentz(name = 'l_mpmmH52',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_mpttmH53 = Lorentz(name = 'l_mpttmH53',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_mpttmH54 = Lorentz(name = 'l_mpttmH54',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_ttpemH55 = Lorentz(name = 'l_ttpemH55',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_ttpemH56 = Lorentz(name = 'l_ttpemH56',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_ttpmmH57 = Lorentz(name = 'l_ttpmmH57',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_ttpmmH58 = Lorentz(name = 'l_ttpmmH58',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_ttpttmH59 = Lorentz(name = 'l_ttpttmH59',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_ttpttmH60 = Lorentz(name = 'l_ttpttmH60',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_ttpttmH61 = Lorentz(name = 'l_ttpttmH61',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_ttpttmH62 = Lorentz(name = 'l_ttpttmH62',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_uxuH63 = Lorentz(name = 'l_uxuH63',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_uxuH64 = Lorentz(name = 'l_uxuH64',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_uxuH65 = Lorentz(name = 'l_uxuH65',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_uxuH66 = Lorentz(name = 'l_uxuH66',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_uxcH67 = Lorentz(name = 'l_uxcH67',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_uxcH68 = Lorentz(name = 'l_uxcH68',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_uxtH69 = Lorentz(name = 'l_uxtH69',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_uxtH70 = Lorentz(name = 'l_uxtH70',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_cxuH71 = Lorentz(name = 'l_cxuH71',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_cxuH72 = Lorentz(name = 'l_cxuH72',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_cxcH73 = Lorentz(name = 'l_cxcH73',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_cxcH74 = Lorentz(name = 'l_cxcH74',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_cxcH75 = Lorentz(name = 'l_cxcH75',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_cxcH76 = Lorentz(name = 'l_cxcH76',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_cxtH77 = Lorentz(name = 'l_cxtH77',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_cxtH78 = Lorentz(name = 'l_cxtH78',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_txuH79 = Lorentz(name = 'l_txuH79',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_txuH80 = Lorentz(name = 'l_txuH80',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_txcH81 = Lorentz(name = 'l_txcH81',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_txcH82 = Lorentz(name = 'l_txcH82',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_txtH83 = Lorentz(name = 'l_txtH83',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_txtH84 = Lorentz(name = 'l_txtH84',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_txtH85 = Lorentz(name = 'l_txtH85',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_txtH86 = Lorentz(name = 'l_txtH86',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_dxdH87 = Lorentz(name = 'l_dxdH87',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_dxdH88 = Lorentz(name = 'l_dxdH88',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_dxdH89 = Lorentz(name = 'l_dxdH89',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_dxdH90 = Lorentz(name = 'l_dxdH90',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_dxsH91 = Lorentz(name = 'l_dxsH91',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_dxsH92 = Lorentz(name = 'l_dxsH92',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_dxbH93 = Lorentz(name = 'l_dxbH93',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_dxbH94 = Lorentz(name = 'l_dxbH94',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_sxdH95 = Lorentz(name = 'l_sxdH95',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_sxdH96 = Lorentz(name = 'l_sxdH96',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_sxsH97 = Lorentz(name = 'l_sxsH97',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_sxsH98 = Lorentz(name = 'l_sxsH98',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjP(2,1)')


l_sxsH99 = Lorentz(name = 'l_sxsH99',
                   spins = [ 2, 2, 1 ],
                   structure = 'ProjM(2,1)')


l_sxsH100 = Lorentz(name = 'l_sxsH100',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_sxbH101 = Lorentz(name = 'l_sxbH101',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_sxbH102 = Lorentz(name = 'l_sxbH102',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_bxdH103 = Lorentz(name = 'l_bxdH103',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_bxdH104 = Lorentz(name = 'l_bxdH104',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_bxsH105 = Lorentz(name = 'l_bxsH105',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_bxsH106 = Lorentz(name = 'l_bxsH106',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_bxbH107 = Lorentz(name = 'l_bxbH107',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_bxbH108 = Lorentz(name = 'l_bxbH108',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_bxbH109 = Lorentz(name = 'l_bxbH109',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjM(2,1)')


l_bxbH110 = Lorentz(name = 'l_bxbH110',
                    spins = [ 2, 2, 1 ],
                    structure = 'ProjP(2,1)')


l_epemG0111 = Lorentz(name = 'l_epemG0111',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_epemG0112 = Lorentz(name = 'l_epemG0112',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_epemG0113 = Lorentz(name = 'l_epemG0113',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_epemG0114 = Lorentz(name = 'l_epemG0114',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_epmmG0115 = Lorentz(name = 'l_epmmG0115',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_epmmG0116 = Lorentz(name = 'l_epmmG0116',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_epttmG0117 = Lorentz(name = 'l_epttmG0117',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_epttmG0118 = Lorentz(name = 'l_epttmG0118',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_mpemG0119 = Lorentz(name = 'l_mpemG0119',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_mpemG0120 = Lorentz(name = 'l_mpemG0120',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_mpmmG0121 = Lorentz(name = 'l_mpmmG0121',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_mpmmG0122 = Lorentz(name = 'l_mpmmG0122',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_mpmmG0123 = Lorentz(name = 'l_mpmmG0123',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_mpmmG0124 = Lorentz(name = 'l_mpmmG0124',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_mpttmG0125 = Lorentz(name = 'l_mpttmG0125',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_mpttmG0126 = Lorentz(name = 'l_mpttmG0126',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_ttpemG0127 = Lorentz(name = 'l_ttpemG0127',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_ttpemG0128 = Lorentz(name = 'l_ttpemG0128',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_ttpmmG0129 = Lorentz(name = 'l_ttpmmG0129',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_ttpmmG0130 = Lorentz(name = 'l_ttpmmG0130',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_ttpttmG0131 = Lorentz(name = 'l_ttpttmG0131',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjM(2,1)')


l_ttpttmG0132 = Lorentz(name = 'l_ttpttmG0132',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjP(2,1)')


l_ttpttmG0133 = Lorentz(name = 'l_ttpttmG0133',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjM(2,1)')


l_ttpttmG0134 = Lorentz(name = 'l_ttpttmG0134',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjP(2,1)')


l_uxuG0135 = Lorentz(name = 'l_uxuG0135',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxuG0136 = Lorentz(name = 'l_uxuG0136',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxuG0137 = Lorentz(name = 'l_uxuG0137',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxuG0138 = Lorentz(name = 'l_uxuG0138',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxcG0139 = Lorentz(name = 'l_uxcG0139',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxcG0140 = Lorentz(name = 'l_uxcG0140',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxtG0141 = Lorentz(name = 'l_uxtG0141',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxtG0142 = Lorentz(name = 'l_uxtG0142',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxuG0143 = Lorentz(name = 'l_cxuG0143',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxuG0144 = Lorentz(name = 'l_cxuG0144',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxcG0145 = Lorentz(name = 'l_cxcG0145',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxcG0146 = Lorentz(name = 'l_cxcG0146',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxcG0147 = Lorentz(name = 'l_cxcG0147',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxcG0148 = Lorentz(name = 'l_cxcG0148',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxtG0149 = Lorentz(name = 'l_cxtG0149',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxtG0150 = Lorentz(name = 'l_cxtG0150',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txuG0151 = Lorentz(name = 'l_txuG0151',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txuG0152 = Lorentz(name = 'l_txuG0152',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txcG0153 = Lorentz(name = 'l_txcG0153',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txcG0154 = Lorentz(name = 'l_txcG0154',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txtG0155 = Lorentz(name = 'l_txtG0155',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txtG0156 = Lorentz(name = 'l_txtG0156',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txtG0157 = Lorentz(name = 'l_txtG0157',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txtG0158 = Lorentz(name = 'l_txtG0158',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxdG0159 = Lorentz(name = 'l_dxdG0159',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxdG0160 = Lorentz(name = 'l_dxdG0160',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxdG0161 = Lorentz(name = 'l_dxdG0161',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxdG0162 = Lorentz(name = 'l_dxdG0162',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxsG0163 = Lorentz(name = 'l_dxsG0163',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxsG0164 = Lorentz(name = 'l_dxsG0164',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxbG0165 = Lorentz(name = 'l_dxbG0165',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxbG0166 = Lorentz(name = 'l_dxbG0166',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxdG0167 = Lorentz(name = 'l_sxdG0167',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxdG0168 = Lorentz(name = 'l_sxdG0168',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxsG0169 = Lorentz(name = 'l_sxsG0169',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxsG0170 = Lorentz(name = 'l_sxsG0170',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxsG0171 = Lorentz(name = 'l_sxsG0171',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxsG0172 = Lorentz(name = 'l_sxsG0172',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxbG0173 = Lorentz(name = 'l_sxbG0173',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxbG0174 = Lorentz(name = 'l_sxbG0174',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxdG0175 = Lorentz(name = 'l_bxdG0175',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxdG0176 = Lorentz(name = 'l_bxdG0176',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxsG0177 = Lorentz(name = 'l_bxsG0177',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxsG0178 = Lorentz(name = 'l_bxsG0178',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxbG0179 = Lorentz(name = 'l_bxbG0179',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxbG0180 = Lorentz(name = 'l_bxbG0180',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxbG0181 = Lorentz(name = 'l_bxbG0181',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxbG0182 = Lorentz(name = 'l_bxbG0182',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxdGp183 = Lorentz(name = 'l_uxdGp183',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxdGp184 = Lorentz(name = 'l_uxdGp184',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxdGp185 = Lorentz(name = 'l_uxdGp185',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxdGp186 = Lorentz(name = 'l_uxdGp186',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxsGp187 = Lorentz(name = 'l_uxsGp187',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxsGp188 = Lorentz(name = 'l_uxsGp188',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxsGp189 = Lorentz(name = 'l_uxsGp189',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxsGp190 = Lorentz(name = 'l_uxsGp190',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxbGp191 = Lorentz(name = 'l_uxbGp191',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxbGp192 = Lorentz(name = 'l_uxbGp192',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_uxbGp193 = Lorentz(name = 'l_uxbGp193',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_uxbGp194 = Lorentz(name = 'l_uxbGp194',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxdGp195 = Lorentz(name = 'l_cxdGp195',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxdGp196 = Lorentz(name = 'l_cxdGp196',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxdGp197 = Lorentz(name = 'l_cxdGp197',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxdGp198 = Lorentz(name = 'l_cxdGp198',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxsGp199 = Lorentz(name = 'l_cxsGp199',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxsGp200 = Lorentz(name = 'l_cxsGp200',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxsGp201 = Lorentz(name = 'l_cxsGp201',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxsGp202 = Lorentz(name = 'l_cxsGp202',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxbGp203 = Lorentz(name = 'l_cxbGp203',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxbGp204 = Lorentz(name = 'l_cxbGp204',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_cxbGp205 = Lorentz(name = 'l_cxbGp205',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_cxbGp206 = Lorentz(name = 'l_cxbGp206',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txdGp207 = Lorentz(name = 'l_txdGp207',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txdGp208 = Lorentz(name = 'l_txdGp208',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txdGp209 = Lorentz(name = 'l_txdGp209',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txdGp210 = Lorentz(name = 'l_txdGp210',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txsGp211 = Lorentz(name = 'l_txsGp211',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txsGp212 = Lorentz(name = 'l_txsGp212',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txsGp213 = Lorentz(name = 'l_txsGp213',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txsGp214 = Lorentz(name = 'l_txsGp214',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txbGp215 = Lorentz(name = 'l_txbGp215',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txbGp216 = Lorentz(name = 'l_txbGp216',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_txbGp217 = Lorentz(name = 'l_txbGp217',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_txbGp218 = Lorentz(name = 'l_txbGp218',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxuGm219 = Lorentz(name = 'l_dxuGm219',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxuGm220 = Lorentz(name = 'l_dxuGm220',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxuGm221 = Lorentz(name = 'l_dxuGm221',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxuGm222 = Lorentz(name = 'l_dxuGm222',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxcGm223 = Lorentz(name = 'l_dxcGm223',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxcGm224 = Lorentz(name = 'l_dxcGm224',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxcGm225 = Lorentz(name = 'l_dxcGm225',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxcGm226 = Lorentz(name = 'l_dxcGm226',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxtGm227 = Lorentz(name = 'l_dxtGm227',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxtGm228 = Lorentz(name = 'l_dxtGm228',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_dxtGm229 = Lorentz(name = 'l_dxtGm229',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_dxtGm230 = Lorentz(name = 'l_dxtGm230',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxuGm231 = Lorentz(name = 'l_sxuGm231',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxuGm232 = Lorentz(name = 'l_sxuGm232',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxuGm233 = Lorentz(name = 'l_sxuGm233',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxuGm234 = Lorentz(name = 'l_sxuGm234',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxcGm235 = Lorentz(name = 'l_sxcGm235',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxcGm236 = Lorentz(name = 'l_sxcGm236',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxcGm237 = Lorentz(name = 'l_sxcGm237',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxcGm238 = Lorentz(name = 'l_sxcGm238',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxtGm239 = Lorentz(name = 'l_sxtGm239',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxtGm240 = Lorentz(name = 'l_sxtGm240',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_sxtGm241 = Lorentz(name = 'l_sxtGm241',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_sxtGm242 = Lorentz(name = 'l_sxtGm242',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxuGm243 = Lorentz(name = 'l_bxuGm243',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxuGm244 = Lorentz(name = 'l_bxuGm244',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxuGm245 = Lorentz(name = 'l_bxuGm245',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxuGm246 = Lorentz(name = 'l_bxuGm246',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxcGm247 = Lorentz(name = 'l_bxcGm247',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxcGm248 = Lorentz(name = 'l_bxcGm248',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxcGm249 = Lorentz(name = 'l_bxcGm249',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxcGm250 = Lorentz(name = 'l_bxcGm250',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxtGm251 = Lorentz(name = 'l_bxtGm251',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxtGm252 = Lorentz(name = 'l_bxtGm252',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_bxtGm253 = Lorentz(name = 'l_bxtGm253',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjM(2,1)')


l_bxtGm254 = Lorentz(name = 'l_bxtGm254',
                     spins = [ 2, 2, 1 ],
                     structure = 'ProjP(2,1)')


l_vexemGp255 = Lorentz(name = 'l_vexemGp255',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_vexemGp256 = Lorentz(name = 'l_vexemGp256',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_vexemGp257 = Lorentz(name = 'l_vexemGp257',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_vexemGp258 = Lorentz(name = 'l_vexemGp258',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_vmxmmGp259 = Lorentz(name = 'l_vmxmmGp259',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_vmxmmGp260 = Lorentz(name = 'l_vmxmmGp260',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_vmxmmGp261 = Lorentz(name = 'l_vmxmmGp261',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_vmxmmGp262 = Lorentz(name = 'l_vmxmmGp262',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_vtxttmGp263 = Lorentz(name = 'l_vtxttmGp263',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjM(2,1)')


l_vtxttmGp264 = Lorentz(name = 'l_vtxttmGp264',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjP(2,1)')


l_vtxttmGp265 = Lorentz(name = 'l_vtxttmGp265',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjM(2,1)')


l_vtxttmGp266 = Lorentz(name = 'l_vtxttmGp266',
                        spins = [ 2, 2, 1 ],
                        structure = 'ProjP(2,1)')


l_epveGm267 = Lorentz(name = 'l_epveGm267',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_epveGm268 = Lorentz(name = 'l_epveGm268',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_epveGm269 = Lorentz(name = 'l_epveGm269',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_epveGm270 = Lorentz(name = 'l_epveGm270',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_mpvmGm271 = Lorentz(name = 'l_mpvmGm271',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_mpvmGm272 = Lorentz(name = 'l_mpvmGm272',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_mpvmGm273 = Lorentz(name = 'l_mpvmGm273',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjM(2,1)')


l_mpvmGm274 = Lorentz(name = 'l_mpvmGm274',
                      spins = [ 2, 2, 1 ],
                      structure = 'ProjP(2,1)')


l_ttpvtGm275 = Lorentz(name = 'l_ttpvtGm275',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_ttpvtGm276 = Lorentz(name = 'l_ttpvtGm276',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_ttpvtGm277 = Lorentz(name = 'l_ttpvtGm277',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjM(2,1)')


l_ttpvtGm278 = Lorentz(name = 'l_ttpvtGm278',
                       spins = [ 2, 2, 1 ],
                       structure = 'ProjP(2,1)')


l_umxumA279 = Lorentz(name = 'l_umxumA279',
                      spins = [ -1, -1, 3 ],
                      structure = 'P(3,1)')


l_umxumA280 = Lorentz(name = 'l_umxumA280',
                      spins = [ -1, -1, 3 ],
                      structure = 'P(3,2)')


l_HuZxuZ281 = Lorentz(name = 'l_HuZxuZ281',
                      spins = [ 1, -1, -1 ],
                      structure = '1')


l_Humxum282 = Lorentz(name = 'l_Humxum282',
                      spins = [ 1, -1, -1 ],
                      structure = '1')


l_Hupxup283 = Lorentz(name = 'l_Hupxup283',
                      spins = [ 1, -1, -1 ],
                      structure = '1')


l_G0upxup284 = Lorentz(name = 'l_G0upxup284',
                       spins = [ 1, -1, -1 ],
                       structure = '1')


l_G0umxum285 = Lorentz(name = 'l_G0umxum285',
                       spins = [ 1, -1, -1 ],
                       structure = '1')


l_GpuZxum286 = Lorentz(name = 'l_GpuZxum286',
                       spins = [ 1, -1, -1 ],
                       structure = '1')


l_GmuZxup287 = Lorentz(name = 'l_GmuZxup287',
                       spins = [ 1, -1, -1 ],
                       structure = '1')


l_GpupxuZ288 = Lorentz(name = 'l_GpupxuZ288',
                       spins = [ 1, -1, -1 ],
                       structure = '1')


l_GmumxuZ289 = Lorentz(name = 'l_GmumxuZ289',
                       spins = [ 1, -1, -1 ],
                       structure = '1')


l_Gpupxugamma290 = Lorentz(name = 'l_Gpupxugamma290',
                           spins = [ 1, -1, -1 ],
                           structure = '1')


l_Gmumxugamma291 = Lorentz(name = 'l_Gmumxugamma291',
                           spins = [ 1, -1, -1 ],
                           structure = '1')


