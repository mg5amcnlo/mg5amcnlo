# This file was automatically created by FeynRules 2.4.78
# Mathematica version: 12.0.0 for Mac OS X x86 (64-bit) (April 7, 2019)
# Date: Wed 1 Apr 2020 19:35:45


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot
try:
   import form_factors as ForFac 
except ImportError:
   pass


FF4 = Lorentz(name = 'FF4',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)*Gamma(-1,2,1)')

FF5 = Lorentz(name = 'FF5',
              spins = [ 2, 2 ],
              structure = 'Identity(2,1)')

FF6 = Lorentz(name = 'FF6',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)**2*Identity(2,1)')

VV5 = Lorentz(name = 'VV5',
              spins = [ 3, 3 ],
              structure = 'Metric(1,2)')

VV6 = Lorentz(name = 'VV6',
              spins = [ 3, 3 ],
              structure = 'P(-1,2)**2*Metric(1,2)')

VV7 = Lorentz(name = 'VV7',
              spins = [ 3, 3 ],
              structure = 'P(1,2)*P(2,2) - (3*P(-1,2)**2*Metric(1,2))/2.')

VV8 = Lorentz(name = 'VV8',
              spins = [ 3, 3 ],
              structure = 'P(1,2)*P(2,2) - P(-1,2)**2*Metric(1,2)')

UUS2 = Lorentz(name = 'UUS2',
               spins = [ -1, -1, 1 ],
               structure = '1')

UUV2 = Lorentz(name = 'UUV2',
               spins = [ -1, -1, 3 ],
               structure = 'P(3,2) + P(3,3)')

SSS6 = Lorentz(name = 'SSS6',
               spins = [ 1, 1, 1 ],
               structure = '1')

SSS7 = Lorentz(name = 'SSS7',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2)')

SSS8 = Lorentz(name = 'SSS8',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,2) - P(-1,1)*P(-1,3)')

SSS9 = Lorentz(name = 'SSS9',
               spins = [ 1, 1, 1 ],
               structure = 'P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')

SSS10 = Lorentz(name = 'SSS10',
                spins = [ 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3)')

FFS31 = Lorentz(name = 'FFS31',
                spins = [ 2, 2, 1 ],
                structure = 'Gamma5(2,1)')

FFS32 = Lorentz(name = 'FFS32',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)**2*Gamma5(2,1)')

FFS33 = Lorentz(name = 'FFS33',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,1) + (P(-1,3)*Gamma(-1,2,1))/2.')

FFS34 = Lorentz(name = 'FFS34',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-2)')

FFS35 = Lorentz(name = 'FFS35',
                spins = [ 2, 2, 1 ],
                structure = '6*P(-1,1)**2*Gamma5(2,1) + 4*P(-1,1)*P(-1,3)*Gamma5(2,1) + 3*P(-1,3)**2*Gamma5(2,1) + P(-2,3)*P(-1,1)*Gamma5(-3,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3) + P(-2,3)*P(-1,1)*Gamma5(-3,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)')

FFS36 = Lorentz(name = 'FFS36',
                spins = [ 2, 2, 1 ],
                structure = '9*P(-1,1)**2*Gamma5(2,1) + 7*P(-1,1)*P(-1,3)*Gamma5(2,1) + (9*P(-1,3)**2*Gamma5(2,1))/2. + P(-2,3)*P(-1,1)*Gamma5(-3,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3) + P(-2,3)*P(-1,1)*Gamma5(-3,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)')

FFS37 = Lorentz(name = 'FFS37',
                spins = [ 2, 2, 1 ],
                structure = 'Identity(2,1)')

FFS38 = Lorentz(name = 'FFS38',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)**2*Identity(2,1)')

FFS39 = Lorentz(name = 'FFS39',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-3)*Gamma(-1,-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-3,1)*Gamma(-1,2,-3) + 6*P(-1,1)**2*Identity(2,1) + 4*P(-1,1)*P(-1,3)*Identity(2,1) + 3*P(-1,3)**2*Identity(2,1)')

FFS40 = Lorentz(name = 'FFS40',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-3)*Gamma(-1,-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-3,1)*Gamma(-1,2,-3) + 9*P(-1,1)**2*Identity(2,1) + 7*P(-1,1)*P(-1,3)*Identity(2,1) + (9*P(-1,3)**2*Identity(2,1))/2.')

FFS41 = Lorentz(name = 'FFS41',
                spins = [ 2, 2, 1 ],
                structure = 'ProjM(2,1)')

FFS42 = Lorentz(name = 'FFS42',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)**2*ProjM(2,1)')

FFS43 = Lorentz(name = 'FFS43',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)**2*ProjM(2,1)')

FFS44 = Lorentz(name = 'FFS44',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS45 = Lorentz(name = 'FFS45',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS46 = Lorentz(name = 'FFS46',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFS47 = Lorentz(name = 'FFS47',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFS48 = Lorentz(name = 'FFS48',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + (3*P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFS49 = Lorentz(name = 'FFS49',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjM(-3,1) + 3*P(-1,1)**2*ProjM(2,1) + 4*P(-1,1)*P(-1,3)*ProjM(2,1) + 3*P(-1,3)**2*ProjM(2,1)')

FFS50 = Lorentz(name = 'FFS50',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjM(-3,1) + (9*P(-1,1)**2*ProjM(2,1))/2. + 7*P(-1,1)*P(-1,3)*ProjM(2,1) + (9*P(-1,3)**2*ProjM(2,1))/2.')

FFS51 = Lorentz(name = 'FFS51',
                spins = [ 2, 2, 1 ],
                structure = 'ProjM(2,1) - ProjP(2,1)')

FFS52 = Lorentz(name = 'FFS52',
                spins = [ 2, 2, 1 ],
                structure = 'ProjP(2,1)')

FFS53 = Lorentz(name = 'FFS53',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,1)**2*ProjP(2,1)')

FFS54 = Lorentz(name = 'FFS54',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)**2*ProjP(2,1)')

FFS55 = Lorentz(name = 'FFS55',
                spins = [ 2, 2, 1 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS56 = Lorentz(name = 'FFS56',
                spins = [ 2, 2, 1 ],
                structure = '-(P(-1,1)*Gamma(-1,2,1)) + P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS57 = Lorentz(name = 'FFS57',
                spins = [ 2, 2, 1 ],
                structure = '-(P(-1,1)*Gamma(-1,2,1)) - P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFS58 = Lorentz(name = 'FFS58',
                spins = [ 2, 2, 1 ],
                structure = '6*P(-1,1)**2*Identity(2,1) + 4*P(-1,1)*P(-1,3)*Identity(2,1) + 3*P(-1,3)**2*Identity(2,1) + P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjP(-3,1)')

FFS59 = Lorentz(name = 'FFS59',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjP(-3,1) + 3*P(-1,1)**2*ProjP(2,1) + 4*P(-1,1)*P(-1,3)*ProjP(2,1) + 3*P(-1,3)**2*ProjP(2,1)')

FFS60 = Lorentz(name = 'FFS60',
                spins = [ 2, 2, 1 ],
                structure = 'P(-2,3)*P(-1,1)*Gamma(-2,2,-4)*Gamma(-1,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-4)*ProjP(-3,1) + (9*P(-1,1)**2*ProjP(2,1))/2. + 7*P(-1,1)*P(-1,3)*ProjP(2,1) + (9*P(-1,3)**2*ProjP(2,1))/2.')

FFV56 = Lorentz(name = 'FFV56',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,3)*Gamma5(2,1)')

FFV57 = Lorentz(name = 'FFV57',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,1)')

FFV58 = Lorentz(name = 'FFV58',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)**2*Gamma(3,2,1)')

FFV59 = Lorentz(name = 'FFV59',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,1) + P(-1,3)**2*Gamma(3,2,1)')

FFV60 = Lorentz(name = 'FFV60',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1)')

FFV61 = Lorentz(name = 'FFV61',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,1)*Identity(2,1)')

FFV62 = Lorentz(name = 'FFV62',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,3)*Identity(2,1)')

FFV63 = Lorentz(name = 'FFV63',
                spins = [ 2, 2, 3 ],
                structure = '(329*P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/66. + P(3,1)*Identity(2,1) - (12*P(3,2)*Identity(2,1))/11. - (166*P(3,3)*Identity(2,1))/33.')

FFV64 = Lorentz(name = 'FFV64',
                spins = [ 2, 2, 3 ],
                structure = '(53*P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/16. + P(3,1)*Identity(2,1) - (21*P(3,2)*Identity(2,1))/16. - (111*P(3,3)*Identity(2,1))/32.')

FFV65 = Lorentz(name = 'FFV65',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,1)*Identity(2,1) + (2*P(3,3)*Identity(2,1))/3.')

FFV66 = Lorentz(name = 'FFV66',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/6. + P(3,1)*Identity(2,1) + (2*P(3,3)*Identity(2,1))/3.')

FFV67 = Lorentz(name = 'FFV67',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1)) + P(3,3)*Identity(2,1)')

FFV68 = Lorentz(name = 'FFV68',
                spins = [ 2, 2, 3 ],
                structure = '(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/3. + P(3,3)*Identity(2,1)')

FFV69 = Lorentz(name = 'FFV69',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,3)*ProjM(2,1)')

FFV70 = Lorentz(name = 'FFV70',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFV71 = Lorentz(name = 'FFV71',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1)')

FFV72 = Lorentz(name = 'FFV72',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) - 3*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1)')

FFV73 = Lorentz(name = 'FFV73',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1)')

FFV74 = Lorentz(name = 'FFV74',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) + (3*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1))/2.')

FFV75 = Lorentz(name = 'FFV75',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) + 3*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1)')

FFV76 = Lorentz(name = 'FFV76',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(3,3)*ProjM(2,1)')

FFV77 = Lorentz(name = 'FFV77',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. + P(3,3)*ProjM(2,1)')

FFV78 = Lorentz(name = 'FFV78',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/6. + P(3,1)*ProjM(2,1) + (2*P(3,3)*ProjM(2,1))/3.')

FFV79 = Lorentz(name = 'FFV79',
                spins = [ 2, 2, 3 ],
                structure = '(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(3,1)*ProjM(2,1) + (4*P(3,3)*ProjM(2,1))/11.')

FFV80 = Lorentz(name = 'FFV80',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFV81 = Lorentz(name = 'FFV81',
                spins = [ 2, 2, 3 ],
                structure = 'P(3,3)*ProjP(2,1)')

FFV82 = Lorentz(name = 'FFV82',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFV83 = Lorentz(name = 'FFV83',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV84 = Lorentz(name = 'FFV84',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) - (5*Gamma(3,2,-1)*ProjP(-1,1))/3.')

FFV85 = Lorentz(name = 'FFV85',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) - Gamma(3,2,-1)*ProjP(-1,1)')

FFV86 = Lorentz(name = 'FFV86',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + (2*Gamma(3,2,-1)*ProjP(-1,1))/3.')

FFV87 = Lorentz(name = 'FFV87',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFV88 = Lorentz(name = 'FFV88',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + (8*Gamma(3,2,-1)*ProjP(-1,1))/5.')

FFV89 = Lorentz(name = 'FFV89',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV90 = Lorentz(name = 'FFV90',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,1) + (16*Gamma(3,2,-1)*ProjM(-1,1))/7. + (16*Gamma(3,2,-1)*ProjP(-1,1))/7.')

FFV91 = Lorentz(name = 'FFV91',
                spins = [ 2, 2, 3 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 4*Gamma(3,2,-1)*ProjP(-1,1)')

FFV92 = Lorentz(name = 'FFV92',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV93 = Lorentz(name = 'FFV93',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFV94 = Lorentz(name = 'FFV94',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) + P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV95 = Lorentz(name = 'FFV95',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,1) + P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) + P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV96 = Lorentz(name = 'FFV96',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) + P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV97 = Lorentz(name = 'FFV97',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) + (3*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1))/2.')

FFV98 = Lorentz(name = 'FFV98',
                spins = [ 2, 2, 3 ],
                structure = 'P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) + P(-1,3)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) + 3*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV99 = Lorentz(name = 'FFV99',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(3,3)*ProjP(2,1)')

FFV100 = Lorentz(name = 'FFV100',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(3,3)*ProjM(2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(3,3)*ProjP(2,1)')

FFV101 = Lorentz(name = 'FFV101',
                 spins = [ 2, 2, 3 ],
                 structure = '(281*P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/66. + P(3,1)*Identity(2,1) - (12*P(3,2)*Identity(2,1))/11. - (118*P(3,3)*Identity(2,1))/33. - (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. - (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11.')

FFV102 = Lorentz(name = 'FFV102',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/2. + P(3,3)*Identity(2,1) - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2.')

FFV103 = Lorentz(name = 'FFV103',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. + P(3,3)*ProjM(2,1) - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2.')

FFV104 = Lorentz(name = 'FFV104',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/2. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. + P(3,3)*ProjM(2,1) - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2. + P(3,3)*ProjP(2,1)')

FFV105 = Lorentz(name = 'FFV105',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/6. + P(3,1)*ProjP(2,1) + (2*P(3,3)*ProjP(2,1))/3.')

FFV106 = Lorentz(name = 'FFV106',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + P(3,1)*ProjP(2,1) + (5*P(3,3)*ProjP(2,1))/13.')

FFV107 = Lorentz(name = 'FFV107',
                 spins = [ 2, 2, 3 ],
                 structure = '(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(3,1)*ProjM(2,1) + (4*P(3,3)*ProjM(2,1))/11. - (P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (13*P(3,1)*ProjP(2,1))/11. + (5*P(3,3)*ProjP(2,1))/11.')

FFV108 = Lorentz(name = 'FFV108',
                 spins = [ 2, 2, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1) + P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFV109 = Lorentz(name = 'FFV109',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV110 = Lorentz(name = 'FFV110',
                 spins = [ 2, 2, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

VSS4 = Lorentz(name = 'VSS4',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2)')

VSS5 = Lorentz(name = 'VSS5',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

VSS6 = Lorentz(name = 'VSS6',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) + P(1,3)/3.')

VVS9 = Lorentz(name = 'VVS9',
               spins = [ 3, 3, 1 ],
               structure = 'Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2)')

VVS10 = Lorentz(name = 'VVS10',
                spins = [ 3, 3, 1 ],
                structure = 'Epsilon(1,2,-1,-2)*P(-2,2)*P(-1,1) + (Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1))/2. + (Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2))/2.')

VVS11 = Lorentz(name = 'VVS11',
                spins = [ 3, 3, 1 ],
                structure = 'Metric(1,2)')

VVS12 = Lorentz(name = 'VVS12',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVS13 = Lorentz(name = 'VVS13',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,1) - (P(-1,1)*P(-1,2)*Metric(1,2))/2. + (P(-1,2)**2*Metric(1,2))/2. + (P(-1,2)*P(-1,3)*Metric(1,2))/2.')

VVS14 = Lorentz(name = 'VVS14',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,1) - (P(1,3)*P(2,1))/2. - (P(1,2)*P(2,3))/2. - (P(-1,1)*P(-1,2)*Metric(1,2))/2. + (P(-1,2)**2*Metric(1,2))/2. + (P(-1,1)*P(-1,3)*Metric(1,2))/2. + P(-1,2)*P(-1,3)*Metric(1,2)')

VVS15 = Lorentz(name = 'VVS15',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,1) - (9*P(1,3)*P(2,1))/26. - (9*P(1,2)*P(2,3))/26. - (17*P(-1,1)*P(-1,2)*Metric(1,2))/26. + (9*P(-1,2)**2*Metric(1,2))/26. + (3*P(-1,1)*P(-1,3)*Metric(1,2))/26. + (6*P(-1,2)*P(-1,3)*Metric(1,2))/13. - (3*P(-1,3)**2*Metric(1,2))/13.')

VVS16 = Lorentz(name = 'VVS16',
                spins = [ 3, 3, 1 ],
                structure = 'P(1,2)*P(2,1) - (13*P(1,3)*P(2,1))/62. - (13*P(1,2)*P(2,3))/62. + (P(1,3)*P(2,3))/62. - (20*P(-1,1)*P(-1,2)*Metric(1,2))/31. + (19*P(-1,2)**2*Metric(1,2))/62. + (2*P(-1,1)*P(-1,3)*Metric(1,2))/31. + (23*P(-1,2)*P(-1,3)*Metric(1,2))/62. - (13*P(-1,3)**2*Metric(1,2))/62.')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2)')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(1,2)*Metric(2,3)')

VVV8 = Lorentz(name = 'VVV8',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVV9 = Lorentz(name = 'VVV9',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,2)*Metric(1,2) - P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) + P(1,3)*Metric(2,3)')

VVV10 = Lorentz(name = 'VVV10',
                spins = [ 3, 3, 3 ],
                structure = '-(P(1,2)*P(2,3)*P(3,1)) + P(1,3)*P(2,1)*P(3,2) + P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2) - P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2) - P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3) + P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3) + P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3) - P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3)')

SSSS7 = Lorentz(name = 'SSSS7',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

SSSS8 = Lorentz(name = 'SSSS8',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4)')

SSSS9 = Lorentz(name = 'SSSS9',
                spins = [ 1, 1, 1, 1 ],
                structure = 'P(-1,1)*P(-1,2) + P(-1,3)*P(-1,4)')

SSSS10 = Lorentz(name = 'SSSS10',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-1,1)*P(-1,2) - P(-1,1)*P(-1,3) - P(-1,2)*P(-1,4) + P(-1,3)*P(-1,4)')

SSSS11 = Lorentz(name = 'SSSS11',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-1,1)*P(-1,2) + (P(-1,1)*P(-1,3))/2. + (P(-1,2)*P(-1,3))/2. + (P(-1,1)*P(-1,4))/2. + (P(-1,2)*P(-1,4))/2. + P(-1,3)*P(-1,4)')

SSSS12 = Lorentz(name = 'SSSS12',
                 spins = [ 1, 1, 1, 1 ],
                 structure = 'P(-1,1)*P(-1,2) + P(-1,1)*P(-1,3) + P(-1,2)*P(-1,3) + P(-1,1)*P(-1,4) + P(-1,2)*P(-1,4) + P(-1,3)*P(-1,4)')

FFSS23 = Lorentz(name = 'FFSS23',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'Gamma5(2,1)')

FFSS24 = Lorentz(name = 'FFSS24',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,1) + (P(-1,3)*Gamma(-1,2,1))/2. + (P(-1,4)*Gamma(-1,2,1))/2.')

FFSS25 = Lorentz(name = 'FFSS25',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-2) - P(-1,4)*Gamma5(-2,1)*Gamma(-1,2,-2)')

FFSS26 = Lorentz(name = 'FFSS26',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'Identity(2,1)')

FFSS27 = Lorentz(name = 'FFSS27',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'ProjM(2,1)')

FFSS28 = Lorentz(name = 'FFSS28',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS29 = Lorentz(name = 'FFSS29',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + (3*P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFSS30 = Lorentz(name = 'FFSS30',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS31 = Lorentz(name = 'FFSS31',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - (P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFSS32 = Lorentz(name = 'FFSS32',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - (P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1))/5.')

FFSS33 = Lorentz(name = 'FFSS33',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2. + P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS34 = Lorentz(name = 'FFSS34',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + (3*P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1))/2.')

FFSS35 = Lorentz(name = 'FFSS35',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'ProjM(2,1) - ProjP(2,1)')

FFSS36 = Lorentz(name = 'FFSS36',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'ProjP(2,1)')

FFSS37 = Lorentz(name = 'FFSS37',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'ProjM(2,1) + ProjP(2,1)')

FFSS38 = Lorentz(name = 'FFSS38',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS39 = Lorentz(name = 'FFSS39',
                 spins = [ 2, 2, 1, 1 ],
                 structure = '-(P(-1,1)*Gamma(-1,2,1)) - P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS40 = Lorentz(name = 'FFSS40',
                 spins = [ 2, 2, 1, 1 ],
                 structure = '-(P(-1,1)*Gamma(-1,2,1)) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS41 = Lorentz(name = 'FFSS41',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1) - 5*P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS42 = Lorentz(name = 'FFSS42',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFSS43 = Lorentz(name = 'FFSS43',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + (P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1))/2. + P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) + (P(-1,3)*Gamma(-1,2,-2)*ProjP(-2,1))/2. + (P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1))/2.')

FFSS44 = Lorentz(name = 'FFSS44',
                 spins = [ 2, 2, 1, 1 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1) + P(-1,4)*Gamma(-1,2,-2)*ProjP(-2,1)')

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

FFVS84 = Lorentz(name = 'FFVS84',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*Gamma5(2,1) + (2*P(3,3)*Gamma5(2,1))/3.')

FFVS85 = Lorentz(name = 'FFVS85',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,3)*Gamma5(2,1) + 2*P(3,4)*Gamma5(2,1)')

FFVS86 = Lorentz(name = 'FFVS86',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'Gamma(3,2,1)')

FFVS87 = Lorentz(name = 'FFVS87',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,3)*Gamma5(2,1) - P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)')

FFVS88 = Lorentz(name = 'FFVS88',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)) + P(-1,3)*Gamma5(-2,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)')

FFVS89 = Lorentz(name = 'FFVS89',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,3)*Identity(2,1)')

FFVS90 = Lorentz(name = 'FFVS90',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,3)*Identity(2,1) + 2*P(3,4)*Identity(2,1)')

FFVS91 = Lorentz(name = 'FFVS91',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,3)*ProjM(2,1)')

FFVS92 = Lorentz(name = 'FFVS92',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,4)*ProjM(2,1)')

FFVS93 = Lorentz(name = 'FFVS93',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,3)*ProjM(2,1) + 2*P(3,4)*ProjM(2,1)')

FFVS94 = Lorentz(name = 'FFVS94',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVS95 = Lorentz(name = 'FFVS95',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(3,3)*ProjM(2,1)')

FFVS96 = Lorentz(name = 'FFVS96',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(-3*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/10. + P(3,3)*ProjM(2,1) + (7*P(3,4)*ProjM(2,1))/5.')

FFVS97 = Lorentz(name = 'FFVS97',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/6. + P(3,1)*ProjM(2,1) + (2*P(3,3)*ProjM(2,1))/3.')

FFVS98 = Lorentz(name = 'FFVS98',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/6. + P(3,1)*ProjM(2,1) + (2*P(3,3)*ProjM(2,1))/3. + P(3,4)*ProjM(2,1)')

FFVS99 = Lorentz(name = 'FFVS99',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/6. + P(3,3)*ProjM(2,1) + (7*P(3,4)*ProjM(2,1))/3.')

FFVS100 = Lorentz(name = 'FFVS100',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(3,1)*ProjM(2,1)')

FFVS101 = Lorentz(name = 'FFVS101',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(3,1)*ProjM(2,1) + (4*P(3,3)*ProjM(2,1))/11.')

FFVS102 = Lorentz(name = 'FFVS102',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(17*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/7. + (88*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/7. + P(3,1)*ProjM(2,1) - (9*P(3,2)*ProjM(2,1))/2. - (85*P(3,3)*ProjM(2,1))/7. - (9*P(3,4)*ProjM(2,1))/2.')

FFVS103 = Lorentz(name = 'FFVS103',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(9*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. + (65*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. + P(3,1)*ProjM(2,1) - 6*P(3,2)*ProjM(2,1) - (59*P(3,3)*ProjM(2,1))/3. - 6*P(3,4)*ProjM(2,1)')

FFVS104 = Lorentz(name = 'FFVS104',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-9*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/20. + (103*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/60. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/20. + P(3,1)*ProjM(2,1) - (3*P(3,2)*ProjM(2,1))/5. - (59*P(3,3)*ProjM(2,1))/30. + P(3,4)*ProjM(2,1)')

FFVS105 = Lorentz(name = 'FFVS105',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-9*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/20. + (151*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/60. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/20. + P(3,1)*ProjM(2,1) - (3*P(3,2)*ProjM(2,1))/5. - (83*P(3,3)*ProjM(2,1))/30. + P(3,4)*ProjM(2,1)')

FFVS106 = Lorentz(name = 'FFVS106',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-17*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/41. + (71*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/41. - (17*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/41. + P(3,1)*ProjM(2,1) - (63*P(3,2)*ProjM(2,1))/82. - (163*P(3,3)*ProjM(2,1))/82. + P(3,4)*ProjM(2,1)')

FFVS107 = Lorentz(name = 'FFVS107',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. - (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/13. + P(3,1)*ProjM(2,1) + (5*P(3,3)*ProjM(2,1))/13. + P(3,4)*ProjM(2,1)')

FFVS108 = Lorentz(name = 'FFVS108',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS109 = Lorentz(name = 'FFVS109',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,3)*ProjP(2,1)')

FFVS110 = Lorentz(name = 'FFVS110',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,4)*ProjP(2,1)')

FFVS111 = Lorentz(name = 'FFVS111',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*ProjM(2,1) - P(3,1)*ProjP(2,1)')

FFVS112 = Lorentz(name = 'FFVS112',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*ProjM(2,1) + P(3,1)*ProjP(2,1)')

FFVS113 = Lorentz(name = 'FFVS113',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,3)*Identity(2,1) + (3*P(3,1)*ProjM(2,1))/2. + (3*P(3,1)*ProjP(2,1))/2.')

FFVS114 = Lorentz(name = 'FFVS114',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*ProjM(2,1) + (2*P(3,3)*ProjM(2,1))/3. - P(3,1)*ProjP(2,1) - (2*P(3,3)*ProjP(2,1))/3.')

FFVS115 = Lorentz(name = 'FFVS115',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*Identity(2,1) + (2*P(3,3)*ProjM(2,1))/3. + (2*P(3,3)*ProjP(2,1))/3.')

FFVS116 = Lorentz(name = 'FFVS116',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*ProjM(2,1) + (2*P(3,3)*ProjM(2,1))/3. + P(3,1)*ProjP(2,1) + (2*P(3,3)*ProjP(2,1))/3.')

FFVS117 = Lorentz(name = 'FFVS117',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1)) + P(3,3)*ProjM(2,1) + P(3,3)*ProjP(2,1)')

FFVS118 = Lorentz(name = 'FFVS118',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,3)*Gamma5(2,1) - 2*P(3,4)*ProjM(2,1) + 2*P(3,4)*ProjP(2,1)')

FFVS119 = Lorentz(name = 'FFVS119',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,3)*ProjP(2,1) + 2*P(3,4)*ProjP(2,1)')

FFVS120 = Lorentz(name = 'FFVS120',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVS121 = Lorentz(name = 'FFVS121',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - (11*Gamma(3,2,-1)*ProjP(-1,1))/3.')

FFVS122 = Lorentz(name = 'FFVS122',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFVS123 = Lorentz(name = 'FFVS123',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - (11*Gamma(3,2,-1)*ProjP(-1,1))/6.')

FFVS124 = Lorentz(name = 'FFVS124',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - Gamma(3,2,-1)*ProjP(-1,1)')

FFVS125 = Lorentz(name = 'FFVS125',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFVS126 = Lorentz(name = 'FFVS126',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,3)*Gamma5(2,1) + (5*P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2))/4. + (9*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/4. - (9*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/4.')

FFVS127 = Lorentz(name = 'FFVS127',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(-1,3)*Gamma(-1,-2,1)*Gamma(3,2,-2) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFVS128 = Lorentz(name = 'FFVS128',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(3,3)*ProjP(2,1)')

FFVS129 = Lorentz(name = 'FFVS129',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(3,3)*ProjM(2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(3,3)*ProjP(2,1)')

FFVS130 = Lorentz(name = 'FFVS130',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/2. + P(3,3)*Identity(2,1) - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2.')

FFVS131 = Lorentz(name = 'FFVS131',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. + P(3,4)*ProjM(2,1) - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. + P(3,4)*ProjP(2,1)')

FFVS132 = Lorentz(name = 'FFVS132',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-3*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/10. + P(3,3)*ProjM(2,1) + (7*P(3,4)*ProjM(2,1))/5. - (3*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/10. - (2*P(3,3)*ProjP(2,1))/5. - (7*P(3,4)*ProjP(2,1))/5.')

FFVS133 = Lorentz(name = 'FFVS133',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,3)*Identity(2,1) - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/4. + (3*P(3,1)*ProjM(2,1))/2. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/4. + (3*P(3,1)*ProjP(2,1))/2. + (3*P(3,4)*ProjP(2,1))/2.')

FFVS134 = Lorentz(name = 'FFVS134',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/6. + P(3,1)*ProjP(2,1) + (2*P(3,3)*ProjP(2,1))/3.')

FFVS135 = Lorentz(name = 'FFVS135',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*Gamma5(2,1) + (2*P(3,3)*Gamma5(2,1))/3. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/6. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/6. + P(3,4)*ProjP(2,1)')

FFVS136 = Lorentz(name = 'FFVS136',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/6. + P(3,1)*ProjP(2,1) + (2*P(3,3)*ProjP(2,1))/3. + P(3,4)*ProjP(2,1)')

FFVS137 = Lorentz(name = 'FFVS137',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/6. + P(3,1)*ProjM(2,1) + (2*P(3,3)*ProjM(2,1))/3. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/6. + P(3,1)*ProjP(2,1) + (2*P(3,3)*ProjP(2,1))/3. + P(3,4)*ProjP(2,1)')

FFVS138 = Lorentz(name = 'FFVS138',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/8. + P(3,3)*ProjP(2,1) + (7*P(3,4)*ProjP(2,1))/4.')

FFVS139 = Lorentz(name = 'FFVS139',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/14. + P(3,4)*ProjP(2,1)')

FFVS140 = Lorentz(name = 'FFVS140',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/6. + P(3,1)*ProjM(2,1) + (2*P(3,3)*ProjM(2,1))/3. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/6. - P(3,1)*ProjP(2,1) - (2*P(3,3)*ProjP(2,1))/3. - P(3,4)*ProjP(2,1)')

FFVS141 = Lorentz(name = 'FFVS141',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + P(3,1)*ProjP(2,1) + (4*P(3,3)*ProjP(2,1))/11.')

FFVS142 = Lorentz(name = 'FFVS142',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-3*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/10. + P(3,3)*ProjM(2,1) + (7*P(3,4)*ProjM(2,1))/5. + (3*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/10. + (2*P(3,3)*ProjP(2,1))/5. + (7*P(3,4)*ProjP(2,1))/5.')

FFVS143 = Lorentz(name = 'FFVS143',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. + P(3,4)*ProjM(2,1) + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. - P(3,4)*ProjP(2,1)')

FFVS144 = Lorentz(name = 'FFVS144',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(3*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/4. + P(3,3)*ProjP(2,1) + (7*P(3,4)*ProjP(2,1))/2.')

FFVS145 = Lorentz(name = 'FFVS145',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1) + P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)')

FFVS146 = Lorentz(name = 'FFVS146',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(3,3)*ProjM(2,1) + P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) - P(3,3)*ProjP(2,1)')

FFVS147 = Lorentz(name = 'FFVS147',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/2. + P(3,3)*Identity(2,1) + (13*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/10. - (9*P(3,3)*ProjM(2,1))/5. + (13*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/10. - (9*P(3,3)*ProjP(2,1))/5.')

FFVS148 = Lorentz(name = 'FFVS148',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,3)*Gamma5(2,1) - P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2) - (9*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/5. + (9*P(3,3)*ProjM(2,1))/5. + (9*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/5. - (9*P(3,3)*ProjP(2,1))/5.')

FFVS149 = Lorentz(name = 'FFVS149',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(17*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/7. + (88*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/7. + P(3,1)*ProjP(2,1) - (9*P(3,2)*ProjP(2,1))/2. - (85*P(3,3)*ProjP(2,1))/7. - (9*P(3,4)*ProjP(2,1))/2.')

FFVS150 = Lorentz(name = 'FFVS150',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(9*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2. + (65*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. + P(3,1)*ProjP(2,1) - 6*P(3,2)*ProjP(2,1) - (59*P(3,3)*ProjP(2,1))/3. - 6*P(3,4)*ProjP(2,1)')

FFVS151 = Lorentz(name = 'FFVS151',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(9*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/2. + (89*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. + P(3,1)*ProjP(2,1) - 6*P(3,2)*ProjP(2,1) - (83*P(3,3)*ProjP(2,1))/3. - 6*P(3,4)*ProjP(2,1)')

FFVS152 = Lorentz(name = 'FFVS152',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-9*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/20. + (103*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/60. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/20. + P(3,1)*ProjP(2,1) - (3*P(3,2)*ProjP(2,1))/5. - (59*P(3,3)*ProjP(2,1))/30. + P(3,4)*ProjP(2,1)')

FFVS153 = Lorentz(name = 'FFVS153',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-17*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/41. + (71*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/41. - (17*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/41. + P(3,1)*ProjP(2,1) - (63*P(3,2)*ProjP(2,1))/82. - (163*P(3,3)*ProjP(2,1))/82. + P(3,4)*ProjP(2,1)')

FFVS154 = Lorentz(name = 'FFVS154',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(-8*P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/33. + P(3,1)*Identity(2,1) - (12*P(3,2)*Identity(2,1))/11. + (16*P(3,3)*Identity(2,1))/33. + (83*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (134*P(3,3)*ProjM(2,1))/33. + (4*P(3,4)*ProjM(2,1))/11. + (83*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (134*P(3,3)*ProjP(2,1))/33. + (4*P(3,4)*ProjP(2,1))/11.')

FFVS155 = Lorentz(name = 'FFVS155',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*Gamma5(2,1) - P(3,2)*Gamma5(2,1) - (59*P(3,3)*Gamma5(2,1))/18. - (4*P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2))/9. - (265*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/72. + (3*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/8. + (P(3,1)*ProjM(2,1))/12. - (P(3,4)*ProjM(2,1))/3. + (265*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/72. - (3*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/8. - (P(3,1)*ProjP(2,1))/12. + (P(3,4)*ProjP(2,1))/3.')

FFVS156 = Lorentz(name = 'FFVS156',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*Gamma5(2,1) - P(3,2)*Gamma5(2,1) - (8*P(3,3)*Gamma5(2,1))/9. + (8*P(-1,3)*Gamma5(-2,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2))/9. - (265*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/72. + (3*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/8. + (P(3,1)*ProjM(2,1))/12. + (67*P(3,3)*ProjM(2,1))/18. - (P(3,4)*ProjM(2,1))/3. + (265*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/72. - (3*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/8. - (P(3,1)*ProjP(2,1))/12. - (67*P(3,3)*ProjP(2,1))/18. + (P(3,4)*ProjP(2,1))/3.')

FFVS157 = Lorentz(name = 'FFVS157',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(4*P(-1,3)*Gamma(-1,2,-2)*Gamma(3,-2,1))/9. + P(3,1)*Identity(2,1) - P(3,2)*Identity(2,1) - (8*P(3,3)*Identity(2,1))/9. + (33*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/8. - (3*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/8. - (P(3,1)*ProjM(2,1))/12. - (67*P(3,3)*ProjM(2,1))/18. + (P(3,4)*ProjM(2,1))/3. + (33*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/8. - (3*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/8. - (P(3,1)*ProjP(2,1))/12. - (67*P(3,3)*ProjP(2,1))/18. + (P(3,4)*ProjP(2,1))/3.')

FFVS158 = Lorentz(name = 'FFVS158',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*Identity(2,1) - P(3,2)*Identity(2,1) + (53*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/21. - (17*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/63. - (5*P(3,1)*ProjM(2,1))/21. - (37*P(3,3)*ProjM(2,1))/14. + (19*P(3,4)*ProjM(2,1))/126. + (53*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/21. - (17*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/63. - (5*P(3,1)*ProjP(2,1))/21. - (37*P(3,3)*ProjP(2,1))/14. + (19*P(3,4)*ProjP(2,1))/126.')

FFVS159 = Lorentz(name = 'FFVS159',
                  spins = [ 2, 2, 3, 1 ],
                  structure = 'P(3,1)*Gamma5(2,1) - P(3,2)*Gamma5(2,1) - (53*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/21. + (17*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/63. + (5*P(3,1)*ProjM(2,1))/21. + (37*P(3,3)*ProjM(2,1))/14. - (19*P(3,4)*ProjM(2,1))/126. + (53*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/21. - (17*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/63. - (5*P(3,1)*ProjP(2,1))/21. - (37*P(3,3)*ProjP(2,1))/14. + (19*P(3,4)*ProjP(2,1))/126.')

FFVS160 = Lorentz(name = 'FFVS160',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. - (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/9. + P(3,3)*ProjM(2,1) + (13*P(3,4)*ProjM(2,1))/9. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. - (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/9. + P(3,3)*ProjP(2,1) + (13*P(3,4)*ProjP(2,1))/9.')

FFVS161 = Lorentz(name = 'FFVS161',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(3,1)*ProjM(2,1) + (4*P(3,3)*ProjM(2,1))/11. - (P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. - (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (13*P(3,1)*ProjP(2,1))/11. + (5*P(3,3)*ProjP(2,1))/11. + (13*P(3,4)*ProjP(2,1))/11.')

FFVS162 = Lorentz(name = 'FFVS162',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. - (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/13. + P(3,1)*ProjP(2,1) + (5*P(3,3)*ProjP(2,1))/13. + P(3,4)*ProjP(2,1)')

FFVS163 = Lorentz(name = 'FFVS163',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (2*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + P(3,1)*ProjM(2,1) + (4*P(3,3)*ProjM(2,1))/11. + (P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. - (13*P(3,1)*ProjP(2,1))/11. - (5*P(3,3)*ProjP(2,1))/11. - (13*P(3,4)*ProjP(2,1))/11.')

FFVS164 = Lorentz(name = 'FFVS164',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/3. - (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/9. + P(3,3)*ProjM(2,1) + (13*P(3,4)*ProjM(2,1))/9. - (P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/3. + (P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/9. - P(3,3)*ProjP(2,1) - (13*P(3,4)*ProjP(2,1))/9.')

FFVS165 = Lorentz(name = 'FFVS165',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS166 = Lorentz(name = 'FFVS166',
                  spins = [ 2, 2, 3, 1 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVV101 = Lorentz(name = 'FFVV101',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,1)*Gamma(3,2,1)')

FFVV102 = Lorentz(name = 'FFVV102',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,2)*Gamma(3,2,1)')

FFVV103 = Lorentz(name = 'FFVV103',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-1)*Gamma(4,-1,1)')

FFVV104 = Lorentz(name = 'FFVV104',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Identity(2,1)*Metric(3,4)')

FFVV105 = Lorentz(name = 'FFVV105',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) - Identity(2,1)*Metric(3,4)')

FFVV106 = Lorentz(name = 'FFVV106',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) + 2*Identity(2,1)*Metric(3,4)')

FFVV107 = Lorentz(name = 'FFVV107',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Metric(3,4)*ProjM(2,1)')

FFVV108 = Lorentz(name = 'FFVV108',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,1)*Gamma(3,2,-1)*ProjM(-1,1)')

FFVV109 = Lorentz(name = 'FFVV109',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,2)*Gamma(3,2,-1)*ProjM(-1,1)')

FFVV110 = Lorentz(name = 'FFVV110',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,3)*Gamma(3,2,-1)*ProjM(-1,1)')

FFVV111 = Lorentz(name = 'FFVV111',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,4)*Gamma(3,2,-1)*ProjM(-1,1)')

FFVV112 = Lorentz(name = 'FFVV112',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,1)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV113 = Lorentz(name = 'FFVV113',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,2)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV114 = Lorentz(name = 'FFVV114',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,3)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV115 = Lorentz(name = 'FFVV115',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV116 = Lorentz(name = 'FFVV116',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV117 = Lorentz(name = 'FFVV117',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVV118 = Lorentz(name = 'FFVV118',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - (8*Metric(3,4)*ProjM(2,1))/3.')

FFVV119 = Lorentz(name = 'FFVV119',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 2*Metric(3,4)*ProjM(2,1)')

FFVV120 = Lorentz(name = 'FFVV120',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - Metric(3,4)*ProjM(2,1)')

FFVV121 = Lorentz(name = 'FFVV121',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (14*Metric(3,4)*ProjM(2,1))/9.')

FFVV122 = Lorentz(name = 'FFVV122',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + 2*Metric(3,4)*ProjM(2,1)')

FFVV123 = Lorentz(name = 'FFVV123',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (26*Metric(3,4)*ProjM(2,1))/5.')

FFVV124 = Lorentz(name = 'FFVV124',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - (8*Metric(3,4)*ProjM(2,1))/3.')

FFVV125 = Lorentz(name = 'FFVV125',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - 2*Metric(3,4)*ProjM(2,1)')

FFVV126 = Lorentz(name = 'FFVV126',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Metric(3,4)*ProjM(2,1)')

FFVV127 = Lorentz(name = 'FFVV127',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + 2*Metric(3,4)*ProjM(2,1)')

FFVV128 = Lorentz(name = 'FFVV128',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV129 = Lorentz(name = 'FFVV129',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV130 = Lorentz(name = 'FFVV130',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV131 = Lorentz(name = 'FFVV131',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1)')

FFVV132 = Lorentz(name = 'FFVV132',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1)')

FFVV133 = Lorentz(name = 'FFVV133',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) + P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) - P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) + P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV134 = Lorentz(name = 'FFVV134',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,4)*Gamma(3,2,-1)*ProjM(-1,1) - P(3,3)*Gamma(4,2,-1)*ProjM(-1,1) + P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV135 = Lorentz(name = 'FFVV135',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '(8*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. - (8*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) + P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (7*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/9. + (P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/9. - P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - (P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/9. + (7*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/9.')

FFVV136 = Lorentz(name = 'FFVV136',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '(7*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. - (7*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) + P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (5*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/9. + (2*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/9. - P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - (2*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/9. + (5*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/9.')

FFVV137 = Lorentz(name = 'FFVV137',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/2. + P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) + (P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/2. - (P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/2. - P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV138 = Lorentz(name = 'FFVV138',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/13. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/13. - (7*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/13. + (7*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/13. + P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) + (5*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/13. - (7*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/13. - (11*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/13.')

FFVV139 = Lorentz(name = 'FFVV139',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/11. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/11. - (7*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/11. + (7*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/11. + P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) + (7*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/11. - (5*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/11. - (13*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/11.')

FFVV140 = Lorentz(name = 'FFVV140',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,4)*Gamma(3,2,-1)*ProjM(-1,1) - P(3,3)*Gamma(4,2,-1)*ProjM(-1,1) + P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV141 = Lorentz(name = 'FFVV141',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1)')

FFVV142 = Lorentz(name = 'FFVV142',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1)')

FFVV143 = Lorentz(name = 'FFVV143',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1)')

FFVV144 = Lorentz(name = 'FFVV144',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) + (complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. + (complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/6. - (complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/6. - (5*complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/6. - (complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/6. + (complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/2. + (complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/2.')

FFVV145 = Lorentz(name = 'FFVV145',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) - (complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. + (complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. + (complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/3. - (complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/3. - (complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/3. + (complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/3. + (complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/3. - (complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/3.')

FFVV146 = Lorentz(name = 'FFVV146',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) + (complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/2. + (complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/2. + (complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/2. + (complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/2. - (complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/6. - (5*complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/6.')

FFVV147 = Lorentz(name = 'FFVV147',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) + (5*complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/3. - (5*complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/3. + (2*complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/3. - (2*complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/3. + 3*complex(0,1)*P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (10*complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/3. + (7*complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/3. - 3*complex(0,1)*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) + complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV148 = Lorentz(name = 'FFVV148',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) + complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) - complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) - complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) - complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1) - complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1) + complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV149 = Lorentz(name = 'FFVV149',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) + (complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/9. - (complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/9. + (20*complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. - (20*complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. + 3*complex(0,1)*P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (14*complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/9. + (5*complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/9. - 3*complex(0,1)*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - (7*complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/9. + (16*complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/9.')

FFVV150 = Lorentz(name = 'FFVV150',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) - (complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/9. + (complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/9. - (20*complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. + (20*complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/9. - 3*complex(0,1)*P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) + (16*complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1))/9. - (7*complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1))/9. + 3*complex(0,1)*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) + (5*complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/9. - (14*complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/9.')

FFVV151 = Lorentz(name = 'FFVV151',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) + 3*complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) - 3*complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + 3*complex(0,1)*P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV152 = Lorentz(name = 'FFVV152',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) - 3*complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + 3*complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) - 3*complex(0,1)*P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV153 = Lorentz(name = 'FFVV153',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) + (5*complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/3. - (5*complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/3. - 4*complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + 4*complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) - 3*complex(0,1)*P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - 3*complex(0,1)*P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) + complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) + 3*complex(0,1)*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) + (7*complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjM(-1,1))/3. - (10*complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjM(-1,1))/3.')

FFVV154 = Lorentz(name = 'FFVV154',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Metric(3,4)*ProjP(2,1)')

FFVV155 = Lorentz(name = 'FFVV155',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma5(-1,1)*Gamma(3,-2,-1)*Gamma(4,2,-2) - (14*Metric(3,4)*ProjM(2,1))/9. - (32*Metric(3,4)*ProjP(2,1))/9.')

FFVV156 = Lorentz(name = 'FFVV156',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) - Metric(3,4)*ProjM(2,1) - Metric(3,4)*ProjP(2,1)')

FFVV157 = Lorentz(name = 'FFVV157',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Metric(3,4)*ProjM(2,1) + Metric(3,4)*ProjP(2,1)')

FFVV158 = Lorentz(name = 'FFVV158',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,1)*Gamma(3,2,-1)*ProjP(-1,1)')

FFVV159 = Lorentz(name = 'FFVV159',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,2)*Gamma(3,2,-1)*ProjP(-1,1)')

FFVV160 = Lorentz(name = 'FFVV160',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,3)*Gamma(3,2,-1)*ProjP(-1,1)')

FFVV161 = Lorentz(name = 'FFVV161',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,4)*Gamma(3,2,-1)*ProjP(-1,1)')

FFVV162 = Lorentz(name = 'FFVV162',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,1)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV163 = Lorentz(name = 'FFVV163',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV164 = Lorentz(name = 'FFVV164',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,3)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV165 = Lorentz(name = 'FFVV165',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(3,4)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV166 = Lorentz(name = 'FFVV166',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV167 = Lorentz(name = 'FFVV167',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

FFVV168 = Lorentz(name = 'FFVV168',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-1)*Gamma(4,-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV169 = Lorentz(name = 'FFVV169',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (26*Metric(3,4)*ProjM(2,1))/5. - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + (36*Metric(3,4)*ProjP(2,1))/5.')

FFVV170 = Lorentz(name = 'FFVV170',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 10*Metric(3,4)*ProjM(2,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - 10*Metric(3,4)*ProjP(2,1)')

FFVV171 = Lorentz(name = 'FFVV171',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - (36*Metric(3,4)*ProjP(2,1))/5.')

FFVV172 = Lorentz(name = 'FFVV172',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 4*Metric(3,4)*ProjM(2,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - 4*Metric(3,4)*ProjP(2,1)')

FFVV173 = Lorentz(name = 'FFVV173',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - (32*Metric(3,4)*ProjP(2,1))/9.')

FFVV174 = Lorentz(name = 'FFVV174',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - (8*Metric(3,4)*ProjP(2,1))/3.')

FFVV175 = Lorentz(name = 'FFVV175',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - 2*Metric(3,4)*ProjP(2,1)')

FFVV176 = Lorentz(name = 'FFVV176',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - Metric(3,4)*ProjP(2,1)')

FFVV177 = Lorentz(name = 'FFVV177',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + 2*Metric(3,4)*ProjP(2,1)')

FFVV178 = Lorentz(name = 'FFVV178',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + 2*Metric(3,4)*ProjM(2,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + 2*Metric(3,4)*ProjP(2,1)')

FFVV179 = Lorentz(name = 'FFVV179',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-1)*Gamma(4,-1,1) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

FFVV180 = Lorentz(name = 'FFVV180',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-1)*Gamma(4,-1,1) + 2*Identity(2,1)*Metric(3,4) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - 2*Metric(3,4)*ProjM(2,1) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - 2*Metric(3,4)*ProjP(2,1)')

FFVV181 = Lorentz(name = 'FFVV181',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - (8*Metric(3,4)*ProjP(2,1))/3.')

FFVV182 = Lorentz(name = 'FFVV182',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - 2*Metric(3,4)*ProjP(2,1)')

FFVV183 = Lorentz(name = 'FFVV183',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Metric(3,4)*ProjP(2,1)')

FFVV184 = Lorentz(name = 'FFVV184',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + 2*Metric(3,4)*ProjP(2,1)')

FFVV185 = Lorentz(name = 'FFVV185',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV186 = Lorentz(name = 'FFVV186',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV187 = Lorentz(name = 'FFVV187',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV188 = Lorentz(name = 'FFVV188',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV189 = Lorentz(name = 'FFVV189',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1)')

FFVV190 = Lorentz(name = 'FFVV190',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1)')

FFVV191 = Lorentz(name = 'FFVV191',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1)) + P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) - P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + P(4,3)*Gamma(3,2,-1)*ProjP(-1,1) - P(4,4)*Gamma(3,2,-1)*ProjP(-1,1) - P(3,3)*Gamma(4,2,-1)*ProjP(-1,1) + P(3,4)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV192 = Lorentz(name = 'FFVV192',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + P(4,3)*Gamma(3,2,-1)*ProjP(-1,1) - P(4,4)*Gamma(3,2,-1)*ProjP(-1,1) - P(3,3)*Gamma(4,2,-1)*ProjP(-1,1) + P(3,4)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV193 = Lorentz(name = 'FFVV193',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjP(-3,1)')

FFVV194 = Lorentz(name = 'FFVV194',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjP(-3,1)')

FFVV195 = Lorentz(name = 'FFVV195',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjP(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjP(-3,1)')

FFVV196 = Lorentz(name = 'FFVV196',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjP(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjP(-3,1) + (complex(0,1)*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. - (complex(0,1)*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. - (complex(0,1)*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/3. + (complex(0,1)*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/3. + (complex(0,1)*P(4,3)*Gamma(3,2,-1)*ProjP(-1,1))/3. - (complex(0,1)*P(4,4)*Gamma(3,2,-1)*ProjP(-1,1))/3. - (complex(0,1)*P(3,3)*Gamma(4,2,-1)*ProjP(-1,1))/3. + (complex(0,1)*P(3,4)*Gamma(4,2,-1)*ProjP(-1,1))/3.')

FFVV197 = Lorentz(name = 'FFVV197',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjM(-3,1) - Epsilon(3,4,-1,-2)*P(-1,3)*Gamma(-2,2,-3)*ProjP(-3,1) + Epsilon(3,4,-1,-2)*P(-1,4)*Gamma(-2,2,-3)*ProjP(-3,1)')

VSSS9 = Lorentz(name = 'VSSS9',
                spins = [ 3, 1, 1, 1 ],
                structure = 'P(1,2)')

VSSS10 = Lorentz(name = 'VSSS10',
                 spins = [ 3, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,3)')

VSSS11 = Lorentz(name = 'VSSS11',
                 spins = [ 3, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3)')

VSSS12 = Lorentz(name = 'VSSS12',
                 spins = [ 3, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) - 2*P(1,4)')

VSSS13 = Lorentz(name = 'VSSS13',
                 spins = [ 3, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,4)/2.')

VSSS14 = Lorentz(name = 'VSSS14',
                 spins = [ 3, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,3)/2. - P(1,4)/2.')

VSSS15 = Lorentz(name = 'VSSS15',
                 spins = [ 3, 1, 1, 1 ],
                 structure = 'P(1,2) - P(1,3)/3. - P(1,4)/3.')

VSSS16 = Lorentz(name = 'VSSS16',
                 spins = [ 3, 1, 1, 1 ],
                 structure = 'P(1,2) + P(1,3) + P(1,4)')

VVSS11 = Lorentz(name = 'VVSS11',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,1)')

VVSS12 = Lorentz(name = 'VVSS12',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1) - Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,1) + 2*Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2) - 2*Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2)')

VVSS13 = Lorentz(name = 'VVSS13',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2) - Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2) - 2*Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,4)')

VVSS14 = Lorentz(name = 'VVSS14',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'Metric(1,2)')

VVSS15 = Lorentz(name = 'VVSS15',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,1) - P(-1,1)*P(-1,2)*Metric(1,2)')

VVSS16 = Lorentz(name = 'VVSS16',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,3)*P(2,1) + P(1,4)*P(2,1) + P(1,2)*P(2,3) + P(1,2)*P(2,4) + 2*P(-1,1)*P(-1,2)*Metric(1,2) - 2*P(-1,2)**2*Metric(1,2) - P(-1,1)*P(-1,3)*Metric(1,2) - 3*P(-1,2)*P(-1,3)*Metric(1,2) - P(-1,1)*P(-1,4)*Metric(1,2) - 3*P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS17 = Lorentz(name = 'VVSS17',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,1) - Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,1) + 2*Epsilon(1,2,-1,-2)*P(-2,3)*P(-1,2) - 2*Epsilon(1,2,-1,-2)*P(-2,4)*P(-1,2) + complex(0,1)*P(1,3)*P(2,1) + complex(0,1)*P(1,4)*P(2,1) + complex(0,1)*P(1,2)*P(2,3) + complex(0,1)*P(1,2)*P(2,4) + 2*complex(0,1)*P(-1,1)*P(-1,2)*Metric(1,2) - 2*complex(0,1)*P(-1,2)**2*Metric(1,2) - complex(0,1)*P(-1,1)*P(-1,3)*Metric(1,2) - 3*complex(0,1)*P(-1,2)*P(-1,3)*Metric(1,2) - complex(0,1)*P(-1,1)*P(-1,4)*Metric(1,2) - 3*complex(0,1)*P(-1,2)*P(-1,4)*Metric(1,2)')

VVSS18 = Lorentz(name = 'VVSS18',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,1) - (P(-1,1)*P(-1,2)*Metric(1,2))/2. + (P(-1,2)**2*Metric(1,2))/2. + (P(-1,2)*P(-1,3)*Metric(1,2))/2. + (P(-1,2)*P(-1,4)*Metric(1,2))/2.')

VVSS19 = Lorentz(name = 'VVSS19',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,1) - (9*P(1,3)*P(2,1))/26. - (9*P(1,4)*P(2,1))/26. - (9*P(1,2)*P(2,3))/26. - (9*P(1,2)*P(2,4))/26. - (17*P(-1,1)*P(-1,2)*Metric(1,2))/26. + (9*P(-1,2)**2*Metric(1,2))/26. + (3*P(-1,1)*P(-1,3)*Metric(1,2))/26. + (6*P(-1,2)*P(-1,3)*Metric(1,2))/13. - (3*P(-1,3)**2*Metric(1,2))/13. + (3*P(-1,1)*P(-1,4)*Metric(1,2))/26. + (6*P(-1,2)*P(-1,4)*Metric(1,2))/13. - (6*P(-1,3)*P(-1,4)*Metric(1,2))/13. - (3*P(-1,4)**2*Metric(1,2))/13.')

VVSS20 = Lorentz(name = 'VVSS20',
                 spins = [ 3, 3, 1, 1 ],
                 structure = 'P(1,2)*P(2,1) - (13*P(1,3)*P(2,1))/62. - (13*P(1,4)*P(2,1))/62. - (13*P(1,2)*P(2,3))/62. + (P(1,3)*P(2,3))/62. + (P(1,4)*P(2,3))/62. - (13*P(1,2)*P(2,4))/62. + (P(1,3)*P(2,4))/62. + (P(1,4)*P(2,4))/62. - (20*P(-1,1)*P(-1,2)*Metric(1,2))/31. + (19*P(-1,2)**2*Metric(1,2))/62. + (2*P(-1,1)*P(-1,3)*Metric(1,2))/31. + (23*P(-1,2)*P(-1,3)*Metric(1,2))/62. - (13*P(-1,3)**2*Metric(1,2))/62. + (2*P(-1,1)*P(-1,4)*Metric(1,2))/31. + (23*P(-1,2)*P(-1,4)*Metric(1,2))/62. - (13*P(-1,3)*P(-1,4)*Metric(1,2))/31. - (13*P(-1,4)**2*Metric(1,2))/62.')

VVVS10 = Lorentz(name = 'VVVS10',
                 spins = [ 3, 3, 3, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2)')

VVVS11 = Lorentz(name = 'VVVS11',
                 spins = [ 3, 3, 3, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,4))')

VVVS12 = Lorentz(name = 'VVVS12',
                 spins = [ 3, 3, 3, 1 ],
                 structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) - Epsilon(1,2,3,-1)*P(-1,2) - Epsilon(1,2,3,-1)*P(-1,3) + 2*Epsilon(1,2,3,-1)*P(-1,4)')

VVVS13 = Lorentz(name = 'VVVS13',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVS14 = Lorentz(name = 'VVVS14',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) - P(1,2)*Metric(2,3)')

VVVS15 = Lorentz(name = 'VVVS15',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVS16 = Lorentz(name = 'VVVS16',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVS17 = Lorentz(name = 'VVVS17',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVS18 = Lorentz(name = 'VVVS18',
                 spins = [ 3, 3, 3, 1 ],
                 structure = 'P(3,4)*Metric(1,2) + P(2,4)*Metric(1,3) + P(1,4)*Metric(2,3)')

VVVV13 = Lorentz(name = 'VVVV13',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Epsilon(1,2,3,4)')

VVVV14 = Lorentz(name = 'VVVV14',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3)')

VVVV15 = Lorentz(name = 'VVVV15',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,3)*Metric(2,4)')

VVVV16 = Lorentz(name = 'VVVV16',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV17 = Lorentz(name = 'VVVV17',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,2)*Metric(3,4)')

VVVV18 = Lorentz(name = 'VVVV18',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV19 = Lorentz(name = 'VVVV19',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV20 = Lorentz(name = 'VVVV20',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV21 = Lorentz(name = 'VVVV21',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVV22 = Lorentz(name = 'VVVV22',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) + Metric(1,2)*Metric(3,4)')

VVVV23 = Lorentz(name = 'VVVV23',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,2)*P(4,1)*Metric(1,2) - P(3,1)*P(4,2)*Metric(1,2) + P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) + P(2,1)*P(4,2)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) + P(2,4)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) - P(2,1)*P(3,2)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(2,3)*P(3,4)*Metric(1,4) - P(1,2)*P(4,1)*Metric(2,3) - P(1,3)*P(4,1)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) - P(1,4)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,2)*P(3,1)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(1,3)*P(3,4)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) - P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,4)*P(2,3)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(1,3)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV24 = Lorentz(name = 'VVVV24',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,1)*Metric(1,2) + P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(3,2)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) - P(2,4)*P(4,1)*Metric(1,3) - P(2,3)*P(4,2)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) + P(2,4)*P(3,1)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(2,1)*P(3,4)*Metric(1,4) - P(1,3)*P(4,1)*Metric(2,3) + P(1,3)*P(4,2)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) - P(1,2)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) - P(1,4)*P(3,1)*Metric(2,4) - P(1,3)*P(3,2)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4) + P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,4)*P(2,1)*Metric(3,4) + P(1,2)*P(2,3)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

SSSSS2 = Lorentz(name = 'SSSSS2',
                 spins = [ 1, 1, 1, 1, 1 ],
                 structure = '1')

FFSSS5 = Lorentz(name = 'FFSSS5',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'Gamma5(2,1)')

FFSSS6 = Lorentz(name = 'FFSSS6',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'Identity(2,1)')

FFSSS7 = Lorentz(name = 'FFSSS7',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjM(2,1)')

FFSSS8 = Lorentz(name = 'FFSSS8',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'ProjP(2,1)')

FFVSS11 = Lorentz(name = 'FFVSS11',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,1)')

FFVSS12 = Lorentz(name = 'FFVSS12',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma5(-1,1)*Gamma(3,2,-1)')

FFVSS13 = Lorentz(name = 'FFVSS13',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVSS14 = Lorentz(name = 'FFVSS14',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVSS15 = Lorentz(name = 'FFVSS15',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 4*Gamma(3,2,-1)*ProjP(-1,1)')

FFVSS16 = Lorentz(name = 'FFVSS16',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFVSS17 = Lorentz(name = 'FFVSS17',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - (8*Gamma(3,2,-1)*ProjP(-1,1))/5.')

FFVSS18 = Lorentz(name = 'FFVSS18',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - Gamma(3,2,-1)*ProjP(-1,1)')

FFVSS19 = Lorentz(name = 'FFVSS19',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) - (6*Gamma(3,2,-1)*ProjP(-1,1))/7.')

FFVSS20 = Lorentz(name = 'FFVSS20',
                  spins = [ 2, 2, 3, 1, 1 ],
                  structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFVVS77 = Lorentz(name = 'FFVVS77',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,2,-1)*Gamma(4,-1,1)')

FFVVS78 = Lorentz(name = 'FFVVS78',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma5(-1,1)*Gamma(3,2,-2)*Gamma(4,-2,-1)')

FFVVS79 = Lorentz(name = 'FFVVS79',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma5(-1,1)*Gamma(3,2,-2)*Gamma(4,-2,-1) - Gamma5(-1,1)*Gamma(3,-2,-1)*Gamma(4,2,-2)')

FFVVS80 = Lorentz(name = 'FFVVS80',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma5(2,1)*Metric(3,4)')

FFVVS81 = Lorentz(name = 'FFVVS81',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Identity(2,1)*Metric(3,4)')

FFVVS82 = Lorentz(name = 'FFVVS82',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma5(-1,1)*Gamma(3,-2,-1)*Gamma(4,2,-2) - Gamma5(2,1)*Metric(3,4)')

FFVVS83 = Lorentz(name = 'FFVVS83',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma5(-1,1)*Gamma(3,-2,-1)*Gamma(4,2,-2) + 2*Gamma5(2,1)*Metric(3,4)')

FFVVS84 = Lorentz(name = 'FFVVS84',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) - (14*Identity(2,1)*Metric(3,4))/3.')

FFVVS85 = Lorentz(name = 'FFVVS85',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) - Identity(2,1)*Metric(3,4)')

FFVVS86 = Lorentz(name = 'FFVVS86',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) + 2*Identity(2,1)*Metric(3,4)')

FFVVS87 = Lorentz(name = 'FFVVS87',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Metric(3,4)*ProjM(2,1)')

FFVVS88 = Lorentz(name = 'FFVVS88',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS89 = Lorentz(name = 'FFVVS89',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVVS90 = Lorentz(name = 'FFVVS90',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - (36*Metric(3,4)*ProjM(2,1))/5.')

FFVVS91 = Lorentz(name = 'FFVVS91',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 4*Metric(3,4)*ProjM(2,1)')

FFVVS92 = Lorentz(name = 'FFVVS92',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - (32*Metric(3,4)*ProjM(2,1))/9.')

FFVVS93 = Lorentz(name = 'FFVVS93',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - (10*Metric(3,4)*ProjM(2,1))/3.')

FFVVS94 = Lorentz(name = 'FFVVS94',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 2*Metric(3,4)*ProjM(2,1)')

FFVVS95 = Lorentz(name = 'FFVVS95',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - Metric(3,4)*ProjM(2,1)')

FFVVS96 = Lorentz(name = 'FFVVS96',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (14*Metric(3,4)*ProjM(2,1))/9.')

FFVVS97 = Lorentz(name = 'FFVVS97',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + 2*Metric(3,4)*ProjM(2,1)')

FFVVS98 = Lorentz(name = 'FFVVS98',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (8*Metric(3,4)*ProjM(2,1))/3.')

FFVVS99 = Lorentz(name = 'FFVVS99',
                  spins = [ 2, 2, 3, 3, 1 ],
                  structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (26*Metric(3,4)*ProjM(2,1))/5.')

FFVVS100 = Lorentz(name = 'FFVVS100',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + 8*Metric(3,4)*ProjM(2,1)')

FFVVS101 = Lorentz(name = 'FFVVS101',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - (36*Metric(3,4)*ProjM(2,1))/5.')

FFVVS102 = Lorentz(name = 'FFVVS102',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - (32*Metric(3,4)*ProjM(2,1))/9.')

FFVVS103 = Lorentz(name = 'FFVVS103',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - (10*Metric(3,4)*ProjM(2,1))/3.')

FFVVS104 = Lorentz(name = 'FFVVS104',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - 2*Metric(3,4)*ProjM(2,1)')

FFVVS105 = Lorentz(name = 'FFVVS105',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Metric(3,4)*ProjM(2,1)')

FFVVS106 = Lorentz(name = 'FFVVS106',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + (14*Metric(3,4)*ProjM(2,1))/9.')

FFVVS107 = Lorentz(name = 'FFVVS107',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + 2*Metric(3,4)*ProjM(2,1)')

FFVVS108 = Lorentz(name = 'FFVVS108',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + (26*Metric(3,4)*ProjM(2,1))/5.')

FFVVS109 = Lorentz(name = 'FFVVS109',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVVS110 = Lorentz(name = 'FFVVS110',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Metric(3,4)*ProjP(2,1)')

FFVVS111 = Lorentz(name = 'FFVVS111',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Identity(2,1)*Metric(3,4) - (95*Metric(3,4)*ProjM(2,1))/2. - (95*Metric(3,4)*ProjP(2,1))/2.')

FFVVS112 = Lorentz(name = 'FFVVS112',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma5(2,1)*Metric(3,4) + (95*Metric(3,4)*ProjM(2,1))/2. - (95*Metric(3,4)*ProjP(2,1))/2.')

FFVVS113 = Lorentz(name = 'FFVVS113',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Identity(2,1)*Metric(3,4) - (235*Metric(3,4)*ProjM(2,1))/28. - (235*Metric(3,4)*ProjP(2,1))/28.')

FFVVS114 = Lorentz(name = 'FFVVS114',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma5(2,1)*Metric(3,4) + (235*Metric(3,4)*ProjM(2,1))/28. - (235*Metric(3,4)*ProjP(2,1))/28.')

FFVVS115 = Lorentz(name = 'FFVVS115',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma5(-1,1)*Gamma(3,-2,-1)*Gamma(4,2,-2) + (28*Identity(2,1)*Metric(3,4))/81. - (154*Metric(3,4)*ProjM(2,1))/81. - (316*Metric(3,4)*ProjP(2,1))/81.')

FFVVS116 = Lorentz(name = 'FFVVS116',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) + (28*Gamma5(2,1)*Metric(3,4))/81. + (154*Metric(3,4)*ProjM(2,1))/81. - (316*Metric(3,4)*ProjP(2,1))/81.')

FFVVS117 = Lorentz(name = 'FFVVS117',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-1,1)*Gamma(4,2,-1) - Metric(3,4)*ProjM(2,1) - Metric(3,4)*ProjP(2,1)')

FFVVS118 = Lorentz(name = 'FFVVS118',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS119 = Lorentz(name = 'FFVVS119',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

FFVVS120 = Lorentz(name = 'FFVVS120',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-1)*Gamma(4,-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS121 = Lorentz(name = 'FFVVS121',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + 2*Metric(3,4)*ProjM(2,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - 2*Metric(3,4)*ProjP(2,1)')

FFVVS122 = Lorentz(name = 'FFVVS122',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 10*Metric(3,4)*ProjM(2,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + 10*Metric(3,4)*ProjP(2,1)')

FFVVS123 = Lorentz(name = 'FFVVS123',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma5(-1,1)*Gamma(3,-2,-1)*Gamma(4,2,-2) - 2*Gamma5(2,1)*Metric(3,4) + (4*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1))/7. - (4*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1))/7.')

FFVVS124 = Lorentz(name = 'FFVVS124',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 10*Metric(3,4)*ProjM(2,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - 10*Metric(3,4)*ProjP(2,1)')

FFVVS125 = Lorentz(name = 'FFVVS125',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - (36*Metric(3,4)*ProjP(2,1))/5.')

FFVVS126 = Lorentz(name = 'FFVVS126',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - 4*Metric(3,4)*ProjP(2,1)')

FFVVS127 = Lorentz(name = 'FFVVS127',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - (32*Metric(3,4)*ProjP(2,1))/9.')

FFVVS128 = Lorentz(name = 'FFVVS128',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - (10*Metric(3,4)*ProjP(2,1))/3.')

FFVVS129 = Lorentz(name = 'FFVVS129',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - 2*Metric(3,4)*ProjP(2,1)')

FFVVS130 = Lorentz(name = 'FFVVS130',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) - Metric(3,4)*ProjP(2,1)')

FFVVS131 = Lorentz(name = 'FFVVS131',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + (14*Metric(3,4)*ProjP(2,1))/9.')

FFVVS132 = Lorentz(name = 'FFVVS132',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + 2*Metric(3,4)*ProjP(2,1)')

FFVVS133 = Lorentz(name = 'FFVVS133',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + 2*Metric(3,4)*ProjM(2,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + 2*Metric(3,4)*ProjP(2,1)')

FFVVS134 = Lorentz(name = 'FFVVS134',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + (8*Metric(3,4)*ProjP(2,1))/3.')

FFVVS135 = Lorentz(name = 'FFVVS135',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + (26*Metric(3,4)*ProjP(2,1))/5.')

FFVVS136 = Lorentz(name = 'FFVVS136',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + 8*Metric(3,4)*ProjP(2,1)')

FFVVS137 = Lorentz(name = 'FFVVS137',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Identity(2,1)*Metric(3,4) - (15*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1))/2. - 40*Metric(3,4)*ProjM(2,1) + (15*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1))/2. - 55*Metric(3,4)*ProjP(2,1)')

FFVVS138 = Lorentz(name = 'FFVVS138',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma5(2,1)*Metric(3,4) + (15*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1))/2. + 40*Metric(3,4)*ProjM(2,1) + (15*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1))/2. - 55*Metric(3,4)*ProjP(2,1)')

FFVVS139 = Lorentz(name = 'FFVVS139',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-1)*Gamma(4,-1,1) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

FFVVS140 = Lorentz(name = 'FFVVS140',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-1)*Gamma(4,-1,1) + 2*Identity(2,1)*Metric(3,4) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - 2*Metric(3,4)*ProjM(2,1) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - 2*Metric(3,4)*ProjP(2,1)')

FFVVS141 = Lorentz(name = 'FFVVS141',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS142 = Lorentz(name = 'FFVVS142',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - (36*Metric(3,4)*ProjP(2,1))/5.')

FFVVS143 = Lorentz(name = 'FFVVS143',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - (32*Metric(3,4)*ProjP(2,1))/9.')

FFVVS144 = Lorentz(name = 'FFVVS144',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - (10*Metric(3,4)*ProjP(2,1))/3.')

FFVVS145 = Lorentz(name = 'FFVVS145',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - 2*Metric(3,4)*ProjP(2,1)')

FFVVS146 = Lorentz(name = 'FFVVS146',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Metric(3,4)*ProjP(2,1)')

FFVVS147 = Lorentz(name = 'FFVVS147',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + (14*Metric(3,4)*ProjP(2,1))/9.')

FFVVS148 = Lorentz(name = 'FFVVS148',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + 2*Metric(3,4)*ProjP(2,1)')

FFVVS149 = Lorentz(name = 'FFVVS149',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + (26*Metric(3,4)*ProjP(2,1))/5.')

FFVVS150 = Lorentz(name = 'FFVVS150',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVS151 = Lorentz(name = 'FFVVS151',
                   spins = [ 2, 2, 3, 3, 1 ],
                   structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVVV79 = Lorentz(name = 'FFVVV79',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*Gamma(5,-1,1)')

FFVVV80 = Lorentz(name = 'FFVVV80',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(5,2,1)*Metric(3,4)')

FFVVV81 = Lorentz(name = 'FFVVV81',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(4,2,1)*Metric(3,5)')

FFVVV82 = Lorentz(name = 'FFVVV82',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(3,2,1)*Metric(4,5)')

FFVVV83 = Lorentz(name = 'FFVVV83',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(3,-3,-2)*Gamma(4,-2,-1)*Gamma(5,2,-3)*ProjM(-1,1)')

FFVVV84 = Lorentz(name = 'FFVVV84',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(3,2,-3)*Gamma(4,-3,-2)*Gamma(5,-2,-1)*ProjM(-1,1)')

FFVVV85 = Lorentz(name = 'FFVVV85',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1)')

FFVVV86 = Lorentz(name = 'FFVVV86',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1)')

FFVVV87 = Lorentz(name = 'FFVVV87',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV88 = Lorentz(name = 'FFVVV88',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '(Gamma(3,2,-3)*Gamma(4,-2,-1)*Gamma(5,-3,-2)*ProjM(-1,1))/8. + (5*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/4. - (5*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV89 = Lorentz(name = 'FFVVV89',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - (182*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/79. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV90 = Lorentz(name = 'FFVVV90',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - (25*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/11. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV91 = Lorentz(name = 'FFVVV91',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '-(Gamma(3,-3,-2)*Gamma(4,2,-3)*Gamma(5,-2,-1)*ProjM(-1,1))/8. + (5*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/4. - (9*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/4. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV92 = Lorentz(name = 'FFVVV92',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - (43*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/20. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV93 = Lorentz(name = 'FFVVV93',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - 2*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV94 = Lorentz(name = 'FFVVV94',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '(-3*Gamma(3,-2,-1)*Gamma(4,-3,-2)*Gamma(5,2,-3)*ProjM(-1,1))/2. + Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - 2*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV95 = Lorentz(name = 'FFVVV95',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '(9*Gamma(3,-3,-2)*Gamma(4,-2,-1)*Gamma(5,2,-3)*ProjM(-1,1))/8. - (5*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/4. - (5*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/4. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV96 = Lorentz(name = 'FFVVV96',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '-(Gamma(3,-2,-1)*Gamma(4,-3,-2)*Gamma(5,2,-3)*ProjM(-1,1)) + Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV97 = Lorentz(name = 'FFVVV97',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '(-9*Gamma(3,-2,-1)*Gamma(4,-3,-2)*Gamma(5,2,-3)*ProjM(-1,1))/10. + Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - (4*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/5. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV98 = Lorentz(name = 'FFVVV98',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '(-11*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/18. - (11*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/18. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV99 = Lorentz(name = 'FFVVV99',
                  spins = [ 2, 2, 3, 3, 3 ],
                  structure = '(-3*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/5. - (3*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/5. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV100 = Lorentz(name = 'FFVVV100',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-5*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/9. - (5*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/9. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV101 = Lorentz(name = 'FFVVV101',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/2. - (Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV102 = Lorentz(name = 'FFVVV102',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-3*Gamma(3,-2,-1)*Gamma(4,-3,-2)*Gamma(5,2,-3)*ProjM(-1,1))/4. + Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - (Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/2. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV103 = Lorentz(name = 'FFVVV103',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-9*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/19. - (9*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/19. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV104 = Lorentz(name = 'FFVVV104',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-20*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/43. - (20*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/43. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV105 = Lorentz(name = 'FFVVV105',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-5*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/11. - (5*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/11. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV106 = Lorentz(name = 'FFVVV106',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-9*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/20. - (9*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/20. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV107 = Lorentz(name = 'FFVVV107',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-11*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/25. - (11*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/25. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV108 = Lorentz(name = 'FFVVV108',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-79*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/182. - (79*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/182. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV109 = Lorentz(name = 'FFVVV109',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-3*Gamma(3,-2,-1)*Gamma(4,-3,-2)*Gamma(5,2,-3)*ProjM(-1,1))/8. + Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + (Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/4. + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV110 = Lorentz(name = 'FFVVV110',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-182*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/79. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV111 = Lorentz(name = 'FFVVV111',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-25*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/11. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV112 = Lorentz(name = 'FFVVV112',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-11*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/5. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV113 = Lorentz(name = 'FFVVV113',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-43*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/20. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV114 = Lorentz(name = 'FFVVV114',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-2*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV115 = Lorentz(name = 'FFVVV115',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-9*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/5. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV116 = Lorentz(name = 'FFVVV116',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-5*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/3. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV117 = Lorentz(name = 'FFVVV117',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-18*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/11. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV118 = Lorentz(name = 'FFVVV118',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1)) - Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV119 = Lorentz(name = 'FFVVV119',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-9*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/10. - (4*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/5. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV120 = Lorentz(name = 'FFVVV120',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(-9*Gamma(3,2,-3)*Gamma(4,-2,-1)*Gamma(5,-3,-2)*ProjM(-1,1))/10. - (4*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/5. + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV121 = Lorentz(name = 'FFVVV121',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV122 = Lorentz(name = 'FFVVV122',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '(3*Gamma(3,-2,-1)*Gamma(4,-3,-2)*Gamma(5,2,-3)*ProjM(-1,1))/2. + Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + 4*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV123 = Lorentz(name = 'FFVVV123',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1))')

FFVVV124 = Lorentz(name = 'FFVVV124',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/6. - (5*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27.')

FFVVV125 = Lorentz(name = 'FFVVV125',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) - (complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - (complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/27. - (complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/27.')

FFVVV126 = Lorentz(name = 'FFVVV126',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (5*complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/27. - (17*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/81. - (5*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/81. - (5*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/81.')

FFVVV127 = Lorentz(name = 'FFVVV127',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) - (complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/9. - (complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/9. - (complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/9.')

FFVVV128 = Lorentz(name = 'FFVVV128',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/9. + (complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/9. + (complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/9.')

FFVVV129 = Lorentz(name = 'FFVVV129',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (5*complex(0,1)*Gamma(3,-3,-2)*Gamma(4,-2,-1)*Gamma(5,2,-3)*ProjM(-1,1))/27. - (5*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/81. - (5*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/81. - (17*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/81.')

FFVVV130 = Lorentz(name = 'FFVVV130',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (11*complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/9. - (29*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - (8*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/27. - (8*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/27.')

FFVVV131 = Lorentz(name = 'FFVVV131',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/6. + (14*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - (complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/3. - (complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/3.')

FFVVV132 = Lorentz(name = 'FFVVV132',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (5*complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/3. - (35*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - (complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/3. - (complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/3.')

FFVVV133 = Lorentz(name = 'FFVVV133',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (4*complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/27. + (37*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/81. - (29*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/81. - (29*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/81.')

FFVVV134 = Lorentz(name = 'FFVVV134',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (4*complex(0,1)*Gamma(3,-3,-2)*Gamma(4,-2,-1)*Gamma(5,2,-3)*ProjM(-1,1))/27. - (29*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/81. - (29*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/81. + (37*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/81.')

FFVVV135 = Lorentz(name = 'FFVVV135',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (11*complex(0,1)*Gamma(3,-3,-2)*Gamma(4,-2,-1)*Gamma(5,2,-3)*ProjM(-1,1))/9. - (8*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - (8*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/27. - (29*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/27.')

FFVVV136 = Lorentz(name = 'FFVVV136',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (5*complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/3. + (125*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - 3*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) - 3*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV137 = Lorentz(name = 'FFVVV137',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (complex(0,1)*Gamma(3,2,-3)*Gamma(4,-2,-1)*Gamma(5,-3,-2)*ProjM(-1,1))/9. - (35*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/9. + (79*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/9. - (37*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/9.')

FFVVV138 = Lorentz(name = 'FFVVV138',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (complex(0,1)*Gamma(3,-3,-2)*Gamma(4,2,-3)*Gamma(5,-2,-1)*ProjM(-1,1))/9. + (35*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/9. - 9*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + (37*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/9.')

FFVVV139 = Lorentz(name = 'FFVVV139',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (23*complex(0,1)*Gamma(3,-2,-1)*Gamma(4,2,-3)*Gamma(5,-3,-2)*ProjM(-1,1))/9. + (185*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - (127*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/27. - (127*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/27.')

FFVVV140 = Lorentz(name = 'FFVVV140',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) - 5*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + 11*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) - 5*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV141 = Lorentz(name = 'FFVVV141',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) - 11*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + 5*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + 5*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV142 = Lorentz(name = 'FFVVV142',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + 5*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) - 11*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + 5*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV143 = Lorentz(name = 'FFVVV143',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + (23*complex(0,1)*Gamma(3,-3,-2)*Gamma(4,-2,-1)*Gamma(5,2,-3)*ProjM(-1,1))/9. - (127*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1))/27. - (127*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1))/27. + (185*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1))/27.')

FFVVV144 = Lorentz(name = 'FFVVV144',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + 5*complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + 5*complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) - 11*complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1)')

FFVVV145 = Lorentz(name = 'FFVVV145',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(3,-3,-2)*Gamma(4,-2,-1)*Gamma(5,2,-3)*ProjP(-1,1)')

FFVVV146 = Lorentz(name = 'FFVVV146',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(3,2,-3)*Gamma(4,-3,-2)*Gamma(5,-2,-1)*ProjP(-1,1)')

FFVVV147 = Lorentz(name = 'FFVVV147',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjP(-1,1)')

FFVVV148 = Lorentz(name = 'FFVVV148',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(4,2,-1)*Metric(3,5)*ProjP(-1,1)')

FFVVV149 = Lorentz(name = 'FFVVV149',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(3,2,-1)*Metric(4,5)*ProjP(-1,1)')

FFVVV150 = Lorentz(name = 'FFVVV150',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1) + (37*Gamma(5,2,-1)*Metric(3,4)*ProjP(-1,1))/64. + (37*Gamma(4,2,-1)*Metric(3,5)*ProjP(-1,1))/64. + (37*Gamma(3,2,-1)*Metric(4,5)*ProjP(-1,1))/64.')

FFVVV151 = Lorentz(name = 'FFVVV151',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjP(-1,1) + Gamma(4,2,-1)*Metric(3,5)*ProjP(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjP(-1,1)')

FFVVV152 = Lorentz(name = 'FFVVV152',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = 'Gamma(5,2,-1)*Metric(3,4)*ProjM(-1,1) + Gamma(4,2,-1)*Metric(3,5)*ProjM(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjM(-1,1) + Gamma(5,2,-1)*Metric(3,4)*ProjP(-1,1) + Gamma(4,2,-1)*Metric(3,5)*ProjP(-1,1) + Gamma(3,2,-1)*Metric(4,5)*ProjP(-1,1)')

FFVVV153 = Lorentz(name = 'FFVVV153',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjP(-2,1))')

FFVVV154 = Lorentz(name = 'FFVVV154',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjM(-2,1)) + Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FFVVV155 = Lorentz(name = 'FFVVV155',
                   spins = [ 2, 2, 3, 3, 3 ],
                   structure = '-(Epsilon(3,4,5,-1)*Gamma(-1,2,-2)*ProjP(-2,1)) - (complex(0,1)*Gamma(5,2,-1)*Metric(3,4)*ProjP(-1,1))/27. - (complex(0,1)*Gamma(4,2,-1)*Metric(3,5)*ProjP(-1,1))/27. - (complex(0,1)*Gamma(3,2,-1)*Metric(4,5)*ProjP(-1,1))/27.')

VSSSS13 = Lorentz(name = 'VSSSS13',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,3)')

VSSSS14 = Lorentz(name = 'VSSSS14',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) + P(1,3) - 2*P(1,4)')

VSSSS15 = Lorentz(name = 'VSSSS15',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,3)/2. - P(1,4)/2.')

VSSSS16 = Lorentz(name = 'VSSSS16',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) + P(1,3) + P(1,4) - 3*P(1,5)')

VSSSS17 = Lorentz(name = 'VSSSS17',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) + P(1,3) - 2*P(1,5)')

VSSSS18 = Lorentz(name = 'VSSSS18',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,3) + P(1,4) - 2*P(1,5)')

VSSSS19 = Lorentz(name = 'VSSSS19',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,5)')

VSSSS20 = Lorentz(name = 'VSSSS20',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) + P(1,3) - P(1,4) - P(1,5)')

VSSSS21 = Lorentz(name = 'VSSSS21',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,4) - P(1,5)')

VSSSS22 = Lorentz(name = 'VSSSS22',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,4)/2. - P(1,5)/2.')

VSSSS23 = Lorentz(name = 'VSSSS23',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,3) - P(1,4)/2. - P(1,5)/2.')

VSSSS24 = Lorentz(name = 'VSSSS24',
                  spins = [ 3, 1, 1, 1, 1 ],
                  structure = 'P(1,2) - P(1,3)/3. - P(1,4)/3. - P(1,5)/3.')

VVSSS2 = Lorentz(name = 'VVSSS2',
                 spins = [ 3, 3, 1, 1, 1 ],
                 structure = 'Metric(1,2)')

VVVSS18 = Lorentz(name = 'VVVSS18',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2)')

VVVSS19 = Lorentz(name = 'VVVSS19',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3)')

VVVSS20 = Lorentz(name = 'VVVSS20',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,4)) + Epsilon(1,2,3,-1)*P(-1,5)')

VVVSS21 = Lorentz(name = 'VVVSS21',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(2,1)*Metric(1,3)')

VVVSS22 = Lorentz(name = 'VVVSS22',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) + P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) - P(1,2)*Metric(2,3)')

VVVSS23 = Lorentz(name = 'VVVSS23',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(2,3)*Metric(1,3) - P(1,3)*Metric(2,3)')

VVVSS24 = Lorentz(name = 'VVVSS24',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,2)*Metric(1,2) + P(2,3)*Metric(1,3) - P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVSS25 = Lorentz(name = 'VVVSS25',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVVSS26 = Lorentz(name = 'VVVSS26',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = 'P(3,4)*Metric(1,2) - P(3,5)*Metric(1,2) + P(2,4)*Metric(1,3) - P(2,5)*Metric(1,3) + P(1,4)*Metric(2,3) - P(1,5)*Metric(2,3)')

VVVSS27 = Lorentz(name = 'VVVSS27',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3) + (complex(0,1)*P(3,4)*Metric(1,2))/12. - (complex(0,1)*P(3,5)*Metric(1,2))/12. + (complex(0,1)*P(2,4)*Metric(1,3))/12. - (complex(0,1)*P(2,5)*Metric(1,3))/12. + (complex(0,1)*P(1,4)*Metric(2,3))/12. - (complex(0,1)*P(1,5)*Metric(2,3))/12.')

VVVSS28 = Lorentz(name = 'VVVSS28',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3) - (complex(0,1)*P(3,4)*Metric(1,2))/12. + (complex(0,1)*P(3,5)*Metric(1,2))/12. - (complex(0,1)*P(2,4)*Metric(1,3))/12. + (complex(0,1)*P(2,5)*Metric(1,3))/12. - (complex(0,1)*P(1,4)*Metric(2,3))/12. + (complex(0,1)*P(1,5)*Metric(2,3))/12.')

VVVSS29 = Lorentz(name = 'VVVSS29',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2) + (complex(0,1)*P(3,4)*Metric(1,2))/6. - (complex(0,1)*P(3,5)*Metric(1,2))/6. + (complex(0,1)*P(2,4)*Metric(1,3))/6. - (complex(0,1)*P(2,5)*Metric(1,3))/6. + (complex(0,1)*P(1,4)*Metric(2,3))/6. - (complex(0,1)*P(1,5)*Metric(2,3))/6.')

VVVSS30 = Lorentz(name = 'VVVSS30',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3) + (complex(0,1)*P(3,4)*Metric(1,2))/6. - (complex(0,1)*P(3,5)*Metric(1,2))/6. + (complex(0,1)*P(2,4)*Metric(1,3))/6. - (complex(0,1)*P(2,5)*Metric(1,3))/6. + (complex(0,1)*P(1,4)*Metric(2,3))/6. - (complex(0,1)*P(1,5)*Metric(2,3))/6.')

VVVSS31 = Lorentz(name = 'VVVSS31',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2) - (complex(0,1)*P(3,4)*Metric(1,2))/6. + (complex(0,1)*P(3,5)*Metric(1,2))/6. - (complex(0,1)*P(2,4)*Metric(1,3))/6. + (complex(0,1)*P(2,5)*Metric(1,3))/6. - (complex(0,1)*P(1,4)*Metric(2,3))/6. + (complex(0,1)*P(1,5)*Metric(2,3))/6.')

VVVSS32 = Lorentz(name = 'VVVSS32',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,2)) + Epsilon(1,2,3,-1)*P(-1,3) - (complex(0,1)*P(3,4)*Metric(1,2))/6. + (complex(0,1)*P(3,5)*Metric(1,2))/6. - (complex(0,1)*P(2,4)*Metric(1,3))/6. + (complex(0,1)*P(2,5)*Metric(1,3))/6. - (complex(0,1)*P(1,4)*Metric(2,3))/6. + (complex(0,1)*P(1,5)*Metric(2,3))/6.')

VVVSS33 = Lorentz(name = 'VVVSS33',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2) + (complex(0,1)*P(3,4)*Metric(1,2))/3. - (complex(0,1)*P(3,5)*Metric(1,2))/3. + (complex(0,1)*P(2,4)*Metric(1,3))/3. - (complex(0,1)*P(2,5)*Metric(1,3))/3. + (complex(0,1)*P(1,4)*Metric(2,3))/3. - (complex(0,1)*P(1,5)*Metric(2,3))/3.')

VVVSS34 = Lorentz(name = 'VVVSS34',
                  spins = [ 3, 3, 3, 1, 1 ],
                  structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2) - (complex(0,1)*P(3,4)*Metric(1,2))/3. + (complex(0,1)*P(3,5)*Metric(1,2))/3. - (complex(0,1)*P(2,4)*Metric(1,3))/3. + (complex(0,1)*P(2,5)*Metric(1,3))/3. - (complex(0,1)*P(1,4)*Metric(2,3))/3. + (complex(0,1)*P(1,5)*Metric(2,3))/3.')

VVVVS11 = Lorentz(name = 'VVVVS11',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Epsilon(1,2,3,4)')

VVVVS12 = Lorentz(name = 'VVVVS12',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3)')

VVVVS13 = Lorentz(name = 'VVVVS13',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4)')

VVVVS14 = Lorentz(name = 'VVVVS14',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVS15 = Lorentz(name = 'VVVVS15',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,2)*Metric(3,4)')

VVVVS16 = Lorentz(name = 'VVVVS16',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVS17 = Lorentz(name = 'VVVVS17',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVS18 = Lorentz(name = 'VVVVS18',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVVS19 = Lorentz(name = 'VVVVS19',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVS20 = Lorentz(name = 'VVVVS20',
                  spins = [ 3, 3, 3, 3, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) + Metric(1,2)*Metric(3,4)')

VVVVV7 = Lorentz(name = 'VVVVV7',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - (P(5,2)*Metric(1,4)*Metric(2,3))/2. - (P(5,3)*Metric(1,4)*Metric(2,3))/2. - P(4,1)*Metric(1,5)*Metric(2,3) + (P(4,2)*Metric(1,5)*Metric(2,3))/2. + (P(4,3)*Metric(1,5)*Metric(2,3))/2. - (P(5,1)*Metric(1,3)*Metric(2,4))/2. + P(5,2)*Metric(1,3)*Metric(2,4) - (P(5,3)*Metric(1,3)*Metric(2,4))/2. + (P(3,1)*Metric(1,5)*Metric(2,4))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(4,1)*Metric(1,3)*Metric(2,5))/2. - P(4,2)*Metric(1,3)*Metric(2,5) + (P(4,3)*Metric(1,3)*Metric(2,5))/2. - (P(3,1)*Metric(1,4)*Metric(2,5))/2. + (P(3,2)*Metric(1,4)*Metric(2,5))/2. - (P(5,1)*Metric(1,2)*Metric(3,4))/2. - (P(5,2)*Metric(1,2)*Metric(3,4))/2. + P(5,3)*Metric(1,2)*Metric(3,4) + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,3)*Metric(2,5)*Metric(3,4))/2. + (P(4,1)*Metric(1,2)*Metric(3,5))/2. + (P(4,2)*Metric(1,2)*Metric(3,5))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(2,1)*Metric(1,4)*Metric(3,5))/2. + (P(2,3)*Metric(1,4)*Metric(3,5))/2. - (P(1,2)*Metric(2,4)*Metric(3,5))/2. + (P(1,3)*Metric(2,4)*Metric(3,5))/2.')

VVVVV8 = Lorentz(name = 'VVVVV8',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + 2*P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + 2*P(3,5)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) - 2*P(2,4)*Metric(1,5)*Metric(3,4) + P(2,5)*Metric(1,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - P(1,5)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) + P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) - 2*P(2,5)*Metric(1,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,1)*Metric(1,2)*Metric(4,5) - P(3,4)*Metric(1,2)*Metric(4,5) - P(3,5)*Metric(1,2)*Metric(4,5) - 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV9 = Lorentz(name = 'VVVVV9',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) + P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(4,2)*Metric(1,3)*Metric(2,5) + P(4,3)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - 2*P(5,1)*Metric(1,2)*Metric(3,4) - 2*P(5,2)*Metric(1,2)*Metric(3,4) + 2*P(5,3)*Metric(1,2)*Metric(3,4) + 2*P(5,4)*Metric(1,2)*Metric(3,4) + 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(1,3)*Metric(2,5)*Metric(3,4) - P(1,4)*Metric(2,5)*Metric(3,4) + P(4,1)*Metric(1,2)*Metric(3,5) + P(4,2)*Metric(1,2)*Metric(3,5) - 2*P(4,3)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) + P(2,3)*Metric(1,4)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,4)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,4)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV10 = Lorentz(name = 'VVVVV10',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + 2*P(4,2)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(5,2)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) - 2*P(3,2)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) + 2*P(4,1)*Metric(1,3)*Metric(2,5) - P(4,2)*Metric(1,3)*Metric(2,5) - P(4,5)*Metric(1,3)*Metric(2,5) - 2*P(3,1)*Metric(1,4)*Metric(2,5) + P(3,2)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(4,1)*Metric(1,2)*Metric(3,5) - P(4,2)*Metric(1,2)*Metric(3,5) + 2*P(4,5)*Metric(1,2)*Metric(3,5) + P(2,1)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) - 2*P(3,5)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV11 = Lorentz(name = 'VVVVV11',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(3,5)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) + P(4,5)*Metric(1,3)*Metric(2,5) - P(3,4)*Metric(1,4)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(2,3)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,5)*Metric(3,4) + 2*P(2,5)*Metric(1,5)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) - 2*P(1,5)*Metric(2,5)*Metric(3,4) - P(2,3)*Metric(1,4)*Metric(3,5) + 2*P(2,4)*Metric(1,4)*Metric(3,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) - 2*P(1,4)*Metric(2,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5) - P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5) + P(1,4)*Metric(2,3)*Metric(4,5) + P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV12 = Lorentz(name = 'VVVVV12',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) + P(5,3)*Metric(1,4)*Metric(2,3) - 2*P(5,4)*Metric(1,4)*Metric(2,3) + P(4,2)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) - 2*P(4,5)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,4)*Metric(1,3)*Metric(2,4) - P(3,2)*Metric(1,5)*Metric(2,4) + P(3,5)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) + P(4,5)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(5,4)*Metric(1,2)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) + P(2,5)*Metric(1,5)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) - P(1,4)*Metric(2,5)*Metric(3,4) - P(1,5)*Metric(2,5)*Metric(3,4) - P(4,2)*Metric(1,2)*Metric(3,5) + P(4,5)*Metric(1,2)*Metric(3,5) - P(2,3)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) + 2*P(3,2)*Metric(1,2)*Metric(4,5) - P(3,4)*Metric(1,2)*Metric(4,5) - P(3,5)*Metric(1,2)*Metric(4,5) + 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5) - P(2,5)*Metric(1,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5) + 2*P(1,4)*Metric(2,3)*Metric(4,5) + 2*P(1,5)*Metric(2,3)*Metric(4,5)')

SSSSSS2 = Lorentz(name = 'SSSSSS2',
                  spins = [ 1, 1, 1, 1, 1, 1 ],
                  structure = '1')

VVSSSS2 = Lorentz(name = 'VVSSSS2',
                  spins = [ 3, 3, 1, 1, 1, 1 ],
                  structure = 'Metric(1,2)')

VVVVSS6 = Lorentz(name = 'VVVVSS6',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVVSS7 = Lorentz(name = 'VVVVSS7',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVVSS8 = Lorentz(name = 'VVVVSS8',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVVSS9 = Lorentz(name = 'VVVVSS9',
                  spins = [ 3, 3, 3, 3, 1, 1 ],
                  structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVVSS10 = Lorentz(name = 'VVVVSS10',
                   spins = [ 3, 3, 3, 3, 1, 1 ],
                   structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVVVV3 = Lorentz(name = 'VVVVVV3',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) + Metric(1,5)*Metric(2,6)*Metric(3,4) - (Metric(1,6)*Metric(2,4)*Metric(3,5))/2. - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - (Metric(1,6)*Metric(2,3)*Metric(4,5))/2. - (Metric(1,3)*Metric(2,6)*Metric(4,5))/2. + Metric(1,2)*Metric(3,6)*Metric(4,5) - (Metric(1,5)*Metric(2,3)*Metric(4,6))/2. - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - 2*Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV4 = Lorentz(name = 'VVVVVV4',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - (Metric(1,5)*Metric(2,6)*Metric(3,4))/2. + Metric(1,6)*Metric(2,4)*Metric(3,5) - (Metric(1,4)*Metric(2,6)*Metric(3,5))/2. - (Metric(1,5)*Metric(2,4)*Metric(3,6))/2. - (Metric(1,4)*Metric(2,5)*Metric(3,6))/2. - 2*Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - (Metric(1,3)*Metric(2,5)*Metric(4,6))/2. - (Metric(1,2)*Metric(3,5)*Metric(4,6))/2. + Metric(1,4)*Metric(2,3)*Metric(5,6) - (Metric(1,3)*Metric(2,4)*Metric(5,6))/2. - (Metric(1,2)*Metric(3,4)*Metric(5,6))/2.')

