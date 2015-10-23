# This file was automatically created by FeynRules $Revision: 634 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Tue 22 Nov 2011 16:45:55


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec



GC_1 = Coupling(name = 'GC_1',
                value = '-(ee*complex(0,1))/3.',
                order = {'QED':1})

GC_2 = Coupling(name = 'GC_2',
                value = '(2*ee*complex(0,1))/3.',
                order = {'QED':1})

GC_3 = Coupling(name = 'GC_3',
                value = '-(ee*complex(0,1))',
                order = {'QED':1})

GC_4 = Coupling(name = 'GC_4',
                value = '-G',
                order = {'QCD':1})

GC_5 = Coupling(name = 'GC_5',
                value = 'complex(0,1)*G',
                order = {'QCD':1})

GC_6 = Coupling(name = 'GC_6',
                value = 'complex(0,1)*G**2',
                order = {'QCD':2})

GC_7 = Coupling(name = 'GC_7',
                value = 'cw*complex(0,1)*gw',
                order = {'QED':1})

GC_8 = Coupling(name = 'GC_8',
                value = '-(complex(0,1)*gw**2)',
                order = {'QED':2})

GC_9 = Coupling(name = 'GC_9',
                value = 'cw**2*complex(0,1)*gw**2',
                order = {'QED':2})

GC_10 = Coupling(name = 'GC_10',
                 value = '(complex(0,1)*kappa)/8.',
                 order = {'QTD':1})

GC_11 = Coupling(name = 'GC_11',
                 value = '-(complex(0,1)*kappa)/2.',
                 order = {'QTD':1})

GC_12 = Coupling(name = 'GC_12',
                 value = '(complex(0,1)*kappa)/2.',
                 order = {'QTD':1})

GC_13 = Coupling(name = 'GC_13',
                 value = '-(ee*complex(0,1)*kappa)/6.',
                 order = {'QED':1,'QTD':1})

GC_14 = Coupling(name = 'GC_14',
                 value = '(ee*complex(0,1)*kappa)/3.',
                 order = {'QED':1,'QTD':1})

GC_15 = Coupling(name = 'GC_15',
                 value = '-(ee*complex(0,1)*kappa)/2.',
                 order = {'QED':1,'QTD':1})

GC_16 = Coupling(name = 'GC_16',
                 value = 'ee**2*complex(0,1)*kappa',
                 order = {'QED':2,'QTD':1})

GC_17 = Coupling(name = 'GC_17',
                 value = '-(complex(0,1)*G*kappa)/4.',
                 order = {'QCD':1,'QTD':1})

GC_18 = Coupling(name = 'GC_18',
                 value = '(G*kappa)/2.',
                 order = {'QCD':1,'QTD':1})

GC_19 = Coupling(name = 'GC_19',
                 value = '(complex(0,1)*G**2*kappa)/2.',
                 order = {'QCD':2,'QTD':1})

GC_20 = Coupling(name = 'GC_20',
                 value = '-6*complex(0,1)*lam',
                 order = {'QED':2})

GC_21 = Coupling(name = 'GC_21',
                 value = '-3*complex(0,1)*kappa*lam',
                 order = {'QED':2,'QTD':1})

GC_22 = Coupling(name = 'GC_22',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_23 = Coupling(name = 'GC_23',
                 value = '-(ee**2*complex(0,1)*kappa)/(4.*sw**2)',
                 order = {'QED':2,'QTD':1})

GC_24 = Coupling(name = 'GC_24',
                 value = '-((ee**2*complex(0,1)*kappa)/sw**2)',
                 order = {'QED':2,'QTD':1})

GC_25 = Coupling(name = 'GC_25',
                 value = '(cw**2*ee**2*complex(0,1)*kappa)/sw**2',
                 order = {'QED':2,'QTD':1})

GC_26 = Coupling(name = 'GC_26',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_27 = Coupling(name = 'GC_27',
                 value = '(CKM11*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_28 = Coupling(name = 'GC_28',
                 value = '(CKM12*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_29 = Coupling(name = 'GC_29',
                 value = '(CKM13*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_30 = Coupling(name = 'GC_30',
                 value = '(CKM21*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_31 = Coupling(name = 'GC_31',
                 value = '(CKM22*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_32 = Coupling(name = 'GC_32',
                 value = '(CKM23*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_33 = Coupling(name = 'GC_33',
                 value = '(CKM31*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_34 = Coupling(name = 'GC_34',
                 value = '(CKM32*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_35 = Coupling(name = 'GC_35',
                 value = '(CKM33*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = '-(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_38 = Coupling(name = 'GC_38',
                 value = '(ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '(CKM11*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(CKM12*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_41 = Coupling(name = 'GC_41',
                 value = '(CKM13*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '(CKM21*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '(CKM22*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_44 = Coupling(name = 'GC_44',
                 value = '(CKM23*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_45 = Coupling(name = 'GC_45',
                 value = '(CKM31*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_46 = Coupling(name = 'GC_46',
                 value = '(CKM32*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_47 = Coupling(name = 'GC_47',
                 value = '(CKM33*ee*complex(0,1)*kappa)/(2.*sw*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_48 = Coupling(name = 'GC_48',
                 value = '-(cw*ee*complex(0,1)*kappa)/(4.*sw)',
                 order = {'QED':1,'QTD':1})

GC_49 = Coupling(name = 'GC_49',
                 value = '(cw*ee*complex(0,1)*kappa)/(4.*sw)',
                 order = {'QED':1,'QTD':1})

GC_50 = Coupling(name = 'GC_50',
                 value = '-(cw*ee*complex(0,1)*kappa)/(2.*sw)',
                 order = {'QED':1,'QTD':1})

GC_51 = Coupling(name = 'GC_51',
                 value = '-(cw*ee**2*complex(0,1)*kappa)/(2.*sw)',
                 order = {'QED':2,'QTD':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '-(ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_53 = Coupling(name = 'GC_53',
                 value = '(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_54 = Coupling(name = 'GC_54',
                 value = 'complex(0,1)*gw*sw',
                 order = {'QED':1})

GC_55 = Coupling(name = 'GC_55',
                 value = '-2*cw*complex(0,1)*gw**2*sw',
                 order = {'QED':2})

GC_56 = Coupling(name = 'GC_56',
                 value = '-(ee*complex(0,1)*kappa*sw)/(12.*cw)',
                 order = {'QED':1,'QTD':1})

GC_57 = Coupling(name = 'GC_57',
                 value = '(ee*complex(0,1)*kappa*sw)/(4.*cw)',
                 order = {'QED':1,'QTD':1})

GC_58 = Coupling(name = 'GC_58',
                 value = 'complex(0,1)*gw**2*sw**2',
                 order = {'QED':2})

GC_59 = Coupling(name = 'GC_59',
                 value = '(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_60 = Coupling(name = 'GC_60',
                 value = '(cw*ee*complex(0,1)*kappa)/(4.*sw) + (ee*complex(0,1)*kappa*sw)/(4.*cw)',
                 order = {'QED':1,'QTD':1})

GC_61 = Coupling(name = 'GC_61',
                 value = 'ee**2*complex(0,1) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_62 = Coupling(name = 'GC_62',
                 value = '-(cw**2*complex(0,1)*kappa)/2. - (complex(0,1)*kappa*sw**2)/2.',
                 order = {'QTD':1})

GC_63 = Coupling(name = 'GC_63',
                 value = '-(ee**2*complex(0,1)*kappa)/2. - (cw**2*ee**2*complex(0,1)*kappa)/(4.*sw**2) - (ee**2*complex(0,1)*kappa*sw**2)/(4.*cw**2)',
                 order = {'QED':2,'QTD':1})

GC_64 = Coupling(name = 'GC_64',
                 value = '-6*complex(0,1)*lam*v',
                 order = {'QED':1})

GC_65 = Coupling(name = 'GC_65',
                 value = '-3*complex(0,1)*kappa*lam*v',
                 order = {'QED':1,'QTD':1})

GC_66 = Coupling(name = 'GC_66',
                 value = '(ee**2*complex(0,1)*v)/(2.*sw**2)',
                 order = {'QED':1})

GC_67 = Coupling(name = 'GC_67',
                 value = '-(ee**2*complex(0,1)*kappa*v)/(4.*sw**2)',
                 order = {'QED':1,'QTD':1})

GC_68 = Coupling(name = 'GC_68',
                 value = '-(ee**2*complex(0,1)*kappa*v**2)/(8.*sw**2)',
                 order = {'QTD':1})

GC_69 = Coupling(name = 'GC_69',
                 value = 'ee**2*complex(0,1)*v + (cw**2*ee**2*complex(0,1)*v)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*v)/(2.*cw**2)',
                 order = {'QED':1})

GC_70 = Coupling(name = 'GC_70',
                 value = '-(ee**2*complex(0,1)*kappa*v)/2. - (cw**2*ee**2*complex(0,1)*kappa*v)/(4.*sw**2) - (ee**2*complex(0,1)*kappa*sw**2*v)/(4.*cw**2)',
                 order = {'QED':1,'QTD':1})

GC_71 = Coupling(name = 'GC_71',
                 value = '(complex(0,1)*kappa*muH**2)/2. - (3*complex(0,1)*kappa*lam*v**2)/2.',
                 order = {'QTD':1})

GC_72 = Coupling(name = 'GC_72',
                 value = '-(ee**2*complex(0,1)*kappa*v**2)/4. - (cw**2*ee**2*complex(0,1)*kappa*v**2)/(8.*sw**2) - (ee**2*complex(0,1)*kappa*sw**2*v**2)/(8.*cw**2)',
                 order = {'QTD':1})

GC_73 = Coupling(name = 'GC_73',
                 value = '-((complex(0,1)*yb)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_74 = Coupling(name = 'GC_74',
                 value = '-(complex(0,1)*kappa*yb)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_75 = Coupling(name = 'GC_75',
                 value = '-(complex(0,1)*kappa*v*yb)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_76 = Coupling(name = 'GC_76',
                 value = '-((complex(0,1)*yc)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '-(complex(0,1)*kappa*yc)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_78 = Coupling(name = 'GC_78',
                 value = '-(complex(0,1)*kappa*v*yc)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_79 = Coupling(name = 'GC_79',
                 value = '-((complex(0,1)*yd)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_80 = Coupling(name = 'GC_80',
                 value = '-(complex(0,1)*kappa*yd)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_81 = Coupling(name = 'GC_81',
                 value = '-(complex(0,1)*kappa*v*yd)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_82 = Coupling(name = 'GC_82',
                 value = '-((complex(0,1)*ye)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_83 = Coupling(name = 'GC_83',
                 value = '-(complex(0,1)*kappa*ye)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_84 = Coupling(name = 'GC_84',
                 value = '-(complex(0,1)*kappa*v*ye)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_85 = Coupling(name = 'GC_85',
                 value = '-((complex(0,1)*ym)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_86 = Coupling(name = 'GC_86',
                 value = '-(complex(0,1)*kappa*ym)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_87 = Coupling(name = 'GC_87',
                 value = '-(complex(0,1)*kappa*v*ym)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_88 = Coupling(name = 'GC_88',
                 value = '-((complex(0,1)*ys)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_89 = Coupling(name = 'GC_89',
                 value = '-(complex(0,1)*kappa*ys)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_90 = Coupling(name = 'GC_90',
                 value = '-(complex(0,1)*kappa*v*ys)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_91 = Coupling(name = 'GC_91',
                 value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_92 = Coupling(name = 'GC_92',
                 value = '-(complex(0,1)*kappa*yt)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_93 = Coupling(name = 'GC_93',
                 value = '-(complex(0,1)*kappa*v*yt)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_94 = Coupling(name = 'GC_94',
                 value = '-((complex(0,1)*ytau)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_95 = Coupling(name = 'GC_95',
                 value = '-(complex(0,1)*kappa*ytau)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_96 = Coupling(name = 'GC_96',
                 value = '-(complex(0,1)*kappa*v*ytau)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '-((complex(0,1)*yu)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_98 = Coupling(name = 'GC_98',
                 value = '-(complex(0,1)*kappa*yu)/(2.*cmath.sqrt(2))',
                 order = {'QED':1,'QTD':1})

GC_99 = Coupling(name = 'GC_99',
                 value = '-(complex(0,1)*kappa*v*yu)/(2.*cmath.sqrt(2))',
                 order = {'QTD':1})

GC_100 = Coupling(name = 'GC_100',
                  value = '(ee*complex(0,1)*complexconjugate(CKM11))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_101 = Coupling(name = 'GC_101',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM11))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_102 = Coupling(name = 'GC_102',
                  value = '(ee*complex(0,1)*complexconjugate(CKM12))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_103 = Coupling(name = 'GC_103',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM12))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_104 = Coupling(name = 'GC_104',
                  value = '(ee*complex(0,1)*complexconjugate(CKM13))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_105 = Coupling(name = 'GC_105',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM13))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_106 = Coupling(name = 'GC_106',
                  value = '(ee*complex(0,1)*complexconjugate(CKM21))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_107 = Coupling(name = 'GC_107',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM21))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_108 = Coupling(name = 'GC_108',
                  value = '(ee*complex(0,1)*complexconjugate(CKM22))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_109 = Coupling(name = 'GC_109',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM22))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_110 = Coupling(name = 'GC_110',
                  value = '(ee*complex(0,1)*complexconjugate(CKM23))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_111 = Coupling(name = 'GC_111',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM23))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_112 = Coupling(name = 'GC_112',
                  value = '(ee*complex(0,1)*complexconjugate(CKM31))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_113 = Coupling(name = 'GC_113',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM31))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_114 = Coupling(name = 'GC_114',
                  value = '(ee*complex(0,1)*complexconjugate(CKM32))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_115 = Coupling(name = 'GC_115',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM32))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

GC_116 = Coupling(name = 'GC_116',
                  value = '(ee*complex(0,1)*complexconjugate(CKM33))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_117 = Coupling(name = 'GC_117',
                  value = '(ee*complex(0,1)*kappa*complexconjugate(CKM33))/(2.*sw*cmath.sqrt(2))',
                  order = {'QED':1,'QTD':1})

