# This file was automatically created by FeynRules 1.7.9
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Fri 18 May 2012 14:43:25


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
                value = '(CPWL2*cw*complex(0,1)*gw)/2.e6',
                order = {'NP':2})

GC_9 = Coupling(name = 'GC_9',
                value = '-(CPWL2*cw**2*complex(0,1)*g1*gw)/2.e6',
                order = {'NP':2,'QED':1})

GC_10 = Coupling(name = 'GC_10',
                 value = '-(complex(0,1)*gw**2)',
                 order = {'QED':2})

GC_11 = Coupling(name = 'GC_11',
                 value = '-(CPWL2*complex(0,1)*gw**2)/2.e6',
                 order = {'NP':2,'QED':1})

GC_12 = Coupling(name = 'GC_12',
                 value = 'cw**2*complex(0,1)*gw**2',
                 order = {'QED':2})

GC_13 = Coupling(name = 'GC_13',
                 value = '(CWL2*complex(0,1)*gw**2)/4.e6',
                 order = {'NP':2,'QED':1})

GC_14 = Coupling(name = 'GC_14',
                 value = '-(CBL2*cw*complex(0,1)*g1*gw**2)/4.e6',
                 order = {'NP':2,'QED':2})

GC_15 = Coupling(name = 'GC_15',
                 value = '(CPWL2*cw*complex(0,1)*g1*gw**2)/2.e6',
                 order = {'NP':2,'QED':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '-(cw*CWL2*complex(0,1)*g1*gw**2)/4.e6',
                 order = {'NP':2,'QED':2})

GC_17 = Coupling(name = 'GC_17',
                 value = '-(CPWL2*cw*complex(0,1)*gw**3)/2.e6',
                 order = {'NP':2,'QED':2})

GC_18 = Coupling(name = 'GC_18',
                 value = '(CPWWWL2*cw*complex(0,1)*gw**3)/2.e6',
                 order = {'NP':2})

GC_19 = Coupling(name = 'GC_19',
                 value = '-(cw*CWL2*complex(0,1)*gw**3)/4.e6',
                 order = {'NP':2,'QED':2})

GC_20 = Coupling(name = 'GC_20',
                 value = '(3*cw*CWWWL2*complex(0,1)*gw**3)/2.e6',
                 order = {'NP':2})

GC_21 = Coupling(name = 'GC_21',
                 value = '-(CPWWWL2*complex(0,1)*gw**4)/1.e6',
                 order = {'NP':2,'QED':1})

GC_22 = Coupling(name = 'GC_22',
                 value = '(CPWWWL2*cw**2*complex(0,1)*gw**4)/1.e6',
                 order = {'NP':2,'QED':1})

GC_23 = Coupling(name = 'GC_23',
                 value = '(CWL2*complex(0,1)*gw**4)/2.e6',
                 order = {'NP':2,'QED':3})

GC_24 = Coupling(name = 'GC_24',
                 value = '(-3*CWWWL2*complex(0,1)*gw**4)/2.e6',
                 order = {'NP':2,'QED':1})

GC_25 = Coupling(name = 'GC_25',
                 value = '(3*cw**2*CWWWL2*complex(0,1)*gw**4)/2.e6',
                 order = {'NP':2,'QED':1})

GC_26 = Coupling(name = 'GC_26',
                 value = '-(CPWWWL2*cw*complex(0,1)*gw**5)/500000.',
                 order = {'NP':2,'QED':2})

GC_27 = Coupling(name = 'GC_27',
                 value = '-(CPWWWL2*cw**3*complex(0,1)*gw**5)/500000.',
                 order = {'NP':2,'QED':2})

GC_28 = Coupling(name = 'GC_28',
                 value = '(-3*cw*CWWWL2*complex(0,1)*gw**5)/2.e6',
                 order = {'NP':2,'QED':2})

GC_29 = Coupling(name = 'GC_29',
                 value = '(-3*cw**3*CWWWL2*complex(0,1)*gw**5)/2.e6',
                 order = {'NP':2,'QED':2})

GC_30 = Coupling(name = 'GC_30',
                 value = '(3*cw**2*CWWWL2*complex(0,1)*gw**6)/500000.',
                 order = {'NP':2,'QED':3})

GC_31 = Coupling(name = 'GC_31',
                 value = '-6*complex(0,1)*lam',
                 order = {'QED':2})

GC_32 = Coupling(name = 'GC_32',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_33 = Coupling(name = 'GC_33',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_34 = Coupling(name = 'GC_34',
                 value = '(CKM1x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_35 = Coupling(name = 'GC_35',
                 value = '(CKM1x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = '(CKM1x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = '(CKM2x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_38 = Coupling(name = 'GC_38',
                 value = '(CKM2x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '(CKM2x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(CKM3x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_41 = Coupling(name = 'GC_41',
                 value = '(CKM3x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '(CKM3x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '-(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_44 = Coupling(name = 'GC_44',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_45 = Coupling(name = 'GC_45',
                 value = '-(ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_46 = Coupling(name = 'GC_46',
                 value = '(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_47 = Coupling(name = 'GC_47',
                 value = 'complex(0,1)*gw*sw',
                 order = {'QED':1})

GC_48 = Coupling(name = 'GC_48',
                 value = '(CPWL2*complex(0,1)*gw*sw)/2.e6',
                 order = {'NP':2})

GC_49 = Coupling(name = 'GC_49',
                 value = '-2*cw*complex(0,1)*gw**2*sw',
                 order = {'QED':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '-(CPWL2*cw*complex(0,1)*gw**2*sw)/2.e6',
                 order = {'NP':2,'QED':1})

GC_51 = Coupling(name = 'GC_51',
                 value = '(CBL2*complex(0,1)*g1*gw**2*sw)/4.e6',
                 order = {'NP':2,'QED':2})

GC_52 = Coupling(name = 'GC_52',
                 value = '-(CPWL2*complex(0,1)*g1*gw**2*sw)/2.e6',
                 order = {'NP':2,'QED':2})

GC_53 = Coupling(name = 'GC_53',
                 value = '-(CWL2*complex(0,1)*g1*gw**2*sw)/4.e6',
                 order = {'NP':2,'QED':2})

GC_54 = Coupling(name = 'GC_54',
                 value = '-(CPWL2*complex(0,1)*gw**3*sw)/2.e6',
                 order = {'NP':2,'QED':2})

GC_55 = Coupling(name = 'GC_55',
                 value = '(CPWWWL2*complex(0,1)*gw**3*sw)/2.e6',
                 order = {'NP':2})

GC_56 = Coupling(name = 'GC_56',
                 value = '-(CWL2*complex(0,1)*gw**3*sw)/4.e6',
                 order = {'NP':2,'QED':2})

GC_57 = Coupling(name = 'GC_57',
                 value = '(3*CWWWL2*complex(0,1)*gw**3*sw)/2.e6',
                 order = {'NP':2})

GC_58 = Coupling(name = 'GC_58',
                 value = '(CPWWWL2*cw*complex(0,1)*gw**4*sw)/1.e6',
                 order = {'NP':2,'QED':1})

GC_59 = Coupling(name = 'GC_59',
                 value = '(-3*cw*CWWWL2*complex(0,1)*gw**4*sw)/2.e6',
                 order = {'NP':2,'QED':1})

GC_60 = Coupling(name = 'GC_60',
                 value = '(CPWWWL2*complex(0,1)*gw**5*sw)/500000.',
                 order = {'NP':2,'QED':2})

GC_61 = Coupling(name = 'GC_61',
                 value = '-(CPWWWL2*cw**2*complex(0,1)*gw**5*sw)/500000.',
                 order = {'NP':2,'QED':2})

GC_62 = Coupling(name = 'GC_62',
                 value = '(-3*CWWWL2*complex(0,1)*gw**5*sw)/2.e6',
                 order = {'NP':2,'QED':2})

GC_63 = Coupling(name = 'GC_63',
                 value = '(-3*cw**2*CWWWL2*complex(0,1)*gw**5*sw)/2.e6',
                 order = {'NP':2,'QED':2})

GC_64 = Coupling(name = 'GC_64',
                 value = '(3*cw*CWWWL2*complex(0,1)*gw**6*sw)/500000.',
                 order = {'NP':2,'QED':3})

GC_65 = Coupling(name = 'GC_65',
                 value = '-(CPWL2*complex(0,1)*g1*gw*sw**2)/2.e6',
                 order = {'NP':2,'QED':1})

GC_66 = Coupling(name = 'GC_66',
                 value = 'complex(0,1)*gw**2*sw**2',
                 order = {'QED':2})

GC_67 = Coupling(name = 'GC_67',
                 value = '(CPWWWL2*complex(0,1)*gw**4*sw**2)/1.e6',
                 order = {'NP':2,'QED':1})

GC_68 = Coupling(name = 'GC_68',
                 value = '(3*CWWWL2*complex(0,1)*gw**4*sw**2)/2.e6',
                 order = {'NP':2,'QED':1})

GC_69 = Coupling(name = 'GC_69',
                 value = '-(CPWWWL2*cw*complex(0,1)*gw**5*sw**2)/500000.',
                 order = {'NP':2,'QED':2})

GC_70 = Coupling(name = 'GC_70',
                 value = '(-3*cw*CWWWL2*complex(0,1)*gw**5*sw**2)/2.e6',
                 order = {'NP':2,'QED':2})

GC_71 = Coupling(name = 'GC_71',
                 value = '(3*CWWWL2*complex(0,1)*gw**6*sw**2)/500000.',
                 order = {'NP':2,'QED':3})

GC_72 = Coupling(name = 'GC_72',
                 value = '-(CPWWWL2*complex(0,1)*gw**5*sw**3)/500000.',
                 order = {'NP':2,'QED':2})

GC_73 = Coupling(name = 'GC_73',
                 value = '(-3*CWWWL2*complex(0,1)*gw**5*sw**3)/1.e6',
                 order = {'NP':2,'QED':2})

GC_74 = Coupling(name = 'GC_74',
                 value = '(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_75 = Coupling(name = 'GC_75',
                 value = '-(CPWL2*cw**2*complex(0,1)*gw**2)/2.e6 - (CPWL2*cw*complex(0,1)*g1*gw*sw)/2.e6',
                 order = {'NP':2,'QED':1})

GC_76 = Coupling(name = 'GC_76',
                 value = '-(CBL2*cw*complex(0,1)*g1**2*sw)/4.e6 + (cw*CWL2*complex(0,1)*gw**2*sw)/4.e6',
                 order = {'NP':2,'QED':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '-(cw**2*CWL2*complex(0,1)*gw**4)/2.e6 - (cw*CWL2*complex(0,1)*g1*gw**3*sw)/2.e6',
                 order = {'NP':2,'QED':3})

GC_78 = Coupling(name = 'GC_78',
                 value = 'ee**2*complex(0,1) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_79 = Coupling(name = 'GC_79',
                 value = '(cw**2*CWL2*complex(0,1)*gw**2)/4.e6 + (CBL2*cw*complex(0,1)*g1*gw*sw)/4.e6 + (cw*CWL2*complex(0,1)*g1*gw*sw)/4.e6 + (CBL2*complex(0,1)*g1**2*sw**2)/4.e6',
                 order = {'NP':2,'QED':1})

GC_80 = Coupling(name = 'GC_80',
                 value = '-(cw**2*CWL2*complex(0,1)*g1*gw)/4.e6 + (CBL2*complex(0,1)*g1*gw*sw**2)/4.e6',
                 order = {'NP':2,'QED':1})

GC_81 = Coupling(name = 'GC_81',
                 value = '-(CBL2*cw**2*complex(0,1)*g1*gw)/4.e6 + (CWL2*complex(0,1)*g1*gw*sw**2)/4.e6',
                 order = {'NP':2,'QED':1})

GC_82 = Coupling(name = 'GC_82',
                 value = '(CPWL2*cw*complex(0,1)*g1*gw*sw)/2.e6 - (CPWL2*complex(0,1)*gw**2*sw**2)/2.e6',
                 order = {'NP':2,'QED':1})

GC_83 = Coupling(name = 'GC_83',
                 value = '(CBL2*cw**2*complex(0,1)*g1**2)/4.e6 - (CBL2*cw*complex(0,1)*g1*gw*sw)/4.e6 - (cw*CWL2*complex(0,1)*g1*gw*sw)/4.e6 + (CWL2*complex(0,1)*gw**2*sw**2)/4.e6',
                 order = {'NP':2,'QED':1})

GC_84 = Coupling(name = 'GC_84',
                 value = '-(cw**2*CWL2*complex(0,1)*g1*gw**3)/2.e6 + (cw*CWL2*complex(0,1)*gw**4*sw)/1.e6 + (CWL2*complex(0,1)*g1*gw**3*sw**2)/2.e6',
                 order = {'NP':2,'QED':3})

GC_85 = Coupling(name = 'GC_85',
                 value = '(cw*CWL2*complex(0,1)*g1*gw**3*sw)/2.e6 - (CWL2*complex(0,1)*gw**4*sw**2)/2.e6',
                 order = {'NP':2,'QED':3})

GC_86 = Coupling(name = 'GC_86',
                 value = '-(CPWL2*cw**2*complex(0,1)*g1*gw*v)/2.e6',
                 order = {'NP':2})

GC_87 = Coupling(name = 'GC_87',
                 value = '-(CPWL2*complex(0,1)*gw**2*v)/2.e6',
                 order = {'NP':2})

GC_88 = Coupling(name = 'GC_88',
                 value = '(CWL2*complex(0,1)*gw**2*v)/4.e6',
                 order = {'NP':2})

GC_89 = Coupling(name = 'GC_89',
                 value = '-(CBL2*cw*complex(0,1)*g1*gw**2*v)/4.e6',
                 order = {'NP':2,'QED':1})

GC_90 = Coupling(name = 'GC_90',
                 value = '(CPWL2*cw*complex(0,1)*g1*gw**2*v)/2.e6',
                 order = {'NP':2,'QED':1})

GC_91 = Coupling(name = 'GC_91',
                 value = '-(cw*CWL2*complex(0,1)*g1*gw**2*v)/4.e6',
                 order = {'NP':2,'QED':1})

GC_92 = Coupling(name = 'GC_92',
                 value = '-(CPWL2*cw*complex(0,1)*gw**3*v)/2.e6',
                 order = {'NP':2,'QED':1})

GC_93 = Coupling(name = 'GC_93',
                 value = '-(cw*CWL2*complex(0,1)*gw**3*v)/4.e6',
                 order = {'NP':2,'QED':1})

GC_94 = Coupling(name = 'GC_94',
                 value = '(CWL2*complex(0,1)*gw**4*v)/2.e6',
                 order = {'NP':2,'QED':2})

GC_95 = Coupling(name = 'GC_95',
                 value = '-6*complex(0,1)*lam*v',
                 order = {'QED':1})

GC_96 = Coupling(name = 'GC_96',
                 value = '(ee**2*complex(0,1)*v)/(2.*sw**2)',
                 order = {'QED':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '-(CPWL2*cw*complex(0,1)*gw**2*sw*v)/2.e6',
                 order = {'NP':2})

GC_98 = Coupling(name = 'GC_98',
                 value = '(CBL2*complex(0,1)*g1*gw**2*sw*v)/4.e6',
                 order = {'NP':2,'QED':1})

GC_99 = Coupling(name = 'GC_99',
                 value = '-(CPWL2*complex(0,1)*g1*gw**2*sw*v)/2.e6',
                 order = {'NP':2,'QED':1})

GC_100 = Coupling(name = 'GC_100',
                  value = '-(CWL2*complex(0,1)*g1*gw**2*sw*v)/4.e6',
                  order = {'NP':2,'QED':1})

GC_101 = Coupling(name = 'GC_101',
                  value = '-(CPWL2*complex(0,1)*gw**3*sw*v)/2.e6',
                  order = {'NP':2,'QED':1})

GC_102 = Coupling(name = 'GC_102',
                  value = '-(CWL2*complex(0,1)*gw**3*sw*v)/4.e6',
                  order = {'NP':2,'QED':1})

GC_103 = Coupling(name = 'GC_103',
                  value = '-(CPWL2*complex(0,1)*g1*gw*sw**2*v)/2.e6',
                  order = {'NP':2})

GC_104 = Coupling(name = 'GC_104',
                  value = '-(CBL2*cw*complex(0,1)*g1*gw**2*v**2)/8.e6',
                  order = {'NP':2})

GC_105 = Coupling(name = 'GC_105',
                  value = '(CPWL2*cw*complex(0,1)*g1*gw**2*v**2)/4.e6',
                  order = {'NP':2})

GC_106 = Coupling(name = 'GC_106',
                  value = '-(cw*CWL2*complex(0,1)*g1*gw**2*v**2)/8.e6',
                  order = {'NP':2})

GC_107 = Coupling(name = 'GC_107',
                  value = '-(CPWL2*cw*complex(0,1)*gw**3*v**2)/4.e6',
                  order = {'NP':2})

GC_108 = Coupling(name = 'GC_108',
                  value = '-(cw*CWL2*complex(0,1)*gw**3*v**2)/8.e6',
                  order = {'NP':2})

GC_109 = Coupling(name = 'GC_109',
                  value = '(CWL2*complex(0,1)*gw**4*v**2)/4.e6',
                  order = {'NP':2,'QED':1})

GC_110 = Coupling(name = 'GC_110',
                  value = '(CBL2*complex(0,1)*g1*gw**2*sw*v**2)/8.e6',
                  order = {'NP':2})

GC_111 = Coupling(name = 'GC_111',
                  value = '-(CPWL2*complex(0,1)*g1*gw**2*sw*v**2)/4.e6',
                  order = {'NP':2})

GC_112 = Coupling(name = 'GC_112',
                  value = '-(CWL2*complex(0,1)*g1*gw**2*sw*v**2)/8.e6',
                  order = {'NP':2})

GC_113 = Coupling(name = 'GC_113',
                  value = '-(CPWL2*complex(0,1)*gw**3*sw*v**2)/4.e6',
                  order = {'NP':2})

GC_114 = Coupling(name = 'GC_114',
                  value = '-(CWL2*complex(0,1)*gw**3*sw*v**2)/8.e6',
                  order = {'NP':2})

GC_115 = Coupling(name = 'GC_115',
                  value = '-(CPWL2*cw**2*complex(0,1)*gw**2*v)/2.e6 - (CPWL2*cw*complex(0,1)*g1*gw*sw*v)/2.e6',
                  order = {'NP':2})

GC_116 = Coupling(name = 'GC_116',
                  value = '-(CBL2*cw*complex(0,1)*g1**2*sw*v)/4.e6 + (cw*CWL2*complex(0,1)*gw**2*sw*v)/4.e6',
                  order = {'NP':2})

GC_117 = Coupling(name = 'GC_117',
                  value = '-(cw**2*CWL2*complex(0,1)*gw**4*v)/2.e6 - (cw*CWL2*complex(0,1)*g1*gw**3*sw*v)/2.e6',
                  order = {'NP':2,'QED':2})

GC_118 = Coupling(name = 'GC_118',
                  value = 'ee**2*complex(0,1)*v + (cw**2*ee**2*complex(0,1)*v)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*v)/(2.*cw**2)',
                  order = {'QED':1})

GC_119 = Coupling(name = 'GC_119',
                  value = '(cw**2*CWL2*complex(0,1)*gw**2*v)/4.e6 + (CBL2*cw*complex(0,1)*g1*gw*sw*v)/4.e6 + (cw*CWL2*complex(0,1)*g1*gw*sw*v)/4.e6 + (CBL2*complex(0,1)*g1**2*sw**2*v)/4.e6',
                  order = {'NP':2})

GC_120 = Coupling(name = 'GC_120',
                  value = '-(cw**2*CWL2*complex(0,1)*g1*gw*v)/4.e6 + (CBL2*complex(0,1)*g1*gw*sw**2*v)/4.e6',
                  order = {'NP':2})

GC_121 = Coupling(name = 'GC_121',
                  value = '-(CBL2*cw**2*complex(0,1)*g1*gw*v)/4.e6 + (CWL2*complex(0,1)*g1*gw*sw**2*v)/4.e6',
                  order = {'NP':2})

GC_122 = Coupling(name = 'GC_122',
                  value = '(CPWL2*cw*complex(0,1)*g1*gw*sw*v)/2.e6 - (CPWL2*complex(0,1)*gw**2*sw**2*v)/2.e6',
                  order = {'NP':2})

GC_123 = Coupling(name = 'GC_123',
                  value = '(CBL2*cw**2*complex(0,1)*g1**2*v)/4.e6 - (CBL2*cw*complex(0,1)*g1*gw*sw*v)/4.e6 - (cw*CWL2*complex(0,1)*g1*gw*sw*v)/4.e6 + (CWL2*complex(0,1)*gw**2*sw**2*v)/4.e6',
                  order = {'NP':2})

GC_124 = Coupling(name = 'GC_124',
                  value = '-(cw**2*CWL2*complex(0,1)*g1*gw**3*v)/2.e6 + (cw*CWL2*complex(0,1)*gw**4*sw*v)/1.e6 + (CWL2*complex(0,1)*g1*gw**3*sw**2*v)/2.e6',
                  order = {'NP':2,'QED':2})

GC_125 = Coupling(name = 'GC_125',
                  value = '(cw*CWL2*complex(0,1)*g1*gw**3*sw*v)/2.e6 - (CWL2*complex(0,1)*gw**4*sw**2*v)/2.e6',
                  order = {'NP':2,'QED':2})

GC_126 = Coupling(name = 'GC_126',
                  value = '-(cw**2*CWL2*complex(0,1)*gw**4*v**2)/4.e6 - (cw*CWL2*complex(0,1)*g1*gw**3*sw*v**2)/4.e6',
                  order = {'NP':2,'QED':1})

GC_127 = Coupling(name = 'GC_127',
                  value = '-(cw**2*CWL2*complex(0,1)*g1*gw**3*v**2)/4.e6 + (cw*CWL2*complex(0,1)*gw**4*sw*v**2)/2.e6 + (CWL2*complex(0,1)*g1*gw**3*sw**2*v**2)/4.e6',
                  order = {'NP':2,'QED':1})

GC_128 = Coupling(name = 'GC_128',
                  value = '(cw*CWL2*complex(0,1)*g1*gw**3*sw*v**2)/4.e6 - (CWL2*complex(0,1)*gw**4*sw**2*v**2)/4.e6',
                  order = {'NP':2,'QED':1})

GC_129 = Coupling(name = 'GC_129',
                  value = '-((complex(0,1)*yb)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_130 = Coupling(name = 'GC_130',
                  value = '-((complex(0,1)*yc)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_131 = Coupling(name = 'GC_131',
                  value = '-((complex(0,1)*ydo)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_132 = Coupling(name = 'GC_132',
                  value = '-((complex(0,1)*ye)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_133 = Coupling(name = 'GC_133',
                  value = '-((complex(0,1)*ym)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_134 = Coupling(name = 'GC_134',
                  value = '-((complex(0,1)*ys)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_135 = Coupling(name = 'GC_135',
                  value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_136 = Coupling(name = 'GC_136',
                  value = '-((complex(0,1)*ytau)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_137 = Coupling(name = 'GC_137',
                  value = '-((complex(0,1)*yup)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_138 = Coupling(name = 'GC_138',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_139 = Coupling(name = 'GC_139',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_140 = Coupling(name = 'GC_140',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_141 = Coupling(name = 'GC_141',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_142 = Coupling(name = 'GC_142',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_143 = Coupling(name = 'GC_143',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_144 = Coupling(name = 'GC_144',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_145 = Coupling(name = 'GC_145',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_146 = Coupling(name = 'GC_146',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

