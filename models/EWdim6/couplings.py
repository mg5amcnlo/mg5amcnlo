# This file was automatically created by FeynRules $Revision: 1167 $
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Fri 4 May 2012 16:30:01


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
                 value = '-6*complex(0,1)*lam',
                 order = {'QED':2})

GC_11 = Coupling(name = 'GC_11',
                 value = '(C3phiq*complex(0,1)*gw*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_12 = Coupling(name = 'GC_12',
                 value = '(CPW*cw*complex(0,1)*gw)/(2.*Lambda**2)',
                 order = {'NP':2})

GC_13 = Coupling(name = 'GC_13',
                 value = '-(CPW*cw**2*complex(0,1)*g1*gw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_14 = Coupling(name = 'GC_14',
                 value = '-(CPW*complex(0,1)*gw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_15 = Coupling(name = 'GC_15',
                 value = '(CPW*cw*complex(0,1)*g1*gw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '-(CPW*cw*complex(0,1)*gw**3)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_17 = Coupling(name = 'GC_17',
                 value = '(CPW*cw*complex(0,1)*gw**3)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_18 = Coupling(name = 'GC_18',
                 value = '(3*cw*CWWW*complex(0,1)*gw**3)/(2.*Lambda**2)',
                 order = {'NP':2})

GC_19 = Coupling(name = 'GC_19',
                 value = '-((CPW*complex(0,1)*gw**4)/Lambda**2)',
                 order = {'NP':2,'QED':3})

GC_20 = Coupling(name = 'GC_20',
                 value = '(CPW*cw**2*complex(0,1)*gw**4)/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_21 = Coupling(name = 'GC_21',
                 value = '(-3*CWWW*complex(0,1)*gw**4)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_22 = Coupling(name = 'GC_22',
                 value = '(3*cw**2*CWWW*complex(0,1)*gw**4)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_23 = Coupling(name = 'GC_23',
                 value = '(-2*CPW*cw*complex(0,1)*gw**5)/Lambda**2',
                 order = {'NP':2,'QED':4})

GC_24 = Coupling(name = 'GC_24',
                 value = '(-2*CPW*cw**3*complex(0,1)*gw**5)/Lambda**2',
                 order = {'NP':2,'QED':4})

GC_25 = Coupling(name = 'GC_25',
                 value = '(-3*cw*CWWW*complex(0,1)*gw**5)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_26 = Coupling(name = 'GC_26',
                 value = '(-3*cw**3*CWWW*complex(0,1)*gw**5)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_27 = Coupling(name = 'GC_27',
                 value = '(6*cw**2*CWWW*complex(0,1)*gw**6)/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_28 = Coupling(name = 'GC_28',
                 value = '-(cw*complex(0,1)*g1*gw**2*HB)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_29 = Coupling(name = 'GC_29',
                 value = '(complex(0,1)*gw**2*HW)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_30 = Coupling(name = 'GC_30',
                 value = '-(cw*complex(0,1)*g1*gw**2*HW)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_31 = Coupling(name = 'GC_31',
                 value = '-(cw*complex(0,1)*gw**3*HW)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_32 = Coupling(name = 'GC_32',
                 value = '(complex(0,1)*gw**4*HW)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':3})

GC_33 = Coupling(name = 'GC_33',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_34 = Coupling(name = 'GC_34',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_35 = Coupling(name = 'GC_35',
                 value = '(CKM1x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = '(CKM1x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = '(CKM1x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_38 = Coupling(name = 'GC_38',
                 value = '(CKM2x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '(CKM2x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(CKM2x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_41 = Coupling(name = 'GC_41',
                 value = '(CKM3x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '(CKM3x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '(CKM3x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_44 = Coupling(name = 'GC_44',
                 value = '-(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_45 = Coupling(name = 'GC_45',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_46 = Coupling(name = 'GC_46',
                 value = '-(ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_47 = Coupling(name = 'GC_47',
                 value = '(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_48 = Coupling(name = 'GC_48',
                 value = 'complex(0,1)*gw*sw',
                 order = {'QED':1})

GC_49 = Coupling(name = 'GC_49',
                 value = '-2*cw*complex(0,1)*gw**2*sw',
                 order = {'QED':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '(CPW*complex(0,1)*gw*sw)/(2.*Lambda**2)',
                 order = {'NP':2})

GC_51 = Coupling(name = 'GC_51',
                 value = '-(CPW*cw*complex(0,1)*gw**2*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '-(CPW*complex(0,1)*g1*gw**2*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_53 = Coupling(name = 'GC_53',
                 value = '-(CPW*complex(0,1)*gw**3*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_54 = Coupling(name = 'GC_54',
                 value = '(CPW*complex(0,1)*gw**3*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_55 = Coupling(name = 'GC_55',
                 value = '(3*CWWW*complex(0,1)*gw**3*sw)/(2.*Lambda**2)',
                 order = {'NP':2})

GC_56 = Coupling(name = 'GC_56',
                 value = '(CPW*cw*complex(0,1)*gw**4*sw)/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_57 = Coupling(name = 'GC_57',
                 value = '(-3*cw*CWWW*complex(0,1)*gw**4*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_58 = Coupling(name = 'GC_58',
                 value = '(2*CPW*complex(0,1)*gw**5*sw)/Lambda**2',
                 order = {'NP':2,'QED':4})

GC_59 = Coupling(name = 'GC_59',
                 value = '(-2*CPW*cw**2*complex(0,1)*gw**5*sw)/Lambda**2',
                 order = {'NP':2,'QED':4})

GC_60 = Coupling(name = 'GC_60',
                 value = '(-3*CWWW*complex(0,1)*gw**5*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_61 = Coupling(name = 'GC_61',
                 value = '(-3*cw**2*CWWW*complex(0,1)*gw**5*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_62 = Coupling(name = 'GC_62',
                 value = '(6*cw*CWWW*complex(0,1)*gw**6*sw)/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_63 = Coupling(name = 'GC_63',
                 value = '(complex(0,1)*g1*gw**2*HB*sw)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_64 = Coupling(name = 'GC_64',
                 value = '-(complex(0,1)*g1*gw**2*HW*sw)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_65 = Coupling(name = 'GC_65',
                 value = '-(complex(0,1)*gw**3*HW*sw)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_66 = Coupling(name = 'GC_66',
                 value = 'complex(0,1)*gw**2*sw**2',
                 order = {'QED':2})

GC_67 = Coupling(name = 'GC_67',
                 value = '-(CPW*complex(0,1)*g1*gw*sw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_68 = Coupling(name = 'GC_68',
                 value = '(CPW*complex(0,1)*gw**4*sw**2)/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_69 = Coupling(name = 'GC_69',
                 value = '(3*CWWW*complex(0,1)*gw**4*sw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_70 = Coupling(name = 'GC_70',
                 value = '(-2*CPW*cw*complex(0,1)*gw**5*sw**2)/Lambda**2',
                 order = {'NP':2,'QED':4})

GC_71 = Coupling(name = 'GC_71',
                 value = '(-3*cw*CWWW*complex(0,1)*gw**5*sw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_72 = Coupling(name = 'GC_72',
                 value = '(6*CWWW*complex(0,1)*gw**6*sw**2)/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_73 = Coupling(name = 'GC_73',
                 value = '(-2*CPW*complex(0,1)*gw**5*sw**3)/Lambda**2',
                 order = {'NP':2,'QED':4})

GC_74 = Coupling(name = 'GC_74',
                 value = '(-3*CWWW*complex(0,1)*gw**5*sw**3)/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_75 = Coupling(name = 'GC_75',
                 value = '(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_76 = Coupling(name = 'GC_76',
                 value = '-((C3phiq*cw*complex(0,1)*gw)/Lambda**2) - (Cphid*cw*complex(0,1)*gw)/Lambda**2 - (Cphiq*cw*complex(0,1)*gw)/Lambda**2 - (C3phiq*complex(0,1)*g1*sw)/Lambda**2 - (Cphid*complex(0,1)*g1*sw)/Lambda**2 - (Cphiq*complex(0,1)*g1*sw)/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_77 = Coupling(name = 'GC_77',
                 value = '(C3phiq*cw*complex(0,1)*gw)/Lambda**2 - (Cphiq*cw*complex(0,1)*gw)/Lambda**2 - (Cphiu*cw*complex(0,1)*gw)/Lambda**2 + (C3phiq*complex(0,1)*g1*sw)/Lambda**2 - (Cphiq*complex(0,1)*g1*sw)/Lambda**2 - (Cphiu*complex(0,1)*g1*sw)/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_78 = Coupling(name = 'GC_78',
                 value = '(C3phiq*cw*complex(0,1)*g1)/Lambda**2 + (Cphid*cw*complex(0,1)*g1)/Lambda**2 + (Cphiq*cw*complex(0,1)*g1)/Lambda**2 - (C3phiq*complex(0,1)*gw*sw)/Lambda**2 - (Cphid*complex(0,1)*gw*sw)/Lambda**2 - (Cphiq*complex(0,1)*gw*sw)/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_79 = Coupling(name = 'GC_79',
                 value = '-((C3phiq*cw*complex(0,1)*g1)/Lambda**2) + (Cphiq*cw*complex(0,1)*g1)/Lambda**2 + (Cphiu*cw*complex(0,1)*g1)/Lambda**2 + (C3phiq*complex(0,1)*gw*sw)/Lambda**2 - (Cphiq*complex(0,1)*gw*sw)/Lambda**2 - (Cphiu*complex(0,1)*gw*sw)/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_80 = Coupling(name = 'GC_80',
                 value = '-(CPW*cw**2*complex(0,1)*gw**2)/(2.*Lambda**2) - (CPW*cw*complex(0,1)*g1*gw*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_81 = Coupling(name = 'GC_81',
                 value = '-(cw*complex(0,1)*g1**2*HB*sw)/(4.*Lambda**2) + (cw*complex(0,1)*gw**2*HW*sw)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_82 = Coupling(name = 'GC_82',
                 value = '-(cw**2*complex(0,1)*gw**4*HW)/(2.*Lambda**2) - (cw*complex(0,1)*g1*gw**3*HW*sw)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':3})

GC_83 = Coupling(name = 'GC_83',
                 value = 'ee**2*complex(0,1) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_84 = Coupling(name = 'GC_84',
                 value = '(CPW*cw*complex(0,1)*g1*gw*sw)/(2.*Lambda**2) - (CPW*complex(0,1)*gw**2*sw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_85 = Coupling(name = 'GC_85',
                 value = '(cw**2*complex(0,1)*gw**2*HW)/(4.*Lambda**2) + (cw*complex(0,1)*g1*gw*HB*sw)/(4.*Lambda**2) + (cw*complex(0,1)*g1*gw*HW*sw)/(4.*Lambda**2) + (complex(0,1)*g1**2*HB*sw**2)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_86 = Coupling(name = 'GC_86',
                 value = '-(cw**2*complex(0,1)*g1*gw*HW)/(4.*Lambda**2) + (complex(0,1)*g1*gw*HB*sw**2)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_87 = Coupling(name = 'GC_87',
                 value = '-(cw**2*complex(0,1)*g1*gw*HB)/(4.*Lambda**2) + (complex(0,1)*g1*gw*HW*sw**2)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_88 = Coupling(name = 'GC_88',
                 value = '(cw**2*complex(0,1)*g1**2*HB)/(4.*Lambda**2) - (cw*complex(0,1)*g1*gw*HB*sw)/(4.*Lambda**2) - (cw*complex(0,1)*g1*gw*HW*sw)/(4.*Lambda**2) + (complex(0,1)*gw**2*HW*sw**2)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_89 = Coupling(name = 'GC_89',
                 value = '-(cw**2*complex(0,1)*g1*gw**3*HW)/(2.*Lambda**2) + (cw*complex(0,1)*gw**4*HW*sw)/Lambda**2 + (complex(0,1)*g1*gw**3*HW*sw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':3})

GC_90 = Coupling(name = 'GC_90',
                 value = '(cw*complex(0,1)*g1*gw**3*HW*sw)/(2.*Lambda**2) - (complex(0,1)*gw**4*HW*sw**2)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':3})

GC_91 = Coupling(name = 'GC_91',
                 value = '-6*complex(0,1)*lam*v',
                 order = {'QED':1})

GC_92 = Coupling(name = 'GC_92',
                 value = '(C3phiq*complex(0,1)*gw*v*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_93 = Coupling(name = 'GC_93',
                 value = '-(CPW*cw**2*complex(0,1)*g1*gw*v)/(2.*Lambda**2)',
                 order = {'NP':2})

GC_94 = Coupling(name = 'GC_94',
                 value = '-(CPW*complex(0,1)*gw**2*v)/(2.*Lambda**2)',
                 order = {'NP':2})

GC_95 = Coupling(name = 'GC_95',
                 value = '(CPW*cw*complex(0,1)*g1*gw**2*v)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_96 = Coupling(name = 'GC_96',
                 value = '-(CPW*cw*complex(0,1)*gw**3*v)/(2.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '-(cw*complex(0,1)*g1*gw**2*HB*v)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_98 = Coupling(name = 'GC_98',
                 value = '(complex(0,1)*gw**2*HW*v)/(4.*Lambda**2)',
                 order = {'NP':2})

GC_99 = Coupling(name = 'GC_99',
                 value = '-(cw*complex(0,1)*g1*gw**2*HW*v)/(4.*Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_100 = Coupling(name = 'GC_100',
                  value = '-(cw*complex(0,1)*gw**3*HW*v)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_101 = Coupling(name = 'GC_101',
                  value = '(complex(0,1)*gw**4*HW*v)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_102 = Coupling(name = 'GC_102',
                  value = '(ee**2*complex(0,1)*v)/(2.*sw**2)',
                  order = {'QED':1})

GC_103 = Coupling(name = 'GC_103',
                  value = '-(CPW*cw*complex(0,1)*gw**2*sw*v)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_104 = Coupling(name = 'GC_104',
                  value = '-(CPW*complex(0,1)*g1*gw**2*sw*v)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_105 = Coupling(name = 'GC_105',
                  value = '-(CPW*complex(0,1)*gw**3*sw*v)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_106 = Coupling(name = 'GC_106',
                  value = '(complex(0,1)*g1*gw**2*HB*sw*v)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_107 = Coupling(name = 'GC_107',
                  value = '-(complex(0,1)*g1*gw**2*HW*sw*v)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_108 = Coupling(name = 'GC_108',
                  value = '-(complex(0,1)*gw**3*HW*sw*v)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_109 = Coupling(name = 'GC_109',
                  value = '-(CPW*complex(0,1)*g1*gw*sw**2*v)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_110 = Coupling(name = 'GC_110',
                  value = '(C3phiq*complex(0,1)*gw*v**2)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2})

GC_111 = Coupling(name = 'GC_111',
                  value = '(CPW*cw*complex(0,1)*g1*gw**2*v**2)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_112 = Coupling(name = 'GC_112',
                  value = '-(CPW*cw*complex(0,1)*gw**3*v**2)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_113 = Coupling(name = 'GC_113',
                  value = '-(cw*complex(0,1)*g1*gw**2*HB*v**2)/(8.*Lambda**2)',
                  order = {'NP':2})

GC_114 = Coupling(name = 'GC_114',
                  value = '-(cw*complex(0,1)*g1*gw**2*HW*v**2)/(8.*Lambda**2)',
                  order = {'NP':2})

GC_115 = Coupling(name = 'GC_115',
                  value = '-(cw*complex(0,1)*gw**3*HW*v**2)/(8.*Lambda**2)',
                  order = {'NP':2})

GC_116 = Coupling(name = 'GC_116',
                  value = '(complex(0,1)*gw**4*HW*v**2)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_117 = Coupling(name = 'GC_117',
                  value = '-(CPW*complex(0,1)*g1*gw**2*sw*v**2)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_118 = Coupling(name = 'GC_118',
                  value = '-(CPW*complex(0,1)*gw**3*sw*v**2)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_119 = Coupling(name = 'GC_119',
                  value = '(complex(0,1)*g1*gw**2*HB*sw*v**2)/(8.*Lambda**2)',
                  order = {'NP':2})

GC_120 = Coupling(name = 'GC_120',
                  value = '-(complex(0,1)*g1*gw**2*HW*sw*v**2)/(8.*Lambda**2)',
                  order = {'NP':2})

GC_121 = Coupling(name = 'GC_121',
                  value = '-(complex(0,1)*gw**3*HW*sw*v**2)/(8.*Lambda**2)',
                  order = {'NP':2})

GC_122 = Coupling(name = 'GC_122',
                  value = '-((C3phiq*cw*complex(0,1)*gw*v)/Lambda**2) - (Cphid*cw*complex(0,1)*gw*v)/Lambda**2 - (Cphiq*cw*complex(0,1)*gw*v)/Lambda**2 - (C3phiq*complex(0,1)*g1*sw*v)/Lambda**2 - (Cphid*complex(0,1)*g1*sw*v)/Lambda**2 - (Cphiq*complex(0,1)*g1*sw*v)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_123 = Coupling(name = 'GC_123',
                  value = '(C3phiq*cw*complex(0,1)*gw*v)/Lambda**2 - (Cphiq*cw*complex(0,1)*gw*v)/Lambda**2 - (Cphiu*cw*complex(0,1)*gw*v)/Lambda**2 + (C3phiq*complex(0,1)*g1*sw*v)/Lambda**2 - (Cphiq*complex(0,1)*g1*sw*v)/Lambda**2 - (Cphiu*complex(0,1)*g1*sw*v)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_124 = Coupling(name = 'GC_124',
                  value = '(C3phiq*cw*complex(0,1)*g1*v)/Lambda**2 + (Cphid*cw*complex(0,1)*g1*v)/Lambda**2 + (Cphiq*cw*complex(0,1)*g1*v)/Lambda**2 - (C3phiq*complex(0,1)*gw*sw*v)/Lambda**2 - (Cphid*complex(0,1)*gw*sw*v)/Lambda**2 - (Cphiq*complex(0,1)*gw*sw*v)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_125 = Coupling(name = 'GC_125',
                  value = '-((C3phiq*cw*complex(0,1)*g1*v)/Lambda**2) + (Cphiq*cw*complex(0,1)*g1*v)/Lambda**2 + (Cphiu*cw*complex(0,1)*g1*v)/Lambda**2 + (C3phiq*complex(0,1)*gw*sw*v)/Lambda**2 - (Cphiq*complex(0,1)*gw*sw*v)/Lambda**2 - (Cphiu*complex(0,1)*gw*sw*v)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_126 = Coupling(name = 'GC_126',
                  value = '-(CPW*cw**2*complex(0,1)*gw**2*v)/(2.*Lambda**2) - (CPW*cw*complex(0,1)*g1*gw*sw*v)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_127 = Coupling(name = 'GC_127',
                  value = '-(cw*complex(0,1)*g1**2*HB*sw*v)/(4.*Lambda**2) + (cw*complex(0,1)*gw**2*HW*sw*v)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_128 = Coupling(name = 'GC_128',
                  value = '-(cw**2*complex(0,1)*gw**4*HW*v)/(2.*Lambda**2) - (cw*complex(0,1)*g1*gw**3*HW*sw*v)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_129 = Coupling(name = 'GC_129',
                  value = 'ee**2*complex(0,1)*v + (cw**2*ee**2*complex(0,1)*v)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*v)/(2.*cw**2)',
                  order = {'QED':1})

GC_130 = Coupling(name = 'GC_130',
                  value = '(CPW*cw*complex(0,1)*g1*gw*sw*v)/(2.*Lambda**2) - (CPW*complex(0,1)*gw**2*sw**2*v)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_131 = Coupling(name = 'GC_131',
                  value = '(cw**2*complex(0,1)*gw**2*HW*v)/(4.*Lambda**2) + (cw*complex(0,1)*g1*gw*HB*sw*v)/(4.*Lambda**2) + (cw*complex(0,1)*g1*gw*HW*sw*v)/(4.*Lambda**2) + (complex(0,1)*g1**2*HB*sw**2*v)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_132 = Coupling(name = 'GC_132',
                  value = '-(cw**2*complex(0,1)*g1*gw*HW*v)/(4.*Lambda**2) + (complex(0,1)*g1*gw*HB*sw**2*v)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_133 = Coupling(name = 'GC_133',
                  value = '-(cw**2*complex(0,1)*g1*gw*HB*v)/(4.*Lambda**2) + (complex(0,1)*g1*gw*HW*sw**2*v)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_134 = Coupling(name = 'GC_134',
                  value = '(cw**2*complex(0,1)*g1**2*HB*v)/(4.*Lambda**2) - (cw*complex(0,1)*g1*gw*HB*sw*v)/(4.*Lambda**2) - (cw*complex(0,1)*g1*gw*HW*sw*v)/(4.*Lambda**2) + (complex(0,1)*gw**2*HW*sw**2*v)/(4.*Lambda**2)',
                  order = {'NP':2})

GC_135 = Coupling(name = 'GC_135',
                  value = '-(cw**2*complex(0,1)*g1*gw**3*HW*v)/(2.*Lambda**2) + (cw*complex(0,1)*gw**4*HW*sw*v)/Lambda**2 + (complex(0,1)*g1*gw**3*HW*sw**2*v)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_136 = Coupling(name = 'GC_136',
                  value = '(cw*complex(0,1)*g1*gw**3*HW*sw*v)/(2.*Lambda**2) - (complex(0,1)*gw**4*HW*sw**2*v)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_137 = Coupling(name = 'GC_137',
                  value = '-(C3phiq*cw*complex(0,1)*gw*v**2)/(2.*Lambda**2) - (Cphid*cw*complex(0,1)*gw*v**2)/(2.*Lambda**2) - (Cphiq*cw*complex(0,1)*gw*v**2)/(2.*Lambda**2) - (C3phiq*complex(0,1)*g1*sw*v**2)/(2.*Lambda**2) - (Cphid*complex(0,1)*g1*sw*v**2)/(2.*Lambda**2) - (Cphiq*complex(0,1)*g1*sw*v**2)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_138 = Coupling(name = 'GC_138',
                  value = '(C3phiq*cw*complex(0,1)*gw*v**2)/(2.*Lambda**2) - (Cphiq*cw*complex(0,1)*gw*v**2)/(2.*Lambda**2) - (Cphiu*cw*complex(0,1)*gw*v**2)/(2.*Lambda**2) + (C3phiq*complex(0,1)*g1*sw*v**2)/(2.*Lambda**2) - (Cphiq*complex(0,1)*g1*sw*v**2)/(2.*Lambda**2) - (Cphiu*complex(0,1)*g1*sw*v**2)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_139 = Coupling(name = 'GC_139',
                  value = '(C3phiq*cw*complex(0,1)*g1*v**2)/(2.*Lambda**2) + (Cphid*cw*complex(0,1)*g1*v**2)/(2.*Lambda**2) + (Cphiq*cw*complex(0,1)*g1*v**2)/(2.*Lambda**2) - (C3phiq*complex(0,1)*gw*sw*v**2)/(2.*Lambda**2) - (Cphid*complex(0,1)*gw*sw*v**2)/(2.*Lambda**2) - (Cphiq*complex(0,1)*gw*sw*v**2)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_140 = Coupling(name = 'GC_140',
                  value = '-(C3phiq*cw*complex(0,1)*g1*v**2)/(2.*Lambda**2) + (Cphiq*cw*complex(0,1)*g1*v**2)/(2.*Lambda**2) + (Cphiu*cw*complex(0,1)*g1*v**2)/(2.*Lambda**2) + (C3phiq*complex(0,1)*gw*sw*v**2)/(2.*Lambda**2) - (Cphiq*complex(0,1)*gw*sw*v**2)/(2.*Lambda**2) - (Cphiu*complex(0,1)*gw*sw*v**2)/(2.*Lambda**2)',
                  order = {'NP':2})

GC_141 = Coupling(name = 'GC_141',
                  value = '-(cw**2*complex(0,1)*gw**4*HW*v**2)/(4.*Lambda**2) - (cw*complex(0,1)*g1*gw**3*HW*sw*v**2)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_142 = Coupling(name = 'GC_142',
                  value = '-(cw**2*complex(0,1)*g1*gw**3*HW*v**2)/(4.*Lambda**2) + (cw*complex(0,1)*gw**4*HW*sw*v**2)/(2.*Lambda**2) + (complex(0,1)*g1*gw**3*HW*sw**2*v**2)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_143 = Coupling(name = 'GC_143',
                  value = '(cw*complex(0,1)*g1*gw**3*HW*sw*v**2)/(4.*Lambda**2) - (complex(0,1)*gw**4*HW*sw**2*v**2)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_144 = Coupling(name = 'GC_144',
                  value = '-((complex(0,1)*yb)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_145 = Coupling(name = 'GC_145',
                  value = '-((complex(0,1)*yc)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_146 = Coupling(name = 'GC_146',
                  value = '-((complex(0,1)*ydo)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_147 = Coupling(name = 'GC_147',
                  value = '-((complex(0,1)*ye)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_148 = Coupling(name = 'GC_148',
                  value = '-((complex(0,1)*ym)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_149 = Coupling(name = 'GC_149',
                  value = '-((complex(0,1)*ys)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_150 = Coupling(name = 'GC_150',
                  value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_151 = Coupling(name = 'GC_151',
                  value = '-((complex(0,1)*ytau)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_152 = Coupling(name = 'GC_152',
                  value = '-((complex(0,1)*yup)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_153 = Coupling(name = 'GC_153',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_154 = Coupling(name = 'GC_154',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_155 = Coupling(name = 'GC_155',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_156 = Coupling(name = 'GC_156',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_157 = Coupling(name = 'GC_157',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_158 = Coupling(name = 'GC_158',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_159 = Coupling(name = 'GC_159',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_160 = Coupling(name = 'GC_160',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_161 = Coupling(name = 'GC_161',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

