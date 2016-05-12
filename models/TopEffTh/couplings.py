# This file was automatically created by FeynRules 2.4.1
# Mathematica version: 10.1.0  for Mac OS X x86 (64-bit) (March 24, 2015)
# Date: Thu 18 Jun 2015 16:26:34


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot



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
                value = 'ee*complex(0,1)',
                order = {'QED':1})

GC_5 = Coupling(name = 'GC_5',
                value = 'ee**2*complex(0,1)',
                order = {'QED':2})

GC_6 = Coupling(name = 'GC_6',
                value = '-G',
                order = {'QCD':1})

GC_7 = Coupling(name = 'GC_7',
                value = 'complex(0,1)*G',
                order = {'QCD':1})

GC_8 = Coupling(name = 'GC_8',
                value = 'complex(0,1)*G**2',
                order = {'QCD':2})

GC_9 = Coupling(name = 'GC_9',
                value = '-6*complex(0,1)*lam',
                order = {'QED':2})

GC_10 = Coupling(name = 'GC_10',
                 value = '(C81qq*complex(0,1))/Lambda**2 - (C83qq*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_11 = Coupling(name = 'GC_11',
                 value = '(C81qq*complex(0,1))/Lambda**2 + (C83qq*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_12 = Coupling(name = 'GC_12',
                 value = '-((C13qq*complex(0,1))/Lambda**2)',
                 order = {'NP':2})

GC_13 = Coupling(name = 'GC_13',
                 value = '(C13qq*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_14 = Coupling(name = 'GC_14',
                 value = '(2*C13qq*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_15 = Coupling(name = 'GC_15',
                 value = '(C1qd*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '(C1qt*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_17 = Coupling(name = 'GC_17',
                 value = '(C1qu*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_18 = Coupling(name = 'GC_18',
                 value = '-((C3phiq*complex(0,1))/Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_19 = Coupling(name = 'GC_19',
                 value = '(C3phiq*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_20 = Coupling(name = 'GC_20',
                 value = '-((C3phiq*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_21 = Coupling(name = 'GC_21',
                 value = '(C3phiq*complex(0,1)*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_22 = Coupling(name = 'GC_22',
                 value = '(C3phiq*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_23 = Coupling(name = 'GC_23',
                 value = '(C81qq*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_24 = Coupling(name = 'GC_24',
                 value = '-((C83qq*complex(0,1))/Lambda**2)',
                 order = {'NP':2})

GC_25 = Coupling(name = 'GC_25',
                 value = '(2*C83qq*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_26 = Coupling(name = 'GC_26',
                 value = '(C8dt*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_27 = Coupling(name = 'GC_27',
                 value = '(C8ut*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_28 = Coupling(name = 'GC_28',
                 value = '(-6*CG)/Lambda**2',
                 order = {'NP':2})

GC_29 = Coupling(name = 'GC_29',
                 value = '(2*CphiG*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_30 = Coupling(name = 'GC_30',
                 value = '(2*CtG)/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_31 = Coupling(name = 'GC_31',
                 value = '(CtG*complex(0,1)*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_32 = Coupling(name = 'GC_32',
                 value = '(CtG*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_33 = Coupling(name = 'GC_33',
                 value = '(CtW*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_34 = Coupling(name = 'GC_34',
                 value = 'CtW/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_35 = Coupling(name = 'GC_35',
                 value = '(CtW*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = '-((CtW*cw)/Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = '(CtW*cw*complex(0,1))/(Lambda**2*cmath.sqrt(2))',
                 order = {'NP':2,'QED':1})

GC_38 = Coupling(name = 'GC_38',
                 value = '(CtW*cw)/(Lambda**2*cmath.sqrt(2))',
                 order = {'NP':2,'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '-((C3phiq*ee*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_40 = Coupling(name = 'GC_40',
                 value = '(C3phiq*ee*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_41 = Coupling(name = 'GC_41',
                 value = '-((CtW*ee)/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_42 = Coupling(name = 'GC_42',
                 value = '-((CtW*ee*complex(0,1))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_43 = Coupling(name = 'GC_43',
                 value = '(CtW*ee*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_44 = Coupling(name = 'GC_44',
                 value = '(6*CG*complex(0,1)*G)/Lambda**2',
                 order = {'NP':2,'QCD':1})

GC_45 = Coupling(name = 'GC_45',
                 value = '(2*CphiG*G)/Lambda**2',
                 order = {'NP':2,'QCD':1,'QED':1})

GC_46 = Coupling(name = 'GC_46',
                 value = '(2*CtG*complex(0,1)*G)/Lambda**2',
                 order = {'NP':2,'QCD':1,'QED':1})

GC_47 = Coupling(name = 'GC_47',
                 value = '-((CtG*G*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QCD':1,'QED':1})

GC_48 = Coupling(name = 'GC_48',
                 value = '(CtG*complex(0,1)*G*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QCD':1,'QED':1})

GC_49 = Coupling(name = 'GC_49',
                 value = '(-3*CG*G**2)/Lambda**2',
                 order = {'NP':2,'QCD':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '(3*CG*G**2)/Lambda**2',
                 order = {'NP':2,'QCD':2})

GC_51 = Coupling(name = 'GC_51',
                 value = '(-2*CphiG*complex(0,1)*G**2)/Lambda**2',
                 order = {'NP':2,'QCD':2,'QED':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '-((CG*complex(0,1)*G**3)/Lambda**2)',
                 order = {'NP':2,'QCD':3})

GC_53 = Coupling(name = 'GC_53',
                 value = '(CG*complex(0,1)*G**3)/Lambda**2',
                 order = {'NP':2,'QCD':3})

GC_54 = Coupling(name = 'GC_54',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_55 = Coupling(name = 'GC_55',
                 value = '-((ee**2*complex(0,1))/sw**2)',
                 order = {'QED':2})

GC_56 = Coupling(name = 'GC_56',
                 value = '(cw**2*ee**2*complex(0,1))/sw**2',
                 order = {'QED':2})

GC_57 = Coupling(name = 'GC_57',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_58 = Coupling(name = 'GC_58',
                 value = '-(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_59 = Coupling(name = 'GC_59',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_60 = Coupling(name = 'GC_60',
                 value = '(cw*ee*complex(0,1))/sw',
                 order = {'QED':1})

GC_61 = Coupling(name = 'GC_61',
                 value = '(-2*cw*ee**2*complex(0,1))/sw',
                 order = {'QED':2})

GC_62 = Coupling(name = 'GC_62',
                 value = '(C3phiq*ee*complex(0,1)*cmath.sqrt(2))/(Lambda**2*sw)',
                 order = {'NP':2,'QED':2})

GC_63 = Coupling(name = 'GC_63',
                 value = '-((CtW*ee)/(Lambda**2*sw))',
                 order = {'NP':2,'QED':2})

GC_64 = Coupling(name = 'GC_64',
                 value = '(CtW*ee*complex(0,1))/(Lambda**2*sw*cmath.sqrt(2))',
                 order = {'NP':2,'QED':2})

GC_65 = Coupling(name = 'GC_65',
                 value = '(CtW*ee)/(Lambda**2*sw*cmath.sqrt(2))',
                 order = {'NP':2,'QED':2})

GC_66 = Coupling(name = 'GC_66',
                 value = '(CtW*cw*ee*complex(0,1))/(Lambda**2*sw)',
                 order = {'NP':2,'QED':2})

GC_67 = Coupling(name = 'GC_67',
                 value = '(CtW*cw*ee)/(Lambda**2*sw)',
                 order = {'NP':2,'QED':2})

GC_68 = Coupling(name = 'GC_68',
                 value = '-((CtW*cw*ee*cmath.sqrt(2))/(Lambda**2*sw))',
                 order = {'NP':2,'QED':2})

GC_69 = Coupling(name = 'GC_69',
                 value = '-(ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_70 = Coupling(name = 'GC_70',
                 value = '(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_71 = Coupling(name = 'GC_71',
                 value = '-((CtW*sw)/Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_72 = Coupling(name = 'GC_72',
                 value = '(CtW*complex(0,1)*sw)/(Lambda**2*cmath.sqrt(2))',
                 order = {'NP':2,'QED':1})

GC_73 = Coupling(name = 'GC_73',
                 value = '(CtW*sw)/(Lambda**2*cmath.sqrt(2))',
                 order = {'NP':2,'QED':1})

GC_74 = Coupling(name = 'GC_74',
                 value = '(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_75 = Coupling(name = 'GC_75',
                 value = 'ee**2*complex(0,1) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_76 = Coupling(name = 'GC_76',
                 value = '-6*complex(0,1)*lam*vev',
                 order = {'QED':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '-((C3phiq*vev*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2})

GC_78 = Coupling(name = 'GC_78',
                 value = '(2*CphiG*complex(0,1)*vev)/Lambda**2',
                 order = {'NP':2})

GC_79 = Coupling(name = 'GC_79',
                 value = '(CtG*complex(0,1)*vev*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2})

GC_80 = Coupling(name = 'GC_80',
                 value = '(CtW*complex(0,1)*vev)/Lambda**2',
                 order = {'NP':2})

GC_81 = Coupling(name = 'GC_81',
                 value = '(CtW*cw*complex(0,1)*vev)/(Lambda**2*cmath.sqrt(2))',
                 order = {'NP':2})

GC_82 = Coupling(name = 'GC_82',
                 value = '(C3phiq*ee*vev*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_83 = Coupling(name = 'GC_83',
                 value = '-((CtW*ee*complex(0,1)*vev)/Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_84 = Coupling(name = 'GC_84',
                 value = '(2*CphiG*G*vev)/Lambda**2',
                 order = {'NP':2,'QCD':1})

GC_85 = Coupling(name = 'GC_85',
                 value = '-((CtG*G*vev*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QCD':1})

GC_86 = Coupling(name = 'GC_86',
                 value = '(-2*CphiG*complex(0,1)*G**2*vev)/Lambda**2',
                 order = {'NP':2,'QCD':2})

GC_87 = Coupling(name = 'GC_87',
                 value = '(ee**2*complex(0,1)*vev)/(2.*sw**2)',
                 order = {'QED':1})

GC_88 = Coupling(name = 'GC_88',
                 value = '(C3phiq*ee*complex(0,1)*vev*cmath.sqrt(2))/(Lambda**2*sw)',
                 order = {'NP':2,'QED':1})

GC_89 = Coupling(name = 'GC_89',
                 value = '(CtW*ee*complex(0,1)*vev)/(Lambda**2*sw*cmath.sqrt(2))',
                 order = {'NP':2,'QED':1})

GC_90 = Coupling(name = 'GC_90',
                 value = '(CtW*cw*ee*complex(0,1)*vev)/(Lambda**2*sw)',
                 order = {'NP':2,'QED':1})

GC_91 = Coupling(name = 'GC_91',
                 value = '(CtW*complex(0,1)*sw*vev)/(Lambda**2*cmath.sqrt(2))',
                 order = {'NP':2})

GC_92 = Coupling(name = 'GC_92',
                 value = '(C3phiq*ee*complex(0,1)*vev**2)/(Lambda**2*sw*cmath.sqrt(2))',
                 order = {'NP':2})

GC_93 = Coupling(name = 'GC_93',
                 value = 'ee**2*complex(0,1)*vev + (cw**2*ee**2*complex(0,1)*vev)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*vev)/(2.*cw**2)',
                 order = {'QED':1})

GC_94 = Coupling(name = 'GC_94',
                 value = '-((complex(0,1)*yb)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_95 = Coupling(name = 'GC_95',
                 value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_96 = Coupling(name = 'GC_96',
                 value = '-((complex(0,1)*ytau)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '-((complex(0,1)*complexconjugate(C3phiq))/Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_98 = Coupling(name = 'GC_98',
                 value = '(complex(0,1)*complexconjugate(C3phiq))/Lambda**2',
                 order = {'NP':2,'QED':1})

GC_99 = Coupling(name = 'GC_99',
                 value = '-((complexconjugate(C3phiq)*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':1})

GC_100 = Coupling(name = 'GC_100',
                  value = '-((complex(0,1)*complexconjugate(C3phiq)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_101 = Coupling(name = 'GC_101',
                  value = '(complexconjugate(C3phiq)*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_102 = Coupling(name = 'GC_102',
                  value = '-((ee*complexconjugate(C3phiq)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_103 = Coupling(name = 'GC_103',
                  value = '-((ee*complex(0,1)*complexconjugate(C3phiq)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_104 = Coupling(name = 'GC_104',
                  value = '(ee*complex(0,1)*complexconjugate(C3phiq)*cmath.sqrt(2))/(Lambda**2*sw)',
                  order = {'NP':2,'QED':2})

GC_105 = Coupling(name = 'GC_105',
                  value = '-((vev*complexconjugate(C3phiq)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2})

GC_106 = Coupling(name = 'GC_106',
                  value = '-((ee*vev*complexconjugate(C3phiq)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_107 = Coupling(name = 'GC_107',
                  value = '(ee*complex(0,1)*vev*complexconjugate(C3phiq)*cmath.sqrt(2))/(Lambda**2*sw)',
                  order = {'NP':2,'QED':1})

GC_108 = Coupling(name = 'GC_108',
                  value = '(ee*complex(0,1)*vev**2*complexconjugate(C3phiq))/(Lambda**2*sw*cmath.sqrt(2))',
                  order = {'NP':2})

GC_109 = Coupling(name = 'GC_109',
                  value = '-C3phiq/(2.*Lambda**2) - complexconjugate(C3phiq)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_110 = Coupling(name = 'GC_110',
                  value = '(C3phiq*complex(0,1))/(2.*Lambda**2) - (complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_111 = Coupling(name = 'GC_111',
                  value = '-(C3phiq*complex(0,1))/(2.*Lambda**2) + (complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_112 = Coupling(name = 'GC_112',
                  value = 'C3phiq/(2.*Lambda**2) + complexconjugate(C3phiq)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_113 = Coupling(name = 'GC_113',
                  value = '-((C3phiq*ee*complex(0,1))/Lambda**2) - (ee*complex(0,1)*complexconjugate(C3phiq))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_114 = Coupling(name = 'GC_114',
                  value = '(C3phiq*ee*complex(0,1))/Lambda**2 + (ee*complex(0,1)*complexconjugate(C3phiq))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_115 = Coupling(name = 'GC_115',
                  value = '(C3phiq*ee)/(2.*Lambda**2*sw) - (ee*complexconjugate(C3phiq))/(2.*Lambda**2*sw)',
                  order = {'NP':2,'QED':2})

GC_116 = Coupling(name = 'GC_116',
                  value = '(C3phiq*ee*complex(0,1))/(2.*Lambda**2*sw) - (ee*complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2*sw)',
                  order = {'NP':2,'QED':2})

GC_117 = Coupling(name = 'GC_117',
                  value = '-(C3phiq*ee*complex(0,1))/(2.*Lambda**2*sw) + (ee*complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2*sw)',
                  order = {'NP':2,'QED':2})

GC_118 = Coupling(name = 'GC_118',
                  value = '-(C3phiq*ee)/(2.*Lambda**2*sw) + (ee*complexconjugate(C3phiq))/(2.*Lambda**2*sw)',
                  order = {'NP':2,'QED':2})

GC_119 = Coupling(name = 'GC_119',
                  value = '-(C3phiq*cw*ee*complex(0,1))/(2.*Lambda**2*sw) - (C3phiq*ee*complex(0,1)*sw)/(2.*cw*Lambda**2) - (cw*ee*complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2*sw) - (ee*complex(0,1)*sw*complexconjugate(C3phiq))/(2.*cw*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_120 = Coupling(name = 'GC_120',
                  value = '(C3phiq*cw*ee*complex(0,1))/(2.*Lambda**2*sw) - (C3phiq*ee*complex(0,1)*sw)/(2.*cw*Lambda**2) + (cw*ee*complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2*sw) - (ee*complex(0,1)*sw*complexconjugate(C3phiq))/(2.*cw*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_121 = Coupling(name = 'GC_121',
                  value = '-(C3phiq*cw*ee*complex(0,1))/(2.*Lambda**2*sw) + (C3phiq*ee*complex(0,1)*sw)/(2.*cw*Lambda**2) - (cw*ee*complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2*sw) + (ee*complex(0,1)*sw*complexconjugate(C3phiq))/(2.*cw*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_122 = Coupling(name = 'GC_122',
                  value = '(C3phiq*cw*ee*complex(0,1))/(2.*Lambda**2*sw) + (C3phiq*ee*complex(0,1)*sw)/(2.*cw*Lambda**2) + (cw*ee*complex(0,1)*complexconjugate(C3phiq))/(2.*Lambda**2*sw) + (ee*complex(0,1)*sw*complexconjugate(C3phiq))/(2.*cw*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_123 = Coupling(name = 'GC_123',
                  value = '(C3phiq*cw*ee)/(Lambda**2*sw*cmath.sqrt(2)) - (C3phiq*ee*sw)/(cw*Lambda**2*cmath.sqrt(2)) - (cw*ee*complexconjugate(C3phiq))/(Lambda**2*sw*cmath.sqrt(2)) - (ee*sw*complexconjugate(C3phiq))/(cw*Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_124 = Coupling(name = 'GC_124',
                  value = '(C3phiq*cw*ee*complex(0,1))/(Lambda**2*sw*cmath.sqrt(2)) + (C3phiq*ee*complex(0,1)*sw)/(cw*Lambda**2*cmath.sqrt(2)) - (cw*ee*complex(0,1)*complexconjugate(C3phiq))/(Lambda**2*sw*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(C3phiq))/(cw*Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_125 = Coupling(name = 'GC_125',
                  value = '-((C3phiq*cw*ee*complex(0,1))/(Lambda**2*sw*cmath.sqrt(2))) + (C3phiq*ee*complex(0,1)*sw)/(cw*Lambda**2*cmath.sqrt(2)) + (cw*ee*complex(0,1)*complexconjugate(C3phiq))/(Lambda**2*sw*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(C3phiq))/(cw*Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_126 = Coupling(name = 'GC_126',
                  value = '(C3phiq*cw*ee)/(Lambda**2*sw*cmath.sqrt(2)) + (C3phiq*ee*sw)/(cw*Lambda**2*cmath.sqrt(2)) - (cw*ee*complexconjugate(C3phiq))/(Lambda**2*sw*cmath.sqrt(2)) + (ee*sw*complexconjugate(C3phiq))/(cw*Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_127 = Coupling(name = 'GC_127',
                  value = '-(C3phiq*vev)/(2.*Lambda**2) - (vev*complexconjugate(C3phiq))/(2.*Lambda**2)',
                  order = {'NP':2})

GC_128 = Coupling(name = 'GC_128',
                  value = '(C3phiq*complex(0,1)*vev)/(2.*Lambda**2) - (complex(0,1)*vev*complexconjugate(C3phiq))/(2.*Lambda**2)',
                  order = {'NP':2})

GC_129 = Coupling(name = 'GC_129',
                  value = '-(C3phiq*complex(0,1)*vev)/(2.*Lambda**2) + (complex(0,1)*vev*complexconjugate(C3phiq))/(2.*Lambda**2)',
                  order = {'NP':2})

GC_130 = Coupling(name = 'GC_130',
                  value = '(C3phiq*vev)/(2.*Lambda**2) + (vev*complexconjugate(C3phiq))/(2.*Lambda**2)',
                  order = {'NP':2})

GC_131 = Coupling(name = 'GC_131',
                  value = '(C3phiq*ee*vev)/(2.*Lambda**2*sw) - (ee*vev*complexconjugate(C3phiq))/(2.*Lambda**2*sw)',
                  order = {'NP':2,'QED':1})

GC_132 = Coupling(name = 'GC_132',
                  value = '-(C3phiq*ee*vev)/(2.*Lambda**2*sw) + (ee*vev*complexconjugate(C3phiq))/(2.*Lambda**2*sw)',
                  order = {'NP':2,'QED':1})

GC_133 = Coupling(name = 'GC_133',
                  value = '-(C3phiq*cw*ee*complex(0,1)*vev)/(2.*Lambda**2*sw) - (C3phiq*ee*complex(0,1)*sw*vev)/(2.*cw*Lambda**2) - (cw*ee*complex(0,1)*vev*complexconjugate(C3phiq))/(2.*Lambda**2*sw) - (ee*complex(0,1)*sw*vev*complexconjugate(C3phiq))/(2.*cw*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_134 = Coupling(name = 'GC_134',
                  value = '(C3phiq*cw*ee*complex(0,1)*vev)/(2.*Lambda**2*sw) + (C3phiq*ee*complex(0,1)*sw*vev)/(2.*cw*Lambda**2) + (cw*ee*complex(0,1)*vev*complexconjugate(C3phiq))/(2.*Lambda**2*sw) + (ee*complex(0,1)*sw*vev*complexconjugate(C3phiq))/(2.*cw*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_135 = Coupling(name = 'GC_135',
                  value = '(C3phiq*cw*ee*vev)/(Lambda**2*sw*cmath.sqrt(2)) - (C3phiq*ee*sw*vev)/(cw*Lambda**2*cmath.sqrt(2)) - (cw*ee*vev*complexconjugate(C3phiq))/(Lambda**2*sw*cmath.sqrt(2)) - (ee*sw*vev*complexconjugate(C3phiq))/(cw*Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_136 = Coupling(name = 'GC_136',
                  value = '(C3phiq*cw*ee*vev)/(Lambda**2*sw*cmath.sqrt(2)) + (C3phiq*ee*sw*vev)/(cw*Lambda**2*cmath.sqrt(2)) - (cw*ee*vev*complexconjugate(C3phiq))/(Lambda**2*sw*cmath.sqrt(2)) + (ee*sw*vev*complexconjugate(C3phiq))/(cw*Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_137 = Coupling(name = 'GC_137',
                  value = '-(C3phiq*cw*ee*complex(0,1)*vev**2)/(4.*Lambda**2*sw) - (C3phiq*ee*complex(0,1)*sw*vev**2)/(4.*cw*Lambda**2) - (cw*ee*complex(0,1)*vev**2*complexconjugate(C3phiq))/(4.*Lambda**2*sw) - (ee*complex(0,1)*sw*vev**2*complexconjugate(C3phiq))/(4.*cw*Lambda**2)',
                  order = {'NP':2})

GC_138 = Coupling(name = 'GC_138',
                  value = '(C3phiq*cw*ee*complex(0,1)*vev**2)/(4.*Lambda**2*sw) + (C3phiq*ee*complex(0,1)*sw*vev**2)/(4.*cw*Lambda**2) + (cw*ee*complex(0,1)*vev**2*complexconjugate(C3phiq))/(4.*Lambda**2*sw) + (ee*complex(0,1)*sw*vev**2*complexconjugate(C3phiq))/(4.*cw*Lambda**2)',
                  order = {'NP':2})

GC_139 = Coupling(name = 'GC_139',
                  value = '(-2*complexconjugate(CtG))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_140 = Coupling(name = 'GC_140',
                  value = '-((complexconjugate(CtG)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_141 = Coupling(name = 'GC_141',
                  value = '(complex(0,1)*complexconjugate(CtG)*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_142 = Coupling(name = 'GC_142',
                  value = '(-2*complex(0,1)*G*complexconjugate(CtG))/Lambda**2',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_143 = Coupling(name = 'GC_143',
                  value = '-((G*complexconjugate(CtG)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_144 = Coupling(name = 'GC_144',
                  value = '-((complex(0,1)*G*complexconjugate(CtG)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_145 = Coupling(name = 'GC_145',
                  value = '(complex(0,1)*vev*complexconjugate(CtG)*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2})

GC_146 = Coupling(name = 'GC_146',
                  value = '-((G*vev*complexconjugate(CtG)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QCD':1})

GC_147 = Coupling(name = 'GC_147',
                  value = '-(complexconjugate(CtW)/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_148 = Coupling(name = 'GC_148',
                  value = '(complex(0,1)*complexconjugate(CtW))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_149 = Coupling(name = 'GC_149',
                  value = '-((complexconjugate(CtW)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_150 = Coupling(name = 'GC_150',
                  value = '(cw*complexconjugate(CtW))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_151 = Coupling(name = 'GC_151',
                  value = '-((cw*complexconjugate(CtW))/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QED':1})

GC_152 = Coupling(name = 'GC_152',
                  value = '(cw*complex(0,1)*complexconjugate(CtW))/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_153 = Coupling(name = 'GC_153',
                  value = '-((ee*complexconjugate(CtW))/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_154 = Coupling(name = 'GC_154',
                  value = '(ee*complex(0,1)*complexconjugate(CtW))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_155 = Coupling(name = 'GC_155',
                  value = '(ee*complexconjugate(CtW)*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_156 = Coupling(name = 'GC_156',
                  value = '(ee*complexconjugate(CtW))/(Lambda**2*sw)',
                  order = {'NP':2,'QED':2})

GC_157 = Coupling(name = 'GC_157',
                  value = '-((ee*complexconjugate(CtW))/(Lambda**2*sw*cmath.sqrt(2)))',
                  order = {'NP':2,'QED':2})

GC_158 = Coupling(name = 'GC_158',
                  value = '(ee*complex(0,1)*complexconjugate(CtW))/(Lambda**2*sw*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_159 = Coupling(name = 'GC_159',
                  value = '-((cw*ee*complex(0,1)*complexconjugate(CtW))/(Lambda**2*sw))',
                  order = {'NP':2,'QED':2})

GC_160 = Coupling(name = 'GC_160',
                  value = '(cw*ee*complexconjugate(CtW))/(Lambda**2*sw)',
                  order = {'NP':2,'QED':2})

GC_161 = Coupling(name = 'GC_161',
                  value = '-((cw*ee*complexconjugate(CtW)*cmath.sqrt(2))/(Lambda**2*sw))',
                  order = {'NP':2,'QED':2})

GC_162 = Coupling(name = 'GC_162',
                  value = '(sw*complexconjugate(CtW))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_163 = Coupling(name = 'GC_163',
                  value = '-((sw*complexconjugate(CtW))/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QED':1})

GC_164 = Coupling(name = 'GC_164',
                  value = '(complex(0,1)*sw*complexconjugate(CtW))/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_165 = Coupling(name = 'GC_165',
                  value = '(complex(0,1)*vev*complexconjugate(CtW))/Lambda**2',
                  order = {'NP':2})

GC_166 = Coupling(name = 'GC_166',
                  value = '(cw*complex(0,1)*vev*complexconjugate(CtW))/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2})

GC_167 = Coupling(name = 'GC_167',
                  value = '(ee*complex(0,1)*vev*complexconjugate(CtW))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_168 = Coupling(name = 'GC_168',
                  value = '(ee*complex(0,1)*vev*complexconjugate(CtW))/(Lambda**2*sw*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_169 = Coupling(name = 'GC_169',
                  value = '-((cw*ee*complex(0,1)*vev*complexconjugate(CtW))/(Lambda**2*sw))',
                  order = {'NP':2,'QED':1})

GC_170 = Coupling(name = 'GC_170',
                  value = '(complex(0,1)*sw*vev*complexconjugate(CtW))/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2})

