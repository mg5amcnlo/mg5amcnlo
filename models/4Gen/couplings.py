# This file was automatically created by FeynRules 1.7.55
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (October 6, 2011)
# Date: Wed 8 Aug 2012 14:29:47


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
                value = 'ee*complex(0,1)',
                order = {'QED':1})

GC_5 = Coupling(name = 'GC_5',
                value = '2*ee**2*complex(0,1)',
                order = {'QED':2})

GC_6 = Coupling(name = 'GC_6',
                value = '-ee**2/(2.*cw)',
                order = {'QED':2})

GC_7 = Coupling(name = 'GC_7',
                value = '(ee**2*complex(0,1))/(2.*cw)',
                order = {'QED':2})

GC_8 = Coupling(name = 'GC_8',
                value = 'ee**2/(2.*cw)',
                order = {'QED':2})

GC_9 = Coupling(name = 'GC_9',
                value = '-G',
                order = {'QCD':1})

GC_10 = Coupling(name = 'GC_10',
                 value = 'complex(0,1)*G',
                 order = {'QCD':1})

GC_11 = Coupling(name = 'GC_11',
                 value = 'complex(0,1)*G**2',
                 order = {'QCD':2})

GC_12 = Coupling(name = 'GC_12',
                 value = 'cw*complex(0,1)*gw',
                 order = {'QED':1})

GC_13 = Coupling(name = 'GC_13',
                 value = '-(complex(0,1)*gw**2)',
                 order = {'QED':2})

GC_14 = Coupling(name = 'GC_14',
                 value = 'cw**2*complex(0,1)*gw**2',
                 order = {'QED':2})

GC_15 = Coupling(name = 'GC_15',
                 value = '-2*complex(0,1)*lam',
                 order = {'QED':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '-4*complex(0,1)*lam',
                 order = {'QED':2})

GC_17 = Coupling(name = 'GC_17',
                 value = '-6*complex(0,1)*lam',
                 order = {'QED':2})

GC_18 = Coupling(name = 'GC_18',
                 value = '-(ee*MW)',
                 order = {'QED':1})

GC_19 = Coupling(name = 'GC_19',
                 value = 'ee*MW',
                 order = {'QED':1})

GC_20 = Coupling(name = 'GC_20',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_21 = Coupling(name = 'GC_21',
                 value = '-(ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_22 = Coupling(name = 'GC_22',
                 value = '(ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_23 = Coupling(name = 'GC_23',
                 value = 'ee/(2.*sw)',
                 order = {'QED':1})

GC_24 = Coupling(name = 'GC_24',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_25 = Coupling(name = 'GC_25',
                 value = '(CKM1x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_26 = Coupling(name = 'GC_26',
                 value = '(CKM1x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_27 = Coupling(name = 'GC_27',
                 value = '(CKM1x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_28 = Coupling(name = 'GC_28',
                 value = '(CKM1x4*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_29 = Coupling(name = 'GC_29',
                 value = '(CKM2x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_30 = Coupling(name = 'GC_30',
                 value = '(CKM2x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_31 = Coupling(name = 'GC_31',
                 value = '(CKM2x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_32 = Coupling(name = 'GC_32',
                 value = '(CKM2x4*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_33 = Coupling(name = 'GC_33',
                 value = '(CKM3x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_34 = Coupling(name = 'GC_34',
                 value = '(CKM3x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_35 = Coupling(name = 'GC_35',
                 value = '(CKM3x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = '(CKM3x4*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = '(CKM4x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_38 = Coupling(name = 'GC_38',
                 value = '(CKM4x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '(CKM4x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(CKM4x4*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_41 = Coupling(name = 'GC_41',
                 value = '-(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '-((cw*ee*complex(0,1))/sw)',
                 order = {'QED':1})

GC_44 = Coupling(name = 'GC_44',
                 value = '(cw*ee*complex(0,1))/sw',
                 order = {'QED':1})

GC_45 = Coupling(name = 'GC_45',
                 value = '-ee**2/(2.*sw)',
                 order = {'QED':2})

GC_46 = Coupling(name = 'GC_46',
                 value = '-(ee**2*complex(0,1))/(2.*sw)',
                 order = {'QED':2})

GC_47 = Coupling(name = 'GC_47',
                 value = 'ee**2/(2.*sw)',
                 order = {'QED':2})

GC_48 = Coupling(name = 'GC_48',
                 value = '-(ee*MW)/(2.*sw)',
                 order = {'QED':1})

GC_49 = Coupling(name = 'GC_49',
                 value = '-(ee*complex(0,1)*MW)/(2.*sw)',
                 order = {'QED':1})

GC_50 = Coupling(name = 'GC_50',
                 value = '(ee*MW)/(2.*sw)',
                 order = {'QED':1})

GC_51 = Coupling(name = 'GC_51',
                 value = '-(ee*MZ)/(2.*sw)',
                 order = {'QED':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '(ee*MZ)/(2.*sw)',
                 order = {'QED':1})

GC_53 = Coupling(name = 'GC_53',
                 value = '-(ee*complex(0,1)*MZ)/(2.*cw*sw)',
                 order = {'QED':1})

GC_54 = Coupling(name = 'GC_54',
                 value = '-(ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_55 = Coupling(name = 'GC_55',
                 value = '(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_56 = Coupling(name = 'GC_56',
                 value = 'complex(0,1)*gw*sw',
                 order = {'QED':1})

GC_57 = Coupling(name = 'GC_57',
                 value = '-2*cw*complex(0,1)*gw**2*sw',
                 order = {'QED':2})

GC_58 = Coupling(name = 'GC_58',
                 value = 'complex(0,1)*gw**2*sw**2',
                 order = {'QED':2})

GC_59 = Coupling(name = 'GC_59',
                 value = '-(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_60 = Coupling(name = 'GC_60',
                 value = '(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_61 = Coupling(name = 'GC_61',
                 value = '(cw*ee)/(2.*sw) + (ee*sw)/(2.*cw)',
                 order = {'QED':1})

GC_62 = Coupling(name = 'GC_62',
                 value = '(cw*ee**2*complex(0,1))/sw - (ee**2*complex(0,1)*sw)/cw',
                 order = {'QED':2})

GC_63 = Coupling(name = 'GC_63',
                 value = '(cw*ee*MW)/(2.*sw) - (ee*MW*sw)/(2.*cw)',
                 order = {'QED':1})

GC_64 = Coupling(name = 'GC_64',
                 value = '-(cw*ee*MW)/(2.*sw) + (ee*MW*sw)/(2.*cw)',
                 order = {'QED':1})

GC_65 = Coupling(name = 'GC_65',
                 value = '-(ee**2*complex(0,1)) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_66 = Coupling(name = 'GC_66',
                 value = 'ee**2*complex(0,1) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_67 = Coupling(name = 'GC_67',
                 value = '-(ee**2*v)/(2.*cw)',
                 order = {'QED':1})

GC_68 = Coupling(name = 'GC_68',
                 value = '(ee**2*v)/(2.*cw)',
                 order = {'QED':1})

GC_69 = Coupling(name = 'GC_69',
                 value = '-2*complex(0,1)*lam*v',
                 order = {'QED':1})

GC_70 = Coupling(name = 'GC_70',
                 value = '-6*complex(0,1)*lam*v',
                 order = {'QED':1})

GC_71 = Coupling(name = 'GC_71',
                 value = '(ee**2*complex(0,1)*v)/(2.*sw**2)',
                 order = {'QED':1})

GC_72 = Coupling(name = 'GC_72',
                 value = '-(ee**2*v)/(2.*sw)',
                 order = {'QED':1})

GC_73 = Coupling(name = 'GC_73',
                 value = '(ee**2*v)/(2.*sw)',
                 order = {'QED':1})

GC_74 = Coupling(name = 'GC_74',
                 value = 'ee**2*complex(0,1)*v + (cw**2*ee**2*complex(0,1)*v)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*v)/(2.*cw**2)',
                 order = {'QED':1})

GC_75 = Coupling(name = 'GC_75',
                 value = '-((complex(0,1)*yb)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_76 = Coupling(name = 'GC_76',
                 value = 'yb/cmath.sqrt(2)',
                 order = {'QED':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '-(CKM1x3*yb)',
                 order = {'QED':1})

GC_78 = Coupling(name = 'GC_78',
                 value = '-(CKM2x3*yb)',
                 order = {'QED':1})

GC_79 = Coupling(name = 'GC_79',
                 value = '-(CKM3x3*yb)',
                 order = {'QED':1})

GC_80 = Coupling(name = 'GC_80',
                 value = '-(CKM4x3*yb)',
                 order = {'QED':1})

GC_81 = Coupling(name = 'GC_81',
                 value = '-((complex(0,1)*ybp)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_82 = Coupling(name = 'GC_82',
                 value = 'ybp/cmath.sqrt(2)',
                 order = {'QED':1})

GC_83 = Coupling(name = 'GC_83',
                 value = '-(CKM1x4*ybp)',
                 order = {'QED':1})

GC_84 = Coupling(name = 'GC_84',
                 value = '-(CKM2x4*ybp)',
                 order = {'QED':1})

GC_85 = Coupling(name = 'GC_85',
                 value = '-(CKM3x4*ybp)',
                 order = {'QED':1})

GC_86 = Coupling(name = 'GC_86',
                 value = '-(CKM4x4*ybp)',
                 order = {'QED':1})

GC_87 = Coupling(name = 'GC_87',
                 value = '-((complex(0,1)*ydo)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_88 = Coupling(name = 'GC_88',
                 value = 'ydo/cmath.sqrt(2)',
                 order = {'QED':1})

GC_89 = Coupling(name = 'GC_89',
                 value = '-(CKM1x1*ydo)',
                 order = {'QED':1})

GC_90 = Coupling(name = 'GC_90',
                 value = '-(CKM2x1*ydo)',
                 order = {'QED':1})

GC_91 = Coupling(name = 'GC_91',
                 value = '-(CKM3x1*ydo)',
                 order = {'QED':1})

GC_92 = Coupling(name = 'GC_92',
                 value = '-(CKM4x1*ydo)',
                 order = {'QED':1})

GC_93 = Coupling(name = 'GC_93',
                 value = '-ye',
                 order = {'QED':1})

GC_94 = Coupling(name = 'GC_94',
                 value = 'ye',
                 order = {'QED':1})

GC_95 = Coupling(name = 'GC_95',
                 value = '-((complex(0,1)*ye)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_96 = Coupling(name = 'GC_96',
                 value = 'ye/cmath.sqrt(2)',
                 order = {'QED':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '-ym',
                 order = {'QED':1})

GC_98 = Coupling(name = 'GC_98',
                 value = 'ym',
                 order = {'QED':1})

GC_99 = Coupling(name = 'GC_99',
                 value = '-((complex(0,1)*ym)/cmath.sqrt(2))',
                 order = {'QED':1})

GC_100 = Coupling(name = 'GC_100',
                  value = 'ym/cmath.sqrt(2)',
                  order = {'QED':1})

GC_101 = Coupling(name = 'GC_101',
                  value = '-((complex(0,1)*ys)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_102 = Coupling(name = 'GC_102',
                  value = 'ys/cmath.sqrt(2)',
                  order = {'QED':1})

GC_103 = Coupling(name = 'GC_103',
                  value = '-(CKM1x2*ys)',
                  order = {'QED':1})

GC_104 = Coupling(name = 'GC_104',
                  value = '-(CKM2x2*ys)',
                  order = {'QED':1})

GC_105 = Coupling(name = 'GC_105',
                  value = '-(CKM3x2*ys)',
                  order = {'QED':1})

GC_106 = Coupling(name = 'GC_106',
                  value = '-(CKM4x2*ys)',
                  order = {'QED':1})

GC_107 = Coupling(name = 'GC_107',
                  value = '-(yt/cmath.sqrt(2))',
                  order = {'QED':1})

GC_108 = Coupling(name = 'GC_108',
                  value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_109 = Coupling(name = 'GC_109',
                  value = 'CKM3x1*yt',
                  order = {'QED':1})

GC_110 = Coupling(name = 'GC_110',
                  value = 'CKM3x2*yt',
                  order = {'QED':1})

GC_111 = Coupling(name = 'GC_111',
                  value = 'CKM3x3*yt',
                  order = {'QED':1})

GC_112 = Coupling(name = 'GC_112',
                  value = 'CKM3x4*yt',
                  order = {'QED':1})

GC_113 = Coupling(name = 'GC_113',
                  value = '-ytau',
                  order = {'QED':1})

GC_114 = Coupling(name = 'GC_114',
                  value = 'ytau',
                  order = {'QED':1})

GC_115 = Coupling(name = 'GC_115',
                  value = '-((complex(0,1)*ytau)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_116 = Coupling(name = 'GC_116',
                  value = 'ytau/cmath.sqrt(2)',
                  order = {'QED':1})

GC_117 = Coupling(name = 'GC_117',
                  value = '-(ytp/cmath.sqrt(2))',
                  order = {'QED':1})

GC_118 = Coupling(name = 'GC_118',
                  value = '-((complex(0,1)*ytp)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_119 = Coupling(name = 'GC_119',
                  value = 'CKM4x1*ytp',
                  order = {'QED':1})

GC_120 = Coupling(name = 'GC_120',
                  value = 'CKM4x2*ytp',
                  order = {'QED':1})

GC_121 = Coupling(name = 'GC_121',
                  value = 'CKM4x3*ytp',
                  order = {'QED':1})

GC_122 = Coupling(name = 'GC_122',
                  value = 'CKM4x4*ytp',
                  order = {'QED':1})

GC_123 = Coupling(name = 'GC_123',
                  value = '-(yup/cmath.sqrt(2))',
                  order = {'QED':1})

GC_124 = Coupling(name = 'GC_124',
                  value = '-((complex(0,1)*yup)/cmath.sqrt(2))',
                  order = {'QED':1})

GC_125 = Coupling(name = 'GC_125',
                  value = 'CKM1x1*yup',
                  order = {'QED':1})

GC_126 = Coupling(name = 'GC_126',
                  value = 'CKM1x2*yup',
                  order = {'QED':1})

GC_127 = Coupling(name = 'GC_127',
                  value = 'CKM1x3*yup',
                  order = {'QED':1})

GC_128 = Coupling(name = 'GC_128',
                  value = 'CKM1x4*yup',
                  order = {'QED':1})

GC_129 = Coupling(name = 'GC_129',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_130 = Coupling(name = 'GC_130',
                  value = 'ydo*complexconjugate(CKM1x1)',
                  order = {'QED':1})

GC_131 = Coupling(name = 'GC_131',
                  value = '-(yup*complexconjugate(CKM1x1))',
                  order = {'QED':1})

GC_132 = Coupling(name = 'GC_132',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_133 = Coupling(name = 'GC_133',
                  value = 'ys*complexconjugate(CKM1x2)',
                  order = {'QED':1})

GC_134 = Coupling(name = 'GC_134',
                  value = '-(yup*complexconjugate(CKM1x2))',
                  order = {'QED':1})

GC_135 = Coupling(name = 'GC_135',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_136 = Coupling(name = 'GC_136',
                  value = 'yb*complexconjugate(CKM1x3)',
                  order = {'QED':1})

GC_137 = Coupling(name = 'GC_137',
                  value = '-(yup*complexconjugate(CKM1x3))',
                  order = {'QED':1})

GC_138 = Coupling(name = 'GC_138',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_139 = Coupling(name = 'GC_139',
                  value = 'ybp*complexconjugate(CKM1x4)',
                  order = {'QED':1})

GC_140 = Coupling(name = 'GC_140',
                  value = '-(yup*complexconjugate(CKM1x4))',
                  order = {'QED':1})

GC_141 = Coupling(name = 'GC_141',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_142 = Coupling(name = 'GC_142',
                  value = 'ydo*complexconjugate(CKM2x1)',
                  order = {'QED':1})

GC_143 = Coupling(name = 'GC_143',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_144 = Coupling(name = 'GC_144',
                  value = 'ys*complexconjugate(CKM2x2)',
                  order = {'QED':1})

GC_145 = Coupling(name = 'GC_145',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_146 = Coupling(name = 'GC_146',
                  value = 'yb*complexconjugate(CKM2x3)',
                  order = {'QED':1})

GC_147 = Coupling(name = 'GC_147',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_148 = Coupling(name = 'GC_148',
                  value = 'ybp*complexconjugate(CKM2x4)',
                  order = {'QED':1})

GC_149 = Coupling(name = 'GC_149',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_150 = Coupling(name = 'GC_150',
                  value = 'ydo*complexconjugate(CKM3x1)',
                  order = {'QED':1})

GC_151 = Coupling(name = 'GC_151',
                  value = '-(yt*complexconjugate(CKM3x1))',
                  order = {'QED':1})

GC_152 = Coupling(name = 'GC_152',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_153 = Coupling(name = 'GC_153',
                  value = 'ys*complexconjugate(CKM3x2)',
                  order = {'QED':1})

GC_154 = Coupling(name = 'GC_154',
                  value = '-(yt*complexconjugate(CKM3x2))',
                  order = {'QED':1})

GC_155 = Coupling(name = 'GC_155',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_156 = Coupling(name = 'GC_156',
                  value = 'yb*complexconjugate(CKM3x3)',
                  order = {'QED':1})

GC_157 = Coupling(name = 'GC_157',
                  value = '-(yt*complexconjugate(CKM3x3))',
                  order = {'QED':1})

GC_158 = Coupling(name = 'GC_158',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_159 = Coupling(name = 'GC_159',
                  value = 'ybp*complexconjugate(CKM3x4)',
                  order = {'QED':1})

GC_160 = Coupling(name = 'GC_160',
                  value = '-(yt*complexconjugate(CKM3x4))',
                  order = {'QED':1})

GC_161 = Coupling(name = 'GC_161',
                  value = '(ee*complex(0,1)*complexconjugate(CKM4x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_162 = Coupling(name = 'GC_162',
                  value = 'ydo*complexconjugate(CKM4x1)',
                  order = {'QED':1})

GC_163 = Coupling(name = 'GC_163',
                  value = '-(ytp*complexconjugate(CKM4x1))',
                  order = {'QED':1})

GC_164 = Coupling(name = 'GC_164',
                  value = '(ee*complex(0,1)*complexconjugate(CKM4x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_165 = Coupling(name = 'GC_165',
                  value = 'ys*complexconjugate(CKM4x2)',
                  order = {'QED':1})

GC_166 = Coupling(name = 'GC_166',
                  value = '-(ytp*complexconjugate(CKM4x2))',
                  order = {'QED':1})

GC_167 = Coupling(name = 'GC_167',
                  value = '(ee*complex(0,1)*complexconjugate(CKM4x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_168 = Coupling(name = 'GC_168',
                  value = 'yb*complexconjugate(CKM4x3)',
                  order = {'QED':1})

GC_169 = Coupling(name = 'GC_169',
                  value = '-(ytp*complexconjugate(CKM4x3))',
                  order = {'QED':1})

GC_170 = Coupling(name = 'GC_170',
                  value = '(ee*complex(0,1)*complexconjugate(CKM4x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_171 = Coupling(name = 'GC_171',
                  value = 'ybp*complexconjugate(CKM4x4)',
                  order = {'QED':1})

GC_172 = Coupling(name = 'GC_172',
                  value = '-(ytp*complexconjugate(CKM4x4))',
                  order = {'QED':1})

