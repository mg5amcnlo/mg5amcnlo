# This file was automatically created by FeynRules 1.7.53
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Tue 31 Jul 2012 19:55:14


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
                value = 'ee**2*complex(0,1)',
                order = {'QED':2})

GC_6 = Coupling(name = 'GC_6',
                value = '-G',
                order = {'QCD':1})

GC_7 = Coupling(name = 'GC_7',
                value = 'complex(0,1)*G',
                order = {'QCD':1})

GC_8 = Coupling(name = 'GC_8',
                value = 'G',
                order = {'QCD':1})

GC_9 = Coupling(name = 'GC_9',
                value = 'complex(0,1)*G**2',
                order = {'QCD':2})

GC_10 = Coupling(name = 'GC_10',
                 value = '(2*ee**2*complex(0,1)*I12x11)/9.',
                 order = {'QED':2})

GC_11 = Coupling(name = 'GC_11',
                 value = '(-2*ee*complex(0,1)*G*I12x11)/3.',
                 order = {'QCD':1,'QED':1})

GC_12 = Coupling(name = 'GC_12',
                 value = 'complex(0,1)*G**2*I12x11',
                 order = {'QCD':2})

GC_13 = Coupling(name = 'GC_13',
                 value = '(2*ee**2*complex(0,1)*I12x22)/9.',
                 order = {'QED':2})

GC_14 = Coupling(name = 'GC_14',
                 value = '(-2*ee*complex(0,1)*G*I12x22)/3.',
                 order = {'QCD':1,'QED':1})

GC_15 = Coupling(name = 'GC_15',
                 value = 'complex(0,1)*G**2*I12x22',
                 order = {'QCD':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '(2*ee**2*complex(0,1)*I12x33)/9. + (2*ee**2*complex(0,1)*I13x33)/9.',
                 order = {'QED':2})

GC_17 = Coupling(name = 'GC_17',
                 value = '(-2*ee*complex(0,1)*G*I12x33)/3. - (2*ee*complex(0,1)*G*I13x33)/3.',
                 order = {'QCD':1,'QED':1})

GC_18 = Coupling(name = 'GC_18',
                 value = 'complex(0,1)*G**2*I12x33 + complex(0,1)*G**2*I13x33',
                 order = {'QCD':2})

GC_19 = Coupling(name = 'GC_19',
                 value = '(2*ee**2*complex(0,1)*I12x36)/9. + (2*ee**2*complex(0,1)*I13x36)/9.',
                 order = {'QED':2})

GC_20 = Coupling(name = 'GC_20',
                 value = '(-2*ee*complex(0,1)*G*I12x36)/3. - (2*ee*complex(0,1)*G*I13x36)/3.',
                 order = {'QCD':1,'QED':1})

GC_21 = Coupling(name = 'GC_21',
                 value = 'complex(0,1)*G**2*I12x36 + complex(0,1)*G**2*I13x36',
                 order = {'QCD':2})

GC_22 = Coupling(name = 'GC_22',
                 value = '(2*ee**2*complex(0,1)*I13x44)/9.',
                 order = {'QED':2})

GC_23 = Coupling(name = 'GC_23',
                 value = '(-2*ee*complex(0,1)*G*I13x44)/3.',
                 order = {'QCD':1,'QED':1})

GC_24 = Coupling(name = 'GC_24',
                 value = 'complex(0,1)*G**2*I13x44',
                 order = {'QCD':2})

GC_25 = Coupling(name = 'GC_25',
                 value = '(2*ee**2*complex(0,1)*I13x55)/9.',
                 order = {'QED':2})

GC_26 = Coupling(name = 'GC_26',
                 value = '(-2*ee*complex(0,1)*G*I13x55)/3.',
                 order = {'QCD':1,'QED':1})

GC_27 = Coupling(name = 'GC_27',
                 value = 'complex(0,1)*G**2*I13x55',
                 order = {'QCD':2})

GC_28 = Coupling(name = 'GC_28',
                 value = '(2*ee**2*complex(0,1)*I12x63)/9. + (2*ee**2*complex(0,1)*I13x63)/9.',
                 order = {'QED':2})

GC_29 = Coupling(name = 'GC_29',
                 value = '(-2*ee*complex(0,1)*G*I12x63)/3. - (2*ee*complex(0,1)*G*I13x63)/3.',
                 order = {'QCD':1,'QED':1})

GC_30 = Coupling(name = 'GC_30',
                 value = 'complex(0,1)*G**2*I12x63 + complex(0,1)*G**2*I13x63',
                 order = {'QCD':2})

GC_31 = Coupling(name = 'GC_31',
                 value = '(2*ee**2*complex(0,1)*I12x66)/9. + (2*ee**2*complex(0,1)*I13x66)/9.',
                 order = {'QED':2})

GC_32 = Coupling(name = 'GC_32',
                 value = '(-2*ee*complex(0,1)*G*I12x66)/3. - (2*ee*complex(0,1)*G*I13x66)/3.',
                 order = {'QCD':1,'QED':1})

GC_33 = Coupling(name = 'GC_33',
                 value = 'complex(0,1)*G**2*I12x66 + complex(0,1)*G**2*I13x66',
                 order = {'QCD':2})

GC_34 = Coupling(name = 'GC_34',
                 value = 'ee*complex(0,1)*I25x11',
                 order = {'QED':1})

GC_35 = Coupling(name = 'GC_35',
                 value = 'ee*complex(0,1)*I25x22',
                 order = {'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = 'ee*complex(0,1)*I25x33 + ee*complex(0,1)*I26x33',
                 order = {'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = 'ee*complex(0,1)*I25x36 + ee*complex(0,1)*I26x36',
                 order = {'QED':1})

GC_38 = Coupling(name = 'GC_38',
                 value = 'ee*complex(0,1)*I26x44',
                 order = {'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = 'ee*complex(0,1)*I26x55',
                 order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '-(ee*complex(0,1)*I25x63) - ee*complex(0,1)*I26x63',
                 order = {'QED':1})

GC_41 = Coupling(name = 'GC_41',
                 value = 'ee*complex(0,1)*I25x66 + ee*complex(0,1)*I26x66',
                 order = {'QED':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '2*ee**2*complex(0,1)*I31x11',
                 order = {'QED':2})

GC_43 = Coupling(name = 'GC_43',
                 value = '2*ee**2*complex(0,1)*I31x22',
                 order = {'QED':2})

GC_44 = Coupling(name = 'GC_44',
                 value = '2*ee**2*complex(0,1)*I31x33 + 2*ee**2*complex(0,1)*I32x33',
                 order = {'QED':2})

GC_45 = Coupling(name = 'GC_45',
                 value = '2*ee**2*complex(0,1)*I31x36 + 2*ee**2*complex(0,1)*I32x36',
                 order = {'QED':2})

GC_46 = Coupling(name = 'GC_46',
                 value = '2*ee**2*complex(0,1)*I32x44',
                 order = {'QED':2})

GC_47 = Coupling(name = 'GC_47',
                 value = '2*ee**2*complex(0,1)*I32x55',
                 order = {'QED':2})

GC_48 = Coupling(name = 'GC_48',
                 value = '2*ee**2*complex(0,1)*I31x63 + 2*ee**2*complex(0,1)*I32x63',
                 order = {'QED':2})

GC_49 = Coupling(name = 'GC_49',
                 value = '2*ee**2*complex(0,1)*I31x66 + 2*ee**2*complex(0,1)*I32x66',
                 order = {'QED':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '(-2*ee*complex(0,1)*I51x11)/3.',
                 order = {'QED':1})

GC_51 = Coupling(name = 'GC_51',
                 value = '-(complex(0,1)*G*I51x11)',
                 order = {'QCD':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '(-2*ee*complex(0,1)*I51x22)/3.',
                 order = {'QED':1})

GC_53 = Coupling(name = 'GC_53',
                 value = '-(complex(0,1)*G*I51x22)',
                 order = {'QCD':1})

GC_54 = Coupling(name = 'GC_54',
                 value = '(-2*ee*complex(0,1)*I51x33)/3. - (2*ee*complex(0,1)*I52x33)/3.',
                 order = {'QED':1})

GC_55 = Coupling(name = 'GC_55',
                 value = '-(complex(0,1)*G*I51x33) - complex(0,1)*G*I52x33',
                 order = {'QCD':1})

GC_56 = Coupling(name = 'GC_56',
                 value = '(-2*ee*complex(0,1)*I51x36)/3. - (2*ee*complex(0,1)*I52x36)/3.',
                 order = {'QED':1})

GC_57 = Coupling(name = 'GC_57',
                 value = '-(complex(0,1)*G*I51x36) - complex(0,1)*G*I52x36',
                 order = {'QCD':1})

GC_58 = Coupling(name = 'GC_58',
                 value = '(-2*ee*complex(0,1)*I52x44)/3.',
                 order = {'QED':1})

GC_59 = Coupling(name = 'GC_59',
                 value = '-(complex(0,1)*G*I52x44)',
                 order = {'QCD':1})

GC_60 = Coupling(name = 'GC_60',
                 value = '(-2*ee*complex(0,1)*I52x55)/3.',
                 order = {'QED':1})

GC_61 = Coupling(name = 'GC_61',
                 value = '-(complex(0,1)*G*I52x55)',
                 order = {'QCD':1})

GC_62 = Coupling(name = 'GC_62',
                 value = '(2*ee*complex(0,1)*I51x63)/3. + (2*ee*complex(0,1)*I52x63)/3.',
                 order = {'QED':1})

GC_63 = Coupling(name = 'GC_63',
                 value = 'complex(0,1)*G*I51x63 + complex(0,1)*G*I52x63',
                 order = {'QCD':1})

GC_64 = Coupling(name = 'GC_64',
                 value = '(-2*ee*complex(0,1)*I51x66)/3. - (2*ee*complex(0,1)*I52x66)/3.',
                 order = {'QED':1})

GC_65 = Coupling(name = 'GC_65',
                 value = '-(complex(0,1)*G*I51x66) - complex(0,1)*G*I52x66',
                 order = {'QCD':1})

GC_66 = Coupling(name = 'GC_66',
                 value = '(ee*complex(0,1)*I5x11)/3.',
                 order = {'QED':1})

GC_67 = Coupling(name = 'GC_67',
                 value = '-(complex(0,1)*G*I5x11)',
                 order = {'QCD':1})

GC_68 = Coupling(name = 'GC_68',
                 value = '(ee*complex(0,1)*I5x22)/3.',
                 order = {'QED':1})

GC_69 = Coupling(name = 'GC_69',
                 value = '-(complex(0,1)*G*I5x22)',
                 order = {'QCD':1})

GC_70 = Coupling(name = 'GC_70',
                 value = '(ee*complex(0,1)*I5x33)/3. + (ee*complex(0,1)*I6x33)/3.',
                 order = {'QED':1})

GC_71 = Coupling(name = 'GC_71',
                 value = '-(complex(0,1)*G*I5x33) - complex(0,1)*G*I6x33',
                 order = {'QCD':1})

GC_72 = Coupling(name = 'GC_72',
                 value = '(ee*complex(0,1)*I5x36)/3. + (ee*complex(0,1)*I6x36)/3.',
                 order = {'QED':1})

GC_73 = Coupling(name = 'GC_73',
                 value = '-(complex(0,1)*G*I5x36) - complex(0,1)*G*I6x36',
                 order = {'QCD':1})

GC_74 = Coupling(name = 'GC_74',
                 value = '(ee*complex(0,1)*I6x44)/3.',
                 order = {'QED':1})

GC_75 = Coupling(name = 'GC_75',
                 value = '-(complex(0,1)*G*I6x44)',
                 order = {'QCD':1})

GC_76 = Coupling(name = 'GC_76',
                 value = '(ee*complex(0,1)*I6x55)/3.',
                 order = {'QED':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '-(complex(0,1)*G*I6x55)',
                 order = {'QCD':1})

GC_78 = Coupling(name = 'GC_78',
                 value = '-(ee*complex(0,1)*I5x63)/3. - (ee*complex(0,1)*I6x63)/3.',
                 order = {'QED':1})

GC_79 = Coupling(name = 'GC_79',
                 value = 'complex(0,1)*G*I5x63 + complex(0,1)*G*I6x63',
                 order = {'QCD':1})

GC_80 = Coupling(name = 'GC_80',
                 value = '(ee*complex(0,1)*I5x66)/3. + (ee*complex(0,1)*I6x66)/3.',
                 order = {'QED':1})

GC_81 = Coupling(name = 'GC_81',
                 value = '-(complex(0,1)*G*I5x66) - complex(0,1)*G*I6x66',
                 order = {'QCD':1})

GC_82 = Coupling(name = 'GC_82',
                 value = '(8*ee**2*complex(0,1)*I74x11)/9.',
                 order = {'QED':2})

GC_83 = Coupling(name = 'GC_83',
                 value = '(4*ee*complex(0,1)*G*I74x11)/3.',
                 order = {'QCD':1,'QED':1})

GC_84 = Coupling(name = 'GC_84',
                 value = 'complex(0,1)*G**2*I74x11',
                 order = {'QCD':2})

GC_85 = Coupling(name = 'GC_85',
                 value = '(8*ee**2*complex(0,1)*I74x22)/9.',
                 order = {'QED':2})

GC_86 = Coupling(name = 'GC_86',
                 value = '(4*ee*complex(0,1)*G*I74x22)/3.',
                 order = {'QCD':1,'QED':1})

GC_87 = Coupling(name = 'GC_87',
                 value = 'complex(0,1)*G**2*I74x22',
                 order = {'QCD':2})

GC_88 = Coupling(name = 'GC_88',
                 value = '(8*ee**2*complex(0,1)*I74x33)/9. + (8*ee**2*complex(0,1)*I75x33)/9.',
                 order = {'QED':2})

GC_89 = Coupling(name = 'GC_89',
                 value = '(4*ee*complex(0,1)*G*I74x33)/3. + (4*ee*complex(0,1)*G*I75x33)/3.',
                 order = {'QCD':1,'QED':1})

GC_90 = Coupling(name = 'GC_90',
                 value = 'complex(0,1)*G**2*I74x33 + complex(0,1)*G**2*I75x33',
                 order = {'QCD':2})

GC_91 = Coupling(name = 'GC_91',
                 value = '(8*ee**2*complex(0,1)*I74x36)/9. + (8*ee**2*complex(0,1)*I75x36)/9.',
                 order = {'QED':2})

GC_92 = Coupling(name = 'GC_92',
                 value = '(4*ee*complex(0,1)*G*I74x36)/3. + (4*ee*complex(0,1)*G*I75x36)/3.',
                 order = {'QCD':1,'QED':1})

GC_93 = Coupling(name = 'GC_93',
                 value = 'complex(0,1)*G**2*I74x36 + complex(0,1)*G**2*I75x36',
                 order = {'QCD':2})

GC_94 = Coupling(name = 'GC_94',
                 value = '(8*ee**2*complex(0,1)*I75x44)/9.',
                 order = {'QED':2})

GC_95 = Coupling(name = 'GC_95',
                 value = '(4*ee*complex(0,1)*G*I75x44)/3.',
                 order = {'QCD':1,'QED':1})

GC_96 = Coupling(name = 'GC_96',
                 value = 'complex(0,1)*G**2*I75x44',
                 order = {'QCD':2})

GC_97 = Coupling(name = 'GC_97',
                 value = '(8*ee**2*complex(0,1)*I75x55)/9.',
                 order = {'QED':2})

GC_98 = Coupling(name = 'GC_98',
                 value = '(4*ee*complex(0,1)*G*I75x55)/3.',
                 order = {'QCD':1,'QED':1})

GC_99 = Coupling(name = 'GC_99',
                 value = 'complex(0,1)*G**2*I75x55',
                 order = {'QCD':2})

GC_100 = Coupling(name = 'GC_100',
                  value = '(8*ee**2*complex(0,1)*I74x63)/9. + (8*ee**2*complex(0,1)*I75x63)/9.',
                  order = {'QED':2})

GC_101 = Coupling(name = 'GC_101',
                  value = '(4*ee*complex(0,1)*G*I74x63)/3. + (4*ee*complex(0,1)*G*I75x63)/3.',
                  order = {'QCD':1,'QED':1})

GC_102 = Coupling(name = 'GC_102',
                  value = 'complex(0,1)*G**2*I74x63 + complex(0,1)*G**2*I75x63',
                  order = {'QCD':2})

GC_103 = Coupling(name = 'GC_103',
                  value = '(8*ee**2*complex(0,1)*I74x66)/9. + (8*ee**2*complex(0,1)*I75x66)/9.',
                  order = {'QED':2})

GC_104 = Coupling(name = 'GC_104',
                  value = '(4*ee*complex(0,1)*G*I74x66)/3. + (4*ee*complex(0,1)*G*I75x66)/3.',
                  order = {'QCD':1,'QED':1})

GC_105 = Coupling(name = 'GC_105',
                  value = 'complex(0,1)*G**2*I74x66 + complex(0,1)*G**2*I75x66',
                  order = {'QCD':2})

GC_106 = Coupling(name = 'GC_106',
                  value = '-(complex(0,1)*G*Rd1x1*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_107 = Coupling(name = 'GC_107',
                  value = '-(complex(0,1)*G*Rd2x2*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_108 = Coupling(name = 'GC_108',
                  value = '-(complex(0,1)*G*Rd3x3*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_109 = Coupling(name = 'GC_109',
                  value = 'complex(0,1)*G*Rd3x6*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_110 = Coupling(name = 'GC_110',
                  value = 'complex(0,1)*G*Rd4x4*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_111 = Coupling(name = 'GC_111',
                  value = 'complex(0,1)*G*Rd5x5*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_112 = Coupling(name = 'GC_112',
                  value = '-(complex(0,1)*G*Rd6x3*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_113 = Coupling(name = 'GC_113',
                  value = 'complex(0,1)*G*Rd6x6*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_114 = Coupling(name = 'GC_114',
                  value = '-(complex(0,1)*G*Ru1x1*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_115 = Coupling(name = 'GC_115',
                  value = '-(complex(0,1)*G*Ru2x2*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_116 = Coupling(name = 'GC_116',
                  value = '-(complex(0,1)*G*Ru3x3*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_117 = Coupling(name = 'GC_117',
                  value = 'complex(0,1)*G*Ru3x6*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_118 = Coupling(name = 'GC_118',
                  value = 'complex(0,1)*G*Ru4x4*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_119 = Coupling(name = 'GC_119',
                  value = 'complex(0,1)*G*Ru5x5*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_120 = Coupling(name = 'GC_120',
                  value = '-(complex(0,1)*G*Ru6x3*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_121 = Coupling(name = 'GC_121',
                  value = 'complex(0,1)*G*Ru6x6*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_122 = Coupling(name = 'GC_122',
                  value = '-(ee**2*complex(0,1)) + (ee**2*complex(0,1))/sw**2',
                  order = {'QED':2})

GC_123 = Coupling(name = 'GC_123',
                  value = '(ee**2*complex(0,1))/(2.*sw**2)',
                  order = {'QED':2})

GC_124 = Coupling(name = 'GC_124',
                  value = '-((ee**2*complex(0,1))/sw**2)',
                  order = {'QED':2})

GC_125 = Coupling(name = 'GC_125',
                  value = '(ee**2*complex(0,1)*I96x11)/(2.*sw**2)',
                  order = {'QED':2})

GC_126 = Coupling(name = 'GC_126',
                  value = '(ee**2*complex(0,1)*I96x22)/(2.*sw**2)',
                  order = {'QED':2})

GC_127 = Coupling(name = 'GC_127',
                  value = '(ee**2*complex(0,1)*I96x33)/(2.*sw**2)',
                  order = {'QED':2})

GC_128 = Coupling(name = 'GC_128',
                  value = '(ee**2*complex(0,1)*I96x36)/(2.*sw**2)',
                  order = {'QED':2})

GC_129 = Coupling(name = 'GC_129',
                  value = '(ee**2*complex(0,1)*I96x63)/(2.*sw**2)',
                  order = {'QED':2})

GC_130 = Coupling(name = 'GC_130',
                  value = '(ee**2*complex(0,1)*I96x66)/(2.*sw**2)',
                  order = {'QED':2})

GC_131 = Coupling(name = 'GC_131',
                  value = '(ee**2*complex(0,1)*I97x11)/(2.*sw**2)',
                  order = {'QED':2})

GC_132 = Coupling(name = 'GC_132',
                  value = '(ee**2*complex(0,1)*I97x22)/(2.*sw**2)',
                  order = {'QED':2})

GC_133 = Coupling(name = 'GC_133',
                  value = '(ee**2*complex(0,1)*I97x33)/(2.*sw**2)',
                  order = {'QED':2})

GC_134 = Coupling(name = 'GC_134',
                  value = '(ee**2*complex(0,1)*I97x36)/(2.*sw**2)',
                  order = {'QED':2})

GC_135 = Coupling(name = 'GC_135',
                  value = '(ee**2*complex(0,1)*I97x63)/(2.*sw**2)',
                  order = {'QED':2})

GC_136 = Coupling(name = 'GC_136',
                  value = '(ee**2*complex(0,1)*I97x66)/(2.*sw**2)',
                  order = {'QED':2})

GC_137 = Coupling(name = 'GC_137',
                  value = '(ee**2*complex(0,1)*I98x11)/(2.*sw**2)',
                  order = {'QED':2})

GC_138 = Coupling(name = 'GC_138',
                  value = '(ee**2*complex(0,1)*I98x22)/(2.*sw**2)',
                  order = {'QED':2})

GC_139 = Coupling(name = 'GC_139',
                  value = '(ee**2*complex(0,1)*I98x33)/(2.*sw**2)',
                  order = {'QED':2})

GC_140 = Coupling(name = 'GC_140',
                  value = '(ee**2*complex(0,1)*I98x36)/(2.*sw**2)',
                  order = {'QED':2})

GC_141 = Coupling(name = 'GC_141',
                  value = '(ee**2*complex(0,1)*I98x63)/(2.*sw**2)',
                  order = {'QED':2})

GC_142 = Coupling(name = 'GC_142',
                  value = '(ee**2*complex(0,1)*I98x66)/(2.*sw**2)',
                  order = {'QED':2})

GC_143 = Coupling(name = 'GC_143',
                  value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_144 = Coupling(name = 'GC_144',
                  value = '(CKM1x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_145 = Coupling(name = 'GC_145',
                  value = '(CKM2x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_146 = Coupling(name = 'GC_146',
                  value = '(CKM3x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_147 = Coupling(name = 'GC_147',
                  value = '(cw*ee*complex(0,1))/sw',
                  order = {'QED':1})

GC_148 = Coupling(name = 'GC_148',
                  value = '(-2*cw*ee**2*complex(0,1))/sw',
                  order = {'QED':2})

GC_149 = Coupling(name = 'GC_149',
                  value = '-((ee**2*complex(0,1)*I39x11)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_150 = Coupling(name = 'GC_150',
                  value = '-((ee**2*complex(0,1)*I39x22)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_151 = Coupling(name = 'GC_151',
                  value = '-((ee**2*complex(0,1)*I39x33)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_152 = Coupling(name = 'GC_152',
                  value = '-((ee**2*complex(0,1)*I39x36)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_153 = Coupling(name = 'GC_153',
                  value = '-((ee**2*complex(0,1)*I45x11)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_154 = Coupling(name = 'GC_154',
                  value = '-((ee**2*complex(0,1)*I45x22)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_155 = Coupling(name = 'GC_155',
                  value = '-((ee**2*complex(0,1)*I45x33)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_156 = Coupling(name = 'GC_156',
                  value = '-((ee**2*complex(0,1)*I45x36)/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_157 = Coupling(name = 'GC_157',
                  value = '(ee**2*complex(0,1)*I53x11)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_158 = Coupling(name = 'GC_158',
                  value = '(ee*complex(0,1)*G*I53x11*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_159 = Coupling(name = 'GC_159',
                  value = '(ee**2*complex(0,1)*I53x22)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_160 = Coupling(name = 'GC_160',
                  value = '(ee*complex(0,1)*G*I53x22*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_161 = Coupling(name = 'GC_161',
                  value = '(ee**2*complex(0,1)*I53x33)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_162 = Coupling(name = 'GC_162',
                  value = '(ee*complex(0,1)*G*I53x33*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_163 = Coupling(name = 'GC_163',
                  value = '(ee**2*complex(0,1)*I53x36)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_164 = Coupling(name = 'GC_164',
                  value = '(ee*complex(0,1)*G*I53x36*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_165 = Coupling(name = 'GC_165',
                  value = '(ee**2*complex(0,1)*I53x63)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_166 = Coupling(name = 'GC_166',
                  value = '(ee*complex(0,1)*G*I53x63*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_167 = Coupling(name = 'GC_167',
                  value = '(ee**2*complex(0,1)*I53x66)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_168 = Coupling(name = 'GC_168',
                  value = '(ee*complex(0,1)*G*I53x66*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_169 = Coupling(name = 'GC_169',
                  value = '(ee**2*complex(0,1)*I66x11)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_170 = Coupling(name = 'GC_170',
                  value = '(ee*complex(0,1)*G*I66x11*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_171 = Coupling(name = 'GC_171',
                  value = '(ee**2*complex(0,1)*I66x22)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_172 = Coupling(name = 'GC_172',
                  value = '(ee*complex(0,1)*G*I66x22*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_173 = Coupling(name = 'GC_173',
                  value = '(ee**2*complex(0,1)*I66x33)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_174 = Coupling(name = 'GC_174',
                  value = '(ee*complex(0,1)*G*I66x33*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_175 = Coupling(name = 'GC_175',
                  value = '(ee**2*complex(0,1)*I66x36)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_176 = Coupling(name = 'GC_176',
                  value = '(ee*complex(0,1)*G*I66x36*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_177 = Coupling(name = 'GC_177',
                  value = '(ee**2*complex(0,1)*I66x63)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_178 = Coupling(name = 'GC_178',
                  value = '(ee*complex(0,1)*G*I66x63*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_179 = Coupling(name = 'GC_179',
                  value = '(ee**2*complex(0,1)*I66x66)/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_180 = Coupling(name = 'GC_180',
                  value = '(ee*complex(0,1)*G*I66x66*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_181 = Coupling(name = 'GC_181',
                  value = '-((ee*complex(0,1)*I92x11)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_182 = Coupling(name = 'GC_182',
                  value = '-((ee*complex(0,1)*I92x22)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_183 = Coupling(name = 'GC_183',
                  value = '-((ee*complex(0,1)*I92x33)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_184 = Coupling(name = 'GC_184',
                  value = '-((ee*complex(0,1)*I92x36)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_185 = Coupling(name = 'GC_185',
                  value = '-((ee*complex(0,1)*I92x63)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_186 = Coupling(name = 'GC_186',
                  value = '-((ee*complex(0,1)*I92x66)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_187 = Coupling(name = 'GC_187',
                  value = '-((ee*complex(0,1)*I93x11)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_188 = Coupling(name = 'GC_188',
                  value = '-((ee*complex(0,1)*I93x22)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_189 = Coupling(name = 'GC_189',
                  value = '-((ee*complex(0,1)*I93x33)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_190 = Coupling(name = 'GC_190',
                  value = '-((ee*complex(0,1)*I93x36)/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_191 = Coupling(name = 'GC_191',
                  value = '(ee*complex(0,1)*I94x11)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_192 = Coupling(name = 'GC_192',
                  value = '(ee*complex(0,1)*I94x22)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_193 = Coupling(name = 'GC_193',
                  value = '(ee*complex(0,1)*I94x33)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_194 = Coupling(name = 'GC_194',
                  value = '(ee*complex(0,1)*I94x36)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_195 = Coupling(name = 'GC_195',
                  value = '(ee*complex(0,1)*I94x63)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_196 = Coupling(name = 'GC_196',
                  value = '(ee*complex(0,1)*I94x66)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_197 = Coupling(name = 'GC_197',
                  value = '(ee*complex(0,1)*I95x11)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_198 = Coupling(name = 'GC_198',
                  value = '(ee*complex(0,1)*I95x22)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_199 = Coupling(name = 'GC_199',
                  value = '(ee*complex(0,1)*I95x33)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_200 = Coupling(name = 'GC_200',
                  value = '(ee*complex(0,1)*I95x36)/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_201 = Coupling(name = 'GC_201',
                  value = '(cw*ee**2*complex(0,1)*I92x11)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_202 = Coupling(name = 'GC_202',
                  value = '(cw*ee**2*complex(0,1)*I92x22)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_203 = Coupling(name = 'GC_203',
                  value = '(cw*ee**2*complex(0,1)*I92x33)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_204 = Coupling(name = 'GC_204',
                  value = '(cw*ee**2*complex(0,1)*I92x36)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_205 = Coupling(name = 'GC_205',
                  value = '(cw*ee**2*complex(0,1)*I92x63)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_206 = Coupling(name = 'GC_206',
                  value = '(cw*ee**2*complex(0,1)*I92x66)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_207 = Coupling(name = 'GC_207',
                  value = '-((cw*ee**2*complex(0,1)*I93x11)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_208 = Coupling(name = 'GC_208',
                  value = '-((cw*ee**2*complex(0,1)*I93x22)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_209 = Coupling(name = 'GC_209',
                  value = '-((cw*ee**2*complex(0,1)*I93x33)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_210 = Coupling(name = 'GC_210',
                  value = '-((cw*ee**2*complex(0,1)*I93x36)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_211 = Coupling(name = 'GC_211',
                  value = '(cw*ee**2*complex(0,1)*I94x11)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_212 = Coupling(name = 'GC_212',
                  value = '(cw*ee**2*complex(0,1)*I94x22)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_213 = Coupling(name = 'GC_213',
                  value = '(cw*ee**2*complex(0,1)*I94x33)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_214 = Coupling(name = 'GC_214',
                  value = '(cw*ee**2*complex(0,1)*I94x36)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_215 = Coupling(name = 'GC_215',
                  value = '(cw*ee**2*complex(0,1)*I94x63)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_216 = Coupling(name = 'GC_216',
                  value = '(cw*ee**2*complex(0,1)*I94x66)/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_217 = Coupling(name = 'GC_217',
                  value = '-((cw*ee**2*complex(0,1)*I95x11)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_218 = Coupling(name = 'GC_218',
                  value = '-((cw*ee**2*complex(0,1)*I95x22)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_219 = Coupling(name = 'GC_219',
                  value = '-((cw*ee**2*complex(0,1)*I95x33)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_220 = Coupling(name = 'GC_220',
                  value = '-((cw*ee**2*complex(0,1)*I95x36)/((-1 + sw)*(1 + sw)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_221 = Coupling(name = 'GC_221',
                  value = '(cw*ee*complex(0,1)*NN1x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_222 = Coupling(name = 'GC_222',
                  value = '(cw*ee*complex(0,1)*NN2x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_223 = Coupling(name = 'GC_223',
                  value = '(cw*ee*complex(0,1)*NN3x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_224 = Coupling(name = 'GC_224',
                  value = '(cw*ee*complex(0,1)*NN4x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_225 = Coupling(name = 'GC_225',
                  value = '(cw*ee*complex(0,1)*NN1x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_226 = Coupling(name = 'GC_226',
                  value = '(cw*ee*complex(0,1)*NN2x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_227 = Coupling(name = 'GC_227',
                  value = '(cw*ee*complex(0,1)*NN3x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_228 = Coupling(name = 'GC_228',
                  value = '(cw*ee*complex(0,1)*NN4x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_229 = Coupling(name = 'GC_229',
                  value = '(cw*ee*complex(0,1)*NN1x1*Rl4x4*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_230 = Coupling(name = 'GC_230',
                  value = '(cw*ee*complex(0,1)*NN2x1*Rl4x4*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_231 = Coupling(name = 'GC_231',
                  value = '(cw*ee*complex(0,1)*NN3x1*Rl4x4*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_232 = Coupling(name = 'GC_232',
                  value = '(cw*ee*complex(0,1)*NN4x1*Rl4x4*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_233 = Coupling(name = 'GC_233',
                  value = '(cw*ee*complex(0,1)*NN1x1*Rl5x5*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_234 = Coupling(name = 'GC_234',
                  value = '(cw*ee*complex(0,1)*NN2x1*Rl5x5*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_235 = Coupling(name = 'GC_235',
                  value = '(cw*ee*complex(0,1)*NN3x1*Rl5x5*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_236 = Coupling(name = 'GC_236',
                  value = '(cw*ee*complex(0,1)*NN4x1*Rl5x5*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_237 = Coupling(name = 'GC_237',
                  value = '(-2*cw*ee*complex(0,1)*NN1x1*Ru4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_238 = Coupling(name = 'GC_238',
                  value = '(-2*cw*ee*complex(0,1)*NN2x1*Ru4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_239 = Coupling(name = 'GC_239',
                  value = '(-2*cw*ee*complex(0,1)*NN3x1*Ru4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_240 = Coupling(name = 'GC_240',
                  value = '(-2*cw*ee*complex(0,1)*NN4x1*Ru4x4*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_241 = Coupling(name = 'GC_241',
                  value = '(-2*cw*ee*complex(0,1)*NN1x1*Ru5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_242 = Coupling(name = 'GC_242',
                  value = '(-2*cw*ee*complex(0,1)*NN2x1*Ru5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_243 = Coupling(name = 'GC_243',
                  value = '(-2*cw*ee*complex(0,1)*NN3x1*Ru5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_244 = Coupling(name = 'GC_244',
                  value = '(-2*cw*ee*complex(0,1)*NN4x1*Ru5x5*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_245 = Coupling(name = 'GC_245',
                  value = '-(ee**2*complex(0,1))/(2.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':2})

GC_246 = Coupling(name = 'GC_246',
                  value = '-(cw*ee*complex(0,1))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_247 = Coupling(name = 'GC_247',
                  value = '(cw*ee*complex(0,1))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_248 = Coupling(name = 'GC_248',
                  value = '-(cw*ee*complex(0,1)*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_249 = Coupling(name = 'GC_249',
                  value = '(2*cw*ee*complex(0,1)*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_250 = Coupling(name = 'GC_250',
                  value = '-((cw*ee*complex(0,1)*sw)/((-1 + sw)*(1 + sw)))',
                  order = {'QED':1})

GC_251 = Coupling(name = 'GC_251',
                  value = '(cw*ee*complex(0,1)*I100x44*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_252 = Coupling(name = 'GC_252',
                  value = '(cw*ee*complex(0,1)*I100x55*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_253 = Coupling(name = 'GC_253',
                  value = '(cw*ee*complex(0,1)*I101x44*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_254 = Coupling(name = 'GC_254',
                  value = '(cw*ee*complex(0,1)*I101x55*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_255 = Coupling(name = 'GC_255',
                  value = '(-2*cw*ee*complex(0,1)*I102x44*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_256 = Coupling(name = 'GC_256',
                  value = '(-2*cw*ee*complex(0,1)*I102x55*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_257 = Coupling(name = 'GC_257',
                  value = '(2*cw*ee**2*complex(0,1)*I26x44*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_258 = Coupling(name = 'GC_258',
                  value = '(2*cw*ee**2*complex(0,1)*I26x55*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_259 = Coupling(name = 'GC_259',
                  value = '(8*cw*ee**2*complex(0,1)*I52x44*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_260 = Coupling(name = 'GC_260',
                  value = '(4*cw*ee*complex(0,1)*G*I52x44*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_261 = Coupling(name = 'GC_261',
                  value = '(8*cw*ee**2*complex(0,1)*I52x55*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_262 = Coupling(name = 'GC_262',
                  value = '(4*cw*ee*complex(0,1)*G*I52x55*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_263 = Coupling(name = 'GC_263',
                  value = '(2*cw*ee**2*complex(0,1)*I6x44*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_264 = Coupling(name = 'GC_264',
                  value = '(-2*cw*ee*complex(0,1)*G*I6x44*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_265 = Coupling(name = 'GC_265',
                  value = '(2*cw*ee**2*complex(0,1)*I6x55*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_266 = Coupling(name = 'GC_266',
                  value = '(-2*cw*ee*complex(0,1)*G*I6x55*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_267 = Coupling(name = 'GC_267',
                  value = '(-2*ee**2*complex(0,1)*I100x44*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_268 = Coupling(name = 'GC_268',
                  value = '(-2*ee**2*complex(0,1)*I100x55*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_269 = Coupling(name = 'GC_269',
                  value = '(-2*ee**2*complex(0,1)*I101x44*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_270 = Coupling(name = 'GC_270',
                  value = '(-2*ee**2*complex(0,1)*I101x55*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_271 = Coupling(name = 'GC_271',
                  value = '(-8*ee**2*complex(0,1)*I102x44*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_272 = Coupling(name = 'GC_272',
                  value = '(-8*ee**2*complex(0,1)*I102x55*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_273 = Coupling(name = 'GC_273',
                  value = '-((cw*ee**2*complex(0,1)*I25x11)/((-1 + sw)*sw*(1 + sw))) + (2*cw*ee**2*complex(0,1)*I25x11*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_274 = Coupling(name = 'GC_274',
                  value = '-((cw*ee**2*complex(0,1)*I25x22)/((-1 + sw)*sw*(1 + sw))) + (2*cw*ee**2*complex(0,1)*I25x22*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_275 = Coupling(name = 'GC_275',
                  value = '-((cw*ee**2*complex(0,1)*I25x33)/((-1 + sw)*sw*(1 + sw))) + (2*cw*ee**2*complex(0,1)*I25x33*sw)/((-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I26x33*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_276 = Coupling(name = 'GC_276',
                  value = '-((cw*ee**2*complex(0,1)*I25x36)/((-1 + sw)*sw*(1 + sw))) + (2*cw*ee**2*complex(0,1)*I25x36*sw)/((-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I26x36*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_277 = Coupling(name = 'GC_277',
                  value = '-((cw*ee**2*complex(0,1)*I25x63)/((-1 + sw)*sw*(1 + sw))) + (2*cw*ee**2*complex(0,1)*I25x63*sw)/((-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I26x63*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_278 = Coupling(name = 'GC_278',
                  value = '-((cw*ee**2*complex(0,1)*I25x66)/((-1 + sw)*sw*(1 + sw))) + (2*cw*ee**2*complex(0,1)*I25x66*sw)/((-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I26x66*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_279 = Coupling(name = 'GC_279',
                  value = '(-2*cw*ee**2*complex(0,1)*I51x11)/(3.*(-1 + sw)*sw*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I51x11*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_280 = Coupling(name = 'GC_280',
                  value = '-((cw*ee*complex(0,1)*G*I51x11)/((-1 + sw)*sw*(1 + sw))) + (4*cw*ee*complex(0,1)*G*I51x11*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_281 = Coupling(name = 'GC_281',
                  value = '(-2*cw*ee**2*complex(0,1)*I51x22)/(3.*(-1 + sw)*sw*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I51x22*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_282 = Coupling(name = 'GC_282',
                  value = '-((cw*ee*complex(0,1)*G*I51x22)/((-1 + sw)*sw*(1 + sw))) + (4*cw*ee*complex(0,1)*G*I51x22*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_283 = Coupling(name = 'GC_283',
                  value = '(-2*cw*ee**2*complex(0,1)*I51x33)/(3.*(-1 + sw)*sw*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I51x33*sw)/(9.*(-1 + sw)*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I52x33*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_284 = Coupling(name = 'GC_284',
                  value = '-((cw*ee*complex(0,1)*G*I51x33)/((-1 + sw)*sw*(1 + sw))) + (4*cw*ee*complex(0,1)*G*I51x33*sw)/(3.*(-1 + sw)*(1 + sw)) + (4*cw*ee*complex(0,1)*G*I52x33*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_285 = Coupling(name = 'GC_285',
                  value = '(-2*cw*ee**2*complex(0,1)*I51x36)/(3.*(-1 + sw)*sw*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I51x36*sw)/(9.*(-1 + sw)*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I52x36*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_286 = Coupling(name = 'GC_286',
                  value = '-((cw*ee*complex(0,1)*G*I51x36)/((-1 + sw)*sw*(1 + sw))) + (4*cw*ee*complex(0,1)*G*I51x36*sw)/(3.*(-1 + sw)*(1 + sw)) + (4*cw*ee*complex(0,1)*G*I52x36*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_287 = Coupling(name = 'GC_287',
                  value = '(-2*cw*ee**2*complex(0,1)*I51x63)/(3.*(-1 + sw)*sw*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I51x63*sw)/(9.*(-1 + sw)*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I52x63*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_288 = Coupling(name = 'GC_288',
                  value = '-((cw*ee*complex(0,1)*G*I51x63)/((-1 + sw)*sw*(1 + sw))) + (4*cw*ee*complex(0,1)*G*I51x63*sw)/(3.*(-1 + sw)*(1 + sw)) + (4*cw*ee*complex(0,1)*G*I52x63*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_289 = Coupling(name = 'GC_289',
                  value = '(-2*cw*ee**2*complex(0,1)*I51x66)/(3.*(-1 + sw)*sw*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I51x66*sw)/(9.*(-1 + sw)*(1 + sw)) + (8*cw*ee**2*complex(0,1)*I52x66*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_290 = Coupling(name = 'GC_290',
                  value = '-((cw*ee*complex(0,1)*G*I51x66)/((-1 + sw)*sw*(1 + sw))) + (4*cw*ee*complex(0,1)*G*I51x66*sw)/(3.*(-1 + sw)*(1 + sw)) + (4*cw*ee*complex(0,1)*G*I52x66*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_291 = Coupling(name = 'GC_291',
                  value = '-(cw*ee**2*complex(0,1)*I5x11)/(3.*(-1 + sw)*sw*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I5x11*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_292 = Coupling(name = 'GC_292',
                  value = '(cw*ee*complex(0,1)*G*I5x11)/((-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I5x11*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_293 = Coupling(name = 'GC_293',
                  value = '-(cw*ee**2*complex(0,1)*I5x22)/(3.*(-1 + sw)*sw*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I5x22*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_294 = Coupling(name = 'GC_294',
                  value = '(cw*ee*complex(0,1)*G*I5x22)/((-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I5x22*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_295 = Coupling(name = 'GC_295',
                  value = '-(cw*ee**2*complex(0,1)*I5x33)/(3.*(-1 + sw)*sw*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I5x33*sw)/(9.*(-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I6x33*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_296 = Coupling(name = 'GC_296',
                  value = '(cw*ee*complex(0,1)*G*I5x33)/((-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I5x33*sw)/(3.*(-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I6x33*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_297 = Coupling(name = 'GC_297',
                  value = '-(cw*ee**2*complex(0,1)*I5x36)/(3.*(-1 + sw)*sw*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I5x36*sw)/(9.*(-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I6x36*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_298 = Coupling(name = 'GC_298',
                  value = '(cw*ee*complex(0,1)*G*I5x36)/((-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I5x36*sw)/(3.*(-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I6x36*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_299 = Coupling(name = 'GC_299',
                  value = '-(cw*ee**2*complex(0,1)*I5x63)/(3.*(-1 + sw)*sw*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I5x63*sw)/(9.*(-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I6x63*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_300 = Coupling(name = 'GC_300',
                  value = '(cw*ee*complex(0,1)*G*I5x63)/((-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I5x63*sw)/(3.*(-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I6x63*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_301 = Coupling(name = 'GC_301',
                  value = '-(cw*ee**2*complex(0,1)*I5x66)/(3.*(-1 + sw)*sw*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I5x66*sw)/(9.*(-1 + sw)*(1 + sw)) + (2*cw*ee**2*complex(0,1)*I6x66*sw)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_302 = Coupling(name = 'GC_302',
                  value = '(cw*ee*complex(0,1)*G*I5x66)/((-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I5x66*sw)/(3.*(-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*G*I6x66*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QCD':1,'QED':1})

GC_303 = Coupling(name = 'GC_303',
                  value = '-(cw*ee*complex(0,1)*I96x11)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I96x11*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_304 = Coupling(name = 'GC_304',
                  value = '-(cw*ee*complex(0,1)*I96x22)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I96x22*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_305 = Coupling(name = 'GC_305',
                  value = '-(cw*ee*complex(0,1)*I96x33)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I100x33*sw)/(3.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*I96x33*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_306 = Coupling(name = 'GC_306',
                  value = '-(cw*ee*complex(0,1)*I96x36)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I100x36*sw)/(3.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*I96x36*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_307 = Coupling(name = 'GC_307',
                  value = '(cw*ee*complex(0,1)*I96x63)/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*I100x63*sw)/(3.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*I96x63*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_308 = Coupling(name = 'GC_308',
                  value = '-(cw*ee*complex(0,1)*I96x66)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I100x66*sw)/(3.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*I96x66*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_309 = Coupling(name = 'GC_309',
                  value = '-(cw*ee*complex(0,1)*I97x11)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I97x11*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_310 = Coupling(name = 'GC_310',
                  value = '-(cw*ee*complex(0,1)*I97x22)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I97x22*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_311 = Coupling(name = 'GC_311',
                  value = '-(cw*ee*complex(0,1)*I97x33)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I101x33*sw)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*I97x33*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_312 = Coupling(name = 'GC_312',
                  value = '-(cw*ee*complex(0,1)*I97x36)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I101x36*sw)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*I97x36*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_313 = Coupling(name = 'GC_313',
                  value = '(cw*ee*complex(0,1)*I97x63)/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*I101x63*sw)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*I97x63*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_314 = Coupling(name = 'GC_314',
                  value = '-(cw*ee*complex(0,1)*I97x66)/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*I101x66*sw)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*I97x66*sw)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_315 = Coupling(name = 'GC_315',
                  value = '(cw*ee*complex(0,1)*I98x11)/(2.*(-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*I98x11*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_316 = Coupling(name = 'GC_316',
                  value = '(cw*ee*complex(0,1)*I98x22)/(2.*(-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*I98x22*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_317 = Coupling(name = 'GC_317',
                  value = '(cw*ee*complex(0,1)*I98x33)/(2.*(-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*I102x33*sw)/(3.*(-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*I98x33*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_318 = Coupling(name = 'GC_318',
                  value = '(cw*ee*complex(0,1)*I98x36)/(2.*(-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*I102x36*sw)/(3.*(-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*I98x36*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_319 = Coupling(name = 'GC_319',
                  value = '-(cw*ee*complex(0,1)*I98x63)/(2.*(-1 + sw)*sw*(1 + sw)) + (2*cw*ee*complex(0,1)*I102x63*sw)/(3.*(-1 + sw)*(1 + sw)) + (2*cw*ee*complex(0,1)*I98x63*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_320 = Coupling(name = 'GC_320',
                  value = '(cw*ee*complex(0,1)*I98x66)/(2.*(-1 + sw)*sw*(1 + sw)) - (2*cw*ee*complex(0,1)*I102x66*sw)/(3.*(-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*I98x66*sw)/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_321 = Coupling(name = 'GC_321',
                  value = '(2*ee**2*complex(0,1)*I96x11)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I96x11)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I96x11*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_322 = Coupling(name = 'GC_322',
                  value = '(2*ee**2*complex(0,1)*I96x22)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I96x22)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I96x22*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_323 = Coupling(name = 'GC_323',
                  value = '(2*ee**2*complex(0,1)*I96x33)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I96x33)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I100x33*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I96x33*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_324 = Coupling(name = 'GC_324',
                  value = '(2*ee**2*complex(0,1)*I96x36)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I96x36)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I100x36*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I96x36*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_325 = Coupling(name = 'GC_325',
                  value = '(2*ee**2*complex(0,1)*I96x63)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I96x63)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I100x63*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I96x63*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_326 = Coupling(name = 'GC_326',
                  value = '(2*ee**2*complex(0,1)*I96x66)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I96x66)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I100x66*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I96x66*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_327 = Coupling(name = 'GC_327',
                  value = '(2*ee**2*complex(0,1)*I97x11)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I97x11)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I97x11*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_328 = Coupling(name = 'GC_328',
                  value = '(2*ee**2*complex(0,1)*I97x22)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I97x22)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I97x22*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_329 = Coupling(name = 'GC_329',
                  value = '(2*ee**2*complex(0,1)*I97x33)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I97x33)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I101x33*sw**2)/((-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I97x33*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_330 = Coupling(name = 'GC_330',
                  value = '(2*ee**2*complex(0,1)*I97x36)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I97x36)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I101x36*sw**2)/((-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I97x36*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_331 = Coupling(name = 'GC_331',
                  value = '(2*ee**2*complex(0,1)*I97x63)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I97x63)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I101x63*sw**2)/((-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I97x63*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_332 = Coupling(name = 'GC_332',
                  value = '(2*ee**2*complex(0,1)*I97x66)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I97x66)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*I101x66*sw**2)/((-1 + sw)*(1 + sw)) - (2*ee**2*complex(0,1)*I97x66*sw**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_333 = Coupling(name = 'GC_333',
                  value = '(4*ee**2*complex(0,1)*I98x11)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I98x11)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (8*ee**2*complex(0,1)*I98x11*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_334 = Coupling(name = 'GC_334',
                  value = '(4*ee**2*complex(0,1)*I98x22)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I98x22)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (8*ee**2*complex(0,1)*I98x22*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_335 = Coupling(name = 'GC_335',
                  value = '(4*ee**2*complex(0,1)*I98x33)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I98x33)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (8*ee**2*complex(0,1)*I102x33*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (8*ee**2*complex(0,1)*I98x33*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_336 = Coupling(name = 'GC_336',
                  value = '(4*ee**2*complex(0,1)*I98x36)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I98x36)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (8*ee**2*complex(0,1)*I102x36*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (8*ee**2*complex(0,1)*I98x36*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_337 = Coupling(name = 'GC_337',
                  value = '(4*ee**2*complex(0,1)*I98x63)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I98x63)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (8*ee**2*complex(0,1)*I102x63*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (8*ee**2*complex(0,1)*I98x63*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_338 = Coupling(name = 'GC_338',
                  value = '(4*ee**2*complex(0,1)*I98x66)/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I98x66)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (8*ee**2*complex(0,1)*I102x66*sw**2)/(9.*(-1 + sw)*(1 + sw)) - (8*ee**2*complex(0,1)*I98x66*sw**2)/(9.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_339 = Coupling(name = 'GC_339',
                  value = '(complex(0,1)*I10x33*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x33*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*Rd3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_340 = Coupling(name = 'GC_340',
                  value = '(complex(0,1)*I10x36*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x36*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_341 = Coupling(name = 'GC_341',
                  value = '(complex(0,1)*I27x33*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x33*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*Rl3x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_342 = Coupling(name = 'GC_342',
                  value = '(complex(0,1)*I27x36*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x36*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*Rl6x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_343 = Coupling(name = 'GC_343',
                  value = '(complex(0,1)*I61x33*NN1x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x33*NN1x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN1x1*Ru3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_344 = Coupling(name = 'GC_344',
                  value = '(complex(0,1)*I61x36*NN1x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x36*NN1x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN1x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_345 = Coupling(name = 'GC_345',
                  value = '(complex(0,1)*I10x33*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x33*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*Rd3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_346 = Coupling(name = 'GC_346',
                  value = '(complex(0,1)*I10x36*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x36*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_347 = Coupling(name = 'GC_347',
                  value = '(complex(0,1)*I27x33*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x33*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*Rl3x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_348 = Coupling(name = 'GC_348',
                  value = '(complex(0,1)*I27x36*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x36*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*Rl6x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_349 = Coupling(name = 'GC_349',
                  value = '(complex(0,1)*I61x33*NN2x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x33*NN2x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN2x1*Ru3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_350 = Coupling(name = 'GC_350',
                  value = '(complex(0,1)*I61x36*NN2x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x36*NN2x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN2x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_351 = Coupling(name = 'GC_351',
                  value = '(complex(0,1)*I10x33*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x33*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*Rd3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_352 = Coupling(name = 'GC_352',
                  value = '(complex(0,1)*I10x36*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x36*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_353 = Coupling(name = 'GC_353',
                  value = '(complex(0,1)*I27x33*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x33*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*Rl3x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_354 = Coupling(name = 'GC_354',
                  value = '(complex(0,1)*I27x36*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x36*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*Rl6x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_355 = Coupling(name = 'GC_355',
                  value = '(complex(0,1)*I61x33*NN3x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x33*NN3x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN3x1*Ru3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_356 = Coupling(name = 'GC_356',
                  value = '(complex(0,1)*I61x36*NN3x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x36*NN3x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN3x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_357 = Coupling(name = 'GC_357',
                  value = '(complex(0,1)*I10x33*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x33*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*Rd3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_358 = Coupling(name = 'GC_358',
                  value = '(complex(0,1)*I10x36*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I10x36*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_359 = Coupling(name = 'GC_359',
                  value = '(complex(0,1)*I27x33*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x33*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*Rl3x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_360 = Coupling(name = 'GC_360',
                  value = '(complex(0,1)*I27x36*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I27x36*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*Rl6x6*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_361 = Coupling(name = 'GC_361',
                  value = '(complex(0,1)*I61x33*NN4x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x33*NN4x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN4x1*Ru3x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_362 = Coupling(name = 'GC_362',
                  value = '(complex(0,1)*I61x36*NN4x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I61x36*NN4x4*sw**2)/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*NN4x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_363 = Coupling(name = 'GC_363',
                  value = '-((ee*complex(0,1)*I82x11*UU1x1)/sw)',
                  order = {'QED':1})

GC_364 = Coupling(name = 'GC_364',
                  value = '-((ee*complex(0,1)*I82x22*UU1x1)/sw)',
                  order = {'QED':1})

GC_365 = Coupling(name = 'GC_365',
                  value = '-((ee*complex(0,1)*I85x11*UU1x1)/sw)',
                  order = {'QED':1})

GC_366 = Coupling(name = 'GC_366',
                  value = '-((ee*complex(0,1)*I85x22*UU1x1)/sw)',
                  order = {'QED':1})

GC_367 = Coupling(name = 'GC_367',
                  value = 'complex(0,1)*I44x33*UU1x2',
                  order = {'QED':1})

GC_368 = Coupling(name = 'GC_368',
                  value = 'complex(0,1)*I64x33*UU1x2',
                  order = {'QED':1})

GC_369 = Coupling(name = 'GC_369',
                  value = 'complex(0,1)*I64x36*UU1x2',
                  order = {'QED':1})

GC_370 = Coupling(name = 'GC_370',
                  value = '-((ee*complex(0,1)*I82x33*UU1x1)/sw) + complex(0,1)*I83x33*UU1x2',
                  order = {'QED':1})

GC_371 = Coupling(name = 'GC_371',
                  value = '-((ee*complex(0,1)*I82x36*UU1x1)/sw) + complex(0,1)*I83x36*UU1x2',
                  order = {'QED':1})

GC_372 = Coupling(name = 'GC_372',
                  value = '-((ee*complex(0,1)*I85x33*UU1x1)/sw) + complex(0,1)*I86x33*UU1x2',
                  order = {'QED':1})

GC_373 = Coupling(name = 'GC_373',
                  value = '-((ee*complex(0,1)*I85x36*UU1x1)/sw) + complex(0,1)*I86x36*UU1x2',
                  order = {'QED':1})

GC_374 = Coupling(name = 'GC_374',
                  value = '-((ee*complex(0,1)*I82x11*UU2x1)/sw)',
                  order = {'QED':1})

GC_375 = Coupling(name = 'GC_375',
                  value = '-((ee*complex(0,1)*I82x22*UU2x1)/sw)',
                  order = {'QED':1})

GC_376 = Coupling(name = 'GC_376',
                  value = '-((ee*complex(0,1)*I85x11*UU2x1)/sw)',
                  order = {'QED':1})

GC_377 = Coupling(name = 'GC_377',
                  value = '-((ee*complex(0,1)*I85x22*UU2x1)/sw)',
                  order = {'QED':1})

GC_378 = Coupling(name = 'GC_378',
                  value = 'complex(0,1)*I44x33*UU2x2',
                  order = {'QED':1})

GC_379 = Coupling(name = 'GC_379',
                  value = 'complex(0,1)*I64x33*UU2x2',
                  order = {'QED':1})

GC_380 = Coupling(name = 'GC_380',
                  value = 'complex(0,1)*I64x36*UU2x2',
                  order = {'QED':1})

GC_381 = Coupling(name = 'GC_381',
                  value = '-((ee*complex(0,1)*I82x33*UU2x1)/sw) + complex(0,1)*I83x33*UU2x2',
                  order = {'QED':1})

GC_382 = Coupling(name = 'GC_382',
                  value = '-((ee*complex(0,1)*I82x36*UU2x1)/sw) + complex(0,1)*I83x36*UU2x2',
                  order = {'QED':1})

GC_383 = Coupling(name = 'GC_383',
                  value = '-((ee*complex(0,1)*I85x33*UU2x1)/sw) + complex(0,1)*I86x33*UU2x2',
                  order = {'QED':1})

GC_384 = Coupling(name = 'GC_384',
                  value = '-((ee*complex(0,1)*I85x36*UU2x1)/sw) + complex(0,1)*I86x36*UU2x2',
                  order = {'QED':1})

GC_385 = Coupling(name = 'GC_385',
                  value = '-((ee*complex(0,1)*I87x11*VV1x1)/sw)',
                  order = {'QED':1})

GC_386 = Coupling(name = 'GC_386',
                  value = '-((ee*complex(0,1)*I87x22*VV1x1)/sw)',
                  order = {'QED':1})

GC_387 = Coupling(name = 'GC_387',
                  value = '-((ee*complex(0,1)*I87x33*VV1x1)/sw)',
                  order = {'QED':1})

GC_388 = Coupling(name = 'GC_388',
                  value = '-((ee*complex(0,1)*I89x11*VV1x1)/sw)',
                  order = {'QED':1})

GC_389 = Coupling(name = 'GC_389',
                  value = '-((ee*complex(0,1)*I89x22*VV1x1)/sw)',
                  order = {'QED':1})

GC_390 = Coupling(name = 'GC_390',
                  value = 'complex(0,1)*I8x33*VV1x2',
                  order = {'QED':1})

GC_391 = Coupling(name = 'GC_391',
                  value = 'complex(0,1)*I8x36*VV1x2',
                  order = {'QED':1})

GC_392 = Coupling(name = 'GC_392',
                  value = '-((ee*complex(0,1)*I89x33*VV1x1)/sw) + complex(0,1)*I90x33*VV1x2',
                  order = {'QED':1})

GC_393 = Coupling(name = 'GC_393',
                  value = '-((ee*complex(0,1)*I89x36*VV1x1)/sw) + complex(0,1)*I90x36*VV1x2',
                  order = {'QED':1})

GC_394 = Coupling(name = 'GC_394',
                  value = '-((ee*complex(0,1)*I87x11*VV2x1)/sw)',
                  order = {'QED':1})

GC_395 = Coupling(name = 'GC_395',
                  value = '-((ee*complex(0,1)*I87x22*VV2x1)/sw)',
                  order = {'QED':1})

GC_396 = Coupling(name = 'GC_396',
                  value = '-((ee*complex(0,1)*I87x33*VV2x1)/sw)',
                  order = {'QED':1})

GC_397 = Coupling(name = 'GC_397',
                  value = '-((ee*complex(0,1)*I89x11*VV2x1)/sw)',
                  order = {'QED':1})

GC_398 = Coupling(name = 'GC_398',
                  value = '-((ee*complex(0,1)*I89x22*VV2x1)/sw)',
                  order = {'QED':1})

GC_399 = Coupling(name = 'GC_399',
                  value = 'complex(0,1)*I8x33*VV2x2',
                  order = {'QED':1})

GC_400 = Coupling(name = 'GC_400',
                  value = 'complex(0,1)*I8x36*VV2x2',
                  order = {'QED':1})

GC_401 = Coupling(name = 'GC_401',
                  value = '-((ee*complex(0,1)*I89x33*VV2x1)/sw) + complex(0,1)*I90x33*VV2x2',
                  order = {'QED':1})

GC_402 = Coupling(name = 'GC_402',
                  value = '-((ee*complex(0,1)*I89x36*VV2x1)/sw) + complex(0,1)*I90x36*VV2x2',
                  order = {'QED':1})

GC_403 = Coupling(name = 'GC_403',
                  value = '(ee*complex(0,1)*complexconjugate(CKM1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_404 = Coupling(name = 'GC_404',
                  value = '(ee*complex(0,1)*complexconjugate(CKM2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_405 = Coupling(name = 'GC_405',
                  value = '(ee*complex(0,1)*complexconjugate(CKM3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_406 = Coupling(name = 'GC_406',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_407 = Coupling(name = 'GC_407',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_408 = Coupling(name = 'GC_408',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_409 = Coupling(name = 'GC_409',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_410 = Coupling(name = 'GC_410',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_411 = Coupling(name = 'GC_411',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_412 = Coupling(name = 'GC_412',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_413 = Coupling(name = 'GC_413',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_414 = Coupling(name = 'GC_414',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_415 = Coupling(name = 'GC_415',
                  value = '(complex(0,1)*I11x33*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x33*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_416 = Coupling(name = 'GC_416',
                  value = '(complex(0,1)*I11x36*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x36*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd6x3*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd6x3*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd6x3*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_417 = Coupling(name = 'GC_417',
                  value = '(complex(0,1)*I28x33*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x33*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_418 = Coupling(name = 'GC_418',
                  value = '(complex(0,1)*I28x36*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x36*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl6x3*complexconjugate(NN1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl6x3*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl6x3*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_419 = Coupling(name = 'GC_419',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN1x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN1x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_420 = Coupling(name = 'GC_420',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN1x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN1x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_421 = Coupling(name = 'GC_421',
                  value = '-(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN1x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN1x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_422 = Coupling(name = 'GC_422',
                  value = '(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN1x3))/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN1x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_423 = Coupling(name = 'GC_423',
                  value = '(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN1x3))/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN1x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_424 = Coupling(name = 'GC_424',
                  value = '(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN1x3))/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN1x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_425 = Coupling(name = 'GC_425',
                  value = '(complex(0,1)*I62x33*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x33*sw**2*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_426 = Coupling(name = 'GC_426',
                  value = '(complex(0,1)*I62x36*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x36*sw**2*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru6x3*complexconjugate(NN1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru6x3*complexconjugate(NN1x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru6x3*sw*complexconjugate(NN1x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_427 = Coupling(name = 'GC_427',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN1x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN1x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_428 = Coupling(name = 'GC_428',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN1x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN1x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_429 = Coupling(name = 'GC_429',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_430 = Coupling(name = 'GC_430',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_431 = Coupling(name = 'GC_431',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN2x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_432 = Coupling(name = 'GC_432',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN2x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_433 = Coupling(name = 'GC_433',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN2x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_434 = Coupling(name = 'GC_434',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN2x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_435 = Coupling(name = 'GC_435',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN2x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_436 = Coupling(name = 'GC_436',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_437 = Coupling(name = 'GC_437',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_438 = Coupling(name = 'GC_438',
                  value = '(complex(0,1)*I11x33*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x33*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_439 = Coupling(name = 'GC_439',
                  value = '(complex(0,1)*I11x36*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x36*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd6x3*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd6x3*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd6x3*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_440 = Coupling(name = 'GC_440',
                  value = '(complex(0,1)*I28x33*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x33*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN2x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_441 = Coupling(name = 'GC_441',
                  value = '(complex(0,1)*I28x36*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x36*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl6x3*complexconjugate(NN2x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl6x3*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl6x3*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_442 = Coupling(name = 'GC_442',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN2x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN2x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_443 = Coupling(name = 'GC_443',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN2x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN2x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_444 = Coupling(name = 'GC_444',
                  value = '-(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN2x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN2x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_445 = Coupling(name = 'GC_445',
                  value = '-(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN2x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN2x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_446 = Coupling(name = 'GC_446',
                  value = '(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN2x3))/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN2x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_447 = Coupling(name = 'GC_447',
                  value = '(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN2x3))/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN2x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_448 = Coupling(name = 'GC_448',
                  value = '(complex(0,1)*I62x33*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x33*sw**2*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_449 = Coupling(name = 'GC_449',
                  value = '(complex(0,1)*I62x36*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x36*sw**2*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru6x3*complexconjugate(NN2x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru6x3*complexconjugate(NN2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru6x3*sw*complexconjugate(NN2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_450 = Coupling(name = 'GC_450',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN2x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN2x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_451 = Coupling(name = 'GC_451',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN2x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN2x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_452 = Coupling(name = 'GC_452',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_453 = Coupling(name = 'GC_453',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_454 = Coupling(name = 'GC_454',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN3x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_455 = Coupling(name = 'GC_455',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN3x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_456 = Coupling(name = 'GC_456',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN3x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_457 = Coupling(name = 'GC_457',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN3x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_458 = Coupling(name = 'GC_458',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN3x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_459 = Coupling(name = 'GC_459',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_460 = Coupling(name = 'GC_460',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_461 = Coupling(name = 'GC_461',
                  value = '(complex(0,1)*I11x33*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x33*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_462 = Coupling(name = 'GC_462',
                  value = '(complex(0,1)*I11x36*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x36*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd6x3*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd6x3*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd6x3*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_463 = Coupling(name = 'GC_463',
                  value = '(complex(0,1)*I28x33*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x33*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN3x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_464 = Coupling(name = 'GC_464',
                  value = '(complex(0,1)*I28x36*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x36*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl6x3*complexconjugate(NN3x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl6x3*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl6x3*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_465 = Coupling(name = 'GC_465',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN3x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_466 = Coupling(name = 'GC_466',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN3x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_467 = Coupling(name = 'GC_467',
                  value = '-(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN3x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN3x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_468 = Coupling(name = 'GC_468',
                  value = '-(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN3x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN3x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_469 = Coupling(name = 'GC_469',
                  value = '-(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN3x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN3x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_470 = Coupling(name = 'GC_470',
                  value = '(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN3x3))/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN3x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_471 = Coupling(name = 'GC_471',
                  value = '(complex(0,1)*I62x33*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x33*sw**2*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_472 = Coupling(name = 'GC_472',
                  value = '(complex(0,1)*I62x36*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x36*sw**2*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru6x3*complexconjugate(NN3x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru6x3*complexconjugate(NN3x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru6x3*sw*complexconjugate(NN3x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_473 = Coupling(name = 'GC_473',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN3x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN3x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_474 = Coupling(name = 'GC_474',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN3x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN3x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_475 = Coupling(name = 'GC_475',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_476 = Coupling(name = 'GC_476',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_477 = Coupling(name = 'GC_477',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN4x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_478 = Coupling(name = 'GC_478',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN4x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_479 = Coupling(name = 'GC_479',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN4x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_480 = Coupling(name = 'GC_480',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN4x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_481 = Coupling(name = 'GC_481',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN4x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_482 = Coupling(name = 'GC_482',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_483 = Coupling(name = 'GC_483',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_484 = Coupling(name = 'GC_484',
                  value = '(complex(0,1)*I11x33*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x33*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_485 = Coupling(name = 'GC_485',
                  value = '(complex(0,1)*I11x36*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I11x36*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Rd6x3*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd6x3*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd6x3*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_486 = Coupling(name = 'GC_486',
                  value = '(complex(0,1)*I28x33*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x33*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN4x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_487 = Coupling(name = 'GC_487',
                  value = '(complex(0,1)*I28x36*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I28x36*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*Rl6x3*complexconjugate(NN4x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Rl6x3*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl6x3*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_488 = Coupling(name = 'GC_488',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN4x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN4x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_489 = Coupling(name = 'GC_489',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN4x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN4x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_490 = Coupling(name = 'GC_490',
                  value = '-(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN4x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN4x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_491 = Coupling(name = 'GC_491',
                  value = '-(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN4x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN4x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_492 = Coupling(name = 'GC_492',
                  value = '-(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN4x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN4x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_493 = Coupling(name = 'GC_493',
                  value = '-(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN4x3))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN4x4))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_494 = Coupling(name = 'GC_494',
                  value = '(complex(0,1)*I62x33*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x33*sw**2*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_495 = Coupling(name = 'GC_495',
                  value = '(complex(0,1)*I62x36*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I62x36*sw**2*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*Ru6x3*complexconjugate(NN4x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru6x3*complexconjugate(NN4x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru6x3*sw*complexconjugate(NN4x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_496 = Coupling(name = 'GC_496',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN4x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN4x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_497 = Coupling(name = 'GC_497',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN4x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN4x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_498 = Coupling(name = 'GC_498',
                  value = '-(complex(0,1)*G*complexconjugate(Rd1x1)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_499 = Coupling(name = 'GC_499',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Rd1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rd1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rd1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_500 = Coupling(name = 'GC_500',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Rd1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rd1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rd1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_501 = Coupling(name = 'GC_501',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Rd1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rd1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rd1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_502 = Coupling(name = 'GC_502',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Rd1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rd1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rd1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_503 = Coupling(name = 'GC_503',
                  value = '-(complex(0,1)*G*complexconjugate(Rd2x2)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_504 = Coupling(name = 'GC_504',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Rd2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rd2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rd2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_505 = Coupling(name = 'GC_505',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Rd2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rd2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rd2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_506 = Coupling(name = 'GC_506',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Rd2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rd2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rd2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_507 = Coupling(name = 'GC_507',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Rd2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rd2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rd2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_508 = Coupling(name = 'GC_508',
                  value = '-(complex(0,1)*G*complexconjugate(Rd3x3)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_509 = Coupling(name = 'GC_509',
                  value = '(complex(0,1)*I3x33*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x33*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*complexconjugate(Rd3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rd3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rd3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_510 = Coupling(name = 'GC_510',
                  value = '(complex(0,1)*I3x33*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x33*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*complexconjugate(Rd3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rd3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rd3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_511 = Coupling(name = 'GC_511',
                  value = '(complex(0,1)*I3x33*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x33*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*complexconjugate(Rd3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rd3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rd3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_512 = Coupling(name = 'GC_512',
                  value = '(complex(0,1)*I3x33*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x33*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*complexconjugate(Rd3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rd3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rd3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_513 = Coupling(name = 'GC_513',
                  value = 'complex(0,1)*G*complexconjugate(Rd3x6)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_514 = Coupling(name = 'GC_514',
                  value = '(complex(0,1)*I4x33*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x33*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rd3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_515 = Coupling(name = 'GC_515',
                  value = '(complex(0,1)*I4x33*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x33*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rd3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_516 = Coupling(name = 'GC_516',
                  value = '(complex(0,1)*I4x33*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x33*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rd3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_517 = Coupling(name = 'GC_517',
                  value = '(complex(0,1)*I4x33*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x33*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rd3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_518 = Coupling(name = 'GC_518',
                  value = 'complex(0,1)*G*complexconjugate(Rd4x4)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_519 = Coupling(name = 'GC_519',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_520 = Coupling(name = 'GC_520',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_521 = Coupling(name = 'GC_521',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_522 = Coupling(name = 'GC_522',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_523 = Coupling(name = 'GC_523',
                  value = 'complex(0,1)*G*complexconjugate(Rd5x5)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_524 = Coupling(name = 'GC_524',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_525 = Coupling(name = 'GC_525',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_526 = Coupling(name = 'GC_526',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_527 = Coupling(name = 'GC_527',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_528 = Coupling(name = 'GC_528',
                  value = '-(complex(0,1)*G*complexconjugate(Rd6x3)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_529 = Coupling(name = 'GC_529',
                  value = '(complex(0,1)*I3x36*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x36*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*complexconjugate(Rd6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rd6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rd6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_530 = Coupling(name = 'GC_530',
                  value = '(complex(0,1)*I3x36*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x36*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*complexconjugate(Rd6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rd6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rd6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_531 = Coupling(name = 'GC_531',
                  value = '(complex(0,1)*I3x36*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x36*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*complexconjugate(Rd6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rd6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rd6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_532 = Coupling(name = 'GC_532',
                  value = '(complex(0,1)*I3x36*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I3x36*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*complexconjugate(Rd6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rd6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rd6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_533 = Coupling(name = 'GC_533',
                  value = 'complex(0,1)*G*complexconjugate(Rd6x6)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_534 = Coupling(name = 'GC_534',
                  value = '(complex(0,1)*I4x36*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x36*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_535 = Coupling(name = 'GC_535',
                  value = '(complex(0,1)*I4x36*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x36*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_536 = Coupling(name = 'GC_536',
                  value = '(complex(0,1)*I4x36*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x36*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_537 = Coupling(name = 'GC_537',
                  value = '(complex(0,1)*I4x36*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I4x36*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_538 = Coupling(name = 'GC_538',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN1x2*complexconjugate(Rl1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_539 = Coupling(name = 'GC_539',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN2x2*complexconjugate(Rl1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_540 = Coupling(name = 'GC_540',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN3x2*complexconjugate(Rl1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_541 = Coupling(name = 'GC_541',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN4x2*complexconjugate(Rl1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rl1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_542 = Coupling(name = 'GC_542',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN1x2*complexconjugate(Rl2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_543 = Coupling(name = 'GC_543',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN2x2*complexconjugate(Rl2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_544 = Coupling(name = 'GC_544',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN3x2*complexconjugate(Rl2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_545 = Coupling(name = 'GC_545',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) - (ee*complex(0,1)*NN4x2*complexconjugate(Rl2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rl2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_546 = Coupling(name = 'GC_546',
                  value = '(complex(0,1)*I23x33*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x33*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN1x1*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rl3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_547 = Coupling(name = 'GC_547',
                  value = '(complex(0,1)*I23x33*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x33*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN2x1*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rl3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_548 = Coupling(name = 'GC_548',
                  value = '(complex(0,1)*I23x33*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x33*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN3x1*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rl3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_549 = Coupling(name = 'GC_549',
                  value = '(complex(0,1)*I23x33*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x33*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN4x1*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rl3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rl3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_550 = Coupling(name = 'GC_550',
                  value = '(complex(0,1)*I24x33*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x33*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rl3x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_551 = Coupling(name = 'GC_551',
                  value = '(complex(0,1)*I24x33*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x33*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rl3x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_552 = Coupling(name = 'GC_552',
                  value = '(complex(0,1)*I24x33*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x33*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rl3x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_553 = Coupling(name = 'GC_553',
                  value = '(complex(0,1)*I24x33*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x33*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rl3x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_554 = Coupling(name = 'GC_554',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_555 = Coupling(name = 'GC_555',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_556 = Coupling(name = 'GC_556',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_557 = Coupling(name = 'GC_557',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_558 = Coupling(name = 'GC_558',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_559 = Coupling(name = 'GC_559',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_560 = Coupling(name = 'GC_560',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_561 = Coupling(name = 'GC_561',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_562 = Coupling(name = 'GC_562',
                  value = '(complex(0,1)*I23x36*NN1x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x36*NN1x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN1x1*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rl6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_563 = Coupling(name = 'GC_563',
                  value = '(complex(0,1)*I23x36*NN2x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x36*NN2x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN2x1*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rl6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_564 = Coupling(name = 'GC_564',
                  value = '(complex(0,1)*I23x36*NN3x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x36*NN3x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN3x1*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rl6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_565 = Coupling(name = 'GC_565',
                  value = '(complex(0,1)*I23x36*NN4x3)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I23x36*NN4x3*sw**2)/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN4x1*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rl6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rl6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_566 = Coupling(name = 'GC_566',
                  value = '(complex(0,1)*I24x36*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x36*sw**2*complexconjugate(NN1x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_567 = Coupling(name = 'GC_567',
                  value = '(complex(0,1)*I24x36*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x36*sw**2*complexconjugate(NN2x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_568 = Coupling(name = 'GC_568',
                  value = '(complex(0,1)*I24x36*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x36*sw**2*complexconjugate(NN3x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_569 = Coupling(name = 'GC_569',
                  value = '(complex(0,1)*I24x36*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I24x36*sw**2*complexconjugate(NN4x3))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_570 = Coupling(name = 'GC_570',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN1x2*complexconjugate(Rn1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_571 = Coupling(name = 'GC_571',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN2x2*complexconjugate(Rn1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_572 = Coupling(name = 'GC_572',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN3x2*complexconjugate(Rn1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_573 = Coupling(name = 'GC_573',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN4x2*complexconjugate(Rn1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rn1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_574 = Coupling(name = 'GC_574',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN1x2*complexconjugate(Rn2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_575 = Coupling(name = 'GC_575',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN2x2*complexconjugate(Rn2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_576 = Coupling(name = 'GC_576',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN3x2*complexconjugate(Rn2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_577 = Coupling(name = 'GC_577',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN4x2*complexconjugate(Rn2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rn2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_578 = Coupling(name = 'GC_578',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN1x2*complexconjugate(Rn3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_579 = Coupling(name = 'GC_579',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN2x2*complexconjugate(Rn3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_580 = Coupling(name = 'GC_580',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN3x2*complexconjugate(Rn3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_581 = Coupling(name = 'GC_581',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))) + (ee*complex(0,1)*NN4x2*complexconjugate(Rn3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rn3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_582 = Coupling(name = 'GC_582',
                  value = '-(complex(0,1)*G*complexconjugate(Ru1x1)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_583 = Coupling(name = 'GC_583',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Ru1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*complexconjugate(Ru1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Ru1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_584 = Coupling(name = 'GC_584',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Ru1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*complexconjugate(Ru1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Ru1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_585 = Coupling(name = 'GC_585',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Ru1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*complexconjugate(Ru1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Ru1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_586 = Coupling(name = 'GC_586',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Ru1x1))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*complexconjugate(Ru1x1))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Ru1x1))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_587 = Coupling(name = 'GC_587',
                  value = '-(complex(0,1)*G*complexconjugate(Ru2x2)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_588 = Coupling(name = 'GC_588',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Ru2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*complexconjugate(Ru2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Ru2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_589 = Coupling(name = 'GC_589',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Ru2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*complexconjugate(Ru2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Ru2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_590 = Coupling(name = 'GC_590',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Ru2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*complexconjugate(Ru2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Ru2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_591 = Coupling(name = 'GC_591',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Ru2x2))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*complexconjugate(Ru2x2))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Ru2x2))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_592 = Coupling(name = 'GC_592',
                  value = '-(complex(0,1)*G*complexconjugate(Ru3x3)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_593 = Coupling(name = 'GC_593',
                  value = '(complex(0,1)*I49x33*NN1x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x33*NN1x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*complexconjugate(Ru3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*complexconjugate(Ru3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Ru3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_594 = Coupling(name = 'GC_594',
                  value = '(complex(0,1)*I49x33*NN2x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x33*NN2x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*complexconjugate(Ru3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*complexconjugate(Ru3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Ru3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_595 = Coupling(name = 'GC_595',
                  value = '(complex(0,1)*I49x33*NN3x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x33*NN3x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*complexconjugate(Ru3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*complexconjugate(Ru3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Ru3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_596 = Coupling(name = 'GC_596',
                  value = '(complex(0,1)*I49x33*NN4x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x33*NN4x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*complexconjugate(Ru3x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*complexconjugate(Ru3x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Ru3x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_597 = Coupling(name = 'GC_597',
                  value = 'complex(0,1)*G*complexconjugate(Ru3x6)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_598 = Coupling(name = 'GC_598',
                  value = '(complex(0,1)*I50x33*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x33*sw**2*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Ru3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_599 = Coupling(name = 'GC_599',
                  value = '(complex(0,1)*I50x33*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x33*sw**2*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Ru3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_600 = Coupling(name = 'GC_600',
                  value = '(complex(0,1)*I50x33*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x33*sw**2*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Ru3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_601 = Coupling(name = 'GC_601',
                  value = '(complex(0,1)*I50x33*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x33*sw**2*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Ru3x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_602 = Coupling(name = 'GC_602',
                  value = 'complex(0,1)*G*complexconjugate(Ru4x4)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_603 = Coupling(name = 'GC_603',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_604 = Coupling(name = 'GC_604',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_605 = Coupling(name = 'GC_605',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_606 = Coupling(name = 'GC_606',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_607 = Coupling(name = 'GC_607',
                  value = 'complex(0,1)*G*complexconjugate(Ru5x5)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_608 = Coupling(name = 'GC_608',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_609 = Coupling(name = 'GC_609',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_610 = Coupling(name = 'GC_610',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_611 = Coupling(name = 'GC_611',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_612 = Coupling(name = 'GC_612',
                  value = '-(complex(0,1)*G*complexconjugate(Ru6x3)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_613 = Coupling(name = 'GC_613',
                  value = '(complex(0,1)*I49x36*NN1x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x36*NN1x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*complexconjugate(Ru6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*complexconjugate(Ru6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Ru6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_614 = Coupling(name = 'GC_614',
                  value = '(complex(0,1)*I49x36*NN2x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x36*NN2x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*complexconjugate(Ru6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*complexconjugate(Ru6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Ru6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_615 = Coupling(name = 'GC_615',
                  value = '(complex(0,1)*I49x36*NN3x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x36*NN3x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*complexconjugate(Ru6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*complexconjugate(Ru6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Ru6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_616 = Coupling(name = 'GC_616',
                  value = '(complex(0,1)*I49x36*NN4x4)/((-1 + sw)*(1 + sw)) - (complex(0,1)*I49x36*NN4x4*sw**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*complexconjugate(Ru6x3))/(3.*(-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*complexconjugate(Ru6x3))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Ru6x3))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_617 = Coupling(name = 'GC_617',
                  value = 'complex(0,1)*G*complexconjugate(Ru6x6)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_618 = Coupling(name = 'GC_618',
                  value = '(complex(0,1)*I50x36*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x36*sw**2*complexconjugate(NN1x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_619 = Coupling(name = 'GC_619',
                  value = '(complex(0,1)*I50x36*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x36*sw**2*complexconjugate(NN2x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_620 = Coupling(name = 'GC_620',
                  value = '(complex(0,1)*I50x36*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x36*sw**2*complexconjugate(NN3x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_621 = Coupling(name = 'GC_621',
                  value = '(complex(0,1)*I50x36*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I50x36*sw**2*complexconjugate(NN4x4))/((-1 + sw)*(1 + sw)) - (2*cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_622 = Coupling(name = 'GC_622',
                  value = '-((ee*complex(0,1)*I29x11*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_623 = Coupling(name = 'GC_623',
                  value = '-((ee*complex(0,1)*I29x22*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_624 = Coupling(name = 'GC_624',
                  value = '-((ee*complex(0,1)*I7x11*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_625 = Coupling(name = 'GC_625',
                  value = '-((ee*complex(0,1)*I7x22*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_626 = Coupling(name = 'GC_626',
                  value = 'complex(0,1)*I88x33*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_627 = Coupling(name = 'GC_627',
                  value = 'complex(0,1)*I91x33*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_628 = Coupling(name = 'GC_628',
                  value = 'complex(0,1)*I91x36*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_629 = Coupling(name = 'GC_629',
                  value = '-((ee*complex(0,1)*I29x33*complexconjugate(UU1x1))/sw) + complex(0,1)*I30x33*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_630 = Coupling(name = 'GC_630',
                  value = '-((ee*complex(0,1)*I29x36*complexconjugate(UU1x1))/sw) + complex(0,1)*I30x36*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_631 = Coupling(name = 'GC_631',
                  value = '-((ee*complex(0,1)*I7x33*complexconjugate(UU1x1))/sw) + complex(0,1)*I9x33*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_632 = Coupling(name = 'GC_632',
                  value = '-((ee*complex(0,1)*I7x36*complexconjugate(UU1x1))/sw) + complex(0,1)*I9x36*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_633 = Coupling(name = 'GC_633',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN1x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_634 = Coupling(name = 'GC_634',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN2x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_635 = Coupling(name = 'GC_635',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN3x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_636 = Coupling(name = 'GC_636',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN4x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_637 = Coupling(name = 'GC_637',
                  value = 'ee*complex(0,1)*UU1x1*complexconjugate(UU1x1) + ee*complex(0,1)*UU1x2*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_638 = Coupling(name = 'GC_638',
                  value = '-((cw*ee*complex(0,1)*UU1x1*complexconjugate(UU1x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*UU1x1*complexconjugate(UU1x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*UU1x2*complexconjugate(UU1x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*UU1x2*complexconjugate(UU1x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_639 = Coupling(name = 'GC_639',
                  value = 'ee*complex(0,1)*UU2x1*complexconjugate(UU1x1) + ee*complex(0,1)*UU2x2*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_640 = Coupling(name = 'GC_640',
                  value = '-((cw*ee*complex(0,1)*UU2x1*complexconjugate(UU1x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*UU2x1*complexconjugate(UU1x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*UU2x2*complexconjugate(UU1x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*UU2x2*complexconjugate(UU1x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_641 = Coupling(name = 'GC_641',
                  value = '-((ee*complex(0,1)*I29x11*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_642 = Coupling(name = 'GC_642',
                  value = '-((ee*complex(0,1)*I29x22*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_643 = Coupling(name = 'GC_643',
                  value = '-((ee*complex(0,1)*I7x11*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_644 = Coupling(name = 'GC_644',
                  value = '-((ee*complex(0,1)*I7x22*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_645 = Coupling(name = 'GC_645',
                  value = 'complex(0,1)*I88x33*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_646 = Coupling(name = 'GC_646',
                  value = 'complex(0,1)*I91x33*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_647 = Coupling(name = 'GC_647',
                  value = 'complex(0,1)*I91x36*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_648 = Coupling(name = 'GC_648',
                  value = '-((ee*complex(0,1)*I29x33*complexconjugate(UU2x1))/sw) + complex(0,1)*I30x33*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_649 = Coupling(name = 'GC_649',
                  value = '-((ee*complex(0,1)*I29x36*complexconjugate(UU2x1))/sw) + complex(0,1)*I30x36*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_650 = Coupling(name = 'GC_650',
                  value = '-((ee*complex(0,1)*I7x33*complexconjugate(UU2x1))/sw) + complex(0,1)*I9x33*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_651 = Coupling(name = 'GC_651',
                  value = '-((ee*complex(0,1)*I7x36*complexconjugate(UU2x1))/sw) + complex(0,1)*I9x36*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_652 = Coupling(name = 'GC_652',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN1x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_653 = Coupling(name = 'GC_653',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN2x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_654 = Coupling(name = 'GC_654',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN3x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_655 = Coupling(name = 'GC_655',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN4x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_656 = Coupling(name = 'GC_656',
                  value = 'ee*complex(0,1)*UU1x1*complexconjugate(UU2x1) + ee*complex(0,1)*UU1x2*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_657 = Coupling(name = 'GC_657',
                  value = '-((cw*ee*complex(0,1)*UU1x1*complexconjugate(UU2x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*UU1x1*complexconjugate(UU2x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*UU1x2*complexconjugate(UU2x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*UU1x2*complexconjugate(UU2x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_658 = Coupling(name = 'GC_658',
                  value = 'ee*complex(0,1)*UU2x1*complexconjugate(UU2x1) + ee*complex(0,1)*UU2x2*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_659 = Coupling(name = 'GC_659',
                  value = '-((cw*ee*complex(0,1)*UU2x1*complexconjugate(UU2x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*UU2x1*complexconjugate(UU2x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*UU2x2*complexconjugate(UU2x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*UU2x2*complexconjugate(UU2x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_660 = Coupling(name = 'GC_660',
                  value = '-((ee*complex(0,1)*I43x11*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_661 = Coupling(name = 'GC_661',
                  value = '-((ee*complex(0,1)*I43x22*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_662 = Coupling(name = 'GC_662',
                  value = '-((ee*complex(0,1)*I43x33*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_663 = Coupling(name = 'GC_663',
                  value = '-((ee*complex(0,1)*I63x11*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_664 = Coupling(name = 'GC_664',
                  value = '-((ee*complex(0,1)*I63x22*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_665 = Coupling(name = 'GC_665',
                  value = 'complex(0,1)*I84x33*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_666 = Coupling(name = 'GC_666',
                  value = 'complex(0,1)*I84x36*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_667 = Coupling(name = 'GC_667',
                  value = '-((ee*complex(0,1)*I63x33*complexconjugate(VV1x1))/sw) + complex(0,1)*I65x33*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_668 = Coupling(name = 'GC_668',
                  value = '-((ee*complex(0,1)*I63x36*complexconjugate(VV1x1))/sw) + complex(0,1)*I65x36*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_669 = Coupling(name = 'GC_669',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN1x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_670 = Coupling(name = 'GC_670',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN2x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_671 = Coupling(name = 'GC_671',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN3x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_672 = Coupling(name = 'GC_672',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN4x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_673 = Coupling(name = 'GC_673',
                  value = 'ee*complex(0,1)*VV1x1*complexconjugate(VV1x1) + ee*complex(0,1)*VV1x2*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_674 = Coupling(name = 'GC_674',
                  value = '-((cw*ee*complex(0,1)*VV1x1*complexconjugate(VV1x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*VV1x1*complexconjugate(VV1x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*VV1x2*complexconjugate(VV1x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*VV1x2*complexconjugate(VV1x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_675 = Coupling(name = 'GC_675',
                  value = 'ee*complex(0,1)*VV2x1*complexconjugate(VV1x1) + ee*complex(0,1)*VV2x2*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_676 = Coupling(name = 'GC_676',
                  value = '-((cw*ee*complex(0,1)*VV2x1*complexconjugate(VV1x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*VV2x1*complexconjugate(VV1x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*VV2x2*complexconjugate(VV1x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*VV2x2*complexconjugate(VV1x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_677 = Coupling(name = 'GC_677',
                  value = '-((ee*complex(0,1)*I43x11*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_678 = Coupling(name = 'GC_678',
                  value = '-((ee*complex(0,1)*I43x22*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_679 = Coupling(name = 'GC_679',
                  value = '-((ee*complex(0,1)*I43x33*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_680 = Coupling(name = 'GC_680',
                  value = '-((ee*complex(0,1)*I63x11*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_681 = Coupling(name = 'GC_681',
                  value = '-((ee*complex(0,1)*I63x22*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_682 = Coupling(name = 'GC_682',
                  value = 'complex(0,1)*I84x33*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_683 = Coupling(name = 'GC_683',
                  value = 'complex(0,1)*I84x36*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_684 = Coupling(name = 'GC_684',
                  value = '-((ee*complex(0,1)*I63x33*complexconjugate(VV2x1))/sw) + complex(0,1)*I65x33*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_685 = Coupling(name = 'GC_685',
                  value = '-((ee*complex(0,1)*I63x36*complexconjugate(VV2x1))/sw) + complex(0,1)*I65x36*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_686 = Coupling(name = 'GC_686',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN1x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_687 = Coupling(name = 'GC_687',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN2x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_688 = Coupling(name = 'GC_688',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN3x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_689 = Coupling(name = 'GC_689',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN4x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_690 = Coupling(name = 'GC_690',
                  value = 'ee*complex(0,1)*VV1x1*complexconjugate(VV2x1) + ee*complex(0,1)*VV1x2*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_691 = Coupling(name = 'GC_691',
                  value = '-((cw*ee*complex(0,1)*VV1x1*complexconjugate(VV2x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*VV1x1*complexconjugate(VV2x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*VV1x2*complexconjugate(VV2x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*VV1x2*complexconjugate(VV2x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_692 = Coupling(name = 'GC_692',
                  value = 'ee*complex(0,1)*VV2x1*complexconjugate(VV2x1) + ee*complex(0,1)*VV2x2*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_693 = Coupling(name = 'GC_693',
                  value = '-((cw*ee*complex(0,1)*VV2x1*complexconjugate(VV2x1))/((-1 + sw)*sw*(1 + sw))) + (cw*ee*complex(0,1)*sw*VV2x1*complexconjugate(VV2x1))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*VV2x2*complexconjugate(VV2x2))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*complex(0,1)*sw*VV2x2*complexconjugate(VV2x2))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_694 = Coupling(name = 'GC_694',
                  value = '-((complex(0,1)*yd3x3*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_695 = Coupling(name = 'GC_695',
                  value = '-((complex(0,1)*ye3x3*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_696 = Coupling(name = 'GC_696',
                  value = '-((complex(0,1)*yu3x3*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_697 = Coupling(name = 'GC_697',
                  value = '-((complex(0,1)*complexconjugate(yd3x3)*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_698 = Coupling(name = 'GC_698',
                  value = '-((complex(0,1)*complexconjugate(ye3x3)*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_699 = Coupling(name = 'GC_699',
                  value = '-((complex(0,1)*complexconjugate(yu3x3)*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_700 = Coupling(name = 'GC_700',
                  value = 'complex(0,1)*I1x33*cmath.cos(beta)',
                  order = {'QED':1})

GC_701 = Coupling(name = 'GC_701',
                  value = 'complex(0,1)*I21x33*cmath.cos(beta)',
                  order = {'QED':1})

GC_702 = Coupling(name = 'GC_702',
                  value = '(yu3x3*cmath.cos(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_703 = Coupling(name = 'GC_703',
                  value = '-((complexconjugate(yu3x3)*cmath.cos(beta))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_704 = Coupling(name = 'GC_704',
                  value = '(ee*complex(0,1)*NN1x4*VV1x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x4*sw*VV1x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*VV1x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_705 = Coupling(name = 'GC_705',
                  value = '(ee*complex(0,1)*NN2x4*VV1x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x4*sw*VV1x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*VV1x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_706 = Coupling(name = 'GC_706',
                  value = '(ee*complex(0,1)*NN3x4*VV1x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x4*sw*VV1x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*VV1x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_707 = Coupling(name = 'GC_707',
                  value = '(ee*complex(0,1)*NN4x4*VV1x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN4x4*sw*VV1x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*VV1x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*VV1x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_708 = Coupling(name = 'GC_708',
                  value = '(ee*complex(0,1)*NN1x4*VV2x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x4*sw*VV2x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*VV2x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_709 = Coupling(name = 'GC_709',
                  value = '(ee*complex(0,1)*NN2x4*VV2x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x4*sw*VV2x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*VV2x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_710 = Coupling(name = 'GC_710',
                  value = '(ee*complex(0,1)*NN3x4*VV2x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x4*sw*VV2x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*VV2x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_711 = Coupling(name = 'GC_711',
                  value = '(ee*complex(0,1)*NN4x4*VV2x1*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN4x4*sw*VV2x1*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*VV2x2*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*VV2x2*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_712 = Coupling(name = 'GC_712',
                  value = '(ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_713 = Coupling(name = 'GC_713',
                  value = '(ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_714 = Coupling(name = 'GC_714',
                  value = '(ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN3x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_715 = Coupling(name = 'GC_715',
                  value = '(ee*complex(0,1)*complexconjugate(NN4x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN4x4)*complexconjugate(VV1x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_716 = Coupling(name = 'GC_716',
                  value = '(ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_717 = Coupling(name = 'GC_717',
                  value = '(ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_718 = Coupling(name = 'GC_718',
                  value = '(ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN3x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_719 = Coupling(name = 'GC_719',
                  value = '(ee*complex(0,1)*complexconjugate(NN4x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN4x4)*complexconjugate(VV2x1)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_720 = Coupling(name = 'GC_720',
                  value = '(complex(0,1)*yd3x3*cmath.sin(alp))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_721 = Coupling(name = 'GC_721',
                  value = '(complex(0,1)*ye3x3*cmath.sin(alp))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_722 = Coupling(name = 'GC_722',
                  value = '-((complex(0,1)*yu3x3*cmath.sin(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_723 = Coupling(name = 'GC_723',
                  value = '(complex(0,1)*complexconjugate(yd3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_724 = Coupling(name = 'GC_724',
                  value = '(complex(0,1)*complexconjugate(ye3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_725 = Coupling(name = 'GC_725',
                  value = '-((complex(0,1)*complexconjugate(yu3x3)*cmath.sin(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_726 = Coupling(name = 'GC_726',
                  value = '(cw*ee*complex(0,1)*NN1x1*NN1x4*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN1x4*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN1x4*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN1x3*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN1x3*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN1x3*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_727 = Coupling(name = 'GC_727',
                  value = '-((cw*ee*complex(0,1)*NN1x1*NN1x3*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*NN1x2*NN1x3*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN1x3*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN1x4*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN1x4*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN1x4*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_728 = Coupling(name = 'GC_728',
                  value = '(cw*ee*complex(0,1)*NN1x4*NN2x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN2x4*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x4*NN2x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN2x4*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x4*NN2x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN2x4*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x3*NN2x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN2x3*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x3*NN2x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN2x3*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x3*NN2x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN2x3*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_729 = Coupling(name = 'GC_729',
                  value = '(cw*ee*complex(0,1)*NN2x1*NN2x4*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN2x4*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN2x4*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN2x3*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN2x3*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN2x3*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_730 = Coupling(name = 'GC_730',
                  value = '-(cw*ee*complex(0,1)*NN1x3*NN2x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN1x1*NN2x3*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x3*NN2x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN2x3*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x3*NN2x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN2x3*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x4*NN2x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN2x4*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x4*NN2x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN2x4*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x4*NN2x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN2x4*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_731 = Coupling(name = 'GC_731',
                  value = '-((cw*ee*complex(0,1)*NN2x1*NN2x3*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*NN2x2*NN2x3*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN2x3*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN2x4*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN2x4*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN2x4*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_732 = Coupling(name = 'GC_732',
                  value = '(cw*ee*complex(0,1)*NN1x4*NN3x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN3x4*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x4*NN3x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN3x4*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x4*NN3x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN3x4*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x3*NN3x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN3x3*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x3*NN3x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN3x3*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x3*NN3x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN3x3*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_733 = Coupling(name = 'GC_733',
                  value = '(cw*ee*complex(0,1)*NN2x4*NN3x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN3x4*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x4*NN3x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN3x4*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x4*NN3x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN3x4*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x3*NN3x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN3x3*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x3*NN3x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN3x3*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x3*NN3x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN3x3*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_734 = Coupling(name = 'GC_734',
                  value = '(cw*ee*complex(0,1)*NN3x1*NN3x4*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN3x4*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN3x2*NN3x4*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*NN3x3*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN3x3*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN3x2*NN3x3*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_735 = Coupling(name = 'GC_735',
                  value = '-(cw*ee*complex(0,1)*NN1x3*NN3x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN1x1*NN3x3*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x3*NN3x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN3x3*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x3*NN3x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN3x3*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x4*NN3x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN3x4*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x4*NN3x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN3x4*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x4*NN3x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN3x4*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_736 = Coupling(name = 'GC_736',
                  value = '-(cw*ee*complex(0,1)*NN2x3*NN3x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN2x1*NN3x3*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x3*NN3x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN3x3*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x3*NN3x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN3x3*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x4*NN3x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN3x4*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x4*NN3x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN3x4*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x4*NN3x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN3x4*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_737 = Coupling(name = 'GC_737',
                  value = '-((cw*ee*complex(0,1)*NN3x1*NN3x3*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*NN3x2*NN3x3*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN3x3*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*NN3x4*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN3x4*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN3x2*NN3x4*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_738 = Coupling(name = 'GC_738',
                  value = '(cw*ee*complex(0,1)*NN1x4*NN4x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN4x4*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x4*NN4x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN4x4*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x4*NN4x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN4x4*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x3*NN4x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN4x3*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x3*NN4x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN4x3*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x3*NN4x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN4x3*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_739 = Coupling(name = 'GC_739',
                  value = '(cw*ee*complex(0,1)*NN2x4*NN4x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN4x4*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x4*NN4x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN4x4*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x4*NN4x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN4x4*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x3*NN4x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN4x3*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x3*NN4x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN4x3*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x3*NN4x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN4x3*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_740 = Coupling(name = 'GC_740',
                  value = '(cw*ee*complex(0,1)*NN3x4*NN4x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*NN4x4*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN3x4*NN4x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN4x4*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN3x4*NN4x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN3x2*NN4x4*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x3*NN4x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*NN4x3*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN3x3*NN4x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN4x3*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN3x3*NN4x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN3x2*NN4x3*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_741 = Coupling(name = 'GC_741',
                  value = '(cw*ee*complex(0,1)*NN4x1*NN4x4*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN4x2*NN4x4*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN4x2*NN4x4*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*NN4x3*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN4x2*NN4x3*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN4x2*NN4x3*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_742 = Coupling(name = 'GC_742',
                  value = '-(cw*ee*complex(0,1)*NN1x3*NN4x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN1x1*NN4x3*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x3*NN4x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN4x3*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x3*NN4x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN4x3*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x4*NN4x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN1x1*NN4x4*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN1x4*NN4x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x2*NN4x4*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN1x4*NN4x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN1x2*NN4x4*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_743 = Coupling(name = 'GC_743',
                  value = '-(cw*ee*complex(0,1)*NN2x3*NN4x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN2x1*NN4x3*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x3*NN4x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN4x3*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x3*NN4x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN4x3*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x4*NN4x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN2x1*NN4x4*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN2x4*NN4x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x2*NN4x4*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN2x4*NN4x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN2x2*NN4x4*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_744 = Coupling(name = 'GC_744',
                  value = '-(cw*ee*complex(0,1)*NN3x3*NN4x1*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN3x1*NN4x3*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN3x3*NN4x2*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN3x2*NN4x3*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x3*NN4x2*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN4x3*sw*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x4*NN4x1*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN3x1*NN4x4*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN3x4*NN4x2*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x2*NN4x4*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN3x4*NN4x2*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*NN3x2*NN4x4*sw*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_745 = Coupling(name = 'GC_745',
                  value = '-((cw*ee*complex(0,1)*NN4x1*NN4x3*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*NN4x2*NN4x3*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN4x2*NN4x3*sw*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*NN4x1*NN4x4*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*NN4x2*NN4x4*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*NN4x2*NN4x4*sw*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_746 = Coupling(name = 'GC_746',
                  value = '(ee**2*complex(0,1)*vu*cmath.cos(alp))/(2.*sw**2) - (ee**2*complex(0,1)*vd*cmath.sin(alp))/(2.*sw**2)',
                  order = {'QED':1})

GC_747 = Coupling(name = 'GC_747',
                  value = '(ee**2*complex(0,1)*I26x44*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x44*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_748 = Coupling(name = 'GC_748',
                  value = '(ee**2*complex(0,1)*I26x55*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x55*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_749 = Coupling(name = 'GC_749',
                  value = '-(ee**2*complex(0,1)*I52x44*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x44*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_750 = Coupling(name = 'GC_750',
                  value = '-(ee**2*complex(0,1)*I52x55*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x55*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_751 = Coupling(name = 'GC_751',
                  value = '(ee**2*complex(0,1)*I6x44*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x44*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_752 = Coupling(name = 'GC_752',
                  value = '(ee**2*complex(0,1)*I6x55*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x55*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_753 = Coupling(name = 'GC_753',
                  value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_754 = Coupling(name = 'GC_754',
                  value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/(2.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.sin(alp))/(2.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_755 = Coupling(name = 'GC_755',
                  value = '-(ee**2*complex(0,1)*I25x11*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x11*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I25x11*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x11*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_756 = Coupling(name = 'GC_756',
                  value = '-(ee**2*complex(0,1)*I25x22*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x22*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I25x22*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x22*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_757 = Coupling(name = 'GC_757',
                  value = '(ee**2*complex(0,1)*I51x11*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x11*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*I51x11*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x11*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_758 = Coupling(name = 'GC_758',
                  value = '(ee**2*complex(0,1)*I51x22*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x22*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*I51x22*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x22*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_759 = Coupling(name = 'GC_759',
                  value = '-(ee**2*complex(0,1)*I5x11*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x11*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I5x11*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x11*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_760 = Coupling(name = 'GC_760',
                  value = '-(ee**2*complex(0,1)*I5x22*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x22*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I5x22*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x22*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_761 = Coupling(name = 'GC_761',
                  value = '-(ee**2*complex(0,1)*I5x33*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x33*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x33*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x33*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x33*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x33*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x33*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I17x33*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x33*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x33*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x33*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x33*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I17x33*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x33*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I14x33*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x33*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I14x33*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x33*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_762 = Coupling(name = 'GC_762',
                  value = '-(ee**2*complex(0,1)*I5x36*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x36*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x36*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x36*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x36*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x36*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x36*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I17x36*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x36*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x36*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x36*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x36*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I17x36*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x36*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I14x36*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x36*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I14x36*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x36*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_763 = Coupling(name = 'GC_763',
                  value = '-(ee**2*complex(0,1)*I5x63*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x63*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x63*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x63*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x63*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x63*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x63*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I17x63*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x63*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x63*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x63*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x63*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I17x63*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x63*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I14x63*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x63*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I14x63*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x63*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_764 = Coupling(name = 'GC_764',
                  value = '-(ee**2*complex(0,1)*I5x66*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x66*vu*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x66*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x66*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x66*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x66*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x66*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I17x66*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x66*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x66*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x66*vd*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x66*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I17x66*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x66*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I14x66*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x66*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I14x66*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x66*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_765 = Coupling(name = 'GC_765',
                  value = '-(ee**2*complex(0,1)*I25x33*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x33*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x33*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x33*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x33*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x33*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x33*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x33*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x33*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (complex(0,1)*I36x33*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x33*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x33*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I36x33*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x33*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I33x33*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x33*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I33x33*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x33*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_766 = Coupling(name = 'GC_766',
                  value = '-(ee**2*complex(0,1)*I25x36*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x36*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x36*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x36*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x36*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x36*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x36*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x36*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x36*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (complex(0,1)*I36x36*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x36*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x36*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I36x36*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x36*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I33x36*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x36*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I33x36*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x36*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_767 = Coupling(name = 'GC_767',
                  value = '-(ee**2*complex(0,1)*I25x63*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x63*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x63*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x63*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x63*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x63*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x63*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x63*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x63*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (complex(0,1)*I36x63*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x63*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x63*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I36x63*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x63*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I33x63*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x63*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I33x63*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x63*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_768 = Coupling(name = 'GC_768',
                  value = '-(ee**2*complex(0,1)*I25x66*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x66*vu*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x66*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x66*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x66*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x66*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x66*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x66*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x66*vd*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (complex(0,1)*I36x66*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x66*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x66*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I36x66*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x66*sw**2*vd*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I33x66*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x66*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I33x66*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x66*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_769 = Coupling(name = 'GC_769',
                  value = '(ee**2*complex(0,1)*vd*cmath.cos(alp))/(2.*sw**2) + (ee**2*complex(0,1)*vu*cmath.sin(alp))/(2.*sw**2)',
                  order = {'QED':1})

GC_770 = Coupling(name = 'GC_770',
                  value = '-(ee**2*complex(0,1)*I26x44*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x44*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_771 = Coupling(name = 'GC_771',
                  value = '-(ee**2*complex(0,1)*I26x55*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x55*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_772 = Coupling(name = 'GC_772',
                  value = '(ee**2*complex(0,1)*I52x44*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x44*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_773 = Coupling(name = 'GC_773',
                  value = '(ee**2*complex(0,1)*I52x55*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x55*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_774 = Coupling(name = 'GC_774',
                  value = '-(ee**2*complex(0,1)*I6x44*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x44*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_775 = Coupling(name = 'GC_775',
                  value = '-(ee**2*complex(0,1)*I6x55*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x55*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_776 = Coupling(name = 'GC_776',
                  value = '(ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_777 = Coupling(name = 'GC_777',
                  value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp))/(2.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/(2.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_778 = Coupling(name = 'GC_778',
                  value = '(ee**2*complex(0,1)*I25x11*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I25x11*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I25x11*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x11*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_779 = Coupling(name = 'GC_779',
                  value = '(ee**2*complex(0,1)*I25x22*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I25x22*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I25x22*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x22*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_780 = Coupling(name = 'GC_780',
                  value = '-(ee**2*complex(0,1)*I51x11*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I51x11*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*I51x11*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x11*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_781 = Coupling(name = 'GC_781',
                  value = '-(ee**2*complex(0,1)*I51x22*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I51x22*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*I51x22*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x22*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_782 = Coupling(name = 'GC_782',
                  value = '(ee**2*complex(0,1)*I5x11*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x11*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I5x11*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x11*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_783 = Coupling(name = 'GC_783',
                  value = '(ee**2*complex(0,1)*I5x22*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x22*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*I5x22*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x22*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_784 = Coupling(name = 'GC_784',
                  value = '-(ee**2*complex(0,1)*I51x33*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I52x33*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I51x33*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I76x33*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I76x33*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x33*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x33*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x33*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x33*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x33*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x33*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x33*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x33*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x33*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x33*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x33*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x33*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x33*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_785 = Coupling(name = 'GC_785',
                  value = '-(ee**2*complex(0,1)*I51x36*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I52x36*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I51x36*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I76x36*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I76x36*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x36*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x36*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x36*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x36*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x36*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x36*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x36*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x36*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x36*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x36*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x36*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x36*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x36*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_786 = Coupling(name = 'GC_786',
                  value = '-(ee**2*complex(0,1)*I51x63*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I52x63*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I51x63*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I76x63*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I76x63*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x63*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x63*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x63*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x63*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x63*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x63*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x63*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x63*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x63*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x63*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x63*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x63*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x63*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_787 = Coupling(name = 'GC_787',
                  value = '-(ee**2*complex(0,1)*I51x66*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I52x66*vd*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I51x66*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I76x66*MUH*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I76x66*MUH*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x66*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x66*sw**2*complexconjugate(MUH)*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x66*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x66*vu*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x66*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x66*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x66*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x66*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x66*sw**2*vu*cmath.sin(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x66*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x66*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x66*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x66*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_788 = Coupling(name = 'GC_788',
                  value = '-((ee*complex(0,1)*UU1x1*VV1x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU1x2*VV1x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_789 = Coupling(name = 'GC_789',
                  value = '-((ee*complex(0,1)*UU2x1*VV1x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU2x2*VV1x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_790 = Coupling(name = 'GC_790',
                  value = '-((ee*complex(0,1)*UU1x2*VV1x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU1x1*VV1x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_791 = Coupling(name = 'GC_791',
                  value = '-((ee*complex(0,1)*UU2x2*VV1x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU2x1*VV1x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_792 = Coupling(name = 'GC_792',
                  value = '-((ee*complex(0,1)*UU1x1*VV2x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU1x2*VV2x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_793 = Coupling(name = 'GC_793',
                  value = '-((ee*complex(0,1)*UU2x1*VV2x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU2x2*VV2x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_794 = Coupling(name = 'GC_794',
                  value = '-((ee*complex(0,1)*UU1x2*VV2x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU1x1*VV2x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_795 = Coupling(name = 'GC_795',
                  value = '-((ee*complex(0,1)*UU2x2*VV2x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU2x1*VV2x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_796 = Coupling(name = 'GC_796',
                  value = '(complex(0,1)*I17x33*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x33*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x33*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I6x33*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x33*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I17x33*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x33*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I14x33*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x33*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I14x33*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x33*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I5x33*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x33*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x33*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x33*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x33*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x33*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x33*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_797 = Coupling(name = 'GC_797',
                  value = '(complex(0,1)*I17x36*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x36*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x36*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I6x36*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x36*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I17x36*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x36*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I14x36*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x36*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I14x36*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x36*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I5x36*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x36*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x36*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x36*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x36*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x36*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x36*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_798 = Coupling(name = 'GC_798',
                  value = '(complex(0,1)*I17x63*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x63*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x63*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I6x63*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x63*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I17x63*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x63*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I14x63*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x63*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I14x63*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x63*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I5x63*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x63*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x63*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x63*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x63*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x63*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x63*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_799 = Coupling(name = 'GC_799',
                  value = '(complex(0,1)*I17x66*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I19x66*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x66*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I6x66*vd*cmath.cos(alp))/(6.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I5x66*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I17x66*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I19x66*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I14x66*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I16x66*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I14x66*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I16x66*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I5x66*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I6x66*vu*cmath.sin(alp))/(6.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I5x66*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I15x66*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I15x66*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I18x66*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I18x66*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_800 = Coupling(name = 'GC_800',
                  value = '(ee**2*complex(0,1)*I25x33*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I26x33*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I36x33*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x33*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I25x33*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I36x33*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x33*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I33x33*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x33*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I33x33*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x33*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x33*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x33*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x33*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x33*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x33*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x33*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x33*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_801 = Coupling(name = 'GC_801',
                  value = '(ee**2*complex(0,1)*I25x36*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I26x36*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I36x36*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x36*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I25x36*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I36x36*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x36*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I33x36*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x36*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I33x36*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x36*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x36*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x36*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x36*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x36*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x36*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x36*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x36*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_802 = Coupling(name = 'GC_802',
                  value = '(ee**2*complex(0,1)*I25x63*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I26x63*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I36x63*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x63*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I25x63*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I36x63*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x63*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I33x63*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x63*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I33x63*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x63*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x63*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x63*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x63*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x63*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x63*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x63*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x63*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_803 = Coupling(name = 'GC_803',
                  value = '(ee**2*complex(0,1)*I25x66*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I26x66*vd*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I36x66*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I38x66*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I25x66*vd*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I36x66*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I38x66*sw**2*vd*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I33x66*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I35x66*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I33x66*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I35x66*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee**2*complex(0,1)*I25x66*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I26x66*vu*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*I25x66*vu*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I34x66*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I34x66*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I37x66*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I37x66*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_804 = Coupling(name = 'GC_804',
                  value = '(ee**2*complex(0,1)*I51x33*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x33*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x33*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x33*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x33*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x33*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x33*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x33*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x33*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x33*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x33*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x33*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x33*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x33*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I76x33*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I76x33*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x33*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x33*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_805 = Coupling(name = 'GC_805',
                  value = '(ee**2*complex(0,1)*I51x36*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x36*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x36*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x36*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x36*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x36*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x36*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x36*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x36*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x36*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x36*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x36*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x36*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x36*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I76x36*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I76x36*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x36*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x36*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_806 = Coupling(name = 'GC_806',
                  value = '(ee**2*complex(0,1)*I51x63*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x63*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x63*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x63*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x63*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x63*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x63*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x63*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x63*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x63*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x63*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x63*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x63*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x63*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I76x63*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I76x63*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x63*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x63*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_807 = Coupling(name = 'GC_807',
                  value = '(ee**2*complex(0,1)*I51x66*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x66*vu*cmath.cos(alp))/(3.*(-1 + sw)*(1 + sw)) + (complex(0,1)*I80x66*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I81x66*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x66*vu*cmath.cos(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (complex(0,1)*I80x66*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (complex(0,1)*I81x66*sw**2*vu*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (complex(0,1)*I77x66*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I78x66*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I77x66*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I78x66*sw**2*cmath.cos(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (ee**2*complex(0,1)*I51x66*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I52x66*vd*cmath.sin(alp))/(3.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*I51x66*vd*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (complex(0,1)*I76x66*MUH*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I76x66*MUH*sw**2*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) + (complex(0,1)*I79x66*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (complex(0,1)*I79x66*sw**2*complexconjugate(MUH)*cmath.sin(alp))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_808 = Coupling(name = 'GC_808',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_809 = Coupling(name = 'GC_809',
                  value = '-((cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_810 = Coupling(name = 'GC_810',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_811 = Coupling(name = 'GC_811',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_812 = Coupling(name = 'GC_812',
                  value = '-(cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_813 = Coupling(name = 'GC_813',
                  value = '-((cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_814 = Coupling(name = 'GC_814',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_815 = Coupling(name = 'GC_815',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_816 = Coupling(name = 'GC_816',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_817 = Coupling(name = 'GC_817',
                  value = '-(cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_818 = Coupling(name = 'GC_818',
                  value = '-(cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_819 = Coupling(name = 'GC_819',
                  value = '-((cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_820 = Coupling(name = 'GC_820',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_821 = Coupling(name = 'GC_821',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_822 = Coupling(name = 'GC_822',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_823 = Coupling(name = 'GC_823',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_824 = Coupling(name = 'GC_824',
                  value = '-(cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_825 = Coupling(name = 'GC_825',
                  value = '-(cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_826 = Coupling(name = 'GC_826',
                  value = '-(cw*ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_827 = Coupling(name = 'GC_827',
                  value = '-((cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw))) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.cos(alp))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.cos(alp))/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.sin(alp))/((-1 + sw)*sw*(1 + sw)) + (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.sin(alp))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_828 = Coupling(name = 'GC_828',
                  value = '-((ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV1x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV1x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_829 = Coupling(name = 'GC_829',
                  value = '-((ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV1x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV1x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_830 = Coupling(name = 'GC_830',
                  value = '-((ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV1x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV1x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_831 = Coupling(name = 'GC_831',
                  value = '-((ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV1x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV1x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_832 = Coupling(name = 'GC_832',
                  value = '-((ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV2x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV2x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_833 = Coupling(name = 'GC_833',
                  value = '-((ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV2x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV2x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_834 = Coupling(name = 'GC_834',
                  value = '-((ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV2x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV2x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_835 = Coupling(name = 'GC_835',
                  value = '-((ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV2x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV2x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_836 = Coupling(name = 'GC_836',
                  value = '(ee**2*complex(0,1)*cmath.cos(alp)**2)/(2.*sw**2) + (ee**2*complex(0,1)*cmath.sin(alp)**2)/(2.*sw**2)',
                  order = {'QED':2})

GC_837 = Coupling(name = 'GC_837',
                  value = '-(ee**2*complex(0,1)*cmath.cos(alp)**2)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*cmath.sin(alp)**2)/(2.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':2})

GC_838 = Coupling(name = 'GC_838',
                  value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (5*ee**2*complex(0,1)*vd*cmath.cos(alp)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (5*ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.sin(alp)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.sin(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_839 = Coupling(name = 'GC_839',
                  value = '(3*ee**2*complex(0,1)*vu*cmath.cos(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (3*ee**2*complex(0,1)*vd*cmath.cos(alp)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (3*ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.sin(alp)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (3*ee**2*complex(0,1)*vd*cmath.sin(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_840 = Coupling(name = 'GC_840',
                  value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (5*ee**2*complex(0,1)*vu*cmath.cos(alp)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (5*ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.sin(alp)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.sin(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_841 = Coupling(name = 'GC_841',
                  value = '(3*ee**2*complex(0,1)*vd*cmath.cos(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (3*ee**2*complex(0,1)*vu*cmath.cos(alp)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (3*ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.sin(alp)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (3*ee**2*complex(0,1)*vu*cmath.sin(alp)**3)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_842 = Coupling(name = 'GC_842',
                  value = 'complex(0,1)*I20x33*cmath.sin(beta)',
                  order = {'QED':1})

GC_843 = Coupling(name = 'GC_843',
                  value = 'complex(0,1)*I22x33*cmath.sin(beta)',
                  order = {'QED':1})

GC_844 = Coupling(name = 'GC_844',
                  value = 'complex(0,1)*I2x33*cmath.sin(beta)',
                  order = {'QED':1})

GC_845 = Coupling(name = 'GC_845',
                  value = 'complex(0,1)*I99x33*cmath.sin(beta)',
                  order = {'QED':1})

GC_846 = Coupling(name = 'GC_846',
                  value = '(yd3x3*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_847 = Coupling(name = 'GC_847',
                  value = '(ye3x3*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_848 = Coupling(name = 'GC_848',
                  value = '-((complexconjugate(yd3x3)*cmath.sin(beta))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_849 = Coupling(name = 'GC_849',
                  value = '-((complexconjugate(ye3x3)*cmath.sin(beta))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_850 = Coupling(name = 'GC_850',
                  value = '-((I15x33*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I18x33*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I14x33*cmath.sin(beta))/cmath.sqrt(2) + (I16x33*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_851 = Coupling(name = 'GC_851',
                  value = '-((I15x36*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I18x36*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I14x36*cmath.sin(beta))/cmath.sqrt(2) + (I16x36*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_852 = Coupling(name = 'GC_852',
                  value = '-((I15x63*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I18x63*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I14x63*cmath.sin(beta))/cmath.sqrt(2) + (I16x63*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_853 = Coupling(name = 'GC_853',
                  value = '-((I15x66*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I18x66*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I14x66*cmath.sin(beta))/cmath.sqrt(2) + (I16x66*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_854 = Coupling(name = 'GC_854',
                  value = '-((I34x33*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I37x33*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I33x33*cmath.sin(beta))/cmath.sqrt(2) + (I35x33*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_855 = Coupling(name = 'GC_855',
                  value = '-((I34x36*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I37x36*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I33x36*cmath.sin(beta))/cmath.sqrt(2) + (I35x36*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_856 = Coupling(name = 'GC_856',
                  value = '-((I34x63*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I37x63*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I33x63*cmath.sin(beta))/cmath.sqrt(2) + (I35x63*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_857 = Coupling(name = 'GC_857',
                  value = '-((I34x66*MUH*cmath.cos(beta))/cmath.sqrt(2)) + (I37x66*complexconjugate(MUH)*cmath.cos(beta))/cmath.sqrt(2) - (I33x66*cmath.sin(beta))/cmath.sqrt(2) + (I35x66*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_858 = Coupling(name = 'GC_858',
                  value = '-((cw*ee*NN1x1*NN1x4*cmath.cos(beta))/((-1 + sw)*(1 + sw))) + (ee*NN1x2*NN1x4*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*NN1x2*NN1x4*sw*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*NN1x1*NN1x3*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (ee*NN1x2*NN1x3*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*NN1x2*NN1x3*sw*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_859 = Coupling(name = 'GC_859',
                  value = '-(cw*ee*NN1x4*NN2x1*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*NN1x1*NN2x4*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN1x4*NN2x2*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN1x2*NN2x4*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN1x4*NN2x2*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN1x2*NN2x4*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN1x3*NN2x1*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN1x1*NN2x3*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN1x3*NN2x2*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN1x2*NN2x3*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN1x3*NN2x2*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN1x2*NN2x3*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_860 = Coupling(name = 'GC_860',
                  value = '-((cw*ee*NN2x1*NN2x4*cmath.cos(beta))/((-1 + sw)*(1 + sw))) + (ee*NN2x2*NN2x4*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*NN2x2*NN2x4*sw*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*NN2x1*NN2x3*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (ee*NN2x2*NN2x3*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*NN2x2*NN2x3*sw*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_861 = Coupling(name = 'GC_861',
                  value = '-(cw*ee*NN1x4*NN3x1*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*NN1x1*NN3x4*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN1x4*NN3x2*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN1x2*NN3x4*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN1x4*NN3x2*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN1x2*NN3x4*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN1x3*NN3x1*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN1x1*NN3x3*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN1x3*NN3x2*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN1x2*NN3x3*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN1x3*NN3x2*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN1x2*NN3x3*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_862 = Coupling(name = 'GC_862',
                  value = '-(cw*ee*NN2x4*NN3x1*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*NN2x1*NN3x4*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN2x4*NN3x2*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN2x2*NN3x4*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN2x4*NN3x2*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN2x2*NN3x4*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN2x3*NN3x1*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN2x1*NN3x3*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN2x3*NN3x2*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN2x2*NN3x3*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN2x3*NN3x2*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN2x2*NN3x3*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_863 = Coupling(name = 'GC_863',
                  value = '-((cw*ee*NN3x1*NN3x4*cmath.cos(beta))/((-1 + sw)*(1 + sw))) + (ee*NN3x2*NN3x4*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*NN3x2*NN3x4*sw*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*NN3x1*NN3x3*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (ee*NN3x2*NN3x3*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*NN3x2*NN3x3*sw*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_864 = Coupling(name = 'GC_864',
                  value = '-(cw*ee*NN1x4*NN4x1*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*NN1x1*NN4x4*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN1x4*NN4x2*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN1x2*NN4x4*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN1x4*NN4x2*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN1x2*NN4x4*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN1x3*NN4x1*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN1x1*NN4x3*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN1x3*NN4x2*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN1x2*NN4x3*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN1x3*NN4x2*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN1x2*NN4x3*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_865 = Coupling(name = 'GC_865',
                  value = '-(cw*ee*NN2x4*NN4x1*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*NN2x1*NN4x4*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN2x4*NN4x2*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN2x2*NN4x4*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN2x4*NN4x2*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN2x2*NN4x4*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN2x3*NN4x1*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN2x1*NN4x3*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN2x3*NN4x2*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN2x2*NN4x3*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN2x3*NN4x2*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN2x2*NN4x3*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_866 = Coupling(name = 'GC_866',
                  value = '-(cw*ee*NN3x4*NN4x1*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*NN3x1*NN4x4*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN3x4*NN4x2*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN3x2*NN4x4*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN3x4*NN4x2*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN3x2*NN4x4*sw*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN3x3*NN4x1*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*NN3x1*NN4x3*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*NN3x3*NN4x2*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*NN3x2*NN4x3*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*NN3x3*NN4x2*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*NN3x2*NN4x3*sw*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_867 = Coupling(name = 'GC_867',
                  value = '-((cw*ee*NN4x1*NN4x4*cmath.cos(beta))/((-1 + sw)*(1 + sw))) + (ee*NN4x2*NN4x4*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*NN4x2*NN4x4*sw*cmath.cos(beta))/((-1 + sw)*(1 + sw)) + (cw*ee*NN4x1*NN4x3*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (ee*NN4x2*NN4x3*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*NN4x2*NN4x3*sw*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_868 = Coupling(name = 'GC_868',
                  value = '(ee*complex(0,1)*NN1x3*UU1x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x3*sw*UU1x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN1x1*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*UU1x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_869 = Coupling(name = 'GC_869',
                  value = '(ee*complex(0,1)*NN2x3*UU1x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x3*sw*UU1x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN2x1*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*UU1x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_870 = Coupling(name = 'GC_870',
                  value = '(ee*complex(0,1)*NN3x3*UU1x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x3*sw*UU1x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN3x1*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*UU1x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_871 = Coupling(name = 'GC_871',
                  value = '(ee*complex(0,1)*NN4x3*UU1x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN4x3*sw*UU1x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN4x1*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*UU1x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*UU1x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_872 = Coupling(name = 'GC_872',
                  value = '(ee*complex(0,1)*NN1x3*UU2x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN1x3*sw*UU2x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN1x1*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*UU2x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_873 = Coupling(name = 'GC_873',
                  value = '(ee*complex(0,1)*NN2x3*UU2x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN2x3*sw*UU2x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN2x1*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*UU2x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_874 = Coupling(name = 'GC_874',
                  value = '(ee*complex(0,1)*NN3x3*UU2x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN3x3*sw*UU2x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN3x1*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*UU2x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_875 = Coupling(name = 'GC_875',
                  value = '(ee*complex(0,1)*NN4x3*UU2x1*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*NN4x3*sw*UU2x1*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*NN4x1*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*UU2x2*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*UU2x2*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_876 = Coupling(name = 'GC_876',
                  value = '-(ee**2*complex(0,1)*I39x11*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I39x11*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_877 = Coupling(name = 'GC_877',
                  value = '-(ee**2*complex(0,1)*I39x22*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I39x22*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_878 = Coupling(name = 'GC_878',
                  value = 'complex(0,1)*I42x33*complexconjugate(MUH)*cmath.cos(beta) - (ee**2*complex(0,1)*I39x33*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I40x33*cmath.sin(beta) + (complex(0,1)*I41x33*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I39x33*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_879 = Coupling(name = 'GC_879',
                  value = 'complex(0,1)*I42x36*complexconjugate(MUH)*cmath.cos(beta) - (ee**2*complex(0,1)*I39x36*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I40x36*cmath.sin(beta) + (complex(0,1)*I41x36*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I39x36*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_880 = Coupling(name = 'GC_880',
                  value = '-(ee**2*complex(0,1)*I45x11*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I45x11*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_881 = Coupling(name = 'GC_881',
                  value = '-(ee**2*complex(0,1)*I45x22*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I45x22*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_882 = Coupling(name = 'GC_882',
                  value = 'complex(0,1)*I46x33*MUH*cmath.cos(beta) - (ee**2*complex(0,1)*I45x33*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I47x33*cmath.sin(beta) + (complex(0,1)*I48x33*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I45x33*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_883 = Coupling(name = 'GC_883',
                  value = 'complex(0,1)*I46x36*MUH*cmath.cos(beta) - (ee**2*complex(0,1)*I45x36*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I47x36*cmath.sin(beta) + (complex(0,1)*I48x36*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I45x36*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_884 = Coupling(name = 'GC_884',
                  value = '-(ee**2*complex(0,1)*I53x11*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I53x11*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_885 = Coupling(name = 'GC_885',
                  value = '-(ee**2*complex(0,1)*I53x22*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I53x22*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_886 = Coupling(name = 'GC_886',
                  value = '-(ee**2*complex(0,1)*I66x11*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I66x11*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_887 = Coupling(name = 'GC_887',
                  value = '-(ee**2*complex(0,1)*I66x22*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*I66x22*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                  order = {'QED':1})

GC_888 = Coupling(name = 'GC_888',
                  value = '(ee**2*complex(0,1)*vu*cmath.cos(beta))/(2.*sw) - (ee**2*complex(0,1)*vd*cmath.sin(beta))/(2.*sw)',
                  order = {'QED':1})

GC_889 = Coupling(name = 'GC_889',
                  value = '(cw*ee**2*complex(0,1)*vu*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee**2*complex(0,1)*vd*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_890 = Coupling(name = 'GC_890',
                  value = 'complex(0,1)*I54x33*cmath.cos(beta) + complex(0,1)*I57x33*complexconjugate(MUH)*cmath.cos(beta) + (complex(0,1)*I59x33*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I60x33*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x33*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I56x33*cmath.sin(beta) + complex(0,1)*I55x33*MUH*cmath.sin(beta) + (complex(0,1)*I58x33*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x33*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I59x33*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_891 = Coupling(name = 'GC_891',
                  value = 'complex(0,1)*I54x36*cmath.cos(beta) + complex(0,1)*I57x36*complexconjugate(MUH)*cmath.cos(beta) + (complex(0,1)*I59x36*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I60x36*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x36*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I56x36*cmath.sin(beta) + complex(0,1)*I55x36*MUH*cmath.sin(beta) + (complex(0,1)*I58x36*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x36*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I59x36*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_892 = Coupling(name = 'GC_892',
                  value = 'complex(0,1)*I54x63*cmath.cos(beta) + complex(0,1)*I57x63*complexconjugate(MUH)*cmath.cos(beta) + (complex(0,1)*I59x63*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I60x63*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x63*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I56x63*cmath.sin(beta) + complex(0,1)*I55x63*MUH*cmath.sin(beta) + (complex(0,1)*I58x63*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x63*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I59x63*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_893 = Coupling(name = 'GC_893',
                  value = 'complex(0,1)*I54x66*cmath.cos(beta) + complex(0,1)*I57x66*complexconjugate(MUH)*cmath.cos(beta) + (complex(0,1)*I59x66*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I60x66*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x66*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I56x66*cmath.sin(beta) + complex(0,1)*I55x66*MUH*cmath.sin(beta) + (complex(0,1)*I58x66*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I53x66*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I59x66*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_894 = Coupling(name = 'GC_894',
                  value = '(ee*UU1x1*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU1x2*VV1x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_895 = Coupling(name = 'GC_895',
                  value = '(ee*UU2x1*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU2x2*VV1x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_896 = Coupling(name = 'GC_896',
                  value = '(ee*UU1x1*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU1x2*VV2x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_897 = Coupling(name = 'GC_897',
                  value = '(ee*UU2x1*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU2x2*VV2x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_898 = Coupling(name = 'GC_898',
                  value = 'complex(0,1)*I69x33*cmath.cos(beta) + complex(0,1)*I67x33*MUH*cmath.cos(beta) + (complex(0,1)*I73x33*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I71x33*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x33*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I68x33*cmath.sin(beta) + complex(0,1)*I72x33*complexconjugate(MUH)*cmath.sin(beta) + (complex(0,1)*I70x33*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x33*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I73x33*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_899 = Coupling(name = 'GC_899',
                  value = 'complex(0,1)*I69x36*cmath.cos(beta) + complex(0,1)*I67x36*MUH*cmath.cos(beta) + (complex(0,1)*I73x36*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I71x36*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x36*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I68x36*cmath.sin(beta) + complex(0,1)*I72x36*complexconjugate(MUH)*cmath.sin(beta) + (complex(0,1)*I70x36*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x36*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I73x36*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_900 = Coupling(name = 'GC_900',
                  value = 'complex(0,1)*I69x63*cmath.cos(beta) + complex(0,1)*I67x63*MUH*cmath.cos(beta) + (complex(0,1)*I73x63*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I71x63*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x63*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I68x63*cmath.sin(beta) + complex(0,1)*I72x63*complexconjugate(MUH)*cmath.sin(beta) + (complex(0,1)*I70x63*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x63*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I73x63*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_901 = Coupling(name = 'GC_901',
                  value = 'complex(0,1)*I69x66*cmath.cos(beta) + complex(0,1)*I67x66*MUH*cmath.cos(beta) + (complex(0,1)*I73x66*vd*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*I71x66*vu*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x66*vu*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + complex(0,1)*I68x66*cmath.sin(beta) + complex(0,1)*I72x66*complexconjugate(MUH)*cmath.sin(beta) + (complex(0,1)*I70x66*vd*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*I66x66*vd*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*I73x66*vu*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_902 = Coupling(name = 'GC_902',
                  value = '-((I77x33*cmath.cos(beta))/cmath.sqrt(2)) + (I78x33*cmath.cos(beta))/cmath.sqrt(2) - (I76x33*MUH*cmath.sin(beta))/cmath.sqrt(2) + (I79x33*complexconjugate(MUH)*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_903 = Coupling(name = 'GC_903',
                  value = '-((I77x36*cmath.cos(beta))/cmath.sqrt(2)) + (I78x36*cmath.cos(beta))/cmath.sqrt(2) - (I76x36*MUH*cmath.sin(beta))/cmath.sqrt(2) + (I79x36*complexconjugate(MUH)*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_904 = Coupling(name = 'GC_904',
                  value = '-((I77x63*cmath.cos(beta))/cmath.sqrt(2)) + (I78x63*cmath.cos(beta))/cmath.sqrt(2) - (I76x63*MUH*cmath.sin(beta))/cmath.sqrt(2) + (I79x63*complexconjugate(MUH)*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_905 = Coupling(name = 'GC_905',
                  value = '-((I77x66*cmath.cos(beta))/cmath.sqrt(2)) + (I78x66*cmath.cos(beta))/cmath.sqrt(2) - (I76x66*MUH*cmath.sin(beta))/cmath.sqrt(2) + (I79x66*complexconjugate(MUH)*cmath.sin(beta))/cmath.sqrt(2)',
                  order = {'QED':1})

GC_906 = Coupling(name = 'GC_906',
                  value = '(cw*ee*complexconjugate(NN1x1)*complexconjugate(NN1x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN1x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_907 = Coupling(name = 'GC_907',
                  value = '(cw*ee*complexconjugate(NN1x4)*complexconjugate(NN2x1)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN2x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN1x3)*complexconjugate(NN2x1)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN2x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_908 = Coupling(name = 'GC_908',
                  value = '(cw*ee*complexconjugate(NN2x1)*complexconjugate(NN2x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN2x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_909 = Coupling(name = 'GC_909',
                  value = '(cw*ee*complexconjugate(NN1x4)*complexconjugate(NN3x1)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN1x3)*complexconjugate(NN3x1)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_910 = Coupling(name = 'GC_910',
                  value = '(cw*ee*complexconjugate(NN2x4)*complexconjugate(NN3x1)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN2x3)*complexconjugate(NN3x1)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_911 = Coupling(name = 'GC_911',
                  value = '(cw*ee*complexconjugate(NN3x1)*complexconjugate(NN3x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN3x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_912 = Coupling(name = 'GC_912',
                  value = '(cw*ee*complexconjugate(NN1x4)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN1x3)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_913 = Coupling(name = 'GC_913',
                  value = '(cw*ee*complexconjugate(NN2x4)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN2x3)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_914 = Coupling(name = 'GC_914',
                  value = '(cw*ee*complexconjugate(NN3x4)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN3x3)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_915 = Coupling(name = 'GC_915',
                  value = '(cw*ee*complexconjugate(NN4x1)*complexconjugate(NN4x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (ee*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.cos(beta))/((-1 + sw)*sw*(1 + sw)) + (ee*sw*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.cos(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complexconjugate(NN4x1)*complexconjugate(NN4x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) + (ee*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*sw*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.sin(beta))/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_916 = Coupling(name = 'GC_916',
                  value = '(ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_917 = Coupling(name = 'GC_917',
                  value = '(ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_918 = Coupling(name = 'GC_918',
                  value = '(ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN3x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_919 = Coupling(name = 'GC_919',
                  value = '(ee*complex(0,1)*complexconjugate(NN4x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN4x3)*complexconjugate(UU1x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_920 = Coupling(name = 'GC_920',
                  value = '(ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN1x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN1x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_921 = Coupling(name = 'GC_921',
                  value = '(ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN2x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN2x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_922 = Coupling(name = 'GC_922',
                  value = '(ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN3x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN3x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_923 = Coupling(name = 'GC_923',
                  value = '(ee*complex(0,1)*complexconjugate(NN4x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)) - (ee*complex(0,1)*sw*complexconjugate(NN4x3)*complexconjugate(UU2x1)*cmath.sin(beta))/((-1 + sw)*(1 + sw)) - (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*sw*(1 + sw)*cmath.sqrt(2)) + (ee*complex(0,1)*sw*complexconjugate(NN4x2)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw)*(1 + sw)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_924 = Coupling(name = 'GC_924',
                  value = '-((ee*complexconjugate(UU1x1)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU1x2)*complexconjugate(VV1x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_925 = Coupling(name = 'GC_925',
                  value = '-((ee*complexconjugate(UU2x1)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU2x2)*complexconjugate(VV1x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_926 = Coupling(name = 'GC_926',
                  value = '-((ee*complexconjugate(UU1x1)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU1x2)*complexconjugate(VV2x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_927 = Coupling(name = 'GC_927',
                  value = '-((ee*complexconjugate(UU2x1)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU2x2)*complexconjugate(VV2x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_928 = Coupling(name = 'GC_928',
                  value = '-(ee*complex(0,1)*cmath.cos(beta)*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*cmath.cos(alp)*cmath.sin(beta))/(2.*sw)',
                  order = {'QED':1})

GC_929 = Coupling(name = 'GC_929',
                  value = '(ee**2*complex(0,1)*cmath.cos(beta)*cmath.sin(alp))/(2.*sw) - (ee**2*complex(0,1)*cmath.cos(alp)*cmath.sin(beta))/(2.*sw)',
                  order = {'QED':2})

GC_930 = Coupling(name = 'GC_930',
                  value = '(cw*ee**2*complex(0,1)*cmath.cos(beta)*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (cw*ee**2*complex(0,1)*cmath.cos(alp)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_931 = Coupling(name = 'GC_931',
                  value = '(cw*ee*cmath.cos(beta)*cmath.sin(alp))/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*cmath.cos(alp)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_932 = Coupling(name = 'GC_932',
                  value = '-(ee*complex(0,1)*cmath.cos(alp)*cmath.cos(beta))/(2.*sw) - (ee*complex(0,1)*cmath.sin(alp)*cmath.sin(beta))/(2.*sw)',
                  order = {'QED':1})

GC_933 = Coupling(name = 'GC_933',
                  value = '(ee**2*complex(0,1)*cmath.cos(alp)*cmath.cos(beta))/(2.*sw) + (ee**2*complex(0,1)*cmath.sin(alp)*cmath.sin(beta))/(2.*sw)',
                  order = {'QED':2})

GC_934 = Coupling(name = 'GC_934',
                  value = '(cw*ee**2*complex(0,1)*cmath.cos(alp)*cmath.cos(beta))/(2.*(-1 + sw)*(1 + sw)) + (cw*ee**2*complex(0,1)*cmath.sin(alp)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_935 = Coupling(name = 'GC_935',
                  value = '(cw*ee*cmath.cos(alp)*cmath.cos(beta))/(2.*(-1 + sw)*sw*(1 + sw)) + (cw*ee*cmath.sin(alp)*cmath.sin(beta))/(2.*(-1 + sw)*sw*(1 + sw))',
                  order = {'QED':1})

GC_936 = Coupling(name = 'GC_936',
                  value = '-(ee*complex(0,1)*cmath.cos(beta)**2) - ee*complex(0,1)*cmath.sin(beta)**2',
                  order = {'QED':1})

GC_937 = Coupling(name = 'GC_937',
                  value = '2*ee**2*complex(0,1)*cmath.cos(beta)**2 + 2*ee**2*complex(0,1)*cmath.sin(beta)**2',
                  order = {'QED':2})

GC_938 = Coupling(name = 'GC_938',
                  value = '(ee**2*complex(0,1)*cmath.cos(beta)**2)/(2.*sw**2) + (ee**2*complex(0,1)*cmath.sin(beta)**2)/(2.*sw**2)',
                  order = {'QED':2})

GC_939 = Coupling(name = 'GC_939',
                  value = '(ee*cmath.cos(beta)**2)/(2.*sw) + (ee*cmath.sin(beta)**2)/(2.*sw)',
                  order = {'QED':1})

GC_940 = Coupling(name = 'GC_940',
                  value = '-(ee**2*cmath.cos(beta)**2)/(2.*sw) - (ee**2*cmath.sin(beta)**2)/(2.*sw)',
                  order = {'QED':2})

GC_941 = Coupling(name = 'GC_941',
                  value = '(ee**2*cmath.cos(beta)**2)/(2.*sw) + (ee**2*cmath.sin(beta)**2)/(2.*sw)',
                  order = {'QED':2})

GC_942 = Coupling(name = 'GC_942',
                  value = '-(cw*ee**2*cmath.cos(beta)**2)/(2.*(-1 + sw)*(1 + sw)) - (cw*ee**2*cmath.sin(beta)**2)/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_943 = Coupling(name = 'GC_943',
                  value = '(cw*ee**2*cmath.cos(beta)**2)/(2.*(-1 + sw)*(1 + sw)) + (cw*ee**2*cmath.sin(beta)**2)/(2.*(-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_944 = Coupling(name = 'GC_944',
                  value = '-(ee**2*complex(0,1)*cmath.cos(beta)**2)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*cmath.sin(beta)**2)/(2.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':2})

GC_945 = Coupling(name = 'GC_945',
                  value = '(cw*ee*complex(0,1)*cmath.cos(beta)**2)/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*sw*cmath.cos(beta)**2)/((-1 + sw)*(1 + sw)) + (cw*ee*complex(0,1)*cmath.sin(beta)**2)/(2.*(-1 + sw)*sw*(1 + sw)) - (cw*ee*complex(0,1)*sw*cmath.sin(beta)**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':1})

GC_946 = Coupling(name = 'GC_946',
                  value = '-((cw*ee**2*complex(0,1)*cmath.cos(beta)**2)/((-1 + sw)*sw*(1 + sw))) + (2*cw*ee**2*complex(0,1)*sw*cmath.cos(beta)**2)/((-1 + sw)*(1 + sw)) - (cw*ee**2*complex(0,1)*cmath.sin(beta)**2)/((-1 + sw)*sw*(1 + sw)) + (2*cw*ee**2*complex(0,1)*sw*cmath.sin(beta)**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_947 = Coupling(name = 'GC_947',
                  value = '(2*ee**2*complex(0,1)*cmath.cos(beta)**2)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*cmath.cos(beta)**2)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*sw**2*cmath.cos(beta)**2)/((-1 + sw)*(1 + sw)) + (2*ee**2*complex(0,1)*cmath.sin(beta)**2)/((-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*cmath.sin(beta)**2)/(2.*(-1 + sw)*sw**2*(1 + sw)) - (2*ee**2*complex(0,1)*sw**2*cmath.sin(beta)**2)/((-1 + sw)*(1 + sw))',
                  order = {'QED':2})

GC_948 = Coupling(name = 'GC_948',
                  value = '(ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.cos(beta)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vd*cmath.sin(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_949 = Coupling(name = 'GC_949',
                  value = '(ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.cos(beta)**2*cmath.sin(alp))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*vd*cmath.cos(beta)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(beta)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(beta)*cmath.sin(beta))/(2.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vu*cmath.cos(beta)*cmath.sin(alp)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.cos(beta)*cmath.sin(alp)*cmath.sin(beta))/(2.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.sin(beta)**2)/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vd*cmath.sin(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_950 = Coupling(name = 'GC_950',
                  value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vu*cmath.cos(beta)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.sin(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

GC_951 = Coupling(name = 'GC_951',
                  value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(beta)**2)/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vu*cmath.cos(beta)**2*cmath.sin(alp))/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(beta)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(beta)*cmath.sin(beta))/(2.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vd*cmath.cos(beta)*cmath.sin(alp)*cmath.sin(beta))/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.cos(beta)*cmath.sin(alp)*cmath.sin(beta))/(2.*(-1 + sw)*sw**2*(1 + sw)) + (ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw)) - (ee**2*complex(0,1)*vu*cmath.sin(alp)*cmath.sin(beta)**2)/(2.*(-1 + sw)*(1 + sw)) + (ee**2*complex(0,1)*vu*cmath.sin(alp)*cmath.sin(beta)**2)/(4.*(-1 + sw)*sw**2*(1 + sw))',
                  order = {'QED':1})

