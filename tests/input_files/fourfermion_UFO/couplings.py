# This file was automatically created by FeynRules 1.7.51
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Thu 2 Aug 2012 10:15:24


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec



GC_1 = Coupling(name = 'GC_1',
                value = '2*complex(0,1)*g1',
                order = {'NP':1})

GC_2 = Coupling(name = 'GC_2',
                value = '2*complex(0,1)*g10',
                order = {'NP':1})

GC_3 = Coupling(name = 'GC_3',
                value = 'complex(0,1)*g11 - g12',
                order = {'NP':1})

GC_4 = Coupling(name = 'GC_4',
                value = 'complex(0,1)*g11 + g12',
                order = {'NP':1})

GC_5 = Coupling(name = 'GC_5',
                value = 'complex(0,1)*g2 - g3',
                order = {'NP':1})

GC_6 = Coupling(name = 'GC_6',
                value = 'complex(0,1)*g2 + g3',
                order = {'NP':1})

GC_7 = Coupling(name = 'GC_7',
                value = 'complex(0,1)*g10 + complex(0,1)*g4',
                order = {'NP':1})

GC_8 = Coupling(name = 'GC_8',
                value = '2*complex(0,1)*g4',
                order = {'NP':1})

GC_9 = Coupling(name = 'GC_9',
                value = 'complex(0,1)*g5 - g6',
                order = {'NP':1})

GC_10 = Coupling(name = 'GC_10',
                 value = 'complex(0,1)*g5 + g6',
                 order = {'NP':1})

GC_11 = Coupling(name = 'GC_11',
                 value = 'complex(0,1)*g1 + complex(0,1)*g7',
                 order = {'NP':1})

GC_12 = Coupling(name = 'GC_12',
                 value = '2*complex(0,1)*g7',
                 order = {'NP':1})

GC_13 = Coupling(name = 'GC_13',
                 value = 'complex(0,1)*g8 - g9',
                 order = {'NP':1})

GC_14 = Coupling(name = 'GC_14',
                 value = 'complex(0,1)*g8 + g9',
                 order = {'NP':1})

GC_15 = Coupling(name = 'GC_15',
                 value = '-((complex(0,1)*g11**2)/Mphi**2) - (complex(0,1)*g12**2)/Mphi**2',
                 order = {'NP':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '-((complex(0,1)*g11**2)/Mphi**2) - (2*g11*g12)/Mphi**2 + (complex(0,1)*g12**2)/Mphi**2',
                 order = {'NP':2})

GC_17 = Coupling(name = 'GC_17',
                 value = '-((complex(0,1)*g11**2)/Mphi**2) + (2*g11*g12)/Mphi**2 + (complex(0,1)*g12**2)/Mphi**2',
                 order = {'NP':2})

GC_18 = Coupling(name = 'GC_18',
                 value = '-((complex(0,1)*g11*g2)/Mphi**2) + (g12*g2)/Mphi**2 - (g11*g3)/Mphi**2 - (complex(0,1)*g12*g3)/Mphi**2',
                 order = {'NP':2})

GC_19 = Coupling(name = 'GC_19',
                 value = '-((complex(0,1)*g11*g2)/Mphi**2) - (g12*g2)/Mphi**2 + (g11*g3)/Mphi**2 - (complex(0,1)*g12*g3)/Mphi**2',
                 order = {'NP':2})

GC_20 = Coupling(name = 'GC_20',
                 value = '-((complex(0,1)*g11*g2)/Mphi**2) - (g12*g2)/Mphi**2 - (g11*g3)/Mphi**2 + (complex(0,1)*g12*g3)/Mphi**2',
                 order = {'NP':2})

GC_21 = Coupling(name = 'GC_21',
                 value = '-((complex(0,1)*g11*g2)/Mphi**2) + (g12*g2)/Mphi**2 + (g11*g3)/Mphi**2 + (complex(0,1)*g12*g3)/Mphi**2',
                 order = {'NP':2})

GC_22 = Coupling(name = 'GC_22',
                 value = '-((complex(0,1)*g2**2)/Mphi**2) - (complex(0,1)*g3**2)/Mphi**2',
                 order = {'NP':2})

GC_23 = Coupling(name = 'GC_23',
                 value = '-((complex(0,1)*g2**2)/Mphi**2) - (2*g2*g3)/Mphi**2 + (complex(0,1)*g3**2)/Mphi**2',
                 order = {'NP':2})

GC_24 = Coupling(name = 'GC_24',
                 value = '-((complex(0,1)*g2**2)/Mphi**2) + (2*g2*g3)/Mphi**2 + (complex(0,1)*g3**2)/Mphi**2',
                 order = {'NP':2})

GC_25 = Coupling(name = 'GC_25',
                 value = '-((complex(0,1)*g10*g11)/Mphi**2) - (g10*g12)/Mphi**2 - (complex(0,1)*g11*g4)/Mphi**2 - (g12*g4)/Mphi**2',
                 order = {'NP':2})

GC_26 = Coupling(name = 'GC_26',
                 value = '-((complex(0,1)*g10*g11)/Mphi**2) + (g10*g12)/Mphi**2 - (complex(0,1)*g11*g4)/Mphi**2 + (g12*g4)/Mphi**2',
                 order = {'NP':2})

GC_27 = Coupling(name = 'GC_27',
                 value = '-((complex(0,1)*g10*g2)/Mphi**2) - (g10*g3)/Mphi**2 - (complex(0,1)*g2*g4)/Mphi**2 - (g3*g4)/Mphi**2',
                 order = {'NP':2})

GC_28 = Coupling(name = 'GC_28',
                 value = '-((complex(0,1)*g10*g2)/Mphi**2) + (g10*g3)/Mphi**2 - (complex(0,1)*g2*g4)/Mphi**2 + (g3*g4)/Mphi**2',
                 order = {'NP':2})

GC_29 = Coupling(name = 'GC_29',
                 value = '-((complex(0,1)*g10**2)/Mphi**2) - (2*complex(0,1)*g10*g4)/Mphi**2 - (complex(0,1)*g4**2)/Mphi**2',
                 order = {'NP':2})

GC_30 = Coupling(name = 'GC_30',
                 value = '-((complex(0,1)*g11*g5)/Mphi**2) + (g12*g5)/Mphi**2 - (g11*g6)/Mphi**2 - (complex(0,1)*g12*g6)/Mphi**2',
                 order = {'NP':2})

GC_31 = Coupling(name = 'GC_31',
                 value = '-((complex(0,1)*g11*g5)/Mphi**2) - (g12*g5)/Mphi**2 + (g11*g6)/Mphi**2 - (complex(0,1)*g12*g6)/Mphi**2',
                 order = {'NP':2})

GC_32 = Coupling(name = 'GC_32',
                 value = '-((complex(0,1)*g11*g5)/Mphi**2) - (g12*g5)/Mphi**2 - (g11*g6)/Mphi**2 + (complex(0,1)*g12*g6)/Mphi**2',
                 order = {'NP':2})

GC_33 = Coupling(name = 'GC_33',
                 value = '-((complex(0,1)*g11*g5)/Mphi**2) + (g12*g5)/Mphi**2 + (g11*g6)/Mphi**2 + (complex(0,1)*g12*g6)/Mphi**2',
                 order = {'NP':2})

GC_34 = Coupling(name = 'GC_34',
                 value = '-((complex(0,1)*g2*g5)/Mphi**2) + (g3*g5)/Mphi**2 - (g2*g6)/Mphi**2 - (complex(0,1)*g3*g6)/Mphi**2',
                 order = {'NP':2})

GC_35 = Coupling(name = 'GC_35',
                 value = '-((complex(0,1)*g2*g5)/Mphi**2) - (g3*g5)/Mphi**2 + (g2*g6)/Mphi**2 - (complex(0,1)*g3*g6)/Mphi**2',
                 order = {'NP':2})

GC_36 = Coupling(name = 'GC_36',
                 value = '-((complex(0,1)*g2*g5)/Mphi**2) - (g3*g5)/Mphi**2 - (g2*g6)/Mphi**2 + (complex(0,1)*g3*g6)/Mphi**2',
                 order = {'NP':2})

GC_37 = Coupling(name = 'GC_37',
                 value = '-((complex(0,1)*g2*g5)/Mphi**2) + (g3*g5)/Mphi**2 + (g2*g6)/Mphi**2 + (complex(0,1)*g3*g6)/Mphi**2',
                 order = {'NP':2})

GC_38 = Coupling(name = 'GC_38',
                 value = '-((complex(0,1)*g10*g5)/Mphi**2) - (complex(0,1)*g4*g5)/Mphi**2 - (g10*g6)/Mphi**2 - (g4*g6)/Mphi**2',
                 order = {'NP':2})

GC_39 = Coupling(name = 'GC_39',
                 value = '-((complex(0,1)*g10*g5)/Mphi**2) - (complex(0,1)*g4*g5)/Mphi**2 + (g10*g6)/Mphi**2 + (g4*g6)/Mphi**2',
                 order = {'NP':2})

GC_40 = Coupling(name = 'GC_40',
                 value = '-((complex(0,1)*g5**2)/Mphi**2) - (complex(0,1)*g6**2)/Mphi**2',
                 order = {'NP':2})

GC_41 = Coupling(name = 'GC_41',
                 value = '-((complex(0,1)*g5**2)/Mphi**2) - (2*g5*g6)/Mphi**2 + (complex(0,1)*g6**2)/Mphi**2',
                 order = {'NP':2})

GC_42 = Coupling(name = 'GC_42',
                 value = '-((complex(0,1)*g5**2)/Mphi**2) + (2*g5*g6)/Mphi**2 + (complex(0,1)*g6**2)/Mphi**2',
                 order = {'NP':2})

GC_43 = Coupling(name = 'GC_43',
                 value = '-((complex(0,1)*g1*g11)/Mphi**2) - (g1*g12)/Mphi**2 - (complex(0,1)*g11*g7)/Mphi**2 - (g12*g7)/Mphi**2',
                 order = {'NP':2})

GC_44 = Coupling(name = 'GC_44',
                 value = '-((complex(0,1)*g1*g11)/Mphi**2) + (g1*g12)/Mphi**2 - (complex(0,1)*g11*g7)/Mphi**2 + (g12*g7)/Mphi**2',
                 order = {'NP':2})

GC_45 = Coupling(name = 'GC_45',
                 value = '-((complex(0,1)*g1*g2)/Mphi**2) - (g1*g3)/Mphi**2 - (complex(0,1)*g2*g7)/Mphi**2 - (g3*g7)/Mphi**2',
                 order = {'NP':2})

GC_46 = Coupling(name = 'GC_46',
                 value = '-((complex(0,1)*g1*g2)/Mphi**2) + (g1*g3)/Mphi**2 - (complex(0,1)*g2*g7)/Mphi**2 + (g3*g7)/Mphi**2',
                 order = {'NP':2})

GC_47 = Coupling(name = 'GC_47',
                 value = '-((complex(0,1)*g1*g10)/Mphi**2) - (complex(0,1)*g1*g4)/Mphi**2 - (complex(0,1)*g10*g7)/Mphi**2 - (complex(0,1)*g4*g7)/Mphi**2',
                 order = {'NP':2})

GC_48 = Coupling(name = 'GC_48',
                 value = '-((complex(0,1)*g1*g5)/Mphi**2) - (g1*g6)/Mphi**2 - (complex(0,1)*g5*g7)/Mphi**2 - (g6*g7)/Mphi**2',
                 order = {'NP':2})

GC_49 = Coupling(name = 'GC_49',
                 value = '-((complex(0,1)*g1*g5)/Mphi**2) + (g1*g6)/Mphi**2 - (complex(0,1)*g5*g7)/Mphi**2 + (g6*g7)/Mphi**2',
                 order = {'NP':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '-((complex(0,1)*g1**2)/Mphi**2) - (2*complex(0,1)*g1*g7)/Mphi**2 - (complex(0,1)*g7**2)/Mphi**2',
                 order = {'NP':2})

GC_51 = Coupling(name = 'GC_51',
                 value = '-((complex(0,1)*g11*g8)/Mphi**2) + (g12*g8)/Mphi**2 - (g11*g9)/Mphi**2 - (complex(0,1)*g12*g9)/Mphi**2',
                 order = {'NP':2})

GC_52 = Coupling(name = 'GC_52',
                 value = '-((complex(0,1)*g11*g8)/Mphi**2) - (g12*g8)/Mphi**2 + (g11*g9)/Mphi**2 - (complex(0,1)*g12*g9)/Mphi**2',
                 order = {'NP':2})

GC_53 = Coupling(name = 'GC_53',
                 value = '-((complex(0,1)*g11*g8)/Mphi**2) - (g12*g8)/Mphi**2 - (g11*g9)/Mphi**2 + (complex(0,1)*g12*g9)/Mphi**2',
                 order = {'NP':2})

GC_54 = Coupling(name = 'GC_54',
                 value = '-((complex(0,1)*g11*g8)/Mphi**2) + (g12*g8)/Mphi**2 + (g11*g9)/Mphi**2 + (complex(0,1)*g12*g9)/Mphi**2',
                 order = {'NP':2})

GC_55 = Coupling(name = 'GC_55',
                 value = '-((complex(0,1)*g2*g8)/Mphi**2) + (g3*g8)/Mphi**2 - (g2*g9)/Mphi**2 - (complex(0,1)*g3*g9)/Mphi**2',
                 order = {'NP':2})

GC_56 = Coupling(name = 'GC_56',
                 value = '-((complex(0,1)*g2*g8)/Mphi**2) - (g3*g8)/Mphi**2 + (g2*g9)/Mphi**2 - (complex(0,1)*g3*g9)/Mphi**2',
                 order = {'NP':2})

GC_57 = Coupling(name = 'GC_57',
                 value = '-((complex(0,1)*g2*g8)/Mphi**2) - (g3*g8)/Mphi**2 - (g2*g9)/Mphi**2 + (complex(0,1)*g3*g9)/Mphi**2',
                 order = {'NP':2})

GC_58 = Coupling(name = 'GC_58',
                 value = '-((complex(0,1)*g2*g8)/Mphi**2) + (g3*g8)/Mphi**2 + (g2*g9)/Mphi**2 + (complex(0,1)*g3*g9)/Mphi**2',
                 order = {'NP':2})

GC_59 = Coupling(name = 'GC_59',
                 value = '-((complex(0,1)*g10*g8)/Mphi**2) - (complex(0,1)*g4*g8)/Mphi**2 - (g10*g9)/Mphi**2 - (g4*g9)/Mphi**2',
                 order = {'NP':2})

GC_60 = Coupling(name = 'GC_60',
                 value = '-((complex(0,1)*g10*g8)/Mphi**2) - (complex(0,1)*g4*g8)/Mphi**2 + (g10*g9)/Mphi**2 + (g4*g9)/Mphi**2',
                 order = {'NP':2})

GC_61 = Coupling(name = 'GC_61',
                 value = '-((complex(0,1)*g5*g8)/Mphi**2) + (g6*g8)/Mphi**2 - (g5*g9)/Mphi**2 - (complex(0,1)*g6*g9)/Mphi**2',
                 order = {'NP':2})

GC_62 = Coupling(name = 'GC_62',
                 value = '-((complex(0,1)*g5*g8)/Mphi**2) - (g6*g8)/Mphi**2 + (g5*g9)/Mphi**2 - (complex(0,1)*g6*g9)/Mphi**2',
                 order = {'NP':2})

GC_63 = Coupling(name = 'GC_63',
                 value = '-((complex(0,1)*g5*g8)/Mphi**2) - (g6*g8)/Mphi**2 - (g5*g9)/Mphi**2 + (complex(0,1)*g6*g9)/Mphi**2',
                 order = {'NP':2})

GC_64 = Coupling(name = 'GC_64',
                 value = '-((complex(0,1)*g5*g8)/Mphi**2) + (g6*g8)/Mphi**2 + (g5*g9)/Mphi**2 + (complex(0,1)*g6*g9)/Mphi**2',
                 order = {'NP':2})

GC_65 = Coupling(name = 'GC_65',
                 value = '-((complex(0,1)*g1*g8)/Mphi**2) - (complex(0,1)*g7*g8)/Mphi**2 - (g1*g9)/Mphi**2 - (g7*g9)/Mphi**2',
                 order = {'NP':2})

GC_66 = Coupling(name = 'GC_66',
                 value = '-((complex(0,1)*g1*g8)/Mphi**2) - (complex(0,1)*g7*g8)/Mphi**2 + (g1*g9)/Mphi**2 + (g7*g9)/Mphi**2',
                 order = {'NP':2})

GC_67 = Coupling(name = 'GC_67',
                 value = '-((complex(0,1)*g8**2)/Mphi**2) - (complex(0,1)*g9**2)/Mphi**2',
                 order = {'NP':2})

GC_68 = Coupling(name = 'GC_68',
                 value = '-((complex(0,1)*g8**2)/Mphi**2) - (2*g8*g9)/Mphi**2 + (complex(0,1)*g9**2)/Mphi**2',
                 order = {'NP':2})

GC_69 = Coupling(name = 'GC_69',
                 value = '-((complex(0,1)*g8**2)/Mphi**2) + (2*g8*g9)/Mphi**2 + (complex(0,1)*g9**2)/Mphi**2',
                 order = {'NP':2})

GC_70 = Coupling(name = 'GC_70',
                 value = '(2*complex(0,1)*g1*g11)/MV**2 - (2*g1*g12)/MV**2',
                 order = {'NP':2})

GC_71 = Coupling(name = 'GC_71',
                 value = '(2*complex(0,1)*g1*g11)/MV**2 + (2*g1*g12)/MV**2',
                 order = {'NP':2})

GC_72 = Coupling(name = 'GC_72',
                 value = '(2*complex(0,1)*g10*g11)/MV**2 - (2*g10*g12)/MV**2',
                 order = {'NP':2})

GC_73 = Coupling(name = 'GC_73',
                 value = '(2*complex(0,1)*g10*g11)/MV**2 + (2*g10*g12)/MV**2',
                 order = {'NP':2})

GC_74 = Coupling(name = 'GC_74',
                 value = '(complex(0,1)*g11**2)/MV**2 - (2*g11*g12)/MV**2 - (complex(0,1)*g12**2)/MV**2',
                 order = {'NP':2})

GC_75 = Coupling(name = 'GC_75',
                 value = '(complex(0,1)*g11**2)/MV**2 + (2*g11*g12)/MV**2 - (complex(0,1)*g12**2)/MV**2',
                 order = {'NP':2})

GC_76 = Coupling(name = 'GC_76',
                 value = '(complex(0,1)*g11**2)/MV**2 + (complex(0,1)*g12**2)/MV**2',
                 order = {'NP':2})

GC_77 = Coupling(name = 'GC_77',
                 value = '(2*complex(0,1)*g1*g2)/MV**2 - (2*g1*g3)/MV**2',
                 order = {'NP':2})

GC_78 = Coupling(name = 'GC_78',
                 value = '(2*complex(0,1)*g1*g2)/MV**2 + (2*g1*g3)/MV**2',
                 order = {'NP':2})

GC_79 = Coupling(name = 'GC_79',
                 value = '(2*complex(0,1)*g10*g2)/MV**2 - (2*g10*g3)/MV**2',
                 order = {'NP':2})

GC_80 = Coupling(name = 'GC_80',
                 value = '(2*complex(0,1)*g10*g2)/MV**2 + (2*g10*g3)/MV**2',
                 order = {'NP':2})

GC_81 = Coupling(name = 'GC_81',
                 value = '(complex(0,1)*g11*g2)/MV**2 - (g12*g2)/MV**2 - (g11*g3)/MV**2 - (complex(0,1)*g12*g3)/MV**2',
                 order = {'NP':2})

GC_82 = Coupling(name = 'GC_82',
                 value = '(complex(0,1)*g11*g2)/MV**2 + (g12*g2)/MV**2 + (g11*g3)/MV**2 - (complex(0,1)*g12*g3)/MV**2',
                 order = {'NP':2})

GC_83 = Coupling(name = 'GC_83',
                 value = '(complex(0,1)*g11*g2)/MV**2 + (g12*g2)/MV**2 - (g11*g3)/MV**2 + (complex(0,1)*g12*g3)/MV**2',
                 order = {'NP':2})

GC_84 = Coupling(name = 'GC_84',
                 value = '(complex(0,1)*g11*g2)/MV**2 - (g12*g2)/MV**2 + (g11*g3)/MV**2 + (complex(0,1)*g12*g3)/MV**2',
                 order = {'NP':2})

GC_85 = Coupling(name = 'GC_85',
                 value = '(complex(0,1)*g2**2)/MV**2 - (2*g2*g3)/MV**2 - (complex(0,1)*g3**2)/MV**2',
                 order = {'NP':2})

GC_86 = Coupling(name = 'GC_86',
                 value = '(complex(0,1)*g2**2)/MV**2 + (2*g2*g3)/MV**2 - (complex(0,1)*g3**2)/MV**2',
                 order = {'NP':2})

GC_87 = Coupling(name = 'GC_87',
                 value = '(complex(0,1)*g2**2)/MV**2 + (complex(0,1)*g3**2)/MV**2',
                 order = {'NP':2})

GC_88 = Coupling(name = 'GC_88',
                 value = '(2*complex(0,1)*g11*g4)/MV**2 - (2*g12*g4)/MV**2',
                 order = {'NP':2})

GC_89 = Coupling(name = 'GC_89',
                 value = '(2*complex(0,1)*g11*g4)/MV**2 + (2*g12*g4)/MV**2',
                 order = {'NP':2})

GC_90 = Coupling(name = 'GC_90',
                 value = '(2*complex(0,1)*g2*g4)/MV**2 - (2*g3*g4)/MV**2',
                 order = {'NP':2})

GC_91 = Coupling(name = 'GC_91',
                 value = '(2*complex(0,1)*g2*g4)/MV**2 + (2*g3*g4)/MV**2',
                 order = {'NP':2})

GC_92 = Coupling(name = 'GC_92',
                 value = '(2*complex(0,1)*g1*g5)/MV**2 - (2*g1*g6)/MV**2',
                 order = {'NP':2})

GC_93 = Coupling(name = 'GC_93',
                 value = '(2*complex(0,1)*g1*g5)/MV**2 + (2*g1*g6)/MV**2',
                 order = {'NP':2})

GC_94 = Coupling(name = 'GC_94',
                 value = '(2*complex(0,1)*g10*g5)/MV**2 - (2*g10*g6)/MV**2',
                 order = {'NP':2})

GC_95 = Coupling(name = 'GC_95',
                 value = '(2*complex(0,1)*g10*g5)/MV**2 + (2*g10*g6)/MV**2',
                 order = {'NP':2})

GC_96 = Coupling(name = 'GC_96',
                 value = '(complex(0,1)*g11*g5)/MV**2 - (g12*g5)/MV**2 - (g11*g6)/MV**2 - (complex(0,1)*g12*g6)/MV**2',
                 order = {'NP':2})

GC_97 = Coupling(name = 'GC_97',
                 value = '(complex(0,1)*g11*g5)/MV**2 + (g12*g5)/MV**2 + (g11*g6)/MV**2 - (complex(0,1)*g12*g6)/MV**2',
                 order = {'NP':2})

GC_98 = Coupling(name = 'GC_98',
                 value = '(complex(0,1)*g11*g5)/MV**2 + (g12*g5)/MV**2 - (g11*g6)/MV**2 + (complex(0,1)*g12*g6)/MV**2',
                 order = {'NP':2})

GC_99 = Coupling(name = 'GC_99',
                 value = '(complex(0,1)*g11*g5)/MV**2 - (g12*g5)/MV**2 + (g11*g6)/MV**2 + (complex(0,1)*g12*g6)/MV**2',
                 order = {'NP':2})

GC_100 = Coupling(name = 'GC_100',
                  value = '(complex(0,1)*g2*g5)/MV**2 - (g3*g5)/MV**2 - (g2*g6)/MV**2 - (complex(0,1)*g3*g6)/MV**2',
                  order = {'NP':2})

GC_101 = Coupling(name = 'GC_101',
                  value = '(complex(0,1)*g2*g5)/MV**2 + (g3*g5)/MV**2 + (g2*g6)/MV**2 - (complex(0,1)*g3*g6)/MV**2',
                  order = {'NP':2})

GC_102 = Coupling(name = 'GC_102',
                  value = '(complex(0,1)*g2*g5)/MV**2 + (g3*g5)/MV**2 - (g2*g6)/MV**2 + (complex(0,1)*g3*g6)/MV**2',
                  order = {'NP':2})

GC_103 = Coupling(name = 'GC_103',
                  value = '(complex(0,1)*g2*g5)/MV**2 - (g3*g5)/MV**2 + (g2*g6)/MV**2 + (complex(0,1)*g3*g6)/MV**2',
                  order = {'NP':2})

GC_104 = Coupling(name = 'GC_104',
                  value = '(2*complex(0,1)*g4*g5)/MV**2 - (2*g4*g6)/MV**2',
                  order = {'NP':2})

GC_105 = Coupling(name = 'GC_105',
                  value = '(2*complex(0,1)*g4*g5)/MV**2 + (2*g4*g6)/MV**2',
                  order = {'NP':2})

GC_106 = Coupling(name = 'GC_106',
                  value = '(complex(0,1)*g5**2)/MV**2 - (2*g5*g6)/MV**2 - (complex(0,1)*g6**2)/MV**2',
                  order = {'NP':2})

GC_107 = Coupling(name = 'GC_107',
                  value = '(complex(0,1)*g5**2)/MV**2 + (2*g5*g6)/MV**2 - (complex(0,1)*g6**2)/MV**2',
                  order = {'NP':2})

GC_108 = Coupling(name = 'GC_108',
                  value = '(complex(0,1)*g5**2)/MV**2 + (complex(0,1)*g6**2)/MV**2',
                  order = {'NP':2})

GC_109 = Coupling(name = 'GC_109',
                  value = '(2*complex(0,1)*g11*g7)/MV**2 - (2*g12*g7)/MV**2',
                  order = {'NP':2})

GC_110 = Coupling(name = 'GC_110',
                  value = '(2*complex(0,1)*g11*g7)/MV**2 + (2*g12*g7)/MV**2',
                  order = {'NP':2})

GC_111 = Coupling(name = 'GC_111',
                  value = '(2*complex(0,1)*g2*g7)/MV**2 - (2*g3*g7)/MV**2',
                  order = {'NP':2})

GC_112 = Coupling(name = 'GC_112',
                  value = '(2*complex(0,1)*g2*g7)/MV**2 + (2*g3*g7)/MV**2',
                  order = {'NP':2})

GC_113 = Coupling(name = 'GC_113',
                  value = '(2*complex(0,1)*g5*g7)/MV**2 - (2*g6*g7)/MV**2',
                  order = {'NP':2})

GC_114 = Coupling(name = 'GC_114',
                  value = '(2*complex(0,1)*g5*g7)/MV**2 + (2*g6*g7)/MV**2',
                  order = {'NP':2})

GC_115 = Coupling(name = 'GC_115',
                  value = '(2*complex(0,1)*g1*g8)/MV**2 - (2*g1*g9)/MV**2',
                  order = {'NP':2})

GC_116 = Coupling(name = 'GC_116',
                  value = '(2*complex(0,1)*g1*g8)/MV**2 + (2*g1*g9)/MV**2',
                  order = {'NP':2})

GC_117 = Coupling(name = 'GC_117',
                  value = '(2*complex(0,1)*g10*g8)/MV**2 - (2*g10*g9)/MV**2',
                  order = {'NP':2})

GC_118 = Coupling(name = 'GC_118',
                  value = '(2*complex(0,1)*g10*g8)/MV**2 + (2*g10*g9)/MV**2',
                  order = {'NP':2})

GC_119 = Coupling(name = 'GC_119',
                  value = '(complex(0,1)*g11*g8)/MV**2 - (g12*g8)/MV**2 - (g11*g9)/MV**2 - (complex(0,1)*g12*g9)/MV**2',
                  order = {'NP':2})

GC_120 = Coupling(name = 'GC_120',
                  value = '(complex(0,1)*g11*g8)/MV**2 + (g12*g8)/MV**2 + (g11*g9)/MV**2 - (complex(0,1)*g12*g9)/MV**2',
                  order = {'NP':2})

GC_121 = Coupling(name = 'GC_121',
                  value = '(complex(0,1)*g11*g8)/MV**2 + (g12*g8)/MV**2 - (g11*g9)/MV**2 + (complex(0,1)*g12*g9)/MV**2',
                  order = {'NP':2})

GC_122 = Coupling(name = 'GC_122',
                  value = '(complex(0,1)*g11*g8)/MV**2 - (g12*g8)/MV**2 + (g11*g9)/MV**2 + (complex(0,1)*g12*g9)/MV**2',
                  order = {'NP':2})

GC_123 = Coupling(name = 'GC_123',
                  value = '(complex(0,1)*g2*g8)/MV**2 - (g3*g8)/MV**2 - (g2*g9)/MV**2 - (complex(0,1)*g3*g9)/MV**2',
                  order = {'NP':2})

GC_124 = Coupling(name = 'GC_124',
                  value = '(complex(0,1)*g2*g8)/MV**2 + (g3*g8)/MV**2 + (g2*g9)/MV**2 - (complex(0,1)*g3*g9)/MV**2',
                  order = {'NP':2})

GC_125 = Coupling(name = 'GC_125',
                  value = '(complex(0,1)*g2*g8)/MV**2 + (g3*g8)/MV**2 - (g2*g9)/MV**2 + (complex(0,1)*g3*g9)/MV**2',
                  order = {'NP':2})

GC_126 = Coupling(name = 'GC_126',
                  value = '(complex(0,1)*g2*g8)/MV**2 - (g3*g8)/MV**2 + (g2*g9)/MV**2 + (complex(0,1)*g3*g9)/MV**2',
                  order = {'NP':2})

GC_127 = Coupling(name = 'GC_127',
                  value = '(2*complex(0,1)*g4*g8)/MV**2 - (2*g4*g9)/MV**2',
                  order = {'NP':2})

GC_128 = Coupling(name = 'GC_128',
                  value = '(2*complex(0,1)*g4*g8)/MV**2 + (2*g4*g9)/MV**2',
                  order = {'NP':2})

GC_129 = Coupling(name = 'GC_129',
                  value = '(complex(0,1)*g5*g8)/MV**2 - (g6*g8)/MV**2 - (g5*g9)/MV**2 - (complex(0,1)*g6*g9)/MV**2',
                  order = {'NP':2})

GC_130 = Coupling(name = 'GC_130',
                  value = '(complex(0,1)*g5*g8)/MV**2 + (g6*g8)/MV**2 + (g5*g9)/MV**2 - (complex(0,1)*g6*g9)/MV**2',
                  order = {'NP':2})

GC_131 = Coupling(name = 'GC_131',
                  value = '(complex(0,1)*g5*g8)/MV**2 + (g6*g8)/MV**2 - (g5*g9)/MV**2 + (complex(0,1)*g6*g9)/MV**2',
                  order = {'NP':2})

GC_132 = Coupling(name = 'GC_132',
                  value = '-((g6*g8)/MV**2) + (g5*g9)/MV**2 + (complex(0,1)*g6*g9)/MV**2',
                  order = {'NP':2})

GC_133 = Coupling(name = 'GC_133',
                  value = '(2*complex(0,1)*g7*g8)/MV**2 - (2*g7*g9)/MV**2',
                  order = {'NP':2})

GC_134 = Coupling(name = 'GC_134',
                  value = '(2*complex(0,1)*g7*g8)/MV**2 + (2*g7*g9)/MV**2',
                  order = {'NP':2})

GC_135 = Coupling(name = 'GC_135',
                  value = '(complex(0,1)*g8**2)/MV**2 - (2*g8*g9)/MV**2 - (complex(0,1)*g9**2)/MV**2',
                  order = {'NP':2})

GC_136 = Coupling(name = 'GC_136',
                  value = '(complex(0,1)*g8**2)/MV**2 + (2*g8*g9)/MV**2 - (complex(0,1)*g9**2)/MV**2',
                  order = {'NP':2})

GC_137 = Coupling(name = 'GC_137',
                  value = '(complex(0,1)*g8**2)/MV**2 + (complex(0,1)*g9**2)/MV**2',
                  order = {'NP':2})

GC_138 = Coupling(name = 'GC_138',
                  value = '(4*complex(0,1)*g1**2)/MV**2',
                  order = {'NP':2})

GC_139 = Coupling(name = 'GC_139',
                  value = '(4*complex(0,1)*g1*g10)/MV**2',
                  order = {'NP':2})

GC_140 = Coupling(name = 'GC_140',
                  value = '(4*complex(0,1)*g10**2)/MV**2',
                  order = {'NP':2})

GC_141 = Coupling(name = 'GC_141',
                  value = '(4*complex(0,1)*g1*g4)/MV**2',
                  order = {'NP':2})

GC_142 = Coupling(name = 'GC_142',
                  value = '(4*complex(0,1)*g10*g4)/MV**2',
                  order = {'NP':2})

GC_143 = Coupling(name = 'GC_143',
                  value = '(4*complex(0,1)*g4**2)/MV**2',
                  order = {'NP':2})

GC_144 = Coupling(name = 'GC_144',
                  value = '(4*complex(0,1)*g1*g7)/MV**2',
                  order = {'NP':2})

GC_145 = Coupling(name = 'GC_145',
                  value = '(4*complex(0,1)*g10*g7)/MV**2',
                  order = {'NP':2})

GC_146 = Coupling(name = 'GC_146',
                  value = '(4*complex(0,1)*g4*g7)/MV**2',
                  order = {'NP':2})

GC_147 = Coupling(name = 'GC_147',
                  value = '(4*complex(0,1)*g7**2)/MV**2',
                  order = {'NP':2})

GC_148 = Coupling(name = 'GC_148',
                  value = '(2*complex(0,1)*g10*g8)/MV**2',
                  order = {'NP':2})

GC_149 = Coupling(name = 'GC_149',
                  value = '(complex(0,1)*g5*g8)/MV**2',
                  order = {'NP':2})

GC_150 = Coupling(name = 'GC_150',
                  value = '(2*g10*g9)/MV**2',
                  order = {'NP':2})

