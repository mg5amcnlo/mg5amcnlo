# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Mon 11 Apr 2011 22:27:17


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
                value = '2*ee**2*complex(0,1)',
                order = {'QED':2})

GC_5 = Coupling(name = 'GC_5',
                value = '-G',
                order = {'QCD':1})

GC_6 = Coupling(name = 'GC_6',
                value = 'complex(0,1)*G',
                order = {'QCD':1})

GC_7 = Coupling(name = 'GC_7',
                value = 'complex(0,1)*G**2',
                order = {'QCD':2})

GC_8 = Coupling(name = 'GC_8',
                value = 'cw*complex(0,1)*gw',
                order = {'QED':1})

GC_9 = Coupling(name = 'GC_9',
                value = '-(complex(0,1)*gw**2)',
                order = {'QED':2})

GC_10 = Coupling(name = 'GC_10',
                 value = 'cw**2*complex(0,1)*gw**2',
                 order = {'QED':2})

GC_11 = Coupling(name = 'GC_11',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_12 = Coupling(name = 'GC_12',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_13 = Coupling(name = 'GC_13',
                 value = '(CKM11*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_14 = Coupling(name = 'GC_14',
                 value = '(CKM12*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_15 = Coupling(name = 'GC_15',
                 value = '(CKM21*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_16 = Coupling(name = 'GC_16',
                 value = '(CKM22*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_17 = Coupling(name = 'GC_17',
                 value = '-(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_18 = Coupling(name = 'GC_18',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_19 = Coupling(name = 'GC_19',
                 value = '-(ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})

GC_20 = Coupling(name = 'GC_20',
                 value = '(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_21 = Coupling(name = 'GC_21',
                 value = 'complex(0,1)*gw*sw',
                 order = {'QED':1})

GC_22 = Coupling(name = 'GC_22',
                 value = '-2*cw*complex(0,1)*gw**2*sw',
                 order = {'QED':2})

GC_23 = Coupling(name = 'GC_23',
                 value = 'complex(0,1)*gw**2*sw**2',
                 order = {'QED':2})

GC_24 = Coupling(name = 'GC_24',
                 value = '-(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_25 = Coupling(name = 'GC_25',
                 value = '(cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})

GC_26 = Coupling(name = 'GC_26',
                 value = '(cw*ee**2*complex(0,1))/sw - (ee**2*complex(0,1)*sw)/cw',
                 order = {'QED':2})

GC_27 = Coupling(name = 'GC_27',
                 value = '-(ee**2*complex(0,1)) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_28 = Coupling(name = 'GC_28',
                 value = '-(ee**2*complex(0,1)*TH21)/(2.*cw) - (ee**2*TH31)/(2.*cw)',
                 order = {'QED':2})

GC_29 = Coupling(name = 'GC_29',
                 value = '-(ee**2*complex(0,1)*TH21)/(2.*cw) + (ee**2*TH31)/(2.*cw)',
                 order = {'QED':2})

GC_30 = Coupling(name = 'GC_30',
                 value = '-(ee*complex(0,1)*TH21)/(2.*sw) + (ee*TH31)/(2.*sw)',
                 order = {'QED':1})

GC_31 = Coupling(name = 'GC_31',
                 value = '(ee*complex(0,1)*TH21)/(2.*sw) + (ee*TH31)/(2.*sw)',
                 order = {'QED':1})

GC_32 = Coupling(name = 'GC_32',
                 value = '(ee**2*complex(0,1)*TH21)/(2.*sw) - (ee**2*TH31)/(2.*sw)',
                 order = {'QED':2})

GC_33 = Coupling(name = 'GC_33',
                 value = '(ee**2*complex(0,1)*TH21)/(2.*sw) + (ee**2*TH31)/(2.*sw)',
                 order = {'QED':2})

GC_34 = Coupling(name = 'GC_34',
                 value = '(ee**2*complex(0,1)*TH11**2)/(2.*sw**2) + (ee**2*complex(0,1)*TH21**2)/(2.*sw**2) + (ee**2*complex(0,1)*TH31**2)/(2.*sw**2)',
                 order = {'QED':2})

GC_35 = Coupling(name = 'GC_35',
                 value = 'ee**2*complex(0,1)*TH11**2 + (cw**2*ee**2*complex(0,1)*TH11**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH11**2)/(2.*cw**2) + ee**2*complex(0,1)*TH21**2 + (cw**2*ee**2*complex(0,1)*TH21**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH21**2)/(2.*cw**2) + ee**2*complex(0,1)*TH31**2 + (cw**2*ee**2*complex(0,1)*TH31**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH31**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_36 = Coupling(name = 'GC_36',
                 value = '-(ee**2*complex(0,1)*TH22)/(2.*cw) - (ee**2*TH32)/(2.*cw)',
                 order = {'QED':2})

GC_37 = Coupling(name = 'GC_37',
                 value = '-(ee**2*complex(0,1)*TH22)/(2.*cw) + (ee**2*TH32)/(2.*cw)',
                 order = {'QED':2})

GC_38 = Coupling(name = 'GC_38',
                 value = '-(ee*complex(0,1)*TH22)/(2.*sw) + (ee*TH32)/(2.*sw)',
                 order = {'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '(ee*complex(0,1)*TH22)/(2.*sw) + (ee*TH32)/(2.*sw)',
                 order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(ee**2*complex(0,1)*TH22)/(2.*sw) - (ee**2*TH32)/(2.*sw)',
                 order = {'QED':2})

GC_41 = Coupling(name = 'GC_41',
                 value = '(ee**2*complex(0,1)*TH22)/(2.*sw) + (ee**2*TH32)/(2.*sw)',
                 order = {'QED':2})

GC_42 = Coupling(name = 'GC_42',
                 value = '-(cw*ee*TH22*TH31)/(2.*sw) - (ee*sw*TH22*TH31)/(2.*cw) + (cw*ee*TH21*TH32)/(2.*sw) + (ee*sw*TH21*TH32)/(2.*cw)',
                 order = {'QED':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '(ee**2*complex(0,1)*TH11*TH12)/(2.*sw**2) + (ee**2*complex(0,1)*TH21*TH22)/(2.*sw**2) + (ee**2*complex(0,1)*TH31*TH32)/(2.*sw**2)',
                 order = {'QED':2})

GC_44 = Coupling(name = 'GC_44',
                 value = 'ee**2*complex(0,1)*TH11*TH12 + (cw**2*ee**2*complex(0,1)*TH11*TH12)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH11*TH12)/(2.*cw**2) + ee**2*complex(0,1)*TH21*TH22 + (cw**2*ee**2*complex(0,1)*TH21*TH22)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH21*TH22)/(2.*cw**2) + ee**2*complex(0,1)*TH31*TH32 + (cw**2*ee**2*complex(0,1)*TH31*TH32)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH31*TH32)/(2.*cw**2)',
                 order = {'QED':2})

GC_45 = Coupling(name = 'GC_45',
                 value = '(ee**2*complex(0,1)*TH12**2)/(2.*sw**2) + (ee**2*complex(0,1)*TH22**2)/(2.*sw**2) + (ee**2*complex(0,1)*TH32**2)/(2.*sw**2)',
                 order = {'QED':2})

GC_46 = Coupling(name = 'GC_46',
                 value = 'ee**2*complex(0,1)*TH12**2 + (cw**2*ee**2*complex(0,1)*TH12**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH12**2)/(2.*cw**2) + ee**2*complex(0,1)*TH22**2 + (cw**2*ee**2*complex(0,1)*TH22**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH22**2)/(2.*cw**2) + ee**2*complex(0,1)*TH32**2 + (cw**2*ee**2*complex(0,1)*TH32**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH32**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_47 = Coupling(name = 'GC_47',
                 value = '-(ee**2*complex(0,1)*TH23)/(2.*cw) - (ee**2*TH33)/(2.*cw)',
                 order = {'QED':2})

GC_48 = Coupling(name = 'GC_48',
                 value = '-(ee**2*complex(0,1)*TH23)/(2.*cw) + (ee**2*TH33)/(2.*cw)',
                 order = {'QED':2})

GC_49 = Coupling(name = 'GC_49',
                 value = '-(ee*complex(0,1)*TH23)/(2.*sw) + (ee*TH33)/(2.*sw)',
                 order = {'QED':1})

GC_50 = Coupling(name = 'GC_50',
                 value = '(ee*complex(0,1)*TH23)/(2.*sw) + (ee*TH33)/(2.*sw)',
                 order = {'QED':1})

GC_51 = Coupling(name = 'GC_51',
                 value = '(ee**2*complex(0,1)*TH23)/(2.*sw) - (ee**2*TH33)/(2.*sw)',
                 order = {'QED':2})

GC_52 = Coupling(name = 'GC_52',
                 value = '(ee**2*complex(0,1)*TH23)/(2.*sw) + (ee**2*TH33)/(2.*sw)',
                 order = {'QED':2})

GC_53 = Coupling(name = 'GC_53',
                 value = '-(cw*ee*TH23*TH31)/(2.*sw) - (ee*sw*TH23*TH31)/(2.*cw) + (cw*ee*TH21*TH33)/(2.*sw) + (ee*sw*TH21*TH33)/(2.*cw)',
                 order = {'QED':1})

GC_54 = Coupling(name = 'GC_54',
                 value = '-(cw*ee*TH23*TH32)/(2.*sw) - (ee*sw*TH23*TH32)/(2.*cw) + (cw*ee*TH22*TH33)/(2.*sw) + (ee*sw*TH22*TH33)/(2.*cw)',
                 order = {'QED':1})

GC_55 = Coupling(name = 'GC_55',
                 value = '(ee**2*complex(0,1)*TH11*TH13)/(2.*sw**2) + (ee**2*complex(0,1)*TH21*TH23)/(2.*sw**2) + (ee**2*complex(0,1)*TH31*TH33)/(2.*sw**2)',
                 order = {'QED':2})

GC_56 = Coupling(name = 'GC_56',
                 value = 'ee**2*complex(0,1)*TH11*TH13 + (cw**2*ee**2*complex(0,1)*TH11*TH13)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH11*TH13)/(2.*cw**2) + ee**2*complex(0,1)*TH21*TH23 + (cw**2*ee**2*complex(0,1)*TH21*TH23)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH21*TH23)/(2.*cw**2) + ee**2*complex(0,1)*TH31*TH33 + (cw**2*ee**2*complex(0,1)*TH31*TH33)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH31*TH33)/(2.*cw**2)',
                 order = {'QED':2})

GC_57 = Coupling(name = 'GC_57',
                 value = '(ee**2*complex(0,1)*TH12*TH13)/(2.*sw**2) + (ee**2*complex(0,1)*TH22*TH23)/(2.*sw**2) + (ee**2*complex(0,1)*TH32*TH33)/(2.*sw**2)',
                 order = {'QED':2})

GC_58 = Coupling(name = 'GC_58',
                 value = 'ee**2*complex(0,1)*TH12*TH13 + (cw**2*ee**2*complex(0,1)*TH12*TH13)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH12*TH13)/(2.*cw**2) + ee**2*complex(0,1)*TH22*TH23 + (cw**2*ee**2*complex(0,1)*TH22*TH23)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH22*TH23)/(2.*cw**2) + ee**2*complex(0,1)*TH32*TH33 + (cw**2*ee**2*complex(0,1)*TH32*TH33)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH32*TH33)/(2.*cw**2)',
                 order = {'QED':2})

GC_59 = Coupling(name = 'GC_59',
                 value = '(ee**2*complex(0,1)*TH13**2)/(2.*sw**2) + (ee**2*complex(0,1)*TH23**2)/(2.*sw**2) + (ee**2*complex(0,1)*TH33**2)/(2.*sw**2)',
                 order = {'QED':2})

GC_60 = Coupling(name = 'GC_60',
                 value = 'ee**2*complex(0,1)*TH13**2 + (cw**2*ee**2*complex(0,1)*TH13**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH13**2)/(2.*cw**2) + ee**2*complex(0,1)*TH23**2 + (cw**2*ee**2*complex(0,1)*TH23**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH23**2)/(2.*cw**2) + ee**2*complex(0,1)*TH33**2 + (cw**2*ee**2*complex(0,1)*TH33**2)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH33**2)/(2.*cw**2)',
                 order = {'QED':2})

GC_61 = Coupling(name = 'GC_61',
                 value = '-((DD11*complex(0,1)*TH11)/v) - (complex(0,1)*GD11*TH21)/v + (GD11*TH31)/v',
                 order = {'QED':1})

GC_62 = Coupling(name = 'GC_62',
                 value = '-((complex(0,1)*GD12*TH21)/v) + (GD12*TH31)/v',
                 order = {'QED':1})

GC_63 = Coupling(name = 'GC_63',
                 value = '-((complex(0,1)*GD13*TH21)/v) + (GD13*TH31)/v',
                 order = {'QED':1})

GC_64 = Coupling(name = 'GC_64',
                 value = '-((complex(0,1)*GD21*TH21)/v) + (GD21*TH31)/v',
                 order = {'QED':1})

GC_65 = Coupling(name = 'GC_65',
                 value = '-((DD22*complex(0,1)*TH11)/v) - (complex(0,1)*GD22*TH21)/v + (GD22*TH31)/v',
                 order = {'QED':1})

GC_66 = Coupling(name = 'GC_66',
                 value = '-((complex(0,1)*GD23*TH21)/v) + (GD23*TH31)/v',
                 order = {'QED':1})

GC_67 = Coupling(name = 'GC_67',
                 value = '-((complex(0,1)*GD31*TH21)/v) + (GD31*TH31)/v',
                 order = {'QED':1})

GC_68 = Coupling(name = 'GC_68',
                 value = '-((complex(0,1)*GD32*TH21)/v) + (GD32*TH31)/v',
                 order = {'QED':1})

GC_69 = Coupling(name = 'GC_69',
                 value = '-((DD33*complex(0,1)*TH11)/v) - (complex(0,1)*GD33*TH21)/v + (GD33*TH31)/v',
                 order = {'QED':1})

GC_70 = Coupling(name = 'GC_70',
                 value = '-((complex(0,1)*GL11*TH21)/v) + (GL11*TH31)/v',
                 order = {'QED':1})

GC_71 = Coupling(name = 'GC_71',
                 value = '-((complex(0,1)*GL12*TH21)/v) + (GL12*TH31)/v',
                 order = {'QED':1})

GC_72 = Coupling(name = 'GC_72',
                 value = '-((complex(0,1)*GL13*TH21)/v) + (GL13*TH31)/v',
                 order = {'QED':1})

GC_73 = Coupling(name = 'GC_73',
                 value = '-((complex(0,1)*GL21*TH21)/v) + (GL21*TH31)/v',
                 order = {'QED':1})

GC_74 = Coupling(name = 'GC_74',
                 value = '-((complex(0,1)*GL22*TH21)/v) + (GL22*TH31)/v',
                 order = {'QED':1})

GC_75 = Coupling(name = 'GC_75',
                 value = '-((complex(0,1)*GL23*TH21)/v) + (GL23*TH31)/v',
                 order = {'QED':1})

GC_76 = Coupling(name = 'GC_76',
                 value = '-((complex(0,1)*GL31*TH21)/v) + (GL31*TH31)/v',
                 order = {'QED':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '-((complex(0,1)*GL32*TH21)/v) + (GL32*TH31)/v',
                 order = {'QED':1})

GC_78 = Coupling(name = 'GC_78',
                 value = '-((complex(0,1)*GL33*TH21)/v) + (GL33*TH31)/v',
                 order = {'QED':1})

GC_79 = Coupling(name = 'GC_79',
                 value = '-((complex(0,1)*GU11*TH21)/v) - (GU11*TH31)/v',
                 order = {'QED':1})

GC_80 = Coupling(name = 'GC_80',
                 value = '-((complex(0,1)*GU12*TH21)/v) - (GU12*TH31)/v',
                 order = {'QED':1})

GC_81 = Coupling(name = 'GC_81',
                 value = '-((complex(0,1)*GU13*TH21)/v) - (GU13*TH31)/v',
                 order = {'QED':1})

GC_82 = Coupling(name = 'GC_82',
                 value = '-((complex(0,1)*GU21*TH21)/v) - (GU21*TH31)/v',
                 order = {'QED':1})

GC_83 = Coupling(name = 'GC_83',
                 value = '-((complex(0,1)*GU22*TH21)/v) - (GU22*TH31)/v',
                 order = {'QED':1})

GC_84 = Coupling(name = 'GC_84',
                 value = '-((complex(0,1)*GU23*TH21)/v) - (GU23*TH31)/v',
                 order = {'QED':1})

GC_85 = Coupling(name = 'GC_85',
                 value = '-((complex(0,1)*GU31*TH21)/v) - (GU31*TH31)/v',
                 order = {'QED':1})

GC_86 = Coupling(name = 'GC_86',
                 value = '-((complex(0,1)*GU32*TH21)/v) - (GU32*TH31)/v',
                 order = {'QED':1})

GC_87 = Coupling(name = 'GC_87',
                 value = '-((complex(0,1)*GU33*TH21)/v) - (GU33*TH31)/v',
                 order = {'QED':1})

GC_88 = Coupling(name = 'GC_88',
                 value = '-((DD11*complex(0,1)*TH12)/v) - (complex(0,1)*GD11*TH22)/v + (GD11*TH32)/v',
                 order = {'QED':1})

GC_89 = Coupling(name = 'GC_89',
                 value = '-((complex(0,1)*GD12*TH22)/v) + (GD12*TH32)/v',
                 order = {'QED':1})

GC_90 = Coupling(name = 'GC_90',
                 value = '-((complex(0,1)*GD13*TH22)/v) + (GD13*TH32)/v',
                 order = {'QED':1})

GC_91 = Coupling(name = 'GC_91',
                 value = '-((complex(0,1)*GD21*TH22)/v) + (GD21*TH32)/v',
                 order = {'QED':1})

GC_92 = Coupling(name = 'GC_92',
                 value = '-((DD22*complex(0,1)*TH12)/v) - (complex(0,1)*GD22*TH22)/v + (GD22*TH32)/v',
                 order = {'QED':1})

GC_93 = Coupling(name = 'GC_93',
                 value = '-((complex(0,1)*GD23*TH22)/v) + (GD23*TH32)/v',
                 order = {'QED':1})

GC_94 = Coupling(name = 'GC_94',
                 value = '-((complex(0,1)*GD31*TH22)/v) + (GD31*TH32)/v',
                 order = {'QED':1})

GC_95 = Coupling(name = 'GC_95',
                 value = '-((complex(0,1)*GD32*TH22)/v) + (GD32*TH32)/v',
                 order = {'QED':1})

GC_96 = Coupling(name = 'GC_96',
                 value = '-((DD33*complex(0,1)*TH12)/v) - (complex(0,1)*GD33*TH22)/v + (GD33*TH32)/v',
                 order = {'QED':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '-((complex(0,1)*GL11*TH22)/v) + (GL11*TH32)/v',
                 order = {'QED':1})

GC_98 = Coupling(name = 'GC_98',
                 value = '-((complex(0,1)*GL12*TH22)/v) + (GL12*TH32)/v',
                 order = {'QED':1})

GC_99 = Coupling(name = 'GC_99',
                 value = '-((complex(0,1)*GL13*TH22)/v) + (GL13*TH32)/v',
                 order = {'QED':1})

GC_100 = Coupling(name = 'GC_100',
                  value = '-((complex(0,1)*GL21*TH22)/v) + (GL21*TH32)/v',
                  order = {'QED':1})

GC_101 = Coupling(name = 'GC_101',
                  value = '-((complex(0,1)*GL22*TH22)/v) + (GL22*TH32)/v',
                  order = {'QED':1})

GC_102 = Coupling(name = 'GC_102',
                  value = '-((complex(0,1)*GL23*TH22)/v) + (GL23*TH32)/v',
                  order = {'QED':1})

GC_103 = Coupling(name = 'GC_103',
                  value = '-((complex(0,1)*GL31*TH22)/v) + (GL31*TH32)/v',
                  order = {'QED':1})

GC_104 = Coupling(name = 'GC_104',
                  value = '-((complex(0,1)*GL32*TH22)/v) + (GL32*TH32)/v',
                  order = {'QED':1})

GC_105 = Coupling(name = 'GC_105',
                  value = '-((complex(0,1)*GL33*TH22)/v) + (GL33*TH32)/v',
                  order = {'QED':1})

GC_106 = Coupling(name = 'GC_106',
                  value = '-((complex(0,1)*GU11*TH22)/v) - (GU11*TH32)/v',
                  order = {'QED':1})

GC_107 = Coupling(name = 'GC_107',
                  value = '-((complex(0,1)*GU12*TH22)/v) - (GU12*TH32)/v',
                  order = {'QED':1})

GC_108 = Coupling(name = 'GC_108',
                  value = '-((complex(0,1)*GU13*TH22)/v) - (GU13*TH32)/v',
                  order = {'QED':1})

GC_109 = Coupling(name = 'GC_109',
                  value = '-((complex(0,1)*GU21*TH22)/v) - (GU21*TH32)/v',
                  order = {'QED':1})

GC_110 = Coupling(name = 'GC_110',
                  value = '-((complex(0,1)*GU22*TH22)/v) - (GU22*TH32)/v',
                  order = {'QED':1})

GC_111 = Coupling(name = 'GC_111',
                  value = '-((complex(0,1)*GU23*TH22)/v) - (GU23*TH32)/v',
                  order = {'QED':1})

GC_112 = Coupling(name = 'GC_112',
                  value = '-((complex(0,1)*GU31*TH22)/v) - (GU31*TH32)/v',
                  order = {'QED':1})

GC_113 = Coupling(name = 'GC_113',
                  value = '-((complex(0,1)*GU32*TH22)/v) - (GU32*TH32)/v',
                  order = {'QED':1})

GC_114 = Coupling(name = 'GC_114',
                  value = '-((complex(0,1)*GU33*TH22)/v) - (GU33*TH32)/v',
                  order = {'QED':1})

GC_115 = Coupling(name = 'GC_115',
                  value = '-((DD11*complex(0,1)*TH13)/v) - (complex(0,1)*GD11*TH23)/v + (GD11*TH33)/v',
                  order = {'QED':1})

GC_116 = Coupling(name = 'GC_116',
                  value = '-((complex(0,1)*GD12*TH23)/v) + (GD12*TH33)/v',
                  order = {'QED':1})

GC_117 = Coupling(name = 'GC_117',
                  value = '-((complex(0,1)*GD13*TH23)/v) + (GD13*TH33)/v',
                  order = {'QED':1})

GC_118 = Coupling(name = 'GC_118',
                  value = '-((complex(0,1)*GD21*TH23)/v) + (GD21*TH33)/v',
                  order = {'QED':1})

GC_119 = Coupling(name = 'GC_119',
                  value = '-((DD22*complex(0,1)*TH13)/v) - (complex(0,1)*GD22*TH23)/v + (GD22*TH33)/v',
                  order = {'QED':1})

GC_120 = Coupling(name = 'GC_120',
                  value = '-((complex(0,1)*GD23*TH23)/v) + (GD23*TH33)/v',
                  order = {'QED':1})

GC_121 = Coupling(name = 'GC_121',
                  value = '-((complex(0,1)*GD31*TH23)/v) + (GD31*TH33)/v',
                  order = {'QED':1})

GC_122 = Coupling(name = 'GC_122',
                  value = '-((complex(0,1)*GD32*TH23)/v) + (GD32*TH33)/v',
                  order = {'QED':1})

GC_123 = Coupling(name = 'GC_123',
                  value = '-((DD33*complex(0,1)*TH13)/v) - (complex(0,1)*GD33*TH23)/v + (GD33*TH33)/v',
                  order = {'QED':1})

GC_124 = Coupling(name = 'GC_124',
                  value = '-((complex(0,1)*GL11*TH23)/v) + (GL11*TH33)/v',
                  order = {'QED':1})

GC_125 = Coupling(name = 'GC_125',
                  value = '-((complex(0,1)*GL12*TH23)/v) + (GL12*TH33)/v',
                  order = {'QED':1})

GC_126 = Coupling(name = 'GC_126',
                  value = '-((complex(0,1)*GL13*TH23)/v) + (GL13*TH33)/v',
                  order = {'QED':1})

GC_127 = Coupling(name = 'GC_127',
                  value = '-((complex(0,1)*GL21*TH23)/v) + (GL21*TH33)/v',
                  order = {'QED':1})

GC_128 = Coupling(name = 'GC_128',
                  value = '-((complex(0,1)*GL22*TH23)/v) + (GL22*TH33)/v',
                  order = {'QED':1})

GC_129 = Coupling(name = 'GC_129',
                  value = '-((complex(0,1)*GL23*TH23)/v) + (GL23*TH33)/v',
                  order = {'QED':1})

GC_130 = Coupling(name = 'GC_130',
                  value = '-((complex(0,1)*GL31*TH23)/v) + (GL31*TH33)/v',
                  order = {'QED':1})

GC_131 = Coupling(name = 'GC_131',
                  value = '-((complex(0,1)*GL32*TH23)/v) + (GL32*TH33)/v',
                  order = {'QED':1})

GC_132 = Coupling(name = 'GC_132',
                  value = '-((complex(0,1)*GL33*TH23)/v) + (GL33*TH33)/v',
                  order = {'QED':1})

GC_133 = Coupling(name = 'GC_133',
                  value = '-((complex(0,1)*GU11*TH23)/v) - (GU11*TH33)/v',
                  order = {'QED':1})

GC_134 = Coupling(name = 'GC_134',
                  value = '-((complex(0,1)*GU12*TH23)/v) - (GU12*TH33)/v',
                  order = {'QED':1})

GC_135 = Coupling(name = 'GC_135',
                  value = '-((complex(0,1)*GU13*TH23)/v) - (GU13*TH33)/v',
                  order = {'QED':1})

GC_136 = Coupling(name = 'GC_136',
                  value = '-((complex(0,1)*GU21*TH23)/v) - (GU21*TH33)/v',
                  order = {'QED':1})

GC_137 = Coupling(name = 'GC_137',
                  value = '-((complex(0,1)*GU22*TH23)/v) - (GU22*TH33)/v',
                  order = {'QED':1})

GC_138 = Coupling(name = 'GC_138',
                  value = '-((complex(0,1)*GU23*TH23)/v) - (GU23*TH33)/v',
                  order = {'QED':1})

GC_139 = Coupling(name = 'GC_139',
                  value = '-((complex(0,1)*GU31*TH23)/v) - (GU31*TH33)/v',
                  order = {'QED':1})

GC_140 = Coupling(name = 'GC_140',
                  value = '-((complex(0,1)*GU32*TH23)/v) - (GU32*TH33)/v',
                  order = {'QED':1})

GC_141 = Coupling(name = 'GC_141',
                  value = '-((complex(0,1)*GU33*TH23)/v) - (GU33*TH33)/v',
                  order = {'QED':1})

GC_142 = Coupling(name = 'GC_142',
                  value = '-((complex(0,1)*GD11*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_143 = Coupling(name = 'GC_143',
                  value = '-((complex(0,1)*GD12*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_144 = Coupling(name = 'GC_144',
                  value = '-((complex(0,1)*GD13*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_145 = Coupling(name = 'GC_145',
                  value = '-((complex(0,1)*GD21*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_146 = Coupling(name = 'GC_146',
                  value = '-((complex(0,1)*GD22*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_147 = Coupling(name = 'GC_147',
                  value = '-((complex(0,1)*GD23*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_148 = Coupling(name = 'GC_148',
                  value = '-((complex(0,1)*GD31*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_149 = Coupling(name = 'GC_149',
                  value = '-((complex(0,1)*GD32*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_150 = Coupling(name = 'GC_150',
                  value = '-((complex(0,1)*GD33*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_151 = Coupling(name = 'GC_151',
                  value = '-((complex(0,1)*GL11*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_152 = Coupling(name = 'GC_152',
                  value = '-((complex(0,1)*GL12*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_153 = Coupling(name = 'GC_153',
                  value = '-((complex(0,1)*GL13*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_154 = Coupling(name = 'GC_154',
                  value = '-((complex(0,1)*GL21*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_155 = Coupling(name = 'GC_155',
                  value = '-((complex(0,1)*GL22*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_156 = Coupling(name = 'GC_156',
                  value = '-((complex(0,1)*GL23*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_157 = Coupling(name = 'GC_157',
                  value = '-((complex(0,1)*GL31*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_158 = Coupling(name = 'GC_158',
                  value = '-((complex(0,1)*GL32*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_159 = Coupling(name = 'GC_159',
                  value = '-((complex(0,1)*GL33*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_160 = Coupling(name = 'GC_160',
                  value = '(complex(0,1)*GU11*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_161 = Coupling(name = 'GC_161',
                  value = '(complex(0,1)*GU12*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_162 = Coupling(name = 'GC_162',
                  value = '(complex(0,1)*GU13*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_163 = Coupling(name = 'GC_163',
                  value = '(complex(0,1)*GU21*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_164 = Coupling(name = 'GC_164',
                  value = '(complex(0,1)*GU22*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_165 = Coupling(name = 'GC_165',
                  value = '(complex(0,1)*GU23*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_166 = Coupling(name = 'GC_166',
                  value = '(complex(0,1)*GU31*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_167 = Coupling(name = 'GC_167',
                  value = '(complex(0,1)*GU32*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_168 = Coupling(name = 'GC_168',
                  value = '(complex(0,1)*GU33*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_169 = Coupling(name = 'GC_169',
                  value = '(ee**2*complex(0,1)*TH11*v)/(2.*sw**2)',
                  order = {'QED':1})

GC_170 = Coupling(name = 'GC_170',
                  value = '(ee**2*complex(0,1)*TH12*v)/(2.*sw**2)',
                  order = {'QED':1})

GC_171 = Coupling(name = 'GC_171',
                  value = '(ee**2*complex(0,1)*TH13*v)/(2.*sw**2)',
                  order = {'QED':1})

GC_172 = Coupling(name = 'GC_172',
                  value = 'ee**2*complex(0,1)*TH11*v + (cw**2*ee**2*complex(0,1)*TH11*v)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH11*v)/(2.*cw**2)',
                  order = {'QED':1})

GC_173 = Coupling(name = 'GC_173',
                  value = 'ee**2*complex(0,1)*TH12*v + (cw**2*ee**2*complex(0,1)*TH12*v)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH12*v)/(2.*cw**2)',
                  order = {'QED':1})

GC_174 = Coupling(name = 'GC_174',
                  value = 'ee**2*complex(0,1)*TH13*v + (cw**2*ee**2*complex(0,1)*TH13*v)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*TH13*v)/(2.*cw**2)',
                  order = {'QED':1})

GC_175 = Coupling(name = 'GC_175',
                  value = '-((complex(0,1)*TH11*yukl1)/v)',
                  order = {'QED':2})

GC_176 = Coupling(name = 'GC_176',
                  value = '-((complex(0,1)*TH12*yukl1)/v)',
                  order = {'QED':2})

GC_177 = Coupling(name = 'GC_177',
                  value = '-((complex(0,1)*TH13*yukl1)/v)',
                  order = {'QED':2})

GC_178 = Coupling(name = 'GC_178',
                  value = '-((complex(0,1)*TH11*yukl2)/v)',
                  order = {'QED':2})

GC_179 = Coupling(name = 'GC_179',
                  value = '-((complex(0,1)*TH12*yukl2)/v)',
                  order = {'QED':2})

GC_180 = Coupling(name = 'GC_180',
                  value = '-((complex(0,1)*TH13*yukl2)/v)',
                  order = {'QED':2})

GC_181 = Coupling(name = 'GC_181',
                  value = '-((complex(0,1)*TH11*yukl3)/v)',
                  order = {'QED':2})

GC_182 = Coupling(name = 'GC_182',
                  value = '-((complex(0,1)*TH12*yukl3)/v)',
                  order = {'QED':2})

GC_183 = Coupling(name = 'GC_183',
                  value = '-((complex(0,1)*TH13*yukl3)/v)',
                  order = {'QED':2})

GC_184 = Coupling(name = 'GC_184',
                  value = '-((complex(0,1)*TH11*yuku1)/v)',
                  order = {'QED':2})

GC_185 = Coupling(name = 'GC_185',
                  value = '-((complex(0,1)*TH12*yuku1)/v)',
                  order = {'QED':2})

GC_186 = Coupling(name = 'GC_186',
                  value = '-((complex(0,1)*TH13*yuku1)/v)',
                  order = {'QED':2})

GC_187 = Coupling(name = 'GC_187',
                  value = '-((complex(0,1)*TH11*yuku2)/v)',
                  order = {'QED':2})

GC_188 = Coupling(name = 'GC_188',
                  value = '-((complex(0,1)*TH12*yuku2)/v)',
                  order = {'QED':2})

GC_189 = Coupling(name = 'GC_189',
                  value = '-((complex(0,1)*TH13*yuku2)/v)',
                  order = {'QED':2})

GC_190 = Coupling(name = 'GC_190',
                  value = '-((complex(0,1)*TH11*yuku3)/v)',
                  order = {'QED':2})

GC_191 = Coupling(name = 'GC_191',
                  value = '-((complex(0,1)*TH12*yuku3)/v)',
                  order = {'QED':2})

GC_192 = Coupling(name = 'GC_192',
                  value = '-((complex(0,1)*TH13*yuku3)/v)',
                  order = {'QED':2})

GC_193 = Coupling(name = 'GC_193',
                  value = '(ee*complex(0,1)*complexconjugate(CKM11))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_194 = Coupling(name = 'GC_194',
                  value = '(ee*complex(0,1)*complexconjugate(CKM12))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_195 = Coupling(name = 'GC_195',
                  value = '(ee*complex(0,1)*complexconjugate(CKM21))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_196 = Coupling(name = 'GC_196',
                  value = '(ee*complex(0,1)*complexconjugate(CKM22))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_197 = Coupling(name = 'GC_197',
                  value = '-((complex(0,1)*complexconjugate(GD11)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_198 = Coupling(name = 'GC_198',
                  value = '-((complex(0,1)*TH11*complexconjugate(DD11))/v) - (complex(0,1)*TH21*complexconjugate(GD11))/v - (TH31*complexconjugate(GD11))/v',
                  order = {'QED':1})

GC_199 = Coupling(name = 'GC_199',
                  value = '-((complex(0,1)*TH12*complexconjugate(DD11))/v) - (complex(0,1)*TH22*complexconjugate(GD11))/v - (TH32*complexconjugate(GD11))/v',
                  order = {'QED':1})

GC_200 = Coupling(name = 'GC_200',
                  value = '-((complex(0,1)*TH13*complexconjugate(DD11))/v) - (complex(0,1)*TH23*complexconjugate(GD11))/v - (TH33*complexconjugate(GD11))/v',
                  order = {'QED':1})

GC_201 = Coupling(name = 'GC_201',
                  value = '-((complex(0,1)*complexconjugate(GD12)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_202 = Coupling(name = 'GC_202',
                  value = '-((complex(0,1)*TH21*complexconjugate(GD12))/v) - (TH31*complexconjugate(GD12))/v',
                  order = {'QED':1})

GC_203 = Coupling(name = 'GC_203',
                  value = '-((complex(0,1)*TH22*complexconjugate(GD12))/v) - (TH32*complexconjugate(GD12))/v',
                  order = {'QED':1})

GC_204 = Coupling(name = 'GC_204',
                  value = '-((complex(0,1)*TH23*complexconjugate(GD12))/v) - (TH33*complexconjugate(GD12))/v',
                  order = {'QED':1})

GC_205 = Coupling(name = 'GC_205',
                  value = '-((complex(0,1)*complexconjugate(GD13)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_206 = Coupling(name = 'GC_206',
                  value = '-((complex(0,1)*TH21*complexconjugate(GD13))/v) - (TH31*complexconjugate(GD13))/v',
                  order = {'QED':1})

GC_207 = Coupling(name = 'GC_207',
                  value = '-((complex(0,1)*TH22*complexconjugate(GD13))/v) - (TH32*complexconjugate(GD13))/v',
                  order = {'QED':1})

GC_208 = Coupling(name = 'GC_208',
                  value = '-((complex(0,1)*TH23*complexconjugate(GD13))/v) - (TH33*complexconjugate(GD13))/v',
                  order = {'QED':1})

GC_209 = Coupling(name = 'GC_209',
                  value = '-((complex(0,1)*complexconjugate(GD21)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_210 = Coupling(name = 'GC_210',
                  value = '-((complex(0,1)*TH21*complexconjugate(GD21))/v) - (TH31*complexconjugate(GD21))/v',
                  order = {'QED':1})

GC_211 = Coupling(name = 'GC_211',
                  value = '-((complex(0,1)*TH22*complexconjugate(GD21))/v) - (TH32*complexconjugate(GD21))/v',
                  order = {'QED':1})

GC_212 = Coupling(name = 'GC_212',
                  value = '-((complex(0,1)*TH23*complexconjugate(GD21))/v) - (TH33*complexconjugate(GD21))/v',
                  order = {'QED':1})

GC_213 = Coupling(name = 'GC_213',
                  value = '-((complex(0,1)*complexconjugate(GD22)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_214 = Coupling(name = 'GC_214',
                  value = '-((complex(0,1)*TH11*complexconjugate(DD22))/v) - (complex(0,1)*TH21*complexconjugate(GD22))/v - (TH31*complexconjugate(GD22))/v',
                  order = {'QED':1})

GC_215 = Coupling(name = 'GC_215',
                  value = '-((complex(0,1)*TH12*complexconjugate(DD22))/v) - (complex(0,1)*TH22*complexconjugate(GD22))/v - (TH32*complexconjugate(GD22))/v',
                  order = {'QED':1})

GC_216 = Coupling(name = 'GC_216',
                  value = '-((complex(0,1)*TH13*complexconjugate(DD22))/v) - (complex(0,1)*TH23*complexconjugate(GD22))/v - (TH33*complexconjugate(GD22))/v',
                  order = {'QED':1})

GC_217 = Coupling(name = 'GC_217',
                  value = '-((complex(0,1)*complexconjugate(GD23)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_218 = Coupling(name = 'GC_218',
                  value = '-((complex(0,1)*TH21*complexconjugate(GD23))/v) - (TH31*complexconjugate(GD23))/v',
                  order = {'QED':1})

GC_219 = Coupling(name = 'GC_219',
                  value = '-((complex(0,1)*TH22*complexconjugate(GD23))/v) - (TH32*complexconjugate(GD23))/v',
                  order = {'QED':1})

GC_220 = Coupling(name = 'GC_220',
                  value = '-((complex(0,1)*TH23*complexconjugate(GD23))/v) - (TH33*complexconjugate(GD23))/v',
                  order = {'QED':1})

GC_221 = Coupling(name = 'GC_221',
                  value = '-((complex(0,1)*complexconjugate(GD31)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_222 = Coupling(name = 'GC_222',
                  value = '-((complex(0,1)*TH21*complexconjugate(GD31))/v) - (TH31*complexconjugate(GD31))/v',
                  order = {'QED':1})

GC_223 = Coupling(name = 'GC_223',
                  value = '-((complex(0,1)*TH22*complexconjugate(GD31))/v) - (TH32*complexconjugate(GD31))/v',
                  order = {'QED':1})

GC_224 = Coupling(name = 'GC_224',
                  value = '-((complex(0,1)*TH23*complexconjugate(GD31))/v) - (TH33*complexconjugate(GD31))/v',
                  order = {'QED':1})

GC_225 = Coupling(name = 'GC_225',
                  value = '-((complex(0,1)*complexconjugate(GD32)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_226 = Coupling(name = 'GC_226',
                  value = '-((complex(0,1)*TH21*complexconjugate(GD32))/v) - (TH31*complexconjugate(GD32))/v',
                  order = {'QED':1})

GC_227 = Coupling(name = 'GC_227',
                  value = '-((complex(0,1)*TH22*complexconjugate(GD32))/v) - (TH32*complexconjugate(GD32))/v',
                  order = {'QED':1})

GC_228 = Coupling(name = 'GC_228',
                  value = '-((complex(0,1)*TH23*complexconjugate(GD32))/v) - (TH33*complexconjugate(GD32))/v',
                  order = {'QED':1})

GC_229 = Coupling(name = 'GC_229',
                  value = '-((complex(0,1)*complexconjugate(GD33)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_230 = Coupling(name = 'GC_230',
                  value = '-((complex(0,1)*TH11*complexconjugate(DD33))/v) - (complex(0,1)*TH21*complexconjugate(GD33))/v - (TH31*complexconjugate(GD33))/v',
                  order = {'QED':1})

GC_231 = Coupling(name = 'GC_231',
                  value = '-((complex(0,1)*TH12*complexconjugate(DD33))/v) - (complex(0,1)*TH22*complexconjugate(GD33))/v - (TH32*complexconjugate(GD33))/v',
                  order = {'QED':1})

GC_232 = Coupling(name = 'GC_232',
                  value = '-((complex(0,1)*TH13*complexconjugate(DD33))/v) - (complex(0,1)*TH23*complexconjugate(GD33))/v - (TH33*complexconjugate(GD33))/v',
                  order = {'QED':1})

GC_233 = Coupling(name = 'GC_233',
                  value = '-((complex(0,1)*complexconjugate(GL11)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_234 = Coupling(name = 'GC_234',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL11))/v) - (TH31*complexconjugate(GL11))/v',
                  order = {'QED':1})

GC_235 = Coupling(name = 'GC_235',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL11))/v) - (TH32*complexconjugate(GL11))/v',
                  order = {'QED':1})

GC_236 = Coupling(name = 'GC_236',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL11))/v) - (TH33*complexconjugate(GL11))/v',
                  order = {'QED':1})

GC_237 = Coupling(name = 'GC_237',
                  value = '-((complex(0,1)*complexconjugate(GL12)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_238 = Coupling(name = 'GC_238',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL12))/v) - (TH31*complexconjugate(GL12))/v',
                  order = {'QED':1})

GC_239 = Coupling(name = 'GC_239',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL12))/v) - (TH32*complexconjugate(GL12))/v',
                  order = {'QED':1})

GC_240 = Coupling(name = 'GC_240',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL12))/v) - (TH33*complexconjugate(GL12))/v',
                  order = {'QED':1})

GC_241 = Coupling(name = 'GC_241',
                  value = '-((complex(0,1)*complexconjugate(GL13)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_242 = Coupling(name = 'GC_242',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL13))/v) - (TH31*complexconjugate(GL13))/v',
                  order = {'QED':1})

GC_243 = Coupling(name = 'GC_243',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL13))/v) - (TH32*complexconjugate(GL13))/v',
                  order = {'QED':1})

GC_244 = Coupling(name = 'GC_244',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL13))/v) - (TH33*complexconjugate(GL13))/v',
                  order = {'QED':1})

GC_245 = Coupling(name = 'GC_245',
                  value = '-((complex(0,1)*complexconjugate(GL21)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_246 = Coupling(name = 'GC_246',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL21))/v) - (TH31*complexconjugate(GL21))/v',
                  order = {'QED':1})

GC_247 = Coupling(name = 'GC_247',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL21))/v) - (TH32*complexconjugate(GL21))/v',
                  order = {'QED':1})

GC_248 = Coupling(name = 'GC_248',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL21))/v) - (TH33*complexconjugate(GL21))/v',
                  order = {'QED':1})

GC_249 = Coupling(name = 'GC_249',
                  value = '-((complex(0,1)*complexconjugate(GL22)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_250 = Coupling(name = 'GC_250',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL22))/v) - (TH31*complexconjugate(GL22))/v',
                  order = {'QED':1})

GC_251 = Coupling(name = 'GC_251',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL22))/v) - (TH32*complexconjugate(GL22))/v',
                  order = {'QED':1})

GC_252 = Coupling(name = 'GC_252',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL22))/v) - (TH33*complexconjugate(GL22))/v',
                  order = {'QED':1})

GC_253 = Coupling(name = 'GC_253',
                  value = '-((complex(0,1)*complexconjugate(GL23)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_254 = Coupling(name = 'GC_254',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL23))/v) - (TH31*complexconjugate(GL23))/v',
                  order = {'QED':1})

GC_255 = Coupling(name = 'GC_255',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL23))/v) - (TH32*complexconjugate(GL23))/v',
                  order = {'QED':1})

GC_256 = Coupling(name = 'GC_256',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL23))/v) - (TH33*complexconjugate(GL23))/v',
                  order = {'QED':1})

GC_257 = Coupling(name = 'GC_257',
                  value = '-((complex(0,1)*complexconjugate(GL31)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_258 = Coupling(name = 'GC_258',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL31))/v) - (TH31*complexconjugate(GL31))/v',
                  order = {'QED':1})

GC_259 = Coupling(name = 'GC_259',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL31))/v) - (TH32*complexconjugate(GL31))/v',
                  order = {'QED':1})

GC_260 = Coupling(name = 'GC_260',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL31))/v) - (TH33*complexconjugate(GL31))/v',
                  order = {'QED':1})

GC_261 = Coupling(name = 'GC_261',
                  value = '-((complex(0,1)*complexconjugate(GL32)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_262 = Coupling(name = 'GC_262',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL32))/v) - (TH31*complexconjugate(GL32))/v',
                  order = {'QED':1})

GC_263 = Coupling(name = 'GC_263',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL32))/v) - (TH32*complexconjugate(GL32))/v',
                  order = {'QED':1})

GC_264 = Coupling(name = 'GC_264',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL32))/v) - (TH33*complexconjugate(GL32))/v',
                  order = {'QED':1})

GC_265 = Coupling(name = 'GC_265',
                  value = '-((complex(0,1)*complexconjugate(GL33)*cmath.sqrt(2))/v)',
                  order = {'QED':1})

GC_266 = Coupling(name = 'GC_266',
                  value = '-((complex(0,1)*TH21*complexconjugate(GL33))/v) - (TH31*complexconjugate(GL33))/v',
                  order = {'QED':1})

GC_267 = Coupling(name = 'GC_267',
                  value = '-((complex(0,1)*TH22*complexconjugate(GL33))/v) - (TH32*complexconjugate(GL33))/v',
                  order = {'QED':1})

GC_268 = Coupling(name = 'GC_268',
                  value = '-((complex(0,1)*TH23*complexconjugate(GL33))/v) - (TH33*complexconjugate(GL33))/v',
                  order = {'QED':1})

GC_269 = Coupling(name = 'GC_269',
                  value = '(complex(0,1)*complexconjugate(GU11)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_270 = Coupling(name = 'GC_270',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU11))/v) + (TH31*complexconjugate(GU11))/v',
                  order = {'QED':1})

GC_271 = Coupling(name = 'GC_271',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU11))/v) + (TH32*complexconjugate(GU11))/v',
                  order = {'QED':1})

GC_272 = Coupling(name = 'GC_272',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU11))/v) + (TH33*complexconjugate(GU11))/v',
                  order = {'QED':1})

GC_273 = Coupling(name = 'GC_273',
                  value = '(complex(0,1)*complexconjugate(GU12)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_274 = Coupling(name = 'GC_274',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU12))/v) + (TH31*complexconjugate(GU12))/v',
                  order = {'QED':1})

GC_275 = Coupling(name = 'GC_275',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU12))/v) + (TH32*complexconjugate(GU12))/v',
                  order = {'QED':1})

GC_276 = Coupling(name = 'GC_276',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU12))/v) + (TH33*complexconjugate(GU12))/v',
                  order = {'QED':1})

GC_277 = Coupling(name = 'GC_277',
                  value = '(complex(0,1)*complexconjugate(GU13)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_278 = Coupling(name = 'GC_278',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU13))/v) + (TH31*complexconjugate(GU13))/v',
                  order = {'QED':1})

GC_279 = Coupling(name = 'GC_279',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU13))/v) + (TH32*complexconjugate(GU13))/v',
                  order = {'QED':1})

GC_280 = Coupling(name = 'GC_280',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU13))/v) + (TH33*complexconjugate(GU13))/v',
                  order = {'QED':1})

GC_281 = Coupling(name = 'GC_281',
                  value = '(complex(0,1)*complexconjugate(GU21)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_282 = Coupling(name = 'GC_282',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU21))/v) + (TH31*complexconjugate(GU21))/v',
                  order = {'QED':1})

GC_283 = Coupling(name = 'GC_283',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU21))/v) + (TH32*complexconjugate(GU21))/v',
                  order = {'QED':1})

GC_284 = Coupling(name = 'GC_284',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU21))/v) + (TH33*complexconjugate(GU21))/v',
                  order = {'QED':1})

GC_285 = Coupling(name = 'GC_285',
                  value = '(complex(0,1)*complexconjugate(GU22)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_286 = Coupling(name = 'GC_286',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU22))/v) + (TH31*complexconjugate(GU22))/v',
                  order = {'QED':1})

GC_287 = Coupling(name = 'GC_287',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU22))/v) + (TH32*complexconjugate(GU22))/v',
                  order = {'QED':1})

GC_288 = Coupling(name = 'GC_288',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU22))/v) + (TH33*complexconjugate(GU22))/v',
                  order = {'QED':1})

GC_289 = Coupling(name = 'GC_289',
                  value = '(complex(0,1)*complexconjugate(GU23)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_290 = Coupling(name = 'GC_290',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU23))/v) + (TH31*complexconjugate(GU23))/v',
                  order = {'QED':1})

GC_291 = Coupling(name = 'GC_291',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU23))/v) + (TH32*complexconjugate(GU23))/v',
                  order = {'QED':1})

GC_292 = Coupling(name = 'GC_292',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU23))/v) + (TH33*complexconjugate(GU23))/v',
                  order = {'QED':1})

GC_293 = Coupling(name = 'GC_293',
                  value = '(complex(0,1)*complexconjugate(GU31)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_294 = Coupling(name = 'GC_294',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU31))/v) + (TH31*complexconjugate(GU31))/v',
                  order = {'QED':1})

GC_295 = Coupling(name = 'GC_295',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU31))/v) + (TH32*complexconjugate(GU31))/v',
                  order = {'QED':1})

GC_296 = Coupling(name = 'GC_296',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU31))/v) + (TH33*complexconjugate(GU31))/v',
                  order = {'QED':1})

GC_297 = Coupling(name = 'GC_297',
                  value = '(complex(0,1)*complexconjugate(GU32)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_298 = Coupling(name = 'GC_298',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU32))/v) + (TH31*complexconjugate(GU32))/v',
                  order = {'QED':1})

GC_299 = Coupling(name = 'GC_299',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU32))/v) + (TH32*complexconjugate(GU32))/v',
                  order = {'QED':1})

GC_300 = Coupling(name = 'GC_300',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU32))/v) + (TH33*complexconjugate(GU32))/v',
                  order = {'QED':1})

GC_301 = Coupling(name = 'GC_301',
                  value = '(complex(0,1)*complexconjugate(GU33)*cmath.sqrt(2))/v',
                  order = {'QED':1})

GC_302 = Coupling(name = 'GC_302',
                  value = '-((complex(0,1)*TH21*complexconjugate(GU33))/v) + (TH31*complexconjugate(GU33))/v',
                  order = {'QED':1})

GC_303 = Coupling(name = 'GC_303',
                  value = '-((complex(0,1)*TH22*complexconjugate(GU33))/v) + (TH32*complexconjugate(GU33))/v',
                  order = {'QED':1})

GC_304 = Coupling(name = 'GC_304',
                  value = '-((complex(0,1)*TH23*complexconjugate(GU33))/v) + (TH33*complexconjugate(GU33))/v',
                  order = {'QED':1})

GC_305 = Coupling(name = 'GC_305',
                  value = '-(complex(0,1)*l3*TH11*v) - (complex(0,1)*l7*TH21*v)/2. + (l7*TH31*v)/2. - (complex(0,1)*TH21*v*complexconjugate(l7))/2. - (TH31*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_306 = Coupling(name = 'GC_306',
                  value = '-6*complex(0,1)*l1*TH11**3*v - (9*complex(0,1)*l6*TH11**2*TH21*v)/2. - 3*complex(0,1)*l3*TH11*TH21**2*v - 3*complex(0,1)*l4*TH11*TH21**2*v - 6*complex(0,1)*l5*TH11*TH21**2*v - (3*complex(0,1)*l7*TH21**3*v)/2. + (9*l6*TH11**2*TH31*v)/2. + (3*l7*TH21**2*TH31*v)/2. - 3*complex(0,1)*l3*TH11*TH31**2*v - 3*complex(0,1)*l4*TH11*TH31**2*v + 6*complex(0,1)*l5*TH11*TH31**2*v - (3*complex(0,1)*l7*TH21*TH31**2*v)/2. + (3*l7*TH31**3*v)/2. - (9*complex(0,1)*TH11**2*TH21*v*complexconjugate(l6))/2. - (9*TH11**2*TH31*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH21**3*v*complexconjugate(l7))/2. - (3*TH21**2*TH31*v*complexconjugate(l7))/2. - (3*complex(0,1)*TH21*TH31**2*v*complexconjugate(l7))/2. - (3*TH31**3*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_307 = Coupling(name = 'GC_307',
                  value = '-(complex(0,1)*l3*TH12*v) - (complex(0,1)*l7*TH22*v)/2. + (l7*TH32*v)/2. - (complex(0,1)*TH22*v*complexconjugate(l7))/2. - (TH32*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_308 = Coupling(name = 'GC_308',
                  value = '-6*complex(0,1)*l1*TH11**2*TH12*v - 3*complex(0,1)*l6*TH11*TH12*TH21*v - complex(0,1)*l3*TH12*TH21**2*v - complex(0,1)*l4*TH12*TH21**2*v - 2*complex(0,1)*l5*TH12*TH21**2*v - (3*complex(0,1)*l6*TH11**2*TH22*v)/2. - 2*complex(0,1)*l3*TH11*TH21*TH22*v - 2*complex(0,1)*l4*TH11*TH21*TH22*v - 4*complex(0,1)*l5*TH11*TH21*TH22*v - (3*complex(0,1)*l7*TH21**2*TH22*v)/2. + 3*l6*TH11*TH12*TH31*v + l7*TH21*TH22*TH31*v - complex(0,1)*l3*TH12*TH31**2*v - complex(0,1)*l4*TH12*TH31**2*v + 2*complex(0,1)*l5*TH12*TH31**2*v - (complex(0,1)*l7*TH22*TH31**2*v)/2. + (3*l6*TH11**2*TH32*v)/2. + (l7*TH21**2*TH32*v)/2. - 2*complex(0,1)*l3*TH11*TH31*TH32*v - 2*complex(0,1)*l4*TH11*TH31*TH32*v + 4*complex(0,1)*l5*TH11*TH31*TH32*v - complex(0,1)*l7*TH21*TH31*TH32*v + (3*l7*TH31**2*TH32*v)/2. - 3*complex(0,1)*TH11*TH12*TH21*v*complexconjugate(l6) - (3*complex(0,1)*TH11**2*TH22*v*complexconjugate(l6))/2. - 3*TH11*TH12*TH31*v*complexconjugate(l6) - (3*TH11**2*TH32*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH21**2*TH22*v*complexconjugate(l7))/2. - TH21*TH22*TH31*v*complexconjugate(l7) - (complex(0,1)*TH22*TH31**2*v*complexconjugate(l7))/2. - (TH21**2*TH32*v*complexconjugate(l7))/2. - complex(0,1)*TH21*TH31*TH32*v*complexconjugate(l7) - (3*TH31**2*TH32*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_309 = Coupling(name = 'GC_309',
                  value = '-6*complex(0,1)*l1*TH11*TH12**2*v - (3*complex(0,1)*l6*TH12**2*TH21*v)/2. - 3*complex(0,1)*l6*TH11*TH12*TH22*v - 2*complex(0,1)*l3*TH12*TH21*TH22*v - 2*complex(0,1)*l4*TH12*TH21*TH22*v - 4*complex(0,1)*l5*TH12*TH21*TH22*v - complex(0,1)*l3*TH11*TH22**2*v - complex(0,1)*l4*TH11*TH22**2*v - 2*complex(0,1)*l5*TH11*TH22**2*v - (3*complex(0,1)*l7*TH21*TH22**2*v)/2. + (3*l6*TH12**2*TH31*v)/2. + (l7*TH22**2*TH31*v)/2. + 3*l6*TH11*TH12*TH32*v + l7*TH21*TH22*TH32*v - 2*complex(0,1)*l3*TH12*TH31*TH32*v - 2*complex(0,1)*l4*TH12*TH31*TH32*v + 4*complex(0,1)*l5*TH12*TH31*TH32*v - complex(0,1)*l7*TH22*TH31*TH32*v - complex(0,1)*l3*TH11*TH32**2*v - complex(0,1)*l4*TH11*TH32**2*v + 2*complex(0,1)*l5*TH11*TH32**2*v - (complex(0,1)*l7*TH21*TH32**2*v)/2. + (3*l7*TH31*TH32**2*v)/2. - (3*complex(0,1)*TH12**2*TH21*v*complexconjugate(l6))/2. - 3*complex(0,1)*TH11*TH12*TH22*v*complexconjugate(l6) - (3*TH12**2*TH31*v*complexconjugate(l6))/2. - 3*TH11*TH12*TH32*v*complexconjugate(l6) - (3*complex(0,1)*TH21*TH22**2*v*complexconjugate(l7))/2. - (TH22**2*TH31*v*complexconjugate(l7))/2. - TH21*TH22*TH32*v*complexconjugate(l7) - complex(0,1)*TH22*TH31*TH32*v*complexconjugate(l7) - (complex(0,1)*TH21*TH32**2*v*complexconjugate(l7))/2. - (3*TH31*TH32**2*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_310 = Coupling(name = 'GC_310',
                  value = '-6*complex(0,1)*l1*TH12**3*v - (9*complex(0,1)*l6*TH12**2*TH22*v)/2. - 3*complex(0,1)*l3*TH12*TH22**2*v - 3*complex(0,1)*l4*TH12*TH22**2*v - 6*complex(0,1)*l5*TH12*TH22**2*v - (3*complex(0,1)*l7*TH22**3*v)/2. + (9*l6*TH12**2*TH32*v)/2. + (3*l7*TH22**2*TH32*v)/2. - 3*complex(0,1)*l3*TH12*TH32**2*v - 3*complex(0,1)*l4*TH12*TH32**2*v + 6*complex(0,1)*l5*TH12*TH32**2*v - (3*complex(0,1)*l7*TH22*TH32**2*v)/2. + (3*l7*TH32**3*v)/2. - (9*complex(0,1)*TH12**2*TH22*v*complexconjugate(l6))/2. - (9*TH12**2*TH32*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH22**3*v*complexconjugate(l7))/2. - (3*TH22**2*TH32*v*complexconjugate(l7))/2. - (3*complex(0,1)*TH22*TH32**2*v*complexconjugate(l7))/2. - (3*TH32**3*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_311 = Coupling(name = 'GC_311',
                  value = '-(complex(0,1)*l3*TH13*v) - (complex(0,1)*l7*TH23*v)/2. + (l7*TH33*v)/2. - (complex(0,1)*TH23*v*complexconjugate(l7))/2. - (TH33*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_312 = Coupling(name = 'GC_312',
                  value = '-6*complex(0,1)*l1*TH11**2*TH13*v - 3*complex(0,1)*l6*TH11*TH13*TH21*v - complex(0,1)*l3*TH13*TH21**2*v - complex(0,1)*l4*TH13*TH21**2*v - 2*complex(0,1)*l5*TH13*TH21**2*v - (3*complex(0,1)*l6*TH11**2*TH23*v)/2. - 2*complex(0,1)*l3*TH11*TH21*TH23*v - 2*complex(0,1)*l4*TH11*TH21*TH23*v - 4*complex(0,1)*l5*TH11*TH21*TH23*v - (3*complex(0,1)*l7*TH21**2*TH23*v)/2. + 3*l6*TH11*TH13*TH31*v + l7*TH21*TH23*TH31*v - complex(0,1)*l3*TH13*TH31**2*v - complex(0,1)*l4*TH13*TH31**2*v + 2*complex(0,1)*l5*TH13*TH31**2*v - (complex(0,1)*l7*TH23*TH31**2*v)/2. + (3*l6*TH11**2*TH33*v)/2. + (l7*TH21**2*TH33*v)/2. - 2*complex(0,1)*l3*TH11*TH31*TH33*v - 2*complex(0,1)*l4*TH11*TH31*TH33*v + 4*complex(0,1)*l5*TH11*TH31*TH33*v - complex(0,1)*l7*TH21*TH31*TH33*v + (3*l7*TH31**2*TH33*v)/2. - 3*complex(0,1)*TH11*TH13*TH21*v*complexconjugate(l6) - (3*complex(0,1)*TH11**2*TH23*v*complexconjugate(l6))/2. - 3*TH11*TH13*TH31*v*complexconjugate(l6) - (3*TH11**2*TH33*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH21**2*TH23*v*complexconjugate(l7))/2. - TH21*TH23*TH31*v*complexconjugate(l7) - (complex(0,1)*TH23*TH31**2*v*complexconjugate(l7))/2. - (TH21**2*TH33*v*complexconjugate(l7))/2. - complex(0,1)*TH21*TH31*TH33*v*complexconjugate(l7) - (3*TH31**2*TH33*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_313 = Coupling(name = 'GC_313',
                  value = '-6*complex(0,1)*l1*TH11*TH12*TH13*v - (3*complex(0,1)*l6*TH12*TH13*TH21*v)/2. - (3*complex(0,1)*l6*TH11*TH13*TH22*v)/2. - complex(0,1)*l3*TH13*TH21*TH22*v - complex(0,1)*l4*TH13*TH21*TH22*v - 2*complex(0,1)*l5*TH13*TH21*TH22*v - (3*complex(0,1)*l6*TH11*TH12*TH23*v)/2. - complex(0,1)*l3*TH12*TH21*TH23*v - complex(0,1)*l4*TH12*TH21*TH23*v - 2*complex(0,1)*l5*TH12*TH21*TH23*v - complex(0,1)*l3*TH11*TH22*TH23*v - complex(0,1)*l4*TH11*TH22*TH23*v - 2*complex(0,1)*l5*TH11*TH22*TH23*v - (3*complex(0,1)*l7*TH21*TH22*TH23*v)/2. + (3*l6*TH12*TH13*TH31*v)/2. + (l7*TH22*TH23*TH31*v)/2. + (3*l6*TH11*TH13*TH32*v)/2. + (l7*TH21*TH23*TH32*v)/2. - complex(0,1)*l3*TH13*TH31*TH32*v - complex(0,1)*l4*TH13*TH31*TH32*v + 2*complex(0,1)*l5*TH13*TH31*TH32*v - (complex(0,1)*l7*TH23*TH31*TH32*v)/2. + (3*l6*TH11*TH12*TH33*v)/2. + (l7*TH21*TH22*TH33*v)/2. - complex(0,1)*l3*TH12*TH31*TH33*v - complex(0,1)*l4*TH12*TH31*TH33*v + 2*complex(0,1)*l5*TH12*TH31*TH33*v - (complex(0,1)*l7*TH22*TH31*TH33*v)/2. - complex(0,1)*l3*TH11*TH32*TH33*v - complex(0,1)*l4*TH11*TH32*TH33*v + 2*complex(0,1)*l5*TH11*TH32*TH33*v - (complex(0,1)*l7*TH21*TH32*TH33*v)/2. + (3*l7*TH31*TH32*TH33*v)/2. - (3*complex(0,1)*TH12*TH13*TH21*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH11*TH13*TH22*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH11*TH12*TH23*v*complexconjugate(l6))/2. - (3*TH12*TH13*TH31*v*complexconjugate(l6))/2. - (3*TH11*TH13*TH32*v*complexconjugate(l6))/2. - (3*TH11*TH12*TH33*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH21*TH22*TH23*v*complexconjugate(l7))/2. - (TH22*TH23*TH31*v*complexconjugate(l7))/2. - (TH21*TH23*TH32*v*complexconjugate(l7))/2. - (complex(0,1)*TH23*TH31*TH32*v*complexconjugate(l7))/2. - (TH21*TH22*TH33*v*complexconjugate(l7))/2. - (complex(0,1)*TH22*TH31*TH33*v*complexconjugate(l7))/2. - (complex(0,1)*TH21*TH32*TH33*v*complexconjugate(l7))/2. - (3*TH31*TH32*TH33*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_314 = Coupling(name = 'GC_314',
                  value = '-6*complex(0,1)*l1*TH12**2*TH13*v - 3*complex(0,1)*l6*TH12*TH13*TH22*v - complex(0,1)*l3*TH13*TH22**2*v - complex(0,1)*l4*TH13*TH22**2*v - 2*complex(0,1)*l5*TH13*TH22**2*v - (3*complex(0,1)*l6*TH12**2*TH23*v)/2. - 2*complex(0,1)*l3*TH12*TH22*TH23*v - 2*complex(0,1)*l4*TH12*TH22*TH23*v - 4*complex(0,1)*l5*TH12*TH22*TH23*v - (3*complex(0,1)*l7*TH22**2*TH23*v)/2. + 3*l6*TH12*TH13*TH32*v + l7*TH22*TH23*TH32*v - complex(0,1)*l3*TH13*TH32**2*v - complex(0,1)*l4*TH13*TH32**2*v + 2*complex(0,1)*l5*TH13*TH32**2*v - (complex(0,1)*l7*TH23*TH32**2*v)/2. + (3*l6*TH12**2*TH33*v)/2. + (l7*TH22**2*TH33*v)/2. - 2*complex(0,1)*l3*TH12*TH32*TH33*v - 2*complex(0,1)*l4*TH12*TH32*TH33*v + 4*complex(0,1)*l5*TH12*TH32*TH33*v - complex(0,1)*l7*TH22*TH32*TH33*v + (3*l7*TH32**2*TH33*v)/2. - 3*complex(0,1)*TH12*TH13*TH22*v*complexconjugate(l6) - (3*complex(0,1)*TH12**2*TH23*v*complexconjugate(l6))/2. - 3*TH12*TH13*TH32*v*complexconjugate(l6) - (3*TH12**2*TH33*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH22**2*TH23*v*complexconjugate(l7))/2. - TH22*TH23*TH32*v*complexconjugate(l7) - (complex(0,1)*TH23*TH32**2*v*complexconjugate(l7))/2. - (TH22**2*TH33*v*complexconjugate(l7))/2. - complex(0,1)*TH22*TH32*TH33*v*complexconjugate(l7) - (3*TH32**2*TH33*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_315 = Coupling(name = 'GC_315',
                  value = '-6*complex(0,1)*l1*TH11*TH13**2*v - (3*complex(0,1)*l6*TH13**2*TH21*v)/2. - 3*complex(0,1)*l6*TH11*TH13*TH23*v - 2*complex(0,1)*l3*TH13*TH21*TH23*v - 2*complex(0,1)*l4*TH13*TH21*TH23*v - 4*complex(0,1)*l5*TH13*TH21*TH23*v - complex(0,1)*l3*TH11*TH23**2*v - complex(0,1)*l4*TH11*TH23**2*v - 2*complex(0,1)*l5*TH11*TH23**2*v - (3*complex(0,1)*l7*TH21*TH23**2*v)/2. + (3*l6*TH13**2*TH31*v)/2. + (l7*TH23**2*TH31*v)/2. + 3*l6*TH11*TH13*TH33*v + l7*TH21*TH23*TH33*v - 2*complex(0,1)*l3*TH13*TH31*TH33*v - 2*complex(0,1)*l4*TH13*TH31*TH33*v + 4*complex(0,1)*l5*TH13*TH31*TH33*v - complex(0,1)*l7*TH23*TH31*TH33*v - complex(0,1)*l3*TH11*TH33**2*v - complex(0,1)*l4*TH11*TH33**2*v + 2*complex(0,1)*l5*TH11*TH33**2*v - (complex(0,1)*l7*TH21*TH33**2*v)/2. + (3*l7*TH31*TH33**2*v)/2. - (3*complex(0,1)*TH13**2*TH21*v*complexconjugate(l6))/2. - 3*complex(0,1)*TH11*TH13*TH23*v*complexconjugate(l6) - (3*TH13**2*TH31*v*complexconjugate(l6))/2. - 3*TH11*TH13*TH33*v*complexconjugate(l6) - (3*complex(0,1)*TH21*TH23**2*v*complexconjugate(l7))/2. - (TH23**2*TH31*v*complexconjugate(l7))/2. - TH21*TH23*TH33*v*complexconjugate(l7) - complex(0,1)*TH23*TH31*TH33*v*complexconjugate(l7) - (complex(0,1)*TH21*TH33**2*v*complexconjugate(l7))/2. - (3*TH31*TH33**2*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_316 = Coupling(name = 'GC_316',
                  value = '-6*complex(0,1)*l1*TH12*TH13**2*v - (3*complex(0,1)*l6*TH13**2*TH22*v)/2. - 3*complex(0,1)*l6*TH12*TH13*TH23*v - 2*complex(0,1)*l3*TH13*TH22*TH23*v - 2*complex(0,1)*l4*TH13*TH22*TH23*v - 4*complex(0,1)*l5*TH13*TH22*TH23*v - complex(0,1)*l3*TH12*TH23**2*v - complex(0,1)*l4*TH12*TH23**2*v - 2*complex(0,1)*l5*TH12*TH23**2*v - (3*complex(0,1)*l7*TH22*TH23**2*v)/2. + (3*l6*TH13**2*TH32*v)/2. + (l7*TH23**2*TH32*v)/2. + 3*l6*TH12*TH13*TH33*v + l7*TH22*TH23*TH33*v - 2*complex(0,1)*l3*TH13*TH32*TH33*v - 2*complex(0,1)*l4*TH13*TH32*TH33*v + 4*complex(0,1)*l5*TH13*TH32*TH33*v - complex(0,1)*l7*TH23*TH32*TH33*v - complex(0,1)*l3*TH12*TH33**2*v - complex(0,1)*l4*TH12*TH33**2*v + 2*complex(0,1)*l5*TH12*TH33**2*v - (complex(0,1)*l7*TH22*TH33**2*v)/2. + (3*l7*TH32*TH33**2*v)/2. - (3*complex(0,1)*TH13**2*TH22*v*complexconjugate(l6))/2. - 3*complex(0,1)*TH12*TH13*TH23*v*complexconjugate(l6) - (3*TH13**2*TH32*v*complexconjugate(l6))/2. - 3*TH12*TH13*TH33*v*complexconjugate(l6) - (3*complex(0,1)*TH22*TH23**2*v*complexconjugate(l7))/2. - (TH23**2*TH32*v*complexconjugate(l7))/2. - TH22*TH23*TH33*v*complexconjugate(l7) - complex(0,1)*TH23*TH32*TH33*v*complexconjugate(l7) - (complex(0,1)*TH22*TH33**2*v*complexconjugate(l7))/2. - (3*TH32*TH33**2*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

GC_317 = Coupling(name = 'GC_317',
                  value = '-6*complex(0,1)*l1*TH13**3*v - (9*complex(0,1)*l6*TH13**2*TH23*v)/2. - 3*complex(0,1)*l3*TH13*TH23**2*v - 3*complex(0,1)*l4*TH13*TH23**2*v - 6*complex(0,1)*l5*TH13*TH23**2*v - (3*complex(0,1)*l7*TH23**3*v)/2. + (9*l6*TH13**2*TH33*v)/2. + (3*l7*TH23**2*TH33*v)/2. - 3*complex(0,1)*l3*TH13*TH33**2*v - 3*complex(0,1)*l4*TH13*TH33**2*v + 6*complex(0,1)*l5*TH13*TH33**2*v - (3*complex(0,1)*l7*TH23*TH33**2*v)/2. + (3*l7*TH33**3*v)/2. - (9*complex(0,1)*TH13**2*TH23*v*complexconjugate(l6))/2. - (9*TH13**2*TH33*v*complexconjugate(l6))/2. - (3*complex(0,1)*TH23**3*v*complexconjugate(l7))/2. - (3*TH23**2*TH33*v*complexconjugate(l7))/2. - (3*complex(0,1)*TH23*TH33**2*v*complexconjugate(l7))/2. - (3*TH33**3*v*complexconjugate(l7))/2.',
                  order = {'QED':1})

