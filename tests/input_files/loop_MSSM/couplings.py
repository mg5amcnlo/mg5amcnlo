# This file was automatically created by FeynRules 2.1.46
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Wed 19 Mar 2014 10:12:28


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot



GC_1 = Coupling(name = 'GC_1',
                value = '-(ee*complex(0,1))/3.',
                order = {'QED':1})

GC_10 = Coupling(name = 'GC_10',
                 value = 'G/2.',
                 order = {'QCD':1})

GC_100 = Coupling(name = 'GC_100',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN1x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1000 = Coupling(name = 'GC_1000',
                   value = '(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*alp))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1001 = Coupling(name = 'GC_1001',
                   value = '-(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1002 = Coupling(name = 'GC_1002',
                   value = '(complex(0,1)*Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rd3x3*sw**2*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(alp)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*alp))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1003 = Coupling(name = 'GC_1003',
                   value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6)*cmath.cos(2*alp))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1004 = Coupling(name = 'GC_1004',
                   value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*alp))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1005 = Coupling(name = 'GC_1005',
                   value = '-(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1006 = Coupling(name = 'GC_1006',
                   value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*alp))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1007 = Coupling(name = 'GC_1007',
                   value = '-(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1008 = Coupling(name = 'GC_1008',
                   value = '(complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rl3x3*sw**2*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(alp)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*alp))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1009 = Coupling(name = 'GC_1009',
                   value = '(complex(0,1)*Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)*cmath.cos(2*alp))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_101 = Coupling(name = 'GC_101',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN1x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1010 = Coupling(name = 'GC_1010',
                   value = '(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1011 = Coupling(name = 'GC_1011',
                   value = '-(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1012 = Coupling(name = 'GC_1012',
                   value = '(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1013 = Coupling(name = 'GC_1013',
                   value = '-(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1014 = Coupling(name = 'GC_1014',
                   value = '(complex(0,1)*Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (complex(0,1)*Ru3x3*sw**2*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(alp)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1015 = Coupling(name = 'GC_1015',
                   value = '(complex(0,1)*Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(alp)**2)/(-1 + sw**2) - (ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6)*cmath.cos(2*alp))/(3.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1016 = Coupling(name = 'GC_1016',
                   value = '(ee**2*complex(0,1))/(8.*sw**2*(-1 + sw**2)) - (3*ee**2*complex(0,1)*cmath.cos(4*alp))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1017 = Coupling(name = 'GC_1017',
                   value = '-(ee**2*cmath.cos(alp - beta))/(4.*sw**2)',
                   order = {'QED':2})

GC_1018 = Coupling(name = 'GC_1018',
                   value = '(ee**2*cmath.cos(alp - beta))/(4.*sw**2)',
                   order = {'QED':2})

GC_1019 = Coupling(name = 'GC_1019',
                   value = '-(ee*complex(0,1)*cmath.cos(alp - beta))/(2.*sw)',
                   order = {'QED':1})

GC_102 = Coupling(name = 'GC_102',
                  value = '-((cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN1x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1020 = Coupling(name = 'GC_1020',
                   value = '(ee*complex(0,1)*cmath.cos(alp - beta))/(2.*sw)',
                   order = {'QED':1})

GC_1021 = Coupling(name = 'GC_1021',
                   value = '(ee**2*complex(0,1)*cmath.cos(alp - beta))/(2.*sw)',
                   order = {'QED':2})

GC_1022 = Coupling(name = 'GC_1022',
                   value = '(cw*ee**2*complex(0,1)*cmath.cos(alp - beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1023 = Coupling(name = 'GC_1023',
                   value = '-(cw*ee*cmath.cos(alp - beta))/(2.*sw*(-1 + sw**2))',
                   order = {'QED':1})

GC_1024 = Coupling(name = 'GC_1024',
                   value = '(cw*ee*cmath.cos(alp - beta))/(2.*sw*(-1 + sw**2))',
                   order = {'QED':1})

GC_1025 = Coupling(name = 'GC_1025',
                   value = '-((CKM3x3*complex(0,1)*Ru6x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(alp - beta))/cmath.sqrt(2))',
                   order = {'QED':2})

GC_1026 = Coupling(name = 'GC_1026',
                   value = '(CKM3x3*complex(0,1)*Ru6x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(alp - beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_1027 = Coupling(name = 'GC_1027',
                   value = '-((complex(0,1)*Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(alp - beta))/cmath.sqrt(2))',
                   order = {'QED':2})

GC_1028 = Coupling(name = 'GC_1028',
                   value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(alp - beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_1029 = Coupling(name = 'GC_1029',
                   value = '-((yd3x3*cmath.cos(beta))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_103 = Coupling(name = 'GC_103',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN1x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1030 = Coupling(name = 'GC_1030',
                   value = '-(complex(0,1)*ye3x3*cmath.cos(beta))',
                   order = {'QED':1})

GC_1031 = Coupling(name = 'GC_1031',
                   value = '-((ye3x3*cmath.cos(beta))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_1032 = Coupling(name = 'GC_1032',
                   value = '(yu3x3*cmath.cos(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1033 = Coupling(name = 'GC_1033',
                   value = 'CKM3x3*complex(0,1)*yu3x3*cmath.cos(beta)',
                   order = {'QED':1})

GC_1034 = Coupling(name = 'GC_1034',
                   value = '-(complex(0,1)*yd3x3*complexconjugate(CKM3x3)*cmath.cos(beta))',
                   order = {'QED':1})

GC_1035 = Coupling(name = 'GC_1035',
                   value = '(complexconjugate(yd3x3)*cmath.cos(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1036 = Coupling(name = 'GC_1036',
                   value = '-(CKM3x3*complex(0,1)*complexconjugate(yd3x3)*cmath.cos(beta))',
                   order = {'QED':1})

GC_1037 = Coupling(name = 'GC_1037',
                   value = '-(complex(0,1)*complexconjugate(ye3x3)*cmath.cos(beta))',
                   order = {'QED':1})

GC_1038 = Coupling(name = 'GC_1038',
                   value = '(complexconjugate(ye3x3)*cmath.cos(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1039 = Coupling(name = 'GC_1039',
                   value = '-((complexconjugate(yu3x3)*cmath.cos(beta))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_104 = Coupling(name = 'GC_104',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN1x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1040 = Coupling(name = 'GC_1040',
                   value = 'complex(0,1)*complexconjugate(CKM3x3)*complexconjugate(yu3x3)*cmath.cos(beta)',
                   order = {'QED':1})

GC_1041 = Coupling(name = 'GC_1041',
                   value = '(ee*complex(0,1)*NN1x3*UU1x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN1x2*UU1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN1x1*UU1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1042 = Coupling(name = 'GC_1042',
                   value = '(ee*complex(0,1)*NN2x3*UU1x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN2x2*UU1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN2x1*UU1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1043 = Coupling(name = 'GC_1043',
                   value = '(ee*complex(0,1)*NN3x3*UU1x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN3x2*UU1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN3x1*UU1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1044 = Coupling(name = 'GC_1044',
                   value = '(ee*complex(0,1)*NN4x3*UU1x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN4x2*UU1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN4x1*UU1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1045 = Coupling(name = 'GC_1045',
                   value = '(ee*complex(0,1)*NN1x3*UU2x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN1x2*UU2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN1x1*UU2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1046 = Coupling(name = 'GC_1046',
                   value = '(ee*complex(0,1)*NN2x3*UU2x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN2x2*UU2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN2x1*UU2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1047 = Coupling(name = 'GC_1047',
                   value = '(ee*complex(0,1)*NN3x3*UU2x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN3x2*UU2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN3x1*UU2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1048 = Coupling(name = 'GC_1048',
                   value = '(ee*complex(0,1)*NN4x3*UU2x1*cmath.cos(beta))/sw - (ee*complex(0,1)*NN4x2*UU2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN4x1*UU2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1049 = Coupling(name = 'GC_1049',
                   value = '-((ee*complex(0,1)*NN1x4*VV1x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN1x2*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN1x1*VV1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_105 = Coupling(name = 'GC_105',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN1x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1050 = Coupling(name = 'GC_1050',
                   value = '-((ee*complex(0,1)*NN2x4*VV1x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN2x2*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN2x1*VV1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1051 = Coupling(name = 'GC_1051',
                   value = '-((ee*complex(0,1)*NN3x4*VV1x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN3x2*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN3x1*VV1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1052 = Coupling(name = 'GC_1052',
                   value = '-((ee*complex(0,1)*NN4x4*VV1x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN4x2*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN4x1*VV1x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1053 = Coupling(name = 'GC_1053',
                   value = '-((ee*complex(0,1)*NN1x4*VV2x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN1x2*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN1x1*VV2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1054 = Coupling(name = 'GC_1054',
                   value = '-((ee*complex(0,1)*NN2x4*VV2x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN2x2*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN2x1*VV2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1055 = Coupling(name = 'GC_1055',
                   value = '-((ee*complex(0,1)*NN3x4*VV2x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN3x2*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN3x1*VV2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1056 = Coupling(name = 'GC_1056',
                   value = '-((ee*complex(0,1)*NN4x4*VV2x1*cmath.cos(beta))/sw) - (ee*complex(0,1)*NN4x2*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN4x1*VV2x2*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1057 = Coupling(name = 'GC_1057',
                   value = '(ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(UU1x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(UU1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(UU1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1058 = Coupling(name = 'GC_1058',
                   value = '(ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(UU1x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(UU1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(UU1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1059 = Coupling(name = 'GC_1059',
                   value = '(ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(UU1x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(UU1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(UU1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_106 = Coupling(name = 'GC_106',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1060 = Coupling(name = 'GC_1060',
                   value = '(ee*complex(0,1)*complexconjugate(NN4x3)*complexconjugate(UU1x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(UU1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(UU1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1061 = Coupling(name = 'GC_1061',
                   value = '(ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(UU2x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(UU2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(UU2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1062 = Coupling(name = 'GC_1062',
                   value = '(ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(UU2x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(UU2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(UU2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1063 = Coupling(name = 'GC_1063',
                   value = '(ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(UU2x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(UU2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(UU2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1064 = Coupling(name = 'GC_1064',
                   value = '(ee*complex(0,1)*complexconjugate(NN4x3)*complexconjugate(UU2x1)*cmath.cos(beta))/sw + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(UU2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(UU2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1065 = Coupling(name = 'GC_1065',
                   value = '-((ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(VV1x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1066 = Coupling(name = 'GC_1066',
                   value = '-((ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(VV1x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1067 = Coupling(name = 'GC_1067',
                   value = '-((ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(VV1x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1068 = Coupling(name = 'GC_1068',
                   value = '-((ee*complex(0,1)*complexconjugate(NN4x4)*complexconjugate(VV1x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(VV1x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1069 = Coupling(name = 'GC_1069',
                   value = '-((ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(VV2x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_107 = Coupling(name = 'GC_107',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1070 = Coupling(name = 'GC_1070',
                   value = '-((ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(VV2x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1071 = Coupling(name = 'GC_1071',
                   value = '-((ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(VV2x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1072 = Coupling(name = 'GC_1072',
                   value = '-((ee*complex(0,1)*complexconjugate(NN4x4)*complexconjugate(VV2x1)*cmath.cos(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(VV2x2)*cmath.cos(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1073 = Coupling(name = 'GC_1073',
                   value = '-(ee**2*complex(0,1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1074 = Coupling(name = 'GC_1074',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1075 = Coupling(name = 'GC_1075',
                   value = '-(CKM1x1*ee**2*Ru1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1076 = Coupling(name = 'GC_1076',
                   value = '(CKM1x1*ee**2*Ru1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1077 = Coupling(name = 'GC_1077',
                   value = '-(CKM2x2*ee**2*Ru2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1078 = Coupling(name = 'GC_1078',
                   value = '(CKM2x2*ee**2*Ru2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1079 = Coupling(name = 'GC_1079',
                   value = '-(ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4)*cmath.cos(2*beta))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_108 = Coupling(name = 'GC_108',
                  value = '(cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1080 = Coupling(name = 'GC_1080',
                   value = '(ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4)*cmath.cos(2*beta))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1081 = Coupling(name = 'GC_1081',
                   value = '-(ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5)*cmath.cos(2*beta))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1082 = Coupling(name = 'GC_1082',
                   value = '(ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5)*cmath.cos(2*beta))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1083 = Coupling(name = 'GC_1083',
                   value = '-(ee**2*Rn1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1084 = Coupling(name = 'GC_1084',
                   value = '(ee**2*Rn1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1085 = Coupling(name = 'GC_1085',
                   value = '-(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1086 = Coupling(name = 'GC_1086',
                   value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1087 = Coupling(name = 'GC_1087',
                   value = '-(ee**2*Rn2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1088 = Coupling(name = 'GC_1088',
                   value = '(ee**2*Rn2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1089 = Coupling(name = 'GC_1089',
                   value = '-(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_109 = Coupling(name = 'GC_109',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN1x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN1x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1090 = Coupling(name = 'GC_1090',
                   value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1091 = Coupling(name = 'GC_1091',
                   value = '-(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1092 = Coupling(name = 'GC_1092',
                   value = '(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1093 = Coupling(name = 'GC_1093',
                   value = '-(ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)*cmath.cos(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1094 = Coupling(name = 'GC_1094',
                   value = '(ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)*cmath.cos(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1095 = Coupling(name = 'GC_1095',
                   value = '-(ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)*cmath.cos(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1096 = Coupling(name = 'GC_1096',
                   value = '(ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)*cmath.cos(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1097 = Coupling(name = 'GC_1097',
                   value = '-(ee**2*Rl1x1*complexconjugate(Rn1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1098 = Coupling(name = 'GC_1098',
                   value = '(ee**2*Rl1x1*complexconjugate(Rn1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1099 = Coupling(name = 'GC_1099',
                   value = '-(ee**2*Rl2x2*complexconjugate(Rn2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_11 = Coupling(name = 'GC_11',
                 value = 'complex(0,1)*G**2',
                 order = {'QCD':2})

GC_110 = Coupling(name = 'GC_110',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN1x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN1x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1100 = Coupling(name = 'GC_1100',
                   value = '(ee**2*Rl2x2*complexconjugate(Rn2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1101 = Coupling(name = 'GC_1101',
                   value = '-(ee**2*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1102 = Coupling(name = 'GC_1102',
                   value = '(ee**2*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1103 = Coupling(name = 'GC_1103',
                   value = '-(ee**2*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1104 = Coupling(name = 'GC_1104',
                   value = '(ee**2*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1105 = Coupling(name = 'GC_1105',
                   value = '-(ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4)*cmath.cos(2*beta))/(3.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1106 = Coupling(name = 'GC_1106',
                   value = '(ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4)*cmath.cos(2*beta))/(3.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1107 = Coupling(name = 'GC_1107',
                   value = '-(ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5)*cmath.cos(2*beta))/(3.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1108 = Coupling(name = 'GC_1108',
                   value = '(ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5)*cmath.cos(2*beta))/(3.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1109 = Coupling(name = 'GC_1109',
                   value = '-(ee**2*complex(0,1)*cmath.cos(2*alp)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_111 = Coupling(name = 'GC_111',
                  value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(NN1x3))/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(NN1x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_1110 = Coupling(name = 'GC_1110',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*alp)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1111 = Coupling(name = 'GC_1111',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*beta)**2)/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1112 = Coupling(name = 'GC_1112',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*beta)**2)/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1113 = Coupling(name = 'GC_1113',
                   value = '(3*ee**2*complex(0,1)*cmath.cos(2*beta)**2)/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1114 = Coupling(name = 'GC_1114',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1115 = Coupling(name = 'GC_1115',
                   value = '(complex(0,1)*Rn3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rn3x3*sw**2*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1116 = Coupling(name = 'GC_1116',
                   value = '-(ee**2*complex(0,1)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1117 = Coupling(name = 'GC_1117',
                   value = '(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1118 = Coupling(name = 'GC_1118',
                   value = '-(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1119 = Coupling(name = 'GC_1119',
                   value = '-(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_112 = Coupling(name = 'GC_112',
                  value = '(complex(0,1)*Rl6x6*ye3x3*complexconjugate(NN1x3))/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(NN1x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_1120 = Coupling(name = 'GC_1120',
                   value = '(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1121 = Coupling(name = 'GC_1121',
                   value = '(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1122 = Coupling(name = 'GC_1122',
                   value = '-(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1123 = Coupling(name = 'GC_1123',
                   value = '-(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1124 = Coupling(name = 'GC_1124',
                   value = '(complex(0,1)*Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rd3x3*sw**2*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1125 = Coupling(name = 'GC_1125',
                   value = '(CKM3x3*complex(0,1)*Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (CKM3x3*complex(0,1)*Rd3x3*sw**2*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1126 = Coupling(name = 'GC_1126',
                   value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6)*cmath.cos(2*beta))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1127 = Coupling(name = 'GC_1127',
                   value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1128 = Coupling(name = 'GC_1128',
                   value = '-(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1129 = Coupling(name = 'GC_1129',
                   value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_113 = Coupling(name = 'GC_113',
                  value = '(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN1x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN1x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1130 = Coupling(name = 'GC_1130',
                   value = '-(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1131 = Coupling(name = 'GC_1131',
                   value = '-((Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/cmath.sqrt(2)) + (ee**2*Rn3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1132 = Coupling(name = 'GC_1132',
                   value = '(complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rl3x3*sw**2*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1133 = Coupling(name = 'GC_1133',
                   value = '(complex(0,1)*Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)*cmath.cos(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1134 = Coupling(name = 'GC_1134',
                   value = '(Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(beta)**2)/cmath.sqrt(2) - (ee**2*Rl3x3*complexconjugate(Rn3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1135 = Coupling(name = 'GC_1135',
                   value = '(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1136 = Coupling(name = 'GC_1136',
                   value = '(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1137 = Coupling(name = 'GC_1137',
                   value = '-(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1138 = Coupling(name = 'GC_1138',
                   value = '-(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1139 = Coupling(name = 'GC_1139',
                   value = '(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_114 = Coupling(name = 'GC_114',
                  value = '(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN1x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN1x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1140 = Coupling(name = 'GC_1140',
                   value = '(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1141 = Coupling(name = 'GC_1141',
                   value = '-(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1142 = Coupling(name = 'GC_1142',
                   value = '-(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1143 = Coupling(name = 'GC_1143',
                   value = '(CKM3x3*complex(0,1)*Ru3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (CKM3x3*complex(0,1)*Ru3x3*sw**2*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1144 = Coupling(name = 'GC_1144',
                   value = '(complex(0,1)*Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (complex(0,1)*Ru3x3*sw**2*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/(-1 + sw**2) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1145 = Coupling(name = 'GC_1145',
                   value = '(complex(0,1)*Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/(-1 + sw**2) - (ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6)*cmath.cos(2*beta))/(3.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1146 = Coupling(name = 'GC_1146',
                   value = '-(ee**2*complex(0,1)*cmath.cos(4*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1147 = Coupling(name = 'GC_1147',
                   value = '(ee**2*complex(0,1))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(4*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1148 = Coupling(name = 'GC_1148',
                   value = '-(ee**2*complex(0,1))/(2.*(-1 + sw**2)) + (3*ee**2*complex(0,1))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(4*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1149 = Coupling(name = 'GC_1149',
                   value = '(ee**2*complex(0,1))/(8.*sw**2*(-1 + sw**2)) - (3*ee**2*complex(0,1)*cmath.cos(4*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_115 = Coupling(name = 'GC_115',
                  value = '(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN1x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN1x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1150 = Coupling(name = 'GC_1150',
                   value = '-(CKM1x1*ee**2*complex(0,1)*Ru1x1*complexconjugate(Rd1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1151 = Coupling(name = 'GC_1151',
                   value = '(CKM1x1*ee**2*complex(0,1)*Ru1x1*complexconjugate(Rd1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1152 = Coupling(name = 'GC_1152',
                   value = '-(CKM2x2*ee**2*complex(0,1)*Ru2x2*complexconjugate(Rd2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1153 = Coupling(name = 'GC_1153',
                   value = '(CKM2x2*ee**2*complex(0,1)*Ru2x2*complexconjugate(Rd2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1154 = Coupling(name = 'GC_1154',
                   value = '-(ee**2*complex(0,1)*Rn1x1*complexconjugate(Rl1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1155 = Coupling(name = 'GC_1155',
                   value = '(ee**2*complex(0,1)*Rn1x1*complexconjugate(Rl1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1156 = Coupling(name = 'GC_1156',
                   value = '-(ee**2*complex(0,1)*Rn2x2*complexconjugate(Rl2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1157 = Coupling(name = 'GC_1157',
                   value = '(ee**2*complex(0,1)*Rn2x2*complexconjugate(Rl2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1158 = Coupling(name = 'GC_1158',
                   value = '-(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rn1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1159 = Coupling(name = 'GC_1159',
                   value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rn1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_116 = Coupling(name = 'GC_116',
                  value = '(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN1x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN1x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1160 = Coupling(name = 'GC_1160',
                   value = '-(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rn2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1161 = Coupling(name = 'GC_1161',
                   value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rn2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1162 = Coupling(name = 'GC_1162',
                   value = '-(ee**2*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1163 = Coupling(name = 'GC_1163',
                   value = '(ee**2*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1164 = Coupling(name = 'GC_1164',
                   value = '-(ee**2*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1165 = Coupling(name = 'GC_1165',
                   value = '(ee**2*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1166 = Coupling(name = 'GC_1166',
                   value = '-((complex(0,1)*Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(alp)*cmath.cos(beta))/cmath.sqrt(2)) + (ee**2*complex(0,1)*Rn3x3*complexconjugate(Rl3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1167 = Coupling(name = 'GC_1167',
                   value = '-((complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(alp)*cmath.cos(beta))/cmath.sqrt(2)) + (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rn3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_1168 = Coupling(name = 'GC_1168',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*(alp - beta)))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(2*(alp - beta)))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(2*(alp + beta)))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1169 = Coupling(name = 'GC_1169',
                   value = '-(ee**2*complex(0,1))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(2*(alp - beta)))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.cos(2*(alp - beta)))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(2*(alp + beta)))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_117 = Coupling(name = 'GC_117',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN1x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN1x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1170 = Coupling(name = 'GC_1170',
                   value = '-(ee**2*complex(0,1))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.cos(2*(alp - beta)))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.cos(2*(alp - beta)))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.cos(2*(alp + beta)))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_118 = Coupling(name = 'GC_118',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN1x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN1x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_119 = Coupling(name = 'GC_119',
                  value = '(complex(0,1)*Ru6x6*yu3x3*complexconjugate(NN1x4))/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(NN1x4))/(-1 + sw**2)',
                  order = {'QED':1})

GC_12 = Coupling(name = 'GC_12',
                 value = '-(complex(0,1)*G*Rd1x1*cmath.sqrt(2))',
                 order = {'QCD':1})

GC_120 = Coupling(name = 'GC_120',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN2x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_121 = Coupling(name = 'GC_121',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN2x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_122 = Coupling(name = 'GC_122',
                  value = '(cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN2x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_123 = Coupling(name = 'GC_123',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN2x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_124 = Coupling(name = 'GC_124',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN2x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_125 = Coupling(name = 'GC_125',
                  value = '-((cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN2x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_126 = Coupling(name = 'GC_126',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN2x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_127 = Coupling(name = 'GC_127',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN2x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_128 = Coupling(name = 'GC_128',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN2x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_129 = Coupling(name = 'GC_129',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN2x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_13 = Coupling(name = 'GC_13',
                 value = '-(complex(0,1)*G*Rd2x2*cmath.sqrt(2))',
                 order = {'QCD':1})

GC_130 = Coupling(name = 'GC_130',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN2x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_131 = Coupling(name = 'GC_131',
                  value = '(cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN2x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_132 = Coupling(name = 'GC_132',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN2x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN2x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_133 = Coupling(name = 'GC_133',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN2x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN2x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_134 = Coupling(name = 'GC_134',
                  value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(NN2x3))/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(NN2x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_135 = Coupling(name = 'GC_135',
                  value = '(complex(0,1)*Rl6x6*ye3x3*complexconjugate(NN2x3))/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(NN2x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_136 = Coupling(name = 'GC_136',
                  value = '-(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN2x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN2x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_137 = Coupling(name = 'GC_137',
                  value = '(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN2x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN2x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_138 = Coupling(name = 'GC_138',
                  value = '(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN2x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN2x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_139 = Coupling(name = 'GC_139',
                  value = '(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN2x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN2x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_14 = Coupling(name = 'GC_14',
                 value = '-(complex(0,1)*G*Rd3x3*cmath.sqrt(2))',
                 order = {'QCD':1})

GC_140 = Coupling(name = 'GC_140',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN2x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN2x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_141 = Coupling(name = 'GC_141',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN2x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN2x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_142 = Coupling(name = 'GC_142',
                  value = '(complex(0,1)*Ru6x6*yu3x3*complexconjugate(NN2x4))/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(NN2x4))/(-1 + sw**2)',
                  order = {'QED':1})

GC_143 = Coupling(name = 'GC_143',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN3x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_144 = Coupling(name = 'GC_144',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN3x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_145 = Coupling(name = 'GC_145',
                  value = '(cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN3x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_146 = Coupling(name = 'GC_146',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN3x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_147 = Coupling(name = 'GC_147',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN3x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_148 = Coupling(name = 'GC_148',
                  value = '-((cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN3x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_149 = Coupling(name = 'GC_149',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN3x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_15 = Coupling(name = 'GC_15',
                 value = 'complex(0,1)*G*Rd4x4*cmath.sqrt(2)',
                 order = {'QCD':1})

GC_150 = Coupling(name = 'GC_150',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN3x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_151 = Coupling(name = 'GC_151',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN3x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_152 = Coupling(name = 'GC_152',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN3x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_153 = Coupling(name = 'GC_153',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN3x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_154 = Coupling(name = 'GC_154',
                  value = '(cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN3x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN3x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN3x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_155 = Coupling(name = 'GC_155',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN3x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_156 = Coupling(name = 'GC_156',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN3x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_157 = Coupling(name = 'GC_157',
                  value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(NN3x3))/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(NN3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_158 = Coupling(name = 'GC_158',
                  value = '(complex(0,1)*Rl6x6*ye3x3*complexconjugate(NN3x3))/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(NN3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_159 = Coupling(name = 'GC_159',
                  value = '-(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN3x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN3x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_16 = Coupling(name = 'GC_16',
                 value = 'complex(0,1)*G*Rd5x5*cmath.sqrt(2)',
                 order = {'QCD':1})

GC_160 = Coupling(name = 'GC_160',
                  value = '-(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN3x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN3x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_161 = Coupling(name = 'GC_161',
                  value = '(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN3x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN3x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_162 = Coupling(name = 'GC_162',
                  value = '(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN3x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN3x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_163 = Coupling(name = 'GC_163',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN3x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN3x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_164 = Coupling(name = 'GC_164',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN3x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN3x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_165 = Coupling(name = 'GC_165',
                  value = '(complex(0,1)*Ru6x6*yu3x3*complexconjugate(NN3x4))/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(NN3x4))/(-1 + sw**2)',
                  order = {'QED':1})

GC_166 = Coupling(name = 'GC_166',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN4x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_167 = Coupling(name = 'GC_167',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN4x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_168 = Coupling(name = 'GC_168',
                  value = '(cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN4x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_169 = Coupling(name = 'GC_169',
                  value = '-((cw*ee*complex(0,1)*Rl1x1*complexconjugate(NN4x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl1x1*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl1x1*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_17 = Coupling(name = 'GC_17',
                 value = 'complex(0,1)*G*Rd6x6*cmath.sqrt(2)',
                 order = {'QCD':1})

GC_170 = Coupling(name = 'GC_170',
                  value = '-((cw*ee*complex(0,1)*Rl2x2*complexconjugate(NN4x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl2x2*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl2x2*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_171 = Coupling(name = 'GC_171',
                  value = '-((cw*ee*complex(0,1)*Rl3x3*complexconjugate(NN4x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*Rl3x3*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rl3x3*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_172 = Coupling(name = 'GC_172',
                  value = '-((cw*ee*complex(0,1)*Rn1x1*complexconjugate(NN4x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn1x1*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn1x1*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_173 = Coupling(name = 'GC_173',
                  value = '-((cw*ee*complex(0,1)*Rn2x2*complexconjugate(NN4x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn2x2*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn2x2*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_174 = Coupling(name = 'GC_174',
                  value = '-((cw*ee*complex(0,1)*Rn3x3*complexconjugate(NN4x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*Rn3x3*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rn3x3*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_175 = Coupling(name = 'GC_175',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(NN4x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru1x1*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru1x1*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_176 = Coupling(name = 'GC_176',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(NN4x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru2x2*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru2x2*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_177 = Coupling(name = 'GC_177',
                  value = '(cw*ee*complex(0,1)*Ru3x3*complexconjugate(NN4x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Ru3x3*complexconjugate(NN4x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Ru3x3*sw*complexconjugate(NN4x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1778 = Coupling(name = 'GC_1778',
                   value = '(complex(0,1)*yd3x3*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1779 = Coupling(name = 'GC_1779',
                   value = '(complex(0,1)*ye3x3*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_178 = Coupling(name = 'GC_178',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(NN4x2))/sw) - (ee*complex(0,1)*UU1x2*complexconjugate(NN4x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1780 = Coupling(name = 'GC_1780',
                   value = '-((complex(0,1)*yu3x3*cmath.sin(alp))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_1781 = Coupling(name = 'GC_1781',
                   value = '(complex(0,1)*complexconjugate(yd3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1782 = Coupling(name = 'GC_1782',
                   value = '(complex(0,1)*complexconjugate(ye3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1783 = Coupling(name = 'GC_1783',
                   value = '-((complex(0,1)*complexconjugate(yu3x3)*cmath.sin(alp))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_1784 = Coupling(name = 'GC_1784',
                   value = '(ee**2*complex(0,1)*cmath.cos(alp)*cmath.sin(alp))/(2*sw**2 - 2*sw**4)',
                   order = {'QED':2})

GC_1785 = Coupling(name = 'GC_1785',
                   value = '(ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)*cmath.cos(alp)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':2})

GC_1786 = Coupling(name = 'GC_1786',
                   value = '(ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)*cmath.cos(alp)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':2})

GC_1787 = Coupling(name = 'GC_1787',
                   value = '(2*ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4)*cmath.cos(alp)*cmath.sin(alp))/(3 - 3*sw**2)',
                   order = {'QED':2})

GC_1788 = Coupling(name = 'GC_1788',
                   value = '(2*ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5)*cmath.cos(alp)*cmath.sin(alp))/(3 - 3*sw**2)',
                   order = {'QED':2})

GC_1789 = Coupling(name = 'GC_1789',
                   value = '(ee*complex(0,1)*NN1x2*NN1x4*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*NN1x1*NN1x4*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN1x2*NN1x3*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN1x1*NN1x3*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_179 = Coupling(name = 'GC_179',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(NN4x2))/sw) - (ee*complex(0,1)*UU2x2*complexconjugate(NN4x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1790 = Coupling(name = 'GC_1790',
                   value = '-((ee*complex(0,1)*NN1x2*NN1x3*cmath.cos(alp))/sw) - (cw*ee*complex(0,1)*NN1x1*NN1x3*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN1x2*NN1x4*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN1x1*NN1x4*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1791 = Coupling(name = 'GC_1791',
                   value = '(ee*complex(0,1)*NN1x4*NN2x2*cmath.cos(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN2x4*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x4*NN2x1*cmath.cos(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN2x4*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*NN1x3*NN2x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN2x3*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x3*NN2x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN2x3*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1792 = Coupling(name = 'GC_1792',
                   value = '(ee*complex(0,1)*NN2x2*NN2x4*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*NN2x1*NN2x4*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN2x2*NN2x3*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN2x1*NN2x3*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1793 = Coupling(name = 'GC_1793',
                   value = '-(ee*complex(0,1)*NN1x3*NN2x2*cmath.cos(alp))/(2.*sw) - (ee*complex(0,1)*NN1x2*NN2x3*cmath.cos(alp))/(2.*sw) - (cw*ee*complex(0,1)*NN1x3*NN2x1*cmath.cos(alp))/(2.*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN1x1*NN2x3*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*NN1x4*NN2x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN2x4*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x4*NN2x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN2x4*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1794 = Coupling(name = 'GC_1794',
                   value = '-((ee*complex(0,1)*NN2x2*NN2x3*cmath.cos(alp))/sw) - (cw*ee*complex(0,1)*NN2x1*NN2x3*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN2x2*NN2x4*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN2x1*NN2x4*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1795 = Coupling(name = 'GC_1795',
                   value = '(ee*complex(0,1)*NN1x4*NN3x2*cmath.cos(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN3x4*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x4*NN3x1*cmath.cos(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN3x4*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*NN1x3*NN3x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN3x3*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x3*NN3x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN3x3*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1796 = Coupling(name = 'GC_1796',
                   value = '(ee*complex(0,1)*NN2x4*NN3x2*cmath.cos(alp))/(2.*sw) + (ee*complex(0,1)*NN2x2*NN3x4*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x4*NN3x1*cmath.cos(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN3x4*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*NN2x3*NN3x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN2x2*NN3x3*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x3*NN3x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN3x3*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1797 = Coupling(name = 'GC_1797',
                   value = '(ee*complex(0,1)*NN3x2*NN3x4*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*NN3x1*NN3x4*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN3x2*NN3x3*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN3x1*NN3x3*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1798 = Coupling(name = 'GC_1798',
                   value = '-(ee*complex(0,1)*NN1x3*NN3x2*cmath.cos(alp))/(2.*sw) - (ee*complex(0,1)*NN1x2*NN3x3*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x3*NN3x1*cmath.cos(alp))/(2.*(1 - sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN3x3*cmath.cos(alp))/(2.*(1 - sw**2)) + (ee*complex(0,1)*NN1x4*NN3x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN3x4*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x4*NN3x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN3x4*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1799 = Coupling(name = 'GC_1799',
                   value = '-(ee*complex(0,1)*NN2x3*NN3x2*cmath.cos(alp))/(2.*sw) - (ee*complex(0,1)*NN2x2*NN3x3*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x3*NN3x1*cmath.cos(alp))/(2.*(1 - sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN3x3*cmath.cos(alp))/(2.*(1 - sw**2)) + (ee*complex(0,1)*NN2x4*NN3x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN2x2*NN3x4*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x4*NN3x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN3x4*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_18 = Coupling(name = 'GC_18',
                 value = '-(complex(0,1)*G*Ru1x1*cmath.sqrt(2))',
                 order = {'QCD':1})

GC_180 = Coupling(name = 'GC_180',
                  value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(NN4x3))/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(NN4x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_1800 = Coupling(name = 'GC_1800',
                   value = '-((ee*complex(0,1)*NN3x2*NN3x3*cmath.cos(alp))/sw) - (cw*ee*complex(0,1)*NN3x1*NN3x3*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN3x2*NN3x4*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN3x1*NN3x4*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1801 = Coupling(name = 'GC_1801',
                   value = '(ee*complex(0,1)*NN1x4*NN4x2*cmath.cos(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN4x4*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x4*NN4x1*cmath.cos(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN4x4*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*NN1x3*NN4x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN4x3*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x3*NN4x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN4x3*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1802 = Coupling(name = 'GC_1802',
                   value = '(ee*complex(0,1)*NN2x4*NN4x2*cmath.cos(alp))/(2.*sw) + (ee*complex(0,1)*NN2x2*NN4x4*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x4*NN4x1*cmath.cos(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN4x4*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*NN2x3*NN4x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN2x2*NN4x3*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x3*NN4x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN4x3*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1803 = Coupling(name = 'GC_1803',
                   value = '(ee*complex(0,1)*NN3x4*NN4x2*cmath.cos(alp))/(2.*sw) + (ee*complex(0,1)*NN3x2*NN4x4*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN3x4*NN4x1*cmath.cos(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN3x1*NN4x4*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*NN3x3*NN4x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN3x2*NN4x3*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN3x3*NN4x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN3x1*NN4x3*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1804 = Coupling(name = 'GC_1804',
                   value = '(ee*complex(0,1)*NN4x2*NN4x4*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*NN4x1*NN4x4*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN4x2*NN4x3*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN4x1*NN4x3*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1805 = Coupling(name = 'GC_1805',
                   value = '-(ee*complex(0,1)*NN1x3*NN4x2*cmath.cos(alp))/(2.*sw) - (ee*complex(0,1)*NN1x2*NN4x3*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x3*NN4x1*cmath.cos(alp))/(2.*(1 - sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN4x3*cmath.cos(alp))/(2.*(1 - sw**2)) + (ee*complex(0,1)*NN1x4*NN4x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN1x2*NN4x4*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN1x4*NN4x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x1*NN4x4*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1806 = Coupling(name = 'GC_1806',
                   value = '-(ee*complex(0,1)*NN2x3*NN4x2*cmath.cos(alp))/(2.*sw) - (ee*complex(0,1)*NN2x2*NN4x3*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x3*NN4x1*cmath.cos(alp))/(2.*(1 - sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN4x3*cmath.cos(alp))/(2.*(1 - sw**2)) + (ee*complex(0,1)*NN2x4*NN4x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN2x2*NN4x4*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN2x4*NN4x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x1*NN4x4*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1807 = Coupling(name = 'GC_1807',
                   value = '-(ee*complex(0,1)*NN3x3*NN4x2*cmath.cos(alp))/(2.*sw) - (ee*complex(0,1)*NN3x2*NN4x3*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN3x3*NN4x1*cmath.cos(alp))/(2.*(1 - sw**2)) + (cw*ee*complex(0,1)*NN3x1*NN4x3*cmath.cos(alp))/(2.*(1 - sw**2)) + (ee*complex(0,1)*NN3x4*NN4x2*cmath.sin(alp))/(2.*sw) + (ee*complex(0,1)*NN3x2*NN4x4*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*NN3x4*NN4x1*cmath.sin(alp))/(2.*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN3x1*NN4x4*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1808 = Coupling(name = 'GC_1808',
                   value = '-((ee*complex(0,1)*NN4x2*NN4x3*cmath.cos(alp))/sw) - (cw*ee*complex(0,1)*NN4x1*NN4x3*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*NN4x2*NN4x4*cmath.sin(alp))/sw + (cw*ee*complex(0,1)*NN4x1*NN4x4*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1809 = Coupling(name = 'GC_1809',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*sw**2) + (ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*sw**2)',
                   order = {'QED':1})

GC_181 = Coupling(name = 'GC_181',
                  value = '(complex(0,1)*Rl6x6*ye3x3*complexconjugate(NN4x3))/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(NN4x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_1810 = Coupling(name = 'GC_1810',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(alp))/(2.*sw**2) - (ee**2*complex(0,1)*vd*cmath.sin(alp))/(2.*sw**2)',
                   order = {'QED':1})

GC_1811 = Coupling(name = 'GC_1811',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/2. - (cw**2*ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*sw**2) - (ee**2*complex(0,1)*sw**2*vu*cmath.cos(alp))/(4.*cw**2) + (ee**2*complex(0,1)*vd*cmath.sin(alp))/2. + (cw**2*ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*sw**2) + (ee**2*complex(0,1)*sw**2*vd*cmath.sin(alp))/(4.*cw**2)',
                   order = {'QED':1})

GC_1812 = Coupling(name = 'GC_1812',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1813 = Coupling(name = 'GC_1813',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/(2.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.sin(alp))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1814 = Coupling(name = 'GC_1814',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*sw**2) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*sw**2)',
                   order = {'QED':1})

GC_1815 = Coupling(name = 'GC_1815',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(alp))/(2.*sw**2) + (ee**2*complex(0,1)*vu*cmath.sin(alp))/(2.*sw**2)',
                   order = {'QED':1})

GC_1816 = Coupling(name = 'GC_1816',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp))/2. - (cw**2*ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*sw**2) - (ee**2*complex(0,1)*sw**2*vd*cmath.cos(alp))/(4.*cw**2) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/2. - (cw**2*ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*sw**2) - (ee**2*complex(0,1)*sw**2*vu*cmath.sin(alp))/(4.*cw**2)',
                   order = {'QED':1})

GC_1817 = Coupling(name = 'GC_1817',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1818 = Coupling(name = 'GC_1818',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp))/(2.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1819 = Coupling(name = 'GC_1819',
                   value = '-((ee*complex(0,1)*UU1x1*VV1x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU1x2*VV1x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_182 = Coupling(name = 'GC_182',
                  value = '-(cw*ee*complex(0,1)*NN1x3*complexconjugate(NN4x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN1x4*complexconjugate(NN4x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1820 = Coupling(name = 'GC_1820',
                   value = '-((ee*complex(0,1)*UU2x1*VV1x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU2x2*VV1x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1821 = Coupling(name = 'GC_1821',
                   value = '-((ee*complex(0,1)*UU1x2*VV1x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU1x1*VV1x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1822 = Coupling(name = 'GC_1822',
                   value = '-((ee*complex(0,1)*UU2x2*VV1x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU2x1*VV1x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1823 = Coupling(name = 'GC_1823',
                   value = '-((ee*complex(0,1)*UU1x1*VV2x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU1x2*VV2x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1824 = Coupling(name = 'GC_1824',
                   value = '-((ee*complex(0,1)*UU2x1*VV2x2*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*UU2x2*VV2x1*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1825 = Coupling(name = 'GC_1825',
                   value = '-((ee*complex(0,1)*UU1x2*VV2x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU1x1*VV2x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1826 = Coupling(name = 'GC_1826',
                   value = '-((ee*complex(0,1)*UU2x2*VV2x1*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*UU2x1*VV2x2*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1827 = Coupling(name = 'GC_1827',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x4)*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x3)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1828 = Coupling(name = 'GC_1828',
                   value = '-((cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x3)*cmath.cos(alp))/(-1 + sw**2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN1x4)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1829 = Coupling(name = 'GC_1829',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_183 = Coupling(name = 'GC_183',
                  value = '-(cw*ee*complex(0,1)*NN2x3*complexconjugate(NN4x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN2x4*complexconjugate(NN4x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1830 = Coupling(name = 'GC_1830',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x4)*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x3)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1831 = Coupling(name = 'GC_1831',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x1)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.cos(alp))/(2.*sw) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x3)*cmath.cos(alp))/(2.*(-1 + sw**2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN2x4)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1832 = Coupling(name = 'GC_1832',
                   value = '-((cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x3)*cmath.cos(alp))/(-1 + sw**2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN2x4)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1833 = Coupling(name = 'GC_1833',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1834 = Coupling(name = 'GC_1834',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1835 = Coupling(name = 'GC_1835',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x4)*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x3)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1836 = Coupling(name = 'GC_1836',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1837 = Coupling(name = 'GC_1837',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x1)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1838 = Coupling(name = 'GC_1838',
                   value = '-((cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x3)*cmath.cos(alp))/(-1 + sw**2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN3x4)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1839 = Coupling(name = 'GC_1839',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_184 = Coupling(name = 'GC_184',
                  value = '-(cw*ee*complex(0,1)*NN3x3*complexconjugate(NN4x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*NN3x4*complexconjugate(NN4x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1840 = Coupling(name = 'GC_1840',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1841 = Coupling(name = 'GC_1841',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1842 = Coupling(name = 'GC_1842',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x4)*cmath.cos(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x3)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1843 = Coupling(name = 'GC_1843',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1844 = Coupling(name = 'GC_1844',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1845 = Coupling(name = 'GC_1845',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x1)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*(1 - sw**2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.cos(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.sin(alp))/(2.*sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.sin(alp))/(2.*sw)',
                   order = {'QED':1})

GC_1846 = Coupling(name = 'GC_1846',
                   value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x3)*cmath.cos(alp))/(1 - sw**2) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.cos(alp))/sw + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(NN4x4)*cmath.sin(alp))/(-1 + sw**2) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.sin(alp))/sw',
                   order = {'QED':1})

GC_1847 = Coupling(name = 'GC_1847',
                   value = '-(ee**2*complex(0,1)*Rd1x1*vu*complexconjugate(Rd1x1)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*vu*complexconjugate(Rd1x1)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*vd*complexconjugate(Rd1x1)*cmath.sin(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*vd*complexconjugate(Rd1x1)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1848 = Coupling(name = 'GC_1848',
                   value = '(ee**2*complex(0,1)*Rd1x1*vd*complexconjugate(Rd1x1)*cmath.cos(alp))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*vd*complexconjugate(Rd1x1)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*vu*complexconjugate(Rd1x1)*cmath.sin(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*vu*complexconjugate(Rd1x1)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1849 = Coupling(name = 'GC_1849',
                   value = '-(ee**2*complex(0,1)*Rd2x2*vu*complexconjugate(Rd2x2)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*vu*complexconjugate(Rd2x2)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*vd*complexconjugate(Rd2x2)*cmath.sin(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*vd*complexconjugate(Rd2x2)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_185 = Coupling(name = 'GC_185',
                  value = '(cw*ee*complex(0,1)*NN4x3*complexconjugate(NN4x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*NN4x4*complexconjugate(NN4x4))/(2.*sw*(-1 + sw**2))',
                  order = {'QED':1})

GC_1850 = Coupling(name = 'GC_1850',
                   value = '(ee**2*complex(0,1)*Rd2x2*vd*complexconjugate(Rd2x2)*cmath.cos(alp))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*vd*complexconjugate(Rd2x2)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*vu*complexconjugate(Rd2x2)*cmath.sin(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*vu*complexconjugate(Rd2x2)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1851 = Coupling(name = 'GC_1851',
                   value = '(complex(0,1)*Rd6x6*yd3x3*complexconjugate(MUH)*complexconjugate(Rd3x3)*cmath.cos(alp))/cmath.sqrt(2) + (complex(0,1)*Rd6x6*td3x3*complexconjugate(Rd3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1852 = Coupling(name = 'GC_1852',
                   value = '(ee**2*complex(0,1)*Rd3x3*vd*complexconjugate(Rd3x3)*cmath.cos(alp))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*vd*complexconjugate(Rd3x3)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Rd3x3*vd*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(alp))/(-1 + sw**2) - (complex(0,1)*Rd3x3*sw**2*vd*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(alp))/(-1 + sw**2) - (ee**2*complex(0,1)*Rd3x3*vu*complexconjugate(Rd3x3)*cmath.sin(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*vu*complexconjugate(Rd3x3)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1853 = Coupling(name = 'GC_1853',
                   value = '-((complex(0,1)*Rd6x6*td3x3*complexconjugate(Rd3x3)*cmath.cos(alp))/cmath.sqrt(2)) + (complex(0,1)*Rd6x6*yd3x3*complexconjugate(MUH)*complexconjugate(Rd3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1854 = Coupling(name = 'GC_1854',
                   value = '(ee**2*complex(0,1)*Rd4x4*vu*complexconjugate(Rd4x4)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd4x4*vd*complexconjugate(Rd4x4)*cmath.sin(alp))/(6.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1855 = Coupling(name = 'GC_1855',
                   value = '-(ee**2*complex(0,1)*Rd4x4*vd*complexconjugate(Rd4x4)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd4x4*vu*complexconjugate(Rd4x4)*cmath.sin(alp))/(6.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1856 = Coupling(name = 'GC_1856',
                   value = '(ee**2*complex(0,1)*Rd5x5*vu*complexconjugate(Rd5x5)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd5x5*vd*complexconjugate(Rd5x5)*cmath.sin(alp))/(6.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1857 = Coupling(name = 'GC_1857',
                   value = '-(ee**2*complex(0,1)*Rd5x5*vd*complexconjugate(Rd5x5)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd5x5*vu*complexconjugate(Rd5x5)*cmath.sin(alp))/(6.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1858 = Coupling(name = 'GC_1858',
                   value = '-(ee**2*complex(0,1)*Rd6x6*vd*complexconjugate(Rd6x6)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (complex(0,1)*Rd6x6*vd*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(alp))/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*vd*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(alp))/(-1 + sw**2) + (ee**2*complex(0,1)*Rd6x6*vu*complexconjugate(Rd6x6)*cmath.sin(alp))/(6.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1859 = Coupling(name = 'GC_1859',
                   value = '-(ee**2*complex(0,1)*Rl1x1*vu*complexconjugate(Rl1x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*vu*complexconjugate(Rl1x1)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*vd*complexconjugate(Rl1x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*vd*complexconjugate(Rl1x1)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_186 = Coupling(name = 'GC_186',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(NN4x2))/sw) + (ee*complex(0,1)*VV1x2*complexconjugate(NN4x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1860 = Coupling(name = 'GC_1860',
                   value = '(ee**2*complex(0,1)*Rl1x1*vd*complexconjugate(Rl1x1)*cmath.cos(alp))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*vd*complexconjugate(Rl1x1)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*vu*complexconjugate(Rl1x1)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*vu*complexconjugate(Rl1x1)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1861 = Coupling(name = 'GC_1861',
                   value = '-(ee**2*complex(0,1)*Rl2x2*vu*complexconjugate(Rl2x2)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*vu*complexconjugate(Rl2x2)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*vd*complexconjugate(Rl2x2)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*vd*complexconjugate(Rl2x2)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1862 = Coupling(name = 'GC_1862',
                   value = '(ee**2*complex(0,1)*Rl2x2*vd*complexconjugate(Rl2x2)*cmath.cos(alp))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*vd*complexconjugate(Rl2x2)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*vu*complexconjugate(Rl2x2)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*vu*complexconjugate(Rl2x2)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1863 = Coupling(name = 'GC_1863',
                   value = '(complex(0,1)*Rl6x6*ye3x3*complexconjugate(MUH)*complexconjugate(Rl3x3)*cmath.cos(alp))/cmath.sqrt(2) + (complex(0,1)*Rl6x6*te3x3*complexconjugate(Rl3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1864 = Coupling(name = 'GC_1864',
                   value = '(ee**2*complex(0,1)*Rl3x3*vd*complexconjugate(Rl3x3)*cmath.cos(alp))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*vd*complexconjugate(Rl3x3)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Rl3x3*vd*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(alp))/(-1 + sw**2) - (complex(0,1)*Rl3x3*sw**2*vd*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(alp))/(-1 + sw**2) - (ee**2*complex(0,1)*Rl3x3*vu*complexconjugate(Rl3x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*vu*complexconjugate(Rl3x3)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1865 = Coupling(name = 'GC_1865',
                   value = '-((complex(0,1)*Rl6x6*te3x3*complexconjugate(Rl3x3)*cmath.cos(alp))/cmath.sqrt(2)) + (complex(0,1)*Rl6x6*ye3x3*complexconjugate(MUH)*complexconjugate(Rl3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1866 = Coupling(name = 'GC_1866',
                   value = '(ee**2*complex(0,1)*Rl4x4*vu*complexconjugate(Rl4x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl4x4*vd*complexconjugate(Rl4x4)*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1867 = Coupling(name = 'GC_1867',
                   value = '-(ee**2*complex(0,1)*Rl4x4*vd*complexconjugate(Rl4x4)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl4x4*vu*complexconjugate(Rl4x4)*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1868 = Coupling(name = 'GC_1868',
                   value = '(ee**2*complex(0,1)*Rl5x5*vu*complexconjugate(Rl5x5)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl5x5*vd*complexconjugate(Rl5x5)*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1869 = Coupling(name = 'GC_1869',
                   value = '-(ee**2*complex(0,1)*Rl5x5*vd*complexconjugate(Rl5x5)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl5x5*vu*complexconjugate(Rl5x5)*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_187 = Coupling(name = 'GC_187',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(NN4x2))/sw) + (ee*complex(0,1)*VV2x2*complexconjugate(NN4x4))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1870 = Coupling(name = 'GC_1870',
                   value = '-(ee**2*complex(0,1)*Rl6x6*vd*complexconjugate(Rl6x6)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl6x6*vd*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(alp))/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*vd*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(alp))/(-1 + sw**2) + (ee**2*complex(0,1)*Rl6x6*vu*complexconjugate(Rl6x6)*cmath.sin(alp))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1871 = Coupling(name = 'GC_1871',
                   value = '(ee**2*complex(0,1)*Ru1x1*vu*complexconjugate(Ru1x1)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*vu*complexconjugate(Ru1x1)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*vd*complexconjugate(Ru1x1)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*vd*complexconjugate(Ru1x1)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1872 = Coupling(name = 'GC_1872',
                   value = '-(ee**2*complex(0,1)*Ru1x1*vd*complexconjugate(Ru1x1)*cmath.cos(alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*vd*complexconjugate(Ru1x1)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*vu*complexconjugate(Ru1x1)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*vu*complexconjugate(Ru1x1)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1873 = Coupling(name = 'GC_1873',
                   value = '(ee**2*complex(0,1)*Ru2x2*vu*complexconjugate(Ru2x2)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*vu*complexconjugate(Ru2x2)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*vd*complexconjugate(Ru2x2)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*vd*complexconjugate(Ru2x2)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1874 = Coupling(name = 'GC_1874',
                   value = '-(ee**2*complex(0,1)*Ru2x2*vd*complexconjugate(Ru2x2)*cmath.cos(alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*vd*complexconjugate(Ru2x2)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*vu*complexconjugate(Ru2x2)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*vu*complexconjugate(Ru2x2)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1875 = Coupling(name = 'GC_1875',
                   value = '(complex(0,1)*Ru6x6*yu3x3*complexconjugate(MUH)*complexconjugate(Ru3x3)*cmath.cos(alp))/cmath.sqrt(2) - (complex(0,1)*Ru6x6*tu3x3*complexconjugate(Ru3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1876 = Coupling(name = 'GC_1876',
                   value = '(ee**2*complex(0,1)*Ru3x3*vu*complexconjugate(Ru3x3)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*vu*complexconjugate(Ru3x3)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Ru3x3*vu*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(alp))/(-1 + sw**2) - (complex(0,1)*Ru3x3*sw**2*vu*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(alp))/(-1 + sw**2) + (ee**2*complex(0,1)*Ru3x3*vd*complexconjugate(Ru3x3)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*vd*complexconjugate(Ru3x3)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1877 = Coupling(name = 'GC_1877',
                   value = '-((complex(0,1)*Ru6x6*tu3x3*complexconjugate(Ru3x3)*cmath.cos(alp))/cmath.sqrt(2)) - (complex(0,1)*Ru6x6*yu3x3*complexconjugate(MUH)*complexconjugate(Ru3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1878 = Coupling(name = 'GC_1878',
                   value = '-(ee**2*complex(0,1)*Ru4x4*vu*complexconjugate(Ru4x4)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru4x4*vd*complexconjugate(Ru4x4)*cmath.sin(alp))/(3.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1879 = Coupling(name = 'GC_1879',
                   value = '(ee**2*complex(0,1)*Ru4x4*vd*complexconjugate(Ru4x4)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru4x4*vu*complexconjugate(Ru4x4)*cmath.sin(alp))/(3.*(-1 + sw**2))',
                   order = {'QED':1})

GC_188 = Coupling(name = 'GC_188',
                  value = '(complex(0,1)*Ru6x6*yu3x3*complexconjugate(NN4x4))/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(NN4x4))/(-1 + sw**2)',
                  order = {'QED':1})

GC_1880 = Coupling(name = 'GC_1880',
                   value = '-(ee**2*complex(0,1)*Ru5x5*vu*complexconjugate(Ru5x5)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru5x5*vd*complexconjugate(Ru5x5)*cmath.sin(alp))/(3.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1881 = Coupling(name = 'GC_1881',
                   value = '(ee**2*complex(0,1)*Ru5x5*vd*complexconjugate(Ru5x5)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru5x5*vu*complexconjugate(Ru5x5)*cmath.sin(alp))/(3.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1882 = Coupling(name = 'GC_1882',
                   value = '-(ee**2*complex(0,1)*Ru6x6*vu*complexconjugate(Ru6x6)*cmath.cos(alp))/(3.*(-1 + sw**2)) + (complex(0,1)*Ru6x6*vu*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(alp))/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*vu*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(alp))/(-1 + sw**2) - (ee**2*complex(0,1)*Ru6x6*vd*complexconjugate(Ru6x6)*cmath.sin(alp))/(3.*(-1 + sw**2))',
                   order = {'QED':1})

GC_1883 = Coupling(name = 'GC_1883',
                   value = '(complex(0,1)*MUH*Rd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(alp))/cmath.sqrt(2) + (complex(0,1)*Rd3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1884 = Coupling(name = 'GC_1884',
                   value = '(complex(0,1)*MUH*Rl3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(alp))/cmath.sqrt(2) + (complex(0,1)*Rl3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1885 = Coupling(name = 'GC_1885',
                   value = '(complex(0,1)*MUH*Ru3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(alp))/cmath.sqrt(2) - (complex(0,1)*Ru3x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1886 = Coupling(name = 'GC_1886',
                   value = '-((ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV1x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV1x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1887 = Coupling(name = 'GC_1887',
                   value = '-((ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV1x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV1x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1888 = Coupling(name = 'GC_1888',
                   value = '-((ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV1x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV1x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1889 = Coupling(name = 'GC_1889',
                   value = '-((ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV1x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV1x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_189 = Coupling(name = 'GC_189',
                  value = '-(complex(0,1)*G*complexconjugate(Rd1x1)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_1890 = Coupling(name = 'GC_1890',
                   value = '-((ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV2x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV2x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1891 = Coupling(name = 'GC_1891',
                   value = '-((ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV2x2)*cmath.cos(alp))/(sw*cmath.sqrt(2))) + (ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV2x1)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1892 = Coupling(name = 'GC_1892',
                   value = '-((ee*complex(0,1)*complexconjugate(UU1x2)*complexconjugate(VV2x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU1x1)*complexconjugate(VV2x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1893 = Coupling(name = 'GC_1893',
                   value = '-((ee*complex(0,1)*complexconjugate(UU2x2)*complexconjugate(VV2x1)*cmath.cos(alp))/(sw*cmath.sqrt(2))) - (ee*complex(0,1)*complexconjugate(UU2x1)*complexconjugate(VV2x2)*cmath.sin(alp))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_1894 = Coupling(name = 'GC_1894',
                   value = '-(ee**2*complex(0,1)*Rd3x3*vu*complexconjugate(Rd3x3)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*vu*complexconjugate(Rd3x3)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*vd*complexconjugate(Rd3x3)*cmath.sin(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*vd*complexconjugate(Rd3x3)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2)) - (complex(0,1)*Rd3x3*vd*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(alp))/(-1 + sw**2) + (complex(0,1)*Rd3x3*sw**2*vd*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1895 = Coupling(name = 'GC_1895',
                   value = '-((complex(0,1)*Rd3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)*cmath.cos(alp))/cmath.sqrt(2)) + (complex(0,1)*MUH*Rd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1896 = Coupling(name = 'GC_1896',
                   value = '(ee**2*complex(0,1)*Rd6x6*vu*complexconjugate(Rd6x6)*cmath.cos(alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd6x6*vd*complexconjugate(Rd6x6)*cmath.sin(alp))/(6.*(-1 + sw**2)) - (complex(0,1)*Rd6x6*vd*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(alp))/(-1 + sw**2) + (complex(0,1)*Rd6x6*sw**2*vd*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1897 = Coupling(name = 'GC_1897',
                   value = '-(ee**2*complex(0,1)*Rl3x3*vu*complexconjugate(Rl3x3)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*vu*complexconjugate(Rl3x3)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*vd*complexconjugate(Rl3x3)*cmath.sin(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*vd*complexconjugate(Rl3x3)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2)) - (complex(0,1)*Rl3x3*vd*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(alp))/(-1 + sw**2) + (complex(0,1)*Rl3x3*sw**2*vd*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1898 = Coupling(name = 'GC_1898',
                   value = '-((complex(0,1)*Rl3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)*cmath.cos(alp))/cmath.sqrt(2)) + (complex(0,1)*MUH*Rl3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1899 = Coupling(name = 'GC_1899',
                   value = '(ee**2*complex(0,1)*Rl6x6*vu*complexconjugate(Rl6x6)*cmath.cos(alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl6x6*vd*complexconjugate(Rl6x6)*cmath.sin(alp))/(2.*(-1 + sw**2)) - (complex(0,1)*Rl6x6*vd*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(alp))/(-1 + sw**2) + (complex(0,1)*Rl6x6*sw**2*vd*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_19 = Coupling(name = 'GC_19',
                 value = '-(complex(0,1)*G*Ru2x2*cmath.sqrt(2))',
                 order = {'QCD':1})

GC_190 = Coupling(name = 'GC_190',
                  value = '(ee*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/3.',
                  order = {'QED':1})

GC_1900 = Coupling(name = 'GC_1900',
                   value = '-(ee**2*complex(0,1)*Ru3x3*vd*complexconjugate(Ru3x3)*cmath.cos(alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3*vd*complexconjugate(Ru3x3)*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3*vu*complexconjugate(Ru3x3)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*vu*complexconjugate(Ru3x3)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Ru3x3*vu*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(alp))/(-1 + sw**2) - (complex(0,1)*Ru3x3*sw**2*vu*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1901 = Coupling(name = 'GC_1901',
                   value = '-((complex(0,1)*Ru3x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)*cmath.cos(alp))/cmath.sqrt(2)) - (complex(0,1)*MUH*Ru3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(alp))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1902 = Coupling(name = 'GC_1902',
                   value = '(ee**2*complex(0,1)*Ru6x6*vd*complexconjugate(Ru6x6)*cmath.cos(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru6x6*vu*complexconjugate(Ru6x6)*cmath.sin(alp))/(3.*(-1 + sw**2)) + (complex(0,1)*Ru6x6*vu*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(alp))/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*vu*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(alp))/(-1 + sw**2)',
                   order = {'QED':1})

GC_1903 = Coupling(name = 'GC_1903',
                   value = '-(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(alp)*cmath.sin(alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(alp)*cmath.sin(alp))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1904 = Coupling(name = 'GC_1904',
                   value = '-(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(alp)*cmath.sin(alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(alp)*cmath.sin(alp))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1905 = Coupling(name = 'GC_1905',
                   value = '(2*ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(alp)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(alp)*cmath.sin(alp))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1906 = Coupling(name = 'GC_1906',
                   value = '(2*ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(alp)*cmath.sin(alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(alp)*cmath.sin(alp))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1907 = Coupling(name = 'GC_1907',
                   value = '(3*ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2)) + (3*ee**2*complex(0,1)*vd*cmath.cos(2*alp)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1908 = Coupling(name = 'GC_1908',
                   value = '(3*ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2)) - (3*ee**2*complex(0,1)*vu*cmath.cos(2*alp)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1909 = Coupling(name = 'GC_1909',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.cos(2*beta)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_191 = Coupling(name = 'GC_191',
                  value = '(2*ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/9.',
                  order = {'QED':2})

GC_1910 = Coupling(name = 'GC_1910',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.cos(2*beta)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1911 = Coupling(name = 'GC_1911',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.cos(2*beta)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1912 = Coupling(name = 'GC_1912',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.cos(2*beta)*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_192 = Coupling(name = 'GC_192',
                  value = '-(complex(0,1)*G*Rd1x1*complexconjugate(Rd1x1))',
                  order = {'QCD':1})

GC_193 = Coupling(name = 'GC_193',
                  value = 'complex(0,1)*G*Rd1x1*complexconjugate(Rd1x1)',
                  order = {'QCD':1})

GC_194 = Coupling(name = 'GC_194',
                  value = '(-2*ee*complex(0,1)*G*Rd1x1*complexconjugate(Rd1x1))/3.',
                  order = {'QCD':1,'QED':1})

GC_1949 = Coupling(name = 'GC_1949',
                   value = '-(ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rd3x3*sw**2*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(alp)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_195 = Coupling(name = 'GC_195',
                  value = 'complex(0,1)*G**2*Rd1x1*complexconjugate(Rd1x1)',
                  order = {'QCD':2})

GC_1950 = Coupling(name = 'GC_1950',
                   value = '(ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6)*cmath.cos(2*alp))/(6.*(-1 + sw**2)) + (complex(0,1)*Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(alp)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_1951 = Coupling(name = 'GC_1951',
                   value = '-(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rl3x3*sw**2*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(alp)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_1952 = Coupling(name = 'GC_1952',
                   value = '(ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)*cmath.cos(2*alp))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(alp)**2)/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(alp)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_1953 = Coupling(name = 'GC_1953',
                   value = '-(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*alp))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(alp)**2)/(-1 + sw**2) - (complex(0,1)*Ru3x3*sw**2*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(alp)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_1954 = Coupling(name = 'GC_1954',
                   value = '(ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6)*cmath.cos(2*alp))/(3.*(-1 + sw**2)) + (complex(0,1)*Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(alp)**2)/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(alp)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_1957 = Coupling(name = 'GC_1957',
                   value = '(ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4)*cmath.sin(2*alp))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1958 = Coupling(name = 'GC_1958',
                   value = '(ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5)*cmath.sin(2*alp))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1959 = Coupling(name = 'GC_1959',
                   value = '-(ee**2*complex(0,1)*cmath.cos(2*beta)*cmath.sin(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_196 = Coupling(name = 'GC_196',
                  value = '(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_1960 = Coupling(name = 'GC_1960',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*beta)*cmath.sin(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1961 = Coupling(name = 'GC_1961',
                   value = '-(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.sin(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1962 = Coupling(name = 'GC_1962',
                   value = '-(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.sin(2*alp))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1963 = Coupling(name = 'GC_1963',
                   value = '-(ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.sin(2*alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.sin(2*alp))/(4.*sw**2*(-1 + sw**2)) - (complex(0,1)*Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) + (complex(0,1)*Rd3x3*sw**2*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1964 = Coupling(name = 'GC_1964',
                   value = '(ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6)*cmath.sin(2*alp))/(6.*(-1 + sw**2)) - (complex(0,1)*Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) + (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1965 = Coupling(name = 'GC_1965',
                   value = '-(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.sin(2*alp))/(4.*sw**2*(-1 + sw**2)) - (complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl3x3*sw**2*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1966 = Coupling(name = 'GC_1966',
                   value = '(ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) - (complex(0,1)*Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1967 = Coupling(name = 'GC_1967',
                   value = '(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.sin(2*alp))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.sin(2*alp))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) - (complex(0,1)*Ru3x3*sw**2*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1968 = Coupling(name = 'GC_1968',
                   value = '-(ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6)*cmath.sin(2*alp))/(3.*(-1 + sw**2)) + (complex(0,1)*Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2)) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(2*alp))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_197 = Coupling(name = 'GC_197',
                  value = '-((CKM1x1*ee*complex(0,1)*Ru1x1*complexconjugate(Rd1x1))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_1970 = Coupling(name = 'GC_1970',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(alp))/(8.*sw**2*(-1 + sw**2)) - (3*ee**2*complex(0,1)*vu*cmath.cos(3*alp))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp))/(8.*sw**2*(-1 + sw**2)) - (3*ee**2*complex(0,1)*vd*cmath.sin(3*alp))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1971 = Coupling(name = 'GC_1971',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(alp))/(8.*sw**2*(-1 + sw**2)) - (3*ee**2*complex(0,1)*vd*cmath.cos(3*alp))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp))/(8.*sw**2*(-1 + sw**2)) + (3*ee**2*complex(0,1)*vu*cmath.sin(3*alp))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_1972 = Coupling(name = 'GC_1972',
                   value = '(-3*ee**2*complex(0,1)*cmath.sin(4*alp))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1973 = Coupling(name = 'GC_1973',
                   value = '(3*ee**2*complex(0,1)*cmath.sin(4*alp))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_1974 = Coupling(name = 'GC_1974',
                   value = '-(ee**2*cmath.sin(alp - beta))/(4.*sw**2)',
                   order = {'QED':2})

GC_1975 = Coupling(name = 'GC_1975',
                   value = '(ee**2*cmath.sin(alp - beta))/(4.*sw**2)',
                   order = {'QED':2})

GC_1976 = Coupling(name = 'GC_1976',
                   value = '-(ee*complex(0,1)*cmath.sin(alp - beta))/(2.*sw)',
                   order = {'QED':1})

GC_1977 = Coupling(name = 'GC_1977',
                   value = '(ee*complex(0,1)*cmath.sin(alp - beta))/(2.*sw)',
                   order = {'QED':1})

GC_1978 = Coupling(name = 'GC_1978',
                   value = '-(ee**2*complex(0,1)*cmath.sin(alp - beta))/(2.*sw)',
                   order = {'QED':2})

GC_1979 = Coupling(name = 'GC_1979',
                   value = '(ee**2*complex(0,1)*cmath.sin(alp - beta))/(2.*sw)',
                   order = {'QED':2})

GC_198 = Coupling(name = 'GC_198',
                  value = '(CKM1x1*ee*complex(0,1)*Ru1x1*complexconjugate(Rd1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_1980 = Coupling(name = 'GC_1980',
                   value = '-(cw*ee**2*complex(0,1)*cmath.sin(alp - beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1981 = Coupling(name = 'GC_1981',
                   value = '(cw*ee**2*complex(0,1)*cmath.sin(alp - beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_1982 = Coupling(name = 'GC_1982',
                   value = '-(cw*ee*cmath.sin(alp - beta))/(2.*sw*(-1 + sw**2))',
                   order = {'QED':1})

GC_1983 = Coupling(name = 'GC_1983',
                   value = '(cw*ee*cmath.sin(alp - beta))/(2.*sw*(-1 + sw**2))',
                   order = {'QED':1})

GC_1984 = Coupling(name = 'GC_1984',
                   value = '-((CKM3x3*complex(0,1)*Ru6x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(alp - beta))/cmath.sqrt(2))',
                   order = {'QED':2})

GC_1985 = Coupling(name = 'GC_1985',
                   value = '-((complex(0,1)*Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(alp - beta))/cmath.sqrt(2))',
                   order = {'QED':2})

GC_1986 = Coupling(name = 'GC_1986',
                   value = '(yd3x3*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1987 = Coupling(name = 'GC_1987',
                   value = 'complex(0,1)*ye3x3*cmath.sin(beta)',
                   order = {'QED':1})

GC_1988 = Coupling(name = 'GC_1988',
                   value = '(ye3x3*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_1989 = Coupling(name = 'GC_1989',
                   value = '(yu3x3*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_199 = Coupling(name = 'GC_199',
                  value = '(CKM1x1*ee**2*complex(0,1)*Ru1x1*complexconjugate(Rd1x1))/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_1990 = Coupling(name = 'GC_1990',
                   value = 'CKM3x3*complex(0,1)*yu3x3*cmath.sin(beta)',
                   order = {'QED':1})

GC_1991 = Coupling(name = 'GC_1991',
                   value = 'complex(0,1)*yd3x3*complexconjugate(CKM3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_1992 = Coupling(name = 'GC_1992',
                   value = '-((complexconjugate(yd3x3)*cmath.sin(beta))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_1993 = Coupling(name = 'GC_1993',
                   value = 'CKM3x3*complex(0,1)*complexconjugate(yd3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_1994 = Coupling(name = 'GC_1994',
                   value = 'complex(0,1)*complexconjugate(ye3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_1995 = Coupling(name = 'GC_1995',
                   value = '-((complexconjugate(ye3x3)*cmath.sin(beta))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_1996 = Coupling(name = 'GC_1996',
                   value = '-((complexconjugate(yu3x3)*cmath.sin(beta))/cmath.sqrt(2))',
                   order = {'QED':1})

GC_1997 = Coupling(name = 'GC_1997',
                   value = 'complex(0,1)*complexconjugate(CKM3x3)*complexconjugate(yu3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_1998 = Coupling(name = 'GC_1998',
                   value = '(ee**2*complex(0,1)*cmath.cos(beta)*cmath.sin(beta))/(2*sw**2 - 2*sw**4)',
                   order = {'QED':2})

GC_1999 = Coupling(name = 'GC_1999',
                   value = '-((CKM1x1*ee**2*Ru1x1*complexconjugate(Rd1x1)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2)))',
                   order = {'QED':2})

GC_2 = Coupling(name = 'GC_2',
                value = '(2*ee*complex(0,1))/3.',
                order = {'QED':1})

GC_20 = Coupling(name = 'GC_20',
                 value = '-(complex(0,1)*G*Ru3x3*cmath.sqrt(2))',
                 order = {'QCD':1})

GC_200 = Coupling(name = 'GC_200',
                  value = '(CKM1x1*ee*complex(0,1)*G*Ru1x1*complexconjugate(Rd1x1)*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_2000 = Coupling(name = 'GC_2000',
                   value = '-((CKM2x2*ee**2*Ru2x2*complexconjugate(Rd2x2)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2)))',
                   order = {'QED':2})

GC_2001 = Coupling(name = 'GC_2001',
                   value = '-((ee**2*Rn1x1*complexconjugate(Rl1x1)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2)))',
                   order = {'QED':2})

GC_2002 = Coupling(name = 'GC_2002',
                   value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.cos(beta)*cmath.sin(beta))/(2*sw**2 - 2*sw**4)',
                   order = {'QED':2})

GC_2003 = Coupling(name = 'GC_2003',
                   value = '-((ee**2*Rn2x2*complexconjugate(Rl2x2)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2)))',
                   order = {'QED':2})

GC_2004 = Coupling(name = 'GC_2004',
                   value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.cos(beta)*cmath.sin(beta))/(2*sw**2 - 2*sw**4)',
                   order = {'QED':2})

GC_2005 = Coupling(name = 'GC_2005',
                   value = '(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(beta)*cmath.sin(beta))/(2*sw**2 - 2*sw**4)',
                   order = {'QED':2})

GC_2006 = Coupling(name = 'GC_2006',
                   value = '(ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)*cmath.cos(beta)*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':2})

GC_2007 = Coupling(name = 'GC_2007',
                   value = '(ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)*cmath.cos(beta)*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':2})

GC_2008 = Coupling(name = 'GC_2008',
                   value = '(ee**2*Rl1x1*complexconjugate(Rn1x1)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2009 = Coupling(name = 'GC_2009',
                   value = '(ee**2*Rl2x2*complexconjugate(Rn2x2)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_201 = Coupling(name = 'GC_201',
                  value = '(CKM1x1*cw*ee**2*complex(0,1)*Ru1x1*complexconjugate(Rd1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_2010 = Coupling(name = 'GC_2010',
                   value = '(ee**2*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2011 = Coupling(name = 'GC_2011',
                   value = '(ee**2*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2012 = Coupling(name = 'GC_2012',
                   value = '(2*ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4)*cmath.cos(beta)*cmath.sin(beta))/(3 - 3*sw**2)',
                   order = {'QED':2})

GC_2013 = Coupling(name = 'GC_2013',
                   value = '(2*ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5)*cmath.cos(beta)*cmath.sin(beta))/(3 - 3*sw**2)',
                   order = {'QED':2})

GC_2014 = Coupling(name = 'GC_2014',
                   value = '(ee**2*complex(0,1)*cmath.cos(alp)*cmath.cos(beta)*cmath.sin(alp)*cmath.sin(beta))/(sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2015 = Coupling(name = 'GC_2015',
                   value = '(2*ee**2*complex(0,1)*cmath.cos(beta)**2*cmath.sin(beta)**2)/(sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2016 = Coupling(name = 'GC_2016',
                   value = '-(ee*NN1x3*NN2x2*cmath.cos(beta))/(2.*sw) - (ee*NN1x2*NN2x3*cmath.cos(beta))/(2.*sw) + (cw*ee*NN1x3*NN2x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN1x1*NN2x3*cmath.cos(beta))/(2.*(1 - sw**2)) - (ee*NN1x4*NN2x2*cmath.sin(beta))/(2.*sw) - (ee*NN1x2*NN2x4*cmath.sin(beta))/(2.*sw) + (cw*ee*NN1x4*NN2x1*cmath.sin(beta))/(2.*(1 - sw**2)) + (cw*ee*NN1x1*NN2x4*cmath.sin(beta))/(2.*(1 - sw**2))',
                   order = {'QED':1})

GC_2017 = Coupling(name = 'GC_2017',
                   value = '-(ee*NN2x3*NN3x2*cmath.cos(beta))/(2.*sw) - (ee*NN2x2*NN3x3*cmath.cos(beta))/(2.*sw) + (cw*ee*NN2x3*NN3x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN2x1*NN3x3*cmath.cos(beta))/(2.*(1 - sw**2)) - (ee*NN2x4*NN3x2*cmath.sin(beta))/(2.*sw) - (ee*NN2x2*NN3x4*cmath.sin(beta))/(2.*sw) + (cw*ee*NN2x4*NN3x1*cmath.sin(beta))/(2.*(1 - sw**2)) + (cw*ee*NN2x1*NN3x4*cmath.sin(beta))/(2.*(1 - sw**2))',
                   order = {'QED':1})

GC_2018 = Coupling(name = 'GC_2018',
                   value = '-(ee*NN1x3*NN4x2*cmath.cos(beta))/(2.*sw) - (ee*NN1x2*NN4x3*cmath.cos(beta))/(2.*sw) + (cw*ee*NN1x3*NN4x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN1x1*NN4x3*cmath.cos(beta))/(2.*(1 - sw**2)) - (ee*NN1x4*NN4x2*cmath.sin(beta))/(2.*sw) - (ee*NN1x2*NN4x4*cmath.sin(beta))/(2.*sw) + (cw*ee*NN1x4*NN4x1*cmath.sin(beta))/(2.*(1 - sw**2)) + (cw*ee*NN1x1*NN4x4*cmath.sin(beta))/(2.*(1 - sw**2))',
                   order = {'QED':1})

GC_2019 = Coupling(name = 'GC_2019',
                   value = '-(ee*NN2x3*NN4x2*cmath.cos(beta))/(2.*sw) - (ee*NN2x2*NN4x3*cmath.cos(beta))/(2.*sw) + (cw*ee*NN2x3*NN4x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN2x1*NN4x3*cmath.cos(beta))/(2.*(1 - sw**2)) - (ee*NN2x4*NN4x2*cmath.sin(beta))/(2.*sw) - (ee*NN2x2*NN4x4*cmath.sin(beta))/(2.*sw) + (cw*ee*NN2x4*NN4x1*cmath.sin(beta))/(2.*(1 - sw**2)) + (cw*ee*NN2x1*NN4x4*cmath.sin(beta))/(2.*(1 - sw**2))',
                   order = {'QED':1})

GC_202 = Coupling(name = 'GC_202',
                  value = '-((CKM1x1*ee*complex(0,1)*UU1x1*complexconjugate(Rd1x1))/sw)',
                  order = {'QED':1})

GC_2020 = Coupling(name = 'GC_2020',
                   value = '-(ee*NN3x3*NN4x2*cmath.cos(beta))/(2.*sw) - (ee*NN3x2*NN4x3*cmath.cos(beta))/(2.*sw) + (cw*ee*NN3x3*NN4x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN3x1*NN4x3*cmath.cos(beta))/(2.*(1 - sw**2)) - (ee*NN3x4*NN4x2*cmath.sin(beta))/(2.*sw) - (ee*NN3x2*NN4x4*cmath.sin(beta))/(2.*sw) + (cw*ee*NN3x4*NN4x1*cmath.sin(beta))/(2.*(1 - sw**2)) + (cw*ee*NN3x1*NN4x4*cmath.sin(beta))/(2.*(1 - sw**2))',
                   order = {'QED':1})

GC_2021 = Coupling(name = 'GC_2021',
                   value = '-((ee*NN1x2*NN1x4*cmath.cos(beta))/sw) + (cw*ee*NN1x1*NN1x4*cmath.cos(beta))/(1 - sw**2) + (ee*NN1x2*NN1x3*cmath.sin(beta))/sw + (cw*ee*NN1x1*NN1x3*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2022 = Coupling(name = 'GC_2022',
                   value = '-((ee*NN1x2*NN1x3*cmath.cos(beta))/sw) - (cw*ee*NN1x1*NN1x3*cmath.cos(beta))/(-1 + sw**2) - (ee*NN1x2*NN1x4*cmath.sin(beta))/sw - (cw*ee*NN1x1*NN1x4*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2023 = Coupling(name = 'GC_2023',
                   value = '-(ee*NN1x4*NN2x2*cmath.cos(beta))/(2.*sw) - (ee*NN1x2*NN2x4*cmath.cos(beta))/(2.*sw) + (cw*ee*NN1x4*NN2x1*cmath.cos(beta))/(2.*(1 - sw**2)) - (cw*ee*NN1x1*NN2x4*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*NN1x3*NN2x2*cmath.sin(beta))/(2.*sw) + (ee*NN1x2*NN2x3*cmath.sin(beta))/(2.*sw) + (cw*ee*NN1x3*NN2x1*cmath.sin(beta))/(2.*(-1 + sw**2)) + (cw*ee*NN1x1*NN2x3*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2024 = Coupling(name = 'GC_2024',
                   value = '-((ee*NN2x2*NN2x4*cmath.cos(beta))/sw) + (cw*ee*NN2x1*NN2x4*cmath.cos(beta))/(1 - sw**2) + (ee*NN2x2*NN2x3*cmath.sin(beta))/sw + (cw*ee*NN2x1*NN2x3*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2025 = Coupling(name = 'GC_2025',
                   value = '-((ee*NN2x2*NN2x3*cmath.cos(beta))/sw) - (cw*ee*NN2x1*NN2x3*cmath.cos(beta))/(-1 + sw**2) - (ee*NN2x2*NN2x4*cmath.sin(beta))/sw - (cw*ee*NN2x1*NN2x4*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2026 = Coupling(name = 'GC_2026',
                   value = '-(ee*NN1x4*NN3x2*cmath.cos(beta))/(2.*sw) - (ee*NN1x2*NN3x4*cmath.cos(beta))/(2.*sw) + (cw*ee*NN1x4*NN3x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN1x1*NN3x4*cmath.cos(beta))/(2.*(1 - sw**2)) + (ee*NN1x3*NN3x2*cmath.sin(beta))/(2.*sw) + (ee*NN1x2*NN3x3*cmath.sin(beta))/(2.*sw) + (cw*ee*NN1x3*NN3x1*cmath.sin(beta))/(2.*(-1 + sw**2)) + (cw*ee*NN1x1*NN3x3*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2027 = Coupling(name = 'GC_2027',
                   value = '-(ee*NN2x4*NN3x2*cmath.cos(beta))/(2.*sw) - (ee*NN2x2*NN3x4*cmath.cos(beta))/(2.*sw) + (cw*ee*NN2x4*NN3x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN2x1*NN3x4*cmath.cos(beta))/(2.*(1 - sw**2)) + (ee*NN2x3*NN3x2*cmath.sin(beta))/(2.*sw) + (ee*NN2x2*NN3x3*cmath.sin(beta))/(2.*sw) + (cw*ee*NN2x3*NN3x1*cmath.sin(beta))/(2.*(-1 + sw**2)) + (cw*ee*NN2x1*NN3x3*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2028 = Coupling(name = 'GC_2028',
                   value = '-((ee*NN3x2*NN3x4*cmath.cos(beta))/sw) - (cw*ee*NN3x1*NN3x4*cmath.cos(beta))/(-1 + sw**2) + (ee*NN3x2*NN3x3*cmath.sin(beta))/sw + (cw*ee*NN3x1*NN3x3*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2029 = Coupling(name = 'GC_2029',
                   value = '-(ee*NN1x3*NN3x2*cmath.cos(beta))/(2.*sw) - (ee*NN1x2*NN3x3*cmath.cos(beta))/(2.*sw) + (cw*ee*NN1x1*NN3x3*cmath.cos(beta))/(2.*(1 - sw**2)) - (cw*ee*NN1x3*NN3x1*cmath.cos(beta))/(2.*(-1 + sw**2)) - (ee*NN1x4*NN3x2*cmath.sin(beta))/(2.*sw) - (ee*NN1x2*NN3x4*cmath.sin(beta))/(2.*sw) + (cw*ee*NN1x4*NN3x1*cmath.sin(beta))/(2.*(1 - sw**2)) - (cw*ee*NN1x1*NN3x4*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_203 = Coupling(name = 'GC_203',
                  value = '-((CKM1x1*ee*complex(0,1)*UU2x1*complexconjugate(Rd1x1))/sw)',
                  order = {'QED':1})

GC_2030 = Coupling(name = 'GC_2030',
                   value = '-((ee*NN3x2*NN3x3*cmath.cos(beta))/sw) - (cw*ee*NN3x1*NN3x3*cmath.cos(beta))/(-1 + sw**2) - (ee*NN3x2*NN3x4*cmath.sin(beta))/sw - (cw*ee*NN3x1*NN3x4*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2031 = Coupling(name = 'GC_2031',
                   value = '-(ee*NN1x4*NN4x2*cmath.cos(beta))/(2.*sw) - (ee*NN1x2*NN4x4*cmath.cos(beta))/(2.*sw) + (cw*ee*NN1x4*NN4x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN1x1*NN4x4*cmath.cos(beta))/(2.*(1 - sw**2)) + (ee*NN1x3*NN4x2*cmath.sin(beta))/(2.*sw) + (ee*NN1x2*NN4x3*cmath.sin(beta))/(2.*sw) + (cw*ee*NN1x3*NN4x1*cmath.sin(beta))/(2.*(-1 + sw**2)) + (cw*ee*NN1x1*NN4x3*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2032 = Coupling(name = 'GC_2032',
                   value = '-(ee*NN2x4*NN4x2*cmath.cos(beta))/(2.*sw) - (ee*NN2x2*NN4x4*cmath.cos(beta))/(2.*sw) + (cw*ee*NN2x4*NN4x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN2x1*NN4x4*cmath.cos(beta))/(2.*(1 - sw**2)) + (ee*NN2x3*NN4x2*cmath.sin(beta))/(2.*sw) + (ee*NN2x2*NN4x3*cmath.sin(beta))/(2.*sw) + (cw*ee*NN2x3*NN4x1*cmath.sin(beta))/(2.*(-1 + sw**2)) + (cw*ee*NN2x1*NN4x3*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2033 = Coupling(name = 'GC_2033',
                   value = '-(ee*NN3x4*NN4x2*cmath.cos(beta))/(2.*sw) - (ee*NN3x2*NN4x4*cmath.cos(beta))/(2.*sw) + (cw*ee*NN3x4*NN4x1*cmath.cos(beta))/(2.*(1 - sw**2)) + (cw*ee*NN3x1*NN4x4*cmath.cos(beta))/(2.*(1 - sw**2)) + (ee*NN3x3*NN4x2*cmath.sin(beta))/(2.*sw) + (ee*NN3x2*NN4x3*cmath.sin(beta))/(2.*sw) + (cw*ee*NN3x3*NN4x1*cmath.sin(beta))/(2.*(-1 + sw**2)) + (cw*ee*NN3x1*NN4x3*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2034 = Coupling(name = 'GC_2034',
                   value = '-((ee*NN4x2*NN4x4*cmath.cos(beta))/sw) - (cw*ee*NN4x1*NN4x4*cmath.cos(beta))/(-1 + sw**2) + (ee*NN4x2*NN4x3*cmath.sin(beta))/sw + (cw*ee*NN4x1*NN4x3*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2035 = Coupling(name = 'GC_2035',
                   value = '-((ee*NN4x2*NN4x3*cmath.cos(beta))/sw) - (cw*ee*NN4x1*NN4x3*cmath.cos(beta))/(-1 + sw**2) - (ee*NN4x2*NN4x4*cmath.sin(beta))/sw - (cw*ee*NN4x1*NN4x4*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':1})

GC_2036 = Coupling(name = 'GC_2036',
                   value = '-((ee*complex(0,1)*NN1x3*UU1x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN1x2*UU1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN1x1*UU1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2037 = Coupling(name = 'GC_2037',
                   value = '-((ee*complex(0,1)*NN2x3*UU1x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN2x2*UU1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN2x1*UU1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2038 = Coupling(name = 'GC_2038',
                   value = '-((ee*complex(0,1)*NN3x3*UU1x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN3x2*UU1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN3x1*UU1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2039 = Coupling(name = 'GC_2039',
                   value = '-((ee*complex(0,1)*NN4x3*UU1x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN4x2*UU1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN4x1*UU1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_204 = Coupling(name = 'GC_204',
                  value = '(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_2040 = Coupling(name = 'GC_2040',
                   value = '-((ee*complex(0,1)*NN1x3*UU2x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN1x2*UU2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN1x1*UU2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2041 = Coupling(name = 'GC_2041',
                   value = '-((ee*complex(0,1)*NN2x3*UU2x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN2x2*UU2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN2x1*UU2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2042 = Coupling(name = 'GC_2042',
                   value = '-((ee*complex(0,1)*NN3x3*UU2x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN3x2*UU2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN3x1*UU2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2043 = Coupling(name = 'GC_2043',
                   value = '-((ee*complex(0,1)*NN4x3*UU2x1*cmath.sin(beta))/sw) + (ee*complex(0,1)*NN4x2*UU2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) - (cw*ee*complex(0,1)*NN4x1*UU2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2044 = Coupling(name = 'GC_2044',
                   value = '(ee**2*vu*cmath.cos(beta))/(4.*sw**2) - (ee**2*vd*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2045 = Coupling(name = 'GC_2045',
                   value = '-(ee**2*vu*cmath.cos(beta))/(4.*sw**2) + (ee**2*vd*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2046 = Coupling(name = 'GC_2046',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(beta))/(4.*cw) + (cw*ee**2*complex(0,1)*vu*cmath.cos(beta))/(4.*sw**2) - (ee**2*complex(0,1)*vd*cmath.sin(beta))/(4.*cw) - (cw*ee**2*complex(0,1)*vd*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2047 = Coupling(name = 'GC_2047',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(beta))/(4.*cw) - (cw*ee**2*complex(0,1)*vu*cmath.cos(beta))/(4.*sw**2) - (ee**2*complex(0,1)*vd*cmath.sin(beta))/(4.*cw) + (cw*ee**2*complex(0,1)*vd*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2048 = Coupling(name = 'GC_2048',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(beta))/(2.*sw) - (ee**2*complex(0,1)*vd*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2049 = Coupling(name = 'GC_2049',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(beta))/(2.*sw) + (ee**2*complex(0,1)*vd*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_205 = Coupling(name = 'GC_205',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Rd1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rd1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rd1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_2050 = Coupling(name = 'GC_2050',
                   value = '(cw*ee**2*complex(0,1)*vu*cmath.cos(beta))/(2.*(-1 + sw**2)) - (cw*ee**2*complex(0,1)*vd*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2051 = Coupling(name = 'GC_2051',
                   value = '-(ee**2*vd*cmath.cos(beta))/(4.*sw**2) - (ee**2*vu*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2052 = Coupling(name = 'GC_2052',
                   value = '(ee**2*vd*cmath.cos(beta))/(4.*sw**2) + (ee**2*vu*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2053 = Coupling(name = 'GC_2053',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(beta))/(4.*cw) - (cw*ee**2*complex(0,1)*vd*cmath.cos(beta))/(4.*sw**2) + (ee**2*complex(0,1)*vu*cmath.sin(beta))/(4.*cw) - (cw*ee**2*complex(0,1)*vu*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2054 = Coupling(name = 'GC_2054',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(beta))/(4.*cw) + (cw*ee**2*complex(0,1)*vd*cmath.cos(beta))/(4.*sw**2) + (ee**2*complex(0,1)*vu*cmath.sin(beta))/(4.*cw) + (cw*ee**2*complex(0,1)*vu*cmath.sin(beta))/(4.*sw**2)',
                   order = {'QED':1})

GC_2055 = Coupling(name = 'GC_2055',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(beta))/(2.*sw) - (ee**2*complex(0,1)*vu*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2056 = Coupling(name = 'GC_2056',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(beta))/(2.*sw) + (ee**2*complex(0,1)*vu*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2057 = Coupling(name = 'GC_2057',
                   value = '(cw*ee**2*complex(0,1)*vd*cmath.cos(beta))/(2.*(-1 + sw**2)) + (cw*ee**2*complex(0,1)*vu*cmath.sin(beta))/(2.*(-1 + sw**2))',
                   order = {'QED':1})

GC_2058 = Coupling(name = 'GC_2058',
                   value = '(ee*UU1x1*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU1x2*VV1x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2059 = Coupling(name = 'GC_2059',
                   value = '(ee*UU2x1*VV1x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU2x2*VV1x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_206 = Coupling(name = 'GC_206',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Rd1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rd1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rd1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_2060 = Coupling(name = 'GC_2060',
                   value = '-((ee*complex(0,1)*NN1x4*VV1x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN1x2*VV1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN1x1*VV1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2061 = Coupling(name = 'GC_2061',
                   value = '-((ee*complex(0,1)*NN2x4*VV1x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN2x2*VV1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN2x1*VV1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2062 = Coupling(name = 'GC_2062',
                   value = '-((ee*complex(0,1)*NN3x4*VV1x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN3x2*VV1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN3x1*VV1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2063 = Coupling(name = 'GC_2063',
                   value = '-((ee*complex(0,1)*NN4x4*VV1x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN4x2*VV1x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN4x1*VV1x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2064 = Coupling(name = 'GC_2064',
                   value = '-((ee*UU1x2*VV1x1*cmath.cos(beta))/(sw*cmath.sqrt(2))) + (ee*UU1x1*VV1x2*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2065 = Coupling(name = 'GC_2065',
                   value = '-((ee*UU2x2*VV1x1*cmath.cos(beta))/(sw*cmath.sqrt(2))) + (ee*UU2x1*VV1x2*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2066 = Coupling(name = 'GC_2066',
                   value = '(ee*UU1x1*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU1x2*VV2x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2067 = Coupling(name = 'GC_2067',
                   value = '(ee*UU2x1*VV2x2*cmath.cos(beta))/(sw*cmath.sqrt(2)) + (ee*UU2x2*VV2x1*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2068 = Coupling(name = 'GC_2068',
                   value = '-((ee*complex(0,1)*NN1x4*VV2x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN1x2*VV2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN1x1*VV2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2069 = Coupling(name = 'GC_2069',
                   value = '-((ee*complex(0,1)*NN2x4*VV2x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN2x2*VV2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN2x1*VV2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_207 = Coupling(name = 'GC_207',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Rd1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rd1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rd1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_2070 = Coupling(name = 'GC_2070',
                   value = '-((ee*complex(0,1)*NN3x4*VV2x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN3x2*VV2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN3x1*VV2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2071 = Coupling(name = 'GC_2071',
                   value = '-((ee*complex(0,1)*NN4x4*VV2x1*cmath.sin(beta))/sw) - (ee*complex(0,1)*NN4x2*VV2x2*cmath.sin(beta))/(sw*cmath.sqrt(2)) + (cw*ee*complex(0,1)*NN4x1*VV2x2*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2072 = Coupling(name = 'GC_2072',
                   value = '-((ee*UU1x2*VV2x1*cmath.cos(beta))/(sw*cmath.sqrt(2))) + (ee*UU1x1*VV2x2*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2073 = Coupling(name = 'GC_2073',
                   value = '-((ee*UU2x2*VV2x1*cmath.cos(beta))/(sw*cmath.sqrt(2))) + (ee*UU2x1*VV2x2*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2074 = Coupling(name = 'GC_2074',
                   value = '(cw*ee*complexconjugate(NN1x1)*complexconjugate(NN1x4)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.cos(beta))/sw - (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN1x3)*cmath.sin(beta))/(-1 + sw**2) - (ee*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_2075 = Coupling(name = 'GC_2075',
                   value = '(cw*ee*complexconjugate(NN1x1)*complexconjugate(NN1x3)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN1x2)*complexconjugate(NN1x3)*cmath.cos(beta))/sw + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN1x4)*cmath.sin(beta))/(-1 + sw**2) + (ee*complexconjugate(NN1x2)*complexconjugate(NN1x4)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_2076 = Coupling(name = 'GC_2076',
                   value = '(cw*ee*complexconjugate(NN1x4)*complexconjugate(NN2x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN2x4)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.cos(beta))/(2.*sw) - (cw*ee*complexconjugate(NN1x3)*complexconjugate(NN2x1)*cmath.sin(beta))/(2.*(-1 + sw**2)) - (ee*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN2x3)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2077 = Coupling(name = 'GC_2077',
                   value = '(cw*ee*complexconjugate(NN2x1)*complexconjugate(NN2x4)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.cos(beta))/sw - (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN2x3)*cmath.sin(beta))/(-1 + sw**2) - (ee*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_2078 = Coupling(name = 'GC_2078',
                   value = '(cw*ee*complexconjugate(NN1x3)*complexconjugate(NN2x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x3)*complexconjugate(NN2x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN2x3)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN2x3)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x4)*complexconjugate(NN2x1)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x4)*complexconjugate(NN2x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN2x4)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN2x4)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2079 = Coupling(name = 'GC_2079',
                   value = '(cw*ee*complexconjugate(NN2x1)*complexconjugate(NN2x3)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN2x2)*complexconjugate(NN2x3)*cmath.cos(beta))/sw + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN2x4)*cmath.sin(beta))/(-1 + sw**2) + (ee*complexconjugate(NN2x2)*complexconjugate(NN2x4)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_208 = Coupling(name = 'GC_208',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Rd1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rd1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rd1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_2080 = Coupling(name = 'GC_2080',
                   value = '(cw*ee*complexconjugate(NN1x4)*complexconjugate(NN3x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x3)*complexconjugate(NN3x1)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2081 = Coupling(name = 'GC_2081',
                   value = '(cw*ee*complexconjugate(NN2x4)*complexconjugate(NN3x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x3)*complexconjugate(NN3x1)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2082 = Coupling(name = 'GC_2082',
                   value = '(cw*ee*complexconjugate(NN3x1)*complexconjugate(NN3x4)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.cos(beta))/sw - (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN3x3)*cmath.sin(beta))/(-1 + sw**2) - (ee*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_2083 = Coupling(name = 'GC_2083',
                   value = '(cw*ee*complexconjugate(NN1x3)*complexconjugate(NN3x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x3)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN3x3)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN3x3)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x4)*complexconjugate(NN3x1)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x4)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN3x4)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN3x4)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2084 = Coupling(name = 'GC_2084',
                   value = '(cw*ee*complexconjugate(NN2x3)*complexconjugate(NN3x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x3)*complexconjugate(NN3x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN3x3)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN3x3)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x4)*complexconjugate(NN3x1)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x4)*complexconjugate(NN3x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN3x4)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN3x4)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2085 = Coupling(name = 'GC_2085',
                   value = '(cw*ee*complexconjugate(NN3x1)*complexconjugate(NN3x3)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN3x2)*complexconjugate(NN3x3)*cmath.cos(beta))/sw + (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN3x4)*cmath.sin(beta))/(-1 + sw**2) + (ee*complexconjugate(NN3x2)*complexconjugate(NN3x4)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_2086 = Coupling(name = 'GC_2086',
                   value = '(cw*ee*complexconjugate(NN1x4)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x3)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2087 = Coupling(name = 'GC_2087',
                   value = '(cw*ee*complexconjugate(NN2x4)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x3)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2088 = Coupling(name = 'GC_2088',
                   value = '(cw*ee*complexconjugate(NN3x4)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN3x3)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*(1 - sw**2)) - (ee*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2089 = Coupling(name = 'GC_2089',
                   value = '(cw*ee*complexconjugate(NN4x1)*complexconjugate(NN4x4)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.cos(beta))/sw - (cw*ee*complexconjugate(NN4x1)*complexconjugate(NN4x3)*cmath.sin(beta))/(-1 + sw**2) - (ee*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_209 = Coupling(name = 'GC_209',
                  value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*Rd1x1*sw*complexconjugate(Rd1x1))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_2090 = Coupling(name = 'GC_2090',
                   value = '(cw*ee*complexconjugate(NN1x3)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x3)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN4x3)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN4x3)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x4)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x4)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN1x1)*complexconjugate(NN4x4)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN1x2)*complexconjugate(NN4x4)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2091 = Coupling(name = 'GC_2091',
                   value = '(cw*ee*complexconjugate(NN2x3)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x3)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN4x3)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN4x3)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x4)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x4)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN2x1)*complexconjugate(NN4x4)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN2x2)*complexconjugate(NN4x4)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2092 = Coupling(name = 'GC_2092',
                   value = '(cw*ee*complexconjugate(NN3x3)*complexconjugate(NN4x1)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN3x3)*complexconjugate(NN4x2)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN4x3)*cmath.cos(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN3x2)*complexconjugate(NN4x3)*cmath.cos(beta))/(2.*sw) + (cw*ee*complexconjugate(NN3x4)*complexconjugate(NN4x1)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN3x4)*complexconjugate(NN4x2)*cmath.sin(beta))/(2.*sw) + (cw*ee*complexconjugate(NN3x1)*complexconjugate(NN4x4)*cmath.sin(beta))/(2.*(-1 + sw**2)) + (ee*complexconjugate(NN3x2)*complexconjugate(NN4x4)*cmath.sin(beta))/(2.*sw)',
                   order = {'QED':1})

GC_2093 = Coupling(name = 'GC_2093',
                   value = '(cw*ee*complexconjugate(NN4x1)*complexconjugate(NN4x3)*cmath.cos(beta))/(-1 + sw**2) + (ee*complexconjugate(NN4x2)*complexconjugate(NN4x3)*cmath.cos(beta))/sw + (cw*ee*complexconjugate(NN4x1)*complexconjugate(NN4x4)*cmath.sin(beta))/(-1 + sw**2) + (ee*complexconjugate(NN4x2)*complexconjugate(NN4x4)*cmath.sin(beta))/sw',
                   order = {'QED':1})

GC_2094 = Coupling(name = 'GC_2094',
                   value = '-(CKM1x1*ee**2*complex(0,1)*Ru1x1*vu*complexconjugate(Rd1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (CKM1x1*ee**2*complex(0,1)*Ru1x1*vd*complexconjugate(Rd1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2095 = Coupling(name = 'GC_2095',
                   value = '(CKM1x1*ee**2*complex(0,1)*Ru1x1*vd*complexconjugate(Rd1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (CKM1x1*ee**2*complex(0,1)*Ru1x1*vu*complexconjugate(Rd1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2096 = Coupling(name = 'GC_2096',
                   value = '-(CKM2x2*ee**2*complex(0,1)*Ru2x2*vu*complexconjugate(Rd2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (CKM2x2*ee**2*complex(0,1)*Ru2x2*vd*complexconjugate(Rd2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2097 = Coupling(name = 'GC_2097',
                   value = '(CKM2x2*ee**2*complex(0,1)*Ru2x2*vd*complexconjugate(Rd2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (CKM2x2*ee**2*complex(0,1)*Ru2x2*vu*complexconjugate(Rd2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2098 = Coupling(name = 'GC_2098',
                   value = '(Rd6x6*yd3x3*complexconjugate(MUH)*complexconjugate(Rd3x3)*cmath.cos(beta))/cmath.sqrt(2) + (Rd6x6*td3x3*complexconjugate(Rd3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2099 = Coupling(name = 'GC_2099',
                   value = '-(CKM3x3*complex(0,1)*Ru6x6*yu3x3*complexconjugate(MUH)*complexconjugate(Rd3x3)*cmath.cos(beta)) + CKM3x3*complex(0,1)*Ru6x6*tu3x3*complexconjugate(Rd3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_21 = Coupling(name = 'GC_21',
                 value = 'complex(0,1)*G*Ru4x4*cmath.sqrt(2)',
                 order = {'QCD':1})

GC_210 = Coupling(name = 'GC_210',
                  value = '-(cw*ee*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*Rd1x1*sw*complexconjugate(Rd1x1))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_2100 = Coupling(name = 'GC_2100',
                   value = '-((Rd6x6*td3x3*complexconjugate(Rd3x3)*cmath.cos(beta))/cmath.sqrt(2)) + (Rd6x6*yd3x3*complexconjugate(MUH)*complexconjugate(Rd3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2101 = Coupling(name = 'GC_2101',
                   value = 'CKM3x3*complex(0,1)*Ru6x6*tu3x3*complexconjugate(Rd3x3)*cmath.cos(beta) + CKM3x3*complex(0,1)*Ru6x6*yu3x3*complexconjugate(MUH)*complexconjugate(Rd3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2102 = Coupling(name = 'GC_2102',
                   value = '-(ee**2*complex(0,1)*Rn1x1*vu*complexconjugate(Rl1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rn1x1*vd*complexconjugate(Rl1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2103 = Coupling(name = 'GC_2103',
                   value = '(ee**2*complex(0,1)*Rn1x1*vd*complexconjugate(Rl1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rn1x1*vu*complexconjugate(Rl1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2104 = Coupling(name = 'GC_2104',
                   value = '-(ee**2*complex(0,1)*Rn2x2*vu*complexconjugate(Rl2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rn2x2*vd*complexconjugate(Rl2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2105 = Coupling(name = 'GC_2105',
                   value = '(ee**2*complex(0,1)*Rn2x2*vd*complexconjugate(Rl2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rn2x2*vu*complexconjugate(Rl2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2106 = Coupling(name = 'GC_2106',
                   value = '(Rl6x6*ye3x3*complexconjugate(MUH)*complexconjugate(Rl3x3)*cmath.cos(beta))/cmath.sqrt(2) + (Rl6x6*te3x3*complexconjugate(Rl3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2107 = Coupling(name = 'GC_2107',
                   value = '(ee**2*complex(0,1)*Rn3x3*vd*complexconjugate(Rl3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (complex(0,1)*Rn3x3*vd*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rn3x3*vu*complexconjugate(Rl3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2108 = Coupling(name = 'GC_2108',
                   value = '-((Rl6x6*te3x3*complexconjugate(Rl3x3)*cmath.cos(beta))/cmath.sqrt(2)) + (Rl6x6*ye3x3*complexconjugate(MUH)*complexconjugate(Rl3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2109 = Coupling(name = 'GC_2109',
                   value = '-(ee**2*complex(0,1)*Rl1x1*vu*complexconjugate(Rn1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rl1x1*vd*complexconjugate(Rn1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_211 = Coupling(name = 'GC_211',
                  value = '-(cw*ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(3.*sw*(-1 + sw**2)) + (2*cw*ee**2*complex(0,1)*Rd1x1*sw*complexconjugate(Rd1x1))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_2110 = Coupling(name = 'GC_2110',
                   value = '(ee**2*complex(0,1)*Rl1x1*vd*complexconjugate(Rn1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rl1x1*vu*complexconjugate(Rn1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2111 = Coupling(name = 'GC_2111',
                   value = '-(ee**2*complex(0,1)*Rl2x2*vu*complexconjugate(Rn2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rl2x2*vd*complexconjugate(Rn2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2112 = Coupling(name = 'GC_2112',
                   value = '(ee**2*complex(0,1)*Rl2x2*vd*complexconjugate(Rn2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rl2x2*vu*complexconjugate(Rn2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2113 = Coupling(name = 'GC_2113',
                   value = 'complex(0,1)*Rl6x6*ye3x3*complexconjugate(MUH)*complexconjugate(Rn3x3)*cmath.cos(beta) + complex(0,1)*Rl6x6*te3x3*complexconjugate(Rn3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2114 = Coupling(name = 'GC_2114',
                   value = '(ee**2*complex(0,1)*Rl3x3*vd*complexconjugate(Rn3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (complex(0,1)*Rl3x3*vd*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rl3x3*vu*complexconjugate(Rn3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2115 = Coupling(name = 'GC_2115',
                   value = '-(complex(0,1)*Rl6x6*te3x3*complexconjugate(Rn3x3)*cmath.cos(beta)) + complex(0,1)*Rl6x6*ye3x3*complexconjugate(MUH)*complexconjugate(Rn3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2116 = Coupling(name = 'GC_2116',
                   value = '-(ee**2*complex(0,1)*Rd1x1*vu*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rd1x1*vd*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2117 = Coupling(name = 'GC_2117',
                   value = '(ee**2*complex(0,1)*Rd1x1*vd*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rd1x1*vu*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2118 = Coupling(name = 'GC_2118',
                   value = '-(ee**2*complex(0,1)*Rd2x2*vu*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rd2x2*vd*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2119 = Coupling(name = 'GC_2119',
                   value = '(ee**2*complex(0,1)*Rd2x2*vd*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rd2x2*vu*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':1})

GC_212 = Coupling(name = 'GC_212',
                  value = '(cw*ee*complex(0,1)*G*Rd1x1*complexconjugate(Rd1x1))/(sw*(-1 + sw**2)) - (2*cw*ee*complex(0,1)*G*Rd1x1*sw*complexconjugate(Rd1x1))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_2120 = Coupling(name = 'GC_2120',
                   value = '-((Ru6x6*yu3x3*complexconjugate(MUH)*complexconjugate(Ru3x3)*cmath.cos(beta))/cmath.sqrt(2)) + (Ru6x6*tu3x3*complexconjugate(Ru3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2121 = Coupling(name = 'GC_2121',
                   value = 'complex(0,1)*Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(MUH)*complexconjugate(Ru3x3)*cmath.cos(beta) + complex(0,1)*Rd6x6*td3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2122 = Coupling(name = 'GC_2122',
                   value = '(Ru6x6*tu3x3*complexconjugate(Ru3x3)*cmath.cos(beta))/cmath.sqrt(2) + (Ru6x6*yu3x3*complexconjugate(MUH)*complexconjugate(Ru3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2123 = Coupling(name = 'GC_2123',
                   value = '-(complex(0,1)*Rd6x6*td3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(beta)) + complex(0,1)*Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(MUH)*complexconjugate(Ru3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2124 = Coupling(name = 'GC_2124',
                   value = '-((MUH*Rd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(beta))/cmath.sqrt(2)) - (Rd3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2125 = Coupling(name = 'GC_2125',
                   value = 'CKM3x3*complex(0,1)*MUH*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(beta) + CKM3x3*complex(0,1)*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2126 = Coupling(name = 'GC_2126',
                   value = '-((MUH*Rl3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(beta))/cmath.sqrt(2)) - (Rl3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2127 = Coupling(name = 'GC_2127',
                   value = 'complex(0,1)*MUH*Rn3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.cos(beta) + complex(0,1)*Rn3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2128 = Coupling(name = 'GC_2128',
                   value = '(MUH*Ru3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(beta))/cmath.sqrt(2) - (Ru3x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2129 = Coupling(name = 'GC_2129',
                   value = '-(complex(0,1)*MUH*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(beta)) + complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(tu3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_213 = Coupling(name = 'GC_213',
                  value = '(2*ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1))/(2.*sw**2*(-1 + sw**2)) - (2*ee**2*complex(0,1)*Rd1x1*sw**2*complexconjugate(Rd1x1))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_2130 = Coupling(name = 'GC_2130',
                   value = '-((ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(UU1x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(UU1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2131 = Coupling(name = 'GC_2131',
                   value = '-((ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(UU1x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(UU1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2132 = Coupling(name = 'GC_2132',
                   value = '-((ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(UU1x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(UU1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2133 = Coupling(name = 'GC_2133',
                   value = '-((ee*complex(0,1)*complexconjugate(NN4x3)*complexconjugate(UU1x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(UU1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(UU1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2134 = Coupling(name = 'GC_2134',
                   value = '-((ee*complex(0,1)*complexconjugate(NN1x3)*complexconjugate(UU2x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(UU2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2135 = Coupling(name = 'GC_2135',
                   value = '-((ee*complex(0,1)*complexconjugate(NN2x3)*complexconjugate(UU2x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(UU2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2136 = Coupling(name = 'GC_2136',
                   value = '-((ee*complex(0,1)*complexconjugate(NN3x3)*complexconjugate(UU2x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(UU2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2137 = Coupling(name = 'GC_2137',
                   value = '-((ee*complex(0,1)*complexconjugate(NN4x3)*complexconjugate(UU2x1)*cmath.sin(beta))/sw) - (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(UU2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(UU2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2138 = Coupling(name = 'GC_2138',
                   value = '-((ee*complexconjugate(UU1x1)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU1x2)*complexconjugate(VV1x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2139 = Coupling(name = 'GC_2139',
                   value = '-((ee*complexconjugate(UU2x1)*complexconjugate(VV1x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU2x2)*complexconjugate(VV1x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_214 = Coupling(name = 'GC_214',
                  value = '(-2*ee**2*complex(0,1)*Rd1x1**2*complexconjugate(Rd1x1)**2)/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1**2*complexconjugate(Rd1x1)**2)/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_2140 = Coupling(name = 'GC_2140',
                   value = '-((ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(VV1x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(VV1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(VV1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2141 = Coupling(name = 'GC_2141',
                   value = '-((ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(VV1x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(VV1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(VV1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2142 = Coupling(name = 'GC_2142',
                   value = '-((ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(VV1x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(VV1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(VV1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2143 = Coupling(name = 'GC_2143',
                   value = '-((ee*complex(0,1)*complexconjugate(NN4x4)*complexconjugate(VV1x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(VV1x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(VV1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2144 = Coupling(name = 'GC_2144',
                   value = '(ee*complexconjugate(UU1x2)*complexconjugate(VV1x1)*cmath.cos(beta))/(sw*cmath.sqrt(2)) - (ee*complexconjugate(UU1x1)*complexconjugate(VV1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2145 = Coupling(name = 'GC_2145',
                   value = '(ee*complexconjugate(UU2x2)*complexconjugate(VV1x1)*cmath.cos(beta))/(sw*cmath.sqrt(2)) - (ee*complexconjugate(UU2x1)*complexconjugate(VV1x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2146 = Coupling(name = 'GC_2146',
                   value = '-((ee*complexconjugate(UU1x1)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU1x2)*complexconjugate(VV2x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2147 = Coupling(name = 'GC_2147',
                   value = '-((ee*complexconjugate(UU2x1)*complexconjugate(VV2x2)*cmath.cos(beta))/(sw*cmath.sqrt(2))) - (ee*complexconjugate(UU2x2)*complexconjugate(VV2x1)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2148 = Coupling(name = 'GC_2148',
                   value = '-((ee*complex(0,1)*complexconjugate(NN1x4)*complexconjugate(VV2x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(VV2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN1x2)*complexconjugate(VV2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2149 = Coupling(name = 'GC_2149',
                   value = '-((ee*complex(0,1)*complexconjugate(NN2x4)*complexconjugate(VV2x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(VV2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN2x2)*complexconjugate(VV2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_215 = Coupling(name = 'GC_215',
                  value = '(complex(0,1)*G**2*Rd1x1**2*complexconjugate(Rd1x1)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Rd1x1**2*sw**2*complexconjugate(Rd1x1)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_2150 = Coupling(name = 'GC_2150',
                   value = '-((ee*complex(0,1)*complexconjugate(NN3x4)*complexconjugate(VV2x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(VV2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN3x2)*complexconjugate(VV2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2151 = Coupling(name = 'GC_2151',
                   value = '-((ee*complex(0,1)*complexconjugate(NN4x4)*complexconjugate(VV2x1)*cmath.sin(beta))/sw) + (cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(VV2x2)*cmath.sin(beta))/((-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*complexconjugate(NN4x2)*complexconjugate(VV2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2152 = Coupling(name = 'GC_2152',
                   value = '(ee*complexconjugate(UU1x2)*complexconjugate(VV2x1)*cmath.cos(beta))/(sw*cmath.sqrt(2)) - (ee*complexconjugate(UU1x1)*complexconjugate(VV2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2153 = Coupling(name = 'GC_2153',
                   value = '(ee*complexconjugate(UU2x2)*complexconjugate(VV2x1)*cmath.cos(beta))/(sw*cmath.sqrt(2)) - (ee*complexconjugate(UU2x1)*complexconjugate(VV2x2)*cmath.sin(beta))/(sw*cmath.sqrt(2))',
                   order = {'QED':1})

GC_2154 = Coupling(name = 'GC_2154',
                   value = '-(CKM3x3*ee**2*complex(0,1)*Ru3x3*vu*complexconjugate(Rd3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + (CKM3x3*complex(0,1)*Ru3x3*vu*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta))/cmath.sqrt(2) - (CKM3x3*ee**2*complex(0,1)*Ru3x3*vd*complexconjugate(Rd3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (CKM3x3*complex(0,1)*Ru3x3*vd*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2155 = Coupling(name = 'GC_2155',
                   value = '(Rd3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)*cmath.cos(beta))/cmath.sqrt(2) - (MUH*Rd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2156 = Coupling(name = 'GC_2156',
                   value = '-(CKM3x3*complex(0,1)*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)*cmath.cos(beta)) + CKM3x3*complex(0,1)*MUH*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2157 = Coupling(name = 'GC_2157',
                   value = '-((CKM3x3*complex(0,1)*Ru6x6*vu*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(beta))/cmath.sqrt(2)) + (CKM3x3*complex(0,1)*Ru6x6*vd*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2158 = Coupling(name = 'GC_2158',
                   value = '(CKM3x3*complex(0,1)*Ru6x6*vd*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.cos(beta))/cmath.sqrt(2) + (CKM3x3*complex(0,1)*Ru6x6*vu*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2159 = Coupling(name = 'GC_2159',
                   value = '-(ee**2*complex(0,1)*Rd3x3*vu*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*Rd3x3*vu*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rd3x3*vd*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*Rd3x3*vd*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_216 = Coupling(name = 'GC_216',
                  value = '-(complex(0,1)*G*complexconjugate(Rd2x2)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_2160 = Coupling(name = 'GC_2160',
                   value = '-(ee**2*complex(0,1)*Rn3x3*vu*complexconjugate(Rl3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rn3x3*vd*complexconjugate(Rl3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*Rn3x3*vd*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2161 = Coupling(name = 'GC_2161',
                   value = '(Rl3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)*cmath.cos(beta))/cmath.sqrt(2) - (MUH*Rl3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2162 = Coupling(name = 'GC_2162',
                   value = '-(complex(0,1)*Rn3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)*cmath.cos(beta)) + complex(0,1)*MUH*Rn3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2163 = Coupling(name = 'GC_2163',
                   value = '-(ee**2*complex(0,1)*Rl3x3*vu*complexconjugate(Rn3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (ee**2*complex(0,1)*Rl3x3*vd*complexconjugate(Rn3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*Rl3x3*vd*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2164 = Coupling(name = 'GC_2164',
                   value = '(CKM3x3*ee**2*complex(0,1)*Ru3x3*vd*complexconjugate(Rd3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (CKM3x3*complex(0,1)*Ru3x3*vd*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(beta))/cmath.sqrt(2) - (CKM3x3*ee**2*complex(0,1)*Ru3x3*vu*complexconjugate(Rd3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (CKM3x3*complex(0,1)*Ru3x3*vu*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2165 = Coupling(name = 'GC_2165',
                   value = '(ee**2*complex(0,1)*Rd3x3*vd*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(beta))/(2.*sw**2*cmath.sqrt(2)) - (complex(0,1)*Rd3x3*vd*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rd3x3*vu*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.sin(beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*Rd3x3*vu*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2166 = Coupling(name = 'GC_2166',
                   value = '-((Ru3x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)*cmath.cos(beta))/cmath.sqrt(2)) - (MUH*Ru3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2167 = Coupling(name = 'GC_2167',
                   value = 'complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(tu3x3)*cmath.cos(beta) + complex(0,1)*MUH*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(beta)',
                   order = {'QED':1})

GC_2168 = Coupling(name = 'GC_2168',
                   value = '-((complex(0,1)*Rd6x6*vu*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(beta))/cmath.sqrt(2)) + (complex(0,1)*Rd6x6*vd*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_2169 = Coupling(name = 'GC_2169',
                   value = '(complex(0,1)*Rd6x6*vd*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.cos(beta))/cmath.sqrt(2) + (complex(0,1)*Rd6x6*vu*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':1})

GC_217 = Coupling(name = 'GC_217',
                  value = '(ee*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/3.',
                  order = {'QED':1})

GC_2170 = Coupling(name = 'GC_2170',
                   value = '(2*ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2171 = Coupling(name = 'GC_2171',
                   value = '-(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2172 = Coupling(name = 'GC_2172',
                   value = '(2*ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2173 = Coupling(name = 'GC_2173',
                   value = '-(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2174 = Coupling(name = 'GC_2174',
                   value = '(2*ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2175 = Coupling(name = 'GC_2175',
                   value = '-(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2176 = Coupling(name = 'GC_2176',
                   value = '(2*ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2177 = Coupling(name = 'GC_2177',
                   value = '-(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2178 = Coupling(name = 'GC_2178',
                   value = '-(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2)) - (CKM3x3*complex(0,1)*Ru3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta)*cmath.sin(beta))/(-1 + sw**2) + (CKM3x3*complex(0,1)*Ru3x3*sw**2*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta)*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':2})

GC_2179 = Coupling(name = 'GC_2179',
                   value = '-((ee**2*Rn3x3*complexconjugate(Rl3x3)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2))) + (Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(beta)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_218 = Coupling(name = 'GC_218',
                  value = '(2*ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/9.',
                  order = {'QED':2})

GC_2180 = Coupling(name = 'GC_2180',
                   value = '(ee**2*Rl3x3*complexconjugate(Rn3x3)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2)) - (Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(beta)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2181 = Coupling(name = 'GC_2181',
                   value = '-((ee**2*complex(0,1)*cmath.cos(beta)*cmath.sin(beta))/(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2)) + complex(0,1)*Rn3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(beta)*cmath.sin(beta)',
                   order = {'QED':2})

GC_2182 = Coupling(name = 'GC_2182',
                   value = '-((CKM3x3*ee**2*Ru3x3*complexconjugate(Rd3x3)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2))) + (CKM3x3*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(beta)*cmath.sin(beta))/cmath.sqrt(2) + (CKM3x3*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2183 = Coupling(name = 'GC_2183',
                   value = '(2*ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(beta)*cmath.sin(beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2)) + (CKM3x3*complex(0,1)*Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta)*cmath.sin(beta))/(-1 + sw**2) - (CKM3x3*complex(0,1)*Rd3x3*sw**2*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta)*cmath.sin(beta))/(-1 + sw**2)',
                   order = {'QED':2})

GC_2184 = Coupling(name = 'GC_2184',
                   value = '(ee**2*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(beta)*cmath.sin(beta))/(sw**2*cmath.sqrt(2)) - (Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta)*cmath.sin(beta))/cmath.sqrt(2) - (Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(beta)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_219 = Coupling(name = 'GC_219',
                  value = '-(complex(0,1)*G*Rd2x2*complexconjugate(Rd2x2))',
                  order = {'QCD':1})

GC_22 = Coupling(name = 'GC_22',
                 value = 'complex(0,1)*G*Ru5x5*cmath.sqrt(2)',
                 order = {'QCD':1})

GC_220 = Coupling(name = 'GC_220',
                  value = 'complex(0,1)*G*Rd2x2*complexconjugate(Rd2x2)',
                  order = {'QCD':1})

GC_221 = Coupling(name = 'GC_221',
                  value = '(-2*ee*complex(0,1)*G*Rd2x2*complexconjugate(Rd2x2))/3.',
                  order = {'QCD':1,'QED':1})

GC_222 = Coupling(name = 'GC_222',
                  value = 'complex(0,1)*G**2*Rd2x2*complexconjugate(Rd2x2)',
                  order = {'QCD':2})

GC_223 = Coupling(name = 'GC_223',
                  value = '(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_2231 = Coupling(name = 'GC_2231',
                   value = '(CKM3x3*complex(0,1)*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(alp)*cmath.cos(beta))/cmath.sqrt(2) - (CKM3x3*ee**2*complex(0,1)*Ru3x3*complexconjugate(Rd3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2)) - (CKM3x3*complex(0,1)*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(alp)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2232 = Coupling(name = 'GC_2232',
                   value = '(complex(0,1)*Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(alp)*cmath.cos(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2)) - (complex(0,1)*Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.sin(alp)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2233 = Coupling(name = 'GC_2233',
                   value = '-(ee**2*complex(0,1)*Rn3x3*complexconjugate(Rl3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2)) - (complex(0,1)*Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(alp)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2234 = Coupling(name = 'GC_2234',
                   value = '-(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rn3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2)) - (complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.sin(alp)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2235 = Coupling(name = 'GC_2235',
                   value = '-((CKM3x3*complex(0,1)*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(alp)*cmath.cos(beta))/cmath.sqrt(2)) + (CKM3x3*ee**2*complex(0,1)*Ru3x3*complexconjugate(Rd3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2)) + (CKM3x3*complex(0,1)*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.sin(alp)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2236 = Coupling(name = 'GC_2236',
                   value = '-((complex(0,1)*Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(alp)*cmath.cos(beta))/cmath.sqrt(2)) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(alp + beta))/(2.*sw**2*cmath.sqrt(2)) + (complex(0,1)*Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(alp)*cmath.sin(beta))/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2237 = Coupling(name = 'GC_2237',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(alp)*cmath.cos(beta)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.cos(beta)*cmath.sin(alp)*cmath.sin(beta))/(2.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_2238 = Coupling(name = 'GC_2238',
                   value = '(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(beta)**2)/(4.*sw**2) - (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(beta)**2)/(12.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.sin(beta)**2)/(4.*sw**2) + (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.sin(beta)**2)/(12.*(-1 + sw**2))',
                   order = {'QED':2})

GC_2239 = Coupling(name = 'GC_2239',
                   value = '(CKM3x3*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/cmath.sqrt(2) - (CKM3x3*ee**2*Ru3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2)) - (CKM3x3*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/cmath.sqrt(2)',
                   order = {'QED':2})

GC_224 = Coupling(name = 'GC_224',
                  value = '-((CKM2x2*ee*complex(0,1)*Ru2x2*complexconjugate(Rd2x2))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_2240 = Coupling(name = 'GC_2240',
                   value = '-(ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rd3x3*sw**2*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2241 = Coupling(name = 'GC_2241',
                   value = '(ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) + (complex(0,1)*Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2242 = Coupling(name = 'GC_2242',
                   value = '-((Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(beta)**2)/cmath.sqrt(2)) + (ee**2*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2)) + (Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2243 = Coupling(name = 'GC_2243',
                   value = '-(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (CKM3x3*complex(0,1)*Ru3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (CKM3x3*complex(0,1)*Ru3x3*sw**2*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2244 = Coupling(name = 'GC_2244',
                   value = '-(ee**2*Rn3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2)) - (Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2245 = Coupling(name = 'GC_2245',
                   value = '-(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rl3x3*sw**2*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2246 = Coupling(name = 'GC_2246',
                   value = '(ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2247 = Coupling(name = 'GC_2247',
                   value = '(ee**2*Rl3x3*complexconjugate(Rn3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2)) + (Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2248 = Coupling(name = 'GC_2248',
                   value = '-(ee**2*complex(0,1)*cmath.cos(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Rn3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (complex(0,1)*Rn3x3*sw**2*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2249 = Coupling(name = 'GC_2249',
                   value = '-((CKM3x3*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/cmath.sqrt(2)) + (CKM3x3*ee**2*Ru3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2)) + (CKM3x3*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/cmath.sqrt(2)',
                   order = {'QED':2})

GC_225 = Coupling(name = 'GC_225',
                  value = '(CKM2x2*ee*complex(0,1)*Ru2x2*complexconjugate(Rd2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_2250 = Coupling(name = 'GC_2250',
                   value = '-(ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (CKM3x3*complex(0,1)*Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (CKM3x3*complex(0,1)*Rd3x3*sw**2*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2251 = Coupling(name = 'GC_2251',
                   value = '-(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (complex(0,1)*Ru3x3*sw**2*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2252 = Coupling(name = 'GC_2252',
                   value = '(Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta)**2)/cmath.sqrt(2) - (ee**2*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.cos(2*beta))/(2.*sw**2*cmath.sqrt(2)) - (Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/cmath.sqrt(2)',
                   order = {'QED':2})

GC_2253 = Coupling(name = 'GC_2253',
                   value = '(ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6)*cmath.cos(2*beta))/(3.*(-1 + sw**2)) + (complex(0,1)*Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/(-1 + sw**2) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(beta)**2)/(-1 + sw**2)',
                   order = {'QED':2})

GC_2258 = Coupling(name = 'GC_2258',
                   value = '(ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4)*cmath.sin(2*beta))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_2259 = Coupling(name = 'GC_2259',
                   value = '(ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5)*cmath.sin(2*beta))/(6.*(-1 + sw**2))',
                   order = {'QED':2})

GC_226 = Coupling(name = 'GC_226',
                  value = '(CKM2x2*ee**2*complex(0,1)*Ru2x2*complexconjugate(Rd2x2))/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_2260 = Coupling(name = 'GC_2260',
                   value = '-(ee**2*complex(0,1)*cmath.cos(2*alp)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2261 = Coupling(name = 'GC_2261',
                   value = '(ee**2*complex(0,1)*cmath.cos(2*alp)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2262 = Coupling(name = 'GC_2262',
                   value = '-(ee**2*complex(0,1)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2263 = Coupling(name = 'GC_2263',
                   value = '-(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2264 = Coupling(name = 'GC_2264',
                   value = '-(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2265 = Coupling(name = 'GC_2265',
                   value = '-(ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.sin(2*beta))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2)) - (complex(0,1)*Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (complex(0,1)*Rd3x3*sw**2*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_2266 = Coupling(name = 'GC_2266',
                   value = '(ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6)*cmath.sin(2*beta))/(6.*(-1 + sw**2)) - (complex(0,1)*Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (complex(0,1)*Rd6x6*sw**2*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_2267 = Coupling(name = 'GC_2267',
                   value = '-(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2)) - (complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl3x3*sw**2*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_2268 = Coupling(name = 'GC_2268',
                   value = '(ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) - (complex(0,1)*Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl6x6*sw**2*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_2269 = Coupling(name = 'GC_2269',
                   value = '(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.sin(2*beta))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2)) + (complex(0,1)*Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) - (complex(0,1)*Ru3x3*sw**2*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_227 = Coupling(name = 'GC_227',
                  value = '(CKM2x2*ee*complex(0,1)*G*Ru2x2*complexconjugate(Rd2x2)*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_2270 = Coupling(name = 'GC_2270',
                   value = '-(ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6)*cmath.sin(2*beta))/(3.*(-1 + sw**2)) + (complex(0,1)*Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2)) - (complex(0,1)*Ru6x6*sw**2*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)*cmath.sin(2*beta))/(2.*(-1 + sw**2))',
                   order = {'QED':2})

GC_2273 = Coupling(name = 'GC_2273',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp)*cmath.sin(2*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_2274 = Coupling(name = 'GC_2274',
                   value = '-(ee**2*complex(0,1)*cmath.sin(4*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2275 = Coupling(name = 'GC_2275',
                   value = '(ee**2*complex(0,1)*cmath.sin(4*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2276 = Coupling(name = 'GC_2276',
                   value = '-(ee**2*complex(0,1)*cmath.sin(4*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2277 = Coupling(name = 'GC_2277',
                   value = '(ee**2*complex(0,1)*cmath.sin(4*beta))/(4.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2278 = Coupling(name = 'GC_2278',
                   value = '(-3*ee**2*complex(0,1)*cmath.sin(4*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2279 = Coupling(name = 'GC_2279',
                   value = '(3*ee**2*complex(0,1)*cmath.sin(4*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_228 = Coupling(name = 'GC_228',
                  value = '(CKM2x2*cw*ee**2*complex(0,1)*Ru2x2*complexconjugate(Rd2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_2280 = Coupling(name = 'GC_2280',
                   value = '-(CKM1x1*ee**2*complex(0,1)*Ru1x1*complexconjugate(Rd1x1)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2281 = Coupling(name = 'GC_2281',
                   value = '-(CKM2x2*ee**2*complex(0,1)*Ru2x2*complexconjugate(Rd2x2)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2282 = Coupling(name = 'GC_2282',
                   value = '-(ee**2*complex(0,1)*Rn1x1*complexconjugate(Rl1x1)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2283 = Coupling(name = 'GC_2283',
                   value = '-(ee**2*complex(0,1)*Rn2x2*complexconjugate(Rl2x2)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2284 = Coupling(name = 'GC_2284',
                   value = '-(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rn1x1)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2285 = Coupling(name = 'GC_2285',
                   value = '-(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rn2x2)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2286 = Coupling(name = 'GC_2286',
                   value = '-(ee**2*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2287 = Coupling(name = 'GC_2287',
                   value = '-(ee**2*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2288 = Coupling(name = 'GC_2288',
                   value = '(CKM3x3*complex(0,1)*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(beta)*cmath.sin(alp))/cmath.sqrt(2) + (CKM3x3*complex(0,1)*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(alp)*cmath.sin(beta))/cmath.sqrt(2) - (CKM3x3*ee**2*complex(0,1)*Ru3x3*complexconjugate(Rd3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2289 = Coupling(name = 'GC_2289',
                   value = '(CKM3x3*complex(0,1)*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)*cmath.cos(beta)*cmath.sin(alp))/cmath.sqrt(2) + (CKM3x3*complex(0,1)*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)*cmath.cos(alp)*cmath.sin(beta))/cmath.sqrt(2) - (CKM3x3*ee**2*complex(0,1)*Ru3x3*complexconjugate(Rd3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_229 = Coupling(name = 'GC_229',
                  value = '-((CKM2x2*ee*complex(0,1)*UU1x1*complexconjugate(Rd2x2))/sw)',
                  order = {'QED':1})

GC_2290 = Coupling(name = 'GC_2290',
                   value = '(complex(0,1)*Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(beta)*cmath.sin(alp))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rn3x3*complexconjugate(Rl3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2291 = Coupling(name = 'GC_2291',
                   value = '(complex(0,1)*Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)*cmath.cos(alp)*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rn3x3*complexconjugate(Rl3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2292 = Coupling(name = 'GC_2292',
                   value = '(complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(beta)*cmath.sin(alp))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rn3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2293 = Coupling(name = 'GC_2293',
                   value = '(complex(0,1)*Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)*cmath.cos(alp)*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rn3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2294 = Coupling(name = 'GC_2294',
                   value = '(complex(0,1)*Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(beta)*cmath.sin(alp))/cmath.sqrt(2) + (complex(0,1)*Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(alp)*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2295 = Coupling(name = 'GC_2295',
                   value = '(complex(0,1)*Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)*cmath.cos(beta)*cmath.sin(alp))/cmath.sqrt(2) + (complex(0,1)*Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)*cmath.cos(alp)*cmath.sin(beta))/cmath.sqrt(2) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.sin(alp + beta))/(2.*sw**2*cmath.sqrt(2))',
                   order = {'QED':2})

GC_2296 = Coupling(name = 'GC_2296',
                   value = '(ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.sin(2*(alp + beta)))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2297 = Coupling(name = 'GC_2297',
                   value = '-(ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.sin(2*(alp + beta)))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2298 = Coupling(name = 'GC_2298',
                   value = '(ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.sin(2*(alp + beta)))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_2299 = Coupling(name = 'GC_2299',
                   value = '-(ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.sin(2*(alp - beta)))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*cmath.sin(2*(alp + beta)))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':2})

GC_23 = Coupling(name = 'GC_23',
                 value = 'complex(0,1)*G*Ru6x6*cmath.sqrt(2)',
                 order = {'QCD':1})

GC_230 = Coupling(name = 'GC_230',
                  value = '-((CKM2x2*ee*complex(0,1)*UU2x1*complexconjugate(Rd2x2))/sw)',
                  order = {'QED':1})

GC_2300 = Coupling(name = 'GC_2300',
                   value = '(ee**2*complex(0,1)*vu*cmath.cos(alp - 2*beta))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.cos(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.cos(alp + 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.sin(alp - 2*beta))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp + 2*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_2301 = Coupling(name = 'GC_2301',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.cos(alp - 2*beta))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.cos(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.cos(alp + 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp - 2*beta))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.sin(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp + 2*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_2302 = Coupling(name = 'GC_2302',
                   value = '-(ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.cos(alp - 2*beta))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.cos(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.cos(alp + 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.sin(alp - 2*beta))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.sin(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.sin(alp + 2*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_2303 = Coupling(name = 'GC_2303',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.cos(alp - 2*beta))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.cos(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.cos(alp + 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp - 2*beta))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp + 2*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_2304 = Coupling(name = 'GC_2304',
                   value = '-(ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.cos(alp))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.cos(alp - 2*beta))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vd*cmath.cos(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.cos(alp + 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp))/(4.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp - 2*beta))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp + 2*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_2305 = Coupling(name = 'GC_2305',
                   value = '(ee**2*complex(0,1)*vd*cmath.cos(alp - 2*beta))/(4.*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.cos(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vd*cmath.cos(alp + 2*beta))/(8.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*vu*cmath.sin(alp - 2*beta))/(4.*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp - 2*beta))/(8.*sw**2*(-1 + sw**2)) + (ee**2*complex(0,1)*vu*cmath.sin(alp + 2*beta))/(8.*sw**2*(-1 + sw**2))',
                   order = {'QED':1})

GC_231 = Coupling(name = 'GC_231',
                  value = '(ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_232 = Coupling(name = 'GC_232',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Rd2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rd2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rd2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_233 = Coupling(name = 'GC_233',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Rd2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rd2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rd2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_234 = Coupling(name = 'GC_234',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Rd2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rd2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rd2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_235 = Coupling(name = 'GC_235',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Rd2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rd2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rd2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_236 = Coupling(name = 'GC_236',
                  value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*Rd2x2*sw*complexconjugate(Rd2x2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_237 = Coupling(name = 'GC_237',
                  value = '-(cw*ee*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*Rd2x2*sw*complexconjugate(Rd2x2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_238 = Coupling(name = 'GC_238',
                  value = '-(cw*ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(3.*sw*(-1 + sw**2)) + (2*cw*ee**2*complex(0,1)*Rd2x2*sw*complexconjugate(Rd2x2))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_239 = Coupling(name = 'GC_239',
                  value = '(cw*ee*complex(0,1)*G*Rd2x2*complexconjugate(Rd2x2))/(sw*(-1 + sw**2)) - (2*cw*ee*complex(0,1)*G*Rd2x2*sw*complexconjugate(Rd2x2))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_24 = Coupling(name = 'GC_24',
                 value = '-(ee**2*complex(0,1)) + (ee**2*complex(0,1))/sw**2',
                 order = {'QED':2})

GC_240 = Coupling(name = 'GC_240',
                  value = '(2*ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*complexconjugate(Rd2x2))/(2.*sw**2*(-1 + sw**2)) - (2*ee**2*complex(0,1)*Rd2x2*sw**2*complexconjugate(Rd2x2))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_241 = Coupling(name = 'GC_241',
                  value = '(-2*ee**2*complex(0,1)*Rd1x1*Rd2x2*complexconjugate(Rd1x1)*complexconjugate(Rd2x2))/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*Rd2x2*complexconjugate(Rd1x1)*complexconjugate(Rd2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_242 = Coupling(name = 'GC_242',
                  value = '(complex(0,1)*G**2*Rd1x1*Rd2x2*complexconjugate(Rd1x1)*complexconjugate(Rd2x2))/(-1 + sw**2) - (complex(0,1)*G**2*Rd1x1*Rd2x2*sw**2*complexconjugate(Rd1x1)*complexconjugate(Rd2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_243 = Coupling(name = 'GC_243',
                  value = '(-2*ee**2*complex(0,1)*Rd2x2**2*complexconjugate(Rd2x2)**2)/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2**2*complexconjugate(Rd2x2)**2)/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_244 = Coupling(name = 'GC_244',
                  value = '(complex(0,1)*G**2*Rd2x2**2*complexconjugate(Rd2x2)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Rd2x2**2*sw**2*complexconjugate(Rd2x2)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_245 = Coupling(name = 'GC_245',
                  value = '-(complex(0,1)*G*complexconjugate(Rd3x3)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_246 = Coupling(name = 'GC_246',
                  value = '(ee*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/3.',
                  order = {'QED':1})

GC_247 = Coupling(name = 'GC_247',
                  value = '(2*ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/9.',
                  order = {'QED':2})

GC_248 = Coupling(name = 'GC_248',
                  value = '-(complex(0,1)*G*Rd3x3*complexconjugate(Rd3x3))',
                  order = {'QCD':1})

GC_249 = Coupling(name = 'GC_249',
                  value = 'complex(0,1)*G*Rd3x3*complexconjugate(Rd3x3)',
                  order = {'QCD':1})

GC_25 = Coupling(name = 'GC_25',
                 value = '2*ee**2*complex(0,1) - (2*ee**2*complex(0,1))/sw**2',
                 order = {'QED':2})

GC_250 = Coupling(name = 'GC_250',
                  value = '(-2*ee*complex(0,1)*G*Rd3x3*complexconjugate(Rd3x3))/3.',
                  order = {'QCD':1,'QED':1})

GC_251 = Coupling(name = 'GC_251',
                  value = 'complex(0,1)*G**2*Rd3x3*complexconjugate(Rd3x3)',
                  order = {'QCD':2})

GC_252 = Coupling(name = 'GC_252',
                  value = '(ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_253 = Coupling(name = 'GC_253',
                  value = '-((CKM3x3*ee*complex(0,1)*Ru3x3*complexconjugate(Rd3x3))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_254 = Coupling(name = 'GC_254',
                  value = '(CKM3x3*ee*complex(0,1)*Ru3x3*complexconjugate(Rd3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_255 = Coupling(name = 'GC_255',
                  value = '(CKM3x3*ee**2*complex(0,1)*Ru3x3*complexconjugate(Rd3x3))/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_256 = Coupling(name = 'GC_256',
                  value = '(CKM3x3*ee*complex(0,1)*G*Ru3x3*complexconjugate(Rd3x3)*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_257 = Coupling(name = 'GC_257',
                  value = '(CKM3x3*cw*ee**2*complex(0,1)*Ru3x3*complexconjugate(Rd3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_258 = Coupling(name = 'GC_258',
                  value = '-((CKM3x3*ee*complex(0,1)*UU1x1*complexconjugate(Rd3x3))/sw)',
                  order = {'QED':1})

GC_259 = Coupling(name = 'GC_259',
                  value = '-((CKM3x3*ee*complex(0,1)*UU2x1*complexconjugate(Rd3x3))/sw)',
                  order = {'QED':1})

GC_26 = Coupling(name = 'GC_26',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_260 = Coupling(name = 'GC_260',
                  value = '(ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_261 = Coupling(name = 'GC_261',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Rd3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*complexconjugate(Rd3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rd3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_262 = Coupling(name = 'GC_262',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Rd3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*complexconjugate(Rd3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rd3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_263 = Coupling(name = 'GC_263',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Rd3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*complexconjugate(Rd3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rd3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_264 = Coupling(name = 'GC_264',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Rd3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*complexconjugate(Rd3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rd3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_265 = Coupling(name = 'GC_265',
                  value = '(cw*ee*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*Rd3x3*sw*complexconjugate(Rd3x3))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_266 = Coupling(name = 'GC_266',
                  value = '-(cw*ee*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*Rd3x3*sw*complexconjugate(Rd3x3))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_267 = Coupling(name = 'GC_267',
                  value = '-(cw*ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(3.*sw*(-1 + sw**2)) + (2*cw*ee**2*complex(0,1)*Rd3x3*sw*complexconjugate(Rd3x3))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_268 = Coupling(name = 'GC_268',
                  value = '(cw*ee*complex(0,1)*G*Rd3x3*complexconjugate(Rd3x3))/(sw*(-1 + sw**2)) - (2*cw*ee*complex(0,1)*G*Rd3x3*sw*complexconjugate(Rd3x3))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_269 = Coupling(name = 'GC_269',
                  value = '(2*ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*complexconjugate(Rd3x3))/(2.*sw**2*(-1 + sw**2)) - (2*ee**2*complex(0,1)*Rd3x3*sw**2*complexconjugate(Rd3x3))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_27 = Coupling(name = 'GC_27',
                 value = '-((ee**2*complex(0,1))/sw**2)',
                 order = {'QED':2})

GC_270 = Coupling(name = 'GC_270',
                  value = '(complex(0,1)*yd3x3*complexconjugate(NN1x3)*complexconjugate(Rd3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yd3x3*complexconjugate(NN1x3)*complexconjugate(Rd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_271 = Coupling(name = 'GC_271',
                  value = '(complex(0,1)*yd3x3*complexconjugate(NN2x3)*complexconjugate(Rd3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yd3x3*complexconjugate(NN2x3)*complexconjugate(Rd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_272 = Coupling(name = 'GC_272',
                  value = '(complex(0,1)*yd3x3*complexconjugate(NN3x3)*complexconjugate(Rd3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yd3x3*complexconjugate(NN3x3)*complexconjugate(Rd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_273 = Coupling(name = 'GC_273',
                  value = '(complex(0,1)*yd3x3*complexconjugate(NN4x3)*complexconjugate(Rd3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yd3x3*complexconjugate(NN4x3)*complexconjugate(Rd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_274 = Coupling(name = 'GC_274',
                  value = '(-2*ee**2*complex(0,1)*Rd1x1*Rd3x3*complexconjugate(Rd1x1)*complexconjugate(Rd3x3))/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*Rd3x3*complexconjugate(Rd1x1)*complexconjugate(Rd3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_275 = Coupling(name = 'GC_275',
                  value = '(complex(0,1)*G**2*Rd1x1*Rd3x3*complexconjugate(Rd1x1)*complexconjugate(Rd3x3))/(-1 + sw**2) - (complex(0,1)*G**2*Rd1x1*Rd3x3*sw**2*complexconjugate(Rd1x1)*complexconjugate(Rd3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_276 = Coupling(name = 'GC_276',
                  value = '(-2*ee**2*complex(0,1)*Rd2x2*Rd3x3*complexconjugate(Rd2x2)*complexconjugate(Rd3x3))/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*Rd3x3*complexconjugate(Rd2x2)*complexconjugate(Rd3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_277 = Coupling(name = 'GC_277',
                  value = '(complex(0,1)*G**2*Rd2x2*Rd3x3*complexconjugate(Rd2x2)*complexconjugate(Rd3x3))/(-1 + sw**2) - (complex(0,1)*G**2*Rd2x2*Rd3x3*sw**2*complexconjugate(Rd2x2)*complexconjugate(Rd3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_278 = Coupling(name = 'GC_278',
                  value = '(-2*ee**2*complex(0,1)*Rd3x3**2*complexconjugate(Rd3x3)**2)/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3**2*complexconjugate(Rd3x3)**2)/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_279 = Coupling(name = 'GC_279',
                  value = '(complex(0,1)*G**2*Rd3x3**2*complexconjugate(Rd3x3)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Rd3x3**2*sw**2*complexconjugate(Rd3x3)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_28 = Coupling(name = 'GC_28',
                 value = '(2*ee**2*complex(0,1))/sw**2',
                 order = {'QED':2})

GC_280 = Coupling(name = 'GC_280',
                  value = 'complex(0,1)*G*complexconjugate(Rd4x4)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_281 = Coupling(name = 'GC_281',
                  value = '(ee*complex(0,1)*Rd4x4*complexconjugate(Rd4x4))/3.',
                  order = {'QED':1})

GC_282 = Coupling(name = 'GC_282',
                  value = '(2*ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4))/9.',
                  order = {'QED':2})

GC_283 = Coupling(name = 'GC_283',
                  value = '-(complex(0,1)*G*Rd4x4*complexconjugate(Rd4x4))',
                  order = {'QCD':1})

GC_284 = Coupling(name = 'GC_284',
                  value = 'complex(0,1)*G*Rd4x4*complexconjugate(Rd4x4)',
                  order = {'QCD':1})

GC_285 = Coupling(name = 'GC_285',
                  value = '(-2*ee*complex(0,1)*G*Rd4x4*complexconjugate(Rd4x4))/3.',
                  order = {'QCD':1,'QED':1})

GC_286 = Coupling(name = 'GC_286',
                  value = 'complex(0,1)*G**2*Rd4x4*complexconjugate(Rd4x4)',
                  order = {'QCD':2})

GC_287 = Coupling(name = 'GC_287',
                  value = '(2*ee**2*complex(0,1)*Rd4x4*sw**2*complexconjugate(Rd4x4))/(9 - 9*sw**2)',
                  order = {'QED':2})

GC_288 = Coupling(name = 'GC_288',
                  value = '(ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_289 = Coupling(name = 'GC_289',
                  value = '(2*cw*ee*complex(0,1)*G*Rd4x4*sw*complexconjugate(Rd4x4))/(3 - 3*sw**2)',
                  order = {'QCD':1,'QED':1})

GC_29 = Coupling(name = 'GC_29',
                 value = '-ee/(2.*sw)',
                 order = {'QED':1})

GC_290 = Coupling(name = 'GC_290',
                  value = '-(cw*ee*complex(0,1)*Rd4x4*sw*complexconjugate(Rd4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_291 = Coupling(name = 'GC_291',
                  value = '(cw*ee*complex(0,1)*Rd4x4*sw*complexconjugate(Rd4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_292 = Coupling(name = 'GC_292',
                  value = '(2*cw*ee**2*complex(0,1)*Rd4x4*sw*complexconjugate(Rd4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_293 = Coupling(name = 'GC_293',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_294 = Coupling(name = 'GC_294',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_295 = Coupling(name = 'GC_295',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_296 = Coupling(name = 'GC_296',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rd4x4)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_297 = Coupling(name = 'GC_297',
                  value = '(ee**2*complex(0,1)*Rd1x1*Rd4x4*complexconjugate(Rd1x1)*complexconjugate(Rd4x4))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_298 = Coupling(name = 'GC_298',
                  value = '(ee**2*complex(0,1)*Rd2x2*Rd4x4*complexconjugate(Rd2x2)*complexconjugate(Rd4x4))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_299 = Coupling(name = 'GC_299',
                  value = '(ee**2*complex(0,1)*Rd3x3*Rd4x4*complexconjugate(Rd3x3)*complexconjugate(Rd4x4))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_3 = Coupling(name = 'GC_3',
                value = '-(ee*complex(0,1))',
                order = {'QED':1})

GC_30 = Coupling(name = 'GC_30',
                 value = 'ee/(2.*sw)',
                 order = {'QED':1})

GC_300 = Coupling(name = 'GC_300',
                  value = '(ee**2*complex(0,1)*Rd4x4**2*complexconjugate(Rd4x4)**2)/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_301 = Coupling(name = 'GC_301',
                  value = '-((complex(0,1)*G**2*Rd1x1*Rd4x4*complexconjugate(Rd1x1)*complexconjugate(Rd4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd1x1*Rd4x4*sw**2*complexconjugate(Rd1x1)*complexconjugate(Rd4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_302 = Coupling(name = 'GC_302',
                  value = '-((complex(0,1)*G**2*Rd2x2*Rd4x4*complexconjugate(Rd2x2)*complexconjugate(Rd4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd2x2*Rd4x4*sw**2*complexconjugate(Rd2x2)*complexconjugate(Rd4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_303 = Coupling(name = 'GC_303',
                  value = '-((complex(0,1)*G**2*Rd3x3*Rd4x4*complexconjugate(Rd3x3)*complexconjugate(Rd4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd3x3*Rd4x4*sw**2*complexconjugate(Rd3x3)*complexconjugate(Rd4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_304 = Coupling(name = 'GC_304',
                  value = '(complex(0,1)*G**2*Rd4x4**2*complexconjugate(Rd4x4)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Rd4x4**2*sw**2*complexconjugate(Rd4x4)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_305 = Coupling(name = 'GC_305',
                  value = 'complex(0,1)*G*complexconjugate(Rd5x5)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_306 = Coupling(name = 'GC_306',
                  value = '(ee*complex(0,1)*Rd5x5*complexconjugate(Rd5x5))/3.',
                  order = {'QED':1})

GC_307 = Coupling(name = 'GC_307',
                  value = '(2*ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5))/9.',
                  order = {'QED':2})

GC_308 = Coupling(name = 'GC_308',
                  value = '-(complex(0,1)*G*Rd5x5*complexconjugate(Rd5x5))',
                  order = {'QCD':1})

GC_309 = Coupling(name = 'GC_309',
                  value = 'complex(0,1)*G*Rd5x5*complexconjugate(Rd5x5)',
                  order = {'QCD':1})

GC_31 = Coupling(name = 'GC_31',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_310 = Coupling(name = 'GC_310',
                  value = '(-2*ee*complex(0,1)*G*Rd5x5*complexconjugate(Rd5x5))/3.',
                  order = {'QCD':1,'QED':1})

GC_311 = Coupling(name = 'GC_311',
                  value = 'complex(0,1)*G**2*Rd5x5*complexconjugate(Rd5x5)',
                  order = {'QCD':2})

GC_312 = Coupling(name = 'GC_312',
                  value = '(2*ee**2*complex(0,1)*Rd5x5*sw**2*complexconjugate(Rd5x5))/(9 - 9*sw**2)',
                  order = {'QED':2})

GC_313 = Coupling(name = 'GC_313',
                  value = '(ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_314 = Coupling(name = 'GC_314',
                  value = '(2*cw*ee*complex(0,1)*G*Rd5x5*sw*complexconjugate(Rd5x5))/(3 - 3*sw**2)',
                  order = {'QCD':1,'QED':1})

GC_315 = Coupling(name = 'GC_315',
                  value = '-(cw*ee*complex(0,1)*Rd5x5*sw*complexconjugate(Rd5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_316 = Coupling(name = 'GC_316',
                  value = '(cw*ee*complex(0,1)*Rd5x5*sw*complexconjugate(Rd5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_317 = Coupling(name = 'GC_317',
                  value = '(2*cw*ee**2*complex(0,1)*Rd5x5*sw*complexconjugate(Rd5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_318 = Coupling(name = 'GC_318',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_319 = Coupling(name = 'GC_319',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_32 = Coupling(name = 'GC_32',
                 value = '(CKM1x1*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_320 = Coupling(name = 'GC_320',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_321 = Coupling(name = 'GC_321',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rd5x5)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_322 = Coupling(name = 'GC_322',
                  value = '(ee**2*complex(0,1)*Rd1x1*Rd5x5*complexconjugate(Rd1x1)*complexconjugate(Rd5x5))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_323 = Coupling(name = 'GC_323',
                  value = '(ee**2*complex(0,1)*Rd2x2*Rd5x5*complexconjugate(Rd2x2)*complexconjugate(Rd5x5))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_324 = Coupling(name = 'GC_324',
                  value = '(ee**2*complex(0,1)*Rd3x3*Rd5x5*complexconjugate(Rd3x3)*complexconjugate(Rd5x5))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_325 = Coupling(name = 'GC_325',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rd5x5*complexconjugate(Rd4x4)*complexconjugate(Rd5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_326 = Coupling(name = 'GC_326',
                  value = '(ee**2*complex(0,1)*Rd5x5**2*complexconjugate(Rd5x5)**2)/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_327 = Coupling(name = 'GC_327',
                  value = '-((complex(0,1)*G**2*Rd1x1*Rd5x5*complexconjugate(Rd1x1)*complexconjugate(Rd5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd1x1*Rd5x5*sw**2*complexconjugate(Rd1x1)*complexconjugate(Rd5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_328 = Coupling(name = 'GC_328',
                  value = '-((complex(0,1)*G**2*Rd2x2*Rd5x5*complexconjugate(Rd2x2)*complexconjugate(Rd5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd2x2*Rd5x5*sw**2*complexconjugate(Rd2x2)*complexconjugate(Rd5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_329 = Coupling(name = 'GC_329',
                  value = '-((complex(0,1)*G**2*Rd3x3*Rd5x5*complexconjugate(Rd3x3)*complexconjugate(Rd5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd3x3*Rd5x5*sw**2*complexconjugate(Rd3x3)*complexconjugate(Rd5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_33 = Coupling(name = 'GC_33',
                 value = '(CKM2x2*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_330 = Coupling(name = 'GC_330',
                  value = '(complex(0,1)*G**2*Rd4x4*Rd5x5*complexconjugate(Rd4x4)*complexconjugate(Rd5x5))/(-1 + sw**2) - (complex(0,1)*G**2*Rd4x4*Rd5x5*sw**2*complexconjugate(Rd4x4)*complexconjugate(Rd5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_331 = Coupling(name = 'GC_331',
                  value = '(complex(0,1)*G**2*Rd5x5**2*complexconjugate(Rd5x5)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Rd5x5**2*sw**2*complexconjugate(Rd5x5)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_332 = Coupling(name = 'GC_332',
                  value = 'complex(0,1)*G*complexconjugate(Rd6x6)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_333 = Coupling(name = 'GC_333',
                  value = '(ee*complex(0,1)*Rd6x6*complexconjugate(Rd6x6))/3.',
                  order = {'QED':1})

GC_334 = Coupling(name = 'GC_334',
                  value = '(2*ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6))/9.',
                  order = {'QED':2})

GC_335 = Coupling(name = 'GC_335',
                  value = '-(complex(0,1)*G*Rd6x6*complexconjugate(Rd6x6))',
                  order = {'QCD':1})

GC_336 = Coupling(name = 'GC_336',
                  value = 'complex(0,1)*G*Rd6x6*complexconjugate(Rd6x6)',
                  order = {'QCD':1})

GC_337 = Coupling(name = 'GC_337',
                  value = '(-2*ee*complex(0,1)*G*Rd6x6*complexconjugate(Rd6x6))/3.',
                  order = {'QCD':1,'QED':1})

GC_338 = Coupling(name = 'GC_338',
                  value = 'complex(0,1)*G**2*Rd6x6*complexconjugate(Rd6x6)',
                  order = {'QCD':2})

GC_339 = Coupling(name = 'GC_339',
                  value = '(2*ee**2*complex(0,1)*Rd6x6*sw**2*complexconjugate(Rd6x6))/(9 - 9*sw**2)',
                  order = {'QED':2})

GC_34 = Coupling(name = 'GC_34',
                 value = '(CKM3x3*ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_340 = Coupling(name = 'GC_340',
                  value = '(ee**2*complex(0,1)*Rd6x6*complexconjugate(Rd6x6))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_341 = Coupling(name = 'GC_341',
                  value = '(2*cw*ee*complex(0,1)*G*Rd6x6*sw*complexconjugate(Rd6x6))/(3 - 3*sw**2)',
                  order = {'QCD':1,'QED':1})

GC_342 = Coupling(name = 'GC_342',
                  value = '-(cw*ee*complex(0,1)*Rd6x6*sw*complexconjugate(Rd6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_343 = Coupling(name = 'GC_343',
                  value = '(cw*ee*complex(0,1)*Rd6x6*sw*complexconjugate(Rd6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_344 = Coupling(name = 'GC_344',
                  value = '(2*cw*ee**2*complex(0,1)*Rd6x6*sw*complexconjugate(Rd6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_345 = Coupling(name = 'GC_345',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_346 = Coupling(name = 'GC_346',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_347 = Coupling(name = 'GC_347',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_348 = Coupling(name = 'GC_348',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rd6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_349 = Coupling(name = 'GC_349',
                  value = '(ee**2*complex(0,1)*Rd1x1*Rd6x6*complexconjugate(Rd1x1)*complexconjugate(Rd6x6))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_35 = Coupling(name = 'GC_35',
                 value = '-((cw*ee*complex(0,1))/sw)',
                 order = {'QED':1})

GC_350 = Coupling(name = 'GC_350',
                  value = '(ee**2*complex(0,1)*Rd2x2*Rd6x6*complexconjugate(Rd2x2)*complexconjugate(Rd6x6))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_351 = Coupling(name = 'GC_351',
                  value = '(ee**2*complex(0,1)*Rd3x3*Rd6x6*complexconjugate(Rd3x3)*complexconjugate(Rd6x6))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_352 = Coupling(name = 'GC_352',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rd6x6*complexconjugate(Rd4x4)*complexconjugate(Rd6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_353 = Coupling(name = 'GC_353',
                  value = '(ee**2*complex(0,1)*Rd5x5*Rd6x6*complexconjugate(Rd5x5)*complexconjugate(Rd6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_354 = Coupling(name = 'GC_354',
                  value = '(ee**2*complex(0,1)*Rd6x6**2*complexconjugate(Rd6x6)**2)/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_355 = Coupling(name = 'GC_355',
                  value = '-((complex(0,1)*G**2*Rd1x1*Rd6x6*complexconjugate(Rd1x1)*complexconjugate(Rd6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd1x1*Rd6x6*sw**2*complexconjugate(Rd1x1)*complexconjugate(Rd6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_356 = Coupling(name = 'GC_356',
                  value = '-((complex(0,1)*G**2*Rd2x2*Rd6x6*complexconjugate(Rd2x2)*complexconjugate(Rd6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd2x2*Rd6x6*sw**2*complexconjugate(Rd2x2)*complexconjugate(Rd6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_357 = Coupling(name = 'GC_357',
                  value = '-((complex(0,1)*G**2*Rd3x3*Rd6x6*complexconjugate(Rd3x3)*complexconjugate(Rd6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd3x3*Rd6x6*sw**2*complexconjugate(Rd3x3)*complexconjugate(Rd6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_358 = Coupling(name = 'GC_358',
                  value = '(complex(0,1)*G**2*Rd4x4*Rd6x6*complexconjugate(Rd4x4)*complexconjugate(Rd6x6))/(-1 + sw**2) - (complex(0,1)*G**2*Rd4x4*Rd6x6*sw**2*complexconjugate(Rd4x4)*complexconjugate(Rd6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_359 = Coupling(name = 'GC_359',
                  value = '(complex(0,1)*G**2*Rd5x5*Rd6x6*complexconjugate(Rd5x5)*complexconjugate(Rd6x6))/(-1 + sw**2) - (complex(0,1)*G**2*Rd5x5*Rd6x6*sw**2*complexconjugate(Rd5x5)*complexconjugate(Rd6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_36 = Coupling(name = 'GC_36',
                 value = '(cw*ee*complex(0,1))/sw',
                 order = {'QED':1})

GC_360 = Coupling(name = 'GC_360',
                  value = '(complex(0,1)*G**2*Rd6x6**2*complexconjugate(Rd6x6)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Rd6x6**2*sw**2*complexconjugate(Rd6x6)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_361 = Coupling(name = 'GC_361',
                  value = 'ee*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)',
                  order = {'QED':1})

GC_362 = Coupling(name = 'GC_362',
                  value = '2*ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1)',
                  order = {'QED':2})

GC_363 = Coupling(name = 'GC_363',
                  value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_364 = Coupling(name = 'GC_364',
                  value = '-((ee*complex(0,1)*Rn1x1*complexconjugate(Rl1x1))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_365 = Coupling(name = 'GC_365',
                  value = '(ee*complex(0,1)*Rn1x1*complexconjugate(Rl1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_366 = Coupling(name = 'GC_366',
                  value = '-((ee**2*complex(0,1)*Rn1x1*complexconjugate(Rl1x1))/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_367 = Coupling(name = 'GC_367',
                  value = '-((cw*ee**2*complex(0,1)*Rn1x1*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_368 = Coupling(name = 'GC_368',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(Rl1x1))/sw)',
                  order = {'QED':1})

GC_369 = Coupling(name = 'GC_369',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(Rl1x1))/sw)',
                  order = {'QED':1})

GC_37 = Coupling(name = 'GC_37',
                 value = '-ee**2/(2.*sw)',
                 order = {'QED':2})

GC_370 = Coupling(name = 'GC_370',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rl1x1*complexconjugate(Rd4x4)*complexconjugate(Rl1x1))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_371 = Coupling(name = 'GC_371',
                  value = '(ee**2*complex(0,1)*Rd5x5*Rl1x1*complexconjugate(Rd5x5)*complexconjugate(Rl1x1))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_372 = Coupling(name = 'GC_372',
                  value = '(ee**2*complex(0,1)*Rd6x6*Rl1x1*complexconjugate(Rd6x6)*complexconjugate(Rl1x1))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_373 = Coupling(name = 'GC_373',
                  value = '(ee**2*complex(0,1)*Rl1x1**2*complexconjugate(Rl1x1)**2)/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_374 = Coupling(name = 'GC_374',
                  value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_375 = Coupling(name = 'GC_375',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN1x2*complexconjugate(Rl1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_376 = Coupling(name = 'GC_376',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN2x2*complexconjugate(Rl1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_377 = Coupling(name = 'GC_377',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN3x2*complexconjugate(Rl1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_378 = Coupling(name = 'GC_378',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN4x2*complexconjugate(Rl1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rl1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_379 = Coupling(name = 'GC_379',
                  value = '(cw*ee*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*Rl1x1*sw*complexconjugate(Rl1x1))/(-1 + sw**2)',
                  order = {'QED':1})

GC_38 = Coupling(name = 'GC_38',
                 value = 'ee**2/(2.*sw)',
                 order = {'QED':2})

GC_380 = Coupling(name = 'GC_380',
                  value = '-(cw*ee*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*Rl1x1*sw*complexconjugate(Rl1x1))/(-1 + sw**2)',
                  order = {'QED':1})

GC_381 = Coupling(name = 'GC_381',
                  value = '-((cw*ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(sw*(-1 + sw**2))) + (2*cw*ee**2*complex(0,1)*Rl1x1*sw*complexconjugate(Rl1x1))/(-1 + sw**2)',
                  order = {'QED':2})

GC_382 = Coupling(name = 'GC_382',
                  value = '(2*ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(-1 + sw**2) - (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(2.*sw**2*(-1 + sw**2)) - (2*ee**2*complex(0,1)*Rl1x1*sw**2*complexconjugate(Rl1x1))/(-1 + sw**2)',
                  order = {'QED':2})

GC_383 = Coupling(name = 'GC_383',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Rl1x1*complexconjugate(Rd1x1)*complexconjugate(Rl1x1))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*Rl1x1*complexconjugate(Rd1x1)*complexconjugate(Rl1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_384 = Coupling(name = 'GC_384',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Rl1x1*complexconjugate(Rd2x2)*complexconjugate(Rl1x1))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*Rl1x1*complexconjugate(Rd2x2)*complexconjugate(Rl1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_385 = Coupling(name = 'GC_385',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Rl1x1*complexconjugate(Rd3x3)*complexconjugate(Rl1x1))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*Rl1x1*complexconjugate(Rd3x3)*complexconjugate(Rl1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_386 = Coupling(name = 'GC_386',
                  value = 'ee*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)',
                  order = {'QED':1})

GC_387 = Coupling(name = 'GC_387',
                  value = '2*ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2)',
                  order = {'QED':2})

GC_388 = Coupling(name = 'GC_388',
                  value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_389 = Coupling(name = 'GC_389',
                  value = '-((ee*complex(0,1)*Rn2x2*complexconjugate(Rl2x2))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_39 = Coupling(name = 'GC_39',
                 value = '(cw*ee**2*complex(0,1))/sw',
                 order = {'QED':2})

GC_390 = Coupling(name = 'GC_390',
                  value = '(ee*complex(0,1)*Rn2x2*complexconjugate(Rl2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_391 = Coupling(name = 'GC_391',
                  value = '-((ee**2*complex(0,1)*Rn2x2*complexconjugate(Rl2x2))/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_392 = Coupling(name = 'GC_392',
                  value = '-((cw*ee**2*complex(0,1)*Rn2x2*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_393 = Coupling(name = 'GC_393',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(Rl2x2))/sw)',
                  order = {'QED':1})

GC_394 = Coupling(name = 'GC_394',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(Rl2x2))/sw)',
                  order = {'QED':1})

GC_395 = Coupling(name = 'GC_395',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rl2x2*complexconjugate(Rd4x4)*complexconjugate(Rl2x2))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_396 = Coupling(name = 'GC_396',
                  value = '(ee**2*complex(0,1)*Rd5x5*Rl2x2*complexconjugate(Rd5x5)*complexconjugate(Rl2x2))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_397 = Coupling(name = 'GC_397',
                  value = '(ee**2*complex(0,1)*Rd6x6*Rl2x2*complexconjugate(Rd6x6)*complexconjugate(Rl2x2))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_398 = Coupling(name = 'GC_398',
                  value = '(ee**2*complex(0,1)*Rl1x1*Rl2x2*complexconjugate(Rl1x1)*complexconjugate(Rl2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_399 = Coupling(name = 'GC_399',
                  value = '(ee**2*complex(0,1)*Rl2x2**2*complexconjugate(Rl2x2)**2)/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_4 = Coupling(name = 'GC_4',
                value = 'ee*complex(0,1)',
                order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(-2*cw*ee**2*complex(0,1))/sw',
                 order = {'QED':2})

GC_400 = Coupling(name = 'GC_400',
                  value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_401 = Coupling(name = 'GC_401',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN1x2*complexconjugate(Rl2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_402 = Coupling(name = 'GC_402',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN2x2*complexconjugate(Rl2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_403 = Coupling(name = 'GC_403',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN3x2*complexconjugate(Rl2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_404 = Coupling(name = 'GC_404',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN4x2*complexconjugate(Rl2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rl2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_405 = Coupling(name = 'GC_405',
                  value = '(cw*ee*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*Rl2x2*sw*complexconjugate(Rl2x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_406 = Coupling(name = 'GC_406',
                  value = '-(cw*ee*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*Rl2x2*sw*complexconjugate(Rl2x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_407 = Coupling(name = 'GC_407',
                  value = '-((cw*ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(sw*(-1 + sw**2))) + (2*cw*ee**2*complex(0,1)*Rl2x2*sw*complexconjugate(Rl2x2))/(-1 + sw**2)',
                  order = {'QED':2})

GC_408 = Coupling(name = 'GC_408',
                  value = '(2*ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(-1 + sw**2) - (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(2.*sw**2*(-1 + sw**2)) - (2*ee**2*complex(0,1)*Rl2x2*sw**2*complexconjugate(Rl2x2))/(-1 + sw**2)',
                  order = {'QED':2})

GC_409 = Coupling(name = 'GC_409',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Rl2x2*complexconjugate(Rd1x1)*complexconjugate(Rl2x2))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*Rl2x2*complexconjugate(Rd1x1)*complexconjugate(Rl2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_41 = Coupling(name = 'GC_41',
                 value = '(2*cw*ee*complex(0,1)*NN1x1*Ru4x4*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_410 = Coupling(name = 'GC_410',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Rl2x2*complexconjugate(Rd2x2)*complexconjugate(Rl2x2))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*Rl2x2*complexconjugate(Rd2x2)*complexconjugate(Rl2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_411 = Coupling(name = 'GC_411',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Rl2x2*complexconjugate(Rd3x3)*complexconjugate(Rl2x2))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*Rl2x2*complexconjugate(Rd3x3)*complexconjugate(Rl2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_412 = Coupling(name = 'GC_412',
                  value = 'ee*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)',
                  order = {'QED':1})

GC_413 = Coupling(name = 'GC_413',
                  value = '2*ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3)',
                  order = {'QED':2})

GC_414 = Coupling(name = 'GC_414',
                  value = '(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_415 = Coupling(name = 'GC_415',
                  value = '-((ee*complex(0,1)*Rn3x3*complexconjugate(Rl3x3))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_416 = Coupling(name = 'GC_416',
                  value = '(ee*complex(0,1)*Rn3x3*complexconjugate(Rl3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_417 = Coupling(name = 'GC_417',
                  value = '-((ee**2*complex(0,1)*Rn3x3*complexconjugate(Rl3x3))/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_418 = Coupling(name = 'GC_418',
                  value = '-((cw*ee**2*complex(0,1)*Rn3x3*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_419 = Coupling(name = 'GC_419',
                  value = '-((ee*complex(0,1)*UU1x1*complexconjugate(Rl3x3))/sw)',
                  order = {'QED':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '(2*cw*ee*complex(0,1)*NN2x1*Ru4x4*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_420 = Coupling(name = 'GC_420',
                  value = '-((ee*complex(0,1)*UU2x1*complexconjugate(Rl3x3))/sw)',
                  order = {'QED':1})

GC_421 = Coupling(name = 'GC_421',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rl3x3*complexconjugate(Rd4x4)*complexconjugate(Rl3x3))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_422 = Coupling(name = 'GC_422',
                  value = '(ee**2*complex(0,1)*Rd5x5*Rl3x3*complexconjugate(Rd5x5)*complexconjugate(Rl3x3))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_423 = Coupling(name = 'GC_423',
                  value = '(ee**2*complex(0,1)*Rd6x6*Rl3x3*complexconjugate(Rd6x6)*complexconjugate(Rl3x3))/(6 - 6*sw**2)',
                  order = {'QED':2})

GC_424 = Coupling(name = 'GC_424',
                  value = '(ee**2*complex(0,1)*Rl1x1*Rl3x3*complexconjugate(Rl1x1)*complexconjugate(Rl3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_425 = Coupling(name = 'GC_425',
                  value = '(ee**2*complex(0,1)*Rl2x2*Rl3x3*complexconjugate(Rl2x2)*complexconjugate(Rl3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_426 = Coupling(name = 'GC_426',
                  value = '(ee**2*complex(0,1)*Rl3x3**2*complexconjugate(Rl3x3)**2)/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_427 = Coupling(name = 'GC_427',
                  value = '(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_428 = Coupling(name = 'GC_428',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN1x2*complexconjugate(Rl3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_429 = Coupling(name = 'GC_429',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN2x2*complexconjugate(Rl3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '(2*cw*ee*complex(0,1)*NN3x1*Ru4x4*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_430 = Coupling(name = 'GC_430',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN3x2*complexconjugate(Rl3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_431 = Coupling(name = 'GC_431',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))) - (ee*complex(0,1)*NN4x2*complexconjugate(Rl3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rl3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_432 = Coupling(name = 'GC_432',
                  value = '(cw*ee*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*Rl3x3*sw*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_433 = Coupling(name = 'GC_433',
                  value = '-(cw*ee*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*Rl3x3*sw*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_434 = Coupling(name = 'GC_434',
                  value = '-((cw*ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(sw*(-1 + sw**2))) + (2*cw*ee**2*complex(0,1)*Rl3x3*sw*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_435 = Coupling(name = 'GC_435',
                  value = '(2*ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(-1 + sw**2) - (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(2.*sw**2*(-1 + sw**2)) - (2*ee**2*complex(0,1)*Rl3x3*sw**2*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_436 = Coupling(name = 'GC_436',
                  value = '(complex(0,1)*ye3x3*complexconjugate(NN1x3)*complexconjugate(Rl3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*ye3x3*complexconjugate(NN1x3)*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_437 = Coupling(name = 'GC_437',
                  value = '(complex(0,1)*ye3x3*complexconjugate(NN2x3)*complexconjugate(Rl3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*ye3x3*complexconjugate(NN2x3)*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_438 = Coupling(name = 'GC_438',
                  value = '(complex(0,1)*ye3x3*complexconjugate(NN3x3)*complexconjugate(Rl3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*ye3x3*complexconjugate(NN3x3)*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_439 = Coupling(name = 'GC_439',
                  value = '(complex(0,1)*ye3x3*complexconjugate(NN4x3)*complexconjugate(Rl3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*ye3x3*complexconjugate(NN4x3)*complexconjugate(Rl3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_44 = Coupling(name = 'GC_44',
                 value = '(2*cw*ee*complex(0,1)*NN4x1*Ru4x4*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_440 = Coupling(name = 'GC_440',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Rl3x3*complexconjugate(Rd1x1)*complexconjugate(Rl3x3))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*Rl3x3*complexconjugate(Rd1x1)*complexconjugate(Rl3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_441 = Coupling(name = 'GC_441',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Rl3x3*complexconjugate(Rd2x2)*complexconjugate(Rl3x3))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd2x2*Rl3x3*complexconjugate(Rd2x2)*complexconjugate(Rl3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_442 = Coupling(name = 'GC_442',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Rl3x3*complexconjugate(Rd3x3)*complexconjugate(Rl3x3))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd3x3*Rl3x3*complexconjugate(Rd3x3)*complexconjugate(Rl3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_443 = Coupling(name = 'GC_443',
                  value = 'ee*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)',
                  order = {'QED':1})

GC_444 = Coupling(name = 'GC_444',
                  value = '2*ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)',
                  order = {'QED':2})

GC_445 = Coupling(name = 'GC_445',
                  value = '(ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_446 = Coupling(name = 'GC_446',
                  value = '-((cw*ee*complex(0,1)*Rl4x4*sw*complexconjugate(Rl4x4))/(-1 + sw**2))',
                  order = {'QED':1})

GC_447 = Coupling(name = 'GC_447',
                  value = '(cw*ee*complex(0,1)*Rl4x4*sw*complexconjugate(Rl4x4))/(-1 + sw**2)',
                  order = {'QED':1})

GC_448 = Coupling(name = 'GC_448',
                  value = '(2*cw*ee**2*complex(0,1)*Rl4x4*sw*complexconjugate(Rl4x4))/(-1 + sw**2)',
                  order = {'QED':2})

GC_449 = Coupling(name = 'GC_449',
                  value = '(-2*ee**2*complex(0,1)*Rl4x4*sw**2*complexconjugate(Rl4x4))/(-1 + sw**2)',
                  order = {'QED':2})

GC_45 = Coupling(name = 'GC_45',
                 value = '(2*cw*ee*complex(0,1)*NN1x1*Ru5x5*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_450 = Coupling(name = 'GC_450',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_451 = Coupling(name = 'GC_451',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_452 = Coupling(name = 'GC_452',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_453 = Coupling(name = 'GC_453',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rl4x4)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_454 = Coupling(name = 'GC_454',
                  value = '(ee**2*complex(0,1)*Rd1x1*Rl4x4*complexconjugate(Rd1x1)*complexconjugate(Rl4x4))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_455 = Coupling(name = 'GC_455',
                  value = '(ee**2*complex(0,1)*Rd2x2*Rl4x4*complexconjugate(Rd2x2)*complexconjugate(Rl4x4))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_456 = Coupling(name = 'GC_456',
                  value = '(ee**2*complex(0,1)*Rd3x3*Rl4x4*complexconjugate(Rd3x3)*complexconjugate(Rl4x4))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_457 = Coupling(name = 'GC_457',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rl4x4*complexconjugate(Rd4x4)*complexconjugate(Rl4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_458 = Coupling(name = 'GC_458',
                  value = '(ee**2*complex(0,1)*Rd5x5*Rl4x4*complexconjugate(Rd5x5)*complexconjugate(Rl4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_459 = Coupling(name = 'GC_459',
                  value = '(ee**2*complex(0,1)*Rd6x6*Rl4x4*complexconjugate(Rd6x6)*complexconjugate(Rl4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_46 = Coupling(name = 'GC_46',
                 value = '(2*cw*ee*complex(0,1)*NN2x1*Ru5x5*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_460 = Coupling(name = 'GC_460',
                  value = '(ee**2*complex(0,1)*Rl1x1*Rl4x4*complexconjugate(Rl1x1)*complexconjugate(Rl4x4))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_461 = Coupling(name = 'GC_461',
                  value = '(ee**2*complex(0,1)*Rl2x2*Rl4x4*complexconjugate(Rl2x2)*complexconjugate(Rl4x4))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_462 = Coupling(name = 'GC_462',
                  value = '(ee**2*complex(0,1)*Rl3x3*Rl4x4*complexconjugate(Rl3x3)*complexconjugate(Rl4x4))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_463 = Coupling(name = 'GC_463',
                  value = '(2*ee**2*complex(0,1)*Rl4x4**2*complexconjugate(Rl4x4)**2)/(-1 + sw**2)',
                  order = {'QED':2})

GC_464 = Coupling(name = 'GC_464',
                  value = 'ee*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)',
                  order = {'QED':1})

GC_465 = Coupling(name = 'GC_465',
                  value = '2*ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)',
                  order = {'QED':2})

GC_466 = Coupling(name = 'GC_466',
                  value = '(ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_467 = Coupling(name = 'GC_467',
                  value = '-((cw*ee*complex(0,1)*Rl5x5*sw*complexconjugate(Rl5x5))/(-1 + sw**2))',
                  order = {'QED':1})

GC_468 = Coupling(name = 'GC_468',
                  value = '(cw*ee*complex(0,1)*Rl5x5*sw*complexconjugate(Rl5x5))/(-1 + sw**2)',
                  order = {'QED':1})

GC_469 = Coupling(name = 'GC_469',
                  value = '(2*cw*ee**2*complex(0,1)*Rl5x5*sw*complexconjugate(Rl5x5))/(-1 + sw**2)',
                  order = {'QED':2})

GC_47 = Coupling(name = 'GC_47',
                 value = '(2*cw*ee*complex(0,1)*NN3x1*Ru5x5*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_470 = Coupling(name = 'GC_470',
                  value = '(-2*ee**2*complex(0,1)*Rl5x5*sw**2*complexconjugate(Rl5x5))/(-1 + sw**2)',
                  order = {'QED':2})

GC_471 = Coupling(name = 'GC_471',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_472 = Coupling(name = 'GC_472',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_473 = Coupling(name = 'GC_473',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_474 = Coupling(name = 'GC_474',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rl5x5)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_475 = Coupling(name = 'GC_475',
                  value = '(ee**2*complex(0,1)*Rd1x1*Rl5x5*complexconjugate(Rd1x1)*complexconjugate(Rl5x5))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_476 = Coupling(name = 'GC_476',
                  value = '(ee**2*complex(0,1)*Rd2x2*Rl5x5*complexconjugate(Rd2x2)*complexconjugate(Rl5x5))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_477 = Coupling(name = 'GC_477',
                  value = '(ee**2*complex(0,1)*Rd3x3*Rl5x5*complexconjugate(Rd3x3)*complexconjugate(Rl5x5))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_478 = Coupling(name = 'GC_478',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rl5x5*complexconjugate(Rd4x4)*complexconjugate(Rl5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_479 = Coupling(name = 'GC_479',
                  value = '(ee**2*complex(0,1)*Rd5x5*Rl5x5*complexconjugate(Rd5x5)*complexconjugate(Rl5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_48 = Coupling(name = 'GC_48',
                 value = '(2*cw*ee*complex(0,1)*NN4x1*Ru5x5*cmath.sqrt(2))/(3 - 3*sw**2)',
                 order = {'QED':1})

GC_480 = Coupling(name = 'GC_480',
                  value = '(ee**2*complex(0,1)*Rd6x6*Rl5x5*complexconjugate(Rd6x6)*complexconjugate(Rl5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_481 = Coupling(name = 'GC_481',
                  value = '(ee**2*complex(0,1)*Rl1x1*Rl5x5*complexconjugate(Rl1x1)*complexconjugate(Rl5x5))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_482 = Coupling(name = 'GC_482',
                  value = '(ee**2*complex(0,1)*Rl2x2*Rl5x5*complexconjugate(Rl2x2)*complexconjugate(Rl5x5))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_483 = Coupling(name = 'GC_483',
                  value = '(ee**2*complex(0,1)*Rl3x3*Rl5x5*complexconjugate(Rl3x3)*complexconjugate(Rl5x5))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_484 = Coupling(name = 'GC_484',
                  value = '(ee**2*complex(0,1)*Rl4x4*Rl5x5*complexconjugate(Rl4x4)*complexconjugate(Rl5x5))/(-1 + sw**2)',
                  order = {'QED':2})

GC_485 = Coupling(name = 'GC_485',
                  value = '(2*ee**2*complex(0,1)*Rl5x5**2*complexconjugate(Rl5x5)**2)/(-1 + sw**2)',
                  order = {'QED':2})

GC_486 = Coupling(name = 'GC_486',
                  value = 'ee*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)',
                  order = {'QED':1})

GC_487 = Coupling(name = 'GC_487',
                  value = '2*ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6)',
                  order = {'QED':2})

GC_488 = Coupling(name = 'GC_488',
                  value = '(ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_489 = Coupling(name = 'GC_489',
                  value = '-((cw*ee*complex(0,1)*Rl6x6*sw*complexconjugate(Rl6x6))/(-1 + sw**2))',
                  order = {'QED':1})

GC_49 = Coupling(name = 'GC_49',
                 value = '(cw*ee**2)/(2 - 2*sw**2)',
                 order = {'QED':2})

GC_490 = Coupling(name = 'GC_490',
                  value = '(cw*ee*complex(0,1)*Rl6x6*sw*complexconjugate(Rl6x6))/(-1 + sw**2)',
                  order = {'QED':1})

GC_491 = Coupling(name = 'GC_491',
                  value = '(2*cw*ee**2*complex(0,1)*Rl6x6*sw*complexconjugate(Rl6x6))/(-1 + sw**2)',
                  order = {'QED':2})

GC_492 = Coupling(name = 'GC_492',
                  value = '(-2*ee**2*complex(0,1)*Rl6x6*sw**2*complexconjugate(Rl6x6))/(-1 + sw**2)',
                  order = {'QED':2})

GC_493 = Coupling(name = 'GC_493',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_494 = Coupling(name = 'GC_494',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_495 = Coupling(name = 'GC_495',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_496 = Coupling(name = 'GC_496',
                  value = '(cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Rl6x6)*cmath.sqrt(2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_497 = Coupling(name = 'GC_497',
                  value = '(ee**2*complex(0,1)*Rd1x1*Rl6x6*complexconjugate(Rd1x1)*complexconjugate(Rl6x6))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_498 = Coupling(name = 'GC_498',
                  value = '(ee**2*complex(0,1)*Rd2x2*Rl6x6*complexconjugate(Rd2x2)*complexconjugate(Rl6x6))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_499 = Coupling(name = 'GC_499',
                  value = '(ee**2*complex(0,1)*Rd3x3*Rl6x6*complexconjugate(Rd3x3)*complexconjugate(Rl6x6))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_5 = Coupling(name = 'GC_5',
                value = 'ee**2*complex(0,1)',
                order = {'QED':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '(cw*ee*complex(0,1)*NN1x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_500 = Coupling(name = 'GC_500',
                  value = '(ee**2*complex(0,1)*Rd4x4*Rl6x6*complexconjugate(Rd4x4)*complexconjugate(Rl6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_501 = Coupling(name = 'GC_501',
                  value = '(ee**2*complex(0,1)*Rd5x5*Rl6x6*complexconjugate(Rd5x5)*complexconjugate(Rl6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_502 = Coupling(name = 'GC_502',
                  value = '(ee**2*complex(0,1)*Rd6x6*Rl6x6*complexconjugate(Rd6x6)*complexconjugate(Rl6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_503 = Coupling(name = 'GC_503',
                  value = '(ee**2*complex(0,1)*Rl1x1*Rl6x6*complexconjugate(Rl1x1)*complexconjugate(Rl6x6))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_504 = Coupling(name = 'GC_504',
                  value = '(ee**2*complex(0,1)*Rl2x2*Rl6x6*complexconjugate(Rl2x2)*complexconjugate(Rl6x6))/(2 - 2*sw**2)',
                  order = {'QED':2})

GC_505 = Coupling(name = 'GC_505',
                  value = '(ee**2*complex(0,1)*Rl4x4*Rl6x6*complexconjugate(Rl4x4)*complexconjugate(Rl6x6))/(-1 + sw**2)',
                  order = {'QED':2})

GC_506 = Coupling(name = 'GC_506',
                  value = '(ee**2*complex(0,1)*Rl5x5*Rl6x6*complexconjugate(Rl5x5)*complexconjugate(Rl6x6))/(-1 + sw**2)',
                  order = {'QED':2})

GC_507 = Coupling(name = 'GC_507',
                  value = '(2*ee**2*complex(0,1)*Rl6x6**2*complexconjugate(Rl6x6)**2)/(-1 + sw**2)',
                  order = {'QED':2})

GC_508 = Coupling(name = 'GC_508',
                  value = '-((ee*complex(0,1)*Rl1x1*complexconjugate(Rn1x1))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_509 = Coupling(name = 'GC_509',
                  value = '(ee*complex(0,1)*Rl1x1*complexconjugate(Rn1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_51 = Coupling(name = 'GC_51',
                 value = '(cw*ee*complex(0,1)*NN2x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_510 = Coupling(name = 'GC_510',
                  value = '-((ee**2*complex(0,1)*Rl1x1*complexconjugate(Rn1x1))/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_511 = Coupling(name = 'GC_511',
                  value = '-((cw*ee**2*complex(0,1)*Rl1x1*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_512 = Coupling(name = 'GC_512',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(Rn1x1))/sw)',
                  order = {'QED':1})

GC_513 = Coupling(name = 'GC_513',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(Rn1x1))/sw)',
                  order = {'QED':1})

GC_514 = Coupling(name = 'GC_514',
                  value = '-(CKM1x1*ee**2*complex(0,1)*Rl1x1*Ru1x1*complexconjugate(Rd1x1)*complexconjugate(Rn1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_515 = Coupling(name = 'GC_515',
                  value = '-(CKM2x2*ee**2*complex(0,1)*Rl1x1*Ru2x2*complexconjugate(Rd2x2)*complexconjugate(Rn1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_516 = Coupling(name = 'GC_516',
                  value = '-(CKM3x3*ee**2*complex(0,1)*Rl1x1*Ru3x3*complexconjugate(Rd3x3)*complexconjugate(Rn1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_517 = Coupling(name = 'GC_517',
                  value = '-(ee**2*complex(0,1)*Rl1x1*Rn2x2*complexconjugate(Rl2x2)*complexconjugate(Rn1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_518 = Coupling(name = 'GC_518',
                  value = '-(ee**2*complex(0,1)*Rl1x1*Rn3x3*complexconjugate(Rl3x3)*complexconjugate(Rn1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_519 = Coupling(name = 'GC_519',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN1x2*complexconjugate(Rn1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '(cw*ee*complex(0,1)*NN3x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_520 = Coupling(name = 'GC_520',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN2x2*complexconjugate(Rn1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_521 = Coupling(name = 'GC_521',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN3x2*complexconjugate(Rn1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_522 = Coupling(name = 'GC_522',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN4x2*complexconjugate(Rn1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rn1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_523 = Coupling(name = 'GC_523',
                  value = '(ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*complexconjugate(Rl1x1))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*Rn1x1*complexconjugate(Rl1x1)*complexconjugate(Rn1x1))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl1x1*Rn1x1*complexconjugate(Rl1x1)*complexconjugate(Rn1x1))/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_524 = Coupling(name = 'GC_524',
                  value = '-((ee*complex(0,1)*Rl2x2*complexconjugate(Rn2x2))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_525 = Coupling(name = 'GC_525',
                  value = '(ee*complex(0,1)*Rl2x2*complexconjugate(Rn2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_526 = Coupling(name = 'GC_526',
                  value = '-((ee**2*complex(0,1)*Rl2x2*complexconjugate(Rn2x2))/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_527 = Coupling(name = 'GC_527',
                  value = '-((cw*ee**2*complex(0,1)*Rl2x2*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_528 = Coupling(name = 'GC_528',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(Rn2x2))/sw)',
                  order = {'QED':1})

GC_529 = Coupling(name = 'GC_529',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(Rn2x2))/sw)',
                  order = {'QED':1})

GC_53 = Coupling(name = 'GC_53',
                 value = '(cw*ee*complex(0,1)*NN4x1*Rd4x4*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_530 = Coupling(name = 'GC_530',
                  value = '-(CKM1x1*ee**2*complex(0,1)*Rl2x2*Ru1x1*complexconjugate(Rd1x1)*complexconjugate(Rn2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_531 = Coupling(name = 'GC_531',
                  value = '-(CKM2x2*ee**2*complex(0,1)*Rl2x2*Ru2x2*complexconjugate(Rd2x2)*complexconjugate(Rn2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_532 = Coupling(name = 'GC_532',
                  value = '-(CKM3x3*ee**2*complex(0,1)*Rl2x2*Ru3x3*complexconjugate(Rd3x3)*complexconjugate(Rn2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_533 = Coupling(name = 'GC_533',
                  value = '-(ee**2*complex(0,1)*Rl2x2*Rn1x1*complexconjugate(Rl1x1)*complexconjugate(Rn2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_534 = Coupling(name = 'GC_534',
                  value = '-(ee**2*complex(0,1)*Rl2x2*Rn3x3*complexconjugate(Rl3x3)*complexconjugate(Rn2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_535 = Coupling(name = 'GC_535',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN1x2*complexconjugate(Rn2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_536 = Coupling(name = 'GC_536',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN2x2*complexconjugate(Rn2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_537 = Coupling(name = 'GC_537',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN3x2*complexconjugate(Rn2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_538 = Coupling(name = 'GC_538',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN4x2*complexconjugate(Rn2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rn2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_539 = Coupling(name = 'GC_539',
                  value = '(ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*complexconjugate(Rl2x2))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*Rn2x2*complexconjugate(Rl2x2)*complexconjugate(Rn2x2))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl2x2*Rn2x2*complexconjugate(Rl2x2)*complexconjugate(Rn2x2))/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_54 = Coupling(name = 'GC_54',
                 value = '(cw*ee*complex(0,1)*NN1x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_540 = Coupling(name = 'GC_540',
                  value = '-((ee*complex(0,1)*Rl3x3*complexconjugate(Rn3x3))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_541 = Coupling(name = 'GC_541',
                  value = '(ee*complex(0,1)*Rl3x3*complexconjugate(Rn3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_542 = Coupling(name = 'GC_542',
                  value = '-((ee**2*complex(0,1)*Rl3x3*complexconjugate(Rn3x3))/(sw*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_543 = Coupling(name = 'GC_543',
                  value = '-((cw*ee**2*complex(0,1)*Rl3x3*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2)))',
                  order = {'QED':2})

GC_544 = Coupling(name = 'GC_544',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(Rn3x3))/sw)',
                  order = {'QED':1})

GC_545 = Coupling(name = 'GC_545',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(Rn3x3))/sw)',
                  order = {'QED':1})

GC_546 = Coupling(name = 'GC_546',
                  value = '-(CKM1x1*ee**2*complex(0,1)*Rl3x3*Ru1x1*complexconjugate(Rd1x1)*complexconjugate(Rn3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_547 = Coupling(name = 'GC_547',
                  value = '-(CKM2x2*ee**2*complex(0,1)*Rl3x3*Ru2x2*complexconjugate(Rd2x2)*complexconjugate(Rn3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_548 = Coupling(name = 'GC_548',
                  value = '-(CKM3x3*ee**2*complex(0,1)*Rl3x3*Ru3x3*complexconjugate(Rd3x3)*complexconjugate(Rn3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_549 = Coupling(name = 'GC_549',
                  value = '-(ee**2*complex(0,1)*Rl3x3*Rn1x1*complexconjugate(Rl1x1)*complexconjugate(Rn3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_55 = Coupling(name = 'GC_55',
                 value = '(cw*ee*complex(0,1)*NN2x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_550 = Coupling(name = 'GC_550',
                  value = '-(ee**2*complex(0,1)*Rl3x3*Rn2x2*complexconjugate(Rl2x2)*complexconjugate(Rn3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_551 = Coupling(name = 'GC_551',
                  value = '-((cw*ee*complex(0,1)*NN1x1*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN1x2*complexconjugate(Rn3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_552 = Coupling(name = 'GC_552',
                  value = '-((cw*ee*complex(0,1)*NN2x1*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN2x2*complexconjugate(Rn3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_553 = Coupling(name = 'GC_553',
                  value = '-((cw*ee*complex(0,1)*NN3x1*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN3x2*complexconjugate(Rn3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_554 = Coupling(name = 'GC_554',
                  value = '-((cw*ee*complex(0,1)*NN4x1*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))) + (ee*complex(0,1)*NN4x2*complexconjugate(Rn3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Rn3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_555 = Coupling(name = 'GC_555',
                  value = '(ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(2.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*complexconjugate(Rl3x3))/(4.*sw**2*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*Rn3x3*complexconjugate(Rl3x3)*complexconjugate(Rn3x3))/(2.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rl3x3*Rn3x3*complexconjugate(Rl3x3)*complexconjugate(Rn3x3))/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_556 = Coupling(name = 'GC_556',
                  value = '-(complex(0,1)*G*complexconjugate(Ru1x1)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_557 = Coupling(name = 'GC_557',
                  value = '(-2*ee*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/3.',
                  order = {'QED':1})

GC_558 = Coupling(name = 'GC_558',
                  value = '(8*ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/9.',
                  order = {'QED':2})

GC_559 = Coupling(name = 'GC_559',
                  value = '-(complex(0,1)*G*Ru1x1*complexconjugate(Ru1x1))',
                  order = {'QCD':1})

GC_56 = Coupling(name = 'GC_56',
                 value = '(cw*ee*complex(0,1)*NN3x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_560 = Coupling(name = 'GC_560',
                  value = 'complex(0,1)*G*Ru1x1*complexconjugate(Ru1x1)',
                  order = {'QCD':1})

GC_561 = Coupling(name = 'GC_561',
                  value = '(4*ee*complex(0,1)*G*Ru1x1*complexconjugate(Ru1x1))/3.',
                  order = {'QCD':1,'QED':1})

GC_562 = Coupling(name = 'GC_562',
                  value = 'complex(0,1)*G**2*Ru1x1*complexconjugate(Ru1x1)',
                  order = {'QCD':2})

GC_563 = Coupling(name = 'GC_563',
                  value = '(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_564 = Coupling(name = 'GC_564',
                  value = '-((ee*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_565 = Coupling(name = 'GC_565',
                  value = '(ee*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_566 = Coupling(name = 'GC_566',
                  value = '(ee**2*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1))/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_567 = Coupling(name = 'GC_567',
                  value = '(ee*complex(0,1)*G*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_568 = Coupling(name = 'GC_568',
                  value = '(cw*ee**2*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_569 = Coupling(name = 'GC_569',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1))/sw)',
                  order = {'QED':1})

GC_57 = Coupling(name = 'GC_57',
                 value = '(cw*ee*complex(0,1)*NN4x1*Rd5x5*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_570 = Coupling(name = 'GC_570',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1))/sw)',
                  order = {'QED':1})

GC_571 = Coupling(name = 'GC_571',
                  value = '-(CKM2x2*ee**2*complex(0,1)*Rd1x1*Ru2x2*complexconjugate(CKM1x1)*complexconjugate(Rd2x2)*complexconjugate(Ru1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_572 = Coupling(name = 'GC_572',
                  value = '-(CKM3x3*ee**2*complex(0,1)*Rd1x1*Ru3x3*complexconjugate(CKM1x1)*complexconjugate(Rd3x3)*complexconjugate(Ru1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_573 = Coupling(name = 'GC_573',
                  value = '(ee**2*complex(0,1)*Rd4x4*Ru1x1*complexconjugate(Rd4x4)*complexconjugate(Ru1x1))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_574 = Coupling(name = 'GC_574',
                  value = '(ee**2*complex(0,1)*Rd5x5*Ru1x1*complexconjugate(Rd5x5)*complexconjugate(Ru1x1))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_575 = Coupling(name = 'GC_575',
                  value = '(ee**2*complex(0,1)*Rd6x6*Ru1x1*complexconjugate(Rd6x6)*complexconjugate(Ru1x1))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_576 = Coupling(name = 'GC_576',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Rn1x1*complexconjugate(CKM1x1)*complexconjugate(Rl1x1)*complexconjugate(Ru1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_577 = Coupling(name = 'GC_577',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Rn2x2*complexconjugate(CKM1x1)*complexconjugate(Rl2x2)*complexconjugate(Ru1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_578 = Coupling(name = 'GC_578',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Rn3x3*complexconjugate(CKM1x1)*complexconjugate(Rl3x3)*complexconjugate(Ru1x1))/(2.*sw**2)',
                  order = {'QED':2})

GC_579 = Coupling(name = 'GC_579',
                  value = '(ee**2*complex(0,1)*Rl4x4*Ru1x1*complexconjugate(Rl4x4)*complexconjugate(Ru1x1))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_58 = Coupling(name = 'GC_58',
                 value = '(cw*ee*complex(0,1)*NN1x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_580 = Coupling(name = 'GC_580',
                  value = '(ee**2*complex(0,1)*Rl5x5*Ru1x1*complexconjugate(Rl5x5)*complexconjugate(Ru1x1))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_581 = Coupling(name = 'GC_581',
                  value = '(ee**2*complex(0,1)*Rl6x6*Ru1x1*complexconjugate(Rl6x6)*complexconjugate(Ru1x1))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_582 = Coupling(name = 'GC_582',
                  value = '-(ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_583 = Coupling(name = 'GC_583',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Ru1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*complexconjugate(Ru1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Ru1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_584 = Coupling(name = 'GC_584',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Ru1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*complexconjugate(Ru1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Ru1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_585 = Coupling(name = 'GC_585',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Ru1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*complexconjugate(Ru1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Ru1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_586 = Coupling(name = 'GC_586',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Ru1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*complexconjugate(Ru1x1))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Ru1x1))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_587 = Coupling(name = 'GC_587',
                  value = '(cw*ee*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(2.*sw*(-1 + sw**2)) - (2*cw*ee*complex(0,1)*Ru1x1*sw*complexconjugate(Ru1x1))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_588 = Coupling(name = 'GC_588',
                  value = '-(cw*ee*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(2.*sw*(-1 + sw**2)) + (2*cw*ee*complex(0,1)*Ru1x1*sw*complexconjugate(Ru1x1))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_589 = Coupling(name = 'GC_589',
                  value = '(-2*cw*ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(3.*sw*(-1 + sw**2)) + (8*cw*ee**2*complex(0,1)*Ru1x1*sw*complexconjugate(Ru1x1))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_59 = Coupling(name = 'GC_59',
                 value = '(cw*ee*complex(0,1)*NN2x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_590 = Coupling(name = 'GC_590',
                  value = '-((cw*ee*complex(0,1)*G*Ru1x1*complexconjugate(Ru1x1))/(sw*(-1 + sw**2))) + (4*cw*ee*complex(0,1)*G*Ru1x1*sw*complexconjugate(Ru1x1))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_591 = Coupling(name = 'GC_591',
                  value = '(4*ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru1x1*complexconjugate(Ru1x1))/(2.*sw**2*(-1 + sw**2)) - (8*ee**2*complex(0,1)*Ru1x1*sw**2*complexconjugate(Ru1x1))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_592 = Coupling(name = 'GC_592',
                  value = '(5*ee**2*complex(0,1)*Rd1x1*Ru1x1*complexconjugate(Rd1x1)*complexconjugate(Ru1x1))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*Ru1x1*complexconjugate(Rd1x1)*complexconjugate(Ru1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_593 = Coupling(name = 'GC_593',
                  value = '(complex(0,1)*G**2*Rd1x1*Ru1x1*complexconjugate(Rd1x1)*complexconjugate(Ru1x1))/(-1 + sw**2) - (complex(0,1)*G**2*Rd1x1*Ru1x1*sw**2*complexconjugate(Rd1x1)*complexconjugate(Ru1x1))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_594 = Coupling(name = 'GC_594',
                  value = '-(CKM1x1*ee**2*complex(0,1)*Rd1x1*Ru1x1*complexconjugate(CKM1x1)*complexconjugate(Rd1x1)*complexconjugate(Ru1x1))/(2.*(-1 + sw**2)) + (CKM1x1*ee**2*complex(0,1)*Rd1x1*Ru1x1*complexconjugate(CKM1x1)*complexconjugate(Rd1x1)*complexconjugate(Ru1x1))/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_595 = Coupling(name = 'GC_595',
                  value = '(5*ee**2*complex(0,1)*Rd2x2*Ru1x1*complexconjugate(Rd2x2)*complexconjugate(Ru1x1))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*Ru1x1*complexconjugate(Rd2x2)*complexconjugate(Ru1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_596 = Coupling(name = 'GC_596',
                  value = '(complex(0,1)*G**2*Rd2x2*Ru1x1*complexconjugate(Rd2x2)*complexconjugate(Ru1x1))/(-1 + sw**2) - (complex(0,1)*G**2*Rd2x2*Ru1x1*sw**2*complexconjugate(Rd2x2)*complexconjugate(Ru1x1))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_597 = Coupling(name = 'GC_597',
                  value = '(5*ee**2*complex(0,1)*Rd3x3*Ru1x1*complexconjugate(Rd3x3)*complexconjugate(Ru1x1))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*Ru1x1*complexconjugate(Rd3x3)*complexconjugate(Ru1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_598 = Coupling(name = 'GC_598',
                  value = '(complex(0,1)*G**2*Rd3x3*Ru1x1*complexconjugate(Rd3x3)*complexconjugate(Ru1x1))/(-1 + sw**2) - (complex(0,1)*G**2*Rd3x3*Ru1x1*sw**2*complexconjugate(Rd3x3)*complexconjugate(Ru1x1))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_599 = Coupling(name = 'GC_599',
                  value = '-((complex(0,1)*G**2*Rd4x4*Ru1x1*complexconjugate(Rd4x4)*complexconjugate(Ru1x1))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd4x4*Ru1x1*sw**2*complexconjugate(Rd4x4)*complexconjugate(Ru1x1))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_6 = Coupling(name = 'GC_6',
                value = '-2*ee**2*complex(0,1)',
                order = {'QED':2})

GC_60 = Coupling(name = 'GC_60',
                 value = '(cw*ee*complex(0,1)*NN3x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_600 = Coupling(name = 'GC_600',
                  value = '-((complex(0,1)*G**2*Rd5x5*Ru1x1*complexconjugate(Rd5x5)*complexconjugate(Ru1x1))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd5x5*Ru1x1*sw**2*complexconjugate(Rd5x5)*complexconjugate(Ru1x1))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_601 = Coupling(name = 'GC_601',
                  value = '-((complex(0,1)*G**2*Rd6x6*Ru1x1*complexconjugate(Rd6x6)*complexconjugate(Ru1x1))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd6x6*Ru1x1*sw**2*complexconjugate(Rd6x6)*complexconjugate(Ru1x1))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_602 = Coupling(name = 'GC_602',
                  value = '(ee**2*complex(0,1)*Rl1x1*Ru1x1*complexconjugate(Rl1x1)*complexconjugate(Ru1x1))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*Ru1x1*complexconjugate(Rl1x1)*complexconjugate(Ru1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_603 = Coupling(name = 'GC_603',
                  value = '(ee**2*complex(0,1)*Rl2x2*Ru1x1*complexconjugate(Rl2x2)*complexconjugate(Ru1x1))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*Ru1x1*complexconjugate(Rl2x2)*complexconjugate(Ru1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_604 = Coupling(name = 'GC_604',
                  value = '(ee**2*complex(0,1)*Rl3x3*Ru1x1*complexconjugate(Rl3x3)*complexconjugate(Ru1x1))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*Ru1x1*complexconjugate(Rl3x3)*complexconjugate(Ru1x1))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_605 = Coupling(name = 'GC_605',
                  value = '(-2*ee**2*complex(0,1)*Ru1x1**2*complexconjugate(Ru1x1)**2)/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1**2*complexconjugate(Ru1x1)**2)/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_606 = Coupling(name = 'GC_606',
                  value = '(complex(0,1)*G**2*Ru1x1**2*complexconjugate(Ru1x1)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Ru1x1**2*sw**2*complexconjugate(Ru1x1)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_607 = Coupling(name = 'GC_607',
                  value = '-(complex(0,1)*G*complexconjugate(Ru2x2)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_608 = Coupling(name = 'GC_608',
                  value = '(-2*ee*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/3.',
                  order = {'QED':1})

GC_609 = Coupling(name = 'GC_609',
                  value = '(2*ee*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/3.',
                  order = {'QED':1})

GC_61 = Coupling(name = 'GC_61',
                 value = '(cw*ee*complex(0,1)*NN4x1*Rd6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_610 = Coupling(name = 'GC_610',
                  value = '(8*ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/9.',
                  order = {'QED':2})

GC_611 = Coupling(name = 'GC_611',
                  value = '-(complex(0,1)*G*Ru2x2*complexconjugate(Ru2x2))',
                  order = {'QCD':1})

GC_612 = Coupling(name = 'GC_612',
                  value = 'complex(0,1)*G*Ru2x2*complexconjugate(Ru2x2)',
                  order = {'QCD':1})

GC_613 = Coupling(name = 'GC_613',
                  value = '(4*ee*complex(0,1)*G*Ru2x2*complexconjugate(Ru2x2))/3.',
                  order = {'QCD':1,'QED':1})

GC_614 = Coupling(name = 'GC_614',
                  value = 'complex(0,1)*G**2*Ru2x2*complexconjugate(Ru2x2)',
                  order = {'QCD':2})

GC_615 = Coupling(name = 'GC_615',
                  value = '(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_616 = Coupling(name = 'GC_616',
                  value = '-((ee*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_617 = Coupling(name = 'GC_617',
                  value = '(ee*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_618 = Coupling(name = 'GC_618',
                  value = '(ee**2*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2))/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_619 = Coupling(name = 'GC_619',
                  value = '(ee*complex(0,1)*G*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_62 = Coupling(name = 'GC_62',
                 value = '(cw*ee*complex(0,1)*NN1x1*Rl4x4*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_620 = Coupling(name = 'GC_620',
                  value = '(cw*ee**2*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_621 = Coupling(name = 'GC_621',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(CKM2x2)*complexconjugate(Ru2x2))/sw)',
                  order = {'QED':1})

GC_622 = Coupling(name = 'GC_622',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(CKM2x2)*complexconjugate(Ru2x2))/sw)',
                  order = {'QED':1})

GC_623 = Coupling(name = 'GC_623',
                  value = '-(CKM1x1*ee**2*complex(0,1)*Rd2x2*Ru1x1*complexconjugate(CKM2x2)*complexconjugate(Rd1x1)*complexconjugate(Ru2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_624 = Coupling(name = 'GC_624',
                  value = '-(CKM3x3*ee**2*complex(0,1)*Rd2x2*Ru3x3*complexconjugate(CKM2x2)*complexconjugate(Rd3x3)*complexconjugate(Ru2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_625 = Coupling(name = 'GC_625',
                  value = '(ee**2*complex(0,1)*Rd4x4*Ru2x2*complexconjugate(Rd4x4)*complexconjugate(Ru2x2))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_626 = Coupling(name = 'GC_626',
                  value = '(ee**2*complex(0,1)*Rd5x5*Ru2x2*complexconjugate(Rd5x5)*complexconjugate(Ru2x2))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_627 = Coupling(name = 'GC_627',
                  value = '(ee**2*complex(0,1)*Rd6x6*Ru2x2*complexconjugate(Rd6x6)*complexconjugate(Ru2x2))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_628 = Coupling(name = 'GC_628',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Rn1x1*complexconjugate(CKM2x2)*complexconjugate(Rl1x1)*complexconjugate(Ru2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_629 = Coupling(name = 'GC_629',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Rn2x2*complexconjugate(CKM2x2)*complexconjugate(Rl2x2)*complexconjugate(Ru2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_63 = Coupling(name = 'GC_63',
                 value = '(cw*ee*complex(0,1)*NN2x1*Rl4x4*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_630 = Coupling(name = 'GC_630',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Rn3x3*complexconjugate(CKM2x2)*complexconjugate(Rl3x3)*complexconjugate(Ru2x2))/(2.*sw**2)',
                  order = {'QED':2})

GC_631 = Coupling(name = 'GC_631',
                  value = '(ee**2*complex(0,1)*Rl4x4*Ru2x2*complexconjugate(Rl4x4)*complexconjugate(Ru2x2))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_632 = Coupling(name = 'GC_632',
                  value = '(ee**2*complex(0,1)*Rl5x5*Ru2x2*complexconjugate(Rl5x5)*complexconjugate(Ru2x2))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_633 = Coupling(name = 'GC_633',
                  value = '(ee**2*complex(0,1)*Rl6x6*Ru2x2*complexconjugate(Rl6x6)*complexconjugate(Ru2x2))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_634 = Coupling(name = 'GC_634',
                  value = '-(ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_635 = Coupling(name = 'GC_635',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Ru2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*complexconjugate(Ru2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Ru2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_636 = Coupling(name = 'GC_636',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Ru2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*complexconjugate(Ru2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Ru2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_637 = Coupling(name = 'GC_637',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Ru2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*complexconjugate(Ru2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Ru2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_638 = Coupling(name = 'GC_638',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Ru2x2))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*complexconjugate(Ru2x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Ru2x2))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_639 = Coupling(name = 'GC_639',
                  value = '(cw*ee*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(2.*sw*(-1 + sw**2)) - (2*cw*ee*complex(0,1)*Ru2x2*sw*complexconjugate(Ru2x2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_64 = Coupling(name = 'GC_64',
                 value = '(cw*ee*complex(0,1)*NN3x1*Rl4x4*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_640 = Coupling(name = 'GC_640',
                  value = '-(cw*ee*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(2.*sw*(-1 + sw**2)) + (2*cw*ee*complex(0,1)*Ru2x2*sw*complexconjugate(Ru2x2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_641 = Coupling(name = 'GC_641',
                  value = '(-2*cw*ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(3.*sw*(-1 + sw**2)) + (8*cw*ee**2*complex(0,1)*Ru2x2*sw*complexconjugate(Ru2x2))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_642 = Coupling(name = 'GC_642',
                  value = '-((cw*ee*complex(0,1)*G*Ru2x2*complexconjugate(Ru2x2))/(sw*(-1 + sw**2))) + (4*cw*ee*complex(0,1)*G*Ru2x2*sw*complexconjugate(Ru2x2))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_643 = Coupling(name = 'GC_643',
                  value = '(4*ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru2x2*complexconjugate(Ru2x2))/(2.*sw**2*(-1 + sw**2)) - (8*ee**2*complex(0,1)*Ru2x2*sw**2*complexconjugate(Ru2x2))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_644 = Coupling(name = 'GC_644',
                  value = '(5*ee**2*complex(0,1)*Rd1x1*Ru2x2*complexconjugate(Rd1x1)*complexconjugate(Ru2x2))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*Ru2x2*complexconjugate(Rd1x1)*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_645 = Coupling(name = 'GC_645',
                  value = '(complex(0,1)*G**2*Rd1x1*Ru2x2*complexconjugate(Rd1x1)*complexconjugate(Ru2x2))/(-1 + sw**2) - (complex(0,1)*G**2*Rd1x1*Ru2x2*sw**2*complexconjugate(Rd1x1)*complexconjugate(Ru2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_646 = Coupling(name = 'GC_646',
                  value = '(5*ee**2*complex(0,1)*Rd2x2*Ru2x2*complexconjugate(Rd2x2)*complexconjugate(Ru2x2))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*Ru2x2*complexconjugate(Rd2x2)*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_647 = Coupling(name = 'GC_647',
                  value = '(complex(0,1)*G**2*Rd2x2*Ru2x2*complexconjugate(Rd2x2)*complexconjugate(Ru2x2))/(-1 + sw**2) - (complex(0,1)*G**2*Rd2x2*Ru2x2*sw**2*complexconjugate(Rd2x2)*complexconjugate(Ru2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_648 = Coupling(name = 'GC_648',
                  value = '-(CKM2x2*ee**2*complex(0,1)*Rd2x2*Ru2x2*complexconjugate(CKM2x2)*complexconjugate(Rd2x2)*complexconjugate(Ru2x2))/(2.*(-1 + sw**2)) + (CKM2x2*ee**2*complex(0,1)*Rd2x2*Ru2x2*complexconjugate(CKM2x2)*complexconjugate(Rd2x2)*complexconjugate(Ru2x2))/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_649 = Coupling(name = 'GC_649',
                  value = '(5*ee**2*complex(0,1)*Rd3x3*Ru2x2*complexconjugate(Rd3x3)*complexconjugate(Ru2x2))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*Ru2x2*complexconjugate(Rd3x3)*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_65 = Coupling(name = 'GC_65',
                 value = '(cw*ee*complex(0,1)*NN4x1*Rl4x4*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_650 = Coupling(name = 'GC_650',
                  value = '(complex(0,1)*G**2*Rd3x3*Ru2x2*complexconjugate(Rd3x3)*complexconjugate(Ru2x2))/(-1 + sw**2) - (complex(0,1)*G**2*Rd3x3*Ru2x2*sw**2*complexconjugate(Rd3x3)*complexconjugate(Ru2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_651 = Coupling(name = 'GC_651',
                  value = '-((complex(0,1)*G**2*Rd4x4*Ru2x2*complexconjugate(Rd4x4)*complexconjugate(Ru2x2))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd4x4*Ru2x2*sw**2*complexconjugate(Rd4x4)*complexconjugate(Ru2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_652 = Coupling(name = 'GC_652',
                  value = '-((complex(0,1)*G**2*Rd5x5*Ru2x2*complexconjugate(Rd5x5)*complexconjugate(Ru2x2))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd5x5*Ru2x2*sw**2*complexconjugate(Rd5x5)*complexconjugate(Ru2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_653 = Coupling(name = 'GC_653',
                  value = '-((complex(0,1)*G**2*Rd6x6*Ru2x2*complexconjugate(Rd6x6)*complexconjugate(Ru2x2))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd6x6*Ru2x2*sw**2*complexconjugate(Rd6x6)*complexconjugate(Ru2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_654 = Coupling(name = 'GC_654',
                  value = '(ee**2*complex(0,1)*Rl1x1*Ru2x2*complexconjugate(Rl1x1)*complexconjugate(Ru2x2))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*Ru2x2*complexconjugate(Rl1x1)*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_655 = Coupling(name = 'GC_655',
                  value = '(ee**2*complex(0,1)*Rl2x2*Ru2x2*complexconjugate(Rl2x2)*complexconjugate(Ru2x2))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*Ru2x2*complexconjugate(Rl2x2)*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_656 = Coupling(name = 'GC_656',
                  value = '(ee**2*complex(0,1)*Rl3x3*Ru2x2*complexconjugate(Rl3x3)*complexconjugate(Ru2x2))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*Ru2x2*complexconjugate(Rl3x3)*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_657 = Coupling(name = 'GC_657',
                  value = '(-2*ee**2*complex(0,1)*Ru1x1*Ru2x2*complexconjugate(Ru1x1)*complexconjugate(Ru2x2))/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*Ru2x2*complexconjugate(Ru1x1)*complexconjugate(Ru2x2))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_658 = Coupling(name = 'GC_658',
                  value = '(complex(0,1)*G**2*Ru1x1*Ru2x2*complexconjugate(Ru1x1)*complexconjugate(Ru2x2))/(-1 + sw**2) - (complex(0,1)*G**2*Ru1x1*Ru2x2*sw**2*complexconjugate(Ru1x1)*complexconjugate(Ru2x2))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_659 = Coupling(name = 'GC_659',
                  value = '(-2*ee**2*complex(0,1)*Ru2x2**2*complexconjugate(Ru2x2)**2)/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2**2*complexconjugate(Ru2x2)**2)/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_66 = Coupling(name = 'GC_66',
                 value = '(cw*ee*complex(0,1)*NN1x1*Rl5x5*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_660 = Coupling(name = 'GC_660',
                  value = '(complex(0,1)*G**2*Ru2x2**2*complexconjugate(Ru2x2)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Ru2x2**2*sw**2*complexconjugate(Ru2x2)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_661 = Coupling(name = 'GC_661',
                  value = '-(complex(0,1)*G*complexconjugate(Ru3x3)*cmath.sqrt(2))',
                  order = {'QCD':1})

GC_662 = Coupling(name = 'GC_662',
                  value = '(-2*ee*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/3.',
                  order = {'QED':1})

GC_663 = Coupling(name = 'GC_663',
                  value = '(2*ee*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/3.',
                  order = {'QED':1})

GC_664 = Coupling(name = 'GC_664',
                  value = '(8*ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/9.',
                  order = {'QED':2})

GC_665 = Coupling(name = 'GC_665',
                  value = '-(complex(0,1)*G*Ru3x3*complexconjugate(Ru3x3))',
                  order = {'QCD':1})

GC_666 = Coupling(name = 'GC_666',
                  value = 'complex(0,1)*G*Ru3x3*complexconjugate(Ru3x3)',
                  order = {'QCD':1})

GC_667 = Coupling(name = 'GC_667',
                  value = '(4*ee*complex(0,1)*G*Ru3x3*complexconjugate(Ru3x3))/3.',
                  order = {'QCD':1,'QED':1})

GC_668 = Coupling(name = 'GC_668',
                  value = 'complex(0,1)*G**2*Ru3x3*complexconjugate(Ru3x3)',
                  order = {'QCD':2})

GC_669 = Coupling(name = 'GC_669',
                  value = '(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_67 = Coupling(name = 'GC_67',
                 value = '(cw*ee*complex(0,1)*NN2x1*Rl5x5*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_670 = Coupling(name = 'GC_670',
                  value = '-((ee*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3))/(sw*cmath.sqrt(2)))',
                  order = {'QED':1})

GC_671 = Coupling(name = 'GC_671',
                  value = '(ee*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_672 = Coupling(name = 'GC_672',
                  value = '(ee**2*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3))/(3.*sw*cmath.sqrt(2))',
                  order = {'QED':2})

GC_673 = Coupling(name = 'GC_673',
                  value = '(ee*complex(0,1)*G*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*cmath.sqrt(2))/sw',
                  order = {'QCD':1,'QED':1})

GC_674 = Coupling(name = 'GC_674',
                  value = '(cw*ee**2*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':2})

GC_675 = Coupling(name = 'GC_675',
                  value = '-((ee*complex(0,1)*VV1x1*complexconjugate(CKM3x3)*complexconjugate(Ru3x3))/sw)',
                  order = {'QED':1})

GC_676 = Coupling(name = 'GC_676',
                  value = '-((ee*complex(0,1)*VV2x1*complexconjugate(CKM3x3)*complexconjugate(Ru3x3))/sw)',
                  order = {'QED':1})

GC_677 = Coupling(name = 'GC_677',
                  value = '-(CKM1x1*ee**2*complex(0,1)*Rd3x3*Ru1x1*complexconjugate(CKM3x3)*complexconjugate(Rd1x1)*complexconjugate(Ru3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_678 = Coupling(name = 'GC_678',
                  value = '-(CKM2x2*ee**2*complex(0,1)*Rd3x3*Ru2x2*complexconjugate(CKM3x3)*complexconjugate(Rd2x2)*complexconjugate(Ru3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_679 = Coupling(name = 'GC_679',
                  value = '(ee**2*complex(0,1)*Rd4x4*Ru3x3*complexconjugate(Rd4x4)*complexconjugate(Ru3x3))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_68 = Coupling(name = 'GC_68',
                 value = '(cw*ee*complex(0,1)*NN3x1*Rl5x5*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_680 = Coupling(name = 'GC_680',
                  value = '(ee**2*complex(0,1)*Rd5x5*Ru3x3*complexconjugate(Rd5x5)*complexconjugate(Ru3x3))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_681 = Coupling(name = 'GC_681',
                  value = '(ee**2*complex(0,1)*Rd6x6*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(Ru3x3))/(18.*(-1 + sw**2))',
                  order = {'QED':2})

GC_682 = Coupling(name = 'GC_682',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Rn1x1*complexconjugate(CKM3x3)*complexconjugate(Rl1x1)*complexconjugate(Ru3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_683 = Coupling(name = 'GC_683',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Rn2x2*complexconjugate(CKM3x3)*complexconjugate(Rl2x2)*complexconjugate(Ru3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_684 = Coupling(name = 'GC_684',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Rn3x3*complexconjugate(CKM3x3)*complexconjugate(Rl3x3)*complexconjugate(Ru3x3))/(2.*sw**2)',
                  order = {'QED':2})

GC_685 = Coupling(name = 'GC_685',
                  value = '(ee**2*complex(0,1)*Rl4x4*Ru3x3*complexconjugate(Rl4x4)*complexconjugate(Ru3x3))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_686 = Coupling(name = 'GC_686',
                  value = '(ee**2*complex(0,1)*Rl5x5*Ru3x3*complexconjugate(Rl5x5)*complexconjugate(Ru3x3))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_687 = Coupling(name = 'GC_687',
                  value = '(ee**2*complex(0,1)*Rl6x6*Ru3x3*complexconjugate(Rl6x6)*complexconjugate(Ru3x3))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_688 = Coupling(name = 'GC_688',
                  value = '-(ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(3.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_689 = Coupling(name = 'GC_689',
                  value = '(cw*ee*complex(0,1)*NN1x1*complexconjugate(Ru3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN1x2*complexconjugate(Ru3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN1x2*sw*complexconjugate(Ru3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_69 = Coupling(name = 'GC_69',
                 value = '(cw*ee*complex(0,1)*NN4x1*Rl5x5*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_690 = Coupling(name = 'GC_690',
                  value = '(cw*ee*complex(0,1)*NN2x1*complexconjugate(Ru3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN2x2*complexconjugate(Ru3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN2x2*sw*complexconjugate(Ru3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_691 = Coupling(name = 'GC_691',
                  value = '(cw*ee*complex(0,1)*NN3x1*complexconjugate(Ru3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN3x2*complexconjugate(Ru3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN3x2*sw*complexconjugate(Ru3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_692 = Coupling(name = 'GC_692',
                  value = '(cw*ee*complex(0,1)*NN4x1*complexconjugate(Ru3x3))/(3.*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*NN4x2*complexconjugate(Ru3x3))/(sw*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*NN4x2*sw*complexconjugate(Ru3x3))/((-1 + sw**2)*cmath.sqrt(2))',
                  order = {'QED':1})

GC_693 = Coupling(name = 'GC_693',
                  value = '(cw*ee*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(2.*sw*(-1 + sw**2)) - (2*cw*ee*complex(0,1)*Ru3x3*sw*complexconjugate(Ru3x3))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_694 = Coupling(name = 'GC_694',
                  value = '-(cw*ee*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(2.*sw*(-1 + sw**2)) + (2*cw*ee*complex(0,1)*Ru3x3*sw*complexconjugate(Ru3x3))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_695 = Coupling(name = 'GC_695',
                  value = '(-2*cw*ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(3.*sw*(-1 + sw**2)) + (8*cw*ee**2*complex(0,1)*Ru3x3*sw*complexconjugate(Ru3x3))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_696 = Coupling(name = 'GC_696',
                  value = '-((cw*ee*complex(0,1)*G*Ru3x3*complexconjugate(Ru3x3))/(sw*(-1 + sw**2))) + (4*cw*ee*complex(0,1)*G*Ru3x3*sw*complexconjugate(Ru3x3))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_697 = Coupling(name = 'GC_697',
                  value = '(4*ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(3.*(-1 + sw**2)) - (ee**2*complex(0,1)*Ru3x3*complexconjugate(Ru3x3))/(2.*sw**2*(-1 + sw**2)) - (8*ee**2*complex(0,1)*Ru3x3*sw**2*complexconjugate(Ru3x3))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_698 = Coupling(name = 'GC_698',
                  value = '(complex(0,1)*yu3x3*complexconjugate(NN1x4)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yu3x3*complexconjugate(NN1x4)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_699 = Coupling(name = 'GC_699',
                  value = '(complex(0,1)*yu3x3*complexconjugate(NN2x4)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yu3x3*complexconjugate(NN2x4)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_7 = Coupling(name = 'GC_7',
                value = '2*ee**2*complex(0,1)',
                order = {'QED':2})

GC_70 = Coupling(name = 'GC_70',
                 value = '(cw*ee*complex(0,1)*NN1x1*Rl6x6*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_700 = Coupling(name = 'GC_700',
                  value = '(complex(0,1)*yu3x3*complexconjugate(NN3x4)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yu3x3*complexconjugate(NN3x4)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_701 = Coupling(name = 'GC_701',
                  value = '(complex(0,1)*yu3x3*complexconjugate(NN4x4)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*sw**2*yu3x3*complexconjugate(NN4x4)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_702 = Coupling(name = 'GC_702',
                  value = '(5*ee**2*complex(0,1)*Rd1x1*Ru3x3*complexconjugate(Rd1x1)*complexconjugate(Ru3x3))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*Ru3x3*complexconjugate(Rd1x1)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_703 = Coupling(name = 'GC_703',
                  value = '(complex(0,1)*G**2*Rd1x1*Ru3x3*complexconjugate(Rd1x1)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*G**2*Rd1x1*Ru3x3*sw**2*complexconjugate(Rd1x1)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_704 = Coupling(name = 'GC_704',
                  value = '(5*ee**2*complex(0,1)*Rd2x2*Ru3x3*complexconjugate(Rd2x2)*complexconjugate(Ru3x3))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd2x2*Ru3x3*complexconjugate(Rd2x2)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_705 = Coupling(name = 'GC_705',
                  value = '(complex(0,1)*G**2*Rd2x2*Ru3x3*complexconjugate(Rd2x2)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*G**2*Rd2x2*Ru3x3*sw**2*complexconjugate(Rd2x2)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_706 = Coupling(name = 'GC_706',
                  value = '(5*ee**2*complex(0,1)*Rd3x3*Ru3x3*complexconjugate(Rd3x3)*complexconjugate(Ru3x3))/(18.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd3x3*Ru3x3*complexconjugate(Rd3x3)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_707 = Coupling(name = 'GC_707',
                  value = '(complex(0,1)*G**2*Rd3x3*Ru3x3*complexconjugate(Rd3x3)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*G**2*Rd3x3*Ru3x3*sw**2*complexconjugate(Rd3x3)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_708 = Coupling(name = 'GC_708',
                  value = '-(CKM3x3*ee**2*complex(0,1)*Rd3x3*Ru3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(Ru3x3))/(2.*(-1 + sw**2)) + (CKM3x3*ee**2*complex(0,1)*Rd3x3*Ru3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(Ru3x3))/(2.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_709 = Coupling(name = 'GC_709',
                  value = '-((complex(0,1)*G**2*Rd4x4*Ru3x3*complexconjugate(Rd4x4)*complexconjugate(Ru3x3))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd4x4*Ru3x3*sw**2*complexconjugate(Rd4x4)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_71 = Coupling(name = 'GC_71',
                 value = '(cw*ee*complex(0,1)*NN2x1*Rl6x6*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_710 = Coupling(name = 'GC_710',
                  value = '-((complex(0,1)*G**2*Rd5x5*Ru3x3*complexconjugate(Rd5x5)*complexconjugate(Ru3x3))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd5x5*Ru3x3*sw**2*complexconjugate(Rd5x5)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_711 = Coupling(name = 'GC_711',
                  value = '-((complex(0,1)*G**2*Rd6x6*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(Ru3x3))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd6x6*Ru3x3*sw**2*complexconjugate(Rd6x6)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_712 = Coupling(name = 'GC_712',
                  value = '(ee**2*complex(0,1)*Rl1x1*Ru3x3*complexconjugate(Rl1x1)*complexconjugate(Ru3x3))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl1x1*Ru3x3*complexconjugate(Rl1x1)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_713 = Coupling(name = 'GC_713',
                  value = '(ee**2*complex(0,1)*Rl2x2*Ru3x3*complexconjugate(Rl2x2)*complexconjugate(Ru3x3))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl2x2*Ru3x3*complexconjugate(Rl2x2)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_714 = Coupling(name = 'GC_714',
                  value = '(ee**2*complex(0,1)*Rl3x3*Ru3x3*complexconjugate(Rl3x3)*complexconjugate(Ru3x3))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rl3x3*Ru3x3*complexconjugate(Rl3x3)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_715 = Coupling(name = 'GC_715',
                  value = '(-2*ee**2*complex(0,1)*Ru1x1*Ru3x3*complexconjugate(Ru1x1)*complexconjugate(Ru3x3))/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru1x1*Ru3x3*complexconjugate(Ru1x1)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_716 = Coupling(name = 'GC_716',
                  value = '(complex(0,1)*G**2*Ru1x1*Ru3x3*complexconjugate(Ru1x1)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*G**2*Ru1x1*Ru3x3*sw**2*complexconjugate(Ru1x1)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_717 = Coupling(name = 'GC_717',
                  value = '(-2*ee**2*complex(0,1)*Ru2x2*Ru3x3*complexconjugate(Ru2x2)*complexconjugate(Ru3x3))/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru2x2*Ru3x3*complexconjugate(Ru2x2)*complexconjugate(Ru3x3))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_718 = Coupling(name = 'GC_718',
                  value = '(complex(0,1)*G**2*Ru2x2*Ru3x3*complexconjugate(Ru2x2)*complexconjugate(Ru3x3))/(-1 + sw**2) - (complex(0,1)*G**2*Ru2x2*Ru3x3*sw**2*complexconjugate(Ru2x2)*complexconjugate(Ru3x3))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_719 = Coupling(name = 'GC_719',
                  value = '(-2*ee**2*complex(0,1)*Ru3x3**2*complexconjugate(Ru3x3)**2)/(9.*(-1 + sw**2)) + (ee**2*complex(0,1)*Ru3x3**2*complexconjugate(Ru3x3)**2)/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_72 = Coupling(name = 'GC_72',
                 value = '(cw*ee*complex(0,1)*NN3x1*Rl6x6*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_720 = Coupling(name = 'GC_720',
                  value = '(complex(0,1)*G**2*Ru3x3**2*complexconjugate(Ru3x3)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Ru3x3**2*sw**2*complexconjugate(Ru3x3)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_721 = Coupling(name = 'GC_721',
                  value = 'complex(0,1)*G*complexconjugate(Ru4x4)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_722 = Coupling(name = 'GC_722',
                  value = '(-2*ee*complex(0,1)*Ru4x4*complexconjugate(Ru4x4))/3.',
                  order = {'QED':1})

GC_723 = Coupling(name = 'GC_723',
                  value = '(2*ee*complex(0,1)*Ru4x4*complexconjugate(Ru4x4))/3.',
                  order = {'QED':1})

GC_724 = Coupling(name = 'GC_724',
                  value = '(8*ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4))/9.',
                  order = {'QED':2})

GC_725 = Coupling(name = 'GC_725',
                  value = '-(complex(0,1)*G*Ru4x4*complexconjugate(Ru4x4))',
                  order = {'QCD':1})

GC_726 = Coupling(name = 'GC_726',
                  value = 'complex(0,1)*G*Ru4x4*complexconjugate(Ru4x4)',
                  order = {'QCD':1})

GC_727 = Coupling(name = 'GC_727',
                  value = '(4*ee*complex(0,1)*G*Ru4x4*complexconjugate(Ru4x4))/3.',
                  order = {'QCD':1,'QED':1})

GC_728 = Coupling(name = 'GC_728',
                  value = 'complex(0,1)*G**2*Ru4x4*complexconjugate(Ru4x4)',
                  order = {'QCD':2})

GC_729 = Coupling(name = 'GC_729',
                  value = '(8*ee**2*complex(0,1)*Ru4x4*sw**2*complexconjugate(Ru4x4))/(9 - 9*sw**2)',
                  order = {'QED':2})

GC_73 = Coupling(name = 'GC_73',
                 value = '(cw*ee*complex(0,1)*NN4x1*Rl6x6*cmath.sqrt(2))/(-1 + sw**2)',
                 order = {'QED':1})

GC_730 = Coupling(name = 'GC_730',
                  value = '(ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_731 = Coupling(name = 'GC_731',
                  value = '(-2*cw*ee*complex(0,1)*Ru4x4*sw*complexconjugate(Ru4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_732 = Coupling(name = 'GC_732',
                  value = '(2*cw*ee*complex(0,1)*Ru4x4*sw*complexconjugate(Ru4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_733 = Coupling(name = 'GC_733',
                  value = '(8*cw*ee**2*complex(0,1)*Ru4x4*sw*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_734 = Coupling(name = 'GC_734',
                  value = '(4*cw*ee*complex(0,1)*G*Ru4x4*sw*complexconjugate(Ru4x4))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_735 = Coupling(name = 'GC_735',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_736 = Coupling(name = 'GC_736',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_737 = Coupling(name = 'GC_737',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_738 = Coupling(name = 'GC_738',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Ru4x4)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_739 = Coupling(name = 'GC_739',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Ru4x4*complexconjugate(Rd1x1)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_74 = Coupling(name = 'GC_74',
                 value = '(-2*cw*ee*complex(0,1)*NN1x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_740 = Coupling(name = 'GC_740',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Ru4x4*complexconjugate(Rd2x2)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_741 = Coupling(name = 'GC_741',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Ru4x4*complexconjugate(Rd3x3)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_742 = Coupling(name = 'GC_742',
                  value = '(-2*ee**2*complex(0,1)*Rd4x4*Ru4x4*complexconjugate(Rd4x4)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_743 = Coupling(name = 'GC_743',
                  value = '(-2*ee**2*complex(0,1)*Rd5x5*Ru4x4*complexconjugate(Rd5x5)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_744 = Coupling(name = 'GC_744',
                  value = '(-2*ee**2*complex(0,1)*Rd6x6*Ru4x4*complexconjugate(Rd6x6)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_745 = Coupling(name = 'GC_745',
                  value = '(ee**2*complex(0,1)*Rl1x1*Ru4x4*complexconjugate(Rl1x1)*complexconjugate(Ru4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_746 = Coupling(name = 'GC_746',
                  value = '(ee**2*complex(0,1)*Rl2x2*Ru4x4*complexconjugate(Rl2x2)*complexconjugate(Ru4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_747 = Coupling(name = 'GC_747',
                  value = '(ee**2*complex(0,1)*Rl3x3*Ru4x4*complexconjugate(Rl3x3)*complexconjugate(Ru4x4))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_748 = Coupling(name = 'GC_748',
                  value = '(2*ee**2*complex(0,1)*Rl4x4*Ru4x4*complexconjugate(Rl4x4)*complexconjugate(Ru4x4))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_749 = Coupling(name = 'GC_749',
                  value = '(2*ee**2*complex(0,1)*Rl5x5*Ru4x4*complexconjugate(Rl5x5)*complexconjugate(Ru4x4))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_75 = Coupling(name = 'GC_75',
                 value = '(-2*cw*ee*complex(0,1)*NN2x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_750 = Coupling(name = 'GC_750',
                  value = '(2*ee**2*complex(0,1)*Rl6x6*Ru4x4*complexconjugate(Rl6x6)*complexconjugate(Ru4x4))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_751 = Coupling(name = 'GC_751',
                  value = '-(ee**2*complex(0,1)*Ru1x1*Ru4x4*complexconjugate(Ru1x1)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_752 = Coupling(name = 'GC_752',
                  value = '-(ee**2*complex(0,1)*Ru2x2*Ru4x4*complexconjugate(Ru2x2)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_753 = Coupling(name = 'GC_753',
                  value = '-(ee**2*complex(0,1)*Ru3x3*Ru4x4*complexconjugate(Ru3x3)*complexconjugate(Ru4x4))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_754 = Coupling(name = 'GC_754',
                  value = '(4*ee**2*complex(0,1)*Ru4x4**2*complexconjugate(Ru4x4)**2)/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_755 = Coupling(name = 'GC_755',
                  value = '-((complex(0,1)*G**2*Rd1x1*Ru4x4*complexconjugate(Rd1x1)*complexconjugate(Ru4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd1x1*Ru4x4*sw**2*complexconjugate(Rd1x1)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_756 = Coupling(name = 'GC_756',
                  value = '-((complex(0,1)*G**2*Rd2x2*Ru4x4*complexconjugate(Rd2x2)*complexconjugate(Ru4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd2x2*Ru4x4*sw**2*complexconjugate(Rd2x2)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_757 = Coupling(name = 'GC_757',
                  value = '-((complex(0,1)*G**2*Rd3x3*Ru4x4*complexconjugate(Rd3x3)*complexconjugate(Ru4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd3x3*Ru4x4*sw**2*complexconjugate(Rd3x3)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_758 = Coupling(name = 'GC_758',
                  value = '(complex(0,1)*G**2*Rd4x4*Ru4x4*complexconjugate(Rd4x4)*complexconjugate(Ru4x4))/(-1 + sw**2) - (complex(0,1)*G**2*Rd4x4*Ru4x4*sw**2*complexconjugate(Rd4x4)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_759 = Coupling(name = 'GC_759',
                  value = '(complex(0,1)*G**2*Rd5x5*Ru4x4*complexconjugate(Rd5x5)*complexconjugate(Ru4x4))/(-1 + sw**2) - (complex(0,1)*G**2*Rd5x5*Ru4x4*sw**2*complexconjugate(Rd5x5)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_76 = Coupling(name = 'GC_76',
                 value = '(-2*cw*ee*complex(0,1)*NN3x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_760 = Coupling(name = 'GC_760',
                  value = '(complex(0,1)*G**2*Rd6x6*Ru4x4*complexconjugate(Rd6x6)*complexconjugate(Ru4x4))/(-1 + sw**2) - (complex(0,1)*G**2*Rd6x6*Ru4x4*sw**2*complexconjugate(Rd6x6)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_761 = Coupling(name = 'GC_761',
                  value = '-((complex(0,1)*G**2*Ru1x1*Ru4x4*complexconjugate(Ru1x1)*complexconjugate(Ru4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru1x1*Ru4x4*sw**2*complexconjugate(Ru1x1)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_762 = Coupling(name = 'GC_762',
                  value = '-((complex(0,1)*G**2*Ru2x2*Ru4x4*complexconjugate(Ru2x2)*complexconjugate(Ru4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru2x2*Ru4x4*sw**2*complexconjugate(Ru2x2)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_763 = Coupling(name = 'GC_763',
                  value = '-((complex(0,1)*G**2*Ru3x3*Ru4x4*complexconjugate(Ru3x3)*complexconjugate(Ru4x4))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru3x3*Ru4x4*sw**2*complexconjugate(Ru3x3)*complexconjugate(Ru4x4))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_764 = Coupling(name = 'GC_764',
                  value = '(complex(0,1)*G**2*Ru4x4**2*complexconjugate(Ru4x4)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Ru4x4**2*sw**2*complexconjugate(Ru4x4)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_765 = Coupling(name = 'GC_765',
                  value = 'complex(0,1)*G*complexconjugate(Ru5x5)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_766 = Coupling(name = 'GC_766',
                  value = '(-2*ee*complex(0,1)*Ru5x5*complexconjugate(Ru5x5))/3.',
                  order = {'QED':1})

GC_767 = Coupling(name = 'GC_767',
                  value = '(2*ee*complex(0,1)*Ru5x5*complexconjugate(Ru5x5))/3.',
                  order = {'QED':1})

GC_768 = Coupling(name = 'GC_768',
                  value = '(8*ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5))/9.',
                  order = {'QED':2})

GC_769 = Coupling(name = 'GC_769',
                  value = '-(complex(0,1)*G*Ru5x5*complexconjugate(Ru5x5))',
                  order = {'QCD':1})

GC_77 = Coupling(name = 'GC_77',
                 value = '(-2*cw*ee*complex(0,1)*NN4x1*Ru6x6*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_770 = Coupling(name = 'GC_770',
                  value = 'complex(0,1)*G*Ru5x5*complexconjugate(Ru5x5)',
                  order = {'QCD':1})

GC_771 = Coupling(name = 'GC_771',
                  value = '(4*ee*complex(0,1)*G*Ru5x5*complexconjugate(Ru5x5))/3.',
                  order = {'QCD':1,'QED':1})

GC_772 = Coupling(name = 'GC_772',
                  value = 'complex(0,1)*G**2*Ru5x5*complexconjugate(Ru5x5)',
                  order = {'QCD':2})

GC_773 = Coupling(name = 'GC_773',
                  value = '(8*ee**2*complex(0,1)*Ru5x5*sw**2*complexconjugate(Ru5x5))/(9 - 9*sw**2)',
                  order = {'QED':2})

GC_774 = Coupling(name = 'GC_774',
                  value = '(ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_775 = Coupling(name = 'GC_775',
                  value = '(-2*cw*ee*complex(0,1)*Ru5x5*sw*complexconjugate(Ru5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_776 = Coupling(name = 'GC_776',
                  value = '(2*cw*ee*complex(0,1)*Ru5x5*sw*complexconjugate(Ru5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_777 = Coupling(name = 'GC_777',
                  value = '(8*cw*ee**2*complex(0,1)*Ru5x5*sw*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_778 = Coupling(name = 'GC_778',
                  value = '(4*cw*ee*complex(0,1)*G*Ru5x5*sw*complexconjugate(Ru5x5))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_779 = Coupling(name = 'GC_779',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_78 = Coupling(name = 'GC_78',
                 value = '(ee**2*complex(0,1))/(4.*sw**2*(-1 + sw**2))',
                 order = {'QED':2})

GC_780 = Coupling(name = 'GC_780',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_781 = Coupling(name = 'GC_781',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_782 = Coupling(name = 'GC_782',
                  value = '(2*cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Ru5x5)*cmath.sqrt(2))/(3 - 3*sw**2)',
                  order = {'QED':1})

GC_783 = Coupling(name = 'GC_783',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Ru5x5*complexconjugate(Rd1x1)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_784 = Coupling(name = 'GC_784',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Ru5x5*complexconjugate(Rd2x2)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_785 = Coupling(name = 'GC_785',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Ru5x5*complexconjugate(Rd3x3)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_786 = Coupling(name = 'GC_786',
                  value = '(-2*ee**2*complex(0,1)*Rd4x4*Ru5x5*complexconjugate(Rd4x4)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_787 = Coupling(name = 'GC_787',
                  value = '(-2*ee**2*complex(0,1)*Rd5x5*Ru5x5*complexconjugate(Rd5x5)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_788 = Coupling(name = 'GC_788',
                  value = '(-2*ee**2*complex(0,1)*Rd6x6*Ru5x5*complexconjugate(Rd6x6)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_789 = Coupling(name = 'GC_789',
                  value = '(ee**2*complex(0,1)*Rl1x1*Ru5x5*complexconjugate(Rl1x1)*complexconjugate(Ru5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_79 = Coupling(name = 'GC_79',
                 value = '(ee**2*complex(0,1))/(2.*sw**2*(-1 + sw**2))',
                 order = {'QED':2})

GC_790 = Coupling(name = 'GC_790',
                  value = '(ee**2*complex(0,1)*Rl2x2*Ru5x5*complexconjugate(Rl2x2)*complexconjugate(Ru5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_791 = Coupling(name = 'GC_791',
                  value = '(ee**2*complex(0,1)*Rl3x3*Ru5x5*complexconjugate(Rl3x3)*complexconjugate(Ru5x5))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_792 = Coupling(name = 'GC_792',
                  value = '(2*ee**2*complex(0,1)*Rl4x4*Ru5x5*complexconjugate(Rl4x4)*complexconjugate(Ru5x5))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_793 = Coupling(name = 'GC_793',
                  value = '(2*ee**2*complex(0,1)*Rl5x5*Ru5x5*complexconjugate(Rl5x5)*complexconjugate(Ru5x5))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_794 = Coupling(name = 'GC_794',
                  value = '(2*ee**2*complex(0,1)*Rl6x6*Ru5x5*complexconjugate(Rl6x6)*complexconjugate(Ru5x5))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_795 = Coupling(name = 'GC_795',
                  value = '-(ee**2*complex(0,1)*Ru1x1*Ru5x5*complexconjugate(Ru1x1)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_796 = Coupling(name = 'GC_796',
                  value = '-(ee**2*complex(0,1)*Ru2x2*Ru5x5*complexconjugate(Ru2x2)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_797 = Coupling(name = 'GC_797',
                  value = '-(ee**2*complex(0,1)*Ru3x3*Ru5x5*complexconjugate(Ru3x3)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_798 = Coupling(name = 'GC_798',
                  value = '(4*ee**2*complex(0,1)*Ru4x4*Ru5x5*complexconjugate(Ru4x4)*complexconjugate(Ru5x5))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_799 = Coupling(name = 'GC_799',
                  value = '(4*ee**2*complex(0,1)*Ru5x5**2*complexconjugate(Ru5x5)**2)/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_8 = Coupling(name = 'GC_8',
                value = '-G',
                order = {'QCD':1})

GC_80 = Coupling(name = 'GC_80',
                 value = '-(cw*ee*complex(0,1))/(2.*sw*(-1 + sw**2))',
                 order = {'QED':1})

GC_800 = Coupling(name = 'GC_800',
                  value = '-((complex(0,1)*G**2*Rd1x1*Ru5x5*complexconjugate(Rd1x1)*complexconjugate(Ru5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd1x1*Ru5x5*sw**2*complexconjugate(Rd1x1)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_801 = Coupling(name = 'GC_801',
                  value = '-((complex(0,1)*G**2*Rd2x2*Ru5x5*complexconjugate(Rd2x2)*complexconjugate(Ru5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd2x2*Ru5x5*sw**2*complexconjugate(Rd2x2)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_802 = Coupling(name = 'GC_802',
                  value = '-((complex(0,1)*G**2*Rd3x3*Ru5x5*complexconjugate(Rd3x3)*complexconjugate(Ru5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd3x3*Ru5x5*sw**2*complexconjugate(Rd3x3)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_803 = Coupling(name = 'GC_803',
                  value = '(complex(0,1)*G**2*Rd4x4*Ru5x5*complexconjugate(Rd4x4)*complexconjugate(Ru5x5))/(-1 + sw**2) - (complex(0,1)*G**2*Rd4x4*Ru5x5*sw**2*complexconjugate(Rd4x4)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_804 = Coupling(name = 'GC_804',
                  value = '(complex(0,1)*G**2*Rd5x5*Ru5x5*complexconjugate(Rd5x5)*complexconjugate(Ru5x5))/(-1 + sw**2) - (complex(0,1)*G**2*Rd5x5*Ru5x5*sw**2*complexconjugate(Rd5x5)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_805 = Coupling(name = 'GC_805',
                  value = '(complex(0,1)*G**2*Rd6x6*Ru5x5*complexconjugate(Rd6x6)*complexconjugate(Ru5x5))/(-1 + sw**2) - (complex(0,1)*G**2*Rd6x6*Ru5x5*sw**2*complexconjugate(Rd6x6)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_806 = Coupling(name = 'GC_806',
                  value = '-((complex(0,1)*G**2*Ru1x1*Ru5x5*complexconjugate(Ru1x1)*complexconjugate(Ru5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru1x1*Ru5x5*sw**2*complexconjugate(Ru1x1)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_807 = Coupling(name = 'GC_807',
                  value = '-((complex(0,1)*G**2*Ru2x2*Ru5x5*complexconjugate(Ru2x2)*complexconjugate(Ru5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru2x2*Ru5x5*sw**2*complexconjugate(Ru2x2)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_808 = Coupling(name = 'GC_808',
                  value = '-((complex(0,1)*G**2*Ru3x3*Ru5x5*complexconjugate(Ru3x3)*complexconjugate(Ru5x5))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru3x3*Ru5x5*sw**2*complexconjugate(Ru3x3)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_809 = Coupling(name = 'GC_809',
                  value = '(complex(0,1)*G**2*Ru4x4*Ru5x5*complexconjugate(Ru4x4)*complexconjugate(Ru5x5))/(-1 + sw**2) - (complex(0,1)*G**2*Ru4x4*Ru5x5*sw**2*complexconjugate(Ru4x4)*complexconjugate(Ru5x5))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_81 = Coupling(name = 'GC_81',
                 value = '(cw*ee*complex(0,1))/(2.*sw*(-1 + sw**2))',
                 order = {'QED':1})

GC_810 = Coupling(name = 'GC_810',
                  value = '(complex(0,1)*G**2*Ru5x5**2*complexconjugate(Ru5x5)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Ru5x5**2*sw**2*complexconjugate(Ru5x5)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_811 = Coupling(name = 'GC_811',
                  value = 'complex(0,1)*G*complexconjugate(Ru6x6)*cmath.sqrt(2)',
                  order = {'QCD':1})

GC_812 = Coupling(name = 'GC_812',
                  value = '(-2*ee*complex(0,1)*Ru6x6*complexconjugate(Ru6x6))/3.',
                  order = {'QED':1})

GC_813 = Coupling(name = 'GC_813',
                  value = '(2*ee*complex(0,1)*Ru6x6*complexconjugate(Ru6x6))/3.',
                  order = {'QED':1})

GC_814 = Coupling(name = 'GC_814',
                  value = '(8*ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6))/9.',
                  order = {'QED':2})

GC_815 = Coupling(name = 'GC_815',
                  value = '-(complex(0,1)*G*Ru6x6*complexconjugate(Ru6x6))',
                  order = {'QCD':1})

GC_816 = Coupling(name = 'GC_816',
                  value = 'complex(0,1)*G*Ru6x6*complexconjugate(Ru6x6)',
                  order = {'QCD':1})

GC_817 = Coupling(name = 'GC_817',
                  value = '(4*ee*complex(0,1)*G*Ru6x6*complexconjugate(Ru6x6))/3.',
                  order = {'QCD':1,'QED':1})

GC_818 = Coupling(name = 'GC_818',
                  value = 'complex(0,1)*G**2*Ru6x6*complexconjugate(Ru6x6)',
                  order = {'QCD':2})

GC_819 = Coupling(name = 'GC_819',
                  value = '(8*ee**2*complex(0,1)*Ru6x6*sw**2*complexconjugate(Ru6x6))/(9 - 9*sw**2)',
                  order = {'QED':2})

GC_82 = Coupling(name = 'GC_82',
                 value = '-(cw*ee*complex(0,1)*sw)/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_820 = Coupling(name = 'GC_820',
                  value = '(ee**2*complex(0,1)*Ru6x6*complexconjugate(Ru6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_821 = Coupling(name = 'GC_821',
                  value = '(-2*cw*ee*complex(0,1)*Ru6x6*sw*complexconjugate(Ru6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_822 = Coupling(name = 'GC_822',
                  value = '(2*cw*ee*complex(0,1)*Ru6x6*sw*complexconjugate(Ru6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_823 = Coupling(name = 'GC_823',
                  value = '(8*cw*ee**2*complex(0,1)*Ru6x6*sw*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_824 = Coupling(name = 'GC_824',
                  value = '(4*cw*ee*complex(0,1)*G*Ru6x6*sw*complexconjugate(Ru6x6))/(3.*(-1 + sw**2))',
                  order = {'QCD':1,'QED':1})

GC_825 = Coupling(name = 'GC_825',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN1x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_826 = Coupling(name = 'GC_826',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN2x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_827 = Coupling(name = 'GC_827',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN3x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_828 = Coupling(name = 'GC_828',
                  value = '(-2*cw*ee*complex(0,1)*complexconjugate(NN4x1)*complexconjugate(Ru6x6)*cmath.sqrt(2))/(3.*(-1 + sw**2))',
                  order = {'QED':1})

GC_829 = Coupling(name = 'GC_829',
                  value = '-(ee**2*complex(0,1)*Rd1x1*Ru6x6*complexconjugate(Rd1x1)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_83 = Coupling(name = 'GC_83',
                 value = '(2*cw*ee*complex(0,1)*sw)/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_830 = Coupling(name = 'GC_830',
                  value = '-(ee**2*complex(0,1)*Rd2x2*Ru6x6*complexconjugate(Rd2x2)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_831 = Coupling(name = 'GC_831',
                  value = '-(ee**2*complex(0,1)*Rd3x3*Ru6x6*complexconjugate(Rd3x3)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_832 = Coupling(name = 'GC_832',
                  value = '(-2*ee**2*complex(0,1)*Rd4x4*Ru6x6*complexconjugate(Rd4x4)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_833 = Coupling(name = 'GC_833',
                  value = '(-2*ee**2*complex(0,1)*Rd5x5*Ru6x6*complexconjugate(Rd5x5)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_834 = Coupling(name = 'GC_834',
                  value = '(-2*ee**2*complex(0,1)*Rd6x6*Ru6x6*complexconjugate(Rd6x6)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_835 = Coupling(name = 'GC_835',
                  value = '(ee**2*complex(0,1)*Rl1x1*Ru6x6*complexconjugate(Rl1x1)*complexconjugate(Ru6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_836 = Coupling(name = 'GC_836',
                  value = '(ee**2*complex(0,1)*Rl2x2*Ru6x6*complexconjugate(Rl2x2)*complexconjugate(Ru6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_837 = Coupling(name = 'GC_837',
                  value = '(ee**2*complex(0,1)*Rl3x3*Ru6x6*complexconjugate(Rl3x3)*complexconjugate(Ru6x6))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_838 = Coupling(name = 'GC_838',
                  value = '(2*ee**2*complex(0,1)*Rl4x4*Ru6x6*complexconjugate(Rl4x4)*complexconjugate(Ru6x6))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_839 = Coupling(name = 'GC_839',
                  value = '(2*ee**2*complex(0,1)*Rl5x5*Ru6x6*complexconjugate(Rl5x5)*complexconjugate(Ru6x6))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_84 = Coupling(name = 'GC_84',
                 value = '-((cw*ee*complex(0,1)*sw)/(-1 + sw**2))',
                 order = {'QED':1})

GC_840 = Coupling(name = 'GC_840',
                  value = '(2*ee**2*complex(0,1)*Rl6x6*Ru6x6*complexconjugate(Rl6x6)*complexconjugate(Ru6x6))/(3 - 3*sw**2)',
                  order = {'QED':2})

GC_841 = Coupling(name = 'GC_841',
                  value = '-(ee**2*complex(0,1)*Ru1x1*Ru6x6*complexconjugate(Ru1x1)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_842 = Coupling(name = 'GC_842',
                  value = '-(ee**2*complex(0,1)*Ru2x2*Ru6x6*complexconjugate(Ru2x2)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_843 = Coupling(name = 'GC_843',
                  value = '-(ee**2*complex(0,1)*Ru3x3*Ru6x6*complexconjugate(Ru3x3)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_844 = Coupling(name = 'GC_844',
                  value = '(4*ee**2*complex(0,1)*Ru4x4*Ru6x6*complexconjugate(Ru4x4)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_845 = Coupling(name = 'GC_845',
                  value = '(4*ee**2*complex(0,1)*Ru5x5*Ru6x6*complexconjugate(Ru5x5)*complexconjugate(Ru6x6))/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_846 = Coupling(name = 'GC_846',
                  value = '(4*ee**2*complex(0,1)*Ru6x6**2*complexconjugate(Ru6x6)**2)/(9.*(-1 + sw**2))',
                  order = {'QED':2})

GC_847 = Coupling(name = 'GC_847',
                  value = '-((complex(0,1)*G**2*Rd1x1*Ru6x6*complexconjugate(Rd1x1)*complexconjugate(Ru6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd1x1*Ru6x6*sw**2*complexconjugate(Rd1x1)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_848 = Coupling(name = 'GC_848',
                  value = '-((complex(0,1)*G**2*Rd2x2*Ru6x6*complexconjugate(Rd2x2)*complexconjugate(Ru6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd2x2*Ru6x6*sw**2*complexconjugate(Rd2x2)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_849 = Coupling(name = 'GC_849',
                  value = '-((complex(0,1)*G**2*Rd3x3*Ru6x6*complexconjugate(Rd3x3)*complexconjugate(Ru6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Rd3x3*Ru6x6*sw**2*complexconjugate(Rd3x3)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_85 = Coupling(name = 'GC_85',
                 value = '(cw*ee**2)/(-2 + 2*sw**2)',
                 order = {'QED':2})

GC_850 = Coupling(name = 'GC_850',
                  value = '(complex(0,1)*G**2*Rd4x4*Ru6x6*complexconjugate(Rd4x4)*complexconjugate(Ru6x6))/(-1 + sw**2) - (complex(0,1)*G**2*Rd4x4*Ru6x6*sw**2*complexconjugate(Rd4x4)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_851 = Coupling(name = 'GC_851',
                  value = '(complex(0,1)*G**2*Rd5x5*Ru6x6*complexconjugate(Rd5x5)*complexconjugate(Ru6x6))/(-1 + sw**2) - (complex(0,1)*G**2*Rd5x5*Ru6x6*sw**2*complexconjugate(Rd5x5)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_852 = Coupling(name = 'GC_852',
                  value = '(complex(0,1)*G**2*Rd6x6*Ru6x6*complexconjugate(Rd6x6)*complexconjugate(Ru6x6))/(-1 + sw**2) - (complex(0,1)*G**2*Rd6x6*Ru6x6*sw**2*complexconjugate(Rd6x6)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_853 = Coupling(name = 'GC_853',
                  value = '-((complex(0,1)*G**2*Ru1x1*Ru6x6*complexconjugate(Ru1x1)*complexconjugate(Ru6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru1x1*Ru6x6*sw**2*complexconjugate(Ru1x1)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_854 = Coupling(name = 'GC_854',
                  value = '-((complex(0,1)*G**2*Ru2x2*Ru6x6*complexconjugate(Ru2x2)*complexconjugate(Ru6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru2x2*Ru6x6*sw**2*complexconjugate(Ru2x2)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_855 = Coupling(name = 'GC_855',
                  value = '-((complex(0,1)*G**2*Ru3x3*Ru6x6*complexconjugate(Ru3x3)*complexconjugate(Ru6x6))/(-1 + sw**2)) + (complex(0,1)*G**2*Ru3x3*Ru6x6*sw**2*complexconjugate(Ru3x3)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_856 = Coupling(name = 'GC_856',
                  value = '(complex(0,1)*G**2*Ru4x4*Ru6x6*complexconjugate(Ru4x4)*complexconjugate(Ru6x6))/(-1 + sw**2) - (complex(0,1)*G**2*Ru4x4*Ru6x6*sw**2*complexconjugate(Ru4x4)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_857 = Coupling(name = 'GC_857',
                  value = '(complex(0,1)*G**2*Ru5x5*Ru6x6*complexconjugate(Ru5x5)*complexconjugate(Ru6x6))/(-1 + sw**2) - (complex(0,1)*G**2*Ru5x5*Ru6x6*sw**2*complexconjugate(Ru5x5)*complexconjugate(Ru6x6))/(-1 + sw**2)',
                  order = {'QCD':2})

GC_858 = Coupling(name = 'GC_858',
                  value = '(complex(0,1)*G**2*Ru6x6**2*complexconjugate(Ru6x6)**2)/(-1 + sw**2) - (complex(0,1)*G**2*Ru6x6**2*sw**2*complexconjugate(Ru6x6)**2)/(-1 + sw**2)',
                  order = {'QCD':2})

GC_859 = Coupling(name = 'GC_859',
                  value = '-((ee*complex(0,1)*Rl1x1*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_86 = Coupling(name = 'GC_86',
                 value = '(cw*ee*complex(0,1))/(2*sw - 2*sw**3)',
                 order = {'QED':1})

GC_860 = Coupling(name = 'GC_860',
                  value = '-((ee*complex(0,1)*Rl2x2*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_861 = Coupling(name = 'GC_861',
                  value = '-((ee*complex(0,1)*Rl3x3*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_862 = Coupling(name = 'GC_862',
                  value = '-((ee*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_863 = Coupling(name = 'GC_863',
                  value = '-((ee*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_864 = Coupling(name = 'GC_864',
                  value = '-((ee*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(UU1x1))/sw)',
                  order = {'QED':1})

GC_865 = Coupling(name = 'GC_865',
                  value = 'complex(0,1)*Rl6x6*ye3x3*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_866 = Coupling(name = 'GC_866',
                  value = 'complex(0,1)*Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_867 = Coupling(name = 'GC_867',
                  value = 'complex(0,1)*ye3x3*complexconjugate(Rn3x3)*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_868 = Coupling(name = 'GC_868',
                  value = 'complex(0,1)*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_869 = Coupling(name = 'GC_869',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN1x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_87 = Coupling(name = 'GC_87',
                 value = '(ee**2*complex(0,1))/(2*sw**2 - 2*sw**4)',
                 order = {'QED':2})

GC_870 = Coupling(name = 'GC_870',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN2x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_871 = Coupling(name = 'GC_871',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN3x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_872 = Coupling(name = 'GC_872',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(UU1x1))/sw) - (ee*complex(0,1)*NN4x3*complexconjugate(UU1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_873 = Coupling(name = 'GC_873',
                  value = 'ee*complex(0,1)*UU1x1*complexconjugate(UU1x1) + ee*complex(0,1)*UU1x2*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_874 = Coupling(name = 'GC_874',
                  value = '-((cw*ee*complex(0,1)*UU1x1*complexconjugate(UU1x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*UU1x1*complexconjugate(UU1x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*UU1x2*complexconjugate(UU1x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*UU1x2*complexconjugate(UU1x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_875 = Coupling(name = 'GC_875',
                  value = 'ee*complex(0,1)*UU2x1*complexconjugate(UU1x1) + ee*complex(0,1)*UU2x2*complexconjugate(UU1x2)',
                  order = {'QED':1})

GC_876 = Coupling(name = 'GC_876',
                  value = '-((cw*ee*complex(0,1)*UU2x1*complexconjugate(UU1x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*UU2x1*complexconjugate(UU1x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*UU2x2*complexconjugate(UU1x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*UU2x2*complexconjugate(UU1x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_877 = Coupling(name = 'GC_877',
                  value = '-((ee*complex(0,1)*Rl1x1*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_878 = Coupling(name = 'GC_878',
                  value = '-((ee*complex(0,1)*Rl2x2*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_879 = Coupling(name = 'GC_879',
                  value = '-((ee*complex(0,1)*Rl3x3*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_88 = Coupling(name = 'GC_88',
                 value = '(cw*ee*complex(0,1))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*sw)/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_880 = Coupling(name = 'GC_880',
                  value = '-((ee*complex(0,1)*Rd1x1*complexconjugate(CKM1x1)*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_881 = Coupling(name = 'GC_881',
                  value = '-((ee*complex(0,1)*Rd2x2*complexconjugate(CKM2x2)*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_882 = Coupling(name = 'GC_882',
                  value = '-((ee*complex(0,1)*Rd3x3*complexconjugate(CKM3x3)*complexconjugate(UU2x1))/sw)',
                  order = {'QED':1})

GC_883 = Coupling(name = 'GC_883',
                  value = 'complex(0,1)*Rl6x6*ye3x3*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_884 = Coupling(name = 'GC_884',
                  value = 'complex(0,1)*Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_885 = Coupling(name = 'GC_885',
                  value = 'complex(0,1)*ye3x3*complexconjugate(Rn3x3)*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_886 = Coupling(name = 'GC_886',
                  value = 'complex(0,1)*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_887 = Coupling(name = 'GC_887',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN1x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_888 = Coupling(name = 'GC_888',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN2x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_889 = Coupling(name = 'GC_889',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN3x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_89 = Coupling(name = 'GC_89',
                 value = '-(cw*ee*complex(0,1))/(2.*sw*(-1 + sw**2)) + (2*cw*ee*complex(0,1)*sw)/(3.*(-1 + sw**2))',
                 order = {'QED':1})

GC_890 = Coupling(name = 'GC_890',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(UU2x1))/sw) - (ee*complex(0,1)*NN4x3*complexconjugate(UU2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_891 = Coupling(name = 'GC_891',
                  value = 'ee*complex(0,1)*UU1x1*complexconjugate(UU2x1) + ee*complex(0,1)*UU1x2*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_892 = Coupling(name = 'GC_892',
                  value = '-((cw*ee*complex(0,1)*UU1x1*complexconjugate(UU2x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*UU1x1*complexconjugate(UU2x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*UU1x2*complexconjugate(UU2x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*UU1x2*complexconjugate(UU2x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_893 = Coupling(name = 'GC_893',
                  value = 'ee*complex(0,1)*UU2x1*complexconjugate(UU2x1) + ee*complex(0,1)*UU2x2*complexconjugate(UU2x2)',
                  order = {'QED':1})

GC_894 = Coupling(name = 'GC_894',
                  value = '-((cw*ee*complex(0,1)*UU2x1*complexconjugate(UU2x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*UU2x1*complexconjugate(UU2x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*UU2x2*complexconjugate(UU2x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*UU2x2*complexconjugate(UU2x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_895 = Coupling(name = 'GC_895',
                  value = '-((ee*complex(0,1)*Rn1x1*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_896 = Coupling(name = 'GC_896',
                  value = '-((ee*complex(0,1)*Rn2x2*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_897 = Coupling(name = 'GC_897',
                  value = '-((ee*complex(0,1)*Rn3x3*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_898 = Coupling(name = 'GC_898',
                  value = '-((CKM1x1*ee*complex(0,1)*Ru1x1*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_899 = Coupling(name = 'GC_899',
                  value = '-((CKM2x2*ee*complex(0,1)*Ru2x2*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_9 = Coupling(name = 'GC_9',
                value = 'complex(0,1)*G',
                order = {'QCD':1})

GC_90 = Coupling(name = 'GC_90',
                 value = '(cw*ee*complex(0,1))/(2.*sw*(-1 + sw**2)) - (cw*ee*complex(0,1)*sw)/(-1 + sw**2)',
                 order = {'QED':1})

GC_900 = Coupling(name = 'GC_900',
                  value = '-((CKM3x3*ee*complex(0,1)*Ru3x3*complexconjugate(VV1x1))/sw)',
                  order = {'QED':1})

GC_901 = Coupling(name = 'GC_901',
                  value = 'CKM3x3*complex(0,1)*Ru6x6*yu3x3*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_902 = Coupling(name = 'GC_902',
                  value = 'CKM3x3*complex(0,1)*yu3x3*complexconjugate(Rd3x3)*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_903 = Coupling(name = 'GC_903',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN1x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_904 = Coupling(name = 'GC_904',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN2x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_905 = Coupling(name = 'GC_905',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN3x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_906 = Coupling(name = 'GC_906',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(VV1x1))/sw) + (ee*complex(0,1)*NN4x4*complexconjugate(VV1x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_907 = Coupling(name = 'GC_907',
                  value = 'ee*complex(0,1)*VV1x1*complexconjugate(VV1x1) + ee*complex(0,1)*VV1x2*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_908 = Coupling(name = 'GC_908',
                  value = '-((cw*ee*complex(0,1)*VV1x1*complexconjugate(VV1x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*VV1x1*complexconjugate(VV1x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*VV1x2*complexconjugate(VV1x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*VV1x2*complexconjugate(VV1x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_909 = Coupling(name = 'GC_909',
                  value = 'ee*complex(0,1)*VV2x1*complexconjugate(VV1x1) + ee*complex(0,1)*VV2x2*complexconjugate(VV1x2)',
                  order = {'QED':1})

GC_91 = Coupling(name = 'GC_91',
                 value = '-(cw*ee*complex(0,1))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw)/(-1 + sw**2)',
                 order = {'QED':1})

GC_910 = Coupling(name = 'GC_910',
                  value = '-((cw*ee*complex(0,1)*VV2x1*complexconjugate(VV1x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*VV2x1*complexconjugate(VV1x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*VV2x2*complexconjugate(VV1x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*VV2x2*complexconjugate(VV1x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_911 = Coupling(name = 'GC_911',
                  value = '-((ee*complex(0,1)*Rn1x1*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_912 = Coupling(name = 'GC_912',
                  value = '-((ee*complex(0,1)*Rn2x2*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_913 = Coupling(name = 'GC_913',
                  value = '-((ee*complex(0,1)*Rn3x3*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_914 = Coupling(name = 'GC_914',
                  value = '-((CKM1x1*ee*complex(0,1)*Ru1x1*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_915 = Coupling(name = 'GC_915',
                  value = '-((CKM2x2*ee*complex(0,1)*Ru2x2*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_916 = Coupling(name = 'GC_916',
                  value = '-((CKM3x3*ee*complex(0,1)*Ru3x3*complexconjugate(VV2x1))/sw)',
                  order = {'QED':1})

GC_917 = Coupling(name = 'GC_917',
                  value = 'CKM3x3*complex(0,1)*Ru6x6*yu3x3*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_918 = Coupling(name = 'GC_918',
                  value = 'CKM3x3*complex(0,1)*yu3x3*complexconjugate(Rd3x3)*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_919 = Coupling(name = 'GC_919',
                  value = '-((ee*complex(0,1)*NN1x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN1x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_92 = Coupling(name = 'GC_92',
                 value = '-((cw*ee**2*complex(0,1))/(sw*(-1 + sw**2))) + (2*cw*ee**2*complex(0,1)*sw)/(-1 + sw**2)',
                 order = {'QED':2})

GC_920 = Coupling(name = 'GC_920',
                  value = '-((ee*complex(0,1)*NN2x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN2x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_921 = Coupling(name = 'GC_921',
                  value = '-((ee*complex(0,1)*NN3x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN3x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_922 = Coupling(name = 'GC_922',
                  value = '-((ee*complex(0,1)*NN4x2*complexconjugate(VV2x1))/sw) + (ee*complex(0,1)*NN4x4*complexconjugate(VV2x2))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})

GC_923 = Coupling(name = 'GC_923',
                  value = 'ee*complex(0,1)*VV1x1*complexconjugate(VV2x1) + ee*complex(0,1)*VV1x2*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_924 = Coupling(name = 'GC_924',
                  value = '-((cw*ee*complex(0,1)*VV1x1*complexconjugate(VV2x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*VV1x1*complexconjugate(VV2x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*VV1x2*complexconjugate(VV2x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*VV1x2*complexconjugate(VV2x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_925 = Coupling(name = 'GC_925',
                  value = 'ee*complex(0,1)*VV2x1*complexconjugate(VV2x1) + ee*complex(0,1)*VV2x2*complexconjugate(VV2x2)',
                  order = {'QED':1})

GC_926 = Coupling(name = 'GC_926',
                  value = '-((cw*ee*complex(0,1)*VV2x1*complexconjugate(VV2x1))/(sw*(-1 + sw**2))) + (cw*ee*complex(0,1)*sw*VV2x1*complexconjugate(VV2x1))/(-1 + sw**2) - (cw*ee*complex(0,1)*VV2x2*complexconjugate(VV2x2))/(2.*sw*(-1 + sw**2)) + (cw*ee*complex(0,1)*sw*VV2x2*complexconjugate(VV2x2))/(-1 + sw**2)',
                  order = {'QED':1})

GC_927 = Coupling(name = 'GC_927',
                  value = 'CKM3x3*complex(0,1)*Ru3x3*UU1x2*complexconjugate(yd3x3)',
                  order = {'QED':1})

GC_928 = Coupling(name = 'GC_928',
                  value = 'CKM3x3*complex(0,1)*Ru3x3*UU2x2*complexconjugate(yd3x3)',
                  order = {'QED':1})

GC_929 = Coupling(name = 'GC_929',
                  value = 'CKM3x3*complex(0,1)*UU1x2*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                  order = {'QED':1})

GC_93 = Coupling(name = 'GC_93',
                 value = '(2*ee**2*complex(0,1))/(-1 + sw**2) - (ee**2*complex(0,1))/(2.*sw**2*(-1 + sw**2)) - (2*ee**2*complex(0,1)*sw**2)/(-1 + sw**2)',
                 order = {'QED':2})

GC_930 = Coupling(name = 'GC_930',
                  value = 'CKM3x3*complex(0,1)*UU2x2*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                  order = {'QED':1})

GC_931 = Coupling(name = 'GC_931',
                  value = '-((CKM3x3*Ru6x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/cmath.sqrt(2))',
                  order = {'QED':2})

GC_932 = Coupling(name = 'GC_932',
                  value = '(CKM3x3*Ru6x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/cmath.sqrt(2)',
                  order = {'QED':2})

GC_933 = Coupling(name = 'GC_933',
                  value = '-(complex(0,1)*Rd3x3*Rl6x6*ye3x3*complexconjugate(Rd6x6)*complexconjugate(Rl3x3)*complexconjugate(yd3x3))',
                  order = {'QED':2})

GC_934 = Coupling(name = 'GC_934',
                  value = '-(CKM3x3*complex(0,1)*Rl6x6*Ru3x3*ye3x3*complexconjugate(Rd6x6)*complexconjugate(Rn3x3)*complexconjugate(yd3x3))',
                  order = {'QED':2})

GC_935 = Coupling(name = 'GC_935',
                  value = '(complex(0,1)*NN1x3*Rd3x3*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN1x3*Rd3x3*sw**2*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_936 = Coupling(name = 'GC_936',
                  value = '(complex(0,1)*NN2x3*Rd3x3*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN2x3*Rd3x3*sw**2*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_937 = Coupling(name = 'GC_937',
                  value = '(complex(0,1)*NN3x3*Rd3x3*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN3x3*Rd3x3*sw**2*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_938 = Coupling(name = 'GC_938',
                  value = '(complex(0,1)*NN4x3*Rd3x3*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN4x3*Rd3x3*sw**2*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_939 = Coupling(name = 'GC_939',
                  value = '(complex(0,1)*NN1x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN1x3*sw**2*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_94 = Coupling(name = 'GC_94',
                 value = '(ee*complex(0,1)*complexconjugate(CKM1x1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_940 = Coupling(name = 'GC_940',
                  value = '(complex(0,1)*NN2x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN2x3*sw**2*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_941 = Coupling(name = 'GC_941',
                  value = '(complex(0,1)*NN3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN3x3*sw**2*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_942 = Coupling(name = 'GC_942',
                  value = '(complex(0,1)*NN4x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*NN4x3*sw**2*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_943 = Coupling(name = 'GC_943',
                  value = '(complex(0,1)*Rd3x3*Rd6x6*yd3x3*complexconjugate(Rd3x3)*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2) - (complex(0,1)*Rd3x3*Rd6x6*sw**2*yd3x3*complexconjugate(Rd3x3)*complexconjugate(Rd6x6)*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_944 = Coupling(name = 'GC_944',
                  value = '(CKM3x3*complex(0,1)*Rd6x6*Ru3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)*complexconjugate(yd3x3))/(-1 + sw**2) - (CKM3x3*complex(0,1)*Rd6x6*Ru3x3*sw**2*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Rd6x6)*complexconjugate(Ru3x3)*complexconjugate(yd3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_945 = Coupling(name = 'GC_945',
                  value = 'complex(0,1)*Rn3x3*UU1x2*complexconjugate(ye3x3)',
                  order = {'QED':1})

GC_946 = Coupling(name = 'GC_946',
                  value = 'complex(0,1)*Rn3x3*UU2x2*complexconjugate(ye3x3)',
                  order = {'QED':1})

GC_947 = Coupling(name = 'GC_947',
                  value = 'complex(0,1)*UU1x2*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                  order = {'QED':1})

GC_948 = Coupling(name = 'GC_948',
                  value = 'complex(0,1)*UU2x2*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                  order = {'QED':1})

GC_949 = Coupling(name = 'GC_949',
                  value = '-(complex(0,1)*Rd6x6*Rl3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(Rl6x6)*complexconjugate(ye3x3))',
                  order = {'QED':2})

GC_95 = Coupling(name = 'GC_95',
                 value = '(ee*complex(0,1)*complexconjugate(CKM2x2))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_950 = Coupling(name = 'GC_950',
                  value = '-(complex(0,1)*Rd6x6*Rn3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Rl6x6)*complexconjugate(Ru3x3)*complexconjugate(ye3x3))',
                  order = {'QED':2})

GC_951 = Coupling(name = 'GC_951',
                  value = '(complex(0,1)*NN1x3*Rl3x3*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN1x3*Rl3x3*sw**2*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_952 = Coupling(name = 'GC_952',
                  value = '(complex(0,1)*NN2x3*Rl3x3*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN2x3*Rl3x3*sw**2*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_953 = Coupling(name = 'GC_953',
                  value = '(complex(0,1)*NN3x3*Rl3x3*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN3x3*Rl3x3*sw**2*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_954 = Coupling(name = 'GC_954',
                  value = '(complex(0,1)*NN4x3*Rl3x3*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN4x3*Rl3x3*sw**2*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_955 = Coupling(name = 'GC_955',
                  value = '(complex(0,1)*NN1x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN1x3*sw**2*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_956 = Coupling(name = 'GC_956',
                  value = '(complex(0,1)*NN2x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN2x3*sw**2*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_957 = Coupling(name = 'GC_957',
                  value = '(complex(0,1)*NN3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN3x3*sw**2*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_958 = Coupling(name = 'GC_958',
                  value = '(complex(0,1)*NN4x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*NN4x3*sw**2*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_959 = Coupling(name = 'GC_959',
                  value = '-(ee**2*complex(0,1)*Rl3x3*Rl6x6*complexconjugate(Rl3x3)*complexconjugate(Rl6x6))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl3x3*Rl6x6*ye3x3*complexconjugate(Rl3x3)*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*Rl3x3*Rl6x6*sw**2*ye3x3*complexconjugate(Rl3x3)*complexconjugate(Rl6x6)*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_96 = Coupling(name = 'GC_96',
                 value = '(ee*complex(0,1)*complexconjugate(CKM3x3))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})

GC_960 = Coupling(name = 'GC_960',
                  value = '-(ee**2*complex(0,1)*Rl6x6*complexconjugate(Rl6x6))/(2.*(-1 + sw**2)) + (complex(0,1)*Rl6x6*Rn3x3*ye3x3*complexconjugate(Rl6x6)*complexconjugate(Rn3x3)*complexconjugate(ye3x3))/(-1 + sw**2) - (complex(0,1)*Rl6x6*Rn3x3*sw**2*ye3x3*complexconjugate(Rl6x6)*complexconjugate(Rn3x3)*complexconjugate(ye3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_961 = Coupling(name = 'GC_961',
                  value = 'complex(0,1)*Rd3x3*VV1x2*complexconjugate(CKM3x3)*complexconjugate(yu3x3)',
                  order = {'QED':1})

GC_962 = Coupling(name = 'GC_962',
                  value = 'complex(0,1)*Rd3x3*VV2x2*complexconjugate(CKM3x3)*complexconjugate(yu3x3)',
                  order = {'QED':1})

GC_963 = Coupling(name = 'GC_963',
                  value = 'complex(0,1)*VV1x2*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                  order = {'QED':1})

GC_964 = Coupling(name = 'GC_964',
                  value = 'complex(0,1)*VV2x2*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                  order = {'QED':1})

GC_965 = Coupling(name = 'GC_965',
                  value = '-((Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/cmath.sqrt(2))',
                  order = {'QED':2})

GC_966 = Coupling(name = 'GC_966',
                  value = '(Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/cmath.sqrt(2)',
                  order = {'QED':2})

GC_967 = Coupling(name = 'GC_967',
                  value = '(complex(0,1)*NN1x4*Ru3x3*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN1x4*Ru3x3*sw**2*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_968 = Coupling(name = 'GC_968',
                  value = '(complex(0,1)*NN2x4*Ru3x3*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN2x4*Ru3x3*sw**2*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_969 = Coupling(name = 'GC_969',
                  value = '(complex(0,1)*NN3x4*Ru3x3*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN3x4*Ru3x3*sw**2*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '(cw*ee*complex(0,1)*Rd1x1*complexconjugate(NN1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd1x1*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd1x1*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                 order = {'QED':1})

GC_970 = Coupling(name = 'GC_970',
                  value = '(complex(0,1)*NN4x4*Ru3x3*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN4x4*Ru3x3*sw**2*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_971 = Coupling(name = 'GC_971',
                  value = '(complex(0,1)*NN1x4*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN1x4*sw**2*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_972 = Coupling(name = 'GC_972',
                  value = '(complex(0,1)*NN2x4*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN2x4*sw**2*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_973 = Coupling(name = 'GC_973',
                  value = '(complex(0,1)*NN3x4*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN3x4*sw**2*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_974 = Coupling(name = 'GC_974',
                  value = '(complex(0,1)*NN4x4*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*NN4x4*sw**2*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':1})

GC_975 = Coupling(name = 'GC_975',
                  value = '(CKM3x3*complex(0,1)*Rd3x3*Ru6x6*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2) - (CKM3x3*complex(0,1)*Rd3x3*Ru6x6*sw**2*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Rd3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_976 = Coupling(name = 'GC_976',
                  value = '(complex(0,1)*Ru3x3*Ru6x6*yu3x3*complexconjugate(Ru3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2) - (complex(0,1)*Ru3x3*Ru6x6*sw**2*yu3x3*complexconjugate(Ru3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3))/(-1 + sw**2)',
                  order = {'QED':2})

GC_977 = Coupling(name = 'GC_977',
                  value = '-((complex(0,1)*yd3x3*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_978 = Coupling(name = 'GC_978',
                  value = '-((complex(0,1)*ye3x3*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_979 = Coupling(name = 'GC_979',
                  value = '-((complex(0,1)*yu3x3*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_98 = Coupling(name = 'GC_98',
                 value = '(cw*ee*complex(0,1)*Rd2x2*complexconjugate(NN1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd2x2*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd2x2*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                 order = {'QED':1})

GC_980 = Coupling(name = 'GC_980',
                  value = '-((complex(0,1)*complexconjugate(yd3x3)*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_981 = Coupling(name = 'GC_981',
                  value = '-((complex(0,1)*complexconjugate(ye3x3)*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_982 = Coupling(name = 'GC_982',
                  value = '-((complex(0,1)*complexconjugate(yu3x3)*cmath.cos(alp))/cmath.sqrt(2))',
                  order = {'QED':1})

GC_983 = Coupling(name = 'GC_983',
                  value = '-(ee**2*complex(0,1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_984 = Coupling(name = 'GC_984',
                  value = '(ee**2*complex(0,1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_985 = Coupling(name = 'GC_985',
                  value = '-(ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4)*cmath.cos(2*alp))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_986 = Coupling(name = 'GC_986',
                  value = '(ee**2*complex(0,1)*Rd4x4*complexconjugate(Rd4x4)*cmath.cos(2*alp))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_987 = Coupling(name = 'GC_987',
                  value = '-(ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5)*cmath.cos(2*alp))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_988 = Coupling(name = 'GC_988',
                  value = '(ee**2*complex(0,1)*Rd5x5*complexconjugate(Rd5x5)*cmath.cos(2*alp))/(6.*(-1 + sw**2))',
                  order = {'QED':2})

GC_989 = Coupling(name = 'GC_989',
                  value = '-(ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)*cmath.cos(2*alp))/(2.*(-1 + sw**2))',
                  order = {'QED':2})

GC_99 = Coupling(name = 'GC_99',
                 value = '(cw*ee*complex(0,1)*Rd3x3*complexconjugate(NN1x1))/(3.*(-1 + sw**2)*cmath.sqrt(2)) - (ee*complex(0,1)*Rd3x3*complexconjugate(NN1x2))/(sw*(-1 + sw**2)*cmath.sqrt(2)) + (ee*complex(0,1)*Rd3x3*sw*complexconjugate(NN1x2))/((-1 + sw**2)*cmath.sqrt(2))',
                 order = {'QED':1})

GC_990 = Coupling(name = 'GC_990',
                  value = '(ee**2*complex(0,1)*Rl4x4*complexconjugate(Rl4x4)*cmath.cos(2*alp))/(2.*(-1 + sw**2))',
                  order = {'QED':2})

GC_991 = Coupling(name = 'GC_991',
                  value = '-(ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)*cmath.cos(2*alp))/(2.*(-1 + sw**2))',
                  order = {'QED':2})

GC_992 = Coupling(name = 'GC_992',
                  value = '(ee**2*complex(0,1)*Rl5x5*complexconjugate(Rl5x5)*cmath.cos(2*alp))/(2.*(-1 + sw**2))',
                  order = {'QED':2})

GC_993 = Coupling(name = 'GC_993',
                  value = '-(ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4)*cmath.cos(2*alp))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_994 = Coupling(name = 'GC_994',
                  value = '(ee**2*complex(0,1)*Ru4x4*complexconjugate(Ru4x4)*cmath.cos(2*alp))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_995 = Coupling(name = 'GC_995',
                  value = '-(ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5)*cmath.cos(2*alp))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_996 = Coupling(name = 'GC_996',
                  value = '(ee**2*complex(0,1)*Ru5x5*complexconjugate(Ru5x5)*cmath.cos(2*alp))/(3.*(-1 + sw**2))',
                  order = {'QED':2})

GC_997 = Coupling(name = 'GC_997',
                  value = '(3*ee**2*complex(0,1)*cmath.cos(2*alp)**2)/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_998 = Coupling(name = 'GC_998',
                  value = '(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*alp))/(6.*(-1 + sw**2)) - (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

GC_999 = Coupling(name = 'GC_999',
                  value = '-(ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*alp))/(6.*(-1 + sw**2)) + (ee**2*complex(0,1)*Rd1x1*complexconjugate(Rd1x1)*cmath.cos(2*alp))/(4.*sw**2*(-1 + sw**2))',
                  order = {'QED':2})

