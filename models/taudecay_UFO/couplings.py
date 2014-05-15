# This file was automatically created by FeynRules 2.0.25
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 8 May 2014 12:30:33


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot



GC_1 = Coupling(name = 'GC_1',
                value = '2*F0*complex(0,1)*Gf*cmath.sqrt(2)',
                order = {'EFT':1,'QED':2})

GC_2 = Coupling(name = 'GC_2',
                value = '-(F1*Gf*cmath.cos(cabi)*cmath.sqrt(2))',
                order = {'EFT':1,'QED':2})

GC_3 = Coupling(name = 'GC_3',
                value = 'F2*Gf*cmath.cos(cabi)*cmath.sqrt(2)',
                order = {'EFT':1,'QED':2})

GC_4 = Coupling(name = 'GC_4',
                value = '-(F3*Fr1*Gf*cmath.cos(cabi)*cmath.sqrt(2))',
                order = {'EFT':1,'QED':2})

GC_5 = Coupling(name = 'GC_5',
                value = 'F3*Fr1*Gf*cmath.cos(cabi)*cmath.sqrt(2)',
                order = {'EFT':1,'QED':2})

GC_6 = Coupling(name = 'GC_6',
                value = '-(F3*Fr2*Gf*cmath.cos(cabi)*cmath.sqrt(2))',
                order = {'EFT':1,'QED':2})

GC_7 = Coupling(name = 'GC_7',
                value = 'F3*Fr2*Gf*cmath.cos(cabi)*cmath.sqrt(2)',
                order = {'EFT':1,'QED':2})

GC_8 = Coupling(name = 'GC_8',
                value = 'F3*Fr1*Gf*Gr1*cmath.cos(cabi)*cmath.sqrt(2) + F3*Fr2*Gf*Gr2*cmath.cos(cabi)*cmath.sqrt(2)',
                order = {'EFT':1,'QED':2})

