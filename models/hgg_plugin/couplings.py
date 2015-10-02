# This file was automatically created by FeynRules 1.7.55
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (October 6, 2011)
# Date: Wed 8 Aug 2012 14:16:24


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

GC_1 = Coupling(name = 'GC_1',
                value = '-(AH*complex(0,1))',
                order = {'HIW':1})

GC_13 = Coupling(name = 'GC_13',
                 value = '-(complex(0,1)*GH)',
                 order = {'HIG':1})

GC_14 = Coupling(name = 'GC_14',
                 value = '-(G*GH)',
                 order = {'HIG':1,'QCD':1})

GC_15 = Coupling(name = 'GC_15',
                 value = 'complex(0,1)*G**2*GH',
                 order = {'HIG':1,'QCD':2})

GC_16 = Coupling(name = 'GC_16',
                 value = '(complex(0,1)*Gphi)/8.',
                 order = {'HIG':1})

GC_17 = Coupling(name = 'GC_17',
                 value = '-(G*Gphi)',
                 order = {'HIG':1,'QCD':1})
