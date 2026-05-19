# This file was automatically created by FeynRules $Revision: 999 $
# Mathematica version: 7.0 for Linux x86 (64-bit) (February 18, 2009)
# Date: Mon 30 Jan 2012 19:57:04


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
                value = '(ee*complex(0,1))',
                order = {'QED':1})


GC_5 = Coupling(name = 'GC_5',
                value = 'ee**2*complex(0,1)',
                order = {'QED':2})

GC_6 = Coupling(name = 'GC_6',
                value = '2*ee**2*complex(0,1)',
                order = {'QED':2})

GC_7 = Coupling(name = 'GC_7',
                value = '-ee**2/(2.*cw)',
                order = {'QED':2})

GC_8 = Coupling(name = 'GC_8',
                value = '(ee**2*complex(0,1))/(2.*cw)',
                order = {'QED':2})

GC_9 = Coupling(name = 'GC_9',
                value = 'ee**2/(2.*cw)',
                order = {'QED':2})

GC_10 = Coupling(name = 'GC_10',
                 value = '-G',
                 order = {'QCD':1})

GC_11 = Coupling(name = 'GC_11',
                 value = 'complex(0,1)*G',
                 order = {'QCD':1})


GC_12 = Coupling(name = 'GC_12',
                 value = 'complex(0,1)*G**2',
                 order = {'QCD':2})

GC_30 = Coupling(name = 'GC_30',
                 value = '-yt',
                 order = {'QED':1})


GC_39 = Coupling(name = 'GC_39',
                 value = 'yt',
                 order = {'QED':1})


GC_49 = Coupling(name = 'GC_49',
                 value = '-2*complex(0,1)*lam*ntadpole',
                 order = {'QED':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '-4*complex(0,1)*lam*ntadpole',
                 order = {'QED':2})

GC_51 = Coupling(name = 'GC_51',
                 value = '-6*complex(0,1)*lam*ntadpole',
                 order = {'QED':2})

GC_52 = Coupling(name = 'GC_52',
                 value = '(ee**2*complex(0,1))/(2.*sw**2)',
                 order = {'QED':2})

GC_53 = Coupling(name = 'GC_53',
                 value = '-((ee**2*complex(0,1))/sw**2)',
                 order = {'QED':2})

GC_54 = Coupling(name = 'GC_54',
                 value = '(cw**2*ee**2*complex(0,1))/sw**2',
                 order = {'QED':2})


GC_58 = Coupling(name = 'GC_58',
                 value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                 order = {'QED':1})


GC_68 = Coupling(name = 'GC_68',
                 value = '-(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})

GC_69 = Coupling(name = 'GC_69',
                 value = '(cw*ee*complex(0,1))/(2.*sw)',
                 order = {'QED':1})


GC_70 = Coupling(name = 'GC_70',
                 value = '-((cw*ee*complex(0,1))/sw)',
                 order = {'QED':1})


GC_72 = Coupling(name = 'GC_72',
                 value = '-ee**2/(2.*sw)',
                 order = {'QED':2})

GC_73 = Coupling(name = 'GC_73',
                 value = '-(ee**2*complex(0,1))/(2.*sw)',
                 order = {'QED':2})

GC_74 = Coupling(name = 'GC_74',
                 value = 'ee**2/(2.*sw)',
                 order = {'QED':2})

GC_75 = Coupling(name = 'GC_75',
                 value = '(-2*cw*ee**2*complex(0,1))/sw',
                 order = {'QED':2})

GC_76 = Coupling(name = 'GC_76',
                 value = '-(ee*complex(0,1)*sw)/(6.*cw)',
                 order = {'QED':1})


GC_77 = Coupling(name = 'GC_77',
                 value = '(ee*complex(0,1)*sw)/(2.*cw)',
                 order = {'QED':1})


GC_80 = Coupling(name = 'GC_80',
                 value = '((cw*ee*complex(0,1))/(2.*sw) + (ee*complex(0,1)*sw)/(2.*cw))',
                 order = {'QED':1})


GC_81 = Coupling(name = 'GC_81',
                 value = '((cw*ee**2*complex(0,1))/sw - (ee**2*complex(0,1)*sw)/cw)',
                 order = {'QED':2})

GC_82 = Coupling(name = 'GC_82',
                 value = '(-(ee**2*complex(0,1)) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2))',
                 order = {'QED':2})

GC_83 = Coupling(name = 'GC_83',
                 value = '(ee**2*complex(0,1) + (cw**2*ee**2*complex(0,1))/(2.*sw**2) + (ee**2*complex(0,1)*sw**2)/(2.*cw**2))',
                 order = {'QED':2})


GC_87 = Coupling(name = 'GC_87',
                 value = '-6*complex(0,1)*lam*vev*ntadpole',
                 order = {'QED':1})

GC_88 = Coupling(name = 'GC_88',
                 value = '-(ee**2*vev)/(4.*sw**2)',
                 order = {'QED':1})

GC_89 = Coupling(name = 'GC_89',
                 value = '-(ee**2*complex(0,1)*vev)/(4.*sw**2)',
                 order = {'QED':1})


GC_91 = Coupling(name = 'GC_91',
                 value = '(ee**2*vev)/(4.*sw**2)',
                 order = {'QED':1})


GC_94 = Coupling(name = 'GC_94',
                 value = '(-(ee**2*vev)/(4.*cw) - (cw*ee**2*vev)/(4.*sw**2))',
                 order = {'QED':1})

GC_95 = Coupling(name = 'GC_95',
                 value = '((ee**2*vev)/(4.*cw) - (cw*ee**2*vev)/(4.*sw**2))',
                 order = {'QED':1})

GC_96 = Coupling(name = 'GC_96',
                 value = '(-(ee**2*vev)/(4.*cw) + (cw*ee**2*vev)/(4.*sw**2))',
                 order = {'QED':1})

GC_97 = Coupling(name = 'GC_97',
                 value = '((ee**2*vev)/(4.*cw) + (cw*ee**2*vev)/(4.*sw**2))',
                 order = {'QED':1})

GC_98 = Coupling(name = 'GC_98',
                 value = '(-(ee**2*complex(0,1)*vev)/2. - (cw**2*ee**2*complex(0,1)*vev)/(4.*sw**2) - (ee**2*complex(0,1)*sw**2*vev)/(4.*cw**2))',
                 order = {'QED':1})


GC_116 = Coupling(name = 'GC_116',
                  value = '-((complex(0,1)*yt)/cmath.sqrt(2))',
                  order = {'QED':1})


GC_117 = Coupling(name = 'GC_117',
                  value = 'yt/cmath.sqrt(2)',
                  order = {'QED':1})



 # SMEFT di-boson
     
GC_SM_4 = Coupling(name = 'GC_SM_4',
                #value = 'SM*ee0*complex(0,1)',
                value = 'ee*complex(0,1)',
                order = {'QED':1})

GC_VV_343 = Coupling(name = 'GC_VV_343',
                  #value = '(2*cpWB*cw0*vev0)/Lambda**2',
                  value = '(2*cpWB*cw*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})

GC_SM_372 = Coupling(name = 'GC_SM_372',
                  #value = 'SM*(ee0**2*vev0)/(2.*sw0)',
                  value = '(ee**2*vev)/(2.*sw)',
                  order = {'QED':1})


GC_VV_535 = Coupling(name = 'GC_VV_535',
                  #value = '-(cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0**3) - (cpWB*cw0*ee0**2*vev0**3)/(2.*Lambda**2*sw0**2) - (c3pl1*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (c3pl2*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (cll1221*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0)',
                  value = '-(cpWB*cw*ee**2*vev**3)/(2.*Lambda**2*sw**2)',
                  order =  {'NP':2,'QED':1})                  
                  

GC_VV_462_m = Coupling(name = 'GC_VV_462_m',
                  #value = '(2*cpWB*cw0**2*complex(0,1)*vev0)/Lambda**2 + (4*cpBB*cw0*complex(0,1)*sw0*vev0)/Lambda**2 - (4*cpW*cw0*complex(0,1)*sw0*vev0)/Lambda**2 - (2*cpWB*complex(0,1)*sw0**2*vev0)/Lambda**2',
                  value = '-((2*cpWB*cw**2*complex(0,1)*vev)/Lambda**2 + (4*cpBB*cw*complex(0,1)*sw*vev)/Lambda**2 - (4*cpW*cw*complex(0,1)*sw*vev)/Lambda**2 - (2*cpWB*complex(0,1)*sw**2*vev)/Lambda**2)',
                  order =  {'NP':2,'QED':1})



GC_VV_543_m = Coupling(name = 'GC_VV_543_m',
                  #value = '-(cpDC*cw0*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**3) + (cpDC*cw0**3*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**3) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*cw0*Lambda**2*sw0) + (cpDC*cw0*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0) + (cpDC*ee0**2*complex(0,1)*sw0*vev0**3)/(8.*cw0*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})

GC_qed_set0 = Coupling(name = 'GC_QED_SET0',
                       value = 'gset0',
                       order = {'QED':1})

GC_np_set0 = Coupling(name = 'GC_NP_SET0',
                       value = 'gset0',
                       order = {'QED':1, 'NP':2})



GC_VV_258 = Coupling(name = 'GC_VV_258',
                  #value = '(6*cWWW*complex(0,1)*sw0)/Lambda**2',
                  value = '(6*cWWW*complex(0,1)*sw)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_VV_417 = Coupling(name = 'GC_VV_417',
                  #value = '(cpWB*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  value = '(cpWB*cw*ee*complex(0,1)*vev**2)/(Lambda**2*sw)',
                  order =  {'NP':2,'QED':1})


GC_VV_469 = Coupling(name = 'GC_VV_469',
                  #value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})

GC_VV_339 = Coupling(name = 'GC_VV_339',
                  #value = '(4*cpW*complex(0,1)*vev0)/Lambda**2',
                  value = '(4*cpW*complex(0,1)*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_SM_363 = Coupling(name = 'GC_SM_363',
                  #value = 'SM*(ee0**2*complex(0,1)*vev0)/(2.*sw0**2)',
                  value = '(ee**2*complex(0,1)*vev)/(2.*sw**2)',
                  order = {'QED':1})


GC_VV_528 = Coupling(name = 'GC_VV_528',
                  #value = '-(c3pl1*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (c3pl2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) + (cdp*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2*sw0**2) + (cll1221*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2)',
                  #value = '(cdp*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2*sw0**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})



GC_VV_124_m = Coupling(name = 'GC_VV_124_m',
                  #value = '(-6*cw0*cWWW*complex(0,1))/Lambda**2',
                  value = '-(-6*cw*cWWW*complex(0,1))/Lambda**2',
                  order =  {'NP':2,'QED':1})



GC_SM_198_m = Coupling(name = 'GC_SM_198_m',
                  #value = 'SM*(-((cw0*ee0*complex(0,1))/sw0))',
                  value = '-(-((cw*ee*complex(0,1))/sw))',
                  order = {'QED':1})


GC_VV_412_m = Coupling(name = 'GC_VV_412_m',
                  #value = '-((cpWB*ee0*complex(0,1)*vev0**2)/Lambda**2)',
                  value = '-(-((cpWB*ee*complex(0,1)*vev**2)/Lambda**2))',
                  order =  {'NP':2,'QED':1})


GC_VV_484_m = Coupling(name = 'GC_VV_484_m',
                  #value = '(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  value = '0',
                  order =  {'NP':2,'QED':1})



GC_VV_401_m = Coupling(name = 'GC_VV_401_m',
                  #value = '(-2*cpWB*sw0*vev0)/Lambda**2',
                  value = '-(-2*cpWB*sw*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})



GC_SM_323_m = Coupling(name = 'GC_SM_323_m',
                  #value = 'SM*(-(ee0**2*vev0)/(2.*cw0))',
                  value = '-(-(ee**2*vev)/(2.*cw))',
                  order = {'QED':1})



GC_VV_541_m = Coupling(name = 'GC_VV_541_m',
                  #value = '(c3pl1*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (c3pl2*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (cll1221*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (cpDC*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (cpDC*ee0**2*vev0**3)/(8.*cw0*Lambda**2*sw0**2) - (cpDC*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cpWB*ee0**2*vev0**3)/(2.*Lambda**2*sw0)',
                  value = '-((cpWB*ee**2*vev**3)/(2.*Lambda**2*sw))',
                  order =  {'NP':2,'QED':1})



GC_VV_402_m = Coupling(name = 'GC_VV_402_m',
                  #value = '(2*cpWB*sw0*vev0)/Lambda**2',
                  value = '-(2*cpWB*sw*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_SM_324_m = Coupling(name = 'GC_SM_324_m',
                  #value = 'SM*(ee0**2*vev0)/(2.*cw0)',
                  value = '-(ee**2*vev)/(2.*cw)',
                  order = {'QED':1})



GC_VV_536_m = Coupling(name = 'GC_VV_536_m',
                  #value = '-(c3pl1*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (c3pl2*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (cll1221*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (cpDC*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (cpDC*ee0**2*vev0**3)/(8.*cw0*Lambda**2*sw0**2) + (cpDC*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cpWB*ee0**2*vev0**3)/(2.*Lambda**2*sw0)',
                  value = '-(- (cpWB*ee**2*vev**3)/(2.*Lambda**2*sw))',
                  order =  {'NP':2,'QED':1})


GC_VV_460 = Coupling(name = 'GC_VV_460',
                  #value = '(4*cpW*cw0**2*complex(0,1)*vev0)/Lambda**2 + (4*cpWB*cw0*complex(0,1)*sw0*vev0)/Lambda**2 + (4*cpBB*complex(0,1)*sw0**2*vev0)/Lambda**2',
                  value = '(4*cpW*cw**2*complex(0,1)*vev)/Lambda**2 + (4*cpWB*cw*complex(0,1)*sw*vev)/Lambda**2 + (4*cpBB*complex(0,1)*sw**2*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_SM_459 = Coupling(name = 'GC_SM_459',
                  #value = 'SM*(ee0**2*complex(0,1)*vev0 + (cw0**2*ee0**2*complex(0,1)*vev0)/(2.*sw0**2) + (ee0**2*complex(0,1)*sw0**2*vev0)/(2.*cw0**2))',
                  value = '(ee**2*complex(0,1)*vev + (cw**2*ee**2*complex(0,1)*vev)/(2.*sw**2) + (ee**2*complex(0,1)*sw**2*vev)/(2.*cw**2))',
                  order = {'QED':1})


GC_VV_545 = Coupling(name = 'GC_VV_545',
                  #value = '-(c3pl1*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2) + (cdp*ee0**2*complex(0,1)*vev0**3)/Lambda**2 + (cll1221*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2) + (3*cpDC*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(4.*cw0**2*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (c3pl1*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (c3pl2*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) + (cdp*cw0**2*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2*sw0**2) + (cll1221*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) + (3*cpDC*cw0**2*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) - (c3pl1*ee0**2*complex(0,1)*sw0**2*vev0**3)/(4.*cw0**2*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*sw0**2*vev0**3)/(4.*cw0**2*Lambda**2) + (cdp*ee0**2*complex(0,1)*sw0**2*vev0**3)/(2.*cw0**2*Lambda**2) + (cll1221*ee0**2*complex(0,1)*sw0**2*vev0**3)/(4.*cw0**2*Lambda**2) + (3*cpDC*ee0**2*complex(0,1)*sw0**2*vev0**3)/(8.*cw0**2*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_SM_3 = Coupling(name = 'GC_SM_3',
                #value = 'SM*(-(ee0*complex(0,1)))',
                value = '(-(ee*complex(0,1)))',
                order = {'QED':1})


GC_VV_493 = Coupling(name = 'GC_VV_493',
                  #value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2) + (cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2) + (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2) + (cpWB*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  value = '(cpWB*cw*ee*complex(0,1)*vev**2)/(Lambda**2*sw)',
                  order =  {'NP':2,'QED':1})


GC_SM_194 = Coupling(name = 'GC_SM_194',
                  #value = 'SM*(ee0*complex(0,1))/(2.*sw0)',
                  value = '(ee*complex(0,1))/(2.*sw)',
                  order = {'QED':1})



GC_VV_416 = Coupling(name = 'GC_VV_416',
                  #value = '(3*cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0)',
                  value = '0',
                  order =  {'NP':2,'QED':1})

###
#GC_VV_477 = Coupling(name = 'GC_VV_477',
#                  value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
#                  order =  {'NP':2,'QED':1})
###

GC_SM_193 = Coupling(name = 'GC_SM_193',
                  #value = 'SM*(-(ee0*complex(0,1))/(2.*sw0))',
                  value = '(-(ee*complex(0,1))/(2.*sw))',
                  order = {'QED':1})


GC_VV_415 = Coupling(name = 'GC_VV_415',
                  #value = '(-3*cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0)',
                  value = '0',
                  order =  {'NP':2,'QED':1})

###
#GC_VV_476 = Coupling(name = 'GC_VV_476',
#                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
#                 order =  {'NP':2,'QED':1})
###


GC_SM_192 = Coupling(name = 'GC_SM_192',
                  #value = 'SM*(-ee0/(2.*sw0))',
                  value = '(-ee/(2.*sw))',
                  order =  {'QED':1})



GC_VV_481 = Coupling(name = 'GC_VV_481',
                  #value = '(c3pl1*ee0*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*ee0*vev0**2)/(4.*Lambda**2*sw0) - (cdp*ee0*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*ee0*vev0**2)/(4.*Lambda**2*sw0) + (cpDC*ee0*vev0**2)/(8.*Lambda**2*sw0)',
                  #value = ' - (cdp*ee0*vev0**2)/(2.*Lambda**2*sw0) + (cpDC*ee0*vev0**2)/(8.*Lambda**2*sw0)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_SM_275_m = Coupling(name = 'GC_SM_275_m',
                  #value = 'SM*((cw0*ee0)/(2.*sw0) + (ee0*sw0)/(2.*cw0))',
                  value = '-((cw*ee)/(2.*sw) + (ee*sw)/(2.*cw))',
                  order = {'QED':1})




GC_VV_511_m = Coupling(name = 'GC_VV_511_m',
                  #value = '(cpDC*cw0*ee0*vev0**2)/(2.*Lambda**2*sw0) + (cpDC*ee0*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_VV_509_m = Coupling(name = 'GC_VV_509_m',
                  #value = '-(cpDC*ee0*vev0**2)/(8.*cw0*Lambda**2*sw0) - (c3pl1*cw0*ee0*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*vev0**2)/(4.*Lambda**2*sw0) + (cdp*cw0*ee0*vev0**2)/(2.*Lambda**2*sw0) + (cll1221*cw0*ee0*vev0**2)/(4.*Lambda**2*sw0) - (c3pl1*ee0*sw0*vev0**2)/(4.*cw0*Lambda**2) - (c3pl2*ee0*sw0*vev0**2)/(4.*cw0*Lambda**2) + (cdp*ee0*sw0*vev0**2)/(2.*cw0*Lambda**2) + (cll1221*ee0*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_SM_274_m = Coupling(name = 'GC_SM_274_m',
                  #value = 'SM*((cw0*ee0*complex(0,1))/(2.*sw0) - (ee0*complex(0,1)*sw0)/(2.*cw0))',
                  value = '-((cw*ee*complex(0,1))/(2.*sw) - (ee*complex(0,1)*sw)/(2.*cw))',
                  order = {'QED':1})



GC_VV_510_m = Coupling(name = 'GC_VV_510_m',
                  #value = '(cpWB*ee0*complex(0,1)*vev0**2)/Lambda**2 + (cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) - (c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cpDC*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  value = '-((cpWB*ee*complex(0,1)*vev**2)/Lambda**2)',
                  order =  {'NP':2,'QED':1})

GC_VV_342 = Coupling(name = 'GC_VV_342',
                  #value = '(-2*cpWB*cw0*vev0)/Lambda**2',
                  value = '(-2*cpWB*cw*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_SM_371 = Coupling(name = 'GC_SM_371',
                  #value = 'SM*(-(ee0**2*vev0)/(2.*sw0))',
                  value = '(-(ee**2*vev)/(2.*sw))',
                  order = {'QED':1})

GC_VV_530 = Coupling(name = 'GC_VV_530',
                  #value = '(cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0**3) + (cpWB*cw0*ee0**2*vev0**3)/(2.*Lambda**2*sw0**2) + (c3pl1*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (c3pl2*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (cll1221*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0)',
                  value = '(cpWB*cw*ee**2*vev**3)/(2.*Lambda**2*sw**2)',
                  order =  {'NP':2,'QED':1})



GC_VV_334 = Coupling(name = 'GC_VV_334',
                  #value = '-((cpDC*complex(0,1)*vev0)/Lambda**2)*ntadpole',
                  value = '0',
                  order =  {'NP':2,'QED':1})

GC_VV_328 = Coupling(name = 'GC_VV_328',
                  #value = '(2*cdp*complex(0,1)*vev0)/Lambda**2*ntadpole',
                  value = '0',
                  order =  {'NP':2,'QED':1})



GC_VV_430 = Coupling(name = 'GC_VV_430',
                  #value = '(c3pl1*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (c3pl2*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (cdp*complex(0,1)*MH**2*vev0)/Lambda**2 - (cll1221*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (cpDC*complex(0,1)*MH**2*vev0)/(4.*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_SM_321 = Coupling(name = 'GC_SM_321',
                  #value = 'SM*(-((complex(0,1)*MH**2)/vev0))*ntadpole',
                  value = '(-((complex(0,1)*MH**2)/vev))*ntadpole',
                  order = {'QED':1})



GC_VV_421 = Coupling(name = 'GC_VV_421',
                  #value = '((2*cdp*complex(0,1)*vev0)/Lambda**2 - (cpDC*complex(0,1)*vev0)/(2.*Lambda**2))*ntadpole',
                  value = '0',
                  order =  {'NP':2,'QED':1})



GC_VV_429 = Coupling(name = 'GC_VV_429',
                  #value = '(c3pl1*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (c3pl2*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (cdp*complex(0,1)*MH**2*vev0)/Lambda**2 - (cll1221*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) - (cpDC*complex(0,1)*MH**2*vev0)/(4.*Lambda**2)',
                  #value = '((cdp*complex(0,1)*MH**2*vev0)/Lambda**2 - (cpDC*complex(0,1)*MH**2*vev0)/(4.*Lambda**2))*ntadpole',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_VV_336 = Coupling(name = 'GC_VV_336',
                  #value = '(4*cpG*complex(0,1)*vev0)/Lambda**2',
                  value = '(4*cpG*complex(0,1)*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_VV_471 = Coupling(name = 'GC_VV_471',
                  #value = '(cpDC*ee0*vev0**2)/(8.*Lambda**2) - (cpDC*ee0*vev0**2)/(8.*Lambda**2*sw0**2) + (cpDC*cw0**2*ee0*vev0**2)/(8.*Lambda**2*sw0**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_VV_461 = Coupling(name = 'GC_VV_461',
                  #value = '(4*cpBB*cw0**2*complex(0,1)*vev0)/Lambda**2 - (4*cpWB*cw0*complex(0,1)*sw0*vev0)/Lambda**2 + (4*cpW*complex(0,1)*sw0**2*vev0)/Lambda**2',
                  value = '(4*cpBB*cw**2*complex(0,1)*vev)/Lambda**2 - (4*cpWB*cw*complex(0,1)*sw*vev)/Lambda**2 + (4*cpW*complex(0,1)*sw**2*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_SM_1 = Coupling(name = 'GC_SM_1',
                #value = '-(ee0*complex(0,1))/3.',
                value = '-(ee*complex(0,1))/3.',
                order = {'QED':1})

GC_VV_466 = Coupling(name = 'GC_VV_466',
                  #value = '(cpDC*ee0*complex(0,1)*vev0**2)/(24.*Lambda**2) - (cpDC*ee0*complex(0,1)*vev0**2)/(24.*Lambda**2*sw0**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_VV_1155 = Coupling(name = 'GC_VV_1155',
                   #value = '(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   value = '0',
                   order =  {'NP':2,'QED':1})


GC_VV_489 = Coupling(name = 'GC_VV_489',
                  #value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2) + (c3pl2*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2) - (cll1221*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2) + (cpWB*cw0*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2*sw0)',
                  value = '(cpWB*cw*ee*complex(0,1)*vev**2)/(3.*Lambda**2*sw)',
                  order =  {'NP':2,'QED':1})




GC_SM_197_m = Coupling(name = 'GC_SM_197_m',
                  #value = '(cw0*ee0*complex(0,1))/(2.*sw0)',
                  value = '-((cw*ee*complex(0,1))/(2.*sw))',
                  order = {'QED':1})


GC_VV_482_m = Coupling(name = 'GC_VV_482_m',
                  #value = '(cpWB*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) + (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(12.*Lambda**2*sw0)',
                  value = '-(cpWB*ee*complex(0,1)*vev**2)/(3.*Lambda**2)',
                  order =  {'NP':2,'QED':1})


GC_VV_1800_m = Coupling(name = 'GC_VV_1800_m',
                   #value = '(cpd*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpd*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   value = '-((cpd*cw*ee*complex(0,1)*vev**2)/(2.*Lambda**2*sw) + (cpd*ee*complex(0,1)*sw*vev**2)/(2.*cw*Lambda**2))',
                   order =  {'NP':2,'QED':1})


GC_VV_1802_m = Coupling(name = 'GC_VV_1802_m',
                   #value = '-(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpq3i*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0) + (cpqMi*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpq3i*ee0*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) + (cpqMi*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   value = '-((cpq3i*cw*ee*complex(0,1)*vev**2)/(Lambda**2*sw) + (cpqMi*cw*ee*complex(0,1)*vev**2)/(2.*Lambda**2*sw) + (cpq3i*ee*complex(0,1)*sw*vev**2)/(cw*Lambda**2) + (cpqMi*ee*complex(0,1)*sw*vev**2)/(2.*cw*Lambda**2))',
                   order =  {'NP':2,'QED':1})


GC_VV_505_m = Coupling(name = 'GC_VV_505_m',
                  #value = '-(cpDC*ee0*complex(0,1)*vev0**2)/(24.*cw0*Lambda**2*sw0) - (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2) - (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2) + (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})

GC_VV_1799_m = Coupling(name = 'GC_VV_1799_m',
                   #value = '-(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpQ3*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0) + (cpQM*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpQ3*ee0*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) + (cpQM*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   value = '-((cpQ3*cw*ee*complex(0,1)*vev**2)/(Lambda**2*sw) + (cpQM*cw0*ee*complex(0,1)*vev**2)/(2.*Lambda**2*sw) + (cpQ3*ee*complex(0,1)*sw*vev**2)/(cw*Lambda**2) + (cpQM*ee*complex(0,1)*sw*vev**2)/(2.*cw*Lambda**2))',
                   order =  {'NP':2,'QED':1})



GC_SM_195 = Coupling(name = 'GC_SM_195',
                  #value = '(ee0*complex(0,1))/(sw0*cmath.sqrt(2))',
                  value = '(ee*complex(0,1))/(sw*cmath.sqrt(2))',
                  order = {'QED':1})


GC_VV_1991 = Coupling(name = 'GC_VV_1991',
                   #value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cpq3i*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0*cmath.sqrt(2))',
                   value = '(cpq3i*ee*complex(0,1)*vev**2)/(Lambda**2*sw*cmath.sqrt(2)) ',
                   order =  {'NP':2,'QED':1})

GC_VV_340 = Coupling(name = 'GC_VV_340',
                  #value = '(ctW*complex(0,1)*vev0)/Lambda**2',
                  value = '(ctW*complex(0,1)*vev)/Lambda**2',
                  order =  {'NP':2,'QED':1})


GC_VV_1992 = Coupling(name = 'GC_VV_1992',
                   #value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cpQ3*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0*cmath.sqrt(2))',
                   value = '(cpQ3*ee*complex(0,1)*vev**2)/(Lambda**2*sw*cmath.sqrt(2))',
                   order =  {'NP':2,'QED':1})


GC_SM_2 = Coupling(name = 'GC_SM_2',
                #value = '(2*ee0*complex(0,1))/3.',
                value = '(2*ee*complex(0,1))/3.',
                order = {'QED':1})



GC_VV_466 = Coupling(name = 'GC_VV_466',
                  #value = '(cpDC*ee0*complex(0,1)*vev0**2)/(24.*Lambda**2) - (cpDC*ee0*complex(0,1)*vev0**2)/(24.*Lambda**2*sw0**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})

GC_VV_1154 = Coupling(name = 'GC_VV_1154',
                   #value = '-(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   value = '0',
                   order =  {'NP':2,'QED':1})

GC_VV_490 = Coupling(name = 'GC_VV_490',
                  #value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (c3pl2*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) + (cll1221*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (2*cpWB*cw0*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2*sw0)',
                  value = '-(2*cpWB*cw*ee*complex(0,1)*vev**2)/(3.*Lambda**2*sw)',
                  order =  {'NP':2,'QED':1})



GC_SM_196_m = Coupling(name = 'GC_SM_196_m',
                  #value = '-(cw0*ee0*complex(0,1))/(2.*sw0)',
                  value = '-(-(cw*ee*complex(0,1))/(2.*sw))',
                  order = {'QED':1})



GC_SM_253_m = Coupling(name = 'GC_SM_253_m',
                  #value = '(ee0*complex(0,1)*sw0)/(6.*cw0)',
                  value = '-((ee*complex(0,1)*sw)/(6.*cw))',
                  order = {'QED':1})


GC_VV_483_m = Coupling(name = 'GC_VV_483_m',
                  #value = '(-2*cpWB*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2*sw0)',
                  value = '-((-2*cpWB*ee*complex(0,1)*vev**2)/(3.*Lambda**2))',
                  order =  {'NP':2,'QED':1})



GC_VV_1804_m = Coupling(name = 'GC_VV_1804_m',
                   #value = '(cpu*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpu*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   value = '-((cpu*cw*ee*complex(0,1)*vev**2)/(2.*Lambda**2*sw) + (cpu*ee*complex(0,1)*sw*vev**2)/(2.*cw*Lambda**2))',
                   order =  {'NP':2,'QED':1})


GC_VV_1801_m = Coupling(name = 'GC_VV_1801_m',
                   #value = '(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpqMi*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpqMi*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   value = '-((cpqMi*cw*ee*complex(0,1)*vev**2)/(2.*Lambda**2*sw) + (cpqMi*ee*complex(0,1)*sw*vev**2)/(2.*cw*Lambda**2))',
                   order =  {'NP':2,'QED':1})


GC_VV_505_m = Coupling(name = 'GC_VV_505_m',
                  #value = '-(cpDC*ee0*complex(0,1)*vev0**2)/(24.*cw0*Lambda**2*sw0) - (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2) - (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2) + (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})


GC_SM_548 = Coupling(name = 'GC_SM_548',
                  #value = '-((ymt*cmath.sqrt(2))/vev0)',
                  value = '-((ymt*cmath.sqrt(2))/vev)',
                  order = {'QED':1})


GC_VV_2079 = Coupling(name = 'GC_VV_2079',
                   #value = '-((cpQ3*vev0*cmath.sqrt(2))/Lambda**2)',
                   value = '-((cpQ3*vev*cmath.sqrt(2))/Lambda**2)',
                   order =  {'NP':2,'QED':1})

"""
GC_VV_2084 = Coupling(name = 'GC_VV_2084',
                   value = '(c3pl1*vev0*ymt)/(Lambda**2*cmath.sqrt(2)) + (c3pl2*vev0*ymt)/(Lambda**2*cmath.sqrt(2)) - (cll1221*vev0*ymt)/(Lambda**2*cmath.sqrt(2))',
                   order =  {'NP':2,'QED':1})
"""

GC_SM_549 = Coupling(name = 'GC_SM_549',
                  #value = '(ymt*cmath.sqrt(2))/vev0',
                  value = '(ymt*cmath.sqrt(2))/vev',
                  order = {'QED':1})
"""
GC_VV_2085 = Coupling(name = 'GC_VV_2085',
                   value = '-((c3pl1*vev0*ymt)/(Lambda**2*cmath.sqrt(2))) - (c3pl2*vev0*ymt)/(Lambda**2*cmath.sqrt(2)) + (cll1221*vev0*ymt)/(Lambda**2*cmath.sqrt(2))',
                   order =  {'NP':2,'QED':1})
"""

GC_SM_547 = Coupling(name = 'GC_SM_547',
                  #value = 'ymt/vev0',
                  value = 'ymt/vev',
                  order = {'QED':1})


GC_VV_1834 = Coupling(name = 'GC_VV_1834',
                   #value = '(cpQM*vev0)/Lambda**2',
                   value = '(cpQM*vev)/Lambda**2',
                   order =  {'NP':2,'QED':1})

GC_VV_1840 = Coupling(name = 'GC_VV_1840',
                   #value = '(cpt*vev0)/Lambda**2',
                   value = '(cpt*vev)/Lambda**2',
                   order =  {'NP':2,'QED':1})


GC_VV_550 = Coupling(name = 'GC_VV_550',
                  #value = '-(c3pl1*vev0*ymt)/(2.*Lambda**2) - (c3pl2*vev0*ymt)/(2.*Lambda**2) + (cll1221*vev0*ymt)/(2.*Lambda**2) - (cpDC*vev0*ymt)/(4.*Lambda**2)',
                  value = '0',
                  order =  {'NP':2,'QED':1})



GC_VV_1842 = Coupling(name = 'GC_VV_1842',
                   #value = '(c3pl1*complex(0,1)*vev0*ymt)/(2.*Lambda**2) + (c3pl2*complex(0,1)*vev0*ymt)/(2.*Lambda**2) - (cdp*complex(0,1)*vev0*ymt)/Lambda**2 - (cll1221*complex(0,1)*vev0*ymt)/(2.*Lambda**2) + (cpDC*complex(0,1)*vev0*ymt)/(4.*Lambda**2) + (ctp*complex(0,1)*vev0**2)/(Lambda**2*cmath.sqrt(2))',
                   value = '(ctp*complex(0,1)*vev**2)/(Lambda**2*cmath.sqrt(2))',
                   order =  {'NP':2,'QED':1})


GC_SM_546 = Coupling(name = 'GC_SM_546',
                  #value = '-((complex(0,1)*ymt)/vev0)',
                  value = '-((complex(0,1)*ymt)/vev)',
                  order = {'QED':1})



GC_VV_1150 = Coupling(name = 'GC_VV_1150',
                   #value = '-(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   value = '0',
                   order =  {'NP':2,'QED':1})



GC_VV_443 = Coupling(name = 'GC_VV_443',
                  #value = '-((ctZ*cw0*complex(0,1)*vev0)/(Lambda**2*sw0*cmath.sqrt(2))) + (ctW*cw0**2*complex(0,1)*vev0)/(Lambda**2*sw0*cmath.sqrt(2)) + (ctW*complex(0,1)*sw0*vev0)/(Lambda**2*cmath.sqrt(2))',
                  value = '-((ctZ*cw*complex(0,1)*vev)/(Lambda**2*sw*cmath.sqrt(2))) + (ctW*cw**2*complex(0,1)*vev)/(Lambda**2*sw*cmath.sqrt(2)) + (ctW*complex(0,1)*sw*vev)/(Lambda**2*cmath.sqrt(2))',
                  order =  {'NP':2,'QED':1})

GC_VV_1151 = Coupling(name = 'GC_VV_1151',
                   #value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (c3pl2*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) + (cll1221*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (2*cpWB*cw0*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2*sw0)',
                   value = '-(2*cpWB*cw*ee*complex(0,1)*vev**2)/(3.*Lambda**2*sw)',
                   order =  {'NP':2,'QED':1})



GC_VV_1227_m = Coupling(name = 'GC_VV_1227_m',
                   #value = '(-2*cpWB*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2*sw0)',
                   value = '-((-2*cpWB*ee*complex(0,1)*vev**2)/(3.*Lambda**2))',
                   order =  {'NP':2,'QED':1})


GC_VV_1803_m = Coupling(name = 'GC_VV_1803_m',
                   #value = '(cpt*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpt*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   value = '-((cpt*cw*ee*complex(0,1)*vev**2)/(2.*Lambda**2*sw) + (cpt*ee*complex(0,1)*sw*vev**2)/(2.*cw*Lambda**2))',
                   order =  {'NP':2,'QED':1})


GC_VV_341_m = Coupling(name = 'GC_VV_341_m',
                  #value = '-((ctZ*complex(0,1)*vev0)/(Lambda**2*cmath.sqrt(2)))',
                  value = '-(-((ctZ*complex(0,1)*vev)/(Lambda**2*cmath.sqrt(2))))',
                  order =  {'NP':2,'QED':1})



GC_VV_1797_m = Coupling(name = 'GC_VV_1797_m',
                   #value = '(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpQM*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpQM*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   value = '-((cpQM*ee*complex(0,1)*sw*vev**2)/(2.*cw*Lambda**2) )',
                   order =  {'NP':2,'QED':1})

GC_VV_1152 = Coupling(name = 'GC_VV_1152',
                   #value = '(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   value = '0',
                   order = {'NP':2,'QED':1})
