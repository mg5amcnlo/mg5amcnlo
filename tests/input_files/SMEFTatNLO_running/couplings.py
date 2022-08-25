# This file was automatically created by FeynRules 2.4.78
# Mathematica version: 12.0.0 for Mac OS X x86 (64-bit) (April 7, 2019)
# Date: Wed 1 Apr 2020 19:35:46

import configuration

from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

if configuration.bottomYukawa:
    
    GC_1842_bot = Coupling(name = 'GC_1842_bot',
                       value = '(c3pl1*complex(0,1)*vev0*ymb)/(2.*Lambda**2) + (c3pl2*complex(0,1)*vev0*ymb)/(2.*Lambda**2) - (cdp*complex(0,1)*vev0*ymb)/Lambda**2 - (cll1221*complex(0,1)*vev0*ymb)/(2.*Lambda**2) + (cpDC*complex(0,1)*vev0*ymb)/(4.*Lambda**2) + (ctp*complex(0,1)*vev0**2)/(Lambda**2*cmath.sqrt(2))',
                       order = {'NP':2,'QED':1})

    GC_546_bot = Coupling(name = 'GC_546_bot',
                      value = '-((complex(0,1)*ymb)/vev0)',
                      order = {'QED':1})

    GC_106_bot = Coupling(name = 'GC_106_bot',
                      value = '(3*cbp*complex(0,1))/(Lambda**2*cmath.sqrt(2))',
                      order = {'NP':2,'QED':3})

    GC_1822_bot = Coupling(name = 'GC_1822_bot',
                       value = '(3*cbp*complex(0,1)*vev0)/(Lambda**2*cmath.sqrt(2))',
                       order = {'NP':2,'QED':2})

GC_1 = Coupling(name = 'GC_1',
                value = '-(ee0*complex(0,1))/3.',
                order = {'QED':1})

GC_10 = Coupling(name = 'GC_10',
                 value = '-G',
                 order = {'QCD':1})

GC_100 = Coupling(name = 'GC_100',
                  value = '(ctlT3*complex(0,1))/(2.*Lambda**2)',
                  order = {'NP':2})

GC_101 = Coupling(name = 'GC_101',
                  value = '(-2*ctp)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_102 = Coupling(name = 'GC_102',
                  value = '-(ctp/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_103 = Coupling(name = 'GC_103',
                  value = 'ctp/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_104 = Coupling(name = 'GC_104',
                  value = '(2*ctp)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_105 = Coupling(name = 'GC_105',
                  value = '(ctp*complex(0,1))/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':3})

GC_106 = Coupling(name = 'GC_106',
                  value = '(3*ctp*complex(0,1))/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':3})

GC_107 = Coupling(name = 'GC_107',
                  value = 'ctp/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':3})

GC_108 = Coupling(name = 'GC_108',
                  value = '(3*ctp)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':3})

GC_109 = Coupling(name = 'GC_109',
                  value = '(ctq1*complex(0,1))/Lambda**2',
                  order = {'NP':2})

GC_11 = Coupling(name = 'GC_11',
                 value = 'complex(0,1)*G',
                 order = {'QCD':1})

GC_110 = Coupling(name = 'GC_110',
                  value = '(ctq8*complex(0,1))/Lambda**2',
                  order = {'NP':2})

GC_111 = Coupling(name = 'GC_111',
                  value = '(2*ctt1*complex(0,1))/Lambda**2',
                  order = {'NP':2})

GC_112 = Coupling(name = 'GC_112',
                  value = '(ctu1*complex(0,1))/Lambda**2',
                  order = {'NP':2})

GC_113 = Coupling(name = 'GC_113',
                  value = '(ctu8*complex(0,1))/Lambda**2',
                  order = {'NP':2})

GC_114 = Coupling(name = 'GC_114',
                  value = '-(ctW/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_115 = Coupling(name = 'GC_115',
                  value = '(ctW*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_1150 = Coupling(name = 'GC_1150',
                   value = '-(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   order = {'NP':2,'QED':1})

GC_1151 = Coupling(name = 'GC_1151',
                   value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (c3pl2*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) + (cll1221*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (2*cpWB*cw0*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2*sw0)',
                   order = {'NP':2,'QED':1})

GC_1152 = Coupling(name = 'GC_1152',
                   value = '(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   order = {'NP':2,'QED':1})

GC_1154 = Coupling(name = 'GC_1154',
                   value = '-(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   order = {'NP':2,'QED':1})

GC_1155 = Coupling(name = 'GC_1155',
                   value = '(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                   order = {'NP':2,'QED':1})

GC_116 = Coupling(name = 'GC_116',
                  value = 'ctW/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_117 = Coupling(name = 'GC_117',
                  value = '-((ctW*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_118 = Coupling(name = 'GC_118',
                  value = '(ctW*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_119 = Coupling(name = 'GC_119',
                  value = '-(ctZ/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QED':2})

GC_12 = Coupling(name = 'GC_12',
                 value = 'complex(0,1)*G**2',
                 order = {'QCD':2})

GC_120 = Coupling(name = 'GC_120',
                  value = '-((ctZ*complex(0,1))/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QED':2})

GC_121 = Coupling(name = 'GC_121',
                  value = '(-2*cpWB*cw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_122 = Coupling(name = 'GC_122',
                  value = '(-2*cpWB*cw0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_1227 = Coupling(name = 'GC_1227',
                   value = '(-2*cpWB*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2*sw0)',
                   order = {'NP':2,'QED':1})

GC_123 = Coupling(name = 'GC_123',
                  value = '(2*cpWB*cw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_124 = Coupling(name = 'GC_124',
                  value = '(-6*cw0*cWWW*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_125 = Coupling(name = 'GC_125',
                  value = '-((c3pl1*ee0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_126 = Coupling(name = 'GC_126',
                  value = '-((c3pl1*ee0*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_127 = Coupling(name = 'GC_127',
                  value = '(c3pl1*ee0*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_128 = Coupling(name = 'GC_128',
                  value = '-((c3pl2*ee0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_129 = Coupling(name = 'GC_129',
                  value = '-((c3pl2*ee0*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_13 = Coupling(name = 'GC_13',
                 value = '(2*cdp*complex(0,1))/Lambda**2 - (cpDC*complex(0,1))/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_130 = Coupling(name = 'GC_130',
                  value = '(c3pl2*ee0*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_131 = Coupling(name = 'GC_131',
                  value = '-((c3pl3*ee0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_132 = Coupling(name = 'GC_132',
                  value = '-((c3pl3*ee0*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_133 = Coupling(name = 'GC_133',
                  value = '(c3pl3*ee0*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_134 = Coupling(name = 'GC_134',
                  value = '(-2*cpDC*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_135 = Coupling(name = 'GC_135',
                  value = '(cpDC*ee0)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_136 = Coupling(name = 'GC_136',
                  value = '(2*cpe*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_137 = Coupling(name = 'GC_137',
                  value = '(2*cpmu*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_138 = Coupling(name = 'GC_138',
                  value = '(2*cpta*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_139 = Coupling(name = 'GC_139',
                  value = '(-4*cpW*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_14 = Coupling(name = 'GC_14',
                 value = '(4*cdp*complex(0,1))/Lambda**2 - (cpDC*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_140 = Coupling(name = 'GC_140',
                  value = '(-2*cpWB*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_141 = Coupling(name = 'GC_141',
                  value = '(2*cpWB*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_142 = Coupling(name = 'GC_142',
                  value = '-((ctW*ee0)/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_143 = Coupling(name = 'GC_143',
                  value = '-((ctW*ee0*complex(0,1))/Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_144 = Coupling(name = 'GC_144',
                  value = '(ctW*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_145 = Coupling(name = 'GC_145',
                  value = '(ctW*ee0*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_146 = Coupling(name = 'GC_146',
                  value = '(-2*cpWB*cw0*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_147 = Coupling(name = 'GC_147',
                  value = '(2*cpWB*cw0*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_148 = Coupling(name = 'GC_148',
                  value = '(2*cpWB*cw0*ee0)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_149 = Coupling(name = 'GC_149',
                  value = '(6*cw0*cWWW*ee0*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_15 = Coupling(name = 'GC_15',
                 value = '-((c3pl1*complex(0,1))/Lambda**2) - (cpl1*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_150 = Coupling(name = 'GC_150',
                  value = '(8*cpDC*ee0**2*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':4})

GC_151 = Coupling(name = 'GC_151',
                  value = '(-4*cpW*ee0**2*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':4})

GC_152 = Coupling(name = 'GC_152',
                  value = '(6*cw0*cWWW*ee0**2*complex(0,1))/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_153 = Coupling(name = 'GC_153',
                  value = '(4*cpG*G)/Lambda**2',
                  order = {'NP':2,'QCD':1,'QED':2})

GC_154 = Coupling(name = 'GC_154',
                  value = '-((ctG*G)/Lambda**2)',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_155 = Coupling(name = 'GC_155',
                  value = '(ctG*G)/Lambda**2',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_156 = Coupling(name = 'GC_156',
                  value = '(ctG*complex(0,1)*G)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_157 = Coupling(name = 'GC_157',
                  value = '(ctG*G)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_158 = Coupling(name = 'GC_158',
                  value = '(-4*cpG*complex(0,1)*G**2)/Lambda**2',
                  order = {'NP':2,'QCD':2,'QED':2})

GC_159 = Coupling(name = 'GC_159',
                  value = '-((ctG*complex(0,1)*G**2)/Lambda**2)',
                  order = {'NP':2,'QCD':2,'QED':1})

GC_16 = Coupling(name = 'GC_16',
                 value = '(c3pl1*complex(0,1))/Lambda**2 - (cpl1*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_160 = Coupling(name = 'GC_160',
                  value = '(ctG*complex(0,1)*G**2)/Lambda**2',
                  order = {'NP':2,'QCD':2,'QED':1})

GC_161 = Coupling(name = 'GC_161',
                  value = '-((ctG*G**2)/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QCD':2,'QED':1})

GC_162 = Coupling(name = 'GC_162',
                  value = '-((ctG*complex(0,1)*G**2)/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QCD':2,'QED':1})

GC_163 = Coupling(name = 'GC_163',
                  value = '(c3pl1*complex(0,1)*MH**2)/Lambda**2 + (c3pl2*complex(0,1)*MH**2)/Lambda**2 - (cll1221*complex(0,1)*MH**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_164 = Coupling(name = 'GC_164',
                  value = '(c3pl1*complex(0,1)*MH**2)/Lambda**2 + (c3pl2*complex(0,1)*MH**2)/Lambda**2 + (2*cdp*complex(0,1)*MH**2)/Lambda**2 - (cll1221*complex(0,1)*MH**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_165 = Coupling(name = 'GC_165',
                  value = '(c3pl1*complex(0,1)*MH**2)/Lambda**2 + (c3pl2*complex(0,1)*MH**2)/Lambda**2 - (cll1221*complex(0,1)*MH**2)/Lambda**2 + (cpDC*complex(0,1)*MH**2)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_166 = Coupling(name = 'GC_166',
                  value = '(2*c3pl1*complex(0,1)*MH**2)/Lambda**2 + (2*c3pl2*complex(0,1)*MH**2)/Lambda**2 + (4*cdp*complex(0,1)*MH**2)/Lambda**2 - (2*cll1221*complex(0,1)*MH**2)/Lambda**2 - (cpDC*complex(0,1)*MH**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_167 = Coupling(name = 'GC_167',
                  value = '(3*c3pl1*complex(0,1)*MH**2)/Lambda**2 + (3*c3pl2*complex(0,1)*MH**2)/Lambda**2 - (6*cdp*complex(0,1)*MH**2)/Lambda**2 - (3*cll1221*complex(0,1)*MH**2)/Lambda**2 + (3*cpDC*complex(0,1)*MH**2)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_168 = Coupling(name = 'GC_168',
                  value = '(3*c3pl1*complex(0,1)*MH**2)/Lambda**2 + (3*c3pl2*complex(0,1)*MH**2)/Lambda**2 + (6*cdp*complex(0,1)*MH**2)/Lambda**2 - (3*cll1221*complex(0,1)*MH**2)/Lambda**2 + (3*cpDC*complex(0,1)*MH**2)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_169 = Coupling(name = 'GC_169',
                  value = '(-3*cpDC*ee0**2)/(2.*cw0*Lambda**2) - (3*cpDC*cw0*ee0**2)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_17 = Coupling(name = 'GC_17',
                 value = '-(c3pl1/Lambda**2) + cpl1/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_170 = Coupling(name = 'GC_170',
                  value = '(cpDC*ee0**2)/(cw0*Lambda**2) - (cpDC*cw0*ee0**2)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_171 = Coupling(name = 'GC_171',
                  value = '-(cpDC*ee0**2)/(2.*cw0*Lambda**2) - (cpDC*cw0*ee0**2)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_172 = Coupling(name = 'GC_172',
                  value = '-(cpDC*ee0**2*complex(0,1))/(2.*cw0*Lambda**2) - (cpDC*cw0*ee0**2*complex(0,1))/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_173 = Coupling(name = 'GC_173',
                  value = '-((cpDC*ee0**2*complex(0,1))/(cw0*Lambda**2)) + (cpDC*cw0*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_174 = Coupling(name = 'GC_174',
                  value = '(-3*cpDC*ee0**2*complex(0,1))/(2.*cw0*Lambda**2) - (3*cpDC*cw0*ee0**2*complex(0,1))/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_175 = Coupling(name = 'GC_175',
                  value = '(cpDC*ee0**2)/(2.*cw0*Lambda**2) + (cpDC*cw0*ee0**2)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_176 = Coupling(name = 'GC_176',
                  value = '-((cpDC*ee0**2)/(cw0*Lambda**2)) + (cpDC*cw0*ee0**2)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_177 = Coupling(name = 'GC_177',
                  value = '(3*cpDC*ee0**2)/(2.*cw0*Lambda**2) + (3*cpDC*cw0*ee0**2)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_178 = Coupling(name = 'GC_178',
                  value = '(24*cw0**2*cWWW*ee0**3*complex(0,1))/(Lambda**2*sw0**3)',
                  order = {'NP':2,'QED':4})

GC_179 = Coupling(name = 'GC_179',
                  value = '(ee0**2*complex(0,1))/(2.*sw0**2)',
                  order = {'QED':2})

GC_1797 = Coupling(name = 'GC_1797',
                   value = '(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpQM*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpQM*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_1799 = Coupling(name = 'GC_1799',
                   value = '-(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpQ3*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0) + (cpQM*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpQ3*ee0*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) + (cpQM*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_18 = Coupling(name = 'GC_18',
                 value = 'c3pl1/Lambda**2 + cpl1/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_180 = Coupling(name = 'GC_180',
                  value = '-((ee0**2*complex(0,1))/sw0**2)',
                  order = {'QED':2})

GC_1800 = Coupling(name = 'GC_1800',
                   value = '(cpd*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpd*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_1801 = Coupling(name = 'GC_1801',
                   value = '(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpqMi*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpqMi*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_1802 = Coupling(name = 'GC_1802',
                   value = '-(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpq3i*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0) + (cpqMi*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpq3i*ee0*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) + (cpqMi*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_1803 = Coupling(name = 'GC_1803',
                   value = '(cpt*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpt*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_1804 = Coupling(name = 'GC_1804',
                   value = '(cpu*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpu*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_181 = Coupling(name = 'GC_181',
                  value = '(cw0**2*ee0**2*complex(0,1))/sw0**2',
                  order = {'QED':2})

GC_1811 = Coupling(name = 'GC_1811',
                   value = 'cpQM/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1813 = Coupling(name = 'GC_1813',
                   value = '(2*cpQ3)/Lambda**2 + cpQM/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1814 = Coupling(name = 'GC_1814',
                   value = 'cpd/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1815 = Coupling(name = 'GC_1815',
                   value = 'cpqMi/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1816 = Coupling(name = 'GC_1816',
                   value = '(2*cpq3i)/Lambda**2 + cpqMi/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1817 = Coupling(name = 'GC_1817',
                   value = 'cpt/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1818 = Coupling(name = 'GC_1818',
                   value = 'cpu/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1819 = Coupling(name = 'GC_1819',
                   value = '-((ctp*vev0)/(Lambda**2*cmath.sqrt(2)))',
                   order = {'NP':2,'QED':2})

GC_182 = Coupling(name = 'GC_182',
                  value = '(-2*cpDC*ee0**2)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_1821 = Coupling(name = 'GC_1821',
                   value = '(ctp*complex(0,1)*vev0)/(Lambda**2*cmath.sqrt(2))',
                   order = {'NP':2,'QED':2})

GC_1822 = Coupling(name = 'GC_1822',
                   value = '(3*ctp*complex(0,1)*vev0)/(Lambda**2*cmath.sqrt(2))',
                   order = {'NP':2,'QED':2})

GC_1823 = Coupling(name = 'GC_1823',
                   value = '(cpQM*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpQM*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_1825 = Coupling(name = 'GC_1825',
                   value = '(2*cpQ3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpQM*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (2*cpQ3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpQM*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_1826 = Coupling(name = 'GC_1826',
                   value = '(cpd*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpd*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_1827 = Coupling(name = 'GC_1827',
                   value = '(cpqMi*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpqMi*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_1828 = Coupling(name = 'GC_1828',
                   value = '(2*cpq3i*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpqMi*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (2*cpq3i*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpqMi*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_1829 = Coupling(name = 'GC_1829',
                   value = '(cpt*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpt*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_183 = Coupling(name = 'GC_183',
                  value = '(cpDC*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_1830 = Coupling(name = 'GC_1830',
                   value = '(cpu*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpu*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_1834 = Coupling(name = 'GC_1834',
                   value = '(cpQM*vev0)/Lambda**2',
                   order = {'NP':2,'QED':1})

GC_1836 = Coupling(name = 'GC_1836',
                   value = '(2*cpQ3*vev0)/Lambda**2 + (cpQM*vev0)/Lambda**2',
                   order = {'NP':2,'QED':1})

GC_1837 = Coupling(name = 'GC_1837',
                   value = '(cpd*vev0)/Lambda**2',
                   order = {'NP':2,'QED':1})

GC_1838 = Coupling(name = 'GC_1838',
                   value = '(cpqMi*vev0)/Lambda**2',
                   order = {'NP':2,'QED':1})

GC_1839 = Coupling(name = 'GC_1839',
                   value = '(2*cpq3i*vev0)/Lambda**2 + (cpqMi*vev0)/Lambda**2',
                   order = {'NP':2,'QED':1})

GC_184 = Coupling(name = 'GC_184',
                  value = '(-2*cpDC*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_1840 = Coupling(name = 'GC_1840',
                   value = '(cpt*vev0)/Lambda**2',
                   order = {'NP':2,'QED':1})

GC_1841 = Coupling(name = 'GC_1841',
                   value = '(cpu*vev0)/Lambda**2',
                   order = {'NP':2,'QED':1})

GC_1842 = Coupling(name = 'GC_1842',
                   value = '(c3pl1*complex(0,1)*vev0*ymt)/(2.*Lambda**2) + (c3pl2*complex(0,1)*vev0*ymt)/(2.*Lambda**2) - (cdp*complex(0,1)*vev0*ymt)/Lambda**2 - (cll1221*complex(0,1)*vev0*ymt)/(2.*Lambda**2) + (cpDC*complex(0,1)*vev0*ymt)/(4.*Lambda**2) + (ctp*complex(0,1)*vev0**2)/(Lambda**2*cmath.sqrt(2))',
                   order = {'NP':2,'QED':1})

GC_1845 = Coupling(name = 'GC_1845',
                   value = '(cpQM*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpQM*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1847 = Coupling(name = 'GC_1847',
                   value = '(2*cpQ3*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpQM*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (2*cpQ3*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpQM*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1848 = Coupling(name = 'GC_1848',
                   value = '(cpd*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpd*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1849 = Coupling(name = 'GC_1849',
                   value = '(cpqMi*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpqMi*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_185 = Coupling(name = 'GC_185',
                  value = '(2*cpDC*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_1850 = Coupling(name = 'GC_1850',
                   value = '(2*cpq3i*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpqMi*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (2*cpq3i*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpqMi*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1851 = Coupling(name = 'GC_1851',
                   value = '(cpt*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpt*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1852 = Coupling(name = 'GC_1852',
                   value = '(cpu*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpu*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_186 = Coupling(name = 'GC_186',
                  value = '(2*cpDC*ee0**2)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_187 = Coupling(name = 'GC_187',
                  value = '(4*cpW*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_188 = Coupling(name = 'GC_188',
                  value = '(-4*cpW*cw0**2*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_189 = Coupling(name = 'GC_189',
                  value = '(6*cw0*cWWW*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_19 = Coupling(name = 'GC_19',
                 value = '-((c3pl2*complex(0,1))/Lambda**2) - (cpl2*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_190 = Coupling(name = 'GC_190',
                  value = '(6*cw0**3*cWWW*ee0**2*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_191 = Coupling(name = 'GC_191',
                  value = '(-24*cw0*cWWW*ee0**3*complex(0,1))/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':4})

GC_192 = Coupling(name = 'GC_192',
                  value = '-ee0/(2.*sw0)',
                  order = {'QED':1})

GC_193 = Coupling(name = 'GC_193',
                  value = '-(ee0*complex(0,1))/(2.*sw0)',
                  order = {'QED':1})

GC_194 = Coupling(name = 'GC_194',
                  value = '(ee0*complex(0,1))/(2.*sw0)',
                  order = {'QED':1})

GC_195 = Coupling(name = 'GC_195',
                  value = '(ee0*complex(0,1))/(sw0*cmath.sqrt(2))',
                  order = {'QED':1})

GC_196 = Coupling(name = 'GC_196',
                  value = '-(cw0*ee0*complex(0,1))/(2.*sw0)',
                  order = {'QED':1})

GC_197 = Coupling(name = 'GC_197',
                  value = '(cw0*ee0*complex(0,1))/(2.*sw0)',
                  order = {'QED':1})

GC_198 = Coupling(name = 'GC_198',
                  value = '-((cw0*ee0*complex(0,1))/sw0)',
                  order = {'QED':1})

GC_199 = Coupling(name = 'GC_199',
                  value = '(cw0*ee0*complex(0,1))/sw0',
                  order = {'QED':1})

GC_1991 = Coupling(name = 'GC_1991',
                   value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cpq3i*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0*cmath.sqrt(2))',
                   order = {'NP':2,'QED':1})

GC_1992 = Coupling(name = 'GC_1992',
                   value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cpQ3*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0*cmath.sqrt(2))',
                   order = {'NP':2,'QED':1})

GC_1993 = Coupling(name = 'GC_1993',
                   value = '(-2*cpQ3*complex(0,1))/Lambda**2 - (cpQM*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1994 = Coupling(name = 'GC_1994',
                   value = '-((cpq3i*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1995 = Coupling(name = 'GC_1995',
                   value = '-((cpq3i*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1996 = Coupling(name = 'GC_1996',
                   value = '(cpq3i*complex(0,1)*cmath.sqrt(2))/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_1997 = Coupling(name = 'GC_1997',
                   value = '-((cpQ3*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1998 = Coupling(name = 'GC_1998',
                   value = '-((cpQ3*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_1999 = Coupling(name = 'GC_1999',
                   value = '(cpQ3*complex(0,1)*cmath.sqrt(2))/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_2 = Coupling(name = 'GC_2',
                value = '(2*ee0*complex(0,1))/3.',
                order = {'QED':1})

GC_20 = Coupling(name = 'GC_20',
                 value = '(c3pl2*complex(0,1))/Lambda**2 - (cpl2*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_200 = Coupling(name = 'GC_200',
                  value = '-ee0**2/(2.*sw0)',
                  order = {'QED':2})

GC_2000 = Coupling(name = 'GC_2000',
                   value = '-((cpQM*complex(0,1))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2002 = Coupling(name = 'GC_2002',
                   value = '-((cpd*complex(0,1))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2003 = Coupling(name = 'GC_2003',
                   value = '(-2*cpq3i*complex(0,1))/Lambda**2 - (cpqMi*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_2004 = Coupling(name = 'GC_2004',
                   value = '-((cpqMi*complex(0,1))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2005 = Coupling(name = 'GC_2005',
                   value = '-((cpt*complex(0,1))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2006 = Coupling(name = 'GC_2006',
                   value = '-((cpu*complex(0,1))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2007 = Coupling(name = 'GC_2007',
                   value = '(4*cpQ3*ee0*complex(0,1))/Lambda**2 + (2*cpQM*ee0*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2008 = Coupling(name = 'GC_2008',
                   value = '-((cpq3i*ee0*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2009 = Coupling(name = 'GC_2009',
                   value = '-((cpq3i*ee0*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_201 = Coupling(name = 'GC_201',
                  value = '-(ee0**2*complex(0,1))/(2.*sw0)',
                  order = {'QED':2})

GC_2010 = Coupling(name = 'GC_2010',
                   value = '(cpq3i*ee0*cmath.sqrt(2))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2011 = Coupling(name = 'GC_2011',
                   value = '-((cpQ3*ee0*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2012 = Coupling(name = 'GC_2012',
                   value = '-((cpQ3*ee0*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2013 = Coupling(name = 'GC_2013',
                   value = '(cpQ3*ee0*cmath.sqrt(2))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2015 = Coupling(name = 'GC_2015',
                   value = '(2*cpQM*ee0*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2016 = Coupling(name = 'GC_2016',
                   value = '(2*cpd*ee0*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2017 = Coupling(name = 'GC_2017',
                   value = '(2*cpqMi*ee0*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2018 = Coupling(name = 'GC_2018',
                   value = '(4*cpq3i*ee0*complex(0,1))/Lambda**2 + (2*cpqMi*ee0*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2019 = Coupling(name = 'GC_2019',
                   value = '(2*cpt*ee0*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_202 = Coupling(name = 'GC_202',
                  value = 'ee0**2/(2.*sw0)',
                  order = {'QED':2})

GC_2020 = Coupling(name = 'GC_2020',
                   value = '(2*cpu*ee0*complex(0,1))/Lambda**2',
                   order = {'NP':2,'QED':3})

GC_2021 = Coupling(name = 'GC_2021',
                   value = '(ctp*complex(0,1)*vev0)/(Lambda**2*cmath.sqrt(2))',
                   order = {'NP':2,'QED':2})

GC_2023 = Coupling(name = 'GC_2023',
                   value = '(ctp*vev0)/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_2024 = Coupling(name = 'GC_2024',
                   value = '-((ctp*vev0)/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2025 = Coupling(name = 'GC_2025',
                   value = '(cpq3i*ee0*complex(0,1)*cmath.sqrt(2))/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2026 = Coupling(name = 'GC_2026',
                   value = '(cpQ3*ee0)/(Lambda**2*sw0) + (cpQM*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2027 = Coupling(name = 'GC_2027',
                   value = '-((cpQ3*ee0*complex(0,1))/(Lambda**2*sw0)) - (cpQM*ee0*complex(0,1))/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2028 = Coupling(name = 'GC_2028',
                   value = '-((cpQ3*ee0)/(Lambda**2*sw0)) - (cpQM*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2029 = Coupling(name = 'GC_2029',
                   value = '(cpQ3*ee0*complex(0,1)*cmath.sqrt(2))/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_203 = Coupling(name = 'GC_203',
                  value = '(2*cw0*ee0**2*complex(0,1))/sw0',
                  order = {'QED':2})

GC_2031 = Coupling(name = 'GC_2031',
                   value = '-((cpQ3*ee0)/(Lambda**2*sw0)) - (cpQM*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2032 = Coupling(name = 'GC_2032',
                   value = '-((cpQ3*ee0*complex(0,1))/(Lambda**2*sw0)) - (cpQM*ee0*complex(0,1))/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2035 = Coupling(name = 'GC_2035',
                   value = '(cpQ3*ee0)/(Lambda**2*sw0) + (cpQM*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2036 = Coupling(name = 'GC_2036',
                   value = '-((cpd*ee0)/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':3})

GC_2037 = Coupling(name = 'GC_2037',
                   value = '-((cpd*ee0*complex(0,1))/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':3})

GC_2038 = Coupling(name = 'GC_2038',
                   value = '(cpd*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2039 = Coupling(name = 'GC_2039',
                   value = '-((cpq3i*ee0)/(Lambda**2*sw0)) - (cpqMi*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_204 = Coupling(name = 'GC_204',
                  value = '(c3pl1*ee0*complex(0,1)*cmath.sqrt(2))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_2040 = Coupling(name = 'GC_2040',
                   value = '-((cpq3i*ee0*complex(0,1))/(Lambda**2*sw0)) - (cpqMi*ee0*complex(0,1))/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2041 = Coupling(name = 'GC_2041',
                   value = '(cpq3i*ee0)/(Lambda**2*sw0) + (cpqMi*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2042 = Coupling(name = 'GC_2042',
                   value = '-((cpt*ee0)/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':3})

GC_2043 = Coupling(name = 'GC_2043',
                   value = '-((cpt*ee0*complex(0,1))/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':3})

GC_2044 = Coupling(name = 'GC_2044',
                   value = '(cpt*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2045 = Coupling(name = 'GC_2045',
                   value = '-((cpu*ee0)/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':3})

GC_2046 = Coupling(name = 'GC_2046',
                   value = '-((cpu*ee0*complex(0,1))/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':3})

GC_2047 = Coupling(name = 'GC_2047',
                   value = '(cpu*ee0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':3})

GC_2049 = Coupling(name = 'GC_2049',
                   value = '(-2*cpQ3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (cpQM*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (2*cpQ3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpQM*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_205 = Coupling(name = 'GC_205',
                  value = '(c3pl2*ee0*complex(0,1)*cmath.sqrt(2))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_2050 = Coupling(name = 'GC_2050',
                   value = '-((cpq3i*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                   order = {'NP':2,'QED':3})

GC_2051 = Coupling(name = 'GC_2051',
                   value = '-((cpq3i*ee0*complex(0,1)*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                   order = {'NP':2,'QED':3})

GC_2052 = Coupling(name = 'GC_2052',
                   value = '(cpq3i*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2053 = Coupling(name = 'GC_2053',
                   value = '-((cpQ3*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                   order = {'NP':2,'QED':3})

GC_2054 = Coupling(name = 'GC_2054',
                   value = '-((cpQ3*ee0*complex(0,1)*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                   order = {'NP':2,'QED':3})

GC_2055 = Coupling(name = 'GC_2055',
                   value = '(cpQ3*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2057 = Coupling(name = 'GC_2057',
                   value = '-((cpQM*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpQM*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2058 = Coupling(name = 'GC_2058',
                   value = '-((cpd*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpd*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2059 = Coupling(name = 'GC_2059',
                   value = '-((cpqMi*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpqMi*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_206 = Coupling(name = 'GC_206',
                  value = '(c3pl3*ee0*complex(0,1)*cmath.sqrt(2))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_2060 = Coupling(name = 'GC_2060',
                   value = '(-2*cpq3i*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (cpqMi*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (2*cpq3i*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpqMi*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2061 = Coupling(name = 'GC_2061',
                   value = '-((cpt*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpt*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_2062 = Coupling(name = 'GC_2062',
                   value = '-((cpu*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpu*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':3})

GC_207 = Coupling(name = 'GC_207',
                  value = '-(cpDC*ee0)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_2078 = Coupling(name = 'GC_2078',
                   value = '-((cpq3i*vev0*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_2079 = Coupling(name = 'GC_2079',
                   value = '-((cpQ3*vev0*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':1})

GC_208 = Coupling(name = 'GC_208',
                  value = '(cpDC*ee0*complex(0,1))/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_2080 = Coupling(name = 'GC_2080',
                   value = '-((cpq3i*ee0*vev0*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2081 = Coupling(name = 'GC_2081',
                   value = '(cpq3i*ee0*vev0*cmath.sqrt(2))/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_2082 = Coupling(name = 'GC_2082',
                   value = '-((cpQ3*ee0*vev0*cmath.sqrt(2))/Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2083 = Coupling(name = 'GC_2083',
                   value = '(cpQ3*ee0*vev0*cmath.sqrt(2))/Lambda**2',
                   order = {'NP':2,'QED':2})

GC_2084 = Coupling(name = 'GC_2084',
                   value = '(c3pl1*vev0*ymt)/(Lambda**2*cmath.sqrt(2)) + (c3pl2*vev0*ymt)/(Lambda**2*cmath.sqrt(2)) - (cll1221*vev0*ymt)/(Lambda**2*cmath.sqrt(2))',
                   order = {'NP':2,'QED':1})

GC_2085 = Coupling(name = 'GC_2085',
                   value = '-((c3pl1*vev0*ymt)/(Lambda**2*cmath.sqrt(2))) - (c3pl2*vev0*ymt)/(Lambda**2*cmath.sqrt(2)) + (cll1221*vev0*ymt)/(Lambda**2*cmath.sqrt(2))',
                   order = {'NP':2,'QED':1})

GC_209 = Coupling(name = 'GC_209',
                  value = '-((cpDC*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_2099 = Coupling(name = 'GC_2099',
                   value = '(cpq3i*ee0*complex(0,1)*vev0*cmath.sqrt(2))/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_21 = Coupling(name = 'GC_21',
                 value = '-(c3pl2/Lambda**2) + cpl2/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_210 = Coupling(name = 'GC_210',
                  value = '(cpDC*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_2100 = Coupling(name = 'GC_2100',
                   value = '(cpQ3*ee0*vev0)/(Lambda**2*sw0) + (cpQM*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2101 = Coupling(name = 'GC_2101',
                   value = '-((cpQ3*ee0*vev0)/(Lambda**2*sw0)) - (cpQM*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2102 = Coupling(name = 'GC_2102',
                   value = '(cpQ3*ee0*complex(0,1)*vev0*cmath.sqrt(2))/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2104 = Coupling(name = 'GC_2104',
                   value = '-((cpQ3*ee0*vev0)/(Lambda**2*sw0)) - (cpQM*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2106 = Coupling(name = 'GC_2106',
                   value = '(cpQ3*ee0*vev0)/(Lambda**2*sw0) + (cpQM*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2107 = Coupling(name = 'GC_2107',
                   value = '-((cpd*ee0*vev0)/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':2})

GC_2108 = Coupling(name = 'GC_2108',
                   value = '(cpd*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2109 = Coupling(name = 'GC_2109',
                   value = '-((cpq3i*ee0*vev0)/(Lambda**2*sw0)) - (cpqMi*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_211 = Coupling(name = 'GC_211',
                  value = '(cpDC*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_2110 = Coupling(name = 'GC_2110',
                   value = '(cpq3i*ee0*vev0)/(Lambda**2*sw0) + (cpqMi*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2111 = Coupling(name = 'GC_2111',
                   value = '-((cpt*ee0*vev0)/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':2})

GC_2112 = Coupling(name = 'GC_2112',
                   value = '(cpt*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_2113 = Coupling(name = 'GC_2113',
                   value = '-((cpu*ee0*vev0)/(Lambda**2*sw0))',
                   order = {'NP':2,'QED':2})

GC_2114 = Coupling(name = 'GC_2114',
                   value = '(cpu*ee0*vev0)/(Lambda**2*sw0)',
                   order = {'NP':2,'QED':2})

GC_212 = Coupling(name = 'GC_212',
                  value = '-((cpe*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_2121 = Coupling(name = 'GC_2121',
                   value = '-((cpq3i*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2))',
                   order = {'NP':2,'QED':2})

GC_2122 = Coupling(name = 'GC_2122',
                   value = '(cpq3i*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_2123 = Coupling(name = 'GC_2123',
                   value = '-((cpQ3*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2))',
                   order = {'NP':2,'QED':2})

GC_2124 = Coupling(name = 'GC_2124',
                   value = '(cpQ3*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2)',
                   order = {'NP':2,'QED':2})

GC_213 = Coupling(name = 'GC_213',
                  value = '-((cpe*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_214 = Coupling(name = 'GC_214',
                  value = '(cpe*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_215 = Coupling(name = 'GC_215',
                  value = '-((cpl1*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_216 = Coupling(name = 'GC_216',
                  value = '-((cpl1*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_217 = Coupling(name = 'GC_217',
                  value = '(cpl1*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_218 = Coupling(name = 'GC_218',
                  value = '-((cpl2*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_219 = Coupling(name = 'GC_219',
                  value = '-((cpl2*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_22 = Coupling(name = 'GC_22',
                 value = 'c3pl2/Lambda**2 + cpl2/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_220 = Coupling(name = 'GC_220',
                  value = '(cpl2*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_221 = Coupling(name = 'GC_221',
                  value = '-((cpl3*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_222 = Coupling(name = 'GC_222',
                  value = '-((cpl3*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_223 = Coupling(name = 'GC_223',
                  value = '(cpl3*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_224 = Coupling(name = 'GC_224',
                  value = '-((cpmu*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_225 = Coupling(name = 'GC_225',
                  value = '-((cpmu*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_226 = Coupling(name = 'GC_226',
                  value = '(cpmu*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_227 = Coupling(name = 'GC_227',
                  value = '-((cpta*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_228 = Coupling(name = 'GC_228',
                  value = '-((cpta*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_229 = Coupling(name = 'GC_229',
                  value = '(cpta*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_23 = Coupling(name = 'GC_23',
                 value = '-((c3pl3*complex(0,1))/Lambda**2) - (cpl3*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_230 = Coupling(name = 'GC_230',
                  value = '-((ctW*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_231 = Coupling(name = 'GC_231',
                  value = '(ctW*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_232 = Coupling(name = 'GC_232',
                  value = '(ctW*ee0*complex(0,1))/(Lambda**2*sw0*cmath.sqrt(2))',
                  order = {'NP':2,'QED':3})

GC_233 = Coupling(name = 'GC_233',
                  value = '(ctW*ee0)/(Lambda**2*sw0*cmath.sqrt(2))',
                  order = {'NP':2,'QED':3})

GC_234 = Coupling(name = 'GC_234',
                  value = '(4*cpW*cw0*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_235 = Coupling(name = 'GC_235',
                  value = '(-2*cpWB*cw0*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_236 = Coupling(name = 'GC_236',
                  value = '(2*cpWB*cw0*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_237 = Coupling(name = 'GC_237',
                  value = '-((ctW*cw0*ee0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_238 = Coupling(name = 'GC_238',
                  value = '-((ctW*cw0*ee0*complex(0,1))/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':3})

GC_239 = Coupling(name = 'GC_239',
                  value = '(ctW*cw0*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_24 = Coupling(name = 'GC_24',
                 value = '(c3pl3*complex(0,1))/Lambda**2 - (cpl3*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_240 = Coupling(name = 'GC_240',
                  value = '(ctW*cw0*ee0*cmath.sqrt(2))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_241 = Coupling(name = 'GC_241',
                  value = '(-2*cpWB*cw0**2*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_242 = Coupling(name = 'GC_242',
                  value = '(2*cpWB*cw0**2*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_243 = Coupling(name = 'GC_243',
                  value = '(2*cpWB*cw0**2*ee0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_244 = Coupling(name = 'GC_244',
                  value = '(-6*cWWW*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_245 = Coupling(name = 'GC_245',
                  value = '(6*cw0**2*cWWW*ee0*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_246 = Coupling(name = 'GC_246',
                  value = '(-2*cpDC*ee0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':4})

GC_247 = Coupling(name = 'GC_247',
                  value = '(-2*cpDC*ee0**2*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':4})

GC_248 = Coupling(name = 'GC_248',
                  value = '(2*cpDC*ee0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':4})

GC_249 = Coupling(name = 'GC_249',
                  value = '(-8*cpW*cw0*ee0**2*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':4})

GC_25 = Coupling(name = 'GC_25',
                 value = '-(c3pl3/Lambda**2) + cpl3/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_250 = Coupling(name = 'GC_250',
                  value = '(-6*cWWW*ee0**2*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_251 = Coupling(name = 'GC_251',
                  value = '(-6*cw0**2*cWWW*ee0**2*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_252 = Coupling(name = 'GC_252',
                  value = '(24*cWWW*ee0**3*complex(0,1))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':4})

GC_253 = Coupling(name = 'GC_253',
                  value = '(ee0*complex(0,1)*sw0)/(6.*cw0)',
                  order = {'QED':1})

GC_254 = Coupling(name = 'GC_254',
                  value = '-(ee0*complex(0,1)*sw0)/(2.*cw0)',
                  order = {'QED':1})

GC_255 = Coupling(name = 'GC_255',
                  value = '(-2*cpWB*sw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_256 = Coupling(name = 'GC_256',
                  value = '(-2*cpWB*complex(0,1)*sw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_257 = Coupling(name = 'GC_257',
                  value = '(2*cpWB*sw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_258 = Coupling(name = 'GC_258',
                  value = '(6*cWWW*complex(0,1)*sw0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_259 = Coupling(name = 'GC_259',
                  value = '(-2*cpWB*ee0*complex(0,1)*sw0)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_26 = Coupling(name = 'GC_26',
                 value = 'c3pl3/Lambda**2 + cpl3/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_260 = Coupling(name = 'GC_260',
                  value = '(2*cpWB*ee0*complex(0,1)*sw0)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_261 = Coupling(name = 'GC_261',
                  value = '(2*cpWB*ee0*sw0)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_262 = Coupling(name = 'GC_262',
                  value = '-((c3pl1*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':3})

GC_263 = Coupling(name = 'GC_263',
                  value = '-((c3pl1*ee0*complex(0,1)*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':3})

GC_264 = Coupling(name = 'GC_264',
                  value = '(c3pl1*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_265 = Coupling(name = 'GC_265',
                  value = '-((c3pl2*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':3})

GC_266 = Coupling(name = 'GC_266',
                  value = '-((c3pl2*ee0*complex(0,1)*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':3})

GC_267 = Coupling(name = 'GC_267',
                  value = '(c3pl2*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_268 = Coupling(name = 'GC_268',
                  value = '-((c3pl3*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':3})

GC_269 = Coupling(name = 'GC_269',
                  value = '-((c3pl3*ee0*complex(0,1)*sw0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':3})

GC_27 = Coupling(name = 'GC_27',
                 value = '(2*cQl31*complex(0,1))/Lambda**2 + (cQlM1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_270 = Coupling(name = 'GC_270',
                  value = '(c3pl3*ee0*sw0*cmath.sqrt(2))/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_271 = Coupling(name = 'GC_271',
                  value = '(6*cWWW*ee0*complex(0,1)*sw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_272 = Coupling(name = 'GC_272',
                  value = '(-12*cWWW*ee0**2*complex(0,1)*sw0)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_273 = Coupling(name = 'GC_273',
                  value = '-(cw0*ee0*complex(0,1))/(2.*sw0) - (ee0*complex(0,1)*sw0)/(2.*cw0)',
                  order = {'QED':1})

GC_274 = Coupling(name = 'GC_274',
                  value = '(cw0*ee0*complex(0,1))/(2.*sw0) - (ee0*complex(0,1)*sw0)/(2.*cw0)',
                  order = {'QED':1})

GC_275 = Coupling(name = 'GC_275',
                  value = '(cw0*ee0)/(2.*sw0) + (ee0*sw0)/(2.*cw0)',
                  order = {'QED':1})

GC_276 = Coupling(name = 'GC_276',
                  value = '-((cw0*ee0**2*complex(0,1))/sw0) + (ee0**2*complex(0,1)*sw0)/cw0',
                  order = {'QED':2})

GC_277 = Coupling(name = 'GC_277',
                  value = '-((ctZ*cw0)/(Lambda**2*sw0)) + (ctW*cw0**2)/(Lambda**2*sw0) - (ctW*sw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_278 = Coupling(name = 'GC_278',
                  value = '(ctZ*cw0)/(Lambda**2*sw0) - (ctW*cw0**2)/(Lambda**2*sw0) + (ctW*sw0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_279 = Coupling(name = 'GC_279',
                  value = '-((ctZ*cw0*complex(0,1))/(Lambda**2*sw0*cmath.sqrt(2))) + (ctW*cw0**2*complex(0,1))/(Lambda**2*sw0*cmath.sqrt(2)) + (ctW*complex(0,1)*sw0)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_28 = Coupling(name = 'GC_28',
                 value = '(2*cQl32*complex(0,1))/Lambda**2 + (cQlM2*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_280 = Coupling(name = 'GC_280',
                  value = '-((ctZ*cw0)/(Lambda**2*sw0*cmath.sqrt(2))) + (ctW*cw0**2)/(Lambda**2*sw0*cmath.sqrt(2)) + (ctW*sw0)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_281 = Coupling(name = 'GC_281',
                  value = '-(cpDC*cw0*ee0*complex(0,1))/(2.*Lambda**2*sw0) - (cpDC*ee0*complex(0,1)*sw0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_282 = Coupling(name = 'GC_282',
                  value = '(cpDC*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (cpDC*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_283 = Coupling(name = 'GC_283',
                  value = '-(cpDC*cw0*ee0)/(2.*Lambda**2*sw0) + (cpDC*ee0*sw0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_284 = Coupling(name = 'GC_284',
                  value = '(cpDC*cw0*ee0)/(2.*Lambda**2*sw0) + (cpDC*ee0*sw0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_285 = Coupling(name = 'GC_285',
                  value = '(3*cpDC*cw0*ee0)/(2.*Lambda**2*sw0) + (3*cpDC*ee0*sw0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_286 = Coupling(name = 'GC_286',
                  value = '-((cpe*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpe*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_287 = Coupling(name = 'GC_287',
                  value = '(cpe*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpe*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_288 = Coupling(name = 'GC_288',
                  value = '(c3pl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (cpl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (c3pl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_289 = Coupling(name = 'GC_289',
                  value = '-((c3pl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (c3pl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_29 = Coupling(name = 'GC_29',
                 value = '(2*cQl33*complex(0,1))/Lambda**2 + (cQlM3*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_290 = Coupling(name = 'GC_290',
                  value = '-((c3pl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) - (cpl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_291 = Coupling(name = 'GC_291',
                  value = '(c3pl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpl1*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl1*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_292 = Coupling(name = 'GC_292',
                  value = '(c3pl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (cpl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (c3pl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_293 = Coupling(name = 'GC_293',
                  value = '-((c3pl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (c3pl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_294 = Coupling(name = 'GC_294',
                  value = '-((c3pl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) - (cpl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (c3pl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_295 = Coupling(name = 'GC_295',
                  value = '(c3pl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpl2*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (c3pl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl2*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_296 = Coupling(name = 'GC_296',
                  value = '(c3pl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (cpl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (c3pl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_297 = Coupling(name = 'GC_297',
                  value = '-((c3pl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) - (c3pl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_298 = Coupling(name = 'GC_298',
                  value = '-((c3pl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) - (cpl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (c3pl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_299 = Coupling(name = 'GC_299',
                  value = '(c3pl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpl3*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (c3pl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_3 = Coupling(name = 'GC_3',
                value = '-(ee0*complex(0,1))',
                order = {'QED':1})

GC_30 = Coupling(name = 'GC_30',
                 value = '(cQq11*complex(0,1))/Lambda**2 - (cQq13*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_300 = Coupling(name = 'GC_300',
                  value = '-((cpmu*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpmu*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_301 = Coupling(name = 'GC_301',
                  value = '(cpmu*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpmu*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_302 = Coupling(name = 'GC_302',
                  value = '-((cpta*cw0*ee0*complex(0,1))/(Lambda**2*sw0)) + (cpta*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_303 = Coupling(name = 'GC_303',
                  value = '(cpta*cw0*ee0*complex(0,1))/(Lambda**2*sw0) + (cpta*ee0*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_304 = Coupling(name = 'GC_304',
                  value = '(cpDC*cw0*ee0**2*complex(0,1))/(Lambda**2*sw0) + (cpDC*ee0**2*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':4})

GC_305 = Coupling(name = 'GC_305',
                  value = '(-4*cpDC*cw0*ee0**2*complex(0,1))/(Lambda**2*sw0) + (4*cpDC*ee0**2*complex(0,1)*sw0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':4})

GC_306 = Coupling(name = 'GC_306',
                  value = '-(ee0**2*complex(0,1)) + (cw0**2*ee0**2*complex(0,1))/(2.*sw0**2) + (ee0**2*complex(0,1)*sw0**2)/(2.*cw0**2)',
                  order = {'QED':2})

GC_307 = Coupling(name = 'GC_307',
                  value = 'ee0**2*complex(0,1) + (cw0**2*ee0**2*complex(0,1))/(2.*sw0**2) + (ee0**2*complex(0,1)*sw0**2)/(2.*cw0**2)',
                  order = {'QED':2})

GC_308 = Coupling(name = 'GC_308',
                  value = '(4*cpW*cw0**2*complex(0,1))/Lambda**2 - (4*cpWB*cw0*complex(0,1)*sw0)/Lambda**2 + (4*cpBB*complex(0,1)*sw0**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_309 = Coupling(name = 'GC_309',
                  value = '(4*cpW*cw0**2*complex(0,1))/Lambda**2 + (4*cpWB*cw0*complex(0,1)*sw0)/Lambda**2 + (4*cpBB*complex(0,1)*sw0**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_31 = Coupling(name = 'GC_31',
                 value = '(cQq11*complex(0,1))/Lambda**2 + (cQq13*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_310 = Coupling(name = 'GC_310',
                  value = '(4*cpBB*cw0**2*complex(0,1))/Lambda**2 - (4*cpWB*cw0*complex(0,1)*sw0)/Lambda**2 + (4*cpW*complex(0,1)*sw0**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_311 = Coupling(name = 'GC_311',
                  value = '(4*cpBB*cw0**2*complex(0,1))/Lambda**2 + (4*cpWB*cw0*complex(0,1)*sw0)/Lambda**2 + (4*cpW*complex(0,1)*sw0**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_312 = Coupling(name = 'GC_312',
                  value = '(2*cpWB*cw0**2*complex(0,1))/Lambda**2 + (4*cpBB*cw0*complex(0,1)*sw0)/Lambda**2 - (4*cpW*cw0*complex(0,1)*sw0)/Lambda**2 - (2*cpWB*complex(0,1)*sw0**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_313 = Coupling(name = 'GC_313',
                  value = '(-2*cpWB*cw0**2*complex(0,1))/Lambda**2 + (4*cpBB*cw0*complex(0,1)*sw0)/Lambda**2 - (4*cpW*cw0*complex(0,1)*sw0)/Lambda**2 + (2*cpWB*complex(0,1)*sw0**2)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_314 = Coupling(name = 'GC_314',
                  value = '-((cpDC*cw0**2*ee0**2*complex(0,1))/(Lambda**2*sw0**2)) + (cpDC*ee0**2*complex(0,1)*sw0**2)/(cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':4})

GC_315 = Coupling(name = 'GC_315',
                  value = '(2*cpDC*ee0**2*complex(0,1))/Lambda**2 + (cpDC*cw0**2*ee0**2*complex(0,1))/(Lambda**2*sw0**2) + (cpDC*ee0**2*complex(0,1)*sw0**2)/(cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':4})

GC_316 = Coupling(name = 'GC_316',
                  value = '(-4*cpDC*ee0**2*complex(0,1))/Lambda**2 + (2*cpDC*cw0**2*ee0**2*complex(0,1))/(Lambda**2*sw0**2) + (2*cpDC*ee0**2*complex(0,1)*sw0**2)/(cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':4})

GC_317 = Coupling(name = 'GC_317',
                  value = '(6*cpDC*ee0**2*complex(0,1))/Lambda**2 + (3*cpDC*cw0**2*ee0**2*complex(0,1))/(Lambda**2*sw0**2) + (3*cpDC*ee0**2*complex(0,1)*sw0**2)/(cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':4})

GC_318 = Coupling(name = 'GC_318',
                  value = '-((complex(0,1)*MH**2)/vev0**2)',
                  order = {'QED':2})

GC_319 = Coupling(name = 'GC_319',
                  value = '(-2*complex(0,1)*MH**2)/vev0**2',
                  order = {'QED':2})

GC_32 = Coupling(name = 'GC_32',
                 value = '(cQQ1*complex(0,1))/Lambda**2 - (cQQ8*complex(0,1))/(6.*Lambda**2)',
                 order = {'NP':2})

GC_320 = Coupling(name = 'GC_320',
                  value = '(-3*complex(0,1)*MH**2)/vev0**2',
                  order = {'QED':2})

GC_321 = Coupling(name = 'GC_321',
                  value = '-((complex(0,1)*MH**2)/vev0)',
                  order = {'QED':1})

GC_322 = Coupling(name = 'GC_322',
                  value = '(-3*complex(0,1)*MH**2)/vev0',
                  order = {'QED':1})

GC_323 = Coupling(name = 'GC_323',
                  value = '-(ee0**2*vev0)/(2.*cw0)',
                  order = {'QED':1})

GC_324 = Coupling(name = 'GC_324',
                  value = '(ee0**2*vev0)/(2.*cw0)',
                  order = {'QED':1})

GC_325 = Coupling(name = 'GC_325',
                  value = '-((c3pl1*vev0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_326 = Coupling(name = 'GC_326',
                  value = '-((c3pl2*vev0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_327 = Coupling(name = 'GC_327',
                  value = '-((c3pl3*vev0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_328 = Coupling(name = 'GC_328',
                  value = '(2*cdp*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_329 = Coupling(name = 'GC_329',
                  value = '(6*cp*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_33 = Coupling(name = 'GC_33',
                 value = '(cQQ1*complex(0,1))/Lambda**2 + (cQQ8*complex(0,1))/(3.*Lambda**2)',
                 order = {'NP':2})

GC_330 = Coupling(name = 'GC_330',
                  value = '(12*cp*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_331 = Coupling(name = 'GC_331',
                  value = '(18*cp*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_332 = Coupling(name = 'GC_332',
                  value = '(90*cp*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_333 = Coupling(name = 'GC_333',
                  value = '-(cpDC*vev0)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_334 = Coupling(name = 'GC_334',
                  value = '-((cpDC*complex(0,1)*vev0)/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_335 = Coupling(name = 'GC_335',
                  value = '(cpe*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_336 = Coupling(name = 'GC_336',
                  value = '(4*cpG*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_337 = Coupling(name = 'GC_337',
                  value = '(cpmu*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_338 = Coupling(name = 'GC_338',
                  value = '(cpta*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_339 = Coupling(name = 'GC_339',
                  value = '(4*cpW*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_34 = Coupling(name = 'GC_34',
                 value = '(cQq81*complex(0,1))/Lambda**2 - (cQq83*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_340 = Coupling(name = 'GC_340',
                  value = '(ctW*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_341 = Coupling(name = 'GC_341',
                  value = '-((ctZ*complex(0,1)*vev0)/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QED':1})

GC_342 = Coupling(name = 'GC_342',
                  value = '(-2*cpWB*cw0*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_343 = Coupling(name = 'GC_343',
                  value = '(2*cpWB*cw0*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_344 = Coupling(name = 'GC_344',
                  value = '-((c3pl1*ee0*vev0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_345 = Coupling(name = 'GC_345',
                  value = '(c3pl1*ee0*vev0*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_346 = Coupling(name = 'GC_346',
                  value = '-((c3pl2*ee0*vev0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_347 = Coupling(name = 'GC_347',
                  value = '(c3pl2*ee0*vev0*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_348 = Coupling(name = 'GC_348',
                  value = '-((c3pl3*ee0*vev0*cmath.sqrt(2))/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_349 = Coupling(name = 'GC_349',
                  value = '(c3pl3*ee0*vev0*cmath.sqrt(2))/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_35 = Coupling(name = 'GC_35',
                 value = '(cQq81*complex(0,1))/Lambda**2 + (cQq83*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_350 = Coupling(name = 'GC_350',
                  value = '(cpDC*ee0*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_351 = Coupling(name = 'GC_351',
                  value = '(-4*cpW*ee0*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_352 = Coupling(name = 'GC_352',
                  value = '(2*cpWB*ee0*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_353 = Coupling(name = 'GC_353',
                  value = '-((ctW*ee0*complex(0,1)*vev0)/Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_354 = Coupling(name = 'GC_354',
                  value = '(ctW*ee0*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_355 = Coupling(name = 'GC_355',
                  value = '(2*cpWB*cw0*ee0*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_356 = Coupling(name = 'GC_356',
                  value = '(-4*cpW*ee0**2*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':3})

GC_357 = Coupling(name = 'GC_357',
                  value = '(4*cpG*G*vev0)/Lambda**2',
                  order = {'NP':2,'QCD':1,'QED':1})

GC_358 = Coupling(name = 'GC_358',
                  value = '(ctG*complex(0,1)*G*vev0)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QCD':1})

GC_359 = Coupling(name = 'GC_359',
                  value = '(-4*cpG*complex(0,1)*G**2*vev0)/Lambda**2',
                  order = {'NP':2,'QCD':2,'QED':1})

GC_36 = Coupling(name = 'GC_36',
                 value = 'ctZ/Lambda**2 - (2*ctW*cw0)/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_360 = Coupling(name = 'GC_360',
                  value = '-((ctG*G**2*vev0)/(Lambda**2*cmath.sqrt(2)))',
                  order = {'NP':2,'QCD':2})

GC_361 = Coupling(name = 'GC_361',
                  value = '-(ee0**2*vev0)/(4.*sw0**2)',
                  order = {'QED':1})

GC_362 = Coupling(name = 'GC_362',
                  value = '-(ee0**2*complex(0,1)*vev0)/(4.*sw0**2)',
                  order = {'QED':1})

GC_363 = Coupling(name = 'GC_363',
                  value = '(ee0**2*complex(0,1)*vev0)/(2.*sw0**2)',
                  order = {'QED':1})

GC_364 = Coupling(name = 'GC_364',
                  value = '(ee0**2*vev0)/(4.*sw0**2)',
                  order = {'QED':1})

GC_365 = Coupling(name = 'GC_365',
                  value = '(-2*cpDC*ee0**2*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_366 = Coupling(name = 'GC_366',
                  value = '(cpDC*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_367 = Coupling(name = 'GC_367',
                  value = '(-2*cpDC*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_368 = Coupling(name = 'GC_368',
                  value = '(2*cpDC*ee0**2*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_369 = Coupling(name = 'GC_369',
                  value = '(4*cpW*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_37 = Coupling(name = 'GC_37',
                 value = '-(ctZ/Lambda**2) + (2*ctW*cw0)/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_370 = Coupling(name = 'GC_370',
                  value = '(-4*cpW*cw0**2*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_371 = Coupling(name = 'GC_371',
                  value = '-(ee0**2*vev0)/(2.*sw0)',
                  order = {'QED':1})

GC_372 = Coupling(name = 'GC_372',
                  value = '(ee0**2*vev0)/(2.*sw0)',
                  order = {'QED':1})

GC_373 = Coupling(name = 'GC_373',
                  value = '(c3pl1*ee0*complex(0,1)*vev0*cmath.sqrt(2))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_374 = Coupling(name = 'GC_374',
                  value = '(c3pl2*ee0*complex(0,1)*vev0*cmath.sqrt(2))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_375 = Coupling(name = 'GC_375',
                  value = '(c3pl3*ee0*complex(0,1)*vev0*cmath.sqrt(2))/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_376 = Coupling(name = 'GC_376',
                  value = '-(cpDC*ee0*vev0)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_377 = Coupling(name = 'GC_377',
                  value = '-((cpDC*ee0*complex(0,1)*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_378 = Coupling(name = 'GC_378',
                  value = '(cpDC*ee0*complex(0,1)*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_379 = Coupling(name = 'GC_379',
                  value = '(cpDC*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_38 = Coupling(name = 'GC_38',
                 value = '(-2*c3pl1*ee0*complex(0,1))/Lambda**2 + (2*cpl1*ee0*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_380 = Coupling(name = 'GC_380',
                  value = '-((cpe*ee0*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_381 = Coupling(name = 'GC_381',
                  value = '(cpe*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_382 = Coupling(name = 'GC_382',
                  value = '-((cpl1*ee0*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_383 = Coupling(name = 'GC_383',
                  value = '(cpl1*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_384 = Coupling(name = 'GC_384',
                  value = '-((cpl2*ee0*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_385 = Coupling(name = 'GC_385',
                  value = '(cpl2*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_386 = Coupling(name = 'GC_386',
                  value = '-((cpl3*ee0*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_387 = Coupling(name = 'GC_387',
                  value = '(cpl3*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_388 = Coupling(name = 'GC_388',
                  value = '-((cpmu*ee0*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_389 = Coupling(name = 'GC_389',
                  value = '(cpmu*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_39 = Coupling(name = 'GC_39',
                 value = '(2*c3pl1*ee0*complex(0,1))/Lambda**2 + (2*cpl1*ee0*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_390 = Coupling(name = 'GC_390',
                  value = '-((cpta*ee0*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_391 = Coupling(name = 'GC_391',
                  value = '(cpta*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_392 = Coupling(name = 'GC_392',
                  value = '(ctW*ee0*complex(0,1)*vev0)/(Lambda**2*sw0*cmath.sqrt(2))',
                  order = {'NP':2,'QED':2})

GC_393 = Coupling(name = 'GC_393',
                  value = '(4*cpW*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_394 = Coupling(name = 'GC_394',
                  value = '(2*cpWB*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_395 = Coupling(name = 'GC_395',
                  value = '-((ctW*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0))',
                  order = {'NP':2,'QED':2})

GC_396 = Coupling(name = 'GC_396',
                  value = '(ctW*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_397 = Coupling(name = 'GC_397',
                  value = '(2*cpWB*cw0**2*ee0*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_398 = Coupling(name = 'GC_398',
                  value = '(-2*cpDC*ee0**2*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_399 = Coupling(name = 'GC_399',
                  value = '(2*cpDC*ee0**2*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_4 = Coupling(name = 'GC_4',
                value = 'ee0*complex(0,1)',
                order = {'QED':1})

GC_40 = Coupling(name = 'GC_40',
                 value = '(-2*c3pl2*ee0*complex(0,1))/Lambda**2 + (2*cpl2*ee0*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_400 = Coupling(name = 'GC_400',
                  value = '(-8*cpW*cw0*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':3})

GC_401 = Coupling(name = 'GC_401',
                  value = '(-2*cpWB*sw0*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_402 = Coupling(name = 'GC_402',
                  value = '(2*cpWB*sw0*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_403 = Coupling(name = 'GC_403',
                  value = '(2*cpWB*ee0*sw0*vev0)/Lambda**2',
                  order = {'NP':2,'QED':2})

GC_404 = Coupling(name = 'GC_404',
                  value = '-((c3pl1*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':2})

GC_405 = Coupling(name = 'GC_405',
                  value = '(c3pl1*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_406 = Coupling(name = 'GC_406',
                  value = '-((c3pl2*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':2})

GC_407 = Coupling(name = 'GC_407',
                  value = '(c3pl2*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_408 = Coupling(name = 'GC_408',
                  value = '-((c3pl3*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2))',
                  order = {'NP':2,'QED':2})

GC_409 = Coupling(name = 'GC_409',
                  value = '(c3pl3*ee0*sw0*vev0*cmath.sqrt(2))/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_41 = Coupling(name = 'GC_41',
                 value = '(2*c3pl2*ee0*complex(0,1))/Lambda**2 + (2*cpl2*ee0*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_410 = Coupling(name = 'GC_410',
                  value = '(6*cp*complex(0,1)*vev0**2)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_411 = Coupling(name = 'GC_411',
                  value = '(36*cp*complex(0,1)*vev0**2)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_412 = Coupling(name = 'GC_412',
                  value = '-((cpWB*ee0*complex(0,1)*vev0**2)/Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_413 = Coupling(name = 'GC_413',
                  value = '(cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_414 = Coupling(name = 'GC_414',
                  value = '-((cpDC*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2))',
                  order = {'NP':2,'QED':2})

GC_415 = Coupling(name = 'GC_415',
                  value = '(-3*cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_416 = Coupling(name = 'GC_416',
                  value = '(3*cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_417 = Coupling(name = 'GC_417',
                  value = '(cpWB*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_418 = Coupling(name = 'GC_418',
                  value = '(3*c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_419 = Coupling(name = 'GC_419',
                  value = '(3*c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_42 = Coupling(name = 'GC_42',
                 value = '(-2*c3pl3*ee0*complex(0,1))/Lambda**2 + (2*cpl3*ee0*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_420 = Coupling(name = 'GC_420',
                  value = '(6*cp*complex(0,1)*vev0**3)/Lambda**2',
                  order = {'NP':2})

GC_421 = Coupling(name = 'GC_421',
                  value = '(2*cdp*complex(0,1)*vev0)/Lambda**2 - (cpDC*complex(0,1)*vev0)/(2.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_422 = Coupling(name = 'GC_422',
                  value = '(4*cdp*complex(0,1)*vev0)/Lambda**2 - (cpDC*complex(0,1)*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_423 = Coupling(name = 'GC_423',
                  value = '-((c3pl1*vev0)/Lambda**2) + (cpl1*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_424 = Coupling(name = 'GC_424',
                  value = '(c3pl1*vev0)/Lambda**2 + (cpl1*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_425 = Coupling(name = 'GC_425',
                  value = '-((c3pl2*vev0)/Lambda**2) + (cpl2*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_426 = Coupling(name = 'GC_426',
                  value = '(c3pl2*vev0)/Lambda**2 + (cpl2*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_427 = Coupling(name = 'GC_427',
                  value = '-((c3pl3*vev0)/Lambda**2) + (cpl3*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_428 = Coupling(name = 'GC_428',
                  value = '(c3pl3*vev0)/Lambda**2 + (cpl3*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_429 = Coupling(name = 'GC_429',
                  value = '(c3pl1*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (c3pl2*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (cdp*complex(0,1)*MH**2*vev0)/Lambda**2 - (cll1221*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) - (cpDC*complex(0,1)*MH**2*vev0)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_43 = Coupling(name = 'GC_43',
                 value = '(2*c3pl3*ee0*complex(0,1))/Lambda**2 + (2*cpl3*ee0*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_430 = Coupling(name = 'GC_430',
                  value = '(c3pl1*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (c3pl2*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (cdp*complex(0,1)*MH**2*vev0)/Lambda**2 - (cll1221*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (cpDC*complex(0,1)*MH**2*vev0)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_431 = Coupling(name = 'GC_431',
                  value = '(3*c3pl1*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (3*c3pl2*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) - (3*cdp*complex(0,1)*MH**2*vev0)/Lambda**2 - (3*cll1221*complex(0,1)*MH**2*vev0)/(2.*Lambda**2) + (3*cpDC*complex(0,1)*MH**2*vev0)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_432 = Coupling(name = 'GC_432',
                  value = '-(ee0**2*vev0)/(4.*cw0) - (cw0*ee0**2*vev0)/(4.*sw0**2)',
                  order = {'QED':1})

GC_433 = Coupling(name = 'GC_433',
                  value = '(ee0**2*vev0)/(4.*cw0) - (cw0*ee0**2*vev0)/(4.*sw0**2)',
                  order = {'QED':1})

GC_434 = Coupling(name = 'GC_434',
                  value = '-(ee0**2*vev0)/(4.*cw0) + (cw0*ee0**2*vev0)/(4.*sw0**2)',
                  order = {'QED':1})

GC_435 = Coupling(name = 'GC_435',
                  value = '(ee0**2*vev0)/(4.*cw0) + (cw0*ee0**2*vev0)/(4.*sw0**2)',
                  order = {'QED':1})

GC_436 = Coupling(name = 'GC_436',
                  value = '(-3*cpDC*ee0**2*vev0)/(2.*cw0*Lambda**2) - (3*cpDC*cw0*ee0**2*vev0)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_437 = Coupling(name = 'GC_437',
                  value = '(cpDC*ee0**2*vev0)/(cw0*Lambda**2) - (cpDC*cw0*ee0**2*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_438 = Coupling(name = 'GC_438',
                  value = '-(cpDC*ee0**2*vev0)/(2.*cw0*Lambda**2) - (cpDC*cw0*ee0**2*vev0)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_439 = Coupling(name = 'GC_439',
                  value = '-(cpDC*ee0**2*complex(0,1)*vev0)/(2.*cw0*Lambda**2) - (cpDC*cw0*ee0**2*complex(0,1)*vev0)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_44 = Coupling(name = 'GC_44',
                 value = '-((c3pl1*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_440 = Coupling(name = 'GC_440',
                  value = '(cpDC*ee0**2*vev0)/(2.*cw0*Lambda**2) + (cpDC*cw0*ee0**2*vev0)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_441 = Coupling(name = 'GC_441',
                  value = '-((cpDC*ee0**2*vev0)/(cw0*Lambda**2)) + (cpDC*cw0*ee0**2*vev0)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_442 = Coupling(name = 'GC_442',
                  value = '(3*cpDC*ee0**2*vev0)/(2.*cw0*Lambda**2) + (3*cpDC*cw0*ee0**2*vev0)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':3})

GC_443 = Coupling(name = 'GC_443',
                  value = '-((ctZ*cw0*complex(0,1)*vev0)/(Lambda**2*sw0*cmath.sqrt(2))) + (ctW*cw0**2*complex(0,1)*vev0)/(Lambda**2*sw0*cmath.sqrt(2)) + (ctW*complex(0,1)*sw0*vev0)/(Lambda**2*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_444 = Coupling(name = 'GC_444',
                  value = '-(cpDC*cw0*ee0*complex(0,1)*vev0)/(2.*Lambda**2*sw0) - (cpDC*ee0*complex(0,1)*sw0*vev0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_445 = Coupling(name = 'GC_445',
                  value = '-(cpDC*cw0*ee0*vev0)/(2.*Lambda**2*sw0) + (cpDC*ee0*sw0*vev0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_446 = Coupling(name = 'GC_446',
                  value = '(cpDC*cw0*ee0*vev0)/(2.*Lambda**2*sw0) + (cpDC*ee0*sw0*vev0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_447 = Coupling(name = 'GC_447',
                  value = '(3*cpDC*cw0*ee0*vev0)/(2.*Lambda**2*sw0) + (3*cpDC*ee0*sw0*vev0)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_448 = Coupling(name = 'GC_448',
                  value = '(cpe*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpe*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_449 = Coupling(name = 'GC_449',
                  value = '-((c3pl1*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0)) + (cpl1*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) - (c3pl1*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpl1*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_45 = Coupling(name = 'GC_45',
                 value = '-((c3pl1*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_450 = Coupling(name = 'GC_450',
                  value = '(c3pl1*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpl1*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpl1*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_451 = Coupling(name = 'GC_451',
                  value = '-((c3pl2*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0)) + (cpl2*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) - (c3pl2*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpl2*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_452 = Coupling(name = 'GC_452',
                  value = '(c3pl2*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpl2*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (c3pl2*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpl2*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_453 = Coupling(name = 'GC_453',
                  value = '-((c3pl3*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0)) + (cpl3*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) - (c3pl3*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_454 = Coupling(name = 'GC_454',
                  value = '(c3pl3*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpl3*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (c3pl3*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_455 = Coupling(name = 'GC_455',
                  value = '(cpmu*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpmu*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_456 = Coupling(name = 'GC_456',
                  value = '(cpta*cw0*ee0*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpta*ee0*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_457 = Coupling(name = 'GC_457',
                  value = '(cpDC*cw0*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0) + (cpDC*ee0**2*complex(0,1)*sw0*vev0)/(cw0*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_458 = Coupling(name = 'GC_458',
                  value = '-(ee0**2*complex(0,1)*vev0)/2. - (cw0**2*ee0**2*complex(0,1)*vev0)/(4.*sw0**2) - (ee0**2*complex(0,1)*sw0**2*vev0)/(4.*cw0**2)',
                  order = {'QED':1})

GC_459 = Coupling(name = 'GC_459',
                  value = 'ee0**2*complex(0,1)*vev0 + (cw0**2*ee0**2*complex(0,1)*vev0)/(2.*sw0**2) + (ee0**2*complex(0,1)*sw0**2*vev0)/(2.*cw0**2)',
                  order = {'QED':1})

GC_46 = Coupling(name = 'GC_46',
                 value = '(c3pl1*complex(0,1)*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_460 = Coupling(name = 'GC_460',
                  value = '(4*cpW*cw0**2*complex(0,1)*vev0)/Lambda**2 + (4*cpWB*cw0*complex(0,1)*sw0*vev0)/Lambda**2 + (4*cpBB*complex(0,1)*sw0**2*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_461 = Coupling(name = 'GC_461',
                  value = '(4*cpBB*cw0**2*complex(0,1)*vev0)/Lambda**2 - (4*cpWB*cw0*complex(0,1)*sw0*vev0)/Lambda**2 + (4*cpW*complex(0,1)*sw0**2*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_462 = Coupling(name = 'GC_462',
                  value = '(2*cpWB*cw0**2*complex(0,1)*vev0)/Lambda**2 + (4*cpBB*cw0*complex(0,1)*sw0*vev0)/Lambda**2 - (4*cpW*cw0*complex(0,1)*sw0*vev0)/Lambda**2 - (2*cpWB*complex(0,1)*sw0**2*vev0)/Lambda**2',
                  order = {'NP':2,'QED':1})

GC_463 = Coupling(name = 'GC_463',
                  value = '-((cpDC*cw0**2*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0**2)) + (cpDC*ee0**2*complex(0,1)*sw0**2*vev0)/(cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_464 = Coupling(name = 'GC_464',
                  value = '(2*cpDC*ee0**2*complex(0,1)*vev0)/Lambda**2 + (cpDC*cw0**2*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0**2) + (cpDC*ee0**2*complex(0,1)*sw0**2*vev0)/(cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_465 = Coupling(name = 'GC_465',
                  value = '(6*cpDC*ee0**2*complex(0,1)*vev0)/Lambda**2 + (3*cpDC*cw0**2*ee0**2*complex(0,1)*vev0)/(Lambda**2*sw0**2) + (3*cpDC*ee0**2*complex(0,1)*sw0**2*vev0)/(cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':3})

GC_466 = Coupling(name = 'GC_466',
                  value = '(cpDC*ee0*complex(0,1)*vev0**2)/(24.*Lambda**2) - (cpDC*ee0*complex(0,1)*vev0**2)/(24.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_467 = Coupling(name = 'GC_467',
                  value = '-(cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2) + (cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_468 = Coupling(name = 'GC_468',
                  value = '-(cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2) + (cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2) - (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_469 = Coupling(name = 'GC_469',
                  value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_47 = Coupling(name = 'GC_47',
                 value = '-((c3pl2*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_470 = Coupling(name = 'GC_470',
                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_471 = Coupling(name = 'GC_471',
                  value = '(cpDC*ee0*vev0**2)/(8.*Lambda**2) - (cpDC*ee0*vev0**2)/(8.*Lambda**2*sw0**2) + (cpDC*cw0**2*ee0*vev0**2)/(8.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_472 = Coupling(name = 'GC_472',
                  value = '(c3pl1*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) + (c3pl2*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) - (cll1221*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':2})

GC_473 = Coupling(name = 'GC_473',
                  value = '-(c3pl1*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (c3pl2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cll1221*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':2})

GC_474 = Coupling(name = 'GC_474',
                  value = '-(c3pl1*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (c3pl2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cdp*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) + (cll1221*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':2})

GC_475 = Coupling(name = 'GC_475',
                  value = '-(c3pl1*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (c3pl2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cll1221*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cpDC*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':2})

GC_476 = Coupling(name = 'GC_476',
                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_477 = Coupling(name = 'GC_477',
                  value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_478 = Coupling(name = 'GC_478',
                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_479 = Coupling(name = 'GC_479',
                  value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_48 = Coupling(name = 'GC_48',
                 value = '-((c3pl2*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_480 = Coupling(name = 'GC_480',
                  value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2)) + (c3pl3*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0*cmath.sqrt(2)) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0*cmath.sqrt(2))',
                  order = {'NP':2,'QED':1})

GC_481 = Coupling(name = 'GC_481',
                  value = '(c3pl1*ee0*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*ee0*vev0**2)/(4.*Lambda**2*sw0) - (cdp*ee0*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*ee0*vev0**2)/(4.*Lambda**2*sw0) + (cpDC*ee0*vev0**2)/(8.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_482 = Coupling(name = 'GC_482',
                  value = '(cpWB*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) + (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(12.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_483 = Coupling(name = 'GC_483',
                  value = '(-2*cpWB*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_484 = Coupling(name = 'GC_484',
                  value = '(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_485 = Coupling(name = 'GC_485',
                  value = '-((cpWB*ee0*complex(0,1)*vev0**2)/Lambda**2) + (c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_486 = Coupling(name = 'GC_486',
                  value = '(cpWB*ee0*complex(0,1)*vev0**2)/Lambda**2 + (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_487 = Coupling(name = 'GC_487',
                  value = '-(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_488 = Coupling(name = 'GC_488',
                  value = '(cpWB*ee0*complex(0,1)*vev0**2)/Lambda**2 - (c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpDC*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_489 = Coupling(name = 'GC_489',
                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2) + (c3pl2*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2) - (cll1221*ee0*complex(0,1)*vev0**2)/(6.*Lambda**2) + (cpWB*cw0*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_49 = Coupling(name = 'GC_49',
                 value = '(c3pl2*complex(0,1)*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_490 = Coupling(name = 'GC_490',
                  value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (c3pl2*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) + (cll1221*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2) - (2*cpWB*cw0*ee0*complex(0,1)*vev0**2)/(3.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_491 = Coupling(name = 'GC_491',
                  value = '-(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2) - (cpWB*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_492 = Coupling(name = 'GC_492',
                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (cpWB*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_493 = Coupling(name = 'GC_493',
                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2) + (cpDC*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2) + (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2) + (cpWB*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_494 = Coupling(name = 'GC_494',
                  value = '(c3pl1*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (c3pl2*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cll1221*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2) + (cpDC*cw0**2*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2) + (cpWB*cw0*ee0*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_495 = Coupling(name = 'GC_495',
                  value = '(cpDC*ee0**2*vev0**2)/(8.*Lambda**2*sw0**3) + (cpWB*cw0*ee0**2*vev0**2)/(2.*Lambda**2*sw0**2) + (c3pl1*ee0**2*vev0**2)/(2.*Lambda**2*sw0) + (c3pl2*ee0**2*vev0**2)/(2.*Lambda**2*sw0) - (cdp*ee0**2*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*ee0**2*vev0**2)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_496 = Coupling(name = 'GC_496',
                  value = '(cpDC*ee0**2*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**3) + (cpWB*cw0*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (c3pl1*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (c3pl2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_497 = Coupling(name = 'GC_497',
                  value = '-(cpDC*ee0**2*vev0**2)/(8.*Lambda**2*sw0**3) - (cpWB*cw0*ee0**2*vev0**2)/(2.*Lambda**2*sw0**2) - (c3pl1*ee0**2*vev0**2)/(2.*Lambda**2*sw0) - (c3pl2*ee0**2*vev0**2)/(2.*Lambda**2*sw0) + (cdp*ee0**2*vev0**2)/(2.*Lambda**2*sw0) + (cll1221*ee0**2*vev0**2)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_498 = Coupling(name = 'GC_498',
                  value = '-(c3pl1*ee0**2*vev0**2)/(2.*cw0*Lambda**2) - (c3pl2*ee0**2*vev0**2)/(2.*cw0*Lambda**2) + (cdp*ee0**2*vev0**2)/(2.*cw0*Lambda**2) + (cll1221*ee0**2*vev0**2)/(2.*cw0*Lambda**2) + (5*cpDC*ee0**2*vev0**2)/(8.*cw0*Lambda**2) - (cpDC*ee0**2*vev0**2)/(8.*cw0*Lambda**2*sw0**2) + (5*cpDC*cw0*ee0**2*vev0**2)/(8.*Lambda**2*sw0**2) - (cpWB*ee0**2*vev0**2)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_499 = Coupling(name = 'GC_499',
                  value = '(c3pl1*ee0**2*complex(0,1)*vev0**2)/(2.*cw0*Lambda**2) + (c3pl2*ee0**2*complex(0,1)*vev0**2)/(2.*cw0*Lambda**2) - (cll1221*ee0**2*complex(0,1)*vev0**2)/(2.*cw0*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2) + (cpDC*ee0**2*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0**2) - (cpDC*cw0*ee0**2*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**2) + (cpWB*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_5 = Coupling(name = 'GC_5',
                value = 'ee0**2*complex(0,1)',
                order = {'QED':2})

GC_50 = Coupling(name = 'GC_50',
                 value = '-((c3pl3*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_500 = Coupling(name = 'GC_500',
                  value = '(c3pl1*ee0**2*vev0**2)/(2.*cw0*Lambda**2) + (c3pl2*ee0**2*vev0**2)/(2.*cw0*Lambda**2) - (cdp*ee0**2*vev0**2)/(2.*cw0*Lambda**2) - (cll1221*ee0**2*vev0**2)/(2.*cw0*Lambda**2) - (5*cpDC*ee0**2*vev0**2)/(8.*cw0*Lambda**2) + (cpDC*ee0**2*vev0**2)/(8.*cw0*Lambda**2*sw0**2) - (5*cpDC*cw0*ee0**2*vev0**2)/(8.*Lambda**2*sw0**2) + (cpWB*ee0**2*vev0**2)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_501 = Coupling(name = 'GC_501',
                  value = '(2*cpWB*ee0**2*complex(0,1)*vev0**2)/Lambda**2 - (cpDC*cw0**3*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**3) - (2*cpWB*cw0**2*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) - (2*c3pl1*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0) - (2*c3pl2*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0) + (2*cll1221*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0) + (cpDC*cw0*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_502 = Coupling(name = 'GC_502',
                  value = '-((c3pl1*ee0**2*complex(0,1)*vev0**2)/Lambda**2) - (c3pl2*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (cll1221*ee0**2*complex(0,1)*vev0**2)/Lambda**2 - (cpDC*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (2*cpWB*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_503 = Coupling(name = 'GC_503',
                  value = '-((c3pl1*cw0**2*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2)) - (c3pl2*cw0**2*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) + (cll1221*cw0**2*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) + (cpDC*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (2*cpWB*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_504 = Coupling(name = 'GC_504',
                  value = '(-2*c3pl1*ee0**2*complex(0,1)*vev0**2)/Lambda**2 - (2*c3pl2*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (2*cll1221*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (cpDC*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (cpDC*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (4*cpWB*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0)',
                  order = {'NP':2,'QED':2})

GC_505 = Coupling(name = 'GC_505',
                  value = '-(cpDC*ee0*complex(0,1)*vev0**2)/(24.*cw0*Lambda**2*sw0) - (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2) - (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2) + (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(12.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_506 = Coupling(name = 'GC_506',
                  value = '(cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_507 = Coupling(name = 'GC_507',
                  value = '(cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) + (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_508 = Coupling(name = 'GC_508',
                  value = '(cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_509 = Coupling(name = 'GC_509',
                  value = '-(cpDC*ee0*vev0**2)/(8.*cw0*Lambda**2*sw0) - (c3pl1*cw0*ee0*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*vev0**2)/(4.*Lambda**2*sw0) + (cdp*cw0*ee0*vev0**2)/(2.*Lambda**2*sw0) + (cll1221*cw0*ee0*vev0**2)/(4.*Lambda**2*sw0) - (c3pl1*ee0*sw0*vev0**2)/(4.*cw0*Lambda**2) - (c3pl2*ee0*sw0*vev0**2)/(4.*cw0*Lambda**2) + (cdp*ee0*sw0*vev0**2)/(2.*cw0*Lambda**2) + (cll1221*ee0*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_51 = Coupling(name = 'GC_51',
                 value = '-((c3pl3*complex(0,1)*cmath.sqrt(2))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_510 = Coupling(name = 'GC_510',
                  value = '(cpWB*ee0*complex(0,1)*vev0**2)/Lambda**2 + (cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) - (c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cpDC*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_511 = Coupling(name = 'GC_511',
                  value = '(cpDC*cw0*ee0*vev0**2)/(2.*Lambda**2*sw0) + (cpDC*ee0*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_512 = Coupling(name = 'GC_512',
                  value = '(cpe*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpe*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_513 = Coupling(name = 'GC_513',
                  value = '(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpl1*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpl1*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_514 = Coupling(name = 'GC_514',
                  value = '(cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) - (c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpl1*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (cpl1*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_515 = Coupling(name = 'GC_515',
                  value = '-(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpl2*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpl2*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_516 = Coupling(name = 'GC_516',
                  value = '(cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) + (c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpl2*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (cpl2*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_517 = Coupling(name = 'GC_517',
                  value = '-(c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl3*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpl3*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (c3pl3*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_518 = Coupling(name = 'GC_518',
                  value = '(cpDC*ee0*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) + (c3pl1*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (c3pl2*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl3*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) - (cll1221*cw0*ee0*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpl3*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (c3pl1*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (c3pl2*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) - (c3pl3*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2) - (cll1221*ee0*complex(0,1)*sw0*vev0**2)/(4.*cw0*Lambda**2) + (cpl3*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_519 = Coupling(name = 'GC_519',
                  value = '(cpmu*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpmu*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_52 = Coupling(name = 'GC_52',
                 value = '(c3pl3*complex(0,1)*cmath.sqrt(2))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_520 = Coupling(name = 'GC_520',
                  value = '(cpta*cw0*ee0*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0) + (cpta*ee0*complex(0,1)*sw0*vev0**2)/(2.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_521 = Coupling(name = 'GC_521',
                  value = '-(cpDC*cw0*ee0**2*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**3) + (cpDC*cw0**3*ee0**2*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**3) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) + (cpDC*cw0*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) + (cpDC*ee0**2*complex(0,1)*sw0*vev0**2)/(8.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_522 = Coupling(name = 'GC_522',
                  value = '(-3*cpWB*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (cpDC*cw0*ee0**2*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**3) + (cpDC*cw0**3*ee0**2*complex(0,1)*vev0**2)/(8.*Lambda**2*sw0**3) + (cpWB*cw0**2*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) - (3*cpDC*ee0**2*complex(0,1)*vev0**2)/(8.*cw0*Lambda**2*sw0) + (c3pl1*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0) + (c3pl2*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0) - (cll1221*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0) - (cpDC*cw0*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0) - (c3pl1*ee0**2*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) + (cll1221*ee0**2*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) + (5*cpDC*ee0**2*complex(0,1)*sw0*vev0**2)/(8.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_523 = Coupling(name = 'GC_523',
                  value = '-((c3pl1*ee0**2*complex(0,1)*vev0**2)/Lambda**2) - (c3pl2*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (cll1221*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (cpDC*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*cw0**2*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2) - (c3pl1*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (c3pl2*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cll1221*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cpDC*cw0**2*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2) - (c3pl1*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) + (cll1221*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) + (cpDC*ee0**2*complex(0,1)*sw0**2*vev0**2)/(4.*cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_524 = Coupling(name = 'GC_524',
                  value = '(c3pl1*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (c3pl2*ee0**2*complex(0,1)*vev0**2)/Lambda**2 - (cll1221*ee0**2*complex(0,1)*vev0**2)/Lambda**2 - (cpDC*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*cw0**2*Lambda**2) + (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2) - (c3pl1*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (c3pl2*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cll1221*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (2*cpWB*cw0*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0) - (2*cpWB*ee0**2*complex(0,1)*sw0*vev0**2)/(cw0*Lambda**2) - (c3pl1*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) + (cll1221*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) + (cpDC*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_525 = Coupling(name = 'GC_525',
                  value = '-((c3pl1*ee0**2*complex(0,1)*vev0**2)/Lambda**2) - (c3pl2*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (2*cdp*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (cll1221*ee0**2*complex(0,1)*vev0**2)/Lambda**2 + (5*cpDC*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*cw0**2*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2) - (c3pl1*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) - (c3pl2*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (cdp*cw0**2*ee0**2*complex(0,1)*vev0**2)/(Lambda**2*sw0**2) + (cll1221*cw0**2*ee0**2*complex(0,1)*vev0**2)/(2.*Lambda**2*sw0**2) + (5*cpDC*cw0**2*ee0**2*complex(0,1)*vev0**2)/(4.*Lambda**2*sw0**2) - (c3pl1*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) + (cdp*ee0**2*complex(0,1)*sw0**2*vev0**2)/(cw0**2*Lambda**2) + (cll1221*ee0**2*complex(0,1)*sw0**2*vev0**2)/(2.*cw0**2*Lambda**2) + (5*cpDC*ee0**2*complex(0,1)*sw0**2*vev0**2)/(4.*cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':2})

GC_526 = Coupling(name = 'GC_526',
                  value = '-(c3pl1*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (c3pl2*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cll1221*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_527 = Coupling(name = 'GC_527',
                  value = '(c3pl1*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) + (c3pl2*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) - (cdp*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (cll1221*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) + (cpDC*ee0**2*complex(0,1)*vev0**3)/(16.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_528 = Coupling(name = 'GC_528',
                  value = '-(c3pl1*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (c3pl2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) + (cdp*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2*sw0**2) + (cll1221*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_529 = Coupling(name = 'GC_529',
                  value = '(c3pl1*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (c3pl2*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cll1221*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0**2)',
                  order = {'NP':2,'QED':1})

GC_53 = Coupling(name = 'GC_53',
                 value = '(cblS3*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_530 = Coupling(name = 'GC_530',
                  value = '(cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0**3) + (cpWB*cw0*ee0**2*vev0**3)/(2.*Lambda**2*sw0**2) + (c3pl1*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (c3pl2*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (cll1221*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_531 = Coupling(name = 'GC_531',
                  value = '(cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) - (cpDC*cw0**2*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) + (cpWB*cw0*ee0**2*vev0**3)/(4.*Lambda**2*sw0**2) - (cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_532 = Coupling(name = 'GC_532',
                  value = '(cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) + (cpDC*cw0**2*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) + (cpWB*cw0*ee0**2*vev0**3)/(4.*Lambda**2*sw0**2) + (c3pl1*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (c3pl2*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (cll1221*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_533 = Coupling(name = 'GC_533',
                  value = '-(cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) + (cpDC*cw0**2*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) - (cpWB*cw0*ee0**2*vev0**3)/(4.*Lambda**2*sw0**2) + (cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_534 = Coupling(name = 'GC_534',
                  value = '-(cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) - (cpDC*cw0**2*ee0**2*vev0**3)/(16.*Lambda**2*sw0**3) - (cpWB*cw0*ee0**2*vev0**3)/(4.*Lambda**2*sw0**2) - (c3pl1*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (c3pl2*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (cll1221*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (cpDC*ee0**2*vev0**3)/(16.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_535 = Coupling(name = 'GC_535',
                  value = '-(cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0**3) - (cpWB*cw0*ee0**2*vev0**3)/(2.*Lambda**2*sw0**2) - (c3pl1*ee0**2*vev0**3)/(4.*Lambda**2*sw0) - (c3pl2*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (cll1221*ee0**2*vev0**3)/(4.*Lambda**2*sw0) + (cpDC*ee0**2*vev0**3)/(8.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_536 = Coupling(name = 'GC_536',
                  value = '-(c3pl1*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (c3pl2*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (cll1221*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (cpDC*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (cpDC*ee0**2*vev0**3)/(8.*cw0*Lambda**2*sw0**2) + (cpDC*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cpWB*ee0**2*vev0**3)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_537 = Coupling(name = 'GC_537',
                  value = '-(c3pl1*ee0**2*vev0**3)/(8.*cw0*Lambda**2) - (c3pl2*ee0**2*vev0**3)/(8.*cw0*Lambda**2) + (cll1221*ee0**2*vev0**3)/(8.*cw0*Lambda**2) - (cpDC*ee0**2*vev0**3)/(16.*cw0*Lambda**2*sw0**2) - (c3pl1*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (c3pl2*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cll1221*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cpWB*ee0**2*vev0**3)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_538 = Coupling(name = 'GC_538',
                  value = '-(c3pl1*ee0**2*vev0**3)/(8.*cw0*Lambda**2) - (c3pl2*ee0**2*vev0**3)/(8.*cw0*Lambda**2) + (cll1221*ee0**2*vev0**3)/(8.*cw0*Lambda**2) - (cpDC*ee0**2*vev0**3)/(16.*cw0*Lambda**2*sw0**2) + (c3pl1*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (c3pl2*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cll1221*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cpDC*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cpWB*ee0**2*vev0**3)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_539 = Coupling(name = 'GC_539',
                  value = '(c3pl1*ee0**2*vev0**3)/(8.*cw0*Lambda**2) + (c3pl2*ee0**2*vev0**3)/(8.*cw0*Lambda**2) - (cll1221*ee0**2*vev0**3)/(8.*cw0*Lambda**2) + (cpDC*ee0**2*vev0**3)/(16.*cw0*Lambda**2*sw0**2) + (c3pl1*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (c3pl2*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (cll1221*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cpWB*ee0**2*vev0**3)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_54 = Coupling(name = 'GC_54',
                 value = '(2*cdp*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_540 = Coupling(name = 'GC_540',
                  value = '(c3pl1*ee0**2*vev0**3)/(8.*cw0*Lambda**2) + (c3pl2*ee0**2*vev0**3)/(8.*cw0*Lambda**2) - (cll1221*ee0**2*vev0**3)/(8.*cw0*Lambda**2) + (cpDC*ee0**2*vev0**3)/(16.*cw0*Lambda**2*sw0**2) - (c3pl1*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) - (c3pl2*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cll1221*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cpDC*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cpWB*ee0**2*vev0**3)/(4.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_541 = Coupling(name = 'GC_541',
                  value = '(c3pl1*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (c3pl2*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (cll1221*ee0**2*vev0**3)/(4.*cw0*Lambda**2) - (cpDC*ee0**2*vev0**3)/(4.*cw0*Lambda**2) + (cpDC*ee0**2*vev0**3)/(8.*cw0*Lambda**2*sw0**2) - (cpDC*cw0*ee0**2*vev0**3)/(8.*Lambda**2*sw0**2) + (cpWB*ee0**2*vev0**3)/(2.*Lambda**2*sw0)',
                  order = {'NP':2,'QED':1})

GC_542 = Coupling(name = 'GC_542',
                  value = '-(cpWB*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2) - (cpDC*cw0*ee0**2*complex(0,1)*vev0**3)/(16.*Lambda**2*sw0**3) + (cpDC*cw0**3*ee0**2*complex(0,1)*vev0**3)/(16.*Lambda**2*sw0**3) - (cpWB*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(16.*cw0*Lambda**2*sw0) + (cpDC*cw0*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0) + (cpDC*ee0**2*complex(0,1)*sw0*vev0**3)/(16.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_543 = Coupling(name = 'GC_543',
                  value = '-(cpDC*cw0*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**3) + (cpDC*cw0**3*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**3) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*cw0*Lambda**2*sw0) + (cpDC*cw0*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0) + (cpDC*ee0**2*complex(0,1)*sw0*vev0**3)/(8.*cw0*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_544 = Coupling(name = 'GC_544',
                  value = '(c3pl1*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2) + (c3pl2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2) - (cdp*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2) - (cll1221*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2) + (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2) + (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*cw0**2*Lambda**2) + (cpDC*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) + (c3pl1*cw0**2*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) + (c3pl2*cw0**2*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) - (cdp*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (cll1221*cw0**2*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) + (cpDC*cw0**2*ee0**2*complex(0,1)*vev0**3)/(16.*Lambda**2*sw0**2) + (cpWB*cw0*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2*sw0) + (cpWB*ee0**2*complex(0,1)*sw0*vev0**3)/(2.*cw0*Lambda**2) + (c3pl1*ee0**2*complex(0,1)*sw0**2*vev0**3)/(8.*cw0**2*Lambda**2) + (c3pl2*ee0**2*complex(0,1)*sw0**2*vev0**3)/(8.*cw0**2*Lambda**2) - (cdp*ee0**2*complex(0,1)*sw0**2*vev0**3)/(4.*cw0**2*Lambda**2) - (cll1221*ee0**2*complex(0,1)*sw0**2*vev0**3)/(8.*cw0**2*Lambda**2) + (cpDC*ee0**2*complex(0,1)*sw0**2*vev0**3)/(16.*cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_545 = Coupling(name = 'GC_545',
                  value = '-(c3pl1*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2) + (cdp*ee0**2*complex(0,1)*vev0**3)/Lambda**2 + (cll1221*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2) + (3*cpDC*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(4.*cw0**2*Lambda**2) - (cpDC*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (c3pl1*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) - (c3pl2*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) + (cdp*cw0**2*ee0**2*complex(0,1)*vev0**3)/(2.*Lambda**2*sw0**2) + (cll1221*cw0**2*ee0**2*complex(0,1)*vev0**3)/(4.*Lambda**2*sw0**2) + (3*cpDC*cw0**2*ee0**2*complex(0,1)*vev0**3)/(8.*Lambda**2*sw0**2) - (c3pl1*ee0**2*complex(0,1)*sw0**2*vev0**3)/(4.*cw0**2*Lambda**2) - (c3pl2*ee0**2*complex(0,1)*sw0**2*vev0**3)/(4.*cw0**2*Lambda**2) + (cdp*ee0**2*complex(0,1)*sw0**2*vev0**3)/(2.*cw0**2*Lambda**2) + (cll1221*ee0**2*complex(0,1)*sw0**2*vev0**3)/(4.*cw0**2*Lambda**2) + (3*cpDC*ee0**2*complex(0,1)*sw0**2*vev0**3)/(8.*cw0**2*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_546 = Coupling(name = 'GC_546',
                  value = '-((complex(0,1)*ymt)/vev0)',
                  order = {'QED':1})

GC_547 = Coupling(name = 'GC_547',
                  value = 'ymt/vev0',
                  order = {'QED':1})

GC_548 = Coupling(name = 'GC_548',
                  value = '-((ymt*cmath.sqrt(2))/vev0)',
                  order = {'QED':1})

GC_549 = Coupling(name = 'GC_549',
                  value = '(ymt*cmath.sqrt(2))/vev0',
                  order = {'QED':1})

GC_55 = Coupling(name = 'GC_55',
                 value = '(4*cdp*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_550 = Coupling(name = 'GC_550',
                  value = '-(c3pl1*vev0*ymt)/(2.*Lambda**2) - (c3pl2*vev0*ymt)/(2.*Lambda**2) + (cll1221*vev0*ymt)/(2.*Lambda**2) - (cpDC*vev0*ymt)/(4.*Lambda**2)',
                  order = {'NP':2,'QED':1})

GC_56 = Coupling(name = 'GC_56',
                 value = '(6*cp*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_57 = Coupling(name = 'GC_57',
                 value = '(12*cp*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_58 = Coupling(name = 'GC_58',
                 value = '(18*cp*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_59 = Coupling(name = 'GC_59',
                 value = '(36*cp*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_6 = Coupling(name = 'GC_6',
                value = '2*ee0**2*complex(0,1)',
                order = {'QED':2})

GC_60 = Coupling(name = 'GC_60',
                 value = '(90*cp*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':3})

GC_61 = Coupling(name = 'GC_61',
                 value = '-cpDC/(2.*Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_62 = Coupling(name = 'GC_62',
                 value = '-((cpDC*complex(0,1))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_63 = Coupling(name = 'GC_63',
                 value = '-((cpe*complex(0,1))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_64 = Coupling(name = 'GC_64',
                 value = 'cpe/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_65 = Coupling(name = 'GC_65',
                 value = '(4*cpG*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_66 = Coupling(name = 'GC_66',
                 value = '-((cpmu*complex(0,1))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_67 = Coupling(name = 'GC_67',
                 value = 'cpmu/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_68 = Coupling(name = 'GC_68',
                 value = '-((cpta*complex(0,1))/Lambda**2)',
                 order = {'NP':2,'QED':2})

GC_69 = Coupling(name = 'GC_69',
                 value = 'cpta/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_7 = Coupling(name = 'GC_7',
                value = '-ee0**2/(2.*cw0)',
                order = {'QED':2})

GC_70 = Coupling(name = 'GC_70',
                 value = '(4*cpW*complex(0,1))/Lambda**2',
                 order = {'NP':2,'QED':2})

GC_71 = Coupling(name = 'GC_71',
                 value = '(cQd1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_72 = Coupling(name = 'GC_72',
                 value = '(cQd8*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_73 = Coupling(name = 'GC_73',
                 value = '(cQe1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_74 = Coupling(name = 'GC_74',
                 value = '(cQe2*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_75 = Coupling(name = 'GC_75',
                 value = '(cQe3*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_76 = Coupling(name = 'GC_76',
                 value = '(2*cQl31*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_77 = Coupling(name = 'GC_77',
                 value = '(2*cQl32*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_78 = Coupling(name = 'GC_78',
                 value = '(2*cQl33*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_79 = Coupling(name = 'GC_79',
                 value = '(cQlM1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_8 = Coupling(name = 'GC_8',
                value = '-(ee0**2*complex(0,1))/(2.*cw0)',
                order = {'QED':2})

GC_80 = Coupling(name = 'GC_80',
                 value = '(cQlM2*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_81 = Coupling(name = 'GC_81',
                 value = '(cQlM3*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_82 = Coupling(name = 'GC_82',
                 value = '(2*cQq13*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_83 = Coupling(name = 'GC_83',
                 value = '(cQQ8*complex(0,1))/(2.*Lambda**2)',
                 order = {'NP':2})

GC_84 = Coupling(name = 'GC_84',
                 value = '(2*cQq83*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_85 = Coupling(name = 'GC_85',
                 value = '(cQt1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_86 = Coupling(name = 'GC_86',
                 value = '(cQt8*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_87 = Coupling(name = 'GC_87',
                 value = '(cQu1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_88 = Coupling(name = 'GC_88',
                 value = '(cQu8*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_89 = Coupling(name = 'GC_89',
                 value = '(ctd1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_9 = Coupling(name = 'GC_9',
                value = 'ee0**2/(2.*cw0)',
                order = {'QED':2})

GC_90 = Coupling(name = 'GC_90',
                 value = '(ctd8*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_91 = Coupling(name = 'GC_91',
                 value = '(cte1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_92 = Coupling(name = 'GC_92',
                 value = '(cte2*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_93 = Coupling(name = 'GC_93',
                 value = '(cte3*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_94 = Coupling(name = 'GC_94',
                 value = '(ctl1*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_95 = Coupling(name = 'GC_95',
                 value = '(ctl2*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_96 = Coupling(name = 'GC_96',
                 value = '(ctl3*complex(0,1))/Lambda**2',
                 order = {'NP':2})

GC_97 = Coupling(name = 'GC_97',
                 value = '-(ctlT3*complex(0,1))/(4.*Lambda**2)',
                 order = {'NP':2})

GC_98 = Coupling(name = 'GC_98',
                 value = '(ctlT3*complex(0,1))/(4.*Lambda**2)',
                 order = {'NP':2})

GC_99 = Coupling(name = 'GC_99',
                 value = '-(ctlT3*complex(0,1))/(2.*Lambda**2)',
                 order = {'NP':2})

