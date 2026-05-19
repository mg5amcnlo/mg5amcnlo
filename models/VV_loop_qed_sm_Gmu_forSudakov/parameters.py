# This file was automatically created by FeynRules $Revision: 999 $
# Mathematica version: 7.0 for Linux x86 (64-bit) (February 18, 2009)
# Date: Mon 30 Jan 2012 19:57:04



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec


# Input SM couplings
Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.0000116637,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1184,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 172.0,
                texname = 'y_{m_t}',
                lhablock = 'SMINPUTS',
                lhacode = [ 6 ])


MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 80.419,
               texname = 'm_W',
               lhablock = 'MASS',
               lhacode = [ 24 ])


MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = 'm_Z',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172,
               texname = 'm_T',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 120,
               texname = 'm_H',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4952,
               texname = '\\Gamma _Z',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.085,
               texname = '\\Gamma _W',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\Gamma _t',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00575308848,
               texname = '\\Gamma _H',
               lhablock = 'DECAY',
               lhacode = [ 25 ])




# For Sudakov

ntadpole = Parameter(name = 'ntadpole',
                      nature = 'external',
                      type = 'real',
                      value = '1.0',
                      texname = 'tadpole',
                      lhablock = 'TADPOLE',
                      lhacode = [ 1 ])                  
                    
                    
# SMEFT                  
                     
Lambda = Parameter(name = 'Lambda',
                   nature = 'external',
                   type = 'real',
                   value = 1000,
                   texname = '\\Lambda',
                   lhablock = 'DIM6',
                   lhacode = [ 1 ])

# Rescale SM tt~g, tt~h, tt~a, tt~z vertices,
# can set = 0 to get tt~ with SMEFT only

"""
SM_tag = Parameter(name = 'SM',
                   nature = 'external',
                   type = 'real',
                   value = 1.0,
                   texname = 'SM',
                   lhablock = 'DIM6',
                   lhacode = [ 2 ])
"""
                                                
# SMEFT di-boson

"""
cpDC = Parameter(name = 'cpDC',
                 nature = 'external',
                 type = 'real',
                 value = 1.0e-1,
                 texname = 'c_{\\text{$\\phi $D}}',
                 lhablock = 'DIM6',
                 lhacode = [ 3 ])
"""

cpWB = Parameter(name = 'cpWB',
                 nature = 'external',
                 type = 'real',
                 value = 1.0e-1,
                 texname = 'c_{\\text{$\\phi $WB}}',
                 lhablock = 'DIM6',
                 lhacode = [ 4 ])

"""
cdp = Parameter(name = 'cdp',
                nature = 'external',
                type = 'real',
                value = 1.0e-1,
                texname = 'c_{\\text{d$\\phi $}}',
                lhablock = 'DIM6',
                lhacode = [ 5 ])
"""

"""
cp = Parameter(name = 'cp',
               nature = 'external',
               type = 'real',
               value = 1.0e-1,
               texname = 'c_{\\phi }',
               lhablock = 'DIM6',
               lhacode = [ 6 ])
"""


cWWW = Parameter(name = 'cWWW',
                 nature = 'external',
                 type = 'real',
                 value = 1.0e-1,
                 texname = 'c_W',
                 lhablock = 'DIM6',
                 lhacode = [ 7 ])

cG = Parameter(name = 'cG',
               nature = 'external',
               type = 'real',
               value = 1.0e-1,
               texname = 'c_G',
               lhablock = 'DIM6',
               lhacode = [ 8 ])

cpG = Parameter(name = 'cpG',
                nature = 'external',
                type = 'real',
                value = 1.0e-1,
                texname = 'c_{\\text{$\\phi $G}}',
                lhablock = 'DIM6',
                lhacode = [ 9 ])

cpW = Parameter(name = 'cpW',
                nature = 'external',
                type = 'real',
                value = 1.0e-1,
                texname = 'c_{\\text{$\\phi $W}}',
                lhablock = 'DIM6',
                lhacode = [ 10 ])

cpBB = Parameter(name = 'cpBB',
                 nature = 'external',
                 type = 'real',
                 value = 1.0e-1,
                 texname = 'c_{\\text{$\\phi $B}}',
                 lhablock = 'DIM6',
                 lhacode = [ 11 ])

gset0= Parameter(name = 'gset0',
                 nature = 'external',
                 type = 'real',
                 value = 1.0e-1,
                 texname = 'c_{\\text{$\\phi $B}}',
                 lhablock = 'DIM6',
                 lhacode = [ 20 ])


cpqMi = Parameter(name = 'cpqMi',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[c,\\text{$\\phi $q},\\text{(-)}]',
                  lhablock = 'DIM62F',
                  lhacode = [ 10 ])



cpq3i = Parameter(name = 'cpq3i',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[c,\\text{$\\phi $q},\\text{(3)}]',
                  lhablock = 'DIM62F',
                  lhacode = [ 11 ])

cpQ3 = Parameter(name = 'cpQ3',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[c,\\text{$\\phi $Q},\\text{(3)}]',
                 lhablock = 'DIM62F',
                 lhacode = [ 12 ])

cpQM = Parameter(name = 'cpQM',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[c,\\text{$\\phi $Q},\\text{(-)}]',
                 lhablock = 'DIM62F',
                 lhacode = [ 13 ])

cpu = Parameter(name = 'cpu',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $u}}',
                lhablock = 'DIM62F',
                lhacode = [ 14 ])


cpt = Parameter(name = 'cpt',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $t}}',
                lhablock = 'DIM62F',
                lhacode = [ 15 ])


cpd = Parameter(name = 'cpd',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $d}}',
                lhablock = 'DIM62F',
                lhacode = [ 16 ])

ctp = Parameter(name = 'ctp',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{t$\\phi $}}',
                lhablock = 'DIM62F',
                lhacode = [ 19 ])



ctZ = Parameter(name = 'ctZ',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{tZ}}',
                lhablock = 'DIM62F',
                lhacode = [ 22 ])


ctW = Parameter(name = 'ctW',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{tW}}',
                lhablock = 'DIM62F',
                lhacode = [ 23 ])





# ---------------------------------------------------------------
# 
#  Internal parameters
#
# ---------------------------------------------------------------

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')


# Other internal parameters
sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - MW**2/MZ**2',
                texname = '\\text{sw2}')


Gfbar = Parameter(name = 'Gfbar',
                  nature = 'internal',
                  type = 'real',
                  value = '(Gf*abs((MW**2*(MZ**2-MW**2))/MZ**2)*(MZ**2/(MW**2*(MZ**2-MW**2))))',
                  texname = '\\bar{G_f}')


# We use Gfbar here which ensures that aEW will be real
aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(2.)*Gfbar*MW**2*(1-MW**2/MZ**2)/cmath.pi',
                texname = '\\alpha _{\\text{EW}}')


aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'internal',
                  type = 'real',
                  value = '1./aEW',
                  texname = '\\text{aEWM1}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - sw2)',
               texname = 'c_w')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(sw2)',
               texname = 's_w')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = 'ee/cw',
               texname = 'g_1')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sw',
               texname = 'g_w')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(2*MW*sw)/ee',
                texname = 'vev')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = 'MH**2/(2.*vev**2)',
                texname = '\\text{lam}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/vev',
               texname = '\\text{yt}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*vev**2)',
                texname = '\\mu ')


# SMEFT di-boson

vev0 = Parameter(name = 'vev0',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1/Gf)/2**0.25',
                 texname = '\\text{vev0}')


ee0 = Parameter(name = 'ee0',
                nature = 'internal',
                type = 'real',
                value = '(2*MW*sw0)/vev0',
                texname = 'e_0')


cw0 = Parameter(name = 'cw0',
                nature = 'internal',
                type = 'real',
                value = 'MW/MZ',
                texname = '\\text{Subsuperscript}[c,w,0]')

muH0 = Parameter(name = 'muH0',
                 nature = 'internal',
                 type = 'real',
                 value = 'MH/cmath.sqrt(2)',
                 texname = '\\mu _0')

sw0 = Parameter(name = 'sw0',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(1 - MW**2/MZ**2)',
                texname = '\\text{Subsuperscript}[s,w,0]')
            
