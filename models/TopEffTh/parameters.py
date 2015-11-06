# This file was automatically created by FeynRules 2.4.1
# Mathematica version: 10.1.0  for Mac OS X x86 (64-bit) (March 24, 2015)
# Date: Thu 18 Jun 2015 16:26:34



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
Lambda = Parameter(name = 'Lambda',
                   nature = 'external',
                   type = 'real',
                   value = 1000,
                   texname = '\\Lambda',
                   lhablock = 'DIM6',
                   lhacode = [ 1 ])

RC3phiq = Parameter(name = 'RC3phiq',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{Subsuperscript}[\\text{RC},\\text{$\\phi $q},\\text{(3)}]',
                    lhablock = 'DIM6',
                    lhacode = [ 2 ])

IC3phiq = Parameter(name = 'IC3phiq',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{Subsuperscript}[\\text{IC},\\text{$\\phi $q},\\text{(3)}]',
                    lhablock = 'DIM6',
                    lhacode = [ 3 ])

RCtW = Parameter(name = 'RCtW',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{tW}}',
                 lhablock = 'DIM6',
                 lhacode = [ 4 ])

ICtW = Parameter(name = 'ICtW',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{tW}}',
                 lhablock = 'DIM6',
                 lhacode = [ 5 ])

RCtG = Parameter(name = 'RCtG',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{RC}_{\\text{tG}}',
                 lhablock = 'DIM6',
                 lhacode = [ 6 ])

ICtG = Parameter(name = 'ICtG',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{IC}_{\\text{tG}}',
                 lhablock = 'DIM6',
                 lhacode = [ 7 ])

CG = Parameter(name = 'CG',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = 'C_G',
               lhablock = 'DIM6',
               lhacode = [ 8 ])

CphiG = Parameter(name = 'CphiG',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = 'C_{\\text{$\\phi $G}}',
                  lhablock = 'DIM6',
                  lhacode = [ 9 ])

C13qq = Parameter(name = 'C13qq',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[C,\\text{qq},\\text{(1,3)}]',
                  lhablock = 'FourFermion',
                  lhacode = [ 1 ])

C81qq = Parameter(name = 'C81qq',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[C,\\text{qq},\\text{(8,1)}]',
                  lhablock = 'FourFermion',
                  lhacode = [ 2 ])

C83qq = Parameter(name = 'C83qq',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[C,\\text{qq},\\text{(8,3)}]',
                  lhablock = 'FourFermion',
                  lhacode = [ 3 ])

C8ut = Parameter(name = 'C8ut',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[C,\\text{ut},\\text{(8)}]',
                 lhablock = 'FourFermion',
                 lhacode = [ 4 ])

C8dt = Parameter(name = 'C8dt',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[C,\\text{dt},\\text{(8)}]',
                 lhablock = 'FourFermion',
                 lhacode = [ 5 ])

C1qu = Parameter(name = 'C1qu',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[C,\\text{qu},\\text{(1)}]',
                 lhablock = 'FourFermion',
                 lhacode = [ 6 ])

C1qd = Parameter(name = 'C1qd',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[C,\\text{qd},\\text{(1)}]',
                 lhablock = 'FourFermion',
                 lhacode = [ 7 ])

C1qt = Parameter(name = 'C1qt',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[C,\\text{qt},\\text{(1)}]',
                 lhablock = 'FourFermion',
                 lhacode = [ 8 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.9,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

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

ymb = Parameter(name = 'ymb',
                nature = 'external',
                type = 'real',
                value = 4.7,
                texname = '\\text{ymb}',
                lhablock = 'YUKAWA',
                lhacode = [ 5 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 172,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MTA = Parameter(name = 'MTA',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{MTA}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.7,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4952,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.085,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00407,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\alpha _{\\text{EW}}')

C3phiq = Parameter(name = 'C3phiq',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*IC3phiq + RC3phiq',
                   texname = '\\text{Subsuperscript}[C,\\text{$\\phi $q},\\text{(3)}]')

CtG = Parameter(name = 'CtG',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICtG + RCtG',
                texname = 'C_{\\text{tG}}')

CtW = Parameter(name = 'CtW',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICtW + RCtW',
                texname = 'C_{\\text{tW}}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
               texname = 'M_W')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - MW**2/MZ**2',
                texname = '\\text{sw2}')

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
                texname = '\\text{vev}')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = 'MH**2/(2.*vev**2)',
                texname = '\\text{lam}')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/vev',
               texname = '\\text{yb}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/vev',
               texname = '\\text{yt}')

ytau = Parameter(name = 'ytau',
                 nature = 'internal',
                 type = 'real',
                 value = '(ymtau*cmath.sqrt(2))/vev',
                 texname = '\\text{ytau}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*vev**2)',
                texname = '\\mu')

