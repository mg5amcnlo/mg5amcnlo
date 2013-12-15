# This file was automatically created by FeynRules $Revision: 821 $
# Mathematica version: 7.0 for Microsoft Windows (32-bit) (February 18, 2009)
# Date: Mon 3 Oct 2011 13:27:06



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
K = Parameter(name = 'K',
              nature = 'external',
              type = 'real',
              value = 100,
              texname = 'K',
              lhablock = 'APPROX',
              lhacode = [ 1 ])

K1 = Parameter(name = 'K1',
               nature = 'external',
               type = 'real',
               value = 100,
               texname = '\\text{K1}',
               lhablock = 'APPROX',
               lhacode = [ 2 ])

K2 = Parameter(name = 'K2',
               nature = 'external',
               type = 'real',
               value = 100,
               texname = '\\text{K2}',
               lhablock = 'APPROX',
               lhacode = [ 3 ])

K3 = Parameter(name = 'K3',
               nature = 'external',
               type = 'real',
               value = 100,
               texname = '\\text{K3}',
               lhablock = 'APPROX',
               lhacode = [ 4 ])

cabi = Parameter(name = 'cabi',
                 nature = 'external',
                 type = 'real',
                 value = 0.227736,
                 texname = '\\theta _c',
                 lhablock = 'CKMBLOCK',
                 lhacode = [ 1 ])

Lambda = Parameter(name = 'Lambda',
                   nature = 'external',
                   type = 'real',
                   value = 1000,
                   texname = '\\Lambda ',
                   lhablock = 'DIM6',
                   lhacode = [ 1 ])

RC3phiq = Parameter(name = 'RC3phiq',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{RC}_{\\text{$\\phi $q}}^{\\text{(3)}}',
                    lhablock = 'DIM6',
                    lhacode = [ 2 ])

IC3phiq = Parameter(name = 'IC3phiq',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{IC}_{\\text{$\\phi $q}}^{\\text{(3)}}',
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
                  texname = 'C_{\\text{qq}}^{\\text{(1,3)}}',
                  lhablock = 'FourFermion',
                  lhacode = [ 1 ])

C81qq = Parameter(name = 'C81qq',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = 'C_{\\text{qq}}^{\\text{(8,1)}}',
                  lhablock = 'FourFermion',
                  lhacode = [ 2 ])

C83qq = Parameter(name = 'C83qq',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = 'C_{\\text{qq}}^{\\text{(8,3)}}',
                  lhablock = 'FourFermion',
                  lhacode = [ 3 ])

C8ut = Parameter(name = 'C8ut',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'C_{\\text{ut}}^{\\text{(8)}}',
                 lhablock = 'FourFermion',
                 lhacode = [ 4 ])

C8dt = Parameter(name = 'C8dt',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'C_{\\text{dt}}^{\\text{(8)}}',
                 lhablock = 'FourFermion',
                 lhacode = [ 5 ])

C1qu = Parameter(name = 'C1qu',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'C_{\\text{qu}}^{\\text{(1)}}',
                 lhablock = 'FourFermion',
                 lhacode = [ 6 ])

C1qd = Parameter(name = 'C1qd',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'C_{\\text{qd}}^{\\text{(1)}}',
                 lhablock = 'FourFermion',
                 lhacode = [ 7 ])

C1qt = Parameter(name = 'C1qt',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'C_{\\text{qt}}^{\\text{(1)}}',
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
               texname = '\\text{aS}',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

ymdo = Parameter(name = 'ymdo',
                 nature = 'external',
                 type = 'real',
                 value = 0.00504,
                 texname = '\\text{ymdo}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 1 ])

ymup = Parameter(name = 'ymup',
                 nature = 'external',
                 type = 'real',
                 value = 0.0025499999999999997,
                 texname = '\\text{ymup}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 2 ])

yms = Parameter(name = 'yms',
                nature = 'external',
                type = 'real',
                value = 0.101,
                texname = '\\text{yms}',
                lhablock = 'YUKAWA',
                lhacode = [ 3 ])

ymc = Parameter(name = 'ymc',
                nature = 'external',
                type = 'real',
                value = 1.27,
                texname = '\\text{ymc}',
                lhablock = 'YUKAWA',
                lhacode = [ 4 ])

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
                value = 172.,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

yme = Parameter(name = 'yme',
                nature = 'external',
                type = 'real',
                value = 0.0005110000000000001,
                texname = '\\text{yme}',
                lhablock = 'YUKAWA',
                lhacode = [ 11 ])

ymm = Parameter(name = 'ymm',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{ymm}',
                lhablock = 'YUKAWA',
                lhacode = [ 13 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.0005110000000000001,
               texname = '\\text{Me}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

MM = Parameter(name = 'MM',
               nature = 'external',
               type = 'real',
               value = 0.10566,
               texname = '\\text{MM}',
               lhablock = 'MASS',
               lhacode = [ 13 ])

MTA = Parameter(name = 'MTA',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{MTA}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.0025499999999999997,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 1.42,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 4 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.00504,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 1 ])

MS = Parameter(name = 'MS',
               nature = 'external',
               type = 'real',
               value = 0.101,
               texname = '\\text{MS}',
               lhablock = 'MASS',
               lhacode = [ 3 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.7,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

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

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 6.38233934e-03,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\text{aEW}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

CKM11 = Parameter(name = 'CKM11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.cos(cabi)',
                  texname = '\\text{CKM11}')

CKM12 = Parameter(name = 'CKM12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.sin(cabi)',
                  texname = '\\text{CKM12}')

CKM13 = Parameter(name = 'CKM13',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{CKM13}')

CKM21 = Parameter(name = 'CKM21',
                  nature = 'internal',
                  type = 'complex',
                  value = '-cmath.sin(cabi)',
                  texname = '\\text{CKM21}')

CKM22 = Parameter(name = 'CKM22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.cos(cabi)',
                  texname = '\\text{CKM22}')

CKM23 = Parameter(name = 'CKM23',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{CKM23}')

CKM31 = Parameter(name = 'CKM31',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{CKM31}')

CKM32 = Parameter(name = 'CKM32',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{CKM32}')

CKM33 = Parameter(name = 'CKM33',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{CKM33}')

C3phiq = Parameter(name = 'C3phiq',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*IC3phiq + RC3phiq',
                   texname = 'C_{\\text{$\\phi $q}}^{\\text{(3)}}')

CtW = Parameter(name = 'CtW',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICtW + RCtW',
                texname = 'C_{\\text{tW}}')

CtG = Parameter(name = 'CtG',
                nature = 'internal',
                type = 'complex',
                value = 'complex(0,1)*ICtG + RCtG',
                texname = 'C_{\\text{tG}}')

MTri = Parameter(name = 'MTri',
                 nature = 'internal',
                 type = 'real',
                 value = 'K*Lambda',
                 texname = 'M_{\\text{Tri}}')

MTri8 = Parameter(name = 'MTri8',
                  nature = 'internal',
                  type = 'real',
                  value = 'K1*Lambda',
                  texname = 'M_{\\text{Tri8}}')

M8t = Parameter(name = 'M8t',
                nature = 'internal',
                type = 'real',
                value = 'K2*Lambda',
                texname = 'M_{8 t}')

M8Q = Parameter(name = 'M8Q',
                nature = 'internal',
                type = 'real',
                value = 'K3*Lambda',
                texname = 'M_{8 Q}')

gT = Parameter(name = 'gT',
               nature = 'internal',
               type = 'real',
               value = '-(K*cmath.sqrt(C13qq))',
               texname = 'g_T')

gTl = Parameter(name = 'gTl',
                nature = 'internal',
                type = 'real',
                value = 'K*cmath.sqrt(C13qq)',
                texname = 'g_{\\text{Tl}}')

gT8 = Parameter(name = 'gT8',
                nature = 'internal',
                type = 'real',
                value = '-(K1*cmath.sqrt(C83qq))',
                texname = 'g_{\\text{T8}}')

gT8l = Parameter(name = 'gT8l',
                 nature = 'internal',
                 type = 'real',
                 value = 'K1*cmath.sqrt(C83qq)',
                 texname = 'g_{\\text{T8l}}')

g8t = Parameter(name = 'g8t',
                nature = 'internal',
                type = 'real',
                value = '-K2',
                texname = 'g_{8 t}')

g8tu = Parameter(name = 'g8tu',
                 nature = 'internal',
                 type = 'real',
                 value = 'C8ut*K2',
                 texname = 'g_{8 \\text{tu}}')

g8td = Parameter(name = 'g8td',
                 nature = 'internal',
                 type = 'real',
                 value = 'C8dt*K2',
                 texname = 'g_{8 \\text{td}}')

g8tq = Parameter(name = 'g8tq',
                 nature = 'internal',
                 type = 'real',
                 value = 'C1qt*K2',
                 texname = 'g_{8 \\text{tq}}')

g8Q = Parameter(name = 'g8Q',
                nature = 'internal',
                type = 'real',
                value = '-K3',
                texname = 'g_{8 Q}')

g8Qu = Parameter(name = 'g8Qu',
                 nature = 'internal',
                 type = 'real',
                 value = 'C1qu*K3',
                 texname = 'g_{8 \\text{Qu}}')

g8Qd = Parameter(name = 'g8Qd',
                 nature = 'internal',
                 type = 'real',
                 value = 'C1qd*K3',
                 texname = 'g_{8 \\text{Qd}}')

g8Qq = Parameter(name = 'g8Qq',
                 nature = 'internal',
                 type = 'real',
                 value = 'C81qq*K3',
                 texname = 'g_{8 \\text{Qq}}')

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

v = Parameter(name = 'v',
              nature = 'internal',
              type = 'real',
              value = '(2*MW*sw)/ee',
              texname = 'v')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = 'MH**2/(2.*v**2)',
                texname = '\\text{lam}')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/v',
               texname = '\\text{yb}')

yc = Parameter(name = 'yc',
               nature = 'internal',
               type = 'real',
               value = '(ymc*cmath.sqrt(2))/v',
               texname = '\\text{yc}')

ydo = Parameter(name = 'ydo',
                nature = 'internal',
                type = 'real',
                value = '(ymdo*cmath.sqrt(2))/v',
                texname = '\\text{ydo}')

ye = Parameter(name = 'ye',
               nature = 'internal',
               type = 'real',
               value = '(yme*cmath.sqrt(2))/v',
               texname = '\\text{ye}')

ym = Parameter(name = 'ym',
               nature = 'internal',
               type = 'real',
               value = '(ymm*cmath.sqrt(2))/v',
               texname = '\\text{ym}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '(yms*cmath.sqrt(2))/v',
               texname = '\\text{ys}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/v',
               texname = '\\text{yt}')

ytau = Parameter(name = 'ytau',
                 nature = 'internal',
                 type = 'real',
                 value = '(ymtau*cmath.sqrt(2))/v',
                 texname = '\\text{ytau}')

yup = Parameter(name = 'yup',
                nature = 'internal',
                type = 'real',
                value = '(ymup*cmath.sqrt(2))/v',
                texname = '\\text{yup}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*v**2)',
                texname = '\\mu ')

