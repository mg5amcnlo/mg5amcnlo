# This file was automatically created by FeynRules $Revision: 999 $
# Mathematica version: 7.0 for Linux x86 (64-bit) (February 18, 2009)
# Date: Mon 30 Jan 2012 19:57:04



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
cabi = Parameter(name = 'cabi',
                 nature = 'external',
                 type = 'real',
                 value = 0.227736,
                 texname = '\\theta _c',
                 lhablock = 'CKMBLOCK',
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
                 value = 0.00255,
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
                value = 172,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

yme = Parameter(name = 'yme',
                nature = 'external',
                type = 'real',
                value = 0.000511,
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

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])
	       
MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 80.48067023374448,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [24])	       

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.000511,
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
               value = 0.00255,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 1.27,
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

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 120,
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
               value = 0.00575308848,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])
	       
aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.pi * MZ**2 / ((MZ**2 - MW**2)*cmath.sqrt(2) * Gf * MW**2) ',
                  texname = '\\text{aEWM1}'
                  )	       

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\alpha _{\\text{EW}}')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = '1',
               texname = 'g_w')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = '1',
               texname = 'g_1')

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

yc = Parameter(name = 'yc',
               nature = 'internal',
               type = 'real',
               value = '(ymc*cmath.sqrt(2))/vev',
               texname = '\\text{yc}')

ydo = Parameter(name = 'ydo',
                nature = 'internal',
                type = 'real',
                value = '(ymdo*cmath.sqrt(2))/vev',
                texname = '\\text{ydo}')

ye = Parameter(name = 'ye',
               nature = 'internal',
               type = 'real',
               value = '(yme*cmath.sqrt(2))/vev',
               texname = '\\text{ye}')

ym = Parameter(name = 'ym',
               nature = 'internal',
               type = 'real',
               value = '(ymm*cmath.sqrt(2))/vev',
               texname = '\\text{ym}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '(yms*cmath.sqrt(2))/vev',
               texname = '\\text{ys}')

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

yup = Parameter(name = 'yup',
                nature = 'internal',
                type = 'real',
                value = '(ymup*cmath.sqrt(2))/vev',
                texname = '\\text{yup}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*vev**2)',
                texname = '\\mu ')

I111 = Parameter(name = 'I111',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ydo*complexconjugate(CKM11)',
                 texname = '\\text{I111}')

I112 = Parameter(name = 'I112',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ydo*complexconjugate(CKM21)',
                 texname = '\\text{I112}')

I113 = Parameter(name = 'I113',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ydo*complexconjugate(CKM31)',
                 texname = '\\text{I113}')

I121 = Parameter(name = 'I121',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ys*complexconjugate(CKM12)',
                 texname = '\\text{I121}')

I122 = Parameter(name = 'I122',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ys*complexconjugate(CKM22)',
                 texname = '\\text{I122}')

I123 = Parameter(name = 'I123',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ys*complexconjugate(CKM32)',
                 texname = '\\text{I123}')

I131 = Parameter(name = 'I131',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yb*complexconjugate(CKM13)',
                 texname = '\\text{I131}')

I132 = Parameter(name = 'I132',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yb*complexconjugate(CKM23)',
                 texname = '\\text{I132}')

I133 = Parameter(name = 'I133',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yb*complexconjugate(CKM33)',
                 texname = '\\text{I133}')

I211 = Parameter(name = 'I211',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yup*complexconjugate(CKM11)',
                 texname = '\\text{I211}')

I212 = Parameter(name = 'I212',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yc*complexconjugate(CKM21)',
                 texname = '\\text{I212}')

I213 = Parameter(name = 'I213',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yt*complexconjugate(CKM31)',
                 texname = '\\text{I213}')

I221 = Parameter(name = 'I221',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yup*complexconjugate(CKM12)',
                 texname = '\\text{I221}')

I222 = Parameter(name = 'I222',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yc*complexconjugate(CKM22)',
                 texname = '\\text{I222}')

I223 = Parameter(name = 'I223',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yt*complexconjugate(CKM32)',
                 texname = '\\text{I223}')

I231 = Parameter(name = 'I231',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yup*complexconjugate(CKM13)',
                 texname = '\\text{I231}')

I232 = Parameter(name = 'I232',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yc*complexconjugate(CKM23)',
                 texname = '\\text{I232}')

I233 = Parameter(name = 'I233',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yt*complexconjugate(CKM33)',
                 texname = '\\text{I233}')

I311 = Parameter(name = 'I311',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM11*yup',
                 texname = '\\text{I311}')

I312 = Parameter(name = 'I312',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM12*yup',
                 texname = '\\text{I312}')

I313 = Parameter(name = 'I313',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM13*yup',
                 texname = '\\text{I313}')

I321 = Parameter(name = 'I321',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM21*yc',
                 texname = '\\text{I321}')

I322 = Parameter(name = 'I322',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM22*yc',
                 texname = '\\text{I322}')

I323 = Parameter(name = 'I323',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM23*yc',
                 texname = '\\text{I323}')

I331 = Parameter(name = 'I331',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM31*yt',
                 texname = '\\text{I331}')

I332 = Parameter(name = 'I332',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM32*yt',
                 texname = '\\text{I332}')

I333 = Parameter(name = 'I333',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM33*yt',
                 texname = '\\text{I333}')

I411 = Parameter(name = 'I411',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM11*ydo',
                 texname = '\\text{I411}')

I412 = Parameter(name = 'I412',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM12*ys',
                 texname = '\\text{I412}')

I413 = Parameter(name = 'I413',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM13*yb',
                 texname = '\\text{I413}')

I421 = Parameter(name = 'I421',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM21*ydo',
                 texname = '\\text{I421}')

I422 = Parameter(name = 'I422',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM22*ys',
                 texname = '\\text{I422}')

I423 = Parameter(name = 'I423',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM23*yb',
                 texname = '\\text{I423}')

I431 = Parameter(name = 'I431',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM31*ydo',
                 texname = '\\text{I431}')

I432 = Parameter(name = 'I432',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM32*ys',
                 texname = '\\text{I432}')

I433 = Parameter(name = 'I433',
                 nature = 'internal',
                 type = 'complex',
                 value = 'CKM33*yb',
                 texname = '\\text{I433}')

