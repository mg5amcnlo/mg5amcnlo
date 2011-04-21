# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Wed 23 Mar 2011 22:53:59



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
ICKM11 = Parameter(name = 'ICKM11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM11}',
                   lhablock = 'ICKM',
                   lhacode = [ 1, 1 ])

ICKM12 = Parameter(name = 'ICKM12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM12}',
                   lhablock = 'ICKM',
                   lhacode = [ 1, 2 ])

ICKM13 = Parameter(name = 'ICKM13',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM13}',
                   lhablock = 'ICKM',
                   lhacode = [ 1, 3 ])

ICKM14 = Parameter(name = 'ICKM14',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM14}',
                   lhablock = 'ICKM',
                   lhacode = [ 1, 4 ])

ICKM21 = Parameter(name = 'ICKM21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM21}',
                   lhablock = 'ICKM',
                   lhacode = [ 2, 1 ])

ICKM22 = Parameter(name = 'ICKM22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM22}',
                   lhablock = 'ICKM',
                   lhacode = [ 2, 2 ])

ICKM23 = Parameter(name = 'ICKM23',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM23}',
                   lhablock = 'ICKM',
                   lhacode = [ 2, 3 ])

ICKM24 = Parameter(name = 'ICKM24',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM24}',
                   lhablock = 'ICKM',
                   lhacode = [ 2, 4 ])

ICKM31 = Parameter(name = 'ICKM31',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM31}',
                   lhablock = 'ICKM',
                   lhacode = [ 3, 1 ])

ICKM32 = Parameter(name = 'ICKM32',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM32}',
                   lhablock = 'ICKM',
                   lhacode = [ 3, 2 ])

ICKM33 = Parameter(name = 'ICKM33',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM33}',
                   lhablock = 'ICKM',
                   lhacode = [ 3, 3 ])

ICKM34 = Parameter(name = 'ICKM34',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM34}',
                   lhablock = 'ICKM',
                   lhacode = [ 3, 4 ])

ICKM41 = Parameter(name = 'ICKM41',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM41}',
                   lhablock = 'ICKM',
                   lhacode = [ 4, 1 ])

ICKM42 = Parameter(name = 'ICKM42',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM42}',
                   lhablock = 'ICKM',
                   lhacode = [ 4, 2 ])

ICKM43 = Parameter(name = 'ICKM43',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM43}',
                   lhablock = 'ICKM',
                   lhacode = [ 4, 3 ])

ICKM44 = Parameter(name = 'ICKM44',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{ICKM44}',
                   lhablock = 'ICKM',
                   lhacode = [ 4, 4 ])

RCKM11 = Parameter(name = 'RCKM11',
                   nature = 'external',
                   type = 'real',
                   value = 1,
                   texname = '\\text{RCKM11}',
                   lhablock = 'RCKM',
                   lhacode = [ 1, 1 ])

RCKM12 = Parameter(name = 'RCKM12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{RCKM12}',
                   lhablock = 'RCKM',
                   lhacode = [ 1, 2 ])

RCKM13 = Parameter(name = 'RCKM13',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{RCKM13}',
                   lhablock = 'RCKM',
                   lhacode = [ 1, 3 ])

RCKM14 = Parameter(name = 'RCKM14',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{RCKM14}',
                   lhablock = 'RCKM',
                   lhacode = [ 1, 4 ])

RCKM21 = Parameter(name = 'RCKM21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{RCKM21}',
                   lhablock = 'RCKM',
                   lhacode = [ 2, 1 ])

RCKM22 = Parameter(name = 'RCKM22',
                   nature = 'external',
                   type = 'real',
                   value = 0.99995,
                   texname = '\\text{RCKM22}',
                   lhablock = 'RCKM',
                   lhacode = [ 2, 2 ])

RCKM23 = Parameter(name = 'RCKM23',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{RCKM23}',
                   lhablock = 'RCKM',
                   lhacode = [ 2, 3 ])

RCKM24 = Parameter(name = 'RCKM24',
                   nature = 'external',
                   type = 'real',
                   value = 0.01,
                   texname = '\\text{RCKM24}',
                   lhablock = 'RCKM',
                   lhacode = [ 2, 4 ])

RCKM31 = Parameter(name = 'RCKM31',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{RCKM31}',
                   lhablock = 'RCKM',
                   lhacode = [ 3, 1 ])

RCKM32 = Parameter(name = 'RCKM32',
                   nature = 'external',
                   type = 'real',
                   value = -0.001,
                   texname = '\\text{RCKM32}',
                   lhablock = 'RCKM',
                   lhacode = [ 3, 2 ])

RCKM33 = Parameter(name = 'RCKM33',
                   nature = 'external',
                   type = 'real',
                   value = 0.995,
                   texname = '\\text{RCKM33}',
                   lhablock = 'RCKM',
                   lhacode = [ 3, 3 ])

RCKM34 = Parameter(name = 'RCKM34',
                   nature = 'external',
                   type = 'real',
                   value = 0.1,
                   texname = '\\text{RCKM34}',
                   lhablock = 'RCKM',
                   lhacode = [ 3, 4 ])

RCKM41 = Parameter(name = 'RCKM41',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{RCKM41}',
                   lhablock = 'RCKM',
                   lhacode = [ 4, 1 ])

RCKM42 = Parameter(name = 'RCKM42',
                   nature = 'external',
                   type = 'real',
                   value = -0.01,
                   texname = '\\text{RCKM42}',
                   lhablock = 'RCKM',
                   lhacode = [ 4, 2 ])

RCKM43 = Parameter(name = 'RCKM43',
                   nature = 'external',
                   type = 'real',
                   value = -0.1,
                   texname = '\\text{RCKM43}',
                   lhablock = 'RCKM',
                   lhacode = [ 4, 3 ])

RCKM44 = Parameter(name = 'RCKM44',
                   nature = 'external',
                   type = 'real',
                   value = 0.99495,
                   texname = '\\text{RCKM44}',
                   lhablock = 'RCKM',
                   lhacode = [ 4, 4 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 132.50698,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.000011663900000000002,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.118,
               texname = '\\text{aS}',
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
                value = 174.3,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

ymbp = Parameter(name = 'ymbp',
                 nature = 'external',
                 type = 'real',
                 value = 500,
                 texname = '\\text{ymbp}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 7 ])

ymtp = Parameter(name = 'ymtp',
                 nature = 'external',
                 type = 'real',
                 value = 700,
                 texname = '\\text{ymtp}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 8 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

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

MTp = Parameter(name = 'MTp',
                nature = 'external',
                type = 'real',
                value = 700,
                texname = '\\text{MTp}',
                lhablock = 'MASS',
                lhacode = [ 8 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.7,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

MBp = Parameter(name = 'MBp',
                nature = 'external',
                type = 'real',
                value = 500,
                texname = '\\text{MBp}',
                lhablock = 'MASS',
                lhacode = [ 7 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.188,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 120,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.4516,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WTp = Parameter(name = 'WTp',
                nature = 'external',
                type = 'real',
                value = 14.109,
                texname = '\\text{WTp}',
                lhablock = 'DECAY',
                lhacode = [ 8 ])

WBp = Parameter(name = 'WBp',
                nature = 'external',
                type = 'real',
                value = 0.28454,
                texname = '\\text{WBp}',
                lhablock = 'DECAY',
                lhacode = [ 7 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.44140351,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.04759951,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00575308848,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

CKM11 = Parameter(name = 'CKM11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM11 + RCKM11',
                  texname = '\\text{CKM11}')

CKM12 = Parameter(name = 'CKM12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM12 + RCKM12',
                  texname = '\\text{CKM12}')

CKM13 = Parameter(name = 'CKM13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM13 + RCKM13',
                  texname = '\\text{CKM13}')

CKM14 = Parameter(name = 'CKM14',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM14 + RCKM14',
                  texname = '\\text{CKM14}')

CKM21 = Parameter(name = 'CKM21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM21 + RCKM21',
                  texname = '\\text{CKM21}')

CKM22 = Parameter(name = 'CKM22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM22 + RCKM22',
                  texname = '\\text{CKM22}')

CKM23 = Parameter(name = 'CKM23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM23 + RCKM23',
                  texname = '\\text{CKM23}')

CKM24 = Parameter(name = 'CKM24',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM24 + RCKM24',
                  texname = '\\text{CKM24}')

CKM31 = Parameter(name = 'CKM31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM31 + RCKM31',
                  texname = '\\text{CKM31}')

CKM32 = Parameter(name = 'CKM32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM32 + RCKM32',
                  texname = '\\text{CKM32}')

CKM33 = Parameter(name = 'CKM33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM33 + RCKM33',
                  texname = '\\text{CKM33}')

CKM34 = Parameter(name = 'CKM34',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM34 + RCKM34',
                  texname = '\\text{CKM34}')

CKM41 = Parameter(name = 'CKM41',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM41 + RCKM41',
                  texname = '\\text{CKM41}')

CKM42 = Parameter(name = 'CKM42',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM42 + RCKM42',
                  texname = '\\text{CKM42}')

CKM43 = Parameter(name = 'CKM43',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM43 + RCKM43',
                  texname = '\\text{CKM43}')

CKM44 = Parameter(name = 'CKM44',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*ICKM44 + RCKM44',
                  texname = '\\text{CKM44}')

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

ydo = Parameter(name = 'ydo',
                nature = 'internal',
                type = 'real',
                value = '0',
                texname = '\\text{ydo}')

ye = Parameter(name = 'ye',
               nature = 'internal',
               type = 'real',
               value = '0',
               texname = '\\text{ye}')

ym = Parameter(name = 'ym',
               nature = 'internal',
               type = 'real',
               value = '0',
               texname = '\\text{ym}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '0',
               texname = '\\text{ys}')

yup = Parameter(name = 'yup',
                nature = 'internal',
                type = 'real',
                value = '0',
                texname = '\\text{yup}')

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

ybp = Parameter(name = 'ybp',
                nature = 'internal',
                type = 'real',
                value = '(ymbp*cmath.sqrt(2))/v',
                texname = '\\text{ybp}')

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

ytp = Parameter(name = 'ytp',
                nature = 'internal',
                type = 'real',
                value = '(ymtp*cmath.sqrt(2))/v',
                texname = '\\text{ytp}')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*v**2)',
                texname = '\\mu')

