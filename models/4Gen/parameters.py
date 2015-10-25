# This file was automatically created by FeynRules 1.7.55
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (October 6, 2011)
# Date: Wed 8 Aug 2012 14:29:46



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
ICKM1x1 = Parameter(name = 'ICKM1x1',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM1x1}',
                    lhablock = 'ICKM',
                    lhacode = [ 1, 1 ])

ICKM1x2 = Parameter(name = 'ICKM1x2',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM1x2}',
                    lhablock = 'ICKM',
                    lhacode = [ 1, 2 ])

ICKM1x3 = Parameter(name = 'ICKM1x3',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM1x3}',
                    lhablock = 'ICKM',
                    lhacode = [ 1, 3 ])

ICKM1x4 = Parameter(name = 'ICKM1x4',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM1x4}',
                    lhablock = 'ICKM',
                    lhacode = [ 1, 4 ])

ICKM2x1 = Parameter(name = 'ICKM2x1',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM2x1}',
                    lhablock = 'ICKM',
                    lhacode = [ 2, 1 ])

ICKM2x2 = Parameter(name = 'ICKM2x2',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM2x2}',
                    lhablock = 'ICKM',
                    lhacode = [ 2, 2 ])

ICKM2x3 = Parameter(name = 'ICKM2x3',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM2x3}',
                    lhablock = 'ICKM',
                    lhacode = [ 2, 3 ])

ICKM2x4 = Parameter(name = 'ICKM2x4',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM2x4}',
                    lhablock = 'ICKM',
                    lhacode = [ 2, 4 ])

ICKM3x1 = Parameter(name = 'ICKM3x1',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM3x1}',
                    lhablock = 'ICKM',
                    lhacode = [ 3, 1 ])

ICKM3x2 = Parameter(name = 'ICKM3x2',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM3x2}',
                    lhablock = 'ICKM',
                    lhacode = [ 3, 2 ])

ICKM3x3 = Parameter(name = 'ICKM3x3',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM3x3}',
                    lhablock = 'ICKM',
                    lhacode = [ 3, 3 ])

ICKM3x4 = Parameter(name = 'ICKM3x4',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM3x4}',
                    lhablock = 'ICKM',
                    lhacode = [ 3, 4 ])

ICKM4x1 = Parameter(name = 'ICKM4x1',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM4x1}',
                    lhablock = 'ICKM',
                    lhacode = [ 4, 1 ])

ICKM4x2 = Parameter(name = 'ICKM4x2',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM4x2}',
                    lhablock = 'ICKM',
                    lhacode = [ 4, 2 ])

ICKM4x3 = Parameter(name = 'ICKM4x3',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM4x3}',
                    lhablock = 'ICKM',
                    lhacode = [ 4, 3 ])

ICKM4x4 = Parameter(name = 'ICKM4x4',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{ICKM4x4}',
                    lhablock = 'ICKM',
                    lhacode = [ 4, 4 ])

RCKM1x1 = Parameter(name = 'RCKM1x1',
                    nature = 'external',
                    type = 'real',
                    value = 1,
                    texname = '\\text{RCKM1x1}',
                    lhablock = 'RCKM',
                    lhacode = [ 1, 1 ])

RCKM1x2 = Parameter(name = 'RCKM1x2',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{RCKM1x2}',
                    lhablock = 'RCKM',
                    lhacode = [ 1, 2 ])

RCKM1x3 = Parameter(name = 'RCKM1x3',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{RCKM1x3}',
                    lhablock = 'RCKM',
                    lhacode = [ 1, 3 ])

RCKM1x4 = Parameter(name = 'RCKM1x4',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{RCKM1x4}',
                    lhablock = 'RCKM',
                    lhacode = [ 1, 4 ])

RCKM2x1 = Parameter(name = 'RCKM2x1',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{RCKM2x1}',
                    lhablock = 'RCKM',
                    lhacode = [ 2, 1 ])

RCKM2x2 = Parameter(name = 'RCKM2x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.99995,
                    texname = '\\text{RCKM2x2}',
                    lhablock = 'RCKM',
                    lhacode = [ 2, 2 ])

RCKM2x3 = Parameter(name = 'RCKM2x3',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{RCKM2x3}',
                    lhablock = 'RCKM',
                    lhacode = [ 2, 3 ])

RCKM2x4 = Parameter(name = 'RCKM2x4',
                    nature = 'external',
                    type = 'real',
                    value = 0.01,
                    texname = '\\text{RCKM2x4}',
                    lhablock = 'RCKM',
                    lhacode = [ 2, 4 ])

RCKM3x1 = Parameter(name = 'RCKM3x1',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{RCKM3x1}',
                    lhablock = 'RCKM',
                    lhacode = [ 3, 1 ])

RCKM3x2 = Parameter(name = 'RCKM3x2',
                    nature = 'external',
                    type = 'real',
                    value = -0.001,
                    texname = '\\text{RCKM3x2}',
                    lhablock = 'RCKM',
                    lhacode = [ 3, 2 ])

RCKM3x3 = Parameter(name = 'RCKM3x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.995,
                    texname = '\\text{RCKM3x3}',
                    lhablock = 'RCKM',
                    lhacode = [ 3, 3 ])

RCKM3x4 = Parameter(name = 'RCKM3x4',
                    nature = 'external',
                    type = 'real',
                    value = 0.1,
                    texname = '\\text{RCKM3x4}',
                    lhablock = 'RCKM',
                    lhacode = [ 3, 4 ])

RCKM4x1 = Parameter(name = 'RCKM4x1',
                    nature = 'external',
                    type = 'real',
                    value = 0,
                    texname = '\\text{RCKM4x1}',
                    lhablock = 'RCKM',
                    lhacode = [ 4, 1 ])

RCKM4x2 = Parameter(name = 'RCKM4x2',
                    nature = 'external',
                    type = 'real',
                    value = -0.01,
                    texname = '\\text{RCKM4x2}',
                    lhablock = 'RCKM',
                    lhacode = [ 4, 2 ])

RCKM4x3 = Parameter(name = 'RCKM4x3',
                    nature = 'external',
                    type = 'real',
                    value = -0.1,
                    texname = '\\text{RCKM4x3}',
                    lhablock = 'RCKM',
                    lhacode = [ 4, 3 ])

RCKM4x4 = Parameter(name = 'RCKM4x4',
                    nature = 'external',
                    type = 'real',
                    value = 0.99495,
                    texname = '\\text{RCKM4x4}',
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
               value = 125,
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
               value = 6.38233934e-03,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

CKM1x1 = Parameter(name = 'CKM1x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM1x1 + RCKM1x1',
                   texname = '\\text{CKM1x1}')

CKM1x2 = Parameter(name = 'CKM1x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM1x2 + RCKM1x2',
                   texname = '\\text{CKM1x2}')

CKM1x3 = Parameter(name = 'CKM1x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM1x3 + RCKM1x3',
                   texname = '\\text{CKM1x3}')

CKM1x4 = Parameter(name = 'CKM1x4',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM1x4 + RCKM1x4',
                   texname = '\\text{CKM1x4}')

CKM2x1 = Parameter(name = 'CKM2x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM2x1 + RCKM2x1',
                   texname = '\\text{CKM2x1}')

CKM2x2 = Parameter(name = 'CKM2x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM2x2 + RCKM2x2',
                   texname = '\\text{CKM2x2}')

CKM2x3 = Parameter(name = 'CKM2x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM2x3 + RCKM2x3',
                   texname = '\\text{CKM2x3}')

CKM2x4 = Parameter(name = 'CKM2x4',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM2x4 + RCKM2x4',
                   texname = '\\text{CKM2x4}')

CKM3x1 = Parameter(name = 'CKM3x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM3x1 + RCKM3x1',
                   texname = '\\text{CKM3x1}')

CKM3x2 = Parameter(name = 'CKM3x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM3x2 + RCKM3x2',
                   texname = '\\text{CKM3x2}')

CKM3x3 = Parameter(name = 'CKM3x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM3x3 + RCKM3x3',
                   texname = '\\text{CKM3x3}')

CKM3x4 = Parameter(name = 'CKM3x4',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM3x4 + RCKM3x4',
                   texname = '\\text{CKM3x4}')

CKM4x1 = Parameter(name = 'CKM4x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM4x1 + RCKM4x1',
                   texname = '\\text{CKM4x1}')

CKM4x2 = Parameter(name = 'CKM4x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM4x2 + RCKM4x2',
                   texname = '\\text{CKM4x2}')

CKM4x3 = Parameter(name = 'CKM4x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM4x3 + RCKM4x3',
                   texname = '\\text{CKM4x3}')

CKM4x4 = Parameter(name = 'CKM4x4',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complex(0,1)*ICKM4x4 + RCKM4x4',
                   texname = '\\text{CKM4x4}')

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

