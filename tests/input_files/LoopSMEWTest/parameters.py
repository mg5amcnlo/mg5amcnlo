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

# Loop related parameters

lhv = Parameter(name = 'lhv',
                 nature = 'internal',
                 type = 'real',
                 value = '1.0',
                 texname = '\lambda_{HV}')

vep = Parameter(name = 'vep',
                nature = 'internal',
                type = 'real',
                value = '0.0',
                texname = '\eps')

Ncol = Parameter(name = 'Ncol',
                 nature = 'internal',
                 type = 'real',
                 value = '3.0',
                 texname = 'N_{col}')

CA = Parameter(name = 'CA',
                 nature = 'internal',
                 type = 'real',
                 value = '3.0',
                 texname = 'C_{A}')

TF = Parameter(name = 'TF',
                 nature = 'internal',
                 type = 'real',
                 value = '0.5',
                 texname = 'T_{F}')

CF = Parameter(name = 'CF',
                 nature = 'internal',
                 type = 'real',
                 value = '(4.0/3.0)',
                 texname = 'C_{F}')

MU_R = Parameter(name = 'MU_R',
              nature = 'external',
              type = 'real',
              value = 91.188,
              texname = '\\text{\\mu_r}',
              lhablock = 'LOOP',
              lhacode = [ 666 ])

# User-defined parameters.

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.9,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

#Gf = Parameter(name = 'Gf',
#               nature = 'external',
#               type = 'real',
#               value = 0.0000116637,
#               texname = 'G_f',
#               lhablock = 'SMINPUTS',
#               lhacode = [ 2 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 80.419,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1184,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

lamWS = Parameter(name = 'lamWS',
                  nature = 'external',
                  type = 'real',
                  value = 0.2253,
                  texname = '\\text{lamWS}',
                  lhablock = 'Wolfenstein',
                  lhacode = [ 1 ])

AWS = Parameter(name = 'AWS',
                nature = 'external',
                type = 'real',
                value = 0.808,
                texname = '\\text{AWS}',
                lhablock = 'Wolfenstein',
                lhacode = [ 2 ])

rhoWS = Parameter(name = 'rhoWS',
                  nature = 'external',
                  type = 'real',
                  value = 0.132,
                  texname = '\\text{rhoWS}',
                  lhablock = 'Wolfenstein',
                  lhacode = [ 3 ])

etaWS = Parameter(name = 'etaWS',
                  nature = 'external',
                  type = 'real',
                  value = 0.341,
                  texname = '\\text{etaWS}',
                  lhablock = 'Wolfenstein',
                  lhacode = [ 4 ])

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

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\alpha _{\\text{EW}}')


G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

CKM11 = Parameter(name = 'CKM11',
                  nature = 'internal',
                  type = 'complex',
                  value = '1 - lamWS**2/2.',
                  texname = '\\text{CKM11}')

CKM12 = Parameter(name = 'CKM12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'lamWS',
                  texname = '\\text{CKM12}')

CKM13 = Parameter(name = 'CKM13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'AWS*lamWS**3*(-(etaWS*complex(0,1)) + rhoWS)',
                  texname = '\\text{CKM13}')

CKM21 = Parameter(name = 'CKM21',
                  nature = 'internal',
                  type = 'complex',
                  value = '-lamWS',
                  texname = '\\text{CKM21}')

CKM22 = Parameter(name = 'CKM22',
                  nature = 'internal',
                  type = 'complex',
                  value = '1 - lamWS**2/2.',
                  texname = '\\text{CKM22}')

CKM23 = Parameter(name = 'CKM23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'AWS*lamWS**2',
                  texname = '\\text{CKM23}')

CKM31 = Parameter(name = 'CKM31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'AWS*lamWS**3*(1 - etaWS*complex(0,1) - rhoWS)',
                  texname = '\\text{CKM31}')

CKM32 = Parameter(name = 'CKM32',
                  nature = 'internal',
                  type = 'complex',
                  value = '-(AWS*lamWS**2)',
                  texname = '\\text{CKM32}')

CKM33 = Parameter(name = 'CKM33',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{CKM33}')

#MW = Parameter(name = 'MW',
#               nature = 'internal',
#               type = 'real',
#               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
#               texname = 'M_W')

Gf = Parameter(name = 'Gf',
               nature = 'internal',
               type = 'complex',
               value = '-aEW*MZ**2*cmath.pi/(cmath.sqrt(2)*MW**2*(MW**2 - MZ**2))',
               texname = 'G_f',)

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
                texname = 'vev')

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

# To facilitate R2 writings

AxialZUp = Parameter(name = 'AxialZUp',
              nature = 'internal',
              type = 'real',
              value = '(3.0/2.0)*(-(ee*sw)/(6.*cw))-(1.0/2.0)*((cw*ee)/(2.*sw))',
              texname = 'AxialZUp')

AxialZDown = Parameter(name = 'AxialZDown',
              nature = 'internal',
              type = 'real',
              value = '(-1.0/2.0)*(-(cw*ee)/(2.*sw))+(-3.0/2.0)*(-(ee*sw)/(6.*cw))',
              texname = 'AxialZdown')

VectorZUp = Parameter(name = 'VectorZUp',
              nature = 'internal',                      
              type = 'real',
              value = '(1.0/2.0)*((cw*ee)/(2.*sw))+(5.0/2.0)*(-(ee*sw)/(6.*cw))',
              texname = 'AxialZUp')

VectorZDown = Parameter(name = 'VectorZDown',
              nature = 'internal',                        
              type = 'real',
              value = '(1.0/2.0)*(-(cw*ee)/(2.*sw))+(-1.0/2.0)*(-(ee*sw)/(6.*cw))',
              texname = 'AxialZdown')

AxialG0Up1 = Parameter(name = 'AxialG0Up1',
              nature = 'internal',
              type = 'real',
              value = '-yup/(cmath.sqrt(2))',
              texname = 'AxialG0Up1')

AxialG0Down1 = Parameter(name = 'AxialG0Down1',
              nature = 'internal',
              type = 'real',
              value = 'ydo/(cmath.sqrt(2))',
              texname = 'AxialG0down1')

AxialG0Up2 = Parameter(name = 'AxialG0Up2',
              nature = 'internal',
              type = 'real',
              value = '-yc/(cmath.sqrt(2))',
              texname = 'AxialG0Up2')

AxialG0Down2 = Parameter(name = 'AxialG0Down2',
              nature = 'internal',
              type = 'real',
              value = 'ys/(cmath.sqrt(2))',
              texname = 'AxialG0down2')

AxialG0Up3 = Parameter(name = 'AxialG0Up3',
              nature = 'internal',
              type = 'real',
              value = '-yt/(cmath.sqrt(2))',
              texname = 'AxialG0Up3')

AxialG0Down3 = Parameter(name = 'AxialG0Down3',
              nature = 'internal',
              type = 'real',
              value = 'yb/(cmath.sqrt(2))',
              texname = 'AxialG0down3')

VectorHUp1 = Parameter(name = 'VectorHUp1',
              nature = 'internal',
              type = 'complex',
              value = '-complex(0,1)*yup/(cmath.sqrt(2))',
              texname = 'VectorHUp1')

VectorHDown1 = Parameter(name = 'VectorHDown1',
              nature = 'internal',
              type = 'complex',
              value = '-complex(0,1)*ydo/(cmath.sqrt(2))',
              texname = 'VectorHdown1')

VectorHUp2 = Parameter(name = 'VectorHUp2',
              nature = 'internal',
              type = 'complex',
              value = '-complex(0,1)*yc/(cmath.sqrt(2))',
              texname = 'VectorHUp2')

VectorHDown2 = Parameter(name = 'VectorHDown2',
              nature = 'internal',
              type = 'complex',
              value = '-complex(0,1)*ys/(cmath.sqrt(2))',
              texname = 'VectorHdown2')

VectorHUp3 = Parameter(name = 'VectorHUp3',
              nature = 'internal',
              type = 'complex',
              value = '-complex(0,1)*yt/(cmath.sqrt(2))',
              texname = 'VectorHUp3')

VectorHDown3 = Parameter(name = 'VectorHDown3',
              nature = 'internal',
              type = 'complex',
              value = '-complex(0,1)*yb/(cmath.sqrt(2))',
              texname = 'VectorHdown3')

VectorAUp = Parameter(name = 'VectorAUp',
              nature = 'internal',                      
              type = 'real',
              value = '(2*ee)/3.',
              texname = 'VectorAUp')

VectorADown = Parameter(name = 'VectorADown',
              nature = 'internal',                        
              type = 'real',
              value = '-(ee)/3.',
              texname = 'VectorADown')

VectorWmDxU = Parameter(name = 'VectorWmDxU',
              nature = 'internal',                        
              type = 'real',
              value = '(1.0/2.0)*((ee)/(sw*cmath.sqrt(2)))',
              texname = 'VectorWmDxU')

AxialWmDxU = Parameter(name = 'AxialWmDxU',
              nature = 'internal',                        
              type = 'real',
              value = '(-1.0/2.0)*((ee)/(sw*cmath.sqrt(2)))',
              texname = 'AxialWmDxU')

VectorWpUxD = Parameter(name = 'VectorWpUxD',
              nature = 'internal',                        
              type = 'real',
              value = '(1.0/2.0)*((ee)/(sw*cmath.sqrt(2)))',
              texname = 'VectorWpUxD')

AxialWpUxD = Parameter(name = 'AxialWpUxD',
              nature = 'internal',                        
              type = 'real',
              value = '-(1.0/2.0)*((ee)/(sw*cmath.sqrt(2)))',
              texname = 'AxialWpUxD')

VectorGpUx1D1 = Parameter(name = 'VectorGpUx1D1',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM11)*(yup-ydo)',
              texname = 'VectorGpUx1D1')

AxialGpUx1D1 = Parameter(name = 'AxialGpUx1D1',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM11)*(yup+ydo)',
              texname = 'AxialGpUx1D1')

VectorGpUx1D2 = Parameter(name = 'VectorGpUx1D2',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM12)*(yup-ys)',
              texname = 'VectorGpUx1D2')

AxialGpUx1D2 = Parameter(name = 'AxialGpUx1D2',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM12)*(yup+ys)',
              texname = 'AxialGpUx1D2')

VectorGpUx1D3 = Parameter(name = 'VectorGpUx1D3',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM13)*(yup-yb)',
              texname = 'VectorGpUx1D3')

AxialGpUx1D3 = Parameter(name = 'AxialGpUx1D3',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM13)*(yup+yb)',
              texname = 'AxialGpUx1D3')

VectorGpUx2D1 = Parameter(name = 'VectorGpUx2D1',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM21)*(yc-ydo)',
              texname = 'VectorGpUx2D1')

AxialGpUx2D1 = Parameter(name = 'AxialGpUx2D1',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM11)*(yc+ydo)',
              texname = 'AxialGpUx2D1')

VectorGpUx2D2 = Parameter(name = 'VectorGpUx2D2',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM22)*(yc-ys)',
              texname = 'VectorGpUx2D2')

AxialGpUx2D2 = Parameter(name = 'AxialGpUx2D2',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM22)*(yc+ys)',
              texname = 'AxialGpUx2D2')

VectorGpUx2D3 = Parameter(name = 'VectorGpUx2D3',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM23)*(yc-yb)',
              texname = 'VectorGpUx2D3')

AxialGpUx2D3 = Parameter(name = 'AxialGpUx2D3',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM23)*(yc+yb)',
              texname = 'AxialGpUx2D3')

VectorGpUx3D1 = Parameter(name = 'VectorGpUx3D1',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM31)*(yt-ydo)',
              texname = 'VectorGpUx3D1')

AxialGpUx3D1 = Parameter(name = 'AxialGpUx3D1',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM31)*(yt+ydo)',
              texname = 'AxialGpUx3D1')

VectorGpUx3D2 = Parameter(name = 'VectorGpUx3D2',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM32)*(yt-ys)',
              texname = 'VectorGpUx3D2')

AxialGpUx3D2 = Parameter(name = 'AxialGpUx3D2',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM32)*(yt+ys)',
              texname = 'AxialGpUx3D2')

VectorGpUx3D3 = Parameter(name = 'VectorGpUx3D3',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM33)*(yt-yb)',
              texname = 'VectorGpUx3D3')

AxialGpUx3D3 = Parameter(name = 'AxialGpUx3D3',
              nature = 'internal',                        
              type = 'complex',
              value = '((-1.)/2.)*complexconjugate(CKM33)*(yt+yb)',
              texname = 'AxialGpUx3D3')

VectorGmDx1U1 = Parameter(name = 'VectorGmDx1U1',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM11*(yup-ydo)',
              texname = 'VectorGmDx1U1')

AxialGmDx1U1 = Parameter(name = 'AxialGmDx1U1',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM11*(yup+ydo)',
              texname = 'AxialGmDx1U1')

VectorGmDx2U1 = Parameter(name = 'VectorGmDx2U1',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM12*(yup-ys)',
              texname = 'VectorGmDx2U1')

AxialGmDx2U1 = Parameter(name = 'AxialGmDx2U1',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM12*(yup+ys)',
              texname = 'AxialGmDx2U1')

VectorGmDx3U1 = Parameter(name = 'VectorGmDx3U1',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM13*(yup-yb)',
              texname = 'VectorGmDx3U1')

AxialGmDx3U1 = Parameter(name = 'AxialGmDx3U1',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM13*(yup+yb)',
              texname = 'AxialGmDx3U1')

VectorGmDx1U2 = Parameter(name = 'VectorGmDx1U2',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM21*(yc-ydo)',
              texname = 'VectorGmDx1U2')

AxialGmDx1U2 = Parameter(name = 'AxialGmDx1U2',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM11*(yc+ydo)',
              texname = 'AxialGmDx1U2')

VectorGmDx2U2 = Parameter(name = 'VectorGmDx2U2',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM22*(yc-ys)',
              texname = 'VectorGmDx2U2')

AxialGmDx2U2 = Parameter(name = 'AxialGmDx2U2',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM22*(yc+ys)',
              texname = 'AxialGmDx2U2')

VectorGmDx3U2 = Parameter(name = 'VectorGmDx3U2',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM23*(yc-yb)',
              texname = 'VectorGmDx3U2')

AxialGmDx3U2 = Parameter(name = 'AxialGmDx3U2',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM23*(yc+yb)',
              texname = 'AxialGmDx3U2')

VectorGmDx1U3 = Parameter(name = 'VectorGmDx1U3',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM31*(yt-ydo)',
              texname = 'VectorGmDx1U3')

AxialGmDx1U3 = Parameter(name = 'AxialGmDx1U3',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM31*(yt+ydo)',
              texname = 'AxialGmDx1U3')

VectorGmDx2U3 = Parameter(name = 'VectorGmDx2U3',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM32*(yt-ys)',
              texname = 'VectorGmDx2U3')

AxialGmDx2U3 = Parameter(name = 'AxialGmDx2U3',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM32*(yt+ys)',
              texname = 'AxialGmDx2U3')

VectorGmDx3U3 = Parameter(name = 'VectorGmDx3U3',
              nature = 'internal',                        
              type = 'complex',
              value = '(1./2.)*CKM33*(yt-yb)',
              texname = 'VectorGmDx3U3')

AxialGmDx3U3 = Parameter(name = 'AxialGmDx3U3',
              nature = 'internal',                        
              type = 'complex',
              value = '(-1./2.)*CKM33*(yt+yb)',
              texname = 'AxialGmDx3U3')

I3d = Parameter(name = 'I3d',
                nature = 'internal',
                type = 'real',
                value = '(-1.0/2.0)',
                texname = 'I_{3d}')

I3u = Parameter(name = 'I3u',
                nature = 'internal',
                type = 'real',
                value = '(1.0/2.0)',
                texname = 'I_{3u}')

I3l = Parameter(name = 'I3l',
                nature = 'internal',
                type = 'real',
                value = '(-1.0/2.0)',
                texname = 'I_{3l}')

I3v = Parameter(name = 'I3v',
                nature = 'internal',
                type = 'real',
                value = '(1.0/2.0)',
                texname = 'I_{3v}')

Qd = Parameter(name = 'Qd',
                nature = 'internal',
                type = 'real',
                value = '(-1.0/3.0)',
                texname = 'Q_{d}')

Qu = Parameter(name = 'Qu',
                nature = 'internal',
                type = 'real',
                value = '(2.0/3.0)',
                texname = 'Q_{u}')

Ql = Parameter(name = 'Ql',
                nature = 'internal',
                type = 'real',
                value = '(-1.0)',
                texname = 'Q_{l}')


SCKM11 = Parameter(name = 'SCKM11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM11*complexconjugate(CKM11)',
                  texname = '\\text{SCKM11}')

SCKM12 = Parameter(name = 'SCKM12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM12*complexconjugate(CKM12)',
                  texname = '\\text{SCKM12}')

SCKM13 = Parameter(name = 'SCKM13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM13*complexconjugate(CKM13)',
                  texname = '\\text{SCKM13}')

SCKM21 = Parameter(name = 'SCKM21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM21*complexconjugate(CKM21)',
                  texname = '\\text{SCKM21}')

SCKM22 = Parameter(name = 'SCKM22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM22*complexconjugate(CKM22)',
                  texname = '\\text{SCKM22}')

SCKM23 = Parameter(name = 'SCKM23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM23*complexconjugate(CKM23)',
                  texname = '\\text{SCKM23}')

SCKM31 = Parameter(name = 'SCKM31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM31*complexconjugate(CKM31)',
                  texname = '\\text{SCKM31}')

SCKM32 = Parameter(name = 'SCKM32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM32*complexconjugate(CKM32)',
                  texname = '\\text{SCKM32}')

SCKM33 = Parameter(name = 'SCKM33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM33*complexconjugate(CKM33)',
                  texname = '\\text{SCKM33}')

CKMucds = Parameter(name = 'CKMucds',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM11*CKM22*complexconjugate(CKM12)*complexconjugate(CKM21)+CKM12*CKM21*complexconjugate(CKM11)*complexconjugate(CKM22)',
                    texname = '\\text{CKMucds}')

CKMucdb = Parameter(name = 'CKMucdb',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM11*CKM23*complexconjugate(CKM13)*complexconjugate(CKM21)+CKM13*CKM21*complexconjugate(CKM11)*complexconjugate(CKM23)',
                    texname = '\\text{CKMucdb}')

CKMucsb = Parameter(name = 'CKMucsb',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM12*CKM23*complexconjugate(CKM13)*complexconjugate(CKM22)+CKM22*CKM13*complexconjugate(CKM12)*complexconjugate(CKM23)',
                    texname = '\\text{CKMucsb}')

CKMutds = Parameter(name = 'CKMutds',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM11*CKM32*complexconjugate(CKM12)*complexconjugate(CKM31)+CKM12*CKM31*complexconjugate(CKM11)*complexconjugate(CKM32)',
                    texname = '\\text{CKMutds}')

CKMutdb = Parameter(name = 'CKMutdb',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM11*CKM33*complexconjugate(CKM13)*complexconjugate(CKM31)+CKM13*CKM31*complexconjugate(CKM11)*complexconjugate(CKM33)',
                    texname = '\\text{CKMutdb}')

CKMutsb = Parameter(name = 'CKMutsb',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM12*CKM33*complexconjugate(CKM13)*complexconjugate(CKM32)+CKM13*CKM32*complexconjugate(CKM12)*complexconjugate(CKM33)',
                    texname = '\\text{CKMutsb}')

CKMctds = Parameter(name = 'CKMctds',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM21*CKM32*complexconjugate(CKM22)*complexconjugate(CKM31)+CKM22*CKM31*complexconjugate(CKM21)*complexconjugate(CKM32)',
                    texname = '\\text{CKMctds}')

CKMctdb = Parameter(name = 'CKMctdb',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM21*CKM33*complexconjugate(CKM23)*complexconjugate(CKM31)+CKM23*CKM31*complexconjugate(CKM21)*complexconjugate(CKM33)',
                    texname = '\\text{CKMctdb}')

CKMctsb = Parameter(name = 'CKMctsb',
                    nature = 'internal',
                    type = 'complex',
                    value = 'CKM22*CKM33*complexconjugate(CKM23)*complexconjugate(CKM32)+CKM32*CKM23*complexconjugate(CKM22)*complexconjugate(CKM33)',
                    texname = '\\text{CKMctsb}')
