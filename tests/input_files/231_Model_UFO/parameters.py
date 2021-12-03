# This file was automatically created by FeynRules 2.3.41
# Mathematica version: 10.4.1 for Linux x86 (64-bit) (April 11, 2016)
# Date: Fri 21 May 2021 11:43:21



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

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

kappa = Parameter(name = 'kappa',
                  nature = 'external',
                  type = 'real',
                  value = 0.5,
                  texname = '\\kappa',
                  lhablock = 'Higgs',
                  lhacode = [ 1 ])

mixh = Parameter(name = 'mixh',
                 nature = 'external',
                 type = 'real',
                 value = 0.55,
                 texname = '\\alpha _{\\text{h1}}',
                 lhablock = 'Higgs',
                 lhacode = [ 2 ])

mixh2 = Parameter(name = 'mixh2',
                  nature = 'external',
                  type = 'real',
                  value = 0.78,
                  texname = '\\alpha _{\\text{h2}}',
                  lhablock = 'Higgs',
                  lhacode = [ 3 ])

mixh3 = Parameter(name = 'mixh3',
                  nature = 'external',
                  type = 'real',
                  value = 0.15,
                  texname = '\\alpha _{\\text{h4}}',
                  lhablock = 'Higgs',
                  lhacode = [ 4 ])

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

vev1 = Parameter(name = 'vev1',
                 nature = 'external',
                 type = 'real',
                 value = 1500,
                 texname = '\\text{vev1}',
                 lhablock = 'FRBlock',
                 lhacode = [ 1 ])

vev = Parameter(name = 'vev',
                nature = 'external',
                type = 'real',
                value = 246.0731,
                texname = '\\text{vev}',
                lhablock = 'FRBlock',
                lhacode = [ 2 ])

tanb = Parameter(name = 'tanb',
                 nature = 'external',
                 type = 'real',
                 value = 5.e-8,
                 texname = '\\text{tanb}',
                 lhablock = 'FRBlock',
                 lhacode = [ 3 ])

Mve = Parameter(name = 'Mve',
                nature = 'external',
                type = 'real',
                value = 0.00010,
                texname = '\\text{Mve}',
                lhablock = 'MASS',
                lhacode = [ 12 ])

Mvm = Parameter(name = 'Mvm',
                nature = 'external',
                type = 'real',
                value = 0.00011,
                texname = '\\text{Mvm}',
                lhablock = 'MASS',
                lhacode = [ 14 ])

Mvt = Parameter(name = 'Mvt',
                nature = 'external',
                type = 'real',
                value = 0.00012,
                texname = '\\text{Mvt}',
                lhablock = 'MASS',
                lhacode = [ 16 ])

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.000511,
               texname = '\\text{Me}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

MMU = Parameter(name = 'MMU',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{MMU}',
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

mh1 = Parameter(name = 'mh1',
                nature = 'external',
                type = 'real',
                value = 125,
                texname = '\\text{mh1}',
                lhablock = 'MASS',
                lhacode = [ 25 ])

mh2 = Parameter(name = 'mh2',
                nature = 'external',
                type = 'real',
                value = 155,
                texname = '\\text{mh2}',
                lhablock = 'MASS',
                lhacode = [ 35 ])

mh4 = Parameter(name = 'mh4',
                nature = 'external',
                type = 'real',
                value = 632,
                texname = '\\text{mh4}',
                lhablock = 'MASS',
                lhacode = [ 777 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 250 ])

mh3 = Parameter(name = 'mh3',
                nature = 'external',
                type = 'real',
                value = 763,
                texname = '\\text{mh3}',
                lhablock = 'MASS',
                lhacode = [ 36 ])

mhc = Parameter(name = 'mhc',
                nature = 'external',
                type = 'real',
                value = 768,
                texname = '\\text{mhc}',
                lhablock = 'MASS',
                lhacode = [ 37 ])

mhcc = Parameter(name = 'mhcc',
                 nature = 'external',
                 type = 'real',
                 value = 773,
                 texname = '\\text{mhcc}',
                 lhablock = 'MASS',
                 lhacode = [ 9000005 ])

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

Wh1 = Parameter(name = 'Wh1',
                nature = 'external',
                type = 'real',
                value = 0.00407,
                texname = '\\text{Wh1}',
                lhablock = 'DECAY',
                lhacode = [ 25 ])

Wh2 = Parameter(name = 'Wh2',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{Wh2}',
                lhablock = 'DECAY',
                lhacode = [ 35 ])

Wh4 = Parameter(name = 'Wh4',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{Wh4}',
                lhablock = 'DECAY',
                lhacode = [ 777 ])

WG0 = Parameter(name = 'WG0',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{WG0}',
                lhablock = 'DECAY',
                lhacode = [ 250 ])

Wh3 = Parameter(name = 'Wh3',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{Wh3}',
                lhablock = 'DECAY',
                lhacode = [ 36 ])

whc = Parameter(name = 'whc',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\text{whc}',
                lhablock = 'DECAY',
                lhacode = [ 37 ])

whcc = Parameter(name = 'whcc',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{whcc}',
                 lhablock = 'DECAY',
                 lhacode = [ 9000005 ])

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

vev2 = Parameter(name = 'vev2',
                 nature = 'internal',
                 type = 'real',
                 value = '(vev*cmath.sqrt(2))/(tanb*cmath.sqrt(4 + 2/tanb**2))',
                 texname = '\\text{vev2}')

vev3 = Parameter(name = 'vev3',
                 nature = 'internal',
                 type = 'real',
                 value = 'vev/cmath.sqrt(4 + 2/tanb**2)',
                 texname = '\\text{vev3}')

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

yc = Parameter(name = 'yc',
               nature = 'internal',
               type = 'real',
               value = '(ymc*cmath.sqrt(2))/vev',
               texname = '\\text{yc}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/vev',
               texname = '\\text{yt}')

ydo = Parameter(name = 'ydo',
                nature = 'internal',
                type = 'real',
                value = '(ymdo*cmath.sqrt(2))/vev',
                texname = '\\text{ydo}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '(yms*cmath.sqrt(2))/vev',
               texname = '\\text{ys}')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/vev',
               texname = '\\text{yb}')

CKM1x1 = Parameter(name = 'CKM1x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.cos(cabi)',
                   texname = '\\text{CKM1x1}')

CKM1x2 = Parameter(name = 'CKM1x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.sin(cabi)',
                   texname = '\\text{CKM1x2}')

CKM1x3 = Parameter(name = 'CKM1x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM1x3}')

CKM2x1 = Parameter(name = 'CKM2x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '-cmath.sin(cabi)',
                   texname = '\\text{CKM2x1}')

CKM2x2 = Parameter(name = 'CKM2x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'cmath.cos(cabi)',
                   texname = '\\text{CKM2x2}')

CKM2x3 = Parameter(name = 'CKM2x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM2x3}')

CKM3x1 = Parameter(name = 'CKM3x1',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM3x1}')

CKM3x2 = Parameter(name = 'CKM3x2',
                   nature = 'internal',
                   type = 'complex',
                   value = '0',
                   texname = '\\text{CKM3x2}')

CKM3x3 = Parameter(name = 'CKM3x3',
                   nature = 'internal',
                   type = 'complex',
                   value = '1',
                   texname = '\\text{CKM3x3}')

UH1x1 = Parameter(name = 'UH1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.cos(mixh)*cmath.cos(mixh2)',
                  texname = '\\text{UH1x1}')

UH1x2 = Parameter(name = 'UH1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.cos(mixh2)*cmath.sin(mixh)',
                  texname = '\\text{UH1x2}')

UH1x3 = Parameter(name = 'UH1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.sin(mixh2)',
                  texname = '\\text{UH1x3}')

UH2x1 = Parameter(name = 'UH2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = '-(cmath.cos(mixh3)*cmath.sin(mixh)) - cmath.cos(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3)',
                  texname = '\\text{UH2x1}')

UH2x2 = Parameter(name = 'UH2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.cos(mixh)*cmath.cos(mixh3) - cmath.sin(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3)',
                  texname = '\\text{UH2x2}')

UH2x3 = Parameter(name = 'UH2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.cos(mixh2)*cmath.sin(mixh3)',
                  texname = '\\text{UH2x3}')

UH3x1 = Parameter(name = 'UH3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = '-(cmath.cos(mixh)*cmath.cos(mixh3)*cmath.sin(mixh2)) + cmath.sin(mixh)*cmath.sin(mixh3)',
                  texname = '\\text{UH3x1}')

UH3x2 = Parameter(name = 'UH3x2',
                  nature = 'internal',
                  type = 'complex',
                  value = '-(cmath.cos(mixh3)*cmath.sin(mixh)*cmath.sin(mixh2)) - cmath.cos(mixh)*cmath.sin(mixh3)',
                  texname = '\\text{UH3x2}')

UH3x3 = Parameter(name = 'UH3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'cmath.cos(mixh2)*cmath.cos(mixh3)',
                  texname = '\\text{UH3x3}')

TH1x1 = Parameter(name = 'TH1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{TH1x1}')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
               texname = 'M_W')

NA = Parameter(name = 'NA',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 + vev3**2/vev1**2 + (4*vev3**2)/vev2**2)',
               texname = 'N_A')

NG = Parameter(name = 'NG',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 + (4*vev3**2)/vev2**2)',
               texname = 'N_G')

b1 = Parameter(name = 'b1',
               nature = 'internal',
               type = 'real',
               value = '(-((mh3**2*vev2**2*vev3**2)/(4*vev1**2*vev3**2 + vev2**2*(vev1**2 + vev3**2))) + mh1**2*cmath.cos(mixh)**2*cmath.cos(mixh2)**2 + mh4**2*(-(cmath.cos(mixh)*cmath.cos(mixh3)*cmath.sin(mixh2)) + cmath.sin(mixh)*cmath.sin(mixh3))**2 + mh2**2*(-(cmath.cos(mixh3)*cmath.sin(mixh)) - cmath.cos(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3))**2)/(2.*vev1**2)',
               texname = '\\beta _1')

b2 = Parameter(name = 'b2',
               nature = 'internal',
               type = 'real',
               value = '((-2*mh3**2*vev1*vev2*vev3**2)/(4*vev1**2*vev3**2 + vev2**2*(vev1**2 + vev3**2)) + mh1**2*cmath.cos(mixh)*cmath.cos(mixh2)**2*cmath.sin(mixh) + mh4**2*(-(cmath.cos(mixh3)*cmath.sin(mixh)*cmath.sin(mixh2)) - cmath.cos(mixh)*cmath.sin(mixh3))*(-(cmath.cos(mixh)*cmath.cos(mixh3)*cmath.sin(mixh2)) + cmath.sin(mixh)*cmath.sin(mixh3)) + mh2**2*(-(cmath.cos(mixh3)*cmath.sin(mixh)) - cmath.cos(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3))*(cmath.cos(mixh)*cmath.cos(mixh3) - cmath.sin(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3)))/(vev1*vev2)',
               texname = '\\beta _2')

b3 = Parameter(name = 'b3',
               nature = 'internal',
               type = 'real',
               value = '((mh3**2*vev1*vev2**2*vev3)/(4*vev1**2*vev3**2 + vev2**2*(vev1**2 + vev3**2)) + mh1**2*cmath.cos(mixh)*cmath.cos(mixh2)*cmath.sin(mixh2) + mh4**2*cmath.cos(mixh2)*cmath.cos(mixh3)*(-(cmath.cos(mixh)*cmath.cos(mixh3)*cmath.sin(mixh2)) + cmath.sin(mixh)*cmath.sin(mixh3)) + mh2**2*cmath.cos(mixh2)*cmath.sin(mixh3)*(-(cmath.cos(mixh3)*cmath.sin(mixh)) - cmath.cos(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3)))/(vev1*vev3)',
               texname = '\\beta _3')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

l1 = Parameter(name = 'l1',
               nature = 'internal',
               type = 'real',
               value = '(mh1**2*cmath.cos(mixh2)**2*cmath.sin(mixh)**2 + mh4**2*(-(cmath.cos(mixh3)*cmath.sin(mixh)*cmath.sin(mixh2)) - cmath.cos(mixh)*cmath.sin(mixh3))**2 + mh2**2*(cmath.cos(mixh)*cmath.cos(mixh3) - cmath.sin(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3))**2)/(2.*vev2**2)',
               texname = '\\lambda _1')

l2 = Parameter(name = 'l2',
               nature = 'internal',
               type = 'real',
               value = '(2*mhcc**2 - (4*mhc**2*vev2**2)/vev**2 + (mh3**2*vev1**2*vev2**2)/(4*vev1**2*vev3**2 + vev2**2*(vev1**2 + vev3**2)) + mh4**2*cmath.cos(mixh2)**2*cmath.cos(mixh3)**2 + mh1**2*cmath.sin(mixh2)**2 + mh2**2*cmath.cos(mixh2)**2*cmath.sin(mixh3)**2)/(2.*vev3**2)',
               texname = '\\lambda _2')

l3 = Parameter(name = 'l3',
               nature = 'internal',
               type = 'real',
               value = '((-4*mhc**2)/vev**2 - (2*mh3**2*vev1**2)/(4*vev1**2*vev3**2 + vev2**2*(vev1**2 + vev3**2)) + mh1**2*cmath.cos(mixh2)*cmath.sin(mixh)*cmath.sin(mixh2) + mh4**2*cmath.cos(mixh2)*cmath.cos(mixh3)*(-(cmath.cos(mixh3)*cmath.sin(mixh)*cmath.sin(mixh2)) - cmath.cos(mixh)*cmath.sin(mixh3)) + mh2**2*cmath.cos(mixh2)*cmath.sin(mixh3)*(cmath.cos(mixh)*cmath.cos(mixh3) - cmath.sin(mixh)*cmath.sin(mixh2)*cmath.sin(mixh3)))/(vev2*vev3)',
               texname = '\\lambda _3')

l4 = Parameter(name = 'l4',
               nature = 'internal',
               type = 'real',
               value = '(-mhcc**2 + (2*mhc**2*vev2**2)/vev**2 - (vev1*vev2**2)/(kappa*vev3))/vev3**2',
               texname = '\\lambda _4')

l5 = Parameter(name = 'l5',
               nature = 'internal',
               type = 'real',
               value = '2*((-2*mhc**2)/vev**2 + (kappa*vev1)/vev3)',
               texname = '\\lambda _5')

yne = Parameter(name = 'yne',
                nature = 'internal',
                type = 'real',
                value = '1.4142135623730952e-12/vev3',
                texname = '\\text{yne}')

ynmu = Parameter(name = 'ynmu',
                 nature = 'internal',
                 type = 'real',
                 value = '1.2162236636408618e-6/vev3',
                 texname = '\\text{ynmu}')

yntau = Parameter(name = 'yntau',
                  nature = 'internal',
                  type = 'real',
                  value = '0.000042426406871192855/vev3',
                  texname = '\\text{yntau}')

mu1 = Parameter(name = 'mu1',
                nature = 'internal',
                type = 'real',
                value = '(-2*b1*vev1**3 - b2*vev1*vev2**2 + kappa*vev2**2*vev3 - b3*vev1*vev3**2)/(2.*vev1)',
                texname = '\\text{mu1}')

mu2 = Parameter(name = 'mu2',
                nature = 'internal',
                type = 'real',
                value = '(-(b2*vev1**2) - 2*l1*vev2**2 + 2*kappa*vev1*vev3 - l3*vev3**2 - l5*vev3**2)/2.',
                texname = '\\text{mu2}')

mu3 = Parameter(name = 'mu3',
                nature = 'internal',
                type = 'real',
                value = '(kappa*vev1*vev2**2 - b3*vev1**2*vev3 - l3*vev2**2*vev3 - l5*vev2**2*vev3 - 2*l2*vev3**3 - 2*l4*vev3**3)/(2.*vev3)',
                texname = '\\text{mu3}')

NJ = Parameter(name = 'NJ',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(NG**4 + vev3**2/vev1**2 + (4*vev3**4)/(vev1**2*vev2**2))',
               texname = 'N_J')

sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - MW**2/MZ**2',
                texname = '\\text{sw2}')

TH1x3 = Parameter(name = 'TH1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'vev3/(NA*vev1)',
                  texname = '\\text{TH1x3}')

TH2x3 = Parameter(name = 'TH2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = '(2*vev3)/(NA*vev2)',
                  texname = '\\text{TH2x3}')

TH3x1 = Parameter(name = 'TH3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = '(-2*vev3)/(NG*vev2)',
                  texname = '\\text{TH3x1}')

TH3x3 = Parameter(name = 'TH3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = '1/NG',
                  texname = '\\text{TH3x3}')

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

TH1x2 = Parameter(name = 'TH1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'NG**2/NJ',
                  texname = '\\text{TH1x2}')

TH2x1 = Parameter(name = 'TH2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = '1/NJ',
                  texname = '\\text{TH2x1}')

TH2x2 = Parameter(name = 'TH2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = '(-2*vev3**2)/(NJ*vev1*vev2)',
                  texname = '\\text{TH2x2}')

TH3x2 = Parameter(name = 'TH3x2',
                  nature = 'internal',
                  type = 'complex',
                  value = '-(vev3/(NJ*vev1))',
                  texname = '\\text{TH3x2}')

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

I1a11 = Parameter(name = 'I1a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ydo*complexconjugate(CKM1x1)',
                  texname = '\\text{I1a11}')

I1a12 = Parameter(name = 'I1a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ydo*complexconjugate(CKM2x1)',
                  texname = '\\text{I1a12}')

I1a13 = Parameter(name = 'I1a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ydo*complexconjugate(CKM3x1)',
                  texname = '\\text{I1a13}')

I1a21 = Parameter(name = 'I1a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ys*complexconjugate(CKM1x2)',
                  texname = '\\text{I1a21}')

I1a22 = Parameter(name = 'I1a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ys*complexconjugate(CKM2x2)',
                  texname = '\\text{I1a22}')

I1a23 = Parameter(name = 'I1a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ys*complexconjugate(CKM3x2)',
                  texname = '\\text{I1a23}')

I1a31 = Parameter(name = 'I1a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yb*complexconjugate(CKM1x3)',
                  texname = '\\text{I1a31}')

I1a32 = Parameter(name = 'I1a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yb*complexconjugate(CKM2x3)',
                  texname = '\\text{I1a32}')

I1a33 = Parameter(name = 'I1a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yb*complexconjugate(CKM3x3)',
                  texname = '\\text{I1a33}')

I10a1 = Parameter(name = 'I10a1',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{I10a1}')

I10a2 = Parameter(name = 'I10a2',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{I10a2}')

I10a3 = Parameter(name = 'I10a3',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{I10a3}')

I11a1 = Parameter(name = 'I11a1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yne',
                  texname = '\\text{I11a1}')

I11a2 = Parameter(name = 'I11a2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ynmu',
                  texname = '\\text{I11a2}')

I11a3 = Parameter(name = 'I11a3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yntau',
                  texname = '\\text{I11a3}')

I12a1 = Parameter(name = 'I12a1',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{I12a1}')

I12a2 = Parameter(name = 'I12a2',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{I12a2}')

I12a3 = Parameter(name = 'I12a3',
                  nature = 'internal',
                  type = 'complex',
                  value = '1',
                  texname = '\\text{I12a3}')

I13a1 = Parameter(name = 'I13a1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yne',
                  texname = '\\text{I13a1}')

I13a2 = Parameter(name = 'I13a2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ynmu',
                  texname = '\\text{I13a2}')

I13a3 = Parameter(name = 'I13a3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yntau',
                  texname = '\\text{I13a3}')

I2a11 = Parameter(name = 'I2a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yup*complexconjugate(CKM1x1)',
                  texname = '\\text{I2a11}')

I2a12 = Parameter(name = 'I2a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yc*complexconjugate(CKM2x1)',
                  texname = '\\text{I2a12}')

I2a13 = Parameter(name = 'I2a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt*complexconjugate(CKM3x1)',
                  texname = '\\text{I2a13}')

I2a21 = Parameter(name = 'I2a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yup*complexconjugate(CKM1x2)',
                  texname = '\\text{I2a21}')

I2a22 = Parameter(name = 'I2a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yc*complexconjugate(CKM2x2)',
                  texname = '\\text{I2a22}')

I2a23 = Parameter(name = 'I2a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt*complexconjugate(CKM3x2)',
                  texname = '\\text{I2a23}')

I2a31 = Parameter(name = 'I2a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yup*complexconjugate(CKM1x3)',
                  texname = '\\text{I2a31}')

I2a32 = Parameter(name = 'I2a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yc*complexconjugate(CKM2x3)',
                  texname = '\\text{I2a32}')

I2a33 = Parameter(name = 'I2a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yt*complexconjugate(CKM3x3)',
                  texname = '\\text{I2a33}')

I3a11 = Parameter(name = 'I3a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x1*yup',
                  texname = '\\text{I3a11}')

I3a12 = Parameter(name = 'I3a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x2*yup',
                  texname = '\\text{I3a12}')

I3a13 = Parameter(name = 'I3a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x3*yup',
                  texname = '\\text{I3a13}')

I3a21 = Parameter(name = 'I3a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x1*yc',
                  texname = '\\text{I3a21}')

I3a22 = Parameter(name = 'I3a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x2*yc',
                  texname = '\\text{I3a22}')

I3a23 = Parameter(name = 'I3a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x3*yc',
                  texname = '\\text{I3a23}')

I3a31 = Parameter(name = 'I3a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x1*yt',
                  texname = '\\text{I3a31}')

I3a32 = Parameter(name = 'I3a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x2*yt',
                  texname = '\\text{I3a32}')

I3a33 = Parameter(name = 'I3a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x3*yt',
                  texname = '\\text{I3a33}')

I4a11 = Parameter(name = 'I4a11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x1*ydo',
                  texname = '\\text{I4a11}')

I4a12 = Parameter(name = 'I4a12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x2*ys',
                  texname = '\\text{I4a12}')

I4a13 = Parameter(name = 'I4a13',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM1x3*yb',
                  texname = '\\text{I4a13}')

I4a21 = Parameter(name = 'I4a21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x1*ydo',
                  texname = '\\text{I4a21}')

I4a22 = Parameter(name = 'I4a22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x2*ys',
                  texname = '\\text{I4a22}')

I4a23 = Parameter(name = 'I4a23',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM2x3*yb',
                  texname = '\\text{I4a23}')

I4a31 = Parameter(name = 'I4a31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x1*ydo',
                  texname = '\\text{I4a31}')

I4a32 = Parameter(name = 'I4a32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x2*ys',
                  texname = '\\text{I4a32}')

I4a33 = Parameter(name = 'I4a33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKM3x3*yb',
                  texname = '\\text{I4a33}')

I5a1 = Parameter(name = 'I5a1',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yne',
                 texname = '\\text{I5a1}')

I5a2 = Parameter(name = 'I5a2',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ynmu',
                 texname = '\\text{I5a2}')

I5a3 = Parameter(name = 'I5a3',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yntau',
                 texname = '\\text{I5a3}')

I6a1 = Parameter(name = 'I6a1',
                 nature = 'internal',
                 type = 'complex',
                 value = '1',
                 texname = '\\text{I6a1}')

I6a2 = Parameter(name = 'I6a2',
                 nature = 'internal',
                 type = 'complex',
                 value = '1',
                 texname = '\\text{I6a2}')

I6a3 = Parameter(name = 'I6a3',
                 nature = 'internal',
                 type = 'complex',
                 value = '1',
                 texname = '\\text{I6a3}')

I7ax = Parameter(name = 'I7ax',
                 nature = 'internal',
                 type = 'complex',
                 value = 'UH1x1 + UH1x2 + UH1x3 + UH2x1 + UH2x2 + UH2x3 + UH3x1 + UH3x2 + UH3x3',
                 texname = '\\text{I7ax}')

I8a1 = Parameter(name = 'I8a1',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yne',
                 texname = '\\text{I8a1}')

I8a2 = Parameter(name = 'I8a2',
                 nature = 'internal',
                 type = 'complex',
                 value = 'ynmu',
                 texname = '\\text{I8a2}')

I8a3 = Parameter(name = 'I8a3',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yntau',
                 texname = '\\text{I8a3}')

I9a1 = Parameter(name = 'I9a1',
                 nature = 'internal',
                 type = 'complex',
                 value = '1',
                 texname = '\\text{I9a1}')

I9a2 = Parameter(name = 'I9a2',
                 nature = 'internal',
                 type = 'complex',
                 value = '1',
                 texname = '\\text{I9a2}')

I9a3 = Parameter(name = 'I9a3',
                 nature = 'internal',
                 type = 'complex',
                 value = '1',
                 texname = '\\text{I9a3}')

