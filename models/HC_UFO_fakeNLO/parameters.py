# This file was automatically created by FeynRules 2.0.6
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Wed 11 Dec 2013 19:27:12



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

MU_R = Parameter(name = 'MU_R',
              nature = 'external',
              type = 'real',
              value = 91.188,
              texname = '\\text{\\mu_r}',
              lhablock = 'LOOP',
              lhacode = [ 1 ])

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

Lambda = Parameter(name = 'Lambda',
                   nature = 'external',
                   type = 'real',
                   value = 1000,
                   texname = '\\Lambda',
                   lhablock = 'FRBlock',
                   lhacode = [ 1 ])

ca = Parameter(name = 'ca',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = 'c_a',
               lhablock = 'FRBlock',
               lhacode = [ 2 ])

kSM = Parameter(name = 'kSM',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{SM}}',
                lhablock = 'FRBlock',
                lhacode = [ 3 ])

kHtt = Parameter(name = 'kHtt',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Htt}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 4 ])

kAtt = Parameter(name = 'kAtt',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Att}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 5 ])

kHbb = Parameter(name = 'kHbb',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Hbb}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 6 ])

kAbb = Parameter(name = 'kAbb',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Abb}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 7 ])

kHll = Parameter(name = 'kHll',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Hll}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 8 ])

kAll = Parameter(name = 'kAll',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{All}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 9 ])

kHaa = Parameter(name = 'kHaa',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Haa}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 10 ])

kAaa = Parameter(name = 'kAaa',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Aaa}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 11 ])

kHza = Parameter(name = 'kHza',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Hza}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 12 ])

kAza = Parameter(name = 'kAza',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Aza}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 13 ])

kHgg = Parameter(name = 'kHgg',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Hgg}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 14 ])

kAgg = Parameter(name = 'kAgg',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\kappa _{\\text{Agg}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 15 ])

kHzz = Parameter(name = 'kHzz',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = '\\kappa _{\\text{Hzz}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 16 ])

kAzz = Parameter(name = 'kAzz',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = '\\kappa _{\\text{Azz}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 17 ])

kHww = Parameter(name = 'kHww',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = '\\kappa _{\\text{Hww}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 18 ])

kAww = Parameter(name = 'kAww',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = '\\kappa _{\\text{Aww}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 19 ])

kHda = Parameter(name = 'kHda',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = '\\kappa _{\\text{Hda}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 20 ])

kHdz = Parameter(name = 'kHdz',
                 nature = 'external',
                 type = 'real',
                 value = 0,
                 texname = '\\kappa _{\\text{Hdz}}',
                 lhablock = 'FRBlock',
                 lhacode = [ 21 ])

kHdwR = Parameter(name = 'kHdwR',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\kappa _{\\text{HdwR}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 22 ])

kHdwI = Parameter(name = 'kHdwI',
                  nature = 'external',
                  type = 'real',
                  value = 0,
                  texname = '\\kappa _{\\text{HdwI}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 23 ])

kHHgg = Parameter(name = 'kHHgg',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\kappa _{\\text{HHgg}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 24 ])

kAAgg = Parameter(name = 'kAAgg',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\kappa _{\\text{AAgg}}',
                  lhablock = 'FRBlock',
                  lhacode = [ 25 ])

kqa = Parameter(name = 'kqa',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{qqa}}',
                lhablock = 'FRBlock',
                lhacode = [ 26 ])

kqb = Parameter(name = 'kqb',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{qqb}}',
                lhablock = 'FRBlock',
                lhacode = [ 27 ])

kla = Parameter(name = 'kla',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{lla}}',
                lhablock = 'FRBlock',
                lhacode = [ 28 ])

klb = Parameter(name = 'klb',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{llb}}',
                lhablock = 'FRBlock',
                lhacode = [ 29 ])

kw1 = Parameter(name = 'kw1',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{w1}}',
                lhablock = 'FRBlock',
                lhacode = [ 30 ])

kw2 = Parameter(name = 'kw2',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{w2}}',
                lhablock = 'FRBlock',
                lhacode = [ 31 ])

kw3 = Parameter(name = 'kw3',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\kappa _{\\text{w3}}',
                lhablock = 'FRBlock',
                lhacode = [ 32 ])

kw4 = Parameter(name = 'kw4',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\kappa _{\\text{w4}}',
                lhablock = 'FRBlock',
                lhacode = [ 33 ])

kw5 = Parameter(name = 'kw5',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\kappa _{\\text{w5}}',
                lhablock = 'FRBlock',
                lhacode = [ 34 ])

kz1 = Parameter(name = 'kz1',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\kappa _{\\text{z1}}',
                lhablock = 'FRBlock',
                lhacode = [ 35 ])

kz3 = Parameter(name = 'kz3',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{z3}}',
                lhablock = 'FRBlock',
                lhacode = [ 36 ])

kz5 = Parameter(name = 'kz5',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\kappa _{\\text{z5}}',
                lhablock = 'FRBlock',
                lhacode = [ 37 ])

kq = Parameter(name = 'kq',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = '\\kappa _q',
               lhablock = 'FRBlock',
               lhacode = [ 38 ])

kq3 = Parameter(name = 'kq3',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '\\kappa _{\\text{q3}}',
                lhablock = 'FRBlock',
                lhacode = [ 39 ])

kl = Parameter(name = 'kl',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = '\\kappa _l',
               lhablock = 'FRBlock',
               lhacode = [ 40 ])

kg = Parameter(name = 'kg',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = '\\kappa _g',
               lhablock = 'FRBlock',
               lhacode = [ 41 ])

ka = Parameter(name = 'ka',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = '\\kappa _a',
               lhablock = 'FRBlock',
               lhacode = [ 42 ])

kz = Parameter(name = 'kz',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = '\\kappa _z',
               lhablock = 'FRBlock',
               lhacode = [ 43 ])

kw = Parameter(name = 'kw',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = '\\kappa _w',
               lhablock = 'FRBlock',
               lhacode = [ 44 ])

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

MX0 = Parameter(name = 'MX0',
                nature = 'external',
                type = 'real',
                value = 125.,
                texname = '\\text{MX0}',
                lhablock = 'MASS',
                lhacode = [ 5000000 ])

MX1 = Parameter(name = 'MX1',
                nature = 'external',
                type = 'real',
                value = 125.,
                texname = '\\text{MX1}',
                lhablock = 'MASS',
                lhacode = [ 5000001 ])

MX2 = Parameter(name = 'MX2',
                nature = 'external',
                type = 'real',
                value = 125.,
                texname = '\\text{MX2}',
                lhablock = 'MASS',
                lhacode = [ 5000002 ])

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

WX0 = Parameter(name = 'WX0',
                nature = 'external',
                type = 'real',
                value = 0.00407,
                texname = '\\text{WX0}',
                lhablock = 'DECAY',
                lhacode = [ 5000000 ])

WX1 = Parameter(name = 'WX1',
                nature = 'external',
                type = 'real',
                value = 0.00407,
                texname = '\\text{WX1}',
                lhablock = 'DECAY',
                lhacode = [ 5000001 ])

WX2 = Parameter(name = 'WX2',
                nature = 'external',
                type = 'real',
                value = 0.00407,
                texname = '\\text{WX2}',
                lhablock = 'DECAY',
                lhacode = [ 5000002 ])

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

sa = Parameter(name = 'sa',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - ca**2)',
               texname = 's_a')

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

kHdw = Parameter(name = 'kHdw',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*kHdwI + kHdwR',
                 texname = '\\kappa _{\\text{Hdw}}')

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

ad = Parameter(name = 'ad',
               nature = 'internal',
               type = 'real',
               value = '(ee*(-0.5 + (2*sw2)/3.))/(2.*cw*sw)',
               texname = 'a_d')

al = Parameter(name = 'al',
               nature = 'internal',
               type = 'real',
               value = '(ee*(-0.5 + 2*sw2))/(2.*cw*sw)',
               texname = 'a_l')

an = Parameter(name = 'an',
               nature = 'internal',
               type = 'real',
               value = 'ee/(4.*cw*sw)',
               texname = '\\text{an}')

au = Parameter(name = 'au',
               nature = 'internal',
               type = 'real',
               value = '(ee*(0.5 - (4*sw2)/3.))/(2.*cw*sw)',
               texname = '\\text{au}')

bd = Parameter(name = 'bd',
               nature = 'internal',
               type = 'real',
               value = '-ee/(4.*cw*sw)',
               texname = 'b_d')

bl = Parameter(name = 'bl',
               nature = 'internal',
               type = 'real',
               value = '-ee/(4.*cw*sw)',
               texname = 'b_l')

bn = Parameter(name = 'bn',
               nature = 'internal',
               type = 'real',
               value = 'ee/(4.*cw*sw)',
               texname = 'b_n')

bu = Parameter(name = 'bu',
               nature = 'internal',
               type = 'real',
               value = 'ee/(4.*cw*sw)',
               texname = 'b_u')

gwwz = Parameter(name = 'gwwz',
                 nature = 'internal',
                 type = 'real',
                 value = '-((cw*ee)/sw)',
                 texname = 'g_{\\text{wwz}}')

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

gAaa = Parameter(name = 'gAaa',
                 nature = 'internal',
                 type = 'real',
                 value = 'ee**2/(3.*cmath.pi**2*vev)',
                 texname = 'g_{\\text{Aaa}}')

gAAgg = Parameter(name = 'gAAgg',
                  nature = 'internal',
                  type = 'real',
                  value = 'G**2/(8.*cmath.pi**2*vev**2)',
                  texname = 'g_{\\text{AAgg}}')

gAgg = Parameter(name = 'gAgg',
                 nature = 'internal',
                 type = 'real',
                 value = 'G**2/(8.*cmath.pi**2*vev)',
                 texname = 'g_{\\text{Agg}}')

gAza = Parameter(name = 'gAza',
                 nature = 'internal',
                 type = 'real',
                 value = '((-5 + 8*cw**2)*cmath.sqrt(ee**2*Gf*MZ**2))/(6.*2**0.75*cmath.pi**2*vev)',
                 texname = 'g_{\\text{Aza}}')

gHaa = Parameter(name = 'gHaa',
                 nature = 'internal',
                 type = 'real',
                 value = '(47*ee**2)/(72.*cmath.pi**2*vev)',
                 texname = 'g_{\\text{Haa}}')

gHgg = Parameter(name = 'gHgg',
                 nature = 'internal',
                 type = 'real',
                 value = '-G**2/(12.*cmath.pi**2*vev)',
                 texname = 'g_{\\text{Hgg}}')

gHHgg = Parameter(name = 'gHHgg',
                  nature = 'internal',
                  type = 'real',
                  value = 'G**2/(12.*cmath.pi**2*vev**2)',
                  texname = 'g_{\\text{HHgg}}')

gHza = Parameter(name = 'gHza',
                 nature = 'internal',
                 type = 'real',
                 value = '((-13 + 94*cw**2)*cmath.sqrt(ee**2*Gf*MZ**2))/(36.*2**0.75*cmath.pi**2*vev)',
                 texname = 'g_{\\text{Hza}}')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = 'MX0**2/(2.*vev**2)',
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

