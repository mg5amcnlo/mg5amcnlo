# This file was automatically created by FeynRules 1.7.69
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Mon 1 Oct 2012 14:58:25


from __future__ import absolute_import
from .object_library import all_parameters, Parameter

from .function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name='ZERO',
                 nature='internal',
                 type='real',
                 value=0.0,
                 texname='0')

# User-defined parameters.
aEWM1 = Parameter(name='aEWM1',
                  nature='external',
                  type='real',
                  value=132.50698,
                  texname='\\text{aEWM1}',
                  lhablock='SMINPUTS',
                  lhacode=[1])

Gf = Parameter(name='Gf',
               nature='external',
               type='real',
               value=0.0000116639,
               texname='G_f',
               lhablock='SMINPUTS',
               lhacode=[2])

aS = Parameter(name='aS',
               nature='external',
               type='real',
               value=0.118,
               texname='\\alpha _s',
               lhablock='SMINPUTS',
               lhacode=[3])

lamWS = Parameter(name='lamWS',
                  nature='external',
                  type='real',
                  value=0.2253,
                  texname='\\text{lamWS}',
                  lhablock='Wolfenstein',
                  lhacode=[1])

AWS = Parameter(name='AWS',
                nature='external',
                type='real',
                value=0.808,
                texname='\\text{AWS}',
                lhablock='Wolfenstein',
                lhacode=[2])

rhoWS = Parameter(name='rhoWS',
                  nature='external',
                  type='real',
                  value=0.132,
                  texname='\\text{rhoWS}',
                  lhablock='Wolfenstein',
                  lhacode=[3])

etaWS = Parameter(name='etaWS',
                  nature='external',
                  type='real',
                  value=0.341,
                  texname='\\text{etaWS}',
                  lhablock='Wolfenstein',
                  lhacode=[4])

ymc = Parameter(name='ymc',
                nature='external',
                type='real',
                value=1.27,
                texname='\\text{ymc}',
                lhablock='YUKAWA',
                lhacode=[4])

ymb = Parameter(name='ymb',
                nature='external',
                type='real',
                value=4.2,
                texname='\\text{ymb}',
                lhablock='YUKAWA',
                lhacode=[5])

ymt = Parameter(name='ymt',
                nature='external',
                type='real',
                value=164.5,
                texname='\\text{ymt}',
                lhablock='YUKAWA',
                lhacode=[6])

yme = Parameter(name='yme',
                nature='external',
                type='real',
                value=0.000511,
                texname='\\text{yme}',
                lhablock='YUKAWA',
                lhacode=[11])

ymm = Parameter(name='ymm',
                nature='external',
                type='real',
                value=0.10566,
                texname='\\text{ymm}',
                lhablock='YUKAWA',
                lhacode=[13])

ymtau = Parameter(name='ymtau',
                  nature='external',
                  type='real',
                  value=1.777,
                  texname='\\text{ymtau}',
                  lhablock='YUKAWA',
                  lhacode=[15])

MZ = Parameter(name='MZ',
               nature='external',
               type='real',
               value=91.188,
               texname='\\text{MZ}',
               lhablock='MASS',
               lhacode=[23])

MC = Parameter(name='MC',
               nature='external',
               type='real',
               value=1.27,
               texname='\\text{MC}',
               lhablock='MASS',
               lhacode=[4])

MT = Parameter(name='MT',
               nature='external',
               type='real',
               value=173.,
               texname='\\text{MT}',
               lhablock='MASS',
               lhacode=[6])

MB = Parameter(name='MB',
               nature='external',
               type='real',
               value=4.7,
               texname='\\text{MB}',
               lhablock='MASS',
               lhacode=[5])

MH = Parameter(name='MH',
               nature='external',
               type='real',
               value=125.,
               texname='\\text{MH}',
               lhablock='MASS',
               lhacode=[25])

Me = Parameter(name='Me',
               nature='external',
               type='real',
               value=0.000511,
               texname='\\text{Me}',
               lhablock='MASS',
               lhacode=[11])

MM = Parameter(name='MM',
               nature='external',
               type='real',
               value=0.10566,
               texname='\\text{MM}',
               lhablock='MASS',
               lhacode=[13])

MTA = Parameter(name='MTA',
                nature='external',
                type='real',
                value=1.777,
                texname='\\text{MTA}',
                lhablock='MASS',
                lhacode=[15])

WZ = Parameter(name='WZ',
               nature='external',
               type='real',
               value=2.44140351,
               texname='\\text{WZ}',
               lhablock='DECAY',
               lhacode=[23])

WW = Parameter(name='WW',
               nature='external',
               type='real',
               value=2.04759951,
               texname='\\text{WW}',
               lhablock='DECAY',
               lhacode=[24])

WT = Parameter(name='WT',
               nature='external',
               type='real',
               value=1.50833649,
               texname='\\text{WT}',
               lhablock='DECAY',
               lhacode=[6])

WH = Parameter(name='WH',
               nature='external',
               type='real',
               value=6.38233934e-03,
               texname='\\text{WH}',
               lhablock='DECAY',
               lhacode=[25])

WTau = Parameter(name='WTau',
                 nature='external',
                 type='real',
                 value=2.27e-12,
                 texname='\\text{WTau}',
                 lhablock='DECAY',
                 lhacode=[15])

CKM1x1 = Parameter(name='CKM1x1',
                   nature='internal',
                   type='complex',
                   value='1 - lamWS**2/2.',
                   texname='\\text{CKM1x1}')

CKM1x2 = Parameter(name='CKM1x2',
                   nature='internal',
                   type='complex',
                   value='lamWS',
                   texname='\\text{CKM1x2}')

CKM1x3 = Parameter(name='CKM1x3',
                   nature='internal',
                   type='complex',
                   value='AWS*lamWS**3*(-(etaWS*complex(0,1)) + rhoWS)',
                   texname='\\text{CKM1x3}')

CKM2x1 = Parameter(name='CKM2x1',
                   nature='internal',
                   type='complex',
                   value='-lamWS',
                   texname='\\text{CKM2x1}')

CKM2x2 = Parameter(name='CKM2x2',
                   nature='internal',
                   type='complex',
                   value='1 - lamWS**2/2.',
                   texname='\\text{CKM2x2}')

CKM2x3 = Parameter(name='CKM2x3',
                   nature='internal',
                   type='complex',
                   value='AWS*lamWS**2',
                   texname='\\text{CKM2x3}')

CKM3x1 = Parameter(name='CKM3x1',
                   nature='internal',
                   type='complex',
                   value='AWS*lamWS**3*(1 - etaWS*complex(0,1) - rhoWS)',
                   texname='\\text{CKM3x1}')

CKM3x2 = Parameter(name='CKM3x2',
                   nature='internal',
                   type='complex',
                   value='-(AWS*lamWS**2)',
                   texname='\\text{CKM3x2}')

CKM3x3 = Parameter(name='CKM3x3',
                   nature='internal',
                   type='complex',
                   value='1',
                   texname='\\text{CKM3x3}')

aEW = Parameter(name='aEW',
                nature='internal',
                type='real',
                value='1/aEWM1',
                texname='\\alpha _{\\text{EW}}')

G = Parameter(name='G',
              nature='internal',
              type='real',
              value='2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname='G')

MW = Parameter(name='MW',
               nature='internal',
               type='real',
               value='cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
               texname='M_W')

ee = Parameter(name='ee',
               nature='internal',
               type='real',
               value='2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname='e')

sw2 = Parameter(name='sw2',
                nature='internal',
                type='real',
                value='1 - MW**2/MZ**2',
                texname='\\text{sw2}')

cw = Parameter(name='cw',
               nature='internal',
               type='real',
               value='cmath.sqrt(1 - sw2)',
               texname='c_w')

sw = Parameter(name='sw',
               nature='internal',
               type='real',
               value='cmath.sqrt(sw2)',
               texname='s_w')

g1 = Parameter(name='g1',
               nature='internal',
               type='real',
               value='ee/cw',
               texname='g_1')

gw = Parameter(name='gw',
               nature='internal',
               type='real',
               value='ee/sw',
               texname='g_w')

vev = Parameter(name='vev',
                nature='internal',
                type='real',
                value='(2*MW*sw)/ee',
                texname='\\text{vev}')

lam = Parameter(name='lam',
                nature='internal',
                type='real',
                value='MH**2/(2.*vev**2)',
                texname='\\text{lam}')

yb = Parameter(name='yb',
               nature='internal',
               type='real',
               value='(ymb*cmath.sqrt(2))/vev',
               texname='\\text{yb}')

yc = Parameter(name='yc',
               nature='internal',
               type='real',
               value='(ymc*cmath.sqrt(2))/vev',
               texname='\\text{yc}')

ye = Parameter(name='ye',
               nature='internal',
               type='real',
               value='(yme*cmath.sqrt(2))/vev',
               texname='\\text{ye}')

ym = Parameter(name='ym',
               nature='internal',
               type='real',
               value='(ymm*cmath.sqrt(2))/vev',
               texname='\\text{ym}')

yt = Parameter(name='yt',
               nature='internal',
               type='real',
               value='(ymt*cmath.sqrt(2))/vev',
               texname='\\text{yt}')

ytau = Parameter(name='ytau',
                 nature='internal',
                 type='real',
                 value='(ymtau*cmath.sqrt(2))/vev',
                 texname='\\text{ytau}')

muH = Parameter(name='muH',
                nature='internal',
                type='real',
                value='cmath.sqrt(lam*vev**2)',
                texname='\\mu')

I1x31 = Parameter(name='I1x31',
                  nature='internal',
                  type='complex',
                  value='yb*complexconjugate(CKM1x3)',
                  texname='\\text{I1x31}')

I1x32 = Parameter(name='I1x32',
                  nature='internal',
                  type='complex',
                  value='yb*complexconjugate(CKM2x3)',
                  texname='\\text{I1x32}')

I1x33 = Parameter(name='I1x33',
                  nature='internal',
                  type='complex',
                  value='yb*complexconjugate(CKM3x3)',
                  texname='\\text{I1x33}')

I2x12 = Parameter(name='I2x12',
                  nature='internal',
                  type='complex',
                  value='yc*complexconjugate(CKM2x1)',
                  texname='\\text{I2x12}')

I2x13 = Parameter(name='I2x13',
                  nature='internal',
                  type='complex',
                  value='yt*complexconjugate(CKM3x1)',
                  texname='\\text{I2x13}')

I2x22 = Parameter(name='I2x22',
                  nature='internal',
                  type='complex',
                  value='yc*complexconjugate(CKM2x2)',
                  texname='\\text{I2x22}')

I2x23 = Parameter(name='I2x23',
                  nature='internal',
                  type='complex',
                  value='yt*complexconjugate(CKM3x2)',
                  texname='\\text{I2x23}')

I2x32 = Parameter(name='I2x32',
                  nature='internal',
                  type='complex',
                  value='yc*complexconjugate(CKM2x3)',
                  texname='\\text{I2x32}')

I2x33 = Parameter(name='I2x33',
                  nature='internal',
                  type='complex',
                  value='yt*complexconjugate(CKM3x3)',
                  texname='\\text{I2x33}')

I3x21 = Parameter(name='I3x21',
                  nature='internal',
                  type='complex',
                  value='CKM2x1*yc',
                  texname='\\text{I3x21}')

I3x22 = Parameter(name='I3x22',
                  nature='internal',
                  type='complex',
                  value='CKM2x2*yc',
                  texname='\\text{I3x22}')

I3x23 = Parameter(name='I3x23',
                  nature='internal',
                  type='complex',
                  value='CKM2x3*yc',
                  texname='\\text{I3x23}')

I3x31 = Parameter(name='I3x31',
                  nature='internal',
                  type='complex',
                  value='CKM3x1*yt',
                  texname='\\text{I3x31}')

I3x32 = Parameter(name='I3x32',
                  nature='internal',
                  type='complex',
                  value='CKM3x2*yt',
                  texname='\\text{I3x32}')

I3x33 = Parameter(name='I3x33',
                  nature='internal',
                  type='complex',
                  value='CKM3x3*yt',
                  texname='\\text{I3x33}')

I4x13 = Parameter(name='I4x13',
                  nature='internal',
                  type='complex',
                  value='CKM1x3*yb',
                  texname='\\text{I4x13}')

I4x23 = Parameter(name='I4x23',
                  nature='internal',
                  type='complex',
                  value='CKM2x3*yb',
                  texname='\\text{I4x23}')

I4x33 = Parameter(name='I4x33',
                  nature='internal',
                  type='complex',
                  value='CKM3x3*yb',
                  texname='\\text{I4x33}')
