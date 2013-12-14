# This file was automatically created by FeynRules $Revision: 634 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 28 Jul 2011 16:28:57



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

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

gSg = Parameter(name = 'gSg',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\text{gSg}',
                lhablock = 'FRBlock',
                lhacode = [ 1 ])

gVg = Parameter(name = 'gVg',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\text{gVg}',
                lhablock = 'FRBlock',
                lhacode = [ 2 ])

gTg = Parameter(name = 'gTg',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\text{gTg}',
                lhablock = 'FRBlock',
                lhacode = [ 3 ])

gPuR11 = Parameter(name = 'gPuR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPuR11}',
                   lhablock = 'FRBlock10',
                   lhacode = [ 1, 1 ])

gPuR12 = Parameter(name = 'gPuR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPuR12}',
                   lhablock = 'FRBlock10',
                   lhacode = [ 1, 2 ])

gPuR22 = Parameter(name = 'gPuR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPuR22}',
                   lhablock = 'FRBlock10',
                   lhacode = [ 2, 2 ])

hUlR11 = Parameter(name = 'hUlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlR11}',
                   lhablock = 'FRBlock100',
                   lhacode = [ 1, 1 ])

hUlR12 = Parameter(name = 'hUlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlR12}',
                   lhablock = 'FRBlock100',
                   lhacode = [ 1, 2 ])

hUlR21 = Parameter(name = 'hUlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlR21}',
                   lhablock = 'FRBlock100',
                   lhacode = [ 2, 1 ])

hUlR22 = Parameter(name = 'hUlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlR22}',
                   lhablock = 'FRBlock100',
                   lhacode = [ 2, 2 ])

hUlI11 = Parameter(name = 'hUlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlI11}',
                   lhablock = 'FRBlock101',
                   lhacode = [ 1, 1 ])

hUlI12 = Parameter(name = 'hUlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlI12}',
                   lhablock = 'FRBlock101',
                   lhacode = [ 1, 2 ])

hUlI21 = Parameter(name = 'hUlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlI21}',
                   lhablock = 'FRBlock101',
                   lhacode = [ 2, 1 ])

hUlI22 = Parameter(name = 'hUlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUlI22}',
                   lhablock = 'FRBlock101',
                   lhacode = [ 2, 2 ])

hYlR11 = Parameter(name = 'hYlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlR11}',
                   lhablock = 'FRBlock103',
                   lhacode = [ 1, 1 ])

hYlR12 = Parameter(name = 'hYlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlR12}',
                   lhablock = 'FRBlock103',
                   lhacode = [ 1, 2 ])

hYlR21 = Parameter(name = 'hYlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlR21}',
                   lhablock = 'FRBlock103',
                   lhacode = [ 2, 1 ])

hYlR22 = Parameter(name = 'hYlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlR22}',
                   lhablock = 'FRBlock103',
                   lhacode = [ 2, 2 ])

hYlI11 = Parameter(name = 'hYlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlI11}',
                   lhablock = 'FRBlock104',
                   lhacode = [ 1, 1 ])

hYlI12 = Parameter(name = 'hYlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlI12}',
                   lhablock = 'FRBlock104',
                   lhacode = [ 1, 2 ])

hYlI21 = Parameter(name = 'hYlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlI21}',
                   lhablock = 'FRBlock104',
                   lhacode = [ 2, 1 ])

hYlI22 = Parameter(name = 'hYlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYlI22}',
                   lhablock = 'FRBlock104',
                   lhacode = [ 2, 2 ])

hZlR11 = Parameter(name = 'hZlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlR11}',
                   lhablock = 'FRBlock106',
                   lhacode = [ 1, 1 ])

hZlR12 = Parameter(name = 'hZlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlR12}',
                   lhablock = 'FRBlock106',
                   lhacode = [ 1, 2 ])

hZlR21 = Parameter(name = 'hZlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlR21}',
                   lhablock = 'FRBlock106',
                   lhacode = [ 2, 1 ])

hZlR22 = Parameter(name = 'hZlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlR22}',
                   lhablock = 'FRBlock106',
                   lhacode = [ 2, 2 ])

hZlI11 = Parameter(name = 'hZlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlI11}',
                   lhablock = 'FRBlock107',
                   lhacode = [ 1, 1 ])

hZlI12 = Parameter(name = 'hZlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlI12}',
                   lhablock = 'FRBlock107',
                   lhacode = [ 1, 2 ])

hZlI21 = Parameter(name = 'hZlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlI21}',
                   lhablock = 'FRBlock107',
                   lhacode = [ 2, 1 ])

hZlI22 = Parameter(name = 'hZlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZlI22}',
                   lhablock = 'FRBlock107',
                   lhacode = [ 2, 2 ])

gPuI12 = Parameter(name = 'gPuI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPuI12}',
                   lhablock = 'FRBlock11',
                   lhacode = [ 1, 2 ])

gSdR11 = Parameter(name = 'gSdR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSdR11}',
                   lhablock = 'FRBlock13',
                   lhacode = [ 1, 1 ])

gSdR12 = Parameter(name = 'gSdR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSdR12}',
                   lhablock = 'FRBlock13',
                   lhacode = [ 1, 2 ])

gSdR22 = Parameter(name = 'gSdR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSdR22}',
                   lhablock = 'FRBlock13',
                   lhacode = [ 2, 2 ])

gSdI12 = Parameter(name = 'gSdI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSdI12}',
                   lhablock = 'FRBlock14',
                   lhacode = [ 1, 2 ])

gPdR11 = Parameter(name = 'gPdR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPdR11}',
                   lhablock = 'FRBlock16',
                   lhacode = [ 1, 1 ])

gPdR12 = Parameter(name = 'gPdR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPdR12}',
                   lhablock = 'FRBlock16',
                   lhacode = [ 1, 2 ])

gPdR22 = Parameter(name = 'gPdR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPdR22}',
                   lhablock = 'FRBlock16',
                   lhacode = [ 2, 2 ])

gPdI12 = Parameter(name = 'gPdI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPdI12}',
                   lhablock = 'FRBlock17',
                   lhacode = [ 1, 2 ])

gSlR11 = Parameter(name = 'gSlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSlR11}',
                   lhablock = 'FRBlock19',
                   lhacode = [ 1, 1 ])

gSlR12 = Parameter(name = 'gSlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSlR12}',
                   lhablock = 'FRBlock19',
                   lhacode = [ 1, 2 ])

gSlR22 = Parameter(name = 'gSlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSlR22}',
                   lhablock = 'FRBlock19',
                   lhacode = [ 2, 2 ])

gSlI12 = Parameter(name = 'gSlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSlI12}',
                   lhablock = 'FRBlock20',
                   lhacode = [ 1, 2 ])

gPlR11 = Parameter(name = 'gPlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPlR11}',
                   lhablock = 'FRBlock22',
                   lhacode = [ 1, 1 ])

gPlR12 = Parameter(name = 'gPlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPlR12}',
                   lhablock = 'FRBlock22',
                   lhacode = [ 1, 2 ])

gPlR22 = Parameter(name = 'gPlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPlR22}',
                   lhablock = 'FRBlock22',
                   lhacode = [ 2, 2 ])

gPlI12 = Parameter(name = 'gPlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gPlI12}',
                   lhablock = 'FRBlock23',
                   lhacode = [ 1, 2 ])

gVuR11 = Parameter(name = 'gVuR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVuR11}',
                   lhablock = 'FRBlock25',
                   lhacode = [ 1, 1 ])

gVuR12 = Parameter(name = 'gVuR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVuR12}',
                   lhablock = 'FRBlock25',
                   lhacode = [ 1, 2 ])

gVuR22 = Parameter(name = 'gVuR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVuR22}',
                   lhablock = 'FRBlock25',
                   lhacode = [ 2, 2 ])

gVuI12 = Parameter(name = 'gVuI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVuI12}',
                   lhablock = 'FRBlock26',
                   lhacode = [ 1, 2 ])

gAuR11 = Parameter(name = 'gAuR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAuR11}',
                   lhablock = 'FRBlock28',
                   lhacode = [ 1, 1 ])

gAuR12 = Parameter(name = 'gAuR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAuR12}',
                   lhablock = 'FRBlock28',
                   lhacode = [ 1, 2 ])

gAuR22 = Parameter(name = 'gAuR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAuR22}',
                   lhablock = 'FRBlock28',
                   lhacode = [ 2, 2 ])

gAuI12 = Parameter(name = 'gAuI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAuI12}',
                   lhablock = 'FRBlock29',
                   lhacode = [ 1, 2 ])

gVdR11 = Parameter(name = 'gVdR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVdR11}',
                   lhablock = 'FRBlock31',
                   lhacode = [ 1, 1 ])

gVdR12 = Parameter(name = 'gVdR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVdR12}',
                   lhablock = 'FRBlock31',
                   lhacode = [ 1, 2 ])

gVdR22 = Parameter(name = 'gVdR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVdR22}',
                   lhablock = 'FRBlock31',
                   lhacode = [ 2, 2 ])

gVdI12 = Parameter(name = 'gVdI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVdI12}',
                   lhablock = 'FRBlock32',
                   lhacode = [ 1, 2 ])

gAdR11 = Parameter(name = 'gAdR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAdR11}',
                   lhablock = 'FRBlock34',
                   lhacode = [ 1, 1 ])

gAdR12 = Parameter(name = 'gAdR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAdR12}',
                   lhablock = 'FRBlock34',
                   lhacode = [ 1, 2 ])

gAdR22 = Parameter(name = 'gAdR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAdR22}',
                   lhablock = 'FRBlock34',
                   lhacode = [ 2, 2 ])

gAdI12 = Parameter(name = 'gAdI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAdI12}',
                   lhablock = 'FRBlock35',
                   lhacode = [ 1, 2 ])

gVlR11 = Parameter(name = 'gVlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVlR11}',
                   lhablock = 'FRBlock37',
                   lhacode = [ 1, 1 ])

gVlR12 = Parameter(name = 'gVlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVlR12}',
                   lhablock = 'FRBlock37',
                   lhacode = [ 1, 2 ])

gVlR22 = Parameter(name = 'gVlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVlR22}',
                   lhablock = 'FRBlock37',
                   lhacode = [ 2, 2 ])

gVlI12 = Parameter(name = 'gVlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gVlI12}',
                   lhablock = 'FRBlock38',
                   lhacode = [ 1, 2 ])

gAlR11 = Parameter(name = 'gAlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAlR11}',
                   lhablock = 'FRBlock40',
                   lhacode = [ 1, 1 ])

gAlR12 = Parameter(name = 'gAlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAlR12}',
                   lhablock = 'FRBlock40',
                   lhacode = [ 1, 2 ])

gAlR22 = Parameter(name = 'gAlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAlR22}',
                   lhablock = 'FRBlock40',
                   lhacode = [ 2, 2 ])

gAlI12 = Parameter(name = 'gAlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gAlI12}',
                   lhablock = 'FRBlock41',
                   lhacode = [ 1, 2 ])

gTuR11 = Parameter(name = 'gTuR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuR11}',
                   lhablock = 'FRBlock43',
                   lhacode = [ 1, 1 ])

gTuR12 = Parameter(name = 'gTuR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuR12}',
                   lhablock = 'FRBlock43',
                   lhacode = [ 1, 2 ])

gTuR21 = Parameter(name = 'gTuR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuR21}',
                   lhablock = 'FRBlock43',
                   lhacode = [ 2, 1 ])

gTuR22 = Parameter(name = 'gTuR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuR22}',
                   lhablock = 'FRBlock43',
                   lhacode = [ 2, 2 ])

gTuI11 = Parameter(name = 'gTuI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuI11}',
                   lhablock = 'FRBlock44',
                   lhacode = [ 1, 1 ])

gTuI12 = Parameter(name = 'gTuI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuI12}',
                   lhablock = 'FRBlock44',
                   lhacode = [ 1, 2 ])

gTuI21 = Parameter(name = 'gTuI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuI21}',
                   lhablock = 'FRBlock44',
                   lhacode = [ 2, 1 ])

gTuI22 = Parameter(name = 'gTuI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTuI22}',
                   lhablock = 'FRBlock44',
                   lhacode = [ 2, 2 ])

gUuR11 = Parameter(name = 'gUuR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuR11}',
                   lhablock = 'FRBlock46',
                   lhacode = [ 1, 1 ])

gUuR12 = Parameter(name = 'gUuR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuR12}',
                   lhablock = 'FRBlock46',
                   lhacode = [ 1, 2 ])

gUuR21 = Parameter(name = 'gUuR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuR21}',
                   lhablock = 'FRBlock46',
                   lhacode = [ 2, 1 ])

gUuR22 = Parameter(name = 'gUuR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuR22}',
                   lhablock = 'FRBlock46',
                   lhacode = [ 2, 2 ])

gUuI11 = Parameter(name = 'gUuI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuI11}',
                   lhablock = 'FRBlock47',
                   lhacode = [ 1, 1 ])

gUuI12 = Parameter(name = 'gUuI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuI12}',
                   lhablock = 'FRBlock47',
                   lhacode = [ 1, 2 ])

gUuI21 = Parameter(name = 'gUuI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuI21}',
                   lhablock = 'FRBlock47',
                   lhacode = [ 2, 1 ])

gUuI22 = Parameter(name = 'gUuI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUuI22}',
                   lhablock = 'FRBlock47',
                   lhacode = [ 2, 2 ])

gTdR11 = Parameter(name = 'gTdR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdR11}',
                   lhablock = 'FRBlock49',
                   lhacode = [ 1, 1 ])

gTdR12 = Parameter(name = 'gTdR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdR12}',
                   lhablock = 'FRBlock49',
                   lhacode = [ 1, 2 ])

gTdR21 = Parameter(name = 'gTdR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdR21}',
                   lhablock = 'FRBlock49',
                   lhacode = [ 2, 1 ])

gTdR22 = Parameter(name = 'gTdR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdR22}',
                   lhablock = 'FRBlock49',
                   lhacode = [ 2, 2 ])

gTdI11 = Parameter(name = 'gTdI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdI11}',
                   lhablock = 'FRBlock50',
                   lhacode = [ 1, 1 ])

gTdI12 = Parameter(name = 'gTdI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdI12}',
                   lhablock = 'FRBlock50',
                   lhacode = [ 1, 2 ])

gTdI21 = Parameter(name = 'gTdI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdI21}',
                   lhablock = 'FRBlock50',
                   lhacode = [ 2, 1 ])

gTdI22 = Parameter(name = 'gTdI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTdI22}',
                   lhablock = 'FRBlock50',
                   lhacode = [ 2, 2 ])

gUdR11 = Parameter(name = 'gUdR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdR11}',
                   lhablock = 'FRBlock52',
                   lhacode = [ 1, 1 ])

gUdR12 = Parameter(name = 'gUdR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdR12}',
                   lhablock = 'FRBlock52',
                   lhacode = [ 1, 2 ])

gUdR21 = Parameter(name = 'gUdR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdR21}',
                   lhablock = 'FRBlock52',
                   lhacode = [ 2, 1 ])

gUdR22 = Parameter(name = 'gUdR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdR22}',
                   lhablock = 'FRBlock52',
                   lhacode = [ 2, 2 ])

gUdI11 = Parameter(name = 'gUdI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdI11}',
                   lhablock = 'FRBlock53',
                   lhacode = [ 1, 1 ])

gUdI12 = Parameter(name = 'gUdI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdI12}',
                   lhablock = 'FRBlock53',
                   lhacode = [ 1, 2 ])

gUdI21 = Parameter(name = 'gUdI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdI21}',
                   lhablock = 'FRBlock53',
                   lhacode = [ 2, 1 ])

gUdI22 = Parameter(name = 'gUdI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUdI22}',
                   lhablock = 'FRBlock53',
                   lhacode = [ 2, 2 ])

gTlR11 = Parameter(name = 'gTlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlR11}',
                   lhablock = 'FRBlock55',
                   lhacode = [ 1, 1 ])

gTlR12 = Parameter(name = 'gTlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlR12}',
                   lhablock = 'FRBlock55',
                   lhacode = [ 1, 2 ])

gTlR21 = Parameter(name = 'gTlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlR21}',
                   lhablock = 'FRBlock55',
                   lhacode = [ 2, 1 ])

gTlR22 = Parameter(name = 'gTlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlR22}',
                   lhablock = 'FRBlock55',
                   lhacode = [ 2, 2 ])

gTlI11 = Parameter(name = 'gTlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlI11}',
                   lhablock = 'FRBlock56',
                   lhacode = [ 1, 1 ])

gTlI12 = Parameter(name = 'gTlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlI12}',
                   lhablock = 'FRBlock56',
                   lhacode = [ 1, 2 ])

gTlI21 = Parameter(name = 'gTlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlI21}',
                   lhablock = 'FRBlock56',
                   lhacode = [ 2, 1 ])

gTlI22 = Parameter(name = 'gTlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gTlI22}',
                   lhablock = 'FRBlock56',
                   lhacode = [ 2, 2 ])

gUlR11 = Parameter(name = 'gUlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlR11}',
                   lhablock = 'FRBlock58',
                   lhacode = [ 1, 1 ])

gUlR12 = Parameter(name = 'gUlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlR12}',
                   lhablock = 'FRBlock58',
                   lhacode = [ 1, 2 ])

gUlR21 = Parameter(name = 'gUlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlR21}',
                   lhablock = 'FRBlock58',
                   lhacode = [ 2, 1 ])

gUlR22 = Parameter(name = 'gUlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlR22}',
                   lhablock = 'FRBlock58',
                   lhacode = [ 2, 2 ])

gUlI11 = Parameter(name = 'gUlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlI11}',
                   lhablock = 'FRBlock59',
                   lhacode = [ 1, 1 ])

gUlI12 = Parameter(name = 'gUlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlI12}',
                   lhablock = 'FRBlock59',
                   lhacode = [ 1, 2 ])

gUlI21 = Parameter(name = 'gUlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlI21}',
                   lhablock = 'FRBlock59',
                   lhacode = [ 2, 1 ])

gUlI22 = Parameter(name = 'gUlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gUlI22}',
                   lhablock = 'FRBlock59',
                   lhacode = [ 2, 2 ])

hSqR11 = Parameter(name = 'hSqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqR11}',
                   lhablock = 'FRBlock61',
                   lhacode = [ 1, 1 ])

hSqR12 = Parameter(name = 'hSqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqR12}',
                   lhablock = 'FRBlock61',
                   lhacode = [ 1, 2 ])

hSqR21 = Parameter(name = 'hSqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqR21}',
                   lhablock = 'FRBlock61',
                   lhacode = [ 2, 1 ])

hSqR22 = Parameter(name = 'hSqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqR22}',
                   lhablock = 'FRBlock61',
                   lhacode = [ 2, 2 ])

hSqI11 = Parameter(name = 'hSqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqI11}',
                   lhablock = 'FRBlock62',
                   lhacode = [ 1, 1 ])

hSqI12 = Parameter(name = 'hSqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqI12}',
                   lhablock = 'FRBlock62',
                   lhacode = [ 1, 2 ])

hSqI21 = Parameter(name = 'hSqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqI21}',
                   lhablock = 'FRBlock62',
                   lhacode = [ 2, 1 ])

hSqI22 = Parameter(name = 'hSqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSqI22}',
                   lhablock = 'FRBlock62',
                   lhacode = [ 2, 2 ])

hPqR11 = Parameter(name = 'hPqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqR11}',
                   lhablock = 'FRBlock64',
                   lhacode = [ 1, 1 ])

hPqR12 = Parameter(name = 'hPqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqR12}',
                   lhablock = 'FRBlock64',
                   lhacode = [ 1, 2 ])

hPqR21 = Parameter(name = 'hPqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqR21}',
                   lhablock = 'FRBlock64',
                   lhacode = [ 2, 1 ])

hPqR22 = Parameter(name = 'hPqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqR22}',
                   lhablock = 'FRBlock64',
                   lhacode = [ 2, 2 ])

hPqI11 = Parameter(name = 'hPqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqI11}',
                   lhablock = 'FRBlock65',
                   lhacode = [ 1, 1 ])

hPqI12 = Parameter(name = 'hPqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqI12}',
                   lhablock = 'FRBlock65',
                   lhacode = [ 1, 2 ])

hPqI21 = Parameter(name = 'hPqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqI21}',
                   lhablock = 'FRBlock65',
                   lhacode = [ 2, 1 ])

hPqI22 = Parameter(name = 'hPqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPqI22}',
                   lhablock = 'FRBlock65',
                   lhacode = [ 2, 2 ])

hSlR11 = Parameter(name = 'hSlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlR11}',
                   lhablock = 'FRBlock67',
                   lhacode = [ 1, 1 ])

hSlR12 = Parameter(name = 'hSlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlR12}',
                   lhablock = 'FRBlock67',
                   lhacode = [ 1, 2 ])

hSlR21 = Parameter(name = 'hSlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlR21}',
                   lhablock = 'FRBlock67',
                   lhacode = [ 2, 1 ])

hSlR22 = Parameter(name = 'hSlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlR22}',
                   lhablock = 'FRBlock67',
                   lhacode = [ 2, 2 ])

hSlI11 = Parameter(name = 'hSlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlI11}',
                   lhablock = 'FRBlock68',
                   lhacode = [ 1, 1 ])

hSlI12 = Parameter(name = 'hSlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlI12}',
                   lhablock = 'FRBlock68',
                   lhacode = [ 1, 2 ])

hSlI21 = Parameter(name = 'hSlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlI21}',
                   lhablock = 'FRBlock68',
                   lhacode = [ 2, 1 ])

hSlI22 = Parameter(name = 'hSlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hSlI22}',
                   lhablock = 'FRBlock68',
                   lhacode = [ 2, 2 ])

gSuR11 = Parameter(name = 'gSuR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSuR11}',
                   lhablock = 'FRBlock7',
                   lhacode = [ 1, 1 ])

gSuR12 = Parameter(name = 'gSuR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSuR12}',
                   lhablock = 'FRBlock7',
                   lhacode = [ 1, 2 ])

gSuR22 = Parameter(name = 'gSuR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSuR22}',
                   lhablock = 'FRBlock7',
                   lhacode = [ 2, 2 ])

hPlR11 = Parameter(name = 'hPlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlR11}',
                   lhablock = 'FRBlock70',
                   lhacode = [ 1, 1 ])

hPlR12 = Parameter(name = 'hPlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlR12}',
                   lhablock = 'FRBlock70',
                   lhacode = [ 1, 2 ])

hPlR21 = Parameter(name = 'hPlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlR21}',
                   lhablock = 'FRBlock70',
                   lhacode = [ 2, 1 ])

hPlR22 = Parameter(name = 'hPlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlR22}',
                   lhablock = 'FRBlock70',
                   lhacode = [ 2, 2 ])

hPlI11 = Parameter(name = 'hPlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlI11}',
                   lhablock = 'FRBlock71',
                   lhacode = [ 1, 1 ])

hPlI12 = Parameter(name = 'hPlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlI12}',
                   lhablock = 'FRBlock71',
                   lhacode = [ 1, 2 ])

hPlI21 = Parameter(name = 'hPlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlI21}',
                   lhablock = 'FRBlock71',
                   lhacode = [ 2, 1 ])

hPlI22 = Parameter(name = 'hPlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hPlI22}',
                   lhablock = 'FRBlock71',
                   lhacode = [ 2, 2 ])

hVqR11 = Parameter(name = 'hVqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqR11}',
                   lhablock = 'FRBlock73',
                   lhacode = [ 1, 1 ])

hVqR12 = Parameter(name = 'hVqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqR12}',
                   lhablock = 'FRBlock73',
                   lhacode = [ 1, 2 ])

hVqR21 = Parameter(name = 'hVqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqR21}',
                   lhablock = 'FRBlock73',
                   lhacode = [ 2, 1 ])

hVqR22 = Parameter(name = 'hVqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqR22}',
                   lhablock = 'FRBlock73',
                   lhacode = [ 2, 2 ])

hVqI11 = Parameter(name = 'hVqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqI11}',
                   lhablock = 'FRBlock74',
                   lhacode = [ 1, 1 ])

hVqI12 = Parameter(name = 'hVqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqI12}',
                   lhablock = 'FRBlock74',
                   lhacode = [ 1, 2 ])

hVqI21 = Parameter(name = 'hVqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqI21}',
                   lhablock = 'FRBlock74',
                   lhacode = [ 2, 1 ])

hVqI22 = Parameter(name = 'hVqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVqI22}',
                   lhablock = 'FRBlock74',
                   lhacode = [ 2, 2 ])

hAqR11 = Parameter(name = 'hAqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqR11}',
                   lhablock = 'FRBlock76',
                   lhacode = [ 1, 1 ])

hAqR12 = Parameter(name = 'hAqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqR12}',
                   lhablock = 'FRBlock76',
                   lhacode = [ 1, 2 ])

hAqR21 = Parameter(name = 'hAqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqR21}',
                   lhablock = 'FRBlock76',
                   lhacode = [ 2, 1 ])

hAqR22 = Parameter(name = 'hAqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqR22}',
                   lhablock = 'FRBlock76',
                   lhacode = [ 2, 2 ])

hAqI11 = Parameter(name = 'hAqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqI11}',
                   lhablock = 'FRBlock77',
                   lhacode = [ 1, 1 ])

hAqI12 = Parameter(name = 'hAqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqI12}',
                   lhablock = 'FRBlock77',
                   lhacode = [ 1, 2 ])

hAqI21 = Parameter(name = 'hAqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqI21}',
                   lhablock = 'FRBlock77',
                   lhacode = [ 2, 1 ])

hAqI22 = Parameter(name = 'hAqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAqI22}',
                   lhablock = 'FRBlock77',
                   lhacode = [ 2, 2 ])

hVlR11 = Parameter(name = 'hVlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlR11}',
                   lhablock = 'FRBlock79',
                   lhacode = [ 1, 1 ])

hVlR12 = Parameter(name = 'hVlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlR12}',
                   lhablock = 'FRBlock79',
                   lhacode = [ 1, 2 ])

hVlR21 = Parameter(name = 'hVlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlR21}',
                   lhablock = 'FRBlock79',
                   lhacode = [ 2, 1 ])

hVlR22 = Parameter(name = 'hVlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlR22}',
                   lhablock = 'FRBlock79',
                   lhacode = [ 2, 2 ])

gSuI12 = Parameter(name = 'gSuI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{gSuI12}',
                   lhablock = 'FRBlock8',
                   lhacode = [ 1, 2 ])

hVlI11 = Parameter(name = 'hVlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlI11}',
                   lhablock = 'FRBlock80',
                   lhacode = [ 1, 1 ])

hVlI12 = Parameter(name = 'hVlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlI12}',
                   lhablock = 'FRBlock80',
                   lhacode = [ 1, 2 ])

hVlI21 = Parameter(name = 'hVlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlI21}',
                   lhablock = 'FRBlock80',
                   lhacode = [ 2, 1 ])

hVlI22 = Parameter(name = 'hVlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hVlI22}',
                   lhablock = 'FRBlock80',
                   lhacode = [ 2, 2 ])

hAlR11 = Parameter(name = 'hAlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlR11}',
                   lhablock = 'FRBlock82',
                   lhacode = [ 1, 1 ])

hAlR12 = Parameter(name = 'hAlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlR12}',
                   lhablock = 'FRBlock82',
                   lhacode = [ 1, 2 ])

hAlR21 = Parameter(name = 'hAlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlR21}',
                   lhablock = 'FRBlock82',
                   lhacode = [ 2, 1 ])

hAlR22 = Parameter(name = 'hAlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlR22}',
                   lhablock = 'FRBlock82',
                   lhacode = [ 2, 2 ])

hAlI11 = Parameter(name = 'hAlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlI11}',
                   lhablock = 'FRBlock83',
                   lhacode = [ 1, 1 ])

hAlI12 = Parameter(name = 'hAlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlI12}',
                   lhablock = 'FRBlock83',
                   lhacode = [ 1, 2 ])

hAlI21 = Parameter(name = 'hAlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlI21}',
                   lhablock = 'FRBlock83',
                   lhacode = [ 2, 1 ])

hAlI22 = Parameter(name = 'hAlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hAlI22}',
                   lhablock = 'FRBlock83',
                   lhacode = [ 2, 2 ])

hTqR11 = Parameter(name = 'hTqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqR11}',
                   lhablock = 'FRBlock85',
                   lhacode = [ 1, 1 ])

hTqR12 = Parameter(name = 'hTqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqR12}',
                   lhablock = 'FRBlock85',
                   lhacode = [ 1, 2 ])

hTqR21 = Parameter(name = 'hTqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqR21}',
                   lhablock = 'FRBlock85',
                   lhacode = [ 2, 1 ])

hTqR22 = Parameter(name = 'hTqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqR22}',
                   lhablock = 'FRBlock85',
                   lhacode = [ 2, 2 ])

hTqI11 = Parameter(name = 'hTqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqI11}',
                   lhablock = 'FRBlock86',
                   lhacode = [ 1, 1 ])

hTqI12 = Parameter(name = 'hTqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqI12}',
                   lhablock = 'FRBlock86',
                   lhacode = [ 1, 2 ])

hTqI21 = Parameter(name = 'hTqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqI21}',
                   lhablock = 'FRBlock86',
                   lhacode = [ 2, 1 ])

hTqI22 = Parameter(name = 'hTqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTqI22}',
                   lhablock = 'FRBlock86',
                   lhacode = [ 2, 2 ])

hUqR11 = Parameter(name = 'hUqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqR11}',
                   lhablock = 'FRBlock88',
                   lhacode = [ 1, 1 ])

hUqR12 = Parameter(name = 'hUqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqR12}',
                   lhablock = 'FRBlock88',
                   lhacode = [ 1, 2 ])

hUqR21 = Parameter(name = 'hUqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqR21}',
                   lhablock = 'FRBlock88',
                   lhacode = [ 2, 1 ])

hUqR22 = Parameter(name = 'hUqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqR22}',
                   lhablock = 'FRBlock88',
                   lhacode = [ 2, 2 ])

hUqI11 = Parameter(name = 'hUqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqI11}',
                   lhablock = 'FRBlock89',
                   lhacode = [ 1, 1 ])

hUqI12 = Parameter(name = 'hUqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqI12}',
                   lhablock = 'FRBlock89',
                   lhacode = [ 1, 2 ])

hUqI21 = Parameter(name = 'hUqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqI21}',
                   lhablock = 'FRBlock89',
                   lhacode = [ 2, 1 ])

hUqI22 = Parameter(name = 'hUqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hUqI22}',
                   lhablock = 'FRBlock89',
                   lhacode = [ 2, 2 ])

hYqR11 = Parameter(name = 'hYqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqR11}',
                   lhablock = 'FRBlock91',
                   lhacode = [ 1, 1 ])

hYqR12 = Parameter(name = 'hYqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqR12}',
                   lhablock = 'FRBlock91',
                   lhacode = [ 1, 2 ])

hYqR21 = Parameter(name = 'hYqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqR21}',
                   lhablock = 'FRBlock91',
                   lhacode = [ 2, 1 ])

hYqR22 = Parameter(name = 'hYqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqR22}',
                   lhablock = 'FRBlock91',
                   lhacode = [ 2, 2 ])

hYqI11 = Parameter(name = 'hYqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqI11}',
                   lhablock = 'FRBlock92',
                   lhacode = [ 1, 1 ])

hYqI12 = Parameter(name = 'hYqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqI12}',
                   lhablock = 'FRBlock92',
                   lhacode = [ 1, 2 ])

hYqI21 = Parameter(name = 'hYqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqI21}',
                   lhablock = 'FRBlock92',
                   lhacode = [ 2, 1 ])

hYqI22 = Parameter(name = 'hYqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hYqI22}',
                   lhablock = 'FRBlock92',
                   lhacode = [ 2, 2 ])

hZqR11 = Parameter(name = 'hZqR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqR11}',
                   lhablock = 'FRBlock94',
                   lhacode = [ 1, 1 ])

hZqR12 = Parameter(name = 'hZqR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqR12}',
                   lhablock = 'FRBlock94',
                   lhacode = [ 1, 2 ])

hZqR21 = Parameter(name = 'hZqR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqR21}',
                   lhablock = 'FRBlock94',
                   lhacode = [ 2, 1 ])

hZqR22 = Parameter(name = 'hZqR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqR22}',
                   lhablock = 'FRBlock94',
                   lhacode = [ 2, 2 ])

hZqI11 = Parameter(name = 'hZqI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqI11}',
                   lhablock = 'FRBlock95',
                   lhacode = [ 1, 1 ])

hZqI12 = Parameter(name = 'hZqI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqI12}',
                   lhablock = 'FRBlock95',
                   lhacode = [ 1, 2 ])

hZqI21 = Parameter(name = 'hZqI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqI21}',
                   lhablock = 'FRBlock95',
                   lhacode = [ 2, 1 ])

hZqI22 = Parameter(name = 'hZqI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hZqI22}',
                   lhablock = 'FRBlock95',
                   lhacode = [ 2, 2 ])

hTlR11 = Parameter(name = 'hTlR11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlR11}',
                   lhablock = 'FRBlock97',
                   lhacode = [ 1, 1 ])

hTlR12 = Parameter(name = 'hTlR12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlR12}',
                   lhablock = 'FRBlock97',
                   lhacode = [ 1, 2 ])

hTlR21 = Parameter(name = 'hTlR21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlR21}',
                   lhablock = 'FRBlock97',
                   lhacode = [ 2, 1 ])

hTlR22 = Parameter(name = 'hTlR22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlR22}',
                   lhablock = 'FRBlock97',
                   lhacode = [ 2, 2 ])

hTlI11 = Parameter(name = 'hTlI11',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlI11}',
                   lhablock = 'FRBlock98',
                   lhacode = [ 1, 1 ])

hTlI12 = Parameter(name = 'hTlI12',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlI12}',
                   lhablock = 'FRBlock98',
                   lhacode = [ 1, 2 ])

hTlI21 = Parameter(name = 'hTlI21',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlI21}',
                   lhablock = 'FRBlock98',
                   lhacode = [ 2, 1 ])

hTlI22 = Parameter(name = 'hTlI22',
                   nature = 'external',
                   type = 'real',
                   value = 0,
                   texname = '\\text{hTlI22}',
                   lhablock = 'FRBlock98',
                   lhacode = [ 2, 2 ])

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
               value = 174.3,
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

MSV = Parameter(name = 'MSV',
                nature = 'external',
                type = 'real',
                value = 1000,
                texname = '\\text{MSV}',
                lhablock = 'MASS',
                lhacode = [ 9000006 ])

MSVP = Parameter(name = 'MSVP',
                 nature = 'external',
                 type = 'real',
                 value = 1000,
                 texname = '\\text{MSVP}',
                 lhablock = 'MASS',
                 lhacode = [ 9000007 ])

MVV = Parameter(name = 'MVV',
                nature = 'external',
                type = 'real',
                value = 1000,
                texname = '\\text{MVV}',
                lhablock = 'MASS',
                lhacode = [ 9000008 ])

MVVP = Parameter(name = 'MVVP',
                 nature = 'external',
                 type = 'real',
                 value = 1000,
                 texname = '\\text{MVVP}',
                 lhablock = 'MASS',
                 lhacode = [ 9000009 ])

MTV = Parameter(name = 'MTV',
                nature = 'external',
                type = 'real',
                value = 1000,
                texname = '\\text{MTV}',
                lhablock = 'MASS',
                lhacode = [ 9000010 ])

MTVP = Parameter(name = 'MTVP',
                 nature = 'external',
                 type = 'real',
                 value = 1000,
                 texname = '\\text{MTVP}',
                 lhablock = 'MASS',
                 lhacode = [ 9000011 ])

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

WSV = Parameter(name = 'WSV',
                nature = 'external',
                type = 'real',
                value = 20,
                texname = '\\text{WSV}',
                lhablock = 'DECAY',
                lhacode = [ 9000006 ])

WSVP = Parameter(name = 'WSVP',
                 nature = 'external',
                 type = 'real',
                 value = 20,
                 texname = '\\text{WSVP}',
                 lhablock = 'DECAY',
                 lhacode = [ 9000007 ])

WVV = Parameter(name = 'WVV',
                nature = 'external',
                type = 'real',
                value = 20,
                texname = '\\text{WVV}',
                lhablock = 'DECAY',
                lhacode = [ 9000008 ])

WVVP = Parameter(name = 'WVVP',
                 nature = 'external',
                 type = 'real',
                 value = 20,
                 texname = '\\text{WVVP}',
                 lhablock = 'DECAY',
                 lhacode = [ 9000009 ])

WTV = Parameter(name = 'WTV',
                nature = 'external',
                type = 'real',
                value = 20,
                texname = '\\text{WTV}',
                lhablock = 'DECAY',
                lhacode = [ 9000010 ])

WTVP = Parameter(name = 'WTVP',
                 nature = 'external',
                 type = 'real',
                 value = 20,
                 texname = '\\text{WTVP}',
                 lhablock = 'DECAY',
                 lhacode = [ 9000011 ])

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

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(lam*v**2)',
                texname = '\\mu')

