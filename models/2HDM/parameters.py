# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Mon 11 Apr 2011 22:27:17



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
CKMI11 = Parameter(name = 'CKMI11',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI11}',
                   lhablock = 'CKMI',
                   lhacode = [ 1, 1 ])

CKMI12 = Parameter(name = 'CKMI12',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI12}',
                   lhablock = 'CKMI',
                   lhacode = [ 1, 2 ])

CKMI13 = Parameter(name = 'CKMI13',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI13}',
                   lhablock = 'CKMI',
                   lhacode = [ 1, 3 ])

CKMI21 = Parameter(name = 'CKMI21',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI21}',
                   lhablock = 'CKMI',
                   lhacode = [ 2, 1 ])

CKMI22 = Parameter(name = 'CKMI22',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI22}',
                   lhablock = 'CKMI',
                   lhacode = [ 2, 2 ])

CKMI23 = Parameter(name = 'CKMI23',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI23}',
                   lhablock = 'CKMI',
                   lhacode = [ 2, 3 ])

CKMI31 = Parameter(name = 'CKMI31',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI31}',
                   lhablock = 'CKMI',
                   lhacode = [ 3, 1 ])

CKMI32 = Parameter(name = 'CKMI32',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI32}',
                   lhablock = 'CKMI',
                   lhacode = [ 3, 2 ])

CKMI33 = Parameter(name = 'CKMI33',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMI33}',
                   lhablock = 'CKMI',
                   lhacode = [ 3, 3 ])

CKMR11 = Parameter(name = 'CKMR11',
                   nature = 'external',
                   type = 'real',
                   value = 0.974589144,
                   texname = '\\text{CKMR11}',
                   lhablock = 'CKMR',
                   lhacode = [ 1, 1 ])

CKMR12 = Parameter(name = 'CKMR12',
                   nature = 'external',
                   type = 'real',
                   value = 0.224,
                   texname = '\\text{CKMR12}',
                   lhablock = 'CKMR',
                   lhacode = [ 1, 2 ])

CKMR13 = Parameter(name = 'CKMR13',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMR13}',
                   lhablock = 'CKMR',
                   lhacode = [ 1, 3 ])

CKMR21 = Parameter(name = 'CKMR21',
                   nature = 'external',
                   type = 'real',
                   value = -0.224,
                   texname = '\\text{CKMR21}',
                   lhablock = 'CKMR',
                   lhacode = [ 2, 1 ])

CKMR22 = Parameter(name = 'CKMR22',
                   nature = 'external',
                   type = 'real',
                   value = 0.974589144,
                   texname = '\\text{CKMR22}',
                   lhablock = 'CKMR',
                   lhacode = [ 2, 2 ])

CKMR23 = Parameter(name = 'CKMR23',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMR23}',
                   lhablock = 'CKMR',
                   lhacode = [ 2, 3 ])

CKMR31 = Parameter(name = 'CKMR31',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMR31}',
                   lhablock = 'CKMR',
                   lhacode = [ 3, 1 ])

CKMR32 = Parameter(name = 'CKMR32',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{CKMR32}',
                   lhablock = 'CKMR',
                   lhacode = [ 3, 2 ])

CKMR33 = Parameter(name = 'CKMR33',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{CKMR33}',
                   lhablock = 'CKMR',
                   lhacode = [ 3, 3 ])

l1 = Parameter(name = 'l1',
               nature = 'external',
               type = 'real',
               value = 1.,
               texname = '\\text{l1}',
               lhablock = 'Higgs',
               lhacode = [ 1 ])

l2 = Parameter(name = 'l2',
               nature = 'external',
               type = 'real',
               value = 1.,
               texname = '\\text{l2}',
               lhablock = 'Higgs',
               lhacode = [ 2 ])

l3 = Parameter(name = 'l3',
               nature = 'external',
               type = 'real',
               value = 1.,
               texname = '\\text{l3}',
               lhablock = 'Higgs',
               lhacode = [ 3 ])

l4 = Parameter(name = 'l4',
               nature = 'external',
               type = 'real',
               value = 0.5,
               texname = '\\text{l4}',
               lhablock = 'Higgs',
               lhacode = [ 4 ])

l5 = Parameter(name = 'l5',
               nature = 'external',
               type = 'real',
               value = 0.4,
               texname = '\\text{l5}',
               lhablock = 'Higgs',
               lhacode = [ 5 ])

lR6 = Parameter(name = 'lR6',
                nature = 'external',
                type = 'real',
                value = 0.3,
                texname = '\\text{lR6}',
                lhablock = 'Higgs',
                lhacode = [ 6 ])

lI6 = Parameter(name = 'lI6',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = '\\text{lI6}',
                lhablock = 'Higgs',
                lhacode = [ 7 ])

lR7 = Parameter(name = 'lR7',
                nature = 'external',
                type = 'real',
                value = 0.2,
                texname = '\\text{lR7}',
                lhablock = 'Higgs',
                lhacode = [ 8 ])

lI7 = Parameter(name = 'lI7',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = '\\text{lI7}',
                lhablock = 'Higgs',
                lhacode = [ 9 ])

TH11 = Parameter(name = 'TH11',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{TH11}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 1, 1 ])

TH12 = Parameter(name = 'TH12',
                 nature = 'external',
                 type = 'real',
                 value = 0.78064408782535,
                 texname = '\\text{TH12}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 1, 2 ])

TH13 = Parameter(name = 'TH13',
                 nature = 'external',
                 type = 'real',
                 value = 0.62497584604793,
                 texname = '\\text{TH13}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 1, 3 ])

TH21 = Parameter(name = 'TH21',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{TH21}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 2, 1 ])

TH22 = Parameter(name = 'TH22',
                 nature = 'external',
                 type = 'real',
                 value = -0.62497584604793,
                 texname = '\\text{TH22}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 2, 2 ])

TH23 = Parameter(name = 'TH23',
                 nature = 'external',
                 type = 'real',
                 value = 0.78064408782535,
                 texname = '\\text{TH23}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 2, 3 ])

TH31 = Parameter(name = 'TH31',
                 nature = 'external',
                 type = 'real',
                 value = 1.,
                 texname = '\\text{TH31}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 3, 1 ])

TH32 = Parameter(name = 'TH32',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{TH32}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 3, 2 ])

TH33 = Parameter(name = 'TH33',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{TH33}',
                 lhablock = 'HiggsMix',
                 lhacode = [ 3, 3 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.934,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.0000116637,
               texname = '\\text{Gf}',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1172,
               texname = '\\text{aS}',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

yukd1 = Parameter(name = 'yukd1',
                  nature = 'external',
                  type = 'complex',
                  value = 0.,
                  texname = '\\text{yukd1}',
                  lhablock = 'YUKAWAD',
                  lhacode = [ 1 ])

yukd2 = Parameter(name = 'yukd2',
                  nature = 'external',
                  type = 'complex',
                  value = 0.,
                  texname = '\\text{yukd2}',
                  lhablock = 'YUKAWAD',
                  lhacode = [ 2 ])

yukd3 = Parameter(name = 'yukd3',
                  nature = 'external',
                  type = 'complex',
                  value = 3.,
                  texname = '\\text{yukd3}',
                  lhablock = 'YUKAWAD',
                  lhacode = [ 3 ])

GDI11 = Parameter(name = 'GDI11',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI11}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 1, 1 ])

GDI12 = Parameter(name = 'GDI12',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI12}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 1, 2 ])

GDI13 = Parameter(name = 'GDI13',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI13}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 1, 3 ])

GDI21 = Parameter(name = 'GDI21',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI21}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 2, 1 ])

GDI22 = Parameter(name = 'GDI22',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI22}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 2, 2 ])

GDI23 = Parameter(name = 'GDI23',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI23}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 2, 3 ])

GDI31 = Parameter(name = 'GDI31',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI31}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 3, 1 ])

GDI32 = Parameter(name = 'GDI32',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI32}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 3, 2 ])

GDI33 = Parameter(name = 'GDI33',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDI33}',
                  lhablock = 'YukawaGDI',
                  lhacode = [ 3, 3 ])

GDR11 = Parameter(name = 'GDR11',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDR11}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 1, 1 ])

GDR12 = Parameter(name = 'GDR12',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDR12}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 1, 2 ])

GDR13 = Parameter(name = 'GDR13',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDR13}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 1, 3 ])

GDR21 = Parameter(name = 'GDR21',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDR21}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 2, 1 ])

GDR22 = Parameter(name = 'GDR22',
                  nature = 'external',
                  type = 'real',
                  value = 0.4,
                  texname = '\\text{GDR22}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 2, 2 ])

GDR23 = Parameter(name = 'GDR23',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDR23}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 2, 3 ])

GDR31 = Parameter(name = 'GDR31',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GDR31}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 3, 1 ])

GDR32 = Parameter(name = 'GDR32',
                  nature = 'external',
                  type = 'real',
                  value = 0.2,
                  texname = '\\text{GDR32}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 3, 2 ])

GDR33 = Parameter(name = 'GDR33',
                  nature = 'external',
                  type = 'real',
                  value = 5.,
                  texname = '\\text{GDR33}',
                  lhablock = 'YukawaGDR',
                  lhacode = [ 3, 3 ])

GLI11 = Parameter(name = 'GLI11',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI11}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 1, 1 ])

GLI12 = Parameter(name = 'GLI12',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI12}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 1, 2 ])

GLI13 = Parameter(name = 'GLI13',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI13}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 1, 3 ])

GLI21 = Parameter(name = 'GLI21',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI21}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 2, 1 ])

GLI22 = Parameter(name = 'GLI22',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI22}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 2, 2 ])

GLI23 = Parameter(name = 'GLI23',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI23}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 2, 3 ])

GLI31 = Parameter(name = 'GLI31',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI31}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 3, 1 ])

GLI32 = Parameter(name = 'GLI32',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI32}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 3, 2 ])

GLI33 = Parameter(name = 'GLI33',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLI33}',
                  lhablock = 'YukawaGLI',
                  lhacode = [ 3, 3 ])

GLR11 = Parameter(name = 'GLR11',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLR11}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 1, 1 ])

GLR12 = Parameter(name = 'GLR12',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLR12}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 1, 2 ])

GLR13 = Parameter(name = 'GLR13',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLR13}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 1, 3 ])

GLR21 = Parameter(name = 'GLR21',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLR21}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 2, 1 ])

GLR22 = Parameter(name = 'GLR22',
                  nature = 'external',
                  type = 'real',
                  value = 0.1,
                  texname = '\\text{GLR22}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 2, 2 ])

GLR23 = Parameter(name = 'GLR23',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLR23}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 2, 3 ])

GLR31 = Parameter(name = 'GLR31',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GLR31}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 3, 1 ])

GLR32 = Parameter(name = 'GLR32',
                  nature = 'external',
                  type = 'real',
                  value = 0.5,
                  texname = '\\text{GLR32}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 3, 2 ])

GLR33 = Parameter(name = 'GLR33',
                  nature = 'external',
                  type = 'real',
                  value = 3.,
                  texname = '\\text{GLR33}',
                  lhablock = 'YukawaGLR',
                  lhacode = [ 3, 3 ])

GUI11 = Parameter(name = 'GUI11',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI11}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 1, 1 ])

GUI12 = Parameter(name = 'GUI12',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI12}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 1, 2 ])

GUI13 = Parameter(name = 'GUI13',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI13}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 1, 3 ])

GUI21 = Parameter(name = 'GUI21',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI21}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 2, 1 ])

GUI22 = Parameter(name = 'GUI22',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI22}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 2, 2 ])

GUI23 = Parameter(name = 'GUI23',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI23}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 2, 3 ])

GUI31 = Parameter(name = 'GUI31',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI31}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 3, 1 ])

GUI32 = Parameter(name = 'GUI32',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI32}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 3, 2 ])

GUI33 = Parameter(name = 'GUI33',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUI33}',
                  lhablock = 'YukawaGUI',
                  lhacode = [ 3, 3 ])

GUR11 = Parameter(name = 'GUR11',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUR11}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 1, 1 ])

GUR12 = Parameter(name = 'GUR12',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUR12}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 1, 2 ])

GUR13 = Parameter(name = 'GUR13',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUR13}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 1, 3 ])

GUR21 = Parameter(name = 'GUR21',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUR21}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 2, 1 ])

GUR22 = Parameter(name = 'GUR22',
                  nature = 'external',
                  type = 'real',
                  value = 2.,
                  texname = '\\text{GUR22}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 2, 2 ])

GUR23 = Parameter(name = 'GUR23',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUR23}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 2, 3 ])

GUR31 = Parameter(name = 'GUR31',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{GUR31}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 3, 1 ])

GUR32 = Parameter(name = 'GUR32',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{GUR32}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 3, 2 ])

GUR33 = Parameter(name = 'GUR33',
                  nature = 'external',
                  type = 'real',
                  value = 100.,
                  texname = '\\text{GUR33}',
                  lhablock = 'YukawaGUR',
                  lhacode = [ 3, 3 ])

yukl1 = Parameter(name = 'yukl1',
                  nature = 'external',
                  type = 'complex',
                  value = 0.,
                  texname = '\\text{yukl1}',
                  lhablock = 'YUKAWAL',
                  lhacode = [ 1 ])

yukl2 = Parameter(name = 'yukl2',
                  nature = 'external',
                  type = 'complex',
                  value = 0.,
                  texname = '\\text{yukl2}',
                  lhablock = 'YUKAWAL',
                  lhacode = [ 2 ])

yukl3 = Parameter(name = 'yukl3',
                  nature = 'external',
                  type = 'complex',
                  value = 1.777,
                  texname = '\\text{yukl3}',
                  lhablock = 'YUKAWAL',
                  lhacode = [ 3 ])

yuku1 = Parameter(name = 'yuku1',
                  nature = 'external',
                  type = 'complex',
                  value = 0.,
                  texname = '\\text{yuku1}',
                  lhablock = 'YUKAWAU',
                  lhacode = [ 1 ])

yuku2 = Parameter(name = 'yuku2',
                  nature = 'external',
                  type = 'complex',
                  value = 0.6,
                  texname = '\\text{yuku2}',
                  lhablock = 'YUKAWAU',
                  lhacode = [ 2 ])

yuku3 = Parameter(name = 'yuku3',
                  nature = 'external',
                  type = 'complex',
                  value = 175.,
                  texname = '\\text{yuku3}',
                  lhablock = 'YUKAWAU',
                  lhacode = [ 3 ])

MM = Parameter(name = 'MM',
               nature = 'external',
               type = 'real',
               value = 0.106,
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

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 1.25,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 4 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 174.3,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MS = Parameter(name = 'MS',
               nature = 'external',
               type = 'real',
               value = 0.105,
               texname = '\\text{MS}',
               lhablock = 'MASS',
               lhacode = [ 3 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.2,
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

mhc = Parameter(name = 'mhc',
                nature = 'external',
                type = 'real',
                value = 300.,
                texname = '\\text{mhc}',
                lhablock = 'MASS',
                lhacode = [ 37 ])

mh1 = Parameter(name = 'mh1',
                nature = 'external',
                type = 'real',
                value = 284.44035350978,
                texname = '\\text{mh1}',
                lhablock = 'MASS',
                lhacode = [ 25 ])

mh2 = Parameter(name = 'mh2',
                nature = 'external',
                type = 'real',
                value = 326.63207182875,
                texname = '\\text{mh2}',
                lhablock = 'MASS',
                lhacode = [ 35 ])

mh3 = Parameter(name = 'mh3',
                nature = 'external',
                type = 'real',
                value = 379.42930373822,
                texname = '\\text{mh3}',
                lhablock = 'MASS',
                lhacode = [ 36 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.5148707555575,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4126275103422,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.0028844418592,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

whc = Parameter(name = 'whc',
                nature = 'external',
                type = 'real',
                value = 1.7272545649498,
                texname = '\\text{whc}',
                lhablock = 'DECAY',
                lhacode = [ 37 ])

Wh1 = Parameter(name = 'Wh1',
                nature = 'external',
                type = 'real',
                value = 0.018006631898186,
                texname = '\\text{Wh1}',
                lhablock = 'DECAY',
                lhacode = [ 25 ])

Wh2 = Parameter(name = 'Wh2',
                nature = 'external',
                type = 'real',
                value = 7.088459362732,
                texname = '\\text{Wh2}',
                lhablock = 'DECAY',
                lhacode = [ 35 ])

Wh3 = Parameter(name = 'Wh3',
                nature = 'external',
                type = 'real',
                value = 9.5259020924187,
                texname = '\\text{Wh3}',
                lhablock = 'DECAY',
                lhacode = [ 36 ])

CKM11 = Parameter(name = 'CKM11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKMR11 + CKMI11*complex(0,1)',
                  texname = '\\text{CKM11}')

CKM12 = Parameter(name = 'CKM12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKMR12 + CKMI12*complex(0,1)',
                  texname = '\\text{CKM12}')

CKM21 = Parameter(name = 'CKM21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKMR21 + CKMI21*complex(0,1)',
                  texname = '\\text{CKM21}')

CKM22 = Parameter(name = 'CKM22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'CKMR22 + CKMI22*complex(0,1)',
                  texname = '\\text{CKM22}')

DD11 = Parameter(name = 'DD11',
                 nature = 'internal',
                 type = 'complex',
                 value = '0',
                 texname = '\\text{DD11}')

DD22 = Parameter(name = 'DD22',
                 nature = 'internal',
                 type = 'complex',
                 value = '0',
                 texname = '\\text{DD22}')

DD33 = Parameter(name = 'DD33',
                 nature = 'internal',
                 type = 'complex',
                 value = '0',
                 texname = '\\text{DD33}')

GD11 = Parameter(name = 'GD11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI11 + GDR11',
                 texname = '\\text{GD11}')

GD12 = Parameter(name = 'GD12',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI12 + GDR12',
                 texname = '\\text{GD12}')

GD13 = Parameter(name = 'GD13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI13 + GDR13',
                 texname = '\\text{GD13}')

GD21 = Parameter(name = 'GD21',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI21 + GDR21',
                 texname = '\\text{GD21}')

GD22 = Parameter(name = 'GD22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI22 + GDR22',
                 texname = '\\text{GD22}')

GD23 = Parameter(name = 'GD23',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI23 + GDR23',
                 texname = '\\text{GD23}')

GD31 = Parameter(name = 'GD31',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI31 + GDR31',
                 texname = '\\text{GD31}')

GD32 = Parameter(name = 'GD32',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI32 + GDR32',
                 texname = '\\text{GD32}')

GD33 = Parameter(name = 'GD33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GDI33 + GDR33',
                 texname = '\\text{GD33}')

GL11 = Parameter(name = 'GL11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI11 + GLR11',
                 texname = '\\text{GL11}')

GL12 = Parameter(name = 'GL12',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI12 + GLR12',
                 texname = '\\text{GL12}')

GL13 = Parameter(name = 'GL13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI13 + GLR13',
                 texname = '\\text{GL13}')

GL21 = Parameter(name = 'GL21',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI21 + GLR21',
                 texname = '\\text{GL21}')

GL22 = Parameter(name = 'GL22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI22 + GLR22',
                 texname = '\\text{GL22}')

GL23 = Parameter(name = 'GL23',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI23 + GLR23',
                 texname = '\\text{GL23}')

GL31 = Parameter(name = 'GL31',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI31 + GLR31',
                 texname = '\\text{GL31}')

GL32 = Parameter(name = 'GL32',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI32 + GLR32',
                 texname = '\\text{GL32}')

GL33 = Parameter(name = 'GL33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GLI33 + GLR33',
                 texname = '\\text{GL33}')

GU11 = Parameter(name = 'GU11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI11 + GUR11',
                 texname = '\\text{GU11}')

GU12 = Parameter(name = 'GU12',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI12 + GUR12',
                 texname = '\\text{GU12}')

GU13 = Parameter(name = 'GU13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI13 + GUR13',
                 texname = '\\text{GU13}')

GU21 = Parameter(name = 'GU21',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI21 + GUR21',
                 texname = '\\text{GU21}')

GU22 = Parameter(name = 'GU22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI22 + GUR22',
                 texname = '\\text{GU22}')

GU23 = Parameter(name = 'GU23',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI23 + GUR23',
                 texname = '\\text{GU23}')

GU31 = Parameter(name = 'GU31',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI31 + GUR31',
                 texname = '\\text{GU31}')

GU32 = Parameter(name = 'GU32',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI32 + GUR32',
                 texname = '\\text{GU32}')

GU33 = Parameter(name = 'GU33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complex(0,1)*GUI33 + GUR33',
                 texname = '\\text{GU33}')

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

l6 = Parameter(name = 'l6',
               nature = 'internal',
               type = 'complex',
               value = 'complex(0,1)*lI6 + lR6',
               texname = '\\text{l6}')

l7 = Parameter(name = 'l7',
               nature = 'internal',
               type = 'complex',
               value = 'complex(0,1)*lI7 + lR7',
               texname = '\\text{l7}')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (aEW*cmath.pi*MZ**2)/(Gf*cmath.sqrt(2))))',
               texname = '\\text{MW}')

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

mu1 = Parameter(name = 'mu1',
                nature = 'internal',
                type = 'real',
                value = '-(l1*v**2)',
                texname = '\\text{mu1}')

mu2 = Parameter(name = 'mu2',
                nature = 'internal',
                type = 'real',
                value = 'mhc**2 - (l3*v**2)/2.',
                texname = '\\text{mu2}')

mu3 = Parameter(name = 'mu3',
                nature = 'internal',
                type = 'complex',
                value = '-(l6*v**2)/2.',
                texname = '\\text{mu3}')

