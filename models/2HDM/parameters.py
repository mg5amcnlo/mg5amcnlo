# This file was automatically created by FeynRules 1.7.55
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (October 6, 2011)
# Date: Wed 8 Aug 2012 14:37:00



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
CKMI1x1 = Parameter(name = 'CKMI1x1',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI1x1}',
                    lhablock = 'CKMI',
                    lhacode = [ 1, 1 ])

CKMI1x2 = Parameter(name = 'CKMI1x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI1x2}',
                    lhablock = 'CKMI',
                    lhacode = [ 1, 2 ])

CKMI1x3 = Parameter(name = 'CKMI1x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI1x3}',
                    lhablock = 'CKMI',
                    lhacode = [ 1, 3 ])

CKMI2x1 = Parameter(name = 'CKMI2x1',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI2x1}',
                    lhablock = 'CKMI',
                    lhacode = [ 2, 1 ])

CKMI2x2 = Parameter(name = 'CKMI2x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI2x2}',
                    lhablock = 'CKMI',
                    lhacode = [ 2, 2 ])

CKMI2x3 = Parameter(name = 'CKMI2x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI2x3}',
                    lhablock = 'CKMI',
                    lhacode = [ 2, 3 ])

CKMI3x1 = Parameter(name = 'CKMI3x1',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI3x1}',
                    lhablock = 'CKMI',
                    lhacode = [ 3, 1 ])

CKMI3x2 = Parameter(name = 'CKMI3x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI3x2}',
                    lhablock = 'CKMI',
                    lhacode = [ 3, 2 ])

CKMI3x3 = Parameter(name = 'CKMI3x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMI3x3}',
                    lhablock = 'CKMI',
                    lhacode = [ 3, 3 ])

CKMR1x1 = Parameter(name = 'CKMR1x1',
                    nature = 'external',
                    type = 'real',
                    value = 0.974589144,
                    texname = '\\text{CKMR1x1}',
                    lhablock = 'CKMR',
                    lhacode = [ 1, 1 ])

CKMR1x2 = Parameter(name = 'CKMR1x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.224,
                    texname = '\\text{CKMR1x2}',
                    lhablock = 'CKMR',
                    lhacode = [ 1, 2 ])

CKMR1x3 = Parameter(name = 'CKMR1x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMR1x3}',
                    lhablock = 'CKMR',
                    lhacode = [ 1, 3 ])

CKMR2x1 = Parameter(name = 'CKMR2x1',
                    nature = 'external',
                    type = 'real',
                    value = -0.224,
                    texname = '\\text{CKMR2x1}',
                    lhablock = 'CKMR',
                    lhacode = [ 2, 1 ])

CKMR2x2 = Parameter(name = 'CKMR2x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.974589144,
                    texname = '\\text{CKMR2x2}',
                    lhablock = 'CKMR',
                    lhacode = [ 2, 2 ])

CKMR2x3 = Parameter(name = 'CKMR2x3',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMR2x3}',
                    lhablock = 'CKMR',
                    lhacode = [ 2, 3 ])

CKMR3x1 = Parameter(name = 'CKMR3x1',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMR3x1}',
                    lhablock = 'CKMR',
                    lhacode = [ 3, 1 ])

CKMR3x2 = Parameter(name = 'CKMR3x2',
                    nature = 'external',
                    type = 'real',
                    value = 0.,
                    texname = '\\text{CKMR3x2}',
                    lhablock = 'CKMR',
                    lhacode = [ 3, 2 ])

CKMR3x3 = Parameter(name = 'CKMR3x3',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{CKMR3x3}',
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

lR5 = Parameter(name = 'lR5',
                nature = 'external',
                type = 'real',
                value = 0.4,
                texname = '\\text{lR5}',
                lhablock = 'Higgs',
                lhacode = [ 5 ])

lI5 = Parameter(name = 'lI5',
                nature = 'external',
                type = 'real',
                value = 0.3,
                texname = '\\text{lI5}',
                lhablock = 'Higgs',
                lhacode = [ 6 ])

lR6 = Parameter(name = 'lR6',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = '\\text{lR6}',
                lhablock = 'Higgs',
                lhacode = [ 7 ])

lI6 = Parameter(name = 'lI6',
                nature = 'external',
                type = 'real',
                value = 0.2,
                texname = '\\text{lI6}',
                lhablock = 'Higgs',
                lhacode = [ 8 ])

lR7 = Parameter(name = 'lR7',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = '\\text{lR7}',
                lhablock = 'Higgs',
                lhacode = [ 9 ])

lI7 = Parameter(name = 'lI7',
                nature = 'external',
                type = 'real',
                value = 0,
                texname = '\\text{lI7}',
                lhablock = 'Higgs',
                lhacode = [ 10 ])

TH1x1 = Parameter(name = 'TH1x1',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{TH1x1}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 1, 1 ])

TH1x2 = Parameter(name = 'TH1x2',
                  nature = 'external',
                  type = 'real',
                  value = 0.78064408782535,
                  texname = '\\text{TH1x2}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 1, 2 ])

TH1x3 = Parameter(name = 'TH1x3',
                  nature = 'external',
                  type = 'real',
                  value = 0.62497584604793,
                  texname = '\\text{TH1x3}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 1, 3 ])

TH2x1 = Parameter(name = 'TH2x1',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{TH2x1}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 2, 1 ])

TH2x2 = Parameter(name = 'TH2x2',
                  nature = 'external',
                  type = 'real',
                  value = -0.62497584604793,
                  texname = '\\text{TH2x2}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 2, 2 ])

TH2x3 = Parameter(name = 'TH2x3',
                  nature = 'external',
                  type = 'real',
                  value = 0.78064408782535,
                  texname = '\\text{TH2x3}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 2, 3 ])

TH3x1 = Parameter(name = 'TH3x1',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{TH3x1}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 3, 1 ])

TH3x2 = Parameter(name = 'TH3x2',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{TH3x2}',
                  lhablock = 'HiggsMix',
                  lhacode = [ 3, 2 ])

TH3x3 = Parameter(name = 'TH3x3',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{TH3x3}',
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

GDI1x1 = Parameter(name = 'GDI1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI1x1}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 1, 1 ])

GDI1x2 = Parameter(name = 'GDI1x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI1x2}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 1, 2 ])

GDI1x3 = Parameter(name = 'GDI1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI1x3}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 1, 3 ])

GDI2x1 = Parameter(name = 'GDI2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI2x1}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 2, 1 ])

GDI2x2 = Parameter(name = 'GDI2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI2x2}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 2, 2 ])

GDI2x3 = Parameter(name = 'GDI2x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI2x3}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 2, 3 ])

GDI3x1 = Parameter(name = 'GDI3x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI3x1}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 3, 1 ])

GDI3x2 = Parameter(name = 'GDI3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI3x2}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 3, 2 ])

GDI3x3 = Parameter(name = 'GDI3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDI3x3}',
                   lhablock = 'YukawaGDI',
                   lhacode = [ 3, 3 ])

GDR1x1 = Parameter(name = 'GDR1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDR1x1}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 1, 1 ])

GDR1x2 = Parameter(name = 'GDR1x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDR1x2}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 1, 2 ])

GDR1x3 = Parameter(name = 'GDR1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDR1x3}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 1, 3 ])

GDR2x1 = Parameter(name = 'GDR2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDR2x1}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 2, 1 ])

GDR2x2 = Parameter(name = 'GDR2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.4,
                   texname = '\\text{GDR2x2}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 2, 2 ])

GDR2x3 = Parameter(name = 'GDR2x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDR2x3}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 2, 3 ])

GDR3x1 = Parameter(name = 'GDR3x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GDR3x1}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 3, 1 ])

GDR3x2 = Parameter(name = 'GDR3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.2,
                   texname = '\\text{GDR3x2}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 3, 2 ])

GDR3x3 = Parameter(name = 'GDR3x3',
                   nature = 'external',
                   type = 'real',
                   value = 5.,
                   texname = '\\text{GDR3x3}',
                   lhablock = 'YukawaGDR',
                   lhacode = [ 3, 3 ])

GLI1x1 = Parameter(name = 'GLI1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI1x1}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 1, 1 ])

GLI1x2 = Parameter(name = 'GLI1x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI1x2}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 1, 2 ])

GLI1x3 = Parameter(name = 'GLI1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI1x3}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 1, 3 ])

GLI2x1 = Parameter(name = 'GLI2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI2x1}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 2, 1 ])

GLI2x2 = Parameter(name = 'GLI2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI2x2}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 2, 2 ])

GLI2x3 = Parameter(name = 'GLI2x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI2x3}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 2, 3 ])

GLI3x1 = Parameter(name = 'GLI3x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI3x1}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 3, 1 ])

GLI3x2 = Parameter(name = 'GLI3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI3x2}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 3, 2 ])

GLI3x3 = Parameter(name = 'GLI3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLI3x3}',
                   lhablock = 'YukawaGLI',
                   lhacode = [ 3, 3 ])

GLR1x1 = Parameter(name = 'GLR1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLR1x1}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 1, 1 ])

GLR1x2 = Parameter(name = 'GLR1x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLR1x2}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 1, 2 ])

GLR1x3 = Parameter(name = 'GLR1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLR1x3}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 1, 3 ])

GLR2x1 = Parameter(name = 'GLR2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLR2x1}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 2, 1 ])

GLR2x2 = Parameter(name = 'GLR2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.1,
                   texname = '\\text{GLR2x2}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 2, 2 ])

GLR2x3 = Parameter(name = 'GLR2x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLR2x3}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 2, 3 ])

GLR3x1 = Parameter(name = 'GLR3x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GLR3x1}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 3, 1 ])

GLR3x2 = Parameter(name = 'GLR3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.5,
                   texname = '\\text{GLR3x2}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 3, 2 ])

GLR3x3 = Parameter(name = 'GLR3x3',
                   nature = 'external',
                   type = 'real',
                   value = 3.,
                   texname = '\\text{GLR3x3}',
                   lhablock = 'YukawaGLR',
                   lhacode = [ 3, 3 ])

GUI1x1 = Parameter(name = 'GUI1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI1x1}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 1, 1 ])

GUI1x2 = Parameter(name = 'GUI1x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI1x2}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 1, 2 ])

GUI1x3 = Parameter(name = 'GUI1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI1x3}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 1, 3 ])

GUI2x1 = Parameter(name = 'GUI2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI2x1}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 2, 1 ])

GUI2x2 = Parameter(name = 'GUI2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI2x2}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 2, 2 ])

GUI2x3 = Parameter(name = 'GUI2x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI2x3}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 2, 3 ])

GUI3x1 = Parameter(name = 'GUI3x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI3x1}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 3, 1 ])

GUI3x2 = Parameter(name = 'GUI3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI3x2}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 3, 2 ])

GUI3x3 = Parameter(name = 'GUI3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUI3x3}',
                   lhablock = 'YukawaGUI',
                   lhacode = [ 3, 3 ])

GUR1x1 = Parameter(name = 'GUR1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUR1x1}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 1, 1 ])

GUR1x2 = Parameter(name = 'GUR1x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUR1x2}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 1, 2 ])

GUR1x3 = Parameter(name = 'GUR1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUR1x3}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 1, 3 ])

GUR2x1 = Parameter(name = 'GUR2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUR2x1}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 2, 1 ])

GUR2x2 = Parameter(name = 'GUR2x2',
                   nature = 'external',
                   type = 'real',
                   value = 2.,
                   texname = '\\text{GUR2x2}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 2, 2 ])

GUR2x3 = Parameter(name = 'GUR2x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUR2x3}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 2, 3 ])

GUR3x1 = Parameter(name = 'GUR3x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.,
                   texname = '\\text{GUR3x1}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 3, 1 ])

GUR3x2 = Parameter(name = 'GUR3x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{GUR3x2}',
                   lhablock = 'YukawaGUR',
                   lhacode = [ 3, 2 ])

GUR3x3 = Parameter(name = 'GUR3x3',
                   nature = 'external',
                   type = 'real',
                   value = 100.,
                   texname = '\\text{GUR3x3}',
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

CKM1x1 = Parameter(name = 'CKM1x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMR1x1 + CKMI1x1*complex(0,1)',
                   texname = '\\text{CKM1x1}')

CKM1x2 = Parameter(name = 'CKM1x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMR1x2 + CKMI1x2*complex(0,1)',
                   texname = '\\text{CKM1x2}')

CKM2x1 = Parameter(name = 'CKM2x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMR2x1 + CKMI2x1*complex(0,1)',
                   texname = '\\text{CKM2x1}')

CKM2x2 = Parameter(name = 'CKM2x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKMR2x2 + CKMI2x2*complex(0,1)',
                   texname = '\\text{CKM2x2}')

DD1x1 = Parameter(name = 'DD1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{DD1x1}')

DD2x2 = Parameter(name = 'DD2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{DD2x2}')

DD3x3 = Parameter(name = 'DD3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = '0',
                  texname = '\\text{DD3x3}')

GD1x1 = Parameter(name = 'GD1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI1x1 + GDR1x1',
                  texname = '\\text{GD1x1}')

GD1x2 = Parameter(name = 'GD1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI1x2 + GDR1x2',
                  texname = '\\text{GD1x2}')

GD1x3 = Parameter(name = 'GD1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI1x3 + GDR1x3',
                  texname = '\\text{GD1x3}')

GD2x1 = Parameter(name = 'GD2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI2x1 + GDR2x1',
                  texname = '\\text{GD2x1}')

GD2x2 = Parameter(name = 'GD2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI2x2 + GDR2x2',
                  texname = '\\text{GD2x2}')

GD2x3 = Parameter(name = 'GD2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI2x3 + GDR2x3',
                  texname = '\\text{GD2x3}')

GD3x1 = Parameter(name = 'GD3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI3x1 + GDR3x1',
                  texname = '\\text{GD3x1}')

GD3x2 = Parameter(name = 'GD3x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI3x2 + GDR3x2',
                  texname = '\\text{GD3x2}')

GD3x3 = Parameter(name = 'GD3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GDI3x3 + GDR3x3',
                  texname = '\\text{GD3x3}')

GL1x1 = Parameter(name = 'GL1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI1x1 + GLR1x1',
                  texname = '\\text{GL1x1}')

GL1x2 = Parameter(name = 'GL1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI1x2 + GLR1x2',
                  texname = '\\text{GL1x2}')

GL1x3 = Parameter(name = 'GL1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI1x3 + GLR1x3',
                  texname = '\\text{GL1x3}')

GL2x1 = Parameter(name = 'GL2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI2x1 + GLR2x1',
                  texname = '\\text{GL2x1}')

GL2x2 = Parameter(name = 'GL2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI2x2 + GLR2x2',
                  texname = '\\text{GL2x2}')

GL2x3 = Parameter(name = 'GL2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI2x3 + GLR2x3',
                  texname = '\\text{GL2x3}')

GL3x1 = Parameter(name = 'GL3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI3x1 + GLR3x1',
                  texname = '\\text{GL3x1}')

GL3x2 = Parameter(name = 'GL3x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI3x2 + GLR3x2',
                  texname = '\\text{GL3x2}')

GL3x3 = Parameter(name = 'GL3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GLI3x3 + GLR3x3',
                  texname = '\\text{GL3x3}')

GU1x1 = Parameter(name = 'GU1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI1x1 + GUR1x1',
                  texname = '\\text{GU1x1}')

GU1x2 = Parameter(name = 'GU1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI1x2 + GUR1x2',
                  texname = '\\text{GU1x2}')

GU1x3 = Parameter(name = 'GU1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI1x3 + GUR1x3',
                  texname = '\\text{GU1x3}')

GU2x1 = Parameter(name = 'GU2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI2x1 + GUR2x1',
                  texname = '\\text{GU2x1}')

GU2x2 = Parameter(name = 'GU2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI2x2 + GUR2x2',
                  texname = '\\text{GU2x2}')

GU2x3 = Parameter(name = 'GU2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI2x3 + GUR2x3',
                  texname = '\\text{GU2x3}')

GU3x1 = Parameter(name = 'GU3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI3x1 + GUR3x1',
                  texname = '\\text{GU3x1}')

GU3x2 = Parameter(name = 'GU3x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI3x2 + GUR3x2',
                  texname = '\\text{GU3x2}')

GU3x3 = Parameter(name = 'GU3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complex(0,1)*GUI3x3 + GUR3x3',
                  texname = '\\text{GU3x3}')

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

l5 = Parameter(name = 'l5',
               nature = 'internal',
               type = 'complex',
               value = 'complex(0,1)*lI5 + lR5',
               texname = '\\text{l5}')

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

