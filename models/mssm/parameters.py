# This file was automatically created by FeynRules $Revision: 364 $
# Mathematica version: 7.0 for Linux x86 (32-bit) (February 18, 2009)
# Date: Sat 4 Dec 2010 12:26:05



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

RRd11 = Parameter(name = 'RRd11',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd11}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 1, 1 ])

RRd22 = Parameter(name = 'RRd22',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd22}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 2, 2 ])

RRd33 = Parameter(name = 'RRd33',
                  nature = 'external',
                  type = 'real',
                  value = 0.938737896,
                  texname = '\\text{RRd33}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 3, 3 ])

RRd36 = Parameter(name = 'RRd36',
                  nature = 'external',
                  type = 'real',
                  value = 0.344631925,
                  texname = '\\text{RRd36}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 3, 6 ])

RRd44 = Parameter(name = 'RRd44',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd44}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 4, 4 ])

RRd55 = Parameter(name = 'RRd55',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd55}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 5, 5 ])

RRd63 = Parameter(name = 'RRd63',
                  nature = 'external',
                  type = 'real',
                  value = -0.344631925,
                  texname = '\\text{RRd63}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 6, 3 ])

RRd66 = Parameter(name = 'RRd66',
                  nature = 'external',
                  type = 'real',
                  value = 0.938737896,
                  texname = '\\text{RRd66}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 6, 6 ])

alp = Parameter(name = 'alp',
                nature = 'external',
                type = 'real',
                value = -0.11382521,
                texname = '\\alpha ',
                lhablock = 'FRALPHA',
                lhacode = [ 1 ])

gp = Parameter(name = 'gp',
               nature = 'external',
               type = 'real',
               value = 0.360966847,
               texname = 'g\'',
               lhablock = 'GAUGE',
               lhacode = [ 1 ])

gw = Parameter(name = 'gw',
               nature = 'external',
               type = 'real',
               value = 0.64648221,
               texname = 'g_w',
               lhablock = 'GAUGE',
               lhacode = [ 2 ])

ggs = Parameter(name = 'ggs',
                nature = 'external',
                type = 'real',
                value = 1.10178679,
                texname = 'g_s',
                lhablock = 'GAUGE',
                lhacode = [ 3 ])

RMU = Parameter(name = 'RMU',
                nature = 'external',
                type = 'real',
                value = 357.680977,
                texname = '\\text{RMU}',
                lhablock = 'HMIX',
                lhacode = [ 1 ])

tb = Parameter(name = 'tb',
               nature = 'external',
               type = 'real',
               value = 9.74862403,
               texname = '\\beta ',
               lhablock = 'HMIX',
               lhacode = [ 2 ])

vev = Parameter(name = 'vev',
                nature = 'external',
                type = 'real',
                value = 244.894549,
                texname = 'v',
                lhablock = 'HMIX',
                lhacode = [ 3 ])

MA2 = Parameter(name = 'MA2',
                nature = 'external',
                type = 'real',
                value = 166439.065,
                texname = 'm_A^2',
                lhablock = 'HMIX',
                lhacode = [ 4 ])

RmDR11 = Parameter(name = 'RmDR11',
                   nature = 'external',
                   type = 'real',
                   value = 273684.674,
                   texname = '\\text{RmDR11}',
                   lhablock = 'MSD2',
                   lhacode = [ 1, 1 ])

RmDR22 = Parameter(name = 'RmDR22',
                   nature = 'external',
                   type = 'real',
                   value = 273684.674,
                   texname = '\\text{RmDR22}',
                   lhablock = 'MSD2',
                   lhacode = [ 2, 2 ])

RmDR33 = Parameter(name = 'RmDR33',
                   nature = 'external',
                   type = 'real',
                   value = 270261.969,
                   texname = '\\text{RmDR33}',
                   lhablock = 'MSD2',
                   lhacode = [ 3, 3 ])

RmER11 = Parameter(name = 'RmER11',
                   nature = 'external',
                   type = 'real',
                   value = 18630.6287,
                   texname = '\\text{RmER11}',
                   lhablock = 'MSE2',
                   lhacode = [ 1, 1 ])

RmER22 = Parameter(name = 'RmER22',
                   nature = 'external',
                   type = 'real',
                   value = 18630.6287,
                   texname = '\\text{RmER22}',
                   lhablock = 'MSE2',
                   lhacode = [ 2, 2 ])

RmER33 = Parameter(name = 'RmER33',
                   nature = 'external',
                   type = 'real',
                   value = 17967.6406,
                   texname = '\\text{RmER33}',
                   lhablock = 'MSE2',
                   lhacode = [ 3, 3 ])

RmLL11 = Parameter(name = 'RmLL11',
                   nature = 'external',
                   type = 'real',
                   value = 38155.67,
                   texname = '\\text{RmLL11}',
                   lhablock = 'MSL2',
                   lhacode = [ 1, 1 ])

RmLL22 = Parameter(name = 'RmLL22',
                   nature = 'external',
                   type = 'real',
                   value = 38155.67,
                   texname = '\\text{RmLL22}',
                   lhablock = 'MSL2',
                   lhacode = [ 2, 2 ])

RmLL33 = Parameter(name = 'RmLL33',
                   nature = 'external',
                   type = 'real',
                   value = 37828.6769,
                   texname = '\\text{RmLL33}',
                   lhablock = 'MSL2',
                   lhacode = [ 3, 3 ])

RMx1 = Parameter(name = 'RMx1',
                 nature = 'external',
                 type = 'real',
                 value = 101.396534,
                 texname = '\\text{RMx1}',
                 lhablock = 'MSOFT',
                 lhacode = [ 1 ])

RMx2 = Parameter(name = 'RMx2',
                 nature = 'external',
                 type = 'real',
                 value = 191.504241,
                 texname = '\\text{RMx2}',
                 lhablock = 'MSOFT',
                 lhacode = [ 2 ])

RMx3 = Parameter(name = 'RMx3',
                 nature = 'external',
                 type = 'real',
                 value = 588.263031,
                 texname = '\\text{RMx3}',
                 lhablock = 'MSOFT',
                 lhacode = [ 3 ])

MHu2 = Parameter(name = 'MHu2',
                 nature = 'external',
                 type = 'real',
                 value = 32337.4943,
                 texname = 'm_{H_u}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 21 ])

MHd2 = Parameter(name = 'MHd2',
                 nature = 'external',
                 type = 'real',
                 value = -128800.134,
                 texname = 'm_{H_d}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 22 ])

RmQL11 = Parameter(name = 'RmQL11',
                   nature = 'external',
                   type = 'real',
                   value = 299836.701,
                   texname = '\\text{RmQL11}',
                   lhablock = 'MSQ2',
                   lhacode = [ 1, 1 ])

RmQL22 = Parameter(name = 'RmQL22',
                   nature = 'external',
                   type = 'real',
                   value = 299836.701,
                   texname = '\\text{RmQL22}',
                   lhablock = 'MSQ2',
                   lhacode = [ 2, 2 ])

RmQL33 = Parameter(name = 'RmQL33',
                   nature = 'external',
                   type = 'real',
                   value = 248765.367,
                   texname = '\\text{RmQL33}',
                   lhablock = 'MSQ2',
                   lhacode = [ 3, 3 ])

RmUR11 = Parameter(name = 'RmUR11',
                   nature = 'external',
                   type = 'real',
                   value = 280382.106,
                   texname = '\\text{RmUR11}',
                   lhablock = 'MSU2',
                   lhacode = [ 1, 1 ])

RmUR22 = Parameter(name = 'RmUR22',
                   nature = 'external',
                   type = 'real',
                   value = 280382.106,
                   texname = '\\text{RmUR22}',
                   lhablock = 'MSU2',
                   lhacode = [ 2, 2 ])

RmUR33 = Parameter(name = 'RmUR33',
                   nature = 'external',
                   type = 'real',
                   value = 179137.072,
                   texname = '\\text{RmUR33}',
                   lhablock = 'MSU2',
                   lhacode = [ 3, 3 ])

RNN11 = Parameter(name = 'RNN11',
                  nature = 'external',
                  type = 'real',
                  value = 0.98636443,
                  texname = '\\text{RNN11}',
                  lhablock = 'NMIX',
                  lhacode = [ 1, 1 ])

RNN12 = Parameter(name = 'RNN12',
                  nature = 'external',
                  type = 'real',
                  value = -0.0531103553,
                  texname = '\\text{RNN12}',
                  lhablock = 'NMIX',
                  lhacode = [ 1, 2 ])

RNN13 = Parameter(name = 'RNN13',
                  nature = 'external',
                  type = 'real',
                  value = 0.146433995,
                  texname = '\\text{RNN13}',
                  lhablock = 'NMIX',
                  lhacode = [ 1, 3 ])

RNN14 = Parameter(name = 'RNN14',
                  nature = 'external',
                  type = 'real',
                  value = -0.0531186117,
                  texname = '\\text{RNN14}',
                  lhablock = 'NMIX',
                  lhacode = [ 1, 4 ])

RNN21 = Parameter(name = 'RNN21',
                  nature = 'external',
                  type = 'real',
                  value = 0.0993505358,
                  texname = '\\text{RNN21}',
                  lhablock = 'NMIX',
                  lhacode = [ 2, 1 ])

RNN22 = Parameter(name = 'RNN22',
                  nature = 'external',
                  type = 'real',
                  value = 0.944949299,
                  texname = '\\text{RNN22}',
                  lhablock = 'NMIX',
                  lhacode = [ 2, 2 ])

RNN23 = Parameter(name = 'RNN23',
                  nature = 'external',
                  type = 'real',
                  value = -0.26984672,
                  texname = '\\text{RNN23}',
                  lhablock = 'NMIX',
                  lhacode = [ 2, 3 ])

RNN24 = Parameter(name = 'RNN24',
                  nature = 'external',
                  type = 'real',
                  value = 0.156150698,
                  texname = '\\text{RNN24}',
                  lhablock = 'NMIX',
                  lhacode = [ 2, 4 ])

RNN31 = Parameter(name = 'RNN31',
                  nature = 'external',
                  type = 'real',
                  value = -0.0603388002,
                  texname = '\\text{RNN31}',
                  lhablock = 'NMIX',
                  lhacode = [ 3, 1 ])

RNN32 = Parameter(name = 'RNN32',
                  nature = 'external',
                  type = 'real',
                  value = 0.0877004854,
                  texname = '\\text{RNN32}',
                  lhablock = 'NMIX',
                  lhacode = [ 3, 2 ])

RNN33 = Parameter(name = 'RNN33',
                  nature = 'external',
                  type = 'real',
                  value = 0.695877493,
                  texname = '\\text{RNN33}',
                  lhablock = 'NMIX',
                  lhacode = [ 3, 3 ])

RNN34 = Parameter(name = 'RNN34',
                  nature = 'external',
                  type = 'real',
                  value = 0.710226984,
                  texname = '\\text{RNN34}',
                  lhablock = 'NMIX',
                  lhacode = [ 3, 4 ])

RNN41 = Parameter(name = 'RNN41',
                  nature = 'external',
                  type = 'real',
                  value = -0.116507132,
                  texname = '\\text{RNN41}',
                  lhablock = 'NMIX',
                  lhacode = [ 4, 1 ])

RNN42 = Parameter(name = 'RNN42',
                  nature = 'external',
                  type = 'real',
                  value = 0.310739017,
                  texname = '\\text{RNN42}',
                  lhablock = 'NMIX',
                  lhacode = [ 4, 2 ])

RNN43 = Parameter(name = 'RNN43',
                  nature = 'external',
                  type = 'real',
                  value = 0.64922596,
                  texname = '\\text{RNN43}',
                  lhablock = 'NMIX',
                  lhacode = [ 4, 3 ])

RNN44 = Parameter(name = 'RNN44',
                  nature = 'external',
                  type = 'real',
                  value = -0.684377823,
                  texname = '\\text{RNN44}',
                  lhablock = 'NMIX',
                  lhacode = [ 4, 4 ])

RRl11 = Parameter(name = 'RRl11',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl11}',
                  lhablock = 'SELMIX',
                  lhacode = [ 1, 1 ])

RRl22 = Parameter(name = 'RRl22',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl22}',
                  lhablock = 'SELMIX',
                  lhacode = [ 2, 2 ])

RRl33 = Parameter(name = 'RRl33',
                  nature = 'external',
                  type = 'real',
                  value = 0.28248719,
                  texname = '\\text{RRl33}',
                  lhablock = 'SELMIX',
                  lhacode = [ 3, 3 ])

RRl36 = Parameter(name = 'RRl36',
                  nature = 'external',
                  type = 'real',
                  value = 0.959271071,
                  texname = '\\text{RRl36}',
                  lhablock = 'SELMIX',
                  lhacode = [ 3, 6 ])

RRl44 = Parameter(name = 'RRl44',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl44}',
                  lhablock = 'SELMIX',
                  lhacode = [ 4, 4 ])

RRl55 = Parameter(name = 'RRl55',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl55}',
                  lhablock = 'SELMIX',
                  lhacode = [ 5, 5 ])

RRl63 = Parameter(name = 'RRl63',
                  nature = 'external',
                  type = 'real',
                  value = 0.959271071,
                  texname = '\\text{RRl63}',
                  lhablock = 'SELMIX',
                  lhacode = [ 6, 3 ])

RRl66 = Parameter(name = 'RRl66',
                  nature = 'external',
                  type = 'real',
                  value = -0.28248719,
                  texname = '\\text{RRl66}',
                  lhablock = 'SELMIX',
                  lhacode = [ 6, 6 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.934,
                  texname = '\\alpha _w^{-1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.0000116637,
               texname = 'G_F',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.118,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

MMZ = Parameter(name = 'MMZ',
                nature = 'external',
                type = 'real',
                value = 91.1876,
                texname = 'm_Z',
                lhablock = 'SMINPUTS',
                lhacode = [ 4 ])

RRn11 = Parameter(name = 'RRn11',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRn11}',
                  lhablock = 'SNUMIX',
                  lhacode = [ 1, 1 ])

RRn22 = Parameter(name = 'RRn22',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRn22}',
                  lhablock = 'SNUMIX',
                  lhacode = [ 2, 2 ])

RRn33 = Parameter(name = 'RRn33',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRn33}',
                  lhablock = 'SNUMIX',
                  lhacode = [ 3, 3 ])

Rtd33 = Parameter(name = 'Rtd33',
                  nature = 'external',
                  type = 'real',
                  value = -110.693742,
                  texname = '\\text{Rtd33}',
                  lhablock = 'TD',
                  lhacode = [ 3, 3 ])

Rte33 = Parameter(name = 'Rte33',
                  nature = 'external',
                  type = 'real',
                  value = -25.4019727,
                  texname = '\\text{Rte33}',
                  lhablock = 'TE',
                  lhacode = [ 3, 3 ])

Rtu33 = Parameter(name = 'Rtu33',
                  nature = 'external',
                  type = 'real',
                  value = -444.752457,
                  texname = '\\text{Rtu33}',
                  lhablock = 'TU',
                  lhacode = [ 3, 3 ])

RUU11 = Parameter(name = 'RUU11',
                  nature = 'external',
                  type = 'real',
                  value = 0.916834859,
                  texname = '\\text{RUU11}',
                  lhablock = 'UMIX',
                  lhacode = [ 1, 1 ])

RUU12 = Parameter(name = 'RUU12',
                  nature = 'external',
                  type = 'real',
                  value = -0.399266629,
                  texname = '\\text{RUU12}',
                  lhablock = 'UMIX',
                  lhacode = [ 1, 2 ])

RUU21 = Parameter(name = 'RUU21',
                  nature = 'external',
                  type = 'real',
                  value = 0.399266629,
                  texname = '\\text{RUU21}',
                  lhablock = 'UMIX',
                  lhacode = [ 2, 1 ])

RUU22 = Parameter(name = 'RUU22',
                  nature = 'external',
                  type = 'real',
                  value = 0.916834859,
                  texname = '\\text{RUU22}',
                  lhablock = 'UMIX',
                  lhacode = [ 2, 2 ])

RMNS11 = Parameter(name = 'RMNS11',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RMNS11}',
                   lhablock = 'UPMNS',
                   lhacode = [ 1, 1 ])

RMNS22 = Parameter(name = 'RMNS22',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RMNS22}',
                   lhablock = 'UPMNS',
                   lhacode = [ 2, 2 ])

RMNS33 = Parameter(name = 'RMNS33',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RMNS33}',
                   lhablock = 'UPMNS',
                   lhacode = [ 3, 3 ])

RRu11 = Parameter(name = 'RRu11',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu11}',
                  lhablock = 'USQMIX',
                  lhacode = [ 1, 1 ])

RRu22 = Parameter(name = 'RRu22',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu22}',
                  lhablock = 'USQMIX',
                  lhacode = [ 2, 2 ])

RRu33 = Parameter(name = 'RRu33',
                  nature = 'external',
                  type = 'real',
                  value = 0.55364496,
                  texname = '\\text{RRu33}',
                  lhablock = 'USQMIX',
                  lhacode = [ 3, 3 ])

RRu36 = Parameter(name = 'RRu36',
                  nature = 'external',
                  type = 'real',
                  value = 0.83275282,
                  texname = '\\text{RRu36}',
                  lhablock = 'USQMIX',
                  lhacode = [ 3, 6 ])

RRu44 = Parameter(name = 'RRu44',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu44}',
                  lhablock = 'USQMIX',
                  lhacode = [ 4, 4 ])

RRu55 = Parameter(name = 'RRu55',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu55}',
                  lhablock = 'USQMIX',
                  lhacode = [ 5, 5 ])

RRu63 = Parameter(name = 'RRu63',
                  nature = 'external',
                  type = 'real',
                  value = 0.83275282,
                  texname = '\\text{RRu63}',
                  lhablock = 'USQMIX',
                  lhacode = [ 6, 3 ])

RRu66 = Parameter(name = 'RRu66',
                  nature = 'external',
                  type = 'real',
                  value = -0.55364496,
                  texname = '\\text{RRu66}',
                  lhablock = 'USQMIX',
                  lhacode = [ 6, 6 ])

RCKM11 = Parameter(name = 'RCKM11',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RCKM11}',
                   lhablock = 'VCKM',
                   lhacode = [ 1, 1 ])

RCKM22 = Parameter(name = 'RCKM22',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RCKM22}',
                   lhablock = 'VCKM',
                   lhacode = [ 2, 2 ])

RCKM33 = Parameter(name = 'RCKM33',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RCKM33}',
                   lhablock = 'VCKM',
                   lhacode = [ 3, 3 ])

RVV11 = Parameter(name = 'RVV11',
                  nature = 'external',
                  type = 'real',
                  value = 0.972557835,
                  texname = '\\text{RVV11}',
                  lhablock = 'VMIX',
                  lhacode = [ 1, 1 ])

RVV12 = Parameter(name = 'RVV12',
                  nature = 'external',
                  type = 'real',
                  value = -0.232661249,
                  texname = '\\text{RVV12}',
                  lhablock = 'VMIX',
                  lhacode = [ 1, 2 ])

RVV21 = Parameter(name = 'RVV21',
                  nature = 'external',
                  type = 'real',
                  value = 0.232661249,
                  texname = '\\text{RVV21}',
                  lhablock = 'VMIX',
                  lhacode = [ 2, 1 ])

RVV22 = Parameter(name = 'RVV22',
                  nature = 'external',
                  type = 'real',
                  value = 0.972557835,
                  texname = '\\text{RVV22}',
                  lhablock = 'VMIX',
                  lhacode = [ 2, 2 ])

Ryd33 = Parameter(name = 'Ryd33',
                  nature = 'external',
                  type = 'real',
                  value = 0.138840206,
                  texname = '\\text{Ryd33}',
                  lhablock = 'YD',
                  lhacode = [ 3, 3 ])

Rye33 = Parameter(name = 'Rye33',
                  nature = 'external',
                  type = 'real',
                  value = 0.10089081,
                  texname = '\\text{Rye33}',
                  lhablock = 'YE',
                  lhacode = [ 3, 3 ])

Ryu33 = Parameter(name = 'Ryu33',
                  nature = 'external',
                  type = 'real',
                  value = 0.89284455,
                  texname = '\\text{Ryu33}',
                  lhablock = 'YU',
                  lhacode = [ 3, 3 ])

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
               value = 175.,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.88991651,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

Mneu1 = Parameter(name = 'Mneu1',
                  nature = 'external',
                  type = 'real',
                  value = 96.6880686,
                  texname = '\\text{Mneu1}',
                  lhablock = 'MASS',
                  lhacode = [ 1000022 ])

Mneu2 = Parameter(name = 'Mneu2',
                  nature = 'external',
                  type = 'real',
                  value = 181.088157,
                  texname = '\\text{Mneu2}',
                  lhablock = 'MASS',
                  lhacode = [ 1000023 ])

Mneu3 = Parameter(name = 'Mneu3',
                  nature = 'external',
                  type = 'real',
                  value = -363.756027,
                  texname = '\\text{Mneu3}',
                  lhablock = 'MASS',
                  lhacode = [ 1000025 ])

Mneu4 = Parameter(name = 'Mneu4',
                  nature = 'external',
                  type = 'real',
                  value = 381.729382,
                  texname = '\\text{Mneu4}',
                  lhablock = 'MASS',
                  lhacode = [ 1000035 ])

Mch1 = Parameter(name = 'Mch1',
                 nature = 'external',
                 type = 'real',
                 value = 181.696474,
                 texname = '\\text{Mch1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000024 ])

Mch2 = Parameter(name = 'Mch2',
                 nature = 'external',
                 type = 'real',
                 value = 379.93932,
                 texname = '\\text{Mch2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000037 ])

Mglu = Parameter(name = 'Mglu',
                 nature = 'external',
                 type = 'real',
                 value = 607.713704,
                 texname = '\\text{Mglu}',
                 lhablock = 'MASS',
                 lhacode = [ 1000021 ])

Msn1 = Parameter(name = 'Msn1',
                 nature = 'external',
                 type = 'real',
                 value = 185.258326,
                 texname = '\\text{Msn1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000012 ])

Msn2 = Parameter(name = 'Msn2',
                 nature = 'external',
                 type = 'real',
                 value = 185.258326,
                 texname = '\\text{Msn2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000014 ])

Msn3 = Parameter(name = 'Msn3',
                 nature = 'external',
                 type = 'real',
                 value = 184.708464,
                 texname = '\\text{Msn3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000016 ])

Msl1 = Parameter(name = 'Msl1',
                 nature = 'external',
                 type = 'real',
                 value = 202.91569,
                 texname = '\\text{Msl1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000011 ])

Msl2 = Parameter(name = 'Msl2',
                 nature = 'external',
                 type = 'real',
                 value = 202.91569,
                 texname = '\\text{Msl2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000013 ])

Msl3 = Parameter(name = 'Msl3',
                 nature = 'external',
                 type = 'real',
                 value = 134.490864,
                 texname = '\\text{Msl3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000015 ])

Msl4 = Parameter(name = 'Msl4',
                 nature = 'external',
                 type = 'real',
                 value = 144.102799,
                 texname = '\\text{Msl4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000011 ])

Msl5 = Parameter(name = 'Msl5',
                 nature = 'external',
                 type = 'real',
                 value = 144.102799,
                 texname = '\\text{Msl5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000013 ])

Msl6 = Parameter(name = 'Msl6',
                 nature = 'external',
                 type = 'real',
                 value = 206.867805,
                 texname = '\\text{Msl6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000015 ])

Musq1 = Parameter(name = 'Musq1',
                  nature = 'external',
                  type = 'real',
                  value = 561.119014,
                  texname = '\\text{Musq1}',
                  lhablock = 'MASS',
                  lhacode = [ 1000002 ])

Musq2 = Parameter(name = 'Musq2',
                  nature = 'external',
                  type = 'real',
                  value = 561.119014,
                  texname = '\\text{Musq2}',
                  lhablock = 'MASS',
                  lhacode = [ 1000004 ])

Musq3 = Parameter(name = 'Musq3',
                  nature = 'external',
                  type = 'real',
                  value = 399.668493,
                  texname = '\\text{Musq3}',
                  lhablock = 'MASS',
                  lhacode = [ 1000006 ])

Musq4 = Parameter(name = 'Musq4',
                  nature = 'external',
                  type = 'real',
                  value = 549.259265,
                  texname = '\\text{Musq4}',
                  lhablock = 'MASS',
                  lhacode = [ 2000002 ])

Musq5 = Parameter(name = 'Musq5',
                  nature = 'external',
                  type = 'real',
                  value = 549.259265,
                  texname = '\\text{Musq5}',
                  lhablock = 'MASS',
                  lhacode = [ 2000004 ])

Musq6 = Parameter(name = 'Musq6',
                  nature = 'external',
                  type = 'real',
                  value = 585.785818,
                  texname = '\\text{Musq6}',
                  lhablock = 'MASS',
                  lhacode = [ 2000006 ])

Mdsq1 = Parameter(name = 'Mdsq1',
                  nature = 'external',
                  type = 'real',
                  value = 568.441109,
                  texname = '\\text{Mdsq1}',
                  lhablock = 'MASS',
                  lhacode = [ 1000001 ])

Mdsq2 = Parameter(name = 'Mdsq2',
                  nature = 'external',
                  type = 'real',
                  value = 568.441109,
                  texname = '\\text{Mdsq2}',
                  lhablock = 'MASS',
                  lhacode = [ 1000003 ])

Mdsq3 = Parameter(name = 'Mdsq3',
                  nature = 'external',
                  type = 'real',
                  value = 513.065179,
                  texname = '\\text{Mdsq3}',
                  lhablock = 'MASS',
                  lhacode = [ 1000005 ])

Mdsq4 = Parameter(name = 'Mdsq4',
                  nature = 'external',
                  type = 'real',
                  value = 545.228462,
                  texname = '\\text{Mdsq4}',
                  lhablock = 'MASS',
                  lhacode = [ 2000001 ])

Mdsq5 = Parameter(name = 'Mdsq5',
                  nature = 'external',
                  type = 'real',
                  value = 545.228462,
                  texname = '\\text{Mdsq5}',
                  lhablock = 'MASS',
                  lhacode = [ 2000003 ])

Mdsq6 = Parameter(name = 'Mdsq6',
                  nature = 'external',
                  type = 'real',
                  value = 543.726676,
                  texname = '\\text{Mdsq6}',
                  lhablock = 'MASS',
                  lhacode = [ 2000005 ])

Mh01 = Parameter(name = 'Mh01',
                 nature = 'external',
                 type = 'real',
                 value = 110.899057,
                 texname = '\\text{Mh01}',
                 lhablock = 'MASS',
                 lhacode = [ 25 ])

Mh02 = Parameter(name = 'Mh02',
                 nature = 'external',
                 type = 'real',
                 value = 399.960116,
                 texname = '\\text{Mh02}',
                 lhablock = 'MASS',
                 lhacode = [ 35 ])

MA0 = Parameter(name = 'MA0',
                nature = 'external',
                type = 'real',
                value = 399.583917,
                texname = '\\text{MA0}',
                lhablock = 'MASS',
                lhacode = [ 36 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 407.879012,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 37 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.56194983,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

Wneu1 = Parameter(name = 'Wneu1',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{Wneu1}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000022 ])

Wneu2 = Parameter(name = 'Wneu2',
                  nature = 'external',
                  type = 'real',
                  value = 0.0207770048,
                  texname = '\\text{Wneu2}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000023 ])

Wneu3 = Parameter(name = 'Wneu3',
                  nature = 'external',
                  type = 'real',
                  value = 1.91598495,
                  texname = '\\text{Wneu3}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000025 ])

Wneu4 = Parameter(name = 'Wneu4',
                  nature = 'external',
                  type = 'real',
                  value = 2.58585079,
                  texname = '\\text{Wneu4}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000035 ])

Wch1 = Parameter(name = 'Wch1',
                 nature = 'external',
                 type = 'real',
                 value = 0.0170414503,
                 texname = '\\text{Wch1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000024 ])

Wch2 = Parameter(name = 'Wch2',
                 nature = 'external',
                 type = 'real',
                 value = 2.4868951,
                 texname = '\\text{Wch2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000037 ])

Wglu = Parameter(name = 'Wglu',
                 nature = 'external',
                 type = 'real',
                 value = 5.50675438,
                 texname = '\\text{Wglu}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000021 ])

Wsn1 = Parameter(name = 'Wsn1',
                 nature = 'external',
                 type = 'real',
                 value = 0.149881634,
                 texname = '\\text{Wsn1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000012 ])

Wsn2 = Parameter(name = 'Wsn2',
                 nature = 'external',
                 type = 'real',
                 value = 0.149881634,
                 texname = '\\text{Wsn2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000014 ])

Wsn3 = Parameter(name = 'Wsn3',
                 nature = 'external',
                 type = 'real',
                 value = 0.147518977,
                 texname = '\\text{Wsn3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000016 ])

Wsl1 = Parameter(name = 'Wsl1',
                 nature = 'external',
                 type = 'real',
                 value = 0.213682161,
                 texname = '\\text{Wsl1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000011 ])

Wsl2 = Parameter(name = 'Wsl2',
                 nature = 'external',
                 type = 'real',
                 value = 0.213682161,
                 texname = '\\text{Wsl2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000013 ])

Wsl3 = Parameter(name = 'Wsl3',
                 nature = 'external',
                 type = 'real',
                 value = 0.148327268,
                 texname = '\\text{Wsl3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000015 ])

Wsl4 = Parameter(name = 'Wsl4',
                 nature = 'external',
                 type = 'real',
                 value = 0.216121626,
                 texname = '\\text{Wsl4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000011 ])

Wsl5 = Parameter(name = 'Wsl5',
                 nature = 'external',
                 type = 'real',
                 value = 0.216121626,
                 texname = '\\text{Wsl5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000013 ])

Wsl6 = Parameter(name = 'Wsl6',
                 nature = 'external',
                 type = 'real',
                 value = 0.269906096,
                 texname = '\\text{Wsl6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000015 ])

Wusq1 = Parameter(name = 'Wusq1',
                  nature = 'external',
                  type = 'real',
                  value = 5.47719539,
                  texname = '\\text{Wusq1}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000002 ])

Wusq2 = Parameter(name = 'Wusq2',
                  nature = 'external',
                  type = 'real',
                  value = 5.47719539,
                  texname = '\\text{Wusq2}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000004 ])

Wusq3 = Parameter(name = 'Wusq3',
                  nature = 'external',
                  type = 'real',
                  value = 2.02159578,
                  texname = '\\text{Wusq3}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000006 ])

Wusq4 = Parameter(name = 'Wusq4',
                  nature = 'external',
                  type = 'real',
                  value = 1.15297292,
                  texname = '\\text{Wusq4}',
                  lhablock = 'DECAY',
                  lhacode = [ 2000002 ])

Wusq5 = Parameter(name = 'Wusq5',
                  nature = 'external',
                  type = 'real',
                  value = 1.15297292,
                  texname = '\\text{Wusq5}',
                  lhablock = 'DECAY',
                  lhacode = [ 2000004 ])

Wusq6 = Parameter(name = 'Wusq6',
                  nature = 'external',
                  type = 'real',
                  value = 7.37313275,
                  texname = '\\text{Wusq6}',
                  lhablock = 'DECAY',
                  lhacode = [ 2000006 ])

Wdsq1 = Parameter(name = 'Wdsq1',
                  nature = 'external',
                  type = 'real',
                  value = 5.31278772,
                  texname = '\\text{Wdsq1}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000001 ])

Wdsq2 = Parameter(name = 'Wdsq2',
                  nature = 'external',
                  type = 'real',
                  value = 5.31278772,
                  texname = '\\text{Wdsq2}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000003 ])

Wdsq3 = Parameter(name = 'Wdsq3',
                  nature = 'external',
                  type = 'real',
                  value = 3.73627601,
                  texname = '\\text{Wdsq3}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000005 ])

Wdsq4 = Parameter(name = 'Wdsq4',
                  nature = 'external',
                  type = 'real',
                  value = 0.285812308,
                  texname = '\\text{Wdsq4}',
                  lhablock = 'DECAY',
                  lhacode = [ 2000001 ])

Wdsq5 = Parameter(name = 'Wdsq5',
                  nature = 'external',
                  type = 'real',
                  value = 0.285812308,
                  texname = '\\text{Wdsq5}',
                  lhablock = 'DECAY',
                  lhacode = [ 2000003 ])

Wdsq6 = Parameter(name = 'Wdsq6',
                  nature = 'external',
                  type = 'real',
                  value = 0.801566294,
                  texname = '\\text{Wdsq6}',
                  lhablock = 'DECAY',
                  lhacode = [ 2000005 ])

Wh01 = Parameter(name = 'Wh01',
                 nature = 'external',
                 type = 'real',
                 value = 0.00198610799,
                 texname = '\\text{Wh01}',
                 lhablock = 'DECAY',
                 lhacode = [ 25 ])

Wh02 = Parameter(name = 'Wh02',
                 nature = 'external',
                 type = 'real',
                 value = 0.574801389,
                 texname = '\\text{Wh02}',
                 lhablock = 'DECAY',
                 lhacode = [ 35 ])

WA0 = Parameter(name = 'WA0',
                nature = 'external',
                type = 'real',
                value = 0.632178488,
                texname = '\\text{WA0}',
                lhablock = 'DECAY',
                lhacode = [ 36 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.546962813,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 37 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.41143316,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.00282196,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

bet = Parameter(name = 'bet',
                nature = 'internal',
                type = 'real',
                value = 'cmath.atan(tb)',
                texname = '\\beta ')

mDR11 = Parameter(name = 'mDR11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmDR11',
                  texname = '\\text{mDR11}')

mDR22 = Parameter(name = 'mDR22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmDR22',
                  texname = '\\text{mDR22}')

mDR33 = Parameter(name = 'mDR33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmDR33',
                  texname = '\\text{mDR33}')

mER11 = Parameter(name = 'mER11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmER11',
                  texname = '\\text{mER11}')

mER22 = Parameter(name = 'mER22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmER22',
                  texname = '\\text{mER22}')

mER33 = Parameter(name = 'mER33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmER33',
                  texname = '\\text{mER33}')

mLL11 = Parameter(name = 'mLL11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmLL11',
                  texname = '\\text{mLL11}')

mLL22 = Parameter(name = 'mLL22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmLL22',
                  texname = '\\text{mLL22}')

mLL33 = Parameter(name = 'mLL33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmLL33',
                  texname = '\\text{mLL33}')

MMMU = Parameter(name = 'MMMU',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RMU',
                 texname = '\\mu ')

mQL11 = Parameter(name = 'mQL11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmQL11',
                  texname = '\\text{mQL11}')

mQL22 = Parameter(name = 'mQL22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmQL22',
                  texname = '\\text{mQL22}')

mQL33 = Parameter(name = 'mQL33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmQL33',
                  texname = '\\text{mQL33}')

mUR11 = Parameter(name = 'mUR11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmUR11',
                  texname = '\\text{mUR11}')

mUR22 = Parameter(name = 'mUR22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmUR22',
                  texname = '\\text{mUR22}')

mUR33 = Parameter(name = 'mUR33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmUR33',
                  texname = '\\text{mUR33}')

Mx1 = Parameter(name = 'Mx1',
                nature = 'internal',
                type = 'complex',
                value = 'RMx1',
                texname = 'M_1')

Mx2 = Parameter(name = 'Mx2',
                nature = 'internal',
                type = 'complex',
                value = 'RMx2',
                texname = 'M_2')

Mx3 = Parameter(name = 'Mx3',
                nature = 'internal',
                type = 'complex',
                value = 'RMx3',
                texname = 'M_3')

MZ = Parameter(name = 'MZ',
               nature = 'internal',
               type = 'real',
               value = 'MMZ',
               texname = 'm_Z')

NN11 = Parameter(name = 'NN11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN11',
                 texname = '\\text{NN11}')

NN12 = Parameter(name = 'NN12',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN12',
                 texname = '\\text{NN12}')

NN13 = Parameter(name = 'NN13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN13',
                 texname = '\\text{NN13}')

NN14 = Parameter(name = 'NN14',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN14',
                 texname = '\\text{NN14}')

NN21 = Parameter(name = 'NN21',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN21',
                 texname = '\\text{NN21}')

NN22 = Parameter(name = 'NN22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN22',
                 texname = '\\text{NN22}')

NN23 = Parameter(name = 'NN23',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN23',
                 texname = '\\text{NN23}')

NN24 = Parameter(name = 'NN24',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN24',
                 texname = '\\text{NN24}')

NN31 = Parameter(name = 'NN31',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN31',
                 texname = '\\text{NN31}')

NN32 = Parameter(name = 'NN32',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN32',
                 texname = '\\text{NN32}')

NN33 = Parameter(name = 'NN33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN33',
                 texname = '\\text{NN33}')

NN34 = Parameter(name = 'NN34',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN34',
                 texname = '\\text{NN34}')

NN41 = Parameter(name = 'NN41',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN41',
                 texname = '\\text{NN41}')

NN42 = Parameter(name = 'NN42',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN42',
                 texname = '\\text{NN42}')

NN43 = Parameter(name = 'NN43',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN43',
                 texname = '\\text{NN43}')

NN44 = Parameter(name = 'NN44',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN44',
                 texname = '\\text{NN44}')

Rd11 = Parameter(name = 'Rd11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd11',
                 texname = '\\text{Rd11}')

Rd22 = Parameter(name = 'Rd22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd22',
                 texname = '\\text{Rd22}')

Rd33 = Parameter(name = 'Rd33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd33',
                 texname = '\\text{Rd33}')

Rd36 = Parameter(name = 'Rd36',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd36',
                 texname = '\\text{Rd36}')

Rd44 = Parameter(name = 'Rd44',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd44',
                 texname = '\\text{Rd44}')

Rd55 = Parameter(name = 'Rd55',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd55',
                 texname = '\\text{Rd55}')

Rd63 = Parameter(name = 'Rd63',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd63',
                 texname = '\\text{Rd63}')

Rd66 = Parameter(name = 'Rd66',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd66',
                 texname = '\\text{Rd66}')

Rl11 = Parameter(name = 'Rl11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl11',
                 texname = '\\text{Rl11}')

Rl22 = Parameter(name = 'Rl22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl22',
                 texname = '\\text{Rl22}')

Rl33 = Parameter(name = 'Rl33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl33',
                 texname = '\\text{Rl33}')

Rl36 = Parameter(name = 'Rl36',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl36',
                 texname = '\\text{Rl36}')

Rl44 = Parameter(name = 'Rl44',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl44',
                 texname = '\\text{Rl44}')

Rl55 = Parameter(name = 'Rl55',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl55',
                 texname = '\\text{Rl55}')

Rl63 = Parameter(name = 'Rl63',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl63',
                 texname = '\\text{Rl63}')

Rl66 = Parameter(name = 'Rl66',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl66',
                 texname = '\\text{Rl66}')

RLn11 = Parameter(name = 'RLn11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn11',
                  texname = '\\text{RLn11}')

RLn22 = Parameter(name = 'RLn22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn22',
                  texname = '\\text{RLn22}')

RLn33 = Parameter(name = 'RLn33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn33',
                  texname = '\\text{RLn33}')

Ru11 = Parameter(name = 'Ru11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu11',
                 texname = '\\text{Ru11}')

Ru22 = Parameter(name = 'Ru22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu22',
                 texname = '\\text{Ru22}')

Ru33 = Parameter(name = 'Ru33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu33',
                 texname = '\\text{Ru33}')

Ru36 = Parameter(name = 'Ru36',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu36',
                 texname = '\\text{Ru36}')

Ru44 = Parameter(name = 'Ru44',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu44',
                 texname = '\\text{Ru44}')

Ru55 = Parameter(name = 'Ru55',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu55',
                 texname = '\\text{Ru55}')

Ru63 = Parameter(name = 'Ru63',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu63',
                 texname = '\\text{Ru63}')

Ru66 = Parameter(name = 'Ru66',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu66',
                 texname = '\\text{Ru66}')

UU11 = Parameter(name = 'UU11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RUU11',
                 texname = '\\text{UU11}')

UU12 = Parameter(name = 'UU12',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RUU12',
                 texname = '\\text{UU12}')

UU21 = Parameter(name = 'UU21',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RUU21',
                 texname = '\\text{UU21}')

UU22 = Parameter(name = 'UU22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RUU22',
                 texname = '\\text{UU22}')

VV11 = Parameter(name = 'VV11',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RVV11',
                 texname = '\\text{VV11}')

VV12 = Parameter(name = 'VV12',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RVV12',
                 texname = '\\text{VV12}')

VV21 = Parameter(name = 'VV21',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RVV21',
                 texname = '\\text{VV21}')

VV22 = Parameter(name = 'VV22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RVV22',
                 texname = '\\text{VV22}')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(1/aEWM1)*cmath.sqrt(cmath.pi)',
               texname = 'e')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

td33 = Parameter(name = 'td33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rtd33',
                 texname = '\\text{td33}')

te33 = Parameter(name = 'te33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rte33',
                 texname = '\\text{te33}')

tu33 = Parameter(name = 'tu33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rtu33',
                 texname = '\\text{tu33}')

yd33 = Parameter(name = 'yd33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Ryd33',
                 texname = '\\text{yd33}')

ye33 = Parameter(name = 'ye33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rye33',
                 texname = '\\text{ye33}')

yu33 = Parameter(name = 'yu33',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Ryu33',
                 texname = '\\text{yu33}')

bb = Parameter(name = 'bb',
               nature = 'internal',
               type = 'complex',
               value = 'MA2*cmath.sin(2*bet)',
               texname = 'b')

MW = Parameter(name = 'MW',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(MZ**2/2. + cmath.sqrt(MZ**4/4. - (cmath.pi*MZ**2)/(aEWM1*Gf*cmath.sqrt(2))))',
               texname = 'm_W')

cw2 = Parameter(name = 'cw2',
                nature = 'internal',
                type = 'real',
                value = 'MW**2/MZ**2',
                texname = 'c_w^2')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(cw2)',
               texname = 'c_w')

sw2 = Parameter(name = 'sw2',
                nature = 'internal',
                type = 'real',
                value = '1 - cw2',
                texname = 's_w^2')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(sw2)',
               texname = 's_w')

vv = Parameter(name = 'vv',
               nature = 'internal',
               type = 'real',
               value = '(2*cw*MZ*sw)/ee',
               texname = 'v')

vd = Parameter(name = 'vd',
               nature = 'internal',
               type = 'real',
               value = 'vv*cmath.cos(bet)',
               texname = 'v_d')

vu = Parameter(name = 'vu',
               nature = 'internal',
               type = 'real',
               value = 'vv*cmath.sin(bet)',
               texname = 'v_u')

I111 = Parameter(name = 'I111',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd11*complexconjugate(Rd11)',
                 texname = '\\text{I111}')

I122 = Parameter(name = 'I122',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd22*complexconjugate(Rd22)',
                 texname = '\\text{I122}')

I133 = Parameter(name = 'I133',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd33*complexconjugate(Rd33)',
                 texname = '\\text{I133}')

I136 = Parameter(name = 'I136',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd63*complexconjugate(Rd33)',
                 texname = '\\text{I136}')

I163 = Parameter(name = 'I163',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd33*complexconjugate(Rd63)',
                 texname = '\\text{I163}')

I166 = Parameter(name = 'I166',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd63*complexconjugate(Rd63)',
                 texname = '\\text{I166}')

I1033 = Parameter(name = 'I1033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*complexconjugate(Rd36)',
                  texname = '\\text{I1033}')

I1036 = Parameter(name = 'I1036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*complexconjugate(Rd36)',
                  texname = '\\text{I1036}')

I1044 = Parameter(name = 'I1044',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd44*complexconjugate(Rd44)',
                  texname = '\\text{I1044}')

I1055 = Parameter(name = 'I1055',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd55*complexconjugate(Rd55)',
                  texname = '\\text{I1055}')

I1063 = Parameter(name = 'I1063',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*complexconjugate(Rd66)',
                  texname = '\\text{I1063}')

I1066 = Parameter(name = 'I1066',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*complexconjugate(Rd66)',
                  texname = '\\text{I1066}')

I10033 = Parameter(name = 'I10033',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru33*complexconjugate(Ru36)*complexconjugate(yu33)',
                   texname = '\\text{I10033}')

I10036 = Parameter(name = 'I10036',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru63*complexconjugate(Ru36)*complexconjugate(yu33)',
                   texname = '\\text{I10036}')

I10063 = Parameter(name = 'I10063',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru33*complexconjugate(Ru66)*complexconjugate(yu33)',
                   texname = '\\text{I10063}')

I10066 = Parameter(name = 'I10066',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru63*complexconjugate(Ru66)*complexconjugate(yu33)',
                   texname = '\\text{I10066}')

I10133 = Parameter(name = 'I10133',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru33*complexconjugate(Ru36)*complexconjugate(tu33)',
                   texname = '\\text{I10133}')

I10136 = Parameter(name = 'I10136',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru63*complexconjugate(Ru36)*complexconjugate(tu33)',
                   texname = '\\text{I10136}')

I10163 = Parameter(name = 'I10163',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru33*complexconjugate(Ru66)*complexconjugate(tu33)',
                   texname = '\\text{I10163}')

I10166 = Parameter(name = 'I10166',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru63*complexconjugate(Ru66)*complexconjugate(tu33)',
                   texname = '\\text{I10166}')

I10233 = Parameter(name = 'I10233',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru36*tu33*complexconjugate(Ru33)',
                   texname = '\\text{I10233}')

I10236 = Parameter(name = 'I10236',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru66*tu33*complexconjugate(Ru33)',
                   texname = '\\text{I10236}')

I10263 = Parameter(name = 'I10263',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru36*tu33*complexconjugate(Ru63)',
                   texname = '\\text{I10263}')

I10266 = Parameter(name = 'I10266',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru66*tu33*complexconjugate(Ru63)',
                   texname = '\\text{I10266}')

I10333 = Parameter(name = 'I10333',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru33*yu33*complexconjugate(Ru33)*complexconjugate(yu33)',
                   texname = '\\text{I10333}')

I10336 = Parameter(name = 'I10336',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru63*yu33*complexconjugate(Ru33)*complexconjugate(yu33)',
                   texname = '\\text{I10336}')

I10363 = Parameter(name = 'I10363',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru33*yu33*complexconjugate(Ru63)*complexconjugate(yu33)',
                   texname = '\\text{I10363}')

I10366 = Parameter(name = 'I10366',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru63*yu33*complexconjugate(Ru63)*complexconjugate(yu33)',
                   texname = '\\text{I10366}')

I10433 = Parameter(name = 'I10433',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru36*yu33*complexconjugate(Ru36)*complexconjugate(yu33)',
                   texname = '\\text{I10433}')

I10436 = Parameter(name = 'I10436',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru66*yu33*complexconjugate(Ru36)*complexconjugate(yu33)',
                   texname = '\\text{I10436}')

I10463 = Parameter(name = 'I10463',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru36*yu33*complexconjugate(Ru66)*complexconjugate(yu33)',
                   texname = '\\text{I10463}')

I10466 = Parameter(name = 'I10466',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru66*yu33*complexconjugate(Ru66)*complexconjugate(yu33)',
                   texname = '\\text{I10466}')

I10533 = Parameter(name = 'I10533',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru36*yu33*complexconjugate(Ru33)',
                   texname = '\\text{I10533}')

I10536 = Parameter(name = 'I10536',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru66*yu33*complexconjugate(Ru33)',
                   texname = '\\text{I10536}')

I10563 = Parameter(name = 'I10563',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru36*yu33*complexconjugate(Ru63)',
                   texname = '\\text{I10563}')

I10566 = Parameter(name = 'I10566',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru66*yu33*complexconjugate(Ru63)',
                   texname = '\\text{I10566}')

I10633 = Parameter(name = 'I10633',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye33',
                   texname = '\\text{I10633}')

I10733 = Parameter(name = 'I10733',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye33',
                   texname = '\\text{I10733}')

I10833 = Parameter(name = 'I10833',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl36*te33*complexconjugate(RLn33)',
                   texname = '\\text{I10833}')

I10836 = Parameter(name = 'I10836',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl66*te33*complexconjugate(RLn33)',
                   texname = '\\text{I10836}')

I10933 = Parameter(name = 'I10933',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl33*ye33*complexconjugate(RLn33)*complexconjugate(ye33)',
                   texname = '\\text{I10933}')

I10936 = Parameter(name = 'I10936',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl63*ye33*complexconjugate(RLn33)*complexconjugate(ye33)',
                   texname = '\\text{I10936}')

I1111 = Parameter(name = 'I1111',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru11*complexconjugate(Ru11)',
                  texname = '\\text{I1111}')

I1122 = Parameter(name = 'I1122',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru22*complexconjugate(Ru22)',
                  texname = '\\text{I1122}')

I1133 = Parameter(name = 'I1133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Ru33)',
                  texname = '\\text{I1133}')

I1136 = Parameter(name = 'I1136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Ru33)',
                  texname = '\\text{I1136}')

I1163 = Parameter(name = 'I1163',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Ru63)',
                  texname = '\\text{I1163}')

I1166 = Parameter(name = 'I1166',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Ru63)',
                  texname = '\\text{I1166}')

I11033 = Parameter(name = 'I11033',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl36*ye33*complexconjugate(RLn33)',
                   texname = '\\text{I11033}')

I11036 = Parameter(name = 'I11036',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl66*ye33*complexconjugate(RLn33)',
                   texname = '\\text{I11036}')

I1233 = Parameter(name = 'I1233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*complexconjugate(Ru36)',
                  texname = '\\text{I1233}')

I1236 = Parameter(name = 'I1236',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*complexconjugate(Ru36)',
                  texname = '\\text{I1236}')

I1244 = Parameter(name = 'I1244',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru44*complexconjugate(Ru44)',
                  texname = '\\text{I1244}')

I1255 = Parameter(name = 'I1255',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru55*complexconjugate(Ru55)',
                  texname = '\\text{I1255}')

I1263 = Parameter(name = 'I1263',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*complexconjugate(Ru66)',
                  texname = '\\text{I1263}')

I1266 = Parameter(name = 'I1266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*complexconjugate(Ru66)',
                  texname = '\\text{I1266}')

I1311 = Parameter(name = 'I1311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru11*complexconjugate(Ru11)',
                  texname = '\\text{I1311}')

I1322 = Parameter(name = 'I1322',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru22*complexconjugate(Ru22)',
                  texname = '\\text{I1322}')

I1333 = Parameter(name = 'I1333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Ru33)',
                  texname = '\\text{I1333}')

I1336 = Parameter(name = 'I1336',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Ru33)',
                  texname = '\\text{I1336}')

I1363 = Parameter(name = 'I1363',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Ru63)',
                  texname = '\\text{I1363}')

I1366 = Parameter(name = 'I1366',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Ru63)',
                  texname = '\\text{I1366}')

I1433 = Parameter(name = 'I1433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*complexconjugate(Ru36)',
                  texname = '\\text{I1433}')

I1436 = Parameter(name = 'I1436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*complexconjugate(Ru36)',
                  texname = '\\text{I1436}')

I1444 = Parameter(name = 'I1444',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru44*complexconjugate(Ru44)',
                  texname = '\\text{I1444}')

I1455 = Parameter(name = 'I1455',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru55*complexconjugate(Ru55)',
                  texname = '\\text{I1455}')

I1463 = Parameter(name = 'I1463',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*complexconjugate(Ru66)',
                  texname = '\\text{I1463}')

I1466 = Parameter(name = 'I1466',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*complexconjugate(Ru66)',
                  texname = '\\text{I1466}')

I1511 = Parameter(name = 'I1511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd11*complexconjugate(Ru11)',
                  texname = '\\text{I1511}')

I1522 = Parameter(name = 'I1522',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd22*complexconjugate(Ru22)',
                  texname = '\\text{I1522}')

I1533 = Parameter(name = 'I1533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru33)',
                  texname = '\\text{I1533}')

I1536 = Parameter(name = 'I1536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru63)',
                  texname = '\\text{I1536}')

I1563 = Parameter(name = 'I1563',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru33)',
                  texname = '\\text{I1563}')

I1566 = Parameter(name = 'I1566',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru63)',
                  texname = '\\text{I1566}')

I1611 = Parameter(name = 'I1611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru11*complexconjugate(Rd11)',
                  texname = '\\text{I1611}')

I1622 = Parameter(name = 'I1622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru22*complexconjugate(Rd22)',
                  texname = '\\text{I1622}')

I1633 = Parameter(name = 'I1633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd33)',
                  texname = '\\text{I1633}')

I1636 = Parameter(name = 'I1636',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd33)',
                  texname = '\\text{I1636}')

I1663 = Parameter(name = 'I1663',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd63)',
                  texname = '\\text{I1663}')

I1666 = Parameter(name = 'I1666',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd63)',
                  texname = '\\text{I1666}')

I1711 = Parameter(name = 'I1711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn11*complexconjugate(Rl11)',
                  texname = '\\text{I1711}')

I1722 = Parameter(name = 'I1722',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn22*complexconjugate(Rl22)',
                  texname = '\\text{I1722}')

I1733 = Parameter(name = 'I1733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl33)',
                  texname = '\\text{I1733}')

I1736 = Parameter(name = 'I1736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl63)',
                  texname = '\\text{I1736}')

I1811 = Parameter(name = 'I1811',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl11*complexconjugate(RLn11)',
                  texname = '\\text{I1811}')

I1822 = Parameter(name = 'I1822',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl22*complexconjugate(RLn22)',
                  texname = '\\text{I1822}')

I1833 = Parameter(name = 'I1833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(RLn33)',
                  texname = '\\text{I1833}')

I1836 = Parameter(name = 'I1836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(RLn33)',
                  texname = '\\text{I1836}')

I1911 = Parameter(name = 'I1911',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl11*complexconjugate(RLn11)',
                  texname = '\\text{I1911}')

I1922 = Parameter(name = 'I1922',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl22*complexconjugate(RLn22)',
                  texname = '\\text{I1922}')

I1933 = Parameter(name = 'I1933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(RLn33)',
                  texname = '\\text{I1933}')

I1936 = Parameter(name = 'I1936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(RLn33)',
                  texname = '\\text{I1936}')

I233 = Parameter(name = 'I233',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd36*complexconjugate(Rd36)',
                 texname = '\\text{I233}')

I236 = Parameter(name = 'I236',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd66*complexconjugate(Rd36)',
                 texname = '\\text{I236}')

I244 = Parameter(name = 'I244',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd44*complexconjugate(Rd44)',
                 texname = '\\text{I244}')

I255 = Parameter(name = 'I255',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd55*complexconjugate(Rd55)',
                 texname = '\\text{I255}')

I263 = Parameter(name = 'I263',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd36*complexconjugate(Rd66)',
                 texname = '\\text{I263}')

I266 = Parameter(name = 'I266',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd66*complexconjugate(Rd66)',
                 texname = '\\text{I266}')

I2011 = Parameter(name = 'I2011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn11*complexconjugate(Rl11)',
                  texname = '\\text{I2011}')

I2022 = Parameter(name = 'I2022',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn22*complexconjugate(Rl22)',
                  texname = '\\text{I2022}')

I2033 = Parameter(name = 'I2033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl33)',
                  texname = '\\text{I2033}')

I2036 = Parameter(name = 'I2036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl63)',
                  texname = '\\text{I2036}')

I2111 = Parameter(name = 'I2111',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd11*complexconjugate(Ru11)',
                  texname = '\\text{I2111}')

I2122 = Parameter(name = 'I2122',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd22*complexconjugate(Ru22)',
                  texname = '\\text{I2122}')

I2133 = Parameter(name = 'I2133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru33)',
                  texname = '\\text{I2133}')

I2136 = Parameter(name = 'I2136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru63)',
                  texname = '\\text{I2136}')

I2163 = Parameter(name = 'I2163',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru33)',
                  texname = '\\text{I2163}')

I2166 = Parameter(name = 'I2166',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru63)',
                  texname = '\\text{I2166}')

I2211 = Parameter(name = 'I2211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd11*complexconjugate(Ru11)',
                  texname = '\\text{I2211}')

I2222 = Parameter(name = 'I2222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd22*complexconjugate(Ru22)',
                  texname = '\\text{I2222}')

I2233 = Parameter(name = 'I2233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru33)',
                  texname = '\\text{I2233}')

I2236 = Parameter(name = 'I2236',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru63)',
                  texname = '\\text{I2236}')

I2263 = Parameter(name = 'I2263',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru33)',
                  texname = '\\text{I2263}')

I2266 = Parameter(name = 'I2266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru63)',
                  texname = '\\text{I2266}')

I2311 = Parameter(name = 'I2311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru11*complexconjugate(Rd11)',
                  texname = '\\text{I2311}')

I2322 = Parameter(name = 'I2322',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru22*complexconjugate(Rd22)',
                  texname = '\\text{I2322}')

I2333 = Parameter(name = 'I2333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd33)',
                  texname = '\\text{I2333}')

I2336 = Parameter(name = 'I2336',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd33)',
                  texname = '\\text{I2336}')

I2363 = Parameter(name = 'I2363',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd63)',
                  texname = '\\text{I2363}')

I2366 = Parameter(name = 'I2366',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd63)',
                  texname = '\\text{I2366}')

I2411 = Parameter(name = 'I2411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru11*complexconjugate(Rd11)',
                  texname = '\\text{I2411}')

I2422 = Parameter(name = 'I2422',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru22*complexconjugate(Rd22)',
                  texname = '\\text{I2422}')

I2433 = Parameter(name = 'I2433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd33)',
                  texname = '\\text{I2433}')

I2436 = Parameter(name = 'I2436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd33)',
                  texname = '\\text{I2436}')

I2463 = Parameter(name = 'I2463',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd63)',
                  texname = '\\text{I2463}')

I2466 = Parameter(name = 'I2466',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd63)',
                  texname = '\\text{I2466}')

I2511 = Parameter(name = 'I2511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl11*complexconjugate(Rl11)',
                  texname = '\\text{I2511}')

I2522 = Parameter(name = 'I2522',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl22*complexconjugate(Rl22)',
                  texname = '\\text{I2522}')

I2533 = Parameter(name = 'I2533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(Rl33)',
                  texname = '\\text{I2533}')

I2536 = Parameter(name = 'I2536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(Rl33)',
                  texname = '\\text{I2536}')

I2563 = Parameter(name = 'I2563',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(Rl63)',
                  texname = '\\text{I2563}')

I2566 = Parameter(name = 'I2566',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(Rl63)',
                  texname = '\\text{I2566}')

I2611 = Parameter(name = 'I2611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru11*complexconjugate(Ru11)',
                  texname = '\\text{I2611}')

I2622 = Parameter(name = 'I2622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru22*complexconjugate(Ru22)',
                  texname = '\\text{I2622}')

I2633 = Parameter(name = 'I2633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Ru33)',
                  texname = '\\text{I2633}')

I2636 = Parameter(name = 'I2636',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Ru33)',
                  texname = '\\text{I2636}')

I2663 = Parameter(name = 'I2663',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Ru63)',
                  texname = '\\text{I2663}')

I2666 = Parameter(name = 'I2666',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Ru63)',
                  texname = '\\text{I2666}')

I2733 = Parameter(name = 'I2733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*complexconjugate(Rl36)',
                  texname = '\\text{I2733}')

I2736 = Parameter(name = 'I2736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*complexconjugate(Rl36)',
                  texname = '\\text{I2736}')

I2744 = Parameter(name = 'I2744',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl44*complexconjugate(Rl44)',
                  texname = '\\text{I2744}')

I2755 = Parameter(name = 'I2755',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl55*complexconjugate(Rl55)',
                  texname = '\\text{I2755}')

I2763 = Parameter(name = 'I2763',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*complexconjugate(Rl66)',
                  texname = '\\text{I2763}')

I2766 = Parameter(name = 'I2766',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*complexconjugate(Rl66)',
                  texname = '\\text{I2766}')

I2833 = Parameter(name = 'I2833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*complexconjugate(Ru36)',
                  texname = '\\text{I2833}')

I2836 = Parameter(name = 'I2836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*complexconjugate(Ru36)',
                  texname = '\\text{I2836}')

I2844 = Parameter(name = 'I2844',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru44*complexconjugate(Ru44)',
                  texname = '\\text{I2844}')

I2855 = Parameter(name = 'I2855',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru55*complexconjugate(Ru55)',
                  texname = '\\text{I2855}')

I2863 = Parameter(name = 'I2863',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*complexconjugate(Ru66)',
                  texname = '\\text{I2863}')

I2866 = Parameter(name = 'I2866',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*complexconjugate(Ru66)',
                  texname = '\\text{I2866}')

I2933 = Parameter(name = 'I2933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I2933}')

I2936 = Parameter(name = 'I2936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I2936}')

I311 = Parameter(name = 'I311',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd11*complexconjugate(Rd11)',
                 texname = '\\text{I311}')

I322 = Parameter(name = 'I322',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd22*complexconjugate(Rd22)',
                 texname = '\\text{I322}')

I333 = Parameter(name = 'I333',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd33*complexconjugate(Rd33)',
                 texname = '\\text{I333}')

I336 = Parameter(name = 'I336',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd63*complexconjugate(Rd33)',
                 texname = '\\text{I336}')

I363 = Parameter(name = 'I363',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd33*complexconjugate(Rd63)',
                 texname = '\\text{I363}')

I366 = Parameter(name = 'I366',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd63*complexconjugate(Rd63)',
                 texname = '\\text{I366}')

I3033 = Parameter(name = 'I3033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd33*complexconjugate(Rd33)',
                  texname = '\\text{I3033}')

I3036 = Parameter(name = 'I3036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd33*complexconjugate(Rd63)',
                  texname = '\\text{I3036}')

I3133 = Parameter(name = 'I3133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(yd33)',
                  texname = '\\text{I3133}')

I3136 = Parameter(name = 'I3136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(yd33)',
                  texname = '\\text{I3136}')

I3233 = Parameter(name = 'I3233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33',
                  texname = '\\text{I3233}')

I3236 = Parameter(name = 'I3236',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33',
                  texname = '\\text{I3236}')

I3311 = Parameter(name = 'I3311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd11',
                  texname = '\\text{I3311}')

I3322 = Parameter(name = 'I3322',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd22',
                  texname = '\\text{I3322}')

I3333 = Parameter(name = 'I3333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33',
                  texname = '\\text{I3333}')

I3336 = Parameter(name = 'I3336',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63',
                  texname = '\\text{I3336}')

I3433 = Parameter(name = 'I3433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(yu33)',
                  texname = '\\text{I3433}')

I3436 = Parameter(name = 'I3436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(yu33)',
                  texname = '\\text{I3436}')

I3533 = Parameter(name = 'I3533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33',
                  texname = '\\text{I3533}')

I3536 = Parameter(name = 'I3536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33',
                  texname = '\\text{I3536}')

I3633 = Parameter(name = 'I3633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl36)*complexconjugate(ye33)',
                  texname = '\\text{I3633}')

I3636 = Parameter(name = 'I3636',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl66)*complexconjugate(ye33)',
                  texname = '\\text{I3636}')

I3733 = Parameter(name = 'I3733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ye33*complexconjugate(Rl33)',
                  texname = '\\text{I3733}')

I3736 = Parameter(name = 'I3736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ye33*complexconjugate(Rl63)',
                  texname = '\\text{I3736}')

I3833 = Parameter(name = 'I3833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(ye33)',
                  texname = '\\text{I3833}')

I3836 = Parameter(name = 'I3836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(ye33)',
                  texname = '\\text{I3836}')

I3933 = Parameter(name = 'I3933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*ye33',
                  texname = '\\text{I3933}')

I3936 = Parameter(name = 'I3936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*ye33',
                  texname = '\\text{I3936}')

I433 = Parameter(name = 'I433',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd36*complexconjugate(Rd36)',
                 texname = '\\text{I433}')

I436 = Parameter(name = 'I436',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd66*complexconjugate(Rd36)',
                 texname = '\\text{I436}')

I444 = Parameter(name = 'I444',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd44*complexconjugate(Rd44)',
                 texname = '\\text{I444}')

I455 = Parameter(name = 'I455',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd55*complexconjugate(Rd55)',
                 texname = '\\text{I455}')

I463 = Parameter(name = 'I463',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd36*complexconjugate(Rd66)',
                 texname = '\\text{I463}')

I466 = Parameter(name = 'I466',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd66*complexconjugate(Rd66)',
                 texname = '\\text{I466}')

I4011 = Parameter(name = 'I4011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl11',
                  texname = '\\text{I4011}')

I4022 = Parameter(name = 'I4022',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl22',
                  texname = '\\text{I4022}')

I4033 = Parameter(name = 'I4033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33',
                  texname = '\\text{I4033}')

I4036 = Parameter(name = 'I4036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63',
                  texname = '\\text{I4036}')

I4133 = Parameter(name = 'I4133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*ye33',
                  texname = '\\text{I4133}')

I4136 = Parameter(name = 'I4136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*ye33',
                  texname = '\\text{I4136}')

I4211 = Parameter(name = 'I4211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn11',
                  texname = '\\text{I4211}')

I4222 = Parameter(name = 'I4222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn22',
                  texname = '\\text{I4222}')

I4233 = Parameter(name = 'I4233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33',
                  texname = '\\text{I4233}')

I4333 = Parameter(name = 'I4333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(ye33)',
                  texname = '\\text{I4333}')

I4433 = Parameter(name = 'I4433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru36)*complexconjugate(yu33)',
                  texname = '\\text{I4433}')

I4436 = Parameter(name = 'I4436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru66)*complexconjugate(yu33)',
                  texname = '\\text{I4436}')

I4533 = Parameter(name = 'I4533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Ru33)',
                  texname = '\\text{I4533}')

I4536 = Parameter(name = 'I4536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Ru63)',
                  texname = '\\text{I4536}')

I4611 = Parameter(name = 'I4611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru11',
                  texname = '\\text{I4611}')

I4622 = Parameter(name = 'I4622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru22',
                  texname = '\\text{I4622}')

I4633 = Parameter(name = 'I4633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33',
                  texname = '\\text{I4633}')

I4636 = Parameter(name = 'I4636',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63',
                  texname = '\\text{I4636}')

I4733 = Parameter(name = 'I4733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(yd33)',
                  texname = '\\text{I4733}')

I4736 = Parameter(name = 'I4736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(yd33)',
                  texname = '\\text{I4736}')

I4833 = Parameter(name = 'I4833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*yu33',
                  texname = '\\text{I4833}')

I4836 = Parameter(name = 'I4836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*yu33',
                  texname = '\\text{I4836}')

I4933 = Parameter(name = 'I4933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(yu33)',
                  texname = '\\text{I4933}')

I4936 = Parameter(name = 'I4936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(yu33)',
                  texname = '\\text{I4936}')

I511 = Parameter(name = 'I511',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl11*complexconjugate(Rl11)',
                 texname = '\\text{I511}')

I522 = Parameter(name = 'I522',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl22*complexconjugate(Rl22)',
                 texname = '\\text{I522}')

I533 = Parameter(name = 'I533',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl33*complexconjugate(Rl33)',
                 texname = '\\text{I533}')

I536 = Parameter(name = 'I536',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl63*complexconjugate(Rl33)',
                 texname = '\\text{I536}')

I563 = Parameter(name = 'I563',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl33*complexconjugate(Rl63)',
                 texname = '\\text{I563}')

I566 = Parameter(name = 'I566',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl63*complexconjugate(Rl63)',
                 texname = '\\text{I566}')

I5033 = Parameter(name = 'I5033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*yu33',
                  texname = '\\text{I5033}')

I5036 = Parameter(name = 'I5036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*yu33',
                  texname = '\\text{I5036}')

I5111 = Parameter(name = 'I5111',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd11)',
                  texname = '\\text{I5111}')

I5122 = Parameter(name = 'I5122',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd22)',
                  texname = '\\text{I5122}')

I5133 = Parameter(name = 'I5133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd33)',
                  texname = '\\text{I5133}')

I5136 = Parameter(name = 'I5136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd63)',
                  texname = '\\text{I5136}')

I5233 = Parameter(name = 'I5233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I5233}')

I5236 = Parameter(name = 'I5236',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I5236}')

I5333 = Parameter(name = 'I5333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Rd33)',
                  texname = '\\text{I5333}')

I5336 = Parameter(name = 'I5336',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Rd63)',
                  texname = '\\text{I5336}')

I5411 = Parameter(name = 'I5411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl11)',
                  texname = '\\text{I5411}')

I5422 = Parameter(name = 'I5422',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl22)',
                  texname = '\\text{I5422}')

I5433 = Parameter(name = 'I5433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl33)',
                  texname = '\\text{I5433}')

I5436 = Parameter(name = 'I5436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl63)',
                  texname = '\\text{I5436}')

I5533 = Parameter(name = 'I5533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl36)*complexconjugate(ye33)',
                  texname = '\\text{I5533}')

I5536 = Parameter(name = 'I5536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl66)*complexconjugate(ye33)',
                  texname = '\\text{I5536}')

I5611 = Parameter(name = 'I5611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(RLn11)',
                  texname = '\\text{I5611}')

I5622 = Parameter(name = 'I5622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(RLn22)',
                  texname = '\\text{I5622}')

I5633 = Parameter(name = 'I5633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(RLn33)',
                  texname = '\\text{I5633}')

I5711 = Parameter(name = 'I5711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru11)',
                  texname = '\\text{I5711}')

I5722 = Parameter(name = 'I5722',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru22)',
                  texname = '\\text{I5722}')

I5733 = Parameter(name = 'I5733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru33)',
                  texname = '\\text{I5733}')

I5736 = Parameter(name = 'I5736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru63)',
                  texname = '\\text{I5736}')

I5833 = Parameter(name = 'I5833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ye33*complexconjugate(RLn33)',
                  texname = '\\text{I5833}')

I5933 = Parameter(name = 'I5933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru36)*complexconjugate(yu33)',
                  texname = '\\text{I5933}')

I5936 = Parameter(name = 'I5936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru66)*complexconjugate(yu33)',
                  texname = '\\text{I5936}')

I633 = Parameter(name = 'I633',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl36*complexconjugate(Rl36)',
                 texname = '\\text{I633}')

I636 = Parameter(name = 'I636',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl66*complexconjugate(Rl36)',
                 texname = '\\text{I636}')

I644 = Parameter(name = 'I644',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl44*complexconjugate(Rl44)',
                 texname = '\\text{I644}')

I655 = Parameter(name = 'I655',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl55*complexconjugate(Rl55)',
                 texname = '\\text{I655}')

I663 = Parameter(name = 'I663',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl36*complexconjugate(Rl66)',
                 texname = '\\text{I663}')

I666 = Parameter(name = 'I666',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl66*complexconjugate(Rl66)',
                 texname = '\\text{I666}')

I6033 = Parameter(name = 'I6033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd33*complexconjugate(Ru33)',
                  texname = '\\text{I6033}')

I6036 = Parameter(name = 'I6036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd33*complexconjugate(Ru63)',
                  texname = '\\text{I6036}')

I6133 = Parameter(name = 'I6133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(yd33)',
                  texname = '\\text{I6133}')

I6233 = Parameter(name = 'I6233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33',
                  texname = '\\text{I6233}')

I6333 = Parameter(name = 'I6333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(ye33)',
                  texname = '\\text{I6333}')

I6433 = Parameter(name = 'I6433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(yu33)',
                  texname = '\\text{I6433}')

I6533 = Parameter(name = 'I6533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd33',
                  texname = '\\text{I6533}')

I6633 = Parameter(name = 'I6633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I6633}')

I6636 = Parameter(name = 'I6636',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I6636}')

I6663 = Parameter(name = 'I6663',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I6663}')

I6666 = Parameter(name = 'I6666',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I6666}')

I6733 = Parameter(name = 'I6733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd36)*complexconjugate(td33)',
                  texname = '\\text{I6733}')

I6736 = Parameter(name = 'I6736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd36)*complexconjugate(td33)',
                  texname = '\\text{I6736}')

I6763 = Parameter(name = 'I6763',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd66)*complexconjugate(td33)',
                  texname = '\\text{I6763}')

I6766 = Parameter(name = 'I6766',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd66)*complexconjugate(td33)',
                  texname = '\\text{I6766}')

I6833 = Parameter(name = 'I6833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*td33*complexconjugate(Rd33)',
                  texname = '\\text{I6833}')

I6836 = Parameter(name = 'I6836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*td33*complexconjugate(Rd33)',
                  texname = '\\text{I6836}')

I6863 = Parameter(name = 'I6863',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*td33*complexconjugate(Rd63)',
                  texname = '\\text{I6863}')

I6866 = Parameter(name = 'I6866',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*td33*complexconjugate(Rd63)',
                  texname = '\\text{I6866}')

I6933 = Parameter(name = 'I6933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*yd33*complexconjugate(Rd33)*complexconjugate(yd33)',
                  texname = '\\text{I6933}')

I6936 = Parameter(name = 'I6936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*yd33*complexconjugate(Rd33)*complexconjugate(yd33)',
                  texname = '\\text{I6936}')

I6963 = Parameter(name = 'I6963',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*yd33*complexconjugate(Rd63)*complexconjugate(yd33)',
                  texname = '\\text{I6963}')

I6966 = Parameter(name = 'I6966',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*yd33*complexconjugate(Rd63)*complexconjugate(yd33)',
                  texname = '\\text{I6966}')

I711 = Parameter(name = 'I711',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl11*complexconjugate(Rl11)',
                 texname = '\\text{I711}')

I722 = Parameter(name = 'I722',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl22*complexconjugate(Rl22)',
                 texname = '\\text{I722}')

I733 = Parameter(name = 'I733',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl33*complexconjugate(Rl33)',
                 texname = '\\text{I733}')

I736 = Parameter(name = 'I736',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl63*complexconjugate(Rl33)',
                 texname = '\\text{I736}')

I763 = Parameter(name = 'I763',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl33*complexconjugate(Rl63)',
                 texname = '\\text{I763}')

I766 = Parameter(name = 'I766',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl63*complexconjugate(Rl63)',
                 texname = '\\text{I766}')

I7033 = Parameter(name = 'I7033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I7033}')

I7036 = Parameter(name = 'I7036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I7036}')

I7063 = Parameter(name = 'I7063',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I7063}')

I7066 = Parameter(name = 'I7066',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I7066}')

I7133 = Parameter(name = 'I7133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Rd33)',
                  texname = '\\text{I7133}')

I7136 = Parameter(name = 'I7136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Rd33)',
                  texname = '\\text{I7136}')

I7163 = Parameter(name = 'I7163',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Rd63)',
                  texname = '\\text{I7163}')

I7166 = Parameter(name = 'I7166',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Rd63)',
                  texname = '\\text{I7166}')

I7233 = Parameter(name = 'I7233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I7233}')

I7236 = Parameter(name = 'I7236',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I7236}')

I7263 = Parameter(name = 'I7263',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I7263}')

I7266 = Parameter(name = 'I7266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I7266}')

I7333 = Parameter(name = 'I7333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd36)*complexconjugate(td33)',
                  texname = '\\text{I7333}')

I7336 = Parameter(name = 'I7336',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd36)*complexconjugate(td33)',
                  texname = '\\text{I7336}')

I7363 = Parameter(name = 'I7363',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Rd66)*complexconjugate(td33)',
                  texname = '\\text{I7363}')

I7366 = Parameter(name = 'I7366',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Rd66)*complexconjugate(td33)',
                  texname = '\\text{I7366}')

I7433 = Parameter(name = 'I7433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*td33*complexconjugate(Rd33)',
                  texname = '\\text{I7433}')

I7436 = Parameter(name = 'I7436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*td33*complexconjugate(Rd33)',
                  texname = '\\text{I7436}')

I7463 = Parameter(name = 'I7463',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*td33*complexconjugate(Rd63)',
                  texname = '\\text{I7463}')

I7466 = Parameter(name = 'I7466',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*td33*complexconjugate(Rd63)',
                  texname = '\\text{I7466}')

I7533 = Parameter(name = 'I7533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Rd33)',
                  texname = '\\text{I7533}')

I7536 = Parameter(name = 'I7536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Rd33)',
                  texname = '\\text{I7536}')

I7563 = Parameter(name = 'I7563',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Rd63)',
                  texname = '\\text{I7563}')

I7566 = Parameter(name = 'I7566',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Rd63)',
                  texname = '\\text{I7566}')

I7633 = Parameter(name = 'I7633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(ye33)',
                  texname = '\\text{I7633}')

I7733 = Parameter(name = 'I7733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(Rl36)*complexconjugate(ye33)',
                  texname = '\\text{I7733}')

I7736 = Parameter(name = 'I7736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(Rl36)*complexconjugate(ye33)',
                  texname = '\\text{I7736}')

I7763 = Parameter(name = 'I7763',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(Rl66)*complexconjugate(ye33)',
                  texname = '\\text{I7763}')

I7766 = Parameter(name = 'I7766',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(Rl66)*complexconjugate(ye33)',
                  texname = '\\text{I7766}')

I7833 = Parameter(name = 'I7833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(Rl36)*complexconjugate(te33)',
                  texname = '\\text{I7833}')

I7836 = Parameter(name = 'I7836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(Rl36)*complexconjugate(te33)',
                  texname = '\\text{I7836}')

I7863 = Parameter(name = 'I7863',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*complexconjugate(Rl66)*complexconjugate(te33)',
                  texname = '\\text{I7863}')

I7866 = Parameter(name = 'I7866',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*complexconjugate(Rl66)*complexconjugate(te33)',
                  texname = '\\text{I7866}')

I7933 = Parameter(name = 'I7933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*te33*complexconjugate(Rl33)',
                  texname = '\\text{I7933}')

I7936 = Parameter(name = 'I7936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*te33*complexconjugate(Rl33)',
                  texname = '\\text{I7936}')

I7963 = Parameter(name = 'I7963',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*te33*complexconjugate(Rl63)',
                  texname = '\\text{I7963}')

I7966 = Parameter(name = 'I7966',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*te33*complexconjugate(Rl63)',
                  texname = '\\text{I7966}')

I833 = Parameter(name = 'I833',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl36*complexconjugate(Rl36)',
                 texname = '\\text{I833}')

I836 = Parameter(name = 'I836',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl66*complexconjugate(Rl36)',
                 texname = '\\text{I836}')

I844 = Parameter(name = 'I844',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl44*complexconjugate(Rl44)',
                 texname = '\\text{I844}')

I855 = Parameter(name = 'I855',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl55*complexconjugate(Rl55)',
                 texname = '\\text{I855}')

I863 = Parameter(name = 'I863',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl36*complexconjugate(Rl66)',
                 texname = '\\text{I863}')

I866 = Parameter(name = 'I866',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rl66*complexconjugate(Rl66)',
                 texname = '\\text{I866}')

I8033 = Parameter(name = 'I8033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*ye33*complexconjugate(Rl33)*complexconjugate(ye33)',
                  texname = '\\text{I8033}')

I8036 = Parameter(name = 'I8036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*ye33*complexconjugate(Rl33)*complexconjugate(ye33)',
                  texname = '\\text{I8036}')

I8063 = Parameter(name = 'I8063',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl33*ye33*complexconjugate(Rl63)*complexconjugate(ye33)',
                  texname = '\\text{I8063}')

I8066 = Parameter(name = 'I8066',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl63*ye33*complexconjugate(Rl63)*complexconjugate(ye33)',
                  texname = '\\text{I8066}')

I8133 = Parameter(name = 'I8133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*ye33*complexconjugate(Rl36)*complexconjugate(ye33)',
                  texname = '\\text{I8133}')

I8136 = Parameter(name = 'I8136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*ye33*complexconjugate(Rl36)*complexconjugate(ye33)',
                  texname = '\\text{I8136}')

I8163 = Parameter(name = 'I8163',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*ye33*complexconjugate(Rl66)*complexconjugate(ye33)',
                  texname = '\\text{I8163}')

I8166 = Parameter(name = 'I8166',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*ye33*complexconjugate(Rl66)*complexconjugate(ye33)',
                  texname = '\\text{I8166}')

I8233 = Parameter(name = 'I8233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*ye33*complexconjugate(Rl33)',
                  texname = '\\text{I8233}')

I8236 = Parameter(name = 'I8236',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*ye33*complexconjugate(Rl33)',
                  texname = '\\text{I8236}')

I8263 = Parameter(name = 'I8263',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl36*ye33*complexconjugate(Rl63)',
                  texname = '\\text{I8263}')

I8266 = Parameter(name = 'I8266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl66*ye33*complexconjugate(Rl63)',
                  texname = '\\text{I8266}')

I8333 = Parameter(name = 'I8333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl36)*complexconjugate(ye33)',
                  texname = '\\text{I8333}')

I8336 = Parameter(name = 'I8336',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl66)*complexconjugate(ye33)',
                  texname = '\\text{I8336}')

I8433 = Parameter(name = 'I8433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl36)*complexconjugate(te33)',
                  texname = '\\text{I8433}')

I8436 = Parameter(name = 'I8436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*complexconjugate(Rl66)*complexconjugate(te33)',
                  texname = '\\text{I8436}')

I8533 = Parameter(name = 'I8533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*ye33*complexconjugate(Rl33)*complexconjugate(ye33)',
                  texname = '\\text{I8533}')

I8536 = Parameter(name = 'I8536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RLn33*ye33*complexconjugate(Rl63)*complexconjugate(ye33)',
                  texname = '\\text{I8536}')

I8633 = Parameter(name = 'I8633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru36)*complexconjugate(yu33)',
                  texname = '\\text{I8633}')

I8636 = Parameter(name = 'I8636',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru66)*complexconjugate(yu33)',
                  texname = '\\text{I8636}')

I8663 = Parameter(name = 'I8663',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru36)*complexconjugate(yu33)',
                  texname = '\\text{I8663}')

I8666 = Parameter(name = 'I8666',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru66)*complexconjugate(yu33)',
                  texname = '\\text{I8666}')

I8733 = Parameter(name = 'I8733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru36)*complexconjugate(tu33)',
                  texname = '\\text{I8733}')

I8736 = Parameter(name = 'I8736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*complexconjugate(Ru66)*complexconjugate(tu33)',
                  texname = '\\text{I8736}')

I8763 = Parameter(name = 'I8763',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru36)*complexconjugate(tu33)',
                  texname = '\\text{I8763}')

I8766 = Parameter(name = 'I8766',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*complexconjugate(Ru66)*complexconjugate(tu33)',
                  texname = '\\text{I8766}')

I8833 = Parameter(name = 'I8833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*td33*complexconjugate(Ru33)',
                  texname = '\\text{I8833}')

I8836 = Parameter(name = 'I8836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*td33*complexconjugate(Ru63)',
                  texname = '\\text{I8836}')

I8863 = Parameter(name = 'I8863',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*td33*complexconjugate(Ru33)',
                  texname = '\\text{I8863}')

I8866 = Parameter(name = 'I8866',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*td33*complexconjugate(Ru63)',
                  texname = '\\text{I8866}')

I8933 = Parameter(name = 'I8933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*yd33*complexconjugate(Ru33)*complexconjugate(yd33)',
                  texname = '\\text{I8933}')

I8936 = Parameter(name = 'I8936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*yd33*complexconjugate(Ru63)*complexconjugate(yd33)',
                  texname = '\\text{I8936}')

I8963 = Parameter(name = 'I8963',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*yd33*complexconjugate(Ru33)*complexconjugate(yd33)',
                  texname = '\\text{I8963}')

I8966 = Parameter(name = 'I8966',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*yd33*complexconjugate(Ru63)*complexconjugate(yd33)',
                  texname = '\\text{I8966}')

I911 = Parameter(name = 'I911',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd11*complexconjugate(Rd11)',
                 texname = '\\text{I911}')

I922 = Parameter(name = 'I922',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd22*complexconjugate(Rd22)',
                 texname = '\\text{I922}')

I933 = Parameter(name = 'I933',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd33*complexconjugate(Rd33)',
                 texname = '\\text{I933}')

I936 = Parameter(name = 'I936',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd63*complexconjugate(Rd33)',
                 texname = '\\text{I936}')

I963 = Parameter(name = 'I963',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd33*complexconjugate(Rd63)',
                 texname = '\\text{I963}')

I966 = Parameter(name = 'I966',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd63*complexconjugate(Rd63)',
                 texname = '\\text{I966}')

I9033 = Parameter(name = 'I9033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Ru36)*complexconjugate(yu33)',
                  texname = '\\text{I9033}')

I9036 = Parameter(name = 'I9036',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Ru66)*complexconjugate(yu33)',
                  texname = '\\text{I9036}')

I9063 = Parameter(name = 'I9063',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Ru36)*complexconjugate(yu33)',
                  texname = '\\text{I9063}')

I9066 = Parameter(name = 'I9066',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Ru66)*complexconjugate(yu33)',
                  texname = '\\text{I9066}')

I9133 = Parameter(name = 'I9133',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Ru33)',
                  texname = '\\text{I9133}')

I9136 = Parameter(name = 'I9136',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd36*yd33*complexconjugate(Ru63)',
                  texname = '\\text{I9136}')

I9163 = Parameter(name = 'I9163',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Ru33)',
                  texname = '\\text{I9163}')

I9166 = Parameter(name = 'I9166',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd66*yd33*complexconjugate(Ru63)',
                  texname = '\\text{I9166}')

I9233 = Parameter(name = 'I9233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*yu33*complexconjugate(Ru33)*complexconjugate(yu33)',
                  texname = '\\text{I9233}')

I9236 = Parameter(name = 'I9236',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd33*yu33*complexconjugate(Ru63)*complexconjugate(yu33)',
                  texname = '\\text{I9236}')

I9263 = Parameter(name = 'I9263',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*yu33*complexconjugate(Ru33)*complexconjugate(yu33)',
                  texname = '\\text{I9263}')

I9266 = Parameter(name = 'I9266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd63*yu33*complexconjugate(Ru63)*complexconjugate(yu33)',
                  texname = '\\text{I9266}')

I9333 = Parameter(name = 'I9333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I9333}')

I9336 = Parameter(name = 'I9336',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I9336}')

I9363 = Parameter(name = 'I9363',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I9363}')

I9366 = Parameter(name = 'I9366',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I9366}')

I9433 = Parameter(name = 'I9433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd36)*complexconjugate(td33)',
                  texname = '\\text{I9433}')

I9436 = Parameter(name = 'I9436',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd36)*complexconjugate(td33)',
                  texname = '\\text{I9436}')

I9463 = Parameter(name = 'I9463',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*complexconjugate(Rd66)*complexconjugate(td33)',
                  texname = '\\text{I9463}')

I9466 = Parameter(name = 'I9466',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*complexconjugate(Rd66)*complexconjugate(td33)',
                  texname = '\\text{I9466}')

I9533 = Parameter(name = 'I9533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*tu33*complexconjugate(Rd33)',
                  texname = '\\text{I9533}')

I9536 = Parameter(name = 'I9536',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*tu33*complexconjugate(Rd33)',
                  texname = '\\text{I9536}')

I9563 = Parameter(name = 'I9563',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*tu33*complexconjugate(Rd63)',
                  texname = '\\text{I9563}')

I9566 = Parameter(name = 'I9566',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*tu33*complexconjugate(Rd63)',
                  texname = '\\text{I9566}')

I9633 = Parameter(name = 'I9633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*yd33*complexconjugate(Rd33)*complexconjugate(yd33)',
                  texname = '\\text{I9633}')

I9636 = Parameter(name = 'I9636',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*yd33*complexconjugate(Rd33)*complexconjugate(yd33)',
                  texname = '\\text{I9636}')

I9663 = Parameter(name = 'I9663',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*yd33*complexconjugate(Rd63)*complexconjugate(yd33)',
                  texname = '\\text{I9663}')

I9666 = Parameter(name = 'I9666',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*yd33*complexconjugate(Rd63)*complexconjugate(yd33)',
                  texname = '\\text{I9666}')

I9733 = Parameter(name = 'I9733',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*yu33*complexconjugate(Rd33)*complexconjugate(yu33)',
                  texname = '\\text{I9733}')

I9736 = Parameter(name = 'I9736',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*yu33*complexconjugate(Rd33)*complexconjugate(yu33)',
                  texname = '\\text{I9736}')

I9763 = Parameter(name = 'I9763',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru33*yu33*complexconjugate(Rd63)*complexconjugate(yu33)',
                  texname = '\\text{I9763}')

I9766 = Parameter(name = 'I9766',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru63*yu33*complexconjugate(Rd63)*complexconjugate(yu33)',
                  texname = '\\text{I9766}')

I9833 = Parameter(name = 'I9833',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*yu33*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I9833}')

I9836 = Parameter(name = 'I9836',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*yu33*complexconjugate(Rd36)*complexconjugate(yd33)',
                  texname = '\\text{I9836}')

I9863 = Parameter(name = 'I9863',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*yu33*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I9863}')

I9866 = Parameter(name = 'I9866',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*yu33*complexconjugate(Rd66)*complexconjugate(yd33)',
                  texname = '\\text{I9866}')

I9933 = Parameter(name = 'I9933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*yu33*complexconjugate(Rd33)',
                  texname = '\\text{I9933}')

I9936 = Parameter(name = 'I9936',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*yu33*complexconjugate(Rd33)',
                  texname = '\\text{I9936}')

I9963 = Parameter(name = 'I9963',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru36*yu33*complexconjugate(Rd63)',
                  texname = '\\text{I9963}')

I9966 = Parameter(name = 'I9966',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru66*yu33*complexconjugate(Rd63)',
                  texname = '\\text{I9966}')

