# This file was automatically created by FeynRules 1.7.53
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Tue 31 Jul 2012 19:55:14



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
RRd1x1 = Parameter(name = 'RRd1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd1x1}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 1, 1 ])

RRd2x2 = Parameter(name = 'RRd2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd2x2}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 2, 2 ])

RRd3x3 = Parameter(name = 'RRd3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.938737896,
                   texname = '\\text{RRd3x3}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 3, 3 ])

RRd3x6 = Parameter(name = 'RRd3x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.344631925,
                   texname = '\\text{RRd3x6}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 3, 6 ])

RRd4x4 = Parameter(name = 'RRd4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd4x4}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 4, 4 ])

RRd5x5 = Parameter(name = 'RRd5x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd5x5}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 5, 5 ])

RRd6x3 = Parameter(name = 'RRd6x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.344631925,
                   texname = '\\text{RRd6x3}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 6, 3 ])

RRd6x6 = Parameter(name = 'RRd6x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.938737896,
                   texname = '\\text{RRd6x6}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 6, 6 ])

alp = Parameter(name = 'alp',
                nature = 'external',
                type = 'real',
                value = -0.11382521,
                texname = '\\alpha',
                lhablock = 'FRALPHA',
                lhacode = [ 1 ])

RMUH = Parameter(name = 'RMUH',
                 nature = 'external',
                 type = 'real',
                 value = 357.680977,
                 texname = '\\text{RMUH}',
                 lhablock = 'HMIX',
                 lhacode = [ 1 ])

tb = Parameter(name = 'tb',
               nature = 'external',
               type = 'real',
               value = 9.74862403,
               texname = 't_b',
               lhablock = 'HMIX',
               lhacode = [ 2 ])

MA2 = Parameter(name = 'MA2',
                nature = 'external',
                type = 'real',
                value = 166439.065,
                texname = 'm_A^2',
                lhablock = 'HMIX',
                lhacode = [ 4 ])

RmD21x1 = Parameter(name = 'RmD21x1',
                    nature = 'external',
                    type = 'real',
                    value = 273684.674,
                    texname = '\\text{RmD21x1}',
                    lhablock = 'MSD2',
                    lhacode = [ 1, 1 ])

RmD22x2 = Parameter(name = 'RmD22x2',
                    nature = 'external',
                    type = 'real',
                    value = 273684.674,
                    texname = '\\text{RmD22x2}',
                    lhablock = 'MSD2',
                    lhacode = [ 2, 2 ])

RmD23x3 = Parameter(name = 'RmD23x3',
                    nature = 'external',
                    type = 'real',
                    value = 270261.969,
                    texname = '\\text{RmD23x3}',
                    lhablock = 'MSD2',
                    lhacode = [ 3, 3 ])

RmE21x1 = Parameter(name = 'RmE21x1',
                    nature = 'external',
                    type = 'real',
                    value = 18630.6287,
                    texname = '\\text{RmE21x1}',
                    lhablock = 'MSE2',
                    lhacode = [ 1, 1 ])

RmE22x2 = Parameter(name = 'RmE22x2',
                    nature = 'external',
                    type = 'real',
                    value = 18630.6287,
                    texname = '\\text{RmE22x2}',
                    lhablock = 'MSE2',
                    lhacode = [ 2, 2 ])

RmE23x3 = Parameter(name = 'RmE23x3',
                    nature = 'external',
                    type = 'real',
                    value = 17967.6406,
                    texname = '\\text{RmE23x3}',
                    lhablock = 'MSE2',
                    lhacode = [ 3, 3 ])

RmL21x1 = Parameter(name = 'RmL21x1',
                    nature = 'external',
                    type = 'real',
                    value = 38155.67,
                    texname = '\\text{RmL21x1}',
                    lhablock = 'MSL2',
                    lhacode = [ 1, 1 ])

RmL22x2 = Parameter(name = 'RmL22x2',
                    nature = 'external',
                    type = 'real',
                    value = 38155.67,
                    texname = '\\text{RmL22x2}',
                    lhablock = 'MSL2',
                    lhacode = [ 2, 2 ])

RmL23x3 = Parameter(name = 'RmL23x3',
                    nature = 'external',
                    type = 'real',
                    value = 37828.6769,
                    texname = '\\text{RmL23x3}',
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

mHd2 = Parameter(name = 'mHd2',
                 nature = 'external',
                 type = 'real',
                 value = 32337.4943,
                 texname = 'm_{H_d}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 21 ])

mHu2 = Parameter(name = 'mHu2',
                 nature = 'external',
                 type = 'real',
                 value = -128800.134,
                 texname = 'm_{H_u}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 22 ])

RmQ21x1 = Parameter(name = 'RmQ21x1',
                    nature = 'external',
                    type = 'real',
                    value = 299836.701,
                    texname = '\\text{RmQ21x1}',
                    lhablock = 'MSQ2',
                    lhacode = [ 1, 1 ])

RmQ22x2 = Parameter(name = 'RmQ22x2',
                    nature = 'external',
                    type = 'real',
                    value = 299836.701,
                    texname = '\\text{RmQ22x2}',
                    lhablock = 'MSQ2',
                    lhacode = [ 2, 2 ])

RmQ23x3 = Parameter(name = 'RmQ23x3',
                    nature = 'external',
                    type = 'real',
                    value = 248765.367,
                    texname = '\\text{RmQ23x3}',
                    lhablock = 'MSQ2',
                    lhacode = [ 3, 3 ])

RmU21x1 = Parameter(name = 'RmU21x1',
                    nature = 'external',
                    type = 'real',
                    value = 280382.106,
                    texname = '\\text{RmU21x1}',
                    lhablock = 'MSU2',
                    lhacode = [ 1, 1 ])

RmU22x2 = Parameter(name = 'RmU22x2',
                    nature = 'external',
                    type = 'real',
                    value = 280382.106,
                    texname = '\\text{RmU22x2}',
                    lhablock = 'MSU2',
                    lhacode = [ 2, 2 ])

RmU23x3 = Parameter(name = 'RmU23x3',
                    nature = 'external',
                    type = 'real',
                    value = 179137.072,
                    texname = '\\text{RmU23x3}',
                    lhablock = 'MSU2',
                    lhacode = [ 3, 3 ])

RNN1x1 = Parameter(name = 'RNN1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.98636443,
                   texname = '\\text{RNN1x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 1 ])

RNN1x2 = Parameter(name = 'RNN1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.0531103553,
                   texname = '\\text{RNN1x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 2 ])

RNN1x3 = Parameter(name = 'RNN1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.146433995,
                   texname = '\\text{RNN1x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 3 ])

RNN1x4 = Parameter(name = 'RNN1x4',
                   nature = 'external',
                   type = 'real',
                   value = -0.0531186117,
                   texname = '\\text{RNN1x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 1, 4 ])

RNN2x1 = Parameter(name = 'RNN2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.0993505358,
                   texname = '\\text{RNN2x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 1 ])

RNN2x2 = Parameter(name = 'RNN2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.944949299,
                   texname = '\\text{RNN2x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 2 ])

RNN2x3 = Parameter(name = 'RNN2x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.26984672,
                   texname = '\\text{RNN2x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 3 ])

RNN2x4 = Parameter(name = 'RNN2x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.156150698,
                   texname = '\\text{RNN2x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 2, 4 ])

RNN3x1 = Parameter(name = 'RNN3x1',
                   nature = 'external',
                   type = 'real',
                   value = -0.0603388002,
                   texname = '\\text{RNN3x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 1 ])

RNN3x2 = Parameter(name = 'RNN3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.0877004854,
                   texname = '\\text{RNN3x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 2 ])

RNN3x3 = Parameter(name = 'RNN3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.695877493,
                   texname = '\\text{RNN3x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 3 ])

RNN3x4 = Parameter(name = 'RNN3x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.710226984,
                   texname = '\\text{RNN3x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 3, 4 ])

RNN4x1 = Parameter(name = 'RNN4x1',
                   nature = 'external',
                   type = 'real',
                   value = -0.116507132,
                   texname = '\\text{RNN4x1}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 1 ])

RNN4x2 = Parameter(name = 'RNN4x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.310739017,
                   texname = '\\text{RNN4x2}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 2 ])

RNN4x3 = Parameter(name = 'RNN4x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.64922596,
                   texname = '\\text{RNN4x3}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 3 ])

RNN4x4 = Parameter(name = 'RNN4x4',
                   nature = 'external',
                   type = 'real',
                   value = -0.684377823,
                   texname = '\\text{RNN4x4}',
                   lhablock = 'NMIX',
                   lhacode = [ 4, 4 ])

RRl1x1 = Parameter(name = 'RRl1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl1x1}',
                   lhablock = 'SELMIX',
                   lhacode = [ 1, 1 ])

RRl2x2 = Parameter(name = 'RRl2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl2x2}',
                   lhablock = 'SELMIX',
                   lhacode = [ 2, 2 ])

RRl3x3 = Parameter(name = 'RRl3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.28248719,
                   texname = '\\text{RRl3x3}',
                   lhablock = 'SELMIX',
                   lhacode = [ 3, 3 ])

RRl3x6 = Parameter(name = 'RRl3x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.959271071,
                   texname = '\\text{RRl3x6}',
                   lhablock = 'SELMIX',
                   lhacode = [ 3, 6 ])

RRl4x4 = Parameter(name = 'RRl4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl4x4}',
                   lhablock = 'SELMIX',
                   lhacode = [ 4, 4 ])

RRl5x5 = Parameter(name = 'RRl5x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl5x5}',
                   lhablock = 'SELMIX',
                   lhacode = [ 5, 5 ])

RRl6x3 = Parameter(name = 'RRl6x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.959271071,
                   texname = '\\text{RRl6x3}',
                   lhablock = 'SELMIX',
                   lhacode = [ 6, 3 ])

RRl6x6 = Parameter(name = 'RRl6x6',
                   nature = 'external',
                   type = 'real',
                   value = -0.28248719,
                   texname = '\\text{RRl6x6}',
                   lhablock = 'SELMIX',
                   lhacode = [ 6, 6 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.934,
                  texname = '\\alpha _w^{-1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.118,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

RRn1x1 = Parameter(name = 'RRn1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn1x1}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 1, 1 ])

RRn2x2 = Parameter(name = 'RRn2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn2x2}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 2, 2 ])

RRn3x3 = Parameter(name = 'RRn3x3',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn3x3}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 3, 3 ])

Rtd3x3 = Parameter(name = 'Rtd3x3',
                   nature = 'external',
                   type = 'real',
                   value = -110.693742,
                   texname = '\\text{Rtd3x3}',
                   lhablock = 'TD',
                   lhacode = [ 3, 3 ])

Rte3x3 = Parameter(name = 'Rte3x3',
                   nature = 'external',
                   type = 'real',
                   value = -25.4019727,
                   texname = '\\text{Rte3x3}',
                   lhablock = 'TE',
                   lhacode = [ 3, 3 ])

Rtu3x3 = Parameter(name = 'Rtu3x3',
                   nature = 'external',
                   type = 'real',
                   value = -444.752457,
                   texname = '\\text{Rtu3x3}',
                   lhablock = 'TU',
                   lhacode = [ 3, 3 ])

RUU1x1 = Parameter(name = 'RUU1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.916834859,
                   texname = '\\text{RUU1x1}',
                   lhablock = 'UMIX',
                   lhacode = [ 1, 1 ])

RUU1x2 = Parameter(name = 'RUU1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.399266629,
                   texname = '\\text{RUU1x2}',
                   lhablock = 'UMIX',
                   lhacode = [ 1, 2 ])

RUU2x1 = Parameter(name = 'RUU2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.399266629,
                   texname = '\\text{RUU2x1}',
                   lhablock = 'UMIX',
                   lhacode = [ 2, 1 ])

RUU2x2 = Parameter(name = 'RUU2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.916834859,
                   texname = '\\text{RUU2x2}',
                   lhablock = 'UMIX',
                   lhacode = [ 2, 2 ])

RMNS1x1 = Parameter(name = 'RMNS1x1',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RMNS1x1}',
                    lhablock = 'UPMNS',
                    lhacode = [ 1, 1 ])

RMNS2x2 = Parameter(name = 'RMNS2x2',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RMNS2x2}',
                    lhablock = 'UPMNS',
                    lhacode = [ 2, 2 ])

RMNS3x3 = Parameter(name = 'RMNS3x3',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RMNS3x3}',
                    lhablock = 'UPMNS',
                    lhacode = [ 3, 3 ])

RRu1x1 = Parameter(name = 'RRu1x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu1x1}',
                   lhablock = 'USQMIX',
                   lhacode = [ 1, 1 ])

RRu2x2 = Parameter(name = 'RRu2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu2x2}',
                   lhablock = 'USQMIX',
                   lhacode = [ 2, 2 ])

RRu3x3 = Parameter(name = 'RRu3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.55364496,
                   texname = '\\text{RRu3x3}',
                   lhablock = 'USQMIX',
                   lhacode = [ 3, 3 ])

RRu3x6 = Parameter(name = 'RRu3x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.83275282,
                   texname = '\\text{RRu3x6}',
                   lhablock = 'USQMIX',
                   lhacode = [ 3, 6 ])

RRu4x4 = Parameter(name = 'RRu4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu4x4}',
                   lhablock = 'USQMIX',
                   lhacode = [ 4, 4 ])

RRu5x5 = Parameter(name = 'RRu5x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu5x5}',
                   lhablock = 'USQMIX',
                   lhacode = [ 5, 5 ])

RRu6x3 = Parameter(name = 'RRu6x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.83275282,
                   texname = '\\text{RRu6x3}',
                   lhablock = 'USQMIX',
                   lhacode = [ 6, 3 ])

RRu6x6 = Parameter(name = 'RRu6x6',
                   nature = 'external',
                   type = 'real',
                   value = -0.55364496,
                   texname = '\\text{RRu6x6}',
                   lhablock = 'USQMIX',
                   lhacode = [ 6, 6 ])

RCKM1x1 = Parameter(name = 'RCKM1x1',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RCKM1x1}',
                    lhablock = 'VCKM',
                    lhacode = [ 1, 1 ])

RCKM2x2 = Parameter(name = 'RCKM2x2',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RCKM2x2}',
                    lhablock = 'VCKM',
                    lhacode = [ 2, 2 ])

RCKM3x3 = Parameter(name = 'RCKM3x3',
                    nature = 'external',
                    type = 'real',
                    value = 1.,
                    texname = '\\text{RCKM3x3}',
                    lhablock = 'VCKM',
                    lhacode = [ 3, 3 ])

RVV1x1 = Parameter(name = 'RVV1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.972557835,
                   texname = '\\text{RVV1x1}',
                   lhablock = 'VMIX',
                   lhacode = [ 1, 1 ])

RVV1x2 = Parameter(name = 'RVV1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.232661249,
                   texname = '\\text{RVV1x2}',
                   lhablock = 'VMIX',
                   lhacode = [ 1, 2 ])

RVV2x1 = Parameter(name = 'RVV2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.232661249,
                   texname = '\\text{RVV2x1}',
                   lhablock = 'VMIX',
                   lhacode = [ 2, 1 ])

RVV2x2 = Parameter(name = 'RVV2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.972557835,
                   texname = '\\text{RVV2x2}',
                   lhablock = 'VMIX',
                   lhacode = [ 2, 2 ])

Ryd3x3 = Parameter(name = 'Ryd3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.138840206,
                   texname = '\\text{Ryd3x3}',
                   lhablock = 'YD',
                   lhacode = [ 3, 3 ])

Rye3x3 = Parameter(name = 'Rye3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.10089081,
                   texname = '\\text{Rye3x3}',
                   lhablock = 'YE',
                   lhacode = [ 3, 3 ])

Ryu3x3 = Parameter(name = 'Ryu3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.89284455,
                   texname = '\\text{Ryu3x3}',
                   lhablock = 'YU',
                   lhacode = [ 3, 3 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.1876,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 79.8290131,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

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

Mgo = Parameter(name = 'Mgo',
                nature = 'external',
                type = 'real',
                value = 607.713704,
                texname = '\\text{Mgo}',
                lhablock = 'MASS',
                lhacode = [ 1000021 ])

MH01 = Parameter(name = 'MH01',
                 nature = 'external',
                 type = 'real',
                 value = 110.899057,
                 texname = '\\text{MH01}',
                 lhablock = 'MASS',
                 lhacode = [ 25 ])

MH02 = Parameter(name = 'MH02',
                 nature = 'external',
                 type = 'real',
                 value = 399.960116,
                 texname = '\\text{MH02}',
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

Mta = Parameter(name = 'Mta',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{Mta}',
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

Msu1 = Parameter(name = 'Msu1',
                 nature = 'external',
                 type = 'real',
                 value = 561.119014,
                 texname = '\\text{Msu1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000002 ])

Msu2 = Parameter(name = 'Msu2',
                 nature = 'external',
                 type = 'real',
                 value = 561.119014,
                 texname = '\\text{Msu2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000004 ])

Msu3 = Parameter(name = 'Msu3',
                 nature = 'external',
                 type = 'real',
                 value = 399.668493,
                 texname = '\\text{Msu3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000006 ])

Msu4 = Parameter(name = 'Msu4',
                 nature = 'external',
                 type = 'real',
                 value = 549.259265,
                 texname = '\\text{Msu4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000002 ])

Msu5 = Parameter(name = 'Msu5',
                 nature = 'external',
                 type = 'real',
                 value = 549.259265,
                 texname = '\\text{Msu5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000004 ])

Msu6 = Parameter(name = 'Msu6',
                 nature = 'external',
                 type = 'real',
                 value = 585.785818,
                 texname = '\\text{Msu6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000006 ])

Msd1 = Parameter(name = 'Msd1',
                 nature = 'external',
                 type = 'real',
                 value = 568.441109,
                 texname = '\\text{Msd1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000001 ])

Msd2 = Parameter(name = 'Msd2',
                 nature = 'external',
                 type = 'real',
                 value = 568.441109,
                 texname = '\\text{Msd2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000003 ])

Msd3 = Parameter(name = 'Msd3',
                 nature = 'external',
                 type = 'real',
                 value = 513.065179,
                 texname = '\\text{Msd3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000005 ])

Msd4 = Parameter(name = 'Msd4',
                 nature = 'external',
                 type = 'real',
                 value = 545.228462,
                 texname = '\\text{Msd4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000001 ])

Msd5 = Parameter(name = 'Msd5',
                 nature = 'external',
                 type = 'real',
                 value = 545.228462,
                 texname = '\\text{Msd5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000003 ])

Msd6 = Parameter(name = 'Msd6',
                 nature = 'external',
                 type = 'real',
                 value = 543.726676,
                 texname = '\\text{Msd6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000005 ])

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

Wgo = Parameter(name = 'Wgo',
                nature = 'external',
                type = 'real',
                value = 5.50675438,
                texname = '\\text{Wgo}',
                lhablock = 'DECAY',
                lhacode = [ 1000021 ])

WH01 = Parameter(name = 'WH01',
                 nature = 'external',
                 type = 'real',
                 value = 0.00198610799,
                 texname = '\\text{WH01}',
                 lhablock = 'DECAY',
                 lhacode = [ 25 ])

WH02 = Parameter(name = 'WH02',
                 nature = 'external',
                 type = 'real',
                 value = 0.574801389,
                 texname = '\\text{WH02}',
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

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.56194983,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

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

Wsu1 = Parameter(name = 'Wsu1',
                 nature = 'external',
                 type = 'real',
                 value = 5.47719539,
                 texname = '\\text{Wsu1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000002 ])

Wsu2 = Parameter(name = 'Wsu2',
                 nature = 'external',
                 type = 'real',
                 value = 5.47719539,
                 texname = '\\text{Wsu2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000004 ])

Wsu3 = Parameter(name = 'Wsu3',
                 nature = 'external',
                 type = 'real',
                 value = 2.02159578,
                 texname = '\\text{Wsu3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000006 ])

Wsu4 = Parameter(name = 'Wsu4',
                 nature = 'external',
                 type = 'real',
                 value = 1.15297292,
                 texname = '\\text{Wsu4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000002 ])

Wsu5 = Parameter(name = 'Wsu5',
                 nature = 'external',
                 type = 'real',
                 value = 1.15297292,
                 texname = '\\text{Wsu5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000004 ])

Wsu6 = Parameter(name = 'Wsu6',
                 nature = 'external',
                 type = 'real',
                 value = 7.37313275,
                 texname = '\\text{Wsu6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000006 ])

Wsd1 = Parameter(name = 'Wsd1',
                 nature = 'external',
                 type = 'real',
                 value = 5.31278772,
                 texname = '\\text{Wsd1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000001 ])

Wsd2 = Parameter(name = 'Wsd2',
                 nature = 'external',
                 type = 'real',
                 value = 5.31278772,
                 texname = '\\text{Wsd2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000003 ])

Wsd3 = Parameter(name = 'Wsd3',
                 nature = 'external',
                 type = 'real',
                 value = 3.73627601,
                 texname = '\\text{Wsd3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000005 ])

Wsd4 = Parameter(name = 'Wsd4',
                 nature = 'external',
                 type = 'real',
                 value = 0.285812308,
                 texname = '\\text{Wsd4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000001 ])

Wsd5 = Parameter(name = 'Wsd5',
                 nature = 'external',
                 type = 'real',
                 value = 0.285812308,
                 texname = '\\text{Wsd5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000003 ])

Wsd6 = Parameter(name = 'Wsd6',
                 nature = 'external',
                 type = 'real',
                 value = 0.801566294,
                 texname = '\\text{Wsd6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000005 ])

beta = Parameter(name = 'beta',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.atan(tb)',
                 texname = '\\beta')

CKM1x1 = Parameter(name = 'CKM1x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RCKM1x1',
                   texname = '\\text{CKM1x1}')

CKM2x2 = Parameter(name = 'CKM2x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RCKM2x2',
                   texname = '\\text{CKM2x2}')

CKM3x3 = Parameter(name = 'CKM3x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RCKM3x3',
                   texname = '\\text{CKM3x3}')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'MW/MZ',
               texname = 'c_w')

mD21x1 = Parameter(name = 'mD21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmD21x1',
                   texname = '\\text{mD21x1}')

mD22x2 = Parameter(name = 'mD22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmD22x2',
                   texname = '\\text{mD22x2}')

mD23x3 = Parameter(name = 'mD23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmD23x3',
                   texname = '\\text{mD23x3}')

mE21x1 = Parameter(name = 'mE21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmE21x1',
                   texname = '\\text{mE21x1}')

mE22x2 = Parameter(name = 'mE22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmE22x2',
                   texname = '\\text{mE22x2}')

mE23x3 = Parameter(name = 'mE23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmE23x3',
                   texname = '\\text{mE23x3}')

mL21x1 = Parameter(name = 'mL21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmL21x1',
                   texname = '\\text{mL21x1}')

mL22x2 = Parameter(name = 'mL22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmL22x2',
                   texname = '\\text{mL22x2}')

mL23x3 = Parameter(name = 'mL23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmL23x3',
                   texname = '\\text{mL23x3}')

mQ21x1 = Parameter(name = 'mQ21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmQ21x1',
                   texname = '\\text{mQ21x1}')

mQ22x2 = Parameter(name = 'mQ22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmQ22x2',
                   texname = '\\text{mQ22x2}')

mQ23x3 = Parameter(name = 'mQ23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmQ23x3',
                   texname = '\\text{mQ23x3}')

mU21x1 = Parameter(name = 'mU21x1',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmU21x1',
                   texname = '\\text{mU21x1}')

mU22x2 = Parameter(name = 'mU22x2',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmU22x2',
                   texname = '\\text{mU22x2}')

mU23x3 = Parameter(name = 'mU23x3',
                   nature = 'internal',
                   type = 'complex',
                   value = 'RmU23x3',
                   texname = '\\text{mU23x3}')

MUH = Parameter(name = 'MUH',
                nature = 'internal',
                type = 'complex',
                value = 'RMUH',
                texname = '\\mu')

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

NN1x1 = Parameter(name = 'NN1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x1',
                  texname = '\\text{NN1x1}')

NN1x2 = Parameter(name = 'NN1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x2',
                  texname = '\\text{NN1x2}')

NN1x3 = Parameter(name = 'NN1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x3',
                  texname = '\\text{NN1x3}')

NN1x4 = Parameter(name = 'NN1x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x4',
                  texname = '\\text{NN1x4}')

NN2x1 = Parameter(name = 'NN2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x1',
                  texname = '\\text{NN2x1}')

NN2x2 = Parameter(name = 'NN2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x2',
                  texname = '\\text{NN2x2}')

NN2x3 = Parameter(name = 'NN2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x3',
                  texname = '\\text{NN2x3}')

NN2x4 = Parameter(name = 'NN2x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x4',
                  texname = '\\text{NN2x4}')

NN3x1 = Parameter(name = 'NN3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x1',
                  texname = '\\text{NN3x1}')

NN3x2 = Parameter(name = 'NN3x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x2',
                  texname = '\\text{NN3x2}')

NN3x3 = Parameter(name = 'NN3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x3',
                  texname = '\\text{NN3x3}')

NN3x4 = Parameter(name = 'NN3x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x4',
                  texname = '\\text{NN3x4}')

NN4x1 = Parameter(name = 'NN4x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x1',
                  texname = '\\text{NN4x1}')

NN4x2 = Parameter(name = 'NN4x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x2',
                  texname = '\\text{NN4x2}')

NN4x3 = Parameter(name = 'NN4x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x3',
                  texname = '\\text{NN4x3}')

NN4x4 = Parameter(name = 'NN4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x4',
                  texname = '\\text{NN4x4}')

Rd1x1 = Parameter(name = 'Rd1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd1x1',
                  texname = '\\text{Rd1x1}')

Rd2x2 = Parameter(name = 'Rd2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd2x2',
                  texname = '\\text{Rd2x2}')

Rd3x3 = Parameter(name = 'Rd3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd3x3',
                  texname = '\\text{Rd3x3}')

Rd3x6 = Parameter(name = 'Rd3x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd3x6',
                  texname = '\\text{Rd3x6}')

Rd4x4 = Parameter(name = 'Rd4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd4x4',
                  texname = '\\text{Rd4x4}')

Rd5x5 = Parameter(name = 'Rd5x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd5x5',
                  texname = '\\text{Rd5x5}')

Rd6x3 = Parameter(name = 'Rd6x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd6x3',
                  texname = '\\text{Rd6x3}')

Rd6x6 = Parameter(name = 'Rd6x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd6x6',
                  texname = '\\text{Rd6x6}')

Rl1x1 = Parameter(name = 'Rl1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl1x1',
                  texname = '\\text{Rl1x1}')

Rl2x2 = Parameter(name = 'Rl2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl2x2',
                  texname = '\\text{Rl2x2}')

Rl3x3 = Parameter(name = 'Rl3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl3x3',
                  texname = '\\text{Rl3x3}')

Rl3x6 = Parameter(name = 'Rl3x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl3x6',
                  texname = '\\text{Rl3x6}')

Rl4x4 = Parameter(name = 'Rl4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl4x4',
                  texname = '\\text{Rl4x4}')

Rl5x5 = Parameter(name = 'Rl5x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl5x5',
                  texname = '\\text{Rl5x5}')

Rl6x3 = Parameter(name = 'Rl6x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl6x3',
                  texname = '\\text{Rl6x3}')

Rl6x6 = Parameter(name = 'Rl6x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl6x6',
                  texname = '\\text{Rl6x6}')

Rn1x1 = Parameter(name = 'Rn1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn1x1',
                  texname = '\\text{Rn1x1}')

Rn2x2 = Parameter(name = 'Rn2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn2x2',
                  texname = '\\text{Rn2x2}')

Rn3x3 = Parameter(name = 'Rn3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn3x3',
                  texname = '\\text{Rn3x3}')

Ru1x1 = Parameter(name = 'Ru1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu1x1',
                  texname = '\\text{Ru1x1}')

Ru2x2 = Parameter(name = 'Ru2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu2x2',
                  texname = '\\text{Ru2x2}')

Ru3x3 = Parameter(name = 'Ru3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu3x3',
                  texname = '\\text{Ru3x3}')

Ru3x6 = Parameter(name = 'Ru3x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu3x6',
                  texname = '\\text{Ru3x6}')

Ru4x4 = Parameter(name = 'Ru4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu4x4',
                  texname = '\\text{Ru4x4}')

Ru5x5 = Parameter(name = 'Ru5x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu5x5',
                  texname = '\\text{Ru5x5}')

Ru6x3 = Parameter(name = 'Ru6x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu6x3',
                  texname = '\\text{Ru6x3}')

Ru6x6 = Parameter(name = 'Ru6x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu6x6',
                  texname = '\\text{Ru6x6}')

UU1x1 = Parameter(name = 'UU1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU1x1',
                  texname = '\\text{UU1x1}')

UU1x2 = Parameter(name = 'UU1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU1x2',
                  texname = '\\text{UU1x2}')

UU2x1 = Parameter(name = 'UU2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU2x1',
                  texname = '\\text{UU2x1}')

UU2x2 = Parameter(name = 'UU2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RUU2x2',
                  texname = '\\text{UU2x2}')

VV1x1 = Parameter(name = 'VV1x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV1x1',
                  texname = '\\text{VV1x1}')

VV1x2 = Parameter(name = 'VV1x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV1x2',
                  texname = '\\text{VV1x2}')

VV2x1 = Parameter(name = 'VV2x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV2x1',
                  texname = '\\text{VV2x1}')

VV2x2 = Parameter(name = 'VV2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RVV2x2',
                  texname = '\\text{VV2x2}')

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

td3x3 = Parameter(name = 'td3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rtd3x3',
                  texname = '\\text{td3x3}')

te3x3 = Parameter(name = 'te3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rte3x3',
                  texname = '\\text{te3x3}')

tu3x3 = Parameter(name = 'tu3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rtu3x3',
                  texname = '\\text{tu3x3}')

yd3x3 = Parameter(name = 'yd3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ryd3x3',
                  texname = '\\text{yd3x3}')

ye3x3 = Parameter(name = 'ye3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rye3x3',
                  texname = '\\text{ye3x3}')

yu3x3 = Parameter(name = 'yu3x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ryu3x3',
                  texname = '\\text{yu3x3}')

bb = Parameter(name = 'bb',
               nature = 'internal',
               type = 'complex',
               value = '((-mHd2 + mHu2 - MZ**2*cmath.cos(2*beta))*cmath.tan(2*beta))/2.',
               texname = 'b')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - cw**2)',
               texname = 's_w')

gp = Parameter(name = 'gp',
               nature = 'internal',
               type = 'real',
               value = 'ee/cw',
               texname = 'g\'')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = 'ee/sw',
               texname = 'g_w')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(2*cw*MZ*sw)/ee',
                texname = 'v')

vd = Parameter(name = 'vd',
               nature = 'internal',
               type = 'real',
               value = 'vev*cmath.cos(beta)',
               texname = 'v_d')

vu = Parameter(name = 'vu',
               nature = 'internal',
               type = 'real',
               value = 'vev*cmath.sin(beta)',
               texname = 'v_u')

I1x33 = Parameter(name = 'I1x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(CKM3x3)*complexconjugate(yu3x3)',
                  texname = '\\text{I1x33}')

I10x33 = Parameter(name = 'I10x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(yd3x3)',
                   texname = '\\text{I10x33}')

I10x36 = Parameter(name = 'I10x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(yd3x3)',
                   texname = '\\text{I10x36}')

I100x33 = Parameter(name = 'I100x33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*complexconjugate(Rd3x6)',
                    texname = '\\text{I100x33}')

I100x36 = Parameter(name = 'I100x36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*complexconjugate(Rd3x6)',
                    texname = '\\text{I100x36}')

I100x44 = Parameter(name = 'I100x44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*complexconjugate(Rd4x4)',
                    texname = '\\text{I100x44}')

I100x55 = Parameter(name = 'I100x55',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd5x5*complexconjugate(Rd5x5)',
                    texname = '\\text{I100x55}')

I100x63 = Parameter(name = 'I100x63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x6*complexconjugate(Rd6x6)',
                    texname = '\\text{I100x63}')

I100x66 = Parameter(name = 'I100x66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd6x6*complexconjugate(Rd6x6)',
                    texname = '\\text{I100x66}')

I101x33 = Parameter(name = 'I101x33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*complexconjugate(Rl3x6)',
                    texname = '\\text{I101x33}')

I101x36 = Parameter(name = 'I101x36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*complexconjugate(Rl3x6)',
                    texname = '\\text{I101x36}')

I101x44 = Parameter(name = 'I101x44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x4*complexconjugate(Rl4x4)',
                    texname = '\\text{I101x44}')

I101x55 = Parameter(name = 'I101x55',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl5x5*complexconjugate(Rl5x5)',
                    texname = '\\text{I101x55}')

I101x63 = Parameter(name = 'I101x63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x6*complexconjugate(Rl6x6)',
                    texname = '\\text{I101x63}')

I101x66 = Parameter(name = 'I101x66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl6x6*complexconjugate(Rl6x6)',
                    texname = '\\text{I101x66}')

I102x33 = Parameter(name = 'I102x33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*complexconjugate(Ru3x6)',
                    texname = '\\text{I102x33}')

I102x36 = Parameter(name = 'I102x36',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*complexconjugate(Ru3x6)',
                    texname = '\\text{I102x36}')

I102x44 = Parameter(name = 'I102x44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru4x4*complexconjugate(Ru4x4)',
                    texname = '\\text{I102x44}')

I102x55 = Parameter(name = 'I102x55',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru5x5*complexconjugate(Ru5x5)',
                    texname = '\\text{I102x55}')

I102x63 = Parameter(name = 'I102x63',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x6*complexconjugate(Ru6x6)',
                    texname = '\\text{I102x63}')

I102x66 = Parameter(name = 'I102x66',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru6x6*complexconjugate(Ru6x6)',
                    texname = '\\text{I102x66}')

I11x33 = Parameter(name = 'I11x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3',
                   texname = '\\text{I11x33}')

I11x36 = Parameter(name = 'I11x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3',
                   texname = '\\text{I11x36}')

I12x11 = Parameter(name = 'I12x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I12x11}')

I12x22 = Parameter(name = 'I12x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I12x22}')

I12x33 = Parameter(name = 'I12x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I12x33}')

I12x36 = Parameter(name = 'I12x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I12x36}')

I12x63 = Parameter(name = 'I12x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I12x63}')

I12x66 = Parameter(name = 'I12x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I12x66}')

I13x33 = Parameter(name = 'I13x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*complexconjugate(Rd3x6)',
                   texname = '\\text{I13x33}')

I13x36 = Parameter(name = 'I13x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*complexconjugate(Rd3x6)',
                   texname = '\\text{I13x36}')

I13x44 = Parameter(name = 'I13x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd4x4*complexconjugate(Rd4x4)',
                   texname = '\\text{I13x44}')

I13x55 = Parameter(name = 'I13x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd5x5*complexconjugate(Rd5x5)',
                   texname = '\\text{I13x55}')

I13x63 = Parameter(name = 'I13x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*complexconjugate(Rd6x6)',
                   texname = '\\text{I13x63}')

I13x66 = Parameter(name = 'I13x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*complexconjugate(Rd6x6)',
                   texname = '\\text{I13x66}')

I14x33 = Parameter(name = 'I14x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                   texname = '\\text{I14x33}')

I14x36 = Parameter(name = 'I14x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                   texname = '\\text{I14x36}')

I14x63 = Parameter(name = 'I14x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                   texname = '\\text{I14x63}')

I14x66 = Parameter(name = 'I14x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                   texname = '\\text{I14x66}')

I15x33 = Parameter(name = 'I15x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I15x33}')

I15x36 = Parameter(name = 'I15x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I15x36}')

I15x63 = Parameter(name = 'I15x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I15x63}')

I15x66 = Parameter(name = 'I15x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I15x66}')

I16x33 = Parameter(name = 'I16x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I16x33}')

I16x36 = Parameter(name = 'I16x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I16x36}')

I16x63 = Parameter(name = 'I16x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I16x63}')

I16x66 = Parameter(name = 'I16x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I16x66}')

I17x33 = Parameter(name = 'I17x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I17x33}')

I17x36 = Parameter(name = 'I17x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I17x36}')

I17x63 = Parameter(name = 'I17x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I17x63}')

I17x66 = Parameter(name = 'I17x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I17x66}')

I18x33 = Parameter(name = 'I18x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I18x33}')

I18x36 = Parameter(name = 'I18x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I18x36}')

I18x63 = Parameter(name = 'I18x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I18x63}')

I18x66 = Parameter(name = 'I18x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I18x66}')

I19x33 = Parameter(name = 'I19x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I19x33}')

I19x36 = Parameter(name = 'I19x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I19x36}')

I19x63 = Parameter(name = 'I19x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I19x63}')

I19x66 = Parameter(name = 'I19x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I19x66}')

I2x33 = Parameter(name = 'I2x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3*complexconjugate(CKM3x3)',
                  texname = '\\text{I2x33}')

I20x33 = Parameter(name = 'I20x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*complexconjugate(yd3x3)',
                   texname = '\\text{I20x33}')

I21x33 = Parameter(name = 'I21x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*yu3x3',
                   texname = '\\text{I21x33}')

I22x33 = Parameter(name = 'I22x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(ye3x3)',
                   texname = '\\text{I22x33}')

I23x33 = Parameter(name = 'I23x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I23x33}')

I23x36 = Parameter(name = 'I23x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I23x36}')

I24x33 = Parameter(name = 'I24x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I24x33}')

I24x36 = Parameter(name = 'I24x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I24x36}')

I25x11 = Parameter(name = 'I25x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I25x11}')

I25x22 = Parameter(name = 'I25x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I25x22}')

I25x33 = Parameter(name = 'I25x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I25x33}')

I25x36 = Parameter(name = 'I25x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I25x36}')

I25x63 = Parameter(name = 'I25x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I25x63}')

I25x66 = Parameter(name = 'I25x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I25x66}')

I26x33 = Parameter(name = 'I26x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I26x33}')

I26x36 = Parameter(name = 'I26x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I26x36}')

I26x44 = Parameter(name = 'I26x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x4*complexconjugate(Rl4x4)',
                   texname = '\\text{I26x44}')

I26x55 = Parameter(name = 'I26x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x5*complexconjugate(Rl5x5)',
                   texname = '\\text{I26x55}')

I26x63 = Parameter(name = 'I26x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I26x63}')

I26x66 = Parameter(name = 'I26x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I26x66}')

I27x33 = Parameter(name = 'I27x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(ye3x3)',
                   texname = '\\text{I27x33}')

I27x36 = Parameter(name = 'I27x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(ye3x3)',
                   texname = '\\text{I27x36}')

I28x33 = Parameter(name = 'I28x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3',
                   texname = '\\text{I28x33}')

I28x36 = Parameter(name = 'I28x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3',
                   texname = '\\text{I28x36}')

I29x11 = Parameter(name = 'I29x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1',
                   texname = '\\text{I29x11}')

I29x22 = Parameter(name = 'I29x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2',
                   texname = '\\text{I29x22}')

I29x33 = Parameter(name = 'I29x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3',
                   texname = '\\text{I29x33}')

I29x36 = Parameter(name = 'I29x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3',
                   texname = '\\text{I29x36}')

I3x33 = Parameter(name = 'I3x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                  texname = '\\text{I3x33}')

I3x36 = Parameter(name = 'I3x36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                  texname = '\\text{I3x36}')

I30x33 = Parameter(name = 'I30x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3',
                   texname = '\\text{I30x33}')

I30x36 = Parameter(name = 'I30x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3',
                   texname = '\\text{I30x36}')

I31x11 = Parameter(name = 'I31x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I31x11}')

I31x22 = Parameter(name = 'I31x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I31x22}')

I31x33 = Parameter(name = 'I31x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I31x33}')

I31x36 = Parameter(name = 'I31x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I31x36}')

I31x63 = Parameter(name = 'I31x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I31x63}')

I31x66 = Parameter(name = 'I31x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I31x66}')

I32x33 = Parameter(name = 'I32x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I32x33}')

I32x36 = Parameter(name = 'I32x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl3x6)',
                   texname = '\\text{I32x36}')

I32x44 = Parameter(name = 'I32x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x4*complexconjugate(Rl4x4)',
                   texname = '\\text{I32x44}')

I32x55 = Parameter(name = 'I32x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x5*complexconjugate(Rl5x5)',
                   texname = '\\text{I32x55}')

I32x63 = Parameter(name = 'I32x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I32x63}')

I32x66 = Parameter(name = 'I32x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*complexconjugate(Rl6x6)',
                   texname = '\\text{I32x66}')

I33x33 = Parameter(name = 'I33x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x6)*complexconjugate(te3x3)',
                   texname = '\\text{I33x33}')

I33x36 = Parameter(name = 'I33x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x6)*complexconjugate(te3x3)',
                   texname = '\\text{I33x36}')

I33x63 = Parameter(name = 'I33x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)',
                   texname = '\\text{I33x63}')

I33x66 = Parameter(name = 'I33x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)',
                   texname = '\\text{I33x66}')

I34x33 = Parameter(name = 'I34x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I34x33}')

I34x36 = Parameter(name = 'I34x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I34x36}')

I34x63 = Parameter(name = 'I34x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I34x63}')

I34x66 = Parameter(name = 'I34x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I34x66}')

I35x33 = Parameter(name = 'I35x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*te3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I35x33}')

I35x36 = Parameter(name = 'I35x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*te3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I35x36}')

I35x63 = Parameter(name = 'I35x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*te3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I35x63}')

I35x66 = Parameter(name = 'I35x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*te3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I35x66}')

I36x33 = Parameter(name = 'I36x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I36x33}')

I36x36 = Parameter(name = 'I36x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I36x36}')

I36x63 = Parameter(name = 'I36x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*ye3x3*complexconjugate(Rl6x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I36x63}')

I36x66 = Parameter(name = 'I36x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*ye3x3*complexconjugate(Rl6x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I36x66}')

I37x33 = Parameter(name = 'I37x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I37x33}')

I37x36 = Parameter(name = 'I37x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I37x36}')

I37x63 = Parameter(name = 'I37x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I37x63}')

I37x66 = Parameter(name = 'I37x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I37x66}')

I38x33 = Parameter(name = 'I38x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I38x33}')

I38x36 = Parameter(name = 'I38x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I38x36}')

I38x63 = Parameter(name = 'I38x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I38x63}')

I38x66 = Parameter(name = 'I38x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I38x66}')

I39x11 = Parameter(name = 'I39x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rn1x1)',
                   texname = '\\text{I39x11}')

I39x22 = Parameter(name = 'I39x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rn2x2)',
                   texname = '\\text{I39x22}')

I39x33 = Parameter(name = 'I39x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I39x33}')

I39x36 = Parameter(name = 'I39x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I39x36}')

I4x33 = Parameter(name = 'I4x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3*complexconjugate(Rd3x3)',
                  texname = '\\text{I4x33}')

I4x36 = Parameter(name = 'I4x36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3*complexconjugate(Rd6x3)',
                  texname = '\\text{I4x36}')

I40x33 = Parameter(name = 'I40x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*te3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I40x33}')

I40x36 = Parameter(name = 'I40x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*te3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I40x36}')

I41x33 = Parameter(name = 'I41x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I41x33}')

I41x36 = Parameter(name = 'I41x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*ye3x3*complexconjugate(Rn3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I41x36}')

I42x33 = Parameter(name = 'I42x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x6*ye3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I42x33}')

I42x36 = Parameter(name = 'I42x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x6*ye3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I42x36}')

I43x11 = Parameter(name = 'I43x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x1',
                   texname = '\\text{I43x11}')

I43x22 = Parameter(name = 'I43x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2',
                   texname = '\\text{I43x22}')

I43x33 = Parameter(name = 'I43x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3',
                   texname = '\\text{I43x33}')

I44x33 = Parameter(name = 'I44x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(ye3x3)',
                   texname = '\\text{I44x33}')

I45x11 = Parameter(name = 'I45x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I45x11}')

I45x22 = Parameter(name = 'I45x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I45x22}')

I45x33 = Parameter(name = 'I45x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I45x33}')

I45x36 = Parameter(name = 'I45x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I45x36}')

I46x33 = Parameter(name = 'I46x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I46x33}')

I46x36 = Parameter(name = 'I46x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I46x36}')

I47x33 = Parameter(name = 'I47x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl3x6)*complexconjugate(te3x3)',
                   texname = '\\text{I47x33}')

I47x36 = Parameter(name = 'I47x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl6x6)*complexconjugate(te3x3)',
                   texname = '\\text{I47x36}')

I48x33 = Parameter(name = 'I48x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*ye3x3*complexconjugate(Rl3x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I48x33}')

I48x36 = Parameter(name = 'I48x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*ye3x3*complexconjugate(Rl6x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I48x36}')

I49x33 = Parameter(name = 'I49x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I49x33}')

I49x36 = Parameter(name = 'I49x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I49x36}')

I5x11 = Parameter(name = 'I5x11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x1*complexconjugate(Rd1x1)',
                  texname = '\\text{I5x11}')

I5x22 = Parameter(name = 'I5x22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x2*complexconjugate(Rd2x2)',
                  texname = '\\text{I5x22}')

I5x33 = Parameter(name = 'I5x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x3*complexconjugate(Rd3x3)',
                  texname = '\\text{I5x33}')

I5x36 = Parameter(name = 'I5x36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x3*complexconjugate(Rd3x3)',
                  texname = '\\text{I5x36}')

I5x63 = Parameter(name = 'I5x63',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x3*complexconjugate(Rd6x3)',
                  texname = '\\text{I5x63}')

I5x66 = Parameter(name = 'I5x66',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x3*complexconjugate(Rd6x3)',
                  texname = '\\text{I5x66}')

I50x33 = Parameter(name = 'I50x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I50x33}')

I50x36 = Parameter(name = 'I50x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I50x36}')

I51x11 = Parameter(name = 'I51x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x1*complexconjugate(Ru1x1)',
                   texname = '\\text{I51x11}')

I51x22 = Parameter(name = 'I51x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x2*complexconjugate(Ru2x2)',
                   texname = '\\text{I51x22}')

I51x33 = Parameter(name = 'I51x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I51x33}')

I51x36 = Parameter(name = 'I51x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I51x36}')

I51x63 = Parameter(name = 'I51x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I51x63}')

I51x66 = Parameter(name = 'I51x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I51x66}')

I52x33 = Parameter(name = 'I52x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*complexconjugate(Ru3x6)',
                   texname = '\\text{I52x33}')

I52x36 = Parameter(name = 'I52x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*complexconjugate(Ru3x6)',
                   texname = '\\text{I52x36}')

I52x44 = Parameter(name = 'I52x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru4x4*complexconjugate(Ru4x4)',
                   texname = '\\text{I52x44}')

I52x55 = Parameter(name = 'I52x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x5*complexconjugate(Ru5x5)',
                   texname = '\\text{I52x55}')

I52x63 = Parameter(name = 'I52x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*complexconjugate(Ru6x6)',
                   texname = '\\text{I52x63}')

I52x66 = Parameter(name = 'I52x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*complexconjugate(Ru6x6)',
                   texname = '\\text{I52x66}')

I53x11 = Parameter(name = 'I53x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)',
                   texname = '\\text{I53x11}')

I53x22 = Parameter(name = 'I53x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)',
                   texname = '\\text{I53x22}')

I53x33 = Parameter(name = 'I53x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I53x33}')

I53x36 = Parameter(name = 'I53x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I53x36}')

I53x63 = Parameter(name = 'I53x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I53x63}')

I53x66 = Parameter(name = 'I53x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I53x66}')

I54x33 = Parameter(name = 'I54x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I54x33}')

I54x36 = Parameter(name = 'I54x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I54x36}')

I54x63 = Parameter(name = 'I54x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I54x63}')

I54x66 = Parameter(name = 'I54x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I54x66}')

I55x33 = Parameter(name = 'I55x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I55x33}')

I55x36 = Parameter(name = 'I55x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I55x36}')

I55x63 = Parameter(name = 'I55x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I55x63}')

I55x66 = Parameter(name = 'I55x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I55x66}')

I56x33 = Parameter(name = 'I56x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I56x33}')

I56x36 = Parameter(name = 'I56x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*td3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I56x36}')

I56x63 = Parameter(name = 'I56x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I56x63}')

I56x66 = Parameter(name = 'I56x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*td3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I56x66}')

I57x33 = Parameter(name = 'I57x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I57x33}')

I57x36 = Parameter(name = 'I57x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I57x36}')

I57x63 = Parameter(name = 'I57x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I57x63}')

I57x66 = Parameter(name = 'I57x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I57x66}')

I58x33 = Parameter(name = 'I58x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x33}')

I58x36 = Parameter(name = 'I58x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x36}')

I58x63 = Parameter(name = 'I58x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x63}')

I58x66 = Parameter(name = 'I58x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x66}')

I59x33 = Parameter(name = 'I59x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x33}')

I59x36 = Parameter(name = 'I59x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x36}')

I59x63 = Parameter(name = 'I59x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x63}')

I59x66 = Parameter(name = 'I59x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x6*yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x66}')

I6x33 = Parameter(name = 'I6x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x6*complexconjugate(Rd3x6)',
                  texname = '\\text{I6x33}')

I6x36 = Parameter(name = 'I6x36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x6*complexconjugate(Rd3x6)',
                  texname = '\\text{I6x36}')

I6x44 = Parameter(name = 'I6x44',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd4x4*complexconjugate(Rd4x4)',
                  texname = '\\text{I6x44}')

I6x55 = Parameter(name = 'I6x55',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd5x5*complexconjugate(Rd5x5)',
                  texname = '\\text{I6x55}')

I6x63 = Parameter(name = 'I6x63',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x6*complexconjugate(Rd6x6)',
                  texname = '\\text{I6x63}')

I6x66 = Parameter(name = 'I6x66',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x6*complexconjugate(Rd6x6)',
                  texname = '\\text{I6x66}')

I60x33 = Parameter(name = 'I60x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I60x33}')

I60x36 = Parameter(name = 'I60x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I60x36}')

I60x63 = Parameter(name = 'I60x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I60x63}')

I60x66 = Parameter(name = 'I60x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*yu3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I60x66}')

I61x33 = Parameter(name = 'I61x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(yu3x3)',
                   texname = '\\text{I61x33}')

I61x36 = Parameter(name = 'I61x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(yu3x3)',
                   texname = '\\text{I61x36}')

I62x33 = Parameter(name = 'I62x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*yu3x3',
                   texname = '\\text{I62x33}')

I62x36 = Parameter(name = 'I62x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*yu3x3',
                   texname = '\\text{I62x36}')

I63x11 = Parameter(name = 'I63x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM1x1*Ru1x1',
                   texname = '\\text{I63x11}')

I63x22 = Parameter(name = 'I63x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM2x2*Ru2x2',
                   texname = '\\text{I63x22}')

I63x33 = Parameter(name = 'I63x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3',
                   texname = '\\text{I63x33}')

I63x36 = Parameter(name = 'I63x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3',
                   texname = '\\text{I63x36}')

I64x33 = Parameter(name = 'I64x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(yd3x3)',
                   texname = '\\text{I64x33}')

I64x36 = Parameter(name = 'I64x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(yd3x3)',
                   texname = '\\text{I64x36}')

I65x33 = Parameter(name = 'I65x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x6*yu3x3',
                   texname = '\\text{I65x33}')

I65x36 = Parameter(name = 'I65x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x6*yu3x3',
                   texname = '\\text{I65x36}')

I66x11 = Parameter(name = 'I66x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM1x1*Ru1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I66x11}')

I66x22 = Parameter(name = 'I66x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM2x2*Ru2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I66x22}')

I66x33 = Parameter(name = 'I66x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I66x33}')

I66x36 = Parameter(name = 'I66x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I66x36}')

I66x63 = Parameter(name = 'I66x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I66x63}')

I66x66 = Parameter(name = 'I66x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I66x66}')

I67x33 = Parameter(name = 'I67x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I67x33}')

I67x36 = Parameter(name = 'I67x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I67x36}')

I67x63 = Parameter(name = 'I67x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I67x63}')

I67x66 = Parameter(name = 'I67x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I67x66}')

I68x33 = Parameter(name = 'I68x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                   texname = '\\text{I68x33}')

I68x36 = Parameter(name = 'I68x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd3x6)*complexconjugate(td3x3)',
                   texname = '\\text{I68x36}')

I68x63 = Parameter(name = 'I68x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                   texname = '\\text{I68x63}')

I68x66 = Parameter(name = 'I68x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd6x6)*complexconjugate(td3x3)',
                   texname = '\\text{I68x66}')

I69x33 = Parameter(name = 'I69x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x6*tu3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I69x33}')

I69x36 = Parameter(name = 'I69x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x6*tu3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I69x36}')

I69x63 = Parameter(name = 'I69x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x6*tu3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I69x63}')

I69x66 = Parameter(name = 'I69x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x6*tu3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I69x66}')

I7x11 = Parameter(name = 'I7x11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x1*complexconjugate(CKM1x1)',
                  texname = '\\text{I7x11}')

I7x22 = Parameter(name = 'I7x22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x2*complexconjugate(CKM2x2)',
                  texname = '\\text{I7x22}')

I7x33 = Parameter(name = 'I7x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x3*complexconjugate(CKM3x3)',
                  texname = '\\text{I7x33}')

I7x36 = Parameter(name = 'I7x36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x3*complexconjugate(CKM3x3)',
                  texname = '\\text{I7x36}')

I70x33 = Parameter(name = 'I70x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I70x33}')

I70x36 = Parameter(name = 'I70x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*yd3x3*complexconjugate(Rd3x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I70x36}')

I70x63 = Parameter(name = 'I70x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I70x63}')

I70x66 = Parameter(name = 'I70x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*yd3x3*complexconjugate(Rd6x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I70x66}')

I71x33 = Parameter(name = 'I71x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I71x33}')

I71x36 = Parameter(name = 'I71x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*yu3x3*complexconjugate(Rd3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I71x36}')

I71x63 = Parameter(name = 'I71x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*yu3x3*complexconjugate(Rd6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I71x63}')

I71x66 = Parameter(name = 'I71x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*yu3x3*complexconjugate(Rd6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I71x66}')

I72x33 = Parameter(name = 'I72x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x6*yu3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I72x33}')

I72x36 = Parameter(name = 'I72x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x6*yu3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I72x36}')

I72x63 = Parameter(name = 'I72x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x6*yu3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I72x63}')

I72x66 = Parameter(name = 'I72x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x6*yu3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I72x66}')

I73x33 = Parameter(name = 'I73x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x6*yu3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I73x33}')

I73x36 = Parameter(name = 'I73x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x6*yu3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I73x36}')

I73x63 = Parameter(name = 'I73x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I73x63}')

I73x66 = Parameter(name = 'I73x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x6*yu3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I73x66}')

I74x11 = Parameter(name = 'I74x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x1*complexconjugate(Ru1x1)',
                   texname = '\\text{I74x11}')

I74x22 = Parameter(name = 'I74x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x2*complexconjugate(Ru2x2)',
                   texname = '\\text{I74x22}')

I74x33 = Parameter(name = 'I74x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I74x33}')

I74x36 = Parameter(name = 'I74x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I74x36}')

I74x63 = Parameter(name = 'I74x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I74x63}')

I74x66 = Parameter(name = 'I74x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I74x66}')

I75x33 = Parameter(name = 'I75x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*complexconjugate(Ru3x6)',
                   texname = '\\text{I75x33}')

I75x36 = Parameter(name = 'I75x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*complexconjugate(Ru3x6)',
                   texname = '\\text{I75x36}')

I75x44 = Parameter(name = 'I75x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru4x4*complexconjugate(Ru4x4)',
                   texname = '\\text{I75x44}')

I75x55 = Parameter(name = 'I75x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x5*complexconjugate(Ru5x5)',
                   texname = '\\text{I75x55}')

I75x63 = Parameter(name = 'I75x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*complexconjugate(Ru6x6)',
                   texname = '\\text{I75x63}')

I75x66 = Parameter(name = 'I75x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*complexconjugate(Ru6x6)',
                   texname = '\\text{I75x66}')

I76x33 = Parameter(name = 'I76x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I76x33}')

I76x36 = Parameter(name = 'I76x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I76x36}')

I76x63 = Parameter(name = 'I76x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I76x63}')

I76x66 = Parameter(name = 'I76x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I76x66}')

I77x33 = Parameter(name = 'I77x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I77x33}')

I77x36 = Parameter(name = 'I77x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru3x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I77x36}')

I77x63 = Parameter(name = 'I77x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I77x63}')

I77x66 = Parameter(name = 'I77x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru6x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I77x66}')

I78x33 = Parameter(name = 'I78x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*tu3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I78x33}')

I78x36 = Parameter(name = 'I78x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*tu3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I78x36}')

I78x63 = Parameter(name = 'I78x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*tu3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I78x63}')

I78x66 = Parameter(name = 'I78x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*tu3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I78x66}')

I79x33 = Parameter(name = 'I79x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*yu3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I79x33}')

I79x36 = Parameter(name = 'I79x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*yu3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I79x36}')

I79x63 = Parameter(name = 'I79x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*yu3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I79x63}')

I79x66 = Parameter(name = 'I79x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*yu3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I79x66}')

I8x33 = Parameter(name = 'I8x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(yu3x3)',
                  texname = '\\text{I8x33}')

I8x36 = Parameter(name = 'I8x36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(yu3x3)',
                  texname = '\\text{I8x36}')

I80x33 = Parameter(name = 'I80x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x33}')

I80x36 = Parameter(name = 'I80x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*yu3x3*complexconjugate(Ru3x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x36}')

I80x63 = Parameter(name = 'I80x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*yu3x3*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x63}')

I80x66 = Parameter(name = 'I80x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*yu3x3*complexconjugate(Ru6x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x66}')

I81x33 = Parameter(name = 'I81x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*yu3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x33}')

I81x36 = Parameter(name = 'I81x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*yu3x3*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x36}')

I81x63 = Parameter(name = 'I81x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x63}')

I81x66 = Parameter(name = 'I81x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x6*yu3x3*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x66}')

I82x11 = Parameter(name = 'I82x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I82x11}')

I82x22 = Parameter(name = 'I82x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I82x22}')

I82x33 = Parameter(name = 'I82x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I82x33}')

I82x36 = Parameter(name = 'I82x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I82x36}')

I83x33 = Parameter(name = 'I83x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*complexconjugate(Rd3x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I83x33}')

I83x36 = Parameter(name = 'I83x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*complexconjugate(Rd6x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I83x36}')

I84x33 = Parameter(name = 'I84x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*yu3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I84x33}')

I84x36 = Parameter(name = 'I84x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*yu3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I84x36}')

I85x11 = Parameter(name = 'I85x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl1x1)',
                   texname = '\\text{I85x11}')

I85x22 = Parameter(name = 'I85x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl2x2)',
                   texname = '\\text{I85x22}')

I85x33 = Parameter(name = 'I85x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl3x3)',
                   texname = '\\text{I85x33}')

I85x36 = Parameter(name = 'I85x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl6x3)',
                   texname = '\\text{I85x36}')

I86x33 = Parameter(name = 'I86x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl3x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I86x33}')

I86x36 = Parameter(name = 'I86x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl6x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I86x36}')

I87x11 = Parameter(name = 'I87x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rn1x1)',
                   texname = '\\text{I87x11}')

I87x22 = Parameter(name = 'I87x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rn2x2)',
                   texname = '\\text{I87x22}')

I87x33 = Parameter(name = 'I87x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rn3x3)',
                   texname = '\\text{I87x33}')

I88x33 = Parameter(name = 'I88x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I88x33}')

I89x11 = Parameter(name = 'I89x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(CKM1x1)*complexconjugate(Ru1x1)',
                   texname = '\\text{I89x11}')

I89x22 = Parameter(name = 'I89x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(CKM2x2)*complexconjugate(Ru2x2)',
                   texname = '\\text{I89x22}')

I89x33 = Parameter(name = 'I89x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I89x33}')

I89x36 = Parameter(name = 'I89x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I89x36}')

I9x33 = Parameter(name = 'I9x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x6*yd3x3*complexconjugate(CKM3x3)',
                  texname = '\\text{I9x33}')

I9x36 = Parameter(name = 'I9x36',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x6*yd3x3*complexconjugate(CKM3x3)',
                  texname = '\\text{I9x36}')

I90x33 = Parameter(name = 'I90x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(CKM3x3)*complexconjugate(Ru3x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I90x33}')

I90x36 = Parameter(name = 'I90x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(CKM3x3)*complexconjugate(Ru6x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I90x36}')

I91x33 = Parameter(name = 'I91x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I91x33}')

I91x36 = Parameter(name = 'I91x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I91x36}')

I92x11 = Parameter(name = 'I92x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM1x1*Ru1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I92x11}')

I92x22 = Parameter(name = 'I92x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM2x2*Ru2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I92x22}')

I92x33 = Parameter(name = 'I92x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I92x33}')

I92x36 = Parameter(name = 'I92x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I92x36}')

I92x63 = Parameter(name = 'I92x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I92x63}')

I92x66 = Parameter(name = 'I92x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'CKM3x3*Ru6x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I92x66}')

I93x11 = Parameter(name = 'I93x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I93x11}')

I93x22 = Parameter(name = 'I93x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I93x22}')

I93x33 = Parameter(name = 'I93x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I93x33}')

I93x36 = Parameter(name = 'I93x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I93x36}')

I94x11 = Parameter(name = 'I94x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x1*complexconjugate(CKM1x1)*complexconjugate(Ru1x1)',
                   texname = '\\text{I94x11}')

I94x22 = Parameter(name = 'I94x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x2*complexconjugate(CKM2x2)*complexconjugate(Ru2x2)',
                   texname = '\\text{I94x22}')

I94x33 = Parameter(name = 'I94x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I94x33}')

I94x36 = Parameter(name = 'I94x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I94x36}')

I94x63 = Parameter(name = 'I94x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru3x3)',
                   texname = '\\text{I94x63}')

I94x66 = Parameter(name = 'I94x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(CKM3x3)*complexconjugate(Ru6x3)',
                   texname = '\\text{I94x66}')

I95x11 = Parameter(name = 'I95x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rn1x1)',
                   texname = '\\text{I95x11}')

I95x22 = Parameter(name = 'I95x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rn2x2)',
                   texname = '\\text{I95x22}')

I95x33 = Parameter(name = 'I95x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I95x33}')

I95x36 = Parameter(name = 'I95x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rn3x3)',
                   texname = '\\text{I95x36}')

I96x11 = Parameter(name = 'I96x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x1*complexconjugate(Rd1x1)',
                   texname = '\\text{I96x11}')

I96x22 = Parameter(name = 'I96x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x2*complexconjugate(Rd2x2)',
                   texname = '\\text{I96x22}')

I96x33 = Parameter(name = 'I96x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I96x33}')

I96x36 = Parameter(name = 'I96x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd3x3)',
                   texname = '\\text{I96x36}')

I96x63 = Parameter(name = 'I96x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I96x63}')

I96x66 = Parameter(name = 'I96x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x3*complexconjugate(Rd6x3)',
                   texname = '\\text{I96x66}')

I97x11 = Parameter(name = 'I97x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x1*complexconjugate(Rl1x1)',
                   texname = '\\text{I97x11}')

I97x22 = Parameter(name = 'I97x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x2*complexconjugate(Rl2x2)',
                   texname = '\\text{I97x22}')

I97x33 = Parameter(name = 'I97x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I97x33}')

I97x36 = Parameter(name = 'I97x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl3x3)',
                   texname = '\\text{I97x36}')

I97x63 = Parameter(name = 'I97x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I97x63}')

I97x66 = Parameter(name = 'I97x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x3*complexconjugate(Rl6x3)',
                   texname = '\\text{I97x66}')

I98x11 = Parameter(name = 'I98x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x1*complexconjugate(Ru1x1)',
                   texname = '\\text{I98x11}')

I98x22 = Parameter(name = 'I98x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x2*complexconjugate(Ru2x2)',
                   texname = '\\text{I98x22}')

I98x33 = Parameter(name = 'I98x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I98x33}')

I98x36 = Parameter(name = 'I98x36',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru3x3)',
                   texname = '\\text{I98x36}')

I98x63 = Parameter(name = 'I98x63',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I98x63}')

I98x66 = Parameter(name = 'I98x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x3*complexconjugate(Ru6x3)',
                   texname = '\\text{I98x66}')

I99x33 = Parameter(name = 'I99x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3',
                   texname = '\\text{I99x33}')

