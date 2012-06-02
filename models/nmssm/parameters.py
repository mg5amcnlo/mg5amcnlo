# This file was automatically created by FeynRules 1.7.20
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Mon 28 May 2012 11:00:02



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
RRd1x3 = Parameter(name = 'RRd1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.990890455,
                   texname = '\\text{RRd1x3}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 1, 3 ])

RRd1x6 = Parameter(name = 'RRd1x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.134670361,
                   texname = '\\text{RRd1x6}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 1, 6 ])

RRd2x3 = Parameter(name = 'RRd2x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.134670361,
                   texname = '\\text{RRd2x3}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 2, 3 ])

RRd2x6 = Parameter(name = 'RRd2x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.990890455,
                   texname = '\\text{RRd2x6}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 2, 6 ])

RRd3x5 = Parameter(name = 'RRd3x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd3x5}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 3, 5 ])

RRd4x4 = Parameter(name = 'RRd4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd4x4}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 4, 4 ])

RRd5x1 = Parameter(name = 'RRd5x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd5x1}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 5, 1 ])

RRd6x2 = Parameter(name = 'RRd6x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRd6x2}',
                   lhablock = 'DSQMIX',
                   lhacode = [ 6, 2 ])

tb = Parameter(name = 'tb',
               nature = 'external',
               type = 'real',
               value = 10.0004319,
               texname = 't_b',
               lhablock = 'HMIX',
               lhacode = [ 2 ])

MA2 = Parameter(name = 'MA2',
                nature = 'external',
                type = 'real',
                value = 1.04827778e6,
                texname = 'm_A^2',
                lhablock = 'HMIX',
                lhacode = [ 4 ])

RmD21x1 = Parameter(name = 'RmD21x1',
                    nature = 'external',
                    type = 'real',
                    value = 963545.439,
                    texname = '\\text{RmD21x1}',
                    lhablock = 'MSD2',
                    lhacode = [ 1, 1 ])

RmD22x2 = Parameter(name = 'RmD22x2',
                    nature = 'external',
                    type = 'real',
                    value = 963545.439,
                    texname = '\\text{RmD22x2}',
                    lhablock = 'MSD2',
                    lhacode = [ 2, 2 ])

RmD23x3 = Parameter(name = 'RmD23x3',
                    nature = 'external',
                    type = 'real',
                    value = 933451.834,
                    texname = '\\text{RmD23x3}',
                    lhablock = 'MSD2',
                    lhacode = [ 3, 3 ])

RmE21x1 = Parameter(name = 'RmE21x1',
                    nature = 'external',
                    type = 'real',
                    value = 66311.1508,
                    texname = '\\text{RmE21x1}',
                    lhablock = 'MSE2',
                    lhacode = [ 1, 1 ])

RmE22x2 = Parameter(name = 'RmE22x2',
                    nature = 'external',
                    type = 'real',
                    value = 66311.1508,
                    texname = '\\text{RmE22x2}',
                    lhablock = 'MSE2',
                    lhacode = [ 2, 2 ])

RmE23x3 = Parameter(name = 'RmE23x3',
                    nature = 'external',
                    type = 'real',
                    value = 48897.3735,
                    texname = '\\text{RmE23x3}',
                    lhablock = 'MSE2',
                    lhacode = [ 3, 3 ])

RmL21x1 = Parameter(name = 'RmL21x1',
                    nature = 'external',
                    type = 'real',
                    value = 142415.235,
                    texname = '\\text{RmL21x1}',
                    lhablock = 'MSL2',
                    lhacode = [ 1, 1 ])

RmL22x2 = Parameter(name = 'RmL22x2',
                    nature = 'external',
                    type = 'real',
                    value = 142415.235,
                    texname = '\\text{RmL22x2}',
                    lhablock = 'MSL2',
                    lhacode = [ 2, 2 ])

RmL23x3 = Parameter(name = 'RmL23x3',
                    nature = 'external',
                    type = 'real',
                    value = 133786.42,
                    texname = '\\text{RmL23x3}',
                    lhablock = 'MSL2',
                    lhacode = [ 3, 3 ])

RMx1 = Parameter(name = 'RMx1',
                 nature = 'external',
                 type = 'real',
                 value = 211.618677,
                 texname = '\\text{RMx1}',
                 lhablock = 'MSOFT',
                 lhacode = [ 1 ])

RMx2 = Parameter(name = 'RMx2',
                 nature = 'external',
                 type = 'real',
                 value = 391.864817,
                 texname = '\\text{RMx2}',
                 lhablock = 'MSOFT',
                 lhacode = [ 2 ])

RMx3 = Parameter(name = 'RMx3',
                 nature = 'external',
                 type = 'real',
                 value = 1112.25552,
                 texname = '\\text{RMx3}',
                 lhablock = 'MSOFT',
                 lhacode = [ 3 ])

mHd2 = Parameter(name = 'mHd2',
                 nature = 'external',
                 type = 'real',
                 value = 89988.5262,
                 texname = 'm_{H_d}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 21 ])

mHu2 = Parameter(name = 'mHu2',
                 nature = 'external',
                 type = 'real',
                 value = -908071.077,
                 texname = 'm_{H_u}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 22 ])

RmQ21x1 = Parameter(name = 'RmQ21x1',
                    nature = 'external',
                    type = 'real',
                    value = 1.04878444e6,
                    texname = '\\text{RmQ21x1}',
                    lhablock = 'MSQ2',
                    lhacode = [ 1, 1 ])

RmQ22x2 = Parameter(name = 'RmQ22x2',
                    nature = 'external',
                    type = 'real',
                    value = 1.04878444e6,
                    texname = '\\text{RmQ22x2}',
                    lhablock = 'MSQ2',
                    lhacode = [ 2, 2 ])

RmQ23x3 = Parameter(name = 'RmQ23x3',
                    nature = 'external',
                    type = 'real',
                    value = 715579.339,
                    texname = '\\text{RmQ23x3}',
                    lhablock = 'MSQ2',
                    lhacode = [ 3, 3 ])

RmU21x1 = Parameter(name = 'RmU21x1',
                    nature = 'external',
                    type = 'real',
                    value = 972428.308,
                    texname = '\\text{RmU21x1}',
                    lhablock = 'MSU2',
                    lhacode = [ 1, 1 ])

RmU22x2 = Parameter(name = 'RmU22x2',
                    nature = 'external',
                    type = 'real',
                    value = 972428.308,
                    texname = '\\text{RmU22x2}',
                    lhablock = 'MSU2',
                    lhacode = [ 2, 2 ])

RmU23x3 = Parameter(name = 'RmU23x3',
                    nature = 'external',
                    type = 'real',
                    value = 319484.921,
                    texname = '\\text{RmU23x3}',
                    lhablock = 'MSU2',
                    lhacode = [ 3, 3 ])

UP1x1 = Parameter(name = 'UP1x1',
                  nature = 'external',
                  type = 'real',
                  value = 0.0501258919,
                  texname = '\\text{UP1x1}',
                  lhablock = 'NMAMIX',
                  lhacode = [ 1, 1 ])

UP1x2 = Parameter(name = 'UP1x2',
                  nature = 'external',
                  type = 'real',
                  value = 0.00501258919,
                  texname = '\\text{UP1x2}',
                  lhablock = 'NMAMIX',
                  lhacode = [ 1, 2 ])

UP1x3 = Parameter(name = 'UP1x3',
                  nature = 'external',
                  type = 'real',
                  value = 0.998730328,
                  texname = '\\text{UP1x3}',
                  lhablock = 'NMAMIX',
                  lhacode = [ 1, 3 ])

UP2x1 = Parameter(name = 'UP2x1',
                  nature = 'external',
                  type = 'real',
                  value = 0.99377382,
                  texname = '\\text{UP2x1}',
                  lhablock = 'NMAMIX',
                  lhacode = [ 2, 1 ])

UP2x2 = Parameter(name = 'UP2x2',
                  nature = 'external',
                  type = 'real',
                  value = 0.099377382,
                  texname = '\\text{UP2x2}',
                  lhablock = 'NMAMIX',
                  lhacode = [ 2, 2 ])

UP2x3 = Parameter(name = 'UP2x3',
                  nature = 'external',
                  type = 'real',
                  value = -0.0503758979,
                  texname = '\\text{UP2x3}',
                  lhablock = 'NMAMIX',
                  lhacode = [ 2, 3 ])

US1x1 = Parameter(name = 'US1x1',
                  nature = 'external',
                  type = 'real',
                  value = 0.101230631,
                  texname = '\\text{US1x1}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 1, 1 ])

US1x2 = Parameter(name = 'US1x2',
                  nature = 'external',
                  type = 'real',
                  value = 0.994841811,
                  texname = '\\text{US1x2}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 1, 2 ])

US1x3 = Parameter(name = 'US1x3',
                  nature = 'external',
                  type = 'real',
                  value = -0.00649079704,
                  texname = '\\text{US1x3}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 1, 3 ])

US2x1 = Parameter(name = 'US2x1',
                  nature = 'external',
                  type = 'real',
                  value = 0.994850372,
                  texname = '\\text{US2x1}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 2, 1 ])

US2x2 = Parameter(name = 'US2x2',
                  nature = 'external',
                  type = 'real',
                  value = -0.10119434,
                  texname = '\\text{US2x2}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 2, 2 ])

US2x3 = Parameter(name = 'US2x3',
                  nature = 'external',
                  type = 'real',
                  value = 0.00569588834,
                  texname = '\\text{US2x3}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 2, 3 ])

US3x1 = Parameter(name = 'US3x1',
                  nature = 'external',
                  type = 'real',
                  value = -0.00500967595,
                  texname = '\\text{US3x1}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 3, 1 ])

US3x2 = Parameter(name = 'US3x2',
                  nature = 'external',
                  type = 'real',
                  value = 0.00703397022,
                  texname = '\\text{US3x2}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 3, 2 ])

US3x3 = Parameter(name = 'US3x3',
                  nature = 'external',
                  type = 'real',
                  value = 0.999962713,
                  texname = '\\text{US3x3}',
                  lhablock = 'NMHMIX',
                  lhacode = [ 3, 3 ])

RNN1x1 = Parameter(name = 'RNN1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.998684518,
                   texname = '\\text{RNN1x1}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 1, 1 ])

RNN1x2 = Parameter(name = 'RNN1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.00814943871,
                   texname = '\\text{RNN1x2}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 1, 2 ])

RNN1x3 = Parameter(name = 'RNN1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.0483530815,
                   texname = '\\text{RNN1x3}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 1, 3 ])

RNN1x4 = Parameter(name = 'RNN1x4',
                   nature = 'external',
                   type = 'real',
                   value = -0.0149871707,
                   texname = '\\text{RNN1x4}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 1, 4 ])

RNN1x5 = Parameter(name = 'RNN1x5',
                   nature = 'external',
                   type = 'real',
                   value = 0.000430389009,
                   texname = '\\text{RNN1x5}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 1, 5 ])

RNN2x1 = Parameter(name = 'RNN2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.0138621789,
                   texname = '\\text{RNN2x1}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 2, 1 ])

RNN2x2 = Parameter(name = 'RNN2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.993268723,
                   texname = '\\text{RNN2x2}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 2, 2 ])

RNN2x3 = Parameter(name = 'RNN2x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.103118961,
                   texname = '\\text{RNN2x3}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 2, 3 ])

RNN2x4 = Parameter(name = 'RNN2x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.05089756,
                   texname = '\\text{RNN2x4}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 2, 4 ])

RNN2x5 = Parameter(name = 'RNN2x5',
                   nature = 'external',
                   type = 'real',
                   value = -0.00100117257,
                   texname = '\\text{RNN2x5}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 2, 5 ])

RNN3x1 = Parameter(name = 'RNN3x1',
                   nature = 'external',
                   type = 'real',
                   value = -0.0232278855,
                   texname = '\\text{RNN3x1}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 3, 1 ])

RNN3x2 = Parameter(name = 'RNN3x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.037295208,
                   texname = '\\text{RNN3x2}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 3, 2 ])

RNN3x3 = Parameter(name = 'RNN3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.705297681,
                   texname = '\\text{RNN3x3}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 3, 3 ])

RNN3x4 = Parameter(name = 'RNN3x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.707534724,
                   texname = '\\text{RNN3x4}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 3, 4 ])

RNN3x5 = Parameter(name = 'RNN3x5',
                   nature = 'external',
                   type = 'real',
                   value = 0.00439627968,
                   texname = '\\text{RNN3x5}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 3, 5 ])

RNN4x1 = Parameter(name = 'RNN4x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.0435606237,
                   texname = '\\text{RNN4x1}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 4, 1 ])

RNN4x2 = Parameter(name = 'RNN4x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.109361086,
                   texname = '\\text{RNN4x2}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 4, 2 ])

RNN4x3 = Parameter(name = 'RNN4x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.69963098,
                   texname = '\\text{RNN4x3}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 4, 3 ])

RNN4x4 = Parameter(name = 'RNN4x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.704673803,
                   texname = '\\text{RNN4x4}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 4, 4 ])

RNN4x5 = Parameter(name = 'RNN4x5',
                   nature = 'external',
                   type = 'real',
                   value = -0.00969268004,
                   texname = '\\text{RNN4x5}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 4, 5 ])

RNN5x1 = Parameter(name = 'RNN5x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.000108397267,
                   texname = '\\text{RNN5x1}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 5, 1 ])

RNN5x2 = Parameter(name = 'RNN5x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.000226034288,
                   texname = '\\text{RNN5x2}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 5, 2 ])

RNN5x3 = Parameter(name = 'RNN5x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.0100066083,
                   texname = '\\text{RNN5x3}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 5, 3 ])

RNN5x4 = Parameter(name = 'RNN5x4',
                   nature = 'external',
                   type = 'real',
                   value = 0.00377728091,
                   texname = '\\text{RNN5x4}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 5, 4 ])

RNN5x5 = Parameter(name = 'RNN5x5',
                   nature = 'external',
                   type = 'real',
                   value = 0.999942767,
                   texname = '\\text{RNN5x5}',
                   lhablock = 'NMNMIX',
                   lhacode = [ 5, 5 ])

NMl = Parameter(name = 'NMl',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = '\\lambda',
                lhablock = 'NMSSMRUN',
                lhacode = [ 1 ])

NMk = Parameter(name = 'NMk',
                nature = 'external',
                type = 'real',
                value = 0.108910706,
                texname = '\\kappa',
                lhablock = 'NMSSMRUN',
                lhacode = [ 2 ])

NMAl = Parameter(name = 'NMAl',
                 nature = 'external',
                 type = 'real',
                 value = -963.907478,
                 texname = 'A_{\\lambda }',
                 lhablock = 'NMSSMRUN',
                 lhacode = [ 3 ])

NMAk = Parameter(name = 'NMAk',
                 nature = 'external',
                 type = 'real',
                 value = -1.58927119,
                 texname = 'A_{\\kappa }',
                 lhablock = 'NMSSMRUN',
                 lhacode = [ 4 ])

mueff = Parameter(name = 'mueff',
                  nature = 'external',
                  type = 'real',
                  value = 970.86792,
                  texname = '\\mu _{\\text{eff}}',
                  lhablock = 'NMSSMRUN',
                  lhacode = [ 5 ])

MS2 = Parameter(name = 'MS2',
                nature = 'external',
                type = 'real',
                value = -2.23503099e6,
                texname = 'M_S^2',
                lhablock = 'NMSSMRUN',
                lhacode = [ 10 ])

bb = Parameter(name = 'bb',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = 'b',
               lhablock = 'NMSSMRUN',
               lhacode = [ 12 ])

RRl1x3 = Parameter(name = 'RRl1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.220980319,
                   texname = '\\text{RRl1x3}',
                   lhablock = 'SELMIX',
                   lhacode = [ 1, 3 ])

RRl1x6 = Parameter(name = 'RRl1x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.975278267,
                   texname = '\\text{RRl1x6}',
                   lhablock = 'SELMIX',
                   lhacode = [ 1, 6 ])

RRl2x4 = Parameter(name = 'RRl2x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl2x4}',
                   lhablock = 'SELMIX',
                   lhacode = [ 2, 4 ])

RRl3x5 = Parameter(name = 'RRl3x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl3x5}',
                   lhablock = 'SELMIX',
                   lhacode = [ 3, 5 ])

RRl4x3 = Parameter(name = 'RRl4x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.975278267,
                   texname = '\\text{RRl4x3}',
                   lhablock = 'SELMIX',
                   lhacode = [ 4, 3 ])

RRl4x6 = Parameter(name = 'RRl4x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.220980319,
                   texname = '\\text{RRl4x6}',
                   lhablock = 'SELMIX',
                   lhacode = [ 4, 6 ])

RRl5x1 = Parameter(name = 'RRl5x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl5x1}',
                   lhablock = 'SELMIX',
                   lhacode = [ 5, 1 ])

RRl6x2 = Parameter(name = 'RRl6x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRl6x2}',
                   lhablock = 'SELMIX',
                   lhacode = [ 6, 2 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.92,
                  texname = '\\alpha _w^{-1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1172,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

RRn1x3 = Parameter(name = 'RRn1x3',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn1x3}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 1, 3 ])

RRn2x2 = Parameter(name = 'RRn2x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn2x2}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 2, 2 ])

RRn3x1 = Parameter(name = 'RRn3x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRn3x1}',
                   lhablock = 'SNUMIX',
                   lhacode = [ 3, 1 ])

Rtd3x3 = Parameter(name = 'Rtd3x3',
                   nature = 'external',
                   type = 'real',
                   value = -342.310014,
                   texname = '\\text{Rtd3x3}',
                   lhablock = 'TD',
                   lhacode = [ 3, 3 ])

Rte3x3 = Parameter(name = 'Rte3x3',
                   nature = 'external',
                   type = 'real',
                   value = -177.121653,
                   texname = '\\text{Rte3x3}',
                   lhablock = 'TE',
                   lhacode = [ 3, 3 ])

Rtu3x3 = Parameter(name = 'Rtu3x3',
                   nature = 'external',
                   type = 'real',
                   value = -1213.64864,
                   texname = '\\text{Rtu3x3}',
                   lhablock = 'TU',
                   lhacode = [ 3, 3 ])

RUU1x1 = Parameter(name = 'RUU1x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.989230572,
                   texname = '\\text{RUU1x1}',
                   lhablock = 'UMIX',
                   lhacode = [ 1, 1 ])

RUU1x2 = Parameter(name = 'RUU1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.146365554,
                   texname = '\\text{RUU1x2}',
                   lhablock = 'UMIX',
                   lhacode = [ 1, 2 ])

RUU2x1 = Parameter(name = 'RUU2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.146365554,
                   texname = '\\text{RUU2x1}',
                   lhablock = 'UMIX',
                   lhacode = [ 2, 1 ])

RUU2x2 = Parameter(name = 'RUU2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.989230572,
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

RRu1x3 = Parameter(name = 'RRu1x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.405775656,
                   texname = '\\text{RRu1x3}',
                   lhablock = 'USQMIX',
                   lhacode = [ 1, 3 ])

RRu1x6 = Parameter(name = 'RRu1x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.913972711,
                   texname = '\\text{RRu1x6}',
                   lhablock = 'USQMIX',
                   lhacode = [ 1, 6 ])

RRu2x3 = Parameter(name = 'RRu2x3',
                   nature = 'external',
                   type = 'real',
                   value = -0.913972711,
                   texname = '\\text{RRu2x3}',
                   lhablock = 'USQMIX',
                   lhacode = [ 2, 3 ])

RRu2x6 = Parameter(name = 'RRu2x6',
                   nature = 'external',
                   type = 'real',
                   value = 0.405775656,
                   texname = '\\text{RRu2x6}',
                   lhablock = 'USQMIX',
                   lhacode = [ 2, 6 ])

RRu3x5 = Parameter(name = 'RRu3x5',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu3x5}',
                   lhablock = 'USQMIX',
                   lhacode = [ 3, 5 ])

RRu4x4 = Parameter(name = 'RRu4x4',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu4x4}',
                   lhablock = 'USQMIX',
                   lhacode = [ 4, 4 ])

RRu5x1 = Parameter(name = 'RRu5x1',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu5x1}',
                   lhablock = 'USQMIX',
                   lhacode = [ 5, 1 ])

RRu6x2 = Parameter(name = 'RRu6x2',
                   nature = 'external',
                   type = 'real',
                   value = 1.,
                   texname = '\\text{RRu6x2}',
                   lhablock = 'USQMIX',
                   lhacode = [ 6, 2 ])

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
                   value = 0.997382381,
                   texname = '\\text{RVV1x1}',
                   lhablock = 'VMIX',
                   lhacode = [ 1, 1 ])

RVV1x2 = Parameter(name = 'RVV1x2',
                   nature = 'external',
                   type = 'real',
                   value = -0.0723075752,
                   texname = '\\text{RVV1x2}',
                   lhablock = 'VMIX',
                   lhacode = [ 1, 2 ])

RVV2x1 = Parameter(name = 'RVV2x1',
                   nature = 'external',
                   type = 'real',
                   value = 0.0723075752,
                   texname = '\\text{RVV2x1}',
                   lhablock = 'VMIX',
                   lhacode = [ 2, 1 ])

RVV2x2 = Parameter(name = 'RVV2x2',
                   nature = 'external',
                   type = 'real',
                   value = 0.997382381,
                   texname = '\\text{RVV2x2}',
                   lhablock = 'VMIX',
                   lhacode = [ 2, 2 ])

Ryd3x3 = Parameter(name = 'Ryd3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.131064265,
                   texname = '\\text{Ryd3x3}',
                   lhablock = 'YD',
                   lhacode = [ 3, 3 ])

Rye3x3 = Parameter(name = 'Rye3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.100464794,
                   texname = '\\text{Rye3x3}',
                   lhablock = 'YE',
                   lhacode = [ 3, 3 ])

Ryu3x3 = Parameter(name = 'Ryu3x3',
                   nature = 'external',
                   type = 'real',
                   value = 0.847827829,
                   texname = '\\text{Ryu3x3}',
                   lhablock = 'YU',
                   lhacode = [ 3, 3 ])

MZ = Parameter(name = 'MZ',
               nature = 'external',
               type = 'real',
               value = 91.187,
               texname = '\\text{MZ}',
               lhablock = 'MASS',
               lhacode = [ 23 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 80.9387517,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

Mneu1 = Parameter(name = 'Mneu1',
                  nature = 'external',
                  type = 'real',
                  value = 208.141578,
                  texname = '\\text{Mneu1}',
                  lhablock = 'MASS',
                  lhacode = [ 1000022 ])

Mneu2 = Parameter(name = 'Mneu2',
                  nature = 'external',
                  type = 'real',
                  value = 397.851055,
                  texname = '\\text{Mneu2}',
                  lhablock = 'MASS',
                  lhacode = [ 1000023 ])

Mneu3 = Parameter(name = 'Mneu3',
                  nature = 'external',
                  type = 'real',
                  value = -963.980547,
                  texname = '\\text{Mneu3}',
                  lhablock = 'MASS',
                  lhacode = [ 1000025 ])

Mneu4 = Parameter(name = 'Mneu4',
                  nature = 'external',
                  type = 'real',
                  value = 969.59391,
                  texname = '\\text{Mneu4}',
                  lhablock = 'MASS',
                  lhacode = [ 1000035 ])

Mneu5 = Parameter(name = 'Mneu5',
                  nature = 'external',
                  type = 'real',
                  value = 2094.27413,
                  texname = '\\text{Mneu5}',
                  lhablock = 'MASS',
                  lhacode = [ 1000045 ])

Mch1 = Parameter(name = 'Mch1',
                 nature = 'external',
                 type = 'real',
                 value = 397.829545,
                 texname = '\\text{Mch1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000024 ])

Mch2 = Parameter(name = 'Mch2',
                 nature = 'external',
                 type = 'real',
                 value = 970.136817,
                 texname = '\\text{Mch2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000037 ])

Mgo = Parameter(name = 'Mgo',
                nature = 'external',
                type = 'real',
                value = 1151.54279,
                texname = '\\text{Mgo}',
                lhablock = 'MASS',
                lhacode = [ 1000021 ])

MH01 = Parameter(name = 'MH01',
                 nature = 'external',
                 type = 'real',
                 value = 119.163922,
                 texname = '\\text{MH01}',
                 lhablock = 'MASS',
                 lhacode = [ 25 ])

MH02 = Parameter(name = 'MH02',
                 nature = 'external',
                 type = 'real',
                 value = 1016.55127,
                 texname = '\\text{MH02}',
                 lhablock = 'MASS',
                 lhacode = [ 35 ])

MH03 = Parameter(name = 'MH03',
                 nature = 'external',
                 type = 'real',
                 value = 2112.57647,
                 texname = '\\text{MH03}',
                 lhablock = 'MASS',
                 lhacode = [ 45 ])

MA01 = Parameter(name = 'MA01',
                 nature = 'external',
                 type = 'real',
                 value = 40.3976968,
                 texname = '\\text{MA01}',
                 lhablock = 'MASS',
                 lhacode = [ 36 ])

MA02 = Parameter(name = 'MA02',
                 nature = 'external',
                 type = 'real',
                 value = 1021.04211,
                 texname = '\\text{MA02}',
                 lhablock = 'MASS',
                 lhacode = [ 46 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 1022.47183,
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
               value = 171.4,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.214,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

Msn1 = Parameter(name = 'Msn1',
                 nature = 'external',
                 type = 'real',
                 value = 360.340595,
                 texname = '\\text{Msn1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000012 ])

Msn2 = Parameter(name = 'Msn2',
                 nature = 'external',
                 type = 'real',
                 value = 372.121162,
                 texname = '\\text{Msn2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000014 ])

Msn3 = Parameter(name = 'Msn3',
                 nature = 'external',
                 type = 'real',
                 value = 372.121162,
                 texname = '\\text{Msn3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000016 ])

Msl1 = Parameter(name = 'Msl1',
                 nature = 'external',
                 type = 'real',
                 value = 214.739576,
                 texname = '\\text{Msl1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000011 ])

Msl2 = Parameter(name = 'Msl2',
                 nature = 'external',
                 type = 'real',
                 value = 261.024365,
                 texname = '\\text{Msl2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000013 ])

Msl3 = Parameter(name = 'Msl3',
                 nature = 'external',
                 type = 'real',
                 value = 261.024365,
                 texname = '\\text{Msl3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000015 ])

Msl4 = Parameter(name = 'Msl4',
                 nature = 'external',
                 type = 'real',
                 value = 374.857439,
                 texname = '\\text{Msl4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000011 ])

Msl5 = Parameter(name = 'Msl5',
                 nature = 'external',
                 type = 'real',
                 value = 380.175937,
                 texname = '\\text{Msl5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000013 ])

Msl6 = Parameter(name = 'Msl6',
                 nature = 'external',
                 type = 'real',
                 value = 380.175937,
                 texname = '\\text{Msl6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000015 ])

Msu1 = Parameter(name = 'Msu1',
                 nature = 'external',
                 type = 'real',
                 value = 499.229271,
                 texname = '\\text{Msu1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000002 ])

Msu2 = Parameter(name = 'Msu2',
                 nature = 'external',
                 type = 'real',
                 value = 935.527355,
                 texname = '\\text{Msu2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000004 ])

Msu3 = Parameter(name = 'Msu3',
                 nature = 'external',
                 type = 'real',
                 value = 1027.25889,
                 texname = '\\text{Msu3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000006 ])

Msu4 = Parameter(name = 'Msu4',
                 nature = 'external',
                 type = 'real',
                 value = 1027.25889,
                 texname = '\\text{Msu4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000002 ])

Msu5 = Parameter(name = 'Msu5',
                 nature = 'external',
                 type = 'real',
                 value = 1063.63463,
                 texname = '\\text{Msu5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000004 ])

Msu6 = Parameter(name = 'Msu6',
                 nature = 'external',
                 type = 'real',
                 value = 1063.63463,
                 texname = '\\text{Msu6}',
                 lhablock = 'MASS',
                 lhacode = [ 2000006 ])

Msd1 = Parameter(name = 'Msd1',
                 nature = 'external',
                 type = 'real',
                 value = 866.365161,
                 texname = '\\text{Msd1}',
                 lhablock = 'MASS',
                 lhacode = [ 1000001 ])

Msd2 = Parameter(name = 'Msd2',
                 nature = 'external',
                 type = 'real',
                 value = 992.138103,
                 texname = '\\text{Msd2}',
                 lhablock = 'MASS',
                 lhacode = [ 1000003 ])

Msd3 = Parameter(name = 'Msd3',
                 nature = 'external',
                 type = 'real',
                 value = 1023.75369,
                 texname = '\\text{Msd3}',
                 lhablock = 'MASS',
                 lhacode = [ 1000005 ])

Msd4 = Parameter(name = 'Msd4',
                 nature = 'external',
                 type = 'real',
                 value = 1023.75369,
                 texname = '\\text{Msd4}',
                 lhablock = 'MASS',
                 lhacode = [ 2000001 ])

Msd5 = Parameter(name = 'Msd5',
                 nature = 'external',
                 type = 'real',
                 value = 1066.51941,
                 texname = '\\text{Msd5}',
                 lhablock = 'MASS',
                 lhacode = [ 2000003 ])

Msd6 = Parameter(name = 'Msd6',
                 nature = 'external',
                 type = 'real',
                 value = 1066.51941,
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
                  value = 2.,
                  texname = '\\text{Wneu2}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000023 ])

Wneu3 = Parameter(name = 'Wneu3',
                  nature = 'external',
                  type = 'real',
                  value = 2.,
                  texname = '\\text{Wneu3}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000025 ])

Wneu4 = Parameter(name = 'Wneu4',
                  nature = 'external',
                  type = 'real',
                  value = 2.,
                  texname = '\\text{Wneu4}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000035 ])

Wneu5 = Parameter(name = 'Wneu5',
                  nature = 'external',
                  type = 'real',
                  value = 2.,
                  texname = '\\text{Wneu5}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000045 ])

Wch1 = Parameter(name = 'Wch1',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wch1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000024 ])

Wch2 = Parameter(name = 'Wch2',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wch2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000037 ])

Wgo = Parameter(name = 'Wgo',
                nature = 'external',
                type = 'real',
                value = 2.,
                texname = '\\text{Wgo}',
                lhablock = 'DECAY',
                lhacode = [ 1000021 ])

WH01 = Parameter(name = 'WH01',
                 nature = 'external',
                 type = 'real',
                 value = 0.0303786329,
                 texname = '\\text{WH01}',
                 lhablock = 'DECAY',
                 lhacode = [ 25 ])

WH02 = Parameter(name = 'WH02',
                 nature = 'external',
                 type = 'real',
                 value = 4.9565785,
                 texname = '\\text{WH02}',
                 lhablock = 'DECAY',
                 lhacode = [ 35 ])

WH03 = Parameter(name = 'WH03',
                 nature = 'external',
                 type = 'real',
                 value = 1.11808339,
                 texname = '\\text{WH03}',
                 lhablock = 'DECAY',
                 lhacode = [ 45 ])

WA01 = Parameter(name = 'WA01',
                 nature = 'external',
                 type = 'real',
                 value = 0.000249558656,
                 texname = '\\text{WA01}',
                 lhablock = 'DECAY',
                 lhacode = [ 36 ])

WA02 = Parameter(name = 'WA02',
                 nature = 'external',
                 type = 'real',
                 value = 3.50947871,
                 texname = '\\text{WA02}',
                 lhablock = 'DECAY',
                 lhacode = [ 46 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 3.25001093,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 37 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.33482521,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

Wsn1 = Parameter(name = 'Wsn1',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsn1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000012 ])

Wsn2 = Parameter(name = 'Wsn2',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsn2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000014 ])

Wsn3 = Parameter(name = 'Wsn3',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsn3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000016 ])

Wsl1 = Parameter(name = 'Wsl1',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsl1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000011 ])

Wsl2 = Parameter(name = 'Wsl2',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsl2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000013 ])

Wsl3 = Parameter(name = 'Wsl3',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsl3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000015 ])

Wsl4 = Parameter(name = 'Wsl4',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsl4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000011 ])

Wsl5 = Parameter(name = 'Wsl5',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsl5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000013 ])

Wsl6 = Parameter(name = 'Wsl6',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsl6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000015 ])

Wsu1 = Parameter(name = 'Wsu1',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsu1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000002 ])

Wsu2 = Parameter(name = 'Wsu2',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsu2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000004 ])

Wsu3 = Parameter(name = 'Wsu3',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsu3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000006 ])

Wsu4 = Parameter(name = 'Wsu4',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsu4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000002 ])

Wsu5 = Parameter(name = 'Wsu5',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsu5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000004 ])

Wsu6 = Parameter(name = 'Wsu6',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsu6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000006 ])

Wsd1 = Parameter(name = 'Wsd1',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsd1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000001 ])

Wsd2 = Parameter(name = 'Wsd2',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsd2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000003 ])

Wsd3 = Parameter(name = 'Wsd3',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsd3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000005 ])

Wsd4 = Parameter(name = 'Wsd4',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsd4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000001 ])

Wsd5 = Parameter(name = 'Wsd5',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsd5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000003 ])

Wsd6 = Parameter(name = 'Wsd6',
                 nature = 'external',
                 type = 'real',
                 value = 2.,
                 texname = '\\text{Wsd6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000005 ])

beta = Parameter(name = 'beta',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.atan(tb)',
                 texname = '\\beta')

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

NN1x5 = Parameter(name = 'NN1x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN1x5',
                  texname = '\\text{NN1x5}')

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

NN2x5 = Parameter(name = 'NN2x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN2x5',
                  texname = '\\text{NN2x5}')

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

NN3x5 = Parameter(name = 'NN3x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN3x5',
                  texname = '\\text{NN3x5}')

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

NN4x5 = Parameter(name = 'NN4x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN4x5',
                  texname = '\\text{NN4x5}')

NN5x1 = Parameter(name = 'NN5x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN5x1',
                  texname = '\\text{NN5x1}')

NN5x2 = Parameter(name = 'NN5x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN5x2',
                  texname = '\\text{NN5x2}')

NN5x3 = Parameter(name = 'NN5x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN5x3',
                  texname = '\\text{NN5x3}')

NN5x4 = Parameter(name = 'NN5x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN5x4',
                  texname = '\\text{NN5x4}')

NN5x5 = Parameter(name = 'NN5x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RNN5x5',
                  texname = '\\text{NN5x5}')

Rd1x3 = Parameter(name = 'Rd1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd1x3',
                  texname = '\\text{Rd1x3}')

Rd1x6 = Parameter(name = 'Rd1x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd1x6',
                  texname = '\\text{Rd1x6}')

Rd2x3 = Parameter(name = 'Rd2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd2x3',
                  texname = '\\text{Rd2x3}')

Rd2x6 = Parameter(name = 'Rd2x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd2x6',
                  texname = '\\text{Rd2x6}')

Rd3x5 = Parameter(name = 'Rd3x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd3x5',
                  texname = '\\text{Rd3x5}')

Rd4x4 = Parameter(name = 'Rd4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd4x4',
                  texname = '\\text{Rd4x4}')

Rd5x1 = Parameter(name = 'Rd5x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd5x1',
                  texname = '\\text{Rd5x1}')

Rd6x2 = Parameter(name = 'Rd6x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRd6x2',
                  texname = '\\text{Rd6x2}')

Rl1x3 = Parameter(name = 'Rl1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl1x3',
                  texname = '\\text{Rl1x3}')

Rl1x6 = Parameter(name = 'Rl1x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl1x6',
                  texname = '\\text{Rl1x6}')

Rl2x4 = Parameter(name = 'Rl2x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl2x4',
                  texname = '\\text{Rl2x4}')

Rl3x5 = Parameter(name = 'Rl3x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl3x5',
                  texname = '\\text{Rl3x5}')

Rl4x3 = Parameter(name = 'Rl4x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl4x3',
                  texname = '\\text{Rl4x3}')

Rl4x6 = Parameter(name = 'Rl4x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl4x6',
                  texname = '\\text{Rl4x6}')

Rl5x1 = Parameter(name = 'Rl5x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl5x1',
                  texname = '\\text{Rl5x1}')

Rl6x2 = Parameter(name = 'Rl6x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRl6x2',
                  texname = '\\text{Rl6x2}')

Rn1x3 = Parameter(name = 'Rn1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn1x3',
                  texname = '\\text{Rn1x3}')

Rn2x2 = Parameter(name = 'Rn2x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn2x2',
                  texname = '\\text{Rn2x2}')

Rn3x1 = Parameter(name = 'Rn3x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRn3x1',
                  texname = '\\text{Rn3x1}')

Ru1x3 = Parameter(name = 'Ru1x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu1x3',
                  texname = '\\text{Ru1x3}')

Ru1x6 = Parameter(name = 'Ru1x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu1x6',
                  texname = '\\text{Ru1x6}')

Ru2x3 = Parameter(name = 'Ru2x3',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu2x3',
                  texname = '\\text{Ru2x3}')

Ru2x6 = Parameter(name = 'Ru2x6',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu2x6',
                  texname = '\\text{Ru2x6}')

Ru3x5 = Parameter(name = 'Ru3x5',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu3x5',
                  texname = '\\text{Ru3x5}')

Ru4x4 = Parameter(name = 'Ru4x4',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu4x4',
                  texname = '\\text{Ru4x4}')

Ru5x1 = Parameter(name = 'Ru5x1',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu5x1',
                  texname = '\\text{Ru5x1}')

Ru6x2 = Parameter(name = 'Ru6x2',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RRu6x2',
                  texname = '\\text{Ru6x2}')

UP31 = Parameter(name = 'UP31',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1 - UP1x1**2 - UP2x1**2)',
                 texname = 'U_P^{31}')

UP32 = Parameter(name = 'UP32',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1 - UP1x2**2 - UP2x2**2)',
                 texname = 'U_P^{32}')

UP33 = Parameter(name = 'UP33',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1 - UP1x3**2 - UP2x3**2)',
                 texname = 'U_P^{33}')

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

vs = Parameter(name = 'vs',
               nature = 'internal',
               type = 'real',
               value = '(mueff*cmath.sqrt(2))/NMl',
               texname = 'v_s')

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
                  value = 'complexconjugate(yu3x3)',
                  texname = '\\text{I1x33}')

I10x31 = Parameter(name = 'I10x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(yd3x3)',
                   texname = '\\text{I10x31}')

I10x32 = Parameter(name = 'I10x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(yd3x3)',
                   texname = '\\text{I10x32}')

I100x11 = Parameter(name = 'I100x11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x6*complexconjugate(Rd1x6)',
                    texname = '\\text{I100x11}')

I100x12 = Parameter(name = 'I100x12',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x6*complexconjugate(Rd1x6)',
                    texname = '\\text{I100x12}')

I100x21 = Parameter(name = 'I100x21',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd1x6*complexconjugate(Rd2x6)',
                    texname = '\\text{I100x21}')

I100x22 = Parameter(name = 'I100x22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd2x6*complexconjugate(Rd2x6)',
                    texname = '\\text{I100x22}')

I100x33 = Parameter(name = 'I100x33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd3x5*complexconjugate(Rd3x5)',
                    texname = '\\text{I100x33}')

I100x44 = Parameter(name = 'I100x44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rd4x4*complexconjugate(Rd4x4)',
                    texname = '\\text{I100x44}')

I101x11 = Parameter(name = 'I101x11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x6*complexconjugate(Rl1x6)',
                    texname = '\\text{I101x11}')

I101x14 = Parameter(name = 'I101x14',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x6*complexconjugate(Rl1x6)',
                    texname = '\\text{I101x14}')

I101x22 = Parameter(name = 'I101x22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl2x4*complexconjugate(Rl2x4)',
                    texname = '\\text{I101x22}')

I101x33 = Parameter(name = 'I101x33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl3x5*complexconjugate(Rl3x5)',
                    texname = '\\text{I101x33}')

I101x41 = Parameter(name = 'I101x41',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl1x6*complexconjugate(Rl4x6)',
                    texname = '\\text{I101x41}')

I101x44 = Parameter(name = 'I101x44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Rl4x6*complexconjugate(Rl4x6)',
                    texname = '\\text{I101x44}')

I102x11 = Parameter(name = 'I102x11',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru1x6*complexconjugate(Ru1x6)',
                    texname = '\\text{I102x11}')

I102x12 = Parameter(name = 'I102x12',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru2x6*complexconjugate(Ru1x6)',
                    texname = '\\text{I102x12}')

I102x21 = Parameter(name = 'I102x21',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru1x6*complexconjugate(Ru2x6)',
                    texname = '\\text{I102x21}')

I102x22 = Parameter(name = 'I102x22',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru2x6*complexconjugate(Ru2x6)',
                    texname = '\\text{I102x22}')

I102x33 = Parameter(name = 'I102x33',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru3x5*complexconjugate(Ru3x5)',
                    texname = '\\text{I102x33}')

I102x44 = Parameter(name = 'I102x44',
                    nature = 'internal',
                    type = 'complex',
                    value = 'Ru4x4*complexconjugate(Ru4x4)',
                    texname = '\\text{I102x44}')

I11x31 = Parameter(name = 'I11x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3',
                   texname = '\\text{I11x31}')

I11x32 = Parameter(name = 'I11x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3',
                   texname = '\\text{I11x32}')

I12x11 = Parameter(name = 'I12x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I12x11}')

I12x12 = Parameter(name = 'I12x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I12x12}')

I12x21 = Parameter(name = 'I12x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I12x21}')

I12x22 = Parameter(name = 'I12x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I12x22}')

I12x55 = Parameter(name = 'I12x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd5x1*complexconjugate(Rd5x1)',
                   texname = '\\text{I12x55}')

I12x66 = Parameter(name = 'I12x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x2*complexconjugate(Rd6x2)',
                   texname = '\\text{I12x66}')

I13x11 = Parameter(name = 'I13x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*complexconjugate(Rd1x6)',
                   texname = '\\text{I13x11}')

I13x12 = Parameter(name = 'I13x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*complexconjugate(Rd1x6)',
                   texname = '\\text{I13x12}')

I13x21 = Parameter(name = 'I13x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*complexconjugate(Rd2x6)',
                   texname = '\\text{I13x21}')

I13x22 = Parameter(name = 'I13x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*complexconjugate(Rd2x6)',
                   texname = '\\text{I13x22}')

I13x33 = Parameter(name = 'I13x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd3x5*complexconjugate(Rd3x5)',
                   texname = '\\text{I13x33}')

I13x44 = Parameter(name = 'I13x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd4x4*complexconjugate(Rd4x4)',
                   texname = '\\text{I13x44}')

I14x33 = Parameter(name = 'I14x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(yd3x3)',
                   texname = '\\text{I14x33}')

I15x33 = Parameter(name = 'I15x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3',
                   texname = '\\text{I15x33}')

I16x33 = Parameter(name = 'I16x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(ye3x3)',
                   texname = '\\text{I16x33}')

I17x31 = Parameter(name = 'I17x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl1x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I17x31}')

I17x34 = Parameter(name = 'I17x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl4x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I17x34}')

I18x31 = Parameter(name = 'I18x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I18x31}')

I18x34 = Parameter(name = 'I18x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I18x34}')

I19x11 = Parameter(name = 'I19x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I19x11}')

I19x14 = Parameter(name = 'I19x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I19x14}')

I19x41 = Parameter(name = 'I19x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I19x41}')

I19x44 = Parameter(name = 'I19x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I19x44}')

I19x55 = Parameter(name = 'I19x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x1*complexconjugate(Rl5x1)',
                   texname = '\\text{I19x55}')

I19x66 = Parameter(name = 'I19x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x2*complexconjugate(Rl6x2)',
                   texname = '\\text{I19x66}')

I2x33 = Parameter(name = 'I2x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3',
                  texname = '\\text{I2x33}')

I20x11 = Parameter(name = 'I20x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*complexconjugate(Rl1x6)',
                   texname = '\\text{I20x11}')

I20x14 = Parameter(name = 'I20x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*complexconjugate(Rl1x6)',
                   texname = '\\text{I20x14}')

I20x22 = Parameter(name = 'I20x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x4*complexconjugate(Rl2x4)',
                   texname = '\\text{I20x22}')

I20x33 = Parameter(name = 'I20x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x5*complexconjugate(Rl3x5)',
                   texname = '\\text{I20x33}')

I20x41 = Parameter(name = 'I20x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*complexconjugate(Rl4x6)',
                   texname = '\\text{I20x41}')

I20x44 = Parameter(name = 'I20x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*complexconjugate(Rl4x6)',
                   texname = '\\text{I20x44}')

I21x31 = Parameter(name = 'I21x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(ye3x3)',
                   texname = '\\text{I21x31}')

I21x34 = Parameter(name = 'I21x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(ye3x3)',
                   texname = '\\text{I21x34}')

I22x31 = Parameter(name = 'I22x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*ye3x3',
                   texname = '\\text{I22x31}')

I22x34 = Parameter(name = 'I22x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*ye3x3',
                   texname = '\\text{I22x34}')

I23x15 = Parameter(name = 'I23x15',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x1',
                   texname = '\\text{I23x15}')

I23x26 = Parameter(name = 'I23x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x2',
                   texname = '\\text{I23x26}')

I23x31 = Parameter(name = 'I23x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3',
                   texname = '\\text{I23x31}')

I23x34 = Parameter(name = 'I23x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3',
                   texname = '\\text{I23x34}')

I24x31 = Parameter(name = 'I24x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*ye3x3',
                   texname = '\\text{I24x31}')

I24x34 = Parameter(name = 'I24x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*ye3x3',
                   texname = '\\text{I24x34}')

I25x11 = Parameter(name = 'I25x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I25x11}')

I25x14 = Parameter(name = 'I25x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I25x14}')

I25x41 = Parameter(name = 'I25x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I25x41}')

I25x44 = Parameter(name = 'I25x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I25x44}')

I25x55 = Parameter(name = 'I25x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x1*complexconjugate(Rl5x1)',
                   texname = '\\text{I25x55}')

I25x66 = Parameter(name = 'I25x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x2*complexconjugate(Rl6x2)',
                   texname = '\\text{I25x66}')

I26x11 = Parameter(name = 'I26x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*complexconjugate(Rl1x6)',
                   texname = '\\text{I26x11}')

I26x14 = Parameter(name = 'I26x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*complexconjugate(Rl1x6)',
                   texname = '\\text{I26x14}')

I26x22 = Parameter(name = 'I26x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl2x4*complexconjugate(Rl2x4)',
                   texname = '\\text{I26x22}')

I26x33 = Parameter(name = 'I26x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl3x5*complexconjugate(Rl3x5)',
                   texname = '\\text{I26x33}')

I26x41 = Parameter(name = 'I26x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*complexconjugate(Rl4x6)',
                   texname = '\\text{I26x41}')

I26x44 = Parameter(name = 'I26x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*complexconjugate(Rl4x6)',
                   texname = '\\text{I26x44}')

I27x11 = Parameter(name = 'I27x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I27x11}')

I27x14 = Parameter(name = 'I27x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I27x14}')

I27x26 = Parameter(name = 'I27x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x2*complexconjugate(Rn2x2)',
                   texname = '\\text{I27x26}')

I27x35 = Parameter(name = 'I27x35',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x1*complexconjugate(Rn3x1)',
                   texname = '\\text{I27x35}')

I28x11 = Parameter(name = 'I28x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*te3x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I28x11}')

I28x14 = Parameter(name = 'I28x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*te3x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I28x14}')

I29x11 = Parameter(name = 'I29x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*ye3x3*complexconjugate(Rn1x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I29x11}')

I29x14 = Parameter(name = 'I29x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*ye3x3*complexconjugate(Rn1x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I29x14}')

I3x31 = Parameter(name = 'I3x31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                  texname = '\\text{I3x31}')

I3x32 = Parameter(name = 'I3x32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                  texname = '\\text{I3x32}')

I30x11 = Parameter(name = 'I30x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*ye3x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I30x11}')

I30x14 = Parameter(name = 'I30x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*ye3x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I30x14}')

I31x13 = Parameter(name = 'I31x13',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x1',
                   texname = '\\text{I31x13}')

I31x22 = Parameter(name = 'I31x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2',
                   texname = '\\text{I31x22}')

I31x31 = Parameter(name = 'I31x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3',
                   texname = '\\text{I31x31}')

I32x31 = Parameter(name = 'I32x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(ye3x3)',
                   texname = '\\text{I32x31}')

I33x11 = Parameter(name = 'I33x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I33x11}')

I33x14 = Parameter(name = 'I33x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I33x14}')

I33x26 = Parameter(name = 'I33x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2*complexconjugate(Rl6x2)',
                   texname = '\\text{I33x26}')

I33x35 = Parameter(name = 'I33x35',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x1*complexconjugate(Rl5x1)',
                   texname = '\\text{I33x35}')

I34x11 = Parameter(name = 'I34x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl1x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I34x11}')

I34x14 = Parameter(name = 'I34x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl4x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I34x14}')

I35x11 = Parameter(name = 'I35x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl1x6)*complexconjugate(te3x3)',
                   texname = '\\text{I35x11}')

I35x14 = Parameter(name = 'I35x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl4x6)*complexconjugate(te3x3)',
                   texname = '\\text{I35x14}')

I36x11 = Parameter(name = 'I36x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*ye3x3*complexconjugate(Rl1x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I36x11}')

I36x14 = Parameter(name = 'I36x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*ye3x3*complexconjugate(Rl4x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I36x14}')

I37x31 = Parameter(name = 'I37x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I37x31}')

I37x32 = Parameter(name = 'I37x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I37x32}')

I38x31 = Parameter(name = 'I38x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I38x31}')

I38x32 = Parameter(name = 'I38x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I38x32}')

I39x11 = Parameter(name = 'I39x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I39x11}')

I39x12 = Parameter(name = 'I39x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I39x12}')

I39x21 = Parameter(name = 'I39x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I39x21}')

I39x22 = Parameter(name = 'I39x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I39x22}')

I39x55 = Parameter(name = 'I39x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x1*complexconjugate(Ru5x1)',
                   texname = '\\text{I39x55}')

I39x66 = Parameter(name = 'I39x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x2*complexconjugate(Ru6x2)',
                   texname = '\\text{I39x66}')

I4x31 = Parameter(name = 'I4x31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3*complexconjugate(Rd1x3)',
                  texname = '\\text{I4x31}')

I4x32 = Parameter(name = 'I4x32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd3x3*complexconjugate(Rd2x3)',
                  texname = '\\text{I4x32}')

I40x11 = Parameter(name = 'I40x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*complexconjugate(Ru1x6)',
                   texname = '\\text{I40x11}')

I40x12 = Parameter(name = 'I40x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*complexconjugate(Ru1x6)',
                   texname = '\\text{I40x12}')

I40x21 = Parameter(name = 'I40x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*complexconjugate(Ru2x6)',
                   texname = '\\text{I40x21}')

I40x22 = Parameter(name = 'I40x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*complexconjugate(Ru2x6)',
                   texname = '\\text{I40x22}')

I40x33 = Parameter(name = 'I40x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x5*complexconjugate(Ru3x5)',
                   texname = '\\text{I40x33}')

I40x44 = Parameter(name = 'I40x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru4x4*complexconjugate(Ru4x4)',
                   texname = '\\text{I40x44}')

I41x11 = Parameter(name = 'I41x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I41x11}')

I41x12 = Parameter(name = 'I41x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I41x12}')

I41x21 = Parameter(name = 'I41x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I41x21}')

I41x22 = Parameter(name = 'I41x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I41x22}')

I41x55 = Parameter(name = 'I41x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd5x1*complexconjugate(Ru5x1)',
                   texname = '\\text{I41x55}')

I41x66 = Parameter(name = 'I41x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x2*complexconjugate(Ru6x2)',
                   texname = '\\text{I41x66}')

I42x11 = Parameter(name = 'I42x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru1x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I42x11}')

I42x12 = Parameter(name = 'I42x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru2x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I42x12}')

I42x21 = Parameter(name = 'I42x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru1x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I42x21}')

I42x22 = Parameter(name = 'I42x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru2x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I42x22}')

I43x11 = Parameter(name = 'I43x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I43x11}')

I43x12 = Parameter(name = 'I43x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I43x12}')

I43x21 = Parameter(name = 'I43x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I43x21}')

I43x22 = Parameter(name = 'I43x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I43x22}')

I44x11 = Parameter(name = 'I44x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*td3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I44x11}')

I44x12 = Parameter(name = 'I44x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*td3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I44x12}')

I44x21 = Parameter(name = 'I44x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*td3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I44x21}')

I44x22 = Parameter(name = 'I44x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*td3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I44x22}')

I45x11 = Parameter(name = 'I45x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I45x11}')

I45x12 = Parameter(name = 'I45x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I45x12}')

I45x21 = Parameter(name = 'I45x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I45x21}')

I45x22 = Parameter(name = 'I45x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I45x22}')

I46x11 = Parameter(name = 'I46x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*yd3x3*complexconjugate(Ru1x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I46x11}')

I46x12 = Parameter(name = 'I46x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*yd3x3*complexconjugate(Ru2x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I46x12}')

I46x21 = Parameter(name = 'I46x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*yd3x3*complexconjugate(Ru1x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I46x21}')

I46x22 = Parameter(name = 'I46x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*yd3x3*complexconjugate(Ru2x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I46x22}')

I47x11 = Parameter(name = 'I47x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I47x11}')

I47x12 = Parameter(name = 'I47x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I47x12}')

I47x21 = Parameter(name = 'I47x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I47x21}')

I47x22 = Parameter(name = 'I47x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I47x22}')

I48x11 = Parameter(name = 'I48x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*yu3x3*complexconjugate(Ru1x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I48x11}')

I48x12 = Parameter(name = 'I48x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*yu3x3*complexconjugate(Ru2x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I48x12}')

I48x21 = Parameter(name = 'I48x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*yu3x3*complexconjugate(Ru1x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I48x21}')

I48x22 = Parameter(name = 'I48x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*yu3x3*complexconjugate(Ru2x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I48x22}')

I49x31 = Parameter(name = 'I49x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(yu3x3)',
                   texname = '\\text{I49x31}')

I49x32 = Parameter(name = 'I49x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(yu3x3)',
                   texname = '\\text{I49x32}')

I5x11 = Parameter(name = 'I5x11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x3*complexconjugate(Rd1x3)',
                  texname = '\\text{I5x11}')

I5x12 = Parameter(name = 'I5x12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x3*complexconjugate(Rd1x3)',
                  texname = '\\text{I5x12}')

I5x21 = Parameter(name = 'I5x21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x3*complexconjugate(Rd2x3)',
                  texname = '\\text{I5x21}')

I5x22 = Parameter(name = 'I5x22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x3*complexconjugate(Rd2x3)',
                  texname = '\\text{I5x22}')

I5x55 = Parameter(name = 'I5x55',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd5x1*complexconjugate(Rd5x1)',
                  texname = '\\text{I5x55}')

I5x66 = Parameter(name = 'I5x66',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x2*complexconjugate(Rd6x2)',
                  texname = '\\text{I5x66}')

I50x31 = Parameter(name = 'I50x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3',
                   texname = '\\text{I50x31}')

I50x32 = Parameter(name = 'I50x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3',
                   texname = '\\text{I50x32}')

I51x15 = Parameter(name = 'I51x15',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x1',
                   texname = '\\text{I51x15}')

I51x26 = Parameter(name = 'I51x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x2',
                   texname = '\\text{I51x26}')

I51x31 = Parameter(name = 'I51x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3',
                   texname = '\\text{I51x31}')

I51x32 = Parameter(name = 'I51x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3',
                   texname = '\\text{I51x32}')

I52x31 = Parameter(name = 'I52x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(yd3x3)',
                   texname = '\\text{I52x31}')

I52x32 = Parameter(name = 'I52x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(yd3x3)',
                   texname = '\\text{I52x32}')

I53x31 = Parameter(name = 'I53x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3',
                   texname = '\\text{I53x31}')

I53x32 = Parameter(name = 'I53x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3',
                   texname = '\\text{I53x32}')

I54x11 = Parameter(name = 'I54x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I54x11}')

I54x12 = Parameter(name = 'I54x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I54x12}')

I54x21 = Parameter(name = 'I54x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I54x21}')

I54x22 = Parameter(name = 'I54x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I54x22}')

I54x55 = Parameter(name = 'I54x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x1*complexconjugate(Rd5x1)',
                   texname = '\\text{I54x55}')

I54x66 = Parameter(name = 'I54x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x2*complexconjugate(Rd6x2)',
                   texname = '\\text{I54x66}')

I55x11 = Parameter(name = 'I55x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I55x11}')

I55x12 = Parameter(name = 'I55x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I55x12}')

I55x21 = Parameter(name = 'I55x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I55x21}')

I55x22 = Parameter(name = 'I55x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I55x22}')

I56x11 = Parameter(name = 'I56x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd1x6)*complexconjugate(td3x3)',
                   texname = '\\text{I56x11}')

I56x12 = Parameter(name = 'I56x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd1x6)*complexconjugate(td3x3)',
                   texname = '\\text{I56x12}')

I56x21 = Parameter(name = 'I56x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd2x6)*complexconjugate(td3x3)',
                   texname = '\\text{I56x21}')

I56x22 = Parameter(name = 'I56x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd2x6)*complexconjugate(td3x3)',
                   texname = '\\text{I56x22}')

I57x11 = Parameter(name = 'I57x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*tu3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I57x11}')

I57x12 = Parameter(name = 'I57x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*tu3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I57x12}')

I57x21 = Parameter(name = 'I57x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*tu3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I57x21}')

I57x22 = Parameter(name = 'I57x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*tu3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I57x22}')

I58x11 = Parameter(name = 'I58x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*yd3x3*complexconjugate(Rd1x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x11}')

I58x12 = Parameter(name = 'I58x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*yd3x3*complexconjugate(Rd1x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x12}')

I58x21 = Parameter(name = 'I58x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*yd3x3*complexconjugate(Rd2x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x21}')

I58x22 = Parameter(name = 'I58x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*yd3x3*complexconjugate(Rd2x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I58x22}')

I59x11 = Parameter(name = 'I59x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*yu3x3*complexconjugate(Rd1x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x11}')

I59x12 = Parameter(name = 'I59x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*yu3x3*complexconjugate(Rd1x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x12}')

I59x21 = Parameter(name = 'I59x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*yu3x3*complexconjugate(Rd2x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x21}')

I59x22 = Parameter(name = 'I59x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*yu3x3*complexconjugate(Rd2x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I59x22}')

I6x11 = Parameter(name = 'I6x11',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x6*complexconjugate(Rd1x6)',
                  texname = '\\text{I6x11}')

I6x12 = Parameter(name = 'I6x12',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x6*complexconjugate(Rd1x6)',
                  texname = '\\text{I6x12}')

I6x21 = Parameter(name = 'I6x21',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x6*complexconjugate(Rd2x6)',
                  texname = '\\text{I6x21}')

I6x22 = Parameter(name = 'I6x22',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x6*complexconjugate(Rd2x6)',
                  texname = '\\text{I6x22}')

I6x33 = Parameter(name = 'I6x33',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd3x5*complexconjugate(Rd3x5)',
                  texname = '\\text{I6x33}')

I6x44 = Parameter(name = 'I6x44',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd4x4*complexconjugate(Rd4x4)',
                  texname = '\\text{I6x44}')

I60x11 = Parameter(name = 'I60x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I60x11}')

I60x12 = Parameter(name = 'I60x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I60x12}')

I60x21 = Parameter(name = 'I60x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I60x21}')

I60x22 = Parameter(name = 'I60x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I60x22}')

I61x11 = Parameter(name = 'I61x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I61x11}')

I61x12 = Parameter(name = 'I61x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I61x12}')

I61x21 = Parameter(name = 'I61x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I61x21}')

I61x22 = Parameter(name = 'I61x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I61x22}')

I62x11 = Parameter(name = 'I62x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I62x11}')

I62x12 = Parameter(name = 'I62x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I62x12}')

I62x21 = Parameter(name = 'I62x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I62x21}')

I62x22 = Parameter(name = 'I62x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I62x22}')

I62x55 = Parameter(name = 'I62x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x1*complexconjugate(Ru5x1)',
                   texname = '\\text{I62x55}')

I62x66 = Parameter(name = 'I62x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x2*complexconjugate(Ru6x2)',
                   texname = '\\text{I62x66}')

I63x11 = Parameter(name = 'I63x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*complexconjugate(Ru1x6)',
                   texname = '\\text{I63x11}')

I63x12 = Parameter(name = 'I63x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*complexconjugate(Ru1x6)',
                   texname = '\\text{I63x12}')

I63x21 = Parameter(name = 'I63x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*complexconjugate(Ru2x6)',
                   texname = '\\text{I63x21}')

I63x22 = Parameter(name = 'I63x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*complexconjugate(Ru2x6)',
                   texname = '\\text{I63x22}')

I63x33 = Parameter(name = 'I63x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru3x5*complexconjugate(Ru3x5)',
                   texname = '\\text{I63x33}')

I63x44 = Parameter(name = 'I63x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru4x4*complexconjugate(Ru4x4)',
                   texname = '\\text{I63x44}')

I64x11 = Parameter(name = 'I64x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd1x6)*complexconjugate(td3x3)',
                   texname = '\\text{I64x11}')

I64x12 = Parameter(name = 'I64x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd1x6)*complexconjugate(td3x3)',
                   texname = '\\text{I64x12}')

I64x21 = Parameter(name = 'I64x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd2x6)*complexconjugate(td3x3)',
                   texname = '\\text{I64x21}')

I64x22 = Parameter(name = 'I64x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd2x6)*complexconjugate(td3x3)',
                   texname = '\\text{I64x22}')

I65x11 = Parameter(name = 'I65x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*td3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I65x11}')

I65x12 = Parameter(name = 'I65x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*td3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I65x12}')

I65x21 = Parameter(name = 'I65x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*td3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I65x21}')

I65x22 = Parameter(name = 'I65x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*td3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I65x22}')

I66x11 = Parameter(name = 'I66x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I66x11}')

I66x12 = Parameter(name = 'I66x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I66x12}')

I66x21 = Parameter(name = 'I66x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I66x21}')

I66x22 = Parameter(name = 'I66x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I66x22}')

I67x11 = Parameter(name = 'I67x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I67x11}')

I67x12 = Parameter(name = 'I67x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I67x12}')

I67x21 = Parameter(name = 'I67x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I67x21}')

I67x22 = Parameter(name = 'I67x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I67x22}')

I68x11 = Parameter(name = 'I68x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl1x6)*complexconjugate(te3x3)',
                   texname = '\\text{I68x11}')

I68x14 = Parameter(name = 'I68x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl1x6)*complexconjugate(te3x3)',
                   texname = '\\text{I68x14}')

I68x41 = Parameter(name = 'I68x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl4x6)*complexconjugate(te3x3)',
                   texname = '\\text{I68x41}')

I68x44 = Parameter(name = 'I68x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl4x6)*complexconjugate(te3x3)',
                   texname = '\\text{I68x44}')

I69x11 = Parameter(name = 'I69x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*te3x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I69x11}')

I69x14 = Parameter(name = 'I69x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*te3x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I69x14}')

I69x41 = Parameter(name = 'I69x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*te3x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I69x41}')

I69x44 = Parameter(name = 'I69x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*te3x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I69x44}')

I7x15 = Parameter(name = 'I7x15',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd5x1',
                  texname = '\\text{I7x15}')

I7x26 = Parameter(name = 'I7x26',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd6x2',
                  texname = '\\text{I7x26}')

I7x31 = Parameter(name = 'I7x31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x3',
                  texname = '\\text{I7x31}')

I7x32 = Parameter(name = 'I7x32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x3',
                  texname = '\\text{I7x32}')

I70x11 = Parameter(name = 'I70x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl1x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I70x11}')

I70x14 = Parameter(name = 'I70x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl1x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I70x14}')

I70x41 = Parameter(name = 'I70x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl4x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I70x41}')

I70x44 = Parameter(name = 'I70x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl4x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I70x44}')

I71x11 = Parameter(name = 'I71x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*ye3x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I71x11}')

I71x14 = Parameter(name = 'I71x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*ye3x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I71x14}')

I71x41 = Parameter(name = 'I71x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*ye3x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I71x41}')

I71x44 = Parameter(name = 'I71x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*ye3x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I71x44}')

I72x11 = Parameter(name = 'I72x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I72x11}')

I72x12 = Parameter(name = 'I72x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I72x12}')

I72x21 = Parameter(name = 'I72x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I72x21}')

I72x22 = Parameter(name = 'I72x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I72x22}')

I73x11 = Parameter(name = 'I73x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru1x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I73x11}')

I73x12 = Parameter(name = 'I73x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru1x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I73x12}')

I73x21 = Parameter(name = 'I73x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru2x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I73x21}')

I73x22 = Parameter(name = 'I73x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru2x6)*complexconjugate(tu3x3)',
                   texname = '\\text{I73x22}')

I74x11 = Parameter(name = 'I74x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*tu3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I74x11}')

I74x12 = Parameter(name = 'I74x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*tu3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I74x12}')

I74x21 = Parameter(name = 'I74x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*tu3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I74x21}')

I74x22 = Parameter(name = 'I74x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*tu3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I74x22}')

I75x11 = Parameter(name = 'I75x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I75x11}')

I75x12 = Parameter(name = 'I75x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I75x12}')

I75x21 = Parameter(name = 'I75x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I75x21}')

I75x22 = Parameter(name = 'I75x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I75x22}')

I76x11 = Parameter(name = 'I76x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*yd3x3*complexconjugate(Rd1x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I76x11}')

I76x12 = Parameter(name = 'I76x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*yd3x3*complexconjugate(Rd1x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I76x12}')

I76x21 = Parameter(name = 'I76x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*yd3x3*complexconjugate(Rd2x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I76x21}')

I76x22 = Parameter(name = 'I76x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*yd3x3*complexconjugate(Rd2x3)*complexconjugate(yd3x3)',
                   texname = '\\text{I76x22}')

I77x11 = Parameter(name = 'I77x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I77x11}')

I77x12 = Parameter(name = 'I77x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I77x12}')

I77x21 = Parameter(name = 'I77x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x6*yd3x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I77x21}')

I77x22 = Parameter(name = 'I77x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x6*yd3x3*complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I77x22}')

I78x11 = Parameter(name = 'I78x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*ye3x3*complexconjugate(Rl1x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I78x11}')

I78x14 = Parameter(name = 'I78x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*ye3x3*complexconjugate(Rl1x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I78x14}')

I78x41 = Parameter(name = 'I78x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*ye3x3*complexconjugate(Rl4x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I78x41}')

I78x44 = Parameter(name = 'I78x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*ye3x3*complexconjugate(Rl4x3)*complexconjugate(ye3x3)',
                   texname = '\\text{I78x44}')

I79x11 = Parameter(name = 'I79x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*ye3x3*complexconjugate(Rl1x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I79x11}')

I79x14 = Parameter(name = 'I79x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*ye3x3*complexconjugate(Rl1x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I79x14}')

I79x41 = Parameter(name = 'I79x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x6*ye3x3*complexconjugate(Rl4x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I79x41}')

I79x44 = Parameter(name = 'I79x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x6*ye3x3*complexconjugate(Rl4x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I79x44}')

I8x31 = Parameter(name = 'I8x31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x3*complexconjugate(yu3x3)',
                  texname = '\\text{I8x31}')

I8x32 = Parameter(name = 'I8x32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x3*complexconjugate(yu3x3)',
                  texname = '\\text{I8x32}')

I80x11 = Parameter(name = 'I80x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*yu3x3*complexconjugate(Ru1x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x11}')

I80x12 = Parameter(name = 'I80x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*yu3x3*complexconjugate(Ru1x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x12}')

I80x21 = Parameter(name = 'I80x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*yu3x3*complexconjugate(Ru2x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x21}')

I80x22 = Parameter(name = 'I80x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*yu3x3*complexconjugate(Ru2x3)*complexconjugate(yu3x3)',
                   texname = '\\text{I80x22}')

I81x11 = Parameter(name = 'I81x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x11}')

I81x12 = Parameter(name = 'I81x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x12}')

I81x21 = Parameter(name = 'I81x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x6*yu3x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x21}')

I81x22 = Parameter(name = 'I81x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x6*yu3x3*complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I81x22}')

I82x15 = Parameter(name = 'I82x15',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rd5x1)',
                   texname = '\\text{I82x15}')

I82x26 = Parameter(name = 'I82x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rd6x2)',
                   texname = '\\text{I82x26}')

I82x31 = Parameter(name = 'I82x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rd1x3)',
                   texname = '\\text{I82x31}')

I82x32 = Parameter(name = 'I82x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rd2x3)',
                   texname = '\\text{I82x32}')

I83x31 = Parameter(name = 'I83x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rd1x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I83x31}')

I83x32 = Parameter(name = 'I83x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rd2x6)*complexconjugate(yd3x3)',
                   texname = '\\text{I83x32}')

I84x31 = Parameter(name = 'I84x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I84x31}')

I84x32 = Parameter(name = 'I84x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yu3x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I84x32}')

I85x15 = Parameter(name = 'I85x15',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl5x1)',
                   texname = '\\text{I85x15}')

I85x26 = Parameter(name = 'I85x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl6x2)',
                   texname = '\\text{I85x26}')

I85x31 = Parameter(name = 'I85x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl1x3)',
                   texname = '\\text{I85x31}')

I85x34 = Parameter(name = 'I85x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl4x3)',
                   texname = '\\text{I85x34}')

I86x31 = Parameter(name = 'I86x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl1x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I86x31}')

I86x34 = Parameter(name = 'I86x34',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rl4x6)*complexconjugate(ye3x3)',
                   texname = '\\text{I86x34}')

I87x13 = Parameter(name = 'I87x13',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rn3x1)',
                   texname = '\\text{I87x13}')

I87x22 = Parameter(name = 'I87x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rn2x2)',
                   texname = '\\text{I87x22}')

I87x31 = Parameter(name = 'I87x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Rn1x3)',
                   texname = '\\text{I87x31}')

I88x31 = Parameter(name = 'I88x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I88x31}')

I89x15 = Parameter(name = 'I89x15',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru5x1)',
                   texname = '\\text{I89x15}')

I89x26 = Parameter(name = 'I89x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru6x2)',
                   texname = '\\text{I89x26}')

I89x31 = Parameter(name = 'I89x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru1x3)',
                   texname = '\\text{I89x31}')

I89x32 = Parameter(name = 'I89x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru2x3)',
                   texname = '\\text{I89x32}')

I9x31 = Parameter(name = 'I9x31',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd1x6*yd3x3',
                  texname = '\\text{I9x31}')

I9x32 = Parameter(name = 'I9x32',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd2x6*yd3x3',
                  texname = '\\text{I9x32}')

I90x31 = Parameter(name = 'I90x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru1x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I90x31}')

I90x32 = Parameter(name = 'I90x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'complexconjugate(Ru2x6)*complexconjugate(yu3x3)',
                   texname = '\\text{I90x32}')

I91x31 = Parameter(name = 'I91x31',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yd3x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I91x31}')

I91x32 = Parameter(name = 'I91x32',
                   nature = 'internal',
                   type = 'complex',
                   value = 'yd3x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I91x32}')

I92x11 = Parameter(name = 'I92x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I92x11}')

I92x12 = Parameter(name = 'I92x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I92x12}')

I92x21 = Parameter(name = 'I92x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I92x21}')

I92x22 = Parameter(name = 'I92x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I92x22}')

I92x55 = Parameter(name = 'I92x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x1*complexconjugate(Rd5x1)',
                   texname = '\\text{I92x55}')

I92x66 = Parameter(name = 'I92x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x2*complexconjugate(Rd6x2)',
                   texname = '\\text{I92x66}')

I93x11 = Parameter(name = 'I93x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I93x11}')

I93x14 = Parameter(name = 'I93x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn1x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I93x14}')

I93x26 = Parameter(name = 'I93x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn2x2*complexconjugate(Rl6x2)',
                   texname = '\\text{I93x26}')

I93x35 = Parameter(name = 'I93x35',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rn3x1*complexconjugate(Rl5x1)',
                   texname = '\\text{I93x35}')

I94x11 = Parameter(name = 'I94x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I94x11}')

I94x12 = Parameter(name = 'I94x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I94x12}')

I94x21 = Parameter(name = 'I94x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I94x21}')

I94x22 = Parameter(name = 'I94x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I94x22}')

I94x55 = Parameter(name = 'I94x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd5x1*complexconjugate(Ru5x1)',
                   texname = '\\text{I94x55}')

I94x66 = Parameter(name = 'I94x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x2*complexconjugate(Ru6x2)',
                   texname = '\\text{I94x66}')

I95x11 = Parameter(name = 'I95x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I95x11}')

I95x14 = Parameter(name = 'I95x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rn1x3)',
                   texname = '\\text{I95x14}')

I95x26 = Parameter(name = 'I95x26',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x2*complexconjugate(Rn2x2)',
                   texname = '\\text{I95x26}')

I95x35 = Parameter(name = 'I95x35',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x1*complexconjugate(Rn3x1)',
                   texname = '\\text{I95x35}')

I96x11 = Parameter(name = 'I96x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I96x11}')

I96x12 = Parameter(name = 'I96x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd1x3)',
                   texname = '\\text{I96x12}')

I96x21 = Parameter(name = 'I96x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd1x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I96x21}')

I96x22 = Parameter(name = 'I96x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd2x3*complexconjugate(Rd2x3)',
                   texname = '\\text{I96x22}')

I96x55 = Parameter(name = 'I96x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd5x1*complexconjugate(Rd5x1)',
                   texname = '\\text{I96x55}')

I96x66 = Parameter(name = 'I96x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd6x2*complexconjugate(Rd6x2)',
                   texname = '\\text{I96x66}')

I97x11 = Parameter(name = 'I97x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I97x11}')

I97x14 = Parameter(name = 'I97x14',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl1x3)',
                   texname = '\\text{I97x14}')

I97x41 = Parameter(name = 'I97x41',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl1x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I97x41}')

I97x44 = Parameter(name = 'I97x44',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl4x3*complexconjugate(Rl4x3)',
                   texname = '\\text{I97x44}')

I97x55 = Parameter(name = 'I97x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl5x1*complexconjugate(Rl5x1)',
                   texname = '\\text{I97x55}')

I97x66 = Parameter(name = 'I97x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl6x2*complexconjugate(Rl6x2)',
                   texname = '\\text{I97x66}')

I98x11 = Parameter(name = 'I98x11',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I98x11}')

I98x12 = Parameter(name = 'I98x12',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru1x3)',
                   texname = '\\text{I98x12}')

I98x21 = Parameter(name = 'I98x21',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru1x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I98x21}')

I98x22 = Parameter(name = 'I98x22',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru2x3*complexconjugate(Ru2x3)',
                   texname = '\\text{I98x22}')

I98x55 = Parameter(name = 'I98x55',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru5x1*complexconjugate(Ru5x1)',
                   texname = '\\text{I98x55}')

I98x66 = Parameter(name = 'I98x66',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru6x2*complexconjugate(Ru6x2)',
                   texname = '\\text{I98x66}')

I99x33 = Parameter(name = 'I99x33',
                   nature = 'internal',
                   type = 'complex',
                   value = 'ye3x3',
                   texname = '\\text{I99x33}')

