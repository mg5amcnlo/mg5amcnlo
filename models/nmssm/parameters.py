# This file was automatically created by FeynRules $Revision: 573 $
# Mathematica version: 7.0 for Linux x86 (64-bit) (February 18, 2009)
# Date: Tue 12 Apr 2011 09:30:14



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
RRd13 = Parameter(name = 'RRd13',
                  nature = 'external',
                  type = 'real',
                  value = 0.990890455,
                  texname = '\\text{RRd13}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 1, 3 ])

RRd16 = Parameter(name = 'RRd16',
                  nature = 'external',
                  type = 'real',
                  value = 0.134670361,
                  texname = '\\text{RRd16}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 1, 6 ])

RRd23 = Parameter(name = 'RRd23',
                  nature = 'external',
                  type = 'real',
                  value = -0.134670361,
                  texname = '\\text{RRd23}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 2, 3 ])

RRd26 = Parameter(name = 'RRd26',
                  nature = 'external',
                  type = 'real',
                  value = 0.990890455,
                  texname = '\\text{RRd26}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 2, 6 ])

RRd35 = Parameter(name = 'RRd35',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd35}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 3, 5 ])

RRd44 = Parameter(name = 'RRd44',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd44}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 4, 4 ])

RRd51 = Parameter(name = 'RRd51',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd51}',
                  lhablock = 'DSQMIX',
                  lhacode = [ 5, 1 ])

RRd62 = Parameter(name = 'RRd62',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRd62}',
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

RmD211 = Parameter(name = 'RmD211',
                   nature = 'external',
                   type = 'real',
                   value = 963545.439,
                   texname = '\\text{RmD211}',
                   lhablock = 'MSD2',
                   lhacode = [ 1, 1 ])

RmD222 = Parameter(name = 'RmD222',
                   nature = 'external',
                   type = 'real',
                   value = 963545.439,
                   texname = '\\text{RmD222}',
                   lhablock = 'MSD2',
                   lhacode = [ 2, 2 ])

RmD233 = Parameter(name = 'RmD233',
                   nature = 'external',
                   type = 'real',
                   value = 933451.834,
                   texname = '\\text{RmD233}',
                   lhablock = 'MSD2',
                   lhacode = [ 3, 3 ])

RmE211 = Parameter(name = 'RmE211',
                   nature = 'external',
                   type = 'real',
                   value = 66311.1508,
                   texname = '\\text{RmE211}',
                   lhablock = 'MSE2',
                   lhacode = [ 1, 1 ])

RmE222 = Parameter(name = 'RmE222',
                   nature = 'external',
                   type = 'real',
                   value = 66311.1508,
                   texname = '\\text{RmE222}',
                   lhablock = 'MSE2',
                   lhacode = [ 2, 2 ])

RmE233 = Parameter(name = 'RmE233',
                   nature = 'external',
                   type = 'real',
                   value = 48897.3735,
                   texname = '\\text{RmE233}',
                   lhablock = 'MSE2',
                   lhacode = [ 3, 3 ])

RmL211 = Parameter(name = 'RmL211',
                   nature = 'external',
                   type = 'real',
                   value = 142415.235,
                   texname = '\\text{RmL211}',
                   lhablock = 'MSL2',
                   lhacode = [ 1, 1 ])

RmL222 = Parameter(name = 'RmL222',
                   nature = 'external',
                   type = 'real',
                   value = 142415.235,
                   texname = '\\text{RmL222}',
                   lhablock = 'MSL2',
                   lhacode = [ 2, 2 ])

RmL233 = Parameter(name = 'RmL233',
                   nature = 'external',
                   type = 'real',
                   value = 133786.42,
                   texname = '\\text{RmL233}',
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

mHu2 = Parameter(name = 'mHu2',
                 nature = 'external',
                 type = 'real',
                 value = 89988.5262,
                 texname = 'm_{H_u}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 21 ])

mHd2 = Parameter(name = 'mHd2',
                 nature = 'external',
                 type = 'real',
                 value = -908071.077,
                 texname = 'm_{H_d}^2',
                 lhablock = 'MSOFT',
                 lhacode = [ 22 ])

RmQ211 = Parameter(name = 'RmQ211',
                   nature = 'external',
                   type = 'real',
                   value = 1.04878444e6,
                   texname = '\\text{RmQ211}',
                   lhablock = 'MSQ2',
                   lhacode = [ 1, 1 ])

RmQ222 = Parameter(name = 'RmQ222',
                   nature = 'external',
                   type = 'real',
                   value = 1.04878444e6,
                   texname = '\\text{RmQ222}',
                   lhablock = 'MSQ2',
                   lhacode = [ 2, 2 ])

RmQ233 = Parameter(name = 'RmQ233',
                   nature = 'external',
                   type = 'real',
                   value = 715579.339,
                   texname = '\\text{RmQ233}',
                   lhablock = 'MSQ2',
                   lhacode = [ 3, 3 ])

RmU211 = Parameter(name = 'RmU211',
                   nature = 'external',
                   type = 'real',
                   value = 972428.308,
                   texname = '\\text{RmU211}',
                   lhablock = 'MSU2',
                   lhacode = [ 1, 1 ])

RmU222 = Parameter(name = 'RmU222',
                   nature = 'external',
                   type = 'real',
                   value = 972428.308,
                   texname = '\\text{RmU222}',
                   lhablock = 'MSU2',
                   lhacode = [ 2, 2 ])

RmU233 = Parameter(name = 'RmU233',
                   nature = 'external',
                   type = 'real',
                   value = 319484.921,
                   texname = '\\text{RmU233}',
                   lhablock = 'MSU2',
                   lhacode = [ 3, 3 ])

UP11 = Parameter(name = 'UP11',
                 nature = 'external',
                 type = 'real',
                 value = 0.0501258919,
                 texname = '\\text{UP11}',
                 lhablock = 'NMAMIX',
                 lhacode = [ 1, 1 ])

UP12 = Parameter(name = 'UP12',
                 nature = 'external',
                 type = 'real',
                 value = 0.00501258919,
                 texname = '\\text{UP12}',
                 lhablock = 'NMAMIX',
                 lhacode = [ 1, 2 ])

UP13 = Parameter(name = 'UP13',
                 nature = 'external',
                 type = 'real',
                 value = 0.998730328,
                 texname = '\\text{UP13}',
                 lhablock = 'NMAMIX',
                 lhacode = [ 1, 3 ])

UP21 = Parameter(name = 'UP21',
                 nature = 'external',
                 type = 'real',
                 value = 0.99377382,
                 texname = '\\text{UP21}',
                 lhablock = 'NMAMIX',
                 lhacode = [ 2, 1 ])

UP22 = Parameter(name = 'UP22',
                 nature = 'external',
                 type = 'real',
                 value = 0.099377382,
                 texname = '\\text{UP22}',
                 lhablock = 'NMAMIX',
                 lhacode = [ 2, 2 ])

UP23 = Parameter(name = 'UP23',
                 nature = 'external',
                 type = 'real',
                 value = -0.0503758979,
                 texname = '\\text{UP23}',
                 lhablock = 'NMAMIX',
                 lhacode = [ 2, 3 ])

US11 = Parameter(name = 'US11',
                 nature = 'external',
                 type = 'real',
                 value = 0.101230631,
                 texname = '\\text{US11}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 1, 1 ])

US12 = Parameter(name = 'US12',
                 nature = 'external',
                 type = 'real',
                 value = 0.994841811,
                 texname = '\\text{US12}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 1, 2 ])

US13 = Parameter(name = 'US13',
                 nature = 'external',
                 type = 'real',
                 value = -0.00649079704,
                 texname = '\\text{US13}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 1, 3 ])

US21 = Parameter(name = 'US21',
                 nature = 'external',
                 type = 'real',
                 value = 0.994850372,
                 texname = '\\text{US21}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 2, 1 ])

US22 = Parameter(name = 'US22',
                 nature = 'external',
                 type = 'real',
                 value = -0.10119434,
                 texname = '\\text{US22}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 2, 2 ])

US23 = Parameter(name = 'US23',
                 nature = 'external',
                 type = 'real',
                 value = 0.00569588834,
                 texname = '\\text{US23}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 2, 3 ])

US31 = Parameter(name = 'US31',
                 nature = 'external',
                 type = 'real',
                 value = -0.00500967595,
                 texname = '\\text{US31}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 3, 1 ])

US32 = Parameter(name = 'US32',
                 nature = 'external',
                 type = 'real',
                 value = 0.00703397022,
                 texname = '\\text{US32}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 3, 2 ])

US33 = Parameter(name = 'US33',
                 nature = 'external',
                 type = 'real',
                 value = 0.999962713,
                 texname = '\\text{US33}',
                 lhablock = 'NMHMIX',
                 lhacode = [ 3, 3 ])

RNN11 = Parameter(name = 'RNN11',
                  nature = 'external',
                  type = 'real',
                  value = 0.998684518,
                  texname = '\\text{RNN11}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 1, 1 ])

RNN12 = Parameter(name = 'RNN12',
                  nature = 'external',
                  type = 'real',
                  value = -0.00814943871,
                  texname = '\\text{RNN12}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 1, 2 ])

RNN13 = Parameter(name = 'RNN13',
                  nature = 'external',
                  type = 'real',
                  value = 0.0483530815,
                  texname = '\\text{RNN13}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 1, 3 ])

RNN14 = Parameter(name = 'RNN14',
                  nature = 'external',
                  type = 'real',
                  value = -0.0149871707,
                  texname = '\\text{RNN14}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 1, 4 ])

RNN15 = Parameter(name = 'RNN15',
                  nature = 'external',
                  type = 'real',
                  value = 0.000430389009,
                  texname = '\\text{RNN15}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 1, 5 ])

RNN21 = Parameter(name = 'RNN21',
                  nature = 'external',
                  type = 'real',
                  value = 0.0138621789,
                  texname = '\\text{RNN21}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 2, 1 ])

RNN22 = Parameter(name = 'RNN22',
                  nature = 'external',
                  type = 'real',
                  value = 0.993268723,
                  texname = '\\text{RNN22}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 2, 2 ])

RNN23 = Parameter(name = 'RNN23',
                  nature = 'external',
                  type = 'real',
                  value = -0.103118961,
                  texname = '\\text{RNN23}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 2, 3 ])

RNN24 = Parameter(name = 'RNN24',
                  nature = 'external',
                  type = 'real',
                  value = 0.05089756,
                  texname = '\\text{RNN24}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 2, 4 ])

RNN25 = Parameter(name = 'RNN25',
                  nature = 'external',
                  type = 'real',
                  value = -0.00100117257,
                  texname = '\\text{RNN25}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 2, 5 ])

RNN31 = Parameter(name = 'RNN31',
                  nature = 'external',
                  type = 'real',
                  value = -0.0232278855,
                  texname = '\\text{RNN31}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 3, 1 ])

RNN32 = Parameter(name = 'RNN32',
                  nature = 'external',
                  type = 'real',
                  value = 0.037295208,
                  texname = '\\text{RNN32}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 3, 2 ])

RNN33 = Parameter(name = 'RNN33',
                  nature = 'external',
                  type = 'real',
                  value = 0.705297681,
                  texname = '\\text{RNN33}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 3, 3 ])

RNN34 = Parameter(name = 'RNN34',
                  nature = 'external',
                  type = 'real',
                  value = 0.707534724,
                  texname = '\\text{RNN34}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 3, 4 ])

RNN35 = Parameter(name = 'RNN35',
                  nature = 'external',
                  type = 'real',
                  value = 0.00439627968,
                  texname = '\\text{RNN35}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 3, 5 ])

RNN41 = Parameter(name = 'RNN41',
                  nature = 'external',
                  type = 'real',
                  value = 0.0435606237,
                  texname = '\\text{RNN41}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 4, 1 ])

RNN42 = Parameter(name = 'RNN42',
                  nature = 'external',
                  type = 'real',
                  value = -0.109361086,
                  texname = '\\text{RNN42}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 4, 2 ])

RNN43 = Parameter(name = 'RNN43',
                  nature = 'external',
                  type = 'real',
                  value = -0.69963098,
                  texname = '\\text{RNN43}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 4, 3 ])

RNN44 = Parameter(name = 'RNN44',
                  nature = 'external',
                  type = 'real',
                  value = 0.704673803,
                  texname = '\\text{RNN44}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 4, 4 ])

RNN45 = Parameter(name = 'RNN45',
                  nature = 'external',
                  type = 'real',
                  value = -0.00969268004,
                  texname = '\\text{RNN45}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 4, 5 ])

RNN51 = Parameter(name = 'RNN51',
                  nature = 'external',
                  type = 'real',
                  value = 0.000108397267,
                  texname = '\\text{RNN51}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 5, 1 ])

RNN52 = Parameter(name = 'RNN52',
                  nature = 'external',
                  type = 'real',
                  value = -0.000226034288,
                  texname = '\\text{RNN52}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 5, 2 ])

RNN53 = Parameter(name = 'RNN53',
                  nature = 'external',
                  type = 'real',
                  value = -0.0100066083,
                  texname = '\\text{RNN53}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 5, 3 ])

RNN54 = Parameter(name = 'RNN54',
                  nature = 'external',
                  type = 'real',
                  value = 0.00377728091,
                  texname = '\\text{RNN54}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 5, 4 ])

RNN55 = Parameter(name = 'RNN55',
                  nature = 'external',
                  type = 'real',
                  value = 0.999942767,
                  texname = '\\text{RNN55}',
                  lhablock = 'NMNMIX',
                  lhacode = [ 5, 5 ])

NMl = Parameter(name = 'NMl',
                nature = 'external',
                type = 'real',
                value = 0.1,
                texname = '\\lambda ',
                lhablock = 'NMSSMRUN',
                lhacode = [ 1 ])

NMk = Parameter(name = 'NMk',
                nature = 'external',
                type = 'real',
                value = 0.108910706,
                texname = '\\kappa ',
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

RRl13 = Parameter(name = 'RRl13',
                  nature = 'external',
                  type = 'real',
                  value = 0.220980319,
                  texname = '\\text{RRl13}',
                  lhablock = 'SELMIX',
                  lhacode = [ 1, 3 ])

RRl16 = Parameter(name = 'RRl16',
                  nature = 'external',
                  type = 'real',
                  value = 0.975278267,
                  texname = '\\text{RRl16}',
                  lhablock = 'SELMIX',
                  lhacode = [ 1, 6 ])

RRl24 = Parameter(name = 'RRl24',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl24}',
                  lhablock = 'SELMIX',
                  lhacode = [ 2, 4 ])

RRl35 = Parameter(name = 'RRl35',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl35}',
                  lhablock = 'SELMIX',
                  lhacode = [ 3, 5 ])

RRl43 = Parameter(name = 'RRl43',
                  nature = 'external',
                  type = 'real',
                  value = -0.975278267,
                  texname = '\\text{RRl43}',
                  lhablock = 'SELMIX',
                  lhacode = [ 4, 3 ])

RRl46 = Parameter(name = 'RRl46',
                  nature = 'external',
                  type = 'real',
                  value = 0.220980319,
                  texname = '\\text{RRl46}',
                  lhablock = 'SELMIX',
                  lhacode = [ 4, 6 ])

RRl51 = Parameter(name = 'RRl51',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl51}',
                  lhablock = 'SELMIX',
                  lhacode = [ 5, 1 ])

RRl62 = Parameter(name = 'RRl62',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRl62}',
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

RRn13 = Parameter(name = 'RRn13',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRn13}',
                  lhablock = 'SNUMIX',
                  lhacode = [ 1, 3 ])

RRn22 = Parameter(name = 'RRn22',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRn22}',
                  lhablock = 'SNUMIX',
                  lhacode = [ 2, 2 ])

RRn31 = Parameter(name = 'RRn31',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRn31}',
                  lhablock = 'SNUMIX',
                  lhacode = [ 3, 1 ])

Rtd33 = Parameter(name = 'Rtd33',
                  nature = 'external',
                  type = 'real',
                  value = -342.310014,
                  texname = '\\text{Rtd33}',
                  lhablock = 'TD',
                  lhacode = [ 3, 3 ])

Rte33 = Parameter(name = 'Rte33',
                  nature = 'external',
                  type = 'real',
                  value = -177.121653,
                  texname = '\\text{Rte33}',
                  lhablock = 'TE',
                  lhacode = [ 3, 3 ])

Rtu33 = Parameter(name = 'Rtu33',
                  nature = 'external',
                  type = 'real',
                  value = -1213.64864,
                  texname = '\\text{Rtu33}',
                  lhablock = 'TU',
                  lhacode = [ 3, 3 ])

RUU11 = Parameter(name = 'RUU11',
                  nature = 'external',
                  type = 'real',
                  value = 0.989230572,
                  texname = '\\text{RUU11}',
                  lhablock = 'UMIX',
                  lhacode = [ 1, 1 ])

RUU12 = Parameter(name = 'RUU12',
                  nature = 'external',
                  type = 'real',
                  value = -0.146365554,
                  texname = '\\text{RUU12}',
                  lhablock = 'UMIX',
                  lhacode = [ 1, 2 ])

RUU21 = Parameter(name = 'RUU21',
                  nature = 'external',
                  type = 'real',
                  value = 0.146365554,
                  texname = '\\text{RUU21}',
                  lhablock = 'UMIX',
                  lhacode = [ 2, 1 ])

RUU22 = Parameter(name = 'RUU22',
                  nature = 'external',
                  type = 'real',
                  value = 0.989230572,
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

RRu13 = Parameter(name = 'RRu13',
                  nature = 'external',
                  type = 'real',
                  value = 0.405775656,
                  texname = '\\text{RRu13}',
                  lhablock = 'USQMIX',
                  lhacode = [ 1, 3 ])

RRu16 = Parameter(name = 'RRu16',
                  nature = 'external',
                  type = 'real',
                  value = 0.913972711,
                  texname = '\\text{RRu16}',
                  lhablock = 'USQMIX',
                  lhacode = [ 1, 6 ])

RRu23 = Parameter(name = 'RRu23',
                  nature = 'external',
                  type = 'real',
                  value = -0.913972711,
                  texname = '\\text{RRu23}',
                  lhablock = 'USQMIX',
                  lhacode = [ 2, 3 ])

RRu26 = Parameter(name = 'RRu26',
                  nature = 'external',
                  type = 'real',
                  value = 0.405775656,
                  texname = '\\text{RRu26}',
                  lhablock = 'USQMIX',
                  lhacode = [ 2, 6 ])

RRu35 = Parameter(name = 'RRu35',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu35}',
                  lhablock = 'USQMIX',
                  lhacode = [ 3, 5 ])

RRu44 = Parameter(name = 'RRu44',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu44}',
                  lhablock = 'USQMIX',
                  lhacode = [ 4, 4 ])

RRu51 = Parameter(name = 'RRu51',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu51}',
                  lhablock = 'USQMIX',
                  lhacode = [ 5, 1 ])

RRu62 = Parameter(name = 'RRu62',
                  nature = 'external',
                  type = 'real',
                  value = 1.,
                  texname = '\\text{RRu62}',
                  lhablock = 'USQMIX',
                  lhacode = [ 6, 2 ])

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
                  value = 0.997382381,
                  texname = '\\text{RVV11}',
                  lhablock = 'VMIX',
                  lhacode = [ 1, 1 ])

RVV12 = Parameter(name = 'RVV12',
                  nature = 'external',
                  type = 'real',
                  value = -0.0723075752,
                  texname = '\\text{RVV12}',
                  lhablock = 'VMIX',
                  lhacode = [ 1, 2 ])

RVV21 = Parameter(name = 'RVV21',
                  nature = 'external',
                  type = 'real',
                  value = 0.0723075752,
                  texname = '\\text{RVV21}',
                  lhablock = 'VMIX',
                  lhacode = [ 2, 1 ])

RVV22 = Parameter(name = 'RVV22',
                  nature = 'external',
                  type = 'real',
                  value = 0.997382381,
                  texname = '\\text{RVV22}',
                  lhablock = 'VMIX',
                  lhacode = [ 2, 2 ])

Ryd33 = Parameter(name = 'Ryd33',
                  nature = 'external',
                  type = 'real',
                  value = 0.131064265,
                  texname = '\\text{Ryd33}',
                  lhablock = 'YD',
                  lhacode = [ 3, 3 ])

Rye33 = Parameter(name = 'Rye33',
                  nature = 'external',
                  type = 'real',
                  value = 0.100464794,
                  texname = '\\text{Rye33}',
                  lhablock = 'YE',
                  lhacode = [ 3, 3 ])

Ryu33 = Parameter(name = 'Ryu33',
                  nature = 'external',
                  type = 'real',
                  value = 0.847827829,
                  texname = '\\text{Ryu33}',
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

Mve = Parameter(name = 'Mve',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = '\\text{Mve}',
                lhablock = 'MASS',
                lhacode = [ 12 ])

Mvm = Parameter(name = 'Mvm',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = '\\text{Mvm}',
                lhablock = 'MASS',
                lhacode = [ 14 ])

Mvt = Parameter(name = 'Mvt',
                nature = 'external',
                type = 'real',
                value = 0.,
                texname = '\\text{Mvt}',
                lhablock = 'MASS',
                lhacode = [ 16 ])

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = '\\text{Me}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

Mm = Parameter(name = 'Mm',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = '\\text{Mm}',
               lhablock = 'MASS',
               lhacode = [ 13 ])

Mta = Parameter(name = 'Mta',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{Mta}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 4 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 171.4,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 1 ])

MS = Parameter(name = 'MS',
               nature = 'external',
               type = 'real',
               value = 0.,
               texname = '\\text{MS}',
               lhablock = 'MASS',
               lhacode = [ 3 ])

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
               value = 2.41143316e+00,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.00282196e+00 ,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

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
                  value = 0.,
                  texname = '\\text{Wneu2}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000023 ])

Wneu3 = Parameter(name = 'Wneu3',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{Wneu3}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000025 ])

Wneu4 = Parameter(name = 'Wneu4',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{Wneu4}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000035 ])

Wneu5 = Parameter(name = 'Wneu5',
                  nature = 'external',
                  type = 'real',
                  value = 0.,
                  texname = '\\text{Wneu5}',
                  lhablock = 'DECAY',
                  lhacode = [ 1000045 ])

Wch1 = Parameter(name = 'Wch1',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wch1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000024 ])

Wch2 = Parameter(name = 'Wch2',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wch2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000037 ])

Wgo = Parameter(name = 'Wgo',
                nature = 'external',
                type = 'real',
                value = 0.,
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
                 value = 0.,
                 texname = '\\text{Wsn1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000012 ])

Wsn2 = Parameter(name = 'Wsn2',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsn2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000014 ])

Wsn3 = Parameter(name = 'Wsn3',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsn3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000016 ])

Wsl1 = Parameter(name = 'Wsl1',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsl1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000011 ])

Wsl2 = Parameter(name = 'Wsl2',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsl2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000013 ])

Wsl3 = Parameter(name = 'Wsl3',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsl3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000015 ])

Wsl4 = Parameter(name = 'Wsl4',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsl4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000011 ])

Wsl5 = Parameter(name = 'Wsl5',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsl5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000013 ])

Wsl6 = Parameter(name = 'Wsl6',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsl6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000015 ])

Wsu1 = Parameter(name = 'Wsu1',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsu1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000002 ])

Wsu2 = Parameter(name = 'Wsu2',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsu2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000004 ])

Wsu3 = Parameter(name = 'Wsu3',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsu3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000006 ])

Wsu4 = Parameter(name = 'Wsu4',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsu4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000002 ])

Wsu5 = Parameter(name = 'Wsu5',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsu5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000004 ])

Wsu6 = Parameter(name = 'Wsu6',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsu6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000006 ])

Wsd1 = Parameter(name = 'Wsd1',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsd1}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000001 ])

Wsd2 = Parameter(name = 'Wsd2',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsd2}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000003 ])

Wsd3 = Parameter(name = 'Wsd3',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsd3}',
                 lhablock = 'DECAY',
                 lhacode = [ 1000005 ])

Wsd4 = Parameter(name = 'Wsd4',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsd4}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000001 ])

Wsd5 = Parameter(name = 'Wsd5',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsd5}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000003 ])

Wsd6 = Parameter(name = 'Wsd6',
                 nature = 'external',
                 type = 'real',
                 value = 0.,
                 texname = '\\text{Wsd6}',
                 lhablock = 'DECAY',
                 lhacode = [ 2000005 ])

beta = Parameter(name = 'beta',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.atan(tb)',
                 texname = '\\beta ')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'MW/MZ',
               texname = 'c_w')

mD211 = Parameter(name = 'mD211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmD211',
                  texname = '\\text{mD211}')

mD222 = Parameter(name = 'mD222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmD222',
                  texname = '\\text{mD222}')

mD233 = Parameter(name = 'mD233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmD233',
                  texname = '\\text{mD233}')

mE211 = Parameter(name = 'mE211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmE211',
                  texname = '\\text{mE211}')

mE222 = Parameter(name = 'mE222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmE222',
                  texname = '\\text{mE222}')

mE233 = Parameter(name = 'mE233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmE233',
                  texname = '\\text{mE233}')

mL211 = Parameter(name = 'mL211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmL211',
                  texname = '\\text{mL211}')

mL222 = Parameter(name = 'mL222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmL222',
                  texname = '\\text{mL222}')

mL233 = Parameter(name = 'mL233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmL233',
                  texname = '\\text{mL233}')

mQ211 = Parameter(name = 'mQ211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmQ211',
                  texname = '\\text{mQ211}')

mQ222 = Parameter(name = 'mQ222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmQ222',
                  texname = '\\text{mQ222}')

mQ233 = Parameter(name = 'mQ233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmQ233',
                  texname = '\\text{mQ233}')

mU211 = Parameter(name = 'mU211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmU211',
                  texname = '\\text{mU211}')

mU222 = Parameter(name = 'mU222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmU222',
                  texname = '\\text{mU222}')

mU233 = Parameter(name = 'mU233',
                  nature = 'internal',
                  type = 'complex',
                  value = 'RmU233',
                  texname = '\\text{mU233}')

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

NN15 = Parameter(name = 'NN15',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN15',
                 texname = '\\text{NN15}')

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

NN25 = Parameter(name = 'NN25',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN25',
                 texname = '\\text{NN25}')

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

NN35 = Parameter(name = 'NN35',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN35',
                 texname = '\\text{NN35}')

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

NN45 = Parameter(name = 'NN45',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN45',
                 texname = '\\text{NN45}')

NN51 = Parameter(name = 'NN51',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN51',
                 texname = '\\text{NN51}')

NN52 = Parameter(name = 'NN52',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN52',
                 texname = '\\text{NN52}')

NN53 = Parameter(name = 'NN53',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN53',
                 texname = '\\text{NN53}')

NN54 = Parameter(name = 'NN54',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN54',
                 texname = '\\text{NN54}')

NN55 = Parameter(name = 'NN55',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RNN55',
                 texname = '\\text{NN55}')

Rd13 = Parameter(name = 'Rd13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd13',
                 texname = '\\text{Rd13}')

Rd16 = Parameter(name = 'Rd16',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd16',
                 texname = '\\text{Rd16}')

Rd23 = Parameter(name = 'Rd23',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd23',
                 texname = '\\text{Rd23}')

Rd26 = Parameter(name = 'Rd26',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd26',
                 texname = '\\text{Rd26}')

Rd35 = Parameter(name = 'Rd35',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd35',
                 texname = '\\text{Rd35}')

Rd44 = Parameter(name = 'Rd44',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd44',
                 texname = '\\text{Rd44}')

Rd51 = Parameter(name = 'Rd51',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd51',
                 texname = '\\text{Rd51}')

Rd62 = Parameter(name = 'Rd62',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRd62',
                 texname = '\\text{Rd62}')

Rl13 = Parameter(name = 'Rl13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl13',
                 texname = '\\text{Rl13}')

Rl16 = Parameter(name = 'Rl16',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl16',
                 texname = '\\text{Rl16}')

Rl24 = Parameter(name = 'Rl24',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl24',
                 texname = '\\text{Rl24}')

Rl35 = Parameter(name = 'Rl35',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl35',
                 texname = '\\text{Rl35}')

Rl43 = Parameter(name = 'Rl43',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl43',
                 texname = '\\text{Rl43}')

Rl46 = Parameter(name = 'Rl46',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl46',
                 texname = '\\text{Rl46}')

Rl51 = Parameter(name = 'Rl51',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl51',
                 texname = '\\text{Rl51}')

Rl62 = Parameter(name = 'Rl62',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRl62',
                 texname = '\\text{Rl62}')

Rn13 = Parameter(name = 'Rn13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRn13',
                 texname = '\\text{Rn13}')

Rn22 = Parameter(name = 'Rn22',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRn22',
                 texname = '\\text{Rn22}')

Rn31 = Parameter(name = 'Rn31',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRn31',
                 texname = '\\text{Rn31}')

Ru13 = Parameter(name = 'Ru13',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu13',
                 texname = '\\text{Ru13}')

Ru16 = Parameter(name = 'Ru16',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu16',
                 texname = '\\text{Ru16}')

Ru23 = Parameter(name = 'Ru23',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu23',
                 texname = '\\text{Ru23}')

Ru26 = Parameter(name = 'Ru26',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu26',
                 texname = '\\text{Ru26}')

Ru35 = Parameter(name = 'Ru35',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu35',
                 texname = '\\text{Ru35}')

Ru44 = Parameter(name = 'Ru44',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu44',
                 texname = '\\text{Ru44}')

Ru51 = Parameter(name = 'Ru51',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu51',
                 texname = '\\text{Ru51}')

Ru62 = Parameter(name = 'Ru62',
                 nature = 'internal',
                 type = 'complex',
                 value = 'RRu62',
                 texname = '\\text{Ru62}')

UP31 = Parameter(name = 'UP31',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1 - UP11**2 - UP21**2)',
                 texname = 'U_P^{31}')

UP32 = Parameter(name = 'UP32',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1 - UP12**2 - UP22**2)',
                 texname = 'U_P^{32}')

UP33 = Parameter(name = 'UP33',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1 - UP13**2 - UP23**2)',
                 texname = 'U_P^{33}')

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

vs = Parameter(name = 'vs',
               nature = 'internal',
               type = 'real',
               value = '(mueff*cmath.sqrt(2))/NMl',
               texname = 'v_s')

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

gp = Parameter(name = 'gp',
               nature = 'internal',
               type = 'real',
               value = '1',
               texname = 'g\'')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = '1',
               texname = 'g_w')

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
               value = '((-mHd2 + mHu2 - MZ**2*cmath.cos(2*beta))*cmath.tan(2*beta))/2.',
               texname = 'b')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = 'cmath.sqrt(1 - cw**2)',
               texname = 's_w')

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

I133 = Parameter(name = 'I133',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complexconjugate(yu33)',
                 texname = '\\text{I133}')

I1031 = Parameter(name = 'I1031',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(yd33)',
                  texname = '\\text{I1031}')

I1032 = Parameter(name = 'I1032',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(yd33)',
                  texname = '\\text{I1032}')

I10011 = Parameter(name = 'I10011',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd16*complexconjugate(Rd16)',
                   texname = '\\text{I10011}')

I10012 = Parameter(name = 'I10012',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd26*complexconjugate(Rd16)',
                   texname = '\\text{I10012}')

I10021 = Parameter(name = 'I10021',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd16*complexconjugate(Rd26)',
                   texname = '\\text{I10021}')

I10022 = Parameter(name = 'I10022',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd26*complexconjugate(Rd26)',
                   texname = '\\text{I10022}')

I10033 = Parameter(name = 'I10033',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd35*complexconjugate(Rd35)',
                   texname = '\\text{I10033}')

I10044 = Parameter(name = 'I10044',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rd44*complexconjugate(Rd44)',
                   texname = '\\text{I10044}')

I10111 = Parameter(name = 'I10111',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl16*complexconjugate(Rl16)',
                   texname = '\\text{I10111}')

I10114 = Parameter(name = 'I10114',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl46*complexconjugate(Rl16)',
                   texname = '\\text{I10114}')

I10122 = Parameter(name = 'I10122',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl24*complexconjugate(Rl24)',
                   texname = '\\text{I10122}')

I10133 = Parameter(name = 'I10133',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl35*complexconjugate(Rl35)',
                   texname = '\\text{I10133}')

I10141 = Parameter(name = 'I10141',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl16*complexconjugate(Rl46)',
                   texname = '\\text{I10141}')

I10144 = Parameter(name = 'I10144',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Rl46*complexconjugate(Rl46)',
                   texname = '\\text{I10144}')

I10211 = Parameter(name = 'I10211',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru16*complexconjugate(Ru16)',
                   texname = '\\text{I10211}')

I10212 = Parameter(name = 'I10212',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru26*complexconjugate(Ru16)',
                   texname = '\\text{I10212}')

I10221 = Parameter(name = 'I10221',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru16*complexconjugate(Ru26)',
                   texname = '\\text{I10221}')

I10222 = Parameter(name = 'I10222',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru26*complexconjugate(Ru26)',
                   texname = '\\text{I10222}')

I10233 = Parameter(name = 'I10233',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru35*complexconjugate(Ru35)',
                   texname = '\\text{I10233}')

I10244 = Parameter(name = 'I10244',
                   nature = 'internal',
                   type = 'complex',
                   value = 'Ru44*complexconjugate(Ru44)',
                   texname = '\\text{I10244}')

I1131 = Parameter(name = 'I1131',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33',
                  texname = '\\text{I1131}')

I1132 = Parameter(name = 'I1132',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33',
                  texname = '\\text{I1132}')

I1211 = Parameter(name = 'I1211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd13)',
                  texname = '\\text{I1211}')

I1212 = Parameter(name = 'I1212',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd13)',
                  texname = '\\text{I1212}')

I1221 = Parameter(name = 'I1221',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd23)',
                  texname = '\\text{I1221}')

I1222 = Parameter(name = 'I1222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd23)',
                  texname = '\\text{I1222}')

I1255 = Parameter(name = 'I1255',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd51*complexconjugate(Rd51)',
                  texname = '\\text{I1255}')

I1266 = Parameter(name = 'I1266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd62*complexconjugate(Rd62)',
                  texname = '\\text{I1266}')

I1311 = Parameter(name = 'I1311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*complexconjugate(Rd16)',
                  texname = '\\text{I1311}')

I1312 = Parameter(name = 'I1312',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*complexconjugate(Rd16)',
                  texname = '\\text{I1312}')

I1321 = Parameter(name = 'I1321',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*complexconjugate(Rd26)',
                  texname = '\\text{I1321}')

I1322 = Parameter(name = 'I1322',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*complexconjugate(Rd26)',
                  texname = '\\text{I1322}')

I1333 = Parameter(name = 'I1333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd35*complexconjugate(Rd35)',
                  texname = '\\text{I1333}')

I1344 = Parameter(name = 'I1344',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd44*complexconjugate(Rd44)',
                  texname = '\\text{I1344}')

I1433 = Parameter(name = 'I1433',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(yd33)',
                  texname = '\\text{I1433}')

I1533 = Parameter(name = 'I1533',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33',
                  texname = '\\text{I1533}')

I1633 = Parameter(name = 'I1633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(ye33)',
                  texname = '\\text{I1633}')

I1731 = Parameter(name = 'I1731',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl16)*complexconjugate(ye33)',
                  texname = '\\text{I1731}')

I1734 = Parameter(name = 'I1734',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl46)*complexconjugate(ye33)',
                  texname = '\\text{I1734}')

I1831 = Parameter(name = 'I1831',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ye33*complexconjugate(Rl13)',
                  texname = '\\text{I1831}')

I1834 = Parameter(name = 'I1834',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ye33*complexconjugate(Rl43)',
                  texname = '\\text{I1834}')

I1911 = Parameter(name = 'I1911',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl13)',
                  texname = '\\text{I1911}')

I1914 = Parameter(name = 'I1914',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl13)',
                  texname = '\\text{I1914}')

I1941 = Parameter(name = 'I1941',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl43)',
                  texname = '\\text{I1941}')

I1944 = Parameter(name = 'I1944',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl43)',
                  texname = '\\text{I1944}')

I1955 = Parameter(name = 'I1955',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl51*complexconjugate(Rl51)',
                  texname = '\\text{I1955}')

I1966 = Parameter(name = 'I1966',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl62*complexconjugate(Rl62)',
                  texname = '\\text{I1966}')

I233 = Parameter(name = 'I233',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yd33',
                 texname = '\\text{I233}')

I2011 = Parameter(name = 'I2011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*complexconjugate(Rl16)',
                  texname = '\\text{I2011}')

I2014 = Parameter(name = 'I2014',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*complexconjugate(Rl16)',
                  texname = '\\text{I2014}')

I2022 = Parameter(name = 'I2022',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl24*complexconjugate(Rl24)',
                  texname = '\\text{I2022}')

I2033 = Parameter(name = 'I2033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl35*complexconjugate(Rl35)',
                  texname = '\\text{I2033}')

I2041 = Parameter(name = 'I2041',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*complexconjugate(Rl46)',
                  texname = '\\text{I2041}')

I2044 = Parameter(name = 'I2044',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*complexconjugate(Rl46)',
                  texname = '\\text{I2044}')

I2131 = Parameter(name = 'I2131',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(ye33)',
                  texname = '\\text{I2131}')

I2134 = Parameter(name = 'I2134',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(ye33)',
                  texname = '\\text{I2134}')

I2231 = Parameter(name = 'I2231',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*ye33',
                  texname = '\\text{I2231}')

I2234 = Parameter(name = 'I2234',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*ye33',
                  texname = '\\text{I2234}')

I2315 = Parameter(name = 'I2315',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl51',
                  texname = '\\text{I2315}')

I2326 = Parameter(name = 'I2326',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl62',
                  texname = '\\text{I2326}')

I2331 = Parameter(name = 'I2331',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13',
                  texname = '\\text{I2331}')

I2334 = Parameter(name = 'I2334',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43',
                  texname = '\\text{I2334}')

I2431 = Parameter(name = 'I2431',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*ye33',
                  texname = '\\text{I2431}')

I2434 = Parameter(name = 'I2434',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*ye33',
                  texname = '\\text{I2434}')

I2511 = Parameter(name = 'I2511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl13)',
                  texname = '\\text{I2511}')

I2514 = Parameter(name = 'I2514',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl13)',
                  texname = '\\text{I2514}')

I2541 = Parameter(name = 'I2541',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl43)',
                  texname = '\\text{I2541}')

I2544 = Parameter(name = 'I2544',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl43)',
                  texname = '\\text{I2544}')

I2555 = Parameter(name = 'I2555',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl51*complexconjugate(Rl51)',
                  texname = '\\text{I2555}')

I2566 = Parameter(name = 'I2566',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl62*complexconjugate(Rl62)',
                  texname = '\\text{I2566}')

I2611 = Parameter(name = 'I2611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*complexconjugate(Rl16)',
                  texname = '\\text{I2611}')

I2614 = Parameter(name = 'I2614',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*complexconjugate(Rl16)',
                  texname = '\\text{I2614}')

I2622 = Parameter(name = 'I2622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl24*complexconjugate(Rl24)',
                  texname = '\\text{I2622}')

I2633 = Parameter(name = 'I2633',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl35*complexconjugate(Rl35)',
                  texname = '\\text{I2633}')

I2641 = Parameter(name = 'I2641',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*complexconjugate(Rl46)',
                  texname = '\\text{I2641}')

I2644 = Parameter(name = 'I2644',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*complexconjugate(Rl46)',
                  texname = '\\text{I2644}')

I2711 = Parameter(name = 'I2711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rn13)',
                  texname = '\\text{I2711}')

I2714 = Parameter(name = 'I2714',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rn13)',
                  texname = '\\text{I2714}')

I2726 = Parameter(name = 'I2726',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl62*complexconjugate(Rn22)',
                  texname = '\\text{I2726}')

I2735 = Parameter(name = 'I2735',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl51*complexconjugate(Rn31)',
                  texname = '\\text{I2735}')

I2811 = Parameter(name = 'I2811',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*te33*complexconjugate(Rn13)',
                  texname = '\\text{I2811}')

I2814 = Parameter(name = 'I2814',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*te33*complexconjugate(Rn13)',
                  texname = '\\text{I2814}')

I2911 = Parameter(name = 'I2911',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*ye33*complexconjugate(Rn13)*complexconjugate(ye33)',
                  texname = '\\text{I2911}')

I2914 = Parameter(name = 'I2914',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*ye33*complexconjugate(Rn13)*complexconjugate(ye33)',
                  texname = '\\text{I2914}')

I331 = Parameter(name = 'I331',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complexconjugate(Rd16)*complexconjugate(yd33)',
                 texname = '\\text{I331}')

I332 = Parameter(name = 'I332',
                 nature = 'internal',
                 type = 'complex',
                 value = 'complexconjugate(Rd26)*complexconjugate(yd33)',
                 texname = '\\text{I332}')

I3011 = Parameter(name = 'I3011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*ye33*complexconjugate(Rn13)',
                  texname = '\\text{I3011}')

I3014 = Parameter(name = 'I3014',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*ye33*complexconjugate(Rn13)',
                  texname = '\\text{I3014}')

I3113 = Parameter(name = 'I3113',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn31',
                  texname = '\\text{I3113}')

I3122 = Parameter(name = 'I3122',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn22',
                  texname = '\\text{I3122}')

I3131 = Parameter(name = 'I3131',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13',
                  texname = '\\text{I3131}')

I3231 = Parameter(name = 'I3231',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(ye33)',
                  texname = '\\text{I3231}')

I3311 = Parameter(name = 'I3311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl13)',
                  texname = '\\text{I3311}')

I3314 = Parameter(name = 'I3314',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl43)',
                  texname = '\\text{I3314}')

I3326 = Parameter(name = 'I3326',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn22*complexconjugate(Rl62)',
                  texname = '\\text{I3326}')

I3335 = Parameter(name = 'I3335',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn31*complexconjugate(Rl51)',
                  texname = '\\text{I3335}')

I3411 = Parameter(name = 'I3411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl16)*complexconjugate(ye33)',
                  texname = '\\text{I3411}')

I3414 = Parameter(name = 'I3414',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl46)*complexconjugate(ye33)',
                  texname = '\\text{I3414}')

I3511 = Parameter(name = 'I3511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl16)*complexconjugate(te33)',
                  texname = '\\text{I3511}')

I3514 = Parameter(name = 'I3514',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl46)*complexconjugate(te33)',
                  texname = '\\text{I3514}')

I3611 = Parameter(name = 'I3611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*ye33*complexconjugate(Rl13)*complexconjugate(ye33)',
                  texname = '\\text{I3611}')

I3614 = Parameter(name = 'I3614',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*ye33*complexconjugate(Rl43)*complexconjugate(ye33)',
                  texname = '\\text{I3614}')

I3731 = Parameter(name = 'I3731',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I3731}')

I3732 = Parameter(name = 'I3732',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I3732}')

I3831 = Parameter(name = 'I3831',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Ru13)',
                  texname = '\\text{I3831}')

I3832 = Parameter(name = 'I3832',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Ru23)',
                  texname = '\\text{I3832}')

I3911 = Parameter(name = 'I3911',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru13)',
                  texname = '\\text{I3911}')

I3912 = Parameter(name = 'I3912',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru13)',
                  texname = '\\text{I3912}')

I3921 = Parameter(name = 'I3921',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru23)',
                  texname = '\\text{I3921}')

I3922 = Parameter(name = 'I3922',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru23)',
                  texname = '\\text{I3922}')

I3955 = Parameter(name = 'I3955',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru51*complexconjugate(Ru51)',
                  texname = '\\text{I3955}')

I3966 = Parameter(name = 'I3966',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru62*complexconjugate(Ru62)',
                  texname = '\\text{I3966}')

I431 = Parameter(name = 'I431',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yd33*complexconjugate(Rd13)',
                 texname = '\\text{I431}')

I432 = Parameter(name = 'I432',
                 nature = 'internal',
                 type = 'complex',
                 value = 'yd33*complexconjugate(Rd23)',
                 texname = '\\text{I432}')

I4011 = Parameter(name = 'I4011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*complexconjugate(Ru16)',
                  texname = '\\text{I4011}')

I4012 = Parameter(name = 'I4012',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*complexconjugate(Ru16)',
                  texname = '\\text{I4012}')

I4021 = Parameter(name = 'I4021',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*complexconjugate(Ru26)',
                  texname = '\\text{I4021}')

I4022 = Parameter(name = 'I4022',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*complexconjugate(Ru26)',
                  texname = '\\text{I4022}')

I4033 = Parameter(name = 'I4033',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru35*complexconjugate(Ru35)',
                  texname = '\\text{I4033}')

I4044 = Parameter(name = 'I4044',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru44*complexconjugate(Ru44)',
                  texname = '\\text{I4044}')

I4111 = Parameter(name = 'I4111',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru13)',
                  texname = '\\text{I4111}')

I4112 = Parameter(name = 'I4112',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru23)',
                  texname = '\\text{I4112}')

I4121 = Parameter(name = 'I4121',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru13)',
                  texname = '\\text{I4121}')

I4122 = Parameter(name = 'I4122',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru23)',
                  texname = '\\text{I4122}')

I4155 = Parameter(name = 'I4155',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd51*complexconjugate(Ru51)',
                  texname = '\\text{I4155}')

I4166 = Parameter(name = 'I4166',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd62*complexconjugate(Ru62)',
                  texname = '\\text{I4166}')

I4211 = Parameter(name = 'I4211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru16)*complexconjugate(tu33)',
                  texname = '\\text{I4211}')

I4212 = Parameter(name = 'I4212',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru26)*complexconjugate(tu33)',
                  texname = '\\text{I4212}')

I4221 = Parameter(name = 'I4221',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru16)*complexconjugate(tu33)',
                  texname = '\\text{I4221}')

I4222 = Parameter(name = 'I4222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru26)*complexconjugate(tu33)',
                  texname = '\\text{I4222}')

I4311 = Parameter(name = 'I4311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I4311}')

I4312 = Parameter(name = 'I4312',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I4312}')

I4321 = Parameter(name = 'I4321',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I4321}')

I4322 = Parameter(name = 'I4322',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I4322}')

I4411 = Parameter(name = 'I4411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*td33*complexconjugate(Ru13)',
                  texname = '\\text{I4411}')

I4412 = Parameter(name = 'I4412',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*td33*complexconjugate(Ru23)',
                  texname = '\\text{I4412}')

I4421 = Parameter(name = 'I4421',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*td33*complexconjugate(Ru13)',
                  texname = '\\text{I4421}')

I4422 = Parameter(name = 'I4422',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*td33*complexconjugate(Ru23)',
                  texname = '\\text{I4422}')

I4511 = Parameter(name = 'I4511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*yd33*complexconjugate(Ru13)*complexconjugate(yd33)',
                  texname = '\\text{I4511}')

I4512 = Parameter(name = 'I4512',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*yd33*complexconjugate(Ru23)*complexconjugate(yd33)',
                  texname = '\\text{I4512}')

I4521 = Parameter(name = 'I4521',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*yd33*complexconjugate(Ru13)*complexconjugate(yd33)',
                  texname = '\\text{I4521}')

I4522 = Parameter(name = 'I4522',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*yd33*complexconjugate(Ru23)*complexconjugate(yd33)',
                  texname = '\\text{I4522}')

I4611 = Parameter(name = 'I4611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Ru13)',
                  texname = '\\text{I4611}')

I4612 = Parameter(name = 'I4612',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Ru23)',
                  texname = '\\text{I4612}')

I4621 = Parameter(name = 'I4621',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Ru13)',
                  texname = '\\text{I4621}')

I4622 = Parameter(name = 'I4622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Ru23)',
                  texname = '\\text{I4622}')

I4711 = Parameter(name = 'I4711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I4711}')

I4712 = Parameter(name = 'I4712',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I4712}')

I4721 = Parameter(name = 'I4721',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I4721}')

I4722 = Parameter(name = 'I4722',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I4722}')

I4811 = Parameter(name = 'I4811',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*yu33*complexconjugate(Ru13)*complexconjugate(yu33)',
                  texname = '\\text{I4811}')

I4812 = Parameter(name = 'I4812',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*yu33*complexconjugate(Ru23)*complexconjugate(yu33)',
                  texname = '\\text{I4812}')

I4821 = Parameter(name = 'I4821',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*yu33*complexconjugate(Ru13)*complexconjugate(yu33)',
                  texname = '\\text{I4821}')

I4822 = Parameter(name = 'I4822',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*yu33*complexconjugate(Ru23)*complexconjugate(yu33)',
                  texname = '\\text{I4822}')

I4931 = Parameter(name = 'I4931',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(yu33)',
                  texname = '\\text{I4931}')

I4932 = Parameter(name = 'I4932',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(yu33)',
                  texname = '\\text{I4932}')

I511 = Parameter(name = 'I511',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd13*complexconjugate(Rd13)',
                 texname = '\\text{I511}')

I512 = Parameter(name = 'I512',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd23*complexconjugate(Rd13)',
                 texname = '\\text{I512}')

I521 = Parameter(name = 'I521',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd13*complexconjugate(Rd23)',
                 texname = '\\text{I521}')

I522 = Parameter(name = 'I522',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd23*complexconjugate(Rd23)',
                 texname = '\\text{I522}')

I555 = Parameter(name = 'I555',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd51*complexconjugate(Rd51)',
                 texname = '\\text{I555}')

I566 = Parameter(name = 'I566',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd62*complexconjugate(Rd62)',
                 texname = '\\text{I566}')

I5031 = Parameter(name = 'I5031',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33',
                  texname = '\\text{I5031}')

I5032 = Parameter(name = 'I5032',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33',
                  texname = '\\text{I5032}')

I5115 = Parameter(name = 'I5115',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru51',
                  texname = '\\text{I5115}')

I5126 = Parameter(name = 'I5126',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru62',
                  texname = '\\text{I5126}')

I5131 = Parameter(name = 'I5131',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13',
                  texname = '\\text{I5131}')

I5132 = Parameter(name = 'I5132',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23',
                  texname = '\\text{I5132}')

I5231 = Parameter(name = 'I5231',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(yd33)',
                  texname = '\\text{I5231}')

I5232 = Parameter(name = 'I5232',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(yd33)',
                  texname = '\\text{I5232}')

I5331 = Parameter(name = 'I5331',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33',
                  texname = '\\text{I5331}')

I5332 = Parameter(name = 'I5332',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33',
                  texname = '\\text{I5332}')

I5411 = Parameter(name = 'I5411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd13)',
                  texname = '\\text{I5411}')

I5412 = Parameter(name = 'I5412',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd13)',
                  texname = '\\text{I5412}')

I5421 = Parameter(name = 'I5421',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd23)',
                  texname = '\\text{I5421}')

I5422 = Parameter(name = 'I5422',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd23)',
                  texname = '\\text{I5422}')

I5455 = Parameter(name = 'I5455',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru51*complexconjugate(Rd51)',
                  texname = '\\text{I5455}')

I5466 = Parameter(name = 'I5466',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru62*complexconjugate(Rd62)',
                  texname = '\\text{I5466}')

I5511 = Parameter(name = 'I5511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I5511}')

I5512 = Parameter(name = 'I5512',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I5512}')

I5521 = Parameter(name = 'I5521',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I5521}')

I5522 = Parameter(name = 'I5522',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I5522}')

I5611 = Parameter(name = 'I5611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd16)*complexconjugate(td33)',
                  texname = '\\text{I5611}')

I5612 = Parameter(name = 'I5612',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd16)*complexconjugate(td33)',
                  texname = '\\text{I5612}')

I5621 = Parameter(name = 'I5621',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd26)*complexconjugate(td33)',
                  texname = '\\text{I5621}')

I5622 = Parameter(name = 'I5622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd26)*complexconjugate(td33)',
                  texname = '\\text{I5622}')

I5711 = Parameter(name = 'I5711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*tu33*complexconjugate(Rd13)',
                  texname = '\\text{I5711}')

I5712 = Parameter(name = 'I5712',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*tu33*complexconjugate(Rd13)',
                  texname = '\\text{I5712}')

I5721 = Parameter(name = 'I5721',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*tu33*complexconjugate(Rd23)',
                  texname = '\\text{I5721}')

I5722 = Parameter(name = 'I5722',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*tu33*complexconjugate(Rd23)',
                  texname = '\\text{I5722}')

I5811 = Parameter(name = 'I5811',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*yd33*complexconjugate(Rd13)*complexconjugate(yd33)',
                  texname = '\\text{I5811}')

I5812 = Parameter(name = 'I5812',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*yd33*complexconjugate(Rd13)*complexconjugate(yd33)',
                  texname = '\\text{I5812}')

I5821 = Parameter(name = 'I5821',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*yd33*complexconjugate(Rd23)*complexconjugate(yd33)',
                  texname = '\\text{I5821}')

I5822 = Parameter(name = 'I5822',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*yd33*complexconjugate(Rd23)*complexconjugate(yd33)',
                  texname = '\\text{I5822}')

I5911 = Parameter(name = 'I5911',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I5911}')

I5912 = Parameter(name = 'I5912',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I5912}')

I5921 = Parameter(name = 'I5921',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I5921}')

I5922 = Parameter(name = 'I5922',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I5922}')

I611 = Parameter(name = 'I611',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd16*complexconjugate(Rd16)',
                 texname = '\\text{I611}')

I612 = Parameter(name = 'I612',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd26*complexconjugate(Rd16)',
                 texname = '\\text{I612}')

I621 = Parameter(name = 'I621',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd16*complexconjugate(Rd26)',
                 texname = '\\text{I621}')

I622 = Parameter(name = 'I622',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd26*complexconjugate(Rd26)',
                 texname = '\\text{I622}')

I633 = Parameter(name = 'I633',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd35*complexconjugate(Rd35)',
                 texname = '\\text{I633}')

I644 = Parameter(name = 'I644',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd44*complexconjugate(Rd44)',
                 texname = '\\text{I644}')

I6011 = Parameter(name = 'I6011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*yu33*complexconjugate(Rd13)*complexconjugate(yu33)',
                  texname = '\\text{I6011}')

I6012 = Parameter(name = 'I6012',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*yu33*complexconjugate(Rd13)*complexconjugate(yu33)',
                  texname = '\\text{I6012}')

I6021 = Parameter(name = 'I6021',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*yu33*complexconjugate(Rd23)*complexconjugate(yu33)',
                  texname = '\\text{I6021}')

I6022 = Parameter(name = 'I6022',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*yu33*complexconjugate(Rd23)*complexconjugate(yu33)',
                  texname = '\\text{I6022}')

I6111 = Parameter(name = 'I6111',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Rd13)',
                  texname = '\\text{I6111}')

I6112 = Parameter(name = 'I6112',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Rd13)',
                  texname = '\\text{I6112}')

I6121 = Parameter(name = 'I6121',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Rd23)',
                  texname = '\\text{I6121}')

I6122 = Parameter(name = 'I6122',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Rd23)',
                  texname = '\\text{I6122}')

I6211 = Parameter(name = 'I6211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru13)',
                  texname = '\\text{I6211}')

I6212 = Parameter(name = 'I6212',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru13)',
                  texname = '\\text{I6212}')

I6221 = Parameter(name = 'I6221',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru23)',
                  texname = '\\text{I6221}')

I6222 = Parameter(name = 'I6222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru23)',
                  texname = '\\text{I6222}')

I6255 = Parameter(name = 'I6255',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru51*complexconjugate(Ru51)',
                  texname = '\\text{I6255}')

I6266 = Parameter(name = 'I6266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru62*complexconjugate(Ru62)',
                  texname = '\\text{I6266}')

I6311 = Parameter(name = 'I6311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*complexconjugate(Ru16)',
                  texname = '\\text{I6311}')

I6312 = Parameter(name = 'I6312',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*complexconjugate(Ru16)',
                  texname = '\\text{I6312}')

I6321 = Parameter(name = 'I6321',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*complexconjugate(Ru26)',
                  texname = '\\text{I6321}')

I6322 = Parameter(name = 'I6322',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*complexconjugate(Ru26)',
                  texname = '\\text{I6322}')

I6333 = Parameter(name = 'I6333',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru35*complexconjugate(Ru35)',
                  texname = '\\text{I6333}')

I6344 = Parameter(name = 'I6344',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru44*complexconjugate(Ru44)',
                  texname = '\\text{I6344}')

I6411 = Parameter(name = 'I6411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd16)*complexconjugate(td33)',
                  texname = '\\text{I6411}')

I6412 = Parameter(name = 'I6412',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd16)*complexconjugate(td33)',
                  texname = '\\text{I6412}')

I6421 = Parameter(name = 'I6421',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd26)*complexconjugate(td33)',
                  texname = '\\text{I6421}')

I6422 = Parameter(name = 'I6422',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd26)*complexconjugate(td33)',
                  texname = '\\text{I6422}')

I6511 = Parameter(name = 'I6511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*td33*complexconjugate(Rd13)',
                  texname = '\\text{I6511}')

I6512 = Parameter(name = 'I6512',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*td33*complexconjugate(Rd13)',
                  texname = '\\text{I6512}')

I6521 = Parameter(name = 'I6521',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*td33*complexconjugate(Rd23)',
                  texname = '\\text{I6521}')

I6522 = Parameter(name = 'I6522',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*td33*complexconjugate(Rd23)',
                  texname = '\\text{I6522}')

I6611 = Parameter(name = 'I6611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I6611}')

I6612 = Parameter(name = 'I6612',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I6612}')

I6621 = Parameter(name = 'I6621',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I6621}')

I6622 = Parameter(name = 'I6622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I6622}')

I6711 = Parameter(name = 'I6711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Rd13)',
                  texname = '\\text{I6711}')

I6712 = Parameter(name = 'I6712',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Rd13)',
                  texname = '\\text{I6712}')

I6721 = Parameter(name = 'I6721',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Rd23)',
                  texname = '\\text{I6721}')

I6722 = Parameter(name = 'I6722',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Rd23)',
                  texname = '\\text{I6722}')

I6811 = Parameter(name = 'I6811',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl16)*complexconjugate(te33)',
                  texname = '\\text{I6811}')

I6814 = Parameter(name = 'I6814',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl16)*complexconjugate(te33)',
                  texname = '\\text{I6814}')

I6841 = Parameter(name = 'I6841',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl46)*complexconjugate(te33)',
                  texname = '\\text{I6841}')

I6844 = Parameter(name = 'I6844',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl46)*complexconjugate(te33)',
                  texname = '\\text{I6844}')

I6911 = Parameter(name = 'I6911',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*te33*complexconjugate(Rl13)',
                  texname = '\\text{I6911}')

I6914 = Parameter(name = 'I6914',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*te33*complexconjugate(Rl13)',
                  texname = '\\text{I6914}')

I6941 = Parameter(name = 'I6941',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*te33*complexconjugate(Rl43)',
                  texname = '\\text{I6941}')

I6944 = Parameter(name = 'I6944',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*te33*complexconjugate(Rl43)',
                  texname = '\\text{I6944}')

I715 = Parameter(name = 'I715',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd51',
                 texname = '\\text{I715}')

I726 = Parameter(name = 'I726',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd62',
                 texname = '\\text{I726}')

I731 = Parameter(name = 'I731',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd13',
                 texname = '\\text{I731}')

I732 = Parameter(name = 'I732',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd23',
                 texname = '\\text{I732}')

I7011 = Parameter(name = 'I7011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl16)*complexconjugate(ye33)',
                  texname = '\\text{I7011}')

I7014 = Parameter(name = 'I7014',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl16)*complexconjugate(ye33)',
                  texname = '\\text{I7014}')

I7041 = Parameter(name = 'I7041',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl46)*complexconjugate(ye33)',
                  texname = '\\text{I7041}')

I7044 = Parameter(name = 'I7044',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl46)*complexconjugate(ye33)',
                  texname = '\\text{I7044}')

I7111 = Parameter(name = 'I7111',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*ye33*complexconjugate(Rl13)',
                  texname = '\\text{I7111}')

I7114 = Parameter(name = 'I7114',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*ye33*complexconjugate(Rl13)',
                  texname = '\\text{I7114}')

I7141 = Parameter(name = 'I7141',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*ye33*complexconjugate(Rl43)',
                  texname = '\\text{I7141}')

I7144 = Parameter(name = 'I7144',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*ye33*complexconjugate(Rl43)',
                  texname = '\\text{I7144}')

I7211 = Parameter(name = 'I7211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I7211}')

I7212 = Parameter(name = 'I7212',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I7212}')

I7221 = Parameter(name = 'I7221',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I7221}')

I7222 = Parameter(name = 'I7222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I7222}')

I7311 = Parameter(name = 'I7311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru16)*complexconjugate(tu33)',
                  texname = '\\text{I7311}')

I7312 = Parameter(name = 'I7312',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru16)*complexconjugate(tu33)',
                  texname = '\\text{I7312}')

I7321 = Parameter(name = 'I7321',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru26)*complexconjugate(tu33)',
                  texname = '\\text{I7321}')

I7322 = Parameter(name = 'I7322',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru26)*complexconjugate(tu33)',
                  texname = '\\text{I7322}')

I7411 = Parameter(name = 'I7411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*tu33*complexconjugate(Ru13)',
                  texname = '\\text{I7411}')

I7412 = Parameter(name = 'I7412',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*tu33*complexconjugate(Ru13)',
                  texname = '\\text{I7412}')

I7421 = Parameter(name = 'I7421',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*tu33*complexconjugate(Ru23)',
                  texname = '\\text{I7421}')

I7422 = Parameter(name = 'I7422',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*tu33*complexconjugate(Ru23)',
                  texname = '\\text{I7422}')

I7511 = Parameter(name = 'I7511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Ru13)',
                  texname = '\\text{I7511}')

I7512 = Parameter(name = 'I7512',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Ru13)',
                  texname = '\\text{I7512}')

I7521 = Parameter(name = 'I7521',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Ru23)',
                  texname = '\\text{I7521}')

I7522 = Parameter(name = 'I7522',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Ru23)',
                  texname = '\\text{I7522}')

I7611 = Parameter(name = 'I7611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I7611}')

I7612 = Parameter(name = 'I7612',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I7612}')

I7621 = Parameter(name = 'I7621',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd16*yd33*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I7621}')

I7622 = Parameter(name = 'I7622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd26*yd33*complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I7622}')

I7711 = Parameter(name = 'I7711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*yd33*complexconjugate(Rd13)*complexconjugate(yd33)',
                  texname = '\\text{I7711}')

I7712 = Parameter(name = 'I7712',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*yd33*complexconjugate(Rd13)*complexconjugate(yd33)',
                  texname = '\\text{I7712}')

I7721 = Parameter(name = 'I7721',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*yd33*complexconjugate(Rd23)*complexconjugate(yd33)',
                  texname = '\\text{I7721}')

I7722 = Parameter(name = 'I7722',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*yd33*complexconjugate(Rd23)*complexconjugate(yd33)',
                  texname = '\\text{I7722}')

I7811 = Parameter(name = 'I7811',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*ye33*complexconjugate(Rl13)*complexconjugate(ye33)',
                  texname = '\\text{I7811}')

I7814 = Parameter(name = 'I7814',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*ye33*complexconjugate(Rl13)*complexconjugate(ye33)',
                  texname = '\\text{I7814}')

I7841 = Parameter(name = 'I7841',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*ye33*complexconjugate(Rl43)*complexconjugate(ye33)',
                  texname = '\\text{I7841}')

I7844 = Parameter(name = 'I7844',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*ye33*complexconjugate(Rl43)*complexconjugate(ye33)',
                  texname = '\\text{I7844}')

I7911 = Parameter(name = 'I7911',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*ye33*complexconjugate(Rl16)*complexconjugate(ye33)',
                  texname = '\\text{I7911}')

I7914 = Parameter(name = 'I7914',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*ye33*complexconjugate(Rl16)*complexconjugate(ye33)',
                  texname = '\\text{I7914}')

I7941 = Parameter(name = 'I7941',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl16*ye33*complexconjugate(Rl46)*complexconjugate(ye33)',
                  texname = '\\text{I7941}')

I7944 = Parameter(name = 'I7944',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl46*ye33*complexconjugate(Rl46)*complexconjugate(ye33)',
                  texname = '\\text{I7944}')

I831 = Parameter(name = 'I831',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd13*complexconjugate(yu33)',
                 texname = '\\text{I831}')

I832 = Parameter(name = 'I832',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd23*complexconjugate(yu33)',
                 texname = '\\text{I832}')

I8011 = Parameter(name = 'I8011',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*yu33*complexconjugate(Ru13)*complexconjugate(yu33)',
                  texname = '\\text{I8011}')

I8012 = Parameter(name = 'I8012',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*yu33*complexconjugate(Ru13)*complexconjugate(yu33)',
                  texname = '\\text{I8012}')

I8021 = Parameter(name = 'I8021',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*yu33*complexconjugate(Ru23)*complexconjugate(yu33)',
                  texname = '\\text{I8021}')

I8022 = Parameter(name = 'I8022',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*yu33*complexconjugate(Ru23)*complexconjugate(yu33)',
                  texname = '\\text{I8022}')

I8111 = Parameter(name = 'I8111',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I8111}')

I8112 = Parameter(name = 'I8112',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I8112}')

I8121 = Parameter(name = 'I8121',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru16*yu33*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I8121}')

I8122 = Parameter(name = 'I8122',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru26*yu33*complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I8122}')

I8215 = Parameter(name = 'I8215',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd51)',
                  texname = '\\text{I8215}')

I8226 = Parameter(name = 'I8226',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd62)',
                  texname = '\\text{I8226}')

I8231 = Parameter(name = 'I8231',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd13)',
                  texname = '\\text{I8231}')

I8232 = Parameter(name = 'I8232',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd23)',
                  texname = '\\text{I8232}')

I8331 = Parameter(name = 'I8331',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd16)*complexconjugate(yd33)',
                  texname = '\\text{I8331}')

I8332 = Parameter(name = 'I8332',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rd26)*complexconjugate(yd33)',
                  texname = '\\text{I8332}')

I8431 = Parameter(name = 'I8431',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Rd13)',
                  texname = '\\text{I8431}')

I8432 = Parameter(name = 'I8432',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yu33*complexconjugate(Rd23)',
                  texname = '\\text{I8432}')

I8515 = Parameter(name = 'I8515',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl51)',
                  texname = '\\text{I8515}')

I8526 = Parameter(name = 'I8526',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl62)',
                  texname = '\\text{I8526}')

I8531 = Parameter(name = 'I8531',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl13)',
                  texname = '\\text{I8531}')

I8534 = Parameter(name = 'I8534',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl43)',
                  texname = '\\text{I8534}')

I8631 = Parameter(name = 'I8631',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl16)*complexconjugate(ye33)',
                  texname = '\\text{I8631}')

I8634 = Parameter(name = 'I8634',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rl46)*complexconjugate(ye33)',
                  texname = '\\text{I8634}')

I8713 = Parameter(name = 'I8713',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rn31)',
                  texname = '\\text{I8713}')

I8722 = Parameter(name = 'I8722',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rn22)',
                  texname = '\\text{I8722}')

I8731 = Parameter(name = 'I8731',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Rn13)',
                  texname = '\\text{I8731}')

I8831 = Parameter(name = 'I8831',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ye33*complexconjugate(Rn13)',
                  texname = '\\text{I8831}')

I8915 = Parameter(name = 'I8915',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru51)',
                  texname = '\\text{I8915}')

I8926 = Parameter(name = 'I8926',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru62)',
                  texname = '\\text{I8926}')

I8931 = Parameter(name = 'I8931',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru13)',
                  texname = '\\text{I8931}')

I8932 = Parameter(name = 'I8932',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru23)',
                  texname = '\\text{I8932}')

I931 = Parameter(name = 'I931',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd16*yd33',
                 texname = '\\text{I931}')

I932 = Parameter(name = 'I932',
                 nature = 'internal',
                 type = 'complex',
                 value = 'Rd26*yd33',
                 texname = '\\text{I932}')

I9031 = Parameter(name = 'I9031',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru16)*complexconjugate(yu33)',
                  texname = '\\text{I9031}')

I9032 = Parameter(name = 'I9032',
                  nature = 'internal',
                  type = 'complex',
                  value = 'complexconjugate(Ru26)*complexconjugate(yu33)',
                  texname = '\\text{I9032}')

I9131 = Parameter(name = 'I9131',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd33*complexconjugate(Ru13)',
                  texname = '\\text{I9131}')

I9132 = Parameter(name = 'I9132',
                  nature = 'internal',
                  type = 'complex',
                  value = 'yd33*complexconjugate(Ru23)',
                  texname = '\\text{I9132}')

I9211 = Parameter(name = 'I9211',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd13)',
                  texname = '\\text{I9211}')

I9212 = Parameter(name = 'I9212',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd13)',
                  texname = '\\text{I9212}')

I9221 = Parameter(name = 'I9221',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Rd23)',
                  texname = '\\text{I9221}')

I9222 = Parameter(name = 'I9222',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Rd23)',
                  texname = '\\text{I9222}')

I9255 = Parameter(name = 'I9255',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru51*complexconjugate(Rd51)',
                  texname = '\\text{I9255}')

I9266 = Parameter(name = 'I9266',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru62*complexconjugate(Rd62)',
                  texname = '\\text{I9266}')

I9311 = Parameter(name = 'I9311',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl13)',
                  texname = '\\text{I9311}')

I9314 = Parameter(name = 'I9314',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn13*complexconjugate(Rl43)',
                  texname = '\\text{I9314}')

I9326 = Parameter(name = 'I9326',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn22*complexconjugate(Rl62)',
                  texname = '\\text{I9326}')

I9335 = Parameter(name = 'I9335',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rn31*complexconjugate(Rl51)',
                  texname = '\\text{I9335}')

I9411 = Parameter(name = 'I9411',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru13)',
                  texname = '\\text{I9411}')

I9412 = Parameter(name = 'I9412',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Ru23)',
                  texname = '\\text{I9412}')

I9421 = Parameter(name = 'I9421',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru13)',
                  texname = '\\text{I9421}')

I9422 = Parameter(name = 'I9422',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Ru23)',
                  texname = '\\text{I9422}')

I9455 = Parameter(name = 'I9455',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd51*complexconjugate(Ru51)',
                  texname = '\\text{I9455}')

I9466 = Parameter(name = 'I9466',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd62*complexconjugate(Ru62)',
                  texname = '\\text{I9466}')

I9511 = Parameter(name = 'I9511',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rn13)',
                  texname = '\\text{I9511}')

I9514 = Parameter(name = 'I9514',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rn13)',
                  texname = '\\text{I9514}')

I9526 = Parameter(name = 'I9526',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl62*complexconjugate(Rn22)',
                  texname = '\\text{I9526}')

I9535 = Parameter(name = 'I9535',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl51*complexconjugate(Rn31)',
                  texname = '\\text{I9535}')

I9611 = Parameter(name = 'I9611',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd13)',
                  texname = '\\text{I9611}')

I9612 = Parameter(name = 'I9612',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd13)',
                  texname = '\\text{I9612}')

I9621 = Parameter(name = 'I9621',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd13*complexconjugate(Rd23)',
                  texname = '\\text{I9621}')

I9622 = Parameter(name = 'I9622',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd23*complexconjugate(Rd23)',
                  texname = '\\text{I9622}')

I9655 = Parameter(name = 'I9655',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd51*complexconjugate(Rd51)',
                  texname = '\\text{I9655}')

I9666 = Parameter(name = 'I9666',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rd62*complexconjugate(Rd62)',
                  texname = '\\text{I9666}')

I9711 = Parameter(name = 'I9711',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl13)',
                  texname = '\\text{I9711}')

I9714 = Parameter(name = 'I9714',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl13)',
                  texname = '\\text{I9714}')

I9741 = Parameter(name = 'I9741',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl13*complexconjugate(Rl43)',
                  texname = '\\text{I9741}')

I9744 = Parameter(name = 'I9744',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl43*complexconjugate(Rl43)',
                  texname = '\\text{I9744}')

I9755 = Parameter(name = 'I9755',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl51*complexconjugate(Rl51)',
                  texname = '\\text{I9755}')

I9766 = Parameter(name = 'I9766',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Rl62*complexconjugate(Rl62)',
                  texname = '\\text{I9766}')

I9811 = Parameter(name = 'I9811',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru13)',
                  texname = '\\text{I9811}')

I9812 = Parameter(name = 'I9812',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru13)',
                  texname = '\\text{I9812}')

I9821 = Parameter(name = 'I9821',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru13*complexconjugate(Ru23)',
                  texname = '\\text{I9821}')

I9822 = Parameter(name = 'I9822',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru23*complexconjugate(Ru23)',
                  texname = '\\text{I9822}')

I9855 = Parameter(name = 'I9855',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru51*complexconjugate(Ru51)',
                  texname = '\\text{I9855}')

I9866 = Parameter(name = 'I9866',
                  nature = 'internal',
                  type = 'complex',
                  value = 'Ru62*complexconjugate(Ru62)',
                  texname = '\\text{I9866}')

I9933 = Parameter(name = 'I9933',
                  nature = 'internal',
                  type = 'complex',
                  value = 'ye33',
                  texname = '\\text{I9933}')

