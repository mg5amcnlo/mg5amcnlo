# This file was automatically created by FeynRules 1.7.51
# Mathematica version: 8.0 for Linux x86 (64-bit) (February 23, 2011)
# Date: Thu 2 Aug 2012 10:15:23



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
g1 = Parameter(name = 'g1',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_1',
               lhablock = 'COUPLINGS',
               lhacode = [ 1 ])

g2 = Parameter(name = 'g2',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_2',
               lhablock = 'COUPLINGS',
               lhacode = [ 2 ])

g3 = Parameter(name = 'g3',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_3',
               lhablock = 'COUPLINGS',
               lhacode = [ 3 ])

g4 = Parameter(name = 'g4',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_4',
               lhablock = 'COUPLINGS',
               lhacode = [ 4 ])

g5 = Parameter(name = 'g5',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_5',
               lhablock = 'COUPLINGS',
               lhacode = [ 5 ])

g6 = Parameter(name = 'g6',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_6',
               lhablock = 'COUPLINGS',
               lhacode = [ 6 ])

g7 = Parameter(name = 'g7',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = 'g_7',
               lhablock = 'COUPLINGS',
               lhacode = [ 7 ])

g8 = Parameter(name = 'g8',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_8',
               lhablock = 'COUPLINGS',
               lhacode = [ 8 ])

g9 = Parameter(name = 'g9',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = 'g_9',
               lhablock = 'COUPLINGS',
               lhacode = [ 9 ])

g10 = Parameter(name = 'g10',
                nature = 'external',
                type = 'real',
                value = 10,
                texname = 'g_{10}',
                lhablock = 'COUPLINGS',
                lhacode = [ 10 ])

g11 = Parameter(name = 'g11',
                nature = 'external',
                type = 'real',
                value = 10,
                texname = 'g_{11}',
                lhablock = 'COUPLINGS',
                lhacode = [ 11 ])

g12 = Parameter(name = 'g12',
                nature = 'external',
                type = 'real',
                value = 10,
                texname = 'g_{12}',
                lhablock = 'COUPLINGS',
                lhacode = [ 12 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1184,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

MA = Parameter(name = 'MA',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = '\\text{MA}',
               lhablock = 'MASS',
               lhacode = [ 9000001 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 9000002 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 9000003 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 10,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 9000004 ])

MmA = Parameter(name = 'MmA',
                nature = 'external',
                type = 'real',
                value = 10,
                texname = '\\text{MmA}',
                lhablock = 'MASS',
                lhacode = [ 9000005 ])

MmB = Parameter(name = 'MmB',
                nature = 'external',
                type = 'real',
                value = 10,
                texname = '\\text{MmB}',
                lhablock = 'MASS',
                lhacode = [ 9000006 ])

MmC = Parameter(name = 'MmC',
                nature = 'external',
                type = 'real',
                value = 10,
                texname = '\\text{MmC}',
                lhablock = 'MASS',
                lhacode = [ 9000007 ])

MmD = Parameter(name = 'MmD',
                nature = 'external',
                type = 'real',
                value = 10,
                texname = '\\text{MmD}',
                lhablock = 'MASS',
                lhacode = [ 9000008 ])

MV = Parameter(name = 'MV',
               nature = 'external',
               type = 'real',
               value = 50000,
               texname = '\\text{MV}',
               lhablock = 'MASS',
               lhacode = [ 9000009 ])

Mphi = Parameter(name = 'Mphi',
                 nature = 'external',
                 type = 'real',
                 value = 50000,
                 texname = '\\text{Mphi}',
                 lhablock = 'MASS',
                 lhacode = [ 9000010 ])

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

