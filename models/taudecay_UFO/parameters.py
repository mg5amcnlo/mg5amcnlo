# This file was automatically created by FeynRules 2.0.25
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 8 May 2014 12:30:33



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

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.000011663787,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

F0 = Parameter(name = 'F0',
               nature = 'external',
               type = 'real',
               value = 1.,
               texname = 'F_0',
               lhablock = 'taudecay',
               lhacode = [ 1 ])

F1 = Parameter(name = 'F1',
               nature = 'external',
               type = 'real',
               value = 0.13041,
               texname = 'F_1',
               lhablock = 'taudecay',
               lhacode = [ 2 ])

F2 = Parameter(name = 'F2',
               nature = 'external',
               type = 'real',
               value = 1.,
               texname = 'F_2',
               lhablock = 'taudecay',
               lhacode = [ 3 ])

F3 = Parameter(name = 'F3',
               nature = 'external',
               type = 'real',
               value = 1.,
               texname = 'F_3',
               lhablock = 'taudecay',
               lhacode = [ 4 ])

Fr1 = Parameter(name = 'Fr1',
                nature = 'external',
                type = 'real',
                value = 1.,
                texname = 'F_{23}',
                lhablock = 'taudecay',
                lhacode = [ 5 ])

Fr2 = Parameter(name = 'Fr2',
                nature = 'external',
                type = 'real',
                value = 1.,
                texname = 'F_{13}',
                lhablock = 'taudecay',
                lhacode = [ 6 ])

Gr1 = Parameter(name = 'Gr1',
                nature = 'external',
                type = 'real',
                value = 1.,
                texname = 'G_{23}',
                lhablock = 'taudecay',
                lhacode = [ 7 ])

Gr2 = Parameter(name = 'Gr2',
                nature = 'external',
                type = 'real',
                value = 1.,
                texname = 'G_{13}',
                lhablock = 'taudecay',
                lhacode = [ 8 ])

Me = Parameter(name = 'Me',
               nature = 'external',
               type = 'real',
               value = 0.000511,
               texname = '\\text{Me}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

MMU = Parameter(name = 'MMU',
                nature = 'external',
                type = 'real',
                value = 0.10566,
                texname = '\\text{MMU}',
                lhablock = 'MASS',
                lhacode = [ 13 ])

MTA = Parameter(name = 'MTA',
                nature = 'external',
                type = 'real',
                value = 1.77682,
                texname = '\\text{MTA}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

Mpic = Parameter(name = 'Mpic',
                 nature = 'external',
                 type = 'real',
                 value = 0.13957018,
                 texname = '\\text{Mpic}',
                 lhablock = 'MASS',
                 lhacode = [ 211 ])

Mpi0 = Parameter(name = 'Mpi0',
                 nature = 'external',
                 type = 'real',
                 value = 0.1349766,
                 texname = '\\text{Mpi0}',
                 lhablock = 'MASS',
                 lhacode = [ 111 ])

WTA = Parameter(name = 'WTA',
                nature = 'external',
                type = 'real',
                value = 2.265e-12,
                texname = '\\text{WTA}',
                lhablock = 'DECAY',
                lhacode = [ 15 ])

