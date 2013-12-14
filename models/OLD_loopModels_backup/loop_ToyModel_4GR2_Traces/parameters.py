# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

lhv = Parameter(name = 'lhv',
                 nature = 'internal',
                 type = 'real',
                 value = '1.0',
                 texname = '\lambda_{HV}')

Ncol = Parameter(name = 'Ncol',
                 nature = 'internal',
                 type = 'real',
                 value = '3.0',
                 texname = 'N_{col}')

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1172,
               texname = '\\text{aS}',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.0025499999999999997,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.00504,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 1 ])

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

RGR2 = Parameter(name = 'RGR2',
              nature = 'internal',
              type = 'real',
              # At leading order without fermion masses, the UV renormalization 
              # in MSbar does not affect the finite part of the virtual, so I
              # put zero here
              value = '-complex(0,1)*G**4/(96.0*cmath.pi**2)',
              texname = '4GR2')

G_UV = Parameter(name = 'G_UV',
              nature = 'internal',
              type = 'real',
              # At leading order without fermion masses, the UV renormalization 
              # in MSbar does not affect the finite part of the virtual, so I
              # put zero here
              value = 'ZERO',
              texname = 'G_{UV}')
