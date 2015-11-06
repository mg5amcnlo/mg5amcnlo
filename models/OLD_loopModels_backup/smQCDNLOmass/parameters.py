# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51


from object_library import all_parameters, Parameter

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

import CT_parameters as CTP
import particles as P

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

CA = Parameter(name = 'CA',
                 nature = 'internal',
                 type = 'real',
                 value = '3.0',
                 texname = 'C_{A}')

TF = Parameter(name = 'TF',
                 nature = 'internal',
                 type = 'real',
                 value = '0.5',
                 texname = 'T_{F}')

CF = Parameter(name = 'CF',
                 nature = 'internal',
                 type = 'real',
                 value = '(4.0/3.0)',
                 texname = 'C_{F}')

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1172,
               texname = '\\text{aS}',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172.,
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

MU_R = Parameter(name = 'MU_R',
              nature = 'external',
              type = 'real',
              value = 91.188,
              texname = '\\text{\\mu_r}',
              lhablock = 'LOOP',
              lhacode = [ 666 ])

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')
