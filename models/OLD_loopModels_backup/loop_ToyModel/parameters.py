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

Nlf = Parameter(name = 'Nlf',
                 nature = 'internal',
                 type = 'real',
                 value = '2.0',
                 texname = 'N_{lf}')

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

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.0,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.0,
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

MU_R = Parameter(name = 'MU_R',
              nature = 'external',
              type = 'real',
              value = 91.188,
              texname = '\\text{\\mu_r}',
              lhablock = 'LOOP',
              lhacode = [ 666 ])

G_UV = Parameter(name = 'G_UV',
              nature = 'internal',
              type = 'real',
              # At leading order without fermion masses, the UV renormalization 
              # in MSbar does not affect the finite part of the virtual.
              # In the way it was done in MGv4, it is not possible to easily
              # disantangle the contribution from the different loops to the 
              # UV counterterms (typically g>dd~ should contain a CA which is 
              # later cancelled by the wf renorm. So for now I leave it like this)
              value = '-((G**2)/(48.0*cmath.pi**2))*(11.0*CA-4.0*TF*Nlf)',
              texname = 'G_{UV}')
