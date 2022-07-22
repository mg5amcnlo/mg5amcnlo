# This file was automatically created by FeynRules 2.4.78
# Mathematica version: 12.0.0 for Mac OS X x86 (64-bit) (April 7, 2019)
# Date: Wed 1 Apr 2020 19:35:44

import configuration

from object_library import all_parameters, Parameter

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

if configuration.bottomYukawa:
    
    ymb = Parameter(name = 'ymb',
                    nature = 'external',
                    type = 'real',
                    value = 4.7,
                    texname = '\\text{ymb}',
                    lhablock = 'YUKAWA',
                    lhacode = [ 5 ])

    cbp = Parameter(name = 'cbp',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = '$c_{b\\phi}$',
                lhablock = 'DIM62F',
                lhacode = [ 18 ])

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# This is a default parameter object representing the renormalization scale (MU_R).
MU_R = Parameter(name = 'MU_R',
                 nature = 'external',
                 type = 'real',
                 value = 91.188,
                 texname = '\\text{\\mu_r}',
                 lhablock = 'LOOP',
                 lhacode = [1])

muprime = Parameter(name = 'muprime',
                    nature = 'external',
                    type = 'real',
                    value = 172.5,
                    texname = '\\text{$\\mu $1}',
                    lhablock = 'LOOP',
                    lhacode = [ 2 ])

# User-defined parameters.
Lambda = Parameter(name = 'Lambda',
                   nature = 'external',
                   type = 'real',
                   value = 1000,
                   texname = '\\Lambda',
                   lhablock = 'DIM6',
                   lhacode = [ 1 ],
                  scale = 172.5)

cpDC = Parameter(name = 'cpDC',
                 nature = 'external',
                 type = 'real',
                 value = 1.,
                 texname = 'c_{\\text{$\\phi $D}}',
                 lhablock = 'DIM6',
                 lhacode = [ 2 ],
                  scale = 172.5)

cpWB = Parameter(name = 'cpWB',
                 nature = 'external',
                 type = 'real',
                 value = 1.,
                 texname = 'c_{\\text{$\\phi $WB}}',
                 lhablock = 'DIM6',
                 lhacode = [ 3 ],
                  scale = 172.5)

cdp = Parameter(name = 'cdp',
                nature = 'external',
                type = 'real',
                value = 1.,
                texname = 'c_{\\text{d$\\phi $}}',
                lhablock = 'DIM6',
                lhacode = [ 4 ],
                  scale = 172.5)

cp = Parameter(name = 'cp',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = 'c_{\\phi }',
               lhablock = 'DIM6',
               lhacode = [ 5 ],
                  scale = 172.5)

cWWW = Parameter(name = 'cWWW',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'c_W',
                 lhablock = 'DIM6',
                 lhacode = [ 6 ],
                  scale = 172.5)

cG = Parameter(name = 'cG',
               nature = 'external',
               type = 'real',
               value = 1,
               texname = 'c_G',
               lhablock = 'DIM6',
               lhacode = [ 7 ],
                  scale = 172.5)

cpG = Parameter(name = 'cpG',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $G}}',
                lhablock = 'DIM6',
                lhacode = [ 8 ],
                  scale = 172.5)

cpW = Parameter(name = 'cpW',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $W}}',
                lhablock = 'DIM6',
                lhacode = [ 9 ],
                  scale = 172.5)

cpBB = Parameter(name = 'cpBB',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'c_{\\text{$\\phi $B}}',
                 lhablock = 'DIM6',
                 lhacode = [ 10 ],
                  scale = 172.5)

cpl1 = Parameter(name = 'cpl1',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[c,\\text{$\\phi $l1},\\text{(1)}]',
                 lhablock = 'DIM62F',
                 lhacode = [ 1 ],
                  scale = 172.5)

cpl2 = Parameter(name = 'cpl2',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[c,\\text{$\\phi $l2},\\text{(1)}]',
                 lhablock = 'DIM62F',
                 lhacode = [ 2 ],
                  scale = 172.5)

cpl3 = Parameter(name = 'cpl3',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[c,\\text{$\\phi $L},\\text{(1)}]',
                 lhablock = 'DIM62F',
                 lhacode = [ 3 ],
                  scale = 172.5)

c3pl1 = Parameter(name = 'c3pl1',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[c,\\text{$\\phi $l1},\\text{(3)}]',
                  lhablock = 'DIM62F',
                  lhacode = [ 4 ],
                  scale = 172.5)

c3pl2 = Parameter(name = 'c3pl2',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[c,\\text{$\\phi $l2},\\text{(3)}]',
                  lhablock = 'DIM62F',
                  lhacode = [ 5 ],
                  scale = 172.5)

c3pl3 = Parameter(name = 'c3pl3',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[c,\\text{$\\phi $L},\\text{(3)}]',
                  lhablock = 'DIM62F',
                  lhacode = [ 6 ],
                  scale = 172.5)

cpe = Parameter(name = 'cpe',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $e}}',
                lhablock = 'DIM62F',
                lhacode = [ 7 ],
                  scale = 172.5)

cpmu = Parameter(name = 'cpmu',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'c_{\\phi \\mu }',
                 lhablock = 'DIM62F',
                 lhacode = [ 8 ],
                  scale = 172.5)

cpta = Parameter(name = 'cpta',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = 'c_{\\phi \\tau }',
                 lhablock = 'DIM62F',
                 lhacode = [ 9 ],
                  scale = 172.5)

cpqMi = Parameter(name = 'cpqMi',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[c,\\text{$\\phi $q},\\text{(-)}]',
                  lhablock = 'DIM62F',
                  lhacode = [ 10 ],
                  scale = 172.5)

cpq3i = Parameter(name = 'cpq3i',
                  nature = 'external',
                  type = 'real',
                  value = 1,
                  texname = '\\text{Subsuperscript}[c,\\text{$\\phi $q},\\text{(3)}]',
                  lhablock = 'DIM62F',
                  lhacode = [ 11 ],
                  scale = 172.5)

cpQ3 = Parameter(name = 'cpQ3',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[c,\\text{$\\phi $Q},\\text{(3)}]',
                 lhablock = 'DIM62F',
                 lhacode = [ 12 ],
                  scale = 172.5)

cpQM = Parameter(name = 'cpQM',
                 nature = 'external',
                 type = 'real',
                 value = 1,
                 texname = '\\text{Subsuperscript}[c,\\text{$\\phi $Q},\\text{(-)}]',
                 lhablock = 'DIM62F',
                 lhacode = [ 13 ],
                  scale = 172.5)

cpu = Parameter(name = 'cpu',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $u}}',
                lhablock = 'DIM62F',
                lhacode = [ 14 ],
                  scale = 172.5)

cpt = Parameter(name = 'cpt',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $t}}',
                lhablock = 'DIM62F',
                lhacode = [ 15 ],
                  scale = 172.5)

cpd = Parameter(name = 'cpd',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{$\\phi $d}}',
                lhablock = 'DIM62F',
                lhacode = [ 16 ],
                  scale = 172.5)

ctp = Parameter(name = 'ctp',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{t$\\phi $}}',
                lhablock = 'DIM62F',
                lhacode = [ 19 ],
                  scale = 172.5)

ctZ = Parameter(name = 'ctZ',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{tZ}}',
                lhablock = 'DIM62F',
                lhacode = [ 22 ],
                  scale = 172.5)

ctW = Parameter(name = 'ctW',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{tW}}',
                lhablock = 'DIM62F',
                lhacode = [ 23 ],
                  scale = 172.5)

ctG = Parameter(name = 'ctG',
                nature = 'external',
                type = 'real',
                value = 1,
                texname = 'c_{\\text{tG}}',
                lhablock = 'DIM62F',
                lhacode = [ 24 ],
                scale = 172.5)

cQq83 = Parameter(name = 'cQq83',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Qq},\\text{(8,3)}]',
                  lhablock = 'DIM64F',
                  lhacode = [ 1 ],
                  scale = 172.5)

cQq81 = Parameter(name = 'cQq81',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Qq},\\text{(8,1)}]',
                  lhablock = 'DIM64F',
                  lhacode = [ 2 ],
                  scale = 172.5)

cQu8 = Parameter(name = 'cQu8',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{Qu},\\text{(8)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 3 ],
                  scale = 172.5)

ctq8 = Parameter(name = 'ctq8',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{tq},\\text{(8)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 4 ],
                 scale = 172.5)

cQd8 = Parameter(name = 'cQd8',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{Qd},\\text{(8)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 6 ],
                 scale = 172.5)

ctu8 = Parameter(name = 'ctu8',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{tu},\\text{(8)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 7 ],
                 scale = 172.5)

ctd8 = Parameter(name = 'ctd8',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{td},\\text{(8)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 8 ],
                 scale = 172.5)

cQq13 = Parameter(name = 'cQq13',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Qq},\\text{(1,3)}]',
                  lhablock = 'DIM64F',
                  lhacode = [ 10 ],
                  scale = 172.5)

cQq11 = Parameter(name = 'cQq11',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Qq},\\text{(1,1)}]',
                  lhablock = 'DIM64F',
                  lhacode = [ 11 ],
                  scale = 172.5)

cQu1 = Parameter(name = 'cQu1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{Qu},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 12 ],
                 scale = 172.5)

ctq1 = Parameter(name = 'ctq1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{tq},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 13 ],
                 scale = 172.5)

cQd1 = Parameter(name = 'cQd1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{Qd},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 14 ],
                 scale = 172.5)

ctu1 = Parameter(name = 'ctu1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{tu},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 16 ],
                 scale = 172.5)

ctd1 = Parameter(name = 'ctd1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{td},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 17 ],
                 scale = 172.5)

cQQ8 = Parameter(name = 'cQQ8',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{QQ},\\text{(8)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 19 ],
                 scale = 172.5)

cQQ1 = Parameter(name = 'cQQ1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{QQ},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 20 ],
                 scale = 172.5)

cQt1 = Parameter(name = 'cQt1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{Qt},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 21 ],
                 scale = 172.5)

ctt1 = Parameter(name = 'ctt1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{tt},\\text{(1)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 23 ],
                 scale = 172.5)

cQt8 = Parameter(name = 'cQt8',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = '\\text{Subsuperscript}[c,\\text{Qt},\\text{(8)}]',
                 lhablock = 'DIM64F',
                 lhacode = [ 25 ],
                 scale = 172.5)

cQlM1 = Parameter(name = 'cQlM1',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Ql1},\\text{(-)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 1 ],
                  scale = 172.5)

cQlM2 = Parameter(name = 'cQlM2',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Ql2},\\text{(-)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 2 ],
                  scale = 172.5)

cQl31 = Parameter(name = 'cQl31',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Ql1},\\text{(3)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 3 ],
                  scale = 172.5)

cQl32 = Parameter(name = 'cQl32',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{Ql2},\\text{(3)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 4 ],
                  scale = 172.5)

cQe1 = Parameter(name = 'cQe1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{Qe1}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 5 ],
                  scale = 172.5)

cQe2 = Parameter(name = 'cQe2',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{Qe2}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 6 ],
                  scale = 172.5)

ctl1 = Parameter(name = 'ctl1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{tl1}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 7 ],
                  scale = 172.5)

ctl2 = Parameter(name = 'ctl2',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{tl2}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 8 ],
                  scale = 172.5)

cte1 = Parameter(name = 'cte1',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{te1}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 9 ],
                  scale = 172.5)

cte2 = Parameter(name = 'cte2',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{te2}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 10 ],
                  scale = 172.5)

cQlM3 = Parameter(name = 'cQlM3',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{QL},\\text{(-)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 13 ],
                  scale = 172.5)

cQl33 = Parameter(name = 'cQl33',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,\\text{QL},\\text{(3)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 14 ],
                  scale = 172.5)

cQe3 = Parameter(name = 'cQe3',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{Qe3}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 15 ],
                  scale = 172.5)

ctl3 = Parameter(name = 'ctl3',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{tL}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 16 ],
                  scale = 172.5)

cte3 = Parameter(name = 'cte3',
                 nature = 'external',
                 type = 'real',
                 value = 1.e-10,
                 texname = 'c_{\\text{te3}}',
                 lhablock = 'DIM64F2L',
                 lhacode = [ 17 ],
                  scale = 172.5)

ctlS3 = Parameter(name = 'ctlS3',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,t,\\text{S(3)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 19 ],
                  scale = 172.5)

ctlT3 = Parameter(name = 'ctlT3',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,t,\\text{T(3)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 20 ],
                  scale = 172.5)

cblS3 = Parameter(name = 'cblS3',
                  nature = 'external',
                  type = 'real',
                  value = 1.e-10,
                  texname = '\\text{Subsuperscript}[c,b,\\text{S(3)}]',
                  lhablock = 'DIM64F2L',
                  lhacode = [ 21 ],
                  scale = 172.5)

cll1111 = Parameter(name = 'cll1111',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},1111]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 1 ],
                  scale = 172.5)

cll2222 = Parameter(name = 'cll2222',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},2222]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 2 ],
                  scale = 172.5)

cll3333 = Parameter(name = 'cll3333',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},3333]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 3 ],
                  scale = 172.5)

cll1122 = Parameter(name = 'cll1122',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},1122]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 4 ],
                  scale = 172.5)

cll1133 = Parameter(name = 'cll1133',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},1133]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 5 ],
                  scale = 172.5)

cll2233 = Parameter(name = 'cll2233',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},2233]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 6 ],
                  scale = 172.5)

cll1221 = Parameter(name = 'cll1221',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},1221]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 7 ])

cll1331 = Parameter(name = 'cll1331',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},1331]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 8 ],
                  scale = 172.5)

cll2332 = Parameter(name = 'cll2332',
                    nature = 'external',
                    type = 'real',
                    value = 1.e-10,
                    texname = '\\text{Subsuperscript}[c,\\text{ll},2332]',
                    lhablock = 'DIM64F4L',
                    lhacode = [ 9 ],
                  scale = 172.5)



Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.0000116637,
               texname = 'G_f',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.1184,
               texname = '\\alpha _s',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 172,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

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
               value = 79.8244,
               texname = '\\text{MW}',
               lhablock = 'MASS',
               lhacode = [ 24 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 172,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MH = Parameter(name = 'MH',
               nature = 'external',
               type = 'real',
               value = 125,
               texname = '\\text{MH}',
               lhablock = 'MASS',
               lhacode = [ 25 ])

WZ = Parameter(name = 'WZ',
               nature = 'external',
               type = 'real',
               value = 2.4952,
               texname = '\\text{WZ}',
               lhablock = 'DECAY',
               lhacode = [ 23 ])

WW = Parameter(name = 'WW',
               nature = 'external',
               type = 'real',
               value = 2.085,
               texname = '\\text{WW}',
               lhablock = 'DECAY',
               lhacode = [ 24 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00575308848,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

cw0 = Parameter(name = 'cw0',
                nature = 'internal',
                type = 'real',
                value = 'MW/MZ',
                texname = '\\text{Subsuperscript}[c,w,0]')

muH0 = Parameter(name = 'muH0',
                 nature = 'internal',
                 type = 'real',
                 value = 'MH/cmath.sqrt(2)',
                 texname = '\\mu _0')

sw0 = Parameter(name = 'sw0',
                nature = 'internal',
                type = 'real',
                value = 'cmath.sqrt(1 - MW**2/MZ**2)',
                texname = '\\text{Subsuperscript}[s,w,0]')

cF3pl1 = Parameter(name = 'cF3pl1',
                   nature = 'internal',
                   type = 'real',
                   value = 'c3pl1',
                   texname = '\\text{Subsuperscript}[c,\\text{F$\\phi $l1},\\text{(3)}]')

cF3pl2 = Parameter(name = 'cF3pl2',
                   nature = 'internal',
                   type = 'real',
                   value = 'c3pl2',
                   texname = '\\text{Subsuperscript}[c,\\text{F$\\phi $l2},\\text{(3)}]')

cFll = Parameter(name = 'cFll',
                 nature = 'internal',
                 type = 'real',
                 value = '2*cll1221',
                 texname = 'c_{\\text{Fll}}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

vev0 = Parameter(name = 'vev0',
                 nature = 'internal',
                 type = 'real',
                 value = 'cmath.sqrt(1/Gf)/2**0.25',
                 texname = '\\text{vev0}')

dT = Parameter(name = 'dT',
               nature = 'internal',
               type = 'real',
               value = '(cpDC*vev0**2)/(2.*Lambda**2)',
               texname = '\\delta _T')

dv = Parameter(name = 'dv',
               nature = 'internal',
               type = 'real',
               value = '((cF3pl1 + cF3pl2 - cFll/2.)*vev0**2)/Lambda**2',
               texname = '\\delta _v')

dWB = Parameter(name = 'dWB',
                nature = 'internal',
                type = 'real',
                value = '(cpWB*vev0**2)/Lambda**2',
                texname = '\\delta _{\\text{WB}}')

ctB = Parameter(name = 'ctB',
                nature = 'internal',
                type = 'real',
                value = '(-ctZ + ctW*cw0)/sw0',
                texname = '\\text{Subsuperscript}[c,\\text{tB},\\text{[int]}]')

dlam = Parameter(name = 'dlam',
                 nature = 'internal',
                 type = 'real',
                 value = '((4*cdp - cpDC)*vev0**2)/(2.*Lambda**2)',
                 texname = '\\text{dlam}')

ee0 = Parameter(name = 'ee0',
                nature = 'internal',
                type = 'real',
                value = '(2*MW*sw0)/vev0',
                texname = 'e_0')

cw = Parameter(name = 'cw',
               nature = 'internal',
               type = 'real',
               value = 'cw0*(1 + dT/2.)',
               texname = 'c_w')

muH = Parameter(name = 'muH',
                nature = 'internal',
                type = 'real',
                value = '(1 - dlam/2.)*muH0',
                texname = '\\mu')

sw = Parameter(name = 'sw',
               nature = 'internal',
               type = 'real',
               value = '(1 - (cw0**2*dT)/(2.*sw0**2))*sw0',
               texname = 's_w')

g1 = Parameter(name = 'g1',
               nature = 'internal',
               type = 'real',
               value = '(ee0*(1 - dv/2. - dT/(2.*sw0**2) - (cw0*dWB)/sw0))/cw0',
               texname = 'g_1')

gw = Parameter(name = 'gw',
               nature = 'internal',
               type = 'real',
               value = '((1 - dv/2.)*ee0)/sw0',
               texname = 'g_w')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = '((1 - dlam - dv)*MH**2)/(2.*vev0**2)',
                texname = '\\text{lam}')

vev = Parameter(name = 'vev',
                nature = 'internal',
                type = 'real',
                value = '(1 + dv/2.)*vev0',
                texname = '\\text{vev}')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '(2*MW*sw*(1 - dv/2. - (cw0*dWB)/sw0))/vev0',
               texname = 'e')

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = 'ee**2/(4.*cmath.pi)',
                texname = '\\alpha _{\\text{EW}}')

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'internal',
                  type = 'real',
                  value = '1/aEW',
                  texname = '\\text{aEWM1}')

