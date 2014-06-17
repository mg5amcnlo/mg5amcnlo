# This file was automatically created by FeynRules 2.0.25
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Thu 8 May 2014 12:30:33


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot


FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1)')

#FFSS1 = Lorentz(name = 'FFSS1',
#                spins = [ 2, 2, 1, 1 ],
#                structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSS1 = Lorentz(name = 'FFSS1',
                spins = [ 2, 2, 1, 1 ],
                structure = 'FFCT2((P(-3,3)+P(-3,4))*(P(-3,3)+P(-3,4))) *(P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1))')

FFFF1 = Lorentz(name = 'FFFF1',
                spins = [ 2, 2, 2, 2 ],
                structure = 'Gamma(-1,2,-2)*Gamma(-1,4,-3)*ProjM(-3,3)*ProjM(-2,1)')

FFSSS1 = Lorentz(name = 'FFSSS1',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1)')

#FFSSS2 = Lorentz(name = 'FFSSS2',
#                 spins = [ 2, 2, 1, 1, 1 ],
#                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1)')
#
#FFSSS3 = Lorentz(name = 'FFSSS3',
#                 spins = [ 2, 2, 1, 1, 1 ],
#                 structure = 'P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1)')
#
#FFSSS4 = Lorentz(name = 'FFSSS4',
#                 spins = [ 2, 2, 1, 1, 1 ],
#                 structure = 'P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1)')

FFSSS2 = Lorentz(name = 'FFSSS2',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'FFCT3((P(-3,3)+P(-3,4)+P(-3,5))*(P(-3,3)+P(-3,4)+P(-3,5))) *FFCT3F1((P(-4,3)+P(-4,5))*(P(-4,3)+P(-4,5))) *(P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1))')

FFSSS3 = Lorentz(name = 'FFSSS3',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'FFCT3((P(-3,3)+P(-3,4)+P(-3,5))*(P(-3,3)+P(-3,4)+P(-3,5))) *FFCT3F1((P(-4,4)+P(-4,5))*(P(-4,4)+P(-4,5))) *(P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1))')

FFSSS4 = Lorentz(name = 'FFSSS4',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = '0.5 *FFCT3((P(-3,3)+P(-3,4)+P(-3,5))*(P(-3,3)+P(-3,4)+P(-3,5))) *( FFCT3F1((P(-4,3)+P(-4,5))*(P(-4,3)+P(-4,5)))*((P(-5,3)+P(-5,4)+P(-5,5))*(P(-5,3)-P(-5,5)))/((P(-6,3)+P(-6,4)+P(-6,5))*(P(-6,3)+P(-6,4)+P(-6,5))) + FFCT3F1((P(-7,4)+P(-7,5))*(P(-7,4)+P(-7,5)))*((P(-8,3)+P(-8,4)+P(-8,5))*(P(-8,4)-P(-8,5)))/((P(-9,3)+P(-9,4)+P(-9,5))*(P(-9,3)+P(-9,4)+P(-9,5))) ) *(P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1))')

FFSSS5 = Lorentz(name = 'FFSSS5',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'FFCT3((P(-3,3)+P(-3,4)+P(-3,5))*(P(-3,3)+P(-3,4)+P(-3,5))) *FFCT3F0((P(-4,3)+P(-4,5))*(P(-4,3)+P(-4,5))) *(P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1))')

FFSSS6 = Lorentz(name = 'FFSSS6',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = 'FFCT3((P(-3,3)+P(-3,4)+P(-3,5))*(P(-3,3)+P(-3,4)+P(-3,5))) *FFCT3F0((P(-4,4)+P(-4,5))*(P(-4,4)+P(-4,5))) *(P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1))')

FFSSS7 = Lorentz(name = 'FFSSS7',
                 spins = [ 2, 2, 1, 1, 1 ],
                 structure = '0.5 *FFCT3((P(-3,3)+P(-3,4)+P(-3,5))*(P(-3,3)+P(-3,4)+P(-3,5))) *( FFCT3F0((P(-4,3)+P(-4,5))*(P(-4,3)+P(-4,5)))*((P(-5,3)+P(-5,4)+P(-5,5))*(P(-5,3)-P(-5,5)))/((P(-6,3)+P(-6,4)+P(-6,5))*(P(-6,3)+P(-6,4)+P(-6,5))) + FFCT3F0((P(-7,4)+P(-7,5))*(P(-7,4)+P(-7,5)))*((P(-8,3)+P(-8,4)+P(-8,5))*(P(-8,4)-P(-8,5)))/((P(-9,3)+P(-9,4)+P(-9,5))*(P(-9,3)+P(-9,4)+P(-9,5))) ) *(P(-1,3)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,5)*Gamma(-1,2,-2)*ProjM(-2,1))')

