# This file was automatically created by FeynRules $Revision: 189 $
# Mathematica version: 7.0 for Linux x86 (32-bit) (November 11, 2008)
# Date: Wed 16 Jun 2010 15:21:39


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate



L_1 = Lorentz(name = 'L_1',
              spins = [ 1, 1, 1 ],
              structure = '1')

L_2 = Lorentz(name = 'L_2',
              spins = [ 1, 1, 3 ],
              structure = 'P(3,1) + P(3,3)')

L_3 = Lorentz(name = 'L_3',
              spins = [ 1, 1, 3 ],
              structure = 'P(3,2) + P(3,3)')

L_4 = Lorentz(name = 'L_4',
              spins = [ 1, 1, 1 ],
              structure = '1')

L_5 = Lorentz(name = 'L_5',
              spins = [ 1, 1, 3 ],
              structure = 'P(3,1) - P(3,2)')

L_6 = Lorentz(name = 'L_6',
              spins = [ 1, 2, 2 ],
              structure = 'Gamma(5,2,3)')

L_7 = Lorentz(name = 'L_7',
              spins = [ 1, 2, 2 ],
              structure = 'Identity(2,3)')

L_8 = Lorentz(name = 'L_8',
              spins = [ 1, 2, 2 ],
              structure = 'ProjM(2,3)')

L_9 = Lorentz(name = 'L_9',
              spins = [ 1, 2, 2 ],
              structure = 'ProjP(2,3)')

L_10 = Lorentz(name = 'L_10',
               spins = [ 1, 3, 3 ],
               structure = 'Metric(2,3)')

L_11 = Lorentz(name = 'L_11',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,1) + P(1,2)')

L_12 = Lorentz(name = 'L_12',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

L_13 = Lorentz(name = 'L_13',
               spins = [ 3, 1, 3 ],
               structure = 'Metric(1,3)')

L_14 = Lorentz(name = 'L_14',
               spins = [ 3, 2, 2 ],
               structure = 'Gamma(1,2,3)')

L_15 = Lorentz(name = 'L_15',
               spins = [ 3, 2, 2 ],
               structure = 'Gamma(1,2,\'s1\')*ProjM(\'s1\',3)')

L_16 = Lorentz(name = 'L_16',
               spins = [ 3, 2, 2 ],
               structure = 'Gamma(1,2,\'s1\')*ProjM(\'s1\',3) - 2*Gamma(1,2,\'s1\')*ProjP(\'s1\',3)')

L_17 = Lorentz(name = 'L_17',
               spins = [ 3, 2, 2 ],
               structure = 'Gamma(1,2,\'s1\')*ProjM(\'s1\',3) + 2*Gamma(1,2,\'s1\')*ProjP(\'s1\',3)')

L_18 = Lorentz(name = 'L_18',
               spins = [ 3, 2, 2 ],
               structure = 'Gamma(1,2,\'s1\')*ProjM(\'s1\',3) + 4*Gamma(1,2,\'s1\')*ProjP(\'s1\',3)')

L_19 = Lorentz(name = 'L_19',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

L_20 = Lorentz(name = 'L_20',
               spins = [ 1, 1, 1, 1 ],
               structure = '1')

L_21 = Lorentz(name = 'L_21',
               spins = [ 1, 1, 3, 3 ],
               structure = 'Metric(3,4)')

L_22 = Lorentz(name = 'L_22',
               spins = [ 3, 1, 1, 3 ],
               structure = 'Metric(1,4)')

L_23 = Lorentz(name = 'L_23',
               spins = [ 3, 3, 1, 1 ],
               structure = 'Metric(1,2)')

L_24 = Lorentz(name = 'L_24',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

L_25 = Lorentz(name = 'L_25',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

L_26 = Lorentz(name = 'L_26',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

L_27 = Lorentz(name = 'L_27',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

L_28 = Lorentz(name = 'L_28',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

