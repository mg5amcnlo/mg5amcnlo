# This file was automatically created by FeynRules $Revision: 161 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (February 19, 2009)
# Date: Thu 20 May 2010 23:00:45


from object_library import all_lorentz, Lorentz


L_1 = Lorentz(name = 'L_1',
              spins = [ 1, 1, 1 ],
              structure = '1')

L_2 = Lorentz(name = 'L_2',
              spins = [ 1, 2, 2 ],
              structure = 'Identity(2,3)')

L_3 = Lorentz(name = 'L_3',
              spins = [ 1, 3, 3 ],
              structure = 'Metric(2,3)')

L_4 = Lorentz(name = 'L_4',
              spins = [ 3, 2, 2 ],
              structure = 'Gamma(1,2,3)')

L_5 = Lorentz(name = 'L_5',
              spins = [ 3, 2, 2 ],
              structure = 'Gamma(1,2,s1)*ProjM(s1,3)')

L_6 = Lorentz(name = 'L_6',
              spins = [ 3, 2, 2 ],
              structure = 'Gamma(1,2,s1)*ProjM(s1,3) - 2*Gamma(1,2,s1)*ProjP(s1,3)')

L_7 = Lorentz(name = 'L_7',
              spins = [ 3, 2, 2 ],
              structure = 'Gamma(1,2,s1)*ProjM(s1,3) + 2*Gamma(1,2,s1)*ProjP(s1,3)')

L_8 = Lorentz(name = 'L_8',
              spins = [ 3, 2, 2 ],
              structure = 'Gamma(1,2,s1)*ProjM(s1,3) + 4*Gamma(1,2,s1)*ProjP(s1,3)')

L_9 = Lorentz(name = 'L_9',
              spins = [ 3, 3, 3 ],
              structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

L_10 = Lorentz(name = 'L_10',
               spins = [ 1, 1, 1, 1 ],
               structure = '1')

L_11 = Lorentz(name = 'L_11',
               spins = [ 1, 1, 3, 3 ],
               structure = 'Metric(3,4)')

L_12 = Lorentz(name = 'L_12',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

L_13 = Lorentz(name = 'L_13',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

L_14 = Lorentz(name = 'L_14',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

L_15 = Lorentz(name = 'L_15',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

L_16 = Lorentz(name = 'L_16',
               spins = [ 3, 3, 3, 3 ],
               structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

