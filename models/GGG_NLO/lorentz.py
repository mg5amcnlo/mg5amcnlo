# This file was automatically created by FeynRules 2.4.54
# Mathematica version: 11.0.0 for Linux x86 (64-bit) (July 28, 2016)
# Date: Tue 25 Oct 2016 14:05:34


from object_library import all_lorentz, Lorentz

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot
try:
   import form_factors as ForFac 
except ImportError:
   pass


FF1 = Lorentz(name = 'FF1',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)*Gamma(-1,2,1)')

FF2 = Lorentz(name = 'FF2',
              spins = [ 2, 2 ],
              structure = 'ProjM(2,1) + ProjP(2,1)')

FF3 = Lorentz(name = 'FF3',
              spins = [ 2, 2 ],
              structure = 'P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

FF4 = Lorentz(name = 'FF4',
              spins = [ 2, 2 ],
              structure = '-(P(-1,1)*Gamma(-1,2,1)) + P(-1,1)*Gamma(-1,2,-2)*ProjM(-2,1) + P(-1,1)*Gamma(-1,2,-2)*ProjP(-2,1)')

VV1 = Lorentz(name = 'VV1',
              spins = [ 3, 3 ],
              structure = 'P(1,2)*P(2,2)')

VV2 = Lorentz(name = 'VV2',
              spins = [ 3, 3 ],
              structure = 'Metric(1,2)')

VV3 = Lorentz(name = 'VV3',
              spins = [ 3, 3 ],
              structure = 'P(-1,2)**2*Metric(1,2)')

VV4 = Lorentz(name = 'VV4',
              spins = [ 3, 3 ],
              structure = 'P(1,2)*P(2,2) - (3*P(-1,2)**2*Metric(1,2))/2.')

VV5 = Lorentz(name = 'VV5',
              spins = [ 3, 3 ],
              structure = 'P(1,2)*P(2,2) - P(-1,2)**2*Metric(1,2)')

VV6 = Lorentz(name = 'VV6',
              spins = [ 3, 3 ],
              structure = 'P(-1,2)**2*P(1,2)*P(2,2) - P(-2,2)**2*P(-1,2)**2*Metric(1,2)')

VV7 = Lorentz(name = 'VV7',
              spins = [ 3, 3 ],
              structure = 'P(-1,2)**2*P(1,2)*P(2,2) - (57*P(-2,2)**2*P(-1,2)**2*Metric(1,2))/58.')

UUS1 = Lorentz(name = 'UUS1',
               spins = [ -1, -1, 1 ],
               structure = '1')

UUV1 = Lorentz(name = 'UUV1',
               spins = [ -1, -1, 3 ],
               structure = 'P(3,2) + P(3,3)')

SSS1 = Lorentz(name = 'SSS1',
               spins = [ 1, 1, 1 ],
               structure = '1')

FFS1 = Lorentz(name = 'FFS1',
               spins = [ 2, 2, 1 ],
               structure = 'Gamma5(2,1)')

FFS2 = Lorentz(name = 'FFS2',
               spins = [ 2, 2, 1 ],
               structure = 'Identity(2,1)')

FFS3 = Lorentz(name = 'FFS3',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1)')

FFS4 = Lorentz(name = 'FFS4',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) - ProjP(2,1)')

FFS5 = Lorentz(name = 'FFS5',
               spins = [ 2, 2, 1 ],
               structure = 'ProjP(2,1)')

FFS6 = Lorentz(name = 'FFS6',
               spins = [ 2, 2, 1 ],
               structure = 'ProjM(2,1) + ProjP(2,1)')

FFV1 = Lorentz(name = 'FFV1',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1)')

FFV2 = Lorentz(name = 'FFV2',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFV3 = Lorentz(name = 'FFV3',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFV4 = Lorentz(name = 'FFV4',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1) - Gamma(3,2,-1)*ProjM(-1,1) - Gamma(3,2,-1)*ProjP(-1,1)')

FFV5 = Lorentz(name = 'FFV5',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFV6 = Lorentz(name = 'FFV6',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,1) + 2*Gamma(3,2,-1)*ProjM(-1,1) + 2*Gamma(3,2,-1)*ProjP(-1,1)')

FFV7 = Lorentz(name = 'FFV7',
               spins = [ 2, 2, 3 ],
               structure = 'Gamma(3,2,-1)*ProjM(-1,1) + 4*Gamma(3,2,-1)*ProjP(-1,1)')

FFV8 = Lorentz(name = 'FFV8',
               spins = [ 2, 2, 3 ],
               structure = '(-2*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. - (2*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11. - (10*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11. + P(3,1)*ProjM(2,1) - P(3,2)*ProjM(2,1) - (2*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (8*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. - (2*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11. - (10*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11. + P(3,1)*ProjP(2,1) - P(3,2)*ProjP(2,1)')

FFV9 = Lorentz(name = 'FFV9',
               spins = [ 2, 2, 3 ],
               structure = '-(P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/19. + (10*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/19. - (P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/19. - (11*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/19. + P(3,1)*ProjM(2,1) - P(3,2)*ProjM(2,1) - (P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/19. + (10*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/19. - (P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/19. - (11*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/19. + P(3,1)*ProjP(2,1) - P(3,2)*ProjP(2,1)')

FFV10 = Lorentz(name = 'FFV10',
                spins = [ 2, 2, 3 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFV11 = Lorentz(name = 'FFV11',
                spins = [ 2, 2, 3 ],
                structure = '-3*P(-1,3)*P(3,1)*Gamma(-1,2,1) - 3*P(-1,3)*P(3,2)*Gamma(-1,2,1) - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1) - 3*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1) - 3*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV12 = Lorentz(name = 'FFV12',
                spins = [ 2, 2, 3 ],
                structure = '-3*P(-1,3)*P(3,2)*Gamma(-1,2,1) - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1) - 3*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) - 3*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1) - 3*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) - 3*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV13 = Lorentz(name = 'FFV13',
                spins = [ 2, 2, 3 ],
                structure = '(-7*P(-1,3)*P(3,1)*Gamma(-1,2,1))/3. - (10*P(-1,3)*P(3,2)*Gamma(-1,2,1))/3. - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1))/18. + (29*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1))/18. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1))/18. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) - P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) + (23*P(-1,1)**2*Gamma(3,2,-2)*ProjM(-2,1))/9. + (7*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjM(-2,1))/3. - (7*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1))/3. - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1))/18. + (29*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1))/18. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1))/18. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) - P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) + (23*P(-1,1)**2*Gamma(3,2,-2)*ProjP(-2,1))/9. + (7*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjP(-2,1))/3. - (7*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1))/3.')

FFV14 = Lorentz(name = 'FFV14',
                spins = [ 2, 2, 3 ],
                structure = '(-10*P(-1,3)*P(3,2)*Gamma(-1,2,1))/3. - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1))/18. + (29*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1))/18. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1))/18. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) - (7*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1))/3. - P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) + (23*P(-1,1)**2*Gamma(3,2,-2)*ProjM(-2,1))/9. + (7*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjM(-2,1))/3. - (7*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1))/3. - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1))/18. + (29*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1))/18. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1))/18. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) - (7*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1))/3. - P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) + (23*P(-1,1)**2*Gamma(3,2,-2)*ProjP(-2,1))/9. + (7*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjP(-2,1))/3. - (7*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1))/3.')

FFV15 = Lorentz(name = 'FFV15',
                spins = [ 2, 2, 3 ],
                structure = '-6*P(-1,3)*P(3,1)*Gamma(-1,2,1) - 6*P(-1,3)*P(3,2)*Gamma(-1,2,1) - (P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjM(-3,1))/2. - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1) - (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1) + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1))/2. - 6*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - (P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjP(-3,1))/2. - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1) - (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1) + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1))/2. - 6*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV16 = Lorentz(name = 'FFV16',
                spins = [ 2, 2, 3 ],
                structure = '-6*P(-1,3)*P(3,2)*Gamma(-1,2,1) - (P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjM(-3,1))/2. - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1) - (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1) + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1))/2. - 6*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) - 6*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - (P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjP(-3,1))/2. - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1) - (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1) + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1))/2. - 6*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) - 6*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV17 = Lorentz(name = 'FFV17',
                spins = [ 2, 2, 3 ],
                structure = '-12*P(-1,3)*P(3,1)*Gamma(-1,2,1) - 12*P(-1,3)*P(3,2)*Gamma(-1,2,1) - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjM(-3,1) - 2*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1) - P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1) + 2*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1) - 12*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjP(-3,1) - 2*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1) - P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1) + 2*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1) - 12*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV18 = Lorentz(name = 'FFV18',
                spins = [ 2, 2, 3 ],
                structure = '-51*P(-1,3)*P(3,1)*Gamma(-1,2,1) - 54*P(-1,3)*P(3,2)*Gamma(-1,2,1) - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjM(-3,1))/2. - (31*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + (37*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1))/2. + (5*P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1))/2. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) - 3*P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) + 26*P(-1,1)**2*Gamma(3,2,-2)*ProjM(-2,1) + 19*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjM(-2,1) - 54*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjP(-3,1))/2. - (31*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + (37*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1))/2. + (5*P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1))/2. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) - 3*P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) + 26*P(-1,1)**2*Gamma(3,2,-2)*ProjP(-2,1) + 19*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjP(-2,1) - 54*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV19 = Lorentz(name = 'FFV19',
                spins = [ 2, 2, 3 ],
                structure = '-54*P(-1,3)*P(3,2)*Gamma(-1,2,1) - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjM(-3,1))/2. - (31*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1))/2. + (37*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1))/2. + (5*P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1))/2. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) - 51*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) - 3*P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjM(-2,1) + 26*P(-1,1)**2*Gamma(3,2,-2)*ProjM(-2,1) + 19*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjM(-2,1) - 54*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - (3*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjP(-3,1))/2. - (31*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + (P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1))/2. + (37*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1))/2. + (5*P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1))/2. + P(-1,1)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) - 51*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) - 3*P(-1,1)*P(3,2)*Gamma(-1,2,-2)*ProjP(-2,1) + 26*P(-1,1)**2*Gamma(3,2,-2)*ProjP(-2,1) + 19*P(-1,1)*P(-1,3)*Gamma(3,2,-2)*ProjP(-2,1) - 54*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

FFV20 = Lorentz(name = 'FFV20',
                spins = [ 2, 2, 3 ],
                structure = '12*P(-1,3)*P(3,2)*Gamma(-1,2,1) + P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjM(-3,1) - P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,2,-5)*Gamma(3,-5,-4)*ProjM(-3,1) + 2*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjM(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjM(-3,1) - 2*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjM(-3,1) - P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjM(-3,1) + 12*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjM(-2,1) + 12*P(-1,3)**2*Gamma(3,2,-2)*ProjM(-2,1) - P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-3)*Gamma(3,-3,-5)*ProjP(-4,1) + P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-4,-3)*Gamma(3,-5,-4)*ProjP(-3,1) + 2*P(-2,3)*P(-1,1)*Gamma(-2,2,-5)*Gamma(-1,-5,-4)*Gamma(3,-4,-3)*ProjP(-3,1) + P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,2,-5)*Gamma(3,-4,-3)*ProjP(-3,1) - 2*P(-2,3)*P(-1,1)*Gamma(-2,-4,-3)*Gamma(-1,-5,-4)*Gamma(3,2,-5)*ProjP(-3,1) - P(-2,3)*P(-1,1)*Gamma(-2,-5,-4)*Gamma(-1,-4,-3)*Gamma(3,2,-5)*ProjP(-3,1) + 12*P(-1,3)*P(3,1)*Gamma(-1,2,-2)*ProjP(-2,1) + 12*P(-1,3)**2*Gamma(3,2,-2)*ProjP(-2,1)')

VSS1 = Lorentz(name = 'VSS1',
               spins = [ 3, 1, 1 ],
               structure = 'P(1,2) - P(1,3)')

VVS1 = Lorentz(name = 'VVS1',
               spins = [ 3, 3, 1 ],
               structure = 'Metric(1,2)')

VVV1 = Lorentz(name = 'VVV1',
               spins = [ 3, 3, 3 ],
               structure = '-(Epsilon(1,2,3,-1)*P(-1,1)) + Epsilon(1,2,3,-1)*P(-1,2)')

VVV2 = Lorentz(name = 'VVV2',
               spins = [ 3, 3, 3 ],
               structure = 'P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) + P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)')

VVV3 = Lorentz(name = 'VVV3',
               spins = [ 3, 3, 3 ],
               structure = '-(P(1,2)*P(2,3)*P(3,1)) + P(1,3)*P(2,1)*P(3,2) + P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2) - P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2) - P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3) + P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3) + P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3) - P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3)')

VVV4 = Lorentz(name = 'VVV4',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,1)*P(2,1)*P(3,1) + 3*P(1,3)*P(2,1)*P(3,1) + P(1,3)*P(2,2)*P(3,1) - 3*P(1,2)*P(2,3)*P(3,1) + 2*P(1,1)*P(2,1)*P(3,2) + 3*P(1,2)*P(2,1)*P(3,2) + 6*P(1,3)*P(2,1)*P(3,2) + P(1,3)*P(2,2)*P(3,2) + P(1,1)*P(2,3)*P(3,2) + 3*P(1,3)*P(2,3)*P(3,2) - P(-1,1)**2*P(3,1)*Metric(1,2) - 2*P(-1,2)**2*P(3,1)*Metric(1,2) - P(-1,1)*P(-1,3)*P(3,1)*Metric(1,2) - P(-1,3)**2*P(3,1)*Metric(1,2) + P(-1,1)**2*P(3,2)*Metric(1,2) - P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2) + P(-1,3)**2*P(3,2)*Metric(1,2) - P(-1,2)**2*P(3,3)*Metric(1,2) - P(-1,2)*P(-1,3)*P(3,3)*Metric(1,2) + P(-1,2)**2*P(2,1)*Metric(1,3) - P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3) + P(-1,3)**2*P(2,1)*Metric(1,3) - P(-1,1)**2*P(2,2)*Metric(1,3) - P(-1,1)*P(-1,2)*P(2,2)*Metric(1,3) - 2*P(-1,1)**2*P(2,3)*Metric(1,3) - P(-1,2)**2*P(2,3)*Metric(1,3) - P(-1,2)*P(-1,3)*P(2,3)*Metric(1,3) - P(-1,3)**2*P(2,3)*Metric(1,3) - P(-1,1)*P(-1,3)*P(1,1)*Metric(2,3) - P(-1,3)**2*P(1,1)*Metric(2,3) - P(-1,1)**2*P(1,2)*Metric(2,3) - P(-1,1)*P(-1,2)*P(1,2)*Metric(2,3) - P(-1,2)**2*P(1,2)*Metric(2,3) - 2*P(-1,3)**2*P(1,2)*Metric(2,3) + P(-1,1)**2*P(1,3)*Metric(2,3) - P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3) + P(-1,2)**2*P(1,3)*Metric(2,3)')

VVV5 = Lorentz(name = 'VVV5',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,1)*P(2,2)*P(3,1) + 2*P(1,2)*P(2,2)*P(3,1) + 2*P(1,1)*P(2,3)*P(3,1) - 2*P(1,1)*P(2,1)*P(3,2) - P(1,1)*P(2,2)*P(3,2) - 2*P(1,3)*P(2,2)*P(3,2) - P(1,1)*P(2,1)*P(3,3) - 2*P(1,3)*P(2,1)*P(3,3) + P(1,2)*P(2,2)*P(3,3) - P(1,3)*P(2,2)*P(3,3) + P(1,1)*P(2,3)*P(3,3) + 2*P(1,2)*P(2,3)*P(3,3) - 2*P(-1,2)**2*P(3,1)*Metric(1,2) - P(-1,3)**2*P(3,1)*Metric(1,2) + 2*P(-1,1)**2*P(3,2)*Metric(1,2) + P(-1,3)**2*P(3,2)*Metric(1,2) + P(-1,1)**2*P(3,3)*Metric(1,2) - P(-1,2)**2*P(3,3)*Metric(1,2) + P(-1,1)*P(-1,3)*P(3,3)*Metric(1,2) - P(-1,2)*P(-1,3)*P(3,3)*Metric(1,2) + P(-1,2)**2*P(2,1)*Metric(1,3) + 2*P(-1,3)**2*P(2,1)*Metric(1,3) - P(-1,1)**2*P(2,2)*Metric(1,3) - P(-1,1)*P(-1,2)*P(2,2)*Metric(1,3) + P(-1,2)*P(-1,3)*P(2,2)*Metric(1,3) + P(-1,3)**2*P(2,2)*Metric(1,3) - 2*P(-1,1)**2*P(2,3)*Metric(1,3) - P(-1,2)**2*P(2,3)*Metric(1,3) + P(-1,1)*P(-1,2)*P(1,1)*Metric(2,3) + P(-1,2)**2*P(1,1)*Metric(2,3) - P(-1,1)*P(-1,3)*P(1,1)*Metric(2,3) - P(-1,3)**2*P(1,1)*Metric(2,3) - P(-1,1)**2*P(1,2)*Metric(2,3) - 2*P(-1,3)**2*P(1,2)*Metric(2,3) + P(-1,1)**2*P(1,3)*Metric(2,3) + 2*P(-1,2)**2*P(1,3)*Metric(2,3)')

VVV6 = Lorentz(name = 'VVV6',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,2)*P(2,1)*P(3,1) - P(1,3)*P(2,1)*P(3,1) + (13*P(1,2)*P(2,3)*P(3,1))/6. + P(1,3)*P(2,3)*P(3,1) - P(1,2)*P(2,1)*P(3,2) - (13*P(1,3)*P(2,1)*P(3,2))/6. + P(1,2)*P(2,3)*P(3,2) - P(1,3)*P(2,3)*P(3,2) - (7*P(-1,1)*P(-1,2)*P(3,1)*Metric(1,2))/9. + (14*P(-1,2)**2*P(3,1)*Metric(1,2))/9. - (P(-1,1)*P(-1,3)*P(3,1)*Metric(1,2))/9. + (11*P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2))/18. + (11*P(-1,3)**2*P(3,1)*Metric(1,2))/9. + (2*P(-1,1)*P(-1,2)*P(3,2)*Metric(1,2))/3. - (5*P(-1,2)**2*P(3,2)*Metric(1,2))/3. + (5*P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2))/6. - (5*P(-1,2)*P(-1,3)*P(3,2)*Metric(1,2))/3. - (4*P(-1,3)**2*P(3,2)*Metric(1,2))/3. + (P(-1,1)*P(-1,2)*P(2,1)*Metric(1,3))/9. - (11*P(-1,2)**2*P(2,1)*Metric(1,3))/9. + (7*P(-1,1)*P(-1,3)*P(2,1)*Metric(1,3))/9. - (11*P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3))/18. - (14*P(-1,3)**2*P(2,1)*Metric(1,3))/9. - (5*P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3))/6. + (4*P(-1,2)**2*P(2,3)*Metric(1,3))/3. - (2*P(-1,1)*P(-1,3)*P(2,3)*Metric(1,3))/3. + (5*P(-1,2)*P(-1,3)*P(2,3)*Metric(1,3))/3. + (5*P(-1,3)**2*P(2,3)*Metric(1,3))/3. + (4*P(-1,2)**2*P(1,2)*Metric(2,3))/3. - (5*P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3))/6. + (P(-1,2)*P(-1,3)*P(1,2)*Metric(2,3))/3. + (4*P(-1,3)**2*P(1,2)*Metric(2,3))/3. + (5*P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3))/6. - (4*P(-1,2)**2*P(1,3)*Metric(2,3))/3. - (P(-1,2)*P(-1,3)*P(1,3)*Metric(2,3))/3. - (4*P(-1,3)**2*P(1,3)*Metric(2,3))/3.')

VVV7 = Lorentz(name = 'VVV7',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,2)*P(2,1)*P(3,1) - P(1,3)*P(2,1)*P(3,1) + (55*P(1,2)*P(2,3)*P(3,1))/12. + P(1,3)*P(2,3)*P(3,1) - P(1,2)*P(2,1)*P(3,2) - (55*P(1,3)*P(2,1)*P(3,2))/12. + P(1,2)*P(2,3)*P(3,2) - P(1,3)*P(2,3)*P(3,2) - (11*P(-1,1)*P(-1,2)*P(3,1)*Metric(1,2))/12. + (49*P(-1,2)**2*P(3,1)*Metric(1,2))/36. - (7*P(-1,1)*P(-1,3)*P(3,1)*Metric(1,2))/18. - (41*P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2))/18. + (8*P(-1,3)**2*P(3,1)*Metric(1,2))/9. + (29*P(-1,1)*P(-1,2)*P(3,2)*Metric(1,2))/36. - (53*P(-1,2)**2*P(3,2)*Metric(1,2))/36. + (95*P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2))/36. - (25*P(-1,2)*P(-1,3)*P(3,2)*Metric(1,2))/12. - (17*P(-1,3)**2*P(3,2)*Metric(1,2))/9. + (7*P(-1,1)*P(-1,2)*P(2,1)*Metric(1,3))/18. - (8*P(-1,2)**2*P(2,1)*Metric(1,3))/9. + (11*P(-1,1)*P(-1,3)*P(2,1)*Metric(1,3))/12. + (41*P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3))/18. - (49*P(-1,3)**2*P(2,1)*Metric(1,3))/36. - (95*P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3))/36. + (17*P(-1,2)**2*P(2,3)*Metric(1,3))/9. - (29*P(-1,1)*P(-1,3)*P(2,3)*Metric(1,3))/36. + (25*P(-1,2)*P(-1,3)*P(2,3)*Metric(1,3))/12. + (53*P(-1,3)**2*P(2,3)*Metric(1,3))/36. + (23*P(-1,2)**2*P(1,2)*Metric(2,3))/18. - (25*P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3))/9. + (3*P(-1,2)*P(-1,3)*P(1,2)*Metric(2,3))/4. + (7*P(-1,3)**2*P(1,2)*Metric(2,3))/4. + (25*P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3))/9. - (7*P(-1,2)**2*P(1,3)*Metric(2,3))/4. - (3*P(-1,2)*P(-1,3)*P(1,3)*Metric(2,3))/4. - (23*P(-1,3)**2*P(1,3)*Metric(2,3))/18.')

VVV8 = Lorentz(name = 'VVV8',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,2)*P(2,1)*P(3,1) - P(1,3)*P(2,1)*P(3,1) + (34*P(1,2)*P(2,3)*P(3,1))/11. + P(1,3)*P(2,3)*P(3,1) - P(1,2)*P(2,1)*P(3,2) - (34*P(1,3)*P(2,1)*P(3,2))/11. + P(1,2)*P(2,3)*P(3,2) - P(1,3)*P(2,3)*P(3,2) + (P(-1,1)**2*P(3,1)*Metric(1,2))/3. - (29*P(-1,1)*P(-1,2)*P(3,1)*Metric(1,2))/44. + (521*P(-1,2)**2*P(3,1)*Metric(1,2))/660. - (P(-1,1)*P(-1,3)*P(3,1)*Metric(1,2))/66. - (1097*P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2))/660. + (53*P(-1,3)**2*P(3,1)*Metric(1,2))/110. - (P(-1,1)**2*P(3,2)*Metric(1,2))/3. + (67*P(-1,1)*P(-1,2)*P(3,2)*Metric(1,2))/132. - (207*P(-1,2)**2*P(3,2)*Metric(1,2))/220. + (229*P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2))/132. - (161*P(-1,2)*P(-1,3)*P(3,2)*Metric(1,2))/165. - (571*P(-1,3)**2*P(3,2)*Metric(1,2))/660. - (P(-1,1)**2*P(2,1)*Metric(1,3))/3. + (P(-1,1)*P(-1,2)*P(2,1)*Metric(1,3))/66. - (53*P(-1,2)**2*P(2,1)*Metric(1,3))/110. + (29*P(-1,1)*P(-1,3)*P(2,1)*Metric(1,3))/44. + (1097*P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3))/660. - (521*P(-1,3)**2*P(2,1)*Metric(1,3))/660. + (P(-1,1)**2*P(2,3)*Metric(1,3))/3. - (229*P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3))/132. + (571*P(-1,2)**2*P(2,3)*Metric(1,3))/660. - (67*P(-1,1)*P(-1,3)*P(2,3)*Metric(1,3))/132. + (161*P(-1,2)*P(-1,3)*P(2,3)*Metric(1,3))/165. + (207*P(-1,3)**2*P(2,3)*Metric(1,3))/220. + (P(-1,1)**2*P(1,2)*Metric(2,3))/3. + (13*P(-1,1)*P(-1,2)*P(1,2)*Metric(2,3))/33. + (49*P(-1,2)**2*P(1,2)*Metric(2,3))/55. - (61*P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3))/33. - (23*P(-1,2)*P(-1,3)*P(1,2)*Metric(2,3))/165. + (124*P(-1,3)**2*P(1,2)*Metric(2,3))/165. - (P(-1,1)**2*P(1,3)*Metric(2,3))/3. + (61*P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3))/33. - (124*P(-1,2)**2*P(1,3)*Metric(2,3))/165. - (13*P(-1,1)*P(-1,3)*P(1,3)*Metric(2,3))/33. + (23*P(-1,2)*P(-1,3)*P(1,3)*Metric(2,3))/165. - (49*P(-1,3)**2*P(1,3)*Metric(2,3))/55.')

VVV9 = Lorentz(name = 'VVV9',
               spins = [ 3, 3, 3 ],
               structure = 'P(1,1)*P(2,1)*P(3,1) - (P(1,2)*P(2,1)*P(3,1))/19. + (58*P(1,3)*P(2,1)*P(3,1))/19. + P(1,3)*P(2,2)*P(3,1) - (60*P(1,2)*P(2,3)*P(3,1))/19. - (P(1,3)*P(2,3)*P(3,1))/19. + 2*P(1,1)*P(2,1)*P(3,2) + (58*P(1,2)*P(2,1)*P(3,2))/19. + (117*P(1,3)*P(2,1)*P(3,2))/19. + P(1,3)*P(2,2)*P(3,2) + P(1,1)*P(2,3)*P(3,2) - (P(1,2)*P(2,3)*P(3,2))/19. + (58*P(1,3)*P(2,3)*P(3,2))/19. - (14*P(-1,1)**2*P(3,1)*Metric(1,2))/19. + (16*P(-1,1)*P(-1,2)*P(3,1)*Metric(1,2))/57. - (92*P(-1,2)**2*P(3,1)*Metric(1,2))/57. - (23*P(-1,1)*P(-1,3)*P(3,1)*Metric(1,2))/57. + (26*P(-1,2)*P(-1,3)*P(3,1)*Metric(1,2))/19. - (17*P(-1,3)**2*P(3,1)*Metric(1,2))/57. + (14*P(-1,1)**2*P(3,2)*Metric(1,2))/19. - (3*P(-1,1)*P(-1,2)*P(3,2)*Metric(1,2))/19. - (5*P(-1,2)**2*P(3,2)*Metric(1,2))/19. - (74*P(-1,1)*P(-1,3)*P(3,2)*Metric(1,2))/57. + (20*P(-1,2)*P(-1,3)*P(3,2)*Metric(1,2))/57. + (71*P(-1,3)**2*P(3,2)*Metric(1,2))/57. - P(-1,2)**2*P(3,3)*Metric(1,2) - P(-1,2)*P(-1,3)*P(3,3)*Metric(1,2) - (5*P(-1,1)**2*P(2,1)*Metric(1,3))/19. - (34*P(-1,1)*P(-1,2)*P(2,1)*Metric(1,3))/57. + (17*P(-1,2)**2*P(2,1)*Metric(1,3))/57. - (16*P(-1,1)*P(-1,3)*P(2,1)*Metric(1,3))/57. - (45*P(-1,2)*P(-1,3)*P(2,1)*Metric(1,3))/19. + (35*P(-1,3)**2*P(2,1)*Metric(1,3))/57. - P(-1,1)**2*P(2,2)*Metric(1,3) - P(-1,1)*P(-1,2)*P(2,2)*Metric(1,3) - (33*P(-1,1)**2*P(2,3)*Metric(1,3))/19. + (17*P(-1,1)*P(-1,2)*P(2,3)*Metric(1,3))/57. - (71*P(-1,2)**2*P(2,3)*Metric(1,3))/57. + (3*P(-1,1)*P(-1,3)*P(2,3)*Metric(1,3))/19. - (77*P(-1,2)*P(-1,3)*P(2,3)*Metric(1,3))/57. - (14*P(-1,3)**2*P(2,3)*Metric(1,3))/19. - P(-1,1)*P(-1,3)*P(1,1)*Metric(2,3) - P(-1,3)**2*P(1,1)*Metric(2,3) - (14*P(-1,1)**2*P(1,2)*Metric(2,3))/19. - (18*P(-1,1)*P(-1,2)*P(1,2)*Metric(2,3))/19. - (16*P(-1,2)**2*P(1,2)*Metric(2,3))/19. + (26*P(-1,1)*P(-1,3)*P(1,2)*Metric(2,3))/57. - (17*P(-1,2)*P(-1,3)*P(1,2)*Metric(2,3))/57. - (119*P(-1,3)**2*P(1,2)*Metric(2,3))/57. + (14*P(-1,1)**2*P(1,3)*Metric(2,3))/19. - (83*P(-1,1)*P(-1,2)*P(1,3)*Metric(2,3))/57. + (62*P(-1,2)**2*P(1,3)*Metric(2,3))/57. - (P(-1,1)*P(-1,3)*P(1,3)*Metric(2,3))/19. + (17*P(-1,2)*P(-1,3)*P(1,3)*Metric(2,3))/57. - (3*P(-1,3)**2*P(1,3)*Metric(2,3))/19.')

SSSS1 = Lorentz(name = 'SSSS1',
                spins = [ 1, 1, 1, 1 ],
                structure = '1')

FFVS1 = Lorentz(name = 'FFVS1',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1)')

FFVS2 = Lorentz(name = 'FFVS2',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1)')

FFVS3 = Lorentz(name = 'FFVS3',
                spins = [ 2, 2, 3, 1 ],
                structure = '(-5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. - (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. + P(3,1)*ProjM(2,1) - P(3,2)*ProjM(2,1)')

FFVS4 = Lorentz(name = 'FFVS4',
                spins = [ 2, 2, 3, 1 ],
                structure = '(-25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/79. + (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. - (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. - (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/79. - (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/158. + (19*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/158. + P(3,1)*ProjM(2,1) - P(3,2)*ProjM(2,1)')

FFVS5 = Lorentz(name = 'FFVS5',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjP(-1,1)')

FFVS6 = Lorentz(name = 'FFVS6',
                spins = [ 2, 2, 3, 1 ],
                structure = 'Gamma(3,2,-1)*ProjM(-1,1) + Gamma(3,2,-1)*ProjP(-1,1)')

FFVS7 = Lorentz(name = 'FFVS7',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) - P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS8 = Lorentz(name = 'FFVS8',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS9 = Lorentz(name = 'FFVS9',
                spins = [ 2, 2, 3, 1 ],
                structure = '-(P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1)')

FFVS10 = Lorentz(name = 'FFVS10',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*Identity(2,1) - P(3,2)*Identity(2,1) - (5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. - (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. - (5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22. - (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22.')

FFVS11 = Lorentz(name = 'FFVS11',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*Gamma5(2,1) - P(3,2)*Gamma5(2,1) + (5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. - (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. + (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. + (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/11. + (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. - (5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22. - (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22.')

FFVS12 = Lorentz(name = 'FFVS12',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*Identity(2,1) - P(3,2)*Identity(2,1) - (5*P(-1,1)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjM(-3,1))/11. - (5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/11. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/22. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. - (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/22. - (5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22. - (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22.')

FFVS13 = Lorentz(name = 'FFVS13',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(-5*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/11. + (7*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (9*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/22. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/11. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22. - (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/22. + P(3,1)*ProjP(2,1) - P(3,2)*ProjP(2,1)')

FFVS14 = Lorentz(name = 'FFVS14',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*Gamma5(2,1) - P(3,2)*Gamma5(2,1) + (25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/79. - (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. + (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. + (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/79. + (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/158. - (19*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/158. - (25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/79. + (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/79. - (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158. + (19*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158.')

FFVS15 = Lorentz(name = 'FFVS15',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*Identity(2,1) - P(3,2)*Identity(2,1) - (25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/79. + (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. - (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. - (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/79. - (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/158. + (19*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/158. - (25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/79. + (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/79. - (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158. + (19*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158.')

FFVS16 = Lorentz(name = 'FFVS16',
                 spins = [ 2, 2, 3, 1 ],
                 structure = 'P(3,1)*Identity(2,1) - P(3,2)*Identity(2,1) + (19*P(-1,4)*Gamma(-1,-2,-3)*Gamma(3,2,-2)*ProjM(-3,1))/158. - (25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/79. + (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. - (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjM(-2,1))/158. - (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/79. - (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjM(-2,1))/158. - (25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/79. + (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/79. - (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158. + (19*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158.')

FFVS17 = Lorentz(name = 'FFVS17',
                 spins = [ 2, 2, 3, 1 ],
                 structure = '(-25*P(-1,1)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/79. + (41*P(-1,3)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (69*P(-1,4)*Gamma(-1,2,-3)*Gamma(3,-3,-2)*ProjP(-2,1))/158. - (25*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/79. - (91*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158. + (19*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*ProjP(-2,1))/158. + P(3,1)*ProjP(2,1) - P(3,2)*ProjP(2,1)')

FFVV1 = Lorentz(name = 'FFVV1',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(4,1)*Gamma(3,2,1)')

FFVV2 = Lorentz(name = 'FFVV2',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(4,2)*Gamma(3,2,1)')

FFVV3 = Lorentz(name = 'FFVV3',
                spins = [ 2, 2, 3, 3 ],
                structure = 'Metric(3,4)*ProjM(2,1)')

FFVV4 = Lorentz(name = 'FFVV4',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(4,1)*Gamma(3,2,-1)*ProjM(-1,1)')

FFVV5 = Lorentz(name = 'FFVV5',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(4,2)*Gamma(3,2,-1)*ProjM(-1,1)')

FFVV6 = Lorentz(name = 'FFVV6',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(4,3)*Gamma(3,2,-1)*ProjM(-1,1)')

FFVV7 = Lorentz(name = 'FFVV7',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(3,1)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV8 = Lorentz(name = 'FFVV8',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(3,2)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV9 = Lorentz(name = 'FFVV9',
                spins = [ 2, 2, 3, 3 ],
                structure = 'P(3,4)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV10 = Lorentz(name = 'FFVV10',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV11 = Lorentz(name = 'FFVV11',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1)')

FFVV12 = Lorentz(name = 'FFVV12',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - 10*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV13 = Lorentz(name = 'FFVV13',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + 5*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1)')

FFVV14 = Lorentz(name = 'FFVV14',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV15 = Lorentz(name = 'FFVV15',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV16 = Lorentz(name = 'FFVV16',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV17 = Lorentz(name = 'FFVV17',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)')

FFVV18 = Lorentz(name = 'FFVV18',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)')

FFVV19 = Lorentz(name = 'FFVV19',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)')

FFVV20 = Lorentz(name = 'FFVV20',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1)')

FFVV21 = Lorentz(name = 'FFVV21',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1)')

FFVV22 = Lorentz(name = 'FFVV22',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1)')

FFVV23 = Lorentz(name = 'FFVV23',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1)')

FFVV24 = Lorentz(name = 'FFVV24',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1)')

FFVV25 = Lorentz(name = 'FFVV25',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1)')

FFVV26 = Lorentz(name = 'FFVV26',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1)')

FFVV27 = Lorentz(name = 'FFVV27',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1)')

FFVV28 = Lorentz(name = 'FFVV28',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1)')

FFVV29 = Lorentz(name = 'FFVV29',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV30 = Lorentz(name = 'FFVV30',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV31 = Lorentz(name = 'FFVV31',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1)')

FFVV32 = Lorentz(name = 'FFVV32',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1)')

FFVV33 = Lorentz(name = 'FFVV33',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1)')

FFVV34 = Lorentz(name = 'FFVV34',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1)')

FFVV35 = Lorentz(name = 'FFVV35',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1)')

FFVV36 = Lorentz(name = 'FFVV36',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/2.')

FFVV37 = Lorentz(name = 'FFVV37',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '(-13*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/3. + (13*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/2. - (13*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/3. - (13*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (7*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/6. + (2*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. + (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. - (P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/4. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/12. - (2*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. - (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. - (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/6. - (11*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/12. - (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. + 2*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (35*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/6. + (35*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/6.')

FFVV38 = Lorentz(name = 'FFVV38',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '(-11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. + (3*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (7*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (3*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. + (7*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. + (3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (9*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. + 2*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (7*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/4. + (7*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/4.')

FFVV39 = Lorentz(name = 'FFVV39',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '(-9*P(-1,4)*Gamma(-1,-2,-4)*Gamma(3,-4,-3)*Gamma(4,2,-2)*ProjM(-3,1))/7. - (11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/7. + (3*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/7. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/7. - (11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/7. - 2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) + (2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/7. + (6*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/7. + (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. - (5*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/14. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/14. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/7. - (3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/7. + (3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/7. - (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/7. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/7. + (8*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/7. + (4*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/7. + (4*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/7. + (4*P(4,1)*Gamma(3,2,-1)*ProjM(-1,1))/7. - (4*P(4,2)*Gamma(3,2,-1)*ProjM(-1,1))/7. - P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) + P(3,2)*Gamma(4,2,-1)*ProjM(-1,1)')

FFVV40 = Lorentz(name = 'FFVV40',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Metric(3,4)*ProjP(2,1)')

FFVV41 = Lorentz(name = 'FFVV41',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(4,1)*Gamma(3,2,-1)*ProjP(-1,1)')

FFVV42 = Lorentz(name = 'FFVV42',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(4,2)*Gamma(3,2,-1)*ProjP(-1,1)')

FFVV43 = Lorentz(name = 'FFVV43',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(4,3)*Gamma(3,2,-1)*ProjP(-1,1)')

FFVV44 = Lorentz(name = 'FFVV44',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(3,1)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV45 = Lorentz(name = 'FFVV45',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV46 = Lorentz(name = 'FFVV46',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(3,4)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV47 = Lorentz(name = 'FFVV47',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV48 = Lorentz(name = 'FFVV48',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1)')

FFVV49 = Lorentz(name = 'FFVV49',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - 10*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) - 10*Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV50 = Lorentz(name = 'FFVV50',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - (4*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1))/7. - (4*Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1))/7. + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV51 = Lorentz(name = 'FFVV51',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) - (Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1))/3. - (Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1))/3. + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV52 = Lorentz(name = 'FFVV52',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + (2*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1))/13. + (2*Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1))/13. + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV53 = Lorentz(name = 'FFVV53',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + (3*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1))/7. + (3*Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1))/7. + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV54 = Lorentz(name = 'FFVV54',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) - (Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1))/10.')

FFVV55 = Lorentz(name = 'FFVV55',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + (Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1))/5.')

FFVV56 = Lorentz(name = 'FFVV56',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV57 = Lorentz(name = 'FFVV57',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (62*Metric(3,4)*ProjM(2,1))/41. + Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + (62*Metric(3,4)*ProjP(2,1))/41.')

FFVV58 = Lorentz(name = 'FFVV58',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + (38*Metric(3,4)*ProjM(2,1))/17. + Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1) + (38*Metric(3,4)*ProjP(2,1))/17.')

FFVV59 = Lorentz(name = 'FFVV59',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjM(-1,1) + 5*Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjM(-1,1) + 5*Gamma(3,2,-2)*Gamma(4,-2,-1)*ProjP(-1,1) + Gamma(3,-2,-1)*Gamma(4,2,-2)*ProjP(-1,1)')

FFVV60 = Lorentz(name = 'FFVV60',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV61 = Lorentz(name = 'FFVV61',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV62 = Lorentz(name = 'FFVV62',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV63 = Lorentz(name = 'FFVV63',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV64 = Lorentz(name = 'FFVV64',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV65 = Lorentz(name = 'FFVV65',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV66 = Lorentz(name = 'FFVV66',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV67 = Lorentz(name = 'FFVV67',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV68 = Lorentz(name = 'FFVV68',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV69 = Lorentz(name = 'FFVV69',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV70 = Lorentz(name = 'FFVV70',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV71 = Lorentz(name = 'FFVV71',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV72 = Lorentz(name = 'FFVV72',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV73 = Lorentz(name = 'FFVV73',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV74 = Lorentz(name = 'FFVV74',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV75 = Lorentz(name = 'FFVV75',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV76 = Lorentz(name = 'FFVV76',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV77 = Lorentz(name = 'FFVV77',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1)')

FFVV78 = Lorentz(name = 'FFVV78',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1)')

FFVV79 = Lorentz(name = 'FFVV79',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1)')

FFVV80 = Lorentz(name = 'FFVV80',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1)')

FFVV81 = Lorentz(name = 'FFVV81',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1)) + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV82 = Lorentz(name = 'FFVV82',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) + 2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV83 = Lorentz(name = 'FFVV83',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) - P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV84 = Lorentz(name = 'FFVV84',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) - 4*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + 4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1)')

FFVV85 = Lorentz(name = 'FFVV85',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/2. + 2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) + P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV86 = Lorentz(name = 'FFVV86',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/2. - 4*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + 4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) + 2*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) + 2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV87 = Lorentz(name = 'FFVV87',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) - 2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) - P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1) + 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1) - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - 2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) + 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) + P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV88 = Lorentz(name = 'FFVV88',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) - (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/2. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/2. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/2. - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) - (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/2. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2.')

FFVV89 = Lorentz(name = 'FFVV89',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1)) - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. - (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/2. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/2. - P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/2. + P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/2. - (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/2. - P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/2. - (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/2. + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/2. + P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2.')

FFVV90 = Lorentz(name = 'FFVV90',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1) - P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) + 2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) + P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1) - 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) - P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) + 2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) + P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) - 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) - P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV91 = Lorentz(name = 'FFVV91',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1) - 2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) - P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) + 2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1) - P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1) + P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1) - P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) - 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) + P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1) - P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1) + 2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) + P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) + 2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) + P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) - 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) + P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1) - P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1)')

FFVV92 = Lorentz(name = 'FFVV92',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '(-23*P(-1,3)*Gamma(-1,2,1)*Metric(3,4))/12. - (23*P(-1,4)*Gamma(-1,2,1)*Metric(3,4))/12. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (31*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (53*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/48. + (5*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/6. - (53*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/48. - (31*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/12. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/24. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/6. + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/48. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/8. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/24. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/12. - (23*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/6. + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) + P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (31*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (53*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/48. + (5*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/6. - (53*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/48. - (31*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/8. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/12. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/24. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/6. + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/48. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/8. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/24. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/12. - (23*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/6. + P(4,1)*Gamma(3,2,-1)*ProjP(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) + P(3,1)*Gamma(4,2,-1)*ProjP(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV93 = Lorentz(name = 'FFVV93',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '(-11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. + (3*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (7*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (3*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. + (7*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. + (3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (9*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. + 2*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (7*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/4. + (7*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/4. + (11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/2. - (3*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/2. + (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/2. + (11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/2. + 7*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - 3*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) - (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/4. + (5*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/4. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/4. - (7*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/4. + (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/2. + (3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/2. - (3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/2. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2. + (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2. + (9*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2. - 4*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) - 2*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) - 2*P(4,1)*Gamma(3,2,-1)*ProjP(-1,1) + 2*P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) + (7*P(3,1)*Gamma(4,2,-1)*ProjP(-1,1))/2. - (7*P(3,2)*Gamma(4,2,-1)*ProjP(-1,1))/2.')

FFVV94 = Lorentz(name = 'FFVV94',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '(-13*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/3. + (13*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/2. - (13*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/3. - (13*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (7*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/6. + (2*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. + (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. - (P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/4. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/12. - (2*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. - (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. - (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/6. - (11*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/12. - (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. + 2*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (35*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/6. + (35*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/6. + (26*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/3. - (13*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/3. + 11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + (26*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/3. + 13*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - (7*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/3. - (4*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. - (2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. + (2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. + P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/2. + (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/6. + (4*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/3. + (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/3. - (2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/3. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/3. + (11*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/6. + (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2. - 4*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) - 2*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) - 2*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) - 2*P(4,1)*Gamma(3,2,-1)*ProjP(-1,1) + 2*P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) + (35*P(3,1)*Gamma(4,2,-1)*ProjP(-1,1))/3. - (35*P(3,2)*Gamma(4,2,-1)*ProjP(-1,1))/3.')

FFVV95 = Lorentz(name = 'FFVV95',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(4,1)*Gamma(3,2,1) - P(4,2)*Gamma(3,2,1) - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (31*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (53*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/48. + (5*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/6. - (53*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/48. - (31*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/12. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/24. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/6. + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/48. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/8. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/24. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/12. - (23*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/6. - (23*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/12. - (23*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/12. + P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (31*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (53*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/48. + (5*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/6. - (53*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/48. - (31*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/8. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/12. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/24. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/6. + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/48. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/8. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/24. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/12. - (23*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/6. - (23*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/12. - (23*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/12. + P(3,1)*Gamma(4,2,-1)*ProjP(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV96 = Lorentz(name = 'FFVV96',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(4,1)*Gamma(3,2,1) - P(4,2)*Gamma(3,2,1) - (P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/15. - (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/9. + (77*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/90. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/90. - (P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/15. - (11*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/90. - (7*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/6. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/45. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/6. + (13*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/9. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/90. + (77*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/90. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/45. + (13*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/90. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/6. - (74*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/45. - (37*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/45. - (37*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/45. + P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - (P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/15. - (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/9. + (77*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/90. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/90. - (P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/15. - (11*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/90. - (7*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/6. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/45. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/6. + (13*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/9. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/90. + (77*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/90. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/45. + (13*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/90. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/6. - (74*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/45. - (37*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/45. - (37*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/45. + P(3,1)*Gamma(4,2,-1)*ProjP(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV97 = Lorentz(name = 'FFVV97',
                 spins = [ 2, 2, 3, 3 ],
                 structure = '-(P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/15. - (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/9. + (77*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/90. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/90. - (P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/15. - (11*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/90. - (7*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/6. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/45. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/6. + (13*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/9. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/90. + (77*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/90. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/45. + (13*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/90. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/6. - (74*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/45. - (37*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/45. - (37*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/45. + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) + P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) - (P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/15. - (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/9. + (77*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/90. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/90. - (P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/15. - (11*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/90. - (7*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/6. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/45. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/6. + (13*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/90. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/9. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/90. + (77*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/90. - (31*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/45. + (13*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/90. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/6. - (74*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/45. - (37*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/45. - (37*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/45. + P(4,1)*Gamma(3,2,-1)*ProjP(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) + P(3,1)*Gamma(4,2,-1)*ProjP(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV98 = Lorentz(name = 'FFVV98',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(4,1)*Gamma(3,2,1) - P(4,2)*Gamma(3,2,1) - (6*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/7. + (5*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/14. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/7. - (3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/7. + (3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/7. + (11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/7. + (13*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/7. - (3*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/7. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/7. + (9*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/7. + (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/7. + (11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/7. - (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/7. + 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1) - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/14. - (8*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/7. - (4*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/7. - (4*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/7. - (4*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/7. + (4*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/7. - (6*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/7. + (5*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/14. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/2. + (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/7. - (3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/7. + (3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/7. + (11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/7. + (13*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/7. - (3*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/7. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/7. + (9*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/7. + (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/7. + (11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/7. - (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/7. + 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/2. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/14. - (8*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/7. - (4*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/7. - (4*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/7. - (4*P(3,1)*Gamma(4,2,-1)*ProjP(-1,1))/7. + (4*P(3,2)*Gamma(4,2,-1)*ProjP(-1,1))/7.')

FFVV99 = Lorentz(name = 'FFVV99',
                 spins = [ 2, 2, 3, 3 ],
                 structure = 'P(4,1)*Gamma(3,2,1) - P(4,2)*Gamma(3,2,1) - (4*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/35. - (2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (4*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/35. - (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (26*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/35. + (33*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/35. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/35. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/35. + (9*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/70. + (11*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/70. + (26*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/35. - (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/5. + (39*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/35. + (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/35. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/14. + (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/70. - (12*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/35. - (6*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/35. - (6*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/35. - (6*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/35. + (6*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/35. - (4*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/35. - (2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (4*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/35. - (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (26*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/35. + (33*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/35. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/35. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/35. + (9*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/70. + (11*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/70. + (26*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/35. - (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/5. + (39*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/35. + (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/35. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/14. + (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/70. - (12*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/35. - (6*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/35. - (6*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/35. - (6*P(3,1)*Gamma(4,2,-1)*ProjP(-1,1))/35. + (6*P(3,2)*Gamma(4,2,-1)*ProjP(-1,1))/35.')

FFVV100 = Lorentz(name = 'FFVV100',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '(-13*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/3. + (13*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (11*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/2. - (13*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/3. - (13*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (7*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/6. + (2*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. + (P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. - (P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/3. - (P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/2. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/4. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/12. - (2*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. - (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/3. - (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/6. - (11*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/12. - (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. + 2*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (35*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/6. + (35*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/6. - (52*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/3. + (26*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/3. - 22*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - (52*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/3. - 26*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) + (14*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/3. + (8*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. + (4*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. - (4*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/3. - 2*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) - P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1) - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/3. - (8*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/3. - (4*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/3. + (4*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/3. - (22*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/3. - (11*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/3. - 3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1) + 8*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + 4*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + 4*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + 4*P(4,1)*Gamma(3,2,-1)*ProjP(-1,1) - 4*P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) - (70*P(3,1)*Gamma(4,2,-1)*ProjP(-1,1))/3. + (70*P(3,2)*Gamma(4,2,-1)*ProjP(-1,1))/3.')

FFVV101 = Lorentz(name = 'FFVV101',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '(-11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. + (3*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/4. - (7*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/2. + (3*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/2. + (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. + (7*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. + (3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. - (9*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/4. + 2*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1) + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) - (7*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/4. + (7*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/4. - 11*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) + 3*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - 13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1) - 11*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) - 14*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) + 2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1) + 6*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1) + (7*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/2. - (5*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/2. - (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/2. + (7*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/2. - 5*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) - 3*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) + 3*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1) - 11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1) - 5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1) - 9*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1) + 8*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + 4*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + 4*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1) + 4*P(4,1)*Gamma(3,2,-1)*ProjP(-1,1) - 4*P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) - 7*P(3,1)*Gamma(4,2,-1)*ProjP(-1,1) + 7*P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV102 = Lorentz(name = 'FFVV102',
                  spins = [ 2, 2, 3, 3 ],
                  structure = '(-23*P(-1,3)*Gamma(-1,2,1)*Metric(3,4))/12. - (23*P(-1,4)*Gamma(-1,2,1)*Metric(3,4))/12. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (31*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (53*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/6. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/48. + (5*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/48. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/6. - (53*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/48. - (31*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/8. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/12. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/24. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/6. + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/48. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/8. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/24. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/12. - (23*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/6. + P(4,1)*Gamma(3,2,-1)*ProjM(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjM(-1,1) + P(3,1)*Gamma(4,2,-1)*ProjM(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjM(-1,1) + (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-2,-4)*Gamma(4,2,-3)*ProjP(-4,1))/24. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (31*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (53*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/6. - (17*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/48. + (5*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/48. - (5*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/6. - (53*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/48. - (31*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/8. - (5*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/12. + (P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/24. - (7*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/6. + (5*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/48. - (17*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/48. - (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/8. - (5*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/12. - (23*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/6. + P(4,1)*Gamma(3,2,-1)*ProjP(-1,1) - P(4,2)*Gamma(3,2,-1)*ProjP(-1,1) + P(3,1)*Gamma(4,2,-1)*ProjP(-1,1) - P(3,2)*Gamma(4,2,-1)*ProjP(-1,1)')

FFVV103 = Lorentz(name = 'FFVV103',
                  spins = [ 2, 2, 3, 3 ],
                  structure = 'P(4,1)*Gamma(3,2,1) - P(4,2)*Gamma(3,2,1) - (4*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/35. - (2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (4*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/35. - (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjM(-2,1))/35. + (26*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/35. + (33*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/35. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjM(-2,1))/35. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/35. + (9*P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/70. + (11*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjM(-2,1))/70. + (26*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/35. - (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/5. + (39*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjM(-2,1))/35. + (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/35. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/14. + (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjM(-2,1))/70. - (12*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/35. - (6*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/35. - (6*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjM(-2,1))/35. - (6*P(3,1)*Gamma(4,2,-1)*ProjM(-1,1))/35. + (6*P(3,2)*Gamma(4,2,-1)*ProjM(-1,1))/35. + (9*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-3)*Gamma(4,-2,-4)*ProjP(-4,1))/70. - (4*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (2*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/35. - (2*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-3,-2)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (4*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/35. - (2*P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (2*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,2,-4)*Gamma(4,-4,-3)*ProjP(-2,1))/35. + (26*P(-1,1)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/35. + (33*P(-1,3)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/35. - (13*P(-1,4)*Gamma(-1,2,-4)*Gamma(3,-4,-3)*Gamma(4,-3,-2)*ProjP(-2,1))/35. + (11*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/35. + (11*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,2,-4)*Gamma(4,-3,-2)*ProjP(-2,1))/70. + (26*P(-1,1)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/35. - (P(-1,3)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/5. + (39*P(-1,4)*Gamma(-1,-3,-2)*Gamma(3,-4,-3)*Gamma(4,2,-4)*ProjP(-2,1))/35. + (3*P(-1,1)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/35. + (P(-1,3)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/14. + (3*P(-1,4)*Gamma(-1,-4,-3)*Gamma(3,-3,-2)*Gamma(4,2,-4)*ProjP(-2,1))/70. - (12*P(-1,1)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/35. - (6*P(-1,3)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/35. - (6*P(-1,4)*Gamma(-1,2,-2)*Metric(3,4)*ProjP(-2,1))/35. - (6*P(3,1)*Gamma(4,2,-1)*ProjP(-1,1))/35. + (6*P(3,2)*Gamma(4,2,-1)*ProjP(-1,1))/35.')

VVSS1 = Lorentz(name = 'VVSS1',
                spins = [ 3, 3, 1, 1 ],
                structure = 'Metric(1,2)')

VVVV1 = Lorentz(name = 'VVVV1',
                spins = [ 3, 3, 3, 3 ],
                structure = 'Epsilon(1,2,3,4)')

VVVV2 = Lorentz(name = 'VVVV2',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,1)*P(4,1)*Metric(1,2)')

VVVV3 = Lorentz(name = 'VVVV3',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,2)*P(4,1)*Metric(1,2)')

VVVV4 = Lorentz(name = 'VVVV4',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,1)*P(4,2)*Metric(1,2)')

VVVV5 = Lorentz(name = 'VVVV5',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,2)*P(4,2)*Metric(1,2)')

VVVV6 = Lorentz(name = 'VVVV6',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,1)*P(4,3)*Metric(1,2)')

VVVV7 = Lorentz(name = 'VVVV7',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,2)*P(4,3)*Metric(1,2)')

VVVV8 = Lorentz(name = 'VVVV8',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,4)*P(4,1)*Metric(1,2)')

VVVV9 = Lorentz(name = 'VVVV9',
                spins = [ 3, 3, 3, 3 ],
                structure = 'P(3,4)*P(4,2)*Metric(1,2)')

VVVV10 = Lorentz(name = 'VVVV10',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,3)*Metric(1,2)')

VVVV11 = Lorentz(name = 'VVVV11',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,1)*P(4,1)*Metric(1,3)')

VVVV12 = Lorentz(name = 'VVVV12',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,1)*P(4,2)*Metric(1,3)')

VVVV13 = Lorentz(name = 'VVVV13',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,3)*P(4,1)*Metric(1,3)')

VVVV14 = Lorentz(name = 'VVVV14',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,3)*P(4,2)*Metric(1,3)')

VVVV15 = Lorentz(name = 'VVVV15',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,1)*P(4,3)*Metric(1,3)')

VVVV16 = Lorentz(name = 'VVVV16',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,3)*P(4,3)*Metric(1,3)')

VVVV17 = Lorentz(name = 'VVVV17',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,4)*P(4,1)*Metric(1,3)')

VVVV18 = Lorentz(name = 'VVVV18',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,4)*P(4,2)*Metric(1,3)')

VVVV19 = Lorentz(name = 'VVVV19',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,4)*P(4,3)*Metric(1,3)')

VVVV20 = Lorentz(name = 'VVVV20',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,1)*P(3,1)*Metric(1,4)')

VVVV21 = Lorentz(name = 'VVVV21',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,1)*P(3,2)*Metric(1,4)')

VVVV22 = Lorentz(name = 'VVVV22',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,3)*P(3,1)*Metric(1,4)')

VVVV23 = Lorentz(name = 'VVVV23',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,3)*P(3,2)*Metric(1,4)')

VVVV24 = Lorentz(name = 'VVVV24',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,4)*P(3,1)*Metric(1,4)')

VVVV25 = Lorentz(name = 'VVVV25',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,4)*P(3,2)*Metric(1,4)')

VVVV26 = Lorentz(name = 'VVVV26',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,1)*P(3,4)*Metric(1,4)')

VVVV27 = Lorentz(name = 'VVVV27',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,3)*P(3,4)*Metric(1,4)')

VVVV28 = Lorentz(name = 'VVVV28',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(2,4)*P(3,4)*Metric(1,4)')

VVVV29 = Lorentz(name = 'VVVV29',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(4,1)*Metric(2,3)')

VVVV30 = Lorentz(name = 'VVVV30',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(4,2)*Metric(2,3)')

VVVV31 = Lorentz(name = 'VVVV31',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(4,1)*Metric(2,3)')

VVVV32 = Lorentz(name = 'VVVV32',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(4,2)*Metric(2,3)')

VVVV33 = Lorentz(name = 'VVVV33',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(4,3)*Metric(2,3)')

VVVV34 = Lorentz(name = 'VVVV34',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(4,3)*Metric(2,3)')

VVVV35 = Lorentz(name = 'VVVV35',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(4,1)*Metric(2,3)')

VVVV36 = Lorentz(name = 'VVVV36',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(4,2)*Metric(2,3)')

VVVV37 = Lorentz(name = 'VVVV37',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(4,3)*Metric(2,3)')

VVVV38 = Lorentz(name = 'VVVV38',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3)')

VVVV39 = Lorentz(name = 'VVVV39',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)**2*Metric(1,4)*Metric(2,3)')

VVVV40 = Lorentz(name = 'VVVV40',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3)')

VVVV41 = Lorentz(name = 'VVVV41',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)**2*Metric(1,4)*Metric(2,3)')

VVVV42 = Lorentz(name = 'VVVV42',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3)')

VVVV43 = Lorentz(name = 'VVVV43',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)*P(-1,3)*Metric(1,4)*Metric(2,3)')

VVVV44 = Lorentz(name = 'VVVV44',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,3)**2*Metric(1,4)*Metric(2,3)')

VVVV45 = Lorentz(name = 'VVVV45',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,4)*Metric(1,4)*Metric(2,3)')

VVVV46 = Lorentz(name = 'VVVV46',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3)')

VVVV47 = Lorentz(name = 'VVVV47',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3)')

VVVV48 = Lorentz(name = 'VVVV48',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,4)**2*Metric(1,4)*Metric(2,3)')

VVVV49 = Lorentz(name = 'VVVV49',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(3,1)*Metric(2,4)')

VVVV50 = Lorentz(name = 'VVVV50',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(3,2)*Metric(2,4)')

VVVV51 = Lorentz(name = 'VVVV51',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(3,1)*Metric(2,4)')

VVVV52 = Lorentz(name = 'VVVV52',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(3,2)*Metric(2,4)')

VVVV53 = Lorentz(name = 'VVVV53',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(3,1)*Metric(2,4)')

VVVV54 = Lorentz(name = 'VVVV54',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(3,2)*Metric(2,4)')

VVVV55 = Lorentz(name = 'VVVV55',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(3,4)*Metric(2,4)')

VVVV56 = Lorentz(name = 'VVVV56',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(3,4)*Metric(2,4)')

VVVV57 = Lorentz(name = 'VVVV57',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(3,4)*Metric(2,4)')

VVVV58 = Lorentz(name = 'VVVV58',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,3)*Metric(2,4)')

VVVV59 = Lorentz(name = 'VVVV59',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)**2*Metric(1,3)*Metric(2,4)')

VVVV60 = Lorentz(name = 'VVVV60',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4)')

VVVV61 = Lorentz(name = 'VVVV61',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)**2*Metric(1,3)*Metric(2,4)')

VVVV62 = Lorentz(name = 'VVVV62',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,3)*Metric(1,3)*Metric(2,4)')

VVVV63 = Lorentz(name = 'VVVV63',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4)')

VVVV64 = Lorentz(name = 'VVVV64',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,3)**2*Metric(1,3)*Metric(2,4)')

VVVV65 = Lorentz(name = 'VVVV65',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4)')

VVVV66 = Lorentz(name = 'VVVV66',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)*P(-1,4)*Metric(1,3)*Metric(2,4)')

VVVV67 = Lorentz(name = 'VVVV67',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4)')

VVVV68 = Lorentz(name = 'VVVV68',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,4)**2*Metric(1,3)*Metric(2,4)')

VVVV69 = Lorentz(name = 'VVVV69',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,3)*Metric(2,4)')

VVVV70 = Lorentz(name = 'VVVV70',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(2,1)*Metric(3,4)')

VVVV71 = Lorentz(name = 'VVVV71',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(2,1)*Metric(3,4)')

VVVV72 = Lorentz(name = 'VVVV72',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(2,3)*Metric(3,4)')

VVVV73 = Lorentz(name = 'VVVV73',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(2,3)*Metric(3,4)')

VVVV74 = Lorentz(name = 'VVVV74',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(2,1)*Metric(3,4)')

VVVV75 = Lorentz(name = 'VVVV75',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(2,3)*Metric(3,4)')

VVVV76 = Lorentz(name = 'VVVV76',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,2)*P(2,4)*Metric(3,4)')

VVVV77 = Lorentz(name = 'VVVV77',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,3)*P(2,4)*Metric(3,4)')

VVVV78 = Lorentz(name = 'VVVV78',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(1,4)*P(2,4)*Metric(3,4)')

VVVV79 = Lorentz(name = 'VVVV79',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,2)*Metric(3,4)')

VVVV80 = Lorentz(name = 'VVVV80',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)**2*Metric(1,2)*Metric(3,4)')

VVVV81 = Lorentz(name = 'VVVV81',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,2)*Metric(1,2)*Metric(3,4)')

VVVV82 = Lorentz(name = 'VVVV82',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)**2*Metric(1,2)*Metric(3,4)')

VVVV83 = Lorentz(name = 'VVVV83',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4)')

VVVV84 = Lorentz(name = 'VVVV84',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4)')

VVVV85 = Lorentz(name = 'VVVV85',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,3)**2*Metric(1,2)*Metric(3,4)')

VVVV86 = Lorentz(name = 'VVVV86',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV87 = Lorentz(name = 'VVVV87',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV88 = Lorentz(name = 'VVVV88',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,3)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV89 = Lorentz(name = 'VVVV89',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(-1,4)**2*Metric(1,2)*Metric(3,4)')

VVVV90 = Lorentz(name = 'VVVV90',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,2)*P(4,1)*Metric(1,2) - P(3,1)*P(4,2)*Metric(1,2) + P(2,1)*P(4,2)*Metric(1,3) + P(2,4)*P(4,3)*Metric(1,3) - P(2,1)*P(3,2)*Metric(1,4) - P(2,3)*P(3,4)*Metric(1,4) - P(1,2)*P(4,1)*Metric(2,3) - P(1,4)*P(4,3)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,2)*P(3,1)*Metric(2,4) + P(1,3)*P(3,4)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) - P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,4)*P(2,3)*Metric(3,4) - P(1,3)*P(2,4)*Metric(3,4)')

VVVV91 = Lorentz(name = 'VVVV91',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) - 2*Metric(1,2)*Metric(3,4)')

VVVV92 = Lorentz(name = 'VVVV92',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - Metric(1,2)*Metric(3,4)')

VVVV93 = Lorentz(name = 'VVVV93',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,3)*Metric(2,4) - Metric(1,2)*Metric(3,4)')

VVVV94 = Lorentz(name = 'VVVV94',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. - (Metric(1,2)*Metric(3,4))/2.')

VVVV95 = Lorentz(name = 'VVVV95',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) + Metric(1,2)*Metric(3,4)')

VVVV96 = Lorentz(name = 'VVVV96',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,1)*Metric(1,2) + P(3,2)*P(4,3)*Metric(1,2) - P(2,4)*P(4,1)*Metric(1,3) - P(2,3)*P(4,2)*Metric(1,3) + P(2,4)*P(3,1)*Metric(1,4) - P(2,1)*P(3,4)*Metric(1,4) + P(1,3)*P(4,2)*Metric(2,3) - P(1,2)*P(4,3)*Metric(2,3) - P(1,4)*P(3,1)*Metric(2,4) - P(1,3)*P(3,2)*Metric(2,4) + P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4) + P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(1,4)*P(2,1)*Metric(3,4) + P(1,2)*P(2,3)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV97 = Lorentz(name = 'VVVV97',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,4)*P(4,2)*Metric(1,2) + P(3,1)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) - P(2,1)*P(4,3)*Metric(1,3) - P(2,3)*P(3,1)*Metric(1,4) - P(2,4)*P(3,2)*Metric(1,4) - P(1,3)*P(4,1)*Metric(2,3) - P(1,4)*P(4,2)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3) + P(1,4)*P(3,2)*Metric(2,4) - P(1,2)*P(3,4)*Metric(2,4) + P(1,3)*P(2,1)*Metric(3,4) + P(1,2)*P(2,4)*Metric(3,4) - P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV98 = Lorentz(name = 'VVVV98',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,3)*P(4,1)*Metric(1,2) + 2*P(3,4)*P(4,1)*Metric(1,2) - P(3,3)*P(4,2)*Metric(1,2) - 2*P(3,4)*P(4,2)*Metric(1,2) - 2*P(3,1)*P(4,3)*Metric(1,2) + 2*P(3,2)*P(4,3)*Metric(1,2) - P(3,1)*P(4,4)*Metric(1,2) + P(3,2)*P(4,4)*Metric(1,2) + P(2,2)*P(4,2)*Metric(1,3) + 4*P(2,1)*P(4,3)*Metric(1,3) + 2*P(2,2)*P(4,3)*Metric(1,3) + 2*P(2,1)*P(4,4)*Metric(1,3) + P(2,2)*P(4,4)*Metric(1,3) + P(2,4)*P(4,4)*Metric(1,3) - P(2,2)*P(3,2)*Metric(1,4) - 2*P(2,1)*P(3,3)*Metric(1,4) - P(2,2)*P(3,3)*Metric(1,4) - P(2,3)*P(3,3)*Metric(1,4) - 4*P(2,1)*P(3,4)*Metric(1,4) - 2*P(2,2)*P(3,4)*Metric(1,4) - P(1,1)*P(4,1)*Metric(2,3) - 2*P(1,1)*P(4,3)*Metric(2,3) - 4*P(1,2)*P(4,3)*Metric(2,3) - P(1,1)*P(4,4)*Metric(2,3) - 2*P(1,2)*P(4,4)*Metric(2,3) - P(1,4)*P(4,4)*Metric(2,3) + P(-1,1)**2*Metric(1,4)*Metric(2,3) + P(-1,2)**2*Metric(1,4)*Metric(2,3) + P(-1,3)**2*Metric(1,4)*Metric(2,3) + P(-1,4)**2*Metric(1,4)*Metric(2,3) + P(1,1)*P(3,1)*Metric(2,4) + P(1,1)*P(3,3)*Metric(2,4) + 2*P(1,2)*P(3,3)*Metric(2,4) + P(1,3)*P(3,3)*Metric(2,4) + 2*P(1,1)*P(3,4)*Metric(2,4) + 4*P(1,2)*P(3,4)*Metric(2,4) - P(-1,1)**2*Metric(1,3)*Metric(2,4) - P(-1,2)**2*Metric(1,3)*Metric(2,4) - P(-1,3)**2*Metric(1,3)*Metric(2,4) - P(-1,4)**2*Metric(1,3)*Metric(2,4) - 2*P(1,3)*P(2,1)*Metric(3,4) + 2*P(1,4)*P(2,1)*Metric(3,4) - P(1,3)*P(2,2)*Metric(3,4) + P(1,4)*P(2,2)*Metric(3,4) + P(1,1)*P(2,3)*Metric(3,4) + 2*P(1,2)*P(2,3)*Metric(3,4) - P(1,1)*P(2,4)*Metric(3,4) - 2*P(1,2)*P(2,4)*Metric(3,4) + P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4) - P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4) + P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4)')

VVVV99 = Lorentz(name = 'VVVV99',
                 spins = [ 3, 3, 3, 3 ],
                 structure = 'P(3,1)*P(4,1)*Metric(1,2) + (110*P(3,2)*P(4,1)*Metric(1,2))/3. - (134*P(3,4)*P(4,1)*Metric(1,2))/3. + (110*P(3,1)*P(4,2)*Metric(1,2))/3. + P(3,2)*P(4,2)*Metric(1,2) - (134*P(3,4)*P(4,2)*Metric(1,2))/3. - (134*P(3,1)*P(4,3)*Metric(1,2))/3. - (134*P(3,2)*P(4,3)*Metric(1,2))/3. + (380*P(3,4)*P(4,3)*Metric(1,2))/3. + P(2,1)*P(4,1)*Metric(1,3) + (110*P(2,3)*P(4,1)*Metric(1,3))/3. - (134*P(2,4)*P(4,1)*Metric(1,3))/3. - (134*P(2,1)*P(4,2)*Metric(1,3))/3. - (134*P(2,3)*P(4,2)*Metric(1,3))/3. + (380*P(2,4)*P(4,2)*Metric(1,3))/3. + (110*P(2,1)*P(4,3)*Metric(1,3))/3. + P(2,3)*P(4,3)*Metric(1,3) - (134*P(2,4)*P(4,3)*Metric(1,3))/3. + P(2,1)*P(3,1)*Metric(1,4) - (134*P(2,3)*P(3,1)*Metric(1,4))/3. + (110*P(2,4)*P(3,1)*Metric(1,4))/3. - (134*P(2,1)*P(3,2)*Metric(1,4))/3. + (380*P(2,3)*P(3,2)*Metric(1,4))/3. - (134*P(2,4)*P(3,2)*Metric(1,4))/3. + (110*P(2,1)*P(3,4)*Metric(1,4))/3. - (134*P(2,3)*P(3,4)*Metric(1,4))/3. + P(2,4)*P(3,4)*Metric(1,4) - (134*P(1,2)*P(4,1)*Metric(2,3))/3. - (134*P(1,3)*P(4,1)*Metric(2,3))/3. + (380*P(1,4)*P(4,1)*Metric(2,3))/3. + P(1,2)*P(4,2)*Metric(2,3) + (110*P(1,3)*P(4,2)*Metric(2,3))/3. - (134*P(1,4)*P(4,2)*Metric(2,3))/3. + (110*P(1,2)*P(4,3)*Metric(2,3))/3. + P(1,3)*P(4,3)*Metric(2,3) - (134*P(1,4)*P(4,3)*Metric(2,3))/3. + P(-1,1)**2*Metric(1,4)*Metric(2,3) + (51*P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3))/2. - (69*P(-1,2)**2*Metric(1,4)*Metric(2,3))/2. + (51*P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3))/2. - (592*P(-1,2)*P(-1,3)*Metric(1,4)*Metric(2,3))/3. - (69*P(-1,3)**2*Metric(1,4)*Metric(2,3))/2. - (187*P(-1,1)*P(-1,4)*Metric(1,4)*Metric(2,3))/2. + (175*P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3))/3. + (175*P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3))/3. + (203*P(-1,4)**2*Metric(1,4)*Metric(2,3))/6. - (134*P(1,2)*P(3,1)*Metric(2,4))/3. + (380*P(1,3)*P(3,1)*Metric(2,4))/3. - (134*P(1,4)*P(3,1)*Metric(2,4))/3. + P(1,2)*P(3,2)*Metric(2,4) - (134*P(1,3)*P(3,2)*Metric(2,4))/3. + (110*P(1,4)*P(3,2)*Metric(2,4))/3. + (110*P(1,2)*P(3,4)*Metric(2,4))/3. - (134*P(1,3)*P(3,4)*Metric(2,4))/3. + P(1,4)*P(3,4)*Metric(2,4) + P(-1,1)**2*Metric(1,3)*Metric(2,4) + (51*P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4))/2. - (69*P(-1,2)**2*Metric(1,3)*Metric(2,4))/2. - (187*P(-1,1)*P(-1,3)*Metric(1,3)*Metric(2,4))/2. + (175*P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4))/3. + (203*P(-1,3)**2*Metric(1,3)*Metric(2,4))/6. + (51*P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4))/2. - (592*P(-1,2)*P(-1,4)*Metric(1,3)*Metric(2,4))/3. + (175*P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4))/3. - (69*P(-1,4)**2*Metric(1,3)*Metric(2,4))/2. + (380*P(1,2)*P(2,1)*Metric(3,4))/3. - (134*P(1,3)*P(2,1)*Metric(3,4))/3. - (134*P(1,4)*P(2,1)*Metric(3,4))/3. - (134*P(1,2)*P(2,3)*Metric(3,4))/3. + P(1,3)*P(2,3)*Metric(3,4) + (110*P(1,4)*P(2,3)*Metric(3,4))/3. - (134*P(1,2)*P(2,4)*Metric(3,4))/3. + (110*P(1,3)*P(2,4)*Metric(3,4))/3. + P(1,4)*P(2,4)*Metric(3,4) + P(-1,1)**2*Metric(1,2)*Metric(3,4) - (187*P(-1,1)*P(-1,2)*Metric(1,2)*Metric(3,4))/2. + (203*P(-1,2)**2*Metric(1,2)*Metric(3,4))/6. + (51*P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4))/2. + (175*P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4))/3. - (69*P(-1,3)**2*Metric(1,2)*Metric(3,4))/2. + (51*P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4))/2. + (175*P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4))/3. - (592*P(-1,3)*P(-1,4)*Metric(1,2)*Metric(3,4))/3. - (69*P(-1,4)**2*Metric(1,2)*Metric(3,4))/2.')

VVVV100 = Lorentz(name = 'VVVV100',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'P(3,1)*P(4,1)*Metric(1,2) - (13*P(3,2)*P(4,1)*Metric(1,2))/10. + (P(3,4)*P(4,1)*Metric(1,2))/10. - (13*P(3,1)*P(4,2)*Metric(1,2))/10. + P(3,2)*P(4,2)*Metric(1,2) + (P(3,4)*P(4,2)*Metric(1,2))/10. + (P(3,1)*P(4,3)*Metric(1,2))/10. + (P(3,2)*P(4,3)*Metric(1,2))/10. + (13*P(3,4)*P(4,3)*Metric(1,2))/5. + P(2,1)*P(4,1)*Metric(1,3) - (13*P(2,3)*P(4,1)*Metric(1,3))/10. + (P(2,4)*P(4,1)*Metric(1,3))/10. + (P(2,1)*P(4,2)*Metric(1,3))/10. + (P(2,3)*P(4,2)*Metric(1,3))/10. + (13*P(2,4)*P(4,2)*Metric(1,3))/5. - (13*P(2,1)*P(4,3)*Metric(1,3))/10. + P(2,3)*P(4,3)*Metric(1,3) + (P(2,4)*P(4,3)*Metric(1,3))/10. + P(2,1)*P(3,1)*Metric(1,4) + (P(2,3)*P(3,1)*Metric(1,4))/10. - (13*P(2,4)*P(3,1)*Metric(1,4))/10. + (P(2,1)*P(3,2)*Metric(1,4))/10. + (13*P(2,3)*P(3,2)*Metric(1,4))/5. + (P(2,4)*P(3,2)*Metric(1,4))/10. - (13*P(2,1)*P(3,4)*Metric(1,4))/10. + (P(2,3)*P(3,4)*Metric(1,4))/10. + P(2,4)*P(3,4)*Metric(1,4) + (P(1,2)*P(4,1)*Metric(2,3))/10. + (P(1,3)*P(4,1)*Metric(2,3))/10. + (13*P(1,4)*P(4,1)*Metric(2,3))/5. + P(1,2)*P(4,2)*Metric(2,3) - (13*P(1,3)*P(4,2)*Metric(2,3))/10. + (P(1,4)*P(4,2)*Metric(2,3))/10. - (13*P(1,2)*P(4,3)*Metric(2,3))/10. + P(1,3)*P(4,3)*Metric(2,3) + (P(1,4)*P(4,3)*Metric(2,3))/10. + (4*P(-1,1)**2*Metric(1,4)*Metric(2,3))/5. + (P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3))/4. - (731*P(-1,2)**2*Metric(1,4)*Metric(2,3))/100. + (P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3))/4. - (897*P(-1,2)*P(-1,3)*Metric(1,4)*Metric(2,3))/50. - (731*P(-1,3)**2*Metric(1,4)*Metric(2,3))/100. - (93*P(-1,1)*P(-1,4)*Metric(1,4)*Metric(2,3))/10. - (733*P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3))/100. - (733*P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3))/100. - (339*P(-1,4)**2*Metric(1,4)*Metric(2,3))/50. + (P(1,2)*P(3,1)*Metric(2,4))/10. + (13*P(1,3)*P(3,1)*Metric(2,4))/5. + (P(1,4)*P(3,1)*Metric(2,4))/10. + P(1,2)*P(3,2)*Metric(2,4) + (P(1,3)*P(3,2)*Metric(2,4))/10. - (13*P(1,4)*P(3,2)*Metric(2,4))/10. - (13*P(1,2)*P(3,4)*Metric(2,4))/10. + (P(1,3)*P(3,4)*Metric(2,4))/10. + P(1,4)*P(3,4)*Metric(2,4) + (4*P(-1,1)**2*Metric(1,3)*Metric(2,4))/5. + (P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4))/4. - (731*P(-1,2)**2*Metric(1,3)*Metric(2,4))/100. - (93*P(-1,1)*P(-1,3)*Metric(1,3)*Metric(2,4))/10. - (733*P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4))/100. - (339*P(-1,3)**2*Metric(1,3)*Metric(2,4))/50. + (P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. - (897*P(-1,2)*P(-1,4)*Metric(1,3)*Metric(2,4))/50. - (733*P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4))/100. - (731*P(-1,4)**2*Metric(1,3)*Metric(2,4))/100. + (13*P(1,2)*P(2,1)*Metric(3,4))/5. + (P(1,3)*P(2,1)*Metric(3,4))/10. + (P(1,4)*P(2,1)*Metric(3,4))/10. + (P(1,2)*P(2,3)*Metric(3,4))/10. + P(1,3)*P(2,3)*Metric(3,4) - (13*P(1,4)*P(2,3)*Metric(3,4))/10. + (P(1,2)*P(2,4)*Metric(3,4))/10. - (13*P(1,3)*P(2,4)*Metric(3,4))/10. + P(1,4)*P(2,4)*Metric(3,4) + (4*P(-1,1)**2*Metric(1,2)*Metric(3,4))/5. - (93*P(-1,1)*P(-1,2)*Metric(1,2)*Metric(3,4))/10. - (339*P(-1,2)**2*Metric(1,2)*Metric(3,4))/50. + (P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4))/4. - (733*P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4))/100. - (731*P(-1,3)**2*Metric(1,2)*Metric(3,4))/100. + (P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4))/4. - (733*P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4))/100. - (897*P(-1,3)*P(-1,4)*Metric(1,2)*Metric(3,4))/50. - (731*P(-1,4)**2*Metric(1,2)*Metric(3,4))/100.')

VVVV101 = Lorentz(name = 'VVVV101',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'P(3,2)*P(4,1)*Metric(1,2) + (P(3,3)*P(4,1)*Metric(1,2))/2. + (P(3,3)*P(4,3)*Metric(1,2))/4. + (P(3,2)*P(4,4)*Metric(1,2))/2. + (P(3,3)*P(4,4)*Metric(1,2))/4. + (P(3,4)*P(4,4)*Metric(1,2))/4. - (P(2,2)*P(4,1)*Metric(1,3))/2. - P(2,3)*P(4,1)*Metric(1,3) - (P(2,2)*P(4,2)*Metric(1,3))/4. - (P(2,2)*P(4,4)*Metric(1,3))/4. - (P(2,3)*P(4,4)*Metric(1,3))/2. - (P(2,4)*P(4,4)*Metric(1,3))/4. + (P(2,2)*P(3,1)*Metric(1,4))/4. + (P(2,3)*P(3,1)*Metric(1,4))/2. - (P(2,1)*P(3,2)*Metric(1,4))/2. + (P(2,4)*P(3,2)*Metric(1,4))/2. - (P(2,1)*P(3,3)*Metric(1,4))/4. + (P(2,4)*P(3,3)*Metric(1,4))/4. - (P(2,2)*P(3,4)*Metric(1,4))/4. - (P(2,3)*P(3,4)*Metric(1,4))/2. - (P(1,2)*P(4,1)*Metric(2,3))/2. + (P(1,3)*P(4,1)*Metric(2,3))/2. + (P(1,1)*P(4,2)*Metric(2,3))/4. + (P(1,4)*P(4,2)*Metric(2,3))/2. - (P(1,1)*P(4,3)*Metric(2,3))/4. - (P(1,4)*P(4,3)*Metric(2,3))/2. - (P(1,2)*P(4,4)*Metric(2,3))/4. + (P(1,3)*P(4,4)*Metric(2,3))/4. + (P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3))/4. - (P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3))/4. - (P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3))/4. + (P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3))/4. - (P(1,1)*P(3,1)*Metric(2,4))/4. - (P(1,1)*P(3,2)*Metric(2,4))/2. - P(1,4)*P(3,2)*Metric(2,4) - (P(1,1)*P(3,3)*Metric(2,4))/4. - (P(1,3)*P(3,3)*Metric(2,4))/4. - (P(1,4)*P(3,3)*Metric(2,4))/2. + (P(-1,1)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,2)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,3)**2*Metric(1,3)*Metric(2,4))/4. + (P(-1,4)**2*Metric(1,3)*Metric(2,4))/4. + (P(1,1)*P(2,1)*Metric(3,4))/4. + (P(1,1)*P(2,2)*Metric(3,4))/4. + (P(1,2)*P(2,2)*Metric(3,4))/4. + (P(1,4)*P(2,2)*Metric(3,4))/2. + (P(1,1)*P(2,3)*Metric(3,4))/2. + P(1,4)*P(2,3)*Metric(3,4) - (P(-1,1)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,2)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,3)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,4)**2*Metric(1,2)*Metric(3,4))/4.')

VVVV102 = Lorentz(name = 'VVVV102',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'P(3,1)*P(4,2)*Metric(1,2) + (P(3,3)*P(4,2)*Metric(1,2))/2. + (P(3,3)*P(4,3)*Metric(1,2))/4. + (P(3,1)*P(4,4)*Metric(1,2))/2. + (P(3,3)*P(4,4)*Metric(1,2))/4. + (P(3,4)*P(4,4)*Metric(1,2))/4. + (P(2,2)*P(4,1)*Metric(1,3))/4. + (P(2,4)*P(4,1)*Metric(1,3))/2. - (P(2,1)*P(4,2)*Metric(1,3))/2. + (P(2,3)*P(4,2)*Metric(1,3))/2. - (P(2,2)*P(4,3)*Metric(1,3))/4. - (P(2,4)*P(4,3)*Metric(1,3))/2. - (P(2,1)*P(4,4)*Metric(1,3))/4. + (P(2,3)*P(4,4)*Metric(1,3))/4. - (P(2,2)*P(3,1)*Metric(1,4))/2. - P(2,4)*P(3,1)*Metric(1,4) - (P(2,2)*P(3,2)*Metric(1,4))/4. - (P(2,2)*P(3,3)*Metric(1,4))/4. - (P(2,3)*P(3,3)*Metric(1,4))/4. - (P(2,4)*P(3,3)*Metric(1,4))/2. - (P(1,1)*P(4,1)*Metric(2,3))/4. - (P(1,1)*P(4,2)*Metric(2,3))/2. - P(1,3)*P(4,2)*Metric(2,3) - (P(1,1)*P(4,4)*Metric(2,3))/4. - (P(1,3)*P(4,4)*Metric(2,3))/2. - (P(1,4)*P(4,4)*Metric(2,3))/4. + (P(-1,1)**2*Metric(1,4)*Metric(2,3))/4. + (P(-1,2)**2*Metric(1,4)*Metric(2,3))/4. + (P(-1,3)**2*Metric(1,4)*Metric(2,3))/4. + (P(-1,4)**2*Metric(1,4)*Metric(2,3))/4. - (P(1,2)*P(3,1)*Metric(2,4))/2. + (P(1,4)*P(3,1)*Metric(2,4))/2. + (P(1,1)*P(3,2)*Metric(2,4))/4. + (P(1,3)*P(3,2)*Metric(2,4))/2. - (P(1,2)*P(3,3)*Metric(2,4))/4. + (P(1,4)*P(3,3)*Metric(2,4))/4. - (P(1,1)*P(3,4)*Metric(2,4))/4. - (P(1,3)*P(3,4)*Metric(2,4))/2. + (P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4))/4. - (P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4))/4. - (P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. + (P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4))/4. + (P(1,1)*P(2,1)*Metric(3,4))/4. + (P(1,1)*P(2,2)*Metric(3,4))/4. + (P(1,2)*P(2,2)*Metric(3,4))/4. + (P(1,3)*P(2,2)*Metric(3,4))/2. + (P(1,1)*P(2,4)*Metric(3,4))/2. + P(1,3)*P(2,4)*Metric(3,4) - (P(-1,1)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,2)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,3)**2*Metric(1,2)*Metric(3,4))/4. - (P(-1,4)**2*Metric(1,2)*Metric(3,4))/4.')

VVVV103 = Lorentz(name = 'VVVV103',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3) + P(-1,2)**2*Metric(1,4)*Metric(2,3) + P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3) + 2*P(-1,2)*P(-1,3)*Metric(1,4)*Metric(2,3) + P(-1,3)**2*Metric(1,4)*Metric(2,3) - P(-1,1)*P(-1,4)*Metric(1,4)*Metric(2,3) - P(-1,4)**2*Metric(1,4)*Metric(2,3) + P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4) + P(-1,2)**2*Metric(1,3)*Metric(2,4) - P(-1,1)*P(-1,3)*Metric(1,3)*Metric(2,4) - P(-1,3)**2*Metric(1,3)*Metric(2,4) + P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4) + 2*P(-1,2)*P(-1,4)*Metric(1,3)*Metric(2,4) + P(-1,4)**2*Metric(1,3)*Metric(2,4) - P(-1,1)*P(-1,2)*Metric(1,2)*Metric(3,4) - P(-1,2)**2*Metric(1,2)*Metric(3,4) + P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4) + P(-1,3)**2*Metric(1,2)*Metric(3,4) + P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4) + 2*P(-1,3)*P(-1,4)*Metric(1,2)*Metric(3,4) + P(-1,4)**2*Metric(1,2)*Metric(3,4)')

VVVV104 = Lorentz(name = 'VVVV104',
                  spins = [ 3, 3, 3, 3 ],
                  structure = 'P(3,2)*P(4,1)*Metric(1,2) - 2*P(3,4)*P(4,1)*Metric(1,2) + P(3,1)*P(4,2)*Metric(1,2) - 2*P(3,4)*P(4,2)*Metric(1,2) - 2*P(3,1)*P(4,3)*Metric(1,2) - 2*P(3,2)*P(4,3)*Metric(1,2) - 10*P(3,4)*P(4,3)*Metric(1,2) + P(2,3)*P(4,1)*Metric(1,3) - 2*P(2,4)*P(4,1)*Metric(1,3) - 2*P(2,1)*P(4,2)*Metric(1,3) - 2*P(2,3)*P(4,2)*Metric(1,3) - 10*P(2,4)*P(4,2)*Metric(1,3) + P(2,1)*P(4,3)*Metric(1,3) - 2*P(2,4)*P(4,3)*Metric(1,3) - 2*P(2,3)*P(3,1)*Metric(1,4) + P(2,4)*P(3,1)*Metric(1,4) - 2*P(2,1)*P(3,2)*Metric(1,4) - 10*P(2,3)*P(3,2)*Metric(1,4) - 2*P(2,4)*P(3,2)*Metric(1,4) + P(2,1)*P(3,4)*Metric(1,4) - 2*P(2,3)*P(3,4)*Metric(1,4) - 2*P(1,2)*P(4,1)*Metric(2,3) - 2*P(1,3)*P(4,1)*Metric(2,3) - 10*P(1,4)*P(4,1)*Metric(2,3) + P(1,3)*P(4,2)*Metric(2,3) - 2*P(1,4)*P(4,2)*Metric(2,3) + P(1,2)*P(4,3)*Metric(2,3) - 2*P(1,4)*P(4,3)*Metric(2,3) + (9*P(-1,1)*P(-1,2)*Metric(1,4)*Metric(2,3))/2. + (3*P(-1,2)**2*Metric(1,4)*Metric(2,3))/2. + (9*P(-1,1)*P(-1,3)*Metric(1,4)*Metric(2,3))/2. + 13*P(-1,2)*P(-1,3)*Metric(1,4)*Metric(2,3) + (3*P(-1,3)**2*Metric(1,4)*Metric(2,3))/2. + 5*P(-1,1)*P(-1,4)*Metric(1,4)*Metric(2,3) - (P(-1,2)*P(-1,4)*Metric(1,4)*Metric(2,3))/2. - (P(-1,3)*P(-1,4)*Metric(1,4)*Metric(2,3))/2. - 5*P(-1,4)**2*Metric(1,4)*Metric(2,3) - 2*P(1,2)*P(3,1)*Metric(2,4) - 10*P(1,3)*P(3,1)*Metric(2,4) - 2*P(1,4)*P(3,1)*Metric(2,4) - 2*P(1,3)*P(3,2)*Metric(2,4) + P(1,4)*P(3,2)*Metric(2,4) + P(1,2)*P(3,4)*Metric(2,4) - 2*P(1,3)*P(3,4)*Metric(2,4) + (9*P(-1,1)*P(-1,2)*Metric(1,3)*Metric(2,4))/2. + (3*P(-1,2)**2*Metric(1,3)*Metric(2,4))/2. + 5*P(-1,1)*P(-1,3)*Metric(1,3)*Metric(2,4) - (P(-1,2)*P(-1,3)*Metric(1,3)*Metric(2,4))/2. - 5*P(-1,3)**2*Metric(1,3)*Metric(2,4) + (9*P(-1,1)*P(-1,4)*Metric(1,3)*Metric(2,4))/2. + 13*P(-1,2)*P(-1,4)*Metric(1,3)*Metric(2,4) - (P(-1,3)*P(-1,4)*Metric(1,3)*Metric(2,4))/2. + (3*P(-1,4)**2*Metric(1,3)*Metric(2,4))/2. - 10*P(1,2)*P(2,1)*Metric(3,4) - 2*P(1,3)*P(2,1)*Metric(3,4) - 2*P(1,4)*P(2,1)*Metric(3,4) - 2*P(1,2)*P(2,3)*Metric(3,4) + P(1,4)*P(2,3)*Metric(3,4) - 2*P(1,2)*P(2,4)*Metric(3,4) + P(1,3)*P(2,4)*Metric(3,4) + 5*P(-1,1)*P(-1,2)*Metric(1,2)*Metric(3,4) - 5*P(-1,2)**2*Metric(1,2)*Metric(3,4) + (9*P(-1,1)*P(-1,3)*Metric(1,2)*Metric(3,4))/2. - (P(-1,2)*P(-1,3)*Metric(1,2)*Metric(3,4))/2. + (3*P(-1,3)**2*Metric(1,2)*Metric(3,4))/2. + (9*P(-1,1)*P(-1,4)*Metric(1,2)*Metric(3,4))/2. - (P(-1,2)*P(-1,4)*Metric(1,2)*Metric(3,4))/2. + 13*P(-1,3)*P(-1,4)*Metric(1,2)*Metric(3,4) + (3*P(-1,4)**2*Metric(1,2)*Metric(3,4))/2.')

VVVVV1 = Lorentz(name = 'VVVVV1',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(4,3)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. - (P(3,3)*Metric(1,5)*Metric(2,4))/2. - P(3,4)*Metric(1,5)*Metric(2,4) - P(4,3)*Metric(1,3)*Metric(2,5) - (P(4,4)*Metric(1,3)*Metric(2,5))/2. + (P(3,3)*Metric(1,4)*Metric(2,5))/2. + P(3,4)*Metric(1,4)*Metric(2,5) - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(2,4)*Metric(1,5)*Metric(3,4))/2. + (P(1,3)*Metric(2,5)*Metric(3,4))/2. - (P(1,4)*Metric(2,5)*Metric(3,4))/2.')

VVVVV2 = Lorentz(name = 'VVVVV2',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(3,1)*Metric(1,5)*Metric(2,4) - P(4,1)*Metric(1,3)*Metric(2,5) + P(3,1)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) + P(4,1)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5)')

VVVVV3 = Lorentz(name = 'VVVVV3',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - P(4,2)*Metric(1,5)*Metric(2,3) + P(4,3)*Metric(1,5)*Metric(2,3) + 2*P(3,2)*Metric(1,5)*Metric(2,4) + P(3,3)*Metric(1,5)*Metric(2,4) - 2*P(3,2)*Metric(1,4)*Metric(2,5) - P(3,3)*Metric(1,4)*Metric(2,5) - P(2,2)*Metric(1,5)*Metric(3,4) - 2*P(2,3)*Metric(1,5)*Metric(3,4) + P(2,2)*Metric(1,4)*Metric(3,5) + 2*P(2,3)*Metric(1,4)*Metric(3,5)')

VVVVV4 = Lorentz(name = 'VVVVV4',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,2)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(3,2)*Metric(1,4)*Metric(2,5))/2. + (P(3,5)*Metric(1,4)*Metric(2,5))/2. - P(5,2)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,2)*Metric(1,5)*Metric(3,4))/2. + P(2,5)*Metric(1,5)*Metric(3,4) + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,5)*Metric(2,5)*Metric(3,4))/2. - (P(2,2)*Metric(1,4)*Metric(3,5))/2. - P(2,5)*Metric(1,4)*Metric(3,5)')

VVVVV5 = Lorentz(name = 'VVVVV5',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,2)*Metric(3,4) - P(5,2)*Metric(1,2)*Metric(3,4) - 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,2)*Metric(1,5)*Metric(3,4) + P(1,1)*Metric(2,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) + P(4,2)*Metric(1,2)*Metric(3,5) + 2*P(2,1)*Metric(1,4)*Metric(3,5) + P(2,2)*Metric(1,4)*Metric(3,5) - P(1,1)*Metric(2,4)*Metric(3,5) - 2*P(1,2)*Metric(2,4)*Metric(3,5)')

VVVVV6 = Lorentz(name = 'VVVVV6',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(4,2)*Metric(1,5)*Metric(2,3) + P(3,2)*Metric(1,5)*Metric(2,4) - P(3,2)*Metric(1,4)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(4,2)*Metric(1,2)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5)')

VVVVV7 = Lorentz(name = 'VVVVV7',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(4,3)*Metric(1,3)*Metric(2,5) + P(2,3)*Metric(1,5)*Metric(3,4) - P(1,3)*Metric(2,5)*Metric(3,4) - P(2,3)*Metric(1,4)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5)')

VVVVV8 = Lorentz(name = 'VVVVV8',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - 2*P(3,1)*Metric(1,5)*Metric(2,4) - P(3,3)*Metric(1,5)*Metric(2,4) - P(4,1)*Metric(1,3)*Metric(2,5) + P(4,3)*Metric(1,3)*Metric(2,5) + 2*P(3,1)*Metric(1,4)*Metric(2,5) + P(3,3)*Metric(1,4)*Metric(2,5) - P(1,1)*Metric(2,5)*Metric(3,4) - 2*P(1,3)*Metric(2,5)*Metric(3,4) + P(1,1)*Metric(2,4)*Metric(3,5) + 2*P(1,3)*Metric(2,4)*Metric(3,5)')

VVVVV9 = Lorentz(name = 'VVVVV9',
                 spins = [ 3, 3, 3, 3, 3 ],
                 structure = 'P(4,2)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(3,4)*Metric(1,5)*Metric(2,4))/2. - (P(2,2)*Metric(1,5)*Metric(3,4))/2. - P(2,4)*Metric(1,5)*Metric(3,4) - P(4,2)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(2,2)*Metric(1,4)*Metric(3,5))/2. + P(2,4)*Metric(1,4)*Metric(3,5) + (P(1,2)*Metric(2,4)*Metric(3,5))/2. - (P(1,4)*Metric(2,4)*Metric(3,5))/2.')

VVVVV10 = Lorentz(name = 'VVVVV10',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,1)*Metric(1,3)*Metric(2,5) + (P(4,4)*Metric(1,3)*Metric(2,5))/2. - (P(3,1)*Metric(1,4)*Metric(2,5))/2. + (P(3,4)*Metric(1,4)*Metric(2,5))/2. - (P(1,1)*Metric(2,5)*Metric(3,4))/2. - P(1,4)*Metric(2,5)*Metric(3,4) - P(4,1)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(2,1)*Metric(1,4)*Metric(3,5))/2. - (P(2,4)*Metric(1,4)*Metric(3,5))/2. + (P(1,1)*Metric(2,4)*Metric(3,5))/2. + P(1,4)*Metric(2,4)*Metric(3,5)')

VVVVV11 = Lorentz(name = 'VVVVV11',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(3,1)*Metric(1,5)*Metric(2,4))/2. + (P(3,5)*Metric(1,5)*Metric(2,4))/2. - P(5,1)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,5)*Metric(1,5)*Metric(3,4))/2. + (P(1,1)*Metric(2,5)*Metric(3,4))/2. + P(1,5)*Metric(2,5)*Metric(3,4) - (P(1,1)*Metric(2,4)*Metric(3,5))/2. - P(1,5)*Metric(2,4)*Metric(3,5)')

VVVVV12 = Lorentz(name = 'VVVVV12',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - P(5,3)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(3,3)*Metric(1,5)*Metric(2,4))/2. + P(3,5)*Metric(1,5)*Metric(2,4) - (P(3,3)*Metric(1,4)*Metric(2,5))/2. - P(3,5)*Metric(1,4)*Metric(2,5) - (P(2,3)*Metric(1,4)*Metric(3,5))/2. + (P(2,5)*Metric(1,4)*Metric(3,5))/2. + (P(1,3)*Metric(2,4)*Metric(3,5))/2. - (P(1,5)*Metric(2,4)*Metric(3,5))/2.')

VVVVV13 = Lorentz(name = 'VVVVV13',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,3)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. - (P(3,3)*Metric(1,5)*Metric(2,4))/2. - P(3,4)*Metric(1,5)*Metric(2,4) + (P(5,3)*Metric(1,2)*Metric(3,4))/2. - (P(5,4)*Metric(1,2)*Metric(3,4))/2. - (P(2,3)*Metric(1,5)*Metric(3,4))/2. + (P(2,4)*Metric(1,5)*Metric(3,4))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,4)*Metric(1,2)*Metric(4,5)')

VVVVV14 = Lorentz(name = 'VVVVV14',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,3)*Metric(1,3)*Metric(2,5) + (P(4,4)*Metric(1,3)*Metric(2,5))/2. - (P(3,3)*Metric(1,4)*Metric(2,5))/2. - P(3,4)*Metric(1,4)*Metric(2,5) + (P(5,3)*Metric(1,2)*Metric(3,4))/2. - (P(5,4)*Metric(1,2)*Metric(3,4))/2. - (P(1,3)*Metric(2,5)*Metric(3,4))/2. + (P(1,4)*Metric(2,5)*Metric(3,4))/2. - P(4,3)*Metric(1,2)*Metric(3,5) - (P(4,4)*Metric(1,2)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,4)*Metric(1,2)*Metric(4,5)')

VVVVV15 = Lorentz(name = 'VVVVV15',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(3,3)*Metric(1,4)*Metric(2,5))/2. - P(3,5)*Metric(1,4)*Metric(2,5) - P(5,3)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,3)*Metric(1,2)*Metric(3,5))/2. - (P(4,5)*Metric(1,2)*Metric(3,5))/2. - (P(2,3)*Metric(1,4)*Metric(3,5))/2. + (P(2,5)*Metric(1,4)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,5)*Metric(1,2)*Metric(4,5)')

VVVVV16 = Lorentz(name = 'VVVVV16',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(3,3)*Metric(1,5)*Metric(2,4))/2. - P(3,5)*Metric(1,5)*Metric(2,4) - P(5,3)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,3)*Metric(1,2)*Metric(3,5))/2. - (P(4,5)*Metric(1,2)*Metric(3,5))/2. - (P(1,3)*Metric(2,4)*Metric(3,5))/2. + (P(1,5)*Metric(2,4)*Metric(3,5))/2. + (P(3,3)*Metric(1,2)*Metric(4,5))/2. + P(3,5)*Metric(1,2)*Metric(4,5)')

VVVVV17 = Lorentz(name = 'VVVVV17',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) + P(4,1)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) - P(5,1)*Metric(1,2)*Metric(3,4) + P(2,1)*Metric(1,5)*Metric(3,4) + P(3,1)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5)')

VVVVV18 = Lorentz(name = 'VVVVV18',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(4,1)*Metric(1,5)*Metric(2,3) - P(5,1)*Metric(1,3)*Metric(2,4) + P(3,1)*Metric(1,5)*Metric(2,4) + P(4,1)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) - P(3,1)*Metric(1,2)*Metric(4,5) + P(2,1)*Metric(1,3)*Metric(4,5)')

VVVVV19 = Lorentz(name = 'VVVVV19',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,3)*Metric(2,4) - P(3,4)*Metric(1,4)*Metric(2,5) - P(5,4)*Metric(1,2)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) + P(2,4)*Metric(1,4)*Metric(3,5) - P(1,4)*Metric(2,4)*Metric(3,5) + P(3,4)*Metric(1,2)*Metric(4,5) - P(2,4)*Metric(1,3)*Metric(4,5)')

VVVVV20 = Lorentz(name = 'VVVVV20',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) + (P(4,4)*Metric(1,5)*Metric(2,3))/2. + (P(5,2)*Metric(1,3)*Metric(2,4))/2. - (P(5,4)*Metric(1,3)*Metric(2,4))/2. - (P(3,2)*Metric(1,5)*Metric(2,4))/2. + (P(3,4)*Metric(1,5)*Metric(2,4))/2. - P(4,2)*Metric(1,3)*Metric(2,5) - (P(4,4)*Metric(1,3)*Metric(2,5))/2. - (P(2,2)*Metric(1,5)*Metric(3,4))/2. - P(2,4)*Metric(1,5)*Metric(3,4) + (P(2,2)*Metric(1,3)*Metric(4,5))/2. + P(2,4)*Metric(1,3)*Metric(4,5)')

VVVVV21 = Lorentz(name = 'VVVVV21',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,3)*Metric(2,4) - P(5,4)*Metric(1,3)*Metric(2,4) - 2*P(4,2)*Metric(1,3)*Metric(2,5) - P(4,4)*Metric(1,3)*Metric(2,5) + 2*P(4,2)*Metric(1,2)*Metric(3,5) + P(4,4)*Metric(1,2)*Metric(3,5) - P(2,2)*Metric(1,4)*Metric(3,5) - 2*P(2,4)*Metric(1,4)*Metric(3,5) - P(1,2)*Metric(2,4)*Metric(3,5) + P(1,4)*Metric(2,4)*Metric(3,5) + P(2,2)*Metric(1,3)*Metric(4,5) + 2*P(2,4)*Metric(1,3)*Metric(4,5)')

VVVVV22 = Lorentz(name = 'VVVVV22',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(4,2)*Metric(1,3)*Metric(2,5))/2. + (P(4,5)*Metric(1,3)*Metric(2,5))/2. - P(5,2)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,2)*Metric(1,5)*Metric(3,4))/2. + P(2,5)*Metric(1,5)*Metric(3,4) + (P(1,2)*Metric(2,5)*Metric(3,4))/2. - (P(1,5)*Metric(2,5)*Metric(3,4))/2. - (P(2,2)*Metric(1,3)*Metric(4,5))/2. - P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV23 = Lorentz(name = 'VVVVV23',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,3)*Metric(2,4) + (P(5,5)*Metric(1,3)*Metric(2,4))/2. - (P(4,4)*Metric(1,3)*Metric(2,5))/2. - P(4,5)*Metric(1,3)*Metric(2,5) - P(5,4)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,4)*Metric(1,2)*Metric(3,5))/2. + P(4,5)*Metric(1,2)*Metric(3,5) + (P(3,4)*Metric(1,2)*Metric(4,5))/2. - (P(3,5)*Metric(1,2)*Metric(4,5))/2. - (P(2,4)*Metric(1,3)*Metric(4,5))/2. + (P(2,5)*Metric(1,3)*Metric(4,5))/2.')

VVVVV24 = Lorentz(name = 'VVVVV24',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(3,5)*Metric(1,5)*Metric(2,4) - P(4,5)*Metric(1,3)*Metric(2,5) - P(2,5)*Metric(1,5)*Metric(3,4) + P(1,5)*Metric(2,5)*Metric(3,4) + P(4,5)*Metric(1,2)*Metric(3,5) - P(1,5)*Metric(2,4)*Metric(3,5) - P(3,5)*Metric(1,2)*Metric(4,5) + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV25 = Lorentz(name = 'VVVVV25',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - P(5,2)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(4,2)*Metric(1,3)*Metric(2,5))/2. - (P(4,5)*Metric(1,3)*Metric(2,5))/2. - (P(3,2)*Metric(1,4)*Metric(2,5))/2. + (P(3,5)*Metric(1,4)*Metric(2,5))/2. - (P(2,2)*Metric(1,4)*Metric(3,5))/2. - P(2,5)*Metric(1,4)*Metric(3,5) + (P(2,2)*Metric(1,3)*Metric(4,5))/2. + P(2,5)*Metric(1,3)*Metric(4,5)')

VVVVV26 = Lorentz(name = 'VVVVV26',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,2)*Metric(3,4) - P(5,2)*Metric(1,2)*Metric(3,4) - 2*P(2,1)*Metric(1,5)*Metric(3,4) - P(2,2)*Metric(1,5)*Metric(3,4) + P(1,1)*Metric(2,5)*Metric(3,4) + 2*P(1,2)*Metric(2,5)*Metric(3,4) - P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) + 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,2)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV27 = Lorentz(name = 'VVVVV27',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,1)*Metric(1,2)*Metric(3,5) - P(4,2)*Metric(1,2)*Metric(3,5) - 2*P(2,1)*Metric(1,4)*Metric(3,5) - P(2,2)*Metric(1,4)*Metric(3,5) + P(1,1)*Metric(2,4)*Metric(3,5) + 2*P(1,2)*Metric(2,4)*Metric(3,5) - P(3,1)*Metric(1,2)*Metric(4,5) + P(3,2)*Metric(1,2)*Metric(4,5) + 2*P(2,1)*Metric(1,3)*Metric(4,5) + P(2,2)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV28 = Lorentz(name = 'VVVVV28',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) + P(5,2)*Metric(1,3)*Metric(2,4) - P(3,2)*Metric(1,5)*Metric(2,4) - P(4,2)*Metric(1,3)*Metric(2,5) - P(5,2)*Metric(1,2)*Metric(3,4) + P(1,2)*Metric(2,5)*Metric(3,4) + P(3,2)*Metric(1,2)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV29 = Lorentz(name = 'VVVVV29',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,2)*Metric(1,3)*Metric(2,4) + P(4,2)*Metric(1,3)*Metric(2,5) - P(3,2)*Metric(1,4)*Metric(2,5) - P(4,2)*Metric(1,2)*Metric(3,5) + P(1,2)*Metric(2,4)*Metric(3,5) + P(3,2)*Metric(1,2)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5)')

VVVVV30 = Lorentz(name = 'VVVVV30',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,1)*Metric(1,3)*Metric(2,5) - P(4,3)*Metric(1,3)*Metric(2,5) - 2*P(3,1)*Metric(1,4)*Metric(2,5) - P(3,3)*Metric(1,4)*Metric(2,5) + P(1,1)*Metric(2,5)*Metric(3,4) + 2*P(1,3)*Metric(2,5)*Metric(3,4) + 2*P(3,1)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV31 = Lorentz(name = 'VVVVV31',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,3)*Metric(2,4) - P(5,3)*Metric(1,3)*Metric(2,4) - 2*P(3,1)*Metric(1,5)*Metric(2,4) - P(3,3)*Metric(1,5)*Metric(2,4) + P(1,1)*Metric(2,4)*Metric(3,5) + 2*P(1,3)*Metric(2,4)*Metric(3,5) + 2*P(3,1)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,1)*Metric(1,3)*Metric(4,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,1)*Metric(2,3)*Metric(4,5) - 2*P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV32 = Lorentz(name = 'VVVVV32',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,3)*Metric(1,4)*Metric(2,3) - P(4,3)*Metric(1,3)*Metric(2,5) - P(5,3)*Metric(1,2)*Metric(3,4) + P(1,3)*Metric(2,5)*Metric(3,4) + P(4,3)*Metric(1,2)*Metric(3,5) - P(2,3)*Metric(1,4)*Metric(3,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV33 = Lorentz(name = 'VVVVV33',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,3)*Metric(1,5)*Metric(2,3) - P(5,3)*Metric(1,3)*Metric(2,4) + P(5,3)*Metric(1,2)*Metric(3,4) - P(2,3)*Metric(1,5)*Metric(3,4) - P(4,3)*Metric(1,2)*Metric(3,5) + P(1,3)*Metric(2,4)*Metric(3,5) + P(2,3)*Metric(1,3)*Metric(4,5) - P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV34 = Lorentz(name = 'VVVVV34',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,2)*Metric(1,5)*Metric(2,3) - P(4,3)*Metric(1,5)*Metric(2,3) - 2*P(3,2)*Metric(1,5)*Metric(2,4) - P(3,3)*Metric(1,5)*Metric(2,4) + P(2,2)*Metric(1,5)*Metric(3,4) + 2*P(2,3)*Metric(1,5)*Metric(3,4) + 2*P(3,2)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,2)*Metric(1,3)*Metric(4,5) - 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV35 = Lorentz(name = 'VVVVV35',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,2)*Metric(1,4)*Metric(2,3) - P(5,3)*Metric(1,4)*Metric(2,3) - 2*P(3,2)*Metric(1,4)*Metric(2,5) - P(3,3)*Metric(1,4)*Metric(2,5) + P(2,2)*Metric(1,4)*Metric(3,5) + 2*P(2,3)*Metric(1,4)*Metric(3,5) + 2*P(3,2)*Metric(1,2)*Metric(4,5) + P(3,3)*Metric(1,2)*Metric(4,5) - P(2,2)*Metric(1,3)*Metric(4,5) - 2*P(2,3)*Metric(1,3)*Metric(4,5) - P(1,2)*Metric(2,3)*Metric(4,5) + P(1,3)*Metric(2,3)*Metric(4,5)')

VVVVV36 = Lorentz(name = 'VVVVV36',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) - P(3,4)*Metric(1,5)*Metric(2,4) - P(5,4)*Metric(1,2)*Metric(3,4) + P(2,4)*Metric(1,5)*Metric(3,4) - P(2,4)*Metric(1,4)*Metric(3,5) + P(1,4)*Metric(2,4)*Metric(3,5) + P(3,4)*Metric(1,2)*Metric(4,5) - P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV37 = Lorentz(name = 'VVVVV37',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,3)*Metric(2,4) + P(3,4)*Metric(1,5)*Metric(2,4) - P(3,4)*Metric(1,4)*Metric(2,5) - P(2,4)*Metric(1,5)*Metric(3,4) + P(1,4)*Metric(2,5)*Metric(3,4) + P(2,4)*Metric(1,3)*Metric(4,5) - P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV38 = Lorentz(name = 'VVVVV38',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - 2*P(4,1)*Metric(1,5)*Metric(2,3) - P(4,4)*Metric(1,5)*Metric(2,3) + 2*P(4,1)*Metric(1,3)*Metric(2,5) + P(4,4)*Metric(1,3)*Metric(2,5) - P(3,1)*Metric(1,4)*Metric(2,5) + P(3,4)*Metric(1,4)*Metric(2,5) - P(1,1)*Metric(2,5)*Metric(3,4) - 2*P(1,4)*Metric(2,5)*Metric(3,4) + P(1,1)*Metric(2,3)*Metric(4,5) + 2*P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV39 = Lorentz(name = 'VVVVV39',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) - P(5,4)*Metric(1,4)*Metric(2,3) - 2*P(4,1)*Metric(1,5)*Metric(2,3) - P(4,4)*Metric(1,5)*Metric(2,3) + 2*P(4,1)*Metric(1,2)*Metric(3,5) + P(4,4)*Metric(1,2)*Metric(3,5) - P(2,1)*Metric(1,4)*Metric(3,5) + P(2,4)*Metric(1,4)*Metric(3,5) - P(1,1)*Metric(2,4)*Metric(3,5) - 2*P(1,4)*Metric(2,4)*Metric(3,5) + P(1,1)*Metric(2,3)*Metric(4,5) + 2*P(1,4)*Metric(2,3)*Metric(4,5)')

VVVVV40 = Lorentz(name = 'VVVVV40',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,5)*Metric(1,5)*Metric(2,3) - P(3,5)*Metric(1,4)*Metric(2,5) - P(2,5)*Metric(1,5)*Metric(3,4) + P(1,5)*Metric(2,5)*Metric(3,4) - P(4,5)*Metric(1,2)*Metric(3,5) + P(2,5)*Metric(1,4)*Metric(3,5) + P(3,5)*Metric(1,2)*Metric(4,5) - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV41 = Lorentz(name = 'VVVVV41',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(4,5)*Metric(1,5)*Metric(2,3) - P(3,5)*Metric(1,5)*Metric(2,4) - P(4,5)*Metric(1,3)*Metric(2,5) + P(3,5)*Metric(1,4)*Metric(2,5) - P(2,5)*Metric(1,4)*Metric(3,5) + P(1,5)*Metric(2,4)*Metric(3,5) + P(2,5)*Metric(1,3)*Metric(4,5) - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV42 = Lorentz(name = 'VVVVV42',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,1)*Metric(1,5)*Metric(2,3))/2. + (P(4,5)*Metric(1,5)*Metric(2,3))/2. - P(5,1)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(2,1)*Metric(1,5)*Metric(3,4))/2. - (P(2,5)*Metric(1,5)*Metric(3,4))/2. + (P(1,1)*Metric(2,5)*Metric(3,4))/2. + P(1,5)*Metric(2,5)*Metric(3,4) - (P(1,1)*Metric(2,3)*Metric(4,5))/2. - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV43 = Lorentz(name = 'VVVVV43',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,1)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,1)*Metric(1,5)*Metric(2,3))/2. + (P(4,5)*Metric(1,5)*Metric(2,3))/2. - P(5,1)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(3,1)*Metric(1,5)*Metric(2,4))/2. - (P(3,5)*Metric(1,5)*Metric(2,4))/2. + (P(1,1)*Metric(2,4)*Metric(3,5))/2. + P(1,5)*Metric(2,4)*Metric(3,5) - (P(1,1)*Metric(2,3)*Metric(4,5))/2. - P(1,5)*Metric(2,3)*Metric(4,5)')

VVVVV44 = Lorentz(name = 'VVVVV44',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,4)*Metric(1,5)*Metric(2,3))/2. - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,4)*Metric(1,2)*Metric(3,4) - (P(5,5)*Metric(1,2)*Metric(3,4))/2. + (P(4,4)*Metric(1,2)*Metric(3,5))/2. + P(4,5)*Metric(1,2)*Metric(3,5) + (P(3,4)*Metric(1,2)*Metric(4,5))/2. - (P(3,5)*Metric(1,2)*Metric(4,5))/2. - (P(1,4)*Metric(2,3)*Metric(4,5))/2. + (P(1,5)*Metric(2,3)*Metric(4,5))/2.')

VVVVV45 = Lorentz(name = 'VVVVV45',
                  spins = [ 3, 3, 3, 3, 3 ],
                  structure = 'P(5,4)*Metric(1,4)*Metric(2,3) + (P(5,5)*Metric(1,4)*Metric(2,3))/2. - (P(4,4)*Metric(1,5)*Metric(2,3))/2. - P(4,5)*Metric(1,5)*Metric(2,3) - P(5,4)*Metric(1,3)*Metric(2,4) - (P(5,5)*Metric(1,3)*Metric(2,4))/2. + (P(4,4)*Metric(1,3)*Metric(2,5))/2. + P(4,5)*Metric(1,3)*Metric(2,5) + (P(2,4)*Metric(1,3)*Metric(4,5))/2. - (P(2,5)*Metric(1,3)*Metric(4,5))/2. - (P(1,4)*Metric(2,3)*Metric(4,5))/2. + (P(1,5)*Metric(2,3)*Metric(4,5))/2.')

VVVVVV1 = Lorentz(name = 'VVVVVV1',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5)')

VVVVVV2 = Lorentz(name = 'VVVVVV2',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6)')

VVVVVV3 = Lorentz(name = 'VVVVVV3',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6)')

VVVVVV4 = Lorentz(name = 'VVVVVV4',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV5 = Lorentz(name = 'VVVVVV5',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV6 = Lorentz(name = 'VVVVVV6',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV7 = Lorentz(name = 'VVVVVV7',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV8 = Lorentz(name = 'VVVVVV8',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV9 = Lorentz(name = 'VVVVVV9',
                  spins = [ 3, 3, 3, 3, 3, 3 ],
                  structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6)')

VVVVVV10 = Lorentz(name = 'VVVVVV10',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5)')

VVVVVV11 = Lorentz(name = 'VVVVVV11',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5)')

VVVVVV12 = Lorentz(name = 'VVVVVV12',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV13 = Lorentz(name = 'VVVVVV13',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV14 = Lorentz(name = 'VVVVVV14',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV15 = Lorentz(name = 'VVVVVV15',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5)')

VVVVVV16 = Lorentz(name = 'VVVVVV16',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV17 = Lorentz(name = 'VVVVVV17',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV18 = Lorentz(name = 'VVVVVV18',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV19 = Lorentz(name = 'VVVVVV19',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6)')

VVVVVV20 = Lorentz(name = 'VVVVVV20',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV21 = Lorentz(name = 'VVVVVV21',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV22 = Lorentz(name = 'VVVVVV22',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV23 = Lorentz(name = 'VVVVVV23',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV24 = Lorentz(name = 'VVVVVV24',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV25 = Lorentz(name = 'VVVVVV25',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV26 = Lorentz(name = 'VVVVVV26',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV27 = Lorentz(name = 'VVVVVV27',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV28 = Lorentz(name = 'VVVVVV28',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV29 = Lorentz(name = 'VVVVVV29',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6)')

VVVVVV30 = Lorentz(name = 'VVVVVV30',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV31 = Lorentz(name = 'VVVVVV31',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV32 = Lorentz(name = 'VVVVVV32',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV33 = Lorentz(name = 'VVVVVV33',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV34 = Lorentz(name = 'VVVVVV34',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV35 = Lorentz(name = 'VVVVVV35',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV36 = Lorentz(name = 'VVVVVV36',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV37 = Lorentz(name = 'VVVVVV37',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV38 = Lorentz(name = 'VVVVVV38',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV39 = Lorentz(name = 'VVVVVV39',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV40 = Lorentz(name = 'VVVVVV40',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV41 = Lorentz(name = 'VVVVVV41',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV42 = Lorentz(name = 'VVVVVV42',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV43 = Lorentz(name = 'VVVVVV43',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV44 = Lorentz(name = 'VVVVVV44',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV45 = Lorentz(name = 'VVVVVV45',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6)')

VVVVVV46 = Lorentz(name = 'VVVVVV46',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV47 = Lorentz(name = 'VVVVVV47',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV48 = Lorentz(name = 'VVVVVV48',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV49 = Lorentz(name = 'VVVVVV49',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV50 = Lorentz(name = 'VVVVVV50',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV51 = Lorentz(name = 'VVVVVV51',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV52 = Lorentz(name = 'VVVVVV52',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV53 = Lorentz(name = 'VVVVVV53',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6)')

VVVVVV54 = Lorentz(name = 'VVVVVV54',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV55 = Lorentz(name = 'VVVVVV55',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV56 = Lorentz(name = 'VVVVVV56',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV57 = Lorentz(name = 'VVVVVV57',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV58 = Lorentz(name = 'VVVVVV58',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV59 = Lorentz(name = 'VVVVVV59',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV60 = Lorentz(name = 'VVVVVV60',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV61 = Lorentz(name = 'VVVVVV61',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV62 = Lorentz(name = 'VVVVVV62',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV63 = Lorentz(name = 'VVVVVV63',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV64 = Lorentz(name = 'VVVVVV64',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV65 = Lorentz(name = 'VVVVVV65',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV66 = Lorentz(name = 'VVVVVV66',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV67 = Lorentz(name = 'VVVVVV67',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV68 = Lorentz(name = 'VVVVVV68',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV69 = Lorentz(name = 'VVVVVV69',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV70 = Lorentz(name = 'VVVVVV70',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV71 = Lorentz(name = 'VVVVVV71',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV72 = Lorentz(name = 'VVVVVV72',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV73 = Lorentz(name = 'VVVVVV73',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,3)*Metric(2,4)*Metric(5,6)')

VVVVVV74 = Lorentz(name = 'VVVVVV74',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV75 = Lorentz(name = 'VVVVVV75',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV76 = Lorentz(name = 'VVVVVV76',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV77 = Lorentz(name = 'VVVVVV77',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV78 = Lorentz(name = 'VVVVVV78',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV79 = Lorentz(name = 'VVVVVV79',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV80 = Lorentz(name = 'VVVVVV80',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV81 = Lorentz(name = 'VVVVVV81',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV82 = Lorentz(name = 'VVVVVV82',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV83 = Lorentz(name = 'VVVVVV83',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV84 = Lorentz(name = 'VVVVVV84',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV85 = Lorentz(name = 'VVVVVV85',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) + Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV86 = Lorentz(name = 'VVVVVV86',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV87 = Lorentz(name = 'VVVVVV87',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV88 = Lorentz(name = 'VVVVVV88',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV89 = Lorentz(name = 'VVVVVV89',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,4)*Metric(2,3)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV90 = Lorentz(name = 'VVVVVV90',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV91 = Lorentz(name = 'VVVVVV91',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV92 = Lorentz(name = 'VVVVVV92',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV93 = Lorentz(name = 'VVVVVV93',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,6)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV94 = Lorentz(name = 'VVVVVV94',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV95 = Lorentz(name = 'VVVVVV95',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,5)*Metric(3,4) - Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,6)*Metric(2,3)*Metric(4,5) + Metric(1,2)*Metric(3,6)*Metric(4,5) + Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV96 = Lorentz(name = 'VVVVVV96',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,6)*Metric(2,4)*Metric(3,5) + Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,5)*Metric(2,3)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV97 = Lorentz(name = 'VVVVVV97',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,5)*Metric(2,6)*Metric(3,4) - Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,5)*Metric(2,4)*Metric(3,6) + Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,3)*Metric(2,5)*Metric(4,6) + Metric(1,2)*Metric(3,5)*Metric(4,6) + Metric(1,3)*Metric(2,4)*Metric(5,6) - Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV98 = Lorentz(name = 'VVVVVV98',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,4)*Metric(2,5)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV99 = Lorentz(name = 'VVVVVV99',
                   spins = [ 3, 3, 3, 3, 3, 3 ],
                   structure = 'Metric(1,6)*Metric(2,3)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV100 = Lorentz(name = 'VVVVVV100',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,4)*Metric(2,6)*Metric(3,5) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV101 = Lorentz(name = 'VVVVVV101',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,3)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,4)*Metric(2,3)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV102 = Lorentz(name = 'VVVVVV102',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,5)*Metric(2,4)*Metric(3,6) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV103 = Lorentz(name = 'VVVVVV103',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,3)*Metric(2,6)*Metric(4,5) - Metric(1,2)*Metric(3,6)*Metric(4,5) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV104 = Lorentz(name = 'VVVVVV104',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,6)*Metric(2,4)*Metric(3,5) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

VVVVVV105 = Lorentz(name = 'VVVVVV105',
                    spins = [ 3, 3, 3, 3, 3, 3 ],
                    structure = 'Metric(1,3)*Metric(2,5)*Metric(4,6) - Metric(1,2)*Metric(3,5)*Metric(4,6) - Metric(1,3)*Metric(2,4)*Metric(5,6) + Metric(1,2)*Metric(3,4)*Metric(5,6)')

