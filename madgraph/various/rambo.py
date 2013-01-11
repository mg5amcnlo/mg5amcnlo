from __future__ import division
import math
import random

class FortranList(list):
    
    def __init__(self, min, max=None):
        if max is None:
            self.min = 1
            self.max = min + 1
        else:
            self.min = min
            self.max = max + 1
        list.__init__(self,[0]*(self.max-self.min))

        
    def __getitem__(self, index):
        assert self.min <= index < self.max, 'outside range %s <= %s < %s' % (self.min, index, self.max)
        return list.__getitem__(self, index - self.min)
    
    def __setitem__(self, index, value):
        assert self.min <= index < self.max
        return list.__setitem__(self, index - self.min , value)

class DoubleFortranList(FortranList):

    def __init__(self, min, max=(None,None)):
        
        min1 = min[0]
        max1 = max[0] 
        FortranList.__init__(self, min1, max1)
        
        min2 = min[1]
        max2 = max[1]
        
        for i in range(len(self)):
            list.__setitem__(self,i,FortranList(min2,max2))
        
    def __getitem__(self, index):
        var1 = index[0]
        var2 = index[1]
        list1 = FortranList.__getitem__(self, var1)
        return list1.__getitem__(var2)
        
    def __setitem__(self, index, value):
        var1 = index[0]
        var2 = index[1]
        
        list1 = FortranList.__getitem__(self, var1)
        list1.__setitem__(var2, value)
                

class RAMBOError(Exception):
    """ A Error class for RAMBO routine """
    pass

def RAMBO(N,ET,XM):
    """***********************************************************************
    *                       RAMBO                                         *
    *    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)                  *
    *                                                                     *
    *    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR                *
    *    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING                 *
    *    -- ADJUSTED BY HANS KUIJF, WEIGHTS ARE LOGARITHMIC (20-08-90)    *
    *    THIS IS PY VERSION 1.0 -  WRITTEN BY O. MATTELAER                *
    *                                                                     *
    *    N  = NUMBER OF PARTICLES                                         *
    *    ET = TOTAL CENTRE-OF-MASS ENERGY                                 *
    *    XM = PARTICLE MASSES ( DIM=NEXTERNAL-nincoming )                 *
    *  RETURN                                                             *
    *    P  = PARTICLE MOMENTA ( DIM=(4,NEXTERNAL-nincoming) )            *
    *    WT = WEIGHT OF THE EVENT                                         *
    ***********************************************************************"""
    # RUN PARAMETER
    acc = 1e-14
    itmax = 6
    ibegin = 0
    iwarn = FortranList(5)
    Nincoming = 2
    
    
    # Object Initialization
    Z = FortranList(N)
    Q = DoubleFortranList((4,N))
    P = DoubleFortranList((4,N))
    R = FortranList(4)
    B = FortranList(3)
    XM2 = FortranList(N)
    P2 = FortranList(N)
    E = FortranList(N)
    V= FortranList(N)
    IWARN = [0,0]
# Check input object
    assert isinstance(XM, FortranList)
    assert XM.min == 1
    assert XM.max == N+1 

# INITIALIZATION STEP: FACTORIALS FOR THE PHASE SPACE WEIGHT                                                                               
    if not ibegin:
        ibegin = 1
        twopi = 8 * math.atan(1)
        po2log = math.log(twopi/4)
        Z[2] = po2log
        for k in range(3, N+1):
            Z[k] = Z[k-1] + po2log - 2.*math.log(k-2) - math.log(k-1)
          
# CHECK ON THE NUMBER OF PARTICLES
        assert 1 < N < 101

# CHECK WHETHER TOTAL ENERGY IS SUFFICIENT; COUNT NONZERO MASSES
    xmt =  0
    nm = 0
    for i in range(1,N+1):
        if XM[i] != 0:
            nm +=1
        xmt += abs(XM[i])
        
    if xmt > ET:
        raise RAMBOError, ' Not enough energy in this case'

#                                                                                                                                          
# THE PARAMETER VALUES ARE NOW ACCEPTED                                                                                                    
#                                                                                                                                          
# GENERATE N MASSLESS MOMENTA IN INFINITE PHASE SPACE                                                                                      
    for i in range(1,N+1):
        r1=random_nb(1)
        c = 2 * r1 -1
        s = math.sqrt(1 - c**2)
        f = twopi * random_nb(2)
        r1 = random_nb(3)
        r2 = random_nb(4)
        
        Q[(4,i)]=-math.log(r1*r2)
        Q[(3,i)]= Q[(4,i)]*c
        Q[(2,i)]=Q[(4,i)]*s*math.cos(f)
        Q[(1,i)]=Q[(4,i)]*s*math.sin(f)
        
# CALCULATE THE PARAMETERS OF THE CONFORMAL TRANSFORMATION                                                                                 
    for i in range(1, N+1):
        for k in range(1,5):
            R[k] = R[k] + Q[(k,i)]
    rmas = math.sqrt(R[4]**2-R[3]**2-R[2]**2-R[1]**2)
    for k in range(1,4):
        B[k] = - R[k]/rmas

    g = R[4] / rmas
    a = 1.0 / (1+g)
    x = ET / rmas

# TRANSFORM THE Q'S CONFORMALLY INTO THE P'S
    for i in range(1, N+1):
        bq = B[1]*Q[(1,i)]+B[2]*Q[(2,i)]+B[3]*Q[(3,i)]
        for k in range(1,4):
            P[k,i] = x*(Q[(k,i)]+B[k]*(Q[(4,i)]+a*bq)) 
        P[(4,i)] = x*(g*Q[(4,i)]+bq)
        
# CALCULATE WEIGHT AND POSSIBLE WARNINGS                                                                                                   
    wt = po2log
    if N != 2:
        wt = (2 * N-4) * math.log(ET) + Z[N]
    if wt < -180 and iwarn[1] < 5:
        print "RAMBO WARNS: WEIGHT = EXP(%f20.9) MAY UNDERFLOW" % wt
        iwarn[1] += 1
    if wt > 174 and iwarn[2] < 5:      
        print " RAMBO WARNS: WEIGHT = EXP(%f20.9) MAY  OVERFLOW" % wt
        iwarn[2] += 1

                                                                                                                                          
# RETURN FOR WEIGHTED MASSLESS MOMENTA                                                                                                     
    if nm == 0: 
        return P, wt


# MASSIVE PARTICLES: RESCALE THE MOMENTA BY A FACTOR X
    xmax = math.sqrt(1-(xmt/ET)**2)
    for i in range(1,N+1):
        XM2[i] = XM[i] **2
        P2[i] = P[(4,i)]**2
    n_iter  = 0
    x= xmax
    accu = ET * acc
  
    while 1:
        f0 = -ET
        g0 = 0
        x2 = x**2
        for i in range(1, N+1):
            E[i] = math.sqrt(XM2[i]+x2*P2[i])
            f0 += E[i]
            g0 += P2[i]/E[i] 
        if abs(f0) <= accu:
            break
        n_iter  += 1
        if n_iter  > itmax:
            print "RAMBO WARNS: %s ITERATIONS DID NOT GIVE THE DESIRED ACCURACY = %s" \
                    %(n_iter , f0)
            break
        x=x-f0/(x*g0)
    for i in range(1, N+1):
        V[i] = x * P[(4,i)]
        for k in range(1,4):
            P[(k,i)] = x * P[(k,i)]
        P[(4,i)] = E[i]
        
# CALCULATE THE MASS-EFFECT WEIGHT FACTOR                                                                                                  
    wt2 = 1.
    wt3 = 0.
    for i in range(1, N+1):
        wt2 *= V[i]/E[i]
        wt3 += V[i]**2/E[i]
    wtm = (2.*N-3.)*math.log(x)+math.log(wt2/wt3*ET)

# RETURN FOR  WEIGHTED MASSIVE MOMENTA                                                                                                     
    wt += wtm
    if(wt < -180 and iwarn[3] < 5):
        print " RAMBO WARNS: WEIGHT = EXP(%s) MAY UNDERFLOW" % wt
        iwarn[3] += 1
    if(wt > 174  and iwarn[4] > 5):
        print " RAMBO WARNS: WEIGHT = EXP(%s) MAY OVERFLOW" % wt
        iwarn[4] += 1

# RETURN
    return P, wt

def random_nb(value):
    """ random number """
    output = 0
    while output < 1e-16:
        output= random.uniform(0,1)
    return output
          


            








