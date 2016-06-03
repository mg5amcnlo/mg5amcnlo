from __future__ import division 
import math
from math import sqrt, pow
from itertools import product

class WaveFunction(list):
    """a objet for a WaveFunction"""
    
    spin_to_size={0:1,
                  1:3,
                  2:6,
                  3:6,
                  4:18,
                  5:18}
    
    def __init__(self, spin= None, size=None):
        """Init the list with zero value"""
        
        if spin:
            size = self.spin_to_size[spin]
        list.__init__(self, [0]*size)
        

def ixxxxx(p,fmass,nhel,nsf):
    """Defines an inflow fermion."""
    
    fi = WaveFunction(2)
    
    fi[0] = complex(-p[0]*nsf,-p[3]*nsf)
    fi[1] = complex(-p[1]*nsf,-p[2]*nsf) 
    nh = nhel*nsf 
    if (fmass != 0.):
        pp = min(p[0],sqrt(p[1]**2 + p[2]**2 + p[3]**2 ))
        if (pp == 0.): 
            sqm = [sqrt(abs(fmass))]
            sqm.append(sign(sqm[0],fmass)) 
            ip = (1+nh)//2 
            im = (1-nh)//2 

            fi[2] = ip*sqm[ip]
            fi[3] = im*nsf*sqm[ip]
            fi[4] = ip*nsf*sqm[im]
            fi[5] = im*sqm[im]

        else:
            sf = [(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5]
            omega = [sqrt(p[0]+pp),fmass/(sqrt(p[0]+pp))]
            ip = (1+nh)//2
            im = (1-nh)//2
            sfomeg = [sf[0]*omega[ip],sf[1]*omega[im]]
            pp3 = max(pp+p[3],0.)
            if (pp3 == 0.):
                chi1 = complex(-nh,0.) 
            else:
                chi1 = complex(nh*p[1]/sqrt(2.*pp*pp3),\
                p[2]/sqrt(2.*pp*pp3))
            chi = [complex(sqrt(pp3*0.5/pp)),chi1]

            fi[2] = sfomeg[0]*chi[im]
            fi[3] = sfomeg[0]*chi[ip]
            fi[4] = sfomeg[1]*chi[im]
            fi[5] = sfomeg[1]*chi[ip] 
    
    else: 
        sqp0p3 = sqrt(max(p[0]+p[3],0.))*nsf
        if (sqp0p3 == 0.):
            chi1 = complex(-nhel*sqrt(2.*p[0]),0.)
        else:
            chi1 = complex(nh*p[1]/sqp0p3,p[2]/sqp0p3)
        chi = [complex(sqp0p3,0.),chi1]
        if (nh == 1):
            fi[2] = complex(0.,0.)
            fi[3] = complex(0.,0.)
            fi[4] = chi[0]
            fi[5] = chi[1] 
        else:
            fi[2] = chi[1]
            fi[3] = chi[0]
            fi[4] = complex(0.,0.)
            fi[5] = complex(0.,0.) 
    
    return fi 

def oxxxxx(p,fmass,nhel,nsf):
    """ initialize an outgoing fermion"""
    
    fo = WaveFunction(2)
    fo[0] = complex(p[0]*nsf,p[3]*nsf)
    fo[1] = complex(p[1]*nsf,p[2]*nsf)
    nh = nhel*nsf
    if (fmass != 0.):
        pp = min(p[0],sqrt(p[1]**2 + p[2]**2 + p[3]**2 ))
        if (pp == 0.): 
            sqm = [sqrt(abs(fmass))]
            sqm.append( sign(sqm[0], fmass)) 
            ip = int(-((1-nh)//2) * nhel)
            im = int((1+nh)//2 * nhel)
            
            fo[2] = im*sqm[abs(im)]
            fo[3] = ip*nsf*sqm[abs(im)]
            fo[4] = im*nsf*sqm[abs(ip)]
            fo[5] = ip*sqm[abs(ip)]

        else:
            sf = [(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5]
            omega = [sqrt(p[0]+pp),fmass/(sqrt(p[0]+pp))]
            ip = (1+nh)//2
            im = (1-nh)//2
            sfomeg = [sf[0]*omega[ip],sf[1]*omega[im]]
            pp3 = max(pp+p[3],0.)
            if (pp3 == 0.):
                chi1 = complex(-nh,0.) 
            else:
                chi1 = complex(nh*p[1]/sqrt(2.*pp*pp3),\
                -p[2]/sqrt(2.*pp*pp3))
            chi = [complex(sqrt(pp3*0.5/pp)),chi1]

            fo[2] = sfomeg[1]*chi[im]
            fo[3] = sfomeg[1]*chi[ip]
            fo[4] = sfomeg[0]*chi[im]
            fo[5] = sfomeg[0]*chi[ip] 
            
    else: 
        sqp0p3 = sqrt(max(p[0]+p[3],0.))*nsf
        if (sqp0p3 == 0.):
            chi1 = complex(-nhel*sqrt(2.*p[0]),0.)
        else:
            chi1 = complex(nh*p[1]/sqp0p3,-p[2]/sqp0p3)
        chi = [complex(sqp0p3,0.),chi1]
        if (nh == 1):
            fo[2] = chi[0]
            fo[3] = chi[1]
            fo[4] = complex(0.,0.)
            fo[5] = complex(0.,0.) 
        else:
            fo[2] = complex(0.,0.)
            fo[3] = complex(0.,0.)
            fo[4] = chi[1]
            fo[5] = chi[0] 
    
    return fo

def vxxxxx(p,vmass,nhel,nsv):
    """ initialize a vector wavefunction. nhel=4 is for checking BRST"""
    
    vc = WaveFunction(3)
    
    sqh = sqrt(0.5)
    nsvahl = nsv*abs(nhel)
    pt2 = p[1]**2 + p[2]**2 
    pp = min(p[0],sqrt(pt2 + p[3]**2))
    pt = min(pp,sqrt(pt2))

    vc[0] = complex(p[0]*nsv,p[3]*nsv)
    vc[1] = complex(p[1]*nsv,p[2]*nsv)

    if (nhel == 4):
        if (vmass == 0.):
            vc[2] = 1.
            vc[3]=p[1]/p[0]
            vc[4]=p[2]/p[0]
            vc[5]=p[3]/p[0]
        else:
            vc[2] = p[0]/vmass
            vc[3] = p[1]/vmass
            vc[4] = p[2]/vmass
            vc[5] = p[3]/vmass
        
        return vc 

    if (vmass != 0.):
        hel0 = 1.-abs(nhel) 

        if (pp == 0.):
            vc[2] = complex(0.,0.)
            vc[3] = complex(-nhel*sqh,0.)
            vc[4] = complex(0.,nsvahl*sqh) 
            vc[5] = complex(hel0,0.)

        else:
            emp = p[0]/(vmass*pp)
            vc[2] = complex(hel0*pp/vmass,0.)
            vc[5] = complex(hel0*p[3]*emp+nhel*pt/pp*sqh)
            if (pt != 0.):
                pzpt = p[3]/(pp*pt)*sqh*nhel
                vc[3] = complex(hel0*p[1]*emp-p[1]*pzpt, \
                    -nsvahl*p[2]/pt*sqh)
                vc[4] = complex(hel0*p[2]*emp-p[2]*pzpt, \
                    nsvahl*p[1]/pt*sqh) 
            else:
                vc[3] = complex(-nhel*sqh,0.)
                vc[4] = complex(0.,nsvahl*sign(sqh,p[3]))
    else: 
        pp = p[0]
        pt = sqrt(p[1]**2 + p[2]**2)
        vc[2] = complex(0.,0.)
        vc[5] = complex(nhel*pt/pp*sqh)
        if (pt != 0.):
            pzpt = p[3]/(pp*pt)*sqh*nhel
            vc[3] = complex(-p[1]*pzpt,-nsv*p[2]/pt*sqh)
            vc[4] = complex(-p[2]*pzpt,nsv*p[1]/pt*sqh)
        else:
            vc[3] = complex(-nhel*sqh,0.)
            vc[4] = complex(0.,nsv*sign(sqh,p[3]))
    
    return vc

def sign(x,y):
    """Fortran's sign transfer function"""
    try:
        cmp = (y < 0.)
    except TypeError:
        # y should be complex
        if abs(y.imag) < 1e-6 * abs(y.real):
            y = y.real
        else:
            raise
    finally:
        if (y < 0.):
            return -abs(x) 
        else:
            return abs(x) 

def sxxxxx(p,nss):
    """initialize a scalar wavefunction"""
    
    sc = WaveFunction(1)
    
    sc[2] = complex(1.,0.)
    sc[0] = complex(p[0]*nss,p[3]*nss)
    sc[1] = complex(p[1]*nss,p[2]*nss)
    return sc


def txxxxx(p, tmass, nhel, nst):
    """ initialize a tensor wavefunction"""
    
    tc = WaveFunction(5)
    
    sqh = sqrt(0.5)
    sqs = sqrt(1/6)

    pt2 = p[1]**2 + p[2]**2
    pp = min(p[0],sqrt(pt2+p[3]**2))
    pt = min(pp,sqrt(pt2))

    ft = {}
    ft[(4,0)] = complex(p[0], p[3]) * nst
    ft[(5,0)] = complex(p[1], p[2]) * nst

    if ( nhel >= 0 ): 
        #construct eps+
        ep = [0] * 4
        
        if ( pp == 0 ):
            #ep[0] = 0
            ep[1] = -sqh
            ep[2] = complex(0, nst*sqh)
            #ep[3] = 0
        else:            
            #ep[0] = 0
            ep[3] = pt/pp*sqh
            if (pt != 0):
               pzpt = p[3]/(pp*pt)*sqh
               ep[1] = complex( -p[1]*pzpt , -nst*p[2]/pt*sqh )
               ep[2] = complex( -p[2]*pzpt ,  nst*p[1]/pt*sqh )
            else:
               ep[1] = -sqh 
               ep[2] = complex( 0 , nst*sign(sqh,p[3]) )
            
         
     
    if ( nhel <= 0 ): 
        #construct eps-
        em = [0] * 4
        if ( pp == 0 ):
            #em[0] = 0
            em[1] = sqh 
            em[2] = complex( 0 , nst*sqh )
            #em[3] = 0
        else:
            #em[0] = 0
            em[3] = -pt/pp*sqh
            if pt:
               pzpt = -p[3]/(pp*pt)*sqh
               em[1] = complex( -p[1]*pzpt , -nst*p[2]/pt*sqh )
               em[2] = complex( -p[2]*pzpt ,  nst*p[1]/pt*sqh )
            else:
               em[1] = sqh
               em[2] = complex( 0 , nst*sign(sqh,p[3]) )
            
    
    if ( abs(nhel) <= 1 ):  
        #construct eps0
        e0 = [0] * 4
        if ( pp == 0 ):
            #e0[0] = complex( rZero )
            #e0[1] = complex( rZero )
            #e0[2] = complex( rZero )
            e0[3] = 1
        else:
            emp = p[0]/(tmass*pp)
            e0[0] = pp/tmass 
            e0[3] = p[3]*emp
            if pt:
               e0[1] = p[1]*emp 
               e0[2] = p[2]*emp 
            #else:
            #   e0[1] = complex( rZero )
            #   e0[2] = complex( rZero )

    if nhel == 2:
        for j in range(4):
            for i in range(4):         
                ft[(i,j)] = ep[i]*ep[j]
    elif nhel == -2:
        for j in range(4):
            for i in range(4):         
                ft[(i,j)] = em[i]*em[j]
    elif tmass == 0:
        for j in range(4):
            for i in range(4):         
                ft[(i,j)] = 0
    elif nhel == 1:
        for j in range(4):
            for i in range(4): 
                ft[(i,j)] = sqh*( ep[i]*e0[j] + e0[i]*ep[j] )
    elif nhel == 0:
        for j in range(4):
            for i in range(4):       
                ft[(i,j)] = sqs*( ep[i]*em[j] + em[i]*ep[j] + 2 *e0[i]*e0[j] )
    elif nhel == -1:
        for j in range(4):
            for i in range(4): 
                ft[(i,j)] = sqh*( em[i]*e0[j] + e0[i]*em[j] )

    else:
        raise Exception, 'invalid helicity TXXXXXX' 



    tc[2] = ft[(0,0)]
    tc[3] = ft[(0,1)]
    tc[4] = ft[(0,2)]
    tc[5] = ft[(0,3)]
    tc[6] = ft[(1,0)]
    tc[7] = ft[(1,1)]
    tc[8] = ft[(1,2)]
    tc[9] = ft[(1,3)]
    tc[10] = ft[(2,0)]
    tc[11] = ft[(2,1)]
    tc[12] = ft[(2,2)]
    tc[13] = ft[(2,3)]
    tc[14] = ft[(3,0)]
    tc[15] = ft[(3,1)]
    tc[16] = ft[(3,2)]
    tc[17] = ft[(3,3)]
    tc[0] = ft[(4,0)]
    tc[1] = ft[(5,0)]

    return tc

def irxxxx(p, mass, nhel, nsr):
    """ initialize a incoming spin 3/2 wavefunction."""
    
    # This routines is a python translation of a routine written by
    # K.Mawatari in fortran dated from the 2008/02/26
    
    ri = WaveFunction(4)
    
    sqh = sqrt(0.5)
    sq2 = sqrt(2)
    sq3 = sqrt(3)
    
    pt2 = p[1]**2 + p[2]**2
    pp = min([p[0], sqrt(pt2+p[3]**2)])
    pt = min([pp,sqrt(pt2)])
    
    rc = {}
    rc[(4,0)] = -1*complex(p[0],p[3])*nsr
    rc[(5,0)] = -1*complex(p[1],p[2])*nsr
    

    nsv = -nsr # nsv=+1 for final, -1 for initial   
        
    # Construct eps+
    if nhel > 0:
        ep = [0] * 4
        if pp:
            #ep[0] = 0
            ep[3] = pt/pp*sqh
            if pt:
                pzpt = p[3]/(pp*pt)*sqh
                ep[1] = complex( -p[1]*pzpt , -nsv*p[2]/pt*sqh )
                ep[2] = complex( -p[2]*pzpt ,  nsv*p[1]/pt*sqh )
            else:
                ep[1] = -sqh 
                ep[2] = complex( 0 , nsv*sign(sqh,p[3]) )
        else:
            #ep[0] = 0d0
            ep[1] = -sqh
            ep[2] = complex(0, nsv * sqh)
            #ep[3] = 0d0

         
    if ( nhel < 0 ): 
        #construct eps-
        em = [0] * 4
        if ( pp == 0 ):
            #em[0] = 0
            em[1] = sqh 
            em[2] = complex( 0 , nsv*sqh )
            #em[3] = 0
        else:
            #em[0] = 0
            em[3] = -pt/pp*sqh
            if pt:
                pzpt = -p[3]/(pp*pt)*sqh
                em[1] = complex( -p[1]*pzpt , -nsv*p[2]/pt*sqh )
                em[2] = complex( -p[2]*pzpt ,  nsv*p[1]/pt*sqh )
            else:
                em[1] = sqh
                em[2] = complex( 0 , nsv*sign(sqh,p[3]) )            
                
    if ( abs(nhel) <= 1 ):  
        #construct eps0
        e0 = [0] * 4
        if ( pp == 0 ):
            #e0[0] = complex( rZero )
            #e0[1] = complex( rZero )
            #e0[2] = complex( rZero )
            e0[3] = 1
        else:
            emp = p[0]/(mass*pp)
            e0[0] = pp/mass 
            e0[3] = p[3]*emp
            if pt:
               e0[1] = p[1]*emp 
               e0[2] = p[2]*emp 
            #else:
            #   e0[1] = complex( rZero )
            #   e0[2] = complex( rZero )

    

    if ( nhel >= -1 ):
        # constract spinor+ 
        fip = [0] * 4
        sf, omega, sfomeg, chi = [0, 0], [0,0], [0,0], [0,0]
        nh = nsr
        if  mass:
            pp = min([p[0],sqrt(p[1]**2+p[2]**2+p[3]**2)])
            if pp == 0:
                sqm = sqrt(mass)
                ip = (1+nh)//2
                im = (1-nh)//2
                fip[0] = ip * sqm
                fip[1] = im * nsr * sqm
                fip[2] = ip * nsr * sqm
                fip[3] = im * sqm
            else:
                sf[0] = float(1+nsr+(1-nsr)*nh)*0.5
                sf[1] = float(1+nsr-(1-nsr)*nh)*0.5
                omega[0] = sqrt(p[0]+pp)
                omega[1] = mass/omega[0]
                ip = ((3+nh)//2) -1 # -1 since they are index 
                im = ((3-nh)//2) -1 # -1 since they are index
                sfomeg[0] = sf[0]*omega[ip]
                sfomeg[1] = sf[1]*omega[im]
                pp3 = max([pp+p[3],0])
                chi[0] = sqrt(pp3*0.5/pp)
                if  pp3 ==0:
                    chi[1] = -nh
                else:
                    chi[1] = complex( nh*p[1] , p[2] )/sqrt(2*pp*pp3)
            
                fip[0] = sfomeg[0]*chi[im]
                fip[1] = sfomeg[0]*chi[ip]
                fip[2] = sfomeg[1]*chi[im]
                fip[3] = sfomeg[1]*chi[ip]
        else:
            sqp0p3 = sqrt(max([p[0]+p[3],0])) * nsr
            chi[0] = sqp0p3
            if  sqp0p3 == 0:
                chi[1] = -nhel *  sqrt(2*p[0])
            else:
                chi[1] = complex( nh*p[1], p[2] )/sqp0p3
            if  nh == 1:
                #fip[0] = complex( rZero )
                #fip[1] = complex( rZero )
                fip[2] = chi[0]
                fip[3] = chi[1]
            else:
                fip[0] = chi[1]
                fip[1] = chi[0]
                #fip(3) = complex( rZero )
                #fip(4) = complex( rZero )
            
    if ( nhel <= 1 ):
        # constract spinor- 
        fim = [0] * 4
        sf, omega, sfomeg, chi = [0, 0], [0,0], [0,0], [0,0]
        nh = -nsr
        if  mass:
            pp = min([p[0],sqrt(p[1]**2+p[2]**2+p[3]**2)])
            if pp == 0:
                sqm = sqrt(mass)
                ip = (1+nh)/2
                im = (1-nh)/2
                fim[0] = ip * sqm
                fim[1] = im * nsr * sqm
                fim[2] = ip * nsr * sqm
                fim[3] = im * sqm
            else:
                sf[0] = float(1+nsr+(1-nsr)*nh)*0.5
                sf[1] = float(1+nsr-(1-nsr)*nh)*0.5
                omega[0] = sqrt(p[0]+pp)
                omega[1] = mass/omega[0]
                ip = (3+nh)//2 -1
                im = (3-nh)//2 -1
                sfomeg[0] = sf[0]*omega[ip]
                sfomeg[1] = sf[1]*omega[im]
                pp3 = max([pp+p[3],0])
                chi[0] = sqrt(pp3*0.5/pp)
                if  pp3 ==0:
                    chi[1] = -nh
                else:
                    chi[1] = complex( nh*p[1] , p[2] )/sqrt(2*pp*pp3)
            
                fim[0] = sfomeg[0]*chi[im]
                fim[1] = sfomeg[0]*chi[ip]
                fim[2] = sfomeg[1]*chi[im]
                fim[3] = sfomeg[1]*chi[ip]
        else:
            sqp0p3 = sqrt(max([p[0]+p[3],0])) * nsr
            chi[0] = sqp0p3
            if  sqp0p3 == 0:
                chi[1] = -nhel *  sqrt(2*p[0])
            else:
                chi[1] = complex( nh*p[1], p[2] )/sqp0p3
            if  nh == 1:
                #fip[0] = complex( rZero )
                #fip[1] = complex( rZero )
                fim[2] = chi[0]
                fim[3] = chi[1]
            else:
                fim[0] = chi[1]
                fim[1] = chi[0]
                #fip(3) = complex( rZero )
                #fip(4) = complex( rZero )        
      
    

    # recurent relation put her for optimization
    cond1  = (pt == 0 and p[3] >= 0)
    cond2  = (pt == 0 and p[3] < 0)
    
    # spin-3/2 fermion wavefunction
    if nhel == 3:
        for i,j in product(range(4), range(4)):
            rc[(i, j)] = ep[i] *fip[j]
    
    elif nhel == 1:
        for i,j in product(range(4), range(4)):
            if cond1:
                rc[(i,j)] = sq2/sq3*e0[i]*fip[j] +1/sq3*ep[i]*fim[j]
            elif cond2:
                rc[(i,j)] = sq2/sq3*e0[i]*fip[j] -1/sq3*ep[i]*fim[j]
            else:
                rc[(i,j)] = sq2/sq3*e0[i]*fip[j] + \
                                   1/sq3*ep[i]*fim[j] *complex(p[1],nsr*p[2])/pt  
    elif nhel == -1:
        for i,j in product(range(4), range(4)):
            if cond1:
                rc[(i,j)] = 1/sq3*em[i]*fip[j] +sq2/sq3*e0[i]*fim[j]
            elif cond2:
                rc[(i,j)] = 1/sq3*em[i]*fip[j] -sq2/sq3*e0[i]*fim[j]
            else:
                rc[(i,j)] = 1/sq3*em[i]*fip[j] + \
                                sq2/sq3*e0[i]*fim[j] *complex(p[1],nsr*p[2])/pt  
    else:
        for i,j in product(range(4), range(4)):
            if cond1:
                rc[(i, j)] = em[i] *fim[j]
            elif cond2:
                rc[(i, j)] = -em[i] *fim[j]
            else:
                rc[(i, j)] = em[i]*fim[j] *complex(p[1],nsr*p[2])/pt 
                
    ri[2] = rc[(0,0)]
    ri[3] = rc[(0,1)]
    ri[4] = rc[(0,2)]
    ri[5] = rc[(0,3)]
    ri[6] = rc[(1,0)]
    ri[7] = rc[(1,1)]
    ri[8] = rc[(1,2)]
    ri[9] = rc[(1,3)]
    ri[10] = rc[(2,0)]
    ri[11] = rc[(2,1)]
    ri[12] = rc[(2,2)]
    ri[13] = rc[(2,3)]
    ri[14] = rc[(3,0)]
    ri[15] = rc[(3,1)]
    ri[16] = rc[(3,2)]
    ri[17] = rc[(3,3)]
    ri[0] = rc[(4,0)]
    ri[1] = rc[(5,0)]              

    return ri

def orxxxx(p, mass, nhel, nsr):
    """ initialize a incoming spin 3/2 wavefunction."""
    
    # This routines is a python translation of a routine written by
    # K.Mawatari in fortran dated from the 2008/02/26

   
    ro = WaveFunction(spin=4)
    
    sqh = sqrt(0.5)
    sq2 = sqrt(2)
    sq3 = sqrt(3)
    
    pt2 = p[1]**2 + p[2]**2
    pp = min([p[0], sqrt(pt2+p[3]**2)])
    pt = min([pp,sqrt(pt2)])
    rc = {}
    rc[(4,0)] = complex(p[0],p[3])*nsr
    rc[(5,0)] = complex(p[1],p[2])*nsr
    

    nsv = nsr # nsv=+1 for final, -1 for initial   
        
    # Construct eps+
    if nhel > 0:
        ep = [0] * 4
        if pp:
            #ep[0] = 0
            ep[3] = pt/pp*sqh
            if pt:
                pzpt = p[3]/(pp*pt)*sqh
                ep[1] = complex( -p[1]*pzpt , -nsv*p[2]/pt*sqh )
                ep[2] = complex( -p[2]*pzpt ,  nsv*p[1]/pt*sqh )
            else:
                ep[1] = -sqh 
                ep[2] = complex( 0 , nsv*sign(sqh,p[3]) )
        else:
            #ep[0] = 0d0
            ep[1] = -sqh
            ep[2] = complex(0, nsv * sqh)
            #ep[3] = 0d0
         
    if ( nhel < 0 ): 
        #construct eps-
        em = [0] * 4
        if ( pp == 0 ):
            #em[0] = 0
            em[1] = sqh 
            em[2] = complex( 0 , nsv*sqh )
            #em[3] = 0
        else:
            #em[0] = 0
            em[3] = -pt/pp*sqh
            if pt:
                pzpt = -p[3]/(pp*pt)*sqh
                em[1] = complex( -p[1]*pzpt , -nsv*p[2]/pt*sqh )
                em[2] = complex( -p[2]*pzpt ,  nsv*p[1]/pt*sqh )
            else:
                em[1] = sqh
                em[2] = complex( 0 , nsv*sign(sqh,p[3]) )            
                
    if ( abs(nhel) <= 1 ):  
        #construct eps0
        e0 = [0] * 4
        if ( pp == 0 ):
            #e0[0] = complex( rZero )
            #e0[1] = complex( rZero )
            #e0[2] = complex( rZero )
            e0[3] = 1
        else:
            emp = p[0]/(mass*pp)
            e0[0] = pp/mass 
            e0[3] = p[3]*emp
            if pt:
               e0[1] = p[1]*emp 
               e0[2] = p[2]*emp 
            #else:
            #   e0[1] = complex( rZero )
            #   e0[2] = complex( rZero )

    if nhel >= -1:
        #constract spinor+ 
        nh = nsr
        sqm, fop, omega, sf, sfomeg = [0]*2,[0]*4,[0]*2,[0]*2,[0]*2
        chi = [0]*2
        if mass:
            pp = min([p[0],sqrt(p[1]**2+p[2]**2+p[3]**2)])
            if ( pp == 0):
                sqm[0] = sqrt(abs(mass)) # possibility of negative fermion masses
                sqm[1] = sign(sqm[0],mass) # possibility of negative fermion masses
                ip = -((1+nh)/2)
                im =  (1-nh)/2
                fop[0] = im     * sqm[im]
                fop[1] = ip*nsr * sqm[im]
                fop[2] = im*nsr * sqm[-ip]
                fop[3] = ip     * sqm[-ip]
            else:
                pp = min(p[0],sqrt(p[1]**2+p[2]**2+p[3]**2))
                sf[0] = (1+nsr+(1-nsr)*nh)*0.5
                sf[1] = (1+nsr-(1-nsr)*nh)*0.5
                omega[0] = sqrt(p[0]+pp)
                omega[1] = mass/omega[0]
                ip = (3+nh)//2  -1 # -1 since this is index
                im = (3-nh)//2  -1 # -1 since this is index 
                sfomeg[0] = sf[0]*omega[ip]
                sfomeg[1] = sf[1]*omega[im]
                pp3 = max([pp+p[3],0])
                chi[0] = sqrt(pp3*0.5/pp)
                if pp3 == 0:
                    chi[1] = -nh 
                else:
                    chi[1] = complex( nh*p[1] , -p[2] )/sqrt(2*pp*pp3)

            
                fop[0] = sfomeg[1]*chi[im]
                fop[1] = sfomeg[1]*chi[ip]
                fop[2] = sfomeg[0]*chi[im]
                fop[3] = sfomeg[0]*chi[ip]

        else:
            if(p[1] == 0 and p[2] == 0 and p[3] < 0):
                sqp0p3 = 0
            else:
                sqp0p3 = sqrt(max(p[0]+p[3], 0))*nsr
                
            chi[0] =  sqp0p3
            if ( sqp0p3 == 0 ):
                chi[1] = complex(-nhel )*sqrt(2*p[0])
            else:
                chi[1] = complex( nh*p[1], -p[2] )/sqp0p3
         
            if ( nh == 1 ):
                fop[0] = chi[0]
                fop[1] = chi[1]
                #fop[2] = 0
                #fop[3] = 0
            else:
                #fop[0] = 0
                #fop[1] = 0
                fop[2] = chi[1]
                fop[3] = chi[0]
         
    
    if ( nhel < 2 ):
        # constract spinor+ 
        sqm, fom, omega, sf, sfomeg = [0]*2,[0]*4,[0]*2,[0]*2,[0]*2
        chi = [0]*2

        
        nh = -nsr
        if mass:
            pp = min([p[0],sqrt(p[1]**2+p[2]**2+p[3]**2)])
            if ( pp == 0):
                sqm[0] = sqrt(abs(mass)) # possibility of negative fermion masses
                sqm[1] = sign(sqm[0],mass) # possibility of negative fermion masses
                ip = -((1+nh)/2)
                im =  (1-nh)/2
            
                fom[0] = im     * sqm[im]
                fom[1] = ip*nsr * sqm[im]
                fom[2] = im*nsr * sqm[-ip]
                fom[3] = ip     * sqm[-ip]
            
            else:
                pp = min([p[0],sqrt(p[1]**2+p[2]**2+p[3]**2)])
                sf[0] = (1+nsr+(1-nsr)*nh)*0.5
                sf[1] = (1+nsr-(1-nsr)*nh)*0.5
                omega[0] = sqrt(p[0]+pp)
                omega[1] = mass/omega[0]
                ip = (3+nh)//2 -1 #-1 since ip is an index
                im = (3-nh)//2 -1 
                sfomeg[0] = sf[0]*omega[ip]
                sfomeg[1] = sf[1]*omega[im]
                pp3 = max([pp+p[3], 0])
                chi[0] = sqrt(pp3*0.5/pp)
                if ( pp3 == 0):
                    chi[1] = -nh
                else:
                    chi[1] = complex( nh*p[1] , -p[2] )/sqrt(2*pp*pp3)
            
            
                fom[0] = sfomeg[1]*chi[im]
                fom[1] = sfomeg[1]*chi[ip]
                fom[2] = sfomeg[0]*chi[im]
                fom[3] = sfomeg[0]*chi[ip]
        else:
            if(p[1] == 0 == p[2] and p[3] < 0):
                sqp0p3 = 0
            else:
                sqp0p3 = sqrt(max([p[0]+p[3],0]))*nsr
            chi[0] = sqp0p3
            if ( sqp0p3 == 0):
                chi[1] = complex(-nhel )*sqrt(2*p[0])
            else:
                chi[1] = complex( nh*p[1], -p[2] )/sqp0p3
            if ( nh == 1 ):
                fom[0] = chi[0]
                fom[1] = chi[1]
                #fom[2] = 0
                #fom[3] = 0
            else:
                #fom[1] = 0
                #fom[2] = 0
                fom[2] = chi[1]
                fom[3] = chi[0]

    cond1 = ( pt==0 and p[3]>=0)
    cond2= (pt==0 and p[3]<0)

   
    # spin-3/2 fermion wavefunction
    if nhel == 3:
        for i,j in product(range(4), range(4)):
            rc[(i, j)] = ep[i] *fop[j]  
    

    elif nhel == 1:
        for i,j in product(range(4), range(4)):
            if cond1:
                rc[(i,j)] = sq2/sq3*e0[i]*fop[j] + 1/sq3*ep[i]*fom[j]
            elif cond2:
                rc[(i,j)] = sq2/sq3*e0[i]*fop[j] - 1/sq3*ep[i]*fom[j]
            else:
                rc[(i,j)] = sq2/sq3*e0[i]*fop[j] + 1/sq3*ep[i]*fom[j] * \
                                                      complex(p[1],-nsr*p[2])/pt  
                
    elif nhel == -1:
        for i,j in product(range(4), range(4)):
            if cond1:
                rc[(i,j)] = 1/sq3*em[i]*fop[j]+sq2/sq3*e0[i]*fom[j]
            elif cond2:
                rc[(i,j)] =1/sq3*em[i]*fop[j]-sq2/sq3*e0[i]*fom[j]
            else:
                rc[(i,j)] =  1/sq3*em[i]*fop[j] + sq2/sq3*e0[i]*fom[j] *\
                                                      complex(p[1],-nsr*p[2])/pt              
    else:
        for i,j in product(range(4), range(4)):
            if cond1:
                rc[(i,j)] = em[i] * fom[j]
            elif cond2:
                rc[(i,j)] = - em[i] * fom[j]
            else:
                rc[(i,j)] = em[i] * fom[j] * complex(p[1],-nsr*p[2])/pt 



    ro[2] = rc[(0,0)]
    ro[3] = rc[(0,1)]
    ro[4] = rc[(0,2)]
    ro[5] = rc[(0,3)]
    ro[6] = rc[(1,0)]
    ro[7] = rc[(1,1)]
    ro[8] = rc[(1,2)]
    ro[9] = rc[(1,3)]
    ro[10] = rc[(2,0)]
    ro[11] = rc[(2,1)]
    ro[12] = rc[(2,2)]
    ro[13] = rc[(2,3)]
    ro[14] = rc[(3,0)]
    ro[15] = rc[(3,1)]
    ro[16] = rc[(3,2)]
    ro[17] = rc[(3,3)]
    ro[0] = rc[(4,0)]
    ro[1] = rc[(5,0)]
    
    return ro
    


