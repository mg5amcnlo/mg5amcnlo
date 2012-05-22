from __future__ import division 
import math
from math import sqrt, pow

class WaveFunction(list):
    """a objet for a WaveFunction"""
    
    spin_to_size={0:1,
                  1:3,
                  2:6,
                  3:6,
                  5:18}
    
    def __init__(self, spin= None, size=None):
        """Init the list with zero value"""
        
        if spin:
            size = self.spin_to_size[spin]
        list.__init__(self, [0]*size)
        

def ixxxxx(p,fmass,nhel,nsf):
    """Defines an inflow fermion."""
    
    fi = WaveFunction(2)
    
    fi[4] = complex(p[0]*nsf,p[3]*nsf)
    fi[5] = complex(p[1]*nsf,p[2]*nsf) 
    
    nh = nhel*nsf 

    if (fmass != 0.):
        pp = min(p[0],sqrt(p[1]**2 + p[2]**2 + p[3]**2 ))
        if (pp == 0.): 
            sqm = sqrt(abs(fmass)) 
            ip = (1+nh)/2 
            im = (1-nh)/2 

            fi[0] = ip*sqm
            fi[1] = im*nsf*sqm
            fi[2] = ip*nsf*sqm
            fi[3] = im*sqm

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

            fi[0] = sfomeg[0]*chi[im]
            fi[1] = sfomeg[0]*chi[ip]
            fi[2] = sfomeg[1]*chi[im]
            fi[3] = sfomeg[1]*chi[ip] 
    
    else: 
        sqp0p3 = sqrt(max(p[0]+p[3],0.))*nsf
        if (sqp0p3 == 0.):
            chi1 = complex(-nhel*sqrt(2.*p[0]),0.)
        else:
            chi1 = complex(nh*p[1]/sqp0p3,p[2]/sqp0p3)
        chi = [complex(sqp0p3,0.),chi1]
        if (nh == 1):
            fi[0] = complex(0.,0.)
            fi[1] = complex(0.,0.)
            fi[2] = chi[0]
            fi[3] = chi[1] 
        else:
            fi[0] = chi[1]
            fi[1] = chi[0]
            fi[2] = complex(0.,0.)
            fi[3] = complex(0.,0.) 
    
    return fi 

def oxxxxx(p,fmass,nhel,nsf):
    """ initialize an outgoing fermion"""
    
    fo = WaveFunction(2)
         
    fo[4] = complex(p[0]*nsf,p[3]*nsf)
    fo[5] = complex(p[1]*nsf,p[2]*nsf)

    nh = nhel*nsf

    if (fmass != 0.):
        pp = min(p[0],sqrt(p[1]**2 + p[2]**2 + p[3]**2 ))
        if (pp == 0.): 
            sqm = sqrt(abs(fmass)) 
            ip = -((1-nh)/2) * nhel
            im = (1+nh)/2 * nhel

            fo[0] = im*sqm
            fo[1] = ip*nsf*sqm
            fo[2] = im*nsf*sqm
            fo[3] = ip*sqm

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

            fo[0] = sfomeg[1]*chi[im]
            fo[1] = sfomeg[1]*chi[ip]
            fo[2] = sfomeg[0]*chi[im]
            fo[3] = sfomeg[0]*chi[ip] 
            
    else: 
        sqp0p3 = sqrt(max(p[0]+p[3],0.))*nsf
        if (sqp0p3 == 0.):
            chi1 = complex(-nhel*sqrt(2.*p[0]),0.)
        else:
            chi1 = complex(nh*p[1]/sqp0p3,-p[2]/sqp0p3)
        chi = [complex(sqp0p3,0.),chi1]
        if (nh == 1):
            fo[0] = chi[0]
            fo[1] = chi[1]
            fo[2] = complex(0.,0.)
            fo[3] = complex(0.,0.) 
        else:
            fo[0] = complex(0.,0.)
            fo[1] = complex(0.,0.)
            fo[2] = chi[1]
            fo[3] = chi[0] 
    
    return fo

def vxxxxx(p,vmass,nhel,nsv):
    """ initialize a vector wavefunction. nhel=4 is for checking BRST"""
    
    vc = WaveFunction(3)
    
    sqh = sqrt(0.5)
    nsvahl = nsv*abs(nhel)
    pt2 = p[1]**2 + p[2]**2 
    pp = min(p[0],sqrt(pt2 + p[3]**2))
    pt = min(pp,sqrt(pt2))

    vc[4] = complex(p[0]*nsv,p[3]*nsv)
    vc[5] = complex(p[1]*nsv,p[2]*nsv)

    if (nhel == 4):
        if (vmass == 0.):
            vc[0] = 1.
            vc[1]=p[1]/p[0]
            vc[2]=p[2]/p[0]
            vc[3]=p[3]/p[0]
        else:
            vc[0] = p[0]/vmass
            vc[1] = p[1]/vmass
            vc[2] = p[2]/vmass
            vc[3] = p[3]/vmass
        
        return vc 

    if (vmass != 0.):
        hel0 = 1.-abs(nhel) 

        if (pp == 0.):
            vc[0] = complex(0.,0.)
            vc[1] = complex(-nhel*sqh,0.)
            vc[2] = complex(0.,nsvahl*sqh) 
            vc[3] = complex(hel0,0.)

        else:
            emp = p[0]/(vmass*pp)
            vc[0] = complex(hel0*pp/vmass,0.)
            vc[3] = complex(hel0*p[3]*emp+nhel*pt/pp*sqh)
            if (pt != 0.):
                pzpt = p[3]/(pp*pt)*sqh*nhel
                vc[1] = complex(hel0*p[1]*emp-p[1]*pzpt, \
                    -nsvahl*p[2]/pt*sqh)
                vc[2] = complex(hel0*p[2]*emp-p[2]*pzpt, \
                    nsvahl*p[1]/pt*sqh) 
            else:
                vc[1] = complex(-nhel*sqh,0.)
                vc[2] = complex(0.,nsvahl*sign(sqh,p[3]))
    else: 
        pp = p[0]
        pt = sqrt(p[1]**2 + p[2]**2)
        vc[0] = complex(0.,0.)
        vc[3] = complex(nhel*pt/pp*sqh)
        if (pt != 0.):
            pzpt = p[3]/(pp*pt)*sqh*nhel
            vc[1] = complex(-p[1]*pzpt,-nsv*p[2]/pt*sqh)
            vc[2] = complex(-p[2]*pzpt,nsv*p[1]/pt*sqh)
        else:
            vc[1] = complex(-nhel*sqh,0.)
            vc[2] = complex(0.,nsv*sign(sqh,p[3]))
    
    return vc

def sign(x,y):
    """Fortran's sign transfer function"""
    if (y < 0.):
        return -abs(x) 
    else:
        return abs(x) 
    

def sxxxxx(p,nss):
    """initialize a scalar wavefunction"""
    
    sc = WaveFunction(1)
    
    sc[0] = complex(1.,0.)
    sc[1] = complex(p[0]*nss,p[3]*nss)
    sc[2] = complex(p[1]*nss,p[2]*nss)
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
            em[2] = cpmplex( 0 , nst*sqh )
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
            #e0[0] = dcmplx( rZero )
            #e0[1] = dcmplx( rZero )
            #e0[2] = dcmplx( rZero )
            e0[3] = 1
        else:
            emp = p[0]/(tmass*pp)
            e0[0] = pp/tmass 
            e0[3] = p[3]*emp
            if pt:
               e0[1] = p[1]*emp 
               e0[2] = p[2]*emp 
            #else:
            #   e0[1] = dcmplx( rZero )
            #   e0[2] = dcmplx( rZero )

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

    tc[0] = ft[(0,0)]
    tc[1] = ft[(0,1)]
    tc[2] = ft[(0,2)]
    tc[3] = ft[(0,3)]
    tc[4] = ft[(1,0)]
    tc[5] = ft[(1,1)]
    tc[6] = ft[(1,2)]
    tc[7] = ft[(1,3)]
    tc[8] = ft[(2,0)]
    tc[9] = ft[(2,1)]
    tc[10] = ft[(2,2)]
    tc[11] = ft[(2,3)]
    tc[12] = ft[(3,0)]
    tc[13] = ft[(3,1)]
    tc[14] = ft[(3,2)]
    tc[15] = ft[(3,3)]
    tc[16] = ft[(4,0)]
    tc[17] = ft[(5,0)]
  
    return tc

    

