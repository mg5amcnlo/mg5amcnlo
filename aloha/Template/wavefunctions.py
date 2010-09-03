import cmath 
import math

class WaveFunction(list):
	"""a objet for a WaveFunction"""
	
	spin_to_size={0:1,
				  1:3,
				  2:6,
				  3:6}
	
	def __init__(self, spin):
		"""Init the list with zero value"""
		
		self.spin = spin
		list.__init__(self, [0]*spin_to_size[spin])
		

def ixxxxx(p,fmass,nhel,nsf):
	"""Defines an inflow fermion."""
	
	fi = WaveFunction(2)
	
	fi[4] = complex(p[0]*nsf,p[3]*nsf)
	fi[5] = complex(p[1]*nsf,p[2]*nsf) 
	
	nh = nhel*nsf 

	if (fmass != 0.):
		pp = min(p[0],math.sqrt(math.pow(p[1],2)+math.pow(p[2],2)+math.pow(p[3],2)))
		if (pp == 0.): 
			sqm = math.sqrt(abs(fmass)) 
			ip = (1+nh)/2 
			im = (1-nh)/2 

			fi[0] = ip*sqm
			fi[1] = im*nsf*sqm
			fi[2] = ip*nsf*sqm
			fi[3] = im*sqm

		else:
			sf = [(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5]
			omega = [math.sqrt(p[0]+pp),fmass/(math.sqrt(p[0]+pp))]
			ip = (1+nh)/2
			im = (1-nh)/2
			sfomeg = [sf[0]*omega[ip],sf[1]*omega[im]]
			pp3 = max(pp+p[3],0.)
			if (pp3 == 0.):
				chi1 = complex(-nh,0.) 
			else:
				chi1 = complex(nh*p[1]/math.sqrt(2.*pp*pp3),\
				p[2]/math.sqrt(2.*pp*pp3))
			chi = [complex(math.sqrt(pp3*0.5/pp)),chi1]

			fi[0] = sfomeg[0]*chi[im]
			fi[1] = sfomeg[0]*chi[ip]
			fi[2] = sfomeg[1]*chi[im]
			fi[3] = sfomeg[1]*chi[ip] 
	
	else: 
		sqp0p3 = math.sqrt(max(p[0]+p[3],0.))*nsf
		if (sqp0p3 == 0.):
			chi1 = complex(-nhel*math.sqrt(2.*p[0]),0.)
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
		pp = min(p[0],math.sqrt(math.pow(p[1],2)+math.pow(p[2],2)+math.pow(p[3],2)))
		if (pp == 0.): 
			sqm = math.sqrt(abs(fmass)) 
			ip = -((1+nh)/2) 
			im = (1-nh)/2 

			fo[0] = im*sqm
			fo[1] = ip*nsf*sqm
			fo[2] = im*nsf*sqm
			fo[3] = ip*sqm

		else:
			sf = [(1+nsf+(1-nsf)*nh)*0.5,(1+nsf-(1-nsf)*nh)*0.5]
			omega = [math.sqrt(p[0]+pp),fmass/(math.sqrt(p[0]+pp))]
			ip = (1+nh)/2
			im = (1-nh)/2
			sfomeg = [sf[0]*omega[ip],sf[1]*omega[im]]
			pp3 = max(pp+p[3],0.)
			if (pp3 == 0.):
				chi1 = complex(-nh,0.) 
			else:
				chi1 = complex(nh*p[1]/math.sqrt(2.*pp*pp3),\
				-p[2]/math.sqrt(2.*pp*pp3))
			chi = [complex(math.sqrt(pp3*0.5/pp)),chi1]

			fo[0] = sfomeg[1]*chi[im]
			fo[1] = sfomeg[1]*chi[ip]
			fo[2] = sfomeg[0]*chi[im]
			fo[3] = sfomeg[0]*chi[ip] 
			
	else: 
		sqp0p3 = math.sqrt(max(p[0]+p[3],0.))*nsf
		if (sqp0p3 == 0.):
			chi1 = complex(-nhel*math.sqrt(2.*p[0]),0.)
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
	
	sqh = math.sqrt(0.5)
	nsvahl = nsv*abs(nhel)
	pt2 = math.pow(p[1],2)+math.pow(p[2],2)
	pp = min(p[0],math.sqrt(pt2+pow(p[3],2)))
	pt = min(pp,math.sqrt(pt2))

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
		pt = math.sqrt(math.pow(p[1],2)+math.pow(p[2],2))
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

