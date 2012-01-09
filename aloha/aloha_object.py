################################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this 
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################
##   Diagram of Class
##
##    Variable <--- aloha_lib.ScalarVariable 
##               |
##               +- LorentzObject <--- Gamma
##                                  |
##                                  +- Sigma
##                                  |
##                                  +- P
##
##    list <--- AddVariable   
##           |
##           +- MultVariable  <--- MultLorentz 
##           
##    list <--- LorentzObjectRepresentation <-- ConstantObject
##
################################################################################
from __future__ import division
import aloha.aloha_lib as aloha_lib

#===============================================================================
# P (Momenta)
#===============================================================================
class P(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    contract_first = 1
    
    def __init__(self, lorentz1, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [lorentz1], [], prefactor,\
                                                               ['P%s'%particle])
    
        
    def create_representation(self):
        self.sub0 = aloha_lib.ScalarVariable('P%s_0' % self.particle)
        self.sub1 = aloha_lib.ScalarVariable('P%s_1' % self.particle)
        self.sub2 = aloha_lib.ScalarVariable('P%s_2' % self.particle)
        self.sub3 = aloha_lib.ScalarVariable('P%s_3' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},                              
                                    self.lorentz_ind, [])

#===============================================================================
# Pslash
#===============================================================================
class PSlash(aloha_lib.LorentzObject):
    """ Gamma Matrices """
    
    #gamma0 = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    #gamma1 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]
    #gamma2 = [[0, 0, 0, -complex(0,1)],[0, 0, complex(0,1), 0],
    #                    [0, complex(0,1), 0, 0], [-complex(0,1), 0, 0, 0]]
    #gamma3 = [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]]
    #    
    #gamma = [gamma0, gamma1, gamma2, gamma3]

    def __init__(self, spin1, spin2, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], \
                                                            prefactor=prefactor)
    
    def create_representation(self):
        """create representation"""
        p0 = aloha_lib.ScalarVariable('P%s_0' % self.particle)
        p1 = aloha_lib.ScalarVariable('P%s_1' % self.particle)
        p2 = aloha_lib.ScalarVariable('P%s_2' % self.particle)
        p3 = aloha_lib.ScalarVariable('P%s_3' % self.particle)    
    
    
        gamma = {
             (0, 0): 0, (0, 1): 0, (0, 2): p0-p3, (0, 3): -1*p1+1j*p2,
             (1, 0): 0, (1, 1): 0, (1, 2): -1*p1-1j*p2, (1, 3): p0+p3,
             (2, 0): p0+p3, (2, 1): p1-1j*p2, (2, 2): 0, (2, 3): 0,
             (3, 0): p1+1j*p2, (3, 1): p0-p3, (3, 2): 0, (3, 3): 0}

                
        self.representation = aloha_lib.LorentzObjectRepresentation(gamma,
                                self.lorentz_ind,self.spin_ind)

#===============================================================================
# Mass
#===============================================================================
class Mass(aloha_lib.LorentzObject):
    """ Helas Object for a Mass"""
    
    def __init__(self, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [], prefactor=prefactor)
    
        
    def create_representation(self):
        mass = aloha_lib.ScalarVariable('M%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

#===============================================================================
# OverMass2
#===============================================================================
class OverMass2(aloha_lib.LorentzObject):
    """ Helas Object for 1/M**2 """
    
    def __init__(self, particle, prefactor=1):
        
        self.particle = particle
        
        tag= ['OM%s' % particle]
        aloha_lib.LorentzObject.__init__(self, [], [], prefactor,tag)
    
        
    def create_representation(self):
        mass = aloha_lib.ScalarVariable('OM%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

#===============================================================================
# Width
#===============================================================================
class Width(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    def __init__(self, particle, prefactor=1):

        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [], prefactor=prefactor)
        
    def create_representation(self):
        width = aloha_lib.ScalarVariable('W%s' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                            width, self.lorentz_ind, self.spin_ind)
        
#===============================================================================
# Scalar
#===============================================================================
class Scalar(aloha_lib.LorentzObject):
    """ Helas Object for a Spinor"""
    
    def __init__(self, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [], prefactor=prefactor)
    
        
    def create_representation(self):
        rep = aloha_lib.ScalarVariable('S%s_1' % self.particle)
        self.representation= aloha_lib.LorentzObjectRepresentation(        
                                                                    rep, [], [])        
        
        
#===============================================================================
# Spinor
#===============================================================================
class Spinor(aloha_lib.LorentzObject):
    """ Helas Object for a Spinor"""
    
    contract_first = 1
    
    def __init__(self, spin1, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [spin1], prefactor=prefactor)
    
        
    def create_representation(self):
        self.sub0 = aloha_lib.ScalarVariable('F%s_1' % self.particle)
        self.sub1 = aloha_lib.ScalarVariable('F%s_2' % self.particle)
        self.sub2 = aloha_lib.ScalarVariable('F%s_3' % self.particle)
        self.sub3 = aloha_lib.ScalarVariable('F%s_4' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},         
                                    [],self.spin_ind)

#===============================================================================
# Vector
#===============================================================================
class Vector(aloha_lib.LorentzObject):
    """ Helas Object for a Vector"""
    
    contract_first = 1
    
    def __init__(self, lorentz, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [lorentz], [], prefactor=prefactor)
    
        
    def create_representation(self):
        self.sub0 = aloha_lib.ScalarVariable('V%s_1' % self.particle)
        self.sub1 = aloha_lib.ScalarVariable('V%s_2' % self.particle)
        self.sub2 = aloha_lib.ScalarVariable('V%s_3' % self.particle)
        self.sub3 = aloha_lib.ScalarVariable('V%s_4' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation( 
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},  
                                    self.lorentz_ind, [])

#===============================================================================
# Spin3/2
#===============================================================================
class Spin2(aloha_lib.LorentzObject):
    """ Helas Object for a Spin2"""
    
    def __init__(self, lorentz, spin, particle, prefactor=1):
        
        self.particle = particle
            
        aloha_lib.LorentzObject.__init__(self, [lorentz], [spin], \
                                 prefactor=prefactor)
    
    def create_representation(self):

        self.sub00 = aloha_lib.ScalarVariable('R%s_1' % self.particle)
        self.sub01 = aloha_lib.ScalarVariable('R%s_2' % self.particle)
        self.sub02 = aloha_lib.ScalarVariable('R%s_3' % self.particle)
        self.sub03 = aloha_lib.ScalarVariable('R%s_4' % self.particle)

        self.sub10 = aloha_lib.ScalarVariable('R%s_5' % self.particle)
        self.sub11 = aloha_lib.ScalarVariable('R%s_6' % self.particle)
        self.sub12 = aloha_lib.ScalarVariable('R%s_7' % self.particle)
        self.sub13 = aloha_lib.ScalarVariable('R%s_8' % self.particle)
    
        self.sub20 = aloha_lib.ScalarVariable('R%s_9' % self.particle)
        self.sub21 = aloha_lib.ScalarVariable('R%s_10' % self.particle)
        self.sub22 = aloha_lib.ScalarVariable('R%s_11' % self.particle)
        self.sub23 = aloha_lib.ScalarVariable('R%s_12' % self.particle)
    
        self.sub30 = aloha_lib.ScalarVariable('R%s_13' % self.particle)
        self.sub31 = aloha_lib.ScalarVariable('R%s_14' % self.particle)
        self.sub32 = aloha_lib.ScalarVariable('R%s_15' % self.particle)
        self.sub33 = aloha_lib.ScalarVariable('R%s_16' % self.particle)
        
        rep = {(0,0): self.sub00, (0,1): self.sub01, (0,2): self.sub02, (0,3): self.sub03,
               (1,0): self.sub10, (1,1): self.sub11, (1,2): self.sub12, (1,3): self.sub13,
               (2,0): self.sub20, (2,1): self.sub21, (2,2): self.sub22, (2,3): self.sub23,
               (3,0): self.sub30, (3,1): self.sub31, (3,2): self.sub32, (3,3): self.sub33}
        
                
        self.representation= aloha_lib.LorentzObjectRepresentation( rep, \
                                    self.lorentz_ind, self.spin_ind)


#===============================================================================
# Spin2
#===============================================================================
class Spin2(aloha_lib.LorentzObject):
    """ Helas Object for a Spin2"""
    
    def __init__(self, lorentz1, lorentz2, particle, prefactor=1):
        
        self.particle = particle
        if lorentz2 < lorentz1:
            lorentz1, lorentz2 = lorentz2, lorentz1
            
        aloha_lib.LorentzObject.__init__(self, [lorentz1, lorentz2], [], \
                                 prefactor=prefactor)
    
    def create_representation(self):

        self.sub00 = aloha_lib.ScalarVariable('T%s_1' % self.particle)
        self.sub01 = aloha_lib.ScalarVariable('T%s_2' % self.particle)
        self.sub02 = aloha_lib.ScalarVariable('T%s_3' % self.particle)
        self.sub03 = aloha_lib.ScalarVariable('T%s_4' % self.particle)

        self.sub10 = aloha_lib.ScalarVariable('T%s_5' % self.particle)
        self.sub11 = aloha_lib.ScalarVariable('T%s_6' % self.particle)
        self.sub12 = aloha_lib.ScalarVariable('T%s_7' % self.particle)
        self.sub13 = aloha_lib.ScalarVariable('T%s_8' % self.particle)
	
        self.sub20 = aloha_lib.ScalarVariable('T%s_9' % self.particle)
        self.sub21 = aloha_lib.ScalarVariable('T%s_10' % self.particle)
        self.sub22 = aloha_lib.ScalarVariable('T%s_11' % self.particle)
        self.sub23 = aloha_lib.ScalarVariable('T%s_12' % self.particle)
	
        self.sub30 = aloha_lib.ScalarVariable('T%s_13' % self.particle)
        self.sub31 = aloha_lib.ScalarVariable('T%s_14' % self.particle)
        self.sub32 = aloha_lib.ScalarVariable('T%s_15' % self.particle)
        self.sub33 = aloha_lib.ScalarVariable('T%s_16' % self.particle)
        
        rep = {(0,0): self.sub00, (0,1): self.sub01, (0,2): self.sub02, (0,3): self.sub03,
               (1,0): self.sub10, (1,1): self.sub11, (1,2): self.sub12, (1,3): self.sub13,
               (2,0): self.sub20, (2,1): self.sub21, (2,2): self.sub22, (2,3): self.sub23,
               (3,0): self.sub30, (3,1): self.sub31, (3,2): self.sub32, (3,3): self.sub33}
        
                
        self.representation= aloha_lib.LorentzObjectRepresentation( rep, \
                                    self.lorentz_ind, [])

#===============================================================================
# Gamma
#===============================================================================
class Gamma(aloha_lib.LorentzObject):
    """ Gamma Matrices """
    
    #gamma0 = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    #gamma1 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]
    #gamma2 = [[0, 0, 0, -complex(0,1)],[0, 0, complex(0,1), 0],
    #                    [0, complex(0,1), 0, 0], [-complex(0,1), 0, 0, 0]]
    #gamma3 = [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]]
    #    
    #gamma = [gamma0, gamma1, gamma2, gamma3]
    gamma = { #Gamma0
             (0, 0, 0): 0, (0, 0, 1): 0, (0, 0, 2): 1, (0, 0, 3): 0,
             (0, 1, 0): 0, (0, 1, 1): 0, (0, 1, 2): 0, (0, 1, 3): 1,
             (0, 2, 0): 1, (0, 2, 1): 0, (0, 2, 2): 0, (0, 2, 3): 0,
             (0, 3, 0): 0, (0, 3, 1): 1, (0, 3, 2): 0, (0, 3, 3): 0,
             #Gamma1
             (1, 0, 0): 0, (1, 0, 1): 0, (1, 0, 2): 0, (1, 0, 3): 1,
             (1, 1, 0): 0, (1, 1, 1): 0, (1, 1, 2): 1, (1, 1, 3): 0,
             (1, 2, 0): 0, (1, 2, 1): -1, (1, 2, 2): 0, (1, 2, 3): 0,
             (1, 3, 0): -1, (1, 3, 1): 0, (1, 3, 2): 0, (1, 3, 3): 0,
             #Gamma2
             (2, 0, 0): 0, (2, 0, 1): 0, (2, 0, 2): 0, (2, 0, 3): -1j,
             (2, 1, 0): 0, (2, 1, 1): 0, (2, 1, 2): 1j, (2, 1, 3): 0,
             (2, 2, 0): 0, (2, 2, 1): 1j, (2, 2, 2): 0, (2, 2, 3): 0,
             (2, 3, 0): -1j, (2, 3, 1): 0, (2, 3, 2): 0, (2, 3, 3): 0,
             #Gamma3
             (3, 0, 0): 0, (3, 0, 1): 0, (3, 0, 2): 1, (3, 0, 3): 0,
             (3, 1, 0): 0, (3, 1, 1): 0, (3, 1, 2): 0, (3, 1, 3): -1,
             (3, 2, 0): -1, (3, 2, 1): 0, (3, 2, 2): 0, (3, 2, 3): 0,
             (3, 3, 0): 0, (3, 3, 1): 1, (3, 3, 2): 0, (3, 3, 3): 0
             }





    def __init__(self, lorentz, spin1, spin2, prefactor=1):
        aloha_lib.LorentzObject.__init__(self,[lorentz], [spin1, spin2], \
                                                            prefactor=prefactor)
    
    def create_representation(self):
                
        self.representation = aloha_lib.LorentzObjectRepresentation(self.gamma,
                                self.lorentz_ind,self.spin_ind)
        
#===============================================================================
# Sigma
#===============================================================================
class Sigma(aloha_lib.LorentzObject):
    """ Sigma Matrices """
    
    
    
    #zero = [[0,0,0,0]]*4
    #i = complex(0,1)
    #sigma01 = [[ 0, -i, 0, 0], [-i, 0, 0, 0], [0, 0, 0, i], [0, 0, i, 0]]
    #sigma02 = [[ 0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]
    #sigma03 = [[-i, 0, 0, 0], [0, i, 0, 0], [0, 0, i, 0], [0, 0, 0, -i]]
    #sigma12 = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    #sigma13 = [[0, i, 0, 0], [-i, 0, 0, 0], [0, 0, 0, i], [0, 0, -i, 0]]
    #sigma23 = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    #def inv(matrice):     
    #    out=[]
    #    for i in range(4):
    #        out2=[]
    #        out.append(out2)
    #        for j in range(4):
    #            out2.append(-1*matrice[i][j])
    #    return out
    #                    
    #sigma =[[zero, sigma01, sigma02, sigma03], \
    #        [inv(sigma01), zero, sigma12, sigma13],\
    #        [inv(sigma02), inv(sigma12), zero, sigma23],\
    #        [inv(sigma03), inv(sigma13), inv(sigma23), zero]]

    sigma={(0, 2, 0, 1): -0.5, (3, 1, 2, 0): 0, (3, 2, 3, 1): 0, (1, 3, 1, 3): 0, 
           (2, 3, 3, 2): 0.5, (2, 1, 3, 1): 0, (0, 2, 2, 1): 0, (3, 1, 0, 0): 0, 
           (2, 3, 3, 1): 0, (3, 3, 1, 2): 0, (3, 1, 0, 3): 0, (1, 1, 0, 3): 0, 
           (0, 1, 2, 2): 0, (3, 2, 3, 2): -0.5, (2, 1, 0, 1): 0, (3, 3, 3, 3): 0, 
           (1, 1, 2, 2): 0, (2, 2, 3, 2): 0, (2, 1, 2, 1): 0, (0, 1, 0, 3): 0, 
           (2, 1, 2, 2): -0.5, (1, 2, 2, 1): 0, (2, 2, 1, 3): 0, (0, 3, 1, 3): 0, 
           (3, 0, 3, 2): 0, (1, 2, 0, 1): 0, (3, 0, 3, 1): 0, (0, 0, 2, 2): 0, 
           (1, 2, 0, 2): 0, (2, 0, 0, 3): 0, (0, 0, 2, 1): 0, (0, 3, 3, 2): 0, 
           (3, 0, 1, 1): -0.5j, (3, 2, 0, 1): -0.5, (1, 0, 1, 0): 0.5j, (0, 0, 0, 1): 0,
            (0, 2, 1, 1): 0, (3, 1, 3, 2): 0.5j, (3, 2, 2, 1): 0, (1, 3, 2, 3): 0.5j, 
            (1, 0, 3, 0): 0, (3, 2, 2, 2): 0, (0, 2, 3, 1): 0, (1, 0, 3, 3): 0, 
            (2, 3, 2, 1): 0, (0, 2, 3, 2): -0.5, (3, 1, 1, 3): 0, (1, 1, 1, 3): 0, 
            (1, 3, 0, 2): 0, (2, 3, 0, 1): 0.5, (1, 1, 1, 0): 0, (2, 3, 0, 2): 0, 
            (3, 3, 0, 3): 0, (1, 1, 3, 0): 0, (0, 1, 3, 3): 0, (2, 2, 0, 1): 0, 
            (2, 1, 1, 0): 0, (3, 3, 2, 2): 0, (2, 3, 1, 0): 0.5, (2, 2, 2, 3): 0, 
            (0, 3, 0, 3): 0, (0, 1, 1, 2): 0, (0, 3, 0, 0): -0.5j, (2, 3, 1, 1): 0, 
            (1, 2, 3, 0): 0, (2, 0, 1, 3): 0, (0, 0, 3, 1): 0, (0, 3, 2, 0): 0, 
            (2, 3, 1, 2): 0, (2, 0, 1, 0): -0.5, (1, 2, 1, 0): 0, (3, 0, 0, 2): 0, 
            (1, 0, 0, 2): 0, (0, 0, 1, 1): 0, (1, 2, 1, 3): 0, (2, 3, 1, 3): 0, 
            (2, 0, 3, 0): 0, (0, 0, 1, 2): 0, (1, 3, 3, 3): 0, (3, 2, 1, 0): -0.5, 
            (1, 3, 3, 0): 0, (1, 0, 2, 3): -0.5j, (0, 2, 0, 0): 0, (3, 1, 2, 3): -0.5j, 
            (3, 2, 3, 0): 0, (1, 3, 1, 0): -0.5j, (3, 2, 3, 3): 0, (0, 2, 2, 0): 0, 
            (2, 3, 3, 0): 0, (3, 3, 1, 3): 0, (0, 2, 2, 3): 0.5, (3, 1, 0, 2): 0, 
            (1, 1, 0, 2): 0, (3, 3, 1, 0): 0, (0, 1, 2, 3): 0.5j, (1, 1, 0, 1): 0,
            (2, 1, 0, 2): 0, (0, 1, 2, 0): 0, (3, 3, 3, 0): 0, (1, 1, 2, 1): 0,
            (2, 2, 3, 3): 0, (0, 1, 0, 0): 0, (2, 2, 3, 0): 0, (2, 1, 2, 3): 0,
            (1, 2, 2, 2): 0.5, (2, 2, 1, 0): 0, (0, 3, 1, 2): 0, (0, 3, 1, 1): 0.5j, 
            (3, 0, 3, 0): 0, (1, 2, 0, 3): 0, (2, 0, 0, 2): 0, (0, 0, 2, 0): 0, 
            (0, 3, 3, 1): 0, (3, 0, 1, 0): 0, (2, 0, 0, 1): 0.5, (3, 2, 0, 2): 0, 
            (3, 0, 1, 3): 0, (1, 0, 1, 3): 0, (0, 0, 0, 0): 0, (0, 2, 1, 2): 0, 
            (3, 1, 3, 3): 0, (0, 0, 0, 3): 0, (1, 3, 2, 2): 0, (3, 1, 3, 0): 0, 
            (3, 2, 2, 3): -0.5, (1, 3, 2, 1): 0, (1, 0, 3, 2): -0.5j, (2, 3, 2, 2): 0, 
            (0, 2, 3, 3): 0, (3, 1, 1, 0): 0.5j, (1, 3, 0, 1): 0.5j, (1, 1, 1, 1): 0, 
            (2, 1, 3, 2): 0, (2, 3, 0, 3): 0, (3, 3, 0, 2): 0, (1, 1, 3, 1): 0, 
            (3, 3, 0, 1): 0, (2, 1, 3, 3): 0.5, (0, 1, 3, 2): 0.5j, (1, 1, 3, 2): 0, 
            (2, 1, 1, 3): 0, (3, 0, 2, 1): 0, (0, 1, 3, 1): 0, (3, 3, 2, 1): 0, 
            (2, 2, 2, 2): 0, (0, 1, 1, 1): 0, (2, 2, 2, 1): 0, (0, 3, 0, 1): 0, 
            (3, 0, 2, 2): -0.5j, (1, 2, 3, 3): -0.5, (0, 0, 3, 2): 0, (0, 3, 2, 1): 0, 
            (2, 0, 1, 1): 0, (2, 2, 0, 0): 0, (0, 3, 2, 2): 0.5j, (3, 0, 0, 3): 0, 
            (1, 0, 0, 3): 0, (1, 2, 1, 2): 0, (2, 0, 3, 1): 0, (1, 0, 0, 0): 0, 
            (0, 0, 1, 3): 0, (2, 0, 3, 2): 0.5, (3, 2, 1, 3): 0, (1, 3, 3, 1): 0, 
            (1, 0, 2, 0): 0, (2, 2, 0, 2): 0, (0, 2, 0, 3): 0, (3, 1, 2, 2): 0, 
            (1, 3, 1, 1): 0, (3, 1, 2, 1): 0, (2, 2, 0, 3): 0, (3, 0, 0, 1): 0, 
            (1, 3, 1, 2): 0, (2, 3, 3, 3): 0, (0, 2, 2, 2): 0, (3, 1, 0, 1): -0.5j, 
            (3, 3, 1, 1): 0, (1, 1, 0, 0): 0, (2, 1, 0, 3): 0, (0, 1, 2, 1): 0, 
            (3, 3, 3, 1): 0, (2, 1, 0, 0): -0.5, (1, 1, 2, 0): 0, (3, 3, 3, 2): 0, 
            (0, 1, 0, 1): -0.5j, (1, 1, 2, 3): 0, (2, 2, 3, 1): 0, (2, 1, 2, 0): 0,
             (0, 1, 0, 2): 0, (1, 2, 2, 3): 0, (2, 0, 2, 1): 0, (2, 2, 1, 1): 0, 
             (1, 2, 2, 0): 0, (2, 2, 1, 2): 0, (0, 3, 1, 0): 0, (3, 0, 3, 3): 0.5j, 
             (2, 1, 3, 0): 0, (1, 2, 0, 0): 0.5, (0, 0, 2, 3): 0, (0, 3, 3, 0): 0, 
             (2, 0, 0, 0): 0, (3, 2, 0, 3): 0, (0, 3, 3, 3): -0.5j, (3, 0, 1, 2): 0, 
             (1, 0, 1, 2): 0, (3, 2, 0, 0): 0, (0, 2, 1, 3): 0, (1, 0, 1, 1): 0, 
             (0, 0, 0, 2): 0, (0, 2, 1, 0): 0.5, (3, 1, 3, 1): 0, (3, 2, 2, 0): 0, 
             (1, 3, 2, 0): 0, (1, 0, 3, 1): 0, (2, 3, 2, 3): 0.5, (0, 2, 3, 0): 0, 
             (3, 1, 1, 1): 0, (2, 3, 2, 0): 0, (1, 3, 0, 0): 0, (3, 1, 1, 2): 0, 
             (1, 1, 1, 2): 0, (1, 3, 0, 3): 0, (2, 3, 0, 0): 0, (2, 0, 2, 0): 0, 
             (3, 3, 0, 0): 0, (1, 1, 3, 3): 0, (2, 1, 1, 2): 0, (0, 1, 3, 0): 0, 
             (3, 3, 2, 0): 0, (2, 1, 1, 1): 0.5, (2, 0, 2, 2): 0, (3, 3, 2, 3): 0, 
             (0, 1, 1, 0): -0.5j, (2, 2, 2, 0): 0, (0, 3, 0, 2): 0, (3, 0, 2, 3): 0, 
             (0, 1, 1, 3): 0, (2, 0, 2, 3): -0.5, (1, 2, 3, 2): 0, (3, 0, 2, 0): 0, 
             (0, 0, 3, 3): 0, (1, 2, 3, 1): 0, (2, 0, 1, 2): 0, (0, 0, 3, 0): 0, 
             (0, 3, 2, 3): 0, (3, 0, 0, 0): 0.5j, (1, 2, 1, 1): -0.5, (1, 0, 0, 1): 0.5j, 
             (0, 0, 1, 0): 0, (2, 0, 3, 3): 0, (3, 2, 1, 2): 0, (1, 3, 3, 2): -0.5j, 
             (1, 0, 2, 1): 0, (3, 2, 1, 1): 0, (0, 2, 0, 2): 0, (1, 0, 2, 2): 0}

    def __init__(self, lorentz1, lorentz2, spin1, spin2, prefactor=1):
        if lorentz1 < lorentz2:
            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], \
                                                  [spin1, spin2], prefactor=prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[lorentz2, lorentz1], 
                                             [spin1, spin2], prefactor=-prefactor)

    def create_representation(self):
                
        self.representation = aloha_lib.LorentzObjectRepresentation(self.sigma,
                                self.lorentz_ind,self.spin_ind)

#===============================================================================
# Gamma5
#===============================================================================        
class Gamma5(aloha_lib.LorentzObject):
    
    #gamma5 = [[-1, 0, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    gamma5 = {(0,0): -1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): -1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 1}
    
    def __init__(self, spin1, spin2, prefactor=1):
        if spin1 > spin2:
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], prefactor)

    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.gamma5,
                                             self.lorentz_ind,self.spin_ind) 
        
#===============================================================================
# Conjugate Matrices
#===============================================================================
class C(aloha_lib.LorentzObject):
    
    #[0, -1, 0, 0] [1,0,0,0] [0,0,0,1],[0,0,-1,0]
    
    Cmetrix = {(0,0): 0, (0,1): -1, (0,2): 0, (0,3): 0,\
              (1,0): 1, (1,1): 0, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 0, (2,3): 1,\
              (3,0): 0, (3,1): 0, (3,2): -1, (3,3): 0} 
    
    def __init__(self, spin1, spin2, prefactor=1):
        #antisymmetric
        if spin1 < spin2:
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], -1*prefactor)

    def create_representation(self):
        self.representation = aloha_lib.LorentzObjectRepresentation(self.Cmetrix,
                                             self.lorentz_ind,self.spin_ind) 
    


#===============================================================================
# EPSILON  
#===============================================================================
#Helpfull function
def give_sign_perm(perm0, perm1):
    """Check if 2 permutations are of equal parity.

    Assume that both permutation lists are of equal length
    and have the same elements. No need to check for these
    conditions.
    """
    assert len(perm0) == len(perm1) 
        
    perm1 = list(perm1) ## copy this into a list so we don't mutate the original
    perm1_map = dict((v, i) for i,v in enumerate(perm1))

    transCount = 0
    for loc, p0 in enumerate(perm0):
        p1 = perm1[loc]
        if p0 != p1:
            sloc = perm1_map[p0]                       # Find position in perm1
            perm1[loc], perm1[sloc] = p0, p1           # Swap in perm1
            perm1_map[p0], perm1_map[p1] = loc, sloc   # Swap the map
            transCount += 1
            
    # Even number of transposition means equal parity
    return -2 * (transCount % 2) + 1
    
# Practical definition of Epsilon
class Epsilon(aloha_lib.LorentzObject):
    """ The fully anti-symmetric object in Lorentz-Space """
 
    def give_parity(self, perm):
        """return the parity of the permutation"""
        assert set(perm) == set([0,1,2,3]) 
        
        i1 , i2, i3, i4 = perm
        #formula found on wikipedia
        return ((i2-i1) * (i3-i1) *(i4-i1) * (i3-i2) * (i4-i2) *(i4-i3))/12 
   
    # DEFINE THE REPRESENTATION OF EPSILON
           
    def __init__(self, lorentz1, lorentz2, lorentz3, lorentz4, prefactor=1):
       
       lorentz_list = [lorentz1 , lorentz2, lorentz3, lorentz4]
       order_lor = list(lorentz_list)
       order_lor.sort()
       
       sign = give_sign_perm(order_lor, lorentz_list)
       
       aloha_lib.LorentzObject.__init__(self, order_lor, \
                                                 [], prefactor=sign * prefactor)


    def create_representation(self):

        if not hasattr(self, 'epsilon'):
            # init all element to zero
            epsilon = dict( ((l1, l2, l3, l4), 0)
                                  for l1 in range(4) \
                                  for l2 in range(4) \
                                  for l3 in range(4) \
                                  for l4 in range(4))        
            # update non trivial one
            epsilon.update(dict(
             ((l1, l2, l3, l4), self.give_parity((l1,l2,l3,l4)))
                                 for l1 in range(4) \
                                 for l2 in range(4) if l2 != l1\
                                 for l3 in range(4) if l3 not in [l1,l2]\
                                 for l4 in range(4) if l4 not in [l1,l2,l3]))

            Epsilon.epsilon = epsilon
        

        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.epsilon,
                                self.lorentz_ind,self.spin_ind)
   
    
            
#===============================================================================
# Metric
#===============================================================================
class Metric(aloha_lib.LorentzObject):
    
    metric = {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): -1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): -1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): -1}
    
    
    #[[1, 0, 0,0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    
    def __init__(self, lorentz1, lorentz2, prefactor=1):
        if lorentz1 < lorentz2:
            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[lorentz2, lorentz1], [], prefactor)
    
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.metric,
                                             self.lorentz_ind,self.spin_ind)     

    def expand(self):
        """Expand the content information. We overload the basic rules in order
        to avoid the computation of Metric(1,2) * Metric(1,2) = 4"""

        if self.power == 2: 
            return aloha_lib.ConstantObject(4)
        else:
            try:
                return self.prefactor * self.representation
            except:
                self.create_representation()
                return self.prefactor * self.representation  
            
    def simplify(self):
        """Return the Denominator in a abstract way"""
        
        if self.power == 2:
            return aloha_lib.ConstantObject(4)
        else:
            return self
         
           
    
#===============================================================================
# Identity
#===============================================================================
class Identity(aloha_lib.LorentzObject):
    
    #identity = [[1, 0, 0,0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    identity = {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): 1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 1}
    
    def __init__(self, spin1, spin2, prefactor=1):
        if spin1 < spin2:
            aloha_lib.LorentzObject.__init__(self,[],[spin1, spin2], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[],[spin2, spin1], prefactor)
            
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.identity,
                                             self.lorentz_ind,self.spin_ind)
##===============================================================================
## IdentityL  (Commented since not use)
##===============================================================================
#class IdentityL(aloha_lib.LorentzObject):
#    
#    identity = [[1, 0, 0,0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
#    
#    def __init__(self, lorentz1, lorentz2, prefactor=1):
#        if lorentz1 < lorentz2:
#            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], [])
#        else:
#            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], [])
#            
#    def create_representation(self):
#        
#        self.representation = aloha_lib.LorentzObjectRepresentation(self.identity,
#                                             self.lorentz_ind,self.spin_ind)
#    
#===============================================================================
# ProjM 
#===============================================================================    
class ProjM(aloha_lib.LorentzObject):
    """ A object for (1-gamma5)/2 """
    
    #projm = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    projm= {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): 1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 0, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 0}
    
    def __init__(self,spin1, spin2, prefactor=1):
        """Initialize the object"""
        if spin1 < spin2:
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], prefactor) 
        
          
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.projm,
                                             self.lorentz_ind,self.spin_ind)    


#===============================================================================
# ProjP 
#===============================================================================    
class ProjP(aloha_lib.LorentzObject):
    """A object for (1+gamma5)/2 """
    
    #projp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    projp = {(0,0): 0, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): 0, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 1}
    
    def __init__(self,spin1, spin2, prefactor=1):
        """Initialize the object"""
        if spin1 < spin2:
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], prefactor) 
        
          
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.projp,
                                            self.lorentz_ind, self.spin_ind)    

#===============================================================================
# Denominator Propagator 
#===============================================================================    
class DenominatorPropagator(aloha_lib.LorentzObject):
    """The Denominator of the Propagator"""
    
    def __init__(self, particle, prefactor=1):
        """Initialize the object"""
        
        self.particle = particle
        tag=['P%s' % particle]
        aloha_lib.LorentzObject.__init__(self, [], [], prefactor, tag)
    
    def simplify(self):
        """Return the Denominator in a abstract way"""

        mass = Mass(self.particle)
        width = Width(self.particle)       
        denominator = P('i1', self.particle) * P('i1', self.particle) - \
                      mass * mass + complex(0,1) * mass* width
         
        return denominator
     
    def create_representation(self):
        """Create the representation for the Vector propagator"""
        
        object = self.simplify()
        self.representation = object.expand()


                
#===============================================================================
# Numerator Propagator 
#===============================================================================            


SpinorPropagator = lambda spin1, spin2, particle: complex(0,1) * (Gamma('mu', spin1, spin2) * \
                    P('mu', particle) + Mass(particle) * Identity(spin1, spin2))
                    
VectorPropagator = lambda l1, l2, part: complex(0,1) * (-1 * Metric(l1, l2) + OverMass2(part) * \
                                    Metric(l1,'I3')* P('I3', part) * P(l2, part))

#Spin3halfPropagator =  lambda mu, nu, s1, s2, part: -1*( Gamma(-1,s1,s2)*P(-1,part) + Identity(s1,s2)*Mass(part)) * (Metric(mu,nu)-Metric(mu,'I3')*P('I3',part)*P(nu,part)*OverMass2(part)) \
#         - 1/3 * (Gamma(mu,s1,-2) + Identity(s1, -2) *  P(mu, part) * Mass(part) * OverMass2(part))* \
#                             (Gamma('alpha',-2,-3) * P('alpha', part) - Identity(-2,-3) * Mass(part)) \
#                             * (Gamma(nu, -3, s2) + Identity(-3, s2) * P(nu, part) * Mass(part) * OverMass2(part) ) \
#            -1*( Gamma(-1,s1,s2)*P(-1,part) + Identity(s1,s2)*Mass(part)) * (Metric(mu,nu)-Metric(mu,'I3')*P('I3',part)*P(nu,part)*OverMass2(part)) \

Spin3halfPropagator =  lambda mu, nu, s1, s2, part:  - 1/3 * (Gamma(mu,s1,-2) + Identity(s1, -2) *  P(mu, part) * Mass(part) * OverMass2(part))* \
                             (PSlash(-2,-3, part) - Identity(-2,-3) * Mass(part)) * \
                             ( Gamma(nu, -3, s2)+ Mass(part) * OverMass2(part) * Identity(-3, s2) * P(nu, part) )

Spin3halfPropagator =  lambda mu, nu, s1, s2, part:  - 1/3 * (Gamma(mu,s1,-2) + Identity(s1, -2) *  P(mu, part) * Mass(part) * OverMass2(part))* \
                             (PSlash(-2,-3, part) - Identity(-2,-3) * Mass(part)) * \
                             ( Gamma(nu, -3, s2)+ Mass(part) * OverMass2(part) * Identity(-3, s2) * P(nu, part) )
                             

Spin2masslessPropagator = lambda mu, nu, alpha, beta: complex(0,1/2)*( Metric(mu, alpha)* Metric(nu, beta) +\
                     Metric(mu, beta) * Metric(nu, alpha) - Metric(mu, nu) * Metric(alpha, beta))



Spin2Propagator =  lambda mu, nu, alpha, beta, part: Spin2masslessPropagator(mu, nu, alpha, beta) + \
                -complex(0, 1/2) * OverMass2(part) * (Metric(mu,alpha)* P(nu, part) * P(beta, part) + \
                                Metric(nu, beta) * P(mu, part) * P(alpha, part) + \
                                Metric(mu, beta) * P(nu, part) * P(alpha, part) + \
                                Metric(nu, alpha) * P(mu, part) * P(beta , part) )+ \
                complex(0, 1/6) * (Metric(mu,nu) + 2 * OverMass2(part) * P(mu, part) * P(nu, part)) * \
                      (Metric(alpha,beta) + 2 * OverMass2(part) * P(alpha, part) * P(beta, part))
    















