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
# P (Impulsion)
#===============================================================================
class P(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    contract_first = 1
    
    def __init__(self, lorentz1, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [lorentz1], [], ['P%s'%particle], \
                                        prefactor=prefactor)
    
        
    def create_representation(self):
        self.sub0 = aloha_lib.ScalarVariable('P%s_0' % self.particle, self.tag)
        self.sub1 = aloha_lib.ScalarVariable('P%s_1' % self.particle, self.tag)
        self.sub2 = aloha_lib.ScalarVariable('P%s_2' % self.particle, self.tag)
        self.sub3 = aloha_lib.ScalarVariable('P%s_3' % self.particle, self.tag)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},                              
                                    self.lorentz_ind,[],self.tag)

#===============================================================================
# Mass
#===============================================================================
class Mass(aloha_lib.LorentzObject):
    """ Helas Object for a Mass"""
    
    def __init__(self, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [], ['M%s' % particle], \
                                        prefactor=prefactor)
    
        
    def create_representation(self):
        mass = aloha_lib.ScalarVariable('M%s' % self.particle, self.tag)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind, self.tag)

#===============================================================================
# OverMass2
#===============================================================================
class OverMass2(aloha_lib.LorentzObject):
    """ Helas Object for 1/M**2 """
    
    def __init__(self, particle, prefactor=1):
        
        self.particle = particle
        
        tag= ['mass%s' %particle, 'OM%s' % particle]
        aloha_lib.LorentzObject.__init__(self, [], [], tag, \
                                        prefactor=prefactor)
    
        
    def create_representation(self):
        mass = aloha_lib.ScalarVariable('OM%s' % self.particle, self.tag)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind, self.tag)

#===============================================================================
# Width
#===============================================================================
class Width(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    def __init__(self, particle, prefactor=1):

        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [], ['W%s' % particle], \
                                         prefactor=prefactor)
        
    def create_representation(self):
        width = aloha_lib.ScalarVariable('W%s' % self.particle, self.tag)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                            width, self.lorentz_ind, self.spin_ind, self.tag)
        
#===============================================================================
# Scalar
#===============================================================================
class Scalar(aloha_lib.LorentzObject):
    """ Helas Object for a Spinor"""
    
    def __init__(self, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [], ['S%s' % particle], \
                                         prefactor=prefactor)
    
        
    def create_representation(self):
        rep = aloha_lib.ScalarVariable('S%s_1' % self.particle, self.tag)
        self.representation= aloha_lib.LorentzObjectRepresentation(        
                                    rep,
                                    [],[],self.tag)        
        
        
#===============================================================================
# Spinor
#===============================================================================
class Spinor(aloha_lib.LorentzObject):
    """ Helas Object for a Spinor"""
    
    contract_first = 1
    
    def __init__(self, spin1, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [spin1], ['F%s' % particle], \
                                         prefactor=prefactor)
    
        
    def create_representation(self):
        self.sub0 = aloha_lib.ScalarVariable('F%s_1' % self.particle, self.tag)
        self.sub1 = aloha_lib.ScalarVariable('F%s_2' % self.particle, self.tag)
        self.sub2 = aloha_lib.ScalarVariable('F%s_3' % self.particle, self.tag)
        self.sub3 = aloha_lib.ScalarVariable('F%s_4' % self.particle, self.tag)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},         
                                    [],self.spin_ind,self.tag)

#===============================================================================
# Vector
#===============================================================================
class Vector(aloha_lib.LorentzObject):
    """ Helas Object for a Vector"""
    
    contract_first = 1
    
    def __init__(self, lorentz, particle, prefactor=1):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [lorentz], [], ['V%s' % particle], \
                                         prefactor=prefactor)
    
        
    def create_representation(self):
        self.sub0 = aloha_lib.ScalarVariable('V%s_1' % self.particle, self.tag)
        self.sub1 = aloha_lib.ScalarVariable('V%s_2' % self.particle, self.tag)
        self.sub2 = aloha_lib.ScalarVariable('V%s_3' % self.particle, self.tag)
        self.sub3 = aloha_lib.ScalarVariable('V%s_4' % self.particle, self.tag)

        self.representation= aloha_lib.LorentzObjectRepresentation( 
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},  
                                    self.lorentz_ind, [], self.tag)
        
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
                                ['T%s' % particle], prefactor=prefactor)
    
    def create_representation(self):

        self.sub00 = aloha_lib.ScalarVariable('T%s_1' % self.particle, self.tag)
        self.sub01 = aloha_lib.ScalarVariable('T%s_2' % self.particle, self.tag)
        self.sub02 = aloha_lib.ScalarVariable('T%s_3' % self.particle, self.tag)
        self.sub03 = aloha_lib.ScalarVariable('T%s_4' % self.particle, self.tag)

        self.sub10 = aloha_lib.ScalarVariable('T%s_5' % self.particle, self.tag)
        self.sub11 = aloha_lib.ScalarVariable('T%s_6' % self.particle, self.tag)
        self.sub12 = aloha_lib.ScalarVariable('T%s_7' % self.particle, self.tag)
        self.sub13 = aloha_lib.ScalarVariable('T%s_8' % self.particle, self.tag)
	
        self.sub20 = aloha_lib.ScalarVariable('T%s_9' % self.particle, self.tag)
        self.sub21 = aloha_lib.ScalarVariable('T%s_10' % self.particle, self.tag)
        self.sub22 = aloha_lib.ScalarVariable('T%s_11' % self.particle, self.tag)
        self.sub23 = aloha_lib.ScalarVariable('T%s_12' % self.particle, self.tag)
	
        self.sub30 = aloha_lib.ScalarVariable('T%s_13' % self.particle, self.tag)
        self.sub31 = aloha_lib.ScalarVariable('T%s_14' % self.particle, self.tag)
        self.sub32 = aloha_lib.ScalarVariable('T%s_15' % self.particle, self.tag)
        self.sub33 = aloha_lib.ScalarVariable('T%s_16' % self.particle, self.tag)
        
        rep = {(0,0): self.sub00, (0,1): self.sub01, (0,2): self.sub02, (0,3): self.sub03,
               (1,0): self.sub10, (1,1): self.sub11, (1,2): self.sub12, (1,3): self.sub13,
               (2,0): self.sub20, (2,1): self.sub21, (2,2): self.sub22, (2,3): self.sub23,
               (3,0): self.sub30, (3,1): self.sub31, (3,2): self.sub32, (3,3): self.sub33}
        
                
        self.representation= aloha_lib.LorentzObjectRepresentation( rep, \
                                    self.lorentz_ind, [], self.tag)

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
        aloha_lib.LorentzObject.__init__(self,[lorentz], [spin1, spin2], [], \
                                                                      prefactor)
    
    def create_representation(self):
                
        self.representation = aloha_lib.LorentzObjectRepresentation(self.gamma,
                                self.lorentz_ind,self.spin_ind,[])
        
#===============================================================================
# Sigma
#===============================================================================
class Sigma(aloha_lib.LorentzObject):
    """ Gamma Matrices """
    
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

    sigma={(0, 2, 0, 1): -1, (3, 1, 2, 0): 0, (3, 2, 3, 1): 0, (1, 3, 1, 3): 0, 
           (2, 3, 3, 2): 1, (2, 1, 3, 1): 0, (0, 2, 2, 1): 0, (3, 1, 0, 0): 0, 
           (2, 3, 3, 1): 0, (3, 3, 1, 2): 0, (3, 1, 0, 3): 0, (1, 1, 0, 3): 0, 
           (0, 1, 2, 2): 0, (3, 2, 3, 2): -1, (2, 1, 0, 1): 0, (3, 3, 3, 3): 0, 
           (1, 1, 2, 2): 0, (2, 2, 3, 2): 0, (2, 1, 2, 1): 0, (0, 1, 0, 3): 0, 
           (2, 1, 2, 2): -1, (1, 2, 2, 1): 0, (2, 2, 1, 3): 0, (0, 3, 1, 3): 0, 
           (3, 0, 3, 2): 0, (1, 2, 0, 1): 0, (3, 0, 3, 1): 0, (0, 0, 2, 2): 0, 
           (1, 2, 0, 2): 0, (2, 0, 0, 3): 0, (0, 0, 2, 1): 0, (0, 3, 3, 2): 0, 
           (3, 0, 1, 1): -1j, (3, 2, 0, 1): -1, (1, 0, 1, 0): 1j, (0, 0, 0, 1): 0,
            (0, 2, 1, 1): 0, (3, 1, 3, 2): 1j, (3, 2, 2, 1): 0, (1, 3, 2, 3): 1j, 
            (1, 0, 3, 0): 0, (3, 2, 2, 2): 0, (0, 2, 3, 1): 0, (1, 0, 3, 3): 0, 
            (2, 3, 2, 1): 0, (0, 2, 3, 2): -1, (3, 1, 1, 3): 0, (1, 1, 1, 3): 0, 
            (1, 3, 0, 2): 0, (2, 3, 0, 1): 1, (1, 1, 1, 0): 0, (2, 3, 0, 2): 0, 
            (3, 3, 0, 3): 0, (1, 1, 3, 0): 0, (0, 1, 3, 3): 0, (2, 2, 0, 1): 0, 
            (2, 1, 1, 0): 0, (3, 3, 2, 2): 0, (2, 3, 1, 0): 1, (2, 2, 2, 3): 0, 
            (0, 3, 0, 3): 0, (0, 1, 1, 2): 0, (0, 3, 0, 0): -1j, (2, 3, 1, 1): 0, 
            (1, 2, 3, 0): 0, (2, 0, 1, 3): 0, (0, 0, 3, 1): 0, (0, 3, 2, 0): 0, 
            (2, 3, 1, 2): 0, (2, 0, 1, 0): -1, (1, 2, 1, 0): 0, (3, 0, 0, 2): 0, 
            (1, 0, 0, 2): 0, (0, 0, 1, 1): 0, (1, 2, 1, 3): 0, (2, 3, 1, 3): 0, 
            (2, 0, 3, 0): 0, (0, 0, 1, 2): 0, (1, 3, 3, 3): 0, (3, 2, 1, 0): -1, 
            (1, 3, 3, 0): 0, (1, 0, 2, 3): -1j, (0, 2, 0, 0): 0, (3, 1, 2, 3): -1j, 
            (3, 2, 3, 0): 0, (1, 3, 1, 0): -1j, (3, 2, 3, 3): 0, (0, 2, 2, 0): 0, 
            (2, 3, 3, 0): 0, (3, 3, 1, 3): 0, (0, 2, 2, 3): 1, (3, 1, 0, 2): 0, 
            (1, 1, 0, 2): 0, (3, 3, 1, 0): 0, (0, 1, 2, 3): 1j, (1, 1, 0, 1): 0,
            (2, 1, 0, 2): 0, (0, 1, 2, 0): 0, (3, 3, 3, 0): 0, (1, 1, 2, 1): 0,
            (2, 2, 3, 3): 0, (0, 1, 0, 0): 0, (2, 2, 3, 0): 0, (2, 1, 2, 3): 0,
            (1, 2, 2, 2): 1, (2, 2, 1, 0): 0, (0, 3, 1, 2): 0, (0, 3, 1, 1): 1j, 
            (3, 0, 3, 0): 0, (1, 2, 0, 3): 0, (2, 0, 0, 2): 0, (0, 0, 2, 0): 0, 
            (0, 3, 3, 1): 0, (3, 0, 1, 0): 0, (2, 0, 0, 1): 1, (3, 2, 0, 2): 0, 
            (3, 0, 1, 3): 0, (1, 0, 1, 3): 0, (0, 0, 0, 0): 0, (0, 2, 1, 2): 0, 
            (3, 1, 3, 3): 0, (0, 0, 0, 3): 0, (1, 3, 2, 2): 0, (3, 1, 3, 0): 0, 
            (3, 2, 2, 3): -1, (1, 3, 2, 1): 0, (1, 0, 3, 2): -1j, (2, 3, 2, 2): 0, 
            (0, 2, 3, 3): 0, (3, 1, 1, 0): 1j, (1, 3, 0, 1): 1j, (1, 1, 1, 1): 0, 
            (2, 1, 3, 2): 0, (2, 3, 0, 3): 0, (3, 3, 0, 2): 0, (1, 1, 3, 1): 0, 
            (3, 3, 0, 1): 0, (2, 1, 3, 3): 1, (0, 1, 3, 2): 1j, (1, 1, 3, 2): 0, 
            (2, 1, 1, 3): 0, (3, 0, 2, 1): 0, (0, 1, 3, 1): 0, (3, 3, 2, 1): 0, 
            (2, 2, 2, 2): 0, (0, 1, 1, 1): 0, (2, 2, 2, 1): 0, (0, 3, 0, 1): 0, 
            (3, 0, 2, 2): -1j, (1, 2, 3, 3): -1, (0, 0, 3, 2): 0, (0, 3, 2, 1): 0, 
            (2, 0, 1, 1): 0, (2, 2, 0, 0): 0, (0, 3, 2, 2): 1j, (3, 0, 0, 3): 0, 
            (1, 0, 0, 3): 0, (1, 2, 1, 2): 0, (2, 0, 3, 1): 0, (1, 0, 0, 0): 0, 
            (0, 0, 1, 3): 0, (2, 0, 3, 2): 1, (3, 2, 1, 3): 0, (1, 3, 3, 1): 0, 
            (1, 0, 2, 0): 0, (2, 2, 0, 2): 0, (0, 2, 0, 3): 0, (3, 1, 2, 2): 0, 
            (1, 3, 1, 1): 0, (3, 1, 2, 1): 0, (2, 2, 0, 3): 0, (3, 0, 0, 1): 0, 
            (1, 3, 1, 2): 0, (2, 3, 3, 3): 0, (0, 2, 2, 2): 0, (3, 1, 0, 1): -1j, 
            (3, 3, 1, 1): 0, (1, 1, 0, 0): 0, (2, 1, 0, 3): 0, (0, 1, 2, 1): 0, 
            (3, 3, 3, 1): 0, (2, 1, 0, 0): -1, (1, 1, 2, 0): 0, (3, 3, 3, 2): 0, 
            (0, 1, 0, 1): -1j, (1, 1, 2, 3): 0, (2, 2, 3, 1): 0, (2, 1, 2, 0): 0,
             (0, 1, 0, 2): 0, (1, 2, 2, 3): 0, (2, 0, 2, 1): 0, (2, 2, 1, 1): 0, 
             (1, 2, 2, 0): 0, (2, 2, 1, 2): 0, (0, 3, 1, 0): 0, (3, 0, 3, 3): 1j, 
             (2, 1, 3, 0): 0, (1, 2, 0, 0): 1, (0, 0, 2, 3): 0, (0, 3, 3, 0): 0, 
             (2, 0, 0, 0): 0, (3, 2, 0, 3): 0, (0, 3, 3, 3): -1j, (3, 0, 1, 2): 0, 
             (1, 0, 1, 2): 0, (3, 2, 0, 0): 0, (0, 2, 1, 3): 0, (1, 0, 1, 1): 0, 
             (0, 0, 0, 2): 0, (0, 2, 1, 0): 1, (3, 1, 3, 1): 0, (3, 2, 2, 0): 0, 
             (1, 3, 2, 0): 0, (1, 0, 3, 1): 0, (2, 3, 2, 3): 1, (0, 2, 3, 0): 0, 
             (3, 1, 1, 1): 0, (2, 3, 2, 0): 0, (1, 3, 0, 0): 0, (3, 1, 1, 2): 0, 
             (1, 1, 1, 2): 0, (1, 3, 0, 3): 0, (2, 3, 0, 0): 0, (2, 0, 2, 0): 0, 
             (3, 3, 0, 0): 0, (1, 1, 3, 3): 0, (2, 1, 1, 2): 0, (0, 1, 3, 0): 0, 
             (3, 3, 2, 0): 0, (2, 1, 1, 1): 1, (2, 0, 2, 2): 0, (3, 3, 2, 3): 0, 
             (0, 1, 1, 0): -1j, (2, 2, 2, 0): 0, (0, 3, 0, 2): 0, (3, 0, 2, 3): 0, 
             (0, 1, 1, 3): 0, (2, 0, 2, 3): -1, (1, 2, 3, 2): 0, (3, 0, 2, 0): 0, 
             (0, 0, 3, 3): 0, (1, 2, 3, 1): 0, (2, 0, 1, 2): 0, (0, 0, 3, 0): 0, 
             (0, 3, 2, 3): 0, (3, 0, 0, 0): 1j, (1, 2, 1, 1): -1, (1, 0, 0, 1): 1j, 
             (0, 0, 1, 0): 0, (2, 0, 3, 3): 0, (3, 2, 1, 2): 0, (1, 3, 3, 2): -1j, 
             (1, 0, 2, 1): 0, (3, 2, 1, 1): 0, (0, 2, 0, 2): 0, (1, 0, 2, 2): 0}

    def __init__(self, lorentz1, lorentz2, spin1, spin2, prefactor=1):
        if lorentz1 < lorentz2:
            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], \
                                                  [spin1, spin2], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[lorentz2, lorentz1], [spin1, spin2], \
                                                                 [], -prefactor)

    def create_representation(self):
                
        self.representation = aloha_lib.LorentzObjectRepresentation(self.sigma,
                                self.lorentz_ind,self.spin_ind,[])

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
        if spin1 < spin2:
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], [], prefactor)

    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.gamma5,
                                             self.lorentz_ind,self.spin_ind,[]) 
        
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
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], [], -1*prefactor)

    def create_representation(self):
        self.representation = aloha_lib.LorentzObjectRepresentation(self.Cmetrix,
                                             self.lorentz_ind,self.spin_ind,[]) 
    
        
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
            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], [], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[lorentz2, lorentz1], [], [], prefactor)
    
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.metric,
                                             self.lorentz_ind,self.spin_ind,[])     

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
            aloha_lib.LorentzObject.__init__(self,[],[spin1, spin2], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[],[spin2, spin1], [], prefactor)
            
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.identity,
                                             self.lorentz_ind,self.spin_ind,[])
##===============================================================================
## IdentityL  (Commented since not use)
##===============================================================================
#class IdentityL(aloha_lib.LorentzObject):
#    
#    identity = [[1, 0, 0,0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
#    
#    def __init__(self, lorentz1, lorentz2, prefactor=1):
#        if lorentz1 < lorentz2:
#            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], [], [])
#        else:
#            aloha_lib.LorentzObject.__init__(self,[lorentz1, lorentz2], [], [])
#            
#    def create_representation(self):
#        
#        self.representation = aloha_lib.LorentzObjectRepresentation(self.identity,
#                                             self.lorentz_ind,self.spin_ind,[])
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
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], [], prefactor) 
        
          
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.projm,
                                             self.lorentz_ind,self.spin_ind,[])    


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
            aloha_lib.LorentzObject.__init__(self,[], [spin1, spin2], [], prefactor)
        else:
            aloha_lib.LorentzObject.__init__(self,[], [spin2, spin1], [], prefactor) 
        
          
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.projp,
                                            self.lorentz_ind, self.spin_ind, [])    

#===============================================================================
# Denominator Propagator 
#===============================================================================    
class DenominatorPropagator(aloha_lib.LorentzObject):
    """The Denominator of the Propagator"""
    
    def __init__(self, particle, prefactor=1):
        """Initialize the object"""
        
        self.particle = particle
        tag=['M%s' % particle,'W%s' % particle, 'P%s' % particle]
        aloha_lib.LorentzObject.__init__(self, [], [], tag, prefactor)
    
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

Spin2masslessPropagator = lambda l1, l2, l3, l4: 1/2 *( Metric(l1, l2)* Metric(l3, l4) +\
                     Metric(l1, l4) * Metric(l2, l3) - Metric(l1, l3) * Metric(l2, l4))



Spin2Propagator =  lambda l1, l2, l3, l4, part: Spin2masslessPropagator(l1, l2, l3, l4) + \
                -1/2 * OverMass2(part) * (Metric(l1,l2)* P(l3, part) * P(l4, part) + \
                                Metric(l3, l4) * P(l1, part) * P(l2, part) + \
                                Metric(l1, l4) * P(l2, part) * P(l3, part) + \
                                Metric(l3, l2) * P(l1, part) * P(l4 , part) )+ \
                1/6 * (Metric(l1,l3) + 2 * OverMass2(part) * P(l1, part) * P(l3, part)) * \
                      (Metric(l2,l4) + 2 * OverMass2(part) * P(l2, part) * P(l4, part))
    
    



















