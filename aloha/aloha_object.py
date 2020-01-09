################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
##   Diagram of Class
##
##    Variable <--- aloha_lib.Variable 
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
import aloha
import cmath

#===============================================================================
# P (Momenta)
#===============================================================================
class L_P(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    contract_first = 1
    
    def __init__(self, name, lorentz1, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name,[lorentz1], [],['P%s'%particle])
        aloha_lib.KERNEL.add_tag((name,))
    
    def create_representation(self):
        self.sub0 = aloha_lib.DVariable('P%s_0' % self.particle)
        self.sub1 = aloha_lib.DVariable('P%s_1' % self.particle)
        self.sub2 = aloha_lib.DVariable('P%s_2' % self.particle)
        self.sub3 = aloha_lib.DVariable('P%s_3' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},                              
                                    self.lorentz_ind, [])


class P(aloha_lib.FactoryLorentz):
    """ Helas Object for an Impulsion """
    
    object_class = L_P
    
    #def __init__(self, lorentz1, particle):
    @classmethod
    def get_unique_name(self, lorentz1, particle):
        return '_P^%s_%s' % (particle, lorentz1)



#===============================================================================
# PBar (Momenta = (E, -Px, -Py, -Pz)
#===============================================================================
class L_PBar(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    contract_first = 1
    
    def __init__(self, name, lorentz1, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name,[lorentz1], [],['PB%s'%particle])
        aloha_lib.KERNEL.add_tag((name,))
    
    def create_representation(self):
        self.sub0 = aloha_lib.DVariable('P%s_0' % self.particle)
        self.sub1 = aloha_lib.DVariable('P%s_1' % self.particle)
        self.sub2 = aloha_lib.DVariable('P%s_2' % self.particle)
        self.sub3 = aloha_lib.DVariable('P%s_3' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): self.sub0, (1,): -self.sub1, \
                                     (2,): -self.sub2, (3,): -self.sub3},                              
                                    self.lorentz_ind, [])


class PBar(aloha_lib.FactoryLorentz):
    """ Helas Object for an Impulsion """
    
    object_class = L_PBar
    
    #def __init__(self, lorentz1, particle):
    @classmethod
    def get_unique_name(self, lorentz1, particle):
        return '_PB^%s_%s' % (particle, lorentz1)

#===============================================================================
# P (Momenta)
#===============================================================================
class L_PVec(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    contract_first = 1
    
    def __init__(self, name, lorentz1, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name,[lorentz1], [],['PVec%s'%particle])
        aloha_lib.KERNEL.add_tag((name,))
    
    def create_representation(self):
        #self.sub0 = aloha_lib.DVariable('P%s_0' % self.particle)
        self.sub1 = aloha_lib.DVariable('P%s_1' % self.particle)
        self.sub2 = aloha_lib.DVariable('P%s_2' % self.particle)
        self.sub3 = aloha_lib.DVariable('P%s_3' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): 0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},                              
                                    self.lorentz_ind, [])


class PVec(aloha_lib.FactoryLorentz):
    """ Helas Object for an Impulsion """
    
    object_class = L_PVec
    
    #def __init__(self, lorentz1, particle):
    @classmethod
    def get_unique_name(self, lorentz1, particle):
        return '_Pvec^%s_%s' % (particle, lorentz1)


    
#===============================================================================
# Three Norm
#===============================================================================
class L_Tnorm(aloha_lib.LorentzObject):
    """ Helas Object for DSQRT(Px^2+Py^2+pz^2) """
      
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [], [], tags=['|p%s|' % particle])
    
    def create_representation(self):
        mass = aloha_lib.DVariable('Tnorm%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

class Tnorm(aloha_lib.FactoryLorentz):
    
    object_class = L_Tnorm
    
    @classmethod
    def get_unique_name(self, particle):
        return '_tnorm_%s' % particle

#===============================================================================
# |p| + pz
#===============================================================================
class L_TnormZ(aloha_lib.LorentzObject):
    """ Helas Object for DSQRT(Px^2+Py^2+pz^2) + PZ """
      
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [], [], tags=['|p%s|_z' % particle])
    
    def create_representation(self):
        mass = aloha_lib.DVariable('TnormZ%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

class TnormZ(aloha_lib.FactoryLorentz):
    
    object_class = L_TnormZ
    
    @classmethod
    def get_unique_name(self, particle):
        return '_tnormz_%s' % particle
    
#===============================================================================
# DSQRT(E+- |p|)
#===============================================================================
class L_FermionWP(aloha_lib.LorentzObject):
    """ Helas Object for DSQRT(Px^2+Py^2+pz^2) """
      
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [], [], tags=['FWP_%s' % particle])
    
    def create_representation(self):
        mass = aloha_lib.DVariable('FWP%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

class FermionWP(aloha_lib.FactoryLorentz):
    
    object_class = L_FermionWP
    
    @classmethod
    def get_unique_name(self, particle):
        return '_FermionWP_%s' % particle
    
#===============================================================================
# DSQRT(E+- |p|)
#===============================================================================
class L_FermionWM(aloha_lib.LorentzObject):
    """ Helas Object for DSQRT(Px^2+Py^2+pz^2) """
      
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [], [], tags=['FWM_%s' % particle])
    
    def create_representation(self):
        mass = aloha_lib.DVariable('FWM%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

class FermionWM(aloha_lib.FactoryLorentz):
    
    object_class = L_FermionWM
    
    @classmethod
    def get_unique_name(self, particle):
        return '_FermionWM_%s' % particle

    
#===============================================================================
# Pslash
#===============================================================================
class L_PSlash(aloha_lib.LorentzObject):
    """ Gamma Matrices """
    
    #gamma0 = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    #gamma1 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]
    #gamma2 = [[0, 0, 0, -complex(0,1)],[0, 0, complex(0,1), 0],
    #                    [0, complex(0,1), 0, 0], [-complex(0,1), 0, 0, 0]]
    #gamma3 = [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]]
    #    
    #gamma = [gamma0, gamma1, gamma2, gamma3]

    def __init__(self, name, spin1, spin2, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name,[], [spin1, spin2])
    
    def create_representation(self):
        """create representation"""
        p0 = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
    
    
        gamma = {
             (0, 0): 0, (0, 1): 0, (0, 2): p0-p3, (0, 3): -1*p1+1j*p2,
             (1, 0): 0, (1, 1): 0, (1, 2): -1*p1-1j*p2, (1, 3): p0+p3,
             (2, 0): p0+p3, (2, 1): p1-1j*p2, (2, 2): 0, (2, 3): 0,
             (3, 0): p1+1j*p2, (3, 1): p0-p3, (3, 2): 0, (3, 3): 0}

                
        self.representation = aloha_lib.LorentzObjectRepresentation(gamma,
                                self.lorentz_ind,self.spin_ind)

class PSlash(aloha_lib.FactoryLorentz):

    object_class = L_PSlash
    
    @classmethod
    def get_unique_name(self, spin1, spin2, particle):
        return '_P%s/_%s_%s' % (particle, spin1,spin2)


#===============================================================================
# Mass
#===============================================================================
class L_Mass(aloha_lib.LorentzObject):
    """ Helas Object for a Mass"""
       
    
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name,[], [])
            
    def create_representation(self):
        mass = aloha_lib.DVariable('M%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

class Mass(aloha_lib.FactoryLorentz):

    object_class = L_Mass
    
    @classmethod
    def get_unique_name(self, particle):
        return '_M%s' % particle

#===============================================================================
# Mass
#===============================================================================
class L_Coup(aloha_lib.LorentzObject):
    """ Helas Object for a Mass"""
       
    
    def __init__(self, name, nb):
        self.nb = nb
        aloha_lib.LorentzObject.__init__(self, name,[], [])
            
    def create_representation(self):
        coup = aloha_lib.Variable('COUP%s' % self.nb)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                coup, self.lorentz_ind, self.spin_ind)

class Coup(aloha_lib.FactoryLorentz):

    object_class = L_Coup
    
    @classmethod
    def get_unique_name(self, nb):
        return 'coup%s' % nb


#===============================================================================
# FCT
#===============================================================================
class L_FCT(aloha_lib.LorentzObject):
    """ Helas Object for a Mass"""
       
    
    def __init__(self, name, id):
        self.fctid = id
        aloha_lib.LorentzObject.__init__(self, name,[], [])
            
    def create_representation(self):
        var = aloha_lib.Variable('FCT%s' % self.fctid)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                var, self.lorentz_ind, self.spin_ind)
  
class FCT(aloha_lib.FactoryLorentz):

    object_class = L_FCT
    
    @classmethod
    def get_unique_name(self, name):
        
        return '_FCT%s' % name


#===============================================================================
# OverMass2
#===============================================================================
class L_OverMass2(aloha_lib.LorentzObject):
    """ Helas Object for 1/M**2 """
      
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [], [], tags=['OM%s' % particle])
    
    def create_representation(self):
        mass = aloha_lib.DVariable('OM%s' % self.particle)

        self.representation = aloha_lib.LorentzObjectRepresentation(
                                mass, self.lorentz_ind, self.spin_ind)

class OverMass2(aloha_lib.FactoryLorentz):
    
    object_class = L_OverMass2
    
    @classmethod
    def get_unique_name(self, particle):
        return '_OM2_%s' % particle


#===============================================================================
# P (Momenta)
#===============================================================================
class L_PT(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
    
    contract_first = 1
    
    def __init__(self, name, lorentz1, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name,[lorentz1], [],['PT%s'%particle])
        aloha_lib.KERNEL.add_tag((name,))
    
    def create_representation(self):
        #self.sub0 = aloha_lib.DVariable('P%s_0' % self.particle)
        self.sub1 = aloha_lib.DVariable('P%s_1' % self.particle)
        self.sub2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #self.sub3 = aloha_lib.DVariable('P%s_3' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): 0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): 0},                              
                                    self.lorentz_ind, [])


class PT(aloha_lib.FactoryLorentz):
    """ Helas Object for an Impulsion """
    
    object_class = L_PT
    
    #def __init__(self, lorentz1, particle):
    @classmethod
    def get_unique_name(self, lorentz1, particle):
        return '_PT^%s_%s' % (particle, lorentz1)




#===============================================================================
# Width
#===============================================================================
class L_Width(aloha_lib.LorentzObject):
    """ Helas Object for an Impulsion """
 
    
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [], [])
        
    def create_representation(self):
        width = aloha_lib.DVariable('W%s' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                            width, self.lorentz_ind, self.spin_ind)

class Width(aloha_lib.FactoryLorentz):
    
    object_class = L_Width
    
    @classmethod
    def get_unique_name(self, particle):
        return '_W%s' % particle

#===============================================================================
# Param
#===============================================================================
class L_Param(aloha_lib.LorentzObject):
    """ Object for a Model Parameter """
 
    
    def __init__(self, Lname, name):
        self.varname = name
        aloha_lib.LorentzObject.__init__(self, name, [], [])
        
    def create_representation(self):
        param = aloha_lib.Variable( self.varname, aloha_lib.ExtVariable)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                            param, [], [])

class Param(aloha_lib.FactoryLorentz):
    
    object_class = L_Param
    
    @classmethod
    def get_unique_name(self, name):
        if name == 'Pi':
            KERNEL.has_pi = True
        return 'Param_%s' % name

#===============================================================================
# Scalar
#===============================================================================
class L_Scalar(aloha_lib.LorentzObject):
    """ Helas Object for a Spinor"""
       
    
    def __init__(self, name, particle):
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [], [])
    

        
    def create_representation(self):
        rep = aloha_lib.Variable('S%s_1' % self.particle)
        self.representation= aloha_lib.LorentzObjectRepresentation(        
                                                                    rep, [], [])        

class Scalar(aloha_lib.FactoryLorentz):
    
    object_class = L_Scalar
    
    @classmethod   
    def get_unique_name(self,particle):
        return '_S%s' % particle       
#===============================================================================
# Spinor
#===============================================================================
class L_Spinor(aloha_lib.LorentzObject):
    """ Helas Object for a Spinor"""
    
    contract_first = 1
        
    def __init__(self, name, spin1, particle, prefactor=1):        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name,[], [spin1])
       
    def create_representation(self):
        self.sub0 = aloha_lib.Variable('F%s_1' % self.particle)
        self.sub1 = aloha_lib.Variable('F%s_2' % self.particle)
        self.sub2 = aloha_lib.Variable('F%s_3' % self.particle)
        self.sub3 = aloha_lib.Variable('F%s_4' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation(
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},         
                                    [],self.spin_ind)

class Spinor(aloha_lib.FactoryLorentz):
    """ Helas Object for a Spinor"""

    object_class = L_Spinor
    
    @classmethod
    def get_unique_name(self,spin1, particle):  
        return '_F%s_%s' % (particle,spin1)

#===============================================================================
# Vector
#===============================================================================
class L_Vector(aloha_lib.LorentzObject):
    """ Helas Object for a Vector"""
    
    contract_first = 1
    
    def __init__(self, name, lorentz, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [lorentz], [])
           
    def create_representation(self):
        self.sub0 = aloha_lib.Variable('V%s_1' % self.particle)
        self.sub1 = aloha_lib.Variable('V%s_2' % self.particle)
        self.sub2 = aloha_lib.Variable('V%s_3' % self.particle)
        self.sub3 = aloha_lib.Variable('V%s_4' % self.particle)

        self.representation= aloha_lib.LorentzObjectRepresentation( 
                                    {(0,): self.sub0, (1,): self.sub1, \
                                     (2,): self.sub2, (3,): self.sub3},  
                                    self.lorentz_ind, [])

class Vector(aloha_lib.FactoryLorentz):
    
    object_class = L_Vector
    
    @classmethod
    def get_unique_name(self, lor, particle):
        return '_V%s_%s' % (particle, lor)    

#===============================================================================
# Spin3/2
#===============================================================================
class L_Spin3Half(aloha_lib.LorentzObject):
    """ Helas Object for a Spin2"""
    
    def __init__(self, name, lorentz, spin, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [lorentz], [spin])

    
    def create_representation(self):

        self.sub00 = aloha_lib.Variable('R%s_1' % self.particle)
        self.sub01 = aloha_lib.Variable('R%s_2' % self.particle)
        self.sub02 = aloha_lib.Variable('R%s_3' % self.particle)
        self.sub03 = aloha_lib.Variable('R%s_4' % self.particle)

        self.sub10 = aloha_lib.Variable('R%s_5' % self.particle)
        self.sub11 = aloha_lib.Variable('R%s_6' % self.particle)
        self.sub12 = aloha_lib.Variable('R%s_7' % self.particle)
        self.sub13 = aloha_lib.Variable('R%s_8' % self.particle)
    
        self.sub20 = aloha_lib.Variable('R%s_9' % self.particle)
        self.sub21 = aloha_lib.Variable('R%s_10' % self.particle)
        self.sub22 = aloha_lib.Variable('R%s_11' % self.particle)
        self.sub23 = aloha_lib.Variable('R%s_12' % self.particle)
    
        self.sub30 = aloha_lib.Variable('R%s_13' % self.particle)
        self.sub31 = aloha_lib.Variable('R%s_14' % self.particle)
        self.sub32 = aloha_lib.Variable('R%s_15' % self.particle)
        self.sub33 = aloha_lib.Variable('R%s_16' % self.particle)
        
        rep = {(0,0): self.sub00, (0,1): self.sub01, (0,2): self.sub02, (0,3): self.sub03,
               (1,0): self.sub10, (1,1): self.sub11, (1,2): self.sub12, (1,3): self.sub13,
               (2,0): self.sub20, (2,1): self.sub21, (2,2): self.sub22, (2,3): self.sub23,
               (3,0): self.sub30, (3,1): self.sub31, (3,2): self.sub32, (3,3): self.sub33}
        
                
        self.representation= aloha_lib.LorentzObjectRepresentation( rep, \
                                    self.lorentz_ind, self.spin_ind)

class Spin3Half(aloha_lib.FactoryLorentz):
    
    object_class = L_Spin3Half

    @classmethod
    def get_unique_name(self, lor, spin, part):
        return 'Spin3Half%s^%s_%s' % (part, lor, spin)

#===============================================================================
# Spin2
#===============================================================================
class L_Spin2(aloha_lib.LorentzObject):
    """ Helas Object for a Spin2"""
    
        
    def __init__(self, name, lorentz1, lorentz2, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, name, [lorentz1, lorentz2], [])
    
    def create_representation(self):

        self.sub00 = aloha_lib.Variable('T%s_1' % self.particle)
        self.sub01 = aloha_lib.Variable('T%s_2' % self.particle)
        self.sub02 = aloha_lib.Variable('T%s_3' % self.particle)
        self.sub03 = aloha_lib.Variable('T%s_4' % self.particle)

        self.sub10 = aloha_lib.Variable('T%s_5' % self.particle)
        self.sub11 = aloha_lib.Variable('T%s_6' % self.particle)
        self.sub12 = aloha_lib.Variable('T%s_7' % self.particle)
        self.sub13 = aloha_lib.Variable('T%s_8' % self.particle)

        self.sub20 = aloha_lib.Variable('T%s_9' % self.particle)
        self.sub21 = aloha_lib.Variable('T%s_10' % self.particle)
        self.sub22 = aloha_lib.Variable('T%s_11' % self.particle)
        self.sub23 = aloha_lib.Variable('T%s_12' % self.particle)

        self.sub30 = aloha_lib.Variable('T%s_13' % self.particle)
        self.sub31 = aloha_lib.Variable('T%s_14' % self.particle)
        self.sub32 = aloha_lib.Variable('T%s_15' % self.particle)
        self.sub33 = aloha_lib.Variable('T%s_16' % self.particle)
        
        rep = {(0,0): self.sub00, (0,1): self.sub01, (0,2): self.sub02, (0,3): self.sub03,
               (1,0): self.sub10, (1,1): self.sub11, (1,2): self.sub12, (1,3): self.sub13,
               (2,0): self.sub20, (2,1): self.sub21, (2,2): self.sub22, (2,3): self.sub23,
               (3,0): self.sub30, (3,1): self.sub31, (3,2): self.sub32, (3,3): self.sub33}
        
                
        self.representation= aloha_lib.LorentzObjectRepresentation( rep, \
                                    self.lorentz_ind, [])

class Spin2(aloha_lib.FactoryLorentz):
    
    object_class = L_Spin2

    @classmethod
    def get_unique_name(self, lor1, lor2, part):
        return 'Spin2^%s_%s_%s' % (part, lor1, lor2)

#===============================================================================
# Gamma
#===============================================================================
class L_Gamma(aloha_lib.LorentzObject):
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

    def __init__(self, name, lorentz, spin1, spin2):
        aloha_lib.LorentzObject.__init__(self,name,[lorentz], [spin1, spin2])
            
    def create_representation(self):
                
        self.representation = aloha_lib.LorentzObjectRepresentation(self.gamma,
                                self.lorentz_ind,self.spin_ind)

class Gamma(aloha_lib.FactoryLorentz):
    
    object_class = L_Gamma
    
    @classmethod
    def get_unique_name(self, lor, spin1, spin2):
        return 'Gamma^%s_%s_%s' % (lor, spin1, spin2)
        
        
#===============================================================================
# Sigma
#===============================================================================
class L_Sigma(aloha_lib.LorentzObject):
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

    def __init__(self, name, lorentz1, lorentz2, spin1, spin2):
            aloha_lib.LorentzObject.__init__(self, name, [lorentz1, lorentz2], \
                                                  [spin1, spin2])

    def create_representation(self):
                
        self.representation = aloha_lib.LorentzObjectRepresentation(self.sigma,
                                self.lorentz_ind,self.spin_ind)

class Sigma(aloha_lib.FactoryLorentz):
    
    object_class = L_Sigma

    @classmethod
    def get_unique_name(self, lorentz1, lorentz2, spin1, spin2):
        return 'Sigma_[%s,%s]^[%s,%s]' % (spin1, spin2, lorentz1, lorentz2)


#===============================================================================
# Gamma5
#===============================================================================        
class L_Gamma5(aloha_lib.LorentzObject):
    
    #gamma5 = [[-1, 0, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    gamma5 = {(0,0): -1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): -1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 1}
    
    def __init__(self, name, spin1, spin2):
        aloha_lib.LorentzObject.__init__(self, name, [], [spin1, spin2])

    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.gamma5,
                                             self.lorentz_ind,self.spin_ind) 

class Gamma5(aloha_lib.FactoryLorentz):
    
    object_class = L_Gamma5

    @classmethod
    def get_unique_name(self, spin1, spin2):
        return 'Gamma5_%s_%s' % (spin1, spin2)
        
#===============================================================================
# Conjugate Matrices
#===============================================================================
class L_C(aloha_lib.LorentzObject):
    
    #[0, -1, 0, 0] [1,0,0,0] [0,0,0,1],[0,0,-1,0]
    
    Cmetrix = {(0,0): 0, (0,1): -1, (0,2): 0, (0,3): 0,\
              (1,0): 1, (1,1): 0, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 0, (2,3): 1,\
              (3,0): 0, (3,1): 0, (3,2): -1, (3,3): 0} 
    
    def __init__(self, name, spin_list):

        # spin_list is automatically ordered. The sign for the symmetrization
        # is set in the Factory routine       
        aloha_lib.LorentzObject.__init__(self, name, [], spin_list)


    def create_representation(self):
        self.representation = aloha_lib.LorentzObjectRepresentation(self.Cmetrix,
                                             self.lorentz_ind,self.spin_ind) 
    
class C(aloha_lib.FactoryLorentz):
    
    object_class = L_C
    
    def __new__(cls, spin1, spin2):
        
       spin_list = [spin1, spin2]
       spin_list.sort()
       sign = give_sign_perm(spin_list, [spin1, spin2])
       name = cls.get_unique_name(spin_list)
       if sign == 1:
           return aloha_lib.FactoryVar.__new__(cls, name, cls.object_class, spin_list)
       else:
           out = aloha_lib.FactoryVar.__new__(cls, name, cls.object_class, spin_list)
           out.prefactor = -1
           return out

    @classmethod
    def get_unique_name(cls, spin_list):
        return "C_%s_%s" % tuple(spin_list)

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
class L_Epsilon(aloha_lib.LorentzObject):
    """ The fully anti-symmetric object in Lorentz-Space """
 
    def give_parity(self, perm):
        """return the parity of the permutation"""
        assert set(perm) == set([0,1,2,3]) 
        
        i1 , i2, i3, i4 = perm
        #formula found on wikipedia
        return -self.sign * ((i2-i1) * (i3-i1) *(i4-i1) * (i3-i2) * (i4-i2) *(i4-i3))/12 
   
    # DEFINE THE REPRESENTATION OF EPSILON
           
    def __init__(self, name, lorentz1, lorentz2, lorentz3, lorentz4):
       
       lorentz_list = [lorentz1 , lorentz2, lorentz3, lorentz4]
       #order_lor = list(lorentz_list)
       #order_lor.sort()
       
       #self.sign = give_sign_perm(order_lor, lorentz_list)
       self.sign=1
       aloha_lib.LorentzObject.__init__(self, name, lorentz_list, [])


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

            L_Epsilon.epsilon = epsilon
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.epsilon,
                                self.lorentz_ind,self.spin_ind)


class Epsilon(aloha_lib.FactoryLorentz):
         
    object_class = L_Epsilon
    
    @classmethod
    def get_unique_name(cls,l1,l2,l3,l4):
        return '_EPSILON_%s_%s_%s_%s' % (l1,l2,l3,l4)
    
            
#===============================================================================
# Metric
#===============================================================================
class L_Metric(aloha_lib.LorentzObject):
    
    metric = {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): -1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): -1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): -1}
    
    
    #[[1, 0, 0,0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        
    def __init__(self, name, lorentz1, lorentz2):
        aloha_lib.LorentzObject.__init__(self,name,[lorentz1, lorentz2], [])
            
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.metric,
                                             self.lorentz_ind,self.spin_ind)     

class Metric(aloha_lib.FactoryLorentz):
         
    object_class = L_Metric
    
    @classmethod
    def get_unique_name(cls,l1,l2):
        if l1<l2:
            return '_ETA_%s_%s' % (l1,l2)
        else:
            return '_ETA_%s_%s' % (l2,l1)
#===============================================================================
# Identity
#===============================================================================
class L_Identity(aloha_lib.LorentzObject):
    
    #identity = [[1, 0, 0,0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    identity = {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): 1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 1}

    
    def __init__(self, name, spin1, spin2):
        aloha_lib.LorentzObject.__init__(self, name, [],[spin1, spin2])
            
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.identity,
                                             self.lorentz_ind,self.spin_ind)

class Identity(aloha_lib.FactoryLorentz):
    
    object_class = L_Identity

    @classmethod
    def get_unique_name(self, spin1, spin2):
        return 'Id_%s_%s' % (spin1, spin2)

#===============================================================================
# IdentityL
#===============================================================================
class L_IdentityL(aloha_lib.LorentzObject):
    
    #identity = [[1, 0, 0,0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    identity = {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): 1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 1}

    
    def __init__(self, name, l1, l2):
        aloha_lib.LorentzObject.__init__(self, name, [l1,l2], [])
            
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.identity,
                                             self.lorentz_ind,self.spin_ind)

class IdentityL(aloha_lib.FactoryLorentz):
    
    object_class = L_IdentityL

    @classmethod
    def get_unique_name(self, l1, l2):
        return 'IdL_%s_%s' % (l1, l2)

#===============================================================================
# ProjM 
#===============================================================================    
class L_ProjM(aloha_lib.LorentzObject):
    """ A object for (1-gamma5)/2 """
    
    #projm = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    projm= {(0,0): 1, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): 1, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 0, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 0}
                
    def __init__(self,name, spin1, spin2):
        """Initialize the object"""
        aloha_lib.LorentzObject.__init__(self, name, [], [spin1, spin2])
            
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.projm,
                                             self.lorentz_ind,self.spin_ind)    

class ProjM(aloha_lib.FactoryLorentz):
    
    object_class = L_ProjM
    
    @classmethod
    def get_unique_name(self, spin1, spin2):
            return 'PROJM_%s_%s' % (spin1, spin2)
#===============================================================================
# ProjP 
#===============================================================================    
class L_ProjP(aloha_lib.LorentzObject):
    """A object for (1+gamma5)/2 """
    
    #projp = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    projp = {(0,0): 0, (0,1): 0, (0,2): 0, (0,3): 0,\
              (1,0): 0, (1,1): 0, (1,2): 0, (1,3): 0,\
              (2,0): 0, (2,1): 0, (2,2): 1, (2,3): 0,\
              (3,0): 0, (3,1): 0, (3,2): 0, (3,3): 1}
    
    def __init__(self,name, spin1, spin2):
        """Initialize the object"""
        aloha_lib.LorentzObject.__init__(self, name, [], [spin1, spin2])

          
    def create_representation(self):
        
        self.representation = aloha_lib.LorentzObjectRepresentation(self.projp,
                                            self.lorentz_ind, self.spin_ind)    

class ProjP(aloha_lib.FactoryLorentz):
    
    object_class = L_ProjP
    
    @classmethod
    def get_unique_name(self, spin1, spin2):
        
        return 'PROJP_%s_%s' % (spin1, spin2)

#===============================================================================
# EPSL (longitudinal part of the epsilon normmaized to avoid denominator)
#===============================================================================
class L_EPSL(aloha_lib.LorentzObject):
    """ eps^mu_L (polarization vector) 
        (kx**2+ky**2+kz**2, E*kz, E*ky, E*kz)
        the normalization 1/m *|\vec{k} is NOT include
    
    """
    
    def __init__(self, name, lor1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [lor1], [])
    
    def create_representation(self):
        """create representation"""
        E = aloha_lib.DVariable('P%s_0' % self.particle)
        m = aloha_lib.DVariable('M%s' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        p3 = aloha_lib.DVariable('P%s_3' % self.particle)    

    
        norm = (p1**2+p2**2+p3**2)
        eps = {
            (0,): norm,
            (1,): p1*E,
            (2,): p2*E ,
            (3,): p3*E ,
            }
#        eps = {
#            (0,): norm-E*E,
#            (1,): 0,
#            (2,): 0 ,
#            (3,): 0 ,
#            }
    
        self.representation = aloha_lib.LorentzObjectRepresentation(eps,
                                self.lorentz_ind,self.spin_ind)

class EPSL(aloha_lib.FactoryLorentz):

    object_class = L_EPSL
    
    @classmethod
    def get_unique_name(self, lor1, particle):
        return '_EPSL%s_%s_' % (particle, lor1)

#===============================================================================
# EPSTR (transverse part of the epsilon normmalized to avoid denominator -- one of the eigenstate)
#        Real part
#===============================================================================
class L_EPST1(aloha_lib.LorentzObject):
    """ eps^mu_1 (polarization vector) 
        (0, kx kz  , ky kz,  -k_T^2)
        the normalization 1/k_T / norm(k) is NOT include
    
    """
    
    def __init__(self, name, lor1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [lor1], [])
    
    def create_representation(self):
        """create representation"""
#        E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        p3 = aloha_lib.DVariable('P%s_3' % self.particle)    

    
        kT = (p1**2+p2**2)

        eps = {
            (0,): 0,
            (1,): (p1)*(p3),
            (2,): p2*(p3),
            (3,):  -kT ,
            }
#        eps = {
#            (0,): norm-E*E,
#            (1,): 0,
#            (2,): 0 ,
#            (3,): 0 ,
#            }
    
        self.representation = aloha_lib.LorentzObjectRepresentation(eps,
                                self.lorentz_ind,self.spin_ind)

class EPST1(aloha_lib.FactoryLorentz):

    object_class = L_EPST1
    
    @classmethod
    def get_unique_name(self, lor1, particle):
        return '_EPST1%s_%s_' % (particle, lor1)

#===============================================================================
# EPSTI (transverse part of the epsilon normmalized to avoid denominator -- one of the eigenstate)
#        Imaginary part
#===============================================================================

class L_EPST2(aloha_lib.LorentzObject):
    """ eps^mu_+ (polarization vector) 
        (0,  ky k_T ,  -i kx k_T, 0)
 
        the normalization 1/(kT*norm(k)) is NOT include here
        as well as the k_T term (to avoid the square-root)
    
    """
    
    def __init__(self, name, lor1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [lor1], [])
    
    def create_representation(self):
        """create representation"""
 #       E = aloha_lib.DVariable('P%s_0' % self.particle)
 #       m = aloha_lib.DVariable('M%s' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
 #       p3 = aloha_lib.DVariable('P%s_3' % self.particle)    


        eps = {
            (0,): 0,
            (1,): -p2,
            (2,): (p1),
            (3,):  0 ,
            }
#        eps = {
#            (0,): norm-E*E,
#            (1,): 0,
#            (2,): 0 ,
#            (3,): 0 ,
#            }
    
        self.representation = aloha_lib.LorentzObjectRepresentation(eps,
                                self.lorentz_ind,self.spin_ind)
        
class EPST2(aloha_lib.FactoryLorentz):

    object_class = L_EPST2
    
    @classmethod
    def get_unique_name(self, lor1, particle):
        return '_EPST2%s_%s_' % (particle, lor1)
#===============================================================================
# UFP U fermion plus component (first two component)
#       the denominator is include in the denominator of the propagator
#===============================================================================
fsign = -1
    
class L_UFP(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): fwm * normz,
            (1,): fwm * (fsign * p1 + complex(0,fsign)*p2),
            (2,): fwp * normz,
            (3,): fwp * (fsign* p1 + complex(0,fsign)*p2),
            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)


class UFP(aloha_lib.FactoryLorentz):

    object_class = L_UFP
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_UFP%s_%s_' % (particle, spin1)

class L_UFM(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): fwp * (-1*fsign*p1 + complex(0,fsign)*p2),
            (1,): fwp * normz,
            (2,): fwm * (-1*fsign*p1 + complex(0,fsign)*p2),
            (3,): fwm * normz,
            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)
        
class UFM(aloha_lib.FactoryLorentz):

    object_class = L_UFM
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_UFM%s_%s_' % (particle, spin1)
   
class L_UFPC(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): fwp * normz,
            (1,): fwp * (fsign* p1 + complex(0,-fsign)*p2),
            (2,): fwm * normz,
            (3,): fwm * (fsign* p1 + complex(0,-fsign)*p2),
            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)

class UFPC(aloha_lib.FactoryLorentz):

    object_class = L_UFPC
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_UFPC%s_%s_' % (particle, spin1)

class L_UFMC(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): fwm * (-fsign*p1 + complex(0,-fsign)*p2),
            (1,): fwm * normz,
            (2,): fwp * (-fsign*p1 + complex(0,-fsign)*p2),
            (3,): fwp * normz,
            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)

class UFMC(aloha_lib.FactoryLorentz):

    object_class = L_UFMC
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_UFMC%s_%s_' % (particle, spin1)

class L_VFP(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): -1*fwp * (-1 *fsign * p1 + complex(0,fsign)*p2),
            (1,): -1*fwp * normz,
            (2,): 1*fwm * (-1 *fsign * p1 + complex(0,fsign)*p2),
            (3,): 1*fwm * normz,
            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)


class VFP(aloha_lib.FactoryLorentz):

    object_class = L_VFP
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_VFP%s_%s_' % (particle, spin1)

class L_VFPC(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): fwm * (-1 *fsign * p1 + complex(0,-1*fsign)*p2),
            (1,): fwm * normz,
            (2,): -1*fwp * (-1 *fsign * p1 + complex(0,-1*fsign)*p2),
            (3,): -1*fwp * normz,
            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)


class VFPC(aloha_lib.FactoryLorentz):

    object_class = L_VFPC
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_VFPC%s_%s_' % (particle, spin1)


class L_VFM(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): fwm * normz,
            (1,): fwm * (fsign * p1 + complex(0,fsign)*p2),
            (2,): -1*fwp * normz,
            (3,): -1*fwp * (fsign * p1 + complex(0,fsign)*p2),
            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)


class VFM(aloha_lib.FactoryLorentz):

    object_class = L_VFM
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_VFM%s_%s_' % (particle, spin1)

class L_VFMC(aloha_lib.LorentzObject):
    """ 

    """
    
    def __init__(self, name, spin1, particle):
        
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self,name, [], [spin1],
                                         tags=['|p%s|_z' % particle,
                                               'FWP%s' % particle,
                                               'FWM%s' % particle])
    
    def create_representation(self):
        """create representation"""
        #E = aloha_lib.DVariable('P%s_0' % self.particle)
        p1 = aloha_lib.DVariable('P%s_1' % self.particle)
        p2 = aloha_lib.DVariable('P%s_2' % self.particle)
        #p3 = aloha_lib.DVariable('P%s_3' % self.particle)    
        normz = aloha_lib.DVariable('TnormZ%s' % self.particle)
        fwp = aloha_lib.DVariable('FWP%s' % self.particle)
        fwm = aloha_lib.DVariable('FWM%s' % self.particle)
    

        u = {
            (0,): -1*fwp * normz,
            (1,): -1*fwp * (fsign * p1 + complex(0,-1*fsign)*p2),
            (2,): fwm * normz,
            (3,): fwm * (fsign * p1 + complex(0,-1*fsign)*p2),

            }

    
        self.representation = aloha_lib.LorentzObjectRepresentation(u,
                                self.lorentz_ind,self.spin_ind)


class VFMC(aloha_lib.FactoryLorentz):

    object_class = L_VFMC
    
    @classmethod
    def get_unique_name(self, spin1, particle):
        return '_VFMC%s_%s_' % (particle, spin1)

#===============================================================================
# Denominator Propagator 
#===============================================================================    
class DenominatorPropagator(aloha_lib.LorentzObject):
    """The Denominator of the Propagator"""
    
    def __new__(cls, particle):
    
        name = 'DenomP%s' % particle
        return  aloha_lib.Variable.__new__(cls, name)    
    
    def __init__(self, particle):
        if self:
            return
        self.particle = particle
        aloha_lib.LorentzObject.__init__(self, [], [])
    
    def get_unique_name(self,*args):
        return 'DenomP%s' % self.particle
    
    
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


SpinorPropagatorout = lambda spin1, spin2, particle: -1 * (Gamma('mu', spin1, spin2) * \
                    P('mu', particle) - Mass(particle) * Identity(spin1, spin2))

SpinorPropagatorin = lambda spin1, spin2, particle: (Gamma('mu', spin1, spin2) * \
                    P('mu', particle) + Mass(particle) * Identity(spin1, spin2))


VectorPropagator = lambda l1, l2, part: complex(0,1)*(-1 * Metric(l1, l2) + OverMass2(part) * \
                                    Metric(l1,'I3')* P('I3', part) * P(l2, part))

VectorPropagatorMassless= lambda l1, l2, part: complex(0,-1) * Metric(l1, l2)
    

Spin3halfPropagatorin =  lambda mu, nu, s1, s2, part: (\
    -1/3 * (Gamma(mu,s1,-2) + Identity(s1, -2) *  P(mu, part) * Mass(part) * OverMass2(part))* \
    (PSlash(-2,-3, part) - Identity(-2,-3) * Mass(part)) * \
    ( Gamma(nu, -3, s2)+ Mass(part) * OverMass2(part) * Identity(-3, s2) * P(nu, part) ) - \
    (PSlash(s1,s2, part) + Mass(part) * Identity(s1,s2)) * (Metric(mu, nu) - OverMass2(part) * P(mu, part) * P(nu,part)))


Spin3halfPropagatorout =  lambda mu, nu, s1, s2, part: ( \
  -1/3 * (Gamma(mu,s1,-2) - Identity(s1, -2) *  P(mu, part) * Mass(part) * OverMass2(part))* \
    (-1*PSlash(-2,-3, part) - Identity(-2,-3) * Mass(part)) * \
    ( Gamma(nu, -3, s2)- Mass(part) * OverMass2(part) * Identity(-3, s2) * P(nu, part) ) - \
    (-1*PSlash(s1,s2, part) 
     + Mass(part) * Identity(s1,s2)) * (Metric(mu, nu) - OverMass2(part) * P(mu, part) * P(nu,part)))


Spin3halfPropagatorMasslessOut = lambda mu, nu, s1, s2, part: Gamma(nu, s1,-1) * PSlash(-1,-2, part) * Gamma(mu,-2, s2)
Spin3halfPropagatorMasslessIn = lambda mu, nu, s1, s2, part: -1 * Gamma(mu, s1,-1) * PSlash(-1,-2, part) * Gamma(nu,-2, s2)


Spin2masslessPropagator = lambda mu, nu, alpha, beta: 1/2 *( Metric(mu, alpha)* Metric(nu, beta) +\
                     Metric(mu, beta) * Metric(nu, alpha) - Metric(mu, nu) * Metric(alpha, beta))



Spin2Propagator =  lambda mu, nu, alpha, beta, part: Spin2masslessPropagator(mu, nu, alpha, beta) + \
                - 1/2 * OverMass2(part) * (Metric(mu,alpha)* P(nu, part) * P(beta, part) + \
                                Metric(nu, beta) * P(mu, part) * P(alpha, part) + \
                                Metric(mu, beta) * P(nu, part) * P(alpha, part) + \
                                Metric(nu, alpha) * P(mu, part) * P(beta , part) )+ \
                1/6 * (Metric(mu,nu) + 2 * OverMass2(part) * P(mu, part) * P(nu, part)) * \
                      (Metric(alpha,beta) + 2 * OverMass2(part) * P(alpha, part) * P(beta, part))
    















