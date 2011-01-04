################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
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
"""Module for finding quantum numbers. Using Monte Carlo trick as in 
   searching for the ground state of Ising model.
   QNumber_Add: a dictionary from pdg_code to quantum number,
                if anti_pdg_code is assigned, return the inverted 
                quantum number (*-1). QNumber_Add can renew itself for all 
                particles (globally) or only flip the value of a particle 
                (locally). The function,'conservation_measure', defines how 
                much the given qnumber is conserved.
   find_QNumber_Add: the main function performs the Monte Carlo search of
                     valid addictive quantum number. It contains a large loop 
                     for trying globally different quantum number and two loops
                     for flipping quantum number of one particle at a time.
   QNumber_Parity: similar to QNumber_Add, but the allow number is only 1 and -1
                   The value is the same for both particle and anti-particle.
                   the generate_global will change only particles that are 
                   listed in parity_free_ids.
   interaction_reduction: the function the reduce the unnecessary interactions.
                          It create a list with each item in it corresponds
                          to a valid interaction. Each item records the 
                          particles involved in this interaction. Note that
                          the number for a single particle is at least 1 and 
                          particle with known parity is removed.
   find_QNumber_Parity: the main function performs the Monte Carlo search of
                        multiplicative quantum number. Executing the
                        interaction_reduction in the begining is optional.
   """

import array
import cmath
import copy
import itertools
import logging
import math
import os
import re
import random

import madgraph.core.base_objects as base_objects
import models.import_ufo as import_ufo
from madgraph import MadGraph5Error, MG5DIR

#===============================================================================
# Logger
#===============================================================================
logger = logging.getLogger('quantum_number')
#logger.setLevel(logging.INFO)
h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s: %(message)s")
h.setFormatter(f)
logger.addHandler(h)

#===============================================================================
# QNumber
#===============================================================================
class QNumber(dict):
    """QNumber is a class for both addive and multiplicative 
       quantum number object."""
    
    qrange = 1
    qtype = 'GENERAL'

    def __init__(self, model, qrange=1):
        """Create a new QNumber object. Use model to import all the particles.
           quantum number for a particle is limited by qrange."""

        dict.__init__(self)
        for part in model['particles']:
            self[part.get_pdg_code()] = 1

        self.qrange = qrange
        self.model = model
        self.generate_global()

    def __missing__(self, key):
        """Return inverse quantum number if anti_pdg_code is given,
           Else raise KeyError."""
        if  -key in self.keys():
            pass
        else:
            raise KeyError

    def get(self, pid):
        """Return quantum number if not anti_particle, else return the inverse
           quantum number"""
        return self[pid]

    def set(self, pid, value):
        """Set the quantum number of particle pid with value in qrange."""
        pass

    def generate_global(self):
        """Generate a (globally) different QNumber and apply on itself"""
        pass

    def generate_local(self, pid):
        """Change the quantum number of particle(pid) to other allow values"""
        pass


#===============================================================================
# QNumber_Add
#===============================================================================
class QNumber_Add(QNumber):
    """QNumber_Add is the addive quantum number object."""
    
    qrange = 1
    qtype = 'ADDICTIVE'

    def __missing__(self, key):
        """Return inverse quantum number if anti_pdg_code is given,
           Else raise KeyError."""
        if  -key in self.keys():
            return -self[-key]
        else:
            raise KeyError

    def set(self, pid, value):
        """Set the quantum number of particle pid with value in qrange."""
        pass

    def generate_global(self):
        """Generate a (globally) different QNumber_Add and apply on itself"""
        for pid in self.keys():
            self[pid] = random.randint(-self.qrange, self.qrange)

    def generate_local(self, pid):
        """Change the quantum number of particle(pid) to other allow values"""
        randint = self[pid]
        while randint == self[pid]:
            randint = random.randint(-self.qrange, self.qrange)

        self[pid] = randint

    def conservation_measure(self, model):
        """Calculate the conservation of quantum number in all interactions.
           Return the measure of conservation."""

        value = 0
        for inter in model['interactions']:
            #Total quantum number in this interaction
            value_int = 0
            for part in inter['particles']:
                value_int += self.get(part.get_pdg_code())
        
            #Square the value to avoid negative result
            value_int = value_int ** 2
            #if value_int != 0:
            #print inter['id']
    
            #Add to total measure for each interaction
            value += value_int

        return value

def find_QNumber_Add(model):
    """Main function of finding quantum numbers. 
       Results are stored in qnumber_list. Monte Carlo method is applied"""

    qnumber_list = []
    qrange = 1
    qnumber = QNumber_Add(model, qrange)
    measure = qnumber.conservation_measure(model)
    
    #Large loop: test globally different qnumber
    for i in range(0, 10):

        #Test if initial measure is zero; 
        #if so, record it and generate a new one
        if measure == 0:
           if not qnumber in qnumber_list:
               qnumber_list.append(copy.copy(qnumber))
               print 'Quantum number found!!! \n', qnumber
               #Generate a qnumber for new search
               qnumber.generate_global()

        #Search locally near the given qnumber (1st local loop)
        for j in range(0, 5):
            #Change locally one by one (2nd local loop)
            for part in model['particles']:
                qnumber_old = copy.copy(qnumber)
                qnumber.generate_local(part.get('pdg_code'))
                measure_new = qnumber.conservation_measure(model)
                #If new qnumber is conserved, 
                #record it and break the local search
                if measure_new == 0:
                    if not qnumber in qnumber_list:
                        qnumber_list.append(copy.copy(qnumber))
                        print 'Quantum number is found!!! \n', qnumber
                        break
                #Move to new qnumber if new measure is closer to zero
                elif measure_new < measure:
                    measure = measure_new
                    #print 'Qnumber renew in (%s , %s)!' % (i, j)
                else:
                    qnumber = qnumber_old
            
            #Break the 1st local search loop if qnumber conserved.
            if measure == 0:
                break

        #No conserved qnumber is found, generate a new one.
        qnumber.generate_global()
        measure = qnumber.conservation_measure(model)
    
    #Record the result
    path = os.path.join(MG5DIR, 'quantum_number', model['name'])
    fdata = open(os.path.join(path, 'qnumber_list_add.dat'), 'w')
    fdata.write(str(qnumber_list))
    fdata.close()

    return qnumber_list

#===============================================================================
# QNumber_Parity
#===============================================================================

"""Global variable used in multiplicative quantum numbers"""
sm_ids = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25]
parity_fix_ids = []
parity_free_ids = []
reduced_interactions = []

class QNumber_Parity(QNumber):
    """QNumber_Parity is the parity-like quantum number object."""
    
    qrange = 1
    qtype = 'Parity'

    def __init__(self, model, qrange=1):
        """Create a new QNumber object. Use model to import all the particles.
           quantum number for a particle is limited by qrange."""

        dict.__init__(self)

        if not parity_free_ids:
            parity_fix_ids.append(sm_ids)
            parity_fix_ids.append([])
            for part in model['particles']:
                self[part.get_pdg_code()] = 1
                if not part.get_pdg_code() in sm_ids:
                    parity_free_ids.append(part.get_pdg_code())
        else:
            for part in model['particles']:
                self[part.get_pdg_code()] = 1
            
        self.qrange = qrange
        self.model = model
        self.generate_global()

    def __missing__(self, key):
        """Return inverse quantum number if anti_pdg_code is given,
           Else raise KeyError."""
        if  -key in self.keys():
            return self[-key]
        else:
            raise KeyError

    def set(self, pid, value):
        """Set the quantum number of particle pid with value in qrange."""
        pass

    def generate_global(self):
        """Generate a (globally) different QNumber_Parity and apply on itself"""
        
        for pid in parity_free_ids:
            if not pid in sm_ids:
                self[pid] = random.choice([-1, 1])

    def generate_local(self, pid):
        """Flip the quantum number of particle(pid)."""
        self[pid] = -self[pid]

    def conservation_measure(self, model):
        """Calculate the conservation of quantum number in all interactions.
           Return the measure of conservation.
           Use the result of interaction_reduction."""

        #Create a primordial reduced_interactions if not exists
        if not reduced_interactions:
            for i, inter in enumerate(model['interactions']):
                reduced_interactions.append([])
                for part in inter['particles']:
                    #if part.get('pdg_code') in parity_free_ids:
                        reduced_interactions[i].append(part.get('pdg_code'))
        
        #Calculate the measure
        value = 0
        for inter in reduced_interactions:
            #Total quantum number in this interaction
            value_int = 1
            for pid in inter:
                value_int *= self[pid]
            #If parity is failed increase the value
            if value_int == -1:
                value += 1

        return value

    def conservation_measure_old(self, model):
        """Calculate the conservation of quantum number in all interactions.
           Return the measure of conservation.
           Use the result of interaction_reduction."""

        value = 0
        for inter in model['interactions']:
            #Total quantum number in this interaction
            value_int = 1
            for part in inter['particles']:
                value_int *= self[part.get('pdg_code')]
            #If parity is failed increase the value
            if value_int == -1:
                value += 1

        return value

    

def interaction_reduction(model):
    """Reduce the interaction by create a list with each element corresponds
       to the particles without fixed parity that are involved in a 
       particular interaction.

       Algorithm:
       
       1. Read all interactions into a list 'reduced_interactions'

       2. Read one non-sm particle at a time, if it has been read before, 
          pop that particle, else record it. (because parity^2 is 1 regardless
          of a)

       3. If the interaction has only one valid particle, set the parity of
          particle as 1 and record in the list 'parity_fix_ids'.

       4. Scan all interactions again. If there is any particle which parity 
          is fixed, remove it.

       5. Repeat step 3 and 4, until the list 'reduced_interactions' doesn't
          change at all."""

    #If reduced_interactions exists, return it
    if reduced_interactions:
        return
    #Initialize the parity_fix_ids and parity_free_ids if haven't been setup.
    if not parity_free_ids:
        parity_fix_ids.append(sm_ids)
        parity_fix_ids.append([])
        parity_free_ids.extend([p.get('pdg_code') for p in model['particles'] \
                                    if not p.get('pdg_code') in sm_ids])

    for i, inter in enumerate(model['interactions']):
        reduced_interactions.append([])
        for part in inter['particles']:
            pid = part.get('pdg_code')
            #Proceed if part is not in parity_fix_ids (including sm particles)
            if pid in parity_free_ids:
                #If pid is not in the interaction yet, append it
                if not pid in reduced_interactions[i]:
                    reduced_interactions[i].append(pid)
                #If pid is there already, remove it since double particle
                #is equivalent to none.
                else:
                    reduced_interactions[i].remove(pid)

        #If only one particle in this interaction, its parity must be 1
        if len(reduced_interactions[i]) == 1:
            #Remove this pid from reduced_interactions and parity_free_ids
            #Add to parity_fix_ids
            p_fix = reduced_interactions[i].pop(0)
            parity_fix_ids[1].append(p_fix)
            parity_free_ids.remove(p_fix)

            
    change = True
    while change:
        #Assume there is no change.
        change = False
        for inter in reduced_interactions:
            #If there is any particle that has been fixed, remove it.
            #And we have 'change' now.
            for p in inter:
                if p in parity_fix_ids:
                    inter.remove(p)
                    change = True

            #Only one particle left...
            if len(inter) == 1:
                #The parity of the last particle must be 1
                p_fix = reduced_interactions[i].pop(0)
                parity_fix_ids[1].append(p_fix)
                parity_free_ids.remove(p_fix)
    
    print 'Numbers of interaction before reduction:', len(reduced_interactions)

    #Remove the interactions without any particle in it
    for inter in reduced_interactions:
        try:
            reduced_interactions.remove([])
        #If ValueError, no empty interaction left.
        except ValueError:
            break

    print 'Numbers of interaction after reduction:', len(reduced_interactions)
    return reduced_interactions
                    
def find_QNumber_Parity(model):
    """Main function of finding quantum numbers. 
       Results are stored in qnumber_list. Monte Carlo method is applied"""

    #Perform interaction_reduction first to get the correct 'parity_free_ids'
    interaction_reduction(model)

    qnumber_list = []
    qnumber = QNumber_Parity(model)
    measure = qnumber.conservation_measure(model)
    
    logger.info('\nNumber of interaction used %d'\
                            % len(reduced_interactions))
    logger.info('\nParticle with free parity after reduction:\n %s'\
                            % str(parity_free_ids))
    logger.info('\nParticle with fix parity after reduction:\n %s' \
                            % str(parity_fix_ids))
    
    #print '\nParticle with free parity after reduction:\n', parity_free_ids
    #print '\nParticle with fix parity after reduction:\n', parity_fix_ids
    #Large loop: test globally different qnumber
    for i in range(0, 10):

        #Test if initial measure is zero; 
        #if so, record it and generate a new one
        if measure == 0:
           if not qnumber in qnumber_list:
               qnumber_list.append(copy.copy(qnumber))
               print 'Quantum number found!!! \n', qnumber
               #Generate a qnumber for new search
               qnumber.generate_global()

        #Search locally near the given qnumber (1st local loop)
        for j in range(0, 5):
            #Change locally one by one (2nd local loop)
            for part in parity_free_ids:
                qnumber_old = copy.copy(qnumber)
                qnumber.generate_local(part)
                measure_new = qnumber.conservation_measure(model)
                #If new qnumber is conserved, 
                #record it and break the local search
                if measure_new == 0:
                    if not qnumber in qnumber_list:
                        qnumber_list.append(copy.copy(qnumber))
                        print 'Quantum number is found!!! \n', qnumber
                        break
                #Move to new qnumber if new measure is closer to zero
                elif measure_new < measure:
                    measure = measure_new
                    #print 'Qnumber renew in (%s , %s)!' % (i, j)
                else:
                    qnumber = qnumber_old
            
            #Break the 1st local search loop if qnumber conserved.
            if measure == 0:
                break

        #No conserved qnumber is found, generate a new one.
        qnumber.generate_global()
        measure = qnumber.conservation_measure(model)
    
    #Record the result
    #path = os.path.join(MG5DIR, 'quantum_number', model['name'])
    #fdata = open(os.path.join(path, 'qnumber_list_parity.dat'), 'w')
    #fdata.write(str(qnumber_list))
    #fdata.close()

    return qnumber_list

def find_QNumber_Parity_old(model):
    """Main function of finding quantum numbers. 
       Results are stored in qnumber_list. Monte Carlo method is applied"""

    qnumber_list = []
    qnumber = QNumber_Parity(model)
    measure = qnumber.conservation_measure_old(model)
    
    print '\nNumber of interaction used %d' % len(model['interactions'])
    print '\nParticle with free parity after reduction:\n', parity_free_ids
    print '\nParticle with fix parity after reduction:\n', parity_fix_ids
    #Large loop: test globally different qnumber
    for i in range(0, 6):

        #Test if initial measure is zero; 
        #if so, record it and generate a new one
        if measure == 0:
           if not qnumber in qnumber_list:
               qnumber_list.append(copy.copy(qnumber))
               print 'Quantum number found!!! \n', qnumber
               #Generate a qnumber for new search
               qnumber.generate_global()

        #Search locally near the given qnumber (1st local loop)
        for j in range(0, 5):
            #Change locally one by one (2nd local loop)
            for part in parity_free_ids:
                qnumber_old = copy.copy(qnumber)
                qnumber.generate_local(part)
                measure_new = qnumber.conservation_measure_old(model)
                #If new qnumber is conserved, 
                #record it and break the local search
                if measure_new == 0:
                    if not qnumber in qnumber_list:
                        qnumber_list.append(copy.copy(qnumber))
                        print 'Quantum number is found!!! \n', qnumber
                        break
                #Move to new qnumber if new measure is closer to zero
                elif measure_new < measure:
                    measure = measure_new
                    #print 'Qnumber renew in (%s , %s)!' % (i, j)
                else:
                    qnumber = qnumber_old
            
            #Break the 1st local search loop if qnumber conserved.
            if measure == 0:
                break

        #No conserved qnumber is found, generate a new one.
        qnumber.generate_global()
        measure = qnumber.conservation_measure_old(model)
    
    #Record the result
    #path = os.path.join(MG5DIR, 'quantum_number', model['name'])
    #fdata = open(os.path.join(path, 'qnumber_list_parity.dat'), 'w')
    #fdata.write(str(qnumber_list))
    #fdata.close()

    return qnumber_list
