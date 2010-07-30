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
"""documentation for decay width calculator"""

import array
import copy
import logging
import itertools
import math

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color

#===============================================================================
# What does logger means...
#===============================================================================

logger = logging.getLogger('madgraph.decay_width')

#===============================================================================
# ParentParticle: will the model be specified by particle?
#===============================================================================
class ParentParticle(base_objects.Particle):
    """ParentParticle is the parent particle for the decay.
    """
    
    def default_setup(self):
        """Default values for all properties"""

        super(ParentParticle, self).default_setup()
        self['twobody_connections']
        self['twobody_decays'] = ChannelList()
        self[

#===============================================================================
# Channel: Each channel for the decay
#===============================================================================
class Channel(base_objects.Process):
    """Class of decay channel 
    """
    def get_num_finalparticles(self)
   
    def get_avg_matrixelement(self)
    
    def get_phase_space(self, ParentParticle)
        if self.get_num_finalparticles()==2:
            
        elif self.get_num_finalparticles()==3:
            
            M=self.???.get("mass")
            M=self.???.get("mass")
            M=self.???.get("mass")
        else:
            
#===============================================================================
# ChannelList: List of all possible  channels for the decay
#===============================================================================
class ChannelList(base_objects.ProcessList):
    """ParentParticle is the parent particle for the 
    """
    
#===============================================================================
# finding_channels: find all the possible channels for the ParentParticle
#===============================================================================
def finding_channels(parentparticle, num_final_particle)
    """find all the possible channels for the ParentParticle
    """
#===============================================================================
# ranking_channels: rank the decay width of all the channels
#===============================================================================
def ranking_channels(channellist, crieterion)
    """rank the approximated decay width
    """
#===============================================================================
# user_interface: for input the initial particle and output for the decay width
#===============================================================================
def input(initialparticle, model)
    """input the parent particle
    """
    parentparticle=ParentParticle._init_(initialparticle)
    parentparticle.def_model(model)

def output()
    """output the approximated decay width
    """

#===============================================================================
# main part of the program
#===============================================================================
input(initialparticle=, model=) # variables from input

finding_channels(parentparticle, 2)
finding_channels(parentparticle, 3)

# Calculation by the madgraph5
pass

ranking_channels(channellist)

output()