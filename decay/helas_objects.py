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

#===============================================================================
# Channel: Each channel for the decay
#===============================================================================
class Channel(base_objects.Process):
    """ParentParticle is the parent particle for the 
    """

#===============================================================================
# Channel: Each channel for the decay
#===============================================================================
class Channel(base_objects.Process):
    """ParentParticle is the parent particle for the 
    """


