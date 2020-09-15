################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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

"""Definitions of the objects needed both for MadFKS from real 
and MadFKS from born"""

from __future__ import absolute_import
from __future__ import print_function
import logging
import madgraph.core.base_objects as MG
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra


logger = logging.getLogger('madgraph.sudakov')
    
    
    
class SudakovError(Exception):
    """Exception for the Sudakov module"""
    pass


def get_sudakov_amps(born_amp):
    """returns all the amplitudes needed to compute EW 
    corrections to born_amp in the sudakov approximation
    """
    logger.warning('get_sudakov_amps does nothing!')
