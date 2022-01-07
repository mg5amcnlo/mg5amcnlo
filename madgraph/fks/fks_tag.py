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

"""Definitions of the objects needed both for MadFKS with tagged particles"""

import madgraph.core.base_objects as MG
import madgraph
import madgraph.various.misc as misc

if madgraph.ordering:
    set = misc.OrderedSet

class MultiTagLeg(MG.MultiLeg):
    """a daughter class of MultiLeg, with the extra possibility of specifying
    whether a given leg is tagged or not, via the "is_tagged" key
    """

    def default_setup(self):
        """Default values for all properties"""
        super(MultiTagLeg, self).default_setup()
        self['is_tagged'] = False
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(MultiTagLeg, self).get_sorted_keys()
        keys += ['is_tagged']
        return keys

    
    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'is_tagged':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError( \
                        "%s is not a valid string for leg 'is_tagged' flag" \
                                                        % str(value))
        return super(MultiTagLeg,self).filter(name, value)
    
     

class TagLeg(MG.Leg):
    """a daughter class of Leg, with the extra possibility of specifying
    whether a given leg is tagged or not, via the "is_tagged" key
    """

    def default_setup(self):
        """Default values for all properties"""
        super(TagLeg, self).default_setup()
        self['is_tagged'] = False
    
    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(TagLeg, self).get_sorted_keys()
        keys += ['is_tagged']
        return keys

    
    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'is_tagged':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError( \
                        "%s is not a valid string for leg 'is_tagged' flag" \
                                                        % str(value))
        return super(TagLeg,self).filter(name, value)
