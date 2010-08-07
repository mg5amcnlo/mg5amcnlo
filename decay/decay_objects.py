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
"""Definition for the objects used in the decay module."""

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

logger = logging.getLogger('madgraph.decay_objects')

#===============================================================================
# DecayParticle
#===============================================================================
class DecayParticle(base_objects.Particle):
    """DecayParticle is the the particle used in the decay module.
       It will list all the corresponding vertices
       (2_body_decay_vertices and 3_body_decay_vertices) with
       the on-shell conditions specified.
    """
    sorted_keys = ['name', 'antiname', 'spin', 'color',
                   'charge', 'mass', 'width', 'pdg_code',
                   'texname', 'antitexname', 'line', 'propagating',
                   'is_part', 'self_antipart', 
                   '2_body_decay_vertexlist','3_body_decay_vertexlist']
    
    def default_setup(self):
        """Default values for all properties"""
        
        super(DecayParticle, self).default_setup()

        # The decay_vertexlist contain a list of real decay vertex
        # and one for pseudo decay vertex.
        # n_body_decay_vertexlist[0]=off-shell decay;
        # n_body_decay_vertexlist[1]=on-shell decay.

        self['2_body_decay_vertexlist'] =[base_objects.VertexList(),
                                          base_objects.VertexList()]
        self['3_body_decay_vertexlist'] =[base_objects.VertexList(),
                                          base_objects.VertexList()]
        
    def filter(self, name, value):
        """Filter for valid decay particle property values."""

        if name in ['2_body_decay_vertexlist', '3_body_decay_vertexlist']:
            # Must be a 2-element list of VertexList
            
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                    "Decay_vertexlist %s is not a list of vertexlist." % str(value)
            elif not len(value)==2:
                raise self.PhysicsObjectError, \
                    "Decay_vertexlist %s is not of length 2." % str(value)

            elif not isinstance(value[0], base_objects.VertexList) or \
                 not isinstance(value[1], base_objects.VertexList):
                
                raise self.PhysicsObjectError, \
                    "Element of Decay_vertexlist %s must be VertexList." % str(value)
                
           
            print 'Warning: It is advised to set the decay_vertex '+ \
                  'properties from method of Model.'
            
        super(DecayParticle, self).filter(self, name, value)

    def FindChannel(self, model):
        """Find the possible decay channel to decay,
           for both on-shell and off-shell
        """
        
        # Raise error if self is not in model.
        if not self['pdg_code'] in keys(model['particles'].generate_dict()):
            raise self.PhysicsObjectError, \
                    "The Particle is not in the model."

        # pass the find channels program

#===============================================================================
# Channel: Each channel for the decay
#===============================================================================
class Channel(base_objects.Diagram):
    """Channel: a diagram that describes a certain on-shell decay channel
                with apprximated (mean) matrix element, phase space area,
                and decay width
                ('apx_matrixelement', 'apx_PSarea', and  'apx_decaywidth')
                Model must be specified.
    """

    sorted_keys = ['vertices',
                   'model',
                   'apx_matrixelement', 'apx_PSarea', 'apx_decaywidth']

    def def_setup(self):
        """Default values for all properties"""
        
        self['vertices'] = VertexList()
        self['model'] = Model()
        self['apx_matrixelement', 'apx_PSarea', 'apx_decaywidth'] = [0., 0., 0.]

    def filter(self, name, value):
        """Filter for valid diagram property values."""
        
        if name in ['apx_matrixelement', 'apx_PSarea', 'apx_decaywidth']:
            if not isinstance(value, float):
                raise self.PhysicsObjectError, \
                    "Value %s is not a float" % str(value)
        
        if name == 'model':
            if not isinstance(value, Model):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Model object" % str(value)

        super(Channel, self0).filter(self, name, value)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return self.sorted_keys

    def nice_string(self):
        pass

    def get_initial_id(self):
        """ Return the list of the id of initial particle"""
        pass

    def get_final_ids(self):
        """ Return the list of the ids of final particles"""
        pass
        
    def get_apx_matrixelement(self):
        """calculate the apx_matrixelement"""
        pass

    def get_apx_PSarea(self):
        """calculate the apx_PSarea"""

        # The initial particle mass
        M = self['model'].get_particle(self.get_initial_id()[0])['mass']

        if len(self.get_final_ids()) == 2:
            
            m_1 = self['model'].get_particle(self.get_final_ids()[0])['mass']
            m_2 = self['model'].get_particle(self.get_final_ids()[1])['mass']

            apx_PSarea = 1 / (32 * math.pi ) * \
                         math.sqrt((M-m_1^2-m_2^2)^2-4*m_1^2*m_2^2)

        elif self.get_num_finalparticles() == 3:
            # Calculate the phase space area for 3 body decay
            m_1 = self['model'].get_particle(self.get_final_ids()[0])['mass']
            m_2 = self['model'].get_particle(self.get_final_ids()[1])['mass']
            m_3 = self['model'].get_particle(self.get_final_ids()[2])['mass']
            
            # The middle point value of the m_1, m_2 C.M. mass
            m_12_mid = (M-m_3+m_1+m_2)/2

            E_2_dag = (m_12^2-m_1^2+m_2^2)/(2*m_12)
            E_3_dag = (M-m_12^2-m_3^2)/(2*m_12)

            apx_PSarea = 4*math.sqrt((E_2_dag^2-m_2^2)*(E_3_dag^2-m_3^2)) \
                         * ((1-m_3)^2-(m_1+m_2)^2)

        else:
            # This version cannot deal with channels with more than 3 final
            # particles.

            raise self.PhysicsObjectError, \
                    "Number of final particles larger than three.\n" \
                        "Not allow in this version."


    def get_apx_decaywidth(self):
        """Calculate the apx_decaywidth"""
        
        self.apx_decaywidth = self.get_apx_matrixelment() * self.get_apx_PSarea()

#===============================================================================
# ChannelList: List of all possible  channels for the decay
#===============================================================================
class ChannelList(base_objects.DiagramList):
    """List of decay Channel
    """

    def is_valid_element(self, obj):
        """ Test if the object is a valid Channel for the list. """

        return isinstance(obj, Channel)

    def nice_string(self, indent=0):
        """Return a nicely formatted string"""

        pass

    
#===============================================================================
# DecayModel: Model object that is used in this module
#===============================================================================
class DecayModel(base_objects.Model):
    """Model object with an attribute to construct the decay vertex list
       for a given particle and a interaction
    """
    
    def FindChannel(self, interaction):
        """ Check whether the interaction is able to decay from mother_part.
            Set the '2_body_decay_vertexlist' and 
            '3_body_decay_vertexlist' of the corresponding particles.
            Utilize in finding all the decay table of the whole model
        """

        pass
