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
                   '2_body_decay_vertexlist','3_body_decay_vertexlist'
                  ]

    vertexlistwritten = False

    def __init__(self, init_dict={}):
        """Creates a new particle object. If a dictionary is given, tries to 
        use it to give values to properties.
        A repeated assignment is to avoid error of inconsistent pdg_code and
        initial particle id of vertex"""

        dict.__init__(self)
        self.default_setup()

        assert isinstance(init_dict, dict), \
                            "Argument %s is not a dictionary" % repr(init_dict)


        for item in init_dict.keys():
            self.set(item, init_dict[item])

        #To avoid the pdg_code remain 0 and then induce the error when
        #set the vertexlist
        for item in init_dict.keys():
            self.set(item, init_dict[item])

    def default_setup(self):
        """Default values for all properties"""
        
        super(DecayParticle, self).default_setup()

        # The decay_vertexlist contain a list of real decay vertex
        # and one for pseudo decay vertex.
        # n_body_decay_vertexlist[0](or [False]): off-shell decay;
        # n_body_decay_vertexlist[1](or [True] ):on-shell decay.

        self['2_body_decay_vertexlist'] =[base_objects.VertexList(),
                                          base_objects.VertexList()]
        self['3_body_decay_vertexlist'] =[base_objects.VertexList(),
                                          base_objects.VertexList()]


    def check_vertexlist(self, partnum, onshell, value, model = {}):
        """Check if the all the vertex in the vertexlist satisfy the following
           conditions. If so, return true; if not, raise error messages.

           1. There is only one initial particle with the id the same as self
           2. The number of final particles is equal to partnum
           3. If model is not None, check the onshell condition
        """

        #Check if partnum is an integer.
        if not isinstance(partnum, int):
            raise self.PhysicsObjectError, \
                "Final particle number %s must be an integer." % str(partnum)

        #Check if partnum is 2 or 3.
        #If so, return the vertexlist with the on-shell condition.
        if partnum not in [2 ,3]:
            raise self.PhysicsObjectError, \
                "Final particle number %s must be 2 or 3." % str(partnum)
        
        #Check if onshell condition is Boolean number.
        if not isinstance(onshell, bool):
            raise self.PhysicsObjectError, \
                "%s must be a Boolean number" % str(onshell)
                
        #Check if the value is a vertexlist
        if not isinstance(value, base_objects.VertexList):
            raise self.PhysicsObjectError, \
                "%s must be a VertexList" % str(value)
        
        #Check if the model is a valid object.
        if not (isinstance(model, base_objects.Model) or model == {}):
            raise self.PhysicsObjectError, \
                "%s must be a Model" % str(model)

        #Determine the number of initial and final particles.
        #Check the id of initial particle as well.

        #Check if the model is given
        if not model:
            #No need to check the on-shell condition
            for vert in value:
                #Reset the number of initial/final particles,
                #initial particle id
                num_ini = 0
                num_final = 0
                ini_part_id = 0
                
                for leg in vert['legs']:
                    if leg['state']:
                        num_final += 1
                    else:
                        num_ini += 1
                        ini_part_id = leg['id']
                #Check the number of final particles is the same as partnum
                if num_final != partnum:
                    raise self.PhysicsObjectError, \
                        "The vertex is a %s-body decay, not a %s-body one."\
                        % (str(num_final), str(partnum))

                #Check if the initial particle number is one.
                if num_ini != 1:
                    raise self.PhysicsObjectError, \
                        "The initial particle number is %s, not 1."\
                        % str(num_ini)
                #Check if the initial particle has the same id as the mother.
                if ini_part_id != self['pdg_code']:
                    raise self.PhysicsObjectError, \
                        "The initial particle id is %s, not %s"\
                        %(str(ini_part_id), str(self['pdg_code']))
        
        #Model is not None, check the on-shell condition
        else:
            for vert in value:

                #Reset the number of initial/final particles,
                #initial particle id, and total and initial mass
                num_ini = 0
                num_final = 0
                ini_part_id = 0
                
                total_mass = 0
                ini_mass = 0

                for leg in vert['legs']:
                    #Calculate the total mass of all the particles
                    total_mass += model.get_particle(leg['id'])['mass']
                    
                    if leg['state']:
                        num_final += 1

                    #Record the particle with 'state' == False (incoming)
                    #Multi-initial particles vertex will be blocked.
                    else:
                        num_ini += 1
                        ini_part_id = leg['id']
                        ini_mass = model.get_particle(leg['id'])['mass']

                #Check the number of final particles is the same as partnum
                if num_final != partnum:
                    raise self.PhysicsObjectError, \
                        "The vertex is a %s -body decay, not a %s -body one."\
                        % (str(num_final), str(partnum))

                #Check if the initial particle number is one.
                if num_ini != 1:
                    raise self.PhysicsObjectError, \
                        "The initial particle number is %s, not 1." % str(num_ini)

                #Check if the initial particle has the same id as the mother.
                if ini_part_id != self['pdg_code']:
                    raise self.PhysicsObjectError, \
                        "The initial particle id is %s, not %.s"\
                        % (str(ini_part_id), str(self['pdg_code']))
                
                if (ini_mass > (total_mass - ini_mass)) != onshell:
                    raise self.PhysicsObjectError, \
                        "The on-shell condition is not satisfied."
        return True


    def filter(self, name, value):
        """Filter for valid decay particle property values."""
        
        if name in ['2_body_decay_vertexlist', '3_body_decay_vertexlist']:

            #Value must be a list of 2 elements.
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                    "Decay_vertexlist %s is not a list of vertexlist." % str(value)
            elif not len(value) == 2:
                raise self.PhysicsObjectError, \
                    "Decay_vertexlist %s must be 2-element list." % str(value)

            #Use the check_vertexlist to check
            elif name == '2_body_decay_vertexlist':
                self.check_vertexlist(2, False, value[0])
                self.check_vertexlist(2, True, value[1])
            else:
                self.check_vertexlist(3, False, value[0])
                self.check_vertexlist(3, True, value[1])

            
        super(DecayParticle, self).filter(name, value)

        return True

    def get_decay_vertexlist(self, partnum ,onshell):
        """Return the n-body decay vertexlist.
           partnum = n.
           If onshell=false, return the on-shell list and vice versa.
        """
        #Check if partnum is an integer.
        if not isinstance(partnum, int):
            raise self.PhysicsObjectError, \
                "Final particle number %s must be an integer." % str(partnum)

        #Check if partnum is 2 or 3.
        #If so, return the vertexlist with the on-shell condition.
        if partnum not in [2 ,3]:
            raise self.PhysicsObjectError, \
                "Final particle number %s must be 2 or 3." % str(partnum)
        
        #Check if onshell condition is Boolean number.
        if not isinstance(onshell, bool):
            raise self.PhysicsObjectError, \
                "%s must be a Boolean number" % str(onshell)

        return self.get(str(partnum)+'_body_decay_vertexlist')[onshell]


    def set_decay_vertexlist(self, partnum ,onshell, value, model = {}):
        """Set the n_body_decay_vertexlist,
           partnum: n, 
           onshell: True for on-shell decay, and False for off-shell
           value: the decay_vertexlist that is tried to assign.
           model: the underlying model for vertexlist
                  Use to check the correctness of on-shell condition.
        """
        #Check the vertexlist by check_vertexlist
        #Error is raised (by check_vertexlist) if value is not valid
        if self.check_vertexlist(partnum, onshell, value, model):            
            self[str(partnum)+'_body_decay_vertexlist'][onshell] = \
                copy.copy(value)

    def find_vertexlist(self, model, option=False):
        """Find the possible decay channel to decay,
           for both on-shell and off-shell.
           If option=False (default), 
           do not rewrite the VertexList if it exists.
           If option=True, rewrite the VertexList anyway.
        """
        
        #Raise error if self is not in model.
        if not self.get_pdg_code() in keys(model['particles'].generate_dict()):
            raise self.PhysicsObjectError, \
                    "The parent particle is not in the model %s." % str(model)
        
        #If 'decay_vertexlist_written' is true and option is false,
        #no action is proceed.
        if self.decay_vertexlist_written and not option:
            return 'The vertexlist has been setup. No action proceeds because of False option.'
        else:
            # pass the find channels program
            pass
        
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
