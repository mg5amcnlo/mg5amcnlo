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
import cmath
import copy
import itertools
import logging
import math
import os
import re

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color
import madgraph.iolibs.import_ufo as import_ufo
from madgraph import MadGraph5Error, MG5DIR

ZERO = 0
#===============================================================================
# Logger for decay_module
#===============================================================================

logger = logging.getLogger('decay')


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
                   'decay_vertexlist', 'decay_channels'
                  ]


    def __init__(self, init_dict={}):
        """Creates a new particle object. If a dictionary is given, tries to 
        use it to give values to properties.
        A repeated assignment is to avoid error of inconsistent pdg_code and
        initial particle id of vertex"""

        dict.__init__(self)
        self.default_setup()

        assert isinstance(init_dict, dict), \
                            "Argument %s is not a dictionary" % repr(init_dict)

        #To avoid the pdg_code remain 0 and then induce the error when
        #set the vertexlist
        try:
            pid = init_dict['pdg_code']
            self.set('pdg_code', pid)
        except KeyError:
            pass
            
        for item in init_dict.keys():
            self.set(item, init_dict[item])

    def default_setup(self):
        """Default values for all properties"""
        
        super(DecayParticle, self).default_setup()

        # The decay_vertexlist contain a list of real decay vertex
        # and one for pseudo decay vertex.
        # n_body_decay_vertexlist[0](or [False]): off-shell decay;
        # n_body_decay_vertexlist[1](or [True] ):on-shell decay.

        self['decay_vertexlist'] = {(2, False) : base_objects.VertexList(),
                                    (2, True)  : base_objects.VertexList(),
                                    (3, False) : base_objects.VertexList(),
                                    (3, True)  : base_objects.VertexList()}
        self['decay_channels'] = {}
        #log of the vertexlist_found history
        self.vertexlist_found = False
        self.max_vertexorder = 0

    def get(self, name):
        """ Overloading get to include vertexlist_found and max_vertexorder"""
        if name == 'vertexlist_found':
            return self.vertexlist_found
        if name == 'max_vertexorder':
            self.get_max_vertexorder()
            return self.max_vertexorder

        return super(DecayParticle, self).get(name)
    
    def check_decay_condition(self, partnum, onshell, 
                              value = base_objects.VertexList(), model = {}):
        """Check the validity of decay condition, including,
           partnum: final state particle number,
           onshell: on-shell condition,
           value  : the assign vertexlist
           model  : the specific model"""

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
                
        #Check if the value is a Vertexlist(in base_objects) or a list of vertex
        if not isinstance(value, base_objects.VertexList):
            raise self.PhysicsObjectError, \
                "%s must be VertexList type." % str(value)
                    
        #Check if the model is a valid object.
        if not (isinstance(model, base_objects.Model) or model == {}):
            raise self.PhysicsObjectError, \
                "%s must be a Model" % str(model)
        elif model:
            #Check if the mother particle is in the 'model'
            if not (self.get_pdg_code() in model.get('particle_dict').keys()):
                raise self.PhysicsObjectError, \
                    "The model, %s, does not contain particle %s." \
                    %(model.get('name'), self.get_name())

                            
    def check_vertexlist(self, partnum, onshell, value, model = {}):
        """Check if the all the vertex in the vertexlist satisfy the following
           conditions. If so, return true; if not, raise error messages.

           1. There is an appropriate leg for initial particle.
           2. The number of final particles equals to partnum.
           3. If model is not None, check the onshell condition and
              the initial particle id is the same as calling particle.
        """
        #Check the validity of arguments first
        self.check_decay_condition(partnum, onshell, value, model)
       
        #Determine the number of final particles.
        #Find all the possible initial particle(s).
        #Check onshell condition if the model is given.
        if model:
            if (self.get('mass') == 'ZERO') and (len(value) != 0):
                raise self.PhysicsObjectError, \
                    "Massless particle %s cannot decay." % self['name']

        for vert in value:
            # Reset the number of initial/final particles,
            # initial particle id, and total and initial mass
            num_ini = 0
            radiation = False
            num_final = 0
                
            if model:
                # Calculate the total mass
                total_mass = sum([eval(model.get_particle(l['id']).get('mass')).real for l in vert['legs']])
                ini_mass = eval(self.get('mass')).real
                
                # Check the onshell condition
                if (ini_mass.real > (total_mass.real - ini_mass.real))!=onshell:
                    raise self.PhysicsObjectError, \
                        "The on-shell condition is not satisfied."

            for leg in vert.get('legs'):
                # Check if all legs are label by true
                if not leg.get('state'):
                    raise self.PhysicsObjectError, \
                        "The state of leg should all be true"

                # Identify the initial particle
                if leg.get('id') == self.get_pdg_code():
                    # Double anti particle is also radiation
                    if num_ini == 1:
                        radiation = True
                    num_ini = 1
                elif leg.get('id') == self.get_anti_pdg_code() and \
                        not self.get('self_antipart'):
                    radiation = True            

            # Calculate the final particle number
            num_final = len(vert.get('legs'))-num_ini

            # Check the number of final particles is the same as partnum
            if num_final != partnum:
                raise self.PhysicsObjectError, \
                    "The vertex is a %s -body decay, not a %s -body one."\
                    % (str(num_final), str(partnum))

            # Check if there is any appropriate leg as initial particle.
            if num_ini == 0:
                raise self.PhysicsObjectError, \
                    "There is no leg satisfied the mother particle %s"\
                    % str(self.get_pdg_code())

            # Check if the vertex is radiation
            if radiation:
                raise self.PhysicsObjectError, \
                    "The vertex is radiactive for mother particle %s"\
                    % str(self.get_pdg_code())

        return True

    def check_channels(self, partnum, onshell, value = [], model = {}):
        """Check the validity of decay channel condition, including,
           partnum: final state particle number,
           onshell: on-shell condition,
           value  : the assign channel list, all channels in it must
                    be consistent to the given partnum and onshell.
           model  : the specific model."""

        # Check if partnum is an integer.
        if not isinstance(partnum, int):
            raise self.PhysicsObjectError, \
                "Final particle number %s must be an integer." % str(partnum)
        
        # Check if onshell condition is Boolean number.
        if not isinstance(onshell, bool):
            raise self.PhysicsObjectError, \
                "%s must be a Boolean number" % str(onshell)
                
        # Check if the value is a ChannelList
        if (not isinstance(value, ChannelList) and value):
            raise self.PhysicsObjectError, \
                "%s must be ChannelList type." % str(value)
                

        # Check if the partnum is correct for all channels in value
        if any(ch for ch in value if \
                   len(ch.get_final_legs()) != partnum):
            raise self.PhysicsObjectError, \
                "The final particle number of channel should be %d."\
                % partnum
        
        # Check if the initial particle in all channels are as self.
        if any(ch for ch in value if \
                   abs(ch.get_initial_id()) != abs(self.get('pdg_code'))):
            raise self.PhysicsObjectError, \
                "The initial particle is not %d or its antipart." \
                % self.get('pdg_code')

        # Check if the onshell condition is right
        if not (isinstance(model, base_objects.Model) or model == {}):
            raise self.PhysicsObjectError, \
                "%s must be a Model" % str(model)
        elif model:
            # Check if the mother particle is in the 'model'
            if not (self.get_pdg_code() in model.get('particle_dict').keys()):
                raise self.PhysicsObjectError, \
                    "The model, %s, does not contain particle %s." \
                    %(model.get('name'), self.get_name())
            if any([ch for ch in value if onshell != ch.get_onshell(model)]):
                raise self.PhysicsObjectError, \
                    "The onshell condition is not consistent with the model."

    def filter(self, name, value):
        """Filter for valid DecayParticle vertexlist."""
        
        if name == 'decay_vertexlist' or name == 'decay_channels':

            #Value must be a dictionary.
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Decay_vertexlist or decay_channels %s must be a dictionary." % str(value)

            # key must be two element tuple
            for key, item in value.items():
                if not isinstance(key, tuple):
                    raise self.PhysicsObjectError,\
                        "Key %s must be a tuple." % str(key)
                
                if len(key) != 2:
                    raise self.PhysicsObjectError,\
                        "Key %s must have two elements." % str(key)
                
                if name == 'decay_vertexlist':
                    self.check_vertexlist(key[0], key[1], item)
                if name == 'decay_channels':
                    self.check_channels(key[0], key[1], item)          

        super(DecayParticle, self).filter(name, value)

        return True

    def get_vertexlist(self, partnum ,onshell):
        """Return the n-body decay vertexlist.
           partnum = n.
           If onshell=false, return the on-shell list and vice versa.
        """
        #check the validity of arguments
        self.check_decay_condition(partnum, onshell)
        
        return self.get('decay_vertexlist')[(partnum, onshell)]

    def set_vertexlist(self, partnum ,onshell, value, model = {}):
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
            self['decay_vertexlist'][(partnum, onshell)] = value

    def get_max_vertexorder(self):
        """ Get the max vertex order of this particle"""
        # Do not include keys without vertexlist in it
        # Both onshell and offshell are consider
        if not self.vertexlist_found:
            print "The vertexlist of this particle has not been searched. ",\
                "Try find_vertexlist first."

        vertnum_list = [k[0] for k in self['decay_vertexlist'].keys() \
             if self['decay_vertexlist'][k]]
        if vertnum_list:
            self.max_vertexorder = max(vertnum_list)
        else:
            self.max_vertexorder = 0

        return self.max_vertexorder

    def find_vertexlist(self, model, option=False):
        """Find the possible decay channel to decay,
           for both on-shell and off-shell.
           If option=False (default), 
           do not rewrite the VertexList if it exists.
           If option=True, rewrite the VertexList anyway.
        """
        
        #Raise error if self is not in model.
        if not (self.get_pdg_code() in model.get('particle_dict').keys()):
            raise self.PhysicsObjectError, \
                    "The parent particle %s is not in the model %s." \
                        % (self.get('name'), model.get('name'))

        #Raise error if option is not Boolean value
        if not isinstance(option, bool):
            raise self.PhysicsObjectError, \
                    "The option %s must be True or False." % str(option)
        
        #If 'vertexlist_found' is true and option is false,
        #no action is proceed.
        if self.vertexlist_found and not option:
            return 'The vertexlist has been setup.', \
                'No action proceeds because of False option.'

        # Reset the decay vertex before finding
        self['decay_vertexlist'] = {(2, False) : base_objects.VertexList(),
                                    (2, True)  : base_objects.VertexList(),
                                    (3, False) : base_objects.VertexList(),
                                    (3, True)  : base_objects.VertexList()}
        
        # Do not include the massless and stable particle
        if self.get('mass') == 'ZERO' or self in model.stable_particles:
            return

        #Go through each interaction...
        for temp_int in model.get('interactions'):
            #Save the particle dictionary (pdg_code & anti_pdg_code to particle)
            partlist = temp_int.get('particles')

            #The final particle number = total particle -1
            partnum = len(partlist)-1
            #Allow only 2 and 3 body decay
            if partnum > 3:
                continue

            #Check if the interaction contains mother particle
            if model.get_particle(self.get_anti_pdg_code()) in partlist:
                #Exclude radiation
                part_id_list = [p.get('pdg_code') for p in partlist]
                if (part_id_list.count(self.get('pdg_code'))) > 1:
                    continue

                total_mass = 0
                ini_mass = eval(self.get('mass')).real
                vert = base_objects.Vertex()
                legs = base_objects.LegList()

                # Setup all the legs and find final_mass
                for part in partlist:
                    legs.append(base_objects.Leg({'id': part.get_pdg_code()}))
                    total_mass += eval(part.get('mass')).real
                    #Initial particle has not been found: ini_found = True
                    if (part == model.get_particle(self.get_anti_pdg_code())):
                        ini_leg = legs.pop()
                        ini_leg.set('id', self.get_pdg_code())
                    
                #Sort the outgoing leglist for comparison sake (removable!)
                legs.sort(legcmp)
                # Append the initial leg
                legs.append(ini_leg)

                vert.set('id', temp_int.get('id'))
                vert.set('legs', legs)
                temp_vertlist = base_objects.VertexList([vert])

                #Check validity of vertex (removable)
                """self.check_vertexlist(partnum,
                ini_mass > final_mass,
                temp_vertlist, model)"""

                #Append current vert to vertexlist
                try:
                    self['decay_vertexlist'][(partnum, \
                                            ini_mass > (total_mass-ini_mass))].\
                                            append(vert)
                except KeyError:
                    self['decay_vertexlist'][(partnum, \
                                            ini_mass > (total_mass-ini_mass))] \
                                            = vert

        #Set the vertexlist_found at the end
        self.vertexlist_found = True
        

    def get_channels(self, partnum ,onshell):
        """Return the n-body decay channels.
           partnum = n.
           If onshell=false, return the on-shell list and vice versa.
        """
        #check the validity of arguments
        self.check_channels(partnum, onshell)
        return self.get('decay_channels')[(partnum, onshell)]

    def set_channels(self, partnum ,onshell, value, model = {}):
        """Set the n_body_decay_vertexlist, value is overloading to both
           ChannelList and list of Channels (auto-transformation will proceed)
           partnum: n, 
           onshell: True for on-shell decay, and False for off-shell
           value: the decay_vertexlist that is tried to assign.
           model: the underlying model for vertexlist
                  Use to check the correctness of on-shell condition.
        """
        #Check the vertexlist by check_vertexlist
        #Error is raised (by check_vertexlist) if value is not valid
        if isinstance(value, ChannelList):
            if self.check_channels(partnum, onshell, value, model):
                self['decay_channels'][(partnum, onshell)] = value
        elif isinstance(value, list) and \
                all([isinstance(c, Channel) for c in value]):
            value_transform = ChannelList(value)
            if self.check_channels(partnum, onshell, value_transform, model):
                self['decay_channels'][(partnum, onshell)] = value_transform
        else:
            raise self.PhysicsObjectError, \
                "The input must be a list of diagrams."
        
              
    def find_channels(self, partnum, model):
        """ Function for finding decay channels up to the final particle
            number given by max_partnum.
            Algorithm:
            1. Any channel must be a. only one vertex, b. an existing channel
               plus one vertex.
            2. Given the maxima channel order, the program start looking for
               2-body decay channels until the channels with the given order.
            3. For each channel order,
               a. First looking for any single vertex with such order and 
                  construct the channel. Setup the has_idpart property.
               b. Then finding the channel from off-shell sub-level channels.
                  Note that any further decay of on-shell sub-level channel
                  is not a new channel. For each sub-level channel, go through
                  the final legs and connect vertex with the right order
                  (with the aid of connect_channel_vertex function).
               c. If the new channel does not exist in 'decay_channels',
                  then if the new channel has no identical particle, append it.
                  If the new channel has identical particle, check if it is not
                  equvialent with the existing channels. Then append it.
         """

        # Check validity of argument
        if not isinstance(partnum, int):
            raise self.PhysicsObjectError, \
                "Max final particle number %s should be integer." % str(partnum)
        if not isinstance(model, DecayModel):
            raise self.PhysicsObjectError, \
                "The second argument %s should be a DecayModel object." \
                % str(model)            
        if not self in model['particles']:
            raise self.PhysicsObjectError, \
                "The model %s does not contain particle %s" \
                % (model.get('name'), self.get('name'))

        # If vertexlist has not been found before, run model.find_vertexlist
        if not model.vertexlist_found:
            print "Vertexlist of this model has not been searched.",\
                "Automatically run the model.find_vertexlist()"
            model.find_vertexlist()

        # If the channel list exist, return.
        if (partnum, True) in self['decay_channels'].keys() or \
                (partnum, False) in self['decay_channels'].keys():
                return
            
        # The initial vertex (identical vertex).
        ini_vert = base_objects.Vertex({'id': 0, 'legs': base_objects.LegList([\
                   base_objects.Leg({'id':self.get_anti_pdg_code(), 
                                     'number':1, 'state': False}),
                   base_objects.Leg({'id':self.get_pdg_code(), 'number':2})])})

        # Find channels from 2-body decay to partnum-body decay channels.
        for clevel in range(2, partnum+1):
            # Initialize the item in dictionary
            self['decay_channels'][(clevel, True)] = ChannelList()
            self['decay_channels'][(clevel, False)] = ChannelList()

            # If there is a vertex in clevel, construct it
            if (clevel, True) in self['decay_vertexlist'].keys() or \
                    (clevel, False) in self['decay_vertexlist'].keys():
                for vert in (self.get_vertexlist(clevel, True) + \
                             self.get_vertexlist(clevel, False)):
                    temp_channel = Channel()
                    temp_vert = copy.deepcopy(vert)
                    # Set the leg number (starting from 2)
                    [l.set('number', i+2) \
                         for i, l in enumerate(temp_vert.get('legs'))]
                    # The final one has 'number' as 2
                    temp_vert.get('legs')[-1]['number'] = 2
                    # Append the true vertex and then the ini_vert
                    temp_channel['vertices'].append(temp_vert)
                    temp_channel['vertices'].append(ini_vert)
                    # Setup the 'has_idpart' property
                    if self.check_idlegs(temp_vert):
                        temp_channel['has_idpart'] = True
                    self.get_channels(clevel, temp_channel.get_onshell(model)).\
                        append(temp_channel)

            # Go through sub-channels and try to add vertex to reach partnum
            for sub_clevel in range(max((clevel - model.get_max_vertexorder()+1),2), clevel):
                # The vertex level that should be combine with sub_clevel
                vlevel = clevel - sub_clevel+1
                # Go through each 'off-shell' channel in the given sub_clevel.
                # Decay of on-shell channel is not a new channel.
                for sub_c in self.get_channels(sub_clevel, False):
                    # Scan each leg to see if there is any appropriate vertex
                    for index, leg in enumerate(sub_c.get_final_legs()):
                        # Get the particle even for anti-particle leg.
                        inter_part = model.get_particle(abs(leg['id']))
                        # Get the vertexlist in vlevel
                        # Both on-shell and off-shell vertex 
                        # should be considered.
                        try:
                            vlist_a = inter_part.get_vertexlist(vlevel, True)
                        except KeyError:
                            vlist_a = []
                        try:
                            vlist_b = inter_part.get_vertexlist(vlevel, False)
                        except KeyError:
                            vlist_b = []

                        # Find appropriate vertex
                        for vert in (vlist_a + vlist_b):
                            # Connect sub_channel to the vertex
                            # the connect_channel_vertex will
                            # inherit the 'has_idpart' from sub_c
                            temp_c = self.connect_channel_vertex(sub_c, index, 
                                                                 vert, model)
                            temp_c_o = temp_c.get_onshell(model)
                            # Append this channel if not exist
                            if not temp_c in self.get_channels(clevel,temp_c_o):
                                # If temp_c has identical particles in it,
                                # check with other existing channels from
                                # duplication.
                                if temp_c.get('has_idpart'):
                                    if not self.check_repeat(clevel,
                                                             temp_c_o, temp_c):
                                        self.get_channels(clevel, temp_c_o).\
                                            append(temp_c)
                                # If no id. particles, append this channel.
                                else:
                                    self.get_channels(clevel, temp_c_o).\
                                        append(temp_c)


    def connect_channel_vertex(self, sub_channel, index, vertex, model):
        """ Helper function to connect a vertex to one of the legs 
            in the channel. The argument 'index' specified the position of
            the leg which will be connected with vertex.
            If leg is for anti-particle, the vertex will be transform into 
            anti-part with minus vertex id."""

        # Copy the vertex to prevent the change of leg number
        new_vertex = copy.deepcopy(vertex)

        # Setup the final leg number that is used in the channel.
        leg_num = max([l['number'] for l in sub_channel.get_final_legs()])

        # Add minus sign to the vertex id and leg id if the leg in sub_channel
        # is for anti-particle
        is_anti_leg = sub_channel.get_final_legs()[index]['id'] < 0
        if is_anti_leg:
            new_vertex['id'] = -new_vertex['id']

        # Legs continue the number of the final one in sub_channel,
        # except the first and the last one ( these two should be connected.)
        for leg in new_vertex['legs']:
            # For the first and final legs, this loop is useful
            # only for reverse leg id in case of mother leg is anti-particle.
            # leg_num is correct only after the second one.
            leg['number'] = leg_num
            leg_num += 1
            if is_anti_leg:
                leg['id']  = model.get_particle(leg['id']).get_anti_pdg_code()

        # Assign correct number to each leg in the vertex.
        # The first leg follows the mother leg.
        new_vertex['legs'][-1]['number'] = \
            sub_channel.get_final_legs()[index]['number']
        new_vertex['legs'][0]['number'] = new_vertex['legs'][-1]['number']

        new_channel = Channel()
        # New vertex is first
        new_channel['vertices'].append(new_vertex)
        # Then extend the vertices of the old channel
        new_channel['vertices'].extend(sub_channel['vertices'])
        # Setup properties of new_channel
        new_channel.get_onshell(model)
        # Descendent of a channel with identicle particles has the same
        # attribute. (but 'idpart_list' will change and should not be inherited)
        new_channel['has_idpart'] = sub_channel['has_idpart']

        return new_channel

    def check_idlegs(self, vert):
        """ Helper function to check if the vertex has several identical legs.
            Return True if id_legs exist, otherwise False.
        """
        pid_list = []
        for l in vert['legs']:
            if l['id'] in pid_list:
                return True
            else:
                pid_list.append(l['id'])

        return False

    def check_repeat(self, clevel, onshell, channel):
        """ Helper function to check if any channel is indeed identical
            the given channel. (This mat happens when identical particle in
            channel)"""
        pass
        """# Return if the channel has no identical particle.
        if not channel.get('has_idpart'):
            return False
        
        # Get the identical particle info and final state of channel
        final_pid_set = set([l.get('id') for l in channel.get_final_legs()])
        id_partlist = channel.get_idpartlist()
        for id_part in id_partlist:
            

        for other_c in self.get_channels(clevel, onshell):
            # Continue if this channel has no identical particle.
            if not other_c.get('has_idpart'):
                continue
            final_pid_set_o = set([l.get('id') \
                                       for l in other_c.get_final_legs()])

            # Continue if the two channels have different final state
            if final_pid_set_o != final_pid_set:
                continue

            id_partlist_o = other_c.get_idpartlist()"""
            
    # This helper function is useless in current algorithm...
    def generate_configlist(self, channel, partnum, model):
        """ Helper function to generate all the configuration to add
            vertetices to channel to create a channel with final 
            particle number as partnum"""

        current_num = len(channel.get_final_legs())
        configlist = []
        limit_list = [model.get_particle(abs(l.get('id'))).get_max_vertexorder() for l in channel.get_final_legs()]

        for leg_position in range(current_num):
            if limit_list[leg_position] >= 2:
                ini_config = [0]* current_num
                ini_config[leg_position] = 1
                configlist.append(ini_config)

        for i in range(partnum - current_num -1):
            new_configlist = []
            # Add particle one by one to each config
            for config in configlist:
                # Add particle to each leg if it does not exceed limit_list
                for leg_position in range(current_num):
                    # the current config + new particle*1 + mother 
                    # <= max_vertexorder
                    if config[leg_position] + 2 <= limit_list[leg_position]:
                        temp_config = copy.deepcopy(config)
                        temp_config[leg_position] += 1
                        if not temp_config in new_configlist:
                            new_configlist.append(temp_config)
            if not new_configlist:
                break
            else:
                configlist = new_configlist

        # Change the format consistent with max_vertexorder
        configlist = [[l+1 for l in config] for config in configlist]
        return configlist
                                            
                    

# Helper function
def legcmp(x, y):
    """Define the leg comparison, useful when testEqual is execute"""
    mycmp = cmp(x['id'], y['id'])
    if mycmp == 0:
        mycmp = cmp(x['state'], y['state'])
    return mycmp

#===============================================================================
# DecayParticleList
#===============================================================================
class DecayParticleList(base_objects.ParticleList):
    """A class to store list of DecayParticle, Particle is also a valid
       element, but will automatically convert to DecayParticle"""

    def append(self, object):
        """Append DecayParticle, even if object is Particle"""

        assert self.is_valid_element(object), \
            "Object %s is not a valid object for the current list" %repr(object)

        if isinstance(object, DecayParticle):
            list.append(self, object)
        else:
            list.append(self, DecayParticle(object))

    def generate_dict(self):
        """Generate a dictionary from particle id to particle.
        Include antiparticles.
        """

        particle_dict = {}

        for particle in self:
            particle_dict[particle.get('pdg_code')] = particle
            if not particle.get('self_antipart'):
                antipart = copy.deepcopy(particle)
                antipart.set('is_part', False)
                particle_dict[antipart.get_pdg_code()] = antipart

        return particle_dict
    
#===============================================================================
# DecayModel: Model object that is used in this module
#===============================================================================
class DecayModel(base_objects.Model):
    """Model object with an attribute to construct the decay vertex list
       for a given particle and a interaction
    """

    def __init__(self, ini_dict = {}):
        """Reset the particle_dict so that items in it is 
           of DecayParitcle type"""
        super(DecayModel, self).__init__(ini_dict)

        self['particle_dict'] = {}
        self.get('particle_dict')
        
    def default_setup(self):
        """The particles is changed to ParticleList"""
        super(DecayModel, self).default_setup()
        self['particles'] = DecayParticleList()
        # Other properties
        self.vertexlist_found = False
        self.max_vertexorder = 0
        self.decay_groups = []
        self.reduced_interactions = []
        self.stable_particles = base_objects.ParticleList()

    def get(self, name):
        """ Evaluate some special properties first if the user request. """

        if name == 'stable_particles' and not self.stable_particles:
            self.find_stable_particles()
            return self.stable_particles
        # reduced_interactions might be empty, cannot judge the evaluation is
        # done or not by it.
        elif (name == 'decay_groups' or name == 'reduced_interactions') and \
                not self.decay_groups:
            self.find_decay_groups_general()
            return eval('self.' + name)
        elif name == 'max_vertexorder' and self.max_vertexorder == 0:
            self.get_max_vertexorder()
            return self.max_vertexorder
        else:       
            # call the mother routine
            return DecayModel.__bases__[0].get(self, name)
        

    def set(self, name, value):
        """Change the Particle into DecayParticle"""
        #Record the validity of set by mother routine
        return_value = super(DecayModel, self).set(name, value)
        #Reset the dictionaries

        if return_value:
            if name == 'particles':
                #Reset dictionaries
                self['particle_dict'] = {}
                self['got_majoranas'] = None
                #Convert to DecayParticleList
                self['particles'] = DecayParticleList(value)
                #Generate new dictionaries with items are DecayParticle
                self.get('particle_dict')
                self.get('got_majoranas')
            if name == 'interactions':
                # Reset dictionaries
                self['interaction_dict'] = {}
                self['ref_dict_to1'] = {}
                self['ref_dict_to0'] = {}
                #Generate interactions with particles are DecayParticleLis
                for inter in self['interactions']:
                    inter['particles']=DecayParticleList([part for part in \
                                                          inter['particles']])
                # Generate new dictionaries
                self.get('interaction_dict')
                self.get('ref_dict_to1')
                self.get('ref_dict_to0')

        return return_value

    def get_max_vertexorder(self):
        """find the maxima vertex order (i.e. decay particle number)"""
        if not self.vertexlist_found:
            print "Use find_vertexlist before get_max_vertexorder!"
            return

        # Do not include key without any vertexlist in it
        self.max_vertexorder = max(sum([[k[0] \
                                         for k in \
                                         p.get('decay_vertexlist').keys() \
                                         if p.get('decay_vertexlist')[k]] \
                                     for p in self.get('particles')], []))
        return self.max_vertexorder
            
    def find_vertexlist(self):
        """ Check whether the interaction is able to decay from mother_part.
            Set the '2_body_decay_vertexlist' and 
            '3_body_decay_vertexlist' of the corresponding particles.
            Utilize in finding all the decay table of the whole model
        """

        # Return if self.vertexlist_found is True.\
        if self.vertexlist_found:
            print "Vertexlist has been searched before."
            return

        ini_list = []
        #Dict to store all the vertexlist (for conveniece only, removable!)
        vertexlist_dict = {}
        for part in self.get('particles'):
            if part['mass'] != 'ZERO':
                #All valid initial particles (mass != 0 and is_part == True)
                ini_list.append(part.get_pdg_code())
            for partnum in [2, 3]:
                for onshell in [False, True]:
                    vertexlist_dict[(part.get_pdg_code(), partnum, onshell)] = \
                        base_objects.VertexList()

        #Prepare the vertexlist
        for inter in self['interactions']:
            #Calculate the particle number and exclude partnum > 3
            partnum = len(inter['particles']) - 1
            if partnum > 3:
                continue
            
            temp_legs = base_objects.LegList()
            total_mass = 0
            validity = False
            for num, part in enumerate(inter['particles']):
                #Check if the interaction contains valid initial particle
                if part.get_anti_pdg_code() in ini_list:
                    validity = True

                #Create the original legs
                temp_legs.append(base_objects.Leg({'id':part.get_pdg_code()}))
                total_mass += eval(part.get('mass')).real
            
            #Exclude interaction without valid initial particle
            if not validity:
                continue

            for num, part in enumerate(inter['particles']):
                #Get anti_pdg_code (pid for incoming particle)
                pid = part.get_anti_pdg_code()
                #Exclude invalid initial particle
                if not pid in ini_list:
                    continue

                #Exclude initial particle appears in final particles
                #i.e. radiation is excluded.
                #count the number of abs(pdg_code)
                pid_list = [p.get('pdg_code') for p in inter.get('particles')]
                if pid_list.count(abs(pid)) > 1:
                    continue

                ini_mass = eval(part.get('mass')).real
                onshell = ini_mass > (total_mass - ini_mass)

                #Create new legs for the sort later
                temp_legs_new = copy.deepcopy(temp_legs)
                temp_legs_new[num]['id'] = pid

                # Put initial leg in the last 
                # and sort other legs for comparison
                inileg = temp_legs_new.pop(num)
                temp_legs_new.sort(legcmp)
                temp_legs_new.append(inileg)
                temp_vertex = base_objects.Vertex({'id': inter.get('id'),
                                                   'legs':temp_legs_new})

                #Record the vertex with key = (interaction_id, part_id)
                if temp_vertex not in \
                        self.get_particle(pid).get_vertexlist(partnum, onshell):
                    vertexlist_dict[(pid, partnum, onshell)].append(temp_vertex)
                    #Assign temp_vertex to antiparticle of part
                    #particle_dict[pid].check_vertexlist(partnum, onshell, 
                    #             base_objects.VertexList([temp_vertex]), self)
                    self.get_particle(pid)['decay_vertexlist'][(\
                            partnum, onshell)].append(temp_vertex)

        # Set the property vertexlist_found as True and for all particles
        self.vertexlist_found = True
        for part in self['particles']:
            part.vertexlist_found = True

        #fdata = open(os.path.join(MG5DIR, 'models', self['name'], 'vertexlist_dict.dat'), 'w')
        #fdata.write(str(vertexlist_dict))
        #fdata.close()



    def read_param_card(self, param_card):
        """Read a param_card and set all parameters and couplings as
        members of this module"""

        if not os.path.isfile(param_card):
            raise MadGraph5Error, \
                  "No such file %s" % param_card

        # Extract external parameters
        external_parameters = self['parameters'][('external',)]

        # Create a dictionary from LHA block name and code to parameter name
        parameter_dict = {}
        for param in external_parameters:
            try:
                dict = parameter_dict[param.lhablock.lower()]
            except KeyError:
                dict = {}
                parameter_dict[param.lhablock.lower()] = dict
            dict[tuple(param.lhacode)] = param.name
            
        # Now read parameters from the param_card

        # Read in param_card
        param_lines = open(param_card, 'r').read().split('\n')

        # Define regular expressions
        re_block = re.compile("^block\s+(?P<name>\w+)")
        re_decay = re.compile("^decay\s+(?P<pid>\d+)\s+(?P<value>[\d\.e\+-]+)")
        re_single_index = re.compile("^\s*(?P<i1>\d+)\s+(?P<value>[\d\.e\+-]+)")
        re_double_index = re.compile(\
                       "^\s*(?P<i1>\d+)\s+(?P<i2>\d+)\s+(?P<value>[\d\.e\+-]+)")
        block = ""
        # Go through lines in param_card
        for line in param_lines:
            if not line.strip() or line[0] == '#':
                continue
            line = line.lower()
            # Look for blocks
            block_match = re_block.match(line)
            if block_match:
                block = block_match.group('name')
                continue
            # Look for single indices
            single_index_match = re_single_index.match(line)
            if block and single_index_match:
                i1 = int(single_index_match.group('i1'))
                value = single_index_match.group('value')
                try:
                    exec("globals()[\'%s\'] = %s" % (parameter_dict[block][(i1,)],
                                      value))
                    logger.info("Set parameter %s = %f" % \
                                (parameter_dict[block][(i1,)],\
                                 eval(parameter_dict[block][(i1,)])))
                except KeyError:
                    logger.warning('No parameter found for block %s index %d' %\
                                   (block, i1))
                continue
            double_index_match = re_double_index.match(line)
            # Look for double indices
            if block and double_index_match:
                i1 = int(double_index_match.group('i1'))
                i2 = int(double_index_match.group('i2'))
                try:
                    exec("globals()[\'%s\'] = %s" % (parameter_dict[block][(i1,i2)],
                                      double_index_match.group('value')))
                    logger.info("Set parameter %s = %f" % \
                                (parameter_dict[block][(i1,i2)],\
                                 eval(parameter_dict[block][(i1,i2)])))
                except KeyError:
                    logger.warning('No parameter found for block %s index %d %d' %\
                                   (block, i1, i2))
                continue
            # Look for decays
            decay_match = re_decay.match(line)
            if decay_match:
                block = ""
                pid = int(decay_match.group('pid'))
                value = decay_match.group('value')
                try:
                    exec("globals()[\'%s\'] = %s" % \
                         (parameter_dict['decay'][(pid,)],
                          value))
                    logger.info("Set decay width %s = %f" % \
                                (parameter_dict['decay'][(pid,)],\
                                 eval(parameter_dict['decay'][(pid,)])))
                except KeyError:
                    logger.warning('No decay parameter found for %d' % pid)
                continue

        # Define all functions used
        for func in self['functions']:
            exec("def %s(%s):\n   return %s" % (func.name,
                                                ",".join(func.arguments),
                                                func.expr))

        # Extract derived parameters
        # TO BE IMPLEMENTED allow running alpha_s coupling
        derived_parameters = []
        try:
            derived_parameters += self['parameters'][()]
        except KeyError:
            pass
        try:
            derived_parameters += self['parameters'][('aEWM1',)]
        except KeyError:
            pass
        try:
            derived_parameters += self['parameters'][('aS',)]
        except KeyError:
            pass
        try:
            derived_parameters += self['parameters'][('aS', 'aEWM1')]
        except KeyError:
            pass
        try:
            derived_parameters += self['parameters'][('aEWM1', 'aS')]
        except KeyError:
            pass

        # Now calculate derived parameters
        # TO BE IMPLEMENTED use running alpha_s for aS-dependent params
        for param in derived_parameters:
            exec("globals()[\'%s\'] = %s" % (param.name, param.expr))
            if not eval(param.name) and eval(param.name) != 0:
                logger.warning("%s has no expression: %s" % (param.name,
                                                             param.expr))
            try:
                logger.info("Calculated parameter %s = %f" % \
                            (param.name, eval(param.name)))
            except TypeError:
                logger.info("Calculated parameter %s = (%f, %f)" % \
                            (param.name,\
                             eval(param.name).real, eval(param.name).imag))
        
        # Extract couplings
        couplings = []
        try:
            couplings += self['couplings'][()]
        except KeyError:
            pass
        try:
            couplings += self['couplings'][('aEWM1',)]
        except KeyError:
            pass
        try:
            couplings += self['couplings'][('aS',)]
        except KeyError:
            pass
        try:
            couplings += self['couplings'][('aS', 'aEWM1')]
        except KeyError:
            pass
        try:
            couplings += self['couplings'][('aEWM1', 'aS')]
        except KeyError:
            pass

        # Now calculate all couplings
        # TO BE IMPLEMENTED use running alpha_s for aS-dependent couplings
        for coup in couplings:
            exec("globals()[\'%s\'] = %s" % (coup.name, coup.expr))
            if not eval(coup.name) and eval(coup.name) != 0:
                logger.warning("%s has no expression: %s" % (coup.name,
                                                             coup.expr))
            logger.info("Calculated coupling %s = (%f, %f)" % \
                        (coup.name,\
                         eval(coup.name).real, eval(coup.name).imag))
                


    def find_decay_groups(self):
        """Find groups of particles which can decay into each other,
        keeping Standard Model particles outside for now. This allows
        to find particles which are absolutely stable based on their
        interactions.

        Algorithm:

        1. Start with any non-SM particle. Look for all
        interactions which has this particle in them.

        2. Any particles with single-particle interactions with this
        particle and with any number of SM particles are in the same
        decay group.

        3. If any of these particles have decay to only SM
        particles, the complete decay group becomes "sm"
        
        5. Iterate through all particles, to cover all particles and
        interactions.
        """

        self.sm_ids = [1,2,3,4,5,6,11,12,13,14,15,16,21,22,23,24]
        self.decay_groups = [[]]

        particles = [p for p in self.get('particles') if \
                     p.get('pdg_code') not in self.sm_ids]

        for particle in particles:
            # Check if particles is already in a decay group
            if particle not in sum(self.decay_groups, []):
                # Insert particle in new decay group
                self.decay_groups.append([particle])
                self.find_decay_groups_for_particle(particle)

    def find_decay_groups_for_particle(self, particle):
        """Recursive routine to find decay groups starting from a
        given particle.

        Algorithm:

        1. Pick out all interactions with this particle

        2. For any interaction which is not a radiation (i.e., has
        this particle twice): 

        a. If there is a single non-sm particle in
        the decay, add particle to this decay group. Otherwise, add to
        SM decay group or new decay group.

        b. If there are more than 1 non-sm particles: if all particles
        in decay groups, merge decay groups according to different
        cases:
        2 non-sm particles: either both are in this group, which means
        this is SM, or one is in this group, so the other has to be
        SM, or both are in the same decay group, then this group is SM.
        3 non-sm particles: either 1 is in this group, then the other
        two must be in same group or 2 is in this group, then third
        must also be in this group, or 2 is in the same group, then
        third must be in this group (not yet implemented). No other
        cases can be dealt with.
        4 or more: Not implemented (not phenomenologically interesting)."""
        
        # interactions with this particle which are not radiation
        interactions = [i for i in self.get('interactions') if \
                            particle in i.get('particles') and \
                            i.get('particles').count(particle) == 1 and \
                            (particle.get('self_antipart') or
                             not self.get_particle(particle.get_anti_pdg_code())\
                                 in i.get('particles'))]
                             
        while interactions:
            interaction = interactions.pop(0)
            non_sm_particles = [p for p in interaction.get('particles') \
                                if p != particle and \
                                not p.get('pdg_code') in self.sm_ids and \
                                not (p.get('is_part') and p in \
                                     self.decay_groups[0] or \
                                     not p.get('is_part') and \
                                     self.get_particle(p.get('pdg_code')) in \
                                     self.decay_groups[0])]
            group_index = [i for (i, g) in enumerate(self.decay_groups) \
                           if particle in g][0]

            if len(non_sm_particles) == 0:
                # The decay group of this particle is the SM group
                if group_index > 0:
                    group = self.decay_groups.pop(group_index)
                    self.decay_groups[0].extend(group)
                    
            elif len(non_sm_particles) == 1:
                # The other particle should be in my decay group
                particle2 = non_sm_particles[0]
                if not particle2.get('is_part'):
                    particle2 = self.get_particle(particle2.get_anti_pdg_code())
                if particle2 in self.decay_groups[group_index]:
                    # This particle is already in this decay group,
                    # and has been treated.
                    continue
                elif particle2 in sum(self.decay_groups, []):
                    # This particle is in a different decay group - merge
                    group_index2 = [i for (i, g) in \
                                    enumerate(self.decay_groups) \
                                    if particle2 in g][0]
                    group = self.decay_groups.pop(max(group_index,
                                                      group_index2))
                    self.decay_groups[min(group_index, group_index2)].\
                                                        extend(group)
                else:
                    # Add particle2 to this decay group
                    self.decay_groups[group_index].append(particle2)

            elif len(non_sm_particles) > 1:
                # Check if any of the particles are not already in any
                # decay group. If there are any, let another particle
                # take care of this interaction instead, later on.

                non_checked_particles = [p for p in non_sm_particles if \
                                         (p.get('is_part') and not p in \
                                          sum(self.decay_groups, []) or \
                                          not p.get('is_part') and not \
                                          self.get_particle(\
                                                     p.get_anti_pdg_code()) in \
                                          sum(self.decay_groups, []))
                                         ]

                if not non_checked_particles:
                    # All particles have been checked. Analyze interaction.

                    if len(non_sm_particles) == 2:
                        # Are any of the particles in my decay group already?
                        this_group_particles = [p for p in non_sm_particles \
                                                if p in self.decay_groups[\
                                                                   group_index]]
                        if len(this_group_particles) == 2:
                            # There can't be any conserved quantum
                            # number! Should be SM group!
                            group = self.decay_groups.pop(group_index)
                            self.decay_groups[0].extend(group)
                            continue
                        elif len(this_group_particles) == 1:
                            # One particle is in the same group as this particle
                            # The other (still non_sm yet) must be SM group.
                            particle2 = [p for p in non_sm_particles \
                                             if p != this_group_particles[0]][0]
                            if not particle2.get('is_part'):
                                particle2 = self.get_particle(particle2.get_anti_pdg_code())

                            group_index2 = [i for (i, g) in \
                                                enumerate(self.decay_groups)\
                                                if particle2 in g][0]
                            group_2 = self.decay_groups.pop(group_index2)
                            self.decay_groups[0].extend(group_2)

                        else:
                            # If the two particles are in another same group,
                            # this particle must be the SM particle.
                            # Transform the 1st non_sm_particle into particle
                            particle1 = non_sm_particles[0]
                            if not particle1.get('is_part'):
                                particle1 = self.get_particle(\
                                    particle1.get_anti_pdg_code())
                            # Find the group of particle1
                            group_index1 = [i for (i, g) in \
                                            enumerate(self.decay_groups) \
                                            if particle1 in g][0]

                            # If the other non_sm_particle is in the same group
                            # as particle1, try to merge this particle to SM
                            if non_sm_particles[1] in \
                                    self.decay_groups[group_index1]:
                                if group_index > 0:
                                    group = self.decay_groups.pop(group_index)
                                    self.decay_groups[0].extend(group)

                    if len(non_sm_particles) == 3:
                        # Are any of the particles in my decay group already?
                        this_group_particles = [p for p in non_sm_particles \
                                                if p in self.decay_groups[\
                                                                   group_index]]
                        if len(this_group_particles) == 2:
                            # Also the 3rd particle has to be in this group.
                            # Merge.
                            particle2 = [p for p in non_sm_particles if p not \
                                         in this_group_particles][0]
                            if not particle2.get('is_part'):
                                particle2 = self.get_particle(\
                                                  particle2.get_anti_pdg_code())
                            group_index2 = [i for (i, g) in \
                                            enumerate(self.decay_groups) \
                                            if particle2 in g][0]
                            group = self.decay_groups.pop(max(group_index,
                                                              group_index2))
                            self.decay_groups[min(group_index, group_index2)].\
                                                                extend(group)
                        if len(this_group_particles) == 1:
                            # The other two particles have to be in
                            # the same group
                            other_group_particles = [p for p in \
                                                     non_sm_particles if p not \
                                                     in this_group_particles]
                            particle1 = other_group_particles[0]
                            if not particle1.get('is_part'):
                                particle1 = self.get_particle(\
                                                  particle1.get_anti_pdg_code())
                            group_index1 = [i for (i, g) in \
                                            enumerate(self.decay_groups) \
                                            if particle1 in g][0]
                            particle2 = other_group_particles[0]
                            if not particle2.get('is_part'):
                                particle2 = self.get_particle(\
                                                  particle2.get_anti_pdg_code())
                            group_index2 = [i for (i, g) in \
                                            enumerate(self.decay_groups) \
                                            if particle2 in g][0]

                            if group_index1 != group_index2:
                                # Merge groups
                                group = self.decay_groups.pop(max(group_index1,
                                                                  group_index2))
                                self.decay_groups[min(group_index1,
                                                      group_index2)].\
                                                                   extend(group)

                        # One more case possible to say something
                        # about: When two of the three particles are
                        # in the same group, the third particle has to
                        # be in the present particle's group. I'm not
                        # doing this case now though.

                    # For cases with number of non-sm particles > 3,
                    # There are also possibilities to say something in
                    # particular situations. Don't implement this now
                    # however.

    def find_decay_groups_general(self, sm_ids = \
                                  [1,2,3,4,5,6,11,12,13,14,15,16,21,22,23,24]):
        """Iteratively find decay groups, suitable to vertex in all orders
           Algrorithm:
           1. Establish the reduced_interactions
              a. Read non-sm particles only
                 (not in sm_ids and not in decay_groups[0])
              b. If the particle appears in this interaction before, 
                 not only stop read it but also remove the existing one.
              c. If the interaction has only one particle,
                 move this particle to SM-like group and void this interaction.
              d. If the interaction has no particle in it, delete it.   
           2. Iteratively reduce the interaction
              a. If there are two particles in this interaction,
                 they must be in the same group. 
                 And we can delete this interaction since we cannot draw more
                 conclusion from it.
              b. If there are only one particle in this interaction,
                 this particle must be SM-like group
                 And we can delete this interaction since we cannot draw more
                 conclusion from it.
              c. If any two particles in this interaction already belong to the 
                 same group, remove the two particles. Delete particles that 
                 become SM-like as well. If this interaction becomes empty 
                 after these deletions, delete this interaction.
              d. If the iteration does not change the reduced_interaction at all
                 stop the iteration. All the remaining reduced_interaction must
                 contain at least three non SM-like particles. And each of 
                 them belongs to different groups.
           3. If there is any particle that has not been classified,
              this particle is lonely i.e. it does not related to 
              other particles. Add this particle to decay_groups.
        """
        
        #Setup the SM particles and initial decay_groups, reduced_interactions
        self.decay_groups = [[]]
        self.reduced_interactions = []

        #Read the interaction information and setup
        for inter in self.get('interactions'):
            temp_int = {'id':inter.get('id'), 'particles':[]}
            for part in inter['particles']:
                #If this particle is anti-particle, convert it.
                if not part.get('is_part'):
                    part = self.get_particle(part.get_anti_pdg_code())
                                  
                #Read this particle if it is not in SM
                if not part.get('pdg_code') in sm_ids and \
                   not part in self.decay_groups[0]:
                    #If pid is not in the interaction yet, append it
                    if not part in temp_int['particles']:
                        temp_int['particles'].append(part)
                    #If pid is there already, remove it since double particles
                    #is equivalent to none.
                    else:
                        temp_int['particles'].remove(part)

            # If there is only one particle in this interaction, this must in SM
            if len(temp_int['particles']) == 1:
                # Remove this particle and add to decay_groups
                part = temp_int['particles'].pop(0)
                self.decay_groups[0].append(part)

            # Finally, append only interaction with nonzero particles
            # to reduced_interactions.
            if len(temp_int['particles']):
                self.reduced_interactions.append(temp_int)
            # So interactions in reduced_interactions are all 
            # with non-zero particles in this stage


        # Now start the iterative interaction reduction
        change = True
        while change:
            change = False
            for inter in self.reduced_interactions:
                #If only two particles in inter, they are in the same group
                if len(inter['particles']) == 2:
                    #If they are in different groups, merge them.
                    #Interaction is useless.

                    # Case for the particle is in decay_groups
                    if inter['particles'][0] in sum(self.decay_groups, []):
                        group_index_0 =[i for (i,g) in\
                                        enumerate(self.decay_groups)\
                                        if inter['particles'][0] in g][0]

                        # If the second one is also in decay_groups, merge them.
                        if inter['particles'][1] in sum(self.decay_groups, []):
                            if not inter['particles'][1] in \
                                    self.decay_groups[group_index_0]:
                                group_index_1 =[i for (i,g) in \
                                                enumerate(self.decay_groups)\
                                                if inter['particles'][1] 
                                                in g][0]
                                # Remove the outer group
                                group_1 = self.decay_groups.pop(max(\
                                          group_index_0, group_index_1))
                                # Merge with the inner one
                                self.decay_groups[min(group_index_0, \
                                                 group_index_1)].extend(group_1)
                        # The other one is no in decay_groups yet
                        # Add inter['particles'][1] to the group of 
                        # inter['particles'][0]
                        else:
                            self.decay_groups[group_index_0].append(
                                inter['particles'][1])
                    # Case for inter['particles'][0] is not in decay_groups yet.
                    else:
                        # If only inter[1] is in decay_groups instead, 
                        # add inter['particles'][0] to its group.
                        if inter['particles'][1] in sum(self.decay_groups, []):
                            group_index_1 =[i for (i,g) in \
                                            enumerate(self.decay_groups)\
                                            if inter['particles'][1] in g][0]
                            # Add inter['particles'][0]
                            self.decay_groups[group_index_1].append(
                                inter['particles'][0])

                        # Both are not in decay_groups
                        # Add both particles to decay_groups
                        else:
                            self.decay_groups.append(inter['particles'])

                    # No matter merging or not the interaction is useless now. 
                    # Kill it.
                    self.reduced_interactions.remove(inter)
                    change = True

                # If only one particle in this interaction,
                # this particle must be SM-like group.
                elif len(inter['particles']) == 1:
                    if inter['particles'][0] in sum(self.decay_groups, []):
                        group_index_1 =[i for (i,g) in \
                                        enumerate(self.decay_groups)\
                                        if inter['particles'][0] in g][0]
                        # If it is not, merge it with SM.
                        if group_index_1 > 0:
                            self.decay_groups[0].extend(self.decay_groups.pop(\
                                                                 group_index_1))

                    # Inter['Particles'][0] not in decay_groups yet, 
                    # add it to SM-like group
                    else:
                        self.decay_groups[0].extend(inter['particles'])

                    # The interaction is useless now. Kill it.
                    self.reduced_interactions.remove(inter)
                    change = True
                
                # Case for more than two particles in this interaction.
                # Remove particles with the same group.
                elif len(inter['particles']) > 2:
                    #List to store the id of each particle's decay group
                    group_ids = []
                    # This list is to prevent removing elements during the 
                    # for loop to create errors.
                    # If the value is normal int, the particle in this position 
                    # is valid. Else, it is already removed. 
                    ref_list = range(len(inter['particles']))
                    for i, part in enumerate(inter['particles']):
                        try:
                            group_ids.append([n for (n,g) in \
                                              enumerate(self.decay_groups) \
                                              if part in g][0])
                        # No group_ids if this particle is not in decay_groups
                        except IndexError:
                            group_ids.append(None)
                            continue
                        
                        # If a particle is SM-like, remove it!
                        # (necessary if some particles turn to SM-like during
                        # the loop then we could reduce the number and decide
                        # groups of the rest particle
                        if group_ids[i] == 0:
                            ref_list[i] = None
                            change = True

                        # See if any valid previous particle has the same group.
                        # If so, both the current one and the previous one
                        # is void
                        for j in range(i):
                            if (group_ids[i] == group_ids[j] and \
                                group_ids[i] != None) and ref_list[j] != None:
                                # Both of the particles is useless for 
                                # the determination of parity
                                ref_list[i] = None
                                ref_list[j] = None
                                change = True
                                break
                    
                    # Remove the particles label with None in ref_list
                    # Remove from the end to prevent errors in list index.
                    for i in range(len(inter['particles'])-1, -1, -1):
                        if ref_list[i] == None:
                            inter['particles'].pop(i)

                    # Remove the interaction if there is no particle in it
                    if not len(inter['particles']):
                        self.reduced_interactions.remove(inter)

                # Start a new iteration...


        # Check if there is any particle that cannot be classified.
        # Such particle is in the group of its own.
        for part in self.get('particles'):
            if not part in sum(self.decay_groups, []) and \
                    not part.get('pdg_code') in sm_ids:
                self.decay_groups.append([part])

    def find_stable_particles(self):
        """ Find stable particles that are protected by parity conservation
            (massless particle is not included). 
            Algorithm:
            1. Find the lightest massive particle in each group.
            2. From reduced_interactions to see if they can decay into
               others.
        """
        # If self.decay_groups is None, find_decay_groups first.
        if not self.decay_groups:
            self.find_decay_groups_general()
        # The list for the stable particle list of all groups
        stable_candidates = [[]]
        large_group_ids = []
        self.stable_particles = base_objects.ParticleList()
        
        # Find lightest particle in each group.
        # SM-like group is excluded.
        for group in self.decay_groups[1:]:
            # The stable particles of each group is described by a sublist
            # (take degeneracy into account). Group index is the index in the
            # stable_candidates.
            stable_candidates.append([])
            # Set the initial mass
            lightest_mass = eval(group[0].get('mass')).real
            for part in group:
                # If there is a massless particle, there is no massive
                # stable particle in this group
                if part.get('mass') != 'ZERO':
                    # If the mass is smaller, replace the the list.
                    if eval(part.get('mass')).real < lightest_mass :
                        stable_candidates[-1] = [part]
                    # If degenerate, append current particle to the list.
                    elif eval(part.get('mass')).real == lightest_mass:
                        stable_candidates[-1].append(part)
                # Escape this loop when massless particle exists
                else:
                    stable_candidates[-1] = []
                    break

        for inter in self.reduced_interactions:
            # Ids for the groups that particles of this inter belong to
            temp_large_group = [[index for i, g in enumerate(self.decay_groups)\
                                 if p in g][0] for p in inter['particles']]

            # Check if any id is repeated in previous groups
            for group_id in temp_large_group:
                for i, g in enumerate(large_group_ids):
                    # If so, merge it to current group
                    # This way enable the multi-merge from several previous
                    # group
                    if group_id in g:
                        # Extend current group with non-existing group_id
                        temp_large_group.extend([p for p in g \
                                                 if not p in temp_large_group])
                        # Empty this already merge group ids
                        large_group_ids[i] = []

            # Add current up-to-date group to large_group_ids
            large_group_ids.append(temp_large_group)


        for common_group in large_group_ids:
            # Avoid common_group with no element (already merged)
            if not common_group:
                # If there is a massless particle in the first group
                # These groups do not have stable particle
                if not stable_candidates[common_group[0]]:
                    break
                # Set initial stable_particles
                temp_partlist = stable_candidates[common_group[0]]
                # Go through each group in common_group to find the lightest
                # particles
                for group_id in common_group:
                    # If the group has no stable particle (i.e. contain
                    # massless particle) break the search and do not add
                    # any particle.
                    if not stable_candidates[group_id]:
                        temp_partlist = []
                        break
                    # If the new group has lighter mass, replace the
                    # temp_partlist
                    elif eval(stable_candidates[group_id][0].get('mass')) < \
                            eval(temp_partlist[0].get('mass')):
                        temp_partlist = stable_candidates[group_id]
                    # If the new group has stable particle with the equal mass
                    # append the stable_candidates to temp_partlist
                    elif eval(stable_candidates[group_id][0].get('mass')) == \
                            eval(temp_partlist[0].get('mass')):
                        temp_partlist.extend(stable_candidates[group_id])

                # If temp_partlist is not empty, add to stable_particles
                if temp_partlist:
                    self.stable_particles.append(temp_partlist)

        # Append the stable particles if their group stand alone
        # (the mixing definition is in large_group_ids)
        for i, stable_particlelist in enumerate(stable_candidates):
            if not i in sum(large_group_ids, []):
                self.stable_particles.append(stable_particlelist)              
 
        return self.stable_particles

    def find_channels(self, part, max_partnum):
        """ Function that find channels for a particle.
            Call the function in DecayParticle."""
        part.find_channels(max_partnum, self)

#===============================================================================
# Channel: A specialized Diagram object for decay
#===============================================================================
class Channel(base_objects.Diagram):
    """Channel: a diagram that describes a certain decay channel
                with apprximated (mean) matrix element, phase space area,
                and decay width
                ('apx_matrixelement', 'apx_PSarea', and  'apx_decaywidth')
                Model must be specified.
    """

    sorted_keys = ['vertices',
                   'orders',
                   'onshell', 'has_idpart', 'id_partlist',
                   'apx_matrixelement', 'apx_PSarea', 'apx_decaywidth']

    def default_setup(self):
        """Default values for all properties"""
        self['vertices'] = base_objects.VertexList()
        self['orders'] = {}
        
        # New properties
        self['onshell'] = False
        # This property denotes whether the channel has 
        # identical particles in it.
        self['has_idpart'] = False
        # The position of the identicle particles.
        self['idpart_list'] = {}
        # Decay width related properties.
        self['apx_matrixelement'] = 0.
        self['apx_PSarea'] = 0.
        self['apx_decaywidth'] = 0.

    def filter(self, name, value):
        """Filter for valid diagram property values."""
        
        if name in ['apx_matrixelement', 'apx_PSarea', 'apx_decaywidth']:
            if not isinstance(value, float):
                raise self.PhysicsObjectError, \
                    "Value %s is not a float" % str(value)
        
        if name == 'onshell' or name == 'has_idpart':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid onshell condition." % str(value)

        return super(Channel, self).filter(name, value)
    
    def get(self, name):
        """ Check the onshell condition before the user get it. """
        
        if name == 'onshell':
            print "It is suggested to get onshell property from get_onshell function"

        return super(Channel, self).get(name)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        return self.sorted_keys

    def get_initial_id(self):
        """ Return the id of initial particle"""
        return self.get('vertices')[-1].get('legs')[-1].get('id')

    def get_final_legs(self):
        """ Return a list of the final state legs."""

        temp_legs = []
        for vert in self.get('vertices'):
            for leg in vert.get('legs'):
                if not leg.get('number') in [l.get('number') \
                                                 for l in temp_legs] and \
                                                 leg.get('number') > 1:
                    temp_legs.append(leg)

        return temp_legs
        
    def get_onshell(self, model):
        """ Evaluate the onshell condition with the aid of get_final_legs"""
        # Check if model is valid
        if not isinstance(model, base_objects.Model):
            raise self.PhysicsObjectError, \
                "The argument %s must be a model." % str(model)

        final_mass = sum([eval(model.get_particle(l.get('id')).get('mass')) \
                              for l in self.get_final_legs()])
        final_mass = final_mass.real
        ini_mass = eval(model.get_particle(self.get_initial_id()).get('mass'))
        ini_mass = ini_mass.real
        self['onshell'] = ini_mass > final_mass

        return self['onshell']

    def get_idpartlist(self):
        """ Get the position of identical particles in this channel.
            The format of idpart_list is a dictionary with the vertex
            which has identical particles, value is the particle id and
            leg index of identicle particles. Eg.
            idpart_list = {vertex_index_1: [(pid_1, [index_1, index_2, ..]),
                                            (pid_2, [index_1, index_2, ..])],
                           vertex_index_2...}
        """

        # Look if there is any two more leg with the same id within a vertex.
        for vindex, vert in enumerate(self.get('vertices')):
            lindex_dict = {}
            pid_list = []
            # Record the occurence of each leg.
            for lindex, leg in enumerate(vert.get('legs')):
                try:
                    lindex_dict[leg['id']].append(lindex)
                except KeyError:
                    lindex_dict[leg['id']] = [lindex]

            for key, indexlist in lindex_dict.items():
                # If more than one index for a key, 
                # there are identical particles.
                if len(indexlist) > 1:
                    self['has_idpart'] = True
                    # Record the index of vertex, leg id, and 
                    # the list of leg index.
                    try:
                        self['idpart_list'][vindex].append((key, indexlist))
                    except KeyError:
                        self['idpart_list'][vindex] = [(key, indexlist)]

        return self['idpart_list']
                
                
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
