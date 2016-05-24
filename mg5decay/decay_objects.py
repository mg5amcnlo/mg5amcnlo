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
"""Definition for the objects used in the decay module.
   DecayParticle: this object contains all the decay related properties
                  including all the decay vertices and decay channels.
                  This object also has the 'is_stable' label to denote
                  wether this particle is stable.
   DecayParticleList: this helper class will help to turn Particle 
                      in base_objects into DecayParticle.
   DecayModel: This model contains DecayParticle. Particle in base_objects
               will be automatically converted into DecayParticle either 
               during the initialization or when a ParticleList is set as
               the particles of DecayModel through set function.
               This model can search all the decay_vertexlist for all
               its particles at one time. The function 'find_stable_particles'
               and 'find_decay_groups_general' will find stable particles
               automatically from interaction and mass without any input.
   Channel: A diagram object specialized for decay process. This includes
            several helper functions for channel generations and the functions
            to calculate the approximate decay width.
   ChannelList: A list of channels.
   
   Users can run DecayParticle.find_channels(final_state_number, model)
   to get the channels with final state number they request. Or they
   may run DecayModel.find_all_channels(final_state_number) to search
   the channels for all particles in the given model."""
   
from __future__ import division

import array
import cmath
import copy
import itertools
import logging
import math
import os
import re
import sys
import time

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation

import madgraph.various.misc as misc

import models.import_ufo as import_ufo
from madgraph import MadGraph5Error, MG5DIR

import models.model_reader as model_reader

ZERO = 0
#===============================================================================
# Logger for decay_module
#===============================================================================

logger = logging.getLogger('madevent.decay_objects')


#===============================================================================
# DecayParticle
#===============================================================================
class DecayParticle(base_objects.Particle):
    """DecayParticle includes the decay vertices and channels for this
       particle. The label 'is_stable' specifies whether the particle
       is stable. The function find_channels will launch all necessary
       functions it needs. If users try to find channels with more final
       state particles, they can run find_channels_nextlevel to get the
       channels in next level.
    """
    sorted_keys = ['name', 'antiname', 'spin', 'color',
                   'charge', 'mass', 'width', 'pdg_code',
                   'line',
                   'is_part', 'self_antipart', 'is_stable',
                   'decay_vertexlist', 'decay_channels', 'apx_decaywidth',
                   'apx_decaywidth_err', '2body_massdiff',
                   'radiations'
                  ]


    def __init__(self, init_dict={}, force=False):
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
            self.set(item, init_dict[item], force)
            
            
    def default_setup(self):
        """Default values for all properties"""
        
        super(DecayParticle, self).default_setup()

        self['is_stable'] = False
        #log of the find_vertexlist history
        self['vertexlist_found'] = False
        self['max_vertexorder'] = 0

        # The decay_vertexlist is a dictionary with vertex list as items and
        # final state particles and on shell condition as keys.
        # decay_channels corresponds to one diagram for each channel, 
        # while decay_amplitudes are a series of diagrams with the same
        # initial and final states.

        self['decay_vertexlist'] = {}
        self['decay_channels'] = {}
        self.decay_amplitudes = {}
        self['apx_decaywidth'] = 0.
        self['apx_decaywidth_err'] = 0.
        self['2body_massdiff'] = 0.
        self['radiations']=[]
        
        
    def get(self, name):
        """ Evaluate some special properties first if the user request. """

        if name == 'apx_decaywidth' \
                and not self[name] \
                and not self['is_stable']:
            self.update_decay_attributes(True, False, True)
            return self[name]
        elif name == 'apx_decaywidth_err' and not self[name]:
            self.estimate_width_error()
            return self[name]
        elif name == 'decay_amplitudes':
            return self.decay_amplitudes
        else:
            # call the mother routine
            return DecayParticle.__bases__[0].get(self, name)

    def set(self, name, value, *args,**opts):
        """assign a value"""

        if name == 'decay_amplitudes':
            self.decay_amplitudes = value
        else:
            out = super(DecayParticle, self).set(name, value, *args, **opts)
            return out
    def check_vertex_condition(self, partnum, onshell, 
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
        self.check_vertex_condition(partnum, onshell, value, model)
       
        #Determine the number of final particles.
        #Find all the possible initial particle(s).
        #Check onshell condition if the model is given.
        if model:
            if (abs(eval(self.get('mass')) == 0.)) and (len(value) != 0):
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
                total_mass = sum([abs(eval(model.get_particle(l['id']).get('mass'))) for l in vert['legs']])
                ini_mass = abs(eval(self.get('mass')))
                
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
                   abs(ch.get_anti_initial_id()) != abs(self.get('pdg_code'))):
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
        return True


    def check_amplitudes(self, partnum, value):
        """Check the validity of DecayAmplitudes condition, including,
           partnum: final state particle number,
           value  : the assign amplitudelist, all amplitudes in it must
                    be consistent to the processess they hold.
        """
        # Check if partnum is an integer.
        if not isinstance(partnum, int):
            raise self.PhysicsObjectError, \
                "Final particle number %s must be an integer." % str(partnum)
        
        # Check if the value is a DecayAmplitudeList
        if (not isinstance(value, DecayAmplitudeList) and value):
            raise self.PhysicsObjectError, \
                "%s must be DecayAmplitudeList type." % str(value)
                
        return True

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

        if name == 'decay_amplitudes':

            #Value must be a dictionary.
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                    "Decay_amplitudes %s must be a dictionary." % str(value)

            # For each key and item, check them with check_amplitudes
            for key, item in value.items():                
                self.check_amplitudes(key, item)
                    
        if name == 'vertexlist_found' or name == 'is_stable':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "Propery %s should be Boolean type." % name

        if name == 'max_vertexorder':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Property %s should be int type." % name

        # Check apx_decaywidth and apx_decaywidth_err
        if name == 'apx_decaywidth' or name == 'apx_decaywidth_err' \
                or name == '2body_massdiff':
            if not isinstance(value, float) and not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Property %s must be float type." % str(value)

        super(DecayParticle, self).filter(name, value)

        return True

    def reset_decay_attributes(self, reset_width, reset_err, reset_br):
        """ Depend on the given arguments, 
            reset the apx_decaywidth, apx_decaywidth_err, and
            branching ratio of the amplitudes in this particle.
            It is necessary when the channels are changed, 
            e.g. find next level."""

        # Reset decay width
        if reset_width:
            self['apx_decaywidth'] = 0.

        # Reset err
        if reset_err:
            self['apx_decaywidth_err'] = 0.
        
        # Reset the branching ratio inside amplitudes
        if reset_br:
            for n, amplist in self.decay_amplitudes.items():
                for amp in amplist:
                    amp.reset_width_br()

    def update_decay_attributes(self, reset_width, reset_err, reset_br, model=None):
        """ This function will update the attributes related to decay width,
            including total width, branching ratios (of each amplitudes),
            and error of width.
            The arguments specify which attributes needs to be updated.
            Note that the width of amplitudes will not be changed.
            If the apx_decaywidth_err needs to be updated, the model
            must be provided!. """
        
        # Reset the related properties
        self.reset_decay_attributes(reset_width, reset_err, reset_br)

        # Update the total width first.
        # (So the decaywidth_err and branching ratios can be calculated with
        # the new width.)
        if reset_width:
            for n, amplist in self.decay_amplitudes.items():
                for amp in amplist:
                    # Do not calculate the br in this moment
                    self['apx_decaywidth'] += amp.get('apx_decaywidth')

        # Update the apx_decaywidth_err
        if reset_err:
            self.estimate_width_error(model)

        # Update the branching ratio in the end with the updated total width
        if reset_br:
            for n, amplist in self.decay_amplitudes.items():
                for amp in amplist:
                    # Reset the br first, so the get function will recalculate
                    # br automatically.
                    amp.get('apx_br')


    def estimate_width_error(self, model=None):
        """ This function will estimate the width error from the highest order
            off-shell channels.
            If model is provided, the apx_decaywidth_err of each channel will
            be (re)calculated. """

        if self['apx_decaywidth']:
            final_level = max([k[0] for k, i in self['decay_channels'].items()])

            # Do not recalculate the apx_decaywidth_nextlevel here
            err = sum([c.get('apx_decaywidth_nextlevel', model) \
                           for c in self.get_channels(final_level, False)])/ \
                           self['apx_decaywidth']

        elif self.get('is_stable'):
            err = 0.
        else:
            err = 1.

        self['apx_decaywidth_err'] = err

        return err

    def decaytable_string(self, format='normal'):
        """ Output the string for the decay table.
            If format is 'full', all the channels in the process will be
            printed."""        

        seperator = str('#'*80)
        output = '\n'+seperator
        output += str('\n#\n#\tPDG\t\tWIDTH\t\tERROR\n')
        output += str('DECAY\t%8d\t%.5e     %.3e  #%s decay\n') \
            %(self.get('pdg_code'), 
              self.get('apx_decaywidth'),
              self.get('apx_decaywidth_err'),
              self.get('name'))
        output += seperator
        # Write the decay table from 2-body decay.
        n = 2
        while n:
            if n in self.get('decay_amplitudes').keys():
                # Do not print empty amplitudes
                if len(self.get_amplitudes(n)):
                    # Titie line
                    output += '\n#\tBR\tNDA       '
                    # ID (ID1, ID2, ID3...)
                    output += '        '.join(['ID%d' %(i+1) for i in range(n)])
                    if format == 'cmp':
                        output += '\tratio'
                    # main content
                    output += '\n%s' \
                        % self.get_amplitudes(n).decaytable_string(format)

                # Increase n to nextlevel    
                n += 1
            else:
                break

        return output+'\n#\n#'
        

    def get_vertexlist(self, partnum ,onshell):
        """Return the n-body on/offshell decay vertexlist. 
           If there is no such vertex, return None.
        
           partnum = number of final particles.
           If onshell=false, return the on-shell list and vice versa.
        """
        #check the validity of arguments
        self.check_vertex_condition(partnum, onshell)
        
        return self.get('decay_vertexlist').get((partnum, onshell), [])

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
        if not self['vertexlist_found']:
            logger.warning("The vertexlist of this particle has not been searched."+"Try find_vertexlist first.")
            return

        vertnum_list = [k[0] for k in self['decay_vertexlist'].keys() \
             if self['decay_vertexlist'][k]]
        if vertnum_list:
            self['max_vertexorder'] = max(vertnum_list)
        else:
            self['max_vertexorder'] = 0

        return self['max_vertexorder']

    # OBSOLETE function. It is recommended to run the find_vertexlist in
    # DecayModel object.
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
        if self['vertexlist_found'] and not option:
            return 'The vertexlist has been setup.', \
                'No action proceeds because of False option.'

        # Reset the decay vertex before finding
        self['decay_vertexlist'] = {(2, False) : base_objects.VertexList(),
                                    (2, True)  : base_objects.VertexList(),
                                    (3, False) : base_objects.VertexList(),
                                    (3, True)  : base_objects.VertexList()}

        #Set the vertexlist_found at the end
        self['vertexlist_found'] = True

        # Do not include the massless and stable particle
        model.get('stable_particles')
        if self.get('is_stable'):
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
                ini_mass = abs(eval(self.get('mass')))
                vert = base_objects.Vertex()
                legs = base_objects.LegList()

                # Setup all the legs and find final_mass
                for part in partlist:
                    legs.append(base_objects.Leg({'id': part.get_pdg_code()}))
                    total_mass += abs(eval(part.get('mass')))
                    #Initial particle has not been found: ini_found = True
                    if (part == model.get_particle(self.get_anti_pdg_code())):
                        ini_leg = legs.pop()
                        ini_leg.set('id', self.get_pdg_code())
                    
                #Sort the outgoing leglist for comparison sake (removable!)
                legs.sort(key= lambda l: l['id'], reverse=True)
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
                                            = base_objects.VertexList([vert])

        

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

    def get_max_level(self):
        """ Get the max channel level that the particle have so far. """
        # Turn off the logger in get_amplitude temporarily
        
        # Initial value
        n = 2
        # Look at the amplitudes or channels to find the max_level
        while self.get_amplitudes(n) or ((n,False) in self['decay_channels'].keys()):
            n += 1

        # n is the failed value, return n-1.
        return (n-1)

    def get_amplitude(self, final_ids):
        """Return the amplitude with the given final pids.
           If no suitable amplitude is found, retun None.
        """
        if not isinstance(final_ids, list):
            raise self.PhysicsObjectError,\
                "The final particle ids %s must be a list of integer." \
                %str(final_ids)

        if any([not isinstance(i, int) for i in final_ids]):
            raise self.PhysicsObjectError,\
                "The final particle ids %s must be a list of integer." \
                %str(final_ids)

        # Sort the given id list first
        final_ids.sort()
        # Search in the amplitude list of given final particle number
        if self.get_amplitudes(len(final_ids)):
            for amp in self.get_amplitudes(len(final_ids)):
                ref_list = [l['id'] \
                                for l in amp.get('process').get_final_legs()]
                ref_list.sort()
                if ref_list == final_ids:
                    return amp
                
        return None

    def get_amplitudes(self, partnum):
        """Return the n-body decay amplitudes.
           partnum = n.
           If the amplitudes do not exist, return none.
        """
        #check the validity of arguments
        if not isinstance(partnum, int):
            raise self.PhysicsObjectError, \
                "The particle number %s must be an integer."  %str(partnum)

        try:
            return self.get('decay_amplitudes')[partnum]
        except KeyError:
            logger.info('The amplitudes of %d for final particle number % d do not exist' % (self['pdg_code'],partnum))
            return []

    def set_amplitudes(self, partnum, value):
        """Set the n_body decay_amplitudes.
           partnum: n, 
           value: the decay_amplitudes that is tried to assign.
        """

        #Check the value by check_amplitudes
        if isinstance(value, DecayAmplitudeList):
            if self.check_amplitudes(partnum, value):
                self.decay_amplitudes[partnum] = value
        elif isinstance(value, list) and \
                all([isinstance(a, DecayAmplitude) for a in value]):
            new_value = DecayAmplitudeList(value)
            if self.check_amplitudes(partnum, new_value):
                self.decay_amplitudes[partnum] = new_value
        else:
            raise self.PhysicsObjectError, \
                "The input must be a list of decay amplitudes."
        
              
    def find_channels(self, partnum, model, min_br=0):
        """ Function for finding decay channels up to the final particle
            number given by max_partnum.
            The branching ratios and err are recalculated in the end.
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
                  equivalent with the existing channels. Then append it.
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
        if not model['vertexlist_found']:
            logger.info("Process the model to add decay informations.")
            model.find_vertexlist()

        # Find stable particles of this model
        model.get('stable_particles')

        # If the channel list exist, return.
        if (partnum, True) in self['decay_channels'].keys() or \
                (partnum, False) in self['decay_channels'].keys():
            logger.info("Particle %s has found channels in %d-body level. " \
                            % (self['name'], partnum) +\
                        "No channel search will not proceed." )
            return

        # Set empty decay_channels for stable particles
        if self.get('is_stable'):
            for num in range(2, partnum+1):
                self['decay_channels'][(num, True)] = ChannelList()
                self['decay_channels'][(num, False)] = ChannelList()
            logger.info("Particle %s is stable. " % self['name'] +
                            "No channel search will not proceed." )
            return

        # Running the coupling constants
        model.running_externals(abs(eval(self.get('mass'))))
        model.running_internals()

        self['apx_decaywidth'] = 0.
        # Find channels from 2-body decay to partnum-body decay channels.
        for clevel in range(2, partnum+1):
            self.find_channels_nextlevel(model, min_br)

        # Update decay width err and branching ratios, but not the total width
        self.update_decay_attributes(False, True, True)

    def find_channels_nextlevel(self, model, min_br=0):
        """ Find channels from all the existing lower level channels to 
            the channels with one more final particle. It is implemented when
            the channels lower than clevel are all found. The user could
            first setup channels by find_channels and then use this function
            if they want to have high level decay.
            NOTE that the width of channels are added sucessively into mother
            particle during the search.
            NO calculation of branching ratios and width err are performed!
            For the search of two-body decay channels, the 2body_massdiff
            is recorded in the end."""

        # Find the next channel level
        try:
            clevel = max([key[0] for key in self['decay_channels'].keys()])+1
        except ValueError:
            clevel = 2

        # Initialize the item in dictionary
        self['decay_channels'][(clevel, True)] = ChannelList()
        self['decay_channels'][(clevel, False)] = ChannelList()

        # Return if this particle is stable
        if self.get('is_stable'):
            logger.info("Particle %s is stable." %self['name'] +\
                            "No channel search will not proceed.")
            return
          
        connect_channel_vertex = self.connect_channel_vertex
        check_repeat = self.check_repeat

        # If there is a vertex in clevel, construct it
        if (clevel, True) in self['decay_vertexlist'].keys() or \
                (clevel, False) in self['decay_vertexlist'].keys():
            for vert in (self.get_vertexlist(clevel, True) + \
                             self.get_vertexlist(clevel, False)):
                
                if len(vert.get('legs')) >3:
                    if not model.keep_Npoint(vert, self):
                        continue
                
                temp_channel = Channel()
                temp_vert = copy.deepcopy(vert)

                # Set the leg number (starting from 2)
                [l.set('number', i) \
                     for i, l in enumerate(temp_vert.get('legs'), start=2)]

                # The final one has 'number' as 1, pid as anti pid of mother
                # (to be consistent with trunk)
                temp_vert.get('legs')[-1]['number'] = 1
                temp_vert.get('legs')[-1]['state'] = False
                temp_vert.get('legs')[-1]['id'] = self.get_anti_pdg_code()
                temp_channel['vertices'].append(temp_vert)

                # Run initial setups for the new channel
                temp_channel.initial_setups(model, True)

                # Setup the 'has_idpart' property
                if Channel.check_idlegs(temp_vert):
                    temp_channel['has_idpart'] = True


                # Add width to total width if onshell
                if temp_channel.get_onshell(model):
                    if temp_channel.has_goldstone(model):
                        continue
                    self['apx_decaywidth'] += temp_channel.get_apx_decaywidth(model)                    

                # Append this channel after all the setups.
                self.get_channels(clevel, temp_channel.get_onshell(model)).\
                    append(temp_channel)
        
        start = time.time()
        repeat_time = 0
        # Go through sub-channels and try to add vertex to reach partnum
        for sub_clevel in range(max((clevel - model.get_max_vertexorder()+1),2),
 clevel):
            # The vertex level that should be combine with sub_clevel
            vlevel = clevel - sub_clevel+1
            # Go through each 'off-shell' channel in the given sub_clevel.
            # Decay of on-shell channel is not a new channel.
            tot = len(self.get_channels(sub_clevel, False))
            for nb,sub_c in enumerate(self.get_channels(sub_clevel, False)):
                if (nb + 1) % 100 == 0:
                    logger.info('%i / %i: %ss' % (nb+1, tot, int(time.time()-start)))
                # Scan each leg to see if there is any appropriate vertex
                for index, leg in enumerate(sub_c.get_final_legs()):

                    # Get the particle even for anti-particle leg.
                    inter_part = model.get_particle(abs(leg['id']))

                    # If this inter_part is stable, do not attach vertices to it
                    if inter_part.get('is_stable'):
                        continue
                    # Get the vertexlist in vlevel
                    # Both on-shell and off-shell vertex 
                    # should be considered.
                    vlist_a = inter_part.get_vertexlist(vlevel, True)
                    vlist_b = inter_part.get_vertexlist(vlevel, False)

                    minv_max = eval(self['mass']) - \
                                    sum([eval(model.get_particle(abs(l['id']))['mass'])
                                         for l in sub_c.get_final_legs() if l!=leg])
                    
                    allow_qcd=True # if two colored particle are lower than the pion. 
                                   # those computation are meaningless.
                    if 0 < minv_max.real < 0.100:
                        logger.warning("WARNING: Mass gap lower than pion mass for decay of %s "
                                                             % self['pdg_code'])
                        logger.warning("decay into colored particle will be remove.")
                        allow_qcd=False
                        if model.get_particle(abs(leg['id']))['color'] != 1:
                            continue
                        
                        
                    # Find appropriate vertex
                    for vert in (vlist_a + vlist_b):
                        # Connect sub_channel to the vertex
                        # the connect_channel_vertex will
                        # inherit the 'has_idpart' from sub_c

                        if not allow_qcd:
                            nb_colored = sum([1 for l in vert['legs'] if \
                                     model.get_particle(abs(l['id']))['color'] != 1])
                            if nb_colored >=1:
                                continue

                        temp_c = self.connect_channel_vertex(sub_c, index, 
                                                             vert, model)
                        temp_c_o = temp_c.get_onshell(model)
                        
                        if temp_c_o and temp_c.has_goldstone(model):
                            continue
                        
                        # Append this channel if it is new
                        rstart = time.time()
                        status = self.check_repeat(clevel, temp_c_o, temp_c)
                        repeat_time += time.time() - rstart
                        if not status:

                            # Check gauge dependence
                            if not temp_c.check_gauge_dependence(model):
                                continue
                            # Calculate the width if onshell
                            # Add to the apx_decaywidth of mother particle
                            if temp_c_o:
                                self['apx_decaywidth'] += temp_c.\
                                    get_apx_decaywidth(model)
                            
                            self.get_channels(clevel, temp_c_o).append(temp_c)
            
            # add valid 3-body decay which start from a valid 2-body decay but 
            #where the sub-decay is forbidden S1 > S2 (S2 > S3 S3)
            #with 2* MS2 < MS1 but 2*MS3> MS2 so S2 is not onshell!
            tot = len(self.get_channels(sub_clevel, True))
            for nb,sub_c in enumerate(self.get_channels(sub_clevel, True)):
                if (nb + 1) % 100 == 0:
                    logger.info('%i / %i: %ss' % (nb+1, tot, int(time.time()-start)))
                # Scan each leg to see if there is any appropriate vertex
                for index, leg in enumerate(sub_c.get_final_legs()):

                    # Get the particle even for anti-particle leg.
                    inter_part = model.get_particle(abs(leg['id']))

                    # If this inter_part is stable, do not attach vertices to it
                    if inter_part.get('is_stable'):
                        continue
                    # Get the vertexlist in vlevel
                    # ONLY off-shell vertex should be considered.
                    vlist_b = inter_part.get_vertexlist(vlevel, False)


                    # Find appropriate vertex
                    for vert in vlist_b:
                        # Connect sub_channel to the vertex
                        # the connect_channel_vertex will
                        # inherit the 'has_idpart' from sub_c

                        temp_c = self.connect_channel_vertex(sub_c, index, 
                                                             vert, model)
                        temp_c_o = temp_c.get_onshell(model)
                        if not temp_c_o:
                            # if offshell, this starts to be extremelly tricky
                            #and time consuming to found all the fake onshell
                            #decay. We therefore neglect all of them. This starts 
                            #to be problematic for 4 body decay (for valid 2-body)
                            #so this should be fine.
                            continue
                        elif temp_c.has_goldstone(model):
                            continue
                        # Append this channel if it is new
                        rstart = time.time()
                        status = self.check_repeat(clevel, temp_c_o, temp_c)
                        repeat_time += time.time() - rstart
                        if not status:
                            # Check gauge dependence
                            if not temp_c.check_gauge_dependence(model):
                                continue
                            # Calculate the width if onshell
                            # Add to the apx_decaywidth of mother particle
                            self['apx_decaywidth'] += temp_c.\
                                    get_apx_decaywidth(model)
                            
                            self.get_channels(clevel, temp_c_o).append(temp_c)               

            
            if hasattr(self, 'check_repeat_tag'):          
                del self.check_repeat_tag

        # For two-body decay, record the maximal mass difference
        if clevel == 2:
            for channel in self.get_channels(2, True):
                mass_diff = abs(eval(self['mass'])) - sum(channel.get('final_mass_list'))
                if mass_diff > self['2body_massdiff']:
                    self.set('2body_massdiff', mass_diff)


        ## Sort the channels by their width
        #self.get_channels(clevel, False).sort(channelcmp_width)


        # Group channels into amplitudes
        self.group_channels_2_amplitudes(clevel, model, min_br)
        

    def connect_channel_vertex(self, sub_channel, index, vertex, model):
        """ Helper function to connect a vertex to one of the legs 
            in the channel. The argument 'index' specified the position of
            the leg in sub_channel final_legs which will be connected with 
            vertex. If leg is for anti-particle, the vertex will be transform
            into anti-part with minus vertex id."""

        # Copy the vertex to prevent the change of leg number
        new_vertex = copy.deepcopy(vertex)
        
        # Setup the final leg number that is used in the channel.
        leg_num = max([l['number'] for l in sub_channel.get_final_legs()])


        # Find the correct vertex id for decays of antiparticles,
        # using dict in model.conj_int_dict
        # particle legs must be in front of anti-particle legs
        if sub_channel.get_final_legs()[index]['id'] < 0:
            new_vertex['id'] = model['conj_int_dict'][vertex['id']]
            for leg in new_vertex['legs']:
                leg['id']  = model.get_particle(leg['id']).get_anti_pdg_code()

            new_vertex['legs'][:-1] = sorted(new_vertex['legs'][:-1],
                                             key=lambda leg: leg['id'],
                                             reverse=True)


        # Legs continue the number of the final one in sub_channel,
        # except the first and the last one ( these two should be connected.)
        for leg in new_vertex['legs'][1:len(new_vertex['legs'])-1]:
            # For the first and final legs, this loop is useless
            # only for reverse leg id in case of mother leg is anti-particle.
            leg_num += 1
            leg['number'] = leg_num


        # Assign correct number to each leg in the vertex.
        # The first leg follows the mother leg.
        new_vertex['legs'][-1]['number'] = \
            sub_channel.get_final_legs()[index]['number']
        new_vertex['legs'][0]['number'] = new_vertex['legs'][-1]['number']
        
        # Combining vertex with channel
        new_channel = Channel()

        # New vertex is first
        new_channel['vertices'].append(new_vertex)

        # Then extend the vertices of the old channel
        # deepcopy is necessary to get new legs        
        new_channel['vertices'].extend(copy.deepcopy(sub_channel['vertices']))



        # Setup properties of new_channel (This slows the time)
        new_channel.initial_setups(model, True)

        # The 'has_idpart' property of descendent of a channel 
        # must derive from the mother channel and the vertex
        # (but 'id_part_list' will change and should not be inherited)
        new_channel['has_idpart'] = (sub_channel['has_idpart'] or \
                                         bool(Channel.check_idlegs(vertex)))

        return new_channel


    def check_repeat(self, clevel, onshell, channel):
        """ Check whether there is any equivalent channels with the given 
            channel. Use the check_channels_equiv function."""

        # old method (much slower), keep-it for comparison
        #previous_answer = any([Channel.check_channels_equiv(other_c, channel)\
        #               for other_c in self.get_channels(clevel, onshell)
        #               if abs(sum(other_c['final_mass_list'])-
        #                 sum(channel['final_mass_list']))< 0.01])

        # To optimize the check, check if the final_mass is the same first
        # this will significantly reduce the number of call to
        # check_channels_equiv
        # further speed improve, store the tag and avoid to recompute the final mass.
        if not hasattr(self, 'check_repeat_tag'):
            self.check_repeat_tag = {(clevel,onshell): {}}
        elif not (clevel,onshell) in  self.check_repeat_tag.keys():
            self.check_repeat_tag[(clevel,onshell)] =  {}
      
        tag = channel.get('tag')
        mass = 100 * sum(channel['final_mass_list']) // 1 #avoid numerical inacuracy
        if mass in self.check_repeat_tag[(clevel,onshell)]:
            repeat = any(tag == other_tag 
                     for other_tag in self.check_repeat_tag[(clevel,onshell)][mass])
        else:
            repeat = False
        
        #add the information to the dictionary to fasten following calls
        if not repeat:
            if mass in self.check_repeat_tag[(clevel,onshell)]:
                self.check_repeat_tag[(clevel,onshell)][mass].append(tag)
            else:
                self.check_repeat_tag[(clevel,onshell)][mass] = [tag]
        return repeat
                  
    def group_channels_2_amplitudes(self, clevel, model, min_br=0):
        """ After the channels is found, combining channels with the same 
            final states into amplitudes.
            NO CALCULATION of branching ratio at this stage!
            clevel: the number of final particles
            model: the model use in find_channels
        """

        if not isinstance(clevel, int):
            raise self.PhysicsObjectError, \
                "The channel level %s must be an integer." % str(clevel)

        if not isinstance(model, DecayModel):
            raise self.PhysicsObjectError, \
                "The model must be an DecayModel object."

        # Reset the value of decay_amplitudes
        self.set_amplitudes(clevel, DecayAmplitudeList())
        # Sort the order of onshell channels according to their final mass list.
        self.get_channels(clevel, True).sort(channelcmp_final)

        total_width = self.get('apx_decaywidth')
        for channel in self.get_channels(clevel, True):
            found = False
            # Record the final particle id.
            final_pid = sorted([l.get('id') for l in channel.get_final_legs()])
        
            # Check if there is a amplitude for it. Since the channels with 
            # the similar final states are put together. Use reversed order
            # of loop.
            for amplt in reversed(self.decay_amplitudes[clevel]):
                # Do not include the first leg (initial id)
                if sorted([l.get('id') for l in amplt['process']['legs'][1:]])\
                        == final_pid:
                    amplt.add_std_diagram(channel)
                    found = True
                    break

            # If no amplitude is satisfied, initiate a new one.
            if not found:
                #print 'start : ', channel.nice_string()
                self.get_amplitudes(clevel).append(DecayAmplitude(channel,
                                                                  model))
        if min_br:
            total_width = self.get('apx_decaywidth')
        # calculate the apx_decaywidth and for every amplitude.
        # apr_br WILL NOT be calculated!
        for amp in list(self.get_amplitudes(clevel)):
            approx_width = amp.get('apx_decaywidth')
            if min_br:
                if approx_width or total_width:
                    br = approx_width / total_width
                else:
                    br = 0
                if br.real < min_br:
                    self.decay_amplitudes[clevel].remove(amp)
                
        self.get_amplitudes(clevel).sort(amplitudecmp_width)




    # This helper function is obselete in current algorithm...
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
                                            
                    
#===============================================================================
# DecayParticleList
#===============================================================================
class DecayParticleList(base_objects.ParticleList):
    """A class to store list of DecayParticle, Particle is also a valid
       element, but will automatically convert to DecayParticle"""

    def __init__(self, init_list=None, force=False):
        """Creates a new particle list object. If a list of physics 
        object is given, add them."""

        list.__init__(self)

        if init_list is not None:
            for object in init_list:
                self.append(object, force)

    def append(self, object, force=False):
        """Append DecayParticle, even if object is Particle"""

        if not force:
            assert self.is_valid_element(object), \
                "Object %s is not a valid object for the current list" %repr(object)

        if isinstance(object, DecayParticle):
            if object.get('type') != 'ghost':
                list.append(self, object)
        else:
            list.append(self, DecayParticle(object, force))

    def generate_dict(self):
        """Generate a dictionary from particle id to particle.
        Include antiparticles.
        """

        particle_dict = {}

        for particle in self:
            particle_dict[particle.get('pdg_code')] = particle
            if not particle.get('self_antipart'):
                antipart = copy.copy(particle)
                antipart.set('is_part', False)
                particle_dict[antipart.get_pdg_code()] = antipart

        return particle_dict
    
#===============================================================================
# DecayModel: Model object that is used in this module
#===============================================================================
class DecayModel(model_reader.ModelReader):
    """DecayModel object is able construct the decay vertices for
       all its particles by find_vertexlist. When the user try to get stable
       particles, it will find all stable particles automatically according to
       the particle mass and interactions by find_stable_particles functions.
       The run of find_channels uses the function of DecayParticle. 
       Note that Particle objects will be converted into DecayParticle 
       either during the initialization or when the set function is used.
    """
    sorted_keys = ['name', 'particles', 'parameters', 'interactions', 
                   'couplings', 'lorentz', 
                   'stable_particles', 'vertexlist_found',
                   'reduced_interactions', 'decay_groups', 'max_vertexorder',
                   'decaywidth_list', 
                   'helascalls',
                   'ab_model', 'abmodel_generated', 'coupling_dict','parameter_dict'
                  ]

    def __init__(self, init_dict = {}, force=False):
        """Reset the particle_dict so that items in it is 
           of DecayParitcle type"""

        dict.__init__(self)
        self.default_setup()

        assert isinstance(init_dict, dict), \
            "Argument %s is not a dictionary" % repr(init_dict)

        # Must set particles first so that the particle_dict
        # can point to DecayParticle
        # Futhermore, the set of interactions can have correct particle_dict
        if 'particles' in init_dict.keys():
            p =  init_dict['particles'][0]
            self.set('particles', init_dict['particles'], force)

        self['particle_dict'] = {}
        self.get('particle_dict')
        # Do not copy the particle_dict, it may be old version and point only
        # to Particle rather than DecayParticle.
        for item in init_dict.keys():
            if item != 'particles' and item != 'particle_dict':
                try:
                    self.set(item, init_dict[item], force)
                except:
                    pass

        
    def default_setup(self):
        """The particles is changed to ParticleList"""
        super(DecayModel, self).default_setup()
        self['particles'] = DecayParticleList()
        # Other properties
        self['vertexlist_found'] = False
        self['max_vertexorder'] = 0
        self['conj_int_dict'] = {}

        self['helascalls'] = {}
        self['decay_groups'] = []
        self['reduced_interactions'] = []
        self['stable_particles'] = []
        # Width from the value of param_card.
        self['decaywidth_list'] = {}
        # Properties for abstract model
        self['ab_model'] = AbstractModel()
        self['abmodel_generated'] = False
        self['invalid_Npoint'] = []
        

    def get_sorted_keys(self):
        return self.sorted_keys

    def get(self, name):
        """ Evaluate some special properties first if the user request. """

        if name == 'stable_particles' and not self['stable_particles']:
            self.find_stable_particles()
            return self['stable_particles']
        # reduced_interactions might be empty, cannot judge the evaluation is
        # done or not by it.
        elif (name == 'decay_groups' or name == 'reduced_interactions') and \
                not self['decay_groups']:
            self.find_decay_groups_general()
            return self[name]
        elif name == 'max_vertexorder' and self['max_vertexorder'] == 0:
            self.get_max_vertexorder()
            return self['max_vertexorder']
        else:
            # call the mother routine
            return DecayModel.__bases__[0].get(self, name)
        
    def filter(self, name, value):
        if name == 'vertexlist_found':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "Property %s should be bool type." % name
        if name == 'max_vertexorder':
            if not isinstance(value, int):
                raise self.PhysicsObjectError,\
                    "Property %s should be int type." % name
        if name == 'stable_particles' or name == 'decay_groups':
            if not isinstance(value, list):
                raise self.PhysicsObjectError,\
                    "Property %s should be a list contains several particle list." % name
            for plist in value:                
                if not isinstance(plist, list):
                    raise self.PhysicsObjectError,\
                    "Property %s should be a list contains several particle list." % name
                for p in plist:
                    if not isinstance(p, DecayParticle):
                        raise self.PhysicsObjectError,\
                            "Property %s should be a list contains several particle list." % name

        super(DecayModel, self).filter(name, value)
        
        return True
            
        
    def set(self, name, value, force=False):
        """Change the Particle into DecayParticle"""
        #Record the validity of set by mother routine
        return_value = super(DecayModel, self).set(name, value, force)
        #Reset the dictionaries
        
        if return_value:
            if name == 'particles':
                #Reset dictionaries and decay related properties.
                self['particle_dict'] = {}
                self['got_majoranas'] = None
                self['vertexlist_found'] = False
                self['max_vertexorder'] = 0
                self['decay_groups'] = []
                self['reduced_interactions'] = []
                self['stable_particles'] = []
                self['decaywidth_list'] = {}

                #Convert to DecayParticleList
                self['particles'] = DecayParticleList(value, force=True)
                #Generate new dictionaries with items are DecayParticle
                self.get('particle_dict')
                self.get('got_majoranas')
            if name == 'interactions':
                # Reset dictionaries and decay related properties.
                self['interaction_dict'] = {}
                self['ref_dict_to1'] = {}
                self['ref_dict_to0'] = {}
                self['vertexlist_found'] = False
                self['max_vertexorder'] = 0
                self['decay_groups'] = []
                self['reduced_interactions'] = []
                self['stable_particles'] = []
                self['decaywidth_list'] = {}

                # Generate interactions with particles are DecayParticleList
                # Get particle from particle_dict so that the particles
                # of interaction is the alias of particles of DecayModel
                for inter in self['interactions']:
                    inter['particles']=DecayParticleList([self.get_particle(part.get_pdg_code()) for part in inter['particles']])
                # Generate new dictionaries
                self.get('interaction_dict')
                self.get('ref_dict_to1')
                self.get('ref_dict_to0')

        return return_value

    def get_max_vertexorder(self):
        """find the maxima vertex order (i.e. decay particle number)"""
        if not self['vertexlist_found']:
            logger.warning("Use find_vertexlist before get_max_vertexorder!")
            return

        # Do not include key without any vertexlist in it
        self['max_vertexorder'] = max(sum([[k[0] \
                                            for k in \
                                            p.get('decay_vertexlist').keys() \
                                            if p.get('decay_vertexlist')[k]] \
                                            for p in self.get('particles')], [])
                                      )
        return self['max_vertexorder']

    def get_helascalls(self, number):
        """ Get the helascalls with the number of final particles. Return
            empty list if such helascalls does not exist."""

        return self.get('helascalls').get(number, [])

    def add_helascalls(self, number, diagram):
        """ Add the helascalls with the number of final particles. 
            Use deepcopy of the diagram."""
        
        # diagram should have std_diagram already
        assert diagram['std_diagram']

        std_channel = Channel()
        # Replace the vertices with standard expression 
        # written by IdentifyHelasTagHelasTag.
        std_channel['vertices'] = copy.deepcopy(diagram['std_diagram']['vertices'])
        std_channel['helastag'] = diagram['helastag']


        try:
            self['helascalls'][number].append(std_channel)
        except KeyError:
            self['helascalls'][number] = base_objects.DiagramList([std_channel])


    def collect_helascalls(self, part, number):
        """ Collect diagrams from the ones in the given particle 
            with given final state number that are not in 'helascalls'."""
            
        
        if not isinstance(number, int) or not isinstance(part, DecayParticle):
            raise self.PhysicsObjectError, \
                "Wrong argument types."

        amps = part.get_amplitudes(number)
        diagrams = sum([[d for d in amp['diagrams']] for amp in amps],[])
        calls = self.get_helascalls(number)

        for d in diagrams:
            helastag, std_d = d.get_helas_properties(self)

            found = False
            for num, call in enumerate(calls):
                if not call['helastag']:
                    raise self.PhysicsObjectError, \
                        "The helastag of helascalls should exist."
                if call['helastag'] == helastag:
                    found = True
                    d['helas_number'] = num
                    break

            if not found:
                d['helas_number'] = len(calls)      
                self.add_helascalls(number, d)
                # for empty calls, update the calls to direct to the actual
                # list rather than a fake empty list [].
                if not calls:
                    calls = self.get_helascalls(number)


    def generate_abstract_model(self, force=False):
        """ Generate the abstract particles in AbstractModel."""

        logger.info("Generating the abstract model...")

        self.get('stable_particles')
        # If vertexlist has not been found before, run model.find_vertexlist
        if not self['vertexlist_found']:
            logger.info("Vertexlist of this model has not been searched."+ \
                "Automatically run the model.find_vertexlist()")
            self.find_vertexlist()

        # Add all particles, including stable ones 
        # (could appear in final states)
        self['ab_model'].setup_particles(self.get('particles'), force)

        # Add interactions, except ones with all stable particles or ones
        # with radiation process.
        self['ab_model'].setup_interactions(self.get('interactions'), 
                                            self['conj_int_dict'],
                                            force)

    def generate_abstract_amplitudes(self, part, clevel):
        """ Generate the abstract amplitudes in AbstractModel."""

        # Skip search if this particle is stable
        if part.get('is_stable'):
            logger.info("Particle %s is stable." %part['name'] +\
                            "No abstract amplitude will be generated.")
            return
        
        logger.info("Generating the abstract amplitudes of %s %d-body decays..." % (part['name'], clevel))

        self['ab_model'].generate_ab_amplitudes(part.get_amplitudes(clevel))

            
    def find_vertexlist(self, force=False):
        """ Check whether the interaction is able to decay from mother_part.
            Set the 'decay_vertexlist' of the corresponding particles.
            Utilize in finding all the decay table of the whole model.
            force: set vertices even though it's already in the vertex list.
            Note: only construct VertexList with particle as mother,
                  fix the antiparticle vertices by conj_int_dict.
        """

        # Return if self['vertexlist_found'] is True.\
        if self['vertexlist_found']:
            logger.info("Vertexlist has been searched before.")
            return
        # Find the stable particles of this model and do not assign decay vertex
        # to them.
        self.get('stable_particles')
        
        # Valid initial particle list
        ini_list = []

        for part in self.get('particles'):
            if not part.get('is_stable'):
                #All valid initial particles (mass != 0 and is_part == True)
                ini_list.append(part.get_pdg_code())


        #Prepare the vertexlist        
        for inter in self['interactions']:

            if 'type' in inter and inter['type'] != 'base':
                continue #need only the LO interactions.
            #Calculate the particle number
            partnum = len(inter['particles']) - 1

            temp_legs = base_objects.LegList()
            total_mass = 0
            validity = False

            for num, part in enumerate(inter['particles']):

                #Check if the interaction contains valid initial particle
                if part.get_anti_pdg_code() in ini_list:
                    validity = True

                #Create the original legs
                temp_legs.append(base_objects.Leg({'id':part.get_pdg_code()}))
                total_mass += abs(eval(part.get('mass')))
            
            #Exclude interaction without valid initial particle
            if not validity:
                continue
                
            for num, part in enumerate(inter['particles']):

                # Get anti_pdg_code (pid for incoming particle)
                pid = part.get_anti_pdg_code()

                # Exclude illegal initial particle 
                # (including unstable antiparticles)
                if not pid in ini_list:
                    #print 'bypass', pid, 'not in', ini_list 
                    continue

                # Exclude initial particle (or its own antiparticle) 
                # appears in final particles
                # e.g. radiation, t > t h, and p > p~ etc.                
                pid_list = [p.get_pdg_code() for p in inter.get('particles')]

                # N.B. can't use #pid + #antipid here, which fail
                # for neutral particles.
                if pid_list.count(pid)+pid_list.count(-pid) > 1:

                    antipid = part.get_pdg_code()

                    # Simply ignore processes that're not radiation-like.
                    if len(pid_list)>3:
                        #print 'wrong line'
                        continue
                    # If particle is not antiparticle, #pid=1 and #antipid=1
                    if antipid != pid and not (pid_list.count(pid) ==1 and \
                                                   pid_list.count(-pid) ==1):
                        continue
                    # If particle is antiparticle, #pid=2
                    elif antipid == pid and pid_list.count(pid) ==2:
                        continue

                    # For radiation like process, we need to collect them to
                    # check processes which are identical to 
                    # radiation + proper decay


                    # skip the 2nd initial particle (e.g. b b a)
                    if num > [i for i, l in enumerate(pid_list) if l==antipid][0]:
                        continue

                    # Extract the radiation
                    for temp_id in pid_list:
                        if temp_id != pid and temp_id != antipid:
                            self.get_particle(pid)['radiations'].append(temp_id)

                    """# If the "radiation" turn particle into antiparticle
                    # (p > ~p + something)
                    # Include both the "radiation" and "anti-radiation", the
                    # product emitted by anti-particle
                    if pid_list.count(antipid) > 1:
                        antiradiation = [self.get_particle(l).get_anti_pdg_code() for l in radiation]
                        if set(antiradiation) != set(radiation):
                            self.get_particle(pid)['radiative_products'].append(antiradiation)"""
                        
                    continue

                ini_mass = abs(eval(part.get('mass')))
                onshell = ini_mass > (total_mass - ini_mass)

                # Create new legs for the sort later,
                # turn pid of initial particle into anti-one
                temp_legs_new = copy.deepcopy(temp_legs)
                temp_legs_new[num]['id'] = pid

                # Put initial leg in the last 
                # and sort other legs for comparison
                inileg = temp_legs_new.pop(num)

                # IMPORTANT
                # Particle legs must be in front of anti-particle legs
                # for the sake of HELAS output
                temp_legs_new.sort(key=lambda leg: leg['id'],reverse=True)

                temp_legs_new.append(inileg)
                temp_vertex = base_objects.Vertex({'id': inter.get('id'),
                                                   'legs':temp_legs_new})


                # Record the vertex with key = (interaction_id, part_id)
                try:
                    # Raise error if this vertex appears already
                    # (indication of mother appears in final state)
                    if temp_vertex in self.get_particle(pid)['decay_vertexlist'][(partnum, onshell)] and not force:

                        raise self.PhysicsObjectError,\
                            "found duplicated %d-point vertices in %s" \
                            % (partnum+1, self.get_particle(pid)['name'])

                    self.get_particle(pid)['decay_vertexlist'][(\
                            partnum, onshell)].append(temp_vertex)

                # Create VertexList if necessary
                except KeyError:
                    self.get_particle(pid)['decay_vertexlist'][(\
                            partnum, onshell)] = base_objects.VertexList(\
                        [temp_vertex])

        # Set the property vertexlist_found as True and for all particles
        self['vertexlist_found'] = True
        for part in self['particles']:
            part['vertexlist_found'] = True

        # Find complex conjugate interaction pairs
        self.find_conjugate_dict()

        # Check gauge dependence
        self.gauge_dependence_helper()


    def find_conjugate_dict(self):
        """ Find the dictionary connecting an interaction to its complex 
            conjuagate. """

        # Setup the conj_int_dict for interaction id to the id of 
        # its complex conjugate.
        interlist = [(i['id'], [p.get_anti_pdg_code() for p in i['particles']])\
                         for i in self['interactions']]

        for inter in self['interactions']:
            pids = [p.get_pdg_code() for p in inter['particles']]
            conj_pids = [p.get_anti_pdg_code() for p in inter['particles']]

            # Skip searching if the interaction is self-conjugate
            if sorted(pids) == sorted(conj_pids):
                continue

            found = False
            for item in interlist:
                if sorted(item[1]) == sorted(pids) and \
                        inter['id'] != item[0]:

                    self['conj_int_dict'][inter['id']] = item[0]
                    found = True
                    break

            if not found:
                logger.warning('No hermitian conjugate is found for interaction #%d' %inter['id'])


        #fdata = open(os.path.join(MG5DIR, 'models', self['name'], 'vertexlist_dict.dat'), 'w')
        #fdata.write(str(vertexlist_dict))
        #fdata.close()

    @misc.mute_logger(names=['madgraph.diagram_generation'], levels=[40])
    def keep_Npoint(self, vertex, initpart):
        """ """
        get_mass = lambda x: eval(self.get_particle(x['id']).get('mass'))
        init_mass = eval(initpart.get('mass'))

        if vertex['id'] in self['invalid_Npoint']:
            return False

        interaction = self.get('interaction_dict')[vertex['id']]
        decay_parts = [p for p in interaction['particles']]
        
        if len([1 for p in decay_parts if abs(p['pdg_code'])==abs(initpart['pdg_code'])]) >1:
            self['invalid_Npoint'].append(vertex['id'])
            return False

        # create a process:
        process = base_objects.ProcessDefinition()
        process['model'] = self
        process['orders'] = interaction['orders']
        for order in self.get('coupling_orders'):
            if order not in interaction['orders']:
                process['orders'][order] = 0
        leglist = base_objects.MultiLegList()
        for i,particle in enumerate(interaction['particles']):
            newleg = base_objects.MultiLeg({#'number': i+1,
                                       'ids': [particle['pdg_code'] * (1 if particle['is_part'] else -1)],
                                       'state': True
                                       })
            if abs(newleg['ids'][0]) == abs(initpart['pdg_code']):
                leglist.insert(0, newleg)
            else:
                leglist.append(newleg)
            
            
            
        process['legs'] = leglist
        
        myprocdef = base_objects.ProcessDefinitionList()
        myprocdef.append(process) 
        myproc = diagram_generation.MultiProcess(myprocdef, optimize=False)

        one_offshell = False
        one_subdiag = False
        for amp in myproc['amplitudes']:
            for proc in amp['diagrams']:
                # check if they are substructure
                if not one_subdiag and len(proc['vertices']) >1:
                    one_subdiag = True
                else:
                    continue
                
                #check that all substructure are valid
                #remove if any radiation and two times the same particle in a vertex
                for v in proc['vertices']:
                    if any([get_mass(l)==0 for l in v.get('legs')]):
                        self['invalid_Npoint'].append(vertex['id'])
                        return False

                    ids = set(abs(l['id']) for l in v.get('legs'))
                    if len(ids) != len(vertex.get('legs')):
                        self['invalid_Npoint'].append(vertex['id'])
                        return False

                # check onshell/offshell status                
                prev_mass = 0
                for v in proc['vertices'][:-1]:
                    propa =  v.get('legs')[-1]
                    if propa['number'] == 1:
                        other_mass = sum([get_mass(p) for p in v.get('legs')[1:]])
                        if init_mass.real > other_mass.real + prev_mass:
                            # update the mass for the nex propagator
                            prev_mass = sum([get_mass(p) for p in v.get('legs')[1:-1]])
                            # special for 3 body to check for fake onshell
                            if len(vertex['legs']) == 4:
                                propa_mass = get_mass(v.get('legs')[-1])
                                other_v = proc['vertices'][-1]
                                end_mass = sum([get_mass(p) for p in other_v['legs']
                                                if p['number']!=1])
                                if propa_mass.real < end_mass.real:
                                    one_offshell = True
                                    continue
                                
                        else:
                            one_offshell = True
                            continue
                                
                                
                            
                            
                
        
        if not one_subdiag:
            return True
        elif one_offshell:
            return True
        else:
            return False                
                        

                                        
                     
        



    def gauge_dependence_helper(self):
        """ Check the potential gauge dependence of vertices, i.e.
            3-pt interaction + radiation. 
            Run inside find_vertexlist, after radiations are identified and
            complex-conjugate interactions are found.
            
            For a N (>3) interactions the algorithm is the following.
            1) generate a process object (same particles and same order)
            2) reject the diagram if some of the diagrams have more than one 
            interactions.
            
            REPLACE BY check_Npoint with a check of onshell/offshell in addiation
        """
        return

            

    def color_multiplicity_def(self, colorlist):
        """ Defines the two-body color multiplicity. It is applied in the 
            get_color_multiplicity in channel object.
            colorlist is the a list of two color indices.
            This function will return a list of all possible
            color index of the "mother" particle and the corresponding
            color multiplicities. """
        
        # Raise error if the colorlist is not the right format.
        if not isinstance(colorlist, list):
            raise self.PhysicsObjectError,\
                "The argument must be a list."

        if any([not isinstance(i, int) for i in colorlist]):
            raise self.PhysicsObjectError,\
                "The argument must be a list of integer elements."

        # Sort the colorlist and 
        colorlist.sort()
        color_tuple = tuple(colorlist)
        # The dictionary of color multiplicity
        # The key is the final_color structure and the value is
        # [(ini_color_1, multiplicity_1), (ini_color_2, multiplicity_2), ...]
        color_dict = { \
            # The trivial key and value
            # These define the convention we used.
            (1, 1): [(1, 1)],
            (1, 3): [(3, 1)],
            (1, 8): [(8, 1./2)],
            (1, 6): [(6, 1)],
            (3, 3): [(1, 3), (8, 0.5), (3, 1), (6, 1)],
            # (3, 6) ->  (3, 2 rather than 3), 
            (3, 6): [(3, 2), (8, 3./4)],
            (3, 8): [(3, 0.5*(3-1./3)), (6, 1)],
            (6, 6): [(1, 6), (8, 4./3)],
            (6, 8): [(3, 2), (6, 2-1./3)],
            (8, 8): [(1, 8)],
            # 3-body decay color structure
            # Quick reference
            (3, 3, 8): [(1, 8), (8, 3-1./3)],
            (1, 3, 8): [(3, 3)],
            (1, 3, 3): [(8, 1)],
            (3, 8, 8): [(3, 64./9)]
            }

        return color_dict[color_tuple]





    def read_param_card(self, param_card):
        """Read a param_card and set all parameters and couplings as
        members of this module"""

        self.set_parameters_and_couplings(param_card)
        for param, value in self.get('parameter_dict').items():
            exec("globals()[\'%s\'] = %s" % (param, value))
        for param, value in self.get('coupling_dict').items():
            exec("globals()[\'%s\'] = %s" % (param, value))        

        for particle in self.get('particles'):
            pid = abs(particle['pdg_code'])
            value = self.get('parameter_dict')[particle['width']]
            self['decaywidth_list'][(pid, True)] = float(value.real)
            
        global amZ0, aS
        amZ0 = aS



    def running_externals(self, q, loopnum=2):
        """ Recalculate external parameters at the given scale. """
        
        # Raise error for wrong type of q
        if not isinstance(q, int) and not isinstance(q, long) and \
                not isinstance(q, float):
            raise self.PhysicsObjectError, \
                "The argument %s should be numerical type." %str(q)

        # Declare global value. amZ0 is the alpha_s at Z pole
        global aS, amZ0, mdl_amZ0
        # Define all functions used
        for func in self['functions']:
            exec("def %s(%s):\n   return %s" % (func.name,
                                                ",".join(func.arguments),
                                                func.expr))

        # Setup the alpha_s at different scale
        amt = 0.
        amb = 0.
        amc = 0.
        a_out = 0.

        # Setup parameters
        # MZ, MB are already read in from param_card
    
    
        # get Z mass
        Z = self.get_particle(23)
        if not Z:
            MZ_ref = 91.118
        else:
            MZ_ref = self['parameter_dict'][Z['mass']].real
        
        B = self.get_particle(5)
        if not B:
            MB_ref = 4.7
        else:
            MB_ref = self['parameter_dict'][B['mass']].real
            if not MB_ref:
                MB_ref = 4.7
                
        C = self.get_particle(4)
        if not C:
            MC_ref = 1.42
        else:
            MC_ref = self['parameter_dict'][C['mass']].real
            if not MC_ref:
                MC_ref = 1.42

        # Calculate alpha_s at the scale q
        if q < MB_ref:
            # Running the alpha_s from MZ_ref to MB_ref (fermion_num = 5)
            t = 2 * math.log(MB_ref/MZ_ref)
            amb = self.newton1(t, amZ0, loopnum, 5.)
            if q< MC_ref:
                # Running the alpha_s from MB_ref to MC_ref (fermion_num = 4)
                t = 2.*math.log(MC_ref/MB_ref)
                amc = self.newton1(t, amb, loopnum, 4.)
                try:
                    # Running the alpha_s from MC_ref to q
                    t = 2.*math.log(q/MC_ref)
                    a_out = self.newton1(t, amc, loopnum, 3.)
                except ValueError:
                    a_out = amc
            else:
                # Running the alpha_s from MB_ref to q (fermion_num = 4)
                t = 2.*math.log(q/MB_ref)
                a_out = self.newton1(t, amb, loopnum, 4.)
        else:
            # Running the alpha_s from MZ_ref to MB_ref (fermion_num = 5)
            t = 2 * math.log(q/MZ_ref)
            a_out = self.newton1(t, amZ0, loopnum, 5.)

        # Save the result alpha_s
        aS = a_out


    def newton1(self, t, a_in, loopnum, nf):
        """ Calculate the running strong coupling constant from a_in
            using the t as the energy running factor, 
            loop number given by loopnum, number of fermions by nf."""

        # Setup the accuracy.
        tol = 5e-5

        # Functions used in the beta function
        b0 = {}
        c1 = {}
        c2 = {}
        delc = {}

        for i in range(3,6):
            b0[i] = (11. - 2.*i/3.)/4./math.pi
            c1[i] = (102. - 38./3.*i)/4./math.pi/(11. - 2.*i/3.)
            c2[i] = (2857./2. - 5033.*i/18. + 325* i**2 /54.)\
                /16./math.pi**2/(11. - 2.*i/3.)
            delc[i] = math.sqrt(4*c2[i] - c1[i]**2)

        f2 = lambda x: (1./x + c1[nf] * math.log((c1[nf]*x)/(1. + c1[nf]*x)))
        f3 = lambda x: (1./x + 0.5*c1[nf] * \
                         math.log((c2[nf]* x**2)\
                                      /(1. +c1[nf]*x +c2[nf]* x**2))\
                            -(c1[nf]**2 - 2.*c2[nf])/delc[nf]* \
                            math.atan((2.*c2[nf]*x + c1[nf])/delc[nf]))

        # Return the 1-loop alpha_s
        if loopnum == 1:
            return a_in/(1. + a_in * b0[nf] * t)
        # For higher order correction, setup the initial value of a_out
        else:
            a_out = a_in/(1. + a_in * b0[nf] * t + \
                              c1[nf] * a_in * math.log(1. + a_in * b0[nf] * t))
            if a_out <= 0.:
                a_out = 0.3
            
        # Start the iteration            
        delta = tol +1
        while delta > tol:
            if loopnum == 2:
                f = b0[nf]*t + f2(a_in) - f2(a_out)
                fp = 1./(a_out**2 * (1. + c1[nf]*a_out))
            if loopnum == 3:
                f = b0[nf]*t + f3(a_in) - f3(a_out)
                fp = 1./(a_out**2 * (1. + c1[nf]*a_out + c2[nf]* a_out**2))
            if fp == 0:
                return a_in
            a_out = a_out - f/fp
            delta = abs(f/fp/a_out)

        return a_out

    def running_internals(self):
        """ Recalculate parameters and couplings which depend on
            running external parameters to the given energy scale.
            RUN running_externals before run this function."""

        # Define all functions used
        for func in self['functions']:
            exec("def %s(%s):\n   return %s" % (func.name,
                                                ",".join(func.arguments),
                                                func.expr))

        # External parameters that must be recalculate for different energy
        # scale.
        run_ext_params = ['aS']

        # Extract derived parameters
        derived_parameters = []
        # Take only keys that depend on running external parameters
        ordered_keys = [key for key in self['parameters'].keys() if \
                            key != ('external',) and \
                            any([run_ext in key for run_ext in run_ext_params])]
        # Sort keys, keys depend on fewer number parameters should be
        # evaluated first.
        ordered_keys.sort(key=len)
        for key in ordered_keys:
            derived_parameters += self['parameters'][key]

        # Now calculate derived parameters
        # TO BE IMPLEMENTED use running alpha_s for aS-dependent params
        for param in derived_parameters:
            exec("globals()[\'%s\'] = %s" % (param.name, param.expr))
            if not eval(param.name) and eval(param.name) != 0:
                logger.warning("%s has no expression: %s" % (param.name,
                                                             param.expr))
#             try:
#                 logger.info("Recalculated parameter %s = %f" % \
#                             (param.name, eval(param.name)))
#             except TypeError:
#                 logger.info("Recalculated parameter %s = (%f, %f)" % \
#                             (param.name,\
#                              eval(param.name).real, eval(param.name).imag))
        
        # Extract couplings from couplings that depend on fewer 
        # number of external parameters.
        couplings = []
        # Only take keys that contain running external parameters.
        ordered_keys = [key for key in self['couplings'].keys() if \
                            key != ('external',) and \
                            any([run_ext in key for run_ext in run_ext_params])]
        # Sort keys
        ordered_keys.sort(key=len)
        for key in ordered_keys:
            couplings += self['couplings'][key]

        # Now calculate all couplings
        # TO BE IMPLEMENTED use running alpha_s for aS-dependent couplings
        for coup in couplings:
            exec("globals()[\'%s\'] = %s" % (coup.name, coup.expr))
            if not eval(coup.name) and eval(coup.name) != 0:
                logger.warning("%s has no expression: %s" % (coup.name,
                                                             coup.expr))
            """logger.info("Recalculated coupling %s = (%f, %f)" % \
                        (coup.name,\
                         eval(coup.name).real, eval(coup.name).imag))"""


    def find_decay_groups(self):
        """ Find groups of particles which can decay into each other,
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
        self['decay_groups'] = [[]]

        particles = [p for p in self.get('particles') if \
                     p.get('pdg_code') not in self.sm_ids]

        for particle in particles:
            # Check if particles is already in a decay group
            if particle not in sum(self['decay_groups'], []):
                # Insert particle in new decay group
                self['decay_groups'].append([particle])
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
                                     self['decay_groups'][0] or \
                                     not p.get('is_part') and \
                                     self.get_particle(p.get('pdg_code')) in \
                                     self['decay_groups'][0])]
            group_index = [i for (i, g) in enumerate(self['decay_groups']) \
                           if particle in g][0]

            if len(non_sm_particles) == 0:
                # The decay group of this particle is the SM group
                if group_index > 0:
                    group = self['decay_groups'].pop(group_index)
                    self['decay_groups'][0].extend(group)
                    
            elif len(non_sm_particles) == 1:
                # The other particle should be in my decay group
                particle2 = non_sm_particles[0]
                if not particle2.get('is_part'):
                    particle2 = self.get_particle(particle2.get_anti_pdg_code())
                if particle2 in self['decay_groups'][group_index]:
                    # This particle is already in this decay group,
                    # and has been treated.
                    continue
                elif particle2 in sum(self['decay_groups'], []):
                    # This particle is in a different decay group - merge
                    group_index2 = [i for (i, g) in \
                                    enumerate(self['decay_groups']) \
                                    if particle2 in g][0]
                    group = self['decay_groups'].pop(max(group_index,
                                                      group_index2))
                    self['decay_groups'][min(group_index, group_index2)].\
                                                        extend(group)
                else:
                    # Add particle2 to this decay group
                    self['decay_groups'][group_index].append(particle2)

            elif len(non_sm_particles) > 1:
                # Check if any of the particles are not already in any
                # decay group. If there are any, let another particle
                # take care of this interaction instead, later on.

                non_checked_particles = [p for p in non_sm_particles if \
                                         (p.get('is_part') and not p in \
                                          sum(self['decay_groups'], []) or \
                                          not p.get('is_part') and not \
                                          self.get_particle(\
                                                     p.get_anti_pdg_code()) in \
                                          sum(self['decay_groups'], []))
                                         ]

                if not non_checked_particles:
                    # All particles have been checked. Analyze interaction.

                    if len(non_sm_particles) == 2:
                        # Are any of the particles in my decay group already?
                        this_group_particles = [p for p in non_sm_particles \
                                                if p in self['decay_groups'][\
                                                                   group_index]]
                        if len(this_group_particles) == 2:
                            # There can't be any conserved quantum
                            # number! Should be SM group!
                            group = self['decay_groups'].pop(group_index)
                            self['decay_groups'][0].extend(group)
                            continue
                        elif len(this_group_particles) == 1:
                            # One particle is in the same group as this particle
                            # The other (still non_sm yet) must be SM group.
                            particle2 = [p for p in non_sm_particles \
                                             if p != this_group_particles[0]][0]
                            if not particle2.get('is_part'):
                                particle2 = self.get_particle(particle2.get_anti_pdg_code())

                            group_index2 = [i for (i, g) in \
                                                enumerate(self['decay_groups'])\
                                                if particle2 in g][0]
                            group_2 = self['decay_groups'].pop(group_index2)
                            self['decay_groups'][0].extend(group_2)

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
                                            enumerate(self['decay_groups']) \
                                            if particle1 in g][0]

                            # If the other non_sm_particle is in the same group
                            # as particle1, try to merge this particle to SM
                            if non_sm_particles[1] in \
                                    self['decay_groups'][group_index1]:
                                if group_index > 0:
                                    group = self['decay_groups'].pop(group_index)
                                    self['decay_groups'][0].extend(group)

                    if len(non_sm_particles) == 3:
                        # Are any of the particles in my decay group already?
                        this_group_particles = [p for p in non_sm_particles \
                                                if p in self['decay_groups'][\
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
                                            enumerate(self['decay_groups']) \
                                            if particle2 in g][0]
                            group = self['decay_groups'].pop(max(group_index,
                                                              group_index2))
                            self['decay_groups'][min(group_index, group_index2)].\
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
                                            enumerate(self['decay_groups']) \
                                            if particle1 in g][0]
                            particle2 = other_group_particles[0]
                            if not particle2.get('is_part'):
                                particle2 = self.get_particle(\
                                                  particle2.get_anti_pdg_code())
                            group_index2 = [i for (i, g) in \
                                            enumerate(self['decay_groups']) \
                                            if particle2 in g][0]

                            if group_index1 != group_index2:
                                # Merge groups
                                group = self['decay_groups'].pop(max(group_index1,
                                                                  group_index2))
                                self['decay_groups'][min(group_index1,
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

    def find_decay_groups_general(self):
        """Iteratively find decay groups, valid for vertices in all orders
           SM particle is defined as all MASSLESS particles.
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
        
        # Setup the SM particles and initial decay_groups, reduced_interactions
        self['decay_groups'] = [[]]
        # self['reduced_interactions'] contains keys in 'id' as interaction id,
        # 'particle' as the reduced particle content, and 'groupid_list' as
        # the decay group id list
        self['reduced_interactions'] = []
        sm_ids = []

        # Setup the original 'SM' particles, i.e. particle without mass.
        sm_ids = [p.get('pdg_code') for p in self.get('particles')\
                      if abs(eval(p.get('mass'))) == 0.]
        self['decay_groups'] = [[p for p in self.get('particles')\
                                     if abs(eval(p.get('mass'))) == 0.]]

        #Read the interaction information and setup
        for inter in self.get('interactions'):
            temp_int = {'id':inter.get('id'), 'particles':[]}
            for part in inter['particles']:
                #If this particle is anti-particle, convert it.
                if not part.get('is_part'):
                    part = self.get_particle(part.get_anti_pdg_code())

                #Read this particle if it is not in SM
                if not part.get('pdg_code') in sm_ids and \
                   not part in self['decay_groups'][0]:
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
                self['decay_groups'][0].append(part)

            # Finally, append only interaction with nonzero particles
            # to reduced_interactions.
            if len(temp_int['particles']):
                self['reduced_interactions'].append(temp_int)
            # So interactions in reduced_interactions are all 
            # with non-zero particles in this stage

        # Now start the iterative interaction reduction
        change = True
        while change:
            change = False
            for inter in self['reduced_interactions']:
                #If only two particles in inter, they are in the same group
                if len(inter['particles']) == 2:
                    #If they are in different groups, merge them.
                    #Interaction is useless.

                    # Case for the particle is in decay_groups
                    if inter['particles'][0] in sum(self['decay_groups'], []):
                        group_index_0 =[i for (i,g) in\
                                        enumerate(self['decay_groups'])\
                                        if inter['particles'][0] in g][0]

                        # If the second one is also in decay_groups, merge them.
                        if inter['particles'][1] in sum(self['decay_groups'], []):
                            if not inter['particles'][1] in \
                                    self['decay_groups'][group_index_0]:
                                group_index_1 =[i for (i,g) in \
                                                enumerate(self['decay_groups'])\
                                                if inter['particles'][1] 
                                                in g][0]
                                # Remove the outer group
                                group_1 = self['decay_groups'].pop(max(\
                                          group_index_0, group_index_1))
                                # Merge with the inner one
                                self['decay_groups'][min(group_index_0, \
                                                 group_index_1)].extend(group_1)
                        # The other one is no in decay_groups yet
                        # Add inter['particles'][1] to the group of 
                        # inter['particles'][0]
                        else:
                            self['decay_groups'][group_index_0].append(
                                inter['particles'][1])
                    # Case for inter['particles'][0] is not in decay_groups yet.
                    else:
                        # If only inter[1] is in decay_groups instead, 
                        # add inter['particles'][0] to its group.
                        if inter['particles'][1] in sum(self['decay_groups'], []):
                            group_index_1 =[i for (i,g) in \
                                            enumerate(self['decay_groups'])\
                                            if inter['particles'][1] in g][0]
                            # Add inter['particles'][0]
                            self['decay_groups'][group_index_1].append(
                                inter['particles'][0])

                        # Both are not in decay_groups
                        # Add both particles to decay_groups
                        else:
                            self['decay_groups'].append(inter['particles'])

                    # No matter merging or not the interaction is useless now. 
                    # Kill it.
                    self['reduced_interactions'].remove(inter)
                    change = True

                # If only one particle in this interaction,
                # this particle must be SM-like group.
                elif len(inter['particles']) == 1:
                    if inter['particles'][0] in sum(self['decay_groups'], []):
                        group_index_1 =[i for (i,g) in \
                                        enumerate(self['decay_groups'])\
                                        if inter['particles'][0] in g][0]
                        # If it is not, merge it with SM.
                        if group_index_1 > 0:
                            self['decay_groups'][0].\
                                extend(self['decay_groups'].pop(group_index_1))

                    # Inter['Particles'][0] not in decay_groups yet, 
                    # add it to SM-like group
                    else:
                        self['decay_groups'][0].extend(inter['particles'])

                    # The interaction is useless now. Kill it.
                    self['reduced_interactions'].remove(inter)
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
                                              enumerate(self['decay_groups']) \
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
                        self['reduced_interactions'].remove(inter)

                # Start a new iteration...


        # Check if there is any particle that cannot be classified.
        # Such particle is in the group of its own.
        for part in self.get('particles'):
            if not part in sum(self['decay_groups'], []) and \
                    not part.get('pdg_code') in sm_ids:
                self['decay_groups'].append([part])

        # For conveniences, record the decay_group id in particles of 
        # reduced interactions
        for inter in self['reduced_interactions']:
            inter['groupid_list'] = [[i \
                                          for i, g in \
                                          enumerate(self['decay_groups']) \
                                          if p in g][0] \
                                         for p in inter['particles']]

        return self['decay_groups']

    def find_stable_particles(self):
        """ Find stable particles that are protected by parity conservation
            (massless particle is not included). 
            Algorithm:
            1. Find the lightest massive particle in each group.
            2. For each reduced interaction, the group of the particle which
               lightest mass is greater than the sum of all other particles 
               is not (may not be) stable.
            3. Replace the lightest mass of unstable group as its decay products
            4. Repeat 2., until no replacement can be made.   

        """

        # If self['decay_groups'] is None, find_decay_groups first.
        if not self['decay_groups']:
            self.find_decay_groups_general()

        # The list for the lightest particle for each groups
        # The first element is preserved for the SM-like group.
        stable_candidates = [[]]
        # The list for the lightest mass for each groups
        lightestmass_list = [0.]

        # Set the massless particles into stable_particles
        self['stable_particles'] = [[]]
        for p in self.get('particles'):
            if abs(eval(p.get('mass'))) == 0. :
                p.set('is_stable', True)
                self['stable_particles'][-1].append(p)

        # Find lightest particle in each group.
        # SM-like group is excluded.
        for group in self['decay_groups'][1:]:
            # The stable particles of each group is described by a sublist
            # (take degeneracy into account). Group index is the index in the
            # stable_candidates. Suppose the lightest particle is the 1st one.
            stable_candidates.append([group[0]])
            for part in group[1:]:
                # If the mass is smaller, replace the the list.
                if abs(eval(part.get('mass'))) < \
                        abs(eval(stable_candidates[-1][0].get('mass'))) :
                    stable_candidates[-1] = [part]
                # If degenerate, append current particle to the list.
                elif abs(eval(part.get('mass'))) == \
                        abs(eval(stable_candidates[-1][0].get('mass'))) :
                    stable_candidates[-1].append(part)
            # Record the lightest mass into lightestmass_list
            lightestmass_list.append(abs(eval(stable_candidates[-1][0].get('mass'))))


        # Deal with the reduced interaction
        change = True
        while change:
            change = False
            for inter in self['reduced_interactions']:
                # Find the minial mass for each particle
                masslist = [lightestmass_list[inter['groupid_list'][i]] \
                                for i in range(len(inter['particles']))]
                
                # Replace the minial mass to possible decay products
                for i, m in enumerate(masslist):
                    if 2*m > sum(masslist):
                        # Clear the stable_candidates in this group
                        stable_candidates[inter['groupid_list'][i]] = []
                        # The lightest mass becomes the mass of decay products.
                        lightestmass_list[inter['groupid_list'][i]] = \
                            sum(masslist)-m
                        change = True
                        break

        # Append the resulting stable particles
        for stable_particlelist in stable_candidates:
            if stable_particlelist:
                self['stable_particles'].append(stable_particlelist)

        # Set the is_stable label for particles in the stable_particles
        for p in sum(self['stable_particles'], []):
            p.set('is_stable', True)
            self.get_particle(p.get_anti_pdg_code()).set('is_stable', True)

        # Run the advance find_stable_particles to ensure that
        # all stable particles are found
        self.find_stable_particles_advance()

        return self['stable_particles']

    def find_stable_particles_advance(self):
        """ Find all stable particles. 
            Algorithm:
            1. For each interaction, if one particle has mass larger than 
               the other, than this particle's mass is replaced by 
               the sum of its decay products' masses. 
               The 'is_stable' label of this particle is False.
               
            2. Repeat 1., until no change was made after the whole check.
            3. Particles that have never been labeled as unstable are now
               stable particles.
        """

        # Record the mass of all particles
        # Record whether particles can decay through stable_list
        mass={}
        stable_list = {}
        for part in self.get('particles'):
            mass[part.get('pdg_code')] = abs(eval(part.get('mass')))
            stable_list[part.get('pdg_code')] = True

        # Record minimal mass for avioding round-off error.
        m_min = min(min([m for i,m in mass.items() if m > 0.]), 10**(-6))

        # Start the iteration
        change = True
        while change:
            change = False
            for inter in self['interactions']:
                total_m = sum([mass[p.get('pdg_code')] \
                                   for p in inter['particles']])

                # Skip interaction with total_m = 0
                if total_m == 0.:
                    continue

                # Find possible decay for each particle
                for part in inter['particles']:
                    # If not stable particle yet.
                    if not part.get('is_stable'):
                        # This condition is to prevent round-off error.
                        if (2*mass[part.get('pdg_code')]-total_m) > \
                                10**3*mass[part.get('pdg_code')]*\
                                sys.float_info.epsilon:
                            mass[part.get('pdg_code')] = \
                                total_m - mass[part.get('pdg_code')]
                            part['is_stable'] = False
                            stable_list[part.get('pdg_code')] = False
                            change = True
                            break

        # Record the stable particle
        for part in self.get('particles'):
            if stable_list[part.get('pdg_code')]:               
                part.set('is_stable', True)
                self.get_particle(part.get_anti_pdg_code()).set('is_stable', 
                                                                True)
                if not part in sum(self['stable_particles'], []):
                    self['stable_particles'].append([part])


    def find_channels(self, part, max_partnum, min_br=0):
        """ Function that find channels for a particle.
            Call the function in DecayParticle."""
        part.find_channels(max_partnum, self, min_br)

    def find_all_channels(self, max_partnum, collect_helascalls = True,
                          generate_abstract=False, min_br=0):
        """ Function that find channels for all particles in this model.
            Call the function in DecayParticle.
            It also write a file to compare the decay width from 
            param_card and from the estimation of this module."""

        # If vertexlist has not been found before, run model.find_vertexlist
        if not self['vertexlist_found']:
            logger.info("Vertexlist of this model has not been searched."+ \
                "Automatically run the model.find_vertexlist()")
            self.find_vertexlist()

        # If vertexlist has not been found before, run model.find_vertexlist
        if generate_abstract and not self['abmodel_generated']:
            logger.info("AbstractModel for this model has not been generated."\
                            +"Automatically run the model.generate_abstract_model()")
            self.generate_abstract_model()

        # Find stable particles of this model
        self.get('stable_particles')
        logger.info("Found %s stable particles" % len(self['stable_particles']))

        # Run the width of all particles from 2-body decay so that the 3-body
        # decay could use the width from 2-body decay.
        for part in self.get('particles'):
            # Skip search if this particle is stable
            if part.get('is_stable'):
                continue

            # Recalculating parameters and coupling constants 
            self.running_externals(abs(eval(part.get('mass'))))
            self.running_internals()
            #logger.info("Find 2-body channels of %s" %part.get('name'))
            part.find_channels_nextlevel(self, min_br)

            # Update the total width for the later error estimation
            part.update_decay_attributes(True, False, False)

            if generate_abstract:
                self.generate_abstract_amplitudes(part, 2)
            if collect_helascalls:
                self.collect_helascalls(part, 2)
 
        for part in self.get('particles'):
            # Skip search if this particle is stable
            if part.get('is_stable'):
                continue
                
            # Recalculating parameters and coupling constants 
            self.running_externals(abs(eval(part.get('mass'))))
            self.running_internals()

            if max_partnum > 2:
                # After recalculating the parameters, find the channels to the
                # requested level.
                for clevel in range(3, max_partnum+1):
                    logger.info("Find %d-body channels of %s" %(clevel,
                                                                part.get('name')))
                    part.find_channels_nextlevel(self)

                    if generate_abstract:
                        self.generate_abstract_amplitudes(part, clevel)
                    
                    if collect_helascalls:
                        self.collect_helascalls(part, clevel)

            # update the decay attributes for both max_partnum >2 or == 2.
            # The update should include branching ratios and apx_decaywidth_err
            # So the apx_decaywidth_err(s) are correct even for max_partnum ==2.
            part.update_decay_attributes(True, True, True, self)


    def find_all_channels_smart(self, precision, 
                                collect_helascalls = True,
                                generate_abstract=False):
        """ Function that find channels for all particles in this model.
            Decay channels more than three final particles are searched
            when the precision is not satisfied.
            If generate_abstract = True, the abstract amplitudes will be
            generated."""

        # Raise error if precision is not a float
        if not isinstance(precision, float):
            raise self.PhysicsObjectError, \
                "The precision %s should be float type." % str(precision)

        # If vertexlist has not been found before, run model.find_vertexlist
        if not self['vertexlist_found']:
            logger.info("Vertexlist of this model has not been searched."+ \
                "Automatically run the model.find_vertexlist()")
            self.find_vertexlist()

        # If vertexlist has not been found before, run model.find_vertexlist
        if generate_abstract and not self['abmodel_generated']:
            logger.info("AbstractModel for this model has not been generated."\
                            +"Automatically run the model.generate_abstract_model()")
            self.generate_abstract_model()

        # Find stable particles of this model
        self.get('stable_particles')

        # Run the width of all particles from 2-body decay so that the 3-body
        # decay could use the width from 2-body decay.
        for part in self.get('particles'):
            # Skip search if this particle is stable
            if part.get('is_stable'):
                logger.info("Particle %s is stable." %part['name'] +\
                                "No channel search will not proceed.")
                continue

            # Recalculating parameters and coupling constants 
            self.running_externals(abs(eval(part.get('mass'))))
            self.running_internals()

            logger.info("Find 2-body channels of %s" %part.get('name'))
            part.find_channels_nextlevel(self)
            if generate_abstract:
                self.generate_abstract_amplitudes(part, 2)

            if collect_helascalls:
                self.collect_helascalls(part, 2)

        # Search for higher final particle states, if the precision
        # is not satisfied.
        for part in self.get('particles'):
            # Skip search if this particle is stable
            if part.get('is_stable'):
                continue

            # Update the decaywidth_err
            part.update_decay_attributes(False,True,False, self)

            # If the error (ratio to apx_decaywidth) is larger then precision,
            # find next level channels.
            # Running coupling constants first.
            if part.get('apx_decaywidth_err') > precision:
                self.running_externals(abs(eval(part.get('mass'))))
                self.running_internals()

            clevel = 3
            while part.get('apx_decaywidth_err') > precision:
                logger.info("Find %d-body channels of %s" \
                                %(clevel,
                                  part.get('name')))
                part.find_channels_nextlevel(self)
                if generate_abstract:
                    self.generate_abstract_amplitudes(part, clevel)

                if collect_helascalls:
                    self.collect_helascalls(part, clevel)

                # Note that the width is updated automatically in the
                # find_nextlevel
                part.update_decay_attributes(False,True,False, self)
                clevel += 1

            # Finally, update the branching ratios
            part.update_decay_attributes(False, False, True)         



    def write_summary_decay_table(self, name=''):
        """ Write a table to list the total width of all the particles
            and compare to the value in param_card."""
    
        # Write the result to decaywidth_MODELNAME.dat in 'mg5decay' directory
        path = os.path.join(MG5DIR, 'mg5decay')
        if not name:
            fdata = open(os.path.join(path, 
                                      (self['name']+'_decay_summary.dat')),
                         'w')
            logger.info("\nWrite decay width summary to %s \n" \
                            % str(os.path.join(path,
                                               (self['name']+'_decay_summary.dat'))))

        elif isinstance(name, str):
            fdata = open(os.path.join(path, name),'w')
            logger.info("\nWrite decay width summary to %s \n" \
                            % str(os.path.join(path, name)))

        else:
            raise PhysicsObjectError,\
                "The file name of the decay table must be str." % str(name)

        summary_chart = ''
        summary_chart = (str('# DECAY WIDTH COMPARISON \n') +\
                            str('# model: %s \n' %self['name']) +\
                            str('#'*80 + '\n')+\
                            str('#Particle ID    card value     apprx. value  ratio') +\
                            str('   level    err \n')
                        )

        for part in self.get('particles'):
            # For non-stable particles
            if not part.get('is_stable'):
                # For width available in the param_card.
                try:
                    summary_chart +=(str('#%11d    %.4e     %.4e    %4.2f  %3d        %.2e\n'\
                                            %(part.get('pdg_code'), 
                                              self['decaywidth_list']\
                                                  [(part.get('pdg_code'), True)],
                                              part['apx_decaywidth'],
                                              part['apx_decaywidth']/self['decaywidth_list'][(part.get('pdg_code'), True)],
                                              part.get_max_level(),
                                              part['apx_decaywidth_err']
                                              )))
                # For width not available, do not calculate the ratio.
                except KeyError:
                    summary_chart += (str('#%11d    %.4e     %.4e    %s\n'\
                                             %(part.get('pdg_code'), 
                                               0.,
                                               part['apx_decaywidth'],
                                               'N/A')))
                # For width in param_card is zero.
                except ZeroDivisionError:
                    summary_chart += (str('#%11d    %.4e     %.4e    %s\n'\
                                             %(part.get('pdg_code'), 
                                               0.,
                                               part['apx_decaywidth'],
                                               'N/A')))
            # For stable particles
            else:
                try:
                    if abs(self['decaywidth_list'][(part.get('pdg_code'), True)]) == 0.:
                        ratio = 1
                    else:
                        ratio = 0
                    summary_chart += (str('#%11d    %.4e     %s    %4.2f\n'\
                                             %(part.get('pdg_code'), 
                                               self['decaywidth_list']\
                                                   [(part.get('pdg_code'), True)],
                                               'stable    ',
                                               ratio)))
                except KeyError:
                    summary_chart += (str('#%11d    %.4e     %s    %s\n'\
                                             %(part.get('pdg_code'), 
                                               0.,
                                               'stable    ',
                                               '1'
                                               )))
        # Write the summary_chart into file
        fdata.write(summary_chart)
        fdata.close()


    def write_decay_table(self, mother_card_path, format='normal',name = ''):
        """ Functions that write the decay table of all the particles 
            in this model that including the channel information and 
            branch ratio (call the estimate_width_error automatically 
            in the execution) in a file.
            format:
                normal: write only amplitudes
                cmp: add ratio of decay_width to the value in MG4 param_card
                full: also write the channels in each amplitude."""

        # The list of current allowing formats
        allow_formats = ['normal','full','cmp']

        # Raise error if format is wrong
        assert format in allow_formats, \
                "The format must be \'normal\' or \'full\' or \'cmp\'." \
                % str(name)

        # Write the result to decaywidth_MODELNAME.dat in 'mg5decay' directory
        path = os.path.join(MG5DIR, 'mg5decay')

        if not name:
            if format == 'full':
                fdata = open(os.path.join(path,
                                          (self['name']+'_decaytable_full.dat')),
                             'w')
                logger.info("\nWrite full decay table to %s\n"\
                                %str(os.path.join(path,
                                                  (self['name']+'_decaytable_full.dat'))))
            else:
                fdata = open(os.path.join(path,
                                          (self['name']+'_decaytable.dat')),
                             'w')
                logger.info("\nWrite %s decay table to %s\n"\
                                %(format, 
                                  str(os.path.join(path,
                                          (self['name']+'_decaytable.dat')))))

        elif isinstance(name, str):
            fdata = open(os.path.join(path, name),'w')
            logger.info("\nWrite %s decay table to %s\n"\
                            %(format, 
                              str(os.path.join(path,
                                               name))))

        else:
            raise PhysicsObjectError,\
                "The file name of the decay table must be str." % str(name)

        # Write the param_card used first
        fdata0 = open(mother_card_path, 'r')
        fdata.write(fdata0.read())
        fdata0.close()

        # Write header of the table
        spart = ''
        nonspart = ''
        summary_chart = ''
        seperator = str('#'*80 + '\n')
        fdata.write('\n' + seperator + '#\n'*2 +\
                        str('##    EST. DECAY TABLE    ## \n') +\
                        '#\n'*2 + seperator)

        # Header of summary data
        summary_chart = (str('# DECAY WIDTH COMPARISON \n') +\
                            str('# model: %s \n' %self['name']) +\
                            str('#'*80 + '\n')+\
                            str('#Particle ID    card value     apprx. value  ratio') +\
                            str('   level    err \n')
                        )
        # Header of stable particle output
        spart = ('\n' + seperator + \
                     '# Stable Particles \n'+ \
                     seperator+ \
                     '#%8s    Predicted \n' %'ID')

        # Output order is based on pdg code
        pids = sorted([p.get_pdg_code() for p in self['particles']])

        for pid in pids:
            p = self.get_particle(pid)

            # Write the table only for particles with finite width.
            if p.get('apx_decaywidth'):
                nonspart += p.decaytable_string(format)
                # Try to calculate the ratio in summary_chart
                try:
                    summary_chart +=(str('#%11d    %.4e     %.4e    %4.2f  %3d        %.2e\n'\
                                             %(p.get('pdg_code'), 
                                               self['decaywidth_list']\
                                                  [(p.get('pdg_code'), True)],
                                               p['apx_decaywidth'],
                                               p['apx_decaywidth']/self['decaywidth_list'][(p.get('pdg_code'), True)],
                                               p.get_max_level(),
                                               p['apx_decaywidth_err']
                                              )))
                # For width not available, do not calculate the ratio.
                except KeyError:
                    summary_chart += (str('#%11d    %.4e     %.4e    %s\n'\
                                              %(p.get('pdg_code'), 
                                                0.,
                                                p['apx_decaywidth'],
                                                'N/A')))
                # For width in param_card is zero.
                except ZeroDivisionError:
                    summary_chart += (str('#%11d    %.4e     %.4e    %s\n'\
                                              %(p.get('pdg_code'), 
                                                0.,
                                                p['apx_decaywidth'],
                                                'N/A')))

            else:
                # If width = 0.,
                # see if the stable property is predicted.
                if p.get('is_stable'):
                    spart += str('#%8d    %9s \n' % (p.get('pdg_code'), 'Yes'))
                else:
                    spart += str('#%8d    %9s \n' % (p.get('pdg_code'), 'No'))

                # Try to calculate the ratio if there is reference width
                try:
                    if abs(self['decaywidth_list'][(p.get('pdg_code'), True)]) == 0.:
                        ratio = 1
                    else:
                        ratio = 0
                    summary_chart += (str('#%11d    %.4e     %s    %4.2f\n'\
                                              %(p.get('pdg_code'), 
                                                self['decaywidth_list']\
                                                    [(p.get('pdg_code'), True)],
                                                'stable    ',
                                                ratio)))

                # If no width available, write the ratio as 1
                except KeyError:
                    summary_chart += (str('#%11d    %.4e     %s    %s\n'\
                                             %(p.get('pdg_code'), 
                                               0.,
                                               'stable    ',
                                               '1'
                                               )))
                    
        # Print summary_chart, stable particles, and finally unstable particles
        fdata.write(summary_chart)
        fdata.write(spart)
        fdata.write(nonspart)
        fdata.close()


    def find_nextlevel_ratio(self):
        """ Find the ratio of matrix element square for channels decay to
            next level."""

        pass


    # Helper function to review helas calls
    def write_helas_collection(self, name = ''):
        """ Functions that write the helascalls of the model, attached with
            a list of actual diagrams that belong to the helas call. """


        # Write the result to decaywidth_MODELNAME.dat in 'mg5decay' directory
        path = os.path.join(MG5DIR, 'mg5decay')

        if not name:
            fdata = open(os.path.join(path,
                                      (self['name']+'_helascollection.dat')),
                         'w')
            logger.info("\nWrite %s Helas collection to %s\n"\
                            %(format, 
                              str(os.path.join(path,
                                               (self['name']+'_helascollection.dat')))))

        elif isinstance(name, str):
            fdata = open(os.path.join(path, name),'w')
            logger.info("\nWrite %s Helas collection to %s\n"\
                            %(format, 
                              str(os.path.join(path,
                                               name))))

        else:
            raise PhysicsObjectError,\
                "The file name of the decay table must be str." % str(name)


        collection = ''

        # Output order is based on pdg code
        pids = sorted([p.get_pdg_code() for p in self['particles']])
        
        for key in sorted(self['helascalls'].keys()):
            collection += '%d-body decay: %d calls\n' %(key, len(self.get_helascalls(key)))
            for num, call in enumerate(self.get_helascalls(key)):
                collection += call.nice_string(decay_info=False)+'\n'
                for pid in pids:
                    p = self.get_particle(pid)
                    for amp in p.get_amplitudes(key):
                        for c in amp['diagrams']:
                            if c['helas_number'] == num:
                                collection += "   "+c.nice_string(decay_info=False)+'\n'
            

                    
        # Print collection
        fdata.write(collection)
        fdata.close()


    # Helper Function for reading MG4 param_card
    # And compare with our apx_decaywidth
    def read_MG4_param_card_decay(self, param_card):
        """Read the decay width in MG4 param_card and 
           compare the width with our estimation."""

        if not os.path.isfile(param_card):
            raise MadGraph5Error, \
                "No such file %s" % param_card
    
        # Read in param_card
        logger.info("\nRead MG4 param_card: %s \n" % str(param_card))
        param_lines = open(param_card, 'r').read().split('\n')

        # Define regular expressions
        re_decay = re.compile(\
            "^decay\s+(?P<pid>\d+)\s+(?P<value>-*\d+\.\d+e(\+|-)\d+)\s*")
        re_two_body_decay = re.compile(\
            "^\s+(?P<br>-*\d+\.\d+e(\+|-)\d+)\s+(?P<nda>\d+)\s+(?P<pid1>-*\d+)\s+(?P<pid2>-*\d+)")
        re_three_body_decay = re.compile(\
            "^\s+(?P<br>-*\d+\.\d+e(\+|-)\d+)\s+(?P<nda>\d+)\s+(?P<pid1>-*\d+)\s+(?P<pid2>-*\d+)\s+(?P<pid3>-*\d+)")

        # Define the decay pid, total width
        pid = 0
        total_width = 0

        # Go through lines in param_card
        for line in param_lines:
            if not line.strip() or line[0] == '#':
                continue
            line = line.lower()
            # Look for decay blocks
            decay_match = re_decay.match(line)
            if decay_match:
                pid = int(decay_match.group('pid'))
                total_width = float(decay_match.group('value'))
                self['decaywidth_list'][(pid, True)] = total_width
                continue
            # If no decay pid available, skip this line.
            if not pid:
                continue

            two_body_match = re_two_body_decay.match(line)
            three_body_match = re_three_body_decay.match(line)

            # Check three_body first!
            # Otherwise it will always to be two body.
            if three_body_match:
                # record the pids and br
                pid1 = int(three_body_match.group('pid1'))
                pid2 = int(three_body_match.group('pid2'))
                pid3 = int(three_body_match.group('pid3'))

                br = float(three_body_match.group('br'))
                final_ids = [pid1, pid2, pid3]
                amp = self.get_particle(pid).get_amplitude(final_ids)
                # If amplitude is found, record the ratio
                if amp:
                    amp['exa_decaywidth'] = br*total_width
                # If not found, show this info
                else:
                    logger.info('No amplitude for %d -> %d %d %d is found.' %\
                                       (pid, pid1, pid2, pid3))

                # Jump to next line. Do not match the two_body_decay
                continue

            # If not three-body, check two-body
            if two_body_match:
                # record the pids and br
                pid1 = int(two_body_match.group('pid1'))
                pid2 = int(two_body_match.group('pid2'))
                br = float(two_body_match.group('br'))
                final_ids = [pid1, pid2]
                amp = self.get_particle(pid).get_amplitude(final_ids)
                # If amplitude is found, record the ratio
                if amp:
                    amp['exa_decaywidth'] = br*total_width
                # If not found, show this info
                else:
                    logger.info('No amplitude for %d -> %d %d is found.' %\
                                       (pid, pid1, pid2))

                # Jump to next line. Do not match the three_body_decay
                continue


#===============================================================================
# Channel: A specialized Diagram object for decay
#===============================================================================

"""parameters for estimating the phase space area"""
c_psarea = 0.8

class Channel(base_objects.Diagram):
    """Channel: a diagram that describes a certain decay channel
                with on shell condition, apprximated matrix element, 
                phase space area, and decay width.
                There are several helper static methods.
                The check_idlegs will return the identical legs of the
                given vertex. The check_channels_equiv will check the
                equivalence of two channels.                
    """

    sorted_keys = ['vertices',
                   'orders',
                   'onshell', 'ini_pid', 'final_legs',
                   'has_idpart', 'id_part_list',
                   'tag', 'helastag', 'helas_number', 'std_diagram',
                   'apx_matrixelement_sq', 'apx_psarea', 'apx_decaywidth',
                   'apx_decaywidth_nextlevel', 'apx_width_calculated']#,
                   #'potential_gauge_dependence']

    def default_setup(self):
        """Default values for all properties"""
        self['vertices'] = base_objects.VertexList()
        self['orders'] = {}
        
        # New properties
        self['onshell'] = 0
        self['ini_pid'] = 0
        self['final_legs'] = base_objects.LegList()

        # This property denotes whether the channel has 
        # identical particles in it.
        self['has_idpart'] = False

        # (real) DiagramTag
        self['tag'] = []
        # IdentifyHelasTag
        self['helastag'] = []
        # the number of the corresponding helas calls
        self['helas_number'] = None
        # diagram written by IdentifyHelasTag
        self['std_diagram'] = None

        # old properties for check_channels_equiv, removable
        # The position of the identicle particles with pid as keys.
        self['id_part_list'] = {}
        self['final_mass_list'] = 0

        # Decay width related properties.
        self['apx_matrixelement_sq'] = 0.
        self['s_factor'] = 1
        self['apx_psarea'] = 0.
        self['apx_decaywidth'] = 0.
        # branch ratio is multiply by 100.
        self['apx_decaywidth_nextlevel'] = 0.
        self['apx_width_calculated'] = False

        # Potential gauge dependence
        #self['potential_gauge_dependence'] = False

        # Properties for abstract Channel
        # [pseudo_ab_inter_ids, pseudo_ab_intermediate_pids, pseudo_ab_finalids]
        # generated by add_ab_diagram in AbstractModel,
        # finalids is in the order of final_legs
        self['abstract_type'] = [[], [], []]
        self['fermionfactor'] = 1
        

    def filter(self, name, value):
        """Filter for valid diagram property values."""
        
        if name in ['apx_matrixelement_sq', 'apx_psarea', 
                    'apx_decaywidth', 'apx_br',
                    'apx_decaywidth_nextlevel']:
            if not isinstance(value, float):
                raise self.PhysicsObjectError, \
                    "Value %s is not a float" % str(value)
        
        if name == 'onshell' or name == 'has_idpart' or \
                name == 'apx_width_calculated':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid onshell condition." % str(value)

        return super(Channel, self).filter(name, value)
    
    def get(self, name, model=None):
        """ Check the onshell condition before the user get it. 
            And recalculate the apx_decaywidth_nextlevel if the 
            model is provided.
        """

        if name == 'onshell':
            logger.info("It is suggested to get onshell property from get_onshell function")

        if name == 'apx_decaywidth_nextlevel' and model:
            return self.get_apx_decaywidth_nextlevel(model)

        if name == 'tag' and not self['tag']:
            self['tag'] = diagram_generation.DiagramTag(self)

        if name == 'helastag' and not self['helastag'] and model:
            self['helastag'] = IdentifyHelasTag(self, model)
            

        return super(Channel, self).get(name)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        return self.sorted_keys

    def calculate_orders(self, model):
        """Calculate the actual coupling orders of this channel,
           negative vertex id is interepret as positive one 
           (the CPT counterpart)."""

        coupling_orders = {}
        for vertex in self['vertices']:
            if vertex.get('id') == 0: continue
            vid = vertex.get('id')
            couplings = model.get('interaction_dict')[abs(vertex.get('id'))].\
                        get('orders')
            for coupling in couplings.keys():
                try:
                    coupling_orders[coupling] += couplings[coupling]
                except:
                    coupling_orders[coupling] = couplings[coupling]

        self.set('orders', coupling_orders)

    def nice_string(self, decay_info=True):
        """ Add width/width_nextlevel to the nice_string"""
        mystr = super(Channel, self).nice_string()
        if self['vertices'] and decay_info:
            if self['onshell']:
                mystr +=" (width = %.3e)" % self['apx_decaywidth']
            else:
                mystr +=" (est. further width = %.3e)" % self['apx_decaywidth_nextlevel']              

            #if self['potential_gauge_dependence']:
            #    mystr +="*PGD"

        return mystr

    def initial_setups(self, model, force):
        """ Setup useful properties for a new channel.
            N.B. 'has_idpart' is not implemented because there is a faster way
            to get it from connect_channel_vertex. """

        self.get_initial_id(model)
        self.get_final_legs(force)
        self.get_onshell(model)
        self.calculate_orders(model)

    def get_anti_initial_id(self):
        """ Return the anti id of initial particle"""

        return self.get('vertices')[-1].get('legs')[-1].get('id')

    def get_initial_id(self, model=None):
        """ Return the id of initial particle"""

        if self['ini_pid'] != 0:
            return self['ini_pid']
        elif model:
            self['ini_pid'] = model.get_particle(self.get_anti_initial_id()).get_anti_pdg_code()
            return self['ini_pid']
        else:
            raise self.PhysicsObjectError, "No model is provided to get initial id."

    def check_gauge_dependence(self, model):
        """ compare the vertex ids to see if anyone of them may be illed."""

        parts = [model.get_particle(l['id']) for l in self.get_final_legs()]
        initial = model.get_particle(self.get_anti_initial_id())
        for i,part in enumerate(parts):
            if part.get('spin') % 2 == 0:
                continue
            if model['parameter_dict'][part.get('mass')] == 0:
                base = [l['id'] for l in self.get_final_legs()]
                base.pop(i)
                base.sort()
                
                for decay in initial.get_channels(len(base), True):
                    tmp = [l['id'] for l in decay.get_final_legs()]
                    tmp.sort()
                    if base == tmp:
                        return False
        return True
            


    def get_final_legs(self, force=False):
        """ Return a list of the final state legs."""

        # Reset leglist of force = True
        if force:
            self['final_legs'] = base_objects.LegList()

        if not self['final_legs']:
            for vert in self.get('vertices'):
                for leg in vert.get('legs')[:-1]:
                    if not leg.get('number') in [l.get('number') \
                                                 for l in self['final_legs']]\
                                                 and leg.get('number') > 1:
                        self['final_legs'].append(leg)

        return self['final_legs']
        
    def get_onshell(self, model):
        """ Evaluate the onshell condition with the aid of get_final_legs"""
        if not isinstance(self['onshell'], bool):
            # Check if model is valid
            if not isinstance(model, base_objects.Model):
                raise self.PhysicsObjectError, \
                    "The argument %s must be a model." % str(model)

            self['final_mass_list'] =sorted([abs(eval(model.get_particle(l.get('id')).get('mass'))) \
                                                for l in self.get_final_legs()])
            # Use the mass of antiparticle for mother mass
            ini_mass = abs(eval(model.get_particle(self.get_anti_initial_id()).get('mass')))
            # ini_mass = ini_mass.real
            self['onshell'] = ini_mass > sum(self['final_mass_list'])

        return self['onshell']
    
    def has_goldstone(self, model):

            # Check if model is valid
            assert isinstance(model, base_objects.Model), "The argument %s must be a model." % str(model)
                    
            return any(model.get_particle(l.get('id'))['type'] =='goldstone' \
                       for l in self.get_final_legs())


    def get_helas_properties(self, model):
        """ return helastag and std_diagram, construct them if necessary. """

        if not self['helastag'] and model:
            self['helastag'] = IdentifyHelasTag(self, model)

        if not self['std_diagram'] and model:
            self['std_diagram'] = self['helastag'].diagram_from_tag(model)

        return self['helastag'], self['std_diagram']

    def get_fermion_factor(self, model):
        """ Get the fermion_factor, same as get_fermion_factor 
            in HelasAmplitude."""

        # Record the fermion_order using numbers as keys
        order_dict = {}

        # Setup the order_dict of external legs
        for l in self.get_final_legs():
            if model.get_particle(l.get('id')).is_fermion:
                order_dict[l.get('id')] = [l.get('number'), []]
            else:
                order_dict[l.get('id')] = []

        # The initial leg is transformed as outgoing legs
        ini_part = model.get_particle(self.get_initial_id(model))
        if ini_part.is_fermion():
            order_dict[ini_part.get_pdg_code()] = [1, []]
        else:
            order_dict[ini_part.get_pdg_code()] = []

        # Find fermion_order in vertices except the identical one and 
        # initial one
        for i, vert in enumerate(self['vertices'][:-1]):

            # If the vert ended up with fermion, find its fermion_mother
            fermion_mother = None
            if ini_part.is_fermion():
                if vert['id'] > 0:
                    inter = model.get_interaction(vert['id'])
                else:
                    inter = model.get_interaction(model['conj_int_dict'][vert['id']])
                pdg_codes = [p.get_pdg_code() for p in inter['particles']]
                ini_index = pdg_codes.index(ini_part.get_anti_pdg_code())
                # part_index is the index of partner in vertex['legs']
                if ini_index % 2 ==0:
                    part_index = ini_index
                    mother_code = pdg_codes[ini_index+1]
                else:
                    part_index = ini_index - 1
                    mother_code = pdg_codes[ini_index-1]

                fermion_mother = vert['legs'][part_index]

                # Check if this vertex has the right order
                if fermion_mother['id'] != mother_code:
                    raise PhysicsObjectError, \
                        "The order of particle in interaction is not standard."

            # Refind the fermion_order except for initial vertex
            if i != len(self['vertices'])-2:
                order_dict[vert['legs'][-1]['id']] = \
                    self.get_fermion_order(vert, order_dict, fermion_mother)

        # Find fermion_factor in initial vertex
        vert = self['vertices'][-2]
        if vert['id'] > 0:
            inter = model.get_interaction(vert['id'])
        else:
            inter = model.get_interaction(model['conj_int_dict'][vert['id']])
        pdg_codes = [p.get_pdg_code() for p in inter['particles']]
        
        # Construct helper_vertex with 
        # legs = [ final+ initial legs in initial vertex, auxiliary_initial_leg]
        help_legs = base_objects.LegList(\
            [base_objects.Leg({'id': code}) for code in pdg_codes])
        help_legs.append(base_objects.Leg({'id': 0}))
        helper_vertex = base_objects.Vertex({'legs': help_legs})

        # Using this helper_vertex to get final fermion_order
        final_order = self.get_fermion_order(helper_vertex, order_dict, None)

        self['fermionfactor'] = self.sign_flips_to_order(final_order)

    @staticmethod    
    def get_fermion_order(vert, order_dict, fermion_mother):
        """ Get the fermion_order. similar to the get_fermion_order in
            HelasWavefunction. """

        new_order = []

        # Include the order of mother first
        if fermion_mother:
            new_order.extend(order_dict[fermion_mother['id']][1])

        # fermion_order of the fermionic legs
        fermions = [order_dict[l['id']] \
                        for l in vert['legs'][:-1] \
                        if len(order_dict[l['id']]) == 2 and \
                        l != fermion_mother]
        # fermion_order of the bosonic legs
        bosons = [order_dict[l['id']] \
                      for l in vert['legs'][:-1] \
                      if len(order_dict[l['id']]) == 1]

        # include fermionic legs first
        for i in range(0, len(fermions), 2):
            new_order.append(fermions[i][0])
            new_order.append(fermions[i+1][0])
            new_order.extend(fermions[i][1])
            new_order.extend(fermions[i+1][1])
        # Then for bosons
        for b in bosons:
            new_order.extend(b)

        # Add mother_number for fermion
        if fermion_mother:
            return [order_dict[fermion_mother['id']][0], new_order]
        else:
            return new_order

    @staticmethod
    def sign_flips_to_order(self, fermions):
        """Gives the sign corresponding to the number of flips needed
        to place the fermion numbers in order.
        N.B. copy from helas_object.py """

        # Perform bubble sort on the fermions, and keep track of
        # the number of flips that are needed

        nflips = 0

        for i in range(len(fermions) - 1):
            for j in range(i + 1, len(fermions)):
                if fermions[j] < fermions[i]:
                    tmp = fermions[i]
                    fermions[i] = fermions[j]
                    fermions[j] = tmp
                    nflips = nflips + 1

        return (-1) ** nflips

    @staticmethod
    def check_idlegs(vert):
        """ Helper function to check if the vertex has several identical legs.
            If id_legs exist, return a dict in the following format,
            {particle_id: [leg_index1, index2, ...]}
            Otherwise return False.
        """
        lindex_dict = {}
        id_part_list = {}
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
                # Record the index of vertex, vertex id (interaction id),
                # leg id, and the list of leg index.
                id_part_list[key] = indexlist

        return id_part_list

    @staticmethod
    def check_channels_equiv(channel_a, channel_b):
        """ Helper function to check if any channel is indeed identical to
            the given channel. (This may happens when identical particle in
            channel.) Use DiagramTag for full comparison.""" 

        return channel_a.get('tag') == channel_b.get('tag')

    # OBSELETE
    def get_idpartlist(self):
        """ Get the position of identical particles in this channel.
            The format of id_part_list is a dictionary with the vertex
            which has identical particles, value is the particle id and
            leg index of identicle particles. Eg.
            id_part_list = {(vertex_index_1, vertex id, pid_1): 
                           [index_1, index_2, ..],
                           (vertex_index_1, vertex id, pid_2): 
                           [index_1, index_2, ..],
                           (vertex_index_2,...): ...}
        """

        if not self['id_part_list']:
            # Check each vertex by check_idlegs
            for vindex, vert in enumerate(self.get('vertices')):
                # Use the id_part_list given by check_idlegs
                id_part_list = Channel.check_idlegs(vert)
                if id_part_list:
                    for key, idpartlist in id_part_list.items():
                        # Record the id_part_list if exists.
                        self['id_part_list'][(vindex, vert.get('id'), 
                                             key)] = id_part_list[key]
                        self['has_idpart'] = True

        return self['id_part_list']

#    # Obselete, replaced by DiagramTag
#    @staticmethod
#    def check_channels_equiv_rec(channel_a, vindex_a, channel_b, vindex_b):
#        """ The recursive function to check the equivalence of channels 
#            starting from the given vertex point.
#            Algorithm:
#            1. Check if the two vertices are the same (in id).
#            2. Compare each the non-identical legs. Either they are both 
#               final legs or their decay chain are the same. The comparision
#               of decay chain is via recursive call of check_channels_equiv_rec
#            3. Check all the identical particle legs, try to match each leg of b
#               for the legs of a of each kind of identical particles. 
#               If a leg of b is fit for one leg of a, do not match this leg of a
#               to other legs of b
#               If any one leg of b cannot be matched, return False.
#            4. If the two channels are the same for all the non-identical legs
#               and are the same for every kind of identical particle,
#               the two channels are the same from the given vertices."""
#        
#        # If vindex_a or vindex_b not in the normal range of index
#        # convert it. (e.g. vindex_b = -1)
#        vindex_a = vindex_a % len(channel_a.get('vertices'))        
#        vindex_b = vindex_b % len(channel_b.get('vertices'))
#
#        # First compare the id of the two vertices.
#        # Do not compare them directly because the number property of legs
#        # may be different.
#        # If vertex id is the same, then the legs id are all the same!
#        if channel_a.get('vertices')[vindex_a]['id'] != \
#                channel_b.get('vertices')[vindex_b]['id']:
#            return False
#
#        # Obsolete when idetical vertex is removed.
#        """# If the vertex is the initial vertex, start from the next vertex
#        if vindex_a == len(channel_a.get('vertices'))-1 and \
#                vindex_b == len(channel_b.get('vertices'))-1 :
#            return Channel.check_channels_equiv_rec(channel_a, -2,
#                                                    channel_b, -2)"""
#        
#        # Find the list of identical particles
#        id_part_list_a=Channel.check_idlegs(channel_a.get('vertices')[vindex_a])
#
#        result = True
#        # For each leg, find their decay chain and compare them.
#        for i, leg_a in \
#                enumerate(channel_a.get('vertices')[vindex_a].get('legs')[:-1]):
#            # The two channels are equivalent as long as the decay chain 
#            # of the two legs must be the same if they are not part of
#            # the identicle particles.
#            if not leg_a.get('id') in id_part_list_a.keys():
#                # The corresponding leg in channel_b
#                leg_b = channel_b.get('vertices')[vindex_b].get('legs')[i]
#
#                # If the 'is final' is inconsistent between a and b,
#                # return False. 
#                # If both are 'is final', end the comparision of these two legs.
#                leg_a_isfinal =  leg_a in channel_a.get_final_legs()
#                leg_b_isfinal =  leg_b in channel_b.get_final_legs()
#                if leg_a_isfinal or leg_b_isfinal:
#                    if leg_a_isfinal and leg_b_isfinal:
#                        continue
#                    else:
#                        # Return false if one is final leg 
#                        # while the other is not.
#                        return False
#                # The case with both legs are not final needs
#                # further analysis
#
#                # Find the next vertex index of the decay chain of 
#                # leg_a and leg_b.
#                for j in range(vindex_a-1, -1, -1):
#                    v = channel_a.get('vertices')[j]
#                    if leg_a in v.get('legs'):
#                        new_vid_a = j
#                        break
#                    
#                for j in range(vindex_b-1, -1, -1):
#                    v = channel_b.get('vertices')[j]
#                    if leg_b in v.get('legs'):
#                        new_vid_b = j
#                        break
#                
#                # Compare the decay chains of the two legs.
#                # If they are already different, return False
#                if not Channel.check_channels_equiv_rec(channel_a, new_vid_a,
#                                                        channel_b, new_vid_b):
#                    return False
#
#        # If the check can proceed out of the loop of legs,
#        # the decay chain is all the same for non-identicle particles.
#        # Return True if there is no identical particles.
#        if not id_part_list_a:
#            return True
#
#
#        # Check each kind of identicle particles
#        for pid, indices_a in id_part_list_a.items():
#            indices_b = copy.copy(indices_a)
#            # Match each leg of channel_b to every leg of channel_a
#            for index_b in indices_b:
#                # Suppose the fit fail
#                this_leg_fit = False
#                # setup leg_b                
#                leg_b = channel_b.get('vertices')[vindex_b].get('legs')[index_b]
#                # Search for match leg in legs from indices_a
#                for i, index_a in enumerate(indices_a):
#                    # setup leg_a
#                    leg_a = channel_a.get('vertices')[vindex_a].get('legs')[index_a]
#                    # Similar to non-identicle particles, but
#                    # we could not return False when one is final leg while
#                    # the other is not since this could due to the wrong
#                    # config used now.
#                    # If the leg is fit (both are final) stop the match of this
#                    # leg.
#                    leg_a_isfinal =  leg_a in channel_a.get_final_legs()
#                    leg_b_isfinal =  leg_b in channel_b.get_final_legs()
#                    if leg_a_isfinal or leg_b_isfinal:
#                        if leg_a_isfinal and leg_b_isfinal:
#                            this_leg_fit = True
#                            indices_a.pop(i)
#                            break
#                        else:
#                            continue
#
#                    # Get the vertex indices for the decay chain of the two
#                    # legs.
#                    for j in range(vindex_a-1, -1, -1):
#                        v = channel_a.get('vertices')[j]
#                        if leg_a in v.get('legs'):
#                            new_vid_a = j
#                            break
#                    for j in range(vindex_b-1, -1, -1):
#                        v = channel_b.get('vertices')[j]
#                        if leg_b in v.get('legs'):
#                            new_vid_b = j
#                            break
#
#                    # If any one of the pairs (leg_a, leg_b) is matched,
#                    # stop the match of this leg
#                    if Channel.check_channels_equiv_rec(channel_a,new_vid_a,
#                                                        channel_b,new_vid_b):
#                        this_leg_fit = True
#                        indices_a.pop(i)
#                        break
#
#                # If this_leg_fit is True, continue to match the next leg of
#                # channel_b. If this_leg_fit remains False, the match of this 
#                # leg cannot match to any leg of channel_a, return False
#                if not this_leg_fit:
#                    return False
#
#        # If no difference is found (i.e. return False),
#        # the two decay chain are the same eventually.
#        return True
#
#    # Helper function (obselete)    
#    @staticmethod
#    def generate_configs(id_part_list):
#        """ Generate all possible configuration for the identical particles in
#            the two channels. E.g. for legs of id=21, index= [1, 3, 5],
#            This function generate a dictionary
#            {leg id ( =21): [[1,3,5], [1,5,3], [3,1,5], [3,5,1], 
#                             [5,1,3], [5,3,1]]}
#            which gives all the possible between the id_legs 
#            in the two channels.
#        """
#        id_part_configs = {}
#        for leg_id, id_parts in id_part_list.items():
#            id_part_configs[leg_id] = [[]]
#
#            # For each index_a, pair it with an index_b.
#            for position, index_a in enumerate(id_parts):
#                # Initiate the new configs of next stage
#                id_part_configs_new = []
#
#                # For each configuration, try to find the index_b
#                # to pair with index_a in the new position.
#                for config in id_part_configs[leg_id]:
#                    # Try to pair index_a with index_b that has not been used
#                    # yet.
#                    for index_b in id_parts:
#                        if not index_b in config:
#                            config_new = copy.copy(config)
#                            config_new.append(index_b)
#                            id_part_configs_new.append(config_new)
#
#                # Finally, replace the configs by the new one.
#                id_part_configs[leg_id] = id_part_configs_new
#
#        return id_part_configs



    def simplify_lorentz(self, match):
        #
        all = match.group('var')
        var = all.split('(')[0]
        if var in ['P','PSlash']:
            nb = int(all.split(',')[1][:-1].strip())
            return '%%(q%s)s' % nb
        else:
            return '1'
        
    @classmethod
    def init_regular_expression(cls):
        dico = dict((`i`, '%s' % ','.join(["\s*-?'?[\w\s]*'?\s*"]*i)) for i in range(1,6))
        cls.lor_pattern = re.compile("""(?<![a-zA-Z])(?P<var>PSlash\(%(3)s\)|
                                        Gamma\(%(3)s\)|
                                        Sigma\(%(4)s\)|
                                        Gamma5\(%(2)s\)|
                                        C\(%(2)s\)|
                                        Epsilon\(%(4)s\)|
                                        Metric\(%(2)s\)|
                                        Identity\(%(2)s\)|
                                        ProjM\(%(2)s\)|
                                        ProjP\(%(2)s\)|
                                        P\(%(2)s\)
                                )
        """ % dico, re.VERBOSE)


    def get_apx_matrixelement_sq(self, model):
        """ Calculate the apx_matrixelement_sq, the estimation for each leg
            is in get_apx_fnrule.
            The color_multiplicity is first searching in the 
            color_multiplicity_def in model object. If no available result,
            use the get_color_multiplicity function.
            For off shell decay, this function will estimate the value
            as if it is on shell."""

        # To ensure the final_mass_list is setup, run get_onshell first
        self.get_onshell(model)
        # Setup the value of matrix element square and the average energy
        # q_dict is to record the flow of energy
        apx_m = 1
        ini_part = model.get_particle(self.get_initial_id())
        # Use the mass of antiparticle
        avg_q = (abs(eval(ini_part.get('mass'))) - sum(self['final_mass_list']))/len(self.get_final_legs())
        q_dict = {}

        # Estimate the width of normal onshell decay.
        if self.get_onshell(model):
            # Go through each vertex and assign factors to apx_m
            # Do not run the identical vertex
            for i, vert in enumerate(self['vertices']):
                
                # Total energy of this vertex
                q_total = 0
                # Color multiplcity
                final_color = []

                q_dict_lor = {}
                # Assign value to apx_m except the mother leg (last leg)
                for lnb,leg in enumerate(vert['legs'][:-1]):
                    # Case for final legs
                    if leg in self.get_final_legs():
                        mass  = abs(eval(model.get_particle(leg.get('id')).\
                                             get('mass')))
                        q_total += (mass + avg_q)
                        q_dict_lor['q%s'%(lnb+1)] = mass + avg_q
                        apx_m *= self.get_apx_fnrule(leg.get('id'),
                                                     avg_q+mass, True, model)
                    # If this is only internal leg, calculate the energy
                    # it accumulated. 
                    # (The value of this leg is assigned before.)
                    else:
                        q_total += q_dict[(leg.get('id'), leg.get('number'))]
                        q_dict_lor['q%s'%(lnb+1)]= q_dict[(leg.get('id'), leg.get('number'))]
                    # Record the color content
                    final_color.append(model.get_particle(leg.get('id')).\
                                           get('color'))
                else:
                    q_dict_lor['q%s'%(lnb+2)] = q_total

                # The energy for mother leg is sum of the energy of its product.
                # Set the q_dict
                q_dict[(vert.get('legs')[-1].get('id'), 
                        vert.get('legs')[-1].get('number'))] = q_total
                # Assign the value if the leg is not initial leg. (propagator),
                # for initial particle, q_total should be M.
                if i < len(self.get('vertices'))-1: 
                    apx_m *= self.get_apx_fnrule(vert.get('legs')[-1].get('id'),
                                                 q_total, False, model)

                else:
                    apx_m *=self.get_apx_fnrule(vert.get('legs')[-1].get('id'),
                                                abs(eval(ini_part.get('mass'))),
                                                True, model)

                # Evaluate the coupling strength
                vertex =  model.get('interaction_dict')[abs(vert.get('id'))]
                lorentz_factor = 0
                for key, v in vertex['couplings'].items():
                    if not hasattr(model, 'lorentz_dict'):
                        model.lorentz_dict = dict([(l.name, l) for l in model['lorentz']])
                        self.init_regular_expression()
                        
                    structure = model.lorentz_dict[vertex['lorentz'][key[1]]].structure 
                    new_structure = self.lor_pattern.sub(self.simplify_lorentz,
                                                         structure)
                    lor_value = eval(new_structure % q_dict_lor)
                    if lor_value == 0:
                        new_structure = new_structure.replace('-','+')
                        lor_value = eval(new_structure % q_dict_lor)
                    lorentz_factor += abs(eval(v))**2 * lor_value**2

                apx_m *= lorentz_factor

                # If final_color contain non-singlet,
                # get the color multiplicity.
                if any([i != 1 for i in final_color]):
                    ini_color = model.get_particle(vert.get('legs')[-1].get('id')).get('color')
                    # Try to find multiplicity directly from model
                    found = False
                    try:
                        color_configs = model.color_multiplicity_def(final_color)
                        for config in color_configs:
                            if config[0] == ini_color:
                                apx_m *= config[1]
                                found = True
                                break
                        # Call the get_color_multiplicity if no suitable
                        # configs in the color_dict.
                        if not found:
                            apx_m *= self.get_color_multiplicity(ini_color,
                                                                 final_color, 
                                                                 model, True)                         
                    # Call the get_color_multiplicity if the final_color
                    # cannot be found directly in the color_dict.
                    except KeyError:
                        apx_m *= self.get_color_multiplicity(ini_color,
                                                             final_color, 
                                                             model, True)
                        
        # A quick estimate of the next-level decay of a off-shell decay
        # Consider all legs are onshell.
        else:
            M = abs(eval(ini_part.get('mass')))
            # The avg_E is lower by one more particle in the next-level.
            avg_E = (M/(len(self.get_final_legs())+1.))

            # Go through each vertex and assign factors to apx_m
            # This will take all propagators into accounts.
            # Do not run the identical vertex
            for i, vert in enumerate(self['vertices']):
                
                # Assign the value if the leg is not initial leg.
                # q is assumed as 1M
                if i < len(self.get('vertices'))-1: 
                    apx_m *= self.get_apx_fnrule(vert.get('legs')[-1].get('id'),
                                                 1*M, False, model, True)

                # Assign the value to initial particle.
                else:
                    apx_m *= self.get_apx_fnrule(vert.get('legs')[-1].get('id'),
                                                 M, True, model)

                # Evaluate the coupling strength
                vertex =  model.get('interaction_dict')[abs(vert.get('id'))]
                lorentz_factor = 0
                q_dict_lor = {}
                for key, v in vertex['couplings'].items():
                    
                    if not hasattr(model, 'lorentz_dict'):
                        model.lorentz_dict = dict([(l.name, l) for l in model['lorentz']])
                        self.init_regular_expression()
                        
                    structure = model.lorentz_dict[vertex['lorentz'][key[1]]].structure 
                    new_structure = self.lor_pattern.sub(self.simplify_lorentz,
                                                         structure)
                    for i, part in enumerate(vertex['particles']):
                        mass  = abs(eval(part.get('mass')))
                        q_dict_lor['q%i' % (i+1)] = mass / 2
                    
                    lor_value = eval(new_structure % q_dict_lor)
                    # Avoid accidental zeros in lor_value
                    if lor_value == 0:
                        new_structure = new_structure.replace('-','+')
                        lor_value = eval(new_structure % q_dict_lor)

                    lorentz_factor += abs(eval(v))**2 * eval(new_structure % q_dict_lor)**2

                apx_m *= lorentz_factor

            # Calculate the contribution from final legs
            for leg in self.get_final_legs():
                apx_m *= self.get_apx_fnrule(leg.get('id'),
                                             avg_E, True, model)

        # For both on-shell and off-shell cases,
        # Correct the factor of spin/color sum of initial particle (average it)
        apx_m *= 1./(ini_part.get('spin'))
        self['apx_matrixelement_sq'] = apx_m
        return apx_m
            
    def get_apx_fnrule(self, pid, q, onshell, model, est = False):
        """ The library that provide the 'approximated Feynmann rule'
            q is the energy of the leg. The onshell label is to decide
            whether this particle is final or intermediate particle."""
        
        part = model.get('particle_dict')[pid]
        mass  = abs(eval(part.get('mass')))

        # Set the propagator value (square is for square of matrix element)
        # The width is included in the propagator.
        if onshell:
            value = 1.
        else:
            if not est:
                value = 1./((q ** 2 - mass ** 2) ** 2 + \
                                mass **2 * part.get('apx_decaywidth') **2)
            # Rough estimation on propagator. Avoid the large propagator when
            # q is close to mass
            else:
                m_large = max([q, mass])
                value = 1./(0.5* m_large**2)**2

        # Set the value according the particle type
        # vector boson case
        if part.get('spin') == 3:
            if onshell:
                # For massive vector boson
                if mass != 0. :
                    value *= (1+ (q/mass) **2)
                # For massless boson
                else:
                    value *= 1
            # The numerator of propagator.
            else:
                value *= (1 - 2* (q/mass) **2 + (q/mass) **4)
        # fermion case
        elif part.get('spin') == 2:
            if onshell:
                value *= 2.*q 
            else:
                value *= q **2
        #spin 3/2 case
        elif part.get('spin') == 4:
            if onshell:
                #spin1 part
                # For massive vector boson
                if mass != 0. :
                    value *= (1+ (q/mass) **2)
                # For massless boson
                else:
                    value *= 1
                #spin1/2 part
                value *= 2.*q 
            # The numerator of propagator.
            else:
                # For massive 
                if mass != 0. :
                    value *= 4/9 * (q)**2 * (1-q**2/mass**2)**2
                # For massless case
                else:
                    value *= q**2
        #spin2 case
        elif part.get('spin') == 5:
            if onshell:
                #spin1**2 
                # For massive vector boson
                if mass != 0. :
                    value *= (1+ (q/mass) **2)**2
                # For massless boson
                else:
                    value *= 1
            # The numerator of propagator.
            else:
                # For massive 
                if mass != 0. :
                    value *= (7/6-4/3*q**2/mass**2+2/3*q**4/mass**4)**2
                # For massless case
                else:
                    value *= 1      
                     

        # Do nothing for scalar
        return value

    def get_color_multiplicity(self, ini_color, final_color, model, base=False):
        """ Get the color multiplicity recursively of the given final_color.
            The multiplicity is obtained based on the color_multiplicity_def
            funtion in the model.
            If the color structure of final_color matches the 
            color_multiplicity_def, return the multiplicity.
            Otherwise, return 1 for the get_color_multiplicity with base = True
            or return 0 for the get_color_multiplicity with base = False."""
            
        # Combine the last two color factor to get the possible configs.
        color_configs = model.color_multiplicity_def([final_color.pop(),
                                                      final_color.pop()])
        c_factor = 1.
        # Try each config
        for config in color_configs:
            # The recursion ends when the length of the final_color now is 0.
            # (i.e. length = 2 before pop)
            if len(final_color) == 0:
                # If the final_color is consistent to ini_color, return the
                # nonzero multiplicity
                if config[0] == ini_color:
                    return config[1]

            else:
                # If next_final_color has more than one element,            
                # creaat a new final_color for recursion.
                next_final_color = copy.copy(final_color)
                next_final_color.append(config[0])

                # Call get_color_multiplicity with next_final_color as argument.
                c_factor = config[1]* self.get_color_multiplicity(ini_color,
                                                                  next_final_color,
                                                                  model)

                # If the c_factor is not zero, the color configs match successfully.
                # Return the c_factor.
                if c_factor != 0:
                    return c_factor

        # If no configs are satisfied...
        # Raise the warning message and return 1 for base get_color_multiplicity
        if base:
            logger.warning("Color structure %s in interaction is not included!" %str(final_color))
            return 1
        # return 0 for intermediate get_color_multiplicity.
        else:
            return 0
        

    def get_apx_psarea(self, model):
        """ Calculate the approximate phase space area. For off-shell case,
            it only approximate it as if it is on-shell.
            For on-shell case, it calls the recursive calculate_apx_psarea
            to get a more accurate estimation. """

        M = abs(eval(model.get_particle(self.get_initial_id()).get('mass')))

        # Off-shell channel only estimate the psarea if next level is onshell.
        if not self.get_onshell(model):
            # The power of extra integration for this level is
            # number of current final particle -2 
            # (3-body decay -> 1 integration)
            self['apx_psarea'] = 1/(8*math.pi)*\
                pow((c_psarea*(M/8./math.pi)**2), len(self.get_final_legs())-2)

        # For onshell case and psarea has not been calculated
        elif not self['apx_psarea']:
            # The initial particle mass
            mass_list = copy.copy(self['final_mass_list'])
            self['apx_psarea'] = self.calculate_apx_psarea(M, mass_list)

        return self['apx_psarea']

    def calculate_apx_psarea(self, M, mass_list):
        """Recursive function to calculate the apx_psarea.
           For the estimation of integration, it takes the final mass in the
           mass_list. The c_psarea is corrected in each integration.
           Symmetric factor of final state is corrected.
           """

        # For more than 2-body decay, estimate the integration from the
        # middle point with c_psarea factor for correction, then
        # calls the function itself with reduced mass_list.
        if len(mass_list) >2 :
            # Mass_n is the mass that use pop out to calculate the ps area.
            mass_n = mass_list.pop()
            # Mean value of the c.m. mass of the rest particles in mass_list
            M_eff_mean = ((M-mass_n) +sum(mass_list))/2
            # The range of the c.m. mass square
            delta_M_eff_sq = ((M-mass_n) ** 2-sum(mass_list) ** 2)
            # Recursive formula for ps area,
            # initial mass is replaced as the square root of M_eff_sq_mean
            return math.sqrt((M ** 2+mass_n ** 2-M_eff_mean**2) ** 2-\
                                 (2*M *mass_n) ** 2)* \
                                 self.calculate_apx_psarea(M_eff_mean, mass_list)*\
                                 delta_M_eff_sq*c_psarea* \
                                 1./(16*(math.pi ** 2)*(M ** 2))
            
        # for two particle decay the phase space area is known.
        else:
            # calculate the symmetric factor first
            self['s_factor'] =1
            id_list = sorted([l.get('id') for l in self.get_final_legs()])
            count =1
            for i, pid in enumerate(id_list):
                if i !=0 and id_list[i-1] == pid:
                    count += 1
                elif count != 1:
                    self['s_factor'] = self['s_factor'] * math.factorial(count)
                    count = 1

            # This complete the s_factor if the idparticle is in the last part
            # of list.
            if count != 1:
                self['s_factor'] = self['s_factor'] * math.factorial(count)
            return math.sqrt((M ** 2+mass_list[0] ** 2-mass_list[1] ** 2) ** 2-\
                                 (2* M *mass_list[0]) ** 2)* \
                                 1./(8*math.pi*(M ** 2)*self['s_factor'])


    def get_apx_decaywidth(self, model):
        """Calculate the apx_decaywidth
           formula: Gamma = ps_area* matrix element square * (1/2M)
           Note that it still simulate the value for off-shell decay."""

        # Return the value now if width has been calculated.
        if self['apx_width_calculated']:
            return self['apx_decaywidth']

        # Calculate width (use mass of antiparticle of initial particle)
        self['apx_decaywidth'] = self.get_apx_matrixelement_sq(model) * \
            self.get_apx_psarea(model)/ \
            (2*abs(eval(model.get_particle(self.get_initial_id()).get('mass'))))
        self['apx_width_calculated'] = True

        return self['apx_decaywidth']

    def get_apx_decaywidth_nextlevel(self, model):
        """ Estimate the sum of all the width of the next-level channels
            it developes."""

        M = abs(eval(model.get_particle(self.get_initial_id()).get('mass')))
        m_now = sum(self.get('final_mass_list'))
        avg_E = (M/(len(self.get_final_legs())+1.))
        # Ratio is the width of next-level channels over current channel.
        err = 0.
        for leg in self.get_final_legs():
            # Use only particle not anti-particle because anti-particle has no
            # width
            part = model.get_particle(abs(leg.get('id')))
            # For legs that are possible to decay.
            if (not part.get('is_stable')) and (M-m_now+part.get('2body_massdiff')) > 0.:
                # Suppose the further decay is two-body.
                # Formula: ratio = width_of_this_leg * (M/m_leg)**(-1) *
                #                  (2 * M * 8 * pi * (c_psarea* (M/8/pi)**2)) *
                #                  1/(leg_mleg(mleg)/leg_mleg(0.5M) *
                #                  Propagator of mleg(M)
                err += (part.get('apx_decaywidth')*\
                            (M/abs(eval(part.get('mass')))) **(-1) *\
                            (c_psarea*(M **3/4/math.pi)) / \
                            (self.get_apx_fnrule(leg.get('id'), avg_E,
                                                 True, model)*\
                                 self.get_apx_fnrule(leg.get('id'), 
                                                     abs(eval(part.get('mass'))), True, model))*\
                            self.get_apx_fnrule(leg.get('id'), M,
                                                False, model, True)
                        )

        self['apx_decaywidth_nextlevel'] = self.get_apx_decaywidth(model)*err

        return self['apx_decaywidth_nextlevel']


#===============================================================================
# ChannelList: List of all possible  channels for the decay
#===============================================================================
class ChannelList(base_objects.DiagramList):
    """List of decay Channel
    """

    def is_valid_element(self, obj):
        """ Test if the object is a valid Channel for the list. """
        return isinstance(obj, Channel)


#===============================================================================
# ChannelList: List of all possible  channels for the decay
#===============================================================================
class IdentifyHelasTag(diagram_generation.DiagramTag):
    """ DiagramTag daughter class to identify helas call. Model is necessary
        for initialization.
        Note: 1.) diagrams that are identical may have different type of
                  mother particles, e.g. t > b w+ and w+ > c s~ are the same
              2.) the output diagram still has the same leg number as its
                  mother diagram. This means the diagrams within the same 
                  amplitude still have consistent leg number.
                  But the leg numbers in a std_diagram may be different
                  from the numbers in model['helascalls'], though the structure
                  is identical.
              3.) in general, the last vertex of output diagram is not initial
                  vertex; however, the initial leg can be found by looking for
                  the end leg with number=1.
    """

    @staticmethod
    def link_from_leg(leg, model):
        """Returns the end link for a leg needed to identify Helas calls
        configs: ((spin, color), (leg id, leg number, leg state) 
        N.B.: only spin and color are used for comparison."""


        part = model.get_particle(leg.get('id'))


        return [((part.get('spin'), part.get('color')),
                 (leg.get('id'), leg.get('number'), leg.get('state')))]
        
    @staticmethod
    def vertex_id_from_vertex(vertex, last_vertex, model, ninitial):
        """Returns the info needed to identify Helas calls:
        ((interaction lorentz, color, keys of couplings), ineraction id)
        N.B.: interaction id is listed for reconstruction purpose,
        but not used for comparison."""

        inter = model.get_interaction(vertex.get('id'))

        return ((inter['lorentz'], inter['color'], 
                 sorted(inter['couplings'].keys())),
                inter['id'])

    @staticmethod
    def leg_from_link(link):
        """Return a leg from a link"""

        if link.end_link:
            # This is an external leg, info in links
            return base_objects.Leg({'number':link.links[0][1][1],
                                     'id':link.links[0][1][0],
                                     'state':link.links[0][1][2],
                                     'onshell':False})

        # This shouldn't happen
        assert False

    @staticmethod
    def id_from_vertex_id(vertex_id):
        """Return the numerical vertex id from a link.vertex_id"""
        return vertex_id[1]


#===============================================================================
# DecayAmplitude: An Amplitude like object contain Process and Channels
#===============================================================================
class DecayAmplitude(diagram_generation.Amplitude):
    """ DecayAmplitude is derived from Amplitude. It collects channels 
        with the same final states and create a Process object to describe it.
        This could be used to generate HELAS amplitude."""

    sorted_keys = ['process', 'diagrams', 'apx_decaywidth', 'apx_br',
                   'exa_decaywidth', 'part_sn_dict', 'inter_sn_dict',
                   'ab2real_dicts', 'has_mirror_process']#,
                   #'potential_gauge_dependence']

    def default_setup(self):
        """Default values for all properties. Property 'diagrams' is now
           as ChannelList object."""

        self['process'] = base_objects.Process()
        self['diagrams'] = ChannelList()
        self['apx_decaywidth'] = 0.
        self['apx_br'] = 0.
        self['exa_decaywidth'] = False
        #self['potential_gauge_dependence'] = False

        # Properties used in abstract amplitude
        self['part_sn_dict'] = {}
        self['inter_sn_dict'] = {}
        self['ab2real_dicts'] = Ab2RealDictList()
        # add has_mirror_process
        self['has_mirror_process'] = False

    def __init__(self, argument=None, model=None):
        """ Allow initialization with a Channel and DecayModel to create 
            the corresponding process."""

        if isinstance(argument, Channel) and isinstance(model, DecayModel):

            super(DecayAmplitude, self).__init__()

            # Set the corresponding process.
            self.set_process(argument, model)
            
            # Set diagram
            self.set('diagrams', ChannelList([argument]))
            #self['potential_gauge_dependence'] = argument['potential_gauge_dependence']

        else:
            super(DecayAmplitude, self).__init__(argument)

    def filter(self, name, value):
        """Filter for valid amplitude property values."""

        if name == 'process':
            if not isinstance(value, base_objects.Process):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Process object." % str(value)
            # Reset the width and br
            self.reset_width_br()

        if name == 'diagrams':
            if not isinstance(value, ChannelList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ChannelList object." % str(value)
            # Reset the width and br
            self.reset_width_br()

        if name == 'apx_decaywidth' and name == 'apx_br':
            if not isinstance(value, float):
                raise self.PhysicsObjectError, \
                        "%s is not a float." % str(value)

        if name == 'exa_decaywidth':
            if not isinstance(value, float) and not isinstance(value,bool):
                raise self.PhysicsObjectError, \
                        "%s is not a float." % str(value)

        return True

    def get(self, name):
        """Get the value of the property name."""

        # When apx_decaywidth is requested, recalculate it if needed.
        # Calculate br in the same time.
        if name == 'apx_decaywidth' and not self[name]:
            model = self['process']['model']
            self['apx_decaywidth'] = sum([c.get('apx_decaywidth') \
                                  for c in self['diagrams']])

        # If apx_br is requested, recalculate from the apx_decaywidth if needed.
        if name == 'apx_br' and not self[name]:
            # diagrams should have already save their initial id.
            ini_part = self['process']['model'].\
                get_particle(self['diagrams'][0].get_initial_id())
            try:
                self['apx_br'] = self.get('apx_decaywidth')/ \
                    ini_part['apx_decaywidth']
            except ZeroDivisionError:
                pass
                #logger.warning("Try to get branch ratio from a zero width particle %s. No action proceed." % ini_part.get('name'))

        return super(DecayAmplitude, self).get(name)

    def get_sorted_keys(self):
        """Return DecayProcess property names as a nicely sorted list."""

        return self.sorted_keys

    def set_process(self, dia, model):
        """ Setup the process from the diagram,
            used in initial setup."""

        # Check the number of initial leg is 1
        if dia['vertices'][-1]['legs'][-1]['number'] != 1:
            raise self.PhysicsObjectError, \
                "The number of initial leg should be setted as 1."

        # Append the initial leg.
        leglist = base_objects.LegList([base_objects.Leg({'id': dia.get_initial_id(model),
                                                          'number':1,
                                                          'state': False})])
        # Extract legs from final legs of Channel.
        leglist.extend(base_objects.LegList(\
                copy.deepcopy(sorted([l for l in dia.get_final_legs()], 
                                     legcmp_bynumber))))
            
        # Set up process and model.
        self.set('process', base_objects.Process({'legs':leglist}))
        self['process'].set('model', model)

    def add_std_diagram(self, new_dia, model=None):
        """ Add new diagram into amplitude
            and check if the number identifiers 
            of outgoing legs are consistent with the process."""

        if not isinstance(new_dia, Channel):
            raise self.PhysicsObjectError,\
                "The argument should be Channel object."

        # Model must be provided in argument or in self['process']
        if not model:
            if not self['process']['model']:
                raise self.PhysicsObjectError,\
                    "No model is given either in argument or in process."
            else:
                model = self['process']['model']

        # If this amplitude has no process
        if not self['process']['legs']:
            self.set_process(new_dia, model)
            self['diagrams'].append(new_dia)
            return
        
        # Edit the potential_gauge_dependence property
#        if new_dia['potential_gauge_dependence']:
#            self['potential_gauge_dependence'] = True


        # non_std_number: number of new_dia
        # take final legs first
        non_std_numbers = [(l.get('id'),l.get('number')) \
                               for l in new_dia.get_final_legs()]


        # initial leg
        non_std_numbers.append((new_dia.get_initial_id(model), 1))
        non_std_numbers.sort(id_num_cmp)

        # std_number: numbers of legs in process
        std_numbers = [(l.get('id'),l.get('number')) \
                           for l in sorted(self['process']['legs'])]
        std_numbers.sort(id_num_cmp)

        # Return if the numbers in diagram is the same as process
        if non_std_numbers == std_numbers:
            self['diagrams'].append(new_dia)
            return

        # Conversion from non_std_number to std_number
        converted_dict = dict([(num[1], std_numbers[i][1])\
                                   for i, num in enumerate(non_std_numbers)])
        
        # 1st stage of converting all legs: change numbering without fixing
        # wrong number flows (e.g. number 3 2 > 3)
        all_numbers_goal = []
        for vert in new_dia.get('vertices'):
            for leg in vert.get('legs'):
                leg['number'] = converted_dict[leg['number']]
                all_numbers_goal.append((leg['id'], leg['number']))

        # 2nd stage of converting all legs: fixing illegal number flows.
        # (except for the first one)
        for pos,vert in enumerate(new_dia.get('vertices')[:-1]):
            lowest_num = vert.get('legs')[0]['number']
            for leg in vert.get('legs')[:-1]:
                if leg['number'] < lowest_num:
                    lowest_num = leg['number']

            mother_leg = vert.get('legs')[-1]
            old_id_number = (mother_leg['id'], mother_leg['number'])
            if old_id_number[1] != lowest_num:
                # Change the number of mother
                mother_leg['number'] = lowest_num                
                
                # Find the leg associated with the mother of this vertex,
                # and change its number to lowest number
                found = False
                for pre_vert in new_dia.get('vertices')[pos:]:
                    for child_leg in pre_vert['legs'][:-1]:
                        if (child_leg['id'], child_leg['number']) == old_id_number:
                            child_leg['number'] = lowest_num
                            found = True
                            break
                    if found:
                        break
            """
            # Put particles before antiparticles            
            vert['legs'][:-1] = sorted(vert['legs'][:-1],
                                       key=lambda leg: leg['id'],
                                       reverse = True)"""

        new_dia.initial_setups(model, True)

        # Add this standard diagram into diagrams
        self['diagrams'].append(new_dia)


    def reset_width_br(self):
        """ Reset the value of decay width and branch ratio.
            Automatically done in the set(filter) function.
            This is needed when content of diagrams or process are changed."""
        
        self['apx_decaywidth'] = 0.
        self['apx_br'] = 0.

    def nice_string(self, indent=0):
        """Returns a nicely formatted string of the amplitude content"""
#        if not self['potential_gauge_dependence']:
#            return self.get('process').nice_string(indent) + "\n" + \
#                self.get('diagrams').nice_string(indent)
#        else:
        return self.get('process').nice_string(indent) + ", *PGD" + "\n" + \
                self.get('diagrams').nice_string(indent)
            

    def decaytable_string(self, format='normal'):
        """ Write the string in the format for decay table.
            format = 'normal': show only branching ratio
                   = 'full'  : show branching ratio and all the channels."""

        output='   %.5e   %d' %(self.get('apx_br'),
                                len(self['process']['legs'])-1)
        output += ''.join(['%11d' %leg.get('id') \
                               for leg in self['process']['legs'][1:]])

        if format == 'cmp':
            if self.get('exa_decaywidth'):
                output += '\t%4.2f' % (self.get('apx_decaywidth')/self.get('exa_decaywidth'))
            else:
                output += '\tN/A'
        
        output += '   #Br(%s)' %self.get('process').input_string()

        # Gauge dependence tag
        #if self['potential_gauge_dependence']:
        #    output += ', *PGD\n'
        #else:
        output += '\n'
        
        # Output the channels if format is full
        if format=='full':
            # Set indent of the beginning for channels
            output += self.get('diagrams').nice_string(6)

        # Return process only, get rid off the final \n
        elif format == 'normal' or format == 'cmp':
            return output[:-1]

        # Raise error if format is wrong
        else:
            raise self.PhysicsObjectError,\
                "Format %s must be \'normal\' or \'full\'." % str(format)

        # Return output for full format case.
        return output


    def abstract_nice_string(self):
        """ Print the nice string of abstract amplitudes,
        which shows the real process of this type."""

        output = super(DecayAmplitude, self).nice_string()
        output += "\nReal process:  "
        output += ', '.join([ref['process'].input_string() for ref in self['ab2real_dicts']])

        return output


    def generate_variables_dicts(self, real_amp, ab2real_id = -1):
        """ Generate the dictionary of abstract to real variables,
        default is to use the latest Ab2RealDict."""

        ab_model = self['process']['model']
        real_model = real_amp['process']['model']
        ab2realdict = self['ab2real_dicts'][ab2real_id]

        # Set the initial particle dict
        ab2realdict['mass_dict']\
            [ab_model.get_particle(self['process'].get_initial_ids()[0])['mass']] = real_model.get_particle(real_amp['process'].get_initial_ids()[0])['mass']

        # Set the final particle dict
        for ab_pid, real_pid in ab2realdict['final_legs_dict'].items():
            if isinstance(real_pid, int):
                ab2realdict['mass_dict'][ab_model.get_particle(ab_pid)['mass']] = real_model.get_particle(real_pid)['mass']
            # For the case of identical particle
            else:
                for pid in real_pid:
                    ab2realdict['mass_dict'][ab_model.get_particle(ab_pid)['mass']] = real_model.get_particle(pid)['mass']

        # Set the interaction id dict and intermediate particles
        for  ab_dia_id, real_dia_id in ab2realdict['dia_sn_dict'].items():

            ab_dia = self['diagrams'][ab_dia_id]
            real_dia = real_amp['diagrams'][real_dia_id]
            
            # Set the intermediate particle and interaction,
            # except for the identical vertex 
            # (the initial particle will be resetted, but it should be fine.)
            for i, v in enumerate(ab_dia['vertices']):

                # Set intermediate particle
                ab2realdict['mass_dict']\
                    [ab_model.get_particle(v['legs'][-1]['id'])['mass']] \
                    = real_model.get_particle(real_dia['vertices'][i]['legs'][-1]['id'])['mass']
                
                # Set the interaction
                for ab_key, real_coup in \
                        ab_model['interaction_coupling_dict'][real_dia['vertices'][i]['id']].items():

                    ab_coup = ab_model.get_interaction(ab_dia['vertices'][i]['id'])['couplings'][ab_key]
                    ab2realdict['coup_dict'][ab_coup] = real_coup
        
#===============================================================================
# DecayAmplitudeList: An Amplitude like object contain Process and Channels
#===============================================================================
class DecayAmplitudeList(diagram_generation.AmplitudeList):
    """ List for DecayAmplitudeList objects.
    """

    def is_valid_element(self, obj):
        """ Test if the object is a valid DecayAmplitude for the list. """
        return isinstance(obj, DecayAmplitude)

    def get_amplitude(self, final_ids):
        """ Get the amplitudes with the given final particles if
        exist in this list. Otherwise, return None.
        Note: Use stored finallist in Channel!"""
        
        for amp in self:
            if sorted([abs(l.get('id')) for l in amp['diagrams'][0]\
                           .get_final_legs()]) == sorted(list_abs(final_ids)):
                return amp
            

        return None

    def nice_string(self):
        """ Nice string from Amplitude """
        mystr = '\n'.join([a.nice_string() for a in self])

        return mystr

    def abstract_nice_string(self):
        """ Nice string for abstract amplitude. """
        mystr = '\n'.join([a.abstract_nice_string() for a in self])

        return mystr


    def decaytable_string(self, format='normal'):
        """ Decaytable string for Amplitudes """

        # Get the level (n-body) of this amplitudelist and
        # the total width inside this DecayAmplitudeList
        level = len(self[0].get('diagrams')[0].get_final_legs())
        total_width = sum([amp.get('apx_decaywidth') for amp in self])

        # Print the header with the total width
        mystr = '## Contribution to total width of %d-body decay: %.5e \n' \
            %(level, total_width)
        mystr += '\n'.join([a.decaytable_string(format) for a in self])

        return mystr

#===============================================================================
# AbstractModel: A Model class in which objects are abstract
#===============================================================================
class AbstractModel(base_objects.Model):
    """ Model in which particles are different only in Lorentz structure
    and color. Interactions may be duplicated in order to generate amplitude
    with the same Lorentz structure vertices or diagrams
    """
    sorted_keys = ['name', 'particles', 'parameters', 'interactions', 
                   'couplings', 'lorentz',
                   'abstract_particles_dict', 'abstract_interactions_dict',
                   'particle_type_dict', 
                   'interaction_type_dict', 'interaction_coupling_dict',
                   'ab_matrix_elements'
                  ]

    def default_setup(self):
        """The particles is changed to ParticleList"""

        super(AbstractModel, self).default_setup()
        # Other properties
        self['particles'] = DecayParticleList()
        # Object type -> Abstract Object list of the given type
        self['abstract_particles_dict'] = {}
        self['abstract_interactions_dict'] = {}

        # Real interaction id -> (Abstract type (spin, color, self-anti),
        #                         pseudo_pdg_code)
        # keys are always positive!!!
        self['particle_type_dict'] = {}

        # Real interaction id -> Abstract type (pseudo abstract particlelist,
        #                                       lorentz,
        #                                       color)
        self['interaction_type_dict'] = {}

        self['interaction_coupling_dict'] = {}

        # Store all the matrix elements
        self['ab_matrix_elements'] = AbstractHelasMatrixElementList()

        self.spin_text_dict = {1:'S', 2: 'F', 3:'V', 5:'T'}
        

    def get_sorted_keys(self):
        return self.sorted_keys

    def get_particle_type(self, part, get_code=False):
        """ Return the tuple (spin, color, self_antipart) 
        of the given particle.
        NOTE: bosons are always treated as self-antipart. """

        if isinstance(part, base_objects.Particle):
            if get_code:
                return abs(self.get_particle_type_code(part))

            if part.is_boson():
                self_antipart = True
            else:
                self_antipart = part['self_antipart']

            return (part['spin'], part['color'], self_antipart)

        # Use particle_type_dict if input is pdg_code
        elif isinstance(part, int):
            if not get_code:
                return self['particle_type_dict'][abs(part)][0]
            else:
                return self['particle_type_dict'][abs(part)][1]
        else:
            raise self.PhysicsObjectError, \
                "Input should be Particle type or pdg_code(int)."


    def get_particle_type_code(self, part):
        """ Return the pseudo pdg_code :
        (9900000+1000*spin+100*color) for non-self-antipart;
        (9910000+1000*spin+100*color) for self-antipart (Majorana fermion)
        of the given particle."""

        # Special pdg_code for Majorana fermions
        if part.is_fermion():
            if part['self_antipart']:
                return 9910000+1000*part['spin']+100*part['color']
            else:
                return int(math.copysign(
                        9900000+1000*part['spin']+100*part['color'],
                        part.get_pdg_code()))
        else:
            return 9900000+1000*part['spin']+100*part['color']


    def add_ab_particle(self, pdg_code, force=False):
        """ Functions to add new particle according to the given particle
        spin and color information. This function will assign the correct
        serial number of new particle and change the abstract_particles."""

        # Check argument type
        if not force and not isinstance(pdg_code, int):
            raise self.PhysicsObjectError,\
                "Argument must be a pdg_code."

        # To ensure pdg_code is positive
        pdg_code = abs(pdg_code)

        # Setup new particle
        ab_part = DecayParticle()
        ab_part['spin'] = self.get_particle_type(pdg_code)[0]
        ab_part['color'] = self.get_particle_type(pdg_code)[1]
        ab_part['self_antipart'] = self.get_particle_type(pdg_code)[2]

        # Check the current abstract_particles
        if self.get_particle_type(pdg_code) \
                in self['abstract_particles_dict'].keys():
            # For existing type, record the serial number
            sn = len(self['abstract_particles_dict']\
                         [self.get_particle_type(pdg_code)])
        else:
            # Setup new type of abstract particle
            self['abstract_particles_dict']\
                [self.get_particle_type(pdg_code)]= DecayParticleList()
            sn = 0
            

        # Set other properties: name = text(spin)+color
        try:
            name_string = self.spin_text_dict[ab_part.get('spin')]
        except KeyError:
            name_string = 'P'

        ab_part['name']= \
            name_string +\
            str(ab_part.get('color')) +\
            '_%02d' %sn

        # mass = M'S or N''spin''color' 
        #                 % S for self-antipart, N for non-self-antipart
        mass_string = 'M'
        if ab_part['self_antipart']:
            mass_string = mass_string + 'S'
        else:
            mass_string = mass_string + 'N'
        mass_string = mass_string + name_string +\
            str(ab_part.get('color')) +\
            '_%02d' %sn
        ab_part['mass'] = mass_string

        # pdg_code = 99'1:S ,0: N'00'spin''color'
        #  e.g. w+: 9903100, gluino: 9912300
        ab_part['pdg_code'] = self.get_particle_type(pdg_code, get_code=True)+sn

        # Set anti-part properties if this is not a self-antiparticle
        if not ab_part['self_antipart']:
            ab_part['antiname'] = ab_part.get('name')+'~'

            # Create the anti-part
            anti_ab_part = DecayParticle(ab_part)
            anti_ab_part['is_part'] = False
        
        # Append ab_part into self['particles'] and abstract_particles_dict,
        self['particles'].append(ab_part)
        self['abstract_particles_dict'][self.get_particle_type(ab_part)].append(ab_part)
        logger.info("Add Abstract Particle %s according to #%s" \
                        %(ab_part.get('name'), pdg_code))


        # Reset the particle dictionary
        self.get('particle_dict')
        self['particle_dict'][ab_part['pdg_code']] = ab_part
        if not ab_part['self_antipart']:
            self['particle_dict'][-ab_part['pdg_code']] = anti_ab_part



    def setup_particle(self, part, force=False):
        """Add real particle into AbstractModel, 
        convert into abstract particle."""

        # Check argument type
        if not force and not isinstance(part, base_objects.Particle):
            raise self.PhysicsObjectError,\
                "Argument must be a Particle object."

        # Setup the particle_type_dict for all particle
        self['particle_type_dict'][part['pdg_code']] = \
            (self.get_particle_type(part), 
             self.get_particle_type(part, get_code=True))

        if self.get_particle_type(part) \
                not in self['abstract_particles_dict'].keys():
            # if not found in existing particle, create a new abstract particle
            self.add_ab_particle(part['pdg_code'], force)


    def setup_particles(self, part_list, force=False):
        """Add real particles into AbstractModel, 
        convert into abstract particles"""

        # Check argument type
        if not force and not isinstance(part_list, base_objects.ParticleList):
            raise self.PhysicsObjectError,\
                "Argument must be a ParticleList."

        # Call add_particle for each particle
        for part in part_list:
            self.setup_particle(part, force)

        # Reset dictionaries in the model
        self.reset_dictionaries()
        


    def get_particlelist_type(self, pdgcode_list, ignore_dup=True,
                              sn_dict={}, std_output=True):
        """ Return a list of the type of the given particlelist,
        and a dictionary records the number of each type of particle.
        The pdg_code will start from the number given by serial_number_dict.
        Note1: If ignore_dup is True,
               the same particles will not be assigned with the same id.
        Note2: std_output is used to get the same abstract_pidlist
               for a given real_pids, no matter what order is.
               Usage e.g.: pids of final_legs in one real amplitude.
               Not used for: comparing two real_pids.
        """

        pseudo_ab_particlelist = []
        # Used the given sn_dict or an empty one
        serial_number_dict = copy.copy(sn_dict)

        if not pdgcode_list:
            return pseudo_ab_particlelist, serial_number_dict
        
        # If standard output is required, input_pdgcode_list
        # is sorted.
        input_pdgcode_list = copy.copy(pdgcode_list)
        if std_output:
            # List of indices to record the original order
            original_order = range(len(pdgcode_list))

            # Sort according to (particle_type, real_pid) of input_pdgcode_list
            temp = zip(input_pdgcode_list, original_order)
            temp.sort(key=lambda item: \
                          (self.get_particle_type(item[0]),item[0]))
            try:
                input_pdgcode_list, original_order = zip(*temp)
            except:
                print original_order, pdgcode_list

        # Construct the abstract pid list
        for i, pdg_code in enumerate(input_pdgcode_list):

            # Use get_particle_type function
            part_type = self.get_particle_type(abs(pdg_code))
            # The sign of part_type_code depends on whether the
            # abstract particle is self-antipart or not
            part_type_code = self.get_particle_type(abs(pdg_code), True)
            if pdg_code < 0 and not part_type[2]:
                part_type_code = -part_type_code

            # Add the pseudo pdg code, if it is the first one in its type
            # appearing in the list.
            # There should be a abstract particle for this type,
            # if the setup_particles has been run correctly.
            if not part_type in serial_number_dict.keys():
                pseudo_ab_particlelist.append(\
                        part_type_code)

                serial_number_dict[part_type] = 1

            # If not ignore duplicate particles,
            # check if there are same particle in the list.
            else:
                set_new = True

                if not ignore_dup:
                    for j, previous_pdgcode in enumerate(input_pdgcode_list):
                        # find duplicate particle, use the abstract particle
                        # already exists.
                        # No need to update the serial_number_dict
                        if j == i:
                            break

                        if abs(previous_pdgcode) == \
                                abs(pdg_code):
                            pseudo_ab_particlelist.append(\
                                int(math.copysign(\
                                        pseudo_ab_particlelist[j],
                                        part_type_code)))
                                
                            set_new = False
                            break

                # Append new abstract particle if not appears before
                if set_new:
                    # If the abstract particle of the given serial number
                    # does not exist, add a new abstract particle and append
                    # it.
                    if serial_number_dict[part_type] >= \
                            len(self['abstract_particles_dict'][part_type]):
                        self.add_ab_particle(pdg_code, True)
                            
                    # Append the pdg_code into the list,
                    # starting from the s/n given by serial_number_dict
                    pseudo_ab_particlelist.append(\
                        int(math.copysign(\
                                abs(part_type_code)+\
                                    serial_number_dict[part_type],
                                part_type_code))
                        )

                    # Update the serial_number_dict
                    serial_number_dict[part_type] += 1

        # For std_output, reorder the pseudo_ab_particlelist
        # to be the same as pdgcode_list
        if std_output:
            temp = zip(original_order, pseudo_ab_particlelist)
            temp.sort()
            original_order, pseudo_ab_particlelist = zip(*temp)
            pseudo_ab_particlelist = list(pseudo_ab_particlelist)

        return pseudo_ab_particlelist, serial_number_dict


    def get_color_string(self, inter):
        """ Return the correct color string according to the 
        sorted order of particle. """
        
        pass


    def get_interaction_type(self, inter_id):
        """ Return the tuple (lorentz, particles_type) of the given interaction
        Raise error if type is not in quick reference dictionary."""

        # Check the quick reference dictionary.
        # If the lorentz type has been imported, use the type
        # provided by the dictionary. Otherwise, raise error.
        # Note: the keys, (lorentz, part_types), are well-sorted in
        # setup_interactions.
        try:
            return self['interaction_type_dict'][inter_id]
        except KeyError:
            raise self.PhysicsObjectError, \
                "Interaction #%d has not been imported into AbstractModel." \
                % inter_id


    def add_ab_interaction(self, inter_id, force=False, color = None):
        """ Functions to set new interaction according to the given interaction
        particles and lorentz sturcture. This function will assign the correct
        serial number of new interaction, change the abstract_interactions.
        Note: Add NEW type of interaction should use setup."""

        # Check argument type
        if not force and not isinstance(inter_id, int):
            raise self.PhysicsObjectError,\
                "Argument must be an Interaction id."
        
        # Setup new interaction
        ab_inter = base_objects.Interaction()
        inter_type = self.get_interaction_type(inter_id)

        # Check the current abstract_interactions
        if inter_type in \
                self['abstract_interactions_dict'].keys():
            # For existing type, record the serial number
            sn = len(self['abstract_interactions_dict'][inter_type])
        else:
            # Setup the new item, serial number = 0
            self['abstract_interactions_dict'][inter_type] = \
                base_objects.InteractionList()
            sn = 0


        # Type_sn is the serial number of the type
        if sn == 0:
            type_sn = len(self['abstract_interactions_dict'].keys())
        else:
            type_sn = \
                self['abstract_interactions_dict'][inter_type][0]['id']/1000

        # id = ___0__
        #        |  |_> the serial number
        #        | 
        #        |_> The serial number of the interaction type
        ab_inter['id'] = 1000*type_sn + sn

        # Get particle from the model.particle_dict
        # to ensure that the particle is DecayParticle, not Particle
        ab_inter['particles'] = DecayParticleList(\
            [self.get_particle(pid) for pid in inter_type[0]])
        ab_inter['lorentz'] = list(inter_type[1])

        # Retrieve the color information
        if color:
            ab_inter['color'] = color
        elif sn != 0:
            # Use the information before
            ab_inter['color'] = \
                self['abstract_interactions_dict'][inter_type][0]['color']
        else:
            raise self.PhysicsObjectError,\
                "Error to add interaction. No color information available."

        # couplings = G###cl##
        #              |  || |-> the serial number
        #              |  ||-> the lorentz identifier
        #              |  |-> the color identifier
        #              |-> The serial number of the interaction type
        ab_inter['couplings'] = {}
        for i, colr in enumerate(ab_inter['color']):
            for j, lorentz in enumerate(ab_inter['lorentz']):
                ab_inter['couplings'][(i, j)] = 'G%03d%1d%1d%02d' \
                    %(type_sn, i, j, sn)


            
        # Append ab_inter into self['interactions'] and 
        # abstract_interactions_dict
        self['interactions'].append(ab_inter)
        self['abstract_interactions_dict'][inter_type].append(ab_inter)

        # Reset the dictionary
        self['interaction_dict'][ab_inter['id']] = ab_inter
        

    def setup_interactions(self, inter_list, anti_dict, force=False):
        """Add real interactions into AbstractModel, 
        convert into abstract interactions.
        The lorentz and color 
        structures keep to be the union of all correpsonding
        real ones. 
        Construct the quick reference dictionary,
        and setup the coupling constants in the end."""

        # Check argument type
        if not force and not isinstance(inter_list, 
                                        base_objects.InteractionList):
            raise self.PhysicsObjectError,\
                "Argument must be an InteractionList."

        # Add all interactions except for
        # 1. radiation interaction, 2. interaction with all stable particles
        for inter in inter_list:
            # Check validity: remove stable particles
            final_list = [p.get('pdg_code')\
                              for p in inter['particles'] \
                              if not p['is_stable'] ]

            # Check validity: do not count duplicate particles,
            # i.e. those cannot decay through this interaction.
            __duplicate__ = True
            for pid in final_list:
                if final_list.count(pid) == 1:
                    __duplicate__ = False
                    break

            # Ignore all stable particle interaction (e.g. g g > g g) (len=0)
            # and interaction with all duplicate particles
            if len(final_list) == 0 or __duplicate__:
                continue


            # Use the get_particlelist_type to find the particles type of this
            # interaction.
            # NOTE: particles do not sort,
            #       ambiguity might happened!!
            abpart_types, sn = \
                self.get_particlelist_type([p.get_pdg_code() for p in\
                                                inter['particles']])

            # The key have to be sorted
            new_key = [tuple(abpart_types),
                       tuple(sorted(inter['lorentz'])),
                       tuple(sorted([str(colorstring) \
                                         for colorstring in inter['color']]))
                       ]
            color_list = inter['color']
            remove_list = []
            is_new_key = True

            # Check if this type is the subset of known lorentz type,
            # or contain the known lorentz type.
            # If so, union the two and continue search.
            # Do the same merge for color structure
            # Note: the abstract_interactions_dict always maintains to have
            # its keys that have no intersect lorentz with others.
            for key, ab_interlist in self['abstract_interactions_dict'].items():
                # Check lorentz if the particles are in the same types
                if key[0] == new_key[0]:
                    # Stop if this lorentz and color are already in
                    # included
                    if set(key[1]).issuperset(new_key[1]) and \
                            set(key[2]).issuperset(new_key[2]):
                        is_new_key = False
                        break
                    # Continue if no joit between the lorentz types
                    # and color strings either.
                    elif set(key[1]).isdisjoint(new_key[1]) and \
                            set(key[2]).isdisjoint(new_key[2]):
                        continue
                    # This key has a related lorentz structure or
                    # color string.
                    # Create the unions and prepare to remove this key
                    else:
                        union_lorentz = \
                            tuple(sorted(set(key[1]).union(inter['lorentz'])))
                        union_color = \
                            tuple(sorted(set(key[2]).union(new_key[2])))
                        new_key[1] = union_lorentz
                        new_key[2] = union_color

                        # Remember to remove this key later
                        remove_list.append(key)

                        # Collect the real color string
                        color_list.extend([c for c in ab_interlist[0]['color']\
                                               if not c in color_list])

            # Set temporate interaction_type_dict,
            # transform list into tuple in order to be eligible for key.
            self['interaction_type_dict'][inter['id']] = tuple(new_key)

            # If it is a new key, add interaction
            # in abstract_interactions_dict
            if is_new_key:
                
                # Use add_ab_interaction to get the correct format,
                # it will find type from interaction_type_dict
                self.add_ab_interaction(inter['id'], color = color_list)

                # Remove subset, if remove_list is not empty
                for remove_key in remove_list:
                    # Remove old abstract interactions in interaction list
                    for old_int in self['abstract_interactions_dict'][remove_key]:
                        self['interactions'].remove(old_int)
                    del self['abstract_interactions_dict'][remove_key]
                
        # Reset the id of all abstract interactions
        # (the deletion could cause some errors.)
        for i, ab_inter in enumerate(self['interactions']):
            
            type_sn = i+1
            # id = ___0__
            #        |  |_> the serial number
            #        | 
            #        |_> The serial number of the interaction type
            ab_inter['id'] = 1000*type_sn

            # couplings = G_______
            #              |  || |-> the serial number
            #              |  ||-> the lorentz identifier
            #              |  |-> the color identifier
            #              |-> The serial number of the interaction type
            ab_inter['couplings'] = {}
            for i, colr in enumerate(ab_inter['color']):
                for j, lorentz in enumerate(ab_inter['lorentz']):
                    ab_inter['couplings'][(i, j)] = 'G%03d%1d%1d00' \
                        %(type_sn, i, j)



        # Update the quick reference dict
        # and setup the interaction_coupling_dict
        for inter in inter_list:

            # Ignore interactions that cannot decay
            if inter['id'] not in self['interaction_type_dict'].keys():
                continue

            # Update the type dict
            old_type = self['interaction_type_dict'][inter['id']]
            for new_type in self['abstract_interactions_dict'].keys():
                if new_type[0] == old_type[0] and \
                        (set(old_type[1]) < set(new_type[1]) or \
                             set(old_type[2]) < set(new_type[2])):

                        self['interaction_type_dict'][inter['id']] = new_type


            # Construct the coupling dict
            self['interaction_coupling_dict'][inter['id']] = {}
            inter_type = self['interaction_type_dict'][inter['id']]            
            ab_inter = self['abstract_interactions_dict'][inter_type][0]
            for key, coup in inter['couplings'].items():
                color = inter['color'][key[0]]
                lorentz = inter['lorentz'][key[1]]
                ab_key = [0, 0]
                # Get new key for the coupling
                ab_key[0] = ab_inter['color'].index(color)
                ab_key[1] = ab_inter['lorentz'].index(lorentz)
                
                self['interaction_coupling_dict'][inter['id']][tuple(ab_key)]\
                    = coup


        # Update dict for anti-iteraction
        for inter in inter_list:
            # For possible anti-interaction, 
            # Update the interaction_type_dict, interaction_coupling_dict
            if inter['id'] in anti_dict.keys():
                    anti_inter_id = anti_dict[inter['id']]                    
            else:
                continue

            # property of -(inter_id) -> property of anti_inter_id
            self['interaction_type_dict'][-inter['id']] = \
                self['interaction_type_dict'][anti_inter_id]
            self['interaction_coupling_dict'][-inter['id']] = \
                self['interaction_coupling_dict'][anti_inter_id]


        # Reset dictionaries in the model
        self.reset_dictionaries()
        self.get('particle_dict')
        self.get('interaction_dict')


    def get_interactionlist_type(self, interid_list, ignore_dup=False,
                                 sn_dict={}):
        """ Return a list of the type of the given interactions,
        and a dictionary records the number of each type of interaction.
        The abstract interaction id will start from the number given 
        by serial_number_dict.
        Note: If ignore_dup is True,
        the same interactions will not assign the same id.
        """

        pseudo_ab_interlist = []
        # Used the given sn_dict or an empty one
        serial_number_dict = copy.copy(sn_dict)
        
        for i, inter_id in enumerate(interid_list):

            # Append identity vertices
            # Not being used by current standard of diagrams
            if inter_id == 0:
                pseudo_ab_interlist.append(0)
                continue

            # Use get_interaction_type function
            inter_type = self.get_interaction_type(inter_id)
            inter_type_code = \
                self['abstract_interactions_dict'][inter_type][0]['id']

            # Add the pseudo id, if it is the first one in its type
            # appearing in the list.
            # There should be a abstract interaction for this type,
            # if the setup_interactions has been run correctly.
            if not inter_type in serial_number_dict.keys():
                pseudo_ab_interlist.append(\
                        inter_type_code)

                serial_number_dict[inter_type] = 1

            # If not ignore duplicate interactions (default),
            # check if there are same interactions in the list.
            else:
                set_new = True

                if not ignore_dup:
                    for j, previous_id in enumerate(interid_list):
                        # find duplicate interaction, 
                        # use the abstract interaction already exists.
                        # No need to update the serial_number_dict
                        if j == i:
                            break

                        if previous_id == \
                                inter_id:
                            pseudo_ab_interlist.append(\
                                pseudo_ab_interlist[j])

                            set_new = False
                            break

                # Append new abstract interaction if not appears before
                if set_new:
                    # If the abstract interaction of the given serial number
                    # does not exist, add a new abstract interaction and append
                    # it.
                    if serial_number_dict[inter_type] >= \
                            len(self['abstract_interactions_dict'][inter_type]):
                        self.add_ab_interaction(inter_id, True)
                            
                    # Append the pdg_code into the list,
                    # starting from the s/n given by serial_number_dict
                    pseudo_ab_interlist.append(\
                        inter_type_code+\
                            serial_number_dict[inter_type]
                        )

                    # Update the serial_number_dict
                    serial_number_dict[inter_type] += 1
                        
                    
        return pseudo_ab_interlist, serial_number_dict


    # Helper function to construct the AbstractMatrixElement
    def compare_diagrams(self, ab_dia, real_dia, ab2realdict=None):
        """ Return True if the two diagrams are in the same abstract type.
        The ab_dia and real_dia must have the same topology in this algorithm.
        Algorithm: 
        a. Compare the pseudo-abstract interaction id list by ORDER
        b. Compare the pseudo-abstract pdg_code list by ORDER.
        """

        if not isinstance(ab_dia, Channel) or not isinstance(real_dia, Channel):
            raise self.PhysicsObjectError,\
                "The first two argument are not Channel objects."
        if ab2realdict != None and not isinstance(ab2realdict, Ab2RealDict):
            raise self.PhysicsObjectError,\
                "The final argument should be Ab2RealDict, otherwise should be omitted."
            

        # Interaction id list
        ab_inter_id_list = ab_dia['abstract_type'][0]
        real_inter_id_list = [v.get('id') for v in real_dia['vertices']]

        # Quick compare from the length.
        if len(ab_inter_id_list) != len(real_inter_id_list):
            return False

        # Full comparision of interaction type.
        # The duplicated intereactions are taken into consideration.
        if self.get_interactionlist_type(real_inter_id_list)[0] != \
                ab_inter_id_list:
            return False

        # Intermediate Particle id list
        real_pdgcode_list = [v.get('legs')[-1]['id'] \
                                 for v in real_dia['vertices'][:-1]]
        

        # Full comparision of intermediate particle type.
        # 1. Duplicated particles will be treated as different.
        # 2. Do not get the std_output. We want to compare two real pids,
        #    not to 
        if self.get_particlelist_type(real_pdgcode_list, std_output=False)[0]\
                != ab_dia['abstract_type'][1]:
            return False

        # Continue to check the final Particle id list
        ini_type = self.get_particle_type(real_dia.get_initial_id())
        if self.get_particle_type(real_dia.get_initial_id(), get_code=True) \
                != ab_dia.get_initial_id():
            return False

        # Final pids
        real_pdgcode_list = [l['id'] for l in real_dia.get_final_legs()]

        # Full comparision of particle type.
        # Duplicated particles will be treated as different.
        try:
            # Try to use final_legs_dict to save time
            temp = [ab2realdict['final_legs_dict'][l.get('id')] for l in ab_dia.get_final_legs()]
            assert_pidlist = []
            # Compare the full list, temp may contain sublist,
            # if one abstract pid corresponds to several real pids.
            for item in temp:
                if isinstance(item, list):
                    assert_pidlist.extend(item)
                else:
                    assert_pidlist.append(item)

            if assert_pidlist != real_pdgcode_list:
                return False

        except:
            if sorted(self.get_particlelist_type(real_pdgcode_list,
                                                 sn_dict = {ini_type: 1})[0])  != sorted(ab_dia['abstract_type'][2]):
                return False

            
        return True


    # Helper function
    def add_ab_diagram(self, ab_amp, real_dia):
        """ Add abstract diagram from real one into the abstract amplitude.
        The abstract_type of abstract diagram is constructed. """

        if not isinstance(ab_amp, DecayAmplitude):
            raise self.PhysicsObjectError,\
                "The first argument  is not DecayAmplitude object."
        if not isinstance(real_dia, Channel):
            raise self.PhysicsObjectError,\
                "The second argument is not Channel object."

        ab_dia = Channel({'onshell': True})
        ab_dia['vertices'] = copy.deepcopy(real_dia['vertices'])

        # Setup the interaction ids
        real_inter_id_list = [v.get('id') for v in real_dia['vertices']]

        # Setup the abstract interaction id type first
        new_inter_ids = self.get_interactionlist_type(real_inter_id_list)[0]
        ab_dia['abstract_type'][0] = new_inter_ids

        # Setup the abstract interaction ids from inter_sn_dict in ab_amp,
        # then recycle the sn record
        new_inter_ids, ab_amp['inter_sn_dict'] = \
            self.get_interactionlist_type(real_inter_id_list,
                                          sn_dict = ab_amp['inter_sn_dict'])
        for i, ab_inter_id in enumerate(new_inter_ids):
            ab_dia['vertices'][i]['id'] = ab_inter_id


        # Setup the initial legs, which pid should be the pid of its
        # antiparticle.
        # Also setup the 'ini_pid' property
        ini_type = self.get_particle_type(real_dia.get_initial_id())
        ini_code = self.get_particle_type(real_dia.get_initial_id(), 
                                          get_code=True)
        ab_dia['ini_pid'] = ini_code
        if ini_type[2]:
            ab_dia['vertices'][-1]['legs'][-1]['id'] = ini_code
        else:
            ab_dia['vertices'][-1]['legs'][-1]['id'] = -ini_code


        # Setup the final abstract particle id
        real_pdgcode_list = [l.get('id') for l in real_dia.get_final_legs()]
        ab_pid_list = self.get_particlelist_type(real_pdgcode_list, 
                                                 sn_dict={ini_type:1})[0]
        # The abstract diagram is copied from real diagram,
        # so the ab_pid_list, given according to the final_legs in
        # real diagram, can be used directly to abstract diagram.
        for i, leg in enumerate(ab_dia.get_final_legs()):
            leg['id'] = ab_pid_list[i]

        # Update the abstract_type
        ab_dia['abstract_type'][2] = ab_pid_list



        # Setup the intermediate Particle id list
        real_pdgcode_list = [v.get('legs')[-1]['id'] \
                                 for v in real_dia['vertices'][:-1]]

        # Setup the abstract particle id type first
        new_part_ids = self.get_particlelist_type(real_pdgcode_list)[0]
        ab_dia['abstract_type'][1] = new_part_ids

        # Setup the abstract interaction ids from the part_sn_dict of ab_amp,
        # then recycle the sn_dict.
        new_part_ids, ab_amp['part_sn_dict'] = \
            self.get_particlelist_type(real_pdgcode_list,
                                       sn_dict=ab_amp['part_sn_dict'])
        for i, ab_pid in enumerate(new_part_ids):
            ab_dia['vertices'][i]['legs'][-1]['id'] = ab_pid

            # Find the previous leg that connect with this one
            for v in ab_dia['vertices'][i+1:]:
                # Use the number to identify the same leg
                for l in v.get('legs')[:-1]:
                    if l['number'] == \
                            ab_dia['vertices'][i]['legs'][-1]['number']:
                        l['id'] = ab_pid
                        break                


        # Add this diagram into the amplitude.
        ab_amp.add_std_diagram(ab_dia, self)

            

    def generate_ab_amplitudes(self, amp_list):
        """ Generate the abstract Amplitudes from real amplitudes of the
        SAME initial particle, 
        then generating the AbstractMatrixElement. """

        # Skip empty list
        if not amp_list:
            return

        # Get the abstract initial id        
        ini_pdg = amp_list[0]['process'].get_initial_ids()[0]
        ab_ini_pdg = self.get_particle_type(ini_pdg, get_code=True)
        ab_ini = self.get_particle(ab_ini_pdg)

        for amp in amp_list:
            # Check if this abstract amplitude exists
            final_ids = [l.get('id') for l in amp['process'].get_final_legs()]
            # PDG code start from the initial one
            pseudo_pids, ini_sn_dict = \
                self.get_particlelist_type(final_ids,\
                                               sn_dict = {self.get_particle_type(ini_pdg): 1})
            ab_amp = ab_ini.get_amplitude(pseudo_pids)

            # Construct abstract Amplitude if not exist
            if not ab_amp:
                # Construct the process
                new_process = base_objects.Process({'model': self})
                new_process['legs'] = copy.deepcopy(amp['process']['legs'])
                i = 0
                for new_leg in new_process['legs']:
                    # Final legs
                    if new_leg.get('state'):
                        new_leg['id'] = pseudo_pids[i]
                        i += 1
                    else:
                        new_leg['id'] = ab_ini_pdg

                # Construct Amplitude
                ab_amp = DecayAmplitude({'part_sn_dict': ini_sn_dict})
                ab_amp['process'] = new_process
                try:
                    ab_ini.decay_amplitudes[len(final_ids)].append(ab_amp)
                except KeyError:
                    ab_ini.decay_amplitudes[len(final_ids)] = DecayAmplitudeList([ab_amp])

            # Create the Ab2RealDict for this real amplitude
            # Set the final_legs_dict, using the pseudo_pids.
            ab_amp['ab2real_dicts'].append(Ab2RealDict({'process':
                                                            amp['process'],
                                                        'ab_process':
                                                            ab_amp['process']}))
            ab_amp['ab2real_dicts'][-1].set_final_legs_dict()

            # Scanning the diagrams in real amplitude,
            # Create new diagrams if necessary
            for i, dia in enumerate(amp['diagrams']):
                not_exist = True
                # Check diagrams in abstract amplitude
                for j, ab_dia in enumerate(ab_amp['diagrams']):

                    if not j in ab_amp['ab2real_dicts'][-1]['dia_sn_dict'].keys() and self.compare_diagrams(ab_dia, dia, ab_amp['ab2real_dicts'][-1]):
                        
                        # Update the dia_sn_dict
                        ab_amp['ab2real_dicts'][-1]['dia_sn_dict'][j] = i
                        not_exist = False
                        break

                # Create new diagram if necessary
                if not_exist:
                    # the new abstract diagram corresponds to this real diagram
                    ab_amp['ab2real_dicts'][-1]['dia_sn_dict']\
                        [len(ab_amp['diagrams'])] = i
                    self.add_ab_diagram(ab_amp, dia)

                        
            # Construct the variable dicts
            ab_amp.generate_variables_dicts(amp)


    def generate_ab_matrixelements(self, ab_amplist):
        """ Generate abstract matrix elements for the given
        abstract amplitude list. """

        ab_matrix_elements = AbstractHelasMatrixElementList()

        for ab_amp in ab_amplist:
            ab_matrix_elements.append(AbstractHelasMatrixElement(ab_amp))

        return ab_matrix_elements


    def generate_ab_matrixelements_all(self):
        """ Generate all abstract matrix elements in this model and
        save inside ab_matrix_elements. """

        ab_matrix_elements = AbstractHelasMatrixElementList()

        # Generate all matrix elements in this model
        for part in self['particles']:            
            for clevel, amps in part.decay_amplitudes.items():
                ab_matrix_elements.extend(self.generate_ab_matrixelements(amps))

        self['ab_matrix_elements'] = ab_matrix_elements

        return ab_matrix_elements

#===============================================================================
# Ab2RealDict
#===============================================================================
class Ab2RealDict(base_objects.PhysicsObject):
    """A Reference dict to mapping the information of an abstract amplitude
    to a real amplitude."""


    sorted_keys = ['process', 'dia_sn_dict', 'mass_dict', 'coup_dict',
                   'final_legs_dict']

    def default_setup(self):
        """Default values for all properties."""

        # Dictionary of from serial numbers of diagrams in abstract amplitude
        # to real amplitude
        self['process'] = base_objects.Process()
        self['ab_process'] = base_objects.Process()
        self['dia_sn_dict'] = {}
        self['mass_dict'] = {}
        self['coup_dict'] = {}
        self['final_legs_dict'] = {}

    def set_final_legs_dict(self, ab_object=None, real_dia=None):
        """ Setup the final_legs_dict."""

        # Set real pids
        if real_dia:
            real_pids = [l.get('id') for l in real_dia.get_final_legs()]
            ini_pid = real_dia.get_initial_id()
        else:
            real_pids = [l.get('id') for l in self['process'].get_final_legs()]
            ini_pid = self['process'].get_initial_ids()[0]

        # Set ab pids
        if not ab_object:
            ab_model = self['ab_process']['model']
            ab_pids = ab_model.get_particlelist_type(real_pids, sn_dict = {ab_model.get_particle_type(ini_pid): 1})[0]
        elif isinstance(ab_object, AbstractModel):
            ab_pids = ab_object.get_particlelist_type(real_pids, sn_dict = {ab_object.get_particle_type(ini_pid): 1})[0]
        else:
            raise self.PhysicsObjectError, \
                "The first argument is not necessary, otherwise it should be AbstractModel."

        # Reset final_legs_dict
        self['final_legs_dict'] = {}

        # So far, each ab_pid only has one real_pid
        for k, ab_pid in enumerate(ab_pids):
            if not ab_pid in self['final_legs_dict'].keys():
                self['final_legs_dict'][ab_pid] = real_pids[k]
            elif isinstance(self['final_legs_dict'][ab_pid], list):
                logger.warning('multiple real id correspondence')
                self['final_legs_dict'][ab_pid].append(real_pids[k])
            else:
                logger.warning('multiple real id correspondence')
                self['final_legs_dict'][ab_pid] = \
                    [self['final_legs_dict'][ab_pid], real_pids[k]]

#===============================================================================
# Ab2RealDictList
#===============================================================================
class Ab2RealDictList(base_objects.PhysicsObjectList):
    """A InteractionList, with additional properties that stores the
       abstract interaction type."""

    def is_valid_element(self, obj):
        """ Test if the object is a valid Ab2RealDictList for the list. """
        return isinstance(obj, Ab2RealDict)


#===============================================================================
# AbstractHelasMatrixElement
#===============================================================================
class AbstractHelasMatrixElement(helas_objects.HelasMatrixElement):
    """A HelasMatrixElement that contains the Ab2RealDict."""


    def default_setup(self):
        """Default values for all properties."""

        super(AbstractHelasMatrixElement, self).default_setup()
        self['ab2real_dicts'] = Ab2RealDictList()


    # Customized constructor
    def __init__(self, amplitude=None, optimization=1,
                 decay_ids=[], gen_color=True):
        """Constructor for the AbstractHelasMatrixElement.
        If DecayAmplitude is provided, copy the Ab2RealDictList.
        """

        super(AbstractHelasMatrixElement, self).__init__()

        if isinstance(amplitude, DecayAmplitude):
            self['ab2real_dicts'] = amplitude['ab2real_dicts']


#===============================================================================
# AbstractHelasMatrixElementList
#===============================================================================
class AbstractHelasMatrixElementList(helas_objects.HelasMatrixElementList):
    """List of AbstractHelasMatrixElement objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasMatrixElement for the list."""

        return isinstance(obj, AbstractHelasMatrixElement)

    
#===============================================================================
# Helper function
#===============================================================================
def list_abs(list):

    return [abs(item) for item in list]

def legcmp(x, y):
    """Define the leg comparison, useful when testEqual is execute"""
    mycmp = cmp(x['id'], y['id'])
    if mycmp == 0:
        mycmp = cmp(x['state'], y['state'])
    return mycmp

def legcmp_bynumber(x, y):
    """Define the leg comparison, useful in generation of process in DecayAmplitude."""
    mycmp = cmp(x['number'], y['number'])
    return mycmp

def id_num_cmp(x, y):
    """Define the leg (id, number) sort."""
    mycmp = cmp(x[0], y[0])
    if mycmp == 0:
        mycmp = cmp(x[1], y[1])
    return mycmp

def channelcmp_width(x, y):
    """ Sort the channels by their width."""
    if x['onshell']:
        mycmp = cmp(x['apx_decaywidth'], y['apx_decaywidth'])
    else:
        mycmp = cmp(x['apx_decaywidth_nextlevel'], y['apx_decaywidth_nextlevel'])
    return -mycmp

def channelcmp_final(x, y):
    """ Sort the channels by their final_mass_list. 
        This will be similar to sort by the final state particles."""

    mycmp = cmp(x['final_mass_list'], y['final_mass_list'])

    return -mycmp

def amplitudecmp_width(x, y):
    """ Sort the amplitudes by their width."""
    mycmp = cmp(x['apx_decaywidth'].real, y['apx_decaywidth'].real)

    return -mycmp

def part_type_cmp(x, y):
    """ Sort the abstract particle type."""
    mycmp = cmp(x[0], y[0])

    if mycmp == 0:
        mycmp = cmp(x[1], y[1])

    return mycmp

def part_cmp(x, y):
    """ Sort the particle according to signed pdg_code."""

    return cmp(x.get_pdg_code(), y.get_pdg_code())


