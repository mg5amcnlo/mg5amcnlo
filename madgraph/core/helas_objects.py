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

import copy
import logging
import re
import itertools
import math
import array

import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation

"""Definitions of objects used to generate Helas calls (language-independent):
HelasWavefunction, HelasAmplitude, HelasDiagram for the generation of
wavefunctions and amplitudes;
HelasParticle, HelasInteraction, HelasModel are language-independent
base classes for the language-specific classes found in the
iolibs directory"""

#===============================================================================
# 
#===============================================================================

#===============================================================================
# HelasWavefunction
#===============================================================================
class HelasWavefunction(base_objects.PhysicsObject):
    """HelasWavefunction object, has the information necessary for
    writing a call to a HELAS wavefunction routine: the PDG number,
    all relevant particle information, a list of mother wavefunctions,
    interaction id, all relevant interaction information, fermion flow
    state, wavefunction number
    """

    def default_setup(self):
        """Default values for all properties"""

        # Properties related to the particle propagator
        # For an electron, would have the following values
        # pdg_code = 11
        # name = 'e-'
        # antiname = 'e+'
        # spin = '1'   defined as 2 x spin + 1  
        # color = '1'  1= singlet, 3 = triplet, 8=octet
        # mass = 'zero'
        # width = 'zero'
        # is_part = 'true'    Particle not antiparticle
        # self_antipart='false'   gluon, photo, h, or majorana would be true
        self['pdg_code'] = 0
        self['name'] = 'none'
        self['antiname'] = 'none'
        self['spin'] = 1
        self['color'] = 1
        self['mass'] = 'zero'
        self['width'] = 'zero'
        self['is_part'] = True
        self['self_antipart'] = False
        # Properties related to the interaction generating the propagator
        # For an e- produced from an e+e-A vertex would have the following
        # proporties:
        # interaction_id = the id of the interaction in the model
        # pdg_codes = the pdg_codes property of the interaction, [11, -11, 22]
        # inter_color = the 'color' property of the interaction: ['C1']
        # lorentz = the 'lorentz' property of the interaction: ['']
        # couplings = the coupling names from the interaction: {(0,0):'MGVX12'}
        self['interaction_id'] = 0
        self['pdg_codes'] = []
        self['inter_color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        # Properties relating to the leg/vertex
        # state = initial/final (for external bosons),
        #         intermediate (for intermediate bosons),
        #         incoming/outgoing (for fermions)
        # number_external = the 'number' property of the corresponding Leg,
        #                   corresponds to the number of the first external
        #                   particle contributing to this leg
        # fermionflow = 1    fermions have +-1 for flow (bosons always +1),
        #                    -1 is used only if there is a fermion flow clash
        #                    due to a Majorana particle 
        self['state'] = 'incoming'
        self['mothers'] = HelasWavefunctionList()
        self['number_external'] = 0
        self['number'] = 0
        self['fermionflow'] = 1
        
    # Customized constructor
    def __init__(self, *arguments):
        """Allow generating a HelasWavefunction from a Leg
        """

        if len(arguments) > 2:
            if isinstance(arguments[0], base_objects.Leg) and \
                   isinstance(arguments[1], int) and \
                   isinstance(arguments[2], base_objects.Model):
                super(HelasWavefunction, self).__init__()
                leg = arguments[0]
                interaction_id = arguments[1]
                model = arguments[2]
                self.set('pdg_code', leg.get('id'), model)
                self.set('number_external', leg.get('number'))
                self.set('number', leg.get('number'))
                self.set('state', leg.get('state'))
                # Set fermion flow state. Initial particle and final
                # antiparticle are incoming, and vice versa for
                # outgoing
                if self.is_fermion():
                    if leg.get('state') == 'initial' and \
                           self.get('is_part') or \
                           leg.get('state') == 'final' and \
                           not self.get('is_part'):
                        self.set('state', 'incoming')
                    else:
                        self.set('state', 'outgoing')
                self.set('interaction_id', interaction_id, model)
        elif arguments:
            super(HelasWavefunction, self).__init__(arguments[0])
            # Set couplings separately, since it needs to be set after
            # color and lorentz
            if 'couplings' in arguments[0].keys():
                self.set('couplings', arguments[0]['couplings'])
        else:
            super(HelasWavefunction, self).__init__()
   
    def filter(self, name, value):
        """Filter for valid wavefunction property values."""

        if name == 'pdg_code':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                      "%s is not a valid pdg_code for wavefunction" % \
                      str(value)

        if name in ['name', 'antiname']:
            # Must start with a letter, followed by letters,  digits,
            # - and + only
            p = re.compile('\A[a-zA-Z]+[\w]*[\-\+]*~?\Z')
            if not p.match(value):
                raise self.PhysicsObjectError, \
                        "%s is not a valid particle name" % value

        if name is 'spin':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Spin %s is not an integer" % repr(value)
            if value < 1 or value > 5:
                raise self.PhysicsObjectError, \
                   "Spin %i is smaller than one" % value

        if name is 'color':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                    "Color %s is not an integer" % repr(value)
            if value not in [1, 3, 6, 8]:
                raise self.PhysicsObjectError, \
                   "Color %i is not valid" % value

        if name in ['mass', 'width']:
            # Must start with a letter, followed by letters, digits or _
            p = re.compile('\A[a-zA-Z]+[\w\_]*\Z')
            if not p.match(value):
                raise self.PhysicsObjectError, \
                        "%s is not a valid name for mass/width variable" % \
                        value

        if name in ['is_part', 'self_antipart']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "%s tag %s is not a boolean" % (name, repr(value))

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer "  % str(value) + \
                        " for wavefunction interaction id"

        if name == 'pdg_codes':
            #Should be a list of strings
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of integers" % str(value)
            for mystr in value:
                if not isinstance(mystr, int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(mystr)

        if name in ['inter_color', 'lorentz']:
            #Should be a list of strings
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of strings" % str(value)
            for mystr in value:
                if not isinstance(mystr, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(mystr)

        if name == 'couplings':
            #Should be a dictionary of strings with (i,j) keys
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary for couplings" % \
                                                                str(value)

            if len(value) != len(self['inter_color']) * len(self['lorentz']):
                raise self.PhysicsObjectError, \
                        "Dictionary " + str(value) + \
                        " for couplings has not the right number of entry"

            for key in value.keys():
                if not isinstance(key, tuple):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple" % str(key)
                if len(key) != 2:
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple with 2 elements" % str(key)
                if not isinstance(key[0], int) or not isinstance(key[1], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple of integer" % str(key)
                if key[0] < 0 or key[1] < 0 or \
                   key[0] >= len(self['inter_color']) or key[1] >= \
                                                    len(self['lorentz']):
                    raise self.PhysicsObjectError, \
                        "%s is not a tuple with valid range" % str(key)
                if not isinstance(value[key], str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(mystr)

        if name == 'state':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for wavefunction state" % \
                                                                    str(value)
            if value not in ['incoming', 'outgoing',
                             'intermediate', 'initial', 'final']:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction "  % str(value) + \
                        "state (incoming|outgoing|intermediate)"
        if name in ['fermionflow']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value)
            if not value in [-1,1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid sign (must be -1 or 1)" % str(value)

        if name in ['number_external', 'number']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value) + \
                        " for wavefunction number"

        if name == 'mothers':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                      "%s is not a valid list of mothers for wavefunction" % \
                      str(value)

        return True

    # Enhanced set function, where we can append a model

    def set(self, *arguments):
        """When setting interaction_id, if model is given (in tuple),
        set all other interaction properties. When setting pdg_code,
        if model is given, set all other particle properties."""

        if len(arguments) < 2:
            raise self.PhysicsObjectError, \
                  "Too few arguments for set"

        name = arguments[0]
        value = arguments[1]
        
        if len(arguments) > 2 and \
               isinstance(value, int) and \
               isinstance(arguments[2], base_objects.Model):
            if name == 'interaction_id':
                self.set('interaction_id', value)
                if value > 0:
                    inter = arguments[2].get('interaction_dict')[value]
                    self.set('pdg_codes',
                             [part.get_pdg_code() for part in \
                              inter.get('particles')])
                    self.set('inter_color', inter.get('color'))
                    self.set('lorentz', inter.get('lorentz'))
                    self.set('couplings', inter.get('couplings'))
                return True
            elif name == 'pdg_code':
                self.set('pdg_code', value)
                part = arguments[2].get('particle_dict')[value]
                self.set('name', part.get('name'))
                self.set('antiname', part.get('antiname'))
                self.set('spin', part.get('spin'))
                self.set('color', part.get('color'))
                self.set('mass', part.get('mass'))
                self.set('width', part.get('width'))
                self.set('is_part', part.get('is_part'))
                self.set('self_antipart', part.get('self_antipart'))
                return True
            else:
                raise self.PhysicsObjectError, \
                      "%s not allowed name for 3-argument set", name
        else:
            return super(HelasWavefunction, self).set(name, value)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['pdg_code', 'name', 'antiname', 'spin', 'color',
                'mass', 'width', 'is_part', 'self_antipart',
                'interaction_id', 'pdg_codes', 'inter_color', 'lorentz',
                'couplings', 'state', 'number_external', 'number',
                'fermionflow', 'mothers']

    # Helper functions

    def is_fermion(self):
        return self.get('spin') % 2 == 0

    def is_boson(self):
        return not self.is_fermion()
    
    def to_array(self):
        """Generate an array with the information needed to uniquely
        determine if a wavefunction has been used before: interaction
        id and mother wavefunction numbers."""
        
        array_rep = array.array('i',[self['interaction_id']])
        array_rep.extend([mother.get('number') for \
                          mother in self['mothers']])
        return array_rep

    def get_pdg_code_outgoing(self):
        """Generate the corresponding pdg_code for an outgoing particle,
        taking into account fermion flow, for mother wavefunctions"""
 
        if self.get('self_antipart'):
            #This is its own antiparticle e.g. a gluon
            return self.get('pdg_code')
 
        if self.is_boson():
            # This is a boson
            return self.get('pdg_code')
 
        if (self.get('state') == 'incoming' and self.get('is_part') \
                or self.get('state') == 'outgoing' and not self.get('is_part')):
            return -self.get('pdg_code')
        else:
            return self.get('pdg_code')

    def get_pdg_code_incoming(self):
        """Generate the corresponding pdg_code for an incoming particle,
        taking into account fermion flow, for mother wavefunctions"""
 
        if self.get('self_antipart'):
            #This is its own antiparticle e.g. gluon
            return self.get('pdg_code')
 
        if self.is_boson():
            # This is a boson
            return -self.get('pdg_code')
 
        if (self.get('state') == 'outgoing' and self.get('is_part') \
                or self.get('state') == 'incoming' and not self.get('is_part')):
            return -self.get('pdg_code')
        else:
            return self.get('pdg_code')


    def set_state_and_particle(self, model):
        """Set incoming/outgoing state according to mother states and
        Lorentz structure of the interaction, and set PDG code
        according to the particles in the interaction"""

        if not isinstance(model, base_objects.Model):
            raise self.PhysicsObjectError, \
                  "%s is not a valid model for call to set_state_and_particle" \
                  % repr(model)
        # Start by setting the state of the wavefunction
        if self.is_boson():
            # For boson, set state to intermediate
            self.set('state', 'intermediate')
        else:
            # For fermion, set state to same as other fermion (in the right way)
            mother_fermions = filter(lambda wf: wf.is_fermion(),
                                     self.get('mothers'))
            if len(mother_fermions) != 1:
                raise self.PhysicsObjectError, \
                      """Multifermion vertices not implemented.
                      Please decompose your vertex into 2-fermion
                      vertices to get fermion flow correct."""
            
            if len(filter(lambda wf: wf.get_with_flow('state') == 'incoming',
                          self.get('mothers'))) > \
                          len(filter(lambda wf: \
                                     wf.get_with_flow('state') == 'outgoing',
                          self.get('mothers'))):
                # If more incoming than outgoing mothers,
                # Pick one with incoming state as mother and set flow
                # Note that this needs to be done more properly if we have
                # 4-fermion vertices
                mother = filter(lambda wf: \
                                wf.get_with_flow('state') == 'incoming',
                                self.get('mothers'))[0]
            else:
                # If more outgoing than incoming mothers,
                # Pick one with outgoing state as mother and set flow
                # Note that this needs to be done more properly if we have
                # 4-fermion vertices
                mother = filter(lambda wf: \
                                wf.get_with_flow('state') == 'outgoing',
                                self.get('mothers'))[0]
            if not self.get('self_antipart'):
                self.set('state', mother.get('state'))
                self.set('fermionflow', mother.get('fermionflow'))
            else:
                self.set('state', mother.get_with_flow('state'))
                self.set('is_part', mother.get_with_flow('is_part'))

        # We want the particle created here to go into the next
        # vertex, so we need to flip identity for incoming
        # antiparticle and outgoing particle.
        if not self.get('self_antipart') and \
               (self.get('state') == 'incoming' and not self.get('is_part') \
                or self.get('state') == 'outgoing' and self.get('is_part')):
            self.set('pdg_code', -self.get('pdg_code'), model)

        return True
        
    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   wf_number):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)) in mothers. This can happen when there is a
        Majorana particle in the diagram, which can flip the fermion
        flow. This is detected either by a wavefunctions or an
        amplitude, with 2 fermion mothers with same state.

        In this case, we need to follow the fermion lines of the
        mother wavefunctions until we find the outermost Majorana
        fermion. For all fermions along the line up to (but not
        including) the Majorana fermion, we need to flip incoming <->
        outgoing and particle id. For all fermions after the Majorana
        fermion, we need to flip the fermionflow property (1 <-> -1).

        The reason for this is that in the Helas calls, we need to
        keep track of where the actual fermion flow clash happens
        (i.e., at the outermost Majorana), as well as having the
        correct fermion flow for all particles along the fermion line.

        This is done by the mothers using
        HelasWavefunctionList.check_and_fix_fermion_flow, which in
        turn calls the recursive function
        check_majorana_and_flip_flow to trace the fermion lines.
        """

        # Use the HelasWavefunctionList helper function
        # Have to keep track of wavefunction number, since we might
        # need to add new wavefunctions.
        wf_number = self.get('mothers').check_and_fix_fermion_flow(\
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   self.get_with_flow('state'),
                                   wf_number)
        return self, wf_number

    def check_majorana_and_flip_flow(self, found_majorana,
                                     wavefunctions,
                                     diagram_wavefunctions,
                                     external_wavefunctions,
                                     wf_number):
        """Recursive function. Check for Majorana fermion. If found,
        continue down to external leg, then flip all the fermion flows
        on the way back up, in the correct way:
        Only flip fermionflow after the last Majorana fermion; for
        wavefunctions before the last Majorana fermion, instead flip
        particle identities and state. Return the new (or old)
        wavefunction, and the present wavefunction number.
        """

        if not found_majorana:
            found_majorana = self.get('self_antipart')

        new_wf = self
        flip_flow = False
        flip_sign = False
        
        # Stop recursion at the external leg
        mothers = copy.copy(self.get('mothers'))
        if not mothers:
            flip_flow = found_majorana
        else:
            # Follow fermion flow up through tree
            fermion_mother = filter(lambda wf: wf.is_fermion() and
                                     wf.get_with_flow('state') == \
                                     self.get_with_flow('state'),
                                     mothers)

            if len(fermion_mother) > 1:
                raise self.PhysicsObjectError,\
                      "6-fermion vertices not yet implemented"
            if len(fermion_mother) == 0:
                raise self.PhysicsObjectError,\
                      "Previous unresolved fermion flow in mother chain"
            
            # Perform recursion by calling on mother
            new_mother, wf_number = fermion_mother[0].\
                                    check_majorana_and_flip_flow(\
                                         found_majorana,
                                         wavefunctions,
                                         diagram_wavefunctions,
                                         external_wavefunctions,
                                         wf_number)

            # If this is Majorana and mother has different fermion
            # flow, it means that we should from now on in the chain
            # flip the particle id and flow state.
            # Otherwise, if mother has different fermion flow, flip
            # flow
            flip_flow = new_mother.get('fermionflow') != \
                        self.get('fermionflow') and \
                        not self.get('self_antipart')
            flip_sign = new_mother.get('fermionflow') != \
                        self.get('fermionflow') and \
                        self.get('self_antipart') or \
                        new_mother.get('state') != self.get('state')
                        
            # Replace old mother with new mother
            mothers[mothers.index(fermion_mother[0])] = new_mother
                
        # Flip sign if needed
        if flip_flow or flip_sign:
            if self in wavefunctions:
                # Need to create a new copy, since we don't want to change
                # the wavefunction for previous diagrams
                new_wf = copy.copy(self)
                # Update wavefunction number
                wf_number = wf_number + 1
                new_wf.set('number', wf_number)
                new_wf.set('mothers', mothers)
                diagram_wavefunctions.append(new_wf)

            # Now flip flow or sign
            if flip_flow:
                # Flip fermion flow
                new_wf.set('fermionflow', -new_wf.get('fermionflow'))
                
            if flip_sign:
                # Flip state and particle identity
                # (to keep particle identity * flow state)
                new_wf.set('state', filter(lambda state: \
                                           state != new_wf.get('state'),
                                           ['incoming', 'outgoing'])[0])
                new_wf.set('is_part', not new_wf.get('is_part'))
                if not new_wf.get('self_antipart'):
                    new_wf.set('pdg_code', -new_wf.get('pdg_code'))

            try:
                # Use the copy in wavefunctions instead.
                # Remove this copy from diagram_wavefunctions
                new_wf = wavefunctions[wavefunctions.index(new_wf)]
                diagram_wavefunctions.remove(new_wf)
                # Since we reuse the old wavefunction, reset wf_number
                wf_number = wf_number - 1
            except ValueError:
                pass

        # Return the new (or old) wavefunction, and the new
        # wavefunction number
        return new_wf, wf_number

    def get_fermion_order(self):
        """Recursive function to get a list of fermion numbers
        corresponding to the order of fermions along fermion lines
        connected to this wavefunction, in the form [n1,n2,...] for a
        boson, and [N,[n1,n2,...]] for a fermion line"""

        # End recursion if external wavefunction
        if not self.get('mothers'):
            if self.is_fermion():
                return [self.get('number_external'),[]]
            else:
                return []

        # Pick out fermion mothers
        out_fermions = filter(lambda wf: wf.get_with_flow('state') == \
                              'outgoing', self.get('mothers'))
        in_fermions = filter(lambda wf: wf.get_with_flow('state') == \
                             'incoming', self.get('mothers'))
        # Pick out bosons
        bosons = filter(lambda wf: wf.is_boson(), self.get('mothers'))

        if self.is_boson() and len(in_fermions) + len(out_fermions) > 2\
               or self.is_fermion() and \
               len(in_fermions) + len(out_fermions) > 1:
            raise self.PhysicsObjectError,\
                  "Multifermion vertices not implemented"

        fermion_number_list = []

        for boson in bosons:
            # Bosons return a list [n1,n2,...]
            fermion_number_list.extend(boson.get_fermion_order())

        if self.is_fermion():
            # Fermions return the result N from their mother
            # and the list from bosons, so [N,[n1,n2,...]]
            fermion_mother = filter(lambda wf: wf.is_fermion(),
                                    self.get('mothers'))[0]
            mother_list = fermion_mother.get_fermion_order()
            fermion_number_list.extend(mother_list[1])
            return [mother_list[0], fermion_number_list]
        elif in_fermions and out_fermions:
            # Combine the incoming and outgoing fermion numbers
            # and add the bosonic numbers: [NI,NO,n1,n2,...]
            in_list = in_fermions[0].get_fermion_order()
            out_list = out_fermions[0].get_fermion_order()
            # Combine to get [N1,N2,n1,n2,...]
            fermion_number_list.append(in_list[0])
            fermion_number_list.append(out_list[0])
            fermion_number_list.extend(in_list[1])
            fermion_number_list.extend(out_list[1])
        elif len(in_fermions) != len(out_fermions):
            raise self.HelasWavefunctionError, \
                  "Error: %d incoming fermions != %d outgoing fermions" % \
                  (len(in_fermions),len(out_fermions))
            
            
        return fermion_number_list

    def needs_hermitian_conjugate(self):
        """Returns true if any of the mothers have negative
        fermionflow"""
        
        return any([wf.get('fermionflow') < 0 for wf in \
                    self.get('mothers')])

    def get_with_flow(self, name):
        """Generate the is_part and state needed for writing out
        wavefunctions, taking into account the fermion flow"""

        if self.get('fermionflow') > 0:
            # Just return (spin, state)
            return self.get(name)
        
        # If fermionflow is -1, need to flip particle identity and state
        if name == 'is_part':
            return not self.get('is_part')
        if name == 'state':
            return filter(lambda state: state != self.get('state'),
                          ['incoming', 'outgoing'])[0]
        return self.get(name)

    def get_spin_state_number(self):
        """Returns the number corresponding to the spin state, with a
        minus sign for incoming fermions"""
        
        state_number = {'incoming': -1, 'outgoing': 1,
                        'intermediate': 1, 'initial': 1, 'final': 1}
        return self.get('fermionflow')* \
               state_number[self.get('state')]* \
               self.get('spin')

    def get_call_key(self):
        """Generate the (spin, state) tuple used as key for the helas call
        dictionaries in HelasModel"""

        res = []
        for mother in self.get('mothers'):
            res.append(mother.get_spin_state_number())

        # Sort according to spin and flow direction
        res.sort()

        res.append(self.get_spin_state_number())
        
        # Check if we need to append a charge conjugation flag
        if self.needs_hermitian_conjugate():
            res.append('C')

        return (tuple(res),tuple(self.get('lorentz')))

    # Overloaded operators
    
    def __eq__(self, other):
        """Overloading the equality operator, to make comparison easy
        when checking if wavefunction is already written, or when
        checking for identical processes. Note that the number for
        this wavefunction, the pdg code, and the interaction id are
        irrelevant, while the numbers for the mothers are important.
        """

        if not isinstance(other,HelasWavefunction):
            return False

        # Check relevant directly defined properties
        if self['number_external'] != other['number_external'] or \
           self['spin'] != other['spin'] or \
           self['is_part'] != other['is_part'] or \
           self['self_antipart'] != other['self_antipart'] or \
           self['fermionflow'] != other['fermionflow'] or \
           self['mass'] != other['mass'] or \
           self['width'] != other['width'] or \
           self['color'] != other['color'] or \
           self['inter_color'] != other['inter_color'] or \
           self['lorentz'] != other['lorentz'] or \
           self['couplings'] != other['couplings'] or \
           self['state'] != other['state']:
            return False

        # Check that mothers have the same numbers (only relevant info)
        return [ mother.get('number') for mother in self['mothers'] ] == \
               [ mother.get('number') for mother in other['mothers'] ]

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

#===============================================================================
# HelasWavefunctionList
#===============================================================================
class HelasWavefunctionList(base_objects.PhysicsObjectList):
    """List of HelasWavefunction objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasWavefunction for the list."""
        
        return isinstance(obj, HelasWavefunction)

    # Helper functions
    
    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   my_state,
                                   wf_number):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)). If found, we need to trace back through the
        mother structure (only looking at fermions), until we find a
        Majorana fermion. Then flip fermion flow along this line all
        the way from the initial clash to the external fermion (in the
        right way, see check_majorana_and_flip_flow), and consider an
        incoming particle with fermionflow -1 as outgoing (and vice
        versa). Continue until we have N(incoming) = N(outgoing).
        
        Since the wavefunction number might get updated, return new
        wavefunction number.
        """

        # Clash is defined by whether the mothers have N(incoming) !=
        # N(outgoing) after this state has been subtracted
        mother_states = [ wf.get_with_flow('state') for wf in \
                          self ]
        try:
            mother_states.remove(my_state)
        except ValueError:
            pass

        Nincoming = len(filter(lambda state: state == 'incoming',
                               mother_states))
        Noutgoing = len(filter(lambda state: state == 'outgoing',
                               mother_states))

        if Nincoming == Noutgoing:
            return wf_number

        fermion_mothers = filter(lambda wf: wf.is_fermion(),
                                 self)
        if my_state in ['incoming', 'outgoing'] and len(fermion_mothers) > 1\
           or len(fermion_mothers) > 2:
            raise self.PhysicsObjectListError, \
                      """Multifermion vertices not implemented.
                      Please decompose your vertex into 2-fermion
                      vertices to get fermion flow correct."""
        
        for mother in fermion_mothers:
            if Nincoming > Noutgoing and \
               mother.get_with_flow('state') == 'outgoing' or \
               Nincoming < Noutgoing and \
               mother.get_with_flow('state') == 'incoming' or \
               Nincoming == Noutgoing:
                # This is not a problematic leg
                continue

            # Call recursive function to check for Majorana fermions
            # and flip fermionflow if found
            found_majorana = False

            new_mother, wf_number = mother.check_majorana_and_flip_flow(\
                                                found_majorana,
                                                wavefunctions,
                                                diagram_wavefunctions,
                                                external_wavefunctions,
                                                wf_number) 
            # Replace old mother with new mother
            self[self.index(mother)] = new_mother

            # Update counters
            mother_states = [ wf.get_with_flow('state') for wf in \
                             self ]
            try:
                mother_states.remove(my_state)
            except ValueError:
                pass

            Nincoming = len(filter(lambda state: state == 'incoming',
                                       mother_states))
            Noutgoing = len(filter(lambda state: state == 'outgoing',
                                       mother_states))
            
        if Nincoming != Noutgoing:
            raise self.PhysicsObjectListError, \
                  "Failed to fix fermion flow, %d != %d" % \
                  (Nincoming, Noutgoing)

        return wf_number

#===============================================================================
# HelasAmplitude
#===============================================================================
class HelasAmplitude(base_objects.PhysicsObject):
    """HelasAmplitude object, has the information necessary for
    writing a call to a HELAS amplitude routine:a list of mother wavefunctions,
    interaction id, amplitude number
    """

    def default_setup(self):
        """Default values for all properties"""

        # Properties related to the interaction generating the propagator
        self['interaction_id'] = 0
        self['pdg_codes'] = []
        self['inter_color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        # Properties relating to the vertex
        self['number'] = 0
        self['mothers'] = HelasWavefunctionList()
        
    # Customized constructor
    def __init__(self, *arguments):
        """Allow generating a HelasAmplitude from a Vertex
        """

        if len(arguments) > 1:
            if isinstance(arguments[0],base_objects.Vertex) and \
               isinstance(arguments[1],base_objects.Model):
                super(HelasAmplitude, self).__init__()
                self.set('interaction_id',
                         arguments[0].get('id'), arguments[1])
        elif arguments:
            super(HelasAmplitude, self).__init__(arguments[0])
            # Set couplings separately, since it needs to be set after
            # color and lorentz
            if 'couplings' in arguments[0].keys():
                self.set('couplings', arguments[0]['couplings'])
        else:
            super(HelasAmplitude, self).__init__()
   
    def filter(self, name, value):
        """Filter for valid property values."""

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for interaction id" % \
                        str(value)

        if name == 'pdg_codes':
            #Should be a list of strings
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of integers" % str(value)
            for mystr in value:
                if not isinstance(mystr, int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(mystr)

        if name in ['inter_color', 'lorentz']:
            #Should be a list of strings
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of strings" % str(value)
            for mystr in value:
                if not isinstance(mystr, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(mystr)

        if name == 'couplings':
            #Should be a dictionary of strings with (i,j) keys
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary for couplings" % \
                                                                str(value)

            if len(value) != len(self['inter_color']) * len(self['lorentz']):
                raise self.PhysicsObjectError, \
                        "Dictionary " + str(value) + \
                        " for couplings has not the right number of entry"

            for key in value.keys():
                if not isinstance(key, tuple):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple" % str(key)
                if len(key) != 2:
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple with 2 elements" % str(key)
                if not isinstance(key[0], int) or not isinstance(key[1], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid tuple of integer" % str(key)
                if key[0] < 0 or key[1] < 0 or \
                   key[0] >= len(self['inter_color']) or key[1] >= \
                                                    len(self['lorentz']):
                    raise self.PhysicsObjectError, \
                        "%s is not a tuple with valid range" % str(key)
                if not isinstance(value[key], str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(mystr)

        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for amplitude number" % \
                        str(value)

        if name == 'mothers':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                      "%s is not a valid list of mothers for amplitude" % \
                      str(value)

        return True

    # Enhanced set function, where we can append a model

    def set(self, *arguments):
        """When setting interaction_id, if model is given (in tuple),
        set all other interaction properties. When setting pdg_code,
        if model is given, set all other particle properties."""

        if len(arguments) < 2:
            raise self.PhysicsObjectError, \
                  "Too few arguments for set"

        name = arguments[0]
        value = arguments[1]
        
        if len(arguments) > 2 and \
               isinstance(value, int) and \
               isinstance(arguments[2], base_objects.Model):
            if name == 'interaction_id':
                self.set('interaction_id', value)
                if value > 0:
                    inter = arguments[2].get('interaction_dict')[value]
                    self.set('pdg_codes',
                             [part.get_pdg_code() for part in \
                              inter.get('particles')])
                    self.set('inter_color', inter.get('color'))
                    self.set('lorentz', inter.get('lorentz'))
                    self.set('couplings', inter.get('couplings'))
                return True
            else:
                raise self.PhysicsObjectError, \
                      "%s not allowed name for 3-argument set", name
        else:
            return super(HelasAmplitude, self).set(name, value)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['interaction_id', 'pdg_codes', 'inter_color', 'lorentz',
                'couplings', 'number', 'mothers']


    # Helper functions

    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   wf_number):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)) in mothers. For documentation, check
        HelasWavefunction.check_and_fix_fermion_flow.
        """

        return self.get('mothers').check_and_fix_fermion_flow(\
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   'nostate',
                                   wf_number)

    def needs_hermitian_conjugate(self):
        """Returns true if any of the mothers have negative
        fermionflow"""
        
        return any([wf.get('fermionflow') < 0 for wf in \
                    self.get('mothers')])

    def get_call_key(self):
        """Generate the (spin, state) tuples used as key for the helas call
        dictionaries in HelasModel"""

        res = []
        for mother in self.get('mothers'):
            res.append(mother.get_spin_state_number())

        # Sort according to spin and flow direction
        res.sort()

        # Check if we need to append a charge conjugation flag
        if self.needs_hermitian_conjugate():
            res.append('C')

        return (tuple(res),tuple(self.get('lorentz')))

    def calculate_fermionfactor(self):
        """Calculate the fermion factor for the diagram corresponding
        to this amplitude"""

        # Pick out fermion mothers
        out_fermions = filter(lambda wf: wf.get_with_flow('state') == \
                              'outgoing', self.get('mothers'))
        in_fermions = filter(lambda wf: wf.get_with_flow('state') == \
                             'incoming', self.get('mothers'))
        # Pick out bosons
        bosons = filter(lambda wf: wf.is_boson(), self.get('mothers'))

        if len(in_fermions) + len(out_fermions) > 2:
            raise self.PhysicsObjectError,\
                  "Multifermion vertices not implemented"

        fermion_number_list = []

        for boson in bosons:
            # Bosons return a list [n1,n2,...]
            fermion_number_list.extend(boson.get_fermion_order())

        if in_fermions and out_fermions:
            # Fermions return the result N from their mother
            # and the list from bosons, so [N,[n1,n2,...]]
            in_list = in_fermions[0].get_fermion_order()
            out_list = out_fermions[0].get_fermion_order()
            # Combine to get [N1,N2,n1,n2,...]
            fermion_number_list.append(in_list[0])
            fermion_number_list.append(out_list[0])
            fermion_number_list.extend(in_list[1])
            fermion_number_list.extend(out_list[1])
        elif len(in_fermions) != len(out_fermions):
            raise self.HelasWavefunctionError, \
                  "Error: %d incoming fermions != %d outgoing fermions" % \
                  (len(in_fermions),len(out_fermions))

        return self.sign_flips_to_order(fermion_number_list)

    def sign_flips_to_order(self, fermions):
        """Gives the sign corresponding to the number of flips needed
        to place the fermion numbers in order"""

        # Perform bubble sort on the fermions, and keep track of
        # the number of flips that are needed

        nflips = 0

        for i in range(len(fermions) - 1):
            for j in range(i+1, len(fermions)):
                if fermions[j] < fermions[i]:
                    tmp  = fermions[i]
                    fermions[i] = fermions[j]
                    fermions[j] = tmp
                    nflips = nflips + 1
            
        return (-1)**nflips

    # Comparison between different amplitudes, to allow check for
    # identical processes. Note that we are then not interested in
    # interaction id, but in all other properties.
    def __eq__(self, other):
        """Comparison between different amplitudes, to allow check for
        identical processes.
        """
        
        if not isinstance(other,HelasAmplitude):
            return False

        # Check relevant directly defined properties
        if self['inter_color'] != other['inter_color'] or \
           self['lorentz'] != other['lorentz'] or \
           self['couplings'] != other['couplings'] or \
           self['number'] != other['number']:
            return False

        # Check that mothers have the same numbers (only relevant info)
        return [ mother.get('number') for mother in self['mothers'] ] == \
               [ mother.get('number') for mother in other['mothers'] ]

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

#===============================================================================
# HelasAmplitudeList
#===============================================================================
class HelasAmplitudeList(base_objects.PhysicsObjectList):
    """List of HelasAmplitude objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasAmplitude for the list."""
        
        return isinstance(obj, HelasAmplitude)


#===============================================================================
# HelasDiagram
#===============================================================================
class HelasDiagram(base_objects.PhysicsObject):
    """HelasDiagram: list of vertices (ordered)
    """

    def default_setup(self):
        """Default values for all properties"""

        self['wavefunctions'] = HelasWavefunctionList()
        self['amplitude'] = HelasAmplitude()
        self['fermionfactor'] = 0

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'wavefunctions':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasWavefunctionList object" % \
                        str(value)
        if name == 'amplitude':
            if not isinstance(value, HelasAmplitude):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasAmplitude object" % str(value)

        if name == 'fermionfactor':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for fermionfactor" % \
                        str(value)
            if not value in [-1,0,1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid fermion factor (-1, 0 or 1)" % \
                        str(value)                

        return True

    def get(self, name):
        """Get the value of the property name."""

        if name == 'fermionfactor' and not self[name]:
            if self['amplitude']:
                self.set('fermionfactor',
                         self.get('amplitude').calculate_fermionfactor())

        return super(HelasDiagram, self).get(name)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['wavefunctions', 'amplitude', 'fermionfactor']
    
#===============================================================================
# HelasDiagramList
#===============================================================================
class HelasDiagramList(base_objects.PhysicsObjectList):
    """List of HelasDiagram objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasDiagram for the list."""

        return isinstance(obj, HelasDiagram)
    
#===============================================================================
# HelasMatrixElement
#===============================================================================
class HelasMatrixElement(base_objects.PhysicsObject):
    """HelasMatrixElement: ordered list of HelasDiagrams
    """

    def default_setup(self):
        """Default values for all properties"""

        self['processes'] = base_objects.ProcessList()
        self['diagrams'] = HelasDiagramList()

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'processes':
            if not isinstance(value, base_objects.ProcessList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ProcessList object" % str(value)
        if name == 'diagrams':
            if not isinstance(value, HelasDiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasDiagramList object" % str(value)
        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['processes', 'diagrams']
    
    # Customized constructor
    def __init__(self, *arguments):
        """Constructor for the HelasMatrixElement. In particular allows
        generating a HelasMatrixElement from an Amplitude, with
        automatic generation of the necessary wavefunctions
        """

        if arguments:
            if isinstance(arguments[0],diagram_generation.Amplitude):
                super(HelasMatrixElement, self).__init__()
                amplitude = arguments[0]
                optimization = 1
                if len(arguments) > 1 and isinstance(arguments[1],int):
                    optimization = arguments[1]

                self.get('processes').append(amplitude.get('process'))
                self.generate_helas_diagrams(amplitude, optimization)
                self.calculate_fermionfactors(amplitude)
            else:
                super(HelasMatrixElement, self).__init__(arguments[0])
        else:
            super(HelasMatrixElement, self).__init__()
   
    # Comparison between different amplitudes, to allow check for
    # identical processes. Note that we are then not interested in
    # interaction id, but in all other properties.
    def __eq__(self, other):
        """Comparison between different matrix elements, to allow check for
        identical processes.
        """
        
        if not isinstance(other,HelasMatrixElement):
            return False

        # Should only check if diagrams and process id are identical
        if self['processes'] and not other['processes'] or \
               self['processes'] and \
               self['processes'][0]['id'] != other['processes'][0]['id'] or \
               self['diagrams'] != other['diagrams']:
            return False

        return True

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

    def generate_helas_diagrams(self, amplitude, optimization = 1):
        """Starting from a list of Diagrams from the diagram
        generation, generate the corresponding HelasDiagrams, i.e.,
        the wave functions and amplitudes. Choose between default
        optimization (= 1, maximum recycling of wavefunctions) or no
        optimization (= 0, no recycling of wavefunctions, useful for
        GPU calculations with very restricted memory).
        """

        if not isinstance(amplitude, diagram_generation.Amplitude) or \
               not isinstance(optimization,int):
            raise self.PhysicsObjectError,\
                  "Missing or erraneous arguments for generate_helas_diagrams"

        diagram_list = amplitude.get('diagrams')
        process = amplitude.get('process')
        model = process.get('model')
        if not diagram_list:
            return

        # All the previously defined wavefunctions
        wavefunctions = []
        # List of minimal information for comparison with previous
        # wavefunctions
        wf_mother_arrays = []
        # Keep track of wavefunction number
        wf_number = 0

        # Generate wavefunctions for the external particles
        external_wavefunctions = dict([(leg.get('number'),
                                        HelasWavefunction(leg, 0, model)) \
                                       for leg in process.get('legs')])
        # Initially, have one wavefunction for each external leg.
        wf_number = len(process.get('legs'))

        # For initial state bosons, need to flip PDG code (if has antipart)
        # since all bosons should be treated as outgoing
        for key in external_wavefunctions.keys():
            wf = external_wavefunctions[key]
            if wf.is_boson() and wf.get('state') == 'initial' and \
               not wf.get('self_antipart'):
                wf.set('pdg_code', -wf.get('pdg_code'))
                wf.set('is_part', not wf.get('is_part'))

        # Now go through the diagrams, looking for undefined wavefunctions

        helas_diagrams = HelasDiagramList()

        for diagram in diagram_list:

            # Dictionary from leg number to wave function, keeps track
            # of the present position in the tree
            number_to_wavefunctions = {}

            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = HelasWavefunctionList()
            
            vertices = copy.copy(diagram.get('vertices'))

            # Single out last vertex, since this will give amplitude
            lastvx = vertices.pop()

            # Check if last vertex is identity vertex
            if lastvx.get('id') == 0:
                # Need to "glue together" last and next-to-last
                # vertext, by replacing the (incoming) last leg of the
                # next-to-last vertex with the (outgoing) leg in the
                # last vertex
                nexttolastvertex = copy.deepcopy(vertices.pop())
                legs = nexttolastvertex.get('legs')
                ntlnumber = legs[-1].get('number')
                lastleg = filter(lambda leg: leg.get('number') != ntlnumber,
                                 lastvx.get('legs'))[0]
                # Replace the last leg of nexttolastvertex
                legs[-1] = lastleg
                lastvx = nexttolastvertex
                # Sort the legs, to get right order of wave functions
                lastvx.get('legs').sort(lambda leg1, leg2: \
                                    leg1.get('number')-leg2.get('number'))

            # Go through all vertices except the last and create
            # wavefunctions
            for vertex in vertices:
                legs = copy.copy(vertex.get('legs'))
                last_leg = legs.pop()
                # Generate list of mothers from legs
                mothers = self.getmothers(legs, number_to_wavefunctions,
                                          external_wavefunctions,
                                          wavefunctions,
                                          diagram_wavefunctions)
                # Now generate new wavefunction for the last leg
                wf = HelasWavefunction(last_leg, vertex.get('id'), model)
                wf.set('mothers', mothers)
                # Need to set incoming/outgoing and
                # particle/antiparticle according to the fermion flow
                # of mothers
                wf.set_state_and_particle(model)
                # Need to check for clashing fermion flow due to
                # Majorana fermions, and modify if necessary
                # Also need to keep track of the wavefunction number.
                wf, wf_number = wf.check_and_fix_fermion_flow(wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wf_number)

# Wavefunction number is given by: number of external
                # wavefunctions + number of non-external wavefunctions
                # in wavefunctions and diagram_wavefunctions
                if not wf in diagram_wavefunctions:
                    wf_number = wf_number + 1
                    wf.set('number', wf_number)
                    # Store wavefunction
                    try:
                        # Use wf_mother_arrays to locate existing wavefunction
                        wf = wavefunctions[wf_mother_arrays.index(wf.to_array())]
                        # Since we reuse the old wavefunction, reset wf_number
                        wf_number = wf_number - 1
                    except ValueError:
                        diagram_wavefunctions.append(wf)
                    number_to_wavefunctions[last_leg.get('number')] = wf

            # Find mothers for the amplitude
            legs = lastvx.get('legs')
            mothers = self.getmothers(legs, number_to_wavefunctions,
                                      external_wavefunctions,
                                      wavefunctions,
                                      diagram_wavefunctions)
                
            # Now generate a HelasAmplitude from the last vertex.
            amp = HelasAmplitude(lastvx, model)
            amp.set('mothers', mothers)
            amp.set('number', diagram_list.index(diagram) + 1)

            # Need to check for clashing fermion flow due to
            # Majorana fermions, and modify if necessary
            wf_number = amp.check_and_fix_fermion_flow(wavefunctions,
                                           diagram_wavefunctions,
                                           external_wavefunctions,
                                           wf_number)

            # Sort the wavefunctions according to number
            diagram_wavefunctions.sort(lambda wf1, wf2: \
                                       wf1.get('number')-wf2.get('number'))

            # Generate HelasDiagram
            helas_diagrams.append(HelasDiagram({ \
                'wavefunctions': diagram_wavefunctions,
                'amplitude': amp
                }))

            if optimization:
                wavefunctions.extend(diagram_wavefunctions)
                wf_mother_arrays.extend([wf.to_array() for wf \
                                         in diagram_wavefunctions])
            else:
                wf_number = len(process.get('legs'))

        self.set('diagrams',helas_diagrams)

    def calculate_fermionfactors(self, amplitude):
        """Generate the fermion factors for all diagrams in the amplitude
        """

        for diagram in self.get('diagrams'):
            diagram.get('fermionfactor')

    # Helper methods

    def getmothers(self, legs, number_to_wavefunctions,
                   external_wavefunctions, wavefunctions,
                   diagram_wavefunctions):
        """Generate list of mothers from number_to_wavefunctions and
        external_wavefunctions"""
        
        mothers = HelasWavefunctionList()

        for leg in legs:
            try:
                # The mother is an existing wavefunction
                wf = number_to_wavefunctions[leg.get('number')]
            except KeyError:
                # This is an external leg, pick from external_wavefunctions
                wf = external_wavefunctions[leg.get('number')]
                number_to_wavefunctions[leg.get('number')] = wf
                if not wf in wavefunctions:
                    diagram_wavefunctions.append(wf)
            mothers.append(wf)

        return mothers

    def get_number_of_wavefunctions(self):
        """Gives the total number of wavefunctions for this amplitude"""
        return sum([ len(d.get('wavefunctions')) for d in \
                       self.get('diagrams')])

    def get_nexternal_ninitial(self):
        """Gives (number or external particles, number of
        incoming particles)"""
        return (len(self.get('processes')[0].get('legs')),
                len(filter(lambda leg: leg.get('state') == 'initial',
                           self.get('processes')[0].get('legs'))))

    def get_helicity_combinations(self):
        """Gives the number of helicity combinations for external
        wavefunctions"""

        if not self.get('processes'):
            return None
        
        model = self.get('processes')[0].get('model')

        return reduce(lambda x, y: x * y,
                      [ len(model.get('particle_dict')[leg.get('id')].\
                            get_helicity_states())\
                        for leg in self.get('processes')[0].get('legs') ])

    def get_helicity_matrix(self):
        """Gives the helicity matrix for external wavefunctions"""

        if not self.get('processes'):
            return None
        
        process = self.get('processes')[0]
        model = process.get('model')

        return apply(itertools.product,[ model.get('particle_dict')[\
                                         leg.get('id')].get_helicity_states()\
                                         for leg in process.get('legs') ])

    def get_denominator_factor(self):
        """Calculate the denominator factor due to:
        Averaging initial state color and spin, and
        identical final state particles"""
        
        model = self.get('processes')[0].get('model')

        initial_legs = filter(lambda leg: leg.get('state') == 'initial', \
                              self.get('processes')[0].get('legs'))

        spin_factor = reduce(lambda x, y: x * y,
                             [ len(model.get('particle_dict')[leg.get('id')].\
                                   get_helicity_states())\
                               for leg in initial_legs ])        

        color_factor = reduce(lambda x, y: x * y,
                              [ model.get('particle_dict')[leg.get('id')].\
                                    get('color')\
                                for leg in initial_legs ])        

        final_legs = filter(lambda leg: leg.get('state') == 'final', \
                              self.get('processes')[0].get('legs'))

        identical_indices = {}
        for leg in final_legs:
            try:
                identical_indices[leg.get('id')] = \
                                    identical_indices[leg.get('id')] + 1
            except KeyError:
                identical_indices[leg.get('id')] = 1
        identical_factor = reduce(lambda x, y: x * y,
                                  [ math.factorial(val) for val in \
                                    identical_indices.values() ])

        return spin_factor * color_factor * identical_factor

#===============================================================================
# HelasMatrixElementList
#===============================================================================
class HelasMatrixElementList(base_objects.PhysicsObjectList):
    """List of HelasMatrixElement objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasMatrixElement for the list."""

        return isinstance(obj, HelasMatrixElement)
    
#===============================================================================
# HelasMultiProcess
#===============================================================================
class HelasMultiProcess(base_objects.PhysicsObject):
    """HelasMultiProcess: Given an AmplitudeList, generate the
    HelasMatrixElements for the Amplitudes, identifying processes with
    identical matrix elements"""

    def default_setup(self):
        """Default values for all properties"""

        self['matrix_elements'] = HelasMatrixElementList()

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'matrix_elements':
            if not isinstance(value, HelasMatrixElementList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasMatrixElementList object" % str(value)

        return True

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['matrix_elements']

    def __init__(self, argument = None):
        """Allow initialization with AmplitudeList"""
        
        if isinstance(argument, diagram_generation.AmplitudeList):
            super(HelasMultiProcess, self).__init__()
            self.generate_matrix_elements(argument)
        elif isinstance(argument, diagram_generation.MultiProcess):
            super(HelasMultiProcess, self).__init__()
            self.generate_matrix_elements(argument.get('amplitudes'))
        elif argument:
            # call the mother routine
            super(HelasMultiProcess, self).__init__(argument)
        else:
            # call the mother routine
            super(HelasMultiProcess, self).__init__()

    def generate_matrix_elements(self, amplitudes):
        """Generate the HelasMatrixElements for the Amplitudes,
        identifying processes with identical matrix elements, as
        defined by HelasMatrixElement.__eq__"""

        if not isinstance(amplitudes, diagram_generation.AmplitudeList):
            raise self.HelasMultiProcessError, \
                  "%s is not valid AmplitudeList" % repr(amplitudes)

        matrix_elements = self.get('matrix_elements')

        for amplitude in amplitudes:
            logging.info("Generating Helas calls for %s" % \
                         amplitude.get('process').nice_string().replace('Process', 'process'))
            matrix_element = HelasMatrixElement(amplitude)
            try:
                # If an identical matrix element is already in the list,
                # then simply add this process to the list of
                # processes for that matrix element
                other_processes = matrix_elements[\
                    matrix_elements.index(matrix_element)].get('processes')
                logging.info("Combining process with %s" % \
                             other_processes[0].nice_string().replace('Process: ',''))
                other_processes.append(amplitude.get('process'))
                
            except ValueError:
                # Otherwise, if the matrix element has any diagrams,
                # add this matrix element.
                if matrix_element.get('processes') and \
                       matrix_element.get('diagrams'):
                    matrix_elements.append(matrix_element)
        

#===============================================================================
# HelasModel
#===============================================================================
class HelasModel(base_objects.PhysicsObject):
    """Language independent base class for writing Helas calls. The
    calls are stored in two dictionaries, wavefunctions and
    amplitudes, with entries being a mapping from a set of spin,
    incoming/outgoing states and Lorentz structure to a function which
    writes the corresponding wavefunction call."""

    def default_setup(self):

        self['name'] = ""
        self['wavefunctions'] = {}
        self['amplitudes'] = {}

    def filter(self, name, value):
        """Filter for model property values"""

        if name == 'name':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a string" % \
                                                            type(value)

        if name == 'wavefunctions':
            # Should be a dictionary of functions returning strings, 
            # with keys (spins, flow state)
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary for wavefunction" % \
                                                                str(value)

            for key in value.keys():
                self.add_wavefunction(key, value[key])

        if name == 'amplitudes':
            # Should be a dictionary of functions returning strings, 
            # with keys (spins, flow state)
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dictionary for amplitude" % \
                                                                str(value)

            for key in value.keys():
                self.add_amplitude(key, value[key])

        return True

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['name', 'wavefunctions', 'amplitudes']

    def get_matrix_element_calls(self, matrix_element):
        """Return a list of strings, corresponding to the Helas calls
        for the matrix element"""

        if not isinstance(matrix_element, HelasMatrixElement):
            raise self.PhysicsObjectError, \
                  "%s not valid argument for get_matrix_element_calls" % \
                  repr(matrix_element)

        res = []
        for diagram in matrix_element.get('diagrams'):
            res.extend([ self.get_wavefunction_call(wf) for \
                         wf in diagram.get('wavefunctions') ])
            res.append(self.get_amplitude_call(diagram.get('amplitude')))

        return res

    def get_wavefunction_call(self, wavefunction):
        """Return the function for writing the wavefunction
        corresponding to the key"""

        try:
            call = self["wavefunctions"][wavefunction.get_call_key()](\
                wavefunction)
            return call
        except KeyError:
            return ""

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude
        corresponding to the key"""

        try:
            call = self["amplitudes"][amplitude.get_call_key()](amplitude)
            return call
        except KeyError:
            return ""

    def add_wavefunction(self, key, function):
        """Set the function for writing the wavefunction
        corresponding to the key"""


        if not isinstance(key, tuple):
            raise self.PhysicsObjectError, \
                  "%s is not a valid tuple for wavefunction key" % \
                  str(key)

        if not callable(function):
            raise self.PhysicsObjectError, \
                  "%s is not a valid function for wavefunction string" % \
                  str(function)

        self.get('wavefunctions')[key] = function
        return True
        
    def add_amplitude(self, key, function):
        """Set the function for writing the amplitude
        corresponding to the key"""


        if not isinstance(key, tuple):
            raise self.PhysicsObjectError, \
                  "%s is not a valid tuple for amplitude key" % \
                  str(key)

        if not callable(function):
            raise self.PhysicsObjectError, \
                  "%s is not a valid function for amplitude string" % \
                  str(function)

        self.get('amplitudes')[key] = function
        return True
        
    # Customized constructor
    def __init__(self, argument = {}):
        """Allow generating a HelasModel from a Model
        """

        if isinstance(argument,base_objects.Model):
            super(HelasModel, self).__init__()
            self.set('name',argument.get('name'))
        else:
            super(HelasModel, self).__init__(argument)
