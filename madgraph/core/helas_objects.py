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
        self['interaction_id'] = 0
        self['inter_color'] = []
        self['lorentz'] = []
        self['couplings'] = { (0, 0):'none'}
        # Properties relating to the leg/vertex
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
                        "%s is not a valid integer for wavefunction interaction id" % str(value)

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
                        "%s is not a valid wavefunction state (incoming|outgoing|intermediate)" % \
                                                                    str(value)
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
                        "%s is not a valid integer for wavefunction number" % str(value)

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
                'interaction_id', 'inter_color', 'lorentz',
                'couplings', 'state', 'number_external', 'number',
                'fermionflow', 'mothers']

    # Helper functions

    def is_fermion(self):
        return self.get('spin') % 2 == 0

    def is_boson(self):
        return not self.is_fermion()
    
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
                                   external_wavefunctions):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)) in mothers
        """

        # Use the HelasWavefunctionList helper function
        self.get('mothers').check_and_fix_fermion_flow(\
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   self.get_with_flow('state'))
        return self

    def check_majorana_and_flip_flow(self, found_majorana,
                                     wavefunctions,
                                     diagram_wavefunctions,
                                     external_wavefunctions):
        """Recursive function. Check for Majorana fermion. If found,
        continue down to external leg, then flip all the fermion flows
        on the way back up, in the correct way:
        Only flip fermionflow after the last Majorana fermion; for
        wavefunctions before the last Majorana fermion, instead flip
        particle identities and state.
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
            new_mother = fermion_mother[0].check_majorana_and_flip_flow(\
                found_majorana,
                wavefunctions,
                diagram_wavefunctions,
                external_wavefunctions)

            # If this is Majorana and mother has different fermion
            # flow, it means that we should from now on in the chain
            # flip the particle id and flow state.
            # Otherwise, if mother has different fermion flow, flip
            # flow
            flip_flow = new_mother.get('fermionflow') != self.get('fermionflow') \
                        and not self.get('self_antipart')
            flip_sign = new_mother.get('fermionflow') != self.get('fermionflow') \
                        and self.get('self_antipart') or \
                        new_mother.get('state') != self.get('state')
                        
            # Replace old mother with new mother
            mothers[mothers.index(fermion_mother[0])] = new_mother
                
        # Flip sign if needed
        if flip_flow or flip_sign:
            if self in wavefunctions:
                # Need to create a new copy, since we don't want to change
                # the wavefunction for previous diagrams
                new_wf = copy.copy(self)
                # Wavefunction number is given by: number of external
                # wavefunctions + number of non-external wavefunctions
                # in wavefunctions and diagram_wavefunctions
                number = len(external_wavefunctions) + 1
                number = number + len(filter(lambda wf: \
                                             wf not in external_wavefunctions.values(),
                                         wavefunctions))
                number = number + len(filter(lambda wf: \
                                             wf not in external_wavefunctions.values(),
                                         diagram_wavefunctions))
                new_wf.set('number',number)
                new_wf.set('mothers',mothers)
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
                new_wf.set('is_part', 1 - new_wf.get('is_part'))
                if not new_wf.get('self_antipart'):
                    new_wf.set('pdg_code', -new_wf.get('pdg_code'))

            if new_wf in wavefunctions:
                # Use the copy in wavefunctions instead.
                # Remove this copy from diagram_wavefunctions
                new_wf = wavefunctions[wavefunctions.index(new_wf)]
                diagram_wavefunctions.remove(new_wf)

        # Return the new (or old) wavefunction
        return new_wf

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
        out_fermions = filter(lambda wf: wf.get('state') == 'outgoing',
                              self.get('mothers'))
        in_fermions = filter(lambda wf: wf.get('state') == 'incoming',
                              self.get('mothers'))
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
            
        return fermion_number_list

    def needs_hermitian_conjugate(self):
        """Returns true if there is a fermion flow clash, i.e.,
        there is an odd number of negative fermion flows"""
        
        return filter(lambda wf: wf.get('fermionflow') < 0,
                      self.get('mothers'))

    def get_with_flow(self, name):
        """Generate the is_part and state needed for writing out
        wavefunctions, taking into account the fermion flow"""

        if self.get('fermionflow') > 0:
            # Just return (spin, state)
            return self.get(name)
        
        # If fermionflow is -1, need to flip state
        if name == 'is_part':
            return 1 - self.get('is_part')
        if name == 'state':
            return filter(lambda state: state != self.get('state'),
                          ['incoming', 'outgoing'])[0]
        return self.get(name)

    def get_spin_state_number(self):

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
        if self['spin'] != other['spin'] or \
           self['color'] != other['color'] or \
           self['mass'] != other['mass'] or \
           self['width'] != other['width'] or \
           self['is_part'] != other['is_part'] or \
           self['self_antipart'] != other['self_antipart'] or \
           self['inter_color'] != other['inter_color'] or \
           self['lorentz'] != other['lorentz'] or \
           self['number_external'] != other['number_external'] or \
           self['couplings'] != other['couplings'] or \
           self['fermionflow'] != other['fermionflow'] or \
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
                                   my_state):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)). If found, we need to trace back through the
        mother structure (only looking at fermions), until we find a
        Majorana fermion. Then flip fermion flow along this line all
        the way from the initial clash to the external fermion (in the
        right way, see check_majorana_and_flip_flow), and consider an
        incoming particle with fermionflow -1 as outgoing (and vice
        versa). Continue until we have N(incoming) = N(outgoing).
        """

        # Clash is defined by whether the mothers have N(incoming) !=
        # N(outgoing) after this state has been subtracted
        mother_states = [ wf.get_with_flow('state') for wf in \
                          self ]
        if my_state in mother_states:
            mother_states.remove(my_state)

        Nincoming = len(filter(lambda state: state == 'incoming',
                               mother_states))
        Noutgoing = len(filter(lambda state: state == 'outgoing',
                               mother_states))

        if Nincoming == Noutgoing:
            return True

        fermion_mothers = filter(lambda wf: wf.is_fermion(),
                                 self)
        if my_state in ['incoming', 'outgoing'] and len(fermion_mothers) > 1\
           or len(fermion_mothers) > 2:
            raise self.PhysicsObjectError, \
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

            new_mother = mother.check_majorana_and_flip_flow(found_majorana,
                                                wavefunctions,
                                                diagram_wavefunctions,
                                                external_wavefunctions) 
            # Replace old mother with new mother
            self[self.index(mother)] = new_mother

            # Update counters
            mother_states = [ wf.get_with_flow('state') for wf in \
                             self ]
            if my_state in mother_states:
                mother_states.remove(my_state)

            Nincoming = len(filter(lambda state: state == 'incoming',
                                       mother_states))
            Noutgoing = len(filter(lambda state: state == 'outgoing',
                                       mother_states))
            
        if Nincoming != Noutgoing:
            raise self.PhysicsObjectListError, \
                  "Failed to fix fermion flow, %d != %d" % \
                  (Nincoming, Noutgoing)

        return True

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
                        "%s is not a valid integer for wavefunction interaction id" % str(value)

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
                        "%s is not a valid integer for amplitude number" % str(value)

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

        return ['interaction_id', 'inter_color', 'lorentz', 'couplings', 
                'number', 'mothers']


    # Helper functions

    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)) in mothers
        """

        return self.get('mothers').check_and_fix_fermion_flow(\
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   'nostate')

    def needs_hermitian_conjugate(self):
        """Returns true if there is a fermion flow clash, i.e.,
        there is an odd number of negative fermion flows"""

        return filter(lambda wf: wf.get('fermionflow') < 0,
                      self.get('mothers'))

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
        out_fermions = filter(lambda wf: wf.get('state') == 'outgoing',
                              self.get('mothers'))
        in_fermions = filter(lambda wf: wf.get('state') == 'incoming',
                              self.get('mothers'))
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
                        "%s is not a valid HelasWavefunctionList object" % str(value)
        if name == 'amplitude':
            if not isinstance(value, HelasAmplitude):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasAmplitude object" % str(value)

        if name == 'fermionfactor':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for fermionfactor" % str(value)
            if not value in [-1,0,1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid fermion factor (must be -1, 0 or 1)" % str(value)                

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

        self['diagrams'] = HelasDiagramList()

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'diagrams':
            if not isinstance(value, HelasDiagramList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasDiagramList object" % str(value)
        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['diagrams']
    
    # Customized constructor
    def __init__(self, *arguments):
        """Constructor for the HelasMatrixElement. In particular allows
        generating a HelasMatrixElement from a DiagramList, with
        automatic generation of the necessary wavefunctions
        """

        if arguments:
            if isinstance(arguments[0],diagram_generation.Amplitude):
                super(HelasMatrixElement, self).__init__()
                amplitude = arguments[0]
                optimization = 1
                if len(arguments) > 1 and isinstance(arguments[1],int):
                    optimization = arguments[1]

                self.generate_helas_diagrams(amplitude, optimization)
                self.calculate_fermionfactors(amplitude)
            else:
                super(HelasMatrixElement, self).__init__(arguments[0])
        else:
            super(HelasMatrixElement, self).__init__()
   
    def generate_helas_diagrams(self, amplitude, optimization = 1):
        """Starting from a list of Diagrams from the diagram
        generation, generate the corresponding HelasDiagrams, i.e.,
        the wave functions and amplitudes. Choose
        between default optimization (= 1) or no optimization (= 0,
        for GPU).
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

        # wavefunctions has all the previously defined wavefunctions
        wavefunctions = []

        # Generate wavefunctions for the external particles
        external_wavefunctions = dict([(leg.get('number'),
                                        HelasWavefunction(leg, 0, model)) \
                                       for leg in process.get('legs')])

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

            # Check if last vertex is indentity vertex
            if lastvx.get('id') == 0:
                # Need to "glue together" last and next-to-last
                # vertext, by replacing the (incoming) last leg of the
                # next-to-last vertex with the (outgoing) leg in the
                # last vertex
                nexttolastvertex = vertices.pop()
                legs = nexttolastvertex.get('legs')
                ntlnumber = legs[len(legs)-1].get('number')
                lastleg = filter(lambda leg: leg.get('number') != ntlnumber,
                                 lastvx.get('legs'))[0]
                # Replace the last leg of nexttolastvertex
                legs[len(legs)-1] = lastleg
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
                wf = wf.check_and_fix_fermion_flow(wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions)
                # Wavefunction number is given by: number of external
                # wavefunctions + number of non-external wavefunctions
                # in wavefunctions and diagram_wavefunctions
                if not wf in diagram_wavefunctions:
                    number = len(external_wavefunctions) + 1
                    number = number + len(filter(lambda wf: \
                                                 wf not in external_wavefunctions.values(),
                                                 wavefunctions))
                    number = number + len(filter(lambda wf: \
                                                 wf not in external_wavefunctions.values(),
                                                 diagram_wavefunctions))
                    wf.set('number',number)
                    # Store wavefunction
                    if wf in wavefunctions:
                        wf = wavefunctions[wavefunctions.index(wf)]
                    else:
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
            amp.check_and_fix_fermion_flow(wavefunctions,
                                           diagram_wavefunctions,
                                           external_wavefunctions)

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
            if not leg.get('number') in number_to_wavefunctions:
                # This is an external leg, pick from external_wavefunctions
                wf = external_wavefunctions[leg.get('number')]
                number_to_wavefunctions[leg.get('number')] = wf
                if not wf in wavefunctions:
                    diagram_wavefunctions.append(wf)
            else:
                # The mother is an existing wavefunction
                wf = number_to_wavefunctions[leg.get('number')]
            mothers.append(wf)

        return mothers

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
                add_amplitude(key, value[key])

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

        if wavefunction.get_call_key() in self.get("wavefunctions").keys():
            call = self["wavefunctions"][wavefunction.get_call_key()](wavefunction)
            return call
        else:
            return ""

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude
        corresponding to the key"""

        if amplitude.get_call_key() in self.get("amplitudes").keys():
            call = self["amplitudes"][amplitude.get_call_key()](amplitude)
            return call
        else:
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

#===============================================================================
# HelasFortranModel
#===============================================================================
class HelasFortranModel(HelasModel):
    """The class for writing Helas calls in Fortran, starting from
    HelasWavefunctions and HelasAmplitudes."""

    mother_dict = {1: 'S', 2: 'O', -2: 'I', 3: 'V', 5: 'T'}
    self_dict = {1: 'H', 2: 'F', -2: 'F', 3: 'J', 5: 'U'}
    sort_wf = {'O': 0, 'I': 1, 'S': 2, 'T': 3, 'V': 4}
    sort_amp = {'S': 1, 'V': 2, 'T': 0, 'O': 3, 'I': 4}


    def default_setup(self):

        super(HelasFortranModel, self).default_setup()

        # Add special fortran Helas calls, which are not automatically
        # generated


        # Gluon 4-vertex division tensor calls ggT for the FR sm and mssm
        key = ((3,3,5),tuple('A'))
        call_function = lambda wf: \
                        "      CALL UVVAXX(W(1,%d),W(1,%d),%s,zero,zero,zero,W(1,%d))" % \
                        (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                         HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                         wf.get('couplings')[(0,0)],
                         wf.get('number'))
        self.add_wavefunction(key,call_function)

        key = ((3,5,3),tuple('A'))
        call_function = lambda wf: \
                        "      CALL JVTAXX(W(1,%d),W(1,%d),%s,zero,zero,W(1,%d))" % \
                        (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                         HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                         wf.get('couplings')[(0,0)],
                         wf.get('number'))
        self.add_wavefunction(key,call_function)

        key = ((3,3,5),tuple('A'))
        call_function = lambda amp: \
                        "      CALL VVTAXX(W(1,%d),W(1,%d),W(1,%d),%s,zero,AMP(%d))" % \
                        (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                         HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                         HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                         amp.get('couplings')[(0,0)],
                         amp.get('number'))
        self.add_amplitude(key,call_function)

        # WWZ/a 3-vertices. Note that W+ and W- have switched places
        # compared to the directions in the vvvxxx file.
        key = ((3,3,3),tuple(['WWV']))
        call_function = lambda amp: \
                        "      CALL VVVXXX(W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
                        (HelasFortranModel.get_wplus(amp)[0].get('number'),
                         HelasFortranModel.get_wminus(amp)[0].get('number'),
                         HelasFortranModel.get_gamma_z(amp)[0].get('number'),
                         amp.get('couplings')[(0,0)],
                         amp.get('number'))
        self.add_amplitude(key,call_function)

        key = ((3,3,3),tuple(['WWV']))
        call_function = HelasFortranModel.wwv_jvv
        self.add_wavefunction(key,call_function)


        # W boson 4-vertices. Note that W+ and W- have switched places
        # compared to the directions in the wwwwnx file.

        key = ((3,3,3,3),('WWWWN',''))
        call_function = lambda amp: \
                        "      CALL WWWWNX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,%s,AMP(%d))" % \
                        (HelasFortranModel.get_wplus(amp)[0].get('number'),
                         HelasFortranModel.get_wminus(amp)[0].get('number'),
                         HelasFortranModel.get_wplus(amp)[1].get('number'),
                         HelasFortranModel.get_wminus(amp)[1].get('number'),
                         amp.get('couplings')[(0,0)],
                         amp.get('couplings')[(0,1)],
                         amp.get('number'))
        self.add_amplitude(key,call_function)

        key = ((3,3,3,3),('WWWWN',''))
        call_function = HelasFortranModel.wwww_jwww
        self.add_wavefunction(key,call_function)

        # WWVV vertices. Note that W+ and W- have switched places
        # compared to the directions in the jwwwnx file.
        
        key = ((3,3,3,3),('WWVVN',''))
        call_function = lambda amp: \
                        "      CALL W3W3NX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,%s,AMP(%d))" % \
                        (HelasFortranModel.get_wplus(amp)[0].get('number'),
                         HelasFortranModel.get_gamma_z(amp)[0].get('number'),
                         HelasFortranModel.get_wminus(amp)[0].get('number'),
                         HelasFortranModel.get_gamma_z(amp)[1].get('number'),
                         amp.get('couplings')[(0,0)],
                         amp.get('couplings')[(0,1)],
                         amp.get('number'))
        self.add_amplitude(key,call_function)
        key = ((3,3,3,3),('WWVVN',''))
        call_function = HelasFortranModel.wwvvn_jw3w
        self.add_wavefunction(key,call_function)


    # Definitions of slightly more complicated Helas calls, which
    # require if statements

    @staticmethod
    def wwv_jvv(wf):
        """JVVXXX call for WWV-type vertices. Note that W+ and W-
        have switched places compared to the directions in the jwwwnx
        file."""
        wminus = HelasFortranModel.get_wminus(wf)
        wplus = HelasFortranModel.get_wplus(wf)
        gamma_z = HelasFortranModel.get_gamma_z(wf)
        call = "      CALL JVVXXX(W(1,%d),W(1,%d),%s,%s,%s,W(1,%d))"
        if not gamma_z:
            return call % \
          (wplus[0].get('number'),
           wminus[0].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))
        if not wminus:
            return call % \
          (wplus[0].get('number'),
           gamma_z[0].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))
        if not wplus:
            return call % \
          (gamma_z[0].get('number'),
           wminus[0].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))

    @staticmethod
    def wwww_jwww(wf):
        """JWWWNX call for WWWWN-type vertices."""
        wminus = HelasFortranModel.get_wminus(wf)
        wplus = HelasFortranModel.get_wplus(wf)
        call = "      CALL JWWWNX(W(1,%d),W(1,%d),W(1,%d),%s,%s,%s,%s,W(1,%d))"
        if len(wminus) == 2:
            return call % \
          (wminus[0].get('number'),
           wplus[0].get('number'),
           wminus[1].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('couplings')[(0,1)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))
        if len(wplus) == 2:
            return call % \
          (wplus[0].get('number'),
           wminus[0].get('number'),
           wplus[1].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('couplings')[(0,1)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))

    @staticmethod
    def wwvvn_jw3w(wf):
        """JW3WNX call for WWVVN-type vertices."""
        wminus = HelasFortranModel.get_wminus(wf)
        wplus = HelasFortranModel.get_wplus(wf)
        gamma_z = HelasFortranModel.get_gamma_z(wf)
        call = "      CALL JW3WNX(W(1,%d),W(1,%d),W(1,%d),%s,%s,%s,%s,W(1,%d))"
        if len(gamma_z) == 1:
            return call % \
          (wminus[0].get('number'),
           gamma_z[0].get('number'),
           wplus[0].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('couplings')[(0,1)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))
        if not wminus:
            return call % \
          (gamma_z[0].get('number'),
           wplus[0].get('number'),
           gamma_z[1].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('couplings')[(0,1)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))
        if not wplus:
            return call % \
          (gamma_z[0].get('number'),
           wminus[0].get('number'),
           gamma_z[1].get('number'),
           wf.get('couplings')[(0,0)],
           wf.get('couplings')[(0,1)],
           wf.get('mass'),
           wf.get('width'),
           wf.get('number'))

    def get_wavefunction_call(self, wavefunction):
        """Return the function for writing the wavefunction
        corresponding to the key"""

        val = super(HelasFortranModel, self).get_wavefunction_call(wavefunction)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(wavefunction.get('mothers')) > 3:
            raise self.PhysicsObjectError,\
                  """Automatic generation of Fortran wavefunctions not
                  implemented for > 3 mothers"""

        self.generate_helas_call(wavefunction)
        return super(HelasFortranModel, self).get_wavefunction_call(wavefunction)

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude
        corresponding to the key"""

        val = super(HelasFortranModel, self).get_amplitude_call(amplitude)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(amplitude.get('mothers')) > 4:
            raise self.PhysicsObjectError,\
                  """Automatic generation of Fortran amplitudes not
                  implemented for > 4 mothers"""

        self.generate_helas_call(amplitude)
        return super(HelasFortranModel, self).get_amplitude_call(amplitude)

    def generate_helas_call(self, argument):
            
        if not isinstance(argument, HelasWavefunction) and \
           not isinstance(argument, HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "get_helas_call must be called with wavefunction or amplitude"

        call = "      CALL "

        call_function = None
            
        if isinstance(argument, HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = call + HelasFortranModel.mother_dict[\
                argument.get_spin_state_number()]
            # Fill out with X up to 6 positions
            call = call + 'X' * (17 - len(call))
            call = call + "(P(0,%d),"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                call = call + "%s,NHEL(%d),"
            call = call + "%d*IC(%d),W(1,%d))"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            elif argument.is_boson():
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state')=='initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For fermions, need particle/antiparticle
                                 -(-1)**wf.get_with_flow('is_part'),
                                 wf.get('number_external'),
                                 wf.get('number'))
        else:
            # String is FOVXXX, FIVXXX, JIOXXX etc.
            if isinstance(argument, HelasWavefunction):
                call = call + \
                       HelasFortranModel.self_dict[\
                argument.get_spin_state_number()]

            mother_letters = HelasFortranModel.sorted_letters(argument)

            call = call +''.join(mother_letters)

            # Add C and other addition (for HEFT etc) if needed
            if argument.get('lorentz')[0]:
                # If Lorentz structure is given, by default add this
                # to call name
                call = call + argument.get('lorentz')[0]
            # Check if we need to append a charge conjugation flag
            if argument.needs_hermitian_conjugate():
                call = call + 'C'

            if len(call) > 17:
                raise self.PhysicsObjectError, \
                      "Too long call to Helas routine %s, should be maximum 6 characters" \
                      % call[11:]

            # Fill out with X up to 6 positions
            call = call + 'X' * (17 - len(call)) + '('
            # Wavefunctions
            call = call + "W(1,%d)," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s,"

            if isinstance(argument, HelasWavefunction):
                # Mass and width
                call = call + "%s,%s,"
                # New wavefunction
                call = call + "W(1,%d))"
            else:
                # Amplitude
                call = call + "AMP(%d))"                

            if isinstance(argument,HelasWavefunction):
                # Create call for wavefunction
                if len(argument.get('mothers')) == 2:
                    call_function = lambda wf: call % \
                                    (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                                     #wf.get_coupling_conjugate().values()[0],
                                     wf.get_with_flow('couplings')[(0,0)],
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
                else:
                    call_function = lambda wf: call % \
                                    (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[2].get('number'),
                                     #wf.get_coupling_conjugate().values()[0],
                                     wf.get_with_flow('couplings')[(0,0)],
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
            else:
                # Create call for amplitude
                if len(argument.get('mothers')) == 3:
                    call_function = lambda amp: call % \
                                    (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                                     #amp.get_coupling_conjugate().values()[0],
                                     amp.get('couplings')[(0,0)],
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[3].get('number'),
                                     #amp.get_coupling_conjugate().values()[0],
                                     amp.get('couplings')[(0,0)],
                                     amp.get('number'))

        if isinstance(argument,HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(),call_function)
        else:
            self.add_amplitude(argument.get_call_key(),call_function)
            
    # Static helper functions

    @staticmethod
    def sorted_mothers(arg):
        """Gives a list of mother wavefunctions sorted according to
        1. the spin order needed in the Fortran Helas calls and
        2. the number for the external leg"""

        if isinstance(arg, HelasWavefunction) or \
           isinstance(arg, HelasAmplitude):
            return HelasWavefunctionList(sorted(arg.get('mothers'),
                                                lambda wf1, wf2: \
                                                HelasFortranModel.sort_amp[\
                HelasFortranModel.mother_dict[wf2.get_spin_state_number()]]\
                                                - HelasFortranModel.sort_amp[\
                HelasFortranModel.mother_dict[wf1.get_spin_state_number()]]\
                                                or wf1.get('number_external') - \
                                                wf2.get('number_external')))
    
    @staticmethod
    def sorted_letters(arg):
        """Gives a list of letters sorted according to
        the order of letters in the Fortran Helas calls"""

        if isinstance(arg, HelasWavefunction):
            return sorted([HelasFortranModel.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          HelasFortranModel.sort_wf[l2] - \
                          HelasFortranModel.sort_wf[l1])

        if isinstance(arg, HelasAmplitude):
            return sorted([HelasFortranModel.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          HelasFortranModel.sort_amp[l2] - \
                          HelasFortranModel.sort_amp[l1])
    
    @staticmethod
    def get_wminus(arg):
        """Extracts outgoing W- from mothers"""

        if isinstance(arg, HelasWavefunction) or \
           isinstance(arg, HelasAmplitude):
            return filter(lambda wf: wf.get('pdg_code') == -24,
                          arg.get('mothers'))

    @staticmethod
    def get_wplus(arg):
        """Extracts outgoing W+ from mothers"""

        if isinstance(arg, HelasWavefunction) or \
           isinstance(arg, HelasAmplitude):
            return filter(lambda wf: wf.get('pdg_code') == 24,
                          arg.get('mothers'))

    @staticmethod
    def get_gamma_z(arg):
        """Extracts W+ from mothers"""

        if isinstance(arg, HelasWavefunction) or \
           isinstance(arg, HelasAmplitude):
            return sorted(filter(lambda wf: wf.get('pdg_code') == 22 or \
                          wf.get('pdg_code') == 23,
                          arg.get('mothers')),
                          lambda wf1,wf2: wf2.get('pdg_code') - wf1.get('pdg_code'))
