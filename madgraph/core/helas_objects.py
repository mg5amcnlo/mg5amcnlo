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
from madgraph.core import base_objects
"""Definitions of objects used to generate language-independent Helas
calls: HelasWavefunction, HelasAmplitude, HelasDiagram for the
generation of wavefunctions and amplitudes, HelasMatrixElement and
HelasMultiProcess for generation of complete matrix elements for
single and multiple processes; and HelasModel, which is the
language-independent base class for the language-specific classes for
writing Helas calls, found in the iolibs directory"""

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
# 
#===============================================================================

logger = logging.getLogger('madgraph.helas_objects')

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
        self['particle'] = base_objects.Particle()
        self['antiparticle'] = base_objects.Particle()
        self['is_part'] = True
        # Properties related to the interaction generating the propagator
        # For an e- produced from an e+e-A vertex would have the following
        # proporties:
        # interaction_id = the id of the interaction in the model
        # pdg_codes = the pdg_codes property of the interaction, [11, -11, 22]
        # inter_color = the 'color' property of the interaction: []
        # lorentz = the 'lorentz' property of the interaction: ['']
        # couplings = the coupling names from the interaction: {(0,0):'MGVX12'}
        self['interaction_id'] = 0
        self['pdg_codes'] = []
        self['orders'] = {}
        self['inter_color'] = None
        self['lorentz'] = ''
        self['coupling'] = 'none'
        # The Lorentz and color index used in this wavefunction
        self['coupl_key'] = (0, 0)
        # Properties relating to the leg/vertex
        # state = initial/final (for external bosons),
        #         intermediate (for intermediate bosons),
        #         incoming/outgoing (for fermions)
        # leg_state = initial/final for initial/final legs
        #             intermediate for non-external wavefunctions
        # number_external = the 'number' property of the corresponding Leg,
        #                   corresponds to the number of the first external
        #                   particle contributing to this leg
        # fermionflow = 1    fermions have +-1 for flow (bosons always +1),
        #                    -1 is used only if there is a fermion flow clash
        #                    due to a Majorana particle 
        self['state'] = 'incoming'
        self['leg_state'] = True
        self['mothers'] = HelasWavefunctionList()
        self['number_external'] = 0
        self['number'] = 0
        self['fermionflow'] = 1
        # The decay flag is used in processes with defined decay chains,
        # to indicate that this wavefunction has a decay defined
        self['decay'] = False
        # The onshell flag is used in processes with defined decay
        # chains, to indicate that this wavefunction is decayed and
        # should be onshell
        self['onshell'] = False

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
                # decay_ids is the pdg codes for particles with decay
                # chains defined
                decay_ids = []
                if len(arguments) > 3:
                    decay_ids = arguments[3]
                self.set('particle', leg.get('id'), model)
                self.set('number_external', leg.get('number'))
                self.set('number', leg.get('number'))
                self.set('state', {False: 'initial', True: 'final'}[leg.get('state')])
                self.set('leg_state', leg.get('state'))
                # Need to set 'decay' to True for particles which will be
                # decayed later, in order to not combine such processes
                # although they might have identical matrix elements before
                # the decay is applied
                if self['state'] == 'final' and self.get('pdg_code') in decay_ids:
                    self.set('decay', True)

                # Set fermion flow state. Initial particle and final
                # antiparticle are incoming, and vice versa for
                # outgoing
                if self.is_fermion():
                    if leg.get('state') == False and \
                           self.get('is_part') or \
                           leg.get('state') == True and \
                           not self.get('is_part'):
                        self.set('state', 'incoming')
                    else:
                        self.set('state', 'outgoing')
                self.set('interaction_id', interaction_id, model)
        elif arguments:
            super(HelasWavefunction, self).__init__(arguments[0])
        else:
            super(HelasWavefunction, self).__init__()

    def filter(self, name, value):
        """Filter for valid wavefunction property values."""

        if name in ['particle', 'antiparticle']:
            if not isinstance(value, base_objects.Particle):
                raise self.PhysicsObjectError, \
                    "%s tag %s is not a particle" % (name, repr(value))            

        if name == 'is_part':
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                    "%s tag %s is not a boolean" % (name, repr(value))

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer " % str(value) + \
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

        if name == 'orders':
            #Should be a dict with valid order names ask keys and int as values
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict for coupling orders" % \
                                                                    str(value)
            for order in value.keys():
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(order)
                if not isinstance(value[order], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value[order])


        if name == 'inter_color':
            # Should be None or a color string
            if value and not isinstance(value, color.ColorString):
                    raise self.PhysicsObjectError, \
                            "%s is not a valid Color String" % str(value)

        if name == 'lorentz':
            #Should be a string
            if not isinstance(value, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(value)

        if name == 'coupling':
            #Should be a string
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid coupling string" % \
                                                                str(value)

        if name == 'coupl_key':
            if value and not isinstance(value, tuple):
                raise self.PhysicsObjectError, \
                      "%s is not a valid tuple" % str(value)
            if len(value) != 2:
                raise self.PhysicsObjectError, \
                      "%s is not a valid tuple with 2 elements" % str(value)
            if not isinstance(value[0], int) or not isinstance(value[1], int):
                raise self.PhysicsObjectError, \
                      "%s is not a valid tuple of integer" % str(value)

        if name == 'state':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for wavefunction state" % \
                                                                    str(value)
            if value not in ['incoming', 'outgoing',
                             'intermediate', 'initial', 'final']:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction " % str(value) + \
                        "state (incoming|outgoing|intermediate)"
        if name == 'leg_state':
            if value not in [False, True]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction " % str(value) + \
                        "state (incoming|outgoing|intermediate)"
        if name in ['fermionflow']:
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value)
            if not value in [-1, 1]:
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

        if name in ['decay', 'onshell']:
            if not isinstance(value, bool):
                raise self.PhysicsObjectError, \
                        "%s is not a valid bool" % str(value) + \
                        " for decay or onshell"

        return True

    # Enhanced get function, where we can directly call the properties of the particle
    def get(self, name):
        """When calling any property related to the particle,
        automatically call the corresponding property of the particle."""

        if name in ['spin', 'mass', 'width', 'self_antipart']:
            return self['particle'].get(name)
        elif name == 'pdg_code':
            return self['particle'].get_pdg_code()
        elif name == 'color':
            return self['particle'].get_color()
        elif name == 'name':
            return self['particle'].get_name()
        elif name == 'antiname':
            return self['particle'].get_anti_name()
        else:
            return super(HelasWavefunction, self).get(name)
        

    # Enhanced set function, where we can append a model

    def set(self, *arguments):
        """When setting interaction_id, if model is given (in tuple),
        set all other interaction properties. When setting pdg_code,
        if model is given, set all other particle properties."""

        assert len(arguments) >1, "Too few arguments for set"

        name = arguments[0]
        value = arguments[1]

        if len(arguments) > 2 and \
               isinstance(value, int) and \
               isinstance(arguments[2], base_objects.Model):
            model = arguments[2]
            if name == 'interaction_id':
                self.set('interaction_id', value)
                if value > 0:
                    inter = model.get('interaction_dict')[value]
                    self.set('pdg_codes',
                             [part.get_pdg_code() for part in \
                              inter.get('particles')])
                    self.set('orders', inter.get('orders'))
                    # Note that the following values might change, if
                    # the relevant color/lorentz/coupling is not index 0
                    if inter.get('color'):
                        self.set('inter_color', inter.get('color')[0])
                    if inter.get('lorentz'):
                        self.set('lorentz', inter.get('lorentz')[0])
                    if inter.get('couplings'):
                        self.set('coupling', inter.get('couplings').values()[0])
                return True
            elif name == 'particle':
                self.set('particle', model.get('particle_dict')[value])
                self.set('is_part', self['particle'].get('is_part'))
                if self['particle'].get('self_antipart'):
                    self.set('antiparticle', self['particle'])
                else:
                    self.set('antiparticle', model.get('particle_dict')[-value])
                return True
            else:
                raise self.PhysicsObjectError, \
                      "%s not allowed name for 3-argument set", name
        else:
            return super(HelasWavefunction, self).set(name, value)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['particle', 'antiparticle', 'is_part',
                'interaction_id', 'pdg_codes', 'orders', 'inter_color', 
                'lorentz', 'coupling', 'coupl_key', 'state', 'number_external',
                'number', 'fermionflow', 'mothers']

    # Helper functions

    def flip_part_antipart(self):
        """Flip between particle and antiparticle."""
        part = self.get('particle')
        self.set('particle', self.get('antiparticle'))
        self.set('antiparticle', part)

    def is_fermion(self):
        return self.get('spin') % 2 == 0

    def is_boson(self):
        return not self.is_fermion()

    def to_array(self):
        """Generate an array with the information needed to uniquely
        determine if a wavefunction has been used before: interaction
        id and mother wavefunction numbers."""

        # Identification based on interaction id
        array_rep = array.array('i', [self['interaction_id']])
        # Need the coupling key, to distinguish between
        # wavefunctions from the same interaction but different
        # Lorentz or color structures
        array_rep.extend(list(self['coupl_key']))
        # Finally, the mother numbers
        array_rep.extend([mother['number'] for \
                          mother in self['mothers']])
        return array_rep

    def get_pdg_code(self):
        """Generate the corresponding pdg_code for an outgoing particle,
        taking into account fermion flow, for mother wavefunctions"""

        return self.get('pdg_code')

    def get_anti_pdg_code(self):
        """Generate the corresponding pdg_code for an incoming particle,
        taking into account fermion flow, for mother wavefunctions"""

        if self.get('self_antipart'):
            #This is its own antiparticle e.g. gluon
            return self.get('pdg_code')

        return - self.get('pdg_code')

    def set_scalar_coupling_sign(self, model):
        """Check if we need to add a minus sign due to non-identical
        bosons in HVS type couplings"""

        inter = model.get('interaction_dict')[self.get('interaction_id')]
        if [p.get('spin') for p in \
                   inter.get('particles')] == [3, 1, 1]:
            particles = inter.get('particles')
            #                   lambda p1, p2: p1.get('spin') - p2.get('spin'))
            if particles[1].get_pdg_code() != particles[2].get_pdg_code() \
                   and self.get('pdg_code') == \
                       particles[1].get_anti_pdg_code():
                # We need a minus sign in front of the coupling
                self.set('coupling', '-' + self.get('coupling'))


    def set_state_and_particle(self, model):
        """Set incoming/outgoing state according to mother states and
        Lorentz structure of the interaction, and set PDG code
        according to the particles in the interaction"""

        assert isinstance(model, base_objects.Model), \
                  "%s is not a valid model for call to set_state_and_particle" \
                  % repr(model)

        # leg_state is final, unless there is exactly one initial 
        # state particle involved in the combination -> t-channel
        if len(filter(lambda mother: mother.get('leg_state') == False,
                      self.get('mothers'))) == 1:
            leg_state = False
        else:
            leg_state = True
        self.set('leg_state', leg_state)

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
                                     wf_number, force_flip_flow=False):
        """Recursive function. Check for Majorana fermion. If found,
        continue down to external leg, then flip all the fermion flows
        on the way back up, in the correct way:
        Only flip fermionflow after the last Majorana fermion; for
        wavefunctions before the last Majorana fermion, instead flip
        particle identities and state. Return the new (or old)
        wavefunction, and the present wavefunction number.

        Arguments:
          found_majorana: boolean
          wavefunctions: HelasWavefunctionList with previously
                         defined wavefunctions
          diagram_wavefunctions: HelasWavefunctionList with the wavefunctions
                         already defined in this diagram
          external_wavefunctions: dictionary from legnumber to external wf
          wf_number: The present wavefunction number
        """

        if not found_majorana:
            found_majorana = self.get('self_antipart')

        new_wf = self
        flip_flow = False
        flip_sign = False

        # Stop recursion at the external leg
        mothers = copy.copy(self.get('mothers'))
        if not mothers:
            if force_flip_flow:
                flip_flow = True
            elif not self.get('self_antipart'):
                flip_flow = found_majorana
            else:
                flip_sign = found_majorana
        else:
            # Follow fermion flow up through tree
            fermion_mother = filter(lambda wf: wf.is_fermion() and
                                     wf.get_with_flow('state') == \
                                     self.get_with_flow('state'),
                                     mothers)

            if len(fermion_mother) > 1:
                raise self.PhysicsObjectError, \
                      "6-fermion vertices not yet implemented"
            if len(fermion_mother) == 0:
                # Fermion flow already resolved for mothers
                fermion_mother = filter(lambda wf: wf.is_fermion(),
                                    mothers)
                new_mother = fermion_mother[0]
            else:
                # Perform recursion by calling on mother
                new_mother, wf_number = fermion_mother[0].\
                                        check_majorana_and_flip_flow(\
                                           found_majorana,
                                           wavefunctions,
                                           diagram_wavefunctions,
                                           external_wavefunctions,
                                           wf_number,
                                           force_flip_flow)

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
                try:
                    # In call from insert_decay, we want to replace
                    # also identical wavefunctions in the same diagram
                    old_wf_index = diagram_wavefunctions.index(self)
                    diagram_wavefunctions[old_wf_index] = new_wf
                except ValueError:
                    diagram_wavefunctions.append(new_wf)
                    # Make sure that new_wf comes before any wavefunction
                    # which has it as mother
                    for i, wf in enumerate(diagram_wavefunctions):
                        if self in wf.get('mothers'):
                            # Remove new_wf, in order to insert it below
                            diagram_wavefunctions.pop()
                            # Update wf numbers
                            new_wf.set('number', wf.get('number'))
                            for w in diagram_wavefunctions[i:]:
                                w.set('number', w.get('number') + 1)
                            # Insert wavefunction
                            diagram_wavefunctions.insert(i, new_wf)
                            break

            # Set new mothers
            new_wf.set('mothers', mothers)

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

            try:
                # Use the copy in wavefunctions instead.
                # Remove this copy from diagram_wavefunctions
                new_wf = wavefunctions[wavefunctions.index(new_wf)]
                index = diagram_wavefunctions.index(new_wf)
                diagram_wavefunctions.pop(index)
                # We need to decrease the wf number for later
                # diagram wavefunctions
                for wf in diagram_wavefunctions[index:]:
                    wf.set('number', wf.get('number') - 1)
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
                return [self.get('number_external'), []]
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
            raise self.PhysicsObjectError, \
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
            raise self.PhysicsObjectError, \
                  "Error: %d incoming fermions != %d outgoing fermions" % \
                  (len(in_fermions), len(out_fermions))


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

        state_number = {'incoming':-1, 'outgoing': 1,
                        'intermediate': 1, 'initial': 1, 'final': 1}
        return self.get('fermionflow') * \
               state_number[self.get('state')] * \
               self.get('spin')

    def find_outgoing_number(self):
        "Return the position of the resulting particles in the interactions"
        # First shot: just the index in the interaction
        if self.get('interaction_id') == 0:
            return 0
        
        wf_indices = self.get('pdg_codes')
        # take the last index in case of identical particles
        wf_indices.reverse() 
        wf_index = len(wf_indices) - wf_indices.index(self.get_anti_pdg_code()) -1
        wf_indices.reverse() # restore the ordering
        #wf_index = self.get('pdg_codes').index(self.get_anti_pdg_code())
        # If fermion, then we need to correct for I/O status
        spin_state = self.get_spin_state_number()
        if spin_state % 2 == 0:
            if wf_index % 2 == 0 and spin_state < 0:
                # Outgoing particle at even slot -> increase by 1
                wf_index += 1
            elif wf_index % 2 == 1 and spin_state > 0:
                # Incoming particle at odd slot -> decrease by 1
                wf_index -= 1
        return wf_index + 1

    def get_call_key(self):
        """Generate the (spin, number, C-state) tuple used as key for
        the helas call dictionaries in HelasModel"""

        res = []
        for mother in self.get('mothers'):
            res.append(mother.get_spin_state_number())

        # Sort according to spin and flow direction
        res.sort()

        res.append(self.get_spin_state_number())

        res.append(self.find_outgoing_number())

        # Check if we need to append a charge conjugation flag
        if self.needs_hermitian_conjugate():
            res.append('C')

        return (tuple(res), self.get('lorentz'))

    def get_base_vertices(self, wf_dict = {}, vx_list = [], optimization = 1):
        """Recursive method to get a base_objects.VertexList
        corresponding to this wavefunction and its mothers."""

        vertices = base_objects.VertexList()

        if not self.get('mothers'):
            return vertices

        # Add vertices for all mothers
        for mother in self.get('mothers'):
            # This is where recursion happens
            vertices.extend(mother.get_base_vertices(wf_dict, vx_list,
                                                     optimization))
        # Generate last vertex
        legs = base_objects.LegList()

        # We use the from_group flag to indicate whether this outgoing
        # leg corresponds to a decaying (onshell) particle or not
        try:
            lastleg = wf_dict[self.get('number')]
        except KeyError:            
            lastleg = base_objects.Leg({
                'id': self.get_pdg_code(),
                'number': self.get('number_external'),
                'state': self.get('leg_state'),
                'from_group': self.get('onshell')
                })
            if optimization != 0:
                wf_dict[self.get('number')] = lastleg

        for mother in self.get('mothers'):
            try:
                leg = wf_dict[mother.get('number')]
            except KeyError:
                leg = base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'from_group': mother.get('onshell')
                    })
                if optimization != 0:
                    wf_dict[mother.get('number')] = leg
            legs.append(leg)

        legs.append(lastleg)

        vertex = base_objects.Vertex({
            'id': self.get('interaction_id'),
            'legs': legs})

        try:
            index = vx_list.index(vertex)
            vertex = vx_list[index]
        except ValueError:
            pass
        
        vertices.append(vertex)

        return vertices

    def get_color_indices(self):
        """Recursive method to get the color indices corresponding to
        this wavefunction and its mothers."""

        if not self.get('mothers'):
            return []

        color_indices = []

        # Add color indices for all mothers
        for mother in self.get('mothers'):
            # This is where recursion happens
            color_indices.extend(mother.get_color_indices())
        # Add this wf's color index
        color_indices.append(self.get('coupl_key')[0])

        return color_indices

    def get_s_and_t_channels(self, ninitial, mother_leg):
        """Returns two lists of vertices corresponding to the s- and
        t-channels that can be traced from this wavefunction, ordered
        from the outermost s-channel and in/down towards the highest
        number initial state leg."""

        schannels = base_objects.VertexList()
        tchannels = base_objects.VertexList()

        mother_leg = copy.copy(mother_leg)

        # Add vertices for all s-channel mothers
        final_mothers = filter(lambda wf: wf.get('number_external') > ninitial,
                               self.get('mothers'))

        for mother in final_mothers:
            schannels.extend(mother.get_base_vertices(optimization = 0))

        # Extract initial state mothers
        init_mothers = filter(lambda wf: wf.get('number_external') <= ninitial,
                              self.get('mothers'))

        assert len(init_mothers) < 3 , \
                   "get_s_and_t_channels can only handle up to 2 initial states"

        if len(init_mothers) == 1:
            # This is an s-channel or t-channel leg, or the initial
            # leg of a decay process. Add vertex and continue stepping
            # down towards external initial state
            legs = base_objects.LegList()

            if ninitial == 1 or init_mothers[0].get('number_external') == 2 \
                   or init_mothers[0].get('leg_state') == True:
                # This is an s-channel or a leg on its way towards the
                # final vertex
                mothers = final_mothers + init_mothers
            else:
                # This is a t-channel leg going up towards leg number 1
                mothers = init_mothers + final_mothers

            for mother in mothers:
                legs.append(base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'from_group': False
                    }))

            if ninitial == 1 or init_mothers[0].get('number_external') == 2 \
                   or init_mothers[0].get('leg_state') == True:
                # For decay processes or if this is an s-channel leg
                # or we are going towards external leg 2, mother leg
                # is one of the mothers
                legs.insert(-1, mother_leg)
                # Also need to switch direction of the resulting s-channel
                legs[-1].set('id', init_mothers[0].get_anti_pdg_code())
            else:
                # If the mother is going
                # towards external leg 1, mother leg is resulting wf
                legs.append(mother_leg)

            # Renumber resulting leg according to minimum leg number
            legs[-1].set('number', min([l.get('number') for l in legs[:-1]]))

            vertex = base_objects.Vertex({
                'id': self.get('interaction_id'),
                'legs': legs})

            # Add s- and t-channels from further down
            mother_s, tchannels = \
                      init_mothers[0].get_s_and_t_channels(ninitial, legs[-1])

            schannels.extend(mother_s)

            if ninitial == 1 or init_mothers[0].get('leg_state') == True:
                # This leg is s-channel
                schannels.append(vertex)
            elif init_mothers[0].get('number_external') == 1:
                # If we are going towards external leg 1, add to
                # t-channels, at end
                tchannels.append(vertex)
            else:
                # If we are going towards external leg 2, add to
                # t-channels, at start
                tchannels.insert(0, vertex)

        elif len(init_mothers) == 2:
            # This is a t-channel junction. Start with the leg going
            # towards external particle 1, and then do external
            # particle 2
            init_mothers1 = filter(lambda wf: wf.get('number_external') == 1,
                                   init_mothers)[0]
            init_mothers2 = filter(lambda wf: wf.get('number_external') == 2,
                                   init_mothers)[0]

            # Create vertex
            legs = base_objects.LegList()
            for mother in final_mothers + [init_mothers1, init_mothers2]:
                legs.append(base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'from_group': False
                    }))
            legs.insert(-1, mother_leg)

            # Renumber resulting leg according to minimum leg number
            legs[-1].set('number', min([l.get('number') for l in legs[:-1]]))

            vertex = base_objects.Vertex({
                'id': self.get('interaction_id'),
                'legs': legs})

            # Add s- and t-channels going down towards leg 1
            mother_s, tchannels = \
                      init_mothers1.get_s_and_t_channels(ninitial, legs[0])
            schannels.extend(mother_s)

            # Add vertex
            tchannels.append(vertex)

            # Add s- and t-channels going down towards leg 2
            mother_s, mother_t = \
                      init_mothers2.get_s_and_t_channels(ninitial, legs[-1])
            schannels.extend(mother_s)
            tchannels.extend(mother_t)

        return schannels, tchannels

    # Overloaded operators

    def __eq__(self, other):
        """Overloading the equality operator, to make comparison easy
        when checking if wavefunction is already written, or when
        checking for identical processes. Note that the number for
        this wavefunction, the pdg code, and the interaction id are
        irrelevant, while the numbers for the mothers are important.
        """

        if not isinstance(other, HelasWavefunction):
            return False

        # Check relevant directly defined properties
        if self['number_external'] != other['number_external'] or \
           self['fermionflow'] != other['fermionflow'] or \
           self['coupl_key'] != other['coupl_key'] or \
           self['lorentz'] != other['lorentz'] or \
           self['coupling'] != other['coupling'] or \
           self['state'] != other['state'] or \
           self['onshell'] != other['onshell'] or \
           self.get('spin') != other.get('spin') or \
           self.get('self_antipart') != other.get('self_antipart') or \
           self.get('mass') != other.get('mass') or \
           self.get('width') != other.get('width') or \
           self.get('color') != other.get('color') or \
           self['decay'] != other['decay'] or \
           self['decay'] and self['particle'] != other['particle']:
            return False

        # Check that mothers have the same numbers (only relevant info)
        return sorted([mother['number'] for mother in self['mothers']]) == \
               sorted([mother['number'] for mother in other['mothers']])

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

#===============================================================================
# HelasWavefunctionList
#===============================================================================
class HelasWavefunctionList(base_objects.PhysicsObjectList):
    """List of HelasWavefunction objects. This class has the routine
    check_and_fix_fermion_flow, which checks for fermion flow clashes
    among the mothers of an amplitude or wavefunction.
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasWavefunction for the list."""

        return isinstance(obj, HelasWavefunction)

    # Helper functions

    def to_array(self):
        return array.array('i', [w['number'] for w in self])

    def check_and_fix_fermion_flow(self,
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   my_state,
                                   wf_number,
                                   force_flip_flow=False):
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

        # Note that the order of mothers is given by the external
        # number meaning that the order will in general be different
        # for t- and s-channel diagrams, hence giving duplication of
        # wave functions. I don't think there's anything to do about
        # this.
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
                                                wf_number,
                                                force_flip_flow)

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
            # No Majorana fermion in any relevant legs - try again,
            # but simply use the first relevant leg
            force_flip_flow = True
            wf_number = self.check_and_fix_fermion_flow(\
                                   wavefunctions,
                                   diagram_wavefunctions,
                                   external_wavefunctions,
                                   my_state,
                                   wf_number,
                                   force_flip_flow)

        return wf_number

    def insert_own_mothers(self):
        """Recursively go through a wavefunction list and insert the
        mothers of all wavefunctions, return the result.
        Assumes that all wavefunctions have unique numbers."""

        res = copy.copy(self)
        # Recursively build up res
        for wf in self:
            index = res.index(wf)
            res = res[:index] + wf.get('mothers').insert_own_mothers() \
                  + res[index:]

        # Make sure no wavefunctions occur twice, by removing doublets
        # from the back
        i = len(res) - 1
        while res[:i]:
            if res[i].get('number') in [w.get('number') for w in res[:i]]:
                res.pop(i)
            i = i - 1

        return res

    @staticmethod
    def extract_wavefunctions(mothers):
        """Recursively extract the wavefunctions from mothers of mothers"""

        wavefunctions = copy.copy(mothers)
        for wf in mothers:
            wavefunctions.extend(HelasWavefunctionList.\
                                 extract_wavefunctions(wf.get('mothers')))

        return wavefunctions

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
        self['orders'] = {}
        self['inter_color'] = None
        self['lorentz'] = ''
        self['coupling'] = 'none'
        # The Lorentz and color index used in this amplitude
        self['coupl_key'] = (0, 0)
        # Properties relating to the vertex
        self['number'] = 0
        self['fermionfactor'] = 0
        self['color_indices'] = []
        self['mothers'] = HelasWavefunctionList()

    # Customized constructor
    def __init__(self, *arguments):
        """Allow generating a HelasAmplitude from a Vertex
        """

        if len(arguments) > 1:
            if isinstance(arguments[0], base_objects.Vertex) and \
               isinstance(arguments[1], base_objects.Model):
                super(HelasAmplitude, self).__init__()
                self.set('interaction_id',
                         arguments[0].get('id'), arguments[1])
        elif arguments:
            super(HelasAmplitude, self).__init__(arguments[0])
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
            #Should be a list of integers
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of integers" % str(value)
            for mystr in value:
                if not isinstance(mystr, int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(mystr)

        if name == 'orders':
            #Should be a dict with valid order names ask keys and int as values
            if not isinstance(value, dict):
                raise self.PhysicsObjectError, \
                        "%s is not a valid dict for coupling orders" % \
                                                                    str(value)
            for order in value.keys():
                if not isinstance(order, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(order)
                if not isinstance(value[order], int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(value[order])

        if name == 'inter_color':
            # Should be None or a color string
            if value and not isinstance(value, color.ColorString):
                    raise self.PhysicsObjectError, \
                            "%s is not a valid Color String" % str(value)

        if name == 'lorentz':
            #Should be a string
            if not isinstance(value, str):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid string" % str(value)

        if name == 'coupling':
            #Should be a string
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid coupling string" % \
                                                                str(value)

        if name == 'coupl_key':
            if value and not isinstance(value, tuple):
                raise self.PhysicsObjectError, \
                      "%s is not a valid tuple" % str(value)
            if len(value) != 2:
                raise self.PhysicsObjectError, \
                      "%s is not a valid tuple with 2 elements" % str(value)
            if not isinstance(value[0], int) or not isinstance(value[1], int):
                raise self.PhysicsObjectError, \
                      "%s is not a valid tuple of integer" % str(value)

        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for amplitude number" % \
                        str(value)

        if name == 'fermionfactor':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for fermionfactor" % \
                        str(value)
            if not value in [-1, 0, 1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid fermion factor (-1, 0 or 1)" % \
                        str(value)

        if name == 'color_indices':
            #Should be a list of integers
            if not isinstance(value, list):
                raise self.PhysicsObjectError, \
                        "%s is not a valid list of integers" % str(value)
            for mystr in value:
                if not isinstance(mystr, int):
                    raise self.PhysicsObjectError, \
                        "%s is not a valid integer" % str(mystr)

        if name == 'mothers':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                      "%s is not a valid list of mothers for amplitude" % \
                      str(value)

        return True

    # Enhanced get function
    def get(self, name):
        """Get the value of the property name."""

        if name == 'fermionfactor' and not self[name]:
            self.calculate_fermionfactor()

        return super(HelasAmplitude, self).get(name)

    # Enhanced set function, where we can append a model

    def set(self, *arguments):
        """When setting interaction_id, if model is given (in tuple),
        set all other interaction properties. When setting pdg_code,
        if model is given, set all other particle properties."""

        assert len(arguments) > 1, "Too few arguments for set"

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
                    self.set('orders', inter.get('orders'))
                    # Note that the following values might change, if
                    # the relevant color/lorentz/coupling is not index 0
                    if inter.get('color'):
                        self.set('inter_color', inter.get('color')[0])
                    if inter.get('lorentz'):
                        self.set('lorentz', inter.get('lorentz')[0])
                    if inter.get('couplings'):
                        self.set('coupling', inter.get('couplings').values()[0])
                return True
            else:
                raise self.PhysicsObjectError, \
                      "%s not allowed name for 3-argument set", name
        else:
            return super(HelasAmplitude, self).set(name, value)

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['interaction_id', 'pdg_codes', 'orders', 'inter_color', 
                'lorentz', 'coupling', 'coupl_key', 'number', 'color_indices',
                'fermionfactor', 'mothers']


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
                                   "Nostate",
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

        return (tuple(res), self.get('lorentz'))

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
            raise self.PhysicsObjectError, \
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
            raise self.PhysicsObjectError, \
                  "Error: %d incoming fermions != %d outgoing fermions" % \
                  (len(in_fermions), len(out_fermions))

        self['fermionfactor'] = self.sign_flips_to_order(fermion_number_list)

    def sign_flips_to_order(self, fermions):
        """Gives the sign corresponding to the number of flips needed
        to place the fermion numbers in order"""

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

    def get_base_diagram(self, wf_dict = {}, vx_list = [], optimization = 1):
        """Return the base_objects.Diagram which corresponds to this
        amplitude, using a recursive method for the wavefunctions."""

        vertices = base_objects.VertexList()

        # Add vertices for all mothers
        for mother in self.get('mothers'):
            vertices.extend(mother.get_base_vertices(wf_dict, vx_list,
                                                     optimization))
        # Generate last vertex
        legs = base_objects.LegList()
        for mother in self.get('mothers'):
            try:
                leg = wf_dict[mother.get('number')]
            except KeyError:
                leg = base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'from_group': mother.get('onshell')
                    })
                if optimization != 0:
                    wf_dict[mother.get('number')] = leg
            legs.append(leg)

        vertices.append(base_objects.Vertex({
            'id': self.get('interaction_id'),
            'legs': legs}))

        return base_objects.Diagram({'vertices': vertices})

    def get_s_and_t_channels(self, ninitial):
        """Returns two lists of vertices corresponding to the s- and
        t-channels of this amplitude/diagram, ordered from the outermost
        s-channel and in/down towards the highest number initial state
        leg."""

        schannels = base_objects.VertexList()
        tchannels = base_objects.VertexList()

        # Add vertices for all s-channel mothers
        final_mothers = filter(lambda wf: wf.get('number_external') > ninitial,
                               self.get('mothers'))

        for mother in final_mothers:
            schannels.extend(mother.get_base_vertices(optimization = 0))

        # Extract initial state mothers
        init_mothers = filter(lambda wf: wf.get('number_external') <= ninitial,
                              self.get('mothers'))

        if len(init_mothers) > 2:
            raise self.PhysicsObjectError, \
                  "get_s_and_t_channels can only handle up to 2 initial states"

        if len(init_mothers) == 1:
            # This is an s-channel leg, or the first vertex in a decay
            # process. Add vertex and start stepping down towards
            # initial state

            # Create vertex
            legs = base_objects.LegList()
            for mother in final_mothers + init_mothers:
                legs.append(base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'from_group': False
                    }))

            # Renumber resulting leg according to minimum leg number
            legs[-1].set('number', min([l.get('number') for l in legs[:-1]]))
            # Change direction of init_mother
            legs[-1].set('id', init_mothers[0].get_anti_pdg_code())

            # Add vertex to s-channels
            schannels.append(base_objects.Vertex({
                'id': self.get('interaction_id'),
                'legs': legs}))

            # Add s- and t-channels from further down
            mother_s, tchannels = init_mothers[0].\
                                  get_s_and_t_channels(ninitial, legs[-1])

            schannels.extend(mother_s)
        else:
            # This is a t-channel leg. Start with the leg going
            # towards external particle 1, and then do external
            # particle 2
            init_mothers1 = filter(lambda wf: wf.get('number_external') == 1,
                                   init_mothers)[0]
            init_mothers2 = filter(lambda wf: wf.get('number_external') == 2,
                                   init_mothers)[0]

            # Create vertex
            legs = base_objects.LegList()
            for mother in [init_mothers1] + final_mothers + [init_mothers2]:
                legs.append(base_objects.Leg({
                    'id': mother.get_pdg_code(),
                    'number': mother.get('number_external'),
                    'state': mother.get('leg_state'),
                    'from_group': False
                    }))
            # Sort the legs
            legs = base_objects.LegList(\
                   sorted(legs[:-1], lambda l1, l2: l1.get('number') - \
                          l2.get('number')) + legs[-1:])
            # Renumber resulting leg according to minimum leg number
            legs[-1].set('number', min([l.get('number') for l in legs[:-1]]))

            vertex = base_objects.Vertex({
                'id': self.get('interaction_id'),
                'legs': legs})

            # Add s- and t-channels going down towards leg 1
            mother_s, tchannels = \
                      init_mothers1.get_s_and_t_channels(ninitial, legs[0])

            schannels.extend(mother_s)

            # Add vertex to t-channels
            tchannels.append(vertex)

            # Add s- and t-channels going down towards leg 2
            mother_s, mother_t = \
                      init_mothers2.get_s_and_t_channels(ninitial, legs[-1])
            schannels.extend(mother_s)
            tchannels.extend(mother_t)

        # Finally go through all vertices, sort the legs and replace
        # leg number with propagator number -1, -2, ...
        number_dict = {}
        nprop = 0
        for vertex in schannels + tchannels:
            # Sort the legs
            legs = vertex.get('legs')[:-1]
            if vertex in schannels:
                legs.sort(lambda l1, l2: l2.get('number') - \
                          l1.get('number'))
            else:
                legs.sort(lambda l1, l2: l1.get('number') - \
                          l2.get('number'))
            for leg in legs:
                try:
                    leg.set('number', number_dict[leg.get('number')])
                except KeyError:
                    pass
            nprop = nprop - 1
            last_leg = vertex.get('legs')[-1]
            number_dict[last_leg.get('number')] = nprop
            last_leg.set('number', nprop)
            legs.append(last_leg)
            vertex.set('legs', base_objects.LegList(legs))

        return schannels, tchannels

    def get_color_indices(self):
        """Get the color indices corresponding to
        this amplitude and its mothers, using a recursive function."""

        if not self.get('mothers'):
            return []

        color_indices = []

        # Add color indices for all mothers
        for mother in self.get('mothers'):
            # This is where recursion happens
            color_indices.extend(mother.get_color_indices())

        # Add this amp's color index
        if self.get('interaction_id'):
            color_indices.append(self.get('coupl_key')[0])

        return color_indices

    # Comparison between different amplitudes, to allow check for
    # identical processes. Note that we are then not interested in
    # interaction id, but in all other properties.
    def __eq__(self, other):
        """Comparison between different amplitudes, to allow check for
        identical processes.
        """

        if not isinstance(other, HelasAmplitude):
            return False

        # Check relevant directly defined properties
        if self['lorentz'] != other['lorentz'] or \
           self['coupling'] != other['coupling'] or \
           self['number'] != other['number']:
            return False

        # Check that mothers have the same numbers (only relevant info)
        return sorted([mother['number'] for mother in self['mothers']]) == \
               sorted([mother['number'] for mother in other['mothers']])

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
    """HelasDiagram: list of HelasWavefunctions and a HelasAmplitude,
    plus the fermion factor associated with the corresponding diagram.
    """

    def default_setup(self):
        """Default values for all properties"""

        self['wavefunctions'] = HelasWavefunctionList()
        # One diagram can have several amplitudes, if there are
        # different Lorentz or color structures associated with this
        # diagram
        self['amplitudes'] = HelasAmplitudeList()
        self['number'] = 0

    def filter(self, name, value):
        """Filter for valid diagram property values."""

        if name == 'wavefunctions':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasWavefunctionList object" % \
                        str(value)
        if name == 'amplitudes':
            if not isinstance(value, HelasAmplitudeList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasAmplitudeList object" % \
                        str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['wavefunctions', 'amplitudes']

    def calculate_orders(self):
        """Calculate the actual coupling orders of this diagram"""

        wavefunctions = HelasWavefunctionList.extract_wavefunctions(\
            self.get('amplitudes')[0].get('mothers'))

        coupling_orders = {}
        for wf in wavefunctions + [self.get('amplitudes')[0]]:
            if not wf.get('orders'): continue
            for order in wf.get('orders').keys():
                try:
                    coupling_orders[order] += wf.get('orders')[order]
                except:
                    coupling_orders[order] = wf.get('orders')[order]

        return coupling_orders


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
    """HelasMatrixElement: list of processes with identical Helas
    calls, and the list of HelasDiagrams associated with the processes.

    If initiated with an Amplitude, HelasMatrixElement calls
    generate_helas_diagrams, which goes through the diagrams of the
    Amplitude and generates the corresponding Helas calls, taking into
    account possible fermion flow clashes due to Majorana
    particles. The optional optimization argument determines whether
    optimization is used (optimization = 1, default), for maximum
    recycling of wavefunctions, or no optimization (optimization = 0)
    when each diagram is written independently of all previous
    diagrams (this is useful for running with restricted memory,
    e.g. on a GPU). For processes with many diagrams, the total number
    or wavefunctions after optimization is ~15% of the number of
    amplitudes (diagrams).

    By default, it will also generate the color information (color
    basis and color matrix) corresponding to the Amplitude.
    """

    def default_setup(self):
        """Default values for all properties"""

        self['processes'] = base_objects.ProcessList()
        self['diagrams'] = HelasDiagramList()
        self['identical_particle_factor'] = 0
        self['color_basis'] = color_amp.ColorBasis()
        self['color_matrix'] = color_amp.ColorMatrix(color_amp.ColorBasis())
        # base_amplitude is the Amplitude to be used in color
        # generation, drawing etc. For decay chain processes, this is
        # the Amplitude which corresponds to the combined process.
        self['base_amplitude'] = None

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
        if name == 'identical_particle_factor':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid int object" % str(value)
        if name == 'color_basis':
            if not isinstance(value, color_amp.ColorBasis):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ColorBasis object" % str(value)
        if name == 'color_matrix':
            if not isinstance(value, color_amp.ColorMatrix):
                raise self.PhysicsObjectError, \
                        "%s is not a valid ColorMatrix object" % str(value)
        if name == 'base_amplitude':
            if value != None and not \
                   isinstance(value, diagram_generation.Amplitude):
                raise self.PhysicsObjectError, \
                        "%s is not a valid Amplitude object" % str(value)
        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['processes', 'identical_particle_factor',
                'diagrams', 'color_basis', 'color_matrix',
                'base_amplitude']

    # Enhanced get function
    def get(self, name):
        """Get the value of the property name."""

        if name == 'base_amplitude' and not self[name]:
            self['base_amplitude'] = self.get_base_amplitude()

        return super(HelasMatrixElement, self).get(name)

    # Customized constructor
    def __init__(self, amplitude=None, optimization=1,
                 decay_ids=[], gen_color=True):
        """Constructor for the HelasMatrixElement. In particular allows
        generating a HelasMatrixElement from an Amplitude, with
        automatic generation of the necessary wavefunctions
        """

        if amplitude != None:
            if isinstance(amplitude, diagram_generation.Amplitude):
                super(HelasMatrixElement, self).__init__()
                self.get('processes').append(amplitude.get('process'))
                self.generate_helas_diagrams(amplitude, optimization, decay_ids)
                self.calculate_fermionfactors()
                self.calculate_identical_particle_factors()
                if gen_color:
                    self.get('color_basis').build(self.get('base_amplitude'))
                    self.set('color_matrix',
                             color_amp.ColorMatrix(self.get('color_basis')))
            else:
                # In this case, try to use amplitude as a dictionary
                super(HelasMatrixElement, self).__init__(amplitude)
        else:
            super(HelasMatrixElement, self).__init__()

    # Comparison between different amplitudes, to allow check for
    # identical processes. Note that we are then not interested in
    # interaction id, but in all other properties.
    def __eq__(self, other):
        """Comparison between different matrix elements, to allow check for
        identical processes.
        """

        if not isinstance(other, HelasMatrixElement):
            return False

        # If no processes, this is an empty matrix element
        if not self['processes'] and not other['processes']:
            return True

        # Should only check if diagrams and process id are identical
        # Except in case of decay processes: then also initial state
        # must be the same
        if self['processes'] and not other['processes'] or \
               self['processes'] and \
               self['processes'][0]['id'] != other['processes'][0]['id'] or \
               self['processes'][0]['is_decay_chain'] or \
               other['processes'][0]['is_decay_chain'] or \
               self['identical_particle_factor'] != \
                           other['identical_particle_factor'] or \
               self['diagrams'] != other['diagrams']:
            return False

        return True

    def __ne__(self, other):
        """Overloading the nonequality operator, to make comparison easy"""
        return not self.__eq__(other)

    def generate_helas_diagrams(self, amplitude, optimization=1,
                                decay_ids=[]):
        """Starting from a list of Diagrams from the diagram
        generation, generate the corresponding HelasDiagrams, i.e.,
        the wave functions and amplitudes. Choose between default
        optimization (= 1, maximum recycling of wavefunctions) or no
        optimization (= 0, no recycling of wavefunctions, useful for
        GPU calculations with very restricted memory).

        Note that we need special treatment for decay chains, since
        the end product then is a wavefunction, not an amplitude.
        """
        
        assert  isinstance(amplitude, diagram_generation.Amplitude), \
                    "Missing or erraneous arguments for generate_helas_diagrams"
        assert isinstance(optimization, int), \
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
                                        HelasWavefunction(leg, 0, model,
                                                          decay_ids)) \
                                       for leg in process.get('legs')])

        # Initially, have one wavefunction for each external leg.
        wf_number = len(process.get('legs'))

        # For initial state bosons, need to flip part-antipart
        # since all bosons should be treated as outgoing
        for key in external_wavefunctions.keys():
            wf = external_wavefunctions[key]
            if wf.is_boson() and wf.get('state') == 'initial' and \
               not wf.get('self_antipart'):
                wf.set('is_part', not wf.get('is_part'))

        # For initial state particles, need to flip PDG code (if has
        # antipart)
        for key in external_wavefunctions.keys():
            wf = external_wavefunctions[key]
            if wf.get('leg_state') == False and \
               not wf.get('self_antipart'):
                wf.flip_part_antipart()

        # Now go through the diagrams, looking for undefined wavefunctions

        helas_diagrams = HelasDiagramList()

        # Keep track of amplitude number and diagram number
        amplitude_number = 0
        diagram_number = 0

        for diagram in diagram_list:

            # List of dictionaries from leg number to wave function,
            # keeps track of the present position in the tree.
            # Need one dictionary per coupling multiplicity (diagram)
            number_to_wavefunctions = [{}]

            # Need to keep track of the color structures for each amplitude
            color_lists = [[]]

            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = HelasWavefunctionList()

            vertices = copy.copy(diagram.get('vertices'))

            # Single out last vertex, since this will give amplitude
            lastvx = vertices.pop()

            # Check if last vertex is identity vertex
            if not process.get('is_decay_chain') and lastvx.get('id') == 0:
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

            # Go through all vertices except the last and create
            # wavefunctions
            for vertex in vertices:

                # In case there are diagrams with multiple Lorentz/color 
                # structures, we need to keep track of the wavefunctions
                # for each such structure separately, and generate
                # one HelasDiagram for each structure.
                # We use the array number_to_wavefunctions to keep
                # track of this, with one dictionary per chain of
                # wavefunctions
                # Note that all wavefunctions relating to this diagram
                # will be written out before the first amplitude is written.
                new_number_to_wavefunctions = []
                new_color_lists = []
                for number_wf_dict, color_list in zip(number_to_wavefunctions,
                                                     color_lists):
                    legs = copy.copy(vertex.get('legs'))
                    last_leg = legs.pop()
                    # Generate list of mothers from legs
                    mothers = self.getmothers(legs, number_wf_dict,
                                              external_wavefunctions,
                                              wavefunctions,
                                              diagram_wavefunctions)
                    inter = model.get('interaction_dict')[vertex.get('id')]

                    # Now generate new wavefunction for the last leg

                    # Need one amplitude for each Lorentz/color structure,
                    # i.e. for each coupling
                    for coupl_key in sorted(inter.get('couplings').keys()):
                        wf = HelasWavefunction(last_leg, vertex.get('id'), model)
                        wf.set('coupling', inter.get('couplings')[coupl_key])
                        # Special feature: For HVS vertices with the two
                        # scalars different, we need extra minus sign in front
                        # of coupling for one of the two scalars since the HVS
                        # is asymmetric in the two scalars
                        if wf.get('spin') == 1:
                            wf.set_scalar_coupling_sign(model)
                        if inter.get('color'):
                            wf.set('inter_color', inter.get('color')[coupl_key[0]])
                        wf.set('lorentz', inter.get('lorentz')[coupl_key[1]])
                        wf.set('coupl_key', coupl_key)
                        wf.set('mothers', mothers)
                        # Need to set incoming/outgoing and
                        # particle/antiparticle according to the fermion flow
                        # of mothers
                        wf.set_state_and_particle(model)
                        # Need to check for clashing fermion flow due to
                        # Majorana fermions, and modify if necessary
                        # Also need to keep track of the wavefunction number.
                        wf, wf_number = wf.check_and_fix_fermion_flow(\
                                                   wavefunctions,
                                                   diagram_wavefunctions,
                                                   external_wavefunctions,
                                                   wf_number)

                        # Create new copy of number_wf_dict
                        new_number_wf_dict = copy.copy(number_wf_dict)

                        # Store wavefunction
                        try:
                            wf = diagram_wavefunctions[\
                                    diagram_wavefunctions.index(wf)]
                        except ValueError:
                            # Update wf number
                            wf_number = wf_number + 1
                            wf.set('number', wf_number)
                            try:
                                # Use wf_mother_arrays to locate existing
                                # wavefunction
                                wf = wavefunctions[wf_mother_arrays.index(\
                                wf.to_array())]
                                # Since we reuse the old wavefunction, reset
                                # wf_number
                                wf_number = wf_number - 1
                            except ValueError:
                                diagram_wavefunctions.append(wf)

                        new_number_wf_dict[last_leg.get('number')] = wf

                        # Store the new copy of number_wf_dict
                        new_number_to_wavefunctions.append(\
                                                        new_number_wf_dict)
                        # Add color index and store new copy of color_lists
                        new_color_list = copy.copy(color_list)
                        new_color_list.append(coupl_key[0])
                        new_color_lists.append(new_color_list)

                number_to_wavefunctions = new_number_to_wavefunctions
                color_lists = new_color_lists

            # Generate all amplitudes corresponding to the different
            # copies of this diagram
            helas_diagram = HelasDiagram()
            diagram_number = diagram_number + 1
            helas_diagram.set('number', diagram_number)
            for number_wf_dict, color_list in zip(number_to_wavefunctions,
                                                  color_lists):
                # Find mothers for the amplitude
                legs = lastvx.get('legs')
                mothers = self.getmothers(legs, number_wf_dict,
                                          external_wavefunctions,
                                          wavefunctions,
                                          diagram_wavefunctions)
                # Need to check for clashing fermion flow due to
                # Majorana fermions, and modify if necessary
                wf_number = mothers.check_and_fix_fermion_flow(wavefunctions,
                                              diagram_wavefunctions,
                                              external_wavefunctions,
                                              "Nostate",
                                              wf_number)

                # Sort the wavefunctions according to number
                diagram_wavefunctions.sort(lambda wf1, wf2: \
                              wf1.get('number') - wf2.get('number'))

                # Now generate HelasAmplitudes from the last vertex.
                if lastvx.get('id'):
                    inter = model.get('interaction_dict')[lastvx.get('id')]
                    keys = sorted(inter.get('couplings').keys())
                else:
                    # Special case for decay chain - amplitude is just a
                    # placeholder for replaced wavefunction
                    inter = None
                    keys = [(0, 0)]
                for i, coupl_key in enumerate(keys):
                    amp = HelasAmplitude(lastvx, model)
                    if inter:
                        amp.set('coupling', inter.get('couplings')[coupl_key])
                        amp.set('lorentz', inter.get('lorentz')[\
                                coupl_key[1]])
                        if inter.get('color'):
                            amp.set('inter_color', inter.get('color')[\
                                coupl_key[0]])
                        amp.set('coupl_key', coupl_key)
                    amp.set('mothers', mothers)
                    amplitude_number = amplitude_number + 1
                    amp.set('number', amplitude_number)
                    # Add the list with color indices to the amplitude
                    new_color_list = copy.copy(color_list)
                    if inter:
                        new_color_list.append(coupl_key[0])
                        
                    amp.set('color_indices', new_color_list)

                    # Generate HelasDiagram
                    helas_diagram.get('amplitudes').append(amp)
                    if diagram_wavefunctions and not \
                                       helas_diagram.get('wavefunctions'):
                        helas_diagram.set('wavefunctions',
                                          diagram_wavefunctions)

                if optimization:
                    wavefunctions.extend(diagram_wavefunctions)
                    wf_mother_arrays.extend([wf.to_array() for wf \
                                             in diagram_wavefunctions])
                else:
                    wf_number = len(process.get('legs'))
            # Append this diagram in the diagram list
            helas_diagrams.append(helas_diagram)


        self.set('diagrams', helas_diagrams)
        # Sort all mothers according to the order wanted in Helas calls

        for wf in self.get_all_wavefunctions():
            wf.set('mothers', HelasMatrixElement.sorted_mothers(wf))
        for amp in self.get_all_amplitudes():
            amp.set('mothers', HelasMatrixElement.sorted_mothers(amp))
            amp.set('color_indices', amp.get_color_indices())

    def insert_decay_chains(self, decay_dict):
        """Iteratively insert decay chains decays into this matrix
        element.        
        * decay_dict: a dictionary from external leg number
          to decay matrix element.
        """

        # We need to keep track of how the
        # wavefunction numbers change
        replace_dict = {}
        for number in decay_dict.keys():
            # Find all wavefunctions corresponding to this external
            # leg number
            replace_dict[number] = [wf for wf in \
                          filter(lambda wf: not wf.get('mothers') and \
                                 wf.get('number_external') == number,
                                 self.get_all_wavefunctions())]

        # Keep track of wavefunction and amplitude numbers, to ensure
        # unique numbers for all new wfs and amps during manipulations
        numbers = [self.get_all_wavefunctions()[-1].get('number'),
                   self.get_all_amplitudes()[-1].get('number')]

        # Check if there are any Majorana particles present in any diagrams
        got_majoranas = False
        for wf in self.get_all_wavefunctions() + \
            sum([d.get_all_wavefunctions() for d in \
                 decay_dict.values()], []):
            if wf.get('self_antipart') and wf.is_fermion():
                got_majoranas = True

        # Now insert decays for all legs that have decays
        for number in decay_dict.keys():

                self.insert_decay(replace_dict[number],
                                  decay_dict[number],
                                  numbers,
                                  got_majoranas)

        # Remove all diagrams that surpass overall coupling orders
        overall_orders = self.get('processes')[0].get('overall_orders')
        if overall_orders:
            ndiag = len(self.get('diagrams'))
            idiag = 0
            while self.get('diagrams')[idiag:]:
                diagram = self.get('diagrams')[idiag]
                orders = diagram.calculate_orders()
                remove_diagram = False
                for order in orders.keys():
                    try:
                        if orders[order] > \
                               overall_orders[order]:
                            remove_diagram = True
                    except KeyError:
                        pass
                if remove_diagram:
                    self.get('diagrams').pop(idiag)
                else:
                    idiag += 1

            if len(self.get('diagrams')) < ndiag:
                # We have removed some diagrams - need to go through
                # diagrams, renumber them and set new wavefunctions
                wf_numbers = []
                ndiagrams = 0
                for diagram in self.get('diagrams'):
                    ndiagrams += 1
                    diagram.set('number', ndiagrams)
                    # Extract all wavefunctions contributing to this amplitude
                    diagram_wfs = HelasWavefunctionList()
                    for amplitude in diagram.get('amplitudes'):
                        wavefunctions = \
                          sorted(HelasWavefunctionList.\
                               extract_wavefunctions(amplitude.get('mothers')),
                                 lambda wf1, wf2: wf1.get('number') - \
                                                  wf2.get('number'))
                        for wf in wavefunctions:
                            # Check if wavefunction already used, otherwise add
                            if wf.get('number') not in wf_numbers and \
                                   wf not in diagram_wfs:
                                diagram_wfs.append(wf)
                                wf_numbers.append(wf.get('number'))
                    diagram.set('wavefunctions', diagram_wfs)

        # Final cleaning out duplicate wavefunctions - needed only if
        # we have multiple fermion flows, i.e., either multiple replaced
        # wavefunctions or  majorana fermions and multiple diagrams
        flows = reduce(lambda i1, i2: i1 * i2,
                       [len(replace_dict[i]) for i in decay_dict.keys()], 1)
        diagrams = reduce(lambda i1, i2: i1 * i2,
                               [len(decay_dict[i].get('diagrams')) for i in \
                                decay_dict.keys()], 1)

        if flows > 1 or (diagrams > 1 and got_majoranas):

            # Clean out any doublet wavefunctions

            earlier_wfs = []

            earlier_wf_arrays = []

            mothers = self.get_all_wavefunctions() + self.get_all_amplitudes()
            mother_arrays = [w['mothers'].to_array() \
                             for w in mothers]

            for diagram in self.get('diagrams'):

                if diagram.get('number') > 1:
                    earlier_wfs.extend(self.get('diagrams')[\
                        diagram.get('number') - 2].get('wavefunctions'))

                i = 0
                diag_wfs = diagram.get('wavefunctions')


                # Remove wavefunctions and replace mothers
                while diag_wfs[i:]:
                    try:
                        new_wf = earlier_wfs[\
                            earlier_wfs.index(diag_wfs[i])]
                        wf = diag_wfs.pop(i)

                        self.update_later_mothers(wf, new_wf, mothers,
                                                  mother_arrays)
                    except ValueError:
                        i = i + 1

        # When we are done with all decays, set wavefunction and
        # amplitude numbers
        for i, wf in enumerate(self.get_all_wavefunctions()):
            wf.set('number', i + 1)
        for i, amp in enumerate(self.get_all_amplitudes()):
            amp.set('number', i + 1)
            # Update fermion factors for all amplitudes
            amp.calculate_fermionfactor()
            # Update color indices
            amp.set('color_indices', amp.get_color_indices())

        # Calculate identical particle factors for
        # this matrix element
        self.identical_decay_chain_factor(decay_dict.values())

    def insert_decay(self, old_wfs, decay, numbers, got_majoranas):
        """Insert a decay chain matrix element into the matrix element.
        * old_wfs: the wavefunctions to be replaced.
          They all correspond to the same external particle, but might
          have different fermion flow directions
        * decay: the matrix element for the decay chain
        * numbers: the present wavefunction and amplitude number,
          to allow for unique numbering
          
        Note that:
        1) All amplitudes and all wavefunctions using the decaying wf
           must be copied as many times as there are amplitudes in the
           decay matrix element
        2) In the presence of Majorana particles, we must make sure
           to flip fermion flow for the decay process if needed.

        The algorithm is the following:
        1) Multiply the diagrams with the number of diagrams Ndiag in
           the decay element
        2) For each diagram in the decay element, work on the diagrams
           which corresponds to it
        3) Flip fermion flow for the decay wavefunctions if needed
        4) Insert all auxiliary wavefunctions into the diagram (i.e., all 
           except the final wavefunctions, which directly replace the
           original final state wavefunctions)
        4) Replace the wavefunctions recursively, so that we always replace
           each old wavefunctions with Namp new ones, where Namp is
           the number of amplitudes in this decay element
           diagram. Do recursion for wavefunctions which have this
           wavefunction as mother. Simultaneously replace any
           amplitudes which have this wavefunction as mother.
        """

        len_decay = len(decay.get('diagrams'))

        number_external = old_wfs[0].get('number_external')

        # Insert the decay process in the process
        for process in self.get('processes'):
            process.get('decay_chains').append(\
                   decay.get('processes')[0])

        # We need one copy of the decay element diagrams for each
        # old_wf to be replaced, since we need different wavefunction
        # numbers for them
        decay_elements = [copy.deepcopy(d) for d in \
                          [ decay.get('diagrams') ] * len(old_wfs)]

        # Need to replace Particle in all wavefunctions to avoid
        # deepcopy
        for decay_element in decay_elements:
            for idiag, diagram in enumerate(decay.get('diagrams')):
                wfs = diagram.get('wavefunctions')
                decay_diag = decay_element[idiag]
                for i, wf in enumerate(decay_diag.get('wavefunctions')):
                    wf.set('particle', wfs[i].get('particle'))
                    wf.set('antiparticle', wfs[i].get('antiparticle'))

        for decay_element in decay_elements:

            # Remove the unwanted initial state wavefunctions from decay
            for decay_diag in decay_element:
                for wf in filter(lambda wf: wf.get('number_external') == 1,
                                 decay_diag.get('wavefunctions')):
                    decay_diag.get('wavefunctions').remove(wf)

            decay_wfs = sum([d.get('wavefunctions') for d in decay_element], [])

            # External wavefunction offset for new wfs
            incr_new = number_external - \
                       decay_wfs[0].get('number_external')

            for wf in decay_wfs:
                # Set number_external for new wavefunctions
                wf.set('number_external', wf.get('number_external') + incr_new)
                # Set unique number for new wavefunctions
                numbers[0] = numbers[0] + 1
                wf.set('number', numbers[0])

            # External wavefunction offset for new wfs
            incr_new = number_external - \
                       decay_wfs[0].get('number_external')
            for wf in decay_wfs:
                wf.set('number_external', wf.get('number_external') + incr_new)


        # External wavefunction offset for old wfs, only the first
        # time this external wavefunction is replaced
        (nex, nin) = decay.get_nexternal_ninitial()
        incr_old = nex - 2 # due to the incoming particle
        wavefunctions = self.get_all_wavefunctions()
        for wf in wavefunctions:
            # Only modify number_external for wavefunctions above old_wf
            if wf.get('number_external') > number_external:
                wf.set('number_external',
                       wf.get('number_external') + incr_old)

        # Multiply the diagrams by Ndiag

        diagrams = HelasDiagramList()
        for diagram in self.get('diagrams'):
            new_diagrams = [copy.copy(diag) for diag in \
                            [ diagram ] * (len_decay - 1)]
            # Update diagram number
            diagram.set('number', (diagram.get('number') - 1) * \
                        len_decay + 1)

            for i, diag in enumerate(new_diagrams):
                # Set diagram number
                diag.set('number', diagram.get('number') + i + 1)
                # Copy over all wavefunctions
                diag.set('wavefunctions',
                         copy.copy(diagram.get('wavefunctions')))
                # Copy over the amplitudes
                amplitudes = HelasAmplitudeList(\
                                [copy.copy(amp) for amp in \
                                 diag.get('amplitudes')])
                # Renumber amplitudes
                for amp in amplitudes:
                    numbers[1] = numbers[1] + 1
                    amp.set('number', numbers[1])
                diag.set('amplitudes', amplitudes)
            # Add old and new diagram to diagrams
            diagrams.append(diagram)
            diagrams.extend(new_diagrams)

        self.set('diagrams', diagrams)

        # Now we work by decay process diagram, parameterized by numdecay
        for numdecay in range(len_decay):

            # Pick out the diagrams which correspond to this decay diagram
            diagrams = [self.get('diagrams')[i] for i in \
                        range(numdecay, len(self.get('diagrams')), len_decay)]

            # Perform replacement for each of the wavefunctions in old_wfs
            for decay_element, old_wf in zip(decay_elements, old_wfs):

                decay_diag = decay_element[numdecay]

                # Find the diagrams which have old_wf
                my_diagrams = filter(lambda diag: (old_wf.get('number') in \
                                            [wf.get('number') for wf in \
                                            diag.get('wavefunctions')]),
                                     diagrams)

                # Ignore possibility for unoptimizated generation for now
                if len(my_diagrams) > 1:
                    raise self.PhysicsObjectError, \
                          "Decay chains not yet prepared for GPU"

                for diagram in my_diagrams:

                    if got_majoranas:
                        # If there are Majorana particles in any of
                        # the matrix elements, we need to check for
                        # fermion flow

                        # Earlier wavefunctions, will be used for fermion flow
                        index = [d.get('number') for d in diagrams].\
                                index(diagram.get('number'))
                        earlier_wavefunctions = \
                                      sum([d.get('wavefunctions') for d in \
                                           diagrams[:index]], [])

                        # Don't want to affect original decay
                        # wavefunctions, so need to deepcopy
                        decay_diag_wfs = copy.deepcopy(\
                                                decay_diag.get('wavefunctions'))
                        # Need to replace Particle in all
                        # wavefunctions to avoid deepcopy
                        for i, wf in enumerate(decay_diag.get('wavefunctions')):
                            decay_diag_wfs[i].set('particle', \
                                                  wf.get('particle'))
                            decay_diag_wfs[i].set('antiparticle', \
                                                  wf.get('antiparticle'))

                        # Complete decay_diag_wfs with the mother wavefunctions
                        # to allow for independent fermion flow flips
                        decay_diag_wfs = decay_diag_wfs.insert_own_mothers()

                        # These are the wavefunctions which directly replace old_wf
                        final_decay_wfs = [amp.get('mothers')[1] for amp in \
                                              decay_diag.get('amplitudes')]

                        # Since we made deepcopy, need to syncronize
                        for i, wf in enumerate(final_decay_wfs):
                            final_decay_wfs[i] = \
                                               decay_diag_wfs[decay_diag_wfs.index(wf)]
                        # Remove final wavefunctions from decay_diag_wfs,
                        # since these will be replaced separately by
                        # replace_wavefunctions
                        for wf in final_decay_wfs:
                            decay_diag_wfs.remove(wf)

                        # Check fermion flow direction
                        if old_wf.is_fermion() and \
                               old_wf.get_with_flow('state') != \
                                     final_decay_wfs[0].get_with_flow('state'):

                            # Not same flow state - need to flip flow of wf

                            for i, wf in enumerate(final_decay_wfs):

                                # We use the function
                                # check_majorana_and_flip_flow, as in the
                                # helas diagram generation.  Since we have
                                # different flow, there is already a Majorana
                                # particle along the fermion line.

                                final_decay_wfs[i], numbers[0] = \
                                                wf.check_majorana_and_flip_flow(\
                                                         True,
                                                         earlier_wavefunctions,
                                                         decay_diag_wfs,
                                                         {},
                                                         numbers[0])

                        # Remove wavefunctions which are already present in
                        # earlier_wavefunctions
                        i = 0
                        earlier_wavefunctions = \
                            sum([d.get('wavefunctions') for d in \
                                 self.get('diagrams')[:diagram.get('number') - 1]], \
                                [])
                        earlier_wf_numbers = [wf.get('number') for wf in \
                                              earlier_wavefunctions]

                        i = 0
                        mother_arrays = [w.get('mothers').to_array() for \
                                         w in final_decay_wfs]
                        while decay_diag_wfs[i:]:
                            wf = decay_diag_wfs[i]
                            try:
                                new_wf = earlier_wavefunctions[\
                                    earlier_wf_numbers.index(wf.get('number'))]
                                # If the wavefunctions are not identical,
                                # then we should keep this wavefunction,
                                # and update its number so it is unique
                                if wf != new_wf:
                                    numbers[0] = numbers[0] + 1
                                    wf.set('number', numbers[0])
                                    continue
                                decay_diag_wfs.pop(i)
                                pres_mother_arrays = [w.get('mothers').to_array() for \
                                                      w in decay_diag_wfs[i:]] + \
                                                      mother_arrays
                                self.update_later_mothers(wf, new_wf,
                                                          decay_diag_wfs[i:] + \
                                                          final_decay_wfs,
                                                          pres_mother_arrays)
                            except ValueError:
                                i = i + 1

                        # Since we made deepcopy, go through mothers and make
                        # sure we are using the ones in earlier_wavefunctions
                        for decay_wf in decay_diag_wfs + final_decay_wfs:
                            mothers = decay_wf.get('mothers')
                            for i, wf in enumerate(mothers):
                                try:
                                    mothers[i] = earlier_wavefunctions[\
                                        earlier_wf_numbers.index(wf.get('number'))]
                                except ValueError:
                                    pass
                    else:
                        # If there are no Majorana particles, the
                        # treatment is much simpler
                        decay_diag_wfs = \
                                       copy.copy(decay_diag.get('wavefunctions'))

                        # These are the wavefunctions which directly
                        # replace old_wf
                        final_decay_wfs = [amp.get('mothers')[1] for amp in \
                                              decay_diag.get('amplitudes')]

                        # Remove final wavefunctions from decay_diag_wfs,
                        # since these will be replaced separately by
                        # replace_wavefunctions
                        for wf in final_decay_wfs:
                            decay_diag_wfs.remove(wf)


                    diagram_wfs = diagram.get('wavefunctions')

                    old_wf_index = [wf.get('number') for wf in \
                                    diagram_wfs].index(old_wf.get('number'))

                    diagram_wfs = diagram_wfs[0:old_wf_index] + \
                                  decay_diag_wfs + diagram_wfs[old_wf_index:]

                    diagram.set('wavefunctions', HelasWavefunctionList(diagram_wfs))

                    # Set the decay flag for final_decay_wfs, to
                    # indicate that these correspond to decayed
                    # particles
                    for wf in final_decay_wfs:
                        wf.set('onshell', True)

                    if len_decay == 1:
                        # Can use simplified treatment, by just modifying old_wf
                        self.replace_single_wavefunction(old_wf,
                                                         final_decay_wfs[0])
                    else:
                        # Multiply wavefunctions and amplitudes and
                        # insert mothers using a recursive function
                        self.replace_wavefunctions(old_wf,
                                                   final_decay_wfs,
                                                   diagrams,
                                                   numbers)

            # Now that we are done with this set of diagrams, we need
            # to clean out duplicate wavefunctions (i.e., remove
            # identical wavefunctions which are already present in
            # earlier diagrams)
            for diagram in diagrams:

                # We can have duplicate wfs only from previous copies of
                # this diagram
                earlier_wfs = sum([d.get('wavefunctions') for d in \
                                   self.get('diagrams')[\
                                     diagram.get('number') - numdecay - 1:\
                                     diagram.get('number') - 1]], [])

                later_wfs = sum([d.get('wavefunctions') for d in \
                                   self.get('diagrams')[\
                                     diagram.get('number'):]], [])

                later_amps = sum([d.get('amplitudes') for d in \
                                   self.get('diagrams')[\
                                     diagram.get('number') - 1:]], [])

                i = 0
                diag_wfs = diagram.get('wavefunctions')

                # Remove wavefunctions and replace mothers, to make
                # sure we only have one copy of each wavefunction
                # number

                mother_arrays = [w.get('mothers').to_array() for \
                                 w in later_wfs + later_amps]

                while diag_wfs[i:]:
                    try:
                        index = [w.get('number') for w in earlier_wfs].\
                                index(diag_wfs[i].get('number'))
                        wf = diag_wfs.pop(i)
                        pres_mother_arrays = [w.get('mothers').to_array() for \
                                              w in diag_wfs[i:]] + \
                                              mother_arrays
                        self.update_later_mothers(wf, earlier_wfs[index],
                                              diag_wfs[i:] + later_wfs + later_amps,
                                              pres_mother_arrays)
                    except ValueError:
                        i = i + 1

    def update_later_mothers(self, wf, new_wf, later_wfs, later_wf_arrays):
        """Update mothers for all later wavefunctions"""

        daughters = filter(lambda tup: wf.get('number') in tup[1],
                              enumerate(later_wf_arrays))

        for (index, mothers) in daughters:
            try:
                # Replace mother
                later_wfs[index].get('mothers')[\
                    mothers.index(wf.get('number'))] = new_wf
            except ValueError:
                pass

    def replace_wavefunctions(self, old_wf, new_wfs,
                              diagrams, numbers):
        """Recursive function to replace old_wf with new_wfs, and
        multiply all wavefunctions or amplitudes that use old_wf

        * old_wf: The wavefunction to be replaced
        * new_wfs: The replacing wavefunction
        * diagrams - the diagrams that are relevant for these new
          wavefunctions.
        * numbers: the present wavefunction and amplitude number,
          to allow for unique numbering
        """

        # Pick out the diagrams which have the old_wf
        my_diagrams = filter(lambda diag: old_wf.get('number') in \
                         [wf.get('number') for wf in diag.get('wavefunctions')],
                         diagrams)

        # Replace old_wf with new_wfs in the diagrams
        for diagram in my_diagrams:

            diagram_wfs = diagram.get('wavefunctions')

            old_wf_index = [wf.get('number') for wf in \
                            diagram.get('wavefunctions')].index(old_wf.get('number'))
            diagram_wfs = diagram_wfs[:old_wf_index] + \
                          new_wfs + diagram_wfs[old_wf_index + 1:]

            diagram.set('wavefunctions', HelasWavefunctionList(diagram_wfs))

        # Now need to take care of amplitudes and wavefunctions which
        # are daughters of old_wf (only among the relevant diagrams)

        # Pick out diagrams with amplitudes which are daughters of old_wf
        amp_diagrams = filter(lambda diag: old_wf.get('number') in \
                          sum([[wf.get('number') for wf in amp.get('mothers')] \
                               for amp in diag.get('amplitudes')], []),
                              diagrams)

        for diagram in amp_diagrams:

            # Amplitudes in this diagram that are daughters of old_wf
            daughter_amps = filter(lambda amp: old_wf.get('number') in \
                                [wf.get('number') for wf in amp.get('mothers')],
                                diagram.get('amplitudes'))

            new_amplitudes = copy.copy(diagram.get('amplitudes'))

            # Loop over daughter_amps, to multiply each amp by the
            # number of replacement wavefunctions and substitute mothers
            for old_amp in daughter_amps:
                # Create copies of this amp
                new_amps = [copy.copy(amp) for amp in \
                            [ old_amp ] * len(new_wfs)]
                # Replace the old mother with the new ones
                for i, (new_amp, new_wf) in enumerate(zip(new_amps, new_wfs)):
                    mothers = copy.copy(new_amp.get('mothers'))
                    old_wf_index = [wf.get('number') for wf in mothers].index(\
                         old_wf.get('number'))
                    # Update mother
                    mothers[old_wf_index] = new_wf
                    new_amp.set('mothers', mothers)
                    # Update amp numbers for replaced amp
                    numbers[1] = numbers[1] + 1
                    new_amp.set('number', numbers[1])

                # Insert the new amplitudes in diagram amplitudes
                index = [a.get('number') for a in new_amplitudes].\
                                   index(old_amp.get('number'))
                new_amplitudes = new_amplitudes[:index] + \
                                 new_amps + new_amplitudes[index + 1:]

            # Replace diagram amplitudes with the new ones
            diagram.set('amplitudes', HelasAmplitudeList(new_amplitudes))

        # Find wavefunctions that are daughters of old_wf
        daughter_wfs = filter(lambda wf: old_wf.get('number') in \
                              [wf1.get('number') for wf1 in wf.get('mothers')],
                              sum([diag.get('wavefunctions') for diag in \
                                   diagrams], []))

        # Loop over daughter_wfs, multiply them and replace mothers
        for daughter_wf in daughter_wfs:

            # Pick out the diagrams where daughter_wf occurs
            wf_diagrams = filter(lambda diag: daughter_wf.get('number') in \
                                 [wf.get('number') for wf in \
                                  diag.get('wavefunctions')],
                                 diagrams)

            if len(wf_diagrams) > 1:
                raise self.PhysicsObjectError, \
                      "Decay chains not yet prepared for GPU"

            for diagram in wf_diagrams:

                # Now create new wfs with updated mothers
                replace_daughters = [ copy.copy(wf) for wf in \
                                      [daughter_wf] * len(new_wfs) ]

                index = [wf.get('number') for wf in \
                         daughter_wf.get('mothers')].index(old_wf.get('number'))

                # Replace the old mother with the new ones, update wf numbers
                for i, (new_daughter, new_wf) in \
                        enumerate(zip(replace_daughters, new_wfs)):
                    mothers = copy.copy(new_daughter.get('mothers'))
                    mothers[index] = new_wf
                    new_daughter.set('mothers', mothers)
                    numbers[0] = numbers[0] + 1
                    new_daughter.set('number', numbers[0])

                # This is where recursion happens.  We need to replace
                # the daughter wavefunction, and fix amplitudes and
                # wavefunctions which have it as mothers.

                self.replace_wavefunctions(daughter_wf,
                                           replace_daughters,
                                           diagrams,
                                           numbers)

    def replace_single_wavefunction(self, old_wf, new_wf):
        """Insert decay chain by simply modifying wavefunction. This
        is possible only if there is only one diagram in the decay."""

        for key in old_wf.keys():
            old_wf.set(key, new_wf.get(key))

    def identical_decay_chain_factor(self, decay_chains):
        """Calculate the denominator factor from identical decay chains"""

        final_legs = [leg.get('id') for leg in \
                      filter(lambda leg: leg.get('state') == True, \
                              self.get('processes')[0].get('legs'))]

        # Leg ids for legs being replaced by decay chains
        decay_ids = [decay.get('legs')[0].get('id') for decay in \
                     self.get('processes')[0].get('decay_chains')]

        # Find all leg ids which are not being replaced by decay chains
        non_decay_legs = filter(lambda id: id not in decay_ids,
                                final_legs)

        # Identical particle factor for legs not being decayed
        identical_indices = {}
        for id in non_decay_legs:
            if id in identical_indices:
                identical_indices[id] = \
                                    identical_indices[id] + 1
            else:
                identical_indices[id] = 1
        non_chain_factor = reduce(lambda x, y: x * y,
                                  [ math.factorial(val) for val in \
                                    identical_indices.values() ], 1)

        # Identical particle factor for decay chains
        # Go through chains to find identical ones
        chains = copy.copy(decay_chains)
        iden_chains_factor = 1
        while chains:
            ident_copies = 1
            first_chain = chains.pop(0)
            i = 0
            while i < len(chains):
                chain = chains[i]
                if HelasMatrixElement.check_equal_decay_processes(\
                                                 first_chain, chain):
                    ident_copies = ident_copies + 1
                    chains.pop(i)
                else:
                    i = i + 1
            iden_chains_factor = iden_chains_factor * \
                                 math.factorial(ident_copies)

        self['identical_particle_factor'] = non_chain_factor * \
                                    iden_chains_factor * \
                                    reduce(lambda x1, x2: x1 * x2,
                                    [me.get('identical_particle_factor') \
                                     for me in decay_chains], 1)

    def calculate_fermionfactors(self):
        """Generate the fermion factors for all diagrams in the matrix element
        """

        for diagram in self.get('diagrams'):
            for amplitude in diagram.get('amplitudes'):
                amplitude.get('fermionfactor')

    def calculate_identical_particle_factors(self):
        """Calculate the denominator factor for identical final state particles
        """

        final_legs = filter(lambda leg: leg.get('state') == True, \
                              self.get('processes')[0].get('legs'))

        identical_indices = {}
        for leg in final_legs:
            if leg.get('id') in identical_indices:
                identical_indices[leg.get('id')] = \
                                    identical_indices[leg.get('id')] + 1
            else:
                identical_indices[leg.get('id')] = 1
        self["identical_particle_factor"] = reduce(lambda x, y: x * y,
                                          [ math.factorial(val) for val in \
                                            identical_indices.values() ])

    def get_base_amplitude(self):
        """Generate a diagram_generation.Amplitude from a
        HelasMatrixElement. This is used to generate both color
        amplitudes and diagram drawing."""

        # Need to take care of diagram numbering for decay chains
        # before this can be used for those!

        optimization = 1
        if len(filter(lambda wf: wf.get('number') == 1,
                      self.get_all_wavefunctions())) > 1:
            optimization = 0

        model = self.get('processes')[0].get('model')

        wf_dict = {}
        vx_list = []
        diagrams = base_objects.DiagramList()
        for diag in self.get('diagrams'):
            diagrams.append(diag.get('amplitudes')[0].get_base_diagram(\
                wf_dict, vx_list, optimization))

        for diag in diagrams:
            diag.calculate_orders(self.get('processes')[0].get('model'))
            
        return diagram_generation.Amplitude({\
            'process': self.get('processes')[0],
            'diagrams': diagrams})

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
                if not wf in wavefunctions and not wf in diagram_wavefunctions:
                    diagram_wavefunctions.append(wf)
            mothers.append(wf)

        return mothers

    def get_number_of_wavefunctions(self):
        """Gives the total number of wavefunctions for this ME"""

        return sum([ len(d.get('wavefunctions')) for d in \
                       self.get('diagrams')])

    def get_all_wavefunctions(self):
        """Gives a list of all wavefunctions for this ME"""

        return sum([d.get('wavefunctions') for d in \
                       self.get('diagrams')], [])

    def get_all_amplitudes(self):
        """Gives a list of all amplitudes for this ME"""

        return sum([d.get('amplitudes') for d in \
                       self.get('diagrams')], [])

    def get_external_wavefunctions(self):
        """Gives the external wavefunctions for this ME"""

        external_wfs = filter(lambda wf: not wf.get('mothers'),
                              self.get('diagrams')[0].get('wavefunctions'))

        external_wfs.sort(lambda w1, w2: w1.get('number_external') - \
             w1.get('number_external'))

        i = 1
        while i < len(external_wfs):
            if external_wfs[i].get('number_external') == \
               external_wfs[i - 1].get('number_external'):
                external_wfs.pop(i)
            else:
                i = i + 1
        return external_wfs

    def get_number_of_amplitudes(self):
        """Gives the total number of amplitudes for this ME"""

        return sum([ len(d.get('amplitudes')) for d in \
                       self.get('diagrams')])

    def get_nexternal_ninitial(self):
        """Gives (number or external particles, number of
        incoming particles)"""

        external_wfs = filter(lambda wf: wf.get('leg_state') != \
                              'intermediate',
                              self.get_all_wavefunctions())

        return (len(set([wf.get('number_external') for wf in \
                         external_wfs])),
                len(set([wf.get('number_external') for wf in \
                         filter(lambda wf: wf.get('leg_state') == False,
                                external_wfs)])))

    def get_helicity_combinations(self):
        """Gives the number of helicity combinations for external
        wavefunctions"""

        if not self.get('processes'):
            return None

        model = self.get('processes')[0].get('model')

        return reduce(lambda x, y: x * y,
                      [ len(model.get('particle_dict')[wf.get('pdg_code')].\
                            get_helicity_states())\
                        for wf in self.get_external_wavefunctions() ], 1)

    def get_helicity_matrix(self):
        """Gives the helicity matrix for external wavefunctions"""

        if not self.get('processes'):
            return None

        process = self.get('processes')[0]
        model = process.get('model')

        return apply(itertools.product, [ model.get('particle_dict')[\
                                  wf.get('pdg_code')].get_helicity_states()\
                                  for wf in self.get_external_wavefunctions()])

    def get_denominator_factor(self):
        """Calculate the denominator factor due to:
        Averaging initial state color and spin, and
        identical final state particles"""

        model = self.get('processes')[0].get('model')

        initial_legs = filter(lambda leg: leg.get('state') == False, \
                              self.get('processes')[0].get('legs'))

        spin_factor = reduce(lambda x, y: x * y,
                             [ len(model.get('particle_dict')[leg.get('id')].\
                                   get_helicity_states())\
                               for leg in initial_legs ])

        color_factor = reduce(lambda x, y: x * y,
                              [ model.get('particle_dict')[leg.get('id')].\
                                    get('color')\
                                for leg in initial_legs ])

        return spin_factor * color_factor * self['identical_particle_factor']

    def get_color_amplitudes(self):
        """Return a list of (coefficient, amplitude number) lists,
        corresponding to the JAMPs for this matrix element. The
        coefficients are given in the format (fermion factor, color
        coeff (frac), imaginary, Nc power)."""

        if not self.get('color_basis'):
            # No color, simply add all amplitudes with correct factor
            # for first color amplitude
            col_amp = []
            for diagram in self.get('diagrams'):
                for amplitude in diagram.get('amplitudes'):
                    col_amp.append(((amplitude.get('fermionfactor'),
                                    1, False, 0),
                                    amplitude.get('number')))
            return [col_amp]

        # There is a color basis - create a list of coefficients and
        # amplitude numbers

        col_amp_list = []
        for i, col_basis_elem in \
                enumerate(sorted(self.get('color_basis').keys())):

            col_amp = []
            for diag_tuple in self.get('color_basis')[col_basis_elem]:
                res_amps = filter(lambda amp: \
                          tuple(amp.get('color_indices')) == diag_tuple[1],
                          self.get('diagrams')[diag_tuple[0]].get('amplitudes'))
                if not res_amps:
                    raise self.PhysicsObjectError, \
                          """No amplitude found for color structure
                            %s and color index chain (%s) (diagram %i)""" % \
                            (col_basis_elem,
                             str(diag_tuple[1]),
                             diag_tuple[0])

                for res_amp in res_amps:
                    col_amp.append(((res_amp.get('fermionfactor'),
                                     diag_tuple[2],
                                     diag_tuple[3],
                                     diag_tuple[4]),
                                    res_amp.get('number')))

            col_amp_list.append(col_amp)

        return col_amp_list

    @staticmethod
    def check_equal_decay_processes(decay1, decay2):
        """Check if two single-sided decay processes
        (HelasMatrixElements) are equal.

        Note that this has to be called before any combination of
        processes has occured.
        
        Since a decay processes for a decay chain is always generated
        such that all final state legs are completely contracted
        before the initial state leg is included, all the diagrams
        will have identical wave function, independently of the order
        of final state particles.
        
        Note that we assume that the process definitions have all
        external particles, corresponding to the external
        wavefunctions.
        """
        
        assert len(decay1.get('processes')) == 1 == len(decay2.get('processes')), \
                  "Can compare only single process HelasMatrixElements"

        assert len(filter(lambda leg: leg.get('state') == False, \
                      decay1.get('processes')[0].get('legs'))) == 1 and \
               len(filter(lambda leg: leg.get('state') == False, \
                      decay2.get('processes')[0].get('legs'))) == 1, \
                  "Call to check_decay_processes_equal requires " + \
                  "both processes to be unique"

        # Compare bulk process properties (number of external legs,
        # identity factors, number of diagrams, number of wavefunctions
        # initial leg, final state legs
        if len(decay1.get('processes')[0].get("legs")) != \
           len(decay1.get('processes')[0].get("legs")) or \
           len(decay1.get('diagrams')) != len(decay2.get('diagrams')) or \
           decay1.get('identical_particle_factor') != \
           decay2.get('identical_particle_factor') or \
           sum(len(d.get('wavefunctions')) for d in \
               decay1.get('diagrams')) != \
           sum(len(d.get('wavefunctions')) for d in \
               decay2.get('diagrams')) or \
           decay1.get('processes')[0].get('legs')[0].get('id') != \
           decay2.get('processes')[0].get('legs')[0].get('id') or \
           sorted([leg.get('id') for leg in \
                   decay1.get('processes')[0].get('legs')[1:]]) != \
           sorted([leg.get('id') for leg in \
                   decay2.get('processes')[0].get('legs')[1:]]):
            return False

        # Run a quick check to see if the processes are already
        # identical (i.e., the wavefunctions are in the same order)
        if [leg.get('id') for leg in \
            decay1.get('processes')[0].get('legs')] == \
           [leg.get('id') for leg in \
            decay2.get('processes')[0].get('legs')] and \
            decay1 == decay2:
            return True

        # Now check if all diagrams are identical. This is done by a
        # recursive function starting from the last wavefunction
        # (corresponding to the initial state), since it is the
        # same steps for each level in mother wavefunctions

        amplitudes2 = copy.copy(reduce(lambda a1, d2: a1 + \
                                       d2.get('amplitudes'),
                                       decay2.get('diagrams'), []))

        for amplitude1 in reduce(lambda a1, d2: a1 + d2.get('amplitudes'),
                                  decay1.get('diagrams'), []):
            foundamplitude = False
            for amplitude2 in amplitudes2:
                if HelasMatrixElement.check_equal_wavefunctions(\
                   amplitude1.get('mothers')[-1],
                   amplitude2.get('mothers')[-1]):
                    foundamplitude = True
                    # Remove amplitude2, since it has already been matched
                    amplitudes2.remove(amplitude2)
                    break
            if not foundamplitude:
                return False

        return True

    @staticmethod
    def check_equal_wavefunctions(wf1, wf2):
        """Recursive function to check if two wavefunctions are equal.
        First check that mothers have identical pdg codes, then repeat for
        all mothers with identical pdg codes."""

        # End recursion with False if the wavefunctions do not have
        # the same mother pdgs
        if sorted([wf.get('pdg_code') for wf in wf1.get('mothers')]) != \
           sorted([wf.get('pdg_code') for wf in wf2.get('mothers')]):
            return False

        # End recursion with True if these are external wavefunctions
        # (note that we have already checked that the pdgs are
        # identical)
        if not wf1.get('mothers') and not wf2.get('mothers'):
            return True

        mothers2 = copy.copy(wf2.get('mothers'))

        for mother1 in wf1.get('mothers'):
            # Compare mother1 with all mothers in wf2 that have not
            # yet been used and have identical pdg codes
            equalmothers = filter(lambda wf: wf.get('pdg_code') == \
                                  mother1.get('pdg_code'),
                                  mothers2)
            foundmother = False
            for mother2 in equalmothers:
                if HelasMatrixElement.check_equal_wavefunctions(\
                    mother1, mother2):
                    foundmother = True
                    # Remove mother2, since it has already been matched
                    mothers2.remove(mother2)
                    break
            if not foundmother:
                return False

        return True

    @staticmethod
    def sorted_mothers(arg):
        """Gives a list of mother wavefunctions sorted according to
        1. The order of the particles in the interaction
        2. Cyclic reordering of particles in same spin group (if boson)
        3. Fermions ordered IOIOIO... according to the pairs in
           the interaction."""

        assert isinstance(arg, (HelasWavefunction, HelasAmplitude)), \
            "%s is not a valid HelasWavefunction or HelasAmplitude" % repr(arg)

        if not arg.get('interaction_id'):
            return arg.get('mothers')

        pdg_codes = copy.copy(arg.get('pdg_codes'))

        # Remove the argument wavefunction code from pdg_codes

        if isinstance(arg, HelasWavefunction):
            my_spin = arg.get_spin_state_number()
            my_index = pdg_codes.index(arg.get_anti_pdg_code())
            pdg_codes.pop(my_index)
        
        mothers = copy.copy(arg.get('mothers'))

        # First sort according to interaction pdg codes

        mother_codes = [ wf.get_pdg_code() for wf \
                         in mothers ]

        sorted_mothers = []
        same_spin_mothers = []
        same_spin_index = -1
        for i, code in enumerate(pdg_codes):
            index = mother_codes.index(code)
            mother_codes.pop(index)
            mother = mothers.pop(index)
            if isinstance(arg, HelasWavefunction) and \
               my_spin % 2 == 1 and \
               mother.get_spin_state_number() == my_spin:
                # For bosons with same spin as this wf, need special treatment
                if same_spin_index < 0:
                    # Remember starting index for same spin states
                    same_spin_index = i
                same_spin_mothers.append(mother)
            else:
                sorted_mothers.append(mother)

        if mothers:
            raise base_objects.PhysicsObject.PhysicsObjectError

        # Make cyclic reordering of mothers with same spin as this wf
        if same_spin_mothers:
            same_spin_mothers = same_spin_mothers[my_index - same_spin_index:] \
                                + same_spin_mothers[:my_index - same_spin_index]

            # Insert same_spin_mothers in sorted_mothers
            sorted_mothers = sorted_mothers[:same_spin_index] + \
                              same_spin_mothers + sorted_mothers[same_spin_index:]
        # If fermion, partner is the corresponding fermion flow partner
        partner = None
        if isinstance(arg, HelasWavefunction) and my_spin % 2 == 0:
            # Fermion case, just pick out the fermion flow partner
            if my_index % 2 == 0:
                # partner is after arg
                partner_index = my_index
            else:
                # partner is before arg
                partner_index = my_index - 1
            partner = sorted_mothers.pop(partner_index)

        # Reorder fermions pairwise according to incoming/outgoing
        for i in range(0, len(sorted_mothers), 2):
            if sorted_mothers[i].get_spin_state_number() % 2 == 0:
                # This is a fermion, order between this fermion and its brother
                if sorted_mothers[i].get_spin_state_number() > 0 and \
                   sorted_mothers[i + 1].get_spin_state_number() < 0:
                    # Switch places between outgoing and incoming
                    sorted_mothers = sorted_mothers[:i] + \
                                      [sorted_mothers[i+1], sorted_mothers[i]] + \
                                      sorted_mothers[i+2:]
                elif sorted_mothers[i].get_spin_state_number() < 0 and \
                   sorted_mothers[i + 1].get_spin_state_number() > 0:
                    # This is the right order
                    pass
                else:
                    # Two incoming or two outgoing in a row - not good!
                    raise base_objects.PhysicsObject.PhysicsObjectError, \
                    "Two incoming or outgoing fermions in a row: %i, %i" % \
                    (sorted_mothers[i].get_spin_state_number(),
                     sorted_mothers[i+1].get_spin_state_number())
            else:
                # No more fermions in sorted_mothers
                break
            
        # Put back partner into sorted_mothers
        if partner:
            sorted_mothers.insert(partner_index, partner)
        
        # Next sort according to spin_state_number
        return HelasWavefunctionList(sorted_mothers)

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
# HelasDecayChainProcess
#===============================================================================
class HelasDecayChainProcess(base_objects.PhysicsObject):
    """HelasDecayChainProcess: If initiated with a DecayChainAmplitude
    object, generates the HelasMatrixElements for the core process(es)
    and decay chains. Then call combine_decay_chain_processes in order
    to generate the matrix elements for all combined processes."""

    def default_setup(self):
        """Default values for all properties"""

        self['core_processes'] = HelasMatrixElementList()
        self['decay_chains'] = HelasDecayChainProcessList()

    def filter(self, name, value):
        """Filter for valid process property values."""

        if name == 'core_processes':
            if not isinstance(value, HelasMatrixElementList):
                raise self.PhysicsObjectError, \
                        "%s is not a valid HelasMatrixElementList object" % \
                        str(value)

        if name == 'decay_chains':
            if not isinstance(value, HelasDecayChainProcessList):
                raise self.PhysicsObjectError, \
                     "%s is not a valid HelasDecayChainProcessList object" % \
                     str(value)

        return True

    def get_sorted_keys(self):
        """Return process property names as a nicely sorted list."""

        return ['core_processes', 'decay_chains']

    def __init__(self, argument=None):
        """Allow initialization with DecayChainAmplitude"""

        if isinstance(argument, diagram_generation.DecayChainAmplitude):
            super(HelasDecayChainProcess, self).__init__()
            self.generate_matrix_elements(argument)
        elif argument:
            # call the mother routine
            super(HelasDecayChainProcess, self).__init__(argument)
        else:
            # call the mother routine
            super(HelasDecayChainProcess, self).__init__()

    def generate_matrix_elements(self, dc_amplitude):
        """Generate the HelasMatrixElements for the core processes and
        decay processes (separately)"""

        assert isinstance(dc_amplitude, diagram_generation.DecayChainAmplitude), \
                        "%s is not a valid DecayChainAmplitude" % dc_amplitude


        matrix_elements = self['core_processes']

        # Extract the pdg codes of all particles decayed by decay chains
        # since these should not be combined in a MultiProcess
        decay_ids = dc_amplitude.get_decay_ids()

        while dc_amplitude.get('amplitudes'):
            # Pop the amplitude to save memory space
            amplitude = dc_amplitude.get('amplitudes').pop(0)

            logger.info("Generating Helas calls for %s" % \
                        amplitude.get('process').nice_string().\
                                            replace('Process', 'process'))
            matrix_element = HelasMatrixElement(amplitude,
                                                decay_ids=decay_ids,
                                                gen_color=False)

            try:
                # If an identical matrix element is already in the list,
                # then simply add this process to the list of
                # processes for that matrix element
                other_processes = matrix_elements[\
                    matrix_elements.index(matrix_element)].get('processes')
                logger.info("Combining process with %s" % \
                      other_processes[0].nice_string().replace('Process: ', ''))
                other_processes.append(amplitude.get('process'))
            except ValueError:
                # Otherwise, if the matrix element has any diagrams,
                # add this matrix element.
                if matrix_element.get('processes') and \
                       matrix_element.get('diagrams'):
                    matrix_elements.append(matrix_element)

        while dc_amplitude.get('decay_chains'):
            # Pop the amplitude to save memory space
            decay_chain = dc_amplitude.get('decay_chains').pop(0)
            self['decay_chains'].append(HelasDecayChainProcess(\
                decay_chain))

    def combine_decay_chain_processes(self):
        """Recursive function to generate complete
        HelasMatrixElements, combining the core process with the decay
        chains.

        * If the number of decay chains is the same as the number of
        decaying particles, apply each decay chain to the corresponding
        final state particle.
        * If the number of decay chains and decaying final state particles
        don't correspond, all decays applying to a given particle type are
        combined (without double counting)."""

        # End recursion when there are no more decay chains
        if not self['decay_chains']:
            # Just return the list of matrix elements
            return self['core_processes']

        # decay_elements is a list of HelasMatrixElementLists with
        # all decay processes
        decay_elements = []

        for decay_chain in self['decay_chains']:
            # This is where recursion happens
            decay_elements.append(decay_chain.combine_decay_chain_processes())

        # Store the result in matrix_elements
        matrix_elements = HelasMatrixElementList()

        # List of list of ids for the initial state legs in all decay
        # processes
        decay_is_ids = [[element.get('processes')[0].get_initial_ids()[0] \
                         for element in elements]
                         for elements in decay_elements]

        while self['core_processes']:
            # Pop the process to save memory space
            core_process = self['core_processes'].pop(0)
            # Get all final state legs that have a decay chain defined
            fs_legs = filter(lambda leg: any([any([id == leg.get('id') for id \
                            in is_ids]) for is_ids in decay_is_ids]),
                            core_process.get('processes')[0].get_final_legs())
            # List of ids for the final state legs
            fs_ids = [leg.get('id') for leg in fs_legs]
            # Create a dictionary from id to (index, leg number)
            fs_numbers = {}
            fs_indices = {}
            for i, leg in enumerate(fs_legs):
                fs_numbers[leg.get('id')] = \
                    fs_numbers.setdefault(leg.get('id'), []) + \
                    [leg.get('number')]
                fs_indices[leg.get('id')] = \
                    fs_indices.setdefault(leg.get('id'), []) + \
                    [i]

            decay_lists = []
            # Loop over unique final state particle ids
            for fs_id in set(fs_ids):
                # decay_list has the leg numbers and decays for this
                # fs particle id:
                # decay_list = [[[n1,d1],[n2,d2]],[[n1,d1'],[n2,d2']],...]

                decay_list = []

                # Two cases: Either number of decay elements is same
                # as number of decaying particles: Then use the
                # corresponding decay for each particle. Or the number
                # of decay elements is different: Then use any decay
                # chain which defines the decay for this particle.

                if len(fs_legs) == len(decay_elements):
                    # The decay of the different fs parts is given
                    # by the different decay chains, respectively.
                    # Chains is a list of matrix element lists
                    chains = []
                    for index in fs_indices[fs_id]:
                        chains.append(filter(lambda me: \
                                             me.get('processes')[0].\
                                             get_initial_ids()[0] == fs_id,
                                             decay_elements[index]))
                else:
                    # All decays for this particle type are used
                    chain = sum([filter(lambda me: \
                                        me.get('processes')[0].\
                                        get_initial_ids()[0] == fs_id,
                                        decay_chain) for decay_chain in \
                                 decay_elements], [])

                    chains = [chain] * len(fs_numbers[fs_id])

                red_decay_chains = []
                for prod in itertools.product(*chains):

                    # Now, need to ensure that we don't append
                    # duplicate chain combinations, e.g. (a>bc, a>de) and
                    # (a>de, a>bc)
                    
                    # Remove double counting between final states
                    if sorted([p.get('processes')[0] for p in prod],
                              lambda x1, x2: x1.compare_for_sort(x2)) \
                              in red_decay_chains:
                        continue
                    
                    # Store already used combinations
                    red_decay_chains.append(\
                    sorted([p.get('processes')[0] for p in prod],
                              lambda x1, x2: x1.compare_for_sort(x2)))

                    # Add the decays to the list
                    decay_list.append(zip(fs_numbers[fs_id], prod))

                decay_lists.append(decay_list)

            # Finally combine all decays for this process,
            # and combine them, decay by decay
            for decays in itertools.product(*decay_lists):
                
                # Generate a dictionary from leg number to decay process
                decay_dict = dict(sum(decays, []))

                # Make sure to not modify the original matrix element
                model_bk = core_process.get('processes')[0].get('model')
                # Avoid Python copying the complete model every time
                for i, process in enumerate(core_process.get('processes')):
                    process.set('model',base_objects.Model())
                matrix_element = copy.deepcopy(core_process)
                # Avoid Python copying the complete model every time
                for i, process in enumerate(matrix_element.get('processes')):
                    process.set('model', model_bk)
                    core_process.get('processes')[i].set('model', model_bk)
                # Need to replace Particle in all wavefunctions to avoid
                # deepcopy
                org_wfs = core_process.get_all_wavefunctions()
                for i, wf in enumerate(matrix_element.get_all_wavefunctions()):
                    wf.set('particle', org_wfs[i].get('particle'))
                    wf.set('antiparticle', org_wfs[i].get('antiparticle'))

                # Insert the decay chains
                logger.info("Combine %s with decays %s" % \
                            (core_process.get('processes')[0].nice_string().\
                             replace('Process: ', ''), \
                             ", ".join([d.get('processes')[0].nice_string().\
                                        replace('Process: ', '') \
                                        for d in decay_dict.values()])))

                matrix_element.insert_decay_chains(decay_dict)

                try:
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    other_processes = matrix_elements[\
                    matrix_elements.index(matrix_element)].get('processes')
                    logger.info("Combining process with %s" % \
                      other_processes[0].nice_string().replace('Process: ', ''))
                    other_processes.extend(matrix_element.get('processes'))
                except ValueError:
                    # Otherwise, if the matrix element has any diagrams,
                    # add this matrix element.
                    if matrix_element.get('processes') and \
                           matrix_element.get('diagrams'):
                        matrix_elements.append(matrix_element)

        return matrix_elements

#===============================================================================
# HelasDecayChainProcessList
#===============================================================================
class HelasDecayChainProcessList(base_objects.PhysicsObjectList):
    """List of HelasDecayChainProcess objects
    """

    def is_valid_element(self, obj):
        """Test if object obj is a valid HelasDecayChainProcess for the list."""

        return isinstance(obj, HelasDecayChainProcess)

#===============================================================================
# HelasMultiProcess
#===============================================================================
class HelasMultiProcess(base_objects.PhysicsObject):
    """HelasMultiProcess: If initiated with an AmplitudeList,
    generates the HelasMatrixElements for the Amplitudes, identifying
    processes with identical matrix elements"""

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

    def __init__(self, argument=None):
        """Allow initialization with AmplitudeList"""

        if isinstance(argument, diagram_generation.AmplitudeList):
            super(HelasMultiProcess, self).__init__()
            self.generate_matrix_elements(argument)
        elif isinstance(argument, diagram_generation.MultiProcess):
            super(HelasMultiProcess, self).__init__()
            self.generate_matrix_elements(argument.get('amplitudes'))
        elif isinstance(argument, diagram_generation.Amplitude):
            super(HelasMultiProcess, self).__init__()
            self.generate_matrix_elements(\
                diagram_generation.AmplitudeList([argument]))
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

        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
                  "%s is not valid AmplitudeList" % repr(amplitudes)

        # Keep track of already generated color objects, to reuse as
        # much as possible
        list_colorize = []
        list_color_basis = []
        list_color_matrices = []

        matrix_elements = self.get('matrix_elements')

        while amplitudes:
            # Pop the amplitude to save memory space
            amplitude = amplitudes.pop(0)
            if isinstance(amplitude, diagram_generation.DecayChainAmplitude):
                matrix_element_list = HelasDecayChainProcess(amplitude).\
                                      combine_decay_chain_processes()
            else:
                logger.info("Generating Helas calls for %s" % \
                         amplitude.get('process').nice_string().\
                                           replace('Process', 'process'))
                matrix_element_list = [HelasMatrixElement(amplitude,
                                                          gen_color=False)]
            for matrix_element in matrix_element_list:
                assert isinstance(matrix_element, HelasMatrixElement), \
                          "Not a HelasMatrixElement: %s" % matrix_element

                try:
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    other_processes = matrix_elements[\
                    matrix_elements.index(matrix_element)].get('processes')
                    logger.info("Combining process with %s" % \
                      other_processes[0].nice_string().replace('Process: ', ''))
                    other_processes.extend(matrix_element.get('processes'))
                except ValueError:
                    # Otherwise, if the matrix element has any diagrams,
                    # add this matrix element.
                    if matrix_element.get('processes') and \
                           matrix_element.get('diagrams'):
                        matrix_elements.append(matrix_element)

                        # Always create an empty color basis, and the
                        # list of raw colorize objects (before
                        # simplification) associated with amplitude
                        col_basis = color_amp.ColorBasis()
                        new_amp = matrix_element.get_base_amplitude()
                        matrix_element.set('base_amplitude', new_amp)
                        colorize_obj = col_basis.create_color_dict_list(new_amp)

                        try:
                            # If the color configuration of the ME has
                            # already been considered before, recycle
                            # the information
                            col_index = list_colorize.index(colorize_obj)
                            logger.info(\
                              "Reusing existing color information for %s" % \
                              matrix_element.get('processes')[0].nice_string().\
                                                 replace('Process', 'process'))
                        except ValueError:
                            # If not, create color basis and color
                            # matrix accordingly
                            list_colorize.append(colorize_obj)
                            col_basis.build()
                            list_color_basis.append(col_basis)
                            col_matrix = color_amp.ColorMatrix(col_basis)
                            list_color_matrices.append(col_matrix)
                            col_index = -1
                            logger.info(\
                              "Processing color information for %s" % \
                              matrix_element.get('processes')[0].nice_string().\
                                             replace('Process', 'process'))

                matrix_element.set('color_basis', list_color_basis[col_index])
                matrix_element.set('color_matrix',
                                   list_color_matrices[col_index])
            



