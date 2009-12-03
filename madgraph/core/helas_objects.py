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
                # Set fermion flow state. Initial particle and final
                # antiparticle are incoming, and vice versa for
                # outgoing
                if leg.get('state') == 'initial' and \
                   self.get('is_part') or \
                   leg.get('state') == 'final' and \
                   not self.get('is_part'):
                    self.set('state', 'incoming')
                else:
                    self.set('state', 'outgoing')
                # For boson, set state to intermediate
                # If initial state, flip PDG code (if has antipart)
                # since all bosons should be treated as outgoing
                if self.get('spin') % 2 == 1:
                    self.set('state', 'intermediate')
                    if leg.get('state') == 'initial':
                        self.set('is_part',not self.get('is_part'))
                        if not self.get('self_antipart'):
                            self.set('pdg_code', -self.get('pdg_code'))
                self.set('interaction_id', interaction_id, model)
        elif arguments:
            super(HelasWavefunction, self).__init__(arguments[0])
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
            if value not in ['incoming', 'outgoing', 'intermediate']:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction state (incoming|outgoing|intermediate)" % \
                                                                    str(value)
        if name == 'fermionflow':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for wavefunction number" % str(value)
            if not value in [-1,0,1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid fermionflow (must be -1, 0 or 1)" % str(value)                

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
                'interaction_id', 'inter_color', 'lorentz', 'couplings',
                'state', 'number_external', 'number', 'fermionflow', 'mothers']

    # Helper functions

    def set_state_and_particle(self, model):
        """Set incoming/outgoing state according to mother states and
        Lorentz structure of the interaction, and set PDG code
        according to the particles in the interaction"""

        if not isinstance(model, base_objects.Model):
            raise self.PhysicsObjectError, \
                  "%s is not a valid model for call to set_state_and_particle" \
                  % repr(model)
        # Start by setting the state of the wavefunction
        if self.get('spin') % 2 == 1:
            # For boson, set state to intermediate
            self.set('state', 'intermediate')
        else:
            # For fermion, set state to same as other fermion (in the right way)
            mother_fermions = filter(lambda wf: wf.get('spin') % 2 == 0,
                                     self.get('mothers'))
            if len(filter(lambda wf: wf.get('state') == 'incoming',
                          self.get('mothers'))) > \
                          len(filter(lambda wf: wf.get('state') == 'outgoing',
                          self.get('mothers'))):
                self.set('state', 'incoming')
            else:
                self.set('state', 'outgoing')

        # We want the particle created here to go into the next
        # vertex, so we need to flip identity for incoming
        # antiparticle and outgoing particle.
        if not self.get('self_antipart') and \
               (self.get('state') == 'incoming' and not self.get('is_part') \
                or self.get('state') == 'outgoing' and self.get('is_part')):
            self.set('pdg_code', -self.get('pdg_code'), model)

        # For a boson, flip code (since going into next vertex)
        if not self.get('self_antipart') and \
               self.get('spin') % 2 == 1:
            self.set('pdg_code', -self.get('pdg_code'), model)

        return True
        
    def check_and_fix_fermion_flow(self, wavefunctions, diagram_wavefunctions):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)). If found, we need to trace back through the
        mother structure (only looking at fermions), until we find a
        Majorana fermion.  Set fermionflow = -1 for this wavefunction,
        as well as all other fermions along this line all the way from
        the initial clash to the external fermion, and consider an
        incoming particle with fermionflow -1 as outgoing (and vice
        versa). Continue until we have N(incoming) = N(outgoing).
        """

        # Clash is defined by whether the mothers have N(incoming) !=
        # N(outgoing) after this state has been subtracted
        mother_states = [ wf.get('state') for wf in self.get('mothers') ]
        if self.get('state') in mother_states:
            mother_states.remove(self.get('state'))

        if len(filter(lambda state: state == 'incoming', mother_states)) == \
           len(filter(lambda state: state == 'outgoing', mother_states)):
            return True

        # IMPLEMENT the rest
        print 'Clashing fermion flow found'

        return True

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

        state_number = {'incoming': -1, 'outgoing': 1, 'intermediate': 1}
        return self.get('fermionflow')* \
               state_number[self.get('state')]* \
               self.get('spin')

    def get_wf_key(self):
        """Generate the (spin, state) tuple used as key for the helas call
        dictionaries in HelasModel"""

        res = []
        for mother in self.get('mothers'):
            res.append(mother.get_spin_state_number())

        # Sort according to spin and flow direction
        res.sort()

        res.append(self.get_spin_state_number())
        
        # IMPLEMENT: Check if we need to append a charge conjugation flag

        return tuple(res)

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

    def check_and_fix_fermion_flow(self, wavefunctions, diagram_wavefunctions):
        """Check for clashing fermion flow (N(incoming) !=
        N(outgoing)). If found, we need to trace back through the
        mother structure (only looking at fermions), until we find a
        Majorana fermion.  Set fermionflow = -1 for this wavefunction,
        as well as all other fermions along this line all the way from
        the initial clash to the external fermion, and consider an
        incoming particle with fermionflow -1 as outgoing (and vice
        versa). Continue until we have N(incoming) = N(outgoing).
        """

        # Clash is defined by whether the mothers have N(incoming) !=
        # N(outgoing) after this state has been subtracted
        mother_states = [ wf.get('state') for wf in self.get('mothers') ]

        if len(filter(lambda state: state == 'incoming', mother_states)) == \
           len(filter(lambda state: state == 'outgoing', mother_states)):
            return True

        # IMPLEMENT the rest
        print 'Clashing fermion flow found'

        return True

    def get_amp_key(self):
        """Generate the (spin, state) tuples used as key for the helas call
        dictionaries in HelasModel"""

        res = []
        for mother in self.get('mothers'):
            res.append(mother.get_spin_state_number())

        # Sort according to spin and flow direction
        res.sort()

        # IMPLEMENT: Check if we need to append a charge conjugation flag

        return tuple(res)

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
        self['fermionfactor'] = 1

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
            if not value in [-1,1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid fermion factor (must be -1 or 1)" % str(value)                

        return True

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
    """HelasMatrixElement: list of HelasDiagrams (ordered)
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
                self.calculate_fermion_factors(amplitude)
            else:
                super(HelasMatrixElement, self).__init__(arguments[0])
        else:
            super(HelasMatrixElement, self).__init__()
   
    def generate_helas_diagrams(self, amplitude, optimization = 1):
        """Starting from a list of Diagrams from the diagram
        generation, generate the corresponding HelasDiagrams, i.e.,
        the wave functions, amplitudes and fermionfactors. Choose
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

        incoming_numbers = [ leg.get('number') for leg in filter(lambda leg: \
                                  leg.get('state') == 'initial',
                                  process.get('legs')) ]

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

            # If wavefunction from incoming particles, flip pdg code
            # (both for s- and t-channel particle)
            lastleg = lastvx.get('legs')[len(lastvx.get('legs')) - 1]
            if lastleg.get('number') in incoming_numbers:
                part = model.get('particle_dict')[lastleg.get('id')]
                lastleg.set('id', part.get_anti_pdg_code())

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
                wf.check_and_fix_fermion_flow(wavefunctions,
                                              diagram_wavefunctions)
                # No number_external for internal propagators
                wf.set('number_external', 0)
                # Wavefunction number is given by: number of external
                # wavefunctions + number of non-external wavefunctions
                # in wavefunctions and diagram_wavefunctions
                number = len(external_wavefunctions) + 1
                number = number + len(filter(lambda wf: wf not in \
                                         external_wavefunctions.values(),
                                         wavefunctions))
                number = number + len(filter(lambda wf: wf not in \
                                         external_wavefunctions.values(),
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
                                           diagram_wavefunctions)
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

    def calculate_fermion_factors(self, amplitude):
        """Starting from a list of Diagrams from the
        diagram generation, generate the corresponding HelasDiagrams,
        i.e., the wave functions, amplitudes and fermionfactors
        """


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
    amplitudes, with entries being a mapping from a set of spin and
    incoming/outgoing states to a function which writes the
    corresponding wavefunction call."""

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

        if wavefunction.get_wf_key() in self.get("wavefunctions").keys():
            return self["wavefunctions"][wavefunction.get_wf_key()](wavefunction)
        else:
            return ""

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude
        corresponding to the key"""

        if amplitude.get_amp_key() in self.get("amplitudes").keys():
            return self["amplitudes"][amplitude.get_amp_key()](amplitude)
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
                # For non-scalars, need mass
                call = call + "%s,NHEL(%d),"
            call = call + "%d*IC(%d),W(1,%d))"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 -(-1)**wf.get_with_flow('is_part'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
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
            # IMPLEMENT Add C and other addition (for HEFT etc) if needed

            # Fill out with X up to 6 positions
            call = call + 'X' * (17 - len(call)) + '('
            # Wavefunctions
            call = call + "W(1,%d)," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s,"

            # IMPLEMENT Here we need to add extra coupling for certain
            # 4-vertices

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
                                     wf.get('couplings').values()[0],
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
                else:
                    call_function = lambda wf: call % \
                                    (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[2].get('number'),
                                     wf.get('couplings').values()[0],
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
                                     amp.get('couplings').values()[0],
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[3].get('number'),
                                     amp.get('couplings').values()[0],
                                     amp.get('number'))                    
                
        if isinstance(argument,HelasWavefunction):
            self.add_wavefunction(argument.get_wf_key(),call_function)
        else:
            self.add_amplitude(argument.get_amp_key(),call_function)
            
    # Static helper functions

    @staticmethod
    def sorted_mothers(arg):
        """Gives a list of mother wavefunctions sorted according to
        the order needed in the Fortran Helas calls"""

        if isinstance(arg, HelasWavefunction) or \
           isinstance(arg, HelasAmplitude):
            return sorted(arg.get('mothers'),
                          lambda wf1, wf2: \
                          HelasFortranModel.sort_amp[HelasFortranModel.mother_dict[\
            wf2.get_spin_state_number()]] - \
                          HelasFortranModel.sort_amp[HelasFortranModel.mother_dict[\
            wf1.get_spin_state_number()]])
        # or \
        #                  wf1.get('number') - wf2.get('number')
    
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
    
