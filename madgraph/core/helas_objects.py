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
    writing a call to a HELAS wavefunction routine: the PDG number, a
    list of mother wavefunctions, interaction id, flow state,
    wavefunction number
    """

    def default_setup(self):
        """Default values for all properties"""

        self['pdg_code'] = 0
        self['mothers'] = HelasWavefunctionList()
        self['interaction_id'] = 0
        self['state'] = 'initial'
        self['number'] = 0
        self['fermionflow'] = 1
        
    def __init__(self, argument = {}):
        """Constructor for the HelasDiagramList. In particular allows
        generating a HelasDiagramList from a DiagramList, with
        automatic generation of the necessary wavefunctions
        """

        if isinstance(argument,base_objects.Leg):
            super(HelasWavefunction, self).__init__()
            self.set('pdg_code', argument.get('id'))
            self.set('number', argument.get('number'))
            self.set('state', argument.get('state'))
        else:
            super(HelasWavefunction, self).__init__(argument)
   
    def filter(self, name, value):
        """Filter for valid wavefunction property values."""

        if name == 'pdg_code':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                      "%s is not a valid pdg_code for wavefunction" % \
                      str(value)

        if name == 'mothers':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                      "%s is not a valid list of mothers for wavefunction" % \
                      str(value)

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for wavefunction interaction id" % str(value)

        if name == 'state':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for wavefunction state" % \
                                                                    str(value)
            if value not in ['initial', 'final', 'intermediate']:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction state (initial|final|intermediate)" % \
                                                                    str(value)
        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for wavefunction number" % str(value)

        if name == 'fermionflow':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for wavefunction number" % str(value)
            if not value in [-1,0,1]:
                raise self.PhysicsObjectError, \
                        "%s is not a valid fermionflow (must be -1, 0 or 1)" % str(value)                

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['pdg_code', 'mothers', 'interaction_id',
                'state', 'number', 'fermionflow']


    # Overloaded operators
    
    def __eq__(self, other):
        """Overloading the equality operator, to make comparison easy
        when checking if wavefunction is already written. Note that
        the number for this wavefunction is irrelevant (not yet
        given), while the number for the mothers is important.
        """

        if not isinstance(other,HelasWavefunction):
            return False

        # Check relevant directly defined properties
        if self['pdg_code'] != other['pdg_code'] or \
           self['interaction_id'] != other['interaction_id'] or \
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

        self['mothers'] = HelasWavefunctionList()
        self['interaction_id'] = 0
        self['number'] = 0
        
    def filter(self, name, value):
        """Filter for valid property values."""

        if name == 'mothers':
            if not isinstance(value, HelasWavefunctionList):
                raise self.PhysicsObjectError, \
                      "%s is not a valid list of mothers for amplitude" % \
                      str(value)

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for amplitude interaction id" % str(value)

        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for amplitude number" % str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['mothers', 'interaction_id', 'number']


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
    
    def __init__(self, *arguments):
        """Constructor for the HelasDiagramList. In particular allows
        generating a HelasDiagramList from a DiagramList, with
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
        external_wavefunctions = [ HelasWavefunction(leg) for leg \
                                   in process.get('legs') ]
        
        incoming_numbers = [ leg.get('number') for leg in filter(lambda leg: \
                                  leg.get('state') == 'initial',
                                  process.get('legs')) ]
        
        # Sort the wavefunctions according to number, just to be sure
        external_wavefunctions.sort(lambda wf1, wf2: \
                                    wf1.get('number')-wf2.get('number'))

        wavefunctions.extend(external_wavefunctions)

        # Now go through the diagrams, looking for undefined wavefunctions

        helas_diagrams = HelasDiagramList()

        for diagram in diagram_list:

            # Dictionary from leg number to wave function, keeps track
            # of the present position in the tree
            number_to_wavefunctions = {}

            # Initialize wavefunctions for this diagram
            diagram_wavefunctions = HelasWavefunctionList()
            if diagram == diagram_list[0] or not optimization:
                diagram_wavefunctions.extend(external_wavefunctions)
            elif not optimization:
                wavefunctions = copy.copy(external_wavefunctions)
            
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

            # If s-channel from incoming particles, flip pdg code

            # I'm not actually sure which particle identity to use for
            # intermediate wavefunction for t-channel particles, since
            # this depends on the order of the particles
            # (e.g., 1<->2), so I just use whatever comes out
            for leg in lastvx.get('legs'):
                if leg.get('number') not in incoming_numbers and \
                   leg.get('state') == 'final':
                    part = model.get('particle_dict')[lastleg.get('id')]
                    lastleg.set('id', part.get_anti_pdg_code())            

            # Go through all vertices except the last and create
            # wavefunctions
            for vertex in vertices:
                legs = copy.copy(vertex.get('legs'))
                last_leg = legs.pop()
                # Generate list of mothers from legs
                mothers = self.getmothers(legs, number_to_wavefunctions,
                                          external_wavefunctions)
                # Now generate new wavefunction for the last leg
                wf = HelasWavefunction(last_leg)
                wf.set('interaction_id',vertex.get('id'))
                wf.set('mothers', mothers)
                wf.set('number', len(wavefunctions) + 1)
                # If s-channel from incoming particles, flip pdg code
                if last_leg.get('number') in incoming_numbers and \
                       leg.get('state') == 'final':
                    part = model.get('particle_dict')[wf.get('pdg_code')]
                    wf.set('pdg_code', part.get_anti_pdg_code())
                wf.set('state','intermediate')
                if wf in wavefunctions and optimization:
                    wf = wavefunctions[wavefunctions.index(wf)]
                else:
                    wavefunctions.append(wf)
                    diagram_wavefunctions.append(wf)
                number_to_wavefunctions[last_leg.get('number')] = wf

            # Find mothers for the amplitude
            legs = lastvx.get('legs')
            mothers = self.getmothers(legs, number_to_wavefunctions,
                                      external_wavefunctions)
                
            # Now generate a HelasAmplitude from the last vertex.
            amp = HelasAmplitude({\
                'interaction_id': lastvx.get('id'),
                'mothers': mothers,
                'number': diagram_list.index(diagram) + 1 })

            # Sort the wavefunctions according to number
            diagram_wavefunctions.sort(lambda wf1, wf2: \
                                       wf1.get('number')-wf2.get('number'))

            # Generate HelasDiagram
            helas_diagrams.append(HelasDiagram({ \
                'wavefunctions': diagram_wavefunctions,
                'amplitude': amp
                }))

        self.set('diagrams',helas_diagrams)

    def calculate_fermion_factors(self, amplitude):
        """Starting from a list of Diagrams from the
        diagram generation, generate the corresponding HelasDiagrams,
        i.e., the wave functions, amplitudes and fermionfactors
        """


    # Helper methods

    def getmothers(self, legs, number_to_wavefunctions,
                   external_wavefunctions):
        """Generate list of mothers from number_to_wavefunctions and
        external_wavefunctions"""
        
        mothers = HelasWavefunctionList()

        for leg in legs:
            if not leg.get('number') in number_to_wavefunctions:
                # This is an external leg, pick from external_wavefunctions
                wf = external_wavefunctions[leg.get('number')-1]
                number_to_wavefunctions[leg.get('number')] = wf
            else:
                # The mother is an existing wavefunction
                wf = number_to_wavefunctions[leg.get('number')]
            mothers.append(wf)

        return mothers

