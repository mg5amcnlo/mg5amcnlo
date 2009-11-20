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

        if len(arguments) > 0:
            if isinstance(arguments[0],base_objects.DiagramList):
                super(HelasMatrixElement, self).__init__()
                diagram_list = arguments[0]
                optimization = 1
                if len(arguments) > 1 and isinstance(arguments[1],int):
                    optimization = arguments[1]

                    self.generate_helas_diagrams(diagram_list,optimization)
                    self.calculate_fermion_factors(diagram_list)
            else:
                super(HelasMatrixElement, self).__init__(arguments[0])
        else:
            super(HelasMatrixElement, self).__init__()
   
    def generate_helas_diagrams(self, diagram_list, optimization):
        """Starting from a list of Diagrams from the diagram
        generation, generate the corresponding HelasDiagrams, i.e.,
        the wave functions, amplitudes and fermionfactors. Choose
        between default optimization (= 1) or no optimization (= 0,
        for GPU).
        """

    def calculate_fermion_factors(self, diagram_list):
        """Starting from a list of Diagrams from the
        diagram generation, generate the corresponding HelasDiagrams,
        i.e., the wave functions, amplitudes and fermionfactors
        """


