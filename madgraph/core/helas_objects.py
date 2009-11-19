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
    writing a call to a HELAS wavefunction routine: the leg with info
    about the PDG number, leg number and virtuality/direction of the
    particle (t/s-channel), a list of mother wavefunctions,
    interaction id, flow direction, wavefunction number
    """

    def default_setup(self):
        """Default values for all properties"""

        self['leg'] = None
        self['mothers'] = HelasWavefunctionList()
        self['interaction_id'] = 0
        self['direction'] = 'incoming'
        self['number'] = 0
        
    def filter(self, name, value):
        """Filter for valid wavefunction property values."""

        if name == 'leg':
            if not isinstance(value, base_objects.Leg):
                raise self.PhysicsObjectError, \
                        "%s is not a valid leg for wavefunction" % \
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

        if name == 'direction':
            if not isinstance(value, str):
                raise self.PhysicsObjectError, \
                        "%s is not a valid string for wavefunction direction" % \
                                                                    str(value)
            if value not in ['incoming', 'outgoing']:
                raise self.PhysicsObjectError, \
                        "%s is not a valid wavefunction direction (incoming|outgoing)" % \
                                                                    str(value)
        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for leg id" % str(value)

        return True

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""

        return ['leg', 'mothers', 'interaction_id', 'direction', 'number']


    # Overloaded operators
    
    def __eq__(self, other):
        """Overloading the equality operator, to make comparison easy
        when checking if wavefunction is already written. Note that
        the number for this wavefunction is irrelevant (not yet
        given), while the number for the mothers is important.
        """

        if not isinstance(other,HelasWavefunction):
            return False

        if not self['leg'] or not other['leg']:
            return False

        if self['interaction_id'] != other['interaction_id'] or \
           self['direction'] != other['direction']:
            return False

        if self['leg'].get('state') != other['leg'].get('state') or \
               self['leg'].get('id') != other['leg'].get('id') or \
               self['leg'].get('number') != other['leg'].get('number'):
            return False

        if self['mothers'] and not other['mothers'] or \
           not self['mothers'] and other['mothers']:
            return False

        if not self['mothers']:
            # We have made all relevant comparisons, so they are equal
            return True

        if len(self['mothers']) != len(other['mothers']):
            return False

        for i in range(len(self['mothers'])):
            if self['mothers'][i].get('number') != other['mothers'][i].get('number'):
                return False
                


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
                      "%s is not a valid list of mothers for wavefunction" % \
                      str(value)

        if name == 'interaction_id':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for wavefunction interaction id" % str(value)

        if name == 'number':
            if not isinstance(value, int):
                raise self.PhysicsObjectError, \
                        "%s is not a valid integer for leg id" % str(value)

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
    
    def __init__(self, *arguments):
        """Constructor for the HelasDiagramList. In particular allows
        generating a HelasDiagramList from a DiagramList, with
        automatic generation of the necessary wavefunctions
        """

        if len(arguments) > 0:
            if isinstance(arguments[0],base_objects.DiagramList):
                list.__init__(self)
                diagram_list = arguments[0]
                optimization = 1
                if len(arguments) > 1 and isinstance(arguments[1],int):
                    optimization = arguments[1]

                self.extend(generate_helas_diagrams(diagram_list, optimization))
            else:
                list.__init__(self, arguments[0])
   
    @staticmethod
    def generate_helas_diagrams(diagram_list,optimization):
        """Static method. Starting from a list of Diagrams from the
        diagram generation, generate the corresponding HelasDiagrams,
        i.e., the wave functions, amplitudes and fermionfactors
        """
        
        return HelasDiagramList()
