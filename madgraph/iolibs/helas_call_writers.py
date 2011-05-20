################################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
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
"""Classes for writing Helas calls. HelasCallWriter is the base class."""

import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.core.color_ordered_amplitudes as color_ordered_amplitudes
import aloha.aloha_writers as aloha_writers
from madgraph import MadGraph5Error


class HelasWriterError(Exception):
    """Class for the error of this module """
    pass

#===============================================================================
# HelasCallWriter
#===============================================================================
class HelasCallWriter(base_objects.PhysicsObject):
    """Language independent base class for writing Helas calls. The
    calls are stored in two dictionaries, wavefunctions and
    amplitudes, with entries being a mapping from a set of spin,
    incoming/outgoing states and Lorentz structure to a function which
    writes the corresponding wavefunction/amplitude call (taking a
    HelasWavefunction/HelasAmplitude as argument)."""

    # Dictionaries used for automatic generation of Helas calls
    # Dictionaries from spin states to letters in Helas call
    mother_dict = {1: 'S', 2: 'O', -2: 'I', 3: 'V', 5: 'T'}

    def default_setup(self):

        self['model'] = base_objects.Model()
        self['wavefunctions'] = {}
        self['amplitudes'] = {}

    def filter(self, name, value):
        """Filter for model property values"""

        if name == 'model':
            if not isinstance(value, base_objects.Model):
                raise self.PhysicsObjectError, \
                    "Object of type %s is not a model" % type(value)

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

        return ['model', 'wavefunctions', 'amplitudes']

    def get_matrix_element_calls(self, matrix_element):
        """Return a list of strings, corresponding to the Helas calls
        for the matrix element"""

        assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
                  "%s not valid argument for get_matrix_element_calls" % \
                  repr(matrix_element)

        res = []
        for diagram in matrix_element.get('diagrams'):
            res.extend([ self.get_wavefunction_call(wf) for \
                         wf in diagram.get('wavefunctions') ])
            res.append("# Amplitude(s) for diagram number %d" % \
                       diagram.get('number'))
            for amplitude in diagram.get('amplitudes'):
                res.append(self.get_amplitude_call(amplitude))

        return res

    def get_wavefunction_calls(self, wavefunctions):
        """Return a list of strings, corresponding to the Helas calls
        for the matrix element"""

        assert isinstance(wavefunctions, helas_objects.HelasWavefunctionList), \
               "%s not valid argument for get_wavefunction_calls" % \
               repr(wavefunctions)

        res = [self.get_wavefunction_call(wf) for wf in wavefunctions]

        return res

    def get_amplitude_calls(self, matrix_element):
        """Return a list of strings, corresponding to the Helas calls
        for the matrix element"""
        
        assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
               "%s not valid argument for get_matrix_element_calls" % \
               repr(matrix_element)            

        res = []
        for diagram in matrix_element.get('diagrams'):
            res.append("# Amplitude(s) for diagram number %d" % \
                       diagram.get('number'))
            for amplitude in diagram.get('amplitudes'):
                res.append(self.get_amplitude_call(amplitude))

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

        assert isinstance(key, tuple), \
                       "%s is not a valid tuple for wavefunction key" % key
        
        assert callable(function), \
                 "%s is not a valid function for wavefunction string" % function

        self.get('wavefunctions')[key] = function
        return True

    def add_amplitude(self, key, function):
        """Set the function for writing the amplitude
        corresponding to the key"""

        assert isinstance(key, tuple), \
                        "%s is not a valid tuple for amplitude key" % str(key)

        assert callable(function), \
            "%s is not a valid function for amplitude string" % str(function)
            
            
        self.get('amplitudes')[key] = function
        return True

    def get_model_name(self):
        """Return the model name"""
        return self['model'].get('name')

    # Customized constructor
    def __init__(self, argument={}):
        """Allow generating a HelasCallWriter from a Model
        """

        if isinstance(argument, base_objects.Model):
            super(HelasCallWriter, self).__init__()
            self.set('model', argument)
        else:
            super(HelasCallWriter, self).__init__(argument)
            
            
            
            
#===============================================================================
# FortranHelasCallWriter
#===============================================================================
class FortranHelasCallWriter(HelasCallWriter):
    """The class for writing Helas calls in Fortran, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the Fortran Helas call based on the Lorentz structure of
    the interaction."""

    # Dictionaries used for automatic generation of Helas calls
    # Dictionaries from spin states to letters in Helas call
    self_dict = {1: 'H', 2: 'F', -2: 'F', 3: 'J', 5: 'U'}
    # Dictionaries used for sorting the letters in the Helas call
    sort_wf = {'O': 0, 'I': 1, 'S': 2, 'T': 3, 'V': 4}
    sort_amp = {'S': 0, 'V': 2, 'T': 1, 'O': 3, 'I': 4}

    def default_setup(self):
        """Set up special Helas calls (wavefunctions and amplitudes)
        that can not be done automatically by generate_helas_call"""

        super(FortranHelasCallWriter, self).default_setup()

        # Add special fortran Helas calls, which are not automatically
        # generated

        # Gluon 4-vertex division tensor calls ggT for the FR sm and mssm

        key = ((3, 3, 5, 3), ('A',))
        call = lambda wf: \
               "CALL UVVAXX(W(1,%d),W(1,%d),%s,zero,zero,zero,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('number'),
                wf.get('coupling')[0],
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 5, 3, 1), ('A',))
        call = lambda wf: \
               "CALL JVTAXX(W(1,%d),W(1,%d),%s,zero,zero,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('number'),
                wf.get('coupling')[0],
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 5), ('A',))
        call = lambda amp: \
               "CALL VVTAXX(W(1,%d),W(1,%d),W(1,%d),%s,zero,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[1].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[2].get('number'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # SM gluon 4-vertex components

        key = ((3, 3, 3, 3, 1), ('gggg3',))
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[1].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[2].get('number'),
                wf.get('coupling')[0],
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), ('gggg1',))
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[1].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[2].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[3].get('number'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3, 1), ('gggg2',))
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[2].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('number'),
                wf.get('coupling')[0],
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), ('gggg2',))
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[2].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[1].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[3].get('number'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3, 1), ('gggg1',))
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[2].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('number'),
                FortranHelasCallWriter.sorted_mothers(wf)[0].get('number'),
                wf.get('coupling')[0],
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), ('gggg3',))
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[1].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[2].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[0].get('number'),
                FortranHelasCallWriter.sorted_mothers(amp)[3].get('number'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # HEFT VVVS calls

        key = ((1, 3, 3, 3, 3), ('',))
        call = lambda wf: \
               "CALL JVVSXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 3, 1, 4), ('',))
        call = lambda wf: \
               "CALL HVVVXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((1, 3, 3, 3), ('',))
        call = lambda amp: \
               "CALL VVVSXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),DUM1,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # HEFT VVVS calls

        key = ((1, 3, 3, 3, 1), ('',))
        call = lambda wf: \
               "CALL JVVSXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 3, 1, 4), ('',))
        call = lambda wf: \
               "CALL HVVVXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((1, 3, 3, 3), ('',))
        call = lambda amp: \
               "CALL VVVSXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),DUM1,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # Spin2 Helas Routine
        key = ((-2, 2, 5), ('',))
        call = lambda amp: \
               "CALL IOTXXX(W(1,%d),W(1,%d),W(1,%d),%s,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('coupling')[0],
                amp.get('mothers')[0].get('mass'),
                amp.get('number'))
        self.add_amplitude(key, call)
        
        key = ((-2, 2, 5, 3), ('',))
        call = lambda wf: \
               "CALL UIOXXX(W(1,%d),W(1,%d),%s,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('coupling')[0],
                wf.get('mothers')[0].get('mass'),
                wf.get('mass'),
                wf.get('width'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        
        key = ((3,3,3,5),('',))
        call = lambda amp: \
               "CALL VVVTXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),1d0,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call) 
 
        key = ((3,3,5),('',))
        call = lambda amp: \
               "CALL VVTXXX(W(1,%d),W(1,%d),W(1,%d),%s,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('coupling')[0],
                amp.get('mothers')[0].get('mass'),
                amp.get('number'))
        self.add_amplitude(key, call)  
        

    def get_wavefunction_call(self, wavefunction):
        """Return the function for writing the wavefunction
        corresponding to the key. If the function doesn't exist,
        generate_helas_call is called to automatically create the
        function."""

        if wavefunction.get('spin') == 1 and \
               wavefunction.get('interaction_id') != 0:
            # Special feature: For HVS vertices with the two
            # scalars different, we need extra minus sign in front
            # of coupling for one of the two scalars since the HVS
            # is asymmetric in the two scalars
            wavefunction.set_scalar_coupling_sign(self['model'])

        val = super(FortranHelasCallWriter, self).get_wavefunction_call(wavefunction)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(wavefunction.get('mothers')) > 3:
            raise self.PhysicsObjectError, \
                  """Automatic generation of Fortran wavefunctions not
                  implemented for > 3 mothers"""

        self.generate_helas_call(wavefunction)
        return super(FortranHelasCallWriter, self).get_wavefunction_call(\
            wavefunction)

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude corresponding
        to the key. If the function doesn't exist, generate_helas_call
        is called to automatically create the function."""

        val = super(FortranHelasCallWriter, self).get_amplitude_call(amplitude)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(amplitude.get('mothers')) > 4:
            raise self.PhysicsObjectError, \
                  """Automatic generation of Fortran amplitudes not
                  implemented for > 4 mothers"""

        self.generate_helas_call(amplitude)
        return super(FortranHelasCallWriter, self).get_amplitude_call(amplitude)

    def generate_helas_call(self, argument):
        """Routine for automatic generation of Fortran Helas calls
        according to just the spin structure of the interaction.

        First the call string is generated, using a dictionary to go
        from the spin state of the calling wavefunction and its
        mothers, or the mothers of the amplitude, to letters.

        Then the call function is generated, as a lambda which fills
        the call string with the information of the calling
        wavefunction or amplitude. The call has different structure,
        depending on the spin of the wavefunction and the number of
        mothers (multiplicity of the vertex). The mother
        wavefunctions, when entering the call, must be sorted in the
        correct way - this is done by the sorted_mothers routine.

        Finally the call function is stored in the relevant
        dictionary, in order to be able to reuse the function the next
        time a wavefunction with the same Lorentz structure is needed.
        """
        
        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "get_helas_call must be called with wavefunction or amplitude"

        call = "CALL "

        call_function = None

        if isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = "#"
            call_function = lambda amp: call
            
            self.add_amplitude(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = call + HelasCallWriter.mother_dict[\
                argument.get_spin_state_number()]
            # Fill out with X up to 6 positions
            call = call + 'X' * (11 - len(call))
            call = call + "(P(0,%d),"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                call = call + "%s,NHEL(%d),"
            call = call + "%+d*IC(%d),W(1,%d))"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            elif argument.is_boson():
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For fermions, need particle/antiparticle
                                 - (-1) ** wf.get_with_flow('is_part'),
                                 wf.get('number_external'),
                                 wf.get('number'))
        else:
            # String is FOVXXX, FIVXXX, JIOXXX etc.
            if isinstance(argument, helas_objects.HelasWavefunction):
                call = call + \
                       FortranHelasCallWriter.self_dict[\
                argument.get_spin_state_number()]

            mother_letters = FortranHelasCallWriter.sorted_letters(argument)

            # If Lorentz structure is given, by default add this
            # to call name
            lor_name = argument.get('lorentz')[0]

            # Take care of special case: WWWW or WWVV calls
            if len(lor_name) > 3 and lor_name[:2] == "WW":
                if lor_name[:4] == "WWWW":
                    mother_letters = "WWWW"[:len(mother_letters)]
                if lor_name[:4] == "WWVV":
                    mother_letters = "W3W3"[:len(mother_letters)]
                lor_name = lor_name[4:]

            call = call + mother_letters
            call = call + lor_name

            # Check if we need to append a charge conjugation flag
            if argument.needs_hermitian_conjugate():
                call = call + 'C'

            assert len(call) < 12, "Call to Helas routine %s should be maximum 6 chars" \
                      % call[5:]

            # Fill out with X up to 6 positions
            call = call + 'X' * (11 - len(call)) + '('
            # Wavefunctions
            call = call + "W(1,%d)," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s,"


            if isinstance(argument, helas_objects.HelasWavefunction):
                # Extra dummy coupling for 4-vector vertices
                if argument.get('lorentz') == 'WWVV':
                    # SM W3W3 vertex
                    call = call + "1D0,"
                elif argument.get('lorentz') == 'WWWW':
                    # SM WWWW vertex
                    call = call + "0D0,"
                elif argument.get('spin') == 3 and \
                       [wf.get('spin') for wf in argument.get('mothers')] == \
                       [3, 3, 3]:
                    # All other 4-vector vertices (FR) - note that gggg
                    # has already been defined
                    call = call + "DUM0,"
                # Mass and width
                call = call + "%s,%s,"
                # New wavefunction
                call = call + "W(1,%d))"
            else:
                # Extra dummy coupling for 4-particle vertices
                # Need to replace later with the correct type
                if argument.get('lorentz') == 'WWVV':
                    # SM W3W3 vertex
                    call = call + "1D0,"
                elif argument.get('lorentz') == 'WWWW':
                    # SM WWWW vertex
                    call = call + "0D0,"
                elif [wf.get('spin') for wf in argument.get('mothers')] == \
                       [3, 3, 3, 3]:
                    # Other 4-vector vertices (FR) - note that gggg
                    # has already been defined
                    call = call + "DUM0,"
                # Amplitude
                call = call + "AMP(%d))"

            if isinstance(argument, helas_objects.HelasWavefunction):
                # Create call for wavefunction
                if len(argument.get('mothers')) == 2:
                    call_function = lambda wf: call % \
                                    (FortranHelasCallWriter.sorted_mothers(wf)[0].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(wf)[1].\
                                     get('number'),
                                     ','.join(wf.get_with_flow('coupling')),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
                else:
                    call_function = lambda wf: call % \
                                    (FortranHelasCallWriter.sorted_mothers(wf)[0].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(wf)[1].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(wf)[2].\
                                     get('number'),
                                     ','.join(wf.get_with_flow('coupling')),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
            else:
                # Create call for amplitude
                if len(argument.get('mothers')) == 3:
                    call_function = lambda amp: call % \
                                    (FortranHelasCallWriter.sorted_mothers(amp)[0].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[1].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[2].\
                                     get('number'),

                                     ','.join(amp.get('coupling')),
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (FortranHelasCallWriter.sorted_mothers(amp)[0].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[1].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[2].\
                                     get('number'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[3].\
                                     get('number'),
                                     ','.join(amp.get('coupling')),
                                     amp.get('number'))

        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

    # Static helper functions

    @staticmethod
    def sorted_letters(arg):
        """Gives a list of letters sorted according to
        the order of letters in the Fortran Helas calls"""

        if isinstance(arg, helas_objects.HelasWavefunction):
            return "".join(sorted([HelasCallWriter.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          FortranHelasCallWriter.sort_wf[l2] - \
                          FortranHelasCallWriter.sort_wf[l1]))

        if isinstance(arg, helas_objects.HelasAmplitude):
            return "".join(sorted([HelasCallWriter.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          FortranHelasCallWriter.sort_amp[l2] - \
                          FortranHelasCallWriter.sort_amp[l1]))

    @staticmethod
    def sorted_mothers(arg):
        """Gives a list of mother wavefunctions sorted according to
        1. The order of the particles in the interaction
        2. Cyclic reordering of particles in same spin group
        3. Fermions ordered IOIOIO... according to the pairs in
           the interaction."""

        assert isinstance(arg, (helas_objects.HelasWavefunction, helas_objects.HelasAmplitude)), \
            "%s is not a valid HelasWavefunction or HelasAmplitude" % repr(arg)

        if not arg.get('interaction_id'):
            return arg.get('mothers')
        my_pdg_code = 0
        my_spin = 0
        if isinstance(arg, helas_objects.HelasWavefunction):
            my_pdg_code = arg.get_anti_pdg_code()
            my_spin = arg.get_spin_state_number()

        sorted_mothers, my_index = arg.get('mothers').sort_by_pdg_codes(\
            arg.get('pdg_codes'), my_pdg_code)

        # If fermion, partner is the corresponding fermion flow partner
        partner = None
        if isinstance(arg, helas_objects.HelasWavefunction) and arg.is_fermion():
            # Fermion case, just pick out the fermion flow partner
            if my_index % 2 == 0:
                # partner is after arg
                partner_index = my_index
            else:
                # partner is before arg
                partner_index = my_index - 1
            partner = sorted_mothers.pop(partner_index)
            # If partner is incoming, move to before arg
            if partner.get_spin_state_number() > 0:
                my_index = partner_index
            else:
                my_index = partner_index + 1

        # Reorder fermions pairwise according to incoming/outgoing
        for i in range(0, len(sorted_mothers), 2):
            if sorted_mothers[i].is_fermion():
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
                # No more fermions in sorted_mothers
                break
            
        # Put back partner into sorted_mothers
        if partner:
            sorted_mothers.insert(partner_index, partner)

        same_spin_mothers = []
        if isinstance(arg, helas_objects.HelasWavefunction):
            # Pick out mothers with same spin, for cyclic reordering
            same_spin_index = -1
            i=0
            while i < len(sorted_mothers):
                if abs(sorted_mothers[i].get_spin_state_number()) == \
                       abs(my_spin):
                    if same_spin_index < 0:
                        # Remember starting index for same spin states
                        same_spin_index = i
                    same_spin_mothers.append(sorted_mothers.pop(i))
                else:
                    i += 1

        # Make cyclic reordering of mothers with same spin as this wf
        if same_spin_mothers:
            same_spin_mothers = same_spin_mothers[my_index - same_spin_index:] \
                                + same_spin_mothers[:my_index - same_spin_index]

            # Insert same_spin_mothers in sorted_mothers
            sorted_mothers = sorted_mothers[:same_spin_index] + \
                              same_spin_mothers + sorted_mothers[same_spin_index:]

        # Next sort according to spin_state_number
        return helas_objects.HelasWavefunctionList(sorted_mothers)


#===============================================================================
# UFOHelasCallWriter
#===============================================================================
class UFOHelasCallWriter(HelasCallWriter):
    """The class for writing Helas calls in Fortran, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the Fortran Helas call based on the Lorentz structure of
    the interaction."""


    def get_wavefunction_call(self, wavefunction, **opt):
        """Return the function for writing the wavefunction
        corresponding to the key. If the function doesn't exist,
        generate_helas_call is called to automatically create the
        function. -UFO ROUTINE-"""
        
        # Special feature: For octet Majorana fermions, need an extra
        # minus sign in the FVI (and FSI?) wavefunction in UFO
        # models. For MG4 models, this is taken care of by calling
        # different routines (in import_v4.py)
        wavefunction.set_octet_majorana_coupling_sign()

        val = super(UFOHelasCallWriter, self).get_wavefunction_call(wavefunction)
        if val:
            return val

        # If function not already existing, try to generate it.
        self.generate_helas_call(wavefunction, **opt)
        return super(UFOHelasCallWriter, self).get_wavefunction_call(\
            wavefunction)

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude corresponding
        to the key. If the function doesn't exist, generate_helas_call
        is called to automatically create the function."""

        val = super(UFOHelasCallWriter, self).get_amplitude_call(amplitude)
        if val:
            return val
        
        # If function not already existing, try to generate it.
        self.generate_helas_call(amplitude)
        return super(UFOHelasCallWriter, self).get_amplitude_call(amplitude)

    # Helper function
    def write_factor(self, factor):
        """Create a suitable string for the factor of the form
        (fraction, is_imaginary?)."""
        imag_dict = {True: "IMAG1", False: "ONE"}
        return str(factor[0]*factor[1]) + "*" + imag_dict[factor[2]]

#===============================================================================
# FortranUFOHelasCallWriter
#===============================================================================
class FortranUFOHelasCallWriter(UFOHelasCallWriter):
    """The class for writing Helas calls in Fortran, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the Fortran Helas call based on the Lorentz structure of
    the interaction."""

    def generate_helas_call(self, argument):
        """Routine for automatic generation of Fortran Helas calls
        according to just the spin structure of the interaction.
        """

        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "get_helas_call must be called with wavefunction or amplitude"
        
        call = "CALL "

        call_function = None

        if isinstance(argument, color_ordered_amplitudes.BGHelasCurrent):
            # Create call for wavefunction
            call += "sumwfs%s(" % "".join([str(m.get('spin')) for \
                                           m in argument.get('mothers')])
            call += "W(1,%d),%s," * len(argument.get('mothers')) + \
                    "W(1,%d))"
            call_function = lambda wf: call % \
                (tuple(sum([[mother.get('number'),
                             self.write_factor(mother.get('factor'))] for \
                            mother in wf.get('mothers')], []) + \
                [wf.get('number')]))
            self.add_wavefunction(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = "#"
            call_function = lambda amp: call
            self.add_amplitude(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = call + HelasCallWriter.mother_dict[\
                argument.get_spin_state_number()]
            # Fill out with X up to 6 positions
            call = call + 'X' * (11 - len(call))
            call = call + "(P(0,%d),"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                call = call + "%s,NHEL(%d),"
            call = call + "%+d*IC(%d),W(1,%d))"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            elif argument.is_boson():
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('number'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For fermions, need particle/antiparticle
                                 - (-1) ** wf.get_with_flow('is_part'),
                                 wf.get('number_external'),
                                 wf.get('number'))
        else:
            # String is LOR1_0, LOR1_2 etc.
            
            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = argument.find_outgoing_number()
            else:
                outgoing = 0

            # Check if we need to append a charge conjugation flag
            l = [str(l) for l in argument.get('lorentz')]
            c_flag = '' 
            if argument.needs_hermitian_conjugate():
                c_flag = "".join(['C%d' % i for i in \
                                  argument.get_conjugate_index()])
            routine_name = aloha_writers.combine_name(
                                        '%s%s' % (l[0], c_flag), l[1:], outgoing)
            call = 'CALL %s' % (routine_name)

            # Add the wave function
            call = call + '('
            # Wavefunctions
            call = call + "W(1,%d)," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s,"

            if isinstance(argument, helas_objects.HelasWavefunction):
                # Create call for wavefunction
                call = call + "%s, %s, W(1,%d))"
                #CALL L_4_011(W(1,%d),W(1,%d),%s,%s, %s, W(1,%d))
                call_function = lambda wf: call % \
                    (tuple([mother.get('number') for mother in wf.get('mothers')]) + \
                    (','.join(wf.get_with_flow('coupling')),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number')))
            else:
                # Amplitude
                call += "AMP(%d))"
                call_function = lambda amp: call % \
                                (tuple([mother.get('number') 
                                          for mother in amp.get('mothers')]) + \
                                (','.join(amp.get('coupling')),
                                amp.get('number')))     
                     
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

#===============================================================================
# CPPUFOHelasCallWriter
#===============================================================================
class CPPUFOHelasCallWriter(UFOHelasCallWriter):
    """The class for writing Helas calls in C++, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the C++ Helas call based on the Lorentz structure of
    the interaction."""

    def generate_helas_call(self, argument):
        """Routine for automatic generation of C++ Helas calls
        according to just the spin structure of the interaction.

        First the call string is generated, using a dictionary to go
        from the spin state of the calling wavefunction and its
        mothers, or the mothers of the amplitude, to difenrentiate wich call is
        done.

        Then the call function is generated, as a lambda which fills
        the call string with the information of the calling
        wavefunction or amplitude. The call has different structure,
        depending on the spin of the wavefunction and the number of
        mothers (multiplicity of the vertex). The mother
        wavefunctions, when entering the call, must be sorted in the
        correct way - this is done by the sorted_mothers routine.

        Finally the call function is stored in the relevant
        dictionary, in order to be able to reuse the function the next
        time a wavefunction with the same Lorentz structure is needed.
        """

        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "get_helas_call must be called with wavefunction or amplitude"
        
        call = ""

        call_function = None

        if isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = "#"
            call_function = lambda amp: call
            self.add_amplitude(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just ixxxxx, oxxxxx, vxxxxx or sxxxxx
            call = call + HelasCallWriter.mother_dict[\
                argument.get_spin_state_number()].lower()
            # Fill out with X up to 6 positions
            call = call + 'x' * (6 - len(call))
            # Specify namespace for Helas calls
            call = call + "(p[perm[%d]],"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                call = call + "mME[%d],hel[%d],"
            call = call + "%+d,w[%d]);"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number')-1)
            elif argument.is_boson():
                call_function = lambda wf: call % \
                                (wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number')-1)
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 # For fermions, need particle/antiparticle
                                 - (-1) ** wf.get_with_flow('is_part'),
                                 wf.get('number')-1)
        else:
            # String is LOR1_0, LOR1_2 etc.

            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = argument.find_outgoing_number()
            else:
                outgoing = 0

            # Check if we need to append a charge conjugation flag
            l = [str(l) for l in argument.get('lorentz')]
            c_flag = '' 
            if argument.needs_hermitian_conjugate():
                c_flag = "".join(['C%d' % i for i in \
                                  argument.get_conjugate_index()])
            routine_name = aloha_writers.combine_name(
                                        '%s%s' % (l[0], c_flag), l[1:], outgoing)
            call = '%s' % (routine_name)

            # Add the wave function
            call = call + '('
            # Wavefunctions
            call = call + "w[%d]," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s,"

            if isinstance(argument, helas_objects.HelasWavefunction):
                # Create call for wavefunction
                call = call + "pars->%s, pars->%s, w[%d]);"
                #CALL L_4_011(W(1,%d),W(1,%d),%s,%s, %s, W(1,%d))
                call_function = lambda wf: call % \
                    (tuple([mother.get('number')-1 for mother in wf.get('mothers')]) + \
                    (','.join(CPPUFOHelasCallWriter.format_coupling(\
                                     wf.get_with_flow('coupling'))),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number')-1))
            else:
                # Amplitude
                call += "amp[%d]);"
                call_function = lambda amp: call % \
                                (tuple([mother.get('number')-1
                                          for mother in amp.get('mothers')]) + \
                                (','.join(CPPUFOHelasCallWriter.format_coupling(\
                                 amp.get('coupling'))),
                                 amp.get('number')-1))
                
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

    @staticmethod
    def format_coupling(couplings):
        """Format the coupling so any minus signs are put in front"""

        output = []
        for coupling in couplings:
            if coupling.startswith('-'):
                output.append("-pars->" + coupling[1:])
            else:
                output.append("pars->" + coupling)
        
        return output
        

#===============================================================================
# PythonUFOHelasCallWriter
#===============================================================================
class PythonUFOHelasCallWriter(UFOHelasCallWriter):
    """The class for writing Helas calls in Python, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the Python Helas call based on the Lorentz structure of
    the interaction."""

    def get_matrix_element_calls(self, matrix_element, gauge_check=False):
        """Return a list of strings, corresponding to the Helas calls
        for the matrix element"""

        assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
                  "%s not valid argument for get_matrix_element_calls" % \
                  repr(matrix_element)

        res = []
        for diagram in matrix_element.get('diagrams'):
            wfs = diagram.get('wavefunctions')
            if gauge_check and diagram.get('number') == 1:
                gauge_check_wfs = [wf for wf in wfs if not wf.get('mothers') \
                                   and wf.get('spin') == 3 \
                                   and wf.get('mass').lower() == 'zero']
                if not gauge_check_wfs:
                    raise HelasWriterError, \
                          'no massless spin one particle for gauge check'
                gauge_check_wf = wfs.pop(wfs.index(gauge_check_wfs[0]))
                res.append(self.generate_helas_call(gauge_check_wf, True)(\
                                                    gauge_check_wf))
            res.extend([ self.get_wavefunction_call(wf) for wf in wfs ])
            res.append("# Amplitude(s) for diagram number %d" % \
                       diagram.get('number'))
            for amplitude in diagram.get('amplitudes'):
                res.append(self.get_amplitude_call(amplitude))
                
        return res



    def generate_helas_call(self, argument, gauge_check=False):
        """Routine for automatic generation of Python Helas calls
        according to just the spin structure of the interaction.
        """

        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "get_helas_call must be called with wavefunction or amplitude"
        
        call_function = None

        if isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = "#"
            call_function = lambda amp: call
            self.add_amplitude(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = "w[%d] = "

            call = call + HelasCallWriter.mother_dict[\
                argument.get_spin_state_number()].lower()
            # Fill out with X up to 6 positions
            call = call + 'x' * (14 - len(call))
            call = call + "(p[%d],"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                if gauge_check and argument.get('spin') == 3 and \
                                                 argument.get('mass') == 'ZERO':
                    call = call + "%s, 4,"
                else:
                    call = call + "%s,hel[%d],"
            call = call + "%+d)"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('number')-1,
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
            elif argument.is_boson():
                if not gauge_check or argument.get('mass') != 'ZERO':
                    call_function = lambda wf: call % \
                                (wf.get('number')-1,
                                 wf.get('number_external')-1,
                                 wf.get('mass'),
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
                else:
                    call_function = lambda wf: call % \
                                (wf.get('number')-1,
                                 wf.get('number_external')-1,
                                 'ZERO',
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('number')-1,
                                 wf.get('number_external')-1,
                                 wf.get('mass'),
                                 wf.get('number_external')-1,
                                 # For fermions, need particle/antiparticle
                                 -(-1)**wf.get_with_flow('is_part'))
        else:
            # String is LOR1_0, LOR1_2 etc.
            
            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = argument.find_outgoing_number()
            else:
                outgoing = 0

            # Check if we need to append a charge conjugation flag
            l = [str(l) for l in argument.get('lorentz')]
            c_flag = '' 
            if argument.needs_hermitian_conjugate():
                c_flag = "".join(['C%d' % i for i in \
                                  argument.get_conjugate_index()])
            routine_name = aloha_writers.combine_name(
                                        '%s%s' % (l[0], c_flag), l[1:], outgoing)


            if isinstance(argument, helas_objects.HelasWavefunction):
                call = 'w[%d] = '
            else:
                call = 'amp[%d] = '
            call += '%s' % routine_name

            # Add the wave function
            call = call + '('
            # Wavefunctions
            call = call + "w[%d]," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s"

            if isinstance(argument, helas_objects.HelasWavefunction):
                # Create call for wavefunction
                call = call + ",%s, %s)"
                #CALL L_4_011(W(1,%d),W(1,%d),%s,%s, %s, W(1,%d))
                call_function = lambda wf: call % \
                                ((wf.get('number')-1,) + \
                                 tuple([mother.get('number')-1 for mother in \
                                        wf.get('mothers')]) + \
                                 (','.join(wf.get_with_flow('coupling')),
                                  wf.get('mass'),
                                  wf.get('width')))
            else:
                call = call + ")"
                # Amplitude
                call_function = lambda amp: call % \
                                ((amp.get('number')-1,) + \
                                 tuple([mother.get('number')-1 
                                        for mother in amp.get('mothers')]) + \
                                 (','.join(amp.get('coupling')),))
        
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            if not gauge_check:
                self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

        return call_function
