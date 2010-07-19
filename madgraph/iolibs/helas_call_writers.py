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

    # Customized constructor
    def __init__(self, argument={}):
        """Allow generating a HelasModel from a Model
        """

        if isinstance(argument, base_objects.Model):
            super(HelasCallWriter, self).__init__()
            self.set('name', argument.get('name'))
        else:
            super(HelasCallWriter, self).__init__(argument)
            
            
            
            
#===============================================================================
# HelasFortranCallWriter
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
    sort_amp = {'S': 1, 'V': 2, 'T': 0, 'O': 3, 'I': 4}

    def default_setup(self):
        """Set up special Helas calls (wavefunctions and amplitudes)
        that can not be done automatically by generate_helas_call"""

        super(FortranHelasCallWriter, self).default_setup()

        # Add special fortran Helas calls, which are not automatically
        # generated

        # Gluon 4-vertex division tensor calls ggT for the FR sm and mssm

        key = ((3, 3, 5), 'A')

        call = lambda wf: \
               "CALL UVVAXX(W(1,%d),W(1,%d),%s,zero,zero,zero,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),

                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 5, 3), 'A')

        call = lambda wf: \
               "CALL JVTAXX(W(1,%d),W(1,%d),%s,zero,zero,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),

                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 5), 'A')

        call = lambda amp: \
               "CALL VVTAXX(W(1,%d),W(1,%d),W(1,%d),%s,zero,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),

                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)

        # SM gluon 4-vertex components

        key = ((3, 3, 3, 3), 'gggg1')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg1')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3), 'gggg2')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (wf.get('mothers')[2].get('number'),
                wf.get('mothers')[0].get('number'),
                wf.get('mothers')[1].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg2')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (amp.get('mothers')[2].get('number'),
                amp.get('mothers')[0].get('number'),
                amp.get('mothers')[1].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3), 'gggg3')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (wf.get('mothers')[1].get('number'),
                wf.get('mothers')[2].get('number'),
                wf.get('mothers')[0].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg3')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (amp.get('mothers')[1].get('number'),
                amp.get('mothers')[2].get('number'),
                amp.get('mothers')[0].get('number'),
                amp.get('mothers')[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)

    def get_wavefunction_call(self, wavefunction):
        """Return the function for writing the wavefunction
        corresponding to the key. If the function doesn't exist,
        generate_helas_call is called to automatically create the
        function."""

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
            call = call + FortranHelasCallWriter.mother_dict[\
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
            addition = argument.get('lorentz')

            # Take care of special case: WWWW or WWVV calls
            if len(argument.get('lorentz')) > 3 and \
                   argument.get('lorentz')[:2] == "WW":
                if argument.get('lorentz')[:4] == "WWWW":
                    mother_letters = "WWWW"[:len(mother_letters)]
                if argument.get('lorentz')[:4] == "WWVV":
                    mother_letters = "W3W3"[:len(mother_letters)]
                addition = argument.get('lorentz')[4:]

            call = call + mother_letters
            call = call + addition

            # Check if we need to append a charge conjugation flag
            if argument.needs_hermitian_conjugate():
                call = call + 'C'

            if len(call) > 11:
                raise self.PhysicsObjectError, \
                      "Call to Helas routine %s should be maximum 6 chars" \
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
                                    (wf.get('mothers')[0].\
                                     get('number'),
                                     wf.get('mothers')[1].\
                                     get('number'),
                                     wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
                else:
                    call_function = lambda wf: call % \
                                    (wf.get('mothers')[0].\
                                     get('number'),
                                     wf.get('mothers')[1].\
                                     get('number'),
                                     wf.get('mothers')[2].\
                                     get('number'),
                                     wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
            else:
                # Create call for amplitude
                if len(argument.get('mothers')) == 3:
                    call_function = lambda amp: call % \
                                    (amp.get('mothers')[0].\
                                     get('number'),
                                     amp.get('mothers')[1].\
                                     get('number'),
                                     amp.get('mothers')[2].\
                                     get('number'),

                                     amp.get('coupling'),
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (amp.get('mothers')[0].\
                                     get('number'),
                                     amp.get('mothers')[1].\
                                     get('number'),
                                     amp.get('mothers')[2].\
                                     get('number'),
                                     amp.get('mothers')[3].\
                                     get('number'),
                                     amp.get('coupling'),
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
            return "".join(sorted([FortranHelasCallWriter.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          FortranHelasCallWriter.sort_wf[l2] - \
                          FortranHelasCallWriter.sort_wf[l1]))

        if isinstance(arg, helas_objects.HelasAmplitude):
            return "".join(sorted([FortranHelasCallWriter.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          FortranHelasCallWriter.sort_amp[l2] - \
                          FortranHelasCallWriter.sort_amp[l1]))

#===============================================================================
# UFOHelasCallWriter
#===============================================================================
class UFOHelasCallWriter(HelasCallWriter):
    """The class for writing Helas calls in Fortran, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the Fortran Helas call based on the Lorentz structure of
    the interaction."""

    def find_outgoing_number(self, wf):
        "Return the position of the resulting particles in the interactions"
        # First shot: just the index in the interaction
        wf_index = wf.get('pdg_codes').index(wf.get_anti_pdg_code())
        # If fermion, then we need to correct for I/O status
        spin_state = wf.get_spin_state_number()
        if spin_state % 2 == 0:
            if wf_index % 2 == 0 and spin_state < 0:
                # Outgoing particle at even slot -> increase by 1
                wf_index += 1
            elif wf_index % 2 == 1 and spin_state > 0:
                # Incoming particle at odd slot -> decrease by 1
                wf_index -= 1
        
        return wf_index

    def get_wavefunction_call(self, wavefunction):
        """Return the function for writing the wavefunction
        corresponding to the key. If the function doesn't exist,
        generate_helas_call is called to automatically create the
        function."""

        val = super(UFOHelasCallWriter, self).get_wavefunction_call(wavefunction)
        if val:
            return val

        # If function not already existing, try to generate it.
        self.generate_helas_call(wavefunction)
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
            call = call + FortranHelasCallWriter.mother_dict[\
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
            # String is LOR1_1110, FIVXXX, JIOXXX etc.
            
            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = self.find_outgoing_number(argument) + 1
                call = 'CALL %s_%s' % (argument.get('lorentz'), outgoing) 
            else:
                outgoing = 0
                call = 'CALL %s_%s' % (argument.get('lorentz'), outgoing)

            # Check if we need to append a charge conjugation flag
            if argument.needs_hermitian_conjugate():
                call = call + '_C'

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
                    (wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number')))
            else:
                # Amplitude
                call += "AMP(%d))"
                call_function = lambda amp: call % \
                                (tuple([mother.get('number') 
                                          for mother in amp.get('mothers')]) + \
                                (amp.get('coupling'),
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
            call = call + CPPUFOHelasCallWriter.mother_dict[\
                argument.get_spin_state_number()]
            # Fill out with X up to 6 positions
            call = call + 'x' * (6 - len(call))
            call = call + "(pME[%d],"
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
            # String is LOR1_1110, FIVXXX, JIOXXX etc.
            
            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = self.find_outgoing_number(argument)
                outgoing = '1' * outgoing + '0' + \
                            '1' * (len(argument.get('mothers')) - outgoing)
                call = '%s_%s' % (argument.get('lorentz'), outgoing) 
            else:
                outgoing = '1' * len(argument.get('mothers'))
                call = '%s_%s' % (argument.get('lorentz'), outgoing)

            # Check if we need to append a charge conjugation flag
            if argument.needs_hermitian_conjugate():
                call = call + '_c'

            # Add the wave function
            call = call + '('
            # Wavefunctions
            call = call + "w[%d]," * len(argument.get('mothers'))
            # Couplings
            call = call + "%s,"

            if isinstance(argument, helas_objects.HelasWavefunction):
                # Create call for wavefunction
                call = call + "%s, %s, w[%d]);"
                #CALL L_4_011(W(1,%d),W(1,%d),%s,%s, %s, W(1,%d))
                call_function = lambda wf: call % \
                    (tuple([mother.get('number')-1 for mother in wf.get('mothers')]) + \
                    (wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number')-1))
            else:
                # Amplitude
                call += "amp[%d]);"
                call_function = lambda amp: call % \
                                (tuple([mother.get('number')-1
                                          for mother in amp.get('mothers')]) + \
                                (amp.get('coupling'),
                                amp.get('number')-1))
                
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)






