################################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Classes for writing Helas calls. HelasCallWriter is the base class."""

import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.loop.loop_helas_objects as loop_helas_objects
import models.check_param_card as check_param_card
import aloha.aloha_writers as aloha_writers
import aloha
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
    mother_dict = {1: 'S', 2: 'O', -2: 'I', 3: 'V', 5: 'T', 4:'OR', -4:'IR',
                   99:'P'}

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

    def get_loop_amp_helas_calls(self, matrix_element):
        """Return a list of strings, corresponding to the Helas calls
        for building loop amplitudes (AMPL) only."""

        assert isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement), \
                  "%s not valid argument for get_loop_amp_helas_calls" % \
                  repr(matrix_element)

        res = []

        for diagram in matrix_element.get_loop_diagrams():
            res.append("# Loop amplitude for loop diagram with ID %d" % \
                       diagram.get('number'))
            for amplitude in diagram.get_loop_amplitudes():
                # Substitute the proc_prefix
                res.append(self.get_amplitude_call(amplitude))

        return res

    def get_sqso_target_skip_code(self, number_checked, sqso_target_numbers,
        continue_label, split_orders, squared_orders, comment):
        """Treat the optimization for the squared order target so that 
        unnecessary computations can be skipped. This function returns the lines
        of codes for doing so, based on the number_checked (typically amplitude number)
        to be checked and the list of squared order contributions 
        sqso_target_numbers specifying for each of them  the maximum contributing
        number. The continue_label informs where the code must 'goto' if indeed
        the rest of this part of the computation must be skipped.
        The split_orders and squared_orders values lists, as well as the string 
        comment template is just for printing out a nice comment in the fortran 
        code explaining what is going on."""
        
        res = []
        for sqso_index, target in enumerate(sqso_target_numbers):
            if target!=number_checked:
                continue
            sqso_name = ' '.join(['%s=%d'%(split_orders[i],val) for i, \
                       val in enumerate(squared_orders[sqso_index][0])])
            sqso_identifier = "(%s), i.e. of split order ID=%d,"\
                                                      %(sqso_name, sqso_index+1)
            res.append(comment%sqso_identifier)
            res.append("IF(FILTER_SO.AND.SQSO_TARGET."+\
                                 "EQ.%d) GOTO %d"%(sqso_index+1,continue_label))
        return res
        
    def get_born_ct_helas_calls(self, matrix_element, include_CT=True,
                                            squared_orders=[], split_orders=[]):
        """Return a two lists of strings, the first corresponding to the Helas
        calls for building the non-loop wavefunctions, the born and CT (only if
        include_CT=True). The second correspond to the Helas calls for the 
        UVCT amplitudes only. The squared_orders can provide the information of 
        what is the maximum contributing CT amp number."""

        assert isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement), \
                  "%s not valid argument for get_born_ct_helas_calls" % \
                  repr(matrix_element)

        res = []

        sqso_max_uvctamp = [sqso[1][0] for sqso in squared_orders]
        sqso_max_ctamp = [sqso[1][1] for sqso in squared_orders]

        for diagram in matrix_element.get_born_diagrams():
            res.extend([ self.get_wavefunction_call(wf) for \
                         wf in diagram.get('wavefunctions') ])
            res.append("# Amplitude(s) for born diagram with ID %d" % \
                       diagram.get('number'))
            for amplitude in diagram.get('amplitudes'):
                res.append(self.get_amplitude_call(amplitude))
                
        for diagram in matrix_element.get_loop_diagrams():
            res.extend([ self.get_wavefunction_call(wf) for \
                         wf in diagram.get('wavefunctions') ])
            if diagram.get_ct_amplitudes() and include_CT:
                res.append("# Counter-term amplitude(s) for loop diagram number %d" % \
                                                          diagram.get('number'))
                for ctamp in diagram.get_ct_amplitudes():
                    res.append(self.get_amplitude_call(ctamp))
                    res.extend(self.get_sqso_target_skip_code(ctamp.get('number'), 
                      sqso_max_ctamp, 2000, split_orders, squared_orders,
                      "# At this point, all CT amps needed for %s are computed."))
        
        if not include_CT:
            return res, []
        
        res_UVCT = []
        for diagram in matrix_element.get_loop_UVCT_diagrams():
            res_UVCT.extend([ self.get_wavefunction_call(wf) for \
                         wf in diagram.get('wavefunctions') ])
            res_UVCT.append("# Amplitude(s) for UVCT diagram with ID %d" % \
                       diagram.get('number'))
            for ctamp in diagram.get('amplitudes'):
                res_UVCT.append(self.get_amplitude_call(ctamp))
                res_UVCT.extend(self.get_sqso_target_skip_code(ctamp.get('number'), 
                  sqso_max_uvctamp, 3000, split_orders, squared_orders,
                  "# At this point, all UVCT amps needed for %s are computed."))

        return res, res_UVCT

    def get_loop_matrix_element_calls(self, loop_matrix_element):
        """Return a list of strings, corresponding to the Helas calls
        for the loop matrix element"""
        
        res_born_CT, res_UVCT = self.get_born_ct_helas_calls(loop_matrix_element)
        res = res_born_CT + res_UVCT + \
                              self.get_loop_amp_helas_calls(loop_matrix_element)
        return res
    

    def get_matrix_element_calls(self, matrix_element):
        """Return a list of strings, corresponding to the Helas calls
        for the matrix element"""

        assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
                  "%s not valid argument for get_matrix_element_calls" % \
                  type(matrix_element)

        # Do not reuse the wavefunctions for loop matrix elements
        if isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement):
            return self.get_loop_matrix_element_calls(matrix_element)
        
        me = matrix_element.get('diagrams')
        matrix_element.reuse_outdated_wavefunctions(me)

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
            call = self["amplitudes"][amplitude.get_call_key()]
        except KeyError, error:
            return ""
        else:
            return call(amplitude)

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
               (FortranHelasCallWriter.sorted_mothers(wf)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('me_id'),
                wf.get('coupling')[0],
                wf.get('me_id'))
        self.add_wavefunction(key, call)

        key = ((3, 5, 3, 1), ('A',))
        call = lambda wf: \
               "CALL JVTAXX(W(1,%d),W(1,%d),%s,zero,zero,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('me_id'),
                wf.get('coupling')[0],
                wf.get('me_id'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 5), ('A',))
        call = lambda amp: \
               "CALL VVTAXX(W(1,%d),W(1,%d),W(1,%d),%s,zero,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[1].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[2].get('me_id'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # SM gluon 4-vertex components

        key = ((3, 3, 3, 3, 1), ('gggg3',))
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[1].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[2].get('me_id'),
                wf.get('coupling')[0],
                wf.get('me_id'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), ('gggg1',))
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[1].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[2].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[3].get('me_id'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3, 1), ('gggg2',))
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[2].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('me_id'),
                wf.get('coupling')[0],
                wf.get('me_id'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), ('gggg2',))
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[2].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[1].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[3].get('me_id'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3, 1), ('gggg1',))
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (FortranHelasCallWriter.sorted_mothers(wf)[2].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[1].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(wf)[0].get('me_id'),
                wf.get('coupling')[0],
                wf.get('me_id'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), ('gggg3',))
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (FortranHelasCallWriter.sorted_mothers(amp)[1].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[2].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[0].get('me_id'),
                FortranHelasCallWriter.sorted_mothers(amp)[3].get('me_id'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # HEFT VVVS calls

        key = ((1, 3, 3, 3, 3), ('',))
        call = lambda wf: \
               "CALL JVVSXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('me_id'),
                wf.get('mothers')[1].get('me_id'),
                wf.get('mothers')[2].get('me_id'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('me_id'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 3, 1, 4), ('',))
        call = lambda wf: \
               "CALL HVVVXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('me_id'),
                wf.get('mothers')[1].get('me_id'),
                wf.get('mothers')[2].get('me_id'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('me_id'))
        self.add_wavefunction(key, call)

        key = ((1, 3, 3, 3), ('',))
        call = lambda amp: \
               "CALL VVVSXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),DUM1,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('me_id'),
                amp.get('mothers')[1].get('me_id'),
                amp.get('mothers')[2].get('me_id'),
                amp.get('mothers')[3].get('me_id'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # HEFT VVVS calls

        key = ((1, 3, 3, 3, 1), ('',))
        call = lambda wf: \
               "CALL JVVSXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('me_id'),
                wf.get('mothers')[1].get('me_id'),
                wf.get('mothers')[2].get('me_id'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('me_id'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 3, 1, 4), ('',))
        call = lambda wf: \
               "CALL HVVVXX(W(1,%d),W(1,%d),W(1,%d),DUM1,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('me_id'),
                wf.get('mothers')[1].get('me_id'),
                wf.get('mothers')[2].get('me_id'),
                wf.get('coupling')[0],
                wf.get('mass'),
                wf.get('width'),
                wf.get('me_id'))
        self.add_wavefunction(key, call)

        key = ((1, 3, 3, 3), ('',))
        call = lambda amp: \
               "CALL VVVSXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),DUM1,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('me_id'),
                amp.get('mothers')[1].get('me_id'),
                amp.get('mothers')[2].get('me_id'),
                amp.get('mothers')[3].get('me_id'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call)

        # Spin2 Helas Routine
        key = ((-2, 2, 5), ('',))
        call = lambda amp: \
               "CALL IOTXXX(W(1,%d),W(1,%d),W(1,%d),%s,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('me_id'),
                amp.get('mothers')[1].get('me_id'),
                amp.get('mothers')[2].get('me_id'),
                amp.get('coupling')[0],
                amp.get('mothers')[0].get('mass'),
                amp.get('number'))
        self.add_amplitude(key, call)
        
        key = ((-2, 2, 5, 3), ('',))
        call = lambda wf: \
               "CALL UIOXXX(W(1,%d),W(1,%d),%s,%s,%s,%s,W(1,%d))" % \
               (wf.get('mothers')[0].get('me_id'),
                wf.get('mothers')[1].get('me_id'),
                wf.get('coupling')[0],
                wf.get('mothers')[0].get('mass'),
                wf.get('mass'),
                wf.get('width'),
                wf.get('me_id'))
        self.add_wavefunction(key, call)
        
        key = ((3,3,3,5),('',))
        call = lambda amp: \
               "CALL VVVTXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),1d0,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('me_id'),
                amp.get('mothers')[1].get('me_id'),
                amp.get('mothers')[2].get('me_id'),
                amp.get('mothers')[3].get('me_id'),
                amp.get('coupling')[0],
                amp.get('number'))
        self.add_amplitude(key, call) 
 
        key = ((3,3,5),('',))
        call = lambda amp: \
               "CALL VVTXXX(W(1,%d),W(1,%d),W(1,%d),%s,%s,AMP(%d))" % \
               (amp.get('mothers')[0].get('me_id'),
                amp.get('mothers')[1].get('me_id'),
                amp.get('mothers')[2].get('me_id'),
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
                                 wf.get('me_id'))
            elif argument.is_boson():
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('number_external'),
                                 wf.get('me_id'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external'),
                                 wf.get('mass'),
                                 wf.get('number_external'),
                                 # For fermions, need particle/antiparticle
                                 - (-1) ** wf.get_with_flow('is_part'),
                                 wf.get('number_external'),
                                 wf.get('me_id'))
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
                if argument.get('lorentz') == ['WWVV']:
                    # SM W3W3 vertex
                    call = call + "1D0,"
                elif argument.get('lorentz') == ['WWWW']:
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
                if argument.get('lorentz') == ['WWVV']:
                    # SM W3W3 vertex
                    call = call + "1D0,"
                elif argument.get('lorentz') == ['WWWW']:
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
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(wf)[1].\
                                     get('me_id'),
                                     ','.join(wf.get_with_flow('coupling')),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('me_id'))
                else:
                    call_function = lambda wf: call % \
                                    (FortranHelasCallWriter.sorted_mothers(wf)[0].\
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(wf)[1].\
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(wf)[2].\
                                     get('me_id'),
                                     ','.join(wf.get_with_flow('coupling')),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('me_id'))
            else:
                # Create call for amplitude
                if len(argument.get('mothers')) == 3:
                    call_function = lambda amp: call % \
                                    (FortranHelasCallWriter.sorted_mothers(amp)[0].\
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[1].\
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[2].\
                                     get('me_id'),

                                     ','.join(amp.get('coupling')),
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (FortranHelasCallWriter.sorted_mothers(amp)[0].\
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[1].\
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[2].\
                                     get('me_id'),
                                     FortranHelasCallWriter.sorted_mothers(amp)[3].\
                                     get('me_id'),
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

    mp_prefix = check_param_card.ParamCard.mp_prefix

    def __init__(self, argument={}, hel_sum = False):
        """Allow generating a HelasCallWriter from a Model.The hel_sum argument
        specifies if amplitude and wavefunctions must be stored specifying the
        helicity, i.e. W(1,i) vs W(1,i,H).
        """
        self.hel_sum = hel_sum
        super(FortranUFOHelasCallWriter, self).__init__(argument)

    def format_helas_object(self, prefix, number):
        """ Returns the string for accessing the wavefunction with number in
        argument. Typical output is {prefix}(1,{number}) """
        
        if self.hel_sum:
            return '%s%s,H)'%(prefix, number)
        else:
            return '%s%s)'%(prefix, number)       

    def get_amplitude_call(self, amplitude,**opts):
        """ We overwrite this function here because we must call 
        set_octet_majorana_coupling_sign for all wavefunction taking part in
        this loopHelasAmplitude. This is not necessary in the optimized mode"""        

        # Special feature: For octet Majorana fermions, need an extra
        # minus sign in the FVI (and FSI?) wavefunction in UFO
        # models.
        if isinstance(amplitude,loop_helas_objects.LoopHelasAmplitude):
            for lwf in amplitude.get('wavefunctions'):
                lwf.set_octet_majorana_coupling_sign()
            amplitude.set('coupling',amplitude.get_couplings())
        
        return super(FortranUFOHelasCallWriter, self).get_amplitude_call(
                                                               amplitude,**opts)        
        


    def generate_loop_amplitude_call(self, loopamp):
        """ Routine for automatic generation of a call to CutTools for loop
        amplitudes."""

        call = "LOOP%(numLoopLines)s"
        if (len(loopamp.get('pairing')) != len(loopamp.get('mothers'))):
            call += "%(numMotherWfs)s%(numCouplings)s(%(numeratorNumber)d,"
            for i in range(len(loopamp.get('pairing'))):
                call = call + "%(Pairing{0})d,".format(i)
        else:
            call += "%(numCouplings)s(%(numeratorNumber)d,"            
        for i in range(len(loopamp.get('mothers'))):
            call = call + "%(MotherID{0})d,".format(i+1)
        for i in range(len(loopamp.get('wavefunctions'))-2):
            call = call + \
            "DCMPLX(%(LoopMass{0})s),CMPLX({1}%(LoopMass{0})s,KIND=16),"\
                                                     .format(i+1,self.mp_prefix)
        for i in range(len(loopamp.get('coupling'))):
            call = call + \
                   "%(LoopCoupling{0})s,%(MPLoopCoupling{0})s,".format(i+1)
        call = call + "%(LoopRank)d,"
        call = call + "%(LoopSymmetryFactor)d,%(LoopMultiplier)d,"
        call = call + "%(ampNumber)d,AMPL(1,%(ampNumber)d),S(%(ampNumber)d))"
        
        def create_loop_amp(amplitude):
            helas_dict = amplitude.get_helas_call_dict()
            # Make sure the potential minus sign on coupling appears at the
            # right place when specifying the mp_coupling. It must be
            # -MP__GC10 and not MP__-GC10
            for i in range(len(loopamp.get('coupling'))):
                coupl = helas_dict['LoopCoupling%i'%(i+1)] 
                helas_dict['MPLoopCoupling%i'%(i+1)]= \
                   '-%s%s'%(self.mp_prefix,coupl[1:]) if coupl.startswith('-') \
                                              else '%s%s'%(self.mp_prefix,coupl)
            # We add here the placeholde for the proc_prefix
            return 'CALL %(proc_prefix)s'+call%helas_dict

        self.add_amplitude(loopamp.get_call_key(), create_loop_amp)
        return

    def generate_helas_call(self, argument, startingExternalWFNumber=0):
        """Routine for automatic generation of Fortran Helas calls
        according to just the spin structure of the interaction.
        """

        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "generate_helas_call must be called with wavefunction or amplitude"
        
        call = "CALL "

        call_function = None

        if isinstance(argument, helas_objects.HelasAmplitude) and \
           not isinstance(argument, loop_helas_objects.LoopHelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = "#"
            call_function = lambda amp: call
            self.add_amplitude(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            self.generate_external_wavefunction(argument)
            return
                  
        if isinstance(argument,loop_helas_objects.LoopHelasAmplitude):
            self.generate_loop_amplitude_call(argument)
            return
            
        self.generate_all_other_helas_objects(argument)

    def generate_external_wavefunction(self,argument):
        """ Generate an external wavefunction """
        
        call="CALL "
        call_function = None
        if argument.get('is_loop'):
            call=call+"LCUT_%(conjugate)s%(lcutspinletter)s(Q(0),I,WL(1,%(number)d))"
        else:
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = call + HelasCallWriter.mother_dict[\
                argument.get_spin_state_number()]
            # Fill out with X up to 6 positions
            call = call + 'X' * (11 - len(call))
            call = call + "(P(0,%(number_external)d),"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                call = call + "%(mass)s,NHEL(%(number_external)d),"
            call = call + "%(state_id)+d*IC(%(number_external)d),{0})".format(\
                                    self.format_helas_object('W(1,','%(me_id)d'))

        call_function = lambda wf: call % wf.get_external_helas_call_dict()
        self.add_wavefunction(argument.get_call_key(), call_function)

    def generate_all_other_helas_objects(self,argument):
        """ Generate all the helas objects for which no special handlers was
        placed in generate_helas_call """
        
                
        if isinstance(argument, helas_objects.HelasWavefunction):
            outgoing = argument.find_outgoing_number()
        else:
            outgoing = 0


        # Check if we need to append a charge conjugation flag
        l = [str(l) for l in argument.get('lorentz')]
        flag = []
        if argument.needs_hermitian_conjugate():
            flag = ['C%d' % i for i in \
                                  argument.get_conjugate_index()]
        if (isinstance(argument, helas_objects.HelasWavefunction) and \
           argument.get('is_loop') or \
           (isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('type')=='loop')):
            flag.insert(0,"L")

        # Creating line formatting:
        call = 'CALL %(routine_name)s(%(wf)s%(coup)s%(mass)s%(out)s)'

        arg = {'routine_name': aloha_writers.combine_name(\
                                        '%s' % l[0], l[1:], outgoing, flag, True),
               'coup': ("%%(coup%d)s," * len(argument.get('coupling'))) % \
                                     tuple(range(len(argument.get('coupling'))))                                            
               }

        # select how to write a single wf
        if (isinstance(argument,helas_objects.HelasWavefunction) \
                 and argument.get('is_loop')) or \
                 ((isinstance(argument,helas_objects.HelasAmplitude) \
                 and argument['type']=='loop')):
            base_wf = "W%(WF{0})s,"
        else:
            base_wf = self.format_helas_object('W(1,','%({0})d')+','
        
        # compute the full list of wf
        wf = ''
        for i in range(len(argument.get('mothers'))):
            wf += base_wf.format(i)
        arg['wf'] = wf

                
        # Treat other argument
        # First WaveFunction    
        if isinstance(argument, helas_objects.HelasWavefunction):
            if argument['is_loop']:
                arg['out'] = 'WL(1,%(out)d)'
                if aloha.complex_mass:
                    arg['mass'] = "ML(%(out)d),"
                else:
                    arg['mass'] = "ML(%(out)d),ZERO,"
            else:
                arg['out']=self.format_helas_object('W(1,','%(out)d')                   
                if aloha.complex_mass:
                    arg['mass'] = "DCMPLX(%(CM)s),"
                else:
                    arg['mass'] = "%(M)s,%(W)s,"
        # Standard Amplitude
        elif argument['type'] == 'base':      
            arg['mass'] = ''  
            arg['out'] = self.format_helas_object('AMP(','%(out)d')              
        # Loop Amplitude
        elif argument['type'] == 'loop':
            arg['mass'] = ''
            arg['out'] = 'BUFF(I)'
        # UV Counterterm (and other)
        else:
            arg['mass'] = ''
            ampl = "AMPL({0},%(out)d)".format(argument.get_epsilon_order()+1)
            arg['out'] = '%s' % ampl
            if isinstance(argument,loop_helas_objects.LoopHelasUVCTAmplitude)\
                   and argument.get_UVCT_couplings()!='1.0d0':
                 # add a second line to take into account the multiplicative factor                 
                 call += "\n %(second_line)s "
                 arg['second_line'] = ampl+"="+ampl+"*(%(uvct)s)"           

        # ALL ARGUMENT FORMATTED ###############################################
        # Store the result.
        call = call % arg
        # Now we have a line correctly formatted
        call_function = lambda wf: call % wf.get_helas_call_dict(\
          OptimizedOutput=False, specifyHel=self.hel_sum)

        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

    def get_loop_amplitude_helas_calls(self, loop_matrix_element):
        """ Returns a list of strings corresponding to the Helas calls for each 
        loop amplitude of this loop matrix element. This function is placed in
        this class and not in HelasWriter, because it contains fortran-specific
        code."""        
        
        res = []
        loopHelasAmpNumberTreated=[]
        for ldiag in loop_matrix_element.get_loop_diagrams():
            for lamp in ldiag.get_loop_amplitudes():
                if lamp.get('number') in loopHelasAmpNumberTreated:
                    continue
                else:
                    loopHelasAmpNumberTreated.append(lamp.get('number'))
                lcutpart=self['model'].get_particle(lamp['type'])
                res.append("ELSEIF (ID.EQ.%d) THEN"%lamp.get('number'))
                res.append("#Loop diagram number %d (might be others, just an example)"\
                                                           %ldiag.get('number'))                
                if lcutpart.get('spin')==1:
                    res.append("DO I=1,1")
                elif lcutpart.get('spin')==2 or lcutpart.get('spin')==3:
                    res.append("DO I=1,4")
                else:
                    raise self.PhysicsObjectError, \
                  "The L-cut particle type is not supported"
                # Temporarily relabel the 'me_id' attribute of the external wfs
                # in this wavefunction's mothers so to have them matching the
                # convention in the loop helas calls.
                # The same relabeling is performed for couplings.
                # We save the original values to reset them afterwards.
                externalWfNumber=1
                originalNumbers=[]
                couplingNumber=1
                originalCouplings=[]
                for lwf in lamp.get('wavefunctions'):
                    if lwf.get('coupling')!=['none']:
                        originalCouplings.append(lwf.get('coupling'))
                        couplings=[]
                        for coup in lwf.get('coupling'):
                            couplings.append("LC(%d)"%couplingNumber)
                            couplingNumber=couplingNumber+1 
                        lwf.set('coupling',couplings)
                    for mother in lwf.get('mothers'):
                        if not mother.get('is_loop'):
                            originalNumbers.append(mother.get('number'))
                            mother.set('me_id',externalWfNumber)
                            externalWfNumber=externalWfNumber+1
                # Now we can generate the call for the starting loop wavefunction
                res.append(self.get_wavefunction_call(\
                                         lamp.get_starting_loop_wavefunction()))
                # And now for all the other wavefunctions
                res.extend([ self.get_wavefunction_call(wf) for \
                          wf in lamp.get('wavefunctions') if wf.get('mothers')])

                # Get the last wf generated and the corresponding loop
                # wavefunction number
                for lwf in lamp.get('amplitudes')[0].get('mothers'):
                    if lwf.get('mothers'):
                        last_lwf_number=lwf.get('number')
                        break
                res.append('BUFF(I)=WL(I+4,%d)'%last_lwf_number)
                # And re-establish the original numbering
                indexMothers=0
                indexWfs=0
                for lwf in lamp.get('wavefunctions'):
                    if lwf.get('coupling')!=['none']:
                        lwf.set('coupling',originalCouplings[indexWfs])
                        indexWfs=indexWfs+1
                    for mother in lwf.get('mothers'):
                        if not mother.get('is_loop'):
                            mother.set('me_id',originalNumbers[indexMothers])
                            indexMothers=indexMothers+1 
                res.append('ENDDO')
                if lcutpart.get('spin')==1:
                    res.append("CALL CLOSE_1(BUFF(1),RES)")
                elif lcutpart.get('spin')==2 or lcutpart.get('spin')==3:
                    res.append("CALL CLOSE_4(BUFF(1),RES)")
        # We must change the first 'ELSE IF' into an 'IF'
        res[0]=res[0][4:]
        # And add an ENDIF at the end
        res.append('ENDIF')
        
        return res

#===============================================================================
# FortranUFOHelasCallWriterOptimized
#===============================================================================
class FortranUFOHelasCallWriterOptimized(FortranUFOHelasCallWriter):
    """ Version of FortranUFOHelasCallWriter meeting the needs of the optimized
    output for loop processes """

    def get_amplitude_call(self, *args, **opts):
        """ We overwrite this function here because in the optimized mode one
        does not need to call the function set_octet_majorana_coupling_sign
        for the wavefunctions of the loop amplitudes. So we directly call
        the mother of the mother, namely UFOHelasCallWriter. """
        
        return super(FortranUFOHelasCallWriter, self).get_amplitude_call(
                                                                   *args,**opts)

    def format_helas_object(self, prefix, number):
        """ Returns the string for accessing the wavefunction with number in
        argument. Typical output is {prefix}(1,{number}) """
        
        if self.hel_sum:
            return '%s%s,H)'%(prefix, number)
        else:
            return '%s%s)'%(prefix, number)

    def get_coef_construction_calls(self, matrix_element, group_loops=True,
                                            squared_orders=[], split_orders=[]):
        """ Return the calls to the helas routines to construct the coefficients
        of the polynomial representation of the loop numerator (i.e. Pozzorini
        method). Group the coefficients of the loop with same denominator 
        together if group_loops is set True. The squared_orders can provide the 
        information of what is the maximum contributing loop amp number."""

        assert isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement), \
                  "%s not valid argument for get_coef_construction_calls" % \
                  repr(matrix_element)
        loop_induced = (not matrix_element.get('processes')[0].get('has_born'))
        res = []
        sqso_max_lamp = [sqso[1][2] for sqso in squared_orders]     

        i=0
        for ldiag in matrix_element.get_loop_diagrams():
            res.append("# Coefficient construction for loop diagram with ID %d"\
                       %ldiag.get('number'))
            for lwf in ldiag.get('loop_wavefunctions'):
                    res.append(self.get_wavefunction_call(lwf))
            for lamp in ldiag.get_loop_amplitudes():
                # If the loop grouping is not desired, then make sure it is 
                # turned off here. It is of course not to be included for 
                # loop-induced processes
                if (not group_loops) or loop_induced:
                    lamp.set('loop_group_id',i)
                    i=i+1
                create_coef=[
                     'CREATE_LOOP_COEFS(WL(1,0,1,%(number)d)',
                     '%(loop_rank)d','%(lcut_size)d',
                     '%(loop_number)d','%(LoopSymmetryFactor)d',
                     '%(LoopMultiplier)d']
                if not loop_induced:
                    create_coef.append('%(amp_number)d,H)')
                else:
                    create_coef.append('%(amp_number)d)')
            
                res.append('CALL %(proc_prefix)s'+','.join(create_coef)%{\
                  'number':lamp.get_final_loop_wavefunction().get('number'),
                  'loop_rank':lamp.get_analytic_info('wavefunction_rank'),
                  'lcut_size':lamp.get_lcut_size(),
        # For the loop_number below, we used the id of the 'loop_group' this 
        # amplitude belongs to. All amplitudes of such loop_group will therefore
        # be added into the same LOOPCOEF array component.
                  'loop_number':(lamp.get('loop_group_id')+1),
                  'amp_number':lamp.get('amplitudes')[0].get('number'),
                  'LoopSymmetryFactor':lamp.get('loopsymmetryfactor'),
                  'LoopMultiplier':lamp.get('multiplier')})
                res.extend(self.get_sqso_target_skip_code(
                      lamp.get('amplitudes')[0].get('number'), 
                      sqso_max_lamp, 4000, split_orders, squared_orders,
                      "# At this point, all loop coefficients needed"+
                                                       " for %s are computed."))
        
        coef_merge=['C  Grouping of loop diagrams now done directly when '+\
                                                      'creating the LOOPCOEFS.']
        
        return res, coef_merge

    def get_loop_CT_calls(self, matrix_element, group_loops=True, 
                                            squared_orders=[], split_orders=[]):
        """ Return the calls to CutTools interface routines to launch the
        computation of the contribution of one loop group. The squared_orders 
        can provide the information of the maximum reference loop group ID for
        each contributing squared loop orders."""

        assert isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement), \
                  "%s not valid argument for get_loop_CT_calls" % \
                  repr(matrix_element)
        
        res = []
        
        sqso_max_lgroup_refs = [sqso[1][3] for sqso in squared_orders]
        
        # Either call CutTools for all loop diagrams or for only the reference
        # amplitude for each group (for which the coefficients are the sum of
        # all others)
        if group_loops and matrix_element.get('processes')[0].get('has_born'):
            # Reformat the loop group list in a convenient form
            loop_group_refs=[(lamps[1][0],lamps[1][1:]) for lamps in \
                                              matrix_element.get('loop_groups')]
            for (lamp_ref, lamps) in loop_group_refs:
                res.append("# CutTools call for loop numbers %s"%\
                   ','.join(['%d'%lamp_ref.get('number'),]+\
                                   ['%d'%lamp.get('number') for lamp in lamps]))
                res.append(self.get_amplitude_call(lamp_ref))
                res.extend(self.get_sqso_target_skip_code(
                  lamp_ref.get('loop_group_id'), sqso_max_lgroup_refs, 5000, 
                  split_orders, squared_orders,
                  "# At this point, all reductions needed for %s are computed."))
        else:
            for ldiag in matrix_element.get_loop_diagrams():
                res.append("# CutTools call for loop # %d"%ldiag.get('number'))
                for lamp in ldiag.get_loop_amplitudes():
                    # Make sure the loop number is used instead of the loop
                    # group number
                    loop_group_id_tmp = lamp.get('loop_group_id')
                    lamp.set('loop_group_id',lamp.get('number')-1)
                    res.append(self.get_amplitude_call(lamp))
                    lamp.set('loop_group_id',loop_group_id_tmp)                    

        return res

    def generate_external_wavefunction(self,argument):
        """ Generate an external wavefunction """
        
        call_function = None
        if argument.get('is_loop'):
            call="LCUT_OPT(PL(0,%(number)d),WL(1,1,1,%(number)d))"
            call_function = lambda wf: "CALL %(proc_prefix)s"+ \
                                              call % {'number':wf.get('number')}
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            # For the tree external wavefunction, just call the mother
            FortranUFOHelasCallWriter.generate_external_wavefunction(self,
                                                                       argument)

    def generate_loop_amplitude_call(self, loopamp):
        """ Routine for automatic generation of a call to CutTools for loop
        amplitudes for the optimized output."""
        
        call = "LOOP%(numLoopLines)s"
        if (len(loopamp.get('pairing')) != len(loopamp.get('mothers'))):
            call += "%(numMotherWfs)s("
            for i in range(len(loopamp.get('pairing'))):
                call = call + "%(Pairing{0})d,".format(i)
        else:
            call += "("            
        for i in range(len(loopamp.get('mothers'))):
            call = call + "%(MotherID{0})d,".format(i+1)
        for i in range(len(loopamp.get('wavefunctions'))-2):
            call = call + \
            "DCMPLX(%(LoopMass{0})s),".format(i+1)
        call = call + "%(LoopRank)d,"
        call = call + "I_SO,%(loop_group_id)d)"
        
        # We add here the placeholde for the proc_prefix
        call_function = lambda amp: 'CALL %(proc_prefix)s'+\
                            call % amp.get_helas_call_dict(OptimizedOutput=True)
        self.add_amplitude(loopamp.get_call_key(), call_function)
        return

    def generate_all_other_helas_objects(self,argument):
        """ Generate all the helas objects for which no special handlers was
        placed in generate_helas_call """
        
        if isinstance(argument, helas_objects.HelasWavefunction):
            outgoing = argument.find_outgoing_number()
        else:
            outgoing = 0
            
        if isinstance(argument, helas_objects.HelasAmplitude) and \
                                                  argument.get('type')=='loop':
           raise MadGraph5Error, 'There should not be any helas call '+\
                                'associated with helas amplitudes of type loop.'

        # Check if we need to append a charge conjugation flag
        l = [str(l) for l in argument.get('lorentz')]
        flag = []
        if argument.needs_hermitian_conjugate():
            flag = ['C%d' % i for i in argument.get_conjugate_index()]
        
        if (isinstance(argument, helas_objects.HelasWavefunction) and \
           argument.get('is_loop')):
            flag.insert(0,"L%d"%argument.get_loop_index())

        # Creating line formatting:
        call = 'CALL %(routine_name)s(%(wf)s%(coup)s%(mass)s%(out)s)'

        arg = {'routine_name': aloha_writers.combine_name(\
                                        '%s' % l[0], l[1:], outgoing, flag, True),
               'coup': ("%%(coup%d)s," * len(argument.get('coupling'))) % \
                                     tuple(range(len(argument.get('coupling'))))                                            
               }

        # select how to write a single wf
        if (isinstance(argument,helas_objects.HelasWavefunction) \
                 and argument.get('is_loop')):
            base_wf = "%(WF{0})s,"
        else:
            base_wf = self.format_helas_object('W(1,','%({0})d')+','
        
        # compute the full list of wf
        wf = ''
        for i in range(len(argument.get('mothers'))):
            wf += base_wf.format(i)
        arg['wf'] = wf

                
        # Treat other argument
        # First WaveFunction
        if isinstance(argument, helas_objects.HelasWavefunction):
            if argument['is_loop']:
                arg['out'] = 'PL(0,%(out)d),COEFS'
            else:
                arg['out']=self.format_helas_object('W(1,','%(out)d')                   
            if aloha.complex_mass:
                arg['mass'] = "DCMPLX(%(CM)s),"
            else:
                arg['mass'] = "%(M)s,%(W)s,"
        # Standard Amplitude
        elif argument['type'] == 'base':      
            arg['mass'] = ''  
            arg['out'] = self.format_helas_object('AMP(','%(out)d')
        # UV Counterterm (and other)
        else:
            arg['mass'] = ''
            ampl = "AMPL({0},%(out)d)".format(argument.get_epsilon_order()+1)
            arg['out'] = '%s' % ampl
            if isinstance(argument,loop_helas_objects.LoopHelasUVCTAmplitude)\
                   and argument.get_UVCT_couplings()!='1.0d0':
                 # add a second line to take into account the multiplicative factor                 
                 call += "\n %(second_line)s "
                 arg['second_line'] = ampl+"="+ampl+"*(%(uvct)s)"        

        # ALL ARGUMENT FORMATTED ###############################################
        # Store the result.
        call = call % arg
        if (isinstance(argument, helas_objects.HelasWavefunction) and \
           argument.get('is_loop')):
            # We add here the call to the UPDATE_COEF subroutine
            call += "\nCALL {0}UPDATE_WL_%(loop_mother_rank)d_%(vertex_rank)d("
            call += "WL(1,0,1,%(loop_mother_number)d),%(lcut_size)d,COEFS,"
            call += "%(in_size)d,%(out_size)d,WL(1,0,1,%(out)d))"
        # Now we have a line correctly formatted, with the proc_prefix
        call_function = lambda wf:\
                        (call%wf.get_helas_call_dict(OptimizedOutput=True, 
                             specifyHel=self.hel_sum)).format('%(proc_prefix)s')

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
                                 wf.get('me_id')-1)
            elif argument.is_boson():
                call_function = lambda wf: call % \
                                (wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1) ** (wf.get('state') == 'initial'),
                                 wf.get('me_id')-1)
            else:
                call_function = lambda wf: call % \
                                (wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 wf.get('number_external')-1,
                                 # For fermions, need particle/antiparticle
                                 - (-1) ** wf.get_with_flow('is_part'),
                                 wf.get('me_id')-1)
        else:
            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = argument.find_outgoing_number()
            else:
                outgoing = 0
                
            # Check if we need to append a charge conjugation flag
            l = [str(l) for l in argument.get('lorentz')]
            flag = [] 
            if argument.needs_hermitian_conjugate():
                flag = ['C%d' % i for i in argument.get_conjugate_index()]
                
                
            # Creating line formatting:
            call = '%(routine_name)s(%(wf)s%(coup)s%(mass)s%(out)s);'
            # compute wf
            arg = {'routine_name': aloha_writers.combine_name(\
                                            '%s' % l[0], l[1:], outgoing, flag,True),
                   'wf': ("w[%%(%d)d]," * len(argument.get('mothers'))) % \
                                      tuple(range(len(argument.get('mothers')))),
                    'coup': ("pars->%%(coup%d)s," * len(argument.get('coupling'))) % \
                                     tuple(range(len(argument.get('coupling'))))           
                   } 
            if isinstance(argument, helas_objects.HelasWavefunction):
                arg['out'] = 'w[%(out)d]'
                if aloha.complex_mass:
                    arg['mass'] = "pars->%(CM)s,"
                else:
                    arg['mass'] = "pars->%(M)s,pars->%(W)s,"
            else:        
                arg['out'] = 'amp[%(out)d]'
                arg['mass'] = ''
                
            call = call % arg
            # Now we have a line correctly formatted
            call_function = lambda wf: self.format_coupling(
                                         call % wf.get_helas_call_dict(index=0))
        
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

    @staticmethod
    def format_coupling(call):
        """Format the coupling so any minus signs are put in front"""

        return call.replace('pars->-', '-pars->')
        

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

        me = matrix_element.get('diagrams')
        matrix_element.reuse_outdated_wavefunctions(me)

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
                                (wf.get('me_id')-1,
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
            elif argument.is_boson():
                if not gauge_check or argument.get('mass') != 'ZERO':
                    call_function = lambda wf: call % \
                                (wf.get('me_id')-1,
                                 wf.get('number_external')-1,
                                 wf.get('mass'),
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
                else:
                    call_function = lambda wf: call % \
                                (wf.get('me_id')-1,
                                 wf.get('number_external')-1,
                                 'ZERO',
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('me_id')-1,
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
            flag = []
            if argument.needs_hermitian_conjugate():
                flag = ['C%d' % i for i in argument.get_conjugate_index()]
                
                
            # Creating line formatting:
            call = '%(out)s= %(routine_name)s(%(wf)s%(coup)s%(mass)s)'
            # compute wf
            arg = {'routine_name': aloha_writers.combine_name(\
                                            '%s' % l[0], l[1:], outgoing, flag, True),
                   'wf': ("w[%%(%d)d]," * len(argument.get('mothers'))) % \
                                      tuple(range(len(argument.get('mothers')))),
                    'coup': ("%%(coup%d)s," * len(argument.get('coupling'))) % \
                                     tuple(range(len(argument.get('coupling'))))           
                   }

            if isinstance(argument, helas_objects.HelasWavefunction):
                arg['out'] = 'w[%(out)d]'
                if aloha.complex_mass:
                    arg['mass'] = "%(CM)s"
                else:
                    arg['mass'] = "%(M)s,%(W)s"
            else:
                arg['coup'] = arg['coup'][:-1] #removing the last coma
                arg['out'] = 'amp[%(out)d]'
                arg['mass'] = ''
                
            call = call % arg
            # Now we have a line correctly formatted
            call_function = lambda wf: call % wf.get_helas_call_dict(index=0)
                
            routine_name = aloha_writers.combine_name(
                                        '%s' % l[0], l[1:], outgoing, flag)
        
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            if not gauge_check:
                self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

        return call_function
