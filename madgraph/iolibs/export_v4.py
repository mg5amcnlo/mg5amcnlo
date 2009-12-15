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

"""Methods and classes to export matrix elements to v4 format."""

import logging
import copy
import re

import madgraph.core.helas_objects as helas_objects

#===============================================================================
# write_amplitude_v4_standalone
#===============================================================================
def write_matrix_element_v4_standalone(fsock, matrix_element, fortran_model):
    """Export a matrix element to a matrix.f file for MG4 standalone"""
 
    helas_calls = fortran_model.get_matrix_element_calls(\
                matrix_element)

    writer = FortranWriter()
    
    for call in helas_calls:
        writer.write_fortran_line(fsock, call)
        
    writer.write_fortran_line(fsock,
                              fortran_model.get_JAMP_line(matrix_element))

    return len(helas_calls)

#===============================================================================
# FortranWriter
#===============================================================================
class FortranWriter():
    """Routines for writing fortran lines. Keeps track of indentation
    and splitting of long lines"""

    keyword_pairs = {'^if.+then': ('^endif', 2),
                     '^do': ('^enddo', 2),
                     '^subroutine': ('^end', 0),
                     'function': ('^end', 0)}
    line_cont_char = '$'
    downcase = True
    line_length = 71
    
    indent = 0
    keyword_list = []
    
    def write_fortran_line(self, fsock, line):
        """Write a fortran line, with correct indent and line splits"""

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise FortranWriterError,\
                  "write_fortran_line must have a single line as argument"

        # Strip leading spaces from line
        myline = line.lstrip()

        # Check if this line is a comment
        comment = False

        # Convert to upper or lower case (unless comment)
        if not comment:
            if self.downcase:
                myline = myline.lower()
            else:
                myline = myline.upper()

        # Check if line starts with dual keyword and adjust indent 
        if self.keyword_list:
            if re.search(self.keyword_pairs[\
                self.keyword_list[len(self.keyword_list) - 1]][0],
                         myline.lower()):
                key = self.keyword_list.pop()
                self.indent = self.indent - self.keyword_pairs[key][1]

        # Strip leading spaces and use our own indent
        res = [" " * (6 + self.indent) + myline]

        # Break line in appropriate places
        while len(res[len(res) - 1]) > self.line_length:
            res.append(" " * 5 + self.line_cont_char + \
                       " " * (self.indent + 1) + \
                       res[len(res) - 1][self.line_length:])
            res[len(res) - 2] = res[len(res) - 2][:self.line_length]

        # Check if line starts with keyword and adjust indent 
        for key in self.keyword_pairs.keys():
            if re.search(key, myline.lower()):
                self.keyword_list.append(key)
                self.indent = self.indent + self.keyword_pairs[key][1]

        # Write line(s) to file
        fsock.write("\n".join(res)+"\n")

        return False

#===============================================================================
# HelasFortranModel
#===============================================================================
class HelasFortranModel(helas_objects.HelasModel):
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
            
        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError, \
                  "get_helas_call must be called with wavefunction or amplitude"

        call = "      CALL "

        call_function = None
            
        if isinstance(argument, helas_objects.HelasWavefunction) and \
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
            if isinstance(argument, helas_objects.HelasWavefunction):
                call = call + \
                       HelasFortranModel.self_dict[\
                argument.get_spin_state_number()]

            mother_letters = HelasFortranModel.sorted_letters(argument)

            # If Lorentz structure is given, by default add this
            # to call name
            addition = argument.get('lorentz')[0]

            # Take care of special case: WWWW or WWVV calls
            if len(argument.get('lorentz')[0]) > 3 and \
                   argument.get('lorentz')[0][:2] == "WW":
                if argument.get('lorentz')[0][:4] == "WWWW":
                    mother_letters = "WWWW"[:len(mother_letters)]
                if argument.get('lorentz')[0][:4] == "WWVV":
                    mother_letters = "W3W3"[:len(mother_letters)]
                addition = argument.get('lorentz')[0][4:]

            call = call + mother_letters
            call = call + addition

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
            call = call + "%s," * min(2,len(argument.get('couplings').keys()))

            if isinstance(argument, helas_objects.HelasWavefunction):
                # Mass and width
                call = call + "%s,%s,"
                # New wavefunction
                call = call + "W(1,%d))"
            else:
                # Amplitude
                call = call + "AMP(%d))"                

            if isinstance(argument,helas_objects.HelasWavefunction):
                # Create call for wavefunction
                if len(argument.get('couplings').keys()) == 1:
                    call_function = lambda wf: call % \
                                    (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                                     wf.get_with_flow('couplings')[(0,0)],
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
                else:
                    call_function = lambda wf: call % \
                                    (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[2].get('number'),
                                     wf.get_with_flow('couplings')[(0,0)],
                                     wf.get_with_flow('couplings')[(0,1)],
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
            else:
                # Create call for amplitude
                if len(argument.get('couplings').keys()) == 1:
                    call_function = lambda amp: call % \
                                    (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                                     amp.get('couplings')[(0,0)],
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[3].get('number'),
                                     amp.get('couplings')[(0,0)],
                                     amp.get('couplings')[(0,1)],
                                     amp.get('number'))

        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument,helas_objects.HelasWavefunction):
            self.add_wavefunction(argument.get_call_key(),call_function)
        else:
            self.add_amplitude(argument.get_call_key(),call_function)
            
    def get_JAMP_line(self, matrix_element):
        """Return the JAMP(1) = sum(fermionfactor * AMP(i)) line"""

        if not isinstance(matrix_element, helas_objects.HelasMatrixElement):
            raise self.PhysicsObjectError, \
                  "%s not valid argument for get_matrix_element_calls" % \
                  repr(matrix_element)

        res = "      JAMP(1)="
        # Add all amplitudes with correct fermion factor
        for diagram in matrix_element.get('diagrams'):
            res = res + "%sAMP(%d)" % (sign(diagram.get('fermionfactor')),
                                         diagram.get('amplitude').get('number'))
        return res

    # Static helper functions

    @staticmethod
    def sorted_mothers(arg):
        """Gives a list of mother wavefunctions sorted according to
        1. the spin order needed in the Fortran Helas calls and
        2. the order of the particles in the interaction (cyclic)
        3. the number for the external leg"""

        if isinstance(arg, helas_objects.HelasWavefunction) or \
           isinstance(arg, helas_objects.HelasAmplitude):
            # First sort according to number_external number
            sorted_mothers1 = sorted(arg.get('mothers'),
                                     lambda wf1, wf2: \
                                     wf1.get('number_external') - \
                                     wf2.get('number_external'))
            # Next sort according to interaction pdg codes
            mother_codes = [ wf.get_pdg_code_outgoing() for wf in sorted_mothers1 ]
            pdg_codes = copy.copy(arg.get('pdg_codes'))
            if isinstance(arg, helas_objects.HelasWavefunction):
                my_code = arg.get_pdg_code_incoming()
                # We need to create the cyclic pdg_codes
                missing_index = pdg_codes.index(my_code)
                pdg_codes_cycl = pdg_codes[missing_index+1:]
                pdg_codes_cycl.extend(pdg_codes[:missing_index])
            else:
                pdg_codes_cycl = pdg_codes

            sorted_mothers2 = helas_objects.HelasWavefunctionList()
            for code in pdg_codes_cycl:
                index = mother_codes.index(code)
                mother_codes.pop(index)
                sorted_mothers2.append(sorted_mothers1.pop(index))

            if sorted_mothers1:
                raise HelasFortranModel.PhysicsObjectError,\
                          "Mismatch of pdg codes, %s != %s" % \
                          (repr(mother_codes),repr(pdg_codes_cycl))

            # Next sort according to spin_state_number
            return sorted(sorted_mothers2, lambda wf1, wf2: \
                          HelasFortranModel.sort_amp[\
                          HelasFortranModel.mother_dict[wf2.get_spin_state_number()]]\
                          - HelasFortranModel.sort_amp[\
                          HelasFortranModel.mother_dict[wf1.get_spin_state_number()]])
        
    @staticmethod
    def sorted_letters(arg):
        """Gives a list of letters sorted according to
        the order of letters in the Fortran Helas calls"""

        if isinstance(arg, helas_objects.HelasWavefunction):
            return "".join(sorted([HelasFortranModel.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          HelasFortranModel.sort_wf[l2] - \
                          HelasFortranModel.sort_wf[l1]))

        if isinstance(arg, helas_objects.HelasAmplitude):
            return "".join(sorted([HelasFortranModel.mother_dict[\
            wf.get_spin_state_number()] for wf in arg.get('mothers')],
                          lambda l1, l2: \
                          HelasFortranModel.sort_amp[l2] - \
                          HelasFortranModel.sort_amp[l1]))
    
#===============================================================================
# Global helper methods
#===============================================================================

def sign(number):
    """Returns '+' if positive, '-' if negative"""

    if number > 0:
        return '+'
    if number < 0:
        return '-'
    
