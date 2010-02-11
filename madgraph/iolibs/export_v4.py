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

import copy
import fractions
import logging
import os
import re

import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.files as files
import madgraph.iolibs.misc as misc

#===============================================================================
# write_matrix_element_v4_standalone
#===============================================================================
def write_matrix_element_v4_standalone(fsock, matrix_element, fortran_model):
    """Export a matrix element to a matrix.f file in MG4 standalone format"""

    if not matrix_element.get('processes') or \
           not matrix_element.get('diagrams'):
        return 0

    writer = FortranWriter()
    # Set lowercase/uppercase Fortran code
    FortranWriter.downcase = True

    replace_dict = {}

    # Extract version number and date from VERSION file
    info_lines = get_mg5_info_lines()
    replace_dict['info_lines'] = info_lines

    # Extract process info lines
    process_lines = get_process_info_lines(matrix_element)
    replace_dict['process_lines'] = process_lines

    # Extract number of external particles
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    replace_dict['nexternal'] = nexternal

    # Extract ncomb
    ncomb = matrix_element.get_helicity_combinations()
    replace_dict['ncomb'] = ncomb

    # Extract helicity lines
    helicity_lines = get_helicity_lines(matrix_element)
    replace_dict['helicity_lines'] = helicity_lines

    # Extract overall denominator
    # Averaging initial state color, spin, and identical FS particles
    den_factor_line = get_den_factor_line(matrix_element)
    replace_dict['den_factor_line'] = den_factor_line

    # Extract ngraphs
    ngraphs = matrix_element.get_number_of_amplitudes()
    replace_dict['ngraphs'] = ngraphs

    # Extract nwavefuncs
    nwavefuncs = matrix_element.get_number_of_wavefunctions()
    replace_dict['nwavefuncs'] = nwavefuncs

    # Extract ncolor
    ncolor = max(1, len(matrix_element.get('color_basis')))
    replace_dict['ncolor'] = ncolor

    # Extract color data lines
    color_data_lines = get_color_data_lines(matrix_element)
    replace_dict['color_data_lines'] = "\n".join(color_data_lines)

    # Extract helas calls
    helas_calls = fortran_model.get_matrix_element_calls(\
                matrix_element)
    replace_dict['helas_calls'] = "\n".join(helas_calls)

    # Extract JAMP lines
    jamp_lines = get_JAMP_lines(matrix_element)
    replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

    file = \
"""      SUBROUTINE SMATRIX(P,ANS)
C  
%(info_lines)s
C 
C MadGraph StandAlone Version
C 
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
      REAL*8 T
      REAL*8 MATRIX
      INTEGER IHEL,IDEN, I
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./
%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
      NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
                 T=MATRIX(P ,NHEL(1,IHEL),JC(1))            
               ANS=ANS+T
               IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
                   GOODHEL(IHEL)=.TRUE.
               ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)
      END
       
       
      REAL*8 FUNCTION MATRIX(P,NHEL,IC)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
C  
C GLOBAL VARIABLES
C  
      include "coupl.inc"
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
%(jamp_lines)s

      MATRIX = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
      END""" % replace_dict

    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return len(filter(lambda call: call.find('#') != 0, helas_calls))


#===============================================================================
# write_nexternal_file
#===============================================================================
def write_nexternal_file(fsock, matrix_element, fortran_model):
    """Write the nexternal.inc file for MG4"""

    writer = FortranWriter()

    replace_dict = {}

    # Extract number of external particles
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    replace_dict['nexternal'] = nexternal
    replace_dict['ninitial'] = ninitial

    file = \
"""   integer    nexternal
      parameter (nexternal=%(nexternal)d)
      integer    nincoming
      parameter (nincoming=%(ninitial)d)""" % replace_dict


    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_pmass_file
#===============================================================================
def write_pmass_file(fsock, matrix_element, fortran_model):
    """Write the pmass.inc file for MG4"""

    writer = FortranWriter()

    process = matrix_element.get('processes')[0]
    model = process.get('model')
    
    lines = []
    for i, leg in enumerate(process.get('legs')):
        lines.append("pmass(%d)=abs(%s)" % \
                     (i + 1, model.get('particle_dict')[leg.get('id')].\
                      get('mass')))

    # Write the file
    for line in lines:
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# write_ngraphs_file
#===============================================================================
def write_ngraphs_file(fsock, matrix_element, fortran_model):
    """Write the ngraphs.inc file for MG4"""

    writer = FortranWriter()

    # Extract number of amplitudes
    ngraphs = matrix_element.get_number_of_amplitudes()

    file = \
"""   integer    n_max_cg
      parameter (n_max_cg=%d)""" % ngraphs


    # Write the file
    for line in file.split('\n'):
        writer.write_fortran_line(fsock, line)

    return True

#===============================================================================
# generate_subprocess_directory_v4_standalone
#===============================================================================
def generate_subprocess_directory_v4_standalone(matrix_element,
                                                fortran_model,
                                                path = os.getcwd()):
    """Generate the Pxxxxx directory for a subprocess in MG4 standalone,
    including the necessary matrix.f and nexternal.inc files"""

    cwd = os.getcwd()

    # Create the directory PN_xx_xxxxx in the specified path
    dirpath = os.path.join(path, \
                   "P%s" % matrix_element.get('processes')[0].shell_string_v4())
    try:
        os.mkdir(dirpath)
    except os.error as error:
        logging.warning(error.strerror + " " + dirpath)
    
    try:
        os.chdir(dirpath)
    except os.error:
        logging.error('Could not cd to directory %s' % dirpath)
        return 0

    logging.info('Creating files in directory %s' % dirpath)

    # Create the matrix.f file and the nexternal.inc file
    filename = 'matrix.f'
    calls = files.write_to_file(filename,
                                write_matrix_element_v4_standalone,
                                matrix_element,
                                fortran_model)

    filename = 'nexternal.inc'
    files.write_to_file(filename,
                        write_nexternal_file,
                        matrix_element,
                        fortran_model)

    filename = 'pmass.inc'
    files.write_to_file(filename,
                        write_pmass_file,
                        matrix_element,
                        fortran_model)

    filename = 'ngraphs.inc'
    files.write_to_file(filename,
                        write_ngraphs_file,
                        matrix_element,
                        fortran_model)

    linkfiles = ['check_sa.f', 'coupl.inc', 'makefile']

    try:
        for file in linkfiles:
            os.symlink(os.path.join('..',file), file)
    except os.error:
        logging.warning('Could not link to ' + os.path.join('..',file))
            
    # Return to original PWD
    os.chdir(cwd)

    if not calls:
        calls = 0
    return calls

#===============================================================================
# Helper functions
#===============================================================================
def get_mg5_info_lines():
    """Return info lines for MG5, suitable to place at beginning of
    Fortran files"""

    info = misc.get_pkg_info()
    info_lines = ""
    if info.has_key('version') and  info.has_key('date'):
        info_lines = "C  Generated by MadGraph 5 v. %s, %s\n" % \
                     (info['version'], info['date'])
        info_lines = info_lines + \
                     "C  By the MadGraph Development Team\n" + \
                     "C  Please visit us at https://launchpad.net/madgraph5"
    return info_lines

def get_process_info_lines(matrix_element):
    """Return info lines describing the processes for this matrix element"""

    return"\n".join([ "C " + process.nice_string().replace('\n', '\nC ') \
                     for process in matrix_element.get('processes')])


def get_helicity_lines(matrix_element):
    """Return the Helicity matrix definition lines for this matrix element"""

    helicity_line_list = []
    i = 0
    for helicities in matrix_element.get_helicity_matrix():
        i = i + 1
        int_list = [i, len(helicities)]
        int_list.extend(helicities)
        helicity_line_list.append(\
            ("DATA (NHEL(IHEL,%4r),IHEL=1,%d) /" + \
             ",".join(['%2r'] * len(helicities)) + "/") % tuple(int_list))

    return "\n".join(helicity_line_list)

def get_color_data_lines(matrix_element, n=6):
    """Return the color matrix definition lines for this matrix element. Split
    rows in chunks of size n."""

    if not matrix_element.get('color_matrix'):
        return ["DATA Denom(1)/1/", "DATA (CF(i,1),i=1,1) /1/"]
    else:
        ret_list = []
        my_cs = color.ColorString()
        for index, denominator in \
            enumerate(matrix_element.get('color_matrix').\
                                             get_line_denominators()):
            # First write the common denominator for this color matrix line
            ret_list.append("DATA Denom(%i)/%i/" % (index + 1, denominator))
            # Then write the numerators for the matrix elements
            num_list = matrix_element.get('color_matrix').\
                                        get_line_numerators(index, denominator)

            for k in xrange(0, len(num_list), n):
                ret_list.append("DATA (CF(i,%3r),i=%3r,%3r) /%s/" % \
                                (index + 1, k + 1, min(k + n, len(num_list)),
                                 ','.join(["%5r" % i for i in num_list[k:k + n]])))

            my_cs.from_immutable(matrix_element.get('color_basis').keys()[index])
            ret_list.append("C %s" % repr(my_cs))
        return ret_list


def get_den_factor_line(matrix_element):
    """Return the denominator factor line for this matrix element"""

    return "DATA IDEN/%2r/" % \
           matrix_element.get_denominator_factor()

def get_JAMP_lines(matrix_element):
    """Return the JAMP(1) = sum(fermionfactor * AMP(i)) line"""

    if not matrix_element.get('color_basis'):
        res = "JAMP(1)="
        # Add all amplitudes with correct fermion factor
        for diagram in matrix_element.get('diagrams'):
            for amplitude in diagram.get('amplitudes'):
                res = res + "%sAMP(%d)" % (coeff(amplitude.get('fermionfactor'),
                                                 1, False, 0),
                                           amplitude.get('number'))
        return [res]
    else:
        res_list = []
        for i, col_basis_elem in \
                enumerate(matrix_element.get('color_basis').keys()):
            res = "JAMP(%i)=" % (i + 1)

            # Optimization: if all contributions to that color basis element have
            # the same coefficient (up to a sign), put it in front
            list_fracs = [abs(diag_tuple[2]) for diag_tuple in \
                          matrix_element.get('color_basis')[col_basis_elem]]
            common_factor = False
            diff_fracs = list(set(list_fracs))
            if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                common_factor = True
                global_factor = diff_fracs[0]
                res = res + '%s(' % coeff(1, global_factor, False, 0)


            for diag_tuple in matrix_element.get('color_basis')[col_basis_elem]:
                res_amp = filter(lambda amp: \
                                 tuple(amp.get('color_indices')) == diag_tuple[1],
                                 matrix_element.get('diagrams')[diag_tuple[0]].get('amplitudes'))
                if res_amp:
                    if len(res_amp) > 1:
                        raise FortranWriter.FortranWriterError, \
                            """More than one amplitude found for color structure
                            %s and color index chain (%s) (diagram %i)""" % \
                            (col_basis_elem,
                             str(diag_tuple[1]),
                             diag_tuple[0])
                    else:
                        if common_factor:
                            res = res + "%sAMP(%d)" % (coeff(res_amp[0].get('fermionfactor'),
                                                     diag_tuple[2] / abs(diag_tuple[2]),
                                                     diag_tuple[3],
                                                     diag_tuple[4]),
                                                     res_amp[0].get('number'))
                        else:
                            res = res + "%sAMP(%d)" % (coeff(res_amp[0].get('fermionfactor'),
                                                     diag_tuple[2],
                                                     diag_tuple[3],
                                                     diag_tuple[4]),
                                                     res_amp[0].get('number'))
                else:
                    raise FortranWriter.FortranWriterError, \
                            """No corresponding amplitude found for color structure
                            %s and color index chain (%s) (diagram %i)""" % \
                            (col_basis_elem,
                             str(diag_tuple[1]),
                             diag_tuple[0])

            if common_factor:
                res = res + ')'

            res_list.append(res)

        return res_list

#===============================================================================
# FortranWriter
#===============================================================================
class FortranWriter():
    """Routines for writing fortran lines. Keeps track of indentation
    and splitting of long lines"""

    class FortranWriterError(Exception):
        """Exception raised if an error occurs in the definition
        or the execution of a FortranWriter."""
        pass

    # Parameters defining the output of the Fortran writer
    keyword_pairs = {'^if.+then\s*$': ('^endif', 2),
                     '^do': ('^enddo\s*$', 2),
                     '^subroutine': ('^end\s*$', 0),
                     'function': ('^end\s*$', 0)}
    single_indents = {'^else\s*$':-2,
                      '^else\s*if.+then\s*$':-2}
    line_cont_char = '$'
    comment_char = 'c'
    downcase = False
    line_length = 71
    max_split = 10
    split_characters = "+-*/,) "
    comment_split_characters = " "

    # Private variables
    __indent = 0
    __keyword_list = []
    __comment_pattern = re.compile(r"^(\s*#|c$|(c\s+([^=]|$)))", re.IGNORECASE)

    def write_fortran_line(self, fsock, line):
        """Write a fortran line, with correct indent and line splits"""

        if not isinstance(line, str) or line.find('\n') >= 0:
            raise self.FortranWriterError, \
                  "write_fortran_line must have a single line as argument"

        # Check if this line is a comment
        comment = False
        if self.__comment_pattern.search(line):
            # This is a comment
            myline = " " * (5 + self.__indent) + line.lstrip()[1:].lstrip()
            if self.downcase:
                self.comment_char = self.comment_char.lower()
            else:
                self.comment_char = self.comment_char.upper()
            myline = self.comment_char + myline
            part = ""
            post_comment = ""
            # Break line in appropriate places
            # defined (in priority order) by the characters in
            # comment_split_characters
            res = self.split_line(myline,
                                  self.comment_split_characters,
                                  self.comment_char + " " * (5 + self.__indent))
        else:
            # This is a regular Fortran line

            # Strip leading spaces from line
            myline = line.lstrip()

            # Convert to upper or lower case
            # Here we need to make exception for anything within quotes.
            (myline, part, post_comment) = myline.partition("!")
            # Replace all double quotes by single quotes
            myline = myline.replace('\"', '\'')
            # Downcase or upcase Fortran code, except for quotes
            splitline = myline.split('\'')
            myline = ""
            i = 0
            while i < len(splitline):
                if i % 2 == 1:
                    # This is a quote - check for escaped \'s
                    while splitline[i][len(splitline[i]) - 1] == '\\':
                        splitline[i] = splitline[i] + '\'' + splitline.pop(i + 1)
                else:
                    # Otherwise downcase/upcase
                    if self.downcase:
                        splitline[i] = splitline[i].lower()
                    else:
                        splitline[i] = splitline[i].upper()
                i = i + 1

            myline = "\'".join(splitline).rstrip()

            # Check if line starts with dual keyword and adjust indent 
            if self.__keyword_list and re.search(self.keyword_pairs[\
                self.__keyword_list[len(self.__keyword_list) - 1]][0],
                                               myline.lower()):
                key = self.__keyword_list.pop()
                self.__indent = self.__indent - self.keyword_pairs[key][1]

            # Check for else and else if
            single_indent = 0
            for key in self.single_indents.keys():
                if re.search(key, myline.lower()):
                    self.__indent = self.__indent + self.single_indents[key]
                    single_indent = -self.single_indents[key]
                    break

            # Break line in appropriate places
            # defined (in priority order) by the characters in split_characters
            res = self.split_line(" " * (6 + self.__indent) + myline,
                                  self.split_characters,
                                  " " * 5 + self.line_cont_char + \
                                  " " * (self.__indent + 1))

            # Check if line starts with keyword and adjust indent for next line
            for key in self.keyword_pairs.keys():
                if re.search(key, myline.lower()):
                    self.__keyword_list.append(key)
                    self.__indent = self.__indent + self.keyword_pairs[key][1]
                    break

            # Correct back for else and else if
            if single_indent != None:
                self.__indent = self.__indent + single_indent
                single_indent = None

        # Write line(s) to file
        fsock.write("\n".join(res) + part + post_comment + "\n")

        return True

    def split_line(self, line, split_characters, line_start):
        """Split a line if it is longer than self.line_length
        columns. Split in preferential order according to
        split_characters, and start each new line with line_start."""

        res_lines = [line]

        while len(res_lines[-1]) > self.line_length:
            split_at = self.line_length
            for character in split_characters:
                index = res_lines[-1][(self.line_length - self.max_split): \
                                      self.line_length].rfind(character)
                if index >= 0:
                    split_at = self.line_length - self.max_split + index
                    break

            res_lines.append(line_start + \
                             res_lines[-1][split_at:])
            res_lines[-2] = res_lines[-2][:split_at]

        return res_lines

#===============================================================================
# HelasFortranModel
#===============================================================================
class HelasFortranModel(helas_objects.HelasModel):
    """The class for writing Helas calls in Fortran, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the Fortran Helas call based on the Lorentz structure of
    the interaction."""

    # Dictionaries used for automatic generation of Helas calls
    # Dictionaries from spin states to letters in Helas call
    mother_dict = {1: 'S', 2: 'O', -2: 'I', 3: 'V', 5: 'T'}
    self_dict = {1: 'H', 2: 'F', -2: 'F', 3: 'J', 5: 'U'}
    # Dictionaries used for sorting the letters in the Helas call
    sort_wf = {'O': 0, 'I': 1, 'S': 2, 'T': 3, 'V': 4}
    sort_amp = {'S': 1, 'V': 2, 'T': 0, 'O': 3, 'I': 4}


    def default_setup(self):
        """Set up special Helas calls (wavefunctions and amplitudes)
        that can not be done automatically by generate_helas_call"""

        super(HelasFortranModel, self).default_setup()

        # Add special fortran Helas calls, which are not automatically
        # generated

        # Gluon 4-vertex division tensor calls ggT for the FR sm and mssm

        key = ((3, 3, 5), 'A')

        call = lambda wf: \
               "CALL UVVAXX(W(1,%d),W(1,%d),%s,zero,zero,zero,W(1,%d))" % \
               (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                HelasFortranModel.sorted_mothers(wf)[1].get('number'),

                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 5, 3), 'A')

        call = lambda wf: \
               "CALL JVTAXX(W(1,%d),W(1,%d),%s,zero,zero,W(1,%d))" % \
               (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                HelasFortranModel.sorted_mothers(wf)[1].get('number'),

                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)

        key = ((3, 3, 5), 'A')

        call = lambda amp: \
               "CALL VVTAXX(W(1,%d),W(1,%d),W(1,%d),%s,zero,AMP(%d))" % \
               (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                HelasFortranModel.sorted_mothers(amp)[2].get('number'),

                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)

        # SM gluon 4-vertex components

        key = ((3, 3, 3, 3), 'gggg1')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                HelasFortranModel.sorted_mothers(wf)[2].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg1')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                HelasFortranModel.sorted_mothers(amp)[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3), 'gggg2')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (HelasFortranModel.sorted_mothers(wf)[2].get('number'),
                HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg2')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                HelasFortranModel.sorted_mothers(amp)[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)
        key = ((3, 3, 3, 3), 'gggg3')
        call = lambda wf: \
               "CALL JGGGXX(W(1,%d),W(1,%d),W(1,%d),%s,W(1,%d))" % \
               (HelasFortranModel.sorted_mothers(wf)[1].get('number'),
                HelasFortranModel.sorted_mothers(wf)[2].get('number'),
                HelasFortranModel.sorted_mothers(wf)[0].get('number'),
                wf.get('coupling'),
                wf.get('number'))
        self.add_wavefunction(key, call)
        key = ((3, 3, 3, 3), 'gggg3')
        call = lambda amp: \
               "CALL GGGGXX(W(1,%d),W(1,%d),W(1,%d),W(1,%d),%s,AMP(%d))" % \
               (HelasFortranModel.sorted_mothers(amp)[1].get('number'),
                HelasFortranModel.sorted_mothers(amp)[2].get('number'),
                HelasFortranModel.sorted_mothers(amp)[0].get('number'),
                HelasFortranModel.sorted_mothers(amp)[3].get('number'),
                amp.get('coupling'),
                amp.get('number'))
        self.add_amplitude(key, call)

    def get_wavefunction_call(self, wavefunction):
        """Return the function for writing the wavefunction
        corresponding to the key. If the function doesn't exist,
        generate_helas_call is called to automatically create the
        function."""

        val = super(HelasFortranModel, self).get_wavefunction_call(wavefunction)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(wavefunction.get('mothers')) > 3:
            raise self.PhysicsObjectError, \
                  """Automatic generation of Fortran wavefunctions not
                  implemented for > 3 mothers"""

        self.generate_helas_call(wavefunction)
        return super(HelasFortranModel, self).get_wavefunction_call(\
            wavefunction)

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude corresponding
        to the key. If the function doesn't exist, generate_helas_call
        is called to automatically create the function."""

        val = super(HelasFortranModel, self).get_amplitude_call(amplitude)

        if val:
            return val

        # If function not already existing, try to generate it.

        if len(amplitude.get('mothers')) > 4:
            raise self.PhysicsObjectError, \
                  """Automatic generation of Fortran amplitudes not
                  implemented for > 4 mothers"""

        self.generate_helas_call(amplitude)
        return super(HelasFortranModel, self).get_amplitude_call(amplitude)

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

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = call + HelasFortranModel.mother_dict[\
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
                       HelasFortranModel.self_dict[\
                argument.get_spin_state_number()]

            mother_letters = HelasFortranModel.sorted_letters(argument)

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
                                    (HelasFortranModel.sorted_mothers(wf)[0].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[1].\
                                     get('number'),
                                     wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
                else:
                    call_function = lambda wf: call % \
                                    (HelasFortranModel.sorted_mothers(wf)[0].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[1].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(wf)[2].\
                                     get('number'),
                                     wf.get_with_flow('coupling'),
                                     wf.get('mass'),
                                     wf.get('width'),
                                     wf.get('number'))
            else:
                # Create call for amplitude
                if len(argument.get('mothers')) == 3:
                    call_function = lambda amp: call % \
                                    (HelasFortranModel.sorted_mothers(amp)[0].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[1].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[2].\
                                     get('number'),

                                     amp.get('coupling'),
                                     amp.get('number'))
                else:
                    call_function = lambda amp: call % \
                                    (HelasFortranModel.sorted_mothers(amp)[0].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[1].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[2].\
                                     get('number'),
                                     HelasFortranModel.sorted_mothers(amp)[3].\
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
    def sorted_mothers(arg):
        """Gives a list of mother wavefunctions sorted according to
        1. the spin order needed in the Fortran Helas calls and
        2. the order of the particles in the interaction (cyclic)
        (3. the number for the external leg)"""

        if isinstance(arg, helas_objects.HelasWavefunction) or \
           isinstance(arg, helas_objects.HelasAmplitude):
            # First sort according to number_external number
            #sorted_mothers1 = sorted(arg.get('mothers'),
            #                         lambda wf1, wf2: \
            #                         wf1.get('number_external') - \
            #                         wf2.get('number_external'))
            sorted_mothers1 = copy.copy(arg.get('mothers'))

            # Next sort according to interaction pdg codes
            mother_codes = [ wf.get_pdg_code_outgoing() for wf \
                             in sorted_mothers1 ]
            pdg_codes = copy.copy(arg.get('pdg_codes'))
            if isinstance(arg, helas_objects.HelasWavefunction):
                my_code = arg.get_pdg_code_incoming()
                # We need to create the cyclic pdg_codes
                missing_index = pdg_codes.index(my_code)
                pdg_codes_cycl = pdg_codes[missing_index + 1:]
                pdg_codes_cycl.extend(pdg_codes[:missing_index])
            else:
                pdg_codes_cycl = pdg_codes

            sorted_mothers2 = helas_objects.HelasWavefunctionList()
            for code in pdg_codes_cycl:
                index = mother_codes.index(code)
                mother_codes.pop(index)
                sorted_mothers2.append(sorted_mothers1.pop(index))

            if sorted_mothers1:
                raise HelasFortranModel.PhysicsObjectError, \
                          "Mismatch of pdg codes, %s != %s" % \
                          (repr(mother_codes), repr(pdg_codes_cycl))

            # Next sort according to spin_state_number
            return sorted(sorted_mothers2, lambda wf1, wf2: \
                          HelasFortranModel.sort_amp[\
                          HelasFortranModel.mother_dict[wf2.\
                                                    get_spin_state_number()]]\
                          - HelasFortranModel.sort_amp[\
                          HelasFortranModel.mother_dict[wf1.\
                                                    get_spin_state_number()]])

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

def coeff(ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
    """Returns a nicely formatted string for the coefficients in JAMP lines"""

    total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

    if total_coeff == 1:
        if is_imaginary:
            return '+complex(0,1)*'
        else:
            return '+'
    elif total_coeff == -1:
        if is_imaginary:
            return '-complex(0,1)*'
        else:
            return '-'

    res_str = '%+i' % total_coeff.numerator

    if total_coeff.denominator != 1:
        # Check if total_coeff is an integer
        res_str = res_str + './%i.' % total_coeff.denominator

    if is_imaginary:
        res_str = res_str + '*complex(0,1)'

    return res_str + '*'





