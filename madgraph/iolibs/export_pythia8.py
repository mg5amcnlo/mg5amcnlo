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

import fractions
import glob
import itertools
import logging
import os
import re
import shutil
import subprocess

import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.import_ufo as import_ufo
import madgraph.iolibs.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as Template
import madgraph.iolibs.ufo_expression_parsers as parsers
from madgraph import MadGraph5Error, MG5DIR

import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_pythia8')

#===============================================================================
# generate_process_files_pythia8
#===============================================================================
def generate_process_files_pythia8(matrix_element,
                                   cpp_model,
                                   path=os.getcwd()):

    """Generate the .h and .cc files needed for Pythia 8, for the
    process described by matrix_element"""

    cwd = os.getcwd()

    os.chdir(path)

    pathdir = os.getcwd()

    process_file_name = get_process_file_name(matrix_element)

    logger.info('Creating files %s.h and %s.cc in directory %s' % \
                (process_file_name, process_file_name, path))

    # Create the files
    filename = '%s.h' % process_file_name
    color_amplitudes = \
                     write_pythia8_process_h_file(writers.CPPWriter(filename),
                                                  matrix_element)

    filename = '%s.cc' % process_file_name
    write_pythia8_process_cc_file(writers.CPPWriter(filename),
                                  matrix_element,
                                  cpp_model,
                                  color_amplitudes)

    os.chdir(cwd)

#===============================================================================
# write_pythia8_process_h_file
#===============================================================================
def write_pythia8_process_h_file(writer, matrix_element):
    """Write the class definition (.h) file for the process described
    by matrix_element"""

    if not matrix_element.get('processes') or \
           not matrix_element.get('diagrams'):
        return 0

    if not isinstance(writer, writers.CPPWriter):
        raise writers.CPPWriter.CPPWriterError(\
            "writer not CPPWriter")

    replace_dict = {}

    # Extract version number and date from VERSION file
    info_lines = get_mg5_info_lines()
    replace_dict['info_lines'] = info_lines

    # Extract model name
    replace_dict['model_name'] = \
                     matrix_element.get('processes')[0].get('model').get('name')

    # Extract process file name
    process_file_name = get_process_file_name(matrix_element)
    replace_dict['process_file_name'] = process_file_name

    # Extract class definitions
    process_class_definitions, color_amplitudes = \
                               get_process_class_definitions(matrix_element)
    replace_dict['process_class_definitions'] = process_class_definitions

    file = read_template_file('pythia8_process_h.inc') % replace_dict

    # Write the file
    writer.writelines(file)

    return color_amplitudes

#===============================================================================
# write_pythia8_process_cc_file
#===============================================================================
def write_pythia8_process_cc_file(writer, matrix_element, cpp_helas_writer,
                                  color_amplitudes):
    """Write the class member definition (.cc) file for the process
    described by matrix_element"""

    if not matrix_element.get('processes') or \
           not matrix_element.get('diagrams'):
        return 0

    if not isinstance(writer, writers.CPPWriter):
        raise writers.CPPWriter.CPPWriterError(\
            "writer not CPPWriter")

    replace_dict = {}

    # Extract version number and date from VERSION file
    info_lines = get_mg5_info_lines()
    replace_dict['info_lines'] = info_lines

    # Extract process file name
    process_file_name = get_process_file_name(matrix_element)
    replace_dict['process_file_name'] = process_file_name

    # Extract model name
    replace_dict['model_name'] = \
                     matrix_element.get('processes')[0].get('model').get('name')

    # Extract class function definitions
    process_function_definitions = \
                              get_process_function_definitions(matrix_element,
                                                               cpp_helas_writer,
                                                               color_amplitudes)
    replace_dict['process_function_definitions'] = process_function_definitions

    file = read_template_file('pythia8_process_cc.inc') % replace_dict

    # Write the file
    writer.writelines(file)

#===============================================================================
# Process export helper functions
#===============================================================================
def get_process_class_definitions(matrix_element):
    """The complete Pythia 8 class definition for the process"""

    replace_dict = {}

    # Extract model name
    replace_dict['model_name'] = \
                     matrix_element.get('processes')[0].get('model').get('name')

    # Extract process info lines
    process_lines = get_process_info_lines(matrix_element)
    replace_dict['process_lines'] = process_lines

    # Extract number of external particles
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    replace_dict['nfinal'] = nexternal - ninitial

    # Extract process class name (for the moment same as file name)
    process_class_name = get_process_file_name(matrix_element)
    replace_dict['process_class_name'] = process_class_name
    
    # Extract process definition
    process_definition = get_process_definition(matrix_element)
    replace_dict['process_definition'] = process_definition

    process = matrix_element.get('processes')[0]
    replace_dict['process_code'] = 10000 + \
                                   process.get('id')
    
    replace_dict['inFlux'] = get_process_influx(matrix_element)

    replace_dict['id_masses'] = get_id_masses(process)

    replace_dict['nexternal'] = nexternal
    # Get color amplitudes
    color_amplitudes = matrix_element.get_color_amplitudes()
    replace_dict['ncolor'] = len(color_amplitudes)

    replace_dict['process_variables'] = get_process_variables(matrix_element)
    
    file = read_template_file('pythia8_process_class.inc') % replace_dict

    return file, color_amplitudes

def get_process_function_definitions(matrix_element, cpp_helas_writer,
                                     color_amplitudes):
    """The complete Pythia 8 class definition for the process"""

    replace_dict = {}

    # Extract model name
    replace_dict['model_name'] = \
                     matrix_element.get('processes')[0].get('model').get('name')

    # Extract process info lines
    replace_dict['process_lines'] = \
                                  get_process_info_lines(matrix_element)

    # Extract process class name (for the moment same as file name)
    process_class_name = get_process_file_name(matrix_element)
    replace_dict['process_class_name'] = process_class_name
                                       

    replace_dict['initProc_lines'] = \
                                   get_initProc_lines(matrix_element)
    
    replace_dict['sigmaKin_lines'] = \
                                   get_sigmaKin_lines(matrix_element)
    
    replace_dict['sigmaHat_lines'] = \
                                   get_sigmaHat_lines(matrix_element)
    
    replace_dict['setIdColAcol_lines'] = \
                                       get_setIdColAcol_lines(matrix_element)
    
    replace_dict['weightDecay_lines'] = \
                                      get_weightDecay_lines(matrix_element)    

    replace_dict['matrix_lines'] = \
                                   get_matrix_lines(matrix_element,
                                                    cpp_helas_writer,
                                                    color_amplitudes)
    
    replace_dict['fixed_parameter_lines'] = \
                                   get_fixed_parameter_lines(matrix_element)
    
    replace_dict['variable_parameter_lines'] = \
                                   get_variable_parameter_lines(matrix_element)
    
    file = read_template_file('pythia8_process_function_definitions.inc') % \
           replace_dict

    return file

def get_mg5_info_lines():
    """Return info lines for MG5, suitable to place at beginning of
    Fortran files"""

    info = misc.get_pkg_info()
    info_lines = ""
    if info and info.has_key('version') and  info.has_key('date'):
        info_lines = "#  by MadGraph 5 v. %s, %s\n" % \
                     (info['version'], info['date'])
        info_lines = info_lines + \
                     "#  By the MadGraph Development Team\n" + \
                     "#  Please visit us at https://launchpad.net/madgraph5"
    else:
        info_lines = "#  by MadGraph 5\n" + \
                     "#  By the MadGraph Development Team\n" + \
                     "#  Please visit us at https://launchpad.net/madgraph5"        

    return info_lines

def get_process_file_name(matrix_element):
    """Return process file name for the process in matrix_element"""

    if not matrix_element.get('processes'):
        raise MadGraph5Error('Matrix element has no processes')

    return "Sigma_%s" % matrix_element.get('processes')[0].shell_string()

def get_process_info_lines(matrix_element):
    """Return info lines describing the processes for this matrix element"""

    return"\n".join([ "# " + process.nice_string().replace('\n', '\n# * ') \
                     for process in matrix_element.get('processes')])


def get_process_definition(matrix_element):
    """Return process file name for the process in matrix_element"""

    if not matrix_element.get('processes'):
        raise MadGraph5Error('Matrix element has no processes')
    
    return "%s (%s)" % \
           (matrix_element.get('processes')[0].nice_string().\
            replace("Process: ", ""),
            matrix_element.get('processes')[0].get('model').get('name'))

def get_process_influx(matrix_element):
    """Return process file name for the process in matrix_element"""

    if not matrix_element.get('processes'):
        raise MadGraph5Error('Matrix element has no processes')
    
    # Create a set with the pairs of incoming partons in definite order,
    # e.g.,  g g >... u d > ... d~ u > ... gives ([21,21], [1,2], [-2,1])
    beams = set([tuple(sorted([process.get('legs')[0].get('id'),
                               process.get('legs')[1].get('id')])) \
                      for process in matrix_element.get('processes')])

    # Define a number of useful sets
    antiquarks = range(-1, -6, -1)
    quarks = range(1,6)
    antileptons = range(-11, -17, -1)
    leptons = range(11, 17, 1)
    allquarks = antiquarks + quarks
    antifermions = antiquarks + antileptons
    fermions = quarks + leptons
    allfermions = allquarks + antileptons + leptons
    downfermions = range(-2, -5, -2) + range(-1, -5, -2) + \
                   range(-12, -17, -2) + range(-11, -17, -2) 
    upfermions = range(1, 5, 2) + range(2, 5, 2) + \
                 range(11, 17, 2) + range(12, 17, 2)

    # The following gives a list from flavor combinations to "inFlux" values
    # allowed by Pythia8, see Pythia 8 document SemiInternalProcesses.html
    set_tuples = [(set([(21, 21)]), "gg"),
                  (set(list(itertools.product(allquarks, [21]))), "qg"),
                  (set(zip(antiquarks, quarks)), "qqbarSame"),
                  (set(list(itertools.product(allquarks,
                                                   allquarks))), "qq"),
                  (set(zip(antifermions, fermions)),"ffbarSame"),
                  (set(zip(downfermions, upfermions)),"ffbarChg"),
                  (set(list(itertools.product(allfermions,
                                                   allfermions))), "ff"),
                  (set(list(itertools.product(allfermions, [22]))), "fgm"),
                  (set([(21, 22)]), "ggm"),
                  (set([(22, 22)]), "gmgm")]

    for set_tuple in set_tuples:
        if beams.issubset(set_tuple[0]):
            return set_tuple[1]

    raise MadGraph5Error('Pythia 8 cannot handle incoming flavors %s' %\
                         repr(beams))

    return 

def get_id_masses(process):
    """Return the lines which define the ids for the final state particles,
    to allow Pythia to know the masses"""

    mass_string = ""
    
    for i in range(2, len(process.get('legs'))):
        mass_string += "int id%dMass() const {return %d;}\n" % \
                       (i + 1, abs(process.get('legs')[i].get('id')))

    return mass_string

def get_initProc_lines(matrix_element):
    """Get initProc_lines for function definition for Pythia 8 .cc file"""

    initProc_lines = ""
    
    return initProc_lines

def get_sigmaKin_lines(matrix_element):
    """Get sigmaKin_lines for function definition for Pythia 8 .cc file"""

    replace_dict = {}

    # Number of helicity combinations
    replace_dict['ncomb'] = matrix_element.get_helicity_combinations()

    # Extract helicity matrix
    replace_dict['helicity_matrix'] = \
                                  get_helicity_matrix(matrix_element)
    # Extract denominator
    replace_dict['den_factor_line'] = \
                                  get_den_factor_line(matrix_element)

    file = read_template_file('pythia8_process_sigmaKin_function.inc') % \
           replace_dict

    return file

def get_matrix_lines(matrix_element, cpp_helas_writer, color_amplitudes):
    """Get sigmaKin_lines for function definition for Pythia 8 .cc file"""

    replace_dict = {}

    # Number of wavefunctions and amplitudes
    replace_dict['nwavefuncs'] = matrix_element.get_number_of_wavefunctions()

    replace_dict['ngraphs'] = matrix_element.get_number_of_amplitudes()

    # Extract color matrix
    replace_dict['color_matrix_lines'] = \
                               get_color_matrix_lines(matrix_element)

    # The Helicity amplitude calls
    replace_dict['amplitude_calls'] = "\n".join(\
        cpp_helas_writer.get_matrix_element_calls(matrix_element))
    replace_dict['jamp_lines'] = "\n".join(get_jamp_lines(matrix_element,
                                                          color_amplitudes))

    file = read_template_file('pythia8_process_matrix.inc') % \
           replace_dict

    return file


def get_sigmaHat_lines(matrix_element):
    """Get sigmaHat_lines for function definition for Pythia 8 .cc file"""

    sigmaHat_lines = "// Already calculated matrix_element in sigmaKin\n"
    sigmaHat_lines += "return matrix_element;"

    return sigmaHat_lines


def get_setIdColAcol_lines(matrix_element):
    """Generate lines to set final-state id and color info for process"""

    res_lines = []
    
    (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
    nfinal = nexternal - ninitial

    # Create a set with the pairs of incoming partons
    beams = set([(process.get('legs')[0].get('id'),
                  process.get('legs')[1].get('id')) \
                 for process in matrix_element.get('processes')])
    id_dict = {}
    for beam_parts in beams:
        # Pick out final state id:s for all processes that have these
        # initial state particles
        id_dict[beam_parts] = \
                 [[leg.get('id') for leg in process.get('legs')[2:]] \
                  for process in filter(lambda process: beam_parts == \
                                        (process.get('legs')[0].get('id'),
                                         process.get('legs')[1].get('id')),
                                        matrix_element.get('processes'))]

    # Now write a selection routine for final state ids
    for ibeam, beam_parts in enumerate(beams):
        if ibeam == 0:
            res_lines.append("if(id1 == %d && id2 == %d){" % beam_parts)
        else:
            res_lines.append("} else if(id1 == %d && id2 == %d){" % beam_parts)            
        final_id_list = id_dict[beam_parts]
        ncombs = len(final_id_list)
        res_lines.append("// Pick one of the flavor combinations %s" % \
                         repr(final_id_list))
        res_lines.append("int flavors[%d][%d] = {%s};" % \
                         (ncombs, nfinal,
                          ",".join(str(id) for id in sum(final_id_list, []))))
        res_lines.append("vector<double> probs(%d, 1./%d.);" % \
                         (ncombs, ncombs))
        res_lines.append("int choice = rndmPtr->pick(probs);")
        for i in range(nfinal):
            res_lines.append("id%d = flavors[choice][%d];" % (i+3, i))
    res_lines.append("}")
    res_lines.append("setId(%s);" % ",".join(["id%d" % i for i in \
                                              range(1,nexternal + 1)]))

    # Now write a selection routine for color flows

    # Here goes the color connections corresponding to the JAMPs
    # Only one output, for the first subproc!
    proc = matrix_element.get('processes')[0]
    if not matrix_element.get('color_basis'):
        # If no color basis, just output trivial color flow
        res_lines.append("setColAcol(%s);" % ",".join(["0"] * 2 * nfinal))
    else:
        # Else, build a color representation dictionnary
        repr_dict = {}
        legs = proc.get_legs_with_decays()
        for l in legs:
            repr_dict[l.get('number')] = \
                proc.get('model').get_particle(l.get('id')).get_color()
        # Get the list of color flows
        color_flow_list = \
            matrix_element.get('color_basis').color_flow_decomposition(\
                                              repr_dict, ninitial)
        # Select a color flow
        ncolor = len(matrix_element.get('color_basis'))
        res_lines.append("""vector<double> probs;
          double sum = %s;
          for(int i=0;i<ncolor;i++)
          probs.push_back(jamp2[i]/sum);
          int ic = rndmPtr->pick(probs);""" % \
                         "+".join(["jamp2[%d]" % i for i in range(ncolor)]))

        color_flows = []
        for color_flow_dict in color_flow_list:
            color_flows.append([color_flow_dict[l.get('number')][i] % 500 \
                                for (l,i) in itertools.product(legs, [0,1])])

        # Write out colors for the selected color flow
        res_lines.append("static int col[2][%d] = {%s};" % \
                         (2*nexternal,
                          ",".join(str(i) for i in sum(color_flows, []))))

        res_lines.append("setColAcol(%s);" % \
                         ",".join(["col[ic][%d]" % i for i in \
                                  range(2*nexternal)]))
        
    return "\n".join(res_lines)


def get_weightDecay_lines(matrix_element)    :
    """Get weightDecay_lines for function definition for Pythia 8 .cc file"""

    weightDecay_lines = "// Just use isotropic decay (default)\n"
    weightDecay_lines += "return 1.;"

    return weightDecay_lines

def get_helicity_matrix(matrix_element):
    """Return the Helicity matrix definition lines for this matrix element"""

    helicity_line = "static const int helicities[ncomb][nexternal] = {";
    helicity_line_list = []
    
    for helicities in matrix_element.get_helicity_matrix():
        helicity_line_list.append(",".join(['%d'] * len(helicities)) % \
                                  tuple(helicities))

    return helicity_line + ",".join(helicity_line_list) + "};"

def get_den_factor_line(matrix_element):
    """Return the denominator factor line for this matrix element"""

    return "const int denominator = %d;" % \
           matrix_element.get_denominator_factor()

def get_color_matrix_lines(matrix_element):
    """Return the color matrix definition lines for this matrix element. Split
    rows in chunks of size n."""

    if not matrix_element.get('color_matrix'):
        return ["static const double denom[1] = {1.};", "static const double cf[1][1] = {1.};"]
    else:
        color_denominators = matrix_element.get('color_matrix').\
                                             get_line_denominators()
        denom_string = "static const double denom[ncolor] = {%s};" % \
                       ",".join(["%i" % denom for denom in color_denominators])
                                                     
        matrix_strings = []
        my_cs = color.ColorString()
        for index, denominator in enumerate(color_denominators):
            # Then write the numerators for the matrix elements
            num_list = matrix_element.get('color_matrix').\
                                        get_line_numerators(index, denominator)

            matrix_strings.append("%s" % \
                                 ",".join(["%d" % i for i in num_list]))
        matrix_string = "static const double cf[ncolor][ncolor] = {" + \
                        ",".join(matrix_strings) + "};"
        return "\n".join([denom_string, matrix_string])

def get_jamp_lines(matrix_element, color_amplitudes):
    """Return the jamp = sum(fermionfactor * amp[i]) lines"""

    res_list = []

    for i, coeff_list in enumerate(color_amplitudes):

        res = "jamp[%i]=" % i

        # Optimization: if all contributions to that color basis element have
        # the same coefficient (up to a sign), put it in front
        list_fracs = [abs(coefficient[0][1]) for coefficient in coeff_list]
        common_factor = False
        diff_fracs = list(set(list_fracs))
        if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
            common_factor = True
            global_factor = diff_fracs[0]
            res = res + '%s(' % coeff(1, global_factor, False, 0)

        for (coefficient, amp_number) in coeff_list:
            if common_factor:
                res = res + "%samp[%d]" % (coeff(coefficient[0],
                                           coefficient[1] / abs(coefficient[1]),
                                           coefficient[2],
                                           coefficient[3]),
                                           amp_number - 1)
            else:
                res = res + "%samp[%d]" % (coeff(coefficient[0],
                                           coefficient[1],
                                           coefficient[2],
                                           coefficient[3]),
                                           amp_number - 1)

        if common_factor:
            res = res + ')'

        res += ';'

        res_list.append(res)

    return res_list

def get_process_variables(matrix_element):
    """Returns a string with all variables (masses, widths and
    couplings) used by the process"""

    variable_lines = []

    nexternal, ninitial = matrix_element.get_nexternal_ninitial()
    masses = set([wf.get('mass') for wf in \
                  matrix_element.get_all_wavefunctions()[nexternal:]])
    masses -= set(['zero', 'ZERO'])
    if masses:
        variable_lines.append("// Propagator masses")
        variable_lines.append("double %s;" % ", ".join(masses))

    widths = set([wf.get('width') for wf in \
                  matrix_element.get_all_wavefunctions()[nexternal:]])
    widths -= set(['zero', 'ZERO'])
    if widths:
        variable_lines.append("// Propagator widths")
        variable_lines.append("double %s;" % ", ".join(widths))

    couplings = set([wf.get('coupling') for wf in \
                     matrix_element.get_all_wavefunctions()[nexternal:] + \
                     matrix_element.get_all_amplitudes()])
    if couplings:
        variable_lines.append("// Couplings")
        variable_lines.append("complex %s;" % ", ".join(couplings))

    return "\n".join(variable_lines)

def get_fixed_parameter_lines(matrix_element):
    """Returns a string setting all fixed parameters (masses, widths
    and couplings)"""

    variable_lines = []

    nexternal, ninitial = matrix_element.get_nexternal_ninitial()
    mass_parts = set([(wf.get('pdg_code'), wf.get('mass'), wf.get('width')) \
                      for wf in filter(lambda wf: \
                                       wf.get('mass').lower() != 'zero',
                         matrix_element.get_all_wavefunctions()[nexternal:])])

    if mass_parts:
        variable_lines.append("// Propagator masses and widths")
        for part in mass_parts:
            variable_lines.append("%s = pd->m0(%d);" % \
                                  (part[1], part[0]))
            if part[2].lower() != 'zero':
                variable_lines.append("%s = pd->mWidth(%d);" % \
                                  (part[2], part[0]))

    couplings = set([wf.get('coupling') for wf in \
                     matrix_element.get_all_wavefunctions()[nexternal:] + \
                     matrix_element.get_all_amplitudes()])
    #if couplings:
    #    variable_lines.append("// Couplings")
    #    variable_lines.append("complex %s;" % ", ".join(couplings))

    return "\n".join(variable_lines)

def get_variable_parameter_lines(matrix_element):
    """Returns a string setting all parameters (couplings) that vary
    from event to event."""

    variable_lines = []
    nexternal, ninitial = matrix_element.get_nexternal_ninitial()
    couplings = set([wf.get('coupling') for wf in \
                     matrix_element.get_all_wavefunctions()[nexternal:] + \
                     matrix_element.get_all_amplitudes()])
    if couplings:
        variable_lines.append("// Couplings")
        for coupling in couplings:
            variable_lines.append("%s = %s;" % (coupling, "expression"))

    return "\n".join(variable_lines)

#===============================================================================
# Global helper methods
#===============================================================================

def read_template_file(filename):
    """Open a template file and return the contents."""

    return open(os.path.join(_file_path, \
                             'iolibs', 'template_files',
                             filename)).read()

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

#===============================================================================
# Routines to output UFO models in Pythia8 format
#===============================================================================

def convert_model_to_pythia8(model, output_dir):
    """Create a full valid Pythia 8 model from an MG5 model (coming from UFO)"""
    
    # create the model parameter files
    model_builder = UFO_model_to_pythia8(model, output_dir)
    model_builder.build()
    
#===============================================================================
# UFO_model_to_pythia8
#===============================================================================

class UFO_model_to_pythia8(object):
    """ A converter of the UFO-MG5 Model to the Pythia 8 format """
    
    # Dictionary from Python type to C++ type
    type_dict = {"real": "double",
                 "complex": "complex"}

    # Regular expressions for cleaning of lines from Aloha files
    compiler_option_re = re.compile('^#\w')
    namespace_re = re.compile('^using namespace')

    # Dictionaries for expression of MG5 SM parameters into Pythia 8
    slha_to_expr = {('SMINPUTS', (1,)): '1./csm->alphaEM(pow(pd->m0(23),2))',
                    ('SMINPUTS', (2,)): 'M_PI*csm->alphaEM(pow(pd->m0(23),2))*pow(pd->m0(23),2)/(sqrt(2.)*pow(pd->m0(24),2)*(pow(pd->m0(23),2)-pow(pd->m0(24),2)))',
                    ('SMINPUTS', (3,)): 'alpS',
                    ('CKMBLOCK', (1,)): 'csm->VCKMgen(1,2)',
                    }

    slha_to_depend = {('SMINPUTS', (3,)): ('aS',),
                      ('SMINPUTS', (1,)): ('aEM',)}

    def __init__(self, model, output_path):
        """ initialization of the objects """
        
        self.model = model
        self.model_name = model['name']
        
        self.dir_path = output_path
        
        # For dependent couplings, only want to update the ones
        # actually used in each process. For other couplings and
        # parameters, just need a list of all.
        self.coups_dep = {}    # name -> import_ufo.ParamExpr
        self.coups_indep = []  # import_ufo.ParamExpr
        self.params_dep = []   # import_ufo.ParamExpr
        self.params_indep = [] # import_ufo.ParamExpr
        self.p_to_cpp = parsers.UFOExpressionParserPythia8()

        # Prepare parameters and couplings for writeout in C++
        self.prepare_parameters()
        self.prepare_couplings()
        
    def write_files(self):
        """Modify the parameters to fit with Pythia8 conventions and
        creates all necessary files"""

        # Write parameter (and coupling) class files
        self.write_parameter_class_files()

        # Write Helas Routines
        self.write_aloha_routines()

    # Routines for preparing parameters and couplings from the model

    def prepare_parameters(self):
        """Extract the parameters from the model, and store them in
        the two lists params_indep and params_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = self.model['parameters'].keys()
        keys.sort(key=len)
        params_ext = []
        for key in keys:
            if key == ('external',):
                params_ext += self.model['parameters'][key]
            elif 'aS' in key:
                for p in self.model['parameters'][key]:
                    self.params_dep.append(import_ufo.ParamExpr(p.name,
                                                 self.p_to_cpp.parse(p.expr),
                                                 p.type,
                                                 p.depend))
            else:
                for p in self.model['parameters'][key]:
                    self.params_indep.append(import_ufo.ParamExpr(p.name,
                                                 self.p_to_cpp.parse(p.expr),
                                                 p.type,
                                                 p.depend))

        # For external parameters, want to use the internal Pythia
        # parameters for SM params and masses and widths. For other
        # parameters, want to read off the SLHA block code (TO BE
        # IMPLEMENTED)
        while params_ext:
            param = params_ext.pop(0)
            key = (param.lhablock, tuple(param.lhacode))
            if 'aS' in self.slha_to_depend.setdefault(key, ()):
                # This value needs to be set event by event
                self.params_dep.insert(0,
                                       import_ufo.ParamExpr(param.name,
                                                         self.slha_to_expr[key],
                                                         'real'))
            else:
                try:
                    # This is an SM parameter defined above
                    self.params_indep.insert(0,
                                             import_ufo.ParamExpr(param.name,
                                                         self.slha_to_expr[key],
                                                         'real'))
                except:
                    # For Yukawa couplings, masses and widths, insert
                    # the Pythia 8 value
                    if param.lhablock == 'YUKAWA':
                        self.slha_to_expr[key] = 'pd->mRun(%i, 120.)' \
                                                 % param.lhacode[0]
                    if param.lhablock == 'MASS':
                        self.slha_to_expr[key] = 'pd->m0(%i)' \
                                            % param.lhacode[0]
                    if param.lhablock == 'DECAY':
                        self.slha_to_expr[key] = 'pd->mWidth(%i)' \
                                            % param.lhacode[0]
                    if key in self.slha_to_expr:
                        self.params_indep.insert(0,\
                                                 import_ufo.ParamExpr(param.name,
                                                          self.slha_to_expr[key],
                                                          'real'))
                    else:
                        # Fix unknown parameters as soon as Pythia has fixed this
                        raise MadGraph5Error, \
                              "Parameter with key " + repr(key) + \
                              " unknown in model export to Pythia 8"

    def prepare_couplings(self):
        """Extract the couplings from the model, and store them in
        the two lists coups_indep and coups_dep"""


        # Keep only dependences on alphaS, to save time in execution
        keys = self.model['couplings'].keys()
        keys.sort(key=len)
        for key, coup_list in self.model['couplings'].items():
            if 'aS' in key:
                for c in coup_list:
                    self.coups_dep[c.name] = import_ufo.ParamExpr(c.name,
                                                 self.p_to_cpp.parse(c.expr),
                                                 c.type,
                                                 c.depend)
            else:
                for c in coup_list:
                    self.coups_indep.append(import_ufo.ParamExpr(c.name,
                                                 self.p_to_cpp.parse(c.expr),
                                                 c.type,
                                                 c.depend))
        
        # Convert coupling expressions from Python to C++
        for coup in self.coups_dep.values() + self.coups_indep:
            coup.expr = self.p_to_cpp.parse(coup.expr)

    # Routines for writing the parameter files

    def write_parameter_class_files(self):
        """Generate the parameters_model.h and parameters_model.cc
        files, which have the parameters and couplings for the model."""

        parameter_h_file = os.path.join(self.dir_path,
                                    'Parameters_%s.h' % self.model.get('name'))
        parameter_cc_file = os.path.join(self.dir_path,
                                     'Parameters_%s.cc' % self.model.get('name'))

        replace_dict = {}

        replace_dict['info_lines'] = get_mg5_info_lines()
        replace_dict['model_name'] = self.model.get('name')

        replace_dict['independent_parameters'] = \
                                   "// Model parameters independent of aS\n" + \
                                   self.write_parameters(self.params_indep)
        replace_dict['independent_couplings'] = \
                                   "// Model parameters dependent on aS\n" + \
                                   self.write_parameters(self.params_dep)
        replace_dict['dependent_parameters'] = \
                                   "// Model couplings independent of aS\n" + \
                                   self.write_parameters(self.coups_indep)
        replace_dict['dependent_couplings'] = \
                                   "// Model couplings dependent on aS\n" + \
                                   self.write_parameters(self.coups_dep.values())

        replace_dict['set_independent_parameters'] = \
                               self.write_set_parameters(self.params_indep)
        replace_dict['set_independent_couplings'] = \
                               self.write_set_parameters(self.coups_indep)
        replace_dict['set_dependent_parameters'] = \
                               self.write_set_parameters(self.params_dep)
        replace_dict['set_dependent_couplings'] = \
                               self.write_set_parameters(self.coups_dep.values())

        file_h = read_template_file('pythia8_model_parameters_h.inc') % \
                 replace_dict
        file_cc = read_template_file('pythia8_model_parameters_cc.inc') % \
                  replace_dict

        # Write the files
        writers.CPPWriter(parameter_h_file).writelines(file_h)
        writers.CPPWriter(parameter_cc_file).writelines(file_cc)

    def write_parameters(self, params):
        """Write out the definitions of parameters"""

        # Create a dictionary from parameter type to list of parameter names
        type_param_dict = {}

        for param in params:
            type_param_dict[param.type] = \
                  type_param_dict.setdefault(param.type, []) + [param.name]

        # For each parameter type, write out the definition string
        # type parameters;
        res_strings = []
        for key in type_param_dict:
            res_strings.append("%s %s;" % (self.type_dict[key],
                                          ",".join(type_param_dict[key])))

        return "\n".join(res_strings)

    def write_set_parameters(self, params):
        """Write out the lines of independent parameters"""

        # For each parameter, write name = expr;

        res_strings = []
        for param in params:
            res_strings.append("%s=%s;" % (param.name, param.expr))

        return "\n".join(res_strings)

    # Routines for writing the ALOHA files
    
    def write_aloha_routines(self):
        """Generate the hel_amps_model.h and hel_amps_model.cc files, which
        have the complete set of generalized Helas routines for the model"""

        model_h_file = os.path.join(self.dir_path,
                                    'hel_amps_%s.h' % self.model.get('name'))
        model_cc_file = os.path.join(self.dir_path,
                                     'hel_amps_%s.cc' % self.model.get('name'))

        replace_dict = {}

        replace_dict['info_lines'] = get_mg5_info_lines()
        replace_dict['model_name'] = self.model.get('name')

        # Read in the template .h and .cc files, stripped of compiler
        # commands and namespaces
        template_h_files = self.read_aloha_template_files(ext = 'h')
        template_cc_files = self.read_aloha_template_files(ext = 'cc')
        
        for abstracthelas in self.model.get('lorentz').values():
            aloha_writer = aloha_writers.ALOHAWriterForCPP(abstracthelas,
                                                        self.dir_path)
            header = aloha_writer.define_header()
            template_h_files.append(self.write_function_declaration(\
                                         aloha_writer, header))
            template_cc_files.append(self.write_function_definition(\
                                          aloha_writer, header))

        replace_dict['function_declarations'] = '\n'.join(template_h_files)
        replace_dict['function_definitions'] = '\n'.join(template_cc_files)

        file_h = read_template_file('pythia8_hel_amps_h.inc') % replace_dict
        file_cc = read_template_file('pythia8_hel_amps_cc.inc') % replace_dict

        # Write the files
        writers.CPPWriter(model_h_file).writelines(file_h)
        writers.CPPWriter(model_cc_file).writelines(file_cc)

    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        template_files = []
        for filename in glob.glob(os.path.join(MG5DIR, 'aloha', 'Template', '*.%s' % ext)):
            file = open(filename, 'r')
            template_file_string = ""
            while file:
                line = file.readline()
                if len(line) == 0: break
                line = self.clean_line(line)
                if not line:
                    continue
                template_file_string += line.strip() + '\n'
            template_files.append(template_file_string)

        return template_files

    def write_function_declaration(self, aloha_writer, header):
        """Write the function declaration for the ALOHA routine"""

        ret_lines = []
        for line in aloha_writer.write_h(header).split('\n'):
            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
                # Strip out compiler flags and namespaces
                continue
            ret_lines.append(line)
        return "\n".join(ret_lines)

    def write_function_definition(self, aloha_writer, header):
        """Write the function definition for the ALOHA routine"""

        ret_lines = []
        for line in aloha_writer.write_cc(header).split('\n'):
            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
                # Strip out compiler flags and namespaces
                continue
            ret_lines.append(line)
        return "\n".join(ret_lines)

    def clean_line(self, line):
        """Strip a line of compiler options and namespace options, and
        replace complex<double> by complex."""

        if self.compiler_option_re.match(line) or self.namespace_re.match(line):
            return ""

        return line
