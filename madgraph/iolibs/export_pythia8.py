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
import madgraph.iolibs.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as Template
import madgraph.iolibs.ufo_expression_parsers as parsers
from madgraph import MadGraph5Error, MG5DIR

import aloha.create_helas as create_helas
import aloha.WriteHelas as WriteHelas

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
def write_pythia8_process_cc_file(writer, matrix_element, cpp_model,
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

    # Extract class function definitions
    process_function_definitions = \
                              get_process_function_definitions(matrix_element,
                                                               cpp_model,
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

def get_process_function_definitions(matrix_element, cpp_model,
                                     color_amplitudes):
    """The complete Pythia 8 class definition for the process"""

    replace_dict = {}

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
                                                    cpp_model,
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

    initProc_lines = "// Set all parameters that are fixed once and for all\n"
    initProc_lines += "set_fixed_parameters();"

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

def get_matrix_lines(matrix_element, cpp_model, color_amplitudes):
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
        cpp_model.get_matrix_element_calls(matrix_element))
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
            variable_lines.append("%s = ParticleData::m0(%d);" % \
                                  (part[1], part[0]))
            if part[2].lower() != 'zero':
                variable_lines.append("%s = ParticleData::mWidth(%d);" % \
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
    
    python_to_fortran = parsers.UFOExpressionParserFortran().parse
    compiler_option_re = re.compile('^#\w')
    namespace_re = re.compile('^using namespace')

    def __init__(self, model, output_path):
        """ initialization of the objects """
        
        self.model = model
        self.model_name = model['name']
        
        self.dir_path = output_path
        
        self.coups_dep = []    # (name, expression, type)
        self.coups_indep = []  # (name, expression, type)
        self.params_dep = []   # (name, expression, type)
        self.params_indep = [] # (name, expression, type)
        self.params_ext = []   # external parameter
        
    def build(self):
        """modify the couplings to fit with MG4 convention and creates all the 
        different files"""

        # Write Helas Routines
        self.write_aloha_routines(model, output_dir)

        # Keep only separation in alphaS        
        keys = self.model['parameters'].keys()
        keys.sort(key=len)
        for key in keys:
            if key == ('external',):
                self.params_ext += self.model['parameters'][key]
            elif 'aS' in key:
                self.params_dep += self.model['parameters'][key]
            else:
                self.params_indep += self.model['parameters'][key]
        # same for couplings
        keys = self.model['couplings'].keys()
        keys.sort(key=len)
        for key, coup_list in self.model['couplings'].items():
            if 'aS' in key:
                self.coups_dep += coup_list
            else:
                self.coups_indep += coup_list
                
        # MG4 use G and not aS as it basic object for alphas related computation
        #Pass G in the  independant list
        index = self.params_dep.index('G')
        self.params_indep.insert(0, self.params_dep.pop(index))
        index = self.params_dep.index('sqrt__aS')
        self.params_indep.insert(0, self.params_dep.pop(index))

        # write the files
        self.write_all()

    def open(self, name, comment='c', format='default'):
        """ Open the file name in the correct directory and with a valid
        header."""
        
        file_path = os.path.join(self.dir_path, name)
        
        if format == 'fortran':
            fsock = writers.FortranWriter(file_path, 'w')
        else:
            fsock = open(file_path, 'w')
        
        file.writelines(fsock, comment * 77 + '\n')
        file.writelines(fsock,'%(comment)s written by the UFO converter\n' % \
                               {'comment': comment + (6 - len(comment)) *  ' '})
        file.writelines(fsock, comment * 77 + '\n\n')
        return fsock       

    
    def write_all(self):
        """ write all the files """
        #write the part related to the external parameter
        self.create_ident_card()
        self.create_param_read()
        
        #write the definition of the parameter
        self.create_input()
        self.create_intparam_def()
        
        
        # definition of the coupling.
        self.create_coupl_inc()
        self.create_write_couplings()
        self.create_couplings()
        
        # the makefile
        self.create_makeinc()
        self.create_param_write()
        
        # The param_card.dat        
        self.create_param_card()
        

        # All the standard files
        self.copy_standard_file()


    # Routines for creating the parameter files

    def copy_standard_file(self):
        """Copy the standard files for the fortran model."""
    
        
        #copy the library files
        file_to_link = ['formats.inc', 'lha_read.f', 'makefile','printout.f', \
                        'rw_para.f', 'testprog.f', 'rw_para.f']
    
        for filename in file_to_link:
            cp( MG5DIR + '/models/Template/fortran/' + filename, self.dir_path)

    def create_coupl_inc(self):
        """ write coupling.inc """
        
        fsock = self.open('coupl.inc', format='fortran')
        
        # Write header
        header = """double precision G
                common/strong/ G
                 
                double complex gal(2)
                common/weak/ gal

                double precision DUM0
                common/FRDUM0/ DUM0

                double precision DUM1
                common/FRDUM1/ DUM1
                """        
        fsock.writelines(header)
        
        # Write the Mass definition/ common block
        masses = [param.name for param in self.params_ext \
                                                    if param.lhablock == 'MASS']
        fsock.writelines('double precision '+','.join(masses)+'\n')
        fsock.writelines('common/masses/ '+','.join(masses)+'\n\n')
        
        # Write the Width definition/ common block
        widths = [param.name for param in self.params_ext \
                                                   if param.lhablock == 'DECAY']
        fsock.writelines('double precision '+','.join(widths)+'\n')
        fsock.writelines('common/widths/ '+','.join(widths)+'\n\n')
        
        # Write the Couplings
        coupling_list = [coupl.name for coupl in self.coups_dep + self.coups_indep]       
        fsock.writelines('double complex '+', '.join(coupling_list)+'\n')
        fsock.writelines('common/couplings/ '+', '.join(coupling_list)+'\n')
        
    def create_write_couplings(self):
        """ write the file coupl_write.inc """
        
        fsock = self.open('coupl_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' Couplings of %s'  
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""" % self.model_name)
        def format(coupl):
            return 'write(*,2) \'%(name)s = \', %(name)s' % {'name': coupl.name}
        
        # Write the Couplings
        lines = [format(coupl) for coupl in self.coups_dep + self.coups_indep]       
        fsock.writelines('\n'.join(lines))
        
        
    def create_input(self):
        """create input.inc containing the definition of the parameters"""
        
        fsock = self.open('input.inc', format='fortran')
        
        real_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'real'
                            and param.name != 'G']
        
        real_parameters += [param.name for param in self.params_ext 
                            if param.type == 'real'and 
                               param.lhablock not in ['MASS', 'DECAY']]
        
        fsock.writelines('double precision '+','.join(real_parameters)+'\n')
        fsock.writelines('common/params_R/ '+','.join(real_parameters)+'\n\n')
        
        complex_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'complex']


        fsock.writelines('double complex '+','.join(complex_parameters)+'\n')
        fsock.writelines('common/params_C/ '+','.join(complex_parameters)+'\n\n')
        
    def create_intparam_def(self):
        """ create intparam_definition.inc """

        fsock = self.open('intparam_definition.inc', format='fortran')
        
        fsock.write_comments(\
                "Parameters that should not be recomputed event by event.\n")
        fsock.writelines("if(readlha) then\n")
        
        for param in self.params_indep:
            fsock.writelines("%s = %s\n" % (param.name, self.python_to_fortran(param.expr) ))
        
        fsock.writelines('endif')
        
        fsock.write_comments('\nParameters that should be recomputed at an event by even basis.\n')
        for param in self.params_dep:
            fsock.writelines("%s = %s\n" % (param.name, self.python_to_fortran(param.expr) ))
           
        fsock.write_comments("\nDefinition of the EW coupling used in the write out of aqed\n")
        fsock.writelines(""" gal(1) = 1d0
                             gal(2) = 1d0
                         """)

        fsock.write_comments("\nDefinition of DUM symbols\n")
        fsock.writelines(""" DUM0 = 0
                             DUM1 = 1
                         """)
    
    def create_couplings(self):
        """ create couplings.f and all couplingsX.f """
        
        nb_def_by_file = 25
        
        self.create_couplings_main(nb_def_by_file)
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        for i in range(nb_coup_indep):
            data = self.coups_indep[nb_def_by_file * i: 
                             min(len(self.coups_indep), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1, data)
            
        for i in range(nb_coup_dep):
            data = self.coups_dep[nb_def_by_file * i: 
                               min(len(self.coups_dep), nb_def_by_file * (i+1))]
            self.create_couplings_part( i + 1 + nb_coup_indep , data)        
        
        
    def create_couplings_main(self, nb_def_by_file=25):
        """ create couplings.f """

        fsock = self.open('couplings.f', format='fortran')
        
        fsock.writelines("""subroutine coup(readlha)

                            implicit none
                            logical readlha
                            double precision PI
                            parameter  (PI=3.141592653589793d0)
                            
                            include \'input.inc\'
                            include \'coupl.inc\'
                            include \'intparam_definition.inc\'\n\n
                         """)
        
        nb_coup_indep = 1 + len(self.coups_indep) // nb_def_by_file 
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        fsock.writelines('if (readlha) then\n')
        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1) for i in range(nb_coup_indep)]))
        fsock.writelines('''\nendif\n''')
        
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')


    def create_couplings_part(self, nb_file, data):
        """ create couplings[nb_file].f containing information coming from data
        """
        
        fsock = self.open('couplings%s.f' % nb_file, format='fortran')
        fsock.writelines("""subroutine coup%s()
        
          implicit none
      
          include 'input.inc'
          include 'coupl.inc'
                        """ % nb_file)
        
        for coupling in data:            
            fsock.writelines('%s = %s' % (coupling.name, \
                                             self.python_to_fortran(coupling.expr)))
        fsock.writelines('end')


    def create_makeinc(self):
        """create makeinc.inc containing the file to compile """
        
        fsock = self.open('makeinc.inc', comment='#')
        text = 'MODEL = couplings.o lha_read.o printout.o rw_para.o '
        
        nb_coup_indep = 1 + len(self.coups_dep) // 25 
        nb_coup_dep = 1 + len(self.coups_indep) // 25
        text += ' '.join(['couplings%s.o' % (i+1) \
                                  for i in range(nb_coup_dep + nb_coup_indep) ])
        fsock.writelines(text)
        
    def create_param_write(self):
        """ create param_write """

        fsock = self.open('param_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' External Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")
        def format(name):
            return 'write(*,*) \'%(name)s = \', %(name)s' % {'name': name}
        
        # Write the external parameter
        lines = [format(param.name) for param in self.params_ext]       
        fsock.writelines('\n'.join(lines))        
        
        fsock.writelines("""write(*,*)  ' Internal Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")        
        lines = [format(data.name) for data in self.params_indep]
        fsock.writelines('\n'.join(lines))
        fsock.writelines("""write(*,*)  ' Internal Params evaluated point by point'
                            write(*,*)  ' ----------------------------------------'
                            write(*,*)  ' '""")         
        lines = [format(data.name) for data in self.params_dep]
        
        fsock.writelines('\n'.join(lines))                
        
 
    
    def create_ident_card(self):
        """ create the ident_card.dat """
    
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            colum = [parameter.lhablock] + \
                    [str(value) for value in parameter.lhacode] + \
                    [parameter.name]
            return ' '.join(colum)+'\n'
    
        fsock = self.open('ident_card.dat')
     
        external_param = [format(param) for param in self.params_ext]
        fsock.writelines('\n'.join(external_param))
        
    def create_param_read(self):    
        """create param_read"""
        
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            template = \
            """ call LHA_get_real(npara,param,value,'%(name)s',%(name)s,%(value)s)"""
            
            return template % {'name': parameter.name, \
                                    'value': self.python_to_fortran(str(parameter.value))}
        
        fsock = self.open('param_read.inc', format='fortran')
        external_param = [format(param) for param in self.params_ext]
        fsock.writelines('\n'.join(external_param))

    def create_param_card(self):
        """ create the param_card.dat """
        
        write_param_card.ParamCardWriter(
                os.path.join(self.dir_path, 'param_card.dat'),
                self.params_ext)


    # Routines for writing the ALOHA files
    
    def write_aloha_routines(self):
        """Generate the hel_amp_model.h and hel_amp_model.cc files, which
        have the complete set of generalized Helas routines for the model"""

        model_h_file = os.path.join(self.output_dir,
                                    'hel_amp_%s.h' % model.get('name'))
        model_cc_file = os.path.join(self.output_dir,
                                     'hel_amp_%s.cc' % model.get('name'))

        replace_dict_h = {}
        replace_dict_cc = {}

        # Read in the template .h and .cc files, stripped of compiler
        # commands and namespaces
        template_h_files = self.read_aloha_template_files(ext = 'h')
        template_cc_files = self.read_aloha_template_files(ext = 'cc')
        
        for abstracthelas in self.model.get('lorentz').values():
            aloha_writer = WriteHelas.HelasWriterForCPP(abstracthelas,
                                                        self.output_dir)
            template_h_files.append(self.write_function_declaration(\
                                         aloha_writer))
            template_cc_files.append(self.write_function_definition(\
                                          aloha_writer))

        file_h = read_template_file('hel_amp_model_h.inc') % replace_dict_h
        file_cc = read_template_file('hel_amp_model_cc.inc') % replace_dict_cc

        # Write the files
        writers.CPPWriter(model_h_file).writelines(file_h)
        writers.CPPWriter(model_cc_file).writelines(file_cc)

    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        template_files = []
        for filename in glob.glob(os.path.join(MG5DIR, 'aloha', 'Template', '*.%s' % ext)):
            print filename
            file = open(filename, 'r')
            template_file_string = ""
            while file:
                line = file.readline()
                if len(line) == 0: break
                if self.compiler_option_re.match(line) or self.namespace_re.match(line):
                    # Strip out compiler flags and namespaces
                    continue
                template_file_string += line.strip() + '\n'
            template_files.append(template_file_string)

        return template_files

    def write_function_declaration(self, aloha_writer):
        """Write the function declaration for the ALOHA routine"""

        ret_string = ""
        for line in aloha_writer.define_header()['headerfile']:
            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
                # Strip out compiler flags and namespaces
                continue
            ret_string += line
        return ret_string

    def write_function_definition(self, aloha_writer):
        """Write the function definition for the ALOHA routine"""

        ret_string = ""
        head = aloha_writer.define_header()['head']
        body = aloha_writer.define_expression()
        foot = aloha_writer.define_foot()
        out = head + body + foot
        for line in out:
            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
                # Strip out compiler flags and namespaces
                continue
            ret_string += line
        return ret_string

