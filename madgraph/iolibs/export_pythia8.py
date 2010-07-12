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
from madgraph import MadGraph5Error 

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
                (process_file_name, process_file_name, subprocdir))

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
# Helper functions
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
        res_lines.append("int choice = Rndm::pick(probs);")
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
          int ic = Rndm::pick(probs);""" % \
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
    masses -= set(['zero'])
    if masses:
        variable_lines.append("// Propagator masses")
        variable_lines.append("double %s;" % ", ".join(masses))

    widths = set([wf.get('width') for wf in \
                  matrix_element.get_all_wavefunctions()[nexternal:]])
    widths -= set(['zero'])
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
                      for wf in filter(lambda wf: wf.get('mass') != 'zero',
                         matrix_element.get_all_wavefunctions()[nexternal:])])

    if mass_parts:
        variable_lines.append("// Propagator masses and widths")
        for part in mass_parts:
            variable_lines.append("%s = ParticleData::m0(%d);" % \
                                  (part[1], part[0]))
            if part[2] != 'zero':
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
# UFOHelasCPPModel
#===============================================================================
class UFOHelasCPPModel(helas_objects.HelasModel):
    """The class for writing Helas calls in C++, starting from
    HelasWavefunctions and HelasAmplitudes.

    Includes the function generate_helas_call, which automatically
    generates the C++ Helas call based on the Lorentz structure of
    the interaction."""

    # Dictionaries from spin states to letters in Helas call
    mother_dict = {1: 's', 2: 'o', -2: 'i', 3: 'v', 5: 't'}

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

        val = super(UFOHelasCPPModel, self).get_wavefunction_call(wavefunction)
        if val:
            return val

        # If function not already existing, try to generate it.
        self.generate_helas_call(wavefunction)
        return super(UFOHelasCPPModel, self).get_wavefunction_call(\
            wavefunction)

    def get_amplitude_call(self, amplitude):
        """Return the function for writing the amplitude corresponding
        to the key. If the function doesn't exist, generate_helas_call
        is called to automatically create the function."""

        val = super(UFOHelasCPPModel, self).get_amplitude_call(amplitude)
        if val:
            return val
        
        # If function not already existing, try to generate it.
        self.generate_helas_call(amplitude)
        return super(UFOHelasCPPModel, self).get_amplitude_call(amplitude)

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
            call = call + UFOHelasCPPModel.mother_dict[\
                argument.get_spin_state_number()]
            # Fill out with X up to 6 positions
            call = call + 'x' * (6 - len(call))
            call = call + "(pME[%d],"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                call = call + "mME[%d],hel[%d],"
            call = call + "%+d,w[%d]));"
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

    # Static helper functions

    @staticmethod
    def sorted_letters(arg):
        """Gives a list of letters sorted according to
        the order of letters in the C++ Helas calls"""

        def convert_bool(logical):
                if logical: 
                    return '1'
                else:
                    return '0'
                
        if isinstance(arg, helas_objects.HelasWavefunction):
            print [dir(wf) for wf in arg.get('mothers')]        
            return "".join([convert_bool(wf['onshell']) for wf in arg.get('mothers')])

        if isinstance(arg, helas_objects.HelasAmplitude):
            print 'amplitude'
            print [dir(wf) for wf in arg.get('mothers')] 
            return "".join(['1' for wf in arg.get('mothers')])


