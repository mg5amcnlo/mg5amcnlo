################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Methods and classes to export models and matrix elements to Pythia 8
and C++ Standalone format."""

from __future__ import absolute_import
import fractions
import glob
import itertools
import logging
from math import fmod
import os
import re
import shutil
import subprocess

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.various.banner as banner_mod
from madgraph import MadGraph5Error, InvalidCmd, MG5DIR
from madgraph.iolibs.files import cp, ln, mv

from madgraph.iolibs.export_v4 import VirtualExporter, ProcessExporterFortran
import madgraph.various.misc as misc

import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
from six.moves import range
from six.moves import zip

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_pythia8')
pjoin = os.path.join


def make_model_cpp(dir_path):
    """Make the model library in a C++ standalone directory"""

    source_dir = os.path.join(dir_path, "src")
    # Run standalone
    logger.info("Running make for src")
    misc.compile(cwd=source_dir)


#===============================================================================
# UFOModelConverterCPP
#===============================================================================

class UFOModelConverterCPP(object):
    """ A converter of the UFO-MG5 Model to the C++ format """

    # Static variables (for inheritance)
    output_name = 'C++ Standalone'
    namespace = 'MG5'
    aloha_writer = 'CPP'
    cc_ext = 'cc'

    # Dictionary from Python type to C++ type
    type_dict = {"real": "double",
                 "complex": "std::complex<double>"}

    # Regular expressions for cleaning of lines from Aloha files
    compiler_option_re = re.compile('^#\w')
    namespace_re = re.compile('^using namespace')

    slha_to_depend = {('SMINPUTS', (3,)): ('aS',),
                      ('SMINPUTS', (1,)): ('aEM',)}

    # Template files to use
    include_dir = '.'
    cc_file_dir = '.'
    param_template_h = 'cpp_model_parameters_h.inc'
    param_template_cc = 'cpp_model_parameters_cc.inc'
    aloha_template_h = 'cpp_hel_amps_h.inc'
    aloha_template_cc = 'cpp_hel_amps_cc.inc'

    copy_include_files = []
    copy_cc_files = []

    def __init__(self, model, output_path, wanted_lorentz = [],
                 wanted_couplings = [], replace_dict={}):
        """ initialization of the objects """

        self.model = model
        self.model_name = ProcessExporterCPP.get_model_name(model['name'])

        self.dir_path = output_path
        self.default_replace_dict = dict(replace_dict)
        # List of needed ALOHA routines
        self.wanted_lorentz = wanted_lorentz

        # For dependent couplings, only want to update the ones
        # actually used in each process. For other couplings and
        # parameters, just need a list of all.
        self.coups_dep = {}    # name -> base_objects.ModelVariable
        self.coups_indep = []  # base_objects.ModelVariable
        self.params_dep = []   # base_objects.ModelVariable
        self.params_indep = [] # base_objects.ModelVariable
        self.p_to_cpp = parsers.UFOExpressionParserCPP()

        # Prepare parameters and couplings for writeout in C++
        self.prepare_parameters()
        self.prepare_couplings(wanted_couplings)

    def write_files(self):
        """Create all necessary files"""

        # Write Helas Routines
        self.write_aloha_routines()

        # Write parameter (and coupling) class files
        self.write_parameter_class_files()

    # Routines for preparing parameters and couplings from the model

    def prepare_parameters(self):
        """Extract the parameters from the model, and store them in
        the two lists params_indep and params_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['parameters'].keys())
        keys.sort(key=len)
        params_ext = []
        for key in keys:
            if key == ('external',):
                params_ext += [p for p in self.model['parameters'][key] if p.name]
            elif 'aS' in key:
                for p in self.model['parameters'][key]:
                    self.params_dep.append(base_objects.ModelVariable(p.name,
                                              p.name + " = " + \
                                              self.p_to_cpp.parse(p.expr) + ";",
                                              p.type,
                                              p.depend))
            else:
                for p in self.model['parameters'][key]:
                    if p.name == 'ZERO':
                        continue
                    self.params_indep.append(base_objects.ModelVariable(p.name,
                                              p.name + " = " + \
                                              self.p_to_cpp.parse(p.expr) + ";",
                                              p.type,
                                              p.depend))

        # For external parameters, want to read off the SLHA block code
        while params_ext:
            param = params_ext.pop(0)
            # Read value from the slha variable
            expression = ""
            assert param.value.imag == 0
            if len(param.lhacode) == 1:
                expression = "%s = slha.get_block_entry(\"%s\", %d, %e);" % \
                             (param.name, param.lhablock.lower(),
                              param.lhacode[0], param.value.real)
            elif len(param.lhacode) == 2:
                expression = "indices[0] = %d;\nindices[1] = %d;\n" % \
                             (param.lhacode[0], param.lhacode[1])
                expression += "%s = slha.get_block_entry(\"%s\", indices, %e);" \
                              % (param.name, param.lhablock.lower(), param.value.real)
            else:
                raise MadGraph5Error("Only support for SLHA blocks with 1 or 2 indices")
            self.params_indep.insert(0,
                                   base_objects.ModelVariable(param.name,
                                                   expression,
                                                              'real'))
            
    def prepare_couplings(self, wanted_couplings = []):
        """Extract the couplings from the model, and store them in
        the two lists coups_indep and coups_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['couplings'].keys())
        keys.sort(key=len)
        for key, coup_list in self.model['couplings'].items():
            if "aS" in key:
                for c in coup_list:
                    if not wanted_couplings or c.name in wanted_couplings:
                        self.coups_dep[c.name] = base_objects.ModelVariable(\
                                                                   c.name,
                                                                   c.expr,
                                                                   c.type,
                                                                   c.depend)
            else:
                for c in coup_list:
                    if not wanted_couplings or c.name in wanted_couplings:
                        self.coups_indep.append(base_objects.ModelVariable(\
                                                                   c.name,
                                                                   c.expr,
                                                                   c.type,
                                                                   c.depend))

        # Convert coupling expressions from Python to C++
        for coup in list(self.coups_dep.values()) + self.coups_indep:
            coup.expr = coup.name + " = " + self.p_to_cpp.parse(coup.expr) + ";"

    # Routines for writing the parameter files

    def write_parameter_class_files(self):
        """Generate the parameters_model.h and parameters_model.cc
        files, which have the parameters and couplings for the model."""

        if not os.path.isdir(os.path.join(self.dir_path, self.include_dir)):
            os.makedirs(os.path.join(self.dir_path, self.include_dir))
        if not os.path.isdir(os.path.join(self.dir_path, self.cc_file_dir)):
            os.makedirs(os.path.join(self.dir_path, self.cc_file_dir))

        parameter_h_file = os.path.join(self.dir_path, self.include_dir,
                                    'Parameters_%s.h' % self.model_name)
        parameter_cc_file = os.path.join(self.dir_path, self.cc_file_dir,
                                     'Parameters_%s.cc' % self.model_name)

        file_h, file_cc = self.generate_parameters_class_files()

        # Write the files
        writers.CPPWriter(parameter_h_file).writelines(file_h)
        writers.CPPWriter(parameter_cc_file).writelines(file_cc)

        # Copy additional needed files
        for copy_file in self.copy_include_files:
            shutil.copy(os.path.join(_file_path, 'iolibs',
                                         'template_files',copy_file),
                        os.path.join(self.dir_path, self.include_dir))
        # Copy additional needed files
        for copy_file in self.copy_cc_files:
            shutil.copy(os.path.join(_file_path, 'iolibs',
                                         'template_files',copy_file),
                        os.path.join(self.dir_path, self.cc_file_dir))

        logger.info("Created files %s and %s in directory" \
                    % (os.path.split(parameter_h_file)[-1],
                       os.path.split(parameter_cc_file)[-1]))
        logger.info("%s and %s" % \
                    (os.path.split(parameter_h_file)[0],
                     os.path.split(parameter_cc_file)[0]))

    def generate_parameters_class_files(self):
        """Create the content of the Parameters_model.h and .cc files"""

        replace_dict = self.default_replace_dict

        replace_dict['info_lines'] = get_mg5_info_lines()
        replace_dict['model_name'] = self.model_name

        replace_dict['independent_parameters'] = \
                                   "// Model parameters independent of aS\n" + \
                                   self.write_parameters(self.params_indep)
        replace_dict['independent_couplings'] = \
                                    "// Model couplings independent of aS\n" + \
                                    self.write_parameters(self.coups_indep)
                                  
                                  
        replace_dict['dependent_parameters'] = \
                                    "// Model parameters dependent on aS\n" + \
                                    self.write_parameters(self.params_dep)
        replace_dict['dependent_couplings'] = \
                                   "// Model couplings dependent on aS\n" + \
                                   self.write_parameters(list(self.coups_dep.values()))

        replace_dict['set_independent_parameters'] = \
                               self.write_set_parameters(self.params_indep)
        replace_dict['set_independent_couplings'] = \
                               self.write_set_parameters(self.coups_indep)
        replace_dict['set_dependent_parameters'] = \
                               self.write_set_parameters(self.params_dep)
        replace_dict['set_dependent_couplings'] = \
                               self.write_set_parameters(list(self.coups_dep.values()))

        replace_dict['print_independent_parameters'] = \
                               self.write_print_parameters(self.params_indep)
        replace_dict['print_independent_couplings'] = \
                               self.write_print_parameters(self.coups_indep)
        replace_dict['print_dependent_parameters'] = \
                               self.write_print_parameters(self.params_dep)
        replace_dict['print_dependent_couplings'] = \
                               self.write_print_parameters(list(self.coups_dep.values()))

        if 'include_prefix' not in replace_dict:
            replace_dict['include_prefix'] = ''


        file_h = self.read_template_file(self.param_template_h) % \
                 replace_dict
        file_cc = self.read_template_file(self.param_template_cc) % \
                  replace_dict
        
        return file_h, file_cc

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
            res_strings.append("%s" % param.expr)

        # Correct width sign for Majorana particles (where the width
        # and mass need to have the same sign)        
        for particle in self.model.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero':
                res_strings.append("if (%s < 0)" % particle.get('mass'))
                res_strings.append("%(width)s = -abs(%(width)s);" % \
                                   {"width": particle.get('width')})

        return "\n".join(res_strings)

    def write_print_parameters(self, params):
        """Write out the lines of independent parameters"""

        # For each parameter, write name = expr;

        res_strings = []
        for param in params:
            res_strings.append("cout << setw(20) << \"%s \" << \"= \" << setiosflags(ios::scientific) << setw(10) << %s << endl;" % (param.name, param.name))

        return "\n".join(res_strings)

    # Routines for writing the ALOHA files

    def write_aloha_routines(self):
        """Generate the hel_amps_model.h and hel_amps_model.cc files, which
        have the complete set of generalized Helas routines for the model"""
        
        if not os.path.isdir(os.path.join(self.dir_path, self.include_dir)):
            os.makedirs(os.path.join(self.dir_path, self.include_dir))
        if not os.path.isdir(os.path.join(self.dir_path, self.cc_file_dir)):
            os.makedirs(os.path.join(self.dir_path, self.cc_file_dir))

        model_h_file = os.path.join(self.dir_path, self.include_dir,
                                    'HelAmps_%s.h' % self.model_name)
        model_cc_file = os.path.join(self.dir_path, self.cc_file_dir,
                                     'HelAmps_%s.%s' % (self.model_name, self.cc_ext))

        replace_dict = {}

        replace_dict['output_name'] = self.output_name
        replace_dict['info_lines'] = get_mg5_info_lines()
        replace_dict['namespace'] = self.namespace
        replace_dict['model_name'] = self.model_name

        # Read in the template .h and .cc files, stripped of compiler
        # commands and namespaces
        template_h_files = self.read_aloha_template_files(ext = 'h')
        template_cc_files = self.read_aloha_template_files(ext = 'cc')

        aloha_model = create_aloha.AbstractALOHAModel(self.model.get('name'),
                                                      explicit_combine=True)
        aloha_model.add_Lorentz_object(self.model.get('lorentz'))
        
        if self.wanted_lorentz:
            aloha_model.compute_subset(self.wanted_lorentz)
        else:
            aloha_model.compute_all(save=False, custom_propa=True)
            
        for abstracthelas in dict(aloha_model).values():
            h_rout, cc_rout = abstracthelas.write(output_dir=None, 
                                                  language=self.aloha_writer, 
                                                  mode='no_include')

            template_h_files.append(h_rout)
            template_cc_files.append(cc_rout)
            
            #aloha_writer = aloha_writers.ALOHAWriterForCPP(abstracthelas,
            #                                               self.dir_path)
            #header = aloha_writer.define_header()
            #template_h_files.append(self.write_function_declaration(\
            #                             aloha_writer, header))
            #template_cc_files.append(self.write_function_definition(\
            #                              aloha_writer, header))

        replace_dict['function_declarations'] = '\n'.join(template_h_files)
        replace_dict['function_definitions'] = '\n'.join(template_cc_files)

        file_h = self.read_template_file(self.aloha_template_h) % replace_dict
        file_cc = self.read_template_file(self.aloha_template_cc) % replace_dict

        # Write the files
        writers.CPPWriter(model_h_file).writelines(file_h)
        writers.CPPWriter(model_cc_file).writelines(file_cc)

        logger.info("Created files %s and %s in directory" \
                    % (os.path.split(model_h_file)[-1],
                       os.path.split(model_cc_file)[-1]))
        logger.info("%s and %s" % \
                    (os.path.split(model_h_file)[0],
                     os.path.split(model_cc_file)[0]))


    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        template_files = []
        for filename in misc.glob('*.%s' % ext, pjoin(MG5DIR, 'aloha','template_files')):
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

#    def write_function_declaration(self, aloha_writer, header):
#        """Write the function declaration for the ALOHA routine"""
#
#        ret_lines = []
#        for line in aloha_writer.write_h(header).split('\n'):
#            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
#                # Strip out compiler flags and namespaces
#                continue
#            ret_lines.append(line)
#        return "\n".join(ret_lines)
#
#    def write_function_definition(self, aloha_writer, header):
#        """Write the function definition for the ALOHA routine"""
#
#        ret_lines = []
#        for line in aloha_writer.write_cc(header).split('\n'):
#            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
#                # Strip out compiler flags and namespaces
#                continue
#            ret_lines.append(line)
#        return "\n".join(ret_lines)

    def clean_line(self, line):
        """Strip a line of compiler options and namespace options."""

        if self.compiler_option_re.match(line) or self.namespace_re.match(line):
            return ""

        return line

    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, filename, classpath=False):
        """Open a template file and return the contents."""
         
        return OneProcessExporterCPP.read_template_file(filename, classpath)


#===============================================================================
# UFOModelConverterGPU
#===============================================================================

class UFOModelConverterGPU(UFOModelConverterCPP):
    
    aloha_writer = 'cudac'
    cc_ext = 'cu'
        # Template files to use
    #include_dir = '.'
    #c_file_dir = '.'
    #param_template_h = 'cpp_model_parameters_h.inc'
    #param_template_cc = 'cpp_model_parameters_cc.inc'
    aloha_template_h = pjoin('gpu','cpp_hel_amps_h.inc')
    aloha_template_cc = pjoin('gpu','cpp_hel_amps_cc.inc')
    helas_h = pjoin('gpu', 'helas.h')
    helas_cc = pjoin('gpu', 'helas.cu')

    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        path = pjoin(MG5DIR, 'aloha','template_files')
        out = []
        
        if ext == 'h':
            out.append(open(pjoin(path, self.helas_h)).read())
        else:
            out.append(open(pjoin(path, self.helas_cc)).read())
    
        return out

    def write_process_h_file(self, writer):
        
        replace_dict = UFOModelConverterCPP.write_process_h_file(self, None)
        replace_dict['include_for_complex'] = '#include "mgOnGpuTypes.h"'
        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict

class OneProcessExporterCPP(object):
    """Class to take care of exporting a set of matrix elements to
    C++ format."""


    # Static variables (for inheritance)
    process_dir = '.'
    include_dir = '.'
    template_path = os.path.join(_file_path, 'iolibs', 'template_files')
    __template_path = os.path.join(_file_path, 'iolibs', 'template_files') 
    process_template_h = 'cpp_process_h.inc'
    process_template_cc = 'cpp_process_cc.inc'
    process_class_template = 'cpp_process_class.inc'
    process_definition_template = 'cpp_process_function_definitions.inc'
    process_wavefunction_template = 'cpp_process_wavefunctions.inc'
    process_sigmaKin_function_template = 'cpp_process_sigmaKin_function.inc'
    single_process_template = 'cpp_process_matrix.inc'
    cc_ext = 'cc'
    support_multichannel = False

    class ProcessExporterCPPError(Exception):
        pass
    
    def __init__(self, matrix_elements, cpp_helas_call_writer, process_string = "",
                 process_number = 0, path = os.getcwd(), prefix=""):
        """Initiate with matrix elements, helas call writer, process
        string, path. Generate the process .h and .cc files."""

        if isinstance(matrix_elements, helas_objects.HelasMultiProcess):
            self.matrix_elements = matrix_elements.get('matrix_elements')
        elif isinstance(matrix_elements, helas_objects.HelasMatrixElement):
            self.matrix_elements = \
                         helas_objects.HelasMatrixElementList([matrix_elements])
        elif isinstance(matrix_elements, helas_objects.HelasMatrixElementList):
            self.matrix_elements = matrix_elements
        else:
            raise base_objects.PhysicsObject.PhysicsObjectError("Wrong object type for matrix_elements: %s" % type(matrix_elements))

        if not self.matrix_elements:
            raise MadGraph5Error("No matrix elements to export")

        self.model = self.matrix_elements[0].get('processes')[0].get('model')
        self.model_name = ProcessExporterCPP.get_model_name(self.model.get('name'))

        self.processes = sum([me.get('processes') for \
                              me in self.matrix_elements], [])
        self.processes.extend(sum([me.get_mirror_processes() for \
                              me in self.matrix_elements], []))

        self.nprocesses = len(self.matrix_elements)
        if any([m.get('has_mirror_process') for m in self.matrix_elements]):
            self.nprocesses = 2*len(self.matrix_elements)

        if process_string:
            self.process_string = process_string
        else:
            self.process_string = self.processes[0].base_string()

        if process_number:
            self.process_number = process_number
        else:
            self.process_number = self.processes[0].get('id')

        self.process_name = self.get_process_name()
        self.process_class = "CPPProcess"

        self.path = path
        self.helas_call_writer = cpp_helas_call_writer

        if not isinstance(self.helas_call_writer, helas_call_writers.CPPUFOHelasCallWriter):
            raise self.ProcessExporterCPPError("helas_call_writer not CPPUFOHelasCallWriter")

        self.nexternal, self.ninitial = \
                        self.matrix_elements[0].get_nexternal_ninitial()
        self.nfinal = self.nexternal - self.ninitial

        # Check if we can use the same helicities for all matrix
        # elements
        
        self.single_helicities = True

        hel_matrix = self.get_helicity_matrix(self.matrix_elements[0])

        for me in self.matrix_elements[1:]:
            if self.get_helicity_matrix(me) != hel_matrix:
                self.single_helicities = False

        if self.single_helicities:
            # If all processes have the same helicity structure, this
            # allows us to reuse the same wavefunctions for the
            # different processes
            
            self.wavefunctions = []
            wf_number = 0

            for me in self.matrix_elements:
                for iwf, wf in enumerate(me.get_all_wavefunctions()):
                    try:
                        old_wf = \
                               self.wavefunctions[self.wavefunctions.index(wf)]
                        wf.set('number', old_wf.get('number'))
                    except ValueError:
                        wf_number += 1
                        wf.set('number', wf_number)
                        self.wavefunctions.append(wf)

            # Also combine amplitudes
            self.amplitudes = helas_objects.HelasAmplitudeList()
            amp_number = 0
            for me in self.matrix_elements:
                for iamp, amp in enumerate(me.get_all_amplitudes()):
                    try:
                        old_amp = \
                               self.amplitudes[self.amplitudes.index(amp)]
                        amp.set('number', old_amp.get('number'))
                    except ValueError:
                        amp_number += 1
                        amp.set('number', amp_number)
                        self.amplitudes.append(amp)
            diagram = helas_objects.HelasDiagram({'amplitudes': self.amplitudes})
            self.amplitudes = helas_objects.HelasMatrixElement({\
                'diagrams': helas_objects.HelasDiagramList([diagram])})


            self.include_multi_channel = False
    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, filename, classpath=False):
        """Open a template file and return the contents."""
         
        if isinstance(filename, tuple):
            file_path = filename[0]
            filename = filename[1]
        elif isinstance(filename, str):
            if classpath:
                file_path = cls.__template_path
            else:
                file_path = cls.template_path
        else:
            raise MadGraph5Error('Argument should be string or tuple.')
        
        return open(os.path.join(file_path, filename)).read()
        
        
                  
    @staticmethod
    def get_multi_channel_dictionary(matrix_element, config_map):

        return ProcessExporterFortran.get_multi_channel_dictionary(matrix_element, config_map)

    # Methods for generation of process files for C++
    def generate_process_files(self):
        """Generate the .h and .cc files needed for C++, for the
        processes described by multi_matrix_element"""

        # Create the files
        if not os.path.isdir(os.path.join(self.path, self.include_dir)):
            os.makedirs(os.path.join(self.path, self.include_dir))
        filename = os.path.join(self.path, self.include_dir,
                                '%s.h' % self.process_class)

        
        self.write_process_h_file(writers.CPPWriter(filename))

        if not os.path.isdir(os.path.join(self.path, self.process_dir)):
            os.makedirs(os.path.join(self.path, self.process_dir))
        filename = os.path.join(self.path, self.process_dir,
                                '%s.%s' % (self.process_class, self.cc_ext)) 
        self.write_process_cc_file(writers.CPPWriter(filename))

        logger.info('Created files %(process)s.h and %(process)s.cc in' % \
                    {'process': self.process_class} + \
                    ' directory %(dir)s' % {'dir': os.path.split(filename)[0]})

    def generate_process_files_madevent(self, proc_id, config_map, subproc_number):


        self.include_multi_channel = config_map
        self.generate_process_files() 
#        raise Exception("working fine but not fully implemented so far")


    def get_default_converter(self):
        
        replace_dict = {}       

        
        return replace_dict
    
    #===========================================================================
    # write_process_h_file
    #===========================================================================
    def write_process_h_file(self, writer):
        """Write the class definition (.h) file for the process"""
        
        if writer and not isinstance(writer, writers.CPPWriter):
            raise writers.CPPWriter.CPPWriterError(\
                "writer not CPPWriter")

        replace_dict = self.get_default_converter()

        # Extract version number and date from VERSION file
        info_lines = get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract model name
        replace_dict['model_name'] = \
                         self.model_name

        # Extract process file name
        replace_dict['process_file_name'] = self.process_name

        # Extract class definitions
        process_class_definitions = self.get_process_class_definitions()
        replace_dict['process_class_definitions'] = process_class_definitions
        replace_dict['include_for_complex'] = ''

        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
    #===========================================================================
    # write_process_cc_file
    #===========================================================================
    def write_process_cc_file(self, writer):
        """Write the class member definition (.cc) file for the process
        described by matrix_element"""

        if writer:
            if not isinstance(writer, writers.CPPWriter):
                raise writers.CPPWriter.CPPWriterError(\
                "writer not CPPWriter")

        replace_dict = self.get_default_converter()

        # Extract version number and date from VERSION file
        info_lines = get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process file name
        replace_dict['process_file_name'] = self.process_name

        # Extract model name
        replace_dict['model_name'] = self.model_name
                         

        # Extract class function definitions
        process_function_definitions = \
                         self.get_process_function_definitions()
        replace_dict['process_function_definitions'] = \
                                                   process_function_definitions

        if writer:
            file = self.read_template_file(self.process_template_cc) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict

    #===========================================================================
    # Process export helper functions
    #===========================================================================
    def get_process_class_definitions(self, write=True):
        """The complete class definition for the process"""

        replace_dict = {}

        # Extract model name
        replace_dict['model_name'] = self.model_name

        # Extract process info lines for all processes
        process_lines = "\n".join([self.get_process_info_lines(me) for me in \
                                   self.matrix_elements])
        
        replace_dict['process_lines'] = process_lines

        # Extract number of external particles
        replace_dict['nfinal'] = self.nfinal

        # Extract number of external particles
        replace_dict['ninitial'] = self.ninitial

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        # Extract process definition
        process_definition = "%s (%s)" % (self.process_string,
                                          self.model_name)
        replace_dict['process_definition'] = process_definition

        process = self.processes[0]

        replace_dict['process_code'] = self.process_number
        replace_dict['nexternal'] = self.nexternal
        replace_dict['nprocesses'] = self.nprocesses
        

        color_amplitudes = self.matrix_elements[0].get_color_amplitudes()
        # Number of color flows
        replace_dict['ncolor'] = len(color_amplitudes)

        if self.single_helicities:
            wfct_size = 18
            # Set the size of Wavefunction
            if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
                wfct_size = 18
            else:
                wfct_size = 6
            
            replace_dict['wfct_size'] = wfct_size
            
            replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          void calculate_wavefunctions(const int perm[], const int hel[]);
                          static const int nwavefuncs = %(nwfct)d;
                          std::complex<double> w[nwavefuncs][%(sizew)d];
                          static const int namplitudes = %(namp)d;
                          std::complex<double> amp[namplitudes];""" % \
                          {'nwfct':len(self.wavefunctions),
                          'sizew': wfct_size,
                          'namp':len(self.amplitudes.get_all_amplitudes())
                          }

            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s();" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

        else:
            replace_dict['all_sigma_kin_definitions'] = \
                          "\n".join(["void sigmaKin_%s();" % \
                                     me.get('processes')[0].shell_string().\
                                     replace("0_", "") \
                                     for me in self.matrix_elements])
            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s(const int hel[]);" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

        if write:
            file = self.read_template_file(self.process_class_template) % replace_dict
            return file
        else:
            return replace_dict
        
    def get_process_function_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = {}

        # Extract model name
        replace_dict['model_name'] = self.model_name

        # Extract process info lines
        replace_dict['process_lines'] = \
                             "\n".join([self.get_process_info_lines(me) for \
                                        me in self.matrix_elements])

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        color_amplitudes = [me.get_color_amplitudes() for me in \
                            self.matrix_elements]

        replace_dict['initProc_lines'] = \
                                self.get_initProc_lines(self.matrix_elements[0],
                                                        color_amplitudes)
        replace_dict['reset_jamp_lines'] = \
                                     self.get_reset_jamp_lines(color_amplitudes)
        replace_dict['sigmaKin_lines'], other_replace = \
                                     self.get_sigmaKin_lines(color_amplitudes)
        replace_dict.update(other_replace)
            
        replace_dict['sigmaHat_lines'] = self.get_sigmaHat_lines()

        replace_dict['all_sigmaKin'] = \
                                  self.get_all_sigmaKin_lines(color_amplitudes,
                                                              'CPPProcess')
        
        replace_dict['nexternal'] = len(self.matrix_elements[0].get('processes')[0].get('legs'))
    
        if write:
            file = self.read_template_file(self.process_definition_template) %\
               replace_dict
            return file
        else:
            return replace_dict

    def get_process_name(self):
        """Return process file name for the process in matrix_element"""

        process_string = self.process_string

        # Extract process number
        proc_number_pattern = re.compile(r"^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(process_string)
        proc_number = 0
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            process_string = proc_number_re.group(1) + \
                             proc_number_re.group(3)

        # Remove order information
        order_pattern = re.compile(r"^(.+)\s+(\w+)\s*=\s*(\d+)\s*$")
        order_re = order_pattern.match(process_string)
        while order_re:
            process_string = order_re.group(1)
            order_re = order_pattern.match(process_string)
        
        process_string = process_string.replace(' ', '')
        process_string = process_string.replace('>', '_')
        process_string = process_string.replace('+', 'p')
        process_string = process_string.replace('-', 'm')
        process_string = process_string.replace('~', 'x')
        process_string = process_string.replace('/', '_no_')
        process_string = process_string.replace('$', '_nos_')
        process_string = process_string.replace('|', '_or_')
        if proc_number != 0:
            process_string = "%d_%s" % (proc_number, process_string)

        process_string = "Sigma_%s_%s" % (self.model_name,
                                          process_string)
        return process_string

    def get_process_info_lines(self, matrix_element):
        """Return info lines describing the processes for this matrix element"""

        return"\n".join([ "# " + process.nice_string().replace('\n', '\n# * ') \
                         for process in matrix_element.get('processes')])


    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set external particle masses for this matrix element")

        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append("mME.push_back(pars->%s);" % part.get('mass'))
        for i, colamp in enumerate(color_amplitudes):
            initProc_lines.append("jamp2[%d] = new double[%d];" % \
                                  (i, len(colamp)))

        return "\n".join(initProc_lines)

    def get_reset_jamp_lines(self, color_amplitudes):
        """Get lines to reset jamps"""

        ret_lines = ""
        for icol, col_amp in enumerate(color_amplitudes):
            ret_lines+= """for(int i=0;i < %(ncolor)d; i++)
            jamp2[%(proc_number)d][i]=0.;\n""" % \
            {"ncolor": len(col_amp), "proc_number": icol}
        return ret_lines
        

    def get_calculate_wavefunctions(self, wavefunctions, amplitudes, write=True):
        """Return the lines for optimized calculation of the
        wavefunctions for all subprocesses"""

        replace_dict = {}

        replace_dict['nwavefuncs'] = len(wavefunctions)
        
        #ensure no recycling of wavefunction ! incompatible with some output
        for me in self.matrix_elements:
            me.restore_original_wavefunctions()

        replace_dict['wavefunction_calls'] = "\n".join(\
            self.helas_call_writer.get_wavefunction_calls(\
            helas_objects.HelasWavefunctionList(wavefunctions)))

        replace_dict['amplitude_calls'] = "\n".join(\
            self.helas_call_writer.get_amplitude_calls(amplitudes))

        if write:
            file = self.read_template_file(self.process_wavefunction_template) % \
                replace_dict
            return file
        else:
            return replace_dict
       

    def get_sigmaKin_lines(self, color_amplitudes, write=True):
        """Get sigmaKin_lines for function definition for Pythia 8 .cc file"""

        if self.include_multi_channel and not self.support_multichannel:
            raise Exception("This standalone format does not support madevent interface")

        
        if self.single_helicities:
            replace_dict = {}

            # Number of helicity combinations
            replace_dict['ncomb'] = \
                            self.matrix_elements[0].get_helicity_combinations()

            # Process name
            replace_dict['process_class_name'] = self.process_name
        
            # Particle ids for the call to setupForME
            replace_dict['id1'] = self.processes[0].get('legs')[0].get('id')
            replace_dict['id2'] = self.processes[0].get('legs')[1].get('id')

            # Extract helicity matrix
            replace_dict['helicity_matrix'] = \
                            self.get_helicity_matrix(self.matrix_elements[0])

            # Extract denominator
            den_factors = [str(me.get_denominator_factor()) for me in \
                               self.matrix_elements]
            if self.nprocesses != len(self.matrix_elements):
                den_factors.extend(den_factors)
            replace_dict['den_factors'] = ",".join(den_factors)
            replace_dict['get_matrix_t_lines'] = "\n".join(
                     ["t[%(iproc)d]=matrix_%(proc_name)s();" % \
                     {"iproc": i, "proc_name": \
                      me.get('processes')[0].shell_string().replace("0_", "")} \
                     for i, me in enumerate(self.matrix_elements)])

            # temporary
            replace_dict['madE_var_reset'] = ''
            replace_dict['madE_caclwfcts_call'] = ''
            replace_dict['madE_update_answer'] = ''



            # Generate lines for mirror matrix element calculation
            mirror_matrix_lines = ""

            if any([m.get('has_mirror_process') for m in self.matrix_elements]):
                mirror_matrix_lines += \
"""             // Mirror initial state momenta for mirror process
                perm[0]=1;
                perm[1]=0;
                // Calculate wavefunctions
                calculate_wavefunctions(perm, helicities[ihel]);
                // Mirror back
                perm[0]=0;
                perm[1]=1;
                // Calculate matrix elements
                """
                
                mirror_matrix_lines += "\n".join(
                    ["t[%(iproc)d]=matrix_%(proc_name)s();" % \
                     {"iproc": i + len(self.matrix_elements), "proc_name": \
                      me.get('processes')[0].shell_string().replace("0_", "")} \
                     for i, me in enumerate(self.matrix_elements) if me.get('has_mirror_process')])
                    
            replace_dict['get_mirror_matrix_lines'] = mirror_matrix_lines

            replace_dict['nproc'] = sum([ 2 if m.get('has_mirror_process') else 1
                                        for m in self.matrix_elements])
            replace_dict['nb_amp'] = len(self.amplitudes.get_all_amplitudes())
            replace_dict['nexternal'] = len(self.processes[0].get('legs'))
            if write:
                file = \
                 self.read_template_file(\
                            self.process_sigmaKin_function_template) %\
                            replace_dict
                return file, replace_dict
            else:
                return replace_dict
        else:
            ret_lines = "// Call the individual sigmaKin for each process\n"
            ret_lines = ret_lines + \
                   "\n".join(["sigmaKin_%s();" % \
                              me.get('processes')[0].shell_string().\
                              replace("0_", "") for \
                              me in self.matrix_elements])
            if write:
                return ret_lines, replace_dict
            else:
                replace_dict['get_mirror_matrix_lines'] = ret_lines
                return replace_dict
              
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for Pythia 8 .cc file"""

        ret_lines = []
        if self.single_helicities:
            ret_lines.append(\
                "void %s::calculate_wavefunctions(const int perm[], const int hel[]){" % \
                class_name)
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append(self.get_calculate_wavefunctions(\
                self.wavefunctions, self.amplitudes))
            ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)


    def get_sigmaKin_single_process(self, i, matrix_element, write=True):
        """Write sigmaKin for each process"""

        # Write sigmaKin for the process

        replace_dict = {}

        # Process name
        replace_dict['proc_name'] = \
          matrix_element.get('processes')[0].shell_string().replace("0_", "")
        
        # Process name
        replace_dict['process_class_name'] = self.process_name
        
        # Process number
        replace_dict['proc_number'] = i

        # Number of helicity combinations
        replace_dict['ncomb'] = matrix_element.get_helicity_combinations()

        # Extract helicity matrix
        replace_dict['helicity_matrix'] = \
                                      self.get_helicity_matrix(matrix_element)
        # Extract denominator
        replace_dict['den_factor'] = matrix_element.get_denominator_factor()
        

        if write:
            file = \
            self.read_template_file('cpp_process_sigmaKin_subproc_function.inc') %\
            replace_dict
            return file
        else:
            return replace_dict
        
    def get_matrix_single_process(self, i, matrix_element, color_amplitudes,
                                  class_name, write=True):
        """Write matrix() for each process"""

        # Write matrix() for the process

        replace_dict = {}

        # Process name
        replace_dict['proc_name'] = \
          matrix_element.get('processes')[0].shell_string().replace("0_", "")
        

        # Wavefunction and amplitude calls
        if self.single_helicities:
            replace_dict['matrix_args'] = ""
            replace_dict['all_wavefunction_calls'] = "" 
        else:
            replace_dict['matrix_args'] = "const int hel[]"
            wavefunctions = matrix_element.get_all_wavefunctions()
            replace_dict['all_wavefunction_calls'] = \
                         """const int nwavefuncs = %d;
                         std::complex<double> w[nwavefuncs][18];
                         """ % len(wavefunctions)+ \
                         self.get_calculate_wavefunctions(wavefunctions, [])

        # Process name
        replace_dict['process_class_name'] = class_name
        
        # Process number
        replace_dict['proc_number'] = i

        # Number of color flows
        replace_dict['ncolor'] = len(color_amplitudes)

        replace_dict['ngraphs'] = matrix_element.get_number_of_amplitudes()

        # Extract color matrix
        replace_dict['color_matrix_lines'] = \
                                     self.get_color_matrix_lines(matrix_element)

                                     
        replace_dict['jamp_lines'] = self.get_jamp_lines(color_amplitudes)


        #specific exporter hack
        replace_dict =  self.get_class_specific_definition_matrix(replace_dict, matrix_element)
        
        if write:
            file = self.read_template_file(self.single_process_template) % \
                replace_dict
            return file
        else:
            return replace_dict
        
    def get_class_specific_definition_matrix(self, converter, matrix_element):
        """place to add some specific hack to a given exporter.
        Please always use Super in that case"""

        return converter

    def get_sigmaHat_lines(self):
        """Get sigmaHat_lines for function definition for Pythia 8 .cc file"""

        # Create a set with the pairs of incoming partons
        beams = set([(process.get('legs')[0].get('id'),
                      process.get('legs')[1].get('id')) \
                     for process in self.processes])
        beams = sorted(list(beams))
        res_lines = []

        # Write a selection routine for the different processes with
        # the same beam particles
        res_lines.append("// Select between the different processes")
        for ibeam, beam_parts in enumerate(beams):
            
            if ibeam == 0:
                res_lines.append("if(id1 == %d && id2 == %d){" % beam_parts)
            else:
                res_lines.append("else if(id1 == %d && id2 == %d){" % beam_parts)            
            
            # Pick out all processes with this beam pair
            beam_processes = [(i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[0].get('id'),
                                process.get('legs')[1].get('id')) \
                               for process in me.get('processes')]]

            # Add mirror processes, 
            beam_processes.extend([(len(self.matrix_elements) + i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[0].get('id'),
                                process.get('legs')[1].get('id')) \
                               for process in me.get_mirror_processes()]])

            # Now add matrix elements for the processes with the right factors
            res_lines.append("// Add matrix elements for processes with beams %s" % \
                             repr(beam_parts))
            res_lines.append("return %s;" % \
                             ("+".join(["matrix_element[%i]*%i" % \
                                        (i, len([proc for proc in \
                                         me.get('processes') if beam_parts == \
                                         (proc.get('legs')[0].get('id'),
                                          proc.get('legs')[1].get('id')) or \
                                         me.get('has_mirror_process') and \
                                         beam_parts == \
                                         (proc.get('legs')[1].get('id'),
                                          proc.get('legs')[0].get('id'))])) \
                                        for (i, me) in beam_processes]).\
                              replace('*1', '')))
            res_lines.append("}")
            

        res_lines.append("else {")
        res_lines.append("// Return 0 if not correct initial state assignment")
        res_lines.append(" return 0.;}")

        return "\n".join(res_lines)


    def get_helicity_matrix(self, matrix_element):
        """Return the Helicity matrix definition lines for this matrix element"""

        helicity_line = "static const int helicities[ncomb][nexternal] = {";
        helicity_line_list = []

        for helicities in matrix_element.get_helicity_matrix(allow_reverse=False):
            helicity_line_list.append("{"+",".join(['%d'] * len(helicities)) % \
                                       tuple(helicities) + "}")

        return helicity_line + ",".join(helicity_line_list) + "};"

    def get_den_factor_line(self, matrix_element):
        """Return the denominator factor line for this matrix element"""

        return "const int denominator = %d;" % \
               matrix_element.get_denominator_factor()

    def get_color_matrix_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return "\n".join(["static const double denom = 1;",
                              "static const int cf[1] = {1};"])
        else:
            color_denominators = matrix_element.get('color_matrix').\
                                                 get_line_denominators()
            denominator = min(color_denominators)
            denom_string = "static const int denom = %i;" % (denominator)

            matrix_strings = []
            my_cs = color.ColorString()
            for index in range(len(color_denominators)):
                # Then write the numerators for the matrix elements
                num_list = matrix_element.get('color_matrix').\
                                            get_line_numerators(index, denominator)

                matrix_strings+= ["%d" % (i if pos==0 else 2*i) for pos,i in enumerate(num_list[index:])]
            matrix_string = "static const int cf[ncolor*(ncolor+1)/2] = {" + \
                            ",".join(matrix_strings) + "};"
            return "\n".join([denom_string, matrix_string])


    @staticmethod
    def coeff(ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Returns a nicely formatted string for the coefficients in JAMP lines"""
    
        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power
    
        if total_coeff == 1:
            if is_imaginary:
                return '+std::complex<double>(0,1)*'
            else:
                return '+'
        elif total_coeff == -1:
            if is_imaginary:
                return '-std::complex<double>(0,1)*'
            else:
                return '-'
    
        res_str = '%+i.' % total_coeff.numerator
    
        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + '/%i.' % total_coeff.denominator
    
        if is_imaginary:
            res_str = res_str + '*std::complex<double>(0,1)'
    
        return res_str + '*'



            
    def get_jamp_lines(self, color_amplitudes):
        """Return the jamp = sum(fermionfactor * amp[i]) lines"""

        res_list = []

        for i, coeff_list in enumerate(color_amplitudes):

            res = "jamp[%i]=" % i

            # Optimization: if all contributions to that color basis element have
            # the same coefficient (up to a sign), put it in front
            list_fracs = [abs(coefficient[0][1]) for coefficient in coeff_list]
            common_factor = False
            diff_fracs = misc.make_unique(list_fracs)
            if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                common_factor = True
                global_factor = diff_fracs[0]
                res = res + '%s(' % self.coeff(1, global_factor, False, 0)

            for (coefficient, amp_number) in coeff_list:

                if common_factor:
                    res = res + "%samp[%d]" % (self.coeff(coefficient[0],
                                               coefficient[1] / abs(coefficient[1]),
                                               coefficient[2],
                                               coefficient[3]),
                                               amp_number - 1)
                else:
                    res = res + "%samp[%d]" % (self.coeff(coefficient[0],
                                               coefficient[1],
                                               coefficient[2],
                                               coefficient[3]),
                                               amp_number - 1)

            if common_factor:
                res = res + ')'

            res += ';'
            res_list.append(res)

        return "\n".join(res_list)
    
coeff = OneProcessExporterCPP.coeff

class OneProcessExporterGPU(OneProcessExporterCPP):

    # Static variables (for inheritance)
    process_dir = '.'
    include_dir = '.'
    template_path = os.path.join(_file_path, 'iolibs', 'template_files')
    __template_path = os.path.join(_file_path, 'iolibs', 'template_files') 
    process_template_h = 'gpu/process_h.inc'
    process_template_cc = 'gpu/process_cc.inc'
    process_class_template = 'gpu/process_class.inc'
    process_definition_template = 'gpu/process_function_definitions.inc'
    process_wavefunction_template = 'cpp_process_wavefunctions.inc'
    process_sigmaKin_function_template = 'gpu/process_sigmaKin_function.inc'
    single_process_template = 'gpu/process_matrix.inc'
    cc_ext = 'cu'
    support_multichannel = True
    multichannel_var = ',fptype& multi_chanel_num, fptype& multi_chanel_denom'

    def __init__(self, *args, **opts):
        
        super(OneProcessExporterGPU, self).__init__(*args, **opts)
        self.process_class = "gCPPProcess"

    def generate_process_files(self):
        
        if self.matrix_elements[0].get('has_mirror_process'):
            self.matrix_elements[0].set('has_mirror_process', False)
            self.nprocesses/=2

        super(OneProcessExporterGPU, self).generate_process_files()

        self.edit_check_sa()
        self.edit_mgonGPU()
        
        # add symbolic link for C++
        files.ln(pjoin(self.path, 'gcheck_sa.cu'), self.path, 'check_sa.cc')
        files.ln(pjoin(self.path, 'gCPPProcess.cu'), self.path, 'CPPProcess.cc')
    

    def edit_check_sa(self):
        
        template = open(pjoin(self.template_path,'gpu','check_sa.cu'),'r').read()
        replace_dict = {}
        replace_dict['nexternal'], _ = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['model'] = self.model_name
        replace_dict['numproc'] = len(self.matrix_elements)

        ff = open(pjoin(self.path, 'gcheck_sa.cu'),'w')
        ff.write(template)
        ff.close()
        
    def edit_mgonGPU(self):
        
        template = open(pjoin(self.template_path,'gpu','mgOnGpuConfig.h'),'r').read()
        replace_dict = {}
        nexternal, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        replace_dict['noutcoming'] = nexternal - nincoming
        
        # Number of helicity combinations
        replace_dict['nbhel'] = \
                            self.matrix_elements[0].get_helicity_combinations()
        replace_dict['nwavefunc'] = \
                          self.matrix_elements[0].get_number_of_wavefunctions()
        replace_dict['wavefuncsize'] = 6
        
        
        ff = open(pjoin(self.path, '..','..','src','mgOnGpuConfig.h'),'w')
        ff.write(template % replace_dict)
        ff.close()        
        

    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set external particle masses for this matrix element")

        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append("mME.push_back(pars->%s);" % part.get('mass'))
        #for i, colamp in enumerate(color_amplitudes):
        #    initProc_lines.append("jamp2[%d] = new double[%d];" % \
        #                          (i, len(colamp)))

        return "\n".join(initProc_lines)

    def get_reset_jamp_lines(self, color_amplitudes):
        """Get lines to reset jamps"""

        ret_lines = ""
        return ret_lines
    
    def get_sigmaKin_lines(self, color_amplitudes, write=True):
        """Get sigmaKin_lines for function definition for Pythia 8 .cc file"""



        replace_dict =  super().get_sigmaKin_lines(color_amplitudes, write=False)

        if self.include_multi_channel:
            replace_dict['madE_var_reset'] = """
            fptype multi_chanel_num = 0.;
            fptype multi_chanel_denom = 0.;
            """
            replace_dict['madE_caclwfcts_call'] = '&multi_chanel_num, &multi_chanel_denom'
            replace_dict['madE_update_answer'] = '   allMEs[iproc*nprocesses + ievt] *= multi_chanel_num/multi_chanel_denom;'

            multi_channel = self.get_multi_channel_dictionary(self.matrix_elements[0].get('diagrams'), self.include_multi_channel)
            replace_dict['nb_channel'] = len(multi_channel)
            replace_dict['nb_color'] = max(1, len(self.matrix_elements[0].get('color_basis')))


        if write:
            file = \
                self.read_template_file(\
                        self.process_sigmaKin_function_template) %\
                        replace_dict
            return file, replace_dict
        else:
            return replace_dict

        if not write:
            return replace_dict, other_replace
        else:
            raise Exception


    @staticmethod
    def coeff(ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Returns a nicely formatted string for the coefficients in JAMP lines"""
    
        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power
    
        if total_coeff == 1:
            if is_imaginary:
                return '+cxtype(0,1)*'
            else:
                return '+'
        elif total_coeff == -1:
            if is_imaginary:
                return '-cxtype(0,1)*'
            else:
                return '-'
    
        res_str = '%+i.' % total_coeff.numerator
    
        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + '/%i.' % total_coeff.denominator
    
        if is_imaginary:
            res_str = res_str + '*cxtype(0,1)'
    
        return res_str + '*'



    def get_process_function_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = super(OneProcessExporterGPU,self).get_process_function_definitions(write=False)


        replace_dict['ncouplings'] = len(self.couplings2order)
        replace_dict['ncouplingstimes2'] = 2 *  replace_dict['ncouplings']
        replace_dict['nparams'] = len(self.params2order)
        replace_dict['nmodels'] = replace_dict['nparams'] + replace_dict['ncouplings']
        replace_dict['coupling_list'] = ' '

        coupling = [''] * len(self.couplings2order)
        params = [''] * len(self.params2order)
        for coup, pos in self.couplings2order.items():
            coupling[pos] = coup
        coup_str = "static cxtype tIPC[%s] = {pars->%s};\n"\
            %(len(self.couplings2order), ',pars->'.join(coupling))
        for para, pos in self.params2order.items():
            params[pos] = para            
        param_str = "static double tIPD[%s] = {pars->%s};\n"\
            %(len(self.params2order), ',pars->'.join(params))            
        
        
        replace_dict['assign_coupling'] = coup_str + param_str
        replace_dict['all_helicities'] = self.get_helicity_matrix(self.matrix_elements[0])
        replace_dict['all_helicities'] = replace_dict['all_helicities'] .replace("helicities", "tHel")
        
        file = self.read_template_file(self.process_definition_template) %\
               replace_dict

        return file

    def get_process_class_definitions(self, write=True):
        
        replace_dict = super(OneProcessExporterGPU,self).get_process_class_definitions(write=False)

        replace_dict['nwavefuncs'] = replace_dict['wfct_size']
        replace_dict['namp'] = len(self.amplitudes.get_all_amplitudes())
        replace_dict['model'] = self.model_name
        
        replace_dict['sizew'] = self.matrix_elements[0].get_number_of_wavefunctions()
        replace_dict['nexternal'], _ = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['ncomb'] = len([x for x in self.matrix_elements[0].get_helicity_matrix()])
        
        replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          __device__ void calculate_wavefunctions(int ihel, double local_mom[%(nexternal)i][3],
                                        cxtype amp[%(namp)d])
                          {
                          const int ncolor =  %(ncolor)d;
                          cxtype jamp[ncolor];

                            cxtype w[%(nwfct)d][%(sizew)d];
                            """ % \
                          {'nwfct':len(self.wavefunctions),
                          'sizew': replace_dict['wfct_size'],
                          'nexternal':replace_dict['nexternal'],
                          'namp':len(self.amplitudes),
                          'ncolor': len(self.matrix_elements[0].get_color_amplitudes())
                          }

        if write:
            file = self.read_template_file(self.process_class_template) % replace_dict
            return file
        else:
            return replace_dict
        
        
#     def get_calculate_wavefunctions(self, wavefunctions, amplitudes, write=True):
#         """Return the lines for optimized calculation of the
#         wavefunctions for all subprocesses"""
# 
#         raise Exception
#         replace_dict = {}
# 
#         replace_dict['nwavefuncs'] = len(wavefunctions)
#         
#         #ensure no recycling of wavefunction ! incompatible with some output
#         #for me in self.matrix_elements:
#         #    me.restore_original_wavefunctions()
# 
#         replace_dict['wavefunction_calls'] = "\n".join(\
#             self.helas_call_writer.get_wavefunction_calls(\
#             helas_objects.HelasWavefunctionList(wavefunctions)))
# 
#         replace_dict['amplitude_calls'] = "\n".join(\
#             self.helas_call_writer.get_amplitude_calls(amplitudes))
# 
#         if write:
#             file = self.read_template_file(self.process_wavefunction_template) % \
#                 replace_dict
#             return file
#         else:
#             return replace_dict
    
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for Pythia 8 .cc file"""

        ret_lines = []
        if self.single_helicities:
            
            template = "__device__ void calculate_wavefunctions(int ihel, const fptype* allmomenta,fptype &meHelSum \n#ifndef __CUDACC__\n                                , const int ievt\n#endif\n %(multi_channel)s                               )\n{"
            
            if self.include_multi_channel:
                info = {'multi_channel': self.multichannel_var}  
            else:
                info = {'multi_channel': ''}

            ret_lines.append( template % info)

            ret_lines.append(" using namespace MG5_%s;" % self.model_name)
            ret_lines.append("mgDebug( 0, __FUNCTION__ );")
            ret_lines.append("cxtype amp[1]; // was %i" % len(self.matrix_elements[0].get_all_amplitudes()))
            ret_lines.append("const int ncolor =  %i;" % len(color_amplitudes[0]))
            ret_lines.append("cxtype jamp[ncolor];")
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append("using namespace MG5_%s;" % self.model_name)


            multi_channel = None
            if self.include_multi_channel:
                if not self.support_multichannel:
                    raise Exception("link with madevent not supported")
                multi_channel = self.get_multi_channel_dictionary(self.matrix_elements[0].get('diagrams'), self.include_multi_channel)

            helas_calls = self.helas_call_writer.get_matrix_element_calls(\
                                                    self.matrix_elements[0],
                                                    color_amplitudes[0],
                                                    multi_channel_map = multi_channel
                                                    )
            assert len(self.matrix_elements) == 1 # how to handle if this is not true?

            self.couplings2order = self.helas_call_writer.couplings2order
            self.params2order = self.helas_call_writer.params2order
            nwavefuncs = self.matrix_elements[0].get_number_of_wavefunctions()
            ret_lines.append("cxtype w[nwf][nw6];")


            ret_lines += helas_calls
            #ret_lines.append(self.get_calculate_wavefunctions(\
            #    self.wavefunctions, self.amplitudes))
            #ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        to_add = []
        to_add.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)

    def write_process_h_file(self, writer):
        """Write the class definition (.h) file for the process"""
        
        replace_dict = super(OneProcessExporterGPU, self).write_process_h_file(False)
        try:
            replace_dict['helamps_h'] = open(pjoin(self.path, os.pardir, os.pardir,'src','HelAmps_%s.h' % self.model_name)).read()
        except FileNotFoundError:
            replace_dict['helamps_h'] = "\n#include \"../../src/HelAmps_%s.h\"" % self.model_name
        
        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
    
    def write_process_cc_file(self, writer):
        """Write the class member definition (.cc) file for the process
        described by matrix_element"""
        
                
        replace_dict = super(OneProcessExporterGPU, self).write_process_cc_file(False)
        #try:
        #    replace_dict['hel_amps_def'] = open(pjoin(self.path, os.pardir, os.pardir,'src','HelAmps_%s.cu' % self.model_name)).read()
        #except FileNotFoundError:
        replace_dict['hel_amps_def'] = "\n#include \"../../src/HelAmps_%s.cu\"" % self.model_name
            
        if writer:
            file = self.read_template_file(self.process_template_cc) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict


    def get_icolamp_lines(self, mapconfigs, matrix_element, num_matrix_element):
        """Return the ICOLAMP matrix, showing which JAMPs contribute to
        which configs (diagrams)."""

        ret_list = []

        booldict = {False: "false", True: "true"}

        # Only want to include leading color flows, so find max_Nc
        color_basis = matrix_element.get('color_basis')
        if not color_basis:
            # No color, so only one color factor. Simply write a ".true." 
            # for each config (i.e., each diagram with only 3 particle
            # vertices
            text = " {{%s}}" % ','.join(['true']* len(mapconfigs))
            return text


        # There is a color basis - create a list showing which JAMPs have
        # contributions to which configs
        # We don't want to include the power of Nc's which come from the potential
        # loop color trace (i.e. in the case of a closed fermion loop for example)
        # so we subtract it here when computing max_Nc
        max_Nc = max(sum([[(v[4]-v[5]) for v in val] for val in 
                                                      color_basis.values()],[]))

        # Create dictionary between diagram number and JAMP number
        diag_jamp = {}
        for ijamp, col_basis_elem in \
                enumerate(sorted(matrix_element.get('color_basis').keys())):
            for diag_tuple in matrix_element.get('color_basis')[col_basis_elem]:
                # Only use color flows with Nc == max_Nc. However, notice that
                # we don't want to include the Nc power coming from the loop
                # in this counting.
                if (diag_tuple[4]-diag_tuple[5]) == max_Nc:
                    diag_num = diag_tuple[0] + 1
                    # Add this JAMP number to this diag_num
                    diag_jamp[diag_num] = diag_jamp.setdefault(diag_num, []) + \
                                          [ijamp+1]
                #else:
                #    self.proc_characteristic['single_color'] = False

        colamps = ijamp + 1
        for iconfig, num_diag in enumerate(mapconfigs):  
            # mapconfigs can be a list or a dictionary.
            # In case of dictionary the num_diag will be the key of the dictionary.    
            if num_diag == 0:
                continue

            # List of True or False 
            bool_list = [(i + 1 in diag_jamp[num_diag]) for i in range(colamps)]
            # Add line
            ret_list.append("{%s};" % ','.join(["%s" % booldict[b] for b in bool_list]))



        return '{%s};' % ','.join(ret_list)


class OneProcessExporterMatchbox(OneProcessExporterCPP):
    """Class to take care of exporting a set of matrix elements to
    Matchbox format."""

    # Static variables (for inheritance)
    process_class_template = 'matchbox_class.inc'
    single_process_template = 'matchbox_matrix.inc'
    process_definition_template = 'matchbox_function_definitions.inc'

    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set external particle masses for this matrix element")

        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append("mME.push_back(pars->%s);" % part.get('mass'))
        return "\n".join(initProc_lines)


    def get_class_specific_definition_matrix(self, converter, matrix_element):
        """ """
        
        converter = super(OneProcessExporterMatchbox, self).get_class_specific_definition_matrix(converter, matrix_element)
        
        # T(....)
        converter['color_sting_lines'] = \
                                     self.get_color_string_lines(matrix_element)
                                     
        return converter
        
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for MAtchbox .cc file"""

        ret_lines = []
        if self.single_helicities:
            ret_lines.append(\
                "void %s::calculate_wavefunctions(const int perm[], const int hel[]){" % \
                class_name)
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append(self.get_calculate_wavefunctions(\
                self.wavefunctions, self.amplitudes))
            ret_lines.append(self.get_jamp_lines(color_amplitudes[0]))
            ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)


    def get_color_string_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return "\n".join(["static const double res[1][1] = {-1.};"])
        
        #start the real work
        color_denominators = matrix_element.get('color_matrix').\
                                                         get_line_denominators()
        matrix_strings = []
        my_cs = color.ColorString()
                
        for i_color in range(len(color_denominators)):
            # Then write the numerators for the matrix elements
            my_cs.from_immutable(sorted(matrix_element.get('color_basis').keys())[i_color])
            t_str=repr(my_cs)
            t_match=re.compile(r"(\w+)\(([\s\d+\,]*)\)")
            # from '1 T(2,4,1) Tr(4,5,6) Epsilon(5,3,2,1) T(1,2)' returns with findall:
            # [('T', '2,4,1'), ('Tr', '4,5,6'), ('Epsilon', '5,3,2,1'), ('T', '1,2')]
            all_matches = t_match.findall(t_str)
            tmp_color = [] 
            for match in all_matches:
                ctype, arg = match[0], [m.strip() for m in match[1].split(',')]
                if ctype not in ['T', 'Tr']:
                    raise self.ProcessExporterCPPError('Color Structure not handle by Matchbox')
                tmp_color.append(arg)
            #compute the maximal size of the vector
            nb_index = sum(len(o) for o in tmp_color)
            max_len = nb_index + (nb_index//2) -1
            #create the list with the 0 separator
            curr_color = tmp_color[0]
            for tcolor in tmp_color[1:]:
                curr_color += ['0'] + tcolor
            curr_color += ['0'] * (max_len- len(curr_color)) 
            #format the output
            matrix_strings.append('{%s}' % ','.join(curr_color))

        matrix_string = 'static const double res[%s][%s] = {%s};' % \
            (len(color_denominators), max_len, ",".join(matrix_strings))    

        return matrix_string


#===============================================================================
# ProcessExporterPythia8
#===============================================================================
class OneProcessExporterPythia8(OneProcessExporterCPP):
    """Class to take care of exporting a set of matrix elements to
    Pythia 8 format."""

    # Static variables (for inheritance)
    process_template_h = 'pythia8_process_h.inc'
    process_template_cc = 'pythia8_process_cc.inc'
    process_class_template = 'pythia8_process_class.inc'
    process_definition_template = 'pythia8_process_function_definitions.inc'
    process_wavefunction_template = 'pythia8_process_wavefunctions.inc'
    process_sigmaKin_function_template = 'pythia8_process_sigmaKin_function.inc'
    template_path = os.path.join(_file_path, 'iolibs', 'template_files', 'pythia8')     


    def __init__(self, *args, **opts):
        """Set process class name"""

        if 'version' in opts:
            self.version = opts['version']
            del opts['version']
        else:
            self.version='8.2'
        super(OneProcessExporterPythia8, self).__init__(*args, **opts)

        # Check if any processes are not 2->1,2,3
        for me in self.matrix_elements:
            if me.get_nexternal_ninitial() not in [(3,2),(4,2),(5,2)]:
                nex,nin = me.get_nexternal_ninitial()
                raise InvalidCmd("Pythia 8 can only handle 2->1,2,3 processes, not %d->%d" % \
                      (nin,nex-nin))
            
        self.process_class = self.process_name
        
    # Methods for generation of process files for Pythia 8

    def get_default_converter(self):
        
        replace_dict = {}       
        # Extract model name
        replace_dict['model_name'] = self.model_name
        if self.version =="8.2":
            replace_dict['include_prefix'] = 'Pythia8/'
        else:
            replace_dict['include_prefix'] = ''
            
        replace_dict['version'] = self.version
        
        return replace_dict
    #===========================================================================
    # Process export helper functions
    #===========================================================================
    def get_process_class_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = self.get_default_converter()


        # Extract process info lines for all processes
        process_lines = "\n".join([self.get_process_info_lines(me) for me in \
                                   self.matrix_elements])
        
        replace_dict['process_lines'] = process_lines

        # Extract number of external particles
        replace_dict['nfinal'] = self.nfinal

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        # Extract process definition
        process_definition = "%s (%s)" % (self.process_string,
                                          self.model_name)
        replace_dict['process_definition'] = process_definition

        process = self.processes[0]
        replace_dict['process_code'] = 10000 + \
                                       100*process.get('id') + \
                                       self.process_number

        replace_dict['inFlux'] = self.get_process_influx()

        replace_dict['id_masses'] = self.get_id_masses(process)
        replace_dict['resonances'] = self.get_resonance_lines()

        replace_dict['nexternal'] = self.nexternal
        replace_dict['nprocesses'] = self.nprocesses
        
        if self.single_helicities:
            replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          void calculate_wavefunctions(const int perm[], const int hel[]);
                          static const int nwavefuncs = %d;
                          std::complex<double> w[nwavefuncs][18];
                          static const int namplitudes = %d;
                          std::complex<double> amp[namplitudes];""" % \
                          (len(self.wavefunctions),
                           len(self.amplitudes.get_all_amplitudes()))
            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s();" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

        else:
            replace_dict['all_sigma_kin_definitions'] = \
                          "\n".join(["void sigmaKin_%s();" % \
                                     me.get('processes')[0].shell_string().\
                                     replace("0_", "") \
                                     for me in self.matrix_elements])
            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s(const int hel[]);" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

        if write:
            file = self.read_template_file('pythia8_process_class.inc') % replace_dict
            return file
        else:
            return replace_dict

    def get_process_function_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""


        replace_dict = self.get_default_converter()

        # Extract process info lines
        replace_dict['process_lines'] = \
                             "\n".join([self.get_process_info_lines(me) for \
                                        me in self.matrix_elements])

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        color_amplitudes = [me.get_color_amplitudes() for me in \
                            self.matrix_elements]

        replace_dict['initProc_lines'] = \
                                     self.get_initProc_lines(color_amplitudes)
        replace_dict['reset_jamp_lines'] = \
                                     self.get_reset_jamp_lines(color_amplitudes)
        

        replace_dict['sigmaKin_lines'], _ = \
                                     self.get_sigmaKin_lines(color_amplitudes)
        replace_dict['sigmaHat_lines'] = \
                                     self.get_sigmaHat_lines()

        replace_dict['setIdColAcol_lines'] = \
                                   self.get_setIdColAcol_lines(color_amplitudes)

        replace_dict['weightDecay_lines'] = \
                                       self.get_weightDecay_lines()    

        replace_dict['all_sigmaKin'] = \
                                  self.get_all_sigmaKin_lines(color_amplitudes,
                                                              self.process_name)
        if write:
            file = self.read_template_file('pythia8_process_function_definitions.inc') %\
               replace_dict
            return file
        else:
            return replace_dict

    def get_process_influx(self):
        """Return process file name for the process in matrix_element"""

        # Create a set with the pairs of incoming partons in definite order,
        # e.g.,  g g >... u d > ... d~ u > ... gives ([21,21], [1,2], [-2,1])
        beams = set([tuple(sorted([process.get('legs')[0].get('id'),
                                   process.get('legs')[1].get('id')])) \
                          for process in self.processes])

        # Define a number of useful sets
        antiquarks = list(range(-1, -6, -1))
        quarks = list(range(1,6))
        antileptons = list(range(-11, -17, -1))
        leptons = list(range(11, 17, 1))
        allquarks = antiquarks + quarks
        antifermions = antiquarks + antileptons
        fermions = quarks + leptons
        allfermions = allquarks + antileptons + leptons
        downfermions = list(range(-2, -5, -2)) + list(range(-1, -5, -2)) + \
                       list(range(-12, -17, -2)) + list(range(-11, -17, -2)) 
        upfermions = list(range(1, 5, 2)) + list(range(2, 5, 2)) + \
                     list(range(11, 17, 2)) + list(range(12, 17, 2))

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

        raise InvalidCmd('Pythia 8 cannot handle incoming flavors %s' %\
                             repr(beams))

        return 

    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, filename):
        """Open a template file and return the contents."""
             
        try:
            return super(OneProcessExporterPythia8, cls).read_template_file(filename)     
        except:
            return super(OneProcessExporterPythia8, cls).read_template_file(filename, classpath=True)

        
    def get_id_masses(self, process):
        """Return the lines which define the ids for the final state particles,
        for the Pythia phase space"""

        if self.nfinal == 1:
            return ""
        
        mass_strings = []
        for i in range(2, len(process.get_legs_with_decays())):
            if self.model.get_particle(process.get_legs_with_decays()[i].get('id')).\
                   get('mass') not in  ['zero', 'ZERO']:
                mass_strings.append("int id%dMass() const {return %d;}" % \
                                (i + 1, abs(process.get_legs_with_decays()[i].get('id'))))

        return "\n".join(mass_strings)

    def get_resonance_lines(self):
        """Return the lines which define the ids for intermediate resonances
        for the Pythia phase space"""

        if self.nfinal == 1:
            return "virtual int resonanceA() const {return %d;}" % \
                           abs(self.processes[0].get('legs')[2].get('id'))
        
        res_strings = []
        res_letters = ['A', 'B']

        sids, singleres, schannel = self.get_resonances()

        for i, sid in enumerate(sids[:2]):
            res_strings.append("virtual int resonance%s() const {return %d;}"\
                                % (res_letters[i], sid))

        if schannel:
           res_strings.append("virtual bool isSChannel() const {return true;}")

        if singleres != 0:
            res_strings.append("virtual int idSChannel() const {return %d;}" \
                               % singleres)
            
        return "\n".join(res_strings)

    def get_resonances(self):
        """Return the PIDs for any resonances in 2->2 and 2->3 processes."""

        model = self.matrix_elements[0].get('processes')[0].get('model')
        new_pdg = model.get_first_non_pdg()
        # Get a list of all resonant s-channel contributions
        diagrams = sum([me.get('diagrams') for me in self.matrix_elements], [])
        resonances = []
        no_t_channels = True
        final_s_channels = []
        for diagram in diagrams:
            schannels, tchannels = diagram.get('amplitudes')[0].\
                                   get_s_and_t_channels(self.ninitial, model,
                                                        new_pdg)
            for schannel in schannels:
                sid = schannel.get('legs')[-1].get('id')
                part = self.model.get_particle(sid)
                if part:
                    width = self.model.get_particle(sid).get('width')
                    if width.lower() != 'zero':
                        # Only care about absolute value of resonance PIDs:
                        resonances.append(abs(sid))
                    else:
                        sid = 0
                    if len(tchannels) == 1 and schannel == schannels[-1]:
                        final_s_channels.append(abs(sid))

            if len(tchannels) > 1:
                # There are t-channel diagrams
                no_t_channels = False
            
        resonance_set = set(resonances)
        final_s_set = set(final_s_channels)

        singleres = 0
        # singleres is set if all diagrams have the same final resonance
        if len(final_s_channels) == len(diagrams) and len(final_s_set) == 1 \
                and final_s_channels[0] != 0:
            singleres = final_s_channels[0]

        resonance_set = misc.make_unique([pid for pid in resonance_set])

        # schannel is True if all diagrams are pure s-channel and there are
        # no QCD vertices
        schannel = no_t_channels and \
                   not any(['QCD' in d.calculate_orders() for d in diagrams])

        return resonance_set, singleres, schannel

    def get_initProc_lines(self, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set massive/massless matrix elements for c/b/mu/tau")
        # Add lines to set c/b/tau/mu kinematics massive/massless
        if not self.model.get_particle(4) or \
               self.model.get_particle(4).get('mass').lower() == 'zero':
            cMassiveME = "0."
        else:
            cMassiveME = "particleDataPtr->m0(4)"
        initProc_lines.append("mcME = %s;" % cMassiveME)
        if not self.model.get_particle(5) or \
               self.model.get_particle(5).get('mass').lower() == 'zero':
            bMassiveME = "0."
        else:
            bMassiveME = "particleDataPtr->m0(5)"
        initProc_lines.append("mbME = %s;" % bMassiveME)
        if not self.model.get_particle(13) or \
               self.model.get_particle(13).get('mass').lower() == 'zero':
            muMassiveME = "0."
        else:
            muMassiveME = "particleDataPtr->m0(13)"
        initProc_lines.append("mmuME = %s;" % muMassiveME)
        if not self.model.get_particle(15) or \
               self.model.get_particle(15).get('mass').lower() == 'zero':
            tauMassiveME = "0."
        else:
            tauMassiveME = "particleDataPtr->m0(15)"
        initProc_lines.append("mtauME = %s;" % tauMassiveME)
            
        for i, me in enumerate(self.matrix_elements):
            initProc_lines.append("jamp2[%d] = new double[%d];" % \
                                  (i, len(color_amplitudes[i])))

        return "\n".join(initProc_lines)

    def get_setIdColAcol_lines(self, color_amplitudes):
        """Generate lines to set final-state id and color info for process"""

        res_lines = []

        # Create a set with the pairs of incoming partons
        beams = set([(process.get('legs')[0].get('id'),
                      process.get('legs')[1].get('id')) \
                     for process in self.processes])
        beams = sorted(list(beams))
        # Now write a selection routine for final state ids
        for ibeam, beam_parts in enumerate(beams):
            if ibeam == 0:
                res_lines.append("if(id1 == %d && id2 == %d){" % beam_parts)
            else:
                res_lines.append("else if(id1 == %d && id2 == %d){" % beam_parts)            
            # Pick out all processes with this beam pair
            beam_processes = [(i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[0].get('id'),
                                process.get('legs')[1].get('id')) \
                               for process in me.get('processes')]]
            # Pick out all mirror processes for this beam pair
            beam_mirror_processes = []
            if beam_parts[0] != beam_parts[1]:
                beam_mirror_processes = [(i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[1].get('id'),
                                process.get('legs')[0].get('id')) \
                               for process in me.get('processes')]]

            final_id_list = []
            final_mirror_id_list = []
            for (i, me) in beam_processes:
                final_id_list.extend([tuple([l.get('id') for l in \
                                             proc.get_legs_with_decays() if l.get('state')]) \
                                      for proc in me.get('processes') \
                                      if beam_parts == \
                                      (proc.get('legs')[0].get('id'),
                                       proc.get('legs')[1].get('id'))])
            for (i, me) in beam_mirror_processes:
                final_mirror_id_list.extend([tuple([l.get('id') for l in \
                                             proc.get_legs_with_decays() if l.get('state')]) \
                                      for proc in me.get_mirror_processes() \
                                      if beam_parts == \
                                      (proc.get('legs')[0].get('id'),
                                       proc.get('legs')[1].get('id'))])
            final_id_list = set(final_id_list)
            final_mirror_id_list = set(final_mirror_id_list)

            if final_id_list and final_mirror_id_list or \
               not final_id_list and not final_mirror_id_list:
                raise self.ProcessExporterCPPError("Missing processes, or both process and mirror process")


            ncombs = len(final_id_list)+len(final_mirror_id_list)

            res_lines.append("// Pick one of the flavor combinations %s" % \
                             ", ".join([repr(ids) for ids in final_id_list]))

            me_weight = []
            for final_ids in final_id_list:
                items = [(i, len([ p for p in me.get('processes') \
                             if [l.get('id') for l in \
                             p.get_legs_with_decays()] == \
                             list(beam_parts) + list(final_ids)])) \
                       for (i, me) in beam_processes]
                me_weight.append("+".join(["matrix_element[%i]*%i" % (i, l) for\
                                           (i, l) in items if l > 0]).\
                                 replace('*1', ''))
                if any([l>1 for (i, l) in items]):
                    raise self.ProcessExporterCPPError("More than one process with identical " + \
                          "external particles is not supported")

            for final_ids in final_mirror_id_list:
                items = [(i, len([ p for p in me.get_mirror_processes() \
                             if [l.get('id') for l in p.get_legs_with_decays()] == \
                             list(beam_parts) + list(final_ids)])) \
                       for (i, me) in beam_mirror_processes]
                me_weight.append("+".join(["matrix_element[%i]*%i" % \
                                           (i+len(self.matrix_elements), l) for\
                                           (i, l) in items if l > 0]).\
                                 replace('*1', ''))
                if any([l>1 for (i, l) in items]):
                    raise self.ProcessExporterCPPError("More than one process with identical " + \
                          "external particles is not supported")

            if final_id_list:
                res_lines.append("int flavors[%d][%d] = {%s};" % \
                                 (ncombs, self.nfinal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in ids]) + "}" for ids \
                                            in final_id_list])))
            elif final_mirror_id_list:
                res_lines.append("int flavors[%d][%d] = {%s};" % \
                                 (ncombs, self.nfinal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in ids]) + "}" for ids \
                                            in final_mirror_id_list])))
            res_lines.append("vector<double> probs;")
            res_lines.append("double sum = %s;" % "+".join(me_weight))
            for me in me_weight:
                res_lines.append("probs.push_back(%s/sum);" % me)
            res_lines.append("int choice = rndmPtr->pick(probs);")
            for i in range(self.nfinal):
                res_lines.append("id%d = flavors[choice][%d];" % (i+3, i))

            res_lines.append("}")

        res_lines.append("setId(%s);" % ",".join(["id%d" % i for i in \
                                                 range(1, self.nexternal + 1)]))

        # Now write a selection routine for color flows

        # We need separate selection for each flavor combination,
        # since the different processes might have different color
        # structures.
        
        # Here goes the color connections corresponding to the JAMPs
        # Only one output, for the first subproc!

        res_lines.append("// Pick color flow")

        res_lines.append("int ncolor[%d] = {%s};" % \
                         (len(color_amplitudes),
                          ",".join([str(len(colamp)) for colamp in \
                                    color_amplitudes])))
                                                 

        for ime, me in enumerate(self.matrix_elements):

            res_lines.append("if((%s)){" % \
                                 ")||(".join(["&&".join(["id%d == %d" % \
                                            (i+1, l.get('id')) for (i, l) in \
                                            enumerate(p.get_legs_with_decays())])\
                                           for p in me.get('processes')]))
            if ime > 0:
                res_lines[-1] = "else " + res_lines[-1]

            proc = me.get('processes')[0]
            if not me.get('color_basis'):
                # If no color basis, just output trivial color flow
                res_lines.append("setColAcol(%s);" % ",".join(["0"]*2*self.nfinal))
            else:
                # Else, build a color representation dictionnary
                repr_dict = {}
                legs = proc.get_legs_with_decays()
                for l in legs:
                    repr_dict[l.get('number')] = \
                        proc.get('model').get_particle(l.get('id')).get_color()
                # Get the list of color flows
                color_flow_list = \
                    me.get('color_basis').color_flow_decomposition(\
                                                      repr_dict, self.ninitial)
                # Select a color flow
                ncolor = len(me.get('color_basis'))
                res_lines.append("""vector<double> probs;
                  double sum = %s;
                  for(int i=0;i<ncolor[%i];i++)
                  probs.push_back(jamp2[%i][i]/sum);
                  int ic = rndmPtr->pick(probs);""" % \
                                 ("+".join(["jamp2[%d][%d]" % (ime, i) for i \
                                            in range(ncolor)]), ime, ime))

                color_flows = []
                for color_flow_dict in color_flow_list:
                    color_flows.append([int(fmod(color_flow_dict[l.get('number')][i], 500)) \
                                        for (l,i) in itertools.product(legs, [0,1])])

                # Write out colors for the selected color flow
                res_lines.append("static int colors[%d][%d] = {%s};" % \
                                 (ncolor, 2 * self.nexternal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in flows]) + "}" for flows \
                                            in color_flows])))

                res_lines.append("setColAcol(%s);" % \
                                 ",".join(["colors[ic][%d]" % i for i in \
                                          range(2 * self.nexternal)]))
            res_lines.append('}')

        # Same thing but for mirror processes
        for ime, me in enumerate(self.matrix_elements):
            if not me.get('has_mirror_process'):
                continue
            res_lines.append("else if((%s)){" % \
                                 ")||(".join(["&&".join(["id%d == %d" % \
                                            (i+1, l.get('id')) for (i, l) in \
                                            enumerate(p.get_legs_with_decays())])\
                                           for p in me.get_mirror_processes()]))

            proc = me.get('processes')[0]
            if not me.get('color_basis'):
                # If no color basis, just output trivial color flow
                res_lines.append("setColAcol(%s);" % ",".join(["0"]*2*self.nfinal))
            else:
                # Else, build a color representation dictionnary
                repr_dict = {}
                legs = proc.get_legs_with_decays()
                legs[0:2] = [legs[1],legs[0]]
                for l in legs:
                    repr_dict[l.get('number')] = \
                        proc.get('model').get_particle(l.get('id')).get_color()
                # Get the list of color flows
                color_flow_list = \
                    me.get('color_basis').color_flow_decomposition(\
                                                      repr_dict, self.ninitial)
                # Select a color flow
                ncolor = len(me.get('color_basis'))
                res_lines.append("""vector<double> probs;
                  double sum = %s;
                  for(int i=0;i<ncolor[%i];i++)
                  probs.push_back(jamp2[%i][i]/sum);
                  int ic = rndmPtr->pick(probs);""" % \
                                 ("+".join(["jamp2[%d][%d]" % (ime, i) for i \
                                            in range(ncolor)]), ime, ime))

                color_flows = []
                for color_flow_dict in color_flow_list:
                    color_flows.append([color_flow_dict[l.get('number')][i] % 500 \
                                        for (l,i) in itertools.product(legs, [0,1])])

                # Write out colors for the selected color flow
                res_lines.append("static int colors[%d][%d] = {%s};" % \
                                 (ncolor, 2 * self.nexternal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in flows]) + "}" for flows \
                                            in color_flows])))

                res_lines.append("setColAcol(%s);" % \
                                 ",".join(["colors[ic][%d]" % i for i in \
                                          range(2 * self.nexternal)]))
            res_lines.append('}')

        return "\n".join(res_lines)


    def get_weightDecay_lines(self):
        """Get weightDecay_lines for function definition for Pythia 8 .cc file"""

        weightDecay_lines = "// Just use isotropic decay (default)\n"
        weightDecay_lines += "return 1.;"

        return weightDecay_lines

    #===============================================================================
    # Routines to export/output UFO models in Pythia8 format
    #===============================================================================
    def convert_model_to_pythia8(self, model, pythia_dir):
        """Create a full valid Pythia 8 model from an MG5 model (coming from UFO)"""
    
        if not os.path.isfile(os.path.join(pythia_dir, 'include', 'Pythia.h'))\
           and not os.path.isfile(os.path.join(pythia_dir, 'include', 'Pythia8', 'Pythia.h')):
            logger.warning('Directory %s is not a valid Pythia 8 main dir.' % pythia_dir)
    
        # create the model parameter files
        model_builder = UFOModelConverterPythia8(model, pythia_dir, replace_dict=self.get_default_converter())
        model_builder.cc_file_dir = "Processes_" + model_builder.model_name
        model_builder.include_dir = model_builder.cc_file_dir
    
        model_builder.write_files()
        # Write makefile
        model_builder.write_makefile()
        # Write param_card
        model_builder.write_param_card()
        return model_builder.model_name, model_builder.cc_file_dir


#===============================================================================
# ProcessExporterCPP
#===============================================================================
class ProcessExporterCPP(VirtualExporter):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    grouped_mode = False
    exporter = 'cpp'

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True
                        }
    
    oneprocessclass = OneProcessExporterCPP
    s= _file_path + 'iolibs/template_files/'
    from_template = {'src': [s+'rambo.h', s+'rambo.cc', s+'read_slha.h', s+'read_slha.cc'],
                     'SubProcesses': [s+'check_sa.cpp']}
    to_link_in_P = ['check_sa.cpp', 'Makefile']
    template_src_make = pjoin(_file_path, 'iolibs', 'template_files','Makefile_sa_cpp_src')
    template_Sub_make = pjoin(_file_path, 'iolibs', 'template_files','Makefile_sa_cpp_sp') 
    create_model_class =  UFOModelConverterCPP
    

    def __init__(self, dir_path = "", opt=None):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = MG5DIR
        self.dir_path = dir_path
        self.model = None

        self.opt = dict(self.default_opt)
        if opt:
            self.opt.update(opt)
        
        #place holder to pass information to the run_interface
        self.proc_characteristic = banner_mod.ProcCharacteristic()    

    def copy_template(self, model):
        """Prepare export_dir as standalone_cpp directory, including:
        src (for RAMBO, model and ALOHA files + makefile)
        lib (with compiled libraries from src)
        SubProcesses (with check_sa.cpp + makefile and Pxxxxx directories)
        """

        try:
            os.mkdir(self.dir_path)
        except os.error as error:
            logger.warning(error.strerror + " " + self.dir_path)
        
        with misc.chdir(self.dir_path):
            logger.info('Creating subdirectories in directory %s' % self.dir_path)

            for d in ['src', 'lib', 'Cards', 'SubProcesses']:
                try:
                    os.mkdir(d)
                except os.error as error:
                    logger.warning(error.strerror + " " + self.dir_path)
    
            # Write param_card
            open(os.path.join("Cards","param_card.dat"), 'w').write(\
                                                       model.write_param_card())

    
            # Copy the needed src files
            for key in self.from_template:
                for f in self.from_template[key]:
                    cp(f, key)

            if self.template_src_make:
                # Copy src Makefile
                makefile = self.read_template_file(self.template_src_make) % \
                               {'model': self.get_model_name(model.get('name')),
                                'cpp_compiler': self.opt['cpp_compiler'] if self.opt['cpp_compiler'] else 'g++'}
                open(os.path.join('src', 'Makefile'), 'w').write(makefile)

            if self.template_Sub_make:
                # Copy SubProcesses Makefile
                makefile = self.read_template_file(self.template_Sub_make) % \
                                        {'model': self.get_model_name(model.get('name')),
                                         'cpp_compiler': self.opt['cpp_compiler'] if self.opt['cpp_compiler'] else 'g++'}
                open(os.path.join('SubProcesses', 'Makefile'), 'w').write(makefile)

    #===========================================================================
    # Helper functions
    #===========================================================================
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""
            
        return False, matrix_element



    def convert_model(self, model, wanted_lorentz = [],
                         wanted_couplings = []):
        # create the model parameter files
        model_builder = self.create_model_class(model,
                                         os.path.join(self.dir_path, 'src'),
                                         wanted_lorentz,
                                         wanted_couplings)
        model_builder.write_files()
    
    def compile_model(self):
        make_model_cpp(self.dir_path)
    
    @classmethod
    def read_template_file(cls, *args, **opts):
        """Open a template file and return the contents."""
         
        return cls.oneprocessclass.read_template_file(*args, **opts) 
        
    #===============================================================================
    # generate_subprocess_directory
    #===============================================================================
    def generate_subprocess_directory(self, matrix_element, cpp_helas_call_writer,
                                      proc_number=None):
        """Generate the Pxxxxx directory for a subprocess in C++ standalone,
        including the necessary .h and .cc files"""

        
        process_exporter_cpp = self.oneprocessclass(matrix_element,cpp_helas_call_writer)

        
        # Create the directory PN_xx_xxxxx in the specified path
        dirpath = pjoin(self.dir_path, 'SubProcesses', "P%d_%s" % (process_exporter_cpp.process_number, 
                                             process_exporter_cpp.process_name))
        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)
    
        with misc.chdir(dirpath):
            logger.info('Creating files in directory %s' % dirpath)
            process_exporter_cpp.path = dirpath
            # Create the process .h and .cc files
            process_exporter_cpp.generate_process_files()
            for file in self.to_link_in_P:
                ln('../%s' % file) 
        return

    @staticmethod
    def get_model_name(name):
        """Replace - with _, + with _plus_ in a model name."""

        name = name.replace('-', '_')
        name = name.replace('+', '_plus_')
        return name
    
    def finalize(self, *args, **opts):
        """ """
        self.compile_model()
        pass

class ProcessExporterMatchbox(ProcessExporterCPP):
    oneprocessclass = OneProcessExporterMatchbox

class ProcessExporterPythia8(ProcessExporterCPP):
    oneprocessclass = OneProcessExporterPythia8
    grouped_mode = 'madevent'
     
    #===============================================================================
    # generate_process_files_pythia8
    #===============================================================================
    def generate_process_directory(self, multi_matrix_element, cpp_helas_call_writer,
                                   process_string = "",
                                   process_number = 0,
                                   version='8.2'):

        """Generate the .h and .cc files needed for Pythia 8, for the
        processes described by multi_matrix_element"""

        process_exporter_pythia8 = OneProcessExporterPythia8(multi_matrix_element,
                                                      cpp_helas_call_writer,
                                                      process_string,
                                                      process_number,
                                                      self.dir_path,
                                                      version=version)
    
        # Set process directory
        model = process_exporter_pythia8.model
        model_name = process_exporter_pythia8.model_name
        process_exporter_pythia8.process_dir = \
                       'Processes_%(model)s' % {'model': \
                        model_name}
        process_exporter_pythia8.include_dir = process_exporter_pythia8.process_dir
        process_exporter_pythia8.generate_process_files()
        return process_exporter_pythia8

    #===============================================================================
    # generate_example_file_pythia8
    #===============================================================================
    @staticmethod
    def generate_example_file_pythia8(path,
                                       model_path,
                                       process_names,
                                       exporter,
                                       main_file_name = "",
                                       example_dir = "examples",
                                       version="8.2"):
        """Generate the main_model_name.cc file and Makefile in the examples dir"""
    
        filepath = os.path.join(path, example_dir)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
    
        replace_dict = {}
    
        # Extract version number and date from VERSION file
        info_lines = get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines
    
        # Extract model name
        replace_dict['model_name'] = exporter.model_name
    
        # Extract include line
        replace_dict['include_lines'] = \
                              "\n".join(["#include \"%s.h\"" % proc_name \
                                         for proc_name in process_names])
    
        # Extract setSigmaPtr line
        replace_dict['sigma_pointer_lines'] = \
               "\n".join(["pythia.setSigmaPtr(new %s());" % proc_name \
                         for proc_name in process_names])
    
        # Extract param_card path
        replace_dict['param_card'] = os.path.join(os.path.pardir,model_path,
                                                  "param_card_%s.dat" % \
                                                  exporter.model_name)
    
        # Create the example main file
        if version =="8.2":
            template_path = 'pythia8.2_main_example_cc.inc'
            makefile_path = 'pythia8.2_main_makefile.inc'
            replace_dict['include_prefix'] = 'Pythia8/'
        else:
            template_path = 'pythia8_main_example_cc.inc'
            makefile_path = 'pythia8_main_makefile.inc'
            replace_dict['include_prefix'] = ''
        
        
        file = ProcessExporterPythia8.read_template_file(template_path) % \
               replace_dict
    
        if not main_file_name:
            num = 1
            while os.path.exists(os.path.join(filepath,
                                        'main_%s_%i.cc' % (exporter.model_name, num))) or \
                  os.path.exists(os.path.join(filepath,
                                        'main_%s_%i' % (exporter.model_name, num))):
                num += 1
            main_file_name = str(num)
    
        main_file = 'main_%s_%s' % (exporter.model_name,
                                    main_file_name)
    
        main_filename = os.path.join(filepath, main_file + '.cc')
    
        # Write the file
        writers.CPPWriter(main_filename).writelines(file)
    
        replace_dict = {}
    
        # Extract version number and date from VERSION file
        replace_dict['info_lines'] = get_mg5_info_lines()
    
        replace_dict['main_file'] = main_file
    
        replace_dict['process_dir'] = model_path
    
        replace_dict['include_dir'] = exporter.include_dir
    
        # Create the makefile
        file = ProcessExporterPythia8.read_template_file(makefile_path) % replace_dict
    
        make_filename = os.path.join(filepath, 'Makefile_%s_%s' % \
                                (exporter.model_name, main_file_name))
    
        # Write the file
        open(make_filename, 'w').write(file)
    
        logger.info("Created files %s and %s in directory %s" \
                    % (os.path.split(main_filename)[-1],
                       os.path.split(make_filename)[-1],
                       os.path.split(make_filename)[0]))
        return main_file, make_filename

    def convert_model(self,*args,**opts):
        pass
    def finalize(self, *args, **opts):
        pass
  
def get_mg5_info_lines():
    """Return info lines for MG5, suitable to place at beginning of
    Fortran files"""

    info = misc.get_pkg_info()
    info_lines = ""
    if info and 'version' in info and  'date' in info:
        info_lines = "#  MadGraph5_aMC@NLO v. %s, %s\n" % \
                     (info['version'], info['date'])
        info_lines = info_lines + \
                     "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                     "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"
    else:
        info_lines = "#  MadGraph5_aMC@NLO\n" + \
                     "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                     "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"        

    return info_lines




#===============================================================================
# ProcessExporterGPU
#===============================================================================
class ProcessExporterGPU(ProcessExporterCPP):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    grouped_mode = False
    exporter = 'gpu'

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True
                        }
    
    oneprocessclass = OneProcessExporterGPU
    s= _file_path + 'iolibs/template_files/'
    from_template = {'src': [s+'gpu/rambo.h', s+'gpu/rambo.cc', s+'read_slha.h', s+'read_slha.cc',
                             s+'gpu/mgOnGpuTypes.h', s+'gpu/grambo.cu'],
                    'SubProcesses': [s+'gpu/timer.h', s+'gpu/Makefile', s+'gpu/nvtx.h',
                                     s+'gpu/nvtx.h', s+'gpu/check.cc',
                                     s+'gpu/timermap.h', s+'gpu/profile.sh',
                                     s+'gpu/perf.py', s+ 'gpu/Memory.h', s + 'gpu/runTest.cc']}
    to_link_in_P = ['Makefile', 'timer.h', 'timermap.h', 'nvtx.h', 'perf.py', 'Memory.h', 'runTest.cc']

    template_src_make = pjoin(_file_path, 'iolibs', 'template_files','gpu','Makefile_src')
    template_Sub_make = pjoin(_file_path, 'iolibs', 'template_files','gpu','Makefile')
    create_model_class =  UFOModelConverterGPU
    
    def compile_model(self):
        return 

class UFOModelConverterCPP(object):
    """ A converter of the UFO-MG5 Model to the C++ format """

    # Static variables (for inheritance)
    output_name = 'C++ Standalone'
    namespace = 'MG5'

    # Dictionary from Python type to C++ type
    type_dict = {"real": "double",
                 "complex": "std::complex<double>"}

    # Regular expressions for cleaning of lines from Aloha files
    compiler_option_re = re.compile(r'^#\w')
    namespace_re = re.compile('^using namespace')

    slha_to_depend = {('SMINPUTS', (3,)): ('aS',),
                      ('SMINPUTS', (1,)): ('aEM',)}

    # Template files to use
    include_dir = '.'
    cc_file_dir = '.'
    param_template_h = 'cpp_model_parameters_h.inc'
    param_template_cc = 'cpp_model_parameters_cc.inc'
    aloha_template_h = 'cpp_hel_amps_h.inc'
    aloha_template_cc = 'cpp_hel_amps_cc.inc'

    copy_include_files = []
    copy_cc_files = []

    def __init__(self, model, output_path, wanted_lorentz = [],
                 wanted_couplings = [], replace_dict={}):
        """ initialization of the objects """

        self.model = model
        self.model_name = ProcessExporterCPP.get_model_name(model['name'])

        self.dir_path = output_path
        self.default_replace_dict = dict(replace_dict)
        # List of needed ALOHA routines
        self.wanted_lorentz = wanted_lorentz

        # For dependent couplings, only want to update the ones
        # actually used in each process. For other couplings and
        # parameters, just need a list of all.
        self.coups_dep = {}    # name -> base_objects.ModelVariable
        self.coups_indep = []  # base_objects.ModelVariable
        self.params_dep = []   # base_objects.ModelVariable
        self.params_indep = [] # base_objects.ModelVariable
        self.p_to_cpp = parsers.UFOExpressionParserCPP()

        # Prepare parameters and couplings for writeout in C++
        self.prepare_parameters()
        self.prepare_couplings(wanted_couplings)

    def write_files(self):
        """Create all necessary files"""

        # Write Helas Routines
        self.write_aloha_routines()

        # Write parameter (and coupling) class files
        self.write_parameter_class_files()

    # Routines for preparing parameters and couplings from the model

    def prepare_parameters(self):
        """Extract the parameters from the model, and store them in
        the two lists params_indep and params_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['parameters'].keys())
        keys.sort(key=len)
        params_ext = []
        for key in keys:
            if key == ('external',):
                params_ext += [p for p in self.model['parameters'][key] if p.name]
            elif 'aS' in key:
                for p in self.model['parameters'][key]:
                    self.params_dep.append(base_objects.ModelVariable(p.name,
                                              p.name + " = " + \
                                              self.p_to_cpp.parse(p.expr) + ";",
                                              p.type,
                                              p.depend))
            else:
                for p in self.model['parameters'][key]:
                    if p.name == 'ZERO':
                        continue
                    self.params_indep.append(base_objects.ModelVariable(p.name,
                                              p.name + " = " + \
                                              self.p_to_cpp.parse(p.expr) + ";",
                                              p.type,
                                              p.depend))

        # For external parameters, want to read off the SLHA block code
        while params_ext:
            param = params_ext.pop(0)
            # Read value from the slha variable
            expression = ""
            assert param.value.imag == 0
            if len(param.lhacode) == 1:
                expression = "%s = slha.get_block_entry(\"%s\", %d, %e);" % \
                             (param.name, param.lhablock.lower(),
                              param.lhacode[0], param.value.real)
            elif len(param.lhacode) == 2:
                expression = "indices[0] = %d;\nindices[1] = %d;\n" % \
                             (param.lhacode[0], param.lhacode[1])
                expression += "%s = slha.get_block_entry(\"%s\", indices, %e);" \
                              % (param.name, param.lhablock.lower(), param.value.real)
            else:
                raise MadGraph5Error("Only support for SLHA blocks with 1 or 2 indices")
            self.params_indep.insert(0,
                                   base_objects.ModelVariable(param.name,
                                                   expression,
                                                              'real'))
            
    def prepare_couplings(self, wanted_couplings = []):
        """Extract the couplings from the model, and store them in
        the two lists coups_indep and coups_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['couplings'].keys())
        keys.sort(key=len)
        for key, coup_list in self.model['couplings'].items():
            if "aS" in key:
                for c in coup_list:
                    if not wanted_couplings or c.name in wanted_couplings:
                        self.coups_dep[c.name] = base_objects.ModelVariable(\
                                                                   c.name,
                                                                   c.expr,
                                                                   c.type,
                                                                   c.depend)
            else:
                for c in coup_list:
                    if not wanted_couplings or c.name in wanted_couplings:
                        self.coups_indep.append(base_objects.ModelVariable(\
                                                                   c.name,
                                                                   c.expr,
                                                                   c.type,
                                                                   c.depend))

        # Convert coupling expressions from Python to C++
        for coup in list(self.coups_dep.values()) + self.coups_indep:
            coup.expr = coup.name + " = " + self.p_to_cpp.parse(coup.expr) + ";"

    # Routines for writing the parameter files

    def write_parameter_class_files(self):
        """Generate the parameters_model.h and parameters_model.cc
        files, which have the parameters and couplings for the model."""

        if not os.path.isdir(os.path.join(self.dir_path, self.include_dir)):
            os.makedirs(os.path.join(self.dir_path, self.include_dir))
        if not os.path.isdir(os.path.join(self.dir_path, self.cc_file_dir)):
            os.makedirs(os.path.join(self.dir_path, self.cc_file_dir))

        parameter_h_file = os.path.join(self.dir_path, self.include_dir,
                                    'Parameters_%s.h' % self.model_name)
        parameter_cc_file = os.path.join(self.dir_path, self.cc_file_dir,
                                     'Parameters_%s.cc' % self.model_name)

        file_h, file_cc = self.generate_parameters_class_files()

        # Write the files
        writers.CPPWriter(parameter_h_file).writelines(file_h)
        writers.CPPWriter(parameter_cc_file).writelines(file_cc)

        # Copy additional needed files
        for copy_file in self.copy_include_files:
            shutil.copy(os.path.join(_file_path, 'iolibs',
                                         'template_files',copy_file),
                        os.path.join(self.dir_path, self.include_dir))
        # Copy additional needed files
        for copy_file in self.copy_cc_files:
            shutil.copy(os.path.join(_file_path, 'iolibs',
                                         'template_files',copy_file),
                        os.path.join(self.dir_path, self.cc_file_dir))

        logger.info("Created files %s and %s in directory" \
                    % (os.path.split(parameter_h_file)[-1],
                       os.path.split(parameter_cc_file)[-1]))
        logger.info("%s and %s" % \
                    (os.path.split(parameter_h_file)[0],
                     os.path.split(parameter_cc_file)[0]))

    def generate_parameters_class_files(self):
        """Create the content of the Parameters_model.h and .cc files"""

        replace_dict = self.default_replace_dict

        replace_dict['info_lines'] = get_mg5_info_lines()
        replace_dict['model_name'] = self.model_name

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
                                   self.write_parameters(list(self.coups_dep.values()))

        replace_dict['set_independent_parameters'] = \
                               self.write_set_parameters(self.params_indep)
        replace_dict['set_independent_couplings'] = \
                               self.write_set_parameters(self.coups_indep)
        replace_dict['set_dependent_parameters'] = \
                               self.write_set_parameters(self.params_dep)
        replace_dict['set_dependent_couplings'] = \
                               self.write_set_parameters(list(self.coups_dep.values()))

        replace_dict['print_independent_parameters'] = \
                               self.write_print_parameters(self.params_indep)
        replace_dict['print_independent_couplings'] = \
                               self.write_print_parameters(self.coups_indep)
        replace_dict['print_dependent_parameters'] = \
                               self.write_print_parameters(self.params_dep)
        replace_dict['print_dependent_couplings'] = \
                               self.write_print_parameters(list(self.coups_dep.values()))

        if 'include_prefix' not in replace_dict:
            replace_dict['include_prefix'] = ''


        file_h = self.read_template_file(self.param_template_h) % \
                 replace_dict
        file_cc = self.read_template_file(self.param_template_cc) % \
                  replace_dict
        
        return file_h, file_cc

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
            res_strings.append("%s" % param.expr)

        # Correct width sign for Majorana particles (where the width
        # and mass need to have the same sign)        
        for particle in self.model.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero':
                res_strings.append("if (%s < 0)" % particle.get('mass'))
                res_strings.append("%(width)s = -abs(%(width)s);" % \
                                   {"width": particle.get('width')})

        return "\n".join(res_strings)

    def write_print_parameters(self, params):
        """Write out the lines of independent parameters"""

        # For each parameter, write name = expr;

        res_strings = []
        for param in params:
            res_strings.append("cout << setw(20) << \"%s \" << \"= \" << setiosflags(ios::scientific) << setw(10) << %s << endl;" % (param.name, param.name))

        return "\n".join(res_strings)

    # Routines for writing the ALOHA files

    def write_aloha_routines(self):
        """Generate the hel_amps_model.h and hel_amps_model.cc files, which
        have the complete set of generalized Helas routines for the model"""
        
        if not os.path.isdir(os.path.join(self.dir_path, self.include_dir)):
            os.makedirs(os.path.join(self.dir_path, self.include_dir))
        if not os.path.isdir(os.path.join(self.dir_path, self.cc_file_dir)):
            os.makedirs(os.path.join(self.dir_path, self.cc_file_dir))

        model_h_file = os.path.join(self.dir_path, self.include_dir,
                                    'HelAmps_%s.h' % self.model_name)
        model_cc_file = os.path.join(self.dir_path, self.cc_file_dir,
                                     'HelAmps_%s.cc' % self.model_name)

        replace_dict = {}

        replace_dict['output_name'] = self.output_name
        replace_dict['info_lines'] = get_mg5_info_lines()
        replace_dict['namespace'] = self.namespace
        replace_dict['model_name'] = self.model_name

        # Read in the template .h and .cc files, stripped of compiler
        # commands and namespaces
        template_h_files = self.read_aloha_template_files(ext = 'h')
        template_cc_files = self.read_aloha_template_files(ext = 'cc')

        aloha_model = create_aloha.AbstractALOHAModel(self.model.get('name'))
        aloha_model.add_Lorentz_object(self.model.get('lorentz'))
        
        if self.wanted_lorentz:
            aloha_model.compute_subset(self.wanted_lorentz)
        else:
            aloha_model.compute_all(save=False, custom_propa=True)
            
        for abstracthelas in dict(aloha_model).values():
            h_rout, cc_rout = abstracthelas.write(output_dir=None, language='CPP', 
                                                              mode='no_include')

            template_h_files.append(h_rout)
            template_cc_files.append(cc_rout)
            
            #aloha_writer = aloha_writers.ALOHAWriterForCPP(abstracthelas,
            #                                               self.dir_path)
            #header = aloha_writer.define_header()
            #template_h_files.append(self.write_function_declaration(\
            #                             aloha_writer, header))
            #template_cc_files.append(self.write_function_definition(\
            #                              aloha_writer, header))

        replace_dict['function_declarations'] = '\n'.join(template_h_files)
        replace_dict['function_definitions'] = '\n'.join(template_cc_files)

        file_h = self.read_template_file(self.aloha_template_h) % replace_dict
        file_cc = self.read_template_file(self.aloha_template_cc) % replace_dict

        # Write the files
        writers.CPPWriter(model_h_file).writelines(file_h)
        writers.CPPWriter(model_cc_file).writelines(file_cc)

        logger.info("Created files %s and %s in directory" \
                    % (os.path.split(model_h_file)[-1],
                       os.path.split(model_cc_file)[-1]))
        logger.info("%s and %s" % \
                    (os.path.split(model_h_file)[0],
                     os.path.split(model_cc_file)[0]))


    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        template_files = []
        for filename in misc.glob('*.%s' % ext, pjoin(MG5DIR, 'aloha','template_files')):
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

#    def write_function_declaration(self, aloha_writer, header):
#        """Write the function declaration for the ALOHA routine"""
#
#        ret_lines = []
#        for line in aloha_writer.write_h(header).split('\n'):
#            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
#                # Strip out compiler flags and namespaces
#                continue
#            ret_lines.append(line)
#        return "\n".join(ret_lines)
#
#    def write_function_definition(self, aloha_writer, header):
#        """Write the function definition for the ALOHA routine"""
#
#        ret_lines = []
#        for line in aloha_writer.write_cc(header).split('\n'):
#            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
#                # Strip out compiler flags and namespaces
#                continue
#            ret_lines.append(line)
#        return "\n".join(ret_lines)

    def clean_line(self, line):
        """Strip a line of compiler options and namespace options."""

        if self.compiler_option_re.match(line) or self.namespace_re.match(line):
            return ""

        return line

    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, filename, classpath=False):
        """Open a template file and return the contents."""
         
        return OneProcessExporterCPP.read_template_file(filename, classpath)


#===============================================================================
# UFOModelConverterPythia8
#===============================================================================

class UFOModelConverterPythia8(UFOModelConverterCPP):
    """ A converter of the UFO-MG5 Model to the Pythia 8 format """

    # Static variables (for inheritance)
    output_name = 'Pythia 8'
    namespace = 'Pythia8'
    
    # Dictionaries for expression of MG5 SM parameters into Pythia 8
    slha_to_expr = {('SMINPUTS', (1,)): '1./csm->alphaEM(((pd->m0(23))*(pd->m0(23))))',
                    ('SMINPUTS', (2,)): 'M_PI*csm->alphaEM(((pd->m0(23))*(pd->m0(23))))*((pd->m0(23))*(pd->m0(23)))/(sqrt(2.)*((pd->m0(24))*(pd->m0(24)))*(((pd->m0(23))*(pd->m0(23)))-((pd->m0(24))*(pd->m0(24)))))',
                    ('SMINPUTS', (3,)): 'alpS',
                    ('CKMBLOCK', (1,)): 'csm->VCKMgen(1,2)',
                    }

    # Template files to use
    param_template_h = 'pythia8_model_parameters_h.inc'
    param_template_cc = 'pythia8_model_parameters_cc.inc'
    template_paths = os.path.join(_file_path, 'iolibs', 'template_files', 'pythia8')     

    def prepare_parameters(self):
        """Extract the model parameters from Pythia 8, and store them in
        the two lists params_indep and params_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['parameters'].keys())
        keys.sort(key=len)
        params_ext = []
        for key in keys:
            if key == ('external',):
                params_ext += [p for p in self.model['parameters'][key] if p.name]
            elif 'aS' in key:
                for p in self.model['parameters'][key]:
                    self.params_dep.append(base_objects.ModelVariable(p.name,
                                                 p.name + " = " + \
                                                 self.p_to_cpp.parse(p.expr) + ';',
                                                 p.type,
                                                 p.depend))
            else:
                for p in self.model['parameters'][key]:
                    self.params_indep.append(base_objects.ModelVariable(p.name,
                                                 p.name + " = " + \
                                                 self.p_to_cpp.parse(p.expr) + ';',
                                                 p.type,
                                                 p.depend))

        # For external parameters, want to use the internal Pythia
        # parameters for SM params and masses and widths. For other
        # parameters, want to read off the SLHA block code
        while params_ext:
            param = params_ext.pop(0)
            key = (param.lhablock, tuple(param.lhacode))
            if 'aS' in self.slha_to_depend.setdefault(key, ()):
                # This value needs to be set event by event
                self.params_dep.insert(0,
                                       base_objects.ModelVariable(param.name,
                                                   param.name + ' = ' + \
                                                   self.slha_to_expr[key] + ';',
                                                   'real'))
            else:
                try:
                    # This is an SM parameter defined above
                    self.params_indep.insert(0,
                                             base_objects.ModelVariable(param.name,
                                                   param.name + ' = ' + \
                                                   self.slha_to_expr[key] + ';',
                                                   'real'))
                except Exception:
                    # For Yukawa couplings, masses and widths, insert
                    # the Pythia 8 value
                    if param.lhablock == 'YUKAWA':
                        self.slha_to_expr[key] = 'pd->mRun(%i, pd->m0(24))' \
                                                 % param.lhacode[0]
                    if param.lhablock == 'MASS':
                        self.slha_to_expr[key] = 'pd->m0(%i)' \
                                            % param.lhacode[0]
                    if param.lhablock == 'DECAY':
                        self.slha_to_expr[key] = \
                                            'pd->mWidth(%i)' % param.lhacode[0]
                    if key in self.slha_to_expr:
                        self.params_indep.insert(0,\
                                     base_objects.ModelVariable(param.name,
                                     param.name + "=" + self.slha_to_expr[key] \
                                                                + ';',
                                                                'real'))
                    else:
                        # This is a BSM parameter which is read from SLHA
                        if len(param.lhacode) == 1:
                            expression = "if(!slhaPtr->getEntry<double>(\"%s\", %d, %s)){\n" % \
                                         (param.lhablock.lower(),
                                          param.lhacode[0],
                                          param.name) + \
                                          ("cout << \"Warning, setting %s to %e\" << endl;\n" \
                                          + "%s = %e;}") % (param.name, param.value.real,
                                                           param.name, param.value.real)
                        elif len(param.lhacode) == 2:
                            expression = "if(!slhaPtr->getEntry<double>(\"%s\", %d, %d, %s)){\n" % \
                                         (param.lhablock.lower(),
                                          param.lhacode[0],
                                          param.lhacode[1],
                                          param.name) + \
                                          ("cout << \"Warning, setting %s to %e\" << endl;\n" \
                                          + "%s = %e;}") % (param.name, param.value.real,
                                                           param.name, param.value.real)
                        elif len(param.lhacode) == 3:
                            expression = "if(!slhaPtr->getEntry<double>(\"%s\", %d, %d, %d, %s)){\n" % \
                                         (param.lhablock.lower(),
                                          param.lhacode[0],
                                          param.lhacode[1],
                                          param.lhacode[2],
                                          param.name) + \
                                          ("cout << \"Warning, setting %s to %e\" << endl;\n" \
                                          + "%s = %e;}") % (param.name, param.value.real,
                                                           param.name, param.value.real)
                        else:
                            raise MadGraph5Error("Only support for SLHA blocks with 1 or 2 indices")
                        self.params_indep.insert(0,
                                               base_objects.ModelVariable(param.name,
                                                                          expression,
                                                                          'real'))

    def write_makefile(self):
        """Generate the Makefile, which creates library files."""

        makefilename = os.path.join(self.dir_path, self.cc_file_dir,
                                    'Makefile')

        replace_dict = {}

        replace_dict['info_lines'] = get_mg5_info_lines()
        replace_dict['model'] = self.model_name

        if self.default_replace_dict['version'] == "8.2":
            path = 'pythia8.2_makefile.inc'
        else:
            path = 'pythia8_makefile.inc'
        makefile = self.read_template_file(path) % replace_dict

        # Write the files
        open(makefilename, 'w').write(makefile)

        logger.info("Created %s in directory %s" \
                    % (os.path.split(makefilename)[-1],
                       os.path.split(makefilename)[0]))

    def write_param_card(self):
        """Generate the param_card for the model."""

        paramcardname = os.path.join(self.dir_path, self.cc_file_dir,
                                    'param_card_%s.dat' % self.model_name)
        # Write out param_card
        open(paramcardname, 'w').write(\
            self.model.write_param_card())

        logger.info("Created %s in directory %s" \
                    % (os.path.split(paramcardname)[-1],
                       os.path.split(paramcardname)[0]))
        
    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, *args, **opts):
        """Open a template file and return the contents."""
         
        return OneProcessExporterPythia8.read_template_file(*args, **opts)

def ExportCPPFactory(cmd, group_subprocesses=False, cmd_options={}):
    """ Determine which Export class is required. cmd is the command 
        interface containing all potential usefull information.
    """

    opt = dict(cmd.options)
    opt['output_options'] = cmd_options
    cformat = cmd._export_format
    
    if cformat == 'pythia8':
        return ProcessExporterPythia8(cmd._export_dir, opt)
    elif cformat == 'standalone_cpp':
        return  ProcessExporterCPP(cmd._export_dir, opt)
    elif cformat == 'standalone_gpu':
        return  ProcessExporterGPU(cmd._export_dir, opt)
    elif cformat == 'matchbox_cpp':
        return  ProcessExporterMatchbox(cmd._export_dir, opt)
    else:
        return cmd._export_plugin(cmd._export_dir, opt)

    


