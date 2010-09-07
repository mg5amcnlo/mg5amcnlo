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

"""Methods and classes to export models and matrix elements to Pythia8
format."""

import fractions
import glob
import itertools
import logging
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
import madgraph.iolibs.misc as misc
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
from madgraph import MadGraph5Error, MG5DIR

import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_pythia8')


#===============================================================================
# generate_process_files_pythia8
#===============================================================================
def generate_process_files_pythia8(multi_matrix_element, cpp_helas_call_writer,
                                   process_string = "", path = os.getcwd()):

    """Generate the .h and .cc files needed for Pythia 8, for the
    processes described by multi_matrix_element"""

    process_exporter_pythia8 = ProcessExporterPythia8(multi_matrix_element,
                                                      cpp_helas_call_writer,
                                                      process_string,
                                                      path)

    process_exporter_pythia8.generate_process_files_pythia8()


#===============================================================================
# ProcessExporterPythia8
#===============================================================================
class ProcessExporterPythia8(object):
    """Class to take care of exporting a set of matrix elements to
    Pythia 8 format."""

    class ProcessExporterPythia8Error(Exception):
        pass
    
    def __init__(self, multi_matrix_element, cpp_helas_call_writer, process_string = "",
                 path = os.getcwd()):
        """Initiate with matrix elements, helas call writer, process
        string, path. Generate the process .h and .cc files."""

        self.matrix_elements = multi_matrix_element.get('matrix_elements')

        if not self.matrix_elements:
            raise MadGraph5Error("No matrix elements to export")

        self.model = self.matrix_elements[0].get('processes')[0].get('model')

        self.processes = sum([me.get('processes') for \
                              me in self.matrix_elements], [])

        if process_string:
            self.process_string = process_string
        else:
            self.process_string = self.processes.base_string()

        self.process_name = self.get_process_name()

        self.path = path
        self.helas_call_writer = cpp_helas_call_writer

        if not isinstance(self.helas_call_writer, helas_call_writers.Pythia8UFOHelasCallWriter):
            raise ProcessExporterPythia8Error, \
                "helas_call_writer not Pythia8UFOHelasCallWriter"

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

            # Redefine equality so that mass differences for external
            # particles don't matter (since Pythia gives the mass
            # explicitly anyway)
            wf_equal = helas_objects.HelasWavefunction.__eq__
            helas_objects.HelasWavefunction.__eq__ = \
                              helas_objects.HelasWavefunction.almost_equal

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

            # Redefined the equality back to the original
            helas_objects.HelasWavefunction.__eq__ = wf_equal

    # Methods for generation of process files for Pythia 8

    def generate_process_files_pythia8(self):

        """Generate the .h and .cc files needed for Pythia 8, for the
        processes described by multi_matrix_element"""

        cwd = os.getcwd()

        os.chdir(self.path)

        pathdir = os.getcwd()

        logger.info('Creating files %s.h and %s.cc in directory %s' % \
                    (self.process_name, self.process_name, self.path))

        # Create the files
        filename = '%s.h' % self.process_name
        self.write_pythia8_process_h_file(writers.CPPWriter(filename))

        filename = '%s.cc' % self.process_name
        self.write_pythia8_process_cc_file(writers.CPPWriter(filename))

        os.chdir(cwd)

    #===========================================================================
    # write_pythia8_process_h_file
    #===========================================================================
    def write_pythia8_process_h_file(self, writer):
        """Write the class definition (.h) file for the process"""

        if not isinstance(writer, writers.CPPWriter):
            raise writers.CPPWriter.CPPWriterError(\
                "writer not CPPWriter")

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract model name
        replace_dict['model_name'] = \
                         self.model.get('name')

        # Extract process file name
        replace_dict['process_file_name'] = self.process_name

        # Extract class definitions
        process_class_definitions = self.get_process_class_definitions()
        replace_dict['process_class_definitions'] = process_class_definitions

        file = read_template_file('pythia8_process_h.inc') % replace_dict

        # Write the file
        writer.writelines(file)

    #===========================================================================
    # write_pythia8_process_cc_file
    #===========================================================================
    def write_pythia8_process_cc_file(self, writer):
        """Write the class member definition (.cc) file for the process
        described by matrix_element"""

        if not isinstance(writer, writers.CPPWriter):
            raise writers.CPPWriter.CPPWriterError(\
                "writer not CPPWriter")

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process file name
        replace_dict['process_file_name'] = self.process_name

        # Extract model name
        replace_dict['model_name'] = self.model.get('name')
                         

        # Extract class function definitions
        process_function_definitions = \
                         self.get_process_function_definitions()
        replace_dict['process_function_definitions'] = \
                                                   process_function_definitions

        file = read_template_file('pythia8_process_cc.inc') % replace_dict

        # Write the file
        writer.writelines(file)

    #===========================================================================
    # Process export helper functions
    #===========================================================================
    def get_process_class_definitions(self):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = {}

        # Extract model name
        replace_dict['model_name'] = self.model.get('name')

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
                                          self.model.get('name'))
        replace_dict['process_definition'] = process_definition

        process = self.processes[0]
        replace_dict['process_code'] = 10000 + \
                                       process.get('id')

        replace_dict['inFlux'] = self.get_process_influx()

        replace_dict['id_masses'] = self.get_id_masses(process)
        replace_dict['resonances'] = self.get_resonance_lines()

        replace_dict['nexternal'] = self.nexternal
        replace_dict['nprocesses'] = len(self.matrix_elements)

        if self.single_helicities:
            replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          void calculate_wavefunctions(const int hel[]);
                          static const int nwavefuncs = %d;
                          complex w[nwavefuncs][18];""" % \
                                                    len(self.wavefunctions)
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


        file = read_template_file('pythia8_process_class.inc') % replace_dict

        return file

    def get_process_function_definitions(self):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = {}

        # Extract model name
        replace_dict['model_name'] = self.model.get('name')

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
        replace_dict['sigmaKin_lines'] = \
                                     self.get_sigmaKin_lines(color_amplitudes)
        replace_dict['sigmaHat_lines'] = \
                                     self.get_sigmaHat_lines()

        replace_dict['setIdColAcol_lines'] = \
                                   self.get_setIdColAcol_lines(color_amplitudes)

        replace_dict['weightDecay_lines'] = \
                                       self.get_weightDecay_lines()    

        replace_dict['all_sigmaKin'] = \
                                  self.get_all_sigmaKin_lines(color_amplitudes)

        file = read_template_file('pythia8_process_function_definitions.inc') %\
               replace_dict

        return file

    def get_process_name(self):
        """Return process file name for the process in matrix_element"""

        process_string = self.process_string

        # Extract process number
        proc_number_pattern = re.compile("^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(process_string)
        proc_number = 0
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            process_string = proc_number_re.group(1) + \
                             proc_number_re.group(3)

        # Remove order information
        order_pattern = re.compile("^(.+)\s+(\w+)\s*=\s*(\d+)\s*$")
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

        process_string = "Sigma_%s_%s" % (self.model.get('name'),
                                          process_string)
        return process_string

    def get_process_info_lines(self, matrix_element):
        """Return info lines describing the processes for this matrix element"""

        return"\n".join([ "# " + process.nice_string().replace('\n', '\n# * ') \
                         for process in matrix_element.get('processes')])


    def get_process_influx(self):
        """Return process file name for the process in matrix_element"""

        # Create a set with the pairs of incoming partons in definite order,
        # e.g.,  g g >... u d > ... d~ u > ... gives ([21,21], [1,2], [-2,1])
        beams = set([tuple(sorted([process.get('legs')[0].get('id'),
                                   process.get('legs')[1].get('id')])) \
                          for process in self.processes])

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

    def get_id_masses(self, process):
        """Return the lines which define the ids for the final state particles,
        for the Pythia phase space"""

        if self.nfinal == 1:
            return ""
        
        mass_strings = []
        for i in range(2, len(process.get('legs'))):
            if self.model.get_particle(process.get('legs')[i].get('id')).\
                   get('mass') not in  ['zero', 'ZERO']:
                mass_strings.append("int id%dMass() const {return %d;}" % \
                                (i + 1, abs(process.get('legs')[i].get('id'))))

        return "\n".join(mass_strings)

    def get_resonance_lines(self):
        """Return the lines which define the ids for the final state particles,
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

        if singleres != 0:
            res_strings.append("virtual int idSChannel() const {return %d;}" \
                               % singleres)
        if schannel:
            res_strings.append("virtual bool isSChannel() const {return true;}")
            
        return "\n".join(res_strings)

    def get_resonances(self):
        """Return the PIDs for any resonances in 2->2 and 2->3 processes."""

        resonances = []

        # Get a list of all resonant s-channel contributions
        diagrams = sum([me.get('diagrams') for me in self.matrix_elements], [])
        for diagram in diagrams:
            schannels, tchannels = diagram.get('amplitudes')[0].\
                                   get_s_and_t_channels(self.ninitial)

            for schannel in schannels:
                sid = schannel.get('legs')[-1].get('id')
                width = self.model.get_particle(sid).get('width')
                if width.lower() != 'zero':
                    resonances.append(sid)

        resonance_set = set(resonances)

        singleres = 0
        # singleres is set if all diagrams have the same resonance
        if len(resonances) == len(diagrams) and len(resonance_set) == 1:
            singleres = resonances[0]

        # Only care about absolute value of resonance PIDs:
        resonance_set = list(set([abs(pid) for pid in resonance_set]))

        # schannel is True if all diagrams are s-channel and there are
        # no QCD vertices
        schannel = not any([\
            len(d.get('amplitudes')[0].get_s_and_t_channels(self.ninitial)[0])\
                 == 0 for d in diagrams]) and \
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

    def get_reset_jamp_lines(self, color_amplitudes):
        """Get lines to reset jamps"""

        ret_lines = ""
        for icol, col_amp in enumerate(color_amplitudes):
            ret_lines+= """for(int i=0;i < %(ncolor)d; i++)
            jamp2[%(proc_number)d][i]=0.;\n""" % \
            {"ncolor": len(col_amp), "proc_number": icol}
        return ret_lines
        

    def get_calculate_wavefunctions(self, wavefunctions):
        """Return the lines for optimized calculation of the
        wavefunctions for all subprocesses"""

        replace_dict = {}

        replace_dict['nwavefuncs'] = len(wavefunctions)

        replace_dict['wavefunction_calls'] = "\n".join(\
            self.helas_call_writer.get_wavefunction_calls(\
            helas_objects.HelasWavefunctionList(wavefunctions)))

        file = read_template_file('pythia8_process_wavefunctions.inc') % \
                replace_dict

        return file
       

    def get_sigmaKin_lines(self, color_amplitudes):
        """Get sigmaKin_lines for function definition for Pythia 8 .cc file"""

        
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
            replace_dict['den_factors'] = \
                     ",".join([str(me.get_denominator_factor()) for me in \
                               self.matrix_elements])

            replace_dict['get_matrix_t_lines'] = "\n".join(
                    ["t[%(iproc)d]=matrix_%(proc_name)s();" % \
                     {"iproc": i, "proc_name": \
                      me.get('processes')[0].shell_string().replace("0_", "")} \
                     for i, me in enumerate(self.matrix_elements)])

            file = \
                 read_template_file(\
                            'pythia8_process_sigmaKin_function.inc') %\
                            replace_dict

            return file

        else:
            ret_lines = "// Call the individual sigmaKin for each process\n"
            return ret_lines + \
                   "\n".join(["sigmaKin_%s();" % \
                              me.get('processes')[0].shell_string().\
                              replace("0_", "") for \
                              me in self.matrix_elements])

    def get_all_sigmaKin_lines(self, color_amplitudes):
        """Get sigmaKin_process for all subprocesses for Pythia 8 .cc file"""

        ret_lines = []
        if self.single_helicities:
            ret_lines.append(\
                "void %s::calculate_wavefunctions(const int hel[]){" % \
                self.process_name)
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append(self.get_calculate_wavefunctions(\
                self.wavefunctions))
            ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                      color_amplitudes[i]) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)


    def get_sigmaKin_single_process(self, i, matrix_element):
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

        file = \
         read_template_file('pythia8_process_sigmaKin_subproc_function.inc') %\
         replace_dict

        return file

    def get_matrix_single_process(self, i, matrix_element, color_amplitudes):
        """Write sigmaKin for each process"""

        # Write matrix() for the process

        replace_dict = {}

        # Process name
        replace_dict['proc_name'] = \
          matrix_element.get('processes')[0].shell_string().replace("0_", "")
        

        if self.single_helicities:
            replace_dict['matrix_args'] = ""
            replace_dict['all_wavefunction_calls'] = "int i, j;"
        else:
            replace_dict['matrix_args'] = "const int hel[]"
            wavefunctions = matrix_element.get_all_wavefunctions()
            replace_dict['all_wavefunction_calls'] = \
                         """const int nwavefuncs = %d;
                         complex w[nwavefuncs][18];\n""" % len(wavefunctions)+ \
                         self.get_calculate_wavefunctions(wavefunctions)

        # Process name
        replace_dict['process_class_name'] = self.process_name
        
        # Process number
        replace_dict['proc_number'] = i

        # Number of color flows
        replace_dict['ncolor'] = len(color_amplitudes)

        replace_dict['ngraphs'] = matrix_element.get_number_of_amplitudes()

        # Extract color matrix
        replace_dict['color_matrix_lines'] = \
                                     self.get_color_matrix_lines(matrix_element)

        # The Helicity amplitude calls
        replace_dict['amplitude_calls'] = "\n".join(\
            self.helas_call_writer.get_amplitude_calls(matrix_element))
        replace_dict['jamp_lines'] = self.get_jamp_lines(color_amplitudes)

        file = read_template_file('pythia8_process_matrix.inc') % \
                replace_dict

        return file


    def get_sigmaHat_lines(self):
        """Get sigmaHat_lines for function definition for Pythia 8 .cc file"""

        # Create a set with the pairs of incoming partons
        beams = set([(process.get('legs')[0].get('id'),
                      process.get('legs')[1].get('id')) \
                     for process in self.processes])

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

            # Now add matrix elements for the processes with the right factors
            res_lines.append("// Add matrix elements for processes with beams %s" % \
                             repr(beam_parts))
            res_lines.append("return %s;" % \
                             ("+".join(["matrix_element[%i]*%i" % \
                                        (i, len([proc for proc in \
                                         me.get('processes') if beam_parts == \
                                         (proc.get('legs')[0].get('id'),
                                          proc.get('legs')[1].get('id'))])) \
                                        for (i, me) in beam_processes]).\
                              replace('*1', '')))
            res_lines.append("}")
            

        res_lines.append("else {")
        res_lines.append("// Return 0 if not correct initial state assignment")
        res_lines.append(" return 0.;}")

        return "\n".join(res_lines)


    def get_setIdColAcol_lines(self, color_amplitudes):
        """Generate lines to set final-state id and color info for process"""

        res_lines = []

        # Create a set with the pairs of incoming partons
        beams = set([(process.get('legs')[0].get('id'),
                      process.get('legs')[1].get('id')) \
                     for process in self.processes])

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

            final_id_list = []
            for (i, me) in beam_processes:
                final_id_list.extend([tuple([l.get('id') for l in \
                                             proc.get('legs') if l.get('state')]) \
                                      for proc in me.get('processes') \
                                      if beam_parts == \
                                      (proc.get('legs')[0].get('id'),
                                       proc.get('legs')[1].get('id'))])
            final_id_list = set(final_id_list)
            ncombs = len(final_id_list)
            #for ids in final_id_list

            res_lines.append("// Pick one of the flavor combinations %s" % \
                             ", ".join([repr(ids) for ids in final_id_list]))

            me_weight = []
            for final_ids in final_id_list:
                items = [(i, len([ p for p in me.get('processes') \
                             if [l.get('id') for l in \
                             p.get('legs')] == \
                             list(beam_parts) + list(final_ids)])) \
                       for (i, me) in beam_processes]
                me_weight.append("+".join(["matrix_element[%i]*%i" % (i, l) for\
                                           (i, l) in items if l > 0]).\
                                 replace('*1', ''))
                if any([l>1 for (i, l) in items]):
                    raise ProcessExporterPythia8Error,\
                          "More than one process with identical " + \
                          "external particles is not supported"

            res_lines.append("int flavors[%d][%d] = {%s};" % \
                             (ncombs, self.nfinal,
                              ",".join(str(id) for id in \
                                       sum([list(ids) for ids in final_id_list],
                                           []))))
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

            res_lines.append("if(%s){" % \
                                 "||".join(["&&".join(["id%d == %d" % \
                                            (i+1, l.get('id')) for (i, l) in \
                                            enumerate(p.get('legs'))])\
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
                    color_flows.append([color_flow_dict[l.get('number')][i] % 500 \
                                        for (l,i) in itertools.product(legs, [0,1])])

                # Write out colors for the selected color flow
                res_lines.append("static int col[%d][%d] = {%s};" % \
                                 (ncolor, 2 * self.nexternal,
                                  ",".join(str(i) for i in sum(color_flows, []))))

                res_lines.append("setColAcol(%s);" % \
                                 ",".join(["col[ic][%d]" % i for i in \
                                          range(2 * self.nexternal)]))
            res_lines.append('}')

        return "\n".join(res_lines)


    def get_weightDecay_lines(self):
        """Get weightDecay_lines for function definition for Pythia 8 .cc file"""

        weightDecay_lines = "// Just use isotropic decay (default)\n"
        weightDecay_lines += "return 1.;"

        return weightDecay_lines

    def get_helicity_matrix(self, matrix_element):
        """Return the Helicity matrix definition lines for this matrix element"""

        helicity_line = "static const int helicities[ncomb][nexternal] = {";
        helicity_line_list = []

        for helicities in matrix_element.get_helicity_matrix():
            helicity_line_list.append(",".join(['%d'] * len(helicities)) % \
                                      tuple(helicities))

        return helicity_line + ",".join(helicity_line_list) + "};"

    def get_den_factor_line(self, matrix_element):
        """Return the denominator factor line for this matrix element"""

        return "const int denominator = %d;" % \
               matrix_element.get_denominator_factor()

    def get_color_matrix_lines(self, matrix_element):
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

    def get_jamp_lines(self, color_amplitudes):
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

        return "\n".join(res_list)

#===============================================================================
# Global helper methods
#===============================================================================

def read_template_file(filename):
    """Open a template file and return the contents."""

    return open(os.path.join(_file_path, \
                             'iolibs', 'template_files',
                             filename)).read()

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

    res_str = '%+i.' % total_coeff.numerator

    if total_coeff.denominator != 1:
        # Check if total_coeff is an integer
        res_str = res_str + '/%i.' % total_coeff.denominator

    if is_imaginary:
        res_str = res_str + '*complex(0,1)'

    return res_str + '*'

#===============================================================================
# Routines to output UFO models in Pythia8 format
#===============================================================================

def convert_model_to_pythia8(self, model, output_dir):
    """Create a full valid Pythia 8 model from an MG5 model (coming from UFO)"""

    # create the model parameter files
    model_builder = UFOModelConverterPythia8(model, output_dir)
    model_builder.build()

#===============================================================================
# UFOModelConverterPythia8
#===============================================================================

class UFOModelConverterPythia8(object):
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
        self.coups_dep = {}    # name -> base_objects.ModelVariable
        self.coups_indep = []  # base_objects.ModelVariable
        self.params_dep = []   # base_objects.ModelVariable
        self.params_indep = [] # base_objects.ModelVariable
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
                    self.params_dep.append(base_objects.ModelVariable(p.name,
                                                 self.p_to_cpp.parse(p.expr),
                                                 p.type,
                                                 p.depend))
            else:
                for p in self.model['parameters'][key]:
                    self.params_indep.append(base_objects.ModelVariable(p.name,
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
                                       base_objects.ModelVariable(param.name,
                                                         self.slha_to_expr[key],
                                                         'real'))
            else:
                try:
                    # This is an SM parameter defined above
                    self.params_indep.insert(0,
                                             base_objects.ModelVariable(param.name,
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
                                                 base_objects.ModelVariable(param.name,
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
                    self.coups_dep[c.name] = base_objects.ModelVariable(c.name,
                                                 self.p_to_cpp.parse(c.expr),
                                                 c.type,
                                                 c.depend)
            else:
                for c in coup_list:
                    self.coups_indep.append(base_objects.ModelVariable(c.name,
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

        replace_dict['print_independent_parameters'] = \
                               self.write_print_parameters(self.params_indep)
        replace_dict['print_independent_couplings'] = \
                               self.write_print_parameters(self.coups_indep)
        replace_dict['print_dependent_parameters'] = \
                               self.write_print_parameters(self.params_dep)
        replace_dict['print_dependent_couplings'] = \
                               self.write_print_parameters(self.coups_dep.values())

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

        aloha_model = create_aloha.AbstractALOHAModel(\
                                         self.model.get('name'))
        aloha_model.compute_all(save=False)
        for abstracthelas in dict(aloha_model).values():
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
        for filename in glob.glob(os.path.join(MG5DIR, 'aloha',
                                               'template_files', '*.%s' % ext)):
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
