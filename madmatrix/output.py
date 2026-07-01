# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
# Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi, Z. Wettersten (2021-2024).
# Integrated with the MadGraph7 project in Feb 2026.

import shutil
import os
import sys
import subprocess

PLUGIN_NAME = __name__.rsplit('.',1)[0]
PLUGINDIR = os.path.dirname( __file__ )

# AV - model_handling includes the custom FileWriter, ALOHAWriter, UFOModelConverter, OneProcessExporter and HelasCallWriter, plus additional patches
from . import model_handling

# AV - create a plugin-specific logger
import logging
logger = logging.getLogger('madgraph.%s.output'%PLUGIN_NAME)
from madgraph import MG5DIR
#------------------------------------------------------------------------------------

from os.path import join as pjoin
import madgraph.iolibs.files as files
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.export_cpp as export_cpp
import madgraph.various.misc as misc

from . import launch_plugin

def relative_path_list(relative_path, files_list):
    return list(map(lambda f: pjoin(relative_path, f), files_list))

# AV - define the plugin's process exporter
# (NB: this is the plugin's main class, enabled in the new_output dictionary in __init__.py)
class ProcessExporterMadMatrix(export_cpp.ProcessExporterMG7):
    # Class structure information
    #  - object
    #  - VirtualExporter(object) [in madgraph/iolibs/export_v4.py]
    #  - ProcessExporterCPP(VirtualExporter) [in madgraph/iolibs/export_cpp.py]
    #  - ProcessExporterMG7(ProcessExporterCPP) [in madgraph/iolibs/export_cpp.py]
    #  - ProcessExporterMadMatrix(ProcessExporterMG7)
    #      This class

    # Below are the class variable that are defined in export_v4.VirtualExporter
    # AV - keep defaults from export_v4.VirtualExporter
    # Check status of the directory. Remove it if already exists
    ###check = True
    # Output type: [Template/dir/None] copy the Template (via copy_template), just create dir or do nothing
    ###output = 'Template'

    # If sa_symmetry is true, generate fewer matrix elements
    # AV - keep OM's default for this plugin (using grouped_mode=False, "can decide to merge uu~ and u~u anyway")
    sa_symmetry = True

    # Below are the class variable that are defined in export_cpp.ProcessExporterGPU
    # AV - keep defaults from export_cpp.ProcessExporterGPU
    # Decide which type of merging is used [madevent/madweight]
    grouped_mode = False
    # Other options
    default_opt = {'clean': False, 'complex_mass':False, 'export_format':'madevent', 'mp': False, 'v5_model': True }

    # AV - keep defaults from export_cpp.ProcessExporterGPU
    # AV - used in MadGraphCmd.do_output to assign export_cpp.ExportCPPFactory to MadGraphCmd._curr_exporter (if cpp or gpu)
    # AV - used in MadGraphCmd.export to assign helas_call_writers.(CPPUFO|GPUFO)HelasCallWriter to MadGraphCmd._curr_helas_model (if cpp or gpu)
    # Language type: 'v4' for f77, 'cpp' for C++ output
    exporter = 'gpu'

    # AV - use a custom OneProcessExporter
    oneprocessclass = model_handling.OneProcessExporterMadMatrix

    # Information to find the template file that we want to include from madgraph
    # you can include additional file from the plugin directory as well
    # AV - use template files from PLUGINDIR instead of MG5DIR and add gpu/mgOnGpuVectors.h
    # [NB: mgOnGpuConfig.h, check_sa.cc and fcheck_sa.f are handled through dedicated methods]
    ###s = MG5DIR + '/madgraph/iolibs/template_files/'

    templates_path = pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files')
    mg7_templates = pjoin(templates_path, 'mg7')
    madmatrix_templates = pjoin(templates_path, 'madmatrix')
    home_path = pjoin(MG5DIR, "madmatrix")

    from_template = {'.': relative_path_list(home_path, ['COPYRIGHT', 'COPYING', 'COPYING.LESSER']),
                     'src': relative_path_list(madmatrix_templates, [
                         'mgOnGpuFptypes.h', 'mgOnGpuCxtypes.h', 'mgOnGpuVectors.h',
                         'constexpr_math.h', 'read_slha.h', 'read_slha.cc'
                     ]),
                     'SubProcesses': relative_path_list(madmatrix_templates, ['nvtx.h', 'GpuRuntime.h', 'GpuAbstraction.h', 'color_sum.h', 'color_sum.cc',
                                      'MemoryAccessHelpers.h', 'MemoryAccessVectors.h',
                                      'MemoryAccessMatrixElements.h', 'MemoryAccessMomenta.h',
                                      'MemoryAccessRandomNumbers.h', 'MemoryAccessWeights.h',
                                      'MemoryAccessAmplitudes.h', 'MemoryAccessWavefunctions.h',
                                      'MemoryAccessGs.h', 'MemoryAccessCouplingsFixed.h',
                                      'MemoryAccessNumerators.h', 'MemoryAccessDenominators.h',
                                      'MemoryAccessChannelIds.h', 'MemoryAccessIflavorVec.h',
                                      'CrossSectionKernels.cc', 'CrossSectionKernels.h',
                                      'MatrixElementKernels.cc', 'MatrixElementKernels.h',
                                      'EventStatistics.h',
                                      'umami.h', 'umami.cc', 'rambo.h']),
                     'Cards': relative_path_list(mg7_templates, ["run_card.toml"])}

    to_link_in_P = ['nvtx.h', 'GpuRuntime.h', 'GpuAbstraction.h', 'color_sum.h',
                    'MemoryAccessHelpers.h', 'MemoryAccessVectors.h',
                    'MemoryAccessMatrixElements.h', 'MemoryAccessMomenta.h',
                    'MemoryAccessRandomNumbers.h', 'MemoryAccessWeights.h',
                    'MemoryAccessAmplitudes.h', 'MemoryAccessWavefunctions.h',
                    'MemoryAccessGs.h', 'MemoryAccessCouplingsFixed.h',
                    'MemoryAccessNumerators.h', 'MemoryAccessDenominators.h',
                    'MemoryAccessChannelIds.h', 'MemoryAccessIflavorVec.h',
                    'CrossSectionKernels.cc', 'CrossSectionKernels.h',
                    'MatrixElementKernels.cc', 'MatrixElementKernels.h',
                    'EventStatistics.h',
                    'MemoryBuffers.h', # this is generated from a template in Subprocesses but we still link it in P1
                    'MemoryAccessCouplings.h', # this is generated from a template in Subprocesses but we still link it in P1
                    'umami.h', 'umami.cc', 'rambo.h']

    template_src_make = pjoin(madmatrix_templates, 'madmatrix_src.mk')
    template_Sub_make = pjoin(madmatrix_templates, 'madmatrix.mk')

    dirs_to_create = ['bin', 'src', 'lib', 'Cards', 'SubProcesses']

    # AV - use a custom UFOModelConverter (model/aloha exporter)
    create_model_class = model_handling.MadMatrixUFOModelConverter

    # AV - use a custom GPUFOHelasCallWriter
    # (NB: use "helas_exporter" - see class MadGraphCmd in madgraph_interface.py - not "aloha_exporter" that is never used!)
    ###helas_exporter = None
    helas_exporter = model_handling.MadMatrixUFOHelasCallWriter # this is one of the main fixes for issue #341!

    # AV (default from OM's tutorial) - add a debug printout
    def __init__(self, *args, **kwargs):
        self.in_madevent_mode = False # see MR #747
        misc.sprint('Entering ProcessExporterMadMatrix.__init__ (initialise the exporter)')
        args[1]["me_lib_format"] = pjoin("lib", "libmadmatrix_{process_id}_{{device}}.so")
        super().__init__(*args, **kwargs)
        # Honor the output command's --mask=True|False (flavor-mask
        # optimization for grouped/merged flavors). Default: enabled.
        self.use_flavor_mask = self._parse_flavor_mask_option()

    def _parse_flavor_mask_option(self):
        """Read --mask=True|False from the output command line (default True)."""
        out_opts = self.opt.get('output_options', {}) if hasattr(self, 'opt') else {}
        val = out_opts.get('mask', True)
        if isinstance(val, str):
            return val.strip().lower() not in ('false', '0', 'no', 'off')
        return bool(val)

    # AV - overload the default version: create CMake directory, do not create lib directory
    def copy_template(self, model):
        misc.sprint('Entering ProcessExporterMadMatrix.copy_template (initialise the directory)')
        super().copy_template(model)
        # Rename Makefile to makefile
        if self.template_src_make:
            shutil.move(os.path.join(self.dir_path, "src", "Makefile"), os.path.join(self.dir_path, "src", "makefile"))
        if self.template_Sub_make:
            shutil.move(os.path.join(self.dir_path, "SubProcesses", "Makefile"), os.path.join(self.dir_path, "SubProcesses", "makefile"))

    # AV - add debug printouts (in addition to the default one from OM's tutorial)
    def generate_subprocess_directory(self, matrix_element, cpp_helas_call_writer, proc_number=None):
        misc.sprint('Entering ProcessExporterMadMatrix.generate_subprocess_directory (create the directory)')
        misc.sprint('  type(matrix_element)=%s'%type(matrix_element)) # e.g. madgraph.core.helas_objects.HelasMatrixElement
        misc.sprint('  type(cpp_helas_call_writer)=%s'%type(cpp_helas_call_writer)) # e.g. madgraph.iolibs.helas_call_writers.GPUFOHelasCallWriter
        misc.sprint('  type(proc_number)=%s me=%s'%(type(proc_number) if proc_number is not None else None, proc_number)) # e.g. int
        misc.sprint("need to link", self.to_link_in_P)
        # Propagate the --mask toggle to the helas call writer that emits the
        # guarded wavefunction/amplitude calls.
        if cpp_helas_call_writer is not None:
            cpp_helas_call_writer.use_flavor_mask = self.use_flavor_mask
        out = super().generate_subprocess_directory(matrix_element, cpp_helas_call_writer, proc_number)
        return out

    # AV (default from OM's tutorial) - add a debug printout
    def convert_model(self, model, wanted_lorentz=[], wanted_couplings=[]):
        if hasattr(model , 'cudacpp_wanted_ordered_couplings'):
            wanted_couplings = model.cudacpp_wanted_ordered_couplings
            del model.cudacpp_wanted_ordered_couplings
        return super().convert_model(model, wanted_lorentz, wanted_couplings)

    # AV (default from OM's tutorial) - overload settings and add a debug printout
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""
        # Irrelevant here since group_mode=False so this function is never called
        misc.sprint('Entering ProcessExporterMadMatrix.modify_grouping')
        return False, matrix_element


# Standalone mode: in addition to the normal madmatrix exports, this writes
# an additional wrapper Makefile (with the template file being madmatrix_standalone.mk) on top of madmatrix.mk,
# so that when running `make` in a P* folder, it builds check_sa.exe as well as the process library (predicatable behaviour)
class ProcessExporterMadMatrixStandalone(ProcessExporterMadMatrix):

    # This wrapper replaces madmatrix.mk
    template_Sub_make = pjoin(ProcessExporterMadMatrix.madmatrix_templates, 'madmatrix_standalone.mk')

    # Standalone-only template files needed to build check_sa.exe
    _standalone_extra_files = ['check_sa.cc',
                               'RamboSamplingKernels.cc', 'RamboSamplingKernels.h',
                               'CommonRandomNumberKernel.cc', 'CommonRandomNumbers.h',
                               'RandomNumberKernels.h',
                               'massless_rambo.h', 'timer.h', 'timermap.h']

    from_template = dict(ProcessExporterMadMatrix.from_template)
    from_template['SubProcesses'] = (ProcessExporterMadMatrix.from_template['SubProcesses']
                                     + relative_path_list(ProcessExporterMadMatrix.madmatrix_templates,
                                                          _standalone_extra_files))

    # We don't need the run_card.toml
    from_template['Cards'] = []

    # Symlink the madmatrix.mk file to each P* folder
    to_link_in_P = ProcessExporterMadMatrix.to_link_in_P + _standalone_extra_files + ['madmatrix.mk']

    def copy_template(self, model):
        super().copy_template(model)
        madmatrix_mk = pjoin(self.madmatrix_templates, 'madmatrix.mk')
        rendered = self.read_template_file(madmatrix_mk) % {
            'model': self.get_model_name(model.get('name')),
            'cpp_compiler': self.opt['cpp_compiler'] if self.opt['cpp_compiler'] else 'g++',
        }
        open(pjoin(self.dir_path, 'SubProcesses', 'madmatrix.mk'), 'w').write(rendered)

        # Write another custom bin/generate_events to orchestrate the standalone mode
        gen_events = pjoin(self.dir_path, 'bin', 'generate_events')
        if os.path.exists(gen_events):
            os.remove(gen_events)
        files.cp(pjoin(self.madmatrix_templates, 'generate_events_standalone'),
                 gen_events)
        os.chmod(gen_events, 0o755)

    def finalize(self, *args, **kwargs):
        # We disable this since we don't need subprocesses.json either
        pass
