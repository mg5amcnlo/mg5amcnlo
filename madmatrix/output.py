# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2021) for the MadGraph7 CUDACPP plugin.
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
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.various.misc as misc

from . import launch_plugin

def relative_path_list(relative_path, files_list):
    return list(map(lambda f: pjoin(relative_path, f), files_list))

def template_sources(dirpath, extensions=('.h', '.cc')):
    """The C++ sources of a template directory, sorted. Only regular files with
    a source extension: whatever else lies there (.DS_Store, editor backups
    such as foo.cc~ or .#foo.cc, merge leftovers) must not be copied into every
    generated output."""
    return sorted(f for f in os.listdir(dirpath)
                  if f.endswith(extensions) and not f.startswith('.')
                  and os.path.isfile(pjoin(dirpath, f)))

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

    # The name this exporter is reached by on the 'output' line, for the error
    # messages that have to name it back to the user.
    format_name = 'mg7'

    # The color sum can run on the (n-2)! Del Duca-Dixon-Maltoni basis for a
    # multi-gluon process, but a color flow still has to be picked among the
    # (n-1)! trace structures, so the trace basis is built alongside and the
    # trace jamps are rebuilt from the DDM ones through the Kleiss-Kuijf
    # relations (see set_color_flow_lines_cpp in model_handling.py).
    support_ddm_color_basis = True
    ddm_needs_flow_basis = True

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
                         'read_slha.h', 'read_slha.cc',
                         # --hel_recycling: the per-thread tables, shared by the subprocess libraries
                         'HelRecyclingScratch.h', 'HelRecyclingScratch.cc'
                     ]),
                     # Backend-owned skeleton files live only under backend/<variant>/ now
                     # (see backend_variants below); only genuinely backend-agnostic files
                     # (no backend/ counterpart) are copied flat into SubProcesses/. umami.h
                     # is the only one needed outside standalone mode too (it's the header
                     # for backend/<variant>/umami.cc's UMAMI API); nvtx.h is
                     # standalone-driver-only (see _standalone_extra_files below). The
                     # rambo/random-number sources live once in src/rambo/
                     'SubProcesses': relative_path_list(madmatrix_templates, ['umami.h']),
                     # run_card.toml is generated in finalize() (ProcessExporterMG7.create_run_card)
                     # from the template, not copied verbatim.
                     # Default cards for the optional post-processing tools
                     # (Pythia8/Delphes/MadSpin/reweight/analysis) so that
                     # bin/generate_events can offer to enable and edit them.
                     # The reweight card is the MG7 one: it documents both
                     # reweighting backends (classic module, MadtRex).
                     'Cards': relative_path_list(pjoin(MG5DIR, 'Template', 'Common', 'Cards'),
                                  ['madspin_card_default.dat',
                                   'density_card_default.dat',
                                   'delphes_card_default.dat']) +
                              relative_path_list(mg7_templates, ['reweight_card_default.dat']) +
                              relative_path_list(pjoin(MG5DIR, 'Template', 'LO', 'Cards'),
                                  ['pythia8_card_default.dat',
                                   'madanalysis5_parton_card_default.dat',
                                   'madanalysis5_hadron_card_default.dat',
                                   'rivet_card_default.dat'])}

    # Backend split: mirror template_files/madmatrix/backend/{common,cpu,simd,gpu}/
    # as a top-level backend/ dir, sibling of SubProcesses/src/lib; madmatrix.mk
    # picks the variant (BACKENDDIR) at make time.
    # backend/common/ holds the files that are identical for every variant; it
    # is searched after backend/<variant>/ and no file name is in both, since a
    # quoted #include resolves in the including file's own directory first.
    backend_variants = ('cpu', 'simd', 'gpu')
    backend_template_dir = pjoin(madmatrix_templates, 'backend')
    for _backend_variant in ('common',) + backend_variants:  # plain loop: comprehension wouldn't see the locals above
        from_template[pjoin('backend', _backend_variant)] = relative_path_list(
            pjoin(backend_template_dir, _backend_variant),
            template_sources(pjoin(backend_template_dir, _backend_variant)))
    del _backend_variant

    # Rambo/random-number files copy in src/rambo/
    rambo_template_dir = pjoin(madmatrix_templates, 'src', 'rambo')
    from_template['src/rambo'] = relative_path_list(
        rambo_template_dir, template_sources(rambo_template_dir))

    # Backend-owned skeleton files (GpuRuntime.h, color_sum.{h,cc}, the
    # MemoryAccess*.h family, MatrixElementKernels/CrossSectionKernels/umami.cc,
    # etc.) are NOT linked into P* at all: they are read straight from the
    # top-level backend/<variant>/ and backend/common/ dirs via the Makefile's
    # INCFLAGS/vpath (see BACKENDDIR in madmatrix.mk). They are still compiled
    # once per P* directory, into its own build.<BACKEND>/, and have to be:
    # they include that directory's generated ProcessData.h, ColorData.h and
    # ProcessTables.h. Only files with no backend/ counterpart - genuinely
    # backend-agnostic - stay here.
    to_link_in_P = ['umami.h']

    template_src_make = pjoin(madmatrix_templates, 'madmatrix_src.mk')
    # SubProcesses/makefile is only a dispatcher over the P* directories: it is
    # what makes 'make -j N' in SubProcesses build all the subprocesses with a
    # single, shared pool of N jobs.
    template_Sub_make = pjoin(madmatrix_templates, 'madmatrix_subprocesses.mk')

    # The actual build rules, rendered into SubProcesses/. Each P* directory
    # links one of them (p_makefile, see OneProcessExporterMadMatrix) as its own
    # 'makefile'.
    p_makefiles = ['madmatrix.mk']

    dirs_to_create = ['bin', 'src', 'src/rambo', 'lib', 'Cards', 'SubProcesses',
                      'backend',
                      'backend/common',
                      'backend/cpu',
                      'backend/simd',
                      'backend/gpu']

    # AV - use a custom UFOModelConverter (model/aloha exporter)
    create_model_class = model_handling.MadMatrixUFOModelConverter

    # AV - use a custom GPUFOHelasCallWriter
    # (NB: use "helas_exporter" - see class MadGraphCmd in madgraph_interface.py - not "aloha_exporter" that is never used!)
    ###helas_exporter = None
    helas_exporter = model_handling.MadMatrixUFOHelasCallWriter # this is one of the main fixes for issue #341!

    # AV (default from OM's tutorial) - add a debug printout
    def __init__(self, *args, **kwargs):
        self.in_madevent_mode = False # see MR #747
        # {model} is resolved by ProcessExporterMG7.generate_subprocess_directory()
        # alongside {process_id}, from the matrix element's own model -- same
        # sanitized name (get_model_name()) used for MADMATRIX_LIB/
        # MADMATRIX_COMMONLIB's %(model)s in madmatrix.mk. {{device}}/{{fptype}}
        # stay literal ("{device}"/"{fptype}") in the resulting me_path: which
        # concrete BACKEND/FPTYPE a given process directory was compiled for is
        # a per-build choice made when the libraries are compiled and loaded
        # (see compile_matrix_elements() in mg7/launch.py), not something
        # subprocesses.json commits to at generation time.
        args[1]["me_lib_format"] = pjoin(
            "lib", "libmadmatrix_{model}_{process_id}_{{device}}_{{fptype}}.so"
        )
        super().__init__(*args, **kwargs)
        # Honor the output command's --mask=True|False (flavor-mask
        # optimization for grouped/merged flavors). Default: enabled.
        self.use_flavor_mask = self._parse_flavor_mask_option()
        # --axial_gauge is read per matrix element by the helas call writer;
        # say here, once per output, if the FD gauge drops it
        helas_call_writers.axial_gauge_requested(
                            self.opt.get('output_options'), warn=True)
        # --hel_recycling: a new output starts with no warm-up of its own; what
        # an earlier one left registered (one that died before its finalize)
        # must never be built into this one
        del model_handling.OneProcessExporterMadMatrix._hr_warmups[:]

    def _parse_flavor_mask_option(self):
        """Read --mask=True|False from the output command line (default True)."""
        out_opts = self.opt.get('output_options', {}) if hasattr(self, 'opt') else {}
        val = out_opts.get('mask', True)
        if isinstance(val, str):
            return val.strip().lower() not in ('false', '0', 'no', 'off')
        return bool(val)

    def get_makefile_replace_dict(self, model):
        """Add what madmatrix.mk needs to know about a host BLAS for the C++
        color sum. Whether a given process actually takes it is decided when
        that process is written out (see cpp_blas_wanted); this only settles
        whether one could be linked at all."""

        replace_dict = super().get_makefile_replace_dict(model)
        flags = self.oneprocessclass.blas_available_flags()
        replace_dict['cpp_blas_default'] = 'hasBlas' if flags else 'hasNoBlas'
        replace_dict['cpp_blas_libflags'] = flags
        return replace_dict

    # AV - overload the default version: create CMake directory, do not create lib directory
    def copy_template(self, model):
        super().copy_template(model)
        # Copy Arithmetics headers for the double-word expansion (FPTYPE=e)
        arithmetics_src = pjoin(self.madmatrix_templates, 'Arithmetics')
        if os.path.isdir(arithmetics_src):
            arithmetics_dst = pjoin(self.dir_path, 'src', 'Arithmetics')
            try:
                os.makedirs(arithmetics_dst, exist_ok=True)
            except os.error:
                pass
            for f in ['Double.h', 'basicOPs.h', 'errorFreeOPs.h']:
                files.cp(pjoin(arithmetics_src, f), arithmetics_dst)

        # Rename Makefile to makefile
        if self.template_src_make:
            shutil.move(os.path.join(self.dir_path, "src", "Makefile"), os.path.join(self.dir_path, "src", "makefile"))
        if self.template_Sub_make:
            shutil.move(os.path.join(self.dir_path, "SubProcesses", "Makefile"), os.path.join(self.dir_path, "SubProcesses", "makefile"))
        self.write_p_makefiles(model)

    def write_p_makefiles(self, model):
        """Render the build rules shared by all the P* directories into
        SubProcesses/ (they are linked from there as each P*/makefile)."""
        # through the hook, not an inline dict: madmatrix.mk also carries the
        # host-BLAS placeholders that get_makefile_replace_dict fills in
        replace_dict = self.get_makefile_replace_dict(model)
        # kept to render the --hel_recycling probe makefile at finalize
        self._p_makefile_replace = replace_dict
        for name in self.p_makefiles:
            rendered = self.read_template_file(pjoin(self.madmatrix_templates, name)) % replace_dict
            open(pjoin(self.dir_path, 'SubProcesses', name), 'w').write(rendered)

    def check_split_orders(self, matrix_element):
        """Report what a squared-order constraint will produce here.

        The jamps carry an amplitude-order index and the color sum pairs them
        (color_sum_cpu_splitorders in backend/{cpu,simd}/color_sum.cc and
        color_sum_kernel in backend/gpu/color_sum.cc, the Fortran GET_MATRIX
        contract), so a '^2' constraint that keeps only some squared orders gets
        the contribution it asked for rather than the total. That is what makes
        the interference case work -- `u u~ > u u~ QED^2==2` keeps every diagram
        and wants the QCD-EW cross term alone, which no amount of dropping
        diagrams at generation can produce.

        The BLAS color sums (CPU: compile time, GPU: CUDACPP_RUNTIME_BLASCOLORSUM)
        cannot pair split orders and are not used for such a process.
        """

        so = export_v4.split_order_tables(matrix_element)
        if not so or so['nampso'] <= 1:
            return
        process = matrix_element.get('processes')[0]
        kept = [n for n, k in zip(so['names'], so['chosen']) if k]
        dropped = [n for n, k in zip(so['names'], so['chosen']) if not k]
        logger.info(
            "%s: '%s' has %d squared-order components (%s); keeping %s%s. "
            "The jamps are split over %d amplitude orders and the color sum "
            "pairs them.",
            self.__class__.format_name,
            process.nice_string().replace('Process: ', ''),
            so['nsqampso'], ', '.join(so['names']),
            ', '.join(kept) if kept else 'nothing',
            '' if not dropped else ', dropping %s' % ', '.join(dropped),
            so['nampso'])

    # AV - add debug printouts (in addition to the default one from OM's tutorial)
    def generate_subprocess_directory(self, matrix_element, cpp_helas_call_writer, proc_number=None):
        self.check_split_orders(matrix_element)
        # Propagate the --mask toggle to the helas call writer that emits the
        # guarded wavefunction/amplitude calls, and the output command line as
        # a whole for the --jamp_optim toggle of the color-flow optimisation.
        if cpp_helas_call_writer is not None:
            cpp_helas_call_writer.use_flavor_mask = self.use_flavor_mask
            cpp_helas_call_writer.cmd_options = self.opt.get('output_options', {})
        return super().generate_subprocess_directory(matrix_element, cpp_helas_call_writer, proc_number)

    # AV (default from OM's tutorial) - add a debug printout
    def convert_model(self, model, wanted_lorentz=[], wanted_couplings=[], **opts):
        # **opts: an option meant for one exporter (npwave, for the dual HELAS
        # libraries of P-wave bound states) is passed by keyword to all of them
        if hasattr(model , 'cudacpp_wanted_ordered_couplings'):
            wanted_couplings = model.cudacpp_wanted_ordered_couplings
            del model.cudacpp_wanted_ordered_couplings
        return super().convert_model(model, wanted_lorentz, wanted_couplings, **opts)

    # AV (default from OM's tutorial) - overload settings and add a debug printout
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""
        # Irrelevant here since group_mode=False so this function is never called
        misc.sprint('Entering ProcessExporterMadMatrix.modify_grouping')
        return False, matrix_element

    def finalize(self, *args, **kwargs):
        super().finalize(*args, **kwargs)
        # --hel_recycling: the warm-up runs once every P* directory is complete
        # and buildable (CPU/SIMD only: the GPU backend ignores the tables)
        self._run_hel_recycling_warmups()

    # The --hel_recycling probe runs through the standalone driver check_sa.exe,
    # which only `output standalone` ships. Any other output is lent it (with
    # the standalone makefile, under another name) for the warm-up build only.
    _hr_probe_files = ['check_sa.cc', 'nvtx.h', 'timer.h', 'timermap.h']
    _hr_probe_makefile = 'hrprobe.mk'

    def _install_hr_probe(self, dirpath):
        """Make check_sa.exe buildable in dirpath: return the extra make
        arguments that build it and the files to remove afterwards."""
        added = []
        for name in self._hr_probe_files:
            target = pjoin(dirpath, name)
            if not os.path.lexists(target):
                files.cp(pjoin(self.madmatrix_templates, name), target)
                added.append(target)
        # madmatrix_standalone.mk includes madmatrix.mk by name
        target = pjoin(dirpath, 'madmatrix.mk')
        if not os.path.lexists(target):
            os.symlink(pjoin(os.pardir, 'madmatrix.mk'), target)
            added.append(target)
        target = pjoin(dirpath, self._hr_probe_makefile)
        rendered = self.read_template_file(pjoin(self.madmatrix_templates,
                        'madmatrix_standalone.mk')) % self._p_makefile_replace
        with open(target, 'w') as fsock:
            fsock.write(rendered)
        added.append(target)
        # the backend marker check_sa.exe's link rule writes
        added.append(pjoin(dirpath, '.check_sa_backend'))
        return ['-f', self._hr_probe_makefile], added

    def _run_hel_recycling_warmups(self):
        """For each --hel_recycling P* directory: build a probe of the ordinary
        code (BACKEND=scalar FPTYPE=d HRPROBE=1), run it once to have its own
        good-helicity scan print the good helicities and the amplitudes that are
        only a cancellation zero at each, and write the recycled build for those.

        The scan is the very sigmaKin_getGoodHel the helicity loop uses at run
        time, so the rows built can never be short of one it asks for (at these
        parameters). On any failure the directory keeps the ordinary code: correct
        but not recycled, which is the right way round to fail."""
        exporters = model_handling.OneProcessExporterMadMatrix._hr_warmups
        # only this output's directories, whatever happens: the list is shared
        # by every exporter of the session
        mine = [e for e in exporters if self._is_own_dir(e.path)]
        del exporters[:]
        for exporter in mine:
            dirpath = exporter.path
            name = os.path.basename(dirpath)
            makeargs, added = [], []
            try:
                makeargs, added = self._install_hr_probe(dirpath)
                built = subprocess.run(
                    ['make'] + makeargs + ['-j%d' % (os.cpu_count() or 1),
                     'BACKEND=scalar', 'FPTYPE=d', 'HRPROBE=1'],
                    cwd=dirpath, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if built.returncode != 0:
                    logger.warning('hel_recycling warm-up build failed in %s; it is '
                        'written without recycling.\n%s', name,
                        built.stdout.decode(errors='replace')[-1500:])
                    continue
                # perf mode: 64 distinct RAMBO events, which the probe's
                # good-helicity scan samples (matrix mode repeats ONE point)
                env = dict(os.environ, MG_DUMP_GOODHEL='1')
                run = subprocess.run([pjoin(dirpath, 'check_sa.exe'), 'perf',
                                      '1', '64', '1'],
                                     cwd=dirpath, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                output = run.stdout.decode(errors='replace')
                if run.returncode != 0:
                    logger.warning('hel_recycling warm-up run failed in %s; it is '
                                   'written without recycling.', name)
                    continue
                good_hels, zero_perhel = self._parse_hel_warmup(output)
                if not good_hels:
                    logger.warning('hel_recycling warm-up found no helicity in %s; '
                                   'it is written without recycling.', name)
                    continue
                nrows = exporter.write_recycled(good_hels, zero_perhel)
                if nrows is None:
                    continue        # declined, and said why
                logger.info('hel_recycling: %s builds %d helicity rows of %d, '
                    'dropping %d (helicity, amplitude) pairs that vanish', name,
                    nrows, exporter.matrix_elements[0].get_helicity_combinations(),
                    len(zero_perhel))
            except Exception as err:
                logger.warning('hel_recycling warm-up error in %s (%s); it is '
                               'written without recycling.', name, err)
            finally:
                self._clean_warmup_build(dirpath, makeargs, added)

    def _is_own_dir(self, path):
        """Whether path lies in this output's directory."""
        own = os.path.realpath(self.dir_path)
        return os.path.commonpath([own, os.path.realpath(path)]) == own

    @staticmethod
    def _parse_hel_warmup(output):
        """The good helicities (1-based rows of cHel) and the (row, amplitude)
        pairs that vanish, as printed by sigmaKin_setGoodHel under MG_HR_PROBE --
        the protocol madevent's helicity filter prints."""
        good, zeros = [], []
        for line in output.splitlines():
            if line.startswith('Matrix Element/Good Helicity:'):
                good.append(int(line.split()[-1]))
            elif line.startswith('HEL/ZEROAMP:'):
                _me, hel, amp = line.split()[1:4]
                zeros.append((int(hel), int(amp)))
        return sorted(set(good)), sorted(set(zeros))

    @staticmethod
    def _clean_warmup_build(dirpath, makeargs=(), added=()):
        """Undo the probe build: its objects must not be reused (the probe is not
        the real thing), and the build tag it leaves makes the next make with any
        other BACKEND/FPTYPE refuse to run. Then take back the driver files the
        probe was lent. (An explicit BACKEND, or make records the one 'auto'
        resolves to in .resolved-backend.)"""
        try:
            subprocess.run(['make'] + list(makeargs) + ['BACKEND=scalar', 'cleanall'],
                           cwd=dirpath, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except Exception:
            pass
        for path in added:
            try:
                os.remove(path)
            except OSError:
                pass


# Standalone mode: in addition to the normal madmatrix exports, this writes
# an additional wrapper makefile (madmatrix_standalone.mk) on top of madmatrix.mk,
# so that when running `make` in a P* folder, it builds check_sa.exe as well as the process library (predicatable behaviour)
class ProcessExporterMadMatrixStandalone(ProcessExporterMadMatrix):

    format_name = 'standalone'

    # Each P* directory links madmatrix_standalone.mk (which itself includes
    # madmatrix.mk) as its 'makefile'; both have to be rendered in SubProcesses/
    p_makefiles = ProcessExporterMadMatrix.p_makefiles + ['madmatrix_standalone.mk']

    # Standalone-only template files needed to build check_sa.exe
    _standalone_extra_files = ['check_sa.cc', 'nvtx.h', 'timer.h', 'timermap.h']

    from_template = dict(ProcessExporterMadMatrix.from_template)
    from_template['SubProcesses'] = (ProcessExporterMadMatrix.from_template['SubProcesses']
                                     + relative_path_list(ProcessExporterMadMatrix.madmatrix_templates,
                                                          _standalone_extra_files))

    # We don't need the run_card.toml
    from_template['Cards'] = []

    # Symlink the madmatrix.mk file to each P* folder (madmatrix_standalone.mk,
    # linked there as 'makefile', includes it by name)
    to_link_in_P = ProcessExporterMadMatrix.to_link_in_P + _standalone_extra_files + ['madmatrix.mk']

    # P*/makefile points at the standalone wrapper, so that a plain 'make' in a
    # P* directory (or from the SubProcesses dispatcher) also builds check_sa.exe
    oneprocessclass = model_handling.OneProcessExporterMadMatrixStandalone

    def copy_template(self, model):
        super().copy_template(model)

        # Write another custom bin/generate_events to orchestrate the standalone mode
        gen_events = pjoin(self.dir_path, 'bin', 'generate_events')
        if os.path.exists(gen_events):
            os.remove(gen_events)
        files.cp(pjoin(self.madmatrix_templates, 'generate_events_standalone'),
                 gen_events)
        os.chmod(gen_events, 0o755)

        # no run_card here (so no create_run_card), but `launch` still needs
        # the model, e.g. for the widths set to 'auto' in the param_card
        self.write_model_reference(model)

    # What a caller evaluating the libraries one phase-space point at a time
    # (madmatrix.density, used by MadSpin) needs to find the library and the
    # flavor index of a given process
    processes_file = 'madmatrix_processes.json'

    def generate_subprocess_directory(self, matrix_element, cpp_helas_call_writer, proc_number=None):
        out = super().generate_subprocess_directory(matrix_element, cpp_helas_call_writer, proc_number)
        if not hasattr(self, 'processes_of_subprocess'):
            self.processes_of_subprocess = []
        self.processes_of_subprocess.append((
            [[leg.get('id') for leg in process.get_legs_with_decays()]
             for process in matrix_element.get('processes')],
            # the legs marked off-shell ('*'), evaluated at the invariant mass of
            # their momentum (offshell_mass, backend/*/SigmaKin.cc)
            sorted(wf.get('number_external')
                   for wf in matrix_element.get_external_wavefunctions()
                   if wf.get('offshell')),
            self.scale_dependence(matrix_element)))
        return out

    @staticmethod
    def scale_dependence(matrix_element):
        """What makes the matrix element depend on the renormalisation scale
        beyond alpha_s: the couplings and the masses/widths it uses that depend
        on MU_R, and the running elements of the model. The libraries compute
        their couplings from G alone and keep MU_R at its param-card value,
        while the Fortran output sets MU_R from the event scale
        (UPDATE_AS_PARAM2): a caller passing per-event scales needs the
        Fortran one when this is not empty."""
        model = matrix_element.get('processes')[0].get('model')
        scale_keys = [key for key in model.get('couplings')
                      if 'MU_R' in key]
        scale_coups = set(coup.name for key in scale_keys
                          for coup in model.get('couplings')[key])
        used = set()
        for entry in matrix_element.get_used_couplings():
            for coup in (entry if isinstance(entry, list) else [entry]):
                if isinstance(coup, str):
                    used.add(coup.lstrip('-'))
                else:
                    used.add(coup.get('name'))
                    used.update(str(c).lstrip('-')
                                for c in coup.get('flavors').values())
        reasons = sorted(used & scale_coups)
        scale_params = set(param.name for key, params in model.get('parameters').items()
                           if 'MU_R' in key for param in params)
        for wf in matrix_element.get_all_wavefunctions():
            for name in (wf.get('mass'), wf.get('width')):
                if name in scale_params and name not in reasons:
                    reasons.append(name)
        if model.get('running_elements'):
            reasons.append('running elements of the model')
        return reasons

    def finalize(self, *args, **kwargs):
        # no subprocesses.json (nor anything else the mg7 launcher needs), only
        # the --hel_recycling warm-up, which can only run once every P* directory
        # is complete and buildable, and the process table of the standalone
        # libraries
        self._run_hel_recycling_warmups()
        self.write_processes_file()

    def _install_hr_probe(self, dirpath):
        # check_sa.exe is part of a standalone output already
        return [], []

    def write_processes_file(self):
        """SubProcesses/madmatrix_processes.json: per subprocess, its library
        (with the make variables LIBTAG, BACKEND and FPTYPE left as '{libtag}',
        '{backend}' and '{fptype}', as for the common library, common_library),
        its legs, the processes it evaluates, its off-shell legs
        and, per flavor index, every pdg combination (in the leg order of the
        process) that this index stands for, and what of it depends on the
        renormalisation scale beyond alpha_s (scale_dependence)."""
        import json
        subprocesses = []
        for info, (processes, offshell, scale) in zip(self.process_info,
                                                      getattr(self, 'processes_of_subprocess', [])):
            path = os.path.basename(info['path'])
            flavors = []
            for flavor in info['flavors']:
                while len(flavors) <= flavor['index']:
                    flavors.append([])
                flavors[flavor['index']].extend(list(pdgs) for pdgs in flavor['options'])
            subprocesses.append({
                'path': path,
                'library': 'libmadmatrix{libtag}_%s_%s_{backend}_{fptype}.so'
                           % (self.lib_model_name, path),
                'ninitial': len(info['incoming']),
                'legs': list(info['incoming']) + list(info['outgoing']),
                'processes': processes,
                'offshell': offshell,
                # what depends on MU_R besides alpha_s (scale_dependence)
                'scale_dependence': scale,
                'flavors': flavors})
        with open(pjoin(self.dir_path, 'SubProcesses', self.processes_file), 'w') as fsock:
            json.dump({'common_library': 'libmadmatrix_common{libtag}_%s_{backend}_{fptype}.so'
                                         % self.lib_model_name,
                       'subprocesses': subprocesses}, fsock, indent=1)
