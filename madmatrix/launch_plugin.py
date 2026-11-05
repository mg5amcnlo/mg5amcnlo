# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: O. Mattelaer (Aug 2023) for the MadGraph7 CUDACPP plugin.
# Further modified by: O. Mattelaer, A. Valassi, Z. Wettersten (2024-2025).
# Integrated with the MadGraph7 project in Feb 2026.

import logging
import os
import subprocess
pjoin = os.path.join
logger = logging.getLogger('cmdprint') # for stdout

try:
    import madgraph
except ImportError:
    import internal.madevent_interface as madevent_interface
    import internal.misc as misc
    import internal.extended_cmd as extended_cmd
    import internal.banner as banner_mod
    import internal.common_run_interface as common_run_interface
else:
    import madgraph.interface.madevent_interface as madevent_interface
    import madgraph.various.misc as misc
    import madgraph.interface.extended_cmd as extended_cmd
    import madgraph.various.banner as banner_mod
    import madgraph.interface.common_run_interface as common_run_interface


# run_card floating_type -> madmatrix.mk FPTYPE letter (raw letters pass through)
FLOATING_TYPE_2_FPTYPE = {'all64': 'd', 'all32': 'f', 'color32': 'm', 'denom64': 'v'}
def _fptype_letter(v):
    return FLOATING_TYPE_2_FPTYPE.get(str(v).lower(), str(v).lower())

# The backends accepted by the 'cudacpp_backend' run_card entry. 'auto' is the
# auto-detecting CPU backend (the makefile resolves it to the widest SIMD flavour
# available on the host); the other CPU entries pin one explicit SIMD width.
CUDACPP_SUPPORTED_BACKENDS = [ 'fortran', 'cuda', 'hip', 'scalar', 'simd_128', 'simd_256', 'avx512y', 'simd_512', 'auto' ]


class CPPMEInterface(madevent_interface.MadEventCmdShell):
    def compile(self, *args, **opts):
        """ """
        import multiprocessing
        if not self.options['nb_core'] or self.options['nb_core'] == 'None':
            self.options['nb_core'] = multiprocessing.cpu_count()    
        if 'cwd' in opts and os.path.basename(opts['cwd']) == 'Source':
            path = pjoin(opts['cwd'], 'make_opts')
            common_run_interface.CommonRunCmd.update_make_opts_full(path,
                {'override FPTYPE': _fptype_letter(self.run_card['floating_type']) })
            misc.sprint('FPTYPE checked')
        cudacpp_supported_backends = CUDACPP_SUPPORTED_BACKENDS
        if args and args[0][0] == 'madevent' and hasattr(self, 'run_card'):
            cudacpp_backend = self.run_card['cudacpp_backend'].lower() # the default value is defined in launch_plugin.py
            if cudacpp_backend == 'auto':
                backend_log = pjoin(opts["cwd"], ".resolved-backend")
                # try to remove old file if present
                try:
                    os.remove(backend_log)
                except FileNotFoundError:
                    pass
                misc.compile(["-f", "cudacpp.mk", f"BACKEND=auto", f"BACKEND_LOG={backend_log}", "detect-backend"], **opts)
                try:
                    with open(backend_log, "r") as f:
                        resolved_backend = f.read().strip()
                    logger.info(f"Backend '{cudacpp_backend}' resolved as '{resolved_backend}'")
                    cudacpp_backend = resolved_backend
                except FileNotFoundError:
                    raise RuntimeError("Could not resolve cudacpp_backend=auto; ensure Makefile detection runs properly.")
            logger.info(f"Building madevent in madevent_interface.py with '{cudacpp_backend}' matrix elements")
            if cudacpp_backend in cudacpp_supported_backends :
                args[0][0] = 'madevent_' + cudacpp_backend + '_link'
            else:
                raise Exception(f"Invalid cudacpp_backend='{cudacpp_backend}': supported backends are [ '" + "', '".join(cudacpp_supported_backends) + "' ]")
            return misc.compile(nb_core=self.options['nb_core'], *args, **opts)
        else:
            return misc.compile(nb_core=self.options['nb_core'], *args, **opts)

# Phase-Space Optimization ------------------------------------------------------------------------------------
template_on = \
"""#***********************************************************************
# SIMD/GPU configuration for the CUDACPP plugin
#************************************************************************
 %(cudacpp_backend)s = cudacpp_backend ! CUDACPP backend: fortran, cuda, hip, auto, scalar, simd_128, simd_256, simd_512, avx512y
"""

template_off = ''
plugin_block = banner_mod.RunBlock('simd', template_on=template_on, template_off=template_off)

class CPPRunCard(banner_mod.RunCardLO):
    blocks = banner_mod.RunCardLO.blocks + [plugin_block]

    def __setitem__(self, name, value, *args, **opts):
        """Refuse an unsupported cudacpp_backend instead of silently falling back.

        The generic ConfigFile machinery reacts to a value outside an 'allowed'
        list by logging a warning and keeping the previous value. For
        cudacpp_backend that would mean building and running on a backend the
        user never asked for, so make it a hard error here. This matters in
        particular for run cards written before the backend renaming, which
        still carry a removed value such as 'cpp' or 'cpu_128b'.
        """
        if isinstance(name, str) and name.strip().lower() == 'cudacpp_backend':
            allowed = getattr(self, 'allowed_value', {}).get('cudacpp_backend', [])
            if allowed and str(value).strip().lower() not in [str(v).lower() for v in allowed]:
                raise banner_mod.InvalidRunCard(
                    "Invalid cudacpp_backend='%s': supported backends are [ '%s' ]. " \
                    % (str(value).strip(), "', '".join(str(v) for v in allowed)) +
                    "The CPU/SIMD backends were renamed: use 'auto' (auto-detected SIMD width) " \
                    "instead of 'cpu'/'cpp', and " \
                    "'scalar'/'simd_128'/'simd_256'/'simd_512'/'avx512y' instead of " \
                    "'cpu_scalar'/'cpu_128b'/'cpu_256b'/'cpu_512b'/'cpu_512b_y'.")
        return super().__setitem__(name, value, *args, **opts)

    def reset_simd(self, old_value, new_value, name):
        if not hasattr(self, 'path'):
            raise Exception('INTERNAL ERROR! CPPRunCard instance has no attribute path') # now ok after fixing #790
        if name == "vector_size" and new_value <= int(old_value):
            # code can handle the new size -> do not recompile
            return

        # ok need to force recompilation of the cpp part
        Sourcedir = pjoin(os.path.dirname(os.path.dirname(self.path)), 'Source')
        subprocess.call(['make', 'cleanall'], cwd=Sourcedir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def reset_makeopts(self, old_value, new_value, name):
        if not hasattr(self, 'path'):
            raise Exception
        if name == 'floating_type':
            common_run_interface.CommonRunCmd.update_make_opts_full({'override FPTYPE': _fptype_letter(new_value)})
        else:
            raise Exception
        Sourcedir = pjoin(os.path.dirname(os.path.dirname(self.path)), 'Source')
        subprocess.call(['make', 'cleanall'], cwd=Sourcedir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def default_setup(self):
        super().default_setup()
        self.add_param('floating_type', 'color32', include=False, hidden=True,
                       fct_mod=(self.reset_makeopts,(),{}),
                       allowed=['color32','all64','all32','denom64','m','d','f','v'],
                       comment='precision: all32, all64, color32 (colour fp32, rest fp64), denom64 (momenta+denom fp64, rest fp32; SIMD falls back to color32)'
                       )
        self.add_param('cudacpp_backend', 'auto', include=False, hidden=False,
                       allowed=CUDACPP_SUPPORTED_BACKENDS)
        self['vector_size'] = 16 # already setup in default class (just change value)
        self['aloha_flag'] = '--fast-math'
        self['matrix_flag'] = '-O3'
        self['limhel'] = 0
        self.display_block.append('simd')
        self.display_block.append('psoptim')

    # OM/AV - overload the default version in banner.py
    def write_one_include_file(self, output_dir, incname, output_file=None):
        """write one include file at the time"""
        if incname == "vector.inc":
            if 'vector_size' not in self.user_set and 'wrap_size' not in self.user_set: return
            if output_file is None: vectorinc=pjoin(output_dir,incname)
            else: vectorinc=output_file
            with open(vectorinc+'.new','w') as fileout:
                with open(vectorinc) as filein:
                    for line in filein:
                        if line.startswith('C'): fileout.write(line)
            super().write_one_include_file(output_dir, incname, output_file)
            with open(vectorinc+'.new','a') as fileout:
                with open(vectorinc) as filein:
                    for line in filein:
                        if not line.startswith('\n'): fileout.write(line)
            os.replace(vectorinc+'.new',vectorinc)
        else:
            super().write_one_include_file(output_dir, incname, output_file)

    def check_validity(self):
        """ensure that PLUGIN information is consistent"""
        super().check_validity()
        if self['SDE_strategy'] != 1:
            logger.warning('SDE_strategy different of 1 is not supported with SMD/GPU mode')
            self['sde_strategy'] = 1
        if self['hel_recycling']:
            self['hel_recycling'] = False

class GPURunCard(CPPRunCard):
    def default_setup(self):
        super().default_setup()
        # change default value:
        self['cudacpp_backend'] = 'cuda'
        self['vector_size'] = 32 # ZW: default to 32, might want to change to 64 to utilise AMD GPUs better as well # 16384 # already setup in default class (just change value)
        self['nb_warp'] = 512 # number of warps per kernel call, for now setting to 16 384 / vector_size

MEINTERFACE = CPPMEInterface
RunCard = CPPRunCard
