################################################################################
#
# Copyright (c) 2024 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Tests for MLM reweighting (REWGT function from reweight.f linked to cluster.f).

This module provides a factory-style test suite to compare MLM reweighting
results between apply_flavor_grouping True and False modes. For a given process,
a specific integration channel is run in each mode and the resulting cross-section
is compared.

The test factory allows easy addition of new processes and channels.
"""
from __future__ import division
from __future__ import absolute_import
import subprocess
import unittest
import os
import re
import shutil
import logging
import tempfile
import math

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
import madgraph.various.banner as banner

from madgraph import MG5DIR

logger = logging.getLogger('test_MLM_reweight')

pjoin = os.path.join


# ============================================================================
# MLM Channel Test Configuration
# ============================================================================
# Each entry defines a test case:
#   'process': the MG5 process string
#   'model': the model to use
#   'defines': list of particle definitions (e.g. ['q = u d s c', 'q~ = u~ d~ s~ c~'])
#   'xqcut': the xqcut value for MLM merging
#   'fg_true': dict with 'subproc' (P* dir pattern) and 'channel' number
#   'fg_false': dict with 'subproc' (P* dir pattern) and 'channel' number
#   'npoints': number of phase-space points per iteration
#   'maxiter': maximum number of iterations
#   'group_subprocesses': whether to group subprocesses (default: False)
#
# Note: channel numbers may differ between modes because flavor grouping
# can merge diagrams into different topologies. The 'subproc' field uses
# a pattern to select the subprocess directory (first match is used).
# Use None to select the first available P* directory.
# For these tests, fg_false often sets group_subprocesses=True so grouped
# subprocess structure is comparable with the flavor-grouped fg_true mode.

MLM_TEST_CHANNELS = {
    'qq_to_qq_tchannel': {
        'process': 'q q~ > q q~',
        'model': 'sm',
        'defines': ['q = u d s c', 'q~ = u~ d~ s~ c~'],
        'xqcut': 20.0,
        'fg_true': {'subproc': None, 'channel': 1},
        'fg_false': {'subproc': None, 'channel': 1, 'group_subprocesses': True},
        'npoints': 1000,
        'maxiter': 5,
    },
    'qq_to_qq_schannel': {
        'process': 'q q~ > q q~',
        'model': 'sm',
        'defines': ['q = u d s c', 'q~ = u~ d~ s~ c~'],
        'xqcut': 20.0,
        'fg_true': {'subproc': None, 'channel': 2},
        'fg_false': {'subproc': None, 'channel': 2, 'group_subprocesses': True},
        'npoints': 1000,
        'maxiter': 5,
    },
    'gg_ttgg': {
        'process': 'g g > t t~ g g',
        'model': 'sm',
        'defines': [],
        'xqcut': 20.0,
        'fg_true': {'subproc': None, 'channel': 1},
        'fg_false': {'subproc': None, 'channel': 1, 'group_subprocesses': True},
        'npoints': 1000,
        'maxiter': 5,
    },
    'DY_uu_to_Z_jjjj': {
        'process': 'u u~ > Z d d~ c c~',
        'model': 'sm',
        'defines': [],
        'xqcut': 20.0,
        'fg_true': {'subproc': None, 'channel': 1},
        'fg_false': {'subproc': None, 'channel': 1, 'group_subprocesses': True},
        'npoints': 1000,
        'maxiter': 5,
    }
}


def get_Pdir(run_dir):
    """Find the subprocess directory (P*) inside SubProcesses."""
    subproc_dir = pjoin(run_dir, 'SubProcesses')
    Pdirs = [d for d in os.listdir(subproc_dir)
             if d.startswith('P') and os.path.isdir(pjoin(subproc_dir, d))]
    if not Pdirs:
        raise RuntimeError("No subprocess directory P* found in %s" % subproc_dir)
    return Pdirs


def select_Pdir(run_dir, subproc_pattern=None):
    """Select a subprocess directory by pattern.

    Args:
        run_dir: the madevent output directory
        subproc_pattern: string pattern to match (e.g. 'P0_uux_uux').
                        If None, returns the first P* directory found.

    Returns:
        Absolute path to the selected subprocess directory.
    """
    Pdirs = get_Pdir(run_dir)
    subproc_dir = pjoin(run_dir, 'SubProcesses')

    if subproc_pattern is None:
        return pjoin(subproc_dir, sorted(Pdirs)[0])

    for d in sorted(Pdirs):
        if subproc_pattern in d or d == subproc_pattern:
            return pjoin(subproc_dir, d)

    raise RuntimeError(
        "No subprocess directory matching '%s' found. Available: %s" %
        (subproc_pattern, sorted(Pdirs))
    )


def _normalize_fortran_data_text(text):
    """Normalize Fortran include content for tolerant DATA parsing."""
    lines = []
    for raw in text.splitlines():
        if not raw:
            continue
        # Drop fixed-form full-line comments.
        if raw[0] in ('c', 'C', '*', '!'):
            continue
        # Strip inline comments.
        line = raw.split('!')[0].strip()
        if line:
            lines.append(line)
    return ' '.join(lines)


def _iter_data_assignments(text):
    """Yield (lhs, rhs) pairs from Fortran DATA statements."""
    normalized = _normalize_fortran_data_text(text)
    for match in re.finditer(r'\bDATA\b\s*(.*?)\s*/\s*(.*?)\s*/', normalized, re.I):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()
        if lhs:
            yield lhs, rhs


def parse_configs_inc(configs_path):
    """Parse configs.inc and extract mapconfig(0), SPROP and TPRID content robustly."""
    if not os.path.exists(configs_path):
        return {'nconfigs': 0, 'sprop_by_config': {}, 'tprid_by_config': {}}

    with open(configs_path) as f:
        text = f.read()

    nconfigs = 0
    sprop_by_config = {}
    tprid_by_config = {}

    for lhs, rhs in _iter_data_assignments(text):
        # Handle direct MAPCONFIG(0) initialization
        if re.search(r'\bMAPCONFIG\s*\(\s*0\s*\)', lhs, re.I):
            numbers = re.findall(r'[+-]?\d+', rhs)
            if numbers:
                nconfigs = max(nconfigs, int(numbers[0]))
            continue

        # Handle MAPCONFIG(I),I=0,... packed initialization
        if re.search(r'\bMAPCONFIG\s*\(\s*I\s*\)', lhs, re.I) and \
           re.search(r'I\s*=\s*0\s*,', lhs, re.I):
            numbers = re.findall(r'[+-]?\d+', rhs)
            if numbers:
                nconfigs = max(nconfigs, int(numbers[0]))
            continue

        # Handle TPRID(branch,config) data statements (t-channel propagators).
        tp_match = re.search(r'\bTPRID\s*\(\s*(-?\d+)\s*,\s*(\d+)\s*\)', lhs, re.I)
        if tp_match:
            config = int(tp_match.group(2))
            pdgs = [int(x) for x in re.findall(r'[+-]?\d+', rhs)]
            tprid_by_config.setdefault(config, []).extend(x for x in pdgs if x != 0)
            continue

        # Handle SPROP data statements.
        # Accept both SPROP(I,branch,config) and SPROP(idx,branch,config) forms.
        sp_match = re.search(
            r'\bSPROP\s*\(\s*[^,]+,\s*(-?\d+)\s*,\s*(\d+)\s*\)',
            lhs, re.I
        )
        if not sp_match:
            continue

        config = int(sp_match.group(2))
        pdgs = [int(x) for x in re.findall(r'[+-]?\d+', rhs)]
        pdgs = [x for x in pdgs if x != 0]
        if config not in sprop_by_config:
            sprop_by_config[config] = []
        sprop_by_config[config].extend(pdgs)

    if nconfigs == 0 and sprop_by_config:
        nconfigs = max(sprop_by_config)

    return {'nconfigs': nconfigs, 'sprop_by_config': sprop_by_config,
            'tprid_by_config': tprid_by_config}


def get_channel_count(Pdir):
    """Read number of integration channels from configs.inc.

    Returns 0 when configs.inc is missing or no config information is found.
    """
    parsed = parse_configs_inc(pjoin(Pdir, 'configs.inc'))
    return parsed['nconfigs']


def _flavor_blind_pdg(pdg):
    """Map a propagator PDG code onto its apply_flavor_grouping class.

    The flavor-grouped output writes merged codes (81 quarks, 82 charged
    leptons, 83 neutrinos) where the ungrouped output has explicit flavors,
    so the explicit codes are mapped onto the merged ones before comparing.
    """
    pdg = abs(pdg)
    if 1 <= pdg <= 5:
        return 81
    if pdg in (11, 13, 15):
        return 82
    if pdg in (12, 14, 16):
        return 83
    return pdg


def get_channel_topology(Pdir):
    """Read channel topology info from configs.inc to allow channel matching.

    Returns a dict mapping config_number -> topology_signature where the
    signature is a pair of sorted tuples (s-channel propagators, t-channel
    propagators), with flavors mapped by _flavor_blind_pdg. This can be used
    to match corresponding channels between apply_flavor_grouping=True/False
    outputs.

    Returns:
        dict: {config_number: topology_signature}
    """
    parsed = parse_configs_inc(pjoin(Pdir, 'configs.inc'))
    topologies = {}
    for config in range(1, parsed['nconfigs'] + 1):
        sprops = parsed['sprop_by_config'].get(config, [])
        tprops = parsed['tprid_by_config'].get(config, [])
        topologies[config] = (tuple(sorted(_flavor_blind_pdg(x) for x in sprops)),
                              tuple(sorted(_flavor_blind_pdg(x) for x in tprops)))
    return topologies


def find_channels_with_topology(Pdirs, signature):
    """Return the (Pdir, channel) pairs, over all Pdirs, whose topology is signature.

    Without subprocess grouping, apply_flavor_grouping=False splits the
    flavor combinations that a single flavor-grouped channel integrates over
    several P directories (e.g. P0_uux_uux and P0_uux_ddx both carry the
    s-channel gluon of q q~ > q q~), so a channel has to be compared with the
    sum of all its counterparts.
    """
    return [(Pdir, ch)
            for Pdir in sorted(Pdirs)
            for ch, sig in sorted(get_channel_topology(Pdir).items())
            if sig == signature]


def read_results_dat(results_path):
    """Read cross-section and error from a results.dat file.

    Returns (xsec, error, nevents) or None if file doesn't exist.
    """
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        xsec = float(parts[0])
        error = float(parts[1])
        nevents = int(parts[3]) if len(parts) >= 4 else 0
        return (xsec, error, nevents)
    except (ValueError, IndexError):
        return None


def generate_process(run_dir, process, model, defines, apply_fg, group_subprocesses=False):
    """Generate a MG5 process output directory.

    Args:
        run_dir: output directory path
        process: process string (e.g. 'q q~ > q q~')
        model: model name (e.g. 'sm')
        defines: list of define strings
        apply_fg: whether to apply flavor grouping
        group_subprocesses: whether to group subprocesses
    """
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    mg_cmd = MGCmd.MasterCmd()
    mg_cmd.no_notification()
    mg_cmd.exec_cmd('set automatic_html_opening False --no_save')
    mg_cmd.exec_cmd('set apply_flavor_grouping %s' % apply_fg)
    mg_cmd.exec_cmd('import model %s' % model)
    mg_cmd.exec_cmd('set group_subprocesses %s' % group_subprocesses)

    for define_str in defines:
        mg_cmd.exec_cmd('define %s' % define_str)

    mg_cmd.exec_cmd('generate %s' % process)
    mg_cmd.exec_cmd('output madevent %s' % run_dir)

    return run_dir


def configure_mlm(run_dir, xqcut):
    """Configure MLM merging parameters in the run card.

    Args:
        run_dir: the madevent output directory
        xqcut: the xqcut value for MLM merging
    """
    run_card_path = pjoin(run_dir, 'Cards', 'run_card.dat')
    run_card = banner.RunCardLO(run_card_path)
    run_card.set('ickkw', 1, user=True)
    run_card.set('xqcut', xqcut, user=True)
    # Pin the compiled-in PDF. This harness builds and runs ./madevent by hand
    # (make in Source, make in the P dir) and never goes through
    # configure_directory, so nothing links LHAPDF or fills lib/PDFsets. With
    # the run_card default (pdlabel=lhapdf) the binary is then built against the
    # internal PDF while the card asks for an LHAPDF set, and the integration
    # silently returns a zero cross-section. Which PDF is used is irrelevant
    # here: the test compares apply_flavor_grouping True vs False on the same
    # one.
    run_card.set('pdlabel', 'nn23lo1', user=True)
    run_card.set('lhaid', 230000, user=True)
    run_card.write(run_card_path)
    # Cards/run_card.dat is not what the binary reads: Source/run_card.inc is,
    # and nothing regenerates it here (make does not, and configure_directory
    # never runs), so it would otherwise keep the values from output time.
    run_card.write_include_file(pjoin(run_dir, 'Source'))


def compile_madevent(run_dir, Pdir):
    """Compile madevent binary in the subprocess directory.

    Tries to build madevent_forhel first (works with matrix*_orig.f from grouped
    subprocesses). Falls back to madevent (works with matrix*.f from ungrouped).

    Args:
        run_dir: the madevent output directory
        Pdir: absolute path to the subprocess directory (P*)

    Returns:
        Name of the compiled binary ('madevent_forhel' or 'madevent'),
        or None if compilation failed.
    """
    # Compile Source libraries
    ret = subprocess.Popen(
        ['make'], cwd=pjoin(run_dir, 'Source'),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).wait()
    if ret != 0:
        return None

    # Try madevent_forhel first (works with matrix*_orig.f)
    ret = subprocess.Popen(
        ['make', 'madevent_forhel'], cwd=Pdir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).wait()
    if ret == 0 and os.path.exists(pjoin(Pdir, 'madevent_forhel')):
        return 'madevent_forhel'

    # Fall back to madevent (works with matrix*.f)
    ret = subprocess.Popen(
        ['make', 'madevent'], cwd=Pdir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).wait()
    if ret == 0 and os.path.exists(pjoin(Pdir, 'madevent')):
        return 'madevent'

    return None


def write_run_config(Pdir, channel, npoints=5000, maxiter=5):
    """Write the run_config.txt (input for madevent_forhel via stdin).

    The format expected by madevent_forhel (get_user_params) is:
        Line 1: npoints maxiter miniter
        Line 2: accuracy
        Line 3: gridmode  (0=none, 2=adjust, -2=adjust parallel)
        Line 4: suppress_amp (0=no, 1=yes)
        Line 5: helicity (0=exact sum, -1=determine zero hels, N=MC N/event)
        Line 6: config_number (positive = single channel)

    A positive config_number selects that single channel (mincfig=maxcfig=config).
    A negative value means integrate all channels from 1 to abs(config).

    Args:
        Pdir: subprocess directory path
        channel: the channel (configuration) number to integrate
        npoints: number of phase-space points per iteration
        maxiter: maximum number of iterations

    Returns:
        path to the written config file
    """
    config_path = pjoin(Pdir, 'run_config_ch%d.txt' % channel)
    with open(config_path, 'w') as f:
        f.write('%d %d %d\n' % (npoints, maxiter, 1))
        f.write('0.1\n')          # accuracy
        f.write('2\n')            # grid adjustment
        f.write('1\n')            # suppress amplitude
        f.write('0\n')            # helicity: exact
        f.write('%d\n' % channel) # config number (positive = single channel)

    return config_path


def run_single_channel(Pdir, channel, npoints=5000, maxiter=5, binary='madevent'):
    """Run madevent for a single integration channel.

    Args:
        Pdir: subprocess directory path (absolute)
        channel: the channel number (positive = single channel)
        npoints: number of PS points per iteration
        maxiter: max iterations
        binary: name of the compiled binary ('madevent' or 'madevent_forhel')

    Returns:
        dict with keys: 'xsec', 'error', 'nevents', 'log', 'return_code'
    """
    # Write input config file
    config_path = write_run_config(Pdir, channel, npoints, maxiter)

    # A Fortran STOP exits with 0 without writing results.dat: never read
    # the one of a previous channel run in the same directory.
    results_path = pjoin(Pdir, 'results.dat')
    if os.path.exists(results_path):
        os.remove(results_path)

    # Run madevent with stdin from config file
    with open(config_path) as config_in:
        proc = subprocess.Popen(
            ['./%s' % binary],
            cwd=Pdir,
            stdin=config_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        stdout, _ = proc.communicate()
    log_text = stdout.decode('ascii', errors='ignore')

    result = {
        'return_code': proc.returncode,
        'log': log_text,
        'xsec': 0.0,
        'error': 0.0,
        'nevents': 0,
    }

    if proc.returncode != 0:
        return result

    # Try to read results from results.dat
    parsed = read_results_dat(results_path)
    if parsed:
        result['xsec'], result['error'], result['nevents'] = parsed

    return result


def sigma_difference(value_a, err_a, value_b, err_b):
    """Return |a-b|/sqrt(err_a^2+err_b^2), handling zero-error edge cases."""
    combined_err = math.sqrt(err_a ** 2 + err_b ** 2)
    if combined_err > 0:
        return abs(value_a - value_b) / combined_err
    if value_a == value_b:
        return 0.0
    return float('inf')


# ============================================================================
# Shared base utilities
# ============================================================================
class MLMReweightTestBase(unittest.TestCase):
    """Shared setup and process-preparation helpers for MLM reweight tests."""

    tmp_prefix = 'acc_test_mlm_'
    debug_dir = 'tmp_test_mlm'

    def setUp(self):
        self.debugging = getattr(unittest, 'debug', False)
        if self.debugging:
            self.path = pjoin(MG5DIR, self.debug_dir)
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.makedirs(self.path)
        else:
            self.path = tempfile.mkdtemp(prefix=self.tmp_prefix)

    def tearDown(self):
        if not self.debugging:
            shutil.rmtree(self.path, ignore_errors=True)

    def prepare_mode_run(self, run_dir, process, model, defines, xqcut,
                         apply_fg, group_subprocesses=False,
                         subproc_pattern=None, pdir_selector=None):
        """Generate, configure MLM, choose subprocess, compile, and return run info."""
        generate_process(
            run_dir=run_dir,
            process=process,
            model=model,
            defines=defines,
            apply_fg=apply_fg,
            group_subprocesses=group_subprocesses
        )
        configure_mlm(run_dir, xqcut)

        if pdir_selector:
            Pdir = pdir_selector(run_dir)
        else:
            Pdir = select_Pdir(run_dir, subproc_pattern)

        binary = compile_madevent(run_dir, Pdir)
        self.assertIsNotNone(
            binary,
            'Compilation failed (apply_flavor_grouping=%s, subproc=%s)' %
            (apply_fg, os.path.basename(Pdir))
        )
        return run_dir, Pdir, binary


# ============================================================================
# Test Factory
# ============================================================================
def make_mlm_channel_test(config_name, config):
    """Factory function that creates a test method for a given MLM channel config.

    This creates a test that:
    1. Generates the process with apply_flavor_grouping=True, runs a specific channel
    2. Generates the process with apply_flavor_grouping=False, runs the corresponding channel
    3. Compares cross-sections: they should agree within statistical uncertainty

    Args:
        config_name: name for the test (used in method name)
        config: dict with test configuration (see MLM_TEST_CHANNELS)

    Returns:
        A test method function.
    """

    def test_method(self):
        """Compare MLM REWGT between flavor_grouping True/False for channel %s."""

        results = {}

        for fg_mode in [True, False]:
            fg_label = 'fg_true' if fg_mode else 'fg_false'
            run_dir = pjoin(self.path, 'MLM_%s_%s' % (config_name, fg_label))
            mode_config = config[fg_label]
            group_subp = mode_config.get('group_subprocesses',
                                         config.get('group_subprocesses', False))

            # 1-3. Generate, configure, select subprocess, compile
            _, Pdir, binary = self.prepare_mode_run(
                run_dir=run_dir,
                process=config['process'],
                model=config['model'],
                defines=config.get('defines', []),
                xqcut=config['xqcut'],
                apply_fg=fg_mode,
                group_subprocesses=group_subp,
                subproc_pattern=mode_config.get('subproc')
            )

            # 4. Determine channel count and validate channel choice
            nchan = get_channel_count(Pdir)
            channel = mode_config['channel']

            if nchan > 0:
                self.assertLessEqual(channel, nchan,
                    'Channel %d exceeds available channels (%d) for %s '
                    '(apply_flavor_grouping=%s, subproc=%s)' %
                    (channel, nchan, config_name, fg_mode, os.path.basename(Pdir)))

            # 5. Run single channel integration
            result = run_single_channel(
                Pdir, channel,
                npoints=config.get('npoints', 5000),
                maxiter=config.get('maxiter', 5),
                binary=binary,
            )

            self.assertEqual(result['return_code'], 0,
                'madevent returned non-zero for %s '
                '(apply_flavor_grouping=%s, subproc=%s, channel=%d)\nLog:\n%s' %
                (config_name, fg_mode, os.path.basename(Pdir), channel,
                 result['log'][-2000:]))

            results[fg_label] = result

            if getattr(unittest, "debug", False):
                misc.sprint('  %s (fg=%s, subproc=%s, ch=%d): xsec=%s +- %s' %
                    (config_name, fg_mode, os.path.basename(Pdir), channel,
                     result['xsec'], result['error']))

        # 6. Compare results between the two modes
        xsec_true = results['fg_true']['xsec']
        err_true = results['fg_true']['error']
        xsec_false = results['fg_false']['xsec']
        err_false = results['fg_false']['error']

        # Both should give non-zero cross-sections
        self.assertGreater(abs(xsec_true), 0,
            'Zero cross-section for apply_flavor_grouping=True in %s' % config_name)
        self.assertGreater(abs(xsec_false), 0,
            'Zero cross-section for apply_flavor_grouping=False in %s' % config_name)

        # Cross-sections should agree within 5 sigma combined uncertainty
        sigma_diff = sigma_difference(xsec_true, err_true, xsec_false, err_false)
        self.assertLess(sigma_diff, 5,
            'Cross-sections differ by %.1f sigma between '
            'apply_flavor_grouping=True (%s +- %s) and '
            'apply_flavor_grouping=False (%s +- %s) for %s' %
            (sigma_diff, xsec_true, err_true, xsec_false, err_false, config_name))

    test_method.__doc__ = (
        'Compare MLM REWGT between flavor_grouping True/False for %s '
        '(subproc/channel: fg_true=%s/%d, fg_false=%s/%d).' %
        (config_name,
         config['fg_true'].get('subproc', 'auto'), config['fg_true']['channel'],
         config['fg_false'].get('subproc', 'auto'), config['fg_false']['channel'])
    )
    return test_method


# ============================================================================
# Test Class
# ============================================================================
class TestMLMReweight(MLMReweightTestBase):
    """Test suite for MLM reweighting comparing apply_flavor_grouping modes.

    Tests verify that the REWGT function (from reweight.f linked to cluster.f)
    produces consistent results when running with apply_flavor_grouping=True
    vs False for specified integration channels.

    To add a new test case, add an entry to MLM_TEST_CHANNELS above.
    Channel numbers may differ between the two modes because different
    flavor grouping can lead to different diagram topologies being generated.
    """


# Dynamically add test methods from the factory
for _name, _config in MLM_TEST_CHANNELS.items():
    _test_method = make_mlm_channel_test(_name, _config)
    _test_method_name = 'test_mlm_rewgt_%s' % _name
    setattr(TestMLMReweight, _test_method_name, _test_method)


class TestMLMReweightAutoMatch(MLMReweightTestBase):
    """Test suite using automatic channel matching between flavor grouping modes.

    Instead of manually specifying channel numbers for each mode, this test
    generates both outputs without subprocess grouping, uses topology matching
    to identify corresponding channels, and compares each flavor-grouped
    channel with the sum of its counterparts over all ungrouped P directories.
    """

    def test_mlm_rewgt_auto_match_qq_to_qq(self):
        """Compare MLM REWGT for q q~ > q q~ using automatic channel matching.

        The s-channel gluon channel is the sensitive one: for the
        q q~ > q' q~' rows the flavor-grouped clustering used to pick the q-q'
        t-channel gluon of the merged graphs, which does not exist for those
        flavors, so the event skipped the alpha_s reweighting of the d d~ > g
        clustering (+13% on that channel at xqcut=20).
        """

        process = 'q q~ > q q~'
        model = 'sm'
        defines = ['q = u d s c', 'q~ = u~ d~ s~ c~']
        xqcut = 20.0
        # 5000 points leave the +13% above at only ~4 sigma
        npoints = 50000
        maxiter = 5

        Pdirs_by_mode = {}

        for fg_mode in [True, False]:
            fg_label = 'fg_true' if fg_mode else 'fg_false'
            run_dir = pjoin(self.path, 'MLM_auto_%s' % fg_label)

            _, Pdir, binary = self.prepare_mode_run(
                run_dir=run_dir,
                process=process,
                model=model,
                defines=defines,
                xqcut=xqcut,
                apply_fg=fg_mode,
                group_subprocesses=False,
            )
            binaries = {Pdir: binary}
            for pname in get_Pdir(run_dir):
                other = pjoin(run_dir, 'SubProcesses', pname)
                if other not in binaries:
                    binaries[other] = compile_madevent(run_dir, other)
                    self.assertIsNotNone(binaries[other],
                        'Compilation failed (apply_flavor_grouping=%s, subproc=%s)' %
                        (fg_mode, pname))
            Pdirs_by_mode[fg_label] = binaries

        def run_sum(binaries, channels):
            xsec, err2 = 0., 0.
            for Pdir, ch in channels:
                result = run_single_channel(Pdir, ch, npoints=npoints,
                                            maxiter=maxiter, binary=binaries[Pdir])
                self.assertEqual(result['return_code'], 0,
                    'Run failed for %s channel %d' % (os.path.basename(Pdir), ch))
                xsec += result['xsec']
                err2 += result['error'] ** 2
            return xsec, math.sqrt(err2)

        def describe(channels):
            return ' + '.join('%s ch=%d' % (os.path.basename(Pdir), ch)
                              for Pdir, ch in channels)

        signatures = set()
        for Pdir in Pdirs_by_mode['fg_true']:
            signatures.update(get_channel_topology(Pdir).values())
        self.assertGreater(len(signatures), 0, 'No channel found for fg=True')

        for signature in sorted(signatures):
            channels_t = find_channels_with_topology(Pdirs_by_mode['fg_true'], signature)
            channels_f = find_channels_with_topology(Pdirs_by_mode['fg_false'], signature)
            self.assertGreater(len(channels_f), 0,
                'No fg=False channel matches the topology %s of %s' %
                (signature, describe(channels_t)))

            xsec_t, err_t = run_sum(Pdirs_by_mode['fg_true'], channels_t)
            xsec_f, err_f = run_sum(Pdirs_by_mode['fg_false'], channels_f)

            if getattr(unittest, "debug", False):
                misc.sprint('  fg=True  %s: %s +- %s' % (describe(channels_t), xsec_t, err_t))
                misc.sprint('  fg=False %s: %s +- %s' % (describe(channels_f), xsec_f, err_f))

            self.assertGreater(abs(xsec_t), 0,
                'Zero xsec for fg=True %s' % describe(channels_t))
            self.assertGreater(abs(xsec_f), 0,
                'Zero xsec for fg=False %s' % describe(channels_f))

            sigma_diff = sigma_difference(xsec_t, err_t, xsec_f, err_f)
            self.assertLess(sigma_diff, 5,
                'Cross-sections differ by %.1f sigma: '
                'fg=True %s (%s +- %s) vs fg=False %s (%s +- %s)' %
                (sigma_diff, describe(channels_t), xsec_t, err_t,
                 describe(channels_f), xsec_f, err_f))


# ============================================================================
# Convenience: run with `python test_MLM_reweight.py`
# ============================================================================
if __name__ == '__main__':
    if not hasattr(unittest, 'debug'):
        unittest.debug = False
    unittest.main()
