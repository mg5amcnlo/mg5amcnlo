################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
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
"""MG7 custom cut, scale, PDF and bias functions ([custom_functions]).

Each custom function is created with bin/init_custom_func in a copy of one
``p p > t t~`` output, given a body that reproduces a built-in setting, and
compared with a run using that built-in setting:

  * cut: pT > 150 GeV for both tops against ``top-pt.min = 150``, and every
    event passes the cut;
  * scale: H_T/2 against ``dynamical_scale_choice = half_transverse_mass``,
    and the LHE scale of every event is H_T/2 of its momenta;
  * pdf: the LHAPDF set of the run card called through the LHAPDF C++ API
    against the built-in interpolation of the same set;
  * bias: b = max((pT / 100 GeV)^2, 1) keeps the cross section and the
    weighted pT tail, while putting more events into the tail;
  * every function in every language with a toolchain here builds, loads and
    returns its default value when evaluated through madspace.

Run locally with::

    ./tests/test_manager.py test_.*custom.*_mg7 -pA -t0

``MG7_CUSTOM_EVENTS`` sets the events per run (default 5000).
"""

from __future__ import absolute_import
from __future__ import division

import glob
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from tests.acceptance_tests.test_check_xsec_processes_mg7 import (
    _REFERENCE_PDF, _edit_run_card, _require_mg7_runtime, _tail)

pjoin = os.path.join

_PROCESS = 'p p > t t~'
_EVENTS = int(os.environ.get('MG7_CUSTOM_EVENTS', 5000))
_NSIGMA = 3
_SEED = 1234
_TOP_PT_CUT = 150.

# bodies of the custom functions, in C++. The cut takes its threshold from the
# run card, through the parameter pt_min of parameters.h.
_CUT_BODY = r'''
#include <cmath>
#include "parameters.h"
extern "C" bool custom_cut(int particle_count, const int* pdg_ids, const double momenta[][4],
                           const CustomParameters* params) {
    for (int i = 2; i < 4; ++i) {
        if (std::hypot(momenta[i][1], momenta[i][2]) < params->pt_min) {
            return false;
        }
    }
    return true;
}
'''
_CUT_PARAMETERS = 'X(pt_min, 20.) X(unused, 1.)'
_CUT_CARD = {'pt_min': _TOP_PT_CUT}

# forward jets only: asymmetric under the beam mirror, so the cut must see the
# momenta the event is written with
_FORWARD_JET_BODY = r'''
#include "parameters.h"
extern "C" bool custom_cut(int particle_count, const int* pdg_ids, const double momenta[][4],
                           const CustomParameters* params) {
    return momenta[4][3] > 0.;
}
'''

_SCALE_BODY = r'''
#include <cmath>
#include "parameters.h"
extern "C" void custom_scale(int particle_count, const int* pdg_ids, const double momenta[][4],
                             const CustomParameters* params, double* ren_scale,
                             double* fact_scale1, double* fact_scale2) {
    double ht = 0.;
    for (int i = 2; i < particle_count; ++i) {
        double e = momenta[i][0], pz = momenta[i][3];
        ht += std::sqrt(std::fmax(e * e - pz * pz, 0.));
    }
    *ren_scale = *fact_scale1 = *fact_scale2 = ht / 2.;
}
'''

_PDF_BODY = r'''
#include <mutex>
#include "LHAPDF/LHAPDF.h"
#include "parameters.h"
extern "C" double custom_pdf(int beam_index, int pdg_id, double x, double q,
                             const CustomParameters* params) {
    static LHAPDF::PDF* pdf = LHAPDF::mkPDF("%(pdf)s", 0);
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    return pdf->xfxQ(pdg_id, x, q);
}
''' % {'pdf': _REFERENCE_PDF}

_BIAS_BODY = r'''
#include <cmath>
#include "parameters.h"
extern "C" double custom_bias(int particle_count, const int* pdg_ids, const double momenta[][4],
                              const CustomParameters* params) {
    double pt2 = momenta[2][1] * momenta[2][1] + momenta[2][2] * momenta[2][2];
    return std::fmax(pt2 / (100. * 100.), 1.);
}
'''

# the same functions in Python with numpy, and the cut with torch
_PY_HEADER = 'import numpy as np\nFRAMEWORK = "numpy"\n'

_CUT_NUMPY = _PY_HEADER + '''
def custom_cut(momenta, pdg_ids):
    pt = np.hypot(momenta[:, 2:4, 1], momenta[:, 2:4, 2])
    return (pt >= %(cut)r).all(axis=1)
''' % {'cut': _TOP_PT_CUT}

# the torch cut takes its threshold from the run card, the numpy one has none
_CUT_TORCH = '''import torch
FRAMEWORK = "torch"
PARAMETERS = {"pt_min": 20.}

def custom_cut(momenta, pdg_ids, params):
    pt = torch.hypot(momenta[:, 2:4, 1], momenta[:, 2:4, 2])
    return (pt >= params.pt_min).all(dim=1)
'''

_SCALE_NUMPY = _PY_HEADER + '''
def custom_scale(momenta, pdg_ids):
    final = momenta[:, 2:]
    return np.sqrt(np.maximum(final[..., 0] ** 2 - final[..., 3] ** 2, 0.)).sum(axis=1) / 2.
'''

_PDF_NUMPY = _PY_HEADER + '''import lhapdf
pdf_set = lhapdf.mkPDF("%(pdf)s", 0)

def custom_pdf(beam_index, pdg_id, x, q):
    return np.array([pdf_set.xfxQ(int(i), a, b) for i, a, b in zip(pdg_id, x, q)])
''' % {'pdf': _REFERENCE_PDF}

_BIAS_NUMPY = _PY_HEADER + '''
def custom_bias(momenta, pdg_ids):
    pt2 = momenta[:, 2, 1] ** 2 + momenta[:, 2, 2] ** 2
    return np.maximum(pt2 / 100. ** 2, 1.)
'''

_EXTENSIONS = {'cpp': 'cpp', 'python-numpy': 'py', 'python-torch': 'py'}


def _set(text, key, value):
    """Set ``key = value`` (first occurrence at the start of a line)."""
    text, count = re.subn(r'(?m)^%s = .*$' % re.escape(key),
                          '%s = %s' % (key, value), text, count=1)
    assert count == 1, 'no %s in the run card' % key
    return text


def _read_lhe(path):
    """(weights, scales, momenta) of an LHE file; momenta[i] lists the
    (E, px, py, pz) of every particle of event i."""
    weights, scales, momenta = [], [], []
    with gzip.open(path, 'rt') as f:
        lines = iter(f)
        for line in lines:
            if not line.startswith('<event'):
                continue
            fields = next(lines).split()
            count = int(fields[0])
            weights.append(float(fields[2]))
            scales.append(float(fields[3]))
            event = []
            for _ in range(count):
                p = next(lines).split()
                event.append((float(p[9]), float(p[6]), float(p[7]), float(p[8])))
            momenta.append(event)
    return weights, scales, momenta


def _pt(p):
    return math.hypot(p[1], p[2])


def _import_madspace():
    """The madspace of this MadGraph, as bin/generate_events imports it, or
    None. Done before anything imports another madspace from the path."""
    try:
        from madgraph.mg7.bootstrap import (
            ensure_madspace, drop_install_path)
        ensure_madspace(interactive=False)
        import madspace
        drop_install_path()
        return madspace
    except Exception:
        return None


class MG7CustomFunctionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ms = _import_madspace()
        cls.path = tempfile.mkdtemp(prefix='mg7_custom_')
        cls.base = pjoin(cls.path, 'BASE')
        cls.reference = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def base_process(self):
        """The ``p p > t t~`` output shared by the tests: fixed scales, the
        PDF set of the xsec references, LHE output and a fixed seed. The first
        call also generates (and compiles) it with a reference run."""
        _require_mg7_runtime(self)
        cls = type(self)
        if cls.reference is None:
            mg = MGCmd.MasterCmd()
            mg.no_notification()
            mg.exec_cmd('set automatic_html_opening False --no_save')
            mg.exec_cmd('generate %s' % _PROCESS)
            mg.exec_cmd('output mg7 %s' % self.base)
            toml = pjoin(self.base, 'Cards', 'run_card.toml')
            _edit_run_card(toml, _EVENTS, disable_jet_cuts=False)
            text = open(toml).read()
            text = _set(text, 'output_format', '"lhe"')
            text = _set(text, 'seed', _SEED)
            text = text.replace('[multiparticles]\n', '[multiparticles]\ntop = [6, -6]\n')
            open(toml, 'w').write(text)
            cls.reference = self.run_process(self.base)
        return self.base

    def copy_process(self, name):
        target = pjoin(self.path, name)
        shutil.copytree(self.base_process(), target, symlinks=True)
        # the copy starts without the events of the reference run
        shutil.rmtree(pjoin(target, 'Events'), ignore_errors=True)
        os.mkdir(pjoin(target, 'Events'))
        return target

    def edit_card(self, proc_dir, edit):
        toml = pjoin(proc_dir, 'Cards', 'run_card.toml')
        text = open(toml).read()
        open(toml, 'w').write(edit(text))

    def init_custom_function(self, proc_dir, kind, body, language='cpp',
                             parameters_h=None, card_parameters=None):
        """bin/init_custom_func for ``kind`` in ``language``, with ``body``;
        the run builds C++ functions. ``parameters_h`` is the X-macro list of
        parameters.h, ``card_parameters`` the values set in the run card."""
        log = pjoin(proc_dir, 'init_custom_func.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(proc_dir, 'bin', 'init_custom_func'),
                 '-f', kind, '-l', language, '-y', '--no-build'],
                cwd=proc_dir, stdout=logfh, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'bin/init_custom_func failed:\n%s' % _tail(log))
        body_file = 'custom_%s.%s' % (kind, _EXTENSIONS[language])
        with open(pjoin(proc_dir, 'custom_%s' % kind, body_file), 'w') as f:
            f.write(body)
        if parameters_h:
            path = pjoin(proc_dir, 'custom_%s' % kind, 'parameters.h')
            text = open(path).read().replace(
                '#define CUSTOM_PARAMETERS(X)\n',
                '#define CUSTOM_PARAMETERS(X) %s\n' % parameters_h, 1)
            open(path, 'w').write(text)
        if card_parameters:
            lines = ''.join('%s_parameters.%s = %r\n' % (kind, name, value)
                            for name, value in card_parameters.items())
            self.edit_card(proc_dir, lambda t: t.replace(
                '[custom_functions]\n', '[custom_functions]\n' + lines, 1))

    def builtin_reference(self, name, edit):
        """The run with the built-in setting ``edit``, shared by the languages."""
        cls = type(self)
        if not hasattr(cls, 'builtin_runs'):
            cls.builtin_runs = {}
        if name not in cls.builtin_runs:
            builtin = self.copy_process(name)
            self.edit_card(builtin, edit)
            cls.builtin_runs[name] = self.run_process(builtin)
        return cls.builtin_runs[name]

    def copy_for(self, kind, language):
        return self.copy_process('%s_%s' % (kind.upper(), language.upper().replace('-', '_')))

    def run_process(self, proc_dir):
        """Run bin/generate_events: (cross section, error, LHE path)."""
        log = pjoin(proc_dir, 'mg7_gen.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(proc_dir, 'bin', 'generate_events'), '-f'],
                cwd=proc_dir, stdout=logfh, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'mg7 generate_events failed:\n%s' % _tail(log))
        infos = sorted(glob.glob(pjoin(proc_dir, 'Events', '*', 'info.json')))
        self.assertTrue(infos, 'no info.json produced:\n%s' % _tail(log))
        with open(infos[-1]) as f:
            info = json.load(f)['process']
        lhe = pjoin(os.path.dirname(infos[-1]), 'events.lhe.gz')
        self.assertTrue(os.path.exists(lhe), 'no LHE file produced')
        return float(info['mean']), float(info['error']), lhe

    def assertCompatible(self, got, ref, what):
        """Cross sections (mean, error) agree within _NSIGMA."""
        sigma = math.hypot(got[1], ref[1])
        self.assertLessEqual(
            abs(got[0] - ref[0]), _NSIGMA * sigma,
            '%s: %.5g +- %.3g pb against %.5g +- %.3g pb'
            % (what, got[0], got[1], ref[0], ref[1]))

    def check_cut(self, language, body, **parameters):
        ref = self.builtin_reference('CUT_BUILTIN', lambda t: t.replace(
            '[cuts]\n', '[cuts]\ntop-pt.min = %r\n' % _TOP_PT_CUT))
        custom = self.copy_for('cut', language)
        self.init_custom_function(custom, 'cut', body, language, **parameters)
        got = self.run_process(custom)
        self.assertCompatible(got, ref, 'custom cut in %s' % language)

        _, _, momenta = _read_lhe(got[2])
        for event in momenta:
            for p in event[2:4]:
                # up to the precision of the LHE file
                self.assertGreaterEqual(_pt(p), _TOP_PT_CUT * (1 - 1e-6))

    def test_custom_cut_mg7(self):
        self.check_cut('cpp', _CUT_BODY, parameters_h=_CUT_PARAMETERS,
                       card_parameters=_CUT_CARD)

    def test_custom_cut_numpy_mg7(self):
        self.check_cut('python-numpy', _CUT_NUMPY)

    def test_custom_cut_torch_mg7(self):
        check = subprocess.run([sys.executable, '-c', 'import numpy, torch'],
                               capture_output=True)
        if check.returncode != 0:
            self.skipTest('torch is unavailable')
        self.check_cut('python-torch', _CUT_TORCH, card_parameters=_CUT_CARD)

    def test_custom_cut_mirror_mg7(self):
        """A pz > 0 jet cut on p p > t t~ j halves the cross section, and every
        written event passes it, also for the mirrored flavors."""
        _require_mg7_runtime(self)
        proc_dir = pjoin(self.path, 'TTJ')
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        mg.exec_cmd('generate p p > t t~ j')
        mg.exec_cmd('output mg7 %s' % proc_dir)
        toml = pjoin(proc_dir, 'Cards', 'run_card.toml')
        _edit_run_card(toml, _EVENTS, disable_jet_cuts=False)
        self.edit_card(proc_dir, lambda t: _set(_set(
            t, 'output_format', '"lhe"'), 'seed', _SEED))
        ref = self.run_process(proc_dir)

        shutil.rmtree(pjoin(proc_dir, 'Events'))
        os.mkdir(pjoin(proc_dir, 'Events'))
        self.init_custom_function(proc_dir, 'cut', _FORWARD_JET_BODY)
        got = self.run_process(proc_dir)
        self.assertCompatible(got, (ref[0] / 2, ref[1] / 2), 'forward jet cut')
        _, _, momenta = _read_lhe(got[2])
        self.assertTrue(all(event[4][3] > 0. for event in momenta),
                        'events with a backward jet were written')

    def check_scale(self, language, body):
        def dynamic(text):
            text = _set(text, 'fixed_ren_scale', 'false')
            text = _set(text, 'fixed_fact_scale', 'false')
            return _set(text, 'dynamical_scale_choice', '"half_transverse_mass"')

        ref = self.builtin_reference('SCALE_BUILTIN', dynamic)
        custom = self.copy_for('scale', language)
        # partonic_energy differs from H_T/2, so the custom function must win
        self.edit_card(custom, lambda t: _set(
            dynamic(t), 'dynamical_scale_choice', '"partonic_energy"'))
        self.init_custom_function(custom, 'scale', body, language)
        got = self.run_process(custom)
        self.assertCompatible(got, ref, 'custom scale in %s' % language)

        _, scales, momenta = _read_lhe(got[2])
        for scale, event in zip(scales, momenta):
            ht = sum(math.sqrt(max(p[0] ** 2 - p[3] ** 2, 0.)) for p in event[2:])
            self.assertAlmostEqual(scale / (ht / 2), 1., places=5)

    def test_custom_scale_mg7(self):
        self.check_scale('cpp', _SCALE_BODY)

    def test_custom_scale_numpy_mg7(self):
        self.check_scale('python-numpy', _SCALE_NUMPY)

    def test_custom_pdf_mg7(self):
        lhapdf_config = misc.resolve_lhapdf().config
        if not lhapdf_config:
            self.skipTest('lhapdf-config not found')
        flags = {}
        for flag in ('cflags', 'libs', 'libdir'):
            flags[flag] = subprocess.check_output(
                [lhapdf_config, '--%s' % flag], text=True).strip()

        custom = self.copy_process('PDF_CUSTOM')
        self.init_custom_function(custom, 'pdf', _PDF_BODY)
        build = subprocess.run(
            ['make', 'CFLAGS=%s' % flags['cflags'],
             'LIBS=%s -Wl,-rpath,%s' % (flags['libs'], flags['libdir'])],
            cwd=pjoin(custom, 'custom_pdf'), capture_output=True, text=True)
        self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
        got = self.run_process(custom)
        self.assertCompatible(got, type(self).reference, 'custom pdf')

    def test_custom_pdf_numpy_mg7(self):
        check = subprocess.run([sys.executable, '-c', 'import lhapdf'],
                               capture_output=True)
        if check.returncode != 0:
            self.skipTest('the lhapdf Python module is unavailable')
        custom = self.copy_for('pdf', 'python-numpy')
        self.init_custom_function(custom, 'pdf', _PDF_NUMPY, 'python-numpy')
        got = self.run_process(custom)
        self.assertCompatible(got, type(self).reference, 'custom pdf in numpy')

    def test_custom_bias_mg7(self):
        self.check_bias('cpp', _BIAS_BODY)

    def test_custom_bias_numpy_mg7(self):
        self.check_bias('python-numpy', _BIAS_NUMPY)

    def check_bias(self, language, body):
        custom = self.copy_for('bias', language)
        ref = type(self).reference
        self.init_custom_function(custom, 'bias', body, language)
        got = self.run_process(custom)
        self.assertCompatible(got, ref, 'biased cross section in %s' % language)

        def tail(lhe, threshold=250.):
            weights, _, momenta = _read_lhe(lhe)
            n = len(weights)
            in_tail = [w for w, event in zip(weights, momenta)
                       if _pt(event[2]) > threshold]
            mean = sum(weights) / n
            error = math.sqrt(sum((w - mean) ** 2 for w in weights) / n / (n - 1))
            xsec = sum(in_tail) / n
            xsec_error = math.sqrt(sum(w ** 2 for w in in_tail)) / n
            return weights, (mean, error), (xsec, xsec_error), len(in_tail) / n

        ref_weights, _, ref_tail, ref_fraction = tail(ref[2])
        weights, mean, tail_xsec, fraction = tail(got[2])
        self.assertGreater(max(weights) / min(weights), 2.,
                           'the biased events should not have a constant weight')
        self.assertCompatible(mean, ref[:2], 'mean biased event weight')
        self.assertCompatible(tail_xsec, ref_tail, 'pT > 250 GeV cross section')
        self.assertGreater(fraction, 2 * ref_fraction,
                           'the bias should put more events into the tail')

    def check_defaults(self, ms, context, apis):
        """The default functions of bin/init_custom_func evaluated through
        madspace: 1 for cut and bias, sqrt(s_hat) for scale, the toy pdf."""
        import numpy as np
        momenta = np.array([[[500., 0., 0., 500.], [500., 0., 0., -500.],
                              [500., 100., 50., 200.], [500., -100., -50., -200.]],
                             [[300., 0., 0., 300.], [200., 0., 0., -200.],
                              [250., 30., 0., 100.], [250., -30., 0., 0.]]])
        for kind in ('cut', 'scale', 'bias'):
            function = ms.CustomFunction(
                apis[kind], getattr(ms.CustomFunction, kind), 4, [[21, 21, 6, -6]])
            values = ms.FunctionRuntime(function.function(), context)(momenta)
            # a scale has three outputs: renormalization and factorization scales
            outputs = values if kind == 'scale' else [values]
            for output in outputs:
                for value, event in zip(np.array(output), momenta):
                    if kind == 'scale':
                        p = [a + b for a, b in zip(event[0], event[1])]
                        target = math.sqrt(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2)
                    else:
                        target = 1.
                    self.assertAlmostEqual(value, target, places=9)
        pdf = ms.PartonDensity(ms.CustomPdf(apis['pdf']), [21, 2], 0)
        values = ms.FunctionRuntime(pdf.function(), context)(
            np.array([0.1, 0.5]), np.array([100., 100.]))
        for x, row in zip([0.1, 0.5], np.array(values)):
            toy = math.sqrt(x) * (1 - x) ** 3
            self.assertAlmostEqual(row[0], 3 * toy, places=12)
            self.assertAlmostEqual(row[1], 0.5 * toy, places=12)

    def test_init_custom_func_languages_mg7(self):
        ms = type(self).ms
        if ms is None or not hasattr(ms, 'CustomFunction'):
            self.skipTest('madspace with custom functions unavailable')
        compilers = {'c': 'cc', 'cpp': 'c++', 'fortran': 'gfortran',
                     'cuda': 'nvcc', 'hip': 'hipcc'}
        from madgraph.mg7 import init_custom_func

        tested = []
        for language, compiler in compilers.items():
            if shutil.which(compiler) is None:
                continue
            me_dir = pjoin(self.path, 'LANG_%s' % language)
            os.makedirs(me_dir)
            for kind in ('cut', 'scale', 'pdf', 'bias'):
                ret = init_custom_func.main(me_dir=me_dir, argv=[
                    '-f', kind, '-l', language, '-y', '--build', '--no-set-run-card'])
                self.assertEqual(ret, 0, '%s %s did not build' % (language, kind))
                self.assertTrue(os.path.exists(pjoin(me_dir, 'lib', 'custom_%s.so' % kind)))
            tested.append(language)
            if language in ('cuda', 'hip'):
                continue
            context = ms.Context(device=ms.cpu_device(), thread_count=1)
            self.check_defaults(ms, context, {
                kind: context.load_matrix_element(
                    pjoin(me_dir, 'lib', 'custom_%s.so' % kind), '')
                for kind in ('cut', 'scale', 'pdf', 'bias')})

        from madgraph.iolibs.template_files.mg7.custom_function_loader import (
            load_custom_function)
        for language in ('python-numpy', 'python-torch'):
            if language == 'python-torch':
                try:
                    import torch
                except Exception:
                    continue
            me_dir = pjoin(self.path, 'LANG_%s' % language)
            os.makedirs(me_dir)
            context = ms.Context(device=ms.cpu_device(), thread_count=2)
            apis = {}
            for kind in ('cut', 'scale', 'pdf', 'bias'):
                ret = init_custom_func.main(me_dir=me_dir, argv=[
                    '-f', kind, '-l', language, '-y', '--no-set-run-card'])
                self.assertEqual(ret, 0, '%s %s failed' % (language, kind))
                path = pjoin(me_dir, 'custom_%s' % kind, 'custom_%s.py' % kind)
                self.assertTrue(os.path.exists(path))
                apis[kind] = load_custom_function(ms, [context], ['cpu'], kind, path)[0]
            self.check_defaults(ms, context, apis)
            tested.append(language)
        if not tested:
            self.skipTest('no compiler found')
