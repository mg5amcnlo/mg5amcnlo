################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
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
"""mg7 runs of '[amplicol]' processes against the same processes with madmatrix.

The AmpliCol subprocesses are integrated at leading colour and re- or
unweighted to full colour, so their cross sections must agree with the
madmatrix ones within the Monte Carlo errors, alone and mixed with madmatrix
processes in one output. Needs madspace, pyamplicol (in pyamplicol/install) and
LHAPDF; the tests skip otherwise. Knobs: MG7_AMPLICOL_EVENTS (default 10000)
and MG7_AMPLICOL_NSIGMA (default 4).

    ./tests/test_manager.py test_mg7_amplicol -pA -t0
"""

import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc

pjoin = os.path.join

_EVENTS = int(os.environ.get('MG7_AMPLICOL_EVENTS', 10000))
_NSIGMA = float(os.environ.get('MG7_AMPLICOL_NSIGMA', 4))
_PDF = 'NNPDF23_lo_as_0130_qed'


def _edit_run_card(path, gridpack=False, output_format=None, **amplicol):
    card = open(path).read()
    if gridpack:
        card = re.sub(r'(?m)^save_gridpack = false', 'save_gridpack = true', card)
        card = re.sub(r'(?m)^compress = true', 'compress = false', card)
        # with events, for the cross section of the run
        card = re.sub(r'(?m)^run_mode = ".*"', 'run_mode = "regular"', card)
    if output_format:
        card = re.sub(r'(?m)^output_format = ".*"', 'output_format = "%s"' %
                      output_format, card)
    card = card.replace('fixed_ren_scale = false', 'fixed_ren_scale = true')
    card = card.replace('fixed_fact_scale = false', 'fixed_fact_scale = true')
    card = re.sub(r'(?m)^(pdf[12]) = ".*"$', r'\1 = "%s"' % _PDF, card)
    card = re.sub(r'(?m)^events = \d+', 'events = %d' % _EVENTS, card)
    card = re.sub(r'(?m)^seed = -1', 'seed = 7', card)
    card = re.sub(r'(?m)^make_plots = true', 'make_plots = false', card)
    card = re.sub(r'(\[systematics\]\nenable = )true', r'\1false', card)
    card = re.sub(r'(\[madnis\]\n(?:#.*\n)*enable = )"auto"', r'\1false', card)
    head, section = card.split('\n[amplicol]\n')
    for key, value in amplicol.items():
        section, count = re.subn(r'(?m)^%s = .*$' % key,
                                 '%s = "%s"' % (key, value), section, count=1)
        assert count == 1, key
    card = head + '\n[amplicol]\n' + section
    open(path, 'w').write(card)


def _color_balanced(lhe_path):
    """Whether every colour line of every event starts and ends once."""
    with gzip.open(lhe_path, 'rt') as stream:
        text = stream.read()
    for event in re.findall(r'<event>\n(.*?)</event>', text, re.S):
        lines = [line.split() for line in event.strip().splitlines()[1:]
                 if line.strip() and not line.startswith('<')]
        tags = {}
        for particle in lines:
            status, color, anticolor = int(particle[1]), int(particle[4]), int(particle[5])
            if status == -1:
                color, anticolor = anticolor, color
            for tag, sign in ((color, 1), (anticolor, -1)):
                if tag:
                    tags[tag] = tags.get(tag, 0) + sign
        if any(tags.values()):
            return False
    return True


def _events(run_dir):
    """The LHE events of the last run."""
    events = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'events.lhe.gz')))
    assert events, run_dir
    return events[-1]


def _resonances(lhe_path):
    """The (pdg, mass) of the status-2 particles of every event."""
    with gzip.open(lhe_path, 'rt') as stream:
        text = stream.read()
    return [[(int(particle[0]), float(particle[10]))
             for particle in (line.split() for line in event.strip().splitlines()[1:])
             if len(particle) > 10 and particle[1] == '2']
            for event in re.findall(r'<event>\n(.*?)</event>', text, re.S)]


class MG7AmplicolTest(unittest.TestCase):

    def setUp(self):
        from madgraph.mg7 import bootstrap
        # not imported: Symbolica would hold its lock against the pyamplicol worker
        missing = [name for name, installed in (
            ('madspace', bootstrap.madspace_is_installed()),
            ('pyamplicol', bootstrap.pyamplicol_is_installed())) if not installed]
        if missing:
            self.skipTest('%s not installed' % ' and '.join(missing))
        if not misc.resolve_lhapdf().find_set(_PDF):
            self.skipTest('%s LHAPDF data not found' % _PDF)
        self.path = tempfile.mkdtemp(prefix='mg7_amplicol_')

    def tearDown(self):
        shutil.rmtree(self.path, ignore_errors=True)

    def run_mg7(self, name, processes, gridpack=False, **amplicol):
        """Output and run the processes, returns the cross section, its error
        and the run directory."""
        run_dir = pjoin(self.path, name)
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        for index, process in enumerate(processes):
            mg.exec_cmd('%s %s' % ('generate' if index == 0 else 'add process',
                                   process))
        mg.exec_cmd('output mg7 %s' % run_dir)
        _edit_run_card(pjoin(run_dir, 'Cards', 'run_card.toml'), gridpack, **amplicol)
        log = pjoin(run_dir, 'generate_events.log')
        with open(log, 'w') as stream:
            status = subprocess.call(
                [sys.executable, pjoin(run_dir, 'bin', 'generate_events'), '-f'],
                cwd=run_dir, stdout=stream, stderr=subprocess.STDOUT)
        self.assertEqual(status, 0, open(log).read()[-3000:])
        info = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'info.json')))[-1]
        with open(info) as stream:
            result = json.load(stream)['process']
        # a failed generation can still exit with 0, after the survey
        self.assertTrue(result['done'], open(log).read()[-3000:])
        return float(result['mean']), float(result['error']), run_dir

    def assert_compatible(self, first, second, factor=1.):
        (mean1, error1), (mean2, error2) = first, second
        error = (error1**2 + (factor * error2)**2)**0.5
        self.assertLess(abs(mean1 - factor * mean2), _NSIGMA * error,
                        '%g +- %g vs %g +- %g' % (mean1, error1, factor * mean2,
                                                  factor * error2))

    def test_gg_ggg(self):
        """Full colour in both modes, and leading colour, which differs by
        (N^2 - 1) / N^2 for five gluons."""
        madmatrix = self.run_mg7('mm', ['g g > g g g'])[:2]
        reweight = self.run_mg7('rw', ['g g > g g g [amplicol]'])
        self.assert_compatible(reweight[:2], madmatrix)
        events = glob.glob(pjoin(reweight[2], 'Events', '*', 'events.lhe.gz'))
        if events:
            self.assertTrue(_color_balanced(events[-1]))
        unweight = self.run_mg7('uw', ['g g > g g g [amplicol]'], mode='unweight')
        self.assert_compatible(unweight[:2], madmatrix)
        leading = self.run_mg7('lc', ['g g > g g g [amplicol]'],
                               color_accuracy='lc')
        self.assert_compatible(madmatrix, leading[:2], 8. / 9.)

    def test_pp_jjj(self):
        """Quark lines and the expansion of the multiparticle labels."""
        madmatrix = self.run_mg7('mm', ['p p > j j j'])[:2]
        amplicol = self.run_mg7('amp', ['p p > j j j [amplicol]'])[:2]
        self.assert_compatible(amplicol, madmatrix)

    def test_gg_ttxg(self):
        """Massive quarks in the colour-ordered phase space."""
        madmatrix = self.run_mg7('mm', ['g g > t t~ g'])[:2]
        amplicol = self.run_mg7('amp', ['g g > t t~ g [amplicol]'])[:2]
        self.assert_compatible(amplicol, madmatrix)

    def assert_resonances(self, lhe_path, pdg, mass, width):
        """Some events write the resonance, within bw_cutoff widths of its
        mass, and no other one."""
        self.assertTrue(_color_balanced(lhe_path))
        resonances = [r for event in _resonances(lhe_path) for r in event]
        self.assertGreater(len(resonances), 0.2 * _EVENTS)
        self.assertEqual({r[0] for r in resonances}, {pdg})
        self.assertLess(max(abs(m - mass) for _, m in resonances), 15 * width)

    def test_pp_eej(self):
        """Colour singlets attached to the quark line in either order and a Z
        resonance: three phase-space channels per colour order, weighted by
        their mapping densities, and the Z in the event record."""
        madmatrix = self.run_mg7('mm', ['p p > e+ e- j'])[:2]
        amplicol = self.run_mg7('amp', ['p p > e+ e- j [amplicol]'],
                                output_format='lhe')
        self.assert_compatible(amplicol[:2], madmatrix)
        self.assert_resonances(_events(amplicol[2]), 23, 91.188, 2.441404)

    def test_pp_wpj(self):
        """A charged resonance: W+, never W-, from the charge of its decay."""
        madmatrix = self.run_mg7('mm', ['p p > e+ ve j'])[:2]
        amplicol = self.run_mg7('amp', ['p p > e+ ve j [amplicol]'],
                                output_format='lhe')
        self.assert_compatible(amplicol[:2], madmatrix)
        self.assert_resonances(_events(amplicol[2]), 24, 80.419, 2.0476)

    def test_mixed(self):
        """[amplicol] and madmatrix processes in one output."""
        jj = self.run_mg7('jj', ['p p > j j'])[:2]
        ggg = self.run_mg7('ggg', ['g g > g g g'])[:2]
        mixed = self.run_mg7('mixed', ['p p > j j', 'g g > g g g [amplicol]'])[:2]
        self.assert_compatible(
            mixed, (jj[0] + ggg[0], (jj[1]**2 + ggg[1]**2)**0.5))

    def test_gridpack(self):
        """A gridpack of [amplicol] and madmatrix processes, with the second
        pass unweighting to full colour, reproduces the cross section of its
        run. The [amplicol] process has several phase-space channels per colour
        order and writes its Z resonance."""
        mean, error, run_dir = self.run_mg7(
            'grid', ['p p > e+ e- j [amplicol]', 'p p > mu+ mu- j'], gridpack=True,
            mode='unweight')
        gridpack = sorted(glob.glob(pjoin(run_dir, 'Events', '*', 'gridpack')))[-1]
        self.assertTrue(glob.glob(pjoin(gridpack, 'SubProcesses', 'A*', 'lc')))
        log = pjoin(self.path, 'gridpack.log')
        with open(log, 'w') as stream:
            status = subprocess.call(
                [sys.executable, pjoin(gridpack, 'bin', 'generate_events'),
                 '--seed', '11', '--output_dir', pjoin(self.path, 'grid_events')],
                cwd=self.path, stdout=stream, stderr=subprocess.STDOUT)
        self.assertEqual(status, 0, open(log).read()[-3000:])
        with open(pjoin(self.path, 'grid_events', 'info.json')) as stream:
            result = json.load(stream)['process']
        self.assertTrue(result['done'], open(log).read()[-3000:])
        self.assert_compatible((float(result['mean']), float(result['error'])),
                               (mean, error))
        self.assert_resonances(pjoin(self.path, 'grid_events', 'events.lhe.gz'),
                               23, 91.188, 2.441404)
