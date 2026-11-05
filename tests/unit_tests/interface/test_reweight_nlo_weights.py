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
"""Ingredients of the NLO-accurate reweighting of aMC@NLO samples.

* The soft counter-terms are eikonal sums of colour-linked Borns, i.e. the real
  emission in the soft limit. ReweightInterface.get_soft_limit_event builds
  that configuration: the counter-term's Born kinematics plus the FKS parton
  with a vanishing energy, along its direction in the real-emission kinematics.
* aMC@NLO evaluates its matrix elements with zero width for the external
  particles, while the banner keeps the param_card as given:
  ReweightInterface.get_amcatnlo_param_card restores the card it really used.
* The reweighting runs in several processes that all rewrite the same
  rw_me/Cards/param_card.dat: ReweightInterface.write_card_atomically.
"""

from __future__ import absolute_import
import copy
import math
import os
import shutil
import tempfile
import threading
import unittest

import madgraph
import madgraph.various.lhe_parser as lhe_parser
import madgraph.models.check_param_card as check_param_card
import madgraph.interface.reweight_interface as reweight_interface

pjoin = os.path.join
_input = pjoin(madgraph.MG5DIR, 'tests', 'input_files')


def _soft_counterterms(nevents=50):
    """(nloweight, contribution) for the soft counter-terms of the first
    events of an aMC@NLO p p > t t~ [QCD] sample"""
    out = []
    for i, event in enumerate(lhe_parser.EventFile(pjoin(_input, 'ttbar_nlo.lhe.gz'))):
        if i >= nevents:
            break
        event.parse_reweight()
        nlo = event.parse_nlo_weight()
        for cevent in nlo.cevents:
            if len(cevent) == cevent.wgts[0].nexternal:
                continue
            for w in cevent.wgts:
                if w.type in (4, 8):
                    out.append((nlo, w))
    return out


def _config(nlo, w, n):
    size = w.nexternal
    return [list(p.get_tuple()) for p in nlo.momenta[size * (n - 1): size * n]]


def _boost_to_cm(moms):
    P = [moms[0][i] + moms[1][i] for i in range(4)]
    b = [-P[i + 1] / P[0] for i in range(3)]
    b2 = sum(x * x for x in b)
    g = 1 / math.sqrt(1 - b2)
    out = []
    for p in moms:
        bp = sum(b[i] * p[i + 1] for i in range(3))
        out.append([g * (p[0] + bp)] + [p[i + 1] + (g - 1) / b2 * bp * b[i] + g * b[i] * p[0]
                                        for i in range(3)] if b2 else list(p))
    return out


class TestSoftLimitEvent(unittest.TestCase):
    """the configuration whose real emission rescales a soft counter-term"""

    def setUp(self):
        self.contribs = _soft_counterterms()
        self.assertTrue(self.contribs)

    def test_kinematics(self):
        """momentum conservation, masses, the soft parton's energy and
        direction, and the Born recovered for lam -> 0"""

        get = reweight_interface.ReweightInterface.get_soft_limit_event
        for nlo, w in self.contribs:
            i_fks, j_fks = [k - 1 for k in w.to_merge_pdg]
            soft = get(nlo, w, lam=1e-6)
            self.assertIsNotNone(soft)
            self.assertEqual(soft.pdgs, list(w.pdgs))
            self.assertEqual(len(soft), w.nexternal)
            p = [list(q.get_tuple()) for q in soft]
            shat = (p[0][0] + p[1][0])**2 - (p[0][3] + p[1][3])**2
            for mu in range(4):
                self.assertAlmostEqual(p[0][mu] + p[1][mu] - sum(q[mu] for q in p[2:]), 0,
                                       delta=1e-9 * math.sqrt(shat))
            # beams along z, massless
            for q in p[:2]:
                self.assertAlmostEqual(q[1], 0); self.assertAlmostEqual(q[2], 0)
                self.assertAlmostEqual(abs(q[3]), q[0], delta=1e-9 * q[0])

            # the Born, in its rest frame
            born = _config(nlo, w, w.born_related)
            sign = 1 if j_fks >= 2 else -1
            born[j_fks] = [born[j_fks][mu] + sign * born[i_fks][mu] for mu in range(4)]
            born = [q for k, q in enumerate(born) if k != i_fks]
            born = _boost_to_cm(born)
            sborn = (born[0][0] + born[1][0])**2
            # the soft parton: energy lam*sqrt(shat_Born), massless. Its
            # direction is the one of the real emission (rest frame of the
            # real) for an initial-state j_fks; for a final-state one the FKS
            # angle y_ij is kept, measured from the Born j_fks (the mother)
            k = p[i_fks]
            self.assertAlmostEqual(k[0], 1e-6 * math.sqrt(sborn), delta=1e-12 * math.sqrt(sborn))
            self.assertAlmostEqual(k[0]**2 - sum(x * x for x in k[1:]), 0, delta=1e-12 * sborn)
            real = _boost_to_cm(_config(nlo, w, w.real_related))
            cos = lambda a, b: sum(a[m] * b[m] for m in (1, 2, 3)) / \
                  math.sqrt(sum(x * x for x in a[1:]) * sum(x * x for x in b[1:]))
            if j_fks < 2:
                self.assertAlmostEqual(cos(k, real[i_fks]), 1, delta=1e-9)
            else:
                self.assertAlmostEqual(cos(k, p[j_fks]), cos(real[i_fks], real[j_fks]),
                                       delta=1e-5)
            # the other momenta are the Born up to O(lam)
            others = [q for n, q in enumerate(p) if n != i_fks]
            for q, b in zip(others, born):
                self.assertAlmostEqual(q[0]**2 - sum(x * x for x in q[1:]),
                                       b[0]**2 - sum(x * x for x in b[1:]), delta=1e-6 * sborn)
                for mu in range(4):
                    self.assertAlmostEqual(q[mu], b[mu], delta=1e-5 * math.sqrt(sborn))

    def test_key_follows_the_flavours(self):
        """two contributions sharing their kinematics but not their flavours
        (several IPROC of one event) get distinct soft configurations"""

        rw = reweight_interface.ReweightInterface
        swap = {1: 3, 3: 1, 2: 4, 4: 2}
        nlo, w = [(nlo, w) for nlo, w in self.contribs
                  if any(abs(p) in swap for p in w.pdgs)][0]
        other = copy.copy(w)
        other.pdgs = [swap[abs(p)] * (1 if p > 0 else -1) if abs(p) in swap else p
                      for p in w.pdgs]
        self.assertNotEqual(other.pdgs, list(w.pdgs))
        self.assertNotEqual(rw.soft_limit_key(w), rw.soft_limit_key(other))
        self.assertEqual(rw.soft_limit_key(w), rw.soft_limit_key(copy.copy(w)))
        self.assertEqual(rw.get_soft_limit_event(nlo, other).pdgs, other.pdgs)

    def test_not_built(self):
        """no soft limit for an initial-state or a non-vector FKS parton: the
        counter-term then follows the Born"""

        get = reweight_interface.ReweightInterface.get_soft_limit_event
        nlo, w = self.contribs[0]
        bad = copy.copy(w)
        bad.to_merge_pdg = [1, w.to_merge_pdg[1]]
        self.assertIsNone(get(nlo, bad))
        bad = copy.copy(w)
        bad.pdgs = list(w.pdgs)
        bad.pdgs[w.to_merge_pdg[0] - 1] = 2
        self.assertIsNone(get(nlo, bad))


class TestAmcatnloParamCard(unittest.TestCase):
    """the widths aMC@NLO evaluated its matrix elements with"""

    card = """
BLOCK MASS
    6 1.730000e+02 # mt
   24 8.041900e+01 # mw
BLOCK SMINPUTS
    3 1.180000e-01 # as
DECAY  6 1.508336e+00 # wt
DECAY 23 2.441404e+00 # wz
DECAY 24 2.047600e+00 # ww
"""

    class Particle(dict):
        pass

    class Model(object):
        def get_particle(self, pdg):
            colour = {6: 3, 24: 1, 21: 8}[abs(pdg)]
            return TestAmcatnloParamCard.Particle(color=colour)

    def interface(self, characteristics=None, sample=True):
        rw = reweight_interface.ReweightInterface.__new__(reweight_interface.ReweightInterface)
        rw.exitted = True      # nothing to clean at deletion
        rw.amcatnlo_sample = sample
        rw.flag_density_matrix = False
        rw.mother = type('Mother', (), {'proc_characteristics': characteristics})() \
                                                         if characteristics is not None else None
        rw.model = self.Model()
        # g g > t t~ w+ w- and its real emission: tags as id_to_path stores them
        rw.id_to_path = {((21, 21), (-24, -6, 6, 24)): None,
                         (((21, 21), (-24, -6, 6, 21, 24)), 'V'): None}
        return rw

    def width(self, card, pdg):
        return card['decay'].get((pdg,)).value

    def test_mixed_expansion(self):
        """the default (nlo_mixed_expansion): every external particle; the
        copy keeps all the parameters, the input card is untouched"""
        card = check_param_card.ParamCard(self.card.splitlines())
        out = self.interface().get_amcatnlo_param_card(card)
        self.assertIsNot(out, card)
        self.assertEqual(self.width(out, 6), 0)
        self.assertEqual(self.width(out, 24), 0)
        self.assertEqual(self.width(out, 23), 2.441404)     # not external
        self.assertEqual(self.width(card, 6), 1.508336)
        self.assertEqual(sum(len(out[b]) for b in out), sum(len(card[b]) for b in card))
        self.assertEqual(out['mass'].get((6,)).value, 173.)

    def test_qcd_expansion(self):
        """without nlo_mixed_expansion: the coloured external particles only"""
        card = check_param_card.ParamCard(self.card.splitlines())
        out = self.interface({'nlo_mixed_expansion': False}).get_amcatnlo_param_card(card)
        self.assertEqual(self.width(out, 6), 0)
        self.assertEqual(self.width(out, 24), 2.0476)

    def test_not_amcatnlo(self):
        """a LO sample keeps its card"""
        card = check_param_card.ParamCard(self.card.splitlines())
        self.assertIs(self.interface(sample=False).get_amcatnlo_param_card(card), card)


class TestWriteCardAtomically(unittest.TestCase):
    """a reader never sees the param_card half written"""

    def test_concurrent_reads(self):
        tmpdir = tempfile.mkdtemp(prefix='rwgt_card')
        try:
            path = os.path.join(tmpdir, 'param_card.dat')
            text = TestAmcatnloParamCard.card * 2000      # ~0.5 MB
            write = reweight_interface.ReweightInterface.write_card_atomically
            write(path, text)
            stop = threading.Event()
            def writer():
                while not stop.is_set():
                    write(path, text)
            thread = threading.Thread(target=writer)
            thread.start()
            try:
                for _ in range(300):
                    self.assertEqual(open(path).read(), text)
            finally:
                stop.set()
                thread.join()
            self.assertEqual(os.listdir(tmpdir), ['param_card.dat'])
        finally:
            shutil.rmtree(tmpdir)


class TestVirtualScale(unittest.TestCase):
    """The virtual's smatrixhel takes mu^2 (the reweighter passes Q_ES^2) and
    must hand it unchanged to <prefix>GET_ME, whose UPDATE_AS_PARAM2 sets
    MU_R = sqrt(SCALE2). The dispatcher used to pass DSQRT(SCALES2): the
    virtual then ran at sqrt(mu) and its parameter dependence (here the EFT
    counter-term logs) was wrong."""

    class FakeModel(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    def test_dispatcher_passes_the_squared_scale(self):
        import madgraph.loop.loop_exporters as loop_exporters
        tmpdir = tempfile.mkdtemp(prefix='rwgt_virt')
        try:
            os.mkdir(os.path.join(tmpdir, 'SubProcesses'))
            exporter = loop_exporters.LoopProcessExporterFortranSA.__new__(
                                    loop_exporters.LoopProcessExporterFortranSA)
            exporter.dir_path = tmpdir
            exporter.opt = {'output_options': {}}
            exporter.density_prefixes = set()
            exporter.model = self.FakeModel({'parameters': {('external',): []},
                                             'running_elements': [],
                                             'merged_particles': {}})
            exporter.prefix_info = {((2, -2, 6, -6), 1): ['ML5_0_', 'tag', 16, 4]}
            exporter.write_f2py_splitter()
            text = open(os.path.join(tmpdir, 'SubProcesses', 'all_matrix.f')).read()
            self.assertIn('CALL ML5_0_GET_ME(P, ALPHAS, SCALES2, NHEL', text)
            self.assertNotIn('SQRT(SCALES2)', text.upper())
            get_me = open(os.path.join(exporter.template_dir, os.pardir,
                                       'loop_optimized', 'check_py.f.inc')).read()
            self.assertIn('UPDATE_AS_PARAM2(SCALE2, ALPHAS)', get_me)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main()
