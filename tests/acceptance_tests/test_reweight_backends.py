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
"""The two reweighting backends of an MG7 output give the same weights, in
the same place.

A small MG7 process (u u~ > t t~, 500 events, LHE output, systematics on)
is generated once; the run is copied, and one copy is reweighted with
MadtRex, the other with the classic module (launch.run_reweighting, as the
launch runs them). The card uses the shared vocabulary: 'change helicity
False' and a parameter given by its param_card name ('set MT 175'). Both
add their weights to the run's own event file (Events/<run>/events.lhe.gz),
after the systematics weights, and agree event by event. A second process
directory with output_format = lhe_npy checks that MadtRex reweights
events.npy in place (generate_events --reweight=CARD) and that npy_to_lhe
carries the new weight.

Skipped unless madspace and madtrex are installed and numpy.f2py runs with
this Python (the classic module compiles its matrix elements with f2py, by
default the one installed with this Python: misc.default_f2py()).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

MG_ROOT = Path(__file__).resolve().parents[2]


def both_backends_available():
    if not (MG_ROOT / 'madspace' / 'install' / 'madspace').is_dir():
        return False
    if not (MG_ROOT / 'madtrex' / 'install' / 'madtrex').is_dir():
        return False
    if shutil.which('gfortran') is None:
        return False
    result = subprocess.run([sys.executable, '-m', 'numpy.f2py', '-v'],
                            capture_output=True, text=True)
    return result.returncode == 0


CARD = "change helicity False\nlaunch --rwgt_name=mt175\nset MT 175\n"

# reweight run <run> of the process directory (the cwd) with <backend>
REWEIGHT = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, {root!r})
    from madgraph.iolibs.template_files.mg7 import bootstrap
    bootstrap.ensure_madspace(interactive=False)
    from madgraph.iolibs.template_files.mg7 import launch
    class Process:
        run_path = os.path.join("Events", {run!r})
        events_path = os.path.join(run_path, "events.lhe.gz")
        run_card = {{"run": {{"output_format": "lhe"}}}}
    launch.run_reweighting(Process(), launch.REWEIGHT_CARD, requested={backend!r})
""")


@unittest.skipUnless(both_backends_available(),
                     'needs madspace, madtrex, gfortran and numpy.f2py')
class ReweightBackendsAgreeTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_checked(self, args, **kwargs):
        result = subprocess.run(args, capture_output=True, text=True, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result

    def make_process(self, name, output_format):
        proc = os.path.join(self.tmp, name)
        script = os.path.join(self.tmp, name + '.mg5')
        with open(script, 'w') as f:
            f.write('import model sm\ngenerate u u~ > t t~\noutput mg7 %s -f\n' % proc)
        self.run_checked([sys.executable, str(MG_ROOT / 'bin' / 'madgraph'), script])
        run_card = os.path.join(proc, 'Cards', 'run_card.toml')
        with open(run_card) as f:
            text = f.read()
        text = text.replace('events = 100000', 'events = 500', 1)
        text = text.replace('output_format = "lhe_npy"', 'output_format = "%s"' % output_format, 1)
        with open(run_card, 'w') as f:
            f.write(text)
        with open(os.path.join(proc, 'Cards', 'reweight_card.dat'), 'w') as f:
            f.write(CARD)
        return proc

    def weights(self, path):
        sys.path.insert(0, str(MG_ROOT))
        import madgraph.various.lhe_parser as lhe_parser
        return [(event.wgt, event.parse_reweight())
                for event in lhe_parser.EventFile(path)]

    def test_classic_and_madtrex_write_the_same_file(self):
        proc = self.make_process('PROC', 'lhe')
        self.run_checked([sys.executable, 'bin/generate_events', '-f'], cwd=proc)
        events = os.path.join(proc, 'Events')
        shutil.copytree(os.path.join(events, 'run_01'), os.path.join(events, 'run_02'))

        for run, backend in (('run_01', 'madtrex'), ('run_02', 'classic')):
            self.run_checked([sys.executable, '-c', REWEIGHT.format(
                root=str(MG_ROOT), run=run, backend=backend)], cwd=proc)
            with open(os.path.join(events, run, 'reweighting.json')) as f:
                record = json.load(f)
            self.assertEqual(record['backend'], backend)
            self.assertEqual(record['events'], os.path.join('Events', run, 'events.lhe.gz'))
            # no separate output file, no stale decompressed copy
            self.assertNotIn('events_reweighted.lhe', os.listdir(os.path.join(events, run)))
            self.assertNotIn('events.lhe', os.listdir(os.path.join(events, run)))

        madtrex = self.weights(os.path.join(events, 'run_01', 'events.lhe.gz'))
        classic = self.weights(os.path.join(events, 'run_02', 'events.lhe.gz'))
        self.assertEqual(len(madtrex), 500)
        self.assertEqual(len(classic), 500)
        for (orig_m, rw_m), (orig_c, rw_c) in zip(madtrex, classic):
            self.assertAlmostEqual(orig_m, orig_c, delta=1e-6 * abs(orig_c))
            # the systematics weights are still there, the new one follows them
            self.assertEqual(list(rw_m)[:109], [str(i) for i in range(1, 110)])
            self.assertEqual(list(rw_m)[-1], 'mt175')
            self.assertIn('1', rw_c)
            self.assertEqual(rw_m['1'], rw_c['1'])
            # the LHE files carry about 7 significant digits
            self.assertAlmostEqual(rw_m['mt175'], rw_c['mt175'], delta=1e-6 * abs(rw_c['mt175']))
        # MT 173 -> 175 changes the weights
        total = sum(w for w, _ in madtrex)
        self.assertNotAlmostEqual(sum(r['mt175'] for _, r in madtrex), total,
                                  delta=1e-3 * abs(total))
        # the shower & co. read unweighted_events.lhe.gz: the same events
        tool_copy = os.path.join(events, 'run_02', 'unweighted_events.lhe.gz')
        self.assertEqual([r['mt175'] for _, r in self.weights(tool_copy)],
                         [r['mt175'] for _, r in classic])

    def test_madtrex_reweights_npy_events_in_place(self):
        import numpy as np
        proc = self.make_process('NPY', 'lhe_npy')
        card = os.path.join(proc, 'Cards', 'reweight_card.dat')
        result = self.run_checked([sys.executable, 'bin/generate_events', '-f',
                                   '--reweight=%s' % card, '--reweight-backend=madtrex'],
                                  cwd=proc)
        self.assertIn('Reweighting with madtrex (--reweight-backend)', result.stdout)
        run = os.path.join(proc, 'Events', 'run_01')
        events = np.load(os.path.join(run, 'events.npy'), mmap_mode='r')
        names = [n for n in events.dtype.names if n.startswith('rwgt_')]
        self.assertEqual(names, ['rwgt_%d' % i for i in range(1, 110)] + ['rwgt_mt175'])
        # MadtRex's stored generation-time matrix elements are not in the output
        self.assertNotIn('matrix_element', events.dtype.names)
        self.assertFalse(any(n.startswith('.madtrex') for n in os.listdir(run)))
        lhe = self.run_checked([sys.executable, '-c', textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(MG_ROOT)!r})
            from madgraph.iolibs.template_files.mg7 import npy_to_lhe
            print(npy_to_lhe.convert({run!r}, compress=False))
        """)], cwd=proc).stdout.split()[-1]
        converted = self.weights(lhe)
        self.assertEqual(len(converted), 500)
        for (_, weights), value in zip(converted, events['rwgt_mt175']):
            self.assertEqual(list(weights)[-1], 'mt175')
            self.assertAlmostEqual(weights['mt175'], value, delta=1e-6 * abs(value))

    def test_madtrex_lhe_run_is_reweighted_as_npy(self):
        """An LHE run MadtRex reweights is generated as npy, reweighted, and
        written as the LHE file it asked for -- also without the weights,
        when the reweighting fails."""
        proc = self.make_process('LHE', 'lhe')
        card = os.path.join(proc, 'Cards', 'reweight_card.dat')
        result = self.run_checked([sys.executable, 'bin/generate_events', '-f',
                                   '--reweight=%s' % card, '--reweight-backend=madtrex'],
                                  cwd=proc)
        self.assertIn('the events are written as npy', result.stdout)
        run = os.path.join(proc, 'Events', 'run_01')
        self.assertIn('events.lhe.gz', os.listdir(run))
        self.assertNotIn('events.npy', os.listdir(run))
        self.assertNotIn('header.lhe', os.listdir(run))
        with open(os.path.join(run, 'reweighting.json')) as f:
            self.assertEqual(json.load(f)['events'], os.path.join(run, 'events.lhe.gz'))
        events = self.weights(os.path.join(run, 'events.lhe.gz'))
        self.assertEqual(len(events), 500)
        for _, weights in events:
            self.assertEqual(list(weights), [str(i) for i in range(1, 110)] + ['mt175'])
        total = sum(w for w, _ in events)
        self.assertNotAlmostEqual(sum(r['mt175'] for _, r in events), total,
                                  delta=1e-3 * abs(total))

        bad = os.path.join(self.tmp, 'bad_card.dat')
        with open(bad, 'w') as f:
            f.write("launch --rwgt_name=bad\nset MASS 99999 175\n")
        result = self.run_checked([sys.executable, 'bin/generate_events', '-f',
                                   '--reweight=%s' % bad, '--reweight-backend=madtrex'],
                                  cwd=proc)
        self.assertIn('reweighting failed', result.stdout + result.stderr)
        run = os.path.join(proc, 'Events', 'run_02')
        self.assertNotIn('events.npy', os.listdir(run))
        events = self.weights(os.path.join(run, 'events.lhe.gz'))
        self.assertEqual(len(events), 500)
        self.assertEqual(list(events[0][1]), [str(i) for i in range(1, 110)])


    def test_madtrex_change_model_divides_by_the_original_matrix_element(self):
        """A "change model" launch is normalised by the matrix element the
        events were generated with: to the same model, it reproduces the
        event weights (no parameter change) and the original model's own
        reweighting (MT 175). The generation stores its matrix elements for
        the first, helicity-summed reweighting; the later passes, on the
        events without them, evaluate the original matrix element (summed
        and at the event's helicity) and agree."""
        import numpy as np
        proc = self.make_process('CM', 'lhe_npy')
        change = 'change model sm\nchange process u u~ > t t~\n'
        card = os.path.join(proc, 'Cards', 'reweight_card.dat')
        with open(card, 'w') as f:
            f.write('change helicity False\nlaunch --rwgt_name=mt175\nset MT 175\n' + change +
                    'launch --rwgt_name=same\nlaunch --rwgt_name=cm175\nset MT 175\n')
        self.run_checked([sys.executable, 'bin/generate_events', '-f',
                          '--reweight=%s' % card, '--reweight-backend=madtrex'], cwd=proc)
        events_path = os.path.join(proc, 'Events', 'run_01', 'events.npy')
        for helicity, suffix in (('False', 'eval'), ('True', 'hel')):
            again = os.path.join(self.tmp, 'again_%s.dat' % suffix)
            with open(again, 'w') as f:
                f.write('change helicity %s\nlaunch --rwgt_name=mt175%s\nset MT 175\n%s'
                        'launch --rwgt_name=same%s\nlaunch --rwgt_name=cm175%s\nset MT 175\n'
                        % (helicity, suffix, change, suffix, suffix))
            self.run_checked([sys.executable, '-c', textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(MG_ROOT)!r})
                from madgraph.iolibs.template_files.mg7 import bootstrap
                bootstrap.ensure_madspace(interactive=False)
                from madgraph.iolibs.template_files.mg7 import madtrex
                madtrex.run_parameter_reweighting({again!r}, events_path={events_path!r},
                                                  process_directory={proc!r})
            """)], cwd=proc)
        events = np.load(events_path)
        self.assertNotIn('matrix_element', events.dtype.names)
        weight = events['weight']

        def close(a, b):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
        for suffix in ('', 'eval', 'hel'):
            close(events['rwgt_same' + suffix], weight)
            close(events['rwgt_cm175' + suffix], events['rwgt_mt175' + suffix])
        # stored and evaluated normalisation (both helicity-summed) agree
        close(events['rwgt_mt175eval'], events['rwgt_mt175'])
        close(events['rwgt_cm175eval'], events['rwgt_cm175'])
        # and MT 175 does change the weights
        self.assertGreater(np.max(np.abs(events['rwgt_mt175'] / weight - 1)), 1e-3)

    def test_madtrex_normalises_with_the_generation_param_card(self):
        """MadtRex reweights from the param_card in the events' header (the
        one they were generated with), whatever Cards/param_card.dat says
        by then, and warns about the difference; events without one in
        their header are normalised with Cards/param_card.dat, with a
        warning that this cannot be checked."""
        import numpy as np
        import re
        proc = self.make_process('PC', 'lhe_npy')
        self.run_checked([sys.executable, 'bin/generate_events', '-f'], cwd=proc)
        run = os.path.join(proc, 'Events', 'run_01')
        card = os.path.join(self.tmp, 'mt175.dat')
        with open(card, 'w') as f:
            f.write(CARD)
        copies = {}
        for name, header in (('as_generated', True), ('edited', True), ('no_header', False)):
            copy = os.path.join(self.tmp, name)
            os.makedirs(copy)
            shutil.copy(os.path.join(run, 'events.npy'), copy)
            if header:
                shutil.copy(os.path.join(run, 'header.lhe'), copy)
            copies[name] = os.path.join(copy, 'events.npy')

        def reweight(events):
            return self.run_checked([sys.executable, '-c', textwrap.dedent(f"""
                import logging, sys
                sys.path.insert(0, {str(MG_ROOT)!r})
                logging.basicConfig(level=logging.WARNING)
                from madgraph.iolibs.template_files.mg7 import bootstrap
                bootstrap.ensure_madspace(interactive=False)
                from madgraph.iolibs.template_files.mg7 import madtrex
                madtrex.run_parameter_reweighting({card!r}, events_path={events!r},
                                                  process_directory={proc!r})
            """)], cwd=proc).stderr

        log = reweight(copies['as_generated'])
        self.assertNotIn('differs from the param_card', log)
        param_card = os.path.join(proc, 'Cards', 'param_card.dat')
        with open(param_card) as f:
            text = f.read()
        edited, count = re.subn(r'(?m)^(\s*6\s+)\S+(\s*#\s*MT\b)', r'\g<1>1.800000e+02\2', text)
        self.assertEqual(count, 1)
        with open(param_card, 'w') as f:
            f.write(edited)
        log = reweight(copies['edited'])
        self.assertIn('differs from the param_card', log)
        self.assertIn('mass 6: 173.0 -> 180.0', log)
        log = reweight(copies['no_header'])
        self.assertIn('carry no param_card', log)

        weights = {name: np.load(path)['rwgt_mt175'] for name, path in copies.items()}
        np.testing.assert_allclose(weights['edited'], weights['as_generated'], rtol=1e-12, atol=0)
        # without the header, normalised with the edited card: MT 180 -> 175
        self.assertGreater(np.max(np.abs(weights['no_header'] / weights['as_generated'] - 1)), 1e-3)

    def test_madtrex_later_group_is_normalised_at_the_generation_parameters(self):
        """A group after a launch that changed parameters (here an fptype
        group, whose normaliser is the first group's library) is normalised
        at the generation's parameters, also when the sample is split into
        batches that reload the libraries."""
        import numpy as np
        proc = self.make_process('GROUPS', 'lhe_npy')
        self.run_checked([sys.executable, 'bin/generate_events', '-f'], cwd=proc)
        events_path = os.path.join(proc, 'Events', 'run_01', 'events.npy')
        card = os.path.join(self.tmp, 'groups.dat')
        with open(card, 'w') as f:
            f.write("change nb_core 4\nchange min_batch 100\n"
                    "launch --rwgt_name=mt175\nset MT 175\n"
                    "launch --rwgt_name=same_d\nchange fptype d\n")
        self.run_checked([sys.executable, '-c', textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(MG_ROOT)!r})
            from madgraph.iolibs.template_files.mg7 import bootstrap
            bootstrap.ensure_madspace(interactive=False)
            from madgraph.iolibs.template_files.mg7 import madtrex
            madtrex.run_parameter_reweighting({card!r}, events_path={events_path!r},
                                              process_directory={proc!r})
        """)], cwd=proc)
        events = np.load(events_path)
        self.assertGreater(np.max(np.abs(events['rwgt_mt175'] / events['weight'] - 1)), 1e-3)
        np.testing.assert_allclose(events['rwgt_same_d'], events['weight'], rtol=1e-5, atol=0)


    def test_madtrex_reweights_mlm_events_with_their_stored_matrix_elements(self):
        """An MLM run's npy events (with per-particle cluster scales) are
        read by MadtRex, and their stored matrix elements are at the
        event's own alpha_s: a no-op helicity-summed launch gives the event
        weight back."""
        import numpy as np
        proc = os.path.join(self.tmp, 'MLM')
        script = os.path.join(self.tmp, 'MLM.mg5')
        with open(script, 'w') as f:
            f.write('import model sm\ngenerate g g > t t~ g\noutput mg7 %s -f\n' % proc)
        self.run_checked([sys.executable, str(MG_ROOT / 'bin' / 'madgraph'), script])
        run_card = os.path.join(proc, 'Cards', 'run_card.toml')
        with open(run_card) as f:
            text = f.read()
        for old, new in (('events = 100000', 'events = 2000'),
                         ('dynamical_scale_choice = "half_transverse_mass"',
                          'dynamical_scale_choice = "mlm"'),
                         ('xqcut = 0.0', 'xqcut = 20.0'),
                         ('[beam]\n', '[beam]\nclustering_history = "diagram"\n')):
            self.assertIn(old, text)
            text = text.replace(old, new, 1)
        with open(run_card, 'w') as f:
            f.write(text)
        card = os.path.join(self.tmp, 'mlm.dat')
        with open(card, 'w') as f:
            f.write("change helicity False\nlaunch --rwgt_name=same\n")
        result = self.run_checked([sys.executable, 'bin/generate_events', '-f',
                                   '--reweight=%s' % card, '--reweight-backend=madtrex'],
                                  cwd=proc)
        self.assertNotIn('differ from the ones', result.stdout + result.stderr)
        events = np.load(os.path.join(proc, 'Events', 'run_01', 'events.npy'))
        self.assertIn('part1_cluster_scale', events.dtype.names)
        np.testing.assert_allclose(events['rwgt_same'], events['weight'], rtol=1e-5, atol=0)


if __name__ == '__main__':
    unittest.main()
