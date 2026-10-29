"""S-veto/H-removal balance diagnostic on synthetic frozen decisions; no shower."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, event, hard_annotation
from tests.unit_tests.various.test_macnlops_component_protocol import packet

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('macnlops_transfer_diagnostic',
    ROOT / 'docs/audits/diagnose_macnlops_transfer.py')
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)

# kind, weight, electron pT, source factor, emitted, acceptance, accepted
SAMPLE = [('S', 2., 1., 1., True, .75, True),
          ('S', -2., 1., 1., True, .9, False),
          ('S', 2., 1., 1., False, 1., True),
          ('H', -2., 1., .25, False, .6, False),
          ('H', 2., 1., .25, False, 1., True),
          ('S', 2., 30., 1., True, .5, False),
          ('H', -2., 30., .5, False, .5, True)]


def proposal(kind, weight, pt, factor=1., partition='sampled'):
    """A Delta-source DY proposal whose electron carries transverse momentum pt."""
    rows = hard_annotation(kind, partition=partition).splitlines()
    rows[0] = rows[0].replace('source="ordinary"', 'source="delta"')
    block = event(kind, weight=weight).replace(
        '\n11 1 1 2 0 0 100 0 0', '\n11 1 1 2 0 0 100 %.17g 0' % pt)
    if partition == 'sampled':
        metadata = rows[1].split()
        metadata[7:10] = ['1', '0', '1']
        rows[1] = ' '.join(metadata)
        rows.append(packet(source_factor=factor,
                           insertion=0. if kind == 'S' else .3).rstrip())
    return block.replace('</event>', '\n'.join(rows) + '\n</event>')


def write_job(directory, sample=SAMPLE, partition='sampled'):
    directory.mkdir(parents=True)
    with open(directory / 'proposals.lhe', 'w') as lhe, \
            open(directory / 'shower.jsonl', 'w') as shower, \
            open(directory / 'events.lhe.gz.decisions.jsonl', 'w') as decisions:
        lhe.write(HEADER)
        for sequence, (kind, weight, pt, factor, emitted, acceptance, accepted) in \
                enumerate(sample, 1):
            block = proposal(kind, weight, pt, factor, partition)
            lhe.write(block)
            shower.write(json.dumps(dict(sequence=sequence, event_kind=kind,
                                         emitted=emitted, source_weight=weight)) + '\n')
            evaluated = emitted if kind == 'S' else weight < 0
            decisions.write(json.dumps(dict(
                sequence=sequence, kind=kind, accepted=accepted, acceptance=acceptance,
                densities=dict(real=1., subtraction=2., shower=4.) if evaluated
                else None)) + '\n')
        lhe.write('</LesHouchesEvents>\n')
    (directory / 'report.json').write_text(json.dumps(
        dict(veto_prescription='component-undamped-delta-v1')))
    return directory


class TestTransferDiagnostic(unittest.TestCase):
    def test_expected_removals_and_their_ratios(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = write_job(Path(temporary) / 'job')
            result = DIAGNOSTIC.diagnose([job], 'dy', (0., 2., 20., 35.), born_xsec=2.)
        soft, hard = result['bins'][0], result['bins'][2]
        self.assertAlmostEqual(soft['X'], 2 * .25 - 2 * .1)
        self.assertAlmostEqual(soft['X_negative_S'], -.2)
        self.assertEqual(soft['X_realized'], -2.)
        self.assertEqual((soft['S_eligible_events'], soft['S_eligible']), (2., 0.))
        self.assertAlmostEqual(soft['Y'], .8)
        self.assertAlmostEqual(soft['Z'], 3.2)
        self.assertAlmostEqual(soft['U'], 8.)
        ratios = soft['ratios']
        self.assertAlmostEqual(ratios['S_removed_per_H_added'][0], .375)
        self.assertAlmostEqual(ratios['S_removed_per_H_added'][1],
                               math.sqrt(.29 + .375**2 * .64) / .8)
        self.assertAlmostEqual(ratios['Eg_over_K'][0], .3 / 3.2)
        # One negative H fixes Y/Z=f exactly.
        self.assertAlmostEqual(ratios['source_factor'][0], .25)
        self.assertAlmostEqual(ratios['source_factor'][1], 0.)
        self.assertAlmostEqual(ratios['imbalance_after_beta'][0], .3 / (2 * .8))
        self.assertAlmostEqual(ratios['largest_Z_share'], 1.)
        self.assertAlmostEqual(result['total']['ratios']['largest_Z_share'], 3.2 / 5.2)
        self.assertEqual(result['bins'][1]['X'], 0.)
        self.assertIsNone(result['bins'][1]['ratios']['S_removed_per_H_added'])
        self.assertEqual((hard['X'], hard['Y'], hard['Z']), (1., 1., 2.))
        self.assertEqual(result['bins'][3]['hi'], None)
        self.assertEqual(result['sigma'], dict(input=2., S=4., H=-2.))
        self.assertEqual(result['beta'], 2.)
        self.assertAlmostEqual(result['total']['ratios']['source_factor'][0], 1.8 / 5.2)
        self.assertEqual(result['veto_prescriptions'], ['component-undamped-delta-v1'])

    def test_collected_output_sums_relocated_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'macnlops'
            for name in ('first', 'second'):
                write_job(base / name)
            # Recorded outputs are absolute paths of the original location.
            (base / 'summary.json').write_text(json.dumps(
                [dict(input='a.lhe', status='complete', output='/moved/first/events.lhe.gz'),
                 dict(input='b.lhe', status='complete', output='/moved/second/events.lhe.gz'),
                 dict(stage='collection', status='complete')]))
            result = DIAGNOSTIC.diagnose([base], 'dy', DIAGNOSTIC.DEFAULT_EDGES)
            self.assertEqual(len(result['jobs']), 2)
            self.assertAlmostEqual(result['total']['X'], 2 * 1.3)
            self.assertAlmostEqual(result['total']['Y'], 2 * 1.8)
            (base / 'summary.json').write_text(json.dumps(
                [dict(input='a.lhe', status='failed', error='native density')]))
            with self.assertRaisesRegex(DIAGNOSTIC.DiagnosticError, 'Incomplete'):
                DIAGNOSTIC.diagnose([base], 'dy', DIAGNOSTIC.DEFAULT_EDGES)

    def test_global_sum_has_no_per_event_source_factor(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = write_job(Path(temporary) / 'job', partition='global_sum')
            result = DIAGNOSTIC.diagnose([job], 'dy', (0., 2.))
        ratios = result['total']['ratios']
        self.assertAlmostEqual(ratios['S_removed_per_H_added'][0], 1.3 / 1.8)
        self.assertNotIn('Eg_over_K', ratios)
        self.assertNotIn('source_factor', ratios)
        self.assertEqual(result['total']['H_negative_factored'], 0.)

    def test_records_must_stay_in_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = write_job(Path(temporary) / 'job')
            decisions = job / 'events.lhe.gz.decisions.jsonl'
            rows = decisions.read_text().splitlines(keepends=True)
            decisions.write_text(''.join(rows[:-1]))
            with self.assertRaisesRegex(DIAGNOSTIC.DiagnosticError, 'out of step'):
                DIAGNOSTIC.diagnose([job], 'dy', (0.,))
            decisions.write_text(''.join(rows) + rows[-1])
            with self.assertRaisesRegex(DIAGNOSTIC.DiagnosticError, 'Extra'):
                DIAGNOSTIC.diagnose([job], 'dy', (0.,))
            first = json.loads(rows[0])
            decisions.write_text(json.dumps(dict(first, kind='H')) + '\n' + ''.join(rows[1:]))
            with self.assertRaisesRegex(DIAGNOSTIC.DiagnosticError, 'kind'):
                DIAGNOSTIC.diagnose([job], 'dy', (0.,))
            with self.assertRaisesRegex(DIAGNOSTIC.DiagnosticError, 'edges'):
                DIAGNOSTIC.diagnose([job], 'dy', (5., 2.))

    def test_jet_systems(self):
        def parton(pdg, pt, y, phi=0.):
            return (pdg, 1, (pt * math.cosh(y), pt * math.cos(phi),
                             pt * math.sin(phi), pt * math.sinh(y)))
        w = (24, 1, (100., 0., -40., 0.))
        near, far = parton(21, 30., .1), parton(1, 10., .3)
        self.assertEqual(len(DIAGNOSTIC.antikt([near[2][1:] + near[2][:1],
                                                 far[2][1:] + far[2][:1]], .5)), 1)
        far = parton(1, 10., 2.)
        jets = DIAGNOSTIC.antikt([p[2][1:] + p[2][:1] for p in (far, near)], .5)
        self.assertEqual([round(DIAGNOSTIC.prior.pt(jet), 9) for jet in jets], [30., 10.])
        wjet = SimpleNamespace(weight=1., kind='H', particles=[w, near, far])
        # W (0,-40) plus the only fiducial jet (30,0).
        self.assertAlmostEqual(DIAGNOSTIC.system_pt(wjet, 'wjet'), 50.)
        # Neither parton passes the 25 GeV fiducial jet requirement here.
        wjet.particles = [w, parton(21, 20., .1), far]
        self.assertIsNone(DIAGNOSTIC.system_pt(wjet, 'wjet'))
        top = SimpleNamespace(weight=1., kind='S', particles=[
            (6, 1, (200., 30., 0., 0.)), parton(2, 30., .1, math.pi)])
        self.assertAlmostEqual(DIAGNOSTIC.system_pt(top, 'single_top'), 0.)

    def test_command_line_table_json_and_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = write_job(Path(temporary) / 'job')
            output, figure = Path(temporary) / 'balance.json', Path(temporary) / 'balance.pdf'
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(DIAGNOSTIC.main([str(job), '--process', 'dy',
                    '--edges', '0,2,20,35', '--born-xsec', '2', '--json', str(output),
                    '--plot', str(figure)]), 0)
            self.assertIn('beta = sigma_S/sigma_B = 2', stream.getvalue())
            self.assertIn('0.375+-', stream.getvalue())
            self.assertIn('>35', stream.getvalue())
            saved = json.loads(output.read_text())
            self.assertEqual(saved['edges'], [0., 2., 20., 35.])
            self.assertGreater(figure.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
