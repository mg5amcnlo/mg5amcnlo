"""Source-Delta versus first-step Sudakov check on synthetic proposals; no shower."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from tests.unit_tests.various.test_macnlops import HEADER
from tests.unit_tests.various.test_macnlops_transfer_diagnostic import HAVE_NUMPY, proposal

ROOT = Path(__file__).resolve().parents[3]
# The check needs NumPy; without it its tests are skipped.
if HAVE_NUMPY:
    SPEC = importlib.util.spec_from_file_location('macnlops_delta_sudakov_check',
        ROOT / 'docs/audits/check_macnlops_delta_sudakov.py')
    CHECK = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(CHECK)
else:
    CHECK = None

# kind, emission/stopping scale (also the lepton-pair pT), source Delta, SCALUP
SAMPLE = [('S', 1., 1., 100.), ('S', 3., 1., 100.), ('S', 30., 1., 100.),
          ('S', 0., 1., 100.), ('S', 0., 1., 2.),
          ('H', 4., .5, -1.), ('H', 4., .25, -1.), ('H', 60., 1., -1.)]


def write_job(directory):
    directory.mkdir(parents=True)
    with open(directory / 'proposals.lhe', 'w') as lhe, \
            open(directory / 'shower.jsonl', 'w') as shower:
        lhe.write(HEADER)
        for sequence, (kind, scale, delta, start) in enumerate(SAMPLE, 1):
            lhe.write(proposal(kind, -2. if kind == 'H' else 2., scale, delta))
            shower.write(json.dumps(dict(
                sequence=sequence, emitted=kind == 'S' and scale > 0, pt_evol=scale,
                shower_scalup=start if kind == 'S' else scale,
                source_dipole_scales=[[1, 2, scale], [2, 1, scale + 1.]] if kind == 'H'
                else [])) + '\n')
        lhe.write('</LesHouchesEvents>\n')
    return directory


@unittest.skipUnless(HAVE_NUMPY, 'NumPy is required by the Delta Sudakov check')
class TestDeltaSudakovCheck(unittest.TestCase):
    def test_paired_source_and_shower_no_emission(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = write_job(Path(temporary) / 'job')
            result = CHECK.check([job], 'dy', (0., 2., 5., 50.))
        self.assertEqual((result['S_events'], result['H_events']), (5, 3))
        for variable in ('evolution', 'system'):
            row = result[variable][1]
            self.assertEqual((row['lo'], row['hi'], row['H_events']), (2., 5., 2))
            inside = row['inside']
            # Undamped weights |w|/Delta = 4, 8 give Delta_src = 4/12.
            self.assertAlmostEqual(inside['undamped_source'], 4. / 12.)
            self.assertAlmostEqual(inside['generated_source'], .375)
            self.assertAlmostEqual(inside['undamped_largest_share'], 8. / 12.)
            # Of the four S events starting above 4 GeV, three emit below it.
            self.assertAlmostEqual(inside['undamped_shower'], .75)
            # All five S events count once the low start is included.
            self.assertAlmostEqual(row['all']['undamped_shower'], .8)
            self.assertEqual(row['delta_one_fraction'], 0.)
            overflow = result[variable][3]
            self.assertEqual((overflow['hi'], overflow['delta_one_fraction']), (None, 1.))
            self.assertIsNone(overflow['inside'])
        self.assertIn('Delta_src (H) versus Delta_PS', CHECK.table(result))

    def test_needs_both_event_kinds(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = write_job(Path(temporary) / 'job')
            rows = (job / 'shower.jsonl').read_text().splitlines(keepends=True)
            (job / 'shower.jsonl').write_text(''.join(rows[:-1]))
            with self.assertRaisesRegex(CHECK.transfer.DiagnosticError, 'out of step'):
                CHECK.check([job], 'dy', (0.,))


if __name__ == '__main__':
    unittest.main()
