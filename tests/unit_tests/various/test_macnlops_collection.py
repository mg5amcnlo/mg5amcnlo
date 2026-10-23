"""The normal collector shuffles matched channels without changing weights."""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER, event, hard_annotation


def blocks(path):
    with mc.open_lhe(path, 'rb') as stream:
        return re.findall(rb'<event(?:\s[^>]*)?>.*?</event>', stream.read(), re.S)


def channel(folder, name, records, input_s, input_h, process=1):
    header = HEADER.replace('sum = event_norm', 'sum = event_norm\n12345 = iseed')
    header = header.replace('1 0 1 1\n</init>', '1 0 1 %d\n</init>' % process)
    header = header.replace('<header>', '<initrwgt><weight id="nominal">original</weight></initrwgt>\n<header>')
    header = header.replace('</header>', '<macnlops normalization="sum" stage="first-emission">'
                            'old channel counts</macnlops>\n</header>')
    packets = [event(kind, weight, label).replace('\n4 1 ', '\n4 %d ' % process)
               for kind, weight, label in records]
    if packets:
        packets[0] = packets[0].replace('</event>', hard_annotation(records[0][0]) + '</event>')
    path = folder / name
    with mc.open_lhe(path, 'wb') as stream:
        stream.write((header + ''.join(packets) + '</LesHouchesEvents>\n').encode())
    counts = dict(input_S=input_s, input_H=input_h,
                  accepted_S=sum(kind == 'S' for kind, weight, label in records),
                  accepted_H=sum(kind == 'H' for kind, weight, label in records))
    return path, dict(counts=counts, seed=100 + len(records),
                      weight_sums=dict(accepted=sum(weight for kind, weight, label in records)))


class TestMacNLOPSCollection(unittest.TestCase):
    def fixture(self, folder):
        return [channel(folder, 'a.lhe.gz', [('H', 2., 1), ('H', -.5, 2), ('S', -1., 3)], 4, 4),
                channel(folder, 'b.lhe', [('S', 3., 4), ('S', 4., 5), ('H', .75, 6)], 3, 2),
                channel(folder, 'empty.lhe.gz', [], 1, 1, process=2)]

    def test_signed_events_are_shuffled_verbatim_with_aggregate_header(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            inputs, reports = map(list, zip(*self.fixture(folder)))
            original = {path: path.read_bytes() for path in inputs}
            expected = [block for path in inputs for block in blocks(path)]
            output = folder / 'matching/events.lhe.gz'
            result = mc.collect_matched_events(inputs, reports, output, seed=7)
            actual = blocks(output)
            with gzip.open(output, 'rb') as stream:
                self.assertEqual(ET.fromstring(stream.read()).tag, 'LesHouchesEvents')
            self.assertEqual(Counter(actual), Counter(expected))
            self.assertNotEqual(actual, expected)
            self.assertEqual({path: path.read_bytes() for path in inputs}, original)
            self.assertEqual(result['counts']['input_S'] + result['counts']['input_H'], 15)
            self.assertEqual(result['sumw'], 8.25)
            self.assertEqual(result['sumw2'], 30.8125)
            with mc.LHEStream(output) as stream:
                header = stream.header
                self.assertEqual(header.count('<macnlops '), 1)
                self.assertNotIn('old channel counts', header)
                self.assertIn('Input events: 15', header)
                self.assertIn('Retained events: 6', header)
                self.assertIn('shuffle_seed="7"', header)
                self.assertIn('7    = iseed', header)
                self.assertEqual(header.count('<initrwgt>'), 1)
                rows = re.search(r'<init>(.*?)</init>', header, re.S)[1].split()
                self.assertEqual(int(rows[9]), 2)
                table = {int(rows[i + 3]): list(map(float, rows[i:i + 3]))
                         for i in range(10, len(rows), 4)}
                self.assertEqual(table[1][0], 8.25)
                self.assertEqual(table[1][2], 4.)
                expected_variance = (8 * 5.25 - .5**2) / 7 + (5 * 25.5625 - 7.75**2) / 4
                self.assertAlmostEqual(table[1][1], math.sqrt(expected_variance))
                self.assertEqual(table[2], [0., 0., 0.])

    def test_single_input_is_shuffled_reproducibly_without_reserializing_events(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, report = channel(folder, 'one.lhe.gz', [('S', i / 7., i) for i in range(1, 12)], 15, 0)
            # Also exercise original CRLF event content; inspection must not
            # turn the copied event blocks into a reserialized text stream.
            with gzip.open(source, 'rb') as stream:
                content = stream.read().replace(b'\n', b'\r\n')
            with gzip.open(source, 'wb') as stream:
                stream.write(content)
            outputs = [folder / ('out%d.lhe.gz' % i) for i in range(3)]
            for output, seed in zip(outputs, (17, 17, 23)):
                mc.collect_matched_events([source], [report], output, seed)
            self.assertEqual(blocks(outputs[0]), blocks(outputs[1]))
            self.assertNotEqual(blocks(outputs[0]), blocks(outputs[2]))
            self.assertEqual(Counter(blocks(outputs[0])), Counter(blocks(source)))

    def test_all_rejected_still_writes_a_valid_empty_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, report = channel(folder, 'empty.lhe.gz', [], 4, 5)
            output = folder / 'events.lhe.gz'
            mc.collect_matched_events([source], [report], output, 8)
            with mc.LHEStream(output) as stream:
                self.assertEqual(list(stream), [])
                self.assertIn('Input events: 9', stream.header)
                self.assertIn('Retained events: 0', stream.header)
                self.assertIn('0.00000000000000000e+00', stream.header)

    def test_incompatible_channels_and_wrong_counts_publish_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            inputs, reports = map(list, zip(*self.fixture(folder)))
            reports[0]['counts']['accepted_H'] = 1
            output = folder / 'events.lhe.gz'
            with self.assertRaisesRegex(mc.MatchingError, 'counts disagree'):
                mc.collect_matched_events(inputs, reports, output, 7)
            self.assertFalse(output.exists())
            reports[0]['counts']['accepted_H'] = 2
            inputs[1].write_text(inputs[1].read_text().replace('6500 6500', '6501 6500'))
            with self.assertRaisesRegex(mc.MatchingError, 'Incompatible LHE'):
                mc.collect_matched_events(inputs, reports, output, 7)
            self.assertFalse(output.exists())

    def test_changed_weights_or_malformed_init_cannot_be_collected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, report = channel(folder, 'one.lhe', [('S', 2., 1)], 3, 0)
            report['weight_sums']['accepted'] = 3.
            output = folder / 'events.lhe.gz'
            with self.assertRaisesRegex(mc.MatchingError, 'weight sums disagree'):
                mc.collect_matched_events([source], [report], output, 7)
            self.assertFalse(output.exists())
            report['weight_sums']['accepted'] = 2.
            source.write_text(source.read_text().replace('1 0 1 1\n</init>', 'bad process row\n</init>'))
            with self.assertRaisesRegex(mc.MatchingError, 'Invalid LHE init'):
                mc.collect_matched_events([source], [report], output, 7)
            self.assertFalse(output.exists())

    def test_failed_collector_and_existing_output_do_not_publish_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            inputs, reports = map(list, zip(*self.fixture(folder)))
            output = folder / 'events.lhe.gz'
            def fail(**kwargs):
                Path(kwargs['output']).write_bytes(b'partial compressed stream')
                raise RuntimeError('compression failed')
            with patch.object(mc.collect_events, 'collect_events', side_effect=fail):
                with self.assertRaisesRegex(mc.MatchingError, 'compression failed'):
                    mc.collect_matched_events(inputs, reports, output, 7)
            self.assertFalse(output.exists())
            self.assertEqual(list(folder.glob('.macnlops_collect_*')), [])
            output.write_bytes(b'existing sample')
            with self.assertRaisesRegex(mc.MatchingError, 'already exists'):
                mc.collect_matched_events(inputs, reports, output, 7)
            self.assertEqual(output.read_bytes(), b'existing sample')

    def test_cli_collects_only_after_every_channel_succeeds(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            pairs = self.fixture(folder)[:2]
            inputs, reports = map(list, zip(*pairs))
            lookup = dict(zip(inputs, reports))
            def run(filename, process, job, driver, args, seed):
                with mc.open_lhe(filename, 'rb') as source, gzip.open(job / 'events.lhe.gz', 'wb') as target:
                    shutil.copyfileobj(source, target)
                return dict(lookup[filename], seed=seed)
            arguments = ['--process-dir', temp, '--pythia8', '/unused', '--seed', '7']
            output = folder / 'matching'
            with patch.object(mc, 'build_shower', return_value='fixture'), patch.object(mc, 'run_file', side_effect=run):
                result = mc.main(arguments + ['--output-dir', str(output), *map(str, inputs)])
            self.assertEqual(result, output / 'events.lhe.gz')
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary), 2)
            self.assertTrue(all(row['collected_output'] == str(result) for row in summary))
            self.assertEqual(json.loads((output / 'collection.json').read_text())['matching_seeds'], [7, 8])
            failed_output = folder / 'failed'
            def fail_second(filename, *args):
                if filename == inputs[1]:
                    raise mc.MatchingError('native matching failed')
                return run(filename, *args)
            with patch.object(mc, 'build_shower', return_value='fixture'), patch.object(mc, 'run_file', side_effect=fail_second):
                with self.assertRaisesRegex(mc.MatchingError, '1 of 2'):
                    mc.main(arguments + ['--output-dir', str(failed_output), *map(str, inputs)])
            self.assertFalse((failed_output / 'events.lhe.gz').exists())

    def test_exported_script_can_import_the_sibling_collector(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for module in (mc, mc.collect_events):
                shutil.copyfile(module.__file__, folder / Path(module.__file__).name)
            result = subprocess.run([sys.executable, str(folder / 'macnlops.py'), '--help'],
                                    cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--output-dir', result.stdout)


if __name__ == '__main__':
    unittest.main()
