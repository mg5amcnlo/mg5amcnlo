"""Splitting must preserve LHE records, including precision and extra tags."""

import gzip
from pathlib import Path
import tempfile
import unittest

from madgraph.various.lhe_parser import EventFile


class TestLHESplit(unittest.TestCase):
    banner = ('<LesHouchesEvents version="3.0">\n<header>original header</header>\n'
              '<init>\n 2212 2212 6500 6500 0 0 0 0 -4 1\n'
              ' 1 0 1 1\n</init>\n')
    footer = '</LesHouchesEvents>\n'

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.events = [
            ('  <event id="%d">\n'
             ' 1 1 -1.234567890123456789e-03 100 0.007 0.118\n'
             ' 21 1 0 0 501 502 0 0 0 100 0 0 9\n'
             '# a comment containing </event> and a Unicode label: μ\n'
             '<rwgt><wgt id="1001">-2.12345678901234567e-3</wgt></rwgt>\n'
             '<scales pt_clust_1="12.345678901234567"/>\n'
             '  </event>\n') % i for i in range(5)]

    def source(self, events=None, compressed=False):
        path = self.root / ('events.lhe.gz' if compressed else 'events.lhe')
        text = self.banner + ''.join(self.events if events is None else events) + self.footer
        if compressed:
            with gzip.open(path, 'wt') as stream:
                stream.write(text)
        else:
            path.write_text(text)
        return path

    def split(self, path, **kwargs):
        stream = EventFile(str(path))
        try:
            return stream.split(**kwargs)
        finally:
            stream.close()

    def assert_part(self, source, index, events, compressed=False):
        path = Path('%s_%d.lhe%s' % (source, index, '.gz' if compressed else ''))
        if compressed:
            with gzip.open(path, 'rt') as stream:
                text = stream.read()
        else:
            text = path.read_text()
        self.assertEqual(text, self.banner + ''.join(events) + self.footer)

    def test_equal_sizes_and_remainder_keep_exact_event_text(self):
        path = self.source()
        self.assertEqual(self.split(path, nb_event=2.0), 3)
        for index, events in enumerate((self.events[:2], self.events[2:4], self.events[4:])):
            self.assert_part(path, index, events)

    def test_gzip_input_and_output_with_partition(self):
        path = self.source(compressed=True)
        self.assertEqual(self.split(path, partition=[1, 3, 1], zip=True), 3)
        for index, events in enumerate((self.events[:1], self.events[1:4], self.events[4:])):
            self.assert_part(path, index, events, compressed=True)

    def test_partition_can_select_a_prefix_without_an_extra_file(self):
        path = self.source()
        self.assertEqual(self.split(path, partition=[2, 1]), 2)
        self.assert_part(path, 0, self.events[:2])
        self.assert_part(path, 1, self.events[2:3])
        self.assertFalse(Path('%s_2.lhe' % path).exists())

    def test_empty_and_single_event_files_are_closed_correctly(self):
        path = self.source([])
        self.assertEqual(self.split(path, nb_event=1), 0)
        path = self.source(self.events[:1])
        self.assertEqual(self.split(path, nb_event=1), 1)
        self.assert_part(path, 0, self.events[:1])

    def test_tags_and_utf8_across_buffer_boundaries(self):
        for compressed in (False, True):
            path = self.source(compressed=compressed)
            for size in (1, 7, 31, 1024):
                with self.subTest(compressed=compressed, chunk_size=size):
                    stream = EventFile(str(path))
                    try:
                        self.assertEqual(list(stream._iter_event_text(size)), self.events)
                    finally:
                        stream.close()

    def test_event_larger_than_buffer_and_final_tag_without_newline(self):
        event = self.events[0].replace('# a comment', '#' + 'x' * (2**20) + '\n# a comment')
        path = self.source([event])
        path.write_text((self.banner + event).rstrip('\n'))
        self.assertEqual(self.split(path, nb_event=10), 1)
        self.assert_part(path, 0, [event])

    def test_incomplete_event_is_reported(self):
        path = self.source()
        path.write_text(self.banner + self.events[0] + '<event>\n1 1 1 1 1 1\n')
        with self.assertRaisesRegex(ValueError, 'Incomplete LHE event'):
            self.split(path, nb_event=10)

    def test_invalid_sizes_do_not_create_outputs(self):
        path = self.source()
        for kwargs in ({'nb_event': 0}, {'nb_event': 1.5}, {'partition': [2, 0]}):
            with self.assertRaises(ValueError):
                self.split(path, **kwargs)
        self.assertEqual(self.split(path, partition=[]), 0)
        self.assertEqual(list(self.root.glob('*_*.lhe')), [])
