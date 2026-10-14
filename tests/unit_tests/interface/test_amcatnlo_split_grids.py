"""Combining fixed-order grids must preserve the pooled virtual mode."""

from pathlib import Path
import tempfile
import unittest

from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd, aMCatNLOError


class TestSplitOrderGrids(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="mg5_split_grids_")
        self.addCleanup(self.tempdir.cleanup)
        self.work = Path(self.tempdir.name)
        self.base = self.work / "all_G1"
        self.base.mkdir()
        self.jobs = []
        for split in (1, 2):
            directory = self.work / ("all_G1_%d" % split)
            directory.mkdir()
            self.jobs.append({"dirname": str(directory)})
        self.mint_data = [
            b"AVE 1 3\nSUM 2 4\nMAX 7 5\nQSM 3 4\nIDE 2 3\nSPE 10 3 4\n"
            b"POL 2\nPOL 1.0 2.0\nPOL 3.0 4.0\n",
            b"AVE 3 5\nSUM 6 8\nMAX 2 9\nQSM 4 3\nIDE 2 3\nSPE 20 5 6\n"
            b"POL 2\nPOL 3.0 4.0\nPOL 5.0 6.0\n",
        ]
    def write_split_grids(self, blocks):
        for index, (job, block) in enumerate(zip(self.jobs, blocks)):
            directory = Path(job["dirname"])
            (directory / "mint_grids").write_bytes(self.mint_data[index] + block)
            (directory / "grid.MC_integer").write_bytes(
                b"AVE 2 4\nSUM 1 3\n" if index == 0
                else b"AVE 6 8\nSUM 5 7\n")

    def combine(self):
        # The combiner depends only on its job argument, not command setup.
        aMCatNLOCmd.combine_split_order_grids(None, self.jobs)

    def assert_combined_numbers(self, content):
        self.assertEqual(content, (
            b"AVE 2.0 4.0\nSUM 8.0 12.0\nMAX 7.0 9.0\nQSM 5.0 5.0\n"
            b"IDE 2 3\nSPE 15 8 5\nPOL 3\nPOL 1.0 2.0\nPOL 3.0 4.0\n"
            b"POL 5.0 6.0\n"))
        self.assertEqual((self.base / "grid.MC_integer").read_bytes(),
                         b"AVE 4.0 6.0\nSUM 6.0 10.0\n")

    def test_legacy_files_without_pooled_mode(self):
        self.write_split_grids([b"", b""])
        self.combine()
        self.assert_combined_numbers((self.base / "mint_grids").read_bytes())

    def test_preserves_frozen_pooled_mode_marker(self):
        marker = b'PVP 1\r\n'
        self.write_split_grids([marker, marker])
        self.combine()
        content = (self.base / 'mint_grids').read_bytes()
        self.assertTrue(content.endswith(marker))
        self.assertEqual(content.count(b'PVP'), 1)
        self.assert_combined_numbers(content[:-len(marker)])

    def test_pooled_mode_mismatch_preserves_existing_outputs(self):
        for blocks in ([b'PVP 1\n', b''], [b'', b'PVP 1\n'],
                       [b'PVP 1\n', b'PVP 0\n']):
            with self.subTest(blocks=blocks):
                self.write_split_grids(blocks)
                for name in ('mint_grids', 'grid.MC_integer'):
                    (self.base / name).write_bytes(b'previous master\n')
                with self.assertRaisesRegex(aMCatNLOError, 'Pooled virtual modes differ or are missing'):
                    self.combine()
                for name in ('mint_grids', 'grid.MC_integer'):
                    self.assertEqual((self.base / name).read_bytes(), b'previous master\n')

    def test_combines_virtual_envelope_factors_and_preserves_marker(self):
        # Componentwise maxima cover the proposal from every split replica.
        self.write_split_grids([b'MAX 2 9\nMAX .7\nPVP 1\nVEP 3\n',
                                b'MAX 5 3\nMAX .3\nPVP 1\nVEP 3\n'])
        self.combine()
        content = (self.base / 'mint_grids').read_bytes()
        self.assertIn(b'MAX 5.0 9.0\n', content)
        self.assertIn(b'MAX 0.7\n', content)
        self.assertTrue(content.endswith(b'PVP 1\nVEP 3\n'))

    def test_virtual_envelope_mismatch_preserves_outputs(self):
        for blocks in ([b'PVP 1\nVEP 1\n', b'PVP 1\n'],
                       [b'PVP 1\nVEP 1\n', b'PVP 1\nVEP 2\n']):
            with self.subTest(blocks=blocks):
                self.write_split_grids(blocks)
                for name in ('mint_grids', 'grid.MC_integer'):
                    (self.base / name).write_bytes(b'previous master\n')
                with self.assertRaisesRegex(aMCatNLOError, 'Virtual envelope formats differ'):
                    self.combine()
                for name in ('mint_grids', 'grid.MC_integer'):
                    self.assertEqual((self.base / name).read_bytes(), b'previous master\n')

    def test_combines_both_velocity_and_born_bounds_by_maximum(self):
        self.write_split_grids([b'MAX .7 .1\nPVP 1\nVEP 4\n',
                                b'MAX .3 .4\nPVP 1\nVEP 4\n'])
        self.combine()
        content = (self.base / 'mint_grids').read_bytes()
        self.assertIn(b'MAX 0.7 0.4\n', content)
        self.assertTrue(content.endswith(b'PVP 1\nVEP 4\n'))

    def test_truncated_velocity_bound_cannot_be_silently_merged(self):
        self.write_split_grids([b'MAX .7 .1\nPVP 1\nVEP 4\n',
                                b'MAX .3\nPVP 1\nVEP 4\n'])
        (self.base / 'mint_grids').write_bytes(b'previous master\n')
        with self.assertRaisesRegex(aMCatNLOError, 'different numbers of values'):
            self.combine()
        self.assertEqual((self.base / 'mint_grids').read_bytes(), b'previous master\n')

    def test_invalid_mc_integer_grid_preserves_both_master_files(self):
        self.write_split_grids([b'PVP 1\nVEP 4\n', b'PVP 1\nVEP 4\n'])
        (Path(self.jobs[1]['dirname']) / 'grid.MC_integer').write_bytes(
            b'AVE 6\nSUM 5 7\n')
        for name in ('mint_grids', 'grid.MC_integer'):
            (self.base / name).write_bytes(b'previous master\n')
        with self.assertRaisesRegex(aMCatNLOError, 'different numbers of values'):
            self.combine()
        for name in ('mint_grids', 'grid.MC_integer'):
            self.assertEqual((self.base / name).read_bytes(), b'previous master\n')
