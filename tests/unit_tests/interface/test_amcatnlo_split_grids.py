"""Combine fixed-order fallback grids while retaining one native virtual fit."""

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

    def test_legacy_files_without_surrogate(self):
        self.write_split_grids([b"", b""])
        self.combine()
        self.assert_combined_numbers((self.base / "mint_grids").read_bytes())

    def test_keeps_one_surrogate_after_the_combined_polynomial_fallback(self):
        first = b'VSU 1 1 4 .25 -2 3 4\nVSU 0 1 4 5 6 7 8\n'
        second = b'VSU 0 1 4 20 30 40 50\nVSU 1 1 4 -1 -2 -3 -4\n'
        self.write_split_grids([first, second])
        self.combine()
        content = (self.base / 'mint_grids').read_bytes()
        self.assertTrue(content.endswith(first))
        self.assert_combined_numbers(content[:-len(first)])

    def test_invalid_mc_integer_grid_preserves_both_master_files(self):
        self.write_split_grids([b'', b''])
        (Path(self.jobs[1]['dirname']) / 'grid.MC_integer').write_bytes(
            b'AVE 6\nSUM 5 7\n')
        for name in ('mint_grids', 'grid.MC_integer'):
            (self.base / name).write_bytes(b'previous master\n')
        with self.assertRaisesRegex(aMCatNLOError, 'different numbers of values'):
            self.combine()
        for name in ('mint_grids', 'grid.MC_integer'):
            self.assertEqual((self.base / name).read_bytes(), b'previous master\n')
