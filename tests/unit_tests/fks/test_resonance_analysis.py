"""IR-safe reconstruction and statistical comparisons for recoil validation."""

import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "Template/NLO/SubProcesses"


@unittest.skipUnless(shutil.which("gfortran") and shutil.which("g++"),
                     "requires gfortran and g++")
class TestResonanceAnalysis(unittest.TestCase):
    def test_reconstruction_cuts_and_soft_collinear_limits(self):
        with tempfile.TemporaryDirectory(prefix="mg5_recoil_analysis_") as tmp:
            work = Path(tmp)
            (work / "nexternal.inc").write_text(
                "      integer nexternal,nincoming\n"
                "      parameter(nexternal=7,nincoming=2)\n")
            (work / "cuts.inc").write_text(
                "      double precision jetradius,jetalgo,ptj,etaj\n"
                "      common /test_cuts/ jetradius,jetalgo,ptj,etaj\n")
            commands = [
                ["g++", "-O1", "-std=c++11", "-c",
                 str(TEMPLATE / "fastjetfortran_madfks_core.cc"),
                 str(TEMPLATE / "fjcore.cc")],
                ["gfortran", "-O2", "-fcheck=all", "-fbacktrace",
                 "-ffixed-line-length-none", "-I", str(work),
                 "-I", str(TEMPLATE),
                 str(ROOT / "tests/input_files/check_resonance_analysis.f90"),
                 str(ROOT / "tests/input_files/analysis_HwU_resonance_recoil.f"),
                 str(ROOT / "tests/input_files/resonance_recoil_cuts.f"),
                 str(TEMPLATE / "fastjet_wrapper.f"),
                 "fastjetfortran_madfks_core.o", "fjcore.o", "-lstdc++",
                 "-o", str(work / "check_analysis")],
                [str(work / "check_analysis")],
            ]
            for command in commands:
                result = subprocess.run(command, cwd=work, text=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("PASS recoil analysis", result.stdout)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy")
class TestRecoilHistogramComparison(unittest.TestCase):
    def test_independent_run_uncertainties(self):
        from tests.input_files import compare_resonance_recoil as comparison
        import numpy as np
        baseline = comparison.Histogram("test", np.array(
            [[0, 1, 4, 0.3], [1, 2, 9, 0.4]], dtype=float))
        local = comparison.Histogram("test", np.array(
            [[0, 1, 5, 0.4], [1, 2, 8, 0.3]], dtype=float))
        rows = comparison.compare_pair({"test": baseline}, {"test": local}, "test")
        self.assertEqual([row["pull"] for row in rows], [2.0, -2.0])
        self.assertAlmostEqual(rows[0]["p_value"], math.erfc(math.sqrt(2)))
        self.assertAlmostEqual(rows[0]["ratio"], 1.25)
        self.assertAlmostEqual(rows[0]["ratio_error"], math.hypot(0.4/4, 5*0.3/16))
        json.dumps(rows, allow_nan=False)

    def test_incomplete_histograms_are_rejected(self):
        from tests.input_files import compare_resonance_recoil as comparison
        with tempfile.TemporaryDirectory(prefix="mg5_recoil_histogram_") as tmp:
            path = Path(tmp) / "input.HwU"
            path.write_text('<histogram> 2 "test"\n0 1 4 0.3\n')
            with self.assertRaisesRegex(ValueError, "Incomplete HwU"):
                comparison.read_hwu(path)
            path.write_text(path.read_text() + '1 2 9 0.4\n<\\histogram>\n')
            result = comparison.read_hwu(path)
            self.assertEqual(result["test"].values.tolist(), [4, 9])
