"""Recorded single-top endpoints must survive native projection and replay.

The inputs include almost beam-collinear TeV light quarks, soft jets and
hard-emission endpoints; their rounded energies carry no reliable E-|pz|.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.unit_tests.fks import test_pythia_local_recoil as recoil_tests
from tests.unit_tests.fks.test_momentum_maps import ROOT


@unittest.skipUnless(shutil.which("gfortran"), "requires gfortran")
class TestPythiaLocalEndpointMaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="mg5_local_endpoints_")
        cls.addClassCleanup(temporary.cleanup)
        driver = Path(temporary.name) / "driver.f90"
        source = (ROOT / "tests/input_files/check_pythia_local_recoil.f90").read_text()
        source = source.replace("if(mode.eq.'native_point')then", """if(mode.eq.'native_point'.or.mode.eq.'native_fsr')then
     if(mode.eq.'native_fsr')then
        jfks=partner
        initial_recoil_leg=beam
        resonance_recoil=.true.
     endif"""
                                )
        source = source.replace("     write(*,*)'PASS native point'", """     write(*,*)'PASS native point',maxval(abs(poutlab-plab))/maxval(abs(plab))"""
                                )
        source = source.replace("     write(*,*)'PASS native point'", """     write(*,*)'CONSERVATION',maxval(abs(sum(poutlab(:,1:2),dim=2)-sum(poutlab(:,3:5),dim=2)))/maxval(abs(poutlab))
     write(*,*)'PASS native point'"""
                                )
        driver.write_text(source)
        original_run = subprocess.run

        def compile_driver(command, *args, **kwargs):
            if isinstance(command, list):
                command = [str(driver) if str(v).endswith("check_pythia_local_recoil.f90") else v
                           for v in command]
            return original_run(command, *args, **kwargs)

        subprocess.run = compile_driver
        try:
            recoil_tests.TestPythiaLocalRecoil.setUpClass()
        finally:
            subprocess.run = original_run
        cls.addClassCleanup(recoil_tests.TestPythiaLocalRecoil.doClassCleanups)
        cls.executable = recoil_tests.TestPythiaLocalRecoil.executable
        cls.points = json.loads((ROOT / "tests/input_files/pythia_local_endpoint_points.json").read_text())

    def check_points(self, category):
        for row in self.points:
            if (row["j"] <= 2) != (category == "ISR"):
                continue
            with self.subTest(source=row["source"], emitter=row["j"]):
                mode = "native_point" if category == "ISR" else "native_fsr"
                data = f"{row['beam']} {row['partner']} 173 .1 .2 .3\n"
                data += "\n".join(" ".join(f"{v:.17g}" for v in p) for p in row["momenta"]) + "\n"
                result = subprocess.run([str(self.executable), mode], input=data, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                relative = float(re.search(r"PASS native point\s+(\S+)", result.stdout).group(1))
                self.assertLess(relative, 2e-8)
                if row.get("check_conservation"):
                    conservation = float(re.search(r"CONSERVATION\s+(\S+)", result.stdout).group(1))
                    self.assertLess(conservation, 1e-12)
                for diagnostic in ("Inconsistent shat", "Kinematics of counterevents", "reconstruction mismatch"):
                    self.assertNotIn(diagnostic, result.stdout)

    def test_recorded_initial_state_endpoints(self):
        self.check_points("ISR")

    def test_recorded_final_state_endpoints(self):
        self.check_points("FSR")
