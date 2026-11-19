#!/usr/bin/env python3
"""Run existing map tests with getangles replaced only in temporary builds."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from central_C_LOC_map_replay import ROOT, GETANGLES


def main():
    source = (ROOT / "Template/NLO/SubProcesses/genps_fks_helpers.f").read_text()
    start = source.index("      subroutine getangles(")
    end = source.index("      end subroutine getangles", start) + len("      end subroutine getangles")
    with tempfile.TemporaryDirectory(prefix="C_LOC_candidate_tests_") as temp:
        helper = Path(temp) / "helpers.f"
        helper.write_text(source[:start] + GETANGLES + source[end:])
        original = subprocess.run

        def substitute(command, *args, **kwargs):
            if isinstance(command, list):
                command = [str(helper) if str(v).endswith("/genps_fks_helpers.f") else v for v in command]
            return original(command, *args, **kwargs)

        subprocess.run = substitute
        try:
            suite = unittest.defaultTestLoader.loadTestsFromNames([
                "tests.unit_tests.fks.test_momentum_maps",
                "tests.unit_tests.fks.test_initial_recoil",
                "tests.unit_tests.fks.test_pythia_local_recoil"])
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        finally:
            subprocess.run = original
    report = {"scope": "Existing tests, temporary candidate getangles compilation only",
              "tests_run": result.testsRun, "errors": len(result.errors),
              "failures": len(result.failures), "passed": result.wasSuccessful()}
    (ROOT / "docs/audits/central_C_LOC_map_candidate_tests.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
