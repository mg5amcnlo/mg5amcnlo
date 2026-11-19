#!/usr/bin/env python3
"""Replay the saved endpoint corpus against current production templates."""
from collections import Counter
import gzip
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from central_C_LOC_map_replay import ROOT, batch_driver, category, TestPythiaLocalRecoil


def main():
    audit = ROOT / "docs/audits"
    records = json.load(gzip.open(audit / "central_C_LOC_map_points.json.gz", "rt"))
    inputs = "".join(f"{r['beam']} {r['partner']} {r['j']} 173 .1 .2 .3\n" +
                     "\n".join(" ".join(f"{v:.17g}" for v in p) for p in r["momenta"]) + "\n"
                     for r in records)
    with tempfile.TemporaryDirectory(prefix="C_LOC_endpoints_") as temp:
        driver = Path(temp) / "driver.f90"
        driver.write_text(batch_driver())
        original = subprocess.run

        def substitute(command, *args, **kwargs):
            if isinstance(command, list) and any(str(v).endswith("check_pythia_local_recoil.f90") for v in command):
                command = [str(driver) if str(v).endswith("check_pythia_local_recoil.f90") else v
                           for v in command if not str(v).startswith("-ffpe-trap")]
            return original(command, *args, **kwargs)

        subprocess.run = substitute
        try:
            TestPythiaLocalRecoil.setUpClass()
        finally:
            subprocess.run = original
        try:
            run = original([str(TestPythiaLocalRecoil.executable), "native_batch"],
                           input=inputs, capture_output=True, text=True)
        finally:
            TestPythiaLocalRecoil.doClassCleanups()
    parsed = re.findall(r"AUDIT_RESULT\s+([TF])\s+(\S+)", run.stdout)
    if run.returncode or len(parsed) != len(records):
        raise RuntimeError(f"Incomplete replay {len(parsed)}/{len(records)} exit {run.returncode}\n{run.stderr}")
    result = {"points": len(records), "groups": {}, "source_sha256": {
        name: hashlib.sha256((ROOT / "Template/NLO/SubProcesses" / name).read_bytes()).hexdigest()
        for name in ("genps_fks.f", "genps_fks_radiation.f", "initial_recoil.f",
                     "genps_fks_helpers.f", "fks_Sij.f", "fks_singular.f")}}
    failures = []
    for group in ("ISR", "top_FSR", "light_FSR"):
        values = [(r, state, float(delta)) for r, (state, delta) in zip(records, parsed) if category(r) == group]
        result["groups"][group] = {"points": len(values), "passed": sum(state == "T" for _, state, _ in values),
                                     "max_relative_error_passed": max((delta for _, state, delta in values if state == "T"), default=None)}
        failures.extend(dict(row, replay_relative_error=delta) for row, state, delta in values if state == "F")
    result["diagnostics"] = dict(Counter(label for label in (
        "Inconsistent shat", "Kinematics of counterevents", "Native MC reconstruction mismatch")
        for _ in range(run.stdout.count(label))))
    (audit / "central_C_LOC_endpoint_replay.json").write_text(json.dumps(result, indent=2) + "\n")
    with gzip.open(audit / "central_C_LOC_endpoint_failures.json.gz", "wt") as stream:
        json.dump(failures, stream)
    (Path(tempfile.gettempdir()) / "C_LOC_endpoint_replay.log").write_text(run.stdout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
